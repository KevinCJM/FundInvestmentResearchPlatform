"""Historical quality diagnostics; immutable reports share reliability persistence."""

import time
import numpy as np
from custom_indicators.errors import ValidationError, ConflictError
from ..v2_contracts import parse_definition_v2, definition_content_hash
from ..v2_service import _content_hash, _json_safe, _source_spec, _definition_output_frequency
from .service import ReliabilityService
from .contracts import QualityRequest, QualityConfirmRequest
from .diagnostics import check_inputs, prepared_plan, run_diagnostics
from .diagnostic_kernels import quality_kernel, horizon_profile_kernel, compare_kernel
from .report import finite, axis
from . import kernels


class QualityService(ReliabilityService):
    request_model = QualityRequest
    confirm_model = QualityConfirmRequest
    report_prefix = "reference-quality"
    storage_name = "historical_regime_reference_quality"

    def _compute(self, request):
        kernels.audit()
        started = time.monotonic()
        definition = parse_definition_v2(
            self.graph.get_definition(request.definition_id, request.revision)
        )
        if (
            definition.study is not None
            and definition.study.purpose != "historical_reference"
        ):
            raise ValidationError(
                "QUALITY_STUDY_REQUIRED", "质量检查仅用于历史状态定义。"
            )
        if any(n.type == "annotation.manual_events" for n in definition.graph.nodes):
            raise ValidationError(
                "QUALITY_MANUAL_EVENTS", "重叠人工事件不能作为互斥历史状态。"
            )
        cutoff = request.as_of.isoformat() if request.as_of else None
        cache = {}
        check_inputs(self.graph, definition, "retrospective", cutoff, cache)
        with prepared_plan(self.graph, definition, request.compile_token) as plan:
            return self._evaluate(request, definition, cutoff, cache, plan, started)

    def _evaluate(self, request, definition, cutoff, cache, plan, started):
        execution = self.graph._execute_graph(
            None, definition, "retrospective", cutoff, plan=plan, source_cache=cache
        )
        points = execution["series"]
        if len(points) > 20000:
            raise ValidationError("RELIABILITY_BUDGET", "输出最多20000个观测。")
        dates, date_ordinals = axis(points)
        codes = {s.id: i for i, s in enumerate(definition.states)}
        labels = np.asarray(
            [codes.get(p.get("state_id"), -1) for p in points], np.int64
        )
        price_info = self._price_semantics(
            definition, execution, request.policy.include_price_returns
        )
        prices = np.asarray(
            [p.get("value") if p.get("value") is not None else np.nan for p in points],
            np.float64,
        )
        if price_info["status"] != "available":
            prices = np.full(len(points), np.nan)
        counts, values = quality_kernel(labels, prices, len(codes))
        horizon_values, horizon_summary = horizon_profile_kernel(labels, date_ordinals, len(codes))
        episode_counts, _ = kernels.state_evidence_kernel(labels, labels, len(codes))
        coverage = compare_kernel(labels, labels, len(codes))[2]
        rows = []
        for i, state in enumerate(definition.states):
            row = {"state_id": state.id}
            for j, name in enumerate(
                (
                    "observations",
                    "segments",
                    "min_length",
                    "median_length",
                    "max_length",
                    "mean_length",
                    "price_return_samples",
                    "mean_price_return",
                )
            ):
                row[name] = finite(values[i, j])
            row["price_return_reason"] = (
                price_info["reason"]
                if price_info["status"] != "available"
                else (None if values[i, 6] else "no_valid_price_segments")
            )
            row["independent_complete_episodes"] = int(episode_counts[i, 3])
            if int(horizon_values[i, 0]) != row["independent_complete_episodes"]:
                raise ConflictError("HORIZON_EPISODE_MISMATCH", "状态持续期与独立区间计数不一致。")
            row["conditional_estimation_status"] = (
                "ready"
                if row["independent_complete_episodes"] >= request.policy.minimum_state_episodes_for_estimation
                else "insufficient_evidence"
            )
            rows.append(row)
        ready_states = [row["state_id"] for row in rows if row["conditional_estimation_status"] == "ready"]
        estimation_status = (
            "ready" if len(ready_states) == len(rows)
            else "partially_ready" if ready_states
            else "insufficient_evidence"
        )
        report = {
            "schema_version": "1.0",
            "kind": "historical_reference_quality",
            "status": "diagnostic_only" if counts[0] else "insufficient_evidence",
            "sample": {
                "input": len(points),
                "classified": int(counts[0]),
                "unknown": int(counts[1]),
                "head_unknown": int(counts[2]),
                "tail_unknown": int(counts[3]),
                "coverage": finite(coverage),
                "first_date": dates[0] if dates else None,
                "last_date": dates[-1] if dates else None,
            },
            "segments": {
                "total": int(counts[4]),
                "transitions": int(counts[5]),
                "per_state": rows,
            },
            "price_returns": price_info,
            "horizon_profile": {
                "method": "empirical_complete_episode_duration",
                "calendar_boundary": "observation_inclusive_next_observation_exclusive",
                "observation_frequency": _definition_output_frequency(definition),
                "calendar_span_days": finite(horizon_summary[2]),
                "classified_coverage": finite(horizon_summary[3]),
                "transitions": int(horizon_summary[0]) if np.isfinite(horizon_summary[0]) else 0,
                "transitions_per_year": finite(horizon_summary[1]),
                "censoring": "open_head_tail_and_unknown_bounded_episodes_excluded",
                "per_state": [
                    {
                        "state_id": state.id,
                        "independent_complete_episodes": int(horizon_values[i, 0]),
                        "duration_observations_p25": finite(horizon_values[i, 1]),
                        "duration_observations_median": finite(horizon_values[i, 2]),
                        "duration_observations_p75": finite(horizon_values[i, 3]),
                        "duration_calendar_days_p25": finite(horizon_values[i, 4]),
                        "duration_calendar_days_median": finite(horizon_values[i, 5]),
                        "duration_calendar_days_p75": finite(horizon_values[i, 6]),
                        "classified_occupancy": finite(horizon_values[i, 7]),
                    }
                    for i, state in enumerate(definition.states)
                ],
                "interpretation": "measured_persistence_not_user_declared_horizon",
            },
            "conditional_estimation": {
                "scope": "episode_sample_sufficiency_only",
                "ltcma_estimated": False,
                "status": estimation_status,
                "ready_states": ready_states,
                "fallback_states": [row["state_id"] for row in rows if row["state_id"] not in ready_states],
                "minimum_state_episodes": request.policy.minimum_state_episodes_for_estimation,
                "fallback_policy": "shrink_or_base_ltcma_for_insufficient_states",
            },
            "stability": run_diagnostics(
                self.graph,
                definition,
                request.policy.stability,
                points,
                "retrospective",
                cutoff,
                cache=cache,
                started=started,
            ),
            "lineage": {
                "definition_hash": definition_content_hash(definition),
                "data_snapshots": execution["result"]["data_snapshots"],
                "evaluation_snapshot": execution["result"].get("evaluation_snapshot"),
            },
            "warnings": [
                "Historical segmentation diagnostics are not ground truth.",
                "This report does not publish a reference or establish historical label availability.",
            ],
            "execution": kernels.audit(),
        }
        if time.monotonic() - started > 120:
            raise ConflictError("RELIABILITY_TIME_BUDGET", "质量检查超过时间预算。")
        canonical = request.model_dump(mode="json", exclude={"compile_token"})
        result = _json_safe({"request": canonical, "report": report})
        return {"preview_hash": _content_hash(result), **result}

    @staticmethod
    def _price_semantics(definition, execution, enabled):
        if not enabled:
            return {
                "status": "disabled",
                "reason": "disabled_by_policy",
                "semantics": None,
            }
        primary = next((t for t in definition.evaluation_targets if t.primary), None)
        if primary is None and definition.evaluation_targets:
            primary = definition.evaluation_targets[0]
        if primary:
            spec = primary.source
        else:
            node = next(
                (
                    n
                    for n in definition.graph.nodes
                    if n.id == execution["result"]["display_source"]
                ),
                None,
            )
            spec = _source_spec(node.type, node.parameters) if node else {}
        kind, field = spec.get("kind"), spec.get("field", "close")
        allowed = {
            "index": {"close", "open", "high", "low"},
            "etf": {"close", "hfq_close", "qfq_close", "adj_close"},
            "fund": {"adj_nav", "adjusted_nav", "unit_nav", "accum_nav"},
        }
        if field not in allowed.get(kind, set()):
            return {
                "status": "unavailable",
                "reason": "no_verified_price_semantics",
                "semantics": None,
            }
        return {
            "status": "available",
            "reason": None,
            "semantics": {
                "kind": kind,
                "field": field,
                "return": "segment_endpoint_simple_return",
                "unit": "decimal",
                "includes_distributions": field
                in {"adj_nav", "adjusted_nav", "hfq_close"},
            },
        }

    def _summary(self, item):
        request = item["request"]
        return {
            "id": item["id"],
            "created_at": item["created_at"],
            "definition_id": request["definition_id"],
            "revision": request["revision"],
            "status": item["report"]["status"],
            "conditional_estimation": item["report"].get("conditional_estimation"),
        }

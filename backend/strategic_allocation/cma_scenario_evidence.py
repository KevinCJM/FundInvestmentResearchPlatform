"""Read-only scenario references and calibrated current-state evidence for CMA.

The scenario center remains the authority for immutable runs, publications and
calibration. This adapter never creates a competing store or grants SAA eligibility.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from backend.custom_indicators.errors import ConflictError, ValidationError
from custom_indicators.errors import IndicatorDomainError as RegimeDomainError
from backend.custom_indicators.errors import IndicatorDomainError


DOMAIN_ERRORS = (IndicatorDomainError, RegimeDomainError)


def _reason(code, message):
    return {"code": code, "message": message}


def _reference(value):
    return value.model_dump(mode="json") if hasattr(value, "model_dump") else value


class CmaScenarioEvidence:
    def __init__(self, evidence):
        self.evidence = evidence
        self.graph = None

    def bind(self, graph):
        # The live scenario center owns this root. Strategic artifacts may live
        # elsewhere when STRATEGIC_ALLOCATION_DATA_DIR is configured separately.
        self.evidence.regime_root = graph.workspace_data_dir
        self.evidence.runs = graph.runs
        self.graph = graph

    def _scope_reasons(self, raw, as_of, mode):
        reasons = []
        if raw.get("schema_version") != "2.0" or raw.get("immutable") is not True or raw.get("mode") != mode:
            reasons.append(_reason("snapshot_required", "请在情景研究中心保存不可变的研究版本。"))
        if mode == "realtime" and raw.get("frequency") != "daily":
            reasons.append(_reason("daily_required", "条件情景按日推演概率，需要日频的当前市场研究；历史区间不受此限制。"))
        if mode == "realtime":
            causality = raw.get("causality") if isinstance(raw.get("causality"), dict) else {}
            if (not causality.get("is_causal") or causality.get("uses_future_data")
                    or causality.get("repaints") or not causality.get("realtime_eligible")):
                reasons.append(_reason("causal_required",
                    "条件情景需要不使用未来信息、且不重绘历史判断的实时研究，请先完成因果识别检查。"))
        if not raw.get("as_of"):
            reasons.append(_reason("cutoff_missing", "这份情景研究没有记录研究日，请重新保存带研究日的版本。"))
        elif raw["as_of"] > str(as_of):
            reasons.append(_reason("after_research_day", "这份情景研究使用了研究日之后的数据，请重新计算截至研究日的版本。"))
        if self.evidence._regime_hash is None:
            reasons.append(_reason("reader_not_ready", "情景读取服务尚未就绪，请稍后重试。"))
        elif self.evidence._regime_hash(raw) != raw.get("content_hash"):
            reasons.append(_reason("snapshot_changed", "这份情景研究的保存内容不完整，请回到情景研究中心重新保存。"))
        return reasons

    def _published_reference(self, raw, reference, as_of):
        if self.graph is None:
            raise ValidationError("LTCMA_SCENARIO_SERVICE_UNAVAILABLE", "情景研究中心尚未连接，请稍后重试。")
        from backend.historical_regimes.reliability.references import resolve_reference
        verified, publication = resolve_reference(self.graph, reference, hydrate=False)
        if verified["id"] != raw["id"] or verified["content_hash"] != raw["content_hash"]:
            raise ConflictError("LTCMA_SCENARIO_REFERENCE_MISMATCH", "所选情景发布版本与历史研究不一致，请重新选择。")
        # Publication records provenance; research eligibility follows the frozen
        # data/label cutoff, not the day the historical study was archived.
        return publication

    def historical_reasons(self, raw, as_of):
        base = self._scope_reasons(raw, as_of, "retrospective")
        if not base:
            try:
                self.evidence.regime(SimpleNamespace(
                    as_of=as_of, run_ref=SimpleNamespace(id=raw["id"], content_hash=raw["content_hash"])), {"dates": []})
            except DOMAIN_ERRORS as exc:
                base.append(_reason(exc.code, exc.message))
        return base

    def _historical_options(self, raw, as_of):
        base = self.historical_reasons(raw, as_of)
        publications = raw.get("publications") or []
        publications = [publication for publication in publications if publication.get("id")]
        common = {key: raw.get(key) for key in ("id", "name", "content_hash", "as_of", "frequency", "states")}
        if not publications:
            return [{**common, "reference": None, "source_kind": "saved_run",
                     "available": not base, "reasons": base}]
        options = []
        # Recognition artifacts bind an exact publication, even for one run/hash.
        for index, publication in enumerate(publications, 1):
            reference = {"run_id": raw["id"], "publication_id": publication["id"],
                         "content_hash": raw.get("content_hash")}
            reasons = list(base)
            try:
                self._published_reference(raw, reference, as_of)
            except DOMAIN_ERRORS as exc:
                reasons.append(_reason(exc.code, exc.message))
            name = common["name"]
            if len(publications) > 1:
                published_on = str(publication.get("published_at") or "")[:10]
                name = f"{name or '历史情景研究'} · 发布 {index}" + (f" · {published_on}" if published_on else "")
            options.append({**common, "name": name, "reference": reference, "source_kind": "published_reference",
                            "available": not reasons, "reasons": reasons})
        return options

    def options(self, as_of):
        historical, realtime = [], []
        for raw in self.evidence._regime_items():
            if raw.get("schema_version") != "2.0":
                continue
            if raw.get("mode") == "retrospective":
                historical.extend(self._historical_options(raw, as_of))
            elif raw.get("mode") == "realtime":
                reasons = self._scope_reasons(raw, as_of, "realtime")
                study = (raw.get("definition") or {}).get("study") or {}
                if not reasons:
                    try:
                        self._current(raw, as_of)
                    except DOMAIN_ERRORS as exc:
                        reasons.append(_reason(exc.code, exc.message))
                realtime.append({**{key: raw.get(key) for key in ("id", "name", "content_hash", "as_of", "frequency")},
                                 "reference": study.get("reference"), "available": not reasons, "reasons": reasons})
        eligible = [row for row in historical if row["available"]]
        return {"historical_references": historical, "realtime_runs": realtime,
                "default_historical_id": eligible[0]["id"] if len(eligible) == 1 else None,
                "as_of": str(as_of)}

    def historical(self, model, evidence):
        result = self.evidence.regime(model, evidence, include_available_dates=True)
        raw = next(row for row in self.evidence._regime_items() if row["id"] == model.run_ref.id)
        reference = _reference(model.historical_reference)
        publication = None
        if reference is not None:
            publication = self._published_reference(raw, reference, model.as_of)
        elif raw.get("publications"):
            # Preserve the exact publication identity even for reconstructed research.
            raise ValidationError("LTCMA_SCENARIO_REFERENCE_REQUIRED", "这份研究已有发布版本，请从情景参考列表重新选择。")
        states, state_ids, audit = result
        audit = {**audit, "reference": reference, "publication": publication,
                 "reference_quality": self._quality(raw, model.as_of),
                 "reference_kind": "published_reference" if reference else "saved_run"}
        return states, state_ids, audit

    def _quality(self, raw, as_of):
        """Only attach diagnostics for the exact definition, inputs and cutoff."""
        if self.graph is None:
            return {"status": "not_available"}
        for summary in self.graph.reference_quality.catalog()["items"]:
            if (summary.get("definition_id") != raw.get("definition_id")
                    or summary.get("revision") != raw.get("definition_revision")):
                continue
            item = self.graph.reference_quality.get(summary["id"])
            request, report = item["request"], item["report"]
            lineage = report.get("lineage", {})
            snapshots = raw.get("data_snapshots")
            if (request.get("as_of") != raw.get("as_of")
                    or item["created_at"][:10] > str(as_of)
                    or lineage.get("definition_hash") != raw.get("definition_snapshot_hash")
                    or not snapshots or lineage.get("data_snapshots") != snapshots):
                continue
            return {"status": "available", "id": item["id"], "content_hash": item["content_hash"],
                    "conditional_estimation": report.get("conditional_estimation"),
                    "horizon_profile": report.get("horizon_profile"), "stability": report.get("stability")}
        return {"status": "not_available"}

    def _current(self, raw, as_of, expected_reference=None, expected_day=None):
        reasons = self._scope_reasons(raw, as_of, "realtime")
        if reasons:
            raise ValidationError("LTCMA_CURRENT_SCENARIO_UNAVAILABLE", reasons[0]["message"])
        if self.graph is None:
            raise ValidationError("LTCMA_SCENARIO_SERVICE_UNAVAILABLE", "情景研究中心尚未连接，请稍后重试。")
        study = (raw.get("definition") or {}).get("study") or {}
        reference = study.get("reference")
        if study.get("purpose") != "realtime_recognition" or not reference:
            raise ValidationError("LTCMA_CURRENT_REFERENCE_REQUIRED", "这份实时研究尚未绑定历史情景参考，请先在情景研究中心完成绑定。")
        if expected_reference is not None and reference != expected_reference:
            raise ValidationError("LTCMA_SCENARIO_AXES_DIFFER", "当前判断与历史情景不属于同一研究版本，请选择匹配的实时研究。")
        historical_raw = self.graph.runs.get(reference["run_id"])
        self._published_reference(historical_raw, reference, as_of)
        from backend.historical_regimes.reliability.consumer import (
            attach_calibration, calibrated_output, consumer_states,
        )
        # The snapshot, PIT and causality checks above authorize this read.
        # TAA publication is a separate downstream gate, not CMA input evidence.
        run = attach_calibration(self.graph.hydrate_run_snapshot(raw), self.graph.reliability)
        if run.get("content_hash") != raw["content_hash"]:
            raise ConflictError("LTCMA_CURRENT_VERSION_CHANGED", "实时情景在读取期间发生变化，请重新选择。")
        states = consumer_states(run)
        points = [p for p in run.get("series", []) if p.get("observation_date", "9999") <= str(as_of)]
        if not points:
            raise ValidationError("LTCMA_CURRENT_STATE_MISSING", "研究日尚无可用的当前情景判断，请先保存对应日期的实时研究。")
        point = max(points, key=lambda p: p["observation_date"])
        if expected_day is not None and point["observation_date"] != expected_day:
            raise ValidationError("LTCMA_CURRENT_STATE_STALE", "实时情景尚未更新到本次收益数据的最后日期，请先更新实时研究。")
        if not point.get("available_at") or point["available_at"][:10] > str(as_of):
            raise ValidationError("LTCMA_CURRENT_DATA_UNAVAILABLE", "当前情景判断使用的数据在研究日尚不可得。")
        probabilities, confidence, reason = calibrated_output(run, point, str(as_of), states)
        if reason:
            messages = {
                "calibration_missing": "这份实时研究还没有概率校准结果，请先完成识别验证。",
                "calibration_not_deployment_eligible": "实时概率尚未通过可用性验证，可先进行长期情景研究。",
                "calibration_not_yet_available": "概率校准结果在研究日尚不可用，请选择当时可用的版本。",
                "calibration_expired": "实时概率的验证有效期已过，请先更新识别验证。",
                "calibration_confidence_below_floor": "当前情景判断不够可靠，暂时不能据此生成条件预测。",
                "calibration_state_not_qualified": "当前情景缺少独立验证证据，可先进行长期情景研究。",
            }
            raise ValidationError("LTCMA_CURRENT_PROBABILITY_UNAVAILABLE", messages.get(reason,
                "当前情景概率的验证或版本绑定尚不完整，请先在情景研究中心完成识别验证。"))
        context = run["_reliability"]
        artifact = context["artifact"]
        qualification = context.get("qualification")
        verification = artifact["report"].get("verification") or {}
        verified = ((qualification.get("qualified_states") or []) if qualification
                    else (verification.get("verified_states") or []))
        if any(probabilities[state] > 0 and state not in verified for state in states):
            raise ValidationError("LTCMA_CURRENT_STATE_UNVERIFIED",
                "当前概率包含尚未通过独立验证的情景，请补充识别验证后再进行条件研究。")
        vector = np.asarray([probabilities[state] for state in states], dtype=np.float64)
        vector.flags.writeable = False
        return {"current_probabilities": vector, "state_ids": states,
                "realtime_ref": {"id": raw["id"], "content_hash": raw["content_hash"]},
                "reference_ref": reference, "observation_date": point["observation_date"],
                "calibration": {"id": artifact["id"], "content_hash": artifact["content_hash"],
                                "method": artifact["report"]["calibration"]["method"],
                                "qualification_id": study.get("qualification_id")},
                "recognition": {"confidence": confidence, "validated": True},
                "forecast_validation": {"status": "not_validated", "downstream_eligible": False},
                "historical_pit_proven": False}

    def conditional(self, model, evidence):
        reference = _reference(model.historical_reference)
        if reference is None:
            raise ValidationError("LTCMA_CONDITIONAL_REFERENCE_REQUIRED", "条件情景研究需要已发布的历史参考，请先选择情景参考。")
        raw = next((row for row in self.evidence._regime_items() if row.get("id") == model.realtime_ref.id), None)
        if raw is None or raw.get("content_hash") != model.realtime_ref.content_hash:
            raise ConflictError("LTCMA_CURRENT_VERSION_CHANGED", "所选实时研究版本已变化或不存在，请重新选择。")
        result = self._current(raw, model.as_of, reference, evidence["dates"][-1])
        if result["state_ids"] != evidence["regime"][1]:
            raise ValidationError("LTCMA_SCENARIO_AXES_DIFFER", "历史情景与当前判断的状态顺序不一致，请重新选择匹配版本。")
        return result

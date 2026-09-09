"""Named scalar orchestration; all formula arithmetic executes in warmed NJIT."""
from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import hmac
import json
import secrets
import os
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np

from cal_indicators.typed_dsl import (
    TypedDslError, TypedIndicatorRuntime, compose_typed_scalar_bundle,
    DEFAULT_MAX_LIVE_ELEMENTS, DEFAULT_MAX_RUNTIME_COST, runtime_validation_execution_audit,
)
from cal_indicators.typed_numba_kernels import STATUS_OK, NUMERIC_KERNEL_VERSION
from cal_indicators.multi_output import MULTI_OUTPUT_PORTS, STATUS_OUTPUT_UNAVAILABLE
from cal_indicators.typed_scalar_bundle import compile_scalar_bundle, CompiledScalarBundle
from cal_indicators.typed_numba_plan import NumbaPlanCompileError
from .errors import ValidationError
from .scalar_outputs import (
    SCALAR_BUNDLE, apply_scalar_output_contract, is_scalar_bundle,
    output_result, project_scalar_output,
)
from .series_provider import load_product_variable_series_batch, select_variable_window, market_data_generation
from .periods import period_cache_reference
from .plan_scoring import score_result_rows
from .parallel_engine import plan_scoring_execution_audit
from .typed_service import _ensure_context, normalize_variable_latex, variable_types


@dataclass(frozen=True)
class PreparedOutputGroup:
    dependencies: tuple[str, ...]
    compiled: CompiledScalarBundle
    output_ids: tuple[str, ...]
    runtimes: tuple[TypedIndicatorRuntime, ...]


@dataclass(frozen=True)
class PreparedBundle:
    validation: dict[str, Any]
    groups: tuple[PreparedOutputGroup, ...]


def definition_key(definition: dict[str, Any]) -> str:
    fields = (
        "name", "description", "context_kind", "dsl_version", "operator_registry_version",
        "numeric_kernel_version", "variable_registry_version", "data_contract_version",
        "context_schema_version", "annual_risk_free_rate_percent", "indicator_type",
    )
    payload = {key: definition.get(key) for key in fields}
    payload["outputs"] = [
        {key: output.get(key) for key in ("id", "label", "expression", "description", "unit", "display_format", "precision", "direction")}
        for output in definition.get("scalar_outputs", [])
    ]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


# Portfolio diagnosis creates short-lived service facades. Reuse only plans
# explicitly prepared in this process and workspace, never compile on a run.
_PREPARED_BY_WORKSPACE: dict[str, dict[str, PreparedBundle]] = {}


class ScalarBundleService:
    def __init__(self, owner: Any) -> None:
        self.owner = owner
        workspace = str(owner.workspace_data_dir.resolve())
        self._prepared = _PREPARED_BY_WORKSPACE.setdefault(workspace, {})
        self._secret = secrets.token_bytes(32)

    def _token(self, key: str) -> str:
        return hmac.new(self._secret, key.encode(), hashlib.sha256).hexdigest()

    def resolve_inline(self, definition: dict[str, Any], token: str | None) -> dict[str, Any]:
        key = definition_key(definition)
        if not token or not hmac.compare_digest(token, self._token(key)):
            raise ValidationError("INLINE_COMPILE_TOKEN_MISMATCH", "公式已更改或尚未校验，请重新校验全部结果。", field="compile_token")
        prepared = self._require_warmed(definition)
        apply_scalar_output_contract(definition, prepared.validation)
        return {**definition, "id": None, "revision": None, "source": "inline"}

    def _require_warmed(self, definition: dict[str, Any]) -> PreparedBundle:
        prepared = self._prepared.get(definition_key(definition))
        if prepared is None:
            raise ValidationError("NJIT_PLAN_NOT_WARMED", "此多结果版本尚未预热，请重新校验或保存。计算没有回退到 Python。", field="indicator_revision")
        return prepared

    def validate(self, fields: dict[str, Any]) -> dict[str, Any]:
        try:
            definition = self.owner._normalize_definition(fields)
            result = copy.deepcopy(self.warm(definition).validation)
            result["compile_token"] = self._token(definition_key(definition))
            return result
        except NumbaPlanCompileError:
            return {"valid": False, "diagnostics": [{"code": "NJIT_PLAN_COMPILE_FAILED", "message": "多个结果无法编译为固定签名计划，请检查公式。", "field": "scalar_outputs"}], "dependencies": [], "dag": None}
        except (ValidationError, TypedDslError) as exc:
            diagnostics = getattr(exc, "diagnostics", None) or [{
                "code": exc.code, "message": exc.message, "field": getattr(exc, "field", None) or "scalar_outputs",
            }]
            return {"valid": False, "diagnostics": diagnostics, "dependencies": [], "dag": None}

    def warm(self, definition: dict[str, Any]) -> PreparedBundle:
        key = definition_key(definition)
        if key in self._prepared:
            return self._prepared[key]
        domain = str(definition.get("context_kind") or "single_product")
        options = {
            "variable_types": variable_types(domain, definition["dsl_version"]),
            "dsl_version": definition["dsl_version"],
            "operator_registry_version": definition["operator_registry_version"],
        }
        inferences, runtimes, diagnostics = {}, {}, []
        for output in definition["scalar_outputs"]:
            projected = project_scalar_output(definition, output["id"])
            # Neutral describes presentation, not the scalar formula's type.
            validation = self.owner._validate_typed({**projected, "name": output["label"], "direction": "higher_better"})
            if not validation["valid"]:
                diagnostics.extend({
                    **item, "output_id": output["id"],
                    "message": f"{output['label']}：{item['message']}",
                    "field": f"scalar_outputs.{output['id']}.expression",
                } for item in validation["diagnostics"])
                continue
            from .service import _get_warmed_typed_plan
            plan = _get_warmed_typed_plan(normalize_variable_latex(output["expression"]), domain, definition["dsl_version"], definition["operator_registry_version"])
            runtimes[output["id"]] = TypedIndicatorRuntime.from_warmed_plan(plan)
            inferences[output["id"]] = {
                **validation, "id": output["id"], "label": output["label"],
                "shape": "scalar", "inferred_type": "scalar", "root_id": plan.root_id,
            }
        if diagnostics:
            raise ValidationError("INVALID_SCALAR_OUTPUTS", "部分结果需要修改。", field="scalar_outputs", diagnostics=diagnostics)
        expressions = {item["id"]: normalize_variable_latex(item["expression"]) for item in definition["scalar_outputs"]}
        full_plan = compose_typed_scalar_bundle(expressions, **options)
        _ensure_context(full_plan, domain)
        grouped: dict[tuple[str, ...], list[str]] = {}
        for output_id, runtime in runtimes.items():
            dependencies = (
                tuple(sorted(self.owner._physical_dependency_signature(runtime.plan.context_requirements)))
                if domain == "single_product" else ()
            )
            grouped.setdefault(dependencies, []).append(output_id)
        groups = []
        for dependencies, output_ids in grouped.items():
            plan = compose_typed_scalar_bundle({key: expressions[key] for key in output_ids}, **options)
            compiled = compile_scalar_bundle(plan)
            groups.append(PreparedOutputGroup(dependencies, compiled, tuple(output_ids), tuple(runtimes[key] for key in output_ids)))
        audit = self.owner._combined_njit_audit([*[group.compiled.metadata() for group in groups], runtime_validation_execution_audit()])
        validation = {
            "valid": True, "diagnostics": [], "dag": full_plan.graph_payload(),
            "dependencies": list(full_plan.context_requirements),
            "output_inferences": inferences, "output_type": SCALAR_BUNDLE,
            "output_measure": SCALAR_BUNDLE, "compiled_plan_id": groups[0].compiled.plan_id,
            "compiled_plan_ids": [group.compiled.plan_id for group in groups],
            "kernel_version": NUMERIC_KERNEL_VERSION, "compile_status": "compiled",
            "compile_token": self._token(key), "compile_token_scope": "current_process_warm_cache",
            "execution": audit, "python_fallback": 0, "python_operator_calls": 0,
            "dsl_version": definition["dsl_version"], "operator_registry_version": definition["operator_registry_version"],
        }
        for output_id, root_id in full_plan.roots.items():
            inferences[output_id]["root_id"] = root_id
        prepared = PreparedBundle(validation, tuple(groups))
        self._prepared[key] = prepared
        return prepared

    def _execute_group(
        self, group: PreparedOutputGroup, definition: dict[str, Any], context: dict[str, Any],
        records: dict[str, dict[str, Any]],
    ) -> None:
        enabled = np.zeros(len(group.output_ids), dtype=np.uint8)
        arguments_by_name: dict[str, Any] = {}
        total_cost, total_elements = 0, 0
        for index, (output_id, runtime) in enumerate(zip(group.output_ids, group.runtimes)):
            record = records[output_id]
            if record["status"] == "unavailable":
                continue
            try:
                arguments, _bindings, trace = runtime.prepare_context(context)
                total_cost += sum(item["runtime_cost"] for item in trace)
                total_elements += sum(max(1, int(np.prod(item["actual_shape"]))) for item in trace)
                arguments_by_name.update(zip(runtime.compiled_plan.context_names, arguments))
                enabled[index] = 1
            except TypedDslError as exc:
                record.update(value=None, status="unavailable")
                record["warnings"].append({"code": exc.code, "message": exc.message})
        if total_cost > DEFAULT_MAX_RUNTIME_COST or total_elements > DEFAULT_MAX_LIVE_ELEMENTS:
            raise ValidationError("COMPUTE_BUDGET_EXCEEDED", "多结果计算超过内存或计算预算，请减少结果或缩短区间。")
        if not enabled.any():
            return
        # Missing inputs belong only to disabled roots. Typed empty buffers are
        # never computed or exposed as substitute data.
        types_by_name = {name: value for runtime in group.runtimes for name, value in runtime.plan.context_requirements.items()}
        arguments = tuple(
            arguments_by_name.get(name, np.empty((0,) * types_by_name[name].rank, dtype=np.float64) if types_by_name[name].rank else 0.0)
            for name in group.compiled.context_names
        )
        values, statuses = group.compiled.compute(arguments, enabled)
        for index, output_id in enumerate(group.output_ids):
            if not enabled[index]:
                continue
            record = records[output_id]
            if statuses[index] == STATUS_OK:
                record["value"] = float(values[index])
                record["status"] = "warning" if record["warnings"] else "ok"
            else:
                record.update(value=None, status="warning")
                plan = group.runtimes[index].plan
                root = plan.nodes[plan.root_id]
                port = None
                if root.kind == "output" and statuses[index] == STATUS_OUTPUT_UNAVAILABLE:
                    parent = plan.nodes[root.inputs[0]]
                    port = next((item for item in MULTI_OUTPUT_PORTS.get(str(parent.operator_id), ()) if item.id == root.label), None)
                record["warnings"].append({
                    "code": "OUTPUT_UNAVAILABLE" if port else "NON_FINITE_RESULT",
                    "message": port.missing_message if port else "此结果不可计算，请检查除零、样本数量或数据范围。",
                })

    def evaluate_definition(
        self, definition: dict[str, Any], targets: list[dict[str, Any]],
        period: str, as_of: str | None, *, cache_stats: dict[str, int] | None = None,
    ) -> list[dict[str, Any]]:
        prepared = self._require_warmed(definition)
        generation = market_data_generation(self.owner.market_data_dir)
        stats = cache_stats if cache_stats is not None else {"hits": 0, "misses": 0}
        outputs = {item["id"]: project_scalar_output(definition, item["id"]) for item in definition["scalar_outputs"]}
        results = []
        for kind in sorted({item["kind"] for item in targets}):
            kind_targets = [target for target in targets if target["kind"] == kind]
            ids = [target["product_id"] for target in kind_targets]
            sources = {
                group.dependencies: load_product_variable_series_batch(kind, ids, group.dependencies, self.owner.market_data_dir, as_of)
                for group in prepared.groups
            }
            for target in kind_targets:
                records = {}
                fingerprints = [(deps, source_map[target["product_id"]].fingerprint) for deps, source_map in sources.items()]
                cache_key = "scalar-bundle:" + hashlib.sha256(repr((definition_key(definition), definition.get("id"), definition.get("revision"), target, period, as_of, period_cache_reference(as_of), generation, fingerprints)).encode()).hexdigest()
                cached = self.owner.cache.get(cache_key)
                if cached is not None:
                    stats["hits"] += 1
                    results.append(cached)
                    continue
                stats["misses"] += 1
                for group in prepared.groups:
                    source = sources[group.dependencies].get(target["product_id"])
                    error = None
                    try:
                        if source is None:
                            raise ValidationError("DATA_NOT_FOUND", "没有找到真实净值数据。")
                        window = select_variable_window(source, period, as_of, max_observations=5000)
                    except ValidationError as exc:
                        window, error = None, exc
                    for output_id in group.output_ids:
                        projected = outputs[output_id]
                        requirements = self.owner._input_requirements_payload(projected, source, window=window, window_error=error)
                        blocked = error is not None or requirements["status"] == "blocked"
                        warnings = list(window.warnings) if window is not None else []
                        if requirements["status"] == "blocked":
                            warnings.append(self.owner._input_requirement_warning(projected, requirements))
                        elif error is not None:
                            warnings.append({"code": error.code, "message": error.message})
                        warnings.extend(self.owner._partial_input_warnings(requirements))
                        records[output_id] = {
                            **self.owner._result_base(projected, target, source.identity.name if source else target["product_id"], period),
                            "output_id": output_id, "value": None,
                            "status": "unavailable" if blocked else "ok", "warnings": warnings,
                            "window": self.owner._window_payload(window) if window is not None else self.owner._empty_window(source, as_of),
                            "input_requirements": requirements, "target_data": self.owner._target_data_payload(source),
                        }
                    if window is not None:
                        elapsed = float((window.frame.iloc[-1]["date"] - window.frame.iloc[0]["date"]).days)
                        context = {**window.context, **self.owner._risk_free_context(definition, elapsed)}
                        self._execute_group(group, definition, context, records)
                result = self._pack(definition, records, period)
                if generation != market_data_generation(self.owner.market_data_dir):
                    raise ValidationError("DATA_GENERATION_CHANGED", "计算期间数据已更新，请重新运行。")
                self.owner.cache.put(cache_key, result)
                results.append(result)
        if generation != market_data_generation(self.owner.market_data_dir):
            raise ValidationError("DATA_GENERATION_CHANGED", "计算期间数据已更新，请重新运行。")
        return results

    def _pack(self, definition: dict[str, Any], records: dict[str, dict[str, Any]], period: str) -> dict[str, Any]:
        ordered = [records[item["id"]] for item in definition["scalar_outputs"]]
        first = ordered[0]
        finite_count = sum(item["value"] is not None for item in ordered)
        common_window = all(item["window"] == first["window"] for item in ordered)
        return {
            **self.owner._result_base(definition, first["target"], first["target"]["name"], period),
            "result_kind": SCALAR_BUNDLE, "value": None,
            "status": "ok" if all(item["status"] == "ok" for item in ordered) else "warning" if finite_count else "unavailable",
            "warnings": [],
            "window": first["window"] if common_window else self.owner._empty_window(None, None),
            "window_scope": "common" if common_window else "per_output",
            "outputs": ordered,
            "output_count": len(ordered), "available_output_count": finite_count,
        }

    def evaluate_mixed(self, definitions: list[dict[str, Any]], targets: list[dict[str, Any]], period: str, as_of: str | None, include_series: bool = False) -> dict[str, Any]:
        if include_series:
            raise ValidationError("OUTPUT_REQUIRED", "多结果指标的滚动曲线需要先选择具体结果并派生时序指标。", field="include_series")
        if sum(len(item.get("scalar_outputs") or [None]) for item in definitions) * len(targets) > 500:
            raise ValidationError("COMBINATION_LIMIT_EXCEEDED", "结果与产品组合数不能超过 500。")
        results, audits = [], []
        cache_stats = {"hits": 0, "misses": 0}
        for definition in definitions:
            if is_scalar_bundle(definition):
                results.extend(self.evaluate_definition(definition, targets, period, as_of, cache_stats=cache_stats))
                audits.append(self._require_warmed(definition).validation["execution"])
            else:
                response = self.owner.evaluate(indicator_ids=[definition["id"]], inline_definition=None, targets=targets, period=period, as_of=as_of, indicator_versions={definition["id"]: definition["revision"]})
                results.extend(response["results"])
                audits.append(response["execution"])
                for key in cache_stats:
                    cache_stats[key] += response.get("cache", {}).get(key, 0)
        return {
            "results": results,
            "summary": {"total": len(results), **{status: sum(item["status"] == status for item in results) for status in ("ok", "warning", "unavailable", "error")}},
            "cache": cache_stats,
            "execution": self.owner._combined_njit_audit(audits),
        }

    def evaluate_references(self, references: list[dict[str, Any]], targets: list[dict[str, Any]], period: str, as_of: str | None, *, prefer_snapshot: bool = True) -> dict[str, Any]:
        if not 1 <= len(references) <= 10:
            raise ValidationError("INDICATOR_LIMIT_EXCEEDED", "每次请选择 1 至 10 个具体结果。")
        groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
        for reference in references:
            definition = self.owner.indicators.get(reference["indicator_id"], reference.get("indicator_revision"))
            project_scalar_output(definition, reference.get("output_id"))
            groups.setdefault((definition["id"], definition["revision"]), []).append(reference)
        results, audits = [], []
        cache_stats = {"hits": 0, "misses": 0}
        for (indicator_id, revision), requested in groups.items():
            response = self.owner.evaluate(indicator_ids=[indicator_id], indicator_versions={indicator_id: revision}, inline_definition=None, targets=targets, period=period, as_of=as_of, prefer_snapshot=prefer_snapshot)
            audits.append(response["execution"])
            for key in cache_stats:
                cache_stats[key] += response.get("cache", {}).get(key, 0)
            for reference in requested:
                results.extend(output_result(result, reference.get("output_id")) for result in response["results"])
        return {
            "results": results, "summary": {"total": len(results), **{status: sum(item["status"] == status for item in results) for status in ("ok", "warning", "unavailable", "error")}},
            "cache": cache_stats, "execution": self.owner._combined_njit_audit(audits),
        }

    def run_plan(self, plan: dict[str, Any], as_of: str | None) -> dict[str, Any]:
        started = time.perf_counter()
        generation = market_data_generation(self.owner.market_data_dir)
        cache_key = self.owner._plan_run_cache_key(plan, as_of, generation)
        cached = self.owner.plan_cache.get(cache_key)
        if cached is not None:
            cached["execution"]["cache"] = {"plan_hits": 1, "plan_misses": 0}
            return cached
        targets = plan["targets"]
        values: list[list[dict[str, Any] | None]] = [[None] * len(plan["indicators"]) for _ in targets]
        names = [target["product_id"] for target in targets]
        target_index = {(target["kind"], target["product_id"]): index for index, target in enumerate(targets)}
        grouped: dict[tuple[str, int, str], list[tuple[int, dict[str, Any]]]] = {}
        for index, item in enumerate(plan["indicators"]):
            grouped.setdefault((item["indicator_id"], int(item["indicator_revision"]), item["period"]), []).append((index, item))
        audits, plan_ids = [], []
        for (indicator_id, revision, period), selected in grouped.items():
            definition = self.owner._decorate_definition(self.owner.indicators.get(indicator_id, revision))
            projections = {index: project_scalar_output(definition, item.get("output_id")) for index, item in selected}
            if is_scalar_bundle(definition):
                prepared = self._require_warmed(definition)
                computed = self.evaluate_definition(definition, targets, period, as_of)
                audits.append(prepared.validation["execution"])
                plan_ids.extend(group.compiled.plan_id for group in prepared.groups)
            else:
                computed = []
                # The existing public scalar path is bounded at 50 targets.
                for start in range(0, len(targets), 50):
                    response = self.owner.evaluate(indicator_ids=[indicator_id], inline_definition=None, indicator_versions={indicator_id: revision}, targets=targets[start:start + 50], period=period, as_of=as_of)
                    computed.extend(response["results"])
                    audits.append(response["execution"])
                    plan_ids.extend(response["execution"].get("compiled_plan_ids", []))
            for result in computed:
                row = target_index[(result["target"]["kind"], result["target"]["product_id"])]
                names[row] = result["target"]["name"]
                for index, item in selected:
                    projected_result = output_result(result, item.get("output_id"))
                    values[row][index] = self.owner._plan_value_payload(item, projections[index], projected_result)
        rows, ranked_count, total_weight = score_result_rows(plan, values, names)
        audit = self.owner._combined_njit_audit([*audits, plan_scoring_execution_audit()])
        result = {
            "plan_id": plan["id"], "plan_revision": int(plan["revision"]),
            "run_at": datetime.now(timezone.utc).isoformat(), "as_of": as_of,
            "rows": rows, "ranked_count": ranked_count, "excluded_count": len(targets) - ranked_count,
            "normalization": {"method": "min_max_0_100", "configured_weight_total": total_weight, "effective_weight_total": 1.0, "missing_policy": "strict"},
            "execution": {
                **audit, "data_generation": generation, "compiled_plan_ids": list(dict.fromkeys(plan_ids)),
                "request_time_compilation": 0, "compile_cache_misses": 0,
                "execution_lanes": {"numba_scalar_bundle": len(grouped), "python_fallback": 0},
                "combinations": len(targets) * len(plan["indicators"]),
                "worker_processes": 0, "worker_pids": [], "numba_threads": 1,
                "shared_memory_bytes": 0, "mmap_bytes": 0,
                "cache": {"plan_hits": 0, "plan_misses": 1},
                "timings_ms": {"total": round((time.perf_counter() - started) * 1000, 3)},
            },
        }
        if generation != market_data_generation(self.owner.market_data_dir):
            raise ValidationError("DATA_GENERATION_CHANGED", "计算期间数据已更新，请重新运行评价方案。")
        inline_limit = max(1, int(os.getenv("INDICATOR_INLINE_RESULT_LIMIT", "2000")))
        if len(targets) > inline_limit:
            result_id = self.owner.run_results.store(result)
            return self.owner.run_results.page(result_id, page=1, page_size=100)
        if len(targets) <= 2000:
            self.owner.plan_cache.put(cache_key, result)
        return result

    def evaluate_portfolio_definition(self, definition: dict[str, Any], snapshot: dict[str, Any]) -> dict[str, Any]:
        prepared = self._require_warmed(definition)
        context_error = None
        try:
            context = self.owner._portfolio_context(snapshot, definition)
        except ValidationError as exc:
            context, context_error = {}, exc
        window = self.owner._portfolio_window(snapshot)
        run_id = str(snapshot.get("id") or "unsaved-run")
        target = {"kind": "portfolio", "product_id": run_id, "name": str(snapshot.get("target_name") or run_id)}
        records = {}
        for output in definition["scalar_outputs"]:
            projected = project_scalar_output(definition, output["id"])
            warnings = copy.deepcopy(snapshot.get("warnings") or [])
            if context_error is not None:
                warnings.append({"code": context_error.code, "message": context_error.message})
            if {"asset_returns", "asset_weights"}.issubset(projected.get("required_variables") or []):
                warnings.append({"code": "STATIC_WEIGHT_HISTORY_ASSUMPTION", "message": "此公式把期末权重应用于整段历史，仅适合截面估算；历史表现应使用组合实际收益序列。"})
            records[output["id"]] = {
                **self.owner._result_base(projected, target, target["name"], "snapshot"),
                "output_id": output["id"], "value": None,
                "status": "unavailable" if context_error else "ok", "warnings": warnings, "window": window,
            }
        if context_error is None:
            for group in prepared.groups:
                self._execute_group(group, definition, context, records)
        return self._pack(definition, records, "snapshot")

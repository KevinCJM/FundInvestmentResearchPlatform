"""Application service for historical regime definitions and immutable runs."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from custom_indicators.errors import IndicatorDomainError, ValidationError
from custom_indicators.repository import utc_now
from custom_indicators.series_provider import DEFAULT_DATA_DIR
from custom_indicators.service import CustomIndicatorService

from .algorithms import UNKNOWN_STATE, run_algorithm
from .analytics import (
    analytics_execution_audit,
    build_segments,
    causality_report,
    compare_runs as compare_run_snapshots,
    conditional_statistics,
    prefix_stability,
    serialise_series,
    transition_matrix,
    walk_forward_report,
)
from .contracts import APPLICATION_TARGETS, RUN_MODES, meta_contract, normalize_definition
from .data import resolve_target
from .formula import (
    FormulaResult,
    evaluate_formula,
    mask_formula_numeric_outputs,
    prepare_formula as prepare_formula_plan,
)
from .numba_kernels import (
    historical_regime_numba_status,
    label_summary_kernel,
    perturb_scalar_kernel,
    predecessor_count_kernel,
    validation_windows_kernel,
    warm_historical_regime_numba_kernels,
)
from .repository import RegimeDefinitionRepository, RegimeRunRepository
from .taa import run_taa_backtest as execute_taa_backtest


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _content_hash(value: Any) -> str:
    encoded = json.dumps(_json_safe(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _business_definition(definition: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "name",
        "description",
        "template_id",
        "target",
        "features",
        "algorithm",
        "states",
        "validation",
        "usage_intent",
        "schema_version",
    )
    return {key: copy.deepcopy(definition.get(key)) for key in keys}


def _perturbed_definition(definition: dict[str, Any], perturbation: float) -> dict[str, Any]:
    alternate = copy.deepcopy(definition)
    parameters = alternate["algorithm"].setdefault("parameters", {})
    changed = False
    for key in ("upper", "lower", "bull_enter", "bear_enter", "threshold", "min_move"):
        value = parameters.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            parameters[key] = float(
                perturb_scalar_kernel(
                    np.float64(value),
                    np.float64(perturbation),
                    np.float64(-np.inf),
                    np.uint8(0),
                )
            )
            changed = True
    if not changed:
        if isinstance(parameters.get("initial_train_size"), (int, float)):
            parameters["initial_train_size"] = int(
                perturb_scalar_kernel(
                    np.float64(parameters["initial_train_size"]),
                    np.float64(perturbation),
                    np.float64(5.0),
                    np.uint8(1),
                )
            )
        else:
            window = int(alternate.get("features", {}).get("window", 20))
            alternate.setdefault("features", {})["window"] = int(
                perturb_scalar_kernel(
                    np.float64(window),
                    np.float64(perturbation),
                    np.float64(2.0),
                    np.uint8(1),
                )
            )
    return alternate


def _formula_provenance_report(
    frame: pd.DataFrame,
    referenced_columns: list[str],
    features: dict[str, Any],
) -> dict[str, Any]:
    configured = features.get("formula_provenance")
    declarations = configured if isinstance(configured, dict) else {}
    observation_dates = pd.to_datetime(frame["observation_date"], errors="coerce")
    row_available = pd.to_datetime(frame["available_at"], errors="coerce")
    columns: list[dict[str, Any]] = []
    for column in referenced_columns:
        if column == "value":
            columns.append(
                {
                    "column": column,
                    "verified": True,
                    "availability_source": "target.available_at",
                    "reason": "主目标字段已通过统一时点数据解析。",
                }
            )
            continue
        declaration = declarations.get(column)
        availability_field = declaration.get("available_at_field") if isinstance(declaration, dict) else None
        if not isinstance(declaration, dict) or declaration.get("point_in_time") is not True:
            columns.append(
                {
                    "column": column,
                    "verified": False,
                    "availability_source": None,
                    "reason": "未声明 point_in_time=true 的字段级来源。",
                }
            )
            continue
        if not isinstance(availability_field, str) or availability_field not in frame.columns:
            columns.append(
                {
                    "column": column,
                    "verified": False,
                    "availability_source": availability_field,
                    "reason": "字段级 available_at_field 不存在。",
                }
            )
            continue
        values_present = pd.to_numeric(frame[column], errors="coerce").notna()
        feature_available = pd.to_datetime(frame[availability_field], errors="coerce")
        valid = (
            ~values_present
            | (
                feature_available.notna()
                & (feature_available >= observation_dates)
                & (feature_available <= row_available)
            )
        )
        invalid_rows = np.flatnonzero(~valid.to_numpy()).astype(int).tolist()
        columns.append(
            {
                "column": column,
                "verified": not invalid_rows,
                "availability_source": availability_field,
                "invalid_rows": invalid_rows[:20],
                "reason": "字段级可得日校验通过。" if not invalid_rows else "字段值早于可得日被使用或可得日无效。",
            }
        )
    return {
        "policy": "value_or_declared_point_in_time_column",
        "verified": all(item["verified"] for item in columns),
        "columns": columns,
    }


class HistoricalRegimeService:
    def __init__(
        self,
        workspace_data_dir: Optional[Path] = None,
        market_data_dir: Optional[Path] = None,
        indicator_service: Optional[CustomIndicatorService] = None,
    ) -> None:
        configured = os.getenv("HISTORICAL_REGIME_DATA_DIR") or os.getenv("CUSTOM_INDICATOR_DATA_DIR")
        self.workspace_data_dir = workspace_data_dir or (Path(configured) if configured else DEFAULT_DATA_DIR)
        self.market_data_dir = market_data_dir or DEFAULT_DATA_DIR
        self.definitions = RegimeDefinitionRepository(self.workspace_data_dir / "historical_regime_definitions.json")
        self.runs = RegimeRunRepository(self.workspace_data_dir / "historical_regime_runs.json")
        self.indicator_service = indicator_service or CustomIndicatorService(
            self.workspace_data_dir,
            self.market_data_dir,
        )
        # Service construction is a lifecycle boundary, never a request path.
        # Fixed signatures are verified here for isolated fixtures and again in
        # the production lifespan before readiness is advertised.
        self.algorithm_runtime = warm_historical_regime_numba_kernels()

    def meta(self) -> dict[str, Any]:
        payload = meta_contract()
        for item in payload["feature_catalog"]:
            item.update(
                {
                    "njit_supported": True,
                    "execution_backend": "numba_njit_fixed_signature",
                }
            )
        for item in payload["algorithm_families"]:
            item.update(
                {
                    "njit_supported": True,
                    "execution_backend": "numba_njit_fixed_signature",
                    "python_fallback": 0,
                }
            )
        formula_meta = payload.get("formula_language") or {}
        formula_meta["operator_catalog"] = [
            {
                "id": operator,
                "njit_supported": True,
                "execution_backend": "numba_njit_fixed_signature",
            }
            for operator in formula_meta.get("operators", [])
        ]
        payload["historical_regime_runtime"] = historical_regime_numba_status()
        items = self.indicator_service.list_indicators(
            context_kind="single_product",
        )["items"]
        payload["indicator_catalog"] = [
            {
                "id": item.get("id"),
                "revision": item.get("revision"),
                "name": item.get("name"),
                "description": item.get("description"),
                "source": item.get("source"),
                "dsl_version": item.get("dsl_version"),
                "operator_registry_version": item.get("operator_registry_version"),
                "compiled_plan_id": item.get("compiled_plan_id"),
                "applicable_product_kinds": copy.deepcopy(
                    item.get("applicable_product_kinds") or []
                ),
                "periods": copy.deepcopy(item.get("periods") or []),
                "execution_backend": "numba_njit_fixed_signature",
                "njit_required": True,
            }
            for item in items
            if str(item.get("dsl_version") or "").startswith("2.")
            and item.get("id")
            and item.get("revision")
        ]
        return payload

    def list_definitions(self) -> list[dict[str, Any]]:
        return self.definitions.list()

    def get_definition(self, definition_id: str, revision: Optional[int] = None) -> dict[str, Any]:
        return self.definitions.get(definition_id, revision)

    def _validate_indicator_reference(
        self,
        definition: dict[str, Any],
    ) -> None:
        target = definition.get("target") or {}
        if target.get("kind") != "indicator":
            return
        indicator = self.indicator_service.get_indicator(
            str(target["indicator_id"]),
            int(target["indicator_revision"]),
        )
        if not str(indicator.get("dsl_version") or "").startswith("2."):
            raise ValidationError(
                "INDICATOR_NJIT_REQUIRED",
                "历史情景定义只能引用 typed DSL 的 NJIT 指标版本。",
                "target.indicator_id",
            )
        if str(indicator.get("context_kind") or "single_product") != "single_product":
            raise ValidationError(
                "INDICATOR_CONTEXT_UNSUPPORTED",
                "历史情景定义不能引用组合指标。",
                "target.indicator_id",
            )
        if target["product_kind"] not in set(
            indicator.get("applicable_product_kinds") or []
        ):
            raise ValidationError(
                "INDICATOR_PRODUCT_KIND_UNSUPPORTED",
                "该指标版本不适用于所选产品类型。",
                "target.product_kind",
            )
        target["name"] = str(
            target.get("name")
            or f"{indicator.get('name')} · {target['product_id']}"
        )
        target["indicator_name"] = indicator.get("name")
        target["indicator_dsl_version"] = indicator.get("dsl_version")
        target["indicator_compiled_plan_id"] = indicator.get(
            "compiled_plan_id"
        )

    def create_definition(self, fields: dict[str, Any]) -> dict[str, Any]:
        normalized = normalize_definition(fields)
        self._validate_indicator_reference(normalized)
        normalized.pop("id", None)
        normalized.pop("revision", None)
        normalized.pop("created_at", None)
        normalized.pop("updated_at", None)
        return self.definitions.create(normalized)

    def update_definition(self, definition_id: str, revision: int, fields: dict[str, Any]) -> dict[str, Any]:
        normalized = normalize_definition(fields)
        self._validate_indicator_reference(normalized)
        for key in ("id", "revision", "created_at", "updated_at"):
            normalized.pop(key, None)
        return self.definitions.update(definition_id, revision, normalized)

    @staticmethod
    def _validate_formula_compatibility(definition: dict[str, Any]) -> str | None:
        formula_expression = definition.get("features", {}).get("formula")
        if formula_expression is None:
            return None
        formula_family = str(definition.get("algorithm", {}).get("family") or "")
        if formula_family in {"merrill_clock", "relative_strength", "ensemble"}:
            raise ValidationError(
                "FORMULA_ALGORITHM_UNSUPPORTED",
                "当前算法使用专用多序列输入，不能安全消费统一公式结果；请改用趋势、峰谷、隐状态或变点算法。",
                "features.formula",
            )
        feature_fields = definition.get("algorithm", {}).get("parameters", {}).get("feature_fields")
        has_explicit_fields = (
            bool(feature_fields)
            if not isinstance(feature_fields, str)
            else bool(feature_fields.strip())
        )
        if has_explicit_fields:
            raise ValidationError(
                "FORMULA_FEATURE_FIELDS_CONFLICT",
                "自定义公式与 feature_fields 都会定义模型输入，请保留其中一种。",
                "features.formula",
            )
        return str(formula_expression)

    def prepare_formula(
        self,
        requested_definition: dict[str, Any],
        mode: str,
        as_of: Optional[str] = None,
    ) -> dict[str, Any]:
        """Explicit preparation boundary; normal runs never compile a plan."""

        if mode not in RUN_MODES:
            raise ValidationError("INVALID_RUN_MODE", "mode 必须是 realtime 或 retrospective。", "mode")
        definition, definition_source = self._resolve_definition(requested_definition)
        self._validate_indicator_reference(definition)
        expression = self._validate_formula_compatibility(definition)
        if expression is None:
            return {
                "required": False,
                "compile_token": None,
                "definition_source": definition_source,
                "request_time_compilation": 0,
            }
        bundle = resolve_target(
            definition["target"],
            mode,
            as_of,
            self.market_data_dir,
            self.indicator_service,
        )
        reporting_frame = bundle.frame.copy()
        if len(reporting_frame) < 5:
            raise ValidationError("INSUFFICIENT_OBSERVATIONS", "历史情景识别至少需要 5 条观测。", "target")
        prepared = prepare_formula_plan(expression, reporting_frame)
        return {
            "required": True,
            "definition_source": definition_source,
            "definition_hash": _content_hash(_business_definition(definition)),
            **prepared,
        }

    def _resolve_definition(self, requested: dict[str, Any]) -> tuple[dict[str, Any], str]:
        if not isinstance(requested, dict):
            raise ValidationError("INVALID_DEFINITION", "run.definition 必须是情景定义或版本引用。", "definition")
        definition_id = requested.get("id")
        if definition_id:
            revision = requested.get("revision")
            only_reference = not requested.get("name") and not requested.get("target") and not requested.get("data")
            if only_reference:
                return self.get_definition(str(definition_id), int(revision) if revision is not None else None), "repository_reference"
            supplied = normalize_definition(requested)
            trial = copy.deepcopy(supplied)
            for key in ("id", "revision", "created_at", "updated_at"):
                trial.pop(key, None)
            try:
                persisted = self.get_definition(str(definition_id), int(revision) if revision is not None else None)
            except IndicatorDomainError:
                return trial, "inline_trial"
            if _content_hash(_business_definition(supplied)) == _content_hash(_business_definition(persisted)):
                return persisted, "repository_reference"
            return trial, "inline_trial"
        return normalize_definition(requested), "inline_trial"

    def run(
        self,
        requested_definition: dict[str, Any],
        mode: str,
        as_of: Optional[str] = None,
        compile_token: Optional[str] = None,
    ) -> dict[str, Any]:
        if mode not in RUN_MODES:
            raise ValidationError("INVALID_RUN_MODE", "mode 必须是 realtime 或 retrospective。", "mode")
        definition, definition_source = self._resolve_definition(requested_definition)
        self._validate_indicator_reference(definition)
        bundle = resolve_target(
            definition["target"],
            mode,
            as_of,
            self.market_data_dir,
            self.indicator_service,
        )
        reporting_frame = bundle.frame.copy()
        if len(reporting_frame) < 5:
            raise ValidationError("INSUFFICIENT_OBSERVATIONS", "历史情景识别至少需要 5 条观测。", "target")
        execution_frame = reporting_frame.copy()
        execution_definition = copy.deepcopy(definition)
        formula_result: Optional[FormulaResult] = None
        formula_expression = self._validate_formula_compatibility(definition)
        if formula_expression is not None:
            formula_result = evaluate_formula(
                formula_expression,
                reporting_frame,
                compile_token=compile_token,
            )
            provenance = _formula_provenance_report(
                reporting_frame,
                list(formula_result.audit["referenced_columns"]),
                definition.get("features", {}),
            )
            formula_result.audit["operator_is_causal"] = True
            formula_result.audit["input_provenance"] = provenance
            formula_result.audit["is_causal"] = bool(provenance["verified"])
            source_values = reporting_frame["value"].to_numpy(dtype=float, copy=True)
            execution_frame["value"] = formula_result.values
            execution_definition.setdefault("features", {})["transform"] = "identity"

        def run_from_reporting_frame(
            raw_frame,
            run_definition: dict[str, Any],
        ):
            derived_frame = raw_frame.copy()
            if formula_expression is not None:
                derived_frame["value"] = evaluate_formula(
                    formula_expression,
                    raw_frame,
                    compile_token=compile_token,
                ).values
            return run_algorithm(derived_frame, run_definition, mode)

        output = run_algorithm(execution_frame, execution_definition, mode)
        if formula_result is not None:
            formula_values = np.ascontiguousarray(
                formula_result.values.to_numpy(dtype=np.float64),
            )
            feature_names = list(output.features)
            numeric_rows = [
                np.ascontiguousarray(output.filtered, dtype=np.float64),
                np.ascontiguousarray(output.scores, dtype=np.float64),
                np.ascontiguousarray(output.confidence, dtype=np.float64),
                *[
                    np.ascontiguousarray(output.features[name], dtype=np.float64)
                    for name in feature_names
                ],
            ]
            try:
                numeric_matrix = np.ascontiguousarray(
                    np.stack(numeric_rows, axis=0),
                    dtype=np.float64,
                )
            except ValueError as exc:
                raise ValidationError(
                    "FORMULA_NUMERIC_POSTPROCESS_INVALID",
                    "公式结果与算法数值结果无法按时间对齐。",
                    "features.formula",
                ) from exc
            masked_numeric, missing_positions, _ = mask_formula_numeric_outputs(
                formula_values,
                numeric_matrix,
            )
            output.filtered = np.ascontiguousarray(masked_numeric[0])
            output.scores = np.ascontiguousarray(masked_numeric[1])
            output.confidence = np.ascontiguousarray(masked_numeric[2])
            for row, name in enumerate(feature_names, start=3):
                output.features[name] = np.ascontiguousarray(masked_numeric[row])
            output.features["formula_input"] = formula_values.copy()
            output.features["source_value"] = source_values
            for index in missing_positions:
                output.labels[int(index)] = UNKNOWN_STATE
                output.probabilities[int(index)] = None
                output.reasons[int(index)] = ["自定义公式在该期结果缺失，未用 0 替代"]
        # 情景状态来自公式派生的执行序列；展示、分段收益与条件表现必须保留原始目标序列口径。
        series = serialise_series(reporting_frame, output, definition["states"])
        frequency = str(definition["target"].get("frequency") or "daily")
        segments = build_segments(series, frequency)
        conditional_metrics = conditional_statistics(series, definition["states"], frequency)
        transition = transition_matrix(series, definition["states"])
        causality = causality_report(output, mode, series)
        data_snapshot = copy.deepcopy(bundle.snapshot)
        if formula_result is not None:
            formula_provenance_verified = bool(formula_result.audit["input_provenance"]["verified"])
            data_snapshot["feature_formula"] = copy.deepcopy(formula_result.audit)
            causality["formula"] = {
                "allowlist_version": formula_result.audit["allowlist_version"],
                "operator_is_causal": True,
                "input_provenance": copy.deepcopy(formula_result.audit["input_provenance"]),
                "is_causal": formula_provenance_verified,
                "uses_future_data": False,
                "repaints": False,
            }
            causality.setdefault("checks", []).append(
                {"id": "formula_allowlist", "passed": True, "version": formula_result.audit["allowlist_version"]}
            )
            causality.setdefault("checks", []).append(
                {"id": "formula_input_provenance", "passed": formula_provenance_verified}
            )
            if not formula_provenance_verified:
                warning = "自定义公式引用了未通过字段级可得日验证的列；结果仅供研究展示，不得用于正式回测或 TAA。"
                causality["is_causal"] = False
                causality["realtime_eligible"] = False
                causality["classification"] = "unverified_input_provenance"
                causality["publish_eligible_usages"] = [
                    usage
                    for usage in causality.get("publish_eligible_usages", [])
                    if usage not in {"formal_backtest", "taa"}
                ]
                causality.setdefault("blockers", []).append(warning)
                causality.setdefault("warnings", []).append(warning)
        if definition_source != "repository_reference":
            causality["publish_eligible_usages"] = []
            causality["publication_blockers"] = ["试算结果没有已保存的定义版本，不能发布或绑定。"]
            causality.setdefault("warnings", []).append("请先保存定义并按版本重新运行，再执行发布。")

        prefix_length, _validation_windows = validation_windows_kernel(
            np.int64(len(execution_frame)),
            np.int64(int(definition.get("validation", {}).get("folds", 4))),
        )
        prefix_length = int(prefix_length)
        prefix_output = None
        try:
            prefix_output = run_from_reporting_frame(
                reporting_frame.iloc[:prefix_length].copy(),
                execution_definition,
            )
        except Exception:
            prefix_output = None
        perturbation = float(definition.get("validation", {}).get("stability_perturbation", 0.1))
        sensitivity_output = None
        try:
            sensitivity_output = run_from_reporting_frame(
                reporting_frame,
                _perturbed_definition(execution_definition, perturbation),
            )
        except Exception:
            sensitivity_output = None
        prefix_metrics = prefix_stability(output, prefix_output, prefix_length)
        state_index = {
            str(state["id"]): index
            for index, state in enumerate(definition["states"])
        }
        label_codes = np.asarray(
            [state_index.get(str(item["state_id"]), -1) for item in series],
            dtype=np.int64,
        )
        label_summary = label_summary_kernel(
            np.ascontiguousarray(label_codes),
        )
        classified_count = int(label_summary[0])
        label_flips = int(label_summary[1])
        stability = {
            "prefix_invariance": prefix_metrics,
            "parameter_sensitivity": {
                "perturbation": perturbation,
                **prefix_stability(output, sensitivity_output, len(execution_frame)),
            },
            "state_switches": int(predecessor_count_kernel(np.int64(len(segments)))),
            "classified_ratio": _json_safe(label_summary[2]),
            "revision_observations": int(data_snapshot.get("revision_observations", 0)),
            "stability_perturbation": perturbation,
            "realtime_monitoring": {
                "prefix_revisions": prefix_metrics.get("revisions"),
                "prefix_revision_rate": prefix_metrics.get("revision_rate"),
                "label_flips": label_flips,
                "label_flip_rate": _json_safe(label_summary[3]),
                "classified_observations": classified_count,
            },
        }

        validation = definition.get("validation", {})
        if validation.get("walk_forward", True):
            walk_forward = walk_forward_report(
                reporting_frame,
                output,
                int(validation.get("folds", 4)),
                lambda prefix: run_from_reporting_frame(prefix, execution_definition),
            )
        else:
            walk_forward = {"status": "not_requested", "folds": []}

        evidence = []
        for item in output.evidence:
            record = dict(item)
            for source_key, target_key in (
                ("index", "observation_date"),
                ("candidate_start_index", "candidate_start"),
                ("recognized_index", "recognized_at"),
            ):
                if source_key in record:
                    point_index = min(max(int(record.pop(source_key)), 0), len(series) - 1)
                    record[target_key] = series[point_index]["observation_date" if target_key != "recognized_at" else "recognized_at"]
            evidence.append(_json_safe(record))
        if formula_result is not None:
            evidence.append({"kind": "formula_compilation", **copy.deepcopy(formula_result.audit)})
        if definition["target"].get("kind") == "indicator":
            evidence.append(
                {
                    "kind": "indicator_version_execution",
                    "indicator_id": data_snapshot.get("indicator_id"),
                    "indicator_revision": data_snapshot.get("indicator_revision"),
                    "indicator_definition_hash": data_snapshot.get(
                        "indicator_definition_hash"
                    ),
                    "series_hash": data_snapshot.get("series_hash"),
                    "compiled_plan_id": (data_snapshot.get("plan") or {}).get(
                        "compiled_plan_id"
                    ),
                }
            )

        definition_snapshot = copy.deepcopy(definition)
        definition_fingerprint = _content_hash(_business_definition(definition))
        definition_id = definition.get("id") if definition_source == "repository_reference" else None
        definition_revision = int(definition.get("revision")) if definition_source == "repository_reference" else None
        diagnostics = [
            {
                "code": "EXECUTION_PATH",
                "level": "info",
                "message": "本次按实时可得数据执行。" if mode == "realtime" else "本次为事后研究口径，不可直接视为当时可交易信号。",
            }
        ]
        diagnostics.extend(
            {"code": "CAUSALITY_BLOCKER", "level": "error", "message": message}
            for message in causality.get("blockers", [])
        )
        if formula_result is not None:
            diagnostics.append(
                {
                    "code": "CAUSAL_FORMULA_COMPILED",
                    "level": "info",
                    "message": f"自定义公式已通过因果白名单校验（版本 {formula_result.audit['allowlist_version']}）。",
                }
            )
        diagnostics.append(
            {
                "code": "NJIT_HISTORICAL_REGIME_EXECUTED",
                "level": "info",
                "message": (
                    "历史情景算法与条件统计已通过启动预热的固定签名 NJIT "
                    "内核执行，Python 回退为 0。"
                ),
            }
        )
        if definition["target"].get("kind") == "indicator":
            diagnostics.append(
                {
                    "code": "NJIT_INDICATOR_VERSION_EXECUTED",
                    "level": "info",
                    "message": (
                        "已按指标中心精确版本执行预热的 typed AST → DAG → NJIT 计划，"
                        "缺失值未替换为 0。"
                    ),
                }
            )
        algorithm_diagnostics = _json_safe(output.diagnostics)
        algorithm_diagnostics["analytics_execution"] = _json_safe(
            analytics_execution_audit()
        )
        if formula_result is not None:
            algorithm_diagnostics["feature_formula"] = copy.deepcopy(formula_result.audit)
        diagnostics.extend(
            {"code": "CAUSALITY_WARNING", "level": "warning", "message": message}
            for message in causality.get("warnings", [])
        )
        calculation_audits: list[dict[str, Any]] = []
        if definition["target"].get("kind") == "indicator":
            indicator_audit = copy.deepcopy(
                data_snapshot.get("calculation_audit") or {}
            )
            calculation_audits.append(
                {"source_kind": "indicator", **indicator_audit}
            )
        if formula_result is not None:
            calculation_audits.append(
                {
                    "source_kind": "formula",
                    "dag": copy.deepcopy(formula_result.audit.get("dag") or {}),
                    "plan": copy.deepcopy(formula_result.audit),
                }
            )
        calculation_audits.extend(
            [
                copy.deepcopy(output.diagnostics["execution_audit"]),
                analytics_execution_audit(),
            ]
        )
        run_payload = {
            "definition_id": definition_id,
            "definition_revision": definition_revision,
            "definition_source": definition_source,
            "definition_snapshot_hash": definition_fingerprint,
            "name": definition["name"],
            "mode": mode,
            "as_of": as_of,
            "schema_version": "1.0",
            "definition": definition_snapshot,
            "target": copy.deepcopy(definition["target"]),
            "algorithm": copy.deepcopy(definition["algorithm"]),
            "states": copy.deepcopy(definition["states"]),
            "data_snapshot": data_snapshot,
            "series": series,
            "segments": segments,
            "conditional_metrics": conditional_metrics,
            "conditional_stats": conditional_metrics,
            "transition": transition,
            "causality": causality,
            "stability": stability,
            "walk_forward": walk_forward,
            "evidence": evidence,
            "diagnostics": diagnostics,
            "algorithm_diagnostics": algorithm_diagnostics,
            "formula_diagnostics": copy.deepcopy(formula_result.audit) if formula_result is not None else None,
            "calculation_audit": calculation_audits[0] if calculation_audits else None,
            "calculation_audits": calculation_audits,
            "application_bindings": [],
        }
        run_payload["content_hash"] = _content_hash(run_payload)
        return self.runs.create(_json_safe(run_payload))

    def list_runs(self, definition_id: Optional[str] = None) -> list[dict[str, Any]]:
        return self.runs.list(definition_id)

    def get_run(self, run_id: str) -> dict[str, Any]:
        return self.runs.get(run_id)

    def publish(
        self,
        run_id: str,
        usage: str | list[str],
        note: str = "",
    ) -> dict[str, Any]:
        run = self.runs.get(run_id)
        usages = [usage] if isinstance(usage, str) else list(usage or [])
        usages = list(dict.fromkeys(str(item) for item in usages))
        if not usages or any(item not in APPLICATION_TARGETS for item in usages):
            raise ValidationError("INVALID_PUBLICATION_USAGE", "usage 包含不支持的应用目标。", "usage")
        if run.get("definition_source") != "repository_reference" or int(run.get("definition_revision") or 0) <= 0:
            raise ValidationError(
                "UNVERSIONED_RUN_PUBLICATION_BLOCKED",
                "试算结果没有已保存的定义版本；请先保存定义并按该版本重新运行。",
                "run_id",
            )
        try:
            persisted_definition = self.definitions.get(
                str(run.get("definition_id")),
                int(run["definition_revision"]),
            )
        except IndicatorDomainError as exc:
            raise ValidationError(
                "RUN_DEFINITION_VERSION_MISSING",
                "运行引用的定义版本已不存在，不能发布。",
                "run_id",
            ) from exc
        persisted_hash = _content_hash(_business_definition(persisted_definition))
        if persisted_hash != run.get("definition_snapshot_hash"):
            raise ValidationError(
                "RUN_DEFINITION_LINEAGE_MISMATCH",
                "运行快照与已保存定义版本不一致，不能发布。",
                "run_id",
            )
        eligible = set(run.get("causality", {}).get("publish_eligible_usages", []))
        blocked = [item for item in usages if item not in eligible]
        if blocked:
            raise ValidationError(
                "NON_CAUSAL_PUBLICATION_BLOCKED",
                f"该运行不能发布到 {', '.join(blocked)}；请使用实时、因果且不重绘的结果。",
                "usage",
                diagnostics=[
                    {
                        "requested": usages,
                        "eligible": sorted(eligible),
                        "causality": run.get("causality"),
                    }
                ],
            )
        if len(note) > 500:
            raise ValidationError("PUBLICATION_NOTE_TOO_LONG", "发布说明不能超过 500 个字符。", "note")
        publications = [
            {
                "id": f"publication-{uuid.uuid4().hex}",
                "usage": item,
                "published_at": utc_now(),
                # `published_at` is when the desk started using this model;
                # `fit_as_of` is what the model itself was allowed to know. A
                # regime published today but fitted on the full history is not
                # a signal a 2018 portfolio could have acted on, and only these
                # two dates side by side make that visible.
                "fit_as_of": run.get("as_of"),
                "fit_mode": run.get("mode"),
                "note": note,
                "run_id": run_id,
                "definition_revision": int(run.get("definition_revision") or 0),
                "run_content_hash": run["content_hash"],
                "gate": "causality_passed" if item in {"taa", "formal_backtest"} else "research_only",
            }
            for item in usages
        ]
        updated = self.runs.add_publications(run_id, publications)
        return {
            "run_id": run_id,
            "publication": publications[0],
            "publications": updated.get("publications", []),
            "application_bindings": updated.get("application_bindings", []),
        }

    def taa_backtest(self, run_id: str, request: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(request, dict):
            raise ValidationError("INVALID_TAA_REQUEST", "TAA 回测请求必须是对象。", "request")
        run = self.runs.get(run_id)
        if run.get("immutable") is not True:
            raise ValidationError("REGIME_RUN_NOT_IMMUTABLE", "TAA 回测只接受不可变的历史情景运行快照。", "run_id")
        if run.get("mode") != "realtime":
            raise ValidationError("TAA_REQUIRES_REALTIME_RUN", "TAA 回测只接受 realtime 历史情景运行。", "run_id")
        causality = run.get("causality") if isinstance(run.get("causality"), dict) else {}
        if (
            not causality.get("is_causal")
            or causality.get("uses_future_data")
            or causality.get("repaints")
            or not causality.get("realtime_eligible")
        ):
            raise ValidationError("TAA_REQUIRES_CAUSAL_RUN", "TAA 回测只接受因果且不重绘的历史情景运行。", "run_id")

        analytical_snapshot = copy.deepcopy(run)
        for key in ("id", "created_at", "immutable", "publications", "content_hash"):
            analytical_snapshot.pop(key, None)
        analytical_snapshot["application_bindings"] = []
        expected_hash = _content_hash(analytical_snapshot)
        if expected_hash != run.get("content_hash"):
            raise ValidationError("REGIME_RUN_SNAPSHOT_MISMATCH", "历史情景运行快照校验失败，不能执行 TAA 回测。", "run_id")

        accepted_publications = []
        for publication in run.get("publications") or []:
            if publication.get("usage") not in {"taa", "formal_backtest"}:
                continue
            if (
                publication.get("run_id") != run_id
                or publication.get("run_content_hash") != run.get("content_hash")
                or publication.get("definition_revision") != run.get("definition_revision")
                or publication.get("gate") != "causality_passed"
            ):
                continue
            accepted_publications.append(publication)
        if not accepted_publications:
            raise ValidationError(
                "TAA_RUN_NOT_PUBLISHED",
                "历史情景运行必须先发布到 TAA 或正式回测，才能执行战术配置回测。",
                "run_id",
            )
        gate = {
            "passed": True,
            "immutable": True,
            "mode": "realtime",
            "causal": True,
            "publication_usages": sorted({item["usage"] for item in accepted_publications}),
            "publication_ids": sorted(str(item["id"]) for item in accepted_publications),
            "run_content_hash": run["content_hash"],
        }
        return execute_taa_backtest(run, request, gate)

    def compare(self, run_ids: list[str], reference_run_id: Optional[str] = None) -> dict[str, Any]:
        unique_ids = list(dict.fromkeys(str(item) for item in run_ids))
        if len(unique_ids) < 2 or len(unique_ids) > 8:
            raise ValidationError("INVALID_COMPARE_RUNS", "模型比较需要 2 至 8 个不同运行。", "run_ids")
        if reference_run_id and reference_run_id not in unique_ids:
            raise ValidationError("INVALID_REFERENCE_RUN", "reference_run_id 必须包含在 run_ids 中。", "reference_run_id")
        runs = [self.runs.get(run_id) for run_id in unique_ids]
        result = compare_run_snapshots(runs, reference_run_id)
        return {**result, "compared_at": utc_now()}

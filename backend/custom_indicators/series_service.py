"""Compile, warm and evaluate named time-series indicator bundles."""

from __future__ import annotations

import ast
import copy
import hashlib
import hmac
import json
import math
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

import numpy as np

from cal_indicators.typed_dsl import (
    TypedDslError,
    TypedSeriesBundlePlan,
    compose_typed_series_bundle,
)
from cal_indicators.typed_latex import (
    MATH_NOTATION_VERSION,
    render_python_expression_latex,
)
from cal_indicators.typed_numba_plan import (
    CompiledNumbaSeriesPlan,
    NumbaPlanCompileError,
    compile_numba_series_plan,
    get_cached_numba_series_plan,
    persist_numba_series_plan,
)
from compute_policy import NJIT_BACKEND, validate_execution_audit

from .errors import ValidationError
from .formula_source import canonical_formula_source, editable_formula_latex
from .excel_export import (
    ExcelExportArtifact,
    SeriesExcelTargetEvidence,
    build_series_indicator_excel_workbook,
)
from .presentation import metric_presentation
from .series_parameters import parameter_hash
from .repository import IndicatorRepository
from .runtime_context import (
    RUNTIME_SCALAR_CONTEXT_NAMES,
    single_product_scalar_context,
)
from .series_definitions import (
    TIME_SERIES_RESULT_KIND,
    normalize_series_parameters,
    normalize_time_series_definition,
    parameter_variable_types,
    resolve_series_output_measure,
    series_expressions,
    series_output_measure_catalog,
)
from .series_provider import (
    DEFAULT_DATA_DIR,
    ProductChartSeries,
    load_product_chart_series,
    market_data_generation,
    select_chart_window,
)
from .variable_registry import (
    canonicalize_variables,
    get_variable,
    variable_latex_symbols,
    variable_types,
)


MAX_SERIES_INSTANCES = 10
MAX_SERIES_DISPLAY_POINTS = 5_000


class ResultCache(Protocol):
    def get(self, key: str) -> dict[str, Any] | None: ...

    def put(self, key: str, value: dict[str, Any]) -> None: ...


_SERIES_PLAN_LOCK = threading.RLock()
_WARMED_SERIES_PLANS: dict[
    str, tuple[TypedSeriesBundlePlan, CompiledNumbaSeriesPlan]
] = {}


_SERIES_FIXED_ARGUMENTS: dict[str, frozenset[str]] = {
    "rolling_mean": frozenset({"window", "min_periods"}),
    "rolling_std": frozenset({"window", "ddof", "min_periods"}),
    "rolling_min": frozenset({"window", "min_periods"}),
    "rolling_max": frozenset({"window", "min_periods"}),
    "recursive_smooth": frozenset({"periods", "initial"}),
    "divide_or_default": frozenset({"default"}),
    "lag": frozenset({"periods"}),
    "difference": frozenset({"periods"}),
    "variance": frozenset({"ddof"}),
    "std": frozenset({"ddof"}),
    "quantile": frozenset({"probability"}),
    "quantile_where": frozenset({"probability"}),
    "clip": frozenset({"lower", "upper"}),
    "power": frozenset({"exponent"}),
}
_SERIES_CONSTANT_OPERATORS = frozenset(
    {"negate", "add", "subtract", "multiply", "divide", "power"}
)


def _array_context_names(
    plan: TypedSeriesBundlePlan,
    parameters: Mapping[str, float] | None = None,
) -> tuple[str, ...]:
    parameter_ids = set(parameters or {})
    return tuple(
        name
        for name, value_type in plan.context_requirements.items()
        if name not in parameter_ids and value_type.rank > 0
    )


def _derived_boundary_observations(dependencies: Iterable[str]) -> int:
    """Return extra source rows needed before a derived return window.

    ``returns[t]`` and ``log_returns[t]`` need the adjusted NAV at ``t-1``.
    A W-observation rolling statistic therefore needs W rows before the first
    displayed date rather than the usual W-1 rows used by direct price series.
    """

    return 1 if {"returns", "log_returns"} & set(dependencies) else 0


def _build_series_runtime_context(
    definition: Mapping[str, Any],
    plan: TypedSeriesBundlePlan,
    compiled: CompiledNumbaSeriesPlan,
    frame: Any,
    parameters: Mapping[str, float],
) -> dict[str, Any]:
    dates = list(frame["date"])
    returns = (
        np.ascontiguousarray(
            frame["returns"].to_numpy(dtype=np.float64, copy=False)
        )
        if "returns" in frame.columns
        else None
    )
    scalar_values = single_product_scalar_context(
        definition,
        dates,
        returns=returns,
    )
    context: dict[str, Any] = {}
    for name in compiled.context_names:
        if name in parameters:
            context[name] = float(parameters[name])
            continue
        value_type = plan.context_requirements[name]
        if value_type.rank == 0:
            if name not in RUNTIME_SCALAR_CONTEXT_NAMES or name not in scalar_values:
                raise ValidationError(
                    "SERIES_SCALAR_CONTEXT_UNSUPPORTED",
                    f"时序指标尚未定义标量上下文 {name} 的运行语义。",
                    field="series_outputs",
                )
            context[name] = float(scalar_values[name])
            continue
        if name not in frame.columns:
            raise ValidationError(
                "VARIABLE_UNAVAILABLE",
                f"时序指标缺少输入变量 {name}。",
                field="series_outputs",
            )
        context[name] = np.ascontiguousarray(
            frame[name].to_numpy(dtype=np.float64, copy=False)
        )
    return context


def _constant_series_node(
    node_id: int,
    nodes: Mapping[int, Any],
    memo: dict[int, bool],
) -> bool:
    cached = memo.get(node_id)
    if cached is not None:
        return cached
    node = nodes[node_id]
    if node.kind == "constant":
        memo[node_id] = True
        return True
    constant = bool(
        node.operator_id in _SERIES_CONSTANT_OPERATORS
        and node.inputs
        and all(_constant_series_node(int(child), nodes, memo) for child in node.inputs)
    )
    memo[node_id] = constant
    return constant


def _validate_fixed_series_configuration(
    plan: TypedSeriesBundlePlan, parameter_ids: set[str] | None = None,
) -> None:
    """Reject data-dependent algorithm configuration before NJIT compilation."""

    nodes = {int(node.node_id): node for node in plan.nodes}
    memo: dict[int, bool] = {}
    diagnostics: list[dict[str, Any]] = []
    for node in plan.nodes:
        fixed_names = _SERIES_FIXED_ARGUMENTS.get(str(node.operator_id or ""))
        if not fixed_names:
            continue
        for parameter_name, input_node_id in node.arguments:
            if parameter_name not in fixed_names:
                continue
            input_node = nodes[int(input_node_id)]
            if input_node.kind == "variable" and str(input_node.label) in (parameter_ids or set()):
                continue
            if _constant_series_node(int(input_node_id), nodes, memo):
                continue
            diagnostics.append(
                {
                    "code": "SERIES_CONFIGURATION_MUST_BE_CONSTANT",
                    "message": (
                        f"{node.operator_id} 的参数 {parameter_name} 必须是定义级有限数值常量；"
                        "修改该参数应创建新的指标公式或版本。"
                    ),
                    "field": "series_outputs",
                    "node_id": int(node.node_id),
                    "operator": node.operator_id,
                    "parameter": parameter_name,
                }
            )
    if diagnostics:
        first = diagnostics[0]
        raise ValidationError(
            str(first["code"]),
            str(first["message"]),
            field="series_outputs",
            diagnostics=diagnostics,
        )


def _definition_key(definition: Mapping[str, Any]) -> str:
    payload = {
        "result_kind": definition.get("result_kind"),
        "series_outputs": [
            {
                "id": item.get("id"),
                "expression": item.get("expression"),
            }
            for item in definition.get("series_outputs") or []
        ],
        "parameter_schema": definition.get("parameter_schema"),
        "parameter_contract_version": definition.get("parameter_contract_version"),
        "axis_anchor": definition.get("axis_anchor"),
        # History policy and output presentation are compiler-derived metadata;
        # neither changes the numerical DAG or its fixed NJIT signature.
        "dsl_version": definition.get("dsl_version"),
        "operator_registry_version": definition.get("operator_registry_version"),
        "variable_registry_version": definition.get("variable_registry_version"),
        "data_contract_version": definition.get("data_contract_version"),
        "context_schema_version": definition.get("context_schema_version"),
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _definition_contract_key(definition: Mapping[str, Any]) -> str:
    payload = {
        "plan_key": _definition_key(definition),
        "result_kind": definition.get("result_kind"),
        "output_contract": definition.get("output_contract"),
        "axis_anchor": definition.get("axis_anchor"),
        "annual_risk_free_rate_percent": float(
            definition.get("annual_risk_free_rate_percent") or 0.0
        ),
        "series_outputs": [
            {
                "id": item.get("id"),
                "label": item.get("label"),
                "expression": item.get("expression"),
                "unit": item.get("unit"),
                "display_format": item.get("display_format"),
                "precision": item.get("precision"),
                "output_measure": item.get("output_measure"),
            }
            for item in definition.get("series_outputs") or []
        ],
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _compile_token(
    definition: Mapping[str, Any], compiled_plan_id: str
) -> str:
    payload = {
        "definition_contract_key": _definition_contract_key(definition),
        "compiled_plan_id": compiled_plan_id,
        "token_version": "time-series-njit-compile-token-v2",
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _compile_definition(
    definition: Mapping[str, Any],
) -> tuple[TypedSeriesBundlePlan, CompiledNumbaSeriesPlan]:
    key = _definition_key(definition)
    with _SERIES_PLAN_LOCK:
        warmed = _WARMED_SERIES_PLANS.get(key)
        if warmed is not None:
            return warmed
    parameter_types = parameter_variable_types(definition)
    context_types = {
        **variable_types("single_product", str(definition["dsl_version"])),
        **parameter_types,
    }
    try:
        plan = compose_typed_series_bundle(
            {
                channel_id: canonical_formula_source(expression)
                for channel_id, expression in series_expressions(definition).items()
            },
            variable_types=context_types,
            dsl_version=str(definition["dsl_version"]),
            operator_registry_version=str(definition["operator_registry_version"]),
        )
        _validate_fixed_series_configuration(plan, set(parameter_types))
        compiled = compile_numba_series_plan(plan)
    except (NumbaPlanCompileError, TypedDslError) as exc:
        diagnostics = [
            exc.to_dict()
            if isinstance(exc, TypedDslError)
            else {
                "code": "NJIT_SERIES_PLAN_COMPILE_FAILED",
                "message": "时序指标无法编译为固定签名 NJIT 计划。",
                "compiled_plan_id": exc.plan_id,
                "operator": exc.operator_id,
            }
        ]
        raise ValidationError(
            "NJIT_SERIES_PLAN_COMPILE_FAILED",
            "时序指标无法编译为固定签名 NJIT 计划。",
            field="series_outputs",
            diagnostics=diagnostics,
        ) from exc
    with _SERIES_PLAN_LOCK:
        _WARMED_SERIES_PLANS[key] = (plan, compiled)
    return plan, compiled


def _get_warmed_definition(
    definition: Mapping[str, Any],
) -> tuple[TypedSeriesBundlePlan, CompiledNumbaSeriesPlan]:
    key = _definition_key(definition)
    with _SERIES_PLAN_LOCK:
        warmed = _WARMED_SERIES_PLANS.get(key)
    if warmed is None:
        raise ValidationError(
            "NJIT_SERIES_PLAN_NOT_WARMED",
            "时序指标版本尚未完成显式预热；运行已关闭，未回退到 Python。",
            field="indicator_revision",
        )
    plan, compiled = warmed
    cached = get_cached_numba_series_plan(plan)
    if cached is None or cached.plan_id != compiled.plan_id:
        raise ValidationError(
            "NJIT_SERIES_PLAN_NOT_WARMED",
            "时序指标固定签名计划未命中运行缓存；运行已关闭。",
            field="indicator_revision",
        )
    return plan, cached


@dataclass(frozen=True)
class _HistoryRequirement:
    full_history: bool
    lookback_observations: int
    minimum_observations: int


def _fixed_parameter_values(definition: Mapping[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    for item in definition.get("parameter_schema") or []:
        try:
            value = float(item.get("default"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values[str(item.get("id") or "")] = value
    for item in definition.get("fixed_parameters") or []:
        try:
            value = float(item.get("value"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values[str(item.get("id") or "")] = value
    return values


def _constant_number(
    node_by_id: Mapping[int, Any],
    node_id: int,
    fixed_parameters: Mapping[str, float],
    memo: dict[int, float | None],
) -> float | None:
    if node_id in memo:
        return memo[node_id]
    node = node_by_id[node_id]
    result: float | None = None
    if node.kind == "constant":
        try:
            result = float(node.label)
        except (TypeError, ValueError):
            result = None
    elif node.kind == "variable" and str(node.label) in fixed_parameters:
        result = float(fixed_parameters[str(node.label)])
    else:
        values = [
            _constant_number(node_by_id, int(input_id), fixed_parameters, memo)
            for input_id in node.inputs
        ]
        operator_id = str(node.operator_id or "")
        if all(value is not None for value in values):
            args = [float(value) for value in values if value is not None]
            try:
                if operator_id == "negate":
                    result = -args[0]
                elif operator_id == "absolute":
                    result = abs(args[0])
                elif operator_id == "sqrt" and args[0] >= 0.0:
                    result = math.sqrt(args[0])
                elif operator_id == "reciprocal" and abs(args[0]) >= 1e-12:
                    result = 1.0 / args[0]
                elif operator_id == "add":
                    result = args[0] + args[1]
                elif operator_id == "subtract":
                    result = args[0] - args[1]
                elif operator_id == "multiply":
                    result = args[0] * args[1]
                elif operator_id == "divide" and abs(args[1]) >= 1e-12:
                    result = args[0] / args[1]
                elif operator_id == "power":
                    result = math.pow(args[0], args[1])
                elif operator_id == "minimum":
                    result = min(args[0], args[1])
                elif operator_id == "maximum":
                    result = max(args[0], args[1])
                elif operator_id == "clip" and args[1] <= args[2]:
                    result = min(max(args[0], args[1]), args[2])
                elif operator_id == "divide_or_default":
                    result = args[2] if abs(args[1]) < 1e-12 else args[0] / args[1]
            except (OverflowError, ValueError, ZeroDivisionError):
                result = None
    if result is not None and not math.isfinite(result):
        result = None
    memo[node_id] = result
    return result


def _required_configuration_number(
    *,
    plan: TypedSeriesBundlePlan,
    node_by_id: Mapping[int, Any],
    node_id: int,
    fixed_parameters: Mapping[str, float],
    memo: dict[int, float | None],
    operator_id: str,
    parameter: str,
    integer: bool,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    value = _constant_number(node_by_id, node_id, fixed_parameters, memo)
    if value is None:
        raise ValidationError(
            "SERIES_CONFIGURATION_MUST_BE_CONSTANT",
            f"{operator_id} 的 {parameter} 必须是公式中固定的有限数值常量。",
            field="series_outputs",
            diagnostics=[
                {
                    "code": "SERIES_CONFIGURATION_MUST_BE_CONSTANT",
                    "message": "窗口、期数、ddof 等算法配置不能来自运行时变量。",
                    "operator": operator_id,
                    "parameter": parameter,
                    "node_id": node_id,
                }
            ],
        )
    if integer and not float(value).is_integer():
        raise ValidationError(
            "SERIES_CONFIGURATION_MUST_BE_INTEGER",
            f"{operator_id} 的 {parameter} 必须是整数常量。",
            field="series_outputs",
        )
    if minimum is not None and value < minimum:
        raise ValidationError(
            "SERIES_CONFIGURATION_OUT_OF_RANGE",
            f"{operator_id} 的 {parameter} 不能小于 {minimum:g}。",
            field="series_outputs",
        )
    if maximum is not None and value > maximum:
        raise ValidationError(
            "SERIES_CONFIGURATION_OUT_OF_RANGE",
            f"{operator_id} 的 {parameter} 不能大于 {maximum:g}。",
            field="series_outputs",
        )
    return value


def _history_requirement(
    plan: TypedSeriesBundlePlan,
    node_id: int,
    definition: Mapping[str, Any],
    memo: dict[int, _HistoryRequirement],
    constant_memo: dict[int, float | None],
) -> _HistoryRequirement:
    if node_id in memo:
        return memo[node_id]
    node_by_id = {node.node_id: node for node in plan.nodes}
    node = node_by_id[node_id]
    if node.kind in {"variable", "constant"}:
        requirement = _HistoryRequirement(False, 1, 1)
        memo[node_id] = requirement
        return requirement

    children = [
        _history_requirement(
            plan,
            int(input_id),
            definition,
            memo,
            constant_memo,
        )
        for input_id in node.inputs
    ]
    requirement = _HistoryRequirement(
        any(item.full_history for item in children),
        max((item.lookback_observations for item in children), default=1),
        max((item.minimum_observations for item in children), default=1),
    )
    operator_id = str(node.operator_id or "")
    fixed_parameters = _fixed_parameter_values(definition)
    maximum_window = 20_000.0

    if operator_id in {"rolling_mean", "rolling_min", "rolling_max", "rolling_std"}:
        values_requirement = children[0]
        window = int(
            _required_configuration_number(
                plan=plan,
                node_by_id=node_by_id,
                node_id=int(node.inputs[1]),
                fixed_parameters=fixed_parameters,
                memo=constant_memo,
                operator_id=operator_id,
                parameter="window",
                integer=True,
                minimum=1.0,
                maximum=maximum_window,
            )
        )
        if operator_id == "rolling_std":
            ddof = int(
                _required_configuration_number(
                    plan=plan,
                    node_by_id=node_by_id,
                    node_id=int(node.inputs[2]),
                    fixed_parameters=fixed_parameters,
                    memo=constant_memo,
                    operator_id=operator_id,
                    parameter="ddof",
                    integer=True,
                    minimum=0.0,
                    maximum=float(window - 1),
                )
            ) if len(node.inputs) >= 3 else 0
            min_periods = int(
                _required_configuration_number(
                    plan=plan,
                    node_by_id=node_by_id,
                    node_id=int(node.inputs[3]),
                    fixed_parameters=fixed_parameters,
                    memo=constant_memo,
                    operator_id=operator_id,
                    parameter="min_periods",
                    integer=True,
                    minimum=1.0,
                    maximum=float(window),
                )
            ) if len(node.inputs) >= 4 else window
            required_valid = max(min_periods, ddof + 1)
        else:
            min_periods = int(
                _required_configuration_number(
                    plan=plan,
                    node_by_id=node_by_id,
                    node_id=int(node.inputs[2]),
                    fixed_parameters=fixed_parameters,
                    memo=constant_memo,
                    operator_id=operator_id,
                    parameter="min_periods",
                    integer=True,
                    minimum=1.0,
                    maximum=float(window),
                )
            ) if len(node.inputs) >= 3 else window
            required_valid = min_periods
        requirement = _HistoryRequirement(
            values_requirement.full_history,
            values_requirement.lookback_observations + window - 1,
            values_requirement.minimum_observations + required_valid - 1,
        )
    elif operator_id in {"lag", "difference"}:
        periods = int(
            _required_configuration_number(
                plan=plan,
                node_by_id=node_by_id,
                node_id=int(node.inputs[1]),
                fixed_parameters=fixed_parameters,
                memo=constant_memo,
                operator_id=operator_id,
                parameter="periods",
                integer=True,
                minimum=1.0,
                maximum=maximum_window,
            )
        ) if len(node.inputs) >= 2 else 1
        values_requirement = children[0]
        requirement = _HistoryRequirement(
            values_requirement.full_history,
            values_requirement.lookback_observations + periods,
            values_requirement.minimum_observations + periods,
        )
    elif operator_id == "recursive_smooth":
        _required_configuration_number(
            plan=plan,
            node_by_id=node_by_id,
            node_id=int(node.inputs[1]),
            fixed_parameters=fixed_parameters,
            memo=constant_memo,
            operator_id=operator_id,
            parameter="periods",
            integer=True,
            minimum=1.0,
            maximum=maximum_window,
        )
        _required_configuration_number(
            plan=plan,
            node_by_id=node_by_id,
            node_id=int(node.inputs[2]),
            fixed_parameters=fixed_parameters,
            memo=constant_memo,
            operator_id=operator_id,
            parameter="initial",
            integer=False,
        )
        requirement = _HistoryRequirement(
            True,
            children[0].lookback_observations,
            children[0].minimum_observations,
        )
    elif operator_id in {
        "cumulative_sum",
        "cumulative_product",
        "cumulative_return",
        "cumulative_max",
        "cumulative_min",
        "drawdown_series",
        "new_high_mask",
    }:
        requirement = _HistoryRequirement(
            True,
            requirement.lookback_observations,
            requirement.minimum_observations,
        )

    memo[node_id] = requirement
    return requirement


def _series_history_contract(
    definition: Mapping[str, Any],
    plan: TypedSeriesBundlePlan,
) -> dict[str, Any]:
    memo: dict[int, _HistoryRequirement] = {}
    constant_memo: dict[int, float | None] = {}
    requirements = [
        _history_requirement(
            plan,
            int(root_id),
            definition,
            memo,
            constant_memo,
        )
        for root_id in plan.roots.values()
    ]
    full_history = any(item.full_history for item in requirements)
    return {
        "history_policy": "full_history" if full_history else "lookback",
        "lookback_observations": max(
            (item.lookback_observations for item in requirements),
            default=1,
        ),
        "minimum_observations": max(
            (item.minimum_observations for item in requirements),
            default=1,
        ),
        "history_inference_source": "typed_dag",
    }


def _proven_numeric_range(
    plan: TypedSeriesBundlePlan,
    node_id: int,
    definition: Mapping[str, Any],
    memo: dict[int, tuple[float, float] | None],
    constant_memo: dict[int, float | None],
) -> tuple[float, float] | None:
    if node_id in memo:
        return memo[node_id]
    node_by_id = {node.node_id: node for node in plan.nodes}
    node = node_by_id[node_id]
    fixed_parameters = _fixed_parameter_values(definition)
    constant = _constant_number(
        node_by_id,
        node_id,
        fixed_parameters,
        constant_memo,
    )
    if constant is not None:
        result: tuple[float, float] | None = (constant, constant)
        memo[node_id] = result
        return result
    child_ranges = [
        _proven_numeric_range(
            plan,
            int(input_id),
            definition,
            memo,
            constant_memo,
        )
        for input_id in node.inputs
    ]
    operator_id = str(node.operator_id or "")
    result = None
    try:
        if operator_id == "clip":
            lower = _constant_number(
                node_by_id, int(node.inputs[1]), fixed_parameters, constant_memo
            )
            upper = _constant_number(
                node_by_id, int(node.inputs[2]), fixed_parameters, constant_memo
            )
            if lower is not None and upper is not None and lower <= upper:
                result = (lower, upper)
        elif operator_id == "sign":
            result = (-1.0, 1.0)
        elif operator_id == "absolute" and child_ranges[0] is not None:
            lower, upper = child_ranges[0]
            result = (0.0, max(abs(lower), abs(upper)))
        elif operator_id in {"rolling_mean", "rolling_min", "rolling_max"}:
            result = child_ranges[0]
        elif operator_id == "recursive_smooth" and child_ranges[0] is not None:
            initial = _constant_number(
                node_by_id, int(node.inputs[2]), fixed_parameters, constant_memo
            )
            if initial is not None:
                result = (
                    min(child_ranges[0][0], initial),
                    max(child_ranges[0][1], initial),
                )
        elif operator_id in {"add", "subtract", "multiply"} and all(
            item is not None for item in child_ranges[:2]
        ):
            left = child_ranges[0]
            right = child_ranges[1]
            assert left is not None and right is not None
            if operator_id == "add":
                result = (left[0] + right[0], left[1] + right[1])
            elif operator_id == "subtract":
                result = (left[0] - right[1], left[1] - right[0])
            else:
                products = (
                    left[0] * right[0],
                    left[0] * right[1],
                    left[1] * right[0],
                    left[1] * right[1],
                )
                result = (min(products), max(products))
        elif operator_id in {"minimum", "maximum"} and all(
            item is not None for item in child_ranges[:2]
        ):
            left = child_ranges[0]
            right = child_ranges[1]
            assert left is not None and right is not None
            if operator_id == "minimum":
                result = (min(left[0], right[0]), min(left[1], right[1]))
            else:
                result = (max(left[0], right[0]), max(left[1], right[1]))
        elif operator_id == "where" and child_ranges[1] is not None and child_ranges[2] is not None:
            result = (
                min(child_ranges[1][0], child_ranges[2][0]),
                max(child_ranges[1][1], child_ranges[2][1]),
            )
    except (IndexError, OverflowError, ValueError):
        result = None
    if result is not None and not all(math.isfinite(value) for value in result):
        result = None
    memo[node_id] = result
    return result


def _series_measure_contract(
    definition: Mapping[str, Any],
    plan: TypedSeriesBundlePlan,
) -> dict[str, dict[str, Any]]:
    metadata = {
        str(item["id"]): item for item in definition.get("series_outputs") or []
    }
    range_memo: dict[int, tuple[float, float] | None] = {}
    constant_memo: dict[int, float | None] = {}
    contract: dict[str, dict[str, Any]] = {}
    for channel_id, root_id in plan.roots.items():
        contract[channel_id] = resolve_series_output_measure(
            str(metadata[channel_id].get("output_measure") or "auto"),
            plan.output_types[channel_id],
            proven_range=_proven_numeric_range(
                plan,
                int(root_id),
                definition,
                range_memo,
                constant_memo,
            ),
            field=f"series_outputs.{channel_id}.output_measure",
        )
    return contract


def _compiled_contract(
    definition: Mapping[str, Any],
    plan: TypedSeriesBundlePlan,
    compiled: CompiledNumbaSeriesPlan,
) -> dict[str, Any]:
    parameter_ids = {
        str(item["id"]) for item in definition.get("parameter_schema") or []
    }
    dependencies = [
        name for name in plan.context_requirements if name not in parameter_ids
    ]
    channel_types = {
        name: value_type.to_dict() for name, value_type in plan.output_types.items()
    }
    return {
        "dependencies": list(canonicalize_variables(dependencies)),
        "channel_types": channel_types,
        "channel_measures": _series_measure_contract(definition, plan),
        **_series_history_contract(definition, plan),
        "compiled_series_plan_id": compiled.plan_id,
        "kernel_version": compiled.kernel_version,
        "engine_version": compiled.engine_version,
    }



def _literal_number_from_node(
    node_id: int,
    nodes: Mapping[int, Any],
    memo: dict[int, float | None],
) -> float | None:
    if node_id in memo:
        return memo[node_id]
    node = nodes[node_id]
    value: float | None = None
    if node.kind == "constant":
        try:
            parsed = ast.parse(str(node.formula_fragment), mode="eval").body
            if isinstance(parsed, ast.Constant) and isinstance(parsed.value, (int, float)):
                number = float(parsed.value)
                value = number if math.isfinite(number) else None
        except (SyntaxError, TypeError, ValueError, OverflowError):
            value = None
    elif node.operator_id == "negate" and len(node.inputs) == 1:
        operand = _literal_number_from_node(int(node.inputs[0]), nodes, memo)
        value = -operand if operand is not None else None
    elif node.operator_id in {"add", "subtract", "multiply", "divide", "power"} and len(node.inputs) == 2:
        left = _literal_number_from_node(int(node.inputs[0]), nodes, memo)
        right = _literal_number_from_node(int(node.inputs[1]), nodes, memo)
        if left is not None and right is not None:
            try:
                if node.operator_id == "add":
                    candidate = left + right
                elif node.operator_id == "subtract":
                    candidate = left - right
                elif node.operator_id == "multiply":
                    candidate = left * right
                elif node.operator_id == "divide":
                    candidate = left / right if abs(right) >= 1e-12 else math.nan
                else:
                    candidate = left**right
                value = float(candidate) if math.isfinite(float(candidate)) else None
            except (ArithmeticError, OverflowError, TypeError, ValueError):
                value = None
    memo[node_id] = value
    return value


def _argument_node_id(node: Any, name: str) -> int | None:
    for parameter_name, input_node_id in node.arguments:
        if parameter_name == name:
            return int(input_node_id)
    return None


def _positive_integer_argument(
    node: Any,
    name: str,
    nodes: Mapping[int, Any],
    memo: dict[int, float | None],
    *,
    default: int | None = None,
    allow_zero: bool = False,
) -> int:
    input_node_id = _argument_node_id(node, name)
    if input_node_id is None:
        if default is None:
            raise ValidationError(
                "SERIES_CONFIGURATION_MUST_BE_CONSTANT",
                f"{node.operator_id} 缺少固定参数 {name}。",
                field="series_outputs",
            )
        return default
    value = _literal_number_from_node(input_node_id, nodes, memo)
    minimum = 0 if allow_zero else 1
    if value is None or not float(value).is_integer() or value < minimum:
        raise ValidationError(
            "SERIES_CONFIGURATION_MUST_BE_CONSTANT",
            f"{node.operator_id} 的参数 {name} 必须是定义级{'非负' if allow_zero else '正'}整数常量。",
            field="series_outputs",
            diagnostics=[{
                "code": "SERIES_CONFIGURATION_MUST_BE_CONSTANT",
                "node_id": int(node.node_id),
                "operator": node.operator_id,
                "parameter": name,
            }],
        )
    return int(value)


def _parameter_constants(plan: TypedSeriesBundlePlan, parameters: Mapping[str, float]) -> dict[int, float | None]:
    return {int(node.node_id): float(parameters[str(node.label)]) for node in plan.nodes
            if node.kind == "variable" and str(node.label) in parameters}


def _infer_series_history(
    plan: TypedSeriesBundlePlan, parameters: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    nodes = {int(node.node_id): node for node in plan.nodes}
    constants = _parameter_constants(plan, parameters or {})
    memo: dict[int, tuple[bool, int, int]] = {}
    full_history_operators = {
        "recursive_smooth",
        "cumulative_sum",
        "cumulative_product",
        "cumulative_return",
        "cumulative_max",
        "cumulative_min",
        "drawdown_series",
        "new_high_mask",
    }

    def visit(node_id: int) -> tuple[bool, int, int]:
        if node_id in memo:
            return memo[node_id]
        node = nodes[node_id]
        children = [visit(int(child)) for child in node.inputs]
        child_full = any(item[0] for item in children)
        child_lookback = max((item[1] for item in children), default=1)
        child_minimum = max((item[2] for item in children), default=1)
        operator_id = str(node.operator_id or "")
        if operator_id in full_history_operators:
            result = (True, 1, child_minimum)
        elif operator_id in {"rolling_mean", "rolling_std", "rolling_min", "rolling_max"}:
            window = _positive_integer_argument(node, "window", nodes, constants)
            default_minimum = window
            min_periods = _positive_integer_argument(
                node,
                "min_periods",
                nodes,
                constants,
                default=default_minimum,
            )
            value_node_id = _argument_node_id(node, "values")
            value_history = visit(value_node_id) if value_node_id is not None else (child_full, child_lookback, child_minimum)
            result = (
                value_history[0],
                value_history[1] + window - 1,
                value_history[2] + min_periods - 1,
            )
        elif operator_id in {"lag", "difference"}:
            periods = _positive_integer_argument(node, "periods", nodes, constants, default=1)
            result = (child_full, child_lookback + periods, child_minimum + periods)
        else:
            result = (child_full, child_lookback, child_minimum)
        memo[node_id] = result
        return result

    requirements = [visit(int(root_id)) for root_id in plan.roots.values()]
    full_history = any(item[0] for item in requirements)
    return {
        "history_policy": "full_history" if full_history else "lookback",
        "lookback_observations": 1 if full_history else max(item[1] for item in requirements),
        "minimum_observations": max(item[2] for item in requirements),
        "history_inference_source": "typed_dag",
    }


def _infer_series_value_ranges(
    plan: TypedSeriesBundlePlan,
    parameters: Mapping[str, float] | None = None,
) -> dict[str, dict[str, Any]]:
    """Conservatively prove simple output bounds from the typed DAG."""

    nodes = {int(node.node_id): node for node in plan.nodes}
    constants = _parameter_constants(plan, parameters or {})
    memo: dict[int, tuple[float | None, float | None, bool]] = {}

    def visit(node_id: int) -> tuple[float | None, float | None, bool]:
        if node_id in memo:
            return memo[node_id]
        node = nodes[node_id]
        literal = _literal_number_from_node(node_id, nodes, constants)
        if literal is not None:
            result = (literal, literal, True)
            memo[node_id] = result
            return result
        children = [visit(int(child)) for child in node.inputs]
        operator_id = str(node.operator_id or "")
        result: tuple[float | None, float | None, bool] = (None, None, False)
        if operator_id == "sign":
            result = (-1.0, 1.0, True)
        elif operator_id == "clip":
            lower_id = _argument_node_id(node, "lower")
            upper_id = _argument_node_id(node, "upper")
            lower = _literal_number_from_node(lower_id, nodes, constants) if lower_id is not None else None
            upper = _literal_number_from_node(upper_id, nodes, constants) if upper_id is not None else None
            if lower is not None and upper is not None and lower <= upper:
                result = (lower, upper, True)
        elif operator_id in {"rolling_mean", "rolling_min", "rolling_max", "recursive_smooth"}:
            values_id = _argument_node_id(node, "values")
            if values_id is not None:
                result = visit(values_id)
                if operator_id == "recursive_smooth":
                    initial_id = _argument_node_id(node, "initial")
                    initial = _literal_number_from_node(initial_id, nodes, constants) if initial_id is not None else None
                    if result[2] and initial is not None:
                        result = (min(result[0], initial), max(result[1], initial), True)  # type: ignore[arg-type]
                    else:
                        result = (None, None, False)
        elif operator_id == "rolling_std":
            result = (0.0, None, False)
        elif operator_id in {"add", "subtract", "multiply"} and len(children) == 2:
            left, right = children
            if left[2] and right[2] and None not in left[:2] and None not in right[:2]:
                l0, l1 = float(left[0]), float(left[1])
                r0, r1 = float(right[0]), float(right[1])
                if operator_id == "add":
                    result = (l0 + r0, l1 + r1, True)
                elif operator_id == "subtract":
                    result = (l0 - r1, l1 - r0, True)
                else:
                    products = (l0 * r0, l0 * r1, l1 * r0, l1 * r1)
                    result = (min(products), max(products), True)
        elif operator_id in {"minimum", "maximum"} and len(children) == 2:
            left, right = children
            if left[2] and right[2] and None not in left[:2] and None not in right[:2]:
                if operator_id == "minimum":
                    result = (min(float(left[0]), float(right[0])), min(float(left[1]), float(right[1])), True)
                else:
                    result = (max(float(left[0]), float(right[0])), max(float(left[1]), float(right[1])), True)
        elif operator_id == "where" and len(children) == 3:
            true_range, false_range = children[1], children[2]
            if true_range[2] and false_range[2] and None not in true_range[:2] and None not in false_range[:2]:
                result = (
                    min(float(true_range[0]), float(false_range[0])),
                    max(float(true_range[1]), float(false_range[1])),
                    True,
                )
        memo[node_id] = result
        return result

    output: dict[str, dict[str, Any]] = {}
    for channel_id, root_id in plan.roots.items():
        minimum, maximum, bounded = visit(int(root_id))
        output[channel_id] = {
            "minimum": minimum,
            "maximum": maximum,
            "bounded": bool(bounded and minimum is not None and maximum is not None),
            "source": "typed_dag" if bounded else "not_proven",
        }
    return output


def _measure_from_semantics_and_range(
    semantic_dimension: str,
    value_range: Mapping[str, Any],
) -> str:
    direct = _semantic_default_measure(semantic_dimension)
    if semantic_dimension != "dimensionless" or not value_range.get("bounded"):
        return direct
    minimum = value_range.get("minimum")
    maximum = value_range.get("maximum")
    if minimum is None or maximum is None:
        return direct
    lower = float(minimum)
    upper = float(maximum)
    tolerance = 1e-12
    if lower >= -tolerance and upper <= 1.0 + tolerance:
        return "bounded_0_1"
    if lower >= -1.0 - tolerance and upper <= 1.0 + tolerance:
        return "bounded_minus1_1"
    if lower >= -tolerance and upper <= 100.0 + tolerance:
        return "oscillator_0_100"
    return direct


def _semantic_default_measure(semantic_dimension: str) -> str:
    direct = {
        "raw_market_price": "raw_market_price",
        "adjusted_nav": "adjusted_nav",
        "reported_nav": "reported_nav",
        "return_decimal": "return_decimal",
        "rate_decimal": "rate_decimal",
        "volume": "volume",
        "currency_amount": "currency_amount",
        "count": "count",
        "calendar_days": "calendar_days",
        "dimensionless": "dimensionless",
    }
    if semantic_dimension in direct:
        return direct[semantic_dimension]
    if semantic_dimension.startswith(("derived:", "squared:", "inverse:")):
        return "derived"
    return "dimensionless"


def _measure_compatible(measure: str, semantic_dimension: str) -> bool:
    if measure == "auto":
        return True
    if measure == "derived":
        return semantic_dimension.startswith(("derived:", "squared:", "inverse:"))
    if measure in {"normalized", "bounded_0_1", "bounded_minus1_1", "oscillator_0_100", "virtual_nav"}:
        return semantic_dimension in {"dimensionless", "adjusted_nav" if measure == "virtual_nav" else "dimensionless"}
    return measure == semantic_dimension


def _ensure_series_compiled_contract(
    definition: Mapping[str, Any],
    plan: TypedSeriesBundlePlan,
    contract: Mapping[str, Any],
    parameters: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    effective = parameters if parameters is not None else {str(item["id"]): item["default"] for item in definition.get("parameter_schema") or []}
    output = dict(contract)
    output.update(_infer_series_history(plan, effective))
    catalog = {str(item["id"]): item for item in series_output_measure_catalog()}
    metadata = {str(item["id"]): item for item in definition.get("series_outputs") or []}
    inferred_ranges = _infer_series_value_ranges(plan, effective)
    measures: dict[str, dict[str, Any]] = {}
    for channel_id, value_type in plan.output_types.items():
        item = metadata[channel_id]
        requested = str(item.get("output_measure") or "auto")
        if requested not in catalog:
            raise ValidationError(
                "INVALID_SERIES_OUTPUT_MEASURE",
                f"不支持的时序输出口径: {requested}。",
                field=f"series_outputs.{channel_id}.output_measure",
            )
        value_range = inferred_ranges[channel_id]
        inferred = _measure_from_semantics_and_range(
            value_type.semantic_dimension,
            value_range,
        )
        resolved = inferred if requested == "auto" else requested
        if not _measure_compatible(resolved, value_type.semantic_dimension):
            raise ValidationError(
                "SERIES_OUTPUT_MEASURE_MISMATCH",
                f"输出通道 {channel_id} 的公式量纲 {value_type.semantic_dimension} 与口径 {resolved} 不兼容。",
                field=f"series_outputs.{channel_id}.output_measure",
            )
        option = catalog[resolved]
        measures[channel_id] = {
            "inferred": inferred,
            "resolved": resolved,
            "source": "compiler" if requested == "auto" else "user",
            "semantic_dimension": value_type.semantic_dimension,
            "price_basis": value_type.price_basis,
            "range": (
                copy.deepcopy(option.get("range"))
                if requested != "auto" and option.get("range")
                else copy.deepcopy(value_range)
            ),
            "default_unit": str(option.get("default_unit") or ""),
            "default_display_format": str(option.get("default_display_format") or "number"),
        }
    output["channel_measures"] = measures
    return output


def _series_latex_symbols(definition: Mapping[str, Any]) -> dict[str, str]:
    symbols = dict(variable_latex_symbols("single_product"))
    for item in definition.get("parameter_schema") or []:
        parameter_id = str(item.get("id") or "")
        escaped = parameter_id.replace("_", r"\_")
        symbols[parameter_id] = rf"\mathrm{{{escaped}}}"
    return symbols


def _reachable_variable_names(
    plan: TypedSeriesBundlePlan,
    root_id: int,
) -> list[str]:
    node_by_id = {node.node_id: node for node in plan.nodes}
    pending = [root_id]
    visited: set[int] = set()
    variables: set[str] = set()
    while pending:
        node_id = int(pending.pop())
        if node_id in visited:
            continue
        visited.add(node_id)
        node = node_by_id[node_id]
        if node.kind == "variable":
            variables.add(str(node.label))
        pending.extend(int(input_id) for input_id in node.inputs)
    return sorted(variables)


def _annotated_series_graph(
    definition: Mapping[str, Any],
    plan: TypedSeriesBundlePlan,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    symbols = _series_latex_symbols(definition)
    graph = copy.deepcopy(plan.graph_payload())
    graph["edges"] = [
        {
            "source": int(input_node_id),
            "target": int(node.node_id),
            "parameter": str(parameter),
            "order": int(order),
        }
        for node in plan.nodes
        for order, (parameter, input_node_id) in enumerate(node.arguments)
    ]
    for node in graph.get("nodes", []):
        value_type = node.get("inferred_type") or {}
        node["value_type"] = value_type.get("display")
        node["shape"] = value_type.get("kind")
        node["axes"] = value_type.get("axes", [])
        node["symbolic_shape"] = value_type.get("shape", [])
        fragment = str(node.get("formula_fragment") or node.get("label") or "")
        try:
            node["latex_fragment"] = render_python_expression_latex(
                fragment,
                symbols,
            )
        except (SyntaxError, ValueError):
            node["latex_fragment"] = None

    output_metadata = {
        str(item["id"]): item for item in definition.get("series_outputs") or []
    }
    python_expressions = dict(plan.python_expressions)
    output_inferences: dict[str, dict[str, Any]] = {}
    for channel_id, root_id in plan.roots.items():
        output_type = plan.output_types[channel_id]
        expression = str(output_metadata[channel_id].get("expression") or "")
        python_expression = str(python_expressions[channel_id])
        channel_metadata = output_metadata[channel_id]
        output_inferences[channel_id] = {
            "id": channel_id,
            "label": str(channel_metadata.get("label") or channel_id),
            "expression": expression,
            "latex": expression,
            "editable_latex": editable_formula_latex(python_expression),
            "display_latex": render_python_expression_latex(
                python_expression,
                symbols,
            ),
            "math_notation_version": MATH_NOTATION_VERSION,
            "python_expression": python_expression,
            "inferred_type": str(output_type),
            "shape": output_type.kind,
            "semantic_dimension": output_type.semantic_dimension,
            "price_basis": output_type.price_basis,
            "output_measure": channel_metadata.get("output_measure") or "auto",
            "inferred_output_measure": channel_metadata.get(
                "inferred_output_measure"
            ),
            "resolved_output_measure": channel_metadata.get(
                "resolved_output_measure"
            ),
            "output_measure_source": channel_metadata.get(
                "output_measure_source"
            ),
            "value_range": copy.deepcopy(channel_metadata.get("value_range")),
            "dependencies": _reachable_variable_names(plan, int(root_id)),
            "root_id": int(root_id),
        }
    return graph, output_inferences


def apply_series_compiled_contract(
    definition: dict[str, Any],
    contract: Mapping[str, Any],
) -> None:
    dependencies = list(canonicalize_variables(contract.get("dependencies") or []))
    definition["required_variables"] = dependencies
    applicable = {"etf", "fund"}
    for variable_id in dependencies:
        variable = get_variable(variable_id)
        if variable is not None:
            applicable &= set(variable.product_kinds)
    definition["applicable_product_kinds"] = sorted(applicable)
    definition["channel_types"] = copy.deepcopy(contract.get("channel_types") or {})
    definition["history_policy"] = str(contract.get("history_policy") or "lookback")
    definition["history_inference_source"] = str(
        contract.get("history_inference_source") or "typed_dag"
    )
    definition["lookback_parameter"] = None
    definition["lookback_observations"] = int(
        contract.get("lookback_observations") or 1
    )
    definition["minimum_observations"] = int(
        contract.get("minimum_observations") or 1
    )
    measure_contract = {
        str(channel_id): copy.deepcopy(item)
        for channel_id, item in (contract.get("channel_measures") or {}).items()
    }
    for output in definition.get("series_outputs") or []:
        channel_id = str(output.get("id") or "")
        measure = measure_contract.get(channel_id)
        if measure is None:
            continue
        output["inferred_output_measure"] = measure["inferred"]
        output["resolved_output_measure"] = measure["resolved"]
        output["output_measure_source"] = measure["source"]
        output["semantic_dimension"] = measure["semantic_dimension"]
        output["price_basis"] = measure["price_basis"]
        output["value_range"] = copy.deepcopy(measure["range"])
        if not output.get("unit") and measure.get("default_unit"):
            output["unit"] = measure["default_unit"]
        if output.get("display_format") not in {"number", "percent"}:
            output["display_format"] = measure["default_display_format"]
    outputs = list(definition.get("series_outputs") or [])
    if outputs:
        primary = outputs[0]
        definition["unit"] = str(primary.get("unit") or "")
        definition["display_format"] = str(
            primary.get("display_format") or "number"
        )
        definition["precision"] = int(primary.get("precision", 4))
    definition["compiled_series_plan_id"] = contract.get(
        "compiled_series_plan_id"
    )
    definition["numeric_kernel_version"] = contract.get("kernel_version")
    definition["availability_status"] = "ready"


def warm_time_series_definition(
    definition: dict[str, Any], runtime_root: Path
) -> dict[str, Any]:
    if definition.get("result_kind") != TIME_SERIES_RESULT_KIND:
        raise TypeError("definition is not a time-series indicator")
    plan, compiled = _compile_definition(definition)
    persist_numba_series_plan(compiled, runtime_root)
    contract = _ensure_series_compiled_contract(
        definition,
        plan,
        _compiled_contract(definition, plan, compiled),
    )
    apply_series_compiled_contract(definition, contract)
    return {
        **contract,
        "plan": plan,
        "execution": compiled.metadata(),
    }


def validate_time_series_definition(
    fields: Mapping[str, Any],
    runtime_root: Path,
    *,
    protocol_defaults: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    definition = normalize_time_series_definition(fields, protocol_defaults)
    token_definition = copy.deepcopy(definition)
    try:
        warm = warm_time_series_definition(definition, runtime_root)
    except ValidationError as exc:
        return definition, {
            "valid": False,
            "diagnostics": exc.diagnostics
            or [
                {
                    "code": exc.code,
                    "message": exc.message,
                    "field": exc.field or "series_outputs",
                }
            ],
            "dependencies": [],
            "dag": None,
        }
    plan: TypedSeriesBundlePlan = warm["plan"]
    compiled: CompiledNumbaSeriesPlan = _get_warmed_definition(definition)[1]
    graph, output_inferences = _annotated_series_graph(definition, plan)
    compile_token = _compile_token(token_definition, compiled.plan_id)
    return definition, {
        "valid": True,
        "diagnostics": [],
        "dependencies": list(definition.get("required_variables") or []),
        "dag": graph,
        "output_type": "series_bundle",
        "result_kind": TIME_SERIES_RESULT_KIND,
        "output_contract": "series_bundle",
        "output_channels": copy.deepcopy(definition["series_outputs"]),
        "output_inferences": output_inferences,
        "channel_types": copy.deepcopy(definition.get("channel_types") or {}),
        "parameter_schema": copy.deepcopy(definition["parameter_schema"]),
        "fixed_parameters": copy.deepcopy(definition.get("fixed_parameters") or []),
        "history_policy": definition.get("history_policy"),
        "history_inference_source": definition.get("history_inference_source"),
        "lookback_observations": int(
            definition.get("lookback_observations") or 1
        ),
        "minimum_observations": int(
            definition.get("minimum_observations") or 1
        ),
        "math_notation_version": MATH_NOTATION_VERSION,
        "estimated_cost": dict(plan.estimated_cost),
        "dsl_version": plan.dsl_version,
        "operator_registry_version": plan.operator_registry_version,
        "compiled_series_plan_id": compiled.plan_id,
        "compile_status": "compiled",
        "compile_ms": compiled.compile_ms,
        "kernel_version": compiled.kernel_version,
        "engine_version": compiled.engine_version,
        "required_workspace_bytes": compiled.required_workspace_bytes,
        "compile_token": compile_token,
        "compile_token_scope": "current_process_warm_cache",
        "execution": compiled.metadata(),
        "python_fallback": 0,
        "python_operator_calls": 0,
    }


def _warning(code: str, message: str) -> dict[str, str]:
    return {"code": code, "message": message}


def _empty_window(
    *,
    as_of: str | None,
    data_latest_date: str | None = None,
) -> dict[str, Any]:
    return {
        "requested_as_of": as_of,
        "effective_as_of": None,
        "start_date": None,
        "end_date": None,
        "observation_count": 0,
        "data_latest_date": data_latest_date,
    }


def _result_base(
    definition: Mapping[str, Any],
    target: Mapping[str, str],
    period: str,
    parameters: Mapping[str, float],
) -> dict[str, Any]:
    return {
        "indicator_id": definition.get("id"),
        "indicator_revision": definition.get("revision"),
        "indicator_name": definition.get("name"),
        "result_kind": TIME_SERIES_RESULT_KIND,
        "target": dict(target),
        "period": period,
        "parameters": dict(parameters),
        "parameter_hash": parameter_hash(parameters),
        "axis_anchor": definition.get("axis_anchor"),
        "history_policy": definition.get("history_policy"),
        "lookback_observations": int(
            definition.get("lookback_observations") or 1
        ),
        "minimum_observations": int(
            definition.get("minimum_observations") or 1
        ),
        "presentation": metric_presentation(dict(definition)),
    }


def _unavailable_result(
    definition: Mapping[str, Any],
    target: Mapping[str, str],
    period: str,
    parameters: Mapping[str, float],
    *,
    code: str,
    message: str,
    as_of: str | None,
    data_latest_date: str | None = None,
) -> dict[str, Any]:
    return {
        **_result_base(definition, target, period, parameters),
        "status": "unavailable",
        "warnings": [_warning(code, message)],
        "window": _empty_window(
            as_of=as_of, data_latest_date=data_latest_date
        ),
        "dates": [],
        "channels": [],
    }


def _series_cache_key(
    definition: Mapping[str, Any],
    target: Mapping[str, str],
    parameters: Mapping[str, float],
    period: str,
    as_of: str | None,
    source: ProductChartSeries,
    data_generation: str,
    max_points: int,
) -> str:
    payload = {
        "max_points": max_points,
        "definition": _definition_contract_key(definition),
        "id": definition.get("id"),
        "revision": definition.get("revision"),
        "target": target,
        "parameters": parameters,
        "period": period,
        "as_of": as_of,
        "source_fingerprint": source.fingerprint,
        "data_generation": data_generation,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _resolve_instance(
    repository: IndicatorRepository,
    instance: Mapping[str, Any],
    runtime_root: Path,
) -> tuple[
    dict[str, Any],
    TypedSeriesBundlePlan,
    CompiledNumbaSeriesPlan,
    dict[str, float],
]:
    inline = instance.get("inline_definition")
    indicator_id = str(instance.get("indicator_id") or "").strip()
    if bool(inline) == bool(indicator_id):
        raise ValidationError(
            "INVALID_SERIES_INSTANCE",
            "时序指标实例必须且只能指定 indicator_id 或 inline_definition。",
            field="indicator_instances",
        )
    if inline:
        definition = normalize_time_series_definition(inline)
        plan, compiled = _get_warmed_definition(definition)
        expected = _compile_token(definition, compiled.plan_id)
        supplied = str(instance.get("compile_token") or "")
        if not supplied or not hmac.compare_digest(supplied, expected):
            raise ValidationError(
                "INLINE_SERIES_COMPILE_TOKEN_MISMATCH",
                "未保存时序公式与已预热计划不匹配，请重新校验。",
                field="compile_token",
            )
        definition = {
            **definition,
            "id": None,
            "revision": None,
            "source": "inline",
            "read_only": False,
        }
    else:
        revision = instance.get("indicator_revision")
        stored_definition = repository.get(
            indicator_id, int(revision) if revision is not None else None
        )
        if stored_definition.get("result_kind", "scalar") != TIME_SERIES_RESULT_KIND:
            raise ValidationError(
                "INDICATOR_RESULT_KIND_MISMATCH",
                "标量指标不能通过时序指标接口运行。",
                field="indicator_id",
            )
        identity = {
            key: copy.deepcopy(stored_definition.get(key))
            for key in (
                "id", "revision", "source", "read_only", "created_at", "updated_at"
            )
            if key in stored_definition
        }
        definition = {
            **stored_definition,
            **normalize_time_series_definition(stored_definition, stored_definition),
            **identity,
        }
        plan, compiled = _get_warmed_definition(definition)
    parameters = normalize_series_parameters(definition, instance.get("parameters"))
    # Instance-specific history/presentation never mutates the stored revision
    # or the shared prewarmed numerical plan.
    contract = _ensure_series_compiled_contract(
        definition, plan, _compiled_contract(definition, plan, compiled), parameters,
    )
    apply_series_compiled_contract(definition, contract)
    return definition, plan, compiled, parameters


def _combined_execution(
    compiled_plans: list[CompiledNumbaSeriesPlan],
) -> dict[str, Any]:
    signatures: dict[str, list[str]] = {}
    for compiled in compiled_plans:
        signatures[compiled.plan_id[:16]] = list(compiled.compiled_signatures)
    audit = validate_execution_audit(
        {
            "execution_backend": NJIT_BACKEND,
            "nopython": all(
                compiled.metadata().get("nopython") is True
                for compiled in compiled_plans
            ),
            "kernel_signatures": signatures,
            "python_fallback": 0,
            "python_operator_calls": 0,
        }
    )
    return {
        **audit,
        "compiled_plan_ids": list(
            dict.fromkeys(compiled.plan_id for compiled in compiled_plans)
        ),
        "request_time_compilation": 0,
    }


class TimeSeriesIndicatorService:
    def __init__(
        self,
        *,
        repository: IndicatorRepository,
        runtime_root: Path,
        market_data_dir: Path = DEFAULT_DATA_DIR,
        cache: ResultCache | None = None,
    ) -> None:
        self.repository = repository
        self.runtime_root = runtime_root
        self.market_data_dir = market_data_dir
        self.cache = cache

    def warm(self, definition: dict[str, Any]) -> dict[str, Any]:
        identity = {
            key: copy.deepcopy(definition.get(key))
            for key in (
                "id", "revision", "source", "read_only", "created_at", "updated_at"
            )
            if key in definition
        }
        normalized = {
            **definition,
            **normalize_time_series_definition(definition, definition),
            **identity,
        }
        return warm_time_series_definition(normalized, self.runtime_root)

    def validate(
        self,
        fields: Mapping[str, Any],
        *,
        protocol_defaults: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return validate_time_series_definition(
            fields, self.runtime_root, protocol_defaults=protocol_defaults
        )

    def evaluate(
        self,
        *,
        indicator_instances: list[dict[str, Any]],
        target: dict[str, Any],
        period: str,
        as_of: str | None = None,
        max_points: int = MAX_SERIES_DISPLAY_POINTS,
        _snapshot_only: bool = False,
    ) -> dict[str, Any]:
        # Internal snapshot jobs reduce the full requested window before chart
        # truncation. This option is deliberately absent from public API models.
        normalized_period = str(period or "").strip().upper()
        if not indicator_instances or len(indicator_instances) > MAX_SERIES_INSTANCES:
            raise ValidationError(
                "SERIES_INSTANCE_LIMIT_EXCEEDED",
                f"每次需要 1 至 {MAX_SERIES_INSTANCES} 个时序指标实例。",
                field="indicator_instances",
            )
        kind = str(target.get("kind") or "")
        product_id = str(target.get("product_id") or "").strip()
        if kind not in {"etf", "fund"} or not product_id:
            raise ValidationError(
                "INVALID_TARGET",
                "时序指标目标产品无效。",
                field="target",
            )
        if not 1 <= int(max_points) <= MAX_SERIES_DISPLAY_POINTS:
            raise ValidationError(
                "INVALID_SERIES_LIMIT",
                f"时序展示点数必须位于 1 至 {MAX_SERIES_DISPLAY_POINTS}。",
                field="max_points",
            )

        resolved = [
            _resolve_instance(
                self.repository, instance, self.runtime_root
            )
            for instance in indicator_instances
        ]
        target_name = product_id
        for definition, _plan, _compiled, _parameters in resolved:
            if definition.get("id"):
                target_name = product_id
                break
        public_target = {
            "kind": kind,
            "product_id": product_id,
            "name": target_name,
        }

        grouped_dependencies: dict[str, set[str]] = {}
        for definition, plan, _compiled, parameters in resolved:
            dependencies = set(_array_context_names(plan, parameters))
            grouped_dependencies.setdefault(
                str(definition["axis_anchor"]), set()
            ).update(dependencies)
        sources: dict[str, ProductChartSeries] = {
            axis_anchor: load_product_chart_series(
                kind,  # type: ignore[arg-type]
                product_id,
                sorted(dependencies),
                axis_anchor,
                self.market_data_dir,
                as_of,
            )
            for axis_anchor, dependencies in grouped_dependencies.items()
        }
        if sources:
            identity = next(iter(sources.values())).identity
            public_target["name"] = identity.name

        results: list[dict[str, Any]] = []
        cache_hits = 0
        cache_misses = 0
        compiled_plans: list[CompiledNumbaSeriesPlan] = []
        for definition, plan, compiled, parameters in resolved:
            compiled_plans.append(compiled)
            if kind not in set(definition.get("applicable_product_kinds") or []):
                results.append(
                    _unavailable_result(
                        definition,
                        public_target,
                        normalized_period,
                        parameters,
                        code="INDICATOR_NOT_APPLICABLE",
                        message="该时序指标所需的真实行情字段不适用于当前产品。",
                        as_of=as_of,
                    )
                )
                continue
            axis_anchor = str(definition["axis_anchor"])
            source = sources[axis_anchor]
            dependencies = set(_array_context_names(plan, parameters))
            missing = [
                name
                for name in dependencies
                if name in source.unavailable_variables
                or name not in source.frame.columns
            ]
            if source.frame.empty or missing:
                labels = "、".join(
                    str(get_variable(name).label if get_variable(name) else name)
                    for name in missing
                )
                results.append(
                    _unavailable_result(
                        definition,
                        public_target,
                        normalized_period,
                        parameters,
                        code="VARIABLE_UNAVAILABLE",
                        message=(
                            f"缺少时序指标输入：{labels}。"
                            if labels
                            else "时序指标日期轴没有可用真实数据。"
                        ),
                        as_of=as_of,
                        data_latest_date=source.data_latest_date,
                    )
                )
                continue
            key = _series_cache_key(
                definition,
                public_target,
                parameters,
                normalized_period,
                as_of,
                source,
                market_data_generation(self.market_data_dir),
                int(max_points),
            )
            if _snapshot_only:
                key = f"{key}:snapshot-last-finite-v1"
            cached = self.cache.get(key) if self.cache is not None else None
            if cached is not None:
                results.append(cached)
                cache_hits += 1
                continue
            lookback = int(definition.get("lookback_observations") or 1)
            source_lookback = lookback + _derived_boundary_observations(
                dependencies
            )
            try:
                window = select_chart_window(
                    source,
                    normalized_period,
                    as_of,
                    history_policy=str(definition["history_policy"]),  # type: ignore[arg-type]
                    lookback_observations=source_lookback,
                    max_display_points=max(1, len(source.frame)) if _snapshot_only else int(max_points),
                )
                context = _build_series_runtime_context(
                    definition,
                    plan,
                    compiled,
                    window.compute_frame,
                    parameters,
                )
                outputs = compiled.compute(
                    tuple(context[name] for name in compiled.context_names)
                )
                channel_by_id = {
                    str(item["id"]): item
                    for item in definition.get("series_outputs") or []
                }
                channels: list[dict[str, Any]] = []
                snapshot_channels: list[dict[str, Any]] = []
                has_finite = False
                for channel_name, values in zip(compiled.channel_names, outputs):
                    array = np.asarray(values, dtype=np.float64)
                    if array.ndim != 1 or array.size != len(window.compute_frame):
                        raise ValidationError(
                            "SERIES_OUTPUT_SHAPE_MISMATCH",
                            "时序 NJIT 输出长度与日期轴不一致。",
                            field=f"series_outputs.{channel_name}",
                        )
                    if _snapshot_only:
                        from .snapshot_execution import last_finite_snapshot_value
                        value, position = last_finite_snapshot_value(
                            np.ascontiguousarray(array), window.display_start, window.display_end,
                        )
                        finite = position >= 0
                        has_finite = has_finite or finite
                        snapshot_channels.append({
                            "id": channel_name, "value": float(value) if finite else None,
                            "value_date": window.compute_frame.iloc[position]["date"].strftime("%Y-%m-%d") if finite else None,
                        })
                        continue
                    visible = array[window.display_start : window.display_end]
                    serialized: list[float | None] = []
                    for value in visible:
                        if math.isfinite(float(value)):
                            serialized.append(float(value))
                            has_finite = True
                        else:
                            serialized.append(None)
                    metadata = channel_by_id[channel_name]
                    channels.append(
                        {
                            "id": channel_name,
                            "label": metadata.get("label") or channel_name,
                            "unit": metadata.get("unit") or "",
                            "display_format": metadata.get("display_format") or "number",
                            "precision": int(metadata.get("precision", 4)),
                            "output_measure": metadata.get(
                                "resolved_output_measure"
                            )
                            or metadata.get("output_measure")
                            or plan.output_types[channel_name].semantic_dimension,
                            "semantic_dimension": metadata.get(
                                "semantic_dimension"
                            )
                            or plan.output_types[channel_name].semantic_dimension,
                            "price_basis": metadata.get("price_basis")
                            or plan.output_types[channel_name].price_basis,
                            "value_range": copy.deepcopy(
                                metadata.get("value_range")
                            ),
                            "null_count": sum(value is None for value in serialized),
                            "values": serialized,
                        }
                    )
                dates = [] if _snapshot_only else [
                    value.strftime("%Y-%m-%d")
                    for value in window.display_frame["date"]
                ]
                warnings = list(window.warnings)
                status = "ok" if has_finite and not warnings else (
                    "warning" if has_finite else "unavailable"
                )
                record = {
                    **_result_base(
                        definition,
                        public_target,
                        normalized_period,
                        parameters,
                    ),
                    "status": status,
                    "warnings": warnings,
                    "window": {
                        "requested_as_of": window.requested_as_of,
                        "effective_as_of": window.effective_as_of,
                        "start_date": window.start_date,
                        "end_date": window.end_date,
                        "observation_count": window.observation_count,
                        "data_latest_date": window.data_latest_date,
                    },
                    "dates": dates,
                    "channels": channels,
                    **({"snapshot_channels": snapshot_channels} if _snapshot_only else {}),
                    "data_lineage": copy.deepcopy(source.lineage),
                    "source_fingerprints": dict(source.fingerprints),
                    "execution": compiled.metadata(),
                }
            except ValidationError as exc:
                record = _unavailable_result(
                    definition,
                    public_target,
                    normalized_period,
                    parameters,
                    code=exc.code,
                    message=exc.message,
                    as_of=as_of,
                    data_latest_date=source.data_latest_date,
                )
            except (FloatingPointError, OverflowError, TypeError, ValueError) as exc:
                record = _unavailable_result(
                    definition,
                    public_target,
                    normalized_period,
                    parameters,
                    code="SERIES_OPERATOR_EXECUTION_FAILED",
                    message="固定签名 NJIT 时序计划执行失败，未回退到 Python。",
                    as_of=as_of,
                    data_latest_date=source.data_latest_date,
                )
                record["diagnostic"] = str(exc)[:200]
            if self.cache is not None:
                self.cache.put(key, record)
            cache_misses += 1
            results.append(record)

        summary = {
            "total": len(results),
            "ok": sum(item["status"] == "ok" for item in results),
            "warning": sum(item["status"] == "warning" for item in results),
            "unavailable": sum(
                item["status"] == "unavailable" for item in results
            ),
            "error": sum(item["status"] == "error" for item in results),
        }
        return {
            "results": results,
            "summary": summary,
            "cache": {"hits": cache_hits, "misses": cache_misses},
            "execution": _combined_execution(compiled_plans),
        }

    def export_excel(
        self,
        *,
        indicator_instance: Mapping[str, Any],
        targets: list[dict[str, Any]],
        period: str,
        as_of: str | None = None,
        output_dir: Path,
    ) -> ExcelExportArtifact:
        """Export raw inputs and shared-DAG Excel formulas for one series indicator."""

        if not targets or len(targets) > 10:
            raise ValidationError(
                "TARGET_LIMIT_EXCEEDED",
                "时序指标 Excel 导出需要 1 至 10 个产品。",
                field="targets",
            )
        definition, plan, compiled, parameters = _resolve_instance(
            self.repository,
            indicator_instance,
            self.runtime_root,
        )
        normalized_period = str(period or "").strip().upper()
        evidence_items: list[SeriesExcelTargetEvidence] = []
        data_generation = market_data_generation(self.market_data_dir)
        dependencies = set(_array_context_names(plan, parameters))
        axis_anchor = str(definition["axis_anchor"])

        for raw_target in targets:
            kind = str(raw_target.get("kind") or "")
            product_id = str(raw_target.get("product_id") or "").strip()
            if kind not in {"etf", "fund"} or not product_id:
                raise ValidationError(
                    "INVALID_TARGET",
                    "时序指标 Excel 导出的产品类型或产品编号无效。",
                    field="targets",
                )
            target_response = self.evaluate(
                indicator_instances=[dict(indicator_instance)],
                target={"kind": kind, "product_id": product_id},
                period=normalized_period,
                as_of=as_of,
                max_points=MAX_SERIES_DISPLAY_POINTS,
            )
            result = dict(target_response["results"][0])
            source = load_product_chart_series(
                kind,  # type: ignore[arg-type]
                product_id,
                sorted(dependencies),
                axis_anchor,
                self.market_data_dir,
                as_of,
            )
            public_target = dict(result.get("target") or raw_target)
            name = str(public_target.get("name") or source.identity.name or product_id)
            missing = [
                item
                for item in dependencies
                if item in source.unavailable_variables
                or item not in source.frame.columns
            ]
            if (
                kind not in set(definition.get("applicable_product_kinds") or [])
                or source.frame.empty
                or missing
            ):
                evidence_items.append(
                    SeriesExcelTargetEvidence(
                        target={"kind": kind, "product_id": product_id},
                        name=name,
                        result=result,
                        context={},
                        dates_by_variable={},
                        backend_outputs={},
                        display_start=0,
                        display_end=0,
                        display_dates=(),
                    )
                )
                continue

            lookback = int(definition.get("lookback_observations") or 1)
            source_lookback = lookback + _derived_boundary_observations(
                dependencies
            )
            try:
                window = select_chart_window(
                    source,
                    normalized_period,
                    as_of,
                    history_policy=str(definition["history_policy"]),  # type: ignore[arg-type]
                    lookback_observations=source_lookback,
                    max_display_points=MAX_SERIES_DISPLAY_POINTS,
                )
                context = _build_series_runtime_context(
                    definition,
                    plan,
                    compiled,
                    window.compute_frame,
                    parameters,
                )
                dates_by_variable: dict[str, list[Any]] = {}
                compute_dates = list(window.compute_frame["date"])
                for name_key in compiled.context_names:
                    if plan.context_requirements[name_key].rank > 0:
                        dates_by_variable[name_key] = compute_dates
                outputs = compiled.compute(
                    tuple(context[name_key] for name_key in compiled.context_names)
                )
                backend_outputs: dict[str, np.ndarray] = {}
                for channel_name, values in zip(
                    compiled.channel_names,
                    outputs,
                    strict=True,
                ):
                    array = np.asarray(values, dtype=np.float64)
                    if array.ndim != 1 or array.size != len(window.compute_frame):
                        raise ValidationError(
                            "SERIES_OUTPUT_SHAPE_MISMATCH",
                            "时序 NJIT 输出长度与 Excel 日期轴不一致。",
                            field=f"series_outputs.{channel_name}",
                        )
                    backend_outputs[channel_name] = array
            except ValidationError:
                evidence_items.append(
                    SeriesExcelTargetEvidence(
                        target={"kind": kind, "product_id": product_id},
                        name=name,
                        result=result,
                        context={},
                        dates_by_variable={},
                        backend_outputs={},
                        display_start=0,
                        display_end=0,
                        display_dates=(),
                    )
                )
                continue
            except (FloatingPointError, OverflowError, TypeError, ValueError) as exc:
                raise ValidationError(
                    "SERIES_EXCEL_EXECUTION_FAILED",
                    "生成时序 Excel 前的固定签名 NJIT 计算失败，未回退到 Python。",
                    field="series_outputs",
                    diagnostics=[{"message": str(exc)[:200]}],
                ) from exc

            evidence_items.append(
                SeriesExcelTargetEvidence(
                    target={"kind": kind, "product_id": product_id},
                    name=name,
                    result=result,
                    context=context,
                    dates_by_variable=dates_by_variable,
                    backend_outputs=backend_outputs,
                    display_start=int(window.display_start),
                    display_end=int(window.display_end),
                    display_dates=list(window.display_frame["date"]),
                )
            )

        if market_data_generation(self.market_data_dir) != data_generation:
            raise ValidationError(
                "EXCEL_EXPORT_DATA_CHANGED",
                "生成时序 Excel 期间市场数据版本发生变化，请重新下载。",
            )
        return build_series_indicator_excel_workbook(
            output_dir=output_dir,
            definition=definition,
            plan=plan,
            targets=evidence_items,
            period=normalized_period,
            as_of=as_of,
            data_generation=data_generation,
            parameters=parameters,
        )


__all__ = [
    "MAX_SERIES_DISPLAY_POINTS",
    "MAX_SERIES_INSTANCES",
    "TimeSeriesIndicatorService",
    "apply_series_compiled_contract",
    "validate_time_series_definition",
    "warm_time_series_definition",
]

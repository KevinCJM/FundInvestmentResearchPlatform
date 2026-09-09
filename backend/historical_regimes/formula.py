"""Causal, typed and NJIT-only formulas for historical regime features.

The historical-regime center deliberately reuses the indicator center's typed
AST -> DAG -> fixed-signature Numba pipeline. There is no Python/pandas
operator fallback in this module. A small causal subset of the global operator
registry is exposed because full-sample reductions would leak future values
when broadcast into a point-in-time feature series.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from cal_indicators.typed_dsl import (
    TYPED_COMPILER_VERSION,
    TYPED_DSL_VERSION,
    TYPED_OPERATOR_REGISTRY_VERSION,
    TypedDslError,
    TypedExpressionPlan,
    TypedIndicatorRuntime,
    ValueType,
    compose_typed_expression,
)
from cal_indicators.typed_numba_kernels import (
    ENGINE_VERSION,
    NUMERIC_KERNEL_VERSION,
    kernel_catalog_entry,
)
from cal_indicators.typed_numba_plan import numba_plan_id
from compute_policy import validate_execution_audit
from custom_indicators.errors import ValidationError
from computation_graph.series_runtime import compute_warmed_series

from .numba_kernels import (
    execution_audit as historical_regime_execution_audit,
    formula_align_block_kernel,
    formula_input_blocks_kernel,
    formula_missing_numeric_mask_kernel,
    require_historical_regime_kernels_ready,
)


FORMULA_LANGUAGE_ID = "historical-regime-typed-causal-series"
FORMULA_ALLOWLIST_VERSION = "typed-njit-causal-2"
FORMULA_EVALUATOR_VERSION = TYPED_COMPILER_VERSION
MAX_EXPRESSION_LENGTH = 1000
MAX_AST_NODES = 128
MAX_AST_DEPTH = 20
MAX_WINDOW = 5000
MAX_TIME_OBSERVATIONS = 20000
FORMULA_NUMERIC_KERNEL_IDS = (
    "formula_input_blocks",
    "formula_align_block",
    "formula_missing_numeric_mask",
)

# Every entry below owns a fixed-signature kernel in typed_numba_kernels.
# Full-sample reductions and terminal selectors are intentionally absent.
CAUSAL_OPERATOR_IDS = frozenset(
    {
        "add",
        "subtract",
        "multiply",
        "divide",
        "divide_or_default",
        "power",
        "minimum",
        "maximum",
        "negate",
        "absolute",
        "sqrt",
        "clip",
        "log",
        "exp",
        "reciprocal",
        "sign",
        "equal",
        "not_equal",
        "less_than",
        "less_equal",
        "greater_than",
        "greater_equal",
        "logical_and",
        "logical_or",
        "logical_not",
        "where",
        "cumulative_sum",
        "cumulative_product",
        "cumulative_max",
        "cumulative_min",
        "drawdown_series",
        "new_high_mask",
        "lag",
        "difference",
        "rolling_mean",
        "rolling_std",
        "rolling_min",
        "rolling_max",
        "recursive_smooth",
    }
)

SYSTEM_COLUMNS = frozenset(
    {
        "observation_date",
        "date",
        "trade_date",
        "available_at",
        "revision",
        "vintage",
        "is_final",
        "final_num",
        "final_den",
        "available_at_num",
        "available_at_den",
    }
)

FUNCTION_SYNTAX = (
    ("log", "log(x)"),
    ("abs", "abs(x)"),
    ("sqrt", "sqrt(x)"),
    ("clip", "clip(x, lower, upper)"),
    ("lag", "lag(x, periods)"),
    ("difference", "difference(x, periods)"),
    ("rolling_mean", "rolling_mean(x, window, min_periods)"),
    ("rolling_std", "rolling_std(x, window, ddof, min_periods)"),
    ("rolling_min", "rolling_min(x, window, min_periods)"),
    ("rolling_max", "rolling_max(x, window, min_periods)"),
    ("recursive_smooth", "recursive_smooth(x, window, initial)"),
    ("cumulative_sum", "cumulative_sum(x)"),
    ("cumulative_product", "cumulative_product(x)"),
    ("cumulative_max", "cumulative_max(x)"),
    ("cumulative_min", "cumulative_min(x)"),
    ("drawdown_series", "drawdown_series(x)"),
    ("where", "where(condition, if_true, if_false)"),
)


@dataclass(frozen=True)
class FormulaResult:
    values: pd.Series
    audit: dict[str, Any]


def formula_language_meta() -> dict[str, Any]:
    functions = []
    aliases = {"abs": "absolute"}
    for public_id, signature in FUNCTION_SYNTAX:
        canonical_id = aliases.get(public_id, public_id)
        kernel = kernel_catalog_entry(canonical_id)
        functions.append(
            {
                "id": public_id,
                "signature": signature,
                "causal": True,
                "njit_supported": bool(kernel["njit_supported"]),
                "execution_backend": "numba_njit_fixed_signature",
                "kernel_version": kernel["kernel_version"],
            }
        )
    return {
        "id": FORMULA_LANGUAGE_ID,
        "allowlist_version": FORMULA_ALLOWLIST_VERSION,
        "evaluator_version": FORMULA_EVALUATOR_VERSION,
        "dsl_version": TYPED_DSL_VERSION,
        "operator_registry_version": TYPED_OPERATOR_REGISTRY_VERSION,
        "numeric_kernel_version": NUMERIC_KERNEL_VERSION,
        "engine_version": ENGINE_VERSION,
        "execution_backend": "numba_njit_fixed_signature",
        "njit_required": True,
        "python_fallback": 0,
        "python_operator_calls": 0,
        "request_time_compilation": 0,
        "compile_protocol": "explicit_prepare_then_execute",
        "prepostprocess_backend": "numba_njit_fixed_signature",
        "prepostprocess_kernel_ids": list(FORMULA_NUMERIC_KERNEL_IDS),
        "operators": [
            "+",
            "-",
            "*",
            "/",
            "**",
            "==",
            "!=",
            "<",
            "<=",
            ">",
            ">=",
        ],
        "functions": functions,
        "variable_rule": "变量必须是数据源中的数值列；所选主字段始终可用别名 value。",
        "provenance": {
            "field": "features.formula_provenance",
            "rule": "value 自动继承目标序列时点；其他列需声明 point_in_time=true 与 available_at_field，并通过逐行可得日校验，方可用于正式回测或 TAA。",
            "example": {
                "growth": {
                    "point_in_time": True,
                    "available_at_field": "growth_available_at",
                }
            },
        },
        "forbidden": [
            "属性访问",
            "下标",
            "任意函数",
            "关键字参数",
            "负数 lag",
            "lead/未来函数",
            "全样本归约",
            "Python/pandas 回退",
        ],
        "limits": {
            "max_expression_length": MAX_EXPRESSION_LENGTH,
            "max_ast_nodes": MAX_AST_NODES,
            "max_ast_depth": MAX_AST_DEPTH,
            "max_window": MAX_WINDOW,
            "max_time_observations": MAX_TIME_OBSERVATIONS,
        },
    }


def _formula_error(
    code: str,
    message: str,
    *,
    details: dict[str, Any] | None = None,
) -> ValidationError:
    diagnostics = [details] if details else None
    return ValidationError(code, message, "features.formula", diagnostics=diagnostics)


def _translate_typed_error(error: TypedDslError) -> ValidationError:
    code_map = {
        "ILLEGAL_AST": "FORMULA_NODE_FORBIDDEN",
        "UNKNOWN_OPERATOR": "FORMULA_FUNCTION_FORBIDDEN",
        "UNKNOWN_VARIABLE": "FORMULA_VARIABLE_NOT_FOUND",
        "INVALID_LITERAL": "FORMULA_LITERAL_FORBIDDEN",
        "INVALID_PARAMETER": "FORMULA_ARGUMENT_ERROR",
        "OUTPUT_CONTRACT_MISMATCH": "FORMULA_MUST_RETURN_SERIES",
        "NJIT_PLAN_COMPILE_FAILED": "FORMULA_NJIT_COMPILE_FAILED",
        "NJIT_PLAN_NOT_WARMED": "FORMULA_NJIT_PLAN_NOT_WARMED",
    }
    return _formula_error(
        code_map.get(error.code, error.code),
        error.message,
        details={"typed_code": error.code, **dict(error.details or {})},
    )


def _numeric_variable_types(frame: pd.DataFrame) -> dict[str, ValueType]:
    result: dict[str, ValueType] = {}
    for raw_name in frame.columns:
        name = str(raw_name)
        if name in SYSTEM_COLUMNS or name.startswith("_") or not name.isidentifier():
            continue
        numeric = pd.to_numeric(frame[raw_name], errors="coerce")
        if name == "value" or bool(numeric.notna().any()):
            result[name] = ValueType.series(
                "T",
                semantic_dimension="dimensionless",
            )
    return result


def _dag_depth(plan: TypedExpressionPlan) -> int:
    depths: dict[int, int] = {}
    for node in plan.nodes:
        depths[node.node_id] = 1 + max(
            (depths[input_id] for input_id in node.inputs),
            default=0,
        )
    return depths.get(plan.root_id, 0)


def _compose_formula(
    expression: str,
    frame: pd.DataFrame,
) -> tuple[TypedExpressionPlan, list[str]]:
    if not expression.strip():
        raise _formula_error("EMPTY_FORMULA", "自定义公式不能为空。")
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise _formula_error(
            "FORMULA_TOO_COMPLEX",
            f"公式长度不能超过 {MAX_EXPRESSION_LENGTH} 个字符。",
        )
    variable_types = _numeric_variable_types(frame)
    try:
        plan = compose_typed_expression(
            expression,
            variable_types=variable_types,
            output_contract="series",
            dsl_version=TYPED_DSL_VERSION,
            operator_registry_version=TYPED_OPERATOR_REGISTRY_VERSION,
            max_nodes=MAX_AST_NODES,
            max_depth=MAX_AST_DEPTH,
        )
    except TypedDslError as exc:
        raise _translate_typed_error(exc) from exc

    dependencies = sorted(plan.context_requirements)
    unavailable = [name for name in dependencies if name not in variable_types]
    if unavailable:
        raise _formula_error(
            "FORMULA_VARIABLE_NOT_FOUND",
            f"数据源中不存在公式变量 {unavailable[0]}。",
            details={
                "variables": unavailable,
                "available_columns": sorted(variable_types),
            },
        )
    non_causal = sorted(
        {
            str(node.operator_id)
            for node in plan.nodes
            if node.operator_id is not None
            and node.operator_id not in CAUSAL_OPERATOR_IDS
        }
    )
    if non_causal:
        raise _formula_error(
            "FORMULA_NON_CAUSAL_OPERATOR",
            "公式包含只能用于全样本指标、不能用于逐期情景识别的算子。",
            details={"operators": non_causal},
        )
    nodes = {node.node_id: node for node in plan.nodes}
    for node in plan.nodes:
        if node.operator_id not in {
            "lag", "difference", "rolling_mean", "rolling_std", "rolling_min",
            "rolling_max", "recursive_smooth",
        } or len(node.inputs) < 2:
            continue
        periods_node = nodes[node.inputs[1]]
        try:
            periods = float(periods_node.label)
        except (TypeError, ValueError):
            continue
        if periods > MAX_WINDOW:
            raise _formula_error(
                "FORMULA_ARGUMENT_ERROR",
                f"窗口或滞后期不能超过 {MAX_WINDOW}。",
                details={"operator": node.operator_id, "periods": periods},
            )
    return plan, dependencies


def _formula_compile_token(plan: TypedExpressionPlan) -> str:
    """Bind an explicit preparation result to one immutable typed DAG."""

    material = "|".join(
        (
            FORMULA_ALLOWLIST_VERSION,
            plan.expression_hash,
            numba_plan_id(plan),
        )
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def prepare_formula(expression: str, frame: pd.DataFrame) -> dict[str, Any]:
    """Explicitly compile and freeze a formula before a numerical run.

    This is the only formula entry point allowed to create a Numba plan.  The
    normal evaluation path below only binds a plan that is already present in
    the process cache, so a cache miss fails closed instead of compiling during
    a calculation request.
    """

    plan, dependencies = _compose_formula(expression, frame)
    try:
        runtime = TypedIndicatorRuntime.from_plan(
            plan,
            max_time=MAX_TIME_OBSERVATIONS,
        )
    except TypedDslError as exc:
        raise _translate_typed_error(exc) from exc
    compiled = runtime.compiled_plan.metadata()
    return {
        "compile_token": _formula_compile_token(plan),
        "compiled_plan_id": compiled["compiled_plan_id"],
        "compile_status": "compiled",
        "expression_hash": plan.expression_hash,
        "referenced_columns": dependencies,
        "execution_backend": "numba_njit_fixed_signature",
        "njit_required": True,
        "nopython": True,
        "request_time_compilation": 0,
        "python_fallback": 0,
        **compiled,
    }


def _require_formula_numeric_runtime() -> None:
    try:
        require_historical_regime_kernels_ready()
    except RuntimeError as exc:
        raise _formula_error(
            "FORMULA_NUMERIC_KERNELS_NOT_WARMED",
            "公式数值预处理内核尚未完成启动预热，当前运行已停止。",
        ) from exc


def mask_formula_numeric_outputs(
    formula_values: np.ndarray,
    numeric_outputs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Apply the formula-missing mask to a batch of downstream numeric rows."""

    _require_formula_numeric_runtime()
    formula_array = np.ascontiguousarray(formula_values, dtype=np.float64)
    output_matrix = np.ascontiguousarray(numeric_outputs, dtype=np.float64)
    if output_matrix.ndim != 2:
        raise _formula_error(
            "FORMULA_NUMERIC_POSTPROCESS_INVALID",
            "公式下游数值结果必须是二维批量矩阵。",
        )
    try:
        return formula_missing_numeric_mask_kernel(formula_array, output_matrix)
    except ValueError as exc:
        raise _formula_error(
            "FORMULA_NUMERIC_POSTPROCESS_INVALID",
            "公式结果与下游数值结果无法按时间对齐。",
        ) from exc


def evaluate_formula(
    expression: str,
    frame: pd.DataFrame,
    *,
    compile_token: str | None = None,
) -> FormulaResult:
    """Execute an explicitly prepared fixed-signature NJIT formula plan."""

    plan, dependencies = _compose_formula(expression, frame)
    expected_token = _formula_compile_token(plan)
    if compile_token != expected_token:
        raise _formula_error(
            "FORMULA_COMPILE_TOKEN_REQUIRED",
            "自定义公式必须先显式编译并取得与当前定义一致的编译凭证。",
            details={"compiled_plan_id": numba_plan_id(plan)},
        )
    _require_formula_numeric_runtime()
    try:
        runtime = TypedIndicatorRuntime.from_warmed_plan(
            plan,
            max_time=MAX_TIME_OBSERVATIONS,
        )
    except TypedDslError as exc:
        raise _translate_typed_error(exc) from exc
    numeric = {
        name: np.ascontiguousarray(
            pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=np.float64),
        )
        for name in dependencies
    }
    numeric_matrix = np.empty(
        (len(dependencies), len(frame)),
        dtype=np.float64,
    )
    for row, name in enumerate(dependencies):
        numeric_matrix[row] = numeric[name]
    complete, block_starts, block_stops, complete_count = (
        formula_input_blocks_kernel(np.ascontiguousarray(numeric_matrix))
    )

    output = np.full(len(frame), np.nan, dtype=np.float64)
    executed_blocks = 0
    for raw_start, raw_stop in zip(block_starts, block_stops):
        start = int(raw_start)
        stop = int(raw_stop)
        context = {
            name: np.ascontiguousarray(values[start:stop], dtype=np.float64)
            for name, values in numeric.items()
        }
        try:
            permits_warmup_missing = any(node.operator_id in {"rolling_mean", "rolling_std", "rolling_min", "rolling_max"} for node in plan.nodes)
            result = np.asarray(compute_warmed_series(runtime, context) if permits_warmup_missing else runtime.compute(context), dtype=np.float64)
        except TypedDslError as exc:
            if exc.code == "INSUFFICIENT_SAMPLE":
                continue
            raise _formula_error(
                "FORMULA_NJIT_EXECUTION_FAILED",
                "公式的 NJIT 计算计划执行失败，请检查输入定义域和参数。",
                details={"typed_code": exc.code, **dict(exc.details or {})},
            ) from exc
        if result.ndim != 1:
            raise _formula_error(
                "FORMULA_NJIT_OUTPUT_INVALID",
                "公式的 NJIT 输出没有形成可对齐的一维时间序列。",
                details={
                    "shape": list(result.shape),
                    "input_size": stop - start,
                },
            )
        try:
            aligned = formula_align_block_kernel(
                np.ascontiguousarray(result),
                np.int64(start),
                np.int64(stop),
                output,
            )
        except ValueError as exc:
            raise _formula_error(
                "FORMULA_NJIT_OUTPUT_INVALID",
                "公式的 NJIT 输出没有形成可对齐的一维时间序列。",
                details={
                    "shape": list(result.shape),
                    "input_size": stop - start,
                },
            ) from exc
        if aligned == 0:
            continue
        executed_blocks += 1

    masked_output, missing_positions, valid_output_count = mask_formula_numeric_outputs(
        output,
        np.ascontiguousarray(output.reshape(1, output.size)),
    )
    output = np.ascontiguousarray(masked_output[0])
    if valid_output_count == 0:
        raise _formula_error(
            "FORMULA_NO_VALID_OUTPUT",
            "公式没有产生任何有效数值，请检查输入列、样本长度和定义域。",
        )

    operator_ids = sorted(
        {
            str(node.operator_id)
            for node in plan.nodes
            if node.operator_id is not None
        }
    )
    functions = sorted(
        {
            "abs" if node.operator_id == "absolute" else str(node.operator_id)
            for node in plan.nodes
            if node.kind == "call" and node.operator_id is not None
        }
    )
    compiled = runtime.compiled_plan.metadata()
    numeric_execution = historical_regime_execution_audit(
        "formula_prepostprocess",
        list(FORMULA_NUMERIC_KERNEL_IDS),
    )
    kernel_signatures = {
        **dict(compiled.get("kernel_signatures") or {}),
        **dict(numeric_execution["kernel_signatures"]),
    }
    fingerprint_payload = "|".join(
        (
            FORMULA_ALLOWLIST_VERSION,
            plan.expression_hash,
            str(compiled["compiled_plan_id"]),
            str(compiled["kernel_version"]),
            str(numeric_execution["kernel_fingerprint"]),
        )
    )
    audit = {
        "language_id": FORMULA_LANGUAGE_ID,
        "allowlist_version": FORMULA_ALLOWLIST_VERSION,
        "evaluator_version": FORMULA_EVALUATOR_VERSION,
        "dsl_version": plan.dsl_version,
        "compiler_version": plan.compiler_version,
        "operator_registry_version": plan.operator_registry_version,
        "numeric_kernel_version": NUMERIC_KERNEL_VERSION,
        "engine_version": ENGINE_VERSION,
        "execution_backend": "numba_njit_fixed_signature",
        "njit_required": True,
        "njit_supported": True,
        "python_fallback": 0,
        "python_operator_calls": 0,
        "request_time_compilation": 0,
        "expression": expression,
        "normalized_expression": plan.python_expression,
        "expression_hash": plan.expression_hash,
        "ast_nodes": len(plan.nodes),
        "ast_depth": _dag_depth(plan),
        "dag": plan.graph_payload(),
        "referenced_columns": dependencies,
        "operators": operator_ids,
        "functions": functions,
        "is_causal": True,
        "uses_future_data": False,
        "repaints": False,
        "executed_blocks": executed_blocks,
        "eligible_blocks": int(block_starts.size),
        "complete_input_observations": int(complete_count),
        "missing_input_observations": int(complete.size - complete_count),
        "valid_output_observations": int(valid_output_count),
        "missing_output_observations": int(missing_positions.size),
        "numeric_prepostprocess": numeric_execution,
        "fingerprint": hashlib.sha256(fingerprint_payload.encode("utf-8")).hexdigest(),
        **compiled,
        "backend": "numba_njit_fixed_signature",
        "execution_backend": "numba_njit_fixed_signature",
        "kernel_signatures": kernel_signatures,
        "nopython": bool(compiled.get("nopython"))
        and bool(numeric_execution.get("nopython")),
        "object_mode": 0,
        "request_time_compilation": 0,
        "python_fallback": 0,
    }
    return FormulaResult(
        values=pd.Series(output, index=frame.index),
        audit=validate_execution_audit(audit),
    )


__all__ = [
    "FORMULA_ALLOWLIST_VERSION",
    "FORMULA_EVALUATOR_VERSION",
    "FormulaResult",
    "evaluate_formula",
    "formula_language_meta",
    "mask_formula_numeric_outputs",
    "prepare_formula",
]

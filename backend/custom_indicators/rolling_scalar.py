"""Compile a locked scalar indicator formula into a fixed-window time series.

The transformation is performed once while a definition is created or warmed.
The resulting expression is an ordinary typed time-series DSL expression, so it
uses the existing fixed-signature NJIT plan, LaTeX renderer, snapshot path and
native-Excel formula exporter without a request-time callback or Python loop.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
from typing import Any, Mapping

from .errors import ValidationError


ROLLING_SCALAR_TRANSFORM_VERSION = "1.0.0"
MIN_ROLLING_WINDOW_OBSERVATIONS = 2
MAX_ROLLING_WINDOW_OBSERVATIONS = 5000

# Scalar reductions that have an exact transparent rolling representation in
# the current typed time-series operator registry.
_REDUCTION_CALLS = frozenset(
    {
        "mean",
        "std",
        "variance",
        "min_value",
        "max_value",
        "sum",
    }
)

# These calls preserve their scalar/series broadcasting semantics after their
# children have been lifted. They are intentionally explicit: an unsupported
# scalar reduction must fail closed rather than silently changing meaning.
_PRESERVED_CALLS = frozenset(
    {
        "absolute",
        "add",
        "clip",
        "divide",
        "divide_or_default",
        "equal",
        "exp",
        "greater_equal",
        "greater_than",
        "less_equal",
        "less_than",
        "log",
        "logical_and",
        "logical_not",
        "logical_or",
        "maximum",
        "minimum",
        "multiply",
        "negate",
        "normal_pdf",
        "normal_ppf",
        "not_equal",
        "power",
        "reciprocal",
        "sign",
        "sqrt",
        "subtract",
        "where",
    }
)

# These run-context scalars keep the same meaning for every observation in the
# generated series. ``observation_count`` is replaced by the locked window.
_BROADCAST_RUNTIME_SCALARS = frozenset(
    {
        "annual_risk_free_rate_decimal",
        "periods_per_year",
        "risk_free_rate_per_observation",
    }
)

# These values describe one complete evaluation window and therefore cannot be
# broadcast from the outer request into every rolling window without changing
# the source indicator's meaning. Dedicated rolling semantics can be added in a
# later transform version.
_UNSUPPORTED_WINDOW_SCALARS = frozenset(
    {
        "risk_free_return_window",
        "window_elapsed_days",
    }
)


def _finite_integer(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        raise ValidationError(
            "INVALID_ROLLING_WINDOW",
            "滚动窗口必须是整数观察数。",
            field=field,
        )
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            "INVALID_ROLLING_WINDOW",
            "滚动窗口必须是整数观察数。",
            field=field,
        ) from exc
    integer = int(number)
    if not math.isfinite(number) or number != integer:
        raise ValidationError(
            "INVALID_ROLLING_WINDOW",
            "滚动窗口必须是整数观察数。",
            field=field,
        )
    return integer


def normalize_rolling_window(
    window_observations: Any,
    min_periods: Any | None = None,
) -> tuple[int, int]:
    """Return a fixed, reproducible observation window and minimum sample."""

    window = _finite_integer(
        window_observations,
        field="rolling_source.window_observations",
    )
    if not MIN_ROLLING_WINDOW_OBSERVATIONS <= window <= MAX_ROLLING_WINDOW_OBSERVATIONS:
        raise ValidationError(
            "INVALID_ROLLING_WINDOW",
            (
                "滚动窗口必须位于 "
                f"{MIN_ROLLING_WINDOW_OBSERVATIONS} 至 "
                f"{MAX_ROLLING_WINDOW_OBSERVATIONS} 个观察值之间。"
            ),
            field="rolling_source.window_observations",
        )
    minimum = window if min_periods is None else _finite_integer(
        min_periods,
        field="rolling_source.min_periods",
    )
    if minimum < 1 or minimum > window:
        raise ValidationError(
            "INVALID_ROLLING_MIN_PERIODS",
            "最少有效观察数必须位于 1 与滚动窗口之间。",
            field="rolling_source.min_periods",
        )
    return window, minimum


def _literal_integer(node: ast.AST, *, operator: str, parameter: str) -> int:
    if (
        not isinstance(node, ast.Constant)
        or isinstance(node.value, bool)
        or not isinstance(node.value, (int, float))
    ):
        raise ValidationError(
            "ROLLING_SCALAR_LITERAL_REQUIRED",
            f"标量指标中 {operator} 的 {parameter} 必须是固定整数常量。",
            field="expression",
        )
    value = float(node.value)
    integer = int(value)
    if not math.isfinite(value) or value != integer or integer < 0:
        raise ValidationError(
            "ROLLING_SCALAR_LITERAL_REQUIRED",
            f"标量指标中 {operator} 的 {parameter} 必须是非负整数常量。",
            field="expression",
        )
    return integer


class _RollingScalarTransformer(ast.NodeTransformer):
    def __init__(self, window: int, minimum: int) -> None:
        self.window = window
        self.minimum = minimum
        self.lifted_reductions = 0

    @staticmethod
    def _name(identifier: str) -> ast.Name:
        return ast.Name(id=identifier, ctx=ast.Load())

    @staticmethod
    def _number(value: int | float) -> ast.Constant:
        return ast.Constant(value=value)

    def _rolling_call(
        self,
        operator: str,
        values: ast.AST,
        *extra: ast.AST,
    ) -> ast.Call:
        return ast.Call(
            func=self._name(operator),
            args=[
                values,
                self._number(self.window),
                *extra,
                self._number(self.minimum),
            ],
            keywords=[],
        )

    def visit_Name(self, node: ast.Name) -> ast.AST:  # noqa: N802
        if node.id == "observation_count":
            return ast.copy_location(self._number(self.window), node)
        if node.id in _UNSUPPORTED_WINDOW_SCALARS:
            raise ValidationError(
                "ROLLING_SCALAR_CONTEXT_UNSUPPORTED",
                (
                    f"标量指标依赖 {node.id}，该变量描述完整计算区间，"
                    "尚不能直接提升为逐观察窗口语义。"
                ),
                field="expression",
                diagnostics=[
                    {
                        "code": "ROLLING_SCALAR_CONTEXT_UNSUPPORTED",
                        "variable": node.id,
                        "message": "请选择不依赖完整区间日历长度的标量指标。",
                    }
                ],
            )
        return node

    def visit_Call(self, node: ast.Call) -> ast.AST:  # noqa: N802
        if not isinstance(node.func, ast.Name) or node.keywords:
            raise ValidationError(
                "ROLLING_SCALAR_EXPRESSION_UNSUPPORTED",
                "滚动提升仅支持受控 DSL 的直接函数调用。",
                field="expression",
            )
        operator = node.func.id
        if operator in _REDUCTION_CALLS:
            if not node.args:
                raise ValidationError(
                    "ROLLING_SCALAR_ARITY_MISMATCH",
                    f"标量归约算子 {operator} 缺少输入序列。",
                    field="expression",
                )
            values = self.visit(copy.deepcopy(node.args[0]))
            self.lifted_reductions += 1
            if operator == "mean":
                if len(node.args) != 1:
                    raise ValidationError(
                        "ROLLING_SCALAR_ARITY_MISMATCH",
                        "mean 只支持一个输入序列。",
                        field="expression",
                    )
                return ast.copy_location(
                    self._rolling_call("rolling_mean", values),
                    node,
                )
            if operator == "std":
                if len(node.args) not in {1, 2}:
                    raise ValidationError(
                        "ROLLING_SCALAR_ARITY_MISMATCH",
                        "std 只支持 values 或 values, ddof。",
                        field="expression",
                    )
                ddof = 1 if len(node.args) == 1 else _literal_integer(
                    node.args[1],
                    operator="std",
                    parameter="ddof",
                )
                return ast.copy_location(
                    self._rolling_call(
                        "rolling_std",
                        values,
                        self._number(ddof),
                    ),
                    node,
                )
            if operator == "variance":
                if len(node.args) not in {1, 2}:
                    raise ValidationError(
                        "ROLLING_SCALAR_ARITY_MISMATCH",
                        "variance 只支持 values 或 values, ddof。",
                        field="expression",
                    )
                ddof = 1 if len(node.args) == 1 else _literal_integer(
                    node.args[1],
                    operator="variance",
                    parameter="ddof",
                )
                rolling_std = self._rolling_call(
                    "rolling_std",
                    values,
                    self._number(ddof),
                )
                return ast.copy_location(
                    ast.BinOp(
                        left=rolling_std,
                        op=ast.Pow(),
                        right=self._number(2),
                    ),
                    node,
                )
            if operator == "min_value":
                if len(node.args) != 1:
                    raise ValidationError(
                        "ROLLING_SCALAR_ARITY_MISMATCH",
                        "min_value 只支持一个输入序列。",
                        field="expression",
                    )
                return ast.copy_location(
                    self._rolling_call("rolling_min", values),
                    node,
                )
            if operator == "max_value":
                if len(node.args) != 1:
                    raise ValidationError(
                        "ROLLING_SCALAR_ARITY_MISMATCH",
                        "max_value 只支持一个输入序列。",
                        field="expression",
                    )
                return ast.copy_location(
                    self._rolling_call("rolling_max", values),
                    node,
                )
            if operator == "sum":
                if len(node.args) != 1:
                    raise ValidationError(
                        "ROLLING_SCALAR_ARITY_MISMATCH",
                        "sum 只支持一个输入序列。",
                        field="expression",
                    )
                if self.minimum != self.window:
                    raise ValidationError(
                        "ROLLING_SCALAR_OPERATOR_UNSUPPORTED",
                        "sum 的自动滚动提升当前要求完整窗口，不能使用较小的 min_periods。",
                        field="expression",
                    )
                return ast.copy_location(
                    ast.BinOp(
                        left=self._rolling_call("rolling_mean", values),
                        op=ast.Mult(),
                        right=self._number(self.window),
                    ),
                    node,
                )

        if operator not in _PRESERVED_CALLS:
            raise ValidationError(
                "ROLLING_SCALAR_OPERATOR_UNSUPPORTED",
                f"标量算子 {operator} 尚无等价的透明滚动实现。",
                field="expression",
                diagnostics=[
                    {
                        "code": "ROLLING_SCALAR_OPERATOR_UNSUPPORTED",
                        "operator": operator,
                        "message": "该标量指标暂不能自动转换为时序指标。",
                    }
                ],
            )
        return ast.copy_location(
            ast.Call(
                func=copy.deepcopy(node.func),
                args=[self.visit(copy.deepcopy(argument)) for argument in node.args],
                keywords=[],
            ),
            node,
        )


def lift_scalar_expression(
    expression: str,
    *,
    window_observations: Any,
    min_periods: Any | None = None,
) -> dict[str, Any]:
    """Lift one scalar expression into a deterministic rolling series formula."""

    window, minimum = normalize_rolling_window(window_observations, min_periods)
    source = str(expression or "").strip()
    if not source:
        raise ValidationError(
            "EMPTY_EXPRESSION",
            "源标量指标公式不能为空。",
            field="expression",
        )
    try:
        parsed = ast.parse(source, mode="eval")
    except SyntaxError as exc:
        raise ValidationError(
            "ROLLING_SCALAR_EXPRESSION_INVALID",
            "源标量指标公式无法解析。",
            field="expression",
        ) from exc
    transformer = _RollingScalarTransformer(window, minimum)
    transformed = transformer.visit(parsed)
    ast.fix_missing_locations(transformed)
    if transformer.lifted_reductions <= 0:
        raise ValidationError(
            "ROLLING_SCALAR_REDUCTION_REQUIRED",
            "源标量指标没有可提升的序列归约步骤。",
            field="expression",
        )
    return {
        "expression": ast.unparse(transformed.body),
        "window_observations": window,
        "min_periods": minimum,
        "transform_version": ROLLING_SCALAR_TRANSFORM_VERSION,
        "lifted_reductions": transformer.lifted_reductions,
    }


def scalar_definition_hash(definition: Mapping[str, Any]) -> str:
    payload = {
        key: copy.deepcopy(definition.get(key))
        for key in (
            "id",
            "revision",
            "expression",
            "dsl_version",
            "operator_registry_version",
            "numeric_kernel_version",
            "variable_registry_version",
            "data_contract_version",
            "context_schema_version",
            "annual_risk_free_rate_percent",
        )
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def rolling_source_metadata(
    source_definition: Mapping[str, Any],
    *,
    window_observations: int,
    min_periods: int,
) -> dict[str, Any]:
    return {
        "kind": "scalar_indicator_rolling_window",
        "indicator_id": str(source_definition.get("id") or ""),
        "indicator_revision": int(source_definition.get("revision") or 0),
        "indicator_name": str(source_definition.get("name") or ""),
        "definition_hash": scalar_definition_hash(source_definition),
        "window_kind": "observations",
        "window_observations": int(window_observations),
        "min_periods": int(min_periods),
        "transform_version": ROLLING_SCALAR_TRANSFORM_VERSION,
        "detached": False,
    }


def validate_scalar_rolling_source(definition: Mapping[str, Any]) -> None:
    if str(definition.get("result_kind") or "scalar") != "scalar":
        raise ValidationError(
            "ROLLING_SOURCE_MUST_BE_SCALAR",
            "只有标量指标可以通过滚动窗口提升为时序指标。",
            field="rolling_source.indicator_id",
        )
    if str(definition.get("context_kind") or "single_product") != "single_product":
        raise ValidationError(
            "ROLLING_SOURCE_CONTEXT_UNSUPPORTED",
            "滚动提升当前只支持单产品标量指标。",
            field="rolling_source.indicator_id",
        )
    if not str(definition.get("dsl_version") or "").startswith("2."):
        raise ValidationError(
            "ROLLING_SOURCE_TYPED_DSL_REQUIRED",
            "滚动提升只支持 typed DSL 标量指标。",
            field="rolling_source.indicator_id",
        )


__all__ = [
    "MAX_ROLLING_WINDOW_OBSERVATIONS",
    "MIN_ROLLING_WINDOW_OBSERVATIONS",
    "ROLLING_SCALAR_TRANSFORM_VERSION",
    "lift_scalar_expression",
    "normalize_rolling_window",
    "rolling_source_metadata",
    "scalar_definition_hash",
    "validate_scalar_rolling_source",
]

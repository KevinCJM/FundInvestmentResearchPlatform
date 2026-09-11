"""Explicit, versioned operator registry for typed indicator DSL v2."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from functools import lru_cache
from statistics import NormalDist
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from cal_indicators.typed_latex import operator_display_latex_template

from cal_indicators.typed_numeric_backend import (
    reduce_max,
    reduce_mean,
    reduce_min,
    reduce_product,
    reduce_std,
    reduce_sum,
    reduce_variance,
    scan_product,
    scan_return,
    scan_sum,
)

from cal_indicators.typed_types import (
    ValueType,
    TypedDslError,
    elementwise_result,
    require_same_type,
    symbolic_elements,
    type_from_axes,
)


LEGACY_TYPED_DSL_VERSION = "2.0.0"
LEGACY_OPERATOR_REGISTRY_VERSION = "2.0.0"
LEGACY_TYPED_COMPILER_VERSION = "typed-ast-1"
COMPAT_TYPED_DSL_VERSION = "2.1.0"
COMPAT_OPERATOR_REGISTRY_VERSION = "2.1.0"
COMPAT_TYPED_COMPILER_VERSION = "typed-ast-2"
PREVIOUS_TYPED_DSL_VERSION = "2.2.0"
PREVIOUS_OPERATOR_REGISTRY_VERSION = "2.2.0"
PREVIOUS_TYPED_COMPILER_VERSION = "typed-numba-3"
ROLLING_TYPED_DSL_VERSION = "2.3.0"
ROLLING_OPERATOR_REGISTRY_VERSION = "2.3.0"
ROLLING_TYPED_COMPILER_VERSION = "typed-numba-4"
TYPED_DSL_VERSION = "2.4.0"
TYPED_COMPILER_VERSION = "typed-numba-5"
TYPED_OPERATOR_REGISTRY_VERSION = "2.4.0"
SUPPORTED_TYPED_DSL_VERSIONS = frozenset(
    {
        LEGACY_TYPED_DSL_VERSION,
        COMPAT_TYPED_DSL_VERSION,
        PREVIOUS_TYPED_DSL_VERSION,
        ROLLING_TYPED_DSL_VERSION,
        TYPED_DSL_VERSION,
    }
)
SUPPORTED_OPERATOR_REGISTRY_VERSIONS = frozenset(
    {
        LEGACY_OPERATOR_REGISTRY_VERSION,
        COMPAT_OPERATOR_REGISTRY_VERSION,
        PREVIOUS_OPERATOR_REGISTRY_VERSION,
        ROLLING_OPERATOR_REGISTRY_VERSION,
        TYPED_OPERATOR_REGISTRY_VERSION,
    }
)
from .operator_lowering import COMPOSITE_OPERATOR_IDS

# Persisted formula spellings are accepted by the compiler, never offered as
# opaque numeric operations in the authoring catalog.
ROLLING_COMPAT_OPERATOR_IDS = frozenset(
    {"rolling_mean", "rolling_std", "rolling_min", "rolling_max"}
)
PUBLIC_OPERATOR_EXCLUSIONS = COMPOSITE_OPERATOR_IDS | ROLLING_COMPAT_OPERATOR_IDS
PRICE_SEMANTIC_DIMENSIONS = frozenset(
    {"adjusted_nav", "reported_nav", "raw_market_price"}
)
PATH_LEVEL_SEMANTIC_DIMENSIONS = PRICE_SEMANTIC_DIMENSIONS | frozenset(
    {"dimensionless"}
)
V22_OPERATOR_IDS = frozenset({"drawdown_series", "new_high_mask"})
V23_OPERATOR_IDS = frozenset(
    {
        "rolling_mean",
        "rolling_std",
        "rolling_min",
        "rolling_max",
        "recursive_smooth",
        "divide_or_default",
    }
)
V24_OPERATOR_IDS = frozenset({"rolling_window"})
ADDITIVE_RATE_DIMENSIONS = frozenset({"return_decimal", "rate_decimal"})
LEGACY_OPERATOR_IDS = frozenset(
    {
        "add",
        "subtract",
        "multiply",
        "divide",
        "power",
        "minimum",
        "maximum",
        "negate",
        "absolute",
        "sqrt",
        "clip",
        "sum",
        "product",
        "mean",
        "min_value",
        "max_value",
        "variance",
        "std",
        "cumulative_sum",
        "cumulative_product",
        "cumulative_return",
        "last",
        "total_return",
        "annualized_return",
        "sum_time",
        "mean_time",
        "product_time",
        "variance_time",
        "std_time",
        "min_time",
        "max_time",
        "sum_asset",
        "mean_asset",
        "product_asset",
        "variance_asset",
        "std_asset",
        "min_asset",
        "max_asset",
        "transpose",
        "dot",
        "outer",
        "matmul",
        "matvec",
        "diag",
        "trace",
        "solve",
        "covariance",
        "correlation",
        "portfolio_returns",
        "quadratic_form",
        "active_returns",
    }
)

InferFunction = Callable[[tuple[ValueType, ...]], ValueType]
RuntimeFunction = Callable[..., Any]
CostFunction = Callable[[tuple[ValueType, ...], ValueType], str]


@dataclass(frozen=True)
class OperatorSignature:
    inputs: tuple[str, ...]
    output: str
    shape_rule: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "inputs": list(self.inputs),
            "output": self.output,
            "shape_rule": self.shape_rule,
        }


@dataclass(frozen=True)
class TypedOperatorSpec:
    """Schema/type metadata plus a test-only NumPy reference oracle.

    Production evaluation never calls ``evaluate``; typed_numba_plan lowers
    operator ids directly to fixed-signature NJIT dispatchers.  Keeping this
    reference callable supports controlled numerical-parity tests only.
    """

    operator_id: str
    version: str
    category: str
    signatures: tuple[OperatorSignature, ...]
    description: str
    infer: InferFunction
    evaluate: RuntimeFunction
    cost_model: str = "elementwise"
    cost: CostFunction | None = None
    aliases: tuple[str, ...] = ()
    latex_template: str = ""
    # Unknown third-party semantics must not acquire rolling capability by shape alone.
    interval_policy: str | None = None

    @property
    def arities(self) -> frozenset[int]:
        return frozenset(len(signature.inputs) for signature in self.signatures)

    def argument_names(self, arity: int) -> tuple[str, ...]:
        names = _OPERATOR_ARGUMENT_NAMES.get((self.operator_id, arity))
        if names is not None:
            return names
        return tuple(f"input_{index}" for index in range(1, arity + 1))

    def infer_output(self, inputs: tuple[ValueType, ...]) -> ValueType:
        if len(inputs) not in self.arities:
            expected = ", ".join(str(arity) for arity in sorted(self.arities))
            raise TypedDslError(
                "ARITY_MISMATCH",
                f"函数 {self.operator_id} 需要 {expected} 个参数，实际为 {len(inputs)} 个。",
                details={
                    "operator": self.operator_id,
                    "expected": sorted(self.arities),
                },
            )
        return self.infer(inputs)

    def cost_expression(
        self,
        inputs: tuple[ValueType, ...],
        output: ValueType,
    ) -> str:
        if self.cost is not None:
            return self.cost(inputs, output)
        return symbolic_elements(output)

    def to_catalog_entry(self) -> dict[str, Any]:
        # Imported lazily to keep the type-inference registry independent from
        # the numeric kernel module during interpreter start-up.
        from cal_indicators.typed_numba_kernels import kernel_catalog_entry

        kernel = kernel_catalog_entry(self.operator_id)
        preferred_arity = min(self.arities)
        return {
            "id": self.operator_id,
            "version": self.version,
            "category": self.category,
            "interval_policy": self.interval_policy,
            "description": self.description,
            "signatures": [
                {
                    **signature.to_dict(),
                    "parameters": list(self.argument_names(len(signature.inputs))),
                }
                for signature in self.signatures
            ],
            "aliases": list(self.aliases),
            "latex_template": self.latex_template,
            "display_latex_template": operator_display_latex_template(
                self.operator_id,
                self.argument_names(preferred_arity),
            ),
            "cost_model": self.cost_model,
            "execution_backend": "numba_njit_fixed_signature",
            **kernel,
        }


_OPERATOR_ARGUMENT_NAMES: Mapping[tuple[str, int], tuple[str, ...]] = {
    ("rolling_apply", 2): ("calculation", "window"),
    ("rolling_apply", 3): ("calculation", "window", "min_periods"),
    ("rolling_apply", 4): ("calculation", "window", "dates", "annual_rate"),
    ("rolling_apply", 5): ("calculation", "window", "dates", "annual_rate", "min_periods"),
    ("finite_mask", 1): ("values",),
    ("last_drawdown_interval", 1): ("drawdowns",),
    ("interval_start", 1): ("interval",),
    ("interval_trough", 1): ("interval",),
    ("interval_recovery", 1): ("interval",),
    ("value_at", 2): ("values", "position"),
    ("require_positive", 1): ("values",),
    ("require_nonnegative", 1): ("values",),
    ("linear_fit", 1): ("values",),
    ("linear_fit", 2): ("x", "y"),
    **{(f"fit_{field}", 1): ("fit",) for field in (
        "slope", "intercept", "residual_sum_squares", "total_sum_squares", "observation_count",
    )},
    ("days_between", 2): ("start_date", "end_date"),
    **{
        (operator_id, 2): ("lhs", "rhs")
        for operator_id in (
            "add",
            "subtract",
            "multiply",
            "divide",
            "power",
            "minimum",
            "maximum",
            "dot",
            "outer",
            "matmul",
            "matvec",
            "solve",
            "quadratic_form",
            "active_returns",
            "equal",
            "not_equal",
            "less_than",
            "less_equal",
            "greater_than",
            "greater_equal",
            "logical_and",
            "logical_or",
        )
    },
    **{
        (operator_id, 1): ("values",)
        for operator_id in (
            "negate",
            "absolute",
            "sqrt",
            "sum",
            "product",
            "mean",
            "variance",
            "std",
            "min_value",
            "max_value",
            "cumulative_sum",
            "cumulative_product",
            "cumulative_return",
            "last",
            "total_return",
            "sum_time",
            "mean_time",
            "product_time",
            "variance_time",
            "std_time",
            "min_time",
            "max_time",
            "sum_asset",
            "mean_asset",
            "product_asset",
            "variance_asset",
            "std_asset",
            "min_asset",
            "max_asset",
            "transpose",
            "diag",
            "trace",
            "log",
            "exp",
            "reciprocal",
            "sign",
            "first",
            "length",
            "lag",
            "difference",
            "logical_not",
            "median",
            "skewness",
            "excess_kurtosis",
            "mean_absolute_deviation",
            "root_mean_square",
            "cumulative_maximum",
            "cumulative_minimum",
            "cumulative_max",
            "cumulative_min",
            "argmin",
            "argmax",
            "max_consecutive_true",
        )
    },
    ("clip", 3): ("values", "lower", "upper"),
    ("divide", 2): ("numerator", "denominator"),
    ("power", 2): ("base", "exponent"),
    ("variance", 2): ("values", "ddof"),
    ("std", 2): ("values", "ddof"),
    ("annualized_return", 2): ("returns", "periods_per_year"),
    ("covariance", 1): ("asset_returns",),
    ("covariance", 2): ("lhs", "rhs"),
    ("correlation", 1): ("asset_returns",),
    ("correlation", 2): ("lhs", "rhs"),
    ("matmul", 2): ("lhs_matrix", "rhs_matrix"),
    ("matvec", 2): ("matrix", "vector"),
    ("solve", 2): ("matrix", "rhs"),
    ("quadratic_form", 2): ("vector", "matrix"),
    ("portfolio_returns", 2): ("asset_returns", "asset_weights"),
    **{
        (operator_id, 2): ("values", "mask")
        for operator_id in (
            "masked_sum",
            "masked_mean",
            "masked_variance",
            "masked_std",
            "masked_min",
            "masked_max",
            "masked_median",
            "sum_where",
            "mean_where",
            "variance_where",
            "std_where",
            "min_where",
            "max_where",
            "median_where",
        )
    },
    ("masked_count", 1): ("mask",),
    ("count_true", 1): ("mask",),
    ("quantile", 2): ("values", "probability"),
    ("normal_pdf", 1): ("values",),
    ("normal_ppf", 1): ("probability",),
    ("masked_quantile", 3): ("values", "mask", "probability"),
    ("quantile_where", 3): ("values", "mask", "probability"),
    ("where", 3): ("mask", "if_true", "if_false"),
    ("linear_slope", 1): ("values",),
    ("linear_slope", 2): ("x", "y"),
    ("linear_intercept", 1): ("values",),
    ("linear_intercept", 2): ("x", "y"),
    ("linear_r_squared", 1): ("values",),
    ("linear_r_squared", 2): ("x", "y"),
    ("regression_standard_error", 1): ("values",),
    ("regression_standard_error", 2): ("x", "y"),
    ("lag", 2): ("values", "periods"),
    ("difference", 2): ("values", "periods"),
    ("rolling_window", 2): ("values", "window"),
    ("rolling_window", 3): ("values", "window", "min_periods"),
    ("rolling_mean", 2): ("values", "window"),
    ("rolling_mean", 3): ("values", "window", "min_periods"),
    ("rolling_std", 2): ("values", "window"),
    ("rolling_std", 3): ("values", "window", "ddof"),
    ("rolling_std", 4): ("values", "window", "ddof", "min_periods"),
    ("rolling_min", 2): ("values", "window"),
    ("rolling_min", 3): ("values", "window", "min_periods"),
    ("rolling_max", 2): ("values", "window"),
    ("rolling_max", 3): ("values", "window", "min_periods"),
    ("recursive_smooth", 3): ("values", "periods", "initial"),
    ("divide_or_default", 3): ("numerator", "denominator", "default"),
    ("drawdown_series", 1): ("levels",),
    ("new_high_mask", 1): ("levels",),
}


def _type_error(
    operator: str, expected: str, actual: Sequence[ValueType]
) -> TypedDslError:
    return TypedDslError(
        "TYPE_MISMATCH",
        f"{operator} 需要 {expected}，实际为 "
        + ", ".join(str(item) for item in actual),
        details={"operator": operator, "actual": [item.to_dict() for item in actual]},
    )


def _require_numeric(operator: str, *values: ValueType) -> None:
    if any(not value.is_numeric for value in values):
        raise _type_error(operator, "numeric input", values)


def _semantic_mismatch(
    operator: str,
    lhs: ValueType,
    rhs: ValueType,
) -> TypedDslError:
    return TypedDslError(
        "SEMANTIC_DIMENSION_MISMATCH",
        f"{operator} 的语义量纲不兼容: {lhs.semantic_dimension} 与 "
        f"{rhs.semantic_dimension}。",
        details={"operator": operator, "left": lhs.to_dict(), "right": rhs.to_dict()},
    )


def _price_basis_mismatch(
    operator: str,
    lhs: ValueType,
    rhs: ValueType,
) -> TypedDslError:
    return TypedDslError(
        "PRICE_BASIS_MISMATCH",
        f"{operator} 不能混用价格基准 {lhs.price_basis} 与 {rhs.price_basis}。",
        details={"operator": operator, "left": lhs.to_dict(), "right": rhs.to_dict()},
    )


def _elementwise_structure(
    operator: str,
    lhs: ValueType,
    rhs: ValueType,
) -> ValueType:
    _require_numeric(operator, lhs, rhs)
    return elementwise_result(lhs, rhs)


def _compatible_semantics(
    operator: str,
    lhs: ValueType,
    rhs: ValueType,
) -> tuple[str, str | None]:
    left_dimension = lhs.semantic_dimension
    right_dimension = rhs.semantic_dimension
    if lhs.is_scalar and left_dimension == "dimensionless":
        dimension = right_dimension
    elif rhs.is_scalar and right_dimension == "dimensionless":
        dimension = left_dimension
    elif left_dimension == right_dimension:
        dimension = left_dimension
    elif {left_dimension, right_dimension} <= ADDITIVE_RATE_DIMENSIONS:
        # A periodic risk-free rate may be subtracted from a return series.
        dimension = "return_decimal"
    else:
        raise _semantic_mismatch(operator, lhs, rhs)
    if (
        dimension in PRICE_SEMANTIC_DIMENSIONS
        and lhs.price_basis is not None
        and rhs.price_basis is not None
        and lhs.price_basis != rhs.price_basis
    ):
        raise _price_basis_mismatch(operator, lhs, rhs)
    return dimension, lhs.price_basis or rhs.price_basis


def _additive_type(
    inputs: tuple[ValueType, ...],
    *,
    operator: str = "add",
) -> ValueType:
    lhs, rhs = inputs
    _require_numeric(operator, lhs, rhs)
    if not lhs.is_scalar and not rhs.is_scalar and lhs.axes != rhs.axes:
        # Nominal time/asset axis mistakes are more fundamental than their
        # business measure mismatch and should retain the stable axis error.
        _elementwise_structure(operator, lhs, rhs)
    dimension, price_basis = _compatible_semantics(operator, lhs, rhs)
    output = _elementwise_structure(operator, lhs, rhs)
    return output.with_semantics(dimension, price_basis)


def _subtract_type(inputs: tuple[ValueType, ...]) -> ValueType:
    return _additive_type(inputs, operator="subtract")


def _minimum_maximum_type(inputs: tuple[ValueType, ...]) -> ValueType:
    return _additive_type(inputs, operator="minimum/maximum")


def _product_semantics(lhs: ValueType, rhs: ValueType) -> tuple[str, str | None]:
    if lhs.semantic_dimension == "dimensionless":
        return rhs.semantic_dimension, rhs.price_basis
    if rhs.semantic_dimension == "dimensionless":
        return lhs.semantic_dimension, lhs.price_basis
    if lhs.semantic_dimension == rhs.semantic_dimension:
        return f"squared:{lhs.semantic_dimension}", None
    return f"derived:{lhs.semantic_dimension}*{rhs.semantic_dimension}", None


def _multiply_type(inputs: tuple[ValueType, ...]) -> ValueType:
    lhs, rhs = inputs
    output = _elementwise_structure("multiply", lhs, rhs)
    dimension, price_basis = _product_semantics(lhs, rhs)
    return output.with_semantics(dimension, price_basis)


def _divide_type(inputs: tuple[ValueType, ...]) -> ValueType:
    lhs, rhs = inputs
    output = _elementwise_structure("divide", lhs, rhs)
    if rhs.is_scalar and rhs.semantic_dimension in {"dimensionless", "count"}:
        if lhs.semantic_dimension == "count" and rhs.semantic_dimension == "count":
            return output.with_semantics("dimensionless")
        return output.with_semantics(lhs.semantic_dimension, lhs.price_basis)
    if lhs.semantic_dimension == rhs.semantic_dimension:
        if (
            lhs.price_basis is not None
            and rhs.price_basis is not None
            and lhs.price_basis != rhs.price_basis
        ):
            raise _price_basis_mismatch("divide", lhs, rhs)
        return output.with_semantics("dimensionless")
    return output.with_semantics(
        f"derived:{lhs.semantic_dimension}/{rhs.semantic_dimension}"
    )


def _power_type(inputs: tuple[ValueType, ...]) -> ValueType:
    base, exponent = inputs
    _elementwise_structure("power", base, exponent)
    if not exponent.is_scalar or exponent.semantic_dimension != "dimensionless":
        raise _type_error(
            "power", "numeric base and dimensionless scalar exponent", inputs
        )
    return (
        base if exponent.is_scalar else _elementwise_structure("power", base, exponent)
    )


def _unary_numeric(inputs: tuple[ValueType, ...]) -> ValueType:
    _require_numeric("unary operator", inputs[0])
    return inputs[0]


def _dimensionless_unary(inputs: tuple[ValueType, ...]) -> ValueType:
    _require_numeric("unary operator", inputs[0])
    return inputs[0].with_semantics("dimensionless")


def _log_exp_type(inputs: tuple[ValueType, ...]) -> ValueType:
    value = _unary_numeric(inputs)
    if value.semantic_dimension not in {
        "dimensionless",
        "return_decimal",
        "rate_decimal",
    }:
        raise TypedDslError(
            "SEMANTIC_DIMENSION_MISMATCH",
            "log/exp 只接受无量纲小数。",
            details={"actual": value.to_dict()},
        )
    return value.with_semantics("dimensionless")


def _strict_dimensionless_unary(inputs: tuple[ValueType, ...]) -> ValueType:
    value = _unary_numeric(inputs)
    if value.semantic_dimension != "dimensionless":
        raise TypedDslError(
            "SEMANTIC_DIMENSION_MISMATCH",
            "该函数只接受无量纲输入。",
            details={"actual": value.to_dict()},
        )
    return value.with_semantics("dimensionless")


def _reciprocal_type(inputs: tuple[ValueType, ...]) -> ValueType:
    value = _unary_numeric(inputs)
    dimension = (
        "dimensionless"
        if value.semantic_dimension == "dimensionless"
        else f"inverse:{value.semantic_dimension}"
    )
    return value.with_semantics(dimension)


def _sqrt_type(inputs: tuple[ValueType, ...]) -> ValueType:
    value = _unary_numeric(inputs)
    dimension = value.semantic_dimension
    if dimension.startswith("squared:"):
        dimension = dimension[len("squared:") :]
    return value.with_semantics(dimension, value.price_basis)


def _comparison_type(inputs: tuple[ValueType, ...]) -> ValueType:
    lhs, rhs = inputs
    output = _elementwise_structure("comparison", lhs, rhs)
    _compatible_semantics("comparison", lhs, rhs)
    return output.as_mask()


def _logical_binary_type(inputs: tuple[ValueType, ...]) -> ValueType:
    lhs, rhs = inputs
    if not lhs.is_mask or not rhs.is_mask:
        raise _type_error("logical operator", "two masks", inputs)
    require_same_type(lhs, rhs, "logical operator")
    return lhs


def _logical_not_type(inputs: tuple[ValueType, ...]) -> ValueType:
    if not inputs[0].is_mask:
        raise _type_error("logical_not", "mask", inputs)
    return inputs[0]


def _where_type(inputs: tuple[ValueType, ...]) -> ValueType:
    mask, if_true, if_false = inputs
    if not mask.is_mask:
        raise _type_error("where", "mask, numeric, numeric", inputs)
    output = _additive_type((if_true, if_false), operator="where")
    if mask.is_scalar:
        return output
    if output.is_scalar:
        return type_from_axes(
            mask.axes,
            mask.shape,
            semantic_dimension=output.semantic_dimension,
            price_basis=output.price_basis,
        )
    if mask.axes != output.axes or mask.shape != output.shape:
        code = "AXIS_MISMATCH" if mask.axes != output.axes else "SHAPE_MISMATCH"
        raise TypedDslError(code, "where 的 mask 必须与数值输入 shape 一致。")
    return output


def _clip_type(inputs: tuple[ValueType, ...]) -> ValueType:
    value, lower, upper = inputs
    _require_numeric("clip", value, lower, upper)
    if not lower.is_scalar or not upper.is_scalar:
        raise _type_error("clip", "numeric, scalar, scalar", inputs)
    _compatible_semantics("clip", value, lower)
    _compatible_semantics("clip", value, upper)
    return value


def _reduce_all(inputs: tuple[ValueType, ...]) -> ValueType:
    value = inputs[0]
    if value.is_scalar or not value.is_numeric:
        raise _type_error("reduction", "series、vector 或 matrix", inputs)
    return ValueType.scalar(
        semantic_dimension=value.semantic_dimension,
        price_basis=value.price_basis,
    )


def _reduce_all_or_window(inputs: tuple[ValueType, ...]) -> ValueType:
    value = inputs[0]
    if value.kind == "window":
        return ValueType.series(
            value.shape[0],
            semantic_dimension=value.semantic_dimension,
            price_basis=value.price_basis,
        )
    return _reduce_all(inputs)


def _dimensionless_reduction_type(inputs: tuple[ValueType, ...]) -> ValueType:
    _reduce_all(inputs)
    return ValueType.scalar(semantic_dimension="dimensionless")


def _preserve_one_dimensional(inputs: tuple[ValueType, ...]) -> ValueType:
    value_type = inputs[0]
    if value_type.kind not in {"series", "vector"} or not value_type.is_numeric:
        raise _type_error("cumulative operator", "series 或 vector", inputs)
    return value_type


def _path_level_input(inputs: tuple[ValueType, ...], operator: str) -> ValueType:
    value_type = inputs[0]
    if value_type.kind != "series" or not value_type.is_numeric:
        raise _type_error(operator, "时间序列", inputs)
    if value_type.semantic_dimension not in PATH_LEVEL_SEMANTIC_DIMENSIONS:
        raise TypedDslError(
            "SEMANTIC_DIMENSION_MISMATCH",
            f"{operator} 只接受净值、价格或无量纲水平序列，不能直接接受收益率序列。",
            details={"operator": operator, "actual": value_type.to_dict()},
        )
    return value_type


def _drawdown_series_type(inputs: tuple[ValueType, ...]) -> ValueType:
    value_type = _path_level_input(inputs, "drawdown_series")
    return ValueType.series(
        value_type.shape[0],
        semantic_dimension="return_decimal",
    )


def _new_high_mask_type(inputs: tuple[ValueType, ...]) -> ValueType:
    return _path_level_input(inputs, "new_high_mask").as_mask()


def _last_type(inputs: tuple[ValueType, ...]) -> ValueType:
    value = inputs[0]
    if value.kind not in {"series", "vector"} or not value.is_numeric:
        raise _type_error("last", "series 或 vector", inputs)
    return ValueType.scalar(
        semantic_dimension=value.semantic_dimension,
        price_basis=value.price_basis,
    )


def _length_type(inputs: tuple[ValueType, ...]) -> ValueType:
    if inputs[0].kind not in {"series", "vector"}:
        raise _type_error("length", "series 或 vector", inputs)
    return ValueType.scalar(semantic_dimension="count")


def _lag_difference_type(inputs: tuple[ValueType, ...]) -> ValueType:
    value = inputs[0]
    if value.kind not in {"series", "vector"} or not value.is_numeric:
        raise _type_error("lag/difference", "numeric series 或 vector", inputs)
    if len(inputs) == 2:
        periods = inputs[1]
        if (
            not periods.is_scalar
            or not periods.is_numeric
            or periods.semantic_dimension not in {"count", "dimensionless"}
        ):
            raise _type_error(
                "lag/difference", "numeric series/vector and scalar periods", inputs
            )
    dimension = value.shape[0]
    next_dimension: str | int
    if isinstance(dimension, int):
        if dimension < 2:
            raise TypedDslError(
                "INSUFFICIENT_SAMPLE", "lag/difference 至少需要两个值。"
            )
        next_dimension = dimension - 1 if len(inputs) == 1 else f"{dimension}-n"
    else:
        next_dimension = f"{dimension}-1" if len(inputs) == 1 else f"{dimension}-n"
    return type_from_axes(
        value.axes,
        (next_dimension,),
        semantic_dimension=value.semantic_dimension,
        price_basis=value.price_basis,
    )


def _series_parameter(
    value: ValueType,
    *,
    operator: str,
    parameter: str,
    allow_dimensionless: bool = True,
) -> None:
    allowed_dimensions = {"count"}
    if allow_dimensionless:
        allowed_dimensions.add("dimensionless")
    if (
        not value.is_scalar
        or not value.is_numeric
        or value.semantic_dimension not in allowed_dimensions
    ):
        raise _type_error(
            operator,
            f"时间序列及标量参数 {parameter}",
            (value,),
        )


def _rolling_type(inputs: tuple[ValueType, ...]) -> ValueType:
    values = inputs[0]
    if values.kind != "series" or not values.is_numeric:
        raise _type_error("rolling", "numeric series<time>", inputs)
    _series_parameter(inputs[1], operator="rolling", parameter="window")
    if len(inputs) >= 3:
        _series_parameter(inputs[-1], operator="rolling", parameter="min_periods")
    return values


def _rolling_window_type(inputs: tuple[ValueType, ...]) -> ValueType:
    values = _rolling_type(inputs)
    return ValueType.window(
        values.shape[0],
        "W",
        semantic_dimension=values.semantic_dimension,
        price_basis=values.price_basis,
    )


def _rolling_std_type(inputs: tuple[ValueType, ...]) -> ValueType:
    values = _rolling_type(inputs)
    if len(inputs) >= 3:
        ddof_index = 2
        _series_parameter(inputs[ddof_index], operator="rolling_std", parameter="ddof")
    return values


def _recursive_smooth_type(inputs: tuple[ValueType, ...]) -> ValueType:
    values, periods, initial = inputs
    if values.kind != "series" or not values.is_numeric:
        raise _type_error("recursive_smooth", "numeric series<time>", inputs)
    _series_parameter(periods, operator="recursive_smooth", parameter="periods")
    if not initial.is_scalar or not initial.is_numeric:
        raise _type_error(
            "recursive_smooth",
            "numeric series<time>, scalar periods, scalar initial",
            inputs,
        )
    _compatible_semantics("recursive_smooth", values, initial)
    return values


def _divide_or_default_type(inputs: tuple[ValueType, ...]) -> ValueType:
    numerator, denominator, default = inputs
    if (
        numerator.kind != "series"
        or denominator.kind != "series"
        or not numerator.is_numeric
        or not denominator.is_numeric
    ):
        raise _type_error(
            "divide_or_default",
            "two numeric series<time> and one scalar default",
            inputs,
        )
    output = _divide_type((numerator, denominator))
    if not default.is_scalar or not default.is_numeric:
        raise _type_error(
            "divide_or_default",
            "two numeric series<time> and one scalar default",
            inputs,
        )
    _compatible_semantics("divide_or_default", output, default)
    return output


def _validated_positive_integer(value: Any, name: str, *, allow_zero: bool = False) -> int:
    number = float(value)
    minimum = 0 if allow_zero else 1
    if not math.isfinite(number) or not number.is_integer() or number < minimum:
        raise TypedDslError(
            "INVALID_PARAMETER",
            f"{name} 必须是{'非负' if allow_zero else '正'}整数。",
            details={"parameter": name, "value": number},
        )
    return int(number)


@dataclass(frozen=True)
class _RollingWindowReference:
    """Test-only logical window reference; production never materializes it."""

    values: np.ndarray
    window: int
    min_periods: int

    def __array__(self, dtype: Any = None) -> np.ndarray:
        output = np.full((self.values.size, self.window), np.nan, dtype=np.float64)
        for index in range(self.values.size):
            start = max(0, index - self.window + 1)
            selected = self.values[start : index + 1]
            output[index, self.window - selected.size :] = selected
        return output.astype(dtype, copy=False) if dtype is not None else output


def _rolling_reference(
    values: Any,
    window: Any,
    min_periods: Any,
    *,
    mode: str,
    ddof: Any = 0.0,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    width = _validated_positive_integer(window, "window")
    minimum = _validated_positive_integer(min_periods, "min_periods")
    if minimum > width:
        raise TypedDslError(
            "INVALID_PARAMETER",
            "min_periods 不能大于 window。",
        )
    degrees = _validated_positive_integer(ddof, "ddof", allow_zero=True)
    output = np.full(array.size, np.nan, dtype=np.float64)
    for index in range(array.size):
        start = max(0, index - width + 1)
        selected = array[start : index + 1]
        selected = selected[np.isfinite(selected)]
        if selected.size < minimum:
            continue
        if mode == "sum":
            output[index] = float(np.sum(selected))
        elif mode == "product":
            output[index] = float(np.prod(selected))
        elif mode == "mean":
            output[index] = float(np.mean(selected))
        elif mode in {"std", "variance"}:
            if selected.size <= degrees:
                continue
            variance = float(np.var(selected, ddof=degrees))
            output[index] = math.sqrt(variance) if mode == "std" else variance
        elif mode == "min":
            output[index] = float(np.min(selected))
        else:
            output[index] = float(np.max(selected))
    return output


def _rolling_window(values: Any, window: Any, min_periods: Any | None = None) -> _RollingWindowReference:
    array = np.asarray(values, dtype=np.float64)
    width = _validated_positive_integer(window, "window")
    minimum = width if min_periods is None else _validated_positive_integer(min_periods, "min_periods")
    if minimum > width:
        raise TypedDslError("INVALID_PARAMETER", "min_periods 不能大于 window。")
    return _RollingWindowReference(array, width, minimum)


def _window_reduction_reference(values: Any, mode: str) -> Any:
    if not isinstance(values, _RollingWindowReference):
        return None
    return _rolling_reference(
        values.values,
        values.window,
        values.min_periods,
        mode=mode,
    )


def _reduce_mean_reference(values: Any) -> Any:
    rolling = _window_reduction_reference(values, "mean")
    return reduce_mean(values) if rolling is None else rolling


def _reduce_min_reference(values: Any) -> Any:
    rolling = _window_reduction_reference(values, "min")
    return reduce_min(values) if rolling is None else rolling


def _reduce_max_reference(values: Any) -> Any:
    rolling = _window_reduction_reference(values, "max")
    return reduce_max(values) if rolling is None else rolling


def _rolling_mean(values: Any, window: Any, min_periods: Any | None = None) -> np.ndarray:
    effective = window if min_periods is None else min_periods
    return _rolling_reference(values, window, effective, mode="mean")


def _rolling_std(
    values: Any,
    window: Any,
    ddof: Any = 0.0,
    min_periods: Any | None = None,
) -> np.ndarray:
    effective = window if min_periods is None else min_periods
    return _rolling_reference(values, window, effective, mode="std", ddof=ddof)


def _rolling_min(values: Any, window: Any, min_periods: Any | None = None) -> np.ndarray:
    effective = window if min_periods is None else min_periods
    return _rolling_reference(values, window, effective, mode="min")


def _rolling_max(values: Any, window: Any, min_periods: Any | None = None) -> np.ndarray:
    effective = window if min_periods is None else min_periods
    return _rolling_reference(values, window, effective, mode="max")


def _recursive_smooth(values: Any, periods: Any, initial: Any) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    width = _validated_positive_integer(periods, "periods")
    previous = float(initial)
    if not math.isfinite(previous):
        raise TypedDslError("INVALID_PARAMETER", "initial 必须是有限数值。")
    output = np.full(array.size, np.nan, dtype=np.float64)
    for index, value in enumerate(array):
        if not np.isfinite(value):
            continue
        previous = ((width - 1.0) * previous + value) / width
        output[index] = previous
    return output


def _divide_or_default(numerator: Any, denominator: Any, default: Any) -> Any:
    lhs = np.asarray(numerator, dtype=np.float64)
    rhs = np.asarray(denominator, dtype=np.float64)
    fallback = float(default)
    return np.where(np.abs(rhs) < 1e-12, fallback, lhs / rhs)


def _arg_type(inputs: tuple[ValueType, ...]) -> ValueType:
    _reduce_all(inputs)
    return ValueType.scalar(semantic_dimension="count")


def _max_consecutive_type(inputs: tuple[ValueType, ...]) -> ValueType:
    value = inputs[0]
    if not value.is_mask or value.kind not in {"series", "vector"}:
        raise _type_error("max_consecutive_true", "one-dimensional mask", inputs)
    return ValueType.scalar(semantic_dimension="count")


def _reduce_time(inputs: tuple[ValueType, ...]) -> ValueType:
    matrix = inputs[0]
    if matrix.kind != "matrix" or matrix.axes != ("time", "asset"):
        raise _type_error("time reduction", "matrix<time,asset>[T,N]", inputs)
    return ValueType.vector(
        matrix.shape[1],
        semantic_dimension=matrix.semantic_dimension,
        price_basis=matrix.price_basis,
    )


def _reduce_asset(inputs: tuple[ValueType, ...]) -> ValueType:
    matrix = inputs[0]
    if matrix.kind != "matrix" or matrix.axes != ("time", "asset"):
        raise _type_error("asset reduction", "matrix<time,asset>[T,N]", inputs)
    return ValueType.series(
        matrix.shape[0],
        semantic_dimension=matrix.semantic_dimension,
        price_basis=matrix.price_basis,
    )


def _variance_time_type(inputs: tuple[ValueType, ...]) -> ValueType:
    output = _reduce_time(inputs)
    return output.with_semantics(f"squared:{output.semantic_dimension}")


def _variance_asset_type(inputs: tuple[ValueType, ...]) -> ValueType:
    output = _reduce_asset(inputs)
    return output.with_semantics(f"squared:{output.semantic_dimension}")


def _transpose_type(inputs: tuple[ValueType, ...]) -> ValueType:
    matrix = inputs[0]
    if matrix.kind != "matrix":
        raise _type_error("transpose", "matrix", inputs)
    return ValueType.matrix(
        (matrix.axes[1], matrix.axes[0]),
        (matrix.shape[1], matrix.shape[0]),
        semantic_dimension=matrix.semantic_dimension,
        price_basis=matrix.price_basis,
    )


def _dot_type(inputs: tuple[ValueType, ...]) -> ValueType:
    lhs, rhs = inputs
    if lhs.kind not in {"series", "vector"}:
        raise _type_error("dot", "两个同类型的一维输入", inputs)
    require_same_type(lhs, rhs, "dot")
    _require_numeric("dot", lhs, rhs)
    dimension, price_basis = _product_semantics(lhs, rhs)
    return ValueType.scalar(semantic_dimension=dimension, price_basis=price_basis)


def _outer_type(inputs: tuple[ValueType, ...]) -> ValueType:
    lhs, rhs = inputs
    if lhs.kind != "vector" or rhs.kind != "vector":
        raise _type_error("outer", "vector<asset>, vector<asset>", inputs)
    return ValueType.matrix(
        ("asset", "asset"),
        (lhs.shape[0], rhs.shape[0]),
        semantic_dimension=_product_semantics(lhs, rhs)[0],
    )


def _matmul_type(inputs: tuple[ValueType, ...]) -> ValueType:
    lhs, rhs = inputs
    if lhs.kind != "matrix" or rhs.kind != "matrix":
        raise _type_error("matmul", "matrix, matrix", inputs)
    if lhs.axes[1] != rhs.axes[0]:
        raise TypedDslError(
            "AXIS_MISMATCH",
            f"matmul 内轴不一致: {lhs.axes[1]} 与 {rhs.axes[0]}",
            details={"left": lhs.to_dict(), "right": rhs.to_dict()},
        )
    if lhs.shape[1] != rhs.shape[0]:
        raise TypedDslError(
            "SHAPE_MISMATCH",
            f"matmul 内维不一致: {lhs.shape[1]} 与 {rhs.shape[0]}",
            details={"left": lhs.to_dict(), "right": rhs.to_dict()},
        )
    return ValueType.matrix(
        (lhs.axes[0], rhs.axes[1]),
        (lhs.shape[0], rhs.shape[1]),
        semantic_dimension=_product_semantics(lhs, rhs)[0],
    )


def _matvec_type(inputs: tuple[ValueType, ...]) -> ValueType:
    matrix, vector = inputs
    if matrix.kind != "matrix" or vector.rank != 1:
        raise _type_error("matvec", "matrix 与匹配内轴的一维输入", inputs)
    if matrix.axes[1] != vector.axes[0]:
        raise TypedDslError("AXIS_MISMATCH", "matvec 的矩阵内轴与向量轴不一致。")
    if matrix.shape[1] != vector.shape[0]:
        raise TypedDslError("SHAPE_MISMATCH", "matvec 的矩阵内维与向量长度不一致。")
    dimension, price_basis = _product_semantics(matrix, vector)
    return type_from_axes(
        (matrix.axes[0],),
        (matrix.shape[0],),
        semantic_dimension=dimension,
        price_basis=price_basis,
    )


def _diag_type(inputs: tuple[ValueType, ...]) -> ValueType:
    value_type = inputs[0]
    if value_type.kind == "vector":
        length = value_type.shape[0]
        return ValueType.matrix(
            ("asset", "asset"),
            (length, length),
            semantic_dimension=value_type.semantic_dimension,
            price_basis=value_type.price_basis,
        )
    if value_type.kind == "matrix" and value_type.axes == ("asset", "asset"):
        if value_type.shape[0] != value_type.shape[1]:
            raise TypedDslError("SHAPE_MISMATCH", "diag 要求方阵。")
        return ValueType.vector(
            value_type.shape[0],
            semantic_dimension=value_type.semantic_dimension,
            price_basis=value_type.price_basis,
        )
    raise _type_error("diag", "vector<asset> 或 matrix<asset,asset>", inputs)


def _trace_type(inputs: tuple[ValueType, ...]) -> ValueType:
    matrix = inputs[0]
    if matrix.kind != "matrix" or matrix.axes[0] != matrix.axes[1]:
        raise _type_error("trace", "同名双轴方阵", inputs)
    if matrix.shape[0] != matrix.shape[1]:
        raise TypedDslError("SHAPE_MISMATCH", "trace 要求方阵。")
    return ValueType.scalar(
        semantic_dimension=matrix.semantic_dimension,
        price_basis=matrix.price_basis,
    )


def _solve_type(inputs: tuple[ValueType, ...]) -> ValueType:
    matrix, vector = inputs
    if matrix.kind != "matrix" or matrix.axes != ("asset", "asset"):
        raise _type_error("solve", "matrix<asset,asset>, vector<asset>", inputs)
    if vector.kind != "vector":
        raise _type_error("solve", "matrix<asset,asset>, vector<asset>", inputs)
    if matrix.shape[0] != matrix.shape[1] or matrix.shape[1] != vector.shape[0]:
        raise TypedDslError("SHAPE_MISMATCH", "solve 的方阵维度必须与向量长度一致。")
    return vector.with_semantics(
        f"derived:{vector.semantic_dimension}/{matrix.semantic_dimension}"
    )


def _covariance_type(inputs: tuple[ValueType, ...]) -> ValueType:
    if len(inputs) == 1:
        matrix = inputs[0]
        if matrix.kind != "matrix" or matrix.axes != ("time", "asset"):
            raise _type_error("covariance", "matrix<time,asset>[T,N]", inputs)
        length = matrix.shape[1]
        return ValueType.matrix(
            ("asset", "asset"),
            (length, length),
            semantic_dimension=f"squared:{matrix.semantic_dimension}",
        )
    lhs, rhs = inputs
    if lhs.kind != "series" or rhs.kind != "series":
        raise _type_error("covariance", "两个 series<time>[T]", inputs)
    require_same_type(lhs, rhs, "covariance")
    return ValueType.scalar(
        semantic_dimension=(
            f"squared:{lhs.semantic_dimension}"
            if lhs.semantic_dimension == rhs.semantic_dimension
            else f"derived:{lhs.semantic_dimension}*{rhs.semantic_dimension}"
        )
    )


def _correlation_type(inputs: tuple[ValueType, ...]) -> ValueType:
    covariance_type = _covariance_type(inputs)
    return covariance_type.with_semantics("dimensionless")


def _portfolio_returns_type(inputs: tuple[ValueType, ...]) -> ValueType:
    matrix, weights = inputs
    if matrix.kind != "matrix" or matrix.axes != ("time", "asset"):
        raise _type_error(
            "portfolio_returns",
            "matrix<time,asset>[T,N], vector<asset>[N]",
            inputs,
        )
    if weights.kind != "vector":
        raise _type_error(
            "portfolio_returns",
            "matrix<time,asset>[T,N], vector<asset>[N]",
            inputs,
        )
    if matrix.shape[1] != weights.shape[0]:
        raise TypedDslError("SHAPE_MISMATCH", "收益矩阵资产数与权重长度不一致。")
    if weights.semantic_dimension != "dimensionless":
        raise _semantic_mismatch("portfolio_returns", matrix, weights)
    return ValueType.series(
        matrix.shape[0],
        semantic_dimension=matrix.semantic_dimension,
        price_basis=matrix.price_basis,
    )


def _quadratic_form_type(inputs: tuple[ValueType, ...]) -> ValueType:
    weights, matrix = inputs
    if weights.kind != "vector" or matrix.kind != "matrix":
        raise _type_error(
            "quadratic_form",
            "vector<asset>[N], matrix<asset,asset>[N,N]",
            inputs,
        )
    if matrix.axes != ("asset", "asset"):
        raise _type_error(
            "quadratic_form",
            "vector<asset>[N], matrix<asset,asset>[N,N]",
            inputs,
        )
    if matrix.shape != (weights.shape[0], weights.shape[0]):
        raise TypedDslError("SHAPE_MISMATCH", "二次型矩阵维度必须与权重长度一致。")
    dimension, price_basis = _product_semantics(weights, matrix)
    return ValueType.scalar(
        semantic_dimension=dimension,
        price_basis=price_basis,
    )


def _active_returns_type(inputs: tuple[ValueType, ...]) -> ValueType:
    lhs, rhs = inputs
    if lhs.kind != "series" or rhs.kind != "series":
        raise _type_error("active_returns", "两个 series<time>[T]", inputs)
    return require_same_type(lhs, rhs, "active_returns")


def _annualized_return_type(inputs: tuple[ValueType, ...]) -> ValueType:
    values, periods_per_year = inputs
    if values.kind != "series" or not periods_per_year.is_scalar:
        raise _type_error(
            "annualized_return",
            "series<time>[T], scalar",
            inputs,
        )
    if periods_per_year.semantic_dimension not in {"count", "dimensionless"}:
        raise _semantic_mismatch("annualized_return", values, periods_per_year)
    return ValueType.scalar(semantic_dimension="return_decimal")


def _validate_variance_inputs(inputs: tuple[ValueType, ...]) -> ValueType:
    values = inputs[0]
    if values.kind != "window" and (values.is_scalar or not values.is_numeric):
        raise _type_error("variance/std", "series、vector、matrix 或滚动窗口", inputs)
    if len(inputs) == 2 and (
        not inputs[1].is_scalar
        or not inputs[1].is_numeric
        or inputs[1].semantic_dimension not in {"count", "dimensionless"}
    ):
        raise _type_error("variance/std", "numeric tensor, scalar ddof", inputs)
    return values


def _variance_type(inputs: tuple[ValueType, ...]) -> ValueType:
    values = _validate_variance_inputs(inputs)
    if values.kind == "window":
        return ValueType.series(
            values.shape[0],
            semantic_dimension=f"squared:{values.semantic_dimension}",
        )
    return ValueType.scalar(semantic_dimension=f"squared:{values.semantic_dimension}")


def _std_type(inputs: tuple[ValueType, ...]) -> ValueType:
    values = _validate_variance_inputs(inputs)
    if values.kind == "window":
        return ValueType.series(
            values.shape[0],
            semantic_dimension=values.semantic_dimension,
            price_basis=values.price_basis,
        )
    return ValueType.scalar(
        semantic_dimension=values.semantic_dimension,
        price_basis=values.price_basis,
    )


def _masked_reduction_type(inputs: tuple[ValueType, ...]) -> ValueType:
    values, mask = inputs[:2]
    if values.is_scalar or not values.is_numeric:
        raise _type_error(
            "masked reduction", "numeric tensor and matching mask", inputs
        )
    if not mask.is_mask:
        raise _type_error(
            "masked reduction", "numeric tensor and matching mask", inputs
        )
    if values.axes != mask.axes or values.shape != mask.shape:
        code = "AXIS_MISMATCH" if values.axes != mask.axes else "SHAPE_MISMATCH"
        raise TypedDslError(code, "masked reduction 的 mask 必须与 values 对齐。")
    if len(inputs) == 3 and (
        not inputs[2].is_scalar or inputs[2].semantic_dimension != "dimensionless"
    ):
        raise _type_error("masked quantile", "values, mask, scalar probability", inputs)
    return ValueType.scalar(
        semantic_dimension=values.semantic_dimension,
        price_basis=values.price_basis,
    )


def _masked_variance_type(inputs: tuple[ValueType, ...]) -> ValueType:
    output = _masked_reduction_type(inputs)
    return output.with_semantics(f"squared:{output.semantic_dimension}")


def _masked_count_type(inputs: tuple[ValueType, ...]) -> ValueType:
    if not inputs[0].is_mask or inputs[0].is_scalar:
        raise _type_error("masked_count", "non-scalar mask", inputs)
    return ValueType.scalar(semantic_dimension="count")


def _quantile_type(inputs: tuple[ValueType, ...]) -> ValueType:
    values, probability = inputs
    if values.is_scalar or not values.is_numeric:
        raise _type_error("quantile", "numeric tensor, scalar probability", inputs)
    if not probability.is_scalar or probability.semantic_dimension != "dimensionless":
        raise _type_error("quantile", "numeric tensor, scalar probability", inputs)
    return ValueType.scalar(
        semantic_dimension=values.semantic_dimension,
        price_basis=values.price_basis,
    )


def _regression_type(inputs: tuple[ValueType, ...]) -> ValueType:
    if len(inputs) == 1:
        values = inputs[0]
        if values.kind != "series" or not values.is_numeric:
            raise _type_error("linear regression", "numeric series", inputs)
        return ValueType.scalar(semantic_dimension=values.semantic_dimension)
    lhs, rhs = inputs
    if lhs.kind != "series" or rhs.kind != "series":
        raise _type_error("linear regression", "two numeric series", inputs)
    _require_numeric("linear regression", lhs, rhs)
    require_same_type(lhs, rhs, "linear regression")
    dimension = (
        "dimensionless"
        if lhs.semantic_dimension == rhs.semantic_dimension
        else f"derived:{rhs.semantic_dimension}/{lhs.semantic_dimension}"
    )
    return ValueType.scalar(semantic_dimension=dimension)


def _regression_intercept_type(inputs: tuple[ValueType, ...]) -> ValueType:
    if len(inputs) == 1:
        values = inputs[0]
        _regression_type(inputs)
        return ValueType.scalar(
            semantic_dimension=values.semantic_dimension,
            price_basis=values.price_basis,
        )
    _regression_type(inputs)
    return ValueType.scalar(
        semantic_dimension=inputs[1].semantic_dimension,
        price_basis=inputs[1].price_basis,
    )


def _r_squared_type(inputs: tuple[ValueType, ...]) -> ValueType:
    _regression_type(inputs)
    return ValueType.scalar(semantic_dimension="dimensionless")


def _cost_input(inputs: tuple[ValueType, ...], _output: ValueType) -> str:
    return symbolic_elements(inputs[0])


def _cost_all_inputs(inputs: tuple[ValueType, ...], output: ValueType) -> str:
    parts = [symbolic_elements(item) for item in inputs]
    parts.append(symbolic_elements(output))
    return "+".join(parts)


def _cost_covariance(inputs: tuple[ValueType, ...], _output: ValueType) -> str:
    if len(inputs) == 2:
        return symbolic_elements(inputs[0])
    matrix = inputs[0]
    return f"{matrix.shape[0]}*{matrix.shape[1]}^2"


def _cost_matmul(inputs: tuple[ValueType, ...], _output: ValueType) -> str:
    lhs, rhs = inputs
    return f"{lhs.shape[0]}*{lhs.shape[1]}*{rhs.shape[1]}"


def _cost_solve(inputs: tuple[ValueType, ...], _output: ValueType) -> str:
    size = inputs[0].shape[0]
    return f"{size}^3"


def _divide(lhs: Any, rhs: Any) -> Any:
    rhs_array = np.asarray(rhs, dtype=np.float64)
    if np.any(np.abs(rhs_array) < 1e-12):
        raise TypedDslError("DIVIDE_BY_ZERO", "除数包含零或接近零的值。")
    return np.asarray(lhs, dtype=np.float64) / rhs_array


def _sqrt(value: Any) -> Any:
    value_array = np.asarray(value, dtype=np.float64)
    if np.any(value_array < 0.0):
        raise TypedDslError("DOMAIN_ERROR", "sqrt 的输入不能为负数。")
    return np.sqrt(value_array)


def _clip(value: Any, lower: Any, upper: Any) -> Any:
    lower_float = float(lower)
    upper_float = float(upper)
    if lower_float > upper_float:
        raise TypedDslError("DOMAIN_ERROR", "clip 的 lower 不能大于 upper。")
    return np.clip(value, lower_float, upper_float)


def _log(value: Any) -> Any:
    array = np.asarray(value, dtype=np.float64)
    if np.any(array <= 0.0):
        raise TypedDslError("DOMAIN_ERROR", "log 的输入必须严格大于零。")
    return np.log(array)


def _reciprocal(value: Any) -> Any:
    return _divide(1.0, value)


def _first(value: Any) -> Any:
    return np.asarray(value, dtype=np.float64)[0]


def _length(value: Any) -> float:
    return float(np.asarray(value).size)


def _validated_periods(value: Any, *, allow_zero: bool) -> int:
    periods = float(value)
    minimum = 0 if allow_zero else 1
    if not math.isfinite(periods) or not periods.is_integer() or periods < minimum:
        qualifier = "非负" if allow_zero else "正"
        raise TypedDslError(
            "INVALID_PARAMETER",
            f"periods 必须是有限的{qualifier}整数。",
            details={"parameter": "periods", "value": periods},
        )
    return int(periods)


def _lag(value: Any, periods: Any = 1.0) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    count = _validated_periods(periods, allow_zero=True)
    if count == 0:
        return array.copy()
    if array.size <= count:
        raise TypedDslError(
            "INSUFFICIENT_SAMPLE", "lag 的 periods 必须小于观察值数量。"
        )
    return array[:-count]


def _difference(value: Any, periods: Any = 1.0) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    count = _validated_periods(periods, allow_zero=False)
    if array.size <= count:
        raise TypedDslError(
            "INSUFFICIENT_SAMPLE", "difference 的 periods 必须小于观察值数量。"
        )
    return array[count:] - array[:-count]


def _probability(value: Any) -> float:
    probability = float(value)
    if not math.isfinite(probability) or not 0.0 < probability < 1.0:
        raise TypedDslError(
            "INVALID_PARAMETER",
            "probability 必须位于开区间 (0, 1)。",
            details={"parameter": "probability", "value": probability},
        )
    return probability


def _quantile(values: Any, probability: Any) -> float:
    return float(
        np.quantile(np.asarray(values, dtype=np.float64), _probability(probability))
    )


def _skewness(values: Any) -> float:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size < 3:
        raise TypedDslError("INSUFFICIENT_SAMPLE", "skewness 至少需要三个观察值。")
    centered = array - np.mean(array)
    second = float(np.mean(centered**2))
    if second <= 0.0:
        raise TypedDslError("DOMAIN_ERROR", "skewness 要求输入具有正方差。")
    third = float(np.mean(centered**3))
    n = array.size
    return float(math.sqrt(n * (n - 1)) / (n - 2) * third / second**1.5)


def _excess_kurtosis(values: Any) -> float:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size < 4:
        raise TypedDslError(
            "INSUFFICIENT_SAMPLE", "excess_kurtosis 至少需要四个观察值。"
        )
    centered = array - np.mean(array)
    second = float(np.mean(centered**2))
    if second <= 0.0:
        raise TypedDslError("DOMAIN_ERROR", "excess_kurtosis 要求输入具有正方差。")
    fourth = float(np.mean(centered**4))
    n = array.size
    population_excess = fourth / second**2 - 3.0
    return float(((n - 1) / ((n - 2) * (n - 3))) * ((n + 1) * population_excess + 6.0))


def _mean_absolute_deviation(values: Any) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.mean(np.abs(array - np.mean(array))))


def _root_mean_square(values: Any) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(array))))


def _max_consecutive_true(mask: Any) -> float:
    array = np.asarray(mask, dtype=np.bool_).reshape(-1)
    longest = current = 0
    for value in array:
        current = current + 1 if bool(value) else 0
        longest = max(longest, current)
    return float(longest)


def _selected(values: Any, mask: Any) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    boolean_mask = np.asarray(mask, dtype=np.bool_)
    if array.shape != boolean_mask.shape:
        raise TypedDslError("RUNTIME_SHAPE_MISMATCH", "mask 与 values shape 不一致。")
    selected = array[boolean_mask]
    if selected.size == 0:
        raise TypedDslError("INSUFFICIENT_SAMPLE", "mask 没有选中任何观察值。")
    return selected


def _masked_sum(values: Any, mask: Any) -> float:
    return float(np.sum(_selected(values, mask)))


def _masked_mean(values: Any, mask: Any) -> float:
    return float(np.mean(_selected(values, mask)))


def _masked_variance(values: Any, mask: Any) -> float:
    selected = _selected(values, mask)
    if selected.size < 2:
        raise TypedDslError(
            "INSUFFICIENT_SAMPLE", "masked_variance 至少需要两个选中观察值。"
        )
    return float(np.var(selected, ddof=1))


def _masked_std(values: Any, mask: Any) -> float:
    selected = _selected(values, mask)
    if selected.size < 2:
        raise TypedDslError(
            "INSUFFICIENT_SAMPLE", "masked_std 至少需要两个选中观察值。"
        )
    return float(np.std(selected, ddof=1))


def _masked_quantile(values: Any, mask: Any, probability: Any) -> float:
    return float(np.quantile(_selected(values, mask), _probability(probability)))


def _linear_fit(*values: Any) -> tuple[float, float, float]:
    if len(values) == 1:
        y = np.asarray(values[0], dtype=np.float64).reshape(-1)
        x = np.arange(y.size, dtype=np.float64)
    else:
        x = np.asarray(values[0], dtype=np.float64).reshape(-1)
        y = np.asarray(values[1], dtype=np.float64).reshape(-1)
    if x.size != y.size:
        raise TypedDslError("RUNTIME_SHAPE_MISMATCH", "线性回归的 x 与 y 长度不一致。")
    if x.size < 2:
        raise TypedDslError("INSUFFICIENT_SAMPLE", "线性回归至少需要两个观察值。")
    centered_x = x - np.mean(x)
    denominator = float(np.dot(centered_x, centered_x))
    if denominator <= 0.0:
        raise TypedDslError("DOMAIN_ERROR", "线性回归的 x 必须具有正方差。")
    centered_y = y - np.mean(y)
    slope = float(np.dot(centered_x, centered_y) / denominator)
    intercept = float(np.mean(y) - slope * np.mean(x))
    fitted = intercept + slope * x
    total = float(np.dot(centered_y, centered_y))
    residual = float(np.dot(y - fitted, y - fitted))
    r_squared = float("nan") if total <= 0.0 else float(1.0 - residual / total)
    return slope, intercept, r_squared


def _linear_slope(*values: Any) -> float:
    return _linear_fit(*values)[0]


def _linear_intercept(*values: Any) -> float:
    return _linear_fit(*values)[1]


def _linear_r_squared(*values: Any) -> float:
    r_squared = _linear_fit(*values)[2]
    if not math.isfinite(r_squared):
        raise TypedDslError("DOMAIN_ERROR", "linear_r_squared 要求 y 具有正方差。")
    return r_squared


def _regression_standard_error(*values: Any) -> float:
    if len(values) == 1:
        y = np.asarray(values[0], dtype=np.float64).reshape(-1)
        x = np.arange(y.size, dtype=np.float64)
    else:
        x = np.asarray(values[0], dtype=np.float64).reshape(-1)
        y = np.asarray(values[1], dtype=np.float64).reshape(-1)
    if y.size < 3:
        raise TypedDslError(
            "INSUFFICIENT_SAMPLE", "regression_standard_error 至少需要三个观察值。"
        )
    slope, intercept, _ = _linear_fit(x, y)
    residual = y - (intercept + slope * x)
    return float(np.sqrt(np.dot(residual, residual) / (y.size - 2)))


def _normal_pdf(value: Any) -> Any:
    array = np.asarray(value, dtype=np.float64)
    return np.exp(-0.5 * array**2) / math.sqrt(2.0 * math.pi)


def _normal_ppf(value: Any) -> Any:
    array = np.asarray(value, dtype=np.float64)
    if np.any((array <= 0.0) | (array >= 1.0)):
        raise TypedDslError("DOMAIN_ERROR", "normal_ppf 的输入必须位于开区间 (0, 1)。")
    inverse = np.vectorize(NormalDist().inv_cdf, otypes=[np.float64])
    return inverse(array)


def _cumulative_return(values: Any) -> np.ndarray:
    return np.cumprod(1.0 + np.asarray(values, dtype=np.float64)) - 1.0


def _validated_level_series(values: Any, operator: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        raise TypedDslError("INSUFFICIENT_SAMPLE", f"{operator} 至少需要一个观察值。")
    if not np.all(np.isfinite(array)) or np.any(array <= 0.0):
        raise TypedDslError(
            "DOMAIN_ERROR",
            f"{operator} 要求所有水平值均为有限正数。",
        )
    return array


def _drawdown_series(values: Any) -> np.ndarray:
    array = _validated_level_series(values, "drawdown_series")
    return array / np.maximum.accumulate(array) - 1.0


def _new_high_mask(values: Any) -> np.ndarray:
    array = _validated_level_series(values, "new_high_mask")
    result = np.zeros(array.size, dtype=np.bool_)
    result[0] = True
    running_peak = array[0]
    for index in range(1, array.size):
        if array[index] > running_peak:
            result[index] = True
            running_peak = array[index]
    return result


def _total_return(values: Any) -> float:
    return float(np.prod(1.0 + np.asarray(values, dtype=np.float64)) - 1.0)


def _annualized_return(values: Any, periods_per_year: Any) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise TypedDslError("INSUFFICIENT_SAMPLE", "年化收益率至少需要一个观察值。")
    growth = float(np.prod(1.0 + array))
    if growth < 0.0:
        raise TypedDslError("DOMAIN_ERROR", "累计增长因子为负，无法计算年化收益率。")
    return growth ** (float(periods_per_year) / array.size) - 1.0


def _validated_ddof(ddof: Any, observation_count: int) -> int:
    value = float(ddof)
    if not math.isfinite(value) or not value.is_integer() or value < 0:
        raise TypedDslError(
            "INVALID_PARAMETER",
            "ddof 必须是有限的非负整数。",
            details={"parameter": "ddof", "value": value},
        )
    integer = int(value)
    if integer >= observation_count:
        raise TypedDslError(
            "INVALID_PARAMETER",
            "ddof 必须小于观察值数量。",
            details={
                "parameter": "ddof",
                "value": integer,
                "observations": observation_count,
            },
        )
    return integer


def _variance(values: Any, ddof: Any = 1.0) -> Any:
    if isinstance(values, _RollingWindowReference):
        return _rolling_reference(
            values.values,
            values.window,
            values.min_periods,
            mode="variance",
            ddof=ddof,
        )
    array = np.asarray(values, dtype=np.float64)
    return reduce_variance(array, _validated_ddof(ddof, array.size))


def _std(values: Any, ddof: Any = 1.0) -> Any:
    if isinstance(values, _RollingWindowReference):
        return _rolling_reference(
            values.values,
            values.window,
            values.min_periods,
            mode="std",
            ddof=ddof,
        )
    array = np.asarray(values, dtype=np.float64)
    return reduce_std(array, _validated_ddof(ddof, array.size))


def _legacy_numpy_variance(values: Any, ddof: Any = 1.0) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.var(array, ddof=_validated_ddof(ddof, array.size)))


def _legacy_numpy_std(values: Any, ddof: Any = 1.0) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.std(array, ddof=_validated_ddof(ddof, array.size)))


def _covariance(*values: Any) -> Any:
    if len(values) == 1:
        matrix = np.asarray(values[0], dtype=np.float64)
        return np.atleast_2d(np.cov(matrix, rowvar=False, ddof=1))
    lhs, rhs = (np.asarray(item, dtype=np.float64) for item in values)
    return float(np.cov(lhs, rhs, ddof=1)[0, 1])


def _correlation(*values: Any) -> Any:
    if len(values) == 1:
        matrix = np.asarray(values[0], dtype=np.float64)
        return np.atleast_2d(np.corrcoef(matrix, rowvar=False))
    lhs, rhs = (np.asarray(item, dtype=np.float64) for item in values)
    return float(np.corrcoef(lhs, rhs)[0, 1])


def _quadratic_form(weights: Any, matrix: Any) -> float:
    weights_array = np.asarray(weights, dtype=np.float64)
    matrix_array = np.asarray(matrix, dtype=np.float64)
    return float(weights_array @ matrix_array @ weights_array)


def _signature(
    inputs: Sequence[str], output: str, shape_rule: str
) -> OperatorSignature:
    return OperatorSignature(tuple(inputs), output, shape_rule)


def _spec(
    operator_id: str,
    category: str,
    signatures: Sequence[OperatorSignature],
    description: str,
    infer: InferFunction,
    evaluate: RuntimeFunction,
    *,
    aliases: Sequence[str] = (),
    latex_template: str = "",
    cost_model: str = "elementwise",
    cost: CostFunction | None = None,
    interval_policy: str = "local",
) -> TypedOperatorSpec:
    return TypedOperatorSpec(
        operator_id=operator_id,
        version=TYPED_OPERATOR_REGISTRY_VERSION,
        category=category,
        signatures=tuple(signatures),
        description=description,
        infer=infer,
        evaluate=evaluate,
        aliases=tuple(aliases),
        latex_template=latex_template,
        cost_model=cost_model,
        cost=cost,
        interval_policy=interval_policy,
    )


@lru_cache(maxsize=1)
def _canonical_specs() -> tuple[TypedOperatorSpec, ...]:
    any_numeric = "scalar | series<T> | vector<N> | matrix<A,B>"
    reducible_numeric = "series<T> | vector<N> | matrix<A,B>"
    same_numeric = f"same({any_numeric})"
    reduction = _signature((reducible_numeric,), "scalar", "reduce all named axes")
    matrix_ta = "matrix<time,asset>[T,N]"
    vector_n = "vector<asset>[N]"
    series_t = "series<time>[T]"

    specs: list[TypedOperatorSpec] = []
    for operator_id, function, aliases, latex in (
        ("add", np.add, (), r"x+y"),
        ("subtract", np.subtract, ("sub",), r"x-y"),
        ("multiply", np.multiply, ("mul",), r"x\cdot y"),
        ("divide", _divide, ("safe_divide",), r"\frac{x}{y}"),
        ("power", np.power, (), r"x^{y}"),
        ("minimum", np.minimum, ("elementwise_min",), r"\operatorname{minimum}(x,y)"),
        ("maximum", np.maximum, ("elementwise_max",), r"\operatorname{maximum}(x,y)"),
    ):
        infer = {
            "add": _additive_type,
            "subtract": _subtract_type,
            "multiply": _multiply_type,
            "divide": _divide_type,
            "power": _power_type,
            "minimum": _minimum_maximum_type,
            "maximum": _minimum_maximum_type,
        }[operator_id]
        specs.append(
            _spec(
                operator_id,
                "basic",
                (
                    _signature(
                        (any_numeric, any_numeric),
                        same_numeric,
                        "scalar broadcast; otherwise axes and shape must match",
                    ),
                ),
                "逐元素数值运算，仅允许标量广播。",
                infer,
                function,
                aliases=aliases,
                latex_template=latex,
            )
        )
    for operator_id, function, aliases, latex in (
        ("negate", np.negative, (), r"-x"),
        ("absolute", np.abs, ("abs",), r"\operatorname{abs}(x)"),
        ("sqrt", _sqrt, (), r"\sqrt{x}"),
    ):
        infer = _sqrt_type if operator_id == "sqrt" else _unary_numeric
        specs.append(
            _spec(
                operator_id,
                "basic",
                (_signature((any_numeric,), same_numeric, "preserve axes and shape"),),
                "逐元素一元运算。",
                infer,
                function,
                aliases=aliases,
                latex_template=latex,
            )
        )
    specs.append(
        _spec(
            "clip",
            "basic",
            (
                _signature(
                    (any_numeric, "scalar", "scalar"),
                    same_numeric,
                    "preserve first input",
                ),
            ),
            "将数值裁剪到标量上下界。",
            _clip_type,
            _clip,
            latex_template=r"\operatorname{clip}(x,lower,upper)",
        )
    )
    specs.append(
        _spec(
            "divide_or_default",
            "basic",
            (
                _signature(
                    (series_t, series_t, "scalar"),
                    series_t,
                    "matching time axes; denominator zero uses default",
                ),
            ),
            "逐元素安全除法；分母接近零时使用指定有限标量。",
            _divide_or_default_type,
            _divide_or_default,
            latex_template=r"\operatorname{divide\_or\_default}(x,y,d)",
        )
    )

    for operator_id, function, infer, aliases, latex in (
        ("log", _log, _log_exp_type, (), r"\log(x)"),
        ("exp", np.exp, _log_exp_type, (), r"\exp(x)"),
        ("reciprocal", _reciprocal, _reciprocal_type, (), r"x^{-1}"),
        ("sign", np.sign, _dimensionless_unary, (), r"\operatorname{sign}(x)"),
        (
            "normal_pdf",
            _normal_pdf,
            _strict_dimensionless_unary,
            (),
            r"\operatorname{normal_pdf}(x)",
        ),
        (
            "normal_ppf",
            _normal_ppf,
            _strict_dimensionless_unary,
            (),
            r"\operatorname{normal_ppf}(p)",
        ),
    ):
        specs.append(
            _spec(
                operator_id,
                "basic" if not operator_id.startswith("normal_") else "statistics",
                (_signature((any_numeric,), same_numeric, "preserve axes and shape"),),
                "逐元素数学函数。",
                infer,
                function,
                aliases=aliases,
                latex_template=latex,
            )
        )

    comparison_functions = (
        ("equal", np.equal, ("eq",)),
        ("not_equal", np.not_equal, ("ne",)),
        ("less_than", np.less, ("lt",)),
        ("less_equal", np.less_equal, ("le",)),
        ("greater_than", np.greater, ("gt",)),
        ("greater_equal", np.greater_equal, ("ge",)),
    )
    for operator_id, function, aliases in comparison_functions:
        specs.append(
            _spec(
                operator_id,
                "comparison",
                (
                    _signature(
                        (any_numeric, any_numeric),
                        "mask",
                        "scalar broadcast or exact shape",
                    ),
                ),
                "逐元素数值比较，输出同 shape 的布尔 mask。",
                _comparison_type,
                function,
                aliases=aliases,
                latex_template=rf"\operatorname{{{operator_id}}}(x,y)",
            )
        )
    for operator_id, function in (
        ("logical_and", np.logical_and),
        ("logical_or", np.logical_or),
    ):
        specs.append(
            _spec(
                operator_id,
                "mask",
                (
                    _signature(
                        ("mask<A>", "same(first)"), "same(first)", "exact mask shape"
                    ),
                ),
                "逐元素布尔逻辑。",
                _logical_binary_type,
                function,
                latex_template=rf"\operatorname{{{operator_id}}}(a,b)",
            )
        )
    specs.extend(
        (
            _spec(
                "logical_not",
                "mask",
                (_signature(("mask<A>",), "same(input)", "preserve mask shape"),),
                "逐元素布尔取反。",
                _logical_not_type,
                np.logical_not,
                latex_template=r"\operatorname{logical_not}(mask)",
            ),
            _spec(
                "where",
                "mask",
                (
                    _signature(
                        ("mask<A>", any_numeric, any_numeric),
                        same_numeric,
                        "mask selects branches",
                    ),
                ),
                "按 mask 逐元素选择两个兼容数值输入。",
                _where_type,
                np.where,
                latex_template=r"\operatorname{where}(mask,x,y)",
            ),
        )
    )

    for operator_id, function, aliases in (
        ("sum", reduce_sum, ("sequence_sum",)),
        ("product", reduce_product, ("prod", "sequence_prod")),
        ("mean", _reduce_mean_reference, ("sequence_mean",)),
        ("min_value", _reduce_min_reference, ("min",)),
        ("max_value", _reduce_max_reference, ("max",)),
    ):
        window_reducer = operator_id in {"mean", "min_value", "max_value"}
        specs.append(
            _spec(
                operator_id,
                "reduction",
                (reduction,),
                (
                    "将普通数值张量归约为标量；滚动窗口输入按每个时点独立归约。"
                    if window_reducer
                    else "将所有命名轴归约为标量。"
                ),
                _reduce_all_or_window if window_reducer else _reduce_all,
                function,
                aliases=aliases,
                latex_template=rf"\operatorname{{{operator_id}}}(x)",
                cost_model="reduction",
                cost=_cost_input,
            )
        )
    specs.extend(
        (
            _spec(
                "drawdown_series",
                "path",
                (
                    _signature(
                        (series_t,),
                        "series<time>[T]<return_decimal>",
                        "preserve time axis",
                    ),
                ),
                "计算每个观察值相对截至当期历史峰值的有符号回撤序列。",
                _drawdown_series_type,
                _drawdown_series,
                latex_template=r"\mathcal{D}(x)",
                cost_model="scan",
                cost=_cost_input,
            ),
            _spec(
                "new_high_mask",
                "path",
                (
                    _signature(
                        (series_t,),
                        "mask<time>[T]",
                        "preserve time axis",
                    ),
                ),
                "首个观察值为真；后续仅严格超过此前历史峰值时为真，平峰不重复计数。",
                _new_high_mask_type,
                _new_high_mask,
                latex_template=r"\mathcal{H}_{\mathrm{new}}(x)",
                cost_model="scan",
                cost=_cost_input,
            ),
        )
    )
    for operator_id, function, aliases in (
        ("variance", _variance, ("var",)),
        ("std", _std, ("sequence_std",)),
    ):
        infer = _variance_type if operator_id == "variance" else _std_type
        specs.append(
            _spec(
                operator_id,
                "reduction",
                (
                    reduction,
                    _signature(
                        (reducible_numeric, "scalar"),
                        "scalar",
                        "reduce all named axes with explicit ddof",
                    ),
                ),
                "将所有命名轴归约为标量，可显式传入有限非负整数 ddof。",
                infer,
                function,
                aliases=aliases,
                latex_template=rf"\operatorname{{{operator_id}}}(x,ddof)",
                cost_model="reduction",
                cost=_cost_input,
            )
        )
    for operator_id, function, aliases in (
        ("cumulative_sum", scan_sum, ()),
        ("cumulative_product", scan_product, ()),
        ("cumulative_return", scan_return, ()),
        ("cumulative_max", np.maximum.accumulate, ("cumulative_maximum",)),
        ("cumulative_min", np.minimum.accumulate, ("cumulative_minimum",)),
    ):
        specs.append(
            _spec(
                operator_id,
                "reduction",
                (
                    _signature(
                        (f"{series_t} | {vector_n}",),
                        "same(input)",
                        "preserve one-dimensional axis",
                    ),
                ),
                "沿唯一命名轴生成累计序列。",
                _preserve_one_dimensional,
                function,
                aliases=aliases,
                latex_template=rf"\operatorname{{{operator_id}}}(x)",
                cost_model="scan",
                cost=_cost_input,
            )
        )
    rolling_window_signatures = (
        _signature(
            (series_t, "scalar<count>"),
            "window<time,window>[T,W]",
            "logical causal windows; no materialized T×W production array",
        ),
        _signature(
            (series_t, "scalar<count>", "scalar<count>"),
            "window<time,window>[T,W]",
            "logical causal windows with explicit minimum observations",
        ),
    )
    specs.append(
        _spec(
            "rolling_window",
            "rolling",
            rolling_window_signatures,
            "只定义截至当前时点的因果滚动观察窗口；统计量由后续普通归约算子决定。",
            _rolling_window_type,
            _rolling_window,
            latex_template=r"\mathcal{W}_{w,m}(x)",
            cost_model="logical_window",
            cost=_cost_input,
        )
    )
    rolling_signatures = (
        _signature((series_t, "scalar<count>"), series_t, "preserve time axis"),
        _signature(
            (series_t, "scalar<count>", "scalar<count>"),
            series_t,
            "preserve time axis with explicit minimum observations",
        ),
    )
    for operator_id, function in (
        ("rolling_mean", _rolling_mean),
        ("rolling_min", _rolling_min),
        ("rolling_max", _rolling_max),
    ):
        specs.append(
            _spec(
                operator_id,
                "rolling",
                rolling_signatures,
                "沿时间轴执行因果滚动计算，前置样本不足时返回缺失值。",
                _rolling_type,
                function,
                latex_template=rf"\operatorname{{{operator_id}}}(x,w,m)",
                cost_model="rolling_scan",
                cost=_cost_input,
            )
        )
    specs.append(
        _spec(
            "rolling_std",
            "rolling",
            (
                _signature((series_t, "scalar<count>"), series_t, "preserve time axis"),
                _signature(
                    (series_t, "scalar<count>", "scalar<count>"),
                    series_t,
                    "preserve time axis with explicit ddof",
                ),
                _signature(
                    (
                        series_t,
                        "scalar<count>",
                        "scalar<count>",
                        "scalar<count>",
                    ),
                    series_t,
                    "preserve time axis with explicit ddof and minimum observations",
                ),
            ),
            "沿时间轴计算因果滚动标准差，支持显式 ddof 与最小观察数。",
            _rolling_std_type,
            _rolling_std,
            latex_template=r"\operatorname{rolling\_std}(x,w,ddof,m)",
            cost_model="rolling_scan",
            cost=_cost_input,
        )
    )
    specs.append(
        _spec(
            "recursive_smooth",
            "rolling",
            (
                _signature(
                    (series_t, "scalar<count>", "scalar"),
                    series_t,
                    "causal recurrence preserving time axis",
                ),
            ),
            "按 ((n-1)×前值+当前值)/n 进行因果递归平滑。",
            _recursive_smooth_type,
            _recursive_smooth,
            interval_policy="history_required",
            latex_template=r"\operatorname{recursive\_smooth}(x,n,x_0)",
            cost_model="scan",
            cost=_cost_input,
        )
    )

    for operator_id, function, infer, output, shape_rule in (
        ("first", _first, _last_type, "scalar", "select first element"),
        ("length", _length, _length_type, "scalar<count>", "count elements"),
        (
            "lag",
            _lag,
            _lag_difference_type,
            "same(input)[K-n]",
            "drop final n elements",
        ),
        (
            "difference",
            _difference,
            _lag_difference_type,
            "same(input)[K-n]",
            "n-period difference",
        ),
    ):
        specs.append(
            _spec(
                operator_id,
                "reduction" if operator_id in {"first", "length"} else "sequence",
                (
                    _signature((f"{series_t} | {vector_n}",), output, shape_rule),
                    _signature(
                        (f"{series_t} | {vector_n}", "scalar<count>"),
                        output,
                        shape_rule,
                    ),
                )
                if operator_id in {"lag", "difference"}
                else (_signature((f"{series_t} | {vector_n}",), output, shape_rule),),
                "一维序列数学运算。",
                infer,
                function,
                latex_template=rf"\operatorname{{{operator_id}}}(x)",
                cost_model="constant" if operator_id in {"first", "length"} else "scan",
                cost=(lambda _inputs, _output: "1")
                if operator_id in {"first", "length"}
                else _cost_input,
            )
        )

    for operator_id, function, infer, aliases in (
        ("median", np.median, _reduce_all, ()),
        ("skewness", _skewness, _dimensionless_reduction_type, ("skew",)),
        (
            "excess_kurtosis",
            _excess_kurtosis,
            _dimensionless_reduction_type,
            ("kurtosis_excess",),
        ),
        ("mean_absolute_deviation", _mean_absolute_deviation, _reduce_all, ("mad",)),
        ("root_mean_square", _root_mean_square, _reduce_all, ("rms",)),
        ("argmin", lambda values: float(np.argmin(values)), _arg_type, ()),
        ("argmax", lambda values: float(np.argmax(values)), _arg_type, ()),
    ):
        specs.append(
            _spec(
                operator_id,
                "statistics",
                (reduction,),
                "将所有命名轴归约为数学统计量。",
                infer,
                function,
                aliases=aliases,
                latex_template=rf"\operatorname{{{operator_id}}}(x)",
                cost_model="reduction",
                cost=_cost_input,
            )
        )
    specs.append(
        _spec(
            "quantile",
            "statistics",
            (_signature((reducible_numeric, "scalar"), "scalar", "reduce all axes"),),
            "计算给定概率的分位数。",
            _quantile_type,
            _quantile,
            latex_template=r"\operatorname{quantile}(x,p)",
            cost_model="selection",
            cost=_cost_input,
        )
    )

    masked_specs = (
        ("sum_where", _masked_sum, _masked_reduction_type, ("masked_sum",)),
        ("mean_where", _masked_mean, _masked_reduction_type, ("masked_mean",)),
        (
            "variance_where",
            _masked_variance,
            _masked_variance_type,
            ("masked_variance",),
        ),
        ("std_where", _masked_std, _masked_reduction_type, ("masked_std",)),
        (
            "min_where",
            lambda values, mask: float(np.min(_selected(values, mask))),
            _masked_reduction_type,
            ("masked_min",),
        ),
        (
            "max_where",
            lambda values, mask: float(np.max(_selected(values, mask))),
            _masked_reduction_type,
            ("masked_max",),
        ),
        (
            "median_where",
            lambda values, mask: float(np.median(_selected(values, mask))),
            _masked_reduction_type,
            ("masked_median",),
        ),
    )
    for operator_id, function, infer, aliases in masked_specs:
        specs.append(
            _spec(
                operator_id,
                "mask",
                (
                    _signature(
                        (reducible_numeric, "same-shape mask"),
                        "scalar",
                        "select then reduce",
                    ),
                ),
                "仅对 mask 选中的元素执行归约。",
                infer,
                function,
                aliases=aliases,
                latex_template=rf"\operatorname{{{operator_id}}}(x,mask)",
                cost_model="masked_reduction",
                cost=_cost_all_inputs,
            )
        )
    specs.extend(
        (
            _spec(
                "quantile_where",
                "mask",
                (
                    _signature(
                        (reducible_numeric, "same-shape mask", "scalar"),
                        "scalar",
                        "select then quantile",
                    ),
                ),
                "对 mask 选中元素计算分位数。",
                _masked_reduction_type,
                _masked_quantile,
                aliases=("masked_quantile",),
                latex_template=r"\operatorname{quantile_where}(x,mask,p)",
                cost_model="masked_reduction",
                cost=_cost_all_inputs,
            ),
            _spec(
                "count_true",
                "mask",
                (
                    _signature(
                        ("non-scalar mask",), "scalar<count>", "count true values"
                    ),
                ),
                "统计 mask 中 True 的数量。",
                _masked_count_type,
                lambda mask: float(np.count_nonzero(mask)),
                aliases=("masked_count",),
                latex_template=r"\operatorname{count_true}(mask)",
                cost_model="reduction",
                cost=_cost_input,
            ),
            _spec(
                "max_consecutive_true",
                "mask",
                (
                    _signature(
                        ("one-dimensional mask",), "scalar<count>", "longest true run"
                    ),
                ),
                "计算一维 mask 中最长连续 True 长度。",
                _max_consecutive_type,
                _max_consecutive_true,
                latex_template=r"\operatorname{max_consecutive_true}(mask)",
                cost_model="scan",
                cost=_cost_input,
            ),
        )
    )

    regression_signatures = (
        _signature((series_t,), "scalar", "regress values on time index"),
        _signature((series_t, series_t), "scalar", "ordinary least squares y on x"),
    )
    for operator_id, function, infer in (
        ("linear_slope", _linear_slope, _regression_type),
        ("linear_intercept", _linear_intercept, _regression_intercept_type),
        ("linear_r_squared", _linear_r_squared, _r_squared_type),
        (
            "regression_standard_error",
            _regression_standard_error,
            _regression_intercept_type,
        ),
    ):
        specs.append(
            _spec(
                operator_id,
                "statistics",
                regression_signatures,
                "一元普通最小二乘回归标量结果。",
                infer,
                function,
                latex_template=rf"\operatorname{{{operator_id}}}(x,y)",
                cost_model="regression",
                cost=_cost_all_inputs,
            )
        )
    specs.extend(
        (
            _spec(
                "last",
                "reduction",
                (
                    _signature(
                        (f"{series_t} | {vector_n}",), "scalar", "select final element"
                    ),
                ),
                "取一维输入的最后一个值。",
                _last_type,
                lambda value: np.asarray(value, dtype=np.float64)[-1],
                latex_template=r"\operatorname{last}(x)",
                cost_model="constant",
                cost=lambda _inputs, _output: "1",
            ),
            _spec(
                "total_return",
                "reduction",
                (_signature((series_t,), "scalar", "prod(1+r)-1"),),
                "计算区间累计收益率。",
                _reduce_all,
                _total_return,
                latex_template=r"\operatorname{total_return}(\mathbf{r})",
                cost_model="reduction",
                cost=_cost_input,
            ),
            _spec(
                "annualized_return",
                "reduction",
                (
                    _signature(
                        (series_t, "scalar"),
                        "scalar",
                        "annualize by periods_per_year / T",
                    ),
                ),
                "根据每年观察数计算年化收益率。",
                _annualized_return_type,
                _annualized_return,
                latex_template=r"\operatorname{annualized_return}(\mathbf{r},p_{year})",
                cost_model="reduction",
                cost=_cost_input,
            ),
        )
    )

    for suffix, function in (
        ("sum", lambda value: np.sum(value, axis=0)),
        ("mean", lambda value: np.mean(value, axis=0)),
        ("product", lambda value: np.prod(value, axis=0)),
        ("variance", lambda value: np.var(value, axis=0, ddof=1)),
        ("std", lambda value: np.std(value, axis=0, ddof=1)),
        ("min", lambda value: np.min(value, axis=0)),
        ("max", lambda value: np.max(value, axis=0)),
    ):
        infer = _variance_time_type if suffix == "variance" else _reduce_time
        specs.append(
            _spec(
                f"{suffix}_time",
                "reduction",
                (_signature((matrix_ta,), vector_n, "remove time axis"),),
                "沿时间轴归约，保留资产轴。",
                infer,
                function,
                latex_template=rf"\operatorname{{{suffix}_time}}(\mathbf{{R}})",
                cost_model="reduction",
                cost=_cost_input,
            )
        )
    for suffix, function in (
        ("sum", lambda value: np.sum(value, axis=1)),
        ("mean", lambda value: np.mean(value, axis=1)),
        ("product", lambda value: np.prod(value, axis=1)),
        ("variance", lambda value: np.var(value, axis=1, ddof=1)),
        ("std", lambda value: np.std(value, axis=1, ddof=1)),
        ("min", lambda value: np.min(value, axis=1)),
        ("max", lambda value: np.max(value, axis=1)),
    ):
        infer = _variance_asset_type if suffix == "variance" else _reduce_asset
        specs.append(
            _spec(
                f"{suffix}_asset",
                "reduction",
                (_signature((matrix_ta,), series_t, "remove asset axis"),),
                "沿资产轴归约，保留时间轴。",
                infer,
                function,
                latex_template=rf"\operatorname{{{suffix}_asset}}(\mathbf{{R}})",
                cost_model="reduction",
                cost=_cost_input,
            )
        )

    specs.extend(
        (
            _spec(
                "transpose",
                "linear_algebra",
                (
                    _signature(
                        ("matrix<A,B>[M,K]",),
                        "matrix<B,A>[K,M]",
                        "swap axes and dimensions",
                    ),
                ),
                "交换矩阵的两个命名轴。",
                _transpose_type,
                np.transpose,
                latex_template=r"\operatorname{transpose}(M)",
                cost_model="view",
                cost=lambda _inputs, _output: "1",
            ),
            _spec(
                "dot",
                "linear_algebra",
                (
                    _signature(
                        ("series<A>[K] | vector<A>[K]", "same(first)"),
                        "scalar",
                        "contract the shared named axis",
                    ),
                ),
                "同轴一维内积。",
                _dot_type,
                np.dot,
                latex_template=r"\operatorname{dot}(x,y)",
                cost_model="dot",
                cost=_cost_input,
            ),
            _spec(
                "outer",
                "linear_algebra",
                (
                    _signature(
                        (vector_n, vector_n),
                        "matrix<asset,asset>[N,M]",
                        "outer product",
                    ),
                ),
                "两个资产向量的外积。",
                _outer_type,
                np.outer,
                latex_template=r"\operatorname{outer}(w_1,w_2)",
                cost_model="outer",
            ),
            _spec(
                "matmul",
                "linear_algebra",
                (
                    _signature(
                        ("matrix<A,B>[M,K]", "matrix<B,C>[K,N]"),
                        "matrix<A,C>[M,N]",
                        "contract matching inner axis B/K",
                    ),
                ),
                "按命名内轴执行矩阵乘法。",
                _matmul_type,
                np.matmul,
                latex_template=r"\operatorname{matmul}(A,B)",
                cost_model="matrix_multiply",
                cost=_cost_matmul,
            ),
            _spec(
                "matvec",
                "linear_algebra",
                (
                    _signature(
                        ("matrix<A,B>[M,K]", "series<B>[K] | vector<B>[K]"),
                        "one_dimensional<A>[M]",
                        "contract matrix inner axis",
                    ),
                ),
                "矩阵与同内轴一维输入相乘。",
                _matvec_type,
                np.matmul,
                latex_template=r"\operatorname{matvec}(M,v)",
                cost_model="matrix_vector",
                cost=_cost_all_inputs,
            ),
            _spec(
                "diag",
                "linear_algebra",
                (
                    _signature(
                        (vector_n,),
                        "matrix<asset,asset>[N,N]",
                        "create diagonal matrix",
                    ),
                    _signature(
                        ("matrix<asset,asset>[N,N]",), vector_n, "extract diagonal"
                    ),
                ),
                "创建或提取资产对角矩阵。",
                _diag_type,
                np.diag,
                latex_template=r"\operatorname{diag}(x)",
                cost_model="diagonal",
            ),
            _spec(
                "trace",
                "linear_algebra",
                (_signature(("matrix<A,A>[N,N]",), "scalar", "sum diagonal"),),
                "计算同名双轴方阵的迹。",
                _trace_type,
                np.trace,
                latex_template=r"\operatorname{trace}(M)",
                cost_model="diagonal",
                cost=_cost_input,
            ),
            _spec(
                "solve",
                "linear_algebra",
                (
                    _signature(
                        ("matrix<asset,asset>[N,N]", vector_n),
                        vector_n,
                        "solve A*x=b; no explicit inverse",
                    ),
                ),
                "求解资产线性方程组，不暴露矩阵求逆。",
                _solve_type,
                np.linalg.solve,
                latex_template=r"\operatorname{solve}(A,b)",
                cost_model="linear_solve",
                cost=_cost_solve,
            ),
        )
    )

    covariance_signatures = (
        _signature(
            (matrix_ta,), "matrix<asset,asset>[N,N]", "sample covariance across time"
        ),
        _signature(
            (series_t, series_t), "scalar", "sample covariance on common time axis"
        ),
    )
    specs.extend(
        (
            _spec(
                "covariance",
                "statistics",
                covariance_signatures,
                "计算样本协方差（ddof=1）。",
                _covariance_type,
                _covariance,
                aliases=("cov",),
                latex_template=r"\operatorname{covariance}(\mathbf{R})",
                cost_model="covariance",
                cost=_cost_covariance,
            ),
            _spec(
                "correlation",
                "statistics",
                tuple(
                    OperatorSignature(
                        item.inputs,
                        item.output,
                        item.shape_rule.replace("covariance", "correlation"),
                    )
                    for item in covariance_signatures
                ),
                "计算 Pearson 相关系数或相关矩阵。",
                _correlation_type,
                _correlation,
                aliases=("corr",),
                latex_template=r"\operatorname{correlation}(\mathbf{R})",
                cost_model="correlation",
                cost=_cost_covariance,
            ),
            _spec(
                "portfolio_returns",
                "portfolio",
                (
                    _signature(
                        (matrix_ta, vector_n),
                        series_t,
                        "matrix-vector contraction over asset",
                    ),
                ),
                "使用资产权重将多资产收益矩阵合成为组合收益序列。",
                _portfolio_returns_type,
                np.matmul,
                latex_template=r"\operatorname{portfolio_returns}(\mathbf{R},\mathbf{w})",
                cost_model="matrix_vector",
                cost=_cost_all_inputs,
            ),
            _spec(
                "quadratic_form",
                "portfolio",
                (
                    _signature(
                        (vector_n, "matrix<asset,asset>[N,N]"), "scalar", "w^T*A*w"
                    ),
                ),
                "计算资产权重与风险矩阵的二次型。",
                _quadratic_form_type,
                _quadratic_form,
                latex_template=r"\operatorname{quadratic_form}(\mathbf{w},A)",
                cost_model="quadratic_form",
                cost=lambda inputs, _output: f"{inputs[0].shape[0]}^2",
            ),
            _spec(
                "active_returns",
                "portfolio",
                (
                    _signature(
                        (series_t, series_t), series_t, "subtract on common time axis"
                    ),
                ),
                "计算资产或组合相对基准的主动收益序列。",
                _active_returns_type,
                np.subtract,
                latex_template=r"\operatorname{active_returns}(\mathbf{r},\mathbf{b})",
            ),
        )
    )
    return tuple(specs)


@lru_cache(maxsize=5)
def get_typed_operator_registry(
    version: str = TYPED_OPERATOR_REGISTRY_VERSION,
) -> Mapping[str, TypedOperatorSpec]:
    if version not in SUPPORTED_OPERATOR_REGISTRY_VERSIONS:
        raise TypedDslError(
            "OPERATOR_VERSION_NOT_FOUND",
            f"未安装算子注册表版本 {version}。",
            details={
                "available_versions": sorted(SUPPORTED_OPERATOR_REGISTRY_VERSIONS)
            },
        )
    registry: dict[str, TypedOperatorSpec] = {}
    legacy_numpy_evaluators: dict[str, RuntimeFunction] = {
        "sum": np.sum,
        "product": np.prod,
        "mean": np.mean,
        "min_value": np.min,
        "max_value": np.max,
        "variance": _legacy_numpy_variance,
        "std": _legacy_numpy_std,
        "cumulative_sum": np.cumsum,
        "cumulative_product": np.cumprod,
        "cumulative_return": _cumulative_return,
    }
    for spec in _canonical_specs():
        if version in {
            LEGACY_OPERATOR_REGISTRY_VERSION,
            COMPAT_OPERATOR_REGISTRY_VERSION,
        } and spec.operator_id in V22_OPERATOR_IDS:
            continue
        if version not in {ROLLING_OPERATOR_REGISTRY_VERSION, TYPED_OPERATOR_REGISTRY_VERSION} and spec.operator_id in V23_OPERATOR_IDS:
            continue
        if version != TYPED_OPERATOR_REGISTRY_VERSION and spec.operator_id in V24_OPERATOR_IDS:
            continue
        if version == LEGACY_OPERATOR_REGISTRY_VERSION:
            if spec.operator_id not in LEGACY_OPERATOR_IDS:
                continue
            spec = replace(
                spec,
                version=LEGACY_OPERATOR_REGISTRY_VERSION,
                evaluate=legacy_numpy_evaluators.get(
                    spec.operator_id, spec.evaluate
                ),
            )
        elif version == COMPAT_OPERATOR_REGISTRY_VERSION:
            spec = replace(spec, version=COMPAT_OPERATOR_REGISTRY_VERSION)
        elif version == PREVIOUS_OPERATOR_REGISTRY_VERSION:
            spec = replace(spec, version=PREVIOUS_OPERATOR_REGISTRY_VERSION)
        elif version == ROLLING_OPERATOR_REGISTRY_VERSION:
            spec = replace(spec, version=ROLLING_OPERATOR_REGISTRY_VERSION)
        if version in {
            COMPAT_OPERATOR_REGISTRY_VERSION,
            PREVIOUS_OPERATOR_REGISTRY_VERSION,
            ROLLING_OPERATOR_REGISTRY_VERSION,
        } and spec.operator_id in {
            "mean", "min_value", "max_value"
        }:
            historical_reducers = {
                "mean": reduce_mean,
                "min_value": reduce_min,
                "max_value": reduce_max,
            }
            spec = replace(
                spec,
                infer=_reduce_all,
                evaluate=historical_reducers[spec.operator_id],
            )
        elif version == TYPED_OPERATOR_REGISTRY_VERSION and spec.operator_id in {
            "mean", "min_value", "max_value"
        }:
            spec = replace(
                spec,
                signatures=spec.signatures + (
                    _signature(
                        ("window<time,window>[T,W]",),
                        "series<time>[T]",
                        "reduce each logical rolling window without materialization",
                    ),
                ),
            )
        elif version == TYPED_OPERATOR_REGISTRY_VERSION and spec.operator_id in {"variance", "std"}:
            spec = replace(
                spec,
                signatures=spec.signatures + (
                    _signature(
                        ("window<time,window>[T,W]",),
                        "series<time>[T]",
                        "reduce each logical rolling window with default ddof",
                    ),
                    _signature(
                        ("window<time,window>[T,W]", "scalar<count>"),
                        "series<time>[T]",
                        "reduce each logical rolling window with explicit ddof",
                    ),
                ),
            )
        registry[spec.operator_id] = spec
        for alias in spec.aliases:
            if alias in registry:
                raise RuntimeError(f"重复算子别名: {alias}")
            registry[alias] = spec
    from .primitive_access import access_operator_specs
    from .regression_state import fit_operator_specs
    for spec in (*access_operator_specs(version), *fit_operator_specs(version)):
        registry[spec.operator_id] = spec
    if version == TYPED_OPERATOR_REGISTRY_VERSION:
        from .rolling_scope import rolling_scope_spec
        scope = rolling_scope_spec(version)
        registry[scope.operator_id] = scope
    if version in {ROLLING_OPERATOR_REGISTRY_VERSION, TYPED_OPERATOR_REGISTRY_VERSION}:
        from .drawdown_interval import interval_operator_specs
        for spec in interval_operator_specs(version):
            registry[spec.operator_id] = spec
    return MappingProxyType(registry)


def get_typed_operator_catalog(
    version: str = TYPED_OPERATOR_REGISTRY_VERSION,
) -> dict[str, Any]:
    registry = get_typed_operator_registry(version)
    historical_ids = {spec.operator_id for spec in _canonical_specs()}
    exclusions = PUBLIC_OPERATOR_EXCLUSIONS if version == TYPED_OPERATOR_REGISTRY_VERSION else frozenset({
        "cumulative_return", "total_return", "annualized_return", "portfolio_returns", "active_returns",
    })
    canonical = sorted(
        {
            spec.operator_id: spec
            for spec in registry.values()
            if spec.operator_id not in exclusions
            and (version == TYPED_OPERATOR_REGISTRY_VERSION or spec.operator_id in historical_ids)
        }.values(),
        key=lambda item: (item.category, item.operator_id),
    )
    return {
        "dsl_version": {
            LEGACY_OPERATOR_REGISTRY_VERSION: LEGACY_TYPED_DSL_VERSION,
            COMPAT_OPERATOR_REGISTRY_VERSION: COMPAT_TYPED_DSL_VERSION,
            PREVIOUS_OPERATOR_REGISTRY_VERSION: PREVIOUS_TYPED_DSL_VERSION,
            ROLLING_OPERATOR_REGISTRY_VERSION: ROLLING_TYPED_DSL_VERSION,
            TYPED_OPERATOR_REGISTRY_VERSION: TYPED_DSL_VERSION,
        }[version],
        "compiler_version": {
            LEGACY_OPERATOR_REGISTRY_VERSION: LEGACY_TYPED_COMPILER_VERSION,
            COMPAT_OPERATOR_REGISTRY_VERSION: COMPAT_TYPED_COMPILER_VERSION,
            PREVIOUS_OPERATOR_REGISTRY_VERSION: PREVIOUS_TYPED_COMPILER_VERSION,
            ROLLING_OPERATOR_REGISTRY_VERSION: ROLLING_TYPED_COMPILER_VERSION,
            TYPED_OPERATOR_REGISTRY_VERSION: TYPED_COMPILER_VERSION,
        }[version],
        "operator_registry_version": version,
        "operators": [spec.to_catalog_entry() for spec in canonical],
    }


def canonical_operator_spec(name: str) -> TypedOperatorSpec:
    registry = get_typed_operator_registry()
    try:
        return registry[name]
    except KeyError as exc:
        raise TypedDslError("UNKNOWN_OPERATOR", f"未知函数或算子: {name}") from exc

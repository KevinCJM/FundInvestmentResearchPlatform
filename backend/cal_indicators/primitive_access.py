"""Small typed indexing and domain guards used by transparent indicator DAGs."""
from __future__ import annotations

import math
from functools import partial

import numba
import numpy as np
from numba import types

from .typed_types import TypedDslError, ValueType

_F1 = types.Array(types.float64, 1, "C", readonly=True)


@numba.njit(types.float64(_F1, types.float64), cache=False, nogil=True)
def value_at_kernel(values: np.ndarray, position: float) -> float:
    if not math.isfinite(position) or position < 0.0 or position != math.floor(position) or position >= values.size:
        return np.nan
    value = values[int(position)]
    return value if math.isfinite(value) else np.nan


@numba.njit(types.float64(types.float64), cache=False, nogil=True)
def require_positive_kernel(value: float) -> float:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("DOMAIN_ERROR")
    return value


@numba.njit(types.float64(types.float64), cache=False, nogil=True)
def require_nonnegative_kernel(value: float) -> float:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("DOMAIN_ERROR")
    return value


@numba.njit(types.uint8[::1](_F1), cache=False, nogil=True)
def finite_mask_kernel(values: np.ndarray) -> np.ndarray:
    """Identify valid observations without copying or filtering the source."""
    result = np.empty(values.size, dtype=np.uint8)
    for index in range(values.size):
        result[index] = 1 if math.isfinite(values[index]) else 0
    return result


ACCESS_KERNELS = {
    "finite_mask": finite_mask_kernel,
    "value_at": value_at_kernel,
    "require_positive": require_positive_kernel,
    "require_nonnegative": require_nonnegative_kernel,
}
for _kernel in ACCESS_KERNELS.values():
    _kernel.disable_compile()


def _infer_value(inputs):
    values, position = inputs
    if values.kind not in {"series", "vector"} or not values.is_numeric:
        raise TypedDslError("TYPE_MISMATCH", "按位置取值需要一个一维数值或日期序列。")
    if not position.is_scalar or not position.is_numeric or position.semantic_dimension not in {"count", "dimensionless"}:
        raise TypedDslError("TYPE_MISMATCH", "位置必须是非负整数，不能是日期或收益率。")
    return ValueType.scalar(semantic_dimension=values.semantic_dimension, price_basis=values.price_basis)


def _infer_guard(inputs):
    value = inputs[0]
    if not value.is_scalar or not value.is_numeric or value.semantic_dimension == "date":
        raise TypedDslError("TYPE_MISMATCH", "数值范围约束只接受一个数值标量。")
    return value


def _infer_finite_mask(inputs):
    value = inputs[0]
    if value.kind != "series" or not value.is_numeric or value.semantic_dimension == "date":
        raise TypedDslError("TYPE_MISMATCH", "有限值判断需要一条数值时间序列。")
    return value.as_mask()


def access_operator_specs(version):
    from .typed_operators import OperatorSignature, TypedOperatorSpec, TYPED_OPERATOR_REGISTRY_VERSION

    specs = [TypedOperatorSpec(
        "value_at", version, "sequence",
        (OperatorSignature(("series<time>[L] | vector<asset>[N]", "scalar"), "scalar", "preserve element measure and price basis; zero-based index"),),
        "按零基位置读取一个值，保留数值或日期类型；缺失、负数、分数和越界位置不可用。",
        _infer_value, value_at_kernel, cost_model="constant", interval_policy="local",
    )]
    for name, label in (("require_positive", "要求输入为有限正数"), ("require_nonnegative", "要求输入为有限非负数")):
        specs.append(TypedOperatorSpec(
            name, version, "basic", (OperatorSignature(("scalar",), "scalar", "validate domain without changing the value"),),
            label, _infer_guard, ACCESS_KERNELS[name], cost_model="constant", interval_policy="local",
        ))
    if version == TYPED_OPERATOR_REGISTRY_VERSION:
        specs.append(TypedOperatorSpec(
            "finite_mask", version, "comparison",
            (OperatorSignature(("series<time>[T]",), "mask<time>[T]", "true for finite observations; false for NaN and positive/negative infinity"),),
            "逐项判断数值是否有限；真实0为有效值，NaN和正负无穷为无效，不删除观察位置。",
            _infer_finite_mask, finite_mask_kernel, cost_model="elementwise", interval_policy="local",
        ))
    return tuple(specs)

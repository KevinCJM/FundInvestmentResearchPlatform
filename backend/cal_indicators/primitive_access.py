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


ACCESS_KERNELS = {
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


def access_operator_specs(version):
    from .typed_operators import OperatorSignature, TypedOperatorSpec

    specs = [TypedOperatorSpec(
        "value_at", version, "sequence",
        (OperatorSignature(("series<time>[L] | vector<asset>[N]", "scalar"), "scalar", "preserve element measure and price basis; zero-based index"),),
        "按零基位置读取一个值，保留数值或日期类型；缺失、负数、分数和越界位置不可用。",
        _infer_value, value_at_kernel, cost_model="constant",
    )]
    for name, label in (("require_positive", "要求输入为有限正数"), ("require_nonnegative", "要求输入为有限非负数")):
        specs.append(TypedOperatorSpec(
            name, version, "basic", (OperatorSignature(("scalar",), "scalar", "validate domain without changing the value"),),
            label, _infer_guard, ACCESS_KERNELS[name], cost_model="constant",
        ))
    return tuple(specs)

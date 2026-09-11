"""One OLS fit with lightweight typed field projections, not a metric bundle."""
from __future__ import annotations

import math
from functools import partial

import numba
import numpy as np
from numba import types

from .typed_types import TypedDslError, ValueType

_F1 = types.Array(types.float64, 1, "C", readonly=True)
_FIT = types.UniTuple(types.float64, 5)
FIT_FIELDS = ("slope", "intercept", "residual_sum_squares", "total_sum_squares", "observation_count")


@numba.njit(_FIT(_F1, _F1, types.boolean), cache=False, nogil=True)
def _fit_kernel(x, y, implicit_x):
    count = y.size
    if count < 2 or (not implicit_x and x.size != count):
        raise ValueError("INSUFFICIENT_SAMPLE")
    sum_x = 0.0
    sum_y = 0.0
    for index in range(count):
        xi = float(index) if implicit_x else x[index]
        yi = y[index]
        if not math.isfinite(xi) or not math.isfinite(yi):
            raise ValueError("DOMAIN_ERROR")
        sum_x += xi
        sum_y += yi
    mean_x = sum_x / count
    mean_y = sum_y / count
    xx = xy = yy = 0.0
    for index in range(count):
        dx = (float(index) if implicit_x else x[index]) - mean_x
        dy = y[index] - mean_y
        xx += dx * dx
        xy += dx * dy
        yy += dy * dy
    if xx <= 0.0 or not math.isfinite(xx):
        raise ValueError("DOMAIN_ERROR")
    slope = xy / xx
    intercept = mean_y - slope * mean_x
    residual = 0.0
    for index in range(count):
        xi = float(index) if implicit_x else x[index]
        error = y[index] - (intercept + slope * xi)
        residual += error * error
    if not math.isfinite(slope) or not math.isfinite(intercept) or not math.isfinite(residual) or not math.isfinite(yy):
        raise ValueError("DOMAIN_ERROR")
    return slope, intercept, residual, yy, float(count)


@numba.njit(_FIT(_F1, _F1), cache=False, nogil=True)
def linear_fit_pair_kernel(x, y):
    return _fit_kernel(x, y, False)


@numba.njit(_FIT(_F1), cache=False, nogil=True)
def linear_fit_time_kernel(values):
    # Reuse the input pointer; the implicit 0..n-1 axis needs no new array.
    return _fit_kernel(values, values, True)


@numba.njit(types.float64(_FIT), cache=False, nogil=True)
def fit_slope_kernel(state):
    return state[0]


@numba.njit(types.float64(_FIT), cache=False, nogil=True)
def fit_intercept_kernel(state):
    return state[1]


@numba.njit(types.float64(_FIT), cache=False, nogil=True)
def fit_residual_sum_squares_kernel(state):
    return state[2]


@numba.njit(types.float64(_FIT), cache=False, nogil=True)
def fit_total_sum_squares_kernel(state):
    return state[3]


@numba.njit(types.float64(_FIT), cache=False, nogil=True)
def fit_observation_count_kernel(state):
    return state[4]


FIT_PROJECTION_KERNELS = {
    "fit_slope": fit_slope_kernel,
    "fit_intercept": fit_intercept_kernel,
    "fit_residual_sum_squares": fit_residual_sum_squares_kernel,
    "fit_total_sum_squares": fit_total_sum_squares_kernel,
    "fit_observation_count": fit_observation_count_kernel,
}
for _kernel in (_fit_kernel, linear_fit_pair_kernel, linear_fit_time_kernel, *FIT_PROJECTION_KERNELS.values()):
    _kernel.disable_compile()


def _infer_fit(inputs):
    from .typed_operators import _regression_type, _regression_intercept_type
    slope = _regression_type(inputs)
    intercept = _regression_intercept_type(inputs)
    measure = intercept.semantic_dimension
    squared = "dimensionless" if measure == "dimensionless" else f"squared:{measure}"
    return ValueType("record", fields=(
        ("slope", slope), ("intercept", intercept),
        ("residual_sum_squares", ValueType.scalar(semantic_dimension=squared, price_basis=intercept.price_basis)),
        ("total_sum_squares", ValueType.scalar(semantic_dimension=squared, price_basis=intercept.price_basis)),
        ("observation_count", ValueType.scalar(semantic_dimension="count")),
    ))


def _infer_projection(inputs, *, field):
    state = inputs[0]
    if state.kind != "record" or tuple(name for name, _ in state.fields) != FIT_FIELDS:
        raise TypedDslError("TYPE_MISMATCH", "请先连接同一次线性拟合的中间结果。")
    return dict(state.fields)[field]


def _reference_fit(*values):
    # Registry callbacks are test oracles only; production lowers directly to
    # the frozen dispatchers above. No second numeric fitting algorithm exists.
    arrays = tuple(np.ascontiguousarray(value, dtype=np.float64) for value in values)
    return linear_fit_time_kernel(arrays[0]) if len(arrays) == 1 else linear_fit_pair_kernel(*arrays)


def fit_operator_specs(version):
    from .typed_operators import OperatorSignature, TypedOperatorSpec
    specs = [TypedOperatorSpec(
        "linear_fit", version, "statistics", (
            OperatorSignature(("series<time>[T]",), "record", "OLS with intercept against the 0..n-1 observation axis"),
            OperatorSignature(("series<time>[T]", "series<time>[T]"), "record", "OLS with intercept on the same observation axis"),
        ), "执行一次带截距线性拟合，后续通过字段提取及基础数学构建独立指标。",
        _infer_fit, _reference_fit, cost_model="regression", cost=lambda inputs, output: str(inputs[0].shape[0]), interval_policy="resettable",
    )]
    for field in FIT_FIELDS:
        name = f"fit_{field}"
        specs.append(TypedOperatorSpec(
            name, version, "statistics", (OperatorSignature(("record",), "scalar", "read one fitted field; never fit again"),),
            "读取同一次线性拟合的一个统计字段，不重复拟合。", partial(_infer_projection, field=field),
            FIT_PROJECTION_KERNELS[name], cost_model="constant", interval_policy="local",
        ))
    return tuple(specs)

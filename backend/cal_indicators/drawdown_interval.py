"""Identify one drawdown event; projections only expose observation positions."""
from __future__ import annotations

import math

import numba
import numpy as np
from numba import types

from .typed_types import TypedDslError, ValueType

_F1 = types.Array(types.float64, 1, "C", readonly=True)
_INTERVAL = types.UniTuple(types.float64, 4)
# status: -1 invalid input, 0 no drawdown event, 1 selected event.
INTERVAL_FIELDS = ("start", "trough", "recovery", "event_status")
INTERVAL_TYPE = ValueType("record", fields=tuple(
    (name, ValueType.scalar(semantic_dimension="count")) for name in INTERVAL_FIELDS
))


@numba.njit(_INTERVAL(_F1), nogil=True, cache=False)
def last_drawdown_interval_kernel(drawdowns):
    if drawdowns.size == 0 or drawdowns[0] != 0.0:
        return np.nan, np.nan, np.nan, -1.0
    deepest = 0.0
    latest_peak = 0.0
    peak = trough = recovery = np.nan
    for index in range(drawdowns.size):
        value = drawdowns[index]
        if not math.isfinite(value) or value > 0.0 or value < -1.0:
            return np.nan, np.nan, np.nan, -1.0
        if value == 0.0:
            latest_peak = float(index)
            if math.isfinite(trough) and not math.isfinite(recovery):
                recovery = float(index)
        elif value <= deepest:
            # Exact ties select the last trough, including within one episode.
            deepest = value
            peak = latest_peak
            trough = float(index)
            recovery = np.nan
    return peak, trough, recovery, 1.0 if deepest < 0.0 else 0.0


@numba.njit(types.float64(_INTERVAL), nogil=True, cache=False)
def interval_start_kernel(interval):
    return interval[0]


@numba.njit(types.float64(_INTERVAL), nogil=True, cache=False)
def interval_trough_kernel(interval):
    return interval[1]


@numba.njit(types.float64(_INTERVAL), nogil=True, cache=False)
def interval_recovery_kernel(interval):
    return interval[2]


@numba.njit(types.float64(types.float64, types.float64), nogil=True, cache=False)
def days_between_kernel(start, end):
    if not math.isfinite(start) or not math.isfinite(end) or start != math.floor(start) or end != math.floor(end) or end < start:
        return np.nan
    return end - start


INTERVAL_KERNELS = {
    "last_drawdown_interval": last_drawdown_interval_kernel,
    "interval_start": interval_start_kernel,
    "interval_trough": interval_trough_kernel,
    "interval_recovery": interval_recovery_kernel,
    "days_between": days_between_kernel,
}
for _kernel in INTERVAL_KERNELS.values():
    _kernel.disable_compile()


def _infer_interval(inputs):
    value = inputs[0]
    if value.kind != "series" or not value.is_numeric:
        raise TypedDslError("TYPE_MISMATCH", "请选择回撤序列。")
    if value.semantic_dimension != "return_decimal":
        raise TypedDslError("SEMANTIC_MISMATCH", "区间识别需要回撤序列，不是净值；请先连接回撤序列算子。")
    return INTERVAL_TYPE


def _infer_field(inputs):
    if inputs[0] != INTERVAL_TYPE:
        raise TypedDslError("TYPE_MISMATCH", "请选择最后一次最大回撤区间。")
    return ValueType.scalar(semantic_dimension="count")


def _infer_days(inputs):
    if any(value.kind != "scalar" or value.semantic_dimension != "date" for value in inputs):
        raise TypedDslError("TYPE_MISMATCH", "日期差需要两个日期，不能用普通数值代替。")
    return ValueType.scalar(semantic_dimension="calendar_days")


def interval_operator_specs(version):
    from .typed_operators import OperatorSignature, TypedOperatorSpec
    specs = [TypedOperatorSpec(
        "last_drawdown_interval", version, "path",
        (OperatorSignature(("series<time>[L]",), "record", "last exact maximum; one interval"),),
        "在回撤序列中选择最后一次最大回撤；同深度取最后谷底，所有日期和时长引用同一区间。",
        _infer_interval, last_drawdown_interval_kernel, cost_model="scan",
        cost=lambda inputs, output: str(inputs[0].shape[0]),
    )]
    for name, label in (("interval_start", "区间峰值位置"), ("interval_trough", "区间谷底位置"), ("interval_recovery", "区间恢复位置")):
        specs.append(TypedOperatorSpec(
            name, version, "path", (OperatorSignature(("record",), "scalar", "read one interval position"),),
            label, _infer_field, INTERVAL_KERNELS[name], cost_model="constant",
        ))
    specs.append(TypedOperatorSpec(
        "days_between", version, "path", (OperatorSignature(("scalar", "scalar"), "scalar", "elapsed calendar days, not observations"),),
        "结束日期减开始日期，返回自然日间隔；缺少任一日期不可计算。", _infer_days, days_between_kernel, cost_model="constant",
    ))
    return tuple(specs)

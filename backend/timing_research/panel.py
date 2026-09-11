"""Small ETF baskets: layout adapters over existing numeric/condition kernels.

The physical layout is member × time, so mapping an existing series plan uses
contiguous row views. There is no second rolling, comparison or mean algorithm.
"""
import numpy as np
from numba import float64, int64, njit, types

from cal_indicators.typed_numba_kernels import axis_reduce_time_fixed
from historical_regimes.condition_numba import condition_compare_kernel
from historical_regimes.v2_numba import unary_transform_kernel
from backend.compute_policy import validate_execution_audit
from .numeric import condition_values_kernel

F = types.Array(float64, 1, "A", readonly=True)
M = types.Array(float64, 2, "A", readonly=True)
IM = types.Array(int64, 2, "A", readonly=True)
FO, MO, IMO = float64[::1], float64[:, ::1], int64[:, ::1]
PANEL_VERSION = "timing-fixed-basket-njit/1.0.0"


@njit(MO(M, int64), cache=True, nogil=True)
def panel_lag_kernel(values, periods):
    if not 2 <= values.shape[0] <= 12 or periods < 1:
        raise ValueError("Basket requires 2-12 members and a positive lag.")
    output = np.empty(values.shape, dtype=np.float64)
    for member in range(values.shape[0]):
        output[member] = unary_transform_kernel(values[member], 4, periods)
    return output


@njit(IMO(M, M, float64, int64), cache=True, nogil=True)
def panel_compare_kernel(left, right, threshold, opcode):
    if not 2 <= left.shape[0] <= 12:
        raise ValueError("Basket requires 2-12 members.")
    has_bounds = right.shape != (0, 0)
    if has_bounds and right.shape != left.shape:
        raise ValueError("Basket axes must match.")
    output = np.empty(left.shape, dtype=np.int64)
    empty = np.empty(0, dtype=np.float64)
    for member in range(left.shape[0]):
        output[member] = condition_compare_kernel(left[member], right[member] if has_bounds else empty, threshold, opcode)
    return output


@njit(FO(M), cache=True, nogil=True)
def cross_mean_kernel(values):
    # The shared time-axis reducer reduces rows in our member × time layout.
    # Its ordinary mean preserves any NaN, i.e. a fixed complete-member basket.
    if not 2 <= values.shape[0] <= 12:
        raise ValueError("Basket requires 2-12 members.")
    output = axis_reduce_time_fixed(121, values)
    # Infinity is not an observed member value. Do not let its row order decide
    # whether the shared arithmetic produces Inf or NaN; completeness is causal.
    for t in range(values.shape[1]):
        for member in range(values.shape[0]):
            if not np.isfinite(values[member, t]):
                output[t] = np.nan
                break
    return output


@njit(FO(IM), cache=True, nogil=True)
def breadth_kernel(conditions):
    if not 2 <= conditions.shape[0] <= 12:
        raise ValueError("Basket requires 2-12 members.")
    values = np.empty(conditions.shape, dtype=np.float64)
    for member in range(conditions.shape[0]):
        values[member] = condition_values_kernel(conditions[member])
    return cross_mean_kernel(values)


PANEL_KERNELS = {"panel_lag": panel_lag_kernel, "panel_compare": panel_compare_kernel,
                 "cross_mean": cross_mean_kernel, "breadth": breadth_kernel}
for kernel in PANEL_KERNELS.values():
    kernel.disable_compile()


def warm_panel_kernels():
    for kernel in PANEL_KERNELS.values():
        if (not kernel.nopython_signatures or len(kernel.signatures) != len(kernel.nopython_signatures)
                or kernel._can_compile):
            raise RuntimeError("ETF 篮子计算内核尚未完成固定签名预热。")
    return {"complete": True, **validate_execution_audit({
        "execution_backend": "numba_njit_fixed_signature", "engine_version": PANEL_VERSION,
        "nopython": True, "python_fallback": 0, "python_operator_calls": 0,
        "request_time_compilation": 0,
        "kernel_signatures": {name: [str(s) for s in kernel.signatures] for name, kernel in PANEL_KERNELS.items()},
    })}

"""Fixed-signature weighting of declared CMA marginal uncertainty half-widths."""
from __future__ import annotations

import os
import numpy as np
from numba import float64, njit, types

V = types.Array(float64, 1, "A", readonly=True)
M = types.Array(float64, 2, "A", readonly=True)
_WARMED_PID = None


@njit((V, M), cache=True, nogil=True)
def weighted_half_width_kernel(weights, half_widths):
    count, assets = half_widths.shape
    if count < 1 or count > 20 or assets < 1 or assets > 30 or weights.size != count:
        raise ValueError("MULTI_CMA_AXIS")
    total = 0.0
    result = np.zeros(assets, dtype=np.float64)
    for m in range(count):
        if not np.isfinite(weights[m]) or weights[m] < 0 or weights[m] > 1:
            raise ValueError("MULTI_CMA_WEIGHT")
        total += weights[m]
        for a in range(assets):
            if not np.isfinite(half_widths[m, a]) or half_widths[m, a] < 0:
                raise ValueError("MULTI_CMA_UNCERTAINTY")
            result[a] += weights[m] * half_widths[m, a]
    if abs(total - 1.0) > 1e-8:
        raise ValueError("MULTI_CMA_WEIGHT")
    return result


weighted_half_width_kernel.disable_compile()


def execution_audit():
    kernel = weighted_half_width_kernel
    complete = (_WARMED_PID == os.getpid() and len(kernel.signatures) == len(kernel.nopython_signatures) == 1
                and not kernel._can_compile and not any(x.objectmode for x in kernel.overloads.values()))
    return {"kernel_version": "multi-cma-half-width/1.0.0", "complete": bool(complete),
            "nopython": bool(complete), "python_fallback": 0, "request_time_compilation": 0,
            "kernel_signatures": {kernel.__name__: [str(s) for s in kernel.signatures]}}


def require_ready():
    if not execution_audit()["complete"]:
        raise RuntimeError("多 CMA 数值内核尚未完成本进程预热。")


def warm():
    global _WARMED_PID
    _WARMED_PID = None
    weights = np.array([0.4, 0.6])[::-1]
    widths = np.array([[0.02, 0.01], [0.01, 0.03]])[::-1, ::-1]
    weights.flags.writeable = widths.flags.writeable = False
    weighted_half_width_kernel(weights, widths)
    _WARMED_PID = os.getpid()
    if not execution_audit()["complete"]:
        _WARMED_PID = None
        raise RuntimeError("多 CMA 数值内核预热失败。")
    return execution_audit()

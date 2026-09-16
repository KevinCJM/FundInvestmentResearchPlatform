"""Forward-only paired blocks; conservative gates, not confidence intervals."""
import os
import threading

import numpy as np
from numba import njit, types

from .kernels import I, F

_LOCK = threading.Lock()
_PID = None


@njit(cache=True)
def forward_blocks_kernel(days, y, pred, q, base, block_size, max_gap):
    """Keep the complete date axis. Missing pairs invalidate whole fixed blocks."""
    n, k = q.shape
    if (len(days) != n or len(y) != n or len(pred) != n or len(base) != k
            or not 2 <= k <= 12 or not 2 <= block_size <= 252 or max_gap < 1):
        raise ValueError("Invalid forward block dimensions")
    if not np.isfinite(base).all() or (base < 0).any() or abs(base.sum() - 1.) > 1e-8:
        raise ValueError("Invalid frozen class base")
    valid = np.zeros(n, np.uint8)
    support = np.zeros((2, k), np.int64)
    cycles = np.zeros(k, np.int64)
    delta = np.full(n, np.nan)
    for i in range(n):
        if i and days[i] <= days[i-1]:
            raise ValueError("Date axis must increase")
        if not (0 <= y[i] < k and 0 <= pred[i] < k):
            continue
        if (not np.isfinite(q[i]).all() or (q[i] < 0).any() or (q[i] > 1).any()
                or abs(q[i].sum() - 1.) > 1e-8):
            continue
        valid[i] = 1
        support[0, y[i]] += 1
        support[1, pred[i]] += 1
        value = 0.
        for c in range(k):
            target = 1. if y[i] == c else 0.
            value += (base[c] - target) ** 2 - (q[i, c] - target) ** 2
        delta[i] = value
    # A complete regime has observed transitions on BOTH sides and no gap.
    start = 0
    while start < n:
        end = start + 1
        while end < n and y[end] == y[start]:
            end += 1
        complete = start > 0 and end < n and y[start] >= 0
        if complete:
            for i in range(start-1, end+1):
                if not valid[i] or (i > start-1 and days[i] - days[i-1] > max_gap):
                    complete = False
            if complete:
                cycles[y[start]] += 1
        start = end
    blocks = np.full(n // block_size, np.nan)
    for b in range(len(blocks)):
        start = b * block_size
        good = True
        total = 0.
        for i in range(start, start + block_size):
            if not valid[i] or (i > start and days[i] - days[i-1] > max_gap):
                good = False
            total += delta[i]
        if good:
            blocks[b] = total / block_size
    complete_blocks = 0
    worst = np.inf
    for value in blocks:
        if np.isfinite(value):
            complete_blocks += 1
            worst = min(worst, value)
    pairs = int(valid.sum())
    coverage = pairs / n if n else np.nan
    return support, cycles, blocks, pairs, coverage, complete_blocks, worst


_SIGNATURE = (I, I, I, F, types.Array(types.float64, 1, "A", readonly=True),
              types.int64, types.int64)


def warm():
    global _PID
    with _LOCK:
        if not forward_blocks_kernel.signatures:
            forward_blocks_kernel.compile(_SIGNATURE)
        forward_blocks_kernel.disable_compile()
        labels = np.empty(0, np.int64)
        q = np.empty((0, 2), np.float64)
        base = np.array([.5, .5])
        for value in (labels, q, base):
            value.flags.writeable = False
        forward_blocks_kernel(labels, labels, labels, q, base, 2, 7)
        _PID = os.getpid()
    return audit()


def audit():
    kernel = forward_blocks_kernel
    if (_PID != os.getpid() or len(kernel.signatures) != 1
            or len(kernel.nopython_signatures) != 1 or kernel._can_compile):
        raise RuntimeError("PROSPECTIVE_RUNTIME_NOT_READY")
    return {"complete": True, "warmed_pid": _PID, "python_fallback": 0,
            "request_time_compilation": 0, "signature": str(kernel.nopython_signatures[0])}

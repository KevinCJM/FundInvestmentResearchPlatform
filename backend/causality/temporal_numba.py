"""Production timestamp-aware versions of the shared P1 comparison primitives."""
import numpy as np
from numba import njit, types, float64, int64
from .probes import TIGHT_RTOL, TIGHT_ATOL, LOOSE_RTOL, LOOSE_ATOL

F2 = float64[:, ::1]
I = int64[::1]


@njit(F2(F2, I, int64, int64), cache=True)
def perturb_available_tail(values, available, cutoff, style):
    out = values.copy()
    indices = np.empty(available.size, dtype=np.int64)
    count = 0
    for i in range(available.size):
        if available[i] > cutoff:
            indices[count] = i
            count += 1
    for k in range(count):
        i = indices[k]
        other = indices[count - k - 1] if style != 1 else i
        factor = 1.9 if style == 0 else 6.0 if style == 1 else 0.013
        for j in range(values.shape[1]):
            if np.isfinite(values[i, j]) and np.isfinite(values[other, j]):
                out[i, j] = values[other, j] * factor
    return out


@njit(types.UniTuple(int64, 3)(F2, I, I, F2, I, I, int64, int64), cache=True)
def compare_available_prefix(base, dates, available, candidate, candidate_dates, candidate_available, cutoff, exact):
    """(0 pass/1 leak/2 grey, comparable valid cells, first mismatching index)."""
    count, grey, first_grey = 0, 0, -1
    for i in range(dates.size):
        if dates[i] > cutoff or available[i] > cutoff:
            continue
        pos = np.searchsorted(candidate_dates, dates[i])
        if pos >= candidate_dates.size or candidate_dates[pos] != dates[i]:
            return 1, count, i
        if candidate.shape[1] != base.shape[1] or candidate_available[pos] > cutoff:
            return 1, count, i
        for j in range(base.shape[1]):
            a, b = base[i, j], candidate[pos, j]
            if np.isnan(a) and np.isnan(b):
                continue
            if a == b:
                if np.isfinite(a) and (exact == 0 or a >= 0):
                    count += 1
                continue
            if not np.isfinite(a) or not np.isfinite(b) or exact:
                return 1, count, i
            count += 1
            delta = abs(a - b)
            if delta > LOOSE_ATOL + LOOSE_RTOL * abs(b):
                return 1, count, i
            if delta > TIGHT_ATOL + TIGHT_RTOL * abs(b):
                grey, first_grey = 2, i
    return grey, count, first_grey


@njit(I(I, int64), cache=True)
def available_positions(available, cutoff):
    count = 0
    for v in available:
        if v <= cutoff:
            count += 1
    out = np.empty(count, dtype=np.int64)
    k = 0
    for i in range(available.size):
        if available[i] <= cutoff:
            out[k] = i
            k += 1
    return out


@njit(float64(F2, F2), cache=True)
def latest_relative_difference(base, candidate):
    if base.shape[0] == 0 or candidate.shape[0] == 0 or base.shape[1] != candidate.shape[1]:
        return np.nan
    worst, seen = 0.0, False
    for j in range(base.shape[1]):
        a, b = base[-1, j], candidate[-1, j]
        if np.isnan(a) and np.isnan(b):
            continue
        if not np.isfinite(a) or not np.isfinite(b):
            return np.nan
        worst = max(worst, abs(a - b) / max(abs(a), 1e-12))
        seen = True
    return worst if seen else np.nan


TEMPORAL_KERNELS = {"temporal_perturb": perturb_available_tail, "temporal_compare": compare_available_prefix,
                    "temporal_positions": available_positions, "temporal_warmup": latest_relative_difference}
for kernel in TEMPORAL_KERNELS.values():
    kernel.disable_compile()

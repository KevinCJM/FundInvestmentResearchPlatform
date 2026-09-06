"""Fixed-signature NJIT data-quality and tolerance checks; no lazy fallback."""
from __future__ import annotations
import numpy as np
from numba import njit, types


@njit(types.int8[::1](types.float64[:, ::1], types.boolean[::1], types.float64[::1], types.float64[::1], types.boolean[::1], types.int64[::1], types.float64[::1], types.int64, types.float64), cache=True, nogil=True)
def quality_flags(values, required, minimum, maximum, positive, ohlc, previous, jump_column, jump_limit):
    flags = np.zeros(values.shape[0], dtype=np.int8)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            if np.isnan(value):
                if required[j]:
                    flags[i] |= 1
                continue
            if not np.isfinite(value) or value < minimum[j] or value > maximum[j] or (positive[j] and value <= 0):
                flags[i] |= 2
        if len(ohlc) == 4:
            opening, high, low, close = values[i, ohlc[0]], values[i, ohlc[1]], values[i, ohlc[2]], values[i, ohlc[3]]
            if np.isfinite(high) and np.isfinite(low) and high < low:
                flags[i] |= 4
            for price in (opening, close):
                if np.isfinite(price) and ((np.isfinite(high) and price > high) or (np.isfinite(low) and price < low)):
                    flags[i] |= 4
        if jump_column >= 0 and jump_limit > 0 and np.isfinite(previous[i]) and previous[i] > 0:
            value = values[i, jump_column]
            if np.isfinite(value) and abs(value / previous[i] - 1.0) > jump_limit:
                flags[i] |= 8
    return flags


@njit(types.boolean[::1](types.float64[::1], types.float64[::1], types.float64[::1], types.float64[::1]), cache=True, nogil=True)
def difference_flags(left, right, absolute, relative):
    output = np.zeros(left.size, dtype=np.bool_)
    for j in range(left.size):
        if np.isfinite(left[j]) and np.isfinite(right[j]):
            tolerance = max(absolute[j], relative[j] * max(abs(left[j]), abs(right[j])))
            output[j] = abs(left[j] - right[j]) > tolerance
    return output


def warm_resolution_kernels() -> dict:
    # Signatures are compiled eagerly at import, before the application can serve.
    return {"complete": bool(quality_flags.nopython_signatures and difference_flags.nopython_signatures),
            "python_fallback": 0, "request_time_compilation": 0}

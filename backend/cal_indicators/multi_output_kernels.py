"""Fixed-signature primitives that produce several related scalar results."""
from __future__ import annotations

import math

import numba
import numpy as np
from numba import types


@numba.njit(
    types.UniTuple(types.float64, 4)(types.Array(types.float64, 1, "C", readonly=True)),
    nogil=True,
    cache=False,
)
def drawdown_analysis_kernel(nav: np.ndarray) -> tuple[float, float, float, float]:
    """One scan, O(1) auxiliary space; durations count observation intervals.

    Equal maxima use the first trough and the latest preceding equal peak.
    Recovery is the first subsequent value at or above that episode's peak.
    An unrecovered worst drawdown has NaN recovery duration, not zero.
    """
    if nav.size == 0:
        return np.nan, np.nan, np.nan, np.nan
    peak = nav[0]
    if not math.isfinite(peak) or peak <= 0.0:
        return np.nan, np.nan, np.nan, np.nan
    peak_index = 0
    worst = 0.0
    worst_peak = 0.0
    worst_trough_index = 0
    decline = 0.0
    recovery = 0.0
    longest = 0.0
    pending_recovery = False
    for index in range(1, nav.size):
        value = nav[index]
        if not math.isfinite(value) or value <= 0.0:
            return np.nan, np.nan, np.nan, np.nan
        if pending_recovery and value >= worst_peak:
            recovery = float(index - worst_trough_index)
            pending_recovery = False
        if value >= peak:
            # Count the recovery interval only when the preceding point was
            # underwater; a continuously rising series has duration zero.
            if index > peak_index + 1:
                longest = max(longest, float(index - peak_index))
            peak = value
            peak_index = index
        else:
            longest = max(longest, float(index - peak_index))
            depth = 1.0 - value / peak
            if depth > worst:
                worst = depth
                worst_peak = peak
                worst_trough_index = index
                decline = float(index - peak_index)
                recovery = np.nan
                pending_recovery = True
    return worst, decline, recovery, longest


drawdown_analysis_kernel.disable_compile()

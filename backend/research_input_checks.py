"""Shared input checks for allocation research; never repair prices by guessing.

The quantity-scale rule is a review gate, not a comprehensive quality certificate.
Fixed signatures accept read-only strided views; scans allocate bounded summaries.
"""
from __future__ import annotations

import numpy as np
from numba import float64, int64, njit, types

M = types.Array(float64, 2, "A", readonly=True)
IM = types.Array(int64, 2, "A", readonly=True)
I = types.Array(int64, 1, "A", readonly=True)


@njit(types.Tuple((int64[::1], int64[::1]))(M), cache=True, nogil=True)
def return_scale_breaks_kernel(returns):
    """First suspect interval and count per asset; ratio <= .1 or >= 10."""
    first = np.full(returns.shape[1], -1, dtype=np.int64)
    counts = np.zeros(returns.shape[1], dtype=np.int64)
    for t in range(returns.shape[0]):
        for a in range(returns.shape[1]):
            value = returns[t, a]
            if np.isfinite(value) and (value <= -.9 or value >= 9.):
                if first[a] < 0:
                    first[a] = t
                counts[a] += 1
    return first, counts


@njit(types.Tuple((int64, int64, int64, int64))(IM, I, int64), cache=True, nogil=True)
def training_readiness_kernel(available, days, cutoff):
    """Count immature/unknown labels and suggest a clock-only feasible split."""
    if available.shape[0] != days.size:
        raise ValueError("Knowledge dates must align with returns.")
    future, unknown, first_future, latest = 0, 0, -1, -1
    best_day, best_distance = -1, 9223372036854775807
    for t in range(days.size):
        for a in range(available.shape[1]):
            value = available[t, a]
            latest = max(latest, value)
            if days[t] <= cutoff:
                if value < 0:
                    unknown += 1
                elif value > cutoff:
                    future += 1
                    if first_future < 0 or value < first_future:
                        first_future = value
        # At least 20 returns on each side. Only clocks, never returns, select
        # the nearest possible boundary; the next observation belongs to OOS.
        if t >= 19 and t < days.size - 20 and latest < days[t + 1]:
            candidate = max(days[t], latest)
            distance = abs(candidate - cutoff)
            if distance < best_distance:
                best_day, best_distance = candidate, distance
    return future, unknown, first_future, best_day


for _kernel in (return_scale_breaks_kernel, training_readiness_kernel):
    _kernel.disable_compile()


def warm_research_input_checks():
    values = np.zeros((40, 2), dtype=np.float64)
    available = np.zeros((40, 2), dtype=np.int64)
    days = np.arange(40, dtype=np.int64)
    return_scale_breaks_kernel(values)
    training_readiness_kernel(available, days, 20)
    if not all(len(k.nopython_signatures) == 1 for k in (return_scale_breaks_kernel, training_readiness_kernel)):
        raise RuntimeError("Research input checks require fixed NJIT signatures.")


def return_quality(returns, dates, assets):
    values = np.asarray(returns, dtype=np.float64)
    if values.ndim != 2 or values.shape != (len(dates), len(assets)):
        raise ValueError("Quality input axes do not match.")
    first, counts = return_scale_breaks_kernel(values)
    issues = []
    for a, asset in enumerate(assets):
        t = int(first[a])
        if t >= 0:
            change = float(values[t, a])
            day = str(dates[t])[:10]
            issues.append({"asset_id": str(asset), "date": day, "value": change,
                           "count": int(counts[a]), "code": "NAV_SCALE_BREAK",
                           "message": f"{asset} 在 {day} 的净值变动为 {change:.2%}，疑似数值尺度断点；请核查复权口径或替换产品后重新构建。"})
    return {"status": "blocked" if issues else "clear", "issues": issues,
            "rule": "adjacent_nav_ratio_lte_0.1_or_gte_10", "certifies_all_data_quality": False}


class ResearchInputError(ValueError):
    def __init__(self, message, *, code="ALLOCATION_INPUT_INVALID", diagnostics=None):
        super().__init__(message)
        self.code, self.diagnostics = code, diagnostics or []

    def detail(self):
        return {"code": self.code, "message": str(self), "diagnostics": self.diagnostics}


def require_return_quality(returns, dates, assets):
    quality = return_quality(returns, dates, assets)
    if quality["issues"]:
        raise ResearchInputError(quality["issues"][0]["message"], code="NAV_SCALE_BREAK", diagnostics=quality["issues"])
    return quality

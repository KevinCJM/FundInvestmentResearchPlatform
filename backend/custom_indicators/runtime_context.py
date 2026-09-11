"""Shared fixed-signature context values for scalar and time-series indicators."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numba
import numpy as np
import pandas as pd


@numba.njit(
    numba.types.UniTuple(numba.float64, 5)(numba.float64, numba.float64),
    cache=True,
    nogil=True,
)
def risk_free_context_kernel(
    annual_percent: float,
    elapsed_days: float,
) -> tuple[float, float, float, float, float]:
    """Return annual, per-observation, legacy, window and annualization values."""

    annual = annual_percent / 100.0
    base = max(0.0, 1.0 + annual)
    per_observation = base ** (1.0 / 252.0) - 1.0
    elapsed = max(0.0, elapsed_days)
    window_return = base ** (elapsed / 365.0) - 1.0
    return annual, per_observation, per_observation, window_return, 252.0


@numba.njit(
    numba.types.UniTuple(numba.float64[::1], 2)(numba.float64[::1]),
    cache=True,
    nogil=True,
)
def aligned_return_series_kernel(
    levels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return date-aligned simple/log returns with the first row as NaN."""

    size = levels.size
    simple = np.full(size, np.nan, dtype=np.float64)
    logarithmic = np.full(size, np.nan, dtype=np.float64)
    for index in range(1, size):
        previous = levels[index - 1]
        current = levels[index]
        if not np.isfinite(previous) or not np.isfinite(current):
            continue
        if previous <= 0.0 or current <= 0.0:
            continue
        ratio = current / previous
        simple[index] = ratio - 1.0
        logarithmic[index] = np.log(ratio)
    return simple, logarithmic


RUNTIME_SCALAR_CONTEXT_NAMES = frozenset(
    {
        "annual_risk_free_rate_decimal",
        "observation_count",
        "periods_per_year",
        "risk_free_rate_per_observation",
        "risk_free_rate_per_period",
        "risk_free_return_window",
        "window_elapsed_days",
    }
)


def single_product_scalar_context(
    definition: Mapping[str, Any],
    dates: Sequence[Any],
    *,
    returns: np.ndarray | None = None,
) -> dict[str, float]:
    """Resolve the scalar context shared by scalar and series typed plans.

    This function performs boundary assembly only.  Risk-free compounding is
    delegated to the explicitly compiled NJIT kernel above.
    """

    if len(dates) > 0:
        start = pd.Timestamp(dates[0])
        end = pd.Timestamp(dates[-1])
        elapsed_days = float(max(0, (end - start).days))
    else:
        elapsed_days = 0.0
    annual, per_period, legacy_per_period, window_return, periods_per_year = (
        risk_free_context_kernel(
            float(definition.get("annual_risk_free_rate_percent") or 0.0),
            elapsed_days,
        )
    )
    if returns is None:
        observation_count = float(max(0, len(dates) - 1))
    else:
        array = np.asarray(returns, dtype=np.float64)
        observation_count = float(np.count_nonzero(np.isfinite(array)))
    return {
        "annual_risk_free_rate_decimal": float(annual),
        "observation_count": observation_count,
        "periods_per_year": float(periods_per_year),
        "risk_free_rate_per_observation": float(per_period),
        "risk_free_rate_per_period": float(legacy_per_period),
        "risk_free_return_window": float(window_return),
        "window_elapsed_days": elapsed_days,
    }


__all__ = [
    "RUNTIME_SCALAR_CONTEXT_NAMES",
    "aligned_return_series_kernel",
    "risk_free_context_kernel",
    "single_product_scalar_context",
]

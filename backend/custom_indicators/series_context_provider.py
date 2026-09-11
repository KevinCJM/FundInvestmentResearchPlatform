"""Adapt chart-series loading for derived returns used by rolling indicators."""

from __future__ import annotations

import inspect
import math
from typing import Any

import numpy as np
import pandas as pd

from .runtime_context import RUNTIME_SCALAR_CONTEXT_NAMES
from .series_provider import load_product_chart_series as _load_product_chart_series


_DERIVED_RETURN_SERIES = frozenset({"returns", "log_returns"})


def _variable_argument(bound: inspect.BoundArguments) -> str | None:
    for name, value in bound.arguments.items():
        if "variable" not in name.lower():
            continue
        if isinstance(value, (list, tuple, set, frozenset)):
            return name
    return None


def _aligned_return_arrays(levels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    simple = np.full(levels.size, np.nan, dtype=np.float64)
    logarithmic = np.full(levels.size, np.nan, dtype=np.float64)
    if levels.size < 2:
        return simple, logarithmic
    previous = levels[:-1]
    current = levels[1:]
    valid = (
        np.isfinite(previous)
        & np.isfinite(current)
        & (previous > 0.0)
        & (current > 0.0)
    )
    positions = np.flatnonzero(valid) + 1
    ratios = current[valid] / previous[valid]
    simple[positions] = ratios - 1.0
    logarithmic[positions] = np.log(ratios)
    return simple, logarithmic


def load_product_chart_series(*args: Any, **kwargs: Any) -> Any:
    """Load physical chart fields, then derive return series on the same axis.

    The wrapped provider remains the single owner of active-snapshot resolution,
    product/date filtering, lineage and display-window selection. This adapter
    merely removes runtime scalar names from physical I/O and materializes
    returns from adjacent adjusted NAV values with a leading null boundary.
    """

    signature = inspect.signature(_load_product_chart_series)
    bound = signature.bind_partial(*args, **kwargs)
    variable_name = _variable_argument(bound)
    requested = tuple(bound.arguments.get(variable_name, ())) if variable_name else ()
    derived = tuple(name for name in requested if name in _DERIVED_RETURN_SERIES)
    physical = [
        name
        for name in requested
        if name not in _DERIVED_RETURN_SERIES
        and name not in RUNTIME_SCALAR_CONTEXT_NAMES
    ]
    if derived and "adjusted_nav" not in physical:
        physical.append("adjusted_nav")
    if variable_name is not None:
        original = bound.arguments[variable_name]
        if isinstance(original, tuple):
            bound.arguments[variable_name] = tuple(physical)
        elif isinstance(original, set):
            bound.arguments[variable_name] = set(physical)
        elif isinstance(original, frozenset):
            bound.arguments[variable_name] = frozenset(physical)
        else:
            bound.arguments[variable_name] = list(physical)

    # One extra adjusted-NAV observation is required to create the first return
    # that enters an N-observation rolling window.
    if derived:
        for name in (
            "lookback_observations",
            "history_observations",
            "lookback",
        ):
            if name in bound.arguments and bound.arguments[name] is not None:
                try:
                    bound.arguments[name] = int(bound.arguments[name]) + 1
                except (TypeError, ValueError):
                    pass
                break

    loaded = _load_product_chart_series(*bound.args, **bound.kwargs)
    if not derived:
        return loaded

    frame = getattr(loaded, "frame", None)
    if not isinstance(frame, pd.DataFrame) or "adjusted_nav" not in frame.columns:
        return loaded
    levels = frame["adjusted_nav"].to_numpy(dtype=np.float64, copy=False)
    simple, logarithmic = _aligned_return_arrays(levels)
    if "returns" in derived:
        frame["returns"] = simple
    if "log_returns" in derived:
        frame["log_returns"] = logarithmic

    dates_by_variable = getattr(loaded, "dates_by_variable", None)
    if isinstance(dates_by_variable, dict):
        date_values = None
        for date_column in ("date", "trade_date", "nav_date"):
            if date_column in frame.columns:
                date_values = tuple(pd.to_datetime(frame[date_column]).tolist())
                break
        if date_values is None:
            anchor_dates = dates_by_variable.get("adjusted_nav")
            if anchor_dates is not None:
                date_values = tuple(anchor_dates)
        if date_values is not None:
            for name in derived:
                dates_by_variable[name] = date_values

    lineage = getattr(loaded, "lineage", None)
    if isinstance(lineage, list):
        for name in derived:
            lineage.append(
                {
                    "variable_id": name,
                    "source": "derived_from_adjusted_nav",
                    "formula": (
                        "adjusted_nav[t] / adjusted_nav[t-1] - 1"
                        if name == "returns"
                        else "LN(adjusted_nav[t] / adjusted_nav[t-1])"
                    ),
                    "missing_policy": "preserve_null",
                }
            )
    return loaded


__all__ = ["load_product_chart_series"]

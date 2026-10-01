"""Fixed-signature NJIT summary for approved derived result series.

Model-visible series summaries (coverage, missingness and the permitted
mean/std/range of a *derived* result) are computed by one compiled kernel with
an explicit signature.  It is compiled and warmed when this module is imported,
never during a request; there is no Python fallback.  The numeric boundary is
float64 (JSON ``null``/NaN and non-finite values count as missing), and the
input array is only read.

See docs/governance/numeric-computing.md: this is an agent-side result-summary
kernel, not a business algorithm, and it does not change any existing service.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np
from numba import njit, types

SUMMARY_SIGNATURE = types.Tuple([types.float64] * 8)(types.Array(types.float64, 1, "C", readonly=True))
KERNEL_VERSION = "agent-series-summary/1"
# Field order of the returned tuple.
SUMMARY_FIELDS = ("point_count", "finite_count", "null_count", "zero_count",
                  "mean", "std", "minimum", "maximum")


@njit(SUMMARY_SIGNATURE, cache=True, nogil=True)
def series_summary(values):  # pragma: no cover - compiled
    n = values.size
    finite = 0
    nulls = 0
    zeros = 0
    mean = 0.0
    m2 = 0.0
    minimum = math.inf
    maximum = -math.inf
    for index in range(n):
        item = values[index]
        if math.isnan(item) or math.isinf(item):
            nulls += 1
            continue
        finite += 1
        if item == 0.0:
            zeros += 1
        # Weighted running mean: avoids both a growing sum and a growing product.
        previous_mean = mean
        weight = 1.0 / finite
        mean = mean * (1.0 - weight) + item * weight
        m2 += (item - previous_mean) * (item - mean)
        if item < minimum:
            minimum = item
        if item > maximum:
            maximum = item
    if finite == 0:
        return (float(n), 0.0, float(nulls), float(zeros), math.nan, math.nan, math.nan, math.nan)
    std = math.sqrt(m2 / (finite - 1)) if finite > 1 and m2 > 0.0 else 0.0
    if not math.isfinite(mean):
        mean = math.nan
    if not math.isfinite(std):
        std = math.nan
    return (float(n), float(finite), float(nulls), float(zeros), mean, std,
            minimum if math.isfinite(minimum) else math.nan,
            maximum if math.isfinite(maximum) else math.nan)


_WARM_VALUE = np.asarray([0.0, 1.0, math.nan], dtype=np.float64)
_WARM_VALUE.flags.writeable = False


def warm() -> None:
    """Compile the fixed signature before any request can use it."""

    series_summary(_WARM_VALUE)
    series_summary.disable_compile()
    if len(series_summary.nopython_signatures) != 1 or series_summary._can_compile:
        raise RuntimeError("AGENT_SUMMARY_NOT_READY")


warm()


def summarize(values: Any) -> Optional[dict[str, Any]]:
    """Return the model-visible summary of one result channel, or None if unusable.

    ``values`` is the service result list; ``None`` entries are missing values.
    """

    if not isinstance(values, list):
        return None
    array = np.empty(len(values), dtype=np.float64)
    for index, item in enumerate(values):
        if item is None or isinstance(item, bool) or not isinstance(item, (int, float)):
            array[index] = math.nan
        else:
            array[index] = float(item)
    array.flags.writeable = False
    counts = series_summary(array)
    summary: dict[str, Any] = {}
    for name, value in zip(SUMMARY_FIELDS, counts):
        if name in {"point_count", "finite_count", "null_count", "zero_count"}:
            summary[name] = int(value)
        elif math.isnan(value):
            summary[name] = None
        else:
            summary[name] = float(value)
    summary["kernel"] = KERNEL_VERSION
    return summary


def coverage_only(values: Any) -> Optional[dict[str, Any]]:
    """Counts only — for page/client series whose derivation cannot be verified.

    No mean/std/extrema: endpoints and extrema of unverified raw series must not
    become model evidence.
    """

    summary = summarize(values)
    if summary is None:
        return None
    return {key: summary[key] for key in ("point_count", "finite_count", "null_count", "zero_count")}

"""Shared completeness and adjusted-NAV quality rules for period metrics."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as parquet

try:
    from backend.instrument_analytics_numba import (
        NAT_DAY,
        adjusted_nav_anomaly_mask_kernel,
        count_true_kernel,
        coverage_ratio_kernel,
        finite_mask_kernel,
        instrument_analytics_numba_execution_audit,
        period_window_quality_kernel,
        warm_instrument_analytics_numba_kernels,
    )
except ModuleNotFoundError:  # pragma: no cover - backend/ direct execution
    from instrument_analytics_numba import (
        NAT_DAY,
        adjusted_nav_anomaly_mask_kernel,
        count_true_kernel,
        coverage_ratio_kernel,
        finite_mask_kernel,
        instrument_analytics_numba_execution_audit,
        period_window_quality_kernel,
        warm_instrument_analytics_numba_kernels,
    )


WINDOW_START_TOLERANCE_DAYS = 10
MIN_OPEN_DAY_COVERAGE = 0.90
FALLBACK_BUSINESS_DAY_COVERAGE = 0.80
MAX_CONSECUTIVE_MISSING_OPEN_DAYS = 5
MAX_CONSECUTIVE_MISSING_FALLBACK_DAYS = 10
ADJ_NAV_DISLOCATION_THRESHOLD = 0.20
REFERENCE_NAV_STABLE_THRESHOLD = 0.10
EXTREME_ADJ_NAV_RETURN_THRESHOLD = 1.0

_QUALITY_REASON_BY_CODE = {
    0: None,
    1: "insufficient_span",
    2: "start_anchor_too_old",
    3: "adjusted_nav_anomaly",
    4: "insufficient_observations",
    5: "insufficient_density",
    6: "internal_gap",
}


def _float64_array(values: pd.Series) -> np.ndarray:
    return np.array(
        pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64),
        dtype=np.float64,
        copy=True,
        order="C",
    )


def _date_days(values: pd.DatetimeIndex | pd.Series) -> np.ndarray:
    if isinstance(values, pd.Series):
        parsed = pd.DatetimeIndex(values)
    else:
        parsed = pd.DatetimeIndex(values)
    return np.array(
        parsed.to_numpy(dtype="datetime64[D]").view(np.int64),
        dtype=np.int64,
        copy=True,
        order="C",
    )


def finite_coverage(values: pd.Series | np.ndarray) -> tuple[int, float]:
    """Count finite observations and their coverage through the warmed NJIT lane."""

    array = np.ascontiguousarray(
        pd.to_numeric(values, errors="coerce"),
        dtype=np.float64,
    )
    mask = finite_mask_kernel(array)
    count = int(count_true_kernel(mask))
    ratio = float(coverage_ratio_kernel(count, int(array.size)))
    return count, 0.0 if not np.isfinite(ratio) else ratio


@dataclass(frozen=True)
class PeriodWindowQuality:
    complete: bool
    reason: Optional[str]
    target_date: pd.Timestamp
    effective_date: pd.Timestamp
    anchor_date: Optional[pd.Timestamp]
    observation_count: int
    expected_observation_count: int
    coverage_ratio: Optional[float]
    max_consecutive_missing: int
    anomaly_count: int


@lru_cache(maxsize=16)
def _load_sse_open_dates_cached(
    path_text: str,
    size: int,
    modified_ns: int,
) -> pd.DatetimeIndex:
    """Cache the immutable calendar by its file fingerprint.

    Returning the cached ``DatetimeIndex`` is safe because callers only slice it.
    Including size and mtime in the key makes an activated data snapshot invalidate
    the cache without requiring a process restart.
    """

    del size, modified_ns
    path = Path(path_text)
    if not path.exists():
        return pd.DatetimeIndex([])
    source = parquet.ParquetFile(path)
    available = set(source.schema.names)
    date_column = "cal_date" if "cal_date" in available else "date" if "date" in available else None
    if date_column is None or "is_open" not in available:
        return pd.DatetimeIndex([])
    columns = [date_column, "is_open"]
    if "exchange" in available:
        columns.append("exchange")
    frame = source.read(columns=columns, use_threads=False).to_pandas()
    if "exchange" in frame.columns:
        frame = frame[frame["exchange"].astype(str).str.upper().eq("SSE")]
    frame = frame[pd.to_numeric(frame["is_open"], errors="coerce").eq(1)]
    dates = pd.to_datetime(frame[date_column], format="%Y%m%d", errors="coerce").dropna()
    if dates.empty:
        return pd.DatetimeIndex([])
    return pd.DatetimeIndex(dates.dt.normalize().drop_duplicates().sort_values())


def load_sse_open_dates(path: Path) -> pd.DatetimeIndex:
    """Read the SSE calendar once per immutable file generation."""

    source = Path(path).expanduser().resolve()
    if not source.exists():
        return pd.DatetimeIndex([])
    stat = source.stat()
    return _load_sse_open_dates_cached(
        str(source),
        int(stat.st_size),
        int(stat.st_mtime_ns),
    )


def adjusted_nav_anomaly_dates(
    frame: pd.DataFrame,
    *,
    value_column: str = "adj_nav",
    date_column: str = "date",
) -> pd.DatetimeIndex:
    """Locate adjusted-NAV jumps unsupported by unit or accumulated NAV.

    A large adjusted-NAV move is not rejected merely because it is large.  It is
    rejected when a simultaneously available reference NAV remains stable, or
    when the adjusted move is so extreme that it cannot be a plausible daily
    fund return.  This catches provider-side adjustment-factor discontinuities
    without rewriting the source series.
    """

    if frame.empty or date_column not in frame.columns or value_column not in frame.columns:
        return pd.DatetimeIndex([])
    working = frame.copy()
    working[date_column] = pd.to_datetime(working[date_column], errors="coerce")
    working = (
        working.dropna(subset=[date_column])
        .sort_values(date_column)
        .drop_duplicates(subset=[date_column], keep="last")
    )
    if len(working) < 2:
        return pd.DatetimeIndex([])
    adjusted = _float64_array(working[value_column])
    missing_reference = np.full(adjusted.size, np.nan, dtype=np.float64)
    accumulated = (
        _float64_array(working["accum_nav"])
        if "accum_nav" in working.columns
        else missing_reference
    )
    unit = (
        _float64_array(working["unit_nav"])
        if "unit_nav" in working.columns
        else missing_reference
    )
    suspicious = adjusted_nav_anomaly_mask_kernel(
        adjusted,
        accumulated,
        unit,
        ADJ_NAV_DISLOCATION_THRESHOLD,
        REFERENCE_NAV_STABLE_THRESHOLD,
        EXTREME_ADJ_NAV_RETURN_THRESHOLD,
    )
    return pd.DatetimeIndex(
        working.loc[suspicious.astype(bool), date_column].dt.normalize()
    )


def assess_period_window(
    dates: Iterable[object],
    *,
    target_date: pd.Timestamp,
    effective_date: pd.Timestamp,
    open_dates: Optional[pd.DatetimeIndex] = None,
    anomaly_dates: Optional[Iterable[object]] = None,
) -> PeriodWindowQuality:
    """Require a near-boundary anchor and dense coverage across the full period."""

    target = pd.Timestamp(target_date).normalize()
    effective = pd.Timestamp(effective_date).normalize()
    clean_dates = pd.DatetimeIndex(pd.to_datetime(list(dates), errors="coerce")).dropna()
    clean_dates = clean_dates.normalize().drop_duplicates().sort_values()
    clean_dates = clean_dates[clean_dates <= effective]
    calendar = pd.DatetimeIndex(open_dates if open_dates is not None else []).dropna()
    calendar = calendar.normalize().drop_duplicates().sort_values()
    has_full_calendar = bool(
        len(calendar)
        and calendar[0] <= target
        and calendar[-1] >= effective
    )
    if has_full_calendar:
        expected = calendar[(calendar >= target) & (calendar <= effective)]
        required_coverage = MIN_OPEN_DAY_COVERAGE
        max_missing_allowed = MAX_CONSECUTIVE_MISSING_OPEN_DAYS
    else:
        expected = pd.bdate_range(target, effective)
        required_coverage = FALLBACK_BUSINESS_DAY_COVERAGE
        max_missing_allowed = MAX_CONSECUTIVE_MISSING_FALLBACK_DAYS

    anomaly_values = [] if anomaly_dates is None else list(anomaly_dates)
    anomalies = pd.DatetimeIndex(
        pd.to_datetime(anomaly_values, errors="coerce")
    ).dropna().normalize()
    (
        complete,
        reason_code,
        anchor_day,
        observation_count,
        expected_count,
        coverage_value,
        max_missing,
        anomaly_count,
    ) = period_window_quality_kernel(
        _date_days(clean_dates),
        int(target.to_datetime64().astype("datetime64[D]").astype(np.int64)),
        int(effective.to_datetime64().astype("datetime64[D]").astype(np.int64)),
        _date_days(expected),
        _date_days(anomalies),
        WINDOW_START_TOLERANCE_DAYS,
        float(required_coverage),
        max_missing_allowed,
    )
    anchor = (
        None
        if anchor_day == NAT_DAY
        else pd.Timestamp(int(anchor_day), unit="D").normalize()
    )
    coverage_ratio = None if not np.isfinite(coverage_value) else float(coverage_value)
    reason = _QUALITY_REASON_BY_CODE[int(reason_code)]
    return PeriodWindowQuality(
        complete=bool(complete),
        reason=reason,
        target_date=target,
        effective_date=effective,
        anchor_date=anchor,
        observation_count=observation_count,
        expected_observation_count=expected_count,
        coverage_ratio=coverage_ratio,
        max_consecutive_missing=max_missing,
        anomaly_count=anomaly_count,
    )


def series_quality_execution_audit() -> dict[str, object]:
    """Return the shared fixed-signature execution proof for this call graph."""

    return instrument_analytics_numba_execution_audit()


def warm_series_quality_numba_kernels() -> dict[str, object]:
    return warm_instrument_analytics_numba_kernels()

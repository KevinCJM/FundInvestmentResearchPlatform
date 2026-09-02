"""Shared completeness and adjusted-NAV quality rules for period metrics."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as parquet


WINDOW_START_TOLERANCE_DAYS = 10
MIN_OPEN_DAY_COVERAGE = 0.90
FALLBACK_BUSINESS_DAY_COVERAGE = 0.80
MAX_CONSECUTIVE_MISSING_OPEN_DAYS = 5
MAX_CONSECUTIVE_MISSING_FALLBACK_DAYS = 10
ADJ_NAV_DISLOCATION_THRESHOLD = 0.20
REFERENCE_NAV_STABLE_THRESHOLD = 0.10
EXTREME_ADJ_NAV_RETURN_THRESHOLD = 1.0


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
    working[value_column] = pd.to_numeric(working[value_column], errors="coerce")
    working = (
        working.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=[date_column, value_column])
        .sort_values(date_column)
        .drop_duplicates(subset=[date_column], keep="last")
    )
    working = working[working[value_column] > 0]
    if len(working) < 2:
        return pd.DatetimeIndex([])
    adjusted_return = working[value_column].pct_change(fill_method=None)
    reference_stable = pd.Series(False, index=working.index)
    for column in ("accum_nav", "unit_nav"):
        if column not in working.columns:
            continue
        reference = pd.to_numeric(working[column], errors="coerce").where(lambda values: values > 0)
        reference_return = reference.pct_change(fill_method=None)
        reference_stable |= reference_return.notna() & reference_return.abs().le(
            REFERENCE_NAV_STABLE_THRESHOLD
        )
    suspicious = (
        adjusted_return.abs().gt(ADJ_NAV_DISLOCATION_THRESHOLD) & reference_stable
    ) | adjusted_return.abs().gt(EXTREME_ADJ_NAV_RETURN_THRESHOLD)
    return pd.DatetimeIndex(working.loc[suspicious.fillna(False), date_column].dt.normalize())


def _longest_missing_run(expected: pd.DatetimeIndex, observed: set[pd.Timestamp]) -> int:
    longest = 0
    current = 0
    for date in expected:
        if pd.Timestamp(date).normalize() in observed:
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest


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
    anchors = clean_dates[clean_dates <= target]
    if anchors.empty:
        return PeriodWindowQuality(False, "insufficient_span", target, effective, None, 0, 0, None, 0, 0)
    anchor = pd.Timestamp(anchors[-1]).normalize()
    selected_dates = clean_dates[clean_dates >= anchor]
    observation_count = int(len(selected_dates))
    if (target - anchor).days > WINDOW_START_TOLERANCE_DAYS:
        return PeriodWindowQuality(
            False,
            "start_anchor_too_old",
            target,
            effective,
            anchor,
            observation_count,
            0,
            None,
            0,
            0,
        )

    calendar = pd.DatetimeIndex(open_dates if open_dates is not None else []).dropna()
    calendar = calendar.normalize().drop_duplicates().sort_values()
    has_full_calendar = bool(
        len(calendar)
        and calendar.min() <= target
        and calendar.max() >= effective
    )
    if has_full_calendar:
        expected = calendar[(calendar >= target) & (calendar <= effective)]
        required_coverage = MIN_OPEN_DAY_COVERAGE
        max_missing_allowed = MAX_CONSECUTIVE_MISSING_OPEN_DAYS
    else:
        expected = pd.bdate_range(target, effective)
        required_coverage = FALLBACK_BUSINESS_DAY_COVERAGE
        max_missing_allowed = MAX_CONSECUTIVE_MISSING_FALLBACK_DAYS

    observed = {pd.Timestamp(date).normalize() for date in selected_dates}
    present_count = sum(pd.Timestamp(date).normalize() in observed for date in expected)
    expected_count = int(len(expected))
    coverage_ratio = (
        min(float(present_count) / expected_count, 1.0) if expected_count else None
    )
    max_missing = _longest_missing_run(expected, observed) if expected_count else 0
    anomaly_values = [] if anomaly_dates is None else list(anomaly_dates)
    anomalies = pd.DatetimeIndex(
        pd.to_datetime(anomaly_values, errors="coerce")
    ).dropna().normalize()
    anomaly_count = int(((anomalies > anchor) & (anomalies <= effective)).sum())

    reason: Optional[str] = None
    if anomaly_count:
        reason = "adjusted_nav_anomaly"
    elif observation_count < 2:
        reason = "insufficient_observations"
    elif coverage_ratio is not None and coverage_ratio < required_coverage:
        reason = "insufficient_density"
    elif max_missing > max_missing_allowed:
        reason = "internal_gap"
    return PeriodWindowQuality(
        complete=reason is None,
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

"""Measure what each dataset can actually prove about its own timeliness.

The catalog says what a dataset claims; this module opens the file and checks.
It reads only the two date columns, so scanning 1.5M NAV rows costs tens of
milliseconds rather than loading the whole frame.

The result is the evidence behind every PIT decision downstream: the grade, the
publication-lag distribution an analyst can eyeball, and the latest date the
dataset can honestly answer questions about.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# In-package references stay relative on purpose. Spelling them
# `backend.pit.catalog` in one place and `pit.catalog` in another loads the
# package twice under two names, and each copy then keeps its own `_CACHE` — a
# warmed audit in one is a 10-second rescan in the other.
from .catalog import (
    DATASETS,
    DATASETS_BY_ID,
    GRADE_LABELS,
    DatasetPitDeclaration,
    grade,
)

try:
    from backend.market_data import resolve_market_data_file
except ModuleNotFoundError:  # pragma: no cover - backend/ direct execution
    from market_data import resolve_market_data_file


# Histogram buckets for the publication lag, in calendar days. The first three
# separate "same day / next day / two days" because that is where 97% of NAV
# rows live and where a one-day mistake actually changes a backtest.
LAG_BUCKETS: tuple[tuple[str, int, Optional[int]], ...] = (
    ("0", 0, 0),
    ("1", 1, 1),
    ("2", 2, 2),
    ("3-5", 3, 5),
    ("6-10", 6, 10),
    ("11-20", 11, 20),
    ("21+", 21, None),
)

# ponytail: process-local memo keyed on the file fingerprint. The audit page
# polls; re-reading 3M date cells per poll is pure waste. Swap for a shared
# cache only if this ever runs multi-process.
_CACHE: dict[str, dict[str, Any]] = {}


def _fingerprint(path: Path) -> str:
    stat = path.stat()
    return f"{path}|{stat.st_size}|{int(stat.st_mtime_ns)}"


def _dates(frame: pd.DataFrame, column: str) -> Optional[pd.Series]:
    if column not in frame.columns:
        return None
    values = pd.to_datetime(frame[column], errors="coerce")
    # Tushare hands dates back as 8-digit strings often enough that a silent
    # all-NaT column would otherwise read as "no coverage".
    if values.isna().all():
        values = pd.to_datetime(frame[column], errors="coerce", format="%Y%m%d")
    return values.dt.normalize()


def _range(values: Optional[pd.Series]) -> dict[str, Optional[str]]:
    if values is None or values.dropna().empty:
        return {"start": None, "end": None}
    cleaned = values.dropna()
    return {
        "start": cleaned.min().strftime("%Y-%m-%d"),
        "end": cleaned.max().strftime("%Y-%m-%d"),
    }


def _lag_profile(lags: pd.Series) -> dict[str, Any]:
    finite = lags.dropna()
    if finite.empty:
        return {"p50": None, "p95": None, "max": None, "negative_rows": 0, "histogram": []}
    values = finite.to_numpy(dtype=np.float64)
    histogram = []
    for label, low, high in LAG_BUCKETS:
        mask = values >= low if high is None else (values >= low) & (values <= high)
        histogram.append({"bucket": label, "rows": int(mask.sum())})
    return {
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
        # A negative lag means the file claims we knew the value before it
        # existed. That is a data bug, not a PIT property, and it must surface.
        "negative_rows": int((values < 0).sum()),
        "histogram": histogram,
    }


def _finish(
    base: dict[str, Any],
    declaration: DatasetPitDeclaration,
    rows: int,
    event_values: Optional[pd.Series],
    availability_values: Optional[pd.Series],
    fingerprint: str,
    coverage: Optional[float] = None,
    lag: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    dataset_grade = grade(declaration, coverage)
    result = {
        **base,
        "present": True,
        "rows": rows,
        "availability_coverage": coverage,
        "event_range": _range(event_values),
        "availability_range": _range(availability_values),
        "lag": lag if lag is not None else _lag_profile(pd.Series(dtype=float)),
        "grade": dataset_grade,
        "grade_label": GRADE_LABELS[dataset_grade],
        "fingerprint": fingerprint,
    }
    _CACHE[declaration.dataset_id] = dict(result)
    return result


def audit_dataset(data_dir: Path, declaration: DatasetPitDeclaration) -> dict[str, Any]:
    """Open one dataset and report what it can prove about its own clocks."""

    base = {
        "dataset_id": declaration.dataset_id,
        "label": declaration.label,
        "file": declaration.file,
        "event_field": declaration.event_field or None,
        "availability_field": declaration.availability_field,
        "declared_lag_days": declaration.declared_lag_days,
        "revisable": declaration.revisable,
        "note": declaration.note,
    }

    try:
        path = resolve_market_data_file(declaration.file, data_dir)
    except Exception:  # noqa: BLE001 - a broken manifest must not blank the page
        path = data_dir / declaration.file
    if not path.exists():
        dataset_grade = grade(declaration, None)
        return {
            **base,
            "present": False,
            "rows": 0,
            "availability_coverage": None,
            "event_range": {"start": None, "end": None},
            "availability_range": {"start": None, "end": None},
            "lag": _lag_profile(pd.Series(dtype=float)),
            "grade": dataset_grade,
            "grade_label": GRADE_LABELS[dataset_grade],
            "fingerprint": None,
        }

    fingerprint = _fingerprint(path)
    cached = _CACHE.get(declaration.dataset_id)
    if cached is not None and cached.get("fingerprint") == fingerprint:
        return dict(cached)

    wanted = [column for column in (declaration.event_field, declaration.availability_field) if column]
    if not wanted:
        # A dataset with no declared clocks needs only its row count, and the
        # parquet footer already carries that. Reading the rows would be waste.
        return _finish(base, declaration, int(pq.read_metadata(path).num_rows), None, None, fingerprint)
    try:
        frame = pd.read_parquet(path, columns=wanted, engine="pyarrow")
    except Exception:  # noqa: BLE001 - a missing declared column is itself a finding
        frame = pd.read_parquet(path)

    rows = int(len(frame))
    event_values = _dates(frame, declaration.event_field) if declaration.event_field else None
    availability_values = (
        _dates(frame, declaration.availability_field) if declaration.availability_field else None
    )

    coverage: Optional[float] = None
    lag = _lag_profile(pd.Series(dtype=float))
    if availability_values is not None:
        coverage = float(availability_values.notna().mean()) if rows else 0.0
        if event_values is not None:
            lag = _lag_profile((availability_values - event_values).dt.days)

    return _finish(base, declaration, rows, event_values, availability_values, fingerprint, coverage, lag)


def _effective_available_end(item: dict[str, Any]) -> Optional[str]:
    """The last date this dataset can honestly answer a question about."""

    availability_end = item["availability_range"]["end"]
    if availability_end:
        return availability_end
    event_end = item["event_range"]["end"]
    if not event_end:
        return None
    lag = int(item.get("declared_lag_days") or 0)
    if lag <= 0:
        return event_end
    return (pd.Timestamp(event_end) + pd.Timedelta(days=lag)).strftime("%Y-%m-%d")


def audit_all(data_dir: Path) -> dict[str, Any]:
    """Full PIT capability sheet plus the headline counters the page shows."""

    datasets = [audit_dataset(data_dir, declaration) for declaration in DATASETS]
    present = [item for item in datasets if item["present"]]
    for item in datasets:
        item["available_through"] = _effective_available_end(item)
    grades = {"A": 0, "B": 0, "C": 0}
    for item in present:
        grades[item["grade"]] = grades.get(item["grade"], 0) + 1
    # Only A/B datasets bound a PIT run: the C-grade dimension tables carry a
    # publication date that says nothing about data currency, and letting one of
    # them set the floor would report the whole platform as a year stale.
    available_ends = [
        item["available_through"]
        for item in present
        if item["available_through"] and item["grade"] in {"A", "B"}
    ]
    return {
        "datasets": datasets,
        "summary": {
            "declared": len(datasets),
            "present": len(present),
            "missing": len(datasets) - len(present),
            "grade_a": grades["A"],
            "grade_b": grades["B"],
            "grade_c": grades["C"],
            "total_rows": int(sum(item["rows"] for item in present)),
            # The pool is only as point-in-time as its least current member.
            "available_through": min(available_ends) if available_ends else None,
            "latest_dataset_end": max(available_ends) if available_ends else None,
            "available_through_basis": "grade_a_b",
        },
    }


def dataset_grade(data_dir: Path, dataset_id: str) -> Optional[str]:
    declaration = DATASETS_BY_ID.get(dataset_id)
    if declaration is None:
        return None
    return audit_dataset(data_dir, declaration)["grade"]


def clear_cache() -> None:
    """Drop the memo; tests rewrite fixture files inside one mtime tick."""

    _CACHE.clear()


__all__ = ["audit_all", "audit_dataset", "clear_cache", "dataset_grade"]

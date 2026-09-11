"""Measure what each dataset can actually prove about its own timeliness.

The catalog says what a dataset claims; this module opens the file and checks.
It reads only the two date columns, so scanning 1.5M NAV rows costs tens of
milliseconds rather than loading the whole frame.

The result is the evidence behind every PIT decision downstream: the grade, the
publication-lag distribution an analyst can eyeball, and the latest date the
dataset can honestly answer questions about.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import threading
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
    SNAPSHOT_FIELD,
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

# The scan itself reads the two clock columns of every declared file. On a cold
# cache over 30M+ NAV rows that is minutes, so it must never sit inside a request:
# the page asks for whatever is already measured and is told what is still running.
_SCAN_LOCK = threading.Lock()
_SCAN: dict[str, Any] = {"state": "idle", "started_at": None, "finished_at": None, "error": None}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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


def history_path(data_dir: Path, declaration: DatasetPitDeclaration) -> Optional[Path]:
    """Where the append-only snapshot log for a revisable table lives, if declared."""

    if not declaration.history_file:
        return None
    try:
        return resolve_market_data_file(declaration.history_file, data_dir)
    except Exception:  # noqa: BLE001 - a broken manifest falls back to the plain layout
        return data_dir / declaration.history_file


def _history_profile(path: Optional[Path]) -> dict[str, Any]:
    """How far back this table's own state can be replayed.

    An empty profile is not a failure — it is the honest answer for a dimension
    table that has only ever been overwritten. `begins_at` is the earliest day a
    universe question about this table can be answered without hindsight.
    """

    empty = {"available": False, "snapshots": 0, "begins_at": None, "latest": None, "file": None}
    if path is None or not path.exists():
        return empty
    try:
        dates = pd.read_parquet(path, columns=[SNAPSHOT_FIELD])[SNAPSHOT_FIELD]
    except Exception:  # noqa: BLE001 - a malformed log must not blank the page
        return empty
    stamps = pd.to_datetime(dates, errors="coerce").dropna()
    if stamps.empty:
        return empty
    return {
        "available": True,
        "snapshots": int(stamps.nunique()),
        "begins_at": stamps.min().strftime("%Y-%m-%d"),
        "latest": stamps.max().strftime("%Y-%m-%d"),
        "file": path.name,
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
    history: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    history = history or _history_profile(None)
    dataset_grade = grade(declaration, coverage, history_snapshots=int(history["snapshots"]))
    result = {
        **base,
        "history": history,
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


def audit_dataset(
    data_dir: Path, declaration: DatasetPitDeclaration, *, scan: bool = True
) -> dict[str, Any]:
    """Open one dataset and report what it can prove about its own clocks.

    `scan=False` answers only from the memo and marks anything unmeasured
    `pending` instead of reading it, which is what lets the page render at once
    while the background scan catches up.
    """

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
    history = _history_profile(history_path(data_dir, declaration))
    if not path.exists():
        dataset_grade = grade(declaration, None, history_snapshots=int(history["snapshots"]))
        return {
            **base,
            "history": history,
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
    if cached is not None and cached.get("fingerprint") == fingerprint and cached.get("history") == history:
        return dict(cached)
    if not scan:
        # Same shape, but honest about knowing nothing yet: reporting grade C for
        # an unread file would look like a measured verdict.
        return {
            **base,
            "history": history,
            "pending": True,
            "present": True,
            "rows": 0,
            "availability_coverage": None,
            "event_range": {"start": None, "end": None},
            "availability_range": {"start": None, "end": None},
            "lag": _lag_profile(pd.Series(dtype=float)),
            "grade": None,
            "grade_label": "扫描中",
            "fingerprint": fingerprint,
        }

    wanted = [column for column in (declaration.event_field, declaration.availability_field) if column]
    if not wanted:
        # A dataset with no declared clocks needs only its row count, and the
        # parquet footer already carries that. Reading the rows would be waste.
        return _finish(
            base,
            declaration,
            int(pq.read_metadata(path).num_rows),
            None,
            None,
            fingerprint,
            history=history,
        )
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

    return _finish(
        base, declaration, rows, event_values, availability_values, fingerprint, coverage, lag, history
    )


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


def audit_all(data_dir: Path, *, scan: bool = True) -> dict[str, Any]:
    """Full PIT capability sheet plus the headline counters the page shows."""

    datasets = [audit_dataset(data_dir, declaration, scan=scan) for declaration in DATASETS]
    present = [item for item in datasets if item["present"] and not item.get("pending")]
    for item in datasets:
        item["available_through"] = _effective_available_end(item)
    grades = {"A": 0, "B": 0, "C": 0}
    for item in present:
        grades[item["grade"]] = grades.get(item["grade"], 0) + 1
    pending = [item["dataset_id"] for item in datasets if item.get("pending")]
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
            "missing": len([item for item in datasets if not item["present"]]),
            "pending": len(pending),
            "grade_a": grades["A"],
            "grade_b": grades["B"],
            "grade_c": grades["C"],
            "total_rows": int(sum(item["rows"] for item in present)),
            # The pool is only as point-in-time as its least current member.
            "available_through": min(available_ends) if available_ends else None,
            "latest_dataset_end": max(available_ends) if available_ends else None,
            "available_through_basis": "grade_a_b",
        },
        "scan": {**_SCAN, "pending": pending},
    }


def start_scan(data_dir: Path) -> dict[str, Any]:
    """Measure every declared dataset off the request thread.

    Idempotent: a second call while one is running joins the running scan rather
    than starting a rival one, because two threads reading the same 1GB files is
    slower than one.
    """

    with _SCAN_LOCK:
        if _SCAN["state"] == "running":
            return dict(_SCAN)
        _SCAN.update(
            {"state": "running", "started_at": _utc_now(), "finished_at": None, "error": None}
        )

    def run() -> None:
        try:
            audit_all(data_dir, scan=True)
        except Exception as exc:  # noqa: BLE001 - a failed scan must be reportable, not fatal
            with _SCAN_LOCK:
                _SCAN.update({"state": "failed", "finished_at": _utc_now(), "error": str(exc)})
            return
        with _SCAN_LOCK:
            _SCAN.update({"state": "ready", "finished_at": _utc_now(), "error": None})

    threading.Thread(target=run, name="pit-audit-scan", daemon=True).start()
    with _SCAN_LOCK:
        return dict(_SCAN)


def scan_status() -> dict[str, Any]:
    with _SCAN_LOCK:
        return dict(_SCAN)


def dataset_grade(data_dir: Path, dataset_id: str) -> Optional[str]:
    declaration = DATASETS_BY_ID.get(dataset_id)
    if declaration is None:
        return None
    return audit_dataset(data_dir, declaration)["grade"]


def clear_cache() -> None:
    """Drop the memo; tests rewrite fixture files inside one mtime tick."""

    _CACHE.clear()
    with _SCAN_LOCK:
        _SCAN.update({"state": "idle", "started_at": None, "finished_at": None, "error": None})


__all__ = [
    "audit_all",
    "audit_dataset",
    "clear_cache",
    "dataset_grade",
    "scan_status",
    "start_scan",
]

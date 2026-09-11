"""One point-in-time read for any declared dataset.

`fit.load_adj_nav_pit` proved the shape on NAV: resolve an availability date,
drop what was not yet published on the research day, then de-duplicate keeping
the newest revision that was knowable *then* rather than the newest that exists
today. Every other dataset needs the same three steps against different column
names — which is precisely what the catalog already declares — plus a fourth
that NAV does not need: replaying an overwrite-only dimension table from its
snapshot log.

NAV keeps its own loader: it merges two files and carries a strict-mode
contract of its own. This module is for everything else.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import pandas as pd

from .audit import history_path

try:
    from backend.market_data import resolve_market_data_file
except ModuleNotFoundError:  # pragma: no cover - backend/ direct execution
    from market_data import resolve_market_data_file

from .catalog import (
    DATASETS_BY_ID,
    RUN_MODE_STRICT,
    SNAPSHOT_FIELD,
    DatasetPitDeclaration,
)
from .context import PitContextError, ResearchContext, parse_as_of

# What a read could prove about the day it claims to stand on.
REPLAYED = "REPLAYED"          # a dated snapshot at or before as_of was used
LATEST_ONLY = "LATEST_ONLY"    # only today's overwrite exists — hindsight is baked in
NOT_APPLICABLE = "NOT_APPLICABLE"  # the dataset is not revisable, so there is nothing to replay

AVAILABLE_FIELD = "available_date"


@dataclass(frozen=True)
class PitFrame:
    """Rows knowable on the research day, plus how that was established."""

    frame: pd.DataFrame
    lineage: dict[str, Any] = field(default_factory=dict)

    @property
    def replayed(self) -> bool:
        return self.lineage.get("coverage") == REPLAYED


def _resolve_availability(
    frame: pd.DataFrame, declaration: DatasetPitDeclaration
) -> tuple[pd.Series, str]:
    """The day each row became knowable, and how that was arrived at.

    Order of preference: a real publication column, then the event date plus the
    declared lag, then nothing at all — a dataset with neither clock cannot be
    cut and says so instead of pretending an empty result is a strict one.
    """

    column = declaration.availability_field
    if column and column in frame.columns:
        stamps = pd.to_datetime(frame[column], errors="coerce")
        if stamps.isna().all() and frame[column].notna().any():
            stamps = pd.to_datetime(frame[column], errors="coerce", format="%Y%m%d")
        return stamps.dt.normalize(), "announcement"
    event = declaration.event_field
    if event and event in frame.columns:
        stamps = pd.to_datetime(frame[event], errors="coerce")
        if stamps.isna().all() and frame[event].notna().any():
            stamps = pd.to_datetime(frame[event], errors="coerce", format="%Y%m%d")
        lag = pd.Timedelta(days=int(declaration.declared_lag_days or 0))
        return (stamps.dt.normalize() + lag), "event_plus_declared_lag"
    return pd.Series(pd.NaT, index=frame.index), "none"


def _replay_dimension(
    log: pd.DataFrame,
    declaration: DatasetPitDeclaration,
    cutoff: Optional[pd.Timestamp],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """The state of an overwrite-only table as last observed on or before `cutoff`.

    The whole point is the row that is *absent*: a fund delisted before today
    still appears in the snapshot taken while it was listed, so a replayed
    universe contains the products that were actually selectable then.
    """

    stamps = pd.to_datetime(log[SNAPSHOT_FIELD], errors="coerce")
    usable = log[stamps.notna()].copy()
    usable[SNAPSHOT_FIELD] = stamps[stamps.notna()]
    detail: dict[str, Any] = {
        "history_snapshots": int(usable[SNAPSHOT_FIELD].nunique()),
        "history_begins_at": None,
        "snapshot_used": None,
    }
    if usable.empty:
        return usable.drop(columns=[SNAPSHOT_FIELD], errors="ignore"), detail
    detail["history_begins_at"] = usable[SNAPSHOT_FIELD].min().strftime("%Y-%m-%d")
    if cutoff is not None:
        usable = usable[usable[SNAPSHOT_FIELD] <= cutoff]
    if usable.empty:
        return usable.drop(columns=[SNAPSHOT_FIELD], errors="ignore"), detail

    chosen = usable[SNAPSHOT_FIELD].max()
    detail["snapshot_used"] = chosen.strftime("%Y-%m-%d")
    # Exactly the last snapshot taken by the cut-off, not the union of every
    # snapshot up to it: each write is the whole table, so a row that is absent
    # from that snapshot was absent from the table — and resurrecting it from an
    # older one is precisely the survivorship error this exists to prevent.
    return usable[usable[SNAPSHOT_FIELD] == chosen].drop(columns=[SNAPSHOT_FIELD]), detail


def read_pit(
    dataset_id: str,
    data_dir: Path,
    context: ResearchContext,
    *,
    columns: Optional[Sequence[str]] = None,
) -> PitFrame:
    """Read one declared dataset as it was knowable on `context.as_of`.

    `as_of=None` returns everything on disk — the pre-PIT behaviour — but the
    lineage says so rather than letting an uncut read look like a cut one.
    """

    declaration = DATASETS_BY_ID.get(dataset_id)
    if declaration is None:
        raise PitContextError(f"未声明 PIT 口径的数据集：{dataset_id}")
    cutoff = parse_as_of(context.as_of)
    strict = context.run_mode == RUN_MODE_STRICT

    lineage: dict[str, Any] = {
        "dataset_id": dataset_id,
        "label": declaration.label,
        "as_of": context.as_of,
        "as_of_applied": cutoff is not None,
        "run_mode": context.run_mode,
        "data_release_id": context.data_release_id,
        "coverage": NOT_APPLICABLE,
        "availability_basis": "none",
        "rows_before_cut": 0,
        "rows_after_cut": 0,
        "rows_dropped_by_as_of": 0,
        "history_snapshots": 0,
        "history_begins_at": None,
        "snapshot_used": None,
        "warnings": [],
    }

    log_path = history_path(data_dir, declaration)
    frame: Optional[pd.DataFrame] = None
    if declaration.revisable:
        lineage["coverage"] = LATEST_ONLY
        if log_path is not None and log_path.exists():
            try:
                log = pd.read_parquet(log_path)
            except Exception as exc:  # noqa: BLE001 - fall back to the latest state
                lineage["warnings"].append(f"维表历史读取失败（{exc}），已回退到最新态。")
                log = None
            if log is not None and SNAPSHOT_FIELD in log.columns:
                replayed, detail = _replay_dimension(log, declaration, cutoff)
                lineage.update(detail)
                if not replayed.empty:
                    frame = replayed
                    lineage["coverage"] = REPLAYED
                elif cutoff is not None and detail["history_begins_at"]:
                    lineage["warnings"].append(
                        f"{context.as_of} 早于维表历史起点 {detail['history_begins_at']}，"
                        "该日的可选产品域无法还原。"
                    )

    if frame is None:
        try:
            path = resolve_market_data_file(declaration.file, data_dir)
        except Exception:  # noqa: BLE001 - a broken manifest falls back to the plain layout
            path = data_dir / declaration.file
        if not path.exists():
            lineage["warnings"].append(f"{declaration.file} 不存在。")
            return PitFrame(pd.DataFrame(), lineage)
        frame = pd.read_parquet(path)

    lineage["rows_before_cut"] = int(len(frame))
    availability, basis = _resolve_availability(frame, declaration)
    lineage["availability_basis"] = basis
    if basis != "none":
        frame = frame.copy()
        frame[AVAILABLE_FIELD] = availability
        if cutoff is not None:
            before = int(len(frame))
            frame = frame[frame[AVAILABLE_FIELD].isna() | (frame[AVAILABLE_FIELD] <= cutoff)]
            lineage["rows_dropped_by_as_of"] = before - int(len(frame))
    elif cutoff is not None and not declaration.revisable:
        lineage["warnings"].append(
            f"{declaration.label} 既没有可得时间列也没有事件日，无法按研究日截断。"
        )

    if strict and lineage["coverage"] == LATEST_ONLY:
        raise PitContextError(
            f"严格 PIT 模式下 {declaration.label} 只有最新态，没有 {context.as_of} 当日的历史版本；"
            "请改用研究模式，或等待维表历史快照积累到该日期。"
        )

    if columns:
        # Projected last, and never before the cut: asking for a subset of
        # columns must not quietly drop the clock the cut is made on.
        keep = [column for column in (*columns, AVAILABLE_FIELD) if column in frame.columns]
        frame = frame[list(dict.fromkeys(keep))]

    lineage["rows_after_cut"] = int(len(frame))
    return PitFrame(frame.reset_index(drop=True), lineage)


__all__ = [
    "AVAILABLE_FIELD",
    "LATEST_ONLY",
    "NOT_APPLICABLE",
    "REPLAYED",
    "PitFrame",
    "read_pit",
]

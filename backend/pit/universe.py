"""Which products were actually selectable on the research day.

The leak this module closes is not in any formula: it is in the candidate list.
Screen today's fund table for a 2018 backtest and every fund that has since been
delisted is missing, while every fund launched in 2021 is present — the universe
itself carries the answer. Catching that with a per-formula probe is impossible,
because each individual computation is perfectly causal.

Three ways to answer, strongest first:

* ``REPLAYED``   — a dated snapshot of the dimension table taken on or before the
  research day. Restores both membership and the attribute values of the time.
* ``INTERVAL``   — no snapshot, but the table carries listing and delisting dates,
  so membership can be reconstructed even though attributes are today's.
* ``LATEST_ONLY``— neither. The universe is today's, and the result says so.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from .catalog import RUN_MODE_STRICT
from .context import PitContextError, ResearchContext, parse_as_of
from .frame import LATEST_ONLY, REPLAYED, read_pit

INTERVAL = "INTERVAL"

# dataset, code column, listing column, the column that ends membership (if any)
KINDS: dict[str, tuple[str, str, str, Optional[str]]] = {
    "fund": ("etf_info", "ts_code", "list_date", "delist_date"),
    "index": ("index_info", "ts_code", "list_date", "exp_date"),
    "stock": ("stock_basic", "ts_code", "list_date", None),
}

KIND_LABELS = {"fund": "基金产品域", "index": "指数域", "stock": "股票域"}


@dataclass(frozen=True)
class UniverseView:
    """The selectable set on one day, with how firmly that was established."""

    as_of: Optional[str]
    kind: str
    codes: tuple[str, ...]
    detail: pd.DataFrame
    coverage: str
    history_begins_at: Optional[str]
    warnings: tuple[str, ...]
    lineage: dict[str, Any]

    @property
    def replayable(self) -> bool:
        return self.coverage in {REPLAYED, INTERVAL}


def _dates(frame: pd.DataFrame, column: Optional[str]) -> Optional[pd.Series]:
    if not column or column not in frame.columns:
        return None
    stamps = pd.to_datetime(frame[column], errors="coerce")
    if stamps.isna().all() and frame[column].notna().any():
        stamps = pd.to_datetime(frame[column], errors="coerce", format="%Y%m%d")
    return stamps


def universe_as_of(
    data_dir: Path,
    context: ResearchContext,
    *,
    kind: str = "fund",
    codes: Optional[list[str]] = None,
) -> UniverseView:
    """The products a screen run on `context.as_of` was allowed to choose from."""

    if kind not in KINDS:
        raise PitContextError(f"不支持的产品域：{kind}")
    dataset_id, code_field, list_field, end_field = KINDS[kind]
    cutoff = parse_as_of(context.as_of)
    loaded = read_pit(dataset_id, data_dir, context)
    frame = loaded.frame
    warnings: list[str] = list(loaded.lineage.get("warnings") or [])
    coverage = REPLAYED if loaded.lineage.get("coverage") == REPLAYED else LATEST_ONLY

    if frame.empty:
        return UniverseView(
            as_of=context.as_of,
            kind=kind,
            codes=(),
            detail=frame,
            coverage=coverage,
            history_begins_at=loaded.lineage.get("history_begins_at"),
            warnings=tuple(warnings),
            lineage=dict(loaded.lineage),
        )

    listed = _dates(frame, list_field)
    ended = _dates(frame, end_field)
    if cutoff is not None:
        mask = pd.Series(True, index=frame.index)
        if listed is not None:
            # A missing listing date is kept: dropping it would silently shrink
            # the universe on a data-quality problem rather than a real one.
            mask &= listed.isna() | (listed <= cutoff)
        if ended is not None:
            mask &= ended.isna() | (ended > cutoff)
        frame = frame[mask]
        if coverage != REPLAYED and listed is not None and ended is not None:
            coverage = INTERVAL
        elif coverage != REPLAYED:
            warnings.append(
                f"{KIND_LABELS[kind]}没有退出日期列，只能按上市日下界还原；"
                "研究日之后退出的成分仍会留在域内。"
            )

    if codes:
        wanted = {str(code) for code in codes if code}
        frame = frame[frame[code_field].astype(str).isin(wanted)]

    if coverage == LATEST_ONLY and cutoff is not None:
        warnings.append(
            f"{KIND_LABELS[kind]}使用的是最新态维表，{context.as_of} 当日的可选集合无法还原——"
            "结果带有幸存者偏差。"
        )
        if context.run_mode == RUN_MODE_STRICT:
            raise PitContextError(
                f"严格 PIT 模式禁止在无法还原的{KIND_LABELS[kind]}上研究：{context.as_of} 没有可用的历史版本。"
            )

    lineage = dict(loaded.lineage)
    lineage.update(
        {
            "kind": kind,
            "kind_label": KIND_LABELS[kind],
            "coverage": coverage,
            "member_count": int(len(frame)),
            "warnings": warnings,
        }
    )
    return UniverseView(
        as_of=context.as_of,
        kind=kind,
        codes=tuple(frame[code_field].astype(str).tolist()) if code_field in frame.columns else (),
        detail=frame.reset_index(drop=True),
        coverage=coverage,
        history_begins_at=loaded.lineage.get("history_begins_at"),
        warnings=tuple(warnings),
        lineage=lineage,
    )


__all__ = ["INTERVAL", "KINDS", "KIND_LABELS", "UniverseView", "universe_as_of"]

"""Refuse, or at least label, a run whose candidate set knows the future.

Two failures live here, and neither is visible in any single formula:

* the **universe** was fixed using data published after the day being simulated —
  a pool screened on 2026 numbers, replayed over 2018;
* the universe cannot be replayed at all, so it is today's survivors wearing a
  historical date.

Research mode warns rather than blocks. That is deliberate: with no dimension
history on disk yet, blocking would stop every historical backtest in the
platform on day one, and a tool that has to be switched off is a tool nobody
reads. Strict mode is where it bites.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

import pandas as pd

from .context import PitContextError, ResearchContext
from .frame import LATEST_ONLY, REPLAYED
from .universe import INTERVAL

UNIVERSE_LOOKAHEAD = "UNIVERSE_LOOKAHEAD"
UNIVERSE_NOT_REPLAYABLE = "UNIVERSE_NOT_REPLAYABLE"


def _day(value: Any) -> Optional[pd.Timestamp]:
    if value in (None, ""):
        return None
    try:
        stamp = pd.Timestamp(value)
    except Exception:  # noqa: BLE001 - an unparseable date simply cannot be checked
        return None
    return None if pd.isna(stamp) else stamp.normalize()


def check_universe(
    context: ResearchContext,
    *,
    established_at: Any = None,
    coverage: Optional[str] = None,
    decision_dates: Optional[Iterable[Any]] = None,
    label: str = "产品池",
) -> list[dict[str, Any]]:
    """Findings about the candidate set behind a run. Empty means clean.

    `established_at` is the day the universe's own inputs were cut — a pool
    version's `data_as_of`, a universe snapshot's `research_date`. `coverage`
    is what :mod:`pit.universe` could prove about replayability.
    """

    findings: list[dict[str, Any]] = []
    established = _day(established_at)

    earliest = None
    dates = [stamp for stamp in (_day(date) for date in (decision_dates or [])) if stamp is not None]
    if dates:
        earliest = min(dates)
    elif context.as_of:
        earliest = _day(context.as_of)

    if established is not None and earliest is not None and established > earliest:
        findings.append(
            {
                "code": UNIVERSE_LOOKAHEAD,
                "label": label,
                "established_at": established.strftime("%Y-%m-%d"),
                "earliest_decision": earliest.strftime("%Y-%m-%d"),
                "message": (
                    f"{label}是用截至 {established.strftime('%Y-%m-%d')} 的数据筛出来的，"
                    f"却被用于 {earliest.strftime('%Y-%m-%d')} 的决策——该决策带入了未来信息。"
                ),
            }
        )

    if coverage == LATEST_ONLY and (earliest is not None or context.as_of):
        findings.append(
            {
                "code": UNIVERSE_NOT_REPLAYABLE,
                "label": label,
                "coverage": coverage,
                "message": (
                    f"{label}只有最新态，无法还原研究日当时的可选集合；"
                    "已退出的产品缺席，结果存在幸存者偏差。"
                ),
            }
        )
    return findings


def assert_no_universe_lookahead(
    context: ResearchContext,
    *,
    established_at: Any = None,
    coverage: Optional[str] = None,
    decision_dates: Optional[Iterable[Any]] = None,
    label: str = "产品池",
) -> list[dict[str, Any]]:
    """Strict mode raises on any finding; research mode hands them back to record."""

    findings = check_universe(
        context,
        established_at=established_at,
        coverage=coverage,
        decision_dates=decision_dates,
        label=label,
    )
    if findings and context.strict:
        raise PitContextError("；".join(item["message"] for item in findings))
    return findings


def universe_lineage(
    findings: list[dict[str, Any]],
    *,
    coverage: Optional[str] = None,
    established_at: Any = None,
    history_begins_at: Optional[str] = None,
    source: Optional[str] = None,
) -> dict[str, Any]:
    """The block a result carries so a reader can see what the candidate set was."""

    established = _day(established_at)
    return {
        "source": source,
        "coverage": coverage,
        "replayable": coverage in {REPLAYED, INTERVAL},
        "established_at": established.strftime("%Y-%m-%d") if established is not None else None,
        "history_begins_at": history_begins_at,
        "findings": findings,
        "clean": not findings,
    }


__all__ = [
    "UNIVERSE_LOOKAHEAD",
    "UNIVERSE_NOT_REPLAYABLE",
    "assert_no_universe_lookahead",
    "check_universe",
    "universe_lineage",
]

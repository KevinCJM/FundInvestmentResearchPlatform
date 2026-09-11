"""The fourth clock: which day a computation stands on.

The catalog models three clocks — when a thing happened, when we could see it,
which vintage we read. They answer "which *rows* were visible on T". They do not
answer "what decision would have been made on T", because a backtest makes not
one decision but a sequence of them, and every rebalance date is its own T.

`slice_fit_data` already cuts the fitting window at the rebalance date, but on
*event* time: a NAV printed for 2018-06-29 and published on 2018-07-03 is inside
a window that closes on 2018-06-30, which no live process could have used. This
module supplies the availability-aware cut and the sweep of research dates that
goes with it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Optional, Sequence

import pandas as pd

from .context import ResearchContext


def _day(value) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


@dataclass(frozen=True)
class DecisionClock:
    """The research days one run steps through, in order.

    The vintage stays fixed across the sweep on purpose: moving `as_of` asks
    "what was knowable then", while moving the release as well would also change
    *which correction of history* is being read, and the two effects would be
    impossible to tell apart in the result.
    """

    dates: tuple[str, ...]
    base: ResearchContext

    def at(self, date) -> ResearchContext:
        """The context a decision taken on `date` runs under."""

        return replace(self.base, as_of=_day(date))

    def __iter__(self):
        return iter(self.dates)

    def __len__(self) -> int:
        return len(self.dates)

    @property
    def first(self) -> Optional[str]:
        return self.dates[0] if self.dates else None

    @property
    def last(self) -> Optional[str]:
        return self.dates[-1] if self.dates else None

    @classmethod
    def from_dates(cls, dates: Iterable, base: ResearchContext) -> "DecisionClock":
        ordered = sorted({_day(date) for date in dates if date is not None})
        # A sweep may not see past the context it runs under: a backtest asked to
        # stand on 2020-12-31 must not quietly rebalance using 2021 knowledge.
        if base.as_of:
            ordered = [date for date in ordered if date <= base.as_of]
        return cls(tuple(ordered), base)

    def lineage(self) -> dict:
        return {
            "swept": bool(self.dates),
            "count": len(self.dates),
            "first": self.first,
            "last": self.last,
            "run_mode": self.base.run_mode,
            "data_release_id": self.base.data_release_id,
        }


def visible_at(
    frame: pd.DataFrame,
    up_to,
    available_at: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Rows a decision taken on `up_to` could actually have read.

    Without an availability series this is the historical behaviour — cut on the
    index — so callers that have no publication clock keep working unchanged.
    Rows whose availability is unknown fall back to their event date rather than
    being dropped, which keeps a data-quality hole from looking like a shorter
    history.
    """

    cutoff = pd.Timestamp(up_to)
    if available_at is None or available_at.empty:
        return frame.loc[:cutoff]
    stamps = pd.to_datetime(available_at.reindex(frame.index), errors="coerce")
    on_time = stamps.notna() & (stamps <= cutoff)
    fallback = stamps.isna() & (frame.index <= cutoff)
    return frame[on_time | fallback]


def availability_from_rows(
    rows: pd.DataFrame,
    *,
    event_field: str,
    available_field: str,
) -> pd.Series:
    """Collapse per-product availability into one date per observation day.

    The latest constituent wins: a class NAV for day d is only computable once
    *every* fund in it has published day d.
    """

    if rows.empty or available_field not in rows.columns:
        return pd.Series(dtype="datetime64[ns]")
    frame = rows[[event_field, available_field]].dropna(subset=[event_field])
    if frame.empty:
        return pd.Series(dtype="datetime64[ns]")
    grouped = frame.groupby(pd.to_datetime(frame[event_field]))[available_field].max()
    return pd.to_datetime(grouped).sort_index()


__all__ = ["DecisionClock", "availability_from_rows", "visible_at"]

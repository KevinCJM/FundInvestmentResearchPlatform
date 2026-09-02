"""Versioned runtime-period catalog and boundary semantics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd


PeriodKind = Literal["rolling", "calendar", "lifetime"]
PeriodUnit = Literal["W", "M", "Y"]


@dataclass(frozen=True)
class PeriodSpec:
    value: str
    label: str
    description: str
    kind: PeriodKind
    group: str
    count: int | None = None
    unit: PeriodUnit | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "value": self.value,
            "label": self.label,
            "description": self.description,
            "kind": self.kind,
            "group": self.group,
        }


@dataclass(frozen=True)
class PeriodBounds:
    """Boundary NAV date and inclusive calendar end for one calculation."""

    anchor_target: pd.Timestamp
    calendar_end: pd.Timestamp


def _rolling(value: str, count: int, unit: PeriodUnit, label: str) -> PeriodSpec:
    unit_label = {"W": "周", "M": "月", "Y": "年"}[unit]
    return PeriodSpec(
        value=value,
        label=label,
        description=f"以有效截止日为终点，向前滚动 {count} {unit_label}。",
        kind="rolling",
        group="rolling",
        count=count,
        unit=unit,
    )


def _calendar(value: str, count: int, unit: PeriodUnit, label: str) -> PeriodSpec:
    unit_label = {"W": "自然周", "M": "自然月", "Y": "自然年度"}[unit]
    return PeriodSpec(
        value=value,
        label=label,
        description=f"截止日之前第 {count} 个完整{unit_label}，不包含当前未结束区间。",
        kind="calendar",
        group={"W": "calendar_week", "M": "calendar_month", "Y": "calendar_year"}[unit],
        count=count,
        unit=unit,
    )


PERIOD_SPECS: tuple[PeriodSpec, ...] = (
    _rolling("1W", 1, "W", "近 1 周"),
    _rolling("1M", 1, "M", "近 1 月"),
    _rolling("3M", 3, "M", "近 3 月"),
    _rolling("6M", 6, "M", "近 6 月"),
    _rolling("1Y", 1, "Y", "近 1 年"),
    _rolling("2Y", 2, "Y", "近 2 年"),
    _rolling("3Y", 3, "Y", "近 3 年"),
    _rolling("5Y", 5, "Y", "近 5 年"),
    _rolling("10Y", 10, "Y", "近 10 年"),
    _rolling("20Y", 20, "Y", "近 20 年"),
    _rolling("30Y", 30, "Y", "近 30 年"),
    _calendar("W1", 1, "W", "上周"),
    _calendar("W2", 2, "W", "上上周"),
    _calendar("M1", 1, "M", "上月"),
    _calendar("M2", 2, "M", "上上月"),
    _calendar("Y1", 1, "Y", "去年"),
    _calendar("Y2", 2, "Y", "前年"),
    PeriodSpec(
        value="ALL",
        label="成立以来",
        description="从首个真实有效净值点计算至有效截止日。",
        kind="lifetime",
        group="lifetime",
    ),
)

PERIOD_SPEC_BY_VALUE = {item.value: item for item in PERIOD_SPECS}
SUPPORTED_PERIODS = tuple(item.value for item in PERIOD_SPECS)


def get_period_spec(period: str) -> PeriodSpec | None:
    return PERIOD_SPEC_BY_VALUE.get(str(period or "").strip().upper())


def period_metadata() -> list[dict[str, object]]:
    return [item.as_dict() for item in PERIOD_SPECS]


def period_cache_reference(as_of: str | None) -> str:
    """Prevent a natural-period result from surviving a calendar-day rollover."""

    if as_of:
        return str(as_of)
    return pd.Timestamp.today().normalize().strftime("%Y-%m-%d")


def resolve_period_bounds(
    spec: PeriodSpec,
    *,
    effective_data_date: pd.Timestamp,
    reference_date: pd.Timestamp,
    first_data_date: pd.Timestamp,
) -> PeriodBounds:
    effective = pd.Timestamp(effective_data_date).normalize()
    reference = pd.Timestamp(reference_date).normalize()
    first = pd.Timestamp(first_data_date).normalize()

    if spec.kind == "lifetime":
        return PeriodBounds(anchor_target=first, calendar_end=effective)

    if spec.kind == "rolling":
        assert spec.count is not None and spec.unit is not None
        offset = (
            pd.DateOffset(weeks=spec.count)
            if spec.unit == "W"
            else pd.DateOffset(months=spec.count)
            if spec.unit == "M"
            else pd.DateOffset(years=spec.count)
        )
        return PeriodBounds(
            anchor_target=(effective - offset).normalize(),
            calendar_end=effective,
        )

    assert spec.count is not None and spec.unit is not None
    if spec.unit == "W":
        current_start = reference - pd.Timedelta(days=reference.weekday())
        period_start = current_start - pd.DateOffset(weeks=spec.count)
        period_end = period_start + pd.Timedelta(days=6)
    elif spec.unit == "M":
        current_start = pd.Timestamp(reference.year, reference.month, 1)
        period_start = current_start - pd.DateOffset(months=spec.count)
        period_end = period_start + pd.offsets.MonthEnd(1)
    else:
        target_year = reference.year - spec.count
        period_start = pd.Timestamp(target_year, 1, 1)
        period_end = pd.Timestamp(target_year, 12, 31)

    # Returns within a complete calendar period need the last NAV on or before
    # the preceding day, so the first in-period return is not silently dropped.
    return PeriodBounds(
        anchor_target=(period_start - pd.Timedelta(days=1)).normalize(),
        calendar_end=pd.Timestamp(period_end).normalize(),
    )


__all__ = [
    "PERIOD_SPECS",
    "SUPPORTED_PERIODS",
    "PeriodBounds",
    "PeriodSpec",
    "get_period_spec",
    "period_cache_reference",
    "period_metadata",
    "resolve_period_bounds",
]

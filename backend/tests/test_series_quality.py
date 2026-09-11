from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = ROOT / "backend"
for path in (ROOT, BACKEND_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from backend.custom_indicators.errors import ValidationError  # noqa: E402
from backend.custom_indicators.series_provider import (  # noqa: E402
    InstrumentIdentity,
    ProductSeries,
    select_period_window,
)
from backend.series_quality import finite_coverage, series_quality_execution_audit  # noqa: E402


def _product_series(frame: pd.DataFrame) -> ProductSeries:
    return ProductSeries(
        identity=InstrumentIdentity("fund", "TEST.OF", "TEST.OF", "测试基金"),
        frame=frame,
        fingerprint="fixture",
        data_latest_date=frame.iloc[-1]["date"].strftime("%Y-%m-%d"),
    )


def test_finite_coverage_uses_the_shared_fixed_signature_njit_lane() -> None:
    count, ratio = finite_coverage(
        pd.Series([1.0, np.nan, np.inf, 2.0], dtype=np.float64)
    )

    assert count == 2
    assert ratio == pytest.approx(0.5)
    audit = series_quality_execution_audit()
    assert audit["backend"] == "numba_njit_fixed_signature"
    assert audit["nopython"] is True
    assert audit["python_fallback"] == 0


def test_period_window_requires_dense_full_interval() -> None:
    dates = pd.date_range("2025-08-31", "2026-08-31", freq="5D")
    frame = pd.DataFrame({"date": dates, "value": 1.0 + np.arange(len(dates)) * 0.001})

    with pytest.raises(ValidationError) as exc_info:
        select_period_window(_product_series(frame), "1Y")

    assert exc_info.value.code == "INCOMPLETE_PERIOD_COVERAGE"


def test_period_window_rejects_adjusted_nav_dislocation() -> None:
    dates = pd.bdate_range("2025-08-28", "2026-08-28")
    smooth = 1.0 + np.arange(len(dates)) * 0.0001
    adjusted = smooth.copy()
    adjusted[dates >= pd.Timestamp("2026-06-02")] *= 100.0
    frame = pd.DataFrame(
        {
            "date": dates,
            "value": adjusted,
            "unit_nav": smooth,
            "accum_nav": smooth,
        }
    )

    with pytest.raises(ValidationError) as exc_info:
        select_period_window(_product_series(frame), "1Y")

    assert exc_info.value.code == "ADJUSTED_NAV_ANOMALY"


def test_period_window_accepts_dense_complete_interval() -> None:
    dates = pd.bdate_range("2025-08-28", "2026-08-28")
    values = 1.0 + np.arange(len(dates)) * 0.0001
    window = select_period_window(
        _product_series(pd.DataFrame({"date": dates, "value": values})),
        "1Y",
    )

    assert window.start_date == "2025-08-28"
    assert window.end_date == "2026-08-28"
    assert window.observation_count == len(dates) - 1


def test_previous_calendar_year_is_not_a_rolling_year() -> None:
    dates = pd.bdate_range("2023-12-29", "2026-08-31")
    values = 1.0 + np.arange(len(dates)) * 0.0001
    source = _product_series(pd.DataFrame({"date": dates, "value": values}))

    last_year = select_period_window(source, "Y1", "2026-08-31")
    year_before_last = select_period_window(source, "Y2", "2026-08-31")

    assert last_year.start_date == "2024-12-31"
    assert last_year.end_date == "2025-12-31"
    assert year_before_last.start_date == "2023-12-29"
    assert year_before_last.end_date == "2024-12-31"


def test_previous_calendar_week_includes_prior_close_anchor() -> None:
    dates = pd.bdate_range("2026-08-14", "2026-09-02")
    values = 1.0 + np.arange(len(dates)) * 0.001
    source = _product_series(pd.DataFrame({"date": dates, "value": values}))

    previous_week = select_period_window(source, "W1", "2026-09-02")
    two_weeks_ago = select_period_window(source, "W2", "2026-09-02")

    assert previous_week.start_date == "2026-08-21"
    assert previous_week.end_date == "2026-08-28"
    assert previous_week.observation_count == 5
    assert two_weeks_ago.start_date == "2026-08-14"
    assert two_weeks_ago.end_date == "2026-08-21"
    assert two_weeks_ago.observation_count == 5


def test_lifetime_period_keeps_first_real_observation() -> None:
    dates = pd.bdate_range("2024-03-04", "2026-08-31")
    values = 1.0 + np.arange(len(dates)) * 0.0001

    window = select_period_window(
        _product_series(pd.DataFrame({"date": dates, "value": values})),
        "ALL",
        "2026-08-31",
    )

    assert window.start_date == "2024-03-04"
    assert window.end_date == "2026-08-31"
    assert window.observation_count == len(dates) - 1

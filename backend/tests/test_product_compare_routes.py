from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
from fastapi import HTTPException


ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = ROOT / "backend"
for path in (ROOT, BACKEND_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from backend.services import instrument_routes  # noqa: E402


def _points() -> list[dict[str, object]]:
    dates = pd.bdate_range("2025-01-02", periods=8)
    return [
        {"date": date.strftime("%Y-%m-%d"), "close": 1.0 + index * 0.01}
        for index, date in enumerate(dates)
    ]


def _request() -> instrument_routes.ProductCompareAnalysisRequest:
    return instrument_routes.ProductCompareAnalysisRequest(
        ranges={
            "performance": {"start_date": None, "end_date": None},
            "risk": {"start_date": None, "end_date": None},
            "efficiency": {"start_date": None, "end_date": None},
        },
        rolling_window_days=3,
        management_fee=0.5,
        custody_fee=0.1,
    )


def test_product_compare_route_delegates_to_fixed_signature_njit(monkeypatch) -> None:
    monkeypatch.setattr(instrument_routes, "_load_timeseries", lambda _kind, _code: _points())

    response = instrument_routes.instrument_product_compare_analysis(
        "510300.SH",
        _request(),
        "etf",
    )

    assert response["product_id"] == "510300.SH"
    assert response["ranges"]["performance"]["window"]["observation_count"] == 8
    assert response["execution"]["execution_backend"] == "numba_njit_fixed_signature"
    assert response["execution"]["nopython"] is True
    assert response["execution"]["object_mode"] == 0
    assert response["execution"]["python_fallback"] == 0
    assert response["execution"]["request_time_compilation"] == 0
    assert all(
        len(signatures) == 1
        for signatures in response["execution"]["kernel_signatures"].values()
    )


def test_product_compare_route_returns_404_without_real_series(monkeypatch) -> None:
    monkeypatch.setattr(instrument_routes, "_load_timeseries", lambda _kind, _code: [])

    with pytest.raises(HTTPException) as exc_info:
        instrument_routes.instrument_product_compare_analysis(
            "510300.SH",
            _request(),
            "etf",
        )

    assert exc_info.value.status_code == 404


def test_product_compare_route_maps_invalid_window_to_chinese_400(monkeypatch) -> None:
    monkeypatch.setattr(instrument_routes, "_load_timeseries", lambda _kind, _code: _points())
    request = _request()
    request.ranges.performance.start_date = "not-a-date"

    with pytest.raises(HTTPException) as exc_info:
        instrument_routes.instrument_product_compare_analysis(
            "510300.SH",
            request,
            "etf",
        )

    assert exc_info.value.status_code == 400
    assert "产品比较计算失败" in exc_info.value.detail

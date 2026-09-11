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

from backend.compute_policy import validate_execution_audit  # noqa: E402
from backend.instrument_analytics_numba import (  # noqa: E402
    instrument_analytics_numba_execution_audit,
    product_compare_analysis_kernel,
    warm_instrument_analytics_numba_kernels,
)
from backend.services.product_compare import build_product_compare_response  # noqa: E402


def _points(count: int) -> list[dict[str, object]]:
    dates = pd.bdate_range("2000-01-03", periods=count)
    positions = np.arange(count, dtype=np.float64)
    values = 1.0 + positions * 0.0002 + np.sin(positions / 17.0) * 0.01
    return [
        {"date": date.strftime("%Y-%m-%d"), "close": float(value)}
        for date, value in zip(dates, values)
    ]


def _parameters(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "ranges": {
            "performance": {"start_date": None, "end_date": None},
            "risk": {"start_date": None, "end_date": None},
            "efficiency": {"start_date": None, "end_date": None},
        },
        "rolling_window_days": 30,
        "management_fee": 0.5,
        "custody_fee": 0.1,
    }
    values.update(overrides)
    return values


def test_product_compare_5000_points_reuses_one_fixed_signature() -> None:
    audit = validate_execution_audit(warm_instrument_analytics_numba_kernels())
    before = tuple(str(signature) for signature in product_compare_analysis_kernel.signatures)

    response = build_product_compare_response(
        product_id="510300.SH",
        points=_points(5_000),
        parameters=_parameters(),
    )

    execution = validate_execution_audit(response["execution"])
    assert execution["execution_backend"] == "numba_njit_fixed_signature"
    assert execution["nopython"] is True
    assert execution["object_mode"] == 0
    assert execution["python_fallback"] == 0
    assert execution["request_time_compilation"] == 0
    assert all(len(signatures) == 1 for signatures in execution["kernel_signatures"].values())
    assert audit["kernel_signatures"]["product_compare_analysis_kernel"]
    assert len(response["ranges"]["performance"]["normalized_nav"]) == 5_000
    assert len(response["ranges"]["risk"]["drawdown"]) == 5_000
    assert len(response["ranges"]["risk"]["rolling_volatility"]) == 5_000
    assert before == tuple(str(signature) for signature in product_compare_analysis_kernel.signatures)
    assert len(product_compare_analysis_kernel.nopython_signatures) == 1


def test_product_compare_metrics_and_series_are_computed_by_kernel() -> None:
    points = [
        {"date": "2024-01-02", "close": 1.0},
        {"date": "2024-01-03", "close": 1.1},
        {"date": "2024-01-04", "close": 0.99},
        {"date": "2024-01-05", "close": 1.2},
    ]

    response = build_product_compare_response(
        product_id="fixture",
        points=points,
        parameters=_parameters(rolling_window_days=3),
    )
    result = response["ranges"]["performance"]

    assert result["metrics"]["cumulativeReturn"] == pytest.approx(20.0)
    assert result["metrics"]["totalFee"] == pytest.approx(0.6)
    assert result["metrics"]["returnToFee"] == pytest.approx(20.0 / 0.6)
    assert result["metrics"]["maxDrawdown"] == pytest.approx(-10.0)
    assert [item["value"] for item in result["normalized_nav"]] == pytest.approx(
        [1.0, 1.1, 0.99, 1.2]
    )
    assert [item["value"] for item in result["drawdown"]] == pytest.approx(
        [0.0, 0.0, -10.0, 0.0]
    )
    assert result["rolling_volatility"][-1]["value"] is not None


@pytest.mark.parametrize(
    ("points", "message"),
    [
        ([{"date": "2024-01-02", "close": 1.0}], "至少需要两个"),
        (
            [
                {"date": "2024-01-02", "close": 1.0},
                {"date": "2024-01-03", "close": 0.0},
            ],
            "有限正数",
        ),
        (
            [
                {"date": "bad-date", "close": 1.0},
                {"date": "2024-01-03", "close": 1.1},
            ],
            "无效日期",
        ),
    ],
)
def test_product_compare_invalid_data_fails_closed(
    points: list[dict[str, object]],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_product_compare_response(
            product_id="fixture",
            points=points,
            parameters=_parameters(),
        )


def test_product_compare_request_range_is_enforced() -> None:
    parameters = _parameters()
    parameters["ranges"] = {
        name: {"start_date": "2024-01-03", "end_date": "2024-01-04"}
        for name in ("performance", "risk", "efficiency")
    }
    response = build_product_compare_response(
        product_id="fixture",
        points=[
            {"date": "2024-01-02", "close": 1.0},
            {"date": "2024-01-03", "close": 1.1},
            {"date": "2024-01-04", "close": 1.2},
            {"date": "2024-01-05", "close": 1.3},
        ],
        parameters=parameters,
    )

    assert response["ranges"]["performance"]["window"] == {
        "start_date": "2024-01-03",
        "end_date": "2024-01-04",
        "observation_count": 2,
    }
    assert response["ranges"]["performance"]["metrics"]["cumulativeReturn"] == pytest.approx(
        (1.2 / 1.1 - 1.0) * 100.0
    )

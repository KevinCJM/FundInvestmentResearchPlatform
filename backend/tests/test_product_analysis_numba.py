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
from backend.product_analysis_numba import (  # noqa: E402
    bollinger_kernel,
    daily_returns_percent_kernel,
    kdj_kernel,
    moving_average_kernel,
    product_analysis_execution_audit,
    return_statistics_kernel,
    warm_product_analysis_numba_kernels,
)
from backend.services.product_analysis import build_product_analysis_response  # noqa: E402


def _parameters(**overrides) -> dict[str, object]:
    values: dict[str, object] = {
        "statistics_period": "ALL",
        "price_ma_periods": [5, 10, 20],
        "volume_ma_periods": [5, 10],
        "boll_period": 20,
        "boll_multiplier": 2.0,
        "kdj_period": 9,
        "kdj_k_smoothing": 3,
        "kdj_d_smoothing": 3,
        "histogram_bin_width": 0.2,
        "simulation_horizon": 21,
        "simulation_path_count": 200,
        "bootstrap_block_length": 10,
        "simulation_target_return": 5.0,
        "simulation_run": 0,
        "regime": None,
    }
    values.update(overrides)
    return values


def _points(count: int) -> list[dict[str, object]]:
    dates = pd.bdate_range("2000-01-03", periods=count)
    positions = np.arange(count, dtype=np.float64)
    close = 1.0 + positions * 0.0002 + np.sin(positions / 13.0) * 0.01
    return [
        {
            "date": date.strftime("%Y-%m-%d"),
            "open": value - 0.001,
            "high": value + 0.01,
            "low": value - 0.01,
            "close": value,
            "volume": 1_000.0 + index,
        }
        for index, (date, value) in enumerate(zip(dates, close))
    ]


def _signature_snapshot() -> dict[str, tuple[str, ...]]:
    return {
        name: tuple(signatures)
        for name, signatures in product_analysis_execution_audit()[
            "kernel_signatures"
        ].items()
    }


def test_product_analysis_warmup_is_fixed_signature_nopython() -> None:
    audit = validate_execution_audit(warm_product_analysis_numba_kernels())

    assert audit["execution_backend"] == "numba_njit_fixed_signature"
    assert audit["kernel_coverage"] == "21/21"
    assert audit["nopython"] is True
    assert audit["object_mode"] == 0
    assert audit["python_fallback"] == 0
    assert all(len(signatures) == 1 for signatures in audit["kernel_signatures"].values())


def test_5000_point_full_response_does_not_compile_new_signatures() -> None:
    warm_product_analysis_numba_kernels()
    before = _signature_snapshot()

    response = build_product_analysis_response(
        product_id="510300.SH",
        points=_points(5_000),
        parameters=_parameters(),
    )

    assert response["returnStatistics"]["sampleSize"] == 4_999
    assert response["simulation"]["parametric"]["method"] == "parametric"
    assert response["simulation"]["blockBootstrap"]["method"] == "block_bootstrap"
    assert sum(
        item["count"]
        for item in response["simulation"]["densities"]["parametric"]["histogram"]
    ) == 200
    assert response["execution"]["python_fallback"] == 0
    assert response["technical"]["availability"] == {
        "ohlc": True,
        "volume": True,
        "kdj": True,
    }
    assert before == _signature_snapshot()


def test_simulations_are_deterministic_and_run_seed_changes_output() -> None:
    points = _points(120)
    first = build_product_analysis_response(
        product_id="510300.SH",
        points=points,
        parameters=_parameters(),
    )
    second = build_product_analysis_response(
        product_id="510300.SH",
        points=points,
        parameters=_parameters(),
    )
    rerun = build_product_analysis_response(
        product_id="510300.SH",
        points=points,
        parameters=_parameters(simulation_run=1),
    )

    assert first["simulation"] == second["simulation"]
    assert (
        first["simulation"]["parametric"]["terminal"]["p50"]
        != rerun["simulation"]["parametric"]["terminal"]["p50"]
    )


def test_technical_and_statistics_kernels_match_controlled_reference() -> None:
    close = np.ascontiguousarray(np.array([1.0, 1.1, 1.2, 1.1, 1.3]))
    periods = np.ascontiguousarray(np.array([3], dtype=np.int64))
    moving = moving_average_kernel(close, periods)
    bollinger = bollinger_kernel(close, 3, 2.0)
    kdj = kdj_kernel(close + 0.1, close - 0.1, close, 3, 3, 3)
    returns = daily_returns_percent_kernel(close)
    statistics = return_statistics_kernel(returns)

    reference_window = close[-3:]
    assert moving[0, -1] == pytest.approx(reference_window.mean())
    assert bollinger[1, -1] == pytest.approx(reference_window.mean())
    assert bollinger[0, -1] == pytest.approx(
        reference_window.mean() + 2.0 * reference_window.std(ddof=0)
    )
    assert np.isfinite(kdj[:, -1]).all()
    reference_returns = (close[1:] / close[:-1] - 1.0) * 100.0
    assert returns == pytest.approx(reference_returns)
    assert statistics[0] == pytest.approx(reference_returns.mean())
    assert statistics[1] == pytest.approx(reference_returns.std(ddof=0))


def test_regime_statistics_exclude_cross_boundary_returns() -> None:
    regime = {
        "states": [
            {"id": "bull", "label": "牛市", "color": "#16a34a"},
            {"id": "range", "label": "震荡", "color": "#f59e0b"},
        ],
        "segments": [
            {"state_id": "bull", "start_date": "2000-01-03", "end_date": "2000-01-05"},
            {"state_id": "range", "start_date": "2000-01-06", "end_date": "2000-01-10"},
        ],
    }
    response = build_product_analysis_response(
        product_id="510300.SH",
        points=_points(40),
        parameters=_parameters(regime=regime),
    )

    statistics = {item["stateId"]: item for item in response["regimeStatistics"]}
    assert statistics["bull"]["returnObservations"] == 2
    assert statistics["range"]["returnObservations"] == 2
    assert statistics["bull"]["observations"] == 3
    assert statistics["range"]["observations"] == 3

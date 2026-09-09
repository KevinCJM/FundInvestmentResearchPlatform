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
        "include_simulation": True,
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

    statistics = {item["stateId"]: item for item in response["regimeAnalysis"]["states"]}
    assert statistics["bull"]["returnObservations"] == 2
    assert statistics["range"]["returnObservations"] == 2
    assert statistics["bull"]["observations"] == 3
    assert statistics["range"]["observations"] == 3


def _regime_for(points, boundaries, **selection):
    return {
        "states": [{"id": "bear", "label": "熊市"}, {"id": "bull", "label": "牛市"}],
        "segments": [
            {"id": f"segment-{index}", "state_id": state, "start_date": points[start]["date"], "end_date": points[end]["date"]}
            for index, (state, start, end) in enumerate(boundaries)
        ],
        **selection,
    }


def test_disjoint_segments_have_independent_paths_and_conditioned_distributions():
    points = _points(5)
    for point, close in zip(points, [100.0, 90.0, 150.0, 110.0, 99.0]):
        point["close"] = close
    regime = _regime_for(points, [("bear", 0, 1), ("bull", 2, 2), ("bear", 3, 4)], state_id="bear")
    response = build_product_analysis_response(product_id="test", points=points, parameters=_parameters(regime=regime))
    bear, bull = response["regimeAnalysis"]["states"]
    assert bear["worstSegmentDrawdown"] == pytest.approx(-0.1)
    assert bear["medianSegmentReturn"] == pytest.approx(-0.1)
    assert bear["segmentCount"] == bear["eligibleSegmentCount"] == 2
    assert bear["returnObservations"] == response["returnStatistics"]["sampleSize"] == 2
    assert response["returnStatistics"]["mean"] == pytest.approx(-10.0)
    assert [item["date"] for item in response["dailyReturns"]] == [points[1]["date"], points[4]["date"]]
    assert bull["worstSegmentDrawdown"] is None
    assert response["regimeAnalysis"]["segments"][1]["cumulativeReturn"] is None
    assert response["simulationStatus"] == "insufficient_sample"


@pytest.mark.parametrize("invalid", [np.nan, np.inf, 0.0, -1.0])
def test_missing_prices_never_bridge_returns_or_qualify_as_complete_paths(invalid):
    points = _points(4)
    for point, value in zip(points, [1.0, invalid, 0.8, 0.88]):
        point["close"] = value
    regime = _regime_for(points, [("bear", 0, 3)], state_id="bear")
    response = build_product_analysis_response(product_id="test", points=points, parameters=_parameters(regime=regime))
    segment = response["regimeAnalysis"]["segments"][0]
    assert segment["observations"] == 4
    assert segment["validObservations"] == 3
    assert segment["returnObservations"] == 1
    assert segment["cumulativeReturn"] is segment["maxDrawdown"] is None
    assert segment["status"] == "missing_data"
    assert response["returnStatistics"]["mean"] == pytest.approx(10.0)
    assert response["regimeAnalysis"]["states"][0]["eligibleSegmentCount"] == 0


def test_single_segment_selection_and_statistics_window_share_the_same_axis():
    points = _points(600)
    regime = _regime_for(points, [("bear", 0, 10), ("bear", 590, 599)], state_id="bear", segment_id="segment-1")
    response = build_product_analysis_response(product_id="test", points=points, parameters=_parameters(statistics_period="1M", regime=regime, include_simulation=False))
    assert [segment["id"] for segment in response["regimeAnalysis"]["segments"]] == ["segment-1"]
    assert response["regimeAnalysis"]["states"][0]["segmentCount"] == 1
    assert response["returnStatistics"]["sampleSize"] == 9
    assert response["researchContext"]["startDate"] == points[590]["date"]
    assert response["researchContext"]["endDate"] == points[599]["date"]
    assert response["researchContext"]["windowStartDate"] < points[590]["date"]
    assert response["researchContext"]["scope"] == "segment"


def test_explicit_simulation_uses_only_the_selected_state_and_keeps_technical_axis():
    points = _points(44)
    regime = _regime_for(points, [("bear", 0, 10), ("bull", 11, 21), ("bear", 22, 32), ("bull", 33, 43)], state_id="bear")
    response = build_product_analysis_response(product_id="test", points=points, parameters=_parameters(regime=regime, include_simulation=False))
    assert response["simulation"] is None
    assert response["simulationStatus"] == "not_requested"
    assert response["researchContext"]["simulationEligible"] is True
    assert len(response["technical"]["priceMa"]["5"]) == len(points)
    response = build_product_analysis_response(product_id="test", points=points, parameters=_parameters(regime=regime, include_simulation=True))
    assert response["simulationStatus"] == "complete"
    assert response["simulation"]["parametric"]["assumptions"]["sourceObservationCount"] == 20
    assert response["simulation"]["blockBootstrap"]["assumptions"]["sourceObservationCount"] == 20
    assert response["schema_version"] == 2
    assert "regimeStatistics" not in response


def test_incomplete_window_and_outside_state_return_explanations_not_zeros():
    points = _points(5)
    regime = _regime_for(points, [("bear", 0, 4)], state_id="bull")
    response = build_product_analysis_response(product_id="test", points=points, parameters=_parameters(regime=regime, statistics_period="1Y"))
    assert response["window"]["complete"] is False
    assert response["returnStatistics"]["sampleSize"] == 0
    assert response["returnStatistics"]["mean"] is None
    assert response["regimeAnalysis"]["segments"] == []
    assert response["researchContext"]["startDate"] is None
    assert response["simulationStatus"] == "insufficient_sample"


def test_overflowed_positive_price_ratio_is_not_a_valid_return():
    points = _points(2)
    points[0]["close"], points[1]["close"] = 1e-300, 1e300
    response = build_product_analysis_response(product_id="test", points=points, parameters=_parameters(include_technical=False, regime=_regime_for(points, [("bear", 0, 1)], state_id="bear")))
    assert response["researchContext"]["returnObservations"] == 0
    assert response["returnStatistics"]["sampleSize"] == 0
    assert response["regimeAnalysis"]["segments"][0]["status"] == "missing_data"


@pytest.mark.parametrize("boundary_kind", ["segments", "missing"])
def test_bootstrap_restarts_at_segment_and_missing_boundaries_without_wrapping(boundary_kind):
    from backend.product_analysis_numba import stationary_block_bootstrap_kernel

    values = np.linspace(-2.0, 2.0, 20)
    if boundary_kind == "missing":
        returns = np.full(40, np.nan)
        returns[::2] = values
        segments = np.zeros(40, dtype=np.int64)
    else:
        returns = values.copy()
        segments = np.arange(20, dtype=np.int64)
    # Each valid return is isolated. Therefore every next draw must restart;
    # a long requested block cannot follow the next compressed array element.
    result = stationary_block_bootstrap_kernel(np.ascontiguousarray(returns), segments, 1.0, 30, 1, 7, 5.0, 20)
    state = 7
    def draw():
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        return (state >> 11) / float(1 << 53)
    source = int(draw() * 20)
    expected = [1.0]
    for day in range(30):
        if day:
            draw()  # restart decision
            source = int(draw() * 20)
        expected.append(expected[-1] * (1.0 + values[source] / 100.0))
    assert result[0][0] == pytest.approx(expected)


def _bootstrap_mass_balance_reference(returns, segment_ids, horizon, path_count, seed, block_length):
    """Solve restart mass from the continuation matrix, independently of CDF weights."""
    positions = np.flatnonzero(np.isfinite(returns) & (returns > -100.0))
    count = len(positions)
    restart_probability = 1.0 / min(count, max(1, block_length))
    continuation = np.zeros((count, count))
    for source, physical_index in enumerate(positions[:-1]):
        following = positions[source + 1]
        if following == physical_index + 1 and segment_ids[following] == segment_ids[physical_index]:
            continuation[source, source + 1] = 1.0 - restart_probability
    uniform = np.full(count, 1.0 / count)
    missing_incoming_mass = uniform - uniform @ continuation
    restart = missing_incoming_mass / missing_incoming_mass.sum()
    transition = continuation + np.outer(1.0 - continuation.sum(axis=1), restart)
    np.testing.assert_allclose(uniform @ transition, uniform, rtol=0.0, atol=1e-14)
    cdf = np.cumsum(restart)
    state = seed

    def draw():
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        return (state >> 11) / float(1 << 53)

    paths = np.ones((path_count, horizon + 1))
    for path in range(path_count):
        # Day one is uniform over observations, not over restart positions.
        source = int(draw() * count)
        for day in range(1, horizon + 1):
            if day > 1:
                decision = draw()
                if source + 1 < count and continuation[source, source + 1] > 0.0 and decision >= restart_probability:
                    source += 1
                else:
                    source = min(count - 1, int(np.searchsorted(cdf, draw(), side="right")))
            paths[path, day] = paths[path, day - 1] * (1.0 + returns[positions[source]] / 100.0)
    return paths


@pytest.mark.parametrize("with_invalid_rows", [False, True])
def test_bootstrap_unequal_segments_match_independent_mass_balance_reference(with_invalid_rows):
    from backend.product_analysis_numba import stationary_block_bootstrap_kernel

    returns = np.linspace(-1.1, 1.1, 23)
    segments = np.repeat(np.arange(5, dtype=np.int64), [1, 2, 4, 7, 9])
    if with_invalid_rows:
        # These gaps split a segment even when the declared segment id stays equal.
        offsets = [5, 10, 18]
        returns = np.insert(returns, offsets, [np.nan, np.inf, -100.0])
        segments = np.insert(segments, offsets, segments[offsets])
    signature_before = _signature_snapshot()
    expected = _bootstrap_mass_balance_reference(returns, segments, 63, 12, 37, 20)
    result = stationary_block_bootstrap_kernel(returns, segments, 1.0, 63, 12, 37, 5.0, 20)
    np.testing.assert_allclose(result[0], expected, rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(result[2], np.sort(expected[:, -1]), rtol=1e-13, atol=1e-14)
    assert np.isfinite(result[0]).all()
    assert _signature_snapshot() == signature_before


@pytest.mark.parametrize("reverse_pair", [False, True])
def test_bootstrap_fragmented_pairs_preserve_uniform_return_marginals(reverse_pair):
    from backend.product_analysis_numba import stationary_block_bootstrap_kernel

    pair = np.array([1.0, -1.0] if reverse_pair else [-1.0, 1.0])
    returns = np.tile(pair, 20)
    segments = np.repeat(np.arange(20, dtype=np.int64), 2)
    result = stationary_block_bootstrap_kernel(returns, segments, 1.0, 504, 1_000, 7, 5.0, 20)
    # All terminal paths are returned, so recover the exact average positive
    # fraction without relying on the first twelve display paths or NAV means.
    positive_counts = (np.log(result[2]) - 504 * np.log(0.99)) / (np.log(1.01) - np.log(0.99))
    assert float(positive_counts.mean() / 504) == pytest.approx(0.5, abs=0.01)
    # The former uniform-restart kernel yields about 2/3 (or 1/3) positives.
    assert result[3][2] == pytest.approx(0.99 ** 252 * 1.01 ** 252, rel=0.10)


def test_bootstrap_initial_draw_is_uniform_for_unequal_segments():
    from backend.product_analysis_numba import stationary_block_bootstrap_kernel

    returns = np.arange(20, dtype=np.float64) / 100.0
    segments = np.repeat(np.arange(4, dtype=np.int64), [1, 2, 7, 10])
    result = stationary_block_bootstrap_kernel(returns, segments, 1.0, 1, 1_000, 19, 5.0, 20)
    state = 19
    initial_positions = []
    for _ in range(1_000):
        state = (state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        draw = (state >> 11) / float(1 << 53)
        initial_positions.append(int(draw * len(returns)))
    expected = np.sort(1.0 + returns[initial_positions] / 100.0)
    np.testing.assert_allclose(result[2], expected, rtol=0.0, atol=1e-14)


@pytest.mark.parametrize("selection", [{"state_id": "absent"}, {"segment_id": "segment-99"}, {"state_id": "bull", "segment_id": "segment-0"}])
def test_invalid_selection_cannot_silently_fall_back_to_full_history(selection):
    points = _points(5)
    with pytest.raises(ValueError):
        build_product_analysis_response(product_id="test", points=points, parameters=_parameters(regime=_regime_for(points, [("bear", 0, 4)], **selection)))

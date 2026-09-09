from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
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
from backend.services.product_analysis import (  # noqa: E402
    SIMULATION_METHODS,
    build_product_analysis_response,
)


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
    assert audit["kernel_coverage"] == "25/25"
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
    assert response["simulation"]["methods"] == list(SIMULATION_METHODS)
    assert all(
        response["simulation"]["byMethod"][method]["method"] == method
        for method in SIMULATION_METHODS
    )
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
        first["simulation"]["byMethod"]["parametric"]["terminal"]["p50"]
        != rerun["simulation"]["byMethod"]["parametric"]["terminal"]["p50"]
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
    assert response["simulation"]["byMethod"]["parametric"]["assumptions"]["sourceObservationCount"] == 20
    assert response["simulation"]["byMethod"]["block_bootstrap"]["assumptions"]["sourceObservationCount"] == 20
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


def _walk_points(start: str, count: int, seed: int, drift: float, vol: float, base: float = 1.0):
    """A geometric walk, so a fixture cannot wander through zero."""

    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=count)
    nav = base * np.exp(np.cumsum(rng.normal(drift, vol, count)))
    return [{"date": date.strftime("%Y-%m-%d"), "close": float(value)} for date, value in zip(dates, nav)]


_RESEARCH_POINTS = _walk_points("2010-01-04", 400, 1, 0.0003, 0.01)


def _future_points(count: int, seed: int, drift: float, vol: float):
    """Continue from the研究日 close — a seam jump would be the fixture's own bug."""

    return _walk_points("2012-01-02", count, seed, drift, vol, base=float(_RESEARCH_POINTS[-1]["close"]))


def _realized_response(future_points, as_of="2011-12-31", horizon=63):
    return build_product_analysis_response(
        product_id="510300.SH",
        points=[],
        parameters=_parameters(simulation_horizon=horizon, simulation_path_count=200, include_technical=False),
        research_points=_RESEARCH_POINTS,
        future_points=future_points,
        as_of=as_of,
    )


def test_realized_path_scores_a_crash_below_the_simulated_band() -> None:
    """The point of the panel: a研究日 whose future is already on disk can be graded."""

    future = _future_points(80, 3, -0.006, 0.02)
    simulation = _realized_response(future)["simulation"]

    assert simulation["realizedStatus"] == "complete"
    realized = simulation["realized"]
    assert realized["coveredDays"] == 63
    assert realized["complete"] is True
    assert realized["nav"][0] == 1.0
    # Day 0 is the anchor, so the path carries one more point than the horizon.
    assert len(realized["nav"]) == len(realized["dates"]) == 64
    assert realized["dates"][1] == "2012-01-02"
    assert realized["terminalReturn"] < -0.2
    for method in ("parametric", "block_bootstrap"):
        score = realized["byMethod"][method]
        assert score["band"] == 0
        assert score["percentileRank"] == 0.0
        # Not asserted against a threshold: one path's containment ratio has a
        # large sampling variance — an ordinary future can sit outside the band
        # for longer than a crash does — so the terminal rank is the score and
        # containment is only descriptive.
        assert 0.0 <= score["containmentRatio"] <= 1.0
        assert score["breachDays"] > 0
        assert score["worstBreachGap"] < 0.0
        assert "低估了下行" in score["verdict"]


def test_realized_path_reports_partial_coverage_without_a_rank() -> None:
    """Half a horizon still scores the days it has; only the exact rank waits."""

    simulation = _realized_response(_future_points(20, 4, 0.0003, 0.01))["simulation"]

    assert simulation["realizedStatus"] == "partial"
    realized = simulation["realized"]
    assert realized["coveredDays"] == 20
    assert realized["requestedDays"] == 63
    assert realized["complete"] is False
    # Ranking a 20-day outcome inside a 63-day terminal distribution would be
    # a category error, so the rank is withheld while the band still reports.
    assert realized["byMethod"]["parametric"]["percentileRank"] is None
    assert realized["byMethod"]["parametric"]["band"] is not None
    assert realized["byMethod"]["parametric"]["containmentRatio"] is not None


def test_realized_path_is_absent_without_a_research_day_or_future_rows() -> None:
    off = _realized_response([], as_of=None)["simulation"]
    assert off["realizedStatus"] == "off"
    assert off["realized"] is None

    empty = _realized_response([])["simulation"]
    assert empty["realizedStatus"] == "no_future_data"
    assert empty["realized"] is None

    blank = _realized_response([{"date": "2012-01-02", "close": None}])["simulation"]
    assert blank["realizedStatus"] == "no_future_data"
    assert blank["realized"] is None


def test_realized_path_never_reaches_the_simulation_inputs() -> None:
    """A leak here would silently turn the whole panel into hindsight."""

    baseline = _realized_response([])["simulation"]
    with_future = _realized_response(_future_points(80, 3, -0.006, 0.02))["simulation"]

    for method in SIMULATION_METHODS:
        assert baseline["byMethod"][method]["terminal"] == with_future["byMethod"][method]["terminal"]
        assert baseline["byMethod"][method]["assumptions"] == with_future["byMethod"][method]["assumptions"]
        assert baseline["byMethod"][method]["percentiles"] == with_future["byMethod"][method]["percentiles"]


# ---------------------------------------------------------------------------
# Filtered historical simulation
# ---------------------------------------------------------------------------


def _garch_series(count: int, seed: int, omega: float, alpha: float, beta: float) -> np.ndarray:
    """A true GARCH(1,1) path in return-percent units, for a recovery check."""

    generator = np.random.default_rng(seed)
    variance = omega / (1.0 - alpha - beta)
    log_returns = np.empty(count, dtype=np.float64)
    for index in range(count):
        log_returns[index] = np.sqrt(variance) * generator.standard_normal()
        variance = omega + alpha * log_returns[index] ** 2 + beta * variance
    return np.ascontiguousarray((np.exp(log_returns) - 1.0) * 100.0)


def test_garch_filter_recovers_the_coefficients_it_was_generated_from() -> None:
    """The fit is the only part of the FHS lanes that could be silently wrong.

    Everything downstream — the residual draws, the forward recursion — is
    arithmetic. If the estimator is off, the lane still produces plausible
    looking paths, which is exactly why it needs a fixture with a known answer.
    """

    from backend.product_analysis_numba import _volatility_filter

    omega, alpha, beta = 2e-6, 0.09, 0.88
    percent = _garch_series(3_000, 7, omega, alpha, beta)
    log_returns = np.ascontiguousarray(np.log1p(percent / 100.0))

    residuals, parameters = _volatility_filter(log_returns, 1, 0.94)

    assert parameters[2] == pytest.approx(alpha, abs=0.02)
    assert parameters[3] == pytest.approx(beta, abs=0.02)
    assert parameters[6] == pytest.approx(alpha + beta, abs=0.01)
    # Variance targeting pins the long-run level to the sample's, and the
    # residuals are rescaled to unit variance so the level is not applied twice.
    assert np.sqrt(parameters[5]) == pytest.approx(np.sqrt(omega / (1.0 - alpha - beta)), rel=0.1)
    assert float(residuals.var()) == pytest.approx(1.0, abs=0.02)

    # EWMA is the same recursion with the coefficients handed to it, and its
    # persistence is exactly 1 — the property that makes it never mean-revert.
    _, ewma = _volatility_filter(log_returns, 0, 0.94)
    assert ewma[1] == 0.0
    assert ewma[2] == pytest.approx(0.06)
    assert ewma[3] == pytest.approx(0.94)
    assert ewma[6] == pytest.approx(1.0)


def test_filtered_lanes_start_from_todays_volatility_and_the_others_do_not() -> None:
    """The one property that separates a conditional model from the rest.

    Two histories share 1200 identical days and differ only in the last 120: one
    ends calm, one ends in a storm. An unconditional lane can only notice the
    change through the whole-sample variance, so its day-one fan barely moves.
    """

    from backend.product_analysis_numba import (
        filtered_historical_simulation_kernel,
        parametric_monte_carlo_kernel,
    )

    generator = np.random.default_rng(11)
    shared = generator.standard_normal(1_200) * 0.008
    tails = {
        "calm": generator.standard_normal(120) * 0.003,
        "storm": generator.standard_normal(120) * 0.035,
    }
    day_one_band: dict[str, dict[str, float]] = {}
    for name, tail in tails.items():
        percent = np.ascontiguousarray((np.exp(np.concatenate([shared, tail])) - 1.0) * 100.0)
        unconditional = parametric_monte_carlo_kernel(percent, 1.0, 63, 600, 5, 5.0, 1)
        conditional = filtered_historical_simulation_kernel(percent, 1.0, 63, 600, 5, 5.0, 1, 0.94)
        day_one_band[name] = {
            "unconditional": float(unconditional[1][4, 1] - unconditional[1][0, 1]),
            "conditional": float(conditional[1][4, 1] - conditional[1][0, 1]),
        }

    unconditional_ratio = day_one_band["storm"]["unconditional"] / day_one_band["calm"]["unconditional"]
    conditional_ratio = day_one_band["storm"]["conditional"] / day_one_band["calm"]["conditional"]
    assert unconditional_ratio < 2.5
    assert conditional_ratio > 5.0


def test_normal_lane_reports_no_shape_calibration_and_differs_from_the_fitted_one() -> None:
    """Two lanes out of one kernel: the switch has to actually switch."""

    from backend.product_analysis_numba import parametric_monte_carlo_kernel

    # Strongly skewed returns, so a lane that ignored the shape is visible.
    generator = np.random.default_rng(3)
    log_returns = -np.abs(generator.standard_normal(1_500)) * 0.02 + 0.004
    percent = np.ascontiguousarray((np.exp(log_returns) - 1.0) * 100.0)

    normal = parametric_monte_carlo_kernel(percent, 1.0, 252, 800, 9, 5.0, 0)
    fitted = parametric_monte_carlo_kernel(percent, 1.0, 252, 800, 9, 5.0, 1)

    horizon = 252
    # Slot 8 is the shape-calibration status; the normal lane reports nothing.
    assert not np.isfinite(normal[4][horizon + 1 + 8])
    assert np.isfinite(fitted[4][horizon + 1 + 8])
    # Same seed, same mean and volatility — only the innovation shape differs,
    # and on a left-skewed sample it has to move the downside quantile.
    assert normal[3][0] != fitted[3][0]

    with pytest.raises(ValueError):
        parametric_monte_carlo_kernel(percent, 1.0, 252, 800, 9, 5.0, 2)


def test_simulation_lanes_share_one_read_only_input() -> None:
    """Zero-copy sharing is only safe if no kernel writes to its inputs.

    All five lanes receive the same `returns` array and run at once on threads,
    so a single in-place write would corrupt the other four non-deterministically
    — the worst possible failure mode to debug. Locking the input's bytes is
    cheaper than discovering that from a flaky percentile.
    """

    from backend.product_analysis_numba import (
        filtered_historical_simulation_kernel,
        parametric_monte_carlo_kernel,
        stationary_block_bootstrap_kernel,
    )

    percent = _garch_series(600, 13, 3e-6, 0.08, 0.9)
    segments = np.zeros(percent.size, dtype=np.int64)
    before = percent.copy()

    calls = (
        (parametric_monte_carlo_kernel, (percent, 1.0, 21, 40, 1, 5.0, 0)),
        (parametric_monte_carlo_kernel, (percent, 1.0, 21, 40, 1, 5.0, 1)),
        (stationary_block_bootstrap_kernel, (percent, segments, 1.0, 21, 40, 2, 5.0, 5)),
        (filtered_historical_simulation_kernel, (percent, 1.0, 21, 40, 3, 5.0, 0, 0.94)),
        (filtered_historical_simulation_kernel, (percent, 1.0, 21, 40, 4, 5.0, 1, 0.94)),
    )
    serial = [kernel(*arguments)[2] for kernel, arguments in calls]
    with ThreadPoolExecutor(max_workers=len(calls)) as pool:
        concurrent = [
            future.result()[2]
            for future in [pool.submit(kernel, *arguments) for kernel, arguments in calls]
        ]

    assert np.array_equal(percent, before)
    for lane, (expected, actual) in enumerate(zip(serial, concurrent)):
        assert np.array_equal(expected, actual), f"lane {lane} differed under threads"


def test_model_comparison_spreads_over_every_lane_that_ran() -> None:
    """The spread has to be max-minus-min, not first-minus-second."""

    from backend.product_analysis_numba import simulation_comparison_kernel

    # Three lanes; the widest gap sits between the first and the third.
    summaries = np.ascontiguousarray(np.array([
        [0.90, 0.95, 1.00, 1.05, 1.10, 0.40, 0.10, 0.12, 0.30, 0.20, -0.10, 0.00],
        [0.92, 0.96, 1.01, 1.06, 1.11, 0.42, 0.11, 0.13, 0.31, 0.21, -0.08, 0.01],
        [0.70, 0.85, 0.95, 1.02, 1.09, 0.55, 0.30, 0.34, 0.20, 0.35, -0.30, -0.05],
    ], dtype=np.float64))

    spread = simulation_comparison_kernel(summaries, 1.0)

    assert spread[0] == pytest.approx(0.92 - 0.70)
    assert spread[1] == pytest.approx(1.01 - 0.95)
    assert spread[2] == pytest.approx(0.55 - 0.40)
    assert spread[3] == pytest.approx(0.34 - 0.12)
    assert spread[4] == 2.0

    with pytest.raises(ValueError):
        simulation_comparison_kernel(np.ascontiguousarray(summaries[:1]), 1.0)


def test_terminal_density_axis_survives_one_runaway_path() -> None:
    """A single compounding path must not make every other one invisible.

    The filtered lanes can produce it: EWMA's persistence is exactly 1, so its
    simulated variance random-walks and over 504 days one path in a thousand can
    reach absurd multiples. The quantiles shrug that off — but the density
    kernel's bandwidth floor was anchored on the mean, so the outlier dragged
    the floor, and with it the chart's whole nav axis, into six figures.
    """

    from backend.product_analysis_numba import terminal_density_kernel

    terminal = np.ascontiguousarray(np.linspace(0.8, 1.3, 1_000))
    percentiles = np.ascontiguousarray(np.tile(np.array([[0.85], [0.95], [1.0], [1.1], [1.25]]), (1, 3)))

    clean = terminal_density_kernel(terminal, percentiles, 81, 1.0)[2]

    runaway = terminal.copy()
    runaway[-1] = 1.5e12
    with_outlier = terminal_density_kernel(np.ascontiguousarray(np.sort(runaway)), percentiles, 81, 1.0)[2]

    # navAxisMin / navAxisMax are slots 5 and 6.
    assert with_outlier[6] < 2.0
    assert with_outlier[5] == pytest.approx(clean[5], abs=0.05)
    assert with_outlier[6] == pytest.approx(clean[6], abs=0.05)

"""Independent target-grid checks; reference solvers are test-only."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize

import optimizer
from backend.qp_numba import feasible_qp_kernel


@pytest.fixture(scope="module", autouse=True)
def warmed():
    optimizer.warm_optimizer_numba_kernels()


def _two_assets():
    raw = np.random.default_rng(84).normal(size=(80, 2))
    raw -= raw.mean(axis=0)
    basis, _ = np.linalg.qr(raw)
    values = basis * np.sqrt(79) * np.array([.1, .2]) / np.sqrt(252) + np.array([.03, .09]) / 252
    return pd.DataFrame(values, columns=["defensive", "growth"])


def _calculate(frame, count=20, **options):
    grid = {"point_count": count, "max_iterations": 300, **options.pop("grid", {})}
    return optimizer.calculate_efficient_frontier_exploration(
        frame, {"metric": "annual", "days": 252},
        options.pop("risk_config", {"metric": "annual_vol", "days": 252}),
        rounds=[{"samples": 100, "step": 1., "buckets": 40}],
        frontier_grid=grid, **options)


@pytest.mark.parametrize("count", [20, 200])
def test_twenty_and_two_hundred_targets_match_analytic_frontier(count):
    result = _calculate(_two_assets(), count)
    grid = result["frontier_grid"]
    assert grid["requested_points"] == grid["attempted_points"] == grid["successful_points"] == grid["solver_calls"] == count
    assert grid["failed_points"] == grid["unattempted_points"] == 0
    assert len(grid["points"]) == len(grid["curve"]) == count
    assert grid["duplicate_solutions"] == 0
    np.testing.assert_allclose([p["target"] for p in grid["points"]], np.linspace(.042, .09, count), atol=1e-9)
    for point in grid["points"]:
        growth = (point["target"] - .03) / .06
        np.testing.assert_allclose(point["weights"], [1 - growth, growth], atol=1e-7)
        expected_risk = np.sqrt(.01 * (1 - growth) ** 2 + .04 * growth ** 2)
        assert point["value"][0] == pytest.approx(expected_risk, abs=1e-8)
        assert point["value"][1] == pytest.approx(point["target"], abs=1e-9)
        assert point["constraint_violation"] <= 1e-7
        assert point["candidate_index"] is not None
        np.testing.assert_allclose(result["scatter"][point["candidate_index"]]["weights"], point["weights"])
    assert result["accepted_candidates"] == result["sampled_candidates"] + result["refined_candidates"] + result["grid_candidates"]
    assert result["accepted_candidates"] == len(result["scatter"])
    assert sum(point is not None for point in grid["curve"]) == count


@pytest.mark.parametrize("count", [20, 200])
def test_each_grid_qp_matches_independent_constrained_solver(count):
    rng = np.random.default_rng(1461)
    values = rng.normal(0, .01, (80, 3)) + rng.uniform(0, .001, 3)
    lows, highs = rng.uniform(.001, .2, 3), rng.uniform(.5, .9, 3)
    bounds = list(zip(lows, highs, strict=True))
    means, covariance = values.mean(axis=0) * 252, np.cov(values, rowvar=False) * 252
    result = _calculate(pd.DataFrame(values), count, single_limits=bounds,
                        group_limits={(0, 1): (.431, .763)}, quantize_step=.005,
                        grid={"accept_continuous_weights": True})
    grid = result["frontier_grid"]
    assert grid["successful_points"] == count
    for point in grid["points"]:
        w = np.asarray(point["weights"])
        assert np.all(w >= lows - 1e-7) and np.all(w <= highs + 1e-7)
        assert .431 - 1e-7 <= w[:2].sum() <= .763 + 1e-7
        assert w.sum() == pytest.approx(1, abs=1e-7)
        assert w @ means >= point["target"] - 1e-7
        constraints = [
            {"type": "eq", "fun": lambda x: x.sum() - 1, "jac": lambda x: np.ones(3)},
            {"type": "ineq", "fun": lambda x: x[:2].sum() - .431, "jac": lambda x: np.array([1., 1., 0.])},
            {"type": "ineq", "fun": lambda x: .763 - x[:2].sum(), "jac": lambda x: np.array([-1., -1., 0.])},
            {"type": "ineq", "fun": lambda x: x @ means - point["target"], "jac": lambda x: means},
        ]
        reference = minimize(lambda x: x @ covariance @ x, w,
                             jac=lambda x: 2 * covariance @ x, method="SLSQP", bounds=bounds,
                             constraints=constraints, options={"maxiter": 1000, "ftol": 1e-12})
        assert reference.success, reference.message
        assert w @ covariance @ w == pytest.approx(reference.fun, abs=1e-8)
        portfolio = values @ w
        assert point["value"][0] == pytest.approx(portfolio.std(ddof=1) * np.sqrt(252), abs=1e-10)
        assert point["value"][1] == pytest.approx(portfolio.mean() * 252, abs=1e-10)


def test_duplicate_targets_are_counted_without_fake_density():
    result = _calculate(_two_assets(), 20, grid={"target_start": .06, "target_end": .06})
    grid = result["frontier_grid"]
    assert grid["attempted_points"] == grid["successful_points"] == 20
    assert grid["duplicate_targets"] == grid["duplicate_solutions"] == 19
    assert grid["added_candidates"] == 1
    assert grid["points"][1]["duplicate_of"] == 0


def test_infeasible_targets_are_not_plotted_or_added():
    result = _calculate(_two_assets(), 20, grid={"target_start": .10, "target_end": .20})
    grid = result["frontier_grid"]
    assert grid["attempted_points"] == grid["failed_points"] == 20
    assert grid["successful_points"] == grid["added_candidates"] == 0
    assert all(point["status"] == "infeasible_target" for point in grid["points"])
    assert grid["curve"] == [None] * 20


def test_exhausted_endpoint_budget_is_not_claimed_as_solved_frontier():
    result = _calculate(_two_assets(), grid={"max_iterations": 1})
    grid = result["frontier_grid"]
    assert grid["successful_points"] == 0
    assert grid["unattempted_points"] == 20
    assert any(point["status"] == "max_iterations" for point in grid["endpoints"])
    assert all(point["status"] == "range_unresolved" for point in grid["points"])


@pytest.mark.parametrize("invalid", [
    {"point_count": 1}, {"point_count": 201}, {"point_count": True}, {"point_count": 20.5},
    {"max_iterations": 0}, {"max_iterations": 1001}, {"target_start": .1},
    {"target_start": .2, "target_end": .1}, {"weight_domain": "discrete"},
])
def test_grid_contract_rejects_invalid_controls(invalid):
    with pytest.raises(ValueError):
        _calculate(_two_assets(), grid=invalid)


def test_sample_quantization_requires_explicit_continuous_grid_consent():
    with pytest.raises(ValueError, match="连续权重"):
        _calculate(_two_assets(), quantize_step=.005)


def test_final_frontier_and_all_representatives_use_grid_union():
    frame = _two_assets()
    result = _calculate(frame, 200, risk_free_rate=.015, use_local_refine=True, refine_iterations=20)
    points = np.asarray([p["value"] for p in result["scatter"]])
    for point in result["frontier"]:
        risk, ret = point["value"]
        dominated = (points[:, 0] <= risk) & (points[:, 1] >= ret) & ((points[:, 0] < risk - 1e-9) | (points[:, 1] > ret + 1e-9))
        assert not dominated.any()
    assert (result["max_sharpe"]["value"][1] - .015) / result["max_sharpe"]["value"][0] == pytest.approx(np.max((points[:, 1] - .015) / points[:, 0]))
    assert result["min_variance"]["value"][0] == pytest.approx(points[:, 0].min())
    assert result["max_return"]["value"][1] == pytest.approx(points[:, 1].max())
    for key, source in result["refinement"]["final_representatives"].items():
        assert result[key] == result["scatter"][source["candidate_index"]]
    assert result == _calculate(frame, 200, risk_free_rate=.015, use_local_refine=True, refine_iterations=20)


@pytest.mark.parametrize("risk", ["ewm_vol", "var", "es", "max_drawdown", "downside_vol"])
def test_other_risks_keep_actual_metric_and_honest_failure_states(risk):
    frame = _two_assets()
    result = _calculate(frame, risk_config={"metric": risk, "days": 252})
    grid = result["frontier_grid"]
    assert len(grid["points"]) == 20
    assert grid["successful_points"] + grid["failed_points"] + grid["unattempted_points"] == 20
    for point in grid["points"]:
        if point["status"] != "converged":
            assert point["candidate_index"] is None
            continue
        portfolio = frame.to_numpy() @ point["weights"]
        assert point["value"][0] == pytest.approx(optimizer.calculate_risk(portfolio, {"metric": risk, "days": 252}), abs=1e-10)
        assert point["value"][1] >= point["target"] - 1e-7
    if risk != "ewm_vol":
        assert grid["optimality_scope"] == "local_numerical_stationarity"


def test_qp_readonly_strided_inputs_and_signatures():
    h_owner = np.zeros((4, 4)); h_owner[::2, ::2] = np.diag([.01, .04])
    h = h_owner[::2, ::2]; h.setflags(write=False)
    a_owner = np.zeros((10, 4)); a_owner[::2, ::2] = np.vstack([np.ones(2), np.eye(2), -np.eye(2)])
    a = a_owner[::2, ::2]; a.setflags(write=False)
    b = np.array([1., 0., 0., -1., -1.]); b.setflags(write=False)
    initial = np.array([.5, .5]); initial.setflags(write=False)
    assert np.shares_memory(h, h_owner) and np.shares_memory(a, a_owner)
    before = [tuple(k.nopython_signatures) for k in optimizer.OPTIMIZER_NUMBA_KERNELS]
    solution, status, _, _ = feasible_qp_kernel(h, np.zeros(2), a, b, initial, 300, 1e-9)
    assert status == 0
    np.testing.assert_allclose(solution, [.8, .2], atol=1e-8)
    _calculate(_two_assets(), 200)
    assert before == [tuple(k.nopython_signatures) for k in optimizer.OPTIMIZER_NUMBA_KERNELS]
    np.testing.assert_array_equal(initial, [.5, .5])
    assert optimizer.optimizer_numba_status()["python_fallback"] == 0


@pytest.mark.parametrize("count", [20, 200])
def test_grid_api_returns_real_targets_and_separate_counts(monkeypatch, tmp_path, count):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from backend.services import analytics_routes as routes
    from pit.context import build_context

    frame = _two_assets()
    nav = np.vstack([np.ones(2), np.cumprod(1 + frame.to_numpy(), axis=0)])
    dates = pd.bdate_range("2024-01-01", periods=len(nav))
    pd.DataFrame([
        {"asset_alloc_name": "grid-api", "asset_name": name, "date": day, "nv": nav[i, j]}
        for i, day in enumerate(dates) for j, name in enumerate(frame.columns)
    ]).to_parquet(tmp_path / "asset_nv.parquet", index=False)
    monkeypatch.setattr(routes, "DATA_DIR", tmp_path)
    monkeypatch.setattr(routes, "resolve_request_context", lambda *_: build_context("2024-12-31"))
    app = FastAPI(); app.include_router(routes.router)
    with TestClient(app) as client:
        response = client.post("/api/efficient-frontier", json={
            "alloc_name": "grid-api", "start_date": "2024-01-01", "end_date": "2024-12-31",
            "return_metric": {"metric": "annual", "days": 252},
            "risk_metric": {"metric": "annual_vol", "days": 252},
            "frontier_grid": {"point_count": count, "max_iterations": 300},
        })
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["frontier_grid"]["successful_points"] == result["frontier_grid"]["solver_calls"] == count
    assert len(result["frontier_grid"]["points"]) == count
    assert result["accepted_candidates"] == len(result["scatter"])
    assert result["frontier_candidates"] == len(result["frontier"])
    assert result["grid_candidates"] == result["frontier_grid"]["added_candidates"]
    assert result["execution"]["python_fallback"] == 0


@pytest.mark.parametrize("payload", [{"point_count": True}, {"point_count": 201}, {"max_iterations": 0}, {"count": 20}])
def test_grid_api_strict_contract_rejects_ambiguous_controls(payload):
    from pydantic import ValidationError
    from backend.services.analytics_routes import FrontierGridRequest
    with pytest.raises(ValidationError):
        FrontierGridRequest(**payload)


def test_grid_reuses_readonly_noncontiguous_return_views():
    original = _two_assets().to_numpy()
    owner = np.zeros((original.shape[0] * 2, 4))
    owner[::2, ::2] = original
    view = owner[::2, ::2]
    view.setflags(write=False)
    bounds = np.array([[0., 1.], [0., 1.]])
    groups = np.empty((0, 2), dtype=np.uint8)
    empty = np.empty(0)
    settings = np.array([252., 252., .94, .94, 60., .95])
    signatures = [tuple(k.nopython_signatures) for k in optimizer.OPTIMIZER_NUMBA_KERNELS]
    result = optimizer.solve_frontier_grid_kernel(view, bounds, groups, empty, empty, np.array([.5, .5]),
                                                 2, 1, settings, 200, 300, False, 0., 0.)
    assert np.shares_memory(view, owner)
    assert np.all(result[3] == 0)
    np.testing.assert_array_equal(view, original)
    assert signatures == [tuple(k.nopython_signatures) for k in optimizer.OPTIMIZER_NUMBA_KERNELS]

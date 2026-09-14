"""Offline reference/ABI tests for the opt-in M2 model boundary."""
import builtins
import copy
import os

import numpy as np
import pytest
from pydantic import ValidationError

from backend.strategic_allocation import cma_model_kernels as kernels
from backend.strategic_allocation.cma_model_contracts import CMA_MODEL_ADAPTER
from backend.strategic_allocation.cma_models import evaluate_cma_model, readonly_float64


@pytest.fixture(autouse=True, scope="module")
def warm_models():
    kernels.warm()


def bl_request():
    return {"method": "black_litterman", "asset_ids": ["equity", "bonds"],
            "as_of": "2026-09-01", "currency": "CNY", "source": "Explicit test risk assumptions",
            "covariance": [[0.04, 0.006], [0.006, 0.01]],
            "risk_covariance_basis": "input_covariance", "market_weights": {"bonds": 0.4, "equity": 0.6},
            "market_weight_source": "User supplied market snapshot", "delta": 2.5, "tau": 0.05,
            "risk_free_rate": 0.02, "views": [view()]}


def view(kind="absolute", **patch):
    return {"kind": kind, "asset_id": "equity", "relative_to": "bonds" if kind == "relative" else None,
            "annual_return": 0.08, "view_std": 0.025, "observed_on": "2026-08-30",
            "available_on": "2026-09-01", "source": "Explicit offline research view", **patch}


def mixture_request(shared=True):
    risk = [[0.04, 0.006], [0.006, 0.01]]
    return {"method": "scenario_mixture", "asset_ids": ["equity", "bonds"],
            "as_of": "2026-09-01", "currency": "CNY", "source": "Explicit scenario assumptions",
            "risk_mode": "shared" if shared else "scenario_specific",
            "shared_covariance": risk if shared else None,
            "scenarios": [
                {"id": "growth", "probability": 0.3, "annual_returns": {"equity": 0.2, "bonds": 0.02},
                 "source": "Growth test assumption", "covariance": None if shared else risk},
                {"id": "recession", "probability": 0.7, "annual_returns": {"bonds": 0.07, "equity": -0.1},
                 "source": "Recession test assumption", "covariance": None if shared else [[0.09, -0.005], [-0.005, 0.02]]}]}


def bl_reference(request):
    sigma = np.array(request["covariance"])
    pi = request["delta"] * sigma @ np.array([request["market_weights"][a] for a in request["asset_ids"]])
    t = request["tau"] * sigma
    p = np.zeros((len(request["views"]), len(pi)))
    for row, v in enumerate(request["views"]):
        p[row, request["asset_ids"].index(v["asset_id"])] = 1
        if v["kind"] == "relative":
            p[row, request["asset_ids"].index(v["relative_to"])] = -1
    if len(p) == 0:
        return pi + request["risk_free_rate"], t
    omega = np.diag([v["view_std"] ** 2 for v in request["views"]])
    q = np.array([v["annual_return"] for v in request["views"]]) - request["risk_free_rate"] * p.sum(axis=1)
    system = p @ t @ p.T + omega
    return (pi + t @ p.T @ np.linalg.solve(system, q - p @ pi) + request["risk_free_rate"],
            t - t @ p.T @ np.linalg.solve(system, p @ t))


@pytest.mark.parametrize("views", [[], [view()], [view("relative")], [view(), view("relative")],
                                   [view(view_std=1e6)], [view(), view()]])
def test_bl_numpy_reference_and_separate_risk(views):
    request = bl_request()
    request["views"] = views
    original = copy.deepcopy(request)
    result = evaluate_cma_model(request)
    means, posterior = bl_reference(request)
    np.testing.assert_allclose(result.effective_returns, means, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(result.posterior_mean_covariance, posterior, rtol=1e-11, atol=1e-15)
    np.testing.assert_array_equal(result.effective_covariance, request["covariance"])
    assert request == original
    assert not result.effective_returns.flags.writeable and not result.effective_covariance.flags.writeable
    assert not result.posterior_mean_covariance.flags.writeable
    assert result.execution["complete"] and result.execution["python_fallback"] == 0
    assert result.to_payload()["definition"]["market_weight_source"] == request["market_weight_source"]


def test_weak_views_converge_to_prior_and_relative_rf_is_not_subtracted():
    request = bl_request()
    request["views"] = [view(view_std=1e6)]
    weak = evaluate_cma_model(request)
    request["views"] = []
    prior = evaluate_cma_model(request)
    np.testing.assert_allclose(weak.effective_returns, prior.effective_returns, atol=1e-14)
    request["views"] = [view("relative")]
    first = evaluate_cma_model(request)
    request["risk_free_rate"] += 0.03
    shifted = evaluate_cma_model(request)
    np.testing.assert_allclose(shifted.effective_returns - first.effective_returns, 0.03, atol=1e-15)
    np.testing.assert_allclose(shifted.posterior_mean_covariance, first.posterior_mean_covariance)


def test_singular_psd_covariance_supported_without_inverse_or_jitter():
    request = bl_request()
    request["covariance"] = [[0.04, 0.02], [0.02, 0.01]]
    request["views"] = [view(), view("relative")]
    result = evaluate_cma_model(request)
    means, posterior = bl_reference(request)
    np.testing.assert_allclose(result.effective_returns, means, atol=1e-14)
    np.testing.assert_allclose(result.posterior_mean_covariance, posterior, atol=1e-14)
    assert not result.model_audit["covariance_repaired"]


@pytest.mark.parametrize("shared", [True, False])
def test_mixture_total_covariance_numpy_reference(shared):
    request = mixture_request(shared)
    result = evaluate_cma_model(request)
    probabilities = np.array([s["probability"] for s in request["scenarios"]])
    means = np.array([[s["annual_returns"][a] for a in request["asset_ids"]] for s in request["scenarios"]])
    mean = probabilities @ means
    expected = np.zeros((2, 2))
    for s, p, mu in zip(request["scenarios"], probabilities, means):
        sigma = request["shared_covariance"] if shared else s["covariance"]
        expected += p * (np.array(sigma) + np.outer(mu - mean, mu - mean))
    np.testing.assert_allclose(result.effective_returns, mean, atol=1e-15)
    np.testing.assert_allclose(result.effective_covariance, expected, atol=1e-15)
    assert result.model_audit["between_mean_covariance"][0][0] > 0
    assert result.posterior_mean_covariance is None


def test_single_scenario_and_zero_probability_are_explicit():
    request = mixture_request()
    request["scenarios"] = request["scenarios"][:1]
    request["scenarios"][0]["probability"] = 1.
    result = evaluate_cma_model(request)
    np.testing.assert_array_equal(result.effective_covariance, request["shared_covariance"])
    np.testing.assert_array_equal(result.model_audit["between_mean_covariance"], np.zeros((2, 2)))
    request = mixture_request()
    request["scenarios"][0]["probability"], request["scenarios"][1]["probability"] = 0., 1.
    np.testing.assert_allclose(evaluate_cma_model(request).effective_returns, [-0.1, 0.07])
    del request["scenarios"][0]["annual_returns"]["bonds"]
    with pytest.raises(ValidationError, match="CMA_SCENARIO_AXIS"):
        evaluate_cma_model(request)


@pytest.mark.parametrize("patch", [
    {"delta": 0}, {"delta": True}, {"delta": "2.5"}, {"delta": float("nan")}, {"tau": 0}, {"tau": -0.1},
    {"risk_free_rate": float("inf")}, {"asset_ids": []}, {"asset_ids": ["equity", "equity"]},
    {"market_weights": {"equity": 1}}, {"market_weights": {"equity": -0.1, "bonds": 1.1}},
    {"market_weights": {"equity": 0.4, "bonds": 0.4}}, {"market_weight_source": ""},
    {"risk_covariance_basis": "posterior_mean_covariance"}, {"covariance": [[0.1]]},
    {"covariance": [[True, 0], [0, 0.01]]}, {"covariance": [[float("nan"), 0], [0, 0.01]]},
    {"views": [view(view_std=0)]}, {"views": [view(view_std=-0.1)]}, {"views": [view(view_std=float("inf"))]},
    {"views": [view(asset_id="missing")]}, {"views": [view("relative", relative_to="equity")]},
    {"views": [view(available_on="2026-09-02")]}, {"views": [view(observed_on="2026-09-02")]},
    {"views": [view(annual_return=-0.6)]}, {"views": [view("relative", annual_return=2.6)]},
])
def test_bl_contract_invalid_inputs(patch):
    with pytest.raises(ValidationError):
        evaluate_cma_model({**bl_request(), **patch})


@pytest.mark.parametrize("covariance", [
    [[0., 0.], [0., 0.01]], [[-0.1, 0.], [0., 0.01]], [[10., 0.], [0., 0.01]],
    [[0.04, 0.01], [0.02, 0.01]], [[0.04, 0.03], [0.03, 0.01]],
    [[1., -0.6, -0.6], [-0.6, 1., -0.6], [-0.6, -0.6, 1.]],
])
def test_invalid_covariance_rejected_without_repair(covariance):
    with pytest.raises(ValueError):
        kernels.covariance_diagnostics_kernel(readonly_float64(covariance, 2))


@pytest.mark.parametrize("patch", [
    {"probability": -0.1}, {"probability": 0.5}, {"probability": float("nan")}, {"probability": True},
    {"annual_returns": {"equity": 0.1}}, {"annual_returns": {"equity": 0.1, "bonds": 2.01}},
    {"annual_returns": {"equity": 0.1, "bonds": float("inf")}},
    {"covariance": [[0.04, 0.0], [0.0, 0.01]]},
])
def test_scenario_contract_invalid_inputs(patch):
    request = mixture_request()
    request["scenarios"][0].update(patch)
    with pytest.raises(ValidationError):
        evaluate_cma_model(request)


def test_missing_or_mixed_risk_and_duplicate_scenarios_rejected():
    for field, value in [("shared_covariance", None), ("risk_mode", "scenario_specific"), ("scenarios", [])]:
        with pytest.raises(ValidationError):
            evaluate_cma_model({**mixture_request(), field: value})
    request = mixture_request(False)
    request["scenarios"][0]["covariance"] = None
    with pytest.raises(ValidationError):
        evaluate_cma_model(request)
    request = mixture_request()
    request["scenarios"][0]["id"] = request["scenarios"][1]["id"]
    with pytest.raises(ValidationError):
        evaluate_cma_model(request)


def test_extreme_std_and_effective_output_ranges_fail_closed():
    for std in (1e-200, 1e200):
        with pytest.raises(ValueError, match="CMA_BL_VIEW_STD"):
            evaluate_cma_model({**bl_request(), "views": [view(view_std=std)]})
    with pytest.raises(ValueError, match="CMA_MODEL_RETURN_RANGE"):
        evaluate_cma_model({**bl_request(), "delta": 1000., "views": []})


def test_positive_tau_above_one_and_large_delta_with_small_risk_are_valid():
    request = {**bl_request(), "tau": 2., "delta": 2000.,
               "covariance": [[0.00004, 0.000006], [0.000006, 0.00001]]}
    result = evaluate_cma_model(request)
    expected, posterior = bl_reference(request)
    np.testing.assert_allclose(result.effective_returns, expected, atol=1e-14)
    np.testing.assert_allclose(result.posterior_mean_covariance, posterior, atol=1e-14)


@pytest.mark.parametrize("guard", [{"asset_ids": ["bonds", "equity"]}, {"as_of": "2026-08-31"}, {"currency": "USD"}])
def test_enclosing_source_context_guards(guard):
    with pytest.raises(ValueError, match="CMA_MODEL_CONTEXT"):
        evaluate_cma_model(bl_request(), **guard)


def test_model_instances_are_revalidated_after_nested_mutation():
    model = CMA_MODEL_ADAPTER.validate_python(bl_request())
    model.market_weights["equity"] = -1
    with pytest.raises(ValidationError):
        evaluate_cma_model(model)


def test_readiness_pid_compile_lock_and_failed_warmup(monkeypatch):
    for marker in (None, os.getpid() + 1):
        monkeypatch.setattr(kernels, "_WARMED_PID", marker)
        assert not kernels.execution_audit()["complete"]
        with pytest.raises(RuntimeError, match="CMA_MODEL_NOT_READY"):
            evaluate_cma_model(bl_request())
    monkeypatch.setattr(kernels, "_WARMED_PID", os.getpid())
    dispatcher = kernels.black_litterman_kernel
    monkeypatch.setattr(dispatcher, "_can_compile", True)
    with pytest.raises(RuntimeError, match="CMA_MODEL_NOT_READY"):
        evaluate_cma_model(bl_request())
    monkeypatch.setattr(dispatcher, "_can_compile", False)
    def fail(*args):
        raise RuntimeError("controlled warmup failure")
    monkeypatch.setattr(kernels, "black_litterman_kernel", fail)
    with pytest.raises(RuntimeError, match="controlled warmup failure"):
        kernels.warm()
    assert kernels._WARMED_PID is None
    monkeypatch.undo()
    kernels.warm()


def test_readonly_stride_alias_lifetime_determinism_and_no_new_signatures():
    owner = np.array([[0.04, 99., 0.006, 99.], [99., 99., 99., 99.],
                      [0.006, 99., 0.01, 99.], [99., 99., 99., 99.]])
    covariance = readonly_float64(owner[::2, ::2], 2)
    weights = readonly_float64(np.array([0.4, 0.6])[::-1], 1)
    picks = readonly_float64(np.array([[-1., 1.]])[:, ::-1], 2)
    q, std = readonly_float64([0.02], 1), readonly_float64([0.025], 1)
    assert np.shares_memory(owner, covariance) and owner.flags.writeable
    assert not covariance.flags.c_contiguous and weights.strides[0] < 0
    saved = owner.copy()
    signatures = kernels.execution_audit()["kernel_signatures"]
    first = kernels.black_litterman_kernel(covariance, weights, picks, q, std, 2.5, 0.05, 0.02)
    for _ in range(3):
        current = kernels.black_litterman_kernel(covariance, weights, picks, q, std, 2.5, 0.05, 0.02)
        for a, b in zip(current, first):
            np.testing.assert_array_equal(a, b)
            assert not np.shares_memory(a, owner)
    np.testing.assert_array_equal(owner, saved)
    del owner
    np.testing.assert_array_equal(covariance, [[0.04, 0.006], [0.006, 0.01]])
    means = readonly_float64(np.array([[0.02, 0.1], [0.03, -0.1]])[:, ::-1], 2)
    risks = covariance[None, :, :]
    assert np.shares_memory(risks, covariance)
    kernels.scenario_mixture_kernel(readonly_float64([0.4, 0.6], 1), means, risks, True)
    assert signatures == kernels.execution_audit()["kernel_signatures"]
    assert all(not k._can_compile and len(k.nopython_signatures) == 1 for k in kernels.KERNELS)


def test_direct_kernel_empty_axis_nan_and_bad_picks_are_checked():
    with pytest.raises(ValueError, match="CMA_BL_AXIS"):
        kernels.black_litterman_kernel(np.empty((0, 0)), np.empty(0), np.empty((0, 0)), np.empty(0), np.empty(0), 2., .05, .02)
    for picks in ([[0., 0.]], [[1., 1.]], [[float("nan"), 0.]]):
        with pytest.raises(ValueError, match="CMA_BL_VIEW_PICK"):
            kernels.black_litterman_kernel(np.eye(2) * .01, np.array([.5, .5]), np.array(picks), np.array([.05]), np.array([.1]), 2., .05, .02)
    with pytest.raises(ValueError, match="CMA_SCENARIO_PROBABILITY"):
        kernels.scenario_mixture_kernel(np.array([np.nan]), np.array([[.1, .2]]), np.eye(2)[None] * .01, True)


def test_preview_has_no_filesystem_side_effects(monkeypatch, tmp_path):
    requests = [bl_request(), mixture_request()]
    def forbidden(*args, **kwargs):
        raise AssertionError("Pure CMA preview attempted filesystem I/O")
    with monkeypatch.context() as context:
        context.setattr(builtins, "open", forbidden)
        context.setattr(os, "mkdir", forbidden)
        for request in requests:
            evaluate_cma_model(request).to_payload()
    assert list(tmp_path.iterdir()) == []

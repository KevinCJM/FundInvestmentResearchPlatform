"""Independent numerical and contract checks for the 2026-09-20 CMA review."""
import copy

import numpy as np
import pytest
from pydantic import ValidationError as ContractError

from backend.strategic_allocation import cma_model_kernels as bl
from backend.strategic_allocation.cma_models import evaluate_cma_model
from backend.tests.test_cma_models import bl_request, view


@pytest.fixture(scope="module", autouse=True)
def ready():
    bl.warm()


def basket_request(basis="relative"):
    request = bl_request()
    request.update(asset_ids=["equity", "bonds", "gold"],
                   covariance=[[.04, .006, .001], [.006, .01, 0.], [.001, 0., .0225]],
                   market_weights={"equity": .5, "bonds": .3, "gold": .2})
    request["views"] = [{k: v for k, v in view().items() if k not in ("asset_id", "relative_to")}]
    request["views"][0].update(kind="basket", basis=basis,
        legs=[{"asset_id": "equity", "coefficient": 1. if basis == "relative" else .5},
              {"asset_id": "bonds", "coefficient": -.5 if basis == "relative" else .25},
              {"asset_id": "gold", "coefficient": -.5 if basis == "relative" else .25}])
    return request


@pytest.mark.parametrize("basis", ["absolute", "relative"])
def test_basket_matches_general_linear_gaussian_reference(basis):
    request = basket_request(basis)
    before = copy.deepcopy(request)
    result = evaluate_cma_model(request)
    sigma = np.array(request["covariance"])
    weights = np.array([request["market_weights"][a] for a in request["asset_ids"]])
    picks = np.array([[leg["coefficient"] for leg in request["views"][0]["legs"]]])
    prior = request["delta"] * sigma @ weights
    prior_cov = request["tau"] * sigma
    omega = np.array([[request["views"][0]["view_std"]**2]])
    residual = np.array([request["views"][0]["annual_return"]]) - request["risk_free_rate"]*picks.sum(1) - picks @ prior
    gain = np.linalg.solve(picks @ prior_cov @ picks.T + omega, picks @ prior_cov).T
    np.testing.assert_allclose(result.effective_returns, prior + request["risk_free_rate"] + gain @ residual, atol=1e-15)
    np.testing.assert_allclose(result.posterior_mean_covariance, prior_cov - gain @ picks @ prior_cov, atol=1e-15)
    audit = result.model_audit
    np.testing.assert_allclose(audit["view_return_contributions"], gain.T*residual[:, None], atol=1e-15)
    np.testing.assert_allclose(audit["posterior_shift"], gain @ residual, atol=1e-15)
    assert audit["implied_market_sharpe"] == pytest.approx(request["delta"]*np.sqrt(weights @ sigma @ weights))
    np.testing.assert_allclose(audit["view_precision_ratio"], np.diag(picks @ prior_cov @ picks.T)/np.diag(omega))
    assert request == before


def test_bl_diagnostics_without_views_and_equivalent_legacy_basket():
    request = bl_request()
    original = evaluate_cma_model(request)
    request["views"] = [{k: v for k, v in request["views"][0].items() if k not in ("asset_id", "relative_to")}]
    request["views"][0].update(kind="basket", basis="absolute", legs=[{"asset_id":"equity", "coefficient":1.}])
    changed = evaluate_cma_model(request)
    np.testing.assert_array_equal(original.effective_returns, changed.effective_returns)
    np.testing.assert_array_equal(original.posterior_mean_covariance, changed.posterior_mean_covariance)
    no_views = evaluate_cma_model({**request, "views": []})
    assert no_views.model_audit["view_precision_ratio"] == []
    assert no_views.model_audit["view_return_contributions"] == []
    assert no_views.model_audit["posterior_shift"] == [0., 0.]


@pytest.mark.parametrize("change", [
    {"legs": []},
    {"legs": [{"asset_id":"equity", "coefficient":0.}]},
    {"legs": [{"asset_id":"equity", "coefficient":1.}, {"asset_id":"equity", "coefficient":-1.}]},
    {"legs": [{"asset_id":"unknown", "coefficient":1.}, {"asset_id":"bonds", "coefficient":-1.}]},
    {"legs": [{"asset_id":"equity", "coefficient":3.}, {"asset_id":"bonds", "coefficient":-3.}]},
    {"legs": [{"asset_id":"equity", "coefficient":float("nan")}]},
    {"basis":"absolute"}, {"available_on":"2026-09-02"}, {"observed_on":"2026-09-02"},
])
def test_basket_invalid_inputs_fail_closed(change):
    request = basket_request()
    request["views"][0].update(change)
    with pytest.raises(ContractError):
        evaluate_cma_model(request)

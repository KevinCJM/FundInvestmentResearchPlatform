"""Ellipsoid calibration, compatibility and actual frozen-policy consumption."""
import copy

import numpy as np
import pytest
from scipy.stats import chi2, f
from pydantic import ValidationError as ContractError

from backend.custom_indicators.errors import ValidationError
from backend.strategic_allocation import kernels, cma_model_kernels, cma_statistical_kernels, reference_evidence_kernels
from backend.strategic_allocation import uncertainty_kernels as numeric
from backend.strategic_allocation.contracts import PolicyRequest, PublishPolicyRequest, CmaRequest
from backend.strategic_allocation.cma_application import frozen_mean_covariance, frozen_policy_assumptions
from backend.strategic_allocation.multi_cma import request_payload
from backend.tests.test_strategic_allocation import workspace, warm, saved_inputs, definition
from backend.tests.test_ltcma_statistics import request, publish


@pytest.fixture(scope="module", autouse=True)
def warm_uncertainty():
    cma_model_kernels.warm()
    reference_evidence_kernels.warm()
    cma_statistical_kernels.warm()
    numeric.warm()


@pytest.mark.parametrize("confidence", [.68, .90, .95])
@pytest.mark.parametrize("dimension", [1, 2, 3, 10, 29, 30])
def test_gaussian_radius_against_scipy(confidence, dimension):
    assert numeric.uncertainty_radius_kernel(confidence, dimension, 0, 0.)**2 == pytest.approx(
        chi2.ppf(confidence, dimension), rel=1e-11, abs=1e-12)


@pytest.mark.parametrize("confidence", [.68, .90, .95])
@pytest.mark.parametrize("degrees", [2.1, 7., 31.5, 1000., 100000.])
@pytest.mark.parametrize("dimension", [1, 3, 30])
def test_niw_t_radius_uses_covariance_not_scale(confidence, degrees, dimension):
    expected = dimension*(degrees-2)/degrees*f.ppf(confidence, dimension, degrees)
    assert numeric.uncertainty_radius_kernel(confidence, dimension, 1, degrees)**2 == pytest.approx(expected, rel=5e-8)


@pytest.mark.parametrize("n,dimension", [(20., 1), (20., 19), (500., 30)])
def test_hotelling_not_gaussian_in_small_samples(n, dimension):
    expected = dimension*(n-1)/(n-dimension)*f.ppf(.95, dimension, n-dimension)
    assert numeric.uncertainty_radius_kernel(.95, dimension, 2, n)**2 == pytest.approx(expected, rel=1e-9)
    with pytest.raises(ValueError):
        numeric.uncertainty_radius_kernel(.95, dimension, 2, float(dimension))


def test_correlated_ellipsoid_and_box_remain_distinct():
    weights = np.array([.5,.5]); means = np.array([.08,.06]); risk = np.diag([.04,.01])
    owner = np.array([[.001,99.,-.0005,99.],[99.,99.,99.,99.],[-.0005,99.,.002,99.],[99.,99.,99.,99.]])
    covariance = owner[::2,::2]; covariance.flags.writeable=False
    assert np.shares_memory(covariance,owner)
    before=owner.copy()
    value,_=kernels.portfolio_moments_ellipsoidal_kernel(weights,means,risk,covariance,5.,2.)
    assert value[2] == pytest.approx(weights @ means - 2*np.sqrt(weights @ covariance @ weights))
    assert value[1] == pytest.approx(np.sqrt(weights @ risk @ weights))
    box,_=kernels.portfolio_moments_kernel(weights,means,risk,2*np.sqrt(np.diag(covariance)),5.,1.)
    assert value[2] != pytest.approx(box[2])
    np.testing.assert_array_equal(owner,before)
    zeros=np.zeros((2,2))
    assert kernels.portfolio_moments_ellipsoidal_kernel(weights,means,risk,zeros,5.,0.)[0][2] == pytest.approx(weights @ means)
    for bad in [np.array([[1.,2.],[2.,1.]]), np.array([[0.,.1],[.1,1.]]),np.full((2,2),np.nan)]:
        with pytest.raises(ValueError):
            kernels.portfolio_moments_ellipsoidal_kernel(weights,means,risk,bad,5.,2.)


def test_old_box_request_echo_has_no_new_fields():
    body=PolicyRequest(mandate_id="mandate",cma_id="cma")
    assert not set(request_payload(body)).intersection({"uncertainty_set","uncertainty_confidence","uncertainty_approximation_acknowledged"})
    for patch in ({"uncertainty_confidence":"95"}, {"uncertainty_set":"ellipsoidal"},
                  {"uncertainty_set":"ellipsoidal","uncertainty_confidence":"95","uncertainty_penalty":2.},
                  {"mode":"parameter_average","cma_id":None,"cma_refs":[{"cma_id":"a","content_hash":"a"*64,"weight":1.}],"uncertainty_set":"ellipsoidal","uncertainty_confidence":"95"}):
        with pytest.raises(ContractError):
            PolicyRequest.model_validate({**body.model_dump(),**patch})


def test_historical_freeze_reuse_publish_and_missing_covariance(workspace,monkeypatch):
    service,_=workspace
    mandate,old,old_request=saved_inputs(service)
    historical=publish(service,request(shrinkage=0.),"ellipse-historical")
    matrix,key=frozen_mean_covariance(historical,service.artifacts)
    assert key=="mean_estimation_covariance" and isinstance(matrix,np.memmap) and not matrix.flags.writeable
    body=PolicyRequest(mandate_id=mandate["id"],cma_id=historical["id"],candidate_count=300,
        uncertainty_set="ellipsoidal",uncertainty_confidence="95")
    monkeypatch.setattr(service.cma,"calculation",lambda *_:pytest.fail("Must not refit a frozen CMA"))
    result=service.preview_policy(body)
    assert result["uncertainty_model"]["calibration"]=="iid_normal_hotelling_sample_mean"
    for candidate in result["candidates"]:
        weights=np.array(list(candidate["weights"].values()))
        expected=candidate["metrics"]["expected_return"]-result["uncertainty_model"]["kappa"]*np.sqrt(weights @ matrix @ weights)
        assert candidate["metrics"]["conservative_return"]==pytest.approx(expected)
    saved=service.publish_policy(PublishPolicyRequest(request=body,preview_hash=result["preview_hash"],
        candidate_id="robust-utility",name="椭球策略",reason="独立核验均值误差的研究范围"))
    assert frozen_policy_assumptions(saved["policy"])==result["assumptions"]
    for remove in (True,False):
        tampered=copy.deepcopy(saved["policy"])
        if remove:tampered.pop("uncertainty_model")
        else:tampered["uncertainty_model"]["kappa"]+=1
        with pytest.raises(ValidationError,match="冻结"):
            frozen_policy_assumptions(tampered)
    before=copy.deepcopy(old)
    assert service.get_cma(old["id"])==before
    with pytest.raises(ValidationError,match="均值协方差"):
        service.preview_policy(body.model_copy(update={"cma_id":old["id"]}))


@pytest.mark.parametrize("covariance,dimension", [
    (np.diag([0., .002]), 1),
    (np.array([[.001, .001], [.001, .001]]), 2),
    (np.diag([.002, 1e-20]), 2),
    (np.zeros((2, 2)), 0),
])
def test_uncertainty_dimension_counts_coordinates_not_numerical_rank(covariance, dimension):
    errors, actual, _ = kernels.mean_covariance_diagnostics_kernel(covariance)
    assert actual == dimension
    np.testing.assert_array_equal(errors, np.sqrt(np.diag(covariance)))


@pytest.mark.parametrize("covariance", [
    np.empty((0, 0)), np.empty((2, 1)), np.diag([.01, np.inf]),
    np.array([[.01, .002], [.001, .01]]), np.diag([.01, -.001]),
])
def test_mean_covariance_invalid_axes_and_values(covariance):
    with pytest.raises(ValueError):
        kernels.mean_covariance_diagnostics_kernel(covariance)


def test_singular_bl_uses_explicit_conservative_coordinate_radius(workspace):
    service, _ = workspace
    mandate, _, _ = saved_inputs(service)
    raw = request().model_dump(mode="json")
    raw["model"] = {
        "method": "black_litterman", "asset_ids": ["股票", "债券"],
        "as_of": raw["as_of"], "currency": "CNY", "source": "奇异协方差边界的固定测试",
        "covariance": [[.04, .02], [.02, .01]], "risk_covariance_basis": "input_covariance",
        "market_weights": {"股票": .6, "债券": .4}, "market_weight_source": "固定边界测试权重",
        "delta": 2., "tau": .05, "risk_free_rate": .02, "views": [],
    }
    cma = publish(service, CmaRequest.model_validate(raw), "ellipse-singular-bl")
    body = PolicyRequest(mandate_id=mandate["id"], cma_id=cma["id"], candidate_count=300,
                         uncertainty_set="ellipsoidal", uncertainty_confidence="95")
    result = service.preview_policy(body)
    evidence = result["uncertainty_model"]
    assert evidence["dimension"] == 2
    assert evidence["dimension_basis"] == "nonzero_variance_coordinates"
    assert evidence["kappa"] ** 2 == pytest.approx(chi2.ppf(.95, 2))
    assert any("保守覆盖" in text for text in evidence["warnings"])
    assert np.linalg.matrix_rank(evidence["mean_covariance"]) == 1


def test_niw_policy_uses_frozen_t_covariance_and_information(workspace, monkeypatch):
    service, _ = workspace
    mandate, _, _ = saved_inputs(service)
    raw = definition().model_dump(mode="json")
    raw.update(schema_version="2.0", moment_semantics="annualized_periodic_arithmetic",
               fee_basis="explicit_assumption", fx_hedging_basis="explicit_assumption")
    prior = publish(service, CmaRequest.model_validate(raw), "ellipse-niw-prior")
    cma = publish(service, request("bayesian_niw",
        prior_ref={"id": prior["id"], "content_hash": prior["content_hash"]},
        mean_prior_observations=20., covariance_prior_observations=30.), "ellipse-niw")
    mapped = []
    arrays = service.artifacts.arrays

    def capture(identifier, names):
        result = arrays(identifier, names)
        if "posterior_mean_covariance" in names:
            mapped.append(result["posterior_mean_covariance"])
        return result

    monkeypatch.setattr(service.artifacts, "arrays", capture)
    monkeypatch.setattr(service.cma, "calculation", lambda *_: pytest.fail("Frozen NIW must not refit"))
    body = PolicyRequest(mandate_id=mandate["id"], cma_id=cma["id"], candidate_count=300,
                         uncertainty_set="ellipsoidal", uncertainty_confidence="90")
    from backend.strategic_allocation.uncertainty import resolve_uncertainty
    matrix, evidence = resolve_uncertainty(body, cma, service.artifacts)
    assert np.shares_memory(matrix, mapped[-1]) and not matrix.flags.writeable
    assert evidence["calibration"] == "conditional_niw_multivariate_t_covariance"
    dimension = len(cma["model_result"]["asset_ids"])
    degrees = cma["model_result"]["model_audit"]["niw_posterior"]["nu"] - dimension + 1
    expected = dimension * (degrees - 2) / degrees * f.ppf(.9, dimension, degrees)
    assert evidence["kappa"] ** 2 == pytest.approx(expected, rel=1e-9)
    result = service.preview_policy(body)
    assert result["uncertainty_model"] == evidence
    for candidate in result["candidates"]:
        weights = np.array(list(candidate["weights"].values()))
        haircut = evidence["kappa"] * np.sqrt(weights @ matrix @ weights)
        assert candidate["metrics"]["conservative_return"] == pytest.approx(
            candidate["metrics"]["expected_return"] - haircut)


def test_shrunk_history_requires_explicit_approximation(workspace):
    service,_=workspace
    mandate,_,_=saved_inputs(service)
    historical=publish(service,request(shrinkage=.1),"ellipse-shrunk-history")
    body=PolicyRequest(mandate_id=mandate["id"],cma_id=historical["id"],candidate_count=300,
        uncertainty_set="ellipsoidal",uncertainty_confidence="90")
    with pytest.raises(ValidationError,match="插件近似"):
        service.preview_policy(body)
    result=service.preview_policy(body.model_copy(update={"uncertainty_approximation_acknowledged":True}))
    assert result["uncertainty_model"]["calibration"]=="gaussian_plugin_not_exact_confidence"


def test_quantile_readiness_and_no_signature_growth(monkeypatch):
    original=numeric.execution_audit()["kernel_signatures"]
    for dimension in (1,30):numeric.uncertainty_radius_kernel(.9,dimension,1,100.)
    assert numeric.execution_audit()["kernel_signatures"]==original
    assert numeric.execution_audit()["request_time_compilation"]==0
    monkeypatch.setattr(numeric,"_WARMED_PID",None)
    with pytest.raises(RuntimeError,match="预热"):
        numeric.require_ready()

"""Research confirmation is deliberately separate from policy admission."""
from datetime import date, timedelta

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.custom_indicators.errors import ValidationError
from backend.strategic_allocation.cma_application import frozen_mean_covariance, frozen_numeric_inputs
from backend.strategic_allocation.cma_selection import (
    downstream_eligibility, require_downstream_eligible,
)
from backend.strategic_allocation.contracts import MandateRequest, MandateStudyRequest, PolicyRequest
from backend.strategic_allocation.routes import build_router
from backend.tests.test_ltcma_scenario_evidence import scenario, payload
from backend.tests.test_ltcma_statistics import publish, request as statistical_request
from backend.tests.test_strategic_allocation import workspace, warm, saved_inputs


def artifact(method, **validation):
    return {"definition": {"model": {"method": method}},
            "model_result": {"model_audit": {"model_validation": validation}}}


def test_conditional_cannot_self_authorize_with_a_metadata_flag():
    item = artifact("conditional_scenario", downstream_eligible=True)
    assert downstream_eligibility(item)["downstream_eligible"] is False
    with pytest.raises(ValidationError, match="条件情景"):
        require_downstream_eligible(item)


def test_long_term_requires_explicit_computation_admission():
    with pytest.raises(ValidationError, match="统计计算依据"):
        require_downstream_eligible(artifact("long_term_scenario"))
    require_downstream_eligible(artifact("long_term_scenario", downstream_eligible=True))


@pytest.mark.parametrize("method", ["historical_statistics", "historical_regime_occupancy", "manual", "bayesian_niw"])
def test_existing_model_contracts_keep_their_admission(method):
    require_downstream_eligible(artifact(method))


@pytest.fixture
def saved_scenarios(scenario):
    service, *_ = scenario
    mandate, _, _ = saved_inputs(service)
    longterm = publish(service, payload(scenario), "selection-longterm")
    conditional = publish(service, payload(scenario, "conditional_scenario"), "selection-conditional")
    app = FastAPI()
    app.include_router(build_router(service))
    return service, TestClient(app), mandate, longterm, conditional


@pytest.mark.parametrize("mode", ["single", "parameter_average", "compatible_all_models"])
def test_saved_conditional_is_rejected_through_actual_policy_endpoints(saved_scenarios, mode):
    service, client, mandate, longterm, conditional = saved_scenarios
    body = {"mandate_id": mandate["id"], "candidate_count": 300, "mode": mode}
    if mode == "single":
        body["cma_id"] = conditional["id"]
    else:
        # A valid first source must not hide the disallowed second source.
        body["cma_refs"] = [{"cma_id": item["id"], "content_hash": item["content_hash"],
                             **({"weight": .5} if mode == "parameter_average" else {})}
                            for item in (longterm, conditional)]
    response = client.post("/api/strategic-allocation/policy/preview", json=body)
    assert response.status_code == 422, response.text
    assert response.json()["detail"]["code"] == "LTCMA_DOWNSTREAM_UNAVAILABLE"
    assert "条件情景" in response.json()["detail"]["message"]
    direct_publish = client.post("/api/strategic-allocation/policies", json={
        "request": body, "preview_hash": "0" * 64, "candidate_id": "nominal-utility",
        "name": "不能绕过的条件研究", "reason": "直接确认也须重新校验来源资格"})
    assert direct_publish.status_code == 422, direct_publish.text
    assert direct_publish.json()["detail"]["code"] == "LTCMA_DOWNSTREAM_UNAVAILABLE"
    assert not service.baselines.list_baselines()
    # A rejected new use leaves the saved research accessible and unchanged.
    assert service.get_cma(conditional["id"]) == conditional


def test_saved_conditional_is_rejected_as_niw_prior_through_cma_endpoint(saved_scenarios):
    service, client, _, _, conditional = saved_scenarios
    body = statistical_request("bayesian_niw",
        prior_ref={"id": conditional["id"], "content_hash": conditional["content_hash"]},
        mean_prior_observations=20., covariance_prior_observations=30.,
        data_reuse_acknowledged=True)
    response = client.post("/api/strategic-allocation/cma/preview", json=body.model_dump(mode="json"))
    assert response.status_code == 422, response.text
    assert response.json()["detail"]["code"] == "LTCMA_DOWNSTREAM_UNAVAILABLE"
    assert "贝叶斯先验" in response.json()["detail"]["message"]
    assert service.get_cma(conditional["id"]) == conditional


def test_saved_conditional_cannot_bypass_admission_via_mandate_diagnosis(saved_scenarios):
    _, client, _, _, conditional = saved_scenarios
    body = MandateStudyRequest(cma_id=conditional["id"], definition=MandateRequest(
        name="条件情景的目标诊断", as_of=date.today(), review_date=date.today() + timedelta(days=90),
        target_return=0., max_volatility=.2, boundary_reason="明确确认的测试风险和流动性边界"))
    response = client.post("/api/strategic-allocation/mandates/preview", json=body.model_dump(mode="json"))
    assert response.status_code == 422, response.text
    assert response.json()["detail"]["code"] == "LTCMA_DOWNSTREAM_UNAVAILABLE"
    assert "条件情景" in response.json()["detail"]["message"]


def test_saved_longterm_uses_frozen_annual_moments_and_bootstrap_box_widths(saved_scenarios, monkeypatch):
    service, client, mandate, longterm, _ = saved_scenarios
    means, risk, widths = frozen_numeric_inputs(longterm, service.artifacts)
    mean_covariance, key = frozen_mean_covariance(longterm, service.artifacts)
    assert key == "mean_estimation_covariance"
    assert isinstance(mean_covariance, np.memmap) and not mean_covariance.flags.writeable
    np.testing.assert_array_equal(mean_covariance, longterm["model_result"]["mean_estimation_covariance"])
    np.testing.assert_array_equal(widths, longterm["model_result"]["mean_uncertainty"])
    assert np.any(widths > 0.)
    assert not np.array_equal(risk, mean_covariance)
    monkeypatch.setattr(service.cma, "calculation", lambda *_: pytest.fail("Saved CMA must not be refitted"))
    penalty = 1.7
    body = PolicyRequest(mandate_id=mandate["id"], cma_id=longterm["id"],
                         uncertainty_penalty=penalty, candidate_count=300)
    response = client.post("/api/strategic-allocation/policy/preview", json=body.model_dump(mode="json"))
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["assumptions"]["moment_semantics"] == "annualized_periodic_arithmetic"
    assert "uncertainty_model" not in result
    for candidate in result["candidates"]:
        weights = np.array([candidate["weights"][name] for name in longterm["model_result"]["asset_ids"]])
        assert candidate["metrics"]["expected_return"] == pytest.approx(weights @ means)
        assert candidate["metrics"]["volatility"] == pytest.approx(np.sqrt(weights @ risk @ weights))
        assert candidate["metrics"]["conservative_return"] == pytest.approx(weights @ means - penalty * (weights @ widths))
    confirmed = client.post("/api/strategic-allocation/policies", json={
        "request": body.model_dump(mode="json"), "preview_hash": result["preview_hash"],
        "candidate_id": "robust-utility", "name": "长期情景区间稳健配置", "reason": "依据冻结的年化矩和时间块重采样半宽"})
    assert confirmed.status_code == 201, confirmed.text
    policy = confirmed.json()["policy"]
    assert policy["model_result"]["content_hash"] == longterm["model_result"]["content_hash"]
    assert policy["selection_request"]["uncertainty_penalty"] == penalty
    assert service.get_cma(longterm["id"]) == longterm


@pytest.mark.parametrize("acknowledged", [False, True])
def test_longterm_joint_ellipsoid_cannot_be_enabled_by_approximation_consent(saved_scenarios, acknowledged):
    _, client, mandate, longterm, _ = saved_scenarios
    body = PolicyRequest(mandate_id=mandate["id"], cma_id=longterm["id"], candidate_count=300,
        uncertainty_set="ellipsoidal", uncertainty_confidence="95",
        uncertainty_approximation_acknowledged=acknowledged)
    response = client.post("/api/strategic-allocation/policy/preview", json=body.model_dump(mode="json"))
    assert response.status_code == 422, response.text
    assert response.json()["detail"]["code"] == "SAA_UNCERTAINTY_SET_UNAVAILABLE"
    assert "联合椭球半径" in response.json()["detail"]["message"]
    assert "区间稳健模式" in response.json()["detail"]["message"]

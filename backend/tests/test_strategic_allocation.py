"""Offline numeric, storage, API and policy-handoff acceptance."""
from datetime import date, timedelta
import copy

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError as InputError

from backend.custom_indicators.errors import ConflictError, ValidationError
from backend.strategic_allocation import kernels
from backend.strategic_allocation.contracts import (
    CmaRequest, MandateRequest, PolicyRequest, PublishCmaRequest,
    PublishPolicyRequest, RiskReferenceRequest,
)
from backend.strategic_allocation.policy_gate import check_policy, require_policy_application
from backend.strategic_allocation.routes import build_router
from backend.strategic_allocation.service import StrategicAllocationService
from backend.tactical_allocation.numeric import warm_tactical_allocation_kernels
from backend.tactical_allocation.contracts import PreviewRequest
from backend.tactical_allocation.service import TacticalAllocationService


@pytest.fixture(scope="module", autouse=True)
def warm():
    from backend.tactical_allocation.data import warm_tactical_data
    warm_tactical_data()
    warm_tactical_allocation_kernels()
    assert kernels.warm_strategic_kernels()["complete"]


@pytest.fixture
def workspace(tmp_path):
    days = pd.bdate_range(end=date.today(), periods=161)
    info = [{"asset_alloc_name": "股债", "asset_name": name, "etf_code": code,
             "etf_name": name + " ETF", "etf_weight": 100.0, "creat_time": days[0],
             "as_of": None, "universe_snapshot_id": None, "data_release_id": None}
            for name, code in [("股票", "510300.SH"), ("债券", "511010.SH")]]
    pd.DataFrame(info).to_parquet(tmp_path / "asset_alloc_info.parquet", index=False)
    returns = np.column_stack((0.0004 + 0.008 * np.sin(np.arange(160)),
                               0.0001 + 0.002 * np.cos(np.arange(160) * 0.7)))
    values = np.vstack((np.ones(2), np.cumprod(1 + returns, axis=0)))
    pd.DataFrame([{"asset_alloc_name": "股债", "asset_name": name, "date": day,
                   "nv": values[t, i], "available_at": day, "as_of": None}
                  for i, name in enumerate(("股票", "债券")) for t, day in enumerate(days)]).to_parquet(
                      tmp_path / "asset_nv.parquet", index=False)
    pd.DataFrame({"exchange": ["SSE"] * len(days),
                  "cal_date": [int(day.strftime("%Y%m%d")) for day in days],
                  "is_open": [1] * len(days)}).to_parquet(tmp_path / "trade_day_df.parquet", index=False)
    return StrategicAllocationService(tmp_path / "research", tmp_path), days


def definition():
    return CmaRequest(name="人民币十年假设", alloc_name="股债", as_of=date.today(),
        currency="CNY", horizon_years=10, source="研究员显式设定的长期总收益假设", basis_confirmed=True,
        assets=[{"id": "股票", "role": "growth", "liquidity": "liquid", "rationale": "广泛权益增长敞口",
                 "annual_return": 0.07, "annual_volatility": 0.18, "mean_uncertainty": 0.03},
                {"id": "债券", "role": "rates", "liquidity": "liquid", "rationale": "利率债防御敞口",
                 "annual_return": 0.025, "annual_volatility": 0.05, "mean_uncertainty": 0.005}],
        correlation=[[1., -0.1], [-0.1, 1.]])


def saved_inputs(service):
    mandate = service.save_mandate(MandateRequest(name="长期配置目标", as_of=date.today(),
        review_date=date.today() + timedelta(days=180), target_return=0.0, max_volatility=0.2,
        min_liquid_weight=0.2, max_tracking_error=0.04))
    request = definition()
    preview = service.preview_cma(request)
    cma = service.publish_cma(PublishCmaRequest(request=request, preview_hash=preview["preview_hash"]))
    policy_request = PolicyRequest(mandate_id=mandate["id"], cma_id=cma["id"], candidate_count=300)
    return mandate, cma, policy_request


def test_covariance_reference_and_readonly_strided_views():
    raw = np.random.default_rng(3).normal(0.0002, 0.01, (100, 6))
    view = raw[::2, ::2]
    raw.setflags(write=False)
    view.setflags(write=False)
    assert np.shares_memory(raw, view)
    before = raw.copy()
    covariance, vol, corr, mean = kernels.historical_risk_kernel(view, 0.2, 252)
    sample = np.cov(view, rowvar=False, ddof=1)
    expected = 252 * (0.8 * sample + 0.2 * np.diag(np.diag(sample)))
    np.testing.assert_allclose(covariance, expected, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(vol, np.sqrt(np.diag(expected)), rtol=1e-12)
    np.testing.assert_allclose(corr, expected / np.outer(vol, vol), rtol=1e-12)
    np.testing.assert_allclose(mean, view.mean(axis=0) * 252, rtol=1e-12)
    np.testing.assert_array_equal(raw, before)


@pytest.mark.parametrize("matrix,error", [
    ([[1., 0.1], [0.2, 1.]], "SYMMETRY"),
    ([[0.9, 0.], [0., 1.]], "DIAGONAL"),
    ([[1., 1.1], [1.1, 1.]], "RANGE"),
    ([[1., np.nan], [np.nan, 1.]], "RANGE"),
    ([[1., np.inf], [np.inf, 1.]], "RANGE"),
])
def test_invalid_covariance_is_not_repaired(matrix, error):
    with pytest.raises(ValueError, match=error):
        kernels.cma_covariance_kernel(np.array([0.1, 0.2]), np.array(matrix))


def test_non_psd_and_singular_psd_are_distinguished():
    with pytest.raises(ValueError, match="PSD"):
        kernels.cma_covariance_kernel(np.ones(3), np.array([[1., .9, .9], [.9, 1., -.9], [.9, -.9, 1.]]))
    covariance, eigenvalue = kernels.cma_covariance_kernel(np.array([0.1, 0.2]), np.ones((2, 2)))
    assert eigenvalue == pytest.approx(0.)
    np.testing.assert_allclose(covariance, [[.01, .02], [.02, .04]])


def test_moments_and_risk_contributions_reconcile():
    weights, means, uncertainty = np.array([.6, .4]), np.array([.07, .03]), np.array([.02, .005])
    cov = np.array([[.04, -.001], [-.001, .0036]])
    metrics, contributions = kernels.portfolio_moments_kernel(weights, means, cov, uncertainty, 4., 1.5)
    variance = weights @ cov @ weights
    np.testing.assert_allclose(metrics, [weights @ means, np.sqrt(variance), weights @ means - 1.5 * weights @ uncertainty,
        weights @ means - 2 * variance, weights @ means - 1.5 * weights @ uncertainty - 2 * variance])
    np.testing.assert_allclose(contributions, weights * (cov @ weights) / variance)
    assert contributions.sum() == pytest.approx(1.)


def test_expected_active_risk_matches_delta_covariance():
    covariance = np.array([[.04, -.001], [-.001, .0036]])
    base = np.array([.6, .4])
    target = np.array([.7, .3])
    delta = target - base
    expected = float(np.sqrt(delta @ covariance @ delta))
    assert kernels.expected_active_risk_kernel(target, base, covariance) == pytest.approx(expected)


def test_preview_no_writes_confirmation_required_and_history_immutable(workspace):
    service, _ = workspace
    preview = service.preview_cma(definition())
    assert not service.artifacts.root.exists()
    assert service.preview_cma(definition())["preview_hash"] == preview["preview_hash"]
    with pytest.raises(ConflictError):
        service.publish_cma(PublishCmaRequest(request=definition(), preview_hash="0" * 64))
    assert not service.artifacts.root.exists()
    first = service.publish_cma(PublishCmaRequest(request=definition(), preview_hash=preview["preview_hash"]))
    second = service.publish_cma(PublishCmaRequest(request=definition(), preview_hash=preview["preview_hash"]))
    assert first["id"] != second["id"]
    assert service.get_cma(first["id"]) == first
    frozen = service.artifacts.arrays(first["id"])["covariance"]
    assert not frozen.flags.writeable
    assert isinstance(frozen, np.memmap)


def test_policy_adoption_enters_existing_taa_store_and_preserves_budgets(workspace):
    service, _ = workspace
    mandate, cma, request = saved_inputs(service)
    before = service.baselines.list_baselines()
    preview = service.preview_policy(request)
    assert service.baselines.list_baselines() == before
    assert len(preview["candidates"]) == 4
    for candidate in preview["candidates"]:
        assert sum(candidate["weights"].values()) == pytest.approx(1)
        assert candidate["metrics"]["volatility"] <= mandate["definition"]["max_volatility"]
    policy = service.publish_policy(PublishPolicyRequest(request=request, preview_hash=preview["preview_hash"],
        candidate_id="robust-utility", name="长期政策一", reason="采用保守假设下的候选配置"))
    assert service.baselines.get_baseline(policy["id"]) == policy
    assert policy["policy"]["cma_hash"] == cma["content_hash"]
    assert policy["policy"]["independent_approval"] is False
    assert policy["group_limits"][0]["lo"] == 0.2
    assert policy["assets"][0]["products"][0]["product_id"] == "510300.SH"
    assert service.catalog()["policies"][0]["id"] == policy["id"]


def test_robustness_zero_penalty_and_candidate_determinism(workspace):
    service, _ = workspace
    _, _, request = saved_inputs(service)
    zero = request.model_copy(update={"uncertainty_penalty": 0.0})
    result = service.preview_policy(zero)
    assert result == service.preview_policy(zero)
    assert result["candidates"][1]["weights"] == result["candidates"][2]["weights"]
    robust = service.preview_policy(request)
    assert robust["candidates"][2]["weights"]["股票"] < robust["candidates"][1]["weights"]["股票"]
    signatures = {k.__name__: list(k.signatures) for k in kernels.KERNELS}
    service.preview_policy(request)
    assert signatures == {k.__name__: list(k.signatures) for k in kernels.KERNELS}


def test_source_changes_and_currency_mismatch_fail_closed(workspace):
    service, _ = workspace
    _, cma, request = saved_inputs(service)
    wrong = service.save_mandate(MandateRequest(name="美元目标", as_of=date.today(), review_date=date.today()+timedelta(days=90), currency="USD"))
    with pytest.raises(ValidationError, match="币种"):
        service.preview_policy(request.model_copy(update={"mandate_id": wrong["id"]}))
    path = service.data.data_dir / "asset_nv.parquet"
    nav = pd.read_parquet(path)
    nav.loc[0, "nv"] += .001
    nav.to_parquet(path, index=False)
    with pytest.raises(ConflictError, match="来源已变"):
        service.preview_policy(request)
    assert service.get_cma(cma["id"])["content_hash"] == cma["content_hash"]


def test_real_risk_reference_requires_exact_unchanged_values(workspace):
    service, days = workspace
    ref_request = RiskReferenceRequest(alloc_name="股债", as_of=date.today(), start_date=days[0].date(), end_date=days[-1].date())
    reference = service.risk_reference(ref_request)
    assert reference["observations"] == 160
    assert not service.artifacts.root.exists()
    body = definition().model_dump()
    body.update(risk_origin="historical_reference", risk_reference=ref_request, risk_reference_hash=reference["preview_hash"])
    body["correlation"] = reference["correlation"]
    for i, asset in enumerate(body["assets"]):
        asset["annual_volatility"] = reference["volatility"][i]
    service.preview_cma(CmaRequest(**body))
    body["assets"][0]["annual_volatility"] += .001
    with pytest.raises(ValidationError, match="人工修改"):
        service.preview_cma(CmaRequest(**body))


def test_risk_reference_rejects_all_asset_missing_trading_day(workspace):
    service, days = workspace
    path = service.data.data_dir / "asset_nv.parquet"
    nav = pd.read_parquet(path)
    missing_day = days[50]
    nav = nav.loc[pd.to_datetime(nav["date"]) != missing_day]
    nav.to_parquet(path, index=False)
    request = RiskReferenceRequest(alloc_name="股债", as_of=date.today(), start_date=days[0].date(), end_date=days[-1].date())
    with pytest.raises(ValidationError, match="SSE 交易日不连续") as caught:
        service.risk_reference(request)
    assert any(item["date"] == missing_day.strftime("%Y-%m-%d") for item in caught.value.diagnostics or [])


def test_risk_reference_rejects_non_daily_annualization(workspace):
    service, days = workspace
    request = RiskReferenceRequest(alloc_name="股债", as_of=date.today(), start_date=days[0].date(), end_date=days[-1].date(), periods_per_year=52)
    with pytest.raises(ValidationError, match="252"):
        service.risk_reference(request)


def test_policy_limits_shared_with_direct_application(workspace):
    service, _ = workspace
    _, _, request = saved_inputs(service)
    preview = service.preview_policy(request)
    policy = service.publish_policy(PublishPolicyRequest(request=request, preview_hash=preview["preview_hash"],
        candidate_id="robust-utility", name="受约束政策", reason="用于后续战术预算校验"))
    weights = {a["id"]: a["base_weight"] for a in policy["assets"]}
    check = check_policy(policy, weights, .04, str(date.today()))
    assert check["within_limits"]
    assert check["expected_tracking_error"] == pytest.approx(0.0)
    with pytest.raises(ValidationError, match="主动风险"):
        require_policy_application(policy, weights, .05, str(date.today()))
    active = copy.deepcopy(policy)
    active["policy"]["mandate"]["max_tracking_error"] = .005
    target = {active["assets"][0]["id"]: weights[active["assets"][0]["id"]] + .10,
              active["assets"][1]["id"]: weights[active["assets"][1]["id"]] - .10}
    active_check = check_policy(active, target, .005, str(date.today()))
    assert active_check["expected_tracking_error"] > .005
    assert not active_check["within_limits"]
    with pytest.raises(ValidationError, match="预期主动风险"):
        require_policy_application(active, target, .005, str(date.today()))
    expired = copy.deepcopy(policy)
    expired["policy"]["expires_on"] = str(date.today() - timedelta(days=1))
    with pytest.raises(ValidationError, match="复核"):
        require_policy_application(expired, weights, .04, str(date.today()))
    tight = copy.deepcopy(policy)
    tight["policy"]["mandate"]["max_volatility"] = .01
    with pytest.raises(ValidationError, match="波动"):
        require_policy_application(tight, weights, .04, str(date.today()))


def test_taa_cannot_expand_adopted_policy_budget(workspace):
    service, days = workspace
    _, _, request = saved_inputs(service)
    preview = service.preview_policy(request)
    policy = service.publish_policy(PublishPolicyRequest(request=request, preview_hash=preview["preview_hash"],
        candidate_id="robust-utility", name="传入TAA", reason="用于战术配置预算测试"))
    taa = TacticalAllocationService(service.artifacts.root.parent.parent, service.data.data_dir)
    req = PreviewRequest(baseline_id=policy["id"], start_date=days[0].date(), train_end_date=days[100].date(),
        end_date=days[-1].date(), as_of=date.today(), signal_mode="manual", manual_tilts={"股票": .01, "债券": -.01})
    with pytest.raises(ValidationError, match="政策预算"):
        taa.preview(req)
    result = taa.preview(req.model_copy(update={"max_tracking_error": .04}))
    assert result["policy_check"]["mandate_id"] == policy["policy"]["mandate_id"]


def test_api_errors_are_actionable_and_client_results_are_forbidden(workspace):
    service, _ = workspace
    app = FastAPI()
    app.include_router(build_router(service))
    with TestClient(app) as client:
        body = definition().model_dump(mode="json")
        body["correlation"] = [[1, .2], [.1, 1]]
        response = client.post("/api/strategic-allocation/cma/preview", json=body)
        assert response.status_code == 422
        assert "对称" in response.json()["detail"]["message"]
        body = definition().model_dump(mode="json")
        body["performance"] = {"return": 999}
        assert client.post("/api/strategic-allocation/cma/preview", json=body).status_code == 422
        assert client.get("/api/strategic-allocation/catalog").status_code == 200


@pytest.mark.parametrize("patch", [{"basis_confirmed": False}, {"assets": []}, {"horizon_years": True}, {"correlation": [[1.]]}])
def test_strict_input_contracts(patch):
    body = definition().model_dump(mode="json")
    body.update(patch)
    with pytest.raises(InputError):
        CmaRequest.model_validate(body)


def test_service_fails_until_worker_is_warm(workspace, monkeypatch):
    service, _ = workspace
    monkeypatch.setattr(kernels, "_WARMED_PID", None)
    with pytest.raises(RuntimeError, match="预热"):
        service.preview_cma(definition())

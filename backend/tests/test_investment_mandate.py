"""Independent funding math and real offline mandate -> CMA -> SAA acceptance."""
from datetime import date, timedelta
import copy

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError as InputError

from backend.tests.test_strategic_allocation import workspace, warm, definition, saved_inputs
from backend.strategic_allocation import goal_kernels as goals, kernels
from backend.strategic_allocation.contracts import (
    MandateRequest, MandateStudyRequest, ConfirmMandateRequest, PublishCmaRequest,
    PolicyRequest, PublishPolicyRequest,
)
from backend.strategic_allocation.planning import funding_inputs
from backend.strategic_allocation.routes import build_router
from backend.strategic_allocation.policy_gate import check_policy, require_policy_application
from backend.custom_indicators.errors import ConflictError, ValidationError


def plan(**updates):
    result = {"total_capital": 1_000_000., "outside_reserve": 100_000., "terminal_target": 1_500_000.,
              "amount_basis": "nominal", "inflation": 0.02, "annual_fee": 0.005,
              "required_probability": 0.8, "liquidity_months": 12, "contribution_stress_ratio": 0.5,
              "drawdown_alert": 0.2, "flows": []}
    result.update(updates)
    return result


def mandate(**updates):
    data = {"name": "十年资金目标", "as_of": str(date.today()),
            "review_date": str(date.today() + timedelta(days=180)), "currency": "CNY", "horizon_years": 10,
            "objective_kind": "funding_goal", "funding_plan": plan(), "target_return": 0.,
            "max_volatility": 0.2, "max_tracking_error": 0., "boundary_reason": "必要支出与损失承受能力经研究员确认"}
    data.update(updates)
    return MandateRequest.model_validate(data)


def test_required_return_matches_compound_growth_and_fee():
    zeros = np.zeros(120)
    rate, status = goals.required_return_kernel(100., zeros, zeros, 200., 0.)
    assert status == 0
    assert rate == pytest.approx(2 ** 0.1 - 1, abs=1e-12)
    with_fee, _ = goals.required_return_kernel(100., zeros, zeros, 200., 0.01)
    assert with_fee == pytest.approx((1 + rate) / 0.99 - 1, abs=1e-12)


def test_schedule_real_values_and_reserve_never_double_deducted():
    body = mandate(funding_plan=plan(amount_basis="real", flows=[
        {"name": "投入", "kind": "contribution", "amount": 2000., "first_month": 1, "last_month": 12, "every_months": 1},
        {"name": "支付", "kind": "withdrawal", "amount": 60_000., "first_month": 6, "last_month": 6, "every_months": 1}]))
    summary, inflows, outflows = funding_inputs(body.model_dump(mode="json"))
    assert summary["investable_capital"] == 900_000.
    assert summary["nominal_terminal_target"] == pytest.approx(1_500_000 * 1.02 ** 10)
    expected = np.array([2000 * 1.02 ** ((m + 1) / 12) if m < 12 else 0 for m in range(120)])
    np.testing.assert_allclose(inflows, expected)
    assert outflows[5] == pytest.approx(60_000 * 1.02 ** 0.5)
    expected_reserve = max(0., np.max(np.cumsum(outflows[:12] - inflows[:12] * .5)))
    assert summary["required_liquid_capital"] == pytest.approx(expected_reserve)
    assert summary["required_liquid_weight"] == pytest.approx(expected_reserve / 900_000)
    assert not inflows.flags.writeable


def test_missed_payment_is_not_erased_by_later_deposit():
    inflows, outflows = np.zeros(12), np.zeros(12)
    outflows[0], inflows[1] = 150., 1000.
    result, fan = goals.funding_paths_kernel(np.zeros((12, 100, 1)), 0., 0., 100., inflows, outflows,
                                            500., 0., .8, .2, 1.)
    assert result[0] == 0 and result[3] == 1
    assert result[5] == 1000 and result[8] == 50.
    assert result[11] == 150.
    assert result[14] == 0  # Lower terminal target does not repair a missed payment.
    assert np.all(fan[0] == 100.)


@pytest.mark.parametrize("probability,count", [(0, 500), (1, 500), (.5, 1000), (.9, 2000)])
def test_wilson_interval_matches_independent_reference(probability, count):
    k = round(probability * count)
    low, high = goals.wilson_interval_kernel(k, count)
    z = 1.959963984540054
    p = k / count
    mid = (p + z*z/(2*count))/(1+z*z/count)
    half = z*np.sqrt(p*(1-p)/count+z*z/(4*count**2))/(1+z*z/count)
    assert low == pytest.approx(max(0., mid-half))
    assert high == pytest.approx(min(1., mid+half))


def test_zero_risk_paths_match_independent_month_end_cashflow_recursion():
    inflows, outflows = np.full(24, 10.), np.full(24, 3.)
    result, fan = goals.funding_paths_kernel(np.zeros((24, 500, 1)), .06, 0., 100., inflows, outflows,
                                            300., .01, .8, .2, 1.)
    expected = 100.
    for _ in range(24):
        expected = expected * (1.06 * .99) ** (1/12) + 10 - 3
    assert result[5] == pytest.approx(expected)
    assert fan[-1, 0] == pytest.approx(expected)
    assert result[0] == float(expected >= 300)
    assert result[9] == 0.


def test_goal_kernel_uses_readonly_strided_views_without_new_signatures():
    rng = np.random.default_rng(79)
    raw = rng.normal(size=(48, 1000, 1))
    draws = raw[::2, ::2, :]
    raw.flags.writeable = False
    draws.flags.writeable = False
    flows = np.zeros(48)
    view = flows[::2]
    view.flags.writeable = False
    assert np.shares_memory(raw, draws) and np.shares_memory(flows, view)
    before = raw.copy()
    signatures = [tuple(k.signatures) for k in goals.KERNELS]
    first = goals.funding_paths_kernel(draws, .07, .15, 100., view, view, 120., 0., .8, .2, 1.)
    second = goals.funding_paths_kernel(draws.copy(), .07, .15, 100., view.copy(), view.copy(), 120., 0., .8, .2, 1.)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(before, raw)
    assert signatures == [tuple(k.signatures) for k in goals.KERNELS]
    assert goals.execution_audit()["python_fallback"] == 0


@pytest.mark.parametrize("patch", [
    {"funding_plan": plan(outside_reserve=1_000_000.)},
    {"funding_plan": plan(flows=[{"name": "超期", "kind": "withdrawal", "amount": 1., "first_month": 121, "last_month": 121}])},
    {"target_return": .08}, {"funding_plan": None},
    {"objective_kind": "absolute_return"},
    {"benchmark": {"name": "错轴", "alloc_name": "股债", "weights": {"股票": .2}, "target_excess_return": .01, "max_tracking_error": .1}},
])
def test_plan_contracts_reject_ambiguous_or_truncated_inputs(patch):
    with pytest.raises(InputError):
        mandate(**patch)


def test_no_cma_has_real_funding_math_but_no_fabricated_probability(workspace):
    service, _ = workspace
    body = MandateStudyRequest(definition=mandate())
    result = service.preview_mandate(body)
    assert result["status"] == "inputs_only"
    assert result["funding"]["required_effective_return"] is not None
    assert result["candidates"] == [] and result["cma"] is None
    assert not service.artifacts.root.exists()
    with pytest.raises(ConflictError):
        service.confirm_mandate(ConfirmMandateRequest(request=body, preview_hash="0"*64, acknowledge_limits=True))
    assert not service.artifacts.root.exists()
    saved = service.confirm_mandate(ConfirmMandateRequest(request=body, preview_hash=result["preview_hash"], acknowledge_limits=True))
    assert saved["assessment"]["status"] == "inputs_only"
    assert service.get_mandate(saved["id"]) == saved


def test_infeasible_liquidity_is_diagnosis_not_a_false_success(workspace):
    service, _ = workspace
    body = mandate(funding_plan=plan(flows=[{"name": "近期大额支付", "kind": "withdrawal", "amount": 2_000_000.,
                                           "first_month": 1, "last_month": 1}]))
    result = service.preview_mandate(MandateStudyRequest(definition=body))
    assert result["status"] == "needs_revision" and result["blockers"]


def test_real_cma_diagnosis_and_saa_consume_the_same_goal_gate(workspace):
    service, _ = workspace
    _, cma, _ = saved_inputs(service)
    body = MandateStudyRequest(definition=mandate(funding_plan=plan(terminal_target=500_000.)), cma_id=cma["id"])
    preview = service.preview_mandate(body)
    assert preview["status"] == "diagnosed"
    for candidate in preview["candidates"]:
        check = candidate["goal_check"]
        assert check["central"]["success_probability"] >= check["central"]["probability_lower"]
        assert check["conservative"]["success_probability"] <= check["central"]["success_probability"] + .03
        assert check["central"]["additional_initial_capital"] >= 0
    saved = service.confirm_mandate(ConfirmMandateRequest(request=body, preview_hash=preview["preview_hash"], acknowledge_limits=True))
    policy_request = PolicyRequest(mandate_id=saved["id"], cma_id=cma["id"])
    policy = service.preview_policy(policy_request)
    assert policy["funding"] == preview["funding"]
    assert policy["candidates"] == preview["candidates"]
    chosen = next(c for c in policy["candidates"] if c["goal_check"]["within_limits"])
    published = service.publish_policy(PublishPolicyRequest(request=policy_request, preview_hash=policy["preview_hash"],
                                      candidate_id=chosen["id"], name="诊断后政策", reason="资金目标通过本次诊断门槛"))
    assert published["policy"]["selection"]["goal_check"]["within_limits"]


def test_missed_goal_cannot_be_adopted_via_direct_api(workspace):
    service, _ = workspace
    _, cma, _ = saved_inputs(service)
    saved = service.save_mandate(mandate(funding_plan=plan(terminal_target=40_000_000.)))
    request = PolicyRequest(mandate_id=saved["id"], cma_id=cma["id"])
    result = service.preview_policy(request)
    assert all(not c["goal_check"]["within_limits"] for c in result["candidates"])
    with pytest.raises(ValidationError, match="成功概率"):
        service.publish_policy(PublishPolicyRequest(request=request, preview_hash=result["preview_hash"],
                               candidate_id="maximum-return", name="不能采用", reason="测试直接入口不能绕过目标门槛"))


def test_benchmark_and_mandate_bounds_are_consumed_not_just_stored(workspace):
    service, _ = workspace
    _, cma, _ = saved_inputs(service)
    definition_data = mandate().model_dump(mode="json")
    definition_data.update(objective_kind="benchmark_relative", funding_plan=None, benchmark={
        "name": "股债六四政策基准", "alloc_name": "股债", "weights": {"股票": .6, "债券": .4},
        "target_excess_return": 0., "max_tracking_error": 0.})
    saved = service.save_mandate(MandateRequest.model_validate(definition_data))
    request = PolicyRequest(mandate_id=saved["id"], cma_id=cma["id"])
    result = service.preview_policy(request)
    for item in result["candidates"]:
        assert item["weights"] == pytest.approx({"股票": .6, "债券": .4})
        assert item["benchmark_check"]["tracking_error"] == pytest.approx(0.)
    definition_data.update(objective_kind="absolute_return", benchmark=None, allocation_scope="股债",
        asset_limits={"股票": {"min_weight": .1, "max_weight": .2, "max_abs_tilt": .01}})
    saved = service.save_mandate(MandateRequest.model_validate(definition_data))
    result = service.preview_policy(request.model_copy(update={"mandate_id": saved["id"]}))
    assert all(.1 - 1e-8 <= c["weights"]["股票"] <= .2 + 1e-8 for c in result["candidates"])


def test_historical_research_is_not_rejected_by_today_but_application_is(workspace):
    service, days = workspace
    historical_day = days[-30].date()
    review_day = days[-10].date()
    saved = service.save_mandate(MandateRequest(name="历史目标", as_of=historical_day, review_date=review_day))
    cma_req = definition().model_copy(update={"as_of": historical_day})
    cma_preview = service.preview_cma(cma_req)
    cma = service.publish_cma(PublishCmaRequest(request=cma_req, preview_hash=cma_preview["preview_hash"]))
    request = PolicyRequest(mandate_id=saved["id"], cma_id=cma["id"])
    result = service.preview_policy(request)
    assert result["current_application_eligible"] is False
    policy = service.publish_policy(PublishPolicyRequest(request=request, preview_hash=result["preview_hash"],
                                    candidate_id="minimum-risk", name="历史研究政策", reason="仅为历史研究保存，不用于当前应用"))
    weights = {a["id"]: a["base_weight"] for a in policy["assets"]}
    assert check_policy(policy, weights, .1, str(historical_day))["within_limits"]
    with pytest.raises(ValidationError, match="复核"):
        require_policy_application(policy, weights, .1, str(historical_day))


def test_new_api_roundtrip_recompute_and_model_mismatch(workspace):
    service, _ = workspace
    app = FastAPI()
    app.include_router(build_router(service))
    request = MandateStudyRequest(definition=mandate()).model_dump(mode="json")
    with TestClient(app) as client:
        response = client.post("/api/strategic-allocation/mandates/preview", json=request)
        assert response.status_code == 200
        preview = response.json()
        changed = copy.deepcopy(request)
        changed["definition"]["funding_plan"]["terminal_target"] += 1
        bad = client.post("/api/strategic-allocation/mandates/confirm", json={
            "request": changed, "preview_hash": preview["preview_hash"], "acknowledge_limits": True})
        assert bad.status_code == 409
        assert not service.artifacts.root.exists()
        saved = client.post("/api/strategic-allocation/mandates/confirm", json={
            "request": request, "preview_hash": preview["preview_hash"], "acknowledge_limits": True})
        assert saved.status_code == 201
        assert client.get('/api/strategic-allocation/mandates/' + saved.json()['id']).json() == saved.json()
        request["simulation_paths"] = 1_000_000
        assert client.post("/api/strategic-allocation/mandates/preview", json=request).status_code == 422


def test_cma_time_currency_and_horizon_cannot_be_silently_substituted(workspace):
    service, _ = workspace
    _, cma, _ = saved_inputs(service)
    for patch in ({"currency": "USD"}, {"horizon_years": 5}, {"as_of": str(date.today()-timedelta(days=1))}):
        with pytest.raises(ValidationError, match="同研究日"):
            service.preview_mandate(MandateStudyRequest(definition=mandate(**patch), cma_id=cma["id"]))


def test_no_feasible_candidate_is_explicit_diagnosis(workspace):
    service, _ = workspace
    _, cma, _ = saved_inputs(service)
    result = service.preview_mandate(MandateStudyRequest(definition=mandate(max_volatility=0.00001), cma_id=cma["id"]))
    assert result["status"] == "needs_revision" and result["blockers"]
    assert result["candidates"] == []


def test_annual_simple_moments_probability_and_quantiles_have_independent_reference():
    import math
    draws, _ = goals.seeded_factor_draws_kernel(12, 30000, 1, 931, 0, 5.)
    mu, sigma, fee = .08, .2, .01
    s2 = np.log1p((sigma / (1 + mu)) ** 2)
    m = np.log1p(mu) - .5 * s2
    gross = np.exp(m + np.sqrt(s2 / 12) * draws[:, :, 0].sum(axis=0))
    assert gross.mean() - 1 == pytest.approx(mu, abs=.006)
    assert gross.std(ddof=1) == pytest.approx(sigma, abs=.006)
    result, _ = goals.funding_paths_kernel(draws, mu, sigma, 100., np.zeros(12), np.zeros(12), 110., fee, .8, .2, 1.)
    balances = 100 * gross * (1 - fee)
    np.testing.assert_allclose(result[4:7], np.quantile(balances, [.05, .5, .95]), rtol=1e-12)
    analytic_probability = .5 * math.erfc((np.log(1.1) - m - np.log1p(-fee)) / np.sqrt(2*s2))
    assert result[0] == pytest.approx(analytic_probability, abs=.012)
    assert result[11] == pytest.approx(np.quantile(110 / (gross * (1-fee)), .8), rel=1e-12)
    assert result[7] == pytest.approx(np.maximum(110-balances, 0).mean(), rel=1e-12)


def test_empty_and_out_of_range_funding_math_fails_closed():
    with pytest.raises(ValueError):
        goals.required_return_kernel(100., np.empty(0), np.empty(0), 100., 0.)
    assert goals.required_return_kernel(100., np.zeros(12), np.zeros(12), 1e9, 0.)[1] == 2
    assert goals.required_return_kernel(100., np.full(12, 1000.), np.zeros(12), 1., 0.)[1] == 1
    with pytest.raises(ValueError):
        goals.funding_schedule_kernel(np.array([[1, 13, 1]], dtype=np.int64), np.array([5.]), 12, 0., 0)
    for value in (np.nan, np.inf):
        with pytest.raises(ValueError):
            goals.funding_paths_kernel(np.zeros((12, 500, 1)), .05, .1, value, np.zeros(12), np.zeros(12), 100., 0., .8, .2, 1.)
    broken = np.zeros((12, 500, 1)); broken[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        goals.funding_paths_kernel(broken, .05, .1, 100., np.zeros(12), np.zeros(12), 100., 0., .8, .2, 1.)


def test_required_capital_reconciles_with_all_payment_dates_not_only_terminal():
    draws, _ = goals.seeded_factor_draws_kernel(24, 1000, 1, 61, 0, 5.)
    inflows, outflows = np.zeros(24), np.zeros(24)
    inflows[12], outflows[0], outflows[18] = 1000., 150., 250.
    result, _ = goals.funding_paths_kernel(draws, .05, .1, 100., inflows, outflows, 500., .01, .8, .2, 1.)
    s2 = np.log1p((.1/1.05)**2)
    growth = np.exp(np.cumsum((np.log(1.05)-s2/2+np.log(.99))/12 + np.sqrt(s2/12)*draws[:, :, 0], axis=0))
    pv = np.cumsum((inflows-outflows)[:, None]/growth, axis=0)
    required = np.maximum(np.maximum(0., (-pv).max(axis=0)), 500/growth[-1]-pv[-1])
    assert result[11] == pytest.approx(np.quantile(required, .8), rel=1e-12)
    assert result[3] > .99  # A later large contribution cannot rescue the missed first payment.


def test_payment_only_goal_is_valid_but_empty_success_definition_is_rejected():
    with pytest.raises(InputError):
        mandate(funding_plan=plan(terminal_target=0.))
    definition_data = mandate(horizon_years=1, funding_plan=plan(terminal_target=0., flows=[
        {"name": "期末必要支付", "kind": "withdrawal", "amount": 900_000., "first_month": 12, "last_month": 12}])).model_dump(mode="json")
    summary, inflows, outflows = funding_inputs(definition_data)
    assert summary["nominal_terminal_target"] == 0.
    result, _ = goals.funding_paths_kernel(np.zeros((12, 500, 1)), 0., 0., 900_000., inflows, outflows, 0., 0., .8, .2, 1.)
    assert result[0] == 1 and result[5] == 0


def test_taa_zero_budget_can_be_researched_without_allowing_weight_tilts(workspace):
    from backend.tactical_allocation.contracts import PreviewRequest
    service, days = workspace
    _, cma, _ = saved_inputs(service)
    saved = service.save_mandate(MandateRequest(name="不允许战术偏离", as_of=date.today(),
        review_date=date.today()+timedelta(days=180), max_tracking_error=0.))
    request = PolicyRequest(mandate_id=saved["id"], cma_id=cma["id"])
    preview = service.preview_policy(request)
    published = service.publish_policy(PublishPolicyRequest(request=request, preview_hash=preview["preview_hash"],
        candidate_id="nominal-utility", name="仅SAA", reason="研究授权不允许主动战术偏离"))
    assert all(asset["max_abs_tilt"] == 0 for asset in published["assets"])
    body = PreviewRequest(baseline_id=published["id"], as_of=date.today(), start_date=days[0].date(),
        train_end_date=days[90].date(), end_date=days[-1].date(), max_tracking_error=0, signal_mode="momentum")
    from backend.tactical_allocation.service import TacticalAllocationService
    tactical = TacticalAllocationService(service.artifacts.root.parent.parent, service.data.data_dir)
    result = tactical.preview(body)
    assert result["recommendation"]["weights"] == pytest.approx({a["id"]: a["base_weight"] for a in published["assets"]})
    assert result["policy_check"]["within_limits"]

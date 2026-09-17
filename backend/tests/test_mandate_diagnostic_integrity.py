"""Independent acceptance-rule, authorization and missing-evidence regressions."""
import numpy as np
import pytest
from pydantic import ValidationError as InputError

from backend.tests.test_investment_mandate import mandate, plan
from backend.tests.test_strategic_allocation import confirmed_mandate, workspace, warm, saved_inputs
from backend.strategic_allocation import goal_kernels as goals
from backend.strategic_allocation.contracts import (
    MandateStudyRequest, ConfirmMandateRequest, PolicyRequest, PublishPolicyRequest,
)
from backend.strategic_allocation.planning import funding_inputs
from backend.custom_indicators.errors import ValidationError


def _wilson_lower(k, n):
    p, z = k / n, 1.959963984540054
    return max(0., (p + z*z/(2*n) - z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))) / (1+z*z/n))


@pytest.mark.parametrize("count,threshold", [(500, .5), (500, .8), (500, .99), (2000, .9)])
def test_capital_matches_the_same_lower_bound_used_for_adoption(count, threshold):
    values = np.arange(1, count+1, dtype=float)
    original = values.copy()
    values.flags.writeable = False
    capital, probability, lower, upper = goals.capital_gate_kernel(values, threshold)
    rank = next(k for k in range(count+1) if _wilson_lower(k, count) >= threshold)
    assert capital == values[rank-1]
    assert lower == pytest.approx(_wilson_lower(rank, count))
    assert probability == rank/count and lower >= threshold and upper >= probability
    assert _wilson_lower(rank-1, count) < threshold
    np.testing.assert_array_equal(values, original)


def test_capital_ties_and_insufficient_sampling_do_not_fake_a_solution():
    capital, probability, lower, _ = goals.capital_gate_kernel(np.full(500, 100.), .9)
    assert capital == 100. and probability == 1 and lower >= .9
    assert all(np.isnan(value) for value in goals.capital_gate_kernel(np.arange(8, dtype=float), .99))
    assert goals.capital_gate_kernel(np.zeros(500), .99)[0] == 0.


@pytest.mark.parametrize("values,threshold", [([], .8), ([2., 1.], .8), ([np.inf], .8), ([np.nan], .8), ([-1.], .8), ([1.], np.nan), ([1.], 1.1)])
def test_invalid_capital_samples_fail_closed(values, threshold):
    with pytest.raises(ValueError, match="FUNDING_CAPITAL_GATE_INPUT"):
        goals.capital_gate_kernel(np.array(values), threshold)


def test_capital_reinjected_into_identical_paths_passes_the_actual_goal_gate():
    draws, _ = goals.seeded_factor_draws_kernel(24, 2000, 1, 329, 0, 5.)
    draws.flags.writeable = False
    inflows, outflows = np.zeros(24), np.zeros(24)
    inflows[12], outflows[0], outflows[18] = 150., 100., 200.
    before = [tuple(k.signatures) for k in goals.KERNELS]
    result, _ = goals.funding_paths_kernel(draws, .06, .12, 100., inflows, outflows, 300., .005, .8, .2, 1.)
    assert result[15] >= result[11]
    assert result[16] == pytest.approx(result[15] - 100.)
    rerun, _ = goals.funding_paths_kernel(draws, .06, .12, result[15], inflows, outflows, 300., .005, .8, .2, 1.)
    assert rerun[1] >= .8
    assert rerun[0] == pytest.approx(result[17], abs=1/2000)
    assert before == [tuple(k.signatures) for k in goals.KERNELS]


@pytest.mark.parametrize("months,threshold", [(12, .5), (120, .8), (360, .99)])
def test_capital_gate_remains_valid_across_horizons_and_probability_thresholds(months, threshold):
    draws, _ = goals.seeded_factor_draws_kernel(months, 500, 1, 90, 0, 5.)
    inflows, outflows = np.zeros(months), np.full(months, 2000.)
    inflows[months // 2] = 100000.
    result, _ = goals.funding_paths_kernel(draws, .07, .17, 900000., inflows, outflows, 2000000., .01, threshold, .2, 1.)
    rerun, _ = goals.funding_paths_kernel(draws, .07, .17, result[15], inflows, outflows, 2000000., .01, threshold, .2, 1.)
    assert rerun[1] >= threshold


def test_capital_gate_accepts_a_readonly_strided_view_without_compilation_or_mutation():
    owner = np.arange(1000, dtype=np.float64)
    view = owner[::2]
    owner.flags.writeable = False
    view.flags.writeable = False
    before = tuple(goals.capital_gate_kernel.signatures)
    result = goals.capital_gate_kernel(view, .8)
    assert np.shares_memory(owner, view)
    assert result == goals.capital_gate_kernel(view.copy(), .8)
    np.testing.assert_array_equal(owner, np.arange(1000, dtype=np.float64))
    assert before == tuple(goals.capital_gate_kernel.signatures)


@pytest.mark.parametrize("scale", [.01, 1., 1000., 100000.])
@pytest.mark.parametrize("seed", [0, 1, 5])
def test_reported_capital_replays_at_display_precision_across_amount_scales(scale, seed):
    draws, _ = goals.seeded_factor_draws_kernel(120, 500, 1, seed, 0, 5.)
    inflows, outflows = np.zeros(120), np.full(120, 2000. * scale)
    inflows[60] = 100000. * scale
    draws.flags.writeable = inflows.flags.writeable = outflows.flags.writeable = False
    signatures = [tuple(k.signatures) for k in goals.KERNELS]
    result, _ = goals.funding_paths_kernel(draws, .07, .17, 900000. * scale, inflows, outflows,
                                         2000000. * scale, .01, .8, .2, 1.)
    displayed_capital = float(f"{result[15]:.2f}")
    replay, _ = goals.funding_paths_kernel(draws, .07, .17, displayed_capital, inflows, outflows,
                                         2000000. * scale, .01, .8, .2, 1.)
    assert result[15] == displayed_capital
    assert replay[1] >= .8
    np.testing.assert_array_equal(replay[:3], result[17:20])
    assert signatures == [tuple(k.signatures) for k in goals.KERNELS]


def test_verified_capital_covers_early_payments_even_with_later_contributions():
    draws = np.zeros((24, 500, 1))
    inflows, outflows = np.zeros(24), np.zeros(24)
    outflows[0], inflows[1] = 1234567890.123, 3e9
    result, _ = goals.funding_paths_kernel(draws, 0., 0., 1e9, inflows, outflows, 1e9, 0., .8, .2, 1.)
    assert result[0] == 0. and result[3] == 1.
    assert result[15] == 1234567890.13
    replay, _ = goals.funding_paths_kernel(draws, 0., 0., result[15], inflows, outflows, 1e9, 0., .8, .2, 1.)
    assert replay[3] == 0. and replay[1] >= .8
    np.testing.assert_array_equal(replay[:3], result[17:20])


def test_payment_buffer_is_zero_return_cash_coverage_not_a_new_cash_deduction():
    definition = mandate(funding_plan=plan(flows=[{
        "name": "首期支付", "kind": "withdrawal", "amount": 200_000., "first_month": 1, "last_month": 1,
    }])).model_dump(mode="json")
    summary, _, _ = funding_inputs(definition)
    assert summary["investable_capital"] == 900_000.
    assert summary["liquidity_payment_buffer"] == 700_000.
    assert summary["liquidity_payment_buffer_ratio"] == pytest.approx(7/9)
    assert summary["liquidity_shortfall_capital"] == 0.
    definition["funding_plan"]["flows"][0]["amount"] = 1_100_000.
    summary, _, _ = funding_inputs(definition)
    assert summary["liquidity_payment_buffer"] == 0 and summary["liquidity_shortfall_capital"] == 200_000.


@pytest.mark.parametrize("malformation", ["missing", "null", "wrong_flag", "nan", "wrong_threshold"])
def test_policy_application_rejects_missing_or_inconsistent_goal_evidence(workspace, monkeypatch, malformation):
    service, _ = workspace
    _, cma, _ = saved_inputs(service)
    saved = confirmed_mandate(service, mandate(funding_plan=plan(terminal_target=30_000_000.)))
    request = PolicyRequest(mandate_id=saved["id"], cma_id=cma["id"])
    preview = service.preview_policy(request)
    candidate = preview["candidates"][0]
    if malformation == "missing":
        candidate.pop("goal_check")
    elif malformation == "null":
        candidate["goal_check"] = None
    elif malformation == "wrong_flag":
        candidate["goal_check"]["within_limits"] = True
    elif malformation == "nan":
        candidate["goal_check"]["central"]["probability_lower"] = np.nan
    else:
        candidate["goal_check"]["threshold"] = 0.
    monkeypatch.setattr(service, "preview_policy", lambda _: preview)
    with pytest.raises(ValidationError, match="资金目标"):
        service.publish_policy(PublishPolicyRequest(request=request, preview_hash=preview["preview_hash"],
            candidate_id=candidate["id"], name="不应创建的政策", reason="不能以缺失或矛盾证据通过采纳"))
    assert service.baselines.list_baselines() == []


def test_preview_cannot_turn_a_missing_diagnosis_into_diagnosed(workspace, monkeypatch):
    service, _ = workspace
    _, cma, _ = saved_inputs(service)
    request = MandateStudyRequest(definition=mandate(), cma_id=cma["id"])
    original = service._candidate_calculation
    def without_check(*args, **kwargs):
        result = original(*args, **kwargs)
        for candidate in result["candidates"]:
            candidate.pop("goal_check")
        return result
    monkeypatch.setattr(service, "_candidate_calculation", without_check)
    with pytest.raises(ValidationError, match="资金目标"):
        service.preview_mandate(request)


@pytest.mark.parametrize("patch", [
    {"asset_limits": {"股票": {"max_weight": .3}}},
    {"group_limits": [{"id": "权益", "assets": ["股票"], "lo": 0., "hi": .4}]},
    {"allocation_scope": "股债", "group_limits": [{"id": "权益", "assets": ["股票", "股票"], "lo": 0., "hi": .4}]},
    {"allocation_scope": "股债", "group_limits": [{"id": "重复", "assets": ["股票"], "lo": 0., "hi": .4}]*2},
])
def test_authorization_requires_unique_members_and_bound_allocation_scope(patch):
    with pytest.raises(InputError):
        mandate(**patch)


def test_saa_search_seed_does_not_replace_confirmed_goal_simulation_seed(workspace):
    service, _ = workspace
    _, cma, _ = saved_inputs(service)
    study = MandateStudyRequest(definition=mandate(funding_plan=plan(terminal_target=500_000.)),
                                cma_id=cma["id"], seed=917, simulation_paths=500)
    assessment = service.preview_mandate(study)
    saved = service.confirm_mandate(ConfirmMandateRequest(
        request=study, preview_hash=assessment["preview_hash"], acknowledge_limits=True))
    request = PolicyRequest(mandate_id=saved["id"], cma_id=cma["id"], seed=19)
    policy = service.preview_policy(request)
    assert policy["request"]["seed"] == 19  # SAA candidate exploration is still configurable.
    assert policy["funding_model"]["seed"] == 917  # The goal's scenario ensemble is frozen.
    assert policy["funding_model"]["paths"] == 500
    changed = service.preview_policy(request.model_copy(update={"seed": 53}))
    assert changed["funding_model"] == policy["funding_model"]
    assert service.get_mandate(saved["id"]) == saved


def test_confirmed_input_only_goal_retains_its_frozen_simulation_settings(workspace):
    service, _ = workspace
    _, cma, _ = saved_inputs(service)
    saved = confirmed_mandate(service, mandate())
    result = service.preview_policy(PolicyRequest(mandate_id=saved["id"], cma_id=cma["id"], seed=19))
    assert result["funding_model"]["seed"] == saved["planning_settings"]["seed"] == 42
    assert result["funding_model"]["paths"] == 2000


def test_same_asset_names_do_not_authorize_another_allocation(workspace):
    service, _ = workspace
    _, cma, _ = saved_inputs(service)
    saved = confirmed_mandate(service, mandate(allocation_scope="另一股债方案",
        asset_limits={"股票": {"max_weight": .3}}))
    with pytest.raises(ValidationError, match="大类方案"):
        service.preview_policy(PolicyRequest(mandate_id=saved["id"], cma_id=cma["id"]))

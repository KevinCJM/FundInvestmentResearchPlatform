"""Real offline risk-scale -> mandate reference search/validation integration."""
import numpy as np
import pytest
from contextlib import contextmanager
from pydantic import ValidationError as InputError

from backend.tests.risk_scale_app import make_service
from backend.tests.test_risk_scale_service import freeze_reference, definition as scale_definition, publish
from backend.tests.test_mandate_boundary_contracts import new_definition, budget, payment
import backend.strategic_allocation.service as service_module
from backend.strategic_allocation.service import StrategicAllocationService
from backend.strategic_allocation.contracts import MandateRequest, MandateStudyRequest, ConfirmMandateRequest
from backend.strategic_allocation import goal_kernels as goals, mandate_kernels as numeric
from backend.strategic_allocation import mandate_diagnosis
from backend.custom_indicators.errors import ConflictError


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    root = tmp_path_factory.mktemp("mandate-reference")
    risk, inputs = make_service(root)
    service = StrategicAllocationService(root / "research", root / "data", universe_dir=root / "data")
    service.artifacts = risk.artifacts
    service.risk_scales = risk
    service.warm()
    reference_input, _ = freeze_reference(risk, inputs)
    version, _, _ = publish(risk, scale_definition(reference_input))
    return service, version, inputs


def request_for(version, **updates):
    data = new_definition(objective_kind="funding_goal", cash_budget=budget(flows=[payment(90.)]),
        funding_target={"amount": 500., "amount_basis": "nominal"}, max_volatility=None,
        risk_authorization={"mode": "funding_suggestion", "source": "离线测试授权最高C3，系统只能给出建议",
            "authorized_max_level": 3, "risk_scale_ref": {k: version[k] for k in ("id", "content_hash")}})
    data.update(updates)
    return MandateStudyRequest(definition=MandateRequest.model_validate(data), simulation_paths=500, seed=43, validation_seed=971)


@pytest.mark.parametrize("objective_kind", ["absolute_return", "benchmark_relative"])
def test_funding_suggestion_rejects_payments_without_explicit_success_condition(reference, objective_kind):
    _, scale, _ = reference
    # A withdrawal is a cash-flow fact, not consent to a probability-based goal.
    with pytest.raises(InputError, match="明确的资金成功条件"):
        request_for(scale, objective_kind=objective_kind, funding_target=None, cash_protection=None)


def test_funding_suggestion_with_payment_protection_runs_independent_validation(reference):
    service, scale, _ = reference
    body = request_for(scale, objective_kind="absolute_return", funding_target=None,
                       cash_protection={"mode": "payments_only"})
    result = service.preview_mandate(body)
    study = result["reference_diagnosis"]
    assert study["status"] == "validated"
    assert study["validation"]["seed"] == body.validation_seed != body.seed
    assert study["validation"]["central"]["probability_lower"] >= .8
    assert result["risk_decision"]["status"] == "recommendation_validated"


def test_confirmation_holds_risk_scale_lock_through_preview_and_save(reference, monkeypatch):
    service, scale, _ = reference
    body = request_for(scale)
    preview = service.preview_mandate(body)
    active = False
    original_locked = service.risk_scales.store.document.locked

    @contextmanager
    def wrapped_locked():
        nonlocal active
        with original_locked():
            active = True
            try:
                yield
            finally:
                active = False

    original_save = service.artifacts.save

    def checked_save(*args, **kwargs):
        assert active
        return original_save(*args, **kwargs)

    monkeypatch.setattr(service.risk_scales.store.document, "locked", wrapped_locked)
    monkeypatch.setattr(service.artifacts, "save", checked_save)
    saved = service.confirm_mandate(ConfirmMandateRequest(
        request=body, preview_hash=preview["preview_hash"], acknowledge_limits=True))
    assert saved["artifact_type"] == "investment_mandate"


def test_frozen_scale_constrained_frontier_then_independent_validation(reference):
    service, scale, _ = reference
    body = request_for(scale)
    result = service.preview_mandate(body)
    study = result["reference_diagnosis"]
    assert study["status"] == "validated", study
    assert len(study["candidates"]) == 101
    assert study["search_seed"] != study["validation"]["seed"]
    assert study["validation"]["candidate_frozen_before_validation"]
    assert study["validation"]["central"]["probability_lower"] >= .8
    assert study["selected_candidate"]["weights"]["cash"] >= .1 - 1e-8
    assert study["minimum_tested_feasible_level"] <= 3
    assert study["distribution"]["engine"] == "base_period_moment_match_v2"
    assert result["risk_decision"]["selection_pending"] is False
    level = result["risk_decision"]["selected_max_level"]
    assert result["definition"]["max_volatility"] == scale["preview"]["result"]["applied_boundaries"][level - 1]
    assert body.definition.max_volatility is None  # Raw inputs remain unchanged.
    saved = service.confirm_mandate(ConfirmMandateRequest(request=body, preview_hash=result["preview_hash"], acknowledge_limits=True))
    assert saved["assessment"]["reference_diagnosis"] == study
    assert service.get_mandate(saved["id"])["content_hash"] == saved["content_hash"]


def test_reference_not_found_is_not_infeasible_proof(reference):
    service, scale, _ = reference
    result = service.preview_mandate(request_for(scale, funding_target={"amount": 1e9}))
    study = result["reference_diagnosis"]
    assert study["status"] == "no_validated_candidate_in_search"
    assert study["validation"] is None
    assert study["minimum_tested_feasible_level"] is None
    assert result["definition"]["max_volatility"] is None
    assert result["status"] == "needs_revision"
    adjustment = study["adjustment_diagnosis"]
    assert adjustment["purpose"] == "fixed_candidate_capital_diagnostic_only"
    assert adjustment["validation"]["central"]["gate_additional_initial_capital"] > 0
    assert adjustment["validation"]["central"]["gate_probability_lower"] >= .8
    assert study["selected_candidate"] is None  # Diagnostic capital does not authorize a portfolio.


def test_validation_failure_never_reselects_using_validation_draws(reference, monkeypatch):
    service, scale, _ = reference
    original = mandate_diagnosis.validate_fixed_candidate
    calls = []
    def fail(metrics, definition, prepared, request, spec):
        value = original(metrics, definition, prepared, request, spec)
        calls.append(metrics.copy())
        value["within_limits"] = False
        return value
    monkeypatch.setattr(mandate_diagnosis, "validate_fixed_candidate", fail)
    result = service.preview_mandate(request_for(scale))
    assert result["reference_diagnosis"]["status"] == "validation_failed"
    assert len(calls) == 1
    assert result["definition"]["max_volatility"] is None
    assert result["reference_diagnosis"]["minimum_tested_feasible_level"] is None


def test_compact_objective_freezes_model_conventions_and_reuses_scale_for_two_frontiers(reference):
    service, scale, _ = reference
    risk = {"mode": "manual_level", "source": "risk_scale_selection", "authorized_max_level": 3,
            "selected_max_level": 3, "risk_scale_ref": {k: scale[k] for k in ("id", "content_hash")}}
    body = request_for(scale, review_date=None, boundary_policy=None, min_cash_weight=.25,
                       risk_authorization=risk)
    result = service.preview_mandate(body)
    frozen = result["definition"]
    study = result["reference_diagnosis"]
    assert frozen["boundary_policy"]["source"] == "investment_objectives_model_convention_v1"
    assert frozen["boundary_policy"]["valid_until"] is None
    assert frozen["boundary_policy_hash"]
    assert study["cash_constraint"]["requested_min_cash_weight"] == pytest.approx(.25)
    assert study["cash_constraint"]["effective_min_cash_weight"] == pytest.approx(
        max(.25, study["cash_constraint"]["cashflow_derived_weight"]))
    assert study["risk_boundaries"] == scale["preview"]["result"]["applied_boundaries"]
    assert len(study["reference_frontier"]) == len(scale["preview"]["result"]["frontier"]) == 101
    assert len(study["constrained_frontier"]) == 101
    reference_points = [(p["volatility"], p["expected_return"]) for p in study["reference_frontier"]]
    constrained_points = [(p["volatility"], p["expected_return"]) for p in study["constrained_frontier"]]
    assert constrained_points != reference_points  # Re-optimized with the cash floor, not relabelled.


def test_funding_suggestion_refreezes_automatic_benchmark_at_recommended_level(reference, monkeypatch):
    service, scale, _ = reference
    body = request_for(scale, objective_kind="benchmark_relative", funding_target=None,
                       cash_protection={"mode": "payments_only"})
    monkeypatch.setattr(service_module, "diagnose_reference",
                         lambda *_args, **_kwargs: {"status": "validated", "minimum_tested_feasible_level": 1})
    result = service.preview_mandate(body)
    level = result["risk_decision"]["selected_max_level"]
    expected = scale["preview"]["result"]["levels"][level - 1]["representative_weights"]
    weights = list(result["definition"]["benchmark"]["weights"].values())
    assert weights == pytest.approx(expected)


def test_funding_suggestion_revalidates_against_refrozen_benchmark(reference, monkeypatch):
    service, scale, _ = reference
    body = request_for(scale, objective_kind="benchmark_relative", funding_target=None,
                       cash_protection={"mode": "payments_only"})
    calls = []

    def diagnose(*_args, **_kwargs):
        definition = _args[2]
        calls.append(definition.get("benchmark"))
        return {"status": "validated", "minimum_tested_feasible_level": 1}

    monkeypatch.setattr(service_module, "diagnose_reference", diagnose)
    service.preview_mandate(body)
    assert len(calls) == 2
    assert calls[0]["source"] == calls[1]["source"] == "risk_scale_reference"
    assert calls[0]["weights"] != calls[1]["weights"]


@pytest.mark.parametrize("second_result", [
    {"status": "validation_failed", "minimum_tested_feasible_level": None},
    {"status": "validated", "minimum_tested_feasible_level": 2},
])
def test_failed_or_unstable_refrozen_benchmark_does_not_authorize(reference, monkeypatch, second_result):
    service, scale, _ = reference
    body = request_for(scale, objective_kind="benchmark_relative", funding_target=None,
                       cash_protection={"mode": "payments_only"})
    results = iter([
        {"status": "validated", "minimum_tested_feasible_level": 1},
        second_result,
    ])
    monkeypatch.setattr(service_module, "diagnose_reference", lambda *_args, **_kwargs: next(results))
    result = service.preview_mandate(body)
    assert result["status"] == "needs_revision"
    assert result["definition"]["max_volatility"] is None
    assert result["risk_decision"]["selection_pending"] is True
    assert result["risk_decision"]["status"] == "awaiting_recommendation"


def test_absolute_return_does_not_repeat_diagnosis_for_unused_benchmark(reference, monkeypatch):
    service, scale, _ = reference
    body = request_for(scale, objective_kind="absolute_return", funding_target=None,
                       cash_protection={"mode": "payments_only"})
    calls = []

    def diagnose(*_args, **_kwargs):
        calls.append(True)
        return {"status": "validated", "minimum_tested_feasible_level": 1}

    monkeypatch.setattr(service_module, "diagnose_reference", diagnose)
    result = service.preview_mandate(body)
    assert len(calls) == 1
    assert result["risk_decision"]["status"] == "recommendation_validated"


def test_relative_goal_uses_selected_risk_scale_representative_as_frozen_benchmark(reference):
    service, scale, _ = reference
    risk = {"mode": "manual_level", "source": "risk_scale_selection", "authorized_max_level": 3,
            "selected_max_level": 3, "risk_scale_ref": {k: scale[k] for k in ("id", "content_hash")}}
    body = request_for(scale, objective_kind="benchmark_relative", target_return=0., target_excess_return=.005,
                       cash_budget=None, funding_target=None, boundary_policy=None, review_date=None,
                       max_volatility=None, risk_authorization=risk)
    result = service.preview_mandate(body)
    benchmark = result["definition"]["benchmark"]
    level = scale["preview"]["result"]["levels"][2]
    ids = scale["preview"]["result"]["ordered_asset_ids"]
    assert benchmark["source"] == "risk_scale_reference"
    assert benchmark["target_excess_return"] == pytest.approx(.005)
    assert benchmark["weights"] == dict(zip(ids, level["representative_weights"], strict=True))
    assert result["request"]["definition"]["benchmark"] is None  # User input stays simple.


def test_manual_cap_is_an_upper_bound_not_a_required_risk_band(reference):
    service, scale, _ = reference
    risk = request_for(scale).definition.risk_authorization.model_dump(mode="json")
    risk.update(mode="manual_level", selected_max_level=3)
    result = service.preview_mandate(request_for(scale, risk_authorization=risk))
    assert result["risk_decision"]["selected_max_level"] == 3
    assert result["reference_diagnosis"]["minimum_tested_feasible_level"] == 1
    assert result["reference_diagnosis"]["selected_candidate"]["volatility"] < scale["preview"]["result"]["levels"][2]["lower_bound"]


def test_risk_scale_is_independent_of_mandate_horizon(reference):
    service, scale, _ = reference
    result = service.preview_mandate(request_for(scale, horizon_years=5))
    assert result["reference_diagnosis"]["status"] == "validated"
    assert result["funding"]["monthly_cashflows"][-1]["month"] == 60
    assert result["definition"]["max_volatility"] is not None


def test_unmapped_scope_is_not_inferred_from_asset_names(reference):
    service, scale, _ = reference
    result = service.preview_mandate(request_for(scale, allocation_scope="a-different-universe"))
    assert result["reference_diagnosis"]["status"] == "awaiting_actual_scope"
    assert result["reference_diagnosis"]["candidates"] == []


def test_scale_hash_is_checked_before_computing(reference):
    service, scale, _ = reference
    body = request_for(scale).model_dump(mode="json")
    body["definition"]["risk_authorization"]["risk_scale_ref"]["content_hash"] = "0" * 64
    with pytest.raises(ConflictError):
        service.preview_mandate(MandateStudyRequest.model_validate(body))


def test_new_adapter_uses_declared_period_and_preserves_old_wrapper(reference):
    mu, vol, fee = .07, .16, .01
    drift, scale = goals.funding_monthly_parameters_kernel(mu, vol, fee, 1, 252)
    variance = np.log1p((vol / np.sqrt(252) / (1 + mu / 252)) ** 2)
    assert drift == pytest.approx((252 * (np.log1p(mu / 252) - variance / 2) + np.log1p(-fee)) / 12)
    assert scale == pytest.approx(np.sqrt(252 * variance / 12))
    draws = np.random.default_rng(37).normal(size=(24, 500, 1))
    flows = np.zeros(24)
    old = goals.funding_paths_kernel(draws, mu, vol, 100., flows, flows, 110., fee, .8, .2, 1.)
    d, s = goals.funding_monthly_parameters_kernel(mu, vol, fee, 0, 1)
    common = goals.funding_paths_from_monthly_kernel(draws, d, s, 100., flows, flows, 110., .8, .2, 1.)
    np.testing.assert_array_equal(old[0], common[0])
    np.testing.assert_array_equal(old[1], common[1])


def test_batched_search_shares_readonly_strided_draws_and_has_no_signature_growth(reference):
    raw = np.zeros((24, 1000, 1))
    draws = raw[::2, ::2]
    draws.flags.writeable = False
    flows = np.zeros(24)[::2]
    flows.flags.writeable = False
    metrics = np.array([[.02, .01], [.06, .15]])
    before = [tuple(k.signatures) for k in numeric.KERNELS]
    first = numeric.reference_funding_search_kernel(metrics, np.ones(2, dtype=np.int64), draws, 100., flows, flows, 90., 0., .8, 0, 1)
    second = numeric.reference_funding_search_kernel(metrics, np.ones(2, dtype=np.int64), draws.copy(), 100., flows.copy(), flows.copy(), 90., 0., .8, 0, 1)
    assert np.shares_memory(raw, draws)
    np.testing.assert_array_equal(first[0], second[0])
    assert first[1] == second[1] == 0
    assert before == [tuple(k.signatures) for k in numeric.KERNELS]
    assert numeric.execution_audit()["complete"] and numeric.execution_audit()["python_fallback"] == 0


def test_search_matches_independent_cashflow_reference_and_keeps_payment_failure(reference):
    draws = np.random.default_rng(503).normal(size=(12, 500, 1))
    metrics = np.array([[.03, .04], [.06, .12]])
    inflows, outflows = np.zeros(12), np.zeros(12)
    outflows[0], inflows[1] = 20., 20.
    result, _ = numeric.reference_funding_search_kernel(metrics, np.ones(2, dtype=np.int64), draws,
        100., inflows, outflows, 102., .005, .5, 0, 1)
    for i, (mean, volatility) in enumerate(metrics):
        variance = np.log1p((volatility / (1 + mean)) ** 2)
        drift = (np.log1p(mean) - variance / 2 + np.log1p(-.005)) / 12
        successes = 0
        for path in range(draws.shape[1]):
            wealth, paid = 100., True
            for month in range(12):
                available = wealth * np.exp(drift + np.sqrt(variance / 12) * draws[month, path, 0]) + inflows[month]
                paid = paid and available >= outflows[month]
                wealth = max(0., available - outflows[month])
            successes += paid and wealth >= 102.
        assert result[i, 0] == successes / draws.shape[1]
    outflows[0], inflows[1] = 1e6, 1e7
    failed, selected = numeric.reference_funding_search_kernel(metrics, np.ones(2, dtype=np.int64), draws,
        100., inflows, outflows, 102., .005, .5, 0, 1)
    assert selected == -1 and np.all(failed[:, 0] == 0)


def test_malformed_search_axes_and_nonfinite_samples_fail_even_without_candidates(reference):
    metrics, eligibility = np.array([[.02, .01]]), np.zeros(1, dtype=np.int64)
    flows = np.zeros(12)
    bad = np.zeros((12, 5, 1)); bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="MANDATE_SEARCH_INPUT"):
        numeric.reference_funding_search_kernel(metrics, eligibility, bad, 100., flows, flows, 90., 0., .8, 0, 1)
    with pytest.raises(ValueError, match="MANDATE_SEARCH_INPUT"):
        numeric.reference_funding_search_kernel(metrics, eligibility, np.zeros((0, 5, 1)), 100., flows, flows, 90., 0., .8, 0, 1)
    with pytest.raises(ValueError, match="MANDATE_CANDIDATE_AXIS"):
        numeric.minimum_reference_candidate_kernel(metrics, np.ones(2, dtype=np.int64))


@pytest.mark.parametrize("mean,vol,fee,method,periods", [(-1., .1, 0., 0, 1), (float("nan"), .1, 0., 0, 1),
    (.1, -.1, 0., 0, 1), (.1, .1, 1., 0, 1), (.1, .1, 0., 1, 0)])
def test_distribution_invalid_input_fails_closed(reference, mean, vol, fee, method, periods):
    with pytest.raises(ValueError):
        goals.funding_monthly_parameters_kernel(mean, vol, fee, method, periods)


def test_reachability_names_the_binding_bound(reference):
    """A target above what the cap can reach must say so, and say which level would."""
    service, scale, _ = reference
    caps = scale["preview"]["result"]["applied_boundaries"]
    absolute = dict(objective_kind="absolute_return", cash_budget=None, funding_target=None,
        risk_authorization={"mode": "manual_level", "source": "离线测试手动采用C3上限",
            "authorized_max_level": 3, "selected_max_level": 3,
            "risk_scale_ref": {k: scale[k] for k in ("id", "content_hash")}})
    modest = service.preview_mandate(request_for(scale, target_return=0.0, **absolute))["reference_diagnosis"]
    assert modest["reachability"]["binding"] == "none"
    headroom = modest["reachability"]["max_return_under_cap"]
    assert headroom is not None and modest["reachability"]["volatility_cap"] == pytest.approx(caps[2])

    greedy = service.preview_mandate(request_for(scale, target_return=headroom + .05, **absolute))["reference_diagnosis"]
    reach = greedy["reachability"]
    assert reach["binding"] in {"volatility_cap", "unreachable_at_any_level"}
    assert reach["max_return_under_cap"] == pytest.approx(headroom)
    if reach["binding"] == "volatility_cap":
        # Reaching the target costs more risk than the authorized cap allows.
        assert reach["min_volatility_for_target"] > reach["volatility_cap"]
        assert reach["required_risk_level"] is None or reach["required_risk_level"] >= 3


def test_reachability_kernel_is_warm_and_ignores_unsolved_rows():
    numeric.require_ready()
    metrics = np.array([[.03, .05, 0, 0, 0, 0, 0], [.06, .12, 0, 0, 0, 0, 0], [.09, .22, 0, 0, 0, 0, 0]])
    solved = np.array([1, 3, 5], dtype=np.int64)
    best, best_row, least, least_row = numeric.reference_reachability_kernel(metrics, solved, .13, .08)
    assert (best, best_row, least, least_row) == (.06, 1, .22, 2)
    assert numeric.reference_reachability_kernel(metrics, np.zeros(3, dtype=np.int64), .13, .08)[1] == -1
    assert numeric.reference_reachability_kernel(metrics, solved, .13, -np.inf)[3] == -1
    assert numeric.execution_audit()["python_fallback"] == 0

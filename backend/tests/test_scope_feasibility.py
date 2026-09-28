"""Early scope diagnostics exercise real evidence and continuous warmed solvers."""
import copy
from datetime import date, timedelta

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend import frontier_moments
from backend.strategic_allocation import compatibility_kernels, goal_kernels
from backend.strategic_allocation.contracts import MandateRequest
from backend.strategic_allocation.routes import build_router
from backend.strategic_allocation.scope_feasibility import diagnose
from backend.strategic_allocation.scope_feasibility_contracts import ScopeFeasibilityRequest
from backend.product_pools.repository import ProductPoolRepository
from backend.product_pools.service import ProductPoolService
from backend.tests.test_cma_scope_facts import scope_case
from backend.tests.test_ltcma_statistics import statistics_warm
from backend.tests.test_multi_cma import warm_multi
from backend.tests.test_strategic_allocation import workspace, warm, confirmed_mandate


@pytest.fixture(scope="module", autouse=True)
def screening_warm(tmp_path_factory):
    from backend.strategic_allocation import risk_scale_kernels, institution_kernels
    from backend.strategic_allocation.reference_sources import ReferenceSources
    frontier_moments.warm()
    compatibility_kernels.warm()
    risk_scale_kernels.warm()
    institution_kernels.warm()
    ReferenceSources(tmp_path_factory.mktemp("scope-source-warm")).warm()


def screening(scope_case, *, target=0., cap=.2, **kwargs):
    service, scope, _ = scope_case
    mandate = confirmed_mandate(service, MandateRequest(name="范围初筛目标", as_of=date.today(),
        review_date=date.today() + timedelta(days=90), target_return=target, max_volatility=cap,
        **kwargs))
    request = ScopeFeasibilityRequest(mandate_id=mandate["id"], as_of=date.today(),
        strategic_definition=scope["definition"], window={"kind": "common_since_inception"})
    return service, mandate, request


def test_frontier_target_and_continuous_candidate_are_read_only(scope_case):
    service, _, request = screening(scope_case)
    before = copy.deepcopy(service.artifacts.list())
    signatures = [tuple(kernel.signatures) for kernel in compatibility_kernels.KERNELS]
    app = FastAPI(); app.include_router(build_router(service))
    response = TestClient(app).post("/api/strategic-allocation/scope-feasibility", json=request.model_dump(mode="json"))
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["status"] == "feasible", result
    assert result["sample"]["observations"] == 160
    assert len(result["frontier"]["points"]) == 41
    assert result["frontier"]["complete"]
    assert result["target_check"]["candidate"]["expected_return"] >= 0.
    assert result["target_check"]["candidate"]["volatility"] <= .2 + 1e-8
    assert result["target_check"]["solver"]["search_domain"] == "continuous_convex_outer_approximation"
    assert service.artifacts.list() == before
    assert signatures == [tuple(kernel.signatures) for kernel in compatibility_kernels.KERNELS]
    assert result["reference_comparison"] == {"status": "not_linked"}


def test_original_reference_is_frozen_read_only_and_never_changes_screening(scope_case, monkeypatch):
    service, mandate, request = screening(scope_case, target=1.)
    baseline = diagnose(service, request)
    points = [{"volatility": .05, "expected_return": 2., "status": "optimal_to_tolerance"},
              {"volatility": None, "expected_return": None, "status": "iteration_limit"}]
    scale = service.artifacts.save("series", {"artifact_type": "risk_scale", "name": "目标原始标尺",
        "preview": {"request_echo": {"definition": {"base_currency": "CNY",
            "risk_basis_id": "annualized-periodic-volatility-v1", "research_as_of": str(request.as_of)}},
            "result": {"frontier": points, "parameter_evidence": {"data_quality": {
                "annualization_method": "arithmetic_mean_and_covariance_times_periods", "periods_per_year": 252,
                "intersection_start": "2019-01-01", "intersection_end": "2019-12-31"}}}}})
    ref = {key: scale[key] for key in ("id", "content_hash")}
    frozen = copy.deepcopy(mandate)
    frozen["definition"]["risk_authorization"] = {"risk_scale_ref": ref}
    frozen["assessment"]["reference_diagnosis"] = {"risk_scale_ref": ref, "constrained_frontier": points[:1]}
    monkeypatch.setattr(service, "_require_active_mandate", lambda _: copy.deepcopy(frozen))
    def no_refit(*args, **kwargs):
        raise AssertionError("reference display must not revalidate arrays, refit, or read today's default")
    monkeypatch.setattr(service.risk_scales, "get_version", no_refit)
    monkeypatch.setattr(service.risk_scales, "frozen_context", no_refit)
    before = copy.deepcopy(service.artifacts.list())
    result = diagnose(service, request)
    comparison = result.pop("reference_comparison")
    baseline.pop("reference_comparison")
    assert result == baseline  # Even the reference's high return cannot pass the current scope.
    assert comparison["status"] == "available"
    assert comparison["points"] == points
    assert comparison["constrained_points"] == points[:1]
    assert comparison["sample_start"] == "2019-01-01"
    assert comparison["risk_scale_ref"] == ref
    assert service.artifacts.list() == before

    from backend.strategic_allocation.scope_feasibility import _reference_comparison
    missing = copy.deepcopy(frozen)
    missing["assessment"]["reference_diagnosis"]["risk_scale_ref"] = {"id": "other", "content_hash": "other"}
    assert _reference_comparison(service, missing, request.as_of)["constrained_points"] == []
    broken = copy.deepcopy(frozen)
    broken["definition"]["risk_authorization"]["risk_scale_ref"]["content_hash"] = "wrong"
    assert _reference_comparison(service, broken, request.as_of) == {"status": "unavailable"}
    for field, value in (("base_currency", "USD"), ("research_as_of", "2999-01-01"), ("risk_basis_id", "other")):
        changed = copy.deepcopy(scale)
        changed["preview"]["request_echo"]["definition"][field] = value
        monkeypatch.setattr(service, "_get", lambda *args: changed)
        assert _reference_comparison(service, frozen, request.as_of) == {"status": "incompatible"}
    changed = copy.deepcopy(scale)
    changed["preview"]["result"]["parameter_evidence"]["data_quality"]["annualization_method"] = "cagr"
    assert _reference_comparison(service, frozen, request.as_of) == {"status": "incompatible"}
    monkeypatch.setattr(service, "_get", lambda *args: no_refit())
    with pytest.raises(AssertionError):  # Unexpected programming errors are not hidden.
        _reference_comparison(service, frozen, request.as_of)


def test_unreachable_return_uses_continuous_upper_bound_and_keeps_plot(scope_case):
    service, _, request = screening(scope_case, target=1.)
    result = diagnose(service, request)
    assert result["status"] == "infeasible", result
    assert result["reason_code"] == "SCOPE_RETURN_SHORTFALL"
    assert result["target_check"]["max_return_upper_bound"] < 1.
    assert result["frontier"]["points"]


def test_volatility_cap_can_make_scope_infeasible(scope_case):
    service, _, request = screening(scope_case, cap=.00001)
    result = diagnose(service, request)
    assert result["status"] == "infeasible", result
    assert result["reason_code"] == "SCOPE_CONSTRAINTS_INFEASIBLE"
    assert result["target_check"]["solver"]["phase_one_lower_bound"] > 0.


def test_insufficient_selected_window_does_not_become_infeasibility(scope_case):
    service, _, request = screening(scope_case)
    request = request.model_copy(update={"window": request.window.model_copy(update={"kind": "5Y"})})
    result = diagnose(service, request)
    assert result["status"] == "undetermined"
    assert result["reason_code"] in {"LTCMA_CALENDAR_COVERAGE", "LTCMA_PROXY_GAPS"}
    assert result["frontier"] is None


def test_unresolved_solver_does_not_become_infeasibility(scope_case, monkeypatch):
    service, _, request = screening(scope_case)
    monkeypatch.setattr("backend.strategic_allocation.scope_feasibility.solve",
        lambda *a, **k: {"weights": None, "status": "iteration_limit"})
    result = diagnose(service, request)
    assert result["status"] == "undetermined"
    assert result["reason_code"] == "SCOPE_SOLVER_UNRESOLVED"
    assert result["frontier"]["points"]


def test_missing_proxy_and_real_scope_change_are_unknown(scope_case):
    service, _, request = screening(scope_case)
    raw = request.model_dump(mode="json")
    raw["strategic_definition"]["assets"][0]["research_proxy"] = None
    result = diagnose(service, ScopeFeasibilityRequest.model_validate(raw))
    assert result["status"] == "undetermined"
    assert result["reason_code"] == "SCOPE_PROXY_REQUIRED"


def test_one_market_asset_has_valid_frontier_point(scope_case):
    service, _, request = screening(scope_case)
    raw = request.model_dump(mode="json")
    raw["strategic_definition"]["assets"] = raw["strategic_definition"]["assets"][:1]
    result = diagnose(service, ScopeFeasibilityRequest.model_validate(raw))
    assert result["status"] == "feasible", result
    assert len(result["target_check"]["candidate"]["weights"]) == 1


def test_benchmark_axis_mismatch_does_not_fabricate_target_point(scope_case, monkeypatch):
    service, mandate, request = screening(scope_case)
    frozen = copy.deepcopy(mandate)
    frozen["definition"].update(objective_kind="benchmark_relative", benchmark={"weights": {"other": 1.},
        "source": "risk_scale_reference", "target_excess_return": .02, "max_tracking_error": .05})
    monkeypatch.setattr(service, "_require_active_mandate", lambda identifier: frozen)
    result = diagnose(service, request)
    assert result["status"] == "undetermined"
    assert result["mandate"]["target_return"] is None
    assert result["reason_code"] == "SCOPE_BENCHMARK_AXIS"
    assert result["frontier"]["points"]


@pytest.mark.parametrize("terminal_target", [10000., 121000., 1e12])
def test_funding_goal_remains_unknown_after_mean_variance_check(scope_case, terminal_target):
    service, mandate, request = screening(scope_case, objective_kind="funding_goal", funding_plan={
        "total_capital": 100000., "terminal_target": terminal_target, "required_probability": .5,
        "liquidity_months": 12, "flows": []})
    result = diagnose(service, request)
    assert result["status"] == "undetermined", result
    assert result["reason_code"] == ("SCOPE_RETURN_CHECK_REQUIRED" if terminal_target == 1e12 else "SCOPE_FUNDING_CHECK_REQUIRED")
    assert result["target_check"]["status"] == "feasible"
    assert result["target_check"]["candidate"] is not None
    assert result["mandate"]["target_return"] is None
    projection = result["funding_comparison"]
    assert projection["probability_validated"] is False
    assert projection["basis"] == "annual_compound_median_gross_of_model_fee"
    assert projection["target_return"] == (result["mandate"]["funding_requirement"]["required_return"]
        if result["mandate"]["funding_requirement"]["status"] == "solved" else None)
    for raw, projected in zip(result["frontier"]["points"], projection["points"], strict=True):
        if raw["expected_return"] is not None:
            mean, vol = raw["expected_return"], raw["volatility"]
            daily_mean = mean / 252
            log_variance = np.log1p(vol ** 2 / 252 / (1 + daily_mean) ** 2)
            expected = np.expm1(252 * (np.log1p(daily_mean) - log_variance / 2))
            assert projected["expected_return"] == pytest.approx(expected, abs=1e-12)
            assert projected["volatility"] == vol
            assert projected["weights"] == raw["weights"]
    if terminal_target == 10000.:
        assert projection["status"] == "passed"
    elif terminal_target == 1e12:
        assert projection["status"] == ("no_candidate" if projection["target_return"] is not None else "unavailable")
    original = mandate["assessment"]["funding"]
    assert result["mandate"]["funding_requirement"] == {
        "required_return": original["cashflow_required_return"],
        "status": original["cashflow_required_return_status"],
        "basis": "annual_effective_gross_of_model_fee",
    }


def test_cash_floor_is_solved_and_zero_cash_variance_is_valid(scope_case):
    service, _, request = screening(scope_case, institutional_context={
        "investor_type": "personal", "purpose": "日常储备", "cash_reserve_weight": .3})
    raw = request.model_dump(mode="json")
    raw["strategic_definition"]["assets"].append({"id": "cash", "name": "现金", "currency": "CNY",
        "role": "liquidity", "liquidity": "liquid", "research_proxy": {"asset_type": "cash", "cash_return": .01}})
    result = diagnose(service, ScopeFeasibilityRequest.model_validate(raw))
    assert result["status"] == "feasible", result
    assert result["constraints"]["cash_floor"] == .3
    assert result["target_check"]["candidate"]["weights"]["cash"] >= .3 - 1e-7
    assert all(point["weights"]["cash"] >= .3 - 1e-7 for point in result["frontier"]["points"])


def test_required_cash_without_a_cash_asset_is_explicit(scope_case):
    service, _, request = screening(scope_case, institutional_context={
        "investor_type": "personal", "purpose": "日常储备", "cash_reserve_weight": .3})
    result = diagnose(service, request)
    assert result["status"] == "infeasible", result
    assert result["reason_code"] == "SAA_CASH_ASSETS_MISSING"
    assert result["frontier"]["constraints_applied"] is False


def product_version(service, *, cap=1., product_id="000300.SH"):
    repository = ProductPoolRepository(service.data.universe_dir / "product_pools.json")
    pool = repository.create_pool({"name": "初筛产品池"})
    _, version = repository.publish_pool(pool["id"], pool["revision"], {
        "effective_from": str(date.today()), "pool_name": "初筛产品池", "effective_to": None,
        "evaluation_plans": [{"plan_id": "equity", "plan_revision": 1, "plan_name": "权益", "as_of": str(date.today())}],
        "members": [{"key": "etf:" + product_id, "kind": "etf", "product_id": product_id, "code": "000300.SH",
            "name": "权益ETF", "research_status": "approved", "usage_status": "normal", "primary_plan_id": "equity", "max_weight": cap}]})
    return repository, version


def test_product_preview_reuses_save_assembly_and_obeys_product_cap(scope_case, monkeypatch):
    service, mandate, _ = screening(scope_case)
    repository, version = product_version(service, cap=.5)
    pools = ProductPoolService(repository, None, strategic_root=service.artifacts.root.parent.parent)
    fields = {"name": "产品范围", "research_date": str(date.today()), "version_ids": [version["id"]]}
    before = repository.store.path.read_bytes()
    preview = pools.preview_universe_snapshot(fields)
    assert repository.store.path.read_bytes() == before
    assert preview["products"][0]["max_weight"] == .5
    monkeypatch.setattr(service.cma.evidence.sources, "catalog", lambda **kwargs: {"items": [{
        "id": "etf:fund_daily:000300.SH", "code": "000300.SH", "reference_capability": {"available": True, "supported_fields": ["close_hfq"]}}]})
    request = ScopeFeasibilityRequest(mandate_id=mandate["id"], as_of=date.today(),
        product_version_ids=[version["id"]], window={"kind": "common_since_inception"})
    result = diagnose(service, request)
    assert result["status"] == "infeasible", result
    assert result["reason_code"] == "SCOPE_CONSTRAINTS_INFEASIBLE"
    assert repository.store.path.read_bytes() == before
    assert all(limit["max_weight"] == .5 for limit in result["constraints"]["asset_limits"].values())
    excluded = request.model_copy(update={"product_excluded_keys": ["etf:000300.SH"]})
    assert diagnose(service, excluded)["reason_code"] == "INVESTABLE_UNIVERSE_EMPTY"
    saved = pools.create_universe_snapshot(fields)
    assert all(saved[key] == value for key, value in preview.items())


def test_risk_reference_expiry_is_checked_before_history(scope_case, monkeypatch):
    service, mandate, request = screening(scope_case)
    frozen = copy.deepcopy(mandate)
    frozen["definition"]["risk_reference_valid_until"] = str(date.today())
    monkeypatch.setattr(service, "_require_active_mandate", lambda identifier: frozen)
    result = diagnose(service, request)
    assert result["reason_code"] == "SCOPE_MANDATE_DATE"
    assert result["status"] == "undetermined"
    assert result["frontier"] is None


def test_product_lookup_uses_code_and_keeps_liquidity_unknown(scope_case, monkeypatch):
    service, mandate, _ = screening(scope_case)
    _, version = product_version(service, product_id="opaque-product-id")
    seen = []
    def catalog(**kwargs):
        seen.append(kwargs["query"])
        return {"items": [{"id": "etf:fund_daily:000300.SH", "code": "000300.SH",
            "reference_capability": {"available": True, "supported_fields": ["close_hfq"]}}]}
    monkeypatch.setattr(service.cma.evidence.sources, "catalog", catalog)
    request = ScopeFeasibilityRequest(mandate_id=mandate["id"], as_of=date.today(),
        product_version_ids=[version["id"]], window={"kind": "common_since_inception"})
    result = diagnose(service, request)
    assert seen == ["000300.SH"]
    assert result["status"] == "undetermined", result
    assert result["reason_code"] == "SCOPE_PRODUCT_LIQUIDITY_REQUIRED"
    assert result["target_check"]["status"] == "feasible"
    assert result["frontier"]["constraints_applied"] is False


@pytest.mark.parametrize("mean,volatility", [(0., 0.), (-.03, .1), (.08, .2), (.05, 0.)])
def test_compound_projection_reuses_daily_moment_model_without_request_compilation(mean, volatility):
    kernel = goal_kernels.funding_compound_return_kernel
    signatures = tuple(kernel.signatures)
    rate = kernel(mean, volatility, 1, 252)
    drift, _ = goal_kernels.funding_monthly_parameters_kernel(mean, volatility, .02, 1, 252)
    # Planned fees already raise the required hurdle; the chart remains gross.
    assert (1 + rate) * .98 == pytest.approx(np.exp(12 * drift), abs=1e-12)
    assert tuple(kernel.signatures) == signatures
    assert len(kernel.nopython_signatures) == 1 and not kernel._can_compile


def test_funding_projection_cannot_pass_on_reference_or_unapplied_constraints():
    from backend.strategic_allocation.scope_feasibility import _funding_comparison
    raw = {"mandate": {"target_return": None, "volatility_cap": .06,
            "funding_requirement": {"status": "solved", "required_return": .04}},
        "frontier": {"constraints_applied": True, "points": [
            {"volatility": .03, "expected_return": .01, "weights": {"a": 1}, "status": "optimal_to_tolerance"},
            {"volatility": .1, "expected_return": .1, "weights": {"a": 1}, "status": "optimal_to_tolerance"}]},
        "reference_comparison": {"points": [{"volatility": .02, "expected_return": .2, "status": "optimal_to_tolerance"}], "constrained_points": []},
        "target_check": {"status": "feasible", "candidate": None}, "additional_checks": {"benchmark": False}}
    before = copy.deepcopy(raw)
    assert _funding_comparison(raw)["status"] == "no_candidate"
    assert raw == before
    raw["frontier"]["points"][0]["expected_return"] = .06
    assert _funding_comparison(raw)["status"] == "passed"
    raw["additional_checks"]["benchmark"] = True
    assert _funding_comparison(raw)["status"] == "no_candidate"
    raw["frontier"]["constraints_applied"] = False
    assert _funding_comparison(raw)["status"] == "unavailable"
    raw["frontier"]["constraints_applied"] = True
    raw["target_check"]["status"] = "undetermined"
    assert _funding_comparison(raw)["status"] == "unavailable"

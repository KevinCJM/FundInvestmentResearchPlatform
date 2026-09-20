"""Offline E2 numerics, immutable policy lifecycle and inherited TAA evidence."""
import copy
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError as InputError

from backend.custom_indicators.errors import ConflictError, ValidationError
from backend.strategic_allocation import cma_model_kernels, cma_statistical_kernels, multi_cma_kernels
from backend.strategic_allocation.cma_center_contracts import CmaCenterPublish, CmaRetire
from backend.strategic_allocation.contracts import CmaRequest, MandateRequest, PolicyRequest, PublishPolicyRequest
from backend.strategic_allocation.policy_gate import check_policy, require_policy_application
from backend.strategic_allocation import multi_cma
from backend.strategic_allocation.routes import build_router
from backend.tests.test_strategic_allocation import workspace, warm, definition, confirmed_mandate, saved_inputs


@pytest.fixture(scope="module", autouse=True)
def warm_multi():
    cma_model_kernels.warm()
    cma_statistical_kernels.warm()
    multi_cma_kernels.warm()


def publish_v2(service, key, *, means=(.07, .025), vol=(.18, .05), patch=None):
    raw = definition().model_dump(mode="json")
    raw.update(schema_version="2.0", name=key, moment_semantics="annualized_periodic_arithmetic",
               fee_basis="source_embedded_no_additional_fee", fx_hedging_basis="same_currency_no_conversion")
    for i, asset in enumerate(raw["assets"]):
        asset.update(annual_return=means[i], annual_volatility=vol[i])
    if patch:
        patch(raw)
    request = CmaRequest.model_validate(raw)
    preview = service.preview_cma(request)
    return service.publish_cma(CmaCenterPublish(request=request, preview_hash=preview["preview_hash"],
                                              confirm=True, idempotency_key="test-multi-" + key))


def setup(service, *, max_volatility=.2, funding=False):
    args = dict(name="融合政策目标", as_of=date.today(), review_date=date.today() + timedelta(days=90),
                target_return=0., max_volatility=max_volatility, max_tracking_error=.1)
    if funding:
        args.update(objective_kind="funding_goal", funding_plan=dict(total_capital=100000., terminal_target=10000.,
                    required_probability=.5, liquidity_months=12, flows=[]))
    mandate = confirmed_mandate(service, MandateRequest(**args))
    a = publish_v2(service, "first")
    b = publish_v2(service, "second", means=(.09, .015), vol=(.28, .07))
    request = PolicyRequest(mandate_id=mandate["id"], mode="parameter_average", candidate_count=300,
        cma_refs=[dict(cma_id=a["id"], content_hash=a["content_hash"], weight=.7),
                  dict(cma_id=b["id"], content_hash=b["content_hash"], weight=.3)])
    return mandate, [a, b], request


def adopt(service, request, preview):
    return service.publish_policy(PublishPolicyRequest(request=request, preview_hash=preview["preview_hash"],
        candidate_id="nominal-utility", name="融合长期政策", reason="依据冻结来源及其交叉评估采纳融合参数"))


def test_e2_uses_average_covariance_not_squared_weights_or_between_risk(workspace):
    service, _ = workspace
    mandate, sources, request = setup(service)
    preview = service.preview_policy(request)
    means = np.array([[a["annual_return"] for a in s["definition"]["assets"]] for s in sources])
    covs = np.array([s["covariance"] for s in sources])
    weights = np.array([.7, .3])
    expected_mean = weights @ means
    expected_cov = np.einsum("m,mij->ij", weights, covs)
    disagreement = np.einsum("m,mi,mj->ij", weights, means - expected_mean, means - expected_mean)
    multi = preview["multi_cma"]
    np.testing.assert_allclose(multi["effective_returns"], expected_mean)
    np.testing.assert_allclose(multi["effective_covariance"], expected_cov)
    np.testing.assert_allclose(multi["model_disagreement"], disagreement)
    assert not np.allclose(expected_cov, np.einsum("m,mij->ij", weights**2, covs))
    assert not np.allclose(expected_cov, expected_cov + disagreement)
    assert preview["cma_id"] is None and preview["cma_hash"] == multi["content_hash"]
    assert multi["uncertainty_status"] == "not_jointly_calibrated"
    assert multi["distribution_adapter"] == "annual_moment_proxy_approximation"
    for candidate in preview["candidates"]:
        w = np.array(list(candidate["weights"].values()))
        assert candidate["metrics"]["volatility"] == pytest.approx(np.sqrt(w @ expected_cov @ w))
        assert len(candidate["cross_model_results"]) == 2
        for row, source in zip(candidate["cross_model_results"], sources, strict=True):
            assert row["metrics"]["volatility"] == pytest.approx(np.sqrt(w @ source["covariance"] @ w))
            assert row["goal_check"] is None


def test_m_one_has_exact_single_candidate_and_funding_results(workspace):
    service, _ = workspace
    mandate, sources, request = setup(service, funding=True)
    source = sources[0]
    single = PolicyRequest(mandate_id=mandate["id"], cma_id=source["id"], candidate_count=300)
    many = request.model_copy(update={"cma_refs": [request.cma_refs[0].model_copy(update={"weight": 1.})]})
    old = service.preview_policy(single)
    new = service.preview_policy(many)
    assert "mode" not in old["request"] and "cma_refs" not in old["request"]
    assert "multi_cma" not in old
    for a, b in zip(old["candidates"], new["candidates"], strict=True):
        assert a == {k: v for k, v in b.items() if k != "cross_model_results"}
        assert b["cross_model_results"][0]["goal_check"] == a["goal_check"]
    assert new["funding_model"] == old["funding_model"]


def test_preview_no_writes_no_model_refit_and_policy_freezes_all_sources(workspace, monkeypatch):
    service, _ = workspace
    _, sources, request = setup(service)
    before = {str(p): p.read_bytes() for p in service.artifacts.root.parent.parent.rglob("*") if p.is_file()}
    import backend.strategic_allocation.cma_application as application
    monkeypatch.setattr(application, "evaluate_cma_model", lambda *a, **k: pytest.fail("frozen model refitted"))
    preview = service.preview_policy(request)
    after = {str(p): p.read_bytes() for p in service.artifacts.root.parent.parent.rglob("*") if p.is_file()}
    assert before == after
    saved = adopt(service, request, preview)
    assert saved["policy"]["cma_id"] is None
    assert saved["policy"]["schema_version"] == "2.0"
    assert saved["policy"]["mode"] == "parameter_average"
    assert [s["artifact"] for s in saved["policy"]["multi_cma"]["sources"]] == sources
    assert service.baselines.get_baseline(saved["id"]) == saved
    assert service.cma.list()["total"] == 2  # Derived moments are not an extra CMA.


def test_weight_change_and_source_hash_cannot_reuse_preview(workspace):
    service, _ = workspace
    _, _, request = setup(service)
    preview = service.preview_policy(request)
    changed = request.model_copy(update={"cma_refs": [ref.model_copy(update={"weight": .5}) for ref in request.cma_refs]})
    with pytest.raises(ConflictError, match="输入已变化"):
        adopt(service, changed, preview)
    changed = request.model_copy(update={"cma_refs": [request.cma_refs[0].model_copy(update={"content_hash": "0" * 64}), request.cma_refs[1]]})
    with pytest.raises(ConflictError, match="版本指纹"):
        service.preview_policy(changed)


def test_retirement_blocks_new_adoption_but_saved_taa_uses_frozen_sources(workspace):
    service, _ = workspace
    _, sources, request = setup(service)
    preview = service.preview_policy(request)
    saved = adopt(service, request, preview)
    source = sources[1]
    service.cma.retire(source["id"], CmaRetire(confirm=True, content_hash=source["content_hash"], reason="停止用于新研究"))
    with pytest.raises(ValidationError, match="停止新引用"):
        adopt(service, request, preview)
    weights = saved["policy"]["selection"]["weights"]
    check = check_policy(saved, weights, .1, str(date.today()))
    assert check["risk_evaluation_mode"] == "parameter_average"
    assert len(check["cross_model_results"]) == 2
    assert all(row["goal_check"] is None and row["expected_tracking_error"] == pytest.approx(0.)
               for row in check["cross_model_results"])


def test_source_failure_is_visible_diagnostic_not_fusion_gate(workspace):
    service, _ = workspace
    _, _, request = setup(service, max_volatility=.08)
    raw = request.model_dump(mode="json")
    raw["constraints"] = {"股票": {"min_weight": .3, "max_weight": .3}, "债券": {"min_weight": .7, "max_weight": .7}}
    request = PolicyRequest.model_validate(raw)
    preview = service.preview_policy(request)
    candidate = preview["candidates"][0]
    assert candidate["metrics"]["volatility"] < .08
    assert candidate["cross_model_results"][0]["within_limits"]
    assert not candidate["cross_model_results"][1]["within_limits"]
    saved = adopt(service, request, preview)
    check = check_policy(saved, candidate["weights"], .1, str(date.today()))
    assert check["within_limits"]
    assert not check["cross_model_results"][1]["within_limits"]


@pytest.mark.parametrize("field,value", [
    ("currency", "USD"), ("horizon_years", 5), ("moment_semantics", "one_year_simple"),
    ("fee_basis", "explicit_assumption"), ("fx_hedging_basis", "explicit_assumption"),
    ("as_of", str(date.today() - timedelta(days=1))),
])
def test_incompatible_basis_fails_closed(workspace, field, value):
    service, _ = workspace
    _, _, request = setup(service)
    second = publish_v2(service, "incompatible", patch=lambda raw: raw.update({field: value}))
    request.cma_refs[1] = request.cma_refs[1].model_copy(update={"cma_id": second["id"], "content_hash": second["content_hash"]})
    with pytest.raises(ValidationError, match="相同研究日"):
        service.preview_policy(request)


def test_legacy_manual_rejected_only_for_new_fusion(workspace):
    service, _ = workspace
    _, old, single = saved_inputs(service)
    assert len(service.preview_policy(single)["candidates"]) == 4
    request = PolicyRequest(mandate_id=single.mandate_id, mode="parameter_average", candidate_count=300,
        cma_refs=[dict(cma_id=old["id"], content_hash=old["content_hash"], weight=1.)])
    with pytest.raises(ValidationError, match="LTCMA 2.0"):
        service.preview_policy(request)


def test_current_source_change_blocks_any_original_cma(workspace):
    service, _ = workspace
    _, _, request = setup(service)
    path = service.data.data_dir / "asset_nv.parquet"
    frame = pd.read_parquet(path)
    frame.loc[0, "nv"] *= 1.01
    frame.to_parquet(path, index=False)
    with pytest.raises(ConflictError, match="来源"):
        service.preview_policy(request)


@pytest.mark.parametrize("tamper", [
    lambda p: p.pop("multi_cma"),
    lambda p: p.update(mode="unknown"),
    lambda p: p.pop("mode"),
    lambda p: [p.pop(key) for key in ("multi_cma", "mode", "schema_version")],
    lambda p: p["assumptions"]["assets"][0].update(annual_return=.9),
    lambda p: p["covariance"][0].__setitem__(0, .99),
    lambda p: p["multi_cma"]["sources"][0]["assumptions"]["assets"][0].update(annual_return=.9),
])
def test_taa_and_application_fail_closed_on_missing_or_tampered_fusion(workspace, tamper):
    service, _ = workspace
    _, _, request = setup(service)
    saved = adopt(service, request, service.preview_policy(request))
    tamper(saved["policy"])
    weights = saved["policy"]["selection"]["weights"]
    with pytest.raises(ValidationError):
        check_policy(saved, weights, .1, str(date.today()))
    with pytest.raises(ValidationError):
        require_policy_application(saved, weights, .1, str(date.today()))


@pytest.mark.parametrize("patch", [
    {"mode": "unknown"}, {"mode": "single", "cma_id": None}, {"cma_id": "another"},
    {"cma_refs": []}, {"cma_refs": [dict(cma_id="a", content_hash="a" * 64, weight=.2)]},
    {"cma_refs": [dict(cma_id="a", content_hash="a" * 64, weight=.5)] * 2},
    {"cma_refs": [dict(cma_id="a", content_hash="a" * 64, weight=True)]},
    {"cma_refs": [dict(cma_id="a", content_hash="a" * 64, weight=float("nan"))]},
    {"cma_refs": [dict(cma_id=str(i), content_hash="a" * 64, weight=1 / 21) for i in range(21)]},
])
def test_invalid_mode_reference_or_weight_contract(patch):
    raw = dict(mandate_id="m", mode="parameter_average", cma_refs=[dict(cma_id="a", content_hash="a" * 64, weight=1.)])
    with pytest.raises(InputError):
        PolicyRequest.model_validate({**raw, **patch})


def test_readonly_strided_inputs_share_memory_and_do_not_add_signatures():
    weights_storage = np.array([.25, 7., .75, 8.])
    mean_storage = np.arange(16., dtype=np.float64).reshape(4, 4) / 1000
    cov_storage = np.tile(np.eye(4)[None, :, :] * .03, (4, 1, 1))
    weights, means, covs = weights_storage[::2], mean_storage[::2, ::2], cov_storage[::2, ::2, ::2]
    widths = means[:, ::-1]
    for value in (weights, means, covs, widths):
        value.flags.writeable = False
    assert np.shares_memory(means, mean_storage) and np.shares_memory(covs, cov_storage)
    dispatchers = [cma_model_kernels.mixture_moments_kernel, multi_cma_kernels.weighted_half_width_kernel]
    signatures = [list(k.signatures) for k in dispatchers]
    snapshot = [value.copy() for value in (weights, means, covs, widths)]
    mean, _, within, between = dispatchers[0](weights, means, covs, False)
    half = dispatchers[1](weights, widths)
    np.testing.assert_allclose(mean, weights @ means)
    np.testing.assert_allclose(within, np.einsum("m,mij->ij", weights, covs))
    np.testing.assert_allclose(half, weights @ widths)
    for value, prior in zip((weights, means, covs, widths), snapshot, strict=True):
        np.testing.assert_array_equal(value, prior)
    assert [list(k.signatures) for k in dispatchers] == signatures
    assert all(len(k.nopython_signatures) == 1 and not k._can_compile for k in dispatchers)
    with pytest.raises(ValueError):
        multi_cma_kernels.weighted_half_width_kernel(np.empty(0), np.empty((0, 2)))
    with pytest.raises(ValueError):
        multi_cma_kernels.weighted_half_width_kernel(np.array([1.]), np.array([[np.inf]]))


def test_api_fusion_readiness_fails_without_request_warmup(workspace, monkeypatch):
    service, _ = workspace
    _, _, request = setup(service)
    app = FastAPI()
    app.include_router(build_router(service))
    monkeypatch.setattr(multi_cma_kernels, "_WARMED_PID", None)
    monkeypatch.setattr(multi_cma_kernels, "warm", lambda: pytest.fail("request must not warm"))
    response = TestClient(app).post("/api/strategic-allocation/policy/preview", json=request.model_dump(mode="json"))
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "SAA_NOT_READY"


def test_calculation_budget_counts_all_models_candidates_paths_and_months(workspace, monkeypatch):
    service, _ = workspace
    mandate, _, request = setup(service, funding=True)
    budget = multi_cma.require_calculation_budget(request, mandate["definition"], 2000)
    assert budget["diagnostic_path_months"] == 3 * 4 * 2000 * 120
    assert budget["maximum_central_and_conservative_path_months"] == 2 * budget["diagnostic_path_months"]
    monkeypatch.setattr(multi_cma, "MAX_DIAGNOSTIC_PATH_MONTHS", 1)
    from backend.strategic_allocation import kernels
    monkeypatch.setattr(kernels, "policy_candidates_with_budget_kernel", lambda *a: pytest.fail("over-budget search"))
    before = service.baselines.list_baselines()
    with pytest.raises(ValidationError, match="路径月计算预算"):
        service.preview_policy(request)
    assert service.baselines.list_baselines() == before


def test_retirement_after_preview_is_rechecked_under_publication_lock(workspace, monkeypatch):
    service, _ = workspace
    _, sources, request = setup(service)
    preview = service.preview_policy(request)
    original = service.preview_policy
    source = sources[-1]
    def retire_after_preview(body):
        result = original(body)
        service.cma.retire(source["id"], CmaRetire(confirm=True, content_hash=source["content_hash"], reason="并发停止新引用"))
        return result
    monkeypatch.setattr(service, "preview_policy", retire_after_preview)
    with pytest.raises(ValidationError, match="停止新引用"):
        adopt(service, request, preview)
    assert service.baselines.list_baselines() == []


def test_metadata_budget_rejects_legal_large_sources_before_any_policy_write(workspace):
    service, _ = workspace
    mandate = confirmed_mandate(service, MandateRequest(name="容量测试目标", as_of=date.today(),
        review_date=date.today() + timedelta(days=90), max_volatility=.3))
    def long_scenarios(raw):
        raw["moment_semantics"] = "one_year_simple"
        raw["model"] = dict(method="scenario_mixture", asset_ids=["股票", "债券"], as_of=str(date.today()), currency="CNY",
            return_basis="annual_arithmetic_total_return", source="显式年度情景", risk_mode="shared",
            shared_covariance=[[.0324, -.0009], [-.0009, .0025]], scenarios=[
                dict(id=f"scenario-{i}", probability=1 / 60, annual_returns={"股票": .07, "债券": .025}, source="研" * 2000)
                for i in range(60)])
    sources = [publish_v2(service, "large-" + str(i), patch=long_scenarios) for i in range(4)]
    request = PolicyRequest(mandate_id=mandate["id"], mode="parameter_average", candidate_count=300,
        cma_refs=[dict(cma_id=s["id"], content_hash=s["content_hash"], weight=.25) for s in sources])
    before = {str(p): p.stat().st_size for p in service.artifacts.root.parent.parent.rglob("*") if p.is_file()}
    with pytest.raises(ValidationError, match="7 MB"):
        service.preview_policy(request)
    assert service.baselines.list_baselines() == []
    assert all(service.get_cma(s["id"]) == s for s in sources)
    assert before == {str(p): p.stat().st_size for p in service.artifacts.root.parent.parent.rglob("*") if p.is_file()}


def test_final_envelope_size_guard_precedes_persistence(workspace, monkeypatch):
    service, _ = workspace
    _, _, request = setup(service)
    preview = service.preview_policy(request)
    original = multi_cma.require_payload_budget
    def reject_final(payload):
        if "policy" in payload:
            raise ValidationError("SAA_MULTI_CMA_PAYLOAD_BUDGET", "最终政策超过 7 MB 保存预算")
        return original(payload)
    monkeypatch.setattr(multi_cma, "require_payload_budget", reject_final)
    with pytest.raises(ValidationError, match="最终政策"):
        adopt(service, request, preview)
    assert service.baselines.list_baselines() == []


@pytest.mark.parametrize("mutate", [
    lambda s: s["definition"]["assets"][0].update(role="rates"),
    lambda s: s["definition"].update(alloc_name="同名资产另一个范围"),
    lambda s: s["source_snapshot"]["lineage"].update(config_hash="different-scope"),
    lambda s: s["semantics"].update(covariance_role="predictive"),
])
def test_shared_compatibility_rejects_role_scope_and_risk_semantic_drift(workspace, mutate):
    service, _ = workspace
    _, sources, _ = setup(service)
    changed = copy.deepcopy(sources)
    mutate(changed[1])
    with pytest.raises(ValidationError):
        multi_cma._compatibility(changed)


def test_pit_change_blocks_fusion_sources_after_initial_preview(workspace, monkeypatch):
    service, _ = workspace
    _, _, request = setup(service)
    service.preview_policy(request)
    monkeypatch.setattr(multi_cma, "automatic_research_day", lambda _: date.today() - timedelta(days=1))
    with pytest.raises(ValidationError, match="知识截止日"):
        service.preview_policy(request)

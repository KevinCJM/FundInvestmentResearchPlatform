"""New budget/authorization semantics and old-contract regression, offline."""
from copy import deepcopy
from datetime import date, timedelta
import numpy as np
import pytest
from pydantic import ValidationError as InputError

from backend.tests.test_strategic_allocation import workspace, warm, saved_inputs
from backend.strategic_allocation.contracts import MandateRequest, MandateStudyRequest, ConfirmMandateRequest, PolicyRequest, PublishPolicyRequest
from backend.strategic_allocation.policy_gate import check_policy
from backend.strategic_allocation.mandate_inputs import cash_success_required, effective_return_floor
from backend.strategic_allocation.planning import funding_inputs, diagnose_funding, require_goal_checks
from backend.custom_indicators.errors import ConflictError, ValidationError


def new_definition(**updates):
    today = date.today()
    result = {"schema_version": "2.0", "name": "有明确来源的研究目标", "as_of": str(today),
        "review_date": str(today + timedelta(days=60)), "currency": "CNY", "horizon_years": 10,
        "objective_kind": "absolute_return", "target_return": 0., "max_volatility": .2,
        "max_tracking_error": 0., "boundary_reason": "本次研究采用已确认的资金和风险边界",
        "boundary_policy": {"name": "研究资金政策", "source": "离线测试的明确政策来源，不是机构默认", "reviewed_on": str(today),
            "valid_until": str(today + timedelta(days=180)), "confirmed": True,
            "required_probability": .8, "liquidity_months": 12, "contribution_stress_ratio": .5,
            "cash_reserve_weight": 0.},
        "risk_authorization": {"mode": "explicit_numeric", "source": "研究员明确确认的数值授权"}}
    result.update(updates)
    return result


def budget(**updates):
    result = {"total_capital": 1000., "outside_reserve": 100., "balance_as_of": str(date.today()),
              "source": "测试余额与预算的明确来源", "amount_basis": "nominal", "inflation": .02,
              "annual_fee": 0., "flows": []}
    result.update(updates)
    return result


def payment(amount=180.):
    return {"name": "必要支付", "kind": "withdrawal", "amount": amount, "first_month": 1, "last_month": 1}


def stream(amount=10.):
    return {"name": "每月必要支付", "kind": "withdrawal", "amount": amount, "first_month": 1,
            "last_month": 120, "every_months": 1}


def test_compact_schema2_allows_optional_review_and_server_frozen_policy():
    data = new_definition(review_date=None, boundary_policy=None, boundary_reason="", min_cash_weight=.15,
        max_volatility=None, risk_authorization={"mode": "manual_level", "source": "risk_scale_selection",
            "authorized_max_level": 3, "selected_max_level": 3})
    value = MandateRequest.model_validate(data)
    assert value.review_date is None and value.boundary_policy is None
    assert value.min_cash_weight == pytest.approx(.15)


def test_legacy_schema_still_requires_review_date():
    data = new_definition(schema_version="1.0", review_date=None, cash_budget=None, boundary_policy=None,
                          risk_authorization=None, max_volatility=.15)
    with pytest.raises(InputError, match="复核日"):
        MandateRequest.model_validate(data)


def test_budget_is_not_the_success_standard_and_has_single_money_source(workspace):
    service, _ = workspace
    data = new_definition(cash_budget=budget(flows=[payment()]), cash_protection={"mode": "payments_only"})
    request = MandateStudyRequest(definition=MandateRequest.model_validate(data))
    before = deepcopy(request.model_dump(mode="json"))
    preview = service.preview_mandate(request)
    assert preview["funding"]["investable_capital"] == 900.
    assert preview["funding"]["required_liquid_weight"] == pytest.approx(.2)
    assert preview["funding"]["nominal_terminal_target"] == 0.
    assert preview["candidates"] == [] and preview["status"] == "inputs_only"
    assert request.model_dump(mode="json") == before
    frozen = service.confirm_mandate(ConfirmMandateRequest(request=request, preview_hash=preview["preview_hash"], acknowledge_limits=True))
    assert frozen["definition"]["cash_budget"]["total_capital"] == 1000.
    assert frozen["definition"]["funding_plan"] is None
    assert frozen["definition"]["boundary_policy_hash"]


def test_cashflows_can_define_a_cash_floor_without_becoming_a_hidden_success_probability(workspace):
    data = new_definition(boundary_policy=None, cash_budget=budget(flows=[payment()]), cash_protection=None)
    definition = MandateRequest.model_validate(data).model_dump(mode="json")
    prepared = funding_inputs(definition)
    assert not cash_success_required(definition)
    assert prepared[0]["required_liquid_weight"] > 0
    assert prepared[0]["required_effective_return"] is None


def test_absolute_return_cash_outflow_is_a_constraint_without_becoming_a_second_success_standard():
    definition = MandateRequest.model_validate(new_definition(boundary_policy=None,
        cash_budget=budget(flows=[payment(180.)]))).model_dump(mode="json")
    result, _, _ = funding_inputs(definition)
    assert not cash_success_required(definition)
    assert result["required_liquid_weight"] == pytest.approx(.2)
    assert result["required_effective_return"] is None
    assert result["nominal_terminal_target"] is None


def test_budget_without_cash_success_does_not_report_empty_one_hundred_percent(workspace):
    service, _ = workspace
    definition = MandateRequest.model_validate(new_definition(cash_budget=budget())).model_dump(mode="json")
    prepared = funding_inputs(definition)
    assert prepared[0]["required_effective_return"] is None
    assert prepared[0]["nominal_terminal_target"] is None
    candidate = {"id": "test", "metrics": {"expected_return": .04, "volatility": .1}}
    result = diagnose_funding(definition, [candidate], paths=500, seed=13)
    assert "goal_check" not in result["candidates"][0]
    assert result["funding"]["investable_capital"] == 900.


def test_real_cashflows_and_nominal_terminal_target_have_separate_inflation_bases():
    data = new_definition(objective_kind="funding_goal", cash_budget=budget(amount_basis="real", flows=[payment()]),
                          funding_target={"amount": 1200., "amount_basis": "nominal"})
    definition = MandateRequest.model_validate(data).model_dump(mode="json")
    result, _, outflows = funding_inputs(definition)
    assert result["nominal_terminal_target"] == 1200.
    assert outflows[0] == pytest.approx(180 * 1.02 ** (1 / 12))
    assert cash_success_required(definition)


def test_new_cash_floor_does_not_treat_tradable_equity_as_operating_cash(workspace):
    service, _ = workspace
    raw = new_definition(cash_budget=budget(flows=[payment()]), cash_protection={"mode": "payments_only"})
    resolved = service.preview_mandate(MandateStudyRequest(definition=MandateRequest.model_validate(raw)))["definition"]
    cma = {"alloc_name": "test", "assets": [
        {"id": "equity", "role": "growth", "liquidity": "liquid"},
        {"id": "cash", "role": "liquidity", "liquidity": "liquid"}]}
    groups, _ = service._constraints(PolicyRequest(mandate_id="x", cma_id="y"), cma, resolved)
    cash = next(g for g in groups if g["id"] == "policy-cash-reserve")
    assert cash["assets"] == ["cash"] and cash["lo"] == pytest.approx(.2)
    liquid = next(g for g in groups if g["id"] == "policy-liquid-reserve")
    assert liquid["lo"] == 0.
    cma["assets"][1]["role"] = "rates"
    with pytest.raises(ValidationError, match="现金"):
        service._constraints(PolicyRequest(mandate_id="x", cma_id="y"), cma, resolved)


def test_new_funding_goal_is_consumed_by_actual_cma_and_saa(workspace):
    service, _ = workspace
    _, cma, _ = saved_inputs(service)
    raw = new_definition(objective_kind="funding_goal", cash_budget=budget(),
                         funding_target={"amount": 300., "amount_basis": "nominal"})
    request = MandateStudyRequest(definition=MandateRequest.model_validate(raw), cma_id=cma["id"])
    preview = service.preview_mandate(request)
    assert preview["status"] == "diagnosed"
    assert all(c["goal_check"]["central"]["drawdown_alert_probability"] is None for c in preview["candidates"])
    saved = service.confirm_mandate(ConfirmMandateRequest(request=request, preview_hash=preview["preview_hash"], acknowledge_limits=True))
    result = service.preview_policy(PolicyRequest(mandate_id=saved["id"], cma_id=cma["id"]))
    assert result["funding"] == preview["funding"]
    assert all("goal_check" in c for c in result["candidates"])
    with pytest.raises(ValidationError):
        require_goal_checks(result["mandate"], [{"id": "missing"}])


@pytest.mark.parametrize("change", [
    lambda d: d.update(funding_plan={"total_capital": 100., "terminal_target": 200., "required_probability": .8}),
    lambda d: d.update(schema_version="1.0"),
    lambda d: d["boundary_policy"].update(confirmed=False),
    lambda d: d["boundary_policy"].update(confirmed=1),
    lambda d: d["boundary_policy"].update(required_probability=None),
    lambda d: d["boundary_policy"].update(liquidity_months=None),
    lambda d: d.update(risk_authorization={"mode": "manual_level", "authorized_max_level": 2, "selected_max_level": 3, "source": "不能放宽原授权"}),
    lambda d: d.update(risk_authorization={"mode": "manual_level", "authorized_max_level": 3, "selected_max_level": 2, "source": "不能另填数值cap"}),
    lambda d: d.update(funding_target=None),
    lambda d: d["cash_budget"].update(balance_as_of="2000-01-01"),
])
def test_new_contract_rejects_unconfirmed_conflicting_or_missing_evidence(change):
    data = new_definition(objective_kind="funding_goal", cash_budget=budget(), funding_target={"amount": 1200.})
    change(data)
    with pytest.raises(InputError):
        MandateRequest.model_validate(data)


def test_published_mandate_edit_and_delete_keep_history_but_only_list_active_version(workspace):
    service, _ = workspace
    first_request = MandateStudyRequest(definition=MandateRequest.model_validate(new_definition(name="第一版投资目标")))
    first_preview = service.preview_mandate(first_request)
    first = service.confirm_mandate(ConfirmMandateRequest(request=first_request,
        preview_hash=first_preview["preview_hash"], acknowledge_limits=True))
    assert first["id"] in {item["id"] for item in service.catalog()["mandates"]}

    second_request = MandateStudyRequest(definition=MandateRequest.model_validate(new_definition(name="修改后的投资目标")))
    second_preview = service.preview_mandate(second_request)
    second = service.confirm_mandate(ConfirmMandateRequest(request=second_request,
        preview_hash=second_preview["preview_hash"], acknowledge_limits=True,
        replaces_mandate_id=first["id"]))
    active = {item["id"] for item in service.catalog()["mandates"]}
    assert second["id"] in active and first["id"] not in active
    assert service.get_mandate(first["id"])["content_hash"] == first["content_hash"]
    with pytest.raises(ConflictError, match="已被新版本替代"):
        service.retire_mandate(first["id"])
    with pytest.raises(ConflictError, match="已被修改或删除"):
        service.confirm_mandate(ConfirmMandateRequest(request=second_request,
            preview_hash=second_preview["preview_hash"], acknowledge_limits=True,
            replaces_mandate_id=first["id"]))

    assert service.retire_mandate(second["id"]) == {"deleted": True, "id": second["id"]}
    assert second["id"] not in {item["id"] for item in service.catalog()["mandates"]}
    assert service.get_mandate(second["id"])["content_hash"] == second["content_hash"]
    assert service.retire_mandate(second["id"]) == {"deleted": True, "id": second["id"]}
    # Deleting the replacement must not resurrect the superseded first version.
    assert first["id"] not in {item["id"] for item in service.catalog()["mandates"]}


@pytest.mark.parametrize("action", ["retire", "replace"])
@pytest.mark.parametrize("during_publish", [False, True])
def test_inactive_mandate_cannot_create_policy_but_history_remains(workspace, monkeypatch, action, during_publish):
    service, _ = workspace
    mandate, _, request = saved_inputs(service)
    preview = service.preview_policy(request)
    publish = PublishPolicyRequest(request=request, preview_hash=preview["preview_hash"],
        candidate_id="minimum-risk", name="目标状态保护测试", reason="离线验证目标生命周期门禁")
    historical = service.publish_policy(publish)
    before = service.baselines.list_baselines()

    def deactivate():
        if action == "retire":
            service.retire_mandate(mandate["id"])
        else:
            study = MandateStudyRequest.model_validate(mandate["assessment"]["request"])
            value = service.preview_mandate(study)
            service.confirm_mandate(ConfirmMandateRequest(request=study, preview_hash=value["preview_hash"],
                acknowledge_limits=True, replaces_mandate_id=mandate["id"]))

    if during_publish:
        original = service.preview_policy
        def preview_then_deactivate(body):
            value = original(body)
            deactivate()
            return value
        monkeypatch.setattr(service, "preview_policy", preview_then_deactivate)
    else:
        deactivate()
        with pytest.raises(ConflictError, match="不能建立新政策"):
            service.preview_policy(request)
    with pytest.raises(ConflictError, match="不能建立新政策"):
        service.publish_policy(publish)
    assert service.baselines.list_baselines() == before
    assert service.baselines.get_baseline(historical["id"]) == historical
    assert service.get_mandate(mandate["id"])["content_hash"] == mandate["content_hash"]


def test_pending_scale_keeps_input_research_but_does_not_invent_cap(workspace):
    service, _ = workspace
    data = new_definition(max_volatility=None, risk_authorization={"mode": "manual_level", "authorized_max_level": 3,
                           "selected_max_level": 2, "source": "明确选择但标尺尚未准备"})
    request = MandateStudyRequest(definition=MandateRequest.model_validate(data))
    result = service.preview_mandate(request)
    assert result["risk_decision"]["status"] == "reference_pending"
    assert result["definition"]["max_volatility"] is None
    saved = service.confirm_mandate(ConfirmMandateRequest(request=request, preview_hash=result["preview_hash"], acknowledge_limits=True))
    _, cma, _ = saved_inputs(service)
    with pytest.raises(ValidationError, match="授权"):
        service.preview_policy(PolicyRequest(mandate_id=saved["id"], cma_id=cma["id"]))


def _new_policy_preview(service):
    service.warm()
    _, cma, _ = saved_inputs(service)
    raw = new_definition(objective_kind="funding_goal", max_tracking_error=.2,
                         cash_budget=budget(), funding_target={"amount": 300., "amount_basis": "nominal"})
    request = MandateStudyRequest(definition=MandateRequest.model_validate(raw), cma_id=cma["id"],
                                  simulation_paths=500, seed=17, validation_seed=29)
    preview = service.preview_mandate(request)
    mandate = service.confirm_mandate(ConfirmMandateRequest(request=request,
        preview_hash=preview["preview_hash"], acknowledge_limits=True))
    policy_request = PolicyRequest(mandate_id=mandate["id"], cma_id=cma["id"], candidate_count=300, seed=19)
    policy_preview = service.preview_policy(policy_request)
    return policy_request, policy_preview


def test_saa_adoption_uses_a_separate_validation_stream_and_freezes_evidence(workspace):
    service, _ = workspace
    request, preview = _new_policy_preview(service)
    baseline = service.publish_policy(PublishPolicyRequest(request=request, preview_hash=preview["preview_hash"],
        candidate_id="minimum-risk", name="新目标独立验证政策", reason="固定候选后使用独立资金验证"))
    validation = baseline["policy"]["funding_validation"]
    assert validation["seed"] == 29 ^ 0x9E3779B9
    assert validation["seed"] not in (17, 19, 29)
    assert validation["within_limits"] and validation["central"]["probability_lower"] >= .8
    weights = {a["id"]: a["base_weight"] for a in baseline["assets"]}
    checked = check_policy(baseline, weights, .2, baseline["as_of"])
    assert checked["cash_reserve_check"]["minimum"] == 0.
    assert checked["goal_diagnostic_scope"] == "strategic_plan_only_not_tactical_probability_guarantee"
    # Corrupt/missing frozen cash evidence cannot silently become zero.
    broken = deepcopy(baseline)
    del broken["policy"]["mandate"]["effective_cash_reserve_weight"]
    with pytest.raises(ValidationError, match="现金"):
        check_policy(broken, weights, .2, baseline["as_of"])
    protected = deepcopy(baseline)
    protected["policy"]["mandate"]["effective_cash_reserve_weight"] = .1
    assert not check_policy(protected, weights, .2, baseline["as_of"])["within_limits"]


def test_saa_does_not_save_a_policy_after_failed_independent_validation(workspace, monkeypatch):
    service, _ = workspace
    request, preview = _new_policy_preview(service)
    before = service.baselines.list_baselines()
    calls = []
    def rejected(*args):
        calls.append(args)
        return {"within_limits": False}
    monkeypatch.setattr("backend.strategic_allocation.service.validate_fixed_candidate", rejected)
    with pytest.raises(ValidationError, match="独立资金验证"):
        service.publish_policy(PublishPolicyRequest(request=request, preview_hash=preview["preview_hash"],
            candidate_id="minimum-risk", name="禁止绕过独立验证", reason="验证失败不能保存政策"))
    assert len(calls) == 1
    assert service.baselines.list_baselines() == before


@pytest.mark.parametrize("validation_seed", [11, 11 ^ 0x9E3779B9])
def test_search_and_validation_seeds_cannot_be_identical(validation_seed):
    with pytest.raises(InputError, match="不同随机种子"):
        MandateStudyRequest(definition=MandateRequest.model_validate(new_definition()), seed=11, validation_seed=validation_seed)


def test_saa_rejects_colliding_seeds_in_an_existing_frozen_mandate(workspace, monkeypatch):
    service, _ = workspace
    request, preview = _new_policy_preview(service)
    saved = deepcopy(service.get_mandate(request.mandate_id))
    saved["planning_settings"]["validation_seed"] = saved["planning_settings"]["seed"] ^ 0x9E3779B9
    monkeypatch.setattr(service, "get_mandate", lambda _: saved)
    before = service.baselines.list_baselines()
    with pytest.raises(ValidationError, match="样本重复"):
        service.publish_policy(PublishPolicyRequest(request=request, preview_hash=preview["preview_hash"],
            candidate_id="minimum-risk", name="重复样本不能发布", reason="审核独立验证样本边界"))
    assert service.baselines.list_baselines() == before


def test_cashflow_required_return_raises_the_stated_return_floor():
    """填了预期收益又填了现金流时，收益下限取两者较大的一个。"""
    # 900 investable capital against 1200 of scheduled payments: the flows themselves demand a return.
    data = new_definition(boundary_policy=None, target_return=.02, cash_budget=budget(flows=[stream()]))
    definition = MandateRequest.model_validate(data).model_dump(mode="json")
    result, _, _ = funding_inputs(definition)
    assert not cash_success_required(definition)
    # The success standard stays undefined; only the deterministic requirement survives.
    assert result["required_effective_return"] is None
    required = result["cashflow_required_return"]
    assert required is not None and required > .02
    assert effective_return_floor(definition, required) == pytest.approx(required)

    modest = MandateRequest.model_validate(new_definition(boundary_policy=None, target_return=.90,
        cash_budget=budget(flows=[stream()]))).model_dump(mode="json")
    assert effective_return_floor(modest, required) == pytest.approx(.90)


def test_terminal_floor_raises_the_required_return_above_a_bare_payment_plan():
    """期末至少保有把收益要求从"账户可归零"抬到"期末还剩这么多"。"""
    bare = MandateRequest.model_validate(new_definition(boundary_policy=None, target_return=.02,
        cash_budget=budget(flows=[stream()]))).model_dump(mode="json")
    protected = MandateRequest.model_validate(new_definition(boundary_policy=None, target_return=.02,
        cash_budget=budget(flows=[stream()]),
        cash_protection={"mode": "payments_and_terminal_floor",
                         "terminal_floor": {"amount": 500., "amount_basis": "nominal"}})).model_dump(mode="json")
    assert not cash_success_required(bare) and cash_success_required(protected)
    loose, _, _ = funding_inputs(bare)
    strict, _, _ = funding_inputs(protected)
    assert strict["cashflow_required_return"] > loose["cashflow_required_return"]
    assert strict["required_effective_return"] == pytest.approx(strict["cashflow_required_return"])
    assert (effective_return_floor(protected, strict["cashflow_required_return"])
            > effective_return_floor(bare, loose["cashflow_required_return"]))


def test_return_floor_only_applies_to_absolute_return_objectives():
    funding_goal = MandateRequest.model_validate(new_definition(boundary_policy=None, target_return=0.,
        objective_kind="funding_goal", cash_budget=budget(flows=[payment()]),
        funding_target={"amount": 500., "amount_basis": "nominal"})).model_dump(mode="json")
    assert effective_return_floor(funding_goal, .5) is None
    plain = MandateRequest.model_validate(new_definition(boundary_policy=None, target_return=.03)).model_dump(mode="json")
    assert effective_return_floor(plain, None) == pytest.approx(.03)


def test_preview_freezes_the_effective_return_floor_for_the_policy_gate(workspace):
    service, _ = workspace
    data = new_definition(target_return=.02, cash_budget=budget(flows=[stream()]))
    request = MandateStudyRequest(definition=MandateRequest.model_validate(data))
    preview = service.preview_mandate(request)
    floor = preview["definition"]["effective_target_return"]
    assert floor == pytest.approx(preview["funding"]["cashflow_required_return"])
    assert floor > .02


def test_funding_echo_is_the_same_arithmetic_as_the_preview(workspace):
    """填写页的实时回显与正式预览走同一条资金算术，只是不跑诊断。"""
    service, _ = workspace
    data = new_definition(target_return=.02, cash_budget=budget(flows=[stream()]))
    request = MandateStudyRequest(definition=MandateRequest.model_validate(data))
    echo, preview = service.mandate_funding(request), service.preview_mandate(request)
    assert echo["funding"]["cashflow_required_return"] == pytest.approx(preview["funding"]["cashflow_required_return"])
    assert echo["effective_target_return"] == pytest.approx(preview["definition"]["effective_target_return"])
    assert echo["effective_target_return"] > .02
    assert echo["execution"]["python_fallback"] == 0 and echo["execution"]["complete"]
    bare = MandateStudyRequest(definition=MandateRequest.model_validate(new_definition(target_return=.03)))
    assert service.mandate_funding(bare) == {"funding": None, "effective_target_return": pytest.approx(.03),
                                             "execution": echo["execution"]}


def test_contract_benchmark_text_is_evidence_only_and_cannot_linger_on_other_objectives():
    """合同基准原文只做留痕：相对目标可以记，其他目标不能残留一个不参与计算的基准。"""
    stated = "沪深300×60%＋中债综合财富×40%"
    relative = new_definition(objective_kind="benchmark_relative", target_excess_return=.03, stated_benchmark=stated)
    assert MandateRequest.model_validate(relative).stated_benchmark == stated
    with pytest.raises(InputError):
        MandateRequest.model_validate(new_definition(stated_benchmark=stated))
    # 留痕字段不得改变任何数值结论：带与不带原文，资金算术必须逐位一致。
    plain = {**relative}
    plain.pop("stated_benchmark")
    assert funding_inputs(relative) == funding_inputs(plain)


def test_retired_scale_blocks_current_policy_and_product_handoff_but_preserves_history(workspace, monkeypatch):
    from backend.tests.risk_scale_app import seed_sources
    from backend.tests.test_risk_scale_service import freeze_reference, definition as scale_definition, publish
    from backend.strategic_allocation.reference_sources import ReferenceSources
    from backend.strategic_allocation.risk_scale_contracts import RetireRequest
    from backend.strategic_allocation.policy_gate import require_policy_application
    from backend.tactical_allocation.portfolio_bridge import validate_decision_application, validate_allocation_source
    from backend.tactical_allocation.data import TacticalAllocationData

    service, _ = workspace
    monkeypatch.delenv('STRATEGIC_ALLOCATION_DATA_DIR', raising=False)
    service.warm()
    source_dir = service.data.data_dir / 'reference-fixture'
    inputs = seed_sources(source_dir)
    service.risk_scales.references.sources = ReferenceSources(source_dir)
    reference, _ = freeze_reference(service.risk_scales, inputs)
    scale, _, _ = publish(service.risk_scales, scale_definition(reference))
    _, cma, _ = saved_inputs(service)
    definition = new_definition(max_volatility=None, risk_authorization={
        'mode': 'manual_level', 'authorized_max_level': 5, 'selected_max_level': 5,
        'risk_scale_ref': {k: scale[k] for k in ('id', 'content_hash')}})
    study = MandateStudyRequest(definition=MandateRequest.model_validate(definition))
    preview = service.preview_mandate(study)
    mandate = service.confirm_mandate(ConfirmMandateRequest(request=study, preview_hash=preview['preview_hash'], acknowledge_limits=True))
    policy_request = PolicyRequest(mandate_id=mandate['id'], cma_id=cma['id'])
    preview = service.preview_policy(policy_request)
    baseline = service.publish_policy(PublishPolicyRequest(request=policy_request, preview_hash=preview['preview_hash'],
        candidate_id='minimum-risk', name='冻结风险标尺政策', reason='离线验证标尺治理应用门禁'))
    weights = {a['id']: a['base_weight'] for a in baseline['assets']}
    root = service.artifacts.root.parent.parent
    context = {'workspace': root, 'data_dir': service.data.data_dir}
    before = check_policy(baseline, weights, 0., baseline['as_of'], **context)
    assert before['current_application_eligible'], before
    service.risk_scales.retire(scale['id'], RetireRequest(confirm=True, expected_revision=0, reason='风险标尺已不再适用'))
    after = check_policy(baseline, weights, 0., baseline['as_of'], **context)
    assert after['within_limits'] and not after['current_application_eligible']
    assert any('退休' in text for text in after['risk_scale_blockers'])
    assert service.baselines.get_baseline(baseline['id']) == baseline
    with pytest.raises(ValidationError) as error:
        require_policy_application(baseline, weights, 0., baseline['as_of'], **context)
    assert error.value.code == 'SAA_RISK_SCALE_INELIGIBLE'
    # Both the TAA export and a direct portfolio request recheck mutable governance.
    decision = service.baselines.save_decision({'preview': {'baseline': baseline,
        'request': {'as_of': baseline['as_of'], 'max_tracking_error': 0.},
        'recommendation': {'weights': weights}}}, arrays={'returns': np.zeros((2, 2))})
    data = TacticalAllocationData(service.data.data_dir, universe_dir=root)
    with pytest.raises(ValidationError) as error:
        validate_decision_application(decision, data)
    assert error.value.code == 'SAA_RISK_SCALE_INELIGIBLE'
    monkeypatch.setenv('TACTICAL_ALLOCATION_DATA_DIR', str(root))
    with pytest.raises(ValidationError) as error:
        validate_allocation_source({'kind': 'taa', 'decision_id': decision['id']}, [], {}, '', root)
    assert error.value.code == 'SAA_RISK_SCALE_INELIGIBLE'
    # Missing context or changed frozen identity must not grant current permission.
    assert not check_policy(baseline, weights, 0., baseline['as_of'])['current_application_eligible']
    bad = deepcopy(baseline)
    bad['policy']['mandate']['risk_authorization']['risk_scale_ref']['content_hash'] = '0' * 64
    assert '指纹' in check_policy(bad, weights, 0., baseline['as_of'], **context)['risk_scale_blockers'][0]

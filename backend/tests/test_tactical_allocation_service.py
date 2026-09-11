"""Real Parquet -> transient preview -> immutable decision -> budget handoff."""
from datetime import date, timedelta
import json

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.custom_indicators.errors import ConflictError, ValidationError
from backend.tactical_allocation.contracts import PreviewRequest, SaveDecisionRequest, ScenarioRequest
from backend.tactical_allocation.service import TacticalAllocationService
from backend.tactical_allocation.portfolio_bridge import validate_allocation_source


@pytest.fixture(scope="module", autouse=True)
def warm():
    from backend.tactical_allocation.numeric import warm_tactical_allocation_kernels
    from backend.tactical_allocation.data import warm_tactical_data
    warm_tactical_data()
    assert warm_tactical_allocation_kernels()["complete"]


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("TACTICAL_ALLOCATION_DATA_DIR", str(tmp_path))
    end = date.today()
    days = pd.date_range(end=end, periods=161)
    info = [{"asset_alloc_name": "配置", "asset_name": key, "etf_code": code,
             "etf_name": key + "ETF", "etf_weight": 100., "creat_time": days[0],
             "as_of": None, "universe_snapshot_id": None, "data_release_id": None}
            for key, code in [("股票", "510300.SH"), ("债券", "511010.SH")]]
    pd.DataFrame(info).to_parquet(tmp_path / "asset_alloc_info.parquet", index=False)
    nav = [{"asset_alloc_name": "配置", "asset_name": key, "date": day,
            "nv": float((1 + rate) ** index), "available_at": day, "as_of": None}
           for key, rate in [("股票", .0015), ("债券", .0002)]
           for index, day in enumerate(days)]
    pd.DataFrame(nav).to_parquet(tmp_path / "asset_nv.parquet", index=False)
    service = TacticalAllocationService(tmp_path, tmp_path)
    baseline = service.create_baseline({"alloc_name": "配置", "name": "SAA 60/40", "as_of": str(end),
                                        "weights": {"股票": .6, "债券": .4},
                                        "group_limits": [{"id": "风险组", "assets": ["股票"], "lo": .3, "hi": .68}]})
    request = PreviewRequest(baseline_id=baseline["id"], start_date=days[0].date(), end_date=end,
                             as_of=end, train_end_date=days[100].date(), lookback=20,
                             max_abs_tilt=.1, transaction_cost_bps=10)
    return service, baseline, request


def test_preview_uses_real_returns_and_is_transient_and_deterministic(workspace):
    service, baseline, request = workspace
    result = service.preview(request)
    assert result["preview_hash"] == service.preview(request)["preview_hash"]
    assert service.repository.list_decisions() == []
    assert result["data"]["observations"] == 160
    assert result["data"]["train_observations"] == 100
    assert result["data"]["validation_observations"] == 60
    assert result["data"]["pit"]["status"] == "research_only"
    assert result["audit"]["formal_pit_eligible"] is False
    assert result["audit"]["selection"]["holdout_used_for_selection"] is False
    assert result["recommendation"]["trade_deltas"] is None
    assert {x["segment"] for x in result["chart"]} == {"train", "validation"}
    assert len(result["candidates"]) == 7
    assert sum(result["recommendation"]["weights"].values()) == pytest.approx(1)
    assert result["recommendation"]["weights"]["股票"] <= .68 + 1e-8


def test_save_recomputes_and_freezes_inputs_and_refuses_stale_hash(workspace):
    service, _, request = workspace
    preview = service.preview(request)
    with pytest.raises(ConflictError, match="重新预览"):
        service.save_decision(SaveDecisionRequest(request=request, preview_hash="0" * 64, name="不应保存"))
    assert service.repository.list_decisions() == []
    decision = service.save_decision(SaveDecisionRequest(request=request, preview_hash=preview["preview_hash"], name="本次研究"))
    frozen = service.repository.decision_arrays(decision["id"])
    assert frozen["returns"].shape == (160, 2)
    assert not frozen["returns"].flags.writeable
    assert decision["preview"] == preview
    assert service.repository.get_decision(decision["id"])["content_hash"] == decision["content_hash"]
    with pytest.raises(ValidationError, match="可投资域"):
        service.product_allocation(decision["id"])


def test_zero_deviation_and_shock_cost_contributions_reconcile(workspace):
    service, _, request = workspace
    zero = request.model_copy(update={"selected_candidate_id": "scale-0"})
    result = service.preview(zero)
    assert result["selected_id"] == "scale-0"
    assert result["recommendation"]["weights"] == {"股票": .6, "债券": .4}
    assert all(row["taa"] == row["baseline"] for row in result["chart"])
    stress = service.scenario(ScenarioRequest(preview_request=request,
                    scenario={"kind": "shock", "name": "股跌债涨", "shocks": {"股票": -.2, "债券": .03}}))
    assert stress["baseline_return"] == pytest.approx(sum(x["baseline"] for x in stress["contributions"]) - stress["cost"]["baseline"])
    assert stress["taa_return"] == pytest.approx(sum(x["taa"] for x in stress["contributions"]) - stress["cost"]["taa"])
    with pytest.raises(ValidationError, match="逐项覆盖"):
        service.scenario(ScenarioRequest(preview_request=request, scenario={"kind": "shock", "shocks": {"股票": -.2}}))
    historical = service.scenario(ScenarioRequest(preview_request=request,
                 scenario={"kind": "historical", "start_date": str(request.start_date),
                           "end_date": str(request.train_end_date), "name": "历史重演"}))
    assert historical["kind"] == "historical"
    assert service.repository.list_decisions() == []


def test_manual_funding_and_current_holdings_are_validated(workspace):
    service, _, request = workspace
    with pytest.raises(ValidationError, match="合计必须为 0"):
        service.preview(request.model_copy(update={"signal_mode": "manual", "manual_tilts": {"股票": .1, "债券": 0}}))
    manual = request.model_copy(update={"signal_mode": "manual", "manual_tilts": {"股票": .1, "债券": -.1},
                                        "current_weights": {"股票": .55, "债券": .45}})
    result = service.preview(manual)
    assert sum(result["recommendation"]["trade_deltas"].values()) == pytest.approx(0)
    assert any("人工观点" in item for item in result["warnings"])


def test_regime_uses_shared_gate_and_recognition_dates(workspace):
    service, _, request = workspace
    first = str(request.start_date)
    recognized = str(request.start_date + timedelta(days=30))
    calls = []
    def resolver(identifier):
        calls.append(identifier)
        return ({"states": [{"id": "risk_on"}], "series": [{"observation_date": first,
                 "recognized_at": recognized, "effective_date": first,
                 "probabilities": {"risk_on": 1.}, "confidence": 1.}]}, {"passed": True})
    service.regime_resolver = resolver
    body = request.model_copy(update={"signal_mode": "regime", "regime_run_id": "published-v2",
                                     "state_tilts": {"risk_on": {"股票": .1, "债券": -.1}},
                                     "max_signal_age_days": 365})
    result = service.preview(body)
    assert calls == ["published-v2"]
    assert result["audit"]["signal"]["gate"]["passed"]
    assert all(row["weights"] == {"股票": .6, "债券": .4} for row in result["weight_path"][:30])
    def blocked(_):
        raise ValidationError("TAA_RUN_NOT_PUBLISHED", "状态未发布")
    service.regime_resolver = blocked
    with pytest.raises(ValidationError, match="未发布"):
        service.preview(body)


def test_product_handoff_conserves_class_budgets_and_rejects_tampering(workspace, monkeypatch):
    service, baseline, request = workspace
    path = service.data.data_dir / "asset_alloc_info.parquet"
    info = pd.read_parquet(path)
    info["universe_snapshot_id"] = "universe-test"
    info.to_parquet(path, index=False)
    universe = {"id": "universe-test", "name": "测试域", "research_date": str(request.as_of),
                "created_at": str(request.as_of), "immutable": True,
                "members": [{"kind": "etf", "product_id": code, "name": code, "eligible": True}
                            for code in ["510300.SH", "511010.SH"]]}
    (service.data.data_dir / "product_pools.json").write_text(json.dumps({"pools": [], "versions": [], "universe_snapshots": [universe]}))
    baseline = service.create_baseline({"alloc_name": "配置", "name": "可应用 SAA", "as_of": str(request.as_of), "weights": {"股票": .6, "债券": .4}})
    request = request.model_copy(update={"baseline_id": baseline["id"]})
    preview = service.preview(request)
    decision = service.save_decision(SaveDecisionRequest(request=request, preview_hash=preview["preview_hash"], name="预算"))
    payload = service.product_allocation(decision["id"])
    assert payload["method"] == "manual"
    assert sum(x["weight"] for x in payload["constituents"]) == pytest.approx(100)
    strategy = {"type": "manual", "weights": [x["weight"] / 100 for x in payload["constituents"]]}
    reference = validate_allocation_source(payload["allocation_source"], payload["constituents"], strategy,
                                          "universe-test", service.data.data_dir)
    assert reference["decision_hash"] == decision["content_hash"]
    strategy["weights"] = [.1, .9]
    with pytest.raises(ValidationError, match="预算不一致"):
        validate_allocation_source(payload["allocation_source"], payload["constituents"], strategy,
                                   "universe-test", service.data.data_dir)


def test_known_late_training_returns_cannot_select_a_validation_policy(workspace):
    service, _, request = workspace
    path = service.data.data_dir / "asset_nv.parquet"
    frame = pd.read_parquet(path)
    frame.loc[frame["date"] == str(request.train_end_date), "available_at"] = pd.Timestamp(request.train_end_date + timedelta(days=10))
    frame.to_parquet(path, index=False)
    baseline = service.create_baseline({"alloc_name": "配置", "name": "晚公布标签", "as_of": str(request.as_of), "weights": {"股票": .6, "债券": .4}})
    request = request.model_copy(update={"baseline_id": baseline["id"]})
    with pytest.raises(ValidationError, match="尚不可得"):
        service.preview(request)


def test_api_contracts_reject_invalid_dates_and_ignore_no_client_gate(workspace, monkeypatch):
    monkeypatch.setenv("CUSTOM_INDICATOR_DATA_DIR", str(workspace[0].data.data_dir / "indicators"))
    monkeypatch.setenv("HISTORICAL_REGIME_DATA_DIR", str(workspace[0].data.data_dir / "regimes"))
    from services import tactical_allocation_routes as routes
    service, _, request = workspace
    monkeypatch.setattr(routes, "tactical_service", service)
    app = FastAPI()
    app.include_router(routes.router)
    with TestClient(app) as client:
        body = request.model_dump(mode="json")
        assert client.post("/api/tactical-allocation/preview", json={**body, "formal_pit_eligible": True}).status_code == 422
        assert client.post("/api/tactical-allocation/preview", json={**body, "train_end_date": body["end_date"]}).status_code == 422
        response = client.post("/api/tactical-allocation/preview", json=body)
        assert response.status_code == 200, response.text
        assert response.json()["execution"]["python_fallback"] == 0
        assert client.get("/api/tactical-allocation/catalog").status_code == 200
        checked = client.post("/api/tactical-allocation/preflight", json=body)
        assert checked.status_code == 200
        assert checked.json()["can_calculate"] is True


def test_preflight_explains_immature_labels_and_does_not_switch_search(workspace):
    service, _, request = workspace
    path = service.data.data_dir / 'asset_nv.parquet'
    frame = pd.read_parquet(path)
    frame['available_at'] = pd.Timestamp(request.as_of)
    frame.to_parquet(path, index=False)
    baseline = service.create_baseline({'alloc_name': '配置', 'name': '当前才可得', 'as_of': str(request.as_of), 'weights': {'股票': .6, '债券': .4}})
    request = request.model_copy(update={'baseline_id': baseline['id']})
    check = service.preflight(request)
    assert not check['can_calculate'] and not check['training']['eligible']
    assert check['training']['unavailable_count'] == 200
    assert check['training']['earliest_available_date'] == str(request.as_of)
    assert [x['action'] for x in check['guidance']] == ['fixed_comparison']
    assert request.search is True
    assert service.preflight(request.model_copy(update={'search': False}))['can_calculate']


def test_preflight_quality_blocks_compute_and_cannot_be_waived(workspace):
    service, _, request = workspace
    path = service.data.data_dir / 'asset_nv.parquet'
    frame = pd.read_parquet(path)
    frame.loc[(frame['asset_name'] == '债券') & (frame['date'] > str(request.train_end_date)), 'nv'] *= .01
    frame.to_parquet(path, index=False)
    baseline = service.create_baseline({'alloc_name': '配置', 'name': '尺度错误', 'as_of': str(request.as_of), 'weights': {'股票': .6, '债券': .4}})
    request = request.model_copy(update={'baseline_id': baseline['id'], 'search': False})
    check = service.preflight(request)
    assert check['quality']['status'] == 'blocked' and not check['can_calculate']
    assert check['quality']['issues'][0]['asset_id'] == '债券'
    with pytest.raises(ValidationError, match='尺度断点'):
        service.preview(request)


def test_save_multiple_scenarios_recomputes_freezes_and_restores(workspace):
    service, _, request = workspace
    preview = service.preview(request)
    scenarios = [{'kind': 'shock', 'name': '股跌债涨', 'shocks': {'股票': -.2, '债券': .03}},
                 {'kind': 'historical', 'name': '训练期压力', 'start_date': str(request.start_date), 'end_date': str(request.train_end_date)}]
    body = SaveDecisionRequest(request=request, preview_hash=preview['preview_hash'], name='含情景研究', scenarios=scenarios)
    result = service.save_decision(body)
    restored = service.repository.get_decision(result['id'])
    assert [item['scenario']['name'] for item in restored['scenarios']] == ['股跌债涨', '训练期压力']
    expected = service.scenario(ScenarioRequest(preview_request=request, scenario=scenarios[0]))
    assert restored['scenarios'][0]['result'] == expected
    arrays = service.repository.decision_arrays(result['id'])
    assert arrays['scenario_1_returns'].shape == (100, 2)
    assert not arrays['scenario_1_returns'].flags.writeable
    assert restored['scenarios'][1]['result']['evidence']['dates'][-1] == str(request.train_end_date)
    with pytest.raises(Exception):
        SaveDecisionRequest(**{**body.model_dump(), 'scenarios': [{**scenarios[0], 'taa_return': 100}]})
    old_hash = restored['content_hash']
    service.save_decision(body.model_copy(update={'name': '另一个版本', 'scenarios': []}))
    assert service.repository.get_decision(result['id'])['content_hash'] == old_hash


@pytest.mark.parametrize('name', ['', '   ', '\n\t', '\u3000'])
def test_scenario_names_reject_blank_before_calculation_or_persistence(workspace, name):
    from pydantic import ValidationError as ContractValidationError
    service, _, request = workspace
    scenario = {'kind': 'shock', 'name': name, 'shocks': {'股票': -.2, '债券': .03}}
    with pytest.raises(ContractValidationError):
        ScenarioRequest(preview_request=request, scenario=scenario)
    with pytest.raises(ContractValidationError):
        SaveDecisionRequest(request=request, preview_hash='0' * 64, name='研究', scenarios=[scenario])
    assert service.repository.list_decisions() == []


@pytest.mark.parametrize('second_name', ['衰退压力', '  衰退压力  ', '\u3000衰退压力\u3000'])
def test_save_rejects_duplicate_normalized_scenario_names(workspace, second_name):
    from pydantic import ValidationError as ContractValidationError
    service, _, request = workspace
    with pytest.raises(ContractValidationError, match='情景名称不能重复'):
        SaveDecisionRequest(request=request, preview_hash='0' * 64, name='研究', scenarios=[
            {'kind': 'shock', 'name': '衰退压力', 'shocks': {'股票': -.2, '债券': .03}},
            {'kind': 'historical', 'name': second_name, 'start_date': request.start_date, 'end_date': request.train_end_date},
        ])
    assert service.repository.list_decisions() == []


@pytest.mark.parametrize('side', ['before_start', 'after_end'])
def test_historical_scenario_cannot_read_outside_preview_frozen_coverage(workspace, monkeypatch, side):
    service, _, original = workspace
    request = original.model_copy(update={
        'start_date': original.start_date + timedelta(days=10),
        'end_date': original.end_date - timedelta(days=10),
    })
    preview = service.preview(request)
    scenario = {'kind': 'historical', 'name': '范围外压力',
                'start_date': original.start_date if side == 'before_start' else request.start_date,
                'end_date': original.end_date if side == 'after_end' else request.end_date}
    # The full source file includes both periods; an extra load would silently
    # evaluate a different input population from the confirmed preview.
    loads = []
    original_load = service.data.load_data
    def tracked_load(baseline, start_date, end_date, as_of):
        loads.append((start_date, end_date, as_of))
        return original_load(baseline, start_date, end_date, as_of)
    monkeypatch.setattr(service.data, 'load_data', tracked_load)
    with pytest.raises(ValidationError) as error:
        service.scenario(ScenarioRequest(preview_request=request, scenario=scenario))
    assert error.value.code == 'TAA_SCENARIO_COVERAGE'
    assert str(request.start_date) in str(error.value)
    assert str(request.end_date) in str(error.value)
    assert loads == [(str(request.start_date), str(request.end_date), str(request.as_of))]
    loads.clear()
    with pytest.raises(ValidationError) as save_error:
        service.save_decision(SaveDecisionRequest(
            request=request, preview_hash=preview['preview_hash'], name='不应部分保存',
            scenarios=[{'kind': 'shock', 'name': '合法冲击', 'shocks': {'股票': -.1, '债券': 0}}, scenario],
        ))
    assert save_error.value.code == 'TAA_SCENARIO_COVERAGE'
    assert loads == [(str(request.start_date), str(request.end_date), str(request.as_of))]
    assert service.repository.list_decisions() == []


def test_historical_scenario_exact_frozen_edges_freeze_same_returns(workspace):
    service, _, request = workspace
    preview = service.preview(request)
    saved = service.save_decision(SaveDecisionRequest(
        request=request, preview_hash=preview['preview_hash'], name='边界区间',
        scenarios=[{'kind': 'historical', 'name': '  整段回放  ', 'start_date': request.start_date, 'end_date': request.end_date}],
    ))
    arrays = service.repository.decision_arrays(saved['id'])
    np.testing.assert_array_equal(arrays['scenario_0_returns'], arrays['returns'])
    np.testing.assert_array_equal(arrays['scenario_0_available_days'], arrays['available_days'])
    assert saved['scenarios'][0]['scenario']['name'] == '整段回放'
    assert saved['scenarios'][0]['result']['name'] == '整段回放'
    assert saved['scenarios'][0]['result']['evidence']['dates'] == [row['date'] for row in preview['weight_path']]


@pytest.mark.parametrize("condition", ["unknown", "warmup", "neutral"])
def test_no_effective_training_signal_blocks_search_but_explicit_fixed_remains(workspace, condition):
    service, _, request = workspace
    if condition in {"unknown", "neutral"}:
        path = service.data.data_dir / 'asset_nv.parquet'
        frame = pd.read_parquet(path)
        if condition == "unknown":
            frame['available_at'] = pd.NaT
        else:
            frame['nv'] = 1.0
        frame.to_parquet(path, index=False)
        baseline = service.create_baseline({'alloc_name': '配置', 'name': condition, 'as_of': str(request.as_of), 'weights': {'股票': .6, '债券': .4}})
        request = request.model_copy(update={'baseline_id': baseline['id']})
    else:
        request = request.model_copy(update={'lookback': 120})
    check = service.preflight(request)
    assert check['training']['train_signal_observations'] == 0
    assert check['training']['eligible'] is False and check['can_calculate'] is False
    assert request.search is True
    assert any(item['action'] == 'fixed_comparison' for item in check['guidance'])
    with pytest.raises(ValidationError) as caught:
        service.preview(request)
    assert caught.value.code == 'TAA_NO_TRAINING_SIGNAL'
    fixed = request.model_copy(update={'search': False})
    assert service.preflight(fixed)['can_calculate']
    result = service.preview(fixed)
    assert result['data']['training']['train_signal_observations'] == 0


def test_momentum_version_in_hash_and_saved_versions_stay_frozen(workspace, monkeypatch):
    from backend.tactical_allocation import numeric
    service, _, request = workspace
    previous = service.preview(request)
    saved = service.save_decision(SaveDecisionRequest(request=request, preview_hash=previous['preview_hash'], name='已冻结'))
    monkeypatch.setattr(numeric, 'MOMENTUM_ALGORITHM_VERSION', 'next-test-version')
    newer = service.preview(request)
    assert newer['preview_hash'] != previous['preview_hash']
    with pytest.raises(ConflictError):
        service.save_decision(SaveDecisionRequest(request=request, preview_hash=previous['preview_hash'], name='旧预览'))
    restored = service.repository.get_decision(saved['id'])
    assert restored['preview'] == previous and restored['content_hash'] == saved['content_hash']
    timing = newer['recommendation']['signal_details'][0]['window']
    assert timing['window_end'] == newer['recommendation']['signal_date']
    assert timing['available_at'] <= timing['period_start']
    assert timing['lag_days'] == 0

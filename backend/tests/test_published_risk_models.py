"""Offline reference and workflow acceptance for the published research loop."""
from __future__ import annotations

import copy
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError as ContractError

from backend.custom_indicators.errors import ValidationError
from backend.scenario_stress.published import PublishedScenarioService
from backend.services.published_scenario_routes import build_router as scenario_router
from backend.services.risk_model_routes import build_router as model_router
from backend.sensitivity.catalog import VariableRegistry
from backend.sensitivity.contracts import ModelFields
from backend.sensitivity.kernels import (
    KERNELS, cashflow_metrics_kernel, cashflow_shock_kernel, ending_weights_kernel,
    execution_audit, lag_features_kernel, resample_transform_kernel,
    warm_sensitivity_kernels, wealth_impact_kernel,
)
from backend.sensitivity.repository import ArtifactRepository, digest_json
from backend.sensitivity.service import ModelResearchService, _fit


@pytest.fixture(scope="module", autouse=True)
def warmed():
    warm_sensitivity_kernels()


def import_values(registry, dates, values, name, roles, unit):
    csv = 'date,value,available_at\n' + '\n'.join(f'{day.date().isoformat()},{value:.12g},{day.date().isoformat()}' for day, value in zip(dates, values))
    return registry.import_series({"name": name, "roles": roles, "unit": unit, "frequency": "monthly",
        "transform": "identity", "source_label": "offline synthetic test fixture, not market observations", "csv_text": csv})


@pytest.fixture()
def environment(tmp_path):
    today = datetime.now(timezone.utc).date()
    final = pd.Timestamp(today).to_period('M') - 1
    dates = pd.period_range(end=final, periods=84, freq='M').end_time.normalize()
    rng = np.random.default_rng(173)
    driver = rng.normal(0, 1, len(dates))
    macro = .4 * driver + rng.normal(0, .005, len(dates))
    market = .025 * macro + rng.normal(0, .0001, len(dates))
    registry = VariableRegistry(tmp_path)
    d = import_values(registry, dates, driver, '供给驱动', ['driver'], 'points')
    m = import_values(registry, dates, macro, '经济变动', ['macro'], 'pp')
    f = import_values(registry, dates, market, '市场收益', ['market'], 'return')
    calendar = pd.date_range(dates[0] - pd.Timedelta(days=70), pd.Timestamp(today) + pd.Timedelta(days=60), freq='D')
    pd.DataFrame({'exchange': 'SSE', 'cal_date': calendar.strftime('%Y%m%d'), 'is_open': (calendar.dayofweek < 5).astype(int)}).to_parquet(tmp_path / 'trade_day_df.parquet', index=False)
    baseline = (dates[0].to_period('M') - 1).end_time.normalize()
    periods = [baseline, *dates]
    closes = [pd.bdate_range(day.to_period('M').start_time, day)[-1] for day in periods]
    records = []
    for code, beta in [('000001.OF', .7), ('000002.OF', .3)]:
        nav = np.r_[1.0, np.cumprod(1.0 + beta * market)]
        records.extend({'ts_code': code, 'nav_date': day.strftime('%Y%m%d'), 'ann_date': day.strftime('%Y%m%d'), 'adj_nav': float(value)} for day, value in zip(closes, nav))
    pd.DataFrame(records).to_parquet(tmp_path / 'fund_nav_df.parquet', index=False)
    pd.DataFrame({'ts_code': ['000001.OF', '000002.OF'], 'name': ['测试基金甲', '测试基金乙']}).to_parquet(tmp_path / 'fund_info_df.parquet', index=False)
    model = ModelResearchService(tmp_path)
    transmission = ModelResearchService(tmp_path, 'transmission')
    scenarios = PublishedScenarioService(tmp_path)
    fields = {'name': '离线敏感度研究', 'stage': 'product', 'inputs': [f['id']], 'targets': [
        {'kind': 'fund', 'product_id': '000001.OF'}, {'kind': 'fund', 'product_id': '000002.OF'}],
        'outputs': [], 'frequency': 'monthly', 'start_date': dates[0].date().isoformat(),
        'end_date': dates[-1].date().isoformat(), 'validation_start': dates[-16].date().isoformat(),
        'as_of': today.isoformat(), 'min_train': 36, 'min_validation': 12}
    return SimpleNamespace(root=tmp_path, dates=dates, registry=registry, driver=d, macro=m, factor=f,
        model=model, transmission=transmission, scenarios=scenarios, fields=fields, today=today.isoformat())


def publish_model(service, fields):
    preview = service.preview(fields)
    assert preview['transient'] is True and preview['publishable'], preview['blockers']
    release = service.publish({'definition': fields, 'preview_hash': preview['preview_hash'],
        'valid_days': 180, 'acknowledge_limitations': True})
    return service.get_run(release['run_id']), release


def direct_scenario(env, values=(-10.0,), **changes):
    fields = {'name': '离线市场冲击', 'entry': 'market', 'frequency': 'monthly',
        'input_ids': [env.factor['id']], 'rows': [[value] for value in values]}
    fields.update(changes)
    preview = env.scenarios.preview(fields)
    release = env.scenarios.publish({'definition': fields, 'preview_hash': preview['preview_hash'],
        'valid_days': 90, 'acknowledge_limitations': True})
    return preview, release


def impact_request(env, exposure, scenario, **changes):
    fields = {'scenario_release_id': scenario['id'], 'exposure_release_id': exposure['id'],
        'target': {'kind': 'product', 'product_key': 'fund:000001.OF'}, 'as_of': env.today,
        'hold_other_factors_constant': True, 'notional': 1000000}
    fields.update(changes)
    return fields


def test_unconfirmed_previews_do_not_persist(environment):
    env = environment
    risk = env.model.preview(env.fields)
    assert risk['transient'] is True and risk['preview_hash']
    assert env.model.artifacts.list('run') == []
    assert env.model.artifacts.list('release') == []

    scenario_fields = {'name': '未发布市场冲击', 'entry': 'market', 'frequency': 'monthly',
        'input_ids': [env.factor['id']], 'rows': [[-8.0]]}
    scenario = env.scenarios.preview(scenario_fields)
    assert scenario['transient'] is True and scenario['preview_hash']
    assert env.scenarios.artifacts.list('preview') == []
    assert env.scenarios.artifacts.list('release') == []

    cashflow = env.model.cashflow_preview({'name': '未发布债券', 'product_id': 'bond-preview', 'as_of': env.today,
        'yield_factor_id': 'cn-gov-yield-bp', 'yield_percent': 3., 'compounding': 2,
        'cashflows': [{'years': 1, 'amount': 103}], 'source_label': 'offline fixture'})
    assert cashflow['transient'] is True and cashflow['preview_hash']
    assert env.model.artifacts.list('run') == []


def test_changed_preview_cannot_publish_or_write_artifacts(environment):
    env = environment
    preview = env.model.preview(env.fields)
    changed = {**env.fields, 'name': '参数已改变'}
    with pytest.raises(ValidationError, match='重新计算'):
        env.model.publish({'definition': changed, 'preview_hash': preview['preview_hash'], 'acknowledge_limitations': True})
    assert env.model.artifacts.list('run') == []
    assert env.model.artifacts.list('release') == []

    scenario_fields = {'name': '原始情景', 'entry': 'market', 'frequency': 'monthly',
        'input_ids': [env.factor['id']], 'rows': [[-8.0]]}
    scenario = env.scenarios.preview(scenario_fields)
    with pytest.raises(ValidationError, match='重新预览'):
        env.scenarios.publish({'definition': {**scenario_fields, 'rows': [[-9.0]]},
            'preview_hash': scenario['preview_hash'], 'acknowledge_limitations': True})
    assert env.scenarios.artifacts.list('preview') == []
    assert env.scenarios.artifacts.list('release') == []


def test_shared_ols_matches_reference_and_holdout_is_independent():
    rng = np.random.default_rng(22)
    x = np.ascontiguousarray(rng.normal(size=(90, 2)) * np.array([.01, 100.0]))
    y = np.ascontiguousarray((x @ np.array([.7, -.0004]) + .002 + rng.normal(0, .0001, 90)).reshape(-1, 1))
    fitted, stats, means, scales = _fit(x, y, 60)
    expected = np.linalg.lstsq(np.column_stack([x[:60], np.ones(60)]), y[:60], rcond=None)[0].ravel()
    np.testing.assert_allclose(fitted[0], expected, rtol=1e-8, atol=1e-10)
    changed = y.copy(); changed[60:] += 10
    other, other_stats, _, _ = _fit(x, changed, 60)
    np.testing.assert_array_equal(other, fitted)
    assert other_stats[0, 3] < stats[0, 3]
    np.testing.assert_allclose(means, x[:60].mean(axis=0))
    np.testing.assert_allclose(scales, x[:60].std(axis=0, ddof=1))


def test_product_research_real_parquet_publish_persist_and_reuse(environment, monkeypatch):
    env = environment
    run, release = publish_model(env.model, env.fields)
    np.testing.assert_allclose(np.array(run['coefficients'])[:, 0], [.7, .3], rtol=1e-8)
    assert len(run['provenance']) == 3 and run['execution']['python_fallback'] == 0
    arrays = env.model.artifacts.arrays(run['id'])
    assert isinstance(arrays['values'], np.memmap) and not arrays['values'].flags.writeable
    assert np.shares_memory(arrays['values'][2:], arrays['values'])
    _, scenario = direct_scenario(env)
    result = env.scenarios.impact(impact_request(env, release, scenario))
    assert result['summary']['terminal_return'] == pytest.approx(-.07)
    assert result['summary']['pnl_amount'] == pytest.approx(-70000)
    monkeypatch.setattr(env.model.data, 'load', lambda *args: pytest.fail('Reading must not train'))
    monkeypatch.setattr(env.scenarios.risks.data, 'load', lambda *args: pytest.fail('Applying must not train'))
    repeated = env.scenarios.impact(impact_request(env, release, scenario))
    assert repeated['id'] == result['id']
    assert result['transient'] is True
    assert env.scenarios.impacts.list('impact') == []
    assert not (env.root / 'scenario_stress/impacts' / result['id'] / 'manifest.json').exists()
    assert 'var_95' not in result['summary'] and 'loss_probability' not in result['summary']


def test_three_entry_points_are_equivalent_and_retain_real_lineage(environment):
    env = environment
    _, risk = publish_model(env.model, env.fields)
    base = {**env.fields, 'targets': [], 'stage': 'event_macro', 'inputs': [env.driver['id']], 'outputs': [env.macro['id']]}
    _, first = publish_model(env.transmission, base)
    _, second = publish_model(env.transmission, {**base, 'stage': 'macro_market', 'inputs': [env.macro['id']], 'outputs': [env.factor['id']]})
    preview = env.scenarios.preview({'name': '事件条件传导', 'entry': 'event', 'frequency': 'monthly',
        'event_template': 'custom', 'event_model_release_id': first['id'], 'macro_model_release_id': second['id'], 'rows': [[-2.0], [.5]]})
    assert len(preview['lineage']) == 2
    macro_path = preview['lineage'][0]['path']
    macro_preview = env.scenarios.preview({'name': '宏观入口', 'entry': 'macro', 'frequency': 'monthly',
        'macro_model_release_id': second['id'], 'rows': macro_path})
    np.testing.assert_allclose(preview['path'], macro_preview['path'])
    direct_preview, direct = direct_scenario(env, tuple(row[0] * 100 for row in preview['path']))
    np.testing.assert_allclose(preview['path'], direct_preview['path'])
    release = env.scenarios.publish({'definition': preview['definition'], 'preview_hash': preview['preview_hash'], 'acknowledge_limitations': True})
    actual = env.scenarios.impact(impact_request(env, risk, release))
    expected = env.scenarios.impact(impact_request(env, risk, direct))
    assert actual['summary'] == pytest.approx(expected['summary'])
    assert actual['lineage'][0]['interpretation'].endswith('not_identified_causality')


def test_missing_period_and_lags_do_not_compress_time():
    dates = np.array([0, 1, 3, 4], dtype=np.int64)
    values = np.array([100., 110., 121., 133.1])
    grid = np.arange(5, dtype=np.int64)
    before = values.copy(); values.flags.writeable = False
    result, known, raw = resample_transform_kernel(dates, dates, dates, values, grid, grid, np.int64(5), np.int64(0), np.int64(1))
    assert np.isnan(result[2]) and np.isnan(result[3])
    assert result[1] == pytest.approx(.1) and result[4] == pytest.approx(.1)
    features = lag_features_kernel(result.reshape(-1, 1), np.int64(1))
    assert np.isnan(features[4, 1])
    np.testing.assert_array_equal(values, before)


def test_future_announcements_are_excluded():
    days = np.arange(4, dtype=np.int64)
    values = np.array([100., 101., 102., 103.])
    known = np.array([0, 1, 5, 3], dtype=np.int64)
    output, _, _ = resample_transform_kernel(days, days, known, values, days, days, np.int64(3), np.int64(0), np.int64(1))
    assert np.isnan(output[2]) and np.isnan(output[3])


def test_cashflow_duration_convexity_and_full_repricing():
    times = np.array([1., 2., 3., 4., 5.]); amounts = np.array([4., 4., 4., 4., 104.])
    metrics, status = cashflow_metrics_kernel(times, amounts, np.float64(3.5), np.int64(2))
    def price(y):
        return np.sum(amounts / (1 + y / 2) ** (2 * times))
    assert status == 0 and metrics[0] == pytest.approx(price(.035))
    h = 1e-5
    duration = -(price(.035 + h) - price(.035 - h)) / (2 * h * price(.035))
    convexity = (price(.035 + h) + price(.035 - h) - 2 * price(.035)) / (h * h * price(.035))
    assert metrics[1] == pytest.approx(duration, rel=1e-7)
    assert metrics[2] == pytest.approx(convexity, rel=2e-5)
    result, status = cashflow_shock_kernel(times, amounts, np.float64(3.5), np.int64(2), np.float64(200.))
    assert status == 0 and result[0] == pytest.approx(price(.055) / price(.035) - 1)
    assert abs(result[0] - result[1]) > 1e-6
    _, invalid = cashflow_metrics_kernel(times, amounts, np.float64(-200), np.int64(2))
    assert invalid != 0


def test_cashflow_cannot_treat_any_bp_factor_as_bond_yield(environment):
    env = environment
    request = {'name': '收益率语义校验', 'product_id': 'bond-test', 'as_of': env.today,
        'yield_factor_id': 'cn-shibor-1w', 'yield_percent': 3., 'compounding': 2,
        'cashflows': [{'years': 1, 'amount': 103}], 'source_label': 'offline fixture'}
    with pytest.raises(ValidationError, match='不能把 Shibor'):
        env.model.cashflow_preview(request)
    assert env.model.artifacts.list('run') == []


def test_cashflow_release_and_single_period_application(environment):
    env = environment
    study = {'name': '固定现金流债券', 'product_id': 'bond-a', 'as_of': env.today,
        'yield_factor_id': 'cn-gov-yield-bp', 'yield_percent': 3., 'compounding': 2,
        'cashflows': [{'years': 1, 'amount': 3}, {'years': 2, 'amount': 103}], 'source_label': 'offline fixture'}
    preview = env.model.cashflow_preview(study)
    release = env.model.publish_cashflow({'study': study, 'preview_hash': preview['preview_hash'], 'acknowledge_limitations': True})
    _, scenario = direct_scenario(env, (100.,), input_ids=['cn-gov-yield-bp'])
    result = env.scenarios.impact(impact_request(env, release, scenario, target={'kind': 'product', 'product_key': 'bond:bond-a'}))
    assert result['summary']['terminal_return'] < 0
    assert abs(result['summary']['factor_reconciliation_error']) < 1e-10
    _, multi = direct_scenario(env, (50., 50.), input_ids=['cn-gov-yield-bp'])
    with pytest.raises(ValidationError, match='单期'):
        env.scenarios.impact(impact_request(env, release, multi, target={'kind': 'product', 'product_key': 'bond:bond-a'}))


def test_wealth_linking_and_end_weights():
    returns = np.array([[.1, -.1], [.1, .0]])
    weights = np.array([.5, .5])
    buy, contribution, _, _, status = wealth_impact_kernel(returns, weights, np.int64(0))
    constant, other, _, _, _ = wealth_impact_kernel(returns, weights, np.int64(1))
    assert status == 0
    assert buy[-1, 1] == pytest.approx(.5 * 1.1 * 1.1 + .5 * .9)
    assert constant[-1, 1] == pytest.approx(1.05)
    assert contribution.sum() == pytest.approx(buy[-1, 1] - 1)
    assert other.sum() == pytest.approx(constant[-1, 1] - 1)
    ending, status = ending_weights_kernel(weights, returns[0])
    np.testing.assert_allclose(ending, [.55, .45])


@pytest.mark.parametrize('complete', [True, False])
def test_monthly_research_requires_only_complete_month_calendar(environment, complete):
    env = environment
    path = env.root / 'trade_day_df.parquet'
    calendar = pd.read_parquet(path)
    last_month_end = env.dates[-1]
    cutoff = last_month_end if complete else last_month_end - pd.Timedelta(days=1)
    calendar = calendar[calendar['cal_date'] <= cutoff.strftime('%Y%m%d')]
    calendar.to_parquet(path, index=False)
    fields = {**env.fields, 'end_date': env.today}
    if complete:
        run = env.model.preview(fields)
        assert run['publishable']
        assert run['data_as_of'] == last_month_end.date().isoformat()
        assert env.model.artifacts.list('run') == []
    else:
        with pytest.raises(ValidationError, match='交易日历未覆盖'):
            env.model.preview(fields)


def test_coefficient_editing_and_wrong_roles_rejected(environment):
    env = environment
    with pytest.raises(ContractError):
        ModelFields.model_validate({**env.fields, 'betas': [[.8]]})
    with pytest.raises(ValidationError, match='不能'):
        env.model.preview({**env.fields, 'inputs': [env.macro['id']]})
    with pytest.raises(ContractError):
        ModelFields.model_validate({**env.fields, 'lags': True})
    with pytest.raises(ContractError):
        ModelFields.model_validate({**env.fields, 'validation_start': env.fields['start_date']})


def test_bad_validation_cannot_publish(environment):
    env = environment
    fields = {**env.fields, 'min_validation': 30}
    run = env.model.preview(fields)
    assert not run['publishable']
    assert env.model.artifacts.list('run') == []
    with pytest.raises(ValidationError, match='验证未通过'):
        env.model.publish({'definition': fields, 'preview_hash': run['preview_hash'], 'acknowledge_limitations': True})


def test_release_dates_and_retirement_do_not_rewrite_history(environment):
    env = environment
    run, release = publish_model(env.model, env.fields)
    yesterday = (datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat()
    with pytest.raises(ValidationError, match='尚未'):
        env.model.resolve_release(release['id'], yesterday)
    with pytest.raises(ValidationError, match='回填'):
        env.model.publish({'definition': env.fields, 'preview_hash': run['preview_hash'], 'effective_from': yesterday, 'acknowledge_limitations': True})
    before = env.model.artifacts.get(release['id'], 'release')
    env.model.retire(release['id'], {'note': 'test'})
    assert env.model.artifacts.get(release['id'], 'release') == before
    with pytest.raises(ValidationError, match='停用'):
        env.model.resolve_release(release['id'])
    assert env.model.get_run(run['id'])['content_hash'] == run['content_hash']


def test_publish_is_idempotent_under_concurrent_calls(environment):
    env = environment
    _, first = publish_model(env.model, env.fields)
    preview = env.model.preview(env.fields)
    payload = {'definition': env.fields, 'preview_hash': preview['preview_hash'], 'valid_days': 180, 'acknowledge_limitations': True}
    with ThreadPoolExecutor(max_workers=3) as pool:
        releases = list(pool.map(lambda _: env.model.publish(payload), range(6)))
    assert {item['id'] for item in releases} == {first['id']}
    assert len(env.model.artifacts.list('release')) == 1


def test_factor_contract_and_frequency_mismatches_fail_closed(environment):
    env = environment
    _, release = publish_model(env.model, env.fields)
    _, wrong_factor = direct_scenario(env, input_ids=['cn-equity-csi300'])
    with pytest.raises(ValidationError, match='单位|来源|缺少'):
        env.scenarios.impact(impact_request(env, release, wrong_factor))
    _, wrong_frequency = direct_scenario(env, frequency='daily')
    with pytest.raises(ValidationError, match='周期'):
        env.scenarios.impact(impact_request(env, release, wrong_frequency))


def test_portfolio_uses_locked_end_holdings_and_blocks_missing_exposures(environment):
    env = environment
    _, release = publish_model(env.model, env.fields)
    _, scenario = direct_scenario(env)
    assets = [{'key': 'fund:000001.OF', 'kind': 'fund', 'product_id': '000001.OF', 'name': '甲'},
              {'key': 'fund:000002.OF', 'kind': 'fund', 'product_id': '000002.OF', 'name': '乙'}]
    snapshot = env.scenarios.portfolios.create({'target_name': '锁定组合', 'assets': assets,
        'daily_weights': [[.5, .5]], 'asset_returns': [[.1, -.1]], 'effective_as_of': env.today})
    result = env.scenarios.impact(impact_request(env, release, scenario, target={'kind': 'portfolio_run', 'portfolio_run_id': snapshot['id']}))
    np.testing.assert_allclose([item['weight'] for item in result['by_asset']], [.55, .45])
    assert result['summary']['terminal_return'] == pytest.approx(-.07 * .55 - .03 * .45)
    unknown = copy.deepcopy(assets); unknown[1].update(key='fund:UNKNOWN', product_id='UNKNOWN')
    missing = env.scenarios.portfolios.create({'target_name': '缺失组合', 'assets': unknown,
        'daily_weights': [[.5, .5]], 'asset_returns': [[0., 0.]], 'effective_as_of': env.today})
    with pytest.raises(ValidationError, match='非零持仓'):
        env.scenarios.impact(impact_request(env, release, scenario, target={'kind': 'portfolio_run', 'portfolio_run_id': missing['id']}))


def test_artifact_tamper_and_path_traversal_are_rejected(tmp_path):
    repository = ArtifactRepository(tmp_path / 'artifacts')
    run = repository.save('run', {'name': 'test'}, {'values': np.array([1., np.nan, 3.])})
    folder = repository.root / run['id']
    assert np.isnan(repository.arrays(run['id'])['values'][1])
    with pytest.raises(ValidationError):
        repository.get('../escape')
    with (folder / 'values.npy').open('ab') as handle:
        handle.write(b'tamper')
    with pytest.raises(ValidationError, match='损坏'):
        repository.arrays(run['id'])
    manifest = json.loads((folder / 'manifest.json').read_text())
    manifest['name'] = 'changed'
    (folder / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValidationError, match='校验失败'):
        repository.get(run['id'])


def test_managed_storage_guard_precedes_any_creation(tmp_path, monkeypatch):
    from backend.sensitivity import repository as module
    root = tmp_path / 'missing-disk' / 'risk_models'
    def offline(*args, **kwargs):
        raise ValidationError('STORAGE_OFFLINE', '磁盘离线')
    monkeypatch.setattr(module, 'guard_path', offline)
    repository = ArtifactRepository(root)
    with pytest.raises(ValidationError, match='离线'):
        repository.save('run', {'name': 'must fail'})
    with pytest.raises(ValidationError, match='离线'):
        repository.list()
    assert not root.exists()


def test_no_new_signatures_or_python_fallback():
    before = {function.py_func.__name__: list(function.signatures) for function in KERNELS}
    warm_sensitivity_kernels()
    after = {function.py_func.__name__: list(function.signatures) for function in KERNELS}
    assert before == after
    audit = execution_audit()
    assert audit['python_fallback'] == 0 and audit['request_time_compilation'] == 0 and audit['object_mode'] == 0
    assert all(function.nopython_signatures for function in KERNELS)


def test_router_workflow_and_strict_public_contract(environment):
    env = environment
    app = FastAPI()
    app.include_router(model_router(env.model))
    app.include_router(model_router(env.transmission))
    app.include_router(scenario_router(env.scenarios))
    with TestClient(app) as client:
        response = client.get('/api/risk-models/catalog')
        assert response.status_code == 200
        assert response.json()['capabilities']['coefficient_editing'] is False
        assert response.json()['capabilities']['persist_only_on_publish'] is True
        assert client.post('/api/risk-models/previews', json={**env.fields, 'beta': .5}).status_code == 422
        run = client.post('/api/risk-models/previews', json=env.fields)
        assert run.status_code == 200, run.text
        assert run.json()['transient'] is True
        assert env.model.artifacts.list('run') == []
        release = client.post('/api/risk-models/releases', json={'definition': env.fields,
            'preview_hash': run.json()['preview_hash'], 'acknowledge_limitations': True, 'valid_days': 180})
        assert release.status_code == 201, release.text
        assert len(env.model.artifacts.list('run')) == 1
        assert client.get('/api/risk-models/releases?product_key=fund:000001.OF').json()['items'][0]['id'] == release.json()['id']
        assert client.get('/api/published-scenarios/impacts').json()['items'] == []
        scenario_fields = {'name': 'HTTP冲击', 'entry': 'market', 'frequency': 'monthly',
            'input_ids': [env.factor['id']], 'rows': [[-10.0]]}
        preview = client.post('/api/published-scenarios/previews', json=scenario_fields)
        assert preview.status_code == 200, preview.text
        assert preview.json()['transient'] is True
        assert env.scenarios.artifacts.list('preview') == []
        scenario = client.post('/api/published-scenarios/releases', json={'definition': scenario_fields,
            'preview_hash': preview.json()['preview_hash'], 'acknowledge_limitations': True})
        assert scenario.status_code == 201, scenario.text
        assert len(env.scenarios.artifacts.list('preview')) == 1
        impact = client.post('/api/published-scenarios/impacts', json=impact_request(env, release.json(), scenario.json()))
        assert impact.status_code == 200, impact.text
        assert impact.json()['transient'] is True
        stored = client.get(f'/api/published-scenarios/impacts/{impact.json()["id"]}')
        assert stored.status_code == 404


def test_collinear_inputs_record_failed_research_instead_of_crashing(environment):
    env = environment
    source = env.registry.imports.arrays(env.factor['id'])['values']
    duplicate = import_values(env.registry, env.dates, source * 2, '共线因子', ['market'], 'return')
    fields = {**env.fields, 'inputs': [env.factor['id'], duplicate['id']]}
    run = env.model.preview(fields)
    assert not run['publishable']
    assert all(row['train_observations'] > 0 for row in run['rows'])
    assert all(row['status'] != 0 for row in run['rows'])
    with pytest.raises(ValidationError, match='验证未通过'):
        env.model.publish({'definition': fields, 'preview_hash': run['preview_hash'], 'acknowledge_limitations': True})


def test_actual_data_end_not_requested_asof_controls_freshness(environment):
    env = environment
    fields = {**env.fields, 'end_date': env.dates[-14].date().isoformat(),
              'validation_start': env.dates[-29].date().isoformat()}
    run = env.model.preview(fields)
    assert run['publishable'], run['blockers']
    assert run['as_of'] == env.today
    assert run['data_as_of'] == fields['end_date']
    with pytest.raises(ValidationError, match='有效|超出'):
        env.model.publish({'definition': fields, 'preview_hash': run['preview_hash'], 'valid_days': 180, 'acknowledge_limitations': True})


def test_duplicate_canonical_product_aliases_rejected(environment):
    env = environment
    fields = {**env.fields, 'targets': [{'kind': 'fund', 'product_id': '000001.OF'},
                                      {'kind': 'fund', 'product_id': '000001'}]}
    with pytest.raises(ValidationError, match='重复'):
        env.model.preview(fields)


def test_revoked_transmission_blocks_new_impacts_but_not_saved_results(environment):
    env = environment
    _, risk = publish_model(env.model, env.fields)
    _, macro = publish_model(env.transmission, {**env.fields, 'stage': 'macro_market', 'targets': [],
        'inputs': [env.macro['id']], 'outputs': [env.factor['id']]})
    preview = env.scenarios.preview({'name': '宏观压测', 'entry': 'macro', 'frequency': 'monthly',
        'macro_model_release_id': macro['id'], 'rows': [[-1.0]]})
    scenario = env.scenarios.publish({'definition': preview['definition'], 'preview_hash': preview['preview_hash'], 'acknowledge_limitations': True})
    request = impact_request(env, risk, scenario)
    result = env.scenarios.impact(request)
    env.transmission.retire(macro['id'], {'note': '模型停用回归测试'})
    assert env.scenarios.releases()['items'][0]['status'] == 'dependency_unavailable'
    with pytest.raises(ValidationError, match='停用'):
        env.scenarios.impact(request)
    assert result['transient'] is True
    assert env.scenarios.impacts.list('impact') == []


def test_concurrent_impacts_reuse_one_artifact(environment):
    env = environment
    _, exposure = publish_model(env.model, env.fields)
    _, scenario = direct_scenario(env)
    request = impact_request(env, exposure, scenario)
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(lambda _: env.scenarios.impact(request), range(4)))
    assert len({item['id'] for item in results}) == 1
    assert all(item['transient'] is True for item in results)
    assert env.scenarios.impacts.list('impact') == []


def test_application_reads_only_coefficients_and_shocks_not_training_panels(environment, monkeypatch):
    from backend.sensitivity import repository as module
    env = environment
    _, exposure = publish_model(env.model, env.fields)
    _, scenario = direct_scenario(env)
    files = []
    original = module.file_hash
    def observe(path):
        if path.parent.name in {exposure['run_id'], scenario['preview_id']}:
            files.append(path.name)
        return original(path)
    monkeypatch.setattr(module, 'file_hash', observe)
    env.scenarios.impact(impact_request(env, exposure, scenario))
    assert sorted(files) == ['coefficients.npy', 'factor_path.npy']
    selected = env.model.artifacts.arrays(exposure['run_id'], names=('coefficients',))
    assert list(selected) == ['coefficients']
    assert not selected['coefficients'].flags.writeable
    with pytest.raises(ValidationError, match='缺少'):
        env.model.artifacts.arrays(exposure['run_id'], names=('missing',))


def test_kernel_empty_shape_and_unsupported_layout_fail_safely():
    empty = np.empty((0, 1), dtype=np.float64)
    assert lag_features_kernel(empty, np.int64(1)).shape == (0, 2)
    with pytest.raises(ValueError, match='dimensions'):
        wealth_impact_kernel(empty, np.ones(1), np.int64(0))
    with pytest.raises(ValueError, match='dimensions'):
        wealth_impact_kernel(np.ones((2, 2)), np.ones(1), np.int64(0))
    with pytest.raises(ValueError, match='negative lag'):
        lag_features_kernel(np.ones((2, 1)), np.int64(-1))
    _, status = cashflow_metrics_kernel(np.ones(1), np.ones(1), np.float64(3), np.int64(0))
    assert status != 0
    signatures = list(lag_features_kernel.signatures)
    with pytest.raises(TypeError):
        lag_features_kernel(np.ones((4, 2))[::2], np.int64(1))
    assert lag_features_kernel.signatures == signatures

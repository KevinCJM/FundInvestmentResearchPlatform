"""最大夏普 / 最小模拟回撤候选：同一有限搜索、NJIT 路径、无风险利率来源顺序。"""
from datetime import date, timedelta
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.scenario_stress.numba_kernels import seeded_factor_draws_kernel
from backend.strategic_allocation import kernels
from backend.strategic_allocation.routes import build_router
from backend.strategic_allocation.service import StrategicAllocationService


@pytest.fixture(scope='module', autouse=True)
def warmed():
    from backend.strategic_allocation import institution_kernels
    kernels.warm_strategic_kernels()
    institution_kernels.warm()


def draws(months=120, paths=64):
    values, _ = seeded_factor_draws_kernel(months, paths, 1, 7, 0, 5.)
    values.flags.writeable = False
    return values[:, :, 0]


def search_inputs():
    covariance = np.array([[.15 ** 2, .002, 0.], [.002, .05 ** 2, 0.], [0., 0., 1e-6]])
    return (np.array([.07, .035, .02]), covariance, np.array([.02, .005, 0.]),
            np.array([[0., 1.]] * 3), np.empty((0, 3), dtype=np.uint8), np.empty(0), np.empty(0),
            5., 1., 0., 1., np.empty(0), 1., 0., 1000, 42, np.empty(0))


def test_drawdown_kernel_matches_plain_compounding_reference():
    shared = draws()
    drift, scale = (np.log1p(.06) - .5 * np.log1p((.12 / 1.06) ** 2)) / 12, np.sqrt(np.log1p((.12 / 1.06) ** 2) / 12)
    growth = np.cumprod(np.exp(drift + scale * np.asarray(shared)), axis=0)
    peak = np.maximum.accumulate(np.vstack([np.ones((1, growth.shape[1])), growth]), axis=0)[1:]
    reference = np.max(1 - growth / peak, axis=0).mean()
    assert kernels.mean_max_drawdown_kernel(.06, .12, shared) == pytest.approx(reference, rel=1e-10)
    assert np.shares_memory(shared, shared.base)  # 3D draws are consumed as a strided view, not copied.


def test_extended_rows_extend_the_same_search_without_changing_base_rows():
    shared = draws()
    base = kernels.policy_candidates_with_budget_kernel(*search_inputs(), 0., np.empty((0, 0)))
    weights, metrics, contributions, accepted = kernels.policy_candidates_with_budget_kernel(*search_inputs(), .02, shared)
    assert accepted == base[3] and weights.shape[0] == 6
    np.testing.assert_array_equal(weights[:4], base[0])
    adjusted = kernels.risk_adjusted_metrics_kernel(metrics, .02, shared)
    # Each extended row is at least as good as any base selection from the same accepted set.
    assert adjusted[4, 0] >= np.nanmax(adjusted[:4, 0]) - 1e-12
    assert adjusted[5, 1] <= np.nanmin(adjusted[:4, 1]) + 1e-12
    np.testing.assert_allclose(weights.sum(axis=1), 1., atol=1e-8)
    assert kernels.execution_audit()['python_fallback'] == 0


def test_nonfinite_risk_free_or_draws_fail_closed():
    with pytest.raises(ValueError, match='POLICY_'):
        kernels.policy_candidates_with_budget_kernel(*search_inputs(), np.nan, draws())
    bad = np.full((12, 4), np.nan)
    with pytest.raises(ValueError, match='POLICY_DRAWDOWN_DRAWS'):
        kernels.mean_max_drawdown_kernel(.05, .1, bad)


def test_risk_free_falls_back_from_scope_cash_to_risk_scale_cash_to_zero(tmp_path):
    service = StrategicAllocationService(tmp_path / 'research', tmp_path / 'market')
    scope = lambda proxy: {'source_snapshot': {'strategic_universe_snapshot': {'definition': {'assets': [
        {'id': 'cash', 'name': '现金', 'research_proxy': proxy}]}}}}
    mandate = {'currency': 'CNY', 'risk_authorization': {'risk_scale_ref': {'id': 'scale-1', 'content_hash': 'h'}}}
    service.risk_scales = SimpleNamespace(
        get_version=lambda _: {'id': 'scale-1', 'name': '系统标尺', 'content_hash': 'h',
                               'preview': {'request_echo': {'definition': {'reference_input_ref': {'id': 'ref', 'content_hash': 'r'}}}}},
        references=SimpleNamespace(get=lambda *_a, **_k: {'definition': {'assets': [
            {'id': 'eq', 'name': '权益', 'asset_type': 'market'}, {'id': 'rs-cash', 'name': '标尺现金', 'asset_type': 'cash', 'cash_return': .018}]}}))
    assert service._risk_free_rate(mandate, scope({'asset_type': 'cash', 'cash_return': .025})) == {
        'rate': .025, 'source': 'scope_cash', 'asset_id': 'cash', 'asset_name': '现金'}
    fallback = service._risk_free_rate(mandate, scope(None))
    assert fallback['source'] == 'risk_scale_cash' and fallback['rate'] == .018
    assert fallback['risk_scale'] == {'id': 'scale-1', 'name': '系统标尺', 'content_hash': 'h'}
    service.risk_scales.store = SimpleNamespace(read=lambda: {'defaults': {}}, key=lambda c, b: c + ':' + b)
    assert service._risk_free_rate({'currency': 'CNY'}, scope(None)) == {'rate': 0., 'source': 'default_zero'}


def test_policy_preview_adds_sharpe_and_drawdown_candidates_with_scope_cash_rate(tmp_path):
    client = TestClient(FastAPI())
    client.app.include_router(build_router(StrategicAllocationService(tmp_path / 'research', tmp_path / 'market')))
    body = {'name': '夏普范围', 'as_of': str(date.today()), 'currency': 'CNY', 'source': '', 'assets': [
        {'id': 'growth', 'name': '增长资产', 'role': 'growth', 'liquidity': 'liquid', 'currency': 'CNY'},
        {'id': 'cash', 'name': '现金', 'role': 'liquidity', 'liquidity': 'liquid', 'currency': 'CNY',
         'research_proxy': {'asset_type': 'cash', 'cash_return': .025}}]}
    preview = client.post('/api/strategic-allocation/universes/preview', json=body)
    scope = client.post('/api/strategic-allocation/universes/confirm', json={'request': body, 'preview_hash': preview.json()['preview_hash']})
    assert scope.status_code == 201, scope.text
    study = {'definition': {
        'name': '夏普目标', 'as_of': str(date.today()), 'review_date': str(date.today() + timedelta(days=180)),
        'currency': 'CNY', 'horizon_years': 5, 'target_return': 0., 'max_volatility': .3, 'max_tracking_error': .1,
        'boundary_reason': '测试夏普与回撤候选。', 'strategic_universe_id': scope.json()['id']}}
    diagnosed = client.post('/api/strategic-allocation/mandates/preview', json=study)
    mandate = client.post('/api/strategic-allocation/mandates/confirm', json={
        'request': study, 'preview_hash': diagnosed.json()['preview_hash'], 'acknowledge_limits': True})
    assert mandate.status_code == 201, mandate.text
    assumptions = {
        'name': '夏普假设', 'strategic_universe_id': scope.json()['id'], 'as_of': str(date.today()),
        'currency': 'CNY', 'horizon_years': 5, 'source': '离线前瞻输入。', 'basis_confirmed': True,
        'assets': [
            {'id': 'growth', 'role': 'growth', 'liquidity': 'liquid', 'rationale': '增长风险预算',
             'annual_return': .08, 'annual_volatility': .18, 'mean_uncertainty': .02},
            {'id': 'cash', 'role': 'liquidity', 'liquidity': 'liquid', 'rationale': '经营资金储备',
             'annual_return': .02, 'annual_volatility': .01, 'mean_uncertainty': .001},
        ], 'correlation': [[1., 0.], [0., 1.]]}
    cma_preview = client.post('/api/strategic-allocation/cma/preview', json=assumptions)
    cma = client.post('/api/strategic-allocation/cma', json={'request': assumptions, 'preview_hash': cma_preview.json()['preview_hash']})
    assert cma.status_code == 201, cma.text
    result = client.post('/api/strategic-allocation/policy/preview', json={
        'mandate_id': mandate.json()['id'], 'cma_id': cma.json()['id'], 'candidate_count': 400})
    assert result.status_code == 200, result.text
    payload = result.json()
    assert [c['id'] for c in payload['candidates']] == [
        'minimum-risk', 'nominal-utility', 'robust-utility', 'maximum-return', 'maximum-sharpe', 'minimum-drawdown']
    assert payload['risk_adjusted_basis']['risk_free'] == {'rate': .025, 'source': 'scope_cash', 'asset_id': 'cash', 'asset_name': '现金'}
    assert payload['risk_adjusted_basis']['drawdown']['months'] == 60
    by_id = {c['id']: c['risk_adjusted'] for c in payload['candidates']}
    assert by_id['maximum-sharpe']['sharpe_ratio'] >= max(v['sharpe_ratio'] for v in by_id.values() if v['sharpe_ratio'] is not None) - 1e-12
    assert by_id['minimum-drawdown']['mean_max_drawdown'] <= min(v['mean_max_drawdown'] for v in by_id.values()) + 1e-12

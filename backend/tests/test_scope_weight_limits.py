"""研究范围大类权重边界：只收紧投资目标，现金下限低于目标时 SAA 仍按目标执行。"""
from datetime import date, timedelta

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.strategic_allocation.routes import build_router
from backend.strategic_allocation.service import StrategicAllocationService


@pytest.fixture(scope='module', autouse=True)
def warmed():
    from backend.strategic_allocation import kernels, institution_kernels
    kernels.warm_strategic_kernels()
    institution_kernels.warm()


def scope_body(growth_limits=None, cash_limits=None):
    assets = [
        {'id': 'growth', 'name': '增长资产', 'role': 'growth', 'liquidity': 'liquid', 'currency': 'CNY'},
        {'id': 'cash', 'name': '现金', 'role': 'liquidity', 'liquidity': 'liquid', 'currency': 'CNY'},
    ]
    for asset, limits in zip(assets, (growth_limits, cash_limits)):
        if limits is not None:
            asset['weight_limits'] = limits
    return {'name': '权重边界范围', 'as_of': str(date.today()), 'currency': 'CNY', 'source': '', 'assets': assets}


def client_for(tmp_path):
    app = FastAPI()
    app.include_router(build_router(StrategicAllocationService(tmp_path / 'research', tmp_path / 'market')))
    return TestClient(app)


def test_unset_limits_keep_legacy_scope_hash(tmp_path):
    client = client_for(tmp_path)
    legacy = client.post('/api/strategic-allocation/universes/preview', json=scope_body())
    assert legacy.status_code == 200, legacy.text
    assert all('weight_limits' not in asset for asset in legacy.json()['definition']['assets'])
    limited = client.post('/api/strategic-allocation/universes/preview', json=scope_body(cash_limits={'min_weight': .1, 'max_weight': 1}))
    assert limited.json()['definition']['assets'][1]['weight_limits'] == {'min_weight': .1, 'max_weight': 1}
    assert limited.json()['preview_hash'] != legacy.json()['preview_hash']


@pytest.mark.parametrize('growth, cash', [({'min_weight': .6, 'max_weight': .5}, None),
                                          ({'min_weight': .7, 'max_weight': 1}, {'min_weight': .4, 'max_weight': 1})])
def test_invalid_limits_are_rejected(tmp_path, growth, cash):
    response = client_for(tmp_path).post('/api/strategic-allocation/universes/preview', json=scope_body(growth, cash))
    assert response.status_code == 422, response.text


def test_saa_tightens_with_scope_limits_but_never_loosens_mandate_cash(tmp_path):
    client = client_for(tmp_path)
    body = scope_body({'min_weight': .5, 'max_weight': 1}, {'min_weight': .05, 'max_weight': 1})
    preview = client.post('/api/strategic-allocation/universes/preview', json=body)
    scope = client.post('/api/strategic-allocation/universes/confirm', json={'request': body, 'preview_hash': preview.json()['preview_hash']})
    assert scope.status_code == 201, scope.text
    study = {'definition': {
        'name': '现金下限目标', 'as_of': str(date.today()), 'review_date': str(date.today() + timedelta(days=180)),
        'currency': 'CNY', 'horizon_years': 10, 'target_return': 0., 'max_volatility': .3, 'max_tracking_error': .1,
        'boundary_reason': '测试范围边界与目标现金下限的合并。', 'strategic_universe_id': scope.json()['id'],
        'institutional_context': {'investor_type': 'corporate_treasury', 'purpose': '经营备用资金', 'cash_reserve_weight': .2}}}
    diagnosed = client.post('/api/strategic-allocation/mandates/preview', json=study)
    mandate = client.post('/api/strategic-allocation/mandates/confirm', json={
        'request': study, 'preview_hash': diagnosed.json()['preview_hash'], 'acknowledge_limits': True})
    assert mandate.status_code == 201, mandate.text
    assumptions = {
        'name': '边界测试假设', 'strategic_universe_id': scope.json()['id'], 'as_of': str(date.today()),
        'currency': 'CNY', 'horizon_years': 10, 'source': '离线前瞻输入。', 'basis_confirmed': True,
        'assets': [
            {'id': 'growth', 'role': 'growth', 'liquidity': 'liquid', 'rationale': '增长风险预算',
             'annual_return': .08, 'annual_volatility': .18, 'mean_uncertainty': .02},
            {'id': 'cash', 'role': 'liquidity', 'liquidity': 'liquid', 'rationale': '经营资金储备',
             'annual_return': .02, 'annual_volatility': .01, 'mean_uncertainty': .001},
        ], 'correlation': [[1., 0.], [0., 1.]]}
    cma_preview = client.post('/api/strategic-allocation/cma/preview', json=assumptions)
    assert cma_preview.status_code == 200, cma_preview.text
    cma = client.post('/api/strategic-allocation/cma', json={'request': assumptions, 'preview_hash': cma_preview.json()['preview_hash']})
    assert cma.status_code == 201, cma.text
    candidates = client.post('/api/strategic-allocation/policy/preview', json={
        'mandate_id': mandate.json()['id'], 'cma_id': cma.json()['id'], 'candidate_count': 200})
    assert candidates.status_code == 200, candidates.text
    assert candidates.json()['candidates']
    for candidate in candidates.json()['candidates']:
        assert candidate['weights']['growth'] >= .5 - 1e-8  # 范围下限收紧
        assert candidate['weights']['cash'] >= .2 - 1e-8  # 范围 5% 不能放宽目标 20%

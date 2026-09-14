"""Coordinator acceptance: real 01→02→03 flow without fabricated product/NAV data."""
from datetime import date, timedelta

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.custom_indicators.errors import ValidationError
from backend.strategic_allocation.service import StrategicAllocationService
from backend.strategic_allocation.routes import build_router
from backend.strategic_allocation.policy_gate import check_policy, require_policy_application
from backend.tactical_allocation.service import TacticalAllocationService
from backend.tactical_allocation.contracts import PreviewRequest


@pytest.fixture(scope='module', autouse=True)
def warmed():
    from backend.strategic_allocation import kernels, institution_kernels
    from backend.tactical_allocation.data import warm_tactical_data
    from backend.tactical_allocation.numeric import warm_tactical_allocation_kernels
    kernels.warm_strategic_kernels()
    institution_kernels.warm()
    warm_tactical_data()
    warm_tactical_allocation_kernels()


def universe_request():
    return {'name': '独立经济机会集', 'as_of': str(date.today()), 'currency': 'CNY', 'source': '离线验收显式范围',
            'assets': [
                {'id': 'growth', 'name': '增长资产', 'role': 'growth', 'liquidity': 'liquid',
                 'currency': 'CNY', 'rationale': '长期增长风险', 'source': '研究员定义'},
                {'id': 'cash', 'name': '经营储备', 'role': 'liquidity', 'liquidity': 'liquid',
                 'currency': 'CNY', 'rationale': '现金用途储备', 'source': '研究员定义'},
            ]}


def setup_journey(tmp_path):
    service = StrategicAllocationService(tmp_path / 'research', tmp_path / 'market')
    app = FastAPI()
    app.include_router(build_router(service))
    client = TestClient(app)
    scope_body = universe_request()
    scope_preview = client.post('/api/strategic-allocation/universes/preview', json=scope_body)
    assert scope_preview.status_code == 200, scope_preview.text
    assert not service.artifacts.root.exists()
    scope = client.post('/api/strategic-allocation/universes/confirm', json={
        'request': scope_body, 'preview_hash': scope_preview.json()['preview_hash']})
    assert scope.status_code == 201, scope.text
    definition = {
        'name': '企业现金用途研究', 'as_of': str(date.today()),
        'review_date': str(date.today() + timedelta(days=180)), 'currency': 'CNY',
        'horizon_years': 10, 'target_return': 0., 'max_volatility': .3,
        'max_tracking_error': .1, 'boundary_reason': '测试明确现金用途，不将可交易权益当成现金。',
        'strategic_universe_id': scope.json()['id'],
        'institutional_context': {'investor_type': 'corporate_treasury', 'purpose': '经营备用与长期闲置资金研究',
                                 'cash_reserve_weight': .2},
    }
    study = {'definition': definition}
    diagnosed = client.post('/api/strategic-allocation/mandates/preview', json=study)
    assert diagnosed.status_code == 200, diagnosed.text
    mandate = client.post('/api/strategic-allocation/mandates/confirm', json={
        'request': study, 'preview_hash': diagnosed.json()['preview_hash'], 'acknowledge_limits': True})
    assert mandate.status_code == 201, mandate.text
    assumptions = {
        'name': '无产品的纯前瞻假设', 'strategic_universe_id': scope.json()['id'],
        'as_of': str(date.today()), 'currency': 'CNY', 'horizon_years': 10,
        'source': '离线研究显式前瞻输入；不包含真实业绩或投资建议。', 'basis_confirmed': True,
        'assets': [
            {'id': 'growth', 'role': 'growth', 'liquidity': 'liquid', 'rationale': '增长风险预算',
             'annual_return': .08, 'annual_volatility': .18, 'mean_uncertainty': .02},
            {'id': 'cash', 'role': 'liquidity', 'liquidity': 'liquid', 'rationale': '经营资金储备',
             'annual_return': .02, 'annual_volatility': .01, 'mean_uncertainty': .001},
        ], 'correlation': [[1., 0.], [0., 1.]],
    }
    cma_preview = client.post('/api/strategic-allocation/cma/preview', json=assumptions)
    assert cma_preview.status_code == 200, cma_preview.text
    cma = client.post('/api/strategic-allocation/cma', json={
        'request': assumptions, 'preview_hash': cma_preview.json()['preview_hash']})
    assert cma.status_code == 201, cma.text
    policy_body = {'mandate_id': mandate.json()['id'], 'cma_id': cma.json()['id'], 'candidate_count': 200}
    candidates = client.post('/api/strategic-allocation/policy/preview', json=policy_body)
    assert candidates.status_code == 200, candidates.text
    for candidate in candidates.json()['candidates']:
        assert candidate['weights']['cash'] >= .2 - 1e-8
    saved = client.post('/api/strategic-allocation/policies', json={
        'request': policy_body, 'preview_hash': candidates.json()['preview_hash'],
        'candidate_id': 'robust-utility', 'name': '保留缺口的战略研究', 'reason': '明确保留缺少产品的战略需求，不假定已经实施。'})
    assert saved.status_code == 201, saved.text
    return service, client, saved.json(), scope.json(), cma.json()


def test_independent_scope_reaches_real_saa_without_any_market_files(tmp_path):
    service, client, baseline, scope, cma = setup_journey(tmp_path)
    assert not (tmp_path / 'market').exists()
    assert baseline['strategic_universe_id'] == scope['id']
    assert baseline['implementation_status'] == 'incomplete'
    assert set(baseline['implementation_gaps']) == {'growth', 'cash'}
    assert all(not asset['products'] for asset in baseline['assets'])
    assert len(baseline['assets']) == 2
    assert baseline['policy']['cma_hash'] == cma['content_hash']
    assert service.baselines.get_baseline(baseline['id']) == baseline
    assert client.get('/api/strategic-allocation/universes/' + scope['id']).json() == scope
    np.testing.assert_allclose(sum(asset['base_weight'] for asset in baseline['assets']), 1.)


def test_unmapped_strategic_budget_cannot_enter_taa_or_disappear(tmp_path):
    service, _, baseline, _, _ = setup_journey(tmp_path)
    tactical = TacticalAllocationService(tmp_path / 'research', tmp_path / 'market')
    request = PreviewRequest(baseline_id=baseline['id'], start_date=date.today() - timedelta(days=160),
                             end_date=date.today(), as_of=date.today(),
                             train_end_date=date.today() - timedelta(days=60), signal_mode='manual',
                             manual_tilts={'growth': .05, 'cash': -.05})
    with pytest.raises(ValidationError, match='映射缺口'):
        tactical.preflight(request)
    assert service.baselines.get_baseline(baseline['id']) == baseline


def test_cash_role_and_manual_reviews_are_independent_application_gates(tmp_path):
    _, _, baseline, _, _ = setup_journey(tmp_path)
    compliant = {a['id']: a['base_weight'] for a in baseline['assets']}
    diagnostic = check_policy(baseline, compliant, .1, str(date.today()))
    assert diagnostic['within_limits'] is True
    assert diagnostic['current_application_eligible'] is False
    assert len(diagnostic['manual_review_blockers']) == 5
    with pytest.raises(ValidationError, match='人工|核验|未评估'):
        require_policy_application(baseline, compliant, .1, str(date.today()))
    violation = check_policy(baseline, {'growth': .95, 'cash': .05}, .1, str(date.today()))
    assert violation['within_limits'] is False
    assert any('现金用途' in text for text in violation['violations'])

"""Offline actual CMA -> unique search -> funding -> saved TAA policy integration."""
import copy
from datetime import date, timedelta

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError as InputError

from backend.tests.test_strategic_allocation import workspace, definition, confirmed_mandate, warm
from backend.strategic_allocation.contracts import CmaRequest, MandateRequest, PolicyRequest, PublishCmaRequest, PublishPolicyRequest
from backend.strategic_allocation import cma_model_kernels
from backend.strategic_allocation.policy_gate import check_policy
from backend.strategic_allocation.routes import build_router
from backend.custom_indicators.errors import ConflictError, ValidationError


def model_request(method='black_litterman'):
    raw = definition().model_dump(mode='json')
    for asset in raw['assets']:
        asset.pop('annual_return'); asset.pop('annual_volatility')
    raw.pop('correlation')
    common = dict(method=method, asset_ids=['股票', '债券'], as_of=str(date.today()), currency='CNY',
                  return_basis='annual_arithmetic_total_return', source='显式研究模型来源')
    if method == 'black_litterman':
        common.update(covariance=[[.04, .001], [.001, .01]], risk_covariance_basis='input_covariance',
                      market_weights={'股票': .6, '债券': .4}, market_weight_source='人工确认市场权重',
                      delta=3., tau=.05, risk_free_rate=.02, views=[])
    else:
        common.update(risk_mode='shared', shared_covariance=[[.01, 0.], [0., .0025]], scenarios=[
            dict(id='增长', probability=.5, annual_returns={'股票': .25, '债券': .01}, source='增长情景研究'),
            dict(id='衰退', probability=.5, annual_returns={'股票': -.15, '债券': .07}, source='衰退情景研究')])
    raw['model'] = common
    return raw


def save(service, raw):
    request = CmaRequest.model_validate(raw)
    preview = service.preview_cma(request)
    return service.publish_cma(PublishCmaRequest(request=request, preview_hash=preview['preview_hash']))


def mandate(service, funding=False):
    args = dict(name='模型集成目标', as_of=date.today(), review_date=date.today()+timedelta(days=90),
                max_volatility=.8, max_tracking_error=.5)
    if funding:
        args.update(objective_kind='funding_goal', funding_plan=dict(total_capital=100000., terminal_target=10000.,
            required_probability=.5, liquidity_months=12, flows=[]))
    return confirmed_mandate(service, MandateRequest(**args))


def policy(service, m, c, **kwargs):
    request = PolicyRequest(mandate_id=m['id'], cma_id=c['id'], candidate_count=300, **kwargs)
    return request, service.preview_policy(request)


def test_bl_prior_and_relative_view_change_actual_policy_and_freeze_lineage(workspace):
    service, _ = workspace
    assert service.warm()['cma_models']['complete']
    raw = model_request()
    before = copy.deepcopy(raw)
    preview = service.preview_cma(CmaRequest.model_validate(raw))
    assert not service.artifacts.root.exists()
    np.testing.assert_allclose(preview['effective_returns'], [.0932, .0338])
    cma = save(service, raw)
    assert raw == before
    m = mandate(service)
    req, prior = policy(service, m, cma)
    raw['model']['views'] = [dict(kind='relative', asset_id='股票', relative_to='债券', annual_return=-.2,
        view_std=.001, observed_on=str(date.today()), available_on=str(date.today()), source='显式相对观点研究')]
    changed = save(service, raw)
    _, posterior = policy(service, m, changed)
    assert prior['candidates'][1]['weights'] != posterior['candidates'][1]['weights']
    assert posterior['assumptions']['assets'][0]['annual_return'] == changed['effective_returns'][0]
    assert service.get_cma(cma['id']) == cma
    assert cma['definition']['assets'][0]['annual_return'] is None
    assert cma['effective_assumptions']['assets'][0]['mean_uncertainty'] == .03
    arrays = service.artifacts.arrays(cma['id'])
    np.testing.assert_array_equal(arrays['effective_returns'], cma['effective_returns'])
    assert not arrays['effective_returns'].flags.writeable
    saved = service.publish_policy(PublishPolicyRequest(request=req, preview_hash=prior['preview_hash'],
        candidate_id='nominal-utility', name='冻结模型政策', reason='验证模型实际被政策采用'))
    assert saved['policy']['raw_assumptions'] == cma['definition']
    assert saved['policy']['model_result'] == cma['model_result']
    assert saved['policy']['assumptions'] == cma['effective_assumptions']


def test_scenario_total_covariance_feeds_funding_and_taa(workspace):
    service, _ = workspace
    service.warm()
    cma = save(service, model_request('scenario_mixture'))
    expected = np.array([[.05, -.006], [-.006, .0034]])
    np.testing.assert_allclose(cma['covariance'], expected)
    m = mandate(service, funding=True)
    req, result = policy(service, m, cma)
    candidate = result['candidates'][1]
    w = np.array(list(candidate['weights'].values()))
    assert candidate['metrics']['volatility'] == pytest.approx(np.sqrt(w @ expected @ w))
    assert candidate['metrics']['expected_return'] == pytest.approx(w @ np.array([.05, .04]))
    from backend.strategic_allocation.planning import diagnose_funding
    reference = copy.deepcopy(result['candidates'])
    diagnose_funding(m['definition'], reference, paths=2000, seed=42)
    assert reference[1]['goal_check'] == candidate['goal_check']
    saved = service.publish_policy(PublishPolicyRequest(request=req, preview_hash=result['preview_hash'],
        candidate_id='nominal-utility', name='情景风险政策', reason='包含情景之间均值差异的风险'))
    check = check_policy(saved, {'股票': .8, '债券': .2}, .5, str(date.today()))
    assert check['expected_volatility'] == pytest.approx(np.sqrt(np.array([.8,.2]) @ expected @ np.array([.8,.2])))
    assert check['expected_return'] == pytest.approx(.048)


@pytest.mark.parametrize('patch', [
    lambda r: r['model'].update(asset_ids=['债券', '股票']),
    lambda r: r['model'].update(as_of=str(date.today()-timedelta(days=1))),
    lambda r: r['model'].update(currency='USD'),
    lambda r: r['model'].update(return_basis='geometric'),
    lambda r: r.update(risk_origin='historical_reference'),
])
def test_context_rejected_at_parent(patch):
    raw = model_request(); patch(raw)
    with pytest.raises(InputError): CmaRequest.model_validate(raw)


def test_manual_still_requires_numbers():
    raw = model_request(); raw.pop('model')
    with pytest.raises(InputError): CmaRequest.model_validate(raw)


def test_recompute_stale_hash_and_preview_no_files(workspace):
    service, _ = workspace; service.warm()
    raw = model_request()
    preview = service.preview_cma(CmaRequest.model_validate(raw))
    raw['model']['delta'] = 4.
    with pytest.raises(ConflictError):
        service.publish_cma(PublishCmaRequest(request=CmaRequest.model_validate(raw), preview_hash=preview['preview_hash']))
    assert not service.artifacts.root.exists()


def test_model_pid_readiness_and_domain_error(workspace, monkeypatch):
    service, _ = workspace; service.warm()
    monkeypatch.setattr(cma_model_kernels, '_WARMED_PID', None)
    with pytest.raises(RuntimeError): service.preview_cma(CmaRequest.model_validate(model_request()))
    service.warm()
    import backend.strategic_allocation.cma_application as application
    def singular(*args, **kwargs): raise np.linalg.LinAlgError('sensitive internal solver detail')
    monkeypatch.setattr(application, 'evaluate_cma_model', singular)
    app = FastAPI(); app.include_router(build_router(service))
    response = TestClient(app).post('/api/strategic-allocation/cma/preview', json=model_request())
    assert response.status_code == 422
    assert 'sensitive' not in response.text
    assert '模型' in response.json()['detail']['message']


def test_budget_optional_fifth_and_publish(workspace):
    service, _ = workspace; service.warm()
    cma = save(service, model_request()); m = mandate(service)
    _, original = policy(service, m, cma)
    req, budget = policy(service, m, cma, risk_budget={'股票': .3, '债券': .7})
    assert len(original['candidates']) == 4 and len(budget['candidates']) == 5
    assert original['candidates'] == budget['candidates'][:4]
    fifth = budget['candidates'][4]
    assert fifth['id'] == 'risk-budget' and fifth['available']
    assert fifth['risk_budget_distance'] == pytest.approx(sum((fifth['risk_contributions'][a]-b)**2 for a,b in req.risk_budget.items()))
    saved = service.publish_policy(PublishPolicyRequest(request=req, preview_hash=budget['preview_hash'],
        candidate_id='risk-budget', name='风险预算候选', reason='采用有限搜索的风险贡献匹配'))
    assert saved['policy']['selection']['id'] == 'risk-budget'
    with pytest.raises(ValidationError): policy(service, m, cma, risk_budget={'未知': 1.})


def test_budget_zero_variance_unavailable_without_fake_weights(workspace):
    service, _ = workspace; service.warm()
    raw = definition().model_dump(mode='json')
    for a in raw['assets']: a['annual_volatility'] = .1
    raw['correlation'] = [[1.,-1.],[-1.,1.]]
    cma = save(service, raw); m = mandate(service)
    req, result = policy(service, m, cma, risk_budget={'股票': .5,'债券':.5},
        constraints={a: dict(min_weight=.5, max_weight=.5) for a in ['股票','债券']})
    fifth = result['candidates'][4]
    assert fifth['available'] is False and not fifth['weights'] and fifth['unavailable_reason']
    with pytest.raises(ValidationError):
        service.publish_policy(PublishPolicyRequest(request=req, preview_hash=result['preview_hash'],
            candidate_id='risk-budget', name='不可用候选', reason='零方差不能发布风险预算'))


@pytest.mark.parametrize('budget', [{'股票': -.1,'债券':1.1}, {'股票': .3,'债券':.3}, {'股票':True}, {'股票':float('nan')}])
def test_budget_contract(budget):
    with pytest.raises(InputError): PolicyRequest(mandate_id='m', cma_id='c', risk_budget=budget)


def test_old_artifact_no_recompute(workspace, monkeypatch):
    service, _ = workspace; service.warm()
    cma = save(service, definition().model_dump(mode='json')); m = mandate(service)
    import backend.strategic_allocation.cma_application as application
    monkeypatch.setattr(application, 'evaluate_cma_model', lambda *a, **k: pytest.fail('old artifact recomputed'))
    _, preview = policy(service, m, cma)
    assert len(preview['candidates']) == 4 and service.get_cma(cma['id']) == cma


def test_fifth_retains_funding_gate_and_stale_budget_hash(workspace):
    service, _ = workspace; service.warm()
    cma = save(service, model_request())
    m = confirmed_mandate(service, MandateRequest(name='困难资金目标', as_of=date.today(),
        review_date=date.today()+timedelta(days=90), max_volatility=.8, objective_kind='funding_goal',
        funding_plan=dict(total_capital=100000., terminal_target=1e12, required_probability=.99, flows=[])))
    req, result = policy(service, m, cma, risk_budget={'股票': .5, '债券': .5})
    assert len(result['candidates']) == 5
    assert result['candidates'][4]['goal_check']['within_limits'] is False
    with pytest.raises(ValidationError, match='门槛'):
        service.publish_policy(PublishPolicyRequest(request=req, preview_hash=result['preview_hash'],
            candidate_id='risk-budget', name='门禁测试', reason='风险预算也不可绕过资金目标'))
    req.risk_budget = {'股票': .3, '债券': .7}
    with pytest.raises(ConflictError):
        service.publish_policy(PublishPolicyRequest(request=req, preview_hash=result['preview_hash'],
            candidate_id='risk-budget', name='过期风险预算', reason='更改预算后必须重新比较'))


def test_policy_reuses_readonly_frozen_arrays_no_model_recompute_or_preview_writes(workspace, monkeypatch):
    service, _ = workspace; service.warm()
    cma = save(service, model_request()); m = mandate(service)
    from backend.strategic_allocation.cma_application import frozen_numeric_inputs
    arrays = frozen_numeric_inputs(cma, service.artifacts)
    assert all(isinstance(a, np.memmap) and not a.flags.writeable for a in arrays)
    import backend.strategic_allocation.cma_application as application
    monkeypatch.setattr(application, 'evaluate_cma_model', lambda *a, **k: pytest.fail('saved model recomputed'))
    before = {str(p): p.read_bytes() for p in service.artifacts.root.parent.parent.rglob('*') if p.is_file()}
    policy(service, m, cma)
    after = {str(p): p.read_bytes() for p in service.artifacts.root.parent.parent.rglob('*') if p.is_file()}
    assert before == after


def test_policy_gate_rejects_mixed_raw_effective_model_lineage(workspace):
    service, _ = workspace; service.warm()
    cma = save(service, model_request()); m = mandate(service)
    req, preview = policy(service, m, cma)
    saved = service.publish_policy(PublishPolicyRequest(request=req, preview_hash=preview['preview_hash'],
        candidate_id='nominal-utility', name='血缘校验政策', reason='冻结原始和有效结果一致性'))
    saved['policy']['assumptions']['assets'][0]['annual_return'] = .9
    with pytest.raises(ValidationError, match='不一致'):
        check_policy(saved, {'股票': .5, '债券': .5}, .5, str(date.today()))


@pytest.mark.parametrize('day', [None, '', '2099-01-01'])
def test_model_research_clock_must_be_known_and_not_future(day):
    raw = model_request(); raw['as_of'] = day
    with pytest.raises(InputError): CmaRequest.model_validate(raw)


def test_api_readiness_fails_closed_and_model_request_never_warms(workspace, monkeypatch):
    service, _ = workspace; service.warm()
    app = FastAPI(); app.include_router(build_router(service))
    monkeypatch.setattr(cma_model_kernels, 'warm', lambda: pytest.fail('request warming'))
    signatures = [list(k.signatures) for k in cma_model_kernels.KERNELS]
    assert TestClient(app).post('/api/strategic-allocation/cma/preview', json=model_request()).status_code == 200
    assert signatures == [list(k.signatures) for k in cma_model_kernels.KERNELS]
    monkeypatch.setattr(cma_model_kernels, '_WARMED_PID', -1)
    response = TestClient(app).post('/api/strategic-allocation/cma/preview', json=model_request())
    assert response.status_code == 503 and response.json()['detail']['code'] == 'SAA_NOT_READY'


def test_model_strategic_first_preserves_unmapped_assets_and_cash_gate(tmp_path):
    from backend.tests.test_bettersaataa_m1 import universe, cma_request, context
    from backend.strategic_allocation.service import StrategicAllocationService
    service = StrategicAllocationService(tmp_path/'research', tmp_path/'no-market')
    service.warm()
    scope = universe(service)
    raw = cma_request(scope).model_dump(mode='json')
    raw['model'] = dict(method='black_litterman', asset_ids=['equity','cash'], as_of=str(date.today()),
        currency='CNY', source='独立战略风险研究', covariance=[[.04,0.],[0.,.0001]],
        risk_covariance_basis='input_covariance', market_weights={'equity':.6,'cash':.4},
        market_weight_source='明确市场权重假设', delta=3., tau=.05, risk_free_rate=.02, views=[])
    for a in raw['assets']: a.pop('annual_return'); a.pop('annual_volatility')
    raw.pop('correlation')
    cma = save(service, raw)
    m = confirmed_mandate(service, MandateRequest(name='独立战略模型目标', as_of=date.today(),
        review_date=date.today()+timedelta(days=90), strategic_universe_id=scope['id'],
        institutional_context=context(), max_volatility=.8))
    req, preview = policy(service, m, cma, risk_budget={'equity':.6,'cash':.4})
    assert len(preview['candidates']) == 5
    assert all(c['weights']['cash'] >= .4-1e-8 for c in preview['candidates'])
    assert preview['current_application_eligible'] is False
    assert not service.data.data_dir.exists()
    saved = service.publish_policy(PublishPolicyRequest(request=req, preview_hash=preview['preview_hash'],
        candidate_id='risk-budget', name='有实施缺口的模型政策', reason='保留全部战略资产且不制造代理净值'))
    assert [a['id'] for a in saved['assets']] == ['equity','cash']
    assert all(not a['products'] for a in saved['assets'])
    from backend.strategic_allocation.policy_gate import require_policy_application
    with pytest.raises(ValidationError):
        require_policy_application(saved, {a['id']:a['base_weight'] for a in saved['assets']}, .1, str(date.today()))

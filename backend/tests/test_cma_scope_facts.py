"""Equivalent saved scopes must remain usable without weakening real boundaries."""
import copy
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

import numpy as np
import pytest

from backend.custom_indicators.errors import ValidationError
from backend.strategic_allocation.contracts import CmaRequest, MandateRequest, PolicyRequest
from backend.strategic_allocation.cma_center_contracts import CmaListItem, CmaRetire
from backend.strategic_allocation.scope_facts import scope_facts, scope_fingerprint, scope_difference
from backend.strategic_allocation.universe_contracts import UniverseRequest, ConfirmUniverseRequest
from backend.tests.test_strategic_allocation import workspace, warm, confirmed_mandate
from backend.tests.test_ltcma_statistics import statistics_warm, request, publish
from backend.tests.test_multi_cma import warm_multi, adopt


def confirm_scope(service, definition, previous=None):
    req = UniverseRequest.model_validate(definition)
    preview = service.scopes.preview_universe(req)
    return service.scopes.confirm_universe(ConfirmUniverseRequest(
        request=req, preview_hash=preview['preview_hash'], replaces_universe_id=previous))


@pytest.fixture
def scope_case(workspace, monkeypatch):
    service, days = workspace
    assets = [{"id": key, "name": key, "currency": "CNY", "role": role, "liquidity": "liquid",
               "research_proxy": {"asset_type": "market", "cash_return": None, "rebalance": "daily", "components": [
                   {"kind": "index", "series_id": f"index:index_daily:{code}", "field": "close", "weight": 1.}]}}
              for key, role, code in [('equity', 'growth', '000300.SH'), ('bond', 'rates', '000012.SH')]]
    scope = confirm_scope(service, {"name": "范围", "as_of": str(date.today()), "currency": "CNY", "assets": assets})
    x = np.arange(len(days) - 1)
    returns = np.column_stack((.0004 + .008 * np.sin(x), .0001 + .002 * np.cos(.7*x)))
    levels = np.vstack((np.ones(2), np.cumprod(1+returns, axis=0)))
    def load(component, _request):
        values = levels[:, 0 if '000300' in component.series_id else 1].copy()
        values.flags.writeable = False
        return {"dates": [str(d.date()) for d in days], "values": values,
                "available_at": [str(d.date()) for d in days],
                "identity": {"name": component.series_id, "series_id": component.series_id}}
    monkeypatch.setattr(service.cma.evidence.sources, 'load', load)
    raw = request().model_dump(mode='json')
    raw.update(alloc_name=None, strategic_universe_id=scope['id'])
    raw['assets'] = [{**a, 'id': key} for a, key in zip(raw['assets'], ['equity', 'bond'])]
    raw['model'].update(asset_ids=['equity', 'bond'], proxy_inputs={
        'name': '代理', 'as_of': str(date.today()), 'currency': 'CNY', 'calendar': 'SSE', 'frequency': 'daily',
        'periods_per_year': 252, 'return_basis': 'selected_index_and_adjusted_product_total_return',
        'fee_basis': 'source_embedded_no_additional_fee', 'fx_basis': 'same_currency_no_conversion',
        'assets': [{"id": a['id'], 'name': a['name'], 'rationale': '', **a['research_proxy']} for a in assets]})
    return service, scope, raw


def test_unchanged_save_is_atomic_and_does_not_create_a_record(scope_case):
    service, scope, _ = scope_case
    original = service.artifacts.get(scope['id'])
    count = len(service.artifacts.list())
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: confirm_scope(service, scope['definition'], scope['id']), range(4)))
    assert {r['id'] for r in results} == {scope['id']}
    assert len(service.artifacts.list()) == count
    assert service.artifacts.get(scope['id']) == original


def test_equivalent_scopes_work_through_niw_publication_and_multi_saa(scope_case):
    service, scope, raw = scope_case
    prior = publish(service, CmaRequest.model_validate(raw), 'facts-prior-original')
    frozen = service.artifacts.get(prior['id'])
    # A real annotation edit is saved, while the economic configuration is unchanged.
    updated = confirm_scope(service, {**scope['definition'], 'source': '补充研究说明'}, scope['id'])
    assert updated['id'] != scope['id']
    assert updated['scope_fingerprint'] == scope['scope_fingerprint']
    raw['strategic_universe_id'] = updated['id']
    second = publish(service, CmaRequest.model_validate(raw), 'facts-prior-new-scope')
    listed = service.cma.list()['items']
    assert listed[0]['scope_facts'] == listed[1]['scope_facts']
    assert listed[0]['scope_fingerprint'] == listed[1]['scope_fingerprint']
    assert all(CmaListItem.model_validate(item) for item in listed)
    raw['model'].update(method='bayesian_niw', prior_ref={'id': prior['id'], 'content_hash': prior['content_hash']},
                        mean_prior_observations=20., covariance_prior_observations=30., data_reuse_acknowledged=True)
    raw['model'].pop('shrinkage', None)
    posterior = publish(service, CmaRequest.model_validate(raw), 'facts-niw-new-scope')
    assert posterior['model_result']['model_audit']['niw_posterior']['kappa'] == 180.
    mandate = confirmed_mandate(service, MandateRequest(name='配置事实目标', as_of=date.today(),
        review_date=date.today()+timedelta(days=90), target_return=0., max_volatility=.2, max_tracking_error=.1))
    for mode in ['single', 'parameter_average']:
        args = {'cma_id': prior['id']} if mode == 'single' else {'cma_refs': [
            {'cma_id': item['id'], 'content_hash': item['content_hash'], 'weight': .5} for item in [prior, second]]}
        policy = PolicyRequest(mandate_id=mandate['id'], mode=mode, candidate_count=300, **args)
        preview = service.preview_policy(policy)
        saved = adopt(service, policy, preview)
        assert saved['policy']['cma_id'] == (prior['id'] if mode == 'single' else None)
        if mode != 'single':
            assert [s['artifact']['definition']['strategic_universe_id'] for s in saved['policy']['multi_cma']['sources']] == [scope['id'], updated['id']]
    assert service.artifacts.get(prior['id']) == frozen


@pytest.mark.parametrize('change,reason', [
    (lambda d: d.update(currency='USD'), 'scopeCurrency'),
    (lambda d: d['assets'].reverse(), 'scopeAssets'),
    (lambda d: d['assets'][0].update(role='credit'), 'scopeRoles'),
    (lambda d: d['assets'][0].update(liquidity='illiquid'), 'scopeLiquidity'),
    (lambda d: d['assets'][0]['research_proxy']['components'][0].update(weight=.9), 'scopeProxyWeights'),
    (lambda d: d['assets'][0]['research_proxy'].update(rebalance='monthly'), 'scopeRebalance'),
])
def test_configuration_changes_are_not_erased(scope_case, change, reason):
    _, scope, _ = scope_case
    modified = copy.deepcopy(scope['definition'])
    change(modified)
    assert scope_difference(scope_facts(scope['definition']), scope_facts(modified)) == reason
    assert scope_fingerprint(scope['definition']) != scope_fingerprint(modified)


def test_real_proxy_change_is_rejected_by_backend_even_with_same_scope_id(scope_case):
    service, scope, raw = scope_case
    prior = publish(service, CmaRequest.model_validate(raw), 'facts-proxy-prior')
    raw['model'].update(method='bayesian_niw', prior_ref={'id': prior['id'], 'content_hash': prior['content_hash']},
                        mean_prior_observations=20., covariance_prior_observations=30., data_reuse_acknowledged=True)
    raw['model'].pop('shrinkage', None)
    raw['model']['proxy_inputs']['assets'][0]['rebalance'] = 'monthly'
    with pytest.raises(ValidationError, match='研究代理'):
        service.preview_cma(CmaRequest.model_validate(raw))
    raw['model']['proxy_inputs']['assets'][0]['rebalance'] = 'daily'
    service.cma.retire(prior['id'], CmaRetire(confirm=True, content_hash=prior['content_hash'], reason='停止新研究引用'))
    with pytest.raises(ValidationError, match='停止新引用'):
        service.preview_cma(CmaRequest.model_validate(raw))
    assert service.cma.study_options()['assumptions'][0]['retired'] is True


def test_niw_rejects_other_fee_basis_and_same_named_changed_scope(scope_case):
    service, scope, raw = scope_case
    original = publish(service, CmaRequest.model_validate(raw), 'facts-rejection-original')
    raw['model'].update(method='bayesian_niw', prior_ref={'id': original['id'], 'content_hash': original['content_hash']},
                        mean_prior_observations=20., covariance_prior_observations=30., data_reuse_acknowledged=True)
    raw['model'].pop('shrinkage', None)
    raw['fee_basis'] = 'explicit_assumption'
    with pytest.raises(ValidationError, match='费用或汇率口径不同'):
        service.preview_cma(CmaRequest.model_validate(raw))
    raw['fee_basis'] = original['definition']['fee_basis']
    changed = copy.deepcopy(scope['definition'])
    changed['assets'][0]['research_proxy']['rebalance'] = 'monthly'
    updated = confirm_scope(service, changed, scope['id'])
    raw['strategic_universe_id'] = updated['id']
    # 范围的经济配置已改版：先验钉住旧范围，按版本规则先要求更新先验。
    with pytest.raises(ValidationError, match='已有新版本'):
        service.preview_cma(CmaRequest.model_validate(raw))


def test_bound_mandate_accepts_equal_facts_but_not_real_scope_changes(scope_case):
    service, scope, raw = scope_case
    current = confirm_scope(service, {**scope['definition'], 'source': '补充说明'}, scope['id'])
    mandate = {'strategic_universe_id': current['id'], 'asset_limits': {'equity': {'max_weight': .6}},
               'max_tracking_error': .1, 'max_volatility': .3, 'min_liquid_weight': 0., 'max_illiquid_weight': 1.}
    _, limits = service._constraints(PolicyRequest(mandate_id='m', cma_id='c'), raw, mandate)
    assert limits['equity']['max_weight'] == .6
    changed = copy.deepcopy(current['definition'])
    changed['assets'][0]['liquidity'] = 'illiquid'
    different = confirm_scope(service, changed, current['id'])
    with pytest.raises(ValidationError, match='流动性'):
        service._constraints(PolicyRequest(mandate_id='m', cma_id='c'), raw, {**mandate, 'strategic_universe_id': different['id']})

"""M1 acceptance uses isolated files only, including true no-products research."""
import copy
import json
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError as InputError

from backend.custom_indicators.errors import ConflictError, ValidationError
from backend.strategic_allocation import institution_kernels as numeric
from backend.strategic_allocation.contracts import (
    CmaRequest, MandateRequest, MandateStudyRequest, ConfirmMandateRequest,
    PublishCmaRequest, PolicyRequest, PublishPolicyRequest,
)
from backend.strategic_allocation.institution_contracts import InstitutionalContext, REVIEW_TOPICS
from backend.strategic_allocation.policy_gate import check_policy, require_policy_application
from backend.strategic_allocation.routes import build_router
from backend.strategic_allocation.service import StrategicAllocationService
from backend.strategic_allocation.universe_contracts import (
    UniverseRequest, ConfirmUniverseRequest, ImplementationMapRequest, ConfirmImplementationMapRequest,
)
from backend.tactical_allocation.numeric import warm_tactical_allocation_kernels

TODAY = str(date.today())
LATER = str(date.today() + timedelta(days=90))


@pytest.fixture(scope='module', autouse=True)
def warmed(tmp_path_factory):
    service = StrategicAllocationService(tmp_path_factory.mktemp('m1-warm'), tmp_path_factory.mktemp('m1-market'))
    assert service.warm()['institution']['complete']
    warm_tactical_allocation_kernels()


@pytest.fixture
def service(tmp_path):
    return StrategicAllocationService(tmp_path / 'research', tmp_path / 'market', universe_dir=tmp_path / 'domain')


def universe_request():
    return UniverseRequest(name='无产品战略', as_of=TODAY, currency='CNY', source='研究员确认的经济风险范围', assets=[
        {'id': 'equity', 'name': '股票', 'currency': 'CNY', 'role': 'growth', 'liquidity': 'liquid', 'rationale': '长期资本增长', 'source': '研究定义一'},
        {'id': 'cash', 'name': '现金', 'currency': 'CNY', 'role': 'liquidity', 'liquidity': 'liquid', 'rationale': '必要现金储备', 'source': '研究定义二'}])


def universe(service):
    request = universe_request()
    preview = service.scopes.preview_universe(request)
    return service.scopes.confirm_universe(ConfirmUniverseRequest(request=request, preview_hash=preview['preview_hash']))


def cma_request(scope, mapping=None):
    return CmaRequest(name='纯前瞻CMA', strategic_universe_id=scope['id'], implementation_mapping_id=mapping,
        as_of=TODAY, source='明确的前瞻研究假设，不使用历史产品净值', basis_confirmed=True,
        assets=[{'id': a['id'], 'role': a['role'], 'liquidity': a['liquidity'], 'rationale': a['rationale'],
                 'annual_return': r, 'annual_volatility': vol, 'mean_uncertainty': .005}
                for a, r, vol in zip(scope['definition']['assets'], [.06, .02], [.15, .01])], correlation=[[1., 0.], [0., 1.]])


def context(checked=False):
    ctx = InstitutionalContext(investor_type='corporate_treasury', purpose='明确的企业储备资金用途', cash_reserve_weight=.4).model_dump(mode='json')
    if checked:
        for item in ctx['review_items']:
            item.update(status='researcher_checked', reason='研究员核查适用条件', evidence='离线测试核查记录', reviewed_on=TODAY, valid_until=LATER)
    return ctx


def policy(service, scope, mapping=None, ctx=None):
    mandate = MandateRequest(name='目标研究', as_of=TODAY, review_date=LATER,
        strategic_universe_id=scope['id'], institutional_context=ctx, max_tracking_error=.1,
        max_volatility=.2, boundary_reason='必要现金与风险容量的研究边界')
    study = MandateStudyRequest(definition=mandate)
    preview = service.preview_mandate(study)
    saved_mandate = service.confirm_mandate(ConfirmMandateRequest(request=study, preview_hash=preview['preview_hash'], acknowledge_limits=True))
    cma = cma_request(scope, mapping)
    preview = service.preview_cma(cma)
    saved_cma = service.publish_cma(PublishCmaRequest(request=cma, preview_hash=preview['preview_hash']))
    request = PolicyRequest(mandate_id=saved_mandate['id'], cma_id=saved_cma['id'], candidate_count=250)
    preview = service.preview_policy(request)
    baseline = service.publish_policy(PublishPolicyRequest(request=request, preview_hash=preview['preview_hash'],
        candidate_id='nominal-utility', name='战略政策', reason='研究员比较后确认的政策研究'))
    return preview, baseline


def test_true_no_products_complete_forward_research_and_taa_refusal(service):
    scope = universe(service)
    preview, baseline = policy(service, scope, ctx=context())
    assert not service.data.data_dir.exists()
    assert not service.data.universe_dir.exists()
    assert baseline['alloc_name'] is None
    assert [a['id'] for a in baseline['assets']] == ['equity', 'cash']
    assert all(not a['products'] for a in baseline['assets'])
    assert len(preview['candidates']) == 4
    assert preview['current_application_eligible'] is False
    assert any(g['id'] == 'policy-cash-reserve' and g['lo'] == .4 for g in baseline['group_limits'])
    assert all(c['weights']['cash'] >= .4 - 1e-8 for c in preview['candidates'])
    with pytest.raises(ValidationError) as rejected:
        service.data.load_data(baseline, '2024-01-01', '2024-01-06', TODAY)
    assert rejected.value.code == 'SAA_IMPLEMENTATION_INCOMPLETE'
    with pytest.raises(ValidationError):
        service.data.validate_application(baseline)
    assert service.baselines.get_baseline(baseline['id']) == baseline
    catalog = service.catalog()
    assert catalog['allocations'] == []
    assert catalog['strategic_universes'][0]['id'] == scope['id']
    assert catalog['assumptions'][0]['strategic_universe_id'] == scope['id']


def test_balance_snapshot_missing_commitments_not_counted_and_negative_net_allowed(service):
    ctx = context()
    ctx['balance_sheet'] = {'as_of': TODAY, 'currency': 'CNY', 'source': '测试研究快照',
        'investable_assets': 100., 'outside_assets': 20., 'confirmed_liabilities': 150., 'uncalled_commitments': 999.}
    mandate = MandateRequest(name='经济诊断', as_of=TODAY, review_date=LATER, institutional_context=ctx)
    result = service.preview_mandate(MandateStudyRequest(definition=mandate))
    sheet = result['institutional_diagnostics']['balance_sheet']
    assert sheet['total_assets'] == 120. and sheet['net_assets_after_confirmed_liabilities'] == -30.
    assert sheet['uncalled_commitments'] == 999.
    assert len(result['institutional_diagnostics']['review_blockers']) == 5
    assert not service.artifacts.root.exists()
    ctx['balance_sheet']['outside_assets'] = None
    result = service.preview_mandate(MandateStudyRequest(definition=mandate.model_copy(update={'institutional_context': InstitutionalContext.model_validate(ctx)})))
    assert result['institutional_diagnostics']['balance_sheet']['total_assets'] is None
    assert result['institutional_diagnostics']['balance_sheet']['net_assets_after_confirmed_liabilities'] is None


@pytest.mark.parametrize('value', [-1., float('inf'), float('nan'), True])
def test_institution_rejects_invalid_numeric(value):
    with pytest.raises(InputError):
        InstitutionalContext(investor_type='corporate_treasury', purpose='现金储備', cash_reserve_weight=value)
    with pytest.raises(InputError):
        InstitutionalContext(investor_type='corporate_treasury', purpose='现金储備', balance_sheet={
            'as_of': TODAY, 'currency': 'CNY', 'source': '离线来源', 'investable_assets': value})


def test_review_contract_and_balance_basis_cannot_be_omitted_or_backdated():
    ctx = context()
    ctx['review_items'] = ctx['review_items'][:-1]
    with pytest.raises(InputError):
        InstitutionalContext.model_validate(ctx)
    ctx = context(); ctx['review_items'][0]['status'] = 'researcher_checked'
    with pytest.raises(InputError):
        InstitutionalContext.model_validate(ctx)
    for day, currency in [('2020-01-01', 'CNY'), (TODAY, 'USD')]:
        ctx = context(); ctx['balance_sheet'] = {'as_of': day, 'currency': currency, 'source': '真实研究来源'}
        with pytest.raises(InputError):
            MandateRequest(name='研究目标', as_of=TODAY, review_date=LATER, institutional_context=ctx)


def test_readonly_strided_numeric_reference_and_readiness(service, monkeypatch):
    owner = np.array([100., -99., 40., -99., 200., -99., 600., -99.])
    owner.flags.writeable = False
    view = owner[::2]; before = owner.copy()
    signatures = [tuple(k.signatures) for k in numeric.KERNELS]
    assert np.shares_memory(owner, view)
    np.testing.assert_array_equal(numeric.balance_sheet_kernel(view), [140., -60.])
    np.testing.assert_array_equal(numeric.balance_sheet_kernel(np.full(4, np.nan)), [np.nan, np.nan])
    mask = np.array([True, False, False, False]); mask.flags.writeable = False
    assert numeric.cash_weight_kernel(np.array([.3, .7]), mask[::2]) == .3
    for bad in [np.ones(0), np.ones(3), np.array([1., 2., 3., np.inf])]:
        with pytest.raises(ValueError):
            numeric.balance_sheet_kernel(bad)
    np.testing.assert_array_equal(before, owner)
    assert signatures == [tuple(k.signatures) for k in numeric.KERNELS]
    assert numeric.execution_audit()['python_fallback'] == 0
    monkeypatch.setattr(numeric, '_WARMED_PID', None)
    mandate = MandateRequest(name='未预热', as_of=TODAY, review_date=LATER, institutional_context=context())
    with pytest.raises(RuntimeError, match='预热'):
        service.preview_mandate(MandateStudyRequest(definition=mandate))


def test_cash_check_independent_of_liquid_group_and_manual_review_research_vs_application(service):
    scope = universe(service)
    preview, baseline = policy(service, scope, ctx=context())
    check = check_policy(baseline, {'equity': .8, 'cash': .2}, .1, TODAY)
    assert not check['within_limits'] and check['cash_reserve_check']['minimum'] == .4
    weights = {a['id']: a['base_weight'] for a in baseline['assets']}
    check = check_policy(baseline, weights, .1, TODAY)
    assert check['within_limits'] and not check['current_application_eligible']
    with pytest.raises(ValidationError) as blocked:
        require_policy_application(baseline, weights, .1, TODAY)
    assert blocked.value.code == 'SAA_MANUAL_REVIEW_REQUIRED'
    baseline['policy']['mandate']['institutional_context'] = context(checked=True)
    with pytest.raises(ValidationError) as incomplete:
        require_policy_application(baseline, weights, .1, TODAY)
    assert incomplete.value.code == "SAA_IMPLEMENTATION_INCOMPLETE"
    baseline['policy']['mandate']['institutional_context']['review_items'][0]['valid_until'] = TODAY
    with pytest.raises(ValidationError):
        require_policy_application(baseline, weights, .1, TODAY)
    request = cma_request(scope).model_dump(mode='json')
    request['assets'][1]['role'] = 'rates'
    with pytest.raises(ValidationError, match='角色'):
        service.preview_cma(CmaRequest.model_validate(request))
    definition = copy.deepcopy(preview['assumptions']); definition['assets'][1]['role'] = 'rates'
    with pytest.raises(ValidationError) as no_cash:
        service._constraints(PolicyRequest(mandate_id='one', cma_id='two'), definition, preview['mandate'])
    assert no_cash.value.code == 'SAA_CASH_ASSETS_MISSING'


def test_scope_routes_are_pure_hash_checked_immutable_and_strict(service):
    app = FastAPI(); app.include_router(build_router(service)); client = TestClient(app)
    request = universe_request().model_dump(mode='json')
    response = client.post('/api/strategic-allocation/universes/preview', json=request)
    assert response.status_code == 200
    assert not service.artifacts.root.exists()
    assert not service.data.data_dir.exists()
    response2 = client.post('/api/strategic-allocation/universes/preview', json=request)
    assert response2.json() == response.json()
    body = {'request': request, 'preview_hash': response.json()['preview_hash']}
    bad = copy.deepcopy(body); bad['request']['name'] = '修改后的范围'
    assert client.post('/api/strategic-allocation/universes/confirm', json=bad).status_code == 409
    saved = client.post('/api/strategic-allocation/universes/confirm', json=body)
    assert saved.status_code == 201
    assert client.get('/api/strategic-allocation/universes/' + saved.json()['id']).json() == saved.json()
    duplicate = copy.deepcopy(request); duplicate['assets'].append(duplicate['assets'][0])
    assert client.post('/api/strategic-allocation/universes/preview', json=duplicate).status_code == 422
    request['assets'][0]['currency'] = 'USD'
    assert client.post('/api/strategic-allocation/universes/preview', json=request).status_code == 422


@pytest.fixture
def mapped(service):
    service.data.data_dir.mkdir(); service.data.universe_dir.mkdir()
    source_assets = [('proxy-bond', '511010.SH'), ('proxy-equity', '510300.SH')]
    pd.DataFrame([{'asset_alloc_name': '真实代理', 'asset_name': name, 'etf_code': code,
        'etf_name': '不能用名称匹配', 'etf_weight': 100., 'as_of': None, 'universe_snapshot_id': 'domain-one'}
        for name, code in source_assets]).to_parquet(service.data.data_dir / 'asset_alloc_info.parquet')
    days = pd.date_range('2024-01-01', periods=6)
    pd.DataFrame([{'asset_alloc_name': '真实代理', 'asset_name': name, 'date': day, 'as_of': None,
        'nv': (1 + rate) ** i, 'available_at': day} for (name, _), rate in zip(source_assets, [.001, .02])
        for i, day in enumerate(days)]).to_parquet(service.data.data_dir / 'asset_nv.parquet')
    snapshot = {'id': 'domain-one', 'name': '产品域', 'research_date': '2024-01-01', 'immutable': True,
        'members': [{'kind': 'etf', 'product_id': code, 'eligible': True} for _, code in source_assets]}
    (service.data.universe_dir / 'product_pools.json').write_text(json.dumps({'pools': [], 'versions': [], 'universe_snapshots': [snapshot]}))
    scope = universe(service)
    request = ImplementationMapRequest(name='明确代理映射', strategic_universe_id=scope['id'], alloc_name='真实代理',
        universe_snapshot_id='domain-one', as_of=TODAY, valid_until=LATER,
        assignments=[{'strategic_asset_id': 'equity', 'proxy_asset_id': 'proxy-equity', 'rationale': '显式研究代理'},
                     {'strategic_asset_id': 'cash', 'proxy_asset_id': 'proxy-bond', 'rationale': '测试映射非投资验证'}])
    return service, scope, request


def save_mapping(service, request):
    preview = service.scopes.preview_mapping(request)
    return service.scopes.confirm_mapping(ConfirmImplementationMapRequest(request=request, preview_hash=preview['preview_hash']))


def test_explicit_mapping_reorders_real_data_once_no_name_inference_or_mutation(mapped):
    service, scope, request = mapped
    before = sorted(p.relative_to(service.artifacts.root) for p in service.artifacts.root.rglob('*'))
    preview = service.scopes.preview_mapping(request)
    assert preview['implementation_status'] == 'complete'
    assert sorted(p.relative_to(service.artifacts.root) for p in service.artifacts.root.rglob('*')) == before
    mapping = save_mapping(service, request)
    _, baseline = policy(service, scope, mapping['id'])
    data = service.data.load_data(baseline, '2024-01-01', '2024-01-06', TODAY)
    np.testing.assert_allclose(data['returns'], np.tile([.02, .001], (5, 1)), atol=1e-14)
    assert not data['returns'].flags.writeable and not data['available_at'].flags.writeable
    assert data['lineage']['implementation_mapping_id'] == mapping['id']
    assert baseline['assets'][0]['products'][0]['product_id'] == '510300.SH'
    assert baseline['assets'][1]['products'][0]['product_id'] == '511010.SH'
    service.data.validate_application(baseline)
    unchanged = copy.deepcopy(baseline)
    service.data.load_data(baseline, '2024-01-01', '2024-01-06', TODAY)
    assert baseline == unchanged
    altered = copy.deepcopy(baseline); altered['assets'][0]['products'][0]['product_id'] = '511010.SH'
    with pytest.raises(ValidationError) as bad:
        service.data.validate_application(altered)
    assert bad.value.code == 'SAA_MAPPING_PRODUCTS'
    altered = copy.deepcopy(baseline); altered['implementation_mapping_snapshot']['definition']['assignments'].reverse()
    with pytest.raises(ValidationError) as bad:
        service.data.validate_application(altered)
    assert bad.value.code == 'SAA_MAPPING_INTEGRITY'


def test_expired_mapping_has_explicit_policy_preview_reason_without_rewriting_history(mapped, monkeypatch):
    service, scope, request = mapped
    mapping = save_mapping(service, request.model_copy(update={'valid_until': date.today() + timedelta(days=2)}))
    initial, baseline = policy(service, scope, mapping['id'])

    class LaterDate(date):
        @classmethod
        def today(cls):
            return date.today() + timedelta(days=3)

    monkeypatch.setattr('backend.strategic_allocation.service.date', LaterDate)
    preview = service.preview_policy(PolicyRequest.model_validate(baseline['policy']['selection_request']))
    assert not preview['current_application_eligible']
    assert any('实施映射尚未生效或已到复核日' in reason for reason in preview['application_blockers'])
    assert preview['candidates'] == initial['candidates']
    assert service.baselines.get_baseline(baseline['id']) == baseline


def test_mapping_partial_domain_duplicates_source_change_and_wrong_universe(mapped):
    service, scope, request = mapped
    partial = request.model_copy(update={'assignments': request.assignments[:1]})
    saved = save_mapping(service, partial)
    assert saved['implementation_gaps'] == ['cash']
    _, baseline = policy(service, scope, saved['id'])
    with pytest.raises(ValidationError) as error:
        service.data.load_data(baseline, '2024-01-01', '2024-01-06', TODAY)
    assert error.value.code == 'SAA_IMPLEMENTATION_INCOMPLETE'
    raw = request.model_dump(mode='json'); raw['assignments'][1]['proxy_asset_id'] = raw['assignments'][0]['proxy_asset_id']
    with pytest.raises(InputError):
        ImplementationMapRequest.model_validate(raw)
    with pytest.raises(ValidationError) as error:
        service.scopes.preview_mapping(request.model_copy(update={'universe_snapshot_id': 'another-domain'}))
    assert error.value.code == 'SAA_MAPPING_DOMAIN'
    other = universe(service)
    with pytest.raises(ValidationError) as error:
        service.preview_cma(cma_request(other, saved['id']))
    assert error.value.code == 'SAA_MAPPING_UNIVERSE'
    preview = service.scopes.preview_mapping(request)
    path = service.data.data_dir / 'asset_alloc_info.parquet'
    frame = pd.read_parquet(path); frame['etf_name'] = '代理来源已变化'; frame.to_parquet(path)
    with pytest.raises(ConflictError):
        service.scopes.confirm_mapping(ConfirmImplementationMapRequest(request=request, preview_hash=preview['preview_hash']))
    with pytest.raises(ConflictError):
        service.preview_cma(cma_request(scope, saved['id']))
    assert service.scopes.get_mapping(saved['id']) == saved


def test_mapping_route_contract_and_read_type(mapped):
    service, scope, request = mapped
    app = FastAPI(); app.include_router(build_router(service)); client = TestClient(app)
    body = request.model_dump(mode='json')
    preview = client.post('/api/strategic-allocation/implementation-maps/preview', json=body)
    assert preview.status_code == 200
    saved = client.post('/api/strategic-allocation/implementation-maps/confirm', json={'request': body, 'preview_hash': preview.json()['preview_hash']})
    assert saved.status_code == 201
    assert client.get('/api/strategic-allocation/implementation-maps/' + saved.json()['id']).json() == saved.json()
    assert client.get('/api/strategic-allocation/universes/' + saved.json()['id']).status_code == 422
    assert service.catalog()['implementation_maps'][0]['id'] == saved.json()['id']


def test_new_clock_application_is_also_checked_by_direct_product_bridge():
    from backend.tactical_allocation.portfolio_bridge import validate_decision_application
    preview = {'request': {'decision_policy': {'mode': 'scheduled'}},
               'application': {'eligible': False, 'reasons': ['等待实际持仓与执行机会']}}
    with pytest.raises(ValidationError) as blocked:
        validate_decision_application({'preview': preview}, None)
    assert blocked.value.code == 'TAA_EXECUTION_INELIGIBLE'


def test_service_readiness_includes_institutional_pid_state(service, monkeypatch):
    monkeypatch.setattr(numeric, 'warm', lambda: None)
    monkeypatch.setattr(numeric, '_WARMED_PID', None)
    with pytest.raises(RuntimeError, match='机构诊断启动预热'):
        service.warm()


def test_complete_mapping_enters_real_taa_service_and_keeps_frozen_scope(mapped):
    from backend.tactical_allocation.service import TacticalAllocationService
    from backend.tactical_allocation.contracts import PreviewRequest
    service, scope, map_request = mapped
    days = pd.bdate_range(end=date.today(), periods=161)
    returns = np.column_stack((.0004 + .002 * np.sin(np.arange(160)), .0001 + .0005 * np.cos(np.arange(160))))
    values = np.vstack((np.ones(2), np.cumprod(1 + returns, axis=0)))
    pd.DataFrame([{'asset_alloc_name': '真实代理', 'asset_name': name, 'date': day, 'as_of': None,
        'nv': values[t, i], 'available_at': day} for i, name in enumerate(['proxy-equity', 'proxy-bond'])
        for t, day in enumerate(days)]).to_parquet(service.data.data_dir / 'asset_nv.parquet')
    mapping = save_mapping(service, map_request)
    _, baseline = policy(service, scope, mapping['id'], ctx=context(checked=True))
    tactical = TacticalAllocationService(service.baselines.artifacts.root.parent.parent, service.data.data_dir,
                                        universe_dir=service.data.universe_dir)
    request = PreviewRequest(baseline_id=baseline['id'], start_date=days[0].date(), end_date=days[-1].date(),
        as_of=date.today(), train_end_date=days[100].date(), signal_mode='manual',
        manual_tilts={'equity': 0., 'cash': 0.}, search=False, max_tracking_error=.1)
    preview = tactical.preview(request)
    assert preview['baseline']['strategic_universe_id'] == scope['id']
    assert preview['data']['lineage']['implementation_mapping_hash'] == mapping['content_hash']
    assert set(preview['recommendation']['weights']) == {'equity', 'cash'}
    assert preview['policy_check']['cash_reserve_check']['weight'] >= .4 - 1e-8
    assert preview['policy_check']['manual_review_blockers'] == []
    np.testing.assert_allclose(tactical.data.load_data(baseline, str(days[0].date()), str(days[-1].date()), TODAY)['returns'], returns, atol=1e-14)

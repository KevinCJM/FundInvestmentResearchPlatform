"""Dedicated offline browser fixture, reusing the existing isolated strategic API.

All data and publication writes stay in a TemporaryDirectory. No production
singletons or live market APIs are imported by this fixture.
"""
import copy
from contextlib import asynccontextmanager
from datetime import date, timedelta
import json

from backend.tests.strategic_allocation_app import app, strategic, days
from backend.strategic_allocation.contracts import CmaRequest, MandateRequest, MandateStudyRequest, ConfirmMandateRequest
from backend.strategic_allocation.cma_center_contracts import CmaCenterPublish
from backend.strategic_allocation.universe_contracts import UniverseRequest, ConfirmUniverseRequest

original_lifespan = app.router.lifespan_context
fixture = {}


def publish(request, key):
    preview = strategic.preview_cma(request)
    return strategic.publish_cma(CmaCenterPublish(request=request, preview_hash=preview['preview_hash'],
                                                confirm=True, idempotency_key=key))


def seed():
    today = str(date.today())
    assets = [
        {'id': '股票', 'role': 'growth', 'liquidity': 'liquid', 'rationale': '离线权益增长研究',
         'annual_return': .07, 'annual_volatility': .18, 'mean_uncertainty': .02},
        {'id': '债券', 'role': 'rates', 'liquidity': 'liquid', 'rationale': '离线利率防御研究',
         'annual_return': .03, 'annual_volatility': .05, 'mean_uncertainty': .005},
    ]
    raw = dict(schema_version='2.0', name='离线基础 LTCMA', alloc_name='浏览器离线股债', as_of=today,
               currency='CNY', horizon_years=10, source='离线合成数据与明确的长期假设，仅供浏览器测试',
               basis_confirmed=True, assets=assets, correlation=[[1., -.1], [-.1, 1.]],
               moment_semantics='annualized_periodic_arithmetic', fee_basis='source_embedded_no_additional_fee',
               fx_hedging_basis='same_currency_no_conversion')
    base = publish(CmaRequest(**raw), 'browser-ltcma-manual')
    context = {'asset_ids': ['股票', '债券'], 'as_of': today, 'currency': 'CNY', 'source': raw['source']}
    history = publish(CmaRequest(**{**raw, 'name': '离线历史 LTCMA',
        'assets': [{**asset, 'annual_return': None, 'annual_volatility': None, 'mean_uncertainty': 0.} for asset in assets], 'model': {
        **context, 'method': 'historical_statistics', 'window': {'kind': 'common_since_inception'}, 'shrinkage': .1,
    }}), 'browser-ltcma-historical')
    bl = publish(CmaRequest(**{**raw, 'name': '离线 BL LTCMA', 'model': {
        **context, 'method': 'black_litterman', 'covariance': [[.0324, -.0009], [-.0009, .0025]],
        'risk_covariance_basis': 'input_covariance', 'market_weights': {'股票': .6, '债券': .4},
        'market_weight_source': '离线明确基准组合', 'delta': 3., 'tau': .05, 'risk_free_rate': .02, 'views': [],
    }}), 'browser-ltcma-bl')
    scenario = publish(CmaRequest(**{**raw, 'name': '离线情景 LTCMA', 'moment_semantics': 'one_year_simple', 'model': {
        **context, 'method': 'scenario_mixture', 'risk_mode': 'shared', 'shared_covariance': [[.0324, 0.], [0., .0025]],
        'scenarios': [{'id': '基准', 'probability': .6, 'annual_returns': {'股票': .07, '债券': .03}, 'source': '明确合成基准情景'},
                      {'id': '压力', 'probability': .4, 'annual_returns': {'股票': -.05, '债券': .02}, 'source': '明确合成压力情景'}],
    }}), 'browser-ltcma-scenario')
    mandate_request = MandateStudyRequest(definition=MandateRequest(
        name='离线 SAA 授权', as_of=date.today(), review_date=date.today() + timedelta(days=180),
        target_return=0., max_volatility=.25, min_liquid_weight=0., max_tracking_error=.1,
        boundary_reason='离线测试明确授权，非真实投资审批'))
    preview = strategic.preview_mandate(mandate_request)
    mandate = strategic.confirm_mandate(ConfirmMandateRequest(request=mandate_request,
        preview_hash=preview['preview_hash'], acknowledge_limits=True))
    common_sources = []
    for index, means in enumerate(((.1, .02), (.02, .1))):
        common_raw = copy.deepcopy(raw)
        common_raw.update(name=f'共同约束模型 {index + 1}', correlation=[[1., 0.], [0., 1.]])
        for asset, mean in zip(common_raw['assets'], means):
            asset.update(annual_return=mean, annual_volatility=.2, mean_uncertainty=0.)
        common_sources.append(publish(CmaRequest(**common_raw), f'browser-common-{index}'))
    common_mandates = []
    for floor in (.058, .061):
        study = MandateStudyRequest(definition=MandateRequest(
            name=f'共同约束收益下限 {floor}', as_of=date.today(), review_date=date.today()+timedelta(days=180),
            target_return=floor, max_volatility=.16, min_liquid_weight=0., max_tracking_error=.1,
            boundary_reason='两个模型共同收益与风险约束的离线数值样例'))
        check = strategic.preview_mandate(study)
        common_mandates.append(strategic.confirm_mandate(ConfirmMandateRequest(
            request=study, preview_hash=check['preview_hash'], acknowledge_limits=True)))
    fixture.update(common_ids=[item['id'] for item in common_sources],
                   common_mandate_id=common_mandates[0]['id'], conflict_mandate_id=common_mandates[1]['id'])
    universe_request = UniverseRequest(name='离线独立战略范围', as_of=date.today(), currency='CNY', source='明确的独立战略范围定义', assets=[
        {**{key: asset[key] for key in ('role', 'liquidity', 'rationale')}, 'id': ('equity', 'bond')[i],
         'name': asset['id'], 'currency': 'CNY', 'source': '独立经济用途定义'} for i, asset in enumerate(assets)])
    universe_preview = strategic.scopes.preview_universe(universe_request)
    universe = strategic.scopes.confirm_universe(ConfirmUniverseRequest(request=universe_request,
        preview_hash=universe_preview['preview_hash']))
    from backend.historical_regimes.v2_service import _stored_run_snapshot_hash
    run = {'id': 'regime-run-ltcma-browser', 'name': '离线事后状态', 'schema_version': '2.0', 'immutable': True,
           'mode': 'retrospective', 'frequency': 'daily', 'as_of': today,
           'states': [{'id': 'up', 'label': '上行'}, {'id': 'down', 'label': '下行'}],
           'application_bindings': [], 'publications': [],
           'series': [{'observation_date': str(day.date()), 'available_at': str(day.date()),
                       'recognized_at': today, 'state_id': 'up' if i % 2 else 'down'} for i, day in enumerate(days)]}
    run['content_hash'] = _stored_run_snapshot_hash(run)
    regime_path = strategic.cma.evidence.regime_root / 'historical_regime_runs.json'
    regime_path.parent.mkdir(parents=True, exist_ok=True)
    regime_path.write_text(json.dumps({'items': [run]}), encoding='utf-8')
    fixture.update(today=today, allocation='浏览器离线股债', manual_id=base['id'], historical_id=history['id'],
                   bl_id=bl['id'], scenario_id=scenario['id'], mandate_id=mandate['id'],
                   universe_id=universe['id'], regime_id=run['id'])


@asynccontextmanager
async def lifespan(instance):
    async with original_lifespan(instance):
        seed()
        yield


app.router.lifespan_context = lifespan


@app.get('/fixture/ltcma')
def fixture_state():
    return fixture

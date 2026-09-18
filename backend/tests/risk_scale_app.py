"""Offline UI fixture: temporary test-only Parquet, real resolvers and NJIT.

Never imports backend.app or production singletons. No network or live artifacts.
"""
from contextlib import asynccontextmanager
from datetime import date, timedelta
import json
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
from fastapi import FastAPI
from backend.sensitivity.repository import ArtifactRepository
from backend.strategic_allocation.reference_sources import ReferenceSources
from backend.strategic_allocation.reference_inputs import ReferenceInputs
from backend.strategic_allocation.risk_scale_store import RiskScaleStore
from backend.strategic_allocation.risk_scale_service import RiskScaleService
from backend.strategic_allocation.risk_scale_routes import build_router


def seed_sources(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    snapshot = root / 'synthetic-test-only'
    snapshot.mkdir()
    days = pd.bdate_range(end=pd.Timestamp(date.today()) - pd.Timedelta(days=3), periods=301)
    codes = ['900001.SH', '900002.SH', '900003.SH']
    extra_index_code = 'H00985.CSI'
    returns = np.column_stack([0.00006 + .0003 * np.sin(np.arange(300)*.27),
        .00025 + .003 * np.cos(np.arange(300)*.19), .0004 + .010 * np.sin(np.arange(300)*.43)])
    levels = np.vstack([np.ones(3), np.cumprod(1 + returns, axis=0)])
    names = ['测试专用现金', '测试专用利率代理', '测试专用权益代理']
    pd.DataFrame([{'ts_code': c, 'name': n} for c,n in zip(codes,names)]).to_parquet(snapshot/'etf_info_df.parquet', index=False)
    pd.DataFrame([{'ts_code': c, 'nav_date': day, 'ann_date': day, 'adj_nav': levels[t,i]}
        for i,c in enumerate(codes) for t,day in enumerate(days)]).to_parquet(snapshot/'etf_daily_df.parquet', index=False)
    index_rows = [{'ts_code': c, 'trade_date': day, 'close': levels[t,i]}
        for i,c in enumerate(codes) for t,day in enumerate(days)]
    index_rows.extend({'ts_code': extra_index_code, 'trade_date': day, 'close': levels[t,2]} for t,day in enumerate(days))
    pd.DataFrame(index_rows).to_parquet(snapshot/'index_daily_df.parquet', index=False)
    index_catalog = [{'ts_code':c,'name':n.replace('代理','价格指数'),'quote_source_api':'index_daily',
        'market':'SSE','publisher':'synthetic-test-only','category':'test'} for c,n in zip(codes,names)]
    index_catalog.append({'ts_code':extra_index_code,'name':'中证全指全收益','quote_source_api':'index_daily',
        'market':'CSI','publisher':'synthetic-test-only','category':'test'})
    pd.DataFrame(index_catalog).to_parquet(snapshot/'index_catalog_df.parquet',index=False)
    pd.DataFrame([{'source_api':'index_daily','ts_code':c,'first_date':days[0], 'latest_date':days[-1],
        'rows':len(days),'domestic_trade_day_coverage':1.0} for c in [*codes, extra_index_code]]).to_parquet(snapshot/'index_coverage_snapshot.parquet',index=False)
    pd.DataFrame({'exchange':['SSE']*len(days),'is_open':[1]*len(days),'cal_date':days}).to_parquet(root/'trade_day_df.parquet', index=False)
    files = {p.name: {'status':'passed'} for p in snapshot.iterdir()}
    (root/'tushare_active.json').write_text(json.dumps({'schema_version':1,'snapshot_dir':snapshot.name,
        'activated_at':str(date.today())+'T00:00:00+00:00','files':files,'validation':{'status':'passed','datasets':{}}}))
    request = {'name':'合成测试专用参考资产', 'currency':'CNY', 'as_of':str(date.today()),
        'calendar':'SSE','frequency':'daily', 'return_basis':'selected_index_and_adjusted_product_total_return', 'assets':[]}
    for i,code in enumerate(codes):
        if i == 0:
            asset = {'id':'cash', 'name':names[i], 'asset_type':'cash',
                'rationale':'合成测试纯现金，不使用任何市场代理', 'cash_return':.01,
                'rebalance':None, 'components':[]}
        else:
            asset = {'id':['cash','bond','equity'][i], 'name':names[i], 'asset_type':'market',
                'rationale':'合成测试来源，不构成真实市场证据', 'cash_return':None, 'rebalance':'daily',
                'components':[{'kind':'etf','series_id':'etf:fund_daily:'+code,'field':'adj_nav','weight':1.0}]}
        request['assets'].append(asset)
    return request


def make_service(root: Path):
    request = seed_sources(root/'data')
    artifacts = ArtifactRepository(root/'research'/'artifacts')
    references = ReferenceInputs(artifacts, ReferenceSources(root/'data'))
    return RiskScaleService(artifacts, RiskScaleStore(root/'research'), references), request


def create_app(root=None):
    storage = tempfile.TemporaryDirectory(prefix='risk-scales-offline-') if root is None else None
    service, fixture_request = make_service(Path(storage.name) if storage else Path(root))
    @asynccontextmanager
    async def lifespan(app):
        service.warm()
        yield
        if storage:
            storage.cleanup()
    app = FastAPI(lifespan=lifespan)
    app.state.risk_scale_service = service
    app.state.fixture_reference_request = fixture_request
    app.include_router(build_router(service))

    @app.get('/api/health')
    @app.get('/ready')
    def health():
        return {'ready':service.execution()['complete'],'fixture':'synthetic-test-only', 'risk_scales':service.execution()}

    @app.get('/api/risk-scale-fixture')
    def fixture():
        return {'reference_request':fixture_request, 'test_only':True}

    @app.get('/api/pit/settings')
    def pit_settings():
        return {'settings':{'active_release_id':None,'as_of':str(date.today()),'run_mode':'RESEARCH','updated_at':None,'note':'离线测试'},
                'effective':{'as_of':str(date.today()),'as_of_source':'explicit','run_mode':'RESEARCH','run_mode_label':'研究',
                    'data_release_id':None,'no_pit':False,'label':'合成测试研究日'},'release':None,'release_error':None,
                'available_releases':[],'can_apply':False}

    @app.get('/api/i18n/settings')
    @app.get('/api/i18n/state')
    def i18n_state():
        return {'revision':0,'preferences_revision':0,'default_locale':'zh-CN','catalog_version':'fixture','overrides':{},
            'locales':[{'id':'zh-CN','label':'中文','fallback_locale':'zh-CN','enabled':True,'builtin':True,'system_pack':True},
                       {'id':'en-US','label':'English','fallback_locale':'zh-CN','enabled':True,'builtin':True,'system_pack':True}]}

    @app.get('/api/i18n/bundle')
    def bundle(locale: str='zh-CN'):
        return {'locale':locale,'default_locale':'zh-CN','revision':0,'catalog_version':'fixture',
                'resources':{'business':{},'shell':{}},'fallback_keys':{'business':[],'shell':[]}}
    return app


# Uvicorn factory is preferred: backend.tests.risk_scale_app:create_app --factory.
# Lazy app is provided for existing Playwright webServer conventions.
app = create_app()

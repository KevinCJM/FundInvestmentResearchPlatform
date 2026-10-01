"""Real host HTTP plus a deterministic local OpenAI wire fixture. Never contacts a provider."""
import json
import os
import re
import sys
import uuid
from pathlib import Path

from fastapi import Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.exceptions import RequestValidationError

from indicator_primitives_app import app, routes, _root
from integrations.portable_agent.routes import install
from research_access.store import ResearchStore
from test_custom_indicator_service import _draft

bridge = install(app, indicator_service=routes.indicator_service, page_services={
    'search': lambda **kwargs: {'items': [{'instrument_type': 'etf', 'ts_code': '510050.SH', 'name': '上证50ETF'}], 'total': 1},
}, store=ResearchStore(_root/'research_access'))
# Real graph services and deterministic market input; no numerical route is mocked.
import pandas as pd
from services import historical_regime_routes
from historical_regimes.v2_service import RegimeGraphV2Service
from historical_regimes.v2_templates import TEMPLATES_V2
from research_access.scenario_contracts import authoring_definition
from copy import deepcopy
_graph = RegimeGraphV2Service(_root, _root)
historical_regime_routes.regime_graph_v2_service = _graph
app.include_router(historical_regime_routes.router)
bridge.pages['graph'] = _graph
pd.DataFrame([{'ts_code': '000300.SH', 'trade_date': stamp.strftime('%Y%m%d'),
               'close': 100+min(i, 80-i), 'date': stamp}
    for i, stamp in enumerate(pd.bdate_range('2020-01-01', periods=80))]).to_parquet(_root/'index_daily_df.parquet', index=False)
_graph_definition = deepcopy(next(t['definition'] for t in TEMPLATES_V2 if t['id']=='peak-trough-daily-v2'))
_graph_definition['name'] = '联调已保存日频峰谷'
_saved_graph = _graph.definitions.create(_graph_definition)
from services import portfolio_routes
from test_portfolio_research import _service as portfolio_fixture, _definition as portfolio_definition
_portfolio_dir = _root/'portfolio'
_portfolio_dir.mkdir()
_portfolio = portfolio_fixture(_portfolio_dir)
_target = _portfolio.create_target({'name': '联调不可变组合', 'definition': portfolio_definition()})
_portfolio_run = _portfolio.run_target(_target['id'])
portfolio_routes.portfolio_service = _portfolio
app.include_router(portfolio_routes.router)
routes.indicator_service.portfolio_runs = _portfolio.runs
bridge.pages.update(run=_portfolio.get_run, diagnosis=_portfolio.diagnose)
from services import instrument_routes, instrument_service
from instrument_analytics_numba import warm_instrument_analytics_numba_kernels
os.environ['TUSHARE_DATA_DIR'] = str(_root)
instrument_service.DATA_DIR = _root
instrument_service.INSTRUMENT_FILES = {kind: _root/f'{kind}_info_df.parquet' for kind in ('etf', 'fund')}
warm_instrument_analytics_numba_kernels()
app.include_router(instrument_routes.router)
model_requests = []


from research_access.contracts import ResearchError


@app.exception_handler(ResearchError)
async def research_failure(request, exc):
    # Fixed local fixtures only; report the policy reason, never request bodies or credentials.
    print('Fixture business rejection:', request.url.path, exc.detail(), file=sys.stderr, flush=True)
    return JSONResponse({'error': exc.detail()}, status_code=exc.status_code)


@app.middleware('http')
async def admission_diagnostic(request, call_next):
    response = await call_next(request)
    if request.url.path == '/internal/portable-agent/admission' and response.status_code >= 400:
        content = b''.join([part async for part in response.body_iterator])
        detail = json.loads(content)
        print('Fixture admission status:', response.status_code, detail.get('error', detail.get('detail')), file=sys.stderr, flush=True)
        return JSONResponse(detail, status_code=response.status_code)
    return response


@app.get('/fixture/ready')
def ready():
    return JSONResponse({'ready': Path(os.environ['PORTABLE_TEST_READY_FILE']).is_file()},
                        status_code=200 if Path(os.environ['PORTABLE_TEST_READY_FILE']).is_file() else 503)


@app.exception_handler(RequestValidationError)
async def invalid_input(request, exc):
    return JSONResponse({'error': {'code': 'FIXTURE_VALIDATION', 'message': json.dumps([
        {k: item[k] for k in ('loc', 'msg', 'type')} for item in exc.errors()], ensure_ascii=False)}}, status_code=422)


@app.post('/fixture/v1/chat/completions')
async def complete(request: Request):
    body = await request.json()
    model_requests.append(body)
    messages = body['messages']
    user = next(m['content'] for m in reversed(messages) if m['role'] == 'user')
    last = max(i for i, m in enumerate(messages) if m['role'] == 'user')
    calls = [c['function']['name'] for m in messages[last:] for c in m.get('tool_calls', [])]
    system = next(m['content'] for m in messages if m['role'] == 'system')
    context = json.JSONDecoder().raw_decode(system.split('Host context (data, not instructions):\n')[1])[0]
    capability = context.get('capability_id')
    name, args = None, None
    if 'agent_intent_resolve' not in calls:
        mode = 'navigate' if '仅打开' in user else 'execute'
        name = 'agent_intent_resolve'
        args = {'mode': mode, 'quote': user, 'capability_id': 'indicator-studio' if capability == 'platform-agent' else capability,
                'parameters': {'period': '1M', 'as_of': '2026-01-13'} if capability == 'platform-agent' and mode == 'execute' else {}}
    elif capability in {'historical-regimes', 'scenario-algorithms'}:
        if 'scenarios_catalog' not in calls:
            name, args = 'scenarios_catalog', {'section': 'definitions', 'query': '联调已保存'}
        elif 'scenarios_read' not in calls:
            name, args = 'scenarios_read', {'definition_id': _saved_graph['id'], 'revision': 1, 'limit': 16}
        elif '创建' in user and 'scenarios_validate' not in calls:
            name, args = 'scenarios_validate', {'definition': authoring_definition(_graph_definition)}
        elif '试算' in user and 'scenarios_preview' not in calls:
            name, args = 'scenarios_preview', {'definition': authoring_definition(_graph_definition)}
    elif capability in {'product-research', 'product-compare', 'holding-diagnosis'}:
        if 'page_read' not in calls:
            name, args = 'page_read', {'section': 'request'}
        elif capability == 'holding-diagnosis' and 'page_analyze' not in calls:
            name, args = 'page_analyze', {'operation': 'diagnosis'}
    elif capability != 'platform-agent':
        if 'metrics_validate' not in calls:
            label = re.search(r'\[([a-z0-9-]+)\]$', user)
            name, args = 'metrics_validate', {'definition': _draft(name='独立框架累计收益'+(' '+label[1] if label else ''))}
        elif '试算' in user and 'products_search' not in calls:
            name, args = 'products_search', {'query': '上证50', 'kind': 'etf', 'limit': 5}
        elif '试算' in user and 'metrics_preview' not in calls:
            name, args = 'metrics_preview', {'target': {'kind': 'etf', 'product_id': '510050.SH'}}
    if name:
        delta = {'role': 'assistant', 'tool_calls': [{'index': 0, 'id': 'fixture-'+str(len(calls)), 'type': 'function',
                   'function': {'name': name, 'arguments': json.dumps(args, ensure_ascii=False)}}]}
        finish = 'tool_calls'
    else:
        delta, finish = {'role': 'assistant', 'content': '已完成本轮处理。正式保存仍需你查看影响后单独确认。'}, 'stop'
    mid = 'fixture-'+uuid.uuid4().hex
    chunks = [{'id': mid, 'object': 'chat.completion.chunk', 'created': 1, 'model': body['model'],
        'choices': [{'index': 0, 'delta': delta, 'finish_reason': None}]},
        {'id': mid, 'object': 'chat.completion.chunk', 'created': 1, 'model': body['model'],
         'choices': [{'index': 0, 'delta': {}, 'finish_reason': finish}]}]
    async def events():
        for chunk in chunks:
            yield 'data: '+json.dumps(chunk, ensure_ascii=False)+'\n\n'
        yield 'data: [DONE]\n\n'
    return StreamingResponse(events(), media_type='text/event-stream')


@app.get('/fixture/model-requests')
def captured_requests():
    return {'items': model_requests}


@app.get('/fixture/portfolio-run')
def portfolio_run():
    return {'id': _portfolio_run['id']}


@app.get('/api/instruments/search')
def search():
    return {'items': [{'kind': 'etf', 'product_id': '510050.SH', 'code': '510050', 'ts_code': '510050.SH', 'name': '上证50ETF'}], 'total': 1}

"""Frozen product page contracts and original-service adapters; offline only."""
import copy
import json
import sys
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from agent import data_policy, research_pages
from agent.contracts import AgentError, AgentMessageRequest, PageContext
from agent.tools import execute_tool
from agent.sessions import stable_json
from custom_indicators.service import CustomIndicatorService
from test_agent_api import FakeIndicatorService

SENTINEL = 987654.321
RUN_ID = 'run-'+'a'*32
TARGETS = [{'kind': 'etf', 'product_id': '510300.SH'}, {'kind': 'fund', 'product_id': '000001.OF'}]
RANGES = {'performance': {'start_date': '2020-01-01', 'end_date': '2020-12-31'},
          'risk': {'start_date': '2021-01-01', 'end_date': '2021-12-31'},
          'efficiency': {'start_date': '2022-01-01', 'end_date': '2022-12-31'}}


def page(name='product-research', targets=None):
    calculation = {'context_kind': 'portfolio', 'run_id': RUN_ID} if name == 'holding-diagnosis' else {
        'context_kind': 'single_product', 'targets': targets or [], 'period': '1Y', 'as_of': None}
    return PageContext.model_validate({'page': name, 'page_instance_id': name+'-instance',
        'context_revision': 1, 'view_state': 'inherit', 'calculation': calculation})


def snapshot(name, request, results=None):
    return {'version': 1, 'snapshot_id': 'snap-'+'b'*32, 'page': name,
            'sections': {'request': request, 'results': results or {'value': SENTINEL, 'values': [SENTINEL]}}}


def request_for(name):
    if name == 'product-research':
        return {'kind': 'etf', 'q': '沪深300', 'fund_type': ['股票型'], 'fund_category': ['ETF'],
            'invest_type': ['被动'], 'market': ['SSE'], 'status': ['L'], 'management': ['某机构'],
            'custodian': ['某托管人'], 'qdii_type': [], 'page': 3, 'page_size': 50,
            'sort_by': 'issue_amount', 'sort_dir': 'asc', 'conditions': [{'field': 'return_1y', 'operator': 'gte', 'value': '0'}],
            'snapshot_metrics': ['return_1y'], 'as_of': None, 'targets': [TARGETS[0]],
            'batch_offset': 0, 'visible_count': 50, 'selected_count': 300, 'selection_mode': 'all_matching',
            'excluded_ids': ['510050.SH'], 'view_mode': 'metrics', 'indicators': []}
    if name == 'product-compare':
        return {'targets': [{**target, 'management_fee': 0.5, 'custody_fee': 0.1} for target in TARGETS],
            'ranges': copy.deepcopy(RANGES), 'rolling_window_days': 17, 'as_of': None, 'source': 'actual', 'indicators': []}
    return {'run_id': RUN_ID, 'indicators': [], 'scenario': {'start_date': '2015-01-01', 'end_date': '2016-01-01'}}


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv('CUSTOM_INDICATOR_DATA_DIR', str(tmp_path))


def call(name, operation, request, service, callbacks, *, section=None):
    targets = [{key: target[key] for key in ('kind', 'product_id')} for target in request.get('targets', [])]
    context = page(name, targets)
    return execute_tool('page.read' if section else 'page.analyze', {'section': section, 'limit': 8000} if section else {'operation': operation},
        session={'scope': 'product_research'}, page_context=context, service=service,
        page_snapshot=snapshot(name, request), page_services=callbacks)


@pytest.mark.parametrize('name', ['product-research', 'product-compare', 'holding-diagnosis'])
def test_new_pages_accept_only_their_frozen_contract_and_hide_client_results(tmp_path, name):
    request = request_for(name); targets = [{key: t[key] for key in ('kind', 'product_id')} for t in request.get('targets', [])]
    context = page(name, targets)
    model = AgentMessageRequest(message_id='m', expected_session_revision=0, text='解释页面', page_context=context,
                                page_snapshot=snapshot(name, request))
    assert model.page_snapshot.page == name
    result = call(name, None, request, FakeIndicatorService(tmp_path), {}, section='results')
    assert '987654' not in stable_json(result) and 'page.analyze' in result['result']['content']
    if name != 'holding-diagnosis':
        malformed = {**request, 'targets': request['targets'] * 11}
        with pytest.raises(ValidationError):
            AgentMessageRequest(message_id='m', expected_session_revision=0, text='解释页面', page_context=context,
                                page_snapshot=snapshot(name, malformed))
    else:
        with pytest.raises(ValidationError):
            AgentMessageRequest(message_id='m', expected_session_revision=0, text='解释页面', page_context=context,
                                page_snapshot=snapshot(name, {'run_id': 'placeholder'}))
    with pytest.raises(AgentError):
        call(name, 'not-real', request, FakeIndicatorService(tmp_path), {})


@pytest.mark.parametrize('status', ['ready', 'loading', 'error', 'stale', 'pending', 'not_requested'])
@pytest.mark.parametrize('name', ['product-research', 'product-compare', 'holding-diagnosis'])
def test_display_references_keep_old_request_and_status_without_client_values(tmp_path, name, status):
    request = request_for(name)
    old_indicator = {'indicator_id': 'indicator-old', 'indicator_revision': 3, 'period': '3M', 'parameters': {'price': SENTINEL}}
    old = {'as_of': '2019-12-31', 'targets': [{**TARGETS[0], 'management_fee': SENTINEL}],
           'ranges': RANGES, 'run_id': RUN_ID, 'start_date': '2015-01-01', 'end_date': '2016-01-01',
           'query': f'nav={SENTINEL}', 'pit_identity': json.dumps({'nav': SENTINEL}),
           'requests': [{'as_of': '2019-12-31', 'period': '3M', 'indicator_refs': [old_indicator],
                         'targets': [TARGETS[0]], 'values': [SENTINEL]}], 'nav': SENTINEL}
    ref_name = {'product-research': 'metrics', 'product-compare': 'comparison', 'holding-diagnosis': 'scenario_result'}[name]
    displayed = {'source': str(SENTINEL), 'refs': {ref_name: {'status': status, 'frozen_request': old,
        'indicators': [old_indicator], 'resolved_indicators': [old_indicator], 'value': SENTINEL},
        'unregistered': {'value': SENTINEL}}}
    frozen = snapshot(name, request, displayed)
    before = copy.deepcopy(frozen)
    result = execute_tool('page.read', {'section': 'results', 'limit': 8000},
        session={'scope': 'product_research'}, page_context=page(name), service=FakeIndicatorService(tmp_path), page_snapshot=frozen)
    view = json.loads(result['result']['content'])
    ref = view['refs'][ref_name]
    assert view['trust'] == 'client_display_reference_not_result_proof'
    assert ref['status'] == status and ref['frozen_request']['as_of'] == '2019-12-31'
    assert ref['frozen_request']['requests'][0]['indicator_refs'][0]['indicator_revision'] == 3
    assert ref['request_hash'] == research_pages.stable_hash(old)
    assert '987654' not in stable_json(view) and 'unregistered' not in view['refs']
    assert '不能冒充旧图表' in view['note'] and frozen == before
    displayed['refs'][ref_name]['status'] = {'value': SENTINEL}
    assert research_pages.results_view(name, frozen)['refs'][ref_name]['status'] == 'unknown'


def test_page_read_model_requests_withhold_current_and_historical_fees(tmp_path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from agent import routes
    from agent.llm import FixtureLLMClient

    monkeypatch.setattr(routes, 'resolve_service', lambda: FakeIndicatorService(tmp_path))
    model = FixtureLLMClient([
        {'tool_calls': [{'name': 'page.read', 'arguments': {'section': 'request', 'limit': 8000}}]},
        {'tool_calls': [{'name': 'page.read', 'arguments': {'section': 'results', 'limit': 8000}}]},
        {'content': '页面仍显示旧结果，当前请求重算不能充当旧图表证据。'},
    ])
    monkeypatch.setattr(routes, '_llm_client', lambda _: model)
    request = request_for('product-compare')
    request['targets'][0].update(management_fee=SENTINEL, custody_fee=SENTINEL + 1)
    request['targets'][1].update(management_fee=0, custody_fee=None)
    old = {**copy.deepcopy(request), 'as_of': '2019-12-31'}
    displayed = {'refs': {'comparison': {'status': 'error', 'frozen_request': old}}}
    context = page('product-compare', TARGETS)
    app = FastAPI(); app.include_router(routes.router)
    with TestClient(app) as client:
        sid = client.post('/api/agent/sessions', json={'page_context': context.model_dump()}).json()['session_id']
        response = client.post(f'/api/agent/sessions/{sid}/messages', json={
            'message_id': 'fee-check', 'expected_session_revision': 0, 'text': '解释当前页面结果',
            'page_context': context.model_dump(), 'page_snapshot': snapshot('product-compare', request, displayed)})
        assert response.status_code == 200, response.text
    assert len(model.requests) == 3 and '987654' not in stable_json(model.requests)
    receipts = [json.loads(message['content']) for message in model.requests[-1]['messages'] if message['role'] == 'tool']
    current, historical = [json.loads(receipt['result']['content']) for receipt in receipts]
    assert all(target['fees_withheld'] for target in current['targets'])
    assert 'management_fee' not in stable_json(current) and 'custody_fee' not in stable_json(historical)
    assert historical['refs']['comparison']['status'] == 'error'
    assert historical['refs']['comparison']['frozen_request']['as_of'] == '2019-12-31'


def test_research_catalog_preserves_every_argument_and_batch_hindsight(tmp_path):
    request = request_for('product-research'); captured = []
    raw = {'items': [{'ts_code': '510300.SH', 'name': '沪深300', 'm_fee': 0, 'c_fee': None,
                     'snapshot_values': {'latest_close': SENTINEL}, 'condition_values': {'x': SENTINEL}}],
           'page': 3, 'page_size': 50, 'total': 300, 'summary': {'filtered_total': 300, 'active_count': 0,
             'avg_m_fee': None, 'junk_value': SENTINEL}, 'pit': {'as_of': '2022-12-31', 'snapshot_is_hindsight': True}}
    before = copy.deepcopy(raw)
    def listing(**kwargs): captured.append(kwargs); return raw
    output = call('product-research', 'catalog', request, FakeIndicatorService(tmp_path), {'catalog': listing})['result']
    expected = research_pages.ProductListRequest.model_validate(request).model_dump()
    for key in ('targets', 'batch_offset', 'visible_count', 'selected_count', 'selection_mode', 'excluded_ids', 'indicators', 'view_mode', 'as_of'):
        expected.pop(key)
    expected['conditions'] = ['return_1y|gte|0']
    assert captured == [expected]
    assert output['catalog']['summary']['active_count'] == 0 and output['catalog']['summary']['avg_m_fee'] is None
    assert output['catalog']['pit']['snapshot_is_hindsight'] is True and output['batch']['count'] == 1
    assert output['batch']['selected_count'] == 300 and output['batch']['scope'] == 'explicit_current_batch_only'
    assert '987654' not in stable_json(output) and raw == before


def test_grouped_metrics_call_locked_revisions_periods_parameters_and_proof(tmp_path):
    real = CustomIndicatorService(tmp_path, tmp_path); calls = []
    class Service(FakeIndicatorService):
        infer = lambda self, fields: real.infer(fields)
        get_indicator = lambda self, id, revision=None: real.get_indicator(id, revision)
        def evaluate(self, **kw):
            calls.append(kw)
            ref = kw['indicator_refs'][0]
            return {'results': [{'indicator_id': ref['indicator_id'], 'indicator_revision': ref['indicator_revision'],
                                  'status': 'ok', 'value': 0, 'window': {'observation_count': 20}, 'series': [SENTINEL]}]}
    request = request_for('product-research')
    request['indicators'] = [{'indicator_id': 'builtin-mean-return-v2', 'indicator_revision': 1, 'period': '1Y'},
                             {'indicator_id': 'builtin-annualized-sharpe-v2', 'indicator_revision': 1, 'period': '3M'}]
    result = call('product-research', 'metrics', request, Service(tmp_path), {})['result']
    assert [kw['period'] for kw in calls] == ['1Y', '3M']
    assert all(kw['targets'] == [TARGETS[0]] and kw['include_series'] is False and kw['prefer_snapshot'] is False for kw in calls)
    assert all(group['results'][0]['value'] == 0 for group in result['evaluations'])
    assert '987654' not in stable_json(result)


def test_compare_keeps_mixed_targets_three_ranges_fees_and_drops_all_curves(tmp_path):
    request = request_for('product-compare'); calls = []; outputs = []
    def compare(target, parameters):
        calls.append((target, parameters))
        result = {'product_id': target['product_id'], 'ranges': {
            name: {'window': {'start_date': bounds['start_date'], 'end_date': bounds['end_date'], 'observation_count': 20},
                   'metrics': {'cumulativeReturn': 0, 'volatility': None, 'last_close': SENTINEL},
                   'normalized_nav': [{'date': '2020-01-01', 'value': SENTINEL}]} for name, bounds in parameters['ranges'].items()}}
        outputs.append(copy.deepcopy(result)); return result
    result = call('product-compare', 'comparison', request, FakeIndicatorService(tmp_path), {'comparison': compare})['result']
    assert [entry[0]['kind'] for entry in calls] == ['etf', 'fund']
    assert all(parameters == {'ranges': RANGES, 'rolling_window_days': 17} for _, parameters in calls)
    assert result['comparisons'][0]['ranges']['risk']['metrics'] == {'cumulativeReturn': 0, 'volatility': None}
    assert 'announcement_pit_not_proven' in result['comparisons'][1]['pit']
    assert '987654' not in stable_json(result)
    assert outputs[0]['ranges']['risk']['normalized_nav'][0]['value'] == SENTINEL
    with pytest.raises(AgentError) as error:
        call('product-compare', 'comparison', {**request, 'source': 'demo'}, FakeIndicatorService(tmp_path), {'comparison': compare})
    assert error.value.code == 'AGENT_DEMO_NOT_EVIDENCE'


def test_compare_matrix_override_preserves_comparison_context_and_restores_after_error(tmp_path, monkeypatch):
    from pit.context import build_context, set_view_override, reset_view_override, view_override
    global_context = build_context('2019-12-31', 'RESEARCH', None)
    request = request_for('product-compare')
    request.update(as_of='2019-12-31', metrics_as_of='2020-12-31')
    context = page('product-compare', TARGETS)
    context.calculation.as_of = '2019-12-31'
    calls = []
    def evaluate(service, request, **kwargs):
        calls.append((kwargs['as_of'], view_override().as_of))
        return []
    monkeypatch.setattr(research_pages, '_evaluate_metrics', evaluate)
    service = FakeIndicatorService(tmp_path)
    token = set_view_override(global_context)
    try:
        result = research_pages.analyze('metrics', context, snapshot('product-compare', request), service, {})
        assert calls == [('2020-12-31', '2020-12-31')]
        assert result['effective_as_of'] == '2020-12-31'
        assert view_override() == global_context
        captured = []
        def compare(target, parameters):
            captured.append(view_override().as_of)
            return {'product_id': target['product_id'], 'ranges': {}}
        research_pages.analyze('comparison', context, snapshot('product-compare', request), service, {'comparison': compare})
        assert captured == ['2019-12-31', '2019-12-31']
        def fail(*args, **kwargs): raise ValueError('fixture')
        monkeypatch.setattr(research_pages, '_evaluate_metrics', fail)
        with pytest.raises(ValueError):
            research_pages.analyze('metrics', context, snapshot('product-compare', request), service, {})
        assert view_override() == global_context
    finally:
        reset_view_override(token)


def test_immutable_holding_summary_scenario_and_missing_run(tmp_path):
    request = request_for('holding-diagnosis'); calls = []
    run = {'id': RUN_ID, 'immutable': True, 'target_id': 'target', 'target_revision': 4,
           'requested_as_of': '2020-12-31', 'effective_as_of': '2020-12-30', 'actual_start_date': '2019-01-02',
           'actual_end_date': '2020-12-30', 'observation_count': 200, 'summary': {'cumulative_return': 0, 'sharpe_ratio': None},
           'asset_returns': [[SENTINEL]], 'portfolio_nav': [SENTINEL], 'data_fingerprints': {'fund': 'hash'}}
    diagnosis = {'run_id': RUN_ID, 'components': [{'kind': 'etf', 'product_id': '510300.SH', 'current_weight': 1,
                   'period_return': 0, 'last_price': SENTINEL}], 'concentration_summary': {'max_weight': 1, 'hhi': 1},
                 'covariance': {'values': [[SENTINEL]]}, 'dates': ['2020-01-01']}
    before = copy.deepcopy((run, diagnosis))
    def scenario(id, **kw): calls.append((id, kw)); return {**run, 'source_run_id': id, 'locked_target_revision': 4}
    callbacks = {'run': lambda id: run, 'diagnosis': lambda id, ids: diagnosis, 'scenario': scenario}
    result = call('holding-diagnosis', 'diagnosis', request, FakeIndicatorService(tmp_path), callbacks)['result']
    assert result['portfolio']['snapshot']['effective_as_of'] == '2020-12-30'
    assert result['portfolio']['metrics']['cumulative_return'] == 0 and result['portfolio']['metrics']['sharpe_ratio'] is None
    assert result['portfolio']['components'][0]['current_weight'] == 1
    assert '987654' not in stable_json(result) and (run, diagnosis) == before
    result = call('holding-diagnosis', 'scenario', request, FakeIndicatorService(tmp_path), callbacks)['result']
    assert calls == [(RUN_ID, request['scenario'])] and 'not_snapshot_replay' in result['provenance']
    from custom_indicators.errors import NotFoundError
    def missing(id): raise NotFoundError('PORTFOLIO_RUN_NOT_FOUND', 'missing')
    with pytest.raises(AgentError) as error:
        call('holding-diagnosis', 'diagnosis', request, FakeIndicatorService(tmp_path), {'run': missing})
    assert error.value.code == 'PORTFOLIO_RUN_NOT_FOUND'


def test_runtime_callbacks_reuse_loaded_native_routes_and_explicit_defaults(monkeypatch):
    captured = []
    class Body:
        def __init__(self, **kw): self.fields = kw
    native = SimpleNamespace(
        instrument_products=lambda **kw: {'captured': kw},
        instrument_product_detail=lambda product_id, **kw: {'metrics': {'m_fee': 0.5, 'c_fee': 0.1}},
        ProductCompareAnalysisRequest=Body,
        instrument_product_compare_analysis=lambda product_id, body, **kw: captured.append((product_id, body.fields, kw)) or {'product_id': product_id})
    portfolio = SimpleNamespace(portfolio_service=SimpleNamespace(get_run=lambda id: {'id': id}, diagnose=lambda id, ids: {}, scenario=lambda id, **kw: {}))
    monkeypatch.setitem(sys.modules, 'services.instrument_routes', native)
    monkeypatch.setitem(sys.modules, 'services.portfolio_routes', portfolio)
    callbacks = research_pages.runtime_callbacks()
    callbacks['comparison']({**TARGETS[0], 'management_fee': 0.5, 'custody_fee': 0.1}, {'ranges': RANGES, 'rolling_window_days': 17})
    assert captured == [('510300.SH', {'ranges': RANGES, 'rolling_window_days': 17, 'management_fee': 0.5, 'custody_fee': 0.1}, {'kind': 'etf'})]
    with pytest.raises(AgentError) as error:
        callbacks['comparison']({**TARGETS[0], 'management_fee': 9, 'custody_fee': 0.1}, {'ranges': RANGES, 'rolling_window_days': 17})
    assert error.value.code == 'AGENT_PAGE_FEES_CHANGED'


@pytest.mark.parametrize('name', ['product-research', 'product-compare', 'holding-diagnosis'])
def test_api_captures_all_model_requests_and_context_reads_without_raw_values(tmp_path, monkeypatch, name):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from agent import routes
    from agent.llm import FixtureLLMClient
    from services.llm_settings_routes import router as settings_router
    service = FakeIndicatorService(tmp_path)
    monkeypatch.setattr(routes, 'resolve_service', lambda: service)
    source = {'items': [{'ts_code': '510300.SH', 'name': '沪深300', 'snapshot_values': {'last_close': SENTINEL}}],
              'summary': {'filtered_total': 1, 'avg_m_fee': 0}, 'page': 3, 'page_size': 50, 'total': 1,
              'pit': {'snapshot_is_hindsight': True, 'as_of': None}}
    class Model(FixtureLLMClient):
        async def complete(self, **kw):
            if len(self.requests) == 2:
                receipt = next(json.loads(message['content']) for message in reversed(kw['messages']) if message['role'] == 'tool')
                self._replies.insert(0, {'tool_calls': [{'name': 'context.read', 'arguments': {'operation_id': receipt['context_ref']}}]})
            return await super().complete(**kw)
    operation = {'product-research': 'catalog', 'product-compare': 'comparison', 'holding-diagnosis': 'diagnosis'}[name]
    model = Model([{'tool_calls': [{'name': 'page.read', 'arguments': {'section': 'request'}}]},
                   {'tool_calls': [{'name': 'page.analyze', 'arguments': {'operation': operation}}]}, {'content': '已读取当前批次摘要。'}])
    monkeypatch.setattr(routes, '_llm_client', lambda id: model)
    app = FastAPI(); app.state.agent_page_services = {
        'catalog': lambda **kw: source,
        'comparison': lambda target, parameters: {'product_id': target['product_id'], 'ranges': {
            key: {'window': {'observation_count': 20}, 'metrics': {'cumulativeReturn': 0, 'volatility': None},
                  'normalized_nav': [{'value': SENTINEL}]} for key in RANGES}},
        'run': lambda id: {'id': id, 'immutable': True, 'observation_count': 20, 'summary': {'cumulative_return': 0}, 'portfolio_nav': [SENTINEL]},
        'diagnosis': lambda id, ids: {'run_id': id, 'components': [], 'asset_returns': [[SENTINEL]]}}

    app.include_router(settings_router); app.include_router(routes.router)
    frozen = request_for(name)
    context = page(name, [{key: target[key] for key in ('kind', 'product_id')} for target in frozen.get('targets', [])])
    with TestClient(app) as client:
        client.put('/api/settings/llm', json={'api_key': 'test-key-123456'})
        sid = client.post('/api/agent/sessions', json={'page_context': context.model_dump()}).json()['session_id']
        response = client.post(f'/api/agent/sessions/{sid}/messages', json={'message_id': 'research', 'text': '解释当前产品列表',
            'expected_session_revision': 0, 'page_context': context.model_dump(), 'page_snapshot': snapshot(name, frozen)})
        assert response.status_code == 200, response.text
        assert len(model.requests) == 4
        for request in model.requests:
            data_policy.enforce_request(**request, system_seal=data_policy.seal_text(request['system'], 'primary'))
        assert '987654' not in stable_json(model.requests)
        assert source['items'][0]['snapshot_values']['last_close'] == SENTINEL


def test_compare_pagination_is_explicit_and_does_not_claim_all_targets(tmp_path):
    request = request_for('product-compare')
    request['targets'] = [{**TARGETS[0], 'product_id': f'ETF{i}', 'management_fee': 0.5, 'custody_fee': 0.1} for i in range(10)]
    seen = []
    def callback(target, parameters):
        seen.append(target['product_id']); return {'product_id': target['product_id'], 'ranges': {}}
    result = call('product-compare', 'comparison', request, FakeIndicatorService(tmp_path), {'comparison': callback})['result']
    assert seen == ['ETF0', 'ETF1', 'ETF2'] and result['pagination']['next_target_offset'] == 3
    assert result['pagination']['target_count'] == 10 and result['batch']['count'] == 3


def test_native_compare_callback_uses_existing_preheated_service(monkeypatch, tmp_path):
    from services import instrument_routes
    from test_product_compare_routes import _points
    monkeypatch.setattr(instrument_routes, '_load_timeseries', lambda kind, id: _points())
    monkeypatch.setattr(instrument_routes, 'instrument_product_detail', lambda id, **kw: {'metrics': {'m_fee': 0.5, 'c_fee': 0.1}})
    output = research_pages.runtime_callbacks()['comparison'](
        {**TARGETS[0], 'management_fee': 0.5, 'custody_fee': 0.1},
        {'ranges': {key: {'start_date': None, 'end_date': None} for key in RANGES}, 'rolling_window_days': 3})
    assert output['execution']['nopython'] is True and output['execution']['python_fallback'] == 0
    assert output['ranges']['performance']['metrics']['cumulativeReturn'] > 0
    assert len(output['ranges']['performance']['normalized_nav']) == 8


def test_numeric_product_search_is_an_explicit_identifier_not_an_observation(tmp_path):
    request = request_for('product-research'); request['q'] = '510300'
    request['targets'] = [{'kind': 'etf', 'product_id': '510300'}]
    result = call('product-research', None, request, FakeIndicatorService(tmp_path), {}, section='request')
    frozen = json.loads(result['result']['content'])
    assert frozen['q'] == '510300' and frozen['targets'][0]['product_id'] == '510300'
    request['q'] = '[{"date":"2026-01-01","x":987654.321}]'
    with pytest.raises(AgentError):
        call('product-research', None, request, FakeIndicatorService(tmp_path), {}, section='request')

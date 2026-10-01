"""Scenario migration parity: real repository, graph validator and host authorization."""
import json
import pytest
from custom_indicators.errors import IndicatorDomainError
from integrations.portable_agent.contracts import ContextInput
from research_access import data_policy, scenarios
from research_access.contracts import ResearchError, PageContext
from research_access.scopes import require_tool
from research_access.tools import parse_arguments
from research_access.scenario_contracts import authoring_definition, page_view
from historical_regimes.v2_templates import TEMPLATES_V2
from historical_regimes.v2_service import RegimeGraphV2Service


def context(mode='retrospective'):
    return PageContext(page='historical-regimes', page_instance_id='test', view_state='explicit',
        calculation={'context_kind': 'scenario', 'workspace': 'graph', 'mode': mode, 'as_of': '2019-12-31'})


def execute_tool(name, arguments, *, session, page_context, service=None, page_services=None):
    require_tool(session['scope'], page_context.context_kind, name)
    try:
        return scenarios.execute(name, parse_arguments(name, arguments), page_context, None, page_services,
            checkpoint=lambda **fields: None, cancelled=lambda: False)
    except IndicatorDomainError as exc:
        raise ResearchError(exc.code, exc.message, status_code=exc.status_code) from exc

@pytest.fixture
def definition():
    return authoring_definition(TEMPLATES_V2[0]["definition"])

@pytest.mark.parametrize("injected", [{"study": {"qualification_id": "forged"}}, {"id": "overwrite"},
                                    {"validation": {"walk_forward": False}}])
def test_cannot_author_business_identity_or_qualification(definition, injected):
    with pytest.raises(ResearchError):
        parse_arguments("scenarios.validate", {"definition": {**definition, **injected}})

@pytest.mark.parametrize("key,value", [("rows", [{"close": 123}]), ("window", [1, 2, 3]),
                                      ("unregistered", 123)])
def test_no_inline_observations_or_unregistered_parameters(definition, key, value):
    definition["graph"]["nodes"][0]["parameters"][key] = value
    assert not data_policy.enforce_arguments("scenarios.validate", json.dumps({"definition": definition}))
    projected, _ = page_view({"definition": definition, "mode": "retrospective"})
    assert "definition" not in projected
    assert "notice" in projected

@pytest.mark.parametrize("changed", [{"mode": "realtime"}, {"as_of": "2020-01-01"}])
def test_page_evidence_must_match_research_context(definition, changed):
    snapshot = {"version": 1, "snapshot_id": "snap-" + "c" * 32, "page": "historical-regimes",
        "sections": {"editing": {"definition": definition, "mode": "retrospective",
                                   "as_of": "2019-12-31", **changed}}}
    with pytest.raises(ValueError, match="情景页面快照与当前研究模式或研究日不一致"):
        ContextInput(
                            page_context=context(), page_snapshot=snapshot)

@pytest.fixture
def saved_algorithm(tmp_path):
    from copy import deepcopy
    from historical_regimes.repository import RegimeDefinitionRepository

    # Use the real versioned repository and service reads, without warming numeric execution.
    service = object.__new__(RegimeGraphV2Service)
    service.definitions = RegimeDefinitionRepository(tmp_path / 'definitions.json')
    definition = deepcopy(next(item['definition'] for item in TEMPLATES_V2
        if any(n['type'] == 'filter.butterworth_zero_phase' for n in item['definition']['graph']['nodes'])))
    definition['name'] = '沪深300日频牛熊 · 零相位平滑峰谷'
    definition['default_mode'] = 'retrospective'
    saved = service.definitions.create(definition)
    return service, saved

def test_saved_algorithm_search_reads_repository_and_normalizes_name(saved_algorithm):
    service, saved = saved_algorithm
    callbacks = {'graph': service}
    for query in (saved['name'], '沪深300 日频牛熊·零相位平滑峰谷', '零相位', '沪深３００'):
        result = execute_tool('scenarios.catalog', {'section': 'definitions', 'query': query},
            session={'scope': 'scenario_center'}, page_context=context(), service=None, page_services=callbacks)['result']
        assert result['total'] == 1
        assert result['items'][0]['id'] == saved['id']
        assert result['items'][0]['revision'] == 1
        assert result['items'][0]['default_mode'] == 'retrospective'
        assert 'graph' not in result['items'][0]
    assert parse_arguments('scenarios.catalog', {}).section == 'definitions'

def test_read_exact_revision_paginated_preserves_graph_and_has_no_writes(saved_algorithm):
    from copy import deepcopy
    service, saved = saved_algorithm
    updated = deepcopy(saved)
    next(n for n in updated['graph']['nodes'] if n['type'] == 'filter.butterworth_zero_phase')['parameters']['period'] = 126
    service.definitions.update(saved['id'], 1, updated)
    before = service.definitions.store.path.read_bytes()
    state = {'scope': 'scenario_center'}
    callbacks = {'graph': service}
    nodes, offset = [], 0
    while offset is not None:
        result = execute_tool('scenarios.read', {'definition_id': saved['id'], 'revision': 1, 'offset': offset, 'limit': 3},
            session=state, page_context=context(), service=None, page_services=callbacks)['result']
        assert result['id'] == saved['id'] and result['revision'] == 1
        assert result['states'] == saved['states']
        assert result['outputs'] == saved['graph']['outputs']
        nodes += result['items']
        offset = result['next_offset']
    assert [n['id'] for n in nodes] == [n['id'] for n in saved['graph']['nodes']]
    for original, actual in zip(saved['graph']['nodes'], nodes):
        assert actual.get('parameters', {}) == original.get('parameters', {})
        assert actual.get('inputs', {}) == original.get('inputs', {})
    smooth = next(n for n in nodes if n['type'] == 'filter.butterworth_zero_phase')
    assert smooth['parameters']['period'] == 63
    assert smooth['contract']['supports_realtime'] is False
    assert smooth['contract']['repaints'] is True
    assert '二阶低通' in smooth['contract']['description']
    assert 'draft' not in state
    assert service.definitions.store.path.read_bytes() == before

def test_saved_catalog_keeps_duplicate_names_and_paginates_before_version_read(saved_algorithm):
    service, saved = saved_algorithm
    second = service.definitions.create(saved)
    callbacks = {'graph': service}
    results = []
    for offset in (0, 1):
        result = execute_tool('scenarios.catalog', {'query': saved['name'], 'offset': offset, 'limit': 1},
            session={'scope': 'scenario_center'}, page_context=context(), service=None, page_services=callbacks)['result']
        assert result['total'] == 2
        assert result['next_offset'] == (1 if offset == 0 else None)
        results.append(result['items'][0]['id'])
    assert set(results) == {saved['id'], second['id']}

def test_saved_read_requires_identity_and_revision_and_respects_scope(saved_algorithm):
    service, saved = saved_algorithm
    callbacks = {'graph': service}
    with pytest.raises(ResearchError):
        parse_arguments('scenarios.read', {'definition_id': saved['id']})
    for definition_id, revision, code in [(saved['id'], 99, 'REGIME_DEFINITION_VERSION_NOT_FOUND'),
                                           ('missing', 1, 'REGIME_DEFINITION_NOT_FOUND')]:
        with pytest.raises(ResearchError) as exc:
            execute_tool('scenarios.read', {'definition_id': definition_id, 'revision': revision},
                session={'scope': 'scenario_center'}, page_context=context(), service=None, page_services=callbacks)
        assert exc.value.code == code
    with pytest.raises(ResearchError):
        execute_tool('scenarios.read', {'definition_id': saved['id'], 'revision': 1},
            session={'scope': 'indicator_center'}, page_context=context(), service=None, page_services=callbacks)

def test_saved_read_withholds_raw_values_and_still_explains_supported_nodes(saved_algorithm):
    from copy import deepcopy
    service, saved = saved_algorithm
    payload = deepcopy(saved)
    payload['graph']['nodes'].insert(0, {'id': 'inline', 'type': 'source.inline', 'inputs': {},
        'parameters': {'rows': [{'date': '2019-01-01', 'value': 123456.789}]}})
    smooth = next(n for n in payload['graph']['nodes'] if n['type'] == 'filter.butterworth_zero_phase')
    smooth['parameters']['unknown_rows'] = [987654.321]
    payload['study'] = {'qualification_id': 'secret-qualification'}
    payload['graph']['edges'] = [{'source': {'node_id': 'inline', 'port': 'value'},
                                 'target': {'node_id': smooth['id'], 'port': 'value'}}]
    service.definitions.update(saved['id'], 1, payload)
    result = execute_tool('scenarios.read', {'definition_id': saved['id'], 'revision': 2, 'limit': 16},
        session={'scope': 'scenario_center'}, page_context=context(), service=None,
        page_services={'graph': service})
    read = result['result']
    assert read['items'][0]['parameters'] == {} and read['items'][0]['notice']
    assert next(n for n in read['items'] if n['id'] == smooth['id'])['parameters'] == {'period': 63}
    assert read['edges'] == payload['graph']['edges']
    assert result['truncated']
    public = json.dumps(read)
    assert all(x not in public for x in ('123456.789', '987654.321', 'secret-qualification', 'unknown_rows'))


def test_real_graph_validator_rejects_realtime_and_missing_links(definition):
    service = object.__new__(RegimeGraphV2Service)
    def run(mode):
        return execute_tool('scenarios.validate', {'definition': definition}, session={'scope': 'scenario_center'},
            page_context=context(mode), page_services={'graph': service})
    valid = run('retrospective')
    assert valid['result']['valid'] and valid['_scenario_payload']['definition'] == definition
    assert not run('realtime')['result']['valid']
    definition['graph']['outputs']['state']['node_id'] = 'missing'
    assert not run('retrospective')['result']['valid']


def test_source_catalog_and_page_read_do_not_expose_observations(definition):
    from types import SimpleNamespace
    from research_access.tools import execute_business
    services = {'sources': SimpleNamespace(catalog=lambda **kw: {'total': 1, 'items': [{
        'id': 'index:index_daily:000300.SH', 'name': '沪深300', 'status': 'available',
        'regime_node_type': 'source.index', 'binding_parameters': {'ts_code': '000300.SH', 'field': 'close'},
        'fields': [{'name': 'close', 'label': '收盘点位'}], 'rows': [{'close': 987654.321}]}]})}
    result = execute_tool('scenarios.catalog', {'section': 'sources'}, session={'scope': 'scenario_center'},
        page_context=context(), page_services=services)
    assert result['result']['items'][0]['source_node']['parameters']['ts_code'] == '000300.SH'
    assert '987654' not in json.dumps(result)
    result = execute_business('page.read', {'section': 'editing', 'limit': 8000}, authoring={'scope': 'scenario_center', 'id': 'test'},
        page_context=context(), service=None, page_snapshot={'page': 'historical-regimes', 'sections': {
            'editing': {'definition': definition, 'mode': 'retrospective', 'as_of': '2019-12-31'}}})
    assert json.loads(result['result']['content'])['definition'] == definition
    assert data_policy.enforce_arguments('scenarios.validate', json.dumps({'definition': definition}))

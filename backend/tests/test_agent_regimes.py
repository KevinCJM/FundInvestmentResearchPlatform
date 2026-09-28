"""Scenario authoring uses the real graph validator and the shared agent boundary."""
import json

import pytest

from agent import data_policy, regimes
from agent.contracts import AgentError, AgentMessageRequest, PageContext
from agent.research_runtime import system_prompt
from agent.scopes import allowed_tools, validate_page_context
from agent.tools import execute_tool, parse_arguments
from historical_regimes.v2_registry import node_catalog
from historical_regimes.v2_templates import TEMPLATES_V2, list_templates_v2
from historical_regimes.v2_service import RegimeGraphV2Service


def context(mode="retrospective"):
    return PageContext(page="regime-workbench", page_instance_id="test", view_state="explicit",
        calculation={"context_kind": "regime_graph", "mode": mode, "as_of": "2019-12-31"})


@pytest.fixture
def definition():
    return regimes.authoring_definition(TEMPLATES_V2[0]["definition"])


@pytest.fixture
def callbacks():
    # infer has no instance state: exercise the production parser, inspector and timing rules.
    return {"regimes.catalog": node_catalog, "regimes.templates": lambda: {"items": list_templates_v2()},
            "regimes.infer": lambda definition, mode: RegimeGraphV2Service.infer(None, definition, mode)}


def test_registered_scenario_scope_is_separate():
    assert set(allowed_tools("scenario_center", "regime_graph")) == {
        "regimes.lookup", "regimes.read", "regimes.template", "regimes.validate", "page.read", "context.read",
        "task.read", "task.plan", "memory.propose"}
    assert validate_page_context(context(), pit_off=False) == "scenario_center"
    wrong = context().model_copy(update={"page": "indicator-studio"})
    with pytest.raises(AgentError):
        validate_page_context(wrong, pit_off=False)


def test_real_validator_creates_only_session_draft_and_checks_mode(definition, callbacks):
    state = {"scope": "scenario_center", "context_hash": "current"}
    result = execute_tool("regimes.validate", {"definition": definition}, session=state,
                          page_context=context(), service=None, page_services=callbacks)
    assert result["ok"] is True
    assert state["draft"]["valid"] is True
    assert state["draft"]["artifact_kind"] == "regime_graph"
    assert state["draft"]["compile_token"] is None
    assert state["draft"]["definition"] == definition
    execute_tool("regimes.validate", {"definition": definition}, session=state,
                 page_context=context("realtime"), service=None, page_services=callbacks)
    assert state["draft"]["valid"] is False
    assert any(d["code"] == "REGIME_REALTIME_UNSUPPORTED" for d in state["draft"]["diagnostics"])


@pytest.mark.parametrize("injected", [{"study": {"qualification_id": "forged"}}, {"id": "overwrite"},
                                    {"validation": {"walk_forward": False}}])
def test_cannot_author_business_identity_or_qualification(definition, injected):
    with pytest.raises(AgentError):
        parse_arguments("regimes.validate", {"definition": {**definition, **injected}})


@pytest.mark.parametrize("key,value", [("rows", [{"close": 123}]), ("window", [1, 2, 3]),
                                      ("unregistered", 123)])
def test_no_inline_observations_or_unregistered_parameters(definition, key, value):
    definition["graph"]["nodes"][0]["parameters"][key] = value
    assert not data_policy.enforce_arguments("regimes.validate", json.dumps({"definition": definition}))
    projected, _ = regimes.page_view({"definition": definition, "mode": "retrospective"})
    assert "definition" not in projected
    assert "notice" in projected


def test_graph_arguments_system_and_page_evidence_pass_model_gate(definition, callbacks):
    assert data_policy.enforce_arguments("regimes.validate", json.dumps({"definition": definition}))
    state = {"scope": "scenario_center", "context_hash": "current"}
    execute_tool("regimes.validate", {"definition": definition}, session=state,
                 page_context=context(), service=None, page_services=callbacks)
    data_policy.check_system_text(system_prompt(state, context(), "test"))
    snapshot = {"version": 1, "snapshot_id": "snap-" + "a" * 32, "page": "regime-workbench",
                "sections": {"editing": {"definition": definition, "mode": "retrospective", "as_of": "2019-12-31"}}}
    AgentMessageRequest(message_id="m", expected_session_revision=0, text="解释算法",
                        page_context=context(), page_snapshot=snapshot)
    result = execute_tool("page.read", {"section": "editing"}, session=state, page_context=context(),
                          service=None, page_snapshot=snapshot)
    assert result["ok"]
    assert "2019-12-31" in json.dumps(result)


def test_missing_mount_and_wrong_scope_fail_closed(definition):
    with pytest.raises(AgentError):
        execute_tool("regimes.validate", {"definition": definition}, session={"scope": "scenario_center"},
                     page_context=context(), service=None)
    with pytest.raises(AgentError):
        execute_tool("regimes.validate", {"definition": definition}, session={"scope": "indicator_center"},
                     page_context=context(), service=None)


@pytest.mark.parametrize("changed", [{"mode": "realtime"}, {"as_of": "2020-01-01"}])
def test_page_evidence_must_match_research_context(definition, changed):
    snapshot = {"version": 1, "snapshot_id": "snap-" + "c" * 32, "page": "regime-workbench",
        "sections": {"editing": {"definition": definition, "mode": "retrospective",
                                   "as_of": "2019-12-31", **changed}}}
    with pytest.raises(ValueError, match="情景页面快照与当前研究模式或研究日不一致"):
        AgentMessageRequest(message_id="m", expected_session_revision=0, text="解释算法",
                            page_context=context(), page_snapshot=snapshot)


def test_catalog_is_contracts_only(callbacks):
    result = execute_tool("regimes.lookup", {"kind": "nodes", "query": "rolling.slope"},
        session={"scope": "scenario_center"}, page_context=context(), service=None, page_services=callbacks)
    assert result["ok"]
    text = json.dumps(result)
    assert "rolling.slope" in text
    assert "runtime" not in text and "startup_prewarm" not in text


def test_invalid_connection_is_not_applicable(definition, callbacks):
    definition["graph"]["outputs"]["state"]["node_id"] = "missing"
    state = {"scope": "scenario_center"}
    execute_tool("regimes.validate", {"definition": definition}, session=state,
        page_context=context(), service=None, page_services=callbacks)
    assert not state["draft"]["valid"]


# The same HTTP harness and deterministic model used by indicator tests.
from test_agent_api import harness  # noqa: E402,F401


def test_http_model_roundtrip_draft_restore_and_scope_isolation(harness, monkeypatch, definition, callbacks):
    harness.client.app.state.agent_page_services = callbacks
    ctx = context().model_dump()
    sid = harness.create_session(ctx)['session_id']
    harness.script(monkeypatch, [
        {'tool_calls': [{'name': 'page.read', 'arguments': {'section': 'editing'}}]},
        {'tool_calls': [{'name': 'regimes.validate', 'arguments': {'definition': definition}}]},
        {'content': '草稿已校验，请检查后应用到编辑器。'},
    ])
    snapshot = {'version': 1, 'snapshot_id': 'snap-' + 'b' * 32, 'page': 'regime-workbench',
        'sections': {'editing': {'definition': definition, 'mode': 'retrospective', 'as_of': '2019-12-31'}}}
    response = harness.message(sid, {'message_id': 'scenario-draft', 'expected_session_revision': 0,
        'text': '校验当前情景算法。', 'page_context': ctx, 'page_snapshot': snapshot})
    assert response.status_code == 200, response.text
    restored = harness.client.get(f'/api/agent/sessions/{sid}').json()
    assert restored['draft']['valid']
    assert restored['draft']['artifact_kind'] == 'regime_graph'
    assert restored['draft']['definition']['graph'] == definition['graph']
    assert restored['active_run']['status'] == 'completed'
    assert not harness.fake.create_calls and not harness.fake.evaluate_calls


def test_source_catalog_exposes_bindings_not_observations(callbacks):
    callbacks['regimes.sources'] = lambda **kw: {'total': 1, 'items': [{
        'id': 'index:index_daily:000300.SH', 'name': '沪深300', 'status': 'available',
        'regime_node_type': 'source.index', 'binding_parameters': {'ts_code': '000300.SH', 'field': 'close'},
        'fields': [{'name': 'close', 'label': '收盘点位'}], 'rows': [{'close': 123.45}],
    }]}
    result = execute_tool('regimes.lookup', {'kind': 'sources', 'query': '沪深300'},
        session={'scope': 'scenario_center'}, page_context=context(), service=None, page_services=callbacks)
    item = result['result']['items'][0]
    assert item['source_node']['parameters']['ts_code'] == '000300.SH'
    assert 'rows' not in item and '123.45' not in json.dumps(result['result'])


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
    callbacks = regimes.runtime_callbacks(service)
    for query in (saved['name'], '沪深300 日频牛熊·零相位平滑峰谷', '零相位', '沪深３００'):
        result = execute_tool('regimes.lookup', {'kind': 'definitions', 'query': query},
            session={'scope': 'scenario_center'}, page_context=context(), service=None, page_services=callbacks)['result']
        assert result['total'] == 1
        assert result['items'][0]['id'] == saved['id']
        assert result['items'][0]['revision'] == 1
        assert result['items'][0]['default_mode'] == 'retrospective'
        assert 'graph' not in result['items'][0]
    assert parse_arguments('regimes.lookup', {}).kind == 'definitions'


def test_read_exact_revision_paginated_preserves_graph_and_has_no_writes(saved_algorithm):
    from copy import deepcopy
    service, saved = saved_algorithm
    updated = deepcopy(saved)
    next(n for n in updated['graph']['nodes'] if n['type'] == 'filter.butterworth_zero_phase')['parameters']['period'] = 126
    service.definitions.update(saved['id'], 1, updated)
    before = service.definitions.store.path.read_bytes()
    state = {'scope': 'scenario_center'}
    callbacks = regimes.runtime_callbacks(service)
    nodes, offset = [], 0
    while offset is not None:
        result = execute_tool('regimes.read', {'definition_id': saved['id'], 'revision': 1, 'offset': offset, 'limit': 3},
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
    callbacks = regimes.runtime_callbacks(service)
    results = []
    for offset in (0, 1):
        result = execute_tool('regimes.lookup', {'query': saved['name'], 'offset': offset, 'limit': 1},
            session={'scope': 'scenario_center'}, page_context=context(), service=None, page_services=callbacks)['result']
        assert result['total'] == 2
        assert result['next_offset'] == (1 if offset == 0 else None)
        results.append(result['items'][0]['id'])
    assert set(results) == {saved['id'], second['id']}


def test_saved_read_requires_identity_and_revision_and_respects_scope(saved_algorithm):
    service, saved = saved_algorithm
    callbacks = regimes.runtime_callbacks(service)
    with pytest.raises(AgentError):
        parse_arguments('regimes.read', {'definition_id': saved['id']})
    for definition_id, revision, code in [(saved['id'], 99, 'REGIME_DEFINITION_VERSION_NOT_FOUND'),
                                           ('missing', 1, 'REGIME_DEFINITION_NOT_FOUND')]:
        with pytest.raises(AgentError) as exc:
            execute_tool('regimes.read', {'definition_id': definition_id, 'revision': revision},
                session={'scope': 'scenario_center'}, page_context=context(), service=None, page_services=callbacks)
        assert exc.value.code == code
    with pytest.raises(AgentError):
        execute_tool('regimes.read', {'definition_id': saved['id'], 'revision': 1},
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
    result = execute_tool('regimes.read', {'definition_id': saved['id'], 'revision': 2, 'limit': 16},
        session={'scope': 'scenario_center'}, page_context=context(), service=None,
        page_services=regimes.runtime_callbacks(service))
    read = result['result']
    assert read['items'][0]['parameters'] == {} and read['items'][0]['notice']
    assert next(n for n in read['items'] if n['id'] == smooth['id'])['parameters'] == {'period': 63}
    assert read['edges'] == payload['graph']['edges']
    assert result['truncated']
    public = json.dumps(read)
    assert all(x not in public for x in ('123456.789', '987654.321', 'secret-qualification', 'unknown_rows'))


def test_http_library_question_delivers_saved_graph_to_model_without_draft(harness, monkeypatch, saved_algorithm):
    from test_agent_api import tool_messages
    service, saved = saved_algorithm
    harness.client.app.state.agent_page_services = regimes.runtime_callbacks(service)
    ctx = context().model_dump()
    sid = harness.create_session(ctx)['session_id']
    script = harness.script(monkeypatch, [
        {'tool_calls': [{'name': 'page.read', 'arguments': {'section': 'editing'}}]},
        {'tool_calls': [{'name': 'regimes.lookup', 'arguments': {'kind': 'definitions', 'query': saved['name']}}]},
        {'tool_calls': [{'name': 'regimes.read', 'arguments': {'definition_id': saved['id'], 'revision': 1, 'limit': 16}}]},
        {'content': '已读取保存版本：先进行零相位平滑，再识别峰谷；这是事后参考，不能作为当时可交易信号。'},
    ])
    snapshot = {'version': 1, 'snapshot_id': 'snap-' + 'd' * 32, 'page': 'regime-workbench',
        'sections': {'editing': {'definition': None, 'mode': 'retrospective', 'as_of': '2019-12-31'}}}
    response = harness.message(sid, {'message_id': 'explain-saved', 'expected_session_revision': 0,
        'text': f'解释“{saved["name"]}”的逻辑和经济含义。', 'page_context': ctx, 'page_snapshot': snapshot})
    assert response.status_code == 200, response.text
    receipts = tool_messages(script, 3)
    read = next(item for item in receipts if item.get('tool') == 'regimes.read')
    assert read['ok'] is True
    assert read['result']['id'] == saved['id']
    smooth = next(n for n in read['result']['items'] if n['type'] == 'filter.butterworth_zero_phase')
    assert smooth['parameters']['period'] == 63
    assert smooth['contract']['supports_realtime'] is False
    restored = harness.client.get(f'/api/agent/sessions/{sid}').json()
    assert not restored.get('draft')
    assert restored['active_run']['status'] == 'completed'
    assert not harness.fake.create_calls and not harness.fake.evaluate_calls

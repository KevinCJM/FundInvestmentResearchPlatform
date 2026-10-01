"""Host contracts tested through actual HTTP and the real indicator service, without an agent runtime import."""
import json
import time
import uuid

import jwt
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from custom_indicators.service import CustomIndicatorService
from integrations.portable_agent.routes import install
from research_access import data_policy
from research_access.authoring import stable_hash
from research_access.store import ResearchStore
from test_custom_indicator_service import _draft, _write_market_data

SERVICE_TOKEN = 'offline-host-tool-credential-00000000000000'
IDENTITY_KEY = 'offline-identity-signing-key-00000000000000'


def test_release_lock_is_required_in_production_and_validated(tmp_path, monkeypatch):
    from integrations.portable_agent.service import release_contract, REQUIRED_CAPABILITIES
    from research_access.contracts import ResearchError
    monkeypatch.setenv('APP_ENV', 'production')
    monkeypatch.delenv('PORTABLE_AGENT_RELEASE_FILE', raising=False)
    with pytest.raises(ResearchError, match='发布版本'):
        release_contract()
    path = tmp_path/'release.json'
    monkeypatch.setenv('PORTABLE_AGENT_RELEASE_FILE', str(path))
    release = {'source_commit': 'a'*40, 'image_digest': 'example/agent@sha256:'+'b'*64,
               'protocol_major': 2, 'widget_version': '0.2.0', 'manifest_sha256': 'c'*64,
               'required_capabilities': REQUIRED_CAPABILITIES}
    path.write_text(json.dumps(release))
    assert release_contract()['expected_release']['manifest_sha256'] == 'c'*64
    for change in ({'required_capabilities': ['profiles']}, {'image_digest': 'example/agent:latest'}, {'protocol_major': 1}):
        path.write_text(json.dumps({**release, **change}))
        with pytest.raises(ResearchError, match='发布配置'):
            release_contract()
    path.write_text(json.dumps(release))
    monkeypatch.setenv('PORTABLE_AGENT_IMAGE', 'example/agent@sha256:'+'d'*64)
    with pytest.raises(ResearchError, match='发布配置'):
        release_contract()


@pytest.fixture
def host(tmp_path, monkeypatch):
    monkeypatch.setenv('CUSTOM_INDICATOR_DATA_DIR', str(tmp_path))
    monkeypatch.setenv('INDICATOR_PROCESS_WORKERS', '1')
    monkeypatch.setenv('PORTABLE_AGENT_AUTH_MODE', 'jwt')
    monkeypatch.setenv('PORTABLE_AGENT_WORKSPACE', 'lab')
    monkeypatch.setenv('PORTABLE_AGENT_IDENTITY_KEY', IDENTITY_KEY)
    monkeypatch.setenv('PORTABLE_AGENT_SERVICE_TOKEN', SERVICE_TOKEN)
    monkeypatch.setenv('PORTABLE_AGENT_ISSUER_KEY', '1'*64)
    _write_market_data(tmp_path)
    service = CustomIndicatorService(tmp_path, tmp_path)
    permissions = ['*']
    app = FastAPI()
    bridge = install(app, indicator_service=service, page_services={}, store=ResearchStore(tmp_path/'research_access'),
                     authority=lambda subject, workspace: {'sub': subject, 'workspace': workspace, 'scopes': list(permissions)})
    now = int(time.time())
    token = jwt.encode({'sub': 'alice', 'workspace': 'lab', 'scopes': ['*'], 'iat': now, 'exp': now+3600,
                        'iss': 'fund-research', 'aud': 'fund-research-platform'}, IDENTITY_KEY, algorithm='HS256')
    with TestClient(app) as client:
        client.headers['Authorization'] = 'Bearer '+token
        yield client, bridge, service, permissions
    service.close_compute_engine()


def context(client):
    response = client.post('/api/integrations/portable-agent/contexts', json={'page_context': {
        'page': 'indicator-studio', 'page_instance_id': 'workbench', 'context_revision': 0, 'view_state': 'unknown',
        'calculation': {'context_kind': 'single_product', 'targets': [], 'period': '1M', 'as_of': None}}})
    assert response.status_code == 200, response.text
    value = response.json()
    boot = client.post('/api/integrations/portable-agent/bootstrap', json={'context_ref': value['ref']})
    assert boot.status_code == 200, boot.text
    return value


def tool(client, ctx, name, args, oid=None):
    oid = oid or uuid.uuid4().hex
    body = {'protocol_version': 2, 'application': 'fund-research', 'subject': 'alice', 'session_id': 'conversation',
            'run_id': 'turn', 'operation_id': oid, 'context': ctx, 'arguments': args}
    headers = {'Authorization': 'Bearer '+SERVICE_TOKEN, 'Idempotency-Key': oid}
    response = client.post('/internal/portable-agent/tools/'+name, json=body, headers=headers)
    assert response.status_code in (200, 202), response.text
    deadline = time.monotonic()+40
    while time.monotonic() < deadline:
        state = client.get('/internal/portable-agent/operations/'+oid, headers=headers)
        assert state.status_code == 200, state.text
        if state.json()['status'] not in {'accepted', 'running', 'stop_requested'}:
            return state.json(), body
        time.sleep(.02)
    raise AssertionError(state.text)


def validated(client, ctx):
    result, payload = tool(client, ctx, 'metrics_validate', {'definition': _draft()})
    assert result['status'] == 'succeeded', result
    assert data_policy.verify(result['model'], 'metrics.validate')
    assert '_progress_payload' not in json.dumps(result)
    aid = result['artifact']['authoring_id']
    draft = client.get('/api/custom-indicators/authorings/'+aid).json()
    assert draft['draft']['valid']
    return aid, draft, result, payload


def confirmation(client, ctx, aid, draft):
    body = {'context_ref': ctx['ref'], 'expected_revision': draft['revision'], 'definition_hash': draft['draft']['definition_hash']}
    response = client.post(f'/api/custom-indicators/authorings/{aid}/confirmations', json=body)
    assert response.status_code == 200, response.text
    return body | {'confirmation_id': response.json()['id'], 'request_id': uuid.uuid4().hex, 'confirmed': True}


def test_draft_confirmation_save_replay_and_no_service_identity_write(host):
    client, bridge, service, _ = host
    ctx = context(client)
    aid, draft, result, payload = validated(client, ctx)
    before = len(service.indicators.list())
    body = confirmation(client, ctx, aid, draft)
    assert len(service.indicators.list()) == before  # Previewing is not approval.
    with bridge.store.db() as db:
        revisions = db.execute('SELECT COUNT(*) FROM authoring_revisions WHERE authoring_id=?', (aid,)).fetchone()[0]
    for name, args in [('metrics_lookup', {'query': 'std'}), ('metrics_infer', {'expression': 'std(returns, 1)'}), ('metrics_draft_read', {})]:
        assert tool(client, ctx, name, args)[0]['status'] == 'succeeded'
    assert client.get('/api/custom-indicators/authorings/'+aid).json()['revision'] == draft['revision']
    with bridge.store.db() as db:
        assert db.execute('SELECT COUNT(*) FROM authoring_revisions WHERE authoring_id=?', (aid,)).fetchone()[0] == revisions
    url = f'/api/custom-indicators/authorings/{aid}/commits'
    assert client.post(url, json=body, headers={'Authorization': 'Bearer '+SERVICE_TOKEN}).status_code == 401
    saved = client.post(url, json=body)
    assert saved.status_code == 200, saved.text
    assert client.post(url, json=body).json()['indicator_id'] == saved.json()['indicator_id']
    next_body = confirmation(client, ctx, aid, draft)
    assert client.post(url, json=next_body).json()['indicator_id'] == saved.json()['indicator_id']
    assert len(service.indicators.list()) == before+1


def test_catalog_write_has_atomic_confirmation_version_guard(host):
    client, _, service, _ = host
    ctx = context(client)
    aid, draft, _, _ = validated(client, ctx)
    body = confirmation(client, ctx, aid, draft)
    service.create_indicator(_draft(name='Another author'))
    before = len(service.indicators.list())
    url = f'/api/custom-indicators/authorings/{aid}/commits'
    rejected = client.post(url, json=body)
    assert rejected.status_code == 409 and rejected.json()['error']['code'] == 'CONFIRMATION_STALE'
    assert len(service.indicators.list()) == before
    assert client.post(url, json=confirmation(client, ctx, aid, draft)).status_code == 200


def test_grant_cannot_be_extended_and_input_cannot_send_raw_market_values(host):
    client, bridge, _, permissions = host
    ctx = context(client)
    headers = {'Authorization': 'Bearer '+SERVICE_TOKEN}
    body = {'application': 'fund-research', 'subject': 'alice', 'context': ctx, 'action': 'run'}
    assert client.post('/internal/portable-agent/authorize', headers=headers, json=body).status_code == 200
    assert client.post('/internal/portable-agent/authorize', headers=headers, json=body | {'context': ctx | {'extra': 'injected'}}).status_code == 403
    raw = {'application': 'fund-research', 'subject': 'alice', 'context': ctx, 'phase': 'input',
           'messages': [{'role': 'user', 'content': 'date,x\n2026-01-02,987654.321'}]}
    assert client.post('/internal/portable-agent/admission', headers=headers, json=raw).status_code == 422
    permissions.clear()
    assert client.post('/internal/portable-agent/authorize', headers=headers, json=body).status_code == 403


def test_signed_tool_wire_and_late_dependency_change(host):
    client, bridge, service, _ = host
    ctx = context(client)
    _, _, result, request = validated(client, ctx)
    wire = {'model': 'offline', 'messages': [{'role': 'system', 'content': 'Help with the registered tools.'},
        {'role': 'user', 'content': '请校验这个指标'},
        {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'c', 'type': 'function', 'function': {
            'name': 'metrics_validate', 'arguments': json.dumps(request['arguments'])}}]},
        {'role': 'tool', 'tool_call_id': 'c', 'content': json.dumps(result['model'])}], 'tools': []}
    body = {'application': 'fund-research', 'subject': 'alice', 'context': ctx, 'phase': 'before', 'wire': wire, 'payload_hash': stable_hash(wire)}
    headers = {'Authorization': 'Bearer '+SERVICE_TOKEN}
    response = client.post('/internal/portable-agent/admission', headers=headers, json=body)
    assert response.status_code == 200, response.text
    receipt = response.json()
    service.create_indicator(_draft(name='Changed catalog'))
    after = {k: v for k, v in body.items() if k != 'wire'} | {'phase': 'after', 'receipt_id': receipt['receipt_id']}
    assert client.post('/internal/portable-agent/admission', headers=headers, json=after).status_code == 409


@pytest.mark.parametrize('changed', ['graph', 'events', 'stress', 'source_manifest', 'source_indicator', 'releases'])
def test_scenario_catalog_mutation_invalidates_model_evidence(host, tmp_path, changed):
    from historical_regimes.v2_service import RegimeGraphV2Service
    from historical_regimes.v2_templates import TEMPLATES_V2
    from scenario_stress.service import ScenarioStressService
    from scenario_stress.published import PublishedScenarioService
    from custom_indicators.repository import IndicatorRepository
    from test_research_series_routes import _write_fixture
    from test_scenario_stress import _template
    client, bridge, _, _ = host
    graph = RegimeGraphV2Service(tmp_path, tmp_path)
    stress = ScenarioStressService(tmp_path, tmp_path)
    published = PublishedScenarioService(tmp_path)
    sources = _write_fixture(tmp_path/'sources')
    bridge.pages.update(graph=graph, stress=stress, published=published, sources=sources)
    body = {'page_context': {'page': 'historical-regimes', 'page_instance_id': 'mutable-catalog',
        'view_state': 'unknown', 'calculation': {'context_kind': 'scenario', 'workspace': 'graph'}}}
    ctx = client.post('/api/integrations/portable-agent/contexts', json=body).json()
    result, _ = tool(client, ctx, 'scenarios_catalog', {'section': 'definitions'})
    assert result['status'] == 'succeeded'
    messages = [{'role': 'user', 'content': '列出情景定义'},
        {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'c', 'type': 'function', 'function': {
            'name': 'scenarios_catalog', 'arguments': '{"section":"definitions"}'}}]},
        {'role': 'tool', 'tool_call_id': 'c', 'content': json.dumps(result['model'])}]
    wire = {'model': 'offline', 'messages': messages, 'tools': []}
    before = {'application': 'fund-research', 'subject': 'alice', 'context': ctx,
              'phase': 'before', 'wire': wire, 'payload_hash': stable_hash(wire)}
    headers = {'Authorization': 'Bearer '+SERVICE_TOKEN}
    admitted = client.post('/internal/portable-agent/admission', headers=headers, json=before)
    assert admitted.status_code == 200, admitted.text
    if changed == 'graph':
        graph.definitions.create(TEMPLATES_V2[0]['definition'])
    elif changed == 'events':
        graph.event_library.create({'name': '新事件', 'windows': [{'id': 'event-window', 'label': '窗口',
            'start_date': '2020-01-01', 'end_date': '2020-01-31', 'rationale': '固定测试窗口'}]})
    elif changed == 'stress':
        stress.create_definition(_template('factor_path'))
    elif changed == 'source_manifest':
        path = sources.data_dir/'tushare_active.json'
        manifest = json.loads(path.read_text())
        path.write_text(json.dumps({**manifest, 'generation': 'next-catalog-generation'}))
    elif changed == 'source_indicator':
        IndicatorRepository(sources.workspace_data_dir/'custom_indicators.json', []).create(_draft(name='新增来源'))
    else:
        published.artifacts.save('release', {'name': '新情景版本', 'lineage': [],
            'effective_at': '2020-01-01T00:00:00+00:00', 'expires_at': '2099-01-01T00:00:00+00:00'})
    after = {key: value for key, value in before.items() if key != 'wire'}
    after.update(phase='after', receipt_id=admitted.json()['receipt_id'])
    assert client.post('/internal/portable-agent/admission', headers=headers, json=after).status_code == 409
    fresh = client.post('/api/integrations/portable-agent/contexts', json=body).json()
    assert fresh['hash'] != ctx['hash']
    prepared = client.post('/internal/portable-agent/admission', headers=headers, json={
        'application': 'fund-research', 'subject': 'alice', 'context': fresh, 'phase': 'prepare', 'messages': messages})
    assert prepared.status_code == 200, prepared.text
    assert json.loads(prepared.json()['messages'][-1]['content'])['status'] == 'historical_stale'
    assert tool(client, fresh, 'scenarios_catalog', {'section': 'definitions'})[0]['status'] == 'succeeded'


@pytest.mark.parametrize('as_of', ['2026-13-01', '2999-01-01'])
def test_scenario_catalog_dates_return_controlled_validation(host, tmp_path, as_of):
    from scenario_stress.published import PublishedScenarioService
    client, bridge, _, _ = host
    bridge.pages['published'] = PublishedScenarioService(tmp_path)
    page = {'page': 'published-scenarios', 'page_instance_id': 'invalid-date', 'view_state': 'unknown',
            'calculation': {'context_kind': 'scenario', 'workspace': 'published', 'as_of': as_of}}
    with TestClient(client.app, raise_server_exceptions=False) as requests:
        requests.headers.update(client.headers)
        response = requests.post('/api/integrations/portable-agent/contexts', json={'page_context': page})
        assert response.status_code == 422, response.text
        assert response.json()['error']['code'] in {'VALIDATION_ERROR', 'FUTURE_RESEARCH_DATE'}
        page['calculation']['as_of'] = '2020-01-01'
        assert requests.post('/api/integrations/portable-agent/contexts', json={'page_context': page}).status_code == 200


@pytest.mark.parametrize('fault', ['manifest_json', 'manifest_target', 'manifest_unreadable', 'source_indicators', 'graph_store', 'release_index',
    'graph_record', 'graph_record_type', 'source_record_type', 'release_index_symlink', 'release_index_oversize', 'release_checksum'])
def test_scenario_catalog_storage_failure_is_controlled_and_recoverable(host, tmp_path, monkeypatch, fault):
    from pathlib import Path
    from historical_regimes.v2_service import RegimeGraphV2Service
    from scenario_stress.published import PublishedScenarioService
    from test_research_series_routes import _write_fixture
    client, bridge, _, _ = host
    graph, published, sources = RegimeGraphV2Service(tmp_path, tmp_path), PublishedScenarioService(tmp_path), _write_fixture(tmp_path/'sources')
    bridge.pages.update(graph=graph, published=published, sources=sources)
    if fault == 'release_checksum':
        release = published.artifacts.save('release', {'name': '正常发布', 'lineage': [],
            'effective_at': '2020-01-01T00:00:00+00:00', 'expires_at': '2099-01-01T00:00:00+00:00'})
    body = {'page_context': {'page': 'historical-regimes', 'page_instance_id': 'unhealthy-catalog',
        'view_state': 'unknown', 'calculation': {'context_kind': 'scenario', 'workspace': 'graph'}}}
    ctx = client.post('/api/integrations/portable-agent/contexts', json=body).json()
    incoming = {'application': 'fund-research', 'subject': 'alice', 'context': ctx,
                'phase': 'input', 'messages': [{'role': 'user', 'content': '继续研究'}]}
    headers = {'Authorization': 'Bearer '+SERVICE_TOKEN}
    path = (sources.workspace_data_dir/'custom_indicators.json' if fault.startswith('source_') else
            graph.definitions.store.path if fault.startswith('graph_') else
            published.artifacts.root/release['id']/'manifest.json' if fault == 'release_checksum' else
            published.artifacts.index.path if fault.startswith('release_') else sources.data_dir/'tushare_active.json')
    original = path.read_text() if path.exists() else None
    path.parent.mkdir(parents=True, exist_ok=True)
    with monkeypatch.context() as patch:
        if fault == 'manifest_unreadable':
            read_text = Path.read_text
            def unreadable(target, *args, **kwargs):
                if target == path:
                    raise PermissionError('fixture: manifest unreadable')
                return read_text(target, *args, **kwargs)
            patch.setattr(Path, 'read_text', unreadable)
        elif fault == 'release_index_symlink':
            target = path.with_name('index-target.json')
            target.write_text(original or '{"items":[]}')
            path.unlink(missing_ok=True)
            path.symlink_to(target)
        elif fault == 'release_index_oversize':
            path.write_text(' ' * 16_000_001)
        elif fault == 'release_checksum':
            path.write_text(json.dumps({**json.loads(original), 'name': '篡改发布'}))
        elif fault in {'graph_record', 'graph_record_type', 'source_record_type'}:
            entry = {} if fault == 'graph_record' else None if fault == 'graph_record_type' else {'current': None}
            path.write_text(json.dumps({'items': [entry]}))
        else:
            path.write_text(json.dumps({**json.loads(original), 'snapshot_dir': 'missing-snapshot'}) if fault == 'manifest_target' else '{broken')
        with TestClient(client.app, raise_server_exceptions=False) as requests:
            requests.headers.update(client.headers)
            for response in (requests.post('/api/integrations/portable-agent/contexts', json=body),
                             requests.post('/internal/portable-agent/admission', headers=headers, json=incoming)):
                assert response.status_code == 503, response.text
                assert response.json()['error']['code'] == 'RESEARCH_SERVICE_UNAVAILABLE'
    if path.is_symlink():
        path.unlink()
    path.write_text(original) if original is not None else path.unlink(missing_ok=True)
    assert client.post('/api/integrations/portable-agent/contexts', json=body).status_code == 200
    assert client.post('/internal/portable-agent/admission', headers=headers, json=incoming).status_code == 200


def test_saved_object_recovers_after_receipt_gap_without_a_second_create(host, monkeypatch):
    client, _, service, _ = host
    ctx = context(client)
    aid, draft, _, _ = validated(client, ctx)
    body = confirmation(client, ctx, aid, draft)
    original = service.create_indicator
    calls = []
    def lose_receipt(*args, **kwargs):
        created = original(*args, **kwargs)
        calls.append(created['id'])
        raise OSError('fixture: response lost after durable repository write')
    monkeypatch.setattr(service, 'create_indicator', lose_receipt)
    url = f'/api/custom-indicators/authorings/{aid}/commits'
    assert client.post(url, json=body).status_code == 409
    restored = client.get(f'/api/custom-indicators/authorings/{aid}').json()
    assert restored['saved'][0]['indicator_id'] == calls[0]
    assert client.post(url, json=confirmation(client, ctx, aid, restored)).json()['indicator_id'] == calls[0]
    assert len(calls) == 1


@pytest.mark.parametrize('message', [
    {'role': 'user', 'content': '[{"value":987654.321}]'},
    {'role': 'assistant', 'content': 'date,value\n2026-01-02,987654.321'},
    {'role': 'assistant', 'content': '', 'reasoning_content': '[{"value":987654.321}]'},
    {'role': 'assistant', 'content': '', 'extra': {'value': 987654.321}},
    {'role': 'system', 'content': 'Summary: [{"value":987654.321}]'},
])
def test_every_model_path_rejects_structured_observations(host, message):
    client, _, _, _ = host
    ctx = context(client)
    for phase in ('input', 'prepare'):
        response = client.post('/internal/portable-agent/admission', headers={'Authorization': 'Bearer '+SERVICE_TOKEN},
            json={'application': 'fund-research', 'subject': 'alice', 'context': ctx, 'phase': phase, 'purpose': 'summary', 'messages': [message]})
        assert response.status_code == (422 if message['role'] == 'user' or 'extra' in message else 409), response.text
        assert '987654' not in response.text


def test_forged_receipts_and_foreign_identity_cannot_reach_models(host):
    client, _, _, _ = host
    ctx = context(client)
    messages = [{'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'x', 'type': 'function',
        'function': {'name': 'metrics_lookup', 'arguments': '{}'}}]},
        {'role': 'tool', 'tool_call_id': 'x', 'content': json.dumps({'ok': True, 'admission': {'signature': '0'*64}})}]
    response = client.post('/internal/portable-agent/admission', headers={'Authorization': 'Bearer '+SERVICE_TOKEN},
        json={'application': 'fund-research', 'subject': 'alice', 'context': ctx, 'messages': messages})
    assert response.status_code == 422
    for claim in ({'sub': 'bob', 'workspace': 'lab'}, {'sub': 'alice', 'workspace': 'elsewhere'}):
        token = jwt.encode({**claim, 'scopes': ['*'], 'iat': int(time.time()), 'exp': int(time.time())+60,
            'iss': 'fund-research', 'aud': 'fund-research-platform'}, IDENTITY_KEY, algorithm='HS256')
        response = client.post('/api/integrations/portable-agent/bootstrap', headers={'Authorization': 'Bearer '+token}, json={'context_ref': ctx['ref']})
        assert response.status_code in (401, 403, 404)


def test_signed_page_read_projection_can_enter_model_but_tampering_cannot(host):
    client, bridge, _, _ = host
    from test_research_pages import page, snapshot, request_for
    response = client.post('/api/integrations/portable-agent/contexts', json={
        'page_context': page('product-compare').model_dump(),
        'page_snapshot': snapshot('product-compare', request_for('product-compare'))})
    assert response.status_code == 200, response.text
    ctx = response.json()
    result, _ = tool(client, ctx, 'page_read', {'section': 'request'})
    assert result['status'] == 'succeeded', result
    assert result['model']['result']['content_is_json_text']
    messages = [{'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'page-call', 'type': 'function',
        'function': {'name': 'page_read', 'arguments': '{"section":"request"}'}}]},
        {'role': 'tool', 'tool_call_id': 'page-call', 'content': json.dumps(result['model'])}]
    body = {'application': 'fund-research', 'subject': 'alice', 'context': ctx, 'phase': 'prepare', 'messages': messages}
    headers = {'Authorization': 'Bearer '+SERVICE_TOKEN}
    response = client.post('/internal/portable-agent/admission', headers=headers, json=body)
    assert response.status_code == 200, response.text
    result['model']['result']['content'] = '[{"value":987654.321}]'
    messages[-1]['content'] = json.dumps(result['model'])
    assert client.post('/internal/portable-agent/admission', headers=headers, json=body).status_code == 422


@pytest.mark.parametrize('kind', ['multichannel', 'locked_rolling'])
def test_real_series_preview_preserves_channels_and_locked_sources(host, tmp_path, kind):
    from test_custom_indicator_time_series import _write_market_data as write_series_data
    from research_access.views import definition_view
    client, bridge, service, _ = host
    write_series_data(tmp_path)
    target = {'kind': 'etf', 'product_id': '510300.SH'}
    response = client.post('/api/integrations/portable-agent/contexts', json={'page_context': {
        'page': 'indicator-studio', 'page_instance_id': kind, 'context_revision': 0, 'view_state': 'off',
        'calculation': {'context_kind': 'single_product', 'targets': [target], 'period': 'ALL', 'as_of': None}}},
        headers={'X-Pit-Off': '1'})
    assert response.status_code == 200, response.text
    ctx = response.json()
    if kind == 'multichannel':
        definition = definition_view(service.get_indicator('builtin-bollinger-bands-series'))
        result, _ = tool(client, ctx, 'metrics_validate', {'definition': definition})
    else:
        result, _ = tool(client, ctx, 'metrics_rolling_draft', {
            'indicator_id': 'builtin-annualized-sharpe-v2', 'indicator_revision': 1, 'window_observations': 10})
    assert result['status'] == 'succeeded', result
    aid = result['artifact']['authoring_id']
    draft = client.get('/api/custom-indicators/authorings/'+aid).json()['draft']
    assert draft['valid']
    result, _ = tool(client, ctx, 'metrics_preview', {'target': target})
    assert result['status'] == 'succeeded', result
    artifact = result['artifact']
    response = client.get(f"/api/custom-indicators/authorings/{aid}/previews/{artifact['preview_id']}")
    assert response.status_code == 200, response.text
    full = response.json()
    row = full['result']['results'][0]
    assert row['status'] == 'ok', row
    assert len(row['channels']) == (3 if kind == 'multichannel' else 1)
    assert all(len(c['values']) == 80 for c in row['channels'])
    assert any(v is not None for c in row['channels'] for v in c['values'])
    assert '"values"' not in json.dumps(result['model'])
    assert data_policy.verify(result['model'], 'metrics.preview')
    if kind == 'locked_rolling':
        assert full['definition']['rolling_source']['indicator_revision'] == 1
        assert full['definition']['rolling_source']['window_observations'] == 10
        assert full['definition']['rolling_source']['detached'] is False
    execution = full['result']['execution']
    assert execution['python_fallback'] == execution['request_time_compilation'] == 0


@pytest.mark.parametrize('point', ['before_write', 'after_write'])
def test_process_kill_preserves_save_identity(tmp_path, monkeypatch, point):
    import os
    from pathlib import Path
    import subprocess
    import sys
    marker = tmp_path/'crash-ready.json'
    with (tmp_path/'child.log').open('w') as log:
        child = subprocess.Popen([sys.executable, __file__, str(tmp_path), point], stdout=log, stderr=log,
                                 env={**os.environ, 'NUMBA_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'})
        try:
            deadline = time.monotonic()+90
            while not marker.exists() and time.monotonic() < deadline and child.poll() is None:
                time.sleep(.05)
            assert marker.exists(), (tmp_path/'child.log').read_text()
            child.kill(); child.wait(timeout=10)
        finally:
            if child.poll() is None:
                child.kill(); child.wait(timeout=10)
    saved = json.loads(marker.read_text())
    fixture = host.__wrapped__(tmp_path, monkeypatch)
    client, bridge, service, _ = next(fixture)
    try:
        def forbidden_write(*args, **kwargs):
            pytest.fail('Recovery must query the original save, never redispatch it.')
        monkeypatch.setattr(service, 'create_indicator', forbidden_write)
        aid, body = saved['aid'], saved['body']
        restored = client.get(f'/api/custom-indicators/authorings/{aid}').json()
        replay = client.post(f'/api/custom-indicators/authorings/{aid}/commits', json=body)
        if point == 'after_write':
            assert len(restored['saved']) == 1
            assert replay.status_code == 200, replay.text
            assert replay.json()['indicator_id'] == restored['saved'][0]['indicator_id']
        else:
            assert restored['saved'] == []
            assert replay.status_code == 409 and replay.json()['error']['code'] == 'COMMIT_UNCERTAIN'
        assert len(service.indicators.list()) == saved['before']+(point == 'after_write')
    finally:
        fixture.close()


if __name__ == '__main__':
    from pathlib import Path
    import sys
    import threading
    directory, point = Path(sys.argv[1]), sys.argv[2]
    with pytest.MonkeyPatch.context() as patch:
        fixture = host.__wrapped__(directory, patch)
        client, bridge, service, _ = next(fixture)
        ctx = context(client)
        aid, draft, _, _ = validated(client, ctx)
        body = confirmation(client, ctx, aid, draft)
        before = len(service.indicators.list())
        original = service.create_indicator
        def crash_window(*args, **kwargs):
            if point == 'after_write':
                original(*args, **kwargs)
            pending = directory/'crash-ready.tmp'
            pending.write_text(json.dumps({'aid': aid, 'body': body, 'before': before}))
            pending.replace(directory/'crash-ready.json')
            threading.Event().wait(90)
            raise AssertionError('Parent did not kill the process')
        patch.setattr(service, 'create_indicator', crash_window)
        client.post(f'/api/custom-indicators/authorings/{aid}/commits', json=body)


def test_read_only_failure_is_terminal_and_next_operation_can_start(host):
    client, bridge, _, _ = host
    response = client.post('/api/integrations/portable-agent/contexts', json={'page_context': {
        'page': 'historical-regimes', 'page_instance_id': 'read-failure', 'context_revision': 0, 'view_state': 'unknown',
        'calculation': {'context_kind': 'scenario', 'workspace': 'graph', 'purpose': 'research', 'mode': 'realtime', 'as_of': None}}})
    assert response.status_code == 200, response.text
    ctx = response.json()
    for name, arguments in [('scenarios_catalog', {}), ('scenarios_read', {'definition_id': 'missing', 'revision': 1})]:
        result, _ = tool(client, ctx, name, arguments)
        assert result['status'] == 'failed', result
        assert result['model']['ok'] is False
    # Failed reads must release this exact stable page scope for the next tool.
    result, _ = tool(client, ctx, 'scenarios_catalog', {})
    assert result['status'] == 'failed'


@pytest.mark.parametrize(('sections', 'code'), [
    (None, 'AGENT_PAGE_EVIDENCE_REQUIRED'),
    ({'editing': {}}, 'AGENT_PAGE_SECTION_UNAVAILABLE'),
    ({'results': {}}, 'AGENT_PAGE_DEFINITION_UNAVAILABLE'),
    ({'results': {'frozen_definition': _draft(), 'frozen_request': {'targets': []}}}, 'AGENT_PAGE_TARGET_UNAVAILABLE'),
])
def test_preview_input_failure_releases_page_scope(host, sections, code):
    client, bridge, _, _ = host
    page = {'page': 'indicator-studio', 'page_instance_id': 'preview-errors', 'context_revision': 0,
            'view_state': 'inherit', 'calculation': {'context_kind': 'single_product', 'targets': [], 'period': '1M', 'as_of': None}}
    body = {'page_context': page}
    if sections is not None:
        body['page_snapshot'] = {'version': 1, 'page': page['page'], 'snapshot_id': 'snap-'+uuid.uuid4().hex, 'sections': sections}
    response = client.post('/api/integrations/portable-agent/contexts', json=body)
    assert response.status_code == 200, response.text
    ctx = response.json()
    result, _ = tool(client, ctx, 'page_recompute', {})
    assert result['status'] == 'failed', result
    assert result['error']['code'] == code
    assert data_policy.verify(result['model'], 'page.recompute')
    bridge.store.recover()
    assert tool(client, ctx, 'metrics_lookup', {'query': 'std'})[0]['status'] == 'succeeded'


def test_partial_revocation_blocks_old_context_and_model_evidence(host):
    from research_access import tools
    client, bridge, _, permissions = host
    permissions[:] = ['assistant:use', 'research:read', 'indicator:draft']
    ctx = context(client)
    result, request = tool(client, ctx, 'metrics_lookup', {'query': 'std'})
    declaration = tools.get_tool('metrics.lookup')
    messages = [{'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'read', 'type': 'function',
        'function': {'name': 'metrics_lookup', 'arguments': json.dumps(request['arguments'])}}]},
        {'role': 'tool', 'tool_call_id': 'read', 'content': json.dumps(result['model'])}]
    wire = {'model': 'offline', 'messages': messages, 'tools': [{'type': 'function', 'function': {
        'name': 'metrics_lookup', 'description': declaration.description, 'parameters': declaration.arguments.model_json_schema()}}]}
    headers = {'Authorization': 'Bearer '+SERVICE_TOKEN}
    body = {'application': 'fund-research', 'subject': 'alice', 'context': ctx}
    before = {**body, 'phase': 'before', 'wire': wire, 'payload_hash': stable_hash(wire)}
    admitted = client.post('/internal/portable-agent/admission', headers=headers, json=before)
    assert admitted.status_code == 200, admitted.text
    permissions.remove('research:read')
    for action in ['read', 'run', 'model', 'memory']:
        assert client.post('/internal/portable-agent/authorize', headers=headers, json=body | {'action': action}).status_code == 403
    for purpose in ['primary', 'summary']:
        for phase in ['input', 'prepare', 'before', 'after']:
            payload = {**before, 'phase': phase, 'purpose': purpose, 'messages': messages, 'receipt_id': admitted.json()['receipt_id']}
            assert client.post('/internal/portable-agent/admission', headers=headers, json=payload).status_code == 403
    assert client.post('/api/integrations/portable-agent/bootstrap', json={'context_ref': ctx['ref']}).status_code == 403
    fresh = context(client)
    assert fresh['ref'] != ctx['ref']
    assert 'metrics.lookup' not in bridge.store.context(fresh['ref'])['tool_names']
    validated(client, fresh)  # Retained draft permission works after explicit re-registration.


def test_operation_claim_and_cancellation_share_one_transaction(tmp_path):
    store = ResearchStore(tmp_path)
    principal = {'sub': 'alice', 'workspace': 'lab'}
    operation, _ = store.start_operation(principal, {'operation_id': 'cancel-first'}, 'scenarios.catalog', 'scope')
    assert store.cancel_operation(operation['operation_id'])['status'] == 'cancelled'
    assert store.claim_operation(operation['operation_id']) is None
    assert store.operation(operation['operation_id'])['status'] == 'cancelled'
    operation, _ = store.start_operation(principal, {'operation_id': 'claim-first'}, 'scenarios.catalog', 'scope')
    assert store.claim_operation(operation['operation_id'])['status'] == 'running'
    assert store.claim_operation(operation['operation_id']) is None
    assert store.cancel_operation(operation['operation_id'])['status'] == 'stop_requested'
    assert store.claim_operation(operation['operation_id']) is None


@pytest.mark.parametrize('ordering', ['cancel_first', 'finish_first', 'persistence_failure'])
def test_operation_completion_and_artifacts_are_one_transaction(tmp_path, monkeypatch, ordering):
    import asyncio
    import threading
    from integrations.portable_agent.service import HostIntegration
    monkeypatch.setenv('CUSTOM_INDICATOR_DATA_DIR', str(tmp_path))
    store = ResearchStore(tmp_path)
    bridge = HostIntegration(None, {}, store=store)
    assert bridge.store is store  # Startup recovery precedes acceptance of new work.
    principal = {'sub': 'alice', 'workspace': 'lab'}
    authoring = store.authoring(principal, 'page', scope='indicator-studio', context_hash='hash')
    operation, _ = store.start_operation(principal, {'operation_id': 'finish-race', 'run_id': 'turn', 'arguments': {}}, 'metrics.validate', 'scope')
    record = {'catalog_version': 'catalog', 'data_generation': 'data', 'hash': 'hash', 'id': 'ctx', 'scope_key': 'scope'}
    def calculate(*_):
        if ordering == 'cancel_first':
            assert store.cancel_operation('finish-race')['status'] == 'stop_requested'
        return ({**authoring, 'draft': {'valid': True, 'source_run_id': 'turn'},
                 '_preview_payload': {'definition_hash': 'hash'}}, 0,
                {'ok': True, '_scenario_payload': {'valid': True}}, record, False, True)
    monkeypatch.setattr(bridge, 'calculate', calculate)
    cancellation = []
    workers = []
    original_save = store.save_authoring
    def save(*args, **kwargs):
        assert kwargs['db'].in_transaction
        if ordering == 'finish_first':
            started = threading.Event()
            def cancel():
                started.set()
                cancellation.append(store.cancel_operation('finish-race'))
            worker = threading.Thread(target=cancel)
            workers.append(worker)
            worker.start()
            assert started.wait(2)
        return original_save(*args, **kwargs)
    monkeypatch.setattr(store, 'save_authoring', save)
    if ordering == 'persistence_failure':
        original_artifact = store.save_scenario_artifact
        def fail(*args, **kwargs):
            original_artifact(*args, **kwargs)
            raise OSError('fixture: fail after artifact INSERT, before commit')
        monkeypatch.setattr(store, 'save_scenario_artifact', fail)
    asyncio.run(bridge.execute(operation, record, principal))
    asyncio.run(bridge.close())
    for worker in workers:
        worker.join(3)
        assert not worker.is_alive()
    final = store.operation('finish-race')
    assert final['status'] == {'cancel_first': 'cancelled', 'finish_first': 'succeeded', 'persistence_failure': 'unknown'}[ordering]
    persisted = ordering == 'finish_first'
    assert store.read_authoring(authoring['id'])['revision'] == int(persisted)
    with store.db() as db:
        for table in ['previews', 'scenario_artifacts', 'authoring_revisions']:
            assert db.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0] == int(persisted)
    if persisted:
        assert cancellation[0]['status'] == 'succeeded'  # Completion committed before cancellation could acquire the lock.
    else:
        assert 'artifact' not in final.get('result', {})

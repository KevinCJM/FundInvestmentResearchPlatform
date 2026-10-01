"""An isolated business import preserves ownership and receipts without re-execution."""
import asyncio
import copy
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.mark.parametrize('interrupted', [0, 1, 2, 'permission', 'authority_response'])
def test_business_archive_dry_run_replay_and_corruption(tmp_path, monkeypatch, interrupted):
    root = Path(__file__).resolve().parents[2]
    import sys
    sys.path[:0] = [str(root), str(root/'backend'), str(root/'backend/tests')]
    spec = importlib.util.spec_from_file_location('research_import', root/'scripts/import_research_archive.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    from research_access.store import ResearchStore
    from research_fixtures import APPROVED_DEFINITION, authoring_context
    from test_custom_indicator_service import _write_market_data
    monkeypatch.setenv('CUSTOM_INDICATOR_DATA_DIR', str(tmp_path/'data'))
    monkeypatch.setenv('INDICATOR_PROCESS_WORKERS', '1')
    import httpx
    principal = {'sub': 'alice', 'workspace': 'lab', 'scopes': ['assistant:use', 'research:read', 'scenario:research']}
    monkeypatch.setenv('PORTABLE_AGENT_AUTH_MODE', 'jwt')
    monkeypatch.setenv('PORTABLE_AGENT_AUTHORITY_URL', 'https://permissions.example.test/current')
    monkeypatch.setenv('PORTABLE_AGENT_AUTHORITY_TOKEN', 'offline-authority-credential-0000000000')
    monkeypatch.setenv('PORTABLE_AGENT_ISSUER_KEY', '1'*64)
    monkeypatch.setenv('APP_ENV', 'development')
    async_client = httpx.AsyncClient
    def permissions(request):
        requested = json.loads(request.content)
        assert requested['workspace'] == 'lab' and requested['subject'] in {'alice', 'bob'}
        return httpx.Response(200, json={**principal, 'sub': requested['subject']})
    monkeypatch.setattr(httpx, 'AsyncClient', lambda **kwargs: async_client(
        transport=httpx.MockTransport(permissions), **kwargs))
    market = tmp_path/'market'; market.mkdir(); _write_market_data(market)
    page = authoring_context(); page['view_state'] = 'unknown'
    digest = module.definition_hash(APPROVED_DEFINITION)
    session = {'source_id': 'old', 'target_id': module.target_id('dataset', 'session', 'old'), 'page_context': page,
        'draft': {'definition': APPROVED_DEFINITION, 'definition_hash': digest, 'draft_revision': 1, 'valid': True},
        'messages': [{'id': 'message', 'run_id': 'run', 'speaker': 'user', 'text': '保留收益定义'},
                     {'id': 'reply', 'run_id': 'run', 'speaker': 'assistant', 'text': '已生成草稿。'}],
        'draft_history': [{'seq': 1, 'source_run_id': 'run', 'draft': {'definition': APPROVED_DEFINITION, 'definition_hash': digest}}]}
    scenario = copy.deepcopy(session)
    definition = {'name': '历史情景', 'graph': {'nodes': [], 'edges': [], 'outputs': {}}}
    scenario.update(source_id='scenario', target_id=module.target_id('dataset', 'session', 'scenario'),
        page_context={**page, 'page': 'historical-regimes', 'page_instance_id': 'archive-scenario',
                      'calculation': {'context_kind': 'scenario', 'workspace': 'graph'}})
    scenario['draft']['definition'] = definition
    scenario['draft_history'][0]['draft']['definition'] = definition
    for message in scenario['messages']:
        message['id'] = 'scenario-'+message['id']
    archive = {'schema_version': 'portable-migration/1', 'dataset_id': 'dataset', 'app': 'fund-research', 'owner': 'alice',
        'workspace': 'lab', 'sessions': [session, scenario], 'memories': [], 'business_commits': [{'source_session_id': 'old',
        'source_request_id': 'save', 'state': 'completed', 'definition_hash': digest,
        'response': {'indicator_id': 'existing-indicator', 'revision': 2, 'name': APPROVED_DEFINITION['name']}}]}
    archive['content_hash'] = module.stable_hash(archive)
    source, output, business = tmp_path/'export.json', tmp_path/'prepared.json', tmp_path/'data/research_access'
    source.write_text(json.dumps(archive))
    run = lambda apply: asyncio.run(module.prepare(source, business, market, output, apply=apply))
    assert run(False)['applied'] is False and not business.exists()
    from research_access.contracts import ResearchError
    scopes = principal['scopes']
    principal['scopes'] = []
    with pytest.raises(ResearchError) as denied:
        run(True)
    assert denied.value.status_code == 403 and not output.exists()
    with ResearchStore(business).db() as db:
        table = db.execute("SELECT name FROM sqlite_master WHERE name='migration_imports'").fetchone()
        assert not table or db.execute('SELECT COUNT(*) FROM migration_imports').fetchone()[0] == 0
    principal['scopes'] = scopes
    if interrupted in {'permission', 'authority_response'}:
        original_register = module.HostIntegration.register
        calls = 0
        async def deny_second(bridge, *args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                if interrupted == 'authority_response':
                    raise ValueError('fixture: authority returned invalid JSON')
                raise ResearchError('FORBIDDEN', '权限已撤销', status_code=403)
            return await original_register(bridge, *args, **kwargs)
        with monkeypatch.context() as patch:
            patch.setattr(module.HostIntegration, 'register', deny_second)
            with pytest.raises(ValueError if interrupted == 'authority_response' else ResearchError):
                run(True)
        assert not output.exists()
        with ResearchStore(business).db() as db:
            assert db.execute('SELECT result FROM migration_imports').fetchone()[0] == ''
    elif interrupted:
        original = module.HostIntegration.public_context
        completed = 0
        def crash_after_authoring(record):
            nonlocal completed
            completed += 1
            if completed == interrupted:
                raise RuntimeError('fixture: interrupted after final authoring revision')
            return original(record)
        with monkeypatch.context() as patch:
            patch.setattr(module.HostIntegration, 'public_context', staticmethod(crash_after_authoring))
            with pytest.raises(RuntimeError, match='after final authoring'):
                run(True)
        modified = {**archive, 'owner': 'bob'}
        modified['content_hash'] = module.stable_hash({k: v for k, v in modified.items() if k != 'content_hash'})
        source.write_text(json.dumps(modified))
        with pytest.raises(ValueError, match='different archive'):
            run(True)
        source.write_text(json.dumps(archive))
        versions = module.HostIntegration.versions
        monkeypatch.setattr(module.HostIntegration, 'versions', lambda self, page: {
            **versions(self, page), 'catalog_version': 'catalog-after-interruption', 'data_generation': 'data-after-interruption'})
    if interrupted == 2:
        for revoked in ('research:read', 'scenario:research'):
            principal['scopes'] = [scope for scope in scopes if scope != revoked]
            with pytest.raises(ResearchError) as denied:
                run(True)
            assert denied.value.status_code == 403 and not output.exists()
            with ResearchStore(business).db() as db:
                assert db.execute('SELECT result FROM migration_imports').fetchone()[0] == ''
            principal['scopes'] = scopes
    first = run(True)
    assert first['sessions'] == 2 and first['quarantined'] == []
    prepared = json.loads(output.read_text())
    assert prepared['sessions'][0]['turns'][0]['output'] == '已生成草稿。'
    authority = prepared['sessions'][0]['context']
    assert authority['summary']['catalog_version'] != 'catalog-after-interruption'
    store = ResearchStore(business)
    authoring = store.read_authoring(authority['authoring_id'], {'sub': 'alice', 'workspace': 'lab'})
    assert authoring['draft']['compile_token'] is None and authoring['draft']['valid'] is False
    # The target owner can read/bootstrap imported history without draft permissions.
    async def resume():
        bridge = module.HostIntegration(None, {}, store=store)
        try:
            for item in prepared['sessions']:
                await bridge.authorize('fund-research', 'alice', item['context'], action='read')
                boot = await bridge.bootstrap(principal, item['context']['ref'])
                import jwt
                scopes = jwt.decode(boot['token'], options={'verify_signature': False})['scopes']
                assert 'tool:metrics_validate' not in scopes
        finally:
            await bridge.close()
    asyncio.run(resume())
    before = output.read_bytes()
    output = tmp_path/'recovered/output.json'
    principal['scopes'] = ['assistant:use']
    with pytest.raises(ResearchError) as denied:
        run(True)
    assert denied.value.status_code == 403 and not output.exists()
    principal['scopes'] = scopes
    assert run(True)['replayed'] and output.read_bytes() == before
    with store.db() as db:
        receipts = [json.loads(row[0]) for row in db.execute('SELECT body FROM commits')]
        assert db.execute('SELECT COUNT(*) FROM scenario_artifacts').fetchone()[0] == 1
    assert len(receipts) == 1 and receipts[0]['indicator_id'] == 'existing-indicator'
    archive['owner'] = 'bob'; source.write_text(json.dumps(archive))
    with pytest.raises(ValueError, match='modified archive'):
        run(True)

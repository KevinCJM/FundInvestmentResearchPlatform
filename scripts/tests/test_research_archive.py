"""An isolated business import preserves ownership and receipts without re-execution."""
import asyncio
import copy
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.mark.parametrize('interrupted', [0, 1, 2])
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
    if interrupted:
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
    first = run(True)
    assert first['sessions'] == 2 and first['quarantined'] == []
    prepared = json.loads(output.read_text())
    assert prepared['sessions'][0]['turns'][0]['output'] == '已生成草稿。'
    authority = prepared['sessions'][0]['context']
    store = ResearchStore(business)
    authoring = store.read_authoring(authority['authoring_id'], {'sub': 'alice', 'workspace': 'lab'})
    assert authoring['draft']['compile_token'] is None and authoring['draft']['valid'] is False
    before = output.read_bytes()
    output = tmp_path/'recovered/output.json'
    assert run(True)['replayed'] and output.read_bytes() == before
    with store.db() as db:
        receipts = [json.loads(row[0]) for row in db.execute('SELECT body FROM commits')]
        assert db.execute('SELECT COUNT(*) FROM scenario_artifacts').fetchone()[0] == 1
    assert len(receipts) == 1 and receipts[0]['indicator_id'] == 'existing-indicator'
    archive['owner'] = 'bob'; source.write_text(json.dumps(archive))
    with pytest.raises(ValueError, match='modified archive'):
        run(True)

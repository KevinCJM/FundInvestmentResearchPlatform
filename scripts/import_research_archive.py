"""Import a versioned business archive and prepare approved references for the external agent.

This consumes a neutral export, not the old agent's executable checkpoints. Run only on an isolated
copy first; --apply is an explicit data migration action, separate from ordinary service startup.
"""
import argparse
import asyncio
import json
import time
import uuid
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'backend')]

from custom_indicators.service import CustomIndicatorService
from integrations.portable_agent.contracts import ContextInput
from integrations.portable_agent.service import HostIntegration
from research_access import data_policy
from research_access.authoring import definition_hash, stable_hash, stable_json
from research_access.store import ResearchStore


def target_id(dataset, kind, source):
    return uuid.uuid5(uuid.NAMESPACE_URL, f'portable-migration/{dataset}/{kind}/{source}').hex


async def prepare(source, business_dir, market_dir, output, *, apply=False):
    archive = json.loads(source.read_text())
    if archive.get('schema_version') != 'portable-migration/1' or archive.get('content_hash') != stable_hash({k: v for k, v in archive.items() if k != 'content_hash'}):
        raise ValueError('Unsupported or modified archive')
    if not archive.get('owner') or not archive.get('workspace'):
        raise ValueError('An explicit owner/workspace assignment is required')
    principal = {'sub': archive['owner'], 'workspace': archive['workspace'], 'scopes': ['*']}
    result = {'schema_version': 'portable-agent-import/1', 'dataset_id': archive['dataset_id'], 'app': archive['app'],
              'owner': archive['owner'], 'workspace': archive['workspace'], 'profiles_hash': archive.get('profiles_hash'),
              'host_reviewed': True, 'sessions': [], 'memories': [], 'quarantined': list(archive.get('quarantined', []))}
    if not apply:
        return {'sessions_to_review': len(archive['sessions']), 'memories_to_review': len(archive['memories']), 'applied': False}
    store = ResearchStore(business_dir)
    service = CustomIndicatorService(business_dir.parent, market_dir)
    integration = HostIntegration(service, {}, store=store, authority=lambda subject, workspace: principal)
    message_ids = {}
    try:
        with store.db() as db:
            db.execute('CREATE TABLE IF NOT EXISTS migration_imports(dataset TEXT PRIMARY KEY, hash TEXT NOT NULL, result TEXT NOT NULL)')
            old = db.execute('SELECT hash,result FROM migration_imports WHERE dataset=?', (archive['dataset_id'],)).fetchone()
            if old:
                if old['hash'] != archive['content_hash']:
                    raise ValueError('Dataset ID already refers to a different archive')
                if old['result']:
                    output.parent.mkdir(parents=True, exist_ok=True)
                    output.write_text(old['result']); output.chmod(0o600)
                    return {'applied': True, 'replayed': True}
            else:
                # Bind interrupted work to this exact archive before writing any imported objects.
                db.execute('INSERT INTO migration_imports VALUES(?,?,?)', (archive['dataset_id'], archive['content_hash'], ''))
        for session in archive['sessions']:
            sid = session['source_id']
            try:
                page = session['page_context']
                if page.get('page') == 'regime-workbench':
                    page = {**page, 'page': 'historical-regimes', 'calculation': {
                        **page['calculation'], 'context_kind': 'scenario', 'workspace': 'graph'}}
                body = ContextInput(page_context=page)
                record = await integration.register(principal, body, pit_off=body.page_context.view_state == 'off')
            except Exception as exc:
                result['quarantined'].append({'kind': 'session', 'id': sid, 'reason': type(exc).__name__})
                continue
            aid = target_id(archive['dataset_id'], 'authoring', sid)
            draft = session.get('draft')
            if draft:
                draft = {**draft, 'definition_hash': definition_hash(draft['definition']), 'compile_token': None,
                         'stale': True, 'valid': False}  # Historical validation is not current execution authority.
            value = {'id': aid, 'revision': 1, 'subject': principal['sub'], 'workspace': principal['workspace'],
                     'scope': record['scope'], 'context_hash': record['hash'], 'draft': draft}
            final_revision = 2 + sum(isinstance(item.get('draft'), dict) and bool(item['draft'].get('definition'))
                                     for item in session.get('draft_history', []))
            with store.db() as db:
                existing = db.execute('SELECT body FROM authorings WHERE id=?', (aid,)).fetchone()
                if existing and json.loads(existing[0]) not in (value, {**value, 'revision': final_revision}):
                    raise ValueError('Imported authoring collides with a different object')
                db.execute('INSERT OR IGNORE INTO authorings VALUES(?,?,?,?,?,?)', (aid, principal['sub'], principal['workspace'],
                    'import:'+archive['dataset_id']+':'+sid, 1, stable_json(value)))
                db.execute('INSERT OR IGNORE INTO authoring_revisions VALUES(?,?,?)', (aid, 1, stable_json(value)))
            # Rebind the business reference; the framework only retains this opaque authority record.
            body = body.model_copy(update={'authoring_id': aid}) if body.page_context.context_kind == 'single_product' else body
            record = await integration.register(principal, body, pit_off=body.page_context.view_state == 'off')
            turns, by_run = [], {}
            for message in session['messages']:
                text = message.get('text') or ''
                if data_policy.user_text_violation(text):
                    result['quarantined'].append({'kind':'message','id':message.get('id'),'reason':'data_admission'})
                    text = '[历史内容已隔离，原文保留在业务审计备份中。]'
                if message.get('speaker') == 'user':
                    rid = target_id(archive['dataset_id'], 'turn', message.get('run_id') or message['id'])
                    old_status = session.get('run_statuses', {}).get(message.get('run_id'))
                    turn = {'id':rid, 'message':text, 'output':'', 'artifacts':[], 'status':'cancelled' if old_status == 'cancelled' else 'completed'}
                    turns.append(turn); by_run[message.get('run_id')] = turn; message_ids[message['id']] = rid
                elif message.get('speaker') == 'assistant':
                    turn = by_run.get(message.get('run_id'))
                    if turn:
                        turn['output'] += ('\n\n' if turn['output'] else '')+text
            definition_hashes = {}
            if session.get('draft'):
                definition_hashes[session['draft']['definition_hash']] = definition_hash(session['draft']['definition'])
            version = 1
            for historical in session.get('draft_history', []):
                past = historical.get('draft')
                turn = by_run.get(historical.get('source_run_id'))
                if not isinstance(past, dict) or not past.get('definition'):
                    continue
                digest = definition_hash(past['definition'])
                definition_hashes[past['definition_hash']] = digest
                version += 1
                historical_value = {**value, 'revision': version, 'draft': {**past, 'definition_hash': digest, 'compile_token': None, 'stale': True}}
                with store.db() as db:
                    db.execute('INSERT OR IGNORE INTO authoring_revisions VALUES(?,?,?)', (aid, version, stable_json(historical_value)))
                if turn:
                    oid = target_id(archive['dataset_id'], 'artifact', str(historical['seq'])+sid)
                    if body.page_context.context_kind == 'scenario':
                        artifact = {'definition': past['definition'], 'valid': False,
                            'validation_scope': 'historical', 'workspace': body.page_context.calculation.workspace,
                            'context_ref': record['id'], 'source_run_id': turn['id']}
                        with store.db() as db:
                            existing = db.execute('SELECT subject,workspace,body FROM scenario_artifacts WHERE id=?', (oid,)).fetchone()
                            if existing and (existing['subject'], existing['workspace'], json.loads(existing['body'])) != (principal['sub'], principal['workspace'], artifact):
                                raise ValueError('Imported scenario artifact collides with a different object')
                            db.execute('INSERT OR IGNORE INTO scenario_artifacts VALUES(?,?,?,?)', (oid, principal['sub'], principal['workspace'], stable_json(artifact)))
                        reference = {'type': 'research.scenario', 'group': 'scenario:'+record['scope_key'],
                                     'title': '历史情景草稿', 'artifact_id': oid, 'historical': True}
                    else:
                        reference = {'type':'research.indicator','group':'indicator:'+aid,
                            'title':'历史指标草稿','authoring_id':aid,'revision':version,'source_run_id':turn['id'],'historical':True}
                    turn['artifacts'].append({'id': oid, 'reference': reference})
            for historical in session.get('preview_history', []):
                reference = historical.get('preview') or {}
                pid = reference.get('preview_id', '')
                if not pid or any(c not in '0123456789abcdef' for c in pid):
                    continue
                manifest_path, parquet_path = source.parent/'previews'/f'{pid}.json', source.parent/'previews'/f'{pid}.parquet'
                if not manifest_path.exists() or not parquet_path.exists():
                    result['quarantined'].append({'kind':'preview','id':pid,'reason':'missing_or_expired'})
                    continue
                manifest = json.loads(manifest_path.read_text())
                if manifest.get('expires_at_epoch',0) <= time.time():
                    result['quarantined'].append({'kind':'preview','id':pid,'reason':'expired'})
                    continue
                import pyarrow.parquet as parquet
                rows = parquet.read_table(parquet_path, columns=['row_json']).column('row_json').to_pylist()
                if len(rows) != 1:
                    raise ValueError('Preview archive must contain one owned artifact')
                artifact = json.loads(rows[0])
                if artifact.get('session_id') != sid or artifact.get('definition_hash') != reference.get('definition_hash') or artifact.get('context_hash') != reference.get('context_hash'):
                    raise ValueError('Preview provenance does not match its owning reference')
                preview_id = target_id(archive['dataset_id'],'preview',pid)
                artifact = {**artifact, 'preview_id':preview_id, 'authoring_id':aid, 'imported':True, 'expires_at':manifest.get('expires_at')}
                artifact.pop('session_id',None)
                turn = by_run.get(historical.get('source_run_id'))
                if turn:
                    artifact['run_id'] = turn['id']
                    for candidate in reversed(turn['artifacts']):
                        if candidate['reference'].get('authoring_id') == aid:
                            candidate['reference']['preview_id'] = preview_id
                            break
                with store.db() as db:
                    db.execute('INSERT OR IGNORE INTO previews VALUES(?,?,?)', (preview_id,aid,stable_json(artifact)))
            # Continue from the final business draft, not an intermediate historical version.
            version += 1
            value['revision'] = version
            with store.db() as db:
                db.execute('UPDATE authorings SET revision=?,body=? WHERE id=?',(version,stable_json(value),aid))
                db.execute('INSERT OR IGNORE INTO authoring_revisions VALUES(?,?,?)',(aid,version,stable_json(value)))
            plans = []
            for index, plan in enumerate(session.get('task_plans', [])):
                source_run = message_ids.get(plan.get('source_message_id'))
                if not source_run:
                    continue
                constraints = []
                for constraint in plan.get('constraints', []):
                    source_run_id = message_ids.get(constraint.get('source_message_id'))
                    original = next((t for t in turns if t['id'] == source_run_id), None)
                    if original and constraint.get('quote') in original['message'] and not data_policy.user_text_violation(constraint.get('quote')):
                        constraints.append({**constraint,'source_message_id':source_run_id,
                            'supersedes_source_message_id':message_ids.get(constraint.get('supersedes_source_message_id'))})
                plans.append({'operation_id':target_id(archive['dataset_id'],'plan',sid+str(index)), 'source_run_id':source_run,
                    'steps':plan.get('capabilities', []), 'questions':plan.get('questions', []),'constraints':constraints,'status':'proposed'})
            result['sessions'].append({'id': session['target_id'], 'source_id':sid, 'context':integration.public_context(record), 'turns': turns, 'plans':plans})
            for item in archive['business_commits']:
                if item['source_session_id'] != sid or item.get('state') != 'completed':
                    continue
                old = item.get('response') or {}
                request_id = target_id(archive['dataset_id'], 'commit', item['source_request_id'])
                digest = old.get('definition_hash') or item.get('definition_hash')
                digest = definition_hashes.get(digest,digest)
                receipt = {'request_id':request_id,'request_hash':stable_hash(item),'authoring_id':aid,'definition_hash':digest,
                    'status':'succeeded','indicator_id':old.get('indicator_id'),'indicator_revision':old.get('revision'),
                    'name':old.get('name'),'imported':True}
                with store.db() as db:
                    db.execute('INSERT OR IGNORE INTO commits VALUES(?,?,?,?,?,?)',(principal['sub'],principal['workspace'],request_id,aid,digest,stable_json(receipt)))
                    db.execute('INSERT OR IGNORE INTO definition_commits VALUES(?,?,?)',(aid,digest,request_id))
        for item in archive['memories']:
            if data_policy.user_text_violation(item.get('text')) or item.get('status') not in {'accepted','revoked','replaced'}:
                result['quarantined'].append({'kind':'memory','id':item.get('memory_id'),'reason':'data_admission_or_status'})
                continue
            identity_key = item.get('key') or 'indicator_definition:'+str(item.get('definition_hash') or '')
            object_id = item.get('object_id') or 'scope'
            if not item.get('source_message_id') and identity_key == 'indicator_definition:'+object_id:
                object_id = 'scope'
            result['memories'].append({'id':target_id(archive['dataset_id'],'memory',item['memory_id']), 'revision':item['version'],
                'key':identity_key,'object_id':object_id,'value':item['text'],'quote':item.get('source_quote',''), 'scope':item['scope'],
                'status':item['status'], 'source_session':target_id(archive['dataset_id'],'session',item['source_session_id']),
                'source_run':message_ids.get(item.get('source_message_id'),''), 'imported_source_verified':True,
                'source_reference':{'kind':'legacy_verified_memory','dataset':archive['dataset_id'],'id':item['memory_id']}})
        result['content_hash'] = stable_hash(result)
        serialized = json.dumps(result, ensure_ascii=False, indent=2)
        with store.db() as db:
            db.execute('UPDATE migration_imports SET result=? WHERE dataset=? AND hash=?', (serialized, archive['dataset_id'], archive['content_hash']))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized); output.chmod(0o600)
        return {'applied':True,'sessions':len(result['sessions']),'memories':len(result['memories']),'quarantined':result['quarantined']}
    finally:
        await integration.close(); service.close_compute_engine()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export', type=Path, required=True)
    parser.add_argument('--business-data', type=Path, required=True)
    parser.add_argument('--market-data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    print(json.dumps(asyncio.run(prepare(args.export,args.business_data,args.market_data,args.output,apply=args.apply)),ensure_ascii=False))

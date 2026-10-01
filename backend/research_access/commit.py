"""Human-confirmed indicator publication, independent of the agent's session database."""
import json
import time
import uuid

from .authoring import stable_hash, stable_json
from .catalog import build_catalog
from .contracts import ResearchError
from custom_indicators.errors import IndicatorDomainError


def reconcile(store, service, principal, aid):
    """Read the repository's atomic creation identity; never repeat a business write."""
    store.read_authoring(aid, principal)
    with store.db() as db:
        pending = [json.loads(row[0]) for row in db.execute("SELECT body FROM commits WHERE authoring_id=? AND json_extract(body,'$.status') IN ('started','unknown')", (aid,))]
    for receipt in pending:
        found = service.indicators.find_creation(receipt['operation_id']) if receipt.get('operation_id') else None
        if not found:
            continue
        with store.db() as db:
            state = store.read_authoring(aid, principal, db=db)
            receipt.update(status='succeeded', indicator_id=found['id'], indicator_revision=found['revision'], name=found.get('name'))
            if state.get('committing') == receipt['request_id']:
                state.pop('committing')
            db.execute('UPDATE authorings SET body=? WHERE id=?', (stable_json(state), aid))
            db.execute('UPDATE commits SET body=? WHERE subject=? AND workspace=? AND request_id=?',
                       (stable_json(receipt), principal['sub'], principal['workspace'], receipt['request_id']))


def preview(store, service, principal, aid, context, expected_revision, definition_hash, target=None):
    current = store.read_authoring(aid, principal)
    draft = current.get('draft') or {}
    if current['revision'] != expected_revision or not draft.get('valid') or draft.get('definition_hash') != definition_hash:
        raise ResearchError('REVISION_CONFLICT', '草稿已改变，请重新读取后预览。', status_code=409)
    if current.get('committing'):
        raise ResearchError('COMMIT_UNCERTAIN', '此前保存尚待核对。', status_code=409)
    if context['page_context']['calculation']['context_kind'] != 'single_product':
        raise ResearchError('DOMAIN_MISMATCH', '只能在单产品研究域保存指标。', status_code=409)
    if target is not None and target not in context['page_context']['calculation'].get('targets', []):
        raise ResearchError('CONTEXT_CHANGED', '保存目标不属于当前冻结上下文。', status_code=409)
    definition = draft['definition']
    repository_revision = service.indicators.catalog_revision()
    validation = service.validate(definition)
    if not validation.get('valid'):
        raise ResearchError('VALIDATION_ERROR', '指标定义未通过校验。', status_code=422, diagnostics=validation.get('diagnostics'))
    catalog = build_catalog(service)
    if repository_revision != service.indicators.catalog_revision():
        raise ResearchError('REVISION_CONFLICT', '预览期间目录已改变，请重试。', status_code=409)
    conflicts = [x for x in catalog['items'] if x.get('name') == definition.get('name')]
    confirmation = {'id': uuid.uuid4().hex, 'authoring_id': aid, 'status': 'pending', 'definition': definition,
        'definition_hash': definition_hash, 'draft_revision': draft['draft_revision'], 'authoring_revision': expected_revision,
        'context_ref': context['id'], 'context_hash': context['hash'], 'target': target, 'catalog_version': catalog['version'],
        'repository_revision': repository_revision,
        'created_at': time.time(), 'expires_at': time.time()+900,
        'impact': {'action': 'create', 'name': definition.get('name'), 'same_name_exists': bool(conflicts),
                   'message': '将新建一个独立指标；不会覆盖同名指标。', 'definition': definition,
                   'target': target, 'calculation': context['page_context']['calculation']}}
    with store.db() as db:
        live = store.read_authoring(aid, principal, db=db)
        if live['revision'] != expected_revision or live.get('committing'):
            raise ResearchError('REVISION_CONFLICT', '预览期间草稿已改变。', status_code=409)
        for row in db.execute("SELECT id,body FROM confirmations WHERE authoring_id=? AND json_extract(body,'$.status')='pending'", (aid,)).fetchall():
            previous = json.loads(row['body']); previous['status'] = 'invalidated'
            db.execute('UPDATE confirmations SET body=? WHERE id=?', (stable_json(previous), row['id']))
        db.execute('INSERT INTO confirmations VALUES(?,?,?)', (confirmation['id'], aid, stable_json(confirmation)))
    return confirmation


def publish(store, service, principal, aid, body):
    request_id = body['request_id']
    digest = stable_hash(body)
    with store.db() as db:
        state = store.read_authoring(aid, principal, db=db)
        old = db.execute('SELECT body FROM commits WHERE subject=? AND workspace=? AND request_id=?',
                         (principal['sub'], principal['workspace'], request_id)).fetchone()
        if old:
            receipt = json.loads(old[0])
            if receipt['request_hash'] != digest or receipt['authoring_id'] != aid:
                raise ResearchError('REQUEST_CONFLICT', '同一保存标识不能改变内容。', status_code=409)
            if receipt['status'] != 'succeeded':
                if receipt['status'] == 'rejected':
                    raise ResearchError('CONFIRMATION_STALE', '原保存被版本校验拒绝，请重新预览确认。', status_code=409)
                found = service.indicators.find_creation(receipt['operation_id']) if receipt.get('operation_id') else None
                if not found:
                    raise ResearchError('COMMIT_UNCERTAIN', '保存结果尚待核对，不能重新创建。', status_code=409)
                receipt.update(status='succeeded', indicator_id=found['id'], indicator_revision=found['revision'], name=found.get('name'))
                state.pop('committing', None)
                db.execute('UPDATE authorings SET body=? WHERE id=?', (stable_json(state), aid))
                db.execute('UPDATE commits SET body=? WHERE subject=? AND workspace=? AND request_id=?',
                           (stable_json(receipt), principal['sub'], principal['workspace'], request_id))
            return receipt
        row = db.execute('SELECT body FROM confirmations WHERE id=? AND authoring_id=?', (body['confirmation_id'], aid)).fetchone()
        confirmation = json.loads(row[0]) if row else {}
        draft = state.get('draft') or {}
        if (not body['confirmed'] or confirmation.get('status') != 'pending' or confirmation.get('expires_at', 0) <= time.time()
                or state['revision'] != body['expected_revision'] or state.get('committing')
                or confirmation.get('authoring_revision') != state['revision']
                or confirmation.get('definition_hash') != body['definition_hash']
                or not draft.get('valid') or draft.get('definition_hash') != body['definition_hash']
                or confirmation.get('draft_revision') != draft.get('draft_revision')):
            raise ResearchError('CONFIRMATION_STALE', '保存确认已失效，请重新展示影响并确认。', status_code=409)
        prior = db.execute('SELECT c.body FROM definition_commits d JOIN commits c ON c.authoring_id=d.authoring_id AND c.request_id=d.request_id WHERE d.authoring_id=? AND d.definition_hash=?',
                           (aid, body['definition_hash'])).fetchone()
        if prior:
            receipt = json.loads(prior[0])
            if receipt['status'] != 'succeeded':
                raise ResearchError('COMMIT_UNCERTAIN', '该定义已有未决保存，不能重复创建。', status_code=409)
            confirmation['status'] = 'accepted'
            db.execute('UPDATE confirmations SET body=? WHERE id=?', (stable_json(confirmation), confirmation['id']))
            replay = receipt | {'request_id': request_id, 'request_hash': digest, 'confirmation_id': confirmation['id'], 'replayed': True}
            db.execute('INSERT INTO commits VALUES(?,?,?,?,?,?)', (principal['sub'], principal['workspace'], request_id, aid, body['definition_hash'], stable_json(replay)))
            return replay
        receipt = {'request_id': request_id, 'request_hash': digest, 'authoring_id': aid, 'definition_hash': body['definition_hash'],
                   'confirmation_id': confirmation['id'], 'status': 'started',
                   'operation_id': stable_hash([principal['sub'], principal['workspace'], aid, request_id])}
        unknown = db.execute("SELECT 1 FROM commits WHERE subject=? AND workspace=? AND definition_hash=? AND json_extract(body,'$.status') IN ('started','unknown')",
                             (principal['sub'], principal['workspace'], body['definition_hash'])).fetchone()
        if unknown:
            raise ResearchError('COMMIT_UNCERTAIN', '另一创作中相同定义的保存尚待核对，不能通过新会话重复创建。', status_code=409)
        db.execute('INSERT INTO commits VALUES(?,?,?,?,?,?)', (principal['sub'], principal['workspace'], request_id, aid, body['definition_hash'], stable_json(receipt)))
        db.execute('INSERT INTO definition_commits VALUES(?,?,?)', (aid, body['definition_hash'], request_id))
        state['committing'] = request_id
        db.execute('UPDATE authorings SET body=? WHERE id=?', (stable_json(state), aid))
        confirmation['status'] = 'accepted'
        db.execute('UPDATE confirmations SET body=? WHERE id=?', (stable_json(confirmation), confirmation['id']))
    # The existing indicator repository is not in this SQLite transaction. An unknown outcome stays fenced.
    try:
        created = service.create_indicator(confirmation['definition'], expected_catalog_revision=confirmation['repository_revision'], operation_id=receipt['operation_id'])
        receipt.update(status='succeeded', indicator_id=created['id'], indicator_revision=created['revision'], name=created.get('name'))
        with store.db() as db:
            state = store.read_authoring(aid, principal, db=db)
            state.pop('committing', None)
            db.execute('UPDATE authorings SET body=? WHERE id=?', (stable_json(state), aid))
            db.execute('UPDATE commits SET body=? WHERE subject=? AND workspace=? AND request_id=?',
                       (stable_json(receipt), principal['sub'], principal['workspace'], request_id))
        return receipt
    except IndicatorDomainError as exc:
        if exc.code != 'CATALOG_REVISION_CONFLICT':
            raise ResearchError('COMMIT_UNCERTAIN', '保存结果尚待核对。', status_code=409) from None
        receipt.update(status='rejected', code=exc.code)
        with store.db() as db:
            state = store.read_authoring(aid, principal, db=db)
            state.pop('committing', None)
            db.execute('UPDATE authorings SET body=? WHERE id=?', (stable_json(state), aid))
            db.execute('UPDATE commits SET body=? WHERE subject=? AND workspace=? AND request_id=?',
                       (stable_json(receipt), principal['sub'], principal['workspace'], request_id))
            db.execute('DELETE FROM definition_commits WHERE authoring_id=? AND definition_hash=? AND request_id=?', (aid, body['definition_hash'], request_id))
        raise ResearchError('CONFIRMATION_STALE', '目录已改变，未执行保存；请重新预览确认。', status_code=409) from None
    except Exception:
        raise ResearchError('COMMIT_UNCERTAIN', '指标可能已保存；请核对原操作，禁止重复创建。', status_code=409) from None

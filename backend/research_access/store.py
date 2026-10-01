"""Business drafts, authority references and operation receipts. No conversations or model state."""
from contextlib import contextmanager
import copy
import json
import os
import sqlite3
import time
import uuid
from pathlib import Path

from .authoring import storage_directory, stable_hash, stable_json, public_draft
from .contracts import ResearchError


class ResearchStore:
    def __init__(self, directory=None):
        self.directory = Path(directory) if directory else storage_directory()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / 'research.sqlite3'
        with self.db() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS contexts(id TEXT PRIMARY KEY, subject TEXT, workspace TEXT, body TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS authorings(id TEXT PRIMARY KEY, subject TEXT, workspace TEXT,
                    correlation TEXT, revision INTEGER NOT NULL, body TEXT NOT NULL, UNIQUE(subject,workspace,correlation));
                CREATE TABLE IF NOT EXISTS authoring_revisions(authoring_id TEXT, revision INTEGER, body TEXT NOT NULL,
                    PRIMARY KEY(authoring_id,revision));
                CREATE TABLE IF NOT EXISTS previews(id TEXT PRIMARY KEY, authoring_id TEXT, body TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS scenario_artifacts(id TEXT PRIMARY KEY, subject TEXT, workspace TEXT, body TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS confirmations(id TEXT PRIMARY KEY, authoring_id TEXT, body TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS commits(subject TEXT, workspace TEXT, request_id TEXT,
                    authoring_id TEXT, definition_hash TEXT, body TEXT NOT NULL, PRIMARY KEY(subject,workspace,request_id));
                CREATE TABLE IF NOT EXISTS definition_commits(authoring_id TEXT, definition_hash TEXT, request_id TEXT,
                    PRIMARY KEY(authoring_id,definition_hash));
                CREATE UNIQUE INDEX IF NOT EXISTS unresolved_definition_writes ON commits(subject,workspace,definition_hash)
                    WHERE json_extract(body,'$.status') IN ('started','unknown');
                CREATE TABLE IF NOT EXISTS operations(id TEXT PRIMARY KEY, subject TEXT, workspace TEXT,
                    scope_key TEXT, status TEXT, body TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS admissions(id TEXT PRIMARY KEY, subject TEXT, context_id TEXT, body TEXT NOT NULL);
            ''')

    @contextmanager
    def db(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA journal_mode=WAL')
        try:
            db.execute('BEGIN IMMEDIATE')
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def register_context(self, principal, value):
        owner = {'subject': principal['sub'], 'workspace': principal['workspace']}
        digest = stable_hash({**owner, **value})
        cid = 'ctx-'+digest[:32]
        record = {**value, **owner, 'id': cid, 'hash': digest, 'grant_id': 'grant-'+digest[:32],
                  'grant_revision': 1, 'revoked': False, 'expires': time.time()+3600}
        with self.db() as db:
            previous = db.execute('SELECT body FROM contexts WHERE id=?', (cid,)).fetchone()
            if previous:
                record = json.loads(previous[0])
                if record['revoked']:
                    raise ResearchError('CONTEXT_REVOKED', '该上下文授权已撤销。', status_code=403)
                record['expires'] = time.time()+3600
            db.execute('INSERT OR REPLACE INTO contexts VALUES(?,?,?,?)', (cid, owner['subject'], owner['workspace'], stable_json(record)))
        return record

    def context(self, cid, *, subject=None, workspace=None):
        with self.db() as db:
            row = db.execute('SELECT body FROM contexts WHERE id=?', (cid,)).fetchone()
        if not row:
            raise ResearchError('CONTEXT_NOT_FOUND', '页面上下文不存在。', status_code=404)
        record = json.loads(row[0])
        if (subject is not None and record['subject'] != subject) or (workspace is not None and record['workspace'] != workspace):
            raise ResearchError('CONTEXT_FORBIDDEN', '页面上下文不属于当前身份。', status_code=403)
        return record

    def renew_context(self, cid, principal):
        with self.db() as db:
            row = db.execute('SELECT body FROM contexts WHERE id=? AND subject=? AND workspace=?',
                             (cid, principal['sub'], principal['workspace'])).fetchone()
            record = json.loads(row[0]) if row else None
            if not record or record['revoked']:
                raise ResearchError('CONTEXT_REVOKED', '该页面授权不可续期。', status_code=403)
            record['expires'] = time.time()+3600
            db.execute('UPDATE contexts SET body=? WHERE id=?', (stable_json(record), cid))
            return record

    def authoring(self, principal, correlation, *, scope, context_hash):
        with self.db() as db:
            row = db.execute('SELECT body FROM authorings WHERE subject=? AND workspace=? AND correlation=?',
                             (principal['sub'], principal['workspace'], correlation)).fetchone()
            if row:
                return json.loads(row[0])
            aid = uuid.uuid4().hex
            value = {'id': aid, 'revision': 0, 'subject': principal['sub'], 'workspace': principal['workspace'],
                     'scope': scope, 'context_hash': context_hash, 'draft': None}
            db.execute('INSERT INTO authorings VALUES(?,?,?,?,?,?)',
                       (aid, principal['sub'], principal['workspace'], correlation, 0, stable_json(value)))
            return value

    def read_authoring(self, aid, principal=None, *, db=None):
        if db is None:
            with self.db() as connection:
                return self.read_authoring(aid, principal, db=connection)
        row = db.execute('SELECT body FROM authorings WHERE id=?', (aid,)).fetchone()
        if not row:
            raise ResearchError('AUTHORING_NOT_FOUND', '指标创作不存在。', status_code=404)
        value = json.loads(row[0])
        if principal and (value['subject'], value['workspace']) != (principal['sub'], principal['workspace']):
            raise ResearchError('AUTHORING_NOT_FOUND', '指标创作不存在。', status_code=404)
        return value

    def save_authoring(self, value, expected_revision, *, preview=None):
        with self.db() as db:
            current = self.read_authoring(value['id'], db=db)
            if current['revision'] != expected_revision or current.get('committing'):
                raise ResearchError('REVISION_CONFLICT', '草稿已更新或正在保存，请重新读取。', status_code=409)
            clean = {k: v for k, v in value.items() if not k.startswith('_')}
            clean['revision'] = expected_revision+1
            if preview:
                pid = uuid.uuid4().hex
                preview = {**preview, 'preview_id': pid, 'authoring_id': value['id']}
                db.execute('INSERT INTO previews VALUES(?,?,?)', (pid, value['id'], stable_json(preview)))
                clean['preview'] = {k: preview.get(k) for k in ('preview_id', 'authoring_id', 'definition_hash', 'draft_revision', 'context_hash', 'run_id', 'target', 'period', 'as_of', 'result_kind')}
            db.execute('UPDATE authorings SET revision=?,body=? WHERE id=?', (clean['revision'], stable_json(clean), value['id']))
            db.execute('INSERT INTO authoring_revisions VALUES(?,?,?)', (value['id'], clean['revision'], stable_json(clean)))
            return clean

    def preview(self, authoring_id, preview_id, historical=True):
        with self.db() as db:
            row = db.execute('SELECT body FROM previews WHERE id=? AND authoring_id=?', (preview_id, authoring_id)).fetchone()
        if not row:
            raise ResearchError('PREVIEW_NOT_FOUND', '试算结果不存在或已过期。', status_code=404)
        return json.loads(row[0])

    def save_scenario_artifact(self, oid, principal, value):
        with self.db() as db:
            db.execute('INSERT INTO scenario_artifacts VALUES(?,?,?,?)', (oid, principal['sub'], principal['workspace'], stable_json(value)))

    def scenario_artifact(self, oid, principal):
        with self.db() as db:
            row = db.execute('SELECT body FROM scenario_artifacts WHERE id=? AND subject=? AND workspace=?',
                             (oid, principal['sub'], principal['workspace'])).fetchone()
        if not row:
            raise ResearchError('ARTIFACT_NOT_FOUND', '情景成果不存在。', status_code=404)
        return json.loads(row[0])

    def public_authoring(self, aid, principal, revision=None):
        value = self.read_authoring(aid, principal)
        current_revision = value['revision']
        current_draft = {k: (value.get('draft') or {}).get(k) for k in ('definition_hash', 'draft_revision', 'valid')}
        with self.db() as db:
            if revision is not None and revision != current_revision:
                row = db.execute('SELECT body FROM authoring_revisions WHERE authoring_id=? AND revision=?', (aid, revision)).fetchone()
                if not row:
                    raise ResearchError('AUTHORING_REVISION_NOT_FOUND', '历史草稿版本不存在。', status_code=404)
                value = json.loads(row[0])
            saved = [json.loads(r[0]) for r in db.execute('SELECT body FROM commits WHERE authoring_id=?', (aid,))]
        return {'id': aid, 'revision': value['revision'], 'current_revision': current_revision, 'current_draft': current_draft,
                'draft': public_draft(value), 'preview': value.get('preview'),
                'saved': [r for r in saved if r.get('status') == 'succeeded']}

    def start_operation(self, principal, payload, name, scope_key):
        oid = payload['operation_id']
        digest = stable_hash({'name': name, 'payload': payload})
        with self.db() as db:
            old = db.execute('SELECT body FROM operations WHERE id=?', (oid,)).fetchone()
            if old:
                value = json.loads(old[0])
                if value['hash'] != digest or value['subject'] != principal['sub'] or value['workspace'] != principal['workspace']:
                    raise ResearchError('REQUEST_CONFLICT', '同一操作标识不能改变参数或身份。', status_code=409)
                return value, False
            busy = db.execute("SELECT 1 FROM operations WHERE scope_key=? AND status IN ('accepted','running','stop_requested','unknown') LIMIT 1", (scope_key,)).fetchone()
            if busy:
                raise ResearchError('RESEARCH_OPERATION_BUSY', '此前业务操作尚未确认结束。', status_code=409)
            value = {'operation_id': oid, 'subject': principal['sub'], 'workspace': principal['workspace'], 'hash': digest,
                     'scope_key': scope_key, 'name': name, 'payload': payload, 'status': 'accepted', 'created_at': time.time()}
            db.execute('INSERT INTO operations VALUES(?,?,?,?,?,?)', (oid, principal['sub'], principal['workspace'], scope_key, 'accepted', stable_json(value)))
            return value, True

    def operation(self, oid):
        with self.db() as db:
            row = db.execute('SELECT body FROM operations WHERE id=?', (oid,)).fetchone()
        if not row:
            raise ResearchError('OPERATION_NOT_FOUND', '业务操作不存在，不能据此重新执行。', status_code=404)
        return json.loads(row[0])

    def update_operation(self, oid, **fields):
        with self.db() as db:
            row = db.execute('SELECT body FROM operations WHERE id=?', (oid,)).fetchone()
            value = json.loads(row[0]) | fields
            db.execute('UPDATE operations SET status=?,body=? WHERE id=?', (value['status'], stable_json(value), oid))
            return value

    def claim_operation(self, oid):
        with self.db() as db:
            row = db.execute('SELECT body FROM operations WHERE id=?', (oid,)).fetchone()
            value = json.loads(row[0]) if row else None
            if not value or value['status'] != 'accepted' or value.get('cancel_requested'):
                return None
            value.update(status='running', executor_pid=os.getpid())
            db.execute('UPDATE operations SET status=?,body=? WHERE id=?', ('running', stable_json(value), oid))
            return value

    def cancel_operation(self, oid):
        with self.db() as db:
            row = db.execute('SELECT body FROM operations WHERE id=?', (oid,)).fetchone()
            if not row:
                raise ResearchError('OPERATION_NOT_FOUND', '业务操作不存在。', status_code=404)
            value = json.loads(row[0])
            if value['status'] not in {'succeeded', 'failed', 'cancelled'}:
                value.update(status='cancelled' if value['status'] == 'accepted' else 'stop_requested', cancel_requested=True)
                db.execute('UPDATE operations SET status=?,body=? WHERE id=?', (value['status'], stable_json(value), oid))
            return value

    def recover(self):
        with self.db() as db:
            for row in db.execute("SELECT subject,workspace,request_id,body FROM commits WHERE json_extract(body,'$.status')='started'").fetchall():
                value = json.loads(row['body'])
                value['status'] = 'unknown'
                db.execute('UPDATE commits SET body=? WHERE subject=? AND workspace=? AND request_id=?',
                           (stable_json(value), row['subject'], row['workspace'], row['request_id']))
            for row in db.execute("SELECT id,body FROM operations WHERE status IN ('accepted','running','stop_requested')").fetchall():
                value = json.loads(row['body'])
                value['status'] = 'unknown'
                db.execute('UPDATE operations SET status=?,body=? WHERE id=?', ('unknown', stable_json(value), row['id']))

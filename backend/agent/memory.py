"""Human memory proposals: accept or reject, append-only audit.

Accepted proposals append one line to the scope memory file under
``agent_memory/``; every decision is also recorded in ``audit.jsonl``.  Both
files are written atomically through the shared path guard.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import uuid
from pathlib import Path
from typing import Any, Optional

from backend.data_storage import guard_path

from .contracts import AgentError, MemoryRequest
from . import data_policy
from .llm_settings import resolve_agent_data_dir
from .sessions import append_event, assert_mutable, utc_now

MEMORY_FILES = {
    "indicator_center": "indicator-center.md",
    "product_research": "product-research.md",
}
MAX_MEMORY_LINE_CHARS = 500


def memory_identity(payload):
    """Keep automatic definition memories in their workspace scope, including signed old records."""
    key = payload.get('key') or 'indicator_definition:' + str(payload.get('definition_hash') or '')
    object_id = payload.get('object_id') or 'scope'
    digest = None
    if not payload.get('source_message_id') and payload.get('scope') in MEMORY_FILES:
        if payload.get('kind') == 'indicator_definition' and payload.get('confirmation_id'):
            digest = payload.get('definition_hash')
        elif key == 'indicator_definition:' + str(object_id):
            # Record callers verify its original seal before projecting this old identity.
            digest = object_id
    if isinstance(digest, str) and re.fullmatch(r'[0-9a-f]{64}', digest):
        return {'key': 'indicator_definition:' + digest, 'object_id': 'scope'}
    return {'key': key, 'object_id': object_id}


def memory_root(root: Optional[Path] = None) -> Path:
    return Path(root) if root is not None else resolve_agent_data_dir() / "agent_memory"


def _atomic_append(path: Path, line: str, *, marker: Optional[str] = None) -> None:
    guard_path(path, write=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    if marker and marker in existing:
        return
    if existing and not existing.endswith("\n"):
        existing += "\n"
    descriptor, temporary_path = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(existing + line + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def resolve(*, store: Any, session_id: str, request: MemoryRequest, root: Optional[Path] = None) -> dict[str, Any]:
    with store.locked(session_id) as state:
        assert_mutable(state)
        decisions = state.setdefault("memory", {})
        existing = decisions.get(request.proposal_id)
        if isinstance(existing, dict):
            if (existing.get("decision") != request.decision
                    or existing.get('replacement') != request.replace_memory_id):
                raise AgentError(
                    "AGENT_MEMORY_DECISION_CONFLICT",
                    "该记忆提案已有不同的人工决定。",
                    status_code=409,
                    field="decision",
                )
            return {**existing, "replayed": True}
        proposals = state.get("memory_proposals") or []
        proposal = next(
            (item for item in proposals if item.get("proposal_id") == request.proposal_id),
            None,
        )
        if proposal is None:
            raise AgentError("AGENT_MEMORY_PROPOSAL_NOT_FOUND", "未找到指定记忆提案。", status_code=404, field="proposal_id")
        memory_path = memory_root(root) / MEMORY_FILES.get(str(state.get("scope")), "shared.md")
        record: dict[str, Any] = {
            "proposal_id": request.proposal_id,
            "decision": request.decision,
            "speaker": request.speaker,
            "scope": state.get("scope"),
            "at": utc_now(),
            "replacement": request.replace_memory_id,
            "file": str(memory_path) if request.decision == "accept" else None,
        }
        if request.decision == "accept":
            summary = str(proposal.get("summary") or proposal.get("definition_hash") or "")[:MAX_MEMORY_LINE_CHARS]
            if data_policy.user_text_violation(summary) or data_policy.user_text_violation(request.speaker):
                raise data_policy._blocked("memory", "structured_memory_payload")
            db = store._local.db
            source_id = proposal.get('source_message_id')
            traceable = False
            if source_id:
                source = db.execute("SELECT body FROM events WHERE session_id=? AND json_extract(body,'$.type')='user.message' AND json_extract(body,'$.id')=?",
                                    (session_id, source_id)).fetchone()
                traceable = source is not None and summary in json.loads(source[0]).get('text', '')
                if not traceable:
                    raise AgentError('AGENT_MEMORY_SOURCE_INVALID', '提案来源已失效，不能接受。', status_code=409)
            elif proposal.get('confirmation_id'):
                traceable = proposal.get('kind') == 'indicator_definition' and bool(proposal.get('definition_hash'))
            identity = memory_identity(proposal)
            try:
                active = [json.loads(row[0]) for row in db.execute('SELECT body FROM memory_records')]
            except (ValueError, TypeError):
                raise AgentError('AGENT_MEMORY_CORRUPT', '记忆存储含损坏记录；本次决定未应用。', status_code=409) from None
            conflicts = [item for item in active if _valid_record(item) and item.get('scope') == state['scope']
                         and memory_identity(item) == identity]
            if conflicts:
                if (len(conflicts) != 1 or request.replace_memory_id != conflicts[0]['memory_id']
                        or request.expected_version != conflicts[0]['version']):
                    raise AgentError('AGENT_MEMORY_CONFLICT', '已有同类偏好；请明确选择替换其当前版本。', status_code=409,
                                     diagnostics=[{'memory_id': item['memory_id'], 'version': item['version']} for item in conflicts])
                prior = data_policy.seal({**conflicts[0], 'status': 'replaced', 'version': conflicts[0]['version'] + 1}, 'memory.record')
                db.execute('UPDATE memory_records SET body=? WHERE id=?', (json.dumps(prior, ensure_ascii=False), prior['memory_id']))
            elif request.replace_memory_id:
                raise AgentError('AGENT_MEMORY_STALE', '待替换的记忆已改变。', status_code=409)
            memory_id = 'memory-' + uuid.uuid4().hex
            versioned = data_policy.seal({'memory_id': memory_id, 'version': 1, 'status': 'accepted',
                'scope': state['scope'], **identity, 'text': summary,
                'source_session_id': session_id, 'source_message_id': proposal.get('source_message_id'),
                'proposal_id': proposal['proposal_id'], 'accepted_at': record['at'],
                'source_verified': traceable,
                'replaces': request.replace_memory_id}, 'memory.record')
            db.execute('INSERT INTO memory_records VALUES (?,?)', (memory_id, json.dumps(versioned, ensure_ascii=False)))
            record.update(memory_id=memory_id, version=1)
            _atomic_append(memory_path, f"- {record['at']} [{request.speaker}] {summary} ({request.proposal_id})",
                           marker=request.proposal_id)
            proposal["status"] = "accepted"
        else:
            proposal["status"] = "rejected"
        _atomic_append(
            memory_root(root) / "audit.jsonl",
            json.dumps(
                {"type": "memory_decision", **record, "summary_chars": len(str(proposal.get("summary") or ""))},
                ensure_ascii=False,
                sort_keys=True,
            ),
            marker=request.proposal_id,
        )
        decisions[request.proposal_id] = record
        state["session_revision"] = int(state.get("session_revision") or 0) + 1
        append_event(
            state,
            {
                "type": "memory",
                "speaker": request.speaker,
                "scope": state.get("scope"),
                "status": request.decision,
                "summary_chars": len(str(proposal.get("summary") or "")),
            },
        )
        store.write(state)
        return record


def propose(*, store, session_id, state, source_message_id, quote, key, object_id='scope'):
    """A tentative local patch; the run's existing owner/cancel transaction applies it."""
    from .sessions import stable_hash
    if data_policy.user_text_violation(quote):
        raise data_policy._blocked('memory', 'structured_memory_payload')
    with store.connection() as db:
        source = db.execute("SELECT body FROM events WHERE session_id=? AND json_extract(body,'$.type')='user.message' AND json_extract(body,'$.id')=?",
                            (session_id, source_message_id)).fetchone()
    if source is None or quote not in str(json.loads(source[0]).get('text') or ''):
        raise AgentError('AGENT_MEMORY_SOURCE_INVALID', '记忆提案必须引用当前会话的有效用户原话。', status_code=409)
    targets = ((state.get('page_context') or {}).get('calculation') or {}).get('targets', [])
    if object_id not in {'scope', *(target['product_id'] for target in targets)}:
        raise AgentError('AGENT_MEMORY_OBJECT_INVALID', '记忆对象必须来自当前页面。', status_code=409)
    proposal_id = 'mem-' + stable_hash([source_message_id, key, object_id, quote])[:32]
    existing = next((p for p in (state.get('memory_proposals') or []) if p['proposal_id'] == proposal_id), None)
    if existing:
        return existing
    proposal = {'proposal_id': proposal_id, 'kind': 'preference', 'key': key, 'object_id': object_id,
                'source_message_id': source_message_id, 'summary': quote, 'scope': state['scope'],
                'status': 'pending', 'created_at': utc_now()}
    state['memory_proposals'] = [*(state.get('memory_proposals') or []), proposal]
    return proposal


def _valid_record(payload):
    if not data_policy.verify(payload, 'memory.record'):
        return False
    return (isinstance(payload.get('text'), str) and not data_policy.user_text_violation(payload['text'])
            and payload.get('status') == 'accepted' and isinstance(payload.get('version'), int)
            and payload.get('source_verified') is True)


def recall(*, store, session_id, object_ids=()):
    """Only signed, accepted, applicable records; legacy Markdown is never an input."""
    with store.connection() as db:
        state = store._read_state(db, session_id)
        records = []
        for row in db.execute('SELECT body FROM memory_records ORDER BY rowid'):
            try:
                record = json.loads(row[0])
            except (ValueError, TypeError):
                continue
            if not _valid_record(record) or record.get('scope') != state['scope']:
                continue
            record = {**record, **memory_identity(record)}
            if record.get('object_id') in {'scope', *object_ids}:
                records.append({key: record.get(key) for key in ('memory_id', 'version', 'scope', 'object_id',
                                'key', 'text', 'source_session_id', 'source_message_id', 'proposal_id', 'accepted_at')})
        # An accidental/corrupted conflict cannot choose a preference implicitly.
        counts = {}
        for record in records:
            identity = (record['scope'], record['object_id'], record['key'])
            counts[identity] = counts.get(identity, 0) + 1
        return [record for record in records if counts[(record['scope'], record['object_id'], record['key'])] == 1]


def revoke(*, store, session_id, request):
    from .sessions import stable_hash
    with store.locked(session_id) as state:
        assert_mutable(state)
        db = store._local.db
        digest = stable_hash(request.model_dump())
        old = db.execute('SELECT body FROM memory_actions WHERE session_id=? AND request_id=?',
                         (session_id, request.request_id)).fetchone()
        if old:
            action = json.loads(old[0])
            if action['request_hash'] != digest:
                raise AgentError('AGENT_MEMORY_DECISION_CONFLICT', '同一操作标识不能改变记忆决定。', status_code=409)
            return {**action['result'], 'replayed': True}
        row = db.execute('SELECT body FROM memory_records WHERE id=?', (request.memory_id,)).fetchone()
        record = json.loads(row[0]) if row else None
        if not _valid_record(record) or record.get('scope') != state['scope'] or record['version'] != request.expected_version:
            raise AgentError('AGENT_MEMORY_STALE', '记忆版本已改变或不属于当前作用域。', status_code=409)
        record = data_policy.seal({**record, 'status': 'revoked', 'version': record['version'] + 1,
                                   'revoked_at': utc_now()}, 'memory.record')
        db.execute('UPDATE memory_records SET body=? WHERE id=?', (json.dumps(record, ensure_ascii=False), request.memory_id))
        result = {'memory_id': request.memory_id, 'version': record['version'], 'status': 'revoked'}
        db.execute('INSERT INTO memory_actions VALUES (?,?,?)', (session_id, request.request_id,
                   json.dumps({'request_hash': digest, 'result': result})))
        state['session_revision'] += 1
        append_event(state, {'type': 'memory.revoked', 'data': result})
        store.write(state)
        return result

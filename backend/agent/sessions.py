"""Transactional agent state, event cursors and run receipts; no business writes."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import sqlite3
import threading
import uuid
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

from backend.data_storage import guard_path
from custom_indicators.errors import IndicatorDomainError
from custom_indicators.run_result_repository import EvaluationRunResultRepository
from .contracts import AgentError, PageContext
from .llm_settings import resolve_agent_data_dir
from .storage import AgentJsonStore

SESSION_SCHEMA_VERSION = 2
SESSION_ID_PATTERN = re.compile(r"^agent-[0-9a-f]{32}$")
ACTIVE_STATUSES = {"queued", "running", "stopping"}
MAX_EVENTS = 200  # Page size, never a lifetime event or turn limit.
# Session state a turn may rewrite; a replacement restores the values its turn started from.
BASE_STATE_KEYS = ("draft", "last_valid_draft", "preview", "product_candidates", "memory_proposals", "task_plans")
RUN_CONTENT_FIELDS = ("checkpoint", "detector_state", "tool_trace")
RUN_METADATA_FIELDS = ("phase", "usage", "catalog_version", "data_generation", "effective_context", "calculation_hash")
CONTROL_EVENTS = {"data.admitted", "data.rejected", "data.reprojected", "model.returned", "run.phase", "run.recovering", "tool.started"}
CONTENT_EVENTS = {"run.started", "run.phase", "run.recovering", "model.completed", "context.compacted", "tool.completed", "draft.updated", "preview.updated", "progress.updated"}
TERMINAL_STATUSES = {"completed", "paused", "cancelled", "failed", "interrupted"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()


def task_plan_id(arguments):
    return 'plan-' + stable_hash(arguments)[:16]


def context_hash(page_context: PageContext) -> str:
    return stable_hash(page_context.model_dump())


def page_snapshot_content(snapshot: Any) -> Optional[dict[str, Any]]:
    """Semantic page content without transport metadata (snapshot id and capture time)."""
    if not isinstance(snapshot, dict):
        return None
    sections = snapshot.get("sections")
    return {"version": snapshot.get("version"), "page": snapshot.get("page"),
            "sections": sections if isinstance(sections, dict) else {}}


def page_snapshot_identity(snapshot: Any, section: Optional[str] = None) -> Optional[str]:
    """Stable identity used to reuse or block page evidence; None means no snapshot at all.

    A section identity covers that section only: another section changing must neither
    unblock an identical read nor let an unchanged read look fresh. The whole-content
    identity (``section=None``) still compares resume snapshots.
    """
    content = page_snapshot_content(snapshot)
    if content is None:
        return None
    if section is None:
        return stable_hash(content)
    return stable_hash({"version": content["version"], "page": content["page"],
                        "section": str(section), "section_content": content["sections"].get(str(section))})


def definition_hash(definition: dict[str, Any]) -> str:
    return stable_hash(definition)


def append_event(state: dict[str, Any], event: dict[str, Any]) -> None:
    seq = int(state.get("next_event_seq", 1))
    state["next_event_seq"] = seq + 1
    state.setdefault("events", []).append({**event, "seq": seq, "created_at": utc_now()})


def reconcile_memory_proposals(state):
    """Human decision receipts outlive tool snapshots and edited turns."""
    decisions = state.get('memory') or {}
    for proposal in state.get('memory_proposals') or []:
        receipt = decisions.get(proposal.get('proposal_id'))
        if isinstance(receipt, dict) and receipt.get('decision') in {'accept', 'reject'}:
            proposal['status'] = 'accepted' if receipt['decision'] == 'accept' else 'rejected'


def stop_message(run):
    if run.get("stop_reason") == "user_cancelled":
        return "已停止自动处理，已提交的进度已保留。"
    change = run.get("context_change") or {}
    labels = {"page": "研究页面", "context_revision": "编辑器版本", "view_state": "PIT 查看口径",
              "calculation.context_kind": "计算对象类型", "calculation.targets": "预览产品",
              "calculation.period": "计算周期", "calculation.as_of": "历史截止日", "calculation.run_id": "组合快照"}
    fields = change.get("fields", [])
    # The frontend may advance the revision solely to fence another field change.
    meaningful = [key for key in fields if key != "context_revision"] or fields
    subject = "、".join(labels[key] for key in meaningful if key in labels)
    if not subject:
        subject = {"catalog": "指标或算子目录", "data": "数据快照"}.get(change.get("source"), "页面研究条件")
    return f"{subject}已变化，本轮已暂停。对话和已完成的进度已保留，点击“继续分析”使用当前条件。"


class AgentSessionStore:
    def __init__(self, root: Optional[Path] = None) -> None:
        self.root = Path(root) if root is not None else resolve_agent_data_dir() / "agent_sessions"
        guard_path(self.root, write=True)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "harness.sqlite3"
        self.previews = EvaluationRunResultRepository(self.root / "previews")
        self._local = threading.local()
        with self.connection() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, body TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY, session_id TEXT NOT NULL, message_id TEXT NOT NULL,
                    request_hash TEXT NOT NULL, status TEXT NOT NULL, body TEXT NOT NULL,
                    UNIQUE(session_id,message_id));
                CREATE INDEX IF NOT EXISTS runs_session ON runs(session_id);
                CREATE TABLE IF NOT EXISTS events (
                    session_id TEXT NOT NULL, seq INTEGER NOT NULL, body TEXT NOT NULL,
                    PRIMARY KEY(session_id,seq));
                CREATE TABLE IF NOT EXISTS tool_calls (
                    run_id TEXT NOT NULL, model_step INTEGER NOT NULL, call_id TEXT NOT NULL,
                    body TEXT NOT NULL, PRIMARY KEY(run_id,model_step,call_id));
                CREATE TABLE IF NOT EXISTS commit_intents (
                    session_id TEXT NOT NULL, request_id TEXT NOT NULL, body TEXT NOT NULL,
                    PRIMARY KEY(session_id,request_id));
                CREATE TABLE IF NOT EXISTS progress_facts (
                    space_id TEXT NOT NULL, kind TEXT NOT NULL, value TEXT NOT NULL,
                    PRIMARY KEY(space_id,kind,value));
                CREATE TABLE IF NOT EXISTS memory_records (id TEXT PRIMARY KEY, body TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS memory_actions (
                    session_id TEXT NOT NULL, request_id TEXT NOT NULL, body TEXT NOT NULL,
                    PRIMARY KEY(session_id,request_id));
                CREATE INDEX IF NOT EXISTS runs_unfinished ON runs(id)
                    WHERE status IN ('queued','running','stopping')
                    OR json_extract(body,'$.execution_blocked_by') IS NOT NULL;
                CREATE INDEX IF NOT EXISTS events_chat ON events(session_id,seq)
                    WHERE json_extract(body,'$.type') IN ('user.message','assistant.message');
            """)
        os.chmod(self.path, 0o600)

    @contextmanager
    def connection(self, *, write=False):
        guard_path(self.path, write=write)
        db = None
        try:
            db = sqlite3.connect(self.path, timeout=3)
            db.row_factory = sqlite3.Row
            if write:
                db.execute("BEGIN IMMEDIATE")
            with db:
                yield db
        except sqlite3.Error as exc:
            code = getattr(exc, "sqlite_errorcode", 0) & 0xff
            message = "智能体存储繁忙，请稍后重试。" if code in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED} else "智能体存储不可用，已保留最后提交的进度，请检查数据盘。"
            raise AgentError("AGENT_STORAGE_UNAVAILABLE", message, status_code=503) from None
        finally:
            if db is not None:
                db.close()

    def _validate_id(self, session_id):
        if not SESSION_ID_PATTERN.fullmatch(str(session_id)):
            raise AgentError("AGENT_SESSION_NOT_FOUND", "未找到指定会话。", status_code=404)

    def save_preview(self, run, payload):
        guard_path(self.previews.directory, write=True)
        artifact = {**payload, "session_id": run["session_id"], "run_id": run["run_id"],
                    "context_hash": context_hash(PageContext.model_validate(run["request"]["page_context"])),
                    "data_generation": run.get("data_generation"), "effective_context": run.get("effective_context"),
                    "created_at": utc_now()}
        preview_id = self.previews.store({"rows": [artifact]})
        return {key: value for key, value in {**artifact, "preview_id": preview_id}.items()
                if key not in {"result", "definition"}}

    def preview(self, session_id, preview_id, *, historical=False):
        state = self.read(session_id)
        reference = state.get("preview") or {}
        if historical:
            with self.connection() as db:
                row = db.execute("SELECT body FROM events WHERE session_id=? AND json_extract(body,'$.type')='preview.updated' AND json_extract(body,'$.data.preview.preview_id')=? ORDER BY seq DESC LIMIT 1", (session_id, preview_id)).fetchone()
            if row is None:
                raise AgentError("AGENT_PREVIEW_STALE", "未找到此会话对应的试算记录。", status_code=409)
            reference = json.loads(row[0])["data"]["preview"]
        elif (reference.get("preview_id") != preview_id or reference.get("context_hash") != state.get("context_hash")
                or reference.get("definition_hash") != (state.get("draft") or {}).get("definition_hash")):
            raise AgentError("AGENT_PREVIEW_STALE", "这份试算已失效，请按当前指标和研究条件重新试算。", status_code=409)
        guard_path(self.previews.directory, write=True)
        try:
            page = self.previews.page(preview_id, page_size=1)
        except IndicatorDomainError as exc:
            if exc.code not in {"RESULT_EXPIRED", "RESULT_NOT_FOUND"}:
                raise
            raise AgentError("AGENT_PREVIEW_EXPIRED", "试算结果已过期或已清理，请让助手重新试算。", status_code=410) from None
        artifact = page["rows"][0]
        if (artifact["session_id"] != session_id or artifact.get("definition_hash") != reference.get("definition_hash")
                or artifact.get("context_hash") != reference.get("context_hash")):
            raise AgentError("AGENT_PREVIEW_STALE", "这份试算不属于当前会话。", status_code=409)
        # Recheck after file I/O: a new context or draft may have invalidated the handle.
        if not historical and self.read(session_id).get("preview") != reference:
            raise AgentError("AGENT_PREVIEW_STALE", "研究条件已变化，请重新试算。", status_code=409)
        return {**artifact, "preview_id": preview_id, "expires_at": page["pagination"]["expires_at"]}

    def _artifacts(self, db, session_id, run_ids):
        if not run_ids:
            return {}
        placeholders = ','.join('?' for _ in run_ids)
        rows = db.execute(f"SELECT body FROM events WHERE session_id=? AND json_extract(body,'$.run_id') IN ({placeholders}) AND json_extract(body,'$.type') IN ('draft.updated','preview.updated') ORDER BY seq", (session_id, *run_ids))
        artifacts = {}
        for row in rows:
            event = json.loads(row[0])
            key = 'draft' if event['type'] == 'draft.updated' else 'preview'
            artifacts.setdefault(event['run_id'], {})[key] = event.get('data', {}).get(key)
        return artifacts

    def _message_artifacts(self, db, session_id, items):
        # Old replies are reconstructed from committed tool events, never from
        # the session's current draft (which may belong to a different turn).
        replies = [e for e in items if e.get('speaker') == 'assistant' and e.get('run_id') and 'artifacts' not in e]
        artifacts = self._artifacts(db, session_id, list({e['run_id'] for e in replies}))
        for reply in replies:
            reply['artifacts'] = artifacts.get(reply['run_id'], {})
        return items

    def _migrate(self, session_id):
        self._validate_id(session_id)
        with self.connection() as db:
            if db.execute("SELECT 1 FROM sessions WHERE id=?", (session_id,)).fetchone():
                return
        legacy = AgentJsonStore(self.root / f"{session_id}.json")
        if not legacy.path.exists():
            return
        with legacy.locked():
            state = legacy.read_unlocked()
        if state.get("session_id") != session_id:
            raise AgentError("AGENT_STORAGE_CORRUPT", "旧会话格式不正确，未迁移。", status_code=500)
        with self.connection(write=True) as db:
            if db.execute("SELECT 1 FROM sessions WHERE id=?", (session_id,)).fetchone():
                return
            old_events = state.pop("events", [])
            history = state.pop("history", [])
            state["conversation"] = {"messages": history, "summary": "", "model_step": 0}
            turns = state.pop("turns", {})
            commits = state.pop("commits", {})
            state.update(schema_version=2, next_event_seq=1, events=[], legacy_history_incomplete=True, active_run_id=None)
            for event in old_events:
                append_event(state, {**event, "legacy_seq": event.get("seq")})
            for item in history:
                role = item.get("role")
                if role in {"user", "assistant"}:
                    append_event(state, {"type": f"{role}.message", "speaker": role, "text": item.get("content", ""), "legacy": True})
            for message_id, receipt in turns.items():
                response = receipt.get("response", {})
                run = {"run_id": f"run-{uuid.uuid4().hex}", "session_id": session_id, "message_id": message_id,
                       "status": "completed", "phase": "thinking", "run_revision": 1,
                       "request_hash": receipt.get("request_fingerprint", ""), "response": response,
                       "legacy": True, "created_at": receipt.get("at", utc_now())}
                self._write_run(db, run)
            for request_id, receipt in commits.items():
                db.execute("INSERT INTO commit_intents VALUES (?,?,?)", (session_id, request_id, stable_json({"state": "completed", **receipt})))
            self._write_state(db, state)

    def create(self, *, page_context: PageContext, scope: str) -> dict[str, Any]:
        now = utc_now()
        state = {"schema_version": 2, "session_id": f"agent-{uuid.uuid4().hex}", "created_at": now,
                 "updated_at": now, "scope": scope, "session_revision": 0, "next_event_seq": 1,
                 "page_context": page_context.model_dump(), "context_hash": context_hash(page_context),
                 "draft": None, "last_valid_draft": None, "pending_confirmation": None,
                 "memory_proposals": [], "memory": {}, "events": [], "active_run_id": None}
        with self.connection(write=True) as db:
            self._write_state(db, state)
        return state

    def _read_state(self, db, session_id):
        row = db.execute("SELECT body FROM sessions WHERE id=?", (session_id,)).fetchone()
        if row is None:
            raise AgentError("AGENT_SESSION_NOT_FOUND", "未找到指定会话。", status_code=404)
        try:
            state = json.loads(row["body"])
        except (ValueError, TypeError):
            raise AgentError("AGENT_STORAGE_CORRUPT", "会话数据格式无效。", status_code=500) from None
        reconcile_memory_proposals(state)
        state["events"] = []
        return state

    @contextmanager
    def locked(self, session_id: str) -> Iterator[dict[str, Any]]:
        self._migrate(session_id)
        with self.connection(write=True) as db:
            self._local.db = db
            try:
                yield self._read_state(db, session_id)
            finally:
                self._local.db = None

    def _write_state(self, db, state):
        reconcile_memory_proposals(state)
        state["updated_at"] = utc_now()
        for event in state.get("events", []):
            db.execute("INSERT INTO events VALUES (?,?,?)", (state["session_id"], event["seq"], stable_json(event)))
        state["events"] = []
        body = {k: v for k, v in state.items() if k != "events"}
        db.execute("INSERT INTO sessions VALUES (?,?) ON CONFLICT(id) DO UPDATE SET body=excluded.body", (state["session_id"], stable_json(body)))

    def write(self, state):
        db = getattr(self._local, "db", None)
        if db is None:
            raise RuntimeError("Session writes require a short transaction")
        self._write_state(db, state)

    def read(self, session_id):
        self._migrate(session_id)
        with self.connection() as db:
            return self._read_state(db, session_id)

    def list_sessions(self, limit=20):
        for path in self.root.glob("agent-*.json"):
            if SESSION_ID_PATTERN.fullmatch(path.stem):
                self._migrate(path.stem)
        with self.connection() as db:
            rows = db.execute("SELECT body FROM sessions ORDER BY json_extract(body,'$.updated_at') DESC LIMIT ?", (limit,))
            return [{k: s.get(k) for k in ("session_id", "scope", "session_revision", "created_at", "updated_at")} for s in (json.loads(row[0]) for row in rows)]

    def events(self, session_id, *, after_seq=0, limit=MAX_EVENTS, public=True):
        self._migrate(session_id)
        with self.connection() as db:
            state = self._read_state(db, session_id)
            rows = list(db.execute("SELECT body FROM events WHERE session_id=? AND seq>? ORDER BY seq LIMIT ?", (session_id, after_seq, limit+1)))
            entries = self._message_artifacts(db, session_id, [json.loads(row[0]) for row in rows[:limit]])
        if public:
            entries = [{k: v for k, v in e.items() if not k.startswith("_")} for e in entries]
        return {"items": entries, "has_more": len(rows)>limit, "last_seq": entries[-1]["seq"] if entries else after_seq,
                "oldest_seq": 1, "next_event_seq": state["next_event_seq"]}

    def public(self, session_id, *, event_from=1):
        self._migrate(session_id)
        with self.connection() as db:
            db.execute("BEGIN")
            state = self._read_state(db, session_id)
            state["events"] = [json.loads(row[0]) for row in db.execute("SELECT body FROM events WHERE session_id=? AND seq>=? ORDER BY seq LIMIT ?", (session_id,event_from,MAX_EVENTS))]
            result = public_state(state, event_from=event_from)
            saved = self.definition_commit(db, session_id, (state.get("draft") or {}).get("definition_hash"))
            result["saved_commit"] = saved.get("response") if saved and saved.get("state") == "completed" else None
            rid = state.get("active_run_id") or state.get("last_run_id")
            result["active_run"] = public_run(self._read_run(db, session_id, rid)) if rid else None
            if rid:
                result["active_run"]["artifacts"] = self._artifacts(db, session_id, [rid]).get(rid, {})
            page = self._message_page(db, session_id, None, MAX_EVENTS)
            result.update(messages=page["items"], older_message_cursor=page["older_cursor"])
            return result

    def _message_page(self, db, session_id, before_seq, limit):
        rows = list(db.execute("SELECT body FROM events WHERE session_id=? AND (? IS NULL OR seq<?) AND json_extract(body,'$.type') IN ('user.message','assistant.message') ORDER BY seq DESC LIMIT ?", (session_id,before_seq,before_seq,limit+1)))
        items = [{k:v for k,v in json.loads(row[0]).items() if not k.startswith('_')} for row in reversed(rows[:limit])]
        return {"items":self._message_artifacts(db, session_id, items), "has_more":len(rows)>limit, "older_cursor":items[0]["seq"] if len(rows)>limit else None}

    def message_page(self, session_id, *, before_seq=None, limit=MAX_EVENTS):
        self._migrate(session_id)
        with self.connection() as db:
            self._read_state(db, session_id)
            return self._message_page(db, session_id, before_seq, limit)

    def _write_run(self, db, run):
        db.execute("INSERT INTO runs VALUES (?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET status=excluded.status,body=excluded.body",
                   (run["run_id"], run["session_id"], run["message_id"], run["request_hash"], run["status"], stable_json(run)))

    def _read_run(self, db, session_id, run_id):
        row = db.execute("SELECT body FROM runs WHERE id=? AND session_id=?", (run_id, session_id)).fetchone()
        if row is None:
            raise AgentError("AGENT_RUN_NOT_FOUND", "未找到指定运行。", status_code=404)
        return json.loads(row[0])

    def get_run(self, session_id, run_id):
        self._validate_id(session_id)
        with self.connection() as db:
            return self._read_run(db, session_id, run_id)

    def read_context(self, session_id, operation_id, offset=0, limit=3000):
        """Read a completed, applied receipt from this session, without replaying it."""
        from . import data_policy
        from .tools import model_view
        from .views import Projection

        with self.connection() as db:
            row = db.execute("""SELECT t.body FROM tool_calls t JOIN runs r ON r.id=t.run_id
                WHERE r.session_id=? AND json_extract(t.body,'$.operation_id')=?
                AND json_extract(r.body,'$.superseded_by') IS NULL
                AND json_extract(t.body,'$.status')='completed'
                AND json_extract(t.body,'$.applied')=1 LIMIT 1""", (session_id, operation_id)).fetchone()
        if row is None:
            raise AgentError('AGENT_EVIDENCE_NOT_FOUND', '该历史证据不可用。', status_code=404)
        receipt = json.loads(row[0])
        tool = receipt.get('tool')
        # A signed receipt is returned byte-identical; anything else is rebuilt through
        # the registered strict view (values without a recheckable proof are omitted).
        evidence = data_policy.reproject_evidence(tool, receipt.get('result', {}),
                                                  view=model_view(tool) if tool else None,
                                                  projection=Projection())
        if not isinstance(evidence, dict):
            evidence = {"ok": False, "status": "policy_reprojected", "message": data_policy.REPROJECT_NOTE}
        text = data_policy.stable_json(evidence)
        end = min(len(text), offset + limit)
        return {'ok': True, 'result': {'context_ref': operation_id, 'tool': tool, 'historical': True,
                'policy_version': data_policy.POLICY_VERSION, 'sha256': stable_hash(evidence), 'offset': offset,
                'total_chars': len(text), 'content': text[offset:end], 'content_is_json_text': True,
                'next_offset': end if end < len(text) else None}}

    def _read_message_run(self, db, session_id, message_id):
        row = db.execute("SELECT body FROM runs WHERE session_id=? AND message_id=?", (session_id, message_id)).fetchone()
        return json.loads(row[0]) if row else None

    def find_message(self, session_id, message_id):
        self._migrate(session_id)
        with self.connection() as db:
            return self._read_message_run(db, session_id, message_id)

    @staticmethod
    def _turn_produced(run):
        """A valid draft or a preview is a real result; an invalid draft is never shown and stays replaceable."""
        artifacts = (run.get("response") or {}).get("artifacts") or {}
        return bool((artifacts.get("draft") or {}).get("valid")) or artifacts.get("preview") is not None

    def _replaced_turn(self, db, session_id, state, request):
        """Pre-turn conversation, events and artifacts of the replaced turn; it never enters later model context."""
        if request.resume_from_run_id:
            raise AgentError("AGENT_MESSAGE_EDIT_CONFLICT", "修改消息不能同时继续旧运行。", status_code=409)
        if not request.text.strip():
            raise AgentError("AGENT_MESSAGE_EDIT_EMPTY", "修改后的消息不能为空。", status_code=422, field="text")
        target = request.edit_of_message_id
        row = db.execute("SELECT seq, body FROM events WHERE session_id=? AND json_extract(body,'$.type')='user.message' AND json_extract(body,'$.id')=?",
                         (session_id, target)).fetchone()
        if row is None:
            raise AgentError("AGENT_MESSAGE_NOT_FOUND", "未找到要修改的消息。", status_code=404)
        last_user_seq = db.execute("SELECT MAX(seq) FROM events WHERE session_id=? AND json_extract(body,'$.type')='user.message'", (session_id,)).fetchone()[0]
        if row["seq"] != last_user_seq:
            raise AgentError("AGENT_MESSAGE_NOT_LAST", "只能修改最后一条尚未获得回复的消息。", status_code=409)
        replaced = self._read_message_run(db, session_id, target)
        if replaced is None or replaced.get("status") != "cancelled" or self._turn_produced(replaced):
            raise AgentError("AGENT_MESSAGE_NOT_EDITABLE", "这一轮仍未停止或已产生回复，不能再修改。", status_code=409)
        # The replaced run's own ancestry is the pre-turn state; a replacement keeps that ancestry so a
        # repeated edit never inherits an earlier stopped turn's text, tool results or compacted memory.
        if "parent_run_id" in replaced:
            ancestry = replaced.get("parent_run_id")
        else:
            # Runs written before this contract carry no ancestry: the previous turn is the nearest
            # earlier user message that is still visible, so its checkpoint is the pre-turn state.
            earlier = db.execute("""SELECT r.body FROM runs r JOIN events e
                ON e.session_id=r.session_id AND json_extract(e.body,'$.id')=r.message_id
                WHERE r.session_id=? AND json_extract(e.body,'$.type')='user.message' AND e.seq<?
                ORDER BY e.seq DESC LIMIT 1""", (session_id, row["seq"])).fetchone()
            ancestry = json.loads(earlier[0])["run_id"] if earlier else None
        previous = None
        if ancestry:
            try:
                previous = self._read_run(db, session_id, ancestry)
            except AgentError:
                previous = None
        base = copy.deepcopy((previous or {}).get("checkpoint") or {})
        if not base:
            base = self._truncate_replaced_turn(db, session_id, replaced, copy.deepcopy(state.get("conversation") or {}))
        base.setdefault("messages", [])
        # The replaced turn leaves the visible conversation entirely; only the stopped run's own
        # stop notice is ever attached to it, and its text stays in the audit record.
        replaced_rows = [(row["seq"], json.loads(row["body"]))]
        replaced_rows += [(item["seq"], json.loads(item["body"])) for item in db.execute(
            "SELECT seq, body FROM events WHERE session_id=? AND json_extract(body,'$.run_id')=? AND json_extract(body,'$.type')='assistant.message'",
            (session_id, replaced["run_id"])).fetchall()]
        # Runs from before this contract carry no snapshot; their per-turn state is handled separately.
        artifacts = copy.deepcopy(replaced["base_state"]) if isinstance(replaced.get("base_state"), dict) else None
        return {"checkpoint": base, "rows": replaced_rows, "ancestry": ancestry, "artifacts": artifacts, "run": replaced}

    def _truncate_replaced_turn(self, db, session_id, replaced, checkpoint):
        """Legacy sessions have no parent snapshot: drop the replaced turn from the stored conversation."""
        messages = list(checkpoint.get("messages") or [])
        for index in range(len(messages) - 1, -1, -1):
            if messages[index].get("role") == "user":
                del messages[index:]
                break
        checkpoint["messages"] = messages
        text = (replaced.get("request") or {}).get("text")
        checkpoint["pinned_user_messages"] = [message for message in checkpoint.get("pinned_user_messages") or []
                                              if not (message.get("role") == "user" and message.get("content") == text)][-4:]
        compacted = db.execute("SELECT 1 FROM events WHERE session_id=? AND json_extract(body,'$.run_id')=? AND json_extract(body,'$.type')='context.compacted' LIMIT 1",
                               (session_id, replaced["run_id"])).fetchone()
        if compacted:
            # The summary was rewritten during the replaced turn; its evidence cannot be split reliably.
            checkpoint["summary"] = ""
            checkpoint.pop("last_compaction_key", None)
        return checkpoint

    def _supersede_messages(self, db, session_id, rows, run, new_message_id):
        """Keep the replaced turn for audit and receipts while closing it to every future use."""
        for seq, body in rows:
            db.execute("UPDATE events SET body=? WHERE session_id=? AND seq=?", (stable_json(
                {**body, "type": f"{body['type']}.superseded", "superseded_by": new_message_id, "edited_at": utc_now()}), session_id, seq))
        run["superseded_by"] = new_message_id
        run["superseded_at"] = utc_now()
        self._write_run(db, run)

    def seen_facts(self, space_id, kind, values):
        if not values:
            return set()
        values = tuple(values)
        with self.connection() as db:
            return {row[0] for row in db.execute("SELECT value FROM progress_facts WHERE space_id=? AND kind=? AND value IN ("+",".join("?" for _ in values)+")", (space_id,kind,*values))}

    def accept(self, session_id, request, owner):
        self._migrate(session_id)
        fingerprint = stable_hash({"message_id": request.message_id, "text": request.text, "context_hash": context_hash(request.page_context),
                                   **({"page_snapshot_hash": stable_hash(request.page_snapshot.model_dump())} if request.page_snapshot else {}),
                                   **({"resume_from_run_id": request.resume_from_run_id} if request.resume_from_run_id else {}),
                                   **({"edit_of_message_id": request.edit_of_message_id} if request.edit_of_message_id else {})})
        with self.connection(write=True) as db:
            state = self._read_state(db, session_id)
            existing = db.execute("SELECT body FROM runs WHERE session_id=? AND message_id=?", (session_id, request.message_id)).fetchone()
            if existing:
                run = json.loads(existing[0])
                if run["request_hash"] != fingerprint:
                    raise AgentError("REVISION_CONFLICT", "同一消息标识不能提交不同内容。", status_code=409)
                return run, True
            assert_mutable(state)
            if request.expected_session_revision != state["session_revision"]:
                raise AgentError("REVISION_CONFLICT", "会话已更新，请刷新后重试。", status_code=409, field="expected_session_revision")
            from .scopes import scope_for_page
            if scope_for_page(request.page_context.page) != state["scope"]:
                raise AgentError("AGENT_SCOPE_PAGE_MISMATCH", "页面与会话作用域不匹配。", status_code=409)
            parent = self._read_run(db, session_id, request.resume_from_run_id) if request.resume_from_run_id else None
            if parent and parent["status"] in ACTIVE_STATUSES:
                raise AgentError("AGENT_SESSION_BUSY", "当前任务仍在处理，请先停止。", status_code=409)
            if parent and parent.get("superseded_by"):
                raise AgentError("AGENT_RUN_SUPERSEDED", "这一轮已被修改，不能再继续原运行。", status_code=409)
            if parent and parent["run_id"] != state.get("last_run_id"):
                raise AgentError("AGENT_RESUME_STALE", "已有更新的对话，请刷新后继续最新一轮。", status_code=409)
            replaced = self._replaced_turn(db, session_id, state, request) if request.edit_of_message_id else None
            if replaced is not None:
                # Tool rollback cannot undo separate human approvals or commit-preview proposals.
                independent_proposals = [item for item in (state.get("memory_proposals") or [])
                    if item.get("proposal_id") in (state.get("memory") or {})
                    or (item.get("kind") == "indicator_definition" and item.get("confirmation_id"))]
                # Write the pre-turn state now: a later failed run must not fall back to the replaced turn.
                state["conversation"] = replaced["checkpoint"]
                state["memory_proposals"] = [item for item in (state.get("memory_proposals") or [])
                                             if item.get("source_message_id") != request.edit_of_message_id]
                if replaced["artifacts"] is not None:
                    for key in BASE_STATE_KEYS:
                        value = replaced["artifacts"].get(key)
                        if value is None:
                            state.pop(key, None)
                        else:
                            state[key] = value
                else:
                    # Old runs have no base snapshot. Applied receipts distinguish a new
                    # proposal from reusing one produced by an earlier, still-valid turn.
                    outputs = [dict(row) for row in db.execute("""SELECT t.run_id, t.body,
                        json_extract(t.body,'$.tool') AS tool,
                        COALESCE(json_extract(t.body,'$.result.result.proposal_id'),
                                 json_extract(t.body,'$.result.result.id')) AS output_id
                        FROM tool_calls t JOIN runs r ON r.id=t.run_id
                        WHERE r.session_id=? AND json_extract(r.body,'$.superseded_by') IS NULL
                        AND json_extract(t.body,'$.tool') IN ('memory.propose','task.plan')
                        AND json_extract(t.body,'$.status')='completed' AND json_extract(t.body,'$.applied')=1""",
                        (session_id,))]
                    for row in outputs:
                        receipt = json.loads(row['body'])
                        # Model views may omit a large plan. Internal ownership uses the
                        # same validated arguments and identity as its original creation.
                        if (row['tool'] == 'task.plan' and isinstance(receipt.get('validated_arguments'), dict)
                                and (receipt.get('result') or {}).get('ok') is not False):
                            row['output_id'] = task_plan_id(receipt['validated_arguments'])
                    for tool, key, identifier in (('memory.propose', 'memory_proposals', 'proposal_id'),
                                                  ('task.plan', 'task_plans', 'id')):
                        own = {row['output_id'] for row in outputs if row['tool'] == tool and row['run_id'] == replaced['run']['run_id']}
                        earlier = {row['output_id'] for row in outputs if row['tool'] == tool and row['run_id'] != replaced['run']['run_id']}
                        if key in state:
                            state[key] = [item for item in (state[key] or []) if item.get(identifier) not in own - earlier]
                    # Runs from before this contract have no snapshot. The gate already proved this turn
                    # holds no valid result, so an invalid current draft can only come from it while the
                    # earlier valid draft and preview must survive the replacement.
                    draft = state.get("draft")
                    if isinstance(draft, dict) and not draft.get("valid"):
                        last_valid = state.get("last_valid_draft")
                        if isinstance(last_valid, dict):
                            state["draft"] = copy.deepcopy(last_valid)
                        else:
                            state.pop("draft", None)
                proposals = {item['proposal_id']: item for item in (state.get('memory_proposals') or [])}
                proposals.update({item['proposal_id']: item for item in independent_proposals})
                state['memory_proposals'] = list(proposals.values())
            changed = apply_context(state, request.page_context)
            state["pending_confirmation"] = None
            state["session_revision"] += 1
            run = {"run_id": f"run-{uuid.uuid4().hex}", "session_id": session_id, "message_id": request.message_id,
                   "request_hash": fingerprint, "request": request.model_dump(), "status": "queued", "phase": "thinking",
                   "stop_reason": None, "run_revision": 0, "owner_epoch": 1, **owner,
                   "created_at": utc_now(), "session_revision": state["session_revision"], "checkpoint": {},
                   "parent_run_id": replaced["ancestry"] if replaced is not None else state.get("last_run_id"),
                   "base_state": {key: copy.deepcopy(state.get(key)) for key in BASE_STATE_KEYS},
                   "edit_of_message_id": request.edit_of_message_id,
                   "detector_state": copy.deepcopy((parent or {}).get("detector_state", {})) if not changed else {},
                   "usage": {"tool_calls": 0, "rounds": 0, "model_steps": 0, "prompt_tokens": None, "completion_tokens": None},
                   "resume_from_run_id": request.resume_from_run_id}
            run["progress_space_id"] = (parent.get("progress_space_id", parent["run_id"]) if parent and not changed else run["run_id"])
            state["active_run_id"] = state["last_run_id"] = run["run_id"]
            append_event(state, {"type": "user.message", "speaker": "user", "id": request.message_id, "message_id": request.message_id,
                                 "run_id": run["run_id"], "text": request.text,
                                 **({"edit_of": request.edit_of_message_id} if request.edit_of_message_id else {})})
            if replaced is not None:
                self._supersede_messages(db, session_id, replaced["rows"], replaced["run"], request.message_id)
            self._write_run(db, run)
            self._write_state(db, state)
            return run, False

    def _transaction(self):
        current = getattr(self._local, "db", None)
        return nullcontext(current) if current is not None else self.connection(write=True)

    def _owned_run(self, db, run):
        current = self._read_run(db, run["session_id"], run["run_id"])
        state = self._read_state(db, run["session_id"])
        if current["owner_epoch"] != run["owner_epoch"]:
            raise AgentError("AGENT_RUN_STALE", "运行执行权已失效。", status_code=409)
        if current["status"] in ACTIVE_STATUSES and state.get("active_run_id") != run["run_id"]:
            raise AgentError("AGENT_RUN_STALE", "运行已不属于当前会话执行。", status_code=409)
        return current, state

    @staticmethod
    def _run_metadata(current, candidate):
        # Identity, cancellation, execution fences and content are store-owned.
        result = copy.deepcopy(current)
        result.update({key: copy.deepcopy(candidate[key]) for key in RUN_METADATA_FIELDS if key in candidate})
        return result

    def _save_transition(self, db, current, run, state, events=()):
        run.update(run_revision=current["run_revision"] + 1, updated_at=utc_now())
        self._write_run(db, run)
        for event in events:
            append_event(state, {**event, "run_id": run["run_id"], "message_id": run["message_id"]})
        if run["status"] not in ACTIVE_STATUSES:
            state["active_run_id"] = None
            if run.get("execution_blocked_by"):
                state["execution_blocked_by"] = run["execution_blocked_by"]
        self._write_state(db, state)

    def admit(self, run, *, events=(), tool_receipt=None):
        """Record execution facts, never the caller's tentative conversation or artifacts."""
        return self._checkpoint(run, content=False, events=events, tool_receipt=tool_receipt)

    def checkpoint(self, run, *, events=(), patch=None, tool_receipt=None, new_facts=()):
        """Commit candidate content and its effects, or discard all of them together."""
        return self._checkpoint(run, content=True, events=events, patch=patch,
                                tool_receipt=tool_receipt, new_facts=new_facts)

    def _checkpoint(self, run, *, content, events=(), patch=None, tool_receipt=None, new_facts=()):
        with self._transaction() as db:
            current, state = self._owned_run(db, run)
            if current["status"] not in ACTIVE_STATUSES:
                return False
            if current["status"] == "running" and run.get("status") == "queued":
                return False
            allowed_events = CONTENT_EVENTS if content else CONTROL_EVENTS
            start_count = sum(event.get("type") == "run.started" for event in events)
            starting = bool(start_count)
            if (run.get("status") not in ACTIVE_STATUSES
                    or start_count > 1
                    or any(event.get("type") not in allowed_events for event in events)
                    or (not content and any(key.startswith("_") for event in events for key in event))
                    or (patch and any(key not in (*BASE_STATE_KEYS, "pending_confirmation") for key in patch))
                    or (tool_receipt and ((tool_receipt.get("status") in {"started", "unknown"}) == content))):
                raise AgentError("AGENT_RUN_TRANSITION_INVALID", "运行状态必须通过对应的提交入口更新。", status_code=409)
            accepted = not current.get("cancel_requested")
            if ((starting and accepted and current["status"] != "queued")
                    or (not content and current["status"] == "queued"
                        and (tool_receipt or any(event.get("type") in {"data.admitted", "model.returned"} for event in events)))):
                raise AgentError("AGENT_RUN_TRANSITION_INVALID", "运行必须先提交起始上下文，才能开始执行。", status_code=409)
            next_run = self._run_metadata(current, run)
            if accepted:
                if starting:
                    next_run["status"] = "running"
                if content:
                    next_run.update({key: copy.deepcopy(run[key]) for key in RUN_CONTENT_FIELDS if key in run})
                    if patch:
                        state.update(copy.deepcopy(patch))
                    db.executemany("INSERT OR IGNORE INTO progress_facts VALUES (?,?,?)",
                        [(current.get("progress_space_id", current["run_id"]), kind, value) for kind, value in new_facts])
            if tool_receipt is not None:
                prior_row = db.execute("SELECT body FROM tool_calls WHERE run_id=? AND model_step=? AND call_id=?",
                    (run["run_id"], tool_receipt["model_step"], tool_receipt["call_id"])).fetchone()
                prior = json.loads(prior_row[0]) if prior_row else None
                tool_status = tool_receipt.get("status")
                operation_id = tool_receipt.get("operation_id")
                physical = tool_status in {"started", "unknown", "completed"}
                valid_identity = (isinstance(operation_id, str) and bool(operation_id)
                                  and isinstance(tool_receipt.get("tool"), str) and bool(tool_receipt["tool"]))
                matching = (prior is not None
                    and all(prior.get(key) == tool_receipt.get(key) for key in ("operation_id", "tool"))
                    and stable_json(prior.get("arguments")) == stable_json(tool_receipt.get("arguments")))
                prior_status = (prior.get("execution_status") or prior.get("status")) if prior else None
                open_receipt = prior_status in {"started", "unknown"}
                if (tool_status not in {"started", "unknown", "completed", "blocked", "not_executed"}
                        or (physical and not valid_identity)
                        or (tool_status == "started" and prior is not None)
                        or (tool_status in {"unknown", "completed"} and not (matching and open_receipt))
                        or (not physical and (operation_id is not None or prior is not None))):
                    raise AgentError("AGENT_RUN_TRANSITION_INVALID", "工具回执必须延续同一个已开始操作，已结束回执不能重写。", status_code=409)
                receipt = {**tool_receipt, "applied": bool(accepted and content and tool_receipt.get("status") == "completed")}
                receipt.pop("execution_status", None)
                if accepted and tool_receipt.get("status") == "started" and operation_id:
                    if current.get("execution_blocked_by") not in (None, operation_id):
                        raise AgentError("AGENT_SESSION_BUSY", "此前工具尚未退出，不能叠加操作。", status_code=409)
                    # Protect the operation before dispatch, not only after timeout.
                    next_run["execution_blocked_by"] = state["execution_blocked_by"] = operation_id
                elif content and tool_receipt.get("status") == "completed" and operation_id:
                    self._release_operation(state, next_run, operation_id)
                # Unknown receipts preserve the durable operation fence as-is; a
                # callback may already have released it after the real tool exited.
                if not accepted:
                    # Content admission and physical completion are separate facts.
                    receipt["execution_status"] = "not_executed" if tool_status == "started" else tool_status
                    receipt["status"] = "discarded"
                db.execute("INSERT INTO tool_calls VALUES (?,?,?,?) ON CONFLICT(run_id,model_step,call_id) DO UPDATE SET body=excluded.body",
                    (run["run_id"], receipt["model_step"], receipt["call_id"], stable_json(receipt)))
                if not accepted and content:
                    self._discarded_tool_message(next_run, receipt)
            committed_events = []
            for event in events:
                if content and not accepted:
                    if event.get("type") != "tool.completed":
                        continue
                    event = {key: value for key, value in event.items() if not key.startswith("_")}
                    event = {**event, "data": {**event.get("data", {}), "status": "discarded", "applied": False}}
                elif event.get("type") in {"run.started", "run.phase", "run.recovering"}:
                    event = {**event, "data": {**event.get("data", {}), "status": next_run["status"]}}
                committed_events.append(event)
            self._save_transition(db, current, next_run, state, committed_events)
        # Only control acknowledgements return to the live candidate. Replacing its
        # content here would invalidate the runner's working-set references.
        for key in ("status", "run_revision", "updated_at", "execution_blocked_by", "cancel_requested", "context_change"):
            if key in next_run:
                run[key] = copy.deepcopy(next_run[key])
        return accepted

    @staticmethod
    def _discarded_tool_message(run, receipt):
        from . import data_policy
        messages = run.get("checkpoint", {}).get("messages", [])
        call_id = receipt["call_id"]
        # Only complete a call from the committed model plan, never a rejected plan.
        plan_index = next((index for index in range(len(messages) - 1, -1, -1)
                           if any(call.get("id") == call_id for call in messages[index].get("tool_calls", []))), None)
        if plan_index is None:
            return
        tool = str(receipt.get("tool") or "unknown")
        notice = {"role": "tool", "tool_call_id": call_id, "content": stable_json(data_policy.seal_notice(
            data_policy.notice(source=tool, status="discarded", message="停止后返回的结果未应用。"), tool))}
        for index in range(plan_index + 1, len(messages)):
            message = messages[index]
            if message.get("role") == "tool" and message.get("tool_call_id") == call_id:
                messages[index] = notice
                break
        else:
            messages.append(notice)

    def finish(self, run, text, *, status="completed", reason=None, reasoning_content=None):
        """Own the final response, recoverable content and terminal state in one transaction."""
        from . import data_policy
        from .context import close_pending
        if status not in TERMINAL_STATUSES:
            raise AgentError("AGENT_RUN_TRANSITION_INVALID", "无效的运行终态。", status_code=409)
        with self._transaction() as db:
            current, state = self._owned_run(db, run)
            if current["status"] not in ACTIVE_STATUSES:
                return current
            result = self._run_metadata(current, run)
            discard = bool(current.get("cancel_requested") or run.get("error"))
            if not result.get("checkpoint"):
                result["checkpoint"] = copy.deepcopy(state.get("conversation") or {})
                user_text = current["request"].get("text")
                if user_text and not data_policy.user_text_violation(user_text):
                    result["checkpoint"].setdefault("messages", []).append({"role": "user", "content": user_text})
            if current.get("cancel_requested") and status not in {"failed", "interrupted"}:
                reason = current.get("stop_reason") or "user_cancelled"
                status = "paused" if reason == "context_changed" else "cancelled"
                text = stop_message(current)
            messages = result["checkpoint"].setdefault("messages", [])
            close_pending(messages)
            if not messages or messages[-1].get("role") != "assistant" or messages[-1].get("content") != text:
                message = {"role": "assistant", "content": text}
                if not discard and reasoning_content is not None:
                    message["reasoning_content"] = reasoning_content
                messages.append(message)
            result.update(status=status, stop_reason=reason)
            if run.get("error"):
                result["error"] = copy.deepcopy(run["error"])
            draft = public_draft(state)
            artifacts = self._artifacts(db, run["session_id"], [run["run_id"]]).get(run["run_id"], {})
            response = {"session_id": current["session_id"], "message_id": current["message_id"], "request_id": current["run_id"],
                "run_id": current["run_id"], "session_revision": state["session_revision"], "reply": {"role": "assistant", "text": text},
                "draft": draft, "preview": state.get("preview"), "artifacts": artifacts, "stop_reason": reason,
                "usage": result["usage"], "tool_trace": result.get("tool_trace", []), "catalog_version": result.get("catalog_version"),
                "context_hash": state.get("context_hash"), "replayed": False,
                "artifact_status": "previewed" if state.get("preview") else "draft_validated" if draft and draft.get("valid") else "draft_invalid" if draft else "none",
                "task_state": result["checkpoint"].get("task_state"), "memory_sources": result["checkpoint"].get("memory_sources", [])}
            result["response"] = response
            state["conversation"] = copy.deepcopy(result["checkpoint"])
            events = [{"type": "assistant.message", "speaker": "assistant", "id": run["run_id"]+"-reply", "text": text, "artifacts": artifacts},
                {"type": "run." + (status if status in {"completed", "paused", "cancelled"} else "failed"),
                 "data": {"status": status, "phase": result["phase"], "stop_reason": reason}}]
            self._save_transition(db, current, result, state, events)
            return result

    def interrupt(self, run):
        """Stop ownership without promoting pending work or inventing an assistant reply."""
        with self._transaction() as db:
            current, state = self._owned_run(db, run)
            if current["status"] not in ACTIVE_STATUSES:
                return current
            result = self._run_metadata(current, run)
            result.update(status="interrupted", stop_reason="process_interrupted")
            self._save_transition(db, current, result, state)
            return result

    def cancel(self, session_id, run_id, *, reason="user_cancelled", context=None, source=None):
        with self.connection(write=True) as db:
            run = self._read_run(db, session_id, run_id)
            state = self._read_state(db, session_id)
            if context is not None:
                frozen = run["request"]["page_context"]
                if context.page_instance_id != frozen["page_instance_id"] or context.context_revision < frozen["context_revision"]:
                    raise AgentError("AGENT_CONTEXT_CHANGED", "页面实例或上下文版本不匹配。", status_code=409)
                incoming = context.model_dump()
                if context_hash(PageContext.model_validate(frozen)) == context_hash(context):
                    return run
            if run["status"] in ACTIVE_STATUSES:
                if reason == "context_changed" and not run.get("cancel_requested"):
                    fields = []
                    if context is not None:
                        fields = [key for key in ("page", "context_revision", "view_state") if frozen.get(key) != incoming.get(key)]
                        fields += ["calculation." + key for key in ("context_kind", "targets", "period", "as_of", "run_id")
                                   if frozen["calculation"].get(key) != incoming["calculation"].get(key)]
                    run["context_change"] = {"source": "page" if context is not None else source or "unknown", "fields": fields}
                    append_event(state, {"type": "context.invalidated", "run_id": run_id, "data": run["context_change"],
                                         **({"_page_context": incoming} if context is not None else {})})
                run.update(cancel_requested=True, status="stopping", stop_reason=reason, run_revision=run["run_revision"]+1)
                self._write_run(db, run)
                append_event(state, {"type": "run.phase", "run_id": run_id, "data": {"status": "stopping", "phase": run["phase"]}})
                self._write_state(db, state)
            return run

    @staticmethod
    def _release_operation(state, run, operation_id):
        if run.get("execution_blocked_by") != operation_id:
            return False
        run["execution_blocked_by"] = None
        if state.get("execution_blocked_by") == operation_id:
            state.pop("execution_blocked_by", None)
        return True

    def release_quarantine(self, session_id, operation_id):
        with self.locked(session_id) as state:
            run_id = state.get("last_run_id")
            if run_id:
                run = self._read_run(self._local.db, session_id, run_id)
                if self._release_operation(state, run, operation_id):
                    run.update(run_revision=run["run_revision"]+1, updated_at=utc_now())
                    self._write_run(self._local.db, run)
                    append_event(state, {"type": "run.phase", "run_id": run_id, "data": {"status": run["status"], "execution_blocked_by": None}})
                    self.write(state)

    def unfinished_runs(self):
        with self.connection() as db:
            return [json.loads(row[0]) for row in db.execute("SELECT body FROM runs WHERE status IN ('queued','running','stopping') OR json_extract(body,'$.execution_blocked_by') IS NOT NULL")]

    def recover(self, run, *, reason="process_interrupted"):
        with self.connection(write=True) as db:
            current = self._read_run(db, run["session_id"], run["run_id"])
            state = self._read_state(db, run["session_id"])
            # The candidate was read before acquiring this transaction (possibly by
            # another process). A changed or already recovered run is not ours to undo.
            if (any(current.get(key) != run.get(key) for key in ("owner_epoch", "run_revision"))
                    or state.get("last_run_id") != run["run_id"]):
                return False
            active = current["status"] in ACTIVE_STATUSES
            operation_id = current.get("execution_blocked_by")
            if (active and state.get("active_run_id") != run["run_id"]) or (not active and not operation_id):
                return False
            current.update(owner_epoch=current.get("owner_epoch", 0)+1, run_revision=current.get("run_revision", 0)+1,
                           execution_blocked_by=None, updated_at=utc_now())
            if active:
                current.update(status="interrupted", stop_reason=reason)
            if state.get("active_run_id") == run["run_id"]:
                state["active_run_id"] = None
            if operation_id and state.get("execution_blocked_by") == operation_id:
                state.pop("execution_blocked_by", None)
            self._write_run(db, current)
            append_event(state, {"type": "run.failed" if active else "run.phase", "run_id": run["run_id"],
                                 "data": {"status": current["status"], "stop_reason": current.get("stop_reason"), "execution_blocked_by": None}})
            self._write_state(db, state)
            return True

    def commit_receipt(self, db, session_id, request_id):
        row = db.execute("SELECT body FROM commit_intents WHERE session_id=? AND request_id=?", (session_id, request_id)).fetchone()
        return json.loads(row[0]) if row else None

    def definition_commit(self, db, session_id, digest):
        # A reload can create a new request ID; the full saved definition is the
        # durable write identity within this session. Older completed receipts
        # already carry the hash in their response.
        row = db.execute("""SELECT body FROM commit_intents WHERE session_id=? AND
            COALESCE(json_extract(body,'$.definition_hash'), json_extract(body,'$.response.definition_hash'))=?
            ORDER BY rowid LIMIT 1""", (session_id, digest)).fetchone()
        return json.loads(row[0]) if row else None

    def put_commit(self, db, session_id, request_id, body):
        db.execute("INSERT INTO commit_intents VALUES (?,?,?) ON CONFLICT(session_id,request_id) DO UPDATE SET body=excluded.body", (session_id, request_id, stable_json(body)))


def assert_mutable(state):
    if state.get("active_run_id") or state.get("execution_blocked_by"):
        raise AgentError("AGENT_SESSION_BUSY", "当前任务仍在处理或停止中，请稍后再操作。", status_code=409)
    if state.get("committing_request_id"):
        raise AgentError("AGENT_COMMIT_UNCERTAIN", "保存正在处理或结果待核对，请勿重复保存。", status_code=409)


def apply_context(state, page_context):
    incoming = context_hash(page_context)
    changed = state.get("context_hash") != incoming
    state.update(page_context=page_context.model_dump(), context_hash=incoming)
    if changed:
        state["pending_confirmation"] = None
        state.pop("product_candidates", None)
        state.pop("preview", None)
        if isinstance(state.get("draft"), dict):
            state["draft"].update(compile_token=None, stale=True, context_hash=incoming)
    return changed


def store_draft(state, *, definition, validation, compile_token):
    digest = definition_hash(definition)
    current = state.get("draft") or {}
    changed = current.get("definition_hash") != digest
    revision = max(1, int(current.get("draft_revision", 0)) + int(changed))
    draft = {"draft_revision": revision, "definition": definition, "definition_hash": digest,
             "valid": bool(validation.get("valid")), "context_hash": state.get("context_hash"),
             "result_kind": definition.get("result_kind", "scalar"), "display_latex": validation.get("display_latex"),
             "editable_latex": validation.get("editable_latex") or definition.get("expression"),
             "dependencies": list(validation.get("dependencies") or [])[:64], "diagnostics": list(validation.get("diagnostics") or [])[:8],
             "compile_token": compile_token if validation.get("valid") else None, "stale": False, "updated_at": utc_now()}
    state["draft"] = draft
    if draft["valid"]:
        state["last_valid_draft"] = copy.deepcopy(draft)
    if changed:
        state["pending_confirmation"] = None
        state.pop("preview", None)
    return draft


def public_draft(state):
    draft = state.get("draft")
    return {k: v for k, v in draft.items() if k != "compile_token"} if isinstance(draft, dict) else None


def public_run(run):
    result = {k: run.get(k) for k in ("run_id", "session_id", "message_id", "status", "phase", "stop_reason", "run_revision", "session_revision", "usage", "response", "created_at", "execution_blocked_by", "error")}
    if run["status"] in ACTIVE_STATUSES:
        result["execution_blocked_by"] = None
    result["events_url"] = f"/api/agent/sessions/{run['session_id']}/events"
    return result


def public_state(state, *, event_from=1):
    from .memory import memory_identity
    confirmation = state.get("pending_confirmation")
    return {"session_id": state["session_id"], "session_revision": state["session_revision"], "scope": state["scope"],
            "legacy_history_incomplete": bool(state.get("legacy_history_incomplete")),
            "page_context": state.get("page_context"), "context_hash": state.get("context_hash"), "next_event_seq": state.get("next_event_seq", 1),
            "state": "turn_running" if state.get("active_run_id") else "confirmation_frozen" if confirmation else "draft_ready" if (state.get("draft") or {}).get("valid") else "idle",
            "draft": public_draft(state), "preview": state.get("preview"), "execution_blocked_by": state.get("execution_blocked_by") if not state.get("active_run_id") else None,
            "pending_confirmation": {k: confirmation.get(k) for k in ("confirmation_id", "definition_hash", "draft_revision", "created_at")} if confirmation else None,
            "memory_proposals": [{**item, **memory_identity(item)} for item in (state.get("memory_proposals") or [])[-10:]],
            "task_state": (state.get("conversation") or {}).get("task_state"),
            "memory_sources": (state.get("conversation") or {}).get("memory_sources", []),
            "events": [{k: v for k, v in event.items() if not k.startswith('_')} for event in state.get("events", []) if event["seq"] >= event_from]}

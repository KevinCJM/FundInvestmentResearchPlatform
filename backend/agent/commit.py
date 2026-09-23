"""Human-triggered commit path: freeze one confirmation, single write entry.

``commit-preview`` validates and freezes the definition hash, context hash and
draft revision; ``commit`` only succeeds while that snapshot is still current
and delegates the actual write to the existing indicator service.
"""

from __future__ import annotations

import uuid
from typing import Any

from custom_indicators.errors import IndicatorDomainError

from .catalog import build_catalog
from .contracts import AgentError, CommitPreviewRequest, CommitRequest
from .memory import memory_identity
from .sessions import (
    append_event,
    apply_context,
    definition_hash,
    utc_now,
    stable_hash,
    assert_mutable,
)


def _target_in_context(page_context, target) -> bool:
    calculation = page_context.calculation
    if calculation.context_kind != "single_product":
        return False
    return any(
        item.kind == target.kind and item.product_id == target.product_id
        for item in calculation.targets
    )


def _same_definition(left, right):
    """Match all JSON fields while allowing browsers to round-trip 0.0 as 0."""
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_same_definition(left[key], right[key]) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_same_definition(a, b) for a, b in zip(left, right))
    same_type = type(left) is type(right) or (type(left) in (int, float) and type(right) in (int, float))
    return same_type and left == right


def preview(*, store: Any, session_id: str, request: CommitPreviewRequest, service: Any) -> dict[str, Any]:
    state = store.read(session_id)
    assert_mutable(state)
    base_revision = state["session_revision"]
    draft = state.get("draft") or {}
    if not draft.get("valid"):
        raise AgentError("AGENT_DRAFT_REQUIRED", "请先校验通过指标草稿再提交预览。", status_code=409)
    if (request.draft_revision != draft.get("draft_revision")
            or not _same_definition(request.definition, draft.get("definition"))):
        raise AgentError("REVISION_CONFLICT", "草稿已更新，请刷新后预览。", status_code=409)
    if request.page_context.context_kind != "single_product":
        raise AgentError("AGENT_TOOL_DOMAIN_MISMATCH", "指标创作仅支持单产品计算域。", status_code=409)
    if request.target is not None and not _target_in_context(request.page_context, request.target):
        raise AgentError("AGENT_CONTEXT_CHANGED", "提交目标不在当前页面上下文内。", status_code=409)
    definition = dict(draft["definition"])
    if definition.get("context_kind", "single_product") != "single_product":
        raise AgentError("AGENT_TOOL_DOMAIN_MISMATCH", "这里只保存单产品指标。", status_code=409)
    name = str(definition.get("name") or "").strip()
    if not name:
        raise AgentError("VALIDATION_ERROR", "指标名称不能为空。", status_code=422, field="definition.name")
    validation = service.validate(definition)
    if not validation.get("valid"):
        raise AgentError("VALIDATION_ERROR", "指标定义未通过提交校验。", status_code=422, diagnostics=validation.get("diagnostics"))
    conflict = _name_conflict(service, name)
    catalog_version = str(build_catalog(service).get("version") or "")
    with store.locked(session_id) as current:
        assert_mutable(current)
        if current["session_revision"] != base_revision:
            raise AgentError("REVISION_CONFLICT", "预览期间会话已改变，请重新预览。", status_code=409)
        apply_context(current, request.page_context)
        confirmation = {"confirmation_id": uuid.uuid4().hex, "definition": definition, "definition_hash": definition_hash(definition),
                        "context_hash": current["context_hash"], "draft_revision": draft["draft_revision"],
                        "target": request.target.model_dump() if request.target else None, "page": request.page_context.page, "created_at": utc_now()}
        current["pending_confirmation"] = confirmation
        current["session_revision"] += 1
        proposal = {"proposal_id": f"mem-{uuid.uuid4().hex}", "kind": "indicator_definition", "summary": f"记住指标草稿「{name}」的定义与上下文版本。",
                    "confirmation_id": confirmation['confirmation_id'],
                    "definition_hash": confirmation["definition_hash"], "scope": current["scope"], "status": "pending", "created_at": utc_now()}
        proposal.update(memory_identity(proposal))
        current["memory_proposals"] = (current.get("memory_proposals", []) + [proposal])[-10:]
        append_event(current, {"type": "commit_preview", "speaker": "human", "status": "ok"})
        store.write(current)
        return {"session_id": session_id, "session_revision": current["session_revision"], **{k:v for k,v in confirmation.items() if k not in {"created_at", "target", "page"}},
                "catalog_version": catalog_version, "preview_status": "valid", "memory_proposal_id": proposal["proposal_id"],
                "impact": {"action": "create", "name": name, "context_kind": definition.get("context_kind", "single_product"), "target": confirmation["target"],
                           "page": request.page_context.page, "name_conflict_indicator_id": conflict, "display_latex": validation.get("display_latex")}}


def _name_conflict(service: Any, name: str) -> str | None:
    try:
        listing = service.list_indicators()
    except IndicatorDomainError:
        return None
    for item in listing.get("items", []):
        if str(item.get("name")) == name and item.get("source") == "custom":
            return str(item.get("id"))
    return None



def commit(*, store: Any, session_id: str, request: CommitRequest, service: Any) -> dict[str, Any]:
    fingerprint = stable_hash(request.model_dump())
    with store.locked(session_id) as state:
        db = store._local.db
        existing = store.commit_receipt(db, session_id, request.request_id)
        if existing:
            if existing.get("request_hash") and existing["request_hash"] != fingerprint:
                raise AgentError("REVISION_CONFLICT", "同一提交标识不能提交不同内容。", status_code=409)
            if existing["state"] == "completed":
                return {**existing["response"], "replayed": True}
            raise AgentError("AGENT_COMMIT_UNCERTAIN", "该保存已有执行记录，结果需人工核对，禁止重复创建。", status_code=409)
        assert_mutable(state)
        if not request.confirmed:
            raise AgentError("AGENT_CONFIRMATION_REQUIRED", "指标写入必须由人类明确确认。", status_code=422)
        confirmation = state.get("pending_confirmation") or {}
        draft = state.get("draft") or {}
        if (confirmation.get("confirmation_id") != request.confirmation_id or confirmation.get("definition_hash") != request.definition_hash
            or confirmation.get("draft_revision") != request.draft_revision or confirmation.get("context_hash") != state["context_hash"]
            or not draft.get("valid") or draft.get("draft_revision") != request.draft_revision
            or draft.get("definition_hash") != request.definition_hash):
            raise AgentError("AGENT_CONFIRMATION_STALE", "确认快照已失效，请重新预览。", status_code=409)
        intent = {"state": "started", "request_hash": fingerprint, "confirmation_id": request.confirmation_id,
                  "definition_hash": request.definition_hash, "at": utc_now()}
        prior = store.definition_commit(db, session_id, request.definition_hash)
        if prior:
            if prior["state"] != "completed":
                raise AgentError("AGENT_COMMIT_UNCERTAIN", "该定义已有保存记录，结果需人工核对，禁止重复创建。", status_code=409)
            state["session_revision"] += 1
            response = {**prior["response"], "request_id": request.request_id,
                        "confirmation_id": request.confirmation_id, "session_revision": state["session_revision"], "replayed": True}
            store.put_commit(db, session_id, request.request_id, {**intent, "state": "completed", "response": response})
            state["pending_confirmation"] = None
            store.write(state)
            return response
        store.put_commit(db, session_id, request.request_id, intent)
        state["committing_request_id"] = request.request_id
        store.write(state)
        definition = dict(confirmation["definition"])
    # Business storage and agent storage are separate commits. Never hold the agent DB lock here.
    try:
        created = service.create_indicator(definition)
    except IndicatorDomainError:
        with store.locked(session_id) as state:
            store.put_commit(store._local.db, session_id, request.request_id, {**intent, "state": "failed"})
            state.pop("committing_request_id", None)
            store.write(state)
        raise
    except Exception:
        with store.locked(session_id) as state:
            store.put_commit(store._local.db, session_id, request.request_id, {**intent, "state": "uncertain"})
        raise AgentError("AGENT_COMMIT_UNCERTAIN", "指标写入结果不确定，请人工核对目录；不会自动重复创建。", status_code=409) from None
    try:
        with store.locked(session_id) as state:
            state.pop("committing_request_id", None)
            state["pending_confirmation"] = None
            state["session_revision"] += 1
            response = {"session_id": session_id, "request_id": request.request_id, "confirmation_id": request.confirmation_id,
                        "indicator_id": created.get("id"), "revision": created.get("revision"), "name": created.get("name"),
                        "definition_hash": request.definition_hash, "session_revision": state["session_revision"],
                        "refresh": {"catalog_required": True}, "replayed": False}
            store.put_commit(store._local.db, session_id, request.request_id, {**intent, "state": "completed", "response": response})
            append_event(state, {"type": "commit", "speaker": "human", "status": "ok"})
            store.write(state)
            return response
    except Exception:
        raise AgentError("AGENT_COMMIT_UNCERTAIN", "指标可能已保存，但回执未确认，请人工核对目录。", status_code=409) from None

"""Versioned, signed admission for everything the agent sends to a model.

Two checks share one policy:

* **Source projection** (:mod:`agent.views`) — every registered tool declares a
  strict, nested allowlist; only its own trusted contract fields survive, so an
  arbitrary dict, an unknown number or a serialized table cannot become model
  context.
* **Send guard** — every outbound request (primary, compaction, no-progress
  finalization, retries, resumed history) is re-verified here: tool receipts
  must carry a server HMAC signature for their own tool, the system prompt must
  carry a server seal, user text must not be a pasted structured table, and no
  message may contain a raw-sequence shape.

A JSON ``admission`` marker proves nothing by itself: the signature is computed
with a key that never leaves this process/data directory.  Rejected payloads
fail closed with value-free reasons; labels/paths use only registry-owned names.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import re
import threading
import tempfile
from typing import Any, Callable, Iterable, Optional

from .contracts import AgentError

POLICY_VERSION = "data-admission/3"
MAX_DEPTH = 12
MAX_LIST_ITEMS = 200
MAX_NUMERIC_RUN = 8
MAX_TEXT_RUN = 64
MAX_KEYS = 200
MAX_TOOL_RESULT_CHARS = 8000

OMIT_NOTE = "完整数据仅保留在后端结果与页面；需要统计或解释请调用已注册的计算工具。"
REPROJECT_NOTE = "这条历史证据没有通过当前数据准入核验，已省略原文；需要时请重新查询或重新计算。"
STRUCTURED_OMIT_NOTE = "消息中的结构化数据未通过数据准入核验，已在交给模型前替换为说明。"
SUMMARY_OMIT_NOTE = "旧摘要未通过当前策略核验，已省略；请以用户原话和当前工具证据为准。"

SYSTEM_PURPOSES = frozenset({"primary", "compaction", "finalization"})
SUMMARY_PURPOSE = "summary"
# Top-level keys the server-composed system prompt may embed as JSON fragments
# (validated calculation context, session draft, page-snapshot metadata).
SYSTEM_TOP_KEYS = frozenset({
    "context_kind", "targets", "period", "as_of", "run_id",
    "draft_revision", "definition", "definition_hash", "valid", "context_hash", "result_kind",
    "display_latex", "editable_latex", "dependencies", "diagnostics", "stale", "updated_at",
    "available", "page", "snapshot_id", "captured_at", "sections", "read_tool", "reason",
})

_KEY_FILENAME = "agent_policy.key"
_key_cache: dict[str, bytes] = {}
_key_lock = threading.Lock()

_TABLE_SPLIT = re.compile(r"[,\t;|]")
_NUMERIC_CELL = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?%?$")
_DATE_CELL = re.compile(r"^\d{4}-\d{2}-\d{2}$|^\d{8}$|^\d{4}/\d{1,2}/\d{1,2}$")


def stable_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


# --------------------------------------------------------------------------- #
# Signing key and seals
# --------------------------------------------------------------------------- #

def _signing_key() -> bytes:
    from .llm_settings import resolve_agent_data_dir

    path = resolve_agent_data_dir() / _KEY_FILENAME
    key = str(path)
    with _key_lock:
        if key in _key_cache:
            return _key_cache[key]
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            descriptor, temporary = tempfile.mkstemp(prefix=".agent-policy-", dir=path.parent)
            try:
                with os.fdopen(descriptor, "w", encoding="ascii") as handle:
                    handle.write(os.urandom(32).hex())
                    handle.flush()
                    os.fsync(handle.fileno())
                try:
                    os.link(temporary, path)  # publish a complete key, once, across workers
                except FileExistsError:
                    pass
            finally:
                os.unlink(temporary)
        raw = bytes.fromhex(path.read_text(encoding="ascii").strip())
        if len(raw) != 32:
            raise RuntimeError("AGENT_POLICY_KEY_INVALID")
        _key_cache[key] = raw
        return raw


def _signature(body: Any, *, purpose: str, source: str) -> str:
    payload = {"policy_version": POLICY_VERSION, "purpose": purpose, "source": source, "body": body}
    # Old page evidence can also be nested in context receipts, task state or a
    # summary. Expire those projections without invalidating confirmed memories.
    if (purpose == "tool_result" and source in {"page.read", "context.read", "task.state"}) or purpose == "summary":
        payload["page_view_version"] = 2
    return hmac.new(_signing_key(), stable_json(payload).encode("utf-8"), hashlib.sha256).hexdigest()


def seal(payload: dict[str, Any], tool: str) -> dict[str, Any]:
    """Stamp one server-produced tool receipt with an HMAC signature."""

    body = {key: value for key, value in payload.items() if key != "admission"}
    return {**body, "admission": {"policy_version": POLICY_VERSION, "source": tool,
                                  "signature": _signature(body, purpose="tool_result", source=tool)}}


def verify(payload: Any, tool: Optional[str]) -> bool:
    if not isinstance(payload, dict) or not tool:
        return False
    envelope = payload.get("admission")
    if not isinstance(envelope, dict) or envelope.get("policy_version") != POLICY_VERSION:
        return False
    if envelope.get("source") != tool:
        return False
    expected = envelope.get("signature")
    if not isinstance(expected, str):
        return False
    body = {key: value for key, value in payload.items() if key != "admission"}
    try:
        actual = _signature(body, purpose="tool_result", source=tool)
    except Exception:
        return False
    return hmac.compare_digest(expected, actual)


def seal_text(text: str, purpose: str) -> str:
    if purpose not in SYSTEM_PURPOSES and purpose != SUMMARY_PURPOSE:
        raise ValueError(f"unknown seal purpose: {purpose}")
    return _signature(text, purpose=purpose, source=purpose)


def verify_text(text: Any, purpose: str, signature: Any) -> bool:
    if not isinstance(text, str) or not isinstance(signature, str):
        return False
    if purpose not in SYSTEM_PURPOSES and purpose != SUMMARY_PURPOSE:
        return False
    try:
        expected = _signature(text, purpose=purpose, source=purpose)
    except Exception:
        return False
    return hmac.compare_digest(expected, signature)


INSTRUCTION_PURPOSES = frozenset({"compaction_source", "history_summary", "task_state"})


def seal_instruction(text: str, purpose: str) -> str:
    """Wrap a server-generated instruction body so the guard can verify its origin."""

    if purpose not in INSTRUCTION_PURPOSES:
        raise ValueError(f"unknown instruction purpose: {purpose}")
    return stable_json({"policy_version": POLICY_VERSION, "purpose": purpose,
                        "signature": _signature(text, purpose=purpose, source=purpose), "text": text})


def verify_instruction(content: Any, purpose: str) -> Optional[str]:
    """Return the sealed instruction body, or None for ordinary user text."""

    if not isinstance(content, str) or purpose not in INSTRUCTION_PURPOSES:
        return None
    try:
        parsed = json.loads(content)
    except ValueError:
        return None
    if not isinstance(parsed, dict) or parsed.get("purpose") != purpose:
        return None
    text = parsed.get("text")
    if not isinstance(text, str):
        return None
    if parsed.get("policy_version") != POLICY_VERSION:
        return None
    try:
        expected = _signature(text, purpose=purpose, source=purpose)
    except Exception:
        return None
    if not hmac.compare_digest(str(parsed.get("signature") or ""), expected):
        return None
    return text


# --------------------------------------------------------------------------- #
# Fail-closed structural invariants (used for guard backstops and embedded text)
# --------------------------------------------------------------------------- #

def _blocked(label: str, code: str) -> AgentError:
    return AgentError(
        "AGENT_DATA_ADMISSION_BLOCKED",
        "当前上下文包含未通过数据准入的载荷，本轮模型请求已停止；已完成的进度已保留。",
        status_code=409,
        field=label,
        diagnostics=[{"code": code, "field": label}],
    )


def check(value: Any, path: str = "messages") -> None:
    """Reject shapes that cannot be a registered summary (labels are registry-owned paths)."""

    _check(value, path, 1)


def _check(value: Any, path: str, depth: int) -> None:
    if depth > MAX_DEPTH:
        raise _blocked(path, "depth_exceeded")
    if isinstance(value, dict):
        if len(value) > MAX_KEYS:
            raise _blocked(path, "too_many_fields")
        for child in value.values():
            _check(child, path, depth + 1)
    elif isinstance(value, list):
        if len(value) > MAX_LIST_ITEMS:
            raise _blocked(path, "sequence_too_long")
        if value and all(_is_number(item) for item in value) and len(value) > MAX_NUMERIC_RUN:
            raise _blocked(path, "raw_value_sequence")
        if value and all(isinstance(item, str) for item in value) and len(value) > MAX_TEXT_RUN:
            raise _blocked(path, "raw_label_sequence")
        for child in value:
            _check(child, path, depth + 1)
    elif isinstance(value, str):
        return
    elif _is_number(value):
        if not math.isfinite(value):
            raise _blocked(path, "non_finite_number")
    elif value is None or isinstance(value, bool):
        return
    else:
        raise _blocked(path, "unsupported_type")


def _iter_json_structures(text: str, depth: int = 0):
    if depth > MAX_DEPTH:
        yield {"omitted": "encoded_depth_exceeded"}
        return
    decoder = json.JSONDecoder()
    index = 0
    length = len(text)
    while index < length:
        if text[index] in '{["':
            try:
                parsed, end = decoder.raw_decode(text, index)
            except ValueError:
                index += 1
                continue
            if isinstance(parsed, (dict, list)):
                yield parsed
                index = max(end, index + 1)
                continue
            if isinstance(parsed, str):
                yield from _iter_json_structures(parsed, depth + 1)
                index = max(end, index + 1)
                continue
        index += 1


def embedded_json_violation(text: Any, path: str = "messages") -> Optional[AgentError]:
    """Free text cannot turn a small serialized payload into trusted metadata."""
    code = user_text_violation(text)
    return _blocked(path, code) if code else None


def check_system_text(text: Any) -> None:
    """The system prompt may only embed registered fragments from server contracts."""

    if not isinstance(text, str):
        raise _blocked("system", "invalid_system_prompt")
    for parsed in _iter_json_structures(text):
        if not isinstance(parsed, dict) or not set(parsed) <= SYSTEM_TOP_KEYS:
            raise _blocked("system", "unregistered_system_payload")
        if "definition" in parsed and parsed["definition"] is not None:
            from .views import definition_view
            if definition_view(parsed["definition"]) != parsed["definition"]:
                raise _blocked("system", "unregistered_definition")
        check(parsed, "system")
        check_text_fields(parsed, "system")


def check_text_fields(value: Any, path: str = "messages", *, formula: bool = False) -> None:
    """Text slots never confer a structured-data contract on their contents."""
    if isinstance(value, dict):
        for key, child in value.items():
            check_text_fields(child, path, formula=key == "expression")
    elif isinstance(value, list):
        for child in value:
            check_text_fields(child, path)
    elif isinstance(value, str) and user_text_violation(value, allow_number=formula):
        raise _blocked(path, "structured_text_field")


def _looks_like_pasted_table(text: str) -> bool:
    """Conservative CSV-like table heuristic; natural-language limits stay documented."""

    previous = ""
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        cells = [cell.strip() for cell in _TABLE_SPLIT.split(stripped) if cell.strip()]
        numeric = sum(bool(_NUMERIC_CELL.fullmatch(cell) or _DATE_CELL.fullmatch(cell)) for cell in cells)
        if len(cells) >= 2 and numeric:
            return True
        if previous and _NUMERIC_CELL.fullmatch(stripped):
            return True
        previous = stripped
    return False


def user_text_violation(text: Any, *, allow_number: bool = False) -> Optional[str]:
    """Structured user input has no registered contract and fails closed."""

    if not isinstance(text, str) or not text.strip():
        return None
    if allow_number and _NUMERIC_CELL.fullmatch(text.strip()):
        return None
    try:
        parsed = json.loads(text)
    except ValueError:
        parsed = None
    if isinstance(parsed, (dict, list)) or _is_number(parsed):
        return "structured_user_payload"
    if isinstance(parsed, str) and parsed != text:
        return user_text_violation(parsed)
    if any(True for _ in _iter_json_structures(text)):
        return "structured_user_payload"
    if _looks_like_pasted_table(text):
        return "pasted_table_shape"
    return None


# --------------------------------------------------------------------------- #
# Capacity (runs after semantics)
# --------------------------------------------------------------------------- #

def bound(payload: Any, limit: Optional[int] = MAX_TOOL_RESULT_CHARS) -> tuple[Any, bool]:
    if limit is None:
        return payload, False
    text = stable_json(payload)
    if len(text) <= limit:
        return payload, False
    if isinstance(payload, dict) and isinstance(payload.get("items"), list):
        result = {**payload, "items": list(payload["items"])}
        result.setdefault("matched_count", len(payload["items"]))
        while result["items"] and len(stable_json(result)) > limit:
            result["items"].pop()
        if len(stable_json(result)) <= limit:
            result["returned_count"] = len(result["items"])
            result["hint"] = "只返回了相关条目；用具体名称或 ID 查询剩余条目。"
            return result, True
    return {"omitted": {"code": "result_too_large", "original_chars": len(text), "note": OMIT_NOTE}}, True


# --------------------------------------------------------------------------- #
# Admitted payloads and notices
# --------------------------------------------------------------------------- #

def admit(payload: Any, *, view: Callable[[Any, Any], tuple[Any, int]], projection: Any,
          limit: Optional[int] = MAX_TOOL_RESULT_CHARS) -> dict[str, Any]:
    """Project one tool payload through its registered view; never a fallback view."""

    projected, redactions = view(payload, projection)
    bounded, truncated = bound(projected, limit)
    return {"ok": True, "truncated": bool(redactions or truncated), "result": bounded,
            "_progress_payload": payload, "_projection_redactions": redactions}


def notice(*, source: str, status: str, message: str, ok: bool = False,
           extra: Optional[dict[str, Any]] = None, evidence_ref: Optional[str] = None) -> dict[str, Any]:
    """A server-authored, value-free tool receipt; sealed by the caller before sending."""

    payload: dict[str, Any] = {"ok": ok, "status": status, "message": message}
    if evidence_ref:
        payload["context_ref"] = evidence_ref
    if extra:
        payload.update(extra)
    return {**payload, "_notice_source": source}


def seal_notice(payload: dict[str, Any], tool: str) -> dict[str, Any]:
    body = {key: value for key, value in payload.items() if not key.startswith("_") and key != "admission"}
    return seal(body, tool)


# --------------------------------------------------------------------------- #
# Send guard
# --------------------------------------------------------------------------- #

def verified_system_purpose(text: Any, signature: Any) -> Optional[str]:
    """Return the registered purpose a system seal verifies for, or None."""

    if not isinstance(text, str) or not isinstance(signature, str):
        return None
    for purpose in SYSTEM_PURPOSES:
        if verify_text(text, purpose, signature):
            return purpose
    return None


def tool_call_sources(messages, *, allow_pending=False):
    """Pair results within the preceding assistant batch; IDs may recur after it closes."""
    sources, pending = {}, {}
    for index, message in enumerate(messages):
        path = f"messages[{index}]"
        if not isinstance(message, dict):
            raise _blocked(path, "invalid_message")
        if message.get("role") == "tool":
            call_id = message.get("tool_call_id")
            if not isinstance(call_id, str) or call_id not in pending:
                raise _blocked(path, "unpaired_tool_result")
            sources[index] = pending.pop(call_id)["function"]["name"]
            continue
        if pending:
            raise _blocked(path, "incomplete_tool_batch")
        if message.get("role") != "assistant":
            continue
        calls = message.get("tool_calls") or []
        if not isinstance(calls, list):
            raise _blocked(path, "invalid_tool_call")
        for call in calls:
            function = call.get("function") if isinstance(call, dict) else None
            call_id = call.get("id") if isinstance(call, dict) else None
            name = function.get("name") if isinstance(function, dict) else None
            if (not isinstance(call_id, str) or not call_id or call_id in pending
                    or not isinstance(name, str) or not name):
                raise _blocked(path, "invalid_tool_call")
            pending[call_id] = call
    if pending and not allow_pending:
        raise _blocked("messages", "incomplete_tool_batch")
    return sources, pending


def pending_tool_calls(messages):
    """The executable remainder comes from the transcript, never a second saved queue."""
    _, pending = tool_call_sources(messages, allow_pending=True)
    calls = []
    for call_id, call in pending.items():
        try:
            arguments = json.loads(call["function"]["arguments"])
        except (KeyError, TypeError, ValueError):
            raise _blocked("messages", "invalid_tool_arguments") from None
        if not isinstance(arguments, dict):
            raise _blocked("messages", "invalid_tool_arguments")
        calls.append({"id": call_id, "name": call["function"]["name"], "arguments": arguments})
    return calls


def enforce_request(*, system: Optional[str] = None, messages: Optional[Iterable[dict[str, Any]]] = None,
                    tools: Optional[Iterable[dict[str, Any]]] = None,
                    system_seal: Optional[str] = None) -> dict[str, Any]:
    """Re-verify one outbound model request; raises AgentError on any bypass."""

    purpose = verified_system_purpose(system, system_seal)
    if purpose is None:
        raise _blocked("system", "unsealed_system_prompt")
    check_system_text(system)
    items = list(messages or ())
    sources, _ = tool_call_sources(items)
    tool_results = 0
    for index, message in enumerate(items):
        if not isinstance(message, dict):
            raise _blocked(f"messages[{index}]", "invalid_message")
        role = message.get("role")
        path = f"messages[{index}]"
        if set(message) - {"role", "content", "tool_calls", "tool_call_id", "reasoning_content"}:
            raise _blocked(path, "unregistered_message_field")
        if role == "tool":
            tool_results += 1
            _enforce_tool_message(message, sources[index], path)
        elif role == "assistant":
            from .tools import TOOL_REGISTRY
            for call_index, call in enumerate(message.get("tool_calls") or []):
                function = call.get("function") if isinstance(call, dict) else None
                raw = function.get("arguments") if isinstance(function, dict) else None
                if (not isinstance(call, dict) or set(call) - {"id", "type", "function"}
                        or not isinstance(function, dict) or set(function) != {"name", "arguments"}
                        or not isinstance(raw, str)
                        or function.get("name") not in {*TOOL_REGISTRY, "unavailable"}):
                    raise _blocked(path, "invalid_tool_call")
                if isinstance(raw, str):
                    try:
                        parsed = json.loads(raw)
                    except ValueError:
                        raise _blocked(path, "invalid_tool_arguments") from None
                    if raw != "{}" and not enforce_arguments(function.get("name"), raw):
                        raise _blocked(path, "unregistered_tool_arguments")
            violation = embedded_json_violation(message.get("content"), path)
            if violation is not None:
                raise violation
            violation = embedded_json_violation(message.get("reasoning_content"), path)
            if violation is not None:
                raise violation
        elif role == "user":
            # A server-sealed instruction body (e.g. the compaction source) is checked
            # structurally; every other user message must be free natural language.
            sealed_body = (verify_instruction(message.get("content"), "compaction_source")
                           or verify_instruction(message.get("content"), "history_summary")
                           or verify_instruction(message.get("content"), "task_state"))
            if sealed_body is not None:
                check(_parse_or_none(sealed_body) or {}, path)
            else:
                code = user_text_violation(message.get("content"))
                if code:
                    raise _blocked(path, code)
        else:
            raise _blocked(path, "unregistered_message_role")
    return {"policy_version": POLICY_VERSION, "messages": len(items), "tools": len(tuple(tools or ())),
            "system_purpose": purpose, "tool_results": tool_results}


def _enforce_tool_message(message: dict[str, Any], tool: Optional[str], path: str) -> None:
    content = message.get("content")
    if not isinstance(content, str):
        raise _blocked(f"{path}.content", "invalid_tool_message")
    try:
        payload = json.loads(content)
    except ValueError:
        raise _blocked(f"{path}.content", "unreadable_tool_message") from None
    if not isinstance(payload, dict):
        raise _blocked(f"{path}.content", "unreadable_tool_message")
    if not verify(payload, tool):
        raise _blocked(f"{path}.admission", "unverified_tool_result")
    check(payload, path)


def enforce_arguments(tool: Optional[str], raw: Any) -> bool:
    """Registered-contract check used when rebuilding stored tool arguments."""

    if not tool or not isinstance(raw, str):
        return False
    try:
        parsed = json.loads(raw)
    except ValueError:
        return False
    if not isinstance(parsed, dict):
        return False
    try:
        check(parsed)
        check_text_fields(parsed)
        from .tools import parse_arguments
        validated = parse_arguments(tool, parsed).model_dump(exclude_unset=True)
        if validated != parsed:
            return False
        if isinstance(parsed.get("definition"), dict):
            from .views import definition_view
            allowed = definition_view(parsed["definition"])
            if any(key not in allowed for key in parsed["definition"]
                   if parsed["definition"][key] is not None):
                return False
    except (AgentError, ValueError, TypeError):
        return False
    return True


# --------------------------------------------------------------------------- #
# Historical reprojection
# --------------------------------------------------------------------------- #

def _evidence_ref(result: Any) -> Optional[str]:
    if not isinstance(result, dict):
        return None
    reference = result.get("context_ref")
    if isinstance(reference, str):
        return reference
    inner = result.get("result")
    if isinstance(inner, dict) and isinstance(inner.get("context_ref"), str):
        return inner["context_ref"]
    return None


def _omission_notice(tool: Optional[str], reference: Optional[str]) -> dict[str, Any]:
    payload = notice(source=tool or "unknown", status="policy_reprojected", message=REPROJECT_NOTE,
                     evidence_ref=reference)
    return seal_notice(payload, tool or "unknown")


def reproject_evidence(tool: Optional[str], result: Any,
                       view: Optional[Callable[[Any, Any], tuple[Any, int]]] = None,
                       projection: Any = None, limit: Optional[int] = MAX_TOOL_RESULT_CHARS) -> dict[str, Any]:
    """Rebuild one persisted receipt under the current policy or omit it explicitly."""

    if verify(result, tool):
        return result
    if not isinstance(result, dict):
        return _omission_notice(tool, None)
    reference = _evidence_ref(result)
    if not isinstance(reference, str) or not re.fullmatch(r"op-[0-9a-f]{32}", reference):
        reference = None
    # An old receipt has lost its source proof. Re-execution is safer than signing
    # a projection of an untrusted historical body (especially paginated text).
    return _omission_notice(tool, reference)


def scrub_checkpoint(checkpoint: dict[str, Any], *,
                     view_for: Optional[Callable[[str], Any]] = None,
                     projection_for: Optional[Callable[[str], Any]] = None,
                     ) -> dict[str, int]:
    """Make a persisted checkpoint sendable: rebuild old receipts, drop opaque records."""

    stats = {"tool_results": 0, "reprojected": 0, "arguments_reset": 0, "summary_dropped": 0,
             "user_payloads": 0}
    messages = checkpoint.get("messages")
    if isinstance(messages, list):
        from .tools import TOOL_REGISTRY
        try:
            sources, _ = tool_call_sources(messages, allow_pending=True)
        except AgentError:
            # Sanitize malformed legacy history without inventing a pairing. The
            # unchanged protocol structure still fails the outbound/compaction gate.
            sources = {}
        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            if role == "tool":
                stats["tool_results"] += 1
                source = sources.get(index)
                tool = source if source in TOOL_REGISTRY else "unavailable" if source else None
                view = view_for(tool) if view_for and tool else None
                projection = projection_for(tool) if projection_for and tool else None
                content = message.get("content")
                rebuilt = reproject_evidence(tool, _parse_or_none(content), view=view,
                                             projection=projection)
                new_content = stable_json(rebuilt)
                if new_content != content:
                    message["content"] = new_content
                    stats["reprojected"] += 1
            elif role == "assistant":
                if user_text_violation(message.get("reasoning_content")):
                    message.pop("reasoning_content", None)
                for call in message.get("tool_calls") or []:
                    function = call.get("function") if isinstance(call, dict) else None
                    if isinstance(function, dict) and function.get("name") not in TOOL_REGISTRY:
                        function["name"] = "unavailable"
                        function["arguments"] = "{}"
                    raw = function.get("arguments") if isinstance(function, dict) else None
                    if isinstance(raw, str) and raw != "{}":
                        tool = function.get("name") if isinstance(function, dict) else None
                        if not enforce_arguments(tool, raw):
                            function["arguments"] = "{}"
                            stats["arguments_reset"] += 1
                violation = embedded_json_violation(message.get("content"))
                if violation is not None:
                    message["content"] = STRUCTURED_OMIT_NOTE
                    stats["reprojected"] += 1
            elif role == "user" and isinstance(message.get("content"), str):
                if user_text_violation(message["content"]):
                    message["content"] = STRUCTURED_OMIT_NOTE
                    stats["user_payloads"] += 1
    pinned = checkpoint.get("pinned_user_messages")
    if isinstance(pinned, list):
        for message in pinned:
            if isinstance(message, dict) and isinstance(message.get("content"), str) and user_text_violation(message["content"]):
                message["content"] = STRUCTURED_OMIT_NOTE
                stats["user_payloads"] += 1
    summary = checkpoint.get("summary")
    if summary:
        if not verify_text(summary, SUMMARY_PURPOSE, checkpoint.get("summary_seal")):
            checkpoint["summary"] = ""
            checkpoint["summary_omitted"] = True
            checkpoint.pop("summary_seal", None)
            stats["summary_dropped"] += 1
    return stats


def _parse_or_none(content: Any) -> Any:
    if not isinstance(content, str):
        return None
    try:
        return json.loads(content)
    except ValueError:
        return None

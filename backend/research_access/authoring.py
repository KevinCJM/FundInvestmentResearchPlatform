import copy
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def stable_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)

def stable_hash(payload: Any) -> str:
    return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()

def definition_hash(definition: dict[str, Any]) -> str:
    def normalized(value):
        if isinstance(value, dict):
            return {k: normalized(v) for k, v in value.items()}
        if isinstance(value, list):
            return [normalized(v) for v in value]
        if isinstance(value, float) and value.is_integer():
            return int(value)
        return value
    return stable_hash(normalized(definition))

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

def storage_directory():
    from backend.data_storage import guard_path
    root = Path(os.environ.get("CUSTOM_INDICATOR_DATA_DIR") or Path(__file__).resolve().parents[2] / "data")
    path = root / "research_access"
    guard_path(path, write=True)
    return path

"""Bounded, deterministic no-progress detection from tool evidence, not model prose."""
from __future__ import annotations

import copy
from typing import Any
from .sessions import stable_hash
from .tools import TOOL_REGISTRY

HISTORY_SIZE = 32
# Only explicit transport/compilation metadata is omitted; dates, values and IDs remain meaningful.
NOISE = {"request_id", "duration_ms", "elapsed_ms", "compile_token", "compile_token_issued", "updated_at", "draft_revision",
         "snapshot_id", "captured_at", "admission"}


def stable_outcome(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: stable_outcome(v) for k, v in value.items() if k not in NOISE and not k.startswith("_")}
    if isinstance(value, list):
        return [stable_outcome(v) for v in value]
    return value


def semantic_definition(definition):
    return {k: v for k, v in (definition or {}).items() if k not in {"name", "description", "display_latex", "updated_at", "revision", "id"}}


def call_key(tool, arguments, dependencies):
    definition = TOOL_REGISTRY.get(tool)
    family = (definition.equivalent_to or tool) if definition else tool
    args = copy.deepcopy(arguments)
    if definition and definition.progress == "draft" and isinstance(args.get("definition"), dict):
        # Only detection ignores cosmetic definition changes; token/cache keys must not.
        args["definition"] = semantic_definition(args["definition"])
    return stable_hash([family, args, dependencies])


def diagnostic_keys(diagnostics):
    return sorted({stable_hash([d.get("code"), d.get("field"), d.get("severity", "error")]) for d in diagnostics or []})


class ProgressGuard:
    def __init__(self, state=None, *, continued=False, seen=None, relevant=None):
        self.state = copy.deepcopy(state or {})
        self.seen = seen or (lambda kind, values: set())
        self.relevant = relevant or (lambda tool, arguments: True)
        self.pending_facts = []
        for key, default in {"history": [], "seen_facts": [], "seen_previews": [], "seen_states": [], "blocked": [],
                             "stagnant": 0, "repair_stall": 0, "recovering": False, "remaining": 0,
                             "pause": False, "reason": None, "best_valid": False, "best_diagnostics": None}.items():
            self.state.setdefault(key, default)
        if continued:
            self.state.update(recovering=False, remaining=0, pause=False, stagnant=0, repair_stall=0)
            # Rejected strategies survive a human "continue" and prompt compaction.

    def snapshot(self):
        return copy.deepcopy(self.state)

    def check_call(self, tool, arguments, dependencies):
        return call_key(tool, arguments, dependencies) in self.state["blocked"]

    def _trigger(self, reason, keys):
        self.state["blocked"] = list(dict.fromkeys(self.state["blocked"] + list(keys)))[-128:]
        self.state["reason"] = reason
        if not self.state["recovering"]:
            self.state.update(recovering=True, remaining=2)
            return True
        return False

    def record_blocked(self, tool, arguments, dependencies):
        self.pending_facts = []
        key = call_key(tool, arguments, dependencies)
        was_recovering = self.state["recovering"]
        started = self._trigger("repeat_no_change", [key])
        if was_recovering:
            self.state["remaining"] -= 1
            self.state["pause"] = self.state["remaining"] <= 0
        return {"progress": False, "reason": self.state["reason"], "recovery_started": started, "pause": self.state["pause"]}

    def record(self, tool, arguments, result, *, draft=None, dependencies=None):
        self.pending_facts = []
        s = self.state
        key = call_key(tool, arguments, dependencies or {})
        payload = result.get("_progress_payload", result.get("result", result))
        outcome = stable_hash(stable_outcome(payload))
        error = result.get("error") or {}
        failed = result.get("ok") is False or (isinstance(payload, dict) and payload.get("valid") is False)
        diags = diagnostic_keys(error.get("diagnostics") or (payload.get("diagnostics") if isinstance(payload, dict) else []))
        error_key = stable_hash([error.get("code"), error.get("field"), diags]) if failed else None
        progress = False
        state_key = None
        definition = TOOL_REGISTRY.get(tool)
        category = definition.progress if definition else None
        if category == "draft" and draft and result.get("ok") is not False:
            state_key = stable_hash([semantic_definition(draft.get("definition")), bool(draft.get("valid")), diagnostic_keys(draft.get("diagnostics"))])
            diags = diagnostic_keys(draft.get("diagnostics"))
            valid = bool(draft.get("valid"))
            progress = valid and not s["best_valid"]
            best = s["best_diagnostics"]
            if not valid and best and set(diags) < set(best):
                progress = state_key not in s["seen_states"]
            if valid:
                s["best_valid"] = True
            if best is None or progress:
                s["best_diagnostics"] = diags
            s["repair_stall"] = 0 if progress else s["repair_stall"] + 1
            s["seen_states"] = list(dict.fromkeys(s["seen_states"] + [state_key]))[-256:]
            failed = not valid
            error_key = stable_hash(diags) if failed else None
        elif not failed and category == "read":
            if definition.ignored_read_fields and isinstance(payload, dict):
                # A different expression string alone is a candidate, not inference progress.
                payload = {k: v for k, v in payload.items() if k not in definition.ignored_read_fields}
            if isinstance(payload, dict) and isinstance(payload.get("items"), list):
                facts = [stable_hash([tool, row.get("id", row.get("name", row.get("product_id"))), row.get("revision"), stable_outcome(row)])
                         for row in payload["items"] if isinstance(row, dict)]
            else:
                facts = [stable_hash([tool, stable_outcome(payload)])]
            candidates = set(facts) - set(s["seen_facts"])
            fresh = candidates - self.seen("read", candidates)
            self.pending_facts = [("read", value) for value in fresh]
            # New directory trivia cannot reset an unresolved formula-repair episode.
            progress = bool(fresh) and self.relevant(tool, arguments) and not (draft and not draft.get("valid"))
            s["seen_facts"] = list(dict.fromkeys(s["seen_facts"] + facts))[-512:]
        elif not failed and category == "preview":
            # Frozen target/definition/data versions identify the completed work.
            # Fresh execution IDs or cache statistics do not create a new preview.
            evidence = key
            progress = evidence not in s["seen_previews"] and not self.seen("preview", {evidence})
            if progress:
                self.pending_facts = [("preview", evidence)]
            s["seen_previews"] = list(dict.fromkeys(s["seen_previews"] + [evidence]))[-256:]
        item = {"key": key, "outcome": outcome, "state": state_key, "error": error_key, "progress": progress}
        history = (s["history"] + [item])[-HISTORY_SIZE:]
        s["history"] = history
        s["stagnant"] = 0 if progress else s["stagnant"] + 1
        started = False
        if progress:
            s.update(recovering=False, remaining=0, pause=False)
        elif s["recovering"]:
            s["remaining"] -= 1
            s["pause"] = s["remaining"] <= 0
        else:
            reason, keys = None, [key]
            if failed and len(history) >= 2 and all(h["key"] == key and h["error"] == error_key for h in history[-2:]):
                reason = "repeat_failure"
            elif len(history) >= 3 and all(h["key"] == key and h["outcome"] == outcome for h in history[-3:]):
                reason = "repeat_no_change"
            else:
                for size in (2, 3, 4):
                    tail = history[-size*3:]
                    signatures = [(h["state"] or h["key"], h["error"] or h["outcome"]) for h in tail]
                    if len(tail) == size*3 and signatures[:size] == signatures[size:2*size] == signatures[2*size:] and not any(h["progress"] for h in tail[size:]):
                        reason, keys = "cycle_no_progress", [h["key"] for h in tail]
                        break
            if reason is None and s["repair_stall"] >= 4:
                reason = "repair_stall"
            if reason is None and s["stagnant"] >= 8:
                reason = "stagnation"
            if reason:
                started = self._trigger(reason, keys)
        return {"progress": progress, "reason": s["reason"] if s["recovering"] else None,
                "recovery_started": started, "pause": s["pause"]}

    def recovery_prompt(self):
        return ("系统已发现当前方法没有新进展（" + str(self.state.get("reason")) + "）。"
                "不要重复相同查询、错误公式或只改名称。请根据真实诊断换一种方法，或直接说明阻碍并向用户澄清。"
                "先前草稿和工具证据已保留；不得改变用户口径来绕过校验。")

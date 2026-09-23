"""Offline, fixed-model evaluation fixtures and explicit scorers (AI-05 / AC-5).

These tasks exercise the harness with ``FixtureLLMClient`` and temporary data
only: zero and missing results, a full-catalog tail lookup, and a prohibited
raw paste.  They are deterministic engineering evidence for this fixed fixture
model — not real-model success rates, token costs or production quality claims.

``run_task`` returns one bounded result per trial with the case id, pass or the
explicit violations, usage and duration, so repeated attempts can be compared.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from contextlib import ExitStack
from unittest.mock import patch
from pathlib import Path
from typing import Any, Callable, Optional

from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent import data_policy
from agent.llm import FixtureLLMClient
from agent.sessions import stable_json
from custom_indicators.service import CustomIndicatorService
from test_agent_api import FakeIndicatorService, authoring_context, evidence_snapshot

TRIALS = 3
SENTINEL = 987654.321


class EvalService(FakeIndicatorService):
    """Fake evaluation surface with the real compiler for derivation proofs."""

    def __init__(self, market_data_dir: Path) -> None:
        super().__init__(market_data_dir)
        self.real = CustomIndicatorService(market_data_dir, market_data_dir)
        self.evaluate_calls: list[dict[str, Any]] = []

    def infer(self, fields: dict[str, Any]) -> dict[str, Any]:
        return self.real.infer(fields)

    def get_indicator(self, indicator_id: str, revision: int | None = None) -> dict[str, Any]:
        return self.real.get_indicator(indicator_id, revision)

    def list_indicators(self, **kwargs: Any) -> dict[str, Any]:
        return {"items": [{"id": f"indicator-{index:03d}", "name": f"指标 {index:03d}", "revision": 1,
                           "context_kind": "single_product", "result_kind": "scalar"}
                          for index in range(1, 402)], "total": 401}

    def meta(self) -> dict[str, Any]:
        return {"engine_version": "eval", "dsl_version": "2.4.0", "operator_registry_version": "op-1",
                "variable_registry_version": "var-1", "periods": [{"id": "1Y"}],
                "variables": [{"name": f"variable_{index:03d}", "label": f"变量 {index:03d}",
                               "description": "研究所需变量"} for index in range(1, 402)]}

    def evaluate(self, **kwargs: Any) -> dict[str, Any]:
        self.evaluate_calls.append(dict(kwargs))
        product_id = kwargs["targets"][0]["product_id"]
        row = {"status": "ok", "value": 0.0, "indicator_name": "AI 零值指标",
               "target": {"kind": "etf", "product_id": product_id},
               "window": {"effective_as_of": "2026-09-18", "observation_count": 240}, "warnings": []}
        if product_id == "512960.SH":
            row.update(status="unavailable", value=None,
                       warnings=[{"code": "NO_FINITE_SERIES_RESULT", "message": "没有有限结果"}])
        return {"results": [row], "execution": {"nopython": True}}


def zero_and_missing_snapshot() -> dict[str, Any]:
    snapshot = evidence_snapshot(rows=2)
    snapshot["sections"]["results"]["frozen_request"]["targets"] = [
        {"kind": "etf", "product_id": "510300.SH"}, {"kind": "etf", "product_id": "512960.SH"}]
    return snapshot


def prohibited_paste_text() -> str:
    return ("date,nav,ret\n"
            f"2026-01-02,{SENTINEL},0.0012\n"
            "2026-01-03,987655.1,0.0008\n"
            "2026-01-04,987655.9,0.0005")


@dataclass(frozen=True)
class EvalCase:
    id: str
    prompt: str
    replies: tuple
    snapshot: Optional[dict] = None
    expect: dict = field(default_factory=dict)
    message_id: str = "eval"


ZERO_TASK = EvalCase(
    id="zero-result",
    prompt="页面上的指标为什么是 0？",
    snapshot=zero_and_missing_snapshot(),
    replies=(
        {"tool_calls": [{"name": "page.read", "arguments": {"section": "results"}}]},
        {"tool_calls": [{"name": "page.recompute", "arguments": {"group_index": 0}}]},
        {"tool_calls": [{"name": "page.recompute", "arguments": {"group_index": 1}}]},
        {"content": "页面数字未经核验，已按冻结口径重算：一个为 0，一个不可用。"},
    ),
    expect={"zero": True, "null": True},
)

MISSING_TASK = EvalCase(
    id="missing-data",
    prompt="缺失数据时这个指标怎么显示？",
    snapshot=zero_and_missing_snapshot(),
    replies=(
        {"tool_calls": [{"name": "page.recompute", "arguments": {"group_index": 1}}]},
        {"content": "该产品没有有限结果，页面显示不可用。"},
    ),
    expect={"null": True},
)

CATALOG_TAIL_TASK = EvalCase(
    id="catalog-tail",
    prompt="查一下最后一个指标 indicator-401 的契约。",
    replies=(
        {"tool_calls": [{"name": "metrics.lookup", "arguments": {"kind": "indicators",
                                                                "query": "indicator-401", "limit": 5}}]},
        {"content": "目录完整参与检索，尾部条目可查到。"},
    ),
    expect={"tail": True},
)

PROHIBITED_TASK = EvalCase(
    id="prohibited-paste",
    prompt=prohibited_paste_text(),
    replies=({"content": "不应被调用"},),
    expect={"prohibited": True},
)

TASKS = (ZERO_TASK, MISSING_TASK, CATALOG_TAIL_TASK, PROHIBITED_TASK)


def _scorer(case: EvalCase, observed: dict[str, Any]) -> list[str]:
    """Explicit, deterministic violations for one trial (empty list means pass)."""

    violations: list[str] = []
    requests = observed.get("requests") or []
    response = observed.get("response") or {}
    detail = observed.get("detail") or {}
    if case.expect.get("prohibited"):
        if requests:
            violations.append("rejected payload reached the model")
        if detail.get("code") != "AGENT_DATA_ADMISSION_BLOCKED":
            violations.append("prohibited paste did not fail closed")
        if str(SENTINEL) in stable_json(response):
            violations.append("rejected value was echoed")
        return violations
    for message in requests:
        try:
            data_policy.enforce_request(system=message["system"], messages=message["messages"],
                                        tools=message["tools"],
                                        system_seal=data_policy.seal_text(message["system"], "primary"))
        except Exception:
            violations.append("captured request violated the current admission policy")
        if str(SENTINEL) in stable_json(message["messages"]):
            violations.append("raw sentinel reached a model request")
    tool_messages = [json.loads(item["content"]) for message in requests for item in message["messages"]
                     if item.get("role") == "tool"]
    by_tool = {item.get("tool"): item for item in tool_messages}
    if case.expect.get("zero") or case.expect.get("null"):
        recomputed = [item for item in tool_messages if item.get("tool") == "page.recompute"]
        rows = [item.get("result", {}).get("results", [{}])[0] for item in recomputed]
        values = [row["value"] for row in rows if "value" in row]
        if any("value" not in row for row in rows):
            violations.append("recomputed evidence omitted the expected value field")
        if case.expect.get("zero") and 0 not in values:
            violations.append("0 was not preserved")
        if case.expect.get("null") and not any(value is None for value in values):
            violations.append("explicit null was not preserved")
        page_read = by_tool.get("page.read", {})
        content = page_read.get("result", {}).get("content")
        if isinstance(content, str) and '"value"' in content:
            violations.append("client result value label reached the model")
    if case.expect.get("tail"):
        lookup = by_tool.get("metrics.lookup", {})
        payload = lookup.get("result", {})
        items = payload.get("items") or []
        if payload.get("total") != 401 or not items or items[0].get("id") != "indicator-401":
            violations.append("catalog tail lookup was incomplete")
    if not response.get("reply", {}).get("text"):
        violations.append("no reply text")
    return violations


def run_task(case: EvalCase, root: Path, *, trials: int = TRIALS) -> list[dict[str, Any]]:
    """Run one fixed task against a fresh offline harness per trial."""

    reports: list[dict[str, Any]] = []
    for trial in range(trials):
        started = time.monotonic()
        workdir = root / f"{case.id}-trial-{trial}"
        workdir.mkdir(parents=True, exist_ok=True)
        import os

        with ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, {"CUSTOM_INDICATOR_DATA_DIR": str(workdir)}))
            from services import custom_indicator_routes
            from agent import routes as agent_routes
            from services.llm_settings_routes import router as settings_router

            service = EvalService(workdir)
            stack.enter_context(patch.object(custom_indicator_routes, "indicator_service", service))
            llm = FixtureLLMClient(list(case.replies))
            stack.enter_context(patch.object(agent_routes, "_llm_client", lambda session_id: llm))
            app = FastAPI()
            app.include_router(settings_router)
            app.include_router(agent_routes.router)
            observed: dict[str, Any] = {"requests": llm.requests}
            with TestClient(app) as client:
                client.put("/api/settings/llm", json={"api_key": "test-key-123456"})
                context = authoring_context()
                session = client.post("/api/agent/sessions", json={"page_context": context}).json()
                body: dict[str, Any] = {"message_id": f"{case.id}-{trial}", "expected_session_revision": 0,
                                        "text": case.prompt, "page_context": context}
                if case.snapshot is not None:
                    body["page_snapshot"] = case.snapshot
                response = client.post(f"/api/agent/sessions/{session['session_id']}/messages", json=body)
                try:
                    payload = response.json()
                except ValueError:
                    payload = {}
                if response.status_code >= 400:
                    observed["detail"] = payload.get("detail", payload)
                else:
                    observed["response"] = payload
            violations = _scorer(case, observed)
        reports.append({
            "case": case.id,
            "trial": trial,
            "status": "pass" if not violations else "violation",
            "violations": violations,
            "usage": (observed.get("response") or {}).get("usage"),
            "duration_ms": int((time.monotonic() - started) * 1000),
        })
    return reports


def run_all(root: Path, *, trials: int = TRIALS) -> list[dict[str, Any]]:
    return [report for case in TASKS for report in run_task(case, root, trials=trials)]

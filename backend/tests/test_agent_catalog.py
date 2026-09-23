"""AC-4: full-catalog search, true total/revision identity, and side-effect-free imports.

A 401-item fixture proves the tail participates in search, ``total`` and the
version fingerprint, and a fresh subprocess proves that importing the pure
request/tool chain creates no business store files in an isolated DATA_DIR.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from agent.catalog import build_catalog, matched_items, search_catalog

REPO_ROOT = Path(__file__).resolve().parents[2]


class CatalogService:
    def __init__(self, count: int = 401) -> None:
        self.items = [
            {"id": f"indicator-{index:03d}", "name": f"指标 {index:03d}", "revision": 1,
             "context_kind": "single_product", "result_kind": "scalar", "description": "目录条目"}
            for index in range(1, count + 1)
        ]

    def meta(self) -> dict[str, Any]:
        return {"engine_version": "t", "dsl_version": "2.4.0", "operator_registry_version": "op-1",
                "variable_registry_version": "var-1", "periods": [{"id": "1Y"}]}

    def list_indicators(self, **kwargs: Any) -> dict[str, Any]:
        return {"items": list(self.items), "total": len(self.items)}


def test_full_catalog_total_tail_search_and_revision_fingerprint():
    service = CatalogService()
    catalog = build_catalog(service)
    assert catalog["total"] == 401 and len(catalog["items"]) == 401
    tail = matched_items(catalog, query="indicator-401")
    assert [item["id"] for item in tail] == ["indicator-401"]
    assert len(search_catalog(catalog, query="", limit=5)) == 5
    assert len(matched_items(catalog, query="")) == 401
    version_before = catalog["version"]
    service.items[-1]["revision"] = 2  # a tail revision change must move the fingerprint
    moved = build_catalog(service)
    assert moved["version"] != version_before and moved["total"] == 401
    service.items[-1]["revision"] = 1
    assert build_catalog(service)["version"] == version_before


def test_tool_lookup_returns_bounded_entries_with_true_match_count(tmp_path):
    from agent.contracts import PageContext
    from agent.sessions import apply_context
    from agent.tools import execute_tool
    from test_agent_api import FakeIndicatorService, authoring_context

    service = FakeIndicatorService(tmp_path)
    service.list_indicators = CatalogService().list_indicators
    service.meta = CatalogService().meta
    context = PageContext.model_validate(authoring_context())
    state = {"scope": "indicator_center"}
    apply_context(state, context)
    result = execute_tool("metrics.lookup", {"kind": "indicators", "query": "indicator-401", "limit": 5},
                          session=state, page_context=context, service=service)
    payload = result["result"]
    assert payload["total"] == 401 and payload["matched_count"] == 1
    assert [item["id"] for item in payload["items"]] == ["indicator-401"]
    from agent import data_policy
    receipt = data_policy.seal({key: value for key, value in result.items() if not key.startswith("_")}, "metrics.lookup")
    assert data_policy.verify(receipt, "metrics.lookup")


def test_pure_request_imports_do_not_construct_service_stores(tmp_path):
    env = {
        "PYTHONPATH": f"{REPO_ROOT}:{REPO_ROOT / 'backend'}",
        "CUSTOM_INDICATOR_DATA_DIR": str(tmp_path),
        "NUMBA_CACHE_DIR": str(tmp_path.parent / "p1-import-numba-cache"),
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
    }
    code = ("import agent.tools, agent.routes, agent.harness, agent.catalog, agent.views, "
            "agent.data_policy, agent.derivation, agent.research_pages, services.custom_indicator_contracts")
    completed = subprocess.run([sys.executable, "-c", code], env=env, cwd=str(REPO_ROOT),
                               capture_output=True, text=True, timeout=300)
    assert completed.returncode == 0, completed.stderr
    created = sorted(path.name for path in tmp_path.iterdir())
    assert created == [], f"纯请求导入不得初始化业务存储：{created}"

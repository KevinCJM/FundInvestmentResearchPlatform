from __future__ import annotations

import copy
import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from custom_indicators.errors import ValidationError as IndicatorValidationError
from product_pools.repository import ProductPoolRepository
from product_pools.service import ProductPoolService
from services import product_pool_routes


class FakeEvaluationGateway:
    def __init__(self) -> None:
        self.plans = {
            "plan-equity": {
                "id": "plan-equity",
                "revision": 4,
                "name": "沪深300被动权益ETF评价",
                "product_kind": "etf",
            },
            "plan-bond": {
                "id": "plan-bond",
                "revision": 2,
                "name": "中短债基金评价",
                "product_kind": "fund",
            },
        }

    def get_plan(self, plan_id: str) -> dict:
        return copy.deepcopy(self.plans[plan_id])

    def run_plan(self, plan_id: str, as_of: str | None = None) -> dict:
        plan = self.plans[plan_id]
        target = (
            {"kind": "etf", "product_id": "510300.SH", "name": "沪深300ETF"}
            if plan_id == "plan-equity"
            else {"kind": "fund", "product_id": "000001.OF", "name": "中短债基金"}
        )
        return {
            "plan_id": plan_id,
            "plan_revision": plan["revision"],
            "run_at": "2026-09-04T10:00:00+00:00",
            "as_of": as_of,
            "rows": [{"rank": 1, "target": target, "score": 90.0, "status": "ranked"}],
            "ranked_count": 1,
            "excluded_count": 0,
            # Deliberately omit result_id: normal small evaluation runs are inline.
        }

    def get_run_page(self, result_id: str, *, page: int, page_size: int) -> dict:
        raise AssertionError("inline result must not request another page")


def _client(monkeypatch, tmp_path: Path) -> TestClient:
    service = ProductPoolService(
        ProductPoolRepository(tmp_path / "product_pools.json"),
        FakeEvaluationGateway(),
    )
    monkeypatch.setattr(product_pool_routes, "product_pool_service", service)
    app = FastAPI()
    app.include_router(product_pool_routes.router)
    return TestClient(app)


def test_inline_runs_can_attach_multiple_plans_and_delete_one(monkeypatch, tmp_path: Path) -> None:
    client = _client(monkeypatch, tmp_path)
    pool = client.post(
        "/api/product-pools",
        json={"name": "核心池", "description": "", "purpose": "长期配置", "owner": "Kevin"},
    ).json()

    first = client.post(
        f"/api/product-pools/{pool['id']}/evaluation-plans",
        json={
            "revision": pool["revision"],
            "plan_id": "plan-equity",
            "selection_mode": "all_ranked",
            "selection_value": None,
        },
    )
    assert first.status_code == 200
    pool = first.json()
    assert pool["evaluation_plans"][0]["result_id"].startswith("inline-")
    assert [item["plan_id"] for item in pool["evaluation_plans"]] == ["plan-equity"]

    second = client.post(
        f"/api/product-pools/{pool['id']}/evaluation-plans",
        json={
            "revision": pool["revision"],
            "plan_id": "plan-bond",
            "selection_mode": "all_ranked",
            "selection_value": None,
        },
    )
    assert second.status_code == 200
    pool = second.json()
    assert [item["plan_id"] for item in pool["evaluation_plans"]] == [
        "plan-equity",
        "plan-bond",
    ]

    removed = client.delete(
        f"/api/product-pools/{pool['id']}/evaluation-plans/plan-equity",
        params={"revision": pool["revision"]},
    )
    assert removed.status_code == 200
    pool = removed.json()
    assert [item["plan_id"] for item in pool["evaluation_plans"]] == ["plan-bond"]
    assert {item["key"] for item in pool["members"]} == {"fund:000001.OF"}

    persisted = client.get(f"/api/product-pools/{pool['id']}")
    assert persisted.status_code == 200
    assert persisted.json()["revision"] == pool["revision"]
    assert [item["plan_id"] for item in persisted.json()["evaluation_plans"]] == ["plan-bond"]


def test_batch_review_route_updates_multiple_members_once(monkeypatch, tmp_path: Path) -> None:
    client = _client(monkeypatch, tmp_path)
    pool = client.post("/api/product-pools", json={"name": "批量复核"}).json()
    for plan_id in ("plan-equity", "plan-bond"):
        response = client.post(
            f"/api/product-pools/{pool['id']}/evaluation-plans",
            json={
                "revision": pool["revision"],
                "plan_id": plan_id,
                "selection_mode": "all_ranked",
            },
        )
        assert response.status_code == 200
        pool = response.json()

    response = client.put(
        f"/api/product-pools/{pool['id']}/members/batch",
        json={
            "revision": pool["revision"],
            "items": [
                {
                    "kind": member["kind"],
                    "product_id": member["product_id"],
                    "research_status": "approved",
                    "usage_status": "normal",
                    "primary_plan_id": member["primary_plan_id"],
                    "max_weight": 0.4,
                    "reasons": ["批量复核通过"],
                    "owner": "Kevin",
                    "review_due_date": "2027-03-31",
                    "valid_until": None,
                    "substitute_group": "核心候选",
                }
                for member in pool["members"]
            ],
        },
    )

    assert response.status_code == 200
    updated = response.json()
    assert updated["revision"] == pool["revision"] + 1
    assert len(updated["members"]) == 2
    assert all(member["research_status"] == "approved" for member in updated["members"])
    assert all(member["owner"] == "Kevin" for member in updated["members"])


def test_investable_universe_route_exposes_summary_and_search(monkeypatch, tmp_path: Path) -> None:
    client = _client(monkeypatch, tmp_path)
    pool = client.post("/api/product-pools", json={"name": "投前产品池"}).json()
    pool = client.post(
        f"/api/product-pools/{pool['id']}/evaluation-plans",
        json={
            "revision": pool["revision"],
            "plan_id": "plan-equity",
            "selection_mode": "all_ranked",
        },
    ).json()
    member = pool["members"][0]
    pool = client.put(
        f"/api/product-pools/{pool['id']}/members/batch",
        json={
            "revision": pool["revision"],
            "items": [
                {
                    "kind": member["kind"],
                    "product_id": member["product_id"],
                    "research_status": "approved",
                    "usage_status": "normal",
                    "primary_plan_id": member["primary_plan_id"],
                    "max_weight": 0.5,
                    "reasons": ["复核通过"],
                    "owner": "Kevin",
                    "review_due_date": None,
                    "valid_until": None,
                    "substitute_group": "",
                }
            ],
        },
    ).json()
    published = client.post(
        f"/api/product-pools/{pool['id']}/publish",
        json={
            "revision": pool["revision"],
            "effective_from": "2026-09-01",
            "effective_to": None,
            "publication_note": "首版",
        },
    ).json()

    created = client.post(
        "/api/investable-universe-snapshots",
        json={
            "name": "投前研究可投资域",
            "research_date": "2026-09-04",
            "version_ids": [published["version"]["id"]],
        },
    )

    assert created.status_code == 201
    snapshot = created.json()
    assert snapshot["summary"]["eligible_count"] == 1
    assert snapshot["members"][0]["evaluation_sources"][0]["source_rank"] == 1

    search = client.get(
        f"/api/investable-universes/{snapshot['id']}/products",
        params={"q": "沪深300", "eligible_only": True, "page": 1, "page_size": 10},
    )
    assert search.status_code == 200
    assert search.json()["total"] == 1
    assert search.json()["items"][0]["product_id"] == "510300.SH"


def _publish_pool_and_snapshot(client, name: str, research_date: str) -> dict:
    pool = client.post("/api/product-pools", json={"name": f"{name} 产品池"}).json()
    pool = client.post(
        f"/api/product-pools/{pool['id']}/evaluation-plans",
        json={"revision": pool["revision"], "plan_id": "plan-equity", "selection_mode": "all_ranked"},
    ).json()
    member = pool["members"][0]
    pool = client.put(
        f"/api/product-pools/{pool['id']}/members/batch",
        json={
            "revision": pool["revision"],
            "items": [{
                "kind": member["kind"], "product_id": member["product_id"],
                "research_status": "approved", "usage_status": "normal",
                "primary_plan_id": member["primary_plan_id"], "max_weight": None,
                "reasons": [], "owner": "", "review_due_date": None,
                "valid_until": None, "substitute_group": "",
            }],
        },
    ).json()
    published = client.post(
        f"/api/product-pools/{pool['id']}/publish",
        json={"revision": pool["revision"], "effective_from": "2026-09-01", "effective_to": None, "publication_note": ""},
    ).json()
    created = client.post(
        "/api/investable-universe-snapshots",
        json={"name": name, "research_date": research_date, "version_ids": [published["version"]["id"]]},
    )
    assert created.status_code == 201, created.text
    return created.json()


def test_universe_snapshot_list_returns_immutable_summaries_without_rewriting(monkeypatch, tmp_path: Path) -> None:
    client = _client(monkeypatch, tmp_path)
    first = _publish_pool_and_snapshot(client, "较早研究范围", "2026-09-04")
    second = _publish_pool_and_snapshot(client, "较新研究范围", "2026-09-05")
    store_path = tmp_path / "product_pools.json"
    stored_before = store_path.read_text(encoding="utf-8")
    full_before = client.get(f"/api/investable-universe-snapshots/{first['id']}").json()

    listed = client.get("/api/investable-universe-snapshots")

    assert listed.status_code == 200
    payload = listed.json()
    assert payload["total"] == 2
    assert {item["id"] for item in payload["items"]} == {first["id"], second["id"]}
    # Deterministic newest-first projection; ties fall back to the id so repeated
    # reads cannot reshuffle the library.
    assert payload["items"] == sorted(
        payload["items"],
        key=lambda item: (str(item["created_at"]), str(item["id"])),
        reverse=True,
    )
    summary = next(item for item in payload["items"] if item["id"] == first["id"])
    assert summary["name"] == "较早研究范围"
    assert summary["research_date"] == "2026-09-04"
    assert summary["version_ids"] == first["version_ids"]
    assert summary["product_count"] == first["product_count"] == 1
    assert summary["content_hash"] == first["content_hash"]
    assert summary["summary"]["pool_count"] == 1
    # The library is metadata only: no frozen member/product payloads.
    assert "members" not in summary and "products" not in summary and "groups" not in summary

    # Reading the library must not rewrite the frozen record or its hash.
    assert store_path.read_text(encoding="utf-8") == stored_before
    assert client.get(f"/api/investable-universe-snapshots/{first['id']}").json() == full_before


def test_universe_snapshot_list_reads_legacy_and_restricted_records_without_rewriting(monkeypatch, tmp_path: Path) -> None:
    """Legacy version_refs-only records stay readable and restricted members are
    never presented as investable counts."""

    client = _client(monkeypatch, tmp_path)
    store_path = tmp_path / "product_pools.json"
    legacy = {
        "id": "universe-legacy",
        "name": "旧版范围",
        "research_date": "2026-08-01",
        "version_refs": [{"version_id": "version-a"}, {"version_id": "version-b"}],
        "members": [
            {"eligible": True, "product_id": "510300.SH"},
            {"eligible": False, "product_id": "000001.OF"},
        ],
        "immutable": True,
        "created_at": "2026-08-01T00:00:00+00:00",
    }
    payload = json.loads(store_path.read_text(encoding="utf-8")) if store_path.exists() else {}
    payload.setdefault("pools", [])
    payload.setdefault("versions", [])
    payload.setdefault("universe_snapshots", []).append(legacy)
    store_path.write_text(json.dumps(payload), encoding="utf-8")
    stored_before = store_path.read_text(encoding="utf-8")

    listed = client.get("/api/investable-universe-snapshots")

    assert listed.status_code == 200
    item = listed.json()["items"][0]
    assert item["version_ids"] == ["version-a", "version-b"]
    assert item["product_count"] == 2
    assert "members" not in item
    # No frozen eligible_count exists for this record, so the summary must not
    # invent one; the client shows a plain product count instead.
    assert "eligible_count" not in item["summary"]
    assert store_path.read_text(encoding="utf-8") == stored_before


def test_attach_preserves_indicator_domain_error_instead_of_returning_500(
    monkeypatch,
    tmp_path: Path,
) -> None:
    client = _client(monkeypatch, tmp_path)
    pool = client.post("/api/product-pools", json={"name": "评价错误透传"}).json()
    service = product_pool_routes.product_pool_service

    def fail_run(*_args, **_kwargs):
        raise IndicatorValidationError(
            "NJIT_BATCH_PLAN_NOT_WARMED",
            "评价方案保存的融合 NJIT 计划未命中预热缓存。",
            field="plan_revision",
        )

    monkeypatch.setattr(service.evaluation_gateway, "run_plan", fail_run)
    response = client.post(
        f"/api/product-pools/{pool['id']}/evaluation-plans",
        json={
            "revision": pool["revision"],
            "plan_id": "plan-equity",
            "selection_mode": "all_ranked",
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"] == {
        "code": "NJIT_BATCH_PLAN_NOT_WARMED",
        "message": "评价方案保存的融合 NJIT 计划未命中预热缓存。",
        "field": "plan_revision",
    }


def test_attach_rejects_a_stale_pool_revision(monkeypatch, tmp_path: Path) -> None:
    client = _client(monkeypatch, tmp_path)
    pool = client.post("/api/product-pools", json={"name": "并发检查"}).json()
    updated = client.put(
        f"/api/product-pools/{pool['id']}",
        json={
            "revision": pool["revision"],
            "name": "并发检查 v2",
            "description": "",
            "purpose": "",
            "owner": "",
        },
    )
    assert updated.status_code == 200

    response = client.post(
        f"/api/product-pools/{pool['id']}/evaluation-plans",
        json={
            "revision": pool["revision"],
            "plan_id": "plan-equity",
            "selection_mode": "all_ranked",
        },
    )

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "PRODUCT_POOL_REVISION_CONFLICT"


def test_pool_plan_run_inherits_platform_pit_when_no_cutoff_given(monkeypatch) -> None:
    """An empty 评价截止日 must read under the platform口径, not to disk's last row."""

    from pit.context import build_context, reset_view_override, set_view_override

    seen: list[str | None] = []

    def record(plan_id: str, as_of: str | None = None) -> dict:
        seen.append(as_of)
        return {"plan_id": plan_id, "as_of": as_of, "rows": []}

    monkeypatch.setattr(product_pool_routes.indicator_service, "run_plan", record)
    gateway = product_pool_routes.IndicatorEvaluationGateway()
    token = set_view_override(build_context("2015-06-30"))
    try:
        gateway.run_plan("plan-equity")
        # A replay states its own research day, which must still win.
        gateway.run_plan("plan-equity", "2020-12-31")
    finally:
        reset_view_override(token)

    assert seen == ["2015-06-30", "2020-12-31"]

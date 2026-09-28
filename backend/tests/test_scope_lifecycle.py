"""Research-scope lifecycle across both stores: unique names, edit, delete."""
from __future__ import annotations

import copy
import threading
from datetime import date, timedelta
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from product_pools.errors import ProductPoolConflictError
from product_pools.repository import ProductPoolRepository
from product_pools.service import ProductPoolService
from services import product_pool_routes
from strategic_allocation.routes import build_router as build_strategic_router
from strategic_allocation.service import StrategicAllocationService
from strategic_allocation.universe_contracts import ConfirmUniverseRequest, UniverseRequest

SNAPSHOTS = "/api/investable-universe-snapshots"
STRATEGIC = "/api/strategic-allocation"


class FakeGateway:
    def get_plan(self, plan_id: str) -> dict:
        return {"id": plan_id, "revision": 1, "name": "评价方案", "product_kind": "etf"}

    def run_plan(self, plan_id: str, as_of: str | None = None) -> dict:
        return {
            "plan_id": plan_id, "plan_revision": 1, "run_at": "2026-09-04T10:00:00+00:00",
            "as_of": as_of, "ranked_count": 1, "excluded_count": 0,
            "rows": [{"rank": 1, "score": 90.0, "status": "ranked",
                      "target": {"kind": "etf", "product_id": "510300.SH", "name": "沪深300ETF"}}],
        }

    def get_run_page(self, result_id: str, *, page: int, page_size: int) -> dict:
        raise AssertionError("inline result must not request another page")


def _product_client(monkeypatch, tmp_path: Path) -> tuple[TestClient, ProductPoolService]:
    service = ProductPoolService(
        ProductPoolRepository(tmp_path / "product_pools.json"),
        FakeGateway(),
        strategic_root=tmp_path,
    )
    monkeypatch.setattr(product_pool_routes, "product_pool_service", service)
    app = FastAPI()
    app.include_router(product_pool_routes.router)
    return TestClient(app), service


def _strategic_client(tmp_path: Path) -> tuple[TestClient, StrategicAllocationService]:
    # Artifacts and the product store share one root so cross-path names are visible.
    service = StrategicAllocationService(tmp_path, tmp_path / "market", universe_dir=tmp_path)
    app = FastAPI()
    app.include_router(build_strategic_router(service))
    return TestClient(app), service


def _publish_version(client: TestClient, name: str) -> str:
    pool = client.post("/api/product-pools", json={"name": name}).json()
    pool = client.post(
        f"/api/product-pools/{pool['id']}/evaluation-plans",
        json={"revision": pool["revision"], "plan_id": "plan-equity", "selection_mode": "all_ranked"},
    ).json()
    member = pool["members"][0]
    pool = client.put(
        f"/api/product-pools/{pool['id']}/members/batch",
        json={"revision": pool["revision"], "items": [{
            "kind": member["kind"], "product_id": member["product_id"],
            "research_status": "approved", "usage_status": "normal",
            "primary_plan_id": member["primary_plan_id"], "max_weight": None,
            "reasons": [], "owner": "", "review_due_date": None,
            "valid_until": None, "substitute_group": "",
        }]},
    ).json()
    published = client.post(
        f"/api/product-pools/{pool['id']}/publish",
        json={"revision": pool["revision"], "effective_from": "2026-09-01",
              "effective_to": None, "publication_note": ""},
    ).json()
    return published["version"]["id"]


def _create_snapshot(client: TestClient, name: str, version_id: str, **extra) -> dict:
    response = client.post(SNAPSHOTS, json={
        "name": name, "research_date": "2026-09-04", "version_ids": [version_id], **extra,
    })
    assert response.status_code == 201, response.text
    return response.json()


def _universe_body(name: str, **overrides) -> dict:
    body = {
        "name": name, "as_of": str(date.today()), "currency": "CNY", "source": "",
        "assets": [{"id": "growth", "name": "增长资产", "currency": "CNY", "role": "growth",
                    "liquidity": "liquid", "rationale": "", "source": ""}],
    }
    body.update(overrides)
    return body


def _confirm_universe(client: TestClient, name: str, **extra) -> tuple[int, dict]:
    body = _universe_body(name)
    preview = client.post(f"{STRATEGIC}/universes/preview", json=body)
    assert preview.status_code == 200, preview.text
    response = client.post(f"{STRATEGIC}/universes/confirm", json={
        "request": body, "preview_hash": preview.json()["preview_hash"], **extra,
    })
    return response.status_code, response.json()


def test_product_scope_edit_replaces_active_and_preserves_frozen_bytes(monkeypatch, tmp_path: Path) -> None:
    client, service = _product_client(monkeypatch, tmp_path)
    version_id = _publish_version(client, "编辑池")
    original = _create_snapshot(client, "原范围", version_id)
    before = next(item for item in service.repository.list_universe_snapshots() if item["id"] == original["id"])
    frozen = copy.deepcopy(before)

    edited = client.post(SNAPSHOTS, json={
        "name": "原范围", "research_date": "2026-09-04", "version_ids": [version_id],
        "replaces_snapshot_id": original["id"],
    })
    assert edited.status_code == 201, edited.text
    replacement = edited.json()
    assert replacement["id"] != original["id"]

    listed = client.get(SNAPSHOTS).json()
    assert [item["id"] for item in listed["items"]] == [replacement["id"]]
    after = next(item for item in service.repository.list_universe_snapshots() if item["id"] == original["id"])
    assert after == frozen
    history = client.get(f"{SNAPSHOTS}/{original['id']}")
    assert history.status_code == 200
    assert history.json()["content_hash"] == original["content_hash"]

    stale = client.post(SNAPSHOTS, json={
        "name": "原范围", "research_date": "2026-09-04", "version_ids": [version_id],
        "replaces_snapshot_id": original["id"],
    })
    assert stale.status_code == 409
    assert stale.json()["detail"]["code"] == "INVESTABLE_UNIVERSE_INACTIVE"
    assert [item["id"] for item in client.get(SNAPSHOTS).json()["items"]] == [replacement["id"]]

    missing = client.post(SNAPSHOTS, json={
        "name": "另一个名字", "research_date": "2026-09-04", "version_ids": [version_id],
        "replaces_snapshot_id": "universe-does-not-exist",
    })
    assert missing.status_code == 404
    assert [item["id"] for item in client.get(SNAPSHOTS).json()["items"]] == [replacement["id"]]


def test_product_scope_duplicate_names_normalize_and_legacy_conflicts(monkeypatch, tmp_path: Path) -> None:
    client, service = _product_client(monkeypatch, tmp_path)
    version_id = _publish_version(client, "重名池")
    created = _create_snapshot(client, "核心范围", version_id)

    for variant in (" 核心范围 ", "核心范围"):
        response = client.post(SNAPSHOTS, json={
            "name": variant, "research_date": "2026-09-04", "version_ids": [version_id],
        })
        assert response.status_code == 409, f"{variant!r}: {response.status_code}"
        assert response.json()["detail"] == {
            "code": "SCOPE_NAME_CONFLICT", "message": "研究范围名称已存在，请修改名称。", "field": "name",
        }

    # Legacy duplicate that predates the rule still blocks a same-name edit.
    payload = service.repository.store.read_unlocked()
    legacy = copy.deepcopy(next(item for item in payload["universe_snapshots"] if item["id"] == created["id"]))
    legacy["id"] = "universe-legacy-duplicate"
    legacy["content_hash"] = "0" * 64
    payload["universe_snapshots"].append(legacy)
    service.repository.store.write_unlocked(payload)
    edit = client.post(SNAPSHOTS, json={
        "name": "核心范围", "research_date": "2026-09-04", "version_ids": [version_id],
        "replaces_snapshot_id": created["id"],
    })
    assert edit.status_code == 409


def test_latin_casefold_and_blank_names_are_rejected(tmp_path: Path) -> None:
    client, _ = _strategic_client(tmp_path)
    status, created = _confirm_universe(client, "Alpha Scope")
    assert status == 201, created

    for variant in ("alpha scope", "ALPHA SCOPE", " Alpha Scope "):
        duplicate_status, duplicate = _confirm_universe(client, variant)
        assert duplicate_status == 409, f"{variant!r}: {duplicate_status}"
        assert duplicate["detail"]["code"] == "UNIVERSE_NAME_CONFLICT"

    for blank in ("", "   ", "\t"):
        body = _universe_body(blank)
        response = client.post(f"{STRATEGIC}/universes/preview", json=body)
        assert response.status_code == 422, f"{blank!r}: {response.status_code}"
    assert len(client.get(f"{STRATEGIC}/catalog").json()["strategic_universes"]) == 1


def test_corrupt_peer_store_blocks_new_scope_instead_of_bypassing_names(monkeypatch, tmp_path: Path) -> None:
    product_client, product_service = _product_client(monkeypatch, tmp_path)
    strategic_client, _ = _strategic_client(tmp_path)
    status, created = _confirm_universe(strategic_client, "损坏前范围")
    assert status == 201, created
    manifest = tmp_path / "strategic_allocation" / "artifacts" / created["id"] / "manifest.json"
    manifest.write_text("{ not json", encoding="utf-8")

    version_id = _publish_version(product_client, "损坏检查池")
    blocked = product_client.post(SNAPSHOTS, json={
        "name": "全新名称", "research_date": "2026-09-04", "version_ids": [version_id],
    })
    assert blocked.status_code == 422, blocked.text
    assert blocked.json()["detail"]["code"] == "RESEARCH_ARTIFACT_CORRUPT"
    assert product_service.list_universe_snapshots()["total"] == 0

    index = tmp_path / "strategic_allocation" / "artifacts" / "index.json"
    index.write_text("{ also not json", encoding="utf-8")
    blocked_again = product_client.post(SNAPSHOTS, json={
        "name": "另一个新名称", "research_date": "2026-09-04", "version_ids": [version_id],
    })
    assert blocked_again.status_code == 500, blocked_again.text
    assert blocked_again.json()["detail"]["code"] == "STORAGE_CORRUPT"
    assert product_service.list_universe_snapshots()["total"] == 0


def test_separate_configured_roots_still_share_one_name_space(monkeypatch, tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    strategic_root = tmp_path / "saa"
    product_service = ProductPoolService(
        ProductPoolRepository(workspace / "product_pools.json"),
        FakeGateway(),
        strategic_root=strategic_root,
    )
    monkeypatch.setattr(product_pool_routes, "product_pool_service", product_service)
    product_app = FastAPI()
    product_app.include_router(product_pool_routes.router)
    product_client = TestClient(product_app)
    strategic_service = StrategicAllocationService(strategic_root, tmp_path / "market", universe_dir=workspace)
    strategic_app = FastAPI()
    strategic_app.include_router(build_strategic_router(strategic_service))
    strategic_client = TestClient(strategic_app)

    version_id = _publish_version(product_client, "独立根池")
    _create_snapshot(product_client, "独立根名称", version_id)
    status, detail = _confirm_universe(strategic_client, "独立根名称")
    assert status == 409, detail
    assert detail["detail"]["code"] == "UNIVERSE_NAME_CONFLICT"

    status, created = _confirm_universe(strategic_client, "独立根战略")
    assert status == 201, created
    duplicate = product_client.post(SNAPSHOTS, json={
        "name": " 独立根战略 ", "research_date": "2026-09-04", "version_ids": [version_id],
    })
    assert duplicate.status_code == 409
    assert duplicate.json()["detail"]["code"] == "SCOPE_NAME_CONFLICT"


def test_concurrent_cross_path_same_name_cannot_both_persist(monkeypatch, tmp_path: Path) -> None:
    product_service = ProductPoolService(
        ProductPoolRepository(tmp_path / "product_pools.json"), FakeGateway(), strategic_root=tmp_path,
    )
    monkeypatch.setattr(product_pool_routes, "product_pool_service", product_service)
    product_app = FastAPI()
    product_app.include_router(product_pool_routes.router)
    product_client = TestClient(product_app)
    version_id = _publish_version(product_client, "跨路径并发池")
    strategic_service = StrategicAllocationService(tmp_path, tmp_path / "market", universe_dir=tmp_path)
    strategic_app = FastAPI()
    strategic_app.include_router(build_strategic_router(strategic_service))
    strategic_client = TestClient(strategic_app)
    body = _universe_body("跨路径并发名称")
    preview = strategic_client.post(f"{STRATEGIC}/universes/preview", json=body).json()

    results: list[str] = []
    barrier = threading.Barrier(2)

    def product_worker() -> None:
        barrier.wait()
        response = product_client.post(SNAPSHOTS, json={
            "name": "跨路径并发名称", "research_date": "2026-09-04", "version_ids": [version_id],
        })
        results.append(f"product:{response.status_code}")

    def strategic_worker() -> None:
        barrier.wait()
        response = strategic_client.post(f"{STRATEGIC}/universes/confirm", json={
            "request": body, "preview_hash": preview["preview_hash"],
        })
        results.append(f"strategic:{response.status_code}")

    threads = [threading.Thread(target=product_worker), threading.Thread(target=strategic_worker)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sorted(results) in (["product:201", "strategic:409"], ["product:409", "strategic:201"]), results
    product_names = [item["name"] for item in product_client.get(SNAPSHOTS).json()["items"]]
    strategic_names = [item["name"] for item in strategic_client.get(f"{STRATEGIC}/catalog").json()["strategic_universes"]]
    assert len(product_names) + len(strategic_names) == 1


def test_corrupt_peer_product_store_is_typed_for_strategic_routes(monkeypatch, tmp_path: Path) -> None:
    product_client, _ = _product_client(monkeypatch, tmp_path)
    strategic_client, _ = _strategic_client(tmp_path)
    _publish_version(product_client, "损坏对端池")
    (tmp_path / "product_pools.json").write_text("{ not json", encoding="utf-8")

    status, detail = _confirm_universe(strategic_client, "损坏对端后的新范围")
    assert status == 422, detail
    assert detail["detail"]["code"] == "PRODUCT_POOL_STORAGE_CORRUPT"
    assert strategic_client.get(f"{STRATEGIC}/catalog").json()["strategic_universes"] == []


def test_missing_index_next_to_artifacts_fails_closed_without_rebuilding(monkeypatch, tmp_path: Path) -> None:
    product_client, product_service = _product_client(monkeypatch, tmp_path)
    client, _ = _strategic_client(tmp_path)
    status, created = _confirm_universe(client, "索引前范围")
    assert status == 201, created
    index = tmp_path / "strategic_allocation" / "artifacts" / "index.json"
    index.unlink()

    status, detail = _confirm_universe(client, "索引缺失后的新范围")
    assert status == 422, detail
    assert detail["detail"]["code"] == "RESEARCH_INDEX_MISSING"
    assert not index.exists()
    assert client.delete(f"{STRATEGIC}/universes/{created['id']}").status_code == 422
    assert not index.exists()

    version_id = _publish_version(product_client, "索引缺失产品池")
    blocked = product_client.post(SNAPSHOTS, json={
        "name": "产品侧新范围", "research_date": "2026-09-04", "version_ids": [version_id],
    })
    assert blocked.status_code == 422, blocked.text
    assert blocked.json()["detail"]["code"] == "RESEARCH_INDEX_MISSING"
    assert product_service.list_universe_snapshots()["total"] == 0
    assert not index.exists()


def test_strategic_legacy_duplicate_requires_rename(tmp_path: Path) -> None:
    client, service = _strategic_client(tmp_path)
    status, created = _confirm_universe(client, "战略范围甲")
    assert status == 201, created
    universe = service.scopes.get_universe(created["id"])
    legacy_fields = {
        key: value
        for key, value in universe.items()
        if key not in {"id", "kind", "schema_version", "created_at", "immutable", "arrays", "content_hash"}
    }
    service.artifacts.save("series", legacy_fields)

    status, detail = _confirm_universe(client, "战略范围甲", replaces_universe_id=created["id"])
    assert status == 409, detail
    assert detail["detail"]["code"] == "UNIVERSE_NAME_CONFLICT"

    status, renamed = _confirm_universe(client, "战略范围甲改名", replaces_universe_id=created["id"])
    assert status == 201, renamed
    assert renamed["id"] != created["id"]


def test_concurrent_strategic_same_name_cannot_both_persist(tmp_path: Path) -> None:
    _, service = _strategic_client(tmp_path)
    request = UniverseRequest(**_universe_body("战略并发名称"))
    preview = service.scopes.preview_universe(request)
    body = ConfirmUniverseRequest(request=request, preview_hash=preview["preview_hash"])
    results: list[object] = []
    barrier = threading.Barrier(2)

    def worker() -> None:
        barrier.wait()
        try:
            results.append(service.scopes.confirm_universe(body))
        except Exception as exc:  # noqa: BLE001 - the conflict type is asserted below
            results.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sum(isinstance(result, dict) for result in results) == 1
    assert sum(type(result).__name__ == "ConflictError" for result in results) == 1
    assert len(service.scopes.active_universe_ids()) == 1




def test_product_scope_delete_is_idempotent_and_history_stays_readable(monkeypatch, tmp_path: Path) -> None:
    client, service = _product_client(monkeypatch, tmp_path)
    version_id = _publish_version(client, "删除池")
    created = _create_snapshot(client, "待删除范围", version_id)
    frozen = copy.deepcopy(next(item for item in service.repository.list_universe_snapshots() if item["id"] == created["id"]))

    assert client.delete(f"{SNAPSHOTS}/{created['id']}").json() == {"deleted": True, "id": created["id"]}
    assert client.get(SNAPSHOTS).json()["total"] == 0
    assert client.get(f"{SNAPSHOTS}/{created['id']}").status_code == 200
    assert client.delete(f"{SNAPSHOTS}/{created['id']}").json() == {"deleted": True, "id": created["id"]}
    after = next(item for item in service.repository.list_universe_snapshots() if item["id"] == created["id"])
    assert after == frozen
    assert client.delete(f"{SNAPSHOTS}/universe-missing").status_code == 404


def test_concurrent_same_name_creates_cannot_both_persist(monkeypatch, tmp_path: Path) -> None:
    client, service = _product_client(monkeypatch, tmp_path)
    version_id = _publish_version(client, "并发池")
    results: list[object] = []
    barrier = threading.Barrier(2)

    def worker() -> None:
        barrier.wait()
        try:
            results.append(service.create_universe_snapshot({
                "name": "并发范围", "research_date": "2026-09-04", "version_ids": [version_id],
            }))
        except ProductPoolConflictError as exc:
            results.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sum(isinstance(result, dict) for result in results) == 1
    assert sum(isinstance(result, ProductPoolConflictError) for result in results) == 1
    assert service.list_universe_snapshots()["total"] == 1


def test_cross_path_names_conflict_in_both_directions(monkeypatch, tmp_path: Path) -> None:
    product_client, _ = _product_client(monkeypatch, tmp_path)
    strategic_client, _ = _strategic_client(tmp_path)
    version_id = _publish_version(product_client, "跨路径池")
    _create_snapshot(product_client, "共享名称", version_id)

    status, detail = _confirm_universe(strategic_client, "共享名称")
    assert status == 409, detail
    assert detail["detail"]["code"] == "UNIVERSE_NAME_CONFLICT"

    status, created = _confirm_universe(strategic_client, "战略名称")
    assert status == 201, created
    product_duplicate = product_client.post(SNAPSHOTS, json={
        "name": " 战略名称 ", "research_date": "2026-09-04", "version_ids": [version_id],
    })
    assert product_duplicate.status_code == 409
    assert product_duplicate.json()["detail"]["code"] == "SCOPE_NAME_CONFLICT"


def test_strategic_scope_edit_delete_and_stale_conflicts(tmp_path: Path) -> None:
    client, service = _strategic_client(tmp_path)
    status, created = _confirm_universe(client, "战略范围甲")
    assert status == 201, created
    original_hash = service.scopes.get_universe(created["id"])["content_hash"]

    duplicate_status, duplicate = _confirm_universe(client, " 战略范围甲 ")
    assert duplicate_status == 409
    assert duplicate["detail"]["code"] == "UNIVERSE_NAME_CONFLICT"

    status, unchanged = _confirm_universe(client, "战略范围甲", replaces_universe_id=created["id"])
    assert unchanged["id"] == created["id"]
    status, replacement = _confirm_universe(client, "战略范围乙", replaces_universe_id=created["id"])
    assert status == 201, replacement
    assert replacement["id"] != created["id"]
    assert service.scopes.get_universe(created["id"])["content_hash"] == original_hash
    catalog = client.get(f"{STRATEGIC}/catalog").json()
    assert [item["id"] for item in catalog["strategic_universes"]] == [replacement["id"]]

    stale_status, stale = _confirm_universe(client, "战略范围甲", replaces_universe_id=created["id"])
    assert stale_status == 409
    assert stale["detail"]["code"] == "SAA_UNIVERSE_INACTIVE"

    assert client.delete(f"{STRATEGIC}/universes/{replacement['id']}").json() == {
        "deleted": True, "id": replacement["id"],
    }
    assert client.get(f"{STRATEGIC}/catalog").json()["strategic_universes"] == []
    assert client.get(f"{STRATEGIC}/universes/{replacement['id']}").status_code == 200
    assert client.delete(f"{STRATEGIC}/universes/{replacement['id']}").status_code == 200
    assert client.delete(f"{STRATEGIC}/universes/strategic-missing").status_code == 404


def test_failed_replacement_keeps_active_library_intact(tmp_path: Path) -> None:
    client, service = _strategic_client(tmp_path)
    status, created = _confirm_universe(client, "失败不丢范围")
    assert status == 201, created
    body = _universe_body("失败不丢范围")
    preview = client.post(f"{STRATEGIC}/universes/preview", json=body).json()
    failed = client.post(f"{STRATEGIC}/universes/confirm", json={
        "request": body, "preview_hash": "f" * 64, "replaces_universe_id": created["id"],
    })
    assert failed.status_code == 409
    assert [item["id"] for item in client.get(f"{STRATEGIC}/catalog").json()["strategic_universes"]] == [created["id"]]
    assert preview["preview_hash"]


def test_scope_mandates_persist_in_both_paths_and_edits_inherit(monkeypatch, tmp_path):
    product, products = _product_client(monkeypatch, tmp_path)
    strategic, service = _strategic_client(tmp_path)
    mandate = service.artifacts.save('series', {'artifact_type': 'investment_mandate', 'name': '目标甲', 'definition': {}})
    other = service.artifacts.save('series', {'artifact_type': 'investment_mandate', 'name': '目标乙', 'definition': {}})
    version_id = _publish_version(product, '绑定测试池')
    snapshot = _create_snapshot(product, '产品范围绑定', version_id, mandate_id=mandate['id'])
    status, universe = _confirm_universe(strategic, '战略范围绑定', mandate_id=mandate['id'])
    assert status == 201
    for record in [snapshot, universe]:
        assert record['mandate_id'] == mandate['id']
        assert record['mandate_hash'] == mandate['content_hash']
    assert product.get(SNAPSHOTS).json()['items'][0]['mandate_id'] == mandate['id']
    assert strategic.get(f'{STRATEGIC}/catalog').json()['strategic_universes'][0]['mandate_id'] == mandate['id']
    reloaded = StrategicAllocationService(tmp_path, tmp_path / 'market', universe_dir=tmp_path)
    assert reloaded.scopes.get_universe(universe['id'])['mandate_id'] == mandate['id']
    replacement = _create_snapshot(product, '产品范围绑定', version_id, replaces_snapshot_id=snapshot['id'])
    status, updated = _confirm_universe(strategic, '战略范围绑定', replaces_universe_id=universe['id'])
    assert status == 201
    assert replacement['mandate_id'] == updated['mandate_id'] == mandate['id']
    assert product.post(SNAPSHOTS, json={'name': replacement['name'], 'research_date': '2026-09-04', 'version_ids': [version_id], 'replaces_snapshot_id': replacement['id'], 'mandate_id': other['id']}).status_code == 409
    assert _confirm_universe(strategic, updated['name'], replaces_universe_id=updated['id'], mandate_id=other['id'])[0] == 409


def test_legacy_scope_binding_is_once_only_and_preserves_frozen_bytes(monkeypatch, tmp_path):
    product, products = _product_client(monkeypatch, tmp_path)
    strategic, service = _strategic_client(tmp_path)
    mandate = service.artifacts.save('series', {'artifact_type': 'investment_mandate', 'name': '旧研究目标', 'definition': {}})
    other = service.artifacts.save('series', {'artifact_type': 'investment_mandate', 'name': '另一个目标', 'definition': {}})
    version_id = _publish_version(product, '旧范围池')
    snapshot = _create_snapshot(product, '旧产品范围', version_id)
    _, universe = _confirm_universe(strategic, '旧战略范围')
    originals = [copy.deepcopy(products.repository.get_universe_snapshot(snapshot['id'])), copy.deepcopy(service.artifacts.get(universe['id'], 'series'))]
    for client, path, record in [(product, SNAPSHOTS, snapshot), (strategic, f'{STRATEGIC}/universes', universe)]:
        endpoint = f"{path}/{record['id']}/mandate"
        invalid = client.post(endpoint, json={'mandate_id': universe['id']})
        assert invalid.status_code == 422, invalid.text
        first = client.post(endpoint, json={'mandate_id': mandate['id']})
        assert first.status_code == 200, first.text
        assert client.post(endpoint, json={'mandate_id': mandate['id']}).json() == first.json()
        assert client.post(endpoint, json={'mandate_id': other['id']}).status_code == 409
        restored = client.get(f"{path}/{record['id']}").json()
        assert restored['mandate_id'] == mandate['id']
        assert restored['content_hash'] == record['content_hash']
    assert products.repository.get_universe_snapshot(snapshot['id']) == originals[0]
    assert service.artifacts.get(universe['id'], 'series') == originals[1]
    assert product.get(SNAPSHOTS).json()['items'][0]['mandate_id'] == mandate['id']
    assert strategic.get(f'{STRATEGIC}/catalog').json()['strategic_universes'][0]['mandate_id'] == mandate['id']

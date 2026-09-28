"""Saved-plan navigation reads frozen lineage, independently of current inputs."""
from copy import deepcopy

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from backend.strategic_allocation.policy_catalog import policy_summary
from backend.strategic_allocation.routes import build_router
from backend.strategic_allocation.service import StrategicAllocationService


def saved_fields():
    return {
        "name": "稳健配置", "as_of": "2019-12-31", "alloc_name": "产品大类",
        "universe_snapshot_id": "product-scope",
        "policy": {
            "mandate_id": "objective", "cma_id": "cma-a",
            "mandate": {"name": "三年目标", "target_return": .07, "max_volatility": .095,
                        "min_cash_weight": .1},
            "assumptions": {"name": "历史统计"},
        },
    }


def test_saved_list_route_uses_frozen_records_and_excludes_non_policy_baselines(tmp_path, monkeypatch):
    service = StrategicAllocationService(tmp_path, tmp_path / "data")
    app = FastAPI()
    app.include_router(build_router(service))
    client = TestClient(app)
    assert client.get("/api/strategic-allocation/policies").json() == {"items": []}
    saved = service.baselines.save_baseline(saved_fields())
    service.baselines.save_baseline({"name": "独立回测基线", "as_of": "2019-12-31"})

    def unavailable(*args, **kwargs):
        raise AssertionError("Saved lineage must not query current research inputs")

    monkeypatch.setattr(service, "catalog", unavailable)
    monkeypatch.setattr(service, "get_cma", unavailable)
    monkeypatch.setattr(service, "get_mandate", unavailable)
    result = client.get("/api/strategic-allocation/policies")
    assert result.status_code == 200
    [item] = result.json()["items"]
    lineage = {key: item.pop(key) for key in ("version", "upstream", "usable")}
    assert item == {
        "id": saved["id"], "name": saved["name"], "as_of": saved["as_of"],
        "created_at": saved["created_at"], "mode": "single",
        "mandate": {"id": "objective", "name": "三年目标", "definition": saved["policy"]["mandate"]},
        "scope": {"research_path": "product_first", "id": "product-scope", "name": "产品大类"},
        "cmas": [{"id": "cma-a", "name": "历史统计"}],
    }
    # 冻结的目标与 LTCMA 已不存在：方案仍可查看，但不能接入新的下游工作。
    assert lineage["version"]["number"] == 1
    assert [(ref["kind"], ref["id"], ref["status"]) for ref in lineage["upstream"]] == [
        ("mandate", "objective", "missing"), ("cma", "cma-a", "missing")]
    assert lineage["usable"]["status"] == "blocked"
    assert service.baselines.get_baseline(saved["id"]) == saved


@pytest.mark.parametrize("mode", ["compatible_all_models", "parameter_average"])
def test_multiple_sources_and_strategic_scope_preserve_frozen_names(mode):
    record = {**saved_fields(), "id": "saa", "created_at": "2026-09-23",
              "strategic_universe_id": "strategic-scope",
              "strategic_universe_snapshot": {"definition": {"name": "三大类范围"}}}
    record["policy"].update(mode=mode, cma_id=None, multi_cma={"sources": [
        {"cma_id": "cma-a", "name": "历史统计"}, {"cma_id": "cma-b", "name": "长期情景"}]})
    original = deepcopy(record)
    result = policy_summary(record)
    assert result["mode"] == mode
    assert result["scope"] == {"research_path": "strategy_first", "id": "strategic-scope", "name": "三大类范围"}
    assert result["cmas"] == [{"id": "cma-a", "name": "历史统计"}, {"id": "cma-b", "name": "长期情景"}]
    result["mandate"]["definition"]["name"] = "changed"
    assert record == original


def test_legacy_missing_names_remain_missing():
    result = policy_summary({"id": "legacy", "name": "旧方案", "as_of": "2019-12-31",
                             "created_at": "2026-09-23", "policy": {"mandate_id": "m", "cma_id": "c"}})
    assert result["mandate"]["name"] is None
    assert result["scope"]["name"] is None
    assert result["cmas"] == [{"id": "c", "name": None}]

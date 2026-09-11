from __future__ import annotations

import copy
from pathlib import Path

import pytest

from product_pools.errors import ProductPoolConflictError, ProductPoolValidationError
from product_pools.repository import ProductPoolRepository
from product_pools.service import ProductPoolService


class FakeEvaluationGateway:
    def __init__(self) -> None:
        self.plans = {
            "plan-equity": {
                "id": "plan-equity",
                "revision": 3,
                "name": "权益 ETF 评价",
                "product_kind": "etf",
            },
            "plan-bond": {
                "id": "plan-bond",
                "revision": 2,
                "name": "固收基金评价",
                "product_kind": "fund",
            },
            "plan-factor": {
                "id": "plan-factor",
                "revision": 1,
                "name": "权益因子评价",
                "product_kind": "etf",
            },
        }
        self.runs = {
            "plan-equity": {
                "result_id": "run-equity",
                "as_of": "2026-08-31",
                "ranked_count": 2,
                "excluded_count": 1,
                "total": 3,
                "page_size": 500,
                "rows": [
                    {"kind": "etf", "product_id": "510300.SH", "code": "510300.SH", "name": "沪深300ETF", "rank": 1, "score": 91.2},
                    {"kind": "etf", "product_id": "510500.SH", "code": "510500.SH", "name": "中证500ETF", "rank": 2, "score": 82.4},
                    {"kind": "etf", "product_id": "159999.SZ", "code": "159999.SZ", "name": "样本不足ETF", "rank": None, "score": None, "exclusion_reason": "样本不足"},
                ],
            },
            "plan-bond": {
                "result_id": "run-bond",
                "as_of": "2026-08-31",
                "ranked_count": 1,
                "excluded_count": 0,
                "total": 1,
                "rows": [
                    {"target": {"kind": "fund", "product_id": "000001.OF", "name": "稳健债券基金"}, "rank": 1, "score": 88.0},
                ],
            },
            "plan-factor": {
                "result_id": "run-factor",
                "as_of": "2026-08-31",
                "ranked_count": 1,
                "excluded_count": 0,
                "total": 1,
                "rows": [
                    {"kind": "etf", "product_id": "510300.SH", "code": "510300.SH", "name": "沪深300ETF", "rank": 1, "score": 77.0},
                ],
            },
        }

    def get_plan(self, plan_id: str):
        return copy.deepcopy(self.plans[plan_id])

    def run_plan(self, plan_id: str, as_of: str | None = None):
        result = copy.deepcopy(self.runs[plan_id])
        if as_of:
            result["as_of"] = as_of
        return result

    def get_run_page(self, result_id: str, *, page: int, page_size: int):
        return {"result_id": result_id, "rows": [], "total": 0, "page": page, "page_size": page_size}


@pytest.fixture()
def service(tmp_path: Path) -> ProductPoolService:
    return ProductPoolService(
        ProductPoolRepository(tmp_path / "product_pools.json"),
        FakeEvaluationGateway(),
    )


def _approve_all(service: ProductPoolService, pool: dict) -> dict:
    current = pool
    for member in list(current["members"]):
        current = service.update_member(
            current["id"],
            current["revision"],
            member["kind"],
            member["product_id"],
            {
                "research_status": "approved",
                "usage_status": "normal",
                "primary_plan_id": member["primary_plan_id"],
                "max_weight": 0.4,
                "reasons": ["定量评价通过并完成人工复核"],
                "owner": "analyst",
                "review_due_date": "2027-02-28",
                "valid_until": None,
                "substitute_group": "",
            },
        )
    return current


def test_pool_directly_contains_evaluation_plans_and_publishes_snapshot(service: ProductPoolService):
    pool = service.create_pool({"name": "核心产品池", "description": "", "purpose": "长期配置", "owner": "Kevin"})
    pool = service.attach_evaluation_plan(
        pool["id"],
        pool["revision"],
        {"plan_id": "plan-equity", "selection_mode": "top_n", "selection_value": 1},
    )
    pool = service.attach_evaluation_plan(
        pool["id"],
        pool["revision"],
        {"plan_id": "plan-bond", "selection_mode": "all_ranked"},
    )

    assert [item["plan_id"] for item in pool["evaluation_plans"]] == ["plan-equity", "plan-bond"]
    assert {member["key"] for member in pool["members"]} == {"etf:510300.SH", "fund:000001.OF"}
    assert all("segment_id" not in member and "group_id" not in member for member in pool["members"])

    pool = _approve_all(service, pool)
    published = service.publish_pool(
        pool["id"],
        pool["revision"],
        {"effective_from": "2026-09-01", "publication_note": "首版"},
    )
    version = published["version"]
    assert version["immutable"] is True
    assert version["version"] == 1
    assert version["investable_count"] == 2
    assert [item["plan_name"] for item in version["evaluation_plans"]] == ["权益 ETF 评价", "固收基金评价"]

    universe = service.create_universe_snapshot(
        {
            "name": "2026年9月投前研究域",
            "research_date": "2026-09-04",
            "version_ids": [version["id"]],
            "excluded_product_keys": [],
        }
    )
    assert universe["immutable"] is True
    assert universe["product_count"] == 2
    assert {item["evaluation_plan_name"] for item in universe["groups"]} == {"权益 ETF 评价", "固收基金评价"}


def test_overlapping_plan_evidence_keeps_one_primary_plan(service: ProductPoolService):
    pool = service.create_pool({"name": "重叠测试", "description": "", "purpose": "", "owner": ""})
    pool = service.attach_evaluation_plan(pool["id"], pool["revision"], {"plan_id": "plan-equity"})
    pool = service.attach_evaluation_plan(pool["id"], pool["revision"], {"plan_id": "plan-factor"})
    member = next(item for item in pool["members"] if item["product_id"] == "510300.SH")
    assert member["primary_plan_id"] == "plan-equity"
    assert {item["plan_id"] for item in member["evidences"]} == {"plan-equity", "plan-factor"}


def test_inline_evaluation_result_without_result_id_can_be_attached(service: ProductPoolService):
    gateway = service.evaluation_gateway
    gateway.runs["plan-equity"].pop("result_id")
    pool = service.create_pool({"name": "内联结果", "description": "", "purpose": "", "owner": ""})

    pool = service.attach_evaluation_plan(
        pool["id"],
        pool["revision"],
        {"plan_id": "plan-equity", "selection_mode": "all_ranked"},
    )

    assert len(pool["evaluation_plans"]) == 1
    assert pool["evaluation_plans"][0]["result_id"].startswith("inline-")
    assert pool["evaluation_plans"][0]["imported_count"] == 2


def test_attached_plans_can_be_removed_independently(service: ProductPoolService):
    pool = service.create_pool({"name": "多方案", "description": "", "purpose": "", "owner": ""})
    pool = service.attach_evaluation_plan(pool["id"], pool["revision"], {"plan_id": "plan-equity"})
    pool = service.add_manual_member(
        pool["id"],
        pool["revision"],
        {
            "plan_id": "plan-equity",
            "kind": "etf",
            "product_id": "512000.SH",
            "name": "券商ETF",
            "reason": "人工补充行业暴露",
        },
    )
    pool = service.attach_evaluation_plan(pool["id"], pool["revision"], {"plan_id": "plan-bond"})

    pool = service.remove_evaluation_plan(pool["id"], pool["revision"], "plan-equity")

    assert [item["plan_id"] for item in pool["evaluation_plans"]] == ["plan-bond"]
    assert {member["key"] for member in pool["members"]} == {"fund:000001.OF"}


def test_manual_exception_must_belong_to_attached_plan(service: ProductPoolService):
    pool = service.create_pool({"name": "人工例外", "description": "", "purpose": "", "owner": ""})
    with pytest.raises(ProductPoolValidationError, match="必须归属于"):
        service.add_manual_member(
            pool["id"],
            pool["revision"],
            {"plan_id": "plan-equity", "kind": "etf", "product_id": "512000.SH", "reason": "补充"},
        )

    pool = service.attach_evaluation_plan(pool["id"], pool["revision"], {"plan_id": "plan-equity", "selection_mode": "top_n", "selection_value": 1})
    pool = service.add_manual_member(
        pool["id"],
        pool["revision"],
        {"plan_id": "plan-equity", "kind": "etf", "product_id": "512000.SH", "name": "券商ETF", "reason": "用于补充行业暴露"},
    )
    member = next(item for item in pool["members"] if item["product_id"] == "512000.SH")
    assert member["manual_exception"] is True
    assert member["primary_plan_id"] == "plan-equity"


def test_publish_rejects_pending_members(service: ProductPoolService):
    pool = service.create_pool({"name": "待审产品池", "description": "", "purpose": "", "owner": ""})
    pool = service.attach_evaluation_plan(pool["id"], pool["revision"], {"plan_id": "plan-bond"})
    with pytest.raises(ProductPoolValidationError, match="未完成人工复核"):
        service.publish_pool(pool["id"], pool["revision"], {"effective_from": "2026-09-01"})


def test_member_batch_update_is_atomic_and_advances_one_revision(service: ProductPoolService):
    pool = service.create_pool({"name": "批量复核", "description": "", "purpose": "", "owner": ""})
    pool = service.attach_evaluation_plan(
        pool["id"],
        pool["revision"],
        {"plan_id": "plan-equity", "selection_mode": "all_ranked"},
    )
    start_revision = pool["revision"]
    updates = [
        {
            "kind": member["kind"],
            "product_id": member["product_id"],
            "research_status": "approved",
            "usage_status": "normal",
            "primary_plan_id": member["primary_plan_id"],
            "max_weight": 0.3,
            "reasons": ["批量复核通过"],
            "owner": "Kevin",
            "review_due_date": "2027-03-31",
            "valid_until": None,
            "substitute_group": "沪深宽基",
        }
        for member in pool["members"]
    ]

    updated = service.update_members(pool["id"], start_revision, updates)

    assert updated["revision"] == start_revision + 1
    assert all(member["research_status"] == "approved" for member in updated["members"])
    assert all(member["owner"] == "Kevin" for member in updated["members"])
    assert all(member["substitute_group"] == "沪深宽基" for member in updated["members"])


def test_invalid_member_batch_does_not_partially_persist(service: ProductPoolService):
    pool = service.create_pool({"name": "批量回滚", "description": "", "purpose": "", "owner": ""})
    pool = service.attach_evaluation_plan(
        pool["id"],
        pool["revision"],
        {"plan_id": "plan-equity", "selection_mode": "all_ranked"},
    )
    updates = []
    for index, member in enumerate(pool["members"]):
        updates.append(
            {
                "kind": member["kind"],
                "product_id": member["product_id"],
                "research_status": "approved",
                "usage_status": "normal",
                "primary_plan_id": member["primary_plan_id"],
                "max_weight": 2.0 if index == 1 else 0.3,
                "reasons": ["批量复核通过"],
                "owner": "Kevin",
                "review_due_date": None,
                "valid_until": None,
                "substitute_group": "",
            }
        )

    with pytest.raises(ProductPoolValidationError) as error:
        service.update_members(pool["id"], pool["revision"], updates)

    assert error.value.code == "INVALID_MAX_WEIGHT"
    current = service.get_pool(pool["id"])
    assert current["revision"] == pool["revision"]
    assert all(member["research_status"] == "pending" for member in current["members"])


def test_universe_snapshot_supports_downstream_summary_and_product_search(
    service: ProductPoolService,
):
    pool = service.create_pool(
        {"name": "投前产品池", "description": "", "purpose": "", "owner": ""}
    )
    pool = service.attach_evaluation_plan(
        pool["id"],
        pool["revision"],
        {"plan_id": "plan-equity", "selection_mode": "all_ranked"},
    )
    pool = _approve_all(service, pool)
    published = service.publish_pool(
        pool["id"],
        pool["revision"],
        {"effective_from": "2026-09-01"},
    )

    snapshot = service.create_universe_snapshot(
        {
            "name": "投前研究可投资域",
            "research_date": "2026-09-04",
            "version_ids": [published["version"]["id"]],
        }
    )

    assert snapshot["product_count"] == 2
    assert snapshot["summary"] == {
        "pool_count": 1,
        "member_count": 2,
        "eligible_count": 2,
        "restricted_count": 0,
        "watch_count": 0,
    }
    member = next(
        item for item in snapshot["members"] if item["product_id"] == "510300.SH"
    )
    assert member["eligible"] is True
    assert member["evaluation_sources"][0]["pool_name"] == "投前产品池"
    assert member["evaluation_sources"][0]["source_rank"] == 1
    assert member["evaluation_sources"][0]["source_score"] == 91.2

    loaded = service.get_universe_snapshot(snapshot["id"])
    assert loaded["summary"]["eligible_count"] == 2

    search = service.search_universe_products(
        snapshot["id"],
        query="沪深300",
        eligible_only=True,
        page=1,
        page_size=10,
    )
    assert search["total"] == 1
    assert search["items"][0]["product_id"] == "510300.SH"
    assert (
        search["items"][0]["evaluation_sources"][0]["evaluation_plan_name"]
        == "权益 ETF 评价"
    )


def test_repository_uses_optimistic_revision(service: ProductPoolService):
    pool = service.create_pool({"name": "并发测试", "description": "", "purpose": "", "owner": ""})
    updated = service.update_pool(pool["id"], pool["revision"], {"name": "新名称", "description": "", "purpose": "", "owner": ""})
    assert updated["revision"] == 2
    with pytest.raises(ProductPoolConflictError):
        service.update_pool(pool["id"], pool["revision"], {"name": "旧写入", "description": "", "purpose": "", "owner": ""})

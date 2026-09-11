from __future__ import annotations

import copy
from pathlib import Path

from product_pools.repository import ProductPoolRepository
from product_pools.service import ProductPoolService


class FakeEvaluationGateway:
    def __init__(self) -> None:
        self.score = 80.0
        self.run_number = 1

    def get_plan(self, plan_id: str) -> dict:
        return {
            "id": plan_id,
            "revision": 3,
            "name": "权益评价方案",
            "product_kind": "etf",
        }

    def run_plan(self, plan_id: str, as_of: str | None = None) -> dict:
        return {
            "result_id": f"run-{self.run_number}",
            "run_at": "2026-09-04T10:00:00+00:00",
            "as_of": as_of,
            "ranked_count": 1,
            "excluded_count": 0,
            "rows": [{
                "rank": 1,
                "score": self.score,
                "status": "ranked",
                "target": {
                    "kind": "etf",
                    "product_id": "510300.SH",
                    "name": "沪深300ETF",
                },
            }],
        }

    def get_run_page(self, result_id: str, *, page: int, page_size: int) -> dict:
        raise AssertionError("inline fixture should not paginate")


def _service(path: Path, gateway: FakeEvaluationGateway | None = None) -> ProductPoolService:
    return ProductPoolService(
        ProductPoolRepository(path / "product_pools.json"),
        gateway or FakeEvaluationGateway(),
    )


def test_product_pool_draft_persists_across_service_restart(tmp_path: Path) -> None:
    service = _service(tmp_path)
    pool = service.create_pool({"name": "核心产品池", "description": "", "purpose": "", "owner": ""})
    pool = service.attach_evaluation_plan(pool["id"], pool["revision"], {"plan_id": "plan-equity"})

    restarted = _service(tmp_path)
    restored = restarted.get_pool(pool["id"])

    assert restored["revision"] == pool["revision"]
    assert [item["plan_id"] for item in restored["evaluation_plans"]] == ["plan-equity"]
    assert restored["members"][0]["primary_plan_id"] == "plan-equity"


def test_rerunning_an_attached_plan_replaces_its_evidence_without_duplicates(tmp_path: Path) -> None:
    gateway = FakeEvaluationGateway()
    service = _service(tmp_path, gateway)
    pool = service.create_pool({"name": "核心产品池", "description": "", "purpose": "", "owner": ""})
    pool = service.attach_evaluation_plan(pool["id"], pool["revision"], {"plan_id": "plan-equity"})

    gateway.score = 92.5
    gateway.run_number = 2
    pool = service.attach_evaluation_plan(pool["id"], pool["revision"], {"plan_id": "plan-equity"})

    assert len(pool["evaluation_plans"]) == 1
    assert pool["evaluation_plans"][0]["result_id"] == "run-2"
    member = copy.deepcopy(pool["members"][0])
    assert len(member["evidences"]) == 1
    assert member["evidences"][0]["score"] == 92.5

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from product_pools.errors import ProductPoolValidationError
from product_pools.repository import ProductPoolRepository
from services import product_pool_routes
from services.product_pool_review import ProductPoolReviewDataService


def _repository(tmp_path: Path) -> tuple[ProductPoolRepository, dict]:
    repository = ProductPoolRepository(tmp_path / "product_pools.json")
    pool = repository.create_pool(
        {
            "name": "复核展示池",
            "description": "",
            "purpose": "",
            "owner": "",
            "state": "draft",
            "evaluation_plans": [],
            "members": [
                {
                    "key": "etf:510300.SH",
                    "kind": "etf",
                    "product_id": "510300.SH",
                    "code": "510300.SH",
                    "name": "沪深300ETF",
                },
                {
                    "key": "etf:510500.SH",
                    "kind": "etf",
                    "product_id": "510500.SH",
                    "code": "510500.SH",
                    "name": "中证500ETF",
                },
            ],
        }
    )
    return repository, pool


def _service(tmp_path: Path) -> tuple[ProductPoolReviewDataService, dict]:
    repository, pool = _repository(tmp_path)
    pd.DataFrame(
        [
            {
                "ts_code": "510300.SH",
                "code": "510300",
                "management": "甲基金",
                "fund_type": "股票型",
                "issue_amount": 52_155.0,
            },
            {
                "ts_code": "510500.SH",
                "code": "510500",
                "management": "乙基金",
                "fund_type": "股票型",
                "issue_amount": 9_800.0,
            },
        ]
    ).to_parquet(tmp_path / "etf_info_df.parquet", index=False)

    def snapshot_loader(kind: str, data_dir: Path):
        assert kind == "etf"
        assert data_dir == tmp_path
        return (
            pd.DataFrame(
                [
                    {"ts_code": "510300.SH", "return_1y": 0.12, "sharpe_1y": 0.8, "as_of": "2026-09-03"},
                    {"ts_code": "510500.SH", "return_1y": 0.08, "sharpe_1y": 0.6, "as_of": "2026-09-03"},
                ]
            ),
            {"status": "ready", "as_of": "2026-09-03", "metric_availability": {}},
        )

    def definition_loader(data_dir: Path):
        assert data_dir == tmp_path
        return {
            "return_1y": {
                "label": "近1年收益率",
                "unit": "ratio",
                "source": "adj_nav",
                "metric_type": "return",
                "metric_type_label": "收益型指标",
                "applicable_product_kinds": ["etf", "fund"],
            },
            "sharpe_1y": {
                "label": "近1年夏普比率",
                "unit": "ratio",
                "source": "adj_nav",
                "metric_type": "risk_adjusted",
                "metric_type_label": "收益风险性价比指标",
                "applicable_product_kinds": ["etf", "fund"],
            },
        }

    return (
        ProductPoolReviewDataService(
            repository,
            tmp_path,
            snapshot_loader=snapshot_loader,
            definition_loader=definition_loader,
        ),
        pool,
    )


def test_review_data_joins_selected_basic_and_snapshot_columns(tmp_path: Path) -> None:
    service, pool = _service(tmp_path)

    result = service.get_review_data(
        pool["id"],
        basic_fields=["management", "issue_amount"],
        snapshot_metrics=["return_1y", "sharpe_1y"],
    )

    assert result["pool_revision"] == pool["revision"]
    assert [row["key"] for row in result["rows"]] == ["etf:510300.SH", "etf:510500.SH"]
    first = result["rows"][0]
    assert first["basic_values"] == {"management": "甲基金", "issue_amount": 52_155.0}
    assert first["snapshot_values"] == {"return_1y": 0.12, "sharpe_1y": 0.8}
    assert first["snapshot_value_dates"]["return_1y"] == "2026-09-03"
    fields = {item["field"]: item for item in result["snapshot_metric_fields"]}
    assert fields["return_1y"]["display_format"] == "percent"
    assert fields["sharpe_1y"]["display_format"] == "number"


def test_review_data_rejects_unknown_or_excess_columns(tmp_path: Path) -> None:
    service, pool = _service(tmp_path)

    with pytest.raises(ProductPoolValidationError) as unknown:
        service.get_review_data(pool["id"], basic_fields=["unknown"])
    assert unknown.value.code == "INVALID_REVIEW_COLUMN"

    with pytest.raises(ProductPoolValidationError) as excess:
        service.get_review_data(
            pool["id"],
            basic_fields=[item["field"] for item in service.get_review_data(pool["id"])["basic_fields"][:11]],
        )
    assert excess.value.code == "TOO_MANY_REVIEW_COLUMNS"


def test_review_data_route_returns_column_catalog_and_values(monkeypatch, tmp_path: Path) -> None:
    service, pool = _service(tmp_path)
    monkeypatch.setattr(product_pool_routes, "product_pool_review_service", service)
    app = FastAPI()
    app.include_router(product_pool_routes.router)
    client = TestClient(app)

    response = client.get(
        f"/api/product-pools/{pool['id']}/review-data",
        params=[("basic_field", "management"), ("snapshot_metric", "return_1y")],
    )

    assert response.status_code == 200
    body = response.json()
    assert body["selected_basic_fields"] == ["management"]
    assert body["selected_snapshot_metrics"] == ["return_1y"]
    assert body["rows"][1]["snapshot_values"]["return_1y"] == 0.08

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BACKEND_DIR = ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from backend.data_model.catalog import get_data_model_catalog  # noqa: E402
from backend.services.data_model_routes import router  # noqa: E402


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_complete_catalog_covers_the_full_platform_business_model() -> None:
    catalog = get_data_model_catalog(scope="all")

    assert catalog["schema_version"] == "1.2.0"
    assert catalog["scope"] == "all"
    assert [item["category_id"] for item in catalog["categories"]] == [
        "governance",
        "master",
        "market",
        "fund",
        "index",
        "macro",
        "marts",
        "research",
        "portfolio",
        "operations",
        "accounting",
        "security",
    ]
    assert catalog["summary"] == {
        "category_count": 12,
        "table_count": 114,
        "field_count": 1761,
        "mapping_target_table_count": 38,
        "mapping_target_field_count": 421,
        "pit_table_count": 83,
        "by_layer": {"control": 68, "master": 15, "canonical": 25, "mart": 6},
        "by_storage_engine": {"sqlite": 75, "parquet": 39},
        "by_delivery_phase": {"core": 46, "next": 68},
    }

    tables = {item["table_id"]: item for item in catalog["tables"]}
    assert {
        "governance.data_source",
        "governance.source_interface",
        "governance.dataset_mapping",
        "governance.field_mapping",
        "governance.download_policy",
        "governance.schedule",
        "governance.data_release",
        "master.instrument",
        "master.instrument_identifier",
        "market.nav_daily",
        "market.quote_daily",
        "fund.holding_disclosure",
        "fund.adjustment_factor",
        "index.membership",
        "macro.observation",
        "mart.instrument_metric_snapshot",
        "research.product_pool_version",
        "research.allocation_plan_version",
        "research.backtest_run",
        "portfolio.asset_owner",
        "portfolio.external_account",
        "portfolio.portfolio",
        "portfolio.target_binding",
        "operations.external_transaction",
        "operations.trade_execution",
        "operations.portfolio_allocation",
        "operations.position_snapshot",
        "operations.reconciliation_case",
        "accounting.booking_event",
        "accounting.journal_line",
        "accounting.pbor_snapshot",
        "accounting.performance_metric",
        "accounting.attribution_result",
        "security.audit_event",
    }.issubset(tables)

    instrument = tables["master.instrument"]
    assert instrument["source_mappable"] is True
    assert instrument["usage"] == "external_import"
    assert instrument["primary_key"] == ["instrument_id", "valid_from"]
    instrument_fields = {item["name"]: item for item in instrument["fields"]}
    assert instrument_fields["instrument_id"]["source_mappable"] is False
    assert instrument_fields["instrument_type"]["nullable"] is False

    nav = tables["market.nav_daily"]
    nav_fields = {item["name"]: item for item in nav["fields"]}
    assert nav["pit_supported"] is True
    assert nav_fields["valuation_date"]["role"] == "observation_time"
    assert nav_fields["available_at"]["role"] == "available_time"
    assert nav_fields["adjusted_nav"]["data_type"] == "float64"
    assert nav_fields["net_assets"]["data_type"] == "decimal128(38, 6)"

    metric_mart = tables["mart.instrument_metric_snapshot"]
    assert metric_mart["source_mappable"] is False
    assert metric_mart["usage"] == "system_internal"
    assert all(field["source_mappable"] is False for field in metric_mart["fields"])


def test_default_catalog_exposes_only_valid_external_mapping_targets() -> None:
    catalog = get_data_model_catalog()

    assert catalog["scope"] == "external"
    assert catalog["summary"] == {
        "category_count": 7,
        "table_count": 38,
        "field_count": 694,
        "mapping_target_table_count": 38,
        "mapping_target_field_count": 421,
        "pit_table_count": 31,
        "by_layer": {"master": 12, "canonical": 25, "control": 1},
        "by_storage_engine": {"parquet": 30, "sqlite": 8},
        "by_delivery_phase": {"core": 14, "next": 24},
    }
    assert all(table["source_mappable"] for table in catalog["tables"])
    assert all(table["usage"] == "external_import" for table in catalog["tables"])
    ids = {table["table_id"] for table in catalog["tables"]}
    assert "governance.data_source" not in ids
    assert "mart.daily_return" not in ids


def test_internal_scope_never_marks_fields_as_external_mapping_targets() -> None:
    catalog = get_data_model_catalog(scope="internal")

    assert catalog["scope"] == "internal"
    assert catalog["summary"]["table_count"] == 76
    assert catalog["summary"]["mapping_target_table_count"] == 0
    assert catalog["summary"]["mapping_target_field_count"] == 0
    assert all(not table["source_mappable"] for table in catalog["tables"])
    assert all(
        not field["source_mappable"]
        for table in catalog["tables"]
        for field in table["fields"]
    )


@pytest.mark.parametrize("table_id", [
    "master.organization_identifier",
    "master.person_identifier",
    "master.instrument_identifier",
    "portfolio.external_account_identifier",
])
def test_code_crosswalks_are_internal_not_external_business_data(table_id: str) -> None:
    external = get_data_model_catalog()
    internal = get_data_model_catalog(scope="internal")
    assert table_id not in {table["table_id"] for table in external["tables"]}
    definition = next(table for table in internal["tables"] if table["table_id"] == table_id)
    assert definition["usage"] == "system_internal"
    assert definition["source_mappable"] is False
    assert all(not field["source_mappable"] for field in definition["fields"])

    client = _client()
    assert client.get(f"/api/data-model/tables/{table_id}").status_code == 404
    response = client.get(f"/api/data-model/tables/{table_id}?scope=internal")
    assert response.status_code == 200
    assert response.json() == definition


def test_real_business_relationships_remain_importable() -> None:
    tables = {table["table_id"] for table in get_data_model_catalog()["tables"]}
    assert {
        "master.organization", "master.person", "master.instrument",
        "master.fund_manager_tenure", "master.instrument_classification",
        "master.instrument_benchmark", "index.membership", "index.weight",
        "market.quote_daily", "market.nav_daily", "portfolio.external_account",
    }.issubset(tables)
    assert not any(table_id.endswith("_identifier") for table_id in tables)


def test_catalog_payload_is_isolated_between_calls() -> None:
    first = get_data_model_catalog(scope="all")
    first["tables"].clear()

    second = get_data_model_catalog(scope="all")
    assert second["summary"]["table_count"] == 114
    assert len(second["tables"]) == 114


def test_data_model_routes_apply_explicit_scope_boundaries() -> None:
    client = _client()

    external_response = client.get("/api/data-model/catalog")
    assert external_response.status_code == 200
    assert external_response.json()["scope"] == "external"
    assert external_response.json()["summary"]["table_count"] == 38

    all_response = client.get("/api/data-model/catalog?scope=all")
    assert all_response.status_code == 200
    assert all_response.json()["summary"]["table_count"] == 114

    table_response = client.get("/api/data-model/tables/market.nav_daily")
    assert table_response.status_code == 200
    assert table_response.json()["table_id"] == "market.nav_daily"

    hidden_internal_response = client.get("/api/data-model/tables/governance.data_source")
    assert hidden_internal_response.status_code == 404

    visible_internal_response = client.get(
        "/api/data-model/tables/governance.data_source?scope=all"
    )
    assert visible_internal_response.status_code == 200
    assert visible_internal_response.json()["usage"] == "system_internal"

    missing_response = client.get("/api/data-model/tables/not.exists?scope=all")
    assert missing_response.status_code == 404
    assert missing_response.json()["detail"]["code"] == "DATA_MODEL_TABLE_NOT_FOUND"

    invalid_scope = client.get("/api/data-model/catalog?scope=unsupported")
    assert invalid_scope.status_code == 422

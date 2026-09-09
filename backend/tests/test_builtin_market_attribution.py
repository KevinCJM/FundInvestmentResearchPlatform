"""The removed indicator must not remain callable through hidden catalog paths."""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from custom_indicators.service import CustomIndicatorService
from custom_indicators.variable_registry import variable_catalog
from services import custom_indicator_routes


@pytest.fixture
def client(monkeypatch, tmp_path):
    service = CustomIndicatorService(tmp_path, tmp_path)
    monkeypatch.setattr(custom_indicator_routes, "indicator_service", service)
    app = FastAPI()
    app.include_router(custom_indicator_routes.router)
    with TestClient(app) as value:
        yield value
    service.close_compute_engine()


@pytest.mark.parametrize("include_compatibility", [False, True])
def test_market_attribution_removed_from_all_catalog_modes(client, include_compatibility):
    response = client.get("/api/custom-indicators", params={"include_compatibility": str(include_compatibility).lower()})
    assert response.status_code == 200
    assert "builtin-market-attribution-csi300" not in {item["id"] for item in response.json()["items"]}


def test_removed_builtin_has_no_hidden_evaluation_entry(client):
    indicator_id = "builtin-market-attribution-csi300"
    assert client.get(f"/api/custom-indicators/{indicator_id}").status_code == 404
    response = client.post("/api/custom-indicators/evaluate", json={
        "indicator_ids": [indicator_id], "targets": [{"kind": "etf", "product_id": "510050.SH"}], "period": "ALL",
    })
    assert response.status_code == 404


def test_fixed_market_variable_removed_but_portfolio_benchmark_contract_remains():
    assert "csi300_returns" not in {item["id"] for item in variable_catalog("single_product")}
    assert "benchmark_returns" in {item["id"] for item in variable_catalog("portfolio")}


def test_used_single_scalar_api_contract_is_not_deleted(client):
    response = client.get("/api/custom-indicators/builtin-maximum-drawdown-v2")
    assert response.status_code == 200
    assert response.json()["output_contract"] == "scalar"
    assert response.json()["revision"] == 1

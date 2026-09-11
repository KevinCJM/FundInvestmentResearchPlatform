from __future__ import annotations

import copy
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from scenario_stress.contracts import meta_contract
from scenario_stress.numba_kernels import warm_scenario_stress_numba_kernels
from scenario_stress.service import ScenarioStressService
from services import scenario_stress_routes


def _template(method: str = "factor_path") -> dict:
    return copy.deepcopy(
        next(item["definition"] for item in meta_contract()["templates"] if item["definition"]["method"] == method)
    )


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> TestClient:
    warm_scenario_stress_numba_kernels()
    service = ScenarioStressService(tmp_path, tmp_path)
    monkeypatch.setattr(scenario_stress_routes, "scenario_stress_service", service)
    app = FastAPI()
    app.include_router(scenario_stress_routes.router)
    return TestClient(app)


def test_production_app_registers_complete_scenario_stress_api() -> None:
    from app import app as production_app

    paths = set(production_app.openapi()["paths"])
    assert {
        "/api/scenario-stress/meta",
        "/api/scenario-stress/weight-summary",
        "/api/scenario-stress/definitions",
        "/api/scenario-stress/definitions/{definition_id}",
        "/api/scenario-stress/run",
        "/api/scenario-stress/batch-run",
        "/api/scenario-stress/runs",
        "/api/scenario-stress/runs/{run_id}",
        "/api/scenario-stress/runs/{run_id}/publish",
        "/api/scenario-stress/compare",
    }.issubset(paths)


def test_meta_and_versioned_definition_crud(client: TestClient) -> None:
    meta = client.get("/api/scenario-stress/meta")
    assert meta.status_code == 200
    assert {item["id"] for item in meta.json()["methods"]} == {
        "historical_replay",
        "factor_path",
        "monte_carlo",
        "regime_conditioned",
        "reverse_stress",
    }
    assert meta.json()["compute_audit"]["engine"] == "numba_njit_fixed_signature"
    assert meta.json()["compute_audit"]["python_fallback"] == 0
    assert meta.json()["compute_audit"]["signatures"]
    created_response = client.post("/api/scenario-stress/definitions", json=_template())
    assert created_response.status_code == 201, created_response.text
    created = created_response.json()
    assert created["revision"] == 1
    assert client.get("/api/scenario-stress/definitions").json()["items"][0]["id"] == created["id"]
    assert client.get(f"/api/scenario-stress/definitions/{created['id']}?revision=1").status_code == 200
    updated_payload = copy.deepcopy(created)
    updated_payload["name"] = "修改后的压力定义"
    updated_response = client.put(f"/api/scenario-stress/definitions/{created['id']}", json=updated_payload)
    assert updated_response.status_code == 200, updated_response.text
    updated = updated_response.json()
    assert updated["revision"] == 2
    conflict = client.put(f"/api/scenario-stress/definitions/{created['id']}", json=updated_payload)
    assert conflict.status_code == 409
    assert conflict.json()["detail"]["code"] == "REVISION_CONFLICT"
    archived = client.delete(f"/api/scenario-stress/definitions/{created['id']}?revision=2")
    assert archived.status_code == 200
    assert archived.json()["archived"] is True
    assert client.get("/api/scenario-stress/definitions").json()["items"] == []
    assert len(client.get("/api/scenario-stress/definitions?include_archived=true").json()["items"]) == 1


def test_run_list_get_publish_and_compare_routes(client: TestClient) -> None:
    created = client.post("/api/scenario-stress/definitions", json=_template()).json()
    first_response = client.post(
        "/api/scenario-stress/run",
        json={"definition": {"id": created["id"], "revision": created["revision"]}},
    )
    assert first_response.status_code == 201, first_response.text
    first = first_response.json()
    assert first["immutable"] is True
    assert first["probabilistic"] is False
    assert first["compute_audit"]["kernel"] == "scenario_stress_numeric_core"
    assert first["compute_audit"]["python_fallback"] == 0
    assert first["results"][0]["path"]
    assert "distribution" not in first["results"][0]
    publish = client.post(
        f"/api/scenario-stress/runs/{first['id']}/publish",
        json={"usage": ["product_research", "taa"], "note": "验收发布"},
    )
    assert publish.status_code == 200, publish.text
    assert len(publish.json()["publications"]) == 2
    assert client.get(f"/api/scenario-stress/runs/{first['id']}").json()["content_hash"] == first["content_hash"]
    assert client.get(f"/api/scenario-stress/runs?definition_id={created['id']}").json()["items"][0]["id"] == first["id"]

    second_response = client.post(
        "/api/scenario-stress/run",
        json={"definition": _template("monte_carlo")},
    )
    assert second_response.status_code == 201, second_response.text
    second = second_response.json()
    compare = client.post(
        "/api/scenario-stress/compare",
        json={"run_ids": [first["id"], second["id"]], "reference_run_id": first["id"]},
    )
    assert compare.status_code == 200, compare.text
    assert compare.json()["reference_run_id"] == first["id"]
    assert len(compare.json()["runs"]) == 2
    assert compare.json()["execution"]["request_time_compilation"] == 0
    assert compare.json()["execution"]["object_mode"] == 0


def test_live_weight_summary_supports_short_positions_and_returns_fixed_njit_audit(
    client: TestClient,
) -> None:
    response = client.post(
        "/api/scenario-stress/weight-summary",
        json={"weights": {"equity": 1.2, "hedge": -0.2}},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["total_weight"] == pytest.approx(1.0)
    assert payload["net_exposure"] == pytest.approx(1.0)
    assert payload["gross_exposure"] == pytest.approx(1.4)
    assert payload["within_tolerance"] is True
    assert payload["execution"]["backend"] == "numba_njit_fixed_signature"
    assert payload["execution"]["nopython"] is True
    assert payload["execution"]["object_mode"] == 0
    assert payload["execution"]["python_fallback"] == 0
    assert payload["execution"]["request_time_compilation"] == 0
    assert payload["execution"]["fully_warmed"] is True

    invalid = client.post(
        "/api/scenario-stress/weight-summary",
        json={"weights": {"equity": 1.2, "hedge": -0.1}},
    )
    assert invalid.status_code == 200, invalid.text
    assert invalid.json()["sums_to_one"] is False
    assert invalid.json()["within_tolerance"] is False


def test_batch_route_creates_one_multi_portfolio_snapshot(client: TestClient) -> None:
    definition = _template()
    second = copy.deepcopy(definition["portfolios"][0])
    second["id"] = "second"
    second["name"] = "第二组合"
    second["weights"] = {"cn_equity": 0.5, "duration_bond": 0.3, "gold": 0.2}
    definition["portfolios"].append(second)
    created = client.post("/api/scenario-stress/definitions", json=definition).json()
    response = client.post(
        "/api/scenario-stress/batch-run",
        json={"definition": {"id": created["id"], "revision": 1}},
    )
    assert response.status_code == 201, response.text
    run = response.json()
    assert run["batch"] is True
    assert run["batch_size"] == 2
    assert len(run["results"]) == 2


def test_routes_return_structured_chinese_business_errors(client: TestClient) -> None:
    invalid = _template()
    invalid["portfolios"][0]["weights"].pop("gold")
    response = client.post("/api/scenario-stress/definitions", json=invalid)
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["code"] == "WEIGHT_DIMENSION_MISMATCH"
    assert "缺失权重不会按 0 处理" in detail["message"]
    assert detail["diagnostics"][0]["missing_assets"] == ["gold"]


def test_batch_route_rejects_single_portfolio_without_persisting_run(client: TestClient) -> None:
    created = client.post("/api/scenario-stress/definitions", json=_template()).json()
    response = client.post(
        "/api/scenario-stress/batch-run",
        json={"definition": {"id": created["id"], "revision": 1}},
    )
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "BATCH_REQUIRES_MULTIPLE_PORTFOLIOS"
    assert client.get("/api/scenario-stress/runs").json()["items"] == []

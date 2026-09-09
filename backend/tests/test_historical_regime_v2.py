from __future__ import annotations

import copy
import json
import math
import threading
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from compute_policy import NJIT_BACKEND, THIRD_PARTY_BACKEND, validate_execution_audit
from historical_regimes import v2_service as v2_service_module
from historical_regimes.v2_contracts import parse_definition_v2
from historical_regimes.v2_numba import KERNELS, regime_graph_numba_status
from historical_regimes.service import HistoricalRegimeService
from historical_regimes.v2_service import RegimeGraphV2Service, hydrate_v2_run_snapshot
from research_series.service import write_upload_artifact
from services import historical_regime_routes


def _rows(count: int = 80) -> list[dict[str, Any]]:
    start = date(2020, 1, 1)
    rows: list[dict[str, Any]] = []
    for index in range(count):
        value = 100.0 + index if index < count // 2 else 100.0 + count - index
        observation_date = start + timedelta(days=index)
        rows.append(
            {
                "observation_date": observation_date.isoformat(),
                "available_at": (observation_date + timedelta(days=1)).isoformat(),
                "value": value,
            }
        )
    return rows


def _definition(rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "schema_version": "2.0",
        "name": "自由图谱验收",
        "description": "从原始时序到市场状态的可编辑 DAG。",
        "graph": {
            "nodes": [
                {
                    "id": "source",
                    "type": "source.inline",
                    "parameters": {"rows": rows if rows is not None else _rows(), "frequency": "daily"},
                },
                {
                    "id": "returns",
                    "type": "transform.return",
                    "parameters": {"window": 1},
                    "inputs": {"value": {"node_id": "source", "port": "value"}},
                },
                {
                    "id": "smooth",
                    "type": "filter.ema",
                    "parameters": {"window": 3},
                    "inputs": {"value": {"node_id": "returns", "port": "value"}},
                },
                {
                    "id": "classifier",
                    "type": "model.threshold",
                    "parameters": {"upper": 0.002, "lower": -0.002},
                    "inputs": {"value": {"node_id": "smooth", "port": "value"}},
                },
                {
                    "id": "confirmed",
                    "type": "post.confirmation",
                    "parameters": {"confirmation": 2, "min_duration": 2},
                    "inputs": {"state": {"node_id": "classifier", "port": "state"}},
                },
            ],
            "outputs": {"state": {"node_id": "confirmed", "port": "state"}},
            "exposed_node_ids": ["returns", "smooth", "classifier"],
        },
        "states": [
            {"id": "bull", "label": "牛市", "role": "positive", "color": "#16a34a", "order": 1},
            {"id": "sideways", "label": "震荡", "role": "neutral", "color": "#64748b", "order": 2},
            {"id": "bear", "label": "熊市", "role": "negative", "color": "#dc2626", "order": 3},
        ],
        "evaluation_targets": [],
        "validation": {"walk_forward": True, "folds": 4},
        "usage_intent": "research_display",
    }


@pytest.fixture
def v2_service(tmp_path: Path) -> RegimeGraphV2Service:
    return RegimeGraphV2Service(tmp_path, tmp_path)


@pytest.fixture
def classic_service(tmp_path: Path) -> HistoricalRegimeService:
    return HistoricalRegimeService(tmp_path, tmp_path)


@pytest.fixture
def client(
    monkeypatch: pytest.MonkeyPatch,
    v2_service: RegimeGraphV2Service,
    classic_service: HistoricalRegimeService,
) -> TestClient:
    monkeypatch.setattr(historical_regime_routes, "regime_graph_v2_service", v2_service)
    monkeypatch.setattr(historical_regime_routes, "historical_regime_service", classic_service)
    app = FastAPI()
    app.include_router(historical_regime_routes.router)
    return TestClient(app)


def _wait_for_preview(client: TestClient, preview_id: str) -> dict[str, Any]:
    for _ in range(200):
        response = client.get(f"/api/historical-regimes/preview-runs/{preview_id}")
        assert response.status_code == 200, response.text
        payload = response.json()
        if payload["status"] in {"completed", "failed", "cancelled"}:
            return payload
        time.sleep(0.01)
    raise AssertionError("preview did not finish")


def test_node_catalog_exposes_typed_ports_and_njit_policy(client: TestClient) -> None:
    response = client.get("/api/historical-regimes/nodes")

    assert response.status_code == 200
    payload = response.json()
    assert payload["schema_version"] == "2.0"
    assert payload["runtime"]["complete"] is True
    items = {item["id"]: item for item in payload["items"]}
    assert {"source.inline", "filter.ema", "model.threshold", "model.quadrant", "post.confirmation"}.issubset(items)
    assert items["model.threshold"]["inputs"][0] == {
        "name": "value",
        "type": "series<float64>",
        "required": True,
        "label": "数值序列",
        "description": "数值序列，数据结构为数值时间序列。",
        "type_label": "数值时间序列",
    }
    index_parameters = items["source.index"]["parameter_schema"]["properties"]
    assert index_parameters["field"]["enum_labels"][index_parameters["field"]["enum"].index("close")] == "收盘点位"
    assert index_parameters["field"]["option_source"] == "research_series.fields"
    assert index_parameters["frequency"]["enum_labels"] == ["日频", "周频", "月频", "季频", "年频"]
    assert all(item["category_label"] for item in items.values())
    for item in items.values():
        for parameter in item["parameter_schema"]["properties"].values():
            assert parameter["title"]
            assert parameter["description"]
        for port in [*item["inputs"], *item["outputs"]]:
            assert port["label"]
            assert port["type_label"]
    assert items["model.threshold"]["njit_policy"]["execution_backend"] == NJIT_BACKEND
    assert items["model.external_optimized"]["njit_policy"]["execution_backend"] == THIRD_PARTY_BACKEND
    assert items["model.external_optimized"]["available"] is False
    assert items["model.external_optimized"]["status"] == "admin_adapter_required"
    assert items["source.relative"]["status"] == "deprecated_use_explicit_graph"
    assert {"series<bool>", "regime_candidate<time,state>", "regime_output<time,state>"}.issubset(
        payload["port_types"]
    )
    assert items["transform.lag"]["available"] is True


def test_templates_are_editable_valid_v2_definitions(client: TestClient) -> None:
    listed = client.get("/api/historical-regimes/templates/v2")
    assert listed.status_code == 200
    template_ids = {item["id"] for item in listed.json()["items"]}
    assert {"blank-three-state", "bull-bear-causal-v2", "merrill-clock-v2", "size-rotation-v2", "growth-value-rotation-v2"}.issubset(template_ids)

    instantiated = client.post("/api/historical-regimes/templates/blank-three-state/instantiate")
    assert instantiated.status_code == 200, instantiated.text
    payload = instantiated.json()
    assert payload["definition"]["schema_version"] == "2.0"
    assert payload["inference"]["valid"] is True
    payload["definition"]["name"] = "用户修改后的模板"
    assert payload["definition"]["graph"]["nodes"]


def test_graph_edges_and_node_inputs_are_bidirectionally_canonicalized(client: TestClient) -> None:
    definition = _definition()
    inferred_from_inputs = client.post(
        "/api/historical-regimes/infer",
        json={"definition": definition},
    )
    assert inferred_from_inputs.status_code == 200, inferred_from_inputs.text

    prepared = client.post(
        "/api/historical-regimes/v2/definitions",
        json={"definition": definition},
    )
    assert prepared.status_code == 201, prepared.text
    saved_graph = prepared.json()["graph"]
    assert len(saved_graph["edges"]) == 4
    assert saved_graph["edges"][0]["target"]["port"] == "value"

    edges_only = _definition()
    edges_only["graph"]["edges"] = saved_graph["edges"]
    for node in edges_only["graph"]["nodes"]:
        node.pop("inputs", None)
    saved_edges_only = client.post(
        "/api/historical-regimes/v2/definitions",
        json={"definition": edges_only},
    )
    assert saved_edges_only.status_code == 201, saved_edges_only.text
    restored_nodes = {node["id"]: node for node in saved_edges_only.json()["graph"]["nodes"]}
    assert restored_nodes["returns"]["inputs"]["value"] == {
        "node_id": "source",
        "port": "value",
    }

    inconsistent = _definition()
    inconsistent["graph"]["edges"] = copy.deepcopy(saved_graph["edges"])
    inconsistent["graph"]["edges"][0]["source"]["node_id"] = "source"
    rejected = client.post(
        "/api/historical-regimes/infer",
        json={"definition": inconsistent},
    )
    assert rejected.status_code == 200
    assert rejected.json()["valid"] is False
    assert rejected.json()["errors"][0]["code"] == "SCHEMA_VALIDATION_ERROR"


def test_infer_reports_cycle_and_port_type_errors_without_running(client: TestClient) -> None:
    definition = _definition()
    definition["graph"]["nodes"][1]["inputs"]["value"] = {"node_id": "confirmed", "port": "state"}
    definition["graph"]["nodes"][4]["inputs"]["state"] = {"node_id": "returns", "port": "value"}

    response = client.post("/api/historical-regimes/infer", json={"definition": definition})

    assert response.status_code == 200
    payload = response.json()
    assert payload["valid"] is False
    codes = {item["code"] for item in payload["errors"]}
    assert "GRAPH_CYCLE" in codes
    assert "PORT_TYPE_MISMATCH" in codes


def test_v2_definition_versions_are_saved_separately(client: TestClient) -> None:
    created_response = client.post(
        "/api/historical-regimes/v2/definitions",
        json={"definition": _definition()},
    )
    assert created_response.status_code == 201, created_response.text
    created = created_response.json()
    assert created["schema_version"] == "2.0"
    assert created["revision"] == 1

    changed = copy.deepcopy(created)
    changed["name"] = "自由图谱验收 v2"
    updated_response = client.put(
        f"/api/historical-regimes/v2/definitions/{created['id']}",
        json={"revision": 1, "definition": changed},
    )
    assert updated_response.status_code == 200, updated_response.text
    assert updated_response.json()["revision"] == 2

    previous = client.get(
        f"/api/historical-regimes/v2/definitions/{created['id']}?revision=1"
    )
    assert previous.status_code == 200
    assert previous.json()["name"] == created["name"]


def test_prepare_reuses_structural_plan_for_parameter_changes(client: TestClient) -> None:
    definition = _definition()
    first = client.post("/api/historical-regimes/prepare", json={"definition": definition})
    assert first.status_code == 200, first.text
    first_plan = first.json()
    assert first_plan["request_time_compilation"] == 0
    validate_execution_audit(first_plan["runtime_audit"])

    definition["graph"]["nodes"][3]["parameters"]["upper"] = 0.004
    second = client.post("/api/historical-regimes/prepare", json={"definition": definition})
    assert second.status_code == 200, second.text
    second_plan = second.json()
    assert second_plan["graph_hash"] == first_plan["graph_hash"]
    assert second_plan["compile_token"] == first_plan["compile_token"]


def test_prepare_does_not_reuse_an_expired_plan(v2_service: RegimeGraphV2Service) -> None:
    first = v2_service.prepare(_definition())
    with v2_service._lock:
        v2_service._plans[first["compile_token"]]["expires_at"] = datetime.now(timezone.utc) - timedelta(seconds=1)
    second = v2_service.prepare(_definition())
    assert second["compile_token"] != first["compile_token"]
    assert first["compile_token"] not in v2_service._plans


def test_prepare_persists_auditable_manifest_without_execution_tokens(
    tmp_path: Path,
) -> None:
    service = RegimeGraphV2Service(tmp_path, tmp_path)
    definition = _definition()
    definition["graph"]["nodes"] = [
        definition["graph"]["nodes"][0],
        {
            "id": "formula",
            "type": "feature.formula",
            "parameters": {
                "expression": "feature_1",
                "variables": {"feature_1": "feature_1"},
            },
            "inputs": {"feature_1": {"node_id": "source", "port": "value"}},
        },
        {
            "id": "classifier",
            "type": "model.threshold",
            "parameters": {"upper": 101.0, "lower": 99.0},
            "inputs": {"value": {"node_id": "formula", "port": "value"}},
        },
        {
            "id": "confirmed",
            "type": "post.confirmation",
            "parameters": {"confirmation": 2, "min_duration": 2},
            "inputs": {"state": {"node_id": "classifier", "port": "state"}},
        },
    ]
    definition["graph"]["outputs"] = {
        "state": {"node_id": "confirmed", "port": "state"}
    }
    definition["graph"]["exposed_node_ids"] = ["formula", "classifier"]

    saved = service.create_definition(definition)
    prepared = service.prepare(saved)
    manifest_path = tmp_path / "historical_regime_v2_plan_manifests.json"
    stored = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = stored["items"][0]

    assert manifest["preparation_hash"] == prepared["preparation_hash"]
    assert manifest["graph_hash"] == prepared["graph_hash"]
    assert manifest["manifest_content_hash"]
    assert manifest["runtime_contract"]["nopython"] is True
    assert manifest["runtime_contract"]["python_fallback"] == 0
    assert manifest["runtime_contract"]["request_time_compilation"] == 0
    assert manifest["token_policy"]["authorizes_execution"] is False
    assert manifest["formula_nodes"]["formula"]["compiled_plan_id"]
    serialized = json.dumps(stored, ensure_ascii=False)
    assert prepared["compile_token"] not in serialized
    assert "rg2-" not in serialized

    def assert_no_token_key(value: Any) -> None:
        if isinstance(value, dict):
            for raw_key, child in value.items():
                key = str(raw_key).lower().replace("-", "_")
                assert key != "token" and not key.endswith("_token")
                assert_no_token_key(child)
        elif isinstance(value, list):
            for child in value:
                assert_no_token_key(child)

    assert_no_token_key(stored)

    with pytest.raises(Exception) as exc_info:
        service.plan_manifests.upsert(
            {
                "preparation_hash": "unsafe",
                "formula_nodes": {"formula": {"compile_token": "formula-secret"}},
            }
        )
    assert getattr(exc_info.value, "code", None) == "REGIME_PLAN_MANIFEST_SECRET_REJECTED"


def test_unsaved_draft_plans_remain_ephemeral_and_do_not_grow_manifest_storage(
    tmp_path: Path,
) -> None:
    service = RegimeGraphV2Service(tmp_path, tmp_path)
    first = service.prepare(_definition())
    changed = _definition()
    changed["graph"]["nodes"][2]["type"] = "filter.sma"
    second = service.prepare(changed)

    assert first["compile_token"] in service._plans
    assert second["compile_token"] in service._plans
    assert service.plan_manifests.list() == []
    assert not (tmp_path / "historical_regime_v2_plan_manifests.json").exists()


def test_saved_definition_is_rewarmed_from_persistent_plan_manifest(
    tmp_path: Path,
) -> None:
    first_service = RegimeGraphV2Service(tmp_path, tmp_path)
    saved = first_service.create_definition(_definition())
    first_plan = first_service.prepare(saved)
    first_manifest = first_service.plan_manifests.get(first_plan["preparation_hash"])

    restarted = RegimeGraphV2Service(tmp_path, tmp_path)
    startup = restarted.startup_prewarm
    assert startup["complete"] is True
    assert startup["prepared_count"] == 1
    assert startup["prepared"][0]["manifest_status"] == "verified_and_rewarmed"
    assert startup["prepared"][0]["manifest_content_hash"] == first_manifest["manifest_content_hash"]
    assert startup["manifest_authorizes_execution"] is False
    assert startup["token_storage"] == "process_local_memory_only"
    assert restarted.catalog()["startup_prewarm"]["prepared_count"] == 1

    restarted_token = restarted._plans_by_graph_hash[first_plan["preparation_hash"]]
    assert restarted_token != first_plan["compile_token"]
    assert restarted_token in restarted._plans
    stored = (tmp_path / "historical_regime_v2_plan_manifests.json").read_text(
        encoding="utf-8"
    )
    assert first_plan["compile_token"] not in stored
    assert restarted_token not in stored


def test_tampered_plan_manifest_fails_integrity_check_and_is_not_reported_verified(
    tmp_path: Path,
) -> None:
    first_service = RegimeGraphV2Service(tmp_path, tmp_path)
    saved = first_service.create_definition(_definition())
    prepared = first_service.prepare(saved)
    manifest_path = tmp_path / "historical_regime_v2_plan_manifests.json"
    stored = json.loads(manifest_path.read_text(encoding="utf-8"))
    stored["items"][0]["runtime_contract"]["nopython"] = False
    manifest_path.write_text(
        json.dumps(stored, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    restarted = RegimeGraphV2Service(tmp_path, tmp_path)
    assert restarted.startup_prewarm["complete"] is False
    assert restarted.startup_prewarm["prepared_count"] == 0
    assert restarted.startup_prewarm["errors"][0]["error"]["code"] == (
        "REGIME_PLAN_MANIFEST_INTEGRITY_MISMATCH"
    )
    with pytest.raises(Exception) as exc_info:
        restarted.plan_manifests.get(prepared["preparation_hash"])
    assert getattr(exc_info.value, "code", None) == "REGIME_PLAN_MANIFEST_INTEGRITY_MISMATCH"
    with pytest.raises(Exception) as prepare_exc:
        restarted.prepare(saved)
    assert getattr(prepare_exc.value, "code", None) == "REGIME_PLAN_MANIFEST_INTEGRITY_MISMATCH"


def test_preview_requires_prepared_plan_and_exposes_final_and_node_series(client: TestClient) -> None:
    definition = _definition()
    missing = client.post(
        "/api/historical-regimes/preview-runs",
        json={"definition": definition, "compile_token": "missing", "mode": "realtime"},
    )
    assert missing.status_code == 422
    assert missing.json()["detail"]["code"] == "REGIME_GRAPH_PLAN_NOT_WARM"

    prepared = client.post("/api/historical-regimes/prepare", json={"definition": definition}).json()
    created = client.post(
        "/api/historical-regimes/preview-runs",
        json={
            "definition": definition,
            "compile_token": prepared["compile_token"],
            "mode": "realtime",
            "ttl_seconds": 60,
        },
    )
    assert created.status_code == 202, created.text
    assert created.json()["status"] in {"queued", "preparing", "running", "completed"}
    finished = _wait_for_preview(client, created.json()["id"])
    assert finished["status"] == "completed", finished
    assert finished["result"]["row_count"] == len(_rows())
    assert finished["result"]["diagnostics"]["request_time_compilation"] == 0
    assert finished["result"]["diagnostics"]["python_fallback"] == 0

    final_series = client.get(
        f"/api/historical-regimes/preview-runs/{finished['id']}/series?offset=0&limit=10"
    )
    assert final_series.status_code == 200, final_series.text
    final_payload = final_series.json()
    assert final_payload["total"] == len(_rows())
    assert len(final_payload["items"]) == 10
    assert {"observation_date", "value", "state_id", "confidence", "probabilities"}.issubset(final_payload["items"][0])

    node_series = client.get(
        f"/api/historical-regimes/preview-runs/{finished['id']}/series",
        params={"node_id": "smooth", "port": "value", "offset": 2, "limit": 5},
    )
    assert node_series.status_code == 200, node_series.text
    node_payload = node_series.json()
    assert node_payload["node_type"] == "filter.ema"
    assert node_payload["value_type"] == "series<float64>"
    assert [item["index"] for item in node_payload["items"]] == [2, 3, 4, 5, 6]


def test_normal_preview_does_not_add_numba_signatures(client: TestClient) -> None:
    before = {kernel_id: tuple(dispatcher.signatures) for kernel_id, dispatcher in KERNELS.items()}
    definition = _definition()
    prepared = client.post("/api/historical-regimes/prepare", json={"definition": definition}).json()
    created = client.post(
        "/api/historical-regimes/preview-runs",
        json={"definition": definition, "compile_token": prepared["compile_token"]},
    )
    finished = _wait_for_preview(client, created.json()["id"])
    assert finished["status"] == "completed", finished
    after = {kernel_id: tuple(dispatcher.signatures) for kernel_id, dispatcher in KERNELS.items()}
    assert after == before
    assert all(dispatcher._can_compile is False for dispatcher in KERNELS.values())
    assert regime_graph_numba_status()["complete"] is True


def test_lag_node_executes_the_fixed_signature_kernel(client: TestClient) -> None:
    definition = _definition()
    definition["graph"]["nodes"][1]["type"] = "transform.lag"
    definition["graph"]["nodes"][1]["parameters"] = {"window": 2}
    prepared = client.post("/api/historical-regimes/prepare", json={"definition": definition}).json()
    created = client.post(
        "/api/historical-regimes/preview-runs",
        json={"definition": definition, "compile_token": prepared["compile_token"]},
    )
    finished = _wait_for_preview(client, created.json()["id"])
    assert finished["status"] == "completed", finished
    lagged = client.get(
        f"/api/historical-regimes/preview-runs/{finished['id']}/series",
        params={"node_id": "returns", "port": "value", "offset": 0, "limit": 4},
    ).json()["items"]
    assert lagged[0]["value"] is None
    assert lagged[1]["value"] is None
    assert lagged[2]["value"] == pytest.approx(_rows()[0]["value"])


def test_pit_asof_stably_sorts_non_monotonic_availability(client: TestClient) -> None:
    start = date(2020, 1, 1)
    anchor_rows = [
        {
            "observation_date": (start + timedelta(days=index)).isoformat(),
            "available_at": (start + timedelta(days=index)).isoformat(),
            "value": 100.0 + index,
        }
        for index in range(8)
    ]
    release_offsets = [4, 2, 5, 3, 6]
    feature_rows = [
        {
            "observation_date": (start + timedelta(days=index)).isoformat(),
            "available_at": (start + timedelta(days=release_offsets[index])).isoformat(),
            "value": float((index + 1) * 10),
        }
        for index in range(5)
    ]
    definition = _definition(anchor_rows)
    definition["graph"] = {
        "nodes": [
            {"id": "anchor", "type": "source.inline", "parameters": {"rows": anchor_rows}},
            {"id": "feature", "type": "source.inline", "parameters": {"rows": feature_rows}},
            {
                "id": "pit",
                "type": "align.pit_asof",
                "parameters": {"max_age_days": 30},
                "inputs": {
                    "anchor": {"node_id": "anchor", "port": "value"},
                    "feature": {"node_id": "feature", "port": "value"},
                },
            },
            {
                "id": "classifier",
                "type": "model.threshold",
                "parameters": {"upper": 35.0, "lower": 15.0},
                "inputs": {"value": {"node_id": "pit", "port": "feature"}},
            },
        ],
        "outputs": {"state": {"node_id": "classifier", "port": "state"}},
        "exposed_node_ids": ["pit"],
    }
    prepared = client.post("/api/historical-regimes/prepare", json={"definition": definition}).json()
    created = client.post(
        "/api/historical-regimes/preview-runs",
        json={"definition": definition, "compile_token": prepared["compile_token"]},
    )
    finished = _wait_for_preview(client, created.json()["id"])
    assert finished["status"] == "completed", finished
    aligned = client.get(
        f"/api/historical-regimes/preview-runs/{finished['id']}/series",
        params={"node_id": "pit", "port": "feature", "offset": 0, "limit": 6},
    ).json()["items"]
    assert [item["value"] for item in aligned[:5]] == [None, None, 20.0, 40.0, 10.0]


def test_probability_output_contract_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    client: TestClient,
) -> None:
    definition = _definition()
    definition["graph"]["outputs"]["probabilities"] = {
        "node_id": "classifier",
        "port": "probabilities",
    }

    def invalid_probabilities(states: Any, state_count: Any):
        return np.full((len(states), int(state_count)), np.nan, dtype=np.float64)

    import numpy as np

    monkeypatch.setattr(
        v2_service_module,
        "state_probabilities_kernel",
        invalid_probabilities,
    )
    prepared = client.post("/api/historical-regimes/prepare", json={"definition": definition}).json()
    created = client.post(
        "/api/historical-regimes/preview-runs",
        json={"definition": definition, "compile_token": prepared["compile_token"]},
    )
    finished = _wait_for_preview(client, created.json()["id"])
    assert finished["status"] == "failed"
    assert finished["error"]["code"] == "REGIME_PROBABILITY_CONTRACT_VIOLATION"


def test_realtime_latent_model_records_locked_initial_training_mapping(client: TestClient) -> None:
    definition = _definition()
    definition["graph"] = {
        "nodes": [
            definition["graph"]["nodes"][0],
            {
                "id": "returns",
                "type": "transform.return",
                "parameters": {"window": 1},
                "inputs": {"value": {"node_id": "source", "port": "value"}},
            },
            {
                "id": "features",
                "type": "feature.matrix",
                "parameters": {},
                "inputs": {"feature_1": {"node_id": "returns", "port": "value"}},
            },
            {
                "id": "latent",
                "type": "model.gmm",
                "parameters": {"components": 3, "initial_train_size": 20, "iterations": 5},
                "inputs": {"features": {"node_id": "features", "port": "features"}},
            },
        ],
        "outputs": {"state": {"node_id": "latent", "port": "state"}},
        "exposed_node_ids": ["features", "latent"],
    }
    prepared = client.post("/api/historical-regimes/prepare", json={"definition": definition})
    assert prepared.status_code == 200, prepared.text
    created = client.post(
        "/api/historical-regimes/preview-runs",
        json={
            "definition": definition,
            "compile_token": prepared.json()["compile_token"],
            "mode": "realtime",
        },
    )
    finished = _wait_for_preview(client, created.json()["id"])
    assert finished["status"] == "completed", finished
    audit = finished["result"]["diagnostics"]["model_audits"]["latent"]
    assert audit["fit_mode"] == "initial_training_interval_locked"
    assert audit["training_count"] == 20
    assert audit["label_mapping_locked_before_classification"] is True
    assert audit["uses_full_sample_for_label_mapping"] is False
    assert sorted(audit["component_order"]) == [0, 1, 2]


def test_formal_v2_run_requires_warm_plan_and_externalizes_large_arrays(
    v2_service: RegimeGraphV2Service,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    execute_graph = v2_service._execute_graph

    def realtime_only(job_id, definition, mode, *args, **kwargs):
        assert mode == "realtime", "实时运行不得隐式调用事后算法"
        return execute_graph(job_id, definition, mode, *args, **kwargs)

    monkeypatch.setattr(v2_service, "_execute_graph", realtime_only)
    upload_frame = pd.DataFrame(_rows())
    upload_frame["vintage"] = None
    upload_frame["revision"] = 1
    upload = write_upload_artifact(v2_service.market_data_dir, upload_frame)
    definition = _definition()
    definition["graph"]["nodes"][0] = {
        "id": "source",
        "type": "source.upload",
        "parameters": {
            "artifact_id": upload["artifact_id"],
            "checksum": upload["checksum"],
            "format": "parquet",
            "frequency": "daily",
            "availability_mode": "point_in_time",
        },
    }
    saved = v2_service.create_definition(definition)
    prepared = v2_service.prepare(saved)
    reference = {
        "schema_version": "2.0",
        "id": saved["id"],
        "revision": saved["revision"],
    }
    response = client.post(
        "/api/historical-regimes/run",
        json={
            "definition": reference,
            "mode": "realtime",
            "compile_token": prepared["compile_token"],
        },
    )
    assert response.status_code == 201, response.text
    run = response.json()
    assert len(run["series"]) == len(_rows())
    assert run["overview"]["run_kind"] == "saved"
    assert run["overview"]["summary"]["total"] == len(_rows())
    assert run["overview"]["definition_id"] == saved["id"]
    assert run["overview"]["primary_series"]["response_field"] == "series"
    assert run["overview"]["states"] == run["states"]
    assert run["overview"]["definition_revision"] == saved["revision"]
    assert run["overview"]["definition_hash"] == run["definition_snapshot_hash"]
    assert run["overview"]["definition_hash"]
    assert run["overview"]["graph_hash"] == prepared["graph_hash"]
    assert run["overview"]["graph_hash"]
    assert run["overview"]["date_range"] == {
        "start": run["series"][0]["observation_date"],
        "end": run["series"][-1]["observation_date"],
    }
    all_intervals = sorted(
        run["overview"]["segments"] + run["overview"]["unknown_intervals"],
        key=lambda interval: interval["start_index"],
    )
    assert [index for interval in all_intervals
            for index in range(interval["start_index"], interval["end_index"] + 1)] == list(range(len(run["series"])))
    for interval in all_intervals:
        assert interval["start_date"] == run["series"][interval["start_index"]]["observation_date"]
        assert interval["end_date"] == run["series"][interval["end_index"]]["observation_date"]
        assert all(row["state_id"] == interval["state_id"] for row in run["series"][interval["start_index"]:interval["end_index"] + 1])
    assert "overview" not in v2_service.runs.get(run["id"])
    assert run["walk_forward"]["status"] == "completed"
    assert run["walk_forward"]["folds"]
    assert run["stability"]["prefix_invariance"]["status"] == "passed"
    assert run["stability"]["realtime_vs_retrospective"]["status"] == "disabled"
    assert run["stability"]["parameter_sensitivity"]["candidates"]
    assert run["stability"]["recognition_delay"]["mean_observations"] == pytest.approx(0.0)
    raw = v2_service.runs.get(run["id"])
    assert "series" not in raw
    assert raw["series_artifact"]["content_addressed"] is True
    assert raw["artifact_manifest"]["node_outputs"]["format"] == "npz"
    assert len(v2_service.get_run(run["id"])["series"]) == len(_rows())
    assert run["algorithm_diagnostics"]["diagnostics"]["probability_contract"]["passed"] is True
    assert set(prepared["kernel_ids"]) == set(run["calculation_audit"]["executed_kernel_ids"])
    listed = v2_service.list_runs()
    assert listed[0]["series_included"] is False
    assert "series" not in listed[0]
    hydrated_without_service = hydrate_v2_run_snapshot(
        raw,
        artifact_dir=v2_service.artifact_dir,
    )
    assert len(hydrated_without_service["series"]) == len(_rows())

    with v2_service._lock:
        v2_service._plans.clear()
        v2_service._plans_by_graph_hash.clear()
    with pytest.raises(Exception) as exc_info:
        v2_service.run_saved(reference, "realtime", compile_token=prepared["compile_token"])
    assert getattr(exc_info.value, "code", None) == "REGIME_PLAN_NOT_WARMED"


def test_identical_source_specs_are_scanned_once_per_graph_execution(
    monkeypatch: pytest.MonkeyPatch,
    v2_service: RegimeGraphV2Service,
) -> None:
    definition = _definition()
    source_parameters = {"rows": _rows(), "frequency": "daily"}
    definition["graph"] = {
        "nodes": [
            {"id": "left", "type": "source.inline", "parameters": source_parameters},
            {"id": "right", "type": "source.inline", "parameters": copy.deepcopy(source_parameters)},
            {
                "id": "aligned",
                "type": "align.strict_intersection",
                "inputs": {
                    "left": {"node_id": "left", "port": "value"},
                    "right": {"node_id": "right", "port": "value"},
                },
            },
            {
                "id": "difference",
                "type": "math.subtract",
                "inputs": {
                    "left": {"node_id": "aligned", "port": "left"},
                    "right": {"node_id": "aligned", "port": "right"},
                },
            },
            {
                "id": "classifier",
                "type": "model.threshold",
                "parameters": {"upper": 0.1, "lower": -0.1},
                "inputs": {"value": {"node_id": "difference", "port": "value"}},
            },
        ],
        "outputs": {"state": {"node_id": "classifier", "port": "state"}},
        "exposed_node_ids": [],
    }
    calls = 0
    original = v2_service_module.resolve_target

    def counted_resolve(*args: Any, **kwargs: Any):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(v2_service_module, "resolve_target", counted_resolve)
    prepared = v2_service.prepare(definition)
    parsed = parse_definition_v2(definition)
    v2_service._execute_graph(None, parsed, "realtime", None, plan=prepared)
    assert calls == 1


def test_graph_source_and_evaluation_target_share_one_resolution(
    monkeypatch: pytest.MonkeyPatch,
    v2_service: RegimeGraphV2Service,
) -> None:
    definition = _definition()
    source_parameters = {
        "rows": _rows(),
        "frequency": "daily",
        "availability_mode": "point_in_time",
        "name": "分类输入",
    }
    definition["graph"]["nodes"][0]["parameters"] = source_parameters
    definition["evaluation_targets"] = [
        {
            "id": "market",
            "name": "市场表现",
            "primary": True,
            "source": {
                "kind": "inline",
                **copy.deepcopy(source_parameters),
                "name": "评价序列",
            },
        }
    ]
    calls = 0
    original = v2_service_module.resolve_target

    def counted_resolve(*args: Any, **kwargs: Any):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(v2_service_module, "resolve_target", counted_resolve)
    prepared = v2_service.prepare(definition)
    execution = v2_service._execute_graph(
        None,
        parse_definition_v2(definition),
        "realtime",
        None,
        plan=prepared,
    )

    assert calls == 1
    assert execution["result"]["display_source"] == "market"
    assert execution["result"]["evaluation_results"]["market"]["snapshot"]


def test_graph_source_and_evaluation_target_with_different_fields_do_not_share(
    monkeypatch: pytest.MonkeyPatch,
    v2_service: RegimeGraphV2Service,
) -> None:
    definition = _definition()
    rows = [
        {**row, "alternate_value": float(row["value"]) * 2.0}
        for row in _rows()
    ]
    definition["graph"]["nodes"][0]["parameters"] = {
        "rows": rows,
        "value_field": "value",
        "frequency": "daily",
    }
    definition["evaluation_targets"] = [
        {
            "id": "alternate",
            "name": "另一字段",
            "primary": True,
            "source": {
                "kind": "inline",
                "rows": copy.deepcopy(rows),
                "value_field": "alternate_value",
                "frequency": "daily",
            },
        }
    ]
    calls = 0
    original = v2_service_module.resolve_target

    def counted_resolve(*args: Any, **kwargs: Any):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(v2_service_module, "resolve_target", counted_resolve)
    prepared = v2_service.prepare(definition)
    execution = v2_service._execute_graph(
        None,
        parse_definition_v2(definition),
        "realtime",
        None,
        plan=prepared,
    )

    assert calls == 2
    assert execution["series"][0]["value"] == pytest.approx(rows[0]["alternate_value"])


def test_source_cache_identity_preserves_resolution_semantics() -> None:
    source = {
        "kind": "relative",
        "field": "close",
        "availability_mode": "point_in_time",
        "vintage": "first-release",
        "transform": "ratio",
    }
    key = v2_service_module._canonical_source_cache_key(
        source,
        "retrospective",
        "2024-12-31",
    )

    semantic_changes = [
        {**source, "field": "open"},
        {**source, "availability_mode": "latest"},
        {**source, "vintage": "final"},
        {**source, "transform": "log_ratio"},
    ]
    assert all(
        v2_service_module._canonical_source_cache_key(
            changed,
            "retrospective",
            "2024-12-31",
        )
        != key
        for changed in semantic_changes
    )
    assert (
        v2_service_module._canonical_source_cache_key(
            source,
            "retrospective",
            "2024-06-30",
        )
        != key
    )
    assert (
        v2_service_module._canonical_source_cache_key(
            {**source, "name": "仅修改展示名称"},
            "retrospective",
            "2024-12-31",
        )
        == key
    )


def test_v1_definition_is_read_only_and_copied_to_v2_without_mutation(
    client: TestClient,
) -> None:
    classic = {
        "name": "经典牛熊",
        "description": "迁移测试",
        "target": {"kind": "inline", "rows": _rows()},
        "features": {"filter": "ema", "window": 5},
        "algorithm": {
            "family": "causal_filter",
            "parameters": {"upper": 0.002, "lower": -0.002},
        },
        "states": _definition()["states"],
        "validation": {"walk_forward": True, "folds": 3},
        "usage_intent": "research_display",
    }
    blocked = client.post("/api/historical-regimes/definitions", json=classic)
    assert blocked.status_code == 409, blocked.text
    assert blocked.json()["detail"]["code"] == "V1_DEFINITION_READ_ONLY"

    # Seed a legacy repository record to prove existing history remains readable
    # and copyable even though HTTP writes are now closed.
    source = historical_regime_routes.historical_regime_service.create_definition(classic)
    copied = client.post(
        f"/api/historical-regimes/definitions/{source['id']}/copy-to-v2",
        params={"revision": source["revision"]},
    )
    assert copied.status_code == 200, copied.text
    payload = copied.json()
    assert payload["definition"]["schema_version"] == "2.0"
    assert payload["definition"]["id"] != source["id"]
    assert payload["definition"]["source_v1"]["id"] == source["id"]
    assert payload["definition"]["source_v1"]["revision"] == source["revision"]
    unchanged = client.get(f"/api/historical-regimes/definitions/{source['id']}")
    assert unchanged.status_code == 200
    assert unchanged.json()["schema_version"] != "2.0"


def test_cancel_is_terminal_and_ttl_expiry_removes_preview(v2_service: RegimeGraphV2Service) -> None:
    definition = _definition()
    prepared = v2_service.prepare(definition)
    original_execute_graph = v2_service._execute_graph
    entered = threading.Event()
    release = threading.Event()

    def blocked_execute_graph(*args: Any, **kwargs: Any) -> dict[str, Any]:
        entered.set()
        release.wait(timeout=2.0)
        return original_execute_graph(*args, **kwargs)

    v2_service._execute_graph = blocked_execute_graph  # type: ignore[method-assign]
    created = v2_service.create_preview(
        definition,
        compile_token=prepared["compile_token"],
        ttl_seconds=60,
    )
    assert entered.wait(timeout=2.0)
    cancelled = v2_service.cancel_preview(created["id"])
    release.set()
    assert cancelled["status"] == "cancelled"
    assert v2_service.get_preview(created["id"])["status"] == "cancelled"

    with v2_service._lock:
        v2_service._jobs[created["id"]]["expires_at"] = datetime.now(timezone.utc) - timedelta(seconds=1)
    with pytest.raises(Exception) as exc_info:
        v2_service.get_preview(created["id"])
    assert getattr(exc_info.value, "code", None) == "REGIME_PREVIEW_NOT_FOUND"


def test_third_party_model_declaration_is_validated_then_adapter_fails_closed(client: TestClient) -> None:
    definition = _definition()
    definition["graph"]["nodes"] = [
        definition["graph"]["nodes"][0],
        {
            "id": "external",
            "type": "model.external_optimized",
            "parameters": {
                "adapter_id": "approved-xgboost-adapter",
                "declaration": {
                    "execution_backend": THIRD_PARTY_BACKEND,
                    "model_family": "machine_learning",
                    "package": "xgboost",
                    "package_version": "3.0.0",
                    "model_name": "regime-classifier",
                    "model_version": "1",
                    "native_backend": "xgboost-native",
                    "model_fingerprint": "a" * 64,
                    "input_dtype": "float64",
                    "output_dtype": "int64",
                    "third_party_package": True,
                    "native_optimized": True,
                    "isolated_array_contract": True,
                    "model_engine_scope": "training_or_inference_only",
                    "feature_pipeline_backend": NJIT_BACKEND,
                    "postprocess_backend": NJIT_BACKEND,
                    "python_callback": False,
                    "python_fallback": 0,
                },
            },
            "inputs": {"features": {"node_id": "source", "port": "value"}},
        },
    ]
    definition["graph"]["outputs"] = {"state": {"node_id": "external", "port": "state"}}
    definition["graph"]["exposed_node_ids"] = ["source"]

    inferred = client.post("/api/historical-regimes/infer", json={"definition": definition})
    assert inferred.status_code == 200
    assert inferred.json()["valid"] is True
    assert "isolated_model_train_or_infer" in inferred.json()["execution_lanes"]

    prepared = client.post("/api/historical-regimes/prepare", json={"definition": definition})
    assert prepared.status_code == 422
    assert prepared.json()["detail"]["code"] == "THIRD_PARTY_MODEL_ADAPTER_UNAVAILABLE"


def test_v1_routes_remain_registered_with_v2_routes(client: TestClient) -> None:
    paths = set(client.app.openapi()["paths"])
    assert "/api/historical-regimes/run" in paths
    assert "/api/historical-regimes/definitions" in paths
    assert "/api/historical-regimes/infer" in paths
    assert "/api/historical-regimes/preview-runs/{preview_id}/series" in paths


def test_nonshared_run_stores_merge_and_get_v2_without_production_path_assumption(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    classic = HistoricalRegimeService(tmp_path / "classic", tmp_path)
    graph = RegimeGraphV2Service(tmp_path / "graph", tmp_path)
    classic_run = classic.runs.create(
        {"definition_id": "classic-definition", "name": "classic", "series": []}
    )
    series_manifest = graph._persist_series(
        [
            {
                "observation_date": "2020-01-01",
                "state_id": "bull",
                "probabilities": {"bull": 1.0},
                "features": {},
                "reasons": ["test"],
            }
        ]
    )
    graph_run = graph.runs.create(
        {
            "schema_version": "2.0",
            "definition_id": "graph-definition",
            "name": "graph",
            "series_artifact": series_manifest,
        }
    )
    monkeypatch.setattr(historical_regime_routes, "historical_regime_service", classic)
    monkeypatch.setattr(historical_regime_routes, "regime_graph_v2_service", graph)
    app = FastAPI()
    app.include_router(historical_regime_routes.router)
    local_client = TestClient(app)

    listed = local_client.get("/api/historical-regimes/runs")
    assert listed.status_code == 200, listed.text
    assert {item["id"] for item in listed.json()["items"]} == {
        classic_run["id"],
        graph_run["id"],
    }
    graph_detail = local_client.get(f"/api/historical-regimes/runs/{graph_run['id']}")
    assert graph_detail.status_code == 200
    assert graph_detail.json()["series"][0]["state_id"] == "bull"
    classic_detail = local_client.get(f"/api/historical-regimes/runs/{classic_run['id']}")
    assert classic_detail.status_code == 200
    assert classic_detail.json()["id"] == classic_run["id"]


def test_queued_preview_deep_freezes_nested_parameters(
    v2_service: RegimeGraphV2Service,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    definition = _definition()
    expected_value = definition["graph"]["nodes"][0]["parameters"]["rows"][0]["value"]
    plan = v2_service.prepare(definition)
    monkeypatch.setattr(v2_service_module.threading.Thread, "start", lambda self: None)
    created = v2_service.create_preview(definition, compile_token=plan["compile_token"])
    definition["graph"]["nodes"][0]["parameters"]["rows"][0]["value"] = -999.0
    frozen = v2_service._jobs[created["id"]]["definition"]
    assert frozen.graph.nodes[0].parameters["rows"][0]["value"] == expected_value
    assert created["definition_hash"] == v2_service_module.definition_content_hash(frozen)


def test_preview_overview_is_complete_frozen_and_does_not_execute_again(
    client: TestClient,
    v2_service: RegimeGraphV2Service,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = _rows(3000)
    for index, row in enumerate(rows):
        row["value"] = 80.0 if index in {500, 2999} else 110.0
    definition = _definition(rows)
    definition["graph"] = {
        "nodes": [
            definition["graph"]["nodes"][0],
            {
                "id": "classifier",
                "type": "model.threshold",
                "parameters": {"upper": 105.0, "lower": 95.0},
                "inputs": {"value": {"node_id": "source", "port": "value"}},
            },
        ],
        "outputs": {"state": {"node_id": "classifier", "port": "state"}},
    }
    prepared = client.post("/api/historical-regimes/prepare", json={"definition": definition})
    assert prepared.status_code == 200, prepared.text
    created = client.post(
        "/api/historical-regimes/preview-runs",
        json={"definition": definition, "compile_token": prepared.json()["compile_token"],
              "mode": "retrospective", "as_of": "2030-01-01"},
    )
    assert created.status_code == 202, created.text
    finished = _wait_for_preview(client, created.json()["id"])
    assert finished["status"] == "completed", finished
    run_id = finished["id"]

    def unexpected_execution(*args: Any, **kwargs: Any):
        raise AssertionError("Overview must never rerun the DAG")

    monkeypatch.setattr(v2_service, "_execute_graph", unexpected_execution)
    definition["states"][0]["color"] = "#000000"
    response = client.get(f"/api/historical-regimes/preview-runs/{run_id}/overview")
    assert response.status_code == 200, response.text
    overview = response.json()
    assert overview["run_kind"] == "preview"
    assert overview["run_id"] == run_id
    assert overview["definition_hash"] == finished["definition_hash"]
    assert overview["graph_hash"] == finished["graph_hash"]
    assert overview["mode"] == "retrospective"
    assert overview["as_of"] == "2030-01-01"
    assert overview["data_snapshots"] == finished["result"]["data_snapshots"]
    assert overview["states"][0]["color"] == "#16a34a"
    assert overview["summary"]["total"] == 3000
    assert overview["summary"]["classified"] == 3000
    assert overview["summary"]["unknown"] == 0
    assert overview["summary"]["state_counts"] == {"bull": 2998, "sideways": 0, "bear": 2}
    assert [(item["start_index"], item["end_index"]) for item in overview["segments"]] == [
        (0, 499), (500, 500), (501, 2998), (2999, 2999)
    ]
    assert overview["capabilities"]["effective"]["available"] is False
    assert overview["capabilities"]["probabilities"]["available"] is False
    assert overview["primary_series"]["value_column"] == "value"
    assert overview["primary_series"]["display_source_id"] == "source"
    assert "series" not in overview
    first_page = client.get(overview["primary_series"]["endpoint"], params={"limit": 500}).json()
    second_page = client.get(overview["primary_series"]["endpoint"], params={"offset": 500, "limit": 5000}).json()
    assert first_page["total"] == second_page["total"] == 3000
    assert first_page["items"][-1]["state_id"] == "bull"
    assert second_page["items"][0]["state_id"] == "bear"
    assert len(first_page["items"]) + len(second_page["items"]) == 3000
    assert client.get(overview["primary_series"]["endpoint"], params={"limit": 5001}).status_code == 422
    overview["states"][0]["color"] = "#111111"
    assert v2_service.preview_overview(run_id)["states"][0]["color"] == "#16a34a"


@pytest.mark.parametrize("job_status", ["queued", "running", "failed", "cancelled"])
def test_preview_overview_rejects_noncompleted_jobs(
    client: TestClient,
    v2_service: RegimeGraphV2Service,
    job_status: str,
) -> None:
    v2_service._jobs["not-ready"] = {
        "status": job_status,
        "expires_at": datetime.now(timezone.utc) + timedelta(minutes=1),
    }
    response = client.get("/api/historical-regimes/preview-runs/not-ready/overview")
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "REGIME_PREVIEW_NOT_COMPLETE"


def test_preview_overview_rejects_expired_and_missing_results(
    client: TestClient,
    v2_service: RegimeGraphV2Service,
) -> None:
    v2_service._jobs["expired"] = {
        "status": "completed",
        "expires_at": datetime.now(timezone.utc) - timedelta(seconds=1),
    }
    assert client.get("/api/historical-regimes/preview-runs/expired/overview").status_code == 404
    v2_service._jobs["missing-result"] = {
        "status": "completed",
        "expires_at": datetime.now(timezone.utc) + timedelta(minutes=1),
        "_series": None,
        "result": {},
    }
    response = client.get("/api/historical-regimes/preview-runs/missing-result/overview")
    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "REGIME_PREVIEW_RESULT_UNAVAILABLE"


@pytest.mark.parametrize("connected", [False, True])
def test_realtime_rejects_retrospective_nodes_before_queueing(
    client: TestClient, v2_service: RegimeGraphV2Service, connected: bool,
) -> None:
    definition = _definition()
    definition["graph"]["nodes"].append({
        "id": "offline", "type": "model.turning_point",
        "parameters": {"window": 3, "min_move": 0.08},
        "inputs": {"value": {"node_id": "source", "port": "value"}},
    })
    if connected:
        definition["graph"]["outputs"] = {"state": {"node_id": "offline", "port": "state"}}
    prepared = v2_service.prepare(definition)
    payload = {"definition": definition, "compile_token": prepared["compile_token"], "mode": "realtime"}
    response = client.post("/api/historical-regimes/preview-runs", json=payload)
    assert response.status_code == 422, response.text
    assert "NON_CAUSAL_REALTIME_GRAPH" in response.text
    assert not v2_service._jobs
    saved = v2_service.create_definition(definition)
    response = client.post("/api/historical-regimes/run", json={
        "definition": {"schema_version": "2.0", "id": saved["id"], "revision": saved["revision"]},
        "mode": "realtime",
    })
    assert response.status_code == 422, response.text
    assert response.json()["detail"]["code"] == "NON_CAUSAL_REALTIME_GRAPH"
    payload["mode"] = "retrospective"
    response = client.post("/api/historical-regimes/preview-runs", json=payload)
    assert response.status_code == 202, response.text
    assert _wait_for_preview(client, response.json()["id"])["status"] == "completed"


@pytest.mark.parametrize("metadata", [
    {"supports_realtime": False}, {"causal": False}, {"repaints": True}, {"supports_realtime": None},
])
def test_realtime_gate_checks_all_causality_flags(
    v2_service: RegimeGraphV2Service, monkeypatch: pytest.MonkeyPatch, metadata: dict,
) -> None:
    from custom_indicators.errors import ValidationError

    schema = dict(v2_service_module.NODE_REGISTRY["filter.ema"], **metadata)
    monkeypatch.setitem(v2_service_module.NODE_REGISTRY, "filter.ema", schema)
    with pytest.raises(ValidationError) as exc:
        v2_service.create_preview(_definition(), compile_token=None, mode="realtime")
    assert exc.value.code == "NON_CAUSAL_REALTIME_GRAPH"
    assert not v2_service._jobs

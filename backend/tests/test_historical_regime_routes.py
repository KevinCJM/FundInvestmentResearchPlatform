from __future__ import annotations

import copy
import math
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from historical_regimes.service import HistoricalRegimeService
from services import historical_regime_routes


def _rows(count: int = 80, *, missing_at: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    start = date(2020, 1, 1)
    for index in range(count):
        if index < 25:
            value = 100.0 + index
        elif index < 45:
            value = 125.0 + 0.08 * math.sin(index)
        else:
            value = 125.0 - 0.8 * (index - 45)
        if index == missing_at:
            value = None
        observation_date = start + timedelta(days=index)
        rows.append(
            {
                "observation_date": observation_date.isoformat(),
                "available_at": (observation_date + timedelta(days=1)).isoformat(),
                "value": value,
                "revision": 1,
            }
        )
    return rows


def _definition(
    rows: list[dict[str, Any]] | None = None,
    *,
    filter_name: str = "ema",
    family: str = "causal_filter",
) -> dict[str, Any]:
    return {
        "name": "沪深300牛熊震荡验收",
        "description": "使用内联固定样本验证历史情景识别契约。",
        "template_id": "bull-bear-causal",
        "target": {
            "kind": "inline",
            "series_id": "000300.SH",
            "name": "沪深300",
            "frequency": "daily",
            "rows": rows if rows is not None else _rows(),
        },
        "features": {
            "transform": "none",
            "filter": filter_name,
            "window": 6,
            "slope_window": 2,
            "volatility_window": 4,
        },
        "algorithm": {
            "family": family,
            "parameters": {
                "bull_enter": 0.15,
                "bull_exit": 0.03,
                "bear_enter": -0.15,
                "bear_exit": -0.03,
                "confirmation": 2,
                "min_duration": 3,
                "window": 5,
                "min_move": 0.03,
            },
        },
        "states": [
            {"id": "bull", "label": "牛市", "role": "positive", "color": "#16a34a", "order": 1},
            {"id": "sideways", "label": "震荡市", "role": "neutral", "color": "#64748b", "order": 2},
            {"id": "bear", "label": "熊市", "role": "negative", "color": "#dc2626", "order": 3},
        ],
        "validation": {
            "walk_forward": True,
            "folds": 2,
            "stability_perturbation": 0.1,
        },
        "usage_intent": "taa",
    }


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> TestClient:
    service = HistoricalRegimeService(tmp_path, tmp_path)
    monkeypatch.setattr(
        historical_regime_routes,
        "historical_regime_service",
        service,
    )
    app = FastAPI()
    app.state.historical_regime_service = service
    app.include_router(historical_regime_routes.router)
    return TestClient(app)


def _service(client: TestClient) -> HistoricalRegimeService:
    return client.app.state.historical_regime_service


def _create_definition(client: TestClient, definition: dict[str, Any] | None = None) -> dict[str, Any]:
    seeded = _service(client).create_definition(definition or _definition())
    response = client.get(
        f"/api/historical-regimes/definitions/{seeded['id']}?revision={seeded['revision']}"
    )
    assert response.status_code == 200, response.text
    assert response.json() == seeded
    return response.json()


def _run_reference(
    client: TestClient,
    definition: dict[str, Any],
    *,
    mode: str = "realtime",
) -> dict[str, Any]:
    seeded = _service(client).run(
        {"id": definition["id"], "revision": definition["revision"]},
        mode,
    )
    response = client.get(f"/api/historical-regimes/runs/{seeded['id']}")
    assert response.status_code == 200, response.text
    assert response.json() == seeded
    return response.json()


def test_production_app_registers_historical_regime_routes() -> None:
    from app import app as production_app

    paths = set(production_app.openapi()["paths"])
    assert {
        "/api/historical-regimes/meta",
        "/api/historical-regimes/definitions",
        "/api/historical-regimes/run",
        "/api/historical-regimes/runs",
        "/api/historical-regimes/runs/{run_id}/publish",
        "/api/historical-regimes/compare",
    }.issubset(paths)


def test_meta_exposes_p0_templates_and_dual_track_contract(client: TestClient) -> None:
    response = client.get("/api/historical-regimes/meta")

    assert response.status_code == 200
    meta = response.json()
    assert meta["schema_version"]
    assert {item["id"] for item in meta["modes"]} == {"realtime", "retrospective"}
    assert {item["id"] for item in meta["templates"]} == {
        "bull-bear-causal",
        "merrill-clock",
        "size-rotation",
        "growth-value-rotation",
    }
    assert {item["id"] for item in meta["application_targets"]} == {
        "research_display",
        "product_research",
        "formal_backtest",
        "taa",
    }
    assert meta["feature_catalog"]
    assert {item["id"] for item in meta["algorithm_families"]}.issuperset(
        {"causal_filter", "turning_point", "merrill_clock", "relative_strength"}
    )


def test_definition_versions_and_run_analytical_snapshot_are_immutable(client: TestClient) -> None:
    created = _create_definition(client)
    first_run = _run_reference(client, created)
    first_snapshot = copy.deepcopy(first_run)

    update = copy.deepcopy(created)
    update["name"] = "沪深300牛熊震荡验收 v2"
    update["algorithm"]["parameters"]["bull_enter"] = 0.25
    update_response = client.put(
        f"/api/historical-regimes/definitions/{created['id']}",
        json=update,
    )
    assert update_response.status_code == 409
    assert update_response.json()["detail"]["code"] == "V1_DEFINITION_READ_ONLY"
    updated = _service(client).update_definition(
        created["id"], created["revision"], update
    )
    assert updated["revision"] == created["revision"] + 1

    historical = client.get(
        f"/api/historical-regimes/definitions/{created['id']}?revision={created['revision']}"
    )
    assert historical.status_code == 200
    assert historical.json()["name"] == created["name"]

    stored_run = client.get(f"/api/historical-regimes/runs/{first_run['id']}")
    assert stored_run.status_code == 200
    assert stored_run.json() == first_snapshot
    assert stored_run.json()["immutable"] is True
    assert stored_run.json()["definition_revision"] == created["revision"]
    assert stored_run.json()["definition"]["name"] == created["name"]
    assert stored_run.json()["data_snapshot"]["fingerprint"]
    assert stored_run.json()["content_hash"]


def test_realtime_run_has_ordered_temporal_fields_and_consistent_segments(client: TestClient) -> None:
    created = _create_definition(client)
    run = _run_reference(client, created)

    assert run["immutable"] is True
    assert run["mode"] == "realtime"
    assert run["series"]
    assert run["segments"]
    assert run["causality"]["realtime_eligible"] is True
    assert run["stability"]["prefix_invariance"]["status"] == "passed"
    assert run["stability"]["prefix_invariance"]["revisions"] == 0

    observation_dates = [point["observation_date"] for point in run["series"]]
    assert observation_dates == sorted(observation_dates)
    for point in run["series"]:
        assert point["date"] == point["observation_date"]
        assert point["observation_date"] <= point["data_available_at"] <= point["recognized_at"]
        if point["executable"]:
            assert point["effective_date"] is not None
            assert point["recognized_at"] <= point["effective_date"]
        else:
            assert point["effective_date"] is None
        assert point["signal_date"] == point["recognized_at"]

    positions = {point["observation_date"]: index for index, point in enumerate(run["series"])}
    covered: set[int] = set()
    for segment in run["segments"]:
        start = positions[segment["start_date"]]
        end = positions[segment["end_date"]]
        assert end >= start
        subset = run["series"][start : end + 1]
        assert segment["duration_observations"] == len(subset)
        assert {point["state_id"] for point in subset} == {segment["state_id"]}
        assert segment["state_label"] == subset[0]["state_label"]
        assert segment["recognized_at"] == subset[0]["recognized_at"]
        assert segment["effective_start"] == subset[0]["effective_date"]
        assert not covered.intersection(range(start, end + 1))
        covered.update(range(start, end + 1))
        if len(subset) >= 2:
            expected_return = subset[-1]["value"] / subset[0]["value"] - 1.0
            assert segment["return"] == pytest.approx(expected_return)

    classified = {
        index
        for index, point in enumerate(run["series"])
        if point["state_id"] != "unclassified"
    }
    assert covered == classified


def test_realtime_output_is_prefix_invariant(client: TestClient) -> None:
    full_rows = _rows(80)
    full_series = _service(client).run(
        _definition(full_rows), "realtime"
    )["series"]

    for prefix_length in (20, 35, 55, 70):
        full = full_series[:prefix_length]
        prefix = _service(client).run(
            _definition(full_rows[:prefix_length]), "realtime"
        )["series"]
        assert len(full) == len(prefix)
        for full_point, prefix_point in zip(full, prefix):
            assert full_point["observation_date"] == prefix_point["observation_date"]
            assert full_point["state_id"] == prefix_point["state_id"]
            assert full_point["filtered_value"] == pytest.approx(prefix_point["filtered_value"], nan_ok=True)
            assert full_point["score"] == pytest.approx(prefix_point["score"], nan_ok=True)
            assert full_point["features"] == pytest.approx(prefix_point["features"], nan_ok=True)


def test_missing_input_remains_unavailable_instead_of_becoming_zero(client: TestClient) -> None:
    missing_index = 30
    run = _service(client).run(
        _definition(_rows(missing_at=missing_index)), "realtime"
    )
    point = run["series"][missing_index]

    assert point["value"] is None
    assert point["state_id"] == "unclassified"
    assert point["confidence"] is None
    assert point["probabilities"] == {}
    assert any("缺失" in reason for reason in point["reasons"])


def test_non_causal_algorithms_are_blocked_from_realtime_and_formal_publication(
    client: TestClient,
) -> None:
    definition = _definition(filter_name="zero_phase")
    created = _create_definition(client, definition)

    realtime = client.post(
        "/api/historical-regimes/run",
        json={
            "definition": {"id": created["id"], "revision": created["revision"]},
            "mode": "realtime",
        },
    )
    assert realtime.status_code == 409
    assert realtime.json()["detail"]["code"] == "V1_RUN_READ_ONLY"

    retrospective = _run_reference(client, created, mode="retrospective")
    assert retrospective["causality"]["is_causal"] is False
    assert retrospective["causality"]["uses_future_data"] is True
    assert retrospective["causality"]["repaints"] is True
    assert set(retrospective["causality"]["publish_eligible_usages"]) == {
        "research_display",
        "product_research",
    }

    blocked = client.post(
        f"/api/historical-regimes/runs/{retrospective['id']}/publish",
        json={"usage": ["taa", "formal_backtest"]},
    )
    assert blocked.status_code == 409
    assert blocked.json()["detail"]["code"] == "V1_RUN_PUBLICATION_READ_ONLY"

    allowed = client.post(
        f"/api/historical-regimes/runs/{retrospective['id']}/publish",
        json={"usage": ["research_display", "product_research"], "note": "仅用于研究解释"},
    )
    assert allowed.status_code == 409
    assert allowed.json()["detail"]["code"] == "V1_RUN_PUBLICATION_READ_ONLY"
    _service(client).publish(
        retrospective["id"],
        ["research_display", "product_research"],
        "既有发布记录",
    )
    stored = client.get(f"/api/historical-regimes/runs/{retrospective['id']}")
    assert stored.status_code == 200
    assert {item["usage"] for item in stored.json()["publications"]} == {
        "research_display",
        "product_research",
    }


def test_causal_publication_keeps_analytical_snapshot_and_has_audit_lineage(client: TestClient) -> None:
    created = _create_definition(client)
    run = _run_reference(client, created)
    analytical_keys = {
        key: copy.deepcopy(run[key])
        for key in (
            "definition",
            "data_snapshot",
            "series",
            "segments",
            "conditional_stats",
            "transition",
            "causality",
            "stability",
            "walk_forward",
            "content_hash",
        )
    }

    blocked = client.post(
        f"/api/historical-regimes/runs/{run['id']}/publish",
        json={"usage": ["formal_backtest", "taa"], "note": "通过实时因果门禁"},
    )
    assert blocked.status_code == 409
    assert blocked.json()["detail"]["code"] == "V1_RUN_PUBLICATION_READ_ONLY"
    response = _service(client).publish(
        run["id"], ["formal_backtest", "taa"], "既有发布记录"
    )
    publications = response["publications"]
    assert {item["usage"] for item in publications} == {"formal_backtest", "taa"}
    for publication in publications:
        assert publication["run_id"] == run["id"]
        assert publication["definition_revision"] == created["revision"]
        assert publication["run_content_hash"] == run["content_hash"]
        assert publication["gate"] == "causality_passed"

    stored = client.get(f"/api/historical-regimes/runs/{run['id']}")
    assert stored.status_code == 200
    assert stored.json()["immutable"] is True
    for key, value in analytical_keys.items():
        assert stored.json()[key] == value


def test_v1_definition_run_and_publication_writes_are_read_only(client: TestClient) -> None:
    create_response = client.post(
        "/api/historical-regimes/definitions",
        json=_definition(),
    )
    assert create_response.status_code == 409
    assert create_response.json()["detail"]["code"] == "V1_DEFINITION_READ_ONLY"

    saved = _create_definition(client)
    update = copy.deepcopy(saved)
    update["name"] = "不允许原地更新"
    update_response = client.put(
        f"/api/historical-regimes/definitions/{saved['id']}",
        json=update,
    )
    assert update_response.status_code == 409
    assert update_response.json()["detail"]["code"] == "V1_DEFINITION_READ_ONLY"

    run_response = client.post(
        "/api/historical-regimes/run",
        json={
            "definition": {"id": saved["id"], "revision": saved["revision"]},
            "mode": "realtime",
        },
    )
    assert run_response.status_code == 409
    assert run_response.json()["detail"]["code"] == "V1_RUN_READ_ONLY"

    historical_run = _run_reference(client, saved)
    publish_response = client.post(
        f"/api/historical-regimes/runs/{historical_run['id']}/publish",
        json={"usage": "research_display"},
    )
    assert publish_response.status_code == 409
    assert publish_response.json()["detail"]["code"] == "V1_RUN_PUBLICATION_READ_ONLY"


def test_ensemble_aggregates_causality_and_rejects_low_consensus(client: TestClient) -> None:
    definition = _definition(_rows(100))
    definition["algorithm"] = {
        "family": "ensemble",
        "parameters": {
            "consensus_threshold": 0.75,
            "members": [
                {
                    "family": "causal_filter",
                    "weight": 0.5,
                    "parameters": {
                        "bull_enter": 0.1,
                        "bear_enter": -0.1,
                        "confirmation": 1,
                    },
                },
                {
                    "family": "causal_filter",
                    "weight": 0.5,
                    "parameters": {
                        "bull_enter": 1000.0,
                        "bear_enter": -1000.0,
                        "confirmation": 1,
                    },
                },
            ],
        },
    }
    definition["validation"] = {"walk_forward": False, "folds": 2}
    saved = _create_definition(client, definition)
    run = _run_reference(client, saved)

    diagnostics = run["algorithm_diagnostics"]
    assert diagnostics["model"] == "ensemble"
    assert diagnostics["consensus_threshold"] == pytest.approx(0.75)
    assert diagnostics["conflict_rejections"] > 0
    assert all(member["is_causal"] for member in diagnostics["members"])
    assert run["causality"]["is_causal"] is True
    assert run["causality"]["uses_future_data"] is False
    assert run["causality"]["repaints"] is False
    assert {"formal_backtest", "taa"}.issubset(run["causality"]["publish_eligible_usages"])
    assert run["stability"]["realtime_monitoring"].keys() >= {
        "prefix_revisions",
        "prefix_revision_rate",
        "label_flips",
        "label_flip_rate",
    }

    rejected = [
        point
        for point in run["series"]
        if any("低于拒判阈值" in reason for reason in point["reasons"])
    ]
    assert rejected
    assert all(point["state_id"] == "unclassified" for point in rejected)
    assert all(point["confidence"] is None for point in rejected)


def test_ensemble_propagates_non_causality_and_blocks_recursive_members(client: TestClient) -> None:
    mixed = _definition(_rows(100))
    mixed["algorithm"] = {
        "family": "ensemble",
        "parameters": {
            "consensus_threshold": 0.6,
            "members": [
                {"family": "causal_filter", "weight": 0.5, "parameters": {}},
                {
                    "family": "turning_point",
                    "weight": 0.5,
                    "parameters": {"window": 5, "min_move": 0.03},
                },
            ],
        },
    }
    mixed["validation"] = {"walk_forward": False, "folds": 2}
    saved = _create_definition(client, mixed)
    run = _run_reference(client, saved, mode="retrospective")
    assert run["causality"]["is_causal"] is False
    assert run["causality"]["uses_future_data"] is True
    assert run["causality"]["repaints"] is True
    assert set(run["causality"]["publish_eligible_usages"]) == {
        "research_display",
        "product_research",
    }
    blocked = client.post(
        f"/api/historical-regimes/runs/{run['id']}/publish",
        json={"usage": "taa"},
    )
    assert blocked.status_code == 409
    assert blocked.json()["detail"]["code"] == "V1_RUN_PUBLICATION_READ_ONLY"

    recursive = _definition(_rows(80))
    recursive["algorithm"] = {
        "family": "ensemble",
        "parameters": {
            "members": [
                {"family": "ensemble", "weight": 1.0, "parameters": {}},
                {"family": "causal_filter", "weight": 1.0, "parameters": {}},
            ]
        },
    }
    response = client.post(
        "/api/historical-regimes/run",
        json={"definition": recursive, "mode": "realtime"},
    )
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "V1_RUN_READ_ONLY"


def test_point_in_time_revision_policy_uses_first_release_for_realtime(client: TestClient) -> None:
    rows = _rows(12)
    observation = rows[5]["observation_date"]
    rows.append(
        {
            "observation_date": observation,
            "available_at": (date.fromisoformat(observation) + timedelta(days=20)).isoformat(),
            "value": 999.0,
            "revision": 2,
        }
    )
    definition = _definition(rows)

    realtime = _service(client).run(definition, "realtime")
    retrospective = _service(client).run(definition, "retrospective")

    realtime_point = next(
        point for point in realtime["series"] if point["observation_date"] == observation
    )
    retrospective_point = next(
        point for point in retrospective["series"] if point["observation_date"] == observation
    )
    assert realtime_point["value"] == pytest.approx(_rows(12)[5]["value"])
    assert realtime_point["is_final"] is False
    assert retrospective_point["value"] == 999.0
    assert retrospective["data_snapshot"]["revision_observations"] == 1


def test_existing_v1_runs_can_still_be_compared(client: TestClient) -> None:
    definition = _create_definition(client)
    first = _run_reference(client, definition, mode="realtime")
    second = _run_reference(client, definition, mode="retrospective")

    response = client.post(
        "/api/historical-regimes/compare",
        json={
            "run_ids": [first["id"], second["id"]],
            "reference_run_id": first["id"],
        },
    )

    assert response.status_code == 200, response.text
    assert response.json()["run_ids"] == [first["id"], second["id"]]

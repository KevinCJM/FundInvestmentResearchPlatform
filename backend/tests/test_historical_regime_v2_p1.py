from __future__ import annotations

import copy
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from historical_regimes.service import HistoricalRegimeService
from historical_regimes.numba_kernels import (
    gmm_fit_kernel,
    historical_regime_numba_status,
    hmm_fit_kernel,
    initialize_gaussian_means_kernel,
    markov_fit_kernel,
)
from historical_regimes.data import DataBundle
from historical_regimes.v2_contracts import parse_definition_v2
from historical_regimes.v2_numba import (
    KERNELS,
    calendar_bucket_kernel,
    calendar_resample_kernel,
    final_output_contract_kernel,
)
import historical_regimes.v2_service as v2_service_module
from historical_regimes.v2_service import PortValue, RegimeGraphV2Service, _definition_output_frequency
from historical_regimes.v2_registry import NODE_REGISTRY
from research_series.service import write_upload_artifact
from services import historical_regime_routes


def _rows(count: int = 96) -> list[dict[str, Any]]:
    start = date(2020, 1, 1)
    return [
        {
            "observation_date": (start + timedelta(days=index)).isoformat(),
            "available_at": (start + timedelta(days=index + 1)).isoformat(),
            "value": 100.0 + index * 0.2 + np.sin(index / 4.0),
        }
        for index in range(count)
    ]


def _definition() -> dict[str, Any]:
    return {
        "schema_version": "2.0",
        "name": "P1 验收图谱",
        "graph": {
            "nodes": [
                {"id": "source", "type": "source.inline", "parameters": {"rows": _rows()}},
                {
                    "id": "returns",
                    "type": "transform.return",
                    "parameters": {"window": 1},
                    "inputs": {"value": {"node_id": "source", "port": "value"}},
                },
                {
                    "id": "threshold",
                    "type": "model.threshold",
                    "parameters": {"upper": 0.001, "lower": -0.001},
                    "inputs": {"value": {"node_id": "returns", "port": "value"}},
                },
                {
                    "id": "hysteresis",
                    "type": "post.hysteresis",
                    "parameters": {
                        "upper_enter": 0.0015,
                        "upper_exit": 0.0002,
                        "lower_enter": -0.0015,
                        "lower_exit": -0.0002,
                    },
                    "inputs": {"score": {"node_id": "threshold", "port": "score"}},
                },
                {
                    "id": "confirmed",
                    "type": "post.confirmation",
                    "parameters": {"confirmation": 2, "min_duration": 2},
                    "inputs": {"state": {"node_id": "hysteresis", "port": "state"}},
                },
            ],
            "outputs": {"state": {"node_id": "confirmed", "port": "state"}},
            "exposed_node_ids": ["returns", "threshold", "hysteresis"],
        },
        "states": [
            {"id": "up", "label": "上行", "role": "positive", "color": "#16a34a", "order": 1},
            {"id": "flat", "label": "震荡", "role": "neutral", "color": "#64748b", "order": 2},
            {"id": "down", "label": "下行", "role": "negative", "color": "#dc2626", "order": 3},
        ],
        "evaluation_targets": [],
        "validation": {
            "walk_forward": True,
            "folds": 3,
            "min_parameter_agreement": 0.0,
            "max_label_flip_rate": 1.0,
            "min_classified_ratio": 0.1,
            "min_walk_forward_classified_ratio": 0.1,
        },
        "usage_intent": "taa",
    }


def _latent_definition(
    source: dict[str, Any] | None = None,
    *,
    model_type: str = "model.markov",
) -> dict[str, Any]:
    definition = _definition()
    definition["name"] = "可重训隐状态图谱"
    definition["graph"] = {
        "nodes": [
            source
            or {
                "id": "source",
                "type": "source.inline",
                "parameters": {"rows": _rows(144)},
            },
            {
                "id": "matrix",
                "type": "feature.matrix",
                "inputs": {"feature_1": {"node_id": "source", "port": "value"}},
            },
            {
                "id": "latent",
                "type": model_type,
                "parameters": {
                    "components": 3,
                    "initial_train_size": 24,
                    "iterations": 5,
                },
                "inputs": {"features": {"node_id": "matrix", "port": "features"}},
            },
        ],
        "outputs": {
            "state": {"node_id": "latent", "port": "state"},
            "probabilities": {"node_id": "latent", "port": "probabilities"},
            "confidence": {"node_id": "latent", "port": "confidence"},
        },
        "exposed_node_ids": ["matrix", "latent"],
    }
    definition["validation"] = {
        "walk_forward": True,
        "folds": 3,
        "stability_perturbation": 0.0,
        "sensitivity_candidates": 1,
        "min_parameter_agreement": 0.0,
        "max_label_flip_rate": 1.0,
        "min_classified_ratio": 0.1,
        "min_walk_forward_classified_ratio": 0.1,
    }
    return definition


def _upload_source(service: RegimeGraphV2Service, rows: list[dict[str, Any]]) -> dict[str, Any]:
    frame = pd.DataFrame(rows)
    frame["vintage"] = frame.get("vintage", None)
    frame["revision"] = frame.get("revision", 1)
    artifact = write_upload_artifact(service.market_data_dir, frame)
    return {
        "id": "source",
        "type": "source.upload",
        "parameters": {
            "artifact_id": artifact["artifact_id"],
            "checksum": artifact["checksum"],
            "format": "parquet",
            "availability_mode": "point_in_time",
            "frequency": "daily",
        },
    }


def _diagnostic_codes(result: dict[str, Any]) -> set[str]:
    return {str(item.get("code")) for item in result.get("errors", [])}


@pytest.fixture
def service(tmp_path: Path) -> RegimeGraphV2Service:
    return RegimeGraphV2Service(tmp_path, tmp_path)


@pytest.fixture
def client(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    service: RegimeGraphV2Service,
) -> TestClient:
    monkeypatch.setattr(historical_regime_routes, "regime_graph_v2_service", service)
    monkeypatch.setattr(
        historical_regime_routes,
        "historical_regime_service",
        HistoricalRegimeService(tmp_path, tmp_path),
    )
    app = FastAPI()
    app.include_router(historical_regime_routes.router)
    return TestClient(app)


def test_infer_returns_shapes_dependencies_causality_and_cost_without_prepare(
    service: RegimeGraphV2Service,
) -> None:
    inference = service.infer(_definition())
    assert inference["valid"] is True
    assert inference["inferred"]["symbolic_shapes"]["threshold.probabilities"][1] == "S=3"
    assert inference["dependencies"]["direct"]["confirmed"] == ["hysteresis"]
    assert "source" in inference["dependencies"]["transitive"]["confirmed"]
    assert inference["causality"] == {
        "causal": True,
        "repaints": False,
        "realtime_supported": True,
        "noncausal_node_ids": [],
        "repaint_node_ids": [],
        "non_realtime_node_ids": [],
    }
    assert inference["cost_estimate"]["total_relative_units_per_observation"] > 0

    before = dict(service._plans)
    saved = service.create_definition(_definition())
    assert service._plans == before
    assert saved["revision"] == 1


def test_user_template_and_subgraph_are_versioned_and_whitelist_only(
    service: RegimeGraphV2Service,
) -> None:
    template = service.create_graph_asset(
        "template",
        {"name": "我的模板", "description": "可编辑", "definition": _definition()},
    )
    instantiated = service.instantiate_graph_asset(template["id"], 1)
    assert instantiated["definition"].get("id") is None
    assert instantiated["source"]["asset_revision"] == 1
    assert instantiated["inference"]["valid"] is True

    changed = service.update_graph_asset(
        template["id"],
        1,
        {"name": "我的模板 v2", "definition": _definition()},
    )
    assert changed["revision"] == 2
    assert service.get_graph_asset(template["id"], 1)["name"] == "我的模板"

    subgraph = service.create_graph_asset(
        "subgraph",
        {
            "name": "收益与阈值",
            "graph": {"nodes": _definition()["graph"]["nodes"][:3], "edges": []},
        },
    )
    assert service.instantiate_graph_asset(subgraph["id"])["kind"] == "subgraph"

    invalid = copy.deepcopy(_definition())
    invalid["graph"]["nodes"][1]["type"] = "model.external_optimized"
    with pytest.raises(Exception) as exc_info:
        service.create_graph_asset("template", {"name": "非法", "definition": invalid})
    assert getattr(exc_info.value, "code", None) in {
        "INVALID_REGIME_GRAPH_V2",
        "REGIME_GRAPH_ASSET_NODE_NOT_AVAILABLE",
    }


@pytest.mark.parametrize(
    ("graph", "expected_code"),
    [
        (
            {
                "nodes": [
                    {"id": "source", "type": "source.inline", "parameters": {"rows": _rows()}},
                    {
                        "id": "confirmed",
                        "type": "post.confirmation",
                        "inputs": {"state": {"node_id": "source", "port": "value"}},
                    },
                ],
                "edges": [],
            },
            "PORT_TYPE_MISMATCH",
        ),
        (
            {
                "nodes": [
                    {
                        "id": "left",
                        "type": "transform.identity",
                        "inputs": {"value": {"node_id": "right", "port": "value"}},
                    },
                    {
                        "id": "right",
                        "type": "transform.identity",
                        "inputs": {"value": {"node_id": "left", "port": "value"}},
                    },
                ],
                "edges": [],
            },
            "GRAPH_CYCLE",
        ),
    ],
)
def test_user_subgraphs_fail_closed_on_semantic_type_and_cycle_errors(
    service: RegimeGraphV2Service,
    graph: dict[str, Any],
    expected_code: str,
) -> None:
    with pytest.raises(Exception) as exc_info:
        service.create_graph_asset(
            "subgraph",
            {"name": "非法子图", "graph": graph},
        )
    assert getattr(exc_info.value, "code", None) == "INVALID_REGIME_SUBGRAPH"
    codes = {
        str(item.get("code"))
        for item in getattr(exc_info.value, "diagnostics", [])
    }
    assert expected_code in codes


def test_user_subgraph_rejects_edges_that_disagree_with_node_inputs(
    service: RegimeGraphV2Service,
) -> None:
    graph = {
        "nodes": [
            {"id": "source", "type": "source.inline", "parameters": {"rows": _rows()}},
            {
                "id": "returns",
                "type": "transform.return",
                "inputs": {"value": {"node_id": "source", "port": "value"}},
            },
        ],
        "edges": [
            {
                "source": {"node_id": "source", "port": "value"},
                "target": {"node_id": "returns", "port": "wrong_port"},
            }
        ],
    }
    with pytest.raises(Exception) as exc_info:
        service.create_graph_asset("subgraph", {"name": "边不一致", "graph": graph})
    assert getattr(exc_info.value, "code", None) == "INVALID_REGIME_GRAPH_V2"


def test_calendar_resample_and_score_hysteresis_use_frozen_njit_signatures(
    service: RegimeGraphV2Service,
) -> None:
    before = {key: tuple(dispatcher.signatures) for key, dispatcher in KERNELS.items()}
    frame = pd.DataFrame(_rows(70))
    dates = np.ascontiguousarray(
        pd.to_datetime(frame["observation_date"]).to_numpy(dtype="datetime64[ns]").view(np.int64)
    )
    available = np.ascontiguousarray(
        pd.to_datetime(frame["available_at"]).to_numpy(dtype="datetime64[ns]").view(np.int64)
    )
    values, sampled_dates, sampled_available = calendar_resample_kernel(
        np.ascontiguousarray(frame["value"].to_numpy(dtype=np.float64)),
        dates,
        available,
        np.int64(2),
        np.int64(2),
    )
    assert values.shape == sampled_dates.shape == sampled_available.shape
    assert 2 <= len(values) <= 3

    prepared = service.prepare(_definition())
    service_definition = parse_definition_v2(_definition())
    execution = service._execute_graph(
        None,
        service_definition,
        "realtime",
        None,
        plan=prepared,
    )
    assert execution["node_outputs"]["hysteresis"]["state"].values.shape[0] == len(_rows())
    assert service_definition.schema_version == "2.0"
    after = {key: tuple(dispatcher.signatures) for key, dispatcher in KERNELS.items()}
    assert after == before
    assert all(dispatcher._can_compile is False for dispatcher in KERNELS.values())


def test_index_source_frequency_is_applied_before_regime_execution(
    service: RegimeGraphV2Service,
    monkeypatch,
) -> None:
    rows = _rows(35)
    frame = pd.DataFrame(rows)
    frame["observation_date"] = pd.to_datetime(frame["observation_date"])
    frame["available_at"] = pd.to_datetime(frame["available_at"])
    raw_bundle = DataBundle(
        frame=frame,
        snapshot={
            "kind": "index",
            "fingerprint": "raw-index-fixture",
            "selected_observations": len(frame),
            "first_observation_date": frame["observation_date"].iloc[0].date().isoformat(),
            "last_observation_date": frame["observation_date"].iloc[-1].date().isoformat(),
            "latest_available_at": frame["available_at"].max().date().isoformat(),
        },
    )
    monkeypatch.setattr(service, "_bound_source_root", lambda *_args, **_kwargs: service.market_data_dir)
    monkeypatch.setattr(v2_service_module, "resolve_target", lambda *_args, **_kwargs: raw_bundle)

    definition = _definition()
    definition["graph"] = {
        "nodes": [
            {
                "id": "source",
                "type": "source.index",
                "parameters": {
                    "ts_code": "000300.SH",
                    "source_api": "index_daily",
                    "field": "close",
                    "frequency": "weekly",
                },
            },
            {
                "id": "threshold",
                "type": "model.threshold",
                "parameters": {"upper": 101.0, "lower": 99.0},
                "inputs": {"value": {"node_id": "source", "port": "value"}},
            },
        ],
        "outputs": {"state": {"node_id": "threshold", "port": "state"}},
        "exposed_node_ids": ["source"],
    }
    parsed = parse_definition_v2(definition)
    result = service._execute_graph(None, parsed, "retrospective", None)

    dates = np.ascontiguousarray(frame["observation_date"].to_numpy(dtype="datetime64[ns]").view(np.int64))
    available = np.ascontiguousarray(frame["available_at"].to_numpy(dtype="datetime64[ns]").view(np.int64))
    expected_values, expected_dates, _ = calendar_resample_kernel(
        np.ascontiguousarray(frame["value"].to_numpy(dtype=np.float64)),
        dates,
        available,
        np.int64(1),
        np.int64(1),
    )
    assert result["result"]["frequency"] == "weekly"
    assert result["result"]["row_count"] == len(expected_values) < len(frame)
    np.testing.assert_array_equal(
        np.array([pd.Timestamp(row["observation_date"]).value for row in result["series"]], dtype=np.int64),
        expected_dates,
    )
    snapshot = result["result"]["data_snapshots"]["source"]
    assert snapshot["raw_selected_observations"] == len(frame)
    assert snapshot["selected_observations"] == len(expected_values)
    assert snapshot["output_frequency"] == "weekly"
    assert snapshot["frequency_sampling"] == "last_observation_per_calendar_bucket"


def test_effective_frequency_follows_the_final_state_axis() -> None:
    definition = _definition()
    definition["graph"] = {
        "nodes": [
            {"id": "source", "type": "source.inline", "parameters": {"rows": _rows(), "frequency": "daily"}},
            {"id": "weekly", "type": "align.resample", "parameters": {"frequency": "weekly", "aggregation": "last"}, "inputs": {"value": {"node_id": "source", "port": "value"}}},
            {"id": "threshold", "type": "model.threshold", "parameters": {"upper": 0.0, "lower": -1.0}, "inputs": {"value": {"node_id": "weekly", "port": "value"}}},
        ],
        "outputs": {"state": {"node_id": "threshold", "port": "state"}},
    }
    assert _definition_output_frequency(parse_definition_v2(definition)) == "weekly"


@pytest.mark.parametrize("observation_count", [5_000, 20_000])
def test_large_v2_graph_execution_never_adds_numba_signatures(
    service: RegimeGraphV2Service,
    observation_count: int,
) -> None:
    definition = _definition()
    definition["graph"]["nodes"][0]["parameters"]["rows"] = _rows(observation_count)
    own_before = {key: tuple(dispatcher.signatures) for key, dispatcher in KERNELS.items()}
    shared_before = {
        item["kernel_id"]: tuple(item["compiled_signatures"])
        for item in historical_regime_numba_status()["kernels"]
    }
    prepared = service.prepare(definition)
    execution = service._execute_graph(
        None,
        parse_definition_v2(definition),
        "realtime",
        None,
        plan=prepared,
    )
    assert execution["result"]["row_count"] == observation_count
    assert {key: tuple(dispatcher.signatures) for key, dispatcher in KERNELS.items()} == own_before
    shared_after = {
        item["kernel_id"]: tuple(item["compiled_signatures"])
        for item in historical_regime_numba_status()["kernels"]
    }
    assert shared_after == shared_before


def test_calendar_resample_respects_week_month_quarter_leap_boundaries_and_aggregations() -> None:
    boundary_dates = pd.to_datetime(
        [
            "2023-12-31",  # Sunday
            "2024-01-01",  # Monday and new year
            "2024-02-28",
            "2024-02-29",  # leap day
            "2024-03-01",
            "2024-03-31",
            "2024-04-01",  # new quarter
            "2100-02-28",  # century year is not a leap year
            "2100-03-01",
        ]
    ).to_numpy(dtype="datetime64[ns]").view(np.int64)
    weekly = [calendar_bucket_kernel(np.int64(item), np.int64(1)) for item in boundary_dates]
    monthly = [calendar_bucket_kernel(np.int64(item), np.int64(2)) for item in boundary_dates]
    quarterly = [calendar_bucket_kernel(np.int64(item), np.int64(3)) for item in boundary_dates]
    yearly = [calendar_bucket_kernel(np.int64(item), np.int64(4)) for item in boundary_dates]
    assert weekly[0] != weekly[1]
    assert monthly[2] == monthly[3] != monthly[4]
    assert monthly[7] != monthly[8]
    assert quarterly[4] == quarterly[5] != quarterly[6]
    assert yearly[0] != yearly[1]

    dates = pd.to_datetime(
        ["2024-01-02", "2024-01-31", "2024-02-01", "2024-02-29"]
    ).to_numpy(dtype="datetime64[ns]").view(np.int64)
    available = pd.to_datetime(
        ["2024-01-03", "2024-02-02", "2024-02-02", "2024-03-02"]
    ).to_numpy(dtype="datetime64[ns]").view(np.int64)
    values = np.ascontiguousarray([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    expected = {
        0: ([1.0, 3.0], [0, 2], [0, 2]),
        1: ([2.0, 4.0], [1, 3], [1, 3]),
        2: ([1.5, 3.5], [1, 3], [1, 3]),
        3: ([3.0, 7.0], [1, 3], [1, 3]),
    }
    for aggregation_code, (expected_values, date_positions, available_positions) in expected.items():
        sampled_values, sampled_dates, sampled_available = calendar_resample_kernel(
            values,
            np.ascontiguousarray(dates),
            np.ascontiguousarray(available),
            np.int64(2),
            np.int64(aggregation_code),
        )
        assert sampled_values.tolist() == expected_values
        assert sampled_dates.tolist() == [int(dates[index]) for index in date_positions]
        assert sampled_available.tolist() == [
            int(available[index]) for index in available_positions
        ]


def test_realtime_calendar_resample_never_emits_an_open_month_bucket(
    service: RegimeGraphV2Service,
) -> None:
    start = date(2020, 1, 1)
    rows = [
        {
            "observation_date": (start + timedelta(days=index)).isoformat(),
            "available_at": (start + timedelta(days=index)).isoformat(),
            "value": 100.0 + index,
        }
        for index in range((date(2020, 8, 1) - start).days + 1)
    ]
    definition = _definition()
    definition["graph"] = {
        "nodes": [
            {
                "id": "source",
                "type": "source.inline",
                "parameters": {"rows": rows, "frequency": "daily"},
            },
            {
                "id": "monthly",
                "type": "align.resample",
                "parameters": {"frequency": "monthly", "aggregation": "last"},
                "inputs": {"value": {"node_id": "source", "port": "value"}},
            },
            {
                "id": "threshold",
                "type": "model.threshold",
                "parameters": {"upper": 0.0, "lower": -1.0},
                "inputs": {"value": {"node_id": "monthly", "port": "value"}},
            },
        ],
        "outputs": {"state": {"node_id": "threshold", "port": "state"}},
        "exposed_node_ids": ["monthly"],
    }
    plan = service.prepare(definition)
    parsed = parse_definition_v2(definition)
    cache: dict[str, Any] = {}
    mid_month = service._execute_graph(
        None,
        parsed,
        "realtime",
        "2020-07-15",
        plan=plan,
        source_cache=cache,
    )
    after_close = service._execute_graph(
        None,
        parsed,
        "realtime",
        "2020-08-01",
        plan=plan,
        source_cache=cache,
    )
    mid_dates = [item["observation_date"] for item in mid_month["series"]]
    closed_dates = [item["observation_date"] for item in after_close["series"]]
    assert mid_dates[-1] == "2020-06-30"
    assert closed_dates[-1] == "2020-07-31"
    assert closed_dates[:-1] == mid_dates


def test_final_outputs_require_one_decision_root_and_one_time_axis(
    service: RegimeGraphV2Service,
) -> None:
    sibling = _definition()
    sibling["graph"] = {
        "nodes": [
            {"id": "source", "type": "source.inline", "parameters": {"rows": _rows()}},
            {
                "id": "state_model",
                "type": "model.threshold",
                "inputs": {"value": {"node_id": "source", "port": "value"}},
            },
            {
                "id": "probability_model",
                "type": "model.threshold",
                "parameters": {"upper": 101.0, "lower": 99.0},
                "inputs": {"value": {"node_id": "source", "port": "value"}},
            },
        ],
        "outputs": {
            "state": {"node_id": "state_model", "port": "state"},
            "probabilities": {
                "node_id": "probability_model",
                "port": "probabilities",
            },
        },
        "exposed_node_ids": [],
    }
    sibling_result = service.infer(sibling)
    assert sibling_result["valid"] is False
    assert "MULTIPLE_STATE_ROOTS" in _diagnostic_codes(sibling_result)

    different_axis = copy.deepcopy(sibling)
    different_axis["graph"]["nodes"].insert(
        2,
        {
            "id": "monthly",
            "type": "align.resample",
            "parameters": {"frequency": "monthly", "aggregation": "last"},
            "inputs": {"value": {"node_id": "source", "port": "value"}},
        },
    )
    different_axis["graph"]["nodes"][-1]["inputs"] = {
        "value": {"node_id": "monthly", "port": "value"}
    }
    different_axis["graph"]["outputs"] = {
        "state": {"node_id": "state_model", "port": "state"},
        "confidence": {"node_id": "probability_model", "port": "confidence"},
    }
    axis_result = service.infer(different_axis)
    assert axis_result["valid"] is False
    assert "FINAL_OUTPUT_AXIS_MISMATCH" in _diagnostic_codes(axis_result)


def test_unknown_final_output_is_rejected_by_schema(
    service: RegimeGraphV2Service,
) -> None:
    definition = _definition()
    definition["graph"]["outputs"]["debug_score"] = {
        "node_id": "threshold",
        "port": "score",
    }
    result = service.infer(definition)
    assert result["valid"] is False
    assert "SCHEMA_VALIDATION_ERROR" in _diagnostic_codes(result)


def test_final_output_runtime_contract_rejects_short_or_invalid_confidence(
    service: RegimeGraphV2Service,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    definition = _definition()
    definition["graph"]["outputs"]["confidence"] = {
        "node_id": "threshold",
        "port": "confidence",
    }
    prepared = service.prepare(definition)
    parsed = parse_definition_v2(definition)
    original = service._execute_numeric_node

    def corrupt_confidence(node: Any, *args: Any, **kwargs: Any) -> dict[str, PortValue]:
        outputs = original(node, *args, **kwargs)
        if node.id == "threshold":
            port = outputs["confidence"]
            outputs["confidence"] = PortValue(
                np.ascontiguousarray(port.values[:-1], dtype=np.float64),
                port.dates,
                port.available,
            )
        return outputs

    monkeypatch.setattr(service, "_execute_numeric_node", corrupt_confidence)
    with pytest.raises(Exception) as exc_info:
        service._execute_graph(None, parsed, "retrospective", None, plan=prepared)
    assert getattr(exc_info.value, "code", None) == "REGIME_FINAL_OUTPUT_CONTRACT_VIOLATION"
    contract = getattr(exc_info.value, "diagnostics", [])[0]
    assert contract["confidence_length_mismatches"] == 1

    def corrupt_state(node: Any, *args: Any, **kwargs: Any) -> dict[str, PortValue]:
        outputs = original(node, *args, **kwargs)
        if node.id == "confirmed":
            port = outputs["state"]
            outputs["state"] = PortValue(
                np.ascontiguousarray(port.values[:-1], dtype=np.int64),
                port.dates,
                port.available,
            )
        return outputs

    monkeypatch.setattr(service, "_execute_numeric_node", corrupt_state)
    with pytest.raises(Exception) as state_exc:
        service._execute_graph(None, parsed, "retrospective", None, plan=prepared)
    assert getattr(state_exc.value, "code", None) == "REGIME_FINAL_OUTPUT_CONTRACT_VIOLATION"
    state_contract = getattr(state_exc.value, "diagnostics", [])[0]
    assert state_contract["value_length"] + 1 == state_contract["date_length"]

    states = np.ascontiguousarray([0, 1, 2], dtype=np.int64)
    diagnostics = final_output_contract_kernel(
        np.ascontiguousarray([1.2, 0.5, np.nan], dtype=np.float64),
        np.ascontiguousarray([0, -1, 4], dtype=np.int64),
        np.ascontiguousarray([0, 1], dtype=np.int64),
        np.ascontiguousarray([0, -1, 0], dtype=np.int64),
        states,
    )
    assert diagnostics.tolist() == [0, 2, 0, 1, 1, 1, 0, 0, 1]
    assert final_output_contract_kernel._can_compile is False


def test_batch_experiment_is_persisted_ranked_and_uses_one_source_scan(
    service: RegimeGraphV2Service,
) -> None:
    saved = service.create_definition(_definition())
    prepared = service.prepare(saved)
    before = {key: tuple(dispatcher.signatures) for key, dispatcher in KERNELS.items()}
    experiment = service.run_batch_experiment(
        {"schema_version": "2.0", "id": saved["id"], "revision": 1},
        [
            {
                "node_id": "threshold",
                "parameter": "upper",
                "values": [0.001, 0.002, 0.003],
            }
        ],
        compile_token=prepared["compile_token"],
    )
    assert experiment["candidate_count"] == 2
    assert [item["rank"] for item in experiment["ranking"]] == [1, 2]
    assert experiment["source_scan_cache"]["unique_source_views"] == 1
    assert experiment["calculation_audit"]["request_time_compilation"] == 0
    assert service.get_experiment(experiment["id"])["content_hash"] == experiment["content_hash"]
    assert {key: tuple(dispatcher.signatures) for key, dispatcher in KERNELS.items()} == before


def test_validation_policy_and_array_parameters_fail_closed(
    service: RegimeGraphV2Service,
) -> None:
    invalid_validation = _definition()
    invalid_validation["validation"]["unknown_gate"] = True
    assert service.infer(invalid_validation)["valid"] is False

    invalid_rate = _definition()
    invalid_rate["validation"]["max_label_flip_rate"] = 1.1
    assert service.infer(invalid_rate)["valid"] is False

    mapping = _definition()
    mapping["graph"]["nodes"].append(
        {
            "id": "mapping",
            "type": "post.component_map",
            "parameters": {"mapping": [0, "bad", 4]},
            "inputs": {"state": {"node_id": "confirmed", "port": "state"}},
        }
    )
    mapping["graph"]["outputs"]["state"] = {"node_id": "mapping", "port": "state"}
    mapping_result = service.infer(mapping)
    assert mapping_result["valid"] is False
    assert {
        "INVALID_NODE_PARAMETER_ITEM",
        "COMPONENT_MAPPING_OUT_OF_RANGE",
    }.issubset(_diagnostic_codes(mapping_result))

    ensemble = _definition()
    ensemble["graph"] = {
        "nodes": [
            {"id": "source", "type": "source.inline", "parameters": {"rows": _rows()}},
            {
                "id": "first",
                "type": "model.threshold",
                "inputs": {"value": {"node_id": "source", "port": "value"}},
            },
            {
                "id": "second",
                "type": "model.threshold",
                "parameters": {"upper": 101.0, "lower": 99.0},
                "inputs": {"value": {"node_id": "source", "port": "value"}},
            },
            {
                "id": "ensemble",
                "type": "model.ensemble",
                "parameters": {"weights": [0.0, 0.0, 0.0]},
                "inputs": {
                    "state_1": {"node_id": "first", "port": "state"},
                    "state_2": {"node_id": "second", "port": "state"},
                },
            },
        ],
        "outputs": {"state": {"node_id": "ensemble", "port": "state"}},
        "exposed_node_ids": [],
    }
    ensemble_result = service.infer(ensemble)
    assert ensemble_result["valid"] is False
    assert {
        "ENSEMBLE_WEIGHT_COUNT_MISMATCH",
        "ENSEMBLE_WEIGHTS_NOT_POSITIVE",
    }.issubset(_diagnostic_codes(ensemble_result))


def test_retrospective_as_of_selects_latest_vintage_available_at_that_date(
    service: RegimeGraphV2Service,
) -> None:
    start = date(2020, 1, 1)
    rows: list[dict[str, Any]] = []
    for index in range(30):
        observation = start + timedelta(days=index)
        rows.extend(
            [
                {
                    "observation_date": observation.isoformat(),
                    "available_at": (observation + timedelta(days=1)).isoformat(),
                    "value": 100.0 + index,
                    "revision": 1,
                    "vintage": "first",
                },
                {
                    "observation_date": observation.isoformat(),
                    "available_at": (observation + timedelta(days=30)).isoformat(),
                    "value": 200.0 + index,
                    "revision": 2,
                    "vintage": "final",
                },
            ]
        )
    definition = _definition()
    definition["graph"]["nodes"][0]["parameters"]["rows"] = rows
    prepared = service.prepare(definition)
    execution = service._execute_graph(
        None,
        parse_definition_v2(definition),
        "retrospective",
        "2020-01-25",
        plan=prepared,
        source_cache={},
    )
    assert execution["series"][0]["value"] == 100.0
    assert max(item["value"] for item in execution["series"] if item["value"] is not None) < 200.0
    snapshot = execution["result"]["data_snapshots"]["source"]
    assert snapshot["as_of"] == "2020-01-25"


def test_markov_uses_its_own_estimator_and_frozen_kernel_plan(
    service: RegimeGraphV2Service,
) -> None:
    definition = _latent_definition()
    prepared = service.prepare(definition)
    assert "markov_fit" in prepared["shared_kernel_ids"]
    assert "hmm_fit" not in prepared["shared_kernel_ids"]
    execution = service._execute_graph(
        None,
        parse_definition_v2(definition),
        "retrospective",
        None,
        plan=prepared,
    )
    audit = execution["result"]["diagnostics"]["model_audits"]["latent"]
    assert audit["estimation_method"] == "hard_gaussian_states_with_markov_transitions"
    assert audit["model_type"] == "model.markov"
    assert execution["result"]["diagnostics"]["request_time_compilation"] == 0


def test_latent_model_registry_exposes_experiment_initialization_controls() -> None:
    for model_type in ("model.hmm", "model.markov", "model.gmm"):
        schema = NODE_REGISTRY[model_type]["parameter_schema"]
        properties = schema["properties"]
        assert properties["components"]["default"] == 3
        assert properties["initialization_strategy"]["enum"] == [
            "quantile",
            "random",
            "explicit",
        ]
        assert properties["initialization_strategy"]["default"] == "quantile"
        assert properties["random_seed"]["default"] == 0
        assert properties["initial_means"]["type"] == "array"
        assert schema["additionalProperties"] is False


def test_latent_initialization_is_seeded_explicit_and_fixed_signature() -> None:
    matrix = np.ascontiguousarray(
        np.column_stack(
            [
                np.linspace(-2.5, 2.5, 24),
                np.cos(np.linspace(0.0, 4.0, 24)),
            ]
        ),
        dtype=np.float64,
    )
    empty = np.empty((0, 0), dtype=np.float64)
    before = {
        dispatcher.__name__: tuple(dispatcher.signatures)
        for dispatcher in (
            initialize_gaussian_means_kernel,
            gmm_fit_kernel,
            markov_fit_kernel,
            hmm_fit_kernel,
        )
    }
    first = initialize_gaussian_means_kernel(matrix, 3, 1, 17, empty)
    repeated = initialize_gaussian_means_kernel(matrix, 3, 1, 17, empty)
    another_seed = initialize_gaussian_means_kernel(matrix, 3, 1, 18, empty)
    explicit = np.ascontiguousarray(
        [[-1.5, -0.5], [0.0, 0.25], [1.5, 0.75]],
        dtype=np.float64,
    )
    explicit_result = initialize_gaussian_means_kernel(matrix, 3, 2, 999, explicit)

    assert np.array_equal(first, repeated)
    assert not np.array_equal(first, another_seed)
    assert np.array_equal(explicit_result, explicit)
    gmm_fit_kernel(matrix, 3, 2, first)
    markov_fit_kernel(matrix, 3, 2, first)
    hmm_fit_kernel(matrix, 3, 2, first)
    after = {
        dispatcher.__name__: tuple(dispatcher.signatures)
        for dispatcher in (
            initialize_gaussian_means_kernel,
            gmm_fit_kernel,
            markov_fit_kernel,
            hmm_fit_kernel,
        )
    }
    assert after == before
    assert all(
        dispatcher._can_compile is False
        for dispatcher in (
            initialize_gaussian_means_kernel,
            gmm_fit_kernel,
            markov_fit_kernel,
            hmm_fit_kernel,
        )
    )


@pytest.mark.parametrize("model_type", ["model.hmm", "model.markov", "model.gmm"])
def test_latent_model_random_seed_is_reproducible_and_audited_without_recompile(
    service: RegimeGraphV2Service,
    model_type: str,
) -> None:
    definition = _latent_definition(model_type=model_type)
    parameters = definition["graph"]["nodes"][2]["parameters"]
    parameters.update({"initialization_strategy": "random", "random_seed": 31})
    parsed = parse_definition_v2(definition)
    prepared = service.prepare(definition)
    before = {
        dispatcher.__name__: tuple(dispatcher.signatures)
        for dispatcher in (
            initialize_gaussian_means_kernel,
            gmm_fit_kernel,
            markov_fit_kernel,
            hmm_fit_kernel,
        )
    }
    first = service._execute_graph(None, parsed, "retrospective", None, plan=prepared)
    repeated = service._execute_graph(None, parsed, "retrospective", None, plan=prepared)
    first_audit = first["result"]["diagnostics"]["model_audits"]["latent"]
    repeated_audit = repeated["result"]["diagnostics"]["model_audits"]["latent"]
    changed_seed_definition = copy.deepcopy(definition)
    changed_seed_definition["graph"]["nodes"][2]["parameters"]["random_seed"] = 32
    changed_seed = service._execute_graph(
        None,
        parse_definition_v2(changed_seed_definition),
        "retrospective",
        None,
        plan=service.prepare(changed_seed_definition),
    )
    changed_seed_audit = changed_seed["result"]["diagnostics"]["model_audits"][
        "latent"
    ]

    assert first["series"] == repeated["series"]
    assert first_audit["initialization_strategy"] == "random"
    assert first_audit["random_seed"] == 31
    assert first_audit["initialization_fingerprint"] == repeated_audit[
        "initialization_fingerprint"
    ]
    assert first_audit["initial_means"] == repeated_audit["initial_means"]
    assert first_audit["initialization_fingerprint"] != changed_seed_audit[
        "initialization_fingerprint"
    ]
    assert first_audit["initialization_backend"] == "numba_njit_fixed_signature"
    assert {
        dispatcher.__name__: tuple(dispatcher.signatures)
        for dispatcher in (
            initialize_gaussian_means_kernel,
            gmm_fit_kernel,
            markov_fit_kernel,
            hmm_fit_kernel,
        )
    } == before


def test_latent_explicit_initial_values_and_experiment_grid_are_supported(
    service: RegimeGraphV2Service,
) -> None:
    definition = _latent_definition(model_type="model.gmm")
    parameters = definition["graph"]["nodes"][2]["parameters"]
    initial_means = [[-1.25], [0.0], [1.25]]
    parameters.update(
        {
            "initialization_strategy": "explicit",
            "initial_means": initial_means,
            "random_seed": 77,
        }
    )
    prepared = service.prepare(definition)
    execution = service._execute_graph(
        None,
        parse_definition_v2(definition),
        "retrospective",
        None,
        plan=prepared,
    )
    audit = execution["result"]["diagnostics"]["model_audits"]["latent"]
    assert audit["initialization_strategy"] == "explicit"
    assert audit["initial_means"] == initial_means

    candidates = service._experiment_grid_candidates(
        parse_definition_v2(_latent_definition(model_type="model.gmm")),
        [
            {"node_id": "latent", "parameter": "components", "values": [2, 3]},
            {
                "node_id": "latent",
                "parameter": "initialization_strategy",
                "values": ["random"],
            },
            {"node_id": "latent", "parameter": "random_seed", "values": [7, 8]},
        ],
    )
    assert len(candidates) == 4
    assert {
        difference["parameter"]
        for _, differences in candidates
        for difference in differences
    } == {"components", "initialization_strategy", "random_seed"}

    explicit_candidates = service._experiment_grid_candidates(
        parse_definition_v2(_latent_definition(model_type="model.gmm")),
        [
            {
                "node_id": "latent",
                "parameter": "initialization_strategy",
                "values": ["explicit"],
            },
            {
                "node_id": "latent",
                "parameter": "initial_means",
                "values": [
                    [[-1.5], [0.0], [1.5]],
                    [[-2.0], [0.25], [1.0]],
                ],
            },
        ],
    )
    assert len(explicit_candidates) == 2
    assert {
        difference["parameter"]
        for _, differences in explicit_candidates
        for difference in differences
    } == {"initialization_strategy", "initial_means"}


def test_publish_records_full_gate_and_blocks_formal_retrospective(
    service: RegimeGraphV2Service,
) -> None:
    definition = _definition()
    upload_frame = pd.DataFrame(_rows())
    upload_frame["vintage"] = None
    upload_frame["revision"] = 1
    artifact = write_upload_artifact(service.market_data_dir, upload_frame)
    definition["graph"]["nodes"][0] = {
        "id": "source",
        "type": "source.upload",
        "parameters": {
            "artifact_id": artifact["artifact_id"],
            "checksum": artifact["checksum"],
            "format": "parquet",
            "availability_mode": "point_in_time",
        },
    }
    saved = service.create_definition(definition)
    prepared = service.prepare(saved)
    reference = {"schema_version": "2.0", "id": saved["id"], "revision": 1}
    realtime = service.run_saved(
        reference,
        "realtime",
        compile_token=prepared["compile_token"],
    )
    assert realtime["governance"]["formal_gate_passed"] is True
    assert set(realtime["governance"]["formal_required_checks"]) == {
        "pit",
        "causality",
        "no_repaint",
        "sample_out",
        "stability",
        "data_coverage",
        "njit_call_graph",
        "temporal_audit",
    }
    publication = service.publish(realtime["id"], "formal_backtest")
    assert publication["publication"]["gate"] == "comprehensive_formal_gate_passed"

    retrospective = service.run_saved(
        reference,
        "retrospective",
        compile_token=prepared["compile_token"],
    )
    assert "pit" in retrospective["governance"]["formal_failures"]
    with pytest.raises(Exception) as exc_info:
        service.publish(retrospective["id"], "taa")
    assert getattr(exc_info.value, "code", None) == "REGIME_PUBLICATION_GATE_FAILED"
    research = service.publish(retrospective["id"], "product_research")
    assert research["publication"]["research_restrictions"]


def test_formal_run_persists_every_evaluation_target_and_conditional_result(
    service: RegimeGraphV2Service,
) -> None:
    definition = _definition()
    definition["graph"]["nodes"][0] = _upload_source(service, _rows())

    equity_rows = _rows()
    bond_rows = _rows()
    for index, row in enumerate(equity_rows):
        row["value"] = 1000.0 + index * 2.0
    for index, row in enumerate(bond_rows):
        row["value"] = 500.0 + index * 0.25
    equity_source = _upload_source(service, equity_rows)["parameters"]
    bond_source = _upload_source(service, bond_rows)["parameters"]
    definition["evaluation_targets"] = [
        {
            "id": "equity",
            "name": "权益评价",
            "primary": True,
            "source": {"kind": "upload", **equity_source},
        },
        {
            "id": "bond",
            "name": "债券评价",
            "primary": False,
            "source": {"kind": "upload", **bond_source},
        },
    ]
    saved = service.create_definition(definition)
    prepared = service.prepare(saved)
    run = service.run_saved(
        {"schema_version": "2.0", "id": saved["id"], "revision": 1},
        "realtime",
        compile_token=prepared["compile_token"],
    )
    assert set(run["evaluation_results"]) == {"equity", "bond"}
    assert run["evaluation_results"]["equity"]["primary"] is True
    assert run["evaluation_results"]["bond"]["primary"] is False
    assert run["conditional_metrics"] == run["evaluation_results"]["equity"]["conditional_metrics"]
    for target in run["evaluation_results"].values():
        assert target["artifact"]["shape"] == [len(_rows())]
        assert target["conditional_metrics"]
    evidence = next(
        item for item in run["evidence"] if item["kind"] == "evaluation_output_artifact"
    )
    assert {item["node_id"] for item in evidence["arrays"]} == {"equity", "bond"}


def test_walk_forward_latent_model_refits_at_each_fold_boundary(
    service: RegimeGraphV2Service,
) -> None:
    definition = _latent_definition(_upload_source(service, _rows(144)))
    saved = service.create_definition(definition)
    prepared = service.prepare(saved)
    run = service.run_saved(
        {"schema_version": "2.0", "id": saved["id"], "revision": 1},
        "realtime",
        compile_token=prepared["compile_token"],
    )
    successful = [
        item
        for item in run["walk_forward"]["folds"]
        if item["status"] == "ok"
    ]
    assert len(successful) >= 2
    training_counts: list[int] = []
    cutoffs: list[str] = []
    for fold in successful:
        audit = fold["model_audits"]["latent"]
        assert fold["model_refit"] == "expanding_window_at_fold_boundary"
        assert audit["walk_forward_refit"] is True
        assert audit["training_cutoff_available_at"] == fold["model_training_available_through"]
        training_counts.append(int(audit["training_count"]))
        cutoffs.append(str(audit["training_cutoff_available_at"]))
    assert training_counts == sorted(training_counts)
    assert len(set(training_counts)) == len(training_counts)
    assert cutoffs == sorted(cutoffs)


def test_builtin_template_instantiation_records_immutable_version_lineage(
    service: RegimeGraphV2Service,
) -> None:
    catalog = service.templates()["items"]
    template = next(item for item in catalog if item["id"] == "blank-three-state")
    assert template["version"] == 1
    assert len(template["content_hash"]) == 64
    instantiated = service.instantiate_template("blank-three-state")
    assert instantiated["template_version"] == template["version"]
    assert instantiated["template_content_hash"] == template["content_hash"]
    assert instantiated["definition"]["template_id"] == "blank-three-state@1"
    assert instantiated["source"] == {
        "template_id": "blank-three-state",
        "template_version": 1,
        "content_hash": template["content_hash"],
    }


def test_v1_definition_write_routes_are_explicitly_read_only(client: TestClient) -> None:
    response = client.post(
        "/api/historical-regimes/definitions",
        json={
            "name": "旧定义",
            "algorithm": {"family": "causal_filter"},
        },
    )
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "V1_DEFINITION_READ_ONLY"

    paths = set(client.app.openapi()["paths"])
    assert "/api/historical-regimes/v2/graph-assets" in paths
    assert "/api/historical-regimes/v2/experiments" in paths


def test_v1_run_and_publication_write_routes_are_explicitly_read_only(
    client: TestClient,
) -> None:
    run_response = client.post(
        "/api/historical-regimes/run",
        json={
            "definition": {"id": "legacy-definition", "revision": 1},
            "mode": "realtime",
        },
    )
    assert run_response.status_code == 409
    assert run_response.json()["detail"]["code"] == "V1_RUN_READ_ONLY"

    legacy_service = historical_regime_routes.historical_regime_service
    legacy_definition = {
        "name": "既有 v1 定义",
        "target": {
            "kind": "inline",
            "name": "历史样本",
            "series_id": "legacy-series",
            "frequency": "daily",
            "rows": _rows(),
        },
        "features": {
            "transform": "none",
            "filter": "ema",
            "window": 6,
            "slope_window": 2,
            "volatility_window": 4,
        },
        "algorithm": {
            "family": "causal_filter",
            "parameters": {
                "bull_enter": 0.15,
                "bull_exit": 0.03,
                "bear_enter": -0.15,
                "bear_exit": -0.03,
                "confirmation": 2,
                "min_duration": 3,
            },
        },
        "states": _definition()["states"],
        "validation": {"walk_forward": False, "folds": 2},
        "usage_intent": "research_display",
    }
    saved = legacy_service.create_definition(legacy_definition)
    legacy_run = legacy_service.run(saved, "realtime")
    assert client.get(f"/api/historical-regimes/runs/{legacy_run['id']}").status_code == 200
    publication_response = client.post(
        f"/api/historical-regimes/runs/{legacy_run['id']}/publish",
        json={"usage": "research_display"},
    )
    assert publication_response.status_code == 409
    assert publication_response.json()["detail"]["code"] == "V1_RUN_PUBLICATION_READ_ONLY"

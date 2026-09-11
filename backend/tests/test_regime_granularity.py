"""Composite expansion, nominal types, fixed-signature execution and parity."""
import copy
import operator

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from custom_indicators.errors import ValidationError
from historical_regimes.condition_numba import (
    COMPARISON_OPCODES, condition_compare_kernel, condition_valid_kernel,
    condition_logic_kernel, select_state_kernel,
)
from historical_regimes.composite_expansion import expand_composite
from historical_regimes.v2_contracts import inspect_definition_v2, parse_definition_v2
from historical_regimes.v2_numba import KERNELS, regime_graph_numba_status
from historical_regimes.v2_registry import NODE_REGISTRY, node_catalog
from historical_regimes.v2_service import RegimeGraphV2Service, _kernel_ids_for_definition
from historical_regimes.v2_templates import MARKET_STATES, CLOCK_STATES


def source(identifier, values):
    return {"id": identifier, "type": "source.inline", "parameters": {"frequency": "daily", "rows": [
        {"observation_date": date.date().isoformat(), "available_at": date.date().isoformat(),
         "value": float(value) if np.isfinite(value) else None}
        for date, value in zip(pd.date_range("2020-01-01", periods=len(values)), values)]}}


def definition(kind="model.threshold", *, seed=0, sideways=False, two_states=False):
    rng = np.random.default_rng(seed)
    values = rng.normal(0, .03, 400)
    values[:6] = [.001, -.001, .0, .002, -.002, np.nan]
    if kind == "model.peak_trough":
        values = 100 * np.exp(np.cumsum(rng.normal(0, .012, 400)))
        values[100:103] = np.nan
        values[200:203] = values[200]
    nodes = [source("data", values)]
    inputs = {"value": {"node_id": "data", "port": "value"}}
    states = copy.deepcopy(MARKET_STATES)
    parameters = {}
    if kind == "model.quadrant":
        other = values[::-1].copy()
        other[15] = np.nan
        nodes.append(source("inflation", other))
        nodes.append({"id": "aligned", "type": "align.strict_intersection", "inputs": {
            "left": inputs["value"], "right": {"node_id": "inflation", "port": "value"}}})
        inputs = {"growth": {"node_id": "aligned", "port": "left"}, "inflation": {"node_id": "aligned", "port": "right"}}
        states = copy.deepcopy(CLOCK_STATES)
    elif kind == "model.peak_trough":
        parameters = {"left_window": 2, "right_window": 3, "head_window": 1, "tail_window": 0,
                      "min_phase": 2, "min_cycle": 6, "amplitude_exception": .1,
                      "sideways_enabled": sideways, "sideways_min_duration": 3}
        if two_states:
            states = [states[0], states[2]]
            states[1]["order"] = 1
    nodes.append({"id": "algorithm", "type": kind, "label": "研究方案节点", "parameters": parameters, "inputs": inputs})
    return {"schema_version": "2.0", "name": "颗粒度回归", "description": "不可变定义测试",
            "graph": {"nodes": nodes, "outputs": {"state": {"node_id": "algorithm", "port": "state"}},
                      "exposed_node_ids": ["algorithm"]}, "states": states, "evaluation_targets": [], "validation": {}}


@pytest.mark.parametrize("operation", list(COMPARISON_OPCODES))
@pytest.mark.parametrize("vector_bound", [True, False])
def test_conditions_reuse_comparisons_preserving_missing_and_equality(operation, vector_bound):
    values = np.array([-1., 0., 1., np.nan, np.inf, -np.inf])
    bounds = np.zeros(values.size) if vector_bound else np.empty(0)
    compare = getattr(operator, operation)
    expected = [int(compare(value, 0)) if np.isfinite(value) else -1 for value in values]
    actual = condition_compare_kernel(values, bounds, 0., COMPARISON_OPCODES[operation])
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.int64
    assert condition_compare_kernel(np.empty(0), np.empty(0), 0., COMPARISON_OPCODES[operation]).size == 0


def test_conditions_validate_axes_and_fixed_dtype():
    with pytest.raises(ValueError):
        condition_compare_kernel(np.ones(5), np.ones(2), 0., 45)
    with pytest.raises(ValueError):
        condition_compare_kernel(np.ones(5), np.empty(0), 0., 50)
    with pytest.raises(TypeError):
        condition_compare_kernel(np.ones(5, dtype=np.float32), np.empty(0), 0., 45)
    np.testing.assert_array_equal(condition_valid_kernel(np.array([0., np.nan, np.inf, -np.inf, 1.])), [1, 0, 0, 0, 1])


@pytest.mark.parametrize("opcode", [0, 1, 2])
def test_condition_logic_truth_table_and_unknown(opcode):
    left = np.repeat(np.array([-1, 0, 1], dtype=np.int64), 3)
    right = np.tile(np.array([-1, 0, 1], dtype=np.int64), 3)
    expected = [(-1 if a < 0 else 1-a) if opcode == 2 else
                (-1 if a < 0 or b < 0 else int((a and b) if opcode == 0 else (a or b)))
                for a, b in zip(left, right)]
    np.testing.assert_array_equal(condition_logic_kernel(left, right, opcode), expected)


def test_state_select_does_not_coerce_missing_or_use_unselected_branch():
    condition = np.array([1, 0, -1, 1], dtype=np.int64)
    yes = np.array([0, -1, 2, 2], dtype=np.int64)
    no = np.array([-1, 1, 1, -1], dtype=np.int64)
    np.testing.assert_array_equal(select_state_kernel(condition, yes, no, 0, 1), [0, 1, -1, 2])
    np.testing.assert_array_equal(select_state_kernel(condition, np.empty(0, dtype=np.int64), no, 0, 1), [0, 1, -1, 0])


@pytest.mark.parametrize("kind,sideways,two_states", [
    ("model.threshold", False, False), ("model.quadrant", False, False),
    ("model.peak_trough", False, False), ("model.peak_trough", True, False),
    ("model.peak_trough", False, True),
])
@pytest.mark.parametrize("seed", [0, 3, 19])
def test_expanded_graph_matches_every_original_port_and_final_timing(tmp_path, kind, sideways, two_states, seed):
    raw = definition(kind, seed=seed, sideways=sideways, two_states=two_states)
    unchanged = copy.deepcopy(raw)
    expanded = expand_composite(raw, "algorithm", "retrospective")
    assert raw == unchanged
    assert all(node["type"] != kind for node in expanded["definition"]["graph"]["nodes"])
    service = RegimeGraphV2Service(tmp_path, tmp_path)
    signatures = {name: list(kernel.signatures) for name, kernel in KERNELS.items()}
    original = service._execute_graph(None, parse_definition_v2(raw), "retrospective", None)
    parsed = parse_definition_v2(expanded["definition"])
    after = service._execute_graph(None, parsed, "retrospective", None)
    for port, reference in expanded["output_map"].items():
        old = original["node_outputs"]["algorithm"][port]
        new = after["node_outputs"][reference["node_id"]][reference["port"]]
        np.testing.assert_array_equal(old.values, new.values, err_msg=port)
        np.testing.assert_array_equal(old.dates, new.dates, err_msg=port)
        np.testing.assert_array_equal(old.available, new.available, err_msg=port)
    for key in ("state_code", "recognition_index", "effective_index", "recognized_at", "effective_date", "executable", "probabilities", "confidence"):
        assert [row[key] for row in original["series"]] == [row[key] for row in after["series"]], key
    if kind == "model.peak_trough":
        for key in ("pivot", "phase_start_index", "phase_end_index", "phase_return", "boundary_line"):
            assert [row["features"].get(key) for row in original["series"]] == [row["features"].get(key) for row in after["series"]], key
        assert not any(row["executable"] for row in after["series"])
    needed = set(_kernel_ids_for_definition(parsed))
    for node in parsed.graph.nodes:
        assert set(NODE_REGISTRY[node.type].get("kernel_dependencies", [])) <= needed
    assert signatures == {name: list(kernel.signatures) for name, kernel in KERNELS.items()}
    audit = regime_graph_numba_status()
    assert audit["complete"] and audit["python_fallback"] == audit["request_time_compilation"] == 0


def test_expansion_reconnects_all_roots_downstream_and_edges_only_input():
    raw = definition()
    raw.update(id="saved", revision=3)
    graph = raw["graph"]
    graph["nodes"].append({"id": "downstream", "type": "post.confirmation", "parameters": {},
                           "inputs": {"state": {"node_id": "algorithm", "port": "state"}}})
    graph["outputs"].update({"state": {"node_id": "downstream", "port": "state"},
                             "confidence": {"node_id": "algorithm", "port": "confidence"},
                             "score": {"node_id": "algorithm", "port": "score"}})
    graph["channel_metadata"] = {"score": {"label": "原始得分", "unit": "", "display_format": "number", "precision": 4}}
    graph["edges"] = [{"source": ref, "target": {"node_id": node["id"], "port": name}}
                      for node in graph["nodes"] for name, ref in node.get("inputs", {}).items()]
    for node in graph["nodes"]:
        node["inputs"] = {}
    result = expand_composite(raw, "algorithm")
    expanded = result["definition"]
    assert expanded["id"] == "saved" and expanded["revision"] == 3
    assert expanded["states"] == raw["states"] and expanded["graph"]["channel_metadata"] == graph["channel_metadata"]
    parsed = parse_definition_v2(expanded)
    assert inspect_definition_v2(parsed)["valid"]
    assert all(ref.node_id != "algorithm" for node in parsed.graph.nodes for ref in node.inputs.values())
    assert expanded["graph"]["outputs"]["confidence"] == result["output_map"]["confidence"]
    assert all(identifier != "algorithm" for identifier in expanded["graph"]["exposed_node_ids"])


@pytest.mark.parametrize("kind", ["model.threshold", "model.quadrant", "model.peak_trough"])
def test_unconnected_composite_drafts_can_expand_without_fabricating_inputs(kind):
    raw = definition(kind)
    raw["graph"]["nodes"] = [raw["graph"]["nodes"][-1]]
    raw["graph"]["nodes"][0]["inputs"] = {}
    raw["graph"]["outputs"] = {}
    result = expand_composite(raw, "algorithm", "retrospective")
    assert result["definition"]["graph"]["outputs"] == {}
    assert not any(node["type"].startswith("source.") for node in result["definition"]["graph"]["nodes"])


def test_long_ids_collisions_and_node_budget_are_bounded():
    raw = definition()
    identifier = "a" * 64
    raw["graph"]["nodes"][-1]["id"] = identifier
    raw["graph"]["outputs"]["state"]["node_id"] = identifier
    raw["graph"]["exposed_node_ids"] = [identifier]
    first = expand_composite(raw, identifier)
    collision = copy.deepcopy(first["definition"]["graph"]["nodes"][1])
    collision["type"] = "transform.identity"; collision["parameters"] = {}; collision["inputs"] = {}
    raw["graph"]["nodes"].append(collision)
    second = expand_composite(raw, identifier)
    assert len(set(second["inserted_node_ids"])) == len(second["inserted_node_ids"])
    assert collision["id"] not in second["inserted_node_ids"]
    assert all(len(value) <= 64 for value in second["inserted_node_ids"])
    raw["graph"]["nodes"].extend({"id": f"unused{i}", "type": "transform.identity", "inputs": {}} for i in range(124))
    with pytest.raises(ValidationError):
        expand_composite(raw, identifier)


@pytest.mark.parametrize("mutation", ["version", "unknown_input", "missing_node", "type", "bounds"])
def test_invalid_definitions_do_not_silently_expand(mutation):
    raw = definition()
    node = raw["graph"]["nodes"][-1]
    if mutation == "version": node["type_version"] = 99
    elif mutation == "unknown_input": node["inputs"]["nonsense"] = node["inputs"]["value"]
    elif mutation == "missing_node": node["inputs"]["value"]["node_id"] = "absent"
    elif mutation == "type": node["type"] = "filter.ema"
    else: node["parameters"].update(upper=-1., lower=1.)
    with pytest.raises(ValidationError):
        expand_composite(raw, "algorithm")


def test_nominal_conditions_cannot_connect_as_numeric_or_market_states():
    raw = expand_composite(definition(), "algorithm")["definition"]
    compare = next(node for node in raw["graph"]["nodes"] if node["type"] == "condition.compare")
    raw["graph"]["outputs"]["state"] = {"node_id": compare["id"], "port": "condition"}
    assert any(error["code"] == "GRAPH_OUTPUT_TYPE_MISMATCH" for error in inspect_definition_v2(parse_definition_v2(raw))["errors"])
    compare["inputs"]["value"] = {"node_id": compare["id"], "port": "condition"}
    raw["graph"]["edges"] = []
    assert any(error["code"] == "PORT_TYPE_MISMATCH" for error in inspect_definition_v2(parse_definition_v2(raw))["errors"])


def test_unused_optional_outputs_are_not_scheduled():
    raw = definition()
    raw["graph"]["exposed_node_ids"] = []
    expanded = expand_composite(raw, "algorithm")["definition"]
    required = set(_kernel_ids_for_definition(parse_definition_v2(expanded)))
    assert {"condition_compare", "select_state"} <= required
    # Final result framing needs membership, but the redundant score identity is not run.
    assert "unary_transform" not in required


def test_ps_realtime_gate_and_full_input_knowledge_survive_expansion(tmp_path, monkeypatch):
    raw = definition("model.peak_trough", sideways=True)
    with pytest.raises(ValidationError, match="事后"):
        expand_composite(raw, "algorithm", "realtime")
    raw["graph"]["nodes"][0]["parameters"]["rows"][1]["available_at"] = "2023-01-01"
    expanded = expand_composite(raw, "algorithm", "retrospective")["definition"]
    service = RegimeGraphV2Service(tmp_path, tmp_path)
    result = service._execute_graph(None, parse_definition_v2(expanded), "retrospective", None)
    classified = [row for row in result["series"] if row["state_code"] >= 0]
    assert classified and all(row["recognized_at"] == "2023-01-01" and not row["executable"] for row in classified)
    monkeypatch.setattr(service, "_resolve_sources", lambda *args, **kwargs: pytest.fail("gate must precede I/O"))
    with pytest.raises(ValidationError, match="实时"):
        service._execute_graph(None, parse_definition_v2(expanded), "realtime", None)


def test_actual_runtime_calls_fixed_signature_domain_kernel(tmp_path, monkeypatch):
    from historical_regimes import granular_runtime
    calls = []
    original = granular_runtime.select_state_kernel
    def observed(*args):
        calls.append(args[0].dtype)
        return original(*args)
    monkeypatch.setattr(granular_runtime, "select_state_kernel", observed)
    raw = expand_composite(definition(), "algorithm")["definition"]
    service = RegimeGraphV2Service(tmp_path, tmp_path)
    service.prepare(raw)
    service._execute_graph(None, parse_definition_v2(raw), "realtime", None)
    assert calls == [np.dtype("int64"), np.dtype("int64")]
    assert original.nopython_signatures and original._can_compile is False


def test_expansion_api_is_pure_and_catalog_describes_granularity():
    from services import historical_regime_routes
    app = FastAPI()
    app.include_router(historical_regime_routes.router)
    with TestClient(app) as client:
        raw = definition()
        response = client.post("/api/historical-regimes/authoring/expand", json={"definition": raw, "node_id": "algorithm", "mode": "realtime"})
        assert response.status_code == 200, response.text
        assert response.json()["contract_version"] == 1
        bad = client.post("/api/historical-regimes/authoring/expand", json={"definition": raw, "node_id": "data"})
        assert bad.status_code == 422
    catalog = {item["id"]: item for item in node_catalog()["items"]}
    assert catalog["model.peak_trough"]["granularity"]["expandable"]
    assert catalog["pivot.ps_filter"]["granularity"]["kind"] == "coupled"
    assert catalog["condition.compare"]["granularity"]["kind"] == "primitive"
    assert catalog["feature.trend_metrics"]["granularity"]["expandable"] is False
    assert "非预测概率" in catalog["state.encode"]["outputs"][0]["label"]

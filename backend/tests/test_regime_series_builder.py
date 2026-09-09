"""Shared registry execution, enum output boundaries and authoring invariants."""
import copy

import numpy as np
import pandas as pd
import pytest

from computation_graph.series_numba import causal_available_kernel, valid_series_output_kernel
from computation_graph.series_operators import series_operator_specs
from historical_regimes.authoring import AuthoringRequest, resolve_authoring
from historical_regimes.v2_contracts import inspect_definition_v2, parse_definition_v2
from historical_regimes.v2_registry import NODE_REGISTRY
from historical_regimes.v2_service import RegimeGraphV2Service
from historical_regimes.v2_templates import instantiate_template_v2


def graph(operator="rolling_mean", parameters=None, values=None):
    data = instantiate_template_v2("peak-trough-daily-v2")
    values = [10., 12., 11., 14., 13., 12., 16., 15.] if values is None else values
    rows = [{"observation_date": date.date().isoformat(), "available_at": date.date().isoformat(), "value": value}
            for date, value in zip(pd.date_range("2020-01-01", periods=len(values)), values)]
    data["graph"] = {"nodes": [
        {"id": "price", "type": "source.inline", "parameters": {"rows": rows, "frequency": "daily"}},
        {"id": "smooth", "type": f"indicator.{operator}", "parameters": parameters or {},
         "inputs": {port["name"]: {"node_id": "price", "port": "value"} for port in NODE_REGISTRY[f"indicator.{operator}"]["inputs"]}},
        {"id": "classifier", "type": "model.range_threshold", "parameters": {"upper": 12., "lower": 11.},
         "inputs": {"value": {"node_id": "smooth", "port": "value"}}}],
        "outputs": {"state": {"node_id": "classifier", "port": "state"}, "trend": {"node_id": "smooth", "port": "value"}},
        "channel_metadata": {"trend": {"label": "趋势线", "unit": "点", "precision": 2}}, "exposed_node_ids": []}
    return data


@pytest.fixture(scope="module")
def service(tmp_path_factory):
    from historical_regimes.numba_kernels import warm_historical_regime_numba_kernels
    warm_historical_regime_numba_kernels()
    root = tmp_path_factory.mktemp("shared-series")
    return RegimeGraphV2Service(root, root)


def execute(service, raw):
    plan = service.prepare(raw)
    return service._execute_graph(None, parse_definition_v2(raw), "realtime", None, plan=plan)


def test_operator_nodes_derive_actual_indicator_signatures():
    for spec in series_operator_specs():
        node = NODE_REGISTRY[f"indicator.{spec.operator_id}"]
        assert node["typed_operator_version"] == spec.version
        assert node["typed_arguments"] == list(spec.argument_names(max(spec.arities)))
        assert node["kernel_id"] == "typed_formula_plan"
        assert node["causal"] and node["supports_realtime"] and not node["repaints"]


@pytest.mark.parametrize("operator,params,reference", [
    ("rolling_mean", {"window": 3, "min_periods": 3}, lambda x: pd.Series(x).rolling(3).mean().to_numpy()),
    ("rolling_std", {"window": 3, "min_periods": 3, "ddof": 1}, lambda x: pd.Series(x).rolling(3).std().to_numpy()),
    ("rolling_min", {"window": 3, "min_periods": 3}, lambda x: pd.Series(x).rolling(3).min().to_numpy()),
    ("rolling_max", {"window": 3, "min_periods": 3}, lambda x: pd.Series(x).rolling(3).max().to_numpy()),
    ("add", {}, lambda x: x + x),
    ("subtract", {}, lambda x: x - x),
    ("clip", {"lower": 11., "upper": 13.}, lambda x: np.clip(x, 11., 13.)),
    ("cumulative_sum", {}, np.cumsum),
])
def test_actual_graph_uses_shared_numeric_plan_and_preserves_prefix(service, operator, params, reference):
    raw = graph(operator, params)
    result = execute(service, raw)
    values = result["node_outputs"]["smooth"]["value"].values
    np.testing.assert_allclose(values, reference(np.array([10., 12., 11., 14., 13., 12., 16., 15.])), equal_nan=True)
    audit = result["result"]["diagnostics"]
    assert audit["formula_audits"]["smooth"]["python_fallback"] == 0
    assert audit["request_time_compilation"] == 0
    extended = copy.deepcopy(raw)
    extended["graph"]["nodes"][0]["parameters"]["rows"].append({"observation_date": "2020-01-09", "available_at": "2020-01-09", "value": 9999.})
    future = execute(service, extended)
    np.testing.assert_array_equal(values, future["node_outputs"]["smooth"]["value"].values[:-1])
    for row, value in zip(result["series"], values):
        assert row["features"]["channel:trend"] == (None if np.isnan(value) else value)
    contracts = result["result"]["series_outputs"]
    assert contracts[0]["value_type"] == "enum" and contracts[0]["missing"] == "unclassified"
    assert contracts[1]["label"] == "趋势线" and contracts[1]["precision"] == 2


def test_formula_arithmetic_reuses_typed_compiler_and_named_channels(service):
    raw = graph()
    source = resolve_authoring(AuthoringRequest(definition=raw, source_kind="graph"))["source"]
    source = "\n".join("smooth = rolling_mean(price.value, 3, 3) + 1" if line.startswith("smooth =") else line for line in source.splitlines())
    resolved = resolve_authoring(AuthoringRequest(definition=raw, source_kind="formula", source=source, mode="realtime"))
    assert resolved["valid"], resolved["diagnostics"]
    result = execute(service, resolved["definition"])
    expected = pd.Series([10., 12., 11., 14., 13., 12., 16., 15.]).rolling(3).mean() + 1
    np.testing.assert_allclose(result["node_outputs"]["smooth"]["value"].values, expected, equal_nan=True)
    assert resolved["compile_status"] == "not_requested"
    bad = source.replace("rolling_mean(price.value, 3, 3) + 1", "mean(price.value)")
    assert not resolve_authoring(AuthoringRequest(definition=raw, source_kind="formula", source=bad))["valid"]


def test_enum_cannot_enter_numeric_operators_or_numeric_channel():
    raw = graph()
    raw["graph"]["outputs"]["trend"] = {"node_id": "classifier", "port": "state"}
    assert not inspect_definition_v2(parse_definition_v2(raw))["valid"]
    raw = graph()
    raw["graph"]["nodes"].append({"id": "invalid", "type": "indicator.rolling_mean", "parameters": {},
                                  "inputs": {"values": {"node_id": "classifier", "port": "state"}}})
    assert not inspect_definition_v2(parse_definition_v2(raw))["valid"]


def test_availability_prefix_fixed_signature_and_empty():
    before = list(causal_available_kernel.signatures)
    for data in [[], [1], [1, 7, 3, 4], [-2, -3, 0]]:
        values = np.asarray(data, dtype=np.int64)
        np.testing.assert_array_equal(causal_available_kernel(values), np.maximum.accumulate(values))
    assert list(causal_available_kernel.signatures) == before
    assert causal_available_kernel.nopython_signatures and not causal_available_kernel._can_compile
    with pytest.raises(TypeError):
        causal_available_kernel(np.ones(2, dtype=np.float64))


def test_legacy_output_contract_and_new_channels_roundtrip():
    raw = instantiate_template_v2("peak-trough-daily-v2")
    assert "channel_metadata" not in parse_definition_v2(raw).model_dump()["graph"]
    raw = graph()
    first = resolve_authoring(AuthoringRequest(definition=raw, source_kind="graph"))
    again = resolve_authoring(AuthoringRequest(definition=raw, source_kind="formula", source=first["source"]))
    assert again["valid"], again["diagnostics"]
    assert first["definition"] == again["definition"]


def test_compact_formula_preserves_labels_versions_and_custom_outputs():
    raw = graph()
    raw["graph"]["nodes"][1]["label"] = "短期滤波线"
    first = resolve_authoring(AuthoringRequest(definition=raw, source_kind="graph", compact=True))
    assert "_label=" not in first["source"] and "_version=" not in first["source"]
    again = resolve_authoring(AuthoringRequest(definition=raw, source_kind="formula", source=first["source"], compact=True))
    assert first["definition"] == again["definition"]


def test_late_publication_is_never_labeled_as_known_earlier(service):
    raw = graph(parameters={"window": 3, "min_periods": 3})
    raw["graph"]["nodes"][0]["parameters"]["rows"][1]["available_at"] = "2020-01-07"
    result = execute(service, raw)
    port = result["node_outputs"]["smooth"]["value"]
    assert port.available[2] == pd.Timestamp("2020-01-07").value
    assert result["series"][2]["recognized_at"] >= "2020-01-07"


def test_missing_blocks_and_series_nan_inf_contract(service):
    raw = graph(parameters={"window": 3, "min_periods": 3}, values=[10., 11., None, 12., 13., 14., 15., 16.])
    result = execute(service, raw)
    expected = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, 13., 14., 15.])
    np.testing.assert_allclose(result["node_outputs"]["smooth"]["value"].values, expected, equal_nan=True)
    assert all(row["state_id"] == "unclassified" for row in result["series"][:5])
    signatures = list(valid_series_output_kernel.signatures)
    assert valid_series_output_kernel(np.array([np.nan, 1.])) == 1
    assert valid_series_output_kernel(np.array([np.inf])) == 0
    assert valid_series_output_kernel(np.array([], dtype=np.float64)) == 1
    with pytest.raises(TypeError):
        valid_series_output_kernel(np.ones(2, dtype=np.float32))
    assert list(valid_series_output_kernel.signatures) == signatures

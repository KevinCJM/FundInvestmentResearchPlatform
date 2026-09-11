import copy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from historical_regimes.indicator_nodes import register_indicator_nodes
from historical_regimes.v2_registry import NODE_REGISTRY, node_catalog
from historical_regimes.v2_service import RegimeGraphV2Service
from historical_regimes.v2_contracts import parse_definition_v2, inspect_definition_v2
from historical_regimes.authoring import AuthoringRequest, resolve_authoring
from test_regime_series_builder import graph, execute


def definition(id="test-bands", expression="rolling_mean(market_close, 3)", **extra):
    return dict(id=id, revision=1, name="测试指标", context_kind="single_product", result_kind="time_series",
                dsl_version="2.4.0", operator_registry_version="2.4.0", parameter_schema=[], series_outputs=[{"id": "middle", "label": "中轨", "expression": expression},
                {"id": "upper", "label": "上轨", "expression": f"({expression}) + 2"}], **extra)


def register(definitions):
    service = SimpleNamespace(indicators=SimpleNamespace(list_all_versions=lambda: definitions),
                              get_indicator=lambda id, revision: next(d for d in definitions if d['id'] == id and d['revision'] == revision))
    register_indicator_nodes(service, NODE_REGISTRY)
    return next(item for item in reversed(list(NODE_REGISTRY.values())) if item.get("indicator_reference", {}).get("id") == definitions[-1]["id"])


@pytest.fixture(scope="module")
def service(tmp_path_factory):
    from historical_regimes.numba_kernels import warm_historical_regime_numba_kernels
    warm_historical_regime_numba_kernels()
    root = tmp_path_factory.mktemp("regime-indicator-nodes")
    return RegimeGraphV2Service(root, root)


def configured(metadata, parameters=None, values=None):
    raw = graph(values=values)
    raw["graph"]["nodes"][1].update(type=metadata["id"], parameters=parameters or {}, inputs={
        p["name"]: {"node_id": "price", "port": "value"} for p in metadata["inputs"]})
    port = metadata["outputs"][0]["name"]
    raw["graph"]["nodes"][2]["inputs"]["value"]["port"] = port
    raw["graph"]["outputs"]["trend"]["port"] = port
    return raw


def test_library_hides_removed_creators_but_keeps_existing_contracts():
    catalog = {n["id"]: n for n in node_catalog()["items"]}
    assert catalog["source.inline"]["authoring_hidden"]
    assert catalog["source.indicator"]["authoring_hidden"]
    assert "source.inline" in NODE_REGISTRY


def test_indicator_multi_output_uses_upstream_njit_and_roundtrips(service, monkeypatch):
    meta = register([definition()])
    assert meta["available"], meta["unavailable_reason"]
    assert meta["category"] == "indicator_calculation"
    assert [p["name"] for p in meta["inputs"]] == ["market_close"]
    raw = configured(meta)
    prepared = service.prepare(raw)
    assert "smooth:middle" in prepared["formula_plans"] and "smooth:upper" in prepared["formula_plans"]
    monkeypatch.setattr(service, "_prepare_formula_nodes", lambda *_: pytest.fail("execution must not compile"))
    actual = service._execute_graph(None, parse_definition_v2(raw), "realtime", None, plan=prepared)
    expected = pd.Series([10.,12.,11.,14.,13.,12.,16.,15.]).rolling(3).mean().to_numpy()
    for port, offset in [("middle", 0), ("upper", 2)]:
        np.testing.assert_allclose(actual["node_outputs"]["smooth"][port].values, expected + offset, equal_nan=True)
        assert actual["result"]["diagnostics"]["formula_audits"][f"smooth:{port}"]["python_fallback"] == 0
    first = resolve_authoring(AuthoringRequest(definition=raw, source_kind="graph"))
    assert first["valid"], first
    again = resolve_authoring(AuthoringRequest(definition=raw, source_kind="formula", source=first["source"]))
    assert again["valid"] and again["definition"] == first["definition"]
    assert "\\" in first["display_latex"]["trend"]


def test_runtime_parameter_validation_and_prefix_missing(service):
    d = definition("test-parameter", "rolling_mean(market_close, n)")
    d["parameter_schema"] = [{"id":"n", "label":"窗口", "type":"integer", "default":3, "minimum":2, "maximum":10, "step":1}]
    meta = register([d])
    assert meta["available"], meta["unavailable_reason"]
    values = [1., 2., 3., 4., np.nan, 6., 7., 8.]
    raw = configured(meta, {"n": 2}, values)
    out = execute(service, raw)["node_outputs"]["smooth"]["middle"].values
    np.testing.assert_allclose(out, pd.Series(values).rolling(2).mean(), equal_nan=True)
    extra = copy.deepcopy(raw)
    extra["graph"]["nodes"][0]["parameters"]["rows"].append({"observation_date":"2020-01-09","available_at":"2020-01-09","value":999.})
    np.testing.assert_equal(out, execute(service, extra)["node_outputs"]["smooth"]["middle"].values[:-1])
    raw["graph"]["nodes"][1]["parameters"] = {"n": 2.5}
    assert not inspect_definition_v2(parse_definition_v2(raw))["valid"]


def test_scalar_rolling_reuses_indicator_transformer(service):
    d = definition("test-mean")
    d.update(result_kind="scalar", expression="mean(market_close)")
    meta = register([d])
    assert meta["available"], meta["unavailable_reason"]
    raw = configured(meta, {"window": 2})
    np.testing.assert_allclose(execute(service, raw)["node_outputs"]["smooth"]["value"].values,
                               pd.Series([10.,12.,11.,14.,13.,12.,16.,15.]).rolling(2).mean(), equal_nan=True)


@pytest.mark.parametrize("expression", ["mean(market_close) + market_close", "lag(market_close, -1)", "market_close + observation_count"])
def test_noncausal_or_context_dependent_series_fail_closed(expression):
    meta = register([definition("test-invalid-" + expression, expression)])
    assert not meta["available"] and meta["unavailable_reason"]


def test_revisions_never_redirect_existing_nodes():
    first = register([definition("test-locked")])
    d = definition("test-locked", "rolling_mean(market_close, 4)"); d["revision"] = 2
    second = register([d])
    assert first["id"] != second["id"]
    assert NODE_REGISTRY[first["id"]]["indicator_reference"]["revision"] == 1
    public = next(n for n in node_catalog()["items"] if n["id"] == second["id"])
    assert "_indicator_definition" not in public
    assert public["outputs"][0]["label"] == "中轨"

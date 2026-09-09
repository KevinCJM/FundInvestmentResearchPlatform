"""Runtime-constant contract, graph roundtrip and real NJIT integration tests."""
from __future__ import annotations

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from test_custom_indicator_time_series import _service, _as_float
from custom_indicators.errors import ValidationError
from custom_indicators.series_parameters import (
    bind_parameter_input, inspect_parameter_inputs, resolve_parameter_values,
    validate_parameter_definition,
)
from custom_indicators.series_definitions import normalize_time_series_definition
from custom_indicators.graph_service import IndicatorGraphService
from custom_indicators.graph_contracts import FormulaResolveRequest, CanvasResolveRequest
from services import custom_indicator_routes


def draft(expression="rolling_mean(market_close, 20)"):
    return {"name": "可调均线", "result_kind": "time_series", "axis_anchor": "market_close",
            "series_outputs": [{"id": "ma", "label": "均线", "expression": expression, "output_measure": "auto"}]}


def opened(expression="rolling_mean(market_close, 20)"):
    value = draft(expression)
    candidate = inspect_parameter_inputs(value)["candidates"][0]
    return bind_parameter_input(value, candidate_id=candidate["id"])


def test_position_aware_binding_shared_parameters_and_unbinding():
    value = draft("rolling_mean(market_close, 20) + rolling_mean(market_close, 20)")
    candidates = inspect_parameter_inputs(value)["candidates"]
    assert len(candidates) == 2
    first = bind_parameter_input(value, candidate_id=candidates[0]["id"])
    assert first["series_outputs"][0]["expression"].count("window_1") == 1
    other = next(item for item in inspect_parameter_inputs(first)["candidates"] if item["parameter_id"] is None)
    shared = bind_parameter_input(first, candidate_id=other["id"], parameter_id="window_1")
    assert shared["series_outputs"][0]["expression"].count("window_1") == 2
    assert len(shared["parameter_schema"]) == 1
    fixed = bind_parameter_input(shared, fixed_parameter_id="window_1")
    assert not fixed["parameter_schema"]
    assert "window_1" not in fixed["series_outputs"][0]["expression"]
    assert value == draft("rolling_mean(market_close, 20) + rolling_mean(market_close, 20)")


def test_defaults_partial_overrides_and_schema_survive_normalization():
    value = opened()
    normalized = normalize_time_series_definition(value)
    assert normalized["parameter_contract_version"] == "1.0"
    assert normalized["parameter_schema"] == value["parameter_schema"]
    assert "window_1" in normalized["expression"]
    assert resolve_parameter_values(normalized, {}) == {"window_1": 20}
    assert resolve_parameter_values(normalized, {"window_1": 5}) == {"window_1": 5}


@pytest.mark.parametrize("value", [None, True, False, "5", 0, -2, 2.5, 20_001, float("inf"), float("nan")])
def test_invalid_runtime_values_fail_closed(value):
    with pytest.raises(ValidationError):
        resolve_parameter_values(opened(), {"window_1": value})


def test_unknown_keys_step_and_relations():
    value = opened("rolling_std(market_close, 20, 1, 10)")
    with pytest.raises(ValidationError):
        resolve_parameter_values(value, {"ddof": 0})
    with pytest.raises(ValidationError):
        resolve_parameter_values(value, {"window_1": 5})
    value["parameter_schema"][0].update(minimum=2, maximum=100, step=2)
    assert resolve_parameter_values(value, {"window_1": 12}) == {"window_1": 12}
    with pytest.raises(ValidationError):
        resolve_parameter_values(value, {"window_1": 11})


def test_cannot_parameterize_data_or_structural_literals():
    assert not inspect_parameter_inputs(draft("3 * market_close - 2"))["candidates"]
    value = opened()
    value["series_outputs"][0]["expression"] = "market_close * window_1"
    with pytest.raises(ValidationError):
        validate_parameter_definition(value)
    value = opened()
    value["parameter_schema"][0]["maximum"] = 30_000
    with pytest.raises(ValidationError):
        normalize_time_series_definition(value)


def test_stale_candidate_is_not_rebound_to_another_constant():
    value = draft()
    candidate = inspect_parameter_inputs(value)["candidates"][0]
    value["series_outputs"][0]["expression"] = "rolling_mean(market_close, 30)"
    with pytest.raises(ValidationError, match="重新识别"):
        bind_parameter_input(value, candidate_id=candidate["id"])


def test_njit_default_override_cache_and_history_are_instance_scoped(tmp_path):
    service, frame = _service(tmp_path)
    saved = service.create_indicator(opened())
    instance = {"indicator_id": saved["id"], "indicator_revision": saved["revision"]}
    result = service.evaluate_series(indicator_instances=[instance, {**instance, "parameters": {"window_1": 5}}],
                                     target={"kind": "etf", "product_id": "510300.SH"}, period="1M")
    left, right = result["results"]
    assert left["parameters"] == {"window_1": 20}
    assert right["parameters"] == {"window_1": 5}
    assert left["parameter_hash"] != right["parameter_hash"]
    assert left["lookback_observations"] == 20
    assert right["lookback_observations"] == 5
    for item, window in ((left, 20), (right, 5)):
        expected = frame["close"].rolling(window).mean().to_numpy()[-len(item["dates"]):]
        np.testing.assert_allclose(_as_float(item["channels"][0]["values"]), expected, rtol=1e-12)
    assert len(result["execution"]["compiled_plan_ids"]) == 1
    assert result["execution"]["request_time_compilation"] == 0
    assert result["execution"]["python_fallback"] == 0
    assert service.indicators.get(saved["id"])["parameter_schema"][0]["default"] == 20
    again = service.evaluate_series(indicator_instances=[{**instance, "parameters": {"window_1": 20}}],
                                    target={"kind": "etf", "product_id": "510300.SH"}, period="1M")
    assert again["cache"]["hits"] == 1


def test_nested_windows_and_no_request_compilation(tmp_path, monkeypatch):
    from custom_indicators import series_service
    service, frame = _service(tmp_path)
    definition = opened("rolling_mean(rolling_mean(market_close, 20), 3)")
    # The first candidate is the outer window; both input positions remain distinct.
    saved = service.create_indicator(definition)
    monkeypatch.setattr(series_service, "_compile_definition", lambda *_args, **_kwargs: pytest.fail("request-time compilation"))
    result = service.evaluate_series(indicator_instances=[{
        "indicator_id": saved["id"], "parameters": {"window_1": 5},
    }], target={"kind": "etf", "product_id": "510300.SH"}, period="1M")["results"][0]
    assert result["lookback_observations"] == 24
    expected = frame["close"].rolling(20).mean().rolling(5).mean().to_numpy()[-len(result["dates"]):]
    np.testing.assert_allclose(_as_float(result["channels"][0]["values"]), expected, rtol=1e-12)


def test_kdj_three_outputs_share_explicit_runtime_parameters(tmp_path):
    from backend.product_analysis_numba import kdj_kernel
    service, frame = _service(tmp_path)
    low = "rolling_min(market_low, n, 1)"
    high = "rolling_max(market_high, n, 1)"
    rsv = f"clip(100 * divide_or_default(market_close - {low}, {high} - {low}, 0.5), 0, 100)"
    k_value = f"recursive_smooth({rsv}, k, 50)"
    d_value = f"recursive_smooth({k_value}, d, 50)"
    definition = {**draft(), "parameter_contract_version": "1.0",
        "parameter_schema": [{"id": name, "label": name, "type": "integer", "default": default,
                              "minimum": 1, "maximum": 250, "step": 1}
                             for name, default in (("n", 9), ("k", 3), ("d", 3))],
        "series_outputs": [{"id": name, "label": name, "expression": expression, "output_measure": "auto"}
                           for name, expression in (("K", k_value), ("D", d_value), ("J", f"3 * ({k_value}) - 2 * ({d_value})"))],
    }
    saved = service.create_indicator(definition)
    result = service.evaluate_series(indicator_instances=[{"indicator_id": saved["id"], "parameters": {"n": 14, "k": 5}}],
                                     target={"kind": "etf", "product_id": "510300.SH"}, period="1M")["results"][0]
    assert result["parameters"] == {"n": 14, "k": 5, "d": 3}
    assert result["history_policy"] == "full_history"
    reference = kdj_kernel(*(np.ascontiguousarray(frame[key].to_numpy(dtype=np.float64)) for key in ("high", "low", "close")), 14, 5, 3)
    for index, channel in enumerate(result["channels"]):
        np.testing.assert_allclose(_as_float(channel["values"]), reference[index, -len(result["dates"]):], rtol=1e-12)


def test_runtime_clip_range_does_not_reuse_default_range(tmp_path):
    service, _ = _service(tmp_path)
    definition = draft("clip(market_close / market_close, 0, 1)")
    candidate = next(item for item in inspect_parameter_inputs(definition)["candidates"] if item["argument"] == "upper")
    saved = service.create_indicator(bind_parameter_input(definition, candidate_id=candidate["id"]))
    result = service.evaluate_series(indicator_instances=[{"indicator_id": saved["id"], "parameters": {"upper_1": 2}}],
                                     target={"kind": "etf", "product_id": "510300.SH"}, period="ALL")["results"][0]
    assert result["channels"][0]["value_range"]["minimum"] == 0.0
    assert result["channels"][0]["value_range"]["maximum"] == 2.0
    assert result["channels"][0]["output_measure"] != "bounded_0_1"


def test_revision_defaults_remain_locked(tmp_path):
    service, _ = _service(tmp_path)
    first = service.create_indicator(opened())
    second = service.update_indicator(first["id"], first["revision"], {**first, "parameter_schema": [{**first["parameter_schema"][0], "default": 5}]})
    result = service.evaluate_series(indicator_instances=[{"indicator_id": first["id"], "indicator_revision": revision}
                                                          for revision in (first["revision"], second["revision"])],
                                     target={"kind": "etf", "product_id": "510300.SH"}, period="1M")["results"]
    assert [item["parameters"]["window_1"] for item in result] == [20, 5]


def test_excel_export_uses_effective_parameters(tmp_path):
    service, frame = _service(tmp_path)
    saved = service.create_indicator(opened())
    artifact = service.export_excel(
        indicator_ids=[saved["id"]], inline_definition=None,
        targets=[{"kind": "etf", "product_id": "510300.SH"}], period="1M",
        parameters={"window_1": 5},
    )
    from openpyxl import load_workbook
    workbook = load_workbook(artifact.path, data_only=False)
    text = " ".join(str(cell.value) for sheet in workbook for row in sheet for cell in row if cell.value is not None)
    assert "window_1=5" in text
    workbook.close()


def test_parameter_graph_formula_roundtrip(tmp_path):
    service, _ = _service(tmp_path)
    definition = normalize_time_series_definition(opened())
    graph_service = IndicatorGraphService(service)
    context = {"result_kind": "time_series", "parameter_contract_version": "1.0", "parameter_schema": definition["parameter_schema"]}
    parsed = graph_service.resolve(FormulaResolveRequest(source_kind="formula", expressions={"ma": definition["expression"]}, **context))
    assert parsed["valid"], parsed
    assert any(node["kind"] == "parameter" for node in parsed["graph"]["nodes"])
    restored = graph_service.resolve(CanvasResolveRequest(source_kind="graph", graph=parsed["graph"], **context))
    assert restored["valid"], restored
    assert restored["expressions"] == parsed["expressions"]
    assert restored["definition_fingerprint"] == parsed["definition_fingerprint"]


def test_api_parameter_contract_and_strict_numeric_inputs(tmp_path, monkeypatch):
    service, _ = _service(tmp_path)
    monkeypatch.setattr(custom_indicator_routes, "indicator_service", service)
    app = FastAPI()
    app.include_router(custom_indicator_routes.router)
    with TestClient(app) as client:
        inspected = client.post("/api/custom-indicators/parameters/inspect", json={"definition": draft()})
        assert inspected.status_code == 200, inspected.text
        bound = client.post("/api/custom-indicators/parameters/bind", json={"definition": draft(), "candidate_id": inspected.json()["candidates"][0]["id"]})
        assert bound.status_code == 200, bound.text
        created = client.post("/api/custom-indicators", json=bound.json()["definition"])
        assert created.status_code == 201, created.text
        saved = created.json()
        for bad in (True, "5", None):
            response = client.post("/api/custom-indicators/evaluate-series", json={
                "indicator_instances": [{"indicator_id": saved["id"], "parameters": {"window_1": bad}}],
                "target": {"kind": "etf", "product_id": "510300.SH"}, "period": "ALL",
            })
            assert response.status_code == 422, response.text

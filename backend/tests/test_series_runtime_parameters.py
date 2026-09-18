"""Runtime-constant contract, graph roundtrip and real NJIT integration tests."""
from __future__ import annotations

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import tempfile
from pathlib import Path

from test_custom_indicator_time_series import _service, _as_float, _write_market_data
from cal_indicators.parameter_policy import configuration_arguments
from cal_indicators.typed_dsl import TYPED_DSL_VERSION
from cal_indicators.typed_operators import (
    TYPED_OPERATOR_REGISTRY_VERSION,
    get_typed_operator_registry,
)
from custom_indicators.service import CustomIndicatorService
from custom_indicators.errors import ValidationError
from custom_indicators.series_parameters import (
    bind_parameter_input, inspect_parameter_inputs, resolve_parameter_values,
    validate_parameter_definition,
)
from custom_indicators.series_definitions import normalize_time_series_definition
from custom_indicators.graph_service import IndicatorGraphService
from custom_indicators.graph_contracts import FormulaResolveRequest, CanvasResolveRequest
from services import custom_indicator_routes


def draft(expression="rolling_mean(adjusted_close, 20)"):
    return {"name": "可调均线", "result_kind": "time_series", "axis_anchor": "adjusted_close",
            "series_outputs": [{"id": "ma", "label": "均线", "expression": expression, "output_measure": "auto"}]}


def opened(expression="rolling_mean(adjusted_close, 20)"):
    value = draft(expression)
    candidate = inspect_parameter_inputs(value)["candidates"][0]
    return bind_parameter_input(value, candidate_id=candidate["id"])


def test_every_configuration_input_is_offered_including_aliases():
    """The contract decides; the user picks. No operator or argument whitelist."""

    value = draft("sequence_std(rolling_window(adjusted_close, 20), 1)")
    candidates = inspect_parameter_inputs(value)["candidates"]
    assert [(item["operator_id"], item["argument"], item["value"]) for item in candidates] == [
        ("sequence_std", "ddof", 1), ("rolling_window", "window", 20)
    ]
    window = next(item for item in candidates if item["argument"] == "window")
    opened_value = bind_parameter_input(value, candidate_id=window["id"])
    assert opened_value["series_outputs"][0]["expression"] == (
        "sequence_std(rolling_window(adjusted_close, window_1), 1)"
    )
    assert opened_value["parameter_schema"][0]["default"] == 20


def test_candidates_and_ranges_follow_the_operator_contract():
    value = draft("clip(quantile(rolling_window(adjusted_close, 20), 0.9), 0.0, 1.0)")
    candidates = {item["argument"]: item for item in inspect_parameter_inputs(value)["candidates"]}
    assert candidates["probability"]["minimum"] == 0 and candidates["probability"]["maximum"] == 1
    assert candidates["probability"]["exclusive_minimum"] is True
    assert candidates["probability"]["exclusive_maximum"] is True
    assert candidates["lower"]["type"] == "number" and candidates["window"]["type"] == "integer"
    # Data and expression positions stay closed because they hold no literal.
    assert "values" not in candidates


def test_configuration_inputs_have_exactly_one_source():
    """The operator signature decides; /meta, the composer and the parameter
    panel only read it.  A reintroduced whitelist breaks this test."""

    registry = get_typed_operator_registry(TYPED_OPERATOR_REGISTRY_VERSION)
    derived = {
        (spec.operator_id, name): policy
        for spec in {id(item): item for item in registry.values()}.values()
        for arity in sorted(spec.arities)
        for name, policy in configuration_arguments(spec, arity).items()
    }
    assert {key[0] for key in derived} == {
        "clip", "difference", "divide_or_default", "lag", "quantile", "quantile_where",
        "recursive_smooth", "rolling_apply", "rolling_max", "rolling_mean", "rolling_min",
        "rolling_std", "rolling_window", "std", "variance",
    }
    assert derived[("rolling_apply", "window")]["maximum"] == 5000, "kernel cap is quoted by the signature"
    assert derived[("rolling_std", "ddof")]["minimum"] == 0, "ddof is a count that may be zero"
    assert ("power", "exponent") not in derived, "an ordinary scalar value is not configuration"

    operators = {item["name"]: item for item in CustomIndicatorService(Path(tempfile.mkdtemp()), Path(tempfile.mkdtemp())).meta()["operators"]}
    for operator_id, name in derived:
        parameters = {item["name"]: item for item in operators.get(operator_id, {}).get("parameters") or []}
        if name not in parameters:
            continue
        assert parameters[name]["source_policy"] == "fixed_constant"
        assert parameters[name]["parameterizable"] is True


def test_position_aware_binding_shared_parameters_and_unbinding():
    value = draft("rolling_mean(adjusted_close, 20) + rolling_mean(adjusted_close, 20)")
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
    assert value == draft("rolling_mean(adjusted_close, 20) + rolling_mean(adjusted_close, 20)")


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
    value = opened("rolling_std(adjusted_close, 20, 1, 10)")
    with pytest.raises(ValidationError):
        resolve_parameter_values(value, {"ddof": 0})
    with pytest.raises(ValidationError):
        resolve_parameter_values(value, {"window_1": 5})
    value["parameter_schema"][0].update(minimum=2, maximum=100, step=2)
    assert resolve_parameter_values(value, {"window_1": 12}) == {"window_1": 12}
    with pytest.raises(ValidationError):
        resolve_parameter_values(value, {"window_1": 11})


def test_cannot_parameterize_data_or_structural_literals():
    assert not inspect_parameter_inputs(draft("3 * adjusted_close - 2"))["candidates"]
    value = opened()
    value["series_outputs"][0]["expression"] = "adjusted_close * window_1"
    with pytest.raises(ValidationError):
        validate_parameter_definition(value)
    value = opened()
    value["parameter_schema"][0]["maximum"] = 30_000
    with pytest.raises(ValidationError):
        normalize_time_series_definition(value)


def test_stale_candidate_is_not_rebound_to_another_constant():
    value = draft()
    candidate = inspect_parameter_inputs(value)["candidates"][0]
    value["series_outputs"][0]["expression"] = "rolling_mean(adjusted_close, 30)"
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
        expected = frame["adj_close"].rolling(window).mean().to_numpy()[-len(item["dates"]):]
        np.testing.assert_allclose(_as_float(item["channels"][0]["values"]), expected, rtol=1e-12)
    assert len(result["execution"]["compiled_plan_ids"]) == 1
    assert result["execution"]["request_time_compilation"] == 0
    assert result["execution"]["python_fallback"] == 0
    assert service.indicators.get(saved["id"])["parameter_schema"][0]["default"] == 20
    again = service.evaluate_series(indicator_instances=[{**instance, "parameters": {"window_1": 20}}],
                                    target={"kind": "etf", "product_id": "510300.SH"}, period="1M")
    assert again["cache"]["hits"] == 1


def test_repeated_instances_are_matched_by_their_own_key(tmp_path):
    """走势图允许同一指标配多条，indicator_id 不再能识别结果属于哪一条。"""

    service, _ = _service(tmp_path)
    saved = service.create_indicator(opened())
    instance = {"indicator_id": saved["id"], "indicator_revision": saved["revision"]}
    response = service.evaluate_series(
        indicator_instances=[
            {**instance, "instance_key": "ov1", "parameters": {"window_1": 5}},
            {**instance, "instance_key": "ov2", "parameters": {"window_1": 20}},
            {**instance, "instance_key": "ov3", "parameters": {"window_1": 5}},
        ],
        target={"kind": "etf", "product_id": "510300.SH"},
        period="1M",
    )
    first, second, third = response["results"]
    assert [item["instance_key"] for item in response["results"]] == ["ov1", "ov2", "ov3"]
    assert [item["indicator_id"] for item in response["results"]] == [saved["id"]] * 3
    assert first["parameters"] == third["parameters"] == {"window_1": 5}
    assert second["parameters"] == {"window_1": 20}
    # 参数相同的两条命中同一份缓存，但各自是独立记录：写第三条的键不能改到第一条。
    assert response["cache"]["hits"] == 1
    assert first is not third
    first_values, second_values, third_values = (
        _as_float(item["channels"][0]["values"]) for item in response["results"]
    )
    np.testing.assert_allclose(first_values, third_values, rtol=0, equal_nan=True)
    assert not np.allclose(first_values, second_values, rtol=0, equal_nan=True)


def test_instance_key_is_optional_and_reaches_the_route(tmp_path, monkeypatch):
    service, _ = _service(tmp_path)
    saved = service.create_indicator(opened())
    plain = service.evaluate_series(
        indicator_instances=[{"indicator_id": saved["id"]}],
        target={"kind": "etf", "product_id": "510300.SH"}, period="1M",
    )["results"][0]
    assert plain["instance_key"] is None

    monkeypatch.setattr(custom_indicator_routes, "indicator_service", service)
    app = FastAPI()
    app.include_router(custom_indicator_routes.router)
    client = TestClient(app)
    payload = client.post("/api/custom-indicators/evaluate-series", json={
        "indicator_instances": [
            {"indicator_id": saved["id"], "instance_key": "ov1"},
            {"indicator_id": saved["id"], "instance_key": "ov2", "parameters": {"window_1": 5.0}},
        ],
        "target": {"kind": "etf", "product_id": "510300.SH"},
        "period": "1M",
    })
    assert payload.status_code == 200
    assert [item["instance_key"] for item in payload.json()["results"]] == ["ov1", "ov2"]


def test_nested_windows_and_no_request_compilation(tmp_path, monkeypatch):
    from custom_indicators import series_service
    service, frame = _service(tmp_path)
    definition = opened("rolling_mean(rolling_mean(adjusted_close, 20), 3)")
    # The first candidate is the outer window; both input positions remain distinct.
    saved = service.create_indicator(definition)
    monkeypatch.setattr(series_service, "_compile_definition", lambda *_args, **_kwargs: pytest.fail("request-time compilation"))
    result = service.evaluate_series(indicator_instances=[{
        "indicator_id": saved["id"], "parameters": {"window_1": 5},
    }], target={"kind": "etf", "product_id": "510300.SH"}, period="1M")["results"][0]
    assert result["lookback_observations"] == 24
    expected = frame["adj_close"].rolling(20).mean().rolling(5).mean().to_numpy()[-len(result["dates"]):]
    np.testing.assert_allclose(_as_float(result["channels"][0]["values"]), expected, rtol=1e-12)


def test_kdj_three_outputs_share_explicit_runtime_parameters(tmp_path):
    from backend.product_analysis_numba import kdj_kernel
    service, frame = _service(tmp_path)
    low = "rolling_min(adjusted_low, n, 1)"
    high = "rolling_max(adjusted_high, n, 1)"
    rsv = f"clip(100 * divide_or_default(adjusted_close - {low}, {high} - {low}, 0.5), 0, 100)"
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
    reference = kdj_kernel(*(np.ascontiguousarray(frame[key].to_numpy(dtype=np.float64)) for key in ("adj_high", "adj_low", "adj_close")), 14, 5, 3)
    for index, channel in enumerate(result["channels"]):
        np.testing.assert_allclose(_as_float(channel["values"]), reference[index, -len(result["dates"]):], rtol=1e-12)


def test_runtime_clip_range_does_not_reuse_default_range(tmp_path):
    service, _ = _service(tmp_path)
    definition = draft("clip(adjusted_close / adjusted_close, 0, 1)")
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


def _scalar_service(tmp_path):
    """The scalar lane warms on demand, so only the kernels are prepared here."""

    from cal_indicators.typed_numba_kernels import warm_numba_kernel_registry

    frame = _write_market_data(tmp_path)
    warm_numba_kernel_registry()
    return CustomIndicatorService(tmp_path, tmp_path), frame


def _scalar_draft(expression="quantile(returns, 0.9)"):
    return {"name": "分位收益", "result_kind": "scalar", "expression": expression,
            "dsl_version": TYPED_DSL_VERSION, "direction": "higher_better"}


def test_scalar_indicator_opens_the_same_configuration_inputs_as_a_series():
    """Result kind is not part of the rule; the operator contract is."""

    candidates = inspect_parameter_inputs(_scalar_draft("clip(quantile(returns, 0.9), 0.0, 1.0)"))["candidates"]
    assert {(item["operator_id"], item["argument"]) for item in candidates} == {
        ("quantile", "probability"), ("clip", "lower"), ("clip", "upper")
    }
    draft_definition = _scalar_draft()
    opened = bind_parameter_input(
        draft_definition,
        candidate_id=inspect_parameter_inputs(draft_definition)["candidates"][0]["id"],
    )
    assert opened["expression"] == "quantile(returns, probability_1)"
    assert opened["parameter_contract_version"] == "1.0"
    # Nothing may be opened on a position the contract calls an ordinary value.
    with pytest.raises(ValidationError):
        validate_parameter_definition({**_scalar_draft("power(returns_total, p_1)"),
                                       "parameter_schema": [{"id": "p_1", "label": "指数", "type": "number",
                                                             "default": 2.0, "minimum": 0.0, "maximum": 5.0, "step": 0.1}]})


def test_scalar_parameter_values_run_on_one_prewarmed_batch_plan(tmp_path):
    """Phase 3-5 end to end: values travel in the parameter vector, so a single
    warm fused NJIT plan serves every value instead of one plan per value."""

    service, frame = _scalar_service(tmp_path)
    draft_definition = _scalar_draft()
    candidate = inspect_parameter_inputs(draft_definition)["candidates"][0]
    saved = service.create_indicator(bind_parameter_input(draft_definition, candidate_id=candidate["id"]))

    def run(probability):
        refs = [{"indicator_id": saved["id"], "indicator_revision": saved["revision"],
                 "parameters": {"probability_1": probability}}]
        service.prepare_evaluation(indicator_ids=[], indicator_refs=refs)
        return service.evaluate(indicator_ids=[], inline_definition=None, indicator_refs=refs,
                                targets=[{"kind": "etf", "product_id": "510300.SH"}], period="ALL")

    nav = frame["adj_nav"].to_numpy(dtype=np.float64)
    returns = nav[1:] / nav[:-1] - 1.0
    plan_ids = set()
    hashes = set()
    for probability in (0.5, 0.9):
        response = run(probability)
        record = response["results"][0]
        assert record["status"] == "ok"
        assert record["value"] == pytest.approx(float(np.quantile(returns, probability)), rel=1e-12)
        assert record["parameters"] == {"probability_1": probability}
        plan_ids.update(response["execution"]["compiled_plan_ids"])
        hashes.add(record["parameter_hash"])
    assert len(plan_ids) == 1, "a parameter value must not specialise a compiled plan"
    assert len(hashes) == 2, "two values must not share a result cache entry"

    with pytest.raises(ValidationError):
        run(1.5)


def test_prepare_evaluation_returns_the_parameters_it_was_given(tmp_path):
    """The client swaps its refs for the prepared ones, so parameters must survive.

    Dropping them here is silent: every indicator falls back to its locked
    default and still reports ``ok``.
    """

    service, frame = _scalar_service(tmp_path)
    candidate = inspect_parameter_inputs(_scalar_draft())["candidates"][0]
    bound = bind_parameter_input(_scalar_draft(), candidate_id=candidate["id"])
    low = service.create_indicator(bound)
    high = service.create_indicator({**bound, "name": "分位收益（高）"})
    refs = [{"indicator_id": low["id"], "indicator_revision": low["revision"], "parameters": {"probability_1": 0.25}},
            {"indicator_id": high["id"], "indicator_revision": high["revision"], "parameters": {"probability_1": 0.75}}]
    prepared = service.prepare_evaluation(indicator_ids=[], indicator_refs=refs)["indicator_refs"]
    assert [item.get("parameters") for item in prepared] == [{"probability_1": 0.25}, {"probability_1": 0.75}]
    response = service.evaluate(indicator_ids=[], inline_definition=None, indicator_refs=prepared,
                                targets=[{"kind": "etf", "product_id": "510300.SH"}], period="ALL")
    nav = frame["adj_nav"].to_numpy(dtype=np.float64)
    returns = nav[1:] / nav[:-1] - 1.0
    assert [record["value"] for record in response["results"]] == [
        pytest.approx(float(np.quantile(returns, 0.25)), rel=1e-12),
        pytest.approx(float(np.quantile(returns, 0.75)), rel=1e-12),
    ]
    # An indicator with no override keeps a bare ref; nothing invents an empty map.
    plain = service.prepare_evaluation(indicator_ids=[], indicator_refs=[{"indicator_id": low["id"], "indicator_revision": low["revision"]}])
    assert plain["indicator_refs"] == [{"indicator_id": low["id"], "indicator_revision": low["revision"]}]


def test_saved_plan_locks_and_replays_its_parameter_values(tmp_path):
    """D2: a plan revision that cannot be replayed is not a saved evaluation."""

    service, frame = _scalar_service(tmp_path)
    draft_definition = _scalar_draft()
    candidate = inspect_parameter_inputs(draft_definition)["candidates"][0]
    saved = service.create_indicator(bind_parameter_input(draft_definition, candidate_id=candidate["id"]))
    plan = service.create_plan({
        "name": "分位方案", "product_kind": "etf",
        "targets": [{"kind": "etf", "product_id": "510300.SH"}],
        "indicators": [{"indicator_id": saved["id"], "indicator_revision": saved["revision"],
                        "period": "ALL", "weight": 1.0, "parameters": {"probability_1": 0.25}}],
    })
    assert plan["indicators"][0]["parameters"] == {"probability_1": 0.25}
    nav = frame["adj_nav"].to_numpy(dtype=np.float64)
    row = service.run_plan(plan["id"])["rows"][0]
    assert row["values"][0]["value"] == pytest.approx(
        float(np.quantile(nav[1:] / nav[:-1] - 1.0, 0.25)), rel=1e-12
    )


@pytest.mark.parametrize("probability", [0.995, 0.005, 0.9995])
def test_probability_open_interval_survives_bind_save_and_runtime(tmp_path, probability):
    service, _ = _scalar_service(tmp_path)
    value = _scalar_draft()
    value["expression"] = f"quantile(returns, {probability})"
    candidate = inspect_parameter_inputs(value)["candidates"][0]
    bound = bind_parameter_input(value, candidate_id=candidate["id"])
    saved = service.create_indicator(bound)
    assert saved["parameter_schema"][0]["exclusive_minimum"] is True
    assert resolve_parameter_values(saved, {}) == {"probability_1": probability}
    for invalid in (0, 1):
        with pytest.raises(ValidationError):
            resolve_parameter_values(saved, {"probability_1": invalid})
    result = service.compose({"indicator_id": saved["id"], "indicator_revision": saved["revision"]})
    assert "probability_1" not in result["expression"]
    assert str(probability) in result["expression"]
    assert result["indicator_origin"]["parameters"] == {"probability_1": probability}
    assert service.indicators.get(saved["id"])["expression"] == saved["expression"]


def test_snapshot_series_parameter_instances_compute_separately(tmp_path):
    from custom_indicators.snapshot_execution import _records
    service, frame = _scalar_service(tmp_path)
    saved = service.create_indicator(opened())
    refs = [{"indicator_id": saved["id"], "indicator_revision": saved["revision"], "channel_id": "ma",
             "reducer": "last_finite", "period": "ALL", "parameters": {"window_1": n}} for n in (5, 20)]
    config = service.update_snapshot_config(service.get_snapshot_config()["revision"], refs)
    records = list(_records(service, config["items"], [{"kind": "etf", "product_id": "510300.SH"}]))
    assert len(records) == 2
    assert len({item["field"] for item, _ in records}) == 2
    for item, record in records:
        n = int(item["parameters"]["window_1"])
        assert record["parameters"]["window_1"] == n
        assert record["value"] == pytest.approx(frame["adj_close"].iloc[-n:].mean())

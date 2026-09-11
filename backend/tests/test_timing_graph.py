"""Real template execution, shared indicator reuse and fail-closed authoring."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest

from backend.timing_research.catalog import build_catalog, templates
from backend.timing_research.contracts import Definition
from backend.timing_research.graph import GraphRuntime, definition_hash
from custom_indicators.errors import ValidationError


def bars(values):
    values = np.asarray(values, dtype=np.float64)
    values.setflags(write=False)
    dates = np.arange(values.size, dtype=np.int64)
    dates.setflags(write=False)
    return SimpleNamespace(dates=dates, close=values, open=values, high=values, low=values, volume=values)


def baskets(values):
    panel = np.vstack((values, values * 1.1))
    panel.setflags(write=False)
    return {"market": panel, "category": panel}


def formula_definition(expression="a", extras=()):
    return Definition(name="测试", nodes=[
        {"id": "price", "label": "收盘价", "op": "source", "parameters": {"field": "close"}},
        {"id": "feature", "label": "特征", "op": "formula", "inputs": {"a": "price.value"}, "parameters": {"expression": expression}},
        {"id": "entry", "label": "入场", "op": "compare", "inputs": {"left": "feature.value"}, "parameters": {"operator": "gt", "threshold": 0.0}},
        *extras,
    ], entry="entry.value")


@pytest.fixture(scope="module")
def runtime():
    return GraphRuntime(build_catalog())


@pytest.mark.parametrize("template", templates(), ids=lambda item: item["id"])
def test_every_real_template_prepares_and_executes_full_axis(runtime, template):
    definition = Definition.model_validate(template["definition"])
    original = definition.model_dump(mode="json")
    prepared = runtime.prepare(definition)
    data = bars(150 + 30 * np.sin(np.arange(700) / 17.0))
    before = {key: list(value.compiled_signatures) for key, value in prepared.formulas.items()}
    channels = runtime.evaluate(prepared, data, baskets(data.close))
    assert channels[definition.entry].shape == data.dates.shape
    assert channels[definition.entry].dtype == np.int64
    assert set(np.unique(channels[definition.entry])) <= {-1, 0, 1}
    assert all(not values.flags.writeable for values in channels.values())
    close_sources = [node.id for node in definition.nodes if node.op == "source" and node.parameters.get("field", "close") == "close"]
    assert any(np.shares_memory(channels[f"{node}.value"], data.close) for node in close_sources)
    assert original == definition.model_dump(mode="json")
    assert before == {key: list(value.compiled_signatures) for key, value in prepared.formulas.items()}
    audit = runtime.audit(prepared)
    assert audit["python_fallback"] == audit["request_time_compilation"] == 0
    assert audit["source_node_copies"] == 0 and audit["kernel_signatures"]
    assert runtime.get(definition, prepared.digest) is prepared


def test_shared_rolling_mean_and_missing_comparison_match_reference(runtime, monkeypatch):
    definition = formula_definition("rolling_mean(a,3,3)")
    prepared = runtime.prepare(definition)
    # Production run cannot prepare a new mathematical signature.
    import backend.timing_research.graph as graph
    monkeypatch.setattr(graph, "compile_numba_series_plan", lambda *_: pytest.fail("run compiled a plan"))
    data = bars([1, 2, 3, 4, np.nan, 6, 7, 8])
    outputs = runtime.evaluate(prepared, data)
    np.testing.assert_allclose(outputs["feature.value"], [np.nan, np.nan, 2, 3, np.nan, np.nan, np.nan, 7], equal_nan=True)
    np.testing.assert_array_equal(outputs["entry.value"], [-1, -1, 1, 1, -1, -1, -1, 1])
    compiled = prepared.formulas["feature.value"]
    assert any(getattr(value, "__name__", None) == "rolling_mean_1d"
               for value in compiled.dispatcher.py_func.__globals__.values())
    assert not compiled.dispatcher._can_compile


def test_same_axis_lag_and_difference_preserve_date_positions(runtime):
    definition = formula_definition()
    definition.nodes[1].op = "indicator.lag"
    definition.nodes[1].inputs = {"values": "price.value"}
    definition.nodes[1].parameters = {"periods": 2}
    data = bars([10, 12, np.nan, 15, 18])
    channels = runtime.evaluate(runtime.prepare(definition), data)
    np.testing.assert_allclose(channels["feature.value"], [np.nan, np.nan, 10, 12, np.nan], equal_nan=True)
    definition.nodes[1].op = "indicator.difference"
    channels = runtime.evaluate(runtime.prepare(definition), data)
    np.testing.assert_allclose(channels["feature.value"], [np.nan, np.nan, np.nan, 3, np.nan], equal_nan=True)


@pytest.mark.parametrize("expression", ["lag(a,-1)", "difference(a,1)", "a/mean(a)", "a + last(a)",
                                        "a[0]", "__import__('os')", "a > 1", "a+b", "rolling_mean(a,1e309,1)"])
def test_invalid_future_global_type_or_unknown_input_rejected(runtime, expression):
    with pytest.raises(ValidationError):
        runtime.prepare(formula_definition(expression))


def test_disconnected_future_branch_is_rejected(runtime):
    extra = {"id": "bad", "label": "未连接未来分支", "op": "formula", "inputs": {"a": "price.value"},
             "parameters": {"expression": "a+last(a)"}}
    with pytest.raises(ValidationError):
        runtime.prepare(formula_definition(extras=[extra]))


def test_cycles_and_condition_numeric_wiring_are_rejected(runtime):
    bad = formula_definition()
    bad.nodes[1].inputs["a"] = "feature.value"
    with pytest.raises(ValidationError, match="循环"):
        runtime.prepare(bad)
    bad = formula_definition()
    bad.nodes[1].inputs["a"] = "entry.value"
    with pytest.raises(ValidationError, match="类型"):
        runtime.prepare(bad)


def test_changed_definition_cannot_use_previous_compile_token(runtime):
    definition = formula_definition()
    prepared = runtime.prepare(definition)
    changed = definition.model_copy(deep=True)
    changed.nodes[2].parameters["threshold"] = 5.0
    assert definition_hash(changed) != prepared.digest
    with pytest.raises(ValidationError, match="重新准备"):
        runtime.get(changed, prepared.digest)


def test_disconnected_formula_not_executed_and_source_owner_is_not_mutated(runtime):
    extra = {"id": "unused", "label": "未连接昂贵分支", "op": "formula", "inputs": {"a": "price.value"},
             "parameters": {"expression": "exp(a)"}}
    prepared = runtime.prepare(formula_definition(extras=[extra]))
    owner = np.full(8, 1000.0)
    data = SimpleNamespace(dates=np.arange(8, dtype=np.int64), close=owner)
    values = runtime.evaluate(prepared, data)
    assert "unused.value" not in values and "unused.value" not in prepared.formulas
    assert owner.flags.writeable and np.shares_memory(values["price.value"], owner)


def test_strided_sources_are_zero_copy_and_formula_boundary_fails_without_copy(runtime):
    definition = formula_definition()
    definition.nodes = [definition.nodes[0], definition.nodes[2]]
    definition.nodes[1].inputs = {"left": "price.value"}
    owner = np.ones((8, 2))
    values = owner[:, 0]
    values.setflags(write=False)
    data = SimpleNamespace(dates=np.arange(8, dtype=np.int64), close=values)
    outputs = runtime.evaluate(runtime.prepare(definition), data)
    assert np.shares_memory(outputs["price.value"], owner)
    with pytest.raises(ValidationError, match="连续数组"):
        runtime.evaluate(runtime.prepare(formula_definition()), data)


def test_runtime_rejects_invalid_dtype_and_infinity(runtime):
    prepared = runtime.prepare(formula_definition())
    with pytest.raises(ValidationError, match="类型"):
        runtime.evaluate(prepared, SimpleNamespace(dates=np.arange(5, dtype=np.int64), close=np.ones(5, dtype=np.float32)))
    with pytest.raises(ValidationError, match="无穷"):
        runtime.evaluate(prepared, bars([1, 2, np.inf]))


def test_division_keeps_zero_and_missing_undefined(runtime):
    prepared = runtime.prepare(formula_definition("a/a"))
    outputs = runtime.evaluate(prepared, bars([2, 0, np.nan, 4]))
    np.testing.assert_allclose(outputs["feature.value"], [1, np.nan, np.nan, 1], equal_nan=True)
    np.testing.assert_array_equal(outputs["entry.value"], [1, -1, -1, 1])


def test_reference_indicator_revision_multi_outputs_use_shared_bundle():
    saved = dict(id="test-mean", revision=3, name="指标中心均线", context_kind="single_product", result_kind="time_series",
                 dsl_version="2.4.0", operator_registry_version="2.4.0", parameter_schema=[], series_outputs=[
                     {"id": "middle", "label": "均线", "expression": "rolling_mean(market_close,3,3)"},
                     {"id": "upper", "label": "上界", "expression": "rolling_mean(market_close,3,3)+2"}])
    service = SimpleNamespace(indicators=SimpleNamespace(list_all_versions=lambda: [saved]),
                              get_indicator=lambda identifier, revision: copy.deepcopy(saved))
    catalog = build_catalog(service)
    reference = next(item for item in catalog.values() if item.get("indicator_reference", {}).get("id") == saved["id"])
    assert reference["indicator_reference"]["revision"] == 3
    definition = Definition(name="复用指标", nodes=[
        {"id": "price", "label": "收盘价", "op": "source"},
        {"id": "mean", "label": "均线", "op": reference["id"], "inputs": {"market_close": "price.value"}},
        {"id": "entry", "label": "入场", "op": "compare", "inputs": {"left": "mean.middle", "right": "mean.upper"}},
    ], entry="entry.value")
    runtime = GraphRuntime(catalog)
    prepared = runtime.prepare(definition)
    assert prepared.formulas["mean.middle"] is prepared.formulas["mean.upper"]
    compiled = prepared.formulas["mean.middle"]
    mean_alias = next(name for name, value in compiled.dispatcher.py_func.__globals__.items()
                      if getattr(value, "__name__", None) == "rolling_mean_1d")
    assert compiled.source.count(mean_alias + "(") == 1
    outputs = runtime.evaluate(prepared, bars([10, 11, 12, 13, 14]))
    np.testing.assert_allclose(outputs["mean.middle"], [np.nan, np.nan, 11, 12, 13], equal_nan=True)
    np.testing.assert_allclose(outputs["mean.upper"], [np.nan, np.nan, 13, 14, 15], equal_nan=True)
    # Unused output does not incur a plan root or result allocation.
    definition.nodes[2].inputs = {"left": "mean.middle"}
    reduced = runtime.prepare(definition)
    assert "mean.upper" not in reduced.formulas


def test_future_perturbations_do_not_change_template_past(runtime):
    data = 150 + 30 * np.sin(np.arange(700) / 17.0)
    changed = data.copy()
    changed[550:] *= 2.0
    for template in templates():
        prepared = runtime.prepare(Definition.model_validate(template["definition"]))
        expected = runtime.evaluate(prepared, bars(data), baskets(data))
        actual = runtime.evaluate(prepared, bars(changed), baskets(changed))
        for key in actual:
            np.testing.assert_array_equal(actual[key][..., :550], expected[key][..., :550])


def test_window_cost_budget_is_checked_before_numerical_execution(runtime):
    definition = formula_definition("rolling_max(rolling_max(rolling_max(rolling_max(a,5000,1),5000,1),5000,1),5000,1)")
    prepared = runtime.prepare(definition)
    with pytest.raises(ValidationError, match="预算"):
        runtime.evaluate(prepared, bars(np.ones(12000)))

"""Fixed ETF basket axes and shared-kernel, read-only execution contracts."""
import numpy as np
import pytest
from types import SimpleNamespace

from backend.timing_research import panel
from cal_indicators.typed_numba_kernels import axis_reduce_time_fixed
from historical_regimes.condition_numba import condition_compare_kernel
from historical_regimes.v2_numba import unary_transform_kernel
from backend.timing_research.numeric import condition_values_kernel


@pytest.fixture(autouse=True, scope="module")
def warmed():
    audit = panel.warm_panel_kernels()
    assert audit["complete"] and audit["nopython"]
    assert audit["request_time_compilation"] == audit["python_fallback"] == 0
    assert all(audit["kernel_signatures"].values())


def test_panel_lag_is_shared_same_axis_not_shifted_or_imputed():
    values = np.array([[1., 2., np.nan, 4., 5.], [10., np.inf, 30., 40., 50.]])
    expected = np.array([[np.nan, np.nan, 1., 2., np.nan], [np.nan, np.nan, 10., np.nan, 30.]])
    actual = panel.panel_lag_kernel(values, 2)
    np.testing.assert_allclose(actual, expected, equal_nan=True)
    for member in range(values.shape[0]):
        np.testing.assert_allclose(actual[member], unary_transform_kernel(values[member], 4, 2), equal_nan=True)
    assert np.isnan(panel.panel_lag_kernel(values, 8)).all()
    for lag in (0, -1):
        with pytest.raises(ValueError, match="positive lag"):
            panel.panel_lag_kernel(values, lag)


@pytest.mark.parametrize("opcode", [40, 41, 42, 43, 44, 45])
def test_panel_comparison_matches_shared_scalar_and_series_boundaries(opcode):
    values = np.array([[1., 2., np.nan, 4.], [2., 1., np.inf, -np.inf]])
    bounds = np.array([[1., 1., 2., np.nan], [2., 2., 1., 0.]])
    empty = np.empty((0, 0), dtype=np.float64)
    scalar_result = panel.panel_compare_kernel(values, empty, 2., opcode)
    series_result = panel.panel_compare_kernel(values, bounds, np.nan, opcode)
    for member in range(2):
        np.testing.assert_array_equal(scalar_result[member], condition_compare_kernel(values[member], np.empty(0), 2., opcode))
        np.testing.assert_array_equal(series_result[member], condition_compare_kernel(values[member], bounds[member], np.nan, opcode))
    assert np.all(scalar_result[:, 2] == -1)
    assert series_result[0, 3] == -1


def test_panel_comparison_rejects_implicit_broadcast_and_ambiguous_empty_bounds():
    values = np.ones((3, 4))
    for right in (np.ones((2, 4)), np.ones((3, 3)), np.empty((0, 4)), np.empty((3, 0))):
        with pytest.raises(ValueError, match="axes"):
            panel.panel_compare_kernel(values, right, 1., 40)
    with pytest.raises(ValueError, match="opcode"):
        panel.panel_compare_kernel(values, np.empty((0, 0)), 1., 0)
    assert np.all(panel.panel_compare_kernel(values, np.empty((0, 0)), np.inf, 40) == -1)


def test_cross_mean_reduces_member_axis_with_all_member_completeness():
    values = np.array([[1., 2., 10., np.nan, 5.], [3., 4., 20., 4., 6.], [8., 9., 30., 8., 7.]])
    actual = panel.cross_mean_kernel(values)
    np.testing.assert_allclose(actual, [4., 5., 20., np.nan, 6.], equal_nan=True)
    np.testing.assert_allclose(actual, axis_reduce_time_fixed(121, values), equal_nan=True)
    assert actual.shape == (5,)
    for row in range(3):
        nonfinite = values.copy()
        nonfinite[row, 2] = np.inf if row % 2 == 0 else -np.inf
        result = panel.cross_mean_kernel(nonfinite)
        assert np.isnan(result[2])
        np.testing.assert_allclose(result[[0, 1, 4]], actual[[0, 1, 4]])


def test_breadth_is_cross_member_fraction_not_time_ratio_and_preserves_missing():
    conditions = np.array([[1, 1, -1, 0, 1], [0, 1, 1, 0, 0], [0, 1, 0, 1, 0]], dtype=np.int64)
    expected = [1 / 3, 1., np.nan, 1 / 3, 1 / 3]
    np.testing.assert_allclose(panel.breadth_kernel(conditions), expected, equal_nan=True)
    shared = np.vstack([condition_values_kernel(row) for row in conditions])
    np.testing.assert_allclose(panel.breadth_kernel(conditions), axis_reduce_time_fixed(121, shared), equal_nan=True)
    for invalid in (-2, 2):
        conditions[0, 2] = invalid
        with pytest.raises(ValueError, match="Conditions"):
            panel.breadth_kernel(conditions)


@pytest.mark.parametrize("members", [0, 1, 13])
@pytest.mark.parametrize("observations", [0, 4])
def test_member_contract_is_consistent_including_empty_member_inputs(members, observations):
    values = np.zeros((members, observations))
    conditions = np.zeros((members, observations), dtype=np.int64)
    for kernel, args in ((panel.panel_lag_kernel, (values, 1)),
                         (panel.panel_compare_kernel, (values, np.empty((0, 0)), 0., 40)),
                         (panel.cross_mean_kernel, (values,)), (panel.breadth_kernel, (conditions,))):
        with pytest.raises(ValueError, match="2-12 members"):
            kernel(*args)


def test_empty_time_axis_is_preserved_with_valid_fixed_members():
    values = np.empty((2, 0))
    assert panel.panel_lag_kernel(values, 1).shape == (2, 0)
    assert panel.panel_compare_kernel(values, values, 0., 40).shape == (2, 0)
    assert panel.cross_mean_kernel(values).shape == (0,)
    assert panel.breadth_kernel(np.empty((2, 0), dtype=np.int64)).shape == (0,)


@pytest.mark.parametrize("layout", ["C", "strided", "F", "reversed"])
def test_readonly_layouts_share_input_memory_and_fixed_signatures(layout):
    values = np.arange(24, dtype=np.float64).reshape(3, 8)
    owner = np.repeat(np.repeat(values, 2, axis=0), 2, axis=1) if layout == "strided" else values.copy(order="F" if layout == "F" else "C")
    view = owner[::2, ::2] if layout == "strided" else owner[:, ::-1] if layout == "reversed" else owner.view()
    snapshot = owner.copy()
    view.setflags(write=False)
    assert np.shares_memory(view, owner)
    dispatchers = [*panel.PANEL_KERNELS.values(), axis_reduce_time_fixed,
                   condition_compare_kernel, unary_transform_kernel, condition_values_kernel]
    signatures = [tuple(kernel.signatures) for kernel in dispatchers]
    lag = panel.panel_lag_kernel(view, 1)
    comparisons = panel.panel_compare_kernel(view, lag, 0., 44)
    comparisons.setflags(write=False)
    assert panel.cross_mean_kernel(view).shape == (8,)
    assert panel.breadth_kernel(comparisons[:, ::-1]).shape == (8,)
    assert signatures == [tuple(kernel.signatures) for kernel in dispatchers]
    assert lag.flags.c_contiguous and comparisons.flags.c_contiguous
    np.testing.assert_array_equal(owner, snapshot)
    assert owner.flags.writeable and not view.flags.writeable


def test_future_members_and_prices_do_not_change_prefix():
    values = np.arange(36, dtype=np.float64).reshape(3, 12)
    mean = panel.cross_mean_kernel(values)
    lag = panel.panel_lag_kernel(values, 2)
    comparisons = panel.panel_compare_kernel(values, lag, 0., 44)
    breadth = panel.breadth_kernel(comparisons)
    values[:, 8:] = np.nan
    np.testing.assert_allclose(panel.cross_mean_kernel(values)[:8], mean[:8])
    updated_lag = panel.panel_lag_kernel(values, 2)
    np.testing.assert_allclose(updated_lag[:, :8], lag[:, :8], equal_nan=True)
    updated_conditions = panel.panel_compare_kernel(values, updated_lag, 0., 44)
    np.testing.assert_allclose(panel.breadth_kernel(updated_conditions)[:8], breadth[:8], equal_nan=True)


def test_no_python_fallback_and_unsupported_dtype_no_new_signature(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Python numerical fallback executed")
    for kernel in [*panel.PANEL_KERNELS.values(), axis_reduce_time_fixed,
                   condition_compare_kernel, unary_transform_kernel, condition_values_kernel]:
        monkeypatch.setattr(kernel, "py_func", forbidden)
    values = np.arange(18, dtype=np.float64).reshape(3, 6)
    lag = panel.panel_lag_kernel(values, 1)
    conditions = panel.panel_compare_kernel(values, lag, 0., 44)
    np.testing.assert_allclose(panel.cross_mean_kernel(values), [6., 7., 8., 9., 10., 11.])
    np.testing.assert_allclose(panel.breadth_kernel(conditions), [np.nan, 1., 1., 1., 1., 1.], equal_nan=True)
    for dtype in (np.float32, object):
        with pytest.raises(TypeError):
            panel.cross_mean_kernel(values.astype(dtype))
    with pytest.raises(TypeError):
        panel.breadth_kernel(conditions.astype(np.int32))
    assert all(not kernel._can_compile for kernel in panel.PANEL_KERNELS.values())


def test_readiness_fails_closed_if_a_dispatcher_can_compile(monkeypatch):
    monkeypatch.setattr(panel.cross_mean_kernel, "_can_compile", True)
    with pytest.raises(RuntimeError, match="预热"):
        panel.warm_panel_kernels()


def basket_graph():
    from backend.timing_research.catalog import build_catalog
    from backend.timing_research.contracts import Definition
    from backend.timing_research.graph import GraphRuntime
    runtime = GraphRuntime(build_catalog())
    definition = Definition(name="固定篮子宽度", nodes=[
        {"id": "basket", "label": "市场篮子", "op": "basket_source", "parameters": {"group": "market"}},
        {"id": "smooth", "label": "成员均线", "op": "panel_formula", "inputs": {"a": "basket.value"},
         "parameters": {"expression": "rolling_mean(a,2,2)"}},
        {"id": "above", "label": "成员在均线上", "op": "panel_compare", "inputs": {"left": "basket.value", "right": "smooth.value"},
         "parameters": {"operator": "gt"}},
        {"id": "breadth", "label": "篮子宽度", "op": "breadth", "inputs": {"value": "above.value"}},
        {"id": "entry", "label": "宽度达标", "op": "compare", "inputs": {"left": "breadth.value"},
         "parameters": {"operator": "ge", "threshold": .5}},
    ], entry="entry.value")
    return runtime, runtime.prepare(definition)


def test_real_basket_graph_executes_shared_plan_without_request_preparation(monkeypatch):
    import backend.timing_research.graph as graph
    runtime, prepared = basket_graph()
    values = np.array([[1., 2., 3., 4., np.nan, 6.], [3., 3., 3., 3., 3., 3.]])
    data = SimpleNamespace(dates=np.arange(6, dtype=np.int64))
    signatures = {name: plan.compiled_signatures for name, plan in prepared.formulas.items()}
    monkeypatch.setattr(graph, "compile_numba_series_plan", lambda *_: pytest.fail("Request compiled a new plan"))
    channels = runtime.evaluate(prepared, data, {"market": values})
    np.testing.assert_allclose(channels["smooth.value"], [[np.nan, 1.5, 2.5, 3.5, np.nan, np.nan],
                                                        [np.nan, 3., 3., 3., 3., 3.]], equal_nan=True)
    np.testing.assert_allclose(channels["breadth.value"], [np.nan, .5, .5, .5, np.nan, np.nan], equal_nan=True)
    np.testing.assert_array_equal(channels["entry.value"], [-1, 1, 1, 1, -1, -1])
    assert np.shares_memory(channels["basket.value"], values)
    assert values.flags.writeable and all(not array.flags.writeable for array in channels.values())
    assert signatures == {name: plan.compiled_signatures for name, plan in prepared.formulas.items()}
    assert runtime.audit(prepared)["request_time_compilation"] == 0


@pytest.mark.parametrize("bad", [np.ones((1, 6)), np.ones((2, 5)), np.ones((2, 6), dtype=np.float32),
                                  np.ones((2, 6), dtype=object), np.full((2, 6), np.inf)])
def test_real_basket_graph_rejects_wrong_member_or_time_axis_dtype_and_inf(bad):
    from custom_indicators.errors import ValidationError
    runtime, prepared = basket_graph()
    with pytest.raises(ValidationError):
        runtime.evaluate(prepared, SimpleNamespace(dates=np.arange(6, dtype=np.int64)), {"market": bad})


def test_real_basket_formula_explicitly_rejects_strided_layout_without_copying():
    from custom_indicators.errors import ValidationError
    runtime, prepared = basket_graph()
    owner = np.ones((2, 12))
    values = owner[:, ::2]
    values.setflags(write=False)
    assert np.shares_memory(values, owner)
    with pytest.raises(ValidationError, match="连续数组"):
        runtime.evaluate(prepared, SimpleNamespace(dates=np.arange(6, dtype=np.int64)), {"market": values})
    assert owner.flags.writeable and not values.flags.writeable


def two_basket_definition():
    from backend.timing_research.contracts import Definition
    return Definition(name="两类篮子分别聚合", nodes=[
        {"id": "market", "label": "市场成员", "op": "basket_source", "parameters": {"group": "market"}},
        {"id": "category", "label": "类别成员", "op": "basket_source", "parameters": {"group": "category"}},
        {"id": "market_mean", "label": "市场等权", "op": "cross_mean", "inputs": {"value": "market.value"}},
        {"id": "category_mean", "label": "类别等权", "op": "cross_mean", "inputs": {"value": "category.value"}},
        {"id": "spread", "label": "篮子聚合之差", "op": "formula",
         "inputs": {"a": "market_mean.value", "b": "category_mean.value"}, "parameters": {"expression": "a-b"}},
        {"id": "entry", "label": "市场领先", "op": "compare", "inputs": {"left": "spread.value"},
         "parameters": {"operator": "gt", "threshold": 0.}},
    ], entry="entry.value")


@pytest.mark.parametrize("op", ["panel_formula", "panel_compare"])
@pytest.mark.parametrize("derived", [False, True])
def test_same_shape_different_nominal_baskets_cannot_be_combined_before_aggregation(op, derived):
    from backend.timing_research.catalog import build_catalog
    from backend.timing_research.contracts import Step
    from backend.timing_research.graph import GraphRuntime
    from custom_indicators.errors import ValidationError
    definition = two_basket_definition()
    left, right = "market.value", "category.value"
    if derived:
        definition.nodes += [
            Step(id="market_lag", label="市场滞后", op="panel_lag", inputs={"values": left}, parameters={"periods": 1}),
            Step(id="category_smooth", label="类别均线", op="panel_formula", inputs={"a": right},
                 parameters={"expression": "rolling_mean(a,2,2)"}),
        ]
        left, right = "market_lag.value", "category_smooth.value"
    inputs = {"a": left, "b": right} if op == "panel_formula" else {"left": left, "right": right}
    parameters = {"expression": "a-b"} if op == "panel_formula" else {"operator": "gt"}
    # Even a disconnected draft branch must not erase nominal member identity.
    definition.nodes.append(Step(id="invalid", label="跨篮子位置混算", op=op, inputs=inputs, parameters=parameters))
    with pytest.raises(ValidationError, match="不同成员篮子"):
        GraphRuntime(build_catalog()).prepare(definition)


def test_different_nominal_baskets_can_combine_after_separate_shared_cross_means(monkeypatch):
    import backend.timing_research.graph as graph
    from backend.timing_research.catalog import build_catalog
    runtime = graph.GraphRuntime(build_catalog())
    prepared = runtime.prepare(two_basket_definition())
    market = np.array([[2., 4., 6., 8.], [4., 6., 8., 10.]])
    category = np.array([[2., 6., np.nan, 9.], [2., 6., 8., 9.]])
    market.setflags(write=False)
    category.setflags(write=False)
    signatures = {name: plan.compiled_signatures for name, plan in prepared.formulas.items()}
    monkeypatch.setattr(graph, "compile_numba_series_plan", lambda *_: pytest.fail("Request compiled a new plan"))
    channels = runtime.evaluate(prepared, SimpleNamespace(dates=np.arange(4, dtype=np.int64)),
                                {"market": market, "category": category})
    assert prepared.panel_groups == {"market.value": "market", "category.value": "category"}
    np.testing.assert_allclose(channels["market_mean.value"], panel.cross_mean_kernel(market), equal_nan=True)
    np.testing.assert_allclose(channels["category_mean.value"], panel.cross_mean_kernel(category), equal_nan=True)
    np.testing.assert_allclose(channels["spread.value"], [1., -1., np.nan, 0.], equal_nan=True)
    np.testing.assert_array_equal(channels["entry.value"], [1, 0, -1, 0])
    assert np.shares_memory(channels["market.value"], market)
    assert np.shares_memory(channels["category.value"], category)
    assert signatures == {name: plan.compiled_signatures for name, plan in prepared.formulas.items()}


def test_panel_group_identity_is_snapshotted_and_part_of_prepare_cache_token():
    from custom_indicators.errors import ValidationError
    runtime, original = basket_graph()
    definition = original.definition.model_copy(deep=True)
    assert runtime.prepare(definition) is original
    definition.nodes[0].parameters["group"] = "category"
    changed = runtime.prepare(definition)
    assert changed is not original and changed.digest != original.digest
    assert {original.panel_groups[ref] for ref in ("basket.value", "smooth.value", "above.value")} == {"market"}
    assert {changed.panel_groups[ref] for ref in ("basket.value", "smooth.value", "above.value")} == {"category"}
    assert original.definition.nodes[0].parameters["group"] == "market"
    definition.nodes[0].parameters["group"] = "market"
    assert changed.definition.nodes[0].parameters["group"] == "category"
    assert runtime.get(definition, original.digest) is original
    with pytest.raises(ValidationError, match="重新准备"):
        runtime.get(definition, changed.digest)


def test_service_panel_previews_keep_real_member_identity_order_and_unknown_values(tmp_path):
    from dataclasses import replace
    from backend.timing_research.contracts import Definition, RunRequest
    from backend.timing_research.service import TimingResearchService
    from test_timing_service import fixture_bars
    bars = fixture_bars()
    members = {"market": ["510500.SH", "510300.SH"], "category": ["159919.SZ", "159915.SZ"]}
    sources = {"510300.SH": bars}
    for index, code in enumerate(("510500.SH", "159919.SZ", "159915.SZ"), start=1):
        close = bars.close.copy() + index * 10.
        close[400 + index] = np.nan
        close.setflags(write=False)
        sources[code] = replace(bars, close=close, lineage={**bars.lineage, "source_hash": code})
    nodes = []
    for group in members:
        nodes.extend([
            {"id": group, "label": f"{group}成员", "op": "basket_source", "parameters": {"group": group}},
            {"id": f"{group}_smooth", "label": f"{group}均线", "op": "panel_formula",
             "inputs": {"a": f"{group}.value"}, "parameters": {"expression": "rolling_mean(a,2,2)"}},
            {"id": f"{group}_above", "label": f"{group}均线上", "op": "panel_compare",
             "inputs": {"left": f"{group}.value", "right": f"{group}_smooth.value"}, "parameters": {"operator": "gt"}},
            {"id": f"{group}_breadth", "label": f"{group}占比", "op": "breadth", "inputs": {"value": f"{group}_above.value"}},
        ])
    nodes.append({"id": "entry", "label": "市场宽度领先", "op": "compare",
                  "inputs": {"left": "market_breadth.value", "right": "category_breadth.value"}, "parameters": {"operator": "ge"}})
    definition = Definition(name="真实成员预览", nodes=nodes, entry="entry.value")
    service = TimingResearchService(tmp_path, loader=lambda base, code, *args: sources[code])
    try:
        token = service.prepare(definition)["compile_token"]
        request = RunRequest(definition=definition, compile_token=token, targets=[{"product_id": "510300.SH"}],
                             start_date="2020-01-01", end_date="2021-12-31", holdout_start="2021-01-01", context_baskets=members)
        prepared = service.graph.get(definition, token)
        result, arrays = service._product(request, prepared, "510300.SH")
        rows = result["channels"]
        by_id = {row["id"]: row for row in rows}
        assert len(rows) == len(by_id)
        start = int(np.searchsorted(bars.dates, np.datetime64("2020-01-01").astype(np.int64)))
        for group, codes in members.items():
            matrix = arrays[f"basket_{group}"]
            assert not matrix.flags.writeable
            expected_means = np.full(matrix.shape, np.nan)
            expected_means[:, 1:] = (matrix[:, :-1] + matrix[:, 1:]) / 2
            expected_conditions = panel.panel_compare_kernel(matrix, expected_means, 0., 44)
            for ref in (f"{group}.value", f"{group}_smooth.value", f"{group}_above.value"):
                matching = [row for row in rows if row["id"].startswith(ref + "[")]
                assert [row["id"] for row in matching] == [f"{ref}[{code}]" for code in codes]
                assert ref not in by_id  # No fabricated single-line panel preview.
            for index, code in enumerate(codes):
                source_preview = by_id[f"{group}.value[{code}]"]
                assert source_preview["label"] == f"{group}成员 · {code}"
                assert source_preview["type"] == "series"
                assert source_preview["values"] == [None if not np.isfinite(value) else float(value) for value in sources[code].close[start:]]
                np.testing.assert_allclose(matrix[index], sources[code].close, equal_nan=True)
                condition_preview = by_id[f"{group}_above.value[{code}]"]
                assert condition_preview["type"] == "condition"
                assert condition_preview["values"] == [None if value == -1 else float(value) for value in expected_conditions[index, start:]]
                assert len(condition_preview["values"]) == len(result["curve"])
    finally:
        service.close()

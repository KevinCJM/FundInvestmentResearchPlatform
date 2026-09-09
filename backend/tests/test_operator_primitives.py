"""Contracts for primitive lowering, shared fitted state and symbolic authoring."""
from __future__ import annotations

import math
import numpy as np
import pytest

from cal_indicators.typed_dsl import TypedIndicatorRuntime, compose_typed_expression
from cal_indicators.typed_numba_plan import compile_numba_batch_plan
from cal_indicators.typed_operators import get_typed_operator_catalog
from cal_indicators.operator_lowering import COMPOSITE_OPERATOR_IDS
from cal_indicators.primitive_access import value_at_kernel
from cal_indicators.regression_state import linear_fit_pair_kernel, linear_fit_time_kernel
from custom_indicators.typed_service import infer_expression, typed_product_meta
from custom_indicators.variable_registry import variable_types, normalize_variable_latex


def plan(expression):
    return compose_typed_expression(normalize_variable_latex(expression), variable_types=variable_types("single_product"))


def test_public_catalog_contains_primitives_not_business_wrappers():
    meta = typed_product_meta()
    operators = {item["name"]: item for item in meta["operators"]}
    assert not COMPOSITE_OPERATOR_IDS.intersection(operators)
    assert not {"date_at", "interval_depth", "drawdown_analysis"}.intersection(operators)
    assert {"value_at", "last_drawdown_interval", "linear_fit", "fit_slope", "fit_intercept", "fit_residual_sum_squares", "fit_total_sum_squares", "fit_observation_count"} <= operators.keys()
    for name, item in operators.items():
        assert item["label"] != name
        assert item["njit_supported"]
    assert operators["linear_fit"]["output_shape"] == "record"
    assert operators["fit_slope"]["parameters"][0]["allowed_shapes"] == ["record"]


@pytest.mark.parametrize("expression,forbidden", [
    ("total_return(returns)", "total_return"),
    ("annualized_return(returns,252)", "annualized_return"),
    ("linear_slope(adjusted_nav)", "linear_slope"),
    ("linear_intercept(adjusted_nav)", "linear_intercept"),
    ("linear_r_squared(adjusted_nav)", "linear_r_squared"),
    ("regression_standard_error(adjusted_nav)", "regression_standard_error"),
])
def test_saved_spellings_expand_to_transparent_dag(expression, forbidden):
    compiled = plan(expression)
    assert forbidden not in compiled.python_expression
    assert forbidden not in {node.operator_id for node in compiled.nodes}
    rebuilt = plan(compiled.python_expression)
    assert rebuilt.expression_hash == compiled.expression_hash


@pytest.mark.parametrize("expression,status", [("1 / std(returns)", 2), ("require_positive(0)", 3), ("sqrt(-1)", 3)])
def test_shared_execution_preserves_scalar_error_and_other_root(expression, status):
    plans = (plan(expression), plan("mean(adjusted_nav)"))
    batch = compile_numba_batch_plan(plans, ({}, {}), ("adjusted_nav",))
    values = np.array([[1., 1., 1.]], dtype=np.float64)
    output = np.empty((1, 2), dtype=np.float64)
    statuses = np.empty((1, 2), dtype=np.int16)
    signatures = batch.serial_dispatcher.signatures[:]
    batch.compute(values, np.array([0], dtype=np.int64), np.array([3], dtype=np.int64),
                  np.array([2.]), output, statuses, parallel=False)
    assert statuses.tolist() == [[status, 0]]
    assert np.isnan(output[0, 0]) and output[0, 1] == 1.
    assert batch.serial_dispatcher.signatures == signatures


@pytest.mark.parametrize("operator", ["covariance", "correlation"])
def test_pairwise_math_notation_includes_both_inputs(operator):
    from cal_indicators.typed_latex import render_python_expression_latex
    latex = render_python_expression_latex(f"{operator}(returns, log_returns)", {"returns": "x", "log_returns": "y"})
    assert "x,y" in latex
    assert operator not in latex


def test_function_and_infix_share_one_primitive_node():
    compiled = plan("add(mean(returns), 1) + (mean(returns) + 1)")
    adds = [node for node in compiled.nodes if node.operator_id == "add"]
    assert len(adds) == 2
    assert adds[-1].inputs[0] == adds[-1].inputs[1]


@pytest.mark.parametrize("position", [-1., .5, 3., np.nan, np.inf])
def test_value_at_never_truncates_wraps_or_fills(position):
    assert math.isnan(value_at_kernel(np.array([2., 4., 6.]), position))


def test_value_at_preserves_measure_and_requires_an_index():
    nav = plan("value_at(adjusted_nav, 1)")
    date = plan("value_at(observation_dates, 1)")
    assert nav.output_type.semantic_dimension == "adjusted_nav"
    assert nav.output_type.price_basis == "adjusted_nav"
    assert date.output_type.semantic_dimension == "date"
    with pytest.raises(ValueError):
        plan("value_at(adjusted_nav, mean(returns))")
    with pytest.raises(ValueError):
        plan("value_at(observation_dates, 1) + 1")
    assert math.isnan(value_at_kernel(np.array([], dtype=np.float64), 0.))


@pytest.mark.parametrize("implicit", [False, True])
def test_fit_matches_independent_lstsq_without_new_signatures(implicit):
    x = np.arange(23, dtype=np.float64) if implicit else np.linspace(-2., 5., 23)
    y = 0.17 + 1.8 * x + np.sin(x) * 0.13
    kernel = linear_fit_time_kernel if implicit else linear_fit_pair_kernel
    before = tuple(kernel.signatures)
    actual = np.array(kernel(y) if implicit else kernel(x, y))
    design = np.column_stack((x, np.ones(x.size)))
    beta, alpha = np.linalg.lstsq(design, y, rcond=None)[0]
    sse = np.sum((y - (alpha + beta * x)) ** 2)
    sst = np.sum((y - np.mean(y)) ** 2)
    np.testing.assert_allclose(actual, (beta, alpha, sse, sst, len(y)), rtol=1e-11, atol=1e-12)
    assert tuple(kernel.signatures) == before
    assert kernel.nopython_signatures


@pytest.mark.parametrize("x,y", [
    ([], []), ([1.], [1.]), ([1., 1.], [1., 2.]), ([1., 2.], [1.]),
    ([1., np.inf], [1., 2.]), ([1., 2.], [np.nan, 2.]),
])
def test_invalid_fit_inputs_fail_closed(x, y):
    with pytest.raises(ValueError):
        linear_fit_pair_kernel(np.array(x), np.array(y))


def test_independent_regression_metrics_fit_once_and_isolate_missing_statistics():
    expressions = ("linear_slope(adjusted_nav)", "linear_intercept(adjusted_nav)", "linear_r_squared(adjusted_nav)", "regression_standard_error(adjusted_nav)")
    plans = tuple(plan(source) for source in expressions)
    compiled = compile_numba_batch_plan(plans, ({},) * len(plans), ("adjusted_nav",))
    audit = compiled.metadata()
    assert audit["operator_call_sites"]["linear_fit"] == 1
    assert not COMPOSITE_OPERATOR_IDS.intersection(audit["operator_call_sites"])
    assert audit["eliminated_node_count"] > 0
    assert audit["python_fallback"] == 0
    before = (tuple(compiled.serial_dispatcher.signatures), tuple(compiled.parallel_dispatcher.signatures))
    for values in (np.array([1., 2., 1.5, 3.]), np.array([2., 2., 2.]), np.array([1., 2.])):
        output = np.full((1, 4), np.nan)
        statuses = np.full((1, 4), -1, dtype=np.int16)
        compiled.compute(np.ascontiguousarray([values]), np.array([0], dtype=np.int64), np.array([values.size], dtype=np.int64), np.array([3.]), output, statuses, parallel=False)
        assert np.all(statuses[0, :2] == 0)
        if np.ptp(values) == 0:
            assert statuses[0, 2] != 0
        if values.size == 2:
            assert statuses[0, 3] != 0
        if values.size > 2 and np.ptp(values) > 0:
            x = np.arange(values.size)
            beta, alpha = np.linalg.lstsq(np.column_stack((x, np.ones(x.size))), values, rcond=None)[0]
            residual = np.sum((values - (alpha + beta * x)) ** 2)
            expected = [beta, alpha, 1-residual/np.sum((values-values.mean())**2), np.sqrt(residual/(values.size-2))]
            np.testing.assert_allclose(output[0], expected, atol=1e-12)
    assert before == (tuple(compiled.serial_dispatcher.signatures), tuple(compiled.parallel_dispatcher.signatures))


@pytest.mark.parametrize("expression,symbol", [
    ("fit_slope(linear_fit(adjusted_nav))", r"\widehat{\beta}"),
    ("fit_intercept(linear_fit(adjusted_nav))", r"\widehat{\alpha}"),
    ("fit_residual_sum_squares(linear_fit(adjusted_nav))", r"\mathrm{SSE}"),
    ("value_at(observation_dates, interval_start(last_drawdown_interval(drawdown_series(adjusted_nav))))", r"\mathcal{I}"),
])
def test_formulas_echo_math_not_function_names(expression, symbol):
    result = infer_expression(expression, "single_product")
    assert symbol in result["display_latex"]
    for name in ("linear_fit", "fit_slope", "fit_intercept", "fit_residual_sum_squares", "value_at", "last_drawdown_interval", "drawdown_series", "interval_start"):
        assert name not in result["display_latex"]
    assert result["editable_latex"]
    assert plan(result["editable_latex"]).expression_hash == plan(result["expression"]).expression_hash


@pytest.mark.parametrize("periods", [0, -1])
def test_annualization_expansion_retains_positive_period_requirement(periods):
    runtime = TypedIndicatorRuntime.from_plan(plan(f"annualized_return(returns,{periods})"))
    with pytest.raises(ValueError):
        runtime.compute({"returns": np.array([.01, .02, .03])})

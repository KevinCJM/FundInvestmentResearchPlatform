"""Whole-interval rolling DAG execution, not leaf-reducer substitution."""
from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
import pytest

from cal_indicators.typed_dsl import compose_typed_expression, compose_typed_series_bundle, TypedDslError
from cal_indicators.typed_numba_plan import compile_numba_series_plan
from cal_indicators.typed_operators import get_typed_operator_registry
from cal_indicators.typed_types import ValueType
from cal_indicators.rolling_scope import (
    analyze_interval, compile_scope, interval_capability, _rolling_loop_source,
)
from custom_indicators.variable_registry import variable_types


def build(body, window="3", extra=None):
    return compose_typed_series_bundle({"value": f"rolling_apply({body}, {window})"},
        variable_types={**variable_types("single_product", "2.4.0"), **(extra or {})})


def evaluate(plan, values, *, readonly=False):
    compiled = compile_numba_series_plan(plan)
    size = next(len(value) for value in values.values() if isinstance(value, np.ndarray))
    context = {"observation_dates": np.arange(size, dtype=np.float64) + 20_000,
               "annual_risk_free_rate_decimal": 0.015, "periods_per_year": 252.0,
               "risk_free_rate_per_observation": 1.015 ** (1 / 252) - 1,
               "observation_count": float(size - 1), "window_elapsed_days": float(size - 1),
               "risk_free_return_window": 1.015 ** ((size - 1) / 365) - 1, **values}
    if readonly:
        context = {key: value.view() if isinstance(value, np.ndarray) else value for key, value in context.items()}
        for value in context.values():
            if isinstance(value, np.ndarray):
                value.flags.writeable = False
    return compiled.compute(tuple(context[name] for name in compiled.context_names))[0]


@pytest.mark.parametrize("body", [
    "mean(returns)", "std(returns, 1)", "min_value(returns)", "max_value(returns)",
    "product(returns + 1) - 1", "quantile(returns, 0.05)",
    "-mean_where(returns, less_equal(returns, quantile(returns, 0.05)))",
    "-min_value(drawdown_series(adjusted_nav))",
    "(mean(returns)-risk_free_rate_per_observation)/std(returns,1)*sqrt(periods_per_year)",
    "(product(returns+1)**(periods_per_year/observation_count)-1)/(-min_value(drawdown_series(adjusted_nav)))",
    "fit_slope(linear_fit(market_close, market_high))",
    "correlation(market_close, market_high)",
    "0.5 * std(returns,1) - 0.5 * min_value(drawdown_series(adjusted_nav))",
])
def test_interval_capability_follows_entire_graph(body):
    capability = interval_capability(body, variable_types=variable_types("single_product", "2.4.0"))
    assert capability["supported"], capability
    assert capability["state_policy"] == "reset_at_window_start"
    build(body)


@pytest.mark.parametrize("body,code", [
    ("returns + 1", "OUTPUT_CONTRACT_MISMATCH"),
    ("5", "ROLLING_TIME_INPUT_REQUIRED"),
    ("last(returns)", "ROLLING_AGGREGATION_REQUIRED"),
    ("length(returns)", "ROLLING_AGGREGATION_REQUIRED"),
    ("mean(recursive_smooth(market_close, 3, 50))", "ROLLING_INTERVAL_POLICY_REQUIRED"),
    ("mean(rolling_mean(market_close, 3))", "ROLLING_INTERMEDIATE_UNSUPPORTED"),
])
def test_shape_alone_does_not_authorize_rolling(body, code):
    result = interval_capability(body, variable_types=variable_types("single_product", "2.4.0"))
    assert not result["supported"]
    assert result["code"] == code


def test_unknown_primitive_semantics_fail_closed():
    plan = compose_typed_expression("mean(returns)")
    registry = dict(get_typed_operator_registry())
    registry["mean"] = replace(registry["mean"], interval_policy=None)
    with pytest.raises(TypedDslError, match="区间闭合"):
        analyze_interval(plan.nodes, plan.root_id, registry)


@pytest.mark.parametrize("body,reference", [
    ("mean(market_close)", np.mean),
    ("std(market_close, 1)", lambda x: np.std(x, ddof=1)),
    ("variance(market_close, 1)", lambda x: np.var(x, ddof=1)),
    ("min_value(market_close)", np.min),
    ("max_value(market_close)", np.max),
    ("quantile(market_close, 0.25)", lambda x: np.quantile(x, 0.25)),
    ("mean_where(market_close, less_equal(market_close, quantile(market_close, 0.25)))", lambda x: np.mean(x[x <= np.quantile(x, 0.25)])),
])
def test_existing_scalar_algorithms_need_no_new_rolling_kernel(body, reference):
    values = np.array([10., 8., 12., 9., 16., 15., 22.])
    actual = evaluate(build(body), {"market_close": values})
    expected = np.r_[np.nan, np.nan, [reference(values[i - 2:i + 1]) for i in range(2, len(values))]]
    np.testing.assert_allclose(actual, expected, equal_nan=True, rtol=1e-12)


def test_maximum_drawdown_resets_peak_at_each_window_start():
    values = np.array([100., 110., 55., 60., 65., 70., 75.])
    plan = build("-min_value(drawdown_series(adjusted_nav))")
    actual = evaluate(plan, {"adjusted_nav": values})
    expected = np.r_[np.nan, np.nan, [-np.min((v := values[i - 2:i + 1]) / np.maximum.accumulate(v) - 1) for i in range(2, len(values))]]
    np.testing.assert_allclose(actual, expected, equal_nan=True)
    assert actual[-1] == 0.0
    changed = values.copy()
    changed[0] = 1_000_000
    assert evaluate(plan, {"adjusted_nav": changed})[-1] == actual[-1]
    # The scalar path must not be executed first on the full history.
    compiled = compile_numba_series_plan(plan)
    body_root = next(node for node in plan.nodes if node.operator_id == "rolling_apply").inputs[0]
    assert f"    n{body_root} =" not in compiled.source


def test_return_and_level_boundaries_and_local_annualization_match_calmar():
    nav = np.array([100., 105., 98., 106., 102., 108., 104., 110.])
    returns = np.r_[np.nan, nav[1:] / nav[:-1] - 1]
    body = "(product(returns+1)**(periods_per_year/observation_count)-1)/(-min_value(drawdown_series(adjusted_nav)))"
    actual = evaluate(build(body), {"adjusted_nav": nav, "returns": returns})
    expected = np.full(len(nav), np.nan)
    for right in range(3, len(nav)):
        local = nav[right - 3:right + 1]
        drawdown = -np.min(local / np.maximum.accumulate(local) - 1)
        expected[right] = ((local[-1] / local[0]) ** (252 / 3) - 1) / drawdown
    np.testing.assert_allclose(actual, expected, equal_nan=True, rtol=1e-11)


@pytest.mark.parametrize("context_name", ["window_elapsed_days", "risk_free_return_window"])
def test_window_context_days_and_risk_free_return_are_rebuilt(context_name):
    x = np.arange(1., 8.)
    dates = np.array([20000., 20001., 20004., 20005., 20006., 20010., 20011.])
    plan = build(f"mean(market_close) / mean(market_close) * {context_name}")
    actual = evaluate(plan, {"market_close": x, "observation_dates": dates})
    days = dates[2:] - dates[:-2]
    values = days if context_name == "window_elapsed_days" else 1.015 ** (days / 365) - 1
    expected = np.r_[np.nan, np.nan, values]
    np.testing.assert_allclose(actual, expected, equal_nan=True)


def test_multiple_inputs_and_future_perturbation():
    x = np.array([1., 2., 3., 5., 4., 6., 7.])
    y = np.array([3., 1., 2., 4., 2., 5., 6.])
    plan = build("correlation(market_close, market_high)")
    actual = evaluate(plan, {"market_close": x, "market_high": y})
    expected = np.r_[np.nan, np.nan, [np.corrcoef(x[i - 2:i + 1], y[i - 2:i + 1])[0, 1] for i in range(2, len(x))]]
    np.testing.assert_allclose(actual, expected, equal_nan=True)
    y[5:] *= -100
    perturbed = evaluate(plan, {"market_close": x, "market_high": y})
    np.testing.assert_allclose(actual[:5], perturbed[:5], equal_nan=True)


def test_missing_input_affects_only_windows_that_contain_it():
    x = np.array([1., 2., np.nan, 4., 5., 6., np.inf, 8., 9., 10.])
    actual = evaluate(build("mean(market_close)"), {"market_close": x})
    assert np.isfinite(actual).nonzero()[0].tolist() == [5, 9]
    assert actual[5] == 5
    assert actual[9] == 9


def test_runtime_window_is_one_frozen_signature_and_readonly_inputs_are_unchanged():
    plan = build("std(market_close,1)", "window_size", {"window_size": ValueType.scalar(semantic_dimension="count")})
    compiled = compile_numba_series_plan(plan)
    signatures = tuple(compiled.dispatcher.signatures)
    x = np.arange(1., 21.)
    before = x.copy()
    for width in (3., 5., 10.):
        result = evaluate(plan, {"market_close": x, "window_size": width}, readonly=True)
        assert result[-1] == pytest.approx(np.std(x[-int(width):], ddof=1))
    np.testing.assert_array_equal(x, before)
    assert tuple(compiled.dispatcher.signatures) == signatures
    assert not compiled.dispatcher._can_compile
    assert compiled.metadata()["request_time_compilation"] == 0


def test_generated_window_arguments_share_original_memory():
    plan = build("mean(market_close)")
    scope = next(node for node in plan.nodes if node.operator_id == "rolling_apply")
    capability = analyze_interval(plan.nodes, scope.inputs[0], get_typed_operator_registry())
    x = np.arange(8., dtype=np.float64)
    observed = []
    def spy(values):
        observed.append(np.shares_memory(values, x))
        return float(values.mean())
    namespace = {"np": np, "math": math, "interval_body": spy}
    # Execute the exact compiler-owned slicing source with an instrumented
    # delegate; production compiles that same source to fixed-signature NJIT.
    exec(_rolling_loop_source(capability.variables, capability), namespace)
    result = namespace["rolling_scope"](x, 3., np.arange(8.) + 20_000, 0.)
    assert observed and all(observed)
    np.testing.assert_allclose(result, evaluate(plan, {"market_close": x}), equal_nan=True)


@pytest.mark.parametrize("window", [0., -1., 2.5, np.inf, 5001.])
def test_invalid_window_fails_before_execution(window):
    plan = build("mean(market_close)", "window_size", {"window_size": ValueType.scalar(semantic_dimension="count")})
    with pytest.raises(ValueError, match="INVALID_PARAMETER"):
        evaluate(plan, {"market_close": np.arange(8.), "window_size": window})


def test_alignment_and_resource_guards():
    with pytest.raises(ValueError, match="ALIGNMENT"):
        evaluate(build("correlation(market_close,market_high)"), {"market_close": np.arange(8.), "market_high": np.arange(7.)})
    with pytest.raises(ValueError, match="DATE_AXIS"):
        evaluate(build("mean(market_close)"), {"market_close": np.arange(8.), "observation_dates": np.ones(8)})
    body = "mean(market_close) + std(market_close,1) + quantile(market_close,0.05)"
    with pytest.raises(ValueError, match="BUDGET"):
        evaluate(build(body, "5000"), {"market_close": np.arange(5000.)})

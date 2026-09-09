"""Independent numerical and one-call contract tests for named scalar ports."""
from __future__ import annotations

import numpy as np
import pytest

from cal_indicators.multi_output import DRAWDOWN_PORTS
from cal_indicators.multi_output_kernels import drawdown_analysis_kernel
from cal_indicators.typed_dsl import TypedDslError, TypedIndicatorRuntime, compose_typed_scalar_bundle
from cal_indicators.typed_scalar_bundle import compile_scalar_bundle
from custom_indicators.variable_registry import variable_types


@pytest.mark.parametrize("values,expected", [
    ([100, 90, 80, 90, 100], [0.2, 2, 2, 4]),
    ([100, 90, 80, 90], [0.2, 2, np.nan, 3]),
    ([100, 101, 102], [0, 0, 0, 0]),
    ([100, 100, 100], [0, 0, 0, 0]),
    ([100], [0, 0, 0, 0]),
    ([100, 100, 80, 100], [0.2, 1, 1, 2]),
    ([100, 80, 80, 100], [0.2, 1, 2, 3]),
    ([100, 80, 100, 90, 90, 90, 100], [0.2, 1, 1, 4]),
    ([100, 80, 100, 80, 100], [0.2, 1, 1, 2]),
    ([100, 90, 100, 110, 70, 110], [1 - 70/110, 1, 1, 2]),
])
def test_drawdown_kernel_contract(values, expected):
    nav = np.asarray(values, dtype=np.float64)
    original = nav.copy()
    signatures = tuple(drawdown_analysis_kernel.signatures)
    np.testing.assert_allclose(drawdown_analysis_kernel(nav), expected, atol=1e-14, equal_nan=True)
    nav.flags.writeable = False
    np.testing.assert_allclose(drawdown_analysis_kernel(nav), expected, atol=1e-14, equal_nan=True)
    np.testing.assert_array_equal(nav, original)
    assert tuple(drawdown_analysis_kernel.signatures) == signatures


@pytest.mark.parametrize("values", [[], [np.nan], [100, np.nan, 90], [100, np.inf], [100, 0], [-1, 1]])
def test_invalid_nav_fails_closed(values):
    assert np.isnan(drawdown_analysis_kernel(np.asarray(values, dtype=np.float64))).all()


def reference(nav):
    peaks = np.maximum.accumulate(nav)
    depths = 1 - nav / peaks
    trough = int(np.argmax(depths))
    depth = depths[trough]
    if depth == 0:
        return np.zeros(4)
    peak = int(np.flatnonzero(nav[:trough] == peaks[trough])[-1])
    recovered = np.flatnonzero(nav[trough+1:] >= peaks[trough])
    recovery = float(recovered[0] + 1) if recovered.size else np.nan
    longest, anchor = 0, 0
    for index in range(1, nav.size):
        if nav[index] >= peaks[index-1]:
            if index > anchor + 1:
                longest = max(longest, index-anchor)
            anchor = index
        else:
            longest = max(longest, index-anchor)
    return np.array([depth, trough-peak, recovery, longest])


def test_randomized_reference_and_scale_invariance():
    rng = np.random.default_rng(542)
    for _ in range(80):
        nav = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, 150)))
        np.testing.assert_allclose(drawdown_analysis_kernel(nav), reference(nav), rtol=1e-12, atol=1e-13, equal_nan=True)
        np.testing.assert_allclose(drawdown_analysis_kernel(nav * 7), reference(nav), rtol=1e-12, atol=1e-13, equal_nan=True)


def test_four_ports_are_one_njit_call_and_can_feed_other_math():
    expressions = {port.id: f"drawdown_analysis(adjusted_nav).{port.id}" for port in DRAWDOWN_PORTS}
    expressions["double"] = "drawdown_analysis(adjusted_nav).max_drawdown * 2"
    plan = compose_typed_scalar_bundle(expressions, variable_types=variable_types("single_product"))
    records = [node for node in plan.nodes if node.inferred_type.kind == "record"]
    assert len(records) == 1
    compiled = compile_scalar_bundle(plan)
    assert compiled.metadata()["multi_output_call_sites"] == 1
    assert compiled.source.count(f"n{records[0].node_id} = k") == 1
    signatures = tuple(compiled.dispatcher.signatures)
    nav = np.array([100., 90., 80., 100.])
    values, status = compiled.compute((nav,), np.ones(5, dtype=np.uint8))
    np.testing.assert_allclose(values, [0.2, 2, 1, 3, 0.4])
    assert np.all(status == 0)
    assert tuple(compiled.dispatcher.signatures) == signatures
    scalar = TypedIndicatorRuntime.from_expression(expressions["double"], variable_types=variable_types("single_product"))
    assert scalar.compute({"adjusted_nav": nav}) == pytest.approx(0.4)


def test_unrecovered_status_is_distinct_from_invalid_input():
    from cal_indicators.multi_output import STATUS_OUTPUT_UNAVAILABLE
    from cal_indicators.typed_numba_kernels import STATUS_NON_FINITE_RESULT
    plan = compose_typed_scalar_bundle({"recovery": "drawdown_analysis(adjusted_nav).recovery_periods"}, variable_types=variable_types("single_product"))
    compiled = compile_scalar_bundle(plan)
    enabled = np.ones(1, dtype=np.uint8)
    values, status = compiled.compute((np.array([100., 90., 80.]),), enabled)
    assert np.isnan(values[0]) and status[0] == STATUS_OUTPUT_UNAVAILABLE
    values, status = compiled.compute((np.array([100., -1., 80.]),), enabled)
    assert np.isnan(values[0]) and status[0] == STATUS_NON_FINITE_RESULT


@pytest.mark.parametrize("expression", [
    "drawdown_analysis(adjusted_nav).__class__", "adjusted_nav.shape",
    "drawdown_analysis(adjusted_nav).missing", "mean(drawdown_analysis(adjusted_nav))",
    "drawdown_analysis(adjusted_nav) + 1", "drawdown_analysis(returns).max_drawdown",
])
def test_record_is_not_a_vector_or_python_object(expression):
    with pytest.raises(TypedDslError):
        TypedIndicatorRuntime.from_expression(expression, variable_types=variable_types("single_product"))

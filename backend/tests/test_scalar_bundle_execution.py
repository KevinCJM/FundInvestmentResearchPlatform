from __future__ import annotations

import numpy as np
import pytest

from cal_indicators.typed_dsl import compose_typed_scalar_bundle, compose_typed_series_bundle, TypedDslError
from cal_indicators.typed_scalar_bundle import compile_scalar_bundle
from custom_indicators.typed_service import variable_types


def test_shared_nodes_and_njit_partial_failure():
    plan = compose_typed_scalar_bundle({
        "mean": "mean(returns)",
        "double": "mean(returns) * 2",
        "bad": "mean(returns) / 0",
    }, variable_types=variable_types("single_product"))
    assert sum(node.operator_id == "mean" for node in plan.nodes) == 1
    compiled = compile_scalar_bundle(plan)
    assert compile_scalar_bundle(plan) is compiled
    signatures = tuple(compiled.dispatcher.signatures)
    values, statuses = compiled.compute((np.array([0.01, 0.02, 0.03]),), np.ones(3, dtype=np.uint8))
    np.testing.assert_allclose(values[:2], [0.02, 0.04])
    assert np.isnan(values[2]) and statuses[2] != 0
    assert tuple(compiled.dispatcher.signatures) == signatures
    assert compiled.metadata()["python_fallback"] == 0
    assert compiled.dispatcher.nopython_signatures


def test_bundle_root_types_and_security_remain_strict():
    with pytest.raises(TypedDslError):
        compose_typed_scalar_bundle({"bad": "returns"})
    with pytest.raises(TypedDslError):
        compose_typed_series_bundle({"bad": "mean(returns)"})
    with pytest.raises(TypedDslError):
        compose_typed_scalar_bundle({"bad": "returns[0]"})
    with pytest.raises(TypedDslError):
        compose_typed_scalar_bundle({"bad": "__import__('os').getcwd()"})


def test_disabled_output_does_not_execute_its_branch():
    plan = compose_typed_scalar_bundle({"value": "2", "bad": "1 / 0"})
    compiled = compile_scalar_bundle(plan)
    values, statuses = compiled.compute((), np.array([1, 0], dtype=np.uint8))
    assert values[0] == 2.0 and np.isnan(values[1])
    assert statuses[0] == 0 and statuses[1] != 0

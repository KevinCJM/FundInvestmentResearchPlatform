from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from numba.core.registry import CPUDispatcher

from cal_indicators.typed_dsl import TypedIndicatorRuntime, compose_typed_expression
from cal_indicators.typed_numba_kernels import (
    AXIS_REDUCTION_OPCODES,
    BASIC_OPCODES,
    CANONICAL_OPERATOR_IDS,
    COMPARISON_OPCODES,
    UNARY_OPCODES,
    axis_reduce_asset,
    axis_reduce_time_fixed,
    binary_1d,
    comparison_1d_right_scalar,
    correlation_2d,
    drawdown_series_1d,
    get_numba_kernel_registry,
    kernel_registry_status,
    new_high_mask_1d,
    solve_2d,
    unary_scalar,
    warm_numba_kernel_registry,
)
from cal_indicators.typed_numba_plan import (
    _risk_free_scalars,
    _risk_free_scalars_kernel,
    compile_numba_batch_plan,
    compile_numba_plan,
    get_cached_numba_batch_plan,
)
from cal_indicators.typed_operators import (
    get_typed_operator_catalog,
    get_typed_operator_registry,
)
from custom_indicators.variable_registry import variable_catalog
from custom_indicators.service import CustomIndicatorService


def test_current_registry_has_complete_fixed_signature_njit_coverage() -> None:
    status = warm_numba_kernel_registry()
    registry = get_numba_kernel_registry()
    canonical = {
        spec.operator_id for spec in get_typed_operator_registry().values()
    }
    frozen_v22 = {
        spec.operator_id for spec in get_typed_operator_registry("2.2.0").values()
    }
    public = get_typed_operator_catalog()["operators"]

    from cal_indicators.operator_lowering import (
        COMPILER_FUSED_OPERATOR_IDS,
        COMPOSITE_OPERATOR_IDS,
        ROLLING_COMPAT_OPERATOR_IDS,
    )
    assert len(set(CANONICAL_OPERATOR_IDS)) == len(CANONICAL_OPERATOR_IDS)
    assert {"linear_fit", "value_at", "fit_slope", "fit_intercept"} <= canonical
    assert set(CANONICAL_OPERATOR_IDS) == set(registry)
    assert canonical == set(registry) | COMPILER_FUSED_OPERATOR_IDS
    assert COMPILER_FUSED_OPERATOR_IDS == {"rolling_window", "rolling_apply"}
    assert {"sum", "std", "drawdown_series", "total_return"} <= frozen_v22
    assert {
        "rolling_mean",
        "rolling_std",
        "rolling_min",
        "rolling_max",
        "recursive_smooth",
        "divide_or_default",
    }.isdisjoint(frozen_v22)
    assert {item["id"] for item in public} == (
        canonical - COMPOSITE_OPERATOR_IDS - ROLLING_COMPAT_OPERATOR_IDS
    )
    rolling_window = next(item for item in public if item["id"] == "rolling_window")
    assert rolling_window["execution_lane"] == "compiler_fused_no_materialization"
    assert rolling_window["njit_supported"] is True
    assert "drawdown_analysis" not in frozen_v22
    assert status["operator_coverage"] == f"{len(registry)}/{len(registry)}"
    assert status["warmed"] is True
    assert status["python_fallback"] == 0
    assert status["python_operator_calls"] == 0
    for spec in registry.values():
        assert spec.kernel_version == "2.2.0"
        assert spec.execution_lane in {"numba", "numba_blas"}
        assert spec.compiled_signatures
        assert all(isinstance(kernel, CPUDispatcher) for kernel in spec.serial_kernels)


def test_all_context_variables_are_numeric_float64_contracts() -> None:
    variables = variable_catalog()

    assert len({variable["id"] for variable in variables}) == len(variables)
    assert next(variable for variable in variables if variable["id"] == "observation_dates")["semantic"] == "date"
    assert any(variable["id"] == "portfolio_returns" for variable in variables)
    assert all(variable["dtype"] == "float64" for variable in variables)
    assert all(
        variable["structural_type"]
        in {"scalar", "series", "vector", "matrix"}
        for variable in variables
    )


def test_numba_operator_families_match_numpy_reference() -> None:
    values = np.ascontiguousarray([1.0, 2.0, 4.0, 8.0], dtype=np.float64)
    matrix = np.ascontiguousarray(
        [[1.0, 3.0], [2.0, 5.0], [4.0, 8.0]], dtype=np.float64
    )

    levels = np.ascontiguousarray([1.10, 1.10, 1.05, 1.20, 1.20])
    assert np.allclose(
        drawdown_series_1d(levels),
        np.asarray([0.0, 0.0, 1.05 / 1.10 - 1.0, 0.0, 0.0]),
    )
    assert np.array_equal(
        new_high_mask_1d(levels),
        np.asarray([1, 0, 0, 1, 0], dtype=np.uint8),
    )

    assert np.allclose(
        binary_1d(BASIC_OPCODES["multiply"], values, values), values * values
    )
    assert np.array_equal(
        comparison_1d_right_scalar(
            COMPARISON_OPCODES["greater_than"], values, 2.0
        ),
        (values > 2.0).astype(np.uint8),
    )
    assert np.allclose(
        axis_reduce_time_fixed(AXIS_REDUCTION_OPCODES["mean"], matrix),
        np.mean(matrix, axis=0),
    )
    assert np.allclose(
        axis_reduce_asset(AXIS_REDUCTION_OPCODES["sum"], matrix),
        np.sum(matrix, axis=1),
    )
    assert np.allclose(correlation_2d(matrix), np.corrcoef(matrix, rowvar=False))
    coefficients = np.ascontiguousarray([[3.0, 1.0], [1.0, 2.0]])
    rhs = np.ascontiguousarray([9.0, 8.0])
    assert np.allclose(solve_2d(coefficients, rhs), np.linalg.solve(coefficients, rhs))
    assert unary_scalar(UNARY_OPCODES["normal_ppf"], 0.5) == pytest.approx(0.0)
    assert unary_scalar(UNARY_OPCODES["normal_ppf"], 0.975) == pytest.approx(
        1.959963984540054, abs=5e-9
    )
    with pytest.raises(ValueError, match="SINGULAR_MATRIX"):
        solve_2d(
            np.ascontiguousarray([[1.0, 2.0], [2.0, 4.0]]),
            np.ascontiguousarray([1.0, 2.0]),
        )


@pytest.mark.parametrize("parallel", [False, True])
def test_fused_batch_plan_matches_single_formula_njit_runtime(parallel: bool) -> None:
    formulas = (
        "product(returns + 1) - 1",
        "mean_where(returns, greater_than(returns, 0))",
        "std(returns, 1)",
    )
    plans = tuple(compose_typed_expression(formula) for formula in formulas)
    definitions = tuple(
        {"annual_risk_free_rate_percent": 1.5} for _ in formulas
    )
    batch = compile_numba_batch_plan(plans, definitions, ("adjusted_nav",))
    first_nav = np.asarray([1.0, 1.01, 1.005, 1.03, 1.04], dtype=np.float64)
    second_nav = np.asarray(
        [2.0, 2.02, 2.04, 2.01, 2.08, 2.10], dtype=np.float64
    )
    values = np.ascontiguousarray([np.concatenate((first_nav, second_nav))])
    starts = np.ascontiguousarray([0, first_nav.size], dtype=np.int64)
    ends = np.ascontiguousarray(
        [first_nav.size, first_nav.size + second_nav.size], dtype=np.int64
    )
    elapsed = np.ascontiguousarray([4.0, 5.0], dtype=np.float64)
    output = np.full((2, len(plans)), np.nan, dtype=np.float64)
    statuses = np.full((2, len(plans)), -1, dtype=np.int16)

    batch.compute(
        values,
        starts,
        ends,
        elapsed,
        output,
        statuses,
        parallel=parallel,
    )

    assert np.all(statuses == 0)
    for row, nav in enumerate((first_nav, second_nav)):
        returns = np.ascontiguousarray(nav[1:] / nav[:-1] - 1.0)
        for metric, plan in enumerate(plans):
            expected = TypedIndicatorRuntime.from_plan(plan).compute(
                {"returns": returns}
            )
            assert math.isclose(
                output[row, metric], expected, rel_tol=1e-10, abs_tol=1e-12
            )
    assert batch.serial_dispatcher.signatures
    assert batch.parallel_dispatcher.signatures
    assert batch.metadata()["execution_backend"] == "numba_njit_fixed_signature"
    assert batch.metadata()["nopython"] is True
    assert batch.metadata()["python_fallback"] == 0


def test_warmed_runtime_never_adds_a_request_signature() -> None:
    plan = compose_typed_expression("mean(returns)")
    compiled = compile_numba_plan(plan)
    runtime = TypedIndicatorRuntime.from_warmed_plan(plan)
    before = tuple(compiled.dispatcher.signatures)

    first = runtime.compute(
        {"returns": np.ascontiguousarray([0.01, -0.02, 0.03], dtype=np.float64)}
    )
    second = runtime.compute(
        {"returns": np.asarray([0.02, 0.01, -0.01], dtype=np.float64)}
    )

    assert first == pytest.approx(0.02 / 3.0)
    assert second == pytest.approx(0.02 / 3.0)
    assert tuple(compiled.dispatcher.signatures) == before
    audit = runtime.trace_payload()
    assert audit["execution_backend"] == "numba_njit_fixed_signature"
    assert audit["nopython"] is True
    assert audit["python_fallback"] == 0
    assert all(audit["kernel_signatures"].values())


def test_runtime_does_not_call_python_operator_registry() -> None:
    runtime = TypedIndicatorRuntime.from_expression("mean(returns)")
    runtime.registry = {
        name: object() for name in runtime.registry
    }  # the compiled plan must be independent of Python callables

    value = runtime.compute(
        {"returns": np.ascontiguousarray([0.01, -0.02, 0.03])}
    )

    assert value == pytest.approx((0.01 - 0.02 + 0.03) / 3.0)
    assert runtime.trace_payload()["python_operator_calls"] == 0
    assert runtime.trace_payload()["python_fallback"] == 0
    count = len(CANONICAL_OPERATOR_IDS)
    assert kernel_registry_status()["operator_coverage"] == f"{count}/{count}"


def test_batch_plan_cache_lookup_never_compiles_on_miss() -> None:
    plan = compose_typed_expression("mean(returns)")
    definitions = ({"annual_risk_free_rate_percent": 12.3456789},)
    columns = ("adjusted_nav",)

    assert get_cached_numba_batch_plan((plan,), definitions, columns) is None
    compiled = compile_numba_batch_plan((plan,), definitions, columns)
    assert get_cached_numba_batch_plan((plan,), definitions, columns) is compiled


@pytest.mark.parametrize(
    "annual_percent",
    (-100.0, -2.5, 0.0, 1.5, 12.3456789, 100.0),
)
def test_risk_free_scalars_use_one_fixed_nopython_signature_without_growth(
    annual_percent: float,
) -> None:
    before = tuple(_risk_free_scalars_kernel.nopython_signatures)

    annual, per_observation = _risk_free_scalars(
        {"annual_risk_free_rate_percent": annual_percent}
    )

    expected_annual = annual_percent / 100.0
    expected_per_observation = max(0.0, 1.0 + expected_annual) ** (1.0 / 252.0) - 1.0
    assert annual == pytest.approx(expected_annual)
    assert per_observation == pytest.approx(expected_per_observation)
    assert len(before) == 1
    assert tuple(_risk_free_scalars_kernel.nopython_signatures) == before


def test_batch_plan_audit_includes_risk_free_conversion_kernel() -> None:
    plan = compose_typed_expression("annual_risk_free_rate_decimal")
    compiled = compile_numba_batch_plan(
        (plan,),
        ({"annual_risk_free_rate_percent": 1.5},),
        ("adjusted_nav",),
    )

    signatures = compiled.metadata()["kernel_signatures"]
    assert len(signatures["risk_free_scalars"]) == 1
    assert signatures["risk_free_scalars"] == [
        str(signature) for signature in _risk_free_scalars_kernel.nopython_signatures
    ]


def test_numba_v3_migration_archives_plans_once_and_clears_run_cache(
    tmp_path: Path,
) -> None:
    old_plan_payload = {
        "schema_version": 1,
        "items": [
            {
                "current": {"id": "plan-old", "revision": 1},
                "history": [],
            }
        ],
    }
    indicator_payload = {"schema_version": 1, "items": []}
    (tmp_path / "evaluation_plans.json").write_text(
        json.dumps(old_plan_payload), encoding="utf-8"
    )
    indicator_path = tmp_path / "custom_indicators.json"
    indicator_path.write_text(json.dumps(indicator_payload), encoding="utf-8")
    run_cache = tmp_path / ".evaluation_run_cache"
    run_cache.mkdir()
    (run_cache / "deadbeef.json").write_text("{}", encoding="utf-8")
    (run_cache / "deadbeef.parquet").write_bytes(b"old-result")

    first = CustomIndicatorService(tmp_path, tmp_path)
    archives = list(
        (tmp_path / "archive").glob(
            "evaluation_plans.pre-typed-numba-3.*.json"
        )
    )
    active = json.loads(
        (tmp_path / "evaluation_plans.json").read_text(encoding="utf-8")
    )

    assert first.migration["applied"] is True
    assert len(archives) == 1
    assert json.loads(archives[0].read_text(encoding="utf-8"))["items"] == (
        old_plan_payload["items"]
    )
    assert active["items"] == []
    assert active["migration"]["marker"] == "typed-numba-3"
    assert not list(run_cache.iterdir())
    assert json.loads(indicator_path.read_text(encoding="utf-8")) == indicator_payload

    second = CustomIndicatorService(tmp_path, tmp_path)

    assert second.migration["applied"] is False
    assert len(
        list(
            (tmp_path / "archive").glob(
                "evaluation_plans.pre-typed-numba-3.*.json"
            )
        )
    ) == 1

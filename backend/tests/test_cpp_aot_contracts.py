"""Integration with the installed native wheel, without starting the service."""

from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from cal_indicators.cpp_aot import CppIndicatorBatchPlan
from compute_policy import validate_execution_audit, ComputePolicyError
from custom_indicators.errors import ValidationError
from custom_indicators.series_parameters import (
    bind_parameter_input,
    inspect_parameter_inputs,
)
from custom_indicators.variable_registry import (
    CONTEXT_SCHEMA_VERSION, DATA_CONTRACT_VERSION, VARIABLE_REGISTRY_VERSION,
)


AdaptiveScheduler = pytest.importorskip("calmetrics_engine").AdaptiveScheduler


def definition(expression="mean(returns)", **extra):
    return {"expression": expression, "dsl_version": "2.4.0", **extra}


def execute(plan, nav, starts=(0,), ends=None, parameters=None, **inputs):
    nav = np.asarray(nav, dtype=np.float64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        return plan.execute(
            scheduler,
            {"adjusted_nav": nav, **inputs},
            np.array(starts, dtype=np.int64),
            np.array(ends or [len(nav)], dtype=np.int64),
            parameters=parameters,
        )


def test_nav_and_returns_have_distinct_interval_axes_and_exact_arithmetic():
    plan = CppIndicatorBatchPlan(
        [
            definition(expr)
            for expr in (
                "length(adjusted_nav)",
                "length(returns)",
                "mean(returns)",
                "mean(log_returns)",
                "observation_count",
            )
        ]
    )
    nav = np.array([1.23, 1.33, 1.17, 100.0, 110.0])
    nav.flags.writeable = False
    result = execute(plan, nav, (0, 3), (3, 5))
    for row, (start, end) in enumerate(((0, 3), (3, 5))):
        ratio = nav[start + 1 : end] / nav[start : end - 1]
        np.testing.assert_allclose(
            result.values[row],
            [
                end - start,
                end - start - 1,
                np.mean(ratio - 1),
                np.mean(np.log(ratio)),
                end - start - 1,
            ],
            rtol=1e-14,
        )
    assert result.values[0, 2] == np.mean(nav[1:3] / nav[:2] - 1)
    np.testing.assert_array_equal(result.statuses, np.zeros((2, 5), dtype=np.int16))
    assert validate_execution_audit(result.audit)["njit_required"] is False


@pytest.mark.parametrize("version", ["2.0.0", "2.1.0", "2.2.0", "2.3.0", "2.4.0"])
def test_persisted_dsl_and_aliases_keep_their_version(version):
    item = definition(
        "mean(returns)", dsl_version=version, operator_registry_version=version
    )
    before = deepcopy(item)
    plan = CppIndicatorBatchPlan([item])
    result = execute(plan, [100.0, 110.0, 121.0])
    assert result.values[0, 0] == pytest.approx(0.1)
    assert item == before
    assert json.loads(result.audit["source_contracts"][0])["registry"] == version
    assert (
        plan.fingerprint != CppIndicatorBatchPlan([definition()]).fingerprint
        or version == "2.4.0"
    )


@pytest.mark.parametrize("version", ["2.0.0", "2.1.0", "2.2.0", "2.3.0", "2.4.0"])
def test_source_contract_defaults_match_platform_and_explicit_versions(version):
    from custom_indicators.service import CustomIndicatorService

    item = definition(dsl_version=version)
    before = deepcopy(item)
    normalized = CustomIndicatorService._normalize_definition({"name": "contract", **item})
    fields = ("data_contract_version", "variable_registry_version", "context_schema_version")
    explicit = {**item, **{field: normalized[field] for field in fields}}
    plan = CppIndicatorBatchPlan([item])
    assert plan.fingerprint == CppIndicatorBatchPlan([explicit]).fingerprint
    result = execute(plan, [100.0, 110.0, 121.0])
    contract = json.loads(result.audit["source_contracts"][0])
    for key, field in zip(("data", "variables", "context"), fields):
        assert contract[key] == normalized[field]
    assert result.values[0, 0] == pytest.approx(0.1)
    assert item == before


@pytest.mark.parametrize("version", ["2.0.0", "2.1.0", "2.2.0", "2.3.0", "2.4.0"])
@pytest.mark.parametrize("field,modern,legacy", [
    ("data_contract_version", DATA_CONTRACT_VERSION, "adjusted-nav-v1"),
    ("variable_registry_version", VARIABLE_REGISTRY_VERSION, "legacy-typed-v2.0"),
    ("context_schema_version", CONTEXT_SCHEMA_VERSION, "multi-asset-v1"),
])
@pytest.mark.parametrize("invalid", ["unknown", "other_version"])
def test_source_contract_versions_reject_unknown_and_mismatched(version, field, modern, legacy, invalid):
    value = "future-contract" if invalid == "unknown" else (modern if version == "2.0.0" else legacy)
    with pytest.raises(ValidationError) as error:
        CppIndicatorBatchPlan([definition(dsl_version=version, **{field: value})])
    assert error.value.code == f"UNSUPPORTED_{field.upper()}"
    assert error.value.field == field


def test_latex_and_unsupported_or_mismatched_versions():
    plan = CppIndicatorBatchPlan(
        [definition(r"\operatorname{mean}\left(\mathbf{r}\right)")]
    )
    assert execute(plan, [100.0, 110.0]).values[0, 0] == pytest.approx(0.1)
    for dsl, registry in [("1.0.0", "1.0.0"), ("2.0.0", "2.1.0")]:
        with pytest.raises(ValueError):
            CppIndicatorBatchPlan(
                [definition(dsl_version=dsl, operator_registry_version=registry)]
            )


def test_per_metric_parameter_slots_do_not_recompile_or_mutate_definitions():
    draft = definition("quantile(returns, 0.9)")
    opened = bind_parameter_input(
        draft, candidate_id=inspect_parameter_inputs(draft)["candidates"][0]["id"]
    )
    before = deepcopy(opened)
    name = opened["parameter_schema"][0]["id"]
    plan = CppIndicatorBatchPlan([opened, opened])
    fingerprint = plan.fingerprint
    nav = np.array([100.0, 102.0, 101.0, 105.0])
    returns = nav[1:] / nav[:-1] - 1
    for values in ((0.2, 0.9), (0.8, 0.1)):
        result = execute(plan, nav, parameters=[{name: values[0]}, {name: values[1]}])
        np.testing.assert_allclose(result.values[0], np.quantile(returns, values))
        assert result.audit["request_time_compilation"] == 0
        assert plan.fingerprint == fingerprint
    assert opened == before
    with pytest.raises(ValidationError):
        plan.parameters([{"unopened": 1}, None])
    with pytest.raises(ValueError):
        plan.parameters([None])


def test_per_metric_risk_free_inputs_and_dates():
    plan = CppIndicatorBatchPlan(
        [
            definition("risk_free_return_window", annual_risk_free_rate_percent=rate)
            for rate in (1.5, 5.0)
        ]
    )
    result = execute(
        plan, [100.0, 102.0, 103.0], observation_dates=np.array([0.0, 100.0, 365.0])
    )
    np.testing.assert_allclose(result.values, [[0.015, 0.05]])


def test_error_statuses_empty_short_and_retained_snapshots():
    plan = CppIndicatorBatchPlan(
        [definition("1/std(returns,1)"), definition("mean(returns)")]
    )
    result = execute(plan, [100.0, 100.0, 100.0], (0, 0, 0), (0, 1, 3))
    np.testing.assert_array_equal(result.statuses, [[1, 1], [1, 1], [2, 0]])
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        prepared = plan.prepare(
            scheduler,
            {"adjusted_nav": np.array([100.0, 110.0, 121.0])},
            np.array([0], np.int64),
            np.array([3], np.int64),
        )
        retained = prepared.run_snapshot()
        fresh = prepared.run_audit()
        borrowed = prepared.run()
        assert np.shares_memory(borrowed, fresh.values)
        assert not np.shares_memory(retained.values, fresh.values)
        validate_execution_audit(retained.audit)
    assert retained.values[0, 1] == pytest.approx(0.1)


@pytest.mark.parametrize("method", ["run", "run_audit", "run_snapshot"])
@pytest.mark.parametrize("invalid_proof", ["missing", "contradictory"])
def test_prepared_execution_rejects_invalid_native_proof(method, invalid_proof):
    plan = CppIndicatorBatchPlan([definition()])
    inputs = {"adjusted_nav": np.array([100.0, 110.0, 121.0])}
    starts, ends = np.array([0], np.int64), np.array([3], np.int64)
    calls = []
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        native = scheduler.prepare_execution(
            plan.graph, inputs, starts, ends, parameters=plan.parameters()
        )

        def corrupt_result(run):
            calls.append(1)
            result = run()
            audit = dict(result.audit)
            if invalid_proof == "missing":
                audit.pop("engine_build_id")
            else:
                audit["backend"] = "numba_njit_fixed_signature"
            return SimpleNamespace(
                values=result.values, statuses=result.statuses, audit=audit
            )

        # Inject only faulty package metadata; computation and ownership remain native.
        faulty_native = SimpleNamespace(
            run=native.run,
            run_audit=lambda: corrupt_result(native.run_audit),
            run_snapshot=lambda: corrupt_result(native.run_snapshot),
        )
        provider = SimpleNamespace(prepare_execution=lambda *a, **kw: faulty_native)
        prepared = plan.prepare(provider, inputs, starts, ends)
        with pytest.raises(ComputePolicyError):
            getattr(prepared, method)()
        assert len(calls) == 1


@pytest.mark.parametrize("method", ["execute", "run", "run_audit", "run_snapshot"])
def test_native_execution_rejects_valid_credentials_from_a_different_graph(method):
    expected = CppIndicatorBatchPlan([definition("mean(returns)")])
    other = CppIndicatorBatchPlan([definition("std(returns,1)")])
    assert expected.fingerprint != other.fingerprint
    inputs = {"adjusted_nav": np.array([100.0, 110.0, 121.0])}
    starts, ends = np.array([0], np.int64), np.array([3], np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        wrong_result = other.execute(scheduler, inputs, starts, ends)
        validate_execution_audit(wrong_result.audit)  # valid credentials, wrong graph
        wrong_prepared = other.prepare(scheduler, inputs, starts, ends)
        faulty_scheduler = SimpleNamespace(
            execute=lambda *a, **kw: wrong_result,
            prepare_execution=lambda *a, **kw: wrong_prepared,
        )
        with pytest.raises(ComputePolicyError):
            if method == "execute":
                expected.execute(faulty_scheduler, inputs, starts, ends)
            else:
                getattr(expected.prepare(faulty_scheduler, inputs, starts, ends), method)()


def test_actual_native_proof_fails_closed_if_any_required_field_is_missing():
    audit = execute(CppIndicatorBatchPlan([definition()]), [100.0, 110.0]).audit
    for key in (
        "engine_build_id",
        "engine_version",
        "plan_fingerprint",
        "native_aot",
        "audit_schema",
        "operator_registry_version",
        "typed_ir_version",
        "python_fallback",
        "python_operator_calls",
        "python_worker_callbacks",
        "request_time_compilation",
        "cpu_tokens",
        "cpu_budget",
        "result_lifetime",
    ):
        invalid = {k: v for k, v in audit.items() if k != key}
        with pytest.raises(ComputePolicyError):
            validate_execution_audit(invalid)
    for override in (
        {"backend": "numba_njit_fixed_signature"},
        {"cpu_tokens": 10000},
        {"python_fallback": True},
        {"python_worker_callbacks": 1},
        {"request_time_compilation": 1},
        {"native_aot": False},
    ):
        with pytest.raises(ComputePolicyError):
            validate_execution_audit({**audit, **override})


def test_native_adapter_matches_actual_njit_batch_values_and_statuses():
    from cal_indicators.typed_dsl import compose_typed_expression
    from cal_indicators.typed_numba_plan import (
        compile_numba_batch_plan,
        batch_parameter_vector,
    )
    from custom_indicators.variable_registry import variable_types

    definitions = tuple(
        definition(expr, annual_risk_free_rate_percent=1.5)
        for expr in (
            "mean(returns)",
            "std(returns,1)",
            "1/std(returns,1)",
            "length(adjusted_nav)",
            "observation_count",
            "risk_free_rate_per_observation",
        )
    )
    plans = tuple(
        compose_typed_expression(
            item["expression"],
            variable_types=variable_types("single_product"),
            dsl_version=item["dsl_version"],
        )
        for item in definitions
    )
    njit = compile_numba_batch_plan(plans, definitions, ("adjusted_nav",))
    nav = np.array([100.0, 100.0, 100.0, 100.0, 102.0, 101.0, 105.0])
    starts = np.array([0, 0, 0, 0, 3], np.int64)
    ends = np.array([0, 1, 2, 3, 7], np.int64)
    expected = np.empty((5, len(definitions)), np.float64)
    statuses = np.empty(expected.shape, np.int16)
    njit.compute(
        nav[None, :],
        starts,
        ends,
        (ends - starts).astype(np.float64),
        expected,
        statuses,
        batch_parameter_vector(definitions),
        parallel=False,
    )
    actual = execute(
        CppIndicatorBatchPlan(definitions), nav, tuple(starts), tuple(ends)
    )
    np.testing.assert_allclose(
        actual.values, expected, rtol=1e-13, atol=1e-15, equal_nan=True
    )
    np.testing.assert_array_equal(actual.statuses, statuses)

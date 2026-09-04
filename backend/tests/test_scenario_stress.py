from __future__ import annotations

import copy
import hashlib
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numba.core.registry import CPUDispatcher

from custom_indicators.errors import ConflictError, ValidationError
from historical_regimes.service import HistoricalRegimeService
from scenario_stress.contracts import meta_contract, normalize_definition
from scenario_stress.engine import execute
from scenario_stress.numba_kernels import (
    SCENARIO_STRESS_NUMBA_KERNELS,
    coverage_weights_kernel,
    deterministic_paths_kernel,
    empirical_regime_projection_kernel,
    portfolio_weight_summary_kernel,
    prepare_historical_state_samples_kernel,
    scenario_stress_numba_status,
    warm_scenario_numba_kernels,
)
from scenario_stress.service import ScenarioStressService, _content_hash


def _template(method: str) -> dict:
    for item in meta_contract()["templates"]:
        if item["definition"]["method"] == method:
            return copy.deepcopy(item["definition"])
    raise AssertionError(method)


@pytest.fixture()
def service(tmp_path: Path) -> ScenarioStressService:
    return ScenarioStressService(tmp_path, tmp_path)


def test_scenario_numba_kernels_are_fixed_signature_warmed_and_have_no_fallback() -> None:
    status = warm_scenario_numba_kernels()
    assert status["fully_warmed"] is True
    assert status["kernel"] == "scenario_stress_numeric_core"
    assert status["engine"] == "numba_njit_fixed_signature"
    assert status["kernel_coverage"] == f"{len(SCENARIO_STRESS_NUMBA_KERNELS)}/{len(SCENARIO_STRESS_NUMBA_KERNELS)}"
    assert status["python_fallback"] == 0
    assert status["object_mode"] == 0
    assert status["request_time_compilation"] == 0
    assert status["optimized_third_party_model"] is None
    assert status["random_generation"]["provider"] == "numba.random"
    assert len(status["fingerprint"]) == 64
    for dispatcher in SCENARIO_STRESS_NUMBA_KERNELS:
        assert isinstance(dispatcher, CPUDispatcher)
        assert len(dispatcher.signatures) == 1
        assert all(compilation.objectmode is False for compilation in dispatcher.overloads.values())


def test_fixed_signature_rejects_wrong_dtype_without_runtime_specialization() -> None:
    warm_scenario_numba_kernels()
    signatures_before = tuple(coverage_weights_kernel.signatures)
    with pytest.raises(TypeError, match="No matching definition"):
        coverage_weights_kernel(
            np.ascontiguousarray([1.0], dtype=np.float32),
            np.ascontiguousarray([1], dtype=np.uint8),
            np.int64(0),
            np.float64(1.0),
        )
    assert tuple(coverage_weights_kernel.signatures) == signatures_before


def test_weight_summary_kernel_supports_long_short_weights_without_signature_growth() -> None:
    warm_scenario_numba_kernels()
    signatures_before = tuple(portfolio_weight_summary_kernel.signatures)
    net, gross, largest, status = portfolio_weight_summary_kernel(
        np.ascontiguousarray([1.2, -0.2], dtype=np.float64),
        np.float64(1.0),
        np.float64(1e-8),
        np.float64(2.0),
        np.float64(3.0),
    )
    assert net == pytest.approx(1.0)
    assert gross == pytest.approx(1.4)
    assert largest == pytest.approx(1.2)
    assert status == 0
    _, _, _, invalid_status = portfolio_weight_summary_kernel(
        np.ascontiguousarray([1.2, -0.1], dtype=np.float64),
        np.float64(1.0),
        np.float64(1e-8),
        np.float64(2.0),
        np.float64(3.0),
    )
    assert invalid_status & 4
    with pytest.raises(TypeError, match="No matching definition"):
        portfolio_weight_summary_kernel(
            np.ascontiguousarray([1.0], dtype=np.float32),
            np.float64(1.0),
            np.float64(1e-8),
            np.float64(2.0),
            np.float64(3.0),
        )
    assert tuple(portfolio_weight_summary_kernel.signatures) == signatures_before


def test_deterministic_numba_kernel_matches_controlled_reference_with_missing_value() -> None:
    output = deterministic_paths_kernel(
        np.ascontiguousarray([[0.10, -0.05], [0.02, np.nan]], dtype=np.float64),
        np.ascontiguousarray([0.6, 0.4], dtype=np.float64),
        np.int64(1),
        np.float64(0.5),
        np.float64(1.0),
    )
    returns, contributions, ratios, statuses, missing, nav, drawdown, summary = output[:8]
    assert returns.tolist() == pytest.approx([0.04, 0.02])
    assert contributions[0].tolist() == pytest.approx([0.06, -0.02])
    assert contributions[1, 0] == pytest.approx(0.02)
    assert np.isnan(contributions[1, 1])
    assert ratios.tolist() == pytest.approx([1.0, 0.6])
    assert statuses.tolist() == [0, 0]
    assert missing[1].tolist() == [0, 1]
    assert nav.tolist() == pytest.approx([1.04, 1.0608])
    assert drawdown.tolist() == pytest.approx([0.0, 0.0])
    assert summary[1] == pytest.approx(0.0608)


@pytest.mark.parametrize(
    "method",
    ["historical_replay", "factor_path", "monte_carlo", "regime_conditioned", "reverse_stress"],
)
def test_each_method_uses_warmed_kernels_without_signature_drift_and_is_reproducible(method: str) -> None:
    warm_scenario_numba_kernels()
    signatures_before = {
        dispatcher.py_func.__name__: tuple(dispatcher.signatures)
        for dispatcher in SCENARIO_STRESS_NUMBA_KERNELS
    }
    normalized = normalize_definition(_template(method))
    first = execute(normalized)
    second = execute(normalized)
    assert first == second
    audit = first["compute_audit"]
    assert audit["fully_warmed"] is True
    assert audit["python_fallback"] == 0
    assert audit["object_mode"] == 0
    assert {
        dispatcher.py_func.__name__: tuple(dispatcher.signatures)
        for dispatcher in SCENARIO_STRESS_NUMBA_KERNELS
    } == signatures_before


@pytest.mark.parametrize("method", ["monte_carlo", "regime_conditioned"])
def test_seeded_simulation_reproduces_at_5000_paths_without_fallback(method: str) -> None:
    warm_scenario_numba_kernels()
    definition = _template(method)
    if method == "monte_carlo":
        definition["scenario"]["path_count"] = 5_000
    else:
        definition["scenario"]["transition"]["path_count"] = 5_000
    normalized = normalize_definition(definition)
    first = execute(normalized)
    second = execute(normalized)
    assert first == second
    assert first["compute_audit"]["python_fallback"] == 0
    assert first["results"][0]["distribution"]["path_count"] == 5_000


def test_probabilistic_methods_do_not_call_python_numpy_rng(monkeypatch: pytest.MonkeyPatch) -> None:
    warm_scenario_numba_kernels()

    def fail_python_rng(*_args, **_kwargs):
        raise AssertionError("Python NumPy RNG must not run in a scenario request")

    monkeypatch.setattr(np.random, "default_rng", fail_python_rng)
    for method in ("monte_carlo", "regime_conditioned"):
        output = execute(normalize_definition(_template(method)))
        assert output["results"][0]["distribution"]["path_count"] >= 100


def test_meta_and_persisted_run_expose_same_numba_compute_audit(service: ScenarioStressService) -> None:
    warmed = warm_scenario_numba_kernels()
    meta_audit = service.meta()["compute_audit"]
    run_audit = service.run(_template("factor_path"))["compute_audit"]
    for audit in (meta_audit, run_audit):
        assert audit["kernel"] == warmed["kernel"]
        assert audit["engine"] == warmed["engine"]
        assert audit["signatures"] == warmed["signatures"]
        assert audit["fingerprint"] == warmed["fingerprint"]
        assert audit["python_fallback"] == 0


def test_meta_has_five_executable_p0_p1_templates() -> None:
    meta = meta_contract()
    templates = {item["definition"]["method"]: item["definition"] for item in meta["templates"]}
    assert set(templates) == {
        "historical_replay",
        "factor_path",
        "monte_carlo",
        "regime_conditioned",
        "reverse_stress",
    }
    for definition in templates.values():
        output = execute(normalize_definition(definition))
        assert output["results"]
    assert templates["historical_replay"]["mapping"]["response_space"] == "direct_simple_return"
    assert templates["regime_conditioned"]["mapping"]["response_space"] == "direct_simple_return"
    assert meta["limits"]["return_unit"] == "decimal"
    assert meta["mapping_contract"]["missing_policy"] == ["block", "degrade"]
    historical_contract = meta["historical_regime_distribution_contract"]
    assert historical_contract["required_content_locks"] == ["run_content_hash", "evaluation_artifact_checksum"]
    assert historical_contract["default_inline_policy"] == "forbid"


def test_definition_versions_conflicts_and_archive_are_preserved(service: ScenarioStressService) -> None:
    created = service.create_definition(_template("factor_path"))
    assert created["revision"] == 1
    draft = copy.deepcopy(created)
    draft["description"] = "第二版"
    updated = service.update_definition(created["id"], created["revision"], draft)
    assert updated["revision"] == 2
    assert service.get_definition(created["id"], 1)["description"] != "第二版"
    with pytest.raises(ConflictError):
        service.update_definition(created["id"], 1, draft)
    archived = service.archive_definition(created["id"], 2)
    assert archived["archived"] is True
    assert service.list_definitions() == []
    assert service.list_definitions(include_archived=True)[0]["revision"] == 3
    with pytest.raises(ValidationError) as error:
        service.run({"id": created["id"], "revision": 3})
    assert error.value.code == "SCENARIO_DEFINITION_ARCHIVED"


def test_factor_path_has_real_nav_drawdown_recovery_contribution_and_breach_time() -> None:
    definition = _template("factor_path")
    definition["horizon"] = 3
    definition["factors"] = [{"id": "shock", "label": "冲击"}]
    definition["mapping"]["factor_to_asset"] = {
        "cn_equity": {"shock": 1.0},
        "duration_bond": {"shock": 1.0},
        "gold": {"shock": 1.0},
    }
    definition["scenario"] = {
        "factor_path": [
            {"step": 1, "date": "2025-01-02", "shocks": {"shock": -0.2}},
            {"step": 2, "date": "2025-01-03", "shocks": {"shock": 0.3}},
            {"step": 3, "date": "2025-01-06", "shocks": {"shock": 0.01}},
        ]
    }
    definition["limits"] = [
        {"id": "dd", "label": "回撤", "metric": "max_drawdown", "operator": "gt", "threshold": 0.1}
    ]
    output = execute(normalize_definition(definition))
    result = output["results"][0]
    assert output["probabilistic"] is False
    assert "distribution" not in result
    assert "var_95" not in result["summary"]
    assert "loss_probability" not in result["summary"]
    assert result["path"][0]["nav"] == pytest.approx(0.8)
    assert result["summary"]["max_drawdown"] == pytest.approx(0.2)
    assert result["summary"]["recovery_step"] == 2
    assert result["summary"]["recovery_steps"] == 1
    assert result["limits"][0]["first_breach_step"] == 1
    assert result["limits"][0]["first_breach_date"] == "2025-01-02"
    assert result["summary"]["first_breach_step"] == 1
    assert sum(value for value in result["path"][0]["contributions"].values() if value is not None) == pytest.approx(-0.2)


def test_deterministic_method_blocks_probability_limits() -> None:
    definition = _template("factor_path")
    definition["limits"] = [
        {"id": "fake", "metric": "loss_probability", "operator": "gt", "threshold": 0.5}
    ]
    with pytest.raises(ValidationError) as error:
        normalize_definition(definition)
    assert error.value.code == "PSEUDO_PROBABILITY_LIMIT_BLOCKED"


def test_legacy_method_aliases_normalize_to_canonical_names() -> None:
    factor = _template("factor_path")
    factor["method"] = "deterministic_path"
    assert normalize_definition(factor)["method"] == "factor_path"
    regime = _template("regime_conditioned")
    regime["method"] = "regime_transition"
    assert normalize_definition(regime)["method"] == "regime_conditioned"


def test_factor_mapping_requires_explicit_method_specific_response_space() -> None:
    definition = _template("factor_path")
    definition["mapping"]["response_space"] = "log_return"
    with pytest.raises(ValidationError) as error:
        normalize_definition(definition)
    assert error.value.code == "MAPPING_RESPONSE_SPACE_MISMATCH"
    del definition["mapping"]["response_space"]
    with pytest.raises(ValidationError) as error:
        normalize_definition(definition)
    assert error.value.code == "MAPPING_RESPONSE_SPACE_REQUIRED"


def test_explicit_factor_path_validates_horizon_dates_and_order() -> None:
    definition = _template("factor_path")
    definition["horizon"] = 2
    definition["scenario"] = {
        "factor_path": [
            {"step": 1, "date": "2025-01-03", "shocks": {"growth": 0, "inflation": 0, "rate": 0}},
            {"step": 2, "date": "2025-01-02", "shocks": {"growth": 0, "inflation": 0, "rate": 0}},
        ]
    }
    with pytest.raises(ValidationError) as error:
        execute(normalize_definition(definition))
    assert error.value.code == "INVALID_FACTOR_PATH_DATE_ORDER"
    definition["scenario"]["factor_path"].pop()
    with pytest.raises(ValidationError) as error:
        execute(normalize_definition(definition))
    assert error.value.code == "INVALID_FACTOR_PATH"


def test_historical_replay_missing_return_is_not_zero_and_block_policy_fails() -> None:
    definition = _template("historical_replay")
    del definition["scenario"]["historical_returns"][0]["returns"]["gold"]
    with pytest.raises(ValidationError) as error:
        execute(normalize_definition(definition))
    assert error.value.code == "SCENARIO_MAPPING_INCOMPLETE"
    assert "gold" in error.value.diagnostics[0]["missing_assets"]


def test_historical_replay_degrades_with_null_contribution_and_disclosed_coverage() -> None:
    definition = _template("historical_replay")
    definition["mapping"] = {"missing_policy": "degrade", "minimum_coverage": 0.8, "factor_to_asset": {}}
    definition["scenario"]["historical_returns"][0]["returns"]["gold"] = None
    result = execute(normalize_definition(definition))["results"][0]
    assert result["coverage"]["status"] == "degraded"
    assert result["coverage"]["ratio"] == pytest.approx(0.85)
    assert result["path"][0]["asset_returns"]["gold"] is None
    assert result["path"][0]["contributions"]["gold"] is None
    assert result["contributions"]["by_asset"]["gold"] is None


def test_null_factor_beta_is_explicit_missing_and_can_follow_degrade_policy() -> None:
    definition = _template("factor_path")
    definition["mapping"]["missing_policy"] = "degrade"
    definition["mapping"]["minimum_coverage"] = 0.8
    definition["mapping"]["factor_to_asset"]["gold"]["growth"] = None
    result = execute(normalize_definition(definition))["results"][0]
    assert result["coverage"]["status"] == "degraded"
    assert result["coverage"]["ratio"] == pytest.approx(0.85)
    assert result["path"][0]["asset_returns"]["gold"] is None


@pytest.mark.parametrize("distribution", ["normal", "student_t"])
def test_seeded_monte_carlo_is_reproducible_and_has_real_tail_metrics(distribution: str) -> None:
    definition = _template("monte_carlo")
    definition["scenario"]["distribution"] = distribution
    definition["scenario"]["path_count"] = 600
    if distribution == "student_t":
        definition["scenario"]["df"] = 4.5
    normalized = normalize_definition(definition)
    first = execute(normalized)
    second = execute(normalized)
    assert first == second
    result = first["results"][0]
    assert first["probabilistic"] is True
    assert len(result["distribution"]["fan"]["quantiles"]["p05"]) == definition["horizon"] + 1
    assert result["distribution"]["fan"]["quantiles"]["p05"][-1] <= result["distribution"]["fan"]["quantiles"]["p95"][-1]
    assert result["summary"]["es_95"] >= result["summary"]["var_95"] >= 0
    assert 0 <= result["summary"]["loss_probability"] <= 1
    assert result["distribution"]["seed"] == definition["scenario"]["seed"]


def test_normal_and_student_t_use_distinct_seeded_paths() -> None:
    normal = _template("monte_carlo")
    normal["scenario"]["path_count"] = 500
    student = copy.deepcopy(normal)
    student["scenario"]["distribution"] = "student_t"
    student["scenario"]["df"] = 4
    normal_result = execute(normalize_definition(normal))["results"][0]["distribution"]
    student_result = execute(normalize_definition(student))["results"][0]["distribution"]
    assert normal_result["sample_paths"] != student_result["sample_paths"]


@pytest.mark.parametrize(
    ("correlation", "code"),
    [
        ([[1, 0], [0, 1]], "CORRELATION_DIMENSION_MISMATCH"),
        ([[1, 0.9, 0.9], [0.9, 1, -0.9], [0.9, -0.9, 1]], "CORRELATION_NOT_POSITIVE_SEMIDEFINITE"),
    ],
)
def test_monte_carlo_rejects_invalid_correlation(correlation: list[list[float]], code: str) -> None:
    definition = _template("monte_carlo")
    definition["scenario"]["correlation"] = correlation
    with pytest.raises(ValidationError) as error:
        execute(normalize_definition(definition))
    assert error.value.code == code


def test_monte_carlo_validates_path_budget_seed_and_student_df() -> None:
    definition = _template("monte_carlo")
    definition["horizon"] = 1200
    definition["scenario"]["path_count"] = 2000
    with pytest.raises(ValidationError) as error:
        execute(normalize_definition(definition))
    assert error.value.code == "SIMULATION_SIZE_EXCEEDED"
    definition = _template("monte_carlo")
    definition["scenario"].update({"distribution": "student_t", "df": 2, "path_count": 100})
    with pytest.raises(ValidationError) as error:
        execute(normalize_definition(definition))
    assert error.value.code == "INVALID_STUDENT_T_DF"


def test_monte_carlo_limits_total_factor_draw_memory_budget() -> None:
    definition = _template("monte_carlo")
    factors = [{"id": f"factor_{index}", "label": f"因子 {index}", "unit": "sigma"} for index in range(17)]
    factor_ids = [item["id"] for item in factors]
    definition["factors"] = factors
    definition["horizon"] = 500
    definition["scenario"].update(
        {
            "path_count": 1000,
            "factor_means": {factor_id: 0 for factor_id in factor_ids},
            "factor_volatilities": {factor_id: 0.1 for factor_id in factor_ids},
            "correlation": [[1.0 if left == right else 0.0 for right in range(17)] for left in range(17)],
        }
    )
    definition["mapping"]["factor_to_asset"] = {
        asset["id"]: {factor_id: 0.001 for factor_id in factor_ids}
        for asset in definition["assets"]
    }
    with pytest.raises(ValidationError) as error:
        execute(normalize_definition(definition))
    assert error.value.code == "FACTOR_SIMULATION_SIZE_EXCEEDED"


def test_regime_conditioned_is_seeded_and_reports_state_probabilities() -> None:
    definition = _template("regime_conditioned")
    definition["scenario"]["transition"]["path_count"] = 500
    first = execute(normalize_definition(definition))
    second = execute(normalize_definition(definition))
    assert first == second
    result = first["results"][0]
    assert result["distribution"]["sample_state_paths"]
    assert len(result["distribution"]["state_probabilities"]) == definition["horizon"]
    assert sum(result["distribution"]["state_probabilities"][0].values()) == pytest.approx(1)
    assert set(path[0] for path in result["distribution"]["sample_state_paths"]) == {"recovery"}


def test_regime_conditioned_rejects_invalid_or_deterministic_transition() -> None:
    definition = _template("regime_conditioned")
    definition["scenario"]["transition"]["matrix"] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    with pytest.raises(ValidationError) as error:
        execute(normalize_definition(definition))
    assert error.value.code == "DEGENERATE_TRANSITION"
    definition["scenario"]["transition"]["matrix"] = [[0.5, 0.5], [0.5, 0.5]]
    with pytest.raises(ValidationError) as error:
        execute(normalize_definition(definition))
    assert error.value.code == "TRANSITION_DIMENSION_MISMATCH"


def test_reverse_stress_returns_single_and_multi_factor_candidates() -> None:
    output = execute(normalize_definition(_template("reverse_stress")))
    result = output["results"][0]
    candidates = result["reverse_stress"]["candidates"]
    assert output["probabilistic"] is False
    assert "distribution" not in result
    assert len(candidates) >= 3
    assert {item["kind"] for item in candidates} >= {"single_factor", "multi_factor"}
    assert any(item["meets_target"] for item in candidates)
    assert all("factor_shocks" in item and "contributions" in item for item in candidates)


def test_reverse_stress_zero_sensitivity_returns_business_error_not_type_error() -> None:
    definition = _template("reverse_stress")
    definition["mapping"]["factor_to_asset"] = {
        asset["id"]: {factor["id"]: 0.0 for factor in definition["factors"]}
        for asset in definition["assets"]
    }
    with pytest.raises(ValidationError) as error:
        execute(normalize_definition(definition))
    assert error.value.code == "REVERSE_STRESS_NO_SENSITIVITY"
    assert error.value.field == "mapping.factor_to_asset"


def test_weight_and_factor_dimensions_fail_closed() -> None:
    definition = _template("factor_path")
    del definition["portfolios"][0]["weights"]["gold"]
    with pytest.raises(ValidationError) as error:
        normalize_definition(definition)
    assert error.value.code == "WEIGHT_DIMENSION_MISMATCH"
    definition = _template("factor_path")
    del definition["scenario"]["shocks"]["growth"]
    with pytest.raises(ValidationError) as error:
        execute(normalize_definition(definition))
    assert error.value.code == "FACTOR_SHOCK_DIMENSION_MISMATCH"


def test_saved_run_snapshot_publication_and_analytical_hash_are_immutable(service: ScenarioStressService) -> None:
    saved = service.create_definition(_template("factor_path"))
    run = service.run({"id": saved["id"], "revision": saved["revision"]})
    before_results = copy.deepcopy(run["results"])
    before_hash = run["content_hash"]
    publication = service.publish(run["id"], ["product_research", "taa"])
    assert len(publication["publications"]) == 2
    stored = service.get_run(run["id"])
    assert stored["results"] == before_results
    assert stored["content_hash"] == before_hash
    assert stored["immutable"] is True
    assert {item["usage"] for item in stored["application_bindings"]} == {"product_research", "taa"}


def test_publish_detects_stored_run_snapshot_tampering(service: ScenarioStressService) -> None:
    saved = service.create_definition(_template("factor_path"))
    run = service.run({"id": saved["id"], "revision": 1})
    with service.runs.store.locked():
        payload = service.runs.store.read_unlocked()
        payload["items"][0]["results"][0]["summary"]["terminal_return"] = 99
        service.runs.store.write_unlocked(payload)
    with pytest.raises(ValidationError) as error:
        service.publish(run["id"], "research_display")
    assert error.value.code == "SCENARIO_RUN_SNAPSHOT_TAMPERED"


def test_inline_trial_and_degraded_run_cannot_cross_publication_gate(service: ScenarioStressService) -> None:
    trial = service.run(_template("factor_path"))
    assert trial["governance"]["publish_eligible_usages"] == []
    with pytest.raises(ValidationError) as error:
        service.publish(trial["id"], "product_research")
    assert error.value.code == "UNVERSIONED_RUN_PUBLICATION_BLOCKED"

    definition = _template("factor_path")
    definition["mapping"]["missing_policy"] = "degrade"
    definition["mapping"]["minimum_coverage"] = 0.8
    del definition["mapping"]["factor_to_asset"]["gold"]
    saved = service.create_definition(definition)
    run = service.run({"id": saved["id"], "revision": 1})
    assert run["governance"]["publish_eligible_usages"] == ["research_display", "product_research"]
    service.publish(run["id"], "product_research")
    with pytest.raises(ValidationError) as error:
        service.publish(run["id"], "taa")
    assert error.value.code == "SCENARIO_PUBLICATION_BLOCKED"


def test_batch_run_is_one_immutable_multi_portfolio_snapshot(service: ScenarioStressService) -> None:
    definition = _template("factor_path")
    second = copy.deepcopy(definition["portfolios"][0])
    second.update({"id": "growth", "name": "进取组合", "label": "进取组合"})
    second["weights"] = {"cn_equity": 0.7, "duration_bond": 0.2, "gold": 0.1}
    definition["portfolios"].append(second)
    saved = service.create_definition(definition)
    run = service.batch_run({"id": saved["id"], "revision": 1})
    assert run["batch"] is True
    assert run["batch_size"] == 2
    assert {item["portfolio_id"] for item in run["results"]} == {"balanced", "growth"}
    assert len(service.list_runs(saved["id"])) == 1


def test_compare_preserves_null_for_non_comparable_probability_metrics(service: ScenarioStressService) -> None:
    deterministic = service.run(_template("factor_path"))
    stochastic = service.run(_template("monte_carlo"))
    comparison = service.compare([deterministic["id"], stochastic["id"]], deterministic["id"])
    stochastic_summary = next(item for item in comparison["runs"] if item["run_id"] == stochastic["id"])
    assert stochastic_summary["deltas_to_reference"]["balanced"]["loss_probability"] is None
    assert comparison["reference_run_id"] == deterministic["id"]


def _historical_definition() -> dict:
    start = date(2020, 1, 1)
    value = 100.0
    rows = []
    for index in range(60):
        change = 0.012 if index < 20 else -0.014 if index < 40 else 0.002
        value *= 1 + change
        rows.append(
            {
                "observation_date": (start + timedelta(days=index)).isoformat(),
                "available_at": (start + timedelta(days=index)).isoformat(),
                "value": value,
            }
        )
    return {
        "name": "历史状态引用测试",
        "target": {"kind": "inline", "series_id": "fixture", "frequency": "daily", "rows": rows},
        "features": {"transform": "identity", "filter": "ema", "window": 3, "slope_window": 2},
        "algorithm": {
            "family": "causal_filter",
            "parameters": {"bull_enter": 0.001, "bear_enter": -0.001, "confirmation": 1, "min_duration": 1},
        },
        "states": [
            {"id": "bull", "label": "牛市"},
            {"id": "sideways", "label": "震荡"},
            {"id": "bear", "label": "熊市"},
        ],
        "validation": {"walk_forward": False, "folds": 2},
        "usage_intent": "research_display",
    }


def test_regime_conditioned_can_reference_only_a_published_historical_snapshot(tmp_path: Path) -> None:
    historical = HistoricalRegimeService(tmp_path, tmp_path)
    historical_definition = historical.create_definition(_historical_definition())
    historical_run = historical.run({"id": historical_definition["id"], "revision": 1}, "realtime")
    publication = historical.publish(historical_run["id"], "research_display")["publication"]
    scenario_service = ScenarioStressService(tmp_path, tmp_path)
    definition = _template("regime_conditioned")
    definition["scenario"] = {
        "historical_run_ref": {"run_id": historical_run["id"], "publication_id": publication["id"]},
        "transition": {
            "states": [
                {"id": "bull", "asset_returns": {"cn_equity": 0.02, "duration_bond": -0.002, "gold": 0.003}},
                {"id": "sideways", "asset_returns": {"cn_equity": 0.001, "duration_bond": 0.002, "gold": 0.001}},
                {"id": "bear", "asset_returns": {"cn_equity": -0.02, "duration_bond": 0.01, "gold": 0.008}},
            ],
            "path_count": 200,
            "seed": 7,
        },
    }
    saved_scenario = scenario_service.create_definition(definition)
    run = scenario_service.run({"id": saved_scenario["id"], "revision": 1})
    source = run["data_snapshot"]["historical_regime_reference"]
    assert source["run_id"] == historical_run["id"]
    assert source["publication_id"] == publication["id"]
    assert source["run_content_hash"] == historical_run["content_hash"]
    assert source["causality_gate_passed"] is True
    assert run["governance"]["publish_eligible_usages"] == ["research_display"]

    taa_publication = historical.publish(historical_run["id"], "taa")["publication"]
    taa_definition = copy.deepcopy(definition)
    taa_definition["name"] = "TAA 历史状态引用"
    taa_definition["scenario"]["historical_run_ref"]["publication_id"] = taa_publication["id"]
    saved_taa_scenario = scenario_service.create_definition(taa_definition)
    taa_run = scenario_service.run({"id": saved_taa_scenario["id"], "revision": 1})
    assert set(taa_run["governance"]["publish_eligible_usages"]) == {"research_display", "taa", "risk_monitoring"}
    assert "portfolio_backtest" not in taa_run["governance"]["publish_eligible_usages"]

    definition["scenario"]["historical_run_ref"]["publication_id"] = "not-published"
    with pytest.raises(ValidationError) as error:
        scenario_service.run(definition)
    assert error.value.code == "HISTORICAL_RUN_NOT_PUBLISHED"


def test_regime_conditioned_hydrates_v2_series_to_infer_latest_initial_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_dir = tmp_path / "historical_regime_v2_artifacts"
    artifact_dir.mkdir()
    frame = pd.DataFrame(
        [
            {
                "observation_date": "2024-01-02",
                "state_id": "bull",
                "probabilities_json": '{"bull":1.0,"bear":0.0}',
                "features_json": "{}",
                "reasons_json": "[]",
            },
            {
                "observation_date": "2024-01-03",
                "state_id": "bear",
                "probabilities_json": '{"bull":0.0,"bear":1.0}',
                "features_json": "{}",
                "reasons_json": "[]",
            },
        ]
    )
    temporary = artifact_dir / "series.parquet"
    frame.to_parquet(temporary, index=False)
    digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
    final_path = artifact_dir / f"{digest}.parquet"
    temporary.replace(final_path)
    series_artifact = {
        "artifact_id": f"regime-series-sha256-{digest}",
        "checksum": f"sha256:{digest}",
        "format": "parquet",
        "row_count": 2,
    }
    analytical = {
        "schema_version": "2.0",
        "mode": "realtime",
        "transition": {
            "states": ["bull", "bear"],
            "probabilities": [[0.8, 0.2], [0.3, 0.7]],
        },
        "series_artifact": series_artifact,
        "causality": {
            "is_causal": True,
            "uses_future_data": False,
            "repaints": False,
        },
        "application_bindings": [],
    }
    content_hash = _content_hash(analytical)
    raw_run = {
        **analytical,
        "id": "regime-run-v2-artifact",
        "created_at": "2024-01-04T00:00:00Z",
        "immutable": True,
        "publications": [
            {
                "id": "publication-v2-artifact",
                "usage": "research_display",
                "run_id": "regime-run-v2-artifact",
                "run_content_hash": content_hash,
            }
        ],
        "content_hash": content_hash,
    }
    monkeypatch.setattr(
        HistoricalRegimeService,
        "get_run",
        lambda _self, _run_id: copy.deepcopy(raw_run),
    )

    definition = _template("regime_conditioned")
    definition["scenario"] = {
        "historical_run_ref": {
            "run_id": raw_run["id"],
            "publication_id": "publication-v2-artifact",
        },
        "transition": {
            "states": [
                {
                    "id": "bull",
                    "asset_returns": {
                        "cn_equity": 0.02,
                        "duration_bond": -0.002,
                        "gold": 0.003,
                    },
                },
                {
                    "id": "bear",
                    "asset_returns": {
                        "cn_equity": -0.02,
                        "duration_bond": 0.01,
                        "gold": 0.008,
                    },
                },
            ],
            "path_count": 200,
            "seed": 17,
        },
    }
    run = ScenarioStressService(tmp_path, tmp_path).run(definition)
    assert run["mapping_diagnostics"]["initial_state"] == "bear"


def test_retrospective_historical_reference_cannot_authorize_formal_downstream_usage(tmp_path: Path) -> None:
    historical = HistoricalRegimeService(tmp_path, tmp_path)
    historical_definition = _historical_definition()
    historical_definition["features"]["filter"] = "zero_phase"
    saved_historical = historical.create_definition(historical_definition)
    historical_run = historical.run({"id": saved_historical["id"], "revision": 1}, "retrospective")
    publication = historical.publish(historical_run["id"], "research_display")["publication"]
    assert historical_run["causality"]["is_causal"] is False

    definition = _template("regime_conditioned")
    definition["scenario"] = {
        "historical_run_ref": {"run_id": historical_run["id"], "publication_id": publication["id"]},
        "transition": {
            "states": [
                {"id": "bull", "asset_returns": {"cn_equity": 0.02, "duration_bond": -0.002, "gold": 0.003}},
                {"id": "sideways", "asset_returns": {"cn_equity": 0.001, "duration_bond": 0.002, "gold": 0.001}},
                {"id": "bear", "asset_returns": {"cn_equity": -0.02, "duration_bond": 0.01, "gold": 0.008}},
            ],
            "path_count": 200,
            "seed": 11,
        },
    }
    scenario_service = ScenarioStressService(tmp_path, tmp_path)
    saved_scenario = scenario_service.create_definition(definition)
    run = scenario_service.run({"id": saved_scenario["id"], "revision": 1})
    source = run["data_snapshot"]["historical_regime_reference"]
    assert source["historical_mode"] == "retrospective"
    assert source["causality_gate_passed"] is False
    assert run["governance"]["publish_eligible_usages"] == ["research_display"]
    with pytest.raises(ValidationError) as error:
        scenario_service.publish(run["id"], "taa")
    assert error.value.code == "SCENARIO_PUBLICATION_BLOCKED"


def _published_v2_evaluation_run(tmp_path: Path) -> tuple[dict, dict]:
    artifact_dir = tmp_path / "historical_regime_v2_artifacts"
    artifact_dir.mkdir(exist_ok=True)
    dates = pd.date_range("2024-01-02", periods=9, freq="B")
    state_values = ["bull", "bull", "bull", "bear", "bear", "bear", "bull", "bull", "bull"]
    series_frame = pd.DataFrame(
        {
            "observation_date": dates.strftime("%Y-%m-%d"),
            "state_id": state_values,
            "probabilities_json": ["{}"] * len(dates),
            "features_json": ["{}"] * len(dates),
            "reasons_json": ["[]"] * len(dates),
        }
    )
    temporary_series = artifact_dir / "evaluation-series.parquet"
    series_frame.to_parquet(temporary_series, index=False)
    series_digest = hashlib.sha256(temporary_series.read_bytes()).hexdigest()
    temporary_series.replace(artifact_dir / f"{series_digest}.parquet")
    series_artifact = {
        "artifact_id": f"regime-series-sha256-{series_digest}",
        "checksum": f"sha256:{series_digest}",
        "format": "parquet",
        "row_count": len(dates),
    }

    date_codes = np.ascontiguousarray(dates.to_numpy(dtype="datetime64[ns]").view(np.int64))
    levels = {
        "equity_eval": np.ascontiguousarray([100, 103, 101, 99, 96, 98, 102, 104, 103], dtype=np.float64),
        "bond_eval": np.ascontiguousarray([100, 99.5, 100, 101, 102, 101.5, 101, 100.5, 101], dtype=np.float64),
        "gold_eval": np.ascontiguousarray([100, 101, 100.5, 102, 104, 103, 102, 103, 104], dtype=np.float64),
    }
    arrays: dict[str, np.ndarray] = {}
    catalog: list[dict] = []
    evaluation_results: dict[str, dict] = {}
    for index, (target_id, values) in enumerate(levels.items()):
        prefix = f"n{index}_p0"
        arrays[f"{prefix}_values"] = values
        arrays[f"{prefix}_dates"] = date_codes
        arrays[f"{prefix}_available"] = date_codes
        entry = {
            "node_id": target_id,
            "node_type": "evaluation_target",
            "port": "value",
            "array_prefix": prefix,
            "shape": [len(dates)],
            "dtype": "float64",
        }
        catalog.append(entry)
        evaluation_results[target_id] = {
            "id": target_id,
            "name": target_id,
            "primary": target_id == "equity_eval",
            "source": {"kind": "index", "frequency": "daily"},
            "snapshot": {"fingerprint": f"snapshot-{target_id}"},
            "conditional_metrics": [
                {"state_id": "bull", "mean_period_return": 0.01},
                {"state_id": "bear", "mean_period_return": -0.01},
            ],
            "artifact": copy.deepcopy(entry),
        }
    temporary_output = artifact_dir / "evaluation-output.npz.tmp"
    with temporary_output.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    output_digest = hashlib.sha256(temporary_output.read_bytes()).hexdigest()
    final_output = artifact_dir / f"{output_digest}.npz"
    temporary_output.replace(final_output)
    evaluation_artifact = {
        "artifact_id": f"regime-output-sha256-{output_digest}",
        "checksum": f"sha256:{output_digest}",
        "format": "npz",
        "schema_version": "regime-node-output-v1",
        "content_addressed": True,
        "size_bytes": final_output.stat().st_size,
        "arrays": catalog,
        "uri": f"historical-regime-artifact://regime-output-sha256-{output_digest}",
    }
    analytical = {
        "schema_version": "2.0",
        "definition_id": "regime-definition-v2",
        "definition_revision": 3,
        "mode": "realtime",
        "states": [{"id": "bull", "label": "牛市"}, {"id": "bear", "label": "熊市"}],
        "transition": {
            "states": ["bull", "bear"],
            "probabilities": [[0.75, 0.25], [0.35, 0.65]],
        },
        "series_artifact": series_artifact,
        "evaluation_results": evaluation_results,
        "artifact_manifest": {"evaluation_targets": evaluation_artifact},
        "causality": {"is_causal": True, "uses_future_data": False, "repaints": False},
        "application_bindings": [],
    }
    content_hash = _content_hash(analytical)
    raw_run = {
        **analytical,
        "id": "published-regime-v2-evaluation",
        "created_at": "2024-02-01T00:00:00Z",
        "immutable": True,
        "publications": [
            {
                "id": "publication-regime-v2-evaluation",
                "usage": "research_display",
                "run_id": "published-regime-v2-evaluation",
                "run_content_hash": content_hash,
            }
        ],
        "content_hash": content_hash,
    }
    source = {
        "kind": "historical_evaluation_targets",
        "run_content_hash": content_hash,
        "evaluation_artifact_checksum": evaluation_artifact["checksum"],
        "sampling": "empirical_bootstrap",
        "minimum_observations_per_state": 2,
        "inline_policy": "forbid",
        "asset_target_map": {
            "cn_equity": {"target_id": "equity_eval", "return_transform": "simple_return"},
            "duration_bond": {"target_id": "bond_eval", "return_transform": "simple_return"},
            "gold": {"target_id": "gold_eval", "return_transform": "simple_return"},
        },
    }
    return raw_run, source


def _historical_distribution_definition(raw_run: dict, source: dict) -> dict:
    definition = _template("regime_conditioned")
    definition["scenario"] = {
        "historical_run_ref": {
            "run_id": raw_run["id"],
            "publication_id": raw_run["publications"][0]["id"],
        },
        "transition": {
            "asset_return_source": copy.deepcopy(source),
            "path_count": 200,
            "seed": 41,
            "target_return": 0.01,
        },
    }
    return definition


def test_regime_conditioned_uses_locked_v2_evaluation_targets_as_empirical_distribution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_run, source = _published_v2_evaluation_run(tmp_path)
    monkeypatch.setattr(HistoricalRegimeService, "get_run", lambda _self, _run_id: copy.deepcopy(raw_run))
    definition = _historical_distribution_definition(raw_run, source)
    service = ScenarioStressService(tmp_path, tmp_path)
    first = service.run(definition)
    second = service.run(definition)
    assert first["results"] == second["results"]
    diagnostics = first["mapping_diagnostics"]
    assert diagnostics["initial_state"] == "bull"
    assert diagnostics["asset_return_sampling"] == "historical_state_empirical_bootstrap"
    distribution_source = first["data_snapshot"]["historical_regime_reference"]["asset_return_distribution"]
    assert distribution_source["observations_by_state"] == {"bull": 5, "bear": 3}
    assert distribution_source["evaluation_artifact_checksum"] == source["evaluation_artifact_checksum"]
    assert distribution_source["joint_sampling"] == "complete_cross_asset_observation"
    result = first["results"][0]
    assert result["distribution"]["asset_return_sampling"] == "historical_state_empirical_bootstrap"
    assert result["contributions"]["method"] == "historical_state_empirical_bootstrap_mean_sum_of_period_contributions"
    assert "_historical_asset_distribution" not in first["definition"]["scenario"]


def test_historical_distribution_rejects_implicit_inline_mix_and_accepts_explicit_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_run, source = _published_v2_evaluation_run(tmp_path)
    monkeypatch.setattr(HistoricalRegimeService, "get_run", lambda _self, _run_id: copy.deepcopy(raw_run))
    definition = _historical_distribution_definition(raw_run, source)
    definition["scenario"]["transition"]["states"] = [
        {"id": "bull", "asset_returns": {"gold": 0.004}},
        {"id": "bear", "asset_returns": {"gold": 0.009}},
    ]
    with pytest.raises(ValidationError) as error:
        ScenarioStressService(tmp_path, tmp_path).run(definition)
    assert error.value.code == "HISTORICAL_DISTRIBUTION_INLINE_MIX_FORBIDDEN"

    definition["scenario"]["transition"]["asset_return_source"]["inline_policy"] = "override"
    del definition["scenario"]["transition"]["asset_return_source"]["asset_target_map"]["gold"]
    result = ScenarioStressService(tmp_path, tmp_path).run(definition)
    source_audit = result["mapping_diagnostics"]["source"]["asset_return_distribution"]
    assert source_audit["inline_overrides"] == ["bear.gold", "bull.gold"]
    assert "gold" not in source_audit["asset_target_map"]


@pytest.mark.parametrize(
    ("mutate", "expected_code"),
    [
        (
            lambda source: source.update({"run_content_hash": "0" * 64}),
            "HISTORICAL_RUN_CONTENT_LOCK_MISMATCH",
        ),
        (
            lambda source: source.update({"evaluation_artifact_checksum": f"sha256:{'0' * 64}"}),
            "HISTORICAL_EVALUATION_ARTIFACT_LOCK_MISMATCH",
        ),
    ],
)
def test_historical_distribution_fails_closed_on_version_lock_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutate,
    expected_code: str,
) -> None:
    raw_run, source = _published_v2_evaluation_run(tmp_path)
    monkeypatch.setattr(HistoricalRegimeService, "get_run", lambda _self, _run_id: copy.deepcopy(raw_run))
    mutate(source)
    with pytest.raises(ValidationError) as error:
        ScenarioStressService(tmp_path, tmp_path).run(_historical_distribution_definition(raw_run, source))
    assert error.value.code == expected_code


def test_historical_distribution_requires_minimum_complete_samples_per_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_run, source = _published_v2_evaluation_run(tmp_path)
    monkeypatch.setattr(HistoricalRegimeService, "get_run", lambda _self, _run_id: copy.deepcopy(raw_run))
    source["minimum_observations_per_state"] = 4
    with pytest.raises(ValidationError) as error:
        ScenarioStressService(tmp_path, tmp_path).run(_historical_distribution_definition(raw_run, source))
    assert error.value.code == "HISTORICAL_STATE_DISTRIBUTION_INSUFFICIENT"
    assert error.value.diagnostics[0]["insufficient"] == {"bear": 3}


def test_historical_distribution_rejects_tampered_evaluation_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_run, source = _published_v2_evaluation_run(tmp_path)
    monkeypatch.setattr(HistoricalRegimeService, "get_run", lambda _self, _run_id: copy.deepcopy(raw_run))
    digest = source["evaluation_artifact_checksum"].removeprefix("sha256:")
    artifact_path = tmp_path / "historical_regime_v2_artifacts" / f"{digest}.npz"
    with artifact_path.open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValidationError) as error:
        ScenarioStressService(tmp_path, tmp_path).run(_historical_distribution_definition(raw_run, source))
    assert error.value.code == "HISTORICAL_EVALUATION_ARTIFACT_SIZE_MISMATCH"


def test_historical_distribution_kernels_are_fixed_signature_and_preserve_joint_rows() -> None:
    warm_scenario_numba_kernels()
    preparation_signatures = tuple(prepare_historical_state_samples_kernel.signatures)
    projection_signatures = tuple(empirical_regime_projection_kernel.signatures)
    levels = np.ascontiguousarray(
        [[100.0, 100.0], [110.0, 90.0], [99.0, 99.0], [108.9, 89.1]],
        dtype=np.float64,
    )
    states = np.ascontiguousarray([0, 0, 1, 1], dtype=np.int64)
    modes = np.ascontiguousarray([0, 0], dtype=np.int64)
    overrides = np.ascontiguousarray([[np.nan, np.nan], [np.nan, np.nan]], dtype=np.float64)
    grouped, offsets, _, means, _, status, _, _ = prepare_historical_state_samples_kernel(
        levels,
        states,
        modes,
        overrides,
        np.int64(2),
    )
    assert status == 0
    assert offsets.tolist() == [0, 2, 3]
    assert means[0].tolist() == pytest.approx([0.0, 0.0])
    state_paths = np.ascontiguousarray([[0, 1], [1, 0]], dtype=np.int64)
    draws = np.ascontiguousarray([[0.0, 0.5], [0.5, 0.0]], dtype=np.float64)
    returns, _, projection_status, _, _, _ = empirical_regime_projection_kernel(
        state_paths,
        draws,
        np.ascontiguousarray(grouped[: int(offsets[-1])]),
        offsets,
        np.ascontiguousarray([0.5, 0.5], dtype=np.float64),
    )
    assert projection_status == 0
    assert returns.shape == (2, 2)
    assert tuple(prepare_historical_state_samples_kernel.signatures) == preparation_signatures
    assert tuple(empirical_regime_projection_kernel.signatures) == projection_signatures


def test_historical_distribution_kernel_excludes_incomplete_joint_row_without_zero_fill() -> None:
    levels = np.ascontiguousarray(
        [[100.0, 100.0], [101.0, np.nan], [102.0, 101.0], [103.0, 102.0]],
        dtype=np.float64,
    )
    grouped, offsets, _, _, _, status, _, _ = prepare_historical_state_samples_kernel(
        levels,
        np.ascontiguousarray([0, 0, 0, 0], dtype=np.int64),
        np.ascontiguousarray([0, 0], dtype=np.int64),
        np.ascontiguousarray([[np.nan, np.nan]], dtype=np.float64),
        np.int64(1),
    )
    assert status == 0
    assert offsets.tolist() == [0, 1]
    assert grouped[0].tolist() == pytest.approx([103.0 / 102.0 - 1.0, 102.0 / 101.0 - 1.0])

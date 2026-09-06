from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cal_indicators.typed_operators import TYPED_COMPILER_VERSION
from custom_indicators.errors import ConflictError, ValidationError
from historical_regimes.service import HistoricalRegimeService


def _rows(length: int = 140) -> list[dict[str, object]]:
    dates = pd.bdate_range("2020-01-02", periods=length)
    changes = np.r_[np.full(length // 3, 0.004), np.full(length // 3, -0.005), np.full(length - 2 * (length // 3), 0.0001)]
    values = 100.0 * np.exp(np.cumsum(changes))
    return [
        {"observation_date": date.date().isoformat(), "available_at": date.date().isoformat(), "value": float(value)}
        for date, value in zip(dates, values)
    ]


def _definition(rows: list[dict[str, object]], family: str = "causal_filter") -> dict[str, object]:
    return {
        "name": "测试历史情景",
        "description": "固定内联样本",
        "template_id": "test-template",
        "target": {"kind": "inline", "series_id": "fixture", "name": "固定样本", "frequency": "daily", "rows": rows},
        "features": {"transform": "log", "filter": "ema", "window": 8, "slope_window": 3, "volatility_window": 8},
        "algorithm": {
            "family": family,
            "parameters": {"bull_enter": 0.001, "bear_enter": -0.001, "confirmation": 2, "min_duration": 2},
        },
        "states": [],
        "validation": {"walk_forward": True, "folds": 3},
        "usage_intent": "taa",
    }


@pytest.fixture()
def service(tmp_path: Path) -> HistoricalRegimeService:
    return HistoricalRegimeService(workspace_data_dir=tmp_path, market_data_dir=tmp_path)


def _run_prepared_formula(
    service: HistoricalRegimeService,
    definition: dict[str, object],
    mode: str = "realtime",
) -> dict[str, object]:
    prepared = service.prepare_formula(definition, mode)
    return service.run(
        definition,
        mode,
        compile_token=prepared.get("compile_token"),
    )


def test_meta_has_four_copyable_canonical_templates(service: HistoricalRegimeService) -> None:
    meta = service.meta()
    template_ids = {item["id"] for item in meta["templates"]}
    assert {"bull-bear-causal", "merrill-clock", "size-rotation", "growth-value-rotation"} <= template_ids
    assert "feature_catalog" in meta
    assert meta["formula_language"]["allowlist_version"] == "typed-njit-causal-1"
    assert meta["formula_language"]["njit_required"] is True
    assert meta["formula_language"]["python_fallback"] == 0
    assert {item["id"] for item in meta["formula_language"]["functions"]} >= {
        "lag",
        "difference",
        "difference",
        "cumulative_sum",
        "cumulative_max",
    }
    for item in meta["templates"]:
        definition = item["definition"]
        assert definition["template_id"] == item["id"]
        assert "target" in definition
        assert "parameters" in definition["algorithm"]


def test_definition_is_versioned_and_accepts_legacy_aliases(service: HistoricalRegimeService) -> None:
    draft = _definition(_rows())
    legacy = copy.deepcopy(draft)
    legacy["data"] = legacy.pop("target")
    legacy["template"] = legacy.pop("template_id")
    algorithm = legacy["algorithm"]
    algorithm["params"] = algorithm.pop("parameters")
    created = service.create_definition(legacy)
    assert created["revision"] == 1
    assert "target" in created and "data" not in created
    assert "parameters" in created["algorithm"] and "params" not in created["algorithm"]
    updated_draft = copy.deepcopy(created)
    updated_draft["description"] = "第二版"
    updated = service.update_definition(created["id"], 1, updated_draft)
    assert updated["revision"] == 2
    assert service.get_definition(created["id"], 1)["description"] == "固定内联样本"
    with pytest.raises(ConflictError):
        service.update_definition(created["id"], 1, updated_draft)


def test_realtime_rule_is_prefix_invariant_and_next_period_effective(service: HistoricalRegimeService) -> None:
    saved = service.create_definition(_definition(_rows()))
    run = service.run({"id": saved["id"], "revision": saved["revision"]}, "realtime")
    assert run["immutable"] is True
    assert run["causality"]["is_causal"] is True
    assert {"formal_backtest", "taa"} <= set(run["causality"]["publish_eligible_usages"])
    assert run["stability"]["prefix_invariance"]["revisions"] == 0
    assert run["series"][-1]["effective_date"] is None
    assert run["series"][-1]["executable"] is False
    for point in run["series"][:-1]:
        assert point["observation_date"] <= point["data_available_at"] <= point["recognized_at"] <= point["effective_date"]
        assert point["effective_date"] > point["recognized_at"]
    before_series = copy.deepcopy(run["series"])
    before_hash = run["content_hash"]
    publication = service.publish(run["id"], "taa")["publication"]
    assert publication["run_id"] == run["id"]
    assert publication["definition_revision"] == 1
    stored = service.get_run(run["id"])
    assert stored["series"] == before_series
    assert stored["content_hash"] == before_hash


def test_turning_point_is_research_only_and_publish_fails_closed(service: HistoricalRegimeService) -> None:
    definition = _definition(_rows(), "turning_point")
    definition["algorithm"]["parameters"] = {"window": 6, "min_move": 0.02}
    saved = service.create_definition(definition)
    run = service.run({"id": saved["id"], "revision": saved["revision"]}, "retrospective")
    assert run["causality"]["uses_future_data"] is True
    assert run["causality"]["repaints"] is True
    assert "taa" not in run["causality"]["publish_eligible_usages"]
    service.publish(run["id"], "research_display")
    with pytest.raises(ValidationError) as error:
        service.publish(run["id"], "taa")
    assert error.value.code == "NON_CAUSAL_PUBLICATION_BLOCKED"


def test_inline_trial_cannot_be_published(service: HistoricalRegimeService) -> None:
    run = service.run(_definition(_rows()), "realtime")
    assert run["definition_source"] == "inline_trial"
    assert run["definition_id"] is None
    assert run["definition_revision"] is None
    assert run["causality"]["publish_eligible_usages"] == []
    with pytest.raises(ValidationError) as error:
        service.publish(run["id"], "research_display")
    assert error.value.code == "UNVERSIONED_RUN_PUBLICATION_BLOCKED"


def test_unsaved_changes_run_as_unpublishable_inline_trial(service: HistoricalRegimeService) -> None:
    saved = service.create_definition(_definition(_rows()))
    forged = copy.deepcopy(saved)
    forged["algorithm"]["parameters"]["bull_enter"] = 99
    run = service.run(forged, "realtime")
    assert run["definition_source"] == "inline_trial"
    assert run["definition_id"] is None
    assert run["definition_revision"] is None
    with pytest.raises(ValidationError) as error:
        service.publish(run["id"], "research_display")
    assert error.value.code == "UNVERSIONED_RUN_PUBLICATION_BLOCKED"


def test_realtime_uses_first_release_and_retrospective_latest_vintage(service: HistoricalRegimeService) -> None:
    dates = pd.date_range("2022-01-01", periods=12, freq="MS")
    rows: list[dict[str, object]] = []
    for index, date in enumerate(dates):
        release = date + pd.Timedelta(days=15)
        rows.append({"observation_date": date.date().isoformat(), "available_at": release.date().isoformat(), "value": 100 + index, "revision": 1, "vintage": "first"})
        rows.append({"observation_date": date.date().isoformat(), "available_at": (release + pd.Timedelta(days=10)).date().isoformat(), "value": 200 + index, "revision": 2, "vintage": "final"})
    definition = _definition(rows)
    definition["target"]["frequency"] = "monthly"
    definition["features"] = {"transform": "identity", "filter": "ema", "window": 3, "slope_window": 1, "volatility_window": 3}
    realtime = service.run(definition, "realtime")
    retrospective = service.run(definition, "retrospective")
    assert realtime["series"][0]["value"] == 100
    assert realtime["series"][0]["is_final"] is False
    assert retrospective["series"][0]["value"] == 200
    assert retrospective["series"][0]["is_final"] is True
    assert "taa" not in retrospective["causality"]["publish_eligible_usages"]


def test_hmm_realtime_filters_after_training_but_retrospective_smooths(service: HistoricalRegimeService) -> None:
    definition = _definition(_rows(120), "hmm")
    definition["algorithm"]["parameters"] = {"states": 3, "initial_train_size": 35, "iterations": 20, "volatility_window": 5}
    definition["validation"] = {"walk_forward": False, "folds": 2}
    realtime = service.run(definition, "realtime")
    retrospective = service.run(definition, "retrospective")
    assert realtime["algorithm_diagnostics"]["parameters"]["probability_type"] == "filtered"
    assert realtime["algorithm_diagnostics"]["feature_fields"] == ["trend_slope", "rolling_volatility"]
    assert realtime["algorithm_diagnostics"]["feature_pipeline"]["filter"] == "ema"
    assert realtime["algorithm_diagnostics"]["feature_pipeline"]["window"] == 8
    assert realtime["causality"]["classification"] == "point_in_time_trained"
    assert realtime["series"][10]["state_id"] == "unclassified"
    assert realtime["series"][10]["probabilities"] == {}
    assert realtime["series"][10]["confidence"] is None
    assert retrospective["algorithm_diagnostics"]["parameters"]["probability_type"] == "smoothed"
    assert retrospective["causality"]["is_causal"] is False
    assert "taa" not in retrospective["causality"]["publish_eligible_usages"]


def test_latent_model_accepts_comma_separated_uploaded_feature_fields(service: HistoricalRegimeService) -> None:
    rows = _rows(90)
    for index, row in enumerate(rows):
        row["momentum"] = float(np.sin(index / 7))
        row["liquidity"] = float(np.cos(index / 9))
    definition = _definition(rows, "gmm")
    definition["features"] = {"transform": "zscore", "filter": "kalman", "window": 11, "slope_window": 4, "volatility_window": 9}
    definition["algorithm"]["parameters"] = {
        "states": 3,
        "initial_train_size": 30,
        "iterations": 10,
        "feature_fields": "momentum, liquidity",
    }
    definition["validation"] = {"walk_forward": False, "folds": 2}
    run = service.run(definition, "realtime")
    diagnostics = run["algorithm_diagnostics"]
    assert diagnostics["feature_fields"] == ["momentum", "liquidity"]
    assert diagnostics["feature_pipeline"]["source"] == "uploaded_fields"
    assert diagnostics["feature_pipeline"]["filter"] == "kalman"
    assert diagnostics["feature_pipeline"]["window"] == 11


@pytest.mark.parametrize("family", ["gmm", "markov"])
def test_other_latent_families_have_distinct_realtime_training_contract(
    service: HistoricalRegimeService,
    family: str,
) -> None:
    definition = _definition(_rows(110), family)
    definition["algorithm"]["parameters"] = {"states": 3, "initial_train_size": 30, "iterations": 12, "volatility_window": 5}
    definition["validation"] = {"walk_forward": False, "folds": 2, "stability_perturbation": 0.05}
    run = service.run(definition, "realtime")
    assert run["causality"]["is_causal"] is True
    assert run["algorithm_diagnostics"]["training_observations"] == 30
    assert run["stability"]["parameter_sensitivity"]["perturbation"] == pytest.approx(0.05)


def test_change_point_has_online_and_centered_paths(service: HistoricalRegimeService) -> None:
    definition = _definition(_rows(100), "change_point")
    definition["algorithm"]["parameters"] = {"window": 6, "threshold": 0.8, "confirmation": 2}
    definition["validation"] = {"walk_forward": False, "folds": 2}
    realtime = service.run(definition, "realtime")
    retrospective = service.run(definition, "retrospective")
    assert realtime["algorithm_diagnostics"]["detector"] == "online_two_window"
    assert realtime["causality"]["uses_future_data"] is False
    assert retrospective["algorithm_diagnostics"]["detector"] == "centered_two_window"
    assert retrospective["causality"]["uses_future_data"] is True


def test_ensemble_records_members_weights_and_conflict_rejections(service: HistoricalRegimeService) -> None:
    definition = _definition(_rows(100), "ensemble")
    definition["algorithm"]["parameters"] = {
        "consensus_threshold": 0.65,
        "members": [
            {
                "family": "causal_filter",
                "weight": 0.6,
                "parameters": {"bull_enter": 0.001, "bear_enter": -0.001, "confirmation": 2},
            },
            {
                "family": "change_point",
                "weight": 0.4,
                "parameters": {"window": 6, "threshold": 0.8, "confirmation": 2},
            },
        ],
    }
    definition["validation"] = {"walk_forward": False, "folds": 2}
    run = service.run(definition, "realtime")
    diagnostics = run["algorithm_diagnostics"]
    assert diagnostics["model"] == "ensemble"
    assert [member["family"] for member in diagnostics["members"]] == ["causal_filter", "change_point"]
    assert diagnostics["consensus_threshold"] == pytest.approx(0.65)
    assert diagnostics["conflict_rejections"] >= 0
    assert run["evidence"][0]["kind"] == "ensemble_configuration"
    assert run["causality"]["is_causal"] is True


def test_recursive_ensemble_is_blocked(service: HistoricalRegimeService) -> None:
    definition = _definition(_rows(80), "ensemble")
    definition["algorithm"]["parameters"] = {
        "members": [
            {"family": "ensemble", "weight": 1, "parameters": {}},
            {"family": "causal_filter", "weight": 1, "parameters": {}},
        ]
    }
    with pytest.raises(ValidationError) as error:
        service.run(definition, "realtime")
    assert error.value.code == "RECURSIVE_ENSEMBLE_BLOCKED"


def test_merrill_clock_uses_release_dates_and_four_states(service: HistoricalRegimeService) -> None:
    dates = pd.date_range("2018-01-01", periods=36, freq="MS")
    rows = [
        {
            "observation_date": date.date().isoformat(),
            "available_at": (date + pd.Timedelta(days=20)).date().isoformat(),
            "growth": float(np.sin(index / 5)),
            "inflation": float(np.cos(index / 6)),
            "vintage": "first",
        }
        for index, date in enumerate(dates)
    ]
    definition = {
        "name": "宏观时钟测试",
        "description": "发布日期口径",
        "template_id": "merrill-clock",
        "target": {"kind": "inline", "frequency": "monthly", "availability_mode": "point_in_time", "rows": rows},
        "features": {"filter": "ema", "window": 3, "slope_window": 1},
        "algorithm": {"family": "merrill_clock", "parameters": {"growth_field": "growth", "inflation_field": "inflation", "confirmation": 1}},
        "states": [],
        "validation": {"walk_forward": False, "folds": 2},
    }
    run = service.run(definition, "realtime")
    assert run["algorithm_diagnostics"]["release_date_aware"] is True
    assert {state["id"] for state in run["states"]} == {"recovery", "overheat", "stagflation", "recession"}
    assert all(point["data_available_at"] >= point["observation_date"] for point in run["series"])


def test_relative_strength_uses_strict_intersection(service: HistoricalRegimeService) -> None:
    dates = pd.bdate_range("2021-01-01", periods=60)
    numerator = [{"date": date.date().isoformat(), "value": 100 + index * 1.2} for index, date in enumerate(dates)]
    denominator = [{"date": date.date().isoformat(), "value": 100 + index * 0.4} for index, date in enumerate(dates[5:])]
    definition = _definition(_rows(60), "relative_strength")
    definition["target"] = {
        "kind": "relative",
        "frequency": "daily",
        "numerator": {"kind": "inline", "rows": numerator},
        "denominator": {"kind": "inline", "rows": denominator},
        "transform": "log_ratio",
    }
    definition["features"] = {"transform": "identity", "filter": "ema", "window": 5, "slope_window": 2, "volatility_window": 5}
    definition["algorithm"]["parameters"] = {"upper": 0.0005, "lower": -0.0005, "confirmation": 2}
    definition["validation"] = {"walk_forward": False, "folds": 2}
    run = service.run(definition, "realtime")
    assert run["data_snapshot"]["alignment"] == "strict_intersection"
    assert len(run["series"]) == 55


def test_index_source_uses_code_filter_column_projection_and_date_range(
    service: HistoricalRegimeService,
    tmp_path: Path,
) -> None:
    dates = pd.bdate_range("2023-01-02", periods=30)
    market_rows = [
        {"ts_code": code, "trade_date": date.date().isoformat(), "close": float(100 + index), "unused_payload": "not-read"}
        for code in ("000300.SH", "000905.SH")
        for index, date in enumerate(dates)
    ]
    pd.DataFrame(market_rows).to_parquet(tmp_path / "index_daily_df.parquet", index=False)
    definition = _definition(_rows(20))
    definition["target"] = {
        "kind": "index",
        "series_id": "000300.SH",
        "name": "沪深300",
        "frequency": "daily",
        "source_api": "index_daily",
        "ts_code": "000300.SH",
        "field": "close",
        "start_date": dates[5].date().isoformat(),
        "end_date": dates[24].date().isoformat(),
    }
    definition["features"] = {"transform": "log", "filter": "ema", "window": 4, "slope_window": 2, "volatility_window": 4}
    definition["validation"] = {"walk_forward": False, "folds": 2}
    run = service.run(definition, "realtime")
    assert len(run["series"]) == 20
    assert run["data_snapshot"]["filters"] == {"ts_code": "000300.SH"}
    assert run["data_snapshot"]["projection"] == ["ts_code", "trade_date", "close"]
    assert run["data_snapshot"]["selected_observations"] == 20
    assert all(point["value"] < 130 for point in run["series"])


@pytest.mark.parametrize("transform", ["identity", "log", "return", "zscore"])
def test_registered_transforms_execute_with_default_parameters(
    service: HistoricalRegimeService,
    transform: str,
) -> None:
    definition = _definition(_rows(80))
    definition["features"]["transform"] = transform
    definition["algorithm"]["parameters"] = {}
    run = service.run(definition, "realtime")
    assert len(run["series"]) == 80
    assert isinstance(run["diagnostics"], list)


def test_causal_formula_is_executed_and_audited(service: HistoricalRegimeService) -> None:
    definition = _definition(_rows())
    definition["features"]["formula"] = "difference(log(value), 1)"
    run = _run_prepared_formula(service, definition)
    audit = run["formula_diagnostics"]
    assert audit["allowlist_version"] == "typed-njit-causal-1"
    assert audit["evaluator_version"] == TYPED_COMPILER_VERSION
    assert audit["referenced_columns"] == ["value"]
    assert audit["functions"] == ["difference", "log"]
    assert audit["execution_backend"] == "numba_njit_fixed_signature"
    assert audit["compile_status"] == "compiled"
    assert audit["python_fallback"] == 0
    assert audit["python_operator_calls"] == 0
    assert audit["dag"]["roots"]["result"] is not None
    assert audit["is_causal"] is True
    assert run["data_snapshot"]["feature_formula"]["fingerprint"] == audit["fingerprint"]
    assert run["algorithm_diagnostics"]["feature_formula"]["normalized_expression"]
    assert any(item["code"] == "CAUSAL_FORMULA_COMPILED" for item in run["diagnostics"])


def test_formula_drives_classification_without_replacing_raw_reporting_values(
    service: HistoricalRegimeService,
) -> None:
    rows = _rows(100)
    definition = _definition(rows)
    definition["features"]["formula"] = "log(value)"
    run = _run_prepared_formula(service, definition)

    raw_values = np.array([float(row["value"]) for row in rows])
    assert [point["value"] for point in run["series"]] == pytest.approx(raw_values.tolist())
    assert [point["features"]["formula_input"] for point in run["series"]] == pytest.approx(
        np.log(raw_values).tolist()
    )

    metric = next(item for item in run["conditional_metrics"] if item["return_observations"] > 0)
    state_mask = np.array([point["state_id"] == metric["state_id"] for point in run["series"]])
    raw_forward_returns = np.r_[raw_values[1:] / raw_values[:-1] - 1.0, np.nan]
    expected_raw_mean = float(np.mean(raw_forward_returns[state_mask & np.isfinite(raw_forward_returns)]))
    formula_values = np.log(raw_values)
    formula_forward_returns = np.r_[formula_values[1:] / formula_values[:-1] - 1.0, np.nan]
    formula_mean = float(np.mean(formula_forward_returns[state_mask & np.isfinite(formula_forward_returns)]))

    assert metric["mean_period_return"] == pytest.approx(expected_raw_mean)
    assert metric["mean_period_return"] != pytest.approx(formula_mean)


def test_custom_formula_changes_regime_segments(service: HistoricalRegimeService) -> None:
    positive = _definition(_rows(120))
    positive["features"]["formula"] = "value"
    negative = copy.deepcopy(positive)
    negative["features"]["formula"] = "-value"
    positive_run = _run_prepared_formula(service, positive)
    negative_run = _run_prepared_formula(service, negative)
    positive_labels = [point["state_id"] for point in positive_run["series"]]
    negative_labels = [point["state_id"] for point in negative_run["series"]]
    assert positive_labels != negative_labels
    assert [(item["state_id"], item["start_date"], item["end_date"]) for item in positive_run["segments"]] != [
        (item["state_id"], item["start_date"], item["end_date"]) for item in negative_run["segments"]
    ]


def test_unverified_formula_column_is_research_only_and_cannot_publish_taa(
    service: HistoricalRegimeService,
) -> None:
    rows = _rows(80)
    for index, row in enumerate(rows):
        row["future_return"] = (
            float(rows[index + 1]["value"]) / float(row["value"]) - 1.0
            if index + 1 < len(rows)
            else None
        )
    definition = _definition(rows)
    definition["features"]["formula"] = "future_return"
    saved = service.create_definition(definition)
    reference = {"id": saved["id"], "revision": saved["revision"]}
    run = _run_prepared_formula(service, reference)

    assert run["formula_diagnostics"]["input_provenance"]["verified"] is False
    assert "formal_backtest" not in run["causality"]["publish_eligible_usages"]
    assert "taa" not in run["causality"]["publish_eligible_usages"]
    assert any("字段级可得日" in warning for warning in run["causality"]["warnings"])
    with pytest.raises(ValidationError) as error:
        service.publish(run["id"], "taa")
    assert error.value.code == "NON_CAUSAL_PUBLICATION_BLOCKED"


def test_declared_formula_column_requires_valid_point_in_time_availability(
    service: HistoricalRegimeService,
) -> None:
    rows = _rows(80)
    for row in rows:
        row["growth"] = float(row["value"])
        row["growth_available_at"] = row["available_at"]
    definition = _definition(rows)
    definition["features"]["formula"] = "growth"
    definition["features"]["formula_provenance"] = {
        "growth": {"point_in_time": True, "available_at_field": "growth_available_at"}
    }
    saved = service.create_definition(definition)
    reference = {"id": saved["id"], "revision": saved["revision"]}
    run = _run_prepared_formula(service, reference)

    assert run["formula_diagnostics"]["input_provenance"]["verified"] is True
    assert {"formal_backtest", "taa"} <= set(run["causality"]["publish_eligible_usages"])


def test_causal_formula_is_prefix_invariant(service: HistoricalRegimeService) -> None:
    full_definition = _definition(_rows(120))
    full_definition["features"]["formula"] = "difference(log(value), 10)"
    prefix_definition = copy.deepcopy(full_definition)
    prefix_definition["target"]["rows"] = prefix_definition["target"]["rows"][:85]
    full = _run_prepared_formula(service, full_definition)
    prefix = _run_prepared_formula(service, prefix_definition)
    assert full["stability"]["prefix_invariance"]["revisions"] == 0
    for full_point, prefix_point in zip(full["series"][:85], prefix["series"]):
        assert full_point["state_id"] == prefix_point["state_id"]
        if full_point["features"]["formula_input"] is None:
            assert prefix_point["features"]["formula_input"] is None
        else:
            assert full_point["features"]["formula_input"] == pytest.approx(
                prefix_point["features"]["formula_input"]
            )


@pytest.mark.parametrize(
    ("formula", "code"),
    [
        ("value.__class__", "FORMULA_NODE_FORBIDDEN"),
        ("lead(value, 1)", "FORMULA_FUNCTION_FORBIDDEN"),
        ("value[-1]", "FORMULA_NODE_FORBIDDEN"),
        ("difference(value, -2)", "FORMULA_ARGUMENT_ERROR"),
        ("__import__('os')", "FORMULA_FUNCTION_FORBIDDEN"),
        ("value + " + "9" * 400, "FORMULA_LITERAL_FORBIDDEN"),
    ],
)
def test_unsafe_formula_is_rejected(
    service: HistoricalRegimeService,
    formula: str,
    code: str,
) -> None:
    definition = _definition(_rows())
    definition["features"]["formula"] = formula
    with pytest.raises(ValidationError) as error:
        service.run(definition, "realtime")
    assert error.value.code == code
    assert error.value.field == "features.formula"


@pytest.mark.parametrize("family", ["merrill_clock", "relative_strength", "ensemble"])
def test_formula_fails_closed_when_algorithm_does_not_consume_unified_input(
    service: HistoricalRegimeService,
    family: str,
) -> None:
    definition = _definition(_rows(), family)
    definition["features"]["formula"] = "log(value)"
    with pytest.raises(ValidationError) as error:
        service.run(definition, "realtime")
    assert error.value.code == "FORMULA_ALGORITHM_UNSUPPORTED"
    assert error.value.field == "features.formula"


def test_formula_missing_value_stays_unclassified_and_not_zero(service: HistoricalRegimeService) -> None:
    rows = _rows(80)
    rows[30]["value"] = None
    definition = _definition(rows)
    definition["features"]["formula"] = "value * 2"
    run = _run_prepared_formula(service, definition)
    point = run["series"][30]
    assert point["value"] is None
    assert point["features"]["formula_input"] is None
    assert point["state_id"] == "unclassified"
    assert point["confidence"] is None
    assert point["probabilities"] == {}
    assert any("未用 0 替代" in reason for reason in point["reasons"])


def test_indicator_reference_remains_explicitly_blocked(service: HistoricalRegimeService) -> None:
    definition = _definition(_rows())
    definition["features"]["indicator_ref"] = {"id": "indicator-demo", "revision": 1}
    with pytest.raises(ValidationError) as error:
        service.run(definition, "realtime")
    assert error.value.code == "UNSUPPORTED_INDICATOR_REFERENCE"


def test_missing_input_is_not_carried_forward_as_a_state(service: HistoricalRegimeService) -> None:
    rows = _rows(80)
    rows[30]["value"] = None
    run = service.run(_definition(rows), "realtime")
    point = run["series"][30]
    assert point["value"] is None
    assert point["filtered_value"] is None
    assert point["state_id"] == "unclassified"
    assert point["confidence"] is None
    assert point["probabilities"] == {}
    assert any("缺失" in reason for reason in point["reasons"])


def test_compare_returns_frontend_canonical_fields(service: HistoricalRegimeService) -> None:
    first = service.run(_definition(_rows()), "realtime")
    second_definition = _definition(_rows())
    second_definition["algorithm"]["parameters"]["confirmation"] = 4
    second = service.run(second_definition, "realtime")
    result = service.compare([first["id"], second["id"]], first["id"])
    assert result["run_ids"] == [first["id"], second["id"]]
    assert result["agreement_rate"] is not None
    assert set(result["pairwise"][0]) >= {"left_run_id", "right_run_id", "agreement_rate", "boundary_distance"}
    assert isinstance(result["disagreement_periods"], list)

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from compute_policy import validate_execution_audit
from cal_indicators.typed_dsl import TypedIndicatorRuntime
from custom_indicators.errors import ValidationError
from historical_regimes import algorithms
from historical_regimes.algorithms import run_algorithm
from historical_regimes.contracts import normalize_definition
from historical_regimes.formula import evaluate_formula, prepare_formula
from historical_regimes.numba_kernels import (
    conflict_rate_kernel,
    formula_align_block_kernel,
    formula_input_blocks_kernel,
    formula_missing_numeric_mask_kernel,
    historical_regime_numba_status,
    relative_transform_kernel,
    warm_historical_regime_numba_kernels,
)
from historical_regimes.service import HistoricalRegimeService


def _frame(length: int = 5_000) -> pd.DataFrame:
    index = np.arange(length, dtype=np.float64)
    value = 100.0 * np.exp(
        np.cumsum(0.0002 + 0.002 * np.sin(index / 37.0))
    )
    dates = pd.bdate_range("2000-01-03", periods=length)
    return pd.DataFrame(
        {
            "observation_date": dates,
            "available_at": dates,
            "value": value,
            "growth": np.sin(index / 43.0),
            "inflation": np.cos(index / 59.0),
        }
    )


def _definition(family: str) -> dict[str, object]:
    parameters: dict[str, object]
    if family == "turning_point":
        parameters = {"window": 6, "min_move": 0.003}
    elif family == "merrill_clock":
        parameters = {
            "growth_field": "growth",
            "inflation_field": "inflation",
            "confirmation": 1,
        }
    elif family in {"hmm", "markov", "gmm"}:
        parameters = {
            "states": 3,
            "initial_train_size": 30,
            "iterations": 3,
        }
    elif family == "change_point":
        parameters = {"window": 8, "threshold": 0.8, "confirmation": 1}
    elif family == "ensemble":
        parameters = {
            "consensus_threshold": 0.6,
            "members": [
                {
                    "family": "causal_filter",
                    "weight": 0.6,
                    "parameters": {"confirmation": 1},
                },
                {
                    "family": "change_point",
                    "weight": 0.4,
                    "parameters": {
                        "window": 8,
                        "threshold": 0.8,
                        "confirmation": 1,
                    },
                },
            ],
        }
    else:
        parameters = {"confirmation": 1, "min_duration": 1}
    return normalize_definition(
        {
            "name": f"{family} NJIT 回归",
            "target": {
                "kind": "inline",
                "frequency": "daily",
                "rows": [
                    {
                        "observation_date": "2020-01-02",
                        "available_at": "2020-01-02",
                        "value": 1.0,
                    }
                    for _ in range(5)
                ],
            },
            "features": {
                "transform": "log",
                "filter": "ema",
                "window": 8,
                "slope_window": 3,
                "volatility_window": 8,
            },
            "algorithm": {"family": family, "parameters": parameters},
            "states": [],
            "validation": {"walk_forward": False, "folds": 2},
        }
    )


@pytest.mark.parametrize(
    ("family", "mode"),
    [
        ("causal_filter", "realtime"),
        ("relative_strength", "realtime"),
        ("turning_point", "retrospective"),
        ("merrill_clock", "realtime"),
        ("hmm", "realtime"),
        ("markov", "realtime"),
        ("gmm", "realtime"),
        ("change_point", "realtime"),
        ("ensemble", "realtime"),
    ],
)
def test_every_algorithm_family_runs_5000_points_through_fixed_njit(
    family: str,
    mode: str,
) -> None:
    warm_historical_regime_numba_kernels()
    output = run_algorithm(_frame(), _definition(family), mode)

    assert len(output.labels) == 5_000
    assert len(output.filtered) == 5_000
    assert len(output.scores) == 5_000
    audit = output.diagnostics["execution_audit"]
    policy_audit = validate_execution_audit(audit)
    assert audit["family"] == family
    assert audit["execution_backend"] == "numba_njit_fixed_signature"
    assert audit["njit_required"] is True
    assert audit["python_fallback"] == 0
    assert audit["python_operator_calls"] == 0
    assert policy_audit["execution_backend"] == "numba_njit_fixed_signature"
    assert audit["kernels"]
    for kernel in audit["kernels"]:
        assert kernel["compile_status"] == "compiled"
        assert kernel["compiled_signatures"]
        assert len(kernel["kernel_fingerprint"]) == 64
        assert kernel["python_fallback"] == 0


def test_warmup_is_complete_and_normal_runs_do_not_add_signatures() -> None:
    before = warm_historical_regime_numba_kernels()
    policy_audit = validate_execution_audit(before)
    signature_snapshot = {
        item["kernel_id"]: tuple(item["compiled_signatures"])
        for item in before["kernels"]
    }

    run_algorithm(_frame(200), _definition("causal_filter"), "realtime")
    after = historical_regime_numba_status()

    assert after["complete"] is True
    assert policy_audit["execution_backend"] == "numba_njit_fixed_signature"
    assert policy_audit["nopython"] is True
    assert after["python_fallback"] == 0
    assert {
        item["kernel_id"]: tuple(item["compiled_signatures"])
        for item in after["kernels"]
    } == signature_snapshot


def test_relative_source_transform_is_fixed_signature_njit() -> None:
    before = tuple(relative_transform_kernel.nopython_signatures)
    numerator = np.ascontiguousarray(np.array([100.0, 110.0], dtype=np.float64))
    denominator = np.ascontiguousarray(np.array([50.0, 55.0], dtype=np.float64))

    ratio = relative_transform_kernel(numerator, denominator, np.int64(0))
    log_ratio = relative_transform_kernel(numerator, denominator, np.int64(1))

    np.testing.assert_allclose(ratio, np.array([2.0, 2.0]))
    np.testing.assert_allclose(log_ratio, np.log(np.array([2.0, 2.0])))
    assert tuple(relative_transform_kernel.nopython_signatures) == before


def test_ensemble_conflict_rate_is_fixed_signature_njit_and_audited() -> None:
    warm_historical_regime_numba_kernels()
    before = tuple(conflict_rate_kernel.nopython_signatures)

    assert conflict_rate_kernel(np.int64(3), np.int64(8)) == pytest.approx(0.375)
    assert np.isnan(conflict_rate_kernel(np.int64(0), np.int64(0)))
    assert tuple(conflict_rate_kernel.nopython_signatures) == before
    assert len(before) == 1
    assert conflict_rate_kernel._can_compile is False

    output = run_algorithm(_frame(200), _definition("ensemble"), "realtime")
    diagnostics = output.diagnostics
    audit = validate_execution_audit(diagnostics["execution_audit"])

    assert diagnostics["conflict_rate"] == pytest.approx(
        conflict_rate_kernel(
            np.int64(diagnostics["conflict_rejections"]),
            np.int64(200),
        )
    )
    assert "conflict_rate" in audit["kernel_signatures"]
    assert len(audit["kernel_signatures"]["conflict_rate"]) == 1
    assert audit["python_fallback"] == 0
    assert audit["request_time_compilation"] == 0


def test_formula_batch_prepostprocess_kernels_cover_5000_points_without_new_signatures() -> None:
    warm_historical_regime_numba_kernels()
    dispatchers = (
        formula_input_blocks_kernel,
        formula_align_block_kernel,
        formula_missing_numeric_mask_kernel,
    )
    before = {
        dispatcher.py_func.__name__: tuple(dispatcher.nopython_signatures)
        for dispatcher in dispatchers
    }
    inputs = np.ascontiguousarray(
        np.vstack(
            (
                np.linspace(1.0, 2.0, 5_000),
                np.linspace(3.0, 4.0, 5_000),
            )
        ),
        dtype=np.float64,
    )
    inputs[0, 100] = np.nan
    inputs[1, 3_000] = np.inf

    complete, starts, stops, complete_count = formula_input_blocks_kernel(inputs)

    assert complete_count == 4_998
    assert complete[[100, 3_000]].tolist() == [0, 0]
    assert starts.tolist() == [0, 101, 3_001]
    assert stops.tolist() == [100, 3_000, 5_000]

    aligned = np.full(5_000, np.nan, dtype=np.float64)
    placed = formula_align_block_kernel(
        np.ascontiguousarray(np.array([10.0, 20.0], dtype=np.float64)),
        np.int64(101),
        np.int64(105),
        aligned,
    )
    assert placed == 2
    np.testing.assert_allclose(aligned[103:105], np.array([10.0, 20.0]))

    numeric_outputs = np.ascontiguousarray(
        np.vstack((np.arange(5_000, dtype=np.float64), np.ones(5_000))),
    )
    formula_values = np.ones(5_000, dtype=np.float64)
    formula_values[[100, 3_000]] = np.nan
    masked, missing_positions, valid_count = formula_missing_numeric_mask_kernel(
        formula_values,
        numeric_outputs,
    )
    assert valid_count == 4_998
    assert missing_positions.tolist() == [100, 3_000]
    assert np.isnan(masked[:, missing_positions]).all()

    after = {
        dispatcher.py_func.__name__: tuple(dispatcher.nopython_signatures)
        for dispatcher in dispatchers
    }
    assert after == before
    assert all(len(signatures) == 1 for signatures in after.values())


def test_prepared_multi_input_formula_run_never_compiles_and_audits_numeric_lane(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warm_historical_regime_numba_kernels()
    frame = _frame(5_000)
    frame.loc[100, "growth"] = np.nan
    prepared = prepare_formula("difference(log(value + growth), 3)", frame)
    signature_snapshot = {
        item["kernel_id"]: tuple(item["compiled_signatures"])
        for item in historical_regime_numba_status()["kernels"]
    }

    def reject_request_compile(*_args, **_kwargs):
        raise AssertionError("normal formula execution must not compile a plan")

    monkeypatch.setattr(TypedIndicatorRuntime, "from_plan", reject_request_compile)
    result = evaluate_formula(
        "difference(log(value + growth), 3)",
        frame,
        compile_token=prepared["compile_token"],
    )

    audit = validate_execution_audit(result.audit)
    assert audit["request_time_compilation"] == 0
    assert audit["python_fallback"] == 0
    assert audit["object_mode"] == 0
    assert {
        "formula_input_blocks",
        "formula_align_block",
        "formula_missing_numeric_mask",
    } <= set(audit["kernel_signatures"])
    assert all(
        len(audit["kernel_signatures"][kernel_id]) == 1
        for kernel_id in (
            "formula_input_blocks",
            "formula_align_block",
            "formula_missing_numeric_mask",
        )
    )
    assert audit["kernel_signatures"]["generated_plan"] == prepared["kernel_signatures"][
        "generated_plan"
    ]
    assert result.values.iloc[100:104].isna().all()
    assert {
        item["kernel_id"]: tuple(item["compiled_signatures"])
        for item in historical_regime_numba_status()["kernels"]
    } == signature_snapshot


@pytest.mark.parametrize("compile_token", [None, "wrong-token"])
def test_formula_execution_fails_closed_without_matching_compile_token(
    compile_token: str | None,
) -> None:
    warm_historical_regime_numba_kernels()
    frame = _frame(100)
    prepare_formula("value + growth", frame)

    with pytest.raises(ValidationError) as error:
        evaluate_formula(
            "value + growth",
            frame,
            compile_token=compile_token,
        )

    assert error.value.code == "FORMULA_COMPILE_TOKEN_REQUIRED"


def test_meta_marks_every_exposed_algorithm_filter_function_and_operator_njit(
    tmp_path: Path,
) -> None:
    meta = HistoricalRegimeService(tmp_path, tmp_path).meta()

    assert all(item["njit_supported"] for item in meta["algorithm_families"])
    assert all(item["njit_supported"] for item in meta["feature_catalog"])
    assert all(
        item["njit_supported"]
        for item in meta["formula_language"]["functions"]
    )
    assert all(
        item["njit_supported"]
        for item in meta["formula_language"]["operator_catalog"]
    )
    runtime = meta["historical_regime_runtime"]
    assert runtime["complete"] is True
    assert runtime["python_fallback"] == 0


def test_non_njit_or_full_sample_paths_fail_closed() -> None:
    warm_historical_regime_numba_kernels()
    unsupported_filter = _definition("causal_filter")
    unsupported_filter["features"]["filter"] = "savgol"
    with pytest.raises(ValidationError) as filter_error:
        run_algorithm(_frame(100), unsupported_filter, "realtime")
    assert filter_error.value.code == "UNSUPPORTED_FILTER"

    with pytest.raises(ValidationError):
        evaluate_formula("mean(value)", _frame(100))


def test_run_snapshot_contains_algorithm_and_analytics_njit_audits(
    tmp_path: Path,
) -> None:
    frame = _frame(160)
    definition = _definition("causal_filter")
    definition["target"]["rows"] = [
        {
            "observation_date": row.observation_date.date().isoformat(),
            "available_at": row.available_at.date().isoformat(),
            "value": float(row.value),
        }
        for row in frame.itertuples(index=False)
    ]
    service = HistoricalRegimeService(tmp_path, tmp_path)
    run = service.run(definition, "realtime")

    algorithm = run["algorithm_diagnostics"]["execution_audit"]
    analytics = run["algorithm_diagnostics"]["analytics_execution"]
    assert algorithm["python_fallback"] == 0
    assert analytics["python_fallback"] == 0
    assert all(item["compiled_signatures"] for item in algorithm["kernels"])
    assert all(item["compiled_signatures"] for item in analytics["kernels"])
    assert {
        item["source_kind"] for item in run["calculation_audits"]
    } >= {"historical_regime_algorithm"}
    assert any(
        item["code"] == "NJIT_HISTORICAL_REGIME_EXECUTED"
        for item in run["diagnostics"]
    )


def test_algorithms_module_has_no_scipy_or_pandas_numerical_path() -> None:
    source = inspect.getsource(algorithms)
    for forbidden in (
        "scipy",
        "logsumexp",
        ".rolling(",
        ".ewm(",
        ".pct_change(",
        "conflicts / len(frame)",
    ):
        assert forbidden not in source

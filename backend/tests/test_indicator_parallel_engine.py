from __future__ import annotations

import os
import math
import time
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cal_indicators.builtin_batch_kernel import (
    BUILTIN_METRIC_CODE,
    compute_builtin_batch_serial,
)
from custom_indicators.parallel_engine import (
    AdaptiveComputeEngine,
    SharedArrayOwner,
    score_matrix,
)
from custom_indicators.errors import IndicatorDomainError
from custom_indicators.service import CustomIndicatorService, _built_in_indicators


def _owner(stack: ExitStack, values: np.ndarray, runtime_dir: Path) -> SharedArrayOwner:
    return stack.enter_context(
        SharedArrayOwner(values, runtime_dir, shm_threshold_bytes=1)
    )


def test_orphan_mmap_cleanup_preserves_live_owner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("INDICATOR_ORPHAN_PROTECTION_SECONDS", "1")
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    orphan = runtime / "indicator-array-999999-orphan.mmap"
    live = runtime / f"indicator-array-{os.getpid()}-live.mmap"
    orphan.write_bytes(b"orphan")
    live.write_bytes(b"live")
    old = time.time() - 5
    os.utime(orphan, (old, old))
    os.utime(live, (old, old))

    AdaptiveComputeEngine(runtime)._cleanup_orphan_arrays()

    assert not orphan.exists()
    assert live.exists()


def test_cpu_token_queue_returns_stable_busy_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("INDICATOR_PROCESS_WORKERS", "1")
    monkeypatch.setenv("INDICATOR_QUEUE_WAIT_SECONDS", "0.01")
    engine = AdaptiveComputeEngine(tmp_path / "runtime")

    with engine.admission(1):
        with pytest.raises(IndicatorDomainError) as error:
            with engine.admission(1):
                pass

    assert error.value.code == "INDICATOR_ENGINE_BUSY"
    assert error.value.status_code == 429


def test_startup_waits_until_every_worker_is_warmed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("INDICATOR_PROCESS_WORKERS", "3")
    monkeypatch.setenv("INDICATOR_STARTUP_WARMUP_TIMEOUT_SECONDS", "60")
    engine = AdaptiveComputeEngine(tmp_path / "runtime")

    engine.start()
    try:
        status = engine.status()
        assert status["fully_warmed"] is True
        assert status["worker_capacity"] == 3
        assert status["worker_processes"] == 3
        assert len(set(status["worker_pids"])) == 3
    finally:
        engine.close()


def test_prange_scoring_only_activates_above_configured_threshold() -> None:
    raw = np.ascontiguousarray(
        np.column_stack((np.arange(32, dtype=np.float64), np.arange(32, 0, -1))),
        dtype=np.float64,
    )
    complete = np.ones(32, dtype=np.bool_)
    normalized, parallel = score_matrix(
        raw,
        np.asarray([0.5, 0.5], dtype=np.float64),
        np.asarray([0, 1], dtype=np.int8),
        complete,
        np.min(raw, axis=0),
        np.max(raw, axis=0),
        min_parallel_elements=1,
        thread_budget=2,
    )

    assert parallel is True
    assert np.allclose(normalized[:, 0], normalized[:, 1])


def test_all_35_fused_builtins_match_typed_runtime(tmp_path: Path) -> None:
    returns = np.asarray(
        [0.01, -0.006, 0.004, -0.003, 0.008, -0.002, 0.005, -0.004] * 6,
        dtype=np.float64,
    )
    nav = np.concatenate((np.asarray([1.0]), np.cumprod(1.0 + returns)))
    volume = np.linspace(1000.0, 2000.0, nav.size)
    market_high = nav * 10.2
    market_low = nav * 9.8
    values = np.ascontiguousarray(
        np.vstack((nav, volume, market_high, market_low)), dtype=np.float64
    )
    definitions = [
        definition
        for definition in _built_in_indicators()
        if definition["id"] in BUILTIN_METRIC_CODE
    ]
    codes = np.asarray(
        [BUILTIN_METRIC_CODE[definition["id"]] for definition in definitions],
        dtype=np.int64,
    )
    primary = np.full(codes.size, -1, dtype=np.int64)
    secondary = np.full(codes.size, -1, dtype=np.int64)
    for index, code in enumerate(codes):
        if code in (30, 31):
            primary[index] = 1
        elif code == 32:
            primary[index] = 2
        elif code == 33:
            primary[index] = 3
        elif code == 34:
            primary[index] = 2
            secondary[index] = 3
    output = np.full((1, len(definitions)), np.nan, dtype=np.float64)
    statuses = np.full((1, len(definitions)), -1, dtype=np.int16)
    risk_free = np.asarray(
        [
            CustomIndicatorService._risk_free_context(definition)[
                "risk_free_rate_per_observation"
            ]
            for definition in definitions
        ],
        dtype=np.float64,
    )
    compute_builtin_batch_serial(
        values,
        np.asarray([0], dtype=np.int64),
        np.asarray([nav.size], dtype=np.int64),
        codes,
        primary,
        secondary,
        risk_free,
        output,
        statuses,
    )
    service = CustomIndicatorService(tmp_path, tmp_path)
    # This test owns its explicit compile phase and must not depend on tests
    # executed earlier having populated the process-wide warm cache.
    service.warm_numba_plans()
    elapsed_days = float(nav.size - 1)
    for index, definition in enumerate(definitions):
        runtime = service._compile_runtime(definition, "1Y")
        context = {
            "returns": returns,
            "log_returns": np.log1p(returns),
            "adjusted_nav": nav,
            "volume": volume,
            "market_high": market_high,
            "market_low": market_low,
            "observation_count": float(returns.size),
            "window_elapsed_days": elapsed_days,
            **service._risk_free_context(definition, elapsed_days),
        }
        expected = float(runtime.compute(context))
        assert int(statuses[0, index]) == 0, definition["id"]
        assert output[0, index] == pytest.approx(
            expected, rel=1e-10, abs=1e-12
        ), definition["id"]


def test_path_metrics_use_nav_path_and_count_only_strict_new_highs() -> None:
    nav = np.ascontiguousarray([1.10, 1.10, 0.99, 1.10, 1.21, 1.21])
    values = np.ascontiguousarray(nav.reshape(1, -1), dtype=np.float64)
    metric_ids = (
        "builtin-maximum-drawdown-v2",
        "builtin-ulcer-index-v2",
        "builtin-new-high-ratio-v2",
    )
    codes = np.asarray(
        [BUILTIN_METRIC_CODE[indicator_id] for indicator_id in metric_ids],
        dtype=np.int64,
    )
    output = np.full((1, len(metric_ids)), np.nan, dtype=np.float64)
    statuses = np.full((1, len(metric_ids)), -1, dtype=np.int16)
    compute_builtin_batch_serial(
        values,
        np.asarray([0], dtype=np.int64),
        np.asarray([nav.size], dtype=np.int64),
        codes,
        np.full(len(metric_ids), -1, dtype=np.int64),
        np.full(len(metric_ids), -1, dtype=np.int64),
        np.zeros(len(metric_ids), dtype=np.float64),
        output,
        statuses,
    )

    drawdowns = nav / np.maximum.accumulate(nav) - 1.0
    assert np.all(statuses == 0)
    assert output[0, 0] == pytest.approx(abs(min(float(np.min(drawdowns)), 0.0)))
    assert output[0, 1] == pytest.approx(
        math.sqrt(float(np.mean(np.minimum(drawdowns, 0.0) ** 2)))
    )
    # 仅首个 1.10 与严格超过历史峰值的 1.21 计数；平峰不重复计数。
    assert output[0, 2] == pytest.approx(2.0 / nav.size)


def test_persistent_worker_computes_over_shared_mmap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("INDICATOR_PROCESS_WORKERS", "1")
    monkeypatch.setenv("INDICATOR_PREWARM_WORKERS", "true")
    engine = AdaptiveComputeEngine(tmp_path / "runtime")
    engine.start()
    try:
        with ExitStack() as stack:
            values = _owner(
                stack,
                np.ascontiguousarray([[1.0, 1.1, 1.21]], dtype=np.float64),
                engine.runtime_dir,
            )
            starts = _owner(stack, np.asarray([0], dtype=np.int64), engine.runtime_dir)
            ends = _owner(stack, np.asarray([3], dtype=np.int64), engine.runtime_dir)
            codes = _owner(stack, np.asarray([0], dtype=np.int64), engine.runtime_dir)
            indices = _owner(stack, np.asarray([-1], dtype=np.int64), engine.runtime_dir)
            risk_free = _owner(
                stack, np.asarray([0.0], dtype=np.float64), engine.runtime_dir
            )
            output = _owner(
                stack, np.full((1, 1), np.nan, dtype=np.float64), engine.runtime_dir
            )
            statuses = _owner(
                stack, np.full((1, 1), -1, dtype=np.int16), engine.runtime_dir
            )
            outcome = engine.run_builtin_shared(
                values=values.descriptor,
                starts=starts.descriptor,
                ends=ends.descriptor,
                codes=codes.descriptor,
                primary_indices=indices.descriptor,
                secondary_indices=indices.descriptor,
                risk_free=risk_free.descriptor,
                output=output.descriptor,
                statuses=statuses.descriptor,
                parallel=False,
                thread_budget=1,
            )
            assert outcome["worker_pid"] != 0
            assert output.descriptor.backend == "mmap"
            assert np.asarray(output.view())[0, 0] == pytest.approx(0.21)
            assert int(np.asarray(statuses.view())[0, 0]) == 0
    finally:
        engine.close()


def test_service_uses_one_shared_dependency_bundle_for_multiple_periods(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("INDICATOR_PROCESS_WORKERS", "2")
    monkeypatch.setenv("INDICATOR_PREWARM_WORKERS", "true")
    monkeypatch.setenv("INDICATOR_SHM_THRESHOLD_BYTES", "1")
    dates = pd.bdate_range("2022-01-03", periods=800)
    products = ["510001.SH", "510002.SH"]
    pd.DataFrame(
        [{"ts_code": product, "name": product} for product in products]
    ).to_parquet(tmp_path / "etf_info_df.parquet", index=False)
    pd.DataFrame(
        [
            {
                "ts_code": product,
                "date": date,
                "adj_nav": (1.0 + 0.0005 + product_index * 0.0001) ** date_index,
            }
            for product_index, product in enumerate(products)
            for date_index, date in enumerate(dates)
        ]
    ).to_parquet(tmp_path / "etf_daily_df.parquet", index=False)
    service = CustomIndicatorService(tmp_path, tmp_path)
    plan = service.create_plan(
        {
            "name": "共享区间测试",
            "product_kind": "etf",
            "indicators": [
                {
                    "indicator_id": "builtin-total-return-v2",
                    "period": "1Y",
                    "weight": 50,
                },
                {
                    "indicator_id": "builtin-total-return-v2",
                    "period": "2Y",
                    "weight": 50,
                },
            ],
            "targets": [
                {"kind": "etf", "product_id": product} for product in products
            ],
            "missing_policy": "strict",
        }
    )
    service.start_compute_engine()
    try:
        startup = service.meta()["numeric_backend"]["startup_warmup"]
        assert startup["complete"] is True
        assert startup["single_metric_batch_plans"] >= 35
        assert startup["workers"]["fully_warmed"] is True

        def reject_request_time_compile(*_args, **_kwargs):
            raise AssertionError("request path attempted NJIT compilation")

        monkeypatch.setattr(
            "custom_indicators.service.compile_numba_batch_plan",
            reject_request_time_compile,
        )
        preview = service.evaluate(
            indicator_ids=[
                "builtin-total-return-v2",
                "builtin-return-volatility-v2",
            ],
            inline_definition=None,
            targets=[{"kind": "etf", "product_id": product} for product in products],
            period="1Y",
        )
        result = service.run_plan(plan["id"])
    finally:
        service.close_compute_engine()

    assert all(item["value"] is not None for item in preview["results"])
    assert result["ranked_count"] == 2
    assert result["execution"]["execution_lanes"] == {
        "numba_fused": 2,
        "numba_blas": 0,
        "python_fallback": 0,
    }
    assert result["execution"]["typed_batch_fallback"] == 0
    assert result["execution"]["python_operator_calls"] == 0
    assert result["execution"]["worker_processes"] >= 1
    assert all(
        value["value"] is not None
        for row in result["rows"]
        for value in row["values"]
    )

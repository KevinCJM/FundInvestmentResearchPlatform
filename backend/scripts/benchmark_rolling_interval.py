"""Reproducible, offline rolling-scope timing and allocator checks.

Run from backend: NUMBA_NRT_STATS=1 python scripts/benchmark_rolling_interval.py
Numerical timings exclude compilation. RSS is the process high-water mark,
including compilation, not a claim about the live size of one output array.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import resource
import statistics
import sys
import time

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[2]), str(Path(__file__).resolve().parents[1])]
from cal_indicators.typed_dsl import compose_typed_series_bundle
from cal_indicators.typed_numba_plan import compile_numba_series_plan
from custom_indicators.variable_registry import variable_types

CASES = {
    "mean": "mean(returns)",
    "std": "std(returns,1)",
    "maximum_drawdown": "-min_value(drawdown_series(adjusted_nav))",
    "sharpe": "(mean(returns)-risk_free_rate_per_observation)/std(returns,1)*sqrt(periods_per_year)",
    "calmar": "(product(returns+1)**(periods_per_year/observation_count)-1)/(-min_value(drawdown_series(adjusted_nav)))",
    "cvar": "-mean_where(returns,less_equal(returns,quantile(returns,0.05)))",
    "custom": "std(returns,1)-min_value(drawdown_series(adjusted_nav))",
}


def context(size: int) -> dict:
    grid = np.arange(size, dtype=np.float64)
    nav = 100.0 + grid * 0.005 + np.sin(grid * 0.23) * 2.0
    returns = np.empty(size, dtype=np.float64)
    returns[0] = np.nan
    returns[1:] = nav[1:] / nav[:-1] - 1
    return {
        "adjusted_nav": nav, "market_close": nav,
        "returns": returns, "observation_dates": 20000.0 + grid,
        "annual_risk_free_rate_decimal": 0.015, "periods_per_year": 252.0,
        "risk_free_rate_per_observation": 1.015 ** (1 / 252) - 1,
        "observation_count": float(size - 1), "window_elapsed_days": float(size - 1),
        "risk_free_return_window": 0.0,
    }


def live_allocations() -> int | None:
    from numba.core.runtime import rtsys
    try:
        stats = rtsys.get_allocation_stats()
    except RuntimeError:
        return None
    return int(stats.alloc - stats.free)


def prepare(body: str, width: int):
    plan = compose_typed_series_bundle({"value": f"rolling_apply({body},{width})"},
                                      variable_types=variable_types("single_product", "2.4.0"))
    return compile_numba_series_plan(plan)


def memory_check(repeats: int = 40) -> dict[str, int]:
    """Test successful and exceptional windows in the actual frozen NJIT path."""
    data = context(128)
    scenarios = [
        ("success", "std(market_close,1)", {}),
        ("zero_denominator", "mean(market_close+1)/std(market_close,1)", {"market_close": np.ones(128)}),
        ("array_division_error", "mean((market_close+1)/(market_close-market_close))", {}),
        ("array_log_error", "mean(log(market_close/market_close-2))", {}),
        ("invalid_path", "-min_value(drawdown_series(adjusted_nav))", {"adjusted_nav": -np.ones(128)}),
        ("invalid_dates", "mean(market_close)", {"observation_dates": np.ones(128)}),
    ]
    results = {}
    for name, body, overrides in scenarios:
        compiled = prepare(body, 5)
        values = {**data, **overrides}
        args = tuple(values[key] for key in compiled.context_names)
        def run():
            try:
                output = compiled.compute(args)
                assert len(output) == 1
            except ValueError as exc:
                assert name == "invalid_dates" and "DATE_AXIS" in str(exc)
        run()
        gc.collect()
        before = live_allocations()
        if before is None:
            raise RuntimeError("Memory verification requires NUMBA_NRT_STATS=1 before Python startup")
        for _ in range(repeats):
            run()
        gc.collect()
        delta = live_allocations() - before
        results[name] = delta
        if delta != 0:
            raise AssertionError(f"{name}: {delta} retained NRT allocations after {repeats} runs")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=4096)
    parser.add_argument("--window", type=int, default=63)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--memory-only", action="store_true")
    args = parser.parse_args()
    if args.size < 2 or not 1 <= args.window <= 5000 or args.repeats < 1:
        parser.error("size >= 2, window in 1..5000, repeats >= 1 are required")
    report = {"fixture": "deterministic_local_arrays", "size": args.size, "window": args.window,
              "repeats": args.repeats, "timings_exclude_compilation": True, "cases": []}
    if not args.memory_only:
        values = context(args.size)
        for name, body in CASES.items():
            started = time.perf_counter()
            compiled = prepare(body, args.window)
            preparation_ms = (time.perf_counter() - started) * 1000
            inputs = tuple(values[key] for key in compiled.context_names)
            compiled.compute(inputs)
            signatures = tuple(compiled.dispatcher.signatures)
            before = live_allocations()
            timings = []
            for _ in range(args.repeats):
                started = time.perf_counter()
                output = compiled.compute(inputs)
                timings.append((time.perf_counter() - started) * 1000)
                assert output[0].shape == (args.size,)
                del output
            gc.collect()
            after = live_allocations()
            assert tuple(compiled.dispatcher.signatures) == signatures
            assert not compiled.dispatcher._can_compile
            report["cases"].append({"name": name, "prepare_ms": round(preparation_ms, 3),
                "median_ms": round(statistics.median(timings), 3), "min_ms": round(min(timings), 3),
                "result_bytes": args.size * 8, "validity_prefix_bytes": (args.size + 1) * 8,
                "retained_nrt_allocations": None if before is None else after - before,
                "request_time_compilation": 0, "python_fallback": 0})
    if os.getenv("NUMBA_NRT_STATS") == "1":
        report["exception_memory_check"] = memory_check()
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report["process_peak_rss_mib_including_compilation"] = round(peak / (1024**2 if sys.platform == "darwin" else 1024), 2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

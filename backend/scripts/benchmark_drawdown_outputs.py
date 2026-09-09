"""Compare shared execution with independently executing the same scalar metrics.

Offline fixture only. Compilation is excluded from timings; this script writes
no market data, definitions, benchmark baseline or source-code copies.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[2]), str(Path(__file__).resolve().parents[1])]
from cal_indicators.typed_dsl import compose_typed_expression  # noqa: E402
from cal_indicators.typed_numba_plan import compile_numba_batch_plan  # noqa: E402
from custom_indicators.variable_registry import variable_types  # noqa: E402


def benchmark(products: int, observations: int, repeats: int) -> dict:
    interval = "last_drawdown_interval(drawdown_series(adjusted_nav))"
    start = f"value_at(observation_dates, interval_start({interval}))"
    trough = f"value_at(observation_dates, interval_trough({interval}))"
    sources = ("-min_value(drawdown_series(adjusted_nav))", start, trough, f"days_between({start}, {trough})")
    plans = tuple(compose_typed_expression(source, variable_types=variable_types("single_product")) for source in sources)
    columns = ("adjusted_nav", "observation_dates")
    shared = compile_numba_batch_plan(plans, ({},) * len(plans), columns)
    singles = [compile_numba_batch_plan((plan,), ({},), columns) for plan in plans]
    random = np.random.default_rng(719)
    nav = np.cumprod(1.0 + random.normal(0.0002, 0.008, (products, observations)), axis=1)
    values = np.ascontiguousarray([nav.ravel(), np.tile(np.arange(observations, dtype=np.float64) + 19000., products)])
    starts = np.arange(products, dtype=np.int64) * observations
    ends = starts + observations
    elapsed = np.full(products, observations - 1., dtype=np.float64)
    output = np.full((products, len(plans)), np.nan)
    statuses = np.full(output.shape, -1, dtype=np.int16)
    expected = np.full_like(output, np.nan)
    expected_status = np.full_like(statuses, -1)
    single_outputs = [np.full((products, 1), np.nan) for _ in plans]
    single_statuses = [np.full((products, 1), -1, dtype=np.int16) for _ in plans]
    signature_before = tuple(shared.serial_dispatcher.signatures)

    def run_shared():
        shared.compute(values, starts, ends, elapsed, output, statuses, parallel=False)

    def run_independent():
        for index, plan in enumerate(singles):
            plan.compute(values, starts, ends, elapsed, single_outputs[index], single_statuses[index], parallel=False)

    run_shared()
    run_independent()
    for index in range(len(plans)):
        expected[:, index] = single_outputs[index][:, 0]
        expected_status[:, index] = single_statuses[index][:, 0]
    np.testing.assert_allclose(output, expected, rtol=1e-12, atol=1e-12, equal_nan=True)
    np.testing.assert_array_equal(statuses, expected_status)

    def timed(function):
        measured = []
        for _ in range(repeats):
            before = time.perf_counter()
            function()
            measured.append(time.perf_counter() - before)
        return min(measured)

    shared_seconds = timed(run_shared)
    independent_seconds = timed(run_independent)
    assert signature_before == tuple(shared.serial_dispatcher.signatures)
    return {
        "products": products, "observations": observations, "independent_metrics": len(plans),
        "shared_seconds": shared_seconds, "separate_seconds": independent_seconds,
        "ratio_separate_over_shared": independent_seconds / shared_seconds,
        "shared_operator_call_sites": shared.metadata()["operator_call_sites"],
        "separate_drawdown_call_sites": sum(item.metadata()["operator_call_sites"].get("drawdown_series", 0) for item in singles),
        "numerical_parity": True, "request_time_compilation": 0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--products", type=int, default=100)
    parser.add_argument("--observations", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if not 1 <= args.products <= 1000 or not 2 <= args.observations <= 5000 or not 1 <= args.repeats <= 100:
        parser.error("products=1..1000, observations=2..5000, repeats=1..100 required")
    print(json.dumps(benchmark(args.products, args.observations, args.repeats), indent=2))

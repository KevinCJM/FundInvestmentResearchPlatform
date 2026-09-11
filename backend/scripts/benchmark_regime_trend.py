"""Repeatable synthetic 10,000-observation benchmark; excludes import/compilation."""
import json
from pathlib import Path
import sys
import timeit

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from historical_regimes.trend_numba import (  # noqa: E402
    TREND_KERNELS, kama_kernel, super_smoother_kernel,
    trend_features_kernel, trend_regime_kernel,
)
from historical_regimes.peak_trough_numba import (  # noqa: E402
    PEAK_TROUGH_KERNELS, peak_trough_asymmetric_kernel, peak_trough_sideways_kernel,
)

from historical_regimes.segment_numba import SEGMENT_KERNELS, local_extrema_kernel, between_pivots_kernel, interval_statistic_kernel, range_threshold_kernel


def main():
    prices = np.ascontiguousarray(np.log(100) + np.random.default_rng(42).normal(0.0002, 0.01, 10_000).cumsum())
    kernels = {**TREND_KERNELS, **PEAK_TROUGH_KERNELS, **SEGMENT_KERNELS}
    signatures = {name: list(kernel.signatures) for name, kernel in kernels.items()}
    assert all(kernel.nopython_signatures and not kernel._can_compile for kernel in kernels.values())

    def run(use_kama):
        trend = kama_kernel(prices, 60, 2, 126) if use_kama else super_smoother_kernel(prices, 126)
        evidence = trend_features_kernel(prices, trend, 60, 20, 60, 0.0001, 5, 0.2, 0.08)
        return trend_regime_kernel(*evidence[:3], 1.0, 0.1, 0.05, 0.25, 3)

    timings = {}
    for use_kama, name in ((False, "super_smoother_pipeline"), (True, "kama_pipeline")):
        run(use_kama)
        timings[name] = [round(seconds * 1000, 3) for seconds in timeit.repeat(lambda: run(use_kama), number=1, repeat=5)]
    raw_prices = np.ascontiguousarray(np.exp(prices))
    def peak_pipeline():
        base = peak_trough_asymmetric_kernel(raw_prices, 8, 2, 4, 16, 6, 1, .2)
        return peak_trough_sideways_kernel(raw_prices, base[0], base[2], base[3], 1, 2, 1, .03, .06, .25, 20)

    peak_pipeline()
    timings["peak_trough_pipeline"] = [round(seconds * 1000, 3) for seconds in timeit.repeat(
        peak_pipeline, number=1, repeat=5)]
    upper, lower = np.full(raw_prices.size, .03), np.full(raw_prices.size, -.03)
    def independent_pipeline():
        pivots, _ = local_extrema_kernel(raw_prices, 8, 8, 6, 6)
        starts, ends = between_pivots_kernel(pivots)
        change = interval_statistic_kernel(raw_prices, starts, ends, 0, 1)
        return range_threshold_kernel(change, upper, lower)
    independent_pipeline()
    timings["independent_peak_pipeline"] = [round(seconds * 1000, 3) for seconds in timeit.repeat(independent_pipeline, number=1, repeat=5)]
    assert signatures == {name: list(kernel.signatures) for name, kernel in kernels.items()}
    print(json.dumps({"observations": len(prices), "milliseconds": timings,
                      "request_time_compilation": 0, "python_fallback": 0}, indent=2))


if __name__ == "__main__":
    main()

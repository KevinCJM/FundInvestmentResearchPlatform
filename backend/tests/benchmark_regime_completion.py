"""Synthetic bounded diagnostics benchmark; excludes I/O, JSON and graph fitting."""

import json
import resource
import statistics
import time
import tracemalloc
import numpy as np
from historical_regimes.reliability import kernels
from historical_regimes.reliability.bootstrap import bootstrap_kernel
from historical_regimes.reliability.diagnostic_kernels import (
    quality_kernel,
    compare_kernel,
    evidence_kernel,
)


def main():
    kernels.warm()
    n = 20000
    owner = np.arange(n * 2, dtype=np.int64) // 60 % 3
    y = owner[::2]
    predictions = np.roll(owner, 4)
    pred = predictions[::2]
    probability_owner = np.eye(3)[predictions] * 0.7 + 0.1
    q = probability_owner[::2]
    price_owner = np.linspace(100.0, 200.0, n * 2)
    price = price_owner[::2]
    base = np.full(3, 1 / 3)
    for a in (y, pred, q, price, base):
        a.flags.writeable = False
    before = kernels.audit()
    copies = [a.copy() for a in (y, pred, q, price)]
    traces = []
    tracemalloc.start()
    for _ in range(5):
        start = time.perf_counter()
        quality_kernel(y, price, 3)
        compare_kernel(y, pred, 3)
        evidence_kernel(pred, q)
        bootstrap_kernel(y, pred, q, base, 0.6, 10, 500, 0.95)
        traces.append(time.perf_counter() - start)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    for a, b in zip((y, pred, q, price), copies):
        np.testing.assert_array_equal(a, b)
    assert before == kernels.audit()
    print(
        json.dumps(
            {
                "observations": n,
                "classes": 3,
                "replicates": 500,
                "repetitions": 5,
                "median_seconds": statistics.median(traces),
                "python_traced_peak_bytes": peak,
                "maxrss_bytes_macos": resource.getrusage(
                    resource.RUSAGE_SELF
                ).ru_maxrss,
                "input_views_share_memory": [
                    np.shares_memory(y, owner),
                    np.shares_memory(pred, predictions),
                    np.shares_memory(q, probability_owner),
                    np.shares_memory(price, price_owner),
                ],
                "bootstrap_prefix_bytes": (n + 1) * 10 * 8,
                "signatures_per_kernel": {
                    k.py_func.__name__: len(k.signatures) for k, _ in kernels.KERNELS
                },
                "python_fallback": 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

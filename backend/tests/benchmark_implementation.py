"""Repeatable, offline product-path latency and sampled process RSS measurement."""

import json
import os
import resource
import threading
import time
import numpy as np
import psutil
from backend.pre_investment import path_kernels


def main():
    path_kernels.warm()
    n, months, paths = 3, 37, 2000
    base = np.random.default_rng(7).standard_normal((months * 2, paths, n))
    draws = base[::2]
    draws.flags.writeable = False
    before = base.copy()
    drift, loading = path_kernels.monthly_joint_lognormal_kernel(
        np.array([0.06, 0.02, 0.0]), np.diag([0.04, 0.0025, 0.0])
    )
    args = (
        draws,
        drift,
        loading,
        np.array([500000.0, 200000.0, 100000.0]),
        np.array([0.625, 0.25, 0.125]),
        np.array([0.0005, 0.0005, 0.0]),
        np.array([0.0005, 0.0005, 0.0]),
        np.array([0, 0, 1], dtype=np.int64),
        np.zeros(months),
        np.full(months, 1000.0),
        100000.0,
        0.002,
        1,
    )
    process = psutil.Process(os.getpid())
    baseline = process.memory_info().rss
    peak = [baseline]
    stopped = threading.Event()

    def sample():
        while not stopped.wait(0.002):
            peak[0] = max(peak[0], process.memory_info().rss)

    monitor = threading.Thread(target=sample)
    monitor.start()
    timings = []
    first = None
    try:
        for _ in range(3):
            start = time.perf_counter()
            result = path_kernels.product_funding_paths_kernel(*args)
            timings.append(time.perf_counter() - start)
            if first is None:
                first = result
            else:
                for a, b in zip(first, result, strict=True):
                    np.testing.assert_array_equal(a, b)
    finally:
        stopped.set()
        monitor.join()
    np.testing.assert_array_equal(base, before)
    print(
        json.dumps(
            {
                "periods": months,
                "paths": paths,
                "products": n,
                "seconds": timings,
                "draw_view_bytes": draws.nbytes,
                "view_shares_memory": np.shares_memory(base, draws),
                "inputs_unchanged": True,
                "deterministic": True,
                "rss_baseline_bytes": baseline,
                "sampled_rss_peak_bytes": peak[0],
                "sample_interval_ms": 2,
                "incremental_sampled_rss_bytes": peak[0] - baseline,
                "process_maxrss_native_units": resource.getrusage(
                    resource.RUSAGE_SELF
                ).ru_maxrss,
                "execution": path_kernels.audit(),
                "note": "RSS includes process runtime; 2ms sampling is observational, not an allocator-level bound.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

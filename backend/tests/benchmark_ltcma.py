"""Repeatable offline LTCMA kernel timing and allocation observation.

Run with PYTHONPATH=.:backend python backend/tests/benchmark_ltcma.py.
Compilation, data generation and reference setup are excluded from timing.
tracemalloc is Python/NumPy-tracked memory, not a complete native-heap profiler.
"""
from __future__ import annotations

import json
import resource
import statistics
import sys
import time
import tracemalloc

import numpy as np

from backend.strategic_allocation import cma_statistical_kernels as kernels
from backend.strategic_allocation import reference_evidence_kernels


def main() -> None:
    reference_evidence_kernels.warm()
    kernels.warm()
    base = np.random.default_rng(271828).normal(.0002, .01, size=(10000, 60))
    base.flags.writeable = False
    panel = base[:, ::2]
    assert np.shares_memory(base, panel) and not panel.flags.writeable
    signatures = {k.__name__: list(k.signatures) for k in kernels.KERNELS}
    fingerprint = panel.tobytes()
    mean, covariance, _, _ = kernels.historical_estimate(panel, .1)
    prior = kernels.recenter_niw_prior(mean, covariance, 20., 30.)
    states = np.arange(panel.shape[0], dtype=np.int64) % 3
    states.flags.writeable = False
    operations = {
        "historical": lambda: kernels.historical_estimate(panel, .1),
        "niw": lambda: kernels.niw_update(panel, *prior),
        "state_statistics": lambda: kernels.conditional_state_moments(panel, states, 3, 0.),
    }
    result = {}
    for name, operation in operations.items():
        timings = []
        for _ in range(5):
            start = time.perf_counter()
            output = operation()
            timings.append((time.perf_counter() - start) * 1000)
            del output
        tracemalloc.start()
        output = operation()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        del output
        result[name] = {"median_ms": round(statistics.median(timings), 3),
                        "peak_traced_bytes": peak}
    assert panel.tobytes() == fingerprint
    assert signatures == {k.__name__: list(k.signatures) for k in kernels.KERNELS}
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(json.dumps({"shape": list(panel.shape), "input_view_shares_memory": True,
        "input_readonly": True, "request_signature_growth": 0, "repeats": 5,
        "process_peak_rss_bytes_including_startup": int(peak_rss * (1 if sys.platform == "darwin" else 1024)),
        "results": result, "python_fallback": kernels.execution_audit()["python_fallback"],
        "limitations": "Timings exclude startup. Traced peak is not total native allocation. RSS is process lifetime high-water mark, not per-kernel usage."}, indent=2))


if __name__ == "__main__":
    main()

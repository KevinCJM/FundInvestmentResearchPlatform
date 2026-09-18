"""Run explicitly: PYTHONPATH=.:backend python backend/tests/benchmark_risk_scale.py.

Synthetic deterministic inputs only. Reports import/JIT, explicit warm, full
frontier (inclusive endpoints), 2-point endpoint request, segmentation, and RSS.
No formal performance promise; source matrices are never copied per grid point.
"""
from __future__ import annotations

import gc
import json
import os
import platform
import resource
import time

import numpy as np
import psutil

_PROCESS = psutil.Process()
_BASE_RSS = _PROCESS.memory_info().rss
_IMPORT_START = time.perf_counter()
from backend import frontier_moments as fm
from backend.strategic_allocation import risk_scale_kernels as rk
_IMPORT_SECONDS = time.perf_counter()-_IMPORT_START


def run(repeats=10):
    started = time.perf_counter()
    fm.warm(); rk.warm()
    warm_seconds = time.perf_counter()-started
    signatures = {k.__name__: tuple(k.signatures) for k in (*fm.DISPATCHERS, *rk.KERNELS)}
    rows = []
    for n in [5, 10, 30]:
        for count in [101, 200]:
            for psd in [False, True]:
                rng = np.random.default_rng(916+n)
                factors = rng.normal(size=(n, n))
                sigma = factors@factors.T*.001+np.eye(n)*.001
                if psd:
                    sigma[0, :] = 0.; sigma[:, 0] = 0.
                mu = np.linspace(.015, .15, n)
                bounds = np.tile([0., 1.], (n, 1))
                groups = np.zeros((2, n)); groups[:, :2] = 1.
                lows, highs = np.zeros(2), np.ones(2)
                args = [mu, sigma, bounds, groups, lows, highs]
                # Physically noncontiguous readonly views, including covariance.
                for i, a in enumerate(args):
                    owner = np.zeros(tuple(2*s for s in a.shape))
                    view = owner[tuple(slice(None, None, 2) for _ in a.shape)]
                    view[...] = a; view.flags.writeable = False
                    assert np.shares_memory(owner, view)
                    args[i] = view
                start = time.perf_counter()
                fm.frontier_constraints_kernel(*args)
                matrix_ms = (time.perf_counter()-start)*1000
                start = time.perf_counter()
                endpoints = fm.solve_frontier(*args, point_count=2)
                endpoint_ms = (time.perf_counter()-start)*1000
                assert np.all(endpoints[8] == 0), endpoints[8]
                elapsed, segment, resident = [], [], []
                for _ in range(repeats):
                    start = time.perf_counter()
                    result = fm.solve_frontier(*args, point_count=count)
                    elapsed.append((time.perf_counter()-start)*1000)
                    assert result[11] == 0 and np.all(result[3] == 0), (n, count, psd, result[8])
                    start = time.perf_counter()
                    segmentation = rk.segment_frontier(result[2], result[3])
                    segment.append((time.perf_counter()-start)*1000)
                    assert segmentation["status"] == 0, segmentation["status"]
                    resident.append(_PROCESS.memory_info().rss)
                rows.append({"N": n, "G": count, "PSD_cash": psd, "readonly_strided": True,
                             "groups": 2, "repeats": repeats, "matrix_validation_ms": matrix_ms,
                             "endpoint_request_ms": endpoint_ms,
                             "frontier_median_ms": float(np.median(elapsed)), "frontier_max_ms": max(elapsed),
                             "segment_median_ms": float(np.median(segment)),
                             "rss_first_mib": resident[0]/2**20, "rss_last_mib": resident[-1]/2**20,
                             "rss_max_mib": max(resident)/2**20,
                             "result_array_bytes": sum(a.nbytes for a in result if isinstance(a, np.ndarray)),
                             "point_iterations_max": int(np.max(result[4])),
                             "fallback_reason": segmentation["fallback_reason"]})
                del result, segmentation
                gc.collect()
    # Portraits consume already constructed portfolio returns; no implied replay.
    portrait = []
    for t in [252, 2520, 10000]:
        returns = np.random.default_rng(t).normal(.0002, .01, t)
        start = time.perf_counter()
        for _ in range(repeats):
            rk.historical_tail_kernel(returns, .95, 5.)
            rk.historical_drawdown_kernel(returns)
        portrait.append({"T": t, "tail_and_drawdown_mean_ms": (time.perf_counter()-start)*1000/repeats})
    assert signatures == {k.__name__: tuple(k.signatures) for k in (*fm.DISPATCHERS, *rk.KERNELS)}
    return {"platform": platform.platform(), "python": platform.python_version(), "pid": os.getpid(),
            "import_including_eager_jit_seconds": _IMPORT_SECONDS, "explicit_warm_seconds": warm_seconds,
            "rss_before_import_mib": _BASE_RSS/2**20, "rss_end_mib": _PROCESS.memory_info().rss/2**20,
            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(2**20 if platform.system() == "Darwin" else 1024),
            "request_new_signatures": 0, "python_fallback": 0, "rows": rows, "portraits": portrait,
            "notes": "Full frontier timing includes readiness audit and endpoints; endpoint request is a separate 2-point run. RSS includes Python/JIT/BLAS. No I/O, persistence, proxy construction or M2 measured."}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))

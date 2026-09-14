"""Reproducible M3 allocation/latency probe; no files or network are used."""
import json
import resource
import time
import tracemalloc
import numpy as np
from backend.tactical_allocation import numeric
from backend.tactical_allocation.contracts import DecisionPolicy


def main():
    numeric.warm_tactical_allocation_kernels()
    owner = np.zeros((4000, 16))
    values = owner[::2, ::2]
    values.setflags(write=False)
    n, a = values.shape
    probabilities = np.ones((n, 1))
    flags = np.ones(n, dtype=np.uint8)
    decisions = np.zeros(n, dtype=np.uint8)
    decisions[::21] = 1
    tilts = np.zeros((1, a)); tilts[0, 0], tilts[0, 1] = .05, -.05
    base = np.full(a, 1.0 / a)
    clock = {'decisions': decisions, 'executions': flags}
    policy = DecisionPolicy()
    times = []
    tracemalloc.start()
    for _ in range(5):
        started = time.perf_counter()
        path = numeric._checked_path(values, probabilities, flags, tilts, base, 10., clock, policy)
        times.append(time.perf_counter() - started)
    _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    print(json.dumps({'shape': list(values.shape), 'shares_memory': bool(np.shares_memory(values, owner)),
                      'readonly': not values.flags.writeable, 'strides': list(values.strides),
                      'path_output_bytes': path.nbytes, 'tracemalloc_peak_bytes': peak,
                      'process_maxrss_platform_units': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      'seconds_5_runs': times, 'signatures': len(numeric._taa_recursive_kernel.signatures),
                      'python_fallback': numeric.execution_audit()['python_fallback']}))


if __name__ == '__main__':
    main()

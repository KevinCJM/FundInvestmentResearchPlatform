"""Offline warmed benchmark. Reports allocations honestly; no Python fallback comparison."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import resource
import time

os.environ.setdefault('NUMBA_NRT_STATS', '1')
import numpy as np
from numba.core.runtime import rtsys
from backend.strategic_allocation import goal_kernels as goals, mandate_kernels as numeric, kernels


def run():
    kernels.warm_strategic_kernels()
    goals.warm_goal_kernels()
    numeric.warm()
    metrics = np.column_stack((np.linspace(.015, .09, 101), np.linspace(.01, .20, 101)))
    eligible = np.ones(101, dtype=np.int64)
    flows = np.zeros(120)
    owner, _ = goals.seeded_factor_draws_kernel(120, 2000, 1, 42, 0, 5.)
    draws = owner.view()
    for array in (metrics, eligible, flows, draws):
        array.flags.writeable = False
    signatures = {k.__name__: list(map(str, k.signatures)) for k in numeric.KERNELS}
    readings = []
    expected = None
    for _ in range(3):
        allocations_before = rtsys.get_allocation_stats()
        started = time.perf_counter()
        result, selected = numeric.reference_funding_search_kernel(metrics, eligible, draws, 1_000_000.,
            flows, flows, 1_500_000., .005, .8, 0, 1)
        elapsed = time.perf_counter() - started
        allocations_after = rtsys.get_allocation_stats()
        if expected is not None:
            np.testing.assert_array_equal(result, expected)
        expected = result
        readings.append({'seconds': elapsed, 'selected_node': int(selected), 'output_bytes': result.nbytes,
            'nrt_allocations': allocations_after.alloc - allocations_before.alloc})
    assert np.shares_memory(owner, draws)
    assert signatures == {k.__name__: list(map(str, k.signatures)) for k in numeric.KERNELS}
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {'platform': platform.platform(), 'candidates': 101, 'months': 120, 'paths': 2000,
        'shared_input_bytes': metrics.nbytes + eligible.nbytes + flows.nbytes + draws.nbytes,
        'draws_share_owner_memory': True, 'input_writeable': bool(draws.flags.writeable),
        'runs': readings, 'process_peak_rss_bytes_including_imports_and_warmup': peak if platform.system() == 'Darwin' else peak * 1024,
        'execution': numeric.execution_audit(),
        'limitations': ['Peak RSS includes imports and startup warmup, not just this kernel.',
            'NRT counts native allocations; no zero-allocation claim is made.', 'No candidate-by-path-by-month result cube is allocated.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = json.dumps(run(), indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(result + '\n')
    print(result)

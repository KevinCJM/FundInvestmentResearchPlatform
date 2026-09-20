"""Maximum-axis common solve benchmark; excludes startup, I/O and funding."""
import json
import resource
import sys
import time
import tracemalloc

import numpy as np
from scipy.optimize import linprog

from backend.strategic_allocation import compatibility_kernels as numeric
from backend.strategic_allocation.compatibility_solver import solve


def benchmark():
    numeric.warm()
    rng = np.random.default_rng(20260919)
    means = rng.uniform(.01, .1, (20, 30))
    risks = np.repeat((np.eye(30)*.0025)[None], 20, axis=0)
    means.flags.writeable = risks.flags.writeable = False
    # The .1 risk cap is nonbinding for this bounded simplex. An independent
    # linear solver can therefore verify the objective at the maximum axes.
    reference = linprog(np.r_[np.zeros(30), 1.],
        A_ub=np.column_stack([-means, -np.ones(20)]), b_ub=np.zeros(20),
        A_eq=np.array([np.r_[np.ones(30), 0.]]), b_eq=[1.],
        bounds=[(0., 1.)]*30+[(-1., 1.)], method='highs')
    assert reference.success
    signatures = {k.__name__: list(k.signatures) for k in numeric.KERNELS}
    scale = 1 if sys.platform == 'darwin' else 1024
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*scale
    tracemalloc.start()
    elapsed = []
    for _ in range(3):
        start = time.perf_counter()
        result = solve(means, risks, np.tile([0., 1.], (30, 1)), np.empty((0,30)),
            np.empty(0), np.empty(0), np.empty(0), 0., .1, 1., 0., np.zeros(20), 0)
        elapsed.append(time.perf_counter()-start)
        assert result['status'] == 'converged', result
        np.testing.assert_allclose(result['objective_value'], reference.fun, atol=1e-8)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert signatures == {k.__name__: list(k.signatures) for k in numeric.KERNELS}
    return {'models': 20, 'assets': 30, 'seed': 20260919, 'elapsed_seconds': elapsed,
            'traced_peak_bytes': peak, 'input_bytes': means.nbytes+risks.nbytes,
            'process_peak_growth_bytes': max(0, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*scale-rss_before),
            'master_iterations': result['master_iterations'], 'objective_gap': result['objective_gap'],
            'python_fallback': 0, 'new_request_signatures': 0,
            'scope': 'maximum_axis_joint_maximin_nonbinding_risk_excludes_startup_anchors_funding_io'}


if __name__ == '__main__':
    print(json.dumps(benchmark(), indent=2))

"""Offline reproducible kernel budget measurement, not investment validation.

PYTHONPATH=.:backend python backend/tests/benchmark_regime_reliability.py
"""
import json
import resource
import time
import tracemalloc
import numpy as np
from historical_regimes.reliability import kernels as k


def main():
    k.warm()
    count = 20000
    owner = (np.arange(count*2,dtype=np.int64)//100)%3
    y = owner[::2]
    pred_owner = np.roll(owner,10)
    pred = pred_owner[::2]
    raw_owner = np.eye(3)[pred_owner] * .8 + .2/3
    raw = raw_owner[::2]
    for array in (y,pred,raw):
        array.flags.writeable = False
    baseline = [array.copy() for array in (y,pred,raw)]
    traces = []
    tracemalloc.start()
    for _ in range(5):
        started = time.perf_counter()
        k.classification_kernel(y,pred,3)
        k.events_kernel(y,pred,3)
        q,counts,base,temperature = k.calibrate_kernel(y,pred,raw,12000,1)
        k.probability_kernel(y[12000:],pred[12000:],q[12000:],10)
        traces.append(time.perf_counter()-started)
    _,peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert all(np.array_equal(a,b) for a,b in zip(baseline,(y,pred,raw)))
    print(json.dumps({"observations":count,"classes":3,"repetitions":5,
        "median_seconds":float(np.median(traces)),"python_traced_peak_bytes":peak,
        "process_maxrss_platform_units":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "new_calibrated_output_bytes":q.nbytes,"calibration_small_buffers_bytes":counts.nbytes+base.nbytes,
        "readonly_strided_views_share_memory":[np.shares_memory(y,owner),np.shares_memory(pred,pred_owner),np.shares_memory(raw,raw_owner)],
        "test_block_shares_memory":np.shares_memory(q[12000:],q),
        "signatures_per_kernel":{fn.py_func.__name__:len(fn.signatures) for fn,_ in k.KERNELS},
        "python_fallback":k.audit()["python_fallback"]},indent=2))


if __name__ == "__main__":
    main()

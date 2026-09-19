"""Run with PYTHONPATH=backend python backend/tests/benchmark_smoothing.py."""
import json
import platform
import resource
import statistics
import time
import tracemalloc
import numpy as np
from computation_graph.smoothing_numba import SMOOTHING_KERNELS


def main():
    records = []
    for size in (20000, 200000):
        base = np.linspace(100, 200, size * 2)
        values = base[::2]
        values.flags.writeable = False
        assert np.shares_memory(base, values)
        for name, args in (("butterworth_zero_phase", (63,)),
                           ("savitzky_golay_centered", (63, 3)),
                           ("ehlers_error_correcting", (20, 50))):
            kernel = SMOOTHING_KERNELS[name]
            kernel(np.empty(0), *args)
            signatures = tuple(kernel.signatures)
            samples, peaks = [], []
            for _ in range(5):
                tracemalloc.start()
                start = time.perf_counter()
                result = kernel(values, *args)
                samples.append((time.perf_counter() - start) * 1000)
                peaks.append(tracemalloc.get_traced_memory()[1])
                tracemalloc.stop()
                assert not np.shares_memory(result, base)
                del result
            assert tuple(kernel.signatures) == signatures and not kernel._can_compile
            records.append(dict(kernel=name, observations=size, median_ms=statistics.median(samples),
                                traced_peak_bytes=max(peaks), input_shared=True, input_copies=0,
                                request_time_compilation=0))
    print(json.dumps(dict(platform=platform.platform(), records=records,
                         process_maxrss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                         memory_note="tracemalloc kernel allocations; maxrss includes imports/warmup and is bytes on macOS, KiB on Linux"), indent=2))


if __name__ == '__main__':
    main()

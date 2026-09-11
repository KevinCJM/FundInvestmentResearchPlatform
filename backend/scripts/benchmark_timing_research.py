"""Reproducible synthetic kernel benchmark; never reads or writes market data.

Run from the repository root with PYTHONPATH=backend:.
"""
import argparse
import json
import time
import tracemalloc
import numpy as np
from backend.timing_research.numeric import simulate_kernel, warm_timing_kernels, timing_execution_audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bars", type=int, default=100000)
    args = parser.parse_args()
    if not 1 <= args.bars <= 1000000:
        parser.error("--bars must be between 1 and 1000000")
    warm_timing_kernels()
    owner = np.full(args.bars * 2, 100.0)
    prices = [owner[::2] for _ in range(4)]
    for value in prices:
        value.setflags(write=False)
    entry = np.ones(args.bars, dtype=np.int64); entry.setflags(write=False)
    exit_ = np.zeros(args.bars, dtype=np.int64); exit_.setflags(write=False)
    before = tuple(simulate_kernel.signatures)
    tracemalloc.start()
    started = time.perf_counter()
    path, trades, count, status = simulate_kernel(*prices, entry, exit_, 0, args.bars, 15, 2, 3., 5., .15, .15)
    elapsed = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    assert before == tuple(simulate_kernel.signatures)
    assert all(np.shares_memory(value, owner) for value in prices)
    assert np.all(owner == 100.0)
    print(json.dumps({"bars": args.bars, "seconds": elapsed, "tracemalloc_peak_bytes": peak,
                      "output_bytes": path.nbytes + trades.nbytes, "closed_trades": count, "status": status,
                      "shared_strided_inputs": True, "new_signatures": 0,
                      "python_fallback": timing_execution_audit()["python_fallback"]}, indent=2))


if __name__ == "__main__":
    main()

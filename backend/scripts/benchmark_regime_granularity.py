"""Compare shared compatibility wrappers with editable numeric DAG execution.

Run: PYTHONPATH=.:backend python backend/scripts/benchmark_regime_granularity.py
Measures warmed numerical node execution only, excluding I/O, serialization,
startup and publication. Retained bytes count unique underlying NumPy buffers,
not peak process memory. No production data is read or written.
"""
from __future__ import annotations

import argparse
import copy
import json
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np

from historical_regimes.composite_expansion import expand_composite
from historical_regimes.v2_contracts import parse_definition_v2, validate_definition_v2
from historical_regimes.v2_numba import KERNELS, regime_graph_numba_status
from historical_regimes.v2_service import PortValue, RegimeGraphV2Service, _required_node_ids
from historical_regimes.v2_templates import CLOCK_STATES, MARKET_STATES


def definition(kind):
    inputs = {"value": {"node_id": "data", "port": "value"}}
    if kind == "model.quadrant":
        inputs = {"growth": inputs["value"], "inflation": inputs["value"]}
    return {"schema_version": "2.0", "name": "颗粒度数值基准", "graph": {
        "nodes": [{"id": "data", "type": "source.inline", "inputs": {}, "parameters": {"rows": [
            {"observation_date": "2020-01-01", "available_at": "2020-01-01", "value": 1.}]}},
            {"id": "algorithm", "type": kind, "inputs": inputs,
             "parameters": {"sideways_enabled": True} if kind == "model.peak_trough" else {}}],
        "outputs": {"state": {"node_id": "algorithm", "port": "state"}}, "exposed_node_ids": ["algorithm"]},
        "states": copy.deepcopy(CLOCK_STATES if kind == "model.quadrant" else MARKET_STATES)}


def make_runner(service, raw, value):
    parsed = parse_definition_v2(raw)
    inspection = validate_definition_v2(parsed)
    required = _required_node_ids(parsed)
    nodes = {node.id: node for node in parsed.graph.nodes}
    plan = [nodes[identifier] for identifier in inspection["topological_order"] if identifier in required and identifier != "data"]

    def execute():
        outputs = {"data": {"value": value}}
        for node in plan:
            outputs[node.id] = service._execute_numeric_node(node, outputs, len(parsed.states), "retrospective", None, None, {}, {}, {})
        return outputs
    return execute


def retained_bytes(outputs):
    buffers = {}
    for ports in outputs.values():
        for port in ports.values():
            for array in (port.values, port.dates, port.available):
                while isinstance(array.base, np.ndarray):
                    array = array.base
                buffers[id(array)] = array.nbytes
    return sum(buffers.values())


def measure(execute, repetitions):
    execute()
    times = []
    for _ in range(repetitions):
        start = time.perf_counter()
        result = execute()
        times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times), retained_bytes(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=int, default=20000)
    parser.add_argument("--repetitions", type=int, default=9)
    args = parser.parse_args()
    if not 100 <= args.observations <= 20000 or not 3 <= args.repetitions <= 100:
        parser.error("observations must be 100..20000 and repetitions 3..100")
    assert regime_graph_numba_status()["complete"]
    signatures = {key: list(kernel.signatures) for key, kernel in KERNELS.items()}
    rng = np.random.default_rng(20260909)
    noise = rng.normal(0., .01, args.observations)
    dates = np.arange(args.observations, dtype=np.int64)
    rows = []
    with tempfile.TemporaryDirectory(prefix="regime-granularity-") as directory:
        service = RegimeGraphV2Service(Path(directory), Path(directory))
        for kind in ("model.threshold", "model.quadrant", "model.peak_trough"):
            values = np.ascontiguousarray(100 * np.exp(noise.cumsum()) if kind == "model.peak_trough" else noise)
            value = PortValue(values, dates, dates)
            raw = definition(kind)
            expanded = expand_composite(raw, "algorithm", "retrospective")
            before = make_runner(service, raw, value)
            after = make_runner(service, expanded["definition"], value)
            old, new = before(), after()
            for port, ref in expanded["output_map"].items():
                np.testing.assert_array_equal(old["algorithm"][port].values, new[ref["node_id"]][ref["port"]].values)
            old_ms, old_bytes = measure(before, args.repetitions)
            new_ms, new_bytes = measure(after, args.repetitions)
            rows.append({"algorithm": kind, "wrapper_median_ms": round(old_ms, 4), "expanded_median_ms": round(new_ms, 4),
                         "wrapper_retained_bytes": old_bytes, "expanded_retained_bytes": new_bytes,
                         "outputs_exactly_equal": True})
    assert signatures == {key: list(kernel.signatures) for key, kernel in KERNELS.items()}
    print(json.dumps({"observations": args.observations, "repetitions": args.repetitions,
                      "scope": "warmed_numeric_node_dispatch_no_io_or_json", "request_time_compilation": 0,
                      "python_fallback": 0, "results": rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

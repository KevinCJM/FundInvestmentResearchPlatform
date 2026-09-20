"""Reproducible maximum-axis moments/diagnostics benchmark, with no external data.

Run: PYTHONPATH=.:backend python backend/tests/test_multi_cma_performance.py
Startup/JIT cache loading and funding simulation are excluded from the timing.
"""
from __future__ import annotations

import json
import resource
import sys
import time
import tracemalloc
import numpy as np

from backend.sensitivity.repository import digest_json
from backend.strategic_allocation import cma_model_kernels, multi_cma_kernels, kernels
from backend.strategic_allocation.multi_cma import cross_model_results


def benchmark(repeats=3):
    kernels.warm_strategic_kernels()
    cma_model_kernels.warm()
    multi_cma_kernels.warm()
    rng = np.random.default_rng(20260919)
    count, assets, candidates = 20, 30, 5
    means = rng.uniform(.01, .09, (count, assets))
    raw = rng.normal(0., .003, (count, assets, assets))
    risks = np.stack([value @ value.T + np.eye(assets) * .002 for value in raw])
    widths = rng.uniform(.001, .015, (count, assets))
    probabilities = np.full(count, 1 / count)
    names = [f"asset-{i}" for i in range(assets)]
    weights = rng.dirichlet(np.ones(assets), size=candidates)
    sources = []
    for m in range(count):
        definition = {"model": None, "assets": [dict(id=names[i], annual_return=float(means[m, i]),
                        mean_uncertainty=float(widths[m, i])) for i in range(assets)]}
        artifact = {"id": f"series-{m}", "definition": definition, "covariance": risks[m].tolist()}
        artifact["content_hash"] = digest_json(artifact)
        sources.append({"cma_id": artifact["id"], "content_hash": artifact["content_hash"], "name": f"model-{m}",
                        "weight": float(probabilities[m]), "definition": definition, "assumptions": definition,
                        "covariance": artifact["covariance"], "artifact": artifact})
    multi = {"mode": "parameter_average", "aggregation_semantics": "parameter_average", "sources": sources,
             "refs": [{"cma_id": s["cma_id"], "content_hash": s["content_hash"], "weight": s["weight"]} for s in sources],
             "assumptions": sources[0]["assumptions"]}
    multi["content_hash"] = digest_json(multi)
    mandate = {"risk_aversion": 5., "max_volatility": .3, "max_tracking_error": .1,
               "objective_kind": "absolute_return", "target_return": 0.}
    for value in (means, risks, widths, probabilities):
        value.flags.writeable = False
    signatures = {k.__name__: list(k.signatures) for k in (cma_model_kernels.mixture_moments_kernel,
                  multi_cma_kernels.weighted_half_width_kernel, kernels.portfolio_moments_kernel)}
    durations = []
    rss_scale = 1 if sys.platform == "darwin" else 1024
    process_peak_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * rss_scale
    tracemalloc.start()
    for _ in range(repeats):
        start = time.perf_counter()
        mean, _, covariance, disagreement = cma_model_kernels.mixture_moments_kernel(probabilities, means, risks, False)
        uncertainty = multi_cma_kernels.weighted_half_width_kernel(probabilities, widths)
        rows = [cross_model_results(multi, dict(zip(names, w.tolist(), strict=True)), mandate) for w in weights]
        durations.append(time.perf_counter() - start)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    process_peak_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * rss_scale
    assert all(len(row) == count for row in rows)
    np.testing.assert_allclose(mean, probabilities @ means)
    np.testing.assert_allclose(covariance, np.einsum("m,mij->ij", probabilities, risks))
    np.testing.assert_allclose(uncertainty, probabilities @ widths)
    assert all(signatures[k.__name__] == list(k.signatures) for k in (cma_model_kernels.mixture_moments_kernel,
               multi_cma_kernels.weighted_half_width_kernel, kernels.portfolio_moments_kernel))
    return {"models": count, "assets": assets, "candidates": candidates, "repeats": repeats,
            "seed": 20260919, "elapsed_seconds": durations, "traced_peak_bytes": peak,
            "process_peak_rss_before_bytes": process_peak_before,
            "process_peak_rss_after_bytes": process_peak_after,
            "process_peak_growth_bytes": max(0, process_peak_after - process_peak_before),
            "moment_input_bytes": sum(v.nbytes for v in (probabilities, means, risks, widths)),
            "scope": "frozen_lineage_checks_e2_moments_and_cross_diagnostics_excludes_startup_and_funding",
            "python_fallback": 0, "new_request_signatures": 0}


def test_maximum_axis_benchmark_matches_references():
    result = benchmark(repeats=1)
    assert result["models"] == 20 and result["assets"] == 30 and result["candidates"] == 5
    assert result["traced_peak_bytes"] > 0


if __name__ == "__main__":
    print(json.dumps(benchmark(), ensure_ascii=False, indent=2))

"""Run one reproducible local benchmark against a persisted evaluation plan."""

from __future__ import annotations

import argparse
import json
import time

try:
    # CustomIndicatorService uses the top-level backend modules when this CLI is
    # executed from backend/.  Import the same registry instance so its warmup
    # proof cannot be split across duplicate module names.
    from cal_indicators.typed_numba_kernels import (
        get_numba_kernel_registry,
        kernel_registry_status,
    )
    from compute_policy import ComputePolicyError, validate_execution_audit
    from custom_indicators.service import CustomIndicatorService
except ModuleNotFoundError:  # pragma: no cover - backend/ direct execution
    from backend.cal_indicators.typed_numba_kernels import (
        get_numba_kernel_registry,
        kernel_registry_status,
    )
    from backend.compute_policy import ComputePolicyError, validate_execution_audit
    from backend.custom_indicators.service import CustomIndicatorService


def validate_benchmark_run_execution(result: dict) -> dict:
    """Fail the benchmark when the measured run cannot prove NJIT execution."""

    execution = result.get("execution") or {}
    lanes = execution.get("execution_lanes") or {}
    if (
        lanes.get("python_fallback") != 0
        or execution.get("typed_batch_fallback") != 0
        or execution.get("python_operator_calls") != 0
    ):
        raise ComputePolicyError("指标基准测试检测到 Python 数值回退")
    registry = get_numba_kernel_registry()
    registry_status = kernel_registry_status()
    signatures = {
        operator_id: list(spec.compiled_signatures)
        for operator_id, spec in registry.items()
    }
    return validate_execution_audit(
        {
            "engine": str(execution.get("engine_version") or "typed-indicator-benchmark"),
            "backend": "numba_njit_fixed_signature",
            "kernel_version": str(
                execution.get("numeric_kernel_version")
                or registry_status.get("kernel_version")
                or "unknown"
            ),
            "kernel_signatures": signatures,
            "nopython": bool(registry_status.get("warmed"))
            and all(bool(values) for values in signatures.values()),
            "python_fallback": 0,
            "compiled_plan_ids": list(execution.get("compiled_plan_ids") or []),
            "typed_batch_fallback": 0,
            "python_operator_calls": 0,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("plan_id")
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--repeat", type=int, default=2)
    arguments = parser.parse_args()

    service = CustomIndicatorService()
    service.start_compute_engine()
    try:
        runs = []
        for index in range(max(1, arguments.repeat)):
            if index == 0:
                service.plan_cache.clear()
            started = time.perf_counter()
            result = service.run_plan(arguments.plan_id, arguments.as_of)
            policy_audit = validate_benchmark_run_execution(result)
            runs.append(
                {
                    "elapsed_seconds": round(time.perf_counter() - started, 6),
                    "ranked_count": result.get("ranked_count"),
                    "excluded_count": result.get("excluded_count"),
                    "execution": result.get("execution"),
                    "policy_audit": policy_audit,
                }
            )
        print(json.dumps({"runs": runs}, ensure_ascii=False, indent=2))
    finally:
        service.close_compute_engine()


if __name__ == "__main__":
    main()

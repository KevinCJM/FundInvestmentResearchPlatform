"""Series boundary around an already warmed indicator calculation plan."""
import numpy as np
from cal_indicators.typed_dsl import TypedDslError
from .series_numba import valid_series_output_kernel


def compute_warmed_series(runtime, context):
    # Retain the shared runtime's input, shape, node and cost checks. Unlike a
    # scalar result, a series may have missing warmup observations.
    arguments, _, trace = runtime.prepare_context(context)
    try:
        values = runtime.compiled_plan.compute(arguments)
    except (TypeError, ValueError, ZeroDivisionError, FloatingPointError) as exc:
        code = str(exc).strip()
        if code not in {"DIVIDE_BY_ZERO", "DOMAIN_ERROR", "INVALID_PARAMETER", "INSUFFICIENT_SAMPLE", "NON_FINITE_RESULT"}:
            code = "NJIT_SIGNATURE_MISMATCH" if isinstance(exc, TypeError) else "OPERATOR_EXECUTION_FAILED"
        raise TypedDslError(code, "时序 NJIT 计划执行失败。") from exc
    if not isinstance(values, np.ndarray) or values.dtype != np.float64 or values.ndim != 1:
        raise TypedDslError("RUNTIME_TYPE_MISMATCH", "时序输出必须是 float64 一维数组。")
    values = np.ascontiguousarray(values)
    if not valid_series_output_kernel(values):
        raise TypedDslError("NON_FINITE_RESULT", "时序输出包含无穷值。")
    runtime.last_trace = tuple(trace)
    return values

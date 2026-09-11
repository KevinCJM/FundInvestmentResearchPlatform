"""Safe typed-DAG to Numba plan compiler.

Only compiler-owned identifiers are emitted.  User text is parsed and typed by
``typed_dsl`` first; this module consumes immutable nodes and never interpolates
formula text, variable names or function names into executable source.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numba
import numpy as np
from numba import types
from numba.core.registry import CPUDispatcher

from cal_indicators import typed_numba_kernels as kernels
from compute_policy import NJIT_BACKEND, validate_execution_audit
from .shared_batch_graph import SHARED_GRAPH_VERSION

if TYPE_CHECKING:
    from cal_indicators.typed_dsl import (
        TypedDagNode,
        TypedExpressionPlan,
        TypedSeriesBundlePlan,
    )


class NumbaPlanCompileError(RuntimeError):
    def __init__(self, message: str, *, plan_id: str, operator_id: str | None = None):
        super().__init__(message)
        self.plan_id = plan_id
        self.operator_id = operator_id


@dataclass(frozen=True)
class CompiledNumbaPlan:
    plan_id: str
    dispatcher: CPUDispatcher
    context_names: tuple[str, ...]
    source: str
    compile_ms: float
    required_workspace_bytes: int
    kernel_version: str = kernels.NUMERIC_KERNEL_VERSION
    engine_version: str = kernels.ENGINE_VERSION

    @property
    def compile_status(self) -> str:
        return "compiled"

    @property
    def compiled_signatures(self) -> tuple[str, ...]:
        return tuple(str(signature) for signature in self.dispatcher.signatures)

    def compute(self, arguments: tuple[Any, ...]) -> Any:
        return self.dispatcher(*arguments)

    def metadata(self) -> dict[str, Any]:
        signatures = list(self.compiled_signatures)
        audit = {
            "compiled_plan_id": self.plan_id,
            "compile_status": self.compile_status,
            "compile_ms": self.compile_ms,
            "kernel_version": self.kernel_version,
            "engine_version": self.engine_version,
            "required_workspace_bytes": self.required_workspace_bytes,
            "compiled_signatures": signatures,
            "kernel_signatures": {"generated_plan": signatures},
            "execution_backend": NJIT_BACKEND,
            "nopython": bool(self.dispatcher.nopython_signatures)
            and len(self.dispatcher.nopython_signatures) == len(signatures),
            "python_fallback": 0,
            "python_operator_calls": 0,
        }
        return validate_execution_audit(audit)


@dataclass(frozen=True)
class CompiledNumbaSeriesPlan:
    plan_id: str
    dispatcher: CPUDispatcher
    context_names: tuple[str, ...]
    channel_names: tuple[str, ...]
    source: str
    compile_ms: float
    required_workspace_bytes: int
    kernel_version: str = kernels.NUMERIC_KERNEL_VERSION
    engine_version: str = kernels.ENGINE_VERSION

    @property
    def compiled_signatures(self) -> tuple[str, ...]:
        return tuple(str(signature) for signature in self.dispatcher.signatures)

    def compute(self, arguments: tuple[Any, ...]) -> tuple[np.ndarray, ...]:
        result = self.dispatcher(*arguments)
        return tuple(result)

    def metadata(self) -> dict[str, Any]:
        signatures = list(self.compiled_signatures)
        return validate_execution_audit(
            {
                "compiled_plan_id": self.plan_id,
                "compile_status": "compiled",
                "compile_ms": self.compile_ms,
                "kernel_version": self.kernel_version,
                "engine_version": self.engine_version,
                "required_workspace_bytes": self.required_workspace_bytes,
                "channel_names": list(self.channel_names),
                "compiled_signatures": signatures,
                "kernel_signatures": {"generated_series_plan": signatures},
                "execution_backend": NJIT_BACKEND,
                "nopython": bool(self.dispatcher.nopython_signatures)
                and len(self.dispatcher.nopython_signatures) == len(signatures),
                "python_fallback": 0,
                "python_operator_calls": 0,
            }
        )


_PLAN_CACHE: dict[str, CompiledNumbaPlan] = {}
_PLAN_CACHE_LOCK = threading.RLock()
_BATCH_PLAN_CACHE: dict[str, "CompiledNumbaBatchPlan"] = {}
_BATCH_PLAN_INPUTS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {}
_SERIES_PLAN_CACHE: dict[str, CompiledNumbaSeriesPlan] = {}


def _rank(node: "TypedDagNode") -> int:
    return int(node.inferred_type.rank)


def _binary_dispatcher(lhs_rank: int, rhs_rank: int, *, comparison: bool = False) -> CPUDispatcher:
    prefix = "comparison" if comparison else "binary"
    if lhs_rank == 0 and rhs_rank == 0:
        return getattr(kernels, f"{prefix}_scalar")
    if lhs_rank == 1 and rhs_rank == 1:
        return getattr(kernels, f"{prefix}_1d")
    if lhs_rank == 1 and rhs_rank == 0:
        return getattr(kernels, f"{prefix}_1d_right_scalar")
    if lhs_rank == 0 and rhs_rank == 1:
        return getattr(kernels, f"{prefix}_1d_left_scalar")
    if lhs_rank == 2 and rhs_rank == 2:
        return getattr(kernels, f"{prefix}_2d")
    if lhs_rank == 2 and rhs_rank == 0:
        return getattr(kernels, f"{prefix}_2d_right_scalar")
    if lhs_rank == 0 and rhs_rank == 2:
        return getattr(kernels, f"{prefix}_2d_left_scalar")
    raise ValueError("unsupported binary rank combination")


def _series_safe_divide_dispatcher(
    lhs_rank: int,
    rhs_rank: int,
) -> CPUDispatcher:
    if lhs_rank == 1 and rhs_rank == 1:
        return kernels.series_safe_divide_1d
    if lhs_rank == 1 and rhs_rank == 0:
        return kernels.series_safe_divide_1d_right_scalar
    if lhs_rank == 0 and rhs_rank == 1:
        return kernels.series_safe_divide_1d_left_scalar
    if lhs_rank == 2 and rhs_rank == 2:
        return kernels.series_safe_divide_2d
    if lhs_rank == 2 and rhs_rank == 0:
        return kernels.series_safe_divide_2d_right_scalar
    if lhs_rank == 0 and rhs_rank == 2:
        return kernels.series_safe_divide_2d_left_scalar
    raise ValueError("unsupported series safe-divide rank combination")


def _call(dispatcher: CPUDispatcher, args: list[str], globals_map: dict[str, Any]) -> str:
    name = f"k{len(globals_map)}"
    globals_map[name] = dispatcher
    return f"{name}({', '.join(args)})"


def _rolling_window_arguments(
    window_node: "TypedDagNode",
    *,
    node_prefix: str,
) -> tuple[str, str, str]:
    by_name = {name: int(input_id) for name, input_id in window_node.arguments}
    try:
        values = f"{node_prefix}{by_name['values']}"
        window = f"{node_prefix}{by_name['window']}"
    except KeyError as exc:
        raise ValueError("rolling_window node is missing required inputs") from exc
    minimum_id = by_name.get("min_periods")
    minimum = f"{node_prefix}{minimum_id}" if minimum_id is not None else window
    return values, window, minimum


def _operator_call(
    node: "TypedDagNode",
    input_nodes: tuple["TypedDagNode", ...],
    globals_map: dict[str, Any],
    *,
    node_prefix: str = "n",
    series_mode: bool = False,
) -> str:
    operator_id = str(node.operator_id)
    args = [f"{node_prefix}{input_node.node_id}" for input_node in input_nodes]
    from .drawdown_interval import INTERVAL_KERNELS
    from .primitive_access import ACCESS_KERNELS
    from .regression_state import FIT_PROJECTION_KERNELS, linear_fit_pair_kernel, linear_fit_time_kernel
    state_kernels = {**INTERVAL_KERNELS, **ACCESS_KERNELS, **FIT_PROJECTION_KERNELS}
    if operator_id in state_kernels:
        return _call(state_kernels[operator_id], args, globals_map)
    if operator_id == "linear_fit":
        return _call(linear_fit_time_kernel if len(args) == 1 else linear_fit_pair_kernel, args, globals_map)
    if operator_id == "rolling_window":
        # Logical authoring node only. It aliases the source series and never
        # materializes a T×W matrix; the consuming reducer below is fused.
        return args[0]

    if input_nodes and input_nodes[0].operator_id == "rolling_window":
        values, window, minimum = _rolling_window_arguments(
            input_nodes[0], node_prefix=node_prefix
        )
        if operator_id == "mean":
            return _call(kernels.rolling_mean_1d, [values, window, minimum], globals_map)
        if operator_id == "min_value":
            return _call(kernels.rolling_min_1d, [values, window, minimum], globals_map)
        if operator_id == "max_value":
            return _call(kernels.rolling_max_1d, [values, window, minimum], globals_map)
        if operator_id in {"std", "variance"}:
            degrees = args[1] if len(args) == 2 else "1.0"
            dispatcher = (
                kernels.rolling_std_1d
                if operator_id == "std"
                else kernels.rolling_variance_1d
            )
            return _call(
                dispatcher,
                [values, window, degrees, minimum],
                globals_map,
            )
        raise ValueError(
            f"operator {operator_id} cannot consume logical rolling_window"
        )

    ranks = tuple(_rank(input_node) for input_node in input_nodes)
    output_rank = _rank(node)

    if (
        series_mode
        and operator_id == "divide"
        and output_rank > 0
    ):
        dispatcher = _series_safe_divide_dispatcher(ranks[0], ranks[1])
        return _call(dispatcher, args, globals_map)
    if operator_id in kernels.BASIC_OPCODES:
        dispatcher = _binary_dispatcher(ranks[0], ranks[1])
        return _call(dispatcher, [str(kernels.BASIC_OPCODES[operator_id]), *args], globals_map)
    if operator_id in kernels.UNARY_OPCODES:
        dispatcher = (kernels.unary_scalar, kernels.unary_1d, kernels.unary_2d)[ranks[0]]
        return _call(dispatcher, [str(kernels.UNARY_OPCODES[operator_id]), *args], globals_map)
    if operator_id == "clip":
        dispatcher = (kernels.clip_scalar, kernels.clip_1d, kernels.clip_2d)[ranks[0]]
        return _call(dispatcher, args, globals_map)
    if operator_id in kernels.COMPARISON_OPCODES:
        dispatcher = _binary_dispatcher(ranks[0], ranks[1], comparison=True)
        return _call(dispatcher, [str(kernels.COMPARISON_OPCODES[operator_id]), *args], globals_map)
    if operator_id in {"logical_and", "logical_or"}:
        dispatcher = (kernels.logical_scalar, kernels.logical_1d, kernels.logical_2d)[ranks[0]]
        opcode = 1 if operator_id == "logical_and" else 2
        return _call(dispatcher, [str(opcode), *args], globals_map)
    if operator_id == "logical_not":
        dispatcher = (kernels.logical_not_scalar, kernels.logical_not_1d, kernels.logical_not_2d)[ranks[0]]
        return _call(dispatcher, args, globals_map)
    if operator_id == "where":
        if output_rank == 0:
            dispatcher = kernels.where_scalar
        elif output_rank == 1:
            dispatcher = kernels.where_1d if ranks[1:] == (1, 1) else (
                kernels.where_1d_false_scalar if ranks[1:] == (1, 0) else kernels.where_1d_true_scalar
            )
        else:
            dispatcher = kernels.where_2d if ranks[1:] == (2, 2) else (
                kernels.where_2d_false_scalar if ranks[1:] == (2, 0) else kernels.where_2d_true_scalar
            )
        return _call(dispatcher, args, globals_map)
    if operator_id in kernels.REDUCTION_OPCODES:
        opcode = kernels.REDUCTION_OPCODES[operator_id]
        if len(args) == 2:
            dispatcher = kernels.reduce_1d_parameter if ranks[0] == 1 else kernels.reduce_2d_parameter
            return _call(dispatcher, [args[0], args[1], str(opcode)], globals_map)
        dispatcher = kernels.reduce_1d if ranks[0] == 1 else kernels.reduce_2d
        return _call(dispatcher, [str(opcode), args[0], "1"], globals_map)
    if operator_id == "quantile":
        dispatcher = kernels.quantile_1d if ranks[0] == 1 else kernels.quantile_2d
        return _call(dispatcher, args, globals_map)
    if operator_id in kernels.SCAN_OPCODES:
        return _call(kernels.scan_1d, [str(kernels.SCAN_OPCODES[operator_id]), args[0]], globals_map)
    if operator_id == "drawdown_series":
        return _call(kernels.drawdown_series_1d, args, globals_map)
    if operator_id == "new_high_mask":
        return _call(kernels.new_high_mask_1d, args, globals_map)
    if operator_id in {"first", "last", "length"}:
        dispatcher = {"first": kernels.first_1d, "last": kernels.last_1d, "length": kernels.length_1d}[operator_id]
        return _call(dispatcher, args, globals_map)
    if operator_id in {"lag", "difference"}:
        dispatcher = kernels.lag_1d_parameter if operator_id == "lag" else kernels.difference_1d_parameter
        period = args[1] if len(args) == 2 else "1.0"
        return _call(dispatcher, [args[0], period], globals_map)
    if operator_id == "rolling_mean":
        minimum = args[2] if len(args) == 3 else args[1]
        return _call(kernels.rolling_mean_1d, [args[0], args[1], minimum], globals_map)
    if operator_id == "rolling_std":
        degrees = args[2] if len(args) >= 3 else "0.0"
        minimum = args[3] if len(args) == 4 else args[1]
        return _call(
            kernels.rolling_std_1d,
            [args[0], args[1], degrees, minimum],
            globals_map,
        )
    if operator_id == "rolling_min":
        minimum = args[2] if len(args) == 3 else args[1]
        return _call(kernels.rolling_min_1d, [args[0], args[1], minimum], globals_map)
    if operator_id == "rolling_max":
        minimum = args[2] if len(args) == 3 else args[1]
        return _call(kernels.rolling_max_1d, [args[0], args[1], minimum], globals_map)
    if operator_id == "recursive_smooth":
        return _call(kernels.recursive_smooth_1d, args, globals_map)
    if operator_id == "divide_or_default":
        return _call(kernels.divide_or_default_1d, args, globals_map)
    if operator_id.endswith("_time") or operator_id.endswith("_asset"):
        suffix = operator_id.rsplit("_", 1)[0]
        opcode = kernels.AXIS_REDUCTION_OPCODES[suffix]
        dispatcher = kernels.axis_reduce_time_fixed if operator_id.endswith("_time") else kernels.axis_reduce_asset
        return _call(dispatcher, [str(opcode), args[0]], globals_map)
    if operator_id in kernels.MASK_REDUCTION_OPCODES:
        dispatcher = kernels.masked_reduce_1d if ranks[0] == 1 else kernels.masked_reduce_2d
        return _call(dispatcher, [str(kernels.MASK_REDUCTION_OPCODES[operator_id]), *args], globals_map)
    if operator_id == "quantile_where":
        dispatcher = kernels.masked_quantile_1d if ranks[0] == 1 else kernels.masked_quantile_2d
        return _call(dispatcher, args, globals_map)
    if operator_id == "count_true":
        dispatcher = kernels.count_true_1d if ranks[0] == 1 else kernels.count_true_2d
        return _call(dispatcher, args, globals_map)
    if operator_id == "max_consecutive_true":
        return _call(kernels.max_consecutive_true_1d, args, globals_map)
    if operator_id == "transpose":
        return _call(kernels.transpose_2d, args, globals_map)
    if operator_id == "dot":
        return _call(kernels.dot_1d, args, globals_map)
    if operator_id == "outer":
        return _call(kernels.outer_1d, args, globals_map)
    if operator_id == "matmul":
        return _call(kernels.matmul_2d, args, globals_map)
    if operator_id == "matvec":
        return _call(kernels.matvec_2d, args, globals_map)
    if operator_id == "diag":
        return _call(kernels.diag_1d if ranks[0] == 1 else kernels.diag_2d, args, globals_map)
    if operator_id == "trace":
        return _call(kernels.trace_2d, args, globals_map)
    if operator_id == "solve":
        return _call(kernels.solve_2d, args, globals_map)
    if operator_id == "covariance":
        return _call(kernels.covariance_2d if len(args) == 1 else kernels.covariance_1d, args, globals_map)
    if operator_id == "correlation":
        return _call(kernels.correlation_2d if len(args) == 1 else kernels.correlation_1d, args, globals_map)
    raise ValueError(f"unknown typed opcode: {operator_id}")


def _numba_type(value_type: Any, *, readonly: bool) -> Any:
    if value_type.rank == 0:
        return types.uint8 if value_type.is_mask else types.float64
    dtype = types.uint8 if value_type.is_mask else types.float64
    return types.Array(dtype, value_type.rank, "C", readonly=readonly)


def _workspace_bytes(plan: "TypedExpressionPlan") -> int:
    # Symbolic dimensions are bound at runtime.  This value is a deterministic
    # lower-bound used by validation responses; the request planner performs
    # the actual 64MB budget check after T/N are known.
    total = 0
    for node in plan.nodes:
        dimensions = [item for item in node.inferred_type.shape if isinstance(item, int)]
        if dimensions:
            elements = 1
            for dimension in dimensions:
                elements *= dimension
            total += elements * (1 if node.inferred_type.is_mask else 8)
    return total


def _plan_id(plan: "TypedExpressionPlan") -> str:
    from .operator_lowering import LOWERING_VERSION
    payload = LOWERING_VERSION + "|" + "|".join(
        (
            plan.expression_hash,
            plan.dsl_version,
            plan.operator_registry_version,
            kernels.NUMERIC_KERNEL_VERSION,
            plan.output_contract,
        )
    )
    if any(node.operator_id == "rolling_apply" for node in plan.nodes):
        from .rolling_scope import SCOPE_VERSION
        payload += "|" + SCOPE_VERSION
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def numba_plan_id(plan: "TypedExpressionPlan") -> str:
    """Return the stable compiler-owned id without compiling the plan."""

    return _plan_id(plan)


def get_cached_numba_plan(
    plan: "TypedExpressionPlan",
) -> CompiledNumbaPlan | None:
    """Return an already-warmed plan and never compile on a request path."""

    plan_id = _plan_id(plan)
    with _PLAN_CACHE_LOCK:
        return _PLAN_CACHE.get(plan_id)


def _emit_nodes(plan, roots, context_positions, globals_map, *, series_mode=False):
    from .rolling_scope import outside_nodes, scope_call
    by_id = {node.node_id: node for node in plan.nodes}
    lines = []
    for node in outside_nodes(plan.nodes, roots):
        if node.kind == "constant":
            expression = repr(float(node.label))
        elif node.kind == "variable":
            expression = f"v{context_positions[node.label]}"
        elif node.operator_id == "rolling_apply":
            expression = scope_call(node, plan.nodes, plan.operator_registry_version, globals_map)
        else:
            expression = _operator_call(node, tuple(by_id[index] for index in node.inputs), globals_map, series_mode=series_mode)
        lines.append(f"    n{node.node_id} = {expression}")
    return lines


def compile_numba_plan(plan: "TypedExpressionPlan") -> CompiledNumbaPlan:
    plan_id = _plan_id(plan)
    with _PLAN_CACHE_LOCK:
        cached = _PLAN_CACHE.get(plan_id)
        if cached is not None:
            return cached

    started = time.perf_counter()
    context_names = tuple(plan.context_requirements)
    context_positions = {name: index for index, name in enumerate(context_names)}
    globals_map: dict[str, Any] = {}
    lines = [f"def generated_plan({', '.join(f'v{index}' for index in range(len(context_names)))}):"]
    failed_operator: str | None = None
    try:
        lines.extend(_emit_nodes(plan, (plan.root_id,), context_positions, globals_map))
        lines.append(f"    return n{plan.root_id}")
        source = "\n".join(lines) + "\n"
        namespace: dict[str, Any] = dict(globals_map)
        exec(compile(source, f"<typed-numba-plan:{plan_id}>", "exec"), namespace)
        dispatcher = numba.njit(cache=False, nogil=True)(namespace["generated_plan"])
        mutable_signature = tuple(_numba_type(plan.context_requirements[name], readonly=False) for name in context_names)
        dispatcher.compile(mutable_signature)
        if any(plan.context_requirements[name].rank > 0 for name in context_names):
            readonly_signature = tuple(_numba_type(plan.context_requirements[name], readonly=True) for name in context_names)
            dispatcher.compile(readonly_signature)
        # A production dispatcher must never specialize itself on first use.
        # Unsupported dtype/layout combinations now fail closed instead of
        # compiling a new signature in the request that happened to hit them.
        dispatcher.disable_compile()
    except Exception as exc:
        raise NumbaPlanCompileError(
            "typed 公式无法编译为固定签名 NJIT 计划。",
            plan_id=plan_id,
            operator_id=failed_operator,
        ) from exc

    compiled = CompiledNumbaPlan(
        plan_id=plan_id,
        dispatcher=dispatcher,
        context_names=context_names,
        source=source,
        compile_ms=round((time.perf_counter() - started) * 1000.0, 3),
        required_workspace_bytes=_workspace_bytes(plan),
    )
    with _PLAN_CACHE_LOCK:
        _PLAN_CACHE[plan_id] = compiled
    return compiled


def _series_plan_id(plan: "TypedSeriesBundlePlan") -> str:
    from .operator_lowering import LOWERING_VERSION
    payload = LOWERING_VERSION + "|" + "|".join(
        (
            plan.expression_hash,
            plan.dsl_version,
            plan.operator_registry_version,
            kernels.NUMERIC_KERNEL_VERSION,
            ",".join(plan.roots),
        )
    )
    if any(node.operator_id == "rolling_apply" for node in plan.nodes):
        from .rolling_scope import SCOPE_VERSION
        payload += "|" + SCOPE_VERSION
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def numba_series_plan_id(plan: "TypedSeriesBundlePlan") -> str:
    return _series_plan_id(plan)


def get_cached_numba_series_plan(
    plan: "TypedSeriesBundlePlan",
) -> CompiledNumbaSeriesPlan | None:
    plan_id = _series_plan_id(plan)
    with _PLAN_CACHE_LOCK:
        return _SERIES_PLAN_CACHE.get(plan_id)


def compile_numba_series_plan(
    plan: "TypedSeriesBundlePlan",
) -> CompiledNumbaSeriesPlan:
    """Compile one shared multi-root time-series DAG and freeze its signatures."""

    plan_id = _series_plan_id(plan)
    with _PLAN_CACHE_LOCK:
        cached = _SERIES_PLAN_CACHE.get(plan_id)
        if cached is not None:
            return cached

    started = time.perf_counter()
    context_names = tuple(plan.context_requirements)
    context_positions = {name: index for index, name in enumerate(context_names)}
    channel_names = tuple(plan.roots)
    globals_map: dict[str, Any] = {}
    lines = [
        f"def generated_series_plan({', '.join(f'v{index}' for index in range(len(context_names)))}):"
    ]
    failed_operator: str | None = None
    try:
        lines.extend(_emit_nodes(plan, tuple(plan.roots.values()), context_positions, globals_map, series_mode=True))
        root_values = ", ".join(f"n{plan.roots[name]}" for name in channel_names)
        if len(channel_names) == 1:
            root_values += ","
        lines.append(f"    return ({root_values})")
        source = "\n".join(lines) + "\n"
        namespace: dict[str, Any] = dict(globals_map)
        exec(
            compile(source, f"<typed-numba-series-plan:{plan_id}>", "exec"),
            namespace,
        )
        dispatcher = numba.njit(cache=False, nogil=True)(
            namespace["generated_series_plan"]
        )
        mutable_signature = tuple(
            _numba_type(plan.context_requirements[name], readonly=False)
            for name in context_names
        )
        dispatcher.compile(mutable_signature)
        if any(plan.context_requirements[name].rank > 0 for name in context_names):
            readonly_signature = tuple(
                _numba_type(plan.context_requirements[name], readonly=True)
                for name in context_names
            )
            dispatcher.compile(readonly_signature)
        dispatcher.disable_compile()
    except Exception as exc:
        raise NumbaPlanCompileError(
            "typed 时序公式无法编译为固定签名 NJIT 计划。",
            plan_id=plan_id,
            operator_id=failed_operator,
        ) from exc

    compiled = CompiledNumbaSeriesPlan(
        plan_id=plan_id,
        dispatcher=dispatcher,
        context_names=context_names,
        channel_names=channel_names,
        source=source,
        compile_ms=round((time.perf_counter() - started) * 1000.0, 3),
        required_workspace_bytes=_workspace_bytes(plan),
    )
    with _PLAN_CACHE_LOCK:
        _SERIES_PLAN_CACHE[plan_id] = compiled
    return compiled


def persist_numba_series_plan(
    compiled: CompiledNumbaSeriesPlan,
    runtime_root: Path,
) -> Path:
    target = runtime_root / "generated_series" / compiled.plan_id
    target.mkdir(parents=True, exist_ok=True)
    (target / "plan.py").write_text(compiled.source, encoding="utf-8")
    (target / "plan.json").write_text(
        json.dumps(compiled.metadata(), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return target


def persist_numba_plan(compiled: CompiledNumbaPlan, runtime_root: Path) -> Path:
    """Persist compiler-owned source and metadata for audit/restart warmup."""

    target = runtime_root / "generated" / compiled.plan_id
    target.mkdir(parents=True, exist_ok=True)
    source_path = target / "plan.py"
    metadata_path = target / "plan.json"
    source_path.write_text(compiled.source, encoding="utf-8")
    metadata_path.write_text(
        json.dumps(compiled.metadata(), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return target


def plan_cache_status() -> dict[str, Any]:
    with _PLAN_CACHE_LOCK:
        return {
            "entries": len(_PLAN_CACHE),
            "batch_entries": len(_BATCH_PLAN_CACHE),
            "series_entries": len(_SERIES_PLAN_CACHE),
            "engine_version": kernels.ENGINE_VERSION,
            "kernel_version": kernels.NUMERIC_KERNEL_VERSION,
        }


@dataclass(frozen=True)
class CompiledNumbaBatchPlan:
    plan_id: str
    serial_dispatcher: CPUDispatcher
    parallel_dispatcher: CPUDispatcher
    metric_count: int
    source_serial: str
    source_parallel: str
    compile_ms: float
    sharing: dict[str, Any]

    @property
    def compiled_signatures(self) -> dict[str, list[str]]:
        return {
            "generated_batch_serial": [
                str(signature) for signature in self.serial_dispatcher.signatures
            ],
            "generated_batch_parallel": [
                str(signature) for signature in self.parallel_dispatcher.signatures
            ],
        }

    def metadata(self) -> dict[str, Any]:
        signatures = {
            **self.compiled_signatures,
            "risk_free_scalars": [
                str(signature)
                for signature in _risk_free_scalars_kernel.nopython_signatures
            ],
        }
        audit = {
            "compiled_plan_id": self.plan_id,
            "compile_status": "compiled",
            "compile_ms": self.compile_ms,
            "kernel_version": kernels.NUMERIC_KERNEL_VERSION,
            "engine_version": kernels.ENGINE_VERSION,
            "metric_count": self.metric_count,
            **self.sharing,
            "compiled_signatures": signatures,
            "kernel_signatures": signatures,
            "execution_backend": NJIT_BACKEND,
            "nopython": bool(self.serial_dispatcher.nopython_signatures)
            and bool(self.parallel_dispatcher.nopython_signatures),
            "python_fallback": 0,
            "python_operator_calls": 0,
        }
        return validate_execution_audit(audit)

    def compute(
        self,
        values: np.ndarray,
        starts: np.ndarray,
        ends: np.ndarray,
        elapsed_days: np.ndarray,
        output: np.ndarray,
        statuses: np.ndarray,
        *,
        parallel: bool,
        enabled: np.ndarray | None = None,
    ) -> None:
        selected = np.ones(self.metric_count, dtype=np.uint8) if enabled is None else np.ascontiguousarray(enabled, dtype=np.uint8)
        if selected.shape != (self.metric_count,):
            raise ValueError("Selected roots do not match the prepared execution plan")
        dispatcher = self.parallel_dispatcher if parallel else self.serial_dispatcher
        dispatcher(values, starts, ends, elapsed_days, output, statuses, selected)


@dataclass(frozen=True)
class SelectedNumbaBatchPlan:
    """A root selection over a prewarmed plan; no compilation or numeric fallback."""
    prepared: CompiledNumbaBatchPlan
    indices: tuple[int, ...]

    @property
    def plan_id(self):
        return self.prepared.plan_id

    def metadata(self):
        return {**self.prepared.metadata(), "selected_roots": list(self.indices), "selected_root_count": len(self.indices)}

    def compute(self, values, starts, ends, elapsed_days, output, statuses, *, parallel):
        selected = np.zeros(self.prepared.metric_count, dtype=np.uint8)
        selected[np.asarray(self.indices, dtype=np.int64)] = 1
        full_values = np.empty((starts.size, self.prepared.metric_count), dtype=np.float64)
        full_status = np.empty(full_values.shape, dtype=np.int16)
        self.prepared.compute(values, starts, ends, elapsed_days, full_values, full_status,
                              parallel=parallel, enabled=selected)
        np.take(full_values, self.indices, axis=1, out=output)
        np.take(full_status, self.indices, axis=1, out=statuses)


@numba.njit(
    types.UniTuple(types.float64, 2)(types.float64),
    cache=False,
    nogil=True,
)
def _risk_free_scalars_kernel(annual_percent: float) -> tuple[float, float]:
    annual = annual_percent / 100.0
    per_observation = max(0.0, 1.0 + annual) ** (1.0 / 252.0) - 1.0
    return annual, per_observation


_risk_free_scalars_kernel.disable_compile()


def _risk_free_scalars(definition: dict[str, Any]) -> tuple[float, float]:
    """Map config to the eagerly compiled risk-free conversion kernel."""

    annual_percent = float(definition.get("annual_risk_free_rate_percent", 0.0))
    return _risk_free_scalars_kernel(annual_percent)


def _batch_variable_expression(
    name: str,
    *,
    column_index: dict[str, int],
    definition: dict[str, Any],
) -> str:
    if name == "returns":
        return "returns_view"
    if name == "log_returns":
        return "log_returns_view"
    if name == "observation_count":
        return "float(end - start - 1)"
    if name == "window_elapsed_days":
        return "elapsed_days[row]"
    if name == "periods_per_year":
        return "252.0"
    annual, per_observation = _risk_free_scalars(definition)
    if name == "annual_risk_free_rate_decimal":
        return repr(annual)
    if name in {"risk_free_rate_per_observation", "risk_free_rate_per_period"}:
        return repr(per_observation)
    if name == "risk_free_return_window":
        return f"({repr(max(0.0, 1.0 + annual))} ** (elapsed_days[row] / 365.0) - 1.0)"
    if name not in column_index:
        raise ValueError(f"batch physical variable unavailable: {name}")
    return f"values[{column_index[name]}, start:end]"


def _batch_source(
    plans: tuple["TypedExpressionPlan", ...],
    definitions: tuple[dict[str, Any], ...],
    column_index: dict[str, int],
    *,
    parallel: bool,
) -> tuple[str, dict[str, Any]]:
    from .shared_batch_graph import build_batch_source
    return build_batch_source(
        plans, definitions, column_index, parallel=parallel,
        variable_expression=_batch_variable_expression, operator_call=_operator_call,
    )


def _batch_plan_id(
    plans: tuple["TypedExpressionPlan", ...],
    definitions: tuple[dict[str, Any], ...],
    physical_columns: tuple[str, ...],
) -> str:
    if not plans or len(plans) != len(definitions):
        raise ValueError("batch plan requires matching plans and definitions")
    key_payload = {
        "sharing_version": SHARED_GRAPH_VERSION,
        "plans": [_plan_id(plan) for plan in plans],
        "risk_free": [definition.get("annual_risk_free_rate_percent", 0.0) for definition in definitions],
        "columns": list(physical_columns),
        "kernel": kernels.NUMERIC_KERNEL_VERSION,
    }
    return hashlib.sha256(
        json.dumps(key_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def get_cached_numba_batch_plan(
    plans: tuple["TypedExpressionPlan", ...],
    definitions: tuple[dict[str, Any], ...],
    physical_columns: tuple[str, ...],
) -> CompiledNumbaBatchPlan | SelectedNumbaBatchPlan | None:
    """Find a prepared exact plan or a pruned superset, never singleton fallbacks."""
    plan_id = _batch_plan_id(plans, definitions, physical_columns)
    keys = tuple(_batch_plan_id((plan,), (definition,), physical_columns) for plan, definition in zip(plans, definitions))
    with _PLAN_CACHE_LOCK:
        exact = _BATCH_PLAN_CACHE.get(plan_id)
        if exact is not None:
            return exact
        candidates = [(len(metrics), key, metrics) for key, (columns, metrics) in _BATCH_PLAN_INPUTS.items()
                      if columns == physical_columns and set(keys).issubset(metrics)]
        if not candidates:
            return None
        _, key, metrics = min(candidates)
        return SelectedNumbaBatchPlan(_BATCH_PLAN_CACHE[key], tuple(metrics.index(item) for item in keys))


def compile_numba_batch_plan(
    plans: tuple["TypedExpressionPlan", ...],
    definitions: tuple[dict[str, Any], ...],
    physical_columns: tuple[str, ...],
) -> CompiledNumbaBatchPlan:
    plan_id = _batch_plan_id(plans, definitions, physical_columns)
    with _PLAN_CACHE_LOCK:
        cached = _BATCH_PLAN_CACHE.get(plan_id)
        if cached is not None:
            return cached
    started = time.perf_counter()
    column_index = {name: index for index, name in enumerate(physical_columns)}
    dispatchers: list[CPUDispatcher] = []
    sources: list[str] = []
    row_source, row_namespace = _batch_source(plans, definitions, column_index, parallel=False)
    exec(compile(row_source, f"<typed-numba-row:{plan_id}>", "exec"), row_namespace)
    row_dispatcher = numba.njit(cache=False, nogil=True)(row_namespace["generated_row"])
    row_dispatcher.compile((types.float64[:, ::1], types.int64, types.int64, types.float64,
                            types.float64[::1], types.int16[::1], types.uint8[::1]))
    row_dispatcher.disable_compile()
    for parallel in (False, True):
        source = (
            "def generated_batch(values, starts, ends, elapsed_days, output, statuses, enabled):\n"
            "    for row in loop(starts.size):\n"
            "        row_kernel(values, starts[row], ends[row], elapsed_days[row], output[row], statuses[row], enabled)\n"
        )
        namespace = {"loop": numba.prange if parallel else range, "row_kernel": row_dispatcher}
        exec(compile(source, f"<typed-numba-batch:{plan_id}>", "exec"), namespace)
        dispatcher = numba.njit(cache=False, nogil=True, parallel=parallel)(
            namespace["generated_batch"]
        )
        signature = (
            types.float64[:, ::1],
            types.int64[::1],
            types.int64[::1],
            types.float64[::1],
            types.float64[:, ::1],
            types.int16[:, ::1],
            types.uint8[::1],
        )
        dispatcher.compile(signature)
        dispatcher.disable_compile()
        dispatchers.append(dispatcher)
        sources.append(row_source + "\n" + source)
    from .shared_batch_graph import merge_metric_plans
    sharing = merge_metric_plans(plans, definitions, _batch_variable_expression, column_index).audit()
    compiled = CompiledNumbaBatchPlan(
        plan_id=plan_id,
        serial_dispatcher=dispatchers[0],
        parallel_dispatcher=dispatchers[1],
        metric_count=len(plans),
        source_serial=sources[0],
        source_parallel=sources[1],
        compile_ms=round((time.perf_counter() - started) * 1000.0, 3),
        sharing=sharing,
    )
    with _PLAN_CACHE_LOCK:
        _BATCH_PLAN_CACHE[plan_id] = compiled
        _BATCH_PLAN_INPUTS[plan_id] = (physical_columns, tuple(
            _batch_plan_id((plan,), (definition,), physical_columns) for plan, definition in zip(plans, definitions)
        ))
    return compiled


def persist_numba_batch_plan(
    compiled: CompiledNumbaBatchPlan,
    runtime_root: Path,
) -> Path:
    """Persist both fixed-signature batch lanes for restart/audit evidence."""

    target = runtime_root / "generated_batches" / compiled.plan_id
    target.mkdir(parents=True, exist_ok=True)
    (target / "serial.py").write_text(compiled.source_serial, encoding="utf-8")
    (target / "parallel.py").write_text(compiled.source_parallel, encoding="utf-8")
    (target / "plan.json").write_text(
        json.dumps(compiled.metadata(), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return target


__all__ = [
    "CompiledNumbaPlan",
    "CompiledNumbaSeriesPlan",
    "NumbaPlanCompileError",
    "compile_numba_plan",
    "compile_numba_series_plan",
    "compile_numba_batch_plan",
    "get_cached_numba_plan",
    "get_cached_numba_series_plan",
    "get_cached_numba_batch_plan",
    "CompiledNumbaBatchPlan",
    "numba_plan_id",
    "numba_series_plan_id",
    "persist_numba_batch_plan",
    "persist_numba_series_plan",
    "persist_numba_plan",
    "plan_cache_status",
]

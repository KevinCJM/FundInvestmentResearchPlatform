"""Compiler-owned rolling execution scope over an interval-scalar DAG.

No user callable is evaluated by Python. The compiler binds registered NJIT
calls before serving requests; each window receives read-only slice views.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import threading
from typing import Any, Mapping, Sequence

import numba
import numpy as np
from numba import types

from .typed_types import TypedDslError, ValueType

SCOPE_VERSION = "interval-rolling-1.5"
SCOPE_OPERATOR = "rolling_apply"
MAX_WINDOW = 5000
MAX_WORK = 100_000_000
MAX_SCRATCH_BYTES = 64 * 1024 * 1024
SCALAR_CONTEXT = frozenset({
    "observation_count", "window_elapsed_days", "risk_free_return_window",
    "periods_per_year", "annual_risk_free_rate_decimal",
    "risk_free_rate_per_observation", "risk_free_rate_per_period",
})
RETURN_INPUTS = frozenset({"returns", "log_returns"})
# These are execution/selection semantics, not a list of supported indicators.
NESTED_SCOPES = frozenset({SCOPE_OPERATOR, "rolling_window", "rolling_mean", "rolling_std", "rolling_min", "rolling_max"})
SELECTION_OPERATORS = frozenset({"first", "last", "length", "argmin", "argmax", "value_at"})


def _fail(code: str, message: str, node_id: int | None = None) -> None:
    raise TypedDslError(code, message, node_id=node_id)


def dependency_nodes(nodes: Sequence[Any], root_id: int) -> tuple[Any, ...]:
    by_id = {node.node_id: node for node in nodes}
    seen: set[int] = set()
    pending = [root_id]
    while pending:
        index = pending.pop()
        if index in seen:
            continue
        seen.add(index)
        pending.extend(by_id[index].inputs)
    return tuple(node for node in nodes if node.node_id in seen)


@dataclass(frozen=True)
class IntervalCapability:
    variables: tuple[Any, ...]
    aggregates: tuple[str, ...]
    reset_operators: tuple[str, ...]
    has_returns: bool
    node_count: int
    array_count: int

    @property
    def needs_preceding_observation(self) -> bool:
        """Only level paths and elapsed-period context need a return baseline.

        Precomputed returns alone may begin at any valid observation; dropping
        their first row would lose the first complete window of a sliced chart.
        """
        return self.has_returns and any(
            (node.inferred_type.rank and node.inferred_type.shape == ('L',))
            or node.label in {'window_elapsed_days', 'risk_free_return_window'}
            for node in self.variables
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "supported": True, "protocol_version": SCOPE_VERSION,
            "message": "完整区间子图可独立重算，无需新增专用滚动算子。",
            "inputs": [node.label for node in self.variables if node.inferred_type.rank],
            "rewritten_reductions": list(self.aggregates),
            "reset_operators": list(self.reset_operators),
            "window_unit": "return_observations" if self.has_returns else "observations",
            "window_unit_label": "收益观察期" if self.has_returns else "观察点",
            "missing_policy": "complete_finite_window",
            "state_policy": "reset_at_window_start",
            "execution": "prewarmed_njit_interval_graph",
            "cost_model": "sum_of_interval_costs",
            "preceding_observations": int(self.needs_preceding_observation),
        }


def analyze_interval(nodes: Sequence[Any], root_id: int, registry: Mapping[str, Any]) -> IntervalCapability:
    """Prove capability structurally from reviewed primitive contracts."""
    selected = dependency_nodes(nodes, root_id)
    by_id = {node.node_id: node for node in selected}
    root = by_id[root_id].inferred_type
    if not root.is_scalar or not root.is_numeric or root.semantic_dimension == "date":
        _fail("ROLLING_SCALAR_REQUIRED", "滚动内容必须输出一个数值标量，不能是序列、日期、条件或中间状态。", root_id)
    variables = tuple(sorted((node for node in selected if node.kind == "variable"), key=lambda node: node.label))
    data = [node for node in variables if node.inferred_type.kind == "series" and node.inferred_type.is_numeric and node.inferred_type.semantic_dimension != "date"]
    if not data:
        _fail("ROLLING_TIME_INPUT_REQUIRED", "区间计算至少需要一条数值时间序列。", root_id)
    from .typed_dsl import DEFAULT_VARIABLE_TYPES
    for node in variables:
        value = node.inferred_type
        if value.rank and (value.kind != "series" or not value.is_numeric):
            _fail("ROLLING_INPUT_UNSUPPORTED", "通用滚动当前只接收按日期对齐的一维数值序列和标量配置。", node.node_id)
        if value.is_scalar and node.label not in SCALAR_CONTEXT and node.label in DEFAULT_VARIABLE_TYPES:
            _fail("ROLLING_CONTEXT_UNSUPPORTED", f"尚未定义 {node.label} 在每个窗口中的上下文语义。", node.node_id)
    aggregates: list[str] = []
    resets: list[str] = []
    for node in selected:
        if node.inferred_type.kind in {"matrix", "vector", "window"}:
            _fail("ROLLING_INTERMEDIATE_UNSUPPORTED", "该子图包含尚未支持的矩阵、资产轴或嵌套窗口中间结果。", node.node_id)
        if not node.operator_id:
            continue
        spec = registry[node.operator_id]
        if node.operator_id in NESTED_SCOPES:
            _fail("ROLLING_NESTED_SCOPE_UNSUPPORTED", "区间内容不能再次包含滚动作用域；请选择原始区间标量计算。", node.node_id)
        if spec.interval_policy not in {"local", "resettable"}:
            _fail("ROLLING_INTERVAL_POLICY_REQUIRED", f"{node.operator_id} 需要区间外历史状态或尚未声明区间闭合语义。", node.node_id)
        inputs_have_time = any("time" in by_id[parent].inferred_type.axes for parent in node.inputs)
        if inputs_have_time and (node.inferred_type.is_scalar or node.inferred_type.kind == "record") and node.operator_id not in SELECTION_OPERATORS:
            aggregates.append(node.operator_id)
        if spec.interval_policy == "resettable" or node.cost_model == "scan":
            resets.append(node.operator_id)
    if not aggregates:
        _fail("ROLLING_AGGREGATION_REQUIRED", "该计算没有区间聚合步骤；仅取值、长度或常数不作为滚动区间指标。", root_id)
    return IntervalCapability(variables, tuple(dict.fromkeys(aggregates)), tuple(dict.fromkeys(resets)),
                              bool(RETURN_INPUTS.intersection(node.label for node in variables)),
                              len(selected), sum(node.inferred_type.rank > 0 for node in selected))


def interval_capability(expression: str, *, variable_types=None, dsl_version=None, operator_registry_version=None) -> dict[str, Any]:
    from .typed_dsl import compose_typed_expression
    from .typed_operators import TYPED_DSL_VERSION, get_typed_operator_registry
    try:
        plan = compose_typed_expression(expression, variable_types=variable_types,
            dsl_version=dsl_version or TYPED_DSL_VERSION, operator_registry_version=operator_registry_version)
        return analyze_interval(plan.nodes, plan.root_id, get_typed_operator_registry(plan.operator_registry_version)).to_dict()
    except TypedDslError as exc:
        return {"supported": False, "protocol_version": SCOPE_VERSION, "code": exc.code, "message": exc.message}


def _infer_scope(inputs: tuple[ValueType, ...]) -> ValueType:
    value, width = inputs[:2]
    if not value.is_scalar or not value.is_numeric or value.semantic_dimension == "date":
        _fail("ROLLING_SCALAR_REQUIRED", "滚动内容必须是区间数值标量计算。")
    if not width.is_scalar or not width.is_numeric or width.semantic_dimension not in {"count", "dimensionless"}:
        _fail("INVALID_PARAMETER", "滚动窗口必须是整数常量或已开放的整数参数。")
    if len(inputs) in {3, 5}:
        minimum = inputs[-1]
        if not minimum.is_scalar or not minimum.is_numeric or minimum.semantic_dimension not in {"count", "dimensionless"}:
            _fail("INVALID_PARAMETER", "最少有效观察数必须是整数常量或已开放的整数参数。")
    if len(inputs) >= 4:
        dates, annual = inputs[2:4]
        if dates.kind != "series" or dates.semantic_dimension != "date" or not annual.is_scalar or not annual.is_numeric:
            _fail("ROLLING_CONTEXT_UNSUPPORTED", "滚动上下文需要真实观察日期和标量年度配置。")
    return ValueType.series(semantic_dimension=value.semantic_dimension, price_basis=value.price_basis)


def _scope_reference(*args: Any) -> Any:
    _fail("ROLLING_SCOPE_REQUIRES_DAG", "滚动计算必须由编译器绑定完整区间子图，不能接收预先计算好的标量。")


def rolling_scope_spec(version: str):
    from .typed_operators import OperatorSignature, TypedOperatorSpec
    return TypedOperatorSpec(SCOPE_OPERATOR, version, "rolling", (
        OperatorSignature(("scalar", "scalar<count>"), "series<time>[T]", "execute scalar subgraph independently in each complete trailing window"),
        OperatorSignature(("scalar", "scalar<count>", "scalar<count>"), "series<time>[T]", "minimum jointly finite observations; raw window views retain missing positions for explicit body handling"),
        OperatorSignature(("scalar", "scalar<count>", "series<time>[T]<date>", "scalar"), "series<time>[T]", "explicit date and annual-rate context bindings"),
        OperatorSignature(("scalar", "scalar<count>", "series<time>[T]<date>", "scalar", "scalar<count>"), "series<time>[T]", "explicit context and minimum observations; no row compression or filling"),
    ), "将整个区间聚合计算逐窗口执行；不是先计算全样本标量再重复。", _infer_scope, _scope_reference,
       cost_model="rolling_scope", interval_policy="scope")


def outside_nodes(nodes: Sequence[Any], roots: Sequence[int]) -> tuple[Any, ...]:
    """Defer only nodes owned exclusively by a rolling body, not shared outer users."""
    by_id = {node.node_id: node for node in nodes}
    needed: set[int] = set()
    pending = list(roots)
    while pending:
        index = pending.pop()
        if index in needed:
            continue
        needed.add(index)
        node = by_id[index]
        if node.operator_id == SCOPE_OPERATOR:
            pending.extend(node.inputs[1:])
            pending.extend(item.node_id for item in dependency_nodes(nodes, node.inputs[0]) if item.kind == "variable")
        else:
            pending.extend(node.inputs)
    return tuple(node for node in nodes if node.node_id in needed)


_SCOPE_CACHE: dict[str, Any] = {}
_SCOPE_LOCK = threading.RLock()


def compile_scope(nodes: Sequence[Any], scope: Any, registry_version: str):
    """Generate one static interval function and one generic rolling loop."""
    from .typed_operators import get_typed_operator_registry
    from .typed_numba_plan import _operator_call, _numba_type
    body = dependency_nodes(nodes, scope.inputs[0])
    capability = analyze_interval(nodes, scope.inputs[0], get_typed_operator_registry(registry_version))
    variables = capability.variables
    partial = len(scope.inputs) == 5
    key = hashlib.sha256(json.dumps({"version": SCOPE_VERSION, "registry": registry_version, "partial": partial,
        "body": next(node.formula_fragment for node in body if node.node_id == scope.inputs[0]),
        "types": [(node.label, node.inferred_type.to_dict()) for node in variables]}, sort_keys=True).encode()).hexdigest()
    with _SCOPE_LOCK:
        cached = _SCOPE_CACHE.get(key)
        if cached is not None:
            return cached, variables
        names = [f"b{index}" for index in range(len(variables))]
        by_id = {node.node_id: node for node in body}
        binding = {node.node_id: names[index] for index, node in enumerate(variables)}
        namespace: dict[str, Any] = {"np": np, "math": math}
        lines = [f"def interval_body({', '.join(names)}):"]
        for node in body:
            if node.kind == "variable":
                expression = binding[node.node_id]
            elif node.kind == "constant":
                expression = repr(float(node.label))
            else:
                expression = _operator_call(node, tuple(by_id[index] for index in node.inputs), namespace)
            # A symbolic T-n axis can have different concrete lengths for two
            # lag/difference branches. Never let a fixed NJIT kernel index an
            # unaligned second input (Numba does not bounds-check by default).
            array_inputs = [index for index in node.inputs if by_id[index].inferred_type.rank == 1]
            if len(array_inputs) > 1:
                mismatch = " or ".join(f"n{index}.size != n{array_inputs[0]}.size" for index in array_inputs[1:])
                lines.extend([f"    if {mismatch}:", "        return np.nan"])
            # Keep common domain failures outside exception paths and do not
            # leak a zero-denominator failure into neighbouring windows.
            if node.operator_id == "divide" and by_id[node.inputs[1]].inferred_type.is_scalar:
                lines.extend([f"    if n{node.inputs[1]} == 0.0:", "        return np.nan"])
            if node.operator_id:
                # Catch at the owning function boundary so already-created
                # intermediate arrays leave by a normal return and are released.
                lines.extend(["    try:", f"        n{node.node_id} = {expression}",
                              "    except Exception:", "        return np.nan"])
            else:
                lines.append(f"    n{node.node_id} = {expression}")
        lines.append(f"    return n{scope.inputs[0]}")
        exec(compile("\n".join(lines), f"<interval-body:{key}>", "exec"), namespace)
        interval = numba.njit(cache=False, nogil=True)(namespace["interval_body"])
        for readonly in (False, True):
            interval.compile(tuple(_numba_type(node.inferred_type, readonly=readonly) for node in variables))
        interval.disable_compile()
        wrapper_source = _rolling_loop_source(variables, capability, partial=partial)
        wrapper_namespace = {"np": np, "math": math, "interval_body": interval}
        exec(compile(wrapper_source, f"<rolling-scope:{key}>", "exec"), wrapper_namespace)
        # Inline the allocating scope into its caller: a validation exception
        # crossing a separately compiled array-call frame leaks borrowed NRT
        # references on supported Numba. The allocator regression covers the
        # actual parent dispatcher, not just this helper in isolation.
        kernel = numba.njit(cache=False, nogil=True, inline="always")(wrapper_namespace["rolling_scope"])
        for readonly in (False, True):
            signature = tuple(_numba_type(node.inferred_type, readonly=readonly) for node in variables) + (
                types.float64, types.Array(types.float64, 1, "C", readonly=readonly), types.float64)
            if partial:
                signature += (types.float64,)
            kernel.compile(signature)
        kernel.disable_compile()
        kernel.rolling_source = wrapper_source
        kernel.interval_source = "\n".join(lines)
        kernel.interval_dispatcher = interval
        _SCOPE_CACHE[key] = kernel
        return kernel, variables


def _rolling_loop_source(variables: Sequence[Any], capability: IntervalCapability, *, partial: bool = False) -> str:
    names = [f"b{index}" for index in range(len(variables))]
    arrays = [(node, names[index]) for index, node in enumerate(variables) if node.inferred_type.rank]
    arguments = [*names, 'window', 'dates', 'annual', *(['minimum'] if partial else [])]
    lines = [f"def rolling_scope({', '.join(arguments)}):",
        f"    if not math.isfinite(window) or window < 1 or window > {MAX_WINDOW} or int(window) != window:",
        "        raise ValueError('INVALID_PARAMETER')", "    width = int(window)", "    size = dates.size",
        f"    if size * min(size, width) * {max(1, capability.node_count)} > {MAX_WORK}:",
        "        raise ValueError('ROLLING_COMPUTE_BUDGET_EXCEEDED')",
        f"    if size * 16 + 8 + (width + 1) * {max(1, capability.array_count)} * 8 > {MAX_SCRATCH_BYTES}:",
        "        raise ValueError('ROLLING_MEMORY_BUDGET_EXCEEDED')",
        "    if not math.isfinite(annual):", "        raise ValueError('INVALID_PARAMETER')"]
    if partial:
        lines.extend(["    if not math.isfinite(minimum) or minimum < 1 or minimum > width or int(minimum) != minimum:",
                      "        raise ValueError('INVALID_MIN_PERIODS')", "    minimum_count = int(minimum)"])
    for _, name in arrays:
        lines.extend([f"    if {name}.size != size:", "        raise ValueError('ROLLING_ALIGNMENT_MISMATCH')"])
    for index, node in enumerate(variables):
        if node.inferred_type.is_scalar and node.label not in {"observation_count", "window_elapsed_days", "risk_free_return_window"}:
            lines.extend([f"    if not math.isfinite(b{index}):", "        raise ValueError('INVALID_PARAMETER')"])
    # Validation must precede allocation: Numba may leak arrays owned by a
    # function that raises. Valid windows then leave through normal returns.
    lines.extend(["    for index in range(size):",
        "        if not math.isfinite(dates[index]) or (index > 0 and dates[index] <= dates[index - 1]):",
        "            raise ValueError('ROLLING_DATE_AXIS_INVALID')",
        "    result = np.full(size, np.nan, dtype=np.float64)",
        "    invalid = np.zeros(size + 1, dtype=np.int64)", "    for index in range(size):"])
    bad = " or ".join(f"not math.isfinite({name}[index])" for _, name in arrays) or "False"
    lines.append(f"        invalid[index + 1] = invalid[index] + (1 if {bad} else 0)")
    if partial:
        first = "minimum_count" if capability.needs_preceding_observation else "minimum_count - 1"
        floor = 1 if capability.needs_preceding_observation else 0
        lines.extend([f"    for right in range({first}, size):", f"        start = max({floor}, right - width + 1)",
                      "        end = right + 1", "        if end - start - (invalid[end] - invalid[start]) < minimum_count:",
                      "            continue"])
    else:
        lines.extend(["    for right in range(width - 1, size):", "        start = right - width + 1", "        end = right + 1"])
        if capability.needs_preceding_observation:
            lines.extend(["        if start < 1:", "            continue"])
        lines.extend(["        if invalid[end] != invalid[start]:", "            continue"])
    boundary = "start - 1" if capability.needs_preceding_observation else "start"
    level_arrays = [name for node, name in arrays if capability.has_returns and node.inferred_type.shape == ("L",)]
    if level_arrays:
        lines.extend(["        if " + " or ".join(f"not math.isfinite({name}[start - 1])" for name in level_arrays) + ":", "            continue"])
    lines.append(f"        elapsed = dates[right] - dates[{boundary}]")
    local_count = partial and capability.has_returns and any(node.label == "observation_count" for node in variables)
    if local_count:
        _, returns_name = min(((node, name) for node, name in arrays if node.label in RETURN_INPUTS),
                              key=lambda item: item[0].label != "returns")
        lines.extend(["        local_observations = 0.0", "        for index in range(start, end):",
                      f"            if math.isfinite({returns_name}[index]):", "                local_observations += 1.0"])
    args: list[str] = []
    for index, node in enumerate(variables):
        name = f"b{index}"
        if node.inferred_type.rank:
            left = "start - 1" if capability.has_returns and node.inferred_type.shape == ("L",) else "start"
            args.append(f"{name}[{left}:end]")
        elif node.label == "observation_count":
            args.append("local_observations" if local_count else "float(end - start)" if capability.has_returns else "float(max(0, end - start - 1))")
        elif node.label == "window_elapsed_days":
            args.append("elapsed")
        elif node.label == "risk_free_return_window":
            args.append("max(0.0, 1.0 + annual) ** (elapsed / 365.0) - 1.0")
        else:
            args.append(name)
    lines.extend(["        try:", f"            value = interval_body({', '.join(args)})",
        "            if math.isfinite(value):", "                result[right] = value",
        "        except Exception:", "            result[right] = np.nan", "    return result"])
    return "\n".join(lines) + "\n"


def scope_call(scope: Any, nodes: Sequence[Any], registry_version: str, namespace: dict[str, Any]) -> str:
    kernel, variables = compile_scope(nodes, scope, registry_version)
    name = f"k{len(namespace)}"
    namespace[name] = kernel
    args = [f"n{node.node_id}" for node in variables] + [f"n{index}" for index in scope.inputs[1:]]
    return f"{name}({', '.join(args)})"

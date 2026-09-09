"""Fixed-signature, shared-node execution of named scalar outputs.

The compiler owns generated code. User text only reaches the restricted typed
parser; no user identifiers are interpolated into executable Python source.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import threading
import time
from typing import Any

import numba
import numpy as np
from numba import types
from numba.core.registry import CPUDispatcher

from . import typed_numba_kernels as kernels
from .multi_output import MULTI_OUTPUT_PORTS, STATUS_OUTPUT_UNAVAILABLE
from .typed_dsl import TypedScalarBundlePlan
from .typed_numba_plan import NumbaPlanCompileError, _numba_type, _operator_call, _workspace_bytes
from compute_policy import NJIT_BACKEND, validate_execution_audit


@dataclass(frozen=True)
class CompiledScalarBundle:
    plan_id: str
    dispatcher: CPUDispatcher
    context_names: tuple[str, ...]
    output_names: tuple[str, ...]
    source: str
    compile_ms: float
    node_count: int
    required_workspace_bytes: int
    multi_output_nodes: tuple[str, ...] = ()

    def compute(self, arguments: tuple[Any, ...], enabled: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.dispatcher(*arguments, enabled)

    def metadata(self) -> dict[str, Any]:
        signatures = [str(value) for value in self.dispatcher.signatures]
        return validate_execution_audit({
            "compiled_plan_id": self.plan_id,
            "compile_status": "compiled",
            "compile_ms": self.compile_ms,
            "kernel_version": kernels.NUMERIC_KERNEL_VERSION,
            "engine_version": kernels.ENGINE_VERSION,
            "required_workspace_bytes": self.required_workspace_bytes,
            "compiled_signatures": signatures,
            "kernel_signatures": {"scalar_bundle": signatures},
            "execution_backend": NJIT_BACKEND,
            "nopython": bool(self.dispatcher.nopython_signatures),
            "python_fallback": 0,
            "python_operator_calls": 0,
            "shared_node_count": self.node_count,
            "output_names": list(self.output_names),
            "multi_output_nodes": list(self.multi_output_nodes),
            "multi_output_call_sites": len(self.multi_output_nodes),
        })


_CACHE: dict[str, CompiledScalarBundle] = {}
_LOCK = threading.RLock()


def bundle_plan_id(plan: TypedScalarBundlePlan) -> str:
    # Input types are part of identity: equal text is not equal computation
    # when context domains, nominal axes or price bases differ.
    payload = {
        "compiler": "scalar-bundle-named-ports-3",
        "expression_hash": plan.expression_hash,
        "dsl": plan.dsl_version,
        "registry": plan.operator_registry_version,
        "kernel": kernels.NUMERIC_KERNEL_VERSION,
        "inputs": {name: value.to_dict() for name, value in plan.context_requirements.items()},
        "outputs": list(plan.roots),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _source(plan: TypedScalarBundlePlan) -> tuple[str, dict[str, Any]]:
    names = tuple(plan.context_requirements)
    positions = {name: index for index, name in enumerate(names)}
    nodes = {node.node_id: node for node in plan.nodes}
    needed_by: dict[int, set[int]] = {node.node_id: set() for node in plan.nodes}
    for output_index, root_id in enumerate(plan.roots.values()):
        pending = [root_id]
        while pending:
            node_id = pending.pop()
            if output_index in needed_by[node_id]:
                continue
            needed_by[node_id].add(output_index)
            pending.extend(nodes[node_id].inputs)
    arguments = [*(f"v{index}" for index in range(len(names))), "enabled"]
    lines = [f"def generated_scalar_bundle({', '.join(arguments)}):"]
    namespace: dict[str, Any] = {"np": np}
    for node in plan.nodes:
        rank = node.inferred_type.rank
        dtype = "np.uint8" if node.inferred_type.is_mask else "np.float64"
        default = (
            f"np.empty({repr((0,) * rank)}, dtype={dtype})" if rank else
            "np.uint8(0)" if node.inferred_type.is_mask else "np.nan"
        )
        if node.inferred_type.kind == "record":
            default = "(" + ", ".join("np.nan" for _ in node.inferred_type.fields) + ",)"
        lines.append(f"    n{node.node_id} = {default}")
        lines.append(f"    ok{node.node_id} = False")
        needed = " or ".join(f"enabled[{index}]" for index in sorted(needed_by[node.node_id]))
        inputs_ok = " and ".join(f"ok{index}" for index in node.inputs)
        condition = f"({needed})" + (f" and {inputs_ok}" if inputs_ok else "")
        if node.kind == "constant":
            expression = repr(float(node.label))
        elif node.kind == "variable":
            expression = f"v{positions[node.label]}"
        else:
            expression = _operator_call(node, tuple(nodes[index] for index in node.inputs), namespace)
        success = "True"
        if node.inferred_type.kind == "record":
            required_ports = [index for index, port in enumerate(MULTI_OUTPUT_PORTS[str(node.operator_id)]) if not port.allow_missing]
            success = " and ".join(f"np.isfinite(n{node.node_id}[{index}])" for index in required_ports) or "True"
        lines.extend([
            f"    if {condition}:",
            "        try:",
            f"            n{node.node_id} = {expression}",
            f"            ok{node.node_id} = {success}",
            "        except Exception:",
            "            pass",
        ])
    lines.extend([
        f"    values = np.full({len(plan.roots)}, np.nan, dtype=np.float64)",
        f"    statuses = np.full({len(plan.roots)}, {kernels.STATUS_NON_FINITE_RESULT}, dtype=np.int16)",
    ])
    for index, root in enumerate(plan.roots.values()):
        if nodes[root].kind == "output":
            lines.extend([
                f"    if enabled[{index}] and ok{root}:",
                f"        statuses[{index}] = {STATUS_OUTPUT_UNAVAILABLE}",
            ])
        lines.extend([
            f"    if enabled[{index}] and ok{root} and np.isfinite(n{root}):",
            f"        values[{index}] = n{root}",
            f"        statuses[{index}] = {kernels.STATUS_OK}",
        ])
    lines.append("    return values, statuses")
    return "\n".join(lines) + "\n", namespace


def compile_scalar_bundle(plan: TypedScalarBundlePlan) -> CompiledScalarBundle:
    """Explicit preparation only. Production evaluation uses warmed lookups."""
    plan_id = bundle_plan_id(plan)
    with _LOCK:
        if plan_id in _CACHE:
            return _CACHE[plan_id]
        start = time.perf_counter()
        try:
            source, namespace = _source(plan)
            exec(compile(source, f"<scalar-bundle:{plan_id}>", "exec"), namespace)
            dispatcher = numba.njit(nogil=True, cache=False)(namespace["generated_scalar_bundle"])
            names = tuple(plan.context_requirements)
            for readonly in (False, True):
                signature = tuple(_numba_type(plan.context_requirements[name], readonly=readonly) for name in names)
                dispatcher.compile((*signature, types.uint8[::1]))
            dispatcher.disable_compile()
        except Exception as exc:
            raise NumbaPlanCompileError("多结果公式无法编译为固定签名 NJIT 计划。", plan_id=plan_id) from exc
        result = CompiledScalarBundle(
            plan_id, dispatcher, names, tuple(plan.roots), source,
            round((time.perf_counter() - start) * 1000, 3), len(plan.nodes), _workspace_bytes(plan),
            tuple(str(node.operator_id) for node in plan.nodes if node.inferred_type.kind == "record"),
        )
        _CACHE[plan_id] = result
        return result

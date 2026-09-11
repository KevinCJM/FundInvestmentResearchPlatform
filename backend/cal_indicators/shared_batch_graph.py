"""Structural common-subexpression elimination across independent metrics.

This module only plans execution. It never evaluates numeric user expressions.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import json
from typing import Any, Callable


SHARED_GRAPH_VERSION = "independent-metrics-cse-2"


@dataclass(frozen=True)
class SharedBatchGraph:
    nodes: tuple[Any, ...]
    roots: tuple[int, ...]
    variable_expressions: dict[int, str]
    needed_by: tuple[tuple[int, ...], ...]
    original_node_count: int

    def audit(self) -> dict[str, Any]:
        operators: dict[str, int] = {}
        for node in self.nodes:
            if node.operator_id:
                operators[node.operator_id] = operators.get(node.operator_id, 0) + 1
        return {
            "sharing_version": SHARED_GRAPH_VERSION,
            "input_node_count": self.original_node_count,
            "shared_node_count": len(self.nodes),
            "eliminated_node_count": self.original_node_count - len(self.nodes),
            "operator_call_sites": operators,
            "root_count": len(self.roots),
        }


def merge_metric_plans(plans, definitions, variable_expression: Callable[..., str], column_index) -> SharedBatchGraph:
    if not plans or len(plans) != len(definitions):
        raise ValueError("Expected matching independent plans and definitions")
    nodes = []
    roots = []
    variables: dict[int, str] = {}
    seen: dict[tuple[Any, ...], int] = {}
    total = 0
    for plan, definition in zip(plans, definitions):
        local = {}
        for node in plan.nodes:
            total += 1
            inputs = tuple(local[parent] for parent in node.inputs)
            structural_type = json.dumps(node.inferred_type.to_dict(), sort_keys=True, separators=(",", ":"))
            binding = ""
            if node.kind == "constant":
                binding = float(node.label).hex()  # Preserve signed zero.
            elif node.kind == "variable":
                binding = variable_expression(node.label, column_index=column_index, definition=definition)
            key = ("operator" if node.operator_id else node.kind, node.operator_id, node.operator_version, structural_type, inputs, binding)
            # Do not equate old/new operator semantics just because their names match.
            if node.operator_id:
                key += (plan.dsl_version, plan.operator_registry_version)
            if key not in seen:
                index = len(nodes)
                seen[key] = index
                nodes.append(replace(node, node_id=index, inputs=inputs,
                                     arguments=tuple((name, local[parent]) for name, parent in node.arguments)))
                if node.kind == "variable":
                    variables[index] = binding
            local[node.node_id] = seen[key]
        roots.append(local[plan.root_id])
    needed = [set() for _ in nodes]
    for metric, root in enumerate(roots):
        pending = [root]
        while pending:
            index = pending.pop()
            if metric in needed[index]:
                continue
            needed[index].add(metric)
            pending.extend(nodes[index].inputs)
    return SharedBatchGraph(tuple(nodes), tuple(roots), variables,
                            tuple(tuple(sorted(items)) for items in needed), total)


def build_batch_source(plans, definitions, column_index, *, parallel, variable_expression, operator_call):
    """Emit one guarded call site per shared node, with requested-root pruning."""
    import math
    import numba
    import numpy as np
    graph = merge_metric_plans(plans, definitions, variable_expression, column_index)
    namespace = {"np": np, "loop": numba.prange if parallel else range, "isfinite": math.isfinite, "math_log": math.log}
    lines = ["def generated_batch(values, starts, ends, elapsed_days, output, statuses, enabled):",
             "    for row in loop(starts.size):", "        start = starts[row]", "        end = ends[row]",
             "        if start < 0 or end - start < 2:", "            for metric in range(output.shape[1]):",
             "                output[row, metric] = np.nan", "                statuses[row, metric] = 1",
             "            continue"]
    # Returns are shared as well, and allocated only if a selected root uses them.
    nav_index = column_index.get("adjusted_nav", 0)
    for binding, logarithmic in (("returns_view", False), ("log_returns_view", True)):
        users = set()
        for node_id, expression in graph.variable_expressions.items():
            if expression == binding:
                users.update(graph.needed_by[node_id])
        if users:
            guard = " or ".join(f"enabled[{index}]" for index in sorted(users))
            value = f"values[{nav_index}, observation + 1] / values[{nav_index}, observation]"
            value = f"math_log({value})" if logarithmic else f"{value} - 1.0"
            lines.extend([f"        {binding} = np.empty(0, dtype=np.float64)", f"        if {guard}:",
                          f"            {binding} = np.empty(end - start - 1, dtype=np.float64)",
                          "            for observation in range(start, end - 1):",
                          f"                {binding}[observation - start] = {value}"])
    for node in graph.nodes:
        index = node.node_id
        rank = node.inferred_type.rank
        dtype = "np.uint8" if node.inferred_type.is_mask else "np.float64"
        default = f"np.empty({repr((0,) * rank)}, dtype={dtype})" if rank else "np.uint8(0)" if node.inferred_type.is_mask else "np.nan"
        if node.inferred_type.kind == "record":
            default = "(" + ", ".join("np.nan" for _ in node.inferred_type.fields) + ",)"
        lines.extend([f"        n{index} = {default}", f"        s{index} = 4"])
        needed = " or ".join(f"enabled[{metric}]" for metric in graph.needed_by[index])
        lines.append(f"        if {needed}:")
        if node.inputs:
            inherited = "max(" + ", ".join(f"s{parent}" for parent in node.inputs) + ")" if len(node.inputs) > 1 else f"s{node.inputs[0]}"
            lines.append(f"            s{index} = {inherited}")
            lines.append(f"            if s{index} == 0:")
            indentation = "                "
        else:
            indentation = "            "
        if node.kind == "constant":
            expression = repr(float(node.label))
        elif node.kind == "variable":
            expression = graph.variable_expressions[index]
        else:
            expression = operator_call(node, tuple(graph.nodes[parent] for parent in node.inputs), namespace)
        if node.inferred_type.kind == "record":
            if node.operator_id == "last_drawdown_interval":
                success = f"n{index}[3] >= 0.0"
            elif node.operator_id == "linear_fit":
                success = f"isfinite(n{index}[0]) and n{index}[4] >= 2.0"
            else:
                raise ValueError("Unregistered computation state contract")
        else:
            success = f"isfinite(n{index})" if rank == 0 else "True"
        unavailable = node.operator_id in {"interval_start", "interval_trough", "interval_recovery", "value_at", "days_between"}
        invalid_status = 7 if unavailable else 4
        if node.operator_id in {"interval_start", "interval_trough", "interval_recovery"}:
            parent = node.inputs[0]
            invalid_status = f"(9 if n{parent}[3] == 0.0 else 10)"
        # Preserve known scalar domain errors without rerunning a failed graph
        # merely to recover its diagnostic. All guards execute inside NJIT.
        guarded_input = None
        comparison = "== 0.0"
        error_status = 2
        if node.operator_id == "divide" and graph.nodes[node.inputs[1]].inferred_type.rank == 0:
            guarded_input = node.inputs[1]
        elif node.operator_id == "reciprocal" and rank == 0:
            guarded_input = node.inputs[0]
        elif node.operator_id in {"sqrt", "log", "require_positive", "require_nonnegative"} and rank == 0:
            guarded_input = node.inputs[0]
            comparison = "<= 0.0" if node.operator_id in {"log", "require_positive"} else "< 0.0"
            error_status = 3
        if guarded_input is not None:
            lines.extend([
                f"{indentation}if n{guarded_input} {comparison}:",
                f"{indentation}    s{index} = {error_status}",
                f"{indentation}else:",
            ])
            indentation += "    "
        # A node failure is isolated. The serial lane is authoritative; the
        # parallel lane keeps the same guards and only parallelizes target rows.
        lines.extend([f"{indentation}try:", f"{indentation}    n{index} = {expression}",
                      f"{indentation}    s{index} = 0 if {success} else {invalid_status}",
                      f"{indentation}except Exception:", f"{indentation}    s{index} = 4"])
    for metric, root in enumerate(graph.roots):
        lines.extend([f"        output[row, {metric}] = n{root} if enabled[{metric}] and s{root} == 0 else np.nan",
                      f"        statuses[row, {metric}] = s{root} if enabled[{metric}] else 8"])
    # Keep exception isolation inside a separately compiled row kernel. A
    # prange loop containing try/except cannot be parallelized by Numba.
    row_lines = ["def generated_row(values, start, end, elapsed_day, output, statuses, enabled):"]
    for line in lines[4:]:
        row_lines.append(line[4:].replace("elapsed_days[row]", "elapsed_day")
                         .replace("output[row, ", "output[").replace("statuses[row, ", "statuses[")
                         .replace("output.shape[1]", "output.size").replace("continue", "return"))
    return "\n".join(row_lines) + "\n", namespace

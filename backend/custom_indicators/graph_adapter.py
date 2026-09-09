"""Bounded authoring DAG to canonical DSL conversion; no numerical execution."""
from __future__ import annotations

from cal_indicators.typed_operators import get_typed_operator_registry
from .errors import ValidationError
from .graph_contracts import AuthoringGraph
from .typed_service import _validate_fixed_constant_arguments
from .series_parameters import PARAMETER_CAPABILITIES

_BINARY = {"add": "+", "subtract": "-", "multiply": "*", "divide": "/", "power": "**"}


def graph_error(code: str, message: str, node_id: str | None = None, parameter: str | None = None) -> ValidationError:
    field = f"nodes.{node_id}" if node_id else "graph"
    return ValidationError(code, message, field, [{
        "code": code, "message": message, "field": field,
        "editor_node_id": node_id, "parameter_id": parameter, "severity": "error",
    }])


def graph_expressions(graph: AuthoringGraph, registry_version: str, result_kind: str,
                      parameter_schema: list[dict] | None = None) -> tuple[dict[str, str], dict[str, str]]:
    """Expand named operands with cycle, depth, arity and size checks."""
    parameters = {item["id"]: item for item in parameter_schema or []}
    output_ids = [output.id for output in graph.outputs]
    if len(output_ids) != len(set(output_ids)):
        raise graph_error("DUPLICATE_OUTPUT", "输出通道 ID 不能重复。")
    if result_kind == "scalar" and output_ids != ["result"]:
        raise graph_error("INVALID_SCALAR_OUTPUT", "标量指标必须只有一个 result 输出。")
    nodes = {node.id: node for node in graph.nodes}
    if len(nodes) != len(graph.nodes):
        raise graph_error("DUPLICATE_NODE", "画布节点 ID 不能重复。")
    if set(nodes) & {f"output_{key}" for key in output_ids}:
        raise graph_error("RESERVED_NODE_ID", "计算节点不能使用最终输出的保留 ID。")
    registry = get_typed_operator_registry(registry_version)
    sources: dict[str, str] = {}
    depths: dict[str, int] = {}
    visiting: set[str] = set()
    dependencies: dict[str, list[str]] = {}
    limit = 1000 if result_kind == "scalar" else 4000

    def select_port(node_id: str, expression: str, port_id: str) -> str:
        if port_id != "value":
            raise graph_error("OUTPUT_PORT_NOT_FOUND", "此节点没有所选结果端口。", node_id)
        return expression

    def visit(node_id: str) -> str:
        if node_id in visiting:
            raise graph_error("GRAPH_CYCLE", "这条连接形成了循环，请断开回到上游的连线。", node_id)
        if node_id in sources:
            return sources[node_id]
        node = nodes.get(node_id)
        if node is None:
            raise graph_error("UNKNOWN_NODE", "连接指向不存在的节点，请重新选择上游。", node_id)
        if len(visiting) >= 20:
            raise graph_error("GRAPH_TOO_DEEP", "计算嵌套不能超过 20 层。", node_id)
        visiting.add(node_id)
        children: list[str] = []
        if node.kind == "variable":
            expression = node.variable_id
        elif node.kind == "parameter":
            if node.parameter_id not in parameters:
                raise graph_error("UNKNOWN_SERIES_PARAMETER", "此参数未在指标中声明。", node_id)
            expression = node.parameter_id
        elif node.kind == "constant":
            expression = repr(node.value)
        else:
            spec = registry.get(node.operator_id)
            if spec is None:
                raise graph_error("UNKNOWN_OPERATOR", f"未找到算子 {node.operator_id}。", node_id)
            arity = node.arity or len(node.arguments)
            if arity not in spec.arities:
                raise graph_error("MISSING_ARGUMENT", "输入尚未完整，请检查节点的必需参数。", node_id)
            names = spec.argument_names(arity)
            missing = set(names) - set(node.arguments)
            unknown = set(node.arguments) - set(names)
            if missing or unknown:
                detail = f"缺少：{', '.join(sorted(missing))}。" if missing else f"未知参数：{', '.join(sorted(unknown))}。"
                raise graph_error("MISSING_ARGUMENT" if missing else "INVALID_ARGUMENTS", detail, node_id, next(iter(sorted(missing or unknown)), None))
            values, fixed_arguments = [], []
            for name in names:
                binding = node.arguments[name]
                if binding.source == "constant":
                    value, fixed_value, fixed = repr(binding.value), binding.value, True
                else:
                    children.append(binding.node_id)
                    value = select_port(binding.node_id, visit(binding.node_id), binding.port_id)
                    child = nodes[binding.node_id]
                    fixed = child.kind == "constant"
                    fixed_value = child.value if fixed else value
                    if child.kind == "parameter" and (node.operator_id, name) in PARAMETER_CAPABILITIES:
                        fixed = True
                        fixed_value = parameters[child.parameter_id]["default"]
                values.append(value)
                fixed_arguments.append({"parameter": name, "source": "constant" if fixed else "expression", "value": fixed_value})
            try:
                _validate_fixed_constant_arguments(node.operator_id, fixed_arguments, registry_version)
            except ValidationError as exc:
                parameter = exc.field.rsplit(".", 1)[-1] if exc.field else None
                raise graph_error(exc.code, exc.message, node_id, parameter) from exc
            if node.operator_id in _BINARY:
                expression = f"({values[0]} {_BINARY[node.operator_id]} {values[1]})"
            elif node.operator_id == "negate":
                expression = f"(-({values[0]}))"
            else:
                expression = f"{spec.operator_id}({', '.join(values)})"
        if len(expression) > limit:
            raise graph_error("FORMULA_TOO_LONG", f"展开公式超过 {limit} 字符；请简化计算步骤。", node_id)
        depths[node_id] = 1 + max((depths[child] for child in children), default=0)
        if depths[node_id] > 20:
            raise graph_error("GRAPH_TOO_DEEP", "计算嵌套不能超过 20 层。", node_id)
        dependencies[node_id] = children
        sources[node_id] = expression
        visiting.remove(node_id)
        return expression

    for node in graph.nodes:
        visit(node.id)
    expressions: dict[str, str] = {}
    reachable: set[str] = set()
    pending = []
    for output in graph.outputs:
        if output.node_id is None:
            raise graph_error("OUTPUT_NOT_CONNECTED", f"请为“{output.label or output.id}”选择最终计算结果。", f"output_{output.id}", "value")
        expressions[output.id] = select_port(output.node_id, visit(output.node_id), output.port_id)
        pending.append(output.node_id)
    while pending:
        current = pending.pop()
        if current not in reachable:
            reachable.add(current)
            pending.extend(dependencies[current])
    unused = set(nodes) - reachable
    if unused:
        first = next(node.id for node in graph.nodes if node.id in unused)
        raise graph_error("UNUSED_NODE", f"有 {len(unused)} 个步骤未接入最终输出。请连接或删除这些步骤。", first)
    return expressions, sources

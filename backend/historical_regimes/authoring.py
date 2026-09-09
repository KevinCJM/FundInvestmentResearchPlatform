"""Pure regime graph/source roundtrip. Never evaluates Python or compiles kernels."""
from __future__ import annotations

import ast
import copy
import keyword
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from custom_indicators.errors import ValidationError
from computation_graph.series_contracts import regime_series_outputs
from computation_graph.series_expression import bind_series_expression
from .formula import CAUSAL_OPERATOR_IDS
from .v2_contracts import inspect_definition_v2, parse_definition_v2
from .v2_registry import NODE_REGISTRY
from .math_presentation import graph_math_presentation

MAX_SOURCE = 100_000


class AuthoringRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    definition: dict[str, Any]
    source_kind: Literal["graph", "formula"]
    source: str = Field(default="", max_length=MAX_SOURCE)
    mode: Literal["realtime", "retrospective"] = "retrospective"
    compact: bool = False


class SourceError(ValueError):
    def __init__(self, message: str, node: ast.AST | None = None):
        super().__init__(message)
        self.line = getattr(node, "lineno", None)


def _symbols(nodes):
    symbols, occupied = {}, {"output", "expose"}
    for node in nodes:
        candidate = node.id.replace("-", "_")
        if not candidate.isidentifier() or keyword.iskeyword(candidate):
            candidate = "node_" + str(len(symbols) + 1)
        root, suffix = candidate, 2
        while candidate in occupied:
            candidate = f"{root}_{suffix}"
            suffix += 1
        symbols[node.id] = candidate
        occupied.add(candidate)
    return symbols


def graph_source(definition, *, compact=False) -> str:
    symbols = _symbols(definition.graph.nodes)
    lines = ["# 每行一个独立计算节点；上下游以 节点.端口 连接。"]
    for node in definition.graph.nodes:
        arguments = []
        for name, value in node.parameters.items():
            if not name.isidentifier() or keyword.iskeyword(name) or name.startswith("_") or name in node.inputs:
                # Source nodes may carry external snapshot fields with arbitrary names.
                break
        else:
            arguments = [f"{name}={value!r}" for name, value in node.parameters.items()]
        if node.parameters and not arguments:
            arguments = [f"_parameters={node.parameters!r}"]
        arguments.extend(f"{name}={symbols[ref.node_id]}.{ref.port}" for name, ref in node.inputs.items())
        if not compact or symbols[node.id] != node.id:
            arguments.append(f"_id={node.id!r}")
        if not compact or node.type_version != NODE_REGISTRY[node.type]["type_version"]:
            arguments.append(f"_version={node.type_version!r}")
        if not compact and node.label is not None:
            arguments.append(f"_label={node.label!r}")
        lines.append(f"{symbols[node.id]} = {node.type.replace('.', '_')}({', '.join(arguments)})")
    lines.append("output(" + ", ".join(f"{name}={symbols[ref.node_id]}.{ref.port}" for name, ref in definition.graph.outputs.items()) + ")")
    lines.append("expose(" + ", ".join(symbols[name] for name in definition.graph.exposed_node_ids) + ")")
    source = "\n".join(lines)
    if len(source) > MAX_SOURCE:
        raise SourceError("完整公式超过 100000 字符，请缩小内联数据或使用已保存数据源。")
    return source


def _literal(node):
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, RecursionError) as exc:
        raise SourceError("参数只接受数值、文本、布尔、列表或字典常量。", node) from exc


def source_graph(source: str, base: dict[str, Any]) -> dict[str, Any]:
    if len(source) > MAX_SOURCE:
        raise SourceError("公式超过长度限制。")
    aliases = {name.replace(".", "_"): name for name in NODE_REGISTRY}
    tree = ast.parse(source, mode="exec")
    if len(tree.body) > 130 or sum(1 for _ in ast.walk(tree)) > 30000:
        raise SourceError("公式节点过多。")
    pending, symbols, commands, expressions = [], {}, {}, []
    old_nodes = {node["id"]: node for node in base.get("graph", {}).get("nodes", [])}
    for statement in tree.body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name):
            symbol = statement.targets[0].id
            call = statement.value
            if symbol in symbols or symbol in {"output", "expose"} or symbol.startswith("_"):
                raise SourceError("节点变量重复或使用保留名称。", statement)
            if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name) or call.func.id not in aliases:
                # Numeric expressions use the exact indicator AST/type compiler.
                symbols[symbol] = symbol
                expressions.append((symbol, call, statement))
                continue
            if call.args:
                raise SourceError("注册图节点使用具名输入和参数；数值公式可以使用指标函数。", statement)
            arguments = {}
            for item in call.keywords:
                if item.arg is None or item.arg in arguments:
                    raise SourceError("不支持参数展开或重复参数。", item)
                arguments[item.arg] = item.value
            identifier = _literal(arguments.pop("_id")) if "_id" in arguments else symbol
            if not isinstance(identifier, str) or identifier in symbols.values():
                raise SourceError("节点 ID 必须是唯一文本。", statement)
            symbols[symbol] = identifier
            pending.append((identifier, aliases[call.func.id], arguments, statement))
        elif isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call) and isinstance(statement.value.func, ast.Name):
            call = statement.value
            if call.func.id not in {"output", "expose"} or call.func.id in commands:
                raise SourceError("仅支持一次 output 和一次 expose 声明。", statement)
            commands[call.func.id] = call
        else:
            raise SourceError("仅支持节点赋值、output 和 expose；禁止执行程序语句。", statement)
    if "output" not in commands:
        raise SourceError("请用 output(state=节点.state) 声明最终状态。")

    def reference(expression):
        if not isinstance(expression, ast.Attribute) or not isinstance(expression.value, ast.Name) or expression.value.id not in symbols:
            raise SourceError("输入须为已声明的 节点.输出端口。", expression)
        return {"node_id": symbols[expression.value.id], "port": expression.attr}

    nodes = []
    for identifier, node_type, arguments, statement in pending:
        metadata = NODE_REGISTRY[node_type]
        slots = {port["name"] for port in metadata["inputs"]}
        old = old_nodes.get(identifier, {})
        version = old.get("type_version", 1) if old.get("type") == node_type else metadata["type_version"]
        node = {"id": identifier, "type": node_type, "type_version": version, "parameters": {}, "inputs": {}}
        if old.get("type") == node_type and old.get("label") is not None:
            node["label"] = old["label"]
        if "_parameters" in arguments:
            params = _literal(arguments.pop("_parameters"))
            if not isinstance(params, dict) or any(not isinstance(k, str) for k in params):
                raise SourceError("_parameters 必须是参数字典。", statement)
            node["parameters"] = params
        for name, value in arguments.items():
            if name == "_version":
                node["type_version"] = _literal(value)
            elif name == "_label":
                node["label"] = _literal(value)
            elif name in slots:
                node["inputs"][name] = reference(value)
            elif name.startswith("_") or name in node["parameters"]:
                raise SourceError("未知保留参数或重复参数。", value)
            else:
                node["parameters"][name] = _literal(value)
        nodes.append(node)
    for identifier, expression, statement in expressions:
        try:
            formula, inputs = bind_series_expression(expression, symbols, CAUSAL_OPERATOR_IDS)
        except ValueError as exc:
            raise SourceError(str(exc), statement) from exc
        old = old_nodes.get(identifier, {})
        nodes.append({"id": identifier, "type": "feature.formula", "type_version": 1,
                      "parameters": {"expression": formula}, "inputs": inputs,
                      **({"label": old["label"]} if old.get("label") else {})})
    root = commands["output"]
    if root.args or any(item.arg is None for item in root.keywords) or len({item.arg for item in root.keywords}) != len(root.keywords):
        raise SourceError("output 必须使用不重复的具名端口。", root)
    outputs = {item.arg: reference(item.value) for item in root.keywords}
    exposed = []
    if "expose" in commands:
        call = commands["expose"]
        if call.keywords:
            raise SourceError("expose 只接受节点变量名。", call)
        for item in call.args:
            if not isinstance(item, ast.Name) or item.id not in symbols:
                raise SourceError("expose 包含未声明节点。", item)
            exposed.append(symbols[item.id])
    metadata = {key: value for key, value in base.get("graph", {}).get("channel_metadata", {}).items() if key in outputs}
    # New numeric roots may be named directly in the formula.
    for key in outputs:
        if key not in {"state", "probabilities", "confidence", "recognition_index", "effective_index", "reason_code"}:
            metadata.setdefault(key, {"label": key})
    return {"nodes": nodes, "outputs": outputs, "exposed_node_ids": exposed,
            **({"channel_metadata": metadata} if metadata else {})}


def resolve_authoring(request: AuthoringRequest) -> dict[str, Any]:
    try:
        raw = copy.deepcopy(request.definition)
        if request.source_kind == "formula":
            raw["graph"] = source_graph(request.source, raw)
        definition = parse_definition_v2(raw)
        inspection = inspect_definition_v2(definition)
        diagnostics = list(inspection["errors"]) + list(inspection["warnings"])
        presentation = {}
        if not inspection["errors"]:
            try:
                presentation = graph_math_presentation(definition)
            except (ValueError, KeyError, TypeError, SyntaxError, RecursionError):
                # Presentation failures must not invalidate or rewrite a saved graph.
                presentation = {"math_error": "当前定义暂时无法生成数学排版，请检查公式及输入连接。"}
        if request.mode == "realtime":
            for node in definition.graph.nodes:
                metadata = NODE_REGISTRY.get(node.type, {})
                if metadata.get("causal") is not True or metadata.get("supports_realtime") is not True or metadata.get("repaints") is not False:
                    diagnostics.append({"code": "NON_CAUSAL_REALTIME_GRAPH", "severity": "error", "path": f"graph.nodes.{node.id}",
                                        "message": "实时模式已禁用事后算法；请移除该节点或切换到事后研究。"})
        return {"valid": not any(item.get("severity", "error") == "error" for item in diagnostics),
                "definition": definition.model_dump(mode="json"), "source": graph_source(definition, compact=request.compact),
                "series_outputs": regime_series_outputs(definition),
                "result_kind": "time_series", "diagnostics": diagnostics, "compile_status": "not_requested", **presentation}
    except (SourceError, SyntaxError, ValidationError, ValueError, RecursionError, KeyError) as exc:
        return {"valid": False, "definition": None, "source": request.source,
                "diagnostics": [{"code": getattr(exc, "code", "INVALID_AUTHORING_SOURCE"), "severity": "error",
                    "message": str(exc), "line": getattr(exc, "line", getattr(exc, "lineno", None))}], "compile_status": "not_requested"}

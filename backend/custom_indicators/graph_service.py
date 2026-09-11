"""Indicator authoring: pure typed checks and independently versioned layout."""
from __future__ import annotations

import ast
import hashlib
import json
from typing import Any

from cal_indicators.typed_dsl import TypedDslError, compose_typed_series_bundle, infer_typed_expression

from .errors import ConflictError, IndicatorDomainError
from .formula_source import canonical_formula_source, editable_formula_latex
from .graph_adapter import graph_error, graph_expressions
from .graph_contracts import AuthoringGraph, CanvasResolveRequest, EditorStateUpdate, FormulaResolveRequest, GraphContext
from .repository import AtomicJsonStore, utc_now
from .series_definitions import normalize_parameter_schema, parameter_variable_types
from .series_parameters import PARAMETER_CONTRACT_VERSION, validate_parameter_definition
from .typed_service import (
    _ensure_context, normalize_variable_latex, render_python_expression_latex,
    variable_latex_symbols, variable_types,
)


def _key(fragment: str) -> str:
    return canonical_formula_source(fragment)


def _infer(expressions: dict[str, str], context: dict[str, Any]):
    limit = 4000 if context["result_kind"] == "time_series" else 1000
    if any(not expression.strip() for expression in expressions.values()):
        raise graph_error("EMPTY_EXPRESSION", "请先输入公式，或从空白画布开始。")
    if any(len(expression) > limit for expression in expressions.values()):
        raise graph_error("FORMULA_TOO_LONG", f"公式不能超过 {limit} 字符。")
    if context["result_kind"] == "scalar" and list(expressions) != ["result"]:
        raise graph_error("INVALID_SCALAR_OUTPUT", "标量指标必须只有一个 result 输出。")
    kwargs = {
        "variable_types": {**variable_types(context["context_kind"], context["dsl_version"]), **parameter_variable_types(context)},
        "dsl_version": context["dsl_version"],
        "operator_registry_version": context["operator_registry_version"],
    }
    normalized = {key: normalize_variable_latex(value) for key, value in expressions.items()}
    if context.get("parameter_contract_version") == PARAMETER_CONTRACT_VERSION:
        validate_parameter_definition({**context, "series_outputs": [
            {"id": key, "expression": canonical_formula_source(value)} for key, value in normalized.items()
        ]})
    if context["result_kind"] == "time_series":
        plan = compose_typed_series_bundle(normalized, **kwargs)
        executable = dict(plan.python_expressions)
    else:
        plan = infer_typed_expression(normalized["result"], allow_non_scalar_root=False, **kwargs)
        executable = {"result": plan.python_expression}
    _ensure_context(plan, context["context_kind"], parameter_variable_types(context))
    canonical = {key: _key(value) for key, value in executable.items()}
    if any(len(value) > limit for value in canonical.values()):
        raise graph_error("FORMULA_TOO_LONG", f"规范公式超过 {limit} 字符。")
    return plan, canonical


def _graph_from_plan(plan, parameter_schema: list[dict] | None = None) -> AuthoringGraph:
    parameters = {item["id"]: item for item in parameter_schema or []}
    payload = plan.graph_payload()
    by_id = {node["id"]: node for node in payload["nodes"]}
    root_ids = set(payload["roots"].values())
    ids = {key: "n_" + hashlib.sha256(_key(node["formula_fragment"]).encode()).hexdigest()[:24] for key, node in by_id.items()}
    def reference(node_id):
        raw = by_id[node_id]
        return {"node_id": ids[node_id]}

    nodes = []
    for raw in payload["nodes"]:
        if raw["kind"] == "constant" and raw["id"] not in root_ids:
            continue
        common = {"id": ids[raw["id"]]}
        if raw["kind"] == "variable":
            if raw["label"] in parameters:
                node = {**common, "kind": "parameter", "parameter_id": raw["label"], "label": parameters[raw["label"]]["label"]}
            else:
                node = {**common, "kind": "variable", "variable_id": raw["label"]}
        elif raw["kind"] == "constant":
            node = {**common, "kind": "constant", "value": ast.literal_eval(raw["formula_fragment"])}
        else:
            arguments = {}
            for argument in raw["arguments"]:
                child = by_id[argument["input_node_id"]]
                child_id = ids[child["id"]]
                if child["kind"] == "constant" and child["id"] not in root_ids:
                    # Equal literals in separate parameters are independently editable.
                    # The execution compiler may still deduplicate their values.
                    owner = f"{ids[raw['id']]}:{argument['name']}"
                    child_id = "c_" + hashlib.sha256(owner.encode()).hexdigest()[:24]
                    nodes.append({"id": child_id, "kind": "constant", "value": ast.literal_eval(child["formula_fragment"])})
                arguments[argument["name"]] = {"source": "node", "node_id": child_id}
            node = {**common, "kind": "operator", "operator_id": raw["operator"]["id"], "arity": len(arguments), "arguments": arguments}
        nodes.append(node)
    if len(nodes) > 128:
        raise graph_error("GRAPH_TOO_LARGE", "展开常量后超过 128 个节点，请简化公式或继续使用公式模式。")
    return AuthoringGraph.model_validate({
        "nodes": nodes,
        "outputs": [{"id": key, **reference(value), "label": "最终结果" if key == "result" else key} for key, value in payload["roots"].items()],
    })


class IndicatorGraphService:
    def __init__(self, indicator_service) -> None:
        self.indicators = indicator_service
        self.store = AtomicJsonStore(indicator_service.workspace_data_dir / "indicator_editor_states.json")

    def _context(self, request: GraphContext) -> dict[str, Any]:
        fields = request.model_dump(include=set(GraphContext.model_fields))
        if not fields["dsl_version"].startswith("2."):
            raise graph_error("GRAPH_DSL_UNSUPPORTED", "此旧版指标暂不支持画布，请继续使用原公式编辑器。")
        # Reuse the existing version gate without compiling or requiring a complete graph.
        normalized = self.indicators._normalize_definition({
            **fields, "name": "画布解析", "expression": "1", "result_kind": "scalar", "output_contract": "scalar",
        })
        schema = normalize_parameter_schema(fields.get("parameter_schema") or [])
        if schema and (fields["result_kind"] != "time_series" or fields.get("parameter_contract_version") != PARAMETER_CONTRACT_VERSION):
            raise graph_error("INVALID_PARAMETER_CONTRACT", "画布可变参数需要当前时序参数契约。")
        return {**{key: normalized.get(key) for key in GraphContext.model_fields},
                "result_kind": fields["result_kind"], "parameter_schema": schema,
                "parameter_contract_version": fields.get("parameter_contract_version")}

    def resolve(self, request: FormulaResolveRequest | CanvasResolveRequest) -> dict[str, Any]:
        sources: dict[str, str] = {}
        graph = request.graph if isinstance(request, CanvasResolveRequest) else None
        context = None
        try:
            context = self._context(request)
            if isinstance(request, FormulaResolveRequest):
                expressions = request.expressions
            else:
                expressions, sources = graph_expressions(graph, context["operator_registry_version"], context["result_kind"], context["parameter_schema"])
            plan, canonical = _infer(expressions, context)
            if graph is None:
                graph = _graph_from_plan(plan, context["parameter_schema"])
                _, sources = graph_expressions(graph, context["operator_registry_version"], context["result_kind"], context["parameter_schema"])
            payload = plan.graph_payload()
            fragments = {_key(node["formula_fragment"]): node for node in payload["nodes"]}
            types, mapping = {}, {}
            for node in graph.nodes:
                compiled = fragments.get(_key(sources[node.id]))
                if compiled is not None:
                    types[node.id], mapping[node.id] = compiled["inferred_type"], compiled["id"]
            symbols = variable_latex_symbols(context["context_kind"])
            fingerprint = hashlib.sha256(json.dumps({"context": context, "expressions": canonical}, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            return {
                "valid": True, "draft_revision": request.draft_revision, "diagnostics": [],
                "graph": graph.model_dump(), "expressions": canonical,
                "editable_latex": {key: editable_formula_latex(value) for key, value in canonical.items()},
                "display_latex": {key: render_python_expression_latex(value, symbols) for key, value in canonical.items()},
                "node_types": types, "editor_to_compiled": mapping, "dag": payload,
                "dependencies": list(plan.context_requirements), "definition_fingerprint": fingerprint,
                "compile_status": "not_requested",
            }
        except (IndicatorDomainError, TypedDslError) as exc:
            diagnostics = getattr(exc, "diagnostics", None) or [{"code": exc.code, "message": exc.message, "field": getattr(exc, "field", None)}]
            # On a type failure only, locate the first invalid editor step. Never compile.
            if isinstance(exc, TypedDslError) and sources and context:
                for node_id, expression in sources.items():
                    try:
                        inferred = infer_typed_expression(
                            expression,
                            variable_types={**variable_types(context["context_kind"], context["dsl_version"]), **parameter_variable_types(context)},
                            dsl_version=context["dsl_version"], operator_registry_version=context["operator_registry_version"],
                        )
                        _ensure_context(inferred, context["context_kind"], parameter_variable_types(context))
                    except (IndicatorDomainError, TypedDslError):
                        diagnostics[0] = {**diagnostics[0], "editor_node_id": node_id}
                        break
            return {"valid": False, "draft_revision": request.draft_revision, "diagnostics": diagnostics, "graph": graph.model_dump() if graph else None, "compile_status": "not_requested"}

    def _saved_resolution(self, indicator_id: str, revision: int) -> dict[str, Any]:
        definition = self.indicators.indicators.get(indicator_id, revision)
        context = {key: definition[key] for key in GraphContext.model_fields if definition.get(key) is not None}
        result_kind = definition.get("result_kind")
        output_field = "series_outputs"
        expressions = (
            {output["id"]: output["expression"] for output in definition.get(output_field, [])}
            if result_kind == "time_series" else {"result": definition["expression"]}
        )
        result = self.resolve(FormulaResolveRequest(source_kind="formula", expressions=expressions, **context))
        if not result["valid"]:
            raise graph_error("EDITOR_STATE_UNAVAILABLE", "此指标版本暂不能还原为画布，请使用原公式编辑器。")
        return result

    def read_state(self, indicator_id: str, revision: int) -> dict[str, Any]:
        self.indicators.indicators.get(indicator_id, revision)
        with self.store.locked():
            payload = self.store.read_unlocked()
            entry = next((item for item in payload["items"] if item["indicator_id"] == indicator_id and item["definition_revision"] == revision), None)
        return entry or {"indicator_id": indicator_id, "definition_revision": revision, "editor_revision": 0, "state": None}

    def save_state(self, indicator_id: str, revision: int, request: EditorStateUpdate) -> dict[str, Any]:
        definition = self.indicators.indicators.get(indicator_id, revision)
        context = {key: definition[key] for key in GraphContext.model_fields if definition.get(key) is not None}
        saved = self._saved_resolution(indicator_id, revision)
        candidate = self.resolve(CanvasResolveRequest(source_kind="graph", graph=request.graph, **context))
        if not candidate["valid"] or candidate["definition_fingerprint"] != saved["definition_fingerprint"]:
            raise ConflictError("EDITOR_DEFINITION_MISMATCH", "画布与此指标版本不一致，请先应用画布并保存指标。")
        with self.store.locked():
            payload = self.store.read_unlocked()
            entry = next((item for item in payload["items"] if item["indicator_id"] == indicator_id and item["definition_revision"] == revision), None)
            current_revision = entry["editor_revision"] if entry else 0
            if current_revision != request.expected_editor_revision:
                raise ConflictError("EDITOR_REVISION_CONFLICT", "布局已被其他操作更新，请重新载入后重试。")
            updated = {
                "indicator_id": indicator_id, "definition_revision": revision,
                "editor_revision": current_revision + 1, "updated_at": utc_now(),
                "definition_fingerprint": saved["definition_fingerprint"],
                "state": request.model_dump(exclude={"expected_editor_revision"}),
            }
            if entry:
                payload["items"].remove(entry)
            payload["items"].append(updated)
            self.store.write_unlocked(payload)
        return updated

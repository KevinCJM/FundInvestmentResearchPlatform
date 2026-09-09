"""Adapt exact Indicator Center definitions to upstream-fed regime calculations.

Only syntax and metadata are adapted here. Calculation uses the existing causal
typed formula plans and their warmed NJIT kernels, with no product-data lookup.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd

from computation_graph.series_operators import (
    is_typed_formula_node as is_shared_formula,
    typed_node_expression as shared_expression,
)
from custom_indicators.errors import ValidationError
from cal_indicators.typed_dsl import TypedDslError
from custom_indicators.formula_source import canonical_formula_source
from custom_indicators.rolling_series import transform_scalar_expression
from custom_indicators.series_parameters import resolve_parameter_values
from custom_indicators.variable_registry import get_variable


def is_typed_formula_node(node, registry):
    return is_shared_formula(node, registry) or bool(registry.get(node.type, {}).get("indicator_reference"))


def typed_node_expressions(node, registry):
    metadata = registry[node.type]
    definition = metadata.get("_indicator_definition")
    if definition is None:
        return {"value": shared_expression(node, registry)}
    if metadata.get("available") is False:
        raise ValueError(metadata["unavailable_reason"])
    if (definition.get("result_kind") or "scalar") == "scalar":
        expression = canonical_formula_source(str(definition.get("expression") or ""))
        return {"value": transform_scalar_expression(expression, node.parameters.get("window", 20)).expression}
    values = resolve_parameter_values(definition, node.parameters)

    class BindParameters(ast.NodeTransformer):
        def visit_Name(self, item):  # noqa: N802
            return ast.copy_location(ast.Constant(values[item.id]), item) if item.id in values else item

    return {
        channel["id"]: ast.unparse(BindParameters().visit(ast.parse(canonical_formula_source(channel["expression"]), mode="eval")))
        for channel in definition["series_outputs"]
    }


def typed_node_expression(node, registry, port=None):
    expressions = typed_node_expressions(node, registry)
    return expressions[port] if port is not None else next(iter(expressions.values()))


def formula_plan_key(node_id, port):
    return node_id if port == "value" else f"{node_id}:{port}"


def validate_typed_series_node(node, registry):
    from .formula import _compose_formula
    frame = pd.DataFrame({name: np.ones(2, dtype=np.float64) for name in node.inputs})
    for expression in typed_node_expressions(node, registry).values():
        _compose_formula(expression, frame)


def register_indicator_nodes(service, registry):
    if service is None or not hasattr(service, "indicators"):
        return
    from .v2_registry import SERIES, _numeric_node, _port
    from .formula import _compose_formula

    # Each exact revision and numerical contract has its own stable node type.
    # Existing nodes are never silently redirected to a newer indicator formula.
    for summary in service.indicators.list_all_versions():
        definition = service.get_indicator(summary["id"], int(summary["revision"]))
        if definition.get("ui_exposed") is False:
            continue
        contract = {key: copy.deepcopy(definition.get(key)) for key in (
            "id", "revision", "result_kind", "context_kind", "expression", "series_outputs",
            "parameter_schema", "dsl_version", "operator_registry_version", "rolling_source",
        )}
        digest = hashlib.sha256(json.dumps(contract, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        identifier = f"indicator.calc_{digest[:24]}"
        if identifier in registry:
            continue
        kind = str(definition.get("result_kind") or "scalar")
        parameters = ({"window": {"type": "integer", "default": 20, "minimum": 2, "maximum": 5000,
                                  "title": "滚动观察数", "description": "在每个时点使用最近这些观察值计算原指标；窗口不足时保留为空。"}}
                      if kind == "scalar" else {
                          item["id"]: {key: item[key] for key in ("type", "label", "default", "minimum", "maximum", "step") if key in item}
                          for item in definition.get("parameter_schema") or []
                      })
        channels = definition.get("series_outputs") if kind == "time_series" else [{"id": "value", "label": "指标时序"}]
        metadata = _numeric_node(identifier, str(definition.get("name") or summary["id"]), "indicator_calculation", [],
                                [{**_port(item["id"], SERIES), "label": item.get("label") or item["id"]} for item in channels or []],
                                {"type": "object", "properties": parameters, "additionalProperties": False})
        metadata.update(kernel_id="typed_formula_plan", kernel_version="typed-indicator-adapter/1",
                        indicator_reference={"id": definition["id"], "revision": definition["revision"],
                                             "definition_hash": digest, "result_kind": kind},
                        _indicator_definition=copy.deepcopy(definition),
                        description=(str(definition.get("description") or "") + (" 按滚动窗口逐期计算。" if kind == "scalar" else "")
                                     + " 输入来自画布上游，无需再次选择产品。"),
                        tags=["指标计算", "指标中心", str(definition.get("name") or "")])
        probe = {identifier: metadata}
        try:
            if kind not in {"scalar", "time_series"}:
                raise ValueError("该指标输出为指标组，暂不能直接接入数值时序端口。")
            if definition.get("context_kind", "single_product") != "single_product":
                raise ValueError("该指标依赖组合上下文，不能用单条上游时序代替。")
            if not str(definition.get("dsl_version", "")).startswith("2."):
                raise ValueError("该版本未使用受支持的 typed NJIT 计算协议。")
            node = SimpleNamespace(type=identifier, parameters={}, inputs={})
            expressions = typed_node_expressions(node, probe)
            inputs = {}
            for expression in expressions.values():
                tree = ast.parse(expression, mode="eval")
                functions = {id(call.func) for call in ast.walk(tree) if isinstance(call, ast.Call)}
                for name in (item.id for item in ast.walk(tree) if isinstance(item, ast.Name) and id(item) not in functions):
                    variable = get_variable(name)
                    if variable is None or variable.kind != "series" or variable.axes not in {("T",), ("time",)}:
                        raise ValueError(f"该指标还需要 {variable.label if variable else name} 上下文，尚不能仅通过数值时序连接计算。")
                    inputs[name] = {**_port(name, SERIES), "label": variable.label, "description": variable.description}
            if not inputs or len(inputs) > 4:
                raise ValueError("指标计算需要 1 至 4 个数值时序输入；请先拆分更复杂的指标。")
            frame = pd.DataFrame({name: np.ones(2, dtype=np.float64) for name in inputs})
            for expression in expressions.values():
                _compose_formula(expression, frame)
            metadata["inputs"] = list(inputs.values())
        except (ValueError, TypeError, KeyError, ValidationError, TypedDslError) as exc:
            metadata.update(available=False, status="unsupported_indicator_contract",
                            unavailable_reason=getattr(exc, "message", str(exc)))
        registry[identifier] = metadata

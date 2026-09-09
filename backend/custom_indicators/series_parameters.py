"""Authoritative, position-aware contracts for runtime-constant series inputs.

This module only handles syntax and small parameter metadata. Market-data
numerics continue to run in the prewarmed fixed-signature NJIT series plan.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import keyword
import math
import re
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping

from cal_indicators.typed_operators import get_typed_operator_registry, TYPED_OPERATOR_REGISTRY_VERSION
from cal_indicators.typed_types import TypedDslError
from .errors import ValidationError
from .formula_source import canonical_formula_source
from .variable_registry import get_variable

PARAMETER_CONTRACT_VERSION = "1.0"
_COUNT = {"type": "integer", "minimum": 1, "maximum": 20_000, "step": 1}
_NUMBER = {"type": "number", "minimum": -1_000_000, "maximum": 1_000_000, "step": 0.01}
PARAMETER_CAPABILITIES: dict[tuple[str, str], dict[str, Any]] = {
    **{(op, "window"): {**_COUNT, "label": "窗口期数"}
       for op in ("rolling_mean", "rolling_std", "rolling_min", "rolling_max")},
    **{(op, "min_periods"): {**_COUNT, "label": "最少有效观察数"}
       for op in ("rolling_mean", "rolling_std", "rolling_min", "rolling_max")},
    ("recursive_smooth", "periods"): {**_COUNT, "label": "平滑周期"},
    ("lag", "periods"): {**_COUNT, "label": "滞后期数"},
    ("difference", "periods"): {**_COUNT, "label": "差分期数"},
    ("clip", "lower"): {**_NUMBER, "label": "裁剪下界"},
    ("clip", "upper"): {**_NUMBER, "label": "裁剪上界"},
}


def _error(message: str, field: str = "parameters", code: str = "INVALID_SERIES_PARAMETER") -> ValidationError:
    return ValidationError(code, message, field=field)


def _trees(definition: Mapping[str, Any]):
    outputs = definition.get("series_outputs") or []
    if not isinstance(outputs, list) or not 1 <= len(outputs) <= 8:
        raise _error("需要 1 至 8 个时序输出。", "series_outputs")
    for index, output in enumerate(outputs):
        expression = str(output.get("expression") or "")
        if not expression or len(expression) > 4000:
            raise _error("请先完成公式，单个通道公式不超过 4000 字符。", "series_outputs")
        try:
            tree = ast.parse(canonical_formula_source(expression), mode="eval")
        except (SyntaxError, ValueError, RecursionError, TypedDslError) as exc:
            raise _error("请先修正公式语法，再设置计算参数。", "series_outputs") from exc
        if sum(1 for _ in ast.walk(tree)) > 4096:
            raise _error("公式节点过多。", "series_outputs")
        yield index, output, tree


def _calls(tree: ast.Expression, registry_version: str | None):
    try:
        registry = get_typed_operator_registry(registry_version or TYPED_OPERATOR_REGISTRY_VERSION)
    except (ValueError, KeyError, TypedDslError) as exc:
        raise _error("不支持的算子版本。", "operator_registry_version") from exc
    for ordinal, node in enumerate(ast.walk(tree)):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        spec = registry.get(node.func.id)
        if spec is None or node.keywords:
            continue
        try:
            names = spec.argument_names(len(node.args))
        except (ValueError, KeyError, TypedDslError):
            continue
        yield ordinal, node, names


def _literal(node: ast.AST) -> int | float | None:
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        try:
            return node.value if math.isfinite(node.value) else None
        except OverflowError:
            return None
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = _literal(node.operand)
        return None if value is None else (-value if isinstance(node.op, ast.USub) else value)
    return None


def _candidate_id(output_id: str, ordinal: int, name: str, tree: ast.AST) -> str:
    digest = hashlib.sha256(ast.dump(tree).encode()).hexdigest()[:16]
    return f"{output_id}:{ordinal}:{name}:{digest}"


def inspect_parameter_inputs(definition: Mapping[str, Any]) -> dict[str, Any]:
    if definition.get("result_kind") != "time_series":
        raise _error("可变参数仅适用于时序指标。", "result_kind")
    schema = {item["id"]: item for item in definition.get("parameter_schema") or []}
    candidates: list[dict[str, Any]] = []
    for _, output, tree in _trees(definition):
        for ordinal, call, names in _calls(tree, definition.get("operator_registry_version")):
            for index, name in enumerate(names):
                policy = PARAMETER_CAPABILITIES.get((call.func.id, name))
                if policy is None:
                    continue
                argument = call.args[index]
                parameter_id = argument.id if isinstance(argument, ast.Name) and argument.id in schema else None
                value = schema[parameter_id]["default"] if parameter_id else _literal(argument)
                if value is None:
                    continue
                candidates.append({
                    "id": _candidate_id(str(output["id"]), ordinal, name, tree),
                    "output_id": output["id"], "output_label": output.get("label") or output["id"],
                    "operator_id": call.func.id, "argument": name,
                    "value": value, "parameter_id": parameter_id,
                    "source_expression": ast.unparse(call)[:400], "position": ordinal + 1, **policy,
                })
    return {"contract_version": PARAMETER_CONTRACT_VERSION, "candidates": candidates}


def _valid_number(value: Any, field: str) -> float:
    try:
        valid = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid:
        raise _error("参数必须是有限数值，不能是空值、布尔值或文本。", field)
    return float(value)


def _on_step(value: float, minimum: float, step: float) -> bool:
    try:
        quotient = (Decimal(str(value)) - Decimal(str(minimum))) / Decimal(str(step))
        return abs(quotient - quotient.to_integral_value()) <= Decimal("0.00000001")
    except (InvalidOperation, ZeroDivisionError):
        return False


def resolve_parameter_values(definition: Mapping[str, Any], supplied: Mapping[str, Any] | None) -> dict[str, int | float]:
    schema = {item["id"]: item for item in definition.get("parameter_schema") or []}
    values = dict(supplied or {})
    unknown = sorted(set(values) - set(schema))
    if unknown:
        raise _error(f"参数 {unknown[0]} 未开放，不能在计算时覆盖。", f"parameters.{unknown[0]}", "UNKNOWN_SERIES_PARAMETER")
    resolved: dict[str, int | float] = {}
    for name, item in schema.items():
        field = f"parameters.{name}"
        value = _valid_number(values[name] if name in values else item["default"], field)
        if item["type"] == "integer" and not value.is_integer():
            raise _error(f"{item['label']} 必须是整数。", field)
        if not item["minimum"] <= value <= item["maximum"]:
            raise _error(f"{item['label']} 必须在 {item['minimum']} 至 {item['maximum']} 之间。", field)
        if not _on_step(value, item["minimum"], item["step"]):
            raise _error(f"{item['label']} 必须从最小值起按步长 {item['step']} 取值。", field)
        resolved[name] = int(value) if item["type"] == "integer" else value
    validate_parameter_relations(definition, resolved)
    return resolved


def validate_parameter_relations(definition: Mapping[str, Any], values: Mapping[str, Any]) -> None:
    for _, _, tree in _trees(definition):
        for _, call, names in _calls(tree, definition.get("operator_registry_version")):
            arguments = {name: values.get(node.id) if isinstance(node, ast.Name) else _literal(node)
                         for name, node in zip(names, call.args)}
            op = call.func.id
            window = arguments.get("window")
            if op.startswith("rolling_") and window is not None:
                minimum = arguments.get("min_periods", window)
                ddof = arguments.get("ddof", 0)
                if minimum is not None and minimum > window:
                    raise _error("最少有效观察数不能大于窗口期数。")
                if ddof is not None and ddof >= window:
                    raise _error("自由度修正必须小于窗口期数。")
            if op == "clip":
                lower, upper = arguments.get("lower"), arguments.get("upper")
                if lower is not None and upper is not None and lower > upper:
                    raise _error("裁剪下界不能大于上界。")


def validate_parameter_definition(definition: Mapping[str, Any]) -> None:
    schema = {item["id"]: item for item in definition.get("parameter_schema") or []}
    if len(schema) != len(definition.get("parameter_schema") or []):
        raise _error("参数代码不能重复。", "parameter_schema")
    registry = get_typed_operator_registry(definition.get("operator_registry_version") or TYPED_OPERATOR_REGISTRY_VERSION)
    for name, item in schema.items():
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", name) or keyword.iskeyword(name) or get_variable(name) or name in registry:
            raise _error("参数代码必须是合法且不与数据变量、算子重名的代码。", f"parameter_schema.{name}.id")
        for field in ("default", "minimum", "maximum", "step"):
            _valid_number(item.get(field), f"parameter_schema.{name}.{field}")
    used: set[str] = set()
    for _, _, tree in _trees(definition):
        allowed: set[int] = set()
        for _, call, names in _calls(tree, definition.get("operator_registry_version")):
            for index, argument_name in enumerate(names):
                node = call.args[index]
                if not isinstance(node, ast.Name) or node.id not in schema:
                    continue
                policy = PARAMETER_CAPABILITIES.get((call.func.id, argument_name))
                if policy is None:
                    raise _error(f"{call.func.id} 的 {argument_name} 不允许参数化。", "series_outputs")
                item = schema[node.id]
                if item["type"] != policy["type"] or item["minimum"] < policy["minimum"] or item["maximum"] > policy["maximum"]:
                    raise _error(f"{item['label']} 的类型或范围超出该输入允许的范围。", f"parameter_schema.{node.id}")
                allowed.add(id(node))
                used.add(node.id)
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in schema and id(node) not in allowed:
                raise _error(f"参数 {node.id} 只能直接用于系统允许的参数输入位置。", "series_outputs")
    unused = set(schema) - used
    if unused:
        raise _error(f"参数 {sorted(unused)[0]} 没有被公式使用，请删除或重新关联。", "parameter_schema")
    resolve_parameter_values(definition, {})


def bind_parameter_input(definition: Mapping[str, Any], *, candidate_id: str | None = None,
                         parameter_id: str | None = None, fixed_parameter_id: str | None = None) -> dict[str, Any]:
    if bool(candidate_id) == bool(fixed_parameter_id) or (parameter_id and not candidate_id):
        raise _error("请选择一个输入进行关联，或选择一个参数固定，不能同时操作。", "candidate_id")
    output = copy.deepcopy(dict(definition))
    if output.get("result_kind") != "time_series":
        raise _error("可变参数仅适用于时序指标。", "result_kind")
    # Old parameter metadata is not an authorization to open historical inputs.
    if output.get("parameter_schema") and output.get("parameter_contract_version") != PARAMETER_CONTRACT_VERSION:
        raise _error("请先按原版本默认值载入并保存固定公式，再开放参数。", "parameter_schema")
    schema = output.setdefault("parameter_schema", [])
    by_id = {item["id"]: item for item in schema}
    trees = list(_trees(output))
    if fixed_parameter_id:
        if fixed_parameter_id not in by_id:
            raise _error("找不到需要固定的参数。")
        value = by_id[fixed_parameter_id]["default"]

        class Fix(ast.NodeTransformer):
            def visit_Name(self, node: ast.Name) -> ast.AST:
                return ast.copy_location(ast.Constant(value=value), node) if node.id == fixed_parameter_id else node

        for _, _, tree in trees:
            Fix().visit(tree)
        output["parameter_schema"] = [item for item in schema if item["id"] != fixed_parameter_id]
    else:
        candidate = next((item for item in inspect_parameter_inputs(output)["candidates"] if item["id"] == candidate_id), None)
        if candidate is None or candidate["parameter_id"]:
            raise _error("可调输入已变化，请重新识别。", "candidate_id", "STALE_PARAMETER_CANDIDATE")
        if parameter_id:
            if parameter_id not in by_id:
                raise _error("找不到要共享的参数。", "parameter_id")
        else:
            base = candidate["argument"]
            index = 1
            parameter_id = f"{base}_{index}"
            while parameter_id in by_id or get_variable(parameter_id):
                index += 1
                parameter_id = f"{base}_{index}"
            step = candidate["step"]
            if candidate["type"] == "number":
                decimal_places = max(0, -Decimal(str(candidate["value"])).as_tuple().exponent)
                step = min(step, 10.0 ** -decimal_places)
            schema.append({"id": parameter_id, "label": candidate["label"], "type": candidate["type"],
                           "default": candidate["value"], "minimum": candidate["minimum"],
                           "maximum": candidate["maximum"], "step": step, "description": ""})
        for _, channel, tree in trees:
            for ordinal, call, names in _calls(tree, output.get("operator_registry_version")):
                for index, name in enumerate(names):
                    if _candidate_id(str(channel["id"]), ordinal, name, tree) == candidate_id:
                        call.args[index] = ast.copy_location(ast.Name(id=parameter_id, ctx=ast.Load()), call.args[index])
                        break
    if len(output["parameter_schema"]) > 16:
        raise _error("最多开放 16 个计算参数。", "parameter_schema")
    for _, channel, tree in trees:
        channel["expression"] = ast.unparse(ast.fix_missing_locations(tree).body)
        channel.pop("editable_latex", None)
    output["expression"] = output["series_outputs"][0]["expression"]
    output["parameter_contract_version"] = PARAMETER_CONTRACT_VERSION
    if output.get("rolling_source"):
        output["rolling_source"]["detached"] = True
    output.pop("rolling_transform", None)
    if isinstance(output.get("template_origin"), dict):
        output["template_origin"]["detached"] = True
    validate_parameter_definition(output)
    return output


def parameter_hash(parameters: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(dict(parameters), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()

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

from cal_indicators.parameter_policy import constant_number as _literal, ARGUMENT_DEFAULTS, configuration_violations, parameter_policy
from cal_indicators.typed_operators import get_typed_operator_registry, TYPED_OPERATOR_REGISTRY_VERSION
from cal_indicators.typed_types import TypedDslError
from .errors import ValidationError
from .formula_source import canonical_formula_source
from .variable_registry import get_variable

PARAMETER_CONTRACT_VERSION = "1.0"
TIME_SERIES_RESULT_KIND = "time_series"


def _error(message: str, field: str = "parameters", code: str = "INVALID_SERIES_PARAMETER") -> ValidationError:
    return ValidationError(code, message, field=field)


def _channels(definition: Mapping[str, Any]) -> tuple[str, Any]:
    """One formula channel per output. A scalar indicator has exactly one, so
    every rule below is written once and applies to both result kinds."""

    if definition.get("result_kind") == TIME_SERIES_RESULT_KIND:
        return "series_outputs", definition.get("series_outputs") or []
    return "expression", [{
        "id": "result",
        "label": definition.get("name") or "结果",
        "expression": definition.get("expression") or "",
    }]


def _trees(definition: Mapping[str, Any]):
    field, outputs = _channels(definition)
    if not isinstance(outputs, list) or not 1 <= len(outputs) <= 8:
        raise _error("需要 1 至 8 个时序输出。", field)
    for index, output in enumerate(outputs):
        expression = str(output.get("expression") or "")
        if not expression or len(expression) > 4000:
            raise _error("请先完成公式，单个通道公式不超过 4000 字符。", field)
        try:
            tree = ast.parse(canonical_formula_source(expression), mode="eval")
        except (SyntaxError, ValueError, RecursionError, TypedDslError) as exc:
            raise _error("请先修正公式语法，再设置计算参数。", field) from exc
        if sum(1 for _ in ast.walk(tree)) > 4096:
            raise _error("公式节点过多。", field)
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
        yield ordinal, node, names, spec


def _candidate_id(output_id: str, ordinal: int, name: str, tree: ast.AST) -> str:
    digest = hashlib.sha256(ast.dump(tree).encode()).hexdigest()[:16]
    return f"{output_id}:{ordinal}:{name}:{digest}"


def inspect_parameter_inputs(definition: Mapping[str, Any]) -> dict[str, Any]:
    schema = {item["id"]: item for item in definition.get("parameter_schema") or []}
    candidates: list[dict[str, Any]] = []
    for _, output, tree in _trees(definition):
        for ordinal, call, names, spec in _calls(tree, definition.get("operator_registry_version")):
            for index, name in enumerate(names):
                policy = parameter_policy(spec, len(call.args), name)
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
        if (not item["minimum"] <= value <= item["maximum"]
                or (item.get("exclusive_minimum") and value == item["minimum"])
                or (item.get("exclusive_maximum") and value == item["maximum"])):
            raise _error(f"{item['label']} 必须在 {item['minimum']} 至 {item['maximum']} 之间。", field)
        if not _on_step(value, item["minimum"], item["step"]):
            raise _error(f"{item['label']} 必须从最小值起按步长 {item['step']} 取值。", field)
        resolved[name] = int(value) if item["type"] == "integer" else value
    validate_parameter_relations(definition, resolved)
    return resolved


def validate_parameter_relations(definition: Mapping[str, Any], values: Mapping[str, Any]) -> None:
    """Cross-argument maths that no single contract can express.

    Keyed by argument name, not by operator id: any operator that accepts both
    of a pair inherits the rule without being listed anywhere.
    """

    def resolve(node: ast.AST | None) -> Any:
        if node is None:
            return None
        return values.get(node.id) if isinstance(node, ast.Name) else _literal(node)

    for _, _, tree in _trees(definition):
        for _, call, names, _spec in _calls(tree, definition.get("operator_registry_version")):
            arguments = {name: resolve(node) for name, node in zip(names, call.args)}
            # A window reduction carries its width on the rolling_window it reduces.
            source = call.args[0] if call.args else None
            if "window" not in arguments and isinstance(source, ast.Call) and isinstance(source.func, ast.Name):
                inner = source.func.id
                if inner == "rolling_window" and len(source.args) >= 2:
                    arguments["window"] = resolve(source.args[1])
            if "ddof" not in arguments and "window" in arguments:
                # Omitted ddof still applies; its declared default is the contract.
                arguments["ddof"] = ARGUMENT_DEFAULTS.get((_spec.operator_id, "ddof"))
            for left, right, message, strict in (
                ("min_periods", "window", "最少有效观察数不能大于窗口期数。", False),
                ("ddof", "window", "自由度修正必须小于窗口期数。", True),
                ("lower", "upper", "裁剪下界不能大于上界。", False),
            ):
                first, second = arguments.get(left), arguments.get(right)
                if first is None or second is None:
                    continue
                if first >= second if strict else first > second:
                    raise _error(message)


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
        for _, call, names, spec in _calls(tree, definition.get("operator_registry_version")):
            for index, argument_name in enumerate(names):
                node = call.args[index]
                if not isinstance(node, ast.Name) or node.id not in schema:
                    continue
                policy = parameter_policy(spec, len(call.args), argument_name)
                if policy is None:
                    raise _error(f"{call.func.id} 的 {argument_name} 不允许参数化。", "series_outputs")
                item = schema[node.id]
                if item["type"] != policy["type"] or item["minimum"] < policy["minimum"] or item["maximum"] > policy["maximum"]:
                    raise _error(f"{item['label']} 的类型或范围超出该输入允许的范围。", f"parameter_schema.{node.id}")
                for bound in ("minimum", "maximum"):
                    if (policy.get(f"exclusive_{bound}") and item[bound] == policy[bound]
                            and not item.get(f"exclusive_{bound}")):
                        raise _error(f"{item['label']} 不能包含该输入的边界值。", f"parameter_schema.{node.id}")
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
                           "maximum": candidate["maximum"], "step": step, "description": "",
                           **{key: candidate[key] for key in ("exclusive_minimum", "exclusive_maximum") if candidate.get(key)}})
        for _, channel, tree in trees:
            for ordinal, call, names, _spec in _calls(tree, output.get("operator_registry_version")):
                for index, name in enumerate(names):
                    if _candidate_id(str(channel["id"]), ordinal, name, tree) == candidate_id:
                        call.args[index] = ast.copy_location(ast.Name(id=parameter_id, ctx=ast.Load()), call.args[index])
                        break
    if len(output["parameter_schema"]) > 16:
        raise _error("最多开放 16 个计算参数。", "parameter_schema")
    for _, channel, tree in trees:
        channel["expression"] = ast.unparse(ast.fix_missing_locations(tree).body)
        channel.pop("editable_latex", None)
    # A scalar channel is synthetic, so the rewritten formula is read back from
    # the channel rather than from ``series_outputs``.
    output["expression"] = trees[0][1]["expression"]
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


def raise_on_configuration_violations(nodes, roots, operator_registry_version, parameter_ids=()) -> None:
    """Fail a compile whose configuration input is fed by data, not a constant."""

    diagnostics = configuration_violations(nodes, roots, operator_registry_version, parameter_ids)
    if not diagnostics:
        return
    first = diagnostics[0]
    raise ValidationError(str(first["code"]), str(first["message"]), field="series_outputs",
                          diagnostics=[{**item, "field": "series_outputs"} for item in diagnostics])

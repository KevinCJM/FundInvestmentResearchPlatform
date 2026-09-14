"""Contract-derived policy for numeric configuration inputs.

``OperatorSignature.inputs`` is the single source of truth.  An argument
position is a *configuration input* when its contract carries a configuration
tag; such a position must hold a definition-level constant, and is exactly the
kind of position that may be opened as a runtime parameter.  There is no
operator whitelist anywhere in the stack: the only way to change the answer is
to change the signature.

Rank-0 scalar contract grammar::

    scalar                    ordinary value; any scalar expression is allowed
    scalar<dimensionless>     ordinary value that must be dimensionless
    scalar<count>             configuration: integer in [1, COUNT_LIMIT]
    scalar<count:LOW..HIGH>   configuration: integer; either bound may be empty
    scalar<probability>       configuration: number in (0, 1)
    scalar<const>             configuration: any finite number

Labels and authoring defaults cannot be derived from a type, so they are
declared here once.  Neither gates anything: an entry only changes the text or
the prefilled number shown in the composer.
"""
from __future__ import annotations

import ast
import math
import re
from typing import Any, Iterable, Mapping, Sequence

from .typed_operators import TypedOperatorSpec

COUNT_LIMIT = 20_000
NUMBER_LIMIT = 1_000_000

_CONTRACT = re.compile(
    r"scalar(?:<(?P<tag>[a-z_]+)"
    r"(?::(?P<low>-?\d+(?:\.\d+)?)?\.\.(?P<high>-?\d+(?:\.\d+)?)?)?>)?"
)
_CONFIGURATION_TAGS: dict[str, dict[str, Any]] = {
    "count": {"constant_kind": "integer", "minimum": 1.0, "maximum": float(COUNT_LIMIT)},
    # A degenerate quantile is min_value/max_value; keep the ends closed out.
    "probability": {"constant_kind": "number", "minimum": 0.0, "maximum": 1.0, "exclusive": True},
    "const": {"constant_kind": "number", "minimum": None, "maximum": None},
}

# Authoring seed values shown in the composer before the user types anything.
ARGUMENT_DEFAULTS: dict[tuple[str, str], float] = {
    ("rolling_apply", "window"): 20, ("rolling_apply", "min_periods"): 1,
    ("rolling_window", "window"): 20, ("rolling_window", "min_periods"): 20,
    ("rolling_mean", "window"): 20, ("rolling_mean", "min_periods"): 20,
    ("rolling_std", "window"): 20, ("rolling_std", "ddof"): 0, ("rolling_std", "min_periods"): 20,
    ("rolling_min", "window"): 20, ("rolling_min", "min_periods"): 20,
    ("rolling_max", "window"): 20, ("rolling_max", "min_periods"): 20,
    ("recursive_smooth", "periods"): 3, ("recursive_smooth", "initial"): 50,
    ("lag", "periods"): 1, ("difference", "periods"): 1,
    ("variance", "ddof"): 1, ("std", "ddof"): 1,
    ("quantile", "probability"): 0.5, ("quantile_where", "probability"): 0.5,
    ("clip", "lower"): 0.0, ("clip", "upper"): 1.0,
    ("divide_or_default", "default"): 0.0,
}

ARGUMENT_LABELS: dict[str, str] = {
    "calculation": "区间计算内容", "dates": "观察日期（自动绑定）", "annual_rate": "年度配置（自动绑定）",
    "values": "输入值", "levels": "净值或价格序列", "drawdowns": "回撤序列",
    "lhs": "输入 A", "rhs": "输入 B", "A": "输入 A", "B": "输入 B",
    "numerator": "分子", "denominator": "分母", "default": "分母为零时的默认值",
    "base": "底数", "exponent": "指数", "lower": "下界", "upper": "上界",
    "ddof": "自由度修正", "probability": "概率", "initial": "递归初始值",
    "matrix": "矩阵", "lhs_matrix": "左侧矩阵", "rhs_matrix": "右侧矩阵", "vector": "向量",
    "asset_returns": "多资产收益矩阵", "x": "自变量 X", "y": "因变量 Y",
    "mask": "布尔条件", "if_true": "条件成立值", "if_false": "条件不成立值",
    "periods": "间隔期数", "window": "窗口期数", "min_periods": "最少有效观察数",
    "start_date": "开始日期", "end_date": "结束日期", "position": "观察位置（从0开始）",
    "fit": "线性拟合结果", "interval": "最大回撤区间",
}
OPERATOR_ARGUMENT_LABELS: dict[tuple[str, str], str] = {
    **{(operator_id, "values"): "待处理数值" for operator_id in (
        "rolling_window", "rolling_mean", "rolling_std", "rolling_min", "rolling_max", "recursive_smooth",
    )},
    ("recursive_smooth", "periods"): "平滑期数",
    ("lag", "periods"): "滞后期数",
    ("difference", "periods"): "差分期数",
    ("clip", "lower"): "裁剪下界",
    ("clip", "upper"): "裁剪上界",
    ("value_at", "values"): "数值或日期序列",
}


def argument_label(operator_id: str, name: str) -> str:
    return OPERATOR_ARGUMENT_LABELS.get((operator_id, name)) or ARGUMENT_LABELS.get(name, name)


def contract_policy(contract: str) -> dict[str, Any] | None:
    """Return the configuration policy declared by one contract, or ``None``."""

    match = _CONTRACT.fullmatch(str(contract).strip())
    if match is None:
        return None
    tag = _CONFIGURATION_TAGS.get(match.group("tag") or "")
    if tag is None:
        return None
    low, high = match.group("low"), match.group("high")
    ranged = ".." in match.group(0)
    return {
        "source_policy": "fixed_constant",
        "constant_kind": tag["constant_kind"],
        # An explicit range replaces the tag bounds; an empty side is unbounded.
        "minimum": float(low) if low is not None else (None if ranged else tag["minimum"]),
        "maximum": float(high) if high is not None else (None if ranged else tag["maximum"]),
        "exclusive": bool(tag.get("exclusive")) and not ranged,
    }


def _narrow(left: dict[str, Any] | None, right: dict[str, Any] | None) -> dict[str, Any] | None:
    """Intersect two policies; a position is configuration only if both agree."""

    if left is None or right is None:
        return None
    bounds = {}
    for field, combine in (("minimum", max), ("maximum", min)):
        values = [item[field] for item in (left, right) if item[field] is not None]
        bounds[field] = combine(values) if values else None
    kind = "integer" if "integer" in (left["constant_kind"], right["constant_kind"]) else "number"
    return {"source_policy": "fixed_constant", "constant_kind": kind,
            "exclusive": bool(left.get("exclusive") or right.get("exclusive")), **bounds}


def configuration_arguments(spec: TypedOperatorSpec, arity: int) -> dict[str, dict[str, Any]]:
    """Configuration inputs declared by every signature of ``arity``."""

    if arity not in spec.arities:
        return {}
    names = spec.argument_names(arity)
    merged: dict[str, dict[str, Any] | None] = {}
    for signature in spec.signatures:
        if len(signature.inputs) != arity:
            continue
        for name, contract in zip(names, signature.inputs):
            policy = contract_policy(contract)
            merged[name] = policy if name not in merged else _narrow(merged[name], policy)
    return {
        name: {**policy, "default": ARGUMENT_DEFAULTS.get((spec.operator_id, name)),
               "label": argument_label(spec.operator_id, name)}
        for name, policy in merged.items() if policy is not None
    }


def parameter_policy(spec: TypedOperatorSpec, arity: int, name: str) -> dict[str, Any] | None:
    """Runtime-parameter schema for one configuration input, or ``None``.

    Runtime parameters need finite bounds and a step; unbounded contracts fall
    back to the registry-wide numeric limits.
    """

    policy = configuration_arguments(spec, arity).get(name)
    if policy is None:
        return None
    integer = policy["constant_kind"] == "integer"
    low, high = (1.0, float(COUNT_LIMIT)) if integer else (-float(NUMBER_LIMIT), float(NUMBER_LIMIT))
    step = 1 if integer else 0.01
    minimum = policy["minimum"] if policy["minimum"] is not None else low
    maximum = policy["maximum"] if policy["maximum"] is not None else high
    return {
        "type": "integer" if integer else "number",
        "minimum": int(minimum) if integer else minimum,
        "maximum": int(maximum) if integer else maximum,
        "step": step,
        **({"exclusive_minimum": True, "exclusive_maximum": True} if policy.get("exclusive") else {}),
        "label": policy["label"],
    }


# Arithmetic over constants is still a constant; nothing else is.
_CONSTANT_FOLDING_OPERATORS = frozenset({"negate", "add", "subtract", "multiply", "divide", "power"})


def _constant_node(node_id: int, nodes: Mapping[int, Any], memo: dict[int, bool]) -> bool:
    cached = memo.get(node_id)
    if cached is not None:
        return cached
    node = nodes[node_id]
    constant = node.kind == "constant" or bool(
        node.operator_id in _CONSTANT_FOLDING_OPERATORS
        and node.inputs
        and all(_constant_node(int(child), nodes, memo) for child in node.inputs)
    )
    memo[node_id] = constant
    return constant


CONFIGURATION_MUST_BE_CONSTANT = "SERIES_CONFIGURATION_MUST_BE_CONSTANT"


def configuration_constant_message(operator_id: str, name: str) -> str:
    """One wording for one rule, whichever lane catches the violation."""

    return (
        f"{operator_id} 的参数 {name} 必须是定义级有限数值常量或已开放的计算参数；"
        "修改该参数应创建新的指标公式或版本。"
    )


def configuration_violations(
    nodes: Sequence[Any],
    roots: Sequence[int],
    operator_registry_version: str,
    parameter_ids: Iterable[str] = (),
) -> list[dict[str, Any]]:
    """Report configuration inputs that are not definition-level constants.

    Shared by both result kinds and by every authoring lane, so the answer can
    never drift between the scalar plan, the series bundle and the canvas.
    Returns diagnostics instead of raising: each caller owns its error type.
    """

    from .rolling_scope import outside_nodes
    from .typed_operators import get_typed_operator_registry

    by_id = {int(node.node_id): node for node in nodes}
    opened = set(parameter_ids)
    registry = get_typed_operator_registry(operator_registry_version)
    memo: dict[int, bool] = {}
    diagnostics: list[dict[str, Any]] = []
    for node in outside_nodes(nodes, tuple(roots)):
        spec = registry.get(str(node.operator_id or ""))
        if spec is None:
            continue
        configuration = configuration_arguments(spec, len(node.arguments))
        if not configuration:
            continue
        for name, input_node_id in node.arguments:
            if name not in configuration:
                continue
            source = by_id[int(input_node_id)]
            if source.kind == "variable" and str(source.label) in opened:
                continue
            if _constant_node(int(input_node_id), by_id, memo):
                continue
            diagnostics.append({
                "code": CONFIGURATION_MUST_BE_CONSTANT,
                "message": configuration_constant_message(str(node.operator_id), name),
                "node_id": int(node.node_id),
                "operator": node.operator_id,
                "parameter": name,
            })
    return diagnostics


def constant_number(node: ast.AST) -> float | None:
    """Fold finite scalar configuration arithmetic without evaluating arbitrary AST."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        try:
            value = float(node.value)
            return value if math.isfinite(value) else None
        except OverflowError:
            return None
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = constant_number(node.operand)
        return None if value is None else (-value if isinstance(node.op, ast.USub) else value)
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
        left, right = constant_number(node.left), constant_number(node.right)
        if left is None or right is None:
            return None
        try:
            if isinstance(node.op, ast.Add): value = left + right
            elif isinstance(node.op, ast.Sub): value = left - right
            elif isinstance(node.op, ast.Mult): value = left * right
            elif isinstance(node.op, ast.Div): value = left / right
            else: value = left ** right
            return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else None
        except (ArithmeticError, ValueError):
            return None
    return None

"""Registry adapters for numeric time-series nodes; no duplicate mathematics."""
from __future__ import annotations

from cal_indicators.typed_operators import TYPED_OPERATOR_REGISTRY_VERSION, get_typed_operator_registry

# Only operators with causal numeric time-series output belong in this adapter.
# Boolean predicates remain available inside typed formulas, not as enum codes.
SERIES_OPERATOR_LABELS = {
    "add": "相加", "subtract": "相减", "multiply": "相乘", "divide": "相除",
    "minimum": "逐点较小值", "maximum": "逐点较大值", "negate": "取相反数",
    "absolute": "绝对值", "sqrt": "平方根", "log": "自然对数", "exp": "指数函数",
    "clip": "上下界裁剪", "cumulative_sum": "累计求和", "cumulative_product": "累计乘积",
    "cumulative_max": "历史最高值", "cumulative_min": "历史最低值",
    "drawdown_series": "回撤序列", "lag": "滞后", "difference": "差分",
    "rolling_mean": "滚动均值", "rolling_std": "滚动标准差",
    "rolling_min": "滚动最低值", "rolling_max": "滚动最高值",
    "recursive_smooth": "递归平滑",
}
PARAMETER_LABELS = {"window": "窗口观察数", "periods": "周期", "ddof": "自由度",
                    "min_periods": "最少观察数", "initial": "初始值", "lower": "下界", "upper": "上界"}


def series_operator_specs():
    registry = get_typed_operator_registry(TYPED_OPERATOR_REGISTRY_VERSION)
    return [registry[name] for name in SERIES_OPERATOR_LABELS]


def register_series_operators(registry, numeric_node, port):
    """Derive ports and fixed arguments from the same indicator signatures."""
    for spec in series_operator_specs():
        signature = max(spec.signatures, key=lambda item: len(item.inputs))
        names = spec.argument_names(len(signature.inputs))
        inputs, parameters = [], {}
        for name, value_type in zip(names, signature.inputs):
            if value_type.startswith("scalar") and "series" not in value_type:
                integer = "count" in value_type
                default = {"window": 20, "periods": 1, "ddof": 1, "min_periods": 1,
                           "initial": 0, "lower": -1, "upper": 1}.get(name, 0)
                schema = {"type": "integer" if integer else "number", "default": default,
                          "title": PARAMETER_LABELS.get(name, name)}
                if integer:
                    schema.update(minimum=0 if name == "ddof" else 1, maximum=5000)
                parameters[name] = schema
            else:
                inputs.append(port(name, "series<float64>"))
        identifier = f"indicator.{spec.operator_id}"
        node = numeric_node(identifier, SERIES_OPERATOR_LABELS[spec.operator_id], "indicator",
                            inputs, [port("value", "series<float64>")],
                            {"type": "object", "properties": parameters, "additionalProperties": False})
        node.update(typed_operator_id=spec.operator_id, typed_operator_version=spec.version,
                    typed_arguments=list(names), kernel_id="typed_formula_plan",
                    description=spec.description, kernel_version=spec.version)
        registry[identifier] = node


def is_typed_formula_node(node, registry):
    return node.type == "feature.formula" or bool(registry.get(node.type, {}).get("typed_operator_id"))


def typed_node_expression(node, registry):
    if node.type == "feature.formula":
        return str(node.parameters.get("expression") or "")
    metadata = registry[node.type]
    defaults = metadata["parameter_schema"]["properties"]
    arguments = [name if name in node.inputs else repr(node.parameters.get(name, defaults[name]["default"]))
                 for name in metadata["typed_arguments"]]
    return f"{metadata['typed_operator_id']}({', '.join(arguments)})"


def validate_typed_series_node(node, registry):
    from cal_indicators.typed_dsl import compose_typed_expression, ValueType
    expression = typed_node_expression(node, registry)
    return compose_typed_expression(expression, variable_types={name: ValueType.series("T") for name in node.inputs},
                                    output_contract="series", max_nodes=128, max_depth=20)

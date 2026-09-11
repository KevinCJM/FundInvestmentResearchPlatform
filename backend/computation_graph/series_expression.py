"""Translate port references into variables for the shared typed DSL compiler."""
from __future__ import annotations

import ast
from cal_indicators.typed_dsl import compose_typed_expression, ValueType
from .causal_series import causal_violations


def bind_series_expression(expression: ast.AST, symbols: dict[str, str], allowed_operators):
    bindings = {}

    class BindPorts(ast.NodeTransformer):
        def visit_Attribute(self, node):
            if not isinstance(node.value, ast.Name) or node.value.id not in symbols or node.attr.startswith("_"):
                raise ValueError("请使用已声明变量的输出端口。")
            reference = (symbols[node.value.id], node.attr)
            if reference not in bindings:
                if len(bindings) >= 4:
                    raise ValueError("单个公式最多引用四个输入，请拆成独立计算步骤。")
                bindings[reference] = f"feature_{len(bindings) + 1}"
            return ast.copy_location(ast.Name(id=bindings[reference], ctx=ast.Load()), node)

    transformed = BindPorts().visit(ast.fix_missing_locations(expression))
    source = ast.unparse(transformed)
    if not bindings:
        raise ValueError("时序公式至少需要一个输入序列。")
    plan = compose_typed_expression(source, variable_types={name: ValueType.series("T") for name in bindings.values()},
                                    output_contract="series", max_nodes=128, max_depth=20)
    if causal_violations(plan, allowed_operators):
        raise ValueError("公式包含未开放或非因果算子。")
    if plan.output_type.dtype != "float64":
        raise ValueError("当前数值公式需返回数值时序；类别由分类算子生成。")
    return source, {name: {"node_id": reference[0], "port": reference[1]} for reference, name in bindings.items()}

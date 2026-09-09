"""Lossless LaTeX editor source and canonical internal DSL.

Editable LaTeX retains every argument; compact display LaTeX is presentation
only. The typed compiler still validates every construct before execution.
"""
from __future__ import annotations

import ast
import math

from cal_indicators.typed_dsl import TypedDslError, TypedExpressionParser

from .variable_registry import normalize_variable_latex, variable_latex_symbols


class _CanonicalConstants(ast.NodeTransformer):
    def visit_Constant(self, node: ast.Constant) -> ast.AST:  # noqa: N802
        value = node.value
        # The builder submits numeric constants as JSON floats. Normalize 15.0
        # to 15 without changing precision or expanding very large literals.
        if (
            isinstance(value, float)
            and math.isfinite(value)
            and value.is_integer()
            and abs(value) <= 2**53
        ):
            return ast.copy_location(ast.Constant(value=int(value)), node)
        return node


def canonical_formula_source(expression: str) -> str:
    """Accept supported DSL/LaTeX and return stable internal DSL syntax."""
    normalized = normalize_variable_latex(expression)
    _, root = TypedExpressionParser(()).parse(normalized)
    root = _CanonicalConstants().visit(root)
    return ast.unparse(ast.fix_missing_locations(root))


class _EditableLatexRenderer(ast.NodeVisitor):
    """Render only the parser's reversible subset, never compact display aliases."""

    def __init__(self) -> None:
        self.symbols = variable_latex_symbols()

    def visit_Name(self, node: ast.Name) -> str:  # noqa: N802
        return self.symbols.get(node.id, node.id)

    def visit_Constant(self, node: ast.Constant) -> str:  # noqa: N802
        return repr(node.value)

    @staticmethod
    def _group(value: str) -> str:
        return rf"\left({value}\right)"

    def visit_UnaryOp(self, node: ast.UnaryOp) -> str:  # noqa: N802
        sign = "-" if isinstance(node.op, ast.USub) else "+"
        return sign + self._group(self.visit(node.operand))

    def visit_BinOp(self, node: ast.BinOp) -> str:  # noqa: N802
        left, right = self.visit(node.left), self.visit(node.right)
        if isinstance(node.op, ast.Div):
            return rf"\frac{{{left}}}{{{right}}}"
        if isinstance(node.op, ast.Pow):
            return rf"{self._group(left)}^{{{right}}}"
        symbol = {ast.Add: "+", ast.Sub: "-", ast.Mult: r"\cdot "}.get(type(node.op))
        if symbol is None:
            return self.generic_visit(node)
        # Preserve associativity instead of flattening operations: a-(b-c),
        # (a+b)*c and a/(b/c) must compile to the same graph after editing.
        return self._group(f"{left}{symbol}{right}")

    def visit_Attribute(self, node: ast.Attribute) -> str:  # noqa: N802
        # Named-port syntax is deliberately retained by the reversible source.
        return f"{self.visit(node.value)}.{node.attr}"

    def visit_Call(self, node: ast.Call) -> str:  # noqa: N802
        if not isinstance(node.func, ast.Name) or node.keywords:
            return self.generic_visit(node)
        arguments = [self.visit(argument) for argument in node.args]
        if node.func.id == "sqrt" and len(arguments) == 1:
            return rf"\sqrt{{{arguments[0]}}}"
        # Keep argument count and aliases exactly, including ddof/min_periods.
        # The preview renderer independently replaces these names with symbols.
        return rf"\operatorname{{{node.func.id}}}\left({','.join(arguments)}\right)"

    def generic_visit(self, node: ast.AST) -> str:
        raise TypedDslError("LATEX_SOURCE_UNSUPPORTED", "该公式无法转换为可编辑 LaTeX。")


def editable_formula_latex(expression: str) -> str:
    """Return reversible LaTeX for the editor without rewriting stored history."""
    canonical = canonical_formula_source(expression)
    root = ast.parse(canonical, mode="eval").body
    latex = _EditableLatexRenderer().visit(root)
    # A display-only abbreviation must never silently replace executable source.
    if canonical_formula_source(latex) != canonical:
        raise TypedDslError("LATEX_SOURCE_ROUNDTRIP_MISMATCH", "LaTeX 转换未保持原公式。")
    return latex

"""Canonical mathematical LaTeX for the typed indicator DSL.

The executable formula is deliberately kept as a restricted DSL/LaTeX source.
This module owns the separate presentation notation so UI clients never render
Python identifiers (and their underscores) as if they were already LaTeX.
"""

from __future__ import annotations

import ast
import math
from collections.abc import Mapping, Sequence


MATH_NOTATION_VERSION = "1.5.0"


_CANONICAL_OPERATOR_ALIASES = {
    "sub": "subtract",
    "mul": "multiply",
    "safe_divide": "divide",
    "elementwise_min": "minimum",
    "elementwise_max": "maximum",
    "abs": "absolute",
    "eq": "equal",
    "ne": "not_equal",
    "lt": "less_than",
    "le": "less_equal",
    "gt": "greater_than",
    "ge": "greater_equal",
    "sequence_sum": "sum",
    "prod": "product",
    "sequence_prod": "product",
    "sequence_mean": "mean",
    "min": "min_value",
    "max": "max_value",
    "var": "variance",
    "sequence_std": "std",
    "cumulative_maximum": "cumulative_max",
    "cumulative_minimum": "cumulative_min",
    "skew": "skewness",
    "kurtosis_excess": "excess_kurtosis",
    "mad": "mean_absolute_deviation",
    "rms": "root_mean_square",
    "masked_sum": "sum_where",
    "masked_mean": "mean_where",
    "masked_variance": "variance_where",
    "masked_std": "std_where",
    "masked_min": "min_where",
    "masked_max": "max_where",
    "masked_median": "median_where",
    "masked_quantile": "quantile_where",
    "masked_count": "count_true",
    "cov": "covariance",
    "corr": "correlation",
}


def escape_latex_text(value: str) -> str:
    """Escape an identifier that must be displayed as text in math mode."""

    replacements = {
        "\\": r"\backslash ",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
        "{": r"\{",
        "}": r"\}",
        "$": r"\$",
    }
    return "".join(replacements.get(character, character) for character in value)


def _group(value: str) -> str:
    return rf"\left({value}\right)"


def _absolute(value: str) -> str:
    return rf"\left\lvert {value}\right\rvert"


def _indexed(value: str, index: str) -> str:
    """Index a complete expression instead of its final TeX atom.

    Nested path operators commonly pass an already-indexed sequence into
    another reduction.  Appending ``_i`` directly to that expression produces
    invalid TeX such as ``(... )_t_i``.  Grouping the operand makes the index
    apply to the complete expression and remains valid at every nesting depth.
    """

    return rf"{_group(value)}_{{{index}}}"


def _argument(arguments: Sequence[str], index: int, fallback: str) -> str:
    return arguments[index] if index < len(arguments) else fallback


def render_operator_latex(operator_id: str, arguments: Sequence[str]) -> str:
    """Render one canonical operator using mathematical notation where possible."""

    operator_id = _CANONICAL_OPERATOR_ALIASES.get(operator_id, operator_id)
    first = _argument(arguments, 0, "x")
    second = _argument(arguments, 1, "y")
    third = _argument(arguments, 2, "z")
    fourth = _argument(arguments, 3, "m")

    binary = {
        "add": rf"{_group(first)}+{_group(second)}",
        "subtract": rf"{_group(first)}-{_group(second)}",
        "multiply": rf"{_group(first)}\cdot{_group(second)}",
        "divide": rf"\frac{{{first}}}{{{second}}}",
        "power": rf"{_group(first)}^{{{second}}}",
        "minimum": rf"\min\left({first},{second}\right)",
        "maximum": rf"\max\left({first},{second}\right)",
        "equal": rf"{first}={second}",
        "not_equal": rf"{first}\ne {second}",
        "less_than": rf"{first}<{second}",
        "less_equal": rf"{first}\le {second}",
        "greater_than": rf"{first}>{second}",
        "greater_equal": rf"{first}\ge {second}",
        "logical_and": rf"{_group(first)}\land{_group(second)}",
        "logical_or": rf"{_group(first)}\lor{_group(second)}",
        "dot": rf"{_group(first)}^{{\mathsf T}}{_group(second)}",
        "outer": rf"{_group(first)}{_group(second)}^{{\mathsf T}}",
        "matmul": rf"{_group(first)}{_group(second)}",
        "matvec": rf"{_group(first)}{_group(second)}",
        "solve": rf"{_group(first)}\backslash{_group(second)}",
        "quadratic_form": rf"{_group(first)}^{{\mathsf T}}{_group(second)}{_group(first)}",
        "active_returns": rf"{_group(first)}-{_group(second)}",
        "value_at": _indexed(first, second),
        "days_between": rf"{_group(second)}-{_group(first)}",
    }
    if operator_id in binary:
        return binary[operator_id]

    unary = {
        "negate": rf"-{_group(first)}",
        "absolute": _absolute(first),
        "sqrt": rf"\sqrt{{{first}}}",
        "log": rf"\ln{_group(first)}",
        "exp": rf"\mathrm{{e}}^{{{first}}}",
        "reciprocal": rf"{_group(first)}^{{-1}}",
        "sign": rf"\operatorname{{sgn}}{_group(first)}",
        "normal_pdf": rf"\phi{_group(first)}",
        "normal_ppf": rf"\Phi^{{-1}}{_group(first)}",
        "logical_not": rf"\neg{_group(first)}",
        "sum": rf"\sum_{{i=1}}^{{n}} {_indexed(first, 'i')}",
        "product": rf"\prod_{{i=1}}^{{n}} {_indexed(first, 'i')}",
        "mean": rf"\overline{{{first}}}",
        "min_value": rf"\min_{{1\le i\le n}} {_indexed(first, 'i')}",
        "max_value": rf"\max_{{1\le i\le n}} {_indexed(first, 'i')}",
        "cumulative_sum": rf"\left(\sum_{{i=1}}^{{t}} {_indexed(first, 'i')}\right)_{{t=1}}^{{T}}",
        "cumulative_product": rf"\left(\prod_{{i=1}}^{{t}} {_indexed(first, 'i')}\right)_{{t=1}}^{{T}}",
        "cumulative_return": rf"\left(\prod_{{i=1}}^{{t}}\left(1+{_indexed(first, 'i')}\right)-1\right)_{{t=1}}^{{T}}",
        "cumulative_max": rf"\left(\max_{{1\le i\le t}} {_indexed(first, 'i')}\right)_{{t=1}}^{{T}}",
        "cumulative_min": rf"\left(\min_{{1\le i\le t}} {_indexed(first, 'i')}\right)_{{t=1}}^{{T}}",
        "drawdown_series": rf"\mathcal{{D}}{_group(first)}",
        "last_drawdown_interval": rf"\mathcal{{I}}^{{*}}_{{\mathrm{{last}}}}{_group(first)}",
        "linear_fit": rf"\mathcal{{F}}_{{\mathrm{{OLS}}}}\left({','.join(arguments)}\right)",
        "fit_slope": rf"\widehat{{\beta}}{_group(first)}",
        "fit_intercept": rf"\widehat{{\alpha}}{_group(first)}",
        "fit_residual_sum_squares": rf"\mathrm{{SSE}}{_group(first)}",
        "fit_total_sum_squares": rf"\mathrm{{SST}}{_group(first)}",
        "fit_observation_count": rf"n{_group(first)}",
        "require_positive": rf"\underbrace{{{first}}}_{{>0}}",
        "require_nonnegative": rf"\underbrace{{{first}}}_{{\ge 0}}",
        "interval_start": rf"p{_group(first)}",
        "interval_trough": rf"\tau{_group(first)}",
        "interval_recovery": rf"\rho{_group(first)}",
        "new_high_mask": rf"\mathcal{{H}}_{{\mathrm{{new}}}}{_group(first)}",
        "first": _indexed(first, "1"),
        "last": _indexed(first, "T"),
        "length": rf"\left\lvert {first}\right\rvert",
        "median": rf"\widetilde{{{first}}}",
        "skewness": rf"\gamma_1{_group(first)}",
        "excess_kurtosis": rf"\gamma_2{_group(first)}",
        "mean_absolute_deviation": rf"\operatorname{{MAD}}{_group(first)}",
        "root_mean_square": rf"\operatorname{{RMS}}{_group(first)}",
        "argmin": rf"\operatorname*{{arg\,min}}_i {_indexed(first, 'i')}",
        "argmax": rf"\operatorname*{{arg\,max}}_i {_indexed(first, 'i')}",
        "linear_slope": rf"\widehat{{\beta}}_1{_group(first)}",
        "linear_intercept": rf"\widehat{{\beta}}_0{_group(first)}",
        "linear_r_squared": rf"R^2{_group(first)}",
        "regression_standard_error": rf"s_{{\varepsilon}}{_group(first)}",
        "transpose": rf"{_group(first)}^{{\mathsf T}}",
        "diag": rf"\operatorname{{diag}}{_group(first)}",
        "trace": rf"\operatorname{{tr}}{_group(first)}",
        "covariance": rf"\operatorname{{Cov}}\left({','.join(arguments)}\right)",
        "correlation": rf"\operatorname{{Corr}}\left({','.join(arguments)}\right)",
        "portfolio_returns": rf"{_group(first)}{_group(second)}",
        "total_return": rf"\prod_i\left(1+{first}_i\right)-1",
    }
    if operator_id in unary:
        return unary[operator_id]

    if operator_id == "clip":
        return rf"\min\left(\max\left({first},{second}\right),{third}\right)"
    if operator_id == "where":
        return rf"\begin{{cases}}{second},&{first}\\{third},&\neg{first}\end{{cases}}"
    if operator_id == "variance":
        ddof = arguments[1] if len(arguments) > 1 else ""
        suffix = rf"_{{\mathrm{{ddof}}={ddof}}}" if ddof else ""
        return rf"\operatorname{{Var}}{suffix}{_group(first)}"
    if operator_id == "std":
        ddof = arguments[1] if len(arguments) > 1 else ""
        suffix = rf"_{{\mathrm{{ddof}}={ddof}}}" if ddof else ""
        return rf"\operatorname{{Std}}{suffix}{_group(first)}"
    if operator_id == "lag":
        return rf"\left({first}_{{t-{second}}}\right)_t"
    if operator_id == "difference":
        return rf"\left(\Delta_{{{second}}}{first}_t\right)_t"
    if operator_id == "rolling_window":
        return rf"\mathcal{{W}}_{{t,{second}}}{_group(first)}"
    if operator_id == "finite_mask":
        return rf"\mathbf{{1}}_{{\operatorname{{finite}}\left({first}\right)}}"
    if operator_id == "rolling_apply":
        window = second if len(arguments) not in {3, 5} else f"{second};{arguments[-1]}"
        return rf"\mathcal{{R}}_{{{window}}}\!\left[{first}\right]_t"
    if operator_id == "rolling_mean":
        # Window availability rules belong in the calculation explanation. The
        # headline formula uses one compact symbol for the complete operator.
        return rf"\mu_{{t,{second}}}{_group(first)}"
    if operator_id == "rolling_std":
        window = second
        ddof = arguments[2] if len(arguments) > 2 else "0"
        if ddof in {"1", "1.0"}:
            return rf"s_{{t,{window}}}{_group(first)}"
        if ddof in {"0", "0.0"}:
            return rf"\sigma_{{t,{window}}}{_group(first)}"
        return rf"\sigma_{{t,{window}}}^{{({ddof})}}{_group(first)}"
    if operator_id == "rolling_min":
        window_set = rf"\mathcal{{W}}_{{t,{second}}}"
        return rf"\min_{{i\in {window_set}}} {_indexed(first, 'i')}"
    if operator_id == "rolling_max":
        window_set = rf"\mathcal{{W}}_{{t,{second}}}"
        return rf"\max_{{i\in {window_set}}} {_indexed(first, 'i')}"
    if operator_id == "recursive_smooth":
        return rf"\mathcal{{S}}_{{{second},{third}}}{_group(first)}_{{t}}"
    if operator_id == "divide_or_default":
        return rf"{_group(first)}\mathbin{{\oslash}}_{{{third}}}{_group(second)}"
    if operator_id == "quantile":
        return rf"Q_{{{second}}}{_group(first)}"
    if operator_id == "count_true":
        return rf"\sum_{{i=1}}^{{n}}\mathbf{{1}}\!\left[{_indexed(first, 'i')}\right]"
    if operator_id == "max_consecutive_true":
        return rf"\max\operatorname{{run}}{_group(first)}"

    conditional = {
        "sum_where": rf"\sum_{{i:{_indexed(second, 'i')}}} {_indexed(first, 'i')}",
        "mean_where": rf"\mathbb{{E}}\!\left[{first}\mid {second}\right]",
        "variance_where": rf"\operatorname{{Var}}\!\left({first}\mid {second}\right)",
        "std_where": rf"\operatorname{{Std}}\!\left({first}\mid {second}\right)",
        "min_where": rf"\min_{{i:{_indexed(second, 'i')}}} {_indexed(first, 'i')}",
        "max_where": rf"\max_{{i:{_indexed(second, 'i')}}} {_indexed(first, 'i')}",
        "median_where": rf"\operatorname{{Med}}\!\left({first}\mid {second}\right)",
        "quantile_where": rf"Q_{{{third}}}\!\left({first}\mid {second}\right)",
    }
    if operator_id in conditional:
        return conditional[operator_id]

    axis_match = None
    for suffix in ("sum", "mean", "product", "variance", "std", "min", "max"):
        for axis in ("time", "asset"):
            if operator_id == f"{suffix}_{axis}":
                axis_match = (suffix, "t" if axis == "time" else "j")
                break
    if axis_match:
        operation, axis = axis_match
        if operation == "sum":
            return rf"\sum_{{{axis}}}{first}"
        if operation == "product":
            return rf"\prod_{{{axis}}}{first}"
        if operation == "mean":
            return rf"\mathbb{{E}}_{{{axis}}}\!\left[{first}\right]"
        if operation == "variance":
            return rf"\operatorname{{Var}}_{{{axis}}}{_group(first)}"
        if operation == "std":
            return rf"\operatorname{{Std}}_{{{axis}}}{_group(first)}"
        return rf"\{operation}_{{{axis}}}{_group(first)}"

    if operator_id == "annualized_return":
        return rf"{_group(first)}^{{{second}}}-1"

    escaped = escape_latex_text(operator_id)
    return rf"\operatorname{{{escaped}}}\left({','.join(arguments)}\right)"


class _MathematicalLatexRenderer(ast.NodeVisitor):
    _PRECEDENCE = {
        ast.Add: 10,
        ast.Sub: 10,
        ast.Mult: 20,
        ast.Div: 20,
        ast.Pow: 30,
    }

    def __init__(self, variable_latex: Mapping[str, str]) -> None:
        self.variable_latex = variable_latex

    def render(self, node: ast.AST) -> str:
        return self.visit(node)

    @classmethod
    def _node_precedence(cls, node: ast.AST) -> int:
        if isinstance(node, ast.BinOp):
            return cls._PRECEDENCE.get(type(node.op), 0)
        if isinstance(node, ast.UnaryOp):
            return 40
        return 50

    def _render_operand(
        self,
        node: ast.AST,
        parent_precedence: int,
        *,
        group_on_equal: bool = False,
    ) -> str:
        rendered = self.render(node)
        precedence = self._node_precedence(node)
        if precedence < parent_precedence or (
            group_on_equal and precedence == parent_precedence
        ):
            return _group(rendered)
        return rendered

    def visit_Constant(self, node: ast.Constant) -> str:  # noqa: N802
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            return rf"\mathrm{{{escape_latex_text(str(node.value))}}}"
        number = float(node.value)
        if math.isfinite(number) and number.is_integer():
            return str(int(number))
        return format(number, ".12g")

    def visit_Name(self, node: ast.Name) -> str:  # noqa: N802
        return self.variable_latex.get(
            node.id,
            rf"\mathrm{{{escape_latex_text(node.id)}}}",
        )

    def visit_UnaryOp(self, node: ast.UnaryOp) -> str:  # noqa: N802
        if isinstance(node.op, ast.USub):
            operand = self._render_operand(node.operand, 40)
            return rf"-{operand}"
        if isinstance(node.op, ast.UAdd):
            return self._render_operand(node.operand, 40)
        return self.generic_visit(node)  # type: ignore[return-value]

    def visit_BinOp(self, node: ast.BinOp) -> str:  # noqa: N802
        precedence = self._PRECEDENCE.get(type(node.op))
        if precedence is None:
            return rf"\mathrm{{{escape_latex_text(ast.unparse(node))}}}"
        if isinstance(node.op, ast.Add):
            return (
                f"{self._render_operand(node.left, precedence)}+"
                f"{self._render_operand(node.right, precedence)}"
            )
        if isinstance(node.op, ast.Sub):
            return (
                f"{self._render_operand(node.left, precedence)}-"
                f"{self._render_operand(node.right, precedence, group_on_equal=True)}"
            )
        if isinstance(node.op, ast.Mult):
            return (
                f"{self._render_operand(node.left, precedence)}"
                rf"\cdot{self._render_operand(node.right, precedence)}"
            )
        if isinstance(node.op, ast.Div):
            return rf"\frac{{{self.render(node.left)}}}{{{self.render(node.right)}}}"
        return (
            rf"{self._render_operand(node.left, precedence, group_on_equal=True)}"
            rf"^{{{self.render(node.right)}}}"
        )

    def visit_Attribute(self, node: ast.Attribute) -> str:  # noqa: N802
        return rf"\left({self.visit(node.value)}\right)_{{\mathrm{{{escape_latex_text(node.attr)}}}}}"

    def visit_Call(self, node: ast.Call) -> str:  # noqa: N802
        if not isinstance(node.func, ast.Name):
            return rf"\mathrm{{{escape_latex_text(ast.unparse(node))}}}"
        operator_id = _CANONICAL_OPERATOR_ALIASES.get(node.func.id, node.func.id)
        if (
            operator_id in {"mean", "std", "variance", "min_value", "max_value"}
            and node.args
            and isinstance(node.args[0], ast.Call)
            and isinstance(node.args[0].func, ast.Name)
            and node.args[0].func.id == "rolling_window"
            and len(node.args[0].args) >= 2
        ):
            window_call = node.args[0]
            values = self.render(window_call.args[0])
            window = self.render(window_call.args[1])
            window_set = rf"\mathcal{{W}}_{{t,{window}}}"
            if operator_id == "mean":
                return rf"\mu_{{t,{window}}}{_group(values)}"
            if operator_id in {"std", "variance"}:
                ddof = self.render(node.args[1]) if len(node.args) > 1 else "1"
                if operator_id == "variance":
                    if ddof in {"1", "1.0"}:
                        return rf"s^2_{{t,{window}}}{_group(values)}"
                    if ddof in {"0", "0.0"}:
                        return rf"\sigma^2_{{t,{window}}}{_group(values)}"
                    return rf"\sigma^{{2,({ddof})}}_{{t,{window}}}{_group(values)}"
                if ddof in {"1", "1.0"}:
                    return rf"s_{{t,{window}}}{_group(values)}"
                if ddof in {"0", "0.0"}:
                    return rf"\sigma_{{t,{window}}}{_group(values)}"
                return rf"\sigma_{{t,{window}}}^{{({ddof})}}{_group(values)}"
            operation = "min" if operator_id == "min_value" else "max"
            return rf"\{operation}_{{i\in {window_set}}} {_indexed(values, 'i')}"
        return render_operator_latex(
            operator_id,
            tuple(self.render(argument) for argument in node.args),
        )

    def generic_visit(self, node: ast.AST) -> str:
        return rf"\mathrm{{{escape_latex_text(ast.unparse(node))}}}"


def render_python_expression_latex(
    python_expression: str,
    variable_latex: Mapping[str, str] | None = None,
) -> str:
    """Render a compiler-normalized Python expression as mathematical LaTeX."""

    parsed = ast.parse(python_expression, mode="eval")
    return _MathematicalLatexRenderer(variable_latex or {}).render(parsed.body)


def operator_display_latex_template(
    operator_id: str,
    parameter_names: Sequence[str],
) -> str:
    placeholders = tuple(
        rf"\mathrm{{{escape_latex_text(name)}}}" for name in parameter_names
    )
    return render_operator_latex(operator_id, placeholders)


__all__ = [
    "MATH_NOTATION_VERSION",
    "escape_latex_text",
    "operator_display_latex_template",
    "render_operator_latex",
    "render_python_expression_latex",
]

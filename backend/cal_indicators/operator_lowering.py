"""Compiler-owned expansion of supported historical formula spellings.

These names are input contracts, not a second numeric execution path. The DAG
builder validates the original arity/types before expanding trusted syntax.
"""
from __future__ import annotations

LOWERING_VERSION = "primitive-operators-3-interval-rolling-1"
ROLLING_COMPAT_OPERATOR_IDS = frozenset(
    {"rolling_mean", "rolling_std", "rolling_min", "rolling_max"}
)
COMPILER_FUSED_OPERATOR_IDS = frozenset({"rolling_window", "rolling_apply"})
COMPOSITE_OPERATOR_IDS = frozenset({
    "total_return", "annualized_return", "cumulative_return", "active_returns",
    "portfolio_returns", "quadratic_form", "linear_slope", "linear_intercept",
    "linear_r_squared", "regression_standard_error",
})


COMPOSITE_DEPENDENCIES = {
    "total_return": ("product", "add", "subtract"),
    "cumulative_return": ("cumulative_product", "add", "subtract"),
    "annualized_return": ("product", "add", "subtract", "power", "divide", "length", "require_positive", "require_nonnegative"),
    "active_returns": ("subtract",),
    "portfolio_returns": ("matvec",),
    "quadratic_form": ("transpose", "matvec", "dot"),
    "linear_slope": ("linear_fit", "fit_slope"),
    "linear_intercept": ("linear_fit", "fit_intercept"),
    "linear_r_squared": ("linear_fit", "fit_residual_sum_squares", "fit_total_sum_squares", "require_positive", "divide", "subtract"),
    "regression_standard_error": ("linear_fit", "fit_residual_sum_squares", "fit_observation_count", "require_positive", "divide", "subtract", "sqrt"),
}


def expand_operator(
    operator_id: str,
    arguments: tuple[str, ...],
    *,
    operator_registry_version: str | None = None,
) -> str | None:
    # 2.3 keeps the original rolling spellings as an immutable historical
    # contract. Current authoring accepts them only as syntax compatibility and
    # lowers them to one logical window node plus an ordinary reduction.
    if operator_id in ROLLING_COMPAT_OPERATOR_IDS:
        if operator_registry_version != "2.4.0":
            return None
        args = tuple(f"({argument})" for argument in arguments)
        first = args[0]
        if operator_id == "rolling_std":
            window = f"rolling_window({first}, {args[1]})" if len(args) < 4 else f"rolling_window({first}, {args[1]}, {args[3]})"
            ddof = args[2] if len(args) >= 3 else "0"
            return f"std({window}, {ddof})"
        window = f"rolling_window({first}, {args[1]})" if len(args) < 3 else f"rolling_window({first}, {args[1]}, {args[2]})"
        reducer = {
            "rolling_mean": "mean",
            "rolling_min": "min_value",
            "rolling_max": "max_value",
        }[operator_id]
        return f"{reducer}({window})"
    if operator_id not in COMPOSITE_OPERATOR_IDS:
        return None
    args = tuple(f"({argument})" for argument in arguments)
    first = args[0]
    if operator_id == "total_return":
        return f"product({first} + 1) - 1"
    if operator_id == "cumulative_return":
        return f"cumulative_product({first} + 1) - 1"
    if operator_id == "annualized_return":
        # Preserve the saved formula's growth and domain conventions, including
        # the original (total_return + 1) floating-point evaluation order.
        return f"require_nonnegative((product({first} + 1) - 1) + 1) ** (require_positive({args[1]}) / length({first})) - 1"
    if operator_id == "active_returns":
        return f"{first} - {args[1]}"
    if operator_id == "portfolio_returns":
        return f"matvec({first}, {args[1]})"
    if operator_id == "quadratic_form":
        return f"dot(matvec(transpose({args[1]}), {first}), {first})"
    fit = f"linear_fit({', '.join(args)})"
    if operator_id == "linear_slope":
        return f"fit_slope({fit})"
    if operator_id == "linear_intercept":
        return f"fit_intercept({fit})"
    if operator_id == "linear_r_squared":
        return f"1 - fit_residual_sum_squares({fit}) / require_positive(fit_total_sum_squares({fit}))"
    return f"sqrt(fit_residual_sum_squares({fit}) / require_positive(fit_observation_count({fit}) - 2))"

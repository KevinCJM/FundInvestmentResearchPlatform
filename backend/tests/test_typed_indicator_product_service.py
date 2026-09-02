from __future__ import annotations

import pytest

from cal_indicators.typed_latex import render_python_expression_latex
from cal_indicators.typed_operators import get_typed_operator_registry
from custom_indicators.errors import ValidationError
from custom_indicators.typed_service import (
    _operator_expression,
    _operator_parameter_names,
    compose_expression,
    infer_expression,
    typed_product_meta,
)
from custom_indicators.variable_registry import normalize_variable_latex


def test_product_catalog_exposes_math_operators_and_hides_deprecated_templates() -> None:
    meta = typed_product_meta()
    operator_ids = {item["name"] for item in meta["operators"]}
    assert {"add", "subtract", "multiply", "divide", "product", "dot", "matmul", "solve"} <= operator_ids
    assert "cumulative_return" not in operator_ids
    assert "total_return" not in operator_ids
    assert "portfolio_returns" not in operator_ids
    assert meta["predefined_calculations"] == []
    assert meta["predefined_calculations_deprecated"] is True
    variables = {item["name"]: item for item in meta["variables"]}
    assert variables["asset_returns"]["shape"] == "matrix"
    assert variables["asset_weights"]["shape"] == "vector"
    assert variables["weight_path"]["symbolic_shape"] == ["T", "N"]
    assert variables["adjusted_nav"]["latex"] == r"\mathbf{p}_{\mathrm{adj}}"
    assert variables["previous_close"]["latex"] == r"\mathbf{c}_{\mathrm{prev}}"
    assert all(
        item["name"] not in item["display_latex_template"]
        for item in meta["operators"]
        if "_" in item["name"]
    )


def test_mathematical_latex_is_separate_from_executable_expression() -> None:
    expression = (
        "mean_where(returns, greater_than(returns, 0)) / "
        "absolute(mean_where(returns, less_than(returns, 0)))"
    )

    inferred = infer_expression(expression, "single_product")

    assert inferred["latex"] == expression
    assert inferred["expression"] == expression
    assert "mean_where" not in inferred["display_latex"]
    assert "greater_than" not in inferred["display_latex"]
    assert "less_than" not in inferred["display_latex"]
    assert r"\mathbb{E}" in inferred["display_latex"]
    assert r"\mathbf{r}" in inferred["display_latex"]
    assert all(
        node.get("latex_fragment")
        for node in inferred["dag"]["nodes"]
    )


def test_operator_compose_uses_catalog_parameter_names_and_accepts_legacy_aliases() -> None:
    operators = typed_product_meta()["operators"]
    add = next(item for item in operators if item["name"] == "add")
    assert [parameter["name"] for parameter in add["parameters"]] == ["lhs", "rhs"]

    canonical = compose_expression(
        {
            "operator_id": "add",
            "context": "single_product",
            "arguments": [
                {"parameter": "lhs", "source": "variable", "value": "returns"},
                {"parameter": "rhs", "source": "variable", "value": "returns"},
            ],
        }
    )
    legacy = compose_expression(
        {
            "operator_id": "add",
            "context": "single_product",
            "arguments": [
                {"parameter": "A", "source": "variable", "value": "returns"},
                {"parameter": "B", "source": "variable", "value": "returns"},
            ],
        }
    )

    assert canonical["shape"] == "series"
    assert legacy["python_expression"] == canonical["python_expression"]

    for operator in operators:
        parameter_names = [item["name"] for item in operator["parameters"]]
        arguments = {name: "1" for name in parameter_names}
        assert _operator_expression(
            operator["name"], arguments, operator["version"]
        )

        legacy_names = _operator_parameter_names(
            operator["name"], len(parameter_names)
        )
        legacy_arguments = {name: "1" for name in legacy_names}
        assert _operator_expression(
            operator["name"], legacy_arguments, operator["version"]
        )

    for version in ("2.0.0", "2.1.0", "2.2.0"):
        for operator_id, spec in get_typed_operator_registry(version).items():
            for arity in spec.arities:
                name_variants = (
                    spec.argument_names(arity),
                    _operator_parameter_names(operator_id, arity),
                    tuple(f"input_{index + 1}" for index in range(arity)),
                )
                for names in name_variants:
                    assert _operator_expression(
                        operator_id,
                        {name: "1" for name in names},
                        version,
                    )


def test_operator_catalog_exposes_real_defaults_and_human_parameter_names() -> None:
    operators = {item["name"]: item for item in typed_product_meta()["operators"]}

    assert operators["normal_pdf"]["parameters"][0]["name"] == "values"
    assert operators["normal_ppf"]["parameters"][0]["name"] == "probability"
    assert operators["std"]["parameters"][1]["name"] == "ddof"
    assert operators["std"]["parameters"][1]["default"] == 1
    assert operators["std"]["parameters"][1]["optional"] is True
    assert operators["lag"]["parameters"][1]["default"] == 1
    assert operators["lag"]["parameters"][1]["optional"] is True

    for operator in operators.values():
        for parameter_set in operator["parameter_sets"]:
            for parameter in parameter_set["parameters"]:
                if parameter["default"] is not None:
                    assert parameter["optional"] is True

    raw_labels = []
    for operator in operators.values():
        for parameter_set in operator["parameter_sets"]:
            raw_labels.extend(
                parameter["name"]
                for parameter in parameter_set["parameters"]
                if parameter["label"] == parameter["name"]
            )
    assert raw_labels == []

    generic_legacy = compose_expression(
        {
            "operator_id": "normal_pdf",
            "context": "single_product",
            "arguments": [
                {"parameter": "input_1", "source": "constant", "value": 0}
            ],
        }
    )
    assert generic_legacy["shape"] == "scalar"


def test_legacy_text_subscripts_remain_parseable() -> None:
    assert normalize_variable_latex(r"\mathbf{p}_{adj}") == "adjusted_nav"
    assert normalize_variable_latex(r"r_{f}^{annual}+p_{year}") == (
        "annual_risk_free_rate_decimal+periods_per_year"
    )


def test_textual_fallback_identifiers_escape_underscores() -> None:
    latex = render_python_expression_latex(
        "custom_function(custom_variable)",
    )

    assert r"\operatorname{custom\_function}" in latex
    assert r"\mathrm{custom\_variable}" in latex
    assert "custom_function" not in latex
    assert "custom_variable" not in latex


def test_nested_path_notation_groups_indices_and_separates_control_words() -> None:
    symbols = {"returns": r"\mathbf{r}"}
    path_latex = render_python_expression_latex(
        "cumulative_max(cumulative_product(returns + 1))",
        symbols,
    )
    cvar_latex = render_python_expression_latex(
        "mean_where(returns, less_equal(returns, quantile(returns, 0.05)))",
        symbols,
    )

    assert "_t_i" not in path_latex
    assert path_latex.count(r"_{t=1}^{T}") == 2
    assert r"\le Q_{0.05}" in cvar_latex


def test_template_compose_enforces_semantic_roles_and_returns_canonical_latex() -> None:
    composed = compose_expression(
        {
            "template_id": "cumulative-return",
            "context": "single_product",
            "arguments": [{"parameter": "values", "source": "variable", "value": "returns"}],
        }
    )
    assert composed["shape"] == "scalar"
    assert "\\prod" in composed["latex"]
    assert "product" not in composed["display_latex"]
    assert r"\prod" in composed["display_latex"]
    assert composed["template_origin"]["template_id"] == "cumulative-return"

    with pytest.raises(ValidationError) as error:
        compose_expression(
            {
                "template_id": "cumulative-return",
                "context": "single_product",
                "arguments": [{"parameter": "values", "source": "variable", "value": "log_returns"}],
            }
        )
    assert error.value.code == "SEMANTIC_ROLE_MISMATCH"


def test_axis_mismatch_and_portfolio_volatility_inference() -> None:
    with pytest.raises(ValidationError) as error:
        infer_expression(r"\mathbf{R}+\mathbf{w}", "portfolio")
    assert error.value.code in {"AXIS_MISMATCH", "SHAPE_MISMATCH"}

    result = compose_expression(
        {
            "template_id": "portfolio-volatility",
            "context": "portfolio",
            "arguments": [
                {"parameter": "returns_matrix", "source": "variable", "value": "asset_returns"},
                {"parameter": "weights", "source": "variable", "value": "asset_weights"},
            ],
        }
    )
    assert result["shape"] == "scalar"
    assert result["dag"]["nodes"]
    assert all("inferred_type" in node for node in result["dag"]["nodes"])


def test_infer_rejects_context_variable_from_other_domain() -> None:
    with pytest.raises(ValidationError) as error:
        infer_expression(r"\operatorname{mean}\left(\mathbf{R}\right)", "single_product")
    assert error.value.code == "CONTEXT_VARIABLE_UNAVAILABLE"

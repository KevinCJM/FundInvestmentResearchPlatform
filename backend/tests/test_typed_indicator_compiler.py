from __future__ import annotations

import pytest

from cal_indicators.typed_dsl import (
    TypedDslError,
    TypedExpressionParser,
    ValueType,
    compose_typed_expression,
    compose_typed_series_bundle,
    infer_typed_expression,
)


def test_parser_supports_multi_asset_latex_variables_and_explicit_functions() -> None:
    expression = (
        r"\operatorname{mean}("
        r"\operatorname{portfolio_returns}(\mathbf{R},\mathbf{w}))"
    )
    plan = compose_typed_expression(expression)

    assert (
        plan.python_expression == "mean(matvec(asset_returns, asset_weights))"
    )
    assert plan.output_type == ValueType.scalar()


def test_parser_supports_fraction_sqrt_and_risk_free_variables() -> None:
    parser = TypedExpressionParser(
        (
            "returns",
            "annual_risk_free_rate_decimal",
            "periods_per_year",
        )
    )

    assert (
        parser.to_python(r"\frac{\sqrt{variance(\mathbf{r})}}{r_{f}^{annual}+p_{year}}")
        == "(sqrt(variance(returns)))/(annual_risk_free_rate_decimal+periods_per_year)"
    )


def test_std_and_variance_accept_optional_scalar_ddof() -> None:
    default_std = compose_typed_expression("std(returns)")
    explicit_std = compose_typed_expression(r"\operatorname{std}(\mathbf{r},1)")
    explicit_variance = compose_typed_expression("variance(returns, 0)")

    assert default_std.output_type == ValueType.scalar()
    assert explicit_std.python_expression == "std(returns,1)"
    assert explicit_variance.output_type == ValueType.scalar()

    with pytest.raises(TypedDslError) as caught:
        infer_typed_expression("std(returns, asset_weights)")
    assert caught.value.code == "TYPE_MISMATCH"


@pytest.mark.parametrize(
    ("expression", "code"),
    (
        ("unknown_series + 1", "UNKNOWN_VARIABLE"),
        ("unknown_function(returns)", "UNKNOWN_OPERATOR"),
        ("returns.__class__", "ILLEGAL_AST"),
        ("returns[0]", "ILLEGAL_AST"),
        ("mean(values=returns)", "ILLEGAL_AST"),
        ("(lambda value: value)(returns)", "ILLEGAL_AST"),
        ("[value for value in returns]", "ILLEGAL_AST"),
        ("'not numeric'", "INVALID_LITERAL"),
    ),
)
def test_restricted_ast_rejects_unsafe_or_unknown_syntax(
    expression: str, code: str
) -> None:
    with pytest.raises(TypedDslError) as caught:
        infer_typed_expression(expression)

    assert caught.value.code == code


def test_function_arity_is_checked_against_registry() -> None:
    with pytest.raises(TypedDslError) as caught:
        infer_typed_expression("mean(returns, 1)")

    assert caught.value.code == "ARITY_MISMATCH"


def test_nominal_axes_prevent_time_asset_accidental_broadcast() -> None:
    with pytest.raises(TypedDslError) as caught:
        infer_typed_expression("returns + asset_weights")

    assert caught.value.code == "AXIS_MISMATCH"


def test_scalar_broadcast_preserves_tensor_axes_and_shape() -> None:
    plan = infer_typed_expression("asset_returns * 2 + 1")

    assert plan.output_type == ValueType.matrix(("time", "asset"), ("T", "N"))


def test_linear_algebra_inference_tracks_named_axes() -> None:
    covariance = infer_typed_expression("covariance(asset_returns)")
    gram = infer_typed_expression("matmul(transpose(asset_returns), asset_returns)")
    portfolio = infer_typed_expression("matvec(asset_returns, asset_weights)")

    expected_covariance = ValueType.matrix(("asset", "asset"), ("N", "N"))
    assert covariance.output_type == expected_covariance
    assert gram.output_type == expected_covariance
    assert portfolio.output_type == ValueType.series()


@pytest.mark.parametrize(
    "operator",
    ("product_time", "variance_time", "min_time", "max_time"),
)
def test_explicit_time_reducers_preserve_asset_axis(operator: str) -> None:
    plan = infer_typed_expression(f"{operator}(asset_returns)")
    assert plan.output_type == ValueType.vector()


@pytest.mark.parametrize(
    "operator",
    ("product_asset", "variance_asset", "min_asset", "max_asset"),
)
def test_explicit_asset_reducers_preserve_time_axis(operator: str) -> None:
    plan = infer_typed_expression(f"{operator}(asset_returns)")
    assert plan.output_type == ValueType.series()


def test_weight_path_latex_variable_is_available_to_typed_parser() -> None:
    plan = compose_typed_expression(r"mean(\mathbf{W})")
    assert plan.python_expression == "mean(weight_path)"
    assert plan.context_requirements["weight_path"] == ValueType.matrix()


def test_matvec_rejects_non_matching_named_axis() -> None:
    with pytest.raises(TypedDslError) as caught:
        infer_typed_expression("matvec(asset_returns, returns)")

    assert caught.value.code == "AXIS_MISMATCH"


@pytest.mark.parametrize(
    ("expression", "code"),
    (
        ("clip(returns, returns, 1)", "TYPE_MISMATCH"),
        ("mean(1)", "TYPE_MISMATCH"),
        ("cumulative_sum(asset_returns)", "TYPE_MISMATCH"),
        ("last(asset_returns)", "TYPE_MISMATCH"),
        ("mean_time(transpose(asset_returns))", "TYPE_MISMATCH"),
        ("mean_asset(transpose(asset_returns))", "TYPE_MISMATCH"),
        ("transpose(returns)", "TYPE_MISMATCH"),
        ("dot(asset_returns, asset_returns)", "TYPE_MISMATCH"),
        ("outer(returns, returns)", "TYPE_MISMATCH"),
        ("matmul(asset_returns, asset_returns)", "AXIS_MISMATCH"),
        ("diag(returns)", "TYPE_MISMATCH"),
        ("trace(asset_returns)", "TYPE_MISMATCH"),
        ("solve(asset_returns, asset_weights)", "TYPE_MISMATCH"),
        ("covariance(returns)", "TYPE_MISMATCH"),
        ("portfolio_returns(transpose(asset_returns), asset_weights)", "TYPE_MISMATCH"),
        ("quadratic_form(returns, covariance(asset_returns))", "TYPE_MISMATCH"),
        ("active_returns(returns, asset_weights)", "TYPE_MISMATCH"),
        ("annualized_return(returns, asset_weights)", "TYPE_MISMATCH"),
    ),
)
def test_operator_type_errors_are_decided_during_inference(
    expression: str,
    code: str,
) -> None:
    with pytest.raises(TypedDslError) as caught:
        infer_typed_expression(expression)
    assert caught.value.code == code


def test_formula_limits_cover_depth_and_unique_nodes() -> None:
    with pytest.raises(TypedDslError) as depth_error:
        infer_typed_expression("((((returns + 1) + 2) + 3) + 4)", max_depth=3)
    assert depth_error.value.code == "FORMULA_TOO_COMPLEX"
    assert depth_error.value.details["dimension"] == "depth"

    with pytest.raises(TypedDslError) as node_error:
        infer_typed_expression("((returns + 1) + 2)", max_nodes=3)
    assert node_error.value.code == "FORMULA_TOO_COMPLEX"
    assert node_error.value.details["dimension"] == "nodes"


def test_rolling_window_is_a_non_publishable_logical_intermediate() -> None:
    plan = infer_typed_expression("rolling_window(returns, 5)")
    assert plan.output_type == ValueType.window(
        semantic_dimension="return_decimal"
    )

    with pytest.raises(TypedDslError) as output_error:
        compose_typed_series_bundle({"value": "rolling_window(returns, 5)"})
    assert output_error.value.code == "OUTPUT_CONTRACT_MISMATCH"

    with pytest.raises(TypedDslError) as arithmetic_error:
        compose_typed_series_bundle(
            {"value": "returns + rolling_window(returns, 5)"}
        )
    assert arithmetic_error.value.code == "TYPE_MISMATCH"


def test_current_rolling_wrapper_spelling_lowers_to_window_plus_reducer_only() -> None:
    current = compose_typed_series_bundle(
        {"value": "rolling_std(returns, 5, 1)"},
        dsl_version="2.4.0",
        operator_registry_version="2.4.0",
    )
    current_operators = [node.operator_id for node in current.nodes if node.operator_id]
    assert dict(current.python_expressions)["value"] == "std(rolling_window(returns, 5), 1)"
    assert "rolling_window" in current_operators
    assert "std" in current_operators
    assert "rolling_std" not in current_operators

    historical = compose_typed_series_bundle(
        {"value": "rolling_std(returns, 5, 1)"},
        dsl_version="2.3.0",
        operator_registry_version="2.3.0",
    )
    historical_operators = [
        node.operator_id for node in historical.nodes if node.operator_id
    ]
    assert dict(historical.python_expressions)["value"] == "rolling_std(returns, 5, 1)"
    assert "rolling_std" in historical_operators
    assert "rolling_window" not in historical_operators


def test_shared_subexpression_is_reused_in_dag() -> None:
    plan = compose_typed_expression("mean(returns) + mean(returns)")
    mean_nodes = [node for node in plan.nodes if node.operator_id == "mean"]

    assert len(mean_nodes) == 1
    root = plan.nodes[plan.root_id]
    assert root.inputs == (mean_nodes[0].node_id, mean_nodes[0].node_id)


@pytest.mark.parametrize(
    ("expression", "code"),
    (
        ("returns + volume", "SEMANTIC_DIMENSION_MISMATCH"),
        ("adjusted_nav + market_close", "SEMANTIC_DIMENSION_MISMATCH"),
        ("unit_nav + accumulated_nav", "PRICE_BASIS_MISMATCH"),
    ),
)
def test_semantic_dimensions_and_price_basis_prevent_invalid_mixing(
    expression: str,
    code: str,
) -> None:
    with pytest.raises(TypedDslError) as caught:
        infer_typed_expression(expression)
    assert caught.value.code == code
    diagnostic = caught.value.to_dict()
    assert diagnostic["node_id"] >= 0
    assert diagnostic["details"]["expected"]
    assert diagnostic["details"]["actual"]


def test_custom_variable_type_mapping_preserves_semantic_contract() -> None:
    with pytest.raises(TypedDslError) as caught:
        infer_typed_expression(
            "lhs + rhs",
            variable_types={
                "lhs": ValueType.series(
                    semantic_dimension="raw_market_price", price_basis="raw"
                ),
                "rhs": ValueType.series(
                    semantic_dimension="raw_market_price", price_basis="adjusted"
                ),
            },
        )
    assert caught.value.code == "PRICE_BASIS_MISMATCH"


def test_path_operators_infer_level_series_and_strict_new_high_mask() -> None:
    drawdown = infer_typed_expression("drawdown_series(adjusted_nav)")
    new_highs = infer_typed_expression("new_high_mask(adjusted_nav)")

    assert drawdown.output_type == ValueType.series(
        "L",
        semantic_dimension="return_decimal"
    )
    assert new_highs.output_type.is_mask
    assert new_highs.output_type.kind == "series"

    with pytest.raises(TypedDslError) as caught:
        infer_typed_expression("drawdown_series(returns)")
    assert caught.value.code == "SEMANTIC_DIMENSION_MISMATCH"


def test_comparisons_masks_where_and_masked_reducers_infer_types() -> None:
    predicate = infer_typed_expression("greater_than(returns, 0)")
    selected = infer_typed_expression("where(greater_than(returns, 0), returns, 0)")
    reduced = compose_typed_expression("mean_where(returns, greater_than(returns, 0))")

    assert predicate.output_type.is_mask
    assert predicate.output_type.shape == ("T",)
    assert selected.output_type == ValueType.series()
    assert selected.output_type.semantic_dimension == "return_decimal"
    assert reduced.output_type.is_numeric and reduced.output_type.is_scalar

    with pytest.raises(TypedDslError) as caught:
        compose_typed_expression("greater_than(returns, 0)")
    assert caught.value.code == "OUTPUT_CONTRACT_MISMATCH"
    assert "有限标量" in caught.value.message
    assert "时间序列布尔掩码" in caught.value.message
    assert "scalar" not in caught.value.message


@pytest.mark.parametrize(
    ("expression", "kind", "shape", "semantic_dimension"),
    (
        ("lag(returns)", "series", ("T-1",), "return_decimal"),
        ("difference(returns, 2)", "series", ("T-n",), "return_decimal"),
        ("cumulative_max(returns)", "series", ("T",), "return_decimal"),
        ("count_true(greater_than(returns, 0))", "scalar", (), "count"),
        ("linear_r_squared(returns)", "scalar", (), "dimensionless"),
        ("normal_pdf(0)", "scalar", (), "dimensionless"),
    ),
)
def test_p0_operator_inference_exposes_type_and_semantics(
    expression: str,
    kind: str,
    shape: tuple[str, ...],
    semantic_dimension: str,
) -> None:
    plan = infer_typed_expression(expression)
    assert plan.output_type.kind == kind
    assert plan.output_type.shape == shape
    assert plan.output_type.semantic_dimension == semantic_dimension


@pytest.mark.parametrize(
    "expression",
    [
        "quantile(returns, 0)",
        "quantile(returns, 1)",
        "lag(returns, 1.5)",
        "difference(returns, 0)",
        "std(returns, periods_per_year)",
    ],
)
def test_v2_1_control_parameters_must_be_valid_constants(expression: str) -> None:
    with pytest.raises(TypedDslError) as caught:
        compose_typed_expression(expression)

    assert caught.value.code == "INVALID_PARAMETER"


def test_legacy_20_compilation_keeps_old_operator_surface() -> None:
    plan = compose_typed_expression(
        "mean(returns)",
        dsl_version="2.0.0",
    )
    assert plan.dsl_version == "2.0.0"
    assert plan.compiler_version == "typed-ast-1"
    assert plan.operator_registry_version == "2.0.0"

    with pytest.raises(TypedDslError) as caught:
        infer_typed_expression(
            "count_true(greater_than(returns, 0))",
            dsl_version="2.0.0",
        )
    assert caught.value.code == "UNKNOWN_OPERATOR"


@pytest.mark.parametrize(
    ("dsl_version", "registry_version"),
    [("2.0.0", "2.1.0"), ("2.1.0", "2.0.0")],
)
def test_dsl_and_operator_registry_versions_cannot_be_mixed(
    dsl_version: str,
    registry_version: str,
) -> None:
    with pytest.raises(TypedDslError) as caught:
        compose_typed_expression(
            "mean(returns)",
            dsl_version=dsl_version,
            operator_registry_version=registry_version,
        )

    assert caught.value.code == "OPERATOR_VERSION_MISMATCH"

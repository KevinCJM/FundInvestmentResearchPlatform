from __future__ import annotations

import pytest

from cal_indicators.typed_dsl import (
    DEFAULT_VARIABLE_TYPES,
    TypedDslError,
    ValueType,
    compose_typed_expression,
    get_typed_dsl_catalog,
    get_typed_operator_catalog,
    get_typed_variable_catalog,
    infer_typed_expression,
)


def test_catalog_is_explicit_versioned_and_contains_core_categories() -> None:
    catalog = get_typed_dsl_catalog()

    assert catalog["dsl_version"] == "2.2.0"
    assert catalog["compiler_version"] == "typed-numba-3"
    assert catalog["operator_registry_version"] == "2.2.0"
    operators = {item["id"]: item for item in catalog["operators"]}
    assert {
        "add",
        "mean",
        "matmul",
        "covariance",
        "correlation",
        "drawdown_series",
        "new_high_mask",
    } <= set(operators)
    assert {
        "cumulative_return",
        "total_return",
        "annualized_return",
        "portfolio_returns",
        "active_returns",
    }.isdisjoint(operators)
    assert {item["category"] for item in operators.values()} >= {
        "basic",
        "reduction",
        "linear_algebra",
        "statistics",
        "portfolio",
        "path",
    }
    assert "inverse" not in operators
    assert all(item["version"] == "2.2.0" for item in operators.values())
    assert all(item["njit_supported"] is True for item in operators.values())
    assert all(item["kernel_version"] == "2.2.0" for item in operators.values())
    assert all(
        signature["parameters"]
        for operator in operators.values()
        for signature in operator["signatures"]
    )
    assert operators["mean"]["signatures"][0]["inputs"] == [
        "series<T> | vector<N> | matrix<A,B>"
    ]
    assert operators["mean"]["latex_template"] == r"\operatorname{mean}(x)"
    assert {
        "log",
        "difference",
        "greater_than",
        "where",
        "sum_where",
        "quantile",
        "skewness",
        "cumulative_max",
        "linear_slope",
        "regression_standard_error",
        "normal_ppf",
    } <= set(operators)
    assert "cumulative_maximum" not in operators
    assert "masked_sum" not in operators


def test_legacy_20_catalog_and_registry_contract_remain_available() -> None:
    legacy = get_typed_operator_catalog("2.0.0")
    operators = {item["id"]: item for item in legacy["operators"]}

    assert legacy["dsl_version"] == "2.0.0"
    assert legacy["compiler_version"] == "typed-ast-1"
    assert legacy["operator_registry_version"] == "2.0.0"
    assert all(item["version"] == "2.0.0" for item in operators.values())
    assert "mean" in operators
    assert "greater_than" not in operators
    assert "drawdown_series" not in operators

    legacy_dsl = get_typed_dsl_catalog("2.0.0")
    assert legacy_dsl["type_system"]["dtype"] == ["float64"]
    assert not any(item["is_mask"] for item in legacy_dsl["type_system"]["types"])
    assert "volume" not in {item["name"] for item in legacy_dsl["variables"]}


def test_variable_catalog_exposes_nominal_axes_and_symbolic_shapes() -> None:
    variables = {
        item["name"]: item["type"] for item in get_typed_variable_catalog()["variables"]
    }

    assert variables["returns"]["display"] == "series<time>[T]"
    assert variables["adjusted_nav"]["shape"] == ["L"]
    assert variables["asset_weights"]["display"] == "vector<asset>[N]"
    assert variables["asset_returns"]["axes"] == ["time", "asset"]
    assert variables["asset_returns"]["shape"] == ["T", "N"]
    assert variables["annual_risk_free_rate_decimal"]["kind"] == "scalar"
    assert variables["returns"]["semantic_dimension"] == "return_decimal"
    assert variables["adjusted_nav"]["semantic_dimension"] == "adjusted_nav"
    assert variables["market_close"]["semantic_dimension"] == "raw_market_price"
    assert variables["volume"]["semantic_dimension"] == "volume"
    assert variables["turnover_amount"]["semantic_dimension"] == "currency_amount"

    weight_path = next(
        item
        for item in get_typed_variable_catalog()["variables"]
        if item["name"] == "weight_path"
    )
    assert weight_path["shape"] == ["T", "N"]
    assert weight_path["semantic_role"] == "asset_weight_path"
    assert weight_path["source"] == "request"
    assert weight_path["context_domains"] == ["portfolio"]
    assert weight_path["label"] == "动态权重路径"

    catalog_names = {item["name"] for item in get_typed_variable_catalog()["variables"]}
    assert "market_open" in catalog_names
    assert "open_price" not in catalog_names
    assert DEFAULT_VARIABLE_TYPES["open_price"] == DEFAULT_VARIABLE_TYPES["market_open"]


def test_unknown_registry_version_has_stable_error_code() -> None:
    with pytest.raises(TypedDslError) as caught:
        get_typed_operator_catalog("99.0.0")

    assert caught.value.code == "OPERATOR_VERSION_NOT_FOUND"


def test_infer_allows_non_scalar_root_but_compose_defaults_to_scalar() -> None:
    inferred = infer_typed_expression("portfolio_returns(asset_returns, asset_weights)")

    assert inferred.output_type == ValueType.series()
    assert inferred.output_contract == "any"

    with pytest.raises(TypedDslError) as caught:
        compose_typed_expression("portfolio_returns(asset_returns, asset_weights)")

    assert caught.value.code == "OUTPUT_CONTRACT_MISMATCH"


def test_typed_dag_nodes_include_type_operator_and_cost_annotations() -> None:
    plan = compose_typed_expression(
        "mean(portfolio_returns(asset_returns, asset_weights))"
    )
    graph = plan.graph_payload()

    assert plan.output_type == ValueType.scalar()
    assert set(plan.context_requirements) == {"asset_returns", "asset_weights"}
    assert graph["roots"] == {"result": plan.root_id}
    call_nodes = [node for node in graph["nodes"] if node["operator"]]
    assert {node["operator"]["id"] for node in call_nodes} == {
        "portfolio_returns",
        "mean",
    }
    assert all("inferred_type" in node and "cost" in node for node in graph["nodes"])
    portfolio_node = next(
        node for node in call_nodes if node["operator"]["id"] == "portfolio_returns"
    )
    assert portfolio_node["arguments"] == [
        {"name": "asset_returns", "input_node_id": portfolio_node["inputs"][0]},
        {"name": "asset_weights", "input_node_id": portfolio_node["inputs"][1]},
    ]
    assert portfolio_node["formula_fragment"] == (
        "portfolio_returns(asset_returns, asset_weights)"
    )
    assert graph["estimated_cost"]["node_count"] == len(graph["nodes"])


def test_custom_variable_types_are_nominal_and_explicit() -> None:
    plan = infer_typed_expression(
        "trace(custom_covariance)",
        variable_types={
            "custom_covariance": ValueType.matrix(("asset", "asset"), (12, 12))
        },
    )

    assert plan.output_type == ValueType.scalar()
    assert plan.context_requirements["custom_covariance"].shape == (12, 12)

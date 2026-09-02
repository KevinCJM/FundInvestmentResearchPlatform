from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from cal_indicators.indicator_runtime import IndicatorRuntime
from cal_indicators.typed_dsl import (
    TypedDslError,
    TypedIndicatorRuntime,
    ValueType,
    evaluate_typed_expression,
    infer_typed_expression,
)


RETURNS = np.array(
    [
        [0.01, 0.02, -0.01],
        [0.03, -0.01, 0.02],
        [-0.02, 0.01, 0.04],
        [0.02, 0.03, 0.01],
    ],
    dtype=np.float64,
)
WEIGHTS = np.array([0.2, 0.3, 0.5], dtype=np.float64)


def test_portfolio_reduction_computes_scalar_indicator() -> None:
    result = evaluate_typed_expression(
        "mean(portfolio_returns(asset_returns, asset_weights))",
        {"asset_returns": RETURNS, "asset_weights": WEIGHTS},
    )

    assert result == pytest.approx(float(np.mean(RETURNS @ WEIGHTS)))


def test_non_scalar_preview_runtime_returns_typed_series() -> None:
    plan = infer_typed_expression(
        r"\operatorname{portfolio_returns}(\mathbf{R},\mathbf{w})"
    )
    result = TypedIndicatorRuntime.from_plan(plan).compute(
        {"asset_returns": RETURNS, "asset_weights": WEIGHTS}
    )

    assert isinstance(result, np.ndarray)
    assert result.shape == (4,)
    assert_allclose(result, RETURNS @ WEIGHTS)


def test_covariance_correlation_and_quadratic_form_match_numpy() -> None:
    covariance_runtime = TypedIndicatorRuntime.from_expression(
        "covariance(asset_returns)",
        output_contract="matrix",
    )
    covariance = covariance_runtime.compute({"asset_returns": RETURNS})
    assert_allclose(covariance, np.cov(RETURNS, rowvar=False, ddof=1))

    correlation = TypedIndicatorRuntime.from_expression(
        "correlation(asset_returns)",
        output_contract="matrix",
    ).compute({"asset_returns": RETURNS})
    assert_allclose(correlation, np.corrcoef(RETURNS, rowvar=False))

    risk = evaluate_typed_expression(
        "quadratic_form(asset_weights, covariance(asset_returns))",
        {"asset_returns": RETURNS, "asset_weights": WEIGHTS},
    )
    expected_covariance = np.cov(RETURNS, rowvar=False, ddof=1)
    assert risk == pytest.approx(float(WEIGHTS @ expected_covariance @ WEIGHTS))


def test_covariance_and_correlation_support_two_time_series() -> None:
    lhs = RETURNS[:, 0]
    rhs = RETURNS[:, 1]
    context = {"returns": lhs, "benchmark_returns": rhs}

    covariance = evaluate_typed_expression(
        "cov(returns, benchmark_returns)",
        context,
    )
    correlation = evaluate_typed_expression(
        "corr(returns, benchmark_returns)",
        context,
    )

    assert covariance == pytest.approx(float(np.cov(lhs, rhs, ddof=1)[0, 1]))
    assert correlation == pytest.approx(float(np.corrcoef(lhs, rhs)[0, 1]))


def test_reduction_and_linear_algebra_operators_compute_shapes() -> None:
    time_mean = TypedIndicatorRuntime.from_expression(
        "mean_time(asset_returns)",
        output_contract="vector",
    ).compute({"asset_returns": RETURNS})
    asset_mean = TypedIndicatorRuntime.from_expression(
        "mean_asset(asset_returns)",
        output_contract="series",
    ).compute({"asset_returns": RETURNS})
    gram_trace = evaluate_typed_expression(
        "trace(matmul(transpose(asset_returns), asset_returns))",
        {"asset_returns": RETURNS},
    )

    assert_allclose(time_mean, np.mean(RETURNS, axis=0))
    assert_allclose(asset_mean, np.mean(RETURNS, axis=1))
    assert gram_trace == pytest.approx(float(np.trace(RETURNS.T @ RETURNS)))


def test_basic_and_one_dimensional_operators_match_numpy() -> None:
    returns = RETURNS[:, 0]
    expression = "mean(clip(maximum(abs(returns), minimum(returns + 1, 0.5)), 0, 1))"
    result = evaluate_typed_expression(expression, {"returns": returns})
    expected = np.mean(
        np.clip(np.maximum(np.abs(returns), np.minimum(returns + 1, 0.5)), 0, 1)
    )

    assert result == pytest.approx(float(expected))
    assert evaluate_typed_expression(
        "last(cumulative_sum(returns))",
        {"returns": returns},
    ) == pytest.approx(float(np.sum(returns)))
    assert evaluate_typed_expression(
        "product(returns + 1)",
        {"returns": returns},
    ) == pytest.approx(float(np.prod(returns + 1)))
    assert evaluate_typed_expression(
        "dot(returns, returns)",
        {"returns": returns},
    ) == pytest.approx(float(np.dot(returns, returns)))


def test_diag_outer_and_regular_solve_match_numpy() -> None:
    covariance_type = ValueType.matrix(("asset", "asset"), ("N", "N"))
    covariance = np.diag(np.array([1.0, 2.0, 4.0], dtype=np.float64))
    context = {"covariance_matrix": covariance, "asset_weights": WEIGHTS}

    diagonal = TypedIndicatorRuntime.from_expression(
        "diag(asset_weights)",
        output_contract="matrix",
    ).compute({"asset_weights": WEIGHTS})
    outer = TypedIndicatorRuntime.from_expression(
        "outer(asset_weights, asset_weights)",
        output_contract="matrix",
    ).compute({"asset_weights": WEIGHTS})
    solved_sum = evaluate_typed_expression(
        "sum(solve(covariance_matrix, asset_weights))",
        context,
        variable_types={"covariance_matrix": covariance_type},
    )

    assert_allclose(diagonal, np.diag(WEIGHTS))
    assert_allclose(outer, np.outer(WEIGHTS, WEIGHTS))
    assert solved_sum == pytest.approx(
        float(np.sum(np.linalg.solve(covariance, WEIGHTS)))
    )


def test_cumulative_total_and_annualized_returns() -> None:
    returns = np.array([0.01, -0.02, 0.03], dtype=np.float64)
    cumulative = TypedIndicatorRuntime.from_expression(
        "cumulative_return(returns)",
        output_contract="series",
    ).compute({"returns": returns})
    total = evaluate_typed_expression("total_return(returns)", {"returns": returns})
    annualized = evaluate_typed_expression(
        "annualized_return(returns, periods_per_year)",
        {"returns": returns, "periods_per_year": 12.0},
    )

    growth = np.prod(1.0 + returns)
    assert_allclose(cumulative, np.cumprod(1.0 + returns) - 1.0)
    assert total == pytest.approx(float(growth - 1.0))
    assert annualized == pytest.approx(float(growth**4.0 - 1.0))


def test_std_and_variance_validate_ddof_at_runtime() -> None:
    returns = RETURNS[:, 0]

    assert evaluate_typed_expression(
        "std(returns)", {"returns": returns}
    ) == pytest.approx(float(np.std(returns, ddof=1)))
    assert evaluate_typed_expression(
        "variance(returns, 0)",
        {"returns": returns},
    ) == pytest.approx(float(np.var(returns, ddof=0)))

    for invalid_ddof in (-1.0, 0.5, 4.0):
        with pytest.raises(TypedDslError) as caught:
            evaluate_typed_expression(
                "std(returns, invalid_ddof)",
                {"returns": returns, "invalid_ddof": invalid_ddof},
                variable_types={"invalid_ddof": ValueType.scalar()},
            )
        assert caught.value.code == "INVALID_PARAMETER"


def test_explicit_matrix_axis_reducers_match_numpy() -> None:
    checks = {
        "product_time": np.prod(RETURNS, axis=0),
        "variance_time": np.var(RETURNS, axis=0, ddof=1),
        "min_time": np.min(RETURNS, axis=0),
        "max_time": np.max(RETURNS, axis=0),
        "product_asset": np.prod(RETURNS, axis=1),
        "variance_asset": np.var(RETURNS, axis=1, ddof=1),
        "min_asset": np.min(RETURNS, axis=1),
        "max_asset": np.max(RETURNS, axis=1),
    }
    for operator, expected in checks.items():
        contract = "vector" if operator.endswith("_time") else "series"
        actual = TypedIndicatorRuntime.from_expression(
            f"{operator}(asset_returns)",
            output_contract=contract,
        ).compute({"asset_returns": RETURNS})
        assert_allclose(actual, expected)


def test_weight_path_rows_must_sum_to_one() -> None:
    valid = np.tile(WEIGHTS, (RETURNS.shape[0], 1))
    runtime = TypedIndicatorRuntime.from_expression("mean(weight_path)")

    assert runtime.compute({"weight_path": valid}) == pytest.approx(
        float(np.mean(valid))
    )
    invalid = valid.copy()
    invalid[1, 0] += 0.1
    with pytest.raises(TypedDslError) as caught:
        runtime.compute({"weight_path": invalid})
    assert caught.value.code == "WEIGHT_SUM_INVALID"


def test_runtime_trace_records_actual_shape_and_cost() -> None:
    runtime = TypedIndicatorRuntime.from_expression(
        "mean(portfolio_returns(asset_returns, asset_weights))"
    )
    runtime.compute({"asset_returns": RETURNS, "asset_weights": WEIGHTS})

    trace = runtime.trace_payload()
    assert len(trace["nodes"]) == len(runtime.plan.nodes)
    portfolio = next(
        item
        for item in trace["nodes"]
        if runtime.plan.nodes[item["node_id"]].operator_id == "portfolio_returns"
    )
    assert portfolio["actual_shape"] == [RETURNS.shape[0]]
    assert trace["total_runtime_cost"] > 0


@pytest.mark.parametrize(
    ("context", "code"),
    (
        ({"asset_returns": RETURNS}, "CONTEXT_VARIABLE_UNAVAILABLE"),
        (
            {
                "asset_returns": RETURNS,
                "asset_weights": np.array([0.2, 0.3, 0.4]),
            },
            "WEIGHT_SUM_INVALID",
        ),
        (
            {
                "asset_returns": RETURNS,
                "asset_weights": np.array([0.2, 0.3, np.nan]),
            },
            "NON_FINITE_INPUT",
        ),
        (
            {
                "asset_returns": RETURNS,
                "asset_weights": np.array([0.25, 0.25, 0.25, 0.25]),
            },
            "RUNTIME_SHAPE_MISMATCH",
        ),
    ),
)
def test_context_validation_has_stable_error_codes(
    context: dict[str, np.ndarray],
    code: str,
) -> None:
    runtime = TypedIndicatorRuntime.from_expression(
        "mean(portfolio_returns(asset_returns, asset_weights))"
    )

    with pytest.raises(TypedDslError) as caught:
        runtime.compute(context)

    assert caught.value.code == code


def test_division_domain_and_non_finite_results_are_rejected() -> None:
    with pytest.raises(TypedDslError) as divide_error:
        evaluate_typed_expression("1 / 0", {})
    assert divide_error.value.code == "DIVIDE_BY_ZERO"

    with pytest.raises(TypedDslError) as sqrt_error:
        evaluate_typed_expression("sqrt(-1)", {})
    assert sqrt_error.value.code == "DOMAIN_ERROR"

    with pytest.raises(TypedDslError) as correlation_error:
        evaluate_typed_expression(
            "correlation(returns, benchmark_returns)",
            {
                "returns": np.ones(4, dtype=np.float64),
                "benchmark_returns": np.ones(4, dtype=np.float64),
            },
        )
    assert correlation_error.value.code == "NON_FINITE_RESULT"


def test_singular_solve_is_reported_without_exposing_numpy_error() -> None:
    covariance_type = ValueType.matrix(("asset", "asset"), ("N", "N"))
    runtime = TypedIndicatorRuntime.from_expression(
        "sum(solve(covariance_matrix, asset_weights))",
        variable_types={"covariance_matrix": covariance_type},
    )

    with pytest.raises(TypedDslError) as caught:
        runtime.compute(
            {
                "covariance_matrix": np.ones((3, 3), dtype=np.float64),
                "asset_weights": WEIGHTS,
            }
        )

    assert caught.value.code == "SINGULAR_MATRIX"


def test_shape_and_compute_limits_are_enforced_before_large_result() -> None:
    time_limited = TypedIndicatorRuntime.from_expression(
        "mean(returns)",
        max_time=2,
    )
    with pytest.raises(TypedDslError) as shape_error:
        time_limited.compute({"returns": RETURNS[:, 0]})
    assert shape_error.value.code == "SHAPE_LIMIT_EXCEEDED"

    cost_limited = TypedIndicatorRuntime.from_expression(
        "trace(covariance(asset_returns))",
        max_runtime_cost=10,
    )
    with pytest.raises(TypedDslError) as cost_error:
        cost_limited.compute({"asset_returns": RETURNS})
    assert cost_error.value.code == "COMPUTE_BUDGET_EXCEEDED"


def test_comparison_mask_where_and_where_reductions_match_numpy() -> None:
    returns = np.array([-0.03, 0.01, 0.04, -0.02, 0.02], dtype=np.float64)
    context = {"returns": returns}
    predicate_expression = "greater_than(returns, 0)"

    predicate = TypedIndicatorRuntime.from_expression(
        predicate_expression,
        output_contract="mask",
    ).compute(context)
    selected = TypedIndicatorRuntime.from_expression(
        f"where({predicate_expression}, returns, 0)",
        output_contract="series",
    ).compute(context)

    assert predicate.dtype == np.bool_
    assert_allclose(predicate, returns > 0)
    assert_allclose(selected, np.where(returns > 0, returns, 0.0))
    assert evaluate_typed_expression(
        f"count_true({predicate_expression})", context
    ) == pytest.approx(3.0)
    assert evaluate_typed_expression(
        f"sum_where(returns, {predicate_expression})", context
    ) == pytest.approx(float(np.sum(returns[returns > 0])))
    assert evaluate_typed_expression(
        f"mean_where(returns, {predicate_expression})", context
    ) == pytest.approx(float(np.mean(returns[returns > 0])))
    assert evaluate_typed_expression(
        f"std_where(returns, {predicate_expression})", context
    ) == pytest.approx(float(np.std(returns[returns > 0], ddof=1)))
    assert evaluate_typed_expression(
        f"quantile_where(returns, {predicate_expression}, 0.5)", context
    ) == pytest.approx(float(np.quantile(returns[returns > 0], 0.5)))


def test_mask_logical_operators_and_runtime_type_validation() -> None:
    returns = np.array([-1.0, 0.5, 2.0, 4.0], dtype=np.float64)
    expression = "logical_and(greater_equal(returns, 0), less_than(returns, 4))"
    actual = TypedIndicatorRuntime.from_expression(
        expression,
        output_contract="mask",
    ).compute({"returns": returns})
    assert_allclose(actual, np.logical_and(returns >= 0, returns < 4))

    runtime = TypedIndicatorRuntime.from_expression(
        "count_true(custom_mask)",
        variable_types={"custom_mask": ValueType.mask(("time",), ("T",))},
    )
    with pytest.raises(TypedDslError) as caught:
        runtime.compute({"custom_mask": np.array([0.0, 1.0])})
    assert caught.value.code == "RUNTIME_TYPE_MISMATCH"


def test_sequence_primitives_support_explicit_periods() -> None:
    values = np.array([1.0, 3.0, 2.0, 8.0], dtype=np.float64)
    context = {"returns": values}

    assert evaluate_typed_expression("first(returns)", context) == 1.0
    assert evaluate_typed_expression("length(returns)", context) == 4.0
    lagged = TypedIndicatorRuntime.from_expression(
        "lag(returns, 2)", output_contract="series"
    ).compute(context)
    differenced = TypedIndicatorRuntime.from_expression(
        "difference(returns, 2)", output_contract="series"
    ).compute(context)
    assert_allclose(lagged, values[:-2])
    assert_allclose(differenced, values[2:] - values[:-2])
    assert_allclose(
        TypedIndicatorRuntime.from_expression(
            "lag(returns, 0)", output_contract="series"
        ).compute(context),
        values,
    )
    assert_allclose(
        TypedIndicatorRuntime.from_expression(
            "cumulative_maximum(returns)", output_contract="series"
        ).compute(context),
        np.maximum.accumulate(values),
    )
    assert_allclose(
        TypedIndicatorRuntime.from_expression(
            "cumulative_min(returns)", output_contract="series"
        ).compute(context),
        np.minimum.accumulate(values),
    )
    assert evaluate_typed_expression("argmin(returns)", context) == 0.0
    assert evaluate_typed_expression("argmax(returns)", context) == 3.0

    for expression in (
        "lag(returns, -1)",
        "difference(returns, 0)",
        "lag(returns, 1.5)",
    ):
        with pytest.raises(TypedDslError) as caught:
            TypedIndicatorRuntime.from_expression(
                expression, output_contract="series"
            ).compute(context)
        assert caught.value.code == "INVALID_PARAMETER"


def test_descriptive_statistics_and_run_length_match_reference_values() -> None:
    values = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float64)
    context = {"returns": values}
    centered = values - np.mean(values)
    population_excess = np.mean(centered**4) / np.mean(centered**2) ** 2 - 3.0
    n = values.size
    corrected_excess = (
        (n - 1) / ((n - 2) * (n - 3)) * ((n + 1) * population_excess + 6.0)
    )

    assert evaluate_typed_expression("median(returns)", context) == 0.0
    assert evaluate_typed_expression("quantile(returns, 0.75)", context) == 1.0
    assert evaluate_typed_expression("skewness(returns)", context) == pytest.approx(0.0)
    assert evaluate_typed_expression(
        "excess_kurtosis(returns)", context
    ) == pytest.approx(float(corrected_excess))
    assert evaluate_typed_expression(
        "mean_absolute_deviation(returns)", context
    ) == pytest.approx(float(np.mean(np.abs(centered))))
    assert evaluate_typed_expression(
        "root_mean_square(returns)", context
    ) == pytest.approx(float(np.sqrt(np.mean(values**2))))
    assert (
        evaluate_typed_expression(
            "max_consecutive_true(greater_than(returns, -1.5))", context
        )
        == 4.0
    )


def test_elementwise_log_exp_reciprocal_and_sign_match_numpy() -> None:
    values = np.array([0.25, 0.5, 2.0], dtype=np.float64)
    variable_types = {"values": ValueType.series(semantic_dimension="dimensionless")}
    context = {"values": values}
    for expression, expected in (
        ("log(values)", np.log(values)),
        ("exp(values)", np.exp(values)),
        ("reciprocal(values)", np.reciprocal(values)),
        ("sign(values - 1)", np.sign(values - 1)),
    ):
        actual = TypedIndicatorRuntime.from_expression(
            expression,
            variable_types=variable_types,
            output_contract="series",
        ).compute(context)
        assert_allclose(actual, expected)


def test_linear_regression_and_normal_distribution_primitives() -> None:
    x = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    y = np.array([1.0, 2.0, 2.5, 5.0], dtype=np.float64)
    variable_types = {
        "x": ValueType.series(),
        "y": ValueType.series(),
    }
    context = {"x": x, "y": y}
    slope, intercept = np.polyfit(x, y, 1)
    fitted = intercept + slope * x
    r_squared = 1.0 - np.sum((y - fitted) ** 2) / np.sum((y - np.mean(y)) ** 2)
    standard_error = np.sqrt(np.sum((y - fitted) ** 2) / (y.size - 2))

    assert evaluate_typed_expression(
        "linear_slope(x, y)", context, variable_types=variable_types
    ) == pytest.approx(float(slope))
    assert evaluate_typed_expression(
        "linear_intercept(x, y)", context, variable_types=variable_types
    ) == pytest.approx(float(intercept))
    assert evaluate_typed_expression(
        "linear_r_squared(x, y)", context, variable_types=variable_types
    ) == pytest.approx(float(r_squared))
    assert evaluate_typed_expression(
        "regression_standard_error(x, y)",
        context,
        variable_types=variable_types,
    ) == pytest.approx(float(standard_error))
    assert evaluate_typed_expression("normal_pdf(0)", {}) == pytest.approx(
        1.0 / np.sqrt(2.0 * np.pi)
    )
    assert evaluate_typed_expression("normal_ppf(0.5)", {}) == pytest.approx(0.0)

    for probability in (0.0, 1.0):
        with pytest.raises(TypedDslError) as caught:
            evaluate_typed_expression(f"normal_ppf({probability})", {})
        assert caught.value.code == "DOMAIN_ERROR"


def test_legacy_v1_runtime_remains_scalar_only_and_unchanged() -> None:
    runtime = IndicatorRuntime.from_definition(
        "平均收益",
        r"\overline{\mathbf{r}}",
        ["1M"],
    )
    result = runtime.compute_period(
        "1M",
        {"returns": np.array([0.01, 0.02, 0.03], dtype=np.float64)},
    )

    assert result["平均收益"] == pytest.approx(0.02)


def test_legacy_typed_20_plan_still_executes_with_20_registry() -> None:
    runtime = TypedIndicatorRuntime.from_expression(
        "mean(returns)",
        dsl_version="2.0.0",
    )
    assert runtime.plan.operator_registry_version == "2.0.0"
    assert runtime.compute({"returns": np.array([1.0, 2.0, 3.0])}) == 2.0

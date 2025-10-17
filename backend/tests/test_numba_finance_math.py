import numpy as np
from numpy.testing import assert_allclose

from cal_indicators.numba_finance_math import (
    cumulative_net_value,
    cumulative_simple_returns,
    sequence_add,
    sequence_compound_annual_growth_rate,
    sequence_cumsum,
    sequence_cumprod,
    sequence_divide,
    sequence_drawdown,
    sequence_excess_return,
    sequence_geometric_mean,
    sequence_max,
    sequence_mean,
    sequence_min,
    sequence_mean_variance_std,
    sequence_multiply,
    sequence_prod,
    sequence_std,
    sequence_subtract,
    sequence_sum,
    sequence_variance,
)


def test_sequence_sum_and_cumsum():
    data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    empty = np.array([], dtype=np.float64)
    assert sequence_sum(data) == 6.0
    assert sequence_sum(empty) == 0.0
    assert_allclose(sequence_cumsum(data), np.array([1.0, 3.0, 6.0], dtype=np.float64))


def test_sequence_prod_and_cumprod():
    data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    empty = np.array([], dtype=np.float64)
    assert sequence_prod(data) == 6.0
    assert sequence_prod(empty) == 1.0
    assert_allclose(sequence_cumprod(data), np.array([1.0, 2.0, 6.0], dtype=np.float64))


def test_sequence_mean_variance_std():
    data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    empty = np.array([], dtype=np.float64)
    mean_val, variance, std = sequence_mean_variance_std(data, 1)
    assert_allclose(mean_val, 2.5)
    assert_allclose(variance, 1.6666666666666667)
    assert_allclose(std, np.sqrt(1.6666666666666667))
    mean_val_ddof0, variance_ddof0, std_ddof0 = sequence_mean_variance_std(data, 0)
    assert_allclose(mean_val_ddof0, 2.5)
    assert_allclose(variance_ddof0, 1.25)
    assert_allclose(std_ddof0, np.sqrt(1.25))
    stats_empty = sequence_mean_variance_std(empty, 1)
    assert np.isnan(stats_empty[0])
    assert np.isnan(stats_empty[1])
    assert np.isnan(stats_empty[2])


def test_sequence_mean_variance_std_individual_functions_consistency():
    data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    _, variance, std = sequence_mean_variance_std(data, 1)
    assert_allclose(variance, sequence_variance(data, 1))
    assert_allclose(std, sequence_std(data, 1))


def test_sequence_mean_variance_std_reuse():
    data = np.array([3.0, 5.0, 7.0, 9.0], dtype=np.float64)
    mean_val, variance, std = sequence_mean_variance_std(data, 1)
    assert_allclose(mean_val, 6.0)
    assert_allclose(variance, 6.666666666666667)
    assert_allclose(std, np.sqrt(6.666666666666667))


def test_sequence_mean_variance_std_single_element():
    data = np.array([2.0], dtype=np.float64)
    mean_val, variance, std = sequence_mean_variance_std(data, 0)
    assert_allclose(mean_val, 2.0)
    assert_allclose(variance, 0.0)
    assert_allclose(std, 0.0)
    mean_val_ddof1, variance_ddof1, std_ddof1 = sequence_mean_variance_std(data, 1)
    assert_allclose(mean_val_ddof1, 2.0)
    assert np.isnan(variance_ddof1)
    assert np.isnan(std_ddof1)


def test_sequence_mean_variance_std_custom_ddof():
    data = np.array([10.0, 12.0, 14.0, 16.0], dtype=np.float64)
    mean_val, variance, std = sequence_mean_variance_std(data, 2)
    assert_allclose(mean_val, 13.0)
    assert_allclose(variance, 10.0)
    assert_allclose(std, np.sqrt(10.0))


def test_sequence_mean_variance_std_and_mean_function_alignment():
    data = np.array([4.0, 8.0, 12.0, 16.0], dtype=np.float64)
    mean_val, _, _ = sequence_mean_variance_std(data, 1)
    assert_allclose(mean_val, sequence_mean(data))


def test_sequence_mean_variance_std_and_variance_edge_ddof():
    data = np.array([1.0, 1.0], dtype=np.float64)
    _, variance_ddof1, std_ddof1 = sequence_mean_variance_std(data, 1)
    assert_allclose(variance_ddof1, 0.0)
    assert_allclose(std_ddof1, 0.0)


def test_individual_mean_variance_std_functions_basics():
    data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    empty = np.array([], dtype=np.float64)
    assert sequence_mean(data) == 2.5
    assert np.isnan(sequence_mean(empty))
    assert_allclose(sequence_variance(data, 1), 1.6666666666666667)
    assert_allclose(sequence_variance(data, 0), 1.25)
    assert np.isnan(sequence_variance(empty, 1))
    assert_allclose(sequence_std(data, 1), np.sqrt(1.6666666666666667))
    assert np.isnan(sequence_std(empty, 1))


def test_sequence_min_max():
    data = np.array([3.0, -1.0, 5.0, 0.0], dtype=np.float64)
    empty = np.array([], dtype=np.float64)
    assert sequence_min(data) == -1.0
    assert sequence_max(data) == 5.0
    assert np.isnan(sequence_min(empty))
    assert np.isnan(sequence_max(empty))


def test_sequence_elementwise_ops():
    lhs = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    rhs = np.array([4.0, 5.0, 6.0], dtype=np.float64)
    div_rhs = np.array([1.0, 0.5, 0.0], dtype=np.float64)
    assert_allclose(sequence_add(lhs, rhs), np.array([5.0, 7.0, 9.0], dtype=np.float64))
    assert_allclose(sequence_subtract(lhs, rhs), np.array([-3.0, -3.0, -3.0], dtype=np.float64))
    assert_allclose(sequence_multiply(lhs, rhs), np.array([4.0, 10.0, 18.0], dtype=np.float64))
    division = sequence_divide(lhs, div_rhs)
    assert_allclose(division[:2], np.array([1.0, 4.0], dtype=np.float64))
    assert np.isnan(division[2])


def test_cumulative_simple_returns_and_nav():
    returns = np.array([0.1, -0.05, 0.2], dtype=np.float64)
    expected_cum = np.array([0.1, 0.045, 0.254], dtype=np.float64)
    assert_allclose(cumulative_simple_returns(returns), expected_cum, rtol=1e-9, atol=1e-9)
    nav_path = cumulative_net_value(100.0, returns)
    assert_allclose(nav_path, np.array([110.0, 104.5, 125.4], dtype=np.float64))


def test_sequence_geometric_mean():
    values = np.array([1.1, 1.2, 1.3], dtype=np.float64)
    expected = (1.1 * 1.2 * 1.3) ** (1.0 / 3.0)
    assert_allclose(sequence_geometric_mean(values), expected)
    invalid = np.array([1.0, -1.0], dtype=np.float64)
    assert np.isnan(sequence_geometric_mean(invalid))
    empty = np.array([], dtype=np.float64)
    assert np.isnan(sequence_geometric_mean(empty))


def test_sequence_compound_annual_growth_rate():
    returns = np.array([0.1, 0.1, 0.1, 0.1], dtype=np.float64)
    cagr = sequence_compound_annual_growth_rate(returns, 4.0)
    assert_allclose(cagr, (1.1 ** 4) - 1.0)
    zero_returns = np.array([], dtype=np.float64)
    assert np.isnan(sequence_compound_annual_growth_rate(zero_returns, 4.0))


def test_sequence_drawdown():
    nav = np.array([100.0, 110.0, 105.0, 120.0, 115.0], dtype=np.float64)
    expected = np.array([0.0, 0.0, -0.045454545454545456, 0.0, -0.041666666666666685], dtype=np.float64)
    assert_allclose(sequence_drawdown(nav), expected)
    nav_with_zero = np.array([0.0, 0.1], dtype=np.float64)
    drawdown = sequence_drawdown(nav_with_zero)
    assert np.isnan(drawdown[0])
    assert_allclose(drawdown[1], 0.0)


def test_sequence_excess_return():
    asset = np.array([0.05, 0.02, -0.01], dtype=np.float64)
    bench = np.array([0.03, 0.01, 0.0], dtype=np.float64)
    expected = np.array([0.02, 0.01, -0.01], dtype=np.float64)
    assert_allclose(sequence_excess_return(asset, bench), expected)

import numpy as np
from numpy.testing import assert_allclose

from cal_indicators.indicator_config import load_indicator_config
from cal_indicators.numba_finance_math import (
    cumulative_net_value,
    sequence_drawdown,
    sequence_mean,
    sequence_min,
    sequence_sum,
    sequence_variance,
)
from cal_indicators.numba_finance_indicators import (
    annualization_factor,
    annualized_return,
    annualized_volatility,
    average_return,
    calmar_ratio,
    cumulative_return,
    excess_return,
    loss_rate,
    max_drawdown_rate_from_min,
    scale_excess_return_for_sharpe,
    sharpe_ratio,
    volatility_from_variance,
    win_rate,
)


def test_indicator_atomic_computations():
    returns = np.array([0.01, -0.02, 0.015, 0.005, -0.01, 0.02], dtype=np.float64)
    config = load_indicator_config()
    total_sum = sequence_sum(returns)
    total_count = returns.shape[0]

    avg_return = average_return(total_sum, total_count)
    assert_allclose(avg_return, np.mean(returns))

    nav_series = cumulative_net_value(config["initial_nav"], returns)
    final_nav = nav_series[-1]
    initial_nav = nav_series[0]
    cum_return = cumulative_return(final_nav, initial_nav)
    expected_cum = np.prod(1.0 + returns) - 1.0
    assert_allclose(cum_return, expected_cum)

    factor = annualization_factor(config["periods_per_year"], total_count)
    ann_return = annualized_return(cum_return, factor)
    expected_ann = (1.0 + expected_cum) ** (config["periods_per_year"] / total_count) - 1.0
    assert_allclose(ann_return, expected_ann)

    variance = sequence_variance(returns, config["ddof"])
    period_volatility = volatility_from_variance(variance)
    expected_period_vol = np.std(returns, ddof=config["ddof"])
    assert_allclose(period_volatility, expected_period_vol)

    ann_volatility = annualized_volatility(period_volatility, config["periods_per_year"])
    expected_ann_vol = expected_period_vol * np.sqrt(config["periods_per_year"])
    assert_allclose(ann_volatility, expected_ann_vol)

    mean_return = sequence_mean(returns)
    per_period_excess = excess_return(mean_return, config["risk_free_rate_per_period"])
    scaled_excess = scale_excess_return_for_sharpe(per_period_excess, config["periods_per_year"])
    expected_scaled_excess = (mean_return - config["risk_free_rate_per_period"]) * np.sqrt(
        config["periods_per_year"]
    )
    assert_allclose(scaled_excess, expected_scaled_excess)

    sharpe = sharpe_ratio(scaled_excess, ann_volatility)
    expected_sharpe = expected_scaled_excess / expected_ann_vol
    assert_allclose(sharpe, expected_sharpe)

    win_threshold = config["win_threshold"]
    loss_threshold = config["loss_threshold"]
    win_count = int(np.sum(returns > win_threshold))
    loss_count = int(np.sum(returns < loss_threshold))
    expected_win_rate = win_count / total_count
    expected_loss_rate = loss_count / total_count
    assert_allclose(win_rate(win_count, total_count), expected_win_rate)
    assert_allclose(loss_rate(loss_count, total_count), expected_loss_rate)

    drawdown_series = sequence_drawdown(nav_series)
    min_drawdown = sequence_min(drawdown_series)
    max_dd = max_drawdown_rate_from_min(min_drawdown)
    expected_max_dd = -min_drawdown
    assert_allclose(max_dd, expected_max_dd)

    calmar = calmar_ratio(ann_return, max_dd)
    expected_calmar = expected_ann / expected_max_dd
    assert_allclose(calmar, expected_calmar)


def test_indicator_edge_cases():
    assert np.isnan(average_return(0.0, 0))
    assert np.isnan(cumulative_return(1.0, 0.0))

    invalid_factor = annualization_factor(252.0, 0)
    assert np.isnan(invalid_factor)
    assert np.isnan(annualized_return(0.1, invalid_factor))
    assert np.isnan(annualized_return(-1.2, 1.0))

    assert np.isnan(volatility_from_variance(-0.1))
    assert np.isnan(annualized_volatility(np.nan, 252.0))
    assert np.isnan(annualized_volatility(0.1, -10.0))

    assert np.isnan(scale_excess_return_for_sharpe(0.01, -252.0))
    assert np.isnan(sharpe_ratio(0.1, 0.0))

    assert np.isnan(win_rate(0, 0))
    assert np.isnan(loss_rate(0, 0))

    assert np.isnan(max_drawdown_rate_from_min(np.nan))
    assert np.isnan(calmar_ratio(np.nan, 0.1))
    assert np.isnan(calmar_ratio(0.1, np.nan))
    assert np.isnan(calmar_ratio(0.1, 0.0))

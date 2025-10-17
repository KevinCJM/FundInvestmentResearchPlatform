import numpy as np
from numpy.testing import assert_allclose

from cal_indicators.indicator_config import load_indicator_config
from cal_indicators.numba_finance_indicators import (
    annualized_return,
    annualized_volatility,
    average_return,
    calmar_ratio,
    cumulative_return,
    loss_rate,
    max_drawdown_rate,
    max_drawdown_recovery_periods,
    sharpe_ratio,
    volatility,
    win_rate,
)


def _compute_nav(returns, initial_nav=1.0):
    n = returns.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.float64)
    nav = np.empty(n, dtype=np.float64)
    nav_value = initial_nav
    for i in range(n):
        nav_value *= 1.0 + returns[i]
        nav[i] = nav_value
    return nav


def _max_drawdown(nav):
    if nav.shape[0] == 0:
        return np.nan
    peaks = np.maximum.accumulate(nav)
    drawdowns = (nav - peaks) / peaks
    return -np.min(drawdowns)


def _max_drawdown_recovery(nav):
    n = nav.shape[0]
    if n == 0:
        return 0
    peak = nav[0]
    peak_index = 0
    drawdown_active = False
    drawdown_peak_index = 0
    max_length = 0
    for i in range(1, n):
        value = nav[i]
        if value >= peak:
            if drawdown_active:
                length = i - drawdown_peak_index
                if length > max_length:
                    max_length = length
                drawdown_active = False
            peak = value
            peak_index = i
        else:
            if not drawdown_active:
                drawdown_active = True
                drawdown_peak_index = peak_index
    if drawdown_active:
        length = (n - 1) - drawdown_peak_index
        if length > max_length:
            max_length = length
    return max_length


def test_indicator_suite_basic():
    returns = np.array([0.01, -0.02, 0.015, 0.005, -0.01, 0.02], dtype=np.float64)
    config = load_indicator_config()
    periods_per_year = config["periods_per_year"]
    risk_free = config["risk_free_rate_per_period"]
    ddof = config["ddof"]
    win_threshold = config["win_threshold"]
    loss_threshold = config["loss_threshold"]
    initial_nav = config["initial_nav"]

    expected_cum = np.prod(1.0 + returns) - 1.0
    assert_allclose(cumulative_return(returns), expected_cum)

    expected_ann_ret = (1.0 + expected_cum) ** (periods_per_year / returns.shape[0]) - 1.0
    assert_allclose(annualized_return(returns, periods_per_year), expected_ann_ret)

    expected_vol = np.std(returns, ddof=ddof)
    assert_allclose(volatility(returns, ddof), expected_vol)

    expected_ann_vol = expected_vol * np.sqrt(periods_per_year)
    assert_allclose(annualized_volatility(returns, periods_per_year, ddof), expected_ann_vol)

    mean_return = np.mean(returns)
    expected_sharpe = (mean_return - risk_free) * np.sqrt(periods_per_year) / expected_vol
    assert_allclose(sharpe_ratio(returns, risk_free, periods_per_year, ddof), expected_sharpe)

    expected_win_rate = np.sum(returns > win_threshold) / returns.shape[0]
    expected_loss_rate = np.sum(returns < loss_threshold) / returns.shape[0]
    assert_allclose(win_rate(returns, win_threshold), expected_win_rate)
    assert_allclose(loss_rate(returns, loss_threshold), expected_loss_rate)

    nav = _compute_nav(returns, initial_nav)
    expected_max_dd = _max_drawdown(nav)
    assert_allclose(max_drawdown_rate(returns, initial_nav), expected_max_dd)

    expected_recovery = _max_drawdown_recovery(nav)
    assert max_drawdown_recovery_periods(returns, initial_nav) == expected_recovery

    expected_calmar = expected_ann_ret / expected_max_dd
    assert_allclose(calmar_ratio(returns, periods_per_year, initial_nav), expected_calmar)

    assert_allclose(average_return(returns), np.mean(returns))


def test_indicators_edge_cases():
    empty = np.array([], dtype=np.float64)
    config = load_indicator_config()
    assert np.isnan(cumulative_return(empty))
    assert np.isnan(annualized_return(empty, config["periods_per_year"]))
    assert np.isnan(volatility(empty, config["ddof"]))
    assert np.isnan(annualized_volatility(empty, config["periods_per_year"], config["ddof"]))
    assert np.isnan(sharpe_ratio(empty, config["risk_free_rate_per_period"], config["periods_per_year"], config["ddof"]))
    assert np.isnan(win_rate(empty, config["win_threshold"]))
    assert np.isnan(loss_rate(empty, config["loss_threshold"]))
    assert np.isnan(max_drawdown_rate(empty, config["initial_nav"]))
    assert max_drawdown_recovery_periods(empty, config["initial_nav"]) == 0
    assert np.isnan(calmar_ratio(empty, config["periods_per_year"], config["initial_nav"]))
    assert np.isnan(average_return(empty))


def test_sharpe_ratio_zero_volatility():
    returns = np.array([0.01, 0.01, 0.01, 0.01], dtype=np.float64)
    config = load_indicator_config()
    periods_per_year = config["periods_per_year"]
    risk_free = config["risk_free_rate_per_period"]
    ddof = config["ddof"]
    assert np.isnan(sharpe_ratio(returns, risk_free, periods_per_year, ddof))


def test_drawdown_metrics_monotonic_increase():
    returns = np.array([0.01, 0.01, 0.01, 0.01], dtype=np.float64)
    config = load_indicator_config()
    initial_nav = config["initial_nav"]
    assert_allclose(max_drawdown_rate(returns, initial_nav), 0.0)
    assert max_drawdown_recovery_periods(returns, initial_nav) == 0


def test_indicator_config_loader_default_path():
    config = load_indicator_config()
    assert config["periods_per_year"] == 252.0
    assert config["ddof"] == 1
    assert config["risk_free_rate_per_period"] == 0.0001
    assert config["win_threshold"] == 0.0
    assert config["loss_threshold"] == 0.0
    assert config["initial_nav"] == 1.0


def test_indicator_config_loader_custom_path(tmp_path):
    custom_path = tmp_path / "custom_config.json"
    custom_path.write_text(
        '{"periods_per_year": 360.0, "risk_free_rate_per_period": 0.0, "ddof": 2, "win_threshold": 0.01, "loss_threshold": -0.01, "initial_nav": 100.0}',
        encoding="utf-8",
    )
    config = load_indicator_config(str(custom_path))
    assert config["periods_per_year"] == 360.0
    assert config["ddof"] == 2
    assert config["risk_free_rate_per_period"] == 0.0
    assert config["win_threshold"] == 0.01
    assert config["loss_threshold"] == -0.01
    assert config["initial_nav"] == 100.0

import numpy as np
from numba import njit

try:
    from backend.cal_indicators.numba_finance_math import (
        cumulative_net_value,
        cumulative_simple_returns,
        sequence_compound_annual_growth_rate,
        sequence_drawdown,
        sequence_mean,
        sequence_mean_variance_std,
        sequence_min,
    )
except ImportError:
    from cal_indicators.numba_finance_math import (
        cumulative_net_value,
        cumulative_simple_returns,
        sequence_compound_annual_growth_rate,
        sequence_drawdown,
        sequence_mean,
        sequence_mean_variance_std,
        sequence_min,
    )


@njit("float64(float64[:])", nogil=True, fastmath=True)
def cumulative_return(returns):
    """
    计算累计收益率，返回最终累计值。
    """
    cumulative = cumulative_simple_returns(returns)
    if cumulative.shape[0] == 0:
        return np.nan
    return cumulative[-1]


@njit("float64(float64[:], float64)", nogil=True, fastmath=True)
def annualized_return(returns, periods_per_year):
    """
    根据普通收益率序列计算年化收益率。
    """
    if periods_per_year <= 0.0:
        return np.nan
    return sequence_compound_annual_growth_rate(returns, periods_per_year)


@njit("float64(float64[:], int64)", nogil=True, fastmath=True)
def volatility(returns, ddof=1):
    """
    计算收益率序列的波动率（标准差）。
    """
    n = returns.shape[0]
    if n == 0 or n <= ddof:
        return np.nan
    _, _, std_val = sequence_mean_variance_std(returns, ddof)
    return std_val


@njit("float64(float64[:], float64, int64)", nogil=True, fastmath=True)
def annualized_volatility(returns, periods_per_year, ddof=1):
    """
    计算收益率序列的年化波动率。
    """
    if periods_per_year <= 0.0:
        return np.nan
    std_val = volatility(returns, ddof)
    if np.isnan(std_val):
        return np.nan
    return std_val * np.sqrt(periods_per_year)


@njit("float64(float64[:], float64, float64, int64)", nogil=True, fastmath=True)
def sharpe_ratio(returns, risk_free_rate_per_period, periods_per_year, ddof=1):
    """
    计算年化夏普比率。
    """
    n = returns.shape[0]
    if n == 0 or n <= ddof or periods_per_year <= 0.0:
        return np.nan
    mean_val, _, std_val = sequence_mean_variance_std(returns, ddof)
    if np.isnan(std_val) or std_val <= 1e-12:
        return np.nan
    excess_mean = mean_val - risk_free_rate_per_period
    return (excess_mean * np.sqrt(periods_per_year)) / std_val


@njit("float64(float64[:], float64)", nogil=True, fastmath=True)
def win_rate(returns, threshold=0.0):
    """
    计算日胜率（收益率大于0的概率）。
    """
    n = returns.shape[0]
    if n == 0:
        return np.nan
    wins = 0
    for i in range(n):
        if returns[i] > threshold:
            wins += 1
    return wins / n


@njit("float64(float64[:], float64)", nogil=True, fastmath=True)
def loss_rate(returns, threshold=0.0):
    """
    计算日败率（收益率小于0的概率）。
    """
    n = returns.shape[0]
    if n == 0:
        return np.nan
    losses = 0
    for i in range(n):
        if returns[i] < threshold:
            losses += 1
    return losses / n


@njit("float64(float64[:], float64)", nogil=True, fastmath=True)
def max_drawdown_rate(returns, initial_nav=1.0):
    """
    计算最大回撤率（正值表示回撤幅度）。
    """
    nav = cumulative_net_value(initial_nav, returns)
    if nav.shape[0] == 0:
        return np.nan
    drawdown = sequence_drawdown(nav)
    if drawdown.shape[0] == 0:
        return np.nan
    min_drawdown = sequence_min(drawdown)
    if np.isnan(min_drawdown):
        return np.nan
    return -min_drawdown


@njit("int64(float64[:], float64)", nogil=True, fastmath=True)
def max_drawdown_recovery_periods(returns, initial_nav=1.0):
    """
    计算最大回撤修复天数（按数据点计）。
    """
    nav = cumulative_net_value(initial_nav, returns)
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


@njit("float64(float64[:], float64, float64)", nogil=True, fastmath=True)
def calmar_ratio(returns, periods_per_year, initial_nav=1.0):
    """
    计算卡玛比率：年化收益率 / 最大回撤率。
    """
    annual_ret = annualized_return(returns, periods_per_year)
    max_dd = max_drawdown_rate(returns, initial_nav)
    if np.isnan(annual_ret) or np.isnan(max_dd) or max_dd <= 1e-12:
        return np.nan
    return annual_ret / max_dd


@njit("float64(float64[:])", nogil=True, fastmath=True)
def average_return(returns):
    """
    计算平均收益率（单次循环结果）。
    """
    return sequence_mean(returns)

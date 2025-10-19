import numpy as np
from numba import njit


@njit("float64(float64, int64)", nogil=True, fastmath=True)
def average_return(total_sum, observation_count):
    """
    根据总和与样本量计算平均收益率。
    """
    if observation_count <= 0:
        return np.nan
    return total_sum / observation_count


@njit("float64(float64, float64)", nogil=True, fastmath=True)
def cumulative_return(final_nav, initial_nav):
    """
    根据期末与期初净值计算累计收益率。
    """
    if initial_nav <= 0.0:
        return np.nan
    return (final_nav / initial_nav) - 1.0


@njit("float64(float64, int64)", nogil=True, fastmath=True)
def annualization_factor(periods_per_year, total_periods):
    """
    计算年化幂次系数。
    """
    if periods_per_year <= 0.0 or total_periods <= 0:
        return np.nan
    return periods_per_year / total_periods


@njit("float64(float64, float64)", nogil=True, fastmath=True)
def annualized_return(cumulative_return_value, annualization_factor_value):
    """
    根据累计收益率与年化幂次系数计算年化收益率。
    """
    if (
            np.isnan(annualization_factor_value)
            or annualization_factor_value <= 0.0
            or (1.0 + cumulative_return_value) <= 0.0
    ):
        return np.nan
    return (1.0 + cumulative_return_value) ** annualization_factor_value - 1.0


@njit("float64(float64)", nogil=True, fastmath=True)
def volatility_from_variance(variance_value):
    """
    根据方差计算波动率（标准差）。
    """
    if variance_value < 0.0:
        return np.nan
    return np.sqrt(variance_value)


@njit("float64(float64, float64)", nogil=True, fastmath=True)
def annualized_volatility(period_volatility, periods_per_year):
    """
    根据周期波动率与年化频率计算年化波动率。
    """
    if periods_per_year <= 0.0 or np.isnan(period_volatility):
        return np.nan
    return period_volatility * np.sqrt(periods_per_year)


@njit("float64(float64, float64)", nogil=True, fastmath=True)
def excess_return(mean_return, risk_free_rate_per_period):
    """
    计算超额收益率（单期）。
    """
    return mean_return - risk_free_rate_per_period


@njit("float64(float64, float64)", nogil=True, fastmath=True)
def scale_excess_return_for_sharpe(excess_return_per_period, periods_per_year):
    """
    将单期超额收益率缩放至年化夏普比率的分子。
    """
    if periods_per_year <= 0.0:
        return np.nan
    return excess_return_per_period * np.sqrt(periods_per_year)


@njit("float64(float64, float64)", nogil=True, fastmath=True)
def sharpe_ratio(annualized_excess_return, annualized_volatility_value):
    """
    根据年化超额收益率与年化波动率计算夏普比率。
    """
    if np.isnan(annualized_volatility_value) or annualized_volatility_value <= 1e-12:
        return np.nan
    return annualized_excess_return / annualized_volatility_value


@njit("float64(int64, int64)", nogil=True, fastmath=True)
def win_rate(win_count, total_count):
    """
    根据胜利次数与样本总数计算胜率。
    """
    if total_count <= 0:
        return np.nan
    return win_count / total_count


@njit("float64(int64, int64)", nogil=True, fastmath=True)
def loss_rate(loss_count, total_count):
    """
    根据失败次数与样本总数计算败率。
    """
    if total_count <= 0:
        return np.nan
    return loss_count / total_count


@njit("float64(float64)", nogil=True, fastmath=True)
def max_drawdown_rate_from_min(min_drawdown_value):
    """
    将最小回撤（负值）转换为正的最大回撤率。
    """
    if np.isnan(min_drawdown_value):
        return np.nan
    return -min_drawdown_value


@njit("float64(float64, float64)", nogil=True, fastmath=True)
def calmar_ratio(annual_return_value, max_drawdown_rate_value):
    """
    根据年化收益率与最大回撤率计算卡玛比率。
    """
    if (
            np.isnan(annual_return_value)
            or np.isnan(max_drawdown_rate_value)
            or max_drawdown_rate_value <= 1e-12
    ):
        return np.nan
    return annual_return_value / max_drawdown_rate_value

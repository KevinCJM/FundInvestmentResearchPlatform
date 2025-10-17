import numpy as np
from numba import njit


@njit("float64(float64[:])", nogil=True, fastmath=True)
def sequence_sum(values):
    """
    计算序列求和，空序列返回0.0。
    """
    total = 0.0
    for i in range(values.shape[0]):
        total += values[i]
    return total


@njit("float64[:](float64[:])", nogil=True, fastmath=True)
def sequence_cumsum(values):
    """
    计算序列的累计和。
    """
    n = values.shape[0]
    result = np.empty(n, dtype=np.float64)
    running = 0.0
    for i in range(n):
        running += values[i]
        result[i] = running
    return result


@njit("float64(float64[:])", nogil=True, fastmath=True)
def sequence_prod(values):
    """
    计算序列乘积，空序列返回1.0。
    """
    product = 1.0
    for i in range(values.shape[0]):
        product *= values[i]
    return product


@njit("float64[:](float64[:])", nogil=True, fastmath=True)
def sequence_cumprod(values):
    """
    计算序列的累计乘积。
    """
    n = values.shape[0]
    result = np.empty(n, dtype=np.float64)
    running = 1.0
    for i in range(n):
        running *= values[i]
        result[i] = running
    return result


@njit("float64(float64[:])", nogil=True, fastmath=True)
def sequence_mean(values):
    """
    计算序列均值，空序列返回np.nan。
    """
    n = values.shape[0]
    if n == 0:
        return np.nan
    return sequence_sum(values) / n


@njit("UniTuple(float64, 3)(float64[:], int64)", nogil=True, fastmath=True)
def sequence_mean_variance_std(values, ddof=1):
    """
    单次循环计算均值、方差和标准差。
    """
    n = values.shape[0]
    if n == 0:
        return np.nan, np.nan, np.nan

    mean_val = 0.0
    m2 = 0.0
    count = 0

    for i in range(n):
        count += 1
        x = values[i]
        delta = x - mean_val
        mean_val += delta / count
        delta2 = x - mean_val
        m2 += delta * delta2

    if count <= ddof:
        variance = np.nan
        std = np.nan
    else:
        variance = m2 / (count - ddof)
        std = np.sqrt(variance)
    return mean_val, variance, std


@njit("float64(float64[:], int64)", nogil=True, fastmath=True)
def sequence_variance(values, ddof=1):
    """
    计算序列样本方差（默认ddof=1），样本不足返回np.nan。
    """
    if values.shape[0] == 0 or values.shape[0] <= ddof:
        return np.nan
    _, variance, _ = sequence_mean_variance_std(values, ddof)
    return variance


@njit("float64(float64[:], int64)", nogil=True, fastmath=True)
def sequence_std(values, ddof=1):
    """
    计算序列样本标准差，样本不足返回np.nan。
    """
    if values.shape[0] == 0 or values.shape[0] <= ddof:
        return np.nan
    _, variance, std = sequence_mean_variance_std(values, ddof)
    return std


@njit("float64(float64[:])", nogil=True, fastmath=True)
def sequence_min(values):
    """
    计算序列最小值，空序列返回np.nan。
    """
    n = values.shape[0]
    if n == 0:
        return np.nan
    current_min = values[0]
    for i in range(1, n):
        if values[i] < current_min:
            current_min = values[i]
    return current_min


@njit("float64(float64[:])", nogil=True, fastmath=True)
def sequence_max(values):
    """
    计算序列最大值，空序列返回np.nan。
    """
    n = values.shape[0]
    if n == 0:
        return np.nan
    current_max = values[0]
    for i in range(1, n):
        if values[i] > current_max:
            current_max = values[i]
    return current_max


@njit("float64[:](float64[:], float64[:])", nogil=True, fastmath=True)
def sequence_add(lhs, rhs):
    """
    对两个等长序列逐元素相加。
    """
    if lhs.shape[0] != rhs.shape[0]:
        raise ValueError("序列长度不一致，无法相加")
    n = lhs.shape[0]
    result = np.empty(n, dtype=np.float64)
    for i in range(n):
        result[i] = lhs[i] + rhs[i]
    return result


@njit("float64[:](float64[:], float64[:])", nogil=True, fastmath=True)
def sequence_subtract(lhs, rhs):
    """
    对两个等长序列逐元素相减。
    """
    if lhs.shape[0] != rhs.shape[0]:
        raise ValueError("序列长度不一致，无法相减")
    n = lhs.shape[0]
    result = np.empty(n, dtype=np.float64)
    for i in range(n):
        result[i] = lhs[i] - rhs[i]
    return result


@njit("float64[:](float64[:], float64[:])", nogil=True, fastmath=True)
def sequence_multiply(lhs, rhs):
    """
    对两个等长序列逐元素相乘。
    """
    if lhs.shape[0] != rhs.shape[0]:
        raise ValueError("序列长度不一致，无法相乘")
    n = lhs.shape[0]
    result = np.empty(n, dtype=np.float64)
    for i in range(n):
        result[i] = lhs[i] * rhs[i]
    return result


@njit("float64[:](float64[:], float64[:])", nogil=True, fastmath=True)
def sequence_divide(lhs, rhs):
    """
    对两个等长序列逐元素相除，除数为0时返回np.nan。
    """
    if lhs.shape[0] != rhs.shape[0]:
        raise ValueError("序列长度不一致，无法相除")
    n = lhs.shape[0]
    result = np.empty(n, dtype=np.float64)
    for i in range(n):
        if np.abs(rhs[i]) < 1e-12:
            result[i] = np.nan
        else:
            result[i] = lhs[i] / rhs[i]
    return result


@njit("float64[:](float64[:])", nogil=True, fastmath=True)
def cumulative_simple_returns(returns):
    """
    计算普通收益率的累计收益率，返回与输入等长的序列。
    """
    n = returns.shape[0]
    cumulative = np.empty(n, dtype=np.float64)
    growth = 1.0
    for i in range(n):
        growth *= 1.0 + returns[i]
        cumulative[i] = growth - 1.0
    return cumulative


@njit("float64(float64[:])", nogil=True, fastmath=True)
def sequence_geometric_mean(values):
    """
    计算序列的几何平均数，遇到非正值返回np.nan。
    """
    n = values.shape[0]
    if n == 0:
        return np.nan
    product = 1.0
    for i in range(n):
        if values[i] <= 0.0:
            return np.nan
        product *= values[i]
    return product ** (1.0 / n)


@njit("float64[:](float64, float64[:])", nogil=True, fastmath=True)
def cumulative_net_value(initial_nav, returns):
    """
    根据初始净值和普通收益率序列计算净值轨迹。
    """
    n = returns.shape[0]
    result = np.empty(n, dtype=np.float64)
    nav = initial_nav
    for i in range(n):
        nav *= 1.0 + returns[i]
        result[i] = nav
    return result


@njit("float64(float64[:], float64)", nogil=True, fastmath=True)
def sequence_compound_annual_growth_rate(returns, periods_per_year):
    """
    根据普通收益率序列估算复合年化增长率(CAGR)。
    """
    cumulative = cumulative_simple_returns(returns)
    if cumulative.shape[0] == 0:
        return np.nan
    final_growth = 1.0 + cumulative[-1]
    total_periods = returns.shape[0]
    if final_growth <= 0.0 or total_periods == 0:
        return np.nan
    years = total_periods / periods_per_year
    if years <= 0.0:
        return np.nan
    return final_growth ** (1.0 / years) - 1.0


@njit("float64[:](float64[:])", nogil=True, fastmath=True)
def sequence_drawdown(nav_series):
    """
    计算净值序列的回撤序列。
    """
    n = nav_series.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.float64)
    result = np.empty(n, dtype=np.float64)
    peak = nav_series[0]
    for i in range(n):
        if nav_series[i] > peak:
            peak = nav_series[i]
        if peak <= 0.0:
            result[i] = np.nan
        else:
            result[i] = (nav_series[i] - peak) / peak
    return result


@njit("float64[:](float64[:], float64[:])", nogil=True, fastmath=True)
def sequence_excess_return(asset_returns, benchmark_returns):
    """
    计算资产相对于基准的超额收益率序列。
    """
    if asset_returns.shape[0] != benchmark_returns.shape[0]:
        raise ValueError("序列长度不一致，无法计算超额收益率")
    n = asset_returns.shape[0]
    result = np.empty(n, dtype=np.float64)
    for i in range(n):
        result[i] = asset_returns[i] - benchmark_returns[i]
    return result

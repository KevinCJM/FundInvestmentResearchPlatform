# -*- encoding: utf-8 -*-
"""
@File: A04_CalFunc.py
@Modify Time: 2025/7/17 14:10       
@Author: Kevin-Chen
@Descriptions: 
"""
from numba import njit
import numpy as np
import pandas as pd


def cal_total_return(return_array: np.ndarray, *args, **kwargs):
    """
    计算累计收益率(总收益率)。
    """
    if return_array is None or len(return_array) == 0:
        return np.nan
    return np.prod(1 + return_array) - 1


@njit(inline='always')
def ind_total_return(
        data, s, e,
        # ---- 以下为标准签名要求的占位参数，本函数中未使用 ----
        end_dates_int, trading_days_int,
        ts_date_int,
        day_type_code, day_count,
        annual_factor, ann_rf
):
    """
    (Numba JIT版) 计算累计收益率。
    - 遵循 prange 框架的标准函数签名。
    - data: (n_src, T) 的二维数组，这里只使用 data[0] (pct_array)。
    - s, e: 当前产品在 data 数组中的起止索引。
    """
    # 1. 检查分段是否有效 (e < s 表示空区间)
    if e < s:
        # 必须返回一个单元素元组
        return (np.nan,)
    # 2. 切片获取当前分段的收益率数据: 假设 pct_array 是源数据(source)中的第一个
    return_segment = data[0, s:e + 1]
    # 3. 计算累计乘积并减1，得到总收益率 (np.prod 在 numba 中受支持)
    total_return = np.prod(1.0 + return_segment) - 1.0
    # 4. 将结果放入一个单元素元组中返回
    return (total_return,)


def cal_mean_return(return_array: np.ndarray, *args, **kwargs):
    """
    计算平均收益率。
    """
    if return_array is None or len(return_array) == 0:
        return np.nan
    return np.mean(return_array)


@njit(inline='always')
def ind_mean_return(
        data, s, e,
        # ---- 以下为标准签名要求的占位参数 ----
        end_dates_int, trading_days_int, ts_date_int,
        day_type_code, day_count, annual_factor, ann_rf
):
    """(Numba JIT版) 计算平均收益率。"""
    if e < s:
        return (np.nan,)
    return_segment = data[0, s:e + 1]
    return (np.mean(return_segment),)


def calc_annualized_return(cache: dict, day_count: np.ndarray, annual_factor: int = 365, *args, **kwargs):
    """
    计算年化收益率。
    """
    tr = cache['TotalReturn']
    # 使用 errstate 上下文管理器来抑制无效值警告
    with np.errstate(invalid='ignore', divide='ignore'):
        # 确保 day_count 不为零
        power = np.divide(annual_factor, day_count, out=np.full_like(day_count, np.nan, dtype=float),
                          where=day_count != 0)
        # (1+tr) 必须为正
        base = 1.0 + tr
        annualized = np.power(base, power, out=np.full_like(base, np.nan), where=base > 0) - 1.0
    return np.where(np.isfinite(annualized), annualized, np.nan)


def calc_bm_annualized_return(cache: dict, day_count: np.ndarray, annual_factor: int = 365, *args, **kwargs):
    """
    计算基准年化收益率。
    """
    tr = cache['TotalReturn_bm']
    with np.errstate(invalid='ignore', divide='ignore'):
        power = np.divide(annual_factor, day_count, out=np.full_like(day_count, np.nan, dtype=float),
                          where=day_count != 0)
        base = 1.0 + tr
        annualized = np.power(base, power, out=np.full_like(base, np.nan), where=base > 0) - 1.0
    return np.where(np.isfinite(annualized), annualized, np.nan)


def cal_relative_return(cache: dict, *args, **kwargs):
    return cache['TotalReturn'] - cache['TotalReturn_bm']


def cal_relative_ann_return(cache: dict, *args, **kwargs):
    return cache['AnnualReturn'] - cache['AnnualReturn_bm']


def cal_abs_odds(return_array: np.ndarray, day_count: int, *args, **kwargs):
    return np.sum(return_array > 0) / day_count


@njit(inline='always')
def ind_abs_odds(
        data, s, e,
        end_dates_int, trading_days_int, ts_date_int,
        day_type_code, day_count, annual_factor, ann_rf
):
    """(Numba JIT版) 计算绝对胜率 (收益率>0的天数占比)。"""
    # day_count 是从外部传入的该分段的总天数，用于分母
    if day_count == 0:
        return (np.nan,)

    if e < s:
        return (0.0,)

    return_segment = data[0, s:e + 1]
    positive_days = np.sum(return_segment > 0)

    return (positive_days / day_count,)


def cal_std(return_array: np.ndarray, *args, **kwargs):
    if return_array is None or len(return_array) < 2:
        return np.nan
    return np.std(return_array, ddof=1)


@njit(inline='always')
def ind_std(
        data, s, e,
        # ---- 以下为标准签名要求的占位参数 ----
        end_dates_int, trading_days_int, ts_date_int,
        day_type_code, day_count, annual_factor, ann_rf
):
    """(Numba JIT版) 计算样本标准差 (ddof=1)。"""
    n_pts = e - s + 1
    # 至少需要2个数据点来计算样本标准差
    if n_pts < 2:
        return (np.nan,)

    return_segment = data[0, s:e + 1]

    # 1. 计算均值
    mean_val = np.mean(return_segment)

    # 2. 计算离差平方和
    sum_sq_diff = 0.0
    for x in return_segment:
        sum_sq_diff += (x - mean_val) ** 2

    # 3. 除以 (N-1) 并开方
    sample_std = np.sqrt(sum_sq_diff / (n_pts - 1))

    return (sample_std,)


def cal_annualized_std(cache: dict, annual_factor: int = 252, *args, **kwargs):
    annualized_std = cache['prodStd'] * np.sqrt(annual_factor)
    return np.where(np.isfinite(annualized_std), annualized_std, np.nan)


def cal_downside_std_not_ann(return_array: dict, *args, **kwargs):
    # 确保有足够的数据进行计算 (至少需要2个数据点来计算样本标准差)
    if len(return_array['pct_array']) < 2:
        return np.nan
    # 1. 计算收益率与最小可接受收益率的差值
    diff = return_array['pct_array'] - return_array['rf_array']
    # 2. 仅保留下行部分的差值（即负数部分）, 其他部分（正数）置为0.
    downside_diff = np.clip(diff, np.NINF, 0)
    # 3. 计算下行差异的平方和.
    downside_sum_sq = np.sum(downside_diff ** 2)
    # 4. 确定样本数量 (N-1).
    n_minus_1 = len(return_array['pct_array']) - 1
    # 5. 计算下行方差, 然后开方得到下行标准差.
    downside_std = np.sqrt(downside_sum_sq / n_minus_1)

    return downside_std


@njit(inline='always')
def ind_downside_std(
        data, s, e,
        # ---- 以下为标准签名要求的占位参数 ----
        end_dates_int, trading_days_int, ts_date_int,
        day_type_code, day_count, annual_factor, ann_rf
):
    """
    (Numba JIT版) 计算下行标准差 (非年化)。

    参数:
    - data: 包含数据的数组，其中第一行是百分比变化数据，第二行是无风险利率数据。
    - s: 开始索引。
    - e: 结束索引。
    - end_dates_int, trading_days_int, ts_date_int, day_type_code, day_count, annual_factor, ann_rf:
      标准签名占位参数，用于保持函数接口一致性，本函数中未使用这些参数。

    返回:
    - 下行标准差（非年化），如果输入数据长度小于2，则返回np.nan。
    """
    # 检查数据长度是否满足计算要求
    if (e - s + 1) < 2:
        return (np.nan,)

    # 提取百分比变化和无风险利率数据段
    pct_segment = data[0, s:e + 1]
    rf_segment = data[1, s:e + 1]

    # 计算每日收益与无风险利率的差值
    diff = pct_segment - rf_segment
    # 仅保留负差值，正差值设为0
    downside_diff = np.where(diff < 0, diff, 0.0)

    # 计算负差值的平方和
    downside_sum_sq = np.sum(downside_diff ** 2)
    # 计算自由度调整后的样本数量
    n_minus_1 = len(pct_segment) - 1

    # 返回下行标准差（非年化）
    return (np.sqrt(downside_sum_sq / n_minus_1),)


def cal_downside_std(cache: dict, annual_factor: int = 252, *args, **kwargs) -> np.array:
    annualized_std = cache['DownsideStd_not_ann'] * np.sqrt(annual_factor)
    return np.where(np.isfinite(annualized_std), annualized_std, np.nan)


def cal_mdd_and_recovery(
        return_array: np.ndarray,  # 收益率数组 (如每日收益率)
        return_days: np.ndarray,  # 收益率对应的时间日期数组
        trading_days: np.ndarray,  # 所有交易日的数组 (用于交易日计数)
        day_type: str = "n_pts",  # 修复时长的计算方式，支持：n_pts(数据点数), n_days(自然日), t_days(交易日)
        *args, **kwargs
):
    """
    计算最大回撤（Maximum Drawdown）和回撤修复时长（Recovery Time）。

    返回：
        包含以下两个值的字典：
        - 'prodMddr': 最大回撤值（浮点数）
        - 'MddrRecoverTime': 回撤修复所需天数（整数）

    约定：
        - 最大回撤 == 0          → 无回撤
        - 修复时长 == 0          → 无回撤
        - 修复时长 == 1_000_000  → 回撤尚未修复
    """
    T = return_array.size
    if T == 0:
        raise ValueError("return_array 不能为空")

    # ---------- 1) 净值曲线 ----------
    nav = np.empty(T + 1, dtype=return_array.dtype)
    nav[0] = 1.0
    nav[1:] = np.multiply.accumulate(1.0 + return_array)

    # ---------- 2) 最大回撤 ----------
    running_max = np.maximum.accumulate(nav)
    drawdown = 1.0 - nav / running_max  # 0 ≤ drawdown ≤ 1
    trough_idx = int(np.argmax(drawdown))  # drawdown 最大的位置
    mdd = drawdown[trough_idx]  # 正值

    if mdd == 0:  # 净值一路上升
        return {'prodMddr': 0.0, 'MddrRecoverTime': 0}

    # ---------- 3) 峰值索引 ----------
    peak_nav = running_max[trough_idx]

    # ---------- 4) 回撤修复点 ----------
    rec_mask = nav[trough_idx:] >= peak_nav
    if rec_mask.any():
        recovery_idx = trough_idx + int(np.argmax(rec_mask))
    else:  # 未修复
        return {'prodMddr': float(mdd), 'MddrRecoverTime': 1_000_000}

    # ---------- 5) nav 索引 → 真实日期 的统一映射 ----------
    first_day = return_days[0]

    # 基点日期：比首个收益率日期“早一天”。
    base_date = first_day - np.timedelta64(1, 'D')

    def nav_idx_to_date(idx: int):
        return base_date if idx == 0 else return_days[idx - 1]

    trough_date = nav_idx_to_date(trough_idx)
    recovery_date = nav_idx_to_date(recovery_idx)

    # ---------- 6) 计算修复时长 ----------
    if day_type == "n_pts":
        recovery = recovery_idx - trough_idx

    elif day_type == "n_days":
        delta = recovery_date - trough_date
        recovery = int(delta.astype('timedelta64[D]') //
                       np.timedelta64(1, 'D'))

    elif day_type == "t_days":
        start = np.searchsorted(trading_days, trough_date, side='left')
        end = np.searchsorted(trading_days, recovery_date, side='right') - 1
        recovery = end - start
    else:
        raise ValueError("day_type 必须是 {'n_pts','n_days','t_days'}")

    return {'prodMddr': float(mdd), 'MddrRecoverTime': int(recovery)}


@njit(inline='always')
def ind_mdd_and_recovery(
        data, s, e,
        end_dates_int, trading_days_int, ts_date_int,
        day_type_code, day_count, annual_factor, ann_rf
):
    """
    (Numba JIT版) 计算最大回撤和回撤修复时长。
    """
    # Step 0: Check for a valid segment
    if e < s:
        return (np.nan, np.nan)

    # Step 1: Create a local data slice and calculate the NAV curve
    return_segment = data[0, s:e + 1]
    T = return_segment.shape[0]
    if T == 0:
        return (0.0, 0.0)

    nav = np.empty(T + 1, dtype=np.float64)
    nav[0] = 1.0
    for i in range(T):
        nav[i + 1] = nav[i] * (1.0 + return_segment[i])

    # Step 2: Calculate Maximum Drawdown (MDD)
    # --- FIX: Manually implement np.maximum.accumulate ---
    # Numba does not support np.maximum.accumulate, so we implement it manually.
    running_max = np.empty_like(nav)
    if len(nav) > 0:
        running_max[0] = nav[0]
        for i in range(1, len(nav)):
            running_max[i] = max(running_max[i - 1], nav[i])
    # --- End of Fix ---

    drawdown = 1.0 - nav / running_max

    trough_idx_local = np.argmax(drawdown)
    mdd = drawdown[trough_idx_local]

    if mdd < 1e-9:
        return (0.0, 0.0)

    # Step 3: Find the recovery point
    peak_nav = running_max[trough_idx_local]
    rec_mask = nav[trough_idx_local:] >= peak_nav

    if np.any(rec_mask):
        recovery_offset = np.argmax(rec_mask)
        recovery_idx_local = trough_idx_local + recovery_offset
    else:
        return (mdd, 1000000.0)

    # Step 4: Calculate the recovery duration using integer dates
    trough_date_int = end_dates_int[s + trough_idx_local - 1] if trough_idx_local > 0 else ts_date_int
    recovery_date_int = end_dates_int[s + recovery_idx_local - 1] if recovery_idx_local > 0 else ts_date_int

    if day_type_code == 1:  # n_pts
        recovery_time = float(recovery_idx_local - trough_idx_local)
    elif day_type_code == 2:  # n_days
        recovery_time = float(recovery_date_int - trough_date_int)
    else:  # t_days
        start_pos = np.searchsorted(trading_days_int, trough_date_int, side='left')
        end_pos = np.searchsorted(trading_days_int, recovery_date_int, side='left')
        recovery_time = float(end_pos - start_pos + 1)

    # Step 5: Return the final tuple
    return (mdd, recovery_time)


def cal_info_ratio(cache: dict, *args, **kwargs) -> np.array:
    tracking_error = cache['AnnualActiveRisk']
    numerator = cache['RelaReAnn']
    return np.divide(numerator, tracking_error, out=np.full_like(numerator, np.nan), where=tracking_error > 1e-9)


def cal_beta(return_array: dict, *args, **kwargs) -> np.array:
    if len(return_array['pct_array']) < 2:
        return np.nan
    cov_matrix = np.cov(return_array['pct_array'], return_array['bench_array'])
    benchmark_variance = cov_matrix[1, 1]
    # 如果基准没有波动（方差接近0），则Beta无意义
    if np.abs(benchmark_variance) < 1e-9:
        return np.nan
    return cov_matrix[0, 1] / benchmark_variance


@njit(inline='always')
def ind_beta(
        data, s, e,
        # ---- 以下为标准签名要求的占位参数 ----
        end_dates_int, trading_days_int, ts_date_int,
        day_type_code, day_count, annual_factor, ann_rf
):
    """
    (Numba JIT版) 计算 Beta 值。

    参数:
    - data: 包含股票和基准数据的二维数组。
    - s: 计算开始的索引。
    - e: 计算结束的索引。
    - 其余参数为占位参数，用于匹配标准函数签名。

    返回:
    - Beta 值，表示股票与基准的相关性。
    """
    # 检查数据长度是否满足计算要求
    if (e - s + 1) < 2:
        return (np.nan,)

    # 提取计算区间内的股票和基准数据
    pct_segment = data[0, s:e + 1]
    bench_segment = data[1, s:e + 1]

    # 计算股票和基准的协方差矩阵
    cov_matrix = np.cov(pct_segment, bench_segment)
    # 提取基准的方差
    benchmark_variance = cov_matrix[1, 1]

    # 如果基准方差接近于零，无法计算 Beta，返回 NaN
    if np.abs(benchmark_variance) < 1e-12:
        return (np.nan,)

    # 计算 Beta 值并返回
    return (cov_matrix[0, 1] / benchmark_variance,)


def cal_alpha_not_ann(cache: dict, *args, **kwargs) -> np.array:
    return cache['MeanReturn'] - (cache['beta'] * cache['MeanReturn_bm'])


def cal_alpha(cache: dict, day_count: np.ndarray, annual_factor: int = 252, *args, **kwargs) -> np.array:
    tr = cache['alpha_not_ann']
    with np.errstate(invalid='ignore', divide='ignore'):
        power = np.divide(annual_factor, day_count, out=np.full_like(day_count, np.nan, dtype=float),
                          where=day_count != 0)
        base = 1.0 + tr
        annualized = np.power(base, power, out=np.full_like(base, np.nan), where=base > 0) - 1.0
    return np.where(np.isfinite(annualized), annualized, np.nan)


def cal_sharpe_ratio_not_ann(cache: dict, *args, **kwargs) -> np.array:
    prod_std = cache['prodStd']
    numerator = cache['TotalReturn'] - cache['TotalReturn_rf']
    return np.divide(numerator, prod_std, out=np.full_like(numerator, np.nan), where=prod_std > 1e-9)


def cal_sharpe_ratio(cache: dict, ann_rf: float = 0.015, *args, **kwargs) -> np.array:
    annual_return = cache['AnnualReturn']
    annual_std = cache['prodAnnualStd']
    numerator = annual_return - ann_rf
    return np.divide(numerator, annual_std, out=np.full_like(numerator, np.nan), where=annual_std > 1e-9)


def cal_calmar_ratio_not_ann(cache: dict, *args, **kwargs) -> np.array:
    max_drawdown = cache['prodMddr']
    numerator = cache['TotalReturn']
    return np.divide(numerator, max_drawdown, out=np.full_like(numerator, np.nan), where=max_drawdown > 1e-9)


def cal_calmar_ratio(cache: dict, *args, **kwargs) -> np.array:
    max_drawdown = cache['prodMddr']
    numerator = cache['AnnualReturn']
    return np.divide(numerator, max_drawdown, out=np.full_like(numerator, np.nan), where=max_drawdown > 1e-9)


def cal_tracking_error_not_ann(return_array: dict, *args, **kwargs) -> np.array:
    excess_rtn = return_array['pct_array'] - return_array['bench_array']
    if len(excess_rtn) < 2:
        return np.nan
    return np.std(excess_rtn, ddof=1)


@njit(inline='always')
def ind_tracking_error(
        data, s, e,
        # --- 标准函数签名的占位参数 ---
        end_dates_int, trading_days_int, ts_date_int,
        day_type_code, day_count, annual_factor, ann_rf
):
    """
    (Numba JIT版) 计算跟踪误差（非年化）。
    该函数手动计算样本标准差（ddof=1），用于衡量基金收益与基准收益之间的偏离程度。

    参数:
        data: 二维数组，包含基金收益数据和基准收益数据。
            - data[0] 是基金收益序列（pct_array）
            - data[1] 是基准收益序列（bench_array）
        s: 数据段的起始索引
        e: 数据段的结束索引
        end_dates_int: 结束日期数组（用于计算持续时间，此处未使用）
        trading_days_int: 交易日数组（此处未使用）
        ts_date_int: 时间戳日期（占位参数，未使用）
        day_type_code: 日期类型编码（占位参数，未使用）
        day_count: 日数（占位参数，未使用）
        annual_factor: 年化因子（占位参数，未使用）
        ann_rf: 年化无风险利率（占位参数，未使用）

    返回:
        跟踪误差（非年化）的单元素元组，若数据不足则返回 (np.nan,)
    """
    # ---------- 1. 获取数据段长度 ----------
    # 计算当前数据段的点数
    n_pts = e - s + 1

    # 至少需要 2 个数据点来计算样本标准差
    if n_pts < 2:
        return (np.nan,)

    # ---------- 2. 提取基金收益和基准收益数据段 ----------
    # 提取基金收益数据段
    pct_segment = data[0, s:e + 1]
    # 提取基准收益数据段
    bench_segment = data[1, s:e + 1]

    # ---------- 3. 计算超额收益 ----------
    # 超额收益 = 基金收益 - 基准收益
    excess_rtn = pct_segment - bench_segment

    # ---------- 4. 手动计算样本标准差 ----------
    # 1. 计算超额收益的均值
    mean_val = np.mean(excess_rtn)

    # 2. 计算离差平方和
    sum_sq_diff = np.sum((excess_rtn - mean_val) ** 2)

    # 3. 计算样本标准差 (ddof=1)
    tracking_err = np.sqrt(sum_sq_diff / (n_pts - 1))

    # ---------- 5. 返回结果 ----------
    # 返回计算得到的跟踪误差
    return (tracking_err,)


def cal_tracking_error(cache: dict, annual_factor: int = 252, *args, **kwargs) -> np.array:
    return cache['ActiveRisk'] * np.sqrt(annual_factor)


def cal_info_ratio_not_ann(cache: dict, *args, **kwargs) -> np.array:
    tracking_error = cache['ActiveRisk']
    numerator = cache['RelaRe']
    return np.divide(numerator, tracking_error, out=np.full_like(numerator, np.nan), where=tracking_error > 1e-9)


def calculate_single_fund_streak_numpy(
        return_array: np.ndarray,
        return_days: np.ndarray,
        trading_days: set,
        day_type: str = 't_days',
        *args, **kwargs
) -> dict:
    """
    (纯 NumPy 高性能版) 计算单只基金的最大连续上涨周期。

    本函数仅使用 NumPy 的向量化操作，不依赖 Numba 或 Pandas 的核心计算。
    它假定所有输入数据都不包含 NaN。

    参数 (Args):
        return_array (np.ndarray): 一维 numpy 数组，基金收益率。
        return_dates (np.ndarray): 一维 numpy 数组，对应的日期。
        trading_days (set):       所有交易日的集合。
        day_type (str, optional): 周期长度的计算方式 ('n_pts', 'n_days', 't_days')。

    返回 (Returns):
        dict: 包含连涨指标的字典 {'r', 'p', 's', 'l'}。
    """
    # --- 1. 准备工作和边缘情况处理 ---
    is_positive = return_array > 0
    if not np.any(is_positive):
        # 如果没有任何正收益，直接返回
        return {'r': 0.0, 'p': 0, 's': np.nan, 'l': np.nan}

    # --- 2. 计算总累计增长和每个独立周期的增长 ---
    growth_factors = return_array + 1.0
    total_cum_prod = np.cumprod(growth_factors)

    # --- 3. 用纯 NumPy 实现前向填充(ffill)来构造除数数组 ---
    # a. 创建一个在重置点（非正收益日）有值，在连涨期为 NaN 的数组
    reset_points = np.where(~is_positive, total_cum_prod, np.nan)

    # b. 这是实现 ffill 的 NumPy 配方
    mask = np.isnan(reset_points)
    idx = np.where(~mask, np.arange(len(mask)), 0)
    np.maximum.accumulate(idx, out=idx)  # 核心：向前传播最后一个有效索引
    streak_divisor = reset_points[idx]

    # c. 处理开头就是连涨的情况（此时 divisor 会是 nan），填充为 1
    streak_divisor = np.nan_to_num(streak_divisor, nan=1.0)

    # --- 4. 计算每个独立周期的累计乘积并找到最佳者 ---
    streak_prod = total_cum_prod / streak_divisor
    # 将非正收益日的结果设置为1，不影响取最大值
    streak_prod[~is_positive] = 1.0

    # 寻找最后一个最大值：翻转数组 -> argmax -> 转换索引
    reversed_streak_prod = streak_prod[::-1]
    argmax_in_reversed = np.argmax(reversed_streak_prod)
    end_index = len(streak_prod) - 1 - argmax_in_reversed
    max_growth = streak_prod[end_index]

    # --- 5. 向量化地寻找开始索引 ---
    # 开始索引 = 最佳周期结束点之前，最后一个非正收益日索引 + 1
    non_positive_indices_before_end = np.where(~is_positive[:end_index + 1])[0]

    if non_positive_indices_before_end.size > 0:
        start_index = non_positive_indices_before_end[-1] + 1
    else:
        # 如果前面没有非正收益日，说明从头开始就是连涨
        start_index = 0

    # --- 6. 计算最终指标 ---
    max_return = max_growth - 1.0
    start_date = return_days[start_index]
    end_date = return_days[end_index]

    if day_type == 'n_pts':
        duration = end_index - start_index + 1
    elif day_type == 'n_days':
        duration = (end_date - start_date).astype('timedelta64[D]').astype(int) + 1
    elif day_type == 't_days':
        # 这里为了方便比较，还是转成 date 对象
        s_date_only = pd.Timestamp(start_date).date()
        e_date_only = pd.Timestamp(end_date).date()
        duration = len([d for d in trading_days if s_date_only <= d <= e_date_only])
    else:
        raise ValueError("day_type 参数必须是 'n_pts', 'n_days', 或 't_days' 中的一个。")

    # --- 7. 格式化并返回结果 ---
    # 使用纯 numpy 的方式格式化日期
    s_date_str = str(np.datetime_as_string(start_date, unit='D')).replace('-', '')
    l_date_str = str(np.datetime_as_string(end_date, unit='D')).replace('-', '')

    return {
        'r': max_return,
        'p': duration,
        's': s_date_str,
        'l': l_date_str
    }


@njit
def _find_best_streak_core(return_array: np.ndarray):
    """
    【Numba 加速核心】遍历数组以找到最大且最近的连续正收益周期的指标。
    """
    if return_array.shape[0] == 0:
        return 0.0, -1, -1

    max_cumulative_prod = 1.0
    best_start_index = -1
    best_end_index = -1
    current_cumulative_prod = 1.0
    current_start_index = -1

    for i in range(return_array.shape[0]):
        if return_array[i] > 0:
            if current_start_index == -1:
                current_start_index = i
                current_cumulative_prod = 1.0 + return_array[i]
            else:
                current_cumulative_prod *= (1.0 + return_array[i])

            # 如果当前收益大于或等于已记录的最大收益，则更新记录。这保证了在收益相同时，总会取后面（最近）的那个周期。
            if current_cumulative_prod >= max_cumulative_prod:
                max_cumulative_prod = current_cumulative_prod
                best_start_index = current_start_index
                best_end_index = i
        else:
            current_start_index = -1
            current_cumulative_prod = 1.0

    if best_start_index == -1:
        return 0.0, -1, -1

    return max_cumulative_prod - 1.0, best_start_index, best_end_index


def cal_largest_continue_raising(
        return_array: np.ndarray,
        return_days: np.ndarray,
        trading_days: set,
        day_type: str = 't_days',
        *args, **kwargs
) -> dict:
    """
    (Numba高性能版) 计算单只基金的最大连续上涨（盈利）周期。
    """
    max_return, start_index, end_index = _find_best_streak_core(return_array)

    if start_index == -1:
        return {'r': 0.0, 'p': 0, 's': np.nan, 'l': np.nan}

    start_date = return_days[start_index]
    end_date = return_days[end_index]

    if day_type == 'n_pts':
        duration = end_index - start_index + 1
    elif day_type == 'n_days':
        duration = (end_date - start_date).days + 1
    elif day_type == 't_days':
        s_date_only = pd.Timestamp(start_date).date()  # 同样建议转换
        e_date_only = pd.Timestamp(end_date).date()  # 同样建议转换
        duration = len([d for d in trading_days if s_date_only <= d <= e_date_only])
    else:
        raise ValueError("day_type 参数必须是 'n_pts', 'n_days', 或 't_days' 中的一个。")

    # 4. 格式化并返回最终结果
    return {
        'LargestContinueRaisingRate': max_return,
        'LargestContinueRaisingDays': duration,
        'LargestContinueRaisingStart': pd.Timestamp(start_date).strftime('%Y%m%d'),
        'LargestContinueRaisingEnd': pd.Timestamp(end_date).strftime('%Y%m%d')
    }


@njit(inline='always')
def ind_largest_continue_raising(
        data, s, e,
        end_dates_int, trading_days_int,  # ← 都是 int64(epoch‑days)
        ts_date_int,  # int64，占位不用
        day_type_code, day_count,
        annual_factor, ann_rf
):
    """
    计算在指定时间段内最大的连续正收益，及其持续天数和对应的开始和结束日期。

    参数:
    - data: 包含收益数据的数组。
    - s, e: 计算范围的起始和结束索引。
    - end_dates_int: 结束日期数组，用于计算持续天数。
    - trading_days_int: 交易日数组，当计算交易日类型时使用。
    - ts_date_int: 占位参数，未使用。
    - day_type_code: 天数计算方式的代码。
    - day_count, annual_factor, ann_rf: 占位参数，未使用。

    返回:
    - max_return: 最大连续正收益。
    - duration: 连续正收益的持续天数。
    - start_int: 连续正收益开始的日期（epoch天数）。
    - end_int: 连续正收益结束的日期（epoch天数）。
    """
    # ---------- 1. 找最大连续正收益段 ----------
    max_prod = 1.0
    best_s = best_e = -1
    cur_prod = 1.0
    cur_s = -1

    # 遍历数据，寻找最大连续正收益的起止点和收益值
    for i in range(s, e + 1):
        r = data[0, i]
        if r > 0.0:
            if cur_s == -1:
                cur_s = i
                cur_prod = 1.0 + r
            else:
                cur_prod *= 1.0 + r

            if cur_prod >= max_prod:
                max_prod = cur_prod
                best_s, best_e = cur_s, i
        else:
            cur_s = -1
            cur_prod = 1.0

    if best_s == -1:
        return 0.0, 0.0, np.nan, np.nan

    # ---------- 2. 持续天数 ----------
    # 根据不同的天数计算方式，计算最大连续正收益的持续天数
    if day_type_code == 1:  # n_pts
        duration = best_e - best_s + 1
    elif day_type_code == 2:  # n_days
        duration = int(end_dates_int[best_e] - end_dates_int[best_s]) + 1
    else:  # t_days
        s_day = end_dates_int[best_s]
        e_day = end_dates_int[best_e]
        cnt = 0
        for j in range(trading_days_int.shape[0]):
            d = trading_days_int[j]
            if d >= s_day and d <= e_day:
                cnt += 1
        duration = cnt

    # ---------- 3. 直接返回 int64 日期（已是 epoch‑days） ----------
    start_int = float(end_dates_int[best_s])
    end_int = float(end_dates_int[best_e])

    return max_prod - 1.0, start_int, end_int, float(duration)


@njit(inline='always')
def ind_largest_continue_falling(
        data, s, e,
        end_dates_int, trading_days_int,
        ts_date_int,
        day_type_code, day_count,
        annual_factor, ann_rf
):
    """
    (Numba JIT版) 计算在指定时间段内最大的连续亏损（最大连跌），
    及其持续天数和对应的开始和结束日期。
    """
    # 步骤0: 检查分段有效性
    if e < s:
        return (np.nan, np.nan, np.nan, np.nan)

    # ---------- 1. 找最大连续负收益段 ----------
    # 寻找最小的累计乘积，代表最大的亏损
    min_prod = 1.0
    best_s = best_e = -1

    cur_prod = 1.0
    cur_s = -1

    # 遍历数据，寻找最大连续亏损的起止点和亏损值
    for i in range(s, e + 1):
        r = data[0, i]
        # 【关键修改】寻找连续负收益
        if r < 0.0:
            if cur_s == -1:
                cur_s = i
                cur_prod = 1.0 + r
            else:
                cur_prod *= 1.0 + r

            # 【关键修改】寻找最小的累计乘积
            # `<=` 确保在亏损相同时，总会取后面（最近）的那个周期
            if cur_prod <= min_prod:
                min_prod = cur_prod
                best_s = cur_s
                best_e = i
        else:
            # 收益为正或零，则连跌中断
            cur_s = -1
            cur_prod = 1.0

    # 如果没有找到任何连跌（即净值一路上升或持平）
    if best_s == -1:
        return (0.0, np.nan, np.nan, 0.0)

    # ---------- 2. 持续天数 ----------
    # 这部分逻辑与最大连涨完全相同
    s_day = end_dates_int[best_s]
    e_day = end_dates_int[best_e]

    if day_type_code == 1:  # n_pts
        duration = float(best_e - best_s + 1)
    elif day_type_code == 2:  # n_days
        duration = float(e_day - s_day + 1)
    else:  # t_days
        start_pos = np.searchsorted(trading_days_int, s_day, side='left')
        end_pos = np.searchsorted(trading_days_int, e_day, side='right')
        duration = float(end_pos - start_pos)

    # ---------- 3. 直接返回 int64 日期（已是 epoch‑days） ----------
    start_int = float(s_day)
    end_int = float(e_day)

    # 【关键修改】min_prod - 1.0 会得到一个负值，代表亏损率
    return min_prod - 1.0, start_int, end_int, duration

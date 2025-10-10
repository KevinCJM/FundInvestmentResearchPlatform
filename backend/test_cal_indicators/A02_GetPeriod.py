# -*- encoding: utf-8 -*-
"""
@File: A02_GetPeriod.py
@Modify Time: 2025/7/16 14:02
@Author: Kevin-Chen
@Descriptions:
"""
import time
import numpy as np
import pandas as pd
from datetime import datetime


# ------------------------------------------------------------------
# 日期分解 / 合成 / 位移 工具
# ------------------------------------------------------------------

def _is_leap_year(y: np.ndarray) -> np.ndarray:
    """
    判断给定的年份是否为闰年。

    闰年定义为能被4整除但不能被100整除，或者能被400整除的年份。

    参数:
    y: np.ndarray - 一个包含年份的NumPy数组。

    返回:
    np.ndarray - 一个布尔类型的NumPy数组，指示每个对应年份是否为闰年。
    """
    # 使用位运算符&和|来同时处理整个NumPy数组的年份判断
    return ((y % 4 == 0) & ((y % 100 != 0) | (y % 400 == 0)))


def _month_length_vec(y: np.ndarray, m: np.ndarray) -> np.ndarray:
    """
    生成一个数组，对应给定年份和月份的天数。

    参数:
    y: np.ndarray - 年份数组。
    m: np.ndarray - 月份数组。

    返回:
    np.ndarray - 月份天数数组。
    """
    # 初始化一个全为31的数组，长度与月份数组m相同
    ml = np.full_like(m, 31, dtype=np.int64)
    # 将4月、6月、9月、11月的天数设置为30天
    ml[(m == 4) | (m == 6) | (m == 9) | (m == 11)] = 30
    # 找出所有二月的记录
    feb = (m == 2)
    # 如果存在二月的记录
    if np.any(feb):
        # 根据是否为闰年，将二月的天数设置为29天或28天
        ml[feb] = np.where(_is_leap_year(y[feb]), 29, 28)
    # 返回月份天数数组
    return ml


def _dt64D_to_ymd(d: np.ndarray):
    """
    将numpy的datetime64对象转换为年、月、日格式。

    参数:
    d: np.ndarray - 包含datetime64对象的numpy数组。

    返回值:
    元组: (年, 月, 日) - 分别包含转换后的年、月、日信息的numpy数组。
    """
    # 将输入的datetime64对象转换为天精度
    d = d.astype('datetime64[D]')
    # 计算年份，通过将日期转换为年精度并转换为整数，然后加上1970年（Unix纪元）
    y = d.astype('datetime64[Y]').astype(int) + 1970
    # 计算月份，通过将日期转换为月精度并取模12，然后加上1以得到1-12的月份
    m = (d.astype('datetime64[M]').astype(int) % 12) + 1
    # 计算日期，通过将日期与当月的第一天相减，然后加上1以得到日期
    day = (d - d.astype('datetime64[M]')).astype(int) + 1
    # 返回年、月、日，确保它们都是64位整数类型
    return y.astype(np.int64), m.astype(np.int64), day.astype(np.int64)


def _ymd_to_dt64D(y, m, day):
    """
    将年月日转换为numpy的datetime64类型。

    此函数接收年(y)、月(m)和日(day)三个参数，并将它们转换为对应的numpy的datetime64对象，
    其中日期单位为天('D')。转换过程利用了numpy的timedelta64类型进行时间单位的换算。

    参数:
    y: 年份，例如2021。
    m: 月份，1到12之间的数字。
    day: 日，月份中的第几天。

    返回值:
    返回转换后的numpy的datetime64对象，日期单位为天。
    """
    # 计算从1970年1月起的月数偏移量
    ym = (y - 1970) * 12 + (m - 1)
    # 将月数偏移量转换为timedelta64[M]类型，并从1970年1月开始计算日期
    base_m = ym.astype('timedelta64[M]') + np.datetime64('1970-01')
    # 返回从base_m日期开始，加上日数偏移量，得到最终的日期
    return base_m + (day - 1).astype('timedelta64[D]')


def _shift_months(d: np.ndarray, n: int) -> np.ndarray:
    """
    将输入的日期数组中的每个日期向前或向后移动指定的月数。

    参数:
    d (np.ndarray): 一个包含日期的 NumPy 数组，元素类型为 datetime64[D]。
    n (int): 要移动的月数。如果为正，则日期向后移动；如果为负，则日期向前移动。

    返回:
    np.ndarray: 移动后的日期数组，保持与输入数组相同的形状。
    """
    # 确保输入的日期数组是日粒度（datetime64[D]）格式，以保证后续操作的一致性
    d = d.astype('datetime64[D]')

    # 将日期数组分解为年、月、日三个部分，以便进行月份的加减运算
    y, m, day = _dt64D_to_ymd(d)

    # 计算总月数：将年份转换为月份，并加上当前月份减1（因为月份从1开始）
    total_m = (y * 12 + (m - 1)) - n

    # 重新计算新的年份和月份：
    # 新年份 = 总月数 // 12 （整除得到年份）
    # 新月份 = 总月数 % 12 + 1 （取余得到月份，范围是1~12）
    y2 = total_m // 12
    m2 = (total_m % 12) + 1

    # 获取新月份的天数，以确保调整后的日期不会超出该月的最大天数
    dim = _month_length_vec(y2, m2)

    # 确保调整后的日期不超过该月的天数：
    # 如果原日期大于该月最大天数，则使用该月最后一天的日期
    day2 = np.minimum(day, dim)

    # 将调整后的年、月、日组合回 datetime64[D] 类型的日期数组并返回
    return _ymd_to_dt64D(y2, m2, day2)


def _shift_years(d: np.ndarray, n: int) -> np.ndarray:
    """
    将日期数组中的每个日期减去n年后，返回新的日期数组。

    该函数首先将输入的日期数组转换为datetime64[D]类型，以确保后续操作的正确性。
    然后，它将每个日期分解为年、月、日组件，并计算减去n年后的年份。
    为了处理闰年和月份天数差异的问题，函数会计算调整年份后每个月的天数，
    并确保调整后的日期不会超出该月的天数范围。

    参数:
    d - 输入的日期数组，类型为np.ndarray。
    n - 需要减去的年数，类型为整数。

    返回:
    返回一个新的日期数组，其中每个日期都比输入日期减少了n年。
    """
    # 将输入日期数组转换为datetime64[D]类型，确保日期处理的一致性和准确性
    d = d.astype('datetime64[D]')

    # 将日期分解为年、月、日组件，以便进行年份的减法操作
    y, m, day = _dt64D_to_ymd(d)

    # 计算减去n年后的年份
    y2 = y - n

    # 计算调整年份后每个月的天数，以处理闰年和不同月份天数差异的问题
    dim = _month_length_vec(y2, m)

    # 确保调整后的日期不会超出该月的天数范围
    day2 = np.minimum(day, dim)

    # 将调整后的年、月、日组件重新组合成日期数组，并返回
    return _ymd_to_dt64D(y2, m, day2)


def _month_start(d: np.ndarray) -> np.ndarray:
    """
    获取给定日期数组中每个日期所在月份的第一天。

    该函数的目的是为了处理日期数组，返回一个新数组，其中每个元素都是输入日期所在月份的第一天。

    参数:
    d: np.ndarray - 一个numpy数组，包含日期（numpy的datetime64对象）。

    返回值:
    np.ndarray - 一个与输入数组长度相同的numpy数组，包含每个输入日期所在月份的第一天的日期。

    过程:
    1. 使用_dt64D_to_ymd函数将输入的日期数组转换为年、月、日三个独立的numpy数组。
    2. 由于目标是获取月份的第一天，因此将日设置为1，这里通过创建一个与月份数组形状相同的全1数组实现。
    3. 使用_ymd_to_dt64D函数将年、月和新设置的日数组转换回numpy的datetime64对象数组。
    """
    # 将日期数组转换为年、月、日三个独立的数组
    y, m, _ = _dt64D_to_ymd(d)
    # 将每个月的日期设置为第一天，并将这些日期转换回datetime64对象数组
    return _ymd_to_dt64D(y, m, np.ones_like(m))


def _month_end(d: np.ndarray) -> np.ndarray:
    """
    计算给定日期数组中每个日期所在月份的最后一天。

    该函数的目的是将输入的日期数组中的每个日期映射到其所在月份的最后一天。
    它通过首先提取每个日期的年份和月份，然后获取该年份和月份的天数，
    最后将年份、月份和天数组合成新的日期来实现这一功能。

    参数:
    d: np.ndarray - 一个numpy数组，包含日期（datetime64.D）。

    返回值:
    np.ndarray - 一个numpy数组，包含输入日期所在月份的最后一天的日期。
    """
    # 将日期数组转换为年、月、日的元组形式
    y, m, _ = _dt64D_to_ymd(d)
    # 获取每个日期所在月份的天数
    dim = _month_length_vec(y, m)
    # 将年、月和月份的天数转换回日期数组，即得到月份的最后一天
    return _ymd_to_dt64D(y, m, dim)


# 分组聚合工具：假设 g 在 [0, G), 利用排序后 reduceat，一次排序可同时得 min/max
def _group_bounds(g_sorted: np.ndarray):
    """
    返回每组起始位置数组

    本函数旨在处理已排序的数组，通过识别不同元素之间的边界，来确定每组元素的起始位置。
    这对于需要对数据进行分组处理或分析的场景特别有用。

    参数:
    g_sorted: np.ndarray - 一个已排序的numpy数组，包含可能重复的元素。元素类型可以是任意的，
              只要它们在排序和比较时保持一致性。

    返回值:
    np.ndarray - 一个包含每组元素起始位置的数组。这些位置索引帮助确定在原始数组中每组元素的边界。
                 通过这些索引，可以轻松地访问和处理每个分组。
    """
    # 通过计算差分来找到每组的起始位置。差分结果中的非零元素表明了元素值的变化，
    # 通过在其索引上加1，我们得到了下一组的起始位置。在结果数组的开始处添加0，
    # 以表示第一组的起始位置。
    return np.r_[0, np.nonzero(np.diff(g_sorted))[0] + 1]


def _group_reduce_minmax_dt(dates_sorted: np.ndarray, grp_starts: np.ndarray):
    ords = dates_sorted.view('int64')
    mins = np.minimum.reduceat(ords, grp_starts)
    maxs = np.maximum.reduceat(ords, grp_starts)
    return mins, maxs  # int64 day ord


def _count_fridays_in_range(ts: np.ndarray, te: np.ndarray) -> np.ndarray:
    """高效地计算一个或多个[开始, 结束]日期区间内周五的数量。"""
    ts_ord = ts.astype('float64')
    te_ord = te.astype('float64')
    # 1970-01-01 是周四。因此，第一个周五是纪元的第1天（0-索引）。
    # 计算到 te 为止的周五数量
    count_te = np.floor((te_ord - 1) / 7) + 1
    # 计算 ts 之前的周五数量
    count_ts_before = np.floor(((ts_ord - 1) - 1) / 7) + 1
    return (count_te - count_ts_before).astype(np.int64)


def _count_month_ends_in_range(ts: np.ndarray, te: np.ndarray) -> np.ndarray:
    """高效地计算一个或多个[开始, 结束]日期区间内月末日期的数量。"""
    ts_y, ts_m, _ = _dt64D_to_ymd(ts)
    te_y, te_m, _ = _dt64D_to_ymd(te)

    # 计算两个月份开始之间的完整月份差异
    month_diff = (te_y - ts_y) * 12 + (te_m - ts_m)

    # 检查 te 本身是否是月末
    is_te_month_end = (te == _month_end(te))

    # 最终计数是月份差异 + te是否为月末的修正
    return month_diff + is_te_month_end.astype(np.int64)


# 区间截取主函数
def get_period_segments_np(
        finpro_codes: np.ndarray,
        end_dates: np.ndarray,
        period_code: str,
        from_today: bool = False,
        row_index=None,
        today=None,
):
    """
    纯 NumPy：按 period_code 为每个产品生成理论日期区间，返回区间内实际数据的起止行索引等统计。

    参数
    ----
    finpro_codes : (N,) 一维数组，产品代码（str/obj/int均可）。
    end_dates    : (N,) 一维数组，dtype 可为 datetime64[...]；内部统一到 [D]。
    period_code  : 'CC','CY','Yk','Mk','nW','nM','nY' 等（与 pandas 版本一致）。
    from_today   : 仅对滚动区间 (nW/nM/nY) 有效。True=统一用 today 为区间尾；False=每产品尾=其数据最大日期。
    row_index    : (N,) 行索引（输出 start/end 即来自此数组）。缺省→np.arange(N)。
    today        : 覆盖当前日期（测试/回放用）。缺省→np.datetime64('today','D')。

    返回
    ----
    start_idx    : (G,) float64，区间内最早行索引；无数据→NaN。
    end_idx      : (G,) float64，区间内最晚行索引；无数据→NaN。
    natural_days : (G,) float64，理论区间自然日长度 (含首尾)；CC 用实际 min/max；始终返回数值。
    data_points  : (G,) float64，区间内实际数据行数；无数据→NaN。
    ts           : (G,) datetime64[D]，理论区间开始日期。
    te           : (G,) datetime64[D]，理论区间结束日期。

    注：输出数组顺序与 `all_products = finpro_codes_unique_in_first_occurrence_order` 对齐。
        你可通过调用 `np.unique(finpro_codes, return_index=True)` 后按 return_index 排序得到产品序列。
    """
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] 正在截取{period_code}区间数据...")
    # ---------- 输入检查 ----------
    finpro_codes = np.asarray(finpro_codes)
    end_dates = np.asarray(end_dates)
    if finpro_codes.ndim != 1 or end_dates.ndim != 1:
        raise ValueError("finpro_codes 与 end_dates 必须为一维。")
    if finpro_codes.shape[0] != end_dates.shape[0]:
        raise ValueError("finpro_codes 与 end_dates 长度不一致。")
    N = finpro_codes.shape[0]

    if row_index is None:
        row_index = np.arange(N, dtype=np.int64)
    else:
        row_index = np.asarray(row_index)
        if row_index.shape != (N,):
            raise ValueError("row_index 长度必须与 finpro_codes 相同。")
        if not np.issubdtype(row_index.dtype, np.integer):
            row_index = row_index.astype(np.int64, copy=False)

    # 日期 -> 日粒度
    end_dates = end_dates.astype('datetime64[D]')

    # today
    if today is None:
        today = np.datetime64('today', 'D')
    else:
        today = np.datetime64(today, 'D')  # ensure unit

    # ---------- 产品唯一值（保持首次出现顺序） ----------
    uniq, first_pos = np.unique(finpro_codes, return_index=True)
    order_u = np.argsort(first_pos, kind='mergesort')
    all_products = uniq[order_u]

    # 构建 mapping→group id
    # 用 dict 因产品 G << N 通常可接受；如需极致性能可 factorize
    mapping = {k: i for i, k in enumerate(all_products.tolist())}
    g = np.fromiter((mapping[x] for x in finpro_codes), count=N, dtype=np.int64)
    G = all_products.shape[0]

    # ---------- 主排序 (按组) ----------
    order = np.argsort(g, kind='mergesort')
    g_s = g[order]
    dates_s = end_dates[order]
    rows_s = row_index[order]

    # 分组边界
    grp_starts = _group_bounds(g_s)

    # 组内实际 min / max 日期（int64 ord）
    grp_min_ord, grp_max_ord = _group_reduce_minmax_dt(dates_s, grp_starts)

    # 转 datetime64[D]
    epoch = np.datetime64('1970-01-01', 'D').view('int64')
    grp_min_dt = (grp_min_ord - epoch).astype('timedelta64[D]') + np.datetime64('1970-01-01', 'D')
    grp_max_dt = (grp_max_ord - epoch).astype('timedelta64[D]') + np.datetime64('1970-01-01', 'D')

    # ---------- 理论区间 ts / te ----------
    ts = np.empty(G, dtype='datetime64[D]')
    te = np.empty(G, dtype='datetime64[D]')

    is_rolling = period_code.endswith(('W', 'M', 'Y')) and period_code[:-1].isdigit()
    if period_code == 'CC':
        # 全历史：理论 = 实际
        ts[:] = grp_min_dt
        te[:] = grp_max_dt

    elif is_rolling:
        n = int(period_code[:-1])
        unit = period_code[-1]
        if from_today:
            te[:] = today
        else:
            te[:] = grp_max_dt  # 每产品独立尾

        if unit == 'W':
            ts = te - np.timedelta64(7 * n, 'D') + np.timedelta64(1, 'D')
        elif unit == 'M':
            ts = _shift_months(te, n) + np.timedelta64(1, 'D')
        else:  # 'Y'
            ts = _shift_years(te, n) + np.timedelta64(1, 'D')

    else:
        if period_code == 'CY':
            te[:] = today
            y_today, _, _ = _dt64D_to_ymd(np.array([today]))
            ts[:] = np.datetime64(f'{int(y_today[0]):04d}-01-01', 'D')
        elif period_code.startswith('Y') and period_code[1:].isdigit():
            n = int(period_code[1:])
            y_today, _, _ = _dt64D_to_ymd(np.array([today]))
            tgt_year = int(y_today[0]) - n
            ts[:] = np.datetime64(f'{tgt_year:04d}-01-01', 'D')
            te[:] = np.datetime64(f'{tgt_year:04d}-12-31', 'D')
        elif period_code.startswith('M') and period_code[1:].isdigit():
            n = int(period_code[1:])
            month_start_today = _month_start(np.array([today]))[0]
            ts_single = _shift_months(np.array([month_start_today]), n)[0]
            te_single = _month_end(np.array([ts_single]))[0]
            ts[:] = ts_single
            te[:] = te_single
        else:
            raise ValueError(f"不支持的期间代码: {period_code}")

    # 若上面 CY 情形忘了 te: 需填
    if period_code == 'CY':
        # 这里补全 te (上面已赋)
        pass

    # ---------- 构造 mask ----------
    end_ord = end_dates.view('int64')
    ts_ord = ts.view('int64')
    te_ord = te.view('int64')
    mask = (end_ord >= ts_ord[g]) & (end_ord <= te_ord[g])

    # ---------- data_points ----------
    data_points = np.bincount(g, weights=mask.astype(np.int64), minlength=G).astype(np.int64)

    # ---------- start / end 索引 ----------
    # 在排序后的数组上求组内 mask 最小/最大 row_index
    mask_s = mask[order]
    big = np.iinfo(rows_s.dtype).max
    sml = np.iinfo(rows_s.dtype).min
    cand_min = np.where(mask_s, rows_s, big)
    cand_max = np.where(mask_s, rows_s, sml)
    start_idx = np.minimum.reduceat(cand_min, grp_starts)
    end_idx = np.maximum.reduceat(cand_max, grp_starts)

    # ---------- 无数据组处理 ----------
    no_data = (data_points == 0)
    start_idx = start_idx.astype(np.float64)
    end_idx = end_idx.astype(np.float64)
    start_idx[no_data] = np.nan
    end_idx[no_data] = np.nan

    # ---------- natural_days ----------
    if period_code == 'CC':
        ndays = (grp_max_ord - grp_min_ord) + 1  # 实际跨度
    else:
        ndays = (te_ord - ts_ord) + 1  # 理论跨度
    ndays = ndays.astype(np.float64)

    # ---------- data_points 输出 ----------
    data_points_out = data_points.astype(np.float64)
    data_points_out[no_data] = np.nan
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] {period_code}区间数据截取完毕...")
    return start_idx, end_idx, ndays, data_points_out, ts, te


def get_period_segments_np_w(
        finpro_codes: np.ndarray,
        end_dates: np.ndarray,
        period_code: str,
        from_today: bool = False,
        row_index=None,
        today=None,
):
    """
    与 get_period_segments_np 的逻辑完全一致，但为代表周频的数据点计算理论区间。
    核心区别：返回理论区间内周五的数量（natural_weeks），而不是天数。

    参数
    ----
    finpro_codes : (N,) 一维数组，产品代码。
    end_dates    : (N,) 一维数组，dtype 必须为 datetime64[D]，但每个日期在语义上代表一个周的结束点。
    period_code  : 'CC','CY','Yk','Mk','nW','nM','nY' 等，逻辑与日频版本完全相同。
    from_today   : 仅对滚动区间有效。
    row_index    : (N,) 行索引。
    today        : 覆盖当前日期。

    返回
    ----
    start_idx      : (G,) float64，区间内最早行索引；无数据→NaN。
    end_idx        : (G,) float64，区间内最晚行索引；无数据→NaN。
    natural_weeks  : (G,) float64，理论区间内周五的数量；CC 用实际 min/max；始终返回数值。
    data_points    : (G,) float64，区间内实际数据行数；无数据→NaN。
    ts             : (G,) datetime64[D]，理论区间开始日期。
    te             : (G,) datetime64[D]，理论区间结束日期。
    """
    # 调用日频版本函数获取所有核心计算结果
    start_idx, end_idx, _, data_points_out, ts, te = get_period_segments_np(
        finpro_codes=finpro_codes,
        end_dates=end_dates,
        period_code=period_code,
        from_today=from_today,
        row_index=row_index,
        today=today
    )

    # 【核心区别】计算区间内周五的数量
    natural_weeks = _count_fridays_in_range(ts, te).astype(np.float64) - 1

    return start_idx, end_idx, natural_weeks, data_points_out, ts, te


def get_period_segments_np_m(
        finpro_codes: np.ndarray,
        end_dates: np.ndarray,
        period_code: str,
        from_today: bool = False,
        row_index=None,
        today=None,
):
    """
    与 get_period_segments_np 的逻辑完全一致，但为代表月频的数据点计算理论区间。
    核心区别：返回理论区间内月末日期的数量（natural_months），而不是天数。

    参数
    ----
    finpro_codes : (N,) 一维数组，产品代码。
    end_dates    : (N,) 一维数组，dtype 必须为 datetime64[D]，但每个日期在语义上代表一个月的结束点。
    period_code  : 'CC','CY','Yk','Mk','nW','nM','nY' 等，逻辑与日频版本完全相同。
    from_today   : 仅对滚动区间有效。
    row_index    : (N,) 行索引。
    today        : 覆盖当前日期。

    返回
    ----
    start_idx      : (G,) float64，区间内最早行索引；无数据→NaN。
    end_idx        : (G,) float64，区间内最晚行索引；无数据→NaN。
    natural_months : (G,) float64，理论区间内月末日期的数量；CC 用实际 min/max；始终返回数值。
    data_points    : (G,) float64，区间内实际数据行数；无数据→NaN。
    ts             : (G,) datetime64[D]，理论区间开始日期。
    te             : (G,) datetime64[D]，理论区间结束日期。
    """
    # 调用日频版本函数获取所有核心计算结果
    start_idx, end_idx, _, data_points_out, ts, te = get_period_segments_np(
        finpro_codes=finpro_codes,
        end_dates=end_dates,
        period_code=period_code,
        from_today=from_today,
        row_index=row_index,
        today=today
    )

    # 【核心区别】计算区间内月末日期的数量
    natural_months = _count_month_ends_in_range(ts, te).astype(np.float64) - 1

    return start_idx, end_idx, natural_months, data_points_out, ts, te


if __name__ == '__main__':
    pass

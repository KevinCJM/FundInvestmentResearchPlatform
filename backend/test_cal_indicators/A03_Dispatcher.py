# -*- encoding: utf-8 -*-
"""
@File: A03_Dispatcher.py
@Modify Time: 2025/7/17 10:17
@Author: Kevin-Chen
@Descriptions:
"""
from collections import defaultdict, deque
from typing import Dict, List, Set
import pandas as pd
import numpy as np
import pandas as pd
from A02_GetPeriod import get_period_segments_np, get_period_segments_np_w, get_period_segments_np_m
from numba import njit, prange
from datetime import datetime

pd.set_option('display.max_columns', 1000)  # 显示字段的数量
pd.set_option('display.width', 1000)  # 表格不分段显示


# ---------------- 辅助：依赖闭包 ----------------
def _collect_closure(requested: List[str], registry: Dict[str, Dict]) -> Set[str]:
    """
    递归向下收集依赖（指标名 + 原始数据名），返回闭包集合。
    原始数据名：在 depends_on 中出现，但不在 registry 中。

    参数:
    requested: List[str] - 请求的节点列表，即需要收集依赖的起始点。
    registry: Dict[str, Dict] - 注册表，包含每个节点及其依赖的字典。

    返回:
    Set[str] - 收集到的依赖闭包集合，包括指标名和原始数据名。
    """
    closure: Set[str] = set()  # 初始化闭包集合，用于存储所有依赖项

    def dfs(node: str):
        """
        深度优先搜索函数，用于递归收集单个节点的所有依赖项。

        参数:
        node: str - 当前正在处理的节点。
        """
        if node in closure:
            return  # 如果节点已经在闭包中，直接返回，避免重复处理
        closure.add(node)  # 将当前节点添加到闭包中
        if node in registry:
            for dep in registry[node].get('depends_on', []):
                dfs(dep)  # 递归处理当前节点的每个依赖项

    for n in requested:
        dfs(n)  # 遍历请求的节点列表，对每个节点执行深度优先搜索
    return closure  # 返回收集到的依赖闭包集合


# ---------------- 辅助：拓扑排序 ----------------
def _topo_sort(nodes: Set[str], registry: Dict[str, Dict]) -> List[str]:
    """
    使用 Kahn 算法对给定的指标节点集合进行拓扑排序。

    该函数主要用于处理指标之间的依赖关系，确保在计算指标时，先计算依赖的指标。
    原始数据节点（即在 `registry` 中没有定义的节点）不会参与排序，但可以作为依赖项存在。

    参数:
    ----------
    nodes : Set[str]
        一个字符串集合，表示需要进行拓扑排序的所有节点（包括指标和原始数据名）。

    registry : Dict[str, Dict]
        指标注册表，定义了所有可用指标的类型、依赖关系、计算函数等。
        每个指标的 `depends_on` 字段列出了它所依赖的其他节点。

    返回:
    ----------
    List[str]
        按照拓扑顺序排列的指标节点列表。顺序确保每个节点的所有依赖项都在它之前。

    流程说明:
    ----------
    1. 初始化图结构 (`graph`) 和入度字典 (`indeg`)。
    2. 遍历所有节点，构建图结构，只处理 `registry` 中存在的节点之间的依赖关系。
    3. 构建入度为 0 的节点队列（即没有依赖的节点），开始进行广度优先搜索。
    4. 每次从队列中取出一个节点，将其加入输出列表，并减少其所有下游节点的入度。
       如果某个下游节点的入度变为 0，则将其加入队列。
    5. 最后检查是否所有节点都被成功排序，如果有环（即存在未被处理的节点），则抛出异常。
    """

    # 使用 defaultdict 构建图结构和入度字典
    graph = defaultdict(list)  # 图的邻接表表示，记录每个节点指向的其他节点
    indeg = defaultdict(int)  # 每个节点的入度，表示有多少依赖项尚未处理

    # 遍历所有节点，构建图结构和入度字典
    for n in nodes:
        # 如果当前节点不在 registry 中，跳过（原始数据节点不参与排序）
        if n not in registry:
            continue
        # 遍历当前节点的所有依赖项
        for dep in registry[n].get('depends_on', []):
            # 如果依赖项也在 registry 中，则将其加入图结构
            if dep in registry:
                graph[dep].append(n)  # 将当前节点 n 加入 dep 的下游节点列表
                indeg[n] += 1  # 当前节点 n 的入度 +1

    # 构建初始入度为 0 的节点队列（即没有依赖的节点）
    q = deque([n for n in nodes if n in registry and indeg[n] == 0])

    out = []  # 存放最终排序结果的列表

    # Kahn 拓扑排序主循环
    while q:
        u = q.popleft()  # 从队列中取出一个节点
        out.append(u)  # 将其加入输出列表
        # 遍历该节点的所有下游节点
        for v in graph[u]:
            indeg[v] -= 1  # 下游节点的入度减 1（表示一个依赖已完成）
            # 如果该下游节点的入度变为 0，说明其所有依赖都已处理完成
            if indeg[v] == 0:
                q.append(v)  # 将该节点加入队列，继续处理

    # 检查是否存在环（即未被处理的节点）
    registry_nodes = [n for n in nodes if n in registry]  # 所有在 registry 中的节点
    if len(out) != len(registry_nodes):
        # 如果输出列表长度不等于 registry 节点数量，说明存在环
        raise RuntimeError("Dependency cycle detected")

    # 返回拓扑排序结果
    return out


# ---------------- 辅助：Numba Prange 计算 ----------------
@njit(parallel=True)
def _run_segment_reduce(
        start_idx: np.ndarray, end_idx: np.ndarray,
        ts_arr: np.ndarray, day_count_arr: np.ndarray,
        data2d: np.ndarray,
        end_dates_int: np.ndarray, trading_days_int: np.ndarray,
        func_ptr,  # 函数指针
        day_type_code: int,
        annual_factor: float, ann_rf: float,
        out_mat: np.ndarray
):
    """
    通用的、并行的分段指标计算引擎。
    此函数通过 prange 并行处理每个产品，调用由 func_ptr 传入的具体指标函数。

    参数:
        start_idx (np.ndarray): 每个产品的数据起始索引数组
        end_idx (np.ndarray): 每个产品的数据结束索引数组
        ts_arr (np.ndarray): 时间序列相关参数数组
        day_count_arr (np.ndarray): 日期计数数组
        data2d (np.ndarray): 二维数据数组，包含所有产品的数据
        end_dates_int (np.ndarray): 结束日期整数数组
        trading_days_int (np.ndarray): 交易日整数数组
        func_ptr: 指向具体指标计算函数的函数指针
        day_type_code (int): 日期类型编码
        annual_factor (float): 年化因子
        ann_rf (float): 年化无风险利率
        out_mat (np.ndarray): 输出矩阵，存储计算结果

    返回值:
        无直接返回值，计算结果存储在 out_mat 参数中
    """
    G = start_idx.size
    n_out = out_mat.shape[0]

    # 并行处理每个产品的数据段
    for i in prange(G):
        s = start_idx[i]
        e = end_idx[i]

        # 检查索引有效性，无效时填充 NaN
        if s < 0 or e < s:
            for k in range(n_out):
                out_mat[k, i] = np.nan
            continue

        # 调用具体的指标计算函数
        res = func_ptr(
            data2d, int(s), int(e),
            end_dates_int, trading_days_int,
            ts_arr[i],
            day_type_code,
            day_count_arr[i],
            annual_factor,
            ann_rf
        )

        # 规定所有 func_ptr 指向的函数都必须返回一个元组。 这个循环对单输出(res=(val1,))和多输出(res=(v1,v2))都有效。
        for k in range(n_out):
            out_mat[k, i] = res[k]


def dt64_to_yyyymmdd_float(dt_arr: np.ndarray) -> np.ndarray:
    """
    dt_arr: dtype 可以是 datetime64[D]、int64(epoch-days) 或 float(含 NaN)。
    返回: float64，形如 20250403.00，NaT/无效值 -> np.nan
    """
    # 1. 统一成 datetime64[D]
    if np.issubdtype(dt_arr.dtype, np.datetime64):
        dt = dt_arr.astype('datetime64[D]')
        mask_nat = dt == np.datetime64('NaT')
    else:
        # 假设是 epoch-days 数字（含 NaN）
        tmp = dt_arr.astype('float64')
        mask_nat = np.isnan(tmp)
        days = tmp[~mask_nat].astype('int64')
        dt = np.empty_like(tmp, dtype='datetime64[D]')
        dt[~mask_nat] = np.datetime64('1970-01-01', 'D') + days.astype('timedelta64[D]')
        dt[mask_nat] = np.datetime64('NaT')
        dt = dt.astype('datetime64[D]')  # 保证 dtype

    # 2. 拆年/月/日
    y = dt.astype('datetime64[Y]').astype(np.int32) + 1970
    m = dt.astype('datetime64[M]').astype(np.int32) % 12 + 1
    d = (dt - dt.astype('datetime64[M]')).astype(np.int32) + 1

    out = (y * 10000 + m * 100 + d).astype(np.float64)
    out[mask_nat] = np.nan
    return out


def execute_indicators_for_period(
        finpro_codes: np.ndarray,
        end_dates: np.ndarray,
        i_dict: Dict[str, Dict],
        registry: Dict[str, Dict],
        period_code: str,
        today: np.datetime64,
        frequency: str,
        from_today: bool = True,
        trading_days_data: np.ndarray = None,
        **raw_arrays: np.ndarray,
) -> pd.DataFrame:
    """
    指标计算调度器，支持依赖原始数据、空 period 指标、以及多源数据的统一调度。
    该函数根据给定的指标字典 `i_dict` 和指标注册表 `registry`，按照依赖顺序调度并计算指标。
    它适用于在特定时间段（由 `period_code` 定义）内为每组理财产品（`finpro_codes`）计算指标。

    参数:
    ----------
    finpro_codes : np.ndarray
        一维数组，包含每个理财产品的产品代码（str/obj/int均可），用于分组计算。

    end_dates : np.ndarray
        一维数组，包含每个数据点的结束日期（datetime64[D]），用于确定每个产品的时间段。

    i_dict : Dict[str, Dict]
        指标字典，定义了用户请求的指标及其元信息，如支持的 `period_code`。

    registry : Dict[str, Dict]
        指标注册表，定义了所有可用指标的类型、依赖关系、计算函数等。

    period_code : str
        时间段代码，如 'CC', 'CY', 'Yk', 'Mk', 'nW', 'nM', 'nY' 等。

    today : np.datetime64
        基准日期，通常为当前日期，用于滚动区间计算。

    frequency : str
        计算频率 ('YS', 'W', 'M')，用于选择正确的区间切分函数。

    from_today : bool, 默认为 True
        仅对滚动区间（如 nW, nM, nY）有效。

    trading_days_data: np.ndarray, optional
        一个包含所有交易日期的有序NumPy数组 (dtype=datetime64[D])。

    **raw_arrays : np.ndarray
        原始数据数组，以关键字参数形式传入。

    返回:
    ----------
    pd.DataFrame
        一个包含 ['FinProCode', 'secassetcatcode', 'EndDate', 'period', 'index_code', 'index_value'] 列的 DataFrame。
    """
    # ---------- 0a. 将产品按顺序排列 ----------
    uniq_codes, first_pos = np.unique(finpro_codes, return_index=True)
    order_u = np.argsort(first_pos, kind='mergesort')
    all_products = uniq_codes[order_u]
    G = len(all_products)

    # ---------- 0b. 创建产品到分类的映射 ----------
    secassetcatcode_array = raw_arrays.get('secassetcatcode')
    if secassetcatcode_array is None:
        raise ValueError("secassetcatcode must be provided in raw_arrays")
    df_cat_map = pd.DataFrame({
        'FinProCode': finpro_codes,
        'secassetcatcode': secassetcatcode_array
    }).drop_duplicates(subset=['FinProCode'])

    # ---------- 0c. 区间切片 ----------
    # 根据频率选择正确的区间切分函数
    freq_upper = frequency.upper()
    if freq_upper == 'W':
        segment_func = get_period_segments_np_w
    elif freq_upper == 'M':
        segment_func = get_period_segments_np_m
    else:  # 默认为 'YS' 或其他日频
        segment_func = get_period_segments_np

    (start_idx,  # 区间起始位置索引（在原始数组中的位置）
     end_idx,  # 区间结束位置索引（在原始数组中的位置）
     period_duration,  # 可能是天数、周数或月数
     n_pts,  # 区间内的数据点数量（即 start_idx 到 end_idx 的长度）
     ts,  # 每个理财产品的起始日期（datetime64[D] 类型）
     te  # 每个理财产品的结束日期（datetime64[D] 类型）
     ) = segment_func(
        finpro_codes=finpro_codes,
        end_dates=end_dates,
        period_code=period_code,
        from_today=from_today,
        row_index=np.arange(len(finpro_codes)),
        today=today
    )

    # ---------- 1. 提取终端指标 ----------
    terminal_metrics = [
        name for name, meta in i_dict.items()
        if period_code in meta.get("periods", [])
    ]

    if not terminal_metrics:
        return pd.DataFrame(columns=['FinProCode', 'secassetcatcode', 'EndDate', 'period', 'index_code', 'index_value'])

    # ---------- 2. 依赖闭包与拓扑排序 ----------
    closure_nodes = _collect_closure(terminal_metrics, registry)  # 收集所有需要的度量指标节点
    topo_order = _topo_sort(closure_nodes, registry)  # 对收集到的节点进行拓扑排序

    # ---------- 4. 初始化缓存 (已包含所有天数类型) ----------
    cache: Dict[str, np.ndarray] = {
        # 关键：将计算出的周期长度（天/周/月）统一放入 'n_days' key 中
        # 后续的年化计算逻辑依赖于此 key，并且 B01_Config 中的 annual_factor 已根据频率调整
        'n_days': period_duration.astype(np.float64, copy=False),
        'n_pts': n_pts.astype(np.float64, copy=False),  # 区间内的数据点数量（如：30个每日数据点）
        't_days': trading_days_data,  # 交易日数组，用于交易日计数（如：剔除非交易日）
        'start_idx': start_idx,  # 每个理财产品的收益率数组起始索引（用于切片）
        'end_idx': end_idx  # 每个理财产品的收益率数组结束索引（用于切片）
    }
    for k, v in raw_arrays.items():
        cache[k] = v

    # ---------- 5. 执行指标计算 ----------
    results: Dict[str, np.ndarray] = {}

    # 定义 day_type 到整数代码的映射，增强可读性和可维护性
    DAY_TYPE_MAPPING = {'t_days': 0, 'n_pts': 1, 'n_days': 2}

    # 提前将日期转换为 Numba 更高效的 int64 格式
    end_dates_int = end_dates.astype(np.int64)
    trading_days_int = (
        trading_days_data.astype(np.int64)
        if trading_days_data is not None else np.empty(0, dtype=np.int64)
    )

    for ind in topo_order:  # 按拓扑排序顺序进行指标计算
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] Executing {ind}...")
        if ind in cache:
            continue

        meta = registry[ind]
        kind = meta['kind']

        if kind == 'segment_reduce':
            # 从元数据中获取功能函数指针和源数据列表
            func_ptr = meta['func']
            source_list = meta['source']

            # 将源数据列表中的每个源数据从raw_arrays中取出并堆叠成一个矩阵
            src_mat = np.ascontiguousarray(np.vstack([raw_arrays[src] for src in source_list]))

            # 获取用于计算天数的源数据，如果没有指定，则默认使用'n_pts'
            day_count_source = meta.get('day_count_source', 'n_pts')

            # 根据day_count_source获取对应的天数类型代码
            day_type_code = DAY_TYPE_MAPPING.get(day_count_source, -1)

            # 从缓存中获取天数计数数组，如果没有，则创建一个全为-1.0的数组
            day_count_arr = cache.get(day_count_source, np.full(G, -1.0))

            # 从元数据中获取输出键列表，如果没有指定，则默认使用[ind]
            output_keys = meta.get('outputs', [ind])

            # 根据输出键的数量创建一个全为NaN的输出矩阵
            n_out = len(output_keys)
            out_mat = np.full((n_out, G), np.nan, dtype=np.float64)

            # 运行segment_reduce函数
            _run_segment_reduce(
                start_idx.astype(np.int64),  # 分段起始索引数组 (np.ndarray, 1-D, int64)
                end_idx.astype(np.int64),  # 分段结束索引数组 (np.ndarray, 1-D, int64)
                ts.astype(np.int64),  # 时间序列数组 (各产品的起始日期) (np.ndarray, 1-D, int64)
                day_count_arr.astype(np.float64),  # 每个分段的天数数组，用于计算年化等  (np.ndarray, 1-D, float64)
                src_mat,  # 输入的二维数据矩阵（n_src × T），用于指标计算 (np.ndarray, 2-D, float64)
                end_dates_int,  # 每个分段的结束日期数组 (np.ndarray, 1-D, int64)
                trading_days_int,  # 交易日历数组 (np.ndarray, 1-D, int64)
                func_ptr,  # 指向具体指标计算函数的函数指针 (函数指针)
                day_type_code,  # 天数类型代码（0: t_days, 1: n_pts, 2: n_days）(int)
                float(meta.get('annual_factor', 365.0)),  # 年化因子，用于将指标转换为年化值 (float)
                float(meta.get('ann_rf', 0.015)),  # 年化无风险利率，用于风险调整计算 (float)
                out_mat  # 输出矩阵，用于存储计算结果 (n_out × G) (np.ndarray, 2-D, float64)
            )

            for i, key in enumerate(output_keys):
                result_array = out_mat[i]
                if key in meta.get('date_outputs', []):
                    result_array = dt64_to_yyyymmdd_float(result_array)
                cache[key] = result_array
                results[key] = result_array

        elif kind == 'vector_transform':
            func = meta['func']
            day_count_source = meta.get('day_count_source', 'n_days')
            out_arr = func(
                cache=cache,
                day_type=day_count_source,
                day_count=cache[day_count_source],
                annual_factor=meta.get('annual_factor', 365)
            )
            cache[ind] = out_arr
            results[ind] = out_arr

        else:
            raise ValueError(f"未知 kind: {kind}")

    # ---------- 6. 格式化输出为 DataFrame ----------
    output_dfs = []
    final_results = {k: results[k] for k in terminal_metrics if k in results}

    for index_code, index_values in final_results.items():
        df_temp = pd.DataFrame({
            'FinProCode': all_products,
            'EndDate': te,
            'period': period_code,
            'index_code': index_code,
            'index_value': index_values
        })
        output_dfs.append(df_temp)

    if not output_dfs:
        return pd.DataFrame(columns=['FinProCode', 'secassetcatcode', 'EndDate', 'period', 'index_code', 'index_value'])

    final_df = pd.concat(output_dfs, ignore_index=True)

    final_df = pd.merge(final_df, df_cat_map, on='FinProCode', how='left')

    final_cols = ['FinProCode', 'secassetcatcode', 'EndDate', 'period', 'index_code', 'index_value']
    final_df = final_df[final_cols]

    final_df_with_ranking = add_stats_and_ranking(final_df, i_dict)

    return final_df_with_ranking


def add_stats_and_ranking(df: pd.DataFrame, indicator_config: Dict[str, Dict]) -> pd.DataFrame:
    """
    为指标结果DataFrame添加排名、总数、中位数和均值。

    Args:
        df (pd.DataFrame): 包含指标计算结果的DataFrame。
        indicator_config (Dict[str, Dict]): 指标配置字典，用于获取排名逻辑。

    Returns:
        pd.DataFrame: 增加了 'ranking', 'totalnum', 'valuemedian', 'valuemean' 列的新DataFrame。
    """
    if df.empty:
        return df

    # 定义分组键
    group_by_cols = ['secassetcatcode', 'period', 'index_code']

    # 计算统计数据
    stats = df.groupby(group_by_cols)['index_value'].agg(['count', 'median', 'mean']).reset_index()
    stats.rename(columns={'count': 'totalnum', 'median': 'valuemedian', 'mean': 'valuemean'}, inplace=True)

    # 合并统计数据
    df = pd.merge(df, stats, on=group_by_cols, how='left')

    df['ranking'] = df.groupby(group_by_cols, group_keys=False).apply(_calculate_ranking_for_group, indicator_config)

    return df


def _calculate_ranking_for_group(group: pd.DataFrame, indicator_config: Dict[str, Dict]) -> pd.Series:
    """
    为单个分组计算排名。
    """
    index_code = group['index_code'].iloc[0]
    config = indicator_config.get(index_code, {})
    order = config.get('index_order', '')
    spec_num = config.get('spec_num')
    spec_rank = config.get('spec_rank', 'N')

    if not order:
        return pd.Series([None] * len(group), index=group.index)

    # 待排名的数据
    values_to_rank = group['index_value'].copy()

    # 处理特殊值
    if spec_num is not None and spec_rank == 'N':
        # 不参与排名的特殊值，暂时设为NaN
        values_to_rank[values_to_rank == spec_num] = np.nan

    # 计算排名
    ascending = (order == 'S')
    # 使用 'min' 方法处理并列排名
    ranking = values_to_rank.rank(method='min', ascending=ascending, na_option='bottom')

    # 将被设为NaN的特殊值排名恢复为None
    if spec_num is not None and spec_rank == 'N':
        ranking[group['index_value'] == spec_num] = None

    return ranking


if __name__ == '__main__':
    import time
    from A01_Scheduler import read_data, read_data_w
    from B01_Config import indicator_dic, get_indicator_registry

    # # the_df_return = read_data()
    # the_df_return = read_data_w(w_data_path="dx_weekly_trade_df.parquet")
    # print(the_df_return.head())
    # the_df_return = the_df_return[['FinProCode', 'EndDate', 'secassetcatcode', 'PctChange', 'Benchmark', 'rf']].dropna(
    #     subset=['PctChange']).reset_index(drop=True)
    # the_df_return = the_df_return[
    #     the_df_return['FinProCode'].isin(['SEC00007IU8Q', 'SEC00007K3ZB', 'SEC00007L2M2', 'SEC00007L8D1']
    #                                      )].reset_index(drop=True)
    # the_df_return.to_parquet("dx_w_test_financial_data.parquet")
    # piv_df_f = the_df_return.pivot(index='EndDate', columns='FinProCode', values='PctChange')
    # # piv_df_f = piv_df_f.resample('D').asfreq()
    # piv_df_b = the_df_return.pivot(index='EndDate', columns='FinProCode', values='Benchmark')
    # # piv_df_b = piv_df_b.resample('D').asfreq()
    # piv_df_rf = the_df_return.pivot(index='EndDate', columns='FinProCode', values='rf')
    # # piv_df_rf = piv_df_rf.resample('D').asfreq()
    # piv_df_f.to_excel('piv_df_f.xlsx')
    # piv_df_b.to_excel('piv_df_b.xlsx')
    # piv_df_rf.to_excel('piv_df_rf.xlsx')

    the_df_return = pd.read_parquet("dx_w_test_financial_data.parquet")
    the_df_return.to_excel("dx_w_test_financial_data.xlsx")
    print(the_df_return.head())

    # 提取数组
    finpro = the_df_return['FinProCode'].to_numpy()
    secassetcatcode = the_df_return['secassetcatcode'].to_numpy()
    end_dt = the_df_return['EndDate'].to_numpy().astype('datetime64[D]')
    pct_array = the_df_return['PctChange'].to_numpy()
    bench_array = the_df_return['Benchmark'].to_numpy()
    rf_array = the_df_return['rf'].to_numpy()

    # 加载交易日数据
    try:
        tradingday_df = pd.read_parquet('tradedate.parquet')
        trading_days_data = tradingday_df[tradingday_df['IfTradingDay'] == 1]['TradingDate'].to_numpy().astype(
            'datetime64[D]')
    except (FileNotFoundError, KeyError) as e:
        print(f"警告: 交易日数据处理失败 ({e})，交易日相关指标将无法计算。")
        trading_days_data = None

    # 计算指标
    s_t = time.time()
    res_df = execute_indicators_for_period(
        finpro_codes=finpro,
        end_dates=end_dt,
        i_dict=indicator_dic,
        registry=get_indicator_registry('W'),
        period_code='6M',
        frequency='W',
        from_today=False,
        today=np.datetime64('2025-07-17'),
        trading_days_data=trading_days_data,
        pct_array=pct_array,
        bench_array=bench_array,
        rf_array=rf_array,
        secassetcatcode=secassetcatcode
    )

    print(res_df.head())
    print(f"指标计算耗时: {(time.time() - s_t):.4f} 秒")

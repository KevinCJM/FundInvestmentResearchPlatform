# -*- encoding: utf-8 -*-
"""
@File: A01_Scheduler.py
@Modify Time: 2025/7/23 15:18
@Author: Kevin-Chen
@Descriptions: 总调度器, 负责进程池功能, 以及共享内存
"""
import multiprocessing
import os
import time
import numpy as np
import pandas as pd
from multiprocessing import shared_memory
from functools import partial
import threading
from datetime import datetime

# --- 检查可选依赖库 ---
try:
    import psutil
except ImportError:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] 警告: 'psutil' 模块未安装。内存监控功能将不可用。")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] 请运行 'pip install psutil' 来安装它。")
    psutil = None

try:
    import openpyxl
except ImportError:
    print(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] 警告: 'openpyxl' 模块未安装。无法将内存日志保存为 .xlsx 文件。")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] 请运行 'pip install openpyxl' 来安装它。")
    openpyxl = None

from A03_Dispatcher import execute_indicators_for_period
from B01_Config import p_list, w_p_list, m_p_list, indicator_dic
from B01_Config import w_indicator_dic, m_indicator_dic, get_indicator_registry


# --- 内存监控 ---
def monitor_memory(stop_event, interval=2, unit='GB', output_file=None):
    """在一个单独的线程中监控并打印全局内存使用情况，并将结果保存到文件。"""
    if not psutil:
        return

    memory_logs = []
    print(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] 内存监控线程已启动... 记录间隔: {interval}秒, 单位: {unit}")

    while not stop_event.is_set():
        timestamp = pd.Timestamp.now()
        mem = psutil.virtual_memory()

        # 根据单位选择除数
        if unit.upper() == 'MB':
            divisor = 1024 ** 2
            display_unit = 'MB'
        else:  # 默认为 GB
            divisor = 1024 ** 3
            display_unit = 'GB'

        used = mem.used / divisor
        total = mem.total / divisor

        # 打印到控制台
        print(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] [内存监控] 系统已用: {used:.2f} {display_unit} / {total:.2f} {display_unit} ({mem.percent}%)")

        # 记录日志
        memory_logs.append({
            'Timestamp': timestamp,
            f'Used ({display_unit})': used,
            f'Total ({display_unit})': total,
            'Percent (%)': mem.percent
        })

        stop_event.wait(interval)

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] 内存监控线程已停止。")

    # 保存到 Excel 文件
    if output_file and memory_logs:
        if not openpyxl:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] 无法保存Excel日志，因为 'openpyxl' 未安装。")
            return

        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] 正在将内存监控日志保存到 {output_file}...")
        try:
            df_logs = pd.DataFrame(memory_logs)
            df_logs['Timestamp'] = df_logs['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            df_logs.to_excel(output_file, index=False, engine='openpyxl')
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] 内存日志已成功保存。")
        except Exception as e:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] 错误：无法保存内存日志文件: {e}")


def add_risk_free_rate_column(
        df: pd.DataFrame,
        annual_rf_rate: float = 0.015,
        annual_days: int = 365
) -> pd.DataFrame:
    """
    为DataFrame添加与每期收益率匹配的无风险利率（rf）列。

    该函数通过计算每条收益记录与其前一条记录之间的时间差（天数），
    然后使用复利公式计算出该持有期间对应的无风险收益率。

    Args:
        df (pd.DataFrame): 必须包含 'FinProCode' 和 'EndDate' 列，且已按产品和日期排序。
        annual_rf_rate (float): 年化无风险利率。
        annual_days (int): 年化基数，通常为 365。

    Returns:
        pd.DataFrame: 增加了 'rf' 列的新DataFrame。
    """
    # 按产品分组，计算每个收益率日期与上一个日期之间的天数差
    # .diff() 会自动处理每个分组的边界，第一个值为 NaT
    days_diff = df.groupby('FinProCode')['EndDate'].diff().dt.days

    # 应用复利公式计算每期的无风险利率
    # (1 + 年利率)^(持有天数 / 年化基数) - 1
    df['rf'] = (1 + annual_rf_rate) ** (days_diff / annual_days) - 1

    return df


# YS 频率数据读取与处理
def read_data(financial_data_path='financial_data.parquet', benchmark_data_path='benchmark_index_nv.parquet'):
    """
    读取并处理基金财务数据和基准数据，进行数据合并、清洗和特征计算

    参数:
        financial_data_path (str): 基金财务数据文件路径，格式为parquet
        benchmark_data_path (str): 基准数据文件路径，格式为parquet

    返回:
        pandas.DataFrame: 处理后的数据框，包含基金收益率、基准收益率和无风险利率等字段
    """
    # 读取基准数据并进行字段重命名和日期格式转换
    b = pd.read_parquet(benchmark_data_path)
    b.rename(columns={'nv': 'benchmark_nv', 'enddate': 'EndDate'}, inplace=True)
    b['EndDate'] = pd.to_datetime(b['EndDate'])

    # 读取基金财务数据并添加资产分类代码，转换日期格式
    f = pd.read_parquet(financial_data_path)
    if 'SecAssetCatCode' in f.columns:
        f.rename(columns={'SecAssetCatCode': 'secassetcatcode'}, inplace=True)
    else:
        f['secassetcatcode'] = 'FCC000001EKW'  # TODO : 这里需要替换为实际的 secassetcatcode 分类
    f['EndDate'] = pd.to_datetime(f['EndDate'])

    # 根据 secassetcatcode 和 enddate 进行合并
    df_nv = pd.merge(f, b[['secassetcatcode', 'EndDate', 'benchmark_nv']], how='left',
                     left_on=['secassetcatcode', 'EndDate'], right_on=['secassetcatcode', 'EndDate'])

    # 按产品代码和结束日期排序，并去除重复记录，保留最新记录
    df_nv = df_nv.sort_values(by=['FinProCode', 'EndDate']).reset_index(drop=True)
    df_nv = df_nv.drop_duplicates(subset=['FinProCode', 'EndDate'], keep='last')

    # 计算基金日涨跌 & 基准日涨跌
    df_nv['PctChange'] = df_nv.groupby('FinProCode')['UnitNVRestored'].pct_change()
    df_nv['Benchmark'] = df_nv.groupby('FinProCode')['benchmark_nv'].pct_change()

    # 调用函数计算并添加无风险利率列
    df_nv = add_risk_free_rate_column(df_nv, 0.015, 365)

    return df_nv.sort_values(by=['FinProCode', 'EndDate']).reset_index(drop=True)


# W 频率数据读取与处理
def read_data_w(w_data_path='weekly_trade_df.parquet'):
    """
    读取并处理基金周度交易数据

    参数:
        w_data_path (str): 数据文件路径，默认为'weekly_trade_df.parquet'

    返回:
        pandas.DataFrame: 处理后的基金数据，包含收益率、基准收益率和无风险利率等列
    """
    df_nv = pd.read_parquet(w_data_path)
    df_nv['EndDate'] = pd.to_datetime(df_nv['EndDate'])

    # 按产品代码和结束日期排序，并去除重复记录，保留最新记录
    df_nv = df_nv.sort_values(by=['FinProCode', 'EndDate']).reset_index(drop=True)
    df_nv = df_nv.drop_duplicates(subset=['FinProCode', 'EndDate'], keep='last')

    # 计算基金日涨跌 & 基准日涨跌
    df_nv['PctChange'] = df_nv.groupby('FinProCode')['UnitNVRestored'].pct_change()
    df_nv['Benchmark'] = df_nv.groupby('FinProCode')['benchmark_nv'].pct_change()

    # 调用函数计算并添加无风险利率列
    df_nv = add_risk_free_rate_column(df_nv, 0.015, 52)

    return df_nv.sort_values(by=['FinProCode', 'EndDate']).reset_index(drop=True)


# M 频率数据读取与处理
def read_data_m(m_data_path='monthly_trade_df.parquet'):
    """
    读取并处理月度交易数据

    参数:
        m_data_path (str): 月度交易数据文件路径，默认为'monthly_trade_df.parquet'

    返回:
        pandas.DataFrame: 处理后的月度交易数据，包含产品代码、结束日期、单位净值、基准净值、
                         日涨跌幅度、基准日涨跌幅度和无风险利率等字段，按产品代码和结束日期排序
    """
    df_nv = pd.read_parquet(m_data_path)
    df_nv['EndDate'] = pd.to_datetime(df_nv['EndDate'])

    # 按产品代码和结束日期排序，并去除重复记录，保留最新记录
    df_nv = df_nv.sort_values(by=['FinProCode', 'EndDate']).reset_index(drop=True)
    df_nv = df_nv.drop_duplicates(subset=['FinProCode', 'EndDate'], keep='last')

    # 计算基金日涨跌 & 基准日涨跌
    df_nv['PctChange'] = df_nv.groupby('FinProCode')['UnitNVRestored'].pct_change()
    df_nv['Benchmark'] = df_nv.groupby('FinProCode')['benchmark_nv'].pct_change()

    # 调用函数计算并添加无风险利率列
    df_nv = add_risk_free_rate_column(df_nv, 0.015, 12)

    return df_nv.sort_values(by=['FinProCode', 'EndDate']).reset_index(drop=True)


# --- 共享内存辅助函数 ---
def attach_to_shared_array(name, shape, dtype):
    """根据名称、形状和数据类型附加到现有的共享内存块，并返回一个NumPy数组视图。"""
    shm = shared_memory.SharedMemory(name=name)
    return np.ndarray(shape, dtype=dtype, buffer=shm.buf), shm


# --- 进程池的工作函数 ---
def worker_task(period_code, shared_data_info, today_date, indicator_config, registry, frequency):
    """
    由进程池中的每个子进程执行的任务。

    参数:
    - period_code: 需要处理的时间区间代码。
    - shared_data_info: 包含共享数据信息的字典。
    - today_date: 今天的日期。
    - indicator_config: 指标配置字典。
    - registry: 指标注册表
    - frequency: 计算频率

    返回:
    - result_df: 处理后生成的结果 DataFrame。
    """
    # 打印子进程开始处理的时间和进程ID以及处理的区间代码
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] 进程 {os.getpid()} 开始处理区间: {period_code}...")
    s_t = time.time()
    shm_references = []
    try:
        # 附加到共享数组并获取金融产品数组
        finpro_arr, finpro_shm = attach_to_shared_array(**shared_data_info['finpro'])
        shm_references.append(finpro_shm)
        finpro = np.array([s.decode('utf-8') for s in finpro_arr])

        # 附加到共享数组并获取 secassetcatcode 数组
        secassetcatcode_arr, secassetcatcode_shm = attach_to_shared_array(**shared_data_info['secassetcatcode'])
        shm_references.append(secassetcatcode_shm)
        secassetcatcode = np.array([s.decode('utf-8') for s in secassetcatcode_arr])

        # 附加到共享数组并获取结束日期数组
        end_dt, end_dt_shm = attach_to_shared_array(**shared_data_info['end_dt'])
        shm_references.append(end_dt_shm)

        # 附加到共享数组并获取百分比数组
        pct_array, pct_array_shm = attach_to_shared_array(**shared_data_info['pct_array'])
        shm_references.append(pct_array_shm)

        # 附加到共享数组并获取基准数组
        bench_array, bench_array_shm = attach_to_shared_array(**shared_data_info['bench_array'])
        shm_references.append(bench_array_shm)

        # 附加到共享数组并获取无风险利率数组
        rf_array, rf_array_shm = attach_to_shared_array(**shared_data_info['rf_array'])
        shm_references.append(rf_array_shm)

        # 附加到共享数组并获取交易日数据（如果存在）
        trading_days_data = None
        if shared_data_info['trading_days_data']:
            trading_days_data, trading_days_shm = attach_to_shared_array(**shared_data_info['trading_days_data'])
            shm_references.append(trading_days_shm)

        # 执行指标计算
        result_df = execute_indicators_for_period(
            finpro_codes=finpro, end_dates=end_dt, i_dict=indicator_config,
            registry=registry, period_code=period_code, from_today=False,
            today=today_date, trading_days_data=trading_days_data,
            pct_array=pct_array, bench_array=bench_array, rf_array=rf_array,
            secassetcatcode=secassetcatcode, frequency=frequency
        )
    finally:
        # 关闭所有共享内存引用
        for shm in shm_references:
            shm.close()

    # 打印子进程完成处理的时间和进程ID以及处理的区间代码和耗时
    e_t = time.time()
    print(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] 进程 {os.getpid()} 完成区间: {period_code} 的计算, 耗时 {e_t - s_t:.2f} 秒")
    return result_df


def prepare_data_and_get_arrays(financial_data_path, benchmark_data_path, trading_day_path, frequency='YS'):
    """加载、准备数据并提取Numpy数组。"""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] 主进程: 正在加载{frequency}频率数据...")
    if frequency.upper() == 'YS':
        the_df_return = read_data(financial_data_path, benchmark_data_path)
    elif frequency.upper() == 'W':
        the_df_return = read_data_w(w_data_path=financial_data_path)
    elif frequency.upper() == 'M':
        the_df_return = read_data_m(m_data_path=financial_data_path)
    else:
        raise ValueError("frequency 参数必须是 'YS'（年）、'W'（周）或 'M'（月）。")

    the_df_return = the_df_return[['FinProCode', 'secassetcatcode', 'EndDate', 'PctChange', 'Benchmark', 'rf']].dropna(
        subset=['PctChange']).reset_index(drop=True)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] 主进程: 数据加载完毕。")

    # 提取核心数组
    finpro_str = the_df_return['FinProCode'].to_numpy()
    max_len = max(len(s.encode('utf-8')) for s in finpro_str) if len(finpro_str) > 0 else 1
    finpro_bytes = np.array([s.encode('utf-8') for s in finpro_str], dtype=f'S{max_len}')

    secassetcatcode_str = the_df_return['secassetcatcode'].to_numpy()
    max_len_cat = max(len(s.encode('utf-8')) for s in secassetcatcode_str) if len(secassetcatcode_str) > 0 else 1
    secassetcatcode_bytes = np.array([s.encode('utf-8') for s in secassetcatcode_str], dtype=f'S{max_len_cat}')

    end_dt = the_df_return['EndDate'].to_numpy(dtype='datetime64[D]')
    pct_array = the_df_return['PctChange'].to_numpy(dtype=np.float64)
    bench_array = the_df_return['Benchmark'].to_numpy(dtype=np.float64)
    rf_array = the_df_return['rf'].to_numpy(dtype=np.float64)

    trading_days_data = None
    try:
        tradingday_df = pd.read_parquet(trading_day_path)
        trading_days_data = tradingday_df[tradingday_df['IfTradingDay'] == 1]['TradingDate'].to_numpy(
            dtype='datetime64[D]')
    except (FileNotFoundError, KeyError) as e:
        print(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] 警告: 无法加载交易日数据 ({e})。依赖交易日的指标将无法计算。")

    return {
        'finpro': finpro_bytes, 'secassetcatcode': secassetcatcode_bytes, 'end_dt': end_dt, 'pct_array': pct_array,
        'bench_array': bench_array, 'rf_array': rf_array, 'trading_days_data': trading_days_data
    }


def setup_shared_memory(arrays_to_share):
    """根据给定的Numpy数组创建共享内存。"""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] 主进程: 正在创建共享内存...")
    shared_data_info = {}
    shm_list = []
    for name, arr in arrays_to_share.items():
        if arr is not None:
            try:
                shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
                shared_arr = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
                shared_arr[:] = arr[:]
                shared_data_info[name] = {'name': shm.name, 'shape': arr.shape, 'dtype': arr.dtype}
                shm_list.append(shm)
            except Exception as e:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] 创建共享内存失败: {name}, 错误: {e}")
                # 清理已创建的共享内存
                for s in shm_list:
                    s.close()
                    s.unlink()
                raise
        else:
            shared_data_info[name] = None
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] 主进程: 共享内存创建完毕。")
    return shared_data_info, shm_list


def run_calculation(shared_data_info, task_list, indicator_config, registry, frequency):
    """设置并运行进程池来执行计算任务。"""
    num_processes = os.cpu_count()
    print(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] 主进程: 即将启动 {num_processes} 个子进程进行计算...")
    today_date = np.datetime64('today', 'D')
    task_with_args = partial(worker_task, shared_data_info=shared_data_info, today_date=today_date,
                             indicator_config=indicator_config, registry=registry, frequency=frequency)

    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(task_with_args, task_list)

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] 主进程: 所有计算任务已完成，正在合并结果...")
    return pd.concat(results, ignore_index=True)


def main(financial_data_path, benchmark_data_path, trading_day_path, output_path, memory_log_path,
         enable_memory_monitoring=True, frequency='YS'):
    """
    主执行函数，协调整个计算流程。

    参数:
        financial_data_path (str): 金融数据文件路径。
        benchmark_data_path (str): 基准数据文件路径。
        trading_day_path (str): 交易日历数据文件路径。
        output_path (str): 计算结果输出文件路径（Parquet格式）。
        memory_log_path (str): 内存监控日志输出路径。
        enable_memory_monitoring (bool): 是否启用内存监控，默认为 True。
        frequency (str): 数据频率类型，支持 'YS'（年）、'W'（周）、'M'（月）

    返回值:
        无返回值。函数主要负责流程控制和资源管理。
    """
    # 启动内存监控线程（如果启用且 psutil 可用）
    monitor_thread = None
    stop_event = threading.Event()
    if psutil and enable_memory_monitoring:
        monitor_thread = threading.Thread(
            target=monitor_memory,
            args=(stop_event,),
            kwargs={'interval': 1, 'unit': 'MB', 'output_file': memory_log_path}
        )
        monitor_thread.start()

    shm_list = []
    overall_start_time = time.time()
    try:
        # 1. 根据频率选择配置
        freq_upper = frequency.upper()
        if freq_upper == 'YS':
            selected_p_list = p_list
            selected_indicator_dic = indicator_dic
        elif freq_upper == 'W':
            selected_p_list = w_p_list
            selected_indicator_dic = w_indicator_dic
        elif freq_upper == 'M':
            selected_p_list = m_p_list
            selected_indicator_dic = m_indicator_dic
        else:
            raise ValueError(f"不支持的频率: {frequency}. 请使用 'YS', 'W', 或 'M'.")

        # 获取对应频率的指标注册表
        registry = get_indicator_registry(frequency)

        # 2. 加载并准备金融数据、基准数据和交易日数据
        arrays_to_share = prepare_data_and_get_arrays(financial_data_path, benchmark_data_path,
                                                      trading_day_path, frequency)

        # 3. 将数据加载到共享内存中，以便多进程访问
        shared_data_info, shm_list = setup_shared_memory(arrays_to_share)

        # 4. 执行核心计算逻辑
        final_df = run_calculation(shared_data_info, selected_p_list, selected_indicator_dic, registry, frequency)

        # 5. 输出计算结果统计信息，并将结果保存至指定路径
        overall_end_time = time.time()
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] --- 计算完成 ---")
        print(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] 总耗时: "
            f"{overall_end_time - overall_start_time:.2f} 秒")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] 总计生成 {len(final_df)} 条指标数据")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] 最终结果抽样:")
        print(final_df.head())
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] 保存结果到文件 {output_path}")
        final_df.to_parquet(output_path, index=False)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] 结果保存完毕")

    finally:
        # 停止内存监控线程（如已启动）
        if monitor_thread and monitor_thread.is_alive():
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] 主进程: 正在停止内存监控...")
            stop_event.set()
            monitor_thread.join()

        # 清理所有创建的共享内存对象
        if shm_list:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] 主进程: 正在清理共享内存...")
            for shm in shm_list:
                shm.close()
                shm.unlink()
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] 主进程: 清理完毕。")


# --- 主执行模块 ---
if __name__ == '__main__':
    # 定义输入输出文件路径
    for FREQ in ['YS', 'W', 'M']:
        if FREQ.upper() == 'YS':
            FINANCIAL_DATA_PATH = "financial_data.parquet"
        elif FREQ.upper() == 'W':
            FINANCIAL_DATA_PATH = "weekly_trade_df.parquet"
        elif FREQ.upper() == 'M':
            FINANCIAL_DATA_PATH = "monthly_trade_df.parquet"
        else:
            raise ValueError(f"不支持的频率: {FREQ}. 请使用 'YS', 'W', 或 'M'.")

        BENCHMARK_DATA_PATH = "benchmark_index_nv.parquet"
        TRADING_DAY_PATH = 'tradedate.parquet'
        OUTPUT_PATH = f'indicator_results_{FREQ}.parquet'
        MEMORY_LOG_PATH = f"memory_usage_log_{FREQ}.xlsx"
        ENABLE_MEMORY_MONITORING = False  # 设置为 True 可以进行内存监控

        # 运行主函数
        main(
            financial_data_path=FINANCIAL_DATA_PATH,
            benchmark_data_path=BENCHMARK_DATA_PATH,
            trading_day_path=TRADING_DAY_PATH,
            output_path=OUTPUT_PATH,
            memory_log_path=MEMORY_LOG_PATH,
            enable_memory_monitoring=ENABLE_MEMORY_MONITORING,
            frequency=FREQ
        )
import pandas as pd
import numpy as np


def calculate_frequency_net_value(
        return_df: pd.DataFrame,
        tradingday_df: pd.DataFrame,
        freq_type: str = 'W',
        based_on: str = 'natural',
        week_start: str = 'mon'
) -> pd.DataFrame:
    """
    根据指定频率和方法计算净值数据。

    Args:
        return_df (pd.DataFrame): 基金日频数据，需包含 FinProCode, EndDate, UnitNVRestored, benchmark_nv。
        tradingday_df (pd.DataFrame): 交易日历数据，需包含 TradingDate, IfTradingDay, IfWeekEnd, IfMonthEnd。
        freq_type (str): 频率类型, 'W' for weekly, 'M' for monthly。默认为 'W'。
        based_on (str): 计算依据, 'natural' for natural day, 'trade' for trading day。默认为 'natural'。
        week_start (str): 自然周的起始日, 'mon' (周一至周日), 'sun' (周日至周六), 'sat' (周六至周五)。默认为 'mon'。

    Returns:
        pd.DataFrame: 计算后的频率净值数据，包含 FinProCode, EndDate, UnitNVRestored, benchmark_nv。
    """
    # --- 数据预处理 ---
    df = return_df.copy().sort_values(by=['FinProCode', 'EndDate']).reset_index(drop=True)
    df['EndDate'] = pd.to_datetime(df['EndDate'])
    # 排序对 merge_asof 和 .last() 的正确性至关重要
    df = df.sort_values(by=['FinProCode', 'EndDate'])

    trade_cal = tradingday_df.copy().sort_values(by=['TradingDate']).reset_index(drop=True)
    trade_cal['TradingDate'] = pd.to_datetime(trade_cal['TradingDate'])
    trade_cal = trade_cal.sort_values(by='TradingDate')

    if based_on == 'natural':
        # --- 基于自然日计算 ---
        # 将 EndDate 设为索引以使用 resample
        df_indexed = df.set_index('EndDate')

        if freq_type.upper() == 'W':
            week_map = {'mon': 'W-SUN', 'sun': 'W-SAT', 'sat': 'W-FRI'}
            freq_str = week_map.get(week_start)
            if not freq_str:
                raise ValueError("week_start 参数必须是 'mon', 'sun', 'sat' 中的一个。")
            result_df = df_indexed.groupby('FinProCode').resample(freq_str).last()
        elif freq_type.upper() == 'M':
            result_df = df_indexed.groupby('FinProCode').resample('M').last()
        else:
            raise ValueError("freq_type 参数必须是 'W' 或 'M'。")

        # 清理结果：删除全为空值的行，丢弃重复的 FinProCode 列，然后重置索引
        result_df = result_df.drop(columns='FinProCode').dropna(how='all').reset_index()

    elif based_on == 'trade':
        # --- 基于交易日计算 ---
        if freq_type.upper() == 'W':
            period_ends_df = trade_cal[trade_cal['IfWeekEnd'] == 1][['TradingDate']].copy()
        elif freq_type.upper() == 'M':
            period_ends_df = trade_cal[trade_cal['IfMonthEnd'] == 1][['TradingDate']].copy()
        else:
            raise ValueError("freq_type 参数必须是 'W' 或 'M'。")

        period_ends_df = period_ends_df.rename(columns={'TradingDate': 'period_end_date'})
        period_ends_df = period_ends_df.drop_duplicates().sort_values('period_end_date')

        # 对每只基金分别进行 merge_asof，因为 period_ends_df 是通用的，不包含 FinProCode
        all_merged = []
        for fin_code, group_df in df.groupby('FinProCode'):
            merged_group = pd.merge_asof(
                group_df,
                period_ends_df,
                left_on='EndDate',
                right_on='period_end_date',
                direction='forward'
            )
            all_merged.append(merged_group)

        if not all_merged:
            merged_df = pd.DataFrame()
        else:
            merged_df = pd.concat(all_merged)

        # 对每个基金的每个周期，取最后一条记录
        result_df = merged_df.dropna(subset=['period_end_date']).groupby(
            ['FinProCode', 'period_end_date']).last().reset_index()

        # 将 EndDate 设置为周期的结束日，以实现日期对齐
        result_df['EndDate'] = result_df['period_end_date']

    else:
        raise ValueError("based_on 参数必须是 'natural' 或 'trade'。")

    # --- 格式化输出 ---
    output_cols = ['FinProCode', 'secassetcatcode', 'EndDate', 'UnitNVRestored', 'benchmark_nv']
    for col in output_cols:
        if col not in result_df.columns:
            result_df[col] = np.nan

    return result_df[output_cols].sort_values(by=['FinProCode', 'EndDate']).reset_index(drop=True)


if __name__ == '__main__':
    from A01_Scheduler import read_data

    print("--- 交易日历数据 ---")
    the_tradingday_df = pd.read_parquet('tradedate.parquet')
    print(the_tradingday_df.head())
    print("\n" + "=" * 60 + "\n")

    # ------------------------------------ 全市场理财 ------------------------------------

    print("--- 全市场理财原始日频数据 ---")
    the_return_df = read_data()
    print(the_return_df.head())
    print("\n" + "=" * 60 + "\n")

    # 计算周频净值 (交易日)
    print("--- 全市场周频净值 (交易日) ---")
    weekly_trade_df = calculate_frequency_net_value(
        the_return_df, the_tradingday_df, freq_type='W', based_on='trade'
    )
    weekly_trade_df.to_parquet('weekly_trade_df.parquet')
    print(weekly_trade_df.head())
    print("\n" + "=" * 60 + "\n")

    # 计算月频净值 (交易日)
    print("--- 全市场月频净值 (交易日) ---")
    monthly_trade_df = calculate_frequency_net_value(
        the_return_df, the_tradingday_df, freq_type='M', based_on='trade'
    )
    monthly_trade_df.to_parquet('monthly_trade_df.parquet')
    print(monthly_trade_df.head())
    print("\n" + "=" * 60 + "\n")

    del the_return_df, weekly_trade_df, monthly_trade_df  # 清理内存

    # ------------------------------------ 代销理财 ------------------------------------

    print("--- 代销理财原始日频数据 ---")
    dx_return_df = read_data('dai_xiao.parquet')
    print(dx_return_df.head())
    print("\n" + "=" * 60 + "\n")

    # 计算周频净值 (交易日)
    print("--- 代销理财周频净值 (交易日) ---")
    dx_weekly_trade_df = calculate_frequency_net_value(
        dx_return_df, the_tradingday_df, freq_type='W', based_on='trade'
    )
    dx_weekly_trade_df.to_parquet('dx_weekly_trade_df.parquet')
    print(dx_weekly_trade_df.head())
    print("\n" + "=" * 60 + "\n")

    # 计算月频净值 (交易日)
    print("--- 代销月频净值 (交易日) ---")
    dx_monthly_trade_df = calculate_frequency_net_value(
        dx_return_df, the_tradingday_df, freq_type='M', based_on='trade'
    )
    dx_monthly_trade_df.to_parquet('dx_monthly_trade_df.parquet')
    print(dx_monthly_trade_df.head())
    print("\n" + "=" * 60 + "\n")

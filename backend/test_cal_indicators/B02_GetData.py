# -*- encoding: utf-8 -*-
"""
@File: B02_GetData.py
@Modify Time: 2025/7/16 13:27
@Author: Kevin-Chen
@Descriptions: 获取全市场存续中的理财产品复权净值数据
"""
import pymysql
import traceback
import pandas as pd
from KYP.DAO import Query_Quote as query_q
from datetime import date
from datetime import datetime


def get_data(host, user, password, database, port=3306):
    """
    从MySQL数据库获取理财产品复权净值数据

    :param host: 数据库主机地址
    :param user: 用户名
    :param password: 密码
    :param database: 数据库名
    :param port: 端口号，默认为3306
    :return: 包含查询结果的 pandas DataFrame
    """
    the_sql = f"""
SELECT
    fnv.FinProCode,
    fnv.EndDate,
    fnv.UnitNVRestored,
    pt.SecAssetCatCode  -- 从 FP_JYProductType 获取的二级分类
FROM
    FP_NetValueRe fnv
JOIN
    (
    SELECT
        FinProCode,
        CASE
            WHEN ActMaturityDate IS NULL THEN MaturityDate
            ELSE ActMaturityDate
        END AS MaturityDate
    FROM FP_BasicInfo
    WHERE SecuCategory = 'FCC0000001TD'
    ) fstc
ON fnv.FinProCode = fstc.FinProCode
JOIN
    FP_JYProductType pt
    ON fnv.FinProCode = pt.FinProCode
WHERE
    fnv.SecuCategory = 'FCC0000001TD'
    AND fstc.MaturityDate is NOT NULL
    AND fstc.MaturityDate >= CURDATE()
    AND pt.IfEffected = 'FCC000000005'  -- 有效产品
    """
    connection = None
    try:
        # 创建数据库连接
        connection = pymysql.connect(host=host,
                                     user=user,
                                     password=password,
                                     database=database,
                                     port=port,
                                     )

        # 使用pandas直接执行SQL查询并返回DataFrame
        df = pd.read_sql_query(the_sql, connection)
        return df

    except Exception as e:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] An error occurred: {e}")
        print(traceback.format_exc())
        return None
    finally:
        # 确保数据库连接被关闭
        if connection:
            connection.close()


def get_benchmark_index_nv():
    """
    获取理财二级分类对应的基准。
    根据预设的 mapping_dict，为每个二级分类生成一个基准序列。
    - 对于单一基准的分类，直接采用基准指数的行情。
    - 对于复合基准的分类，根据权重合成新净值(nv)序列。
    返回结果会扩展到<自然日>维度
    """
    mapping_dict = {
        'FCC000001EMV': {  # 现金管理类
            'benchmark': [{'name': '中证货币基金指数', 'innercode': '8473', 'weight': '1'}]
        },
        'FCC000001EL5': {  # 纯债类
            'benchmark': [{'name': '中证全债', 'innercode': '6455', 'weight': '1'}]
        },
        'FCC000001EL6': {  # 债券+非标类
            'benchmark': [{'name': '中证全债', 'innercode': '6455', 'weight': '1'}]
        },
        'FCC000001EL7': {  # 固收+
            'benchmark': [{'name': '中证全债', 'innercode': '6455', 'weight': '1'}]
        },
        'FCC000001EL0': {  # 偏债混合
            'benchmark': [
                {'name': '中证800', 'innercode': '4982', 'weight': '0.3'},
                {'name': '中证全债', 'innercode': '6455', 'weight': '0.7'}
            ]
        },
        'FCC000001EL2': {  # 灵活配置
            'benchmark': [
                {'name': '中证800', 'innercode': '4982', 'weight': '0.5'},
                {'name': '中证全债', 'innercode': '6455', 'weight': '0.5'}
            ]
        },
        'FCC000001EL1': {  # 股债平衡
            'benchmark': [
                {'name': '中证800', 'innercode': '4982', 'weight': '0.5'},
                {'name': '中证全债', 'innercode': '6455', 'weight': '0.5'}
            ]
        },
        'FCC000001EKZ': {  # 偏股混合
            'benchmark': [
                {'name': '中证800', 'innercode': '4982', 'weight': '0.7'},
                {'name': '中证全债', 'innercode': '6455', 'weight': '0.3'}
            ]
        },
        'FCC000001EKW': {  # 股票类
            'benchmark': [{'name': '中证800', 'innercode': '4982', 'weight': '1'}]
        },
        'FCC000001EKX': {  # 股权类
            'benchmark': [{'name': '中证800', 'innercode': '4982', 'weight': '1'}]
        },
        'FCC000001I93': {  # 商品及金融衍生品类
            'benchmark': [{'name': '南华商品', 'innercode': '203159', 'weight': '1'}]
        },
        'FCC000001EKY': {  # 结构化产品类
            'benchmark': [{'name': '中证全债', 'innercode': '6455', 'weight': '1'}]
        }
    }

    # 1. 一次性获取所有需要的基准指数 innercode
    all_innercodes = list(set(
        b['innercode']
        for config in mapping_dict.values()
        for b in config['benchmark']
    ))

    # 2. 一次性查询所有基准指数的数据
    today_date_str = date.today().strftime('%Y-%m-%d')
    index_df = query_q.query_index_for_all('1896-01-01', today_date_str,
                                           all_innercodes,
                                           with_innercode=True)

    # 3. 重命名列，并将日期列转为 datetime 对象
    index_df = index_df.rename(columns={'bm_unitnv': 'nv', 'benchmark_gr': 'gr'})
    index_df['enddate'] = pd.to_datetime(index_df['enddate'])
    if 'innercode' in index_df.columns:
        index_df['innercode'] = index_df['innercode'].astype(str)

    # 4. 创建 innercode -> secucode 的映射字典
    inner_to_secu_map = {}
    if 'innercode' in index_df.columns and 'secucode' in index_df.columns:
        inner_to_secu_map = index_df.drop_duplicates(subset=['innercode']).set_index('innercode')['secucode'].to_dict()

    # 5. 为提高后续查询效率，设置多级索引
    index_df_indexed = index_df.set_index(['innercode', 'enddate']).sort_index()

    res_list = []

    # 6. 遍历 mapping_dict，为每个二级分类生成基准序列
    for secassetcatcode, config in mapping_dict.items():
        benchmarks = config['benchmark']

        # 使用 secucode 构造拼接字符串
        secucode_str = '+'.join([
            f"{inner_to_secu_map.get(b['innercode'], b['innercode'])}_" \
            f"{int(float(b['weight']) * 100)}" for b in benchmarks
        ])

        # 使用 innercode 构造拼接字符串
        innercode_str = '+'.join([
            f"{b['innercode']}_{int(float(b['weight']) * 100)}" for b in benchmarks
        ])

        if len(benchmarks) == 1:
            innercode = benchmarks[0]['innercode']
            df_single = index_df[index_df['innercode'] == innercode].copy()
            if df_single.empty:
                continue

            df_single['secassetcatcode'] = secassetcatcode
            if 'secucode' not in df_single.columns or df_single['secucode'].isnull().all():
                df_single['secucode'] = innercode

            res_list.append(df_single[['secassetcatcode', 'secucode', 'innercode', 'enddate', 'nv', 'gr']])
            continue

        weighted_gr_parts = []
        for b in benchmarks:
            innercode = b['innercode']
            weight = float(b['weight'])
            try:
                component_gr = index_df_indexed.loc[innercode]['gr']
                weighted_gr_parts.append(component_gr * weight)
            except KeyError:
                continue

        if not weighted_gr_parts:
            continue

        df_composite_gr = pd.concat(weighted_gr_parts, axis=1, join='outer').fillna(0)
        composite_gr = df_composite_gr.sum(axis=1)
        composite_nv = (1 + composite_gr).cumprod()

        df_result = pd.DataFrame({
            'enddate': composite_gr.index,
            'gr': composite_gr.values,
            'nv': composite_nv.values
        }).reset_index(drop=True)

        df_result['secassetcatcode'] = secassetcatcode
        df_result['secucode'] = secucode_str
        df_result['innercode'] = innercode_str

        res_list.append(df_result)

    if not res_list:
        return pd.DataFrame()

    final_df = pd.concat(res_list, ignore_index=True)
    final_df['enddate'] = pd.to_datetime(final_df['enddate']).dt.strftime('%Y-%m-%d')

    # === 按 secassetcatcode 维度分别扩展自然日 ===
    expanded_list = []
    grouped = final_df.groupby('secassetcatcode')

    for secassetcatcode, group in grouped:
        group = group.copy()
        group['enddate'] = pd.to_datetime(group['enddate'])

        start_date = group['enddate'].min()
        end_date = group['enddate'].max()
        full_dates = pd.date_range(start=start_date, end=end_date, freq='D')

        # 为每个 group 的 secucode、innercode 唯一组合补全日期
        for (secucode, innercode), sub_group in group.groupby(['secucode', 'innercode']):
            sub_group = sub_group.set_index('enddate').reindex(full_dates)
            sub_group['secassetcatcode'] = secassetcatcode
            sub_group['secucode'] = secucode
            sub_group['innercode'] = innercode
            sub_group['nv'] = sub_group['nv'].ffill()
            sub_group = sub_group.reset_index().rename(columns={'index': 'enddate'})
            expanded_list.append(sub_group)

    expanded_df = pd.concat(expanded_list, ignore_index=True)
    expanded_df['enddate'] = expanded_df['enddate'].dt.strftime('%Y-%m-%d')

    return expanded_df[['secassetcatcode', 'secucode', 'innercode', 'enddate', 'nv']]


def get_SecAssetCatCode(host, user, password, database, port=3306):
    """
    从MySQL数据库获取理财产品对应的二级分类代码

    :param host: 数据库主机地址
    :param user: 用户名
    :param password: 密码
    :param database: 数据库名
    :param port: 端口号，默认为3306
    :return: 包含查询结果的 pandas DataFrame
    """
    select_sql = f"""
SELECT
    a.FinProCode,
    b.SecuCode,
    a.SecAssetCatCode
FROM 
    FP_JYProductType a
JOIN 
    FP_SecuMain b
    ON a.FinProCode = b.FinProCode
JOIN 
    FP_BasicInfo c
    ON a.FinProCode = c.FinProCode
WHERE 
    c.SecuCategory = 'FCC0000001TD'   -- 银行理财
  AND a.IfEffected = 'FCC000000005' -- 有效
    """
    connection = None
    try:
        # 创建数据库连接
        connection = pymysql.connect(host=host,
                                     user=user,
                                     password=password,
                                     database=database,
                                     port=port,
                                     )

        # 使用pandas直接执行SQL查询并返回DataFrame
        df = pd.read_sql_query(select_sql, connection)
        return df

    except Exception as e:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] An error occurred: {e}")
        return None
    finally:
        # 确保数据库连接被关闭
        if connection:
            connection.close()


def get_trading_day_info(host, user, password, database, port=3306):
    the_sql = f"""
select 
    TradingDate, 
    IfTradingDay,
    IfWeekEnd,
    IfMonthEnd,
    IfQuarterEnd,
    IfYearEnd
from QT_TradingDayNew
where SecuMarket = '83'
    """
    connection = None
    try:
        # 创建数据库连接
        connection = pymysql.connect(host=host,
                                     user=user,
                                     password=password,
                                     database=database,
                                     port=port,
                                     )

        # 使用pandas直接执行SQL查询并返回DataFrame
        df = pd.read_sql_query(the_sql, connection)
        return df

    except Exception as e:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] An error occurred: {e}")
        print(traceback.format_exc())
        return None
    finally:
        # 确保数据库连接被关闭
        if connection:
            connection.close()


if __name__ == '__main__':
    # 数据库连接信息
    db_host = '120.76.157.156'
    db_user = 'root'
    db_password = 'Root!bi2016'
    db_name = 'sma'
    db_port = 3306

    # 银行理财净值数据
    financial_data = get_data(db_host, db_user, db_password, db_name, db_port)
    if financial_data is not None:
        print("Successfully retrieved data:")
        print(financial_data.head())
        financial_data.to_parquet('financial_data.parquet', index=False)

    res = get_benchmark_index_nv()
    print(res)
    res.to_parquet('benchmark_index_nv.parquet', index=False)

    # 交易日历信息
    trading_day_info = get_trading_day_info(db_host, db_user, db_password, db_name, db_port)
    if trading_day_info is not None:
        print("Successfully retrieved trading day info:")
        print(trading_day_info.head())
        trading_day_info.to_parquet('tradedate.parquet', index=False)

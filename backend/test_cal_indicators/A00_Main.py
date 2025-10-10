# -*- encoding: utf-8 -*-
"""
@File: A00_Main.py
@Modify Time: 2025/7/24 14:39
@Author: Kevin-Chen
@Descriptions:
"""
import os
import traceback
import numpy as np
import pandas as pd
import oracledb  # 直接使用 oracledb 库
from datetime import datetime
from sqlalchemy import create_engine


def insert_data_to_oracle_direct(df: pd.DataFrame, db_config: dict, target_table: str, staging_table: str):
    """
    使用 oracledb 库、暂存表和MERGE语句高效地将DataFrame数据插入或更新到Oracle数据库。
    (增强版：包含严格的数据清洗和调试功能)

    Args:
        df (pd.DataFrame): 待插入的数据。
        db_config (dict): 数据库连接配置。
        target_table (str): 目标Oracle表的名称。
        staging_table (str): 暂存Oracle表的名称。
    """
    # --- 1. 准备数据 ---

    print("Preparing DataFrame for database insertion...")
    df_to_insert = df.copy()

    df_to_insert.rename(columns={
        'FinProCode': 'SECUCODE',
        'EndDate': 'ENDDATE',
        'period': 'PERIOD_CODE',
        'index_code': 'INDEX_CODE',
        'index_value': 'INDEX_VALUE',
        'ranking': 'RANKING',
        'totalnum': 'TOTALNUM',
        'valuemedian': 'VALUEMEDIAN',
        'valuemean': 'VALUEMEAN'
    }, inplace=True)

    df_to_insert['ENDDATE'] = pd.to_datetime(df_to_insert['ENDDATE']).dt.strftime('%Y%m%d')
    df_to_insert['UPDATETIME'] = datetime.now().strftime('%Y%m%d%H%M%S')
    df_to_insert['STR_ID'] = df_to_insert['SECUCODE'].astype(str) + df_to_insert['PERIOD_CODE'].astype(str) + \
                             df_to_insert['INDEX_CODE'].astype(str)

    final_cols = [
        'SECUCODE', 'ENDDATE', 'PERIOD_CODE', 'INDEX_CODE', 'INDEX_VALUE',
        'RANKING', 'TOTALNUM', 'VALUEMEDIAN', 'VALUEMEAN', 'STR_ID', 'UPDATETIME'
    ]
    df_to_insert = df_to_insert[final_cols]

    # --- 数据类型转换 ---
    # 调用者应确保数据是干净的。这里只做最后的类型转换。
    print("Converting final data types for insertion...")

    # 强制转换为数值类型，将无法转换的值设为 NaN
    for col in ['INDEX_VALUE', 'VALUEMEDIAN', 'VALUEMEAN', 'RANKING', 'TOTALNUM']:
        if col in df_to_insert.columns:
            df_to_insert[col] = pd.to_numeric(df_to_insert[col], errors='coerce')

    # 将浮点数列四舍五入
    for col in ['INDEX_VALUE', 'VALUEMEDIAN', 'VALUEMEAN']:
        if col in df_to_insert.columns:
            df_to_insert[col] = df_to_insert[col].round(6)
    # 将整数列转换为可空整数
    for col in ['RANKING', 'TOTALNUM']:
        if col in df_to_insert.columns:
            df_to_insert[col] = df_to_insert[col].astype(pd.Int64Dtype())

    # 将所有 pandas 的 NA 表现形式 (NaN, NaT, <NA>) 替换为 None
    df_to_insert = df_to_insert.replace({pd.NaT: None}).fillna(np.nan).replace([np.nan], [None])

    # --- 新增：处理异常大的浮点数值，避免 ORA-01438 错误 ---
    # 设定一个合理的阈值，例如 1e15 (1后面15个零)，如果数值的绝对值超过此阈值，则视为异常并替换为 None
    # 这个阈值需要根据实际业务情况和数据库列的定义来调整
    print("Checking for extremely large/small float values and replacing with None if abnormal...")
    abnormal_value_threshold = 1e15  # 10^15

    for col in ['INDEX_VALUE', 'VALUEMEDIAN', 'VALUEMEAN']:
        if col in df_to_insert.columns:
            # 确保列是数值类型，否则跳过
            if pd.api.types.is_numeric_dtype(df_to_insert[col]):
                df_to_insert[col] = df_to_insert[col].apply(
                    lambda x: None if pd.notna(x) and (abs(x) > abnormal_value_threshold) else x
                )

    # --- 最终清洗和转换，确保数据对 oracledb 安全 ---
    # 替换无穷大值为 None
    df_to_insert.replace([np.inf, -np.inf], None, inplace=True)
    # 确保所有 NA/NaN/NaT 都被替换为 None
    df_to_insert = df_to_insert.replace({pd.NaT: None}).fillna(np.nan).replace([np.nan], [None])
    # 再次确保所有列中的空字符串都替换为 None
    for col in df_to_insert.columns:
        if df_to_insert[col].dtype == 'object':
            df_to_insert[col] = df_to_insert[col].replace('', None)

    # 将DataFrame转换为元组列表以用于 executemany
    data_tuples = [tuple(x) for x in df_to_insert.to_numpy()]

    print(f"DataFrame prepared. Shape: {df_to_insert.shape}")

    # --- 2. 数据库操作 ---
    connection = None
    try:
        print("Connecting to Oracle database...")
        connection = oracledb.connect(
            user=db_config['user'],
            password=db_config['password'],
            dsn=f"{db_config['host']}:{db_config['port']}/{db_config['sid']}"
        )
        cursor = connection.cursor()
        print("Connection successful.")

        # 步骤 a: 创建暂存表
        try:
            cursor.execute(f"DROP TABLE {staging_table}")
        except oracledb.DatabaseError as e:
            if "ORA-00942" not in str(e): raise

        create_stage_sql = f"CREATE TABLE {staging_table} AS SELECT * FROM {target_table} WHERE 1=0"
        cursor.execute(create_stage_sql)
        print(f"Staging table '{staging_table}' created.")

        # 步骤 b: 使用 executemany 批量插入
        print(f"Writing {len(data_tuples)} rows to staging table...")
        insert_sql = f"INSERT INTO {staging_table} VALUES (:1, :2, :3, :4, :5, :6, :7, :8, :9, :10, :11)"
        cursor.executemany(insert_sql, data_tuples, batcherrors=True)

        # 增强的错误调试
        batch_errors = cursor.getbatcherrors()
        if batch_errors:
            error_count = 0
            for error in batch_errors:
                error_count += 1
                print("--- Batch Insert Error ---")
                print(f"Error: {error.message}")
                print(f"Row Offset: {error.offset}")
                # 打印问题行的详细数据
                if error.offset < len(data_tuples):
                    problem_row = data_tuples[error.offset]
                    print("Problematic Data:")
                    for col_name, value in zip(final_cols, problem_row):
                        print(f"  {col_name}: {value} (Type: {type(value)})")
                print("--------------------------")
            raise Exception(f"{error_count} errors occurred during batch insert. See details above.")

        print("Staging table populated successfully.")

        # 步骤 c: 执行 MERGE
        print("Executing MERGE statement...")
        merge_sql = f"""
        MERGE INTO {target_table} T
        USING {staging_table} S
        ON (T.STR_ID = S.STR_ID)
        WHEN MATCHED THEN
            UPDATE SET
                T.INDEX_VALUE = S.INDEX_VALUE, T.RANKING = S.RANKING, T.TOTALNUM = S.TOTALNUM,
                T.VALUEMEDIAN = S.VALUEMEDIAN, T.VALUEMEAN = S.VALUEMEAN, T.UPDATETIME = S.UPDATETIME
        WHEN NOT MATCHED THEN
            INSERT ({', '.join(final_cols)})
            VALUES (S.SECUCODE, S.ENDDATE, S.PERIOD_CODE, S.INDEX_CODE, S.INDEX_VALUE, S.RANKING, S.TOTALNUM, S.VALUEMEDIAN, S.VALUEMEAN, S.STR_ID, S.UPDATETIME)
        """
        cursor.execute(merge_sql)
        print(f"MERGE statement executed. {cursor.rowcount} rows affected.")

        # 步骤 d: 提交事务
        connection.commit()
        print("Transaction committed.")

        # 步骤 e: 删除暂存表
        cursor.execute(f"DROP TABLE {staging_table}")
        print("Staging table dropped.")

    except Exception as e:
        print(f"An error occurred: {e}")
        print(traceback.format_exc())
        if connection:
            try:
                connection.rollback()
                print("Transaction rolled back.")
            except oracledb.DatabaseError as rb_e:
                print(f"Error during rollback: {rb_e}")
                print(traceback.format_exc())
    finally:
        if 'cursor' in locals() and cursor:
            cursor.close()
        if connection:
            connection.close()
            print("Database connection closed.")


if __name__ == '__main__':
    # --- 数据库配置 ---
    db_config = {
        'host': '10.20.156.94',
        'port': 1521,
        'sid': 'orcl',
        'user': 'kyp_prod30',
        'password': 'handsome'
    }

    # --- 频率配置 ---
    frequency_configs = {
        'ys': {
            'parquet_file': 'indicator_results_ys.parquet',
            'target_table': 'ADS_LC_FUND_INDICATOR'
        },
        'w': {
            'parquet_file': 'indicator_results_w.parquet',
            'target_table': 'ADS_LC_FUND_INDICATOR_W'
        },
        'm': {
            'parquet_file': 'indicator_results_m.parquet',
            'target_table': 'ADS_LC_FUND_INDICATOR_M'
        }
    }

    for freq, config in frequency_configs.items():
        parquet_file = config['parquet_file']
        target_table = config['target_table']
        staging_table = f"{target_table}_STAGE"  # 暂存表名

        print(
            f"\n--- Processing {freq.upper()} frequency data (File: {parquet_file}, Target Table: {target_table}) ---")

        # --- 读取指标结果文件 ---
        print(f"Reading indicator results from {parquet_file}...")
        try:
            df_results = pd.read_parquet(parquet_file)
            print(df_results.head())
            print(f"Loaded {len(df_results)} rows from {parquet_file}.")
        except FileNotFoundError:
            print(f"Error: Parquet file '{parquet_file}' not found. Skipping this frequency.")
            print(traceback.format_exc())
            continue
        except Exception as e:
            print(f"Error reading {parquet_file}: {e}. Skipping this frequency.")
            print(traceback.format_exc())
            continue

        # 删除包含 NaN 或 Inf 的行 ---
        numeric_cols_to_check = ['index_value', 'valuemedian', 'valuemean', 'ranking', 'totalnum']
        cols_to_check = [col for col in numeric_cols_to_check if col in df_results.columns]

        print(f"Dropping rows with NaN/inf in columns: {cols_to_check}")
        df_results.dropna(subset=cols_to_check, inplace=True)
        print(f"Cleaned DataFrame shape: {df_results.shape}")

        # --- 执行数据插入 ---
        if not df_results.empty:
            insert_data_to_oracle_direct(df_results, db_config, target_table, staging_table)
        else:
            print("DataFrame is empty after cleaning, no data to insert.")

"""ETF 数据抓取脚本（多线程 + 重试 + Parquet 输出）

使用 Tushare Pro 拉取 ETF 基本信息与净值日行情：
- 使用线程池并发抓取 `fund_nav`。
- 失败自动重试 n 次（带退避）。
- 将结果合并保存为 Parquet 文件。
- 速率限制：按每分钟不超过 80 次接口调用进行节流；若触发官方限流报错，暂停一段时间再继续。

注意：参数在 `if __name__ == "__main__":` 下方直接设置（不使用环境变量）。
"""

from __future__ import annotations

import os
import time
import pandas as pd
import tushare as ts
from collections import deque
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import require_tushare_token

# df展示多行
pd.set_option('display.max_rows', 500)
# df不换行展示
pd.set_option('expand_frame_repr', False)


def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def fetch_etf_list(output_dir, pro) -> pd.DataFrame:
    df = pro.fund_basic(market="E")
    # 规范列名 & 保留常用字段
    keep = ["ts_code", "name", "management", "found_date"]
    cols = [c for c in keep if c in df.columns]
    df = df[cols].copy() if cols else df
    df.to_parquet(os.path.join(output_dir, "etf_info_df.parquet"))
    return df


class RateLimiter:
    """简单的时间窗口限流（max_calls 次 / period 秒）。线程安全。"""

    def __init__(self, max_calls: int, period: float) -> None:
        self.max_calls = max_calls
        self.period = period
        self._dq = deque()  # 存放最近一次调用时间（monotonic 秒）

    def acquire(self) -> None:
        while True:
            now = time.monotonic()
            # 清理过期时间戳
            while self._dq and now - self._dq[0] >= self.period:
                self._dq.popleft()
            if len(self._dq) < self.max_calls:
                self._dq.append(now)
                return
            # 需要等待直到最早的一次过期
            wait = self.period - (now - self._dq[0])
            if wait > 0:
                time.sleep(wait)


def fetch_nav_once(pro, ts_code: str, limiter: RateLimiter) -> Optional[pd.DataFrame]:
    # 限流：确保不超过 80 次/分钟
    limiter.acquire()
    df = pro.fund_nav(ts_code=ts_code)
    if df is None or df.empty:
        return None
    # 只保留需要的列
    keep = ["ts_code", "nav_date", "unit_nav", "accum_nav", "adj_nav"]
    cols = [c for c in keep if c in df.columns]
    df = df[cols].copy()
    # 转换日期
    if "nav_date" in df.columns:
        df["nav_date"] = pd.to_datetime(df["nav_date"])
        df["date"] = df["nav_date"]
    return df


def fetch_nav_with_retry(
        pro,
        ts_code: str,
        name: str,
        max_retries: int,
        backoff_sec: float,
        limiter: RateLimiter,
        wait_on_rate_limit_sec: float,
) -> Optional[pd.DataFrame]:
    last_err: Optional[Exception] = None
    backoff = backoff_sec
    for attempt in range(1, max_retries + 1):
        try:
            df = fetch_nav_once(pro, ts_code, limiter)
            if df is not None and not df.empty:
                df["name"] = name
                df = df.sort_values("nav_date").reset_index(drop=True)
                return df
            # 空数据也视为失败，进行重试
            last_err = RuntimeError("empty response")
        except Exception as e:  # noqa: BLE001
            last_err = e
            # 若触发官方限流报错，等待一段时间再继续
            msg = str(last_err)
            if "每分钟最多访问该接口80次" in msg or "doc_id=108" in msg:
                print(
                    f"[INFO] 触发接口限流，对 {ts_code}-{name} 暂停 {wait_on_rate_limit_sec}s 后重试。"
                )
                time.sleep(wait_on_rate_limit_sec)
        # 重试等待
        if attempt < max_retries:
            time.sleep(backoff)
            backoff *= 2
    # 最终失败
    print(f"[WARN] 拉取 {ts_code} - {name} 失败，错误：{last_err}")
    return None


def main_nv(
        pro,
        *,
        max_workers: int = 8,
        max_retries: int = 3,
        backoff_sec: float = 1.5,
        output_dir: str = "data",
        max_calls_per_minute: int = 80,
        wait_on_rate_limit_sec: float = 10.0,
) -> None:
    ensure_output_dir(output_dir)

    limiter = RateLimiter(max_calls=max_calls_per_minute, period=60.0)
    # 1) 获取 ETF 列表
    etf_info_df = fetch_etf_list(output_dir, pro)
    # 可选：保存元数据（便于排查），受 .gitignore 保护
    try:
        etf_info_df.to_excel(os.path.join(output_dir, "etf_info_df.xlsx"), index=False)
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] 写入 ETF 基本信息 Excel 失败：{e}")

    etf_dict: Dict[str, str] = {i: j for i, j in zip(
        etf_info_df.get("ts_code", []), etf_info_df.get("name", []))}
    total = len(etf_dict)
    print(
        f"准备抓取 {total} 只 ETF 的净值数据，线程数={max_workers}，重试={max_retries}，限流≤{max_calls_per_minute}/min"
    )

    # 2) 并发抓取净值
    results: List[pd.DataFrame] = []
    failures: List[str] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_code = {
            executor.submit(
                fetch_nav_with_retry,
                pro,
                code,
                name,
                max_retries,
                backoff_sec,
                limiter,
                wait_on_rate_limit_sec,
            ): (code, name)
            for code, name in etf_dict.items()
        }
        for idx, future in enumerate(as_completed(future_to_code), 1):
            code, name = future_to_code[future]
            try:
                df = future.result()
                if df is not None and not df.empty:
                    results.append(df)
                else:
                    failures.append(code)
            except Exception as e:  # noqa: BLE001
                print(f"[WARN] 任务异常 {code}-{name}：{e}")
                failures.append(code)
            if idx % 50 == 0 or idx == total:
                print(f"进度：{idx}/{total}，成功 {len(results)}，失败 {len(failures)}")

    if not results:
        raise RuntimeError("抓取失败：未获得任何 ETF 净值数据。")

    # 3) 合并与保存
    etf_daily_df = pd.concat(results, axis=0, ignore_index=True)
    # 去重（以 ts_code + date 唯一）
    if {"ts_code", "date"}.issubset(etf_daily_df.columns):
        etf_daily_df = etf_daily_df.drop_duplicates(subset=["ts_code", "date"]).reset_index(drop=True)

    out_path = os.path.join(output_dir, "etf_daily_df.parquet")
    etf_daily_df.to_parquet(out_path, index=False)
    print(f"已保存 Parquet：{out_path}，共 {len(etf_daily_df)} 行；失败 {len(failures)} 支。")


def fetch_candle_once(pro, ts_code: str, limiter: RateLimiter) -> Optional[pd.DataFrame]:
    """
    分段拉取单个ETF的日线行情数据，以规避单次2000行的限制。
    """
    all_df = []
    end_date = pd.to_datetime("today").strftime("%Y%m%d")
    # 循环获取，每次取约8年数据（2000交易日近似值），最多取5次（40年）
    for _ in range(5):
        start_date = (pd.to_datetime(end_date) - pd.DateOffset(years=8)).strftime(
            "%Y%m%d"
        )
        limiter.acquire()
        try:
            df_chunk = pro.fund_daily(
                ts_code=ts_code, start_date=start_date, end_date=end_date
            )
        except Exception:
            # Tushare 可能会在某些情况下抛出异常而不是返回None
            df_chunk = None

        if df_chunk is None or df_chunk.empty:
            break

        all_df.append(df_chunk)

        # 如果返回的数据量小于2000，说明已经取完所有历史数据
        if len(df_chunk) < 2000:
            break

        # 准备获取更早的数据
        oldest_date_in_chunk = pd.to_datetime(df_chunk["trade_date"].min())
        end_date = (oldest_date_in_chunk - pd.DateOffset(days=1)).strftime("%Y%m%d")

    if not all_df:
        return None

    # 合并所有分块并去重
    full_df = pd.concat(all_df, ignore_index=True)
    full_df = full_df.drop_duplicates(subset=["ts_code", "trade_date"]).reset_index(
        drop=True
    )

    # 规范化
    if "trade_date" in full_df.columns:
        full_df["date"] = pd.to_datetime(full_df["trade_date"])

    return full_df


def fetch_candle_with_retry(
        pro,
        ts_code: str,
        name: str,
        max_retries: int,
        backoff_sec: float,
        limiter: RateLimiter,
        wait_on_rate_limit_sec: float,
) -> Optional[pd.DataFrame]:
    last_err: Optional[Exception] = None
    backoff = backoff_sec
    for attempt in range(1, max_retries + 1):
        try:
            df = fetch_candle_once(pro, ts_code, limiter)
            if df is not None and not df.empty:
                df["name"] = name
                df = df.sort_values("date").reset_index(drop=True)
                return df
            last_err = RuntimeError("empty response")
        except Exception as e:
            last_err = e
            msg = str(last_err)
            if (
                    "每分钟最多访问该接口80次" in msg
                    or "doc_id=108" in msg
                    or "rate limit" in msg.lower()
            ):
                print(
                    f"[INFO][CANDLE] 触发接口限流，对 {ts_code}-{name} 暂停 {wait_on_rate_limit_sec}s 后重试。"
                )
                time.sleep(wait_on_rate_limit_sec)
        if attempt < max_retries:
            time.sleep(backoff)
            backoff *= 2
    print(f"[WARN][CANDLE] 拉取 {ts_code} - {name} 失败，错误：{last_err}")
    return None


def main_candle(
        pro,
        *,
        max_workers: int = 8,
        max_retries: int = 3,
        backoff_sec: float = 1.5,
        output_dir: str = "data",
        max_calls_per_minute: int = 80,
        wait_on_rate_limit_sec: float = 10.0,
) -> None:
    ensure_output_dir(output_dir)

    limiter = RateLimiter(max_calls=max_calls_per_minute, period=60.0)

    # 1) 获取 ETF 列表 (复用已有函数)
    etf_info_df = fetch_etf_list(output_dir, pro)
    etf_dict: Dict[str, str] = {
        i: j
        for i, j in zip(
            etf_info_df.get("ts_code", []), etf_info_df.get("name", [])
        )
    }
    total = len(etf_dict)
    print(
        f"\n准备抓取 {total} 只 ETF 的日线行情(K线)数据，线程数={max_workers}，重试={max_retries}，限流≤{max_calls_per_minute}/min"
    )

    # 2) 并发抓取日线行情
    results: List[pd.DataFrame] = []
    failures: List[str] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_code = {
            executor.submit(
                fetch_candle_with_retry,
                pro,
                code,
                name,
                max_retries,
                backoff_sec,
                limiter,
                wait_on_rate_limit_sec,
            ): (code, name)
            for code, name in etf_dict.items()
        }
        for idx, future in enumerate(as_completed(future_to_code), 1):
            code, name = future_to_code[future]
            try:
                df = future.result()
                if df is not None and not df.empty:
                    results.append(df)
                else:
                    failures.append(code)
            except Exception as e:
                print(f"[WARN][CANDLE] 任务异常 {code}-{name}：{e}")
                failures.append(code)
            if idx % 50 == 0 or idx == total:
                print(
                    f"[CANDLE] 进度：{idx}/{total}，成功 {len(results)}，失败 {len(failures)}"
                )

    if not results:
        print("[WARN][CANDLE] 抓取结束：未获得任何 ETF 日线行情数据。")
        return

    # 3) 合并与保存
    etf_candle_df = pd.concat(results, axis=0, ignore_index=True)
    if {"ts_code", "trade_date"}.issubset(etf_candle_df.columns):
        etf_candle_df = etf_candle_df.drop_duplicates(
            subset=["ts_code", "trade_date"]
        ).reset_index(drop=True)

    out_path = os.path.join(output_dir, "etf_daily_candle_df.parquet")
    etf_candle_df.to_parquet(out_path, index=False)
    print(
        f"已保存日线行情 Parquet：{out_path}，共 {len(etf_candle_df)} 行；失败 {len(failures)} 支。"
    )


if __name__ == "__main__":
    # 在此处配置运行参数（不使用环境变量）
    MAX_WORKERS = 8
    MAX_RETRIES = 3
    RETRY_BACKOFF_SEC = 1.5
    OUTPUT_DIR = "data"
    token = require_tushare_token()
    ts.set_token(token)
    the_pro = ts.pro_api(token)

    # # 获取 ETF 净值数据
    # main_nv(
    #     pro=the_pro,
    #     max_workers=MAX_WORKERS,
    #     max_retries=MAX_RETRIES,
    #     backoff_sec=RETRY_BACKOFF_SEC,
    #     output_dir=OUTPUT_DIR,
    # )

    # # 获取 ETF 日线行情数据
    # main_candle(
    #     pro=the_pro,
    #     max_workers=MAX_WORKERS,
    #     max_retries=MAX_RETRIES,
    #     backoff_sec=RETRY_BACKOFF_SEC,
    #     output_dir=OUTPUT_DIR,
    # )
    #
    # 获取交易日历数据
    trade_day_df = the_pro.trade_cal(exchange='SSE', start_date='20100101', end_date=pd.to_datetime('today').strftime('%Y%m%d'))
    trade_day_df.to_parquet(os.path.join(OUTPUT_DIR, "trade_day_df.parquet"), index=False)
    print(f"已保存交易日历 Parquet：{os.path.join(OUTPUT_DIR, 'trade_day_df.parquet')}，共 {len(trade_day_df)} 行")
    df = pd.read_parquet("data/etf_daily_df.parquet")
    print(df)

    data = the_pro.stock_basic(exchange='', list_status='L',
                               fields='ts_code,symbol,name,fullname,market,exchange,area,industry,list_date,list_status')
    data.to_parquet("data/stock_basic.parquet", index=False)
    data.to_excel("data/stock_basic.xlsx", index=False)
    print(data)

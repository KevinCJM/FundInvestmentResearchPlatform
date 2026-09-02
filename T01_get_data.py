"""Tushare-only data acquisition for the research platform.

Examples:
    python T01_get_data.py --etf-info --calendar --stock-basic
    python T01_get_data.py --nav --candle --start-date 20100101
    python T01_get_data.py --smoke --all
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import tempfile
import threading
import time
import uuid
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as parquet
import tushare as ts

from backend.services.refresh_runtime import InterProcessFileLock, atomic_write_json, read_json_object
from config import read_tushare_token, require_tushare_token


DEFAULT_START_DATE = "20100101"
TODAY = pd.Timestamp.today().strftime("%Y%m%d")
API_ROW_LIMITS = {
    "fund_basic": 15000,
    "etf_basic": 5000,
    "fund_daily": 5000,
    "index_classify": 5000,
    "ths_index": 5000,
    "dc_index": 5000,
    "tdx_index": 1000,
    "index_daily": 8000,
    "sw_daily": 4000,
    "ci_daily": 4000,
    "ths_daily": 3000,
    "dc_daily": 2000,
    "tdx_daily": 3000,
    "index_global": 4000,
    "fut_index_daily": 2000,
    "index_dailybasic": 3000,
    "index_weight": 1000,
}
DEFAULT_FUND_NAV_PAGE_SIZE = 10000
DEFAULT_INCREMENTAL_BATCH_DAYS = 20
PROJECT_ROOT = Path(__file__).resolve().parent
GLOBAL_REFRESH_LOCK_PATH = PROJECT_ROOT / "data" / ".tushare_refresh.lock"
GLOBAL_REFRESH_STATE_PATH = PROJECT_ROOT / "data" / ".tushare_refresh_status.json"
PARENT_LOCK_ENV = "TUSHARE_REFRESH_LOCK_HELD_BY_PARENT"
CLI_HEARTBEAT_INTERVAL_SECONDS = 15.0


class ResponseTruncatedError(RuntimeError):
    """Raised when an API response reaches its documented single-call cap."""

FUND_BASIC_FIELDS = [
    "ts_code",
    "name",
    "management",
    "custodian",
    "fund_type",
    "found_date",
    "due_date",
    "list_date",
    "issue_date",
    "delist_date",
    "issue_amount",
    "m_fee",
    "c_fee",
    "duration_year",
    "p_value",
    "min_amount",
    "exp_return",
    "benchmark",
    "status",
    "invest_type",
    "type",
    "trustee",
    "purc_startdate",
    "redm_startdate",
    "market",
]

ETF_BASIC_FIELDS = [
    "ts_code",
    "csname",
    "extname",
    "cname",
    "index_code",
    "index_name",
    "setup_date",
    "list_date",
    "list_status",
    "exchange",
    "mgr_name",
    "custod_name",
    "mgt_fee",
    "etf_type",
]

FUND_NAV_FIELDS = [
    "ts_code",
    "ann_date",
    "nav_date",
    "unit_nav",
    "accum_nav",
    "accum_div",
    "net_asset",
    "total_netasset",
    "adj_nav",
]

FUND_DAILY_FIELDS = [
    "ts_code",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "pre_close",
    "change",
    "pct_chg",
    "vol",
    "amount",
]

FUND_COMPANY_FIELDS = [
    "name",
    "shortname",
    "short_enname",
    "province",
    "city",
    "address",
    "phone",
    "office",
    "website",
    "chairman",
    "manager",
    "reg_capital",
    "setup_date",
    "end_date",
    "employees",
    "main_business",
    "org_code",
    "credit_code",
]

STOCK_BASIC_FIELDS = [
    "ts_code",
    "symbol",
    "name",
    "fullname",
    "market",
    "exchange",
    "area",
    "industry",
    "list_date",
    "list_status",
]

INDEX_BASIC_FIELDS = [
    "ts_code",
    "name",
    "fullname",
    "market",
    "publisher",
    "index_type",
    "category",
    "base_date",
    "base_point",
    "list_date",
    "weight_rule",
    "desc",
    "exp_date",
]

ETF_INDEX_FIELDS = [
    "ts_code",
    "indx_name",
    "indx_csname",
    "pub_party_name",
    "pub_date",
    "base_date",
    "bp",
    "adj_circle",
]

INDEX_SCOPE_ACTIONS = {
    "catalog": "index_catalog",
    "domestic": "index_domestic",
    "industry": "index_industry",
    "concept": "index_concept",
    "global": "index_global",
    "futures": "index_futures",
    "valuation": "index_valuation",
    "constituents": "index_constituents",
}
INDEX_ACTIONS = set(INDEX_SCOPE_ACTIONS.values()) | {"index_coverage"}
INDEX_HISTORY_FILES = {
    "index_daily": "index_daily_df.parquet",
    "sw_daily": "index_sw_daily_df.parquet",
    "ci_daily": "index_ci_daily_df.parquet",
    "ths_daily": "index_ths_daily_df.parquet",
    "dc_daily": "index_dc_daily_df.parquet",
    "tdx_daily": "index_tdx_daily_df.parquet",
    "index_global": "index_global_daily_df.parquet",
    "fut_index_daily": "index_futures_daily_df.parquet",
    "index_dailybasic": "index_daily_basic_df.parquet",
}
INDEX_SCOPE_APIS = {
    "index_domestic": ("index_daily",),
    "index_industry": ("sw_daily", "ci_daily"),
    "index_concept": ("ths_daily", "dc_daily", "tdx_daily"),
    "index_global": ("index_global",),
    "index_futures": ("fut_index_daily",),
    "index_valuation": ("index_dailybasic",),
}
INDEX_CATALOG_APIS = ("index_classify", "ths_index", "dc_index", "tdx_index")
DATE_SCOPED_INDEX_CATALOG_APIS = {"dc_index", "tdx_index"}
INDEX_QUOTE_SOURCE = {
    "index_basic": "index_daily",
    "etf_index": "index_daily",
    "index_classify:SW": "sw_daily",
    "index_classify:CI": "ci_daily",
    "ths_index": "ths_daily",
    "dc_index": "dc_daily",
    "tdx_index": "tdx_daily",
}
VALUATION_INDEX_CODES = (
    "000001.SH", "399001.SZ", "000300.SH", "000905.SH", "000016.SH", "399005.SZ"
)

# Tushare fut_index_daily 文档提供的完整南华指数代码表。该接口运行时实际
# 要求 ts_code，因此不能像 index_global 一样通过“单日全市场”请求发现目录。
FUTURES_INDEX_UNIVERSE = (
    ("NHAI.NH", "南华农产品指数"),
    ("NHCI.NH", "南华商品指数"),
    ("NHECI.NH", "南华能化指数"),
    ("NHFI.NH", "南华黑色指数"),
    ("NHII.NH", "南华工业品指数"),
    ("NHMI.NH", "南华金属指数"),
    ("NHNFI.NH", "南华有色金属"),
    ("NHPMI.NH", "南华贵金属指数"),
    ("A.NH", "南华连大豆指数"),
    ("AG.NH", "南华沪银指数"),
    ("AL.NH", "南华沪铝指数"),
    ("AP.NH", "南华郑苹果指数"),
    ("AU.NH", "南华沪黄金指数"),
    ("BB.NH", "南华连胶合板指数"),
    ("BU.NH", "南华沪石油沥青指数"),
    ("C.NH", "南华连玉米指数"),
    ("CF.NH", "南华郑棉花指数"),
    ("CS.NH", "南华连玉米淀粉指数"),
    ("CU.NH", "南华沪铜指数"),
    ("CY.NH", "南华棉纱指数"),
    ("ER.NH", "南华郑籼稻指数"),
    ("FB.NH", "南华连纤维板指数"),
    ("FG.NH", "南华郑玻璃指数"),
    ("FU.NH", "南华沪燃油指数"),
    ("HC.NH", "南华沪热轧卷板指数"),
    ("I.NH", "南华连铁矿石指数"),
    ("J.NH", "南华连焦炭指数"),
    ("JD.NH", "南华连鸡蛋指数"),
    ("JM.NH", "南华连焦煤指数"),
    ("JR.NH", "南华郑粳稻指数"),
    ("L.NH", "南华连乙烯指数"),
    ("LR.NH", "南华郑晚籼稻指数"),
    ("M.NH", "南华连豆粕指数"),
    ("ME.NH", "南华郑甲醇指数"),
    ("NI.NH", "南华沪镍指数"),
    ("P.NH", "南华连棕油指数"),
    ("PB.NH", "南华沪铅指数"),
    ("PP.NH", "南华连聚丙烯指数"),
    ("RB.NH", "南华沪螺钢指数"),
    ("RM.NH", "南华郑菜籽粕指数"),
    ("RO.NH", "南华郑菜油指数"),
    ("RS.NH", "南华郑油菜籽指数"),
    ("RU.NH", "南华沪天胶指数"),
    ("SC.NH", "南华原油指数"),
    ("SF.NH", "南华郑硅铁指数"),
    ("SM.NH", "南华郑锰硅指数"),
    ("SN.NH", "南华沪锡指数"),
    ("SP.NH", "南华纸浆指数"),
    ("SR.NH", "南华郑白糖指数"),
    ("TA.NH", "南华郑精对苯二甲酸指数"),
    ("TC.NH", "南华郑动力煤指数"),
    ("V.NH", "南华连聚氯乙烯指数"),
    ("WR.NH", "南华沪线材指数"),
    ("WS.NH", "南华郑强麦指数"),
    ("Y.NH", "南华连豆油指数"),
    ("ZN.NH", "南华沪锌指数"),
)

ETF_INFO_COLUMNS = [
    "ts_code",
    "code",
    "name",
    "instrument_type",
    "management",
    "custodian",
    "trustee",
    "fund_type",
    "type",
    "invest_type",
    "market",
    "market_code",
    "status",
    "status_code",
    "benchmark",
    "index_code",
    "index_name",
    "issue_amount",
    "m_fee",
    "c_fee",
    "exp_return",
    "duration_year",
    "p_value",
    "min_amount",
    "list_date",
    "found_date",
    "issue_date",
    "due_date",
    "delist_date",
    "purc_startdate",
    "redm_startdate",
]


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_yyyymmdd(value: Optional[str], *, default: str) -> str:
    if not value:
        return default
    text = str(value).strip().replace("-", "")
    if len(text) != 8 or not text.isdigit():
        raise ValueError(f"日期格式应为 YYYYMMDD 或 YYYY-MM-DD: {value}")
    return text


def fields_arg(fields: Iterable[str]) -> str:
    return ",".join(fields)


def safe_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    text = str(value).strip()
    return text or None


def first_nonempty(*values: Any) -> Optional[Any]:
    for value in values:
        if safe_text(value) is not None:
            return value
    return None


def date_series(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, format="%Y%m%d", errors="coerce")


def normalise_adj_nav(series: pd.Series) -> pd.Series:
    """Keep source rows while converting unusable adjusted NAV values to null."""

    values = pd.to_numeric(series, errors="coerce")
    return values.mask(values.le(0) | values.eq(float("inf")) | values.eq(float("-inf")))


def numeric_series(series: pd.Series, *, multiply: float = 1.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce") * multiply


def iter_date_chunks(start_date: str, end_date: str, max_calendar_days: int) -> Iterable[tuple[str, str]]:
    if max_calendar_days < 1:
        raise ValueError("max_calendar_days must be positive")
    current = pd.to_datetime(start_date, format="%Y%m%d")
    end = pd.to_datetime(end_date, format="%Y%m%d")
    while current <= end:
        chunk_end = min(current + pd.Timedelta(days=max_calendar_days - 1), end)
        yield current.strftime("%Y%m%d"), chunk_end.strftime("%Y%m%d")
        current = chunk_end + pd.Timedelta(days=1)


def is_project_data_dir(path: Path) -> bool:
    project_data_dir = (Path(__file__).resolve().parent / "data").resolve()
    return path.resolve() == project_data_dir


class RateLimiter:
    """Thread-safe fixed-window limiter with minimum spacing between calls."""

    def __init__(self, max_calls: int, period: float = 60.0, min_interval_sec: float = 0.0) -> None:
        if max_calls < 1:
            raise ValueError("max_calls must be positive")
        if min_interval_sec < 0:
            raise ValueError("min_interval_sec must not be negative")
        self.max_calls = max_calls
        self.period = period
        self.min_interval_sec = min_interval_sec
        self._calls: deque[float] = deque()
        self._next_allowed_at = 0.0
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                while self._calls and now - self._calls[0] >= self.period:
                    self._calls.popleft()
                window_wait = 0.0
                if len(self._calls) >= self.max_calls:
                    window_wait = self.period - (now - self._calls[0])
                interval_wait = self._next_allowed_at - now
                wait = max(window_wait, interval_wait, 0.0)
                if wait <= 0:
                    self._calls.append(now)
                    self._next_allowed_at = now + self.min_interval_sec
                    return
            time.sleep(wait)


def ensure_response_not_truncated(df: pd.DataFrame, api_name: str, context: str) -> None:
    row_limit = API_ROW_LIMITS.get(api_name)
    if row_limit is not None and len(df) >= row_limit:
        raise ResponseTruncatedError(
            f"{context} 返回 {len(df)} 行，达到 {api_name} 单次上限 {row_limit}，"
            "结果可能被截断；请缩小日期范围或增加分区条件。"
        )


def call_tushare_api(
    func: Callable[..., pd.DataFrame],
    limiter: RateLimiter,
    *,
    max_retries: int,
    backoff_sec: float,
    wait_on_rate_limit_sec: float,
    context: str,
    api_name: Optional[str] = None,
    retry_jitter_sec: float = 0.25,
    allow_capped_response: bool = False,
    **kwargs: Any,
) -> pd.DataFrame:
    last_err: Optional[Exception] = None
    delay = backoff_sec
    for attempt in range(1, max_retries + 1):
        try:
            limiter.acquire()
            df = func(**kwargs)
            if df is None:
                return pd.DataFrame()
            if not allow_capped_response:
                ensure_response_not_truncated(df, api_name or context.split(" ", 1)[0], context)
            return df
        except Exception as exc:  # noqa: BLE001
            if isinstance(exc, ResponseTruncatedError):
                raise
            last_err = exc
            msg = str(exc).lower()
            is_rate_limit = (
                "rate limit" in msg
                or "每分钟最多" in str(exc)
                or "doc_id=108" in msg
                or "访问频次" in str(exc)
                or "访问频率" in str(exc)
            )
            is_permission_error = any(
                marker in msg for marker in ("permission", "no privilege", "not authorized")
            ) or any(marker in str(exc) for marker in ("权限不足", "无权限", "积分不足"))
            if is_permission_error:
                raise PermissionError(f"{context} 权限不足: {exc}") from exc
            if attempt >= max_retries:
                break
            base_wait = max(wait_on_rate_limit_sec, delay) if is_rate_limit else delay
            jitter = random.uniform(0.0, max(retry_jitter_sec, 0.0))
            wait = base_wait + jitter
            reason = "触发限流" if is_rate_limit else "请求异常"
            print(f"[INFO] {context} {reason}，等待 {wait:.2f}s 后进行第 {attempt + 1}/{max_retries} 次尝试。")
            time.sleep(wait)
            delay *= 2
    raise RuntimeError(f"{context} 请求失败: {last_err}") from last_err


def _empty_response_retry_delay(args: argparse.Namespace, attempt: int) -> float:
    """Return a bounded exponential delay for confirming an empty response."""

    base = max(float(getattr(args, "backoff_sec", 1.5)), 0.0) * (2**attempt)
    jitter = random.uniform(0.0, max(float(getattr(args, "retry_jitter_sec", 0.25)), 0.0))
    return base + jitter


def fetch_with_empty_confirmation(
    fetch: Callable[[], pd.DataFrame],
    *,
    args: argparse.Namespace,
    context: str,
) -> pd.DataFrame:
    """Confirm an empty Tushare response before treating it as legitimate.

    Exception retries remain owned by :func:`call_tushare_api`.  This separate,
    bounded loop covers the common upstream failure mode where the request is
    successful but unexpectedly returns no rows.
    """

    retries = max(int(getattr(args, "empty_response_retries", 1)), 0)
    for attempt in range(retries + 1):
        frame = fetch()
        if frame is not None and not frame.empty:
            return frame
        if attempt >= retries:
            return pd.DataFrame() if frame is None else frame
        delay = _empty_response_retry_delay(args, attempt)
        print(
            f"[INFO] {context} 返回空结果，等待 {delay:.2f}s 后进行"
            f"第 {attempt + 2}/{retries + 1} 次确认。"
        )
        time.sleep(delay)
    return pd.DataFrame()


def fetch_fund_basic(
    pro: Any,
    limiter: RateLimiter,
    args: argparse.Namespace,
    *,
    market: str = "E",
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for status in ("L", "I", "D"):
        page_size = API_ROW_LIMITS["fund_basic"]
        max_pages = getattr(args, "max_fund_basic_pages", 20)
        seen_codes: set[str] = set()
        offset = 0
        for page in range(max_pages):
            context = f"fund_basic market={market} status={status} offset={offset}"
            frame = fetch_with_empty_confirmation(
                lambda: call_tushare_api(
                    pro.fund_basic,
                    limiter,
                    max_retries=args.max_retries,
                    backoff_sec=args.backoff_sec,
                    wait_on_rate_limit_sec=args.wait_on_rate_limit_sec,
                    retry_jitter_sec=getattr(args, "retry_jitter_sec", 0.25),
                    context=context,
                    api_name="fund_basic",
                    allow_capped_response=True,
                    market=market,
                    status=status,
                    offset=offset,
                    limit=page_size,
                    fields=fields_arg(FUND_BASIC_FIELDS),
                ),
                args=args,
                context=context,
            )
            if len(frame) > page_size:
                raise RuntimeError(f"fund_basic 分页返回 {len(frame)} 行，超过请求上限 {page_size}。")
            if frame.empty:
                break
            page_codes = set(frame["ts_code"].dropna().astype(str)) if "ts_code" in frame.columns else set()
            if page > 0 and page_codes and page_codes.issubset(seen_codes):
                raise RuntimeError("fund_basic 分页未向前推进，疑似服务端忽略 offset；停止以避免无限循环。")
            seen_codes.update(page_codes)
            frames.append(frame)
            print(
                f"[INFO] fund_basic market={market} status={status} "
                f"分页 {page + 1}，offset={offset}，{len(frame)} 行。"
            )
            offset += len(frame)
        else:
            raise RuntimeError(
                f"fund_basic market={market} status={status} 达到最大分页数 {max_pages}，"
                "为避免遗漏或无限循环，本批次停止。"
            )
    if not frames:
        return pd.DataFrame()
    clean_frames = [frame.dropna(axis=1, how="all") for frame in frames]
    return pd.concat(clean_frames, ignore_index=True).drop_duplicates(subset=["ts_code"]).reset_index(drop=True)


def fetch_etf_basic(pro: Any, limiter: RateLimiter, args: argparse.Namespace) -> pd.DataFrame:
    try:
        frames: list[pd.DataFrame] = []
        for exchange in ("SH", "SZ"):
            for list_status in ("L", "P", "D"):
                frame = call_tushare_api(
                    pro.etf_basic,
                    limiter,
                    max_retries=args.max_retries,
                    backoff_sec=args.backoff_sec,
                    wait_on_rate_limit_sec=args.wait_on_rate_limit_sec,
                    retry_jitter_sec=getattr(args, "retry_jitter_sec", 0.25),
                    context=f"etf_basic exchange={exchange} list_status={list_status}",
                    api_name="etf_basic",
                    exchange=exchange,
                    list_status=list_status,
                    fields=fields_arg(ETF_BASIC_FIELDS),
                )
                if not frame.empty:
                    frames.append(frame)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True).drop_duplicates(subset=["ts_code"]).reset_index(drop=True)
    except ResponseTruncatedError:
        raise
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] etf_basic 获取失败，仅使用 fund_basic 字段: {exc}")
        return pd.DataFrame()


def build_etf_info_df(fund_df: pd.DataFrame, etf_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Build the local ETF info schema from Tushare fund_basic/etf_basic data."""

    if fund_df is None or fund_df.empty:
        return pd.DataFrame(columns=ETF_INFO_COLUMNS)

    working = fund_df.copy()
    working.columns = [str(col) for col in working.columns]
    if "ts_code" not in working.columns:
        raise ValueError("fund_basic response must include ts_code")

    if etf_df is not None and not etf_df.empty and "ts_code" in etf_df.columns:
        extra = etf_df.copy()
        extra.columns = [str(col) for col in extra.columns]
        keep = [col for col in ETF_BASIC_FIELDS if col in extra.columns]
        extra = extra[keep].drop_duplicates(subset=["ts_code"])
        working = working.merge(extra, on="ts_code", how="left", suffixes=("", "_etf"))

    out = pd.DataFrame(index=working.index)
    out["ts_code"] = working["ts_code"].astype(str).str.strip()
    out["code"] = out["ts_code"].str.split(".", regex=False).str[0]
    out["instrument_type"] = "etf"

    name_candidates = [working.get("name"), working.get("csname"), working.get("cname"), working.get("extname")]
    out["name"] = name_candidates[0]
    for candidate in name_candidates[1:]:
        if candidate is not None:
            out["name"] = out["name"].where(out["name"].apply(safe_text).notna(), candidate)

    out["management"] = working.get("management")
    if "mgr_name" in working.columns:
        out["management"] = out["management"].where(out["management"].apply(safe_text).notna(), working["mgr_name"])

    out["custodian"] = working.get("custodian")
    if "custod_name" in working.columns:
        out["custodian"] = out["custodian"].where(out["custodian"].apply(safe_text).notna(), working["custod_name"])
    out["trustee"] = working.get("trustee")
    out["fund_type"] = working.get("fund_type")
    if "etf_type" in working.columns:
        out["fund_type"] = out["fund_type"].where(out["fund_type"].apply(safe_text).notna(), working["etf_type"])
    out["type"] = working.get("type")
    out["invest_type"] = working.get("invest_type")

    code_suffix = working["ts_code"].astype(str).str.split(".", regex=False).str[-1]
    market_code = working.get("exchange")
    if market_code is None:
        market_code = code_suffix
    else:
        market_code = market_code.where(market_code.apply(safe_text).notna(), code_suffix)
    out["market_code"] = market_code
    out["market"] = pd.Series(market_code, index=working.index).map(
        {
            "SH": "上交所",
            "SSE": "上交所",
            "SZ": "深交所",
            "SZSE": "深交所",
            "BJ": "北交所",
            "BSE": "北交所",
            "E": "场内",
        }
    )
    if "market" in working.columns:
        fallback_market = working["market"].map({"E": "场内", "O": "场外"}).fillna(working["market"])
        out["market"] = out["market"].where(out["market"].apply(safe_text).notna(), fallback_market)

    status_code = working.get("list_status")
    if status_code is None:
        status_code = working.get("status")
    elif "status" in working.columns:
        status_code = status_code.where(status_code.apply(safe_text).notna(), working["status"])
    out["status_code"] = status_code
    out["status"] = pd.Series(status_code, index=working.index).map(
        {
            "L": "上市交易",
            "I": "发行中",
            "P": "待上市",
            "D": "摘牌",
        }
    )
    if "status" in working.columns:
        out["status"] = out["status"].where(out["status"].apply(safe_text).notna(), working["status"])

    benchmark = working.get("benchmark")
    if benchmark is None:
        benchmark = pd.Series([None] * len(working), index=working.index)
    if "index_name" in working.columns:
        benchmark = benchmark.where(benchmark.apply(safe_text).notna(), working["index_name"])
    out["benchmark"] = benchmark
    out["index_code"] = working.get("index_code")
    out["index_name"] = working.get("index_name")

    numeric_cols = ["m_fee", "c_fee", "exp_return", "duration_year", "p_value", "min_amount"]
    for col in numeric_cols:
        out[col] = numeric_series(working[col]) if col in working.columns else pd.NA
    if "issue_amount" in working.columns:
        out["issue_amount"] = numeric_series(working["issue_amount"], multiply=10000.0)
    else:
        out["issue_amount"] = pd.NA

    date_cols = [
        "list_date",
        "found_date",
        "issue_date",
        "due_date",
        "delist_date",
        "purc_startdate",
        "redm_startdate",
    ]
    for col in date_cols:
        source = working.get(col)
        if source is None and col == "found_date" and "setup_date" in working.columns:
            source = working["setup_date"]
        out[col] = date_series(source) if source is not None else pd.NaT

    out = out[ETF_INFO_COLUMNS].drop_duplicates(subset=["ts_code"]).reset_index(drop=True)
    out = out.sort_values(["market", "ts_code"], na_position="last").reset_index(drop=True)
    return out


def build_public_fund_info_df(fund_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize off-exchange public funds without treating them as tradable ETFs."""

    out = build_etf_info_df(fund_df)
    if out.empty:
        return out
    out["instrument_type"] = "fund"
    out["market_code"] = "O"
    out["market"] = "场外"
    status_labels = {"L": "存续", "I": "发行中", "D": "到期/终止"}
    out["status"] = out["status_code"].map(status_labels).fillna(out["status"])
    return out


def filter_fund_basic_to_etfs(fund_df: pd.DataFrame, etf_df: pd.DataFrame) -> pd.DataFrame:
    """Use Tushare etf_basic as the authoritative ETF universe."""

    if etf_df is None or etf_df.empty or "ts_code" not in etf_df.columns:
        raise ValueError("etf_basic response is required to identify ETF universe")
    if fund_df is None or fund_df.empty or "ts_code" not in fund_df.columns:
        raise ValueError("fund_basic response must include ts_code")
    etf_codes = set(etf_df["ts_code"].dropna().astype(str))
    return fund_df[fund_df["ts_code"].astype(str).isin(etf_codes)].copy()


def save_dataframe(
    df: pd.DataFrame,
    path: Path,
    *,
    excel_path: Optional[Path] = None,
    quiet: bool = False,
) -> None:
    ensure_output_dir(path.parent)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        df.to_parquet(temp_path, index=False)
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)
    if not quiet:
        print(f"[OK] 保存 {path}，{len(df)} 行。")
    if excel_path is not None:
        try:
            df.to_excel(excel_path, index=False)
            print(f"[OK] 保存 {excel_path}，{len(df)} 行。")
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Excel 写入失败 {excel_path}: {exc}")


def history_checkpoint_dir(out_path: Path, args: argparse.Namespace) -> Path:
    """Keep restartable shards isolated by dataset and requested date range."""

    return out_path.parent / f".{out_path.stem}_parts_{args.start_date}_{args.end_date}"


def history_checkpoint_paths(checkpoint_dir: Path, ts_code: str) -> tuple[Path, Path]:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", ts_code):
        raise ValueError(f"不安全的 Tushare 代码，无法创建检查点: {ts_code!r}")
    return checkpoint_dir / f"{ts_code}.parquet", checkpoint_dir / f"{ts_code}.empty"


def mark_empty_checkpoint(path: Path) -> None:
    """Atomically remember a legitimate no-data response."""

    ensure_output_dir(path.parent)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write("no data\n")
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def consolidate_parquet_parts(
    part_paths: list[Path],
    out_path: Path,
    *,
    base_path: Optional[Path] = None,
    primary_column: str = "ts_code",
) -> int:
    """Merge per-instrument checkpoints into one sorted, atomic parquet."""

    sources = ([base_path] if base_path is not None and base_path.exists() else []) + part_paths
    if not sources:
        raise RuntimeError(f"{out_path.name} 没有可合并的数据分片。")

    def normalise_key(value: Any) -> str:
        if value is None or bool(pd.isna(value)):
            raise ValueError(f"{primary_column} 不允许为空，无法安全流式归并。")
        return str(value)

    base_source: Optional[Path] = None
    part_sources: list[tuple[str, Path]] = []
    output_schema: Optional[pa.Schema] = None
    for source_value in sources:
        source = Path(source_value)
        source_file = parquet.ParquetFile(source)
        schema = source_file.schema_arrow.remove_metadata()
        if primary_column not in schema.names:
            raise ValueError(f"{source.name} 缺少 {primary_column}，无法流式归并。")
        output_schema = schema if output_schema is None else _union_arrow_schema(output_schema, schema)
        if base_path is not None and source == Path(base_path):
            base_source = source
            continue
        first_key: Optional[str] = None
        for batch in source_file.iter_batches(
            batch_size=1,
            columns=[primary_column],
            use_threads=False,
        ):
            if batch.num_rows:
                first_key = normalise_key(batch.column(0)[0].as_py())
                break
        if first_key is not None:
            part_sources.append((first_key, source))

    if output_schema is None:
        raise RuntimeError(f"{out_path.name} 的所有数据分片均为空。")
    part_sources.sort(key=lambda item: item[0])
    part_keys = [key for key, _path in part_sources]
    if len(part_keys) != len(set(part_keys)):
        raise ValueError("检查点包含重复 ts_code，拒绝生成含重复代码的基线文件。")

    ensure_output_dir(out_path.parent)
    fd, temp_name = tempfile.mkstemp(prefix=f".{out_path.name}.", suffix=".tmp", dir=out_path.parent)
    os.close(fd)
    temp_path = Path(temp_name)
    writer: Optional[parquet.ParquetWriter] = None
    row_count = 0
    part_index = 0
    try:
        writer = parquet.ParquetWriter(temp_path, output_schema, compression="snappy")

        def write_table(table: pa.Table) -> None:
            nonlocal row_count
            aligned = _align_arrow_table(table.replace_schema_metadata(None), output_schema)
            writer.write_table(aligned)
            row_count += aligned.num_rows

        def write_part(path: Path, expected_key: str) -> None:
            source_file = parquet.ParquetFile(path)
            observed = False
            for batch in source_file.iter_batches(batch_size=16_384, use_threads=False):
                table = pa.Table.from_batches([batch])
                keys = {normalise_key(value) for value in table.column(primary_column).to_pylist()}
                if keys != {expected_key}:
                    raise ValueError(f"检查点 {path.name} 混入多个 ts_code: {sorted(keys)[:3]}")
                observed = True
                write_table(table)
            if not observed:
                raise ValueError(f"检查点 {path.name} 为空。")

        def write_parts_before(target_key: Optional[str]) -> None:
            nonlocal part_index
            while part_index < len(part_sources):
                key, path = part_sources[part_index]
                if target_key is not None and key >= target_key:
                    break
                write_part(path, key)
                part_index += 1

        active_base_key: Optional[str] = None
        if base_source is not None:
            source_file = parquet.ParquetFile(base_source)
            for batch in source_file.iter_batches(batch_size=16_384, use_threads=False):
                table = pa.Table.from_batches([batch])
                keys = [normalise_key(value) for value in table.column(primary_column).to_pylist()]
                run_start = 0
                for index in range(1, len(keys) + 1):
                    if index < len(keys) and keys[index] == keys[run_start]:
                        continue
                    run_key = keys[run_start]
                    if active_base_key is not None and run_key < active_base_key:
                        raise ValueError(f"{base_source.name} 未按 ts_code 连续排序，拒绝增量归并。")
                    if run_key != active_base_key:
                        write_parts_before(run_key)
                        if part_index < len(part_sources) and part_sources[part_index][0] == run_key:
                            raise ValueError(f"检查点与基线重复包含 ts_code={run_key}。")
                        active_base_key = run_key
                    write_table(table.slice(run_start, index - run_start))
                    run_start = index
        write_parts_before(None)
        if row_count == 0:
            raise RuntimeError(f"{out_path.name} 的所有数据分片均为空。")
        writer.close()
        writer = None
        metadata = parquet.ParquetFile(temp_path).metadata
        if metadata.num_rows != row_count:
            raise RuntimeError(
                f"{out_path.name} 合并行数校验失败: metadata={metadata.num_rows}, expected={row_count}"
            )
        os.replace(temp_path, out_path)
    finally:
        if writer is not None:
            writer.close()
        temp_path.unlink(missing_ok=True)
    print(f"[OK] 有序流式合并 {out_path}，{row_count} 行，来源分片 {len(part_paths)} 个。")
    return row_count


def save_full_history_with_checkpoints(
    *,
    universe: pd.DataFrame,
    out_path: Path,
    label: str,
    fetcher: Callable[[str, str], Optional[pd.DataFrame]],
    duplicate_subset: list[str],
    sort_cols: list[str],
    args: argparse.Namespace,
) -> None:
    """Fetch one instrument per shard, resume safely, then stream to final parquet."""

    universe = (
        universe[["ts_code", "name"]]
        .dropna(subset=["ts_code"])
        .drop_duplicates(subset=["ts_code"])
        .sort_values("ts_code")
        .reset_index(drop=True)
    )
    checkpoint_dir = history_checkpoint_dir(out_path, args)
    ensure_output_dir(checkpoint_dir)

    pending: list[tuple[str, str]] = []
    part_by_code: dict[str, Path] = {}
    empty_by_code: dict[str, Path] = {}
    for code_value, name_value in universe.itertuples(index=False, name=None):
        code = str(code_value)
        part_path, empty_path = history_checkpoint_paths(checkpoint_dir, code)
        part_by_code[code] = part_path
        empty_by_code[code] = empty_path
        if part_path.exists() or empty_path.exists():
            continue
        pending.append((code, str(name_value)))

    total = len(universe)
    successful = sum(path.exists() for path in part_by_code.values())
    no_data = sum(
        empty_by_code[code].exists() for code, part_path in part_by_code.items() if not part_path.exists()
    )
    print(
        f"[INFO] {label}：universe={total}，已成功={successful}，"
        f"已确认无数据={no_data}，待抓取={len(pending)}，workers={args.max_workers}。"
    )

    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        pending_iter = iter(pending)
        futures: dict[Any, str] = {}

        def submit_next() -> bool:
            try:
                code, name = next(pending_iter)
            except StopIteration:
                return False
            futures[executor.submit(fetcher, code, name)] = code
            return True

        for _ in range(min(len(pending), args.max_workers * 2)):
            submit_next()

        processed_this_run = 0
        while futures:
            completed_futures, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in completed_futures:
                code = futures.pop(future)
                part_path = part_by_code[code]
                empty_path = empty_by_code[code]
                try:
                    frame = future.result()
                    if frame is not None and not frame.empty:
                        frame = (
                            frame.drop_duplicates(subset=duplicate_subset)
                            .sort_values(sort_cols)
                            .reset_index(drop=True)
                        )
                        save_dataframe(frame, part_path, quiet=True)
                        empty_path.unlink(missing_ok=True)
                        successful += 1
                    else:
                        mark_empty_checkpoint(empty_path)
                        no_data += 1
                except Exception as exc:  # noqa: BLE001
                    errors.append(code)
                    print(f"[WARN] {label} {code} 失败: {exc}")
                processed_this_run += 1
                processed = total - len(pending) + processed_this_run
                if processed_this_run % 50 == 0 or processed_this_run == len(pending):
                    print(
                        f"[INFO] {label} 进度 {processed}/{total}，成功 {successful}，"
                        f"无数据 {no_data}，本轮异常 {len(errors)}。"
                    )
                submit_next()

    if errors:
        raise RuntimeError(
            f"{label} 有 {len(errors)} 只请求异常；已保留检查点，重启将仅重试失败项: "
            f"{', '.join(errors[:10])}"
        )

    part_paths = [part_by_code[str(code)] for code in universe["ts_code"] if part_by_code[str(code)].exists()]
    completed = len(part_paths) + sum(
        empty_by_code[str(code)].exists()
        for code in universe["ts_code"]
        if not part_by_code[str(code)].exists()
    )
    if completed != total:
        raise RuntimeError(f"{label} 检查点不完整: completed={completed}, expected={total}")
    if not part_paths:
        if args.missing_only and out_path.exists():
            print(f"[WARN] {label} 未获得新增数据，保留现有文件不变。")
            return
        raise RuntimeError(f"{label} 未获得任何历史数据。")

    base_path = out_path if args.missing_only and out_path.exists() else None
    consolidate_parquet_parts(part_paths, out_path, base_path=base_path)


def filter_missing_universe(universe: pd.DataFrame, existing_path: Path, *, label: str) -> pd.DataFrame:
    if not existing_path.exists():
        print(f"[INFO] {label} 增量文件不存在，改为全量抓取。")
        return universe
    source = parquet.ParquetFile(existing_path)
    if "ts_code" not in source.schema.names:
        raise ValueError(f"{existing_path.name} 缺少 ts_code，无法判断缺失代码。")
    existing_codes: set[str] = set()
    for batch in source.iter_batches(batch_size=65_536, columns=["ts_code"], use_threads=False):
        existing_codes.update(
            str(value)
            for value in batch.column(0).to_pylist()
            if value is not None and str(value).strip()
        )
    missing = universe[~universe["ts_code"].astype(str).isin(existing_codes)].copy()
    print(
        f"[INFO] {label} 增量：universe={len(universe)}，"
        f"已落盘代码={len(existing_codes)}，待补={len(missing)}。"
    )
    return missing


def merge_existing_rows(
    new_df: pd.DataFrame,
    existing_path: Path,
    *,
    subset: list[str],
    sort_cols: list[str],
) -> pd.DataFrame:
    if not existing_path.exists():
        return new_df
    existing = pd.read_parquet(existing_path)
    merged = pd.concat([existing, new_df], ignore_index=True)
    merged = merged.drop_duplicates(subset=subset).sort_values(sort_cols).reset_index(drop=True)
    print(f"[INFO] 合并 {existing_path.name}: old={len(existing)}，new={len(new_df)}，merged={len(merged)}。")
    return merged


def _parse_date_scalar(value: Any) -> pd.Timestamp | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, (datetime, pd.Timestamp)):
        parsed = pd.to_datetime(value, errors="coerce")
    else:
        text = str(value).strip()
        normalized = text.replace("-", "")
        if len(normalized) >= 8 and normalized[:8].isdigit():
            parsed = pd.to_datetime(normalized[:8], format="%Y%m%d", errors="coerce")
        else:
            parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return None
    return pd.Timestamp(parsed).normalize()


def _latest_date_from_series(values: pd.Series) -> pd.Timestamp | None:
    if values.empty:
        return None
    if pd.api.types.is_datetime64_any_dtype(values):
        parsed = pd.to_datetime(values, errors="coerce")
    else:
        normalized = values.astype(str).str.strip().str.replace("-", "", regex=False).str[:8]
        parsed = pd.to_datetime(normalized, format="%Y%m%d", errors="coerce")
    parsed = parsed.dropna()
    return None if parsed.empty else pd.Timestamp(parsed.max()).normalize()


def latest_parquet_date(path: Path, column: str) -> pd.Timestamp:
    """Read row-group statistics first and scan only groups lacking date stats."""

    if not path.exists():
        raise FileNotFoundError(
            f"增量更新要求已有基线文件 {path}；请先使用非 --latest 模式完成一次全量初始化。"
        )
    source = parquet.ParquetFile(path)
    if column not in source.schema_arrow.names or source.metadata.num_rows == 0:
        raise ValueError(f"{path.name} 缺少可用于增量更新的日期列 {column}。")
    column_index = source.schema_arrow.names.index(column)
    latest: pd.Timestamp | None = None
    for row_group_index in range(source.metadata.num_row_groups):
        statistics = source.metadata.row_group(row_group_index).column(column_index).statistics
        candidate = _parse_date_scalar(statistics.max) if statistics is not None and statistics.has_min_max else None
        if candidate is None:
            for batch in source.iter_batches(
                batch_size=16_384,
                row_groups=[row_group_index],
                columns=[column],
                use_threads=False,
            ):
                batch_candidate = _latest_date_from_series(batch.column(0).to_pandas())
                if batch_candidate is not None and (candidate is None or batch_candidate > candidate):
                    candidate = batch_candidate
        if candidate is not None and (latest is None or candidate > latest):
            latest = candidate
    if latest is None:
        raise ValueError(f"{path.name}.{column} 没有有效日期，无法安全增量更新。")
    return latest


def _union_arrow_schema(existing: pa.Schema, incoming: pa.Schema) -> pa.Schema:
    fields = list(existing)
    known = set(existing.names)
    fields.extend(field for field in incoming if field.name not in known)
    return pa.schema(fields)


def _align_arrow_table(table: pa.Table, schema: pa.Schema) -> pa.Table:
    arrays = []
    for field in schema:
        if field.name not in table.column_names:
            arrays.append(pa.nulls(table.num_rows, type=field.type))
            continue
        column = table.column(field.name)
        if column.type != field.type:
            column = column.cast(field.type, safe=False)
        arrays.append(column)
    return pa.Table.from_arrays(arrays, schema=schema)


def append_incremental_rows(
    new_df: pd.DataFrame,
    existing_path: Path,
    *,
    subset: list[str],
    sort_cols: list[str],
    date_column: str,
) -> int:
    """Atomically upsert incremental rows while preserving global sort order.

    ``new_df`` is allowed to overlap the historical parquet so delayed records
    and upstream revisions can be recovered.  An incoming row replaces the old
    row with the same ``subset`` key; every other historical row is retained.
    The historical parquet is consumed in 16K Arrow batches and only one
    instrument is buffered at a time, so memory does not grow with the complete
    history.
    """

    if new_df.empty:
        return 0
    required_columns = set(subset) | set(sort_cols) | {date_column}
    missing_columns = sorted(required_columns - set(new_df.columns))
    if missing_columns:
        raise ValueError(f"增量数据缺少归并列: {', '.join(missing_columns)}。")
    incoming = new_df.drop_duplicates(subset=subset, keep="last").copy()
    incoming_dates = incoming[date_column].apply(_parse_date_scalar)
    if incoming_dates.isna().any():
        raise ValueError(f"增量数据的 {date_column} 包含无效日期。")
    incoming[date_column] = incoming_dates
    incoming = incoming.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    if not existing_path.exists():
        save_dataframe(incoming, existing_path)
        return len(incoming)

    source = parquet.ParquetFile(existing_path)
    missing_existing_columns = sorted(required_columns - set(source.schema_arrow.names))
    if missing_existing_columns:
        raise ValueError(
            f"{existing_path.name} 缺少归并列: {', '.join(missing_existing_columns)}。"
        )
    incoming_schema = pa.Schema.from_pandas(incoming, preserve_index=False).remove_metadata()
    output_schema = _union_arrow_schema(source.schema_arrow.remove_metadata(), incoming_schema)
    output_date_type = output_schema.field(date_column).type
    primary_column = sort_cols[0] if sort_cols else None
    if not primary_column or primary_column not in output_schema.names:
        raise ValueError("增量流式归并要求 sort_cols 的首列存在于 parquet schema。")
    schema_changed = not output_schema.equals(
        source.schema_arrow.remove_metadata(), check_metadata=False
    )

    def normalise_key(value: Any) -> str:
        if value is None:
            raise ValueError(f"{primary_column} 不允许为空，无法安全流式归并。")
        return str(value)

    def normalise_frame_dates(frame: pd.DataFrame, *, for_storage: bool) -> pd.DataFrame:
        result = frame.copy()
        parsed = result[date_column].apply(_parse_date_scalar)
        if parsed.isna().any():
            raise ValueError(f"{existing_path.name} 的 {date_column} 包含无效日期。")
        parsed = pd.to_datetime(parsed)
        if not for_storage or pa.types.is_timestamp(output_date_type):
            result[date_column] = parsed
        elif pa.types.is_string(output_date_type) or pa.types.is_large_string(output_date_type):
            result[date_column] = parsed.dt.strftime("%Y%m%d")
        elif pa.types.is_integer(output_date_type):
            result[date_column] = parsed.dt.strftime("%Y%m%d").astype("int64")
        elif pa.types.is_date(output_date_type):
            result[date_column] = parsed.dt.date
        else:
            raise ValueError(
                f"{existing_path.name} 的 {date_column} 使用不支持的 Arrow 类型: {output_date_type}。"
            )
        return result

    def incoming_matches_existing() -> bool:
        """Check the small overlap with parquet filters before rewriting history."""

        if schema_changed:
            return False
        filter_values = sorted(
            {normalise_key(value) for value in incoming[primary_column].tolist()}
        )
        try:
            matching = parquet.read_table(
                existing_path,
                columns=output_schema.names,
                filters=[(primary_column, "in", filter_values)],
                use_threads=True,
            ).to_pandas()
        except (OSError, TypeError, ValueError, pa.ArrowException):
            return False
        if matching.empty:
            return False
        matching = normalise_frame_dates(matching, for_storage=False)
        candidate = normalise_frame_dates(incoming, for_storage=False)
        matching[primary_column] = matching[primary_column].map(normalise_key)
        candidate[primary_column] = candidate[primary_column].map(normalise_key)
        existing_indexed = matching.set_index(subset)
        incoming_indexed = candidate.set_index(subset)
        if existing_indexed.index.has_duplicates or not incoming_indexed.index.isin(
            existing_indexed.index
        ).all():
            return False
        existing_rows = existing_indexed.reindex(incoming_indexed.index)
        for column in output_schema.names:
            if column in subset:
                continue
            incoming_values = (
                incoming_indexed[column]
                if column in incoming_indexed.columns
                else pd.Series(pd.NA, index=incoming_indexed.index)
            )
            existing_values = existing_rows[column]
            equal = incoming_values.eq(existing_values) | (
                incoming_values.isna() & existing_values.isna()
            )
            if not bool(equal.fillna(False).all()):
                return False
        return True

    if incoming_matches_existing():
        print(
            f"[OK] 增量预检查 {existing_path}：new={len(incoming)}，"
            "内容无变化，跳过历史文件重写。"
        )
        return source.metadata.num_rows

    incoming_keys = [normalise_key(value) for value in incoming[primary_column].tolist()]
    incoming_groups: list[tuple[str, int, int]] = []
    group_start = 0
    for index in range(1, len(incoming_keys) + 1):
        if index == len(incoming_keys) or incoming_keys[index] != incoming_keys[group_start]:
            incoming_groups.append((incoming_keys[group_start], group_start, index - group_start))
            group_start = index
    ensure_output_dir(existing_path.parent)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{existing_path.name}.",
        suffix=".tmp",
        dir=existing_path.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    writer: Optional[parquet.ParquetWriter] = None
    row_count = 0
    data_changed = schema_changed
    incoming_group_index = 0
    active_existing_key: Optional[str] = None
    active_existing_parts: list[pa.Table] = []
    try:
        writer = parquet.ParquetWriter(temporary_path, output_schema, compression="snappy")

        def write_table(table: pa.Table) -> None:
            nonlocal row_count
            aligned = _align_arrow_table(table.replace_schema_metadata(None), output_schema)
            writer.write_table(aligned, row_group_size=16_384)
            row_count += aligned.num_rows

        def write_frame(frame: pd.DataFrame) -> None:
            if frame.empty:
                return
            storage_frame = normalise_frame_dates(frame, for_storage=True)
            write_table(pa.Table.from_pandas(storage_frame, preserve_index=False))

        def write_incoming_before(target_key: Optional[str]) -> None:
            nonlocal data_changed, incoming_group_index
            while incoming_group_index < len(incoming_groups):
                key, start, length = incoming_groups[incoming_group_index]
                if target_key is not None and key >= target_key:
                    break
                write_frame(incoming.iloc[start : start + length])
                data_changed = True
                incoming_group_index += 1

        def flush_existing_group() -> None:
            """Merge one bounded instrument group and write it in date order."""

            nonlocal data_changed, incoming_group_index, active_existing_key, active_existing_parts
            if active_existing_key is None:
                return
            write_incoming_before(active_existing_key)
            incoming_frame: Optional[pd.DataFrame] = None
            if incoming_group_index < len(incoming_groups):
                key, start, length = incoming_groups[incoming_group_index]
                if key == active_existing_key:
                    incoming_frame = incoming.iloc[start : start + length]
                    incoming_group_index += 1
            existing_table = (
                active_existing_parts[0]
                if len(active_existing_parts) == 1
                else pa.concat_tables(active_existing_parts)
            )
            if incoming_frame is None:
                # Most instruments have no rows in a small incremental window.
                # Keep those groups in Arrow to avoid tens of thousands of
                # pandas conversions while the immutable parquet is copied.
                write_table(existing_table)
                active_existing_key = None
                active_existing_parts = []
                return
            existing_frame = existing_table.to_pandas()
            frames = [existing_frame, incoming_frame]
            merged = normalise_frame_dates(pd.concat(frames, ignore_index=True), for_storage=False)
            merged = (
                merged.drop_duplicates(subset=subset, keep="last")
                .sort_values(sort_cols, kind="mergesort")
                .reset_index(drop=True)
            )
            merged_storage = normalise_frame_dates(merged, for_storage=True)
            merged_table = _align_arrow_table(
                pa.Table.from_pandas(merged_storage, preserve_index=False), output_schema
            )
            existing_aligned = _align_arrow_table(
                existing_table.replace_schema_metadata(None), output_schema
            )
            if not existing_aligned.equals(merged_table, check_metadata=False):
                data_changed = True
            write_table(merged_table)
            active_existing_key = None
            active_existing_parts = []

        for batch in source.iter_batches(batch_size=16_384, use_threads=False):
            table = pa.Table.from_batches([batch])
            keys = [normalise_key(value) for value in table.column(primary_column).to_pylist()]
            run_start = 0
            for index in range(1, len(keys) + 1):
                if index < len(keys) and keys[index] == keys[run_start]:
                    continue
                run_key = keys[run_start]
                if active_existing_key is not None and run_key < active_existing_key:
                    raise ValueError(f"{existing_path.name} 未按 {primary_column} 连续排序，拒绝增量归并。")
                if active_existing_key is not None and run_key != active_existing_key:
                    flush_existing_group()
                if active_existing_key is None:
                    active_existing_key = run_key
                active_existing_parts.append(table.slice(run_start, index - run_start))
                run_start = index

        flush_existing_group()
        write_incoming_before(None)
        writer.close()
        writer = None
        if parquet.ParquetFile(temporary_path).metadata.num_rows != row_count:
            raise RuntimeError(f"{existing_path.name} 增量写入行数校验失败。")
        if data_changed:
            os.replace(temporary_path, existing_path)
    finally:
        try:
            if writer is not None:
                writer.close()
        finally:
            temporary_path.unlink(missing_ok=True)
    suffix = "" if data_changed else "，内容无变化，保留原文件"
    print(
        f"[OK] 流式增量归并 {existing_path}：old={source.metadata.num_rows}，"
        f"new={len(incoming)}，merged={row_count}{suffix}。"
    )
    return row_count


def _sanitise_cli_error(value: Any) -> str:
    """Bound and redact exception text before writing persistent CLI state."""

    output = str(value)
    try:
        token_values = [read_tushare_token()]
    except RuntimeError:
        token_values = []
    for token in filter(None, token_values):
        output = output.replace(token, "[REDACTED]")
    output = re.sub(r"(?i)(TUSHARE_TOKEN\s*=\s*)\S+", r"\1[REDACTED]", output)
    return output[-2000:]


def next_calendar_date(value: pd.Timestamp) -> str:
    return (pd.Timestamp(value).normalize() + pd.Timedelta(days=1)).strftime("%Y%m%d")


def incremental_start_date(
    output_dir: Path,
    latest_date: pd.Timestamp,
    *,
    lookback_days: int = 5,
) -> str:
    """Return the first of the recent open days that should be refetched.

    The latest stored trading day counts as one lookback day.  Refetching this
    small overlap lets Tushare revisions and delayed fund NAV announcements
    overwrite or fill historical keys without replaying the complete history.
    """

    if lookback_days < 1:
        raise ValueError("incremental lookback days must be positive")
    path = output_dir / "trade_day_df.parquet"
    if not path.exists():
        raise FileNotFoundError(f"缺少交易日历 {path}，无法确定增量回看日期。")
    frame = pd.read_parquet(path, columns=["exchange", "cal_date", "is_open"])
    frame = frame[frame["exchange"].astype(str).str.upper() == "SSE"].copy()
    if pd.api.types.is_datetime64_any_dtype(frame["cal_date"]):
        frame["date"] = pd.to_datetime(frame["cal_date"], errors="coerce")
    else:
        compact = frame["cal_date"].astype(str).str.replace("-", "", regex=False).str[:8]
        frame["date"] = pd.to_datetime(compact, format="%Y%m%d", errors="coerce")
    latest = pd.Timestamp(latest_date).normalize()
    candidates = sorted(
        frame.loc[(frame["is_open"].astype(int) == 1) & (frame["date"] <= latest), "date"]
        .dropna()
        .unique()
        .tolist()
    )
    if not candidates:
        # A partial calendar fixture may begin after the local baseline.  In
        # that case there is nothing available to refetch, so continue forward.
        return next_calendar_date(latest)
    return pd.Timestamp(candidates[-lookback_days]).strftime("%Y%m%d")


def load_open_trade_dates(
    output_dir: Path,
    *,
    start_date: str,
    end_date: str,
    max_days: int,
) -> list[str]:
    path = output_dir / "trade_day_df.parquet"
    if not path.exists():
        raise FileNotFoundError(f"缺少交易日历 {path}，无法确定增量日期。")
    frame = pd.read_parquet(path, columns=["exchange", "cal_date", "is_open"])
    frame = frame[frame["exchange"].astype(str).str.upper() == "SSE"].copy()
    if pd.api.types.is_datetime64_any_dtype(frame["cal_date"]):
        frame["date"] = pd.to_datetime(frame["cal_date"], errors="coerce")
    else:
        compact = frame["cal_date"].astype(str).str.replace("-", "", regex=False).str[:8]
        frame["date"] = pd.to_datetime(compact, format="%Y%m%d", errors="coerce")
    start = pd.to_datetime(start_date, format="%Y%m%d")
    end = pd.to_datetime(end_date, format="%Y%m%d")
    frame = frame[(frame["is_open"].astype(int) == 1) & frame["date"].between(start, end)]
    dates = sorted(frame["date"].dropna().dt.strftime("%Y%m%d").unique().tolist())
    if len(dates) > max_days:
        raise ValueError(
            f"待更新交易日共 {len(dates)} 天，超过安全上限 {max_days}；"
            "请先在命令行核对日期范围并提高 --max-latest-days。"
        )
    return dates


def attach_universe_names(frame: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    names = universe[["ts_code", "name"]].drop_duplicates(subset=["ts_code"])
    allowed = set(names["ts_code"].dropna().astype(str))
    out = frame[frame["ts_code"].astype(str).isin(allowed)].copy()
    return out.drop(columns=["name"], errors="ignore").merge(names, on="ts_code", how="left")


def fetch_latest_dates(
    *,
    api_func: Callable[..., pd.DataFrame],
    api_name: str,
    date_param: str,
    dates: list[str],
    fields: list[str],
    universe: pd.DataFrame,
    limiter: RateLimiter,
    args: argparse.Namespace,
    market: Optional[str] = None,
    universe_label: str = "ETF",
    page_size: Optional[int] = None,
    allow_terminal_empty: bool = True,
) -> list[pd.DataFrame]:
    if not dates:
        return []

    def fetch_one_date(index: int, date_value: str) -> Optional[pd.DataFrame]:
        base_params: dict[str, Any] = {date_param: date_value, "fields": fields_arg(fields)}
        if market:
            base_params["market"] = market
        pages: list[pd.DataFrame] = []
        seen_keys: set[str] = set()
        offset = 0
        max_pages = getattr(args, "max_fund_nav_pages", 20)
        for page in range(max_pages if page_size else 1):
            params = dict(base_params)
            if page_size:
                params.update({"offset": offset, "limit": page_size})
            context = f"{api_name} {date_value} offset={offset}"
            frame = fetch_with_empty_confirmation(
                lambda: call_tushare_api(
                    api_func,
                    limiter,
                    max_retries=args.max_retries,
                    backoff_sec=args.backoff_sec,
                    wait_on_rate_limit_sec=args.wait_on_rate_limit_sec,
                    retry_jitter_sec=getattr(args, "retry_jitter_sec", 0.25),
                    context=context,
                    api_name=api_name,
                    **params,
                ),
                args=args,
                context=context,
            )
            if frame.empty:
                break
            if page_size:
                keys = set(frame["ts_code"].dropna().astype(str)) if "ts_code" in frame.columns else set()
                if page > 0 and keys and keys.issubset(seen_keys):
                    raise RuntimeError(f"{api_name} {date_value} 分页未向前推进，停止以避免无限循环。")
                seen_keys.update(keys)
                print(f"[INFO] {api_name} {date_value} 分页 {page + 1}，offset={offset}，{len(frame)} 行。")
                offset += len(frame)
            pages.append(frame)
            if not page_size:
                break
        else:
            raise RuntimeError(f"{api_name} {date_value} 达到最大分页数 {max_pages}，本批次停止。")
        frame = pd.concat(pages, ignore_index=True) if pages else pd.DataFrame()
        if frame.empty:
            if allow_terminal_empty and index == len(dates) - 1:
                print(f"[INFO] {api_name} {date_value} 尚无数据，保留给下次更新。")
                return None
            raise RuntimeError(f"{api_name} {date_value} 返回空数据；为避免日期空洞，本批次不落盘。")
        frame = attach_universe_names(frame, universe)
        if frame.empty:
            raise RuntimeError(f"{api_name} {date_value} 未匹配当前 {universe_label} universe；本批次不落盘。")
        return frame

    worker_count = min(max(int(getattr(args, "max_workers", 1)), 1), len(dates))
    ordered_results: list[Optional[pd.DataFrame]] = [None] * len(dates)
    if worker_count == 1:
        for index, date_value in enumerate(dates):
            ordered_results[index] = fetch_one_date(index, date_value)
            frame = ordered_results[index]
            if frame is not None:
                print(f"[INFO] {api_name} 增量进度 {index + 1}/{len(dates)}，{len(frame)} 行。")
    else:
        print(f"[INFO] {api_name} 按日期并发抓取，dates={len(dates)}，workers={worker_count}。")
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(fetch_one_date, index, date_value): (index, date_value)
                for index, date_value in enumerate(dates)
            }
            completed_count = 0
            for future in as_completed(futures):
                index, date_value = futures[future]
                ordered_results[index] = future.result()
                completed_count += 1
                frame = ordered_results[index]
                row_count = 0 if frame is None else len(frame)
                print(
                    f"[INFO] {api_name} 增量进度 {completed_count}/{len(dates)}，"
                    f"日期 {date_value}，{row_count} 行。"
                )
    return [frame for frame in ordered_results if frame is not None]


def iter_date_batches(values: list[str], batch_size: int) -> Iterable[list[str]]:
    if batch_size < 1:
        raise ValueError("incremental batch size must be positive")
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def append_latest_batches(
    *,
    dates: list[str],
    out_path: Path,
    fetch_batch: Callable[[list[str], bool], list[pd.DataFrame]],
    prepare_rows: Callable[[pd.DataFrame], pd.DataFrame],
    subset: list[str],
    sort_cols: list[str],
    batch_days: int,
) -> None:
    """Bound incremental memory by atomically merging a few dates at a time."""

    for date_batch in iter_date_batches(dates, batch_days):
        frames = fetch_batch(date_batch, date_batch[-1] == dates[-1])
        if not frames:
            continue
        incoming = prepare_rows(pd.concat(frames, ignore_index=True))
        if incoming.empty:
            continue
        append_incremental_rows(
            incoming,
            out_path,
            subset=subset,
            sort_cols=sort_cols,
            date_column="date",
        )


def save_latest_nav(
    pro: Any,
    output_dir: Path,
    limiter: RateLimiter,
    args: argparse.Namespace,
    etf_info: Optional[pd.DataFrame] = None,
) -> None:
    out_path = output_dir / "etf_daily_df.parquet"
    lookback_days = getattr(args, "incremental_lookback_days", 5)
    start_date = incremental_start_date(
        output_dir,
        latest_parquet_date(out_path, "date"),
        lookback_days=lookback_days,
    )
    dates = load_open_trade_dates(
        output_dir,
        start_date=start_date,
        end_date=args.end_date,
        max_days=args.max_latest_days + max(lookback_days - 1, 0),
    )
    if not dates:
        print("[INFO] fund_nav 已是最新，无需更新。")
        return
    universe = load_or_create_etf_universe(pro, output_dir, limiter, args, etf_info)
    def prepare(frame: pd.DataFrame) -> pd.DataFrame:
        frame["date"] = date_series(frame["nav_date"])
        frame["adj_nav"] = normalise_adj_nav(frame["adj_nav"])
        return frame.dropna(subset=["date"])

    append_latest_batches(
        dates=dates,
        out_path=out_path,
        fetch_batch=lambda batch, allow_empty: fetch_latest_dates(
            api_func=pro.fund_nav,
            api_name="fund_nav",
            date_param="nav_date",
            dates=batch,
            fields=FUND_NAV_FIELDS,
            universe=universe,
            limiter=limiter,
            args=args,
            market="E",
            universe_label="ETF",
            allow_terminal_empty=allow_empty,
        ),
        prepare_rows=prepare,
        subset=["ts_code", "date"],
        sort_cols=["ts_code", "date"],
        batch_days=getattr(args, "incremental_batch_days", DEFAULT_INCREMENTAL_BATCH_DAYS),
    )


def save_latest_candles(
    pro: Any,
    output_dir: Path,
    limiter: RateLimiter,
    args: argparse.Namespace,
    etf_info: Optional[pd.DataFrame] = None,
) -> None:
    out_path = output_dir / "etf_daily_candle_df.parquet"
    lookback_days = getattr(args, "incremental_lookback_days", 5)
    start_date = incremental_start_date(
        output_dir,
        latest_parquet_date(out_path, "date"),
        lookback_days=lookback_days,
    )
    dates = load_open_trade_dates(
        output_dir,
        start_date=start_date,
        end_date=args.end_date,
        max_days=args.max_latest_days + max(lookback_days - 1, 0),
    )
    if not dates:
        print("[INFO] fund_daily 已是最新，无需更新。")
        return
    universe = load_or_create_etf_universe(pro, output_dir, limiter, args, etf_info)
    def prepare(frame: pd.DataFrame) -> pd.DataFrame:
        frame["date"] = date_series(frame["trade_date"])
        return frame.dropna(subset=["date"])

    append_latest_batches(
        dates=dates,
        out_path=out_path,
        fetch_batch=lambda batch, allow_empty: fetch_latest_dates(
            api_func=pro.fund_daily,
            api_name="fund_daily",
            date_param="trade_date",
            dates=batch,
            fields=FUND_DAILY_FIELDS,
            universe=universe,
            limiter=limiter,
            args=args,
            allow_terminal_empty=allow_empty,
        ),
        prepare_rows=prepare,
        subset=["ts_code", "trade_date"],
        sort_cols=["ts_code", "date"],
        batch_days=getattr(args, "incremental_batch_days", DEFAULT_INCREMENTAL_BATCH_DAYS),
    )


def save_etf_info(pro: Any, output_dir: Path, limiter: RateLimiter, args: argparse.Namespace) -> pd.DataFrame:
    fund_df = fetch_fund_basic(pro, limiter, args, market="E")
    etf_df = fetch_etf_basic(pro, limiter, args)
    fund_df = filter_fund_basic_to_etfs(fund_df, etf_df)
    if args.limit:
        active = fund_df[fund_df.get("status", "").astype(str).eq("L")].copy()
        if active.empty:
            active = fund_df.copy()
        sample_date = active.get("found_date", active.get("list_date"))
        active["_sample_date"] = pd.to_datetime(sample_date, format="%Y%m%d", errors="coerce")
        fund_df = active.sort_values(["_sample_date", "ts_code"], na_position="last").head(args.limit)
        fund_df = fund_df.drop(columns=["_sample_date"])
        if not etf_df.empty:
            etf_df = etf_df[etf_df["ts_code"].isin(fund_df["ts_code"])].copy()
    df = build_etf_info_df(fund_df, etf_df)
    if df.empty:
        raise RuntimeError("未获取到 ETF 产品信息。")
    save_dataframe(df, output_dir / "etf_info_df.parquet", excel_path=output_dir / "etf_info_df.xlsx")
    return df


def save_public_fund_info(
    pro: Any,
    output_dir: Path,
    limiter: RateLimiter,
    args: argparse.Namespace,
) -> pd.DataFrame:
    fund_df = fetch_fund_basic(pro, limiter, args, market="O")
    if args.limit:
        active = fund_df[fund_df.get("status", "").astype(str).eq("L")].copy()
        if active.empty:
            active = fund_df.copy()
        sample_date = active.get("found_date", active.get("list_date"))
        active["_sample_date"] = pd.to_datetime(sample_date, format="%Y%m%d", errors="coerce")
        fund_df = active.sort_values(["_sample_date", "ts_code"], na_position="last").head(args.limit)
        fund_df = fund_df.drop(columns=["_sample_date"])
    df = build_public_fund_info_df(fund_df)
    if df.empty:
        raise RuntimeError("未获取到场外公募基金信息。")
    save_dataframe(df, output_dir / "fund_info_df.parquet", excel_path=output_dir / "fund_info_df.xlsx")
    return df


def load_or_create_etf_universe(
    pro: Any,
    output_dir: Path,
    limiter: RateLimiter,
    args: argparse.Namespace,
    current_info: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    if current_info is not None and not current_info.empty:
        df = current_info
    else:
        path = output_dir / "etf_info_df.parquet"
        if path.exists():
            df = pd.read_parquet(path)
        else:
            df = save_etf_info(pro, output_dir, limiter, args)
    if args.limit:
        df = df.head(args.limit)
    required = {"ts_code", "name"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"etf_info_df 缺少必要列: {sorted(missing)}")
    return df.dropna(subset=["ts_code"]).copy()


def load_or_create_public_fund_universe(
    pro: Any,
    output_dir: Path,
    limiter: RateLimiter,
    args: argparse.Namespace,
    current_info: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    if current_info is not None and not current_info.empty:
        df = current_info
    else:
        path = output_dir / "fund_info_df.parquet"
        df = pd.read_parquet(path) if path.exists() else save_public_fund_info(pro, output_dir, limiter, args)
    if args.limit:
        df = df.head(args.limit)
    required = {"ts_code", "name"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"fund_info_df 缺少必要列: {sorted(missing)}")
    return df.dropna(subset=["ts_code"]).copy()


def _fetch_history_chunks_with_empty_retry(
    chunks: list[tuple[str, str]],
    fetch_chunk: Callable[[str, str], pd.DataFrame],
    *,
    args: argparse.Namespace,
    context: str,
) -> list[pd.DataFrame]:
    """Fetch each historical window with independent empty confirmation.

    A non-empty neighbouring window must never cause a transient empty window
    to be accepted.  Callers clip ``chunks`` to the instrument lifecycle first,
    so legitimate pre-launch/post-termination windows are not requested.
    """

    frames: list[pd.DataFrame] = []
    for start, end in chunks:
        frame = fetch_with_empty_confirmation(
            lambda start=start, end=end: fetch_chunk(start, end),
            args=args,
            context=f"{context} {start}-{end}",
        )
        if not frame.empty:
            frames.append(frame)
    return frames


def _history_bound(value: Any) -> Optional[str]:
    parsed = _parse_date_scalar(value)
    return parsed.strftime("%Y%m%d") if parsed is not None else None


def history_bounds_by_code(
    universe: pd.DataFrame,
    *,
    start_columns: Iterable[str],
    end_columns: Iterable[str],
) -> dict[str, tuple[Optional[str], Optional[str]]]:
    """Extract per-instrument lifecycle bounds without guessing share classes."""

    bounds: dict[str, tuple[Optional[str], Optional[str]]] = {}
    for row in universe.to_dict(orient="records"):
        code = str(row.get("ts_code") or "").strip()
        if not code:
            continue
        starts = [_history_bound(row.get(column)) for column in start_columns if column in row]
        ends = [_history_bound(row.get(column)) for column in end_columns if column in row]
        valid_starts = [value for value in starts if value]
        valid_ends = [value for value in ends if value]
        bounds[code] = (
            min(valid_starts) if valid_starts else None,
            max(valid_ends) if valid_ends else None,
        )
    return bounds


def clipped_history_chunks(
    args: argparse.Namespace,
    *,
    history_start_date: Optional[str] = None,
    history_end_date: Optional[str] = None,
) -> list[tuple[str, str]]:
    start = max(args.start_date, history_start_date or args.start_date)
    end = min(args.end_date, history_end_date or args.end_date)
    if start > end:
        return []
    return list(iter_date_chunks(start, end, getattr(args, "history_chunk_days", 3650)))


def fetch_fund_nav(
    pro: Any,
    ts_code: str,
    name: str,
    limiter: RateLimiter,
    args: argparse.Namespace,
    *,
    market: str = "E",
    history_start_date: Optional[str] = None,
    history_end_date: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    chunks = clipped_history_chunks(
        args,
        history_start_date=history_start_date,
        history_end_date=history_end_date,
    )
    frames = _fetch_history_chunks_with_empty_retry(
        chunks,
        lambda chunk_start, chunk_end: call_tushare_api(
            pro.fund_nav,
            limiter,
            max_retries=args.max_retries,
            backoff_sec=args.backoff_sec,
            wait_on_rate_limit_sec=args.wait_on_rate_limit_sec,
            retry_jitter_sec=getattr(args, "retry_jitter_sec", 0.25),
            context=f"fund_nav market={market} {ts_code} {chunk_start}-{chunk_end}",
            api_name="fund_nav",
            ts_code=ts_code,
            market=market,
            start_date=chunk_start,
            end_date=chunk_end,
            fields=fields_arg(FUND_NAV_FIELDS),
        ),
        args=args,
        context=f"fund_nav market={market} {ts_code}",
    )
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["ts_code", "nav_date"])
    for column in FUND_NAV_FIELDS:
        if column not in df.columns:
            df[column] = pd.NA
    for column in ["ts_code", "ann_date", "nav_date"]:
        df[column] = df[column].astype("string")
    for column in set(FUND_NAV_FIELDS) - {"ts_code", "ann_date", "nav_date"}:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["adj_nav"] = normalise_adj_nav(df["adj_nav"])
    df = df[FUND_NAV_FIELDS].copy()
    df["name"] = pd.Series([name] * len(df), dtype="string")
    df["date"] = date_series(df["nav_date"])
    return df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


def save_nav(
    pro: Any,
    output_dir: Path,
    limiter: RateLimiter,
    args: argparse.Namespace,
    etf_info: Optional[pd.DataFrame] = None,
) -> None:
    universe = load_or_create_etf_universe(pro, output_dir, limiter, args, etf_info)
    lifecycle = history_bounds_by_code(
        universe,
        start_columns=("setup_date", "list_date", "found_date", "issue_date"),
        end_columns=("delist_date", "due_date"),
    )
    out_path = output_dir / "etf_daily_df.parquet"
    if args.missing_only:
        universe = filter_missing_universe(universe, out_path, label="fund_nav")
        if universe.empty:
            print("[INFO] fund_nav 无缺失代码，无需增量抓取。")
            return
    save_full_history_with_checkpoints(
        universe=universe,
        out_path=out_path,
        label="ETF fund_nav",
        fetcher=lambda code, name: fetch_fund_nav(
            pro,
            code,
            name,
            limiter,
            args,
            history_start_date=lifecycle.get(code, (None, None))[0],
            history_end_date=lifecycle.get(code, (None, None))[1],
        ),
        duplicate_subset=["ts_code", "date"],
        sort_cols=["ts_code", "date"],
        args=args,
    )


def save_latest_public_fund_nav(
    pro: Any,
    output_dir: Path,
    limiter: RateLimiter,
    args: argparse.Namespace,
    fund_info: Optional[pd.DataFrame] = None,
) -> None:
    out_path = output_dir / "fund_nav_df.parquet"
    lookback_days = getattr(args, "incremental_lookback_days", 5)
    start_date = incremental_start_date(
        output_dir,
        latest_parquet_date(out_path, "date"),
        lookback_days=lookback_days,
    )
    dates = load_open_trade_dates(
        output_dir,
        start_date=start_date,
        end_date=args.end_date,
        max_days=args.max_latest_days + max(lookback_days - 1, 0),
    )
    if not dates:
        print("[INFO] 场外公募基金 fund_nav 已是最新，无需更新。")
        return
    universe = load_or_create_public_fund_universe(pro, output_dir, limiter, args, fund_info)
    def prepare(frame: pd.DataFrame) -> pd.DataFrame:
        frame["date"] = date_series(frame["nav_date"])
        frame["adj_nav"] = normalise_adj_nav(frame["adj_nav"])
        return frame.dropna(subset=["date"])

    append_latest_batches(
        dates=dates,
        out_path=out_path,
        fetch_batch=lambda batch, allow_empty: fetch_latest_dates(
            api_func=pro.fund_nav,
            api_name="fund_nav",
            date_param="nav_date",
            dates=batch,
            fields=FUND_NAV_FIELDS,
            universe=universe,
            limiter=limiter,
            args=args,
            market="O",
            universe_label="场外公募基金",
            page_size=getattr(args, "fund_nav_page_size", DEFAULT_FUND_NAV_PAGE_SIZE),
            allow_terminal_empty=allow_empty,
        ),
        prepare_rows=prepare,
        subset=["ts_code", "date"],
        sort_cols=["ts_code", "date"],
        batch_days=getattr(args, "incremental_batch_days", DEFAULT_INCREMENTAL_BATCH_DAYS),
    )


def save_public_fund_nav(
    pro: Any,
    output_dir: Path,
    limiter: RateLimiter,
    args: argparse.Namespace,
    fund_info: Optional[pd.DataFrame] = None,
) -> None:
    universe = load_or_create_public_fund_universe(pro, output_dir, limiter, args, fund_info)
    lifecycle = history_bounds_by_code(
        universe,
        start_columns=("found_date", "issue_date", "list_date"),
        end_columns=("due_date", "delist_date"),
    )
    out_path = output_dir / "fund_nav_df.parquet"
    if args.missing_only:
        universe = filter_missing_universe(universe, out_path, label="场外公募基金 fund_nav")
        if universe.empty:
            print("[INFO] 场外公募基金 fund_nav 无缺失代码，无需补抓。")
            return
    save_full_history_with_checkpoints(
        universe=universe,
        out_path=out_path,
        label="场外公募基金 fund_nav",
        fetcher=lambda code, name: fetch_fund_nav(
            pro,
            code,
            name,
            limiter,
            args,
            market="O",
            history_start_date=lifecycle.get(code, (None, None))[0],
            history_end_date=lifecycle.get(code, (None, None))[1],
        ),
        duplicate_subset=["ts_code", "date"],
        sort_cols=["ts_code", "date"],
        args=args,
    )


def fetch_fund_daily(
    pro: Any,
    ts_code: str,
    name: str,
    limiter: RateLimiter,
    args: argparse.Namespace,
    *,
    history_start_date: Optional[str] = None,
    history_end_date: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    chunks = clipped_history_chunks(
        args,
        history_start_date=history_start_date,
        history_end_date=history_end_date,
    )
    frames = _fetch_history_chunks_with_empty_retry(
        chunks,
        lambda chunk_start, chunk_end: call_tushare_api(
            pro.fund_daily,
            limiter,
            max_retries=args.max_retries,
            backoff_sec=args.backoff_sec,
            wait_on_rate_limit_sec=args.wait_on_rate_limit_sec,
            retry_jitter_sec=getattr(args, "retry_jitter_sec", 0.25),
            context=f"fund_daily {ts_code} {chunk_start}-{chunk_end}",
            api_name="fund_daily",
            ts_code=ts_code,
            start_date=chunk_start,
            end_date=chunk_end,
            fields=fields_arg(FUND_DAILY_FIELDS),
        ),
        args=args,
        context=f"fund_daily {ts_code}",
    )
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["ts_code", "trade_date"])
    for column in FUND_DAILY_FIELDS:
        if column not in df.columns:
            df[column] = pd.NA
    for column in ["ts_code", "trade_date"]:
        df[column] = df[column].astype("string")
    for column in set(FUND_DAILY_FIELDS) - {"ts_code", "trade_date"}:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df[FUND_DAILY_FIELDS].copy()
    df["name"] = pd.Series([name] * len(df), dtype="string")
    df["date"] = date_series(df["trade_date"])
    return df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


def save_candles(
    pro: Any,
    output_dir: Path,
    limiter: RateLimiter,
    args: argparse.Namespace,
    etf_info: Optional[pd.DataFrame] = None,
) -> None:
    universe = load_or_create_etf_universe(pro, output_dir, limiter, args, etf_info)
    lifecycle = history_bounds_by_code(
        universe,
        start_columns=("setup_date", "list_date", "found_date", "issue_date"),
        end_columns=("delist_date", "due_date"),
    )
    out_path = output_dir / "etf_daily_candle_df.parquet"
    if args.missing_only:
        universe = filter_missing_universe(universe, out_path, label="fund_daily")
        if universe.empty:
            print("[INFO] fund_daily 无缺失代码，无需增量抓取。")
            return
    save_full_history_with_checkpoints(
        universe=universe,
        out_path=out_path,
        label="ETF fund_daily",
        fetcher=lambda code, name: fetch_fund_daily(
            pro,
            code,
            name,
            limiter,
            args,
            history_start_date=lifecycle.get(code, (None, None))[0],
            history_end_date=lifecycle.get(code, (None, None))[1],
        ),
        duplicate_subset=["ts_code", "trade_date"],
        sort_cols=["ts_code", "date"],
        args=args,
    )


def save_trade_calendar(
    pro: Any,
    output_dir: Path,
    limiter: RateLimiter,
    args: argparse.Namespace,
    *,
    incremental_start: Optional[str] = None,
) -> None:
    out_path = output_dir / "trade_day_df.parquet"
    start_date = incremental_start or args.start_date
    if start_date > args.end_date:
        print("[INFO] 交易日历已覆盖目标日期，无需更新。")
        return
    df = call_tushare_api(
        pro.trade_cal,
        limiter,
        max_retries=args.max_retries,
        backoff_sec=args.backoff_sec,
        wait_on_rate_limit_sec=args.wait_on_rate_limit_sec,
        context="trade_cal",
        exchange="SSE",
        start_date=start_date,
        end_date=args.end_date,
    )
    required = {"exchange", "cal_date", "is_open"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"trade_cal 缺少必要列: {sorted(missing)}")
    if incremental_start:
        append_incremental_rows(
            df,
            out_path,
            subset=["exchange", "cal_date"],
            sort_cols=["exchange", "cal_date"],
            date_column="cal_date",
        )
        return
    save_dataframe(df, out_path)


def save_stock_basic(pro: Any, output_dir: Path, limiter: RateLimiter, args: argparse.Namespace) -> None:
    df = call_tushare_api(
        pro.stock_basic,
        limiter,
        max_retries=args.max_retries,
        backoff_sec=args.backoff_sec,
        wait_on_rate_limit_sec=args.wait_on_rate_limit_sec,
        context="stock_basic",
        exchange="",
        list_status="L",
        fields=fields_arg(STOCK_BASIC_FIELDS),
    )
    if args.limit:
        df = df.head(args.limit)
    save_dataframe(df, output_dir / "stock_basic.parquet", excel_path=output_dir / "stock_basic.xlsx")


def _looks_like_permission_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in ("权限", "积分", "permission", "没有访问", "无权限"))


def _index_api_method(pro: Any, api_name: str) -> Callable[..., pd.DataFrame]:
    method = getattr(pro, api_name, None)
    if callable(method):
        return method
    query = getattr(pro, "query", None)
    if callable(query):
        return lambda **kwargs: query(api_name, **kwargs)
    raise AttributeError(f"Tushare client 不支持 {api_name}。")


def _call_index_api(
    pro: Any,
    api_name: str,
    limiter: RateLimiter,
    args: argparse.Namespace,
    *,
    allow_capped_response: bool = False,
    context: Optional[str] = None,
    **kwargs: Any,
) -> pd.DataFrame:
    return call_tushare_api(
        _index_api_method(pro, api_name),
        limiter,
        max_retries=args.max_retries,
        backoff_sec=args.backoff_sec,
        wait_on_rate_limit_sec=args.wait_on_rate_limit_sec,
        retry_jitter_sec=getattr(args, "retry_jitter_sec", 0.25),
        context=context or api_name,
        api_name=api_name,
        allow_capped_response=allow_capped_response,
        **kwargs,
    )


def _first_existing_column(frame: pd.DataFrame, candidates: Iterable[str]) -> pd.Series:
    for column in candidates:
        if column in frame.columns:
            return frame[column]
    return pd.Series([None] * len(frame), index=frame.index, dtype="object")


def _normalise_catalog_frame(frame: pd.DataFrame, source_api: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    src = _first_existing_column(frame, ["src"]).fillna("").astype(str).str.upper()
    quote_source = pd.Series(INDEX_QUOTE_SOURCE.get(source_api), index=frame.index, dtype="object")
    if source_api == "index_classify":
        quote_source = src.map(lambda value: "ci_daily" if value.startswith("CI") else "sw_daily")
    result = pd.DataFrame(
        {
            "source_api": source_api,
            "ts_code": _first_existing_column(frame, ["ts_code", "index_code", "code"]),
            "name": _first_existing_column(
                frame, ["name", "fullname", "index_name", "industry_name", "indx_name", "indx_csname"]
            ),
            "category": _first_existing_column(
                frame, ["category", "idx_type", "index_type", "type", "level", "industry_name"]
            ),
            "market": _first_existing_column(frame, ["market", "exchange"]),
            "publisher": _first_existing_column(
                frame, ["publisher", "publisher_name", "pub_party_name", "provider"]
            ),
            "list_date": _first_existing_column(frame, ["list_date", "pub_date", "launch_date"]),
            "exp_date": _first_existing_column(frame, ["exp_date", "delist_date", "end_date"]),
            "quote_source_api": quote_source,
        }
    )
    result["ts_code"] = result["ts_code"].fillna("").astype(str).str.strip()
    result = result[result["ts_code"].ne("")].copy()
    for column in ("list_date", "exp_date"):
        result[column] = pd.to_datetime(result[column], errors="coerce")
    today = pd.Timestamp.today().normalize()
    result["status"] = "active"
    result.loc[result["exp_date"].notna() & result["exp_date"].lt(today), "status"] = "inactive"
    return result.drop_duplicates(["source_api", "ts_code"], keep="last")


def _catalog_raw_path(output_dir: Path, api_name: str) -> Path:
    legacy = {"index_basic": "index_info.parquet", "etf_index": "etf_index.parquet"}
    return output_dir / legacy.get(api_name, f"{api_name}_df.parquet")


def fetch_index_catalog_request(
    pro: Any,
    api_name: str,
    params: dict[str, Any],
    limiter: RateLimiter,
    args: argparse.Namespace,
) -> pd.DataFrame:
    """Fetch a catalog request without accidentally accepting capped history.

    dc_index and tdx_index are daily snapshots rather than timeless directory
    endpoints.  Query the latest available day (with a short holiday/publication
    fallback) so one request never aggregates their entire history to the cap.
    """

    if api_name not in DATE_SCOPED_INDEX_CATALOG_APIS:
        return _call_index_api(
            pro,
            api_name,
            limiter,
            args,
            context=f"{api_name} {params or 'all'}",
            **params,
        )
    end = pd.Timestamp(args.end_date)
    for offset in range(10):
        trade_date = (end - pd.Timedelta(days=offset)).strftime("%Y%m%d")
        dated_params = {**params, "trade_date": trade_date}
        frame = _call_index_api(
            pro,
            api_name,
            limiter,
            args,
            context=f"{api_name} {dated_params}",
            **dated_params,
        )
        if not frame.empty:
            return frame
    return pd.DataFrame()


def save_index_catalog(pro: Any, output_dir: Path, limiter: RateLimiter, args: argparse.Namespace) -> None:
    unified_path = output_dir / "index_catalog_df.parquet"
    existing_catalog = pd.read_parquet(unified_path) if unified_path.exists() else pd.DataFrame()
    save_index_basic(pro, output_dir, limiter, args)
    save_etf_index(pro, output_dir, limiter, args)
    raw_frames: dict[str, list[pd.DataFrame]] = {
        "index_basic": [pd.read_parquet(_catalog_raw_path(output_dir, "index_basic"))],
        "etf_index": [pd.read_parquet(_catalog_raw_path(output_dir, "etf_index"))],
    }
    catalog_requests: list[tuple[str, dict[str, Any]]] = []
    for api_name in INDEX_CATALOG_APIS:
        requests: list[dict[str, Any]] = [{}]
        if api_name == "index_classify":
            requests = [
                {"level": level, "src": source}
                for source in ("SW2021", "CI")
                for level in ("L1", "L2", "L3")
            ]
        elif api_name == "dc_index":
            requests = [
                {"idx_type": index_type}
                for index_type in ("行业板块", "概念板块", "地域板块")
            ]
        elif api_name == "tdx_index":
            requests = [
                {"idx_type": index_type}
                for index_type in ("行业板块", "概念板块", "风格板块", "地区板块")
            ]
        catalog_requests.extend((api_name, params) for params in requests)

    def fetch_catalog_request(api_name: str, params: dict[str, Any]):
        try:
            frame = fetch_index_catalog_request(
                pro, api_name, params, limiter, args
            )
            return api_name, params, frame, None
        except Exception as exc:  # noqa: BLE001
            return api_name, params, pd.DataFrame(), exc

    fetched_catalog: dict[str, list[pd.DataFrame]] = {
        api_name: [] for api_name in INDEX_CATALOG_APIS
    }
    worker_count = min(
        max(int(getattr(args, "max_workers", 1)), 1),
        max(len(catalog_requests), 1),
    )
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(fetch_catalog_request, api_name, params): (api_name, params)
            for api_name, params in catalog_requests
        }
        for future in as_completed(futures):
            api_name, params, frame, error = future.result()
            if error is not None:
                reason = "权限不足" if _looks_like_permission_error(error) else "请求失败"
                print(
                    f"[WARN] {api_name} {params or 'all'} {reason}，"
                    f"该目录将标记为缺失: {error}"
                )
                continue
            if args.limit:
                frame = frame.head(args.limit)
            if not frame.empty:
                fetched_catalog[api_name].append(frame)

    for api_name in INDEX_CATALOG_APIS:
        frames = fetched_catalog[api_name]
        if frames:
            raw = pd.concat(frames, ignore_index=True).drop_duplicates()
            save_dataframe(raw, _catalog_raw_path(output_dir, api_name))
            raw_frames[api_name] = [raw]
    normalised = [
        _normalise_catalog_frame(frame, api_name)
        for api_name, frames in raw_frames.items()
        for frame in frames
    ]
    normalised.append(
        pd.DataFrame(
            {
                "source_api": "fut_index_daily",
                "ts_code": [code for code, _name in FUTURES_INDEX_UNIVERSE],
                "name": [name for _code, name in FUTURES_INDEX_UNIVERSE],
                "category": "商品期货指数",
                "market": "FUTURES",
                "publisher": "南华期货",
                "list_date": pd.NaT,
                "exp_date": pd.NaT,
                "quote_source_api": "fut_index_daily",
                "status": "active",
            }
        )
    )
    if not existing_catalog.empty and "source_api" in existing_catalog:
        refreshed_sources = {*raw_frames, "fut_index_daily"}
        preserved = existing_catalog[
            ~existing_catalog["source_api"].astype(str).isin(refreshed_sources)
        ].copy()
        if not preserved.empty:
            normalised.append(preserved)
    catalog = pd.concat([item for item in normalised if not item.empty], ignore_index=True)
    catalog = catalog.drop_duplicates(["source_api", "ts_code"], keep="last").sort_values(
        ["source_api", "ts_code"]
    )
    save_dataframe(catalog, unified_path)


def _normalise_index_history(frame: pd.DataFrame, api_name: str, code: Optional[str]) -> pd.DataFrame:
    if frame.empty:
        return frame
    result = frame.copy()
    if "ts_code" not in result:
        result["ts_code"] = code
    if "trade_date" not in result:
        result["trade_date"] = _first_existing_column(result, ["date", "cal_date"])
    result["source_api"] = api_name
    result["ts_code"] = result["ts_code"].fillna(code or "").astype(str).str.strip()
    result["trade_date"] = pd.to_datetime(result["trade_date"], errors="coerce")
    result = result.dropna(subset=["trade_date"])
    result = result[result["ts_code"].ne("")]
    return result


def fetch_index_date_window(
    *,
    pro: Any,
    api_name: str,
    limiter: RateLimiter,
    args: argparse.Namespace,
    code: Optional[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Fetch a complete range, recursively splitting any capped response."""

    kwargs: dict[str, Any] = {"start_date": start_date, "end_date": end_date}
    if code:
        kwargs["ts_code"] = code
    frame = _call_index_api(
        pro, api_name, limiter, args,
        allow_capped_response=True,
        context=f"{api_name} {code or 'all'} {start_date}-{end_date}",
        **kwargs,
    )
    limit = API_ROW_LIMITS.get(api_name)
    if limit is not None and len(frame) >= limit:
        start = pd.to_datetime(start_date, format="%Y%m%d")
        end = pd.to_datetime(end_date, format="%Y%m%d")
        if start >= end:
            raise ResponseTruncatedError(
                f"{api_name} {code or 'all'} 单日返回 {len(frame)} 行，达到上限 {limit}。"
            )
        middle = start + (end - start) // 2
        left = fetch_index_date_window(
            pro=pro, api_name=api_name, limiter=limiter, args=args, code=code,
            start_date=start.strftime("%Y%m%d"), end_date=middle.strftime("%Y%m%d"),
        )
        right_start = (middle + pd.Timedelta(days=1)).strftime("%Y%m%d")
        right = fetch_index_date_window(
            pro=pro, api_name=api_name, limiter=limiter, args=args, code=code,
            start_date=right_start, end_date=end.strftime("%Y%m%d"),
        )
        frame = pd.concat([left, right], ignore_index=True)
    return _normalise_index_history(frame, api_name, code)


def _index_universe(output_dir: Path, api_name: str, args: argparse.Namespace) -> pd.DataFrame:
    if api_name == "index_dailybasic":
        frame = pd.DataFrame({"ts_code": VALUATION_INDEX_CODES, "name": VALUATION_INDEX_CODES})
    elif api_name == "fut_index_daily":
        frame = pd.DataFrame(FUTURES_INDEX_UNIVERSE, columns=["ts_code", "name"])
    else:
        path = output_dir / "index_catalog_df.parquet"
        if not path.exists():
            raise FileNotFoundError("缺少 index_catalog_df.parquet，请先更新指数目录。")
        catalog = pd.read_parquet(path)
        frame = catalog[catalog["quote_source_api"].astype(str).eq(api_name)][["ts_code", "name"]]
        if frame.empty and api_name == "index_global":
            frame = pd.DataFrame([{"ts_code": "__all__", "name": "全部"}])
    frame = frame.drop_duplicates("ts_code").sort_values("ts_code")
    return frame.head(args.limit) if args.limit else frame


def _index_incremental_start(
    output_dir: Path,
    path: Path,
    args: argparse.Namespace,
) -> str:
    if not path.exists():
        return args.start_date
    latest = latest_parquet_date(path, "trade_date")
    if pd.isna(latest):
        return args.start_date
    start = incremental_start_date(
        output_dir,
        pd.Timestamp(latest),
        lookback_days=max(int(getattr(args, "incremental_lookback_days", 5)), 1),
    )
    return max(pd.Timestamp(args.start_date), pd.Timestamp(start)).strftime("%Y%m%d")


def _discover_index_universe(
    pro: Any,
    api_name: str,
    limiter: RateLimiter,
    args: argparse.Namespace,
) -> pd.DataFrame:
    """Discover market-wide APIs once, then keep full history code-partitioned."""

    end = pd.Timestamp(args.end_date)
    for offset in range(10):
        trade_date = (end - pd.Timedelta(days=offset)).strftime("%Y%m%d")
        frame = fetch_index_date_window(
            pro=pro, api_name=api_name, limiter=limiter, args=args, code=None,
            start_date=trade_date, end_date=trade_date,
        )
        if frame.empty:
            continue
        names = _first_existing_column(frame, ["name", "index_name", "fullname"])
        universe = pd.DataFrame({"ts_code": frame["ts_code"], "name": names})
        universe["name"] = universe["name"].fillna(universe["ts_code"])
        return universe.drop_duplicates("ts_code").sort_values("ts_code")
    raise RuntimeError(f"{api_name} 最近 10 个日历日均无数据，无法发现代码目录。")


def _merge_discovered_catalog(output_dir: Path, api_name: str, universe: pd.DataFrame) -> None:
    path = output_dir / "index_catalog_df.parquet"
    catalog = pd.read_parquet(path)
    addition = pd.DataFrame(
        {
            "source_api": api_name,
            "ts_code": universe["ts_code"].astype(str),
            "name": universe["name"],
            "category": "国际指数" if api_name == "index_global" else "商品期货指数",
            "market": "GLOBAL" if api_name == "index_global" else "FUTURES",
            "publisher": None,
            "list_date": pd.NaT,
            "exp_date": pd.NaT,
            "quote_source_api": api_name,
            "status": "active",
        }
    )
    merged = pd.concat([catalog, addition], ignore_index=True).drop_duplicates(
        ["source_api", "ts_code"], keep="last"
    ).sort_values(["source_api", "ts_code"])
    save_dataframe(merged, path, quiet=True)


def save_index_full_history_with_segment_checkpoints(
    *,
    universe: pd.DataFrame,
    out_path: Path,
    api_name: str,
    start_date: str,
    pro: Any,
    limiter: RateLimiter,
    args: argparse.Namespace,
) -> None:
    """Fetch codes concurrently while persisting every date segment on the main thread."""

    universe = universe[["ts_code", "name"]].drop_duplicates("ts_code").sort_values("ts_code")
    checkpoint_dir = history_checkpoint_dir(out_path, args)
    segment_dir = checkpoint_dir / "segments"
    ensure_output_dir(segment_dir)
    chunks = list(iter_date_chunks(start_date, args.end_date, max(int(args.history_chunk_days), 1)))

    def segment_paths(code: str, chunk_start: str, chunk_end: str) -> tuple[Path, Path]:
        safe_code = re.sub(r"[^A-Za-z0-9_.-]", "_", code)
        stem = f"{safe_code}__{chunk_start}_{chunk_end}"
        return segment_dir / f"{stem}.parquet", segment_dir / f"{stem}.empty"

    pending: list[tuple[str, str, list[tuple[str, str]]]] = []
    code_parts: dict[str, Path] = {}
    code_empty: dict[str, Path] = {}
    for code_value, name_value in universe.itertuples(index=False, name=None):
        code = str(code_value)
        part_path, empty_path = history_checkpoint_paths(checkpoint_dir, code)
        code_parts[code] = part_path
        code_empty[code] = empty_path
        if part_path.exists() or empty_path.exists():
            continue
        missing_chunks = [
            chunk
            for chunk in chunks
            if not any(path.exists() for path in segment_paths(code, *chunk))
        ]
        pending.append((code, str(name_value), missing_chunks))

    def fetch_code(code: str, missing_chunks: list[tuple[str, str]]):
        completed: list[tuple[str, str, pd.DataFrame]] = []
        try:
            for chunk_start, chunk_end in missing_chunks:
                frame = fetch_index_date_window(
                    pro=pro, api_name=api_name, limiter=limiter, args=args, code=code,
                    start_date=chunk_start, end_date=chunk_end,
                )
                completed.append((chunk_start, chunk_end, frame))
            return code, completed, None
        except Exception as exc:  # noqa: BLE001
            return code, completed, exc

    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(fetch_code, code, missing): code
            for code, _name, missing in pending
        }
        processed = 0
        for future in as_completed(futures):
            code, completed, error = future.result()
            for chunk_start, chunk_end, frame in completed:
                part_path, empty_path = segment_paths(code, chunk_start, chunk_end)
                if frame.empty:
                    mark_empty_checkpoint(empty_path)
                else:
                    save_dataframe(
                        frame.drop_duplicates(["source_api", "ts_code", "trade_date"])
                        .sort_values(["ts_code", "trade_date"]),
                        part_path,
                        quiet=True,
                    )
            if error is not None:
                errors.append(code)
                print(f"[WARN] {api_name} {code} 分段失败，已保留成功日期段: {error}")
            processed += 1
            if processed % 50 == 0 or processed == len(pending):
                print(f"[INFO] {api_name} 分段进度 {processed}/{len(pending)}，异常 {len(errors)}。")
    if errors:
        raise RuntimeError(
            f"{api_name} 有 {len(errors)} 个代码的日期段失败；重启仅重试缺失日期段: "
            f"{', '.join(errors[:10])}"
        )

    for code in universe["ts_code"].astype(str):
        final_part = code_parts[code]
        if final_part.exists() or code_empty[code].exists():
            continue
        segment_files = [
            segment_paths(code, chunk_start, chunk_end)[0]
            for chunk_start, chunk_end in chunks
            if segment_paths(code, chunk_start, chunk_end)[0].exists()
        ]
        completed_count = sum(
            any(path.exists() for path in segment_paths(code, chunk_start, chunk_end))
            for chunk_start, chunk_end in chunks
        )
        if completed_count != len(chunks):
            raise RuntimeError(f"{api_name} {code} 的日期段检查点不完整。")
        if not segment_files:
            mark_empty_checkpoint(code_empty[code])
            continue
        frame = pd.concat((pd.read_parquet(path) for path in segment_files), ignore_index=True)
        frame = frame.drop_duplicates(["source_api", "ts_code", "trade_date"]).sort_values(
            ["ts_code", "trade_date"]
        )
        save_dataframe(frame, final_part, quiet=True)
    part_paths = [code_parts[code] for code in universe["ts_code"].astype(str) if code_parts[code].exists()]
    if not part_paths:
        raise RuntimeError(f"{api_name} 未获得任何历史数据。")
    consolidate_parquet_parts(part_paths, out_path)


def save_index_history_api(
    pro: Any,
    output_dir: Path,
    limiter: RateLimiter,
    args: argparse.Namespace,
    api_name: str,
) -> None:
    out_path = output_dir / INDEX_HISTORY_FILES[api_name]
    universe = _index_universe(output_dir, api_name, args)
    if universe["ts_code"].astype(str).eq("__all__").any():
        universe = _discover_index_universe(pro, api_name, limiter, args)
        _merge_discovered_catalog(output_dir, api_name, universe)
        if args.limit:
            universe = universe.head(args.limit)
    if universe.empty:
        empty = pd.DataFrame(
            {
                "source_api": pd.Series(dtype="string"),
                "ts_code": pd.Series(dtype="string"),
                "trade_date": pd.Series(dtype="datetime64[ns]"),
            }
        )
        save_dataframe(empty, out_path, quiet=True)
        print(f"[WARN] {api_name} 没有可抓取的指数代码，已写入空数据状态。")
        return
    start_date = (
        _index_incremental_start(output_dir, out_path, args)
        if args.latest
        else args.start_date
    )
    if args.smoke:
        start_date = max(
            pd.Timestamp(start_date), pd.Timestamp(args.end_date) - pd.Timedelta(days=31)
        ).strftime("%Y%m%d")

    def fetcher(code: str, _name: str) -> pd.DataFrame:
        actual_code = None if code == "__all__" else code
        parts = [
            fetch_index_date_window(
                pro=pro, api_name=api_name, limiter=limiter, args=args, code=actual_code,
                start_date=chunk_start, end_date=chunk_end,
            )
            for chunk_start, chunk_end in iter_date_chunks(
                start_date, args.end_date, max(int(args.history_chunk_days), 1)
            )
        ]
        return pd.concat([part for part in parts if not part.empty], ignore_index=True) if any(
            not part.empty for part in parts
        ) else pd.DataFrame()

    if not args.latest:
        save_index_full_history_with_segment_checkpoints(
            universe=universe,
            out_path=out_path,
            api_name=api_name,
            start_date=start_date,
            pro=pro,
            limiter=limiter,
            args=args,
        )
        return
    frames: list[pd.DataFrame] = []
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(fetcher, str(row.ts_code), str(row.name)): str(row.ts_code)
            for row in universe.itertuples(index=False)
        }
        for future in as_completed(futures):
            code = futures[future]
            try:
                frame = future.result()
                if not frame.empty:
                    frames.append(frame)
            except Exception as exc:  # noqa: BLE001
                errors.append(code)
                print(f"[WARN] {api_name} {code} 增量失败: {exc}")
    if errors:
        raise RuntimeError(f"{api_name} 增量有 {len(errors)} 个代码失败，拒绝写入不完整结果。")
    if frames:
        incoming = pd.concat(frames, ignore_index=True)
        append_incremental_rows(
            incoming,
            out_path,
            subset=["source_api", "ts_code", "trade_date"],
            sort_cols=["ts_code", "trade_date"],
            date_column="trade_date",
        )


def save_index_history_scope(
    pro: Any,
    output_dir: Path,
    limiter: RateLimiter,
    args: argparse.Namespace,
    action: str,
) -> None:
    for api_name in INDEX_SCOPE_APIS[action]:
        save_index_history_api(pro, output_dir, limiter, args, api_name)


def _normalise_member_frame(frame: pd.DataFrame, api_name: str, index_code: Optional[str]) -> pd.DataFrame:
    if frame.empty:
        return frame
    result = pd.DataFrame(
        {
            "source_api": api_name,
            "index_code": _first_existing_column(frame, ["index_code", "l1_code", "ts_code"]),
            "con_code": _first_existing_column(frame, ["con_code", "con_ts_code", "stock_code", "member_code"]),
            "member_name": _first_existing_column(frame, ["con_name", "name", "stock_name", "member_name"]),
            "in_date": _first_existing_column(frame, ["in_date", "in_time"]),
            "out_date": _first_existing_column(frame, ["out_date", "out_time"]),
            "is_new": _first_existing_column(frame, ["is_new"]),
            "weight": _first_existing_column(frame, ["weight"]),
            "trade_date": _first_existing_column(frame, ["trade_date"]),
        }
    )
    if index_code:
        result["index_code"] = result["index_code"].fillna(index_code)
    for column in ("in_date", "out_date", "trade_date"):
        result[column] = pd.to_datetime(result[column], errors="coerce")
    return result.dropna(subset=["index_code"])


def save_index_constituents(
    pro: Any, output_dir: Path, limiter: RateLimiter, args: argparse.Namespace
) -> None:
    catalog = pd.read_parquet(output_dir / "index_catalog_df.parquet")
    members_path = output_dir / "index_members_df.parquet"
    member_stage_marker = output_dir / ".tushare_stage_index_constituents_members.json"
    resume_key = _action_resume_key(args)
    member_stage = read_json_object(member_stage_marker) if getattr(args, "resume", False) else None
    reuse_members = bool(
        members_path.exists()
        and member_stage is not None
        and member_stage.get("resume_key") == resume_key
    )
    if reuse_members:
        print("[INFO] 指数成分已在当前候选目录完成，续跑直接进入最新权重。")
    else:
        member_frames: list[pd.DataFrame] = []
        member_requests: list[tuple[str, Optional[str]]] = [
            (api_name, None) for api_name in ("index_member_all", "ci_index_member")
        ]
        for api_name, source_api in (("ths_member", "ths_index"), ("dc_member", "dc_index"), ("tdx_member", "tdx_index")):
            codes = catalog.loc[catalog["source_api"].eq(source_api), "ts_code"].dropna().astype(str).unique()
            if args.limit:
                codes = codes[: args.limit]
            member_requests.extend((api_name, str(code)) for code in codes)

        def fetch_member(api_name: str, code: Optional[str]):
            kwargs = {"ts_code": code} if code else {}
            try:
                return api_name, code, _call_index_api(pro, api_name, limiter, args, **kwargs), None
            except Exception as exc:  # noqa: BLE001
                return api_name, code, pd.DataFrame(), exc

        member_worker_count = min(
            max(int(getattr(args, "max_workers", 1)), 1),
            max(len(member_requests), 1),
        )
        member_failures = 0
        with ThreadPoolExecutor(max_workers=member_worker_count) as executor:
            futures = {
                executor.submit(fetch_member, api_name, code): (api_name, code)
                for api_name, code in member_requests
            }
            processed = 0
            for future in as_completed(futures):
                futures.pop(future, None)
                api_name, code, frame, error = future.result()
                if error is not None:
                    member_failures += 1
                    suffix = f" {code}" if code else ""
                    print(f"[WARN] {api_name}{suffix} 获取失败: {error}")
                elif not frame.empty:
                    member_frames.append(_normalise_member_frame(frame, api_name, code))
                processed += 1
                if processed % 100 == 0 or processed == len(member_requests):
                    print(
                        f"[INFO] 指数成分进度 {processed}/{len(member_requests)}，"
                        f"workers={member_worker_count}，异常 {member_failures}。"
                    )
        valid_member_frames = [item for item in member_frames if not item.empty]
        members = (
            pd.concat(valid_member_frames, ignore_index=True)
            if valid_member_frames
            else pd.DataFrame(
                columns=[
                    "source_api", "index_code", "con_code", "member_name", "in_date",
                    "out_date", "is_new", "weight", "trade_date",
                ]
            )
        )
        if members.empty:
            raise RuntimeError("未获得任何指数成分数据。")
        members = members.drop_duplicates(
            ["source_api", "index_code", "con_code"], keep="last"
        )
        save_dataframe(members, members_path)
        atomic_write_json(
            member_stage_marker,
            {
                "schema_version": 1,
                "stage": "index_constituents_members",
                "resume_key": resume_key,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        del members, member_frames, valid_member_frames

    active = catalog.get("status", pd.Series("active", index=catalog.index)).fillna("active").eq("active")
    codes = catalog.loc[
        catalog["quote_source_api"].eq("index_daily") & active,
        "ts_code",
    ].dropna().astype(str).unique()
    if args.limit:
        codes = codes[: args.limit]

    weights_path = output_dir / "index_weights_df.parquet"
    weight_checkpoint_dir = history_checkpoint_dir(weights_path, args)
    ensure_output_dir(weight_checkpoint_dir)

    def fetch_latest_weight(code: str):
        window_end = pd.Timestamp(args.end_date)
        window_start = max(pd.Timestamp(args.start_date), window_end - pd.Timedelta(days=119))
        # index_weight 是月度数据。当前规模只需要最新可用权重；若最近
        # 120 天无披露则记为无数据，不能为数千个不支持的指数回溯十余年。
        try:
            frame = _call_index_api(
                pro,
                "index_weight",
                limiter,
                args,
                allow_capped_response=True,
                context=(
                    f"index_weight {code} "
                    f"{window_start.strftime('%Y%m%d')}-{window_end.strftime('%Y%m%d')}"
                ),
                index_code=code,
                start_date=window_start.strftime("%Y%m%d"),
                end_date=window_end.strftime("%Y%m%d"),
            )
        except Exception as exc:  # noqa: BLE001
            return code, pd.DataFrame(), exc
        return code, frame, None

    code_paths = {
        str(code): history_checkpoint_paths(weight_checkpoint_dir, str(code))
        for code in codes
    }
    pending_codes = [
        str(code)
        for code in codes
        if not any(path.exists() for path in code_paths[str(code)])
    ]

    weight_worker_count = min(
        max(int(getattr(args, "max_workers", 1)), 1),
        max(len(pending_codes), 1),
    )
    weight_failures = 0
    with ThreadPoolExecutor(max_workers=weight_worker_count) as executor:
        futures = {
            executor.submit(fetch_latest_weight, str(code)): str(code)
            for code in pending_codes
        }
        processed = len(codes) - len(pending_codes)
        if processed:
            print(f"[INFO] 指数权重续跑复用检查点 {processed}/{len(codes)}。")
        for future in as_completed(futures):
            futures.pop(future, None)
            code, frame, error = future.result()
            if error is not None:
                weight_failures += 1
                print(f"[WARN] index_weight {code} 获取失败: {error}")
            else:
                normalised = _normalise_member_frame(frame, "index_weight", code)
                if not normalised.empty and normalised["trade_date"].notna().any():
                    latest = normalised["trade_date"].max()
                    normalised = normalised[normalised["trade_date"].eq(latest)]
                part_path, empty_path = code_paths[code]
                if normalised.empty:
                    mark_empty_checkpoint(empty_path)
                else:
                    save_dataframe(
                        normalised.drop_duplicates(
                            ["source_api", "index_code", "con_code"], keep="last"
                        ).sort_values(["index_code", "con_code"]),
                        part_path,
                        quiet=True,
                    )
            processed += 1
            if processed % 100 == 0 or processed == len(codes):
                print(
                    f"[INFO] 指数权重进度 {processed}/{len(codes)}，"
                    f"workers={weight_worker_count}，异常 {weight_failures}。"
                )
    if weight_failures:
        raise RuntimeError(
            f"index_weight 有 {weight_failures} 个代码失败；已保留成功检查点，续跑只重试失败项。"
        )
    incomplete_codes = [
        code for code, paths in code_paths.items() if not any(path.exists() for path in paths)
    ]
    if incomplete_codes:
        raise RuntimeError(f"index_weight 仍有 {len(incomplete_codes)} 个代码缺少检查点。")
    part_paths = [paths[0] for paths in code_paths.values() if paths[0].exists()]
    if not part_paths:
        raise RuntimeError("未获得任何最新指数权重。")
    consolidate_parquet_parts(
        part_paths,
        weights_path,
        primary_column="index_code",
    )


def save_index_coverage(output_dir: Path) -> None:
    from backend.services.index_data import build_index_coverage_snapshot

    result = build_index_coverage_snapshot(output_dir)
    print(f"[OK] 指数覆盖快照 {result['rows']} 行。")


def save_index_basic(pro: Any, output_dir: Path, limiter: RateLimiter, args: argparse.Namespace) -> None:
    markets = ["SSE", "SZSE", "CSI", "CICC", "SW", "MSCI", "OTH"]
    if args.smoke:
        markets = markets[:1]
    def fetch_market(market: str):
        try:
            df = call_tushare_api(
                pro.index_basic,
                limiter,
                max_retries=args.max_retries,
                backoff_sec=args.backoff_sec,
                wait_on_rate_limit_sec=args.wait_on_rate_limit_sec,
                context=f"index_basic {market}",
                market=market,
                fields=fields_arg(INDEX_BASIC_FIELDS),
            )
        except Exception as exc:  # noqa: BLE001
            return market, pd.DataFrame(), exc
        return market, df, None

    frames: list[pd.DataFrame] = []
    worker_count = min(
        max(int(getattr(args, "max_workers", 1)), 1),
        max(len(markets), 1),
    )
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {executor.submit(fetch_market, market): market for market in markets}
        for future in as_completed(futures):
            market, df, error = future.result()
            if error is not None:
                print(f"[WARN] index_basic {market} 获取失败: {error}")
                continue
            if args.limit:
                df = df.head(args.limit)
            if not df.empty:
                frames.append(df)
    if not frames:
        raise RuntimeError("未获得指数基础信息。")
    out = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=["ts_code"])
        .sort_values("ts_code")
    )
    save_dataframe(out, output_dir / "index_info.parquet")


def save_etf_index(pro: Any, output_dir: Path, limiter: RateLimiter, args: argparse.Namespace) -> None:
    df = call_tushare_api(
        pro.etf_index,
        limiter,
        max_retries=args.max_retries,
        backoff_sec=args.backoff_sec,
        wait_on_rate_limit_sec=args.wait_on_rate_limit_sec,
        context="etf_index",
        fields=fields_arg(ETF_INDEX_FIELDS),
    )
    if args.limit:
        df = df.head(args.limit)
    save_dataframe(df, output_dir / "etf_index.parquet")


def save_fund_company(pro: Any, output_dir: Path, limiter: RateLimiter, args: argparse.Namespace) -> None:
    df = call_tushare_api(
        pro.fund_company,
        limiter,
        max_retries=args.max_retries,
        backoff_sec=args.backoff_sec,
        wait_on_rate_limit_sec=args.wait_on_rate_limit_sec,
        retry_jitter_sec=getattr(args, "retry_jitter_sec", 0.25),
        context="fund_company",
        fields=fields_arg(FUND_COMPANY_FIELDS),
    )
    if df.empty:
        raise RuntimeError("未获得公募基金管理人信息。")
    if "setup_date" in df.columns:
        df["setup_date"] = date_series(df["setup_date"])
    if "end_date" in df.columns:
        df["end_date"] = date_series(df["end_date"])
    save_dataframe(df, output_dir / "fund_company_df.parquet")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch platform data from Tushare only.")
    parser.add_argument("--output-dir", type=Path, default=None, help="输出目录；默认 data，smoke 默认临时目录。")
    parser.add_argument("--start-date", default=DEFAULT_START_DATE, help="开始日期 YYYYMMDD。")
    parser.add_argument("--end-date", default=TODAY, help="结束日期 YYYYMMDD。")
    parser.add_argument("--max-workers", type=int, default=16, help="Tushare 下载线程池并发数。")
    parser.add_argument("--max-retries", type=int, default=3, help="单次接口最大重试次数。")
    parser.add_argument(
        "--empty-response-retries",
        type=int,
        default=1,
        help="按代码全历史请求全部为空时的额外确认次数，默认 1；仅用于避免瞬时空响应被写入检查点。",
    )
    parser.add_argument("--backoff-sec", type=float, default=1.5, help="重试退避初始秒数。")
    parser.add_argument("--max-calls-per-minute", type=int, default=450, help="Tushare 每分钟调用安全上限。")
    parser.add_argument(
        "--min-call-interval-sec",
        type=float,
        default=0.13,
        help="任意两次 Tushare 调用的最小间隔秒数，默认 0.13 秒。",
    )
    parser.add_argument("--wait-on-rate-limit-sec", type=float, default=10.0, help="限流错误后的等待秒数。")
    parser.add_argument("--retry-jitter-sec", type=float, default=0.5, help="重试等待的随机抖动上限秒数。")
    parser.add_argument(
        "--history-chunk-days",
        type=int,
        default=3650,
        help="历史 fund_daily 单次请求的最大日历窗口，默认约 10 年。",
    )
    parser.add_argument(
        "--max-fund-basic-pages",
        type=int,
        default=DEFAULT_INCREMENTAL_BATCH_DAYS,
        help="fund_basic 每个市场/状态允许的最大分页数，防止服务端异常导致无限循环。",
    )
    parser.add_argument(
        "--max-fund-nav-pages",
        type=int,
        default=20,
        help="场外 fund_nav 每个净值日期允许的最大分页数。",
    )
    parser.add_argument(
        "--fund-nav-page-size",
        type=int,
        default=DEFAULT_FUND_NAV_PAGE_SIZE,
        help="场外 fund_nav 按日期增量的分页大小，默认 10000。",
    )
    parser.add_argument("--limit", type=int, default=None, help="限制处理标的数量，便于验证。")
    parser.add_argument("--smoke", action="store_true", help="小样本验证，默认写临时目录且限制 2 个标的。")
    parser.add_argument("--missing-only", action="store_true", help="仅补抓现有净值/K线 parquet 中缺失的 ETF。")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="续跑同一隔离候选目录；跳过已完成任务，未完成历史仅重试缺失检查点。",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="按本地 parquet 最新日期逐交易日增量更新 ETF 信息、日历、净值和行情。",
    )
    parser.add_argument(
        "--max-latest-days",
        type=int,
        default=120,
        help="--latest 单数据集允许补抓的最大新交易日数，默认 120；回看日另计。",
    )
    parser.add_argument(
        "--incremental-batch-days",
        type=int,
        default=20,
        help="增量结果每累计多少个交易日执行一次流式原子归并，默认 20。",
    )
    parser.add_argument(
        "--incremental-lookback-days",
        type=int,
        default=5,
        help="增量更新重复拉取最近多少个交易日以回补迟报和修订，默认 5。",
    )

    parser.add_argument("--all", action="store_true", help="执行全部数据任务。")
    parser.add_argument("--etf-info", action="store_true", help="更新 data/etf_info_df.parquet。")
    parser.add_argument("--nav", action="store_true", help="更新 data/etf_daily_df.parquet。")
    parser.add_argument("--candle", action="store_true", help="更新 data/etf_daily_candle_df.parquet。")
    parser.add_argument("--calendar", action="store_true", help="更新 data/trade_day_df.parquet。")
    parser.add_argument("--stock-basic", action="store_true", help="更新 data/stock_basic.parquet。")
    parser.add_argument("--index-info", action="store_true", help="更新 data/index_info.parquet。")
    parser.add_argument("--etf-index", action="store_true", help="更新 data/etf_index.parquet。")
    parser.add_argument("--fund-info", action="store_true", help="更新场外公募基金 data/fund_info_df.parquet。")
    parser.add_argument("--fund-nav", action="store_true", help="更新场外公募基金 data/fund_nav_df.parquet。")
    parser.add_argument("--fund-company", action="store_true", help="更新 data/fund_company_df.parquet。")
    parser.add_argument("--index-catalog", action="store_true", help="更新指数原始目录与统一目录。")
    parser.add_argument("--index-domestic", action="store_true", help="更新境内指数日线。")
    parser.add_argument("--index-industry", action="store_true", help="更新申万与中信行业指数日线。")
    parser.add_argument("--index-concept", action="store_true", help="更新同花顺、东财与通达信概念指数。")
    parser.add_argument("--index-global", action="store_true", help="更新国际指数日线。")
    parser.add_argument("--index-futures", action="store_true", help="更新商品期货指数日线。")
    parser.add_argument("--index-valuation", action="store_true", help="更新指数估值数据。")
    parser.add_argument("--index-constituents", action="store_true", help="更新指数成分与最新权重。")

    args = parser.parse_args()
    args.output_dir_explicit = args.output_dir is not None
    args.start_date = normalize_yyyymmdd(args.start_date, default=DEFAULT_START_DATE)
    args.end_date = normalize_yyyymmdd(args.end_date, default=TODAY)
    if args.start_date > args.end_date:
        parser.error("--start-date 不能晚于 --end-date。")
    if args.latest and args.smoke:
        parser.error("--latest 与 --smoke 不能同时使用。")
    if args.latest and args.missing_only:
        parser.error("--latest 与 --missing-only 语义不同，不能同时使用。")
    if args.latest and args.resume:
        parser.error("--latest 与 --resume 不能同时使用。")
    if args.max_latest_days < 1:
        parser.error("--max-latest-days 必须大于 0。")
    if args.max_workers < 1:
        parser.error("--max-workers 必须大于 0。")
    if args.empty_response_retries < 0:
        parser.error("--empty-response-retries 不能小于 0。")
    if args.incremental_batch_days < 1:
        parser.error("--incremental-batch-days 必须大于 0。")
    if args.incremental_lookback_days < 1:
        parser.error("--incremental-lookback-days 必须大于 0。")
    if args.max_calls_per_minute < 1:
        parser.error("--max-calls-per-minute 必须大于 0。")
    if args.min_call_interval_sec < 0:
        parser.error("--min-call-interval-sec 不能小于 0。")
    if args.retry_jitter_sec < 0:
        parser.error("--retry-jitter-sec 不能小于 0。")
    if args.history_chunk_days < 1:
        parser.error("--history-chunk-days 必须大于 0。")
    if args.max_fund_basic_pages < 1:
        parser.error("--max-fund-basic-pages 必须大于 0。")
    if args.max_fund_nav_pages < 1:
        parser.error("--max-fund-nav-pages 必须大于 0。")
    if args.fund_nav_page_size < 1:
        parser.error("--fund-nav-page-size 必须大于 0。")
    if args.smoke:
        args.limit = args.limit or 2
        args.max_workers = min(args.max_workers, 2)
        args.max_calls_per_minute = min(args.max_calls_per_minute, 20)
        if args.output_dir is None:
            args.output_dir = Path(tempfile.mkdtemp(prefix="tushare_smoke_"))
        elif is_project_data_dir(args.output_dir):
            parser.error("--smoke 不能写入项目 data 目录，请移除 --output-dir 或指定临时目录。")
    else:
        args.output_dir = args.output_dir or Path("data")
    return args


def selected_actions(args: argparse.Namespace) -> list[str]:
    if args.latest and args.all:
        return [
            "etf_info",
            "fund_info",
            "fund_company",
            "calendar",
            "stock_basic",
            "nav",
            "candle",
            "fund_nav",
            *INDEX_SCOPE_ACTIONS.values(),
            "index_coverage",
        ]
    actions = []
    if args.all or args.etf_info:
        actions.append("etf_info")
    if args.all or args.nav:
        actions.append("nav")
    if args.all or args.candle:
        actions.append("candle")
    if args.all or args.calendar:
        actions.append("calendar")
    if args.all or args.stock_basic:
        actions.append("stock_basic")
    if args.all or args.index_info:
        actions.append("index_info")
    if args.all or args.etf_index:
        actions.append("etf_index")
    if args.all or args.fund_info:
        actions.append("fund_info")
    if args.all or args.fund_nav:
        actions.append("fund_nav")
    if args.all or args.fund_company:
        actions.append("fund_company")
    for scope, action in INDEX_SCOPE_ACTIONS.items():
        if args.all or bool(getattr(args, f"index_{scope}", False)):
            actions.append(action)
    if any(action in INDEX_ACTIONS for action in actions):
        if "index_catalog" not in actions:
            actions.append("index_catalog")
        actions.append("index_coverage")
    if "index_catalog" in actions:
        actions = [item for item in actions if item not in {"index_info", "etf_index"}]
    return list(dict.fromkeys(actions))


def _action_resume_key(args: argparse.Namespace) -> str:
    payload = {
        "start_date": args.start_date,
        "end_date": args.end_date,
        "history_chunk_days": getattr(args, "history_chunk_days", None),
        "missing_only": bool(getattr(args, "missing_only", False)),
        "latest": bool(getattr(args, "latest", False)),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20]


def _action_marker_path(output_dir: Path, action: str) -> Path:
    if not re.fullmatch(r"[a-z_]+", action):
        raise ValueError(f"不安全的任务标识: {action!r}")
    return output_dir / f".tushare_action_{action}.json"


def _run_action_once(
    args: argparse.Namespace,
    action: str,
    operation: Callable[[], Any],
) -> Any:
    """Run one action and durably mark it complete for same-candidate resume."""

    marker = _action_marker_path(args.output_dir, action)
    resume_key = _action_resume_key(args)
    if getattr(args, "resume", False):
        payload = read_json_object(marker)
        if payload is not None and payload.get("resume_key") == resume_key:
            print(f"[INFO] {action} 已在当前候选目录完成，续跑跳过。")
            return None
    result = operation()
    atomic_write_json(
        marker,
        {
            "schema_version": 1,
            "action": action,
            "resume_key": resume_key,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    return result


def _run_actions(args: argparse.Namespace, actions: list[str]) -> None:
    if args.latest:
        unsupported = set(actions) - {
            "etf_info",
            "fund_info",
            "fund_company",
            "calendar",
            "stock_basic",
            "index_info",
            "etf_index",
            "nav",
            "candle",
            "fund_nav",
            *INDEX_ACTIONS,
        }
        if unsupported:
            raise ValueError(f"--latest 不支持这些任务: {', '.join(sorted(unsupported))}")

    ensure_output_dir(args.output_dir)
    token = require_tushare_token()
    pro = ts.pro_api(token)
    limiter = RateLimiter(args.max_calls_per_minute, min_interval_sec=args.min_call_interval_sec)

    print(f"[INFO] 输出目录: {args.output_dir.resolve()}")
    mode = "按交易日增量" if args.latest else "指定范围"
    print(f"[INFO] 日期范围: {args.start_date} - {args.end_date}；模式: {mode}；任务: {', '.join(actions)}")

    calendar_start: Optional[str] = None
    if args.latest:
        starts: list[str] = []
        if "calendar" in actions:
            starts.append(next_calendar_date(latest_parquet_date(args.output_dir / "trade_day_df.parquet", "cal_date")))
        if "nav" in actions:
            starts.append(next_calendar_date(latest_parquet_date(args.output_dir / "etf_daily_df.parquet", "date")))
        if "candle" in actions:
            starts.append(next_calendar_date(latest_parquet_date(args.output_dir / "etf_daily_candle_df.parquet", "date")))
        if "fund_nav" in actions:
            starts.append(next_calendar_date(latest_parquet_date(args.output_dir / "fund_nav_df.parquet", "date")))
        calendar_start = min(starts) if starts else None

    etf_info: Optional[pd.DataFrame] = None
    fund_info: Optional[pd.DataFrame] = None
    if "etf_info" in actions:
        etf_info = _run_action_once(
            args, "etf_info", lambda: save_etf_info(pro, args.output_dir, limiter, args)
        )
    if "fund_info" in actions:
        fund_info = _run_action_once(
            args, "fund_info", lambda: save_public_fund_info(pro, args.output_dir, limiter, args)
        )
    if "fund_company" in actions:
        _run_action_once(
            args, "fund_company", lambda: save_fund_company(pro, args.output_dir, limiter, args)
        )
    if "calendar" in actions:
        _run_action_once(
            args,
            "calendar",
            lambda: save_trade_calendar(
                pro, args.output_dir, limiter, args, incremental_start=calendar_start
            ),
        )
    if "nav" in actions:
        _run_action_once(
            args,
            "nav",
            lambda: save_latest_nav(pro, args.output_dir, limiter, args, etf_info)
            if args.latest
            else save_nav(pro, args.output_dir, limiter, args, etf_info),
        )
    if "candle" in actions:
        _run_action_once(
            args,
            "candle",
            lambda: save_latest_candles(pro, args.output_dir, limiter, args, etf_info)
            if args.latest
            else save_candles(pro, args.output_dir, limiter, args, etf_info),
        )
    if "fund_nav" in actions:
        _run_action_once(
            args,
            "fund_nav",
            lambda: save_latest_public_fund_nav(pro, args.output_dir, limiter, args, fund_info)
            if args.latest
            else save_public_fund_nav(pro, args.output_dir, limiter, args, fund_info),
        )
    if "stock_basic" in actions:
        _run_action_once(
            args, "stock_basic", lambda: save_stock_basic(pro, args.output_dir, limiter, args)
        )
    if "index_info" in actions:
        _run_action_once(
            args, "index_info", lambda: save_index_basic(pro, args.output_dir, limiter, args)
        )
    if "etf_index" in actions:
        _run_action_once(
            args, "etf_index", lambda: save_etf_index(pro, args.output_dir, limiter, args)
        )
    if "index_catalog" in actions:
        _run_action_once(
            args, "index_catalog", lambda: save_index_catalog(pro, args.output_dir, limiter, args)
        )
    for action in (
        "index_domestic", "index_industry", "index_concept", "index_global",
        "index_futures", "index_valuation",
    ):
        if action in actions:
            _run_action_once(
                args,
                action,
                lambda selected=action: save_index_history_scope(
                    pro, args.output_dir, limiter, args, selected
                ),
            )
    if "index_constituents" in actions:
        _run_action_once(
            args,
            "index_constituents",
            lambda: save_index_constituents(pro, args.output_dir, limiter, args),
        )
    if "index_coverage" in actions:
        _run_action_once(
            args, "index_coverage", lambda: save_index_coverage(args.output_dir)
        )


def _modules_for_actions(actions: list[str]) -> list[str]:
    selected = set(actions)
    modules = []
    if selected & {"calendar", "stock_basic", "index_info", "fund_company"}:
        modules.append("base")
    if selected & {"etf_info", "nav", "candle", "etf_index"}:
        modules.append("etf")
    if selected & {"fund_info", "fund_nav"}:
        modules.append("fund")
    if selected & INDEX_ACTIONS:
        modules.append("index")
    return modules


def _index_scopes_for_actions(actions: list[str]) -> list[str]:
    selected = set(actions)
    return [scope for scope, action in INDEX_SCOPE_ACTIONS.items() if action in selected]


def _write_cli_refresh_state(job: dict[str, Any]) -> None:
    atomic_write_json(
        GLOBAL_REFRESH_STATE_PATH,
        {
            "schema_version": 1,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "job": job,
        },
    )


def _cli_heartbeat_loop(
    job: dict[str, Any],
    stop_event: threading.Event,
    *,
    interval_seconds: Optional[float] = None,
) -> None:
    resolved_interval = CLI_HEARTBEAT_INTERVAL_SECONDS if interval_seconds is None else interval_seconds
    while not stop_event.wait(max(resolved_interval, 0.01)):
        if job.get("status") != "running":
            return
        job["heartbeat_at"] = datetime.now(timezone.utc).isoformat()
        try:
            _write_cli_refresh_state(job)
        except OSError:
            # The lock remains authoritative.  A final state write is retried
            # by the main thread after acquisition and local post-processing.
            pass


def _rebuild_cli_analytics(data_dir: Path) -> dict[str, Any]:
    from backend.services.instrument_analytics import rebuild_analytics_snapshot

    return rebuild_analytics_snapshot(data_dir)


def _stop_cli_heartbeat(stop_event: threading.Event, thread: threading.Thread) -> None:
    stop_event.set()
    if thread.ident is not None:
        thread.join(timeout=5.0)


def _cli_request_fingerprint(args: argparse.Namespace, actions: list[str]) -> str:
    payload = {
        "actions": actions,
        "start_date": getattr(args, "start_date", None),
        "end_date": getattr(args, "end_date", None),
        "history_chunk_days": getattr(args, "history_chunk_days", None),
        "missing_only": bool(getattr(args, "missing_only", False)),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


def _safe_cli_candidate(value: Any) -> Optional[Path]:
    if not value:
        return None
    root = (PROJECT_ROOT / "data").resolve()
    try:
        candidate = Path(str(value)).expanduser().resolve()
    except (OSError, RuntimeError, TypeError, ValueError):
        return None
    if root not in candidate.parents or not candidate.name.startswith("tushare_snapshot_"):
        return None
    return candidate if candidate.is_dir() else None


def _prepare_direct_cli_output(
    args: argparse.Namespace,
    actions: list[str],
    job_id: str,
) -> tuple[bool, bool, str]:
    """Resolve active incremental data or an isolated resumable full candidate."""

    if bool(getattr(args, "smoke", False)):
        return False, False, _cli_request_fingerprint(args, actions)
    from backend.market_data import resolve_tushare_data_dir

    data_root = (PROJECT_ROOT / "data").resolve()
    active_dir = resolve_tushare_data_dir(data_root)
    explicit = bool(getattr(args, "output_dir_explicit", True))
    explicit_path = Path(args.output_dir).expanduser().resolve()
    if explicit and explicit_path not in {data_root, active_dir}:
        return False, False, _cli_request_fingerprint(args, actions)
    if bool(getattr(args, "latest", False)):
        args.output_dir = active_dir
        return False, False, _cli_request_fingerprint(args, actions)

    from backend.services.data_refresh import prepare_full_refresh_staging

    fingerprint = _cli_request_fingerprint(args, actions)
    persisted = read_json_object(GLOBAL_REFRESH_STATE_PATH) or {}
    previous = persisted.get("job") if isinstance(persisted.get("job"), dict) else {}
    analytics = previous.get("analytics_snapshot") if isinstance(previous, dict) else None
    eligible = bool(
        isinstance(previous, dict)
        and previous.get("mode") == "full"
        and previous.get("request_fingerprint") == fingerprint
        and (
            previous.get("status") == "failed"
            or (isinstance(analytics, dict) and analytics.get("status") == "failed")
        )
    )
    candidate = _safe_cli_candidate(previous.get("staging_data_dir")) if eligible else None
    fetch_complete = bool(previous.get("fetch_complete")) if candidate is not None else False
    if candidate is None:
        candidate = prepare_full_refresh_staging(
            data_root=data_root,
            source_dir=active_dir,
            job_id=job_id,
        )
    else:
        args.resume = True
    args.output_dir = candidate
    return True, fetch_complete, fingerprint


def main() -> None:
    args = parse_args()
    actions = selected_actions(args)
    if not actions:
        print("未指定数据任务。使用 --help 查看可用参数。")
        return
    if os.getenv(PARENT_LOCK_ENV, "").strip() == "1":
        _run_actions(args, actions)
        return

    job_id = f"cli-{uuid.uuid4().hex}"
    process_lock = InterProcessFileLock(GLOBAL_REFRESH_LOCK_PATH)
    started_at = datetime.now(timezone.utc).isoformat()
    owner = f"job_id={job_id}\npid={os.getpid()}\nstarted_at={started_at}\n"
    if not process_lock.acquire(owner=owner):
        raise RuntimeError("已有 Tushare 数据任务正在运行；为避免重复抓取，本次命令未启动。")
    try:
        managed_full, fetch_complete, request_fingerprint = _prepare_direct_cli_output(
            args, actions, job_id
        )
    except Exception:
        process_lock.release()
        raise
    job = {
        "job_id": job_id,
        "status": "running",
        "started_at": started_at,
        "finished_at": None,
        "modules": _modules_for_actions(actions),
        "index_scopes": _index_scopes_for_actions(actions),
        "mode": "incremental" if args.latest else "full",
        "message": "正在从命令行执行 Tushare 数据更新",
        "log_tail": "",
        "owner_pid": os.getpid(),
        "worker_pid": os.getpid(),
        "heartbeat_at": started_at,
        "analytics_snapshot": None,
        "warnings": [],
        "output_dir": str(args.output_dir.resolve()),
        "request_fingerprint": request_fingerprint,
        "staging_data_dir": str(args.output_dir.resolve()) if managed_full else None,
        "fetch_complete": fetch_complete,
        "resumed": bool(managed_full and getattr(args, "resume", False)),
    }
    heartbeat_stop = threading.Event()
    heartbeat_thread = threading.Thread(
        target=_cli_heartbeat_loop,
        args=(job, heartbeat_stop),
        name=f"tushare-cli-heartbeat-{job_id[-8:]}",
        daemon=True,
    )
    try:
        _write_cli_refresh_state(job)
        heartbeat_thread.start()
        if fetch_complete:
            print(
                "[INFO] 已复用抓取完成的候选数据，本次仅重建并验收分析快照，"
                "不调用 Tushare。"
            )
        else:
            _run_actions(args, actions)
            job["fetch_complete"] = True
            _write_cli_refresh_state(job)
    except Exception as exc:
        _stop_cli_heartbeat(heartbeat_stop, heartbeat_thread)
        job.update(
            {
                "status": "failed",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "message": f"Tushare 命令行数据更新失败: {_sanitise_cli_error(exc)}",
            }
        )
        _write_cli_refresh_state(job)
        raise
    else:
        job["message"] = "Tushare 数据抓取完成，正在本地重建分析快照"
        _write_cli_refresh_state(job)
        try:
            snapshot_result = _rebuild_cli_analytics(args.output_dir)
            job["analytics_snapshot"] = {
                "status": "succeeded",
                "rebuilt_at": datetime.now(timezone.utc).isoformat(),
                **snapshot_result,
            }
            if managed_full:
                from backend.services.data_refresh import validate_and_activate_full_refresh

                promotion = validate_and_activate_full_refresh(
                    args.output_dir,
                    data_root=(PROJECT_ROOT / "data").resolve(),
                    **(
                        {"index_scopes": _index_scopes_for_actions(actions)}
                        if _index_scopes_for_actions(actions)
                        else {}
                    ),
                )
                job["analytics_snapshot"]["validation_status"] = promotion["validation"][
                    "status"
                ]
                job["analytics_snapshot"]["activated_snapshot_dir"] = promotion["manifest"][
                    "snapshot_dir"
                ]
                message = "Tushare 命令行全量数据验收完成并已原子切换"
            else:
                message = "Tushare 命令行数据更新及分析快照重建完成"
        except Exception as exc:  # noqa: BLE001
            warning_message = _sanitise_cli_error(exc)
            job["analytics_snapshot"] = {
                "status": "failed",
                "failed_at": datetime.now(timezone.utc).isoformat(),
                "message": warning_message,
            }
            job["warnings"] = [
                {"code": "ANALYTICS_REBUILD_FAILED", "message": warning_message}
            ]
            message = (
                "Tushare 命令行全量抓取完成，但候选未通过快照验收/接入；"
                "旧版本继续服务，重启同请求将复用候选，无需重新拉取数据"
                if managed_full
                else "Tushare 命令行数据抓取完成；分析快照重建失败，可单独重建，"
                "无需重新拉取数据"
            )
        _stop_cli_heartbeat(heartbeat_stop, heartbeat_thread)
        job.update(
            {
                "status": "succeeded",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "message": message,
            }
        )
        _write_cli_refresh_state(job)
    finally:
        process_lock.release()


if __name__ == "__main__":
    main()

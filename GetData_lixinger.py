# -*- coding: utf-8 -*-
"""
理杏仁 API 数据抓取整合脚本：
1) 拉取全量基金基础信息 -> 保存
2) 从中筛选 ETF 基金（交易所 in {'SZ','SH'}）-> 保存
3) 拉取 ETF 的基金档案（分批）-> 保存
4) 拉取“指数信息API”（可全量或按代码分批）-> 保存

API:
- 基金基础信息: POST https://open.lixinger.com/api/cn/fund
  Body: {"token": "...", "stockCodes": []}  # stockCodes 可省略=全量
- 基金档案:   POST https://open.lixinger.com/api/cn/fund/profile
  Body: {"token": "...", "stockCodes": [...]}
- 指数信息:   POST https://open.lixinger.com/api/cn/index
  Body: {"token": "..."} 或 {"token": "...", "stockCodes": [...]}

保存：Excel + Parquet
"""
import math
import time
import json
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests

# ======== pandas 显示选项 ========
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)


# =========================
# 一、通用：HTTP Client + 保存工具
# =========================
class ApiClient:
    """带 Session 的轻量重试 HTTP 客户端"""

    def __init__(self, base_timeout: int = 30, max_retries: int = 3, backoff: float = 1.5):
        self.sess = requests.Session()
        self.base_timeout = base_timeout
        self.max_retries = max_retries
        self.backoff = backoff
        self.headers = {"Content-Type": "application/json"}

    def post_json(self, url: str, payload: Dict[str, Any]) -> Any:
        last_err: Optional[Exception] = None
        for i in range(self.max_retries):
            try:
                resp = self.sess.post(url, headers=self.headers,
                                      data=json.dumps(payload), timeout=self.base_timeout)
                # 明确处理 4xx/5xx
                if 500 <= resp.status_code:
                    raise requests.HTTPError(f"Server {resp.status_code}: {resp.text[:200]}")
                if 400 <= resp.status_code < 500:
                    raise requests.HTTPError(f"Client {resp.status_code}: {resp.text[:200]}")
                data = resp.json()
                return data
            except Exception as e:
                last_err = e
                time.sleep(self.backoff ** (i + 1))
        raise last_err

    @staticmethod
    def unwrap_data(resp: Any) -> List[Dict[str, Any]]:
        """兼容两种返回结构：list 或 {'data': list}"""
        if isinstance(resp, list):
            return resp
        if isinstance(resp, dict) and "data" in resp and isinstance(resp["data"], list):
            return resp["data"]
        raise ValueError(f"未知返回结构：{type(resp)} -> {str(resp)[:200]}")


def save_table(df: pd.DataFrame, xlsx_path: str, pq_path: str,
               datetime_cols: Tuple[str, ...] = ()) -> None:
    if df.empty:
        print(f"[WARN] DataFrame 为空：{xlsx_path}, {pq_path} 未写出。")
        return

    # 1) 优先处理指定的时间列
    for c in datetime_cols:
        if c in df.columns:
            # 明确指定格式（带时区的 ISO8601）
            df[c] = pd.to_datetime(df[c], format="%Y-%m-%dT%H:%M:%S%z",
                                   errors="coerce", utc=True).dt.tz_localize(None)

    # 2) 扫描其它所有 tz-aware 的列
    for c in df.columns:
        s = df[c]
        if isinstance(s.dtype, pd.DatetimeTZDtype):
            df[c] = s.dt.tz_convert("UTC").dt.tz_localize(None)

    # 保存
    df.to_parquet(pq_path, index=False)
    with pd.ExcelWriter(xlsx_path, engine="xlsxwriter",
                        datetime_format="yyyy-mm-dd HH:MM:SS") as writer:
        df.to_excel(writer, sheet_name="data", index=False)
        ws = writer.sheets["data"]
        for i, col in enumerate(df.columns):
            width = min(max(len(col), int(df[col].astype(str).head(500).str.len().max()) + 2), 60)
            ws.set_column(i, i, width)


# =========================
# 二、数据层：基金 / 档案 / 指数
# =========================
def fetch_all_funds(token: str, client: ApiClient) -> pd.DataFrame:
    """基金基础信息（全量或按 stockCodes）；此处默认全量"""
    url = "https://open.lixinger.com/api/cn/fund"
    payload = {"token": token}  # 不传 stockCodes 即全量
    data = client.post_json(url, payload)
    rows = client.unwrap_data(data)
    df = pd.DataFrame(rows)
    # 常用字段优先排序
    preferred = [
        "name", "shortName", "stockCode",
        "fundFirstLevel", "fundSecondLevel",
        "areaCode", "market", "exchange", "inceptionDate",
    ]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    df = df.loc[:, cols]
    # 去重（以 stockCode + market）
    keys = [c for c in ["stockCode", "market"] if c in df.columns]
    if keys:
        df = df.drop_duplicates(subset=keys, keep="first")
    else:
        df = df.drop_duplicates(keep="first")
    return df


def filter_etf(df_fund: pd.DataFrame) -> pd.DataFrame:
    """筛选 ETF（交易所字段 SZ/SH 常见；大小写统一）"""
    if "exchange" not in df_fund.columns:
        return df_fund.iloc[0:0].copy()
    x = df_fund["exchange"].astype(str).str.lower()
    etf = df_fund[x.isin({"sz", "sh"})].copy()
    etf.reset_index(drop=True, inplace=True)
    return etf


def fetch_fund_profiles(token: str, codes: List[str], client: ApiClient,
                        batch_size: int = 100) -> pd.DataFrame:
    """基金档案（分批拉取合并）"""
    url = "https://open.lixinger.com/api/cn/fund/profile"
    codes = [c.strip() for c in codes if str(c).strip()]
    if not codes:
        return pd.DataFrame()

    all_rows: List[Dict[str, Any]] = []
    for s in range(0, len(codes), batch_size):
        sub = codes[s: s + batch_size]
        payload = {"token": token, "stockCodes": sub}
        data = client.post_json(url, payload)
        rows = client.unwrap_data(data)
        if rows:
            all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    return df


def fetch_index_info(token: str,
                     client: ApiClient,
                     stock_codes: Optional[Iterable[str]] = None,
                     batch_size: int = 200) -> pd.DataFrame:
    """
    指数信息API：全量或按代码分批
    返回字段（常见）：name, stockCode, areaCode, market, fsTableType, source,
                     currency, series, launchDate, rebalancingFrequency, caculationMethod, ...
    """
    url = "https://open.lixinger.com/api/cn/index"

    def _one(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        data = client.post_json(url, payload)
        return client.unwrap_data(data)

    rows: List[Dict[str, Any]] = []
    if not stock_codes:
        rows = _one({"token": token})
    else:
        codes = [c.strip() for c in stock_codes if str(c).strip()]
        for s in range(0, len(codes), batch_size):
            sub = codes[s: s + batch_size]
            rows.extend(_one({"token": token, "stockCodes": sub}))

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # 按常用字段排布、去重
    preferred = [
        "name", "stockCode", "areaCode", "market",
        "fsTableType", "source", "currency", "series",
        "launchDate", "rebalancingFrequency", "caculationMethod",
    ]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    df = df.loc[:, cols]

    keys = [c for c in ["stockCode", "market"] if c in df.columns]
    if keys:
        df = df.drop_duplicates(subset=keys, keep="first")
    else:
        df = df.drop_duplicates(keep="first")

    return df


if __name__ == "__main__":
    # 请替换为你的 Token
    LIXINGER_TOKEN = "d555552c-a42a-403c-a498-0c863ee219c3"

    client = ApiClient(base_timeout=30, max_retries=3, backoff=1.5)

    # # 1) 拉取全市场基金
    # print("[1/4] 拉取全市场基金基础信息 ...")
    # df_all_funds = fetch_all_funds(LIXINGER_TOKEN, client)
    # save_table(df_all_funds, "all_funds.xlsx", "all_funds.parquet",
    #            datetime_cols=("inceptionDate",))
    # print(f"  -> 基金条数: {len(df_all_funds)}")
    #
    # # 2) 筛选 ETF
    # print("[2/4] 从基金中筛选 ETF ...")
    # df_etf = filter_etf(df_all_funds)
    # save_table(df_etf, "etf_funds.xlsx", "etf_funds.parquet",
    #            datetime_cols=("inceptionDate",))
    # print(f"  -> ETF 条数: {len(df_etf)}")
    #
    # # 3) 拉取 ETF 档案
    # print("[3/4] 拉取 ETF 基金档案 ...")
    # etf_codes = df_etf["stockCode"].astype(str).tolist() if "stockCode" in df_etf.columns else []
    # df_profiles = fetch_fund_profiles(LIXINGER_TOKEN, etf_codes, client, batch_size=100)
    # save_table(df_profiles, "fund_profiles.xlsx", "fund_profiles.parquet")
    # print(f"  -> 档案条数: {len(df_profiles)}")

    # 4) 拉取 指数信息（可全量）
    print("[4/4] 拉取 指数信息API（全量） ...")
    df_index = fetch_index_info(LIXINGER_TOKEN, client, stock_codes=None, batch_size=200)
    # 若只想拉部分指数，可传 stock_codes=list(...)
    save_table(df_index, "index_info.xlsx", "index_info.parquet",
               datetime_cols=("launchDate",))
    print(f"  -> 指数条数: {len(df_index)}")

    print("✅ 全部完成：all_funds / etf_funds / fund_profiles / index_info 已生成。")

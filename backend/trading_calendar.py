"""Trading calendar utilities sourced from data/trade_day_df.parquet."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@lru_cache(maxsize=4)
def _load_calendar(exchange: str = "SSE") -> pd.DatetimeIndex:
    path = DATA_DIR / "trade_day_df.parquet"
    if not path.exists():
        raise FileNotFoundError(f"未找到交易日数据文件: {path}")

    df = pd.read_parquet(path, columns=["exchange", "cal_date", "is_open"])
    df = df[df["exchange"].str.upper() == exchange.upper()].copy()
    if df.empty:
        raise ValueError(f"交易日数据中没有交易所 {exchange} 的记录")

    df["cal_date"] = pd.to_datetime(df["cal_date"].astype(str), format="%Y%m%d", errors="coerce")
    df = df[df["is_open"].astype(int) == 1]
    df = df[df["cal_date"].notna()].sort_values("cal_date")

    return pd.DatetimeIndex(df["cal_date"].unique())


def get_trading_days(
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
    exchange: str = "SSE",
) -> pd.DatetimeIndex:
    """Return trading days for the given exchange filtered by [start, end]."""

    cal = _load_calendar(exchange)
    if start is not None:
        cal = cal[cal >= pd.Timestamp(start).normalize()]
    if end is not None:
        cal = cal[cal <= pd.Timestamp(end).normalize()]
    return cal

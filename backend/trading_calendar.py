"""Trading calendar utilities sourced from data/trade_day_df.parquet."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd

try:
    from backend.market_data import resolve_market_data_file
except ModuleNotFoundError:  # pragma: no cover - backend/ direct execution
    from market_data import resolve_market_data_file


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@lru_cache(maxsize=8)
def _load_calendar_file(
    path_text: str,
    _mtime_ns: int,
    _size: int,
    exchange: str,
) -> pd.DatetimeIndex:
    """Cache one immutable calendar file identity, not only its exchange."""

    del _mtime_ns, _size
    df = pd.read_parquet(path_text, columns=["exchange", "cal_date", "is_open"])
    df = df[df["exchange"].str.upper() == exchange.upper()].copy()
    if df.empty:
        raise ValueError(f"交易日数据中没有交易所 {exchange} 的记录")

    df["cal_date"] = pd.to_datetime(df["cal_date"].astype(str), format="%Y%m%d", errors="coerce")
    df = df[df["is_open"].astype(int) == 1]
    df = df[df["cal_date"].notna()].sort_values("cal_date")

    return pd.DatetimeIndex(df["cal_date"].unique())


def _load_calendar(exchange: str = "SSE") -> pd.DatetimeIndex:
    path = resolve_market_data_file("trade_day_df.parquet", DATA_DIR)
    if not path.exists():
        raise FileNotFoundError(f"未找到交易日数据文件: {path}")
    stat = path.stat()
    return _load_calendar_file(str(path), stat.st_mtime_ns, stat.st_size, exchange)


# Preserve the existing invalidation hook used by refresh orchestration.
_load_calendar.cache_clear = _load_calendar_file.cache_clear  # type: ignore[attr-defined]


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

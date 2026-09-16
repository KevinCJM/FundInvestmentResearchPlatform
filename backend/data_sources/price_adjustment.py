"""后复权因子与复权 OHLC：写回行情表的日频派生列。

因子口径固定为后复权，每只标的首个交易日为 1.0。数据源因子与 pre_close 推导
因子必须归一到同一基准，两条路径才可比、可交叉校验。
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

ADJUSTED_FIELDS = ("open", "high", "low", "close")
ADJUSTED_COLUMNS = tuple(f"adj_{name}" for name in ADJUSTED_FIELDS)
FACTOR_POLICIES = ("source", "source_then_pre_close", "pre_close")
DEFAULT_FACTOR_POLICY = "source"


class AdjustmentPolicyError(ValueError):
    """请求了未登记的复权因子口径。"""


def _codes_with_full_coverage(codes: pd.Series, values: pd.Series) -> pd.Index:
    complete = values.notna().groupby(codes).all()
    return complete[complete].index


def _normalise_by_first(codes: pd.Series, values: pd.Series) -> pd.Series:
    base = values.groupby(codes).transform("first")
    normalised = values / base
    return normalised.where(np.isfinite(normalised) & (normalised > 0))


def source_factor(frame: pd.DataFrame, official: pd.DataFrame | None) -> pd.Series:
    """数据源复权因子，按标的归一到首个交易日；覆盖不全的标的整体留空。"""

    empty = pd.Series(np.nan, index=frame.index, dtype="float64")
    if official is None or official.empty or "adj_factor" not in official.columns:
        return empty
    table = official[["ts_code", "date", "adj_factor"]].copy()
    table["date"] = pd.to_datetime(table["date"], errors="coerce")
    table["adj_factor"] = pd.to_numeric(table["adj_factor"], errors="coerce")
    table = table.dropna(subset=["ts_code", "date"]).sort_values(["ts_code", "date"], kind="mergesort")
    table = table.drop_duplicates(["ts_code", "date"], keep="last")
    merged = frame[["ts_code", "date"]].merge(table, on=["ts_code", "date"], how="left")
    values = pd.Series(merged["adj_factor"].to_numpy(), index=frame.index, dtype="float64")
    values = values.where(np.isfinite(values) & (values > 0))
    covered = _codes_with_full_coverage(frame["ts_code"], values)
    return _normalise_by_first(frame["ts_code"], values.where(frame["ts_code"].isin(covered)))


def pre_close_factor(frame: pd.DataFrame) -> pd.Series:
    """由前收盘价推导的后复权因子：F[0]=1，F[t]=F[t-1]·close[t-1]/pre_close[t]。

    pre_close 是除权调整后的前收盘价，所以 close[t]/pre_close[t] 就是含分红的
    当日真实收益，累乘即后复权因子。某一天比值不可用时该标的其后全部留空，
    不用 1 或相邻值补齐。
    """

    close = pd.to_numeric(frame["close"], errors="coerce")
    previous = pd.to_numeric(frame["pre_close"], errors="coerce")
    ratio = close.groupby(frame["ts_code"]).shift(1) / previous
    ratio = ratio.where(np.isfinite(ratio) & (ratio > 0))
    ratio = ratio.mask(~frame["ts_code"].eq(frame["ts_code"].shift(1)), 1.0)
    # cumprod 会跳过缺口继续累乘，等于用未除权的比值冒充复权；这里从第一个
    # 缺口起整段作废。
    broken = ratio.isna().groupby(frame["ts_code"]).cummax()
    return ratio.groupby(frame["ts_code"]).cumprod().mask(broken)


def factor_divergence(left: pd.Series, right: pd.Series) -> dict[str, Any]:
    """两条因子路径在同时可用的行上的相对偏离；用于暴露供应商口径差异。"""

    both = left.notna() & right.notna()
    if not bool(both.any()):
        return {"compared_rows": 0, "max_relative_difference": None}
    relative = ((left[both] - right[both]).abs() / right[both].abs()).replace([np.inf, -np.inf], np.nan)
    return {
        "compared_rows": int(both.sum()),
        "max_relative_difference": float(relative.max()) if relative.notna().any() else None,
    }


def attach_adjusted_prices(
    candle: pd.DataFrame,
    official: pd.DataFrame | None = None,
    *,
    policy: str = DEFAULT_FACTOR_POLICY,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """为行情表补上 adj_factor / adj_factor_source / 复权 OHLC。

    没有可用因子的标的这些列留空；下游据此报告复权口径指标不可计算，
    不会退回未复权价格冒充复权价格。
    """

    if policy not in FACTOR_POLICIES:
        raise AdjustmentPolicyError(f"未登记的复权因子口径: {policy}；可选 {', '.join(FACTOR_POLICIES)}。")
    required = {"ts_code", "date", "close", "pre_close", *ADJUSTED_FIELDS}
    missing = sorted(required - set(candle.columns))
    if missing:
        raise ValueError(f"行情表缺少复权计算所需字段: {missing}")

    frame = candle.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.sort_values(["ts_code", "date"], kind="mergesort").reset_index(drop=True)

    from_source = source_factor(frame, official) if policy != "pre_close" else pd.Series(
        np.nan, index=frame.index, dtype="float64"
    )
    from_pre_close = pre_close_factor(frame) if policy != "source" else pd.Series(
        np.nan, index=frame.index, dtype="float64"
    )
    divergence = factor_divergence(from_source, from_pre_close)

    factor = from_source.where(from_source.notna(), from_pre_close)
    origin = pd.Series(pd.NA, index=frame.index, dtype="string")
    origin[from_pre_close.notna()] = "pre_close"
    origin[from_source.notna()] = "source"
    origin[factor.isna()] = pd.NA

    frame["adj_factor"] = factor
    frame["adj_factor_source"] = origin
    for name in ADJUSTED_FIELDS:
        frame[f"adj_{name}"] = pd.to_numeric(frame[name], errors="coerce") * factor

    by_code = origin.groupby(frame["ts_code"]).agg(lambda values: values.dropna().iloc[0] if values.notna().any() else "none")
    final_factor = factor.groupby(frame["ts_code"]).agg(lambda values: values.dropna().iloc[-1] if values.notna().any() else np.nan)
    stats = {
        "policy": policy,
        "rows": int(len(frame)),
        "codes": int(frame["ts_code"].nunique()),
        "codes_by_factor_source": {str(key): int(value) for key, value in by_code.value_counts().items()},
        "codes_with_adjustment_events": int((final_factor.notna() & (final_factor.round(10) != 1.0)).sum()),
        "rows_without_factor": int(factor.isna().sum()),
        "source_vs_pre_close": divergence,
    }
    return frame, stats


__all__ = [
    "ADJUSTED_COLUMNS",
    "ADJUSTED_FIELDS",
    "AdjustmentPolicyError",
    "DEFAULT_FACTOR_POLICY",
    "FACTOR_POLICIES",
    "attach_adjusted_prices",
    "factor_divergence",
    "pre_close_factor",
    "source_factor",
]

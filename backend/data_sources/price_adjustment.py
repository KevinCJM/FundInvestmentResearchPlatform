"""后复权因子与复权 OHLC：写回行情表的日频派生列。

因子口径固定为后复权，每只标的首个交易日为 1.0。数据源因子与 pre_close 推导
因子必须归一到同一基准，两条路径才可比、可交叉校验。

本模块只做输入边界：读表、对齐、转成稳定 dtype 的 NumPy 数组、封装结果。
全部数值变换与分组聚合在 price_adjustment_numba 的固定签名 NJIT 内核里完成。
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .price_adjustment_numba import (
    ORIGIN_NONE,
    ORIGIN_PRE_CLOSE,
    ORIGIN_SOURCE,
    combine_factor_kernel,
    divergence_kernel,
    factor_summary_kernel,
    pre_close_factor_kernel,
    scale_kernel,
    source_factor_kernel,
)

ADJUSTED_FIELDS = ("open", "high", "low", "close")
ADJUSTED_COLUMNS = tuple(f"adj_{name}" for name in ADJUSTED_FIELDS)
FACTOR_POLICIES = ("source", "source_then_pre_close", "pre_close")
# The vendor factor covers a handful of ETFs, so `source` alone leaves a full
# sync with almost no adjusted prices. Prefer it where it exists and derive the
# rest from pre_close, which is the same quantity out of a different field.
DEFAULT_FACTOR_POLICY = "source_then_pre_close"

# The kernels answer in origin codes; only this table turns them back into the
# stored labels, so the mapping stays single-sourced and reversible.
ORIGIN_LABELS: dict[int, str] = {
    ORIGIN_NONE: "none",
    ORIGIN_SOURCE: "source",
    ORIGIN_PRE_CLOSE: "pre_close",
}
_STORED_LABELS = np.array([None, ORIGIN_LABELS[ORIGIN_SOURCE], ORIGIN_LABELS[ORIGIN_PRE_CLOSE]], dtype=object)


class AdjustmentPolicyError(ValueError):
    """请求了未登记的复权因子口径。"""


def _group_codes(codes: pd.Series) -> np.ndarray:
    """标的标识转稳定 int64 组号；调用方已按 (ts_code, date) 排序，组内连续。"""

    values, _ = pd.factorize(codes, sort=False)
    return np.ascontiguousarray(values, dtype=np.int64)


def _floats(values: Any) -> np.ndarray:
    return np.ascontiguousarray(pd.to_numeric(values, errors="coerce").to_numpy(dtype="float64"))


def source_factor(frame: pd.DataFrame, official: pd.DataFrame | None) -> np.ndarray:
    """数据源复权因子，按标的归一到首个交易日；覆盖不全的标的整体留空。"""

    if official is None or official.empty or "adj_factor" not in official.columns:
        return np.full(len(frame), np.nan)
    table = official[["ts_code", "date", "adj_factor"]].copy()
    table["date"] = pd.to_datetime(table["date"], errors="coerce")
    table = table.dropna(subset=["ts_code", "date"]).sort_values(["ts_code", "date"], kind="mergesort")
    table = table.drop_duplicates(["ts_code", "date"], keep="last")
    # Key alignment, not arithmetic: the join only decides which vendor row
    # belongs to which candle row before the values enter the kernel.
    merged = frame[["ts_code", "date"]].merge(table, on=["ts_code", "date"], how="left")
    return source_factor_kernel(_group_codes(frame["ts_code"]), _floats(merged["adj_factor"]))


def pre_close_factor(frame: pd.DataFrame) -> np.ndarray:
    """由前收盘价推导的后复权因子；某一天比值不可用时该标的其后全部留空。"""

    return pre_close_factor_kernel(
        _group_codes(frame["ts_code"]), _floats(frame["close"]), _floats(frame["pre_close"])
    )


def factor_divergence(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    """两条因子路径在同时可用的行上的相对偏离；用于暴露供应商口径差异。"""

    compared, worst = divergence_kernel(left, right)
    return {
        "compared_rows": int(compared),
        "max_relative_difference": None if np.isnan(worst) else float(worst),
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

    empty = np.full(len(frame), np.nan)
    from_source = source_factor(frame, official) if policy != "pre_close" else empty
    from_pre_close = pre_close_factor(frame) if policy != "source" else empty
    divergence = factor_divergence(from_source, from_pre_close)

    factor, origin = combine_factor_kernel(from_source, from_pre_close)
    frame["adj_factor"] = factor
    frame["adj_factor_source"] = pd.array(_STORED_LABELS[origin], dtype="string")
    for name in ADJUSTED_FIELDS:
        frame[f"adj_{name}"] = scale_kernel(_floats(frame[name]), factor)

    codes = _group_codes(frame["ts_code"])
    counts, events, rows_without_factor = factor_summary_kernel(codes, origin, factor)
    by_source = sorted(
        ((int(count), ORIGIN_LABELS[code]) for code, count in enumerate(counts) if count),
        key=lambda item: (-item[0], item[1]),
    )
    stats = {
        "policy": policy,
        "rows": int(len(frame)),
        "codes": int(codes.max() + 1) if codes.size else 0,
        "codes_by_factor_source": {label: count for count, label in by_source},
        "codes_with_adjustment_events": int(events),
        "rows_without_factor": int(rows_without_factor),
        "source_vs_pre_close": divergence,
    }
    return frame, stats


__all__ = [
    "ADJUSTED_COLUMNS",
    "ADJUSTED_FIELDS",
    "AdjustmentPolicyError",
    "DEFAULT_FACTOR_POLICY",
    "FACTOR_POLICIES",
    "ORIGIN_LABELS",
    "attach_adjusted_prices",
    "factor_divergence",
    "pre_close_factor",
    "source_factor",
]

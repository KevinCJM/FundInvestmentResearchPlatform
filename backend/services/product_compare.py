"""Server-side product comparison metrics and chart-series assembly.

Python is restricted to field/date preparation and response serialization.  All
financial arithmetic is delegated to the shared fixed-signature instrument
analytics kernel.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

try:
    from backend.compute_policy import validate_execution_audit
    from backend.instrument_analytics_numba import (
        instrument_analytics_numba_execution_audit,
        product_compare_analysis_kernel,
    )
except ModuleNotFoundError:  # pragma: no cover - backend/ direct execution
    from compute_policy import validate_execution_audit
    from instrument_analytics_numba import (
        instrument_analytics_numba_execution_audit,
        product_compare_analysis_kernel,
    )


RANGE_NAMES = ("performance", "risk", "efficiency")
METRIC_NAMES = (
    "cumulativeReturn",
    "annualizedReturn",
    "volatility",
    "maxDrawdown",
    "totalFee",
    "returnToFee",
    "sharpeRatio",
    "calmarRatio",
)


def _prepare_frame(points: Sequence[Mapping[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(points)
    if frame.empty or "date" not in frame.columns or "close" not in frame.columns:
        raise ValueError("产品行情数据缺少 date/close 字段")
    frame = frame.loc[:, ["date", "close"]].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    if frame["date"].isna().any():
        raise ValueError("产品行情包含无效日期")
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = (
        frame.sort_values("date", kind="mergesort")
        .drop_duplicates(subset=["date"], keep="last")
        .reset_index(drop=True)
    )
    if len(frame) < 2:
        raise ValueError("产品行情至少需要两个净值观察值")
    return frame


def _parse_boundary(value: object, field_name: str) -> pd.Timestamp | None:
    if value is None or str(value).strip() == "":
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"{field_name} 必须是有效日期")
    timestamp = pd.Timestamp(parsed)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_localize(None)
    return timestamp.normalize()


def _select_window(
    frame: pd.DataFrame,
    specification: Mapping[str, object],
    range_name: str,
) -> pd.DataFrame:
    start = _parse_boundary(specification.get("start_date"), f"{range_name}.start_date")
    end = _parse_boundary(specification.get("end_date"), f"{range_name}.end_date")
    if start is not None and end is not None and start > end:
        raise ValueError(f"{range_name} 的开始日期不能晚于结束日期")
    selected = frame
    if start is not None:
        selected = selected[selected["date"] >= start]
    if end is not None:
        selected = selected[selected["date"] <= end]
    if len(selected) < 2:
        raise ValueError(f"{range_name} 区间至少需要两个净值观察值")
    return selected.reset_index(drop=True)


def _optional_float(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _fee_value(value: object) -> float:
    return np.nan if value is None else float(value)


def _serialize_window(
    frame: pd.DataFrame,
    rolling_window_days: int,
    management_fee: float,
    custody_fee: float,
) -> dict[str, object]:
    dates = frame["date"].dt.strftime("%Y-%m-%d").tolist()
    nav_values = np.ascontiguousarray(frame["close"].to_numpy(dtype=np.float64))
    date_days = np.ascontiguousarray(
        frame["date"].to_numpy(dtype="datetime64[D]").astype(np.int64)
    )
    metrics, normalized, drawdown, rolling_volatility = product_compare_analysis_kernel(
        nav_values,
        date_days,
        int(rolling_window_days),
        float(management_fee),
        float(custody_fee),
    )
    return {
        "window": {
            "start_date": dates[0],
            "end_date": dates[-1],
            "observation_count": len(dates),
        },
        "metrics": {
            name: _optional_float(metrics[index])
            for index, name in enumerate(METRIC_NAMES)
        },
        "normalized_nav": [
            {"date": date, "value": float(normalized[index])}
            for index, date in enumerate(dates)
        ],
        "drawdown": [
            {"date": date, "value": float(drawdown[index])}
            for index, date in enumerate(dates)
        ],
        "rolling_volatility": [
            {"date": date, "value": _optional_float(rolling_volatility[index])}
            for index, date in enumerate(dates)
        ],
    }


def build_product_compare_response(
    *,
    product_id: str,
    points: Sequence[Mapping[str, object]],
    parameters: Mapping[str, object],
) -> dict[str, object]:
    frame = _prepare_frame(points)
    raw_ranges = parameters.get("ranges")
    if not isinstance(raw_ranges, Mapping):
        raise ValueError("产品比较请求缺少 ranges")
    rolling_window_days = int(parameters.get("rolling_window_days", 30))
    management_fee = _fee_value(parameters.get("management_fee"))
    custody_fee = _fee_value(parameters.get("custody_fee"))
    ranges: dict[str, object] = {}
    for range_name in RANGE_NAMES:
        specification = raw_ranges.get(range_name)
        if not isinstance(specification, Mapping):
            raise ValueError(f"产品比较请求缺少 {range_name} 区间")
        ranges[range_name] = _serialize_window(
            _select_window(frame, specification, range_name),
            rolling_window_days,
            management_fee,
            custody_fee,
        )
    return {
        "schema_version": 1,
        "product_id": product_id,
        "ranges": ranges,
        "execution": validate_execution_audit(
            instrument_analytics_numba_execution_audit()
        ),
    }


__all__ = ["build_product_compare_response"]

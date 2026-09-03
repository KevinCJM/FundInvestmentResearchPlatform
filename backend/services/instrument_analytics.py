"""Unified dashboard analytics for ETFs and off-exchange public funds.

Request-time analytics deliberately read only the small instrument information
files and the one-row-per-instrument metrics snapshot.  Full NAV and candle
history is consumed only by :func:`rebuild_analytics_snapshot`, one parquet row
group (and one instrument) at a time.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as parquet

try:
    from backend.market_data import resolve_tushare_data_dir
    from backend.series_quality import (
        PeriodWindowQuality,
        adjusted_nav_anomaly_dates,
        assess_period_window,
        load_sse_open_dates,
    )
except ModuleNotFoundError:  # pragma: no cover - backend/ direct execution
    from market_data import resolve_tushare_data_dir
    from series_quality import (
        PeriodWindowQuality,
        adjusted_nav_anomaly_dates,
        assess_period_window,
        load_sse_open_dates,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
SNAPSHOT_FILENAME = "instrument_metrics_snapshot.parquet"
SNAPSHOT_METADATA_FILENAME = "instrument_metrics_snapshot.meta.json"

InstrumentKind = Literal["all", "etf", "fund"]
SingleInstrumentKind = Literal["etf", "fund"]

ANALYTICS_FILTER_COLUMNS = ("fund_type", "invest_type", "status", "management", "market")
SNAPSHOT_COLUMNS = (
    "instrument_type",
    "ts_code",
    "as_of",
    "first_date",
    "latest_date",
    "observation_count",
    "observation_count_1m",
    "observation_count_3m",
    "observation_count_1y",
    "observation_count_3y",
    "coverage_ratio_1m",
    "coverage_ratio_3m",
    "coverage_ratio_1y",
    "coverage_ratio_3y",
    "quality_reason_1m",
    "quality_reason_3m",
    "quality_reason_1y",
    "quality_reason_3y",
    "adj_nav_anomaly_count",
    "latest_adj_nav",
    "current_size",
    "current_size_as_of",
    "current_share",
    "current_unit_nav",
    "return_1m",
    "return_3m",
    "return_1y",
    "return_3y",
    "annual_volatility_1y",
    "max_drawdown_3y",
    "sharpe_1y",
    "calmar_3y",
    "stale_days",
    "nav_source_fingerprint",
    "latest_close",
    "latest_candle_date",
    "latest_unit_nav",
    "premium_discount_latest",
    "premium_discount_date",
    "amount_avg_20d",
    "volume_avg_20d",
    "candle_source_fingerprint",
    "share_source_fingerprint",
)

METRIC_DEFINITIONS: dict[str, dict[str, Any]] = {
    "return_1m": {"label": "近1月收益率", "unit": "ratio", "source": "完整区间 adj_nav"},
    "return_3m": {"label": "近3月收益率", "unit": "ratio", "source": "完整区间 adj_nav"},
    "return_1y": {"label": "近1年收益率", "unit": "ratio", "source": "完整区间 adj_nav"},
    "return_3y": {"label": "近3年收益率", "unit": "ratio", "source": "完整区间 adj_nav"},
    "annual_volatility_1y": {
        "label": "近1年年化波动率",
        "unit": "ratio",
        "source": "完整1年区间 adj_nav daily return, annual_factor=252",
    },
    "max_drawdown_3y": {"label": "近3年最大回撤", "unit": "ratio", "source": "完整3年区间 adj_nav"},
    "sharpe_1y": {
        "label": "近1年夏普比率",
        "unit": "ratio",
        "source": "完整1年区间 adj_nav daily return, risk_free_rate=0",
    },
    "calmar_3y": {
        "label": "近3年卡玛比率",
        "unit": "ratio",
        "source": "完整3年区间 adj_nav annualized return / abs(max_drawdown)",
    },
    "issue_amount": {"label": "披露发行规模", "unit": "project_normalized_wan", "source": "fund_basic"},
    "m_fee": {"label": "管理费率", "unit": "percent", "source": "fund_basic"},
    "c_fee": {"label": "托管费率", "unit": "percent", "source": "fund_basic"},
    "amount_avg_20d": {"label": "近20期平均成交额", "unit": "source_amount", "source": "fund_daily"},
    "volume_avg_20d": {"label": "近20期平均成交量", "unit": "source_volume", "source": "fund_daily"},
    "premium_discount_latest": {
        "label": "最新同日溢折价率",
        "unit": "ratio",
        "source": "close / unit_nav - 1",
    },
    "current_size": {
        "label": "当前规模",
        "unit": "project_normalized_wan",
        "source": "ETF 总份额 × 同期单位净值",
    },
}

LEGACY_SNAPSHOT_METRIC_TYPES: dict[str, tuple[str, str]] = {
    "return_1m": ("return", "收益型指标"),
    "return_3m": ("return", "收益型指标"),
    "return_1y": ("return", "收益型指标"),
    "return_3y": ("return", "收益型指标"),
    "annual_volatility_1y": ("risk", "风险型指标"),
    "max_drawdown_3y": ("path", "路径与回撤指标"),
    "sharpe_1y": ("risk_adjusted", "收益风险性价比指标"),
    "calmar_3y": ("risk_adjusted", "收益风险性价比指标"),
    "amount_avg_20d": ("market_liquidity", "交易与流动性指标"),
    "volume_avg_20d": ("market_liquidity", "交易与流动性指标"),
    "premium_discount_latest": ("market_liquidity", "交易与流动性指标"),
    "current_size": ("scale", "规模指标"),
}
LEGACY_DERIVED_SNAPSHOT_METRICS = {
    "amount_avg_20d",
    "volume_avg_20d",
    "premium_discount_latest",
    "current_size",
}
LEGACY_COMPATIBILITY_SNAPSHOT_METRICS = LEGACY_DERIVED_SNAPSHOT_METRICS

SNAPSHOT_RANKING_METRICS = {
    "return_1m",
    "return_3m",
    "return_1y",
    "return_3y",
    "annual_volatility_1y",
    "max_drawdown_3y",
    "sharpe_1y",
    "calmar_3y",
    "amount_avg_20d",
    "volume_avg_20d",
    "premium_discount_latest",
}
PRODUCT_FILTER_METRICS = tuple(sorted(SNAPSHOT_RANKING_METRICS | {"current_size"}))
INFO_RANKING_METRICS = {"issue_amount", "m_fee", "c_fee"}
RANKING_METRICS = SNAPSHOT_RANKING_METRICS | INFO_RANKING_METRICS
ETF_ONLY_METRICS = {
    "amount_avg_20d",
    "volume_avg_20d",
    "premium_discount_latest",
    "current_size",
}
PRODUCT_SNAPSHOT_CONTEXT_FIELDS = (
    "as_of",
    "current_size_as_of",
    "current_share",
    "current_unit_nav",
)
RANKING_CONTEXT_METRICS = (
    "return_3m",
    "return_1y",
    "return_3y",
    "annual_volatility_1y",
    "max_drawdown_3y",
    "sharpe_1y",
)

METRIC_HORIZONS = {
    "return_1m": "1m",
    "return_3m": "3m",
    "return_1y": "1y",
    "annual_volatility_1y": "1y",
    "sharpe_1y": "1y",
    "return_3y": "3y",
    "max_drawdown_3y": "3y",
    "calmar_3y": "3y",
}


def _snapshot_indicator_metadata(data_dir: Path) -> dict[str, Any] | None:
    path = Path(data_dir) / SNAPSHOT_METADATA_FILENAME
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _legacy_snapshot_metric_definition(
    name: str,
    definition: Mapping[str, Any],
) -> dict[str, Any]:
    metric_type, metric_type_label = LEGACY_SNAPSHOT_METRIC_TYPES.get(
        name, ("other", "其他指标")
    )
    is_etf_only = name in ETF_ONLY_METRICS
    return {
        **dict(definition),
        "metric_source": (
            "system_derived"
            if name in LEGACY_DERIVED_SNAPSHOT_METRICS
            else "built_in"
        ),
        "metric_type": metric_type,
        "metric_type_label": metric_type_label,
        "applicable_product_kinds": ["etf"] if is_etf_only else ["etf", "fund"],
    }


def snapshot_metric_definitions(data_dir: Path) -> dict[str, dict[str, Any]]:
    """Return the public snapshot metric catalog for the active generation."""

    metadata = _snapshot_indicator_metadata(data_dir)
    if not metadata or not isinstance(metadata.get("items"), list):
        return {
            name: _legacy_snapshot_metric_definition(name, definition)
            for name, definition in METRIC_DEFINITIONS.items()
            if name in PRODUCT_FILTER_METRICS
        }
    # Indicator Center entries own configurable metrics. A small set of
    # compatibility metrics remains public because it is calculated directly
    # during snapshot construction and has no separate Indicator Center entry.
    definitions: dict[str, dict[str, Any]] = {
        name: _legacy_snapshot_metric_definition(name, METRIC_DEFINITIONS[name])
        for name in LEGACY_COMPATIBILITY_SNAPSHOT_METRICS
    }
    for item in metadata["items"]:
        if not isinstance(item, dict) or not item.get("field"):
            continue
        presentation = item.get("presentation") or {}
        period = str(item.get("period") or "")
        indicator_source = str(
            item.get("source") or presentation.get("source") or "custom"
        )
        indicator_type = str(
            presentation.get("indicator_type")
            or presentation.get("category")
            or "other"
        )
        definitions[str(item["field"])] = {
            "label": f"{item.get('name') or item.get('indicator_id')}（{period}）",
            "unit": (
                "ratio"
                if presentation.get("display_format") == "percent"
                else str(presentation.get("unit") or "number")
            ),
            "source": "指标中心预计算",
            "metric_source": indicator_source,
            "metric_type": indicator_type,
            "metric_type_label": str(
                presentation.get("category_label") or "其他指标"
            ),
            "indicator_id": item.get("indicator_id"),
            "indicator_revision": item.get("indicator_revision"),
            "period": period,
            "presentation": presentation,
            "applicable_product_kinds": list(
                presentation.get("applicable_product_kinds") or ["etf", "fund"]
            ),
        }
    return definitions


def snapshot_ranking_metrics(data_dir: Path) -> set[str]:
    return set(snapshot_metric_definitions(data_dir)) - {"current_size"}


def snapshot_etf_only_metrics(data_dir: Path) -> set[str]:
    return {
        field
        for field, definition in snapshot_metric_definitions(data_dir).items()
        if "fund" not in definition.get("applicable_product_kinds", ["etf", "fund"])
    }


def _requested_kinds(kind: InstrumentKind) -> tuple[SingleInstrumentKind, ...]:
    return ("etf", "fund") if kind == "all" else (kind,)


def _default_info_files(data_dir: Path) -> dict[str, Path]:
    return {
        "etf": data_dir / "etf_info_df.parquet",
        "fund": data_dir / "fund_info_df.parquet",
    }


def _path_key(path: Path) -> tuple[str, int, int]:
    stat = path.stat()
    return str(path), stat.st_mtime_ns, stat.st_size


@lru_cache(maxsize=12)
def _read_small_parquet_cached(path_text: str, mtime_ns: int, size: int) -> pd.DataFrame:
    del mtime_ns, size
    return pd.read_parquet(path_text, dtype_backend="pyarrow")


def _load_small_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    # Cached frames are treated as immutable.  Request handlers only allocate
    # when normalization or an actual filter is required; this keeps a cold
    # dashboard request below the memory budget.
    return _read_small_parquet_cached(*_path_key(path))


def _load_info(
    kind: SingleInstrumentKind,
    data_dir: Path,
    info_files: Optional[Mapping[str, Path]] = None,
) -> pd.DataFrame:
    path = Path((info_files or _default_info_files(data_dir))[kind])
    frame = _load_small_parquet(path)
    if frame.empty:
        return frame
    if any(not isinstance(column, str) for column in frame.columns):
        frame = frame.copy()
        frame.columns = [str(column) for column in frame.columns]
    if "ts_code" not in frame.columns:
        return pd.DataFrame()
    code_series = frame["ts_code"]
    codes = (
        code_series.str.strip()
        if pd.api.types.is_string_dtype(code_series.dtype)
        else code_series.astype(str).str.strip()
    )
    has_kind = "instrument_type" in frame.columns and frame["instrument_type"].eq(kind).all()
    needs_codes = not codes.equals(frame["ts_code"])
    if not has_kind or needs_codes:
        frame = frame.copy()
        if needs_codes:
            frame["ts_code"] = codes
        frame["instrument_type"] = kind
    if frame.duplicated(subset=["instrument_type", "ts_code"]).any():
        return frame.drop_duplicates(
            subset=["instrument_type", "ts_code"], keep="last"
        ).reset_index(drop=True)
    return frame


def _load_snapshot(data_dir: Path) -> pd.DataFrame:
    frame = _load_small_parquet(data_dir / SNAPSHOT_FILENAME)
    if frame.empty or not {"instrument_type", "ts_code"}.issubset(frame.columns):
        return pd.DataFrame(columns=SNAPSHOT_COLUMNS)
    instrument_type = frame["instrument_type"]
    ts_code = frame["ts_code"]
    if not pd.api.types.is_string_dtype(instrument_type.dtype):
        instrument_type = instrument_type.astype(str)
    if not pd.api.types.is_string_dtype(ts_code.dtype):
        ts_code = ts_code.astype(str)
    if not instrument_type.equals(frame["instrument_type"]) or not ts_code.equals(frame["ts_code"]):
        frame = frame.copy()
        frame["instrument_type"] = instrument_type
        frame["ts_code"] = ts_code
    if frame.duplicated(subset=["instrument_type", "ts_code"]).any():
        return frame.drop_duplicates(
            subset=["instrument_type", "ts_code"], keep="last"
        ).reset_index(drop=True)
    return frame


def _clean_filter_values(values: Optional[Iterable[str]]) -> list[str]:
    cleaned: list[str] = []
    for value in values or []:
        cleaned.extend(part.strip() for part in str(value).split(",") if part.strip())
    return list(dict.fromkeys(cleaned))


def _normalised_text(series: pd.Series) -> pd.Series:
    return (
        series.astype("string[pyarrow]")
        .fillna("未知")
        .str.strip()
        .replace({"": "未知", "nan": "未知"})
    )


def _apply_filters(frame: pd.DataFrame, filters: Optional[Mapping[str, Iterable[str]]]) -> pd.DataFrame:
    working = frame
    applied = False
    for column in ANALYTICS_FILTER_COLUMNS:
        values = _clean_filter_values((filters or {}).get(column))
        if not values or column not in working.columns:
            continue
        candidates = {value.casefold() for value in values}
        working = working[_normalised_text(working[column]).str.casefold().isin(candidates)]
        applied = True
    return working if applied else frame


def _distribution(frame: pd.DataFrame, column: str) -> list[dict[str, Any]]:
    if frame.empty or column not in frame.columns:
        return []
    counts = _normalised_text(frame[column]).value_counts()
    return [{"name": str(name), "value": int(count)} for name, count in counts.items()]


def _fee_distribution(frame: pd.DataFrame, column: str) -> list[dict[str, Any]]:
    """Return stable percentage buckets while retaining undisclosed rows."""

    if frame.empty or column not in frame.columns:
        return []
    values = pd.to_numeric(frame[column], errors="coerce")
    labels = pd.Series("未披露", index=frame.index, dtype=object)
    finite = values.notna() & np.isfinite(values)
    labels.loc[finite & values.le(0.25)] = "≤0.25%"
    labels.loc[finite & values.gt(0.25) & values.le(0.50)] = "0.25%–0.50%"
    labels.loc[finite & values.gt(0.50) & values.le(1.00)] = "0.50%–1.00%"
    labels.loc[finite & values.gt(1.00)] = ">1.00%"
    order = ("≤0.25%", "0.25%–0.50%", "0.50%–1.00%", ">1.00%", "未披露")
    counts = labels.value_counts()
    return [{"name": label, "value": int(counts[label])} for label in order if label in counts]


def _filter_options(frame: pd.DataFrame, column: str) -> list[dict[str, Any]]:
    return [
        {"value": item["name"], "label": item["name"], "count": item["value"]}
        for item in _distribution(frame, column)
    ]


def _combined_filter_options(
    frames: Iterable[pd.DataFrame],
    column: str,
) -> list[dict[str, Any]]:
    totals: dict[str, int] = {}
    for frame in frames:
        for item in _distribution(frame, column):
            totals[item["name"]] = totals.get(item["name"], 0) + int(item["value"])
    return [
        {"value": name, "label": name, "count": count}
        for name, count in sorted(totals.items(), key=lambda item: (-item[1], item[0]))
    ]


def _status_masks(frame: pd.DataFrame, kind: SingleInstrumentKind) -> dict[str, pd.Series]:
    if frame.empty:
        empty = pd.Series(dtype=bool, index=frame.index)
        return {name: empty for name in ("active", "issuing", "inactive", "unknown")}
    if "status_code" in frame.columns:
        codes = frame["status_code"].fillna("").astype(str).str.upper().str.strip()
        active = codes.eq("L")
        issuing = codes.eq("I")
        inactive = codes.eq("D")
        known = active | issuing | inactive
        return {"active": active, "issuing": issuing, "inactive": inactive, "unknown": ~known}

    status = _normalised_text(frame.get("status", pd.Series(index=frame.index, dtype=object)))
    active_label = "上市交易" if kind == "etf" else "存续"
    active = status.eq(active_label)
    issuing = status.str.contains("发行", na=False)
    inactive = status.str.contains("摘牌|终止|到期|清盘|退市", na=False)
    known = active | issuing | inactive
    return {"active": active, "issuing": issuing, "inactive": inactive, "unknown": ~known}


def _safe_sum(series: Optional[pd.Series]) -> Optional[float]:
    if series is None:
        return None
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return None if values.empty else float(values.sum())


def _safe_max_date(series: Optional[pd.Series]) -> Optional[str]:
    if series is None:
        return None
    values = pd.to_datetime(series, errors="coerce").dropna()
    return None if values.empty else values.max().strftime("%Y-%m-%d")


def _segment_summary(
    frame: pd.DataFrame,
    snapshot: pd.DataFrame,
    kind: SingleInstrumentKind,
) -> dict[str, Any]:
    masks = _status_masks(frame, kind)
    code_count = int(len(frame))
    codes = set(frame["ts_code"].astype(str)) if "ts_code" in frame.columns else set()
    snapshot_columns = [
        column
        for column in (
            "ts_code",
            "latest_date",
            "latest_candle_date",
            "amount_avg_20d",
        )
        if column in snapshot.columns
    ]
    snap = (
        snapshot.loc[
            snapshot["instrument_type"].eq(kind) & snapshot["ts_code"].astype(str).isin(codes),
            snapshot_columns,
        ]
        if not snapshot.empty
        else snapshot
    )
    covered = int(snap["ts_code"].nunique()) if not snap.empty else 0
    issue_values = pd.to_numeric(frame.get("issue_amount"), errors="coerce") if "issue_amount" in frame else pd.Series(dtype=float)
    index_covered = (
        int(_normalised_text(frame["index_code"]).ne("未知").sum())
        if kind == "etf" and "index_code" in frame
        else None
    )
    liquidity_covered = (
        int(pd.to_numeric(snap.get("amount_avg_20d"), errors="coerce").notna().sum())
        if kind == "etf" and not snap.empty and "amount_avg_20d" in snap
        else (0 if kind == "etf" else None)
    )
    purchase_redemption_covered = None
    if kind == "fund":
        purchase = pd.to_datetime(frame.get("purc_startdate"), errors="coerce")
        redemption = pd.to_datetime(frame.get("redm_startdate"), errors="coerce")
        if isinstance(purchase, pd.Series) and isinstance(redemption, pd.Series):
            purchase_redemption_covered = int((purchase.notna() & redemption.notna()).sum())
        else:
            purchase_redemption_covered = 0
    return {
        "share_code_count": code_count,
        "active_count": int(masks["active"].sum()),
        "issuing_count": int(masks["issuing"].sum()),
        "inactive_count": int(masks["inactive"].sum()),
        "unknown_status_count": int(masks["unknown"].sum()),
        "unique_managements": int(frame["management"].nunique(dropna=True)) if "management" in frame else 0,
        "nav_covered_count": covered,
        "nav_coverage_rate": (covered / code_count) if code_count else None,
        "latest_nav_date": _safe_max_date(snap.get("latest_date")) if not snap.empty else None,
        "latest_candle_date": (
            _safe_max_date(snap.get("latest_candle_date"))
            if kind == "etf" and not snap.empty
            else None
        ),
        "index_covered_count": index_covered,
        "index_coverage_rate": (index_covered / code_count) if index_covered is not None and code_count else None,
        "liquidity_covered_count": liquidity_covered,
        "liquidity_coverage_rate": (
            liquidity_covered / code_count
            if liquidity_covered is not None and code_count
            else None
        ),
        "purchase_redemption_covered_count": purchase_redemption_covered,
        "purchase_redemption_coverage_rate": (
            purchase_redemption_covered / code_count
            if purchase_redemption_covered is not None and code_count
            else None
        ),
        "issue_amount_total": _safe_sum(issue_values),
        "issue_amount_coverage_rate": (int(issue_values.notna().sum()) / code_count) if code_count else None,
    }


def _combined_summary(segments: Mapping[str, dict[str, Any]], frames: Iterable[pd.DataFrame]) -> dict[str, Any]:
    segment_summaries = [segment["summary"] for segment in segments.values()]
    code_count = sum(int(summary["share_code_count"]) for summary in segment_summaries)
    covered = sum(int(summary["nav_covered_count"]) for summary in segment_summaries)
    management_values: set[str] = set()
    for frame in frames:
        if "management" in frame.columns:
            management_values.update(_normalised_text(frame["management"]).loc[lambda value: value.ne("未知")])
    # Deliberately omit issue_amount_total: ETF and fund share-class issuance
    # amounts are not a safe cross-kind AUM or market-size aggregate.
    return {
        "share_code_count": code_count,
        "active_count": sum(int(summary["active_count"]) for summary in segment_summaries),
        "issuing_count": sum(int(summary["issuing_count"]) for summary in segment_summaries),
        "inactive_count": sum(int(summary["inactive_count"]) for summary in segment_summaries),
        "unknown_status_count": sum(int(summary["unknown_status_count"]) for summary in segment_summaries),
        "unique_managements": len(management_values),
        "nav_covered_count": covered,
        "nav_coverage_rate": (covered / code_count) if code_count else None,
    }


def _event_trend(frame: pd.DataFrame, kind: SingleInstrumentKind) -> dict[str, Any]:
    date_field = "list_date" if kind == "etf" else "found_date"
    label = "ETF上市趋势" if kind == "etf" else "场外公募基金成立趋势"
    if frame.empty or date_field not in frame.columns:
        return {"date_field": date_field, "label": label, "points": []}
    dates = pd.to_datetime(frame[date_field], errors="coerce")
    valid = dates.notna()
    if not valid.any():
        return {"date_field": date_field, "label": label, "points": []}
    years = dates.loc[valid].dt.year
    counts = years.value_counts().sort_index()
    issue = (
        pd.to_numeric(frame.loc[valid, "issue_amount"], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .groupby(years)
        .sum(min_count=1)
        if "issue_amount" in frame.columns
        else pd.Series(dtype=float)
    )
    points = [
        {
            "year": int(year),
            "count": int(count),
            "total_issue_amount": None if issue.empty or pd.isna(issue.get(year)) else float(issue.get(year)),
        }
        for year, count in counts.items()
    ]
    return {"date_field": date_field, "label": label, "points": points}


def _latest_products(
    frame: pd.DataFrame,
    kind: SingleInstrumentKind,
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    date_field = "list_date" if kind == "etf" else "found_date"
    if frame.empty or date_field not in frame.columns:
        return []
    dates = pd.to_datetime(frame[date_field], errors="coerce")
    valid = dates.notna()
    if not valid.any():
        return []
    ordering = pd.DataFrame(
        {
            date_field: dates.loc[valid],
            "ts_code": frame.loc[valid, "ts_code"].astype(str),
        },
        index=frame.index[valid],
    ).sort_values([date_field, "ts_code"], ascending=[False, True], kind="mergesort")
    fields = (
        "ts_code",
        "name",
        "management",
        "fund_type",
        "invest_type",
        "market",
        "index_code",
        "index_name",
        "list_date",
        "found_date",
        "due_date",
        "delist_date",
        "min_amount",
        "purc_startdate",
        "redm_startdate",
    )
    date_fields = {
        "list_date",
        "found_date",
        "due_date",
        "delist_date",
        "purc_startdate",
        "redm_startdate",
    }
    return [
        {
            field: (_json_date(row.get(field)) if field in date_fields else _json_scalar(row.get(field)))
            for field in fields
        }
        for index in ordering.head(limit).index
        for row in (frame.loc[index],)
    ]


def _kind_snapshot_state(
    data_dir: Path,
    snapshot: pd.DataFrame,
    kind: SingleInstrumentKind,
) -> dict[str, Any]:
    required_columns = [
        column
        for column in (
            "as_of",
            "latest_date",
            "nav_source_fingerprint",
            "candle_source_fingerprint",
            "share_source_fingerprint",
        )
        if column in snapshot.columns
    ]
    rows = (
        snapshot.loc[snapshot["instrument_type"].eq(kind), required_columns]
        if not snapshot.empty
        else snapshot
    )
    nav_path = data_dir / ("etf_daily_df.parquet" if kind == "etf" else "fund_nav_df.parquet")
    state = "missing"
    reason = "snapshot_rows_missing"
    if nav_path.exists() and not rows.empty:
        expected_nav = _source_fingerprint(nav_path)
        observed_nav = set(rows.get("nav_source_fingerprint", pd.Series(dtype=object)).dropna().astype(str))
        if observed_nav == {expected_nav}:
            state = "ready"
            reason = None
        else:
            state = "stale"
            reason = "nav_source_fingerprint_mismatch"
    elif not nav_path.exists():
        reason = "nav_source_missing"

    if kind == "etf" and state == "ready":
        candle_path = data_dir / "etf_daily_candle_df.parquet"
        if candle_path.exists():
            expected_candle = _source_fingerprint(candle_path)
            observed_candle = set(
                rows.get("candle_source_fingerprint", pd.Series(dtype=object)).dropna().astype(str)
            )
            if observed_candle and observed_candle != {expected_candle}:
                state = "stale"
                reason = "candle_source_fingerprint_mismatch"
    metric_availability: dict[str, str] = {}
    if kind == "etf":
        share_path = data_dir / "etf_share_size_df.parquet"
        if not share_path.exists():
            metric_availability["current_size"] = "missing"
        else:
            expected_share = _source_fingerprint(share_path)
            observed_share = set(
                rows.get("share_source_fingerprint", pd.Series(dtype=object))
                .dropna()
                .astype(str)
            )
            metric_availability["current_size"] = (
                "ready" if observed_share == {expected_share} else "stale"
            )
    return {
        "status": state,
        "reason": reason,
        "rows": int(len(rows)),
        "as_of": _safe_max_date(rows.get("as_of")) if not rows.empty else None,
        "latest_date": _safe_max_date(rows.get("latest_date")) if not rows.empty else None,
        "metric_availability": metric_availability,
    }


def load_product_filter_snapshot(
    kind: SingleInstrumentKind,
    data_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return the validated small metric snapshot used by product filters."""

    snapshot = _load_snapshot(data_dir)
    state = _kind_snapshot_state(data_dir, snapshot, kind)
    if state["status"] != "ready" or snapshot.empty:
        return snapshot.iloc[0:0], state
    metric_fields = sorted(snapshot_ranking_metrics(data_dir))
    metric_context_fields = [
        f"{field}__{suffix}"
        for field in metric_fields
        for suffix in (
            "status",
            "observation_count",
            "start_date",
            "end_date",
            "effective_as_of",
            "warning_code",
            "warning_message",
        )
    ]
    columns = [
        column
        for column in dict.fromkeys((
            "instrument_type",
            "ts_code",
            *metric_fields,
            *metric_context_fields,
            "current_size",
            *PRODUCT_SNAPSHOT_CONTEXT_FIELDS,
        ))
        if column in snapshot.columns
    ]
    selected = snapshot.loc[snapshot["instrument_type"].eq(kind), columns].copy()
    if state.get("metric_availability", {}).get("current_size") != "ready":
        for column in ("current_size", "current_size_as_of", "current_share", "current_unit_nav"):
            if column in selected.columns:
                selected[column] = pd.NA
    return selected, state


def _snapshot_metadata(
    data_dir: Path,
    snapshot: pd.DataFrame,
    kinds: Iterable[SingleInstrumentKind] = ("etf", "fund"),
) -> dict[str, Any]:
    path = data_dir / SNAPSHOT_FILENAME
    if not path.exists():
        return {
            "exists": False,
            "rows": 0,
            "updated_at": None,
            "as_of": None,
            "segments": {kind: _kind_snapshot_state(data_dir, snapshot, kind) for kind in kinds},
        }
    stat = path.stat()
    requested = tuple(kinds)
    relevant_mask = snapshot["instrument_type"].isin(requested) if not snapshot.empty else None
    relevant_rows = int(relevant_mask.sum()) if relevant_mask is not None else 0
    relevant_as_of = (
        snapshot.loc[relevant_mask, "as_of"]
        if relevant_mask is not None and "as_of" in snapshot.columns
        else None
    )
    return {
        "exists": True,
        "rows": relevant_rows,
        "updated_at": pd.Timestamp(stat.st_mtime, unit="s", tz="UTC").isoformat(),
        "as_of": _safe_max_date(relevant_as_of),
        "segments": {kind: _kind_snapshot_state(data_dir, snapshot, kind) for kind in requested},
    }


def _status_for_availability(availability: Iterable[str]) -> str:
    states = list(availability)
    if states and all(state == "ready" for state in states):
        return "complete"
    if any(state == "ready" for state in states):
        return "partial"
    return "unavailable"


def build_analytics_response(
    *,
    kind: InstrumentKind,
    data_dir: Path = DEFAULT_DATA_DIR,
    info_files: Optional[Mapping[str, Path]] = None,
    filters: Optional[Mapping[str, Iterable[str]]] = None,
) -> dict[str, Any]:
    requested = _requested_kinds(kind)
    snapshot = _load_snapshot(data_dir)
    snapshot_meta = _snapshot_metadata(data_dir, snapshot, requested)
    raw_frames = {item: _load_info(item, data_dir, info_files) for item in requested}
    filtered_frames = {item: _apply_filters(frame, filters) for item, frame in raw_frames.items()}
    segments: dict[str, dict[str, Any]] = {}
    warnings: list[dict[str, str]] = []
    quality_segments: dict[str, dict[str, Any]] = {}

    for item in requested:
        raw = raw_frames[item]
        frame = filtered_frames[item]
        availability = "ready" if not raw.empty else "missing"
        snapshot_state = snapshot_meta["segments"][item]
        usable_snapshot = snapshot if snapshot_state["status"] == "ready" else snapshot.iloc[0:0]
        summary = _segment_summary(frame, usable_snapshot, item)
        segment_warning: list[dict[str, str]] = []
        if availability == "missing":
            warning = {
                "code": f"{item.upper()}_INFO_MISSING",
                "message": f"未找到 {'ETF' if item == 'etf' else '场外公募基金'}基础信息。",
                "kind": item,
            }
            warnings.append(warning)
            segment_warning.append(warning)
        elif summary["nav_covered_count"] < summary["share_code_count"]:
            warning = {
                "code": f"{item.upper()}_NAV_PARTIAL",
                "message": "部分产品代码缺少分析快照，相关业绩指标将显示为空。",
                "kind": item,
            }
            warnings.append(warning)
            segment_warning.append(warning)
        if snapshot_state["status"] != "ready":
            warning = {
                "code": (
                    f"{item.upper()}_ANALYTICS_SNAPSHOT_STALE"
                    if snapshot_state["status"] == "stale"
                    else f"{item.upper()}_ANALYTICS_SNAPSHOT_MISSING"
                ),
                "message": "分析快照与当前源数据不一致，业绩指标已局部降级。"
                if snapshot_state["status"] == "stale"
                else "该品类分析快照尚未生成，业绩指标暂不可用。",
                "kind": item,
            }
            warnings.append(warning)
            segment_warning.append(warning)
        segments[item] = {
            "availability": availability,
            "snapshot_availability": snapshot_state["status"],
            "summary": summary,
            "distributions": {
                column: _distribution(frame, column)
                for column in (
                    "fund_type",
                    "invest_type",
                    "market",
                    "status",
                    "management",
                    "index_name",
                )
            },
            "event_trend": _event_trend(frame, item),
            "latest_products": _latest_products(frame, item),
        }
        segments[item]["distributions"].update(
            {column: _fee_distribution(frame, column) for column in ("m_fee", "c_fee")}
        )
        info_path = Path((info_files or _default_info_files(data_dir))[item])
        quality_segments[item] = {
            "info_file_exists": info_path.exists(),
            "info_rows": int(len(raw)),
            "snapshot_rows": int(summary["nav_covered_count"]),
            "nav_coverage_rate": summary["nav_coverage_rate"],
            "warnings": segment_warning,
        }

    if not snapshot_meta["exists"]:
        warnings.append(
            {
                "code": "ANALYTICS_SNAPSHOT_MISSING",
                "message": "分析快照尚未生成，结构统计可用，业绩与风险排行暂不可用。",
            }
        )
    summary: dict[str, Any] = {item: segments[item]["summary"] for item in requested}
    if kind == "all":
        summary = {
            "all": _combined_summary(segments, filtered_frames.values()),
            "etf": segments["etf"]["summary"],
            "fund": segments["fund"]["summary"],
        }
    availability = {
        f"{item}_info": segment["availability"]
        for item, segment in segments.items()
    }
    snapshot_states = [snapshot_meta["segments"][item]["status"] for item in requested]
    availability["analysis_snapshot"] = (
        "ready"
        if snapshot_states and all(state == "ready" for state in snapshot_states)
        else ("stale" if any(state == "stale" for state in snapshot_states) else "missing")
    )
    base_status = _status_for_availability(segment["availability"] for segment in segments.values())
    response_status = (
        "partial"
        if base_status == "complete" and availability["analysis_snapshot"] != "ready"
        else base_status
    )
    return {
        "schema_version": 1,
        "kind": kind,
        "status": response_status,
        "as_of": snapshot_meta["as_of"],
        "availability": availability,
        "summary": summary,
        "segments": segments,
        "available_filters": {
            column: _combined_filter_options(raw_frames.values(), column)
            for column in ANALYTICS_FILTER_COLUMNS
        },
        "data_quality": {
            "snapshot": snapshot_meta,
            "segments": quality_segments,
            "warnings": warnings,
        },
        "metric_definitions": {
            **{name: value for name, value in METRIC_DEFINITIONS.items() if name in INFO_RANKING_METRICS},
            **snapshot_metric_definitions(data_dir),
        },
        "units": {
            name: definition["unit"]
            for name, definition in {
                **{name: value for name, value in METRIC_DEFINITIONS.items() if name in INFO_RANKING_METRICS},
                **snapshot_metric_definitions(data_dir),
            }.items()
        },
    }


def build_trend_response(
    *,
    kind: InstrumentKind,
    dimension: Literal["all", "fund_type", "invest_type", "management"] = "all",
    values: Optional[Iterable[str]] = None,
    data_dir: Path = DEFAULT_DATA_DIR,
    info_files: Optional[Mapping[str, Path]] = None,
    filters: Optional[Mapping[str, Iterable[str]]] = None,
) -> dict[str, Any]:
    requested = _requested_kinds(kind)
    raw_frames = {item: _load_info(item, data_dir, info_files) for item in requested}
    applied_values = _clean_filter_values(values)
    series: dict[str, Any] = {}
    availability: list[str] = []
    warnings: list[dict[str, str]] = []
    for item, raw in raw_frames.items():
        availability.append("ready" if not raw.empty else "missing")
        working = _apply_filters(raw, filters)
        if dimension != "all":
            if not applied_values or dimension not in working.columns:
                working = working.iloc[0:0]
            else:
                candidates = {value.casefold() for value in applied_values}
                working = working[_normalised_text(working[dimension]).str.casefold().isin(candidates)]
        series[item] = _event_trend(working, item)
        if raw.empty:
            warnings.append(
                {
                    "code": f"{item.upper()}_INFO_MISSING",
                    "message": f"未找到 {'ETF' if item == 'etf' else '场外公募基金'}基础信息。",
                    "kind": item,
                }
            )
    return {
        "schema_version": 1,
        "kind": kind,
        "status": _status_for_availability(availability),
        "date_semantic": {
            item: ("list_date" if item == "etf" else "found_date") for item in requested
        },
        "dimension": dimension,
        "values": applied_values,
        "series": series,
        "available_values": (
            []
            if dimension == "all"
            else _combined_filter_options(raw_frames.values(), dimension)
        ),
        "data_quality": {"warnings": warnings},
    }


def build_rankings_response(
    *,
    kind: SingleInstrumentKind,
    metric: str,
    sort_dir: Literal["asc", "desc"] = "desc",
    page: int = 1,
    page_size: int = 10,
    active_only: bool = True,
    data_dir: Path = DEFAULT_DATA_DIR,
    info_files: Optional[Mapping[str, Path]] = None,
    filters: Optional[Mapping[str, Iterable[str]]] = None,
) -> dict[str, Any]:
    dynamic_snapshot_metrics = snapshot_ranking_metrics(data_dir)
    metric_definitions = {
        **{name: value for name, value in METRIC_DEFINITIONS.items() if name in INFO_RANKING_METRICS},
        **snapshot_metric_definitions(data_dir),
    }
    if metric not in dynamic_snapshot_metrics | INFO_RANKING_METRICS:
        raise ValueError(f"不支持的排行指标: {metric}")
    if kind == "fund" and metric in snapshot_etf_only_metrics(data_dir):
        raise ValueError(f"{metric} 仅适用于 ETF")
    if not active_only:
        raise ValueError("排行仅允许纳入 status_code=L 的上市/存续份额。")
    metric_horizon = METRIC_HORIZONS.get(metric)
    dynamic_observation_column = f"{metric}__observation_count"
    metric_observation_column = (
        dynamic_observation_column
        if metric in dynamic_snapshot_metrics and dynamic_observation_column in _load_snapshot(data_dir).columns
        else f"observation_count_{metric_horizon}" if metric_horizon else "observation_count"
    )
    info = _apply_filters(_load_info(kind, data_dir, info_files), filters)
    snapshot = _load_snapshot(data_dir)
    snapshot = snapshot[snapshot["instrument_type"].eq(kind)].copy() if not snapshot.empty else snapshot
    snapshot_state = _kind_snapshot_state(data_dir, snapshot, kind)
    if snapshot_state["status"] != "ready":
        snapshot = snapshot.iloc[0:0]
    if not snapshot.empty and "latest_date" in snapshot.columns:
        parsed_latest = pd.to_datetime(snapshot["latest_date"], errors="coerce")
        segment_latest = parsed_latest.max()
        if pd.notna(segment_latest):
            snapshot["_stale_days"] = (segment_latest - parsed_latest).dt.days
            snapshot = snapshot[snapshot["_stale_days"].fillna(8).le(7)]
    warnings: list[dict[str, str]] = []
    if info.empty:
        warnings.append({"code": f"{kind.upper()}_INFO_MISSING", "message": "产品基础信息不可用。", "kind": kind})
    if snapshot.empty:
        warnings.append(
            {
                "code": "ANALYTICS_SNAPSHOT_STALE"
                if snapshot_state["status"] == "stale"
                else "ANALYTICS_SNAPSHOT_MISSING",
                "message": "分析快照尚未生成或与源数据不一致，排行暂不可用。",
            }
        )
    status = "complete"
    if info.empty or snapshot.empty:
        status = "unavailable"
        working = pd.DataFrame()
    else:
        info = info[_status_masks(info, kind)["active"]]
        info_columns = [
            column
            for column in (
                "instrument_type",
                "ts_code",
                "name",
                "management",
                "fund_type",
                "invest_type",
                "status",
                "issue_amount",
                "m_fee",
                "c_fee",
            )
            if column in info.columns
        ]
        working = info[info_columns].copy()
        eligibility_columns = [
            column
            for column in ("instrument_type", "ts_code", "latest_date", "observation_count", "_stale_days")
            if column in snapshot.columns
        ]
        if metric in dynamic_snapshot_metrics:
            snapshot_columns = [
                column
                for column in dict.fromkeys((
                    "instrument_type",
                    "ts_code",
                    "latest_date",
                    "observation_count",
                    metric_observation_column,
                    "_stale_days",
                    metric,
                    *RANKING_CONTEXT_METRICS,
                ))
                if column in snapshot.columns
            ]
            working = working.merge(snapshot[snapshot_columns], on=["instrument_type", "ts_code"], how="inner")
        else:
            eligibility_columns = list(
                dict.fromkeys([*eligibility_columns, *[
                    name for name in RANKING_CONTEXT_METRICS if name in snapshot.columns
                ]])
            )
            working = working.merge(
                snapshot[eligibility_columns], on=["instrument_type", "ts_code"], how="inner"
            )
        if metric_observation_column in working.columns:
            working["_metric_observation_count"] = working[metric_observation_column]
        else:
            working["_metric_observation_count"] = working.get("observation_count")
        working["value"] = pd.to_numeric(working[metric], errors="coerce")
        working = working.replace([np.inf, -np.inf], np.nan).dropna(subset=["value"])
        working = working.sort_values(
            ["value", "ts_code"],
            ascending=[sort_dir == "asc", True],
            kind="mergesort",
        )
    total = int(len(working))
    start = (page - 1) * page_size
    page_frame = working.iloc[start : start + page_size]
    items: list[dict[str, Any]] = []
    for _, row in page_frame.iterrows():
        item = {
            "instrument_type": kind,
            "ts_code": str(row.get("ts_code")),
            "name": _json_scalar(row.get("name")),
            "management": _json_scalar(row.get("management")),
            "fund_type": _json_scalar(row.get("fund_type")),
            "invest_type": _json_scalar(row.get("invest_type")),
            "status": _json_scalar(row.get("status")),
            "latest_date": _json_date(row.get("latest_date")),
            "observation_count": _json_int(row.get("_metric_observation_count")),
            "value": float(row["value"]),
            "metrics": {
                name: _json_scalar(row.get(name)) for name in RANKING_CONTEXT_METRICS
            },
        }
        items.append(item)
    snapshot_meta = _snapshot_metadata(data_dir, snapshot, (kind,))
    return {
        "schema_version": 1,
        "kind": kind,
        "status": status,
        "metric": metric,
        "metric_definition": metric_definitions[metric],
        "sort_dir": sort_dir,
        "page": page,
        "page_size": page_size,
        "total": total,
        "as_of": snapshot_meta["as_of"],
        "items": items,
        "data_quality": {"warnings": warnings},
    }


def _json_scalar(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    text = str(value).strip()
    return text or None


def _json_date(value: Any) -> Optional[str]:
    parsed = pd.to_datetime(value, errors="coerce")
    return None if pd.isna(parsed) else parsed.strftime("%Y-%m-%d")


def _json_int(value: Any) -> Optional[int]:
    try:
        return None if value is None or pd.isna(value) else int(value)
    except (TypeError, ValueError):
        return None


def _source_fingerprint(path: Path) -> str:
    stat = path.stat()
    raw = f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def _parquet_columns(path: Path) -> set[str]:
    return set(parquet.ParquetFile(path).schema.names)


def _iter_instrument_frames(path: Path, requested_columns: Iterable[str]):
    """Yield contiguous per-code frames with bounded parquet batch memory.

    Tushare acquisition writes time-series files sorted by ``ts_code,date``.
    Failing closed on a non-contiguous code protects metric correctness without
    falling back to a full-file pandas read.  Fixed-size record batches also
    bound memory for legacy parquet files whose row groups contain one million
    or more rows; newly consolidated files may still use one row group per code.
    """

    source = parquet.ParquetFile(path)
    available = set(source.schema.names)
    columns = [column for column in requested_columns if column in available]
    if not {"ts_code", "date"}.issubset(columns):
        raise ValueError(f"{path.name} 缺少 ts_code/date 列")
    pending_code: Optional[str] = None
    pending_parts: list[pd.DataFrame] = []
    emitted: set[str] = set()
    for batch in source.iter_batches(batch_size=16_384, columns=columns, use_threads=False):
        frame = batch.to_pandas()
        if frame.empty:
            continue
        frame["ts_code"] = frame["ts_code"].astype(str).str.strip()
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame = frame.dropna(subset=["ts_code", "date"]).sort_values(["ts_code", "date"], kind="mergesort")
        for code, group in frame.groupby("ts_code", sort=False):
            code = str(code)
            if pending_code is None:
                pending_code = code
                pending_parts = [group]
                continue
            if code == pending_code:
                pending_parts.append(group)
                continue
            if code in emitted:
                raise ValueError(f"{path.name} 未按 ts_code 连续排列，无法流式构建快照: {code}")
            combined = pd.concat(pending_parts, ignore_index=True)
            emitted.add(pending_code)
            yield pending_code, combined
            pending_code = code
            pending_parts = [group]
    if pending_code is not None:
        if pending_code in emitted:
            raise ValueError(f"{path.name} 未按 ts_code 连续排列，无法流式构建快照: {pending_code}")
        yield pending_code, pd.concat(pending_parts, ignore_index=True)


def _complete_metric_window(
    frame: pd.DataFrame,
    *,
    months: int,
    open_dates: Optional[pd.DatetimeIndex],
    anomaly_dates: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, PeriodWindowQuality]:
    effective = pd.Timestamp(frame["date"].max()).normalize()
    quality = assess_period_window(
        frame["date"],
        target_date=effective - pd.DateOffset(months=months),
        effective_date=effective,
        open_dates=open_dates,
        anomaly_dates=anomaly_dates,
    )
    if quality.anchor_date is None:
        return frame.iloc[0:0], quality
    selected = frame[
        frame["date"].between(quality.anchor_date, quality.effective_date)
    ].copy()
    return selected, quality


def _window_return(frame: pd.DataFrame, quality: PeriodWindowQuality) -> Optional[float]:
    if not quality.complete or len(frame) < 2:
        return None
    first_value = float(frame.iloc[0]["adj_nav"])
    latest_value = float(frame.iloc[-1]["adj_nav"])
    result = latest_value / first_value - 1 if first_value > 0 else np.nan
    return float(result) if np.isfinite(result) else None


def _compute_nav_record(
    kind: SingleInstrumentKind,
    code: str,
    frame: pd.DataFrame,
    fingerprint: str,
    open_dates: Optional[pd.DatetimeIndex] = None,
    include_legacy_metrics: bool = True,
) -> tuple[dict[str, Any], dict[str, float]]:
    working = frame.copy()
    working["adj_nav"] = pd.to_numeric(working.get("adj_nav"), errors="coerce")
    for reference_column in ("unit_nav", "accum_nav"):
        if reference_column in working.columns:
            working[reference_column] = pd.to_numeric(
                working[reference_column], errors="coerce"
            )
    working = working.replace([np.inf, -np.inf], np.nan).dropna(subset=["date", "adj_nav"])
    working = working[working["adj_nav"] > 0].sort_values("date").drop_duplicates(subset=["date"], keep="last")
    if working.empty:
        return {}, {}
    latest = working.iloc[-1]
    latest_date = pd.Timestamp(latest["date"])
    latest_value = float(latest["adj_nav"])
    anomaly_dates = adjusted_nav_anomaly_dates(working)
    horizon_months = {"1m": 1, "3m": 3, "1y": 12, "3y": 36}
    windows: dict[str, pd.DataFrame] = {}
    qualities: dict[str, PeriodWindowQuality] = {}
    for label, months in horizon_months.items():
        windows[label], qualities[label] = _complete_metric_window(
            working,
            months=months,
            open_dates=open_dates,
            anomaly_dates=anomaly_dates,
        )

    annual_volatility: Optional[float] = None
    sharpe: Optional[float] = None
    max_drawdown: Optional[float] = None
    calmar: Optional[float] = None
    if include_legacy_metrics:
        one_year = windows["1y"] if qualities["1y"].complete else working.iloc[0:0]
        three_year = windows["3y"] if qualities["3y"].complete else working.iloc[0:0]
        returns = one_year["adj_nav"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).dropna()
        if len(returns) >= 60:
            daily_std = float(returns.std(ddof=1))
            if np.isfinite(daily_std):
                annual_volatility = daily_std * np.sqrt(252.0)
                if daily_std > 1e-12:
                    sharpe = float(returns.mean() / daily_std * np.sqrt(252.0))
        navs = three_year["adj_nav"].to_numpy(dtype=float)
        if len(navs) - 1 >= 180:
            peaks = np.maximum.accumulate(navs)
            drawdowns = navs / peaks - 1.0
            max_drawdown = float(drawdowns.min())
        if len(three_year) >= 2 and max_drawdown is not None and abs(max_drawdown) > 1e-12:
            elapsed_days = max((three_year.iloc[-1]["date"] - three_year.iloc[0]["date"]).days, 1)
            annualized = (float(three_year.iloc[-1]["adj_nav"]) / float(three_year.iloc[0]["adj_nav"])) ** (
                365.25 / elapsed_days
            ) - 1.0
            if np.isfinite(annualized):
                calmar = float(annualized / abs(max_drawdown))
    record: dict[str, Any] = {
        "instrument_type": kind,
        "ts_code": code,
        "as_of": latest_date,
        "first_date": pd.Timestamp(working.iloc[0]["date"]),
        "latest_date": latest_date,
        "observation_count": int(len(working)),
        **{
            f"observation_count_{label}": int(quality.observation_count)
            for label, quality in qualities.items()
        },
        **{
            f"coverage_ratio_{label}": quality.coverage_ratio
            for label, quality in qualities.items()
        },
        **{
            f"quality_reason_{label}": quality.reason
            for label, quality in qualities.items()
        },
        "adj_nav_anomaly_count": int(len(anomaly_dates)),
        "latest_adj_nav": latest_value,
        "return_1m": _window_return(windows["1m"], qualities["1m"]) if include_legacy_metrics else None,
        "return_3m": _window_return(windows["3m"], qualities["3m"]) if include_legacy_metrics else None,
        "return_1y": _window_return(windows["1y"], qualities["1y"]) if include_legacy_metrics else None,
        "return_3y": _window_return(windows["3y"], qualities["3y"]) if include_legacy_metrics else None,
        "annual_volatility_1y": annual_volatility,
        "max_drawdown_3y": max_drawdown,
        "sharpe_1y": sharpe,
        "calmar_3y": calmar,
        "stale_days": 0,
        "nav_source_fingerprint": fingerprint,
    }
    unit_nav_tail: dict[str, float] = {}
    if kind == "etf" and "unit_nav" in working.columns:
        unit_values = pd.to_numeric(working["unit_nav"], errors="coerce")
        unit_frame = working.assign(_unit_nav=unit_values).dropna(subset=["_unit_nav"]).tail(31)
        unit_nav_tail = {
            pd.Timestamp(date).strftime("%Y-%m-%d"): float(unit_nav)
            for date, unit_nav in zip(unit_frame["date"], unit_frame["_unit_nav"])
            if float(unit_nav) > 0
        }
    return record, unit_nav_tail


def _compute_candle_record(frame: pd.DataFrame, unit_nav_tail: Mapping[str, float], fingerprint: str) -> dict[str, Any]:
    working = frame.copy()
    for column in ("close", "amount", "vol"):
        if column in working.columns:
            working[column] = pd.to_numeric(working[column], errors="coerce")
    working = working.replace([np.inf, -np.inf], np.nan).dropna(subset=["date", "close"])
    working = working[working["close"] > 0].sort_values("date").drop_duplicates(subset=["date"], keep="last")
    if working.empty:
        return {}
    latest = working.iloc[-1]
    tail = working.tail(20)
    common = working.assign(_date_key=working["date"].dt.strftime("%Y-%m-%d"))
    common = common[common["_date_key"].isin(unit_nav_tail)]
    premium: Optional[float] = None
    premium_date: Optional[pd.Timestamp] = None
    latest_unit_nav: Optional[float] = None
    if not common.empty:
        common_row = common.iloc[-1]
        latest_unit_nav = float(unit_nav_tail[str(common_row["_date_key"])])
        premium_value = float(common_row["close"]) / latest_unit_nav - 1.0
        premium = float(premium_value) if np.isfinite(premium_value) else None
        premium_date = pd.Timestamp(common_row["date"])
    return {
        "latest_close": float(latest["close"]),
        "latest_candle_date": pd.Timestamp(latest["date"]),
        "latest_unit_nav": latest_unit_nav,
        "premium_discount_latest": premium,
        "premium_discount_date": premium_date,
        "amount_avg_20d": _finite_mean(tail.get("amount"), required_count=20),
        "volume_avg_20d": _finite_mean(tail.get("vol"), required_count=20),
        "candle_source_fingerprint": fingerprint,
    }


def _compute_share_record(frame: pd.DataFrame, fingerprint: str) -> dict[str, Any]:
    """Calculate the latest ETF size in 万元 from 万份 × 元/份."""

    working = frame.copy()
    for column in ("total_share", "nav"):
        working[column] = pd.to_numeric(working.get(column), errors="coerce")
    working = working.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["date", "total_share", "nav"]
    )
    working = working[
        working["total_share"].gt(0) & working["nav"].gt(0)
    ].sort_values("date").drop_duplicates(subset=["date"], keep="last")
    if working.empty:
        return {}
    latest = working.iloc[-1]
    current_share = float(latest["total_share"])
    current_unit_nav = float(latest["nav"])
    current_size = current_share * current_unit_nav
    if not np.isfinite(current_size):
        return {}
    return {
        "current_size": float(current_size),
        "current_size_as_of": pd.Timestamp(latest["date"]),
        "current_share": current_share,
        "current_unit_nav": current_unit_nav,
        "share_source_fingerprint": fingerprint,
    }


def _finite_mean(series: Optional[pd.Series], *, required_count: int | None = None) -> Optional[float]:
    if series is None:
        return None
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if required_count is not None and len(values) < required_count:
        return None
    return None if values.empty else float(values.mean())


def _atomic_write_snapshot(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        frame.to_parquet(temporary_path, index=False)
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _atomic_write_json(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _configured_snapshot_values(
    output: pd.DataFrame,
    *,
    market_data_dir: Path,
    workspace_data_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Evaluate configured Indicator Center definitions into snapshot columns."""

    try:
        from backend.custom_indicators.service import CustomIndicatorService
        from backend.custom_indicators.series_provider import market_data_generation
    except ModuleNotFoundError:  # pragma: no cover - backend/ direct execution
        from custom_indicators.service import CustomIndicatorService
        from custom_indicators.series_provider import market_data_generation

    service = CustomIndicatorService(
        workspace_data_dir=workspace_data_dir,
        market_data_dir=market_data_dir,
    )
    config = service.get_snapshot_config()
    configured = [item for item in config.get("items", []) if item.get("status") == "ready"]
    service.warm_snapshot_numba_plans(configured)
    result = output.copy()
    for item in configured:
        field = str(item["field"])
        result[field] = np.nan
        result[f"{field}__status"] = "unavailable"
        result[f"{field}__observation_count"] = 0
        for suffix in ("start_date", "end_date", "effective_as_of", "warning_code", "warning_message"):
            result[f"{field}__{suffix}"] = None

    targets = [
        {"kind": str(row.instrument_type), "product_id": str(row.ts_code)}
        for row in result[["instrument_type", "ts_code"]].itertuples(index=False)
    ]
    row_indexes = {
        (str(row.instrument_type), str(row.ts_code)): row.Index
        for row in result[["instrument_type", "ts_code"]].itertuples()
    }
    status_counts = {"ok": 0, "warning": 0, "unavailable": 0, "error": 0}
    failures: list[dict[str, str]] = []
    groups: dict[str, list[dict[str, Any]]] = {}
    for item in configured:
        groups.setdefault(str(item["period"]), []).append(item)

    for period, period_items in groups.items():
        for metric_start in range(0, len(period_items), 10):
            metric_batch = period_items[metric_start : metric_start + 10]
            indicator_ids = [str(item["indicator_id"]) for item in metric_batch]
            versions = {
                str(item["indicator_id"]): int(item["indicator_revision"])
                for item in metric_batch
            }
            fields = {str(item["indicator_id"]): str(item["field"]) for item in metric_batch}
            for target_start in range(0, len(targets), 50):
                target_batch = targets[target_start : target_start + 50]
                response = service.evaluate(
                    indicator_ids=indicator_ids,
                    indicator_versions=versions,
                    inline_definition=None,
                    targets=target_batch,
                    period=period,
                    include_series=False,
                    prefer_snapshot=False,
                )
                for item in response.get("results", []):
                    status = str(item.get("status") or "error")
                    status_counts[status if status in status_counts else "error"] += 1
                    value = item.get("value")
                    target = item.get("target") or {}
                    field = fields.get(str(item.get("indicator_id")))
                    if not field:
                        continue
                    row_index = row_indexes.get(
                        (str(target.get("kind")), str(target.get("product_id")))
                    )
                    if row_index is None:
                        continue
                    result.at[row_index, f"{field}__status"] = status
                    window = item.get("window") or {}
                    result.at[row_index, f"{field}__observation_count"] = int(
                        window.get("observation_count") or 0
                    )
                    for suffix in ("start_date", "end_date", "effective_as_of"):
                        result.at[row_index, f"{field}__{suffix}"] = window.get(suffix)
                    warnings = item.get("warnings") or []
                    if warnings:
                        result.at[row_index, f"{field}__warning_code"] = warnings[0].get("code")
                        result.at[row_index, f"{field}__warning_message"] = warnings[0].get("message")
                    if value is not None and np.isfinite(float(value)):
                        result.at[row_index, field] = float(value)

    metadata_items = [
        {
            "field": item["field"],
            "indicator_id": item["indicator_id"],
            "indicator_revision": item["indicator_revision"],
            "period": item["period"],
            "name": item.get("name"),
            "source": item.get("source"),
            "presentation": item.get("presentation"),
        }
        for item in configured
    ]
    missing_definitions = [
        {
            "indicator_id": str(item.get("indicator_id")),
            "message": str(item.get("status_message") or "指标版本不存在。"),
        }
        for item in config.get("items", [])
        if item.get("status") != "ready"
    ]
    failures.extend(missing_definitions)
    return result, {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_generation": market_data_generation(market_data_dir),
        "config_revision": config.get("revision"),
        "configured_count": len(configured),
        "items": metadata_items,
        "status_counts": status_counts,
        "failures": failures,
        "legacy_columns": [
            "amount_avg_20d",
            "volume_avg_20d",
            "premium_discount_latest",
            "current_size",
        ],
        "legacy_note": "这些列是兼容属性；新增快照指标只能从指标中心配置。",
    }


def rebuild_analytics_snapshot(
    data_dir: Path | None = None,
    *,
    workspace_data_dir: Path | None = None,
) -> dict[str, Any]:
    """Build the local dashboard snapshot without any network access.

    The function raises on malformed input and atomically preserves any prior
    snapshot on failure.  It is safe for refresh orchestration to run as a
    post-processing step and report its exception as a warning.
    """

    target_dir = (
        Path(data_dir).expanduser().resolve()
        if data_dir is not None
        else resolve_tushare_data_dir(DEFAULT_DATA_DIR)
    )
    nav_sources: tuple[tuple[SingleInstrumentKind, Path], ...] = (
        ("etf", target_dir / "etf_daily_df.parquet"),
        ("fund", target_dir / "fund_nav_df.parquet"),
    )
    existing_nav_sources = [(kind, path) for kind, path in nav_sources if path.exists()]
    if not existing_nav_sources:
        raise FileNotFoundError("未找到 ETF 或场外公募基金净值文件，无法构建分析快照。")

    records: dict[tuple[str, str], dict[str, Any]] = {}
    etf_unit_nav_tails: dict[str, dict[str, float]] = {}
    source_files: list[str] = []
    open_dates = load_sse_open_dates(target_dir / "trade_day_df.parquet")
    for kind, path in existing_nav_sources:
        source_files.append(path.name)
        fingerprint = _source_fingerprint(path)
        requested = ("ts_code", "date", "adj_nav", "unit_nav", "accum_nav")
        for code, frame in _iter_instrument_frames(path, requested):
            record, unit_nav_tail = _compute_nav_record(
                kind,
                code,
                frame,
                fingerprint,
                open_dates,
                include_legacy_metrics=workspace_data_dir is None,
            )
            if not record:
                continue
            records[(kind, code)] = record
            if kind == "etf" and unit_nav_tail:
                etf_unit_nav_tails[code] = unit_nav_tail

    candle_path = target_dir / "etf_daily_candle_df.parquet"
    if candle_path.exists():
        source_files.append(candle_path.name)
        fingerprint = _source_fingerprint(candle_path)
        requested = ("ts_code", "date", "close", "amount", "vol")
        for code, frame in _iter_instrument_frames(candle_path, requested):
            key = ("etf", code)
            if key not in records:
                continue
            records[key].update(_compute_candle_record(frame, etf_unit_nav_tails.get(code, {}), fingerprint))

    share_path = target_dir / "etf_share_size_df.parquet"
    if share_path.exists():
        source_files.append(share_path.name)
        fingerprint = _source_fingerprint(share_path)
        requested = ("ts_code", "date", "total_share", "nav")
        for code, frame in _iter_instrument_frames(share_path, requested):
            key = ("etf", code)
            if key not in records:
                continue
            records[key].update(_compute_share_record(frame, fingerprint))

    output = pd.DataFrame(list(records.values()))
    for column in SNAPSHOT_COLUMNS:
        if column not in output.columns:
            output[column] = pd.NA
    output = output[list(SNAPSHOT_COLUMNS)].sort_values(["instrument_type", "ts_code"]).reset_index(drop=True)
    output["latest_date"] = pd.to_datetime(output["latest_date"], errors="coerce")
    for kind in ("etf", "fund"):
        mask = output["instrument_type"].eq(kind)
        segment_latest = output.loc[mask, "latest_date"].max()
        if pd.notna(segment_latest):
            output.loc[mask, "stale_days"] = (
                segment_latest - output.loc[mask, "latest_date"]
            ).dt.days.clip(lower=0)
    configured_metadata: dict[str, Any] | None = None
    if workspace_data_dir is not None:
        output, configured_metadata = _configured_snapshot_values(
            output,
            market_data_dir=target_dir,
            workspace_data_dir=Path(workspace_data_dir).expanduser().resolve(),
        )
    output_path = target_dir / SNAPSHOT_FILENAME
    _atomic_write_snapshot(output, output_path)
    if configured_metadata is not None:
        _atomic_write_json(
            configured_metadata,
            target_dir / SNAPSHOT_METADATA_FILENAME,
        )
    _read_small_parquet_cached.cache_clear()
    by_kind = {
        kind: int((output["instrument_type"] == kind).sum()) for kind in ("etf", "fund")
    }
    return {
        "path": str(output_path),
        "rows": int(len(output)),
        "by_kind": by_kind,
        "as_of": _safe_max_date(output.get("as_of")),
        "source_files": source_files,
        "snapshot_indicators": (
            {
                "config_revision": configured_metadata.get("config_revision"),
                "configured_count": configured_metadata.get("configured_count"),
                "status_counts": configured_metadata.get("status_counts"),
            }
            if configured_metadata is not None
            else {"mode": "legacy_compatibility"}
        ),
    }

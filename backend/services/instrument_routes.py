"""Unified read-only product APIs for ETFs and off-exchange public funds."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as arrow_parquet
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from custom_indicators.series_provider import load_price_points
from services.instrument_analytics import (
    ETF_ONLY_METRICS,
    METRIC_DEFINITIONS,
    PRODUCT_FILTER_METRICS,
    build_analytics_response,
    build_rankings_response,
    build_trend_response,
    load_product_filter_snapshot,
)
try:
    from backend.market_data import resolve_tushare_data_dir
except ModuleNotFoundError:  # pragma: no cover - backend/ direct execution
    from market_data import resolve_tushare_data_dir


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
INSTRUMENT_FILES = {
    "etf": DATA_DIR / "etf_info_df.parquet",
    "fund": DATA_DIR / "fund_info_df.parquet",
}
FILTER_COLUMNS = ("fund_type", "type", "invest_type", "market", "status", "management", "custodian")
COMPARISON_OPERATORS = {
    "gte": {"label": "大于等于", "symbol": "≥"},
    "lte": {"label": "小于等于", "symbol": "≤"},
    "gt": {"label": "大于", "symbol": ">"},
    "lt": {"label": "小于", "symbol": "<"},
    "eq": {"label": "等于", "symbol": "="},
}
PERCENT_INPUT_METRICS = {
    "return_1m",
    "return_3m",
    "return_1y",
    "return_3y",
    "annual_volatility_1y",
    "max_drawdown_3y",
    "premium_discount_latest",
}
MAX_PRODUCT_CONDITIONS = 12
MAX_PRODUCT_SELECTION = 50_000

router = APIRouter(prefix="/api/instruments", tags=["instruments"])


@dataclass(frozen=True)
class ProductCondition:
    field: str
    operator: str
    raw_value: str
    value: date | float
    data_type: Literal["date", "number"]
    input_scale: float = 1.0


def _current_data_dir() -> Path:
    return resolve_tushare_data_dir(DATA_DIR)


def _current_info_files() -> dict[str, Path]:
    configured = {kind: Path(path) for kind, path in INSTRUMENT_FILES.items()}
    defaults = {
        "etf": Path(DATA_DIR) / "etf_info_df.parquet",
        "fund": Path(DATA_DIR) / "fund_info_df.parquet",
    }
    # Preserve explicit test/operator file overrides; the default mapping is
    # the only one that should follow the active manifest pointer.
    if configured != defaults:
        return configured
    current = _current_data_dir()
    if current != Path(DATA_DIR).resolve():
        return {
            "etf": current / "etf_info_df.parquet",
            "fund": current / "fund_info_df.parquet",
        }
    return configured


def _load_instruments(kind: Literal["all", "etf", "fund"]) -> pd.DataFrame:
    kinds = ("etf", "fund") if kind == "all" else (kind,)
    frames: list[pd.DataFrame] = []
    for item_kind in kinds:
        path = _current_info_files()[item_kind]
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        if frame.empty:
            continue
        frame = frame.copy()
        frame["instrument_type"] = item_kind
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False).drop_duplicates(
        subset=["instrument_type", "ts_code"], keep="last"
    )


def _coerce_filter_list(raw: Optional[list[str]]) -> list[str]:
    values: list[str] = []
    for entry in raw or []:
        values.extend(part.strip() for part in str(entry).split(",") if part.strip())
    return values


def _serialize(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (pd.Timestamp, datetime, np.datetime64)):
        parsed = pd.to_datetime(value, errors="coerce")
        return None if pd.isna(parsed) else parsed.strftime("%Y-%m-%d")
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    text = str(value).strip()
    return text or None


def _safe_stat(series: Optional[pd.Series], operation: str) -> Optional[float]:
    if series is None or series.empty:
        return None
    numeric = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if numeric.empty:
        return None
    value = getattr(numeric, operation)()
    return None if pd.isna(value) else float(value)


def _filter_options(df: pd.DataFrame, column: str) -> list[dict[str, Any]]:
    if column not in df.columns:
        return []
    clean = df[column].fillna("未知").astype(str).replace({"": "未知", "nan": "未知"})
    counts = clean.value_counts()
    return [{"value": str(value), "label": str(value), "count": int(count)} for value, count in counts.items()]


def _date_condition_field(kind: Literal["etf", "fund"]) -> str:
    return "list_date" if kind == "etf" else "found_date"


def _allowed_metric_fields(kind: Literal["etf", "fund"]) -> tuple[str, ...]:
    return tuple(
        field
        for field in PRODUCT_FILTER_METRICS
        if kind == "etf" or field not in ETF_ONLY_METRICS
    )


def _condition_fields(
    kind: Literal["etf", "fund"],
    snapshot_state: dict[str, Any],
) -> list[dict[str, Any]]:
    date_field = _date_condition_field(kind)
    result = [
        {
            "field": date_field,
            "label": "上市日期" if kind == "etf" else "成立日期",
            "data_type": "date",
            "unit_label": None,
            "input_scale": 1.0,
            "source": "fund_basic",
            "available": True,
        }
    ]
    snapshot_ready = snapshot_state.get("status") == "ready"
    for field in _allowed_metric_fields(kind):
        definition = METRIC_DEFINITIONS[field]
        is_percent = field in PERCENT_INPUT_METRICS
        result.append(
            {
                "field": field,
                "label": definition["label"],
                "data_type": "number",
                "unit_label": "%" if is_percent else None,
                "input_scale": 100.0 if is_percent else 1.0,
                "source": "instrument_metrics_snapshot",
                "available": snapshot_ready,
            }
        )
    return result


def _parse_product_conditions(
    raw_conditions: Optional[list[str]],
    kind: Literal["etf", "fund"],
) -> list[ProductCondition]:
    entries = raw_conditions if isinstance(raw_conditions, list) else []
    if len(entries) > MAX_PRODUCT_CONDITIONS:
        raise HTTPException(status_code=400, detail=f"筛选条件最多允许 {MAX_PRODUCT_CONDITIONS} 条。")
    date_field = _date_condition_field(kind)
    allowed_metrics = set(_allowed_metric_fields(kind))
    parsed: list[ProductCondition] = []
    for entry in entries:
        parts = str(entry).split("|", 2)
        if len(parts) != 3:
            raise HTTPException(status_code=400, detail="筛选条件格式无效。")
        field, operator, raw_value = (part.strip() for part in parts)
        if operator not in COMPARISON_OPERATORS:
            raise HTTPException(status_code=400, detail=f"不支持的比较方式: {operator}")
        if field == date_field:
            try:
                value = date.fromisoformat(raw_value)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=f"{date_field} 必须使用 YYYY-MM-DD 日期。") from exc
            parsed.append(ProductCondition(field, operator, raw_value, value, "date"))
            continue
        if field not in allowed_metrics:
            raise HTTPException(status_code=400, detail=f"不支持的筛选字段: {field}")
        try:
            numeric_value = float(raw_value)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"{field} 必须是数字。") from exc
        if not np.isfinite(numeric_value):
            raise HTTPException(status_code=400, detail=f"{field} 必须是有限数字。")
        parsed.append(
            ProductCondition(
                field,
                operator,
                raw_value,
                numeric_value,
                "number",
                100.0 if field in PERCENT_INPUT_METRICS else 1.0,
            )
        )
    return parsed


def _comparison_mask(series: pd.Series, condition: ProductCondition) -> pd.Series:
    if condition.data_type == "date":
        values = pd.to_datetime(series, errors="coerce").dt.date
        target = condition.value
    else:
        values = pd.to_numeric(series, errors="coerce") * condition.input_scale
        values = values.replace([np.inf, -np.inf], np.nan)
        target = float(condition.value)
    if condition.operator == "gte":
        return values.ge(target).fillna(False)
    if condition.operator == "lte":
        return values.le(target).fillna(False)
    if condition.operator == "gt":
        return values.gt(target).fillna(False)
    if condition.operator == "lt":
        return values.lt(target).fillna(False)
    if condition.data_type == "number":
        numeric = pd.to_numeric(values, errors="coerce")
        return pd.Series(
            np.isclose(numeric, target, rtol=1e-9, atol=1e-9, equal_nan=False),
            index=series.index,
        )
    return values.eq(target).fillna(False)


def _apply_product_conditions(
    working: pd.DataFrame,
    conditions: list[ProductCondition],
    kind: Literal["etf", "fund"],
    data_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    snapshot, snapshot_state = load_product_filter_snapshot(kind, data_dir)
    metric_fields = list(dict.fromkeys(
        condition.field for condition in conditions if condition.data_type == "number"
    ))
    if metric_fields:
        if snapshot_state.get("status") != "ready":
            raise HTTPException(status_code=409, detail="分析快照未就绪，暂时不能按已计算指标筛选。")
        available_fields = [field for field in metric_fields if field in snapshot.columns]
        if len(available_fields) != len(metric_fields):
            missing = sorted(set(metric_fields) - set(available_fields))
            raise HTTPException(status_code=409, detail=f"分析快照缺少指标字段: {', '.join(missing)}")
        working = working.merge(
            snapshot[["instrument_type", "ts_code", *available_fields]],
            on=["instrument_type", "ts_code"],
            how="inner",
        )
    for condition in conditions:
        if condition.field not in working.columns:
            working = working.iloc[0:0]
            break
        working = working[_comparison_mask(working[condition.field], condition)]
    return working, snapshot_state


def _serialized_conditions(conditions: list[ProductCondition]) -> list[dict[str, Any]]:
    return [
        {
            "field": condition.field,
            "operator": condition.operator,
            "operator_label": COMPARISON_OPERATORS[condition.operator]["label"],
            "operator_symbol": COMPARISON_OPERATORS[condition.operator]["symbol"],
            "value": condition.raw_value,
        }
        for condition in conditions
    ]


def _active_count(df: pd.DataFrame) -> Optional[int]:
    if "status" not in df.columns:
        return None
    status = df["status"].fillna("未知").astype(str)
    inactive = ("终止", "退市", "清盘", "暂停", "到期", "摘牌")
    return int((~status.str.contains("|".join(inactive), case=False, na=False)).sum())


def _match_instrument(df: pd.DataFrame, identifier: str) -> Optional[pd.Series]:
    token = identifier.strip().lower()
    token_nosfx = token.split(".", 1)[0]
    for column in ("ts_code", "code", "name"):
        if column not in df.columns:
            continue
        values = df[column].astype(str).str.lower()
        exact = df[values == token]
        if not exact.empty:
            return exact.iloc[0]
        if column in {"ts_code", "code"}:
            without_suffix = df[values.str.split(".").str[0] == token_nosfx]
            if not without_suffix.empty:
                return without_suffix.iloc[0]
    return None


def _load_timeseries(kind: str, ts_code: str) -> list[dict[str, Any]]:
    if kind not in {"etf", "fund"}:
        return []
    return load_price_points(kind, ts_code, _current_data_dir())


def _load_current_size(kind: str, ts_code: str) -> dict[str, Any]:
    """Read the latest disclosed net asset without scanning unrelated products."""

    if kind not in {"etf", "fund"}:
        return {"current_size": None, "current_size_as_of": None, "current_size_source": None}
    filename = "etf_daily_df.parquet" if kind == "etf" else "fund_nav_df.parquet"
    path = _current_data_dir() / filename
    if not path.exists():
        return {"current_size": None, "current_size_as_of": None, "current_size_source": None}
    try:
        available = set(arrow_parquet.read_schema(path).names)
        asset_columns = [column for column in ("total_netasset", "net_asset") if column in available]
        if "ts_code" not in available or not asset_columns:
            return {"current_size": None, "current_size_as_of": None, "current_size_source": None}
        date_columns = [column for column in ("nav_date", "date", "ann_date") if column in available]
        frame = pd.read_parquet(
            path,
            columns=["ts_code", *date_columns, *asset_columns],
            filters=[("ts_code", "==", ts_code)],
        )
    except (OSError, ValueError, TypeError):
        return {"current_size": None, "current_size_as_of": None, "current_size_source": None}
    if frame.empty:
        return {"current_size": None, "current_size_as_of": None, "current_size_source": None}

    for column in asset_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame.loc[~np.isfinite(frame[column]) | frame[column].le(0), column] = np.nan
    valid_assets = frame[asset_columns].notna().any(axis=1)
    frame = frame.loc[valid_assets].copy()
    if frame.empty:
        return {"current_size": None, "current_size_as_of": None, "current_size_source": None}

    date_column = next((column for column in ("nav_date", "date", "ann_date") if column in frame), None)
    if date_column is not None:
        compact_dates = frame[date_column].astype(str).str.strip()
        parsed_dates = pd.to_datetime(compact_dates, format="%Y%m%d", errors="coerce")
        parsed_dates = parsed_dates.fillna(pd.to_datetime(frame[date_column], errors="coerce"))
        frame["_asset_date"] = parsed_dates
        frame = frame.sort_values("_asset_date", ascending=False, na_position="last", kind="mergesort")

    row = frame.iloc[0]
    source = "total_netasset" if pd.notna(row.get("total_netasset")) else "net_asset"
    amount_yuan = float(row[source])
    as_of = _serialize(row.get("_asset_date")) if "_asset_date" in frame else None
    return {
        # Product-facing amount fields use ten-thousand yuan as the base unit.
        "current_size": amount_yuan / 10_000.0,
        "current_size_as_of": as_of,
        "current_size_source": source,
    }


def _analytics_filters(
    fund_type: Optional[list[str]],
    invest_type: Optional[list[str]],
    status: Optional[list[str]],
    management: Optional[list[str]],
    market: Optional[list[str]],
) -> dict[str, list[str]]:
    return {
        "fund_type": _coerce_filter_list(fund_type),
        "invest_type": _coerce_filter_list(invest_type),
        "status": _coerce_filter_list(status),
        "management": _coerce_filter_list(management),
        "market": _coerce_filter_list(market),
    }


@router.get("/analytics")
def instrument_analytics(
    kind: Literal["all", "etf", "fund"] = Query(default="all"),
    fund_type: Optional[list[str]] = Query(default=None),
    invest_type: Optional[list[str]] = Query(default=None),
    status: Optional[list[str]] = Query(default=None),
    management: Optional[list[str]] = Query(default=None),
    market: Optional[list[str]] = Query(default=None),
):
    """Return small-table dashboard analytics with partial-data semantics."""

    return build_analytics_response(
        kind=kind,
        data_dir=_current_data_dir(),
        info_files=_current_info_files(),
        filters=_analytics_filters(fund_type, invest_type, status, management, market),
    )


@router.get("/analytics/trend")
def instrument_analytics_trend(
    kind: Literal["all", "etf", "fund"] = Query(default="all"),
    dimension: Literal["all", "fund_type", "invest_type", "management"] = Query(default="all"),
    values: Optional[list[str]] = Query(default=None),
    fund_type: Optional[list[str]] = Query(default=None),
    invest_type: Optional[list[str]] = Query(default=None),
    status: Optional[list[str]] = Query(default=None),
    management: Optional[list[str]] = Query(default=None),
    market: Optional[list[str]] = Query(default=None),
):
    return build_trend_response(
        kind=kind,
        dimension=dimension,
        values=_coerce_filter_list(values),
        data_dir=_current_data_dir(),
        info_files=_current_info_files(),
        filters=_analytics_filters(fund_type, invest_type, status, management, market),
    )


@router.get("/analytics/rankings")
def instrument_analytics_rankings(
    kind: Literal["etf", "fund"] = Query(default="etf"),
    metric: str = Query(default="return_1y"),
    sort_dir: Literal["asc", "desc"] = Query(default="desc"),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=10, ge=1, le=50),
    active_only: bool = Query(default=True),
    fund_type: Optional[list[str]] = Query(default=None),
    invest_type: Optional[list[str]] = Query(default=None),
    status: Optional[list[str]] = Query(default=None),
    management: Optional[list[str]] = Query(default=None),
    market: Optional[list[str]] = Query(default=None),
):
    if not active_only:
        raise HTTPException(
            status_code=400,
            detail="排行仅允许纳入 status_code=L 的上市/存续份额。",
        )
    try:
        return build_rankings_response(
            kind=kind,
            metric=metric,
            sort_dir=sort_dir,
            page=page,
            page_size=page_size,
            active_only=active_only,
            data_dir=_current_data_dir(),
            info_files=_current_info_files(),
            filters=_analytics_filters(fund_type, invest_type, status, management, market),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/search")
def instrument_search(
    q: str = Query(default=""),
    kind: Literal["all", "etf", "fund"] = Query(default="all"),
    sort_by: str = Query(default="name"),
    sort_dir: Literal["asc", "desc"] = Query(default="asc"),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=10, ge=1, le=200),
):
    df = _load_instruments(kind)
    keyword = q.strip().lower()
    if keyword and not df.empty:
        mask = pd.Series(False, index=df.index)
        for column in ("ts_code", "code", "name", "management"):
            if column in df.columns:
                mask |= df[column].astype(str).str.lower().str.contains(keyword, regex=False, na=False)
        df = df[mask]
    sort_column = sort_by if sort_by in {"name", "code", "management", "found_date"} else "name"
    if sort_column in df.columns:
        df = df.sort_values(sort_column, ascending=sort_dir == "asc", na_position="last", kind="mergesort")
    total = int(len(df))
    start = (page - 1) * page_size
    items = []
    for _, row in df.iloc[start : start + page_size].iterrows():
        ts_code = _serialize(row.get("ts_code"))
        items.append(
            {
                "code": ts_code or _serialize(row.get("code")),
                "ts_code": ts_code,
                "name": _serialize(row.get("name")),
                "management": _serialize(row.get("management")),
                "found_date": _serialize(row.get("found_date")),
                "instrument_type": _serialize(row.get("instrument_type")),
            }
        )
    return {"items": items, "total": total, "page": page, "page_size": page_size, "kind": kind}


def _filter_product_frame(
    *,
    kind: Literal["etf", "fund"],
    q: str,
    filters: dict[str, list[str]],
    conditions: Optional[list[str]],
    sort_by: str,
    sort_dir: Literal["asc", "desc"],
) -> tuple[pd.DataFrame, pd.DataFrame, list[ProductCondition], dict[str, Any], str]:
    """Apply one filter contract to the paged list and full-selection endpoints."""
    universe = _load_instruments(kind)
    if universe.empty:
        filename = _current_info_files()[kind].name
        raise HTTPException(status_code=404, detail=f"未找到 {filename} 数据文件")
    working = universe.copy()
    keyword = q.strip().lower()
    if keyword:
        mask = pd.Series(False, index=working.index)
        for column in ("ts_code", "code", "name", "management"):
            if column in working.columns:
                mask |= working[column].astype(str).str.lower().str.contains(keyword, regex=False, na=False)
        working = working[mask]

    for column, values in filters.items():
        if not values or column not in working.columns:
            continue
        candidates = {value.lower() for value in values}
        clean = working[column].fillna("未知").astype(str).replace({"": "未知", "nan": "未知"})
        working = working[clean.str.lower().isin(candidates)]

    parsed_conditions = _parse_product_conditions(conditions, kind)
    working, snapshot_state = _apply_product_conditions(
        working,
        parsed_conditions,
        kind,
        _current_data_dir(),
    )
    sortable = {
        "issue_amount",
        "m_fee",
        "c_fee",
        "exp_return",
        "duration_year",
        "list_date",
        "found_date",
        "issue_date",
        "name",
    }
    sort_column = sort_by if sort_by in sortable else "issue_amount"
    if sort_column in working.columns:
        working = working.sort_values(
            sort_column,
            ascending=sort_dir == "asc",
            na_position="last",
            kind="mergesort",
        )
    return universe, working, parsed_conditions, snapshot_state, sort_column


def _product_filters(
    fund_type: Optional[list[str]],
    fund_category: Optional[list[str]],
    invest_type: Optional[list[str]],
    market: Optional[list[str]],
    status: Optional[list[str]],
    management: Optional[list[str]],
    custodian: Optional[list[str]],
) -> dict[str, list[str]]:
    return {
        "fund_type": _coerce_filter_list(fund_type),
        "type": _coerce_filter_list(fund_category),
        "invest_type": _coerce_filter_list(invest_type),
        "market": _coerce_filter_list(market),
        "status": _coerce_filter_list(status),
        "management": _coerce_filter_list(management),
        "custodian": _coerce_filter_list(custodian),
    }


@router.get("/products")
def instrument_products(
    kind: Literal["etf", "fund"] = Query(default="etf"),
    q: str = Query(default=""),
    fund_type: Optional[list[str]] = Query(default=None),
    fund_category: Optional[list[str]] = Query(default=None, alias="type"),
    invest_type: Optional[list[str]] = Query(default=None),
    market: Optional[list[str]] = Query(default=None),
    status: Optional[list[str]] = Query(default=None),
    management: Optional[list[str]] = Query(default=None),
    custodian: Optional[list[str]] = Query(default=None),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=10, ge=1, le=200),
    sort_by: str = Query(default="issue_amount"),
    sort_dir: Literal["asc", "desc"] = Query(default="desc"),
    conditions: Optional[list[str]] = Query(default=None, alias="condition"),
):
    filters = _product_filters(
        fund_type,
        fund_category,
        invest_type,
        market,
        status,
        management,
        custodian,
    )
    universe, working, parsed_conditions, snapshot_state, sort_column = _filter_product_frame(
        kind=kind,
        q=q,
        filters=filters,
        conditions=conditions,
        sort_by=sort_by,
        sort_dir=sort_dir,
    )

    total = int(len(working))
    start = (page - 1) * page_size
    preferred = [
        "ts_code",
        "code",
        "name",
        "instrument_type",
        "management",
        "custodian",
        "fund_type",
        "type",
        "invest_type",
        "market",
        "status",
        "benchmark",
        "issue_amount",
        "m_fee",
        "c_fee",
        "exp_return",
        "duration_year",
        "list_date",
        "found_date",
        "issue_date",
        "due_date",
        "purc_startdate",
        "redm_startdate",
    ]
    present = [column for column in preferred if column in working.columns]
    condition_fields = list(dict.fromkeys(condition.field for condition in parsed_conditions))
    items = []
    for _, row in working.iloc[start : start + page_size].iterrows():
        item = {column: _serialize(row[column]) for column in present}
        item["condition_values"] = {
            field: _serialize(row.get(field)) for field in condition_fields
        }
        items.append(item)
    recent_cutoff = pd.Timestamp.today().normalize() - pd.DateOffset(months=12)
    recent_count = None
    if "list_date" in working.columns:
        dates = pd.to_datetime(working["list_date"], errors="coerce")
        recent_count = int((dates >= recent_cutoff).sum())
    summary = {
        "universe_total": int(len(universe)),
        "filtered_total": total,
        "active_count": _active_count(working),
        "recent_listings_12m": recent_count,
        "avg_m_fee": _safe_stat(working.get("m_fee"), "mean"),
        "avg_c_fee": _safe_stat(working.get("c_fee"), "mean"),
        "total_issue_amount": _safe_stat(working.get("issue_amount"), "sum"),
        "median_issue_amount": _safe_stat(working.get("issue_amount"), "median"),
        "unique_managements": int(working["management"].nunique(dropna=True)) if "management" in working.columns else None,
    }
    return {
        "items": items,
        "page": page,
        "page_size": page_size,
        "total": total,
        "summary": summary,
        "available_filters": {column: _filter_options(universe, column) for column in FILTER_COLUMNS},
        "condition_fields": _condition_fields(kind, snapshot_state),
        "condition_operators": [
            {"value": key, **metadata} for key, metadata in COMPARISON_OPERATORS.items()
        ],
        "applied_conditions": _serialized_conditions(parsed_conditions),
        "snapshot": snapshot_state,
        "sort_by": sort_column,
        "sort_dir": sort_dir,
        "kind": kind,
    }


@router.get("/products/selection")
def instrument_product_selection(
    kind: Literal["etf", "fund"] = Query(default="etf"),
    q: str = Query(default=""),
    fund_type: Optional[list[str]] = Query(default=None),
    fund_category: Optional[list[str]] = Query(default=None, alias="type"),
    invest_type: Optional[list[str]] = Query(default=None),
    market: Optional[list[str]] = Query(default=None),
    status: Optional[list[str]] = Query(default=None),
    management: Optional[list[str]] = Query(default=None),
    custodian: Optional[list[str]] = Query(default=None),
    sort_by: str = Query(default="name"),
    sort_dir: Literal["asc", "desc"] = Query(default="asc"),
    conditions: Optional[list[str]] = Query(default=None, alias="condition"),
):
    """Return only identities for all matching products without paging metadata."""
    filters = _product_filters(
        fund_type,
        fund_category,
        invest_type,
        market,
        status,
        management,
        custodian,
    )
    _, working, _, _, _ = _filter_product_frame(
        kind=kind,
        q=q,
        filters=filters,
        conditions=conditions,
        sort_by=sort_by,
        sort_dir=sort_dir,
    )
    total = int(len(working))
    if total > MAX_PRODUCT_SELECTION:
        raise HTTPException(
            status_code=422,
            detail=f"筛选结果超过 {MAX_PRODUCT_SELECTION} 个产品，请增加筛选条件后重试。",
        )
    items = []
    for _, row in working.iterrows():
        ts_code = _serialize(row.get("ts_code"))
        code = ts_code or _serialize(row.get("code"))
        if not code:
            continue
        items.append(
            {
                "code": code,
                "ts_code": ts_code,
                "name": _serialize(row.get("name")) or code,
                "instrument_type": kind,
            }
        )
    return {"items": items, "total": len(items), "kind": kind}


@router.get("/products/{product_id}")
def instrument_product_detail(
    product_id: str,
    kind: Literal["etf", "fund"] = Query(default="etf"),
):
    df = _load_instruments(kind)
    if df.empty:
        return JSONResponse(status_code=404, content={"detail": f"未找到 {_current_info_files()[kind].name} 数据文件"})
    record = _match_instrument(df, product_id)
    if record is None:
        return JSONResponse(status_code=404, content={"detail": f"未找到编号为 {product_id} 的产品"})
    base_columns = [
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
        "status",
        "benchmark",
        "found_date",
        "issue_date",
        "list_date",
        "due_date",
        "delist_date",
        "purc_startdate",
        "redm_startdate",
    ]
    metric_columns = ("issue_amount", "m_fee", "c_fee", "exp_return", "duration_year")
    base_info = {column: _serialize(record.get(column)) for column in base_columns if column in record.index}
    metrics = {column: _serialize(record.get(column)) for column in metric_columns}
    ts_code = str(record.get("ts_code") or product_id)
    metrics.update(_load_current_size(kind, ts_code))
    return {
        "product_id": ts_code,
        "name": _serialize(record.get("name")) or ts_code,
        "management": _serialize(record.get("management")),
        "custodian": _serialize(record.get("custodian")),
        "status": _serialize(record.get("status")),
        "base_info": base_info,
        "metrics": metrics,
        "timeseries": _load_timeseries(kind, ts_code),
        "kind": kind,
    }

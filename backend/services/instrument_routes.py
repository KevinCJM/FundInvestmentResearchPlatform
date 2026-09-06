"""Unified read-only product APIs for ETFs and off-exchange public funds."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Annotated, Any, Literal, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from custom_indicators.errors import NotFoundError
from custom_indicators.series_provider import load_price_points
from historical_regimes.repository import RegimeRunRepository
from services.instrument_analytics import (
    ETF_ONLY_METRICS,
    METRIC_DEFINITIONS,
    PRODUCT_FILTER_METRICS,
    build_analytics_response,
    build_rankings_response,
    build_trend_response,
    load_product_filter_snapshot,
    snapshot_etf_only_metrics,
    snapshot_metric_definitions,
)
from services.product_analysis import build_product_analysis_response
from services.product_compare import build_product_compare_response
try:
    from backend.compute_policy import validate_execution_audit
    from backend.instrument_analytics_numba import (
        count_true_kernel,
        coverage_ratio_kernel,
        encoded_category_counts_kernel,
        encoded_unique_count_kernel,
        instrument_analytics_numba_execution_audit,
        numeric_comparison_mask_kernel,
        numeric_sort_order_kernel,
        numeric_stat_kernel,
    )
    from backend.market_data import resolve_tushare_data_dir
except ModuleNotFoundError:  # pragma: no cover - backend/ direct execution
    from compute_policy import validate_execution_audit
    from instrument_analytics_numba import (
        count_true_kernel,
        coverage_ratio_kernel,
        encoded_category_counts_kernel,
        encoded_unique_count_kernel,
        instrument_analytics_numba_execution_audit,
        numeric_comparison_mask_kernel,
        numeric_sort_order_kernel,
        numeric_stat_kernel,
    )
    from market_data import resolve_tushare_data_dir


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
INSTRUMENT_FILES = {
    "etf": DATA_DIR / "etf_info_df.parquet",
    "fund": DATA_DIR / "fund_info_df.parquet",
}
FILTER_COLUMNS = (
    "fund_type",
    "type",
    "invest_type",
    "qdii_type",
    "market",
    "status",
    "management",
    "custodian",
)
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


def _historical_regime_workspace_dir() -> Path:
    configured = os.getenv("HISTORICAL_REGIME_DATA_DIR") or os.getenv(
        "CUSTOM_INDICATOR_DATA_DIR"
    )
    return Path(configured).expanduser() if configured else DATA_DIR


def _historical_regime_snapshot_hash(run: dict[str, Any]) -> str:
    """Recreate the immutable analytical hash before repository metadata."""

    analytical = copy.deepcopy(run)
    for key in ("id", "created_at", "immutable", "publications", "content_hash"):
        analytical.pop(key, None)
    analytical["application_bindings"] = []
    encoded = json.dumps(
        analytical,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _resolve_product_regime_reference(
    reference: ProductAnalysisRegime,
) -> tuple[dict[str, Any], dict[str, Any]]:
    repository = RegimeRunRepository(
        _historical_regime_workspace_dir() / "historical_regime_runs.json"
    )
    try:
        run = repository.get(reference.run_id)
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail="未找到指定的历史情景运行版本。") from exc
    if run.get("immutable") is not True:
        raise HTTPException(status_code=409, detail="产品研究只能引用不可变的历史情景运行。")
    if _historical_regime_snapshot_hash(run) != run.get("content_hash"):
        raise HTTPException(status_code=409, detail="历史情景运行快照校验失败，已阻断产品研究引用。")
    publication = next(
        (
            item
            for item in run.get("publications") or []
            if item.get("id") == reference.publication_id
        ),
        None,
    )
    if not isinstance(publication, dict):
        raise HTTPException(status_code=422, detail="历史情景运行没有匹配的发布记录。")
    if publication.get("usage") != "product_research":
        raise HTTPException(status_code=422, detail="该历史情景版本未发布到产品研究。")
    if (
        publication.get("run_id") != run.get("id")
        or publication.get("run_content_hash") != run.get("content_hash")
        or publication.get("definition_revision") != run.get("definition_revision")
    ):
        raise HTTPException(status_code=409, detail="历史情景发布记录与运行版本不一致。")
    states = run.get("states")
    segments = run.get("segments")
    if not isinstance(states, list) or not states or not isinstance(segments, list):
        raise HTTPException(status_code=422, detail="历史情景运行缺少产品研究所需的状态或区间。")
    analytical_input = {
        "states": [
            {
                "id": str(item.get("id") or ""),
                "label": str(item.get("label") or item.get("id") or ""),
                "color": str(item.get("color") or "#64748b"),
            }
            for item in states
            if isinstance(item, dict) and item.get("id")
        ],
        "segments": [
            {
                "state_id": str(item.get("state_id") or ""),
                "start_date": str(item.get("start_date") or ""),
                "end_date": str(item.get("end_date") or ""),
            }
            for item in segments
            if isinstance(item, dict)
            and item.get("state_id")
            and item.get("start_date")
            and item.get("end_date")
        ],
    }
    lineage = {
        "run_id": run["id"],
        "publication_id": publication["id"],
        "definition_id": run.get("definition_id"),
        "definition_revision": run.get("definition_revision"),
        "run_content_hash": run["content_hash"],
        "usage": publication["usage"],
    }
    return analytical_input, lineage


MAX_PRODUCT_CONDITIONS = 12
MAX_PRODUCT_SELECTION = 50_000
MAX_DISPLAY_SNAPSHOT_METRICS = 8
SNAPSHOT_METRIC_SOURCE_LABELS = {
    "built_in": "内置指标",
    "custom": "工作区指标",
    "system_derived": "系统衍生指标",
}
NUMERIC_PRODUCT_SORT_FIELDS = {
    "issue_amount",
    "m_fee",
    "c_fee",
    "exp_return",
    "duration_year",
}
_COMPARISON_OPCODE = {"gte": 0, "lte": 1, "gt": 2, "lt": 3, "eq": 4}
_STAT_OPCODE = {"mean": 0, "sum": 1, "median": 2}

router = APIRouter(prefix="/api/instruments", tags=["instruments"])


@dataclass(frozen=True)
class ProductCondition:
    field: str
    operator: str
    raw_value: str
    value: date | float
    data_type: Literal["date", "number"]
    input_scale: float = 1.0


class ProductAnalysisRegimeState(BaseModel):
    id: str = Field(min_length=1, max_length=80)
    label: str = Field(min_length=1, max_length=120)
    color: str = Field(default="#64748b", max_length=32)


class ProductAnalysisRegimeSegment(BaseModel):
    state_id: str = Field(min_length=1, max_length=80)
    start_date: str = Field(min_length=8, max_length=32)
    end_date: str = Field(min_length=8, max_length=32)


class ProductAnalysisRegime(BaseModel):
    """Immutable published historical-regime reference for product research."""

    run_id: str = Field(min_length=1, max_length=128)
    publication_id: str = Field(min_length=1, max_length=128)


ProductAnalysisMaPeriod = Annotated[int, Field(ge=2, le=500)]


class ProductAnalysisRequest(BaseModel):
    statistics_period: Literal["ALL", "1M", "3M", "6M", "1Y", "3Y", "5Y"] = "ALL"
    include_technical: bool = True
    price_ma_periods: list[ProductAnalysisMaPeriod] = Field(default_factory=lambda: [5, 10, 20], max_length=8)
    volume_ma_periods: list[ProductAnalysisMaPeriod] = Field(default_factory=lambda: [5, 10], max_length=8)
    boll_period: int = Field(default=20, ge=2, le=500)
    boll_multiplier: float = Field(default=2.0, ge=0.5, le=10.0, allow_inf_nan=False)
    kdj_period: int = Field(default=9, ge=2, le=500)
    kdj_k_smoothing: int = Field(default=3, ge=1, le=100)
    kdj_d_smoothing: int = Field(default=3, ge=1, le=100)
    histogram_bin_width: float = Field(default=0.2, ge=0.01, le=100.0, allow_inf_nan=False)
    simulation_horizon: int = Field(default=252, ge=1, le=504)
    simulation_path_count: int = Field(default=500, ge=1, le=1_000)
    bootstrap_block_length: int = Field(default=20, ge=1, le=504)
    simulation_target_return: float = Field(default=5.0, ge=-99.0, le=1_000.0, allow_inf_nan=False)
    simulation_run: int = Field(default=0, ge=0, le=2_147_483_647)
    regime: ProductAnalysisRegime | None = None


class ProductCompareRange(BaseModel):
    start_date: str | None = Field(default=None, max_length=32)
    end_date: str | None = Field(default=None, max_length=32)


class ProductCompareRanges(BaseModel):
    performance: ProductCompareRange
    risk: ProductCompareRange
    efficiency: ProductCompareRange


class ProductCompareAnalysisRequest(BaseModel):
    ranges: ProductCompareRanges
    rolling_window_days: int = Field(default=30, ge=2, le=252)
    management_fee: float | None = Field(default=None, allow_inf_nan=False)
    custody_fee: float | None = Field(default=None, allow_inf_nan=False)


def _instrument_execution_audit() -> dict[str, object]:
    return validate_execution_audit(instrument_analytics_numba_execution_audit())


def _numeric_array(series: pd.Series) -> np.ndarray:
    return np.array(
        pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64),
        dtype=np.float64,
        copy=True,
        order="C",
    )


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
        names = frame.get("name", pd.Series([None] * len(frame), index=frame.index))
        name_qdii = names.astype("string").str.contains("QDII", case=False, na=False)
        negative_label = "非QDII" if item_kind == "fund" else "待确认"
        derived_qdii = name_qdii.map({True: "QDII", False: negative_label})
        if "qdii_type" not in frame.columns:
            frame["qdii_type"] = derived_qdii
        else:
            clean_qdii = frame["qdii_type"].astype("string").str.strip()
            frame["qdii_type"] = clean_qdii.where(clean_qdii.ne("") & clean_qdii.notna(), derived_qdii)
        if "qdii_source" not in frame.columns:
            frame["qdii_source"] = "legacy_info.unavailable" if item_kind == "etf" else "legacy_info.name_marker"
            frame.loc[name_qdii, "qdii_source"] = "legacy_info.name_marker"
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False).drop_duplicates(
        subset=["instrument_type", "ts_code"], keep="last"
    )


def _coerce_filter_list(raw: Optional[list[str]]) -> list[str]:
    if not isinstance(raw, (list, tuple)):
        return []
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
    if operation not in _STAT_OPCODE:
        raise ValueError(f"不支持的统计方法: {operation}")
    value = numeric_stat_kernel(_numeric_array(series), _STAT_OPCODE[operation])
    return float(value) if np.isfinite(value) else None


def _category_codes(series: pd.Series) -> tuple[list[str], np.ndarray]:
    """Map text fields to stable integer codes; counting remains in NJIT."""

    labels: list[str] = []
    code_by_label: dict[str, int] = {}
    codes = np.full(len(series), -1, dtype=np.int64)
    for position, raw_value in enumerate(series.tolist()):
        if pd.isna(raw_value):
            continue
        label = str(raw_value)
        code = code_by_label.get(label)
        if code is None:
            code = len(labels)
            code_by_label[label] = code
            labels.append(label)
        codes[position] = code
    return labels, np.ascontiguousarray(codes)


def _category_count_rows(series: pd.Series) -> list[tuple[str, int]]:
    labels, codes = _category_codes(series)
    if not labels:
        return []
    counts = encoded_category_counts_kernel(codes, len(labels))
    order = numeric_sort_order_kernel(
        np.ascontiguousarray(counts.astype(np.float64)),
        np.arange(len(labels), dtype=np.int64),
        np.uint8(0),
    )
    return [(labels[index], int(counts[index])) for index in order]


def _category_unique_count(series: pd.Series) -> int:
    _, codes = _category_codes(series)
    return int(encoded_unique_count_kernel(codes))


def _filter_options(df: pd.DataFrame, column: str) -> list[dict[str, Any]]:
    if column not in df.columns:
        return []
    clean = df[column].fillna("未知").astype(str).replace({"": "未知", "nan": "未知"})
    return [
        {"value": value, "label": value, "count": count}
        for value, count in _category_count_rows(clean)
    ]


def _date_condition_field(kind: Literal["etf", "fund"]) -> str:
    return "list_date" if kind == "etf" else "found_date"


def _allowed_metric_fields(kind: Literal["etf", "fund"]) -> tuple[str, ...]:
    data_dir = _current_data_dir()
    fields = snapshot_metric_definitions(data_dir)
    etf_only = snapshot_etf_only_metrics(data_dir)
    return tuple(
        field
        for field in fields
        if kind == "etf" or field not in etf_only
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
    definitions = snapshot_metric_definitions(_current_data_dir())
    for field in _allowed_metric_fields(kind):
        definition = definitions[field]
        is_percent = (
            definition.get("unit") == "ratio"
            or definition.get("presentation", {}).get("display_format") == "percent"
        )
        metric_ready = (
            snapshot_state.get("metric_availability", {}).get(field, "ready") == "ready"
        )
        result.append(
            {
                "field": field,
                "label": definition["label"],
                "data_type": "number",
                "unit_label": "%" if is_percent else None,
                "input_scale": 100.0 if is_percent else 1.0,
                "source": "instrument_metrics_snapshot",
                "available": snapshot_ready and metric_ready,
            }
        )
    return result


def _snapshot_metric_fields(
    kind: Literal["etf", "fund"],
    snapshot_state: dict[str, Any],
) -> list[dict[str, Any]]:
    definitions = snapshot_metric_definitions(_current_data_dir())
    return [
        {
            **field,
            "unit": definitions[field["field"]]["unit"],
            "description": definitions[field["field"]]["source"],
            "metric_source": definitions[field["field"]].get(
                "metric_source", "system_derived"
            ),
            "metric_source_label": SNAPSHOT_METRIC_SOURCE_LABELS.get(
                definitions[field["field"]].get("metric_source", "system_derived"),
                "其他来源",
            ),
            "metric_type": definitions[field["field"]].get("metric_type", "other"),
            "metric_type_label": definitions[field["field"]].get(
                "metric_type_label", "其他指标"
            ),
            "indicator_id": definitions[field["field"]].get("indicator_id"),
            "indicator_revision": definitions[field["field"]].get("indicator_revision"),
            "period": definitions[field["field"]].get("period"),
            "presentation": definitions[field["field"]].get("presentation"),
        }
        for field in _condition_fields(kind, snapshot_state)
        if field["source"] == "instrument_metrics_snapshot"
    ]


def _parse_snapshot_metrics(
    raw_metrics: Optional[list[str]],
    kind: Literal["etf", "fund"],
) -> list[str]:
    entries = raw_metrics if isinstance(raw_metrics, list) else []
    metrics = list(dict.fromkeys(_coerce_filter_list(entries)))
    if len(metrics) > MAX_DISPLAY_SNAPSHOT_METRICS:
        raise HTTPException(
            status_code=400,
            detail=f"产品列表最多展示 {MAX_DISPLAY_SNAPSHOT_METRICS} 个快照指标。",
        )
    allowed = set(_allowed_metric_fields(kind))
    invalid = [metric for metric in metrics if metric not in allowed]
    if invalid:
        raise HTTPException(status_code=400, detail=f"不支持的快照指标: {', '.join(invalid)}")
    return metrics


def _parse_product_conditions(
    raw_conditions: Optional[list[str]],
    kind: Literal["etf", "fund"],
) -> list[ProductCondition]:
    entries = raw_conditions if isinstance(raw_conditions, list) else []
    if len(entries) > MAX_PRODUCT_CONDITIONS:
        raise HTTPException(status_code=400, detail=f"筛选条件最多允许 {MAX_PRODUCT_CONDITIONS} 条。")
    date_field = _date_condition_field(kind)
    allowed_metrics = set(_allowed_metric_fields(kind))
    metric_definitions = snapshot_metric_definitions(_current_data_dir())
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
                100.0
                if (
                    field in PERCENT_INPUT_METRICS
                    or metric_definitions.get(field, {}).get("unit") == "ratio"
                    or metric_definitions.get(field, {}).get("presentation", {}).get("display_format") == "percent"
                )
                else 1.0,
            )
        )
    return parsed


def _comparison_mask(series: pd.Series, condition: ProductCondition) -> pd.Series:
    if condition.data_type == "date":
        values = pd.to_datetime(series, errors="coerce").dt.date
        target = condition.value
    else:
        mask = numeric_comparison_mask_kernel(
            _numeric_array(series),
            float(condition.value),
            float(condition.input_scale),
            _COMPARISON_OPCODE[condition.operator],
        )
        return pd.Series(mask.astype(bool), index=series.index)
    if condition.operator == "gte":
        return values.ge(target).fillna(False)
    if condition.operator == "lte":
        return values.le(target).fillna(False)
    if condition.operator == "gt":
        return values.gt(target).fillna(False)
    if condition.operator == "lt":
        return values.lt(target).fillna(False)
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
        unavailable = [
            field
            for field in metric_fields
            if snapshot_state.get("metric_availability", {}).get(field, "ready") != "ready"
        ]
        if unavailable:
            raise HTTPException(
                status_code=409,
                detail=f"分析快照中的指标尚未就绪: {', '.join(unavailable)}",
            )
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
    mask = (~status.str.contains("|".join(inactive), case=False, na=False)).to_numpy(
        dtype=np.uint8
    )
    return int(
        count_true_kernel(
            np.array(mask, dtype=np.uint8, copy=True, order="C")
        )
    )


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
    return load_price_points(kind, ts_code, _current_data_dir(), preserve_missing=True)


def _empty_current_size() -> dict[str, Any]:
    return {
        "current_size": None,
        "current_size_as_of": None,
        "current_size_source": None,
        "current_share": None,
        "current_unit_nav": None,
    }


def _load_current_size(kind: str, ts_code: str) -> dict[str, Any]:
    """Read the derived current-size metric from the validated local snapshot."""

    if kind != "etf":
        return _empty_current_size()
    snapshot, state = load_product_filter_snapshot("etf", _current_data_dir())
    if state.get("status") != "ready" or snapshot.empty:
        return _empty_current_size()
    rows = snapshot[snapshot["ts_code"].astype(str).str.strip().eq(ts_code.strip())]
    if rows.empty:
        return _empty_current_size()
    row = rows.iloc[0]
    current_size = _serialize(row.get("current_size"))
    if current_size is None:
        return _empty_current_size()
    return {
        "current_size": current_size,
        "current_size_as_of": _serialize(row.get("current_size_as_of")),
        "current_size_source": "instrument_metrics_snapshot",
        "current_share": _serialize(row.get("current_share")),
        "current_unit_nav": _serialize(row.get("current_unit_nav")),
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
    if sort_column in working.columns and sort_column in NUMERIC_PRODUCT_SORT_FIELDS:
        sort_values = _numeric_array(working[sort_column])
        stable_rank = np.arange(len(working), dtype=np.int64)
        sort_order = numeric_sort_order_kernel(
            sort_values,
            stable_rank,
            np.uint8(1 if sort_dir == "asc" else 0),
        )
        working = working.iloc[sort_order]
    elif sort_column in working.columns:
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
    qdii_type: Optional[list[str]] = None,
) -> dict[str, list[str]]:
    return {
        "fund_type": _coerce_filter_list(fund_type),
        "type": _coerce_filter_list(fund_category),
        "invest_type": _coerce_filter_list(invest_type),
        "market": _coerce_filter_list(market),
        "status": _coerce_filter_list(status),
        "management": _coerce_filter_list(management),
        "custodian": _coerce_filter_list(custodian),
        "qdii_type": _coerce_filter_list(qdii_type),
    }


def _attach_snapshot_metrics(
    frame: pd.DataFrame,
    *,
    kind: Literal["etf", "fund"],
    metrics: list[str],
) -> pd.DataFrame:
    if not metrics or frame.empty:
        return frame
    snapshot, state = load_product_filter_snapshot(kind, _current_data_dir())
    if state.get("status") != "ready" or snapshot.empty:
        return frame
    context_fields = [
        field
        for field in ("as_of", "current_size_as_of")
        if field in snapshot.columns
    ]
    value_fields = [field for field in metrics if field in snapshot.columns]
    metric_context_fields = [
        f"{field}__{suffix}"
        for field in value_fields
        for suffix in ("status", "effective_as_of", "warning_message")
        if f"{field}__{suffix}" in snapshot.columns
    ]
    snapshot_columns = list(dict.fromkeys([
        "instrument_type",
        "ts_code",
        *value_fields,
        *metric_context_fields,
        *context_fields,
    ]))
    overlapping = [
        column
        for column in (*value_fields, *context_fields)
        if column in frame.columns
    ]
    return frame.drop(columns=overlapping).merge(
        snapshot[snapshot_columns],
        on=["instrument_type", "ts_code"],
        how="left",
    )


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
    snapshot_metrics: Optional[list[str]] = Query(default=None, alias="snapshot_metric"),
    qdii_type: Optional[list[str]] = Query(default=None),
):
    filters = _product_filters(
        fund_type,
        fund_category,
        invest_type,
        market,
        status,
        management,
        custodian,
        qdii_type,
    )
    universe, working, parsed_conditions, snapshot_state, sort_column = _filter_product_frame(
        kind=kind,
        q=q,
        filters=filters,
        conditions=conditions,
        sort_by=sort_by,
        sort_dir=sort_dir,
    )

    selected_snapshot_metrics = _parse_snapshot_metrics(snapshot_metrics, kind)
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
        "qdii_type",
        "qdii_source",
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
    page_frame = _attach_snapshot_metrics(
        working.iloc[start : start + page_size].copy(),
        kind=kind,
        metrics=selected_snapshot_metrics,
    )
    items = []
    for _, row in page_frame.iterrows():
        item = {column: _serialize(row[column]) for column in present}
        item["condition_values"] = {
            field: _serialize(row.get(field)) for field in condition_fields
        }
        item["snapshot_values"] = {
            field: _serialize(row.get(field)) for field in selected_snapshot_metrics
        }
        item["snapshot_value_dates"] = {
            field: _serialize(
                row.get(f"{field}__effective_as_of")
                if f"{field}__effective_as_of" in row.index
                else row.get("current_size_as_of") if field == "current_size" else row.get("as_of")
            )
            for field in selected_snapshot_metrics
        }
        item["snapshot_statuses"] = {
            field: _serialize(row.get(f"{field}__status"))
            for field in selected_snapshot_metrics
        }
        item["snapshot_warnings"] = {
            field: _serialize(row.get(f"{field}__warning_message"))
            for field in selected_snapshot_metrics
        }
        items.append(item)
    recent_cutoff = pd.Timestamp.today().normalize() - pd.DateOffset(months=12)
    recent_count = None
    if "list_date" in working.columns:
        dates = pd.to_datetime(working["list_date"], errors="coerce")
        recent_mask = (dates >= recent_cutoff).to_numpy(dtype=np.uint8)
        recent_count = int(
            count_true_kernel(
                np.array(recent_mask, dtype=np.uint8, copy=True, order="C")
            )
        )
    active_count = _active_count(working)
    active_rate = (
        None
        if active_count is None
        else _serialize(coverage_ratio_kernel(active_count, total))
    )
    summary = {
        "universe_total": int(len(universe)),
        "filtered_total": total,
        "active_count": active_count,
        "active_rate": active_rate,
        "recent_listings_12m": recent_count,
        "avg_m_fee": _safe_stat(working.get("m_fee"), "mean"),
        "avg_c_fee": _safe_stat(working.get("c_fee"), "mean"),
        "avg_exp_return": _safe_stat(working.get("exp_return"), "mean"),
        "avg_duration_year": _safe_stat(working.get("duration_year"), "mean"),
        "total_issue_amount": _safe_stat(working.get("issue_amount"), "sum"),
        "median_issue_amount": _safe_stat(working.get("issue_amount"), "median"),
        "unique_managements": (
            _category_unique_count(working["management"])
            if "management" in working.columns
            else None
        ),
    }
    return {
        "items": items,
        "page": page,
        "page_size": page_size,
        "total": total,
        "summary": summary,
        "available_filters": {column: _filter_options(universe, column) for column in FILTER_COLUMNS},
        "condition_fields": _condition_fields(kind, snapshot_state),
        "snapshot_metric_fields": _snapshot_metric_fields(kind, snapshot_state),
        "selected_snapshot_metrics": selected_snapshot_metrics,
        "condition_operators": [
            {"value": key, **metadata} for key, metadata in COMPARISON_OPERATORS.items()
        ],
        "applied_conditions": _serialized_conditions(parsed_conditions),
        "snapshot": snapshot_state,
        "sort_by": sort_column,
        "sort_dir": sort_dir,
        "kind": kind,
        "execution": _instrument_execution_audit(),
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
    qdii_type: Optional[list[str]] = Query(default=None),
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
        qdii_type,
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


@router.post("/products/{product_id}/analysis")
def instrument_product_analysis(
    product_id: str,
    request: ProductAnalysisRequest,
    kind: Literal["etf", "fund"] = Query(default="etf"),
):
    """Run every ProductDetail numerical panel through prewarmed NJIT kernels."""

    instruments = _load_instruments(kind)
    if instruments.empty:
        raise HTTPException(
            status_code=404,
            detail=f"未找到 {_current_info_files()[kind].name} 数据文件",
        )
    record = _match_instrument(instruments, product_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"未找到编号为 {product_id} 的产品")
    ts_code = str(record.get("ts_code") or product_id)
    points = _load_timeseries(kind, ts_code)
    if not points:
        raise HTTPException(status_code=422, detail="产品真实行情数据不足，无法执行分析。")
    try:
        parameters = request.model_dump()
        regime_lineage = None
        if request.regime is not None:
            regime_input, regime_lineage = _resolve_product_regime_reference(
                request.regime
            )
            parameters["regime"] = regime_input
        response = build_product_analysis_response(
            product_id=ts_code,
            points=points,
            parameters=parameters,
        )
        if regime_lineage is not None:
            response["regimeReference"] = regime_lineage
        return response
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/products/{product_id}/compare-analysis")
def instrument_product_compare_analysis(
    product_id: str,
    request: ProductCompareAnalysisRequest,
    kind: Literal["etf", "fund"] = Query(default="etf"),
):
    """Load one real product series and delegate every comparison metric to NJIT."""

    points = _load_timeseries(kind, product_id)
    if not points:
        raise HTTPException(
            status_code=404,
            detail=f"未找到编号为 {product_id} 的产品真实行情数据",
        )
    try:
        return build_product_compare_response(
            product_id=product_id,
            points=points,
            parameters=request.model_dump(),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"产品比较计算失败：{exc}") from exc


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
        "qdii_type",
        "qdii_source",
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
        "execution": _instrument_execution_audit(),
    }

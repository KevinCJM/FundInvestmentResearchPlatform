"""Shared real-data series provider for product pages and indicator evaluation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Literal, Optional

import numpy as np
import pandas as pd
import pyarrow.dataset as arrow_dataset
import pyarrow.parquet as arrow_parquet

try:  # Package imports in tests; top-level imports when uvicorn starts in backend/.
    from backend.instrument_analytics_numba import simple_log_returns_kernel
    from backend.market_data import resolve_tushare_data_dir
    from backend.series_quality import (
        adjusted_nav_anomaly_dates,
        assess_period_window,
        finite_coverage,
        load_sse_open_dates,
        period_window_quality_kernel,
    )
except ModuleNotFoundError:  # pragma: no cover - exercised by integrated app startup
    from instrument_analytics_numba import simple_log_returns_kernel
    from market_data import resolve_tushare_data_dir
    from series_quality import (
        adjusted_nav_anomaly_dates,
        assess_period_window,
        finite_coverage,
        load_sse_open_dates,
        period_window_quality_kernel,
    )

from .errors import ValidationError
from .periods import get_period_spec, resolve_period_bounds
from .runtime_context import aligned_return_series_kernel
from .variable_registry import (
    DATA_CONTRACT_VERSION,
    canonicalize_variables,
    get_variable,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
@dataclass
class InstrumentIdentity:
    kind: Literal["etf", "fund"]
    product_id: str
    ts_code: str
    name: str
    found_date: str | None = None
    list_date: str | None = None


@dataclass
class ProductSeries:
    identity: InstrumentIdentity
    frame: pd.DataFrame
    fingerprint: str
    data_latest_date: str
    open_dates: pd.DatetimeIndex = field(default_factory=lambda: pd.DatetimeIndex([]))


@dataclass
class PeriodWindow:
    frame: pd.DataFrame
    returns: np.ndarray
    log_returns: np.ndarray
    requested_as_of: Optional[str]
    effective_as_of: str
    start_date: str
    end_date: str
    observation_count: int
    data_latest_date: str
    warnings: list[dict[str, str]] = field(default_factory=list)


@dataclass
class ProductVariableSeries:
    """Dependency-specific product frame with source provenance.

    ``frame`` contains only dates where every requested physical input exists.
    It is intentionally an inner join: missing prices, volumes, NAV disclosures,
    or announcement dates are never forward-filled or replaced with zero.
    """

    identity: InstrumentIdentity
    frame: pd.DataFrame
    fingerprints: dict[str, str]
    fingerprint: str
    data_latest_date: str | None
    requested_variables: tuple[str, ...]
    open_dates: pd.DatetimeIndex = field(default_factory=lambda: pd.DatetimeIndex([]))
    lineage: list[dict[str, Any]] = field(default_factory=list)
    coverage: dict[str, dict[str, Any]] = field(default_factory=dict)
    warnings: list[dict[str, str]] = field(default_factory=list)
    unavailable_variables: dict[str, dict[str, str]] = field(default_factory=dict)


@dataclass
class ProductChartSeries:
    """Axis-preserving product inputs for time-series indicator charts.

    The anchor variable owns the dates. Other variables are left-joined and
    keep missing values as NaN so chart positions are never compressed.
    """

    identity: InstrumentIdentity
    frame: pd.DataFrame
    axis_anchor: str
    requested_variables: tuple[str, ...]
    fingerprints: dict[str, str]
    fingerprint: str
    data_latest_date: str | None
    lineage: list[dict[str, Any]] = field(default_factory=list)
    coverage: dict[str, dict[str, Any]] = field(default_factory=dict)
    warnings: list[dict[str, str]] = field(default_factory=list)
    unavailable_variables: dict[str, dict[str, str]] = field(default_factory=dict)


@dataclass
class ChartPeriodWindow:
    compute_frame: pd.DataFrame
    display_start: int
    display_end: int
    requested_as_of: str | None
    effective_as_of: str
    start_date: str
    end_date: str
    observation_count: int
    data_latest_date: str | None
    warnings: list[dict[str, str]] = field(default_factory=list)

    @property
    def display_frame(self) -> pd.DataFrame:
        return self.compute_frame.iloc[self.display_start : self.display_end]


@dataclass
class VariablePeriodWindow(PeriodWindow):
    context: dict[str, Any] = field(default_factory=dict)
    fingerprints: dict[str, str] = field(default_factory=dict)
    lineage: list[dict[str, Any]] = field(default_factory=list)
    coverage: dict[str, dict[str, Any]] = field(default_factory=dict)
    common_date_hash: str = ""
    unavailable_variables: dict[str, dict[str, str]] = field(default_factory=dict)


@dataclass(frozen=True)
class VariableWindowIndex:
    """NumPy search index reused by every period for one product series."""

    date_days: np.ndarray
    anomaly_days: np.ndarray
    open_days: np.ndarray


def _return_arrays(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    simple_returns, log_returns, status = simple_log_returns_kernel(
        np.ascontiguousarray(values, dtype=np.float64)
    )
    if status == 1:
        raise ValidationError("INSUFFICIENT_SAMPLE", "至少需要两个有效净值点。")
    if status == 2:
        raise ValidationError(
            "INVALID_NAV_SERIES",
            "净值序列必须全部为有限正数，不能计算收益率。",
        )
    return simple_returns, log_returns


def _effective_data_dir(data_dir: Path) -> Path:
    candidate = Path(data_dir).expanduser().resolve()
    if candidate == DEFAULT_DATA_DIR.resolve():
        return resolve_tushare_data_dir(DEFAULT_DATA_DIR)
    return candidate


def _file_fingerprint(path: Path) -> str:
    stat = path.stat()
    raw = f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def market_data_generation(data_dir: Path = DEFAULT_DATA_DIR) -> str:
    """Return a cheap, deterministic generation for active indicator inputs."""

    resolved = _effective_data_dir(data_dir)
    inventory: list[tuple[str, int, int]] = []
    for filename in (
        "etf_info_df.parquet",
        "fund_info_df.parquet",
        "etf_daily_df.parquet",
        "fund_nav_df.parquet",
        "etf_daily_candle_df.parquet",
        "index_daily_df.parquet",
        "trade_day_df.parquet",
    ):
        path = resolved / filename
        if path.exists():
            stat = path.stat()
            inventory.append((filename, int(stat.st_size), int(stat.st_mtime_ns)))
    payload = {"directory": str(resolved), "files": inventory}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:20]


def _match_code(frame: pd.DataFrame, identifier: str) -> pd.DataFrame:
    if "ts_code" not in frame.columns:
        return frame.iloc[0:0]
    token = identifier.strip().lower()
    token_no_suffix = token.split(".", 1)[0]
    codes = frame["ts_code"].astype(str).str.lower()
    matched = frame[codes == token]
    if matched.empty:
        matched = frame[codes.str.split(".").str[0] == token_no_suffix]
    return matched


@lru_cache(maxsize=16)
def _identity_rows(
    path_text: str,
    size: int,
    modified_ns: int,
) -> tuple[tuple[str, str, str, str | None, str | None], ...]:
    del size, modified_ns
    path = Path(path_text)
    if not path.exists():
        return ()
    columns = set(arrow_parquet.read_schema(path).names)
    requested = [name for name in ("ts_code", "code", "name", "found_date", "list_date") if name in columns]
    if "ts_code" not in requested:
        return ()
    frame = pd.read_parquet(path, columns=requested)
    rows = []
    for row in frame.itertuples(index=False):
        payload = row._asdict()
        ts_code = str(payload.get("ts_code") or "")
        if not ts_code:
            continue
        rows.append(
            (
                ts_code,
                str(payload.get("code") or ts_code.split(".", 1)[0]),
                str(payload.get("name") or ts_code),
                _identity_date(payload.get("found_date")),
                _identity_date(payload.get("list_date")),
            )
        )
    return tuple(rows)


def _identity_date(value: Any) -> str | None:
    """Normalize info-table dates without substituting listing for inception."""
    if value is None or pd.isna(value):
        return None
    text = str(value).removesuffix(".0")
    parsed = pd.to_datetime(text, format="%Y%m%d" if len(text) == 8 and text.isdigit() else None, errors="coerce")
    return None if pd.isna(parsed) else parsed.strftime("%Y-%m-%d")


def resolve_identities(
    kind: Literal["etf", "fund"],
    product_ids: Iterable[str],
    data_dir: Path = DEFAULT_DATA_DIR,
) -> dict[str, InstrumentIdentity]:
    """Resolve a target set with one cached info-table read."""

    resolved_dir = _effective_data_dir(data_dir)
    info_path = resolved_dir / (
        "etf_info_df.parquet" if kind == "etf" else "fund_info_df.parquet"
    )
    rows = ()
    if info_path.exists():
        stat = info_path.stat()
        rows = _identity_rows(
            str(info_path.resolve()), int(stat.st_size), int(stat.st_mtime_ns)
        )
    by_token = {}
    for ts_code, code, name, found_date, list_date in rows:
        for token in (ts_code, code, ts_code.split(".", 1)[0], name):
            by_token[token.strip().lower()] = (ts_code, name, found_date, list_date)
    output: dict[str, InstrumentIdentity] = {}
    for raw_id in product_ids:
        product_id = str(raw_id)
        match = by_token.get(product_id.strip().lower())
        ts_code, name, found_date, list_date = match if match is not None else (product_id, product_id, None, None)
        output[product_id] = InstrumentIdentity(kind, ts_code, ts_code, name, found_date, list_date)
    return output


def resolve_identity(
    kind: Literal["etf", "fund"],
    product_id: str,
    data_dir: Path = DEFAULT_DATA_DIR,
) -> InstrumentIdentity:
    return resolve_identities(kind, [product_id], data_dir)[product_id]


def _read_candidate(path: Path, ts_code: str, value_column: str) -> pd.DataFrame:
    try:
        frame = pd.read_parquet(path, filters=[("ts_code", "==", ts_code)])
    except Exception:
        frame = pd.read_parquet(path)
    if frame.empty or "ts_code" not in frame.columns:
        return frame.iloc[0:0]
    frame = _match_code(frame, ts_code)
    if frame.empty or "date" not in frame.columns or value_column not in frame.columns:
        return frame.iloc[0:0]
    optional_columns = [
        column
        for column in ("open", "high", "low", "vol", "unit_nav", "accum_nav")
        if column in frame.columns and column != value_column
    ]
    result = frame[["date", value_column, *optional_columns]].copy().rename(
        columns={value_column: "value", "vol": "volume"}
    )
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    result["value"] = pd.to_numeric(result["value"], errors="coerce")
    for column in ("open", "high", "low", "volume"):
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result.replace([np.inf, -np.inf], np.nan).dropna(subset=["date", "value"])
    result = result[result["value"] > 0]
    return result.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)


def load_product_series(
    kind: Literal["etf", "fund"],
    product_id: str,
    data_dir: Path = DEFAULT_DATA_DIR,
) -> Optional[ProductSeries]:
    data_dir = _effective_data_dir(data_dir)
    identity = resolve_identity(kind, product_id, data_dir)
    open_dates = load_sse_open_dates(data_dir / "trade_day_df.parquet")
    candidates = (
        [
            (data_dir / "etf_daily_candle_df.parquet", "close"),
            (data_dir / "etf_daily_df.parquet", "adj_nav"),
        ]
        if kind == "etf"
        else [(data_dir / "fund_nav_df.parquet", "adj_nav")]
    )
    for path, value_column in candidates:
        if not path.exists():
            continue
        frame = _read_candidate(path, identity.ts_code, value_column)
        if frame.empty and identity.ts_code != product_id:
            frame = _read_candidate(path, product_id, value_column)
        if frame.empty:
            continue
        resolved_id = identity.ts_code
        return ProductSeries(
            identity=InstrumentIdentity(kind, resolved_id, resolved_id, identity.name),
            frame=frame,
            fingerprint=_file_fingerprint(path),
            data_latest_date=frame.iloc[-1]["date"].strftime("%Y-%m-%d"),
            open_dates=open_dates,
        )
    return None


def load_adjusted_product_series(
    kind: Literal["etf", "fund"],
    product_id: str,
    data_dir: Path = DEFAULT_DATA_DIR,
) -> Optional[ProductSeries]:
    """Load the adjusted-NAV series used by typed-v2 portfolio research.

    The legacy single-product evaluator intentionally keeps its existing
    ``load_product_series`` precedence.  Typed-v2 portfolio calculations use a
    separate entry point so their data contract cannot silently fall back to
    exchange close prices.
    """

    data_dir = _effective_data_dir(data_dir)
    identity = resolve_identity(kind, product_id, data_dir)
    open_dates = load_sse_open_dates(data_dir / "trade_day_df.parquet")
    path = data_dir / ("etf_daily_df.parquet" if kind == "etf" else "fund_nav_df.parquet")
    if not path.exists():
        return None
    frame = _read_candidate(path, identity.ts_code, "adj_nav")
    if frame.empty and identity.ts_code != product_id:
        frame = _read_candidate(path, product_id, "adj_nav")
    if frame.empty:
        return None
    resolved_id = identity.ts_code
    return ProductSeries(
        identity=InstrumentIdentity(kind, resolved_id, resolved_id, identity.name),
        frame=frame,
        fingerprint=_file_fingerprint(path),
        data_latest_date=frame.iloc[-1]["date"].strftime("%Y-%m-%d"),
        open_dates=open_dates,
    )


def load_price_points(
    kind: Literal["etf", "fund"],
    product_id: str,
    data_dir: Path = DEFAULT_DATA_DIR,
    *,
    preserve_missing: bool = False,
) -> list[dict[str, object]]:
    product_series = load_product_series(kind, product_id, data_dir)
    if product_series is None:
        return []
    points: list[dict[str, object]] = []
    for _, row in product_series.frame.iterrows():
        value = float(row["value"])
        if preserve_missing:
            raw_open = row.get("open", np.nan)
            raw_high = row.get("high", np.nan)
            raw_low = row.get("low", np.nan)
            raw_volume = row.get("volume", np.nan)
            open_value = float(raw_open) if pd.notna(raw_open) else None
            high_value = float(raw_high) if pd.notna(raw_high) else None
            low_value = float(raw_low) if pd.notna(raw_low) else None
            volume_value = float(raw_volume) if pd.notna(raw_volume) else None
        else:
            open_value = float(row.get("open", value)) if pd.notna(row.get("open", value)) else value
            high_value = float(row.get("high", value)) if pd.notna(row.get("high", value)) else value
            low_value = float(row.get("low", value)) if pd.notna(row.get("low", value)) else value
            volume_value = float(row.get("volume", 0)) if pd.notna(row.get("volume", 0)) else 0.0
        points.append(
            {
                "date": row["date"].strftime("%Y-%m-%d"),
                "open": open_value,
                "high": high_value,
                "low": low_value,
                "close": value,
                "volume": volume_value,
            }
        )
    return points


def _parse_as_of(as_of: Optional[str]) -> Optional[pd.Timestamp]:
    if not as_of:
        return None
    try:
        parsed = pd.Timestamp(as_of)
    except (TypeError, ValueError) as exc:
        raise ValidationError("INVALID_AS_OF", "as_of 必须是有效日期。", field="as_of") from exc
    if pd.isna(parsed):
        raise ValidationError("INVALID_AS_OF", "as_of 必须是有效日期。", field="as_of")
    if parsed.tzinfo is not None:
        parsed = parsed.tz_localize(None)
    return parsed.normalize()


def select_period_window(
    product_series: ProductSeries,
    period: str,
    as_of: Optional[str] = None,
    max_observations: int = 5000,
) -> PeriodWindow:
    spec = get_period_spec(period)
    if spec is None:
        raise ValidationError("INVALID_PERIOD", "不支持的评价周期。", field="period")
    requested_date = _parse_as_of(as_of)
    frame = product_series.frame
    if requested_date is not None:
        eligible = frame[frame["date"] <= requested_date].copy()
    else:
        eligible = frame.copy()
    if eligible.empty:
        raise ValidationError("NO_DATA_AS_OF", "截止日期之前没有可用净值数据。", field="as_of")

    effective_reference = pd.Timestamp(eligible.iloc[-1]["date"]).normalize()
    reference_date = requested_date or pd.Timestamp.today().normalize()
    bounds = resolve_period_bounds(
        spec,
        effective_data_date=effective_reference,
        reference_date=reference_date,
        first_data_date=pd.Timestamp(eligible.iloc[0]["date"]),
    )
    period_eligible = eligible[eligible["date"] <= bounds.calendar_end].copy()
    if period_eligible.empty:
        raise ValidationError(
            "NO_DATA_FOR_PERIOD",
            f"{spec.label}没有可用净值数据。",
            field="period",
        )
    actual_end = pd.Timestamp(period_eligible.iloc[-1]["date"]).normalize()
    anomalies = adjusted_nav_anomaly_dates(period_eligible, value_column="value")
    quality = assess_period_window(
        period_eligible["date"],
        target_date=bounds.anchor_target,
        effective_date=bounds.calendar_end,
        open_dates=product_series.open_dates,
        anomaly_dates=anomalies,
    )
    if quality.reason in {"insufficient_span", "start_anchor_too_old"}:
        raise ValidationError(
            "INSUFFICIENT_SAMPLE",
            f"现有历史未完整覆盖 {period} 自然周期。",
        )
    if quality.reason in {"insufficient_density", "internal_gap", "insufficient_observations"}:
        coverage = (
            f"{quality.coverage_ratio * 100:.1f}%"
            if quality.coverage_ratio is not None
            else "不可计算"
        )
        raise ValidationError(
            "INCOMPLETE_PERIOD_COVERAGE",
            f"{period} 区间内净值覆盖不完整（覆盖率 {coverage}），指标不予计算。",
        )
    if quality.reason == "adjusted_nav_anomaly":
        raise ValidationError(
            "ADJUSTED_NAV_ANOMALY",
            f"{period} 区间内复权净值存在异常跳变，指标不予计算。",
        )
    if quality.anchor_date is None:
        raise ValidationError("INSUFFICIENT_SAMPLE", f"现有历史不足以覆盖 {period} 自然周期。")
    selected = period_eligible[period_eligible["date"] >= quality.anchor_date].copy()

    warnings: list[dict[str, str]] = []
    max_points = max_observations + 1
    if len(selected) > max_points:
        selected = selected.iloc[-max_points:].copy()
        warnings.append(
            {
                "code": "SERIES_TRUNCATED",
                "message": f"输入序列已截取最近 {max_observations} 个收益观察值。",
            }
        )
    values = np.ascontiguousarray(selected["value"].to_numpy(dtype=np.float64))
    returns, log_returns = _return_arrays(values)
    return PeriodWindow(
        frame=selected,
        returns=returns,
        log_returns=log_returns,
        requested_as_of=as_of,
        effective_as_of=actual_end.strftime("%Y-%m-%d"),
        start_date=selected.iloc[0]["date"].strftime("%Y-%m-%d"),
        end_date=selected.iloc[-1]["date"].strftime("%Y-%m-%d"),
        observation_count=int(returns.size),
        data_latest_date=product_series.data_latest_date,
        warnings=warnings,
    )


@lru_cache(maxsize=32)
def _parquet_columns_cached(
    path_text: str,
    size: int,
    modified_ns: int,
) -> frozenset[str]:
    del size, modified_ns
    try:
        return frozenset(arrow_parquet.read_schema(path_text).names)
    except Exception:
        try:
            return frozenset(pd.read_parquet(path_text).columns)
        except Exception:
            return frozenset()


def _parquet_columns(path: Path) -> set[str]:
    """Read a Parquet schema once per immutable file generation."""

    source = Path(path).expanduser().resolve()
    if not source.exists():
        return set()
    stat = source.stat()
    return set(
        _parquet_columns_cached(
            str(source), int(stat.st_size), int(stat.st_mtime_ns)
        )
    )


def _date_values(values: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(values):
        return pd.to_datetime(values, errors="coerce").dt.tz_localize(None)
    tokens = values.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    compact = tokens.str.fullmatch(r"\d{8}", na=False)
    parsed = pd.to_datetime(tokens, errors="coerce")
    if compact.any():
        parsed.loc[compact] = pd.to_datetime(tokens.loc[compact], format="%Y%m%d", errors="coerce")
    try:
        return parsed.dt.tz_localize(None)
    except TypeError:
        return parsed


def _source_path(kind: Literal["etf", "fund"], dataset: str, data_dir: Path) -> Path | None:
    if dataset == "nav":
        return data_dir / ("etf_daily_df.parquet" if kind == "etf" else "fund_nav_df.parquet")
    if dataset == "candle" and kind == "etf":
        return data_dir / "etf_daily_candle_df.parquet"
    if dataset == "share" and kind == "etf":
        return data_dir / "etf_share_size_df.parquet"
    return None


def _source_unavailable_detail(
    variable_id: str,
    kind: Literal["etf", "fund"],
) -> dict[str, str]:
    definition = get_variable(variable_id)
    label = definition.label if definition is not None else variable_id
    kind_label = "ETF" if kind == "etf" else "场外公募基金"
    return {
        "code": "SOURCE_UNAVAILABLE_FOR_PRODUCT",
        "message": f"{kind_label}没有可供计算“{label}”的真实数据源。",
    }


def _variable_label(variable_id: str) -> str:
    definition = get_variable(variable_id)
    return definition.label if definition is not None else variable_id


def _no_observations_detail(
    variable_id: str, identity: InstrumentIdentity, as_of: str | None,
    first_date: str | None, rows_before: int, rows_after_date: int,
    rows_after: int, availability_field: str,
) -> dict[str, str]:
    label = _variable_label(variable_id)
    # Inception is explanatory metadata, never an extra data filter: predecessor
    # history may legitimately predate the current fund contract.
    if as_of and rows_before and not rows_after_date and first_date and first_date > as_of:
        if identity.found_date and identity.found_date > as_of:
            return {
                "code": "PRODUCT_NOT_ESTABLISHED_AS_OF",
                "message": f"当前计算截止日为 {as_of}，早于产品成立日期 {identity.found_date}；本地数据从 {first_date} 开始，当时没有可用的“{label}”数据。",
            }
        return {
            "code": "NO_DATA_BEFORE_CUTOFF",
            "message": f"当前计算截止日为 {as_of}，但本地数据从 {first_date} 才开始，截止日之前没有“{label}”数据。数据起点不等于产品成立日期。",
        }
    if as_of and rows_after_date and not rows_after and availability_field == "ann_date":
        return {
            "code": "NO_DISCLOSURES_AS_OF",
            "message": f"已有截止 {as_of} 的净值记录，但公告日期晚于截止日或缺失，无法确认当时已披露；因此未用于计算“{label}”。",
        }
    return {
        "code": "VARIABLE_NO_OBSERVATIONS",
        "message": f"“{label}”在当前产品和截止日没有有效观察值；请检查该字段的缺失值和数据覆盖范围。",
    }


def input_date_context(source: Any, as_of: str | None) -> dict[str, Any] | None:
    """Explain the loaded inputs using existing provenance, without another scan."""
    if source is None:
        return None
    labels = {
        "etf_daily_df.parquet": "ETF 净值",
        "fund_nav_df.parquet": "基金净值",
        "etf_daily_candle_df.parquet": "ETF 行情",
        "etf_share_size_df.parquet": "ETF 份额与规模",
    }
    sources = []
    for item in getattr(source, "lineage", []):
        if "rows_before_as_of" not in item:
            continue
        sources.append({
            "label": labels.get(item.get("dataset"), "指标输入数据"),
            "first_date": item.get("dataset_first_date"),
            "latest_date": item.get("dataset_latest_date"),
            "rows_before_as_of": item.get("rows_before_as_of"),
            "rows_after_date_filter": item.get("rows_after_date_filter"),
            "rows_after_as_of": item.get("rows_after_as_of"),
            "uses_disclosure_date": item.get("availability_field") == "ann_date",
        })
    return {
        "found_date": source.identity.found_date,
        "list_date": source.identity.list_date,
        "as_of": as_of,
        "sources": sources,
    }


def _read_source_frame(
    *,
    identity: InstrumentIdentity,
    product_id: str,
    dataset: str,
    variable_ids: tuple[str, ...],
    data_dir: Path,
    as_of: Optional[str],
) -> tuple[
    pd.DataFrame,
    str | None,
    dict[str, dict[str, Any]],
    dict[str, dict[str, str]],
    list[dict[str, str]],
    dict[str, Any] | None,
    str | None,
]:
    """Load a single physical dataset and expose canonical columns."""

    path = _source_path(identity.kind, dataset, data_dir)
    coverage: dict[str, dict[str, Any]] = {}
    unavailable: dict[str, dict[str, str]] = {}
    warnings: list[dict[str, str]] = []
    if path is None:
        for variable_id in variable_ids:
            unavailable[variable_id] = _source_unavailable_detail(
                variable_id, identity.kind
            )
        return pd.DataFrame(), None, coverage, unavailable, warnings, None, None
    if not path.exists():
        for variable_id in variable_ids:
            unavailable[variable_id] = {
                "code": "SOURCE_DATASET_MISSING",
                "message": f"“{_variable_label(variable_id)}”所需的本地数据集不存在。",
            }
        return pd.DataFrame(), None, coverage, unavailable, warnings, None, None

    schema = _parquet_columns(path)
    date_field = "date" if "date" in schema else "nav_date" if "nav_date" in schema else None
    if date_field is None or "ts_code" not in schema:
        for variable_id in variable_ids:
            unavailable[variable_id] = {
                "code": "SOURCE_SCHEMA_MISMATCH",
                "message": f"变量 {variable_id} 的数据集缺少 ts_code 或日期字段。",
            }
        return pd.DataFrame(), _file_fingerprint(path), coverage, unavailable, warnings, None, None
    if dataset == "nav" and as_of is not None and "ann_date" not in schema:
        for variable_id in variable_ids:
            unavailable[variable_id] = {
                "code": "ANN_DATE_UNAVAILABLE",
                "message": (
                    f"变量 {variable_id} 的净值数据缺少 ann_date，"
                    "无法证明历史截止日当时已经披露。"
                ),
            }
        warnings.append(
            {
                "code": "ANN_DATE_UNAVAILABLE",
                "message": "历史 as-of 计算要求 ann_date；未使用 nav_date 代替公告时点。",
            }
        )
        return (
            pd.DataFrame(),
            _file_fingerprint(path),
            coverage,
            unavailable,
            warnings,
            {
                "dataset": path.name,
                "fingerprint": _file_fingerprint(path),
                "source_fields": [],
                "rows_before_as_of": 0,
                "rows_after_as_of": 0,
                "availability_filter": "ann_date <= as_of (required)",
            },
            None,
        )

    fields_by_variable = {
        variable_id: str(get_variable(variable_id).source_field)
        for variable_id in variable_ids
        if get_variable(variable_id) is not None and get_variable(variable_id).source_field
    }
    present_fields = {field_name for field_name in fields_by_variable.values() if field_name in schema}
    requested_columns = ["ts_code", date_field, *sorted(present_fields)]
    if dataset == "nav" and "ann_date" in schema:
        requested_columns.append("ann_date")
    try:
        raw = pd.read_parquet(
            path,
            columns=list(dict.fromkeys(requested_columns)),
            filters=[("ts_code", "==", identity.ts_code)],
        )
    except Exception:
        raw = pd.read_parquet(path, columns=list(dict.fromkeys(requested_columns)))
    matched = _match_code(raw, identity.ts_code)
    if matched.empty and identity.ts_code != product_id:
        matched = _match_code(raw, product_id)
    raw = matched.copy()
    fingerprint = _file_fingerprint(path)
    if raw.empty:
        for variable_id in variable_ids:
            unavailable[variable_id] = {
                "code": "PRODUCT_DATA_NOT_FOUND",
                "message": f"数据集未找到产品 {product_id} 的变量 {variable_id}。",
            }
        lineage = {
            "dataset": path.name,
            "fingerprint": fingerprint,
            "source_fields": sorted(present_fields),
            "rows_before_as_of": 0,
            "rows_after_as_of": 0,
            "availability_filter": "ann_date <= as_of" if dataset == "nav" else "date <= as_of",
        }
        return pd.DataFrame(), fingerprint, coverage, unavailable, warnings, lineage, None

    raw["date"] = _date_values(raw[date_field])
    raw = raw.dropna(subset=["date"])
    dataset_first = raw["date"].min().strftime("%Y-%m-%d") if not raw.empty else None
    dataset_latest = raw["date"].max().strftime("%Y-%m-%d") if not raw.empty else None
    rows_before = len(raw)
    cutoff = _parse_as_of(as_of)
    if cutoff is not None:
        raw = raw[raw["date"] <= cutoff].copy()
    rows_after_date = len(raw)
    availability_field = "date"
    if dataset == "nav":
        if "ann_date" in raw.columns:
            raw["available_date"] = _date_values(raw["ann_date"])
            availability_field = "ann_date"
            if cutoff is not None:
                raw = raw[raw["available_date"].notna() & (raw["available_date"] <= cutoff)].copy()
        elif cutoff is not None:
            warnings.append(
                {
                    "code": "ANN_DATE_UNAVAILABLE",
                    "message": "净值数据缺少 ann_date，按净值日期截断；该数据集不具备严格公告时点证明。",
                }
            )
    sort_columns = ["date"] + (["available_date"] if "available_date" in raw.columns else [])
    raw = raw.sort_values(sort_columns).drop_duplicates(subset=["date"], keep="last")

    output = raw[["date"]].copy()
    if "available_date" in raw.columns:
        output["_available_date_nav"] = raw["available_date"]
    for variable_id in variable_ids:
        definition = get_variable(variable_id)
        source_field = fields_by_variable.get(variable_id)
        if definition is None or source_field is None:
            unavailable[variable_id] = {
                "code": "UNKNOWN_VARIABLE",
                "message": f"未知变量 {variable_id}。",
            }
            continue
        if source_field not in schema:
            unavailable[variable_id] = {
                "code": "SOURCE_FIELD_MISSING",
                "message": f"数据集缺少“{_variable_label(variable_id)}”所需字段 {source_field}。",
            }
            coverage[variable_id] = {
                "source_rows": len(raw),
                "non_null_rows": 0,
                "coverage_ratio": 0.0,
                "first_date": None,
                "latest_date": None,
            }
            continue
        values = pd.to_numeric(raw[source_field], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if definition.transform == "pct_chg / 100":
            values = values / 100.0
        valid = values.notna()
        non_null_rows, coverage_ratio = finite_coverage(values)
        dates = raw.loc[valid, "date"]
        coverage[variable_id] = {
            "source_rows": int(len(raw)),
            "non_null_rows": non_null_rows,
            "coverage_ratio": round(coverage_ratio, 8),
            "first_date": dates.min().strftime("%Y-%m-%d") if not dates.empty else None,
            "latest_date": dates.max().strftime("%Y-%m-%d") if not dates.empty else None,
            "conditional": bool(definition.conditional),
        }
        if not valid.any():
            unavailable[variable_id] = _no_observations_detail(
                variable_id, identity, as_of, dataset_first, rows_before,
                rows_after_date, len(raw), availability_field,
            )
            continue
        output[variable_id] = values

    lineage = {
        "dataset": path.name,
        "fingerprint": fingerprint,
        "source_fields": sorted(present_fields),
        "rows_before_as_of": int(rows_before),
        "rows_after_date_filter": int(rows_after_date),
        "rows_after_as_of": int(len(raw)),
        "dataset_first_date": dataset_first,
        "dataset_latest_date": dataset_latest,
        "availability_field": availability_field,
        "as_of": as_of,
    }
    return output, fingerprint, coverage, unavailable, warnings, lineage, dataset_latest


def _scan_source_batch(
    *,
    kind: Literal["etf", "fund"],
    dataset: str,
    variable_ids: tuple[str, ...],
    identities: dict[str, InstrumentIdentity],
    data_dir: Path,
    as_of: Optional[str],
) -> tuple[
    dict[str, pd.DataFrame],
    str | None,
    dict[str, dict[str, dict[str, Any]]],
    dict[str, dict[str, dict[str, str]]],
    dict[str, list[dict[str, str]]],
    dict[str, dict[str, Any]],
    dict[str, str | None],
]:
    """Scan one physical Parquet dataset once for an entire target set."""

    path = _source_path(kind, dataset, data_dir)
    frames = {product_id: pd.DataFrame() for product_id in identities}
    coverage = {product_id: {} for product_id in identities}
    unavailable = {product_id: {} for product_id in identities}
    warnings = {product_id: [] for product_id in identities}
    lineage: dict[str, dict[str, Any]] = {}
    latest_dates: dict[str, str | None] = {product_id: None for product_id in identities}

    if path is None or not path.exists():
        code = "SOURCE_UNAVAILABLE_FOR_PRODUCT" if path is None else "SOURCE_DATASET_MISSING"
        for product_id in identities:
            for variable_id in variable_ids:
                unavailable[product_id][variable_id] = (
                    _source_unavailable_detail(variable_id, kind)
                    if path is None
                    else {
                        "code": code,
                        "message": f"“{_variable_label(variable_id)}”所需的本地数据集不存在。",
                    }
                )
        return frames, None, coverage, unavailable, warnings, lineage, latest_dates

    schema = _parquet_columns(path)
    fingerprint = _file_fingerprint(path)
    date_field = next((name for name in ("date", "nav_date", "trade_date") if name in schema), None)
    if date_field is None or "ts_code" not in schema:
        for product_id in identities:
            for variable_id in variable_ids:
                unavailable[product_id][variable_id] = {
                    "code": "SOURCE_SCHEMA_MISMATCH",
                    "message": f"变量 {variable_id} 的数据集缺少 ts_code 或日期字段。",
                }
        return frames, fingerprint, coverage, unavailable, warnings, lineage, latest_dates

    if dataset == "nav" and as_of is not None and "ann_date" not in schema:
        for product_id in identities:
            for variable_id in variable_ids:
                unavailable[product_id][variable_id] = {
                    "code": "ANN_DATE_UNAVAILABLE",
                    "message": (
                        f"变量 {variable_id} 的净值数据缺少 ann_date，"
                        "无法证明历史截止日当时已经披露。"
                    ),
                }
            warnings[product_id].append(
                {
                    "code": "ANN_DATE_UNAVAILABLE",
                    "message": "历史 as-of 计算要求 ann_date；未使用 nav_date 代替公告时点。",
                }
            )
        return frames, fingerprint, coverage, unavailable, warnings, lineage, latest_dates

    fields_by_variable = {
        variable_id: str(get_variable(variable_id).source_field)
        for variable_id in variable_ids
        if get_variable(variable_id) is not None and get_variable(variable_id).source_field
    }
    present_fields = {
        field_name for field_name in fields_by_variable.values() if field_name in schema
    }
    requested_columns = ["ts_code", date_field, *sorted(present_fields)]
    if dataset == "nav" and "ann_date" in schema:
        requested_columns.append("ann_date")
    requested_columns = list(dict.fromkeys(requested_columns))
    codes = sorted({identity.ts_code for identity in identities.values()})
    try:
        source = arrow_dataset.dataset(path, format="parquet")
        table = source.to_table(
            columns=requested_columns,
            filter=arrow_dataset.field("ts_code").isin(codes),
            use_threads=True,
        )
        raw = table.to_pandas(split_blocks=True, self_destruct=True)
    except Exception:
        raw = pd.read_parquet(path, columns=requested_columns)
        raw = raw[raw["ts_code"].astype(str).isin(codes)]

    if raw.empty:
        groups: dict[str, pd.DataFrame] = {}
    else:
        raw["_code_key"] = raw["ts_code"].astype(str).str.lower()
        groups = {
            str(code): frame.drop(columns=["_code_key"])
            for code, frame in raw.groupby("_code_key", sort=False)
        }
    cutoff = _parse_as_of(as_of)

    for product_id, identity in identities.items():
        product_raw = groups.get(identity.ts_code.lower(), pd.DataFrame()).copy()
        if product_raw.empty:
            for variable_id in variable_ids:
                unavailable[product_id][variable_id] = {
                    "code": "PRODUCT_DATA_NOT_FOUND",
                    "message": f"数据集未找到产品 {product_id} 的变量 {variable_id}。",
                }
            lineage[product_id] = {
                "dataset": path.name,
                "fingerprint": fingerprint,
                "source_fields": sorted(present_fields),
                "rows_before_as_of": 0,
                "rows_after_as_of": 0,
                "availability_filter": "ann_date <= as_of" if dataset == "nav" else "date <= as_of",
                "batch_scan": True,
            }
            continue

        product_raw["date"] = _date_values(product_raw[date_field])
        product_raw = product_raw.dropna(subset=["date"])
        dataset_first = product_raw["date"].min().strftime("%Y-%m-%d") if not product_raw.empty else None
        dataset_latest = (
            product_raw["date"].max().strftime("%Y-%m-%d")
            if not product_raw.empty
            else None
        )
        latest_dates[product_id] = dataset_latest
        rows_before = len(product_raw)
        if cutoff is not None:
            product_raw = product_raw[product_raw["date"] <= cutoff].copy()
        rows_after_date = len(product_raw)
        availability_field = "date"
        if dataset == "nav" and "ann_date" in product_raw.columns:
            product_raw["available_date"] = _date_values(product_raw["ann_date"])
            availability_field = "ann_date"
            if cutoff is not None:
                product_raw = product_raw[
                    product_raw["available_date"].notna()
                    & (product_raw["available_date"] <= cutoff)
                ].copy()
        sort_columns = ["date"] + (
            ["available_date"] if "available_date" in product_raw.columns else []
        )
        product_raw = product_raw.sort_values(sort_columns).drop_duplicates(
            subset=["date"], keep="last"
        )
        output = product_raw[["date"]].copy()
        if "available_date" in product_raw.columns:
            output["_available_date_nav"] = product_raw["available_date"]

        for variable_id in variable_ids:
            definition = get_variable(variable_id)
            source_field = fields_by_variable.get(variable_id)
            if definition is None or source_field is None:
                unavailable[product_id][variable_id] = {
                    "code": "UNKNOWN_VARIABLE",
                    "message": f"未知变量 {variable_id}。",
                }
                continue
            if source_field not in schema:
                unavailable[product_id][variable_id] = {
                    "code": "SOURCE_FIELD_MISSING",
                    "message": f"数据集缺少“{_variable_label(variable_id)}”所需字段 {source_field}。",
                }
                coverage[product_id][variable_id] = {
                    "source_rows": int(len(product_raw)),
                    "non_null_rows": 0,
                    "coverage_ratio": 0.0,
                    "first_date": None,
                    "latest_date": None,
                }
                continue
            values = pd.to_numeric(product_raw[source_field], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
            if definition.transform == "pct_chg / 100":
                values = values / 100.0
            valid = values.notna()
            non_null_rows, coverage_ratio = finite_coverage(values)
            dates = product_raw.loc[valid, "date"]
            coverage[product_id][variable_id] = {
                "source_rows": int(len(product_raw)),
                "non_null_rows": non_null_rows,
                "coverage_ratio": round(coverage_ratio, 8),
                "first_date": dates.min().strftime("%Y-%m-%d") if not dates.empty else None,
                "latest_date": dates.max().strftime("%Y-%m-%d") if not dates.empty else None,
                "conditional": bool(definition.conditional),
            }
            if not valid.any():
                unavailable[product_id][variable_id] = _no_observations_detail(
                    variable_id, identity, as_of, dataset_first, rows_before,
                    rows_after_date, len(product_raw), availability_field,
                )
                continue
            output[variable_id] = values.to_numpy(copy=False)

        frames[product_id] = output
        lineage[product_id] = {
            "dataset": path.name,
            "fingerprint": fingerprint,
            "source_fields": sorted(present_fields),
            "rows_before_as_of": int(rows_before),
            "rows_after_date_filter": int(rows_after_date),
            "rows_after_as_of": int(len(product_raw)),
            "dataset_first_date": dataset_first,
            "dataset_latest_date": dataset_latest,
            "availability_field": availability_field,
            "as_of": as_of,
            "batch_scan": True,
        }

    return frames, fingerprint, coverage, unavailable, warnings, lineage, latest_dates


def load_product_variable_series_batch(
    kind: Literal["etf", "fund"],
    product_ids: Iterable[str],
    dependencies: Iterable[str],
    data_dir: Path = DEFAULT_DATA_DIR,
    as_of: Optional[str] = None,
) -> dict[str, ProductVariableSeries]:
    """Load one dependency signature for many products with one scan per file."""

    resolved_dir = _effective_data_dir(data_dir)
    ordered_ids = tuple(dict.fromkeys(str(item) for item in product_ids))
    identities = resolve_identities(kind, ordered_ids, resolved_dir)
    open_dates = load_sse_open_dates(resolved_dir / "trade_day_df.parquet")
    requested = canonicalize_variables(dependencies)
    physical = list(requested)
    if "adjusted_nav" not in physical:
        physical.append("adjusted_nav")

    base_unavailable: dict[str, dict[str, dict[str, str]]] = {
        product_id: {} for product_id in ordered_ids
    }
    by_dataset: dict[str, list[str]] = {}
    for variable_id in physical:
        definition = get_variable(variable_id)
        for product_id in ordered_ids:
            if definition is None:
                base_unavailable[product_id][variable_id] = {
                    "code": "UNKNOWN_VARIABLE",
                    "message": f"未知变量 {variable_id}。",
                }
            elif "single_product" not in definition.domains:
                base_unavailable[product_id][variable_id] = {
                    "code": "VARIABLE_CONTEXT_MISMATCH",
                    "message": f"变量 {variable_id} 不适用于单产品域。",
                }
            elif kind not in definition.product_kinds:
                base_unavailable[product_id][variable_id] = (
                    _source_unavailable_detail(variable_id, kind)
                )
        if (
            definition is not None
            and "single_product" in definition.domains
            and kind in definition.product_kinds
            and definition.source_dataset
        ):
            by_dataset.setdefault(definition.source_dataset, []).append(variable_id)

    source_frames: dict[str, list[pd.DataFrame]] = {product_id: [] for product_id in ordered_ids}
    fingerprints: dict[str, str] = {}
    coverages: dict[str, dict[str, Any]] = {product_id: {} for product_id in ordered_ids}
    warnings: dict[str, list[dict[str, str]]] = {product_id: [] for product_id in ordered_ids}
    lineages: dict[str, list[dict[str, Any]]] = {product_id: [] for product_id in ordered_ids}
    latest_dates: dict[str, list[str]] = {product_id: [] for product_id in ordered_ids}

    for dataset, variable_ids in by_dataset.items():
        (
            frames,
            fingerprint,
            source_coverage,
            source_unavailable,
            source_warnings,
            source_lineage,
            source_latest,
        ) = _scan_source_batch(
            kind=kind,
            dataset=dataset,
            variable_ids=tuple(variable_ids),
            identities=identities,
            data_dir=resolved_dir,
            as_of=as_of,
        )
        if fingerprint:
            fingerprints[dataset] = fingerprint
        for product_id in ordered_ids:
            coverages[product_id].update(source_coverage[product_id])
            base_unavailable[product_id].update(source_unavailable[product_id])
            warnings[product_id].extend(source_warnings[product_id])
            if product_id in source_lineage:
                lineages[product_id].append(source_lineage[product_id])
            if source_latest[product_id]:
                latest_dates[product_id].append(str(source_latest[product_id]))
            frame = frames[product_id]
            available_columns = [name for name in variable_ids if name in frame.columns]
            technical_columns = [
                name for name in frame.columns if name.startswith("_available_date_")
            ]
            if available_columns:
                source_frames[product_id].append(
                    frame[["date", *technical_columns, *available_columns]].copy()
                )

    combined_payload = {
        "contract": DATA_CONTRACT_VERSION,
        "files": fingerprints,
        "variables": requested,
    }
    combined = hashlib.sha256(
        json.dumps(combined_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:20]
    output: dict[str, ProductVariableSeries] = {}
    for product_id in ordered_ids:
        frames = source_frames[product_id]
        if not frames or "adjusted_nav" in base_unavailable[product_id]:
            merged = pd.DataFrame()
        else:
            merged = frames[0]
            for frame in frames[1:]:
                merged = merged.merge(frame, on="date", how="inner", validate="one_to_one")
            required_physical = [name for name in physical if name in merged.columns]
            merged = merged.replace([np.inf, -np.inf], np.nan).dropna(
                subset=required_physical
            )
            merged = merged[merged["adjusted_nav"] > 0]
            merged = (
                merged.sort_values("date")
                .drop_duplicates(subset=["date"], keep="last")
                .reset_index(drop=True)
            )
        for variable_id in requested:
            item = coverages[product_id].setdefault(variable_id, {})
            item["common_rows"] = (
                int(len(merged)) if variable_id in merged.columns else 0
            )
            item["common_latest_date"] = (
                merged.iloc[-1]["date"].strftime("%Y-%m-%d")
                if variable_id in merged.columns and not merged.empty
                else None
            )
        output[product_id] = ProductVariableSeries(
            identity=identities[product_id],
            frame=merged,
            fingerprints=dict(fingerprints),
            fingerprint=combined if not merged.empty else "missing",
            data_latest_date=(
                min(latest_dates[product_id]) if latest_dates[product_id] else None
            ),
            requested_variables=requested,
            open_dates=open_dates,
            lineage=lineages[product_id],
            coverage=coverages[product_id],
            warnings=warnings[product_id],
            unavailable_variables=base_unavailable[product_id],
        )
    return output


def load_product_variable_series(
    kind: Literal["etf", "fund"],
    product_id: str,
    dependencies: Iterable[str],
    data_dir: Path = DEFAULT_DATA_DIR,
    as_of: Optional[str] = None,
) -> ProductVariableSeries | None:
    """Load only the physical variables required by one typed expression."""

    data_dir = _effective_data_dir(data_dir)
    identity = resolve_identity(kind, product_id, data_dir)
    open_dates = load_sse_open_dates(data_dir / "trade_day_df.parquet")
    requested = canonicalize_variables(dependencies)
    physical = list(requested)
    # Every single-product window is anchored to adjusted NAV.  This keeps one
    # stable period contract even when the formula itself only uses quote data.
    if "adjusted_nav" not in physical:
        physical.append("adjusted_nav")

    by_dataset: dict[str, list[str]] = {}
    unavailable: dict[str, dict[str, str]] = {}
    for variable_id in physical:
        definition = get_variable(variable_id)
        if definition is None:
            unavailable[variable_id] = {
                "code": "UNKNOWN_VARIABLE",
                "message": f"未知变量 {variable_id}。",
            }
            continue
        if "single_product" not in definition.domains:
            unavailable[variable_id] = {
                "code": "VARIABLE_CONTEXT_MISMATCH",
                "message": f"变量 {variable_id} 不适用于单产品域。",
            }
            continue
        if kind not in definition.product_kinds:
            unavailable[variable_id] = _source_unavailable_detail(variable_id, kind)
            continue
        if definition.source_dataset:
            by_dataset.setdefault(definition.source_dataset, []).append(variable_id)

    frames: list[pd.DataFrame] = []
    fingerprints: dict[str, str] = {}
    coverage: dict[str, dict[str, Any]] = {}
    warnings: list[dict[str, str]] = []
    lineage: list[dict[str, Any]] = []
    latest_dates: list[str] = []
    for dataset, variable_ids in by_dataset.items():
        (
            frame,
            fingerprint,
            source_coverage,
            source_unavailable,
            source_warnings,
            source_lineage,
            dataset_latest,
        ) = _read_source_frame(
            identity=identity,
            product_id=product_id,
            dataset=dataset,
            variable_ids=tuple(variable_ids),
            data_dir=data_dir,
            as_of=as_of,
        )
        if fingerprint:
            fingerprints[dataset] = fingerprint
        coverage.update(source_coverage)
        unavailable.update(source_unavailable)
        warnings.extend(source_warnings)
        if source_lineage:
            lineage.append(source_lineage)
        if dataset_latest:
            latest_dates.append(dataset_latest)
        available_columns = [name for name in variable_ids if name in frame.columns]
        technical_columns = [name for name in frame.columns if name.startswith("_available_date_")]
        if available_columns:
            frames.append(frame[["date", *technical_columns, *available_columns]].copy())

    if not frames or "adjusted_nav" in unavailable:
        return ProductVariableSeries(
            identity=identity,
            frame=pd.DataFrame(),
            fingerprints=fingerprints,
            fingerprint="missing",
            data_latest_date=min(latest_dates) if latest_dates else None,
            requested_variables=requested,
            open_dates=open_dates,
            lineage=lineage,
            coverage=coverage,
            warnings=warnings,
            unavailable_variables=unavailable,
        )

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="date", how="inner", validate="one_to_one")
    required_physical = [name for name in physical if name in merged.columns]
    merged = merged.replace([np.inf, -np.inf], np.nan).dropna(subset=required_physical)
    merged = merged[merged["adjusted_nav"] > 0]
    merged = merged.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    for variable_id in requested:
        item = coverage.setdefault(variable_id, {})
        item["common_rows"] = int(len(merged)) if variable_id in merged.columns else 0
        item["common_latest_date"] = (
            merged.iloc[-1]["date"].strftime("%Y-%m-%d")
            if variable_id in merged.columns and not merged.empty
            else None
        )
    combined_payload = {
        "contract": DATA_CONTRACT_VERSION,
        "files": fingerprints,
        "variables": requested,
    }
    combined = hashlib.sha256(
        json.dumps(combined_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:20]
    return ProductVariableSeries(
        identity=identity,
        frame=merged,
        fingerprints=fingerprints,
        fingerprint=combined,
        data_latest_date=min(latest_dates) if latest_dates else None,
        requested_variables=requested,
        open_dates=open_dates,
        lineage=lineage,
        coverage=coverage,
        warnings=warnings,
        unavailable_variables=unavailable,
    )


def _window_source_variables(
    product_series: ProductVariableSeries,
    selected: pd.DataFrame,
    context: dict[str, Any],
) -> None:
    """Map physical levels and the aligned date axis after fixing the window."""
    dates = selected["date"].to_numpy(dtype="datetime64[D]")
    if np.isnat(dates).any() or (dates.size > 1 and not np.all(dates[1:] > dates[:-1])):
        raise ValidationError("INVALID_DATE_AXIS", "净值日期必须唯一、有效且严格递增。")
    context["observation_dates"] = np.ascontiguousarray(dates.astype(np.float64))
    for variable_id in product_series.requested_variables:
        if variable_id in {"returns", "log_returns", "observation_count", "window_elapsed_days"}:
            continue
        if variable_id not in selected.columns:
            continue
        values = np.ascontiguousarray(selected[variable_id].to_numpy(dtype=np.float64, copy=False))
        context[variable_id] = values


def select_variable_window(
    product_series: ProductVariableSeries,
    period: str,
    as_of: Optional[str] = None,
    max_observations: int = 5000,
) -> VariablePeriodWindow:
    unavailable = {
        name: detail
        for name, detail in product_series.unavailable_variables.items()
        if name in product_series.requested_variables or name == "adjusted_nav"
    }
    if unavailable:
        first_name, detail = next(iter(unavailable.items()))
        raise ValidationError(
            detail.get("code") or "VARIABLE_UNAVAILABLE",
            detail.get("message") or f"变量 {first_name} 不可用。",
            field="expression",
            diagnostics=[{"variable": name, **item} for name, item in unavailable.items()],
        )
    if product_series.frame.empty:
        raise ValidationError("DATA_NOT_FOUND", "没有满足所需变量共同日期的数据。")
    eligible_frame = product_series.frame
    cutoff = _parse_as_of(as_of)
    if cutoff is not None and "_available_date_nav" in eligible_frame.columns:
        eligible_frame = eligible_frame[
            eligible_frame["_available_date_nav"].notna()
            & (eligible_frame["_available_date_nav"] <= cutoff)
        ].copy()
    anchor = ProductSeries(
        identity=product_series.identity,
        frame=eligible_frame.rename(columns={"adjusted_nav": "value"}),
        fingerprint=product_series.fingerprint,
        data_latest_date=product_series.data_latest_date or product_series.frame.iloc[-1]["date"].strftime("%Y-%m-%d"),
        open_dates=product_series.open_dates,
    )
    base = select_period_window(anchor, period, as_of, max_observations)
    selected = base.frame.rename(columns={"value": "adjusted_nav"})
    elapsed_days = int((selected.iloc[-1]["date"] - selected.iloc[0]["date"]).days)
    context: dict[str, Any] = {
        "returns": base.returns,
        "log_returns": base.log_returns,
        "observation_count": float(base.observation_count),
        "window_elapsed_days": float(elapsed_days),
    }
    _window_source_variables(product_series, selected, context)
    copy_coverage = {
        name: dict(details) for name, details in product_series.coverage.items()
    }
    for variable_id in product_series.requested_variables:
        details = copy_coverage.setdefault(variable_id, {})
        details["window_rows"] = (
            finite_coverage(selected[variable_id])[0]
            if variable_id in selected.columns
            else base.observation_count
        )
        if isinstance(context.get(variable_id), np.ndarray):
            details["window_rows"] = int(context[variable_id].size)
        details["window_start_date"] = base.start_date
        details["window_end_date"] = base.end_date
    date_tokens = "|".join(selected["date"].dt.strftime("%Y-%m-%d").tolist())
    common_date_hash = hashlib.sha256(date_tokens.encode("utf-8")).hexdigest()[:20]
    return VariablePeriodWindow(
        frame=selected,
        returns=base.returns,
        log_returns=base.log_returns,
        requested_as_of=base.requested_as_of,
        effective_as_of=base.effective_as_of,
        start_date=base.start_date,
        end_date=base.end_date,
        observation_count=base.observation_count,
        data_latest_date=base.data_latest_date,
        warnings=[*product_series.warnings, *base.warnings],
        context=context,
        fingerprints=dict(product_series.fingerprints),
        lineage=[dict(item) for item in product_series.lineage],
        coverage=copy_coverage,
        common_date_hash=common_date_hash,
        unavailable_variables=dict(product_series.unavailable_variables),
    )


def prepare_variable_window_index(
    product_series: ProductVariableSeries,
) -> VariableWindowIndex:
    """Precompute date and anomaly arrays once for all requested periods."""

    if product_series.frame.empty:
        empty = np.empty(0, dtype=np.int64)
        return VariableWindowIndex(empty, empty, empty)
    dates = pd.DatetimeIndex(product_series.frame["date"])
    date_days = np.ascontiguousarray(dates.asi8 // 86_400_000_000_000, dtype=np.int64)
    anomaly_frame = product_series.frame.rename(columns={"adjusted_nav": "value"})
    anomaly_dates = adjusted_nav_anomaly_dates(anomaly_frame, value_column="value")
    anomaly_days = np.ascontiguousarray(
        anomaly_dates.asi8 // 86_400_000_000_000, dtype=np.int64
    )
    open_days = np.ascontiguousarray(
        product_series.open_dates.asi8 // 86_400_000_000_000, dtype=np.int64
    )
    return VariableWindowIndex(date_days, anomaly_days, open_days)


def select_variable_window_fast(
    product_series: ProductVariableSeries,
    period: str,
    as_of: Optional[str] = None,
    max_observations: int = 5000,
    *,
    index: VariableWindowIndex | None = None,
) -> VariablePeriodWindow:
    """Select a period with reusable NumPy indices and zero-copy source views."""

    unavailable = {
        name: detail
        for name, detail in product_series.unavailable_variables.items()
        if name in product_series.requested_variables or name == "adjusted_nav"
    }
    if unavailable:
        first_name, detail = next(iter(unavailable.items()))
        raise ValidationError(
            detail.get("code") or "VARIABLE_UNAVAILABLE",
            detail.get("message") or f"变量 {first_name} 不可用。",
            field="expression",
            diagnostics=[{"variable": name, **item} for name, item in unavailable.items()],
        )
    if product_series.frame.empty:
        raise ValidationError("DATA_NOT_FOUND", "没有满足所需变量共同日期的数据。")
    spec = get_period_spec(period)
    if spec is None:
        raise ValidationError("INVALID_PERIOD", "不支持的评价周期。", field="period")

    prepared = index or prepare_variable_window_index(product_series)
    date_days = prepared.date_days
    requested_date = _parse_as_of(as_of)
    if requested_date is None:
        effective_position = date_days.size - 1
    else:
        requested_day = int(requested_date.value // 86_400_000_000_000)
        effective_position = int(np.searchsorted(date_days, requested_day, side="right") - 1)
    if effective_position < 0:
        raise ValidationError("NO_DATA_AS_OF", "截止日期之前没有可用净值数据。", field="as_of")
    effective_reference = pd.Timestamp(
        product_series.frame.iloc[effective_position]["date"]
    ).normalize()
    reference_date = requested_date or pd.Timestamp.today().normalize()
    bounds = resolve_period_bounds(
        spec,
        effective_data_date=effective_reference,
        reference_date=reference_date,
        first_data_date=pd.Timestamp(product_series.frame.iloc[0]["date"]),
    )
    calendar_end_day = int(bounds.calendar_end.value // 86_400_000_000_000)
    effective_position = int(
        np.searchsorted(date_days, calendar_end_day, side="right") - 1
    )
    if effective_position < 0:
        raise ValidationError(
            "NO_DATA_FOR_PERIOD",
            f"{spec.label}没有可用净值数据。",
            field="period",
        )
    effective_date = pd.Timestamp(
        product_series.frame.iloc[effective_position]["date"]
    ).normalize()
    boundary_day = int(bounds.anchor_target.value // 86_400_000_000_000)
    anchor_position = int(
        np.searchsorted(
            date_days[: effective_position + 1], boundary_day, side="right"
        )
        - 1
    )
    if anchor_position < 0:
        raise ValidationError("INSUFFICIENT_SAMPLE", f"现有历史不足以覆盖 {period} 自然周期。")
    anchor_day = int(date_days[anchor_position])
    if spec.kind != "lifetime" and boundary_day - anchor_day > 10:
        raise ValidationError("INSUFFICIENT_SAMPLE", f"现有历史未完整覆盖 {period} 自然周期。")

    open_days = prepared.open_days
    has_full_calendar = bool(
        open_days.size
        and open_days[0] <= boundary_day
        and open_days[-1] >= calendar_end_day
    )
    if has_full_calendar:
        expected_start = int(np.searchsorted(open_days, boundary_day, side="left"))
        expected_end = int(
            np.searchsorted(open_days, calendar_end_day, side="right")
        )
        expected = open_days[expected_start:expected_end]
        required_coverage = 0.90
        max_missing_allowed = 5
    else:
        expected_dates = pd.bdate_range(bounds.anchor_target, bounds.calendar_end)
        expected = np.ascontiguousarray(
            expected_dates.asi8 // 86_400_000_000_000, dtype=np.int64
        )
        required_coverage = 0.80
        max_missing_allowed = 10
    (
        quality_complete,
        quality_reason,
        _quality_anchor,
        _quality_observations,
        _quality_expected,
        coverage_value,
        _quality_max_missing,
        _quality_anomalies,
    ) = period_window_quality_kernel(
        np.ascontiguousarray(date_days[: effective_position + 1]),
        boundary_day,
        int(date_days[effective_position]),
        np.ascontiguousarray(expected),
        np.ascontiguousarray(prepared.anomaly_days),
        10,
        required_coverage,
        max_missing_allowed,
    )
    coverage_ratio = (
        None if not np.isfinite(coverage_value) else float(coverage_value)
    )
    if int(quality_reason) == 3:
        raise ValidationError(
            "ADJUSTED_NAV_ANOMALY",
            f"{period} 区间内复权净值存在异常跳变，指标不予计算。",
        )
    if not bool(quality_complete):
        if int(quality_reason) in {1, 2}:
            raise ValidationError(
                "INSUFFICIENT_SAMPLE",
                f"现有历史未完整覆盖 {period} 自然周期。",
            )
        if coverage_ratio is None:
            message = f"{period} 区间内净值覆盖不完整，指标不予计算。"
        else:
            message = (
                f"{period} 区间内净值覆盖不完整"
                f"（覆盖率 {coverage_ratio * 100:.1f}%），指标不予计算。"
            )
        raise ValidationError(
            "INCOMPLETE_PERIOD_COVERAGE",
            message,
        )

    warnings: list[dict[str, str]] = []
    start_position = anchor_position
    max_points = max_observations + 1
    if effective_position - start_position + 1 > max_points:
        start_position = effective_position - max_points + 1
        warnings.append(
            {
                "code": "SERIES_TRUNCATED",
                "message": f"输入序列已截取最近 {max_observations} 个收益观察值。",
            }
        )
    selected = product_series.frame.iloc[start_position : effective_position + 1]
    adjusted_nav = selected["adjusted_nav"].to_numpy(dtype=np.float64, copy=False)
    returns, log_returns = _return_arrays(adjusted_nav)
    elapsed_days = int((selected.iloc[-1]["date"] - selected.iloc[0]["date"]).days)
    context: dict[str, Any] = {
        "returns": returns,
        "log_returns": log_returns,
        "observation_count": float(returns.size),
        "window_elapsed_days": float(elapsed_days),
    }
    _window_source_variables(product_series, selected, context)
    copy_coverage = {
        name: dict(details) for name, details in product_series.coverage.items()
    }
    start_date = selected.iloc[0]["date"].strftime("%Y-%m-%d")
    end_date = selected.iloc[-1]["date"].strftime("%Y-%m-%d")
    for variable_id in product_series.requested_variables:
        details = copy_coverage.setdefault(variable_id, {})
        details["window_rows"] = (
            int(finite_coverage(selected[variable_id])[0])
            if variable_id in selected.columns
            else int(returns.size)
        )
        if isinstance(context.get(variable_id), np.ndarray):
            details["window_rows"] = int(context[variable_id].size)
        details["window_start_date"] = start_date
        details["window_end_date"] = end_date
    selected_day_view = prepared.date_days[start_position : effective_position + 1]
    common_date_hash = hashlib.sha256(selected_day_view.tobytes()).hexdigest()[:20]
    return VariablePeriodWindow(
        frame=selected,
        returns=returns,
        log_returns=log_returns,
        requested_as_of=as_of,
        effective_as_of=effective_date.strftime("%Y-%m-%d"),
        start_date=start_date,
        end_date=end_date,
        observation_count=int(returns.size),
        data_latest_date=(
            product_series.data_latest_date
            or product_series.frame.iloc[-1]["date"].strftime("%Y-%m-%d")
        ),
        warnings=[*product_series.warnings, *warnings],
        context=context,
        fingerprints=dict(product_series.fingerprints),
        lineage=[dict(item) for item in product_series.lineage],
        coverage=copy_coverage,
        common_date_hash=common_date_hash,
        unavailable_variables=dict(product_series.unavailable_variables),
    )


def load_product_chart_series(
    kind: Literal["etf", "fund"],
    product_id: str,
    dependencies: Iterable[str],
    axis_anchor: str,
    data_dir: Path = DEFAULT_DATA_DIR,
    as_of: Optional[str] = None,
) -> ProductChartSeries:
    """Load an axis-preserving frame, including derived return series.

    ``returns`` and ``log_returns`` are public time-series variables but are not
    physical Parquet columns.  They are derived from the same-date adjusted NAV
    path after the backing dataset is loaded.  The first row remains NaN so the
    derived arrays stay aligned with the chart date axis.
    """

    resolved_dir = _effective_data_dir(data_dir)
    identity = resolve_identity(kind, product_id, resolved_dir)
    requested = tuple(
        dict.fromkeys((*canonicalize_variables(dependencies), str(axis_anchor)))
    )
    derived_backing = {
        "returns": "adjusted_nav",
        "log_returns": "adjusted_nav",
    }
    physical_requested: list[str] = []
    unavailable: dict[str, dict[str, str]] = {}
    for variable_id in requested:
        definition = get_variable(variable_id)
        if definition is None:
            unavailable[variable_id] = {
                "code": "UNKNOWN_VARIABLE",
                "message": f"未知变量 {variable_id}。",
            }
            continue
        if definition.kind != "series" or "single_product" not in definition.domains:
            unavailable[variable_id] = {
                "code": "VARIABLE_CONTEXT_MISMATCH",
                "message": f"变量 {variable_id} 不是单产品时间序列。",
            }
            continue
        backing = derived_backing.get(variable_id, variable_id)
        if backing not in physical_requested:
            physical_requested.append(backing)

    by_dataset: dict[str, list[str]] = {}
    for variable_id in physical_requested:
        definition = get_variable(variable_id)
        if definition is None:
            unavailable[variable_id] = {
                "code": "UNKNOWN_VARIABLE",
                "message": f"未知变量 {variable_id}。",
            }
            continue
        if kind not in definition.product_kinds or not definition.source_dataset:
            unavailable[variable_id] = _source_unavailable_detail(variable_id, kind)
            continue
        by_dataset.setdefault(definition.source_dataset, []).append(variable_id)

    source_frames: dict[str, pd.DataFrame] = {}
    fingerprints: dict[str, str] = {}
    coverage: dict[str, dict[str, Any]] = {}
    warnings: list[dict[str, str]] = []
    lineage: list[dict[str, Any]] = []
    latest_dates: dict[str, str | None] = {}
    for dataset, variable_ids in by_dataset.items():
        (
            frame,
            fingerprint,
            source_coverage,
            source_unavailable,
            source_warnings,
            source_lineage,
            dataset_latest,
        ) = _read_source_frame(
            identity=identity,
            product_id=product_id,
            dataset=dataset,
            variable_ids=tuple(variable_ids),
            data_dir=resolved_dir,
            as_of=as_of,
        )
        source_frames[dataset] = frame
        if fingerprint:
            fingerprints[dataset] = fingerprint
        coverage.update(source_coverage)
        unavailable.update(source_unavailable)
        warnings.extend(source_warnings)
        if source_lineage:
            lineage.append(source_lineage)
        latest_dates[dataset] = dataset_latest

    anchor_physical = derived_backing.get(str(axis_anchor), str(axis_anchor))
    anchor_definition = get_variable(anchor_physical)
    anchor_dataset = (
        anchor_definition.source_dataset if anchor_definition is not None else None
    )
    anchor_frame = source_frames.get(str(anchor_dataset), pd.DataFrame())
    if (
        anchor_physical in unavailable
        or anchor_frame.empty
        or anchor_physical not in anchor_frame.columns
    ):
        merged = pd.DataFrame()
    else:
        anchor_columns = [
            name
            for name in anchor_frame.columns
            if name == "date"
            or name.startswith("_available_date_")
            or name in physical_requested
        ]
        merged = anchor_frame[anchor_columns].copy()
        anchor_values = pd.to_numeric(
            merged[anchor_physical], errors="coerce"
        ).replace([np.inf, -np.inf], np.nan)
        merged[anchor_physical] = anchor_values
        # The date axis belongs to observations, not their numerical validity.
        # Dropping an invalid anchor would bridge data gaps in both rolling
        # windows and derived returns. Keep NaN positions for the NJIT contract.
        merged = merged.dropna(subset=["date"])
        for dataset, frame in source_frames.items():
            if dataset == anchor_dataset or frame.empty:
                continue
            join_columns = [
                name
                for name in frame.columns
                if name == "date"
                or name.startswith("_available_date_")
                or name in physical_requested
            ]
            if len(join_columns) <= 1:
                continue
            merged = merged.merge(
                frame[join_columns],
                on="date",
                how="left",
                validate="one_to_one",
            )
        merged = (
            merged.sort_values("date")
            .drop_duplicates(subset=["date"], keep="last")
            .reset_index(drop=True)
        )
        for variable_id in physical_requested:
            if variable_id not in merged.columns:
                merged[variable_id] = np.nan
            else:
                merged[variable_id] = pd.to_numeric(
                    merged[variable_id], errors="coerce"
                ).replace([np.inf, -np.inf], np.nan)

        if any(name in requested for name in derived_backing):
            levels = np.ascontiguousarray(
                merged["adjusted_nav"].to_numpy(dtype=np.float64, copy=False)
            )
            simple_returns, log_returns = aligned_return_series_kernel(levels)
            if "returns" in requested:
                merged["returns"] = simple_returns
            if "log_returns" in requested:
                merged["log_returns"] = log_returns
            for variable_id, values in (
                ("returns", simple_returns),
                ("log_returns", log_returns),
            ):
                if variable_id not in requested:
                    continue
                finite = np.isfinite(values)
                dates = merged.loc[finite, "date"]
                coverage[variable_id] = {
                    "source_rows": int(values.size),
                    "non_null_rows": int(np.count_nonzero(finite)),
                    "coverage_ratio": round(
                        float(np.count_nonzero(finite) / values.size)
                        if values.size
                        else 0.0,
                        8,
                    ),
                    "first_date": (
                        dates.min().strftime("%Y-%m-%d")
                        if not dates.empty
                        else None
                    ),
                    "latest_date": (
                        dates.max().strftime("%Y-%m-%d")
                        if not dates.empty
                        else None
                    ),
                    "derived_from": "adjusted_nav",
                }
            lineage.append(
                {
                    "dataset": "derived_context",
                    "source_fields": ["adjusted_nav"],
                    "derived_fields": [
                        name for name in ("returns", "log_returns") if name in requested
                    ],
                    "transform": "adj_nav[t] / adj_nav[t-1] - 1; log(adj_nav[t] / adj_nav[t-1])",
                    "alignment": "same date axis; first observation is missing",
                }
            )

        # A return anchor shares its backing NAV's dates, including the first
        # undefined return. Never drop that baseline or internal missing rows.

    for derived_name, backing_name in derived_backing.items():
        if derived_name in requested and backing_name in unavailable:
            backing_detail = unavailable[backing_name]
            unavailable[derived_name] = {
                "code": backing_detail.get("code", "VARIABLE_UNAVAILABLE"),
                "message": (
                    f"“{_variable_label(derived_name)}”需要可用的"
                    f"“{_variable_label(backing_name)}”：{backing_detail.get('message', '')}"
                ),
            }

    fingerprint_payload = {
        "contract": DATA_CONTRACT_VERSION,
        "axis_anchor": axis_anchor,
        "variables": requested,
        "derived_backing": {
            name: backing
            for name, backing in derived_backing.items()
            if name in requested
        },
        "files": fingerprints,
    }
    combined = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:20]
    if merged.empty:
        combined = "missing"
    anchor_latest = latest_dates.get(str(anchor_dataset))
    return ProductChartSeries(
        identity=identity,
        frame=merged,
        axis_anchor=str(axis_anchor),
        requested_variables=requested,
        fingerprints=fingerprints,
        fingerprint=combined,
        data_latest_date=anchor_latest,
        lineage=lineage,
        coverage=coverage,
        warnings=warnings,
        unavailable_variables=unavailable,
    )


def select_chart_window(
    product_series: ProductChartSeries,
    period: str,
    as_of: Optional[str] = None,
    *,
    history_policy: Literal["lookback", "full_history"] = "lookback",
    lookback_observations: int = 1,
    max_display_points: int = 5_000,
    max_compute_points: int = 20_000,
) -> ChartPeriodWindow:
    """Separate the visible period from the causal computation history."""

    spec = get_period_spec(period)
    if spec is None:
        raise ValidationError("INVALID_PERIOD", "不支持的评价周期。", field="period")
    if history_policy not in {"lookback", "full_history"}:
        raise ValidationError(
            "INVALID_HISTORY_POLICY",
            "时序历史策略无效。",
            field="history_policy",
        )
    if lookback_observations < 1:
        raise ValidationError(
            "INVALID_LOOKBACK",
            "时序回看观察数必须为正整数。",
            field="lookback_observations",
        )
    frame = product_series.frame
    requested_date = _parse_as_of(as_of)
    eligible = (
        frame[frame["date"] <= requested_date].copy()
        if requested_date is not None
        else frame.copy()
    )
    if eligible.empty:
        raise ValidationError(
            "NO_DATA_AS_OF",
            "截止日期之前没有可用的时序指标日期轴。",
            field="as_of",
        )
    effective = pd.Timestamp(eligible.iloc[-1]["date"]).normalize()
    reference = requested_date or pd.Timestamp.today().normalize()
    bounds = resolve_period_bounds(
        spec,
        effective_data_date=effective,
        reference_date=reference,
        first_data_date=pd.Timestamp(eligible.iloc[0]["date"]),
    )
    bounded = eligible[eligible["date"] <= bounds.calendar_end].copy()
    if bounded.empty:
        raise ValidationError(
            "NO_DATA_FOR_PERIOD",
            f"{spec.label}没有可用时序数据。",
            field="period",
        )
    display_floor = bounds.anchor_target
    if spec.kind == "calendar":
        display_floor = display_floor + pd.Timedelta(days=1)
    display_positions = np.flatnonzero(
        (bounded["date"] >= display_floor).to_numpy(dtype=np.bool_)
    )
    if display_positions.size == 0:
        raise ValidationError(
            "NO_DATA_FOR_PERIOD",
            f"{spec.label}没有可展示的时序数据。",
            field="period",
        )
    first_display = int(display_positions[0])
    last_display = int(display_positions[-1]) + 1
    warnings = list(product_series.warnings)
    visible_count = last_display - first_display
    if visible_count > max_display_points:
        first_display = last_display - max_display_points
        warnings.append(
            {
                "code": "SERIES_DISPLAY_TRUNCATED",
                "message": f"时序展示已保留最近 {max_display_points} 个观察值。",
            }
        )
    compute_start = (
        0
        if history_policy == "full_history"
        else max(0, first_display - max(0, lookback_observations - 1))
    )
    compute_end = last_display
    if compute_end - compute_start > max_compute_points:
        if history_policy == "full_history":
            raise ValidationError(
                "SERIES_HISTORY_LIMIT_EXCEEDED",
                f"递归时序指标历史超过 {max_compute_points} 个观察值，不能截断后重置状态。",
                field="period",
            )
        compute_start = compute_end - max_compute_points
        warnings.append(
            {
                "code": "SERIES_COMPUTE_TRUNCATED",
                "message": f"时序计算已保留最近 {max_compute_points} 个观察值。",
            }
        )
    compute_frame = bounded.iloc[compute_start:compute_end].reset_index(drop=True)
    display_start = first_display - compute_start
    display_end = last_display - compute_start
    display = compute_frame.iloc[display_start:display_end]
    return ChartPeriodWindow(
        compute_frame=compute_frame,
        display_start=display_start,
        display_end=display_end,
        requested_as_of=as_of,
        effective_as_of=effective.strftime("%Y-%m-%d"),
        start_date=display.iloc[0]["date"].strftime("%Y-%m-%d"),
        end_date=display.iloc[-1]["date"].strftime("%Y-%m-%d"),
        observation_count=int(len(display)),
        data_latest_date=product_series.data_latest_date,
        warnings=warnings,
    )


__all__ = [
    "DEFAULT_DATA_DIR",
    "InstrumentIdentity",
    "PeriodWindow",
    "ProductSeries",
    "ProductVariableSeries",
    "ProductChartSeries",
    "ChartPeriodWindow",
    "VariableWindowIndex",
    "VariablePeriodWindow",
    "load_adjusted_product_series",
    "load_price_points",
    "load_product_series",
    "load_product_chart_series",
    "load_product_variable_series",
    "load_product_variable_series_batch",
    "market_data_generation",
    "resolve_identity",
    "resolve_identities",
    "select_period_window",
    "select_chart_window",
    "select_variable_window",
    "prepare_variable_window_index",
    "select_variable_window_fast",
]

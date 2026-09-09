"""Point-in-time data resolution for historical regime identification."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import pyarrow.dataset as arrow_dataset

try:
    from backend.instrument_analytics_numba import count_true_kernel
    from backend.market_data import resolve_tushare_data_dir
    from backend.historical_regimes.numba_kernels import relative_transform_kernel
except ModuleNotFoundError:  # pragma: no cover - backend/ direct execution
    from instrument_analytics_numba import count_true_kernel
    from market_data import resolve_tushare_data_dir
    from historical_regimes.numba_kernels import relative_transform_kernel

from custom_indicators.errors import NotFoundError, ValidationError
from research_series.product_sources import PRODUCT_SOURCES, ProductSourceError, product_pit, read_product_observations, apply_product_adjustment, ETF_ADJUSTED_FIELDS, product_source_spec


INDEX_HISTORY_FILES = {
    "index_daily": "index_daily_df.parquet",
    "sw_daily": "index_sw_daily_df.parquet",
    "ci_daily": "index_ci_daily_df.parquet",
    "ths_daily": "index_ths_daily_df.parquet",
    "dc_daily": "index_dc_daily_df.parquet",
    "tdx_daily": "index_tdx_daily_df.parquet",
    "index_global": "index_global_daily_df.parquet",
    "fut_index_daily": "index_futures_daily_df.parquet",
    "index_dailybasic": "index_daily_basic_df.parquet",
}


@dataclass(frozen=True)
class DataBundle:
    frame: pd.DataFrame
    snapshot: dict[str, Any]


def _parse_date(value: Any, field: str) -> pd.Timestamp:
    try:
        result = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError("INVALID_DATE", f"{field} 必须是有效日期。", field) from exc
    if pd.isna(result):
        raise ValidationError("INVALID_DATE", f"{field} 必须是有效日期。", field)
    if result.tzinfo is not None:
        result = result.tz_localize(None)
    return result.normalize()


def _hash_frame(frame: pd.DataFrame) -> str:
    canonical = frame.copy()
    for column in canonical.select_dtypes(include=["datetime", "datetimetz"]).columns:
        canonical[column] = canonical[column].astype(str)
    digest = hashlib.sha256()
    digest.update(pd.util.hash_pandas_object(canonical, index=True).values.tobytes())
    return digest.hexdigest()


def _normalise_observations(
    raw: pd.DataFrame,
    mode: str,
    as_of: Optional[str],
    *,
    value_field: str = "value",
    availability_mode: Optional[str] = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if raw.empty:
        raise ValidationError("EMPTY_SERIES", "数据源没有可用于情景识别的观测。", "target")
    frame = raw.copy()
    if "observation_date" not in frame.columns:
        if "date" in frame.columns:
            frame["observation_date"] = frame["date"]
        elif "trade_date" in frame.columns:
            frame["observation_date"] = frame["trade_date"]
        else:
            raise ValidationError("MISSING_OBSERVATION_DATE", "每条数据必须包含 observation_date 或 date。", "target")

    parsed_dates = pd.to_datetime(frame["observation_date"], errors="coerce")
    if parsed_dates.isna().any():
        raise ValidationError("INVALID_OBSERVATION_DATE", "观测日期中存在无效值。", "target.observation_date")
    if getattr(parsed_dates.dt, "tz", None) is not None:
        parsed_dates = parsed_dates.dt.tz_localize(None)
    frame["observation_date"] = parsed_dates.dt.normalize()

    if "available_at" not in frame.columns:
        frame["available_at"] = frame["observation_date"]
    parsed_available = pd.to_datetime(frame["available_at"], errors="coerce")
    if parsed_available.isna().any():
        raise ValidationError("INVALID_AVAILABLE_AT", "数据可得日期中存在无效值。", "target.available_at")
    if getattr(parsed_available.dt, "tz", None) is not None:
        parsed_available = parsed_available.dt.tz_localize(None)
    frame["available_at"] = parsed_available.dt.normalize()
    if (frame["available_at"] < frame["observation_date"]).any():
        raise ValidationError(
            "AVAILABLE_BEFORE_OBSERVATION",
            "available_at 不能早于 observation_date。",
            "target.available_at",
        )

    if as_of:
        cutoff = _parse_date(as_of, "as_of")
        frame = frame.loc[frame["available_at"] <= cutoff].copy()
    if frame.empty:
        raise ValidationError("NO_DATA_AS_OF", "截至所选日期尚无可得数据。", "as_of")

    if value_field in frame.columns and value_field != "value":
        frame["value"] = frame[value_field]
    elif "value" not in frame.columns:
        frame["value"] = np.nan
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame["revision"] = frame.get("revision", 1)
    frame["vintage"] = frame.get("vintage", None)
    frame["_row_id"] = np.arange(len(frame))
    revision_numeric = pd.to_numeric(frame["revision"], errors="coerce")
    frame["_revision_order"] = revision_numeric.where(revision_numeric.notna(), 0)
    frame = frame.sort_values(["observation_date", "available_at", "_revision_order", "_row_id"])

    revision_counts = frame.groupby("observation_date").size()
    final_row_ids = set(frame.groupby("observation_date").tail(1)["_row_id"].tolist())
    revision_policy = availability_mode or ("point_in_time" if mode == "realtime" else "latest")
    if revision_policy not in {"point_in_time", "latest"}:
        raise ValidationError("INVALID_AVAILABILITY_MODE", "availability_mode 必须是 point_in_time 或 latest。", "target.availability_mode")
    if mode == "realtime" and revision_policy == "latest":
        raise ValidationError(
            "LATEST_VINTAGE_REALTIME_REQUEST",
            "实时模式不能使用最终修订值；请选择 point_in_time 或改用 retrospective。",
            "target.availability_mode",
        )
    if revision_policy == "point_in_time":
        selected = frame.groupby("observation_date", as_index=False, sort=False).head(1).copy()
        selected_revision_policy = "first_release"
    else:
        selected = frame.groupby("observation_date", as_index=False, sort=False).tail(1).copy()
        selected_revision_policy = "latest_vintage"
    selected["is_final"] = selected["_row_id"].isin(final_row_ids)
    selected = selected.sort_values("observation_date").reset_index(drop=True)

    metadata = {
        "raw_observations": int(len(frame)),
        "selected_observations": int(len(selected)),
        "revision_observations": int(
            count_true_kernel(
                np.ascontiguousarray((revision_counts > 1).to_numpy(dtype=np.uint8))
            )
        ),
        "revision_policy": selected_revision_policy,
        "first_observation_date": selected["observation_date"].iloc[0].date().isoformat(),
        "last_observation_date": selected["observation_date"].iloc[-1].date().isoformat(),
        "latest_available_at": selected["available_at"].max().date().isoformat(),
    }
    return selected.drop(columns=["_row_id", "_revision_order"]), metadata


def _inline_bundle(spec: dict[str, Any], mode: str, as_of: Optional[str]) -> DataBundle:
    rows = spec.get("rows")
    if rows is None:
        rows = spec.get("points")
    if rows is None:
        rows = spec.get("series")
    if not isinstance(rows, list) or not rows:
        raise ValidationError("EMPTY_INLINE_DATA", "内联数据至少需要一条观测。", "target.rows")
    if len(rows) > 20000:
        raise ValidationError("INLINE_DATA_TOO_LARGE", "内联数据不能超过 20000 条。", "target.rows")
    if not all(isinstance(row, dict) for row in rows):
        raise ValidationError("INVALID_INLINE_ROW", "内联数据的每条观测必须是对象。", "target.rows")
    frame, revision_meta = _normalise_observations(
        pd.DataFrame(rows),
        mode,
        as_of,
        value_field=str(spec.get("value_field") or "value"),
        availability_mode=str(spec["availability_mode"]) if spec.get("availability_mode") else None,
    )
    if spec.get("start_date"):
        frame = frame.loc[frame["observation_date"] >= _parse_date(spec["start_date"], "target.start_date")].copy()
    if spec.get("end_date"):
        frame = frame.loc[frame["observation_date"] <= _parse_date(spec["end_date"], "target.end_date")].copy()
    if frame.empty:
        raise ValidationError("EMPTY_DATE_RANGE", "所选日期区间没有内联数据。", "target")
    revision_meta.update(
        {
            "selected_observations": int(len(frame)),
            "first_observation_date": frame["observation_date"].iloc[0].date().isoformat(),
            "last_observation_date": frame["observation_date"].iloc[-1].date().isoformat(),
            "latest_available_at": frame["available_at"].max().date().isoformat(),
        }
    )
    snapshot = {
        "kind": "inline",
        "fingerprint": _hash_frame(frame),
        **revision_meta,
    }
    return DataBundle(frame=frame, snapshot=snapshot)


def _index_bundle(
    spec: dict[str, Any],
    mode: str,
    as_of: Optional[str],
    market_data_dir: Path,
) -> DataBundle:
    source_api = str(spec.get("source_api") or "index_daily")
    filename = INDEX_HISTORY_FILES.get(source_api)
    if filename is None:
        raise ValidationError("UNSUPPORTED_INDEX_SOURCE", "不支持的指数数据源。", "target.source_api")
    ts_code = str(spec.get("ts_code") or spec.get("code") or "").strip()
    if not ts_code:
        raise ValidationError("MISSING_INDEX_CODE", "指数数据源必须配置 ts_code。", "target.ts_code")
    root = resolve_tushare_data_dir(market_data_dir)
    path = root / filename
    if not path.exists():
        raise NotFoundError("INDEX_DATA_NOT_FOUND", f"指数数据文件 {filename} 不存在。")

    dataset = arrow_dataset.dataset(path, format="parquet")
    names = set(dataset.schema.names)
    code_column = next((name for name in ("ts_code", "index_code", "code") if name in names), None)
    date_column = next((name for name in ("trade_date", "observation_date", "date") if name in names), None)
    value_column = str(spec.get("field") or "close")
    if code_column is None or date_column is None or value_column not in names:
        raise ValidationError("INDEX_SCHEMA_MISMATCH", "指数文件缺少代码、日期或所选数值列。", "target")
    optional = [name for name in ("available_at", "revision", "vintage") if name in names]
    columns = list(dict.fromkeys([code_column, date_column, value_column, *optional]))
    # Projection and predicate pushdown are mandatory here: never materialise a full market table.
    table = dataset.to_table(columns=columns, filter=arrow_dataset.field(code_column) == ts_code)
    raw = table.to_pandas()
    if raw.empty:
        raise NotFoundError("INDEX_SERIES_NOT_FOUND", "未找到所选指数的历史序列。")
    raw = raw.rename(columns={date_column: "observation_date"})
    raw["value"] = raw[value_column]
    frame, revision_meta = _normalise_observations(
        raw,
        mode,
        as_of,
        availability_mode=str(spec["availability_mode"]) if spec.get("availability_mode") else None,
    )
    start_date = spec.get("start_date")
    end_date = spec.get("end_date")
    if start_date:
        frame = frame.loc[frame["observation_date"] >= _parse_date(start_date, "target.start_date")].copy()
    if end_date:
        frame = frame.loc[frame["observation_date"] <= _parse_date(end_date, "target.end_date")].copy()
    if frame.empty:
        raise ValidationError("EMPTY_DATE_RANGE", "所选日期区间没有指数数据。", "target")
    revision_meta.update(
        {
            "selected_observations": int(len(frame)),
            "first_observation_date": frame["observation_date"].iloc[0].date().isoformat(),
            "last_observation_date": frame["observation_date"].iloc[-1].date().isoformat(),
            "latest_available_at": frame["available_at"].max().date().isoformat(),
        }
    )
    stat = path.stat()
    fingerprint_payload = {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "source_api": source_api,
        "ts_code": ts_code,
        "field": value_column,
        "frame": _hash_frame(frame),
    }
    snapshot = {
        "kind": "index",
        "source_api": source_api,
        "ts_code": ts_code,
        "field": value_column,
        "file": filename,
        "projection": columns,
        "filters": {code_column: ts_code},
        "fingerprint": hashlib.sha256(json.dumps(fingerprint_payload, sort_keys=True).encode()).hexdigest(),
        **revision_meta,
    }
    return DataBundle(frame=frame, snapshot=snapshot)


def _product_bundle(spec: dict[str, Any], mode: str, as_of: Optional[str], market_data_dir: Path) -> DataBundle:
    kind = str(spec["kind"])
    root = resolve_tushare_data_dir(market_data_dir)
    try:
        raw = read_product_observations(root, kind, spec, mode)
    except ProductSourceError as exc:
        raise ValidationError(exc.code, exc.message, "target") from exc
    frame, revision_meta = _normalise_observations(
        raw, mode, as_of,
        availability_mode=str(spec["availability_mode"]) if spec.get("availability_mode") else None,
    )
    for key, lower in (("start_date", True), ("end_date", False)):
        if spec.get(key):
            bound = _parse_date(spec[key], f"target.{key}")
            frame = frame.loc[frame["observation_date"] >= bound if lower else frame["observation_date"] <= bound].copy()
    if frame.empty:
        raise ValidationError("EMPTY_DATE_RANGE", "所选日期区间没有产品行情。", "target")
    source = product_source_spec(kind, spec.get("field"))
    field = str(spec.get("field") or source["default_field"])
    try:
        frame = apply_product_adjustment(frame, kind, field)
    except ProductSourceError as exc:
        raise ValidationError(exc.code, exc.message, "target.field") from exc
    pit = product_pit(kind, field)
    if (kind == "etf" and field in ETF_ADJUSTED_FIELDS) or field == "adj_nav":
        pit.update(supported=False, availability_status="retrospective_adjustment")
    if frame["availability_unknown"].any():
        pit.update(supported=False, availability_status="unknown_retrospective_only")
    snapshot = {
        **revision_meta, "kind": kind, "ts_code": str(spec["ts_code"]),
        **({"adjustment": raw.attrs["adjustment"]} if raw.attrs.get("adjustment") else {}),
        "source_api": source["source_api"], "field": spec.get("field") or source["default_field"],
        "file": source["filename"], "fingerprint": _hash_frame(frame), "pit": pit,
        "selected_observations": len(frame),
        "first_observation_date": frame["observation_date"].iloc[0].date().isoformat(),
        "last_observation_date": frame["observation_date"].iloc[-1].date().isoformat(),
        "latest_available_at": frame["available_at"].max().date().isoformat(),
    }
    return DataBundle(frame=frame, snapshot=snapshot)


def _indicator_bundle(
    spec: dict[str, Any],
    mode: str,
    as_of: Optional[str],
    indicator_service: Any,
) -> DataBundle:
    if indicator_service is None:
        raise ValidationError(
            "INDICATOR_SERVICE_UNAVAILABLE",
            "指标中心当前不可用，无法解析版本化指标序列。",
            "target.kind",
        )
    result = indicator_service.evaluate_historical_series(
        indicator_id=str(spec["indicator_id"]),
        indicator_revision=int(spec["indicator_revision"]),
        product_kind=str(spec["product_kind"]),
        product_id=str(spec["product_id"]),
        period=str(spec["period"]),
        as_of=as_of,
        start_date=str(spec["start_date"]) if spec.get("start_date") else None,
        end_date=str(spec["end_date"]) if spec.get("end_date") else None,
        max_points=5000,
    )
    points = result.get("series")
    if not isinstance(points, list) or not points:
        raise ValidationError(
            "EMPTY_INDICATOR_SERIES",
            "指标中心没有返回可用于历史识别的逐期序列。",
            "target",
        )
    frame, revision_meta = _normalise_observations(
        pd.DataFrame(points),
        mode,
        as_of,
        availability_mode="point_in_time",
    )
    snapshot = dict(result.get("snapshot") or {})
    snapshot.update(revision_meta)
    snapshot["fingerprint"] = str(
        snapshot.get("series_hash") or _hash_frame(frame)
    )
    snapshot["calculation_audit"] = dict(result.get("audit") or {})
    return DataBundle(frame=frame, snapshot=snapshot)


def resolve_target(
    spec: dict[str, Any],
    mode: str,
    as_of: Optional[str],
    market_data_dir: Path,
    indicator_service: Any = None,
) -> DataBundle:
    """Resolve point-in-time inline, index, indicator or relative data."""

    kind = str(spec.get("kind") or "")
    if kind == "inline":
        return _inline_bundle(spec, mode, as_of)
    if kind == "index":
        return _index_bundle(spec, mode, as_of, market_data_dir)
    if kind in PRODUCT_SOURCES:
        return _product_bundle(spec, mode, as_of, market_data_dir)
    if kind == "indicator":
        return _indicator_bundle(spec, mode, as_of, indicator_service)
    if kind != "relative":
        raise ValidationError("INVALID_DATA_SOURCE", "不支持的数据源类型。", "target.kind")

    numerator = spec.get("numerator")
    denominator = spec.get("denominator")
    if not isinstance(numerator, dict) or not isinstance(denominator, dict):
        raise ValidationError("INVALID_RELATIVE_SOURCE", "相对强弱需要 numerator 与 denominator。", "target")
    availability_mode = spec.get("availability_mode")
    numerator_spec = {**numerator, **({"availability_mode": availability_mode} if availability_mode and "availability_mode" not in numerator else {})}
    denominator_spec = {**denominator, **({"availability_mode": availability_mode} if availability_mode and "availability_mode" not in denominator else {})}
    numerator_bundle = resolve_target(
        numerator_spec, mode, as_of, market_data_dir, indicator_service
    )
    denominator_bundle = resolve_target(
        denominator_spec, mode, as_of, market_data_dir, indicator_service
    )
    left = numerator_bundle.frame[["observation_date", "available_at", "value", "is_final"]].rename(
        columns={"available_at": "available_at_num", "value": "numerator", "is_final": "final_num"}
    )
    right = denominator_bundle.frame[["observation_date", "available_at", "value", "is_final"]].rename(
        columns={"available_at": "available_at_den", "value": "denominator", "is_final": "final_den"}
    )
    frame = left.merge(right, on="observation_date", how="inner", validate="one_to_one")
    frame = frame.loc[(frame["numerator"] > 0) & (frame["denominator"] > 0)].copy()
    if frame.empty:
        raise ValidationError("NO_RELATIVE_OVERLAP", "两条序列没有可比较的正值交集。", "target")
    frame["available_at"] = frame[["available_at_num", "available_at_den"]].max(axis=1)
    transform = str(spec.get("transform") or "log_ratio")
    transform_code = {"ratio": 0, "log_ratio": 1}.get(transform)
    if transform_code is None:
        raise ValidationError("INVALID_RELATIVE_TRANSFORM", "相对序列仅支持 ratio 或 log_ratio。", "target.transform")
    frame["value"] = relative_transform_kernel(
        np.ascontiguousarray(frame["numerator"].to_numpy(dtype=np.float64)),
        np.ascontiguousarray(frame["denominator"].to_numpy(dtype=np.float64)),
        np.int64(transform_code),
    )
    frame["revision"] = 1
    frame["vintage"] = None
    frame["is_final"] = frame["final_num"] & frame["final_den"]
    frame = frame.sort_values("observation_date").reset_index(drop=True)
    if spec.get("start_date"):
        frame = frame.loc[frame["observation_date"] >= _parse_date(spec["start_date"], "target.start_date")].copy()
    if spec.get("end_date"):
        frame = frame.loc[frame["observation_date"] <= _parse_date(spec["end_date"], "target.end_date")].copy()
    if frame.empty:
        raise ValidationError("EMPTY_DATE_RANGE", "所选日期区间没有相对序列交集。", "target")
    snapshot = {
        "kind": "relative",
        "transform": transform,
        "alignment": "strict_intersection",
        "fingerprint": _hash_frame(frame),
        "selected_observations": int(len(frame)),
        "numerator": numerator_bundle.snapshot,
        "denominator": denominator_bundle.snapshot,
    }
    return DataBundle(frame=frame, snapshot=snapshot)

"""Read-only research-series catalog and full-sample profile service."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import threading
from typing import Any, Literal

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

try:
    from backend.market_data import MarketDataManifestError, read_active_manifest
except ModuleNotFoundError:  # pragma: no cover - backend/ direct execution
    from market_data import MarketDataManifestError, read_active_manifest

from .numba_kernels import (
    align_values_kernel,
    complete_case_indices_kernel,
    complement_rate_kernel,
    distribution_summary_kernel,
    finite_mask_kernel,
    infinite_count_kernel,
    index_profile_kernel,
    macro_profile_kernel,
    pair_valid_indices_kernel,
    pearson_matrix_kernel,
    release_lag_days_kernel,
    research_series_numba_execution_audit,
    sample_indices_kernel,
    standardize_matrix_kernel,
    strict_intersection_dates_kernel,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
CATALOG_SCHEMA_VERSION = "research-series-catalog-v1"
PROFILE_SCHEMA_VERSION = "research-series-profile-v1"
MAX_PROFILE_OBSERVATIONS = 200_000
MAX_INLINE_ROWS = 20_000
UPLOAD_ARTIFACT_DIRNAME = "research_series_uploads"
_UPLOAD_ARTIFACT_PATTERN = re.compile(r"^upload-sha256-([0-9a-f]{64})$")
_FILE_CHECKSUM_CACHE: dict[tuple[str, int, int], str] = {}
_FILE_CHECKSUM_LOCK = threading.RLock()

INDEX_SOURCE_FILES = {
    "index_daily": "index_daily_df.parquet",
    "sw_daily": "index_sw_daily_df.parquet",
    "ci_daily": "index_ci_daily_df.parquet",
    "ths_daily": "index_ths_daily_df.parquet",
    "dc_daily": "index_dc_daily_df.parquet",
    "tdx_daily": "index_tdx_daily_df.parquet",
    "index_global": "index_global_daily_df.parquet",
    "fut_index_daily": "index_futures_daily_df.parquet",
}

INDEX_PROFILE_OPERATIONS = [
    "raw",
    "normalized",
    "return",
    "cumulative_return",
    "drawdown",
    "distribution",
    "rolling_volatility",
]
MACRO_PROFILE_OPERATIONS = [
    "raw",
    "yoy",
    "mom",
    "quantile",
    "distribution",
    "release_lag",
    "vintage",
]
INLINE_PROFILE_OPERATIONS = list(
    dict.fromkeys([*INDEX_PROFILE_OPERATIONS, *MACRO_PROFILE_OPERATIONS])
)
GOVERNANCE_FIELDS = {
    "revision",
    "source_api",
    "ingested_at",
    "observation_date",
    "available_at",
    "availability_status",
    "vintage",
    "ts_code",
    "trade_date",
}

FIELD_LABELS = {
    # 指数行情与估值
    "open": "开盘点位",
    "high": "最高点位",
    "low": "最低点位",
    "close": "收盘点位",
    "pre_close": "前收盘点位",
    "avg_price": "平均成交价",
    "change": "涨跌点数",
    "pct_chg": "涨跌幅",
    "vol": "成交量",
    "amount": "成交额",
    "swing": "振幅",
    "turnover_rate": "换手率",
    "turnover_rate_f": "自由流通换手率",
    "pe": "市盈率",
    "pe_ttm": "滚动市盈率",
    "pb": "市净率",
    "total_mv": "总市值",
    "float_mv": "流通市值",
    "total_share": "总股本",
    "float_share": "流通股本",
    "free_share": "自由流通股本",
    # 国内宏观接口常用字段
    "nt_val": "本期值",
    "pre_val": "上期值",
    "accu_val": "累计值",
    "accu_pre": "上年同期累计值",
    "gdp": "国内生产总值",
    "gdp_yoy": "国内生产总值同比增速",
    "pi": "第一产业增加值",
    "si": "第二产业增加值",
    "ti": "第三产业增加值",
    "ppi_yoy": "工业生产者出厂价格同比",
    "ppi_mom": "工业生产者出厂价格环比",
    "cpi_yoy": "居民消费价格同比",
    "cpi_mom": "居民消费价格环比",
    "pmi": "制造业采购经理指数",
    "pmi010000": "制造业采购经理指数",
    "non_manu_pmi": "非制造业商务活动指数",
    "composite_pmi": "综合采购经理指数",
    "m0": "流通中现金",
    "m0_yoy": "流通中现金同比增速",
    "m1": "狭义货币供应量",
    "m1_yoy": "狭义货币供应量同比增速",
    "m2": "广义货币供应量",
    "m2_yoy": "广义货币供应量同比增速",
    "shibor_on": "隔夜 Shibor",
    "shibor_1w": "一周 Shibor",
    "shibor_1m": "一个月 Shibor",
    "shibor_3m": "三个月 Shibor",
    "lpr_1y": "一年期贷款市场报价利率",
    "lpr_5y": "五年期以上贷款市场报价利率",
}


@dataclass(frozen=True)
class MacroDatasetSpec:
    filename: str
    source_api: str
    name: str
    category: str
    frequency: str
    year_over_year_lag: int

    @property
    def stem(self) -> str:
        return self.filename.removesuffix(".parquet")


MACRO_DATASETS = (
    MacroDatasetSpec("macro_cn_gdp_df.parquet", "cn_gdp", "国内生产总值", "经济增长", "quarterly", 4),
    MacroDatasetSpec("macro_cn_cpi_df.parquet", "cn_cpi", "居民消费价格", "通胀", "monthly", 12),
    MacroDatasetSpec("macro_cn_ppi_df.parquet", "cn_ppi", "工业生产者价格", "通胀", "monthly", 12),
    MacroDatasetSpec("macro_cn_pmi_df.parquet", "cn_pmi", "采购经理指数", "经济景气", "monthly", 12),
    MacroDatasetSpec("macro_cn_money_df.parquet", "cn_m", "货币供应量", "货币信用", "monthly", 12),
    MacroDatasetSpec(
        "macro_cn_social_financing_df.parquet",
        "sf_month",
        "社会融资规模",
        "货币信用",
        "monthly",
        12,
    ),
    MacroDatasetSpec("macro_shibor_df.parquet", "shibor", "Shibor", "利率", "daily", 252),
    MacroDatasetSpec("macro_lpr_df.parquet", "shibor_lpr", "贷款市场报价利率", "利率", "monthly", 12),
    MacroDatasetSpec("macro_repo_daily_df.parquet", "repo_daily", "回购利率", "利率", "daily", 252),
    MacroDatasetSpec(
        "macro_cn_schedule_df.parquet",
        "cn_schedule",
        "宏观发布日历",
        "发布治理",
        "event",
        1,
    ),
)
MACRO_BY_STEM = {spec.stem: spec for spec in MACRO_DATASETS}


class ResearchSeriesError(ValueError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        status_code: int = 400,
        field: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.field = field

    def detail(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.field:
            payload["field"] = self.field
        return payload


def _text(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _date_text(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _file_checksum(path: Path) -> str:
    stat = path.stat()
    key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    with _FILE_CHECKSUM_LOCK:
        cached = _FILE_CHECKSUM_CACHE.get(key)
        if cached is not None:
            return cached
        digest = hashlib.sha256()
        with path.open("rb") as source:
            while True:
                block = source.read(8 * 1024 * 1024)
                if not block:
                    break
                digest.update(block)
        checksum = f"sha256:{digest.hexdigest()}"
        stale = [candidate for candidate in _FILE_CHECKSUM_CACHE if candidate[0] == key[0]]
        for candidate in stale:
            _FILE_CHECKSUM_CACHE.pop(candidate, None)
        _FILE_CHECKSUM_CACHE[key] = checksum
        return checksum


def _snapshot_identity(snapshot: Path, manifest: dict[str, object]) -> dict[str, str]:
    snapshot_id = str(manifest.get("snapshot_id") or snapshot.name)
    generation = str(manifest.get("generation") or snapshot.name)
    return {"snapshot_id": snapshot_id, "snapshot_generation": generation}


def _parse_optional_date(value: str | None, field: str) -> pd.Timestamp | None:
    if not value:
        return None
    try:
        parsed = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ResearchSeriesError("INVALID_DATE", f"{field} 必须是有效日期。", field=field) from exc
    if pd.isna(parsed):
        raise ResearchSeriesError("INVALID_DATE", f"{field} 必须是有效日期。", field=field)
    if parsed.tzinfo is not None:
        parsed = parsed.tz_localize(None)
    return parsed.normalize()


def _numeric_fields(path: Path) -> list[str]:
    schema = pq.ParquetFile(path).schema_arrow
    fields: list[str] = []
    for field in schema:
        if field.name in GOVERNANCE_FIELDS:
            continue
        if pa.types.is_integer(field.type) or pa.types.is_floating(field.type) or pa.types.is_decimal(field.type):
            fields.append(field.name)
    return fields


def _field_unit(field: str, kind: Literal["index", "macro"]) -> str:
    normalized = field.lower()
    if kind == "index":
        if normalized in {"open", "high", "low", "close", "pre_close", "avg_price", "change"}:
            return "index_point"
        if "pct" in normalized or normalized in {"swing", "turnover_rate", "turnover_rate_f"}:
            return "percent"
        if normalized.endswith("_num") or normalized in {"lu_days"}:
            return "count"
        if normalized.startswith("pe") or normalized == "pb" or normalized.endswith("ratio"):
            return "ratio"
    return "source_unit"


def _field_descriptor(field: str, kind: Literal["index", "macro"], missing: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "name": field,
        "label": FIELD_LABELS.get(field.lower(), "数值字段"),
        "unit": _field_unit(field, kind),
        "dtype": "float64",
        "nullable": True,
        **({"missing": missing} if missing is not None else {}),
    }


def _nullable_scalar(value: Any) -> float | None:
    try:
        array = np.ascontiguousarray(np.array([float(value)], dtype=np.float64))
    except (TypeError, ValueError):
        return None
    mask = finite_mask_kernel(array)
    return float(array[0]) if mask[0] == 1 else None


def _nullable_values(values: np.ndarray, indices: np.ndarray) -> list[float | None]:
    mask = finite_mask_kernel(values)
    return [float(values[index]) if mask[index] == 1 else None for index in indices]


_DISTRIBUTION_KEYS = (
    "valid_count",
    "missing_count",
    "missing_rate",
    "mean",
    "std",
    "min",
    "p05",
    "p25",
    "median",
    "p75",
    "p95",
    "max",
)


def _distribution_payload(values: np.ndarray) -> dict[str, Any]:
    summary = distribution_summary_kernel(values)
    mask = finite_mask_kernel(summary)
    payload: dict[str, Any] = {}
    for index, key in enumerate(_DISTRIBUTION_KEYS):
        if mask[index] == 0:
            payload[key] = None
        elif key in {"valid_count", "missing_count"}:
            payload[key] = int(summary[index])
        else:
            payload[key] = float(summary[index])
    return payload


def _datetime_days(values: pd.Series) -> np.ndarray:
    timestamps = pd.to_datetime(values, errors="coerce").to_numpy(dtype="datetime64[ns]")
    return np.ascontiguousarray(timestamps.astype("datetime64[D]").astype(np.int64))


def _inline_periods(frequency: str) -> tuple[int, float]:
    normalized = frequency.strip().lower()
    mapping = {
        "daily": (252, 252.0),
        "weekly": (52, 52.0),
        "monthly": (12, 12.0),
        "quarterly": (4, 4.0),
        "annual": (1, 1.0),
        "irregular": (1, 1.0),
    }
    if normalized not in mapping:
        raise ResearchSeriesError(
            "INLINE_FREQUENCY_INVALID",
            "frequency 仅支持 daily、weekly、monthly、quarterly、annual 或 irregular。",
            field="frequency",
        )
    return mapping[normalized]


def _upload_artifact_digest(frame: pd.DataFrame) -> str:
    values = np.ascontiguousarray(frame["value"].to_numpy(dtype=np.float64))
    mask = finite_mask_kernel(values)
    rows: list[dict[str, Any]] = []
    for index in range(len(frame)):
        raw_vintage = frame["vintage"].iloc[index]
        rows.append(
            {
                "observation_date": _date_text(frame["observation_date"].iloc[index]),
                "available_at": _date_text(frame["available_at"].iloc[index]),
                "value": float(values[index]) if mask[index] == 1 else None,
                "vintage": (
                    None
                    if raw_vintage is None or pd.isna(raw_vintage)
                    else str(raw_vintage)
                ),
                "revision": int(frame["revision"].iloc[index]),
            }
        )
    material = json.dumps(
        rows,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def write_upload_artifact(data_dir: Path, frame: pd.DataFrame) -> dict[str, Any]:
    """Atomically register a validated, content-addressed upload artifact."""

    required = {
        "observation_date",
        "available_at",
        "value",
        "vintage",
        "revision",
    }
    if not required.issubset(frame.columns) or frame.empty:
        raise ResearchSeriesError(
            "UPLOAD_ARTIFACT_SCHEMA_INVALID",
            "上传数据无法生成版本化数据源。",
        )
    if len(frame) > MAX_INLINE_ROWS:
        raise ResearchSeriesError(
            "INLINE_ROWS_LIMIT_EXCEEDED",
            f"内联分析最多允许 {MAX_INLINE_ROWS} 条数据。",
            status_code=413,
        )
    canonical = frame[
        ["observation_date", "available_at", "value", "vintage", "revision"]
    ].copy()
    canonical["observation_date"] = pd.to_datetime(
        canonical["observation_date"], errors="raise"
    ).dt.normalize()
    canonical["available_at"] = pd.to_datetime(
        canonical["available_at"], errors="raise"
    ).dt.normalize()
    canonical["value"] = pd.to_numeric(canonical["value"], errors="coerce").astype(
        "float64"
    )
    canonical["vintage"] = canonical["vintage"].astype("string")
    canonical["revision"] = pd.to_numeric(
        canonical["revision"], errors="raise"
    ).astype("int64")
    canonical = canonical.sort_values(
        ["observation_date", "available_at", "revision"],
        kind="stable",
    ).reset_index(drop=True)
    digest = _upload_artifact_digest(canonical)
    artifact_id = f"upload-sha256-{digest}"
    checksum = f"sha256:{digest}"
    root = (Path(data_dir).expanduser().resolve() / UPLOAD_ARTIFACT_DIRNAME).resolve()
    root.mkdir(parents=True, exist_ok=True)
    final_path = root / f"{digest}.parquet"
    if not final_path.exists():
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".upload-",
            suffix=".parquet.tmp",
            dir=root,
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        try:
            table = pa.Table.from_pandas(canonical, preserve_index=False)
            metadata = dict(table.schema.metadata or {})
            metadata.update(
                {
                    b"research_series_artifact_id": artifact_id.encode("ascii"),
                    b"research_series_checksum": checksum.encode("ascii"),
                    b"research_series_schema": b"upload-series-v1",
                }
            )
            pq.write_table(
                table.replace_schema_metadata(metadata),
                temporary_path,
                compression="zstd",
            )
            with temporary_path.open("rb") as artifact_handle:
                os.fsync(artifact_handle.fileno())
            try:
                os.link(temporary_path, final_path)
            except FileExistsError:
                pass
        finally:
            temporary_path.unlink(missing_ok=True)
    return {
        "artifact_id": artifact_id,
        "checksum": checksum,
        "checksum_scope": "canonical_rows_v1",
        "format": "parquet",
        "schema_version": "upload-series-v1",
        "observations": len(canonical),
        "uri": f"research-series-upload://{artifact_id}",
        "persisted": True,
    }


def read_upload_artifact(
    data_dir: Path,
    artifact_id: str,
    checksum: str | None = None,
) -> pd.DataFrame:
    """Read and verify an immutable upload artifact without accepting a path."""

    match = _UPLOAD_ARTIFACT_PATTERN.fullmatch(str(artifact_id))
    if match is None:
        raise ResearchSeriesError(
            "UPLOAD_ARTIFACT_ID_INVALID",
            "上传数据版本标识无效。",
            field="artifact_id",
        )
    digest = match.group(1)
    expected_checksum = f"sha256:{digest}"
    if checksum is not None and checksum != expected_checksum:
        raise ResearchSeriesError(
            "UPLOAD_ARTIFACT_CHECKSUM_MISMATCH",
            "上传数据版本校验值不匹配。",
            field="checksum",
        )
    root = (Path(data_dir).expanduser().resolve() / UPLOAD_ARTIFACT_DIRNAME).resolve()
    path = (root / f"{digest}.parquet").resolve()
    if path.parent != root:
        raise ResearchSeriesError(
            "UPLOAD_ARTIFACT_ID_INVALID",
            "上传数据版本标识无效。",
            field="artifact_id",
        )
    if not path.exists():
        raise ResearchSeriesError(
            "UPLOAD_ARTIFACT_NOT_FOUND",
            "上传数据版本不存在或已不可用。",
            status_code=404,
            field="artifact_id",
        )
    parquet = pq.ParquetFile(path)
    metadata = parquet.schema_arrow.metadata or {}
    try:
        observed_metadata = (
            metadata.get(b"research_series_artifact_id", b"").decode("ascii"),
            metadata.get(b"research_series_checksum", b"").decode("ascii"),
            metadata.get(b"research_series_schema", b"").decode("ascii"),
        )
    except UnicodeDecodeError:
        observed_metadata = ("", "", "")
    if observed_metadata != (artifact_id, expected_checksum, "upload-series-v1"):
        raise ResearchSeriesError(
            "UPLOAD_ARTIFACT_METADATA_INVALID",
            "上传数据版本元数据校验失败。",
            status_code=409,
        )
    frame = pd.read_parquet(path)
    required = {
        "observation_date",
        "available_at",
        "value",
        "vintage",
        "revision",
    }
    if frame.empty or len(frame) > MAX_INLINE_ROWS or not required.issubset(frame.columns):
        raise ResearchSeriesError(
            "UPLOAD_ARTIFACT_SCHEMA_INVALID",
            "上传数据版本结构校验失败。",
            status_code=409,
        )
    observation_dates = pd.to_datetime(frame["observation_date"], errors="coerce")
    available_dates = pd.to_datetime(frame["available_at"], errors="coerce")
    revisions = pd.to_numeric(frame["revision"], errors="coerce")
    values = np.ascontiguousarray(
        pd.to_numeric(frame["value"], errors="coerce").to_numpy(dtype=np.float64)
    )
    if (
        observation_dates.isna().any()
        or available_dates.isna().any()
        or (available_dates < observation_dates).any()
        or revisions.isna().any()
        or (revisions < 1).any()
        or infinite_count_kernel(values) > 0
    ):
        raise ResearchSeriesError(
            "UPLOAD_ARTIFACT_VALUES_INVALID",
            "上传数据版本包含非法日期、修订号或无穷值。",
            status_code=409,
        )
    if _upload_artifact_digest(frame) != digest:
        raise ResearchSeriesError(
            "UPLOAD_ARTIFACT_CHECKSUM_MISMATCH",
            "上传数据版本内容校验失败。",
            status_code=409,
        )
    return frame


def _upload_artifact_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    values = np.ascontiguousarray(frame["value"].to_numpy(dtype=np.float64))
    mask = finite_mask_kernel(values)
    rows: list[dict[str, Any]] = []
    for index in range(len(frame)):
        vintage = frame["vintage"].iloc[index]
        rows.append(
            {
                "observation_date": _date_text(frame["observation_date"].iloc[index]),
                "available_at": _date_text(frame["available_at"].iloc[index]),
                "value": float(values[index]) if mask[index] == 1 else None,
                "vintage": (
                    None if vintage is None or pd.isna(vintage) else str(vintage)
                ),
                "revision": int(frame["revision"].iloc[index]),
            }
        )
    return rows


class ResearchSeriesService:
    def __init__(
        self,
        data_dir: Path | None = None,
        workspace_data_dir: Path | None = None,
    ) -> None:
        self.data_dir = (data_dir or DEFAULT_DATA_DIR).expanduser().resolve()
        self.workspace_data_dir = (workspace_data_dir or self.data_dir).expanduser().resolve()

    def _active_snapshot(self) -> tuple[Path, dict[str, object]]:
        try:
            manifest = read_active_manifest(self.data_dir)
        except MarketDataManifestError as exc:
            raise ResearchSeriesError(
                "ACTIVE_SNAPSHOT_INVALID",
                "Tushare 活跃快照 manifest 无效。",
                status_code=503,
            ) from exc
        if manifest is None:
            raise ResearchSeriesError(
                "ACTIVE_SNAPSHOT_REQUIRED",
                "研究数据实验室只读取已激活的 Tushare 快照。",
                status_code=503,
            )
        raw_directory = Path(str(manifest["snapshot_dir"]))
        snapshot = (
            raw_directory.resolve()
            if raw_directory.is_absolute()
            else (self.data_dir / raw_directory).resolve()
        )
        if snapshot != self.data_dir and self.data_dir not in snapshot.parents:
            raise ResearchSeriesError(
                "ACTIVE_SNAPSHOT_OUTSIDE_DATA_DIR",
                "Tushare 活跃快照必须位于项目 data 目录内。",
                status_code=503,
            )
        return snapshot, manifest

    @staticmethod
    def _snapshot_payload(snapshot: Path, manifest: dict[str, object]) -> dict[str, Any]:
        validation = manifest.get("validation") if isinstance(manifest.get("validation"), dict) else {}
        return {
            **_snapshot_identity(snapshot, manifest),
            "directory": snapshot.name,
            "activated_at": manifest.get("activated_at"),
            "manifest_schema_version": manifest.get("schema_version"),
            "validation_status": validation.get("status") if isinstance(validation, dict) else None,
            "source": "active_tushare_snapshot",
            "legacy_fallback": False,
        }

    @staticmethod
    def _coverage_lookup(snapshot: Path) -> dict[tuple[str, str], dict[str, Any]]:
        path = snapshot / "index_coverage_snapshot.parquet"
        if not path.exists() or pq.ParquetFile(path).metadata.num_rows == 0:
            return {}
        frame = pd.read_parquet(path)
        required = {"source_api", "ts_code"}
        if not required.issubset(frame.columns):
            return {}
        return {
            (str(row["source_api"]), str(row["ts_code"])): row.to_dict()
            for _, row in frame.iterrows()
        }

    def _index_catalog_items(
        self,
        snapshot: Path,
        manifest: dict[str, object],
    ) -> list[dict[str, Any]]:
        catalog_path = snapshot / "index_catalog_df.parquet"
        if not catalog_path.exists() or pq.ParquetFile(catalog_path).metadata.num_rows == 0:
            return [self._missing_index_catalog_item()]
        catalog = pd.read_parquet(catalog_path)
        if not {"ts_code", "quote_source_api"}.issubset(catalog.columns):
            return [self._missing_index_catalog_item()]
        coverage_lookup = self._coverage_lookup(snapshot)
        schema_fields: dict[str, list[str]] = {}
        file_available: dict[str, bool] = {}
        for source_api, filename in INDEX_SOURCE_FILES.items():
            path = snapshot / filename
            available = path.exists() and pq.ParquetFile(path).metadata.num_rows > 0
            file_available[source_api] = available
            schema_fields[source_api] = _numeric_fields(path) if available else []

        items: list[dict[str, Any]] = []
        for _, row in catalog.iterrows():
            code = _text(row.get("ts_code"))
            source_api = _text(row.get("quote_source_api"))
            if not code or not source_api:
                continue
            coverage = coverage_lookup.get((source_api, code))
            observation_count = int(coverage.get("rows") or 0) if coverage else 0
            available = file_available.get(source_api, False) and observation_count > 0
            fields = schema_fields.get(source_api, []) if available else []
            default_field = "close" if "close" in fields else fields[0] if fields else None
            coverage_rate = _nullable_scalar(
                coverage.get("domestic_trade_day_coverage") if coverage else None
            )
            missing_rate = complement_rate_kernel(
                np.ascontiguousarray(
                    np.array(
                        [np.nan if coverage_rate is None else coverage_rate],
                        dtype=np.float64,
                    )
                )
            )[0]
            missing_rate_value = _nullable_scalar(missing_rate)
            series_id = f"index:{source_api}:{code}"
            data_path = snapshot / str(INDEX_SOURCE_FILES.get(source_api) or "")
            file_checksum = _file_checksum(data_path) if available else None
            items.append(
                {
                    "id": series_id,
                    "kind": "index",
                    "name": _text(row.get("name")) or code,
                    "code": code,
                    "category": _text(row.get("category")) or "指数",
                    "market": _text(row.get("market")),
                    "publisher": _text(row.get("publisher")),
                    "status": "available" if available else "not_downloaded",
                    "status_reason": None if available else "series_not_present_in_active_snapshot",
                    "source_api": source_api,
                    "dataset": INDEX_SOURCE_FILES.get(source_api),
                    "default_field": default_field,
                    "fields": [_field_descriptor(field, "index") for field in fields],
                    "unit": _field_unit(default_field, "index") if default_field else None,
                    "frequency": "daily",
                    "coverage": {
                        "start_date": _date_text(coverage.get("first_date")) if coverage else None,
                        "end_date": _date_text(coverage.get("latest_date")) if coverage else None,
                        "observations": observation_count,
                        "calendar_coverage_rate": coverage_rate,
                    },
                    "missing": {
                        "count": None,
                        "rate": missing_rate_value,
                        "definition": "1 - active snapshot domestic trade-day coverage",
                    },
                    "pit": {
                        "supported": True,
                        "observation_field": "trade_date",
                        "available_at_field": "trade_date",
                        "availability_status": "date_only_market_close",
                    },
                    "vintage": {"supported": False, "field": None},
                    "profile_operations": INDEX_PROFILE_OPERATIONS if available else [],
                    "regime_node_type": "source.index" if available else None,
                    "binding_parameters": {
                        "ts_code": code,
                        "source_api": source_api,
                        "field": default_field,
                        "frequency": "daily",
                        "name": _text(row.get("name")) or code,
                        **_snapshot_identity(snapshot, manifest),
                        "source_file": INDEX_SOURCE_FILES.get(source_api),
                        "file_checksum": file_checksum,
                    },
                }
            )
        unique: dict[str, dict[str, Any]] = {}
        for item in items:
            unique.setdefault(str(item["id"]), item)
        return list(unique.values())

    @staticmethod
    def _missing_index_catalog_item() -> dict[str, Any]:
        return {
            "id": "index:index_catalog",
            "kind": "index",
            "name": "指数目录",
            "code": None,
            "status": "not_downloaded",
            "status_reason": "index_catalog_not_present_in_active_snapshot",
            "source_api": "index_catalog",
            "dataset": "index_catalog_df.parquet",
            "default_field": None,
            "fields": [],
            "unit": None,
            "frequency": "daily",
            "coverage": {"start_date": None, "end_date": None, "observations": 0},
            "missing": {"count": None, "rate": None},
            "pit": {"supported": False},
            "vintage": {"supported": False},
            "profile_operations": [],
            "regime_node_type": None,
            "binding_parameters": {},
        }

    def _macro_catalog_items(
        self,
        snapshot: Path,
        manifest: dict[str, object],
    ) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for spec in MACRO_DATASETS:
            path = snapshot / spec.filename
            if not path.exists() or pq.ParquetFile(path).metadata.num_rows == 0:
                items.append(self._macro_item(spec, None, None, snapshot, manifest))
                continue
            numeric_fields = _numeric_fields(path)
            columns = ["observation_date", *numeric_fields]
            schema_names = set(pq.ParquetFile(path).schema_arrow.names)
            for optional in (
                "available_at",
                "availability_status",
                "vintage",
                "revision",
                "ingested_at",
                "ts_code",
            ):
                if optional in schema_names:
                    columns.append(optional)
            frame = pd.read_parquet(path, columns=list(dict.fromkeys(columns)))
            if "ts_code" in frame.columns:
                codes = sorted(str(code) for code in frame["ts_code"].dropna().unique())
                if codes:
                    for code in codes:
                        items.append(
                            self._macro_item(
                                spec,
                                frame[frame["ts_code"].astype(str) == code],
                                code,
                                snapshot,
                                manifest,
                            )
                        )
                    continue
            items.append(self._macro_item(spec, frame, None, snapshot, manifest))
        return items

    def _macro_item(
        self,
        spec: MacroDatasetSpec,
        frame: pd.DataFrame | None,
        code: str | None,
        snapshot: Path,
        manifest: dict[str, object],
    ) -> dict[str, Any]:
        available = frame is not None and not frame.empty
        numeric_fields = (
            [field for field in frame.columns if field not in GOVERNANCE_FIELDS and pd.api.types.is_numeric_dtype(frame[field])]
            if available
            else []
        )
        field_rows: list[dict[str, Any]] = []
        for field in numeric_fields:
            values = np.ascontiguousarray(pd.to_numeric(frame[field], errors="coerce").to_numpy(dtype=np.float64))
            distribution = _distribution_payload(values)
            field_rows.append(
                _field_descriptor(
                    field,
                    "macro",
                    {
                        "count": distribution["missing_count"],
                        "rate": distribution["missing_rate"],
                    },
                )
            )
        default_field = numeric_fields[0] if numeric_fields else None
        dates = pd.to_datetime(frame["observation_date"], errors="coerce") if available and "observation_date" in frame else pd.Series(dtype="datetime64[ns]")
        valid_dates = dates.dropna()
        default_distribution = (
            _distribution_payload(
                np.ascontiguousarray(pd.to_numeric(frame[default_field], errors="coerce").to_numpy(dtype=np.float64))
            )
            if available and default_field
            else {"valid_count": 0, "missing_count": 0, "missing_rate": None}
        )
        release_known = 0
        if available and "available_at" in frame and "observation_date" in frame:
            release_lag = release_lag_days_kernel(
                _datetime_days(frame["observation_date"]),
                _datetime_days(frame["available_at"]),
            )
            release_known = int(_distribution_payload(release_lag)["valid_count"])
        suffix = f":{code}" if code else ""
        series_id = f"macro:{spec.stem}{suffix}"
        return {
            "id": series_id,
            "kind": "macro",
            "name": f"{spec.name} {code}" if code else spec.name,
            "code": code,
            "category": spec.category,
            "status": "available" if available else "not_downloaded",
            "status_reason": None if available else "dataset_not_present_in_active_snapshot",
            "source_api": spec.source_api,
            "dataset": spec.filename,
            "default_field": default_field,
            "fields": field_rows,
            "unit": _field_unit(default_field, "macro") if default_field else None,
            "frequency": spec.frequency,
            "coverage": {
                "start_date": _date_text(valid_dates.min()) if not valid_dates.empty else None,
                "end_date": _date_text(valid_dates.max()) if not valid_dates.empty else None,
                "observations": len(frame) if available else 0,
            },
            "missing": {
                "count": default_distribution["missing_count"],
                "rate": default_distribution["missing_rate"],
                "field": default_field,
            },
            "pit": {
                "supported": bool(available and "available_at" in frame),
                "observation_field": "observation_date",
                "available_at_field": "available_at" if available and "available_at" in frame else None,
                "known_release_dates": release_known,
                "availability_status_field": "availability_status" if available and "availability_status" in frame else None,
            },
            "vintage": {
                "supported": bool(available and ("vintage" in frame or "revision" in frame)),
                "field": "vintage" if available and "vintage" in frame else None,
                "revision_field": "revision" if available and "revision" in frame else None,
            },
            "profile_operations": MACRO_PROFILE_OPERATIONS if available and default_field else [],
            "regime_node_type": "source.macro" if available and default_field else None,
            "binding_parameters": {
                "dataset": spec.filename,
                "source_api": spec.source_api,
                "ts_code": code,
                "field": default_field,
                "frequency": spec.frequency,
                "name": f"{spec.name} {code}" if code else spec.name,
                **_snapshot_identity(snapshot, manifest),
                "source_file": spec.filename,
                "file_checksum": (
                    _file_checksum(snapshot / spec.filename)
                    if available
                    else None
                ),
            },
        }

    def _indicator_catalog_items(self) -> list[dict[str, Any]]:
        path = self.workspace_data_dir / "custom_indicators.json"
        if not path.exists():
            return [self._missing_indicator_item()]
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return [self._missing_indicator_item()]
        items: list[dict[str, Any]] = []
        for entry in payload.get("items", []) if isinstance(payload, dict) else []:
            versions = [entry.get("current"), *(entry.get("history") or [])] if isinstance(entry, dict) else []
            for version in versions:
                if not isinstance(version, dict):
                    continue
                indicator_id = _text(version.get("id"))
                revision = int(version.get("revision") or 1)
                if not indicator_id:
                    continue
                series_id = f"indicator:{indicator_id}@{revision}"
                periods = [str(period) for period in version.get("periods", [])]
                items.append(
                    {
                        "id": series_id,
                        "kind": "indicator",
                        "name": _text(version.get("name")) or indicator_id,
                        "code": indicator_id,
                        "category": "指标版本",
                        "status": "available" if version.get("expression") else "not_downloaded",
                        "status_reason": None if version.get("expression") else "indicator_expression_missing",
                        "source_api": "workspace_indicator_registry",
                        "dataset": "custom_indicators.json",
                        "default_field": "value",
                        "fields": [
                            {
                                "name": "value",
                                "label": _text(version.get("name")) or "指标值",
                                "unit": _text(version.get("unit")) or "dimensionless",
                                "dtype": "float64",
                                "nullable": True,
                            }
                        ],
                        "unit": _text(version.get("unit")) or "dimensionless",
                        "frequency": "period_defined",
                        "periods": periods,
                        "coverage": {"start_date": None, "end_date": None, "observations": None},
                        "missing": {"count": None, "rate": None},
                        "pit": {"supported": False, "reason": "depends_on_bound_source"},
                        "vintage": {"supported": True, "revision": revision},
                        "profile_operations": [],
                        "indicator_version": {
                            "indicator_id": indicator_id,
                            "revision": revision,
                            "dsl_version": version.get("dsl_version"),
                            "operator_registry_version": version.get("operator_registry_version"),
                        },
                        "regime_node_type": "source.indicator",
                        "binding_parameters": {
                            "indicator_id": indicator_id,
                            "indicator_revision": revision,
                            "product_kind": "",
                            "product_id": "",
                            "period": periods[0] if periods else "",
                            "name": _text(version.get("name")) or indicator_id,
                        },
                        "binding_required_inputs": ["product_kind", "product_id"],
                    }
                )
        return items or [self._missing_indicator_item()]

    @staticmethod
    def _missing_indicator_item() -> dict[str, Any]:
        return {
            "id": "indicator:workspace",
            "kind": "indicator",
            "name": "工作区指标版本",
            "code": None,
            "status": "not_downloaded",
            "status_reason": "indicator_registry_empty",
            "source_api": "workspace_indicator_registry",
            "dataset": "custom_indicators.json",
            "default_field": None,
            "fields": [],
            "unit": None,
            "frequency": "period_defined",
            "coverage": {"start_date": None, "end_date": None, "observations": None},
            "missing": {"count": None, "rate": None},
            "pit": {"supported": False},
            "vintage": {"supported": False},
            "profile_operations": [],
            "regime_node_type": None,
            "binding_parameters": {},
        }

    @staticmethod
    def _upload_catalog_item() -> dict[str, Any]:
        return {
            "id": "upload:time_series",
            "kind": "upload",
            "name": "用户上传时间序列",
            "code": None,
            "category": "用户数据",
            "status": "available",
            "status_reason": None,
            "source_api": "user_upload",
            "dataset": None,
            "default_field": "value",
            "fields": [
                {
                    "name": "value",
                    "label": "数值",
                    "unit": "user_defined",
                    "dtype": "float64",
                    "nullable": True,
                }
            ],
            "unit": "user_defined",
            "frequency": "user_defined",
            "coverage": {"start_date": None, "end_date": None, "observations": 0},
            "missing": {"count": None, "rate": None},
            "pit": {"supported": False},
            "vintage": {"supported": False},
            "profile_operations": INLINE_PROFILE_OPERATIONS,
            "regime_node_type": "source.upload",
            "binding_parameters": {
                "artifact_id": None,
                "checksum": None,
                "format": "parquet",
                "value_field": "value",
                "date_field": "observation_date",
                "available_at_field": "available_at",
                "vintage_field": "vintage",
                "revision_field": "revision",
                "frequency": "daily",
                "availability_mode": "point_in_time",
            },
            "binding_required_inputs": ["artifact_id", "checksum"],
            "capability": {
                "available": True,
                "accepted_formats": ["csv", "json"],
                "parsing_location": "frontend",
                "transport": "profile_registration",
                "persisted": True,
                "persistence": "content_addressed_immutable_parquet",
                "max_rows": MAX_INLINE_ROWS,
                "required_fields": ["date", "value"],
                "optional_fields": ["available_at", "vintage", "revision"],
            },
        }

    def catalog(
        self,
        *,
        kind: str | None = None,
        status: str | None = None,
        query: str | None = None,
        offset: int = 0,
        limit: int = 200,
    ) -> dict[str, Any]:
        snapshot, manifest = self._active_snapshot()
        items: list[dict[str, Any]] = []
        if kind in {None, "index"}:
            items.extend(self._index_catalog_items(snapshot, manifest))
        if kind in {None, "macro"}:
            items.extend(self._macro_catalog_items(snapshot, manifest))
        if kind in {None, "indicator"}:
            items.extend(self._indicator_catalog_items())
        if kind in {None, "upload"}:
            items.append(self._upload_catalog_item())
        if status:
            items = [item for item in items if item["status"] == status]
        if query:
            needle = query.strip().casefold()
            items = [
                item
                for item in items
                if needle in " ".join(
                    str(item.get(key) or "") for key in ("id", "name", "code", "category", "source_api")
                ).casefold()
            ]
        items.sort(key=lambda item: (str(item["kind"]), str(item.get("name") or ""), str(item["id"])))
        total = len(items)
        return {
            "schema_version": CATALOG_SCHEMA_VERSION,
            "snapshot": self._snapshot_payload(snapshot, manifest),
            "items": items[offset : offset + limit],
            "total": total,
            "offset": offset,
            "limit": limit,
            "capabilities": {
                "profile": {
                    "available": True,
                    "full_sample_computation": True,
                    "display_sampling": "deterministic_even_spacing",
                },
                "upload": self._upload_catalog_item()["capability"],
            },
            "execution": research_series_numba_execution_audit(),
        }

    @staticmethod
    def _filter_profile_dates(
        frame: pd.DataFrame,
        date_column: str,
        start_date: str | None,
        end_date: str | None,
    ) -> pd.DataFrame:
        start = _parse_optional_date(start_date, "start_date")
        end = _parse_optional_date(end_date, "end_date")
        if start is not None and end is not None and start > end:
            raise ResearchSeriesError("INVALID_DATE_RANGE", "start_date 不能晚于 end_date。")
        selected = frame.copy()
        selected["_observation_date"] = pd.to_datetime(selected[date_column], errors="coerce")
        selected = selected[selected["_observation_date"].notna()]
        if start is not None:
            selected = selected[selected["_observation_date"] >= start]
        if end is not None:
            selected = selected[selected["_observation_date"] <= end]
        return selected

    @staticmethod
    def _validate_profile_size(frame: pd.DataFrame) -> None:
        if frame.empty:
            raise ResearchSeriesError(
                "SERIES_WINDOW_EMPTY",
                "选定区间内没有可分析的观察值。",
                status_code=404,
            )
        if len(frame) > MAX_PROFILE_OBSERVATIONS:
            raise ResearchSeriesError(
                "PROFILE_OBSERVATION_LIMIT_EXCEEDED",
                f"单次分析最多允许 {MAX_PROFILE_OBSERVATIONS} 个观察值。",
                status_code=413,
            )

    @staticmethod
    def _sampled_date_text(frame: pd.DataFrame, indices: np.ndarray) -> list[str]:
        dates = frame["_observation_date"].tolist()
        return [_date_text(dates[index]) or "" for index in indices]

    @staticmethod
    def _index_identity(snapshot: Path, source_api: str, code: str) -> dict[str, Any]:
        path = snapshot / "index_catalog_df.parquet"
        if not path.exists():
            return {"name": code, "category": "指数", "market": None, "publisher": None}
        catalog = pd.read_parquet(path)
        selected = catalog[
            (catalog["ts_code"].astype(str) == code)
            & (catalog["quote_source_api"].astype(str) == source_api)
        ]
        if selected.empty:
            return {"name": code, "category": "指数", "market": None, "publisher": None}
        row = selected.iloc[-1]
        return {
            "name": _text(row.get("name")) or code,
            "category": _text(row.get("category")) or "指数",
            "market": _text(row.get("market")),
            "publisher": _text(row.get("publisher")),
        }

    def _profile_index(
        self,
        snapshot: Path,
        manifest: dict[str, object],
        *,
        series_id: str,
        field: str | None,
        start_date: str | None,
        end_date: str | None,
        rolling_window: int,
        sample_limit: int,
    ) -> dict[str, Any]:
        parts = series_id.split(":", 2)
        if len(parts) != 3 or parts[0] != "index":
            raise ResearchSeriesError("INVALID_SERIES_ID", "指数 series_id 格式无效。", field="series_id")
        source_api, code = parts[1], parts[2]
        filename = INDEX_SOURCE_FILES.get(source_api)
        if not filename:
            raise ResearchSeriesError("INDEX_SOURCE_UNSUPPORTED", "该指数行情来源暂不支持分析。", field="series_id")
        path = snapshot / filename
        if not path.exists() or pq.ParquetFile(path).metadata.num_rows == 0:
            raise ResearchSeriesError("SERIES_NOT_DOWNLOADED", "该指数序列未下载到活跃快照。", status_code=404)
        numeric_fields = _numeric_fields(path)
        selected_field = field or ("close" if "close" in numeric_fields else numeric_fields[0] if numeric_fields else None)
        if not selected_field or selected_field not in numeric_fields:
            raise ResearchSeriesError("PROFILE_FIELD_INVALID", "请选择可计算的数值字段。", field="field")
        frame = pd.read_parquet(
            path,
            columns=["ts_code", "trade_date", selected_field],
            filters=[("ts_code", "==", code)],
        )
        frame = frame[frame["ts_code"].astype(str) == code]
        frame = self._filter_profile_dates(frame, "trade_date", start_date, end_date)
        frame = frame.sort_values("_observation_date").drop_duplicates("_observation_date", keep="last")
        self._validate_profile_size(frame)
        values = np.ascontiguousarray(pd.to_numeric(frame[selected_field], errors="coerce").to_numpy(dtype=np.float64))
        normalized, returns, cumulative, drawdown, rolling_volatility = index_profile_kernel(
            values,
            np.int64(rolling_window),
            np.float64(252.0),
        )
        indices = sample_indices_kernel(np.int64(values.size), np.int64(sample_limit))
        raw_distribution = _distribution_payload(values)
        identity = self._index_identity(snapshot, source_api, code)
        binding_parameters = {
            "ts_code": code,
            "source_api": source_api,
            "field": selected_field,
            "frequency": "daily",
            "name": identity["name"],
            **_snapshot_identity(snapshot, manifest),
            "source_file": filename,
            "file_checksum": _file_checksum(path),
        }
        return {
            "schema_version": PROFILE_SCHEMA_VERSION,
            "series": {
                "id": series_id,
                "kind": "index",
                "code": code,
                "field": selected_field,
                "unit": _field_unit(selected_field, "index"),
                "frequency": "daily",
                "source_api": source_api,
                "dataset": filename,
                **identity,
            },
            "coverage": {
                "start_date": _date_text(frame["_observation_date"].iloc[0]),
                "end_date": _date_text(frame["_observation_date"].iloc[-1]),
                "observations": len(frame),
                "valid_observations": raw_distribution["valid_count"],
            },
            "missing": {
                "count": raw_distribution["missing_count"],
                "rate": raw_distribution["missing_rate"],
                "null_preserved": True,
            },
            "sampling": {
                "method": "deterministic_even_spacing",
                "computed_observations": len(frame),
                "displayed_observations": int(indices.size),
                "sample_limit": sample_limit,
                "computed_before_sampling": True,
            },
            "dates": self._sampled_date_text(frame, indices),
            "values": {
                "raw": _nullable_values(values, indices),
                "normalized": _nullable_values(normalized, indices),
                "return": _nullable_values(returns, indices),
                "cumulative_return": _nullable_values(cumulative, indices),
                "drawdown": _nullable_values(drawdown, indices),
                "rolling_volatility": _nullable_values(rolling_volatility, indices),
            },
            "distribution": {
                "raw": raw_distribution,
                "return": _distribution_payload(returns),
            },
            "pit": {
                "supported": True,
                "observation_field": "trade_date",
                "available_at_field": "trade_date",
                "availability_status": "date_only_market_close",
            },
            "vintage": {"supported": False, "values": []},
            "transform_definitions": {
                "normalized": "full selected sample z-score",
                "return": "adjacent simple return; missing observations are not bridged",
                "cumulative_return": "current value / first finite value - 1",
                "drawdown": "current value / running finite peak - 1",
                "rolling_volatility": f"{rolling_window}-observation sample volatility annualized by sqrt(252)",
            },
            "regime_node_type": "source.index",
            "binding_parameters": binding_parameters,
            "binding": {
                "node_type": "source.index",
                "parameters": binding_parameters,
            },
            "execution": research_series_numba_execution_audit(),
        }

    @staticmethod
    def _macro_series_parts(series_id: str) -> tuple[MacroDatasetSpec, str | None]:
        parts = series_id.split(":", 2)
        if len(parts) < 2 or parts[0] != "macro":
            raise ResearchSeriesError("INVALID_SERIES_ID", "宏观 series_id 格式无效。", field="series_id")
        spec = MACRO_BY_STEM.get(parts[1])
        if spec is None:
            raise ResearchSeriesError("MACRO_SOURCE_UNSUPPORTED", "该宏观数据源暂不支持分析。", field="series_id")
        return spec, parts[2] if len(parts) == 3 else None

    def _profile_macro(
        self,
        snapshot: Path,
        manifest: dict[str, object],
        *,
        series_id: str,
        field: str | None,
        start_date: str | None,
        end_date: str | None,
        as_of: str | None,
        vintage: str | None,
        sample_limit: int,
    ) -> dict[str, Any]:
        spec, code = self._macro_series_parts(series_id)
        path = snapshot / spec.filename
        if not path.exists() or pq.ParquetFile(path).metadata.num_rows == 0:
            raise ResearchSeriesError("SERIES_NOT_DOWNLOADED", "该宏观序列未下载到活跃快照。", status_code=404)
        schema_names = pq.ParquetFile(path).schema_arrow.names
        numeric_fields = _numeric_fields(path)
        selected_field = field or (numeric_fields[0] if numeric_fields else None)
        if not selected_field or selected_field not in numeric_fields:
            raise ResearchSeriesError("PROFILE_FIELD_INVALID", "请选择可计算的宏观数值字段。", field="field")
        columns = ["observation_date", selected_field]
        for optional in ("available_at", "availability_status", "vintage", "revision", "ingested_at", "ts_code"):
            if optional in schema_names:
                columns.append(optional)
        frame = pd.read_parquet(path, columns=columns)
        if code:
            if "ts_code" not in frame:
                raise ResearchSeriesError("SERIES_CODE_UNSUPPORTED", "该宏观数据集没有代码维度。", field="series_id")
            frame = frame[frame["ts_code"].astype(str) == code]
        frame = self._filter_profile_dates(frame, "observation_date", start_date, end_date)
        as_of_date = _parse_optional_date(as_of, "as_of")
        unknown_availability_excluded = 0
        if as_of_date is not None:
            if "available_at" not in frame:
                raise ResearchSeriesError(
                    "PIT_AVAILABILITY_UNKNOWN",
                    "该宏观数据没有 available_at，不能执行历史时点筛选。",
                    field="as_of",
                )
            frame["_available_at"] = pd.to_datetime(
                frame["available_at"], errors="coerce", utc=True
            ).dt.tz_convert(None)
            unknown_availability_excluded = len(frame[frame["_available_at"].isna()])
            frame = frame[
                frame["_available_at"].notna()
                & (frame["_available_at"].dt.normalize() <= as_of_date)
            ]
        elif "available_at" in frame:
            frame["_available_at"] = pd.to_datetime(
                frame["available_at"], errors="coerce", utc=True
            ).dt.tz_convert(None)
        if vintage is not None:
            if "vintage" not in frame:
                raise ResearchSeriesError("VINTAGE_UNAVAILABLE", "该宏观数据没有 vintage 字段。", field="vintage")
            frame = frame[frame["vintage"].astype(str) == vintage]
        sort_columns = ["_observation_date"]
        if "revision" in frame:
            frame["_revision"] = pd.to_numeric(frame["revision"], errors="coerce")
            sort_columns.append("_revision")
        if "_available_at" in frame:
            sort_columns.append("_available_at")
        if "ingested_at" in frame:
            frame["_ingested_at"] = pd.to_datetime(frame["ingested_at"], errors="coerce")
            sort_columns.append("_ingested_at")
        frame = frame.sort_values(sort_columns).drop_duplicates("_observation_date", keep="last")
        self._validate_profile_size(frame)
        values = np.ascontiguousarray(pd.to_numeric(frame[selected_field], errors="coerce").to_numpy(dtype=np.float64))
        year_over_year, month_over_month, quantile = macro_profile_kernel(
            values,
            np.int64(spec.year_over_year_lag),
        )
        if "available_at" in frame:
            release_lag = release_lag_days_kernel(
                _datetime_days(frame["_observation_date"]),
                _datetime_days(frame["available_at"]),
            )
        else:
            release_lag = np.ascontiguousarray(np.full(values.size, np.nan, dtype=np.float64))
        indices = sample_indices_kernel(np.int64(values.size), np.int64(sample_limit))
        raw_distribution = _distribution_payload(values)
        available_values = (
            pd.to_datetime(frame["available_at"], errors="coerce").tolist()
            if "available_at" in frame
            else [None] * len(frame)
        )
        vintage_values = frame["vintage"].tolist() if "vintage" in frame else [None] * len(frame)
        revision_values = frame["revision"].tolist() if "revision" in frame else [None] * len(frame)
        binding_parameters = {
            "dataset": spec.filename,
            "source_api": spec.source_api,
            "ts_code": code,
            "field": selected_field,
            "frequency": spec.frequency,
            "name": f"{spec.name} {code}" if code else spec.name,
            **_snapshot_identity(snapshot, manifest),
            "source_file": spec.filename,
            "file_checksum": _file_checksum(path),
        }
        return {
            "schema_version": PROFILE_SCHEMA_VERSION,
            "series": {
                "id": series_id,
                "kind": "macro",
                "name": f"{spec.name} {code}" if code else spec.name,
                "code": code,
                "field": selected_field,
                "unit": _field_unit(selected_field, "macro"),
                "frequency": spec.frequency,
                "source_api": spec.source_api,
                "dataset": spec.filename,
                "category": spec.category,
            },
            "coverage": {
                "start_date": _date_text(frame["_observation_date"].iloc[0]),
                "end_date": _date_text(frame["_observation_date"].iloc[-1]),
                "observations": len(frame),
                "valid_observations": raw_distribution["valid_count"],
            },
            "missing": {
                "count": raw_distribution["missing_count"],
                "rate": raw_distribution["missing_rate"],
                "null_preserved": True,
            },
            "sampling": {
                "method": "deterministic_even_spacing",
                "computed_observations": len(frame),
                "displayed_observations": int(indices.size),
                "sample_limit": sample_limit,
                "computed_before_sampling": True,
            },
            "dates": self._sampled_date_text(frame, indices),
            "values": {
                "raw": _nullable_values(values, indices),
                "yoy": _nullable_values(year_over_year, indices),
                "mom": _nullable_values(month_over_month, indices),
                "quantile": _nullable_values(quantile, indices),
                "release_lag_days": _nullable_values(release_lag, indices),
            },
            "distribution": {
                "raw": raw_distribution,
                "yoy": _distribution_payload(year_over_year),
                "mom": _distribution_payload(month_over_month),
                "release_lag_days": _distribution_payload(release_lag),
            },
            "pit": {
                "supported": "available_at" in frame,
                "as_of": _date_text(as_of_date),
                "observation_field": "observation_date",
                "available_at_field": "available_at" if "available_at" in frame else None,
                "unknown_availability_excluded": unknown_availability_excluded,
                "available_at": [_date_text(available_values[index]) for index in indices],
            },
            "vintage": {
                "supported": "vintage" in frame or "revision" in frame,
                "requested": vintage,
                "values": [None if pd.isna(vintage_values[index]) else str(vintage_values[index]) for index in indices],
                "revisions": [None if pd.isna(revision_values[index]) else int(revision_values[index]) for index in indices],
            },
            "transform_definitions": {
                "yoy": f"adjacent {spec.year_over_year_lag}-observation percent change",
                "mom": "adjacent one-observation percent change",
                "quantile": "full selected sample average-rank percentile",
                "release_lag_days": "available_at date - observation_date",
            },
            "regime_node_type": "source.macro",
            "binding_parameters": binding_parameters,
            "binding": {
                "node_type": "source.macro",
                "parameters": binding_parameters,
            },
            "execution": research_series_numba_execution_audit(),
        }

    def _profile_inline(
        self,
        *,
        series_id: str,
        field: str | None,
        inline_rows: list[dict[str, Any]] | None,
        rows: list[dict[str, Any]] | None,
        name: str | None,
        frequency: str,
        availability_mode: str,
        register_artifact: bool,
        start_date: str | None,
        end_date: str | None,
        as_of: str | None,
        vintage: str | None,
        rolling_window: int,
        sample_limit: int,
    ) -> dict[str, Any]:
        if inline_rows is not None and rows is not None:
            raise ResearchSeriesError(
                "INLINE_ROWS_AMBIGUOUS",
                "inline_rows 与 rows 只能提供一个。",
                field="inline_rows",
            )
        source_rows = inline_rows if inline_rows is not None else rows
        if not isinstance(source_rows, list) or not source_rows:
            raise ResearchSeriesError(
                "INLINE_ROWS_REQUIRED",
                "内联分析至少需要一条数据。",
                field="inline_rows",
            )
        if len(source_rows) > MAX_INLINE_ROWS:
            raise ResearchSeriesError(
                "INLINE_ROWS_LIMIT_EXCEEDED",
                f"内联分析最多允许 {MAX_INLINE_ROWS} 条数据。",
                status_code=413,
                field="inline_rows",
            )
        if field not in {None, "value"}:
            raise ResearchSeriesError(
                "INLINE_FIELD_INVALID",
                "内联数据的数值字段必须是 value。",
                field="field",
            )
        year_over_year_lag, annualization = _inline_periods(frequency)
        normalized_frequency = frequency.strip().lower()
        if availability_mode not in {"point_in_time", "latest"}:
            raise ResearchSeriesError(
                "INLINE_AVAILABILITY_MODE_INVALID",
                "availability_mode 必须是 point_in_time 或 latest。",
                field="availability_mode",
            )

        parsed_rows: list[dict[str, Any]] = []
        for index, row in enumerate(source_rows):
            if not isinstance(row, dict):
                raise ResearchSeriesError(
                    "INLINE_ROW_INVALID",
                    f"第 {index + 1} 行必须是对象。",
                    field=f"inline_rows.{index}",
                )
            date_value = row.get("observation_date", row.get("date"))
            if not isinstance(date_value, str) or not date_value.strip():
                raise ResearchSeriesError(
                    "INLINE_DATE_REQUIRED",
                    f"第 {index + 1} 行缺少有效 date。",
                    field=f"inline_rows.{index}.date",
                )
            observation_date = _parse_optional_date(
                date_value,
                f"inline_rows.{index}.date",
            )
            assert observation_date is not None
            available_value = row.get("available_at", date_value)
            if not isinstance(available_value, str) or not available_value.strip():
                raise ResearchSeriesError(
                    "INLINE_AVAILABLE_AT_INVALID",
                    f"第 {index + 1} 行的 available_at 无效。",
                    field=f"inline_rows.{index}.available_at",
                )
            available_at = _parse_optional_date(
                available_value,
                f"inline_rows.{index}.available_at",
            )
            assert available_at is not None
            if available_at < observation_date:
                raise ResearchSeriesError(
                    "INLINE_AVAILABLE_BEFORE_OBSERVATION",
                    f"第 {index + 1} 行的 available_at 不能早于 date。",
                    field=f"inline_rows.{index}.available_at",
                )
            if "value" not in row:
                raise ResearchSeriesError(
                    "INLINE_VALUE_REQUIRED",
                    f"第 {index + 1} 行缺少 value。",
                    field=f"inline_rows.{index}.value",
                )
            raw_value = row.get("value")
            if raw_value is None:
                numeric_value = np.nan
            else:
                if isinstance(raw_value, bool):
                    raise ResearchSeriesError(
                        "INLINE_VALUE_INVALID",
                        f"第 {index + 1} 行的 value 必须是数值或 null。",
                        field=f"inline_rows.{index}.value",
                    )
                try:
                    numeric_value = float(raw_value)
                except (TypeError, ValueError) as exc:
                    raise ResearchSeriesError(
                        "INLINE_VALUE_INVALID",
                        f"第 {index + 1} 行的 value 必须是数值或 null。",
                        field=f"inline_rows.{index}.value",
                    ) from exc
                if not math.isfinite(numeric_value):
                    raise ResearchSeriesError(
                        "INLINE_VALUE_INVALID",
                        f"第 {index + 1} 行的 value 不能是 NaN 或无穷值；缺失请使用 null。",
                        field=f"inline_rows.{index}.value",
                    )
            raw_vintage = row.get("vintage")
            vintage_value: str | None = None
            if raw_vintage is not None:
                vintage_value = str(raw_vintage).strip()
                if not vintage_value or len(vintage_value) > 160:
                    raise ResearchSeriesError(
                        "INLINE_VINTAGE_INVALID",
                        f"第 {index + 1} 行的 vintage 必须是 1 到 160 个字符。",
                        field=f"inline_rows.{index}.vintage",
                    )
            raw_revision = row.get("revision", 1)
            if isinstance(raw_revision, bool):
                revision = 0
            else:
                try:
                    revision = int(raw_revision)
                    if float(raw_revision) != float(revision):
                        revision = 0
                except (TypeError, ValueError, OverflowError):
                    revision = 0
            if revision < 1:
                raise ResearchSeriesError(
                    "INLINE_REVISION_INVALID",
                    f"第 {index + 1} 行的 revision 必须是正整数。",
                    field=f"inline_rows.{index}.revision",
                )
            parsed_rows.append(
                {
                    "observation_date": observation_date,
                    "available_at": available_at,
                    "value": numeric_value,
                    "vintage": vintage_value,
                    "revision": revision,
                    "_row_id": index,
                }
            )

        frame = pd.DataFrame(parsed_rows)
        artifact_source = frame.copy()
        frame = self._filter_profile_dates(
            frame,
            "observation_date",
            start_date,
            end_date,
        )
        as_of_date = _parse_optional_date(as_of, "as_of")
        if as_of_date is not None:
            frame = frame[frame["available_at"] <= as_of_date]
        if vintage is not None:
            requested_vintage = vintage.strip()
            if not requested_vintage:
                raise ResearchSeriesError(
                    "INLINE_VINTAGE_INVALID",
                    "vintage 不能为空。",
                    field="vintage",
                )
            frame = frame[frame["vintage"] == requested_vintage]
        self._validate_profile_size(frame)
        frame = frame.sort_values(
            ["_observation_date", "available_at", "revision", "_row_id"]
        )
        grouped = frame.groupby("_observation_date", as_index=False, sort=False)
        frame = (
            grouped.head(1)
            if availability_mode == "point_in_time"
            else grouped.tail(1)
        )
        frame = frame.sort_values("_observation_date").reset_index(drop=True)
        self._validate_profile_size(frame)
        artifact = (
            write_upload_artifact(self.data_dir, artifact_source)
            if register_artifact
            else None
        )

        values = np.ascontiguousarray(frame["value"].to_numpy(dtype=np.float64))
        normalized, returns, cumulative, drawdown, rolling_volatility = (
            index_profile_kernel(
                values,
                np.int64(rolling_window),
                np.float64(annualization),
            )
        )
        year_over_year, month_over_month, quantile = macro_profile_kernel(
            values,
            np.int64(year_over_year_lag),
        )
        release_lag = release_lag_days_kernel(
            _datetime_days(frame["_observation_date"]),
            _datetime_days(frame["available_at"]),
        )
        indices = sample_indices_kernel(
            np.int64(values.size),
            np.int64(sample_limit),
        )
        raw_distribution = _distribution_payload(values)
        series_name = name.strip() if name and name.strip() else "内联时间序列"
        binding_parameters = (
            {
                "artifact_id": artifact["artifact_id"],
                "checksum": artifact["checksum"],
                "format": "parquet",
                "value_field": "value",
                "date_field": "observation_date",
                "available_at_field": "available_at",
                "vintage_field": "vintage",
                "revision_field": "revision",
                "frequency": normalized_frequency,
                "availability_mode": availability_mode,
                "name": series_name,
            }
            if artifact is not None
            else {}
        )
        fingerprint = (
            str(artifact["checksum"]).removeprefix("sha256:")
            if artifact is not None
            else _upload_artifact_digest(frame)
        )
        vintage_values = frame["vintage"].tolist()
        revision_values = frame["revision"].tolist()
        available_values = frame["available_at"].tolist()
        return {
            "schema_version": PROFILE_SCHEMA_VERSION,
            "series": {
                "id": series_id,
                "kind": "upload",
                "name": series_name,
                "code": None,
                "field": "value",
                "unit": "user_defined",
                "frequency": normalized_frequency,
                "source_api": "user_upload",
                "dataset": None,
            },
            "coverage": {
                "start_date": _date_text(frame["_observation_date"].iloc[0]),
                "end_date": _date_text(frame["_observation_date"].iloc[-1]),
                "input_observations": len(source_rows),
                "observations": len(frame),
                "valid_observations": raw_distribution["valid_count"],
            },
            "missing": {
                "count": raw_distribution["missing_count"],
                "rate": raw_distribution["missing_rate"],
                "null_preserved": True,
            },
            "sampling": {
                "method": "deterministic_even_spacing",
                "computed_observations": len(frame),
                "displayed_observations": int(indices.size),
                "sample_limit": sample_limit,
                "computed_before_sampling": True,
            },
            "dates": self._sampled_date_text(frame, indices),
            "values": {
                "raw": _nullable_values(values, indices),
                "normalized": _nullable_values(normalized, indices),
                "return": _nullable_values(returns, indices),
                "cumulative_return": _nullable_values(cumulative, indices),
                "drawdown": _nullable_values(drawdown, indices),
                "rolling_volatility": _nullable_values(rolling_volatility, indices),
                "yoy": _nullable_values(year_over_year, indices),
                "mom": _nullable_values(month_over_month, indices),
                "quantile": _nullable_values(quantile, indices),
                "release_lag_days": _nullable_values(release_lag, indices),
            },
            "distribution": {
                "raw": raw_distribution,
                "return": _distribution_payload(returns),
                "yoy": _distribution_payload(year_over_year),
                "mom": _distribution_payload(month_over_month),
                "release_lag_days": _distribution_payload(release_lag),
            },
            "pit": {
                "supported": True,
                "as_of": _date_text(as_of_date),
                "observation_field": "observation_date",
                "available_at_field": "available_at",
                "availability_mode": availability_mode,
                "available_at": [
                    _date_text(available_values[index]) for index in indices
                ],
            },
            "vintage": {
                "supported": any(value is not None for value in vintage_values),
                "requested": vintage,
                "values": [
                    None if vintage_values[index] is None else str(vintage_values[index])
                    for index in indices
                ],
                "revisions": [int(revision_values[index]) for index in indices],
            },
            "transform_definitions": {
                "normalized": "full selected sample z-score",
                "return": "adjacent simple return; missing observations are not bridged",
                "cumulative_return": "current value / first finite value - 1",
                "drawdown": "current value / running finite peak - 1",
                "rolling_volatility": (
                    f"{rolling_window}-observation sample volatility annualized by "
                    f"sqrt({annualization:g})"
                ),
                "yoy": f"adjacent {year_over_year_lag}-observation percent change",
                "mom": "adjacent one-observation percent change",
                "quantile": "full selected sample average-rank percentile",
                "release_lag_days": "available_at date - observation_date",
            },
            "artifact": artifact,
            "regime_node_type": "source.upload" if artifact is not None else None,
            "binding_parameters": binding_parameters,
            "binding": (
                {
                    "node_type": "source.upload",
                    "parameters": binding_parameters,
                    "fingerprint": fingerprint,
                }
                if artifact is not None
                else None
            ),
            "snapshot": {
                "source": (
                    "content_addressed_upload_artifact"
                    if artifact is not None
                    else "request_inline_rows_draft"
                ),
                "persisted": artifact is not None,
                "legacy_fallback": False,
                "fingerprint": fingerprint,
            },
            "execution": research_series_numba_execution_audit(),
        }

    def profile(
        self,
        *,
        series_id: str = "upload:time_series",
        field: str | None = None,
        inline_rows: list[dict[str, Any]] | None = None,
        rows: list[dict[str, Any]] | None = None,
        name: str | None = None,
        frequency: str = "daily",
        availability_mode: str = "point_in_time",
        register_artifact: bool = True,
        artifact_id: str | None = None,
        checksum: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        as_of: str | None = None,
        vintage: str | None = None,
        rolling_window: int = 20,
        sample_limit: int = 500,
    ) -> dict[str, Any]:
        if artifact_id is not None:
            if inline_rows is not None or rows is not None:
                raise ResearchSeriesError(
                    "UPLOAD_SOURCE_AMBIGUOUS",
                    "artifact_id 不能与 inline_rows 或 rows 同时提供。",
                    field="artifact_id",
                )
            inline_rows = _upload_artifact_rows(
                read_upload_artifact(self.data_dir, artifact_id, checksum)
            )
            register_artifact = True
        elif checksum is not None:
            raise ResearchSeriesError(
                "UPLOAD_ARTIFACT_ID_REQUIRED",
                "提供 checksum 时必须同时提供 artifact_id。",
                field="artifact_id",
            )
        has_inline_rows = inline_rows is not None or rows is not None
        is_inline_series = series_id.startswith(("upload:", "inline:"))
        if has_inline_rows or is_inline_series:
            if not is_inline_series:
                raise ResearchSeriesError(
                    "INLINE_SERIES_ID_CONFLICT",
                    "提供 inline_rows 时 series_id 必须是 upload: 或 inline: 序列。",
                    field="series_id",
                )
            return self._profile_inline(
                series_id=series_id,
                field=field,
                inline_rows=inline_rows,
                rows=rows,
                name=name,
                frequency=frequency,
                availability_mode=availability_mode,
                register_artifact=register_artifact,
                start_date=start_date,
                end_date=end_date,
                as_of=as_of,
                vintage=vintage,
                rolling_window=rolling_window,
                sample_limit=sample_limit,
            )
        snapshot, manifest = self._active_snapshot()
        if series_id.startswith("index:"):
            result = self._profile_index(
                snapshot,
                manifest,
                series_id=series_id,
                field=field,
                start_date=start_date,
                end_date=end_date,
                rolling_window=rolling_window,
                sample_limit=sample_limit,
            )
        elif series_id.startswith("macro:"):
            result = self._profile_macro(
                snapshot,
                manifest,
                series_id=series_id,
                field=field,
                start_date=start_date,
                end_date=end_date,
                as_of=as_of,
                vintage=vintage,
                sample_limit=sample_limit,
            )
        else:
            raise ResearchSeriesError(
                "PROFILE_KIND_UNSUPPORTED",
                "当前 profile 仅支持指数与宏观序列。",
                field="series_id",
            )
        result["snapshot"] = self._snapshot_payload(snapshot, manifest)
        return result

    def compare(
        self,
        *,
        sources: list[dict[str, Any]],
        sample_limit: int = 500,
    ) -> dict[str, Any]:
        if not isinstance(sources, list) or not 2 <= len(sources) <= 4:
            raise ResearchSeriesError(
                "COMPARE_SOURCE_COUNT_INVALID",
                "多序列比较必须选择 2 到 4 条序列。",
                field="sources",
            )
        profiles: list[dict[str, Any]] = []
        source_ids: list[str] = []
        labels: list[str] = []
        date_arrays: list[np.ndarray] = []
        value_arrays: list[np.ndarray] = []
        for index, source in enumerate(sources):
            if not isinstance(source, dict):
                raise ResearchSeriesError(
                    "COMPARE_SOURCE_INVALID",
                    f"第 {index + 1} 个比较源必须是对象。",
                    field=f"sources.{index}",
                )
            payload = dict(source)
            requested_id = payload.pop("id", None)
            requested_label = payload.pop("label", None)
            payload["sample_limit"] = MAX_PROFILE_OBSERVATIONS
            if payload.get("inline_rows") is not None or payload.get("rows") is not None:
                payload["register_artifact"] = False
            profile = self.profile(**payload)
            source_id = str(requested_id or f"series-{index + 1}")
            if source_id in source_ids:
                raise ResearchSeriesError(
                    "COMPARE_SOURCE_ID_DUPLICATED",
                    "比较源 id 不能重复。",
                    field=f"sources.{index}.id",
                )
            source_ids.append(source_id)
            labels.append(str(requested_label or profile["series"]["name"]))
            raw_dates = pd.to_datetime(profile["dates"], errors="coerce")
            if raw_dates.isna().any():
                raise ResearchSeriesError(
                    "COMPARE_DATE_INVALID",
                    "比较源包含无效日期。",
                    status_code=409,
                )
            date_arrays.append(
                np.ascontiguousarray(
                    raw_dates.to_numpy(dtype="datetime64[D]").astype(np.int64)
                )
            )
            value_arrays.append(
                np.ascontiguousarray(
                    np.array(
                        [
                            np.nan if value is None else float(value)
                            for value in profile["values"]["raw"]
                        ],
                        dtype=np.float64,
                    )
                )
            )
            profiles.append(profile)

        common_dates = date_arrays[0]
        for dates in date_arrays[1:]:
            common_dates = strict_intersection_dates_kernel(common_dates, dates)
        if common_dates.size == 0:
            raise ResearchSeriesError(
                "COMPARE_NO_DATE_OVERLAP",
                "所选序列没有共同日期，无法比较。",
                status_code=404,
            )
        aligned = np.empty((common_dates.size, len(sources)), dtype=np.float64)
        for column, (dates, values) in enumerate(zip(date_arrays, value_arrays)):
            aligned[:, column] = align_values_kernel(common_dates, dates, values)
        aligned = np.ascontiguousarray(aligned)
        standardized = standardize_matrix_kernel(aligned)
        correlations, pair_counts = pearson_matrix_kernel(aligned)
        complete_cases = complete_case_indices_kernel(aligned)
        display_indices = sample_indices_kernel(
            np.int64(common_dates.size),
            np.int64(sample_limit),
        )

        def date_at(position: int) -> str:
            return np.datetime_as_string(
                np.datetime64(int(common_dates[position]), "D"),
                unit="D",
            )

        correlation_mask = finite_mask_kernel(
            np.ascontiguousarray(correlations.reshape(-1))
        ).reshape(correlations.shape)
        correlation_payload = [
            [
                (
                    float(correlations[left, right])
                    if correlation_mask[left, right] == 1
                    else None
                )
                for right in range(len(sources))
            ]
            for left in range(len(sources))
        ]
        series_payload: list[dict[str, Any]] = []
        for column, profile in enumerate(profiles):
            raw_column = np.ascontiguousarray(aligned[:, column])
            standardized_column = np.ascontiguousarray(standardized[:, column])
            series_payload.append(
                {
                    "id": source_ids[column],
                    "label": labels[column],
                    "series": profile["series"],
                    "values": _nullable_values(raw_column, display_indices),
                    "standardized": _nullable_values(
                        standardized_column,
                        display_indices,
                    ),
                    "profile_observations": profile["coverage"]["observations"],
                    "snapshot": profile.get("snapshot"),
                    "regime_node_type": profile.get("regime_node_type"),
                    "binding_parameters": profile.get("binding_parameters"),
                    "binding": profile.get("binding"),
                }
            )

        scatter_pairs: list[dict[str, Any]] = []
        for left in range(len(sources)):
            for right in range(left + 1, len(sources)):
                valid_rows = pair_valid_indices_kernel(
                    aligned,
                    np.int64(left),
                    np.int64(right),
                )
                pair_sample_positions = sample_indices_kernel(
                    np.int64(valid_rows.size),
                    np.int64(sample_limit),
                )
                sampled_rows = valid_rows[pair_sample_positions]
                scatter_pairs.append(
                    {
                        "left_id": source_ids[left],
                        "right_id": source_ids[right],
                        "observation_count": int(valid_rows.size),
                        "displayed_observations": int(sampled_rows.size),
                        "dates": [date_at(int(row)) for row in sampled_rows],
                        "x": [float(aligned[row, left]) for row in sampled_rows],
                        "y": [float(aligned[row, right]) for row in sampled_rows],
                        "standardized_x": [
                            float(standardized[row, left]) for row in sampled_rows
                        ],
                        "standardized_y": [
                            float(standardized[row, right]) for row in sampled_rows
                        ],
                        "correlation": correlation_payload[left][right],
                    }
                )

        return {
            "schema_version": "research-series-compare-v1",
            "alignment": {
                "method": "strict_date_intersection",
                "intersected_observations": int(common_dates.size),
                "source_observations": [
                    int(profile["coverage"]["observations"])
                    for profile in profiles
                ],
                "computed_before_sampling": True,
            },
            "sampling": {
                "method": "deterministic_even_spacing",
                "computed_observations": int(common_dates.size),
                "displayed_observations": int(display_indices.size),
                "sample_limit": sample_limit,
                "computed_before_sampling": True,
            },
            "dates": [date_at(int(index)) for index in display_indices],
            "series": series_payload,
            "correlation": {
                "method": "pearson_pairwise_finite_after_strict_date_intersection",
                "source_ids": source_ids,
                "matrix": correlation_payload,
                "observation_counts": pair_counts.tolist(),
            },
            "scatter_pairs": scatter_pairs,
            "common_valid": {
                "definition": "all selected series finite on a strictly intersected date",
                "observation_count": int(complete_cases.size),
                "start_date": (
                    date_at(int(complete_cases[0])) if complete_cases.size else None
                ),
                "end_date": (
                    date_at(int(complete_cases[-1])) if complete_cases.size else None
                ),
            },
            "execution": research_series_numba_execution_audit(),
        }


__all__ = [
    "CATALOG_SCHEMA_VERSION",
    "DEFAULT_DATA_DIR",
    "MAX_INLINE_ROWS",
    "PROFILE_SCHEMA_VERSION",
    "ResearchSeriesError",
    "ResearchSeriesService",
    "UPLOAD_ARTIFACT_DIRNAME",
    "read_upload_artifact",
    "write_upload_artifact",
]

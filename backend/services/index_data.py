"""Index catalog, coverage snapshots and read-only query helpers."""

from __future__ import annotations

import hashlib
import math
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as parquet

try:
    from backend.compute_policy import validate_execution_audit
    from backend.instrument_analytics_numba import (
        count_true_kernel,
        coverage_ratio_kernel,
        encoded_category_counts_kernel,
        encoded_unique_count_kernel,
        instrument_analytics_numba_execution_audit,
        int_range_count_kernel,
        numeric_sort_order_kernel,
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
        int_range_count_kernel,
        numeric_sort_order_kernel,
    )
    from market_data import resolve_tushare_data_dir


INDEX_SCOPES = (
    "catalog",
    "domestic",
    "industry",
    "concept",
    "global",
    "futures",
    "valuation",
    "constituents",
)
DEFAULT_INDEX_SCOPES = ("catalog", "domestic", "industry", "global")
INDEX_SCOPE_LABELS = {
    "catalog": "指数目录",
    "domestic": "境内指数",
    "industry": "行业指数",
    "concept": "概念板块",
    "global": "国际指数",
    "futures": "商品期货指数",
    "valuation": "指数估值",
    "constituents": "成分与权重",
}
INDEX_SCOPE_FILES = {
    "catalog": (
        "index_info.parquet",
        "etf_index.parquet",
        "index_catalog_df.parquet",
    ),
    "domestic": ("index_daily_df.parquet",),
    "industry": ("index_sw_daily_df.parquet", "index_ci_daily_df.parquet"),
    "concept": (
        "index_ths_daily_df.parquet",
        "index_dc_daily_df.parquet",
        "index_tdx_daily_df.parquet",
    ),
    "global": ("index_global_daily_df.parquet",),
    "futures": ("index_futures_daily_df.parquet",),
    "valuation": ("index_daily_basic_df.parquet",),
    "constituents": ("index_members_df.parquet", "index_weights_df.parquet"),
}
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
INDEX_DATASET_SPECS = {
    "index_catalog": ("index_catalog_df.parquet", "list_date", "catalog"),
    "index_domestic": ("index_daily_df.parquet", "trade_date", "domestic"),
    "index_sw": ("index_sw_daily_df.parquet", "trade_date", "industry"),
    "index_ci": ("index_ci_daily_df.parquet", "trade_date", "industry"),
    "index_ths": ("index_ths_daily_df.parquet", "trade_date", "concept"),
    "index_dc": ("index_dc_daily_df.parquet", "trade_date", "concept"),
    "index_tdx": ("index_tdx_daily_df.parquet", "trade_date", "concept"),
    "index_global": ("index_global_daily_df.parquet", "trade_date", "global"),
    "index_futures": ("index_futures_daily_df.parquet", "trade_date", "futures"),
    "index_valuation": ("index_daily_basic_df.parquet", "trade_date", "valuation"),
    "index_members": ("index_members_df.parquet", None, "constituents"),
    "index_weights": ("index_weights_df.parquet", "trade_date", "constituents"),
    "index_coverage": ("index_coverage_snapshot.parquet", "latest_date", "coverage"),
}


class IndexDataValidationError(RuntimeError):
    """Raised when a selected index dataset cannot be safely activated."""


def normalise_index_scopes(scopes: Iterable[str] | None) -> list[str]:
    selected = list(dict.fromkeys(str(item).strip().lower() for item in (scopes or []) if str(item).strip()))
    if not selected:
        selected = list(DEFAULT_INDEX_SCOPES)
    unknown = set(selected) - set(INDEX_SCOPES)
    if unknown:
        raise ValueError(f"不支持的指数范围: {', '.join(sorted(unknown))}")
    if "catalog" not in selected:
        selected.insert(0, "catalog")
    return selected


def index_required_files(scopes: Iterable[str] | None) -> tuple[str, ...]:
    selected = normalise_index_scopes(scopes)
    files: list[str] = []
    for scope in selected:
        files.extend(INDEX_SCOPE_FILES[scope])
    files.append("index_coverage_snapshot.parquet")
    return tuple(dict.fromkeys(files))


def _fingerprint(path: Path) -> str:
    stat = path.stat()
    raw = f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def _normalise_date(value: Any) -> pd.Timestamp | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        text = str(value).strip().replace("-", "")
        parsed = pd.to_datetime(text[:8], format="%Y%m%d", errors="coerce")
    return None if pd.isna(parsed) else pd.Timestamp(parsed).normalize()


def _category_codes(series: pd.Series) -> tuple[list[str], np.ndarray]:
    """Map source labels in Python; all counts and ordering remain in NJIT."""

    labels: list[str] = []
    code_by_label: dict[str, int] = {}
    codes = np.full(len(series), -1, dtype=np.int64)
    for position, raw_value in enumerate(series.tolist()):
        label = "未知" if pd.isna(raw_value) else str(raw_value).strip() or "未知"
        code = code_by_label.get(label)
        if code is None:
            code = len(labels)
            code_by_label[label] = code
            labels.append(label)
        codes[position] = code
    return labels, np.ascontiguousarray(codes)


def _source_rows(series: pd.Series) -> tuple[list[dict[str, Any]], int]:
    labels, codes = _category_codes(series)
    counts = encoded_category_counts_kernel(codes, len(labels))
    order = numeric_sort_order_kernel(
        np.ascontiguousarray(counts.astype(np.float64)),
        np.arange(len(labels), dtype=np.int64),
        np.uint8(0),
    )
    rows = [
        {"source_api": labels[index], "count": int(counts[index])}
        for index in order
    ]
    return rows, int(encoded_unique_count_kernel(codes))


def _execution_audit() -> dict[str, object]:
    return validate_execution_audit(instrument_analytics_numba_execution_audit())


def _calendar_open_dates(data_dir: Path) -> pd.DatetimeIndex:
    path = data_dir / "trade_day_df.parquet"
    if not path.exists():
        return pd.DatetimeIndex([])
    try:
        frame = pd.read_parquet(path, columns=["exchange", "cal_date", "is_open"])
    except (KeyError, OSError, ValueError):
        return pd.DatetimeIndex([])
    frame = frame[
        frame["exchange"].astype(str).eq("SSE")
        & pd.to_numeric(frame["is_open"], errors="coerce").eq(1)
    ]
    if pd.api.types.is_datetime64_any_dtype(frame["cal_date"]):
        parsed = pd.to_datetime(frame["cal_date"], errors="coerce")
    else:
        compact = frame["cal_date"].astype(str).str.replace("-", "", regex=False).str[:8]
        parsed = pd.to_datetime(compact, format="%Y%m%d", errors="coerce")
    return pd.DatetimeIndex(parsed.dropna().unique()).sort_values()


def build_index_coverage_snapshot(data_dir: Path) -> dict[str, Any]:
    """Stream index histories and create one compact row per source/code."""

    root = Path(data_dir)
    open_dates = _calendar_open_dates(root)
    records: list[dict[str, Any]] = []
    for source_api, filename in INDEX_HISTORY_FILES.items():
        path = root / filename
        if not path.exists():
            continue
        source = parquet.ParquetFile(path)
        names = set(source.schema.names)
        if not {"ts_code", "trade_date"}.issubset(names):
            raise IndexDataValidationError(f"{filename} 缺少 ts_code/trade_date。")
        stats: dict[str, list[Any]] = {}
        source_column = "source_api" if "source_api" in names else None
        columns = ["ts_code", "trade_date"] + ([source_column] if source_column else [])
        for batch in source.iter_batches(batch_size=65_536, columns=columns, use_threads=False):
            frame = batch.to_pandas()
            frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
            frame = frame.dropna(subset=["ts_code", "trade_date"])
            if source_column:
                grouped = (
                    (str(observed_source), str(code), group)
                    for (observed_source, code), group in frame.groupby(
                        [source_column, "ts_code"], sort=False
                    )
                )
            else:
                grouped = (
                    (source_api, str(code), group)
                    for code, group in frame.groupby("ts_code", sort=False)
                )
            for observed_source, code, group in grouped:
                key = f"{observed_source}\0{code}"
                earliest = group["trade_date"].min()
                latest = group["trade_date"].max()
                record = stats.setdefault(key, [str(observed_source), str(code), earliest, latest, 0])
                record[2] = min(record[2], earliest)
                record[3] = max(record[3], latest)
                record[4] += int(len(group))
        today = pd.Timestamp(date.today())
        for observed_source, code, earliest, latest, rows in stats.values():
            expected = None
            if observed_source in {"index_daily", "sw_daily", "ci_daily", "ths_daily", "dc_daily", "tdx_daily", "index_dailybasic"} and len(open_dates):
                expected = int(
                    int_range_count_kernel(
                        np.ascontiguousarray(open_dates.asi8, dtype=np.int64),
                        int(earliest.value),
                        int(latest.value),
                    )
                )
            records.append(
                {
                    "source_api": observed_source,
                    "ts_code": code,
                    "first_date": earliest,
                    "latest_date": latest,
                    "rows": rows,
                    "stale_days": max(int((today - latest.normalize()).days), 0),
                    "domestic_trade_day_coverage": (
                        float(coverage_ratio_kernel(rows, expected))
                        if expected
                        else None
                    ),
                    "source_file": filename,
                    "source_fingerprint": _fingerprint(path),
                }
            )
    frame = pd.DataFrame.from_records(
        records,
        columns=[
            "source_api", "ts_code", "first_date", "latest_date", "rows", "stale_days",
            "domestic_trade_day_coverage", "source_file", "source_fingerprint",
        ],
    )
    if not frame.empty:
        frame = frame.drop_duplicates(["source_api", "ts_code"], keep="last").sort_values(
            ["source_api", "ts_code"]
        )
    path = root / "index_coverage_snapshot.parquet"
    temp = path.with_name(f".{path.name}.{hashlib.sha1(str(path).encode()).hexdigest()[:8]}.tmp")
    try:
        frame.to_parquet(temp, index=False)
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)
    return {"path": str(path), "rows": int(len(frame))}


def validate_index_snapshot(data_dir: Path, scopes: Iterable[str] | None) -> dict[str, Any]:
    root = Path(data_dir).expanduser().resolve()
    selected = normalise_index_scopes(scopes)
    missing = [name for name in index_required_files(selected) if not (root / name).is_file()]
    if missing:
        raise IndexDataValidationError(f"指数候选快照缺少文件: {', '.join(missing)}")
    catalog_path = root / "index_catalog_df.parquet"
    catalog = pd.read_parquet(catalog_path)
    required_catalog = {"source_api", "ts_code", "name", "quote_source_api"}
    absent = required_catalog - set(catalog.columns)
    if absent:
        raise IndexDataValidationError(f"index_catalog_df.parquet 缺少列: {sorted(absent)}")
    if catalog[["source_api", "ts_code"]].isna().any(axis=None):
        raise IndexDataValidationError("指数目录包含空主键。")
    duplicates = int(
        count_true_kernel(
            np.ascontiguousarray(
                catalog.duplicated(["source_api", "ts_code"]).to_numpy(dtype=np.uint8)
            )
        )
    )
    if duplicates:
        raise IndexDataValidationError(f"指数目录包含 {duplicates} 个重复主键。")
    datasets: dict[str, Any] = {
        "index_catalog": {"rows": int(len(catalog)), "unique_codes": int(catalog["ts_code"].nunique())}
    }
    unavailable_history_sources: set[str] = set()
    history_source_by_file = {filename: source for source, filename in INDEX_HISTORY_FILES.items()}
    for scope in selected:
        for filename in INDEX_SCOPE_FILES[scope]:
            path = root / filename
            metadata = parquet.ParquetFile(path).metadata
            if metadata.num_rows <= 0:
                history_source = history_source_by_file.get(filename)
                catalog_has_source = bool(
                    history_source
                    and catalog["quote_source_api"].astype(str).eq(history_source).any()
                )
                if not history_source or catalog_has_source:
                    raise IndexDataValidationError(f"{filename} 为空。")
                required_history_columns = {"source_api", "ts_code", "trade_date"}
                missing_history_columns = required_history_columns - set(
                    parquet.ParquetFile(path).schema.names
                )
                if missing_history_columns:
                    raise IndexDataValidationError(
                        f"{filename} 空数据文件缺少列: {sorted(missing_history_columns)}"
                    )
                unavailable_history_sources.add(history_source)
                datasets[filename] = {"rows": 0, "status": "unavailable"}
                continue
            datasets[filename] = {"rows": metadata.num_rows, "status": "ready"}
    coverage = pd.read_parquet(root / "index_coverage_snapshot.parquet")
    required_coverage_columns = {"source_api", "ts_code", "rows", "first_date", "latest_date"}
    missing_coverage_columns = required_coverage_columns - set(coverage.columns)
    if missing_coverage_columns:
        raise IndexDataValidationError(
            f"index_coverage_snapshot.parquet 缺少列: {sorted(missing_coverage_columns)}"
        )
    if coverage.duplicated(["source_api", "ts_code"]).any():
        raise IndexDataValidationError("指数覆盖快照存在重复主键。")
    selected_history_sources = {
        source_api
        for source_api, filename in INDEX_HISTORY_FILES.items()
        if any(filename in INDEX_SCOPE_FILES[scope] for scope in selected)
    }
    covered_sources = set(coverage["source_api"].dropna().astype(str))
    absent_sources = selected_history_sources - unavailable_history_sources - covered_sources
    if absent_sources:
        raise IndexDataValidationError(
            f"指数覆盖快照缺少所选行情来源: {', '.join(sorted(absent_sources))}"
        )
    catalog_history_keys = set(
        zip(catalog["quote_source_api"].astype(str), catalog["ts_code"].astype(str))
    )
    coverage_history_keys = set(
        zip(coverage["source_api"].astype(str), coverage["ts_code"].astype(str))
    )
    catalog_codes = set(catalog["ts_code"].dropna().astype(str))
    unexpected_history_keys = {
        (source_api, code)
        for source_api, code in coverage_history_keys - catalog_history_keys
        if not (source_api == "index_dailybasic" and code in catalog_codes)
    }
    if unexpected_history_keys:
        sample = sorted(unexpected_history_keys)[:5]
        raise IndexDataValidationError(f"指数覆盖快照包含目录外代码: {sample}")
    datasets["index_coverage"] = {"rows": int(len(coverage))}
    return {"status": "passed", "scopes": selected, "datasets": datasets}


@lru_cache(maxsize=8)
def _read_small_tables_cached(
    catalog_path: str,
    catalog_mtime_ns: int,
    catalog_size: int,
    coverage_path: str,
    coverage_mtime_ns: int,
    coverage_size: int,
) -> pd.DataFrame:
    del catalog_mtime_ns, catalog_size, coverage_mtime_ns, coverage_size
    catalog = pd.read_parquet(catalog_path)
    coverage_file = Path(coverage_path)
    if coverage_file.exists() and coverage_file.stat().st_size:
        coverage = pd.read_parquet(coverage_file)
    else:
        coverage = pd.DataFrame(columns=[
            "source_api", "ts_code", "first_date", "latest_date", "rows", "stale_days",
            "domestic_trade_day_coverage", "source_file", "source_fingerprint",
        ])
    if "quote_source_api" not in catalog:
        catalog["quote_source_api"] = catalog["source_api"]
    merged = catalog.merge(
        coverage,
        left_on=["quote_source_api", "ts_code"],
        right_on=["source_api", "ts_code"],
        how="left",
        suffixes=("", "_coverage"),
    )
    merged["coverage_status"] = "missing"
    has_history = pd.to_numeric(merged.get("rows"), errors="coerce").fillna(0).gt(0)
    stale = pd.to_numeric(merged.get("stale_days"), errors="coerce").fillna(10_000).gt(10)
    merged.loc[has_history, "coverage_status"] = "ready"
    merged.loc[has_history & stale, "coverage_status"] = "stale"
    return merged


def load_index_catalog(data_dir: Path | None = None) -> pd.DataFrame:
    root = resolve_tushare_data_dir(data_dir)
    catalog_path = root / "index_catalog_df.parquet"
    coverage_path = root / "index_coverage_snapshot.parquet"
    if not catalog_path.exists():
        return pd.DataFrame()
    catalog_stat = catalog_path.stat()
    coverage_stat = coverage_path.stat() if coverage_path.exists() else None
    return _read_small_tables_cached(
        str(catalog_path), catalog_stat.st_mtime_ns, catalog_stat.st_size,
        str(coverage_path), coverage_stat.st_mtime_ns if coverage_stat else 0,
        coverage_stat.st_size if coverage_stat else 0,
    ).copy()


def _json_value(value: Any) -> Any:
    if value is None or bool(pd.isna(value)):
        return None
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [{key: _json_value(value) for key, value in row.items()} for row in frame.to_dict("records")]


def _parquet_date_bounds(path: Path, date_column: str | None) -> tuple[str | None, str | None]:
    if not date_column:
        return None, None
    source = parquet.ParquetFile(path)
    if date_column not in source.schema.names:
        return None, None
    column_index = source.schema.names.index(date_column)
    earliest: pd.Timestamp | None = None
    latest: pd.Timestamp | None = None
    for group_index in range(source.metadata.num_row_groups):
        statistics = source.metadata.row_group(group_index).column(column_index).statistics
        if statistics is None or not statistics.has_min_max:
            continue
        minimum = _normalise_date(statistics.min)
        maximum = _normalise_date(statistics.max)
        if minimum is not None and (earliest is None or minimum < earliest):
            earliest = minimum
        if maximum is not None and (latest is None or maximum > latest):
            latest = maximum
    return (
        earliest.strftime("%Y-%m-%d") if earliest is not None else None,
        latest.strftime("%Y-%m-%d") if latest is not None else None,
    )


def index_summary(data_dir: Path | None = None) -> dict[str, Any]:
    frame = load_index_catalog(data_dir)
    if frame.empty:
        return {
            "schema_version": 1, "status": "unavailable", "catalog_count": 0,
            "source_count": 0, "covered_count": 0, "coverage_rate": None,
            "latest_date": None,
            "missing_count": 0, "stale_count": 0, "sources": [], "datasets": [],
            "execution": _execution_audit(),
        }
    latest = pd.to_datetime(frame.get("latest_date"), errors="coerce").max()
    sources, source_count = _source_rows(frame["source_api"])
    catalog_count = int(len(frame))
    ready_count = int(
        count_true_kernel(
            np.ascontiguousarray(
                frame["coverage_status"].eq("ready").to_numpy(dtype=np.uint8)
            )
        )
    )
    covered_count = int(
        count_true_kernel(
            np.ascontiguousarray(
                frame["coverage_status"].ne("missing").to_numpy(dtype=np.uint8)
            )
        )
    )
    coverage_rate = float(coverage_ratio_kernel(covered_count, catalog_count))
    datasets: list[dict[str, Any]] = []
    root = resolve_tushare_data_dir(data_dir)
    for key, (filename, date_column, scope) in INDEX_DATASET_SPECS.items():
        path = root / filename
        exists = path.exists()
        rows = parquet.ParquetFile(path).metadata.num_rows if exists else 0
        earliest, dataset_latest = _parquet_date_bounds(path, date_column) if exists else (None, None)
        datasets.append(
            {
                "key": key,
                "scope": scope,
                "file": filename,
                "exists": exists,
                "status": "ready" if exists and rows > 0 else "missing",
                "rows": rows,
                "earliest_date": earliest,
                "latest_date": dataset_latest,
            }
        )
    return {
        "schema_version": 1,
        "status": "complete" if ready_count == catalog_count else "partial",
        "catalog_count": catalog_count,
        "source_count": source_count,
        "covered_count": covered_count,
        "coverage_rate": coverage_rate,
        "latest_date": None if pd.isna(latest) else latest.strftime("%Y-%m-%d"),
        "missing_count": int(
            count_true_kernel(
                np.ascontiguousarray(
                    frame["coverage_status"].eq("missing").to_numpy(dtype=np.uint8)
                )
            )
        ),
        "stale_count": int(
            count_true_kernel(
                np.ascontiguousarray(
                    frame["coverage_status"].eq("stale").to_numpy(dtype=np.uint8)
                )
            )
        ),
        "sources": sources,
        "datasets": datasets,
        "execution": _execution_audit(),
    }


def list_indices(
    *,
    query: str | None = None,
    source: str | None = None,
    market: str | None = None,
    category: str | None = None,
    active: str | None = None,
    coverage: str | None = None,
    page: int = 1,
    page_size: int = 20,
    data_dir: Path | None = None,
) -> dict[str, Any]:
    frame = load_index_catalog(data_dir)
    if frame.empty:
        return {"schema_version": 1, "status": "unavailable", "page": page, "page_size": page_size, "total": 0, "items": [], "filters": {}}
    working = frame.copy()
    if query:
        needle = str(query).strip().lower()
        haystack = (
            working.get("ts_code", "").fillna("").astype(str) + " "
            + working.get("name", "").fillna("").astype(str) + " "
            + working.get("publisher", "").fillna("").astype(str)
        ).str.lower()
        working = working[haystack.str.contains(needle, regex=False)]
    for column, value in (("source_api", source), ("market", market), ("category", category), ("coverage_status", coverage)):
        if value:
            working = working[working[column].fillna("").astype(str).eq(value)]
    if active in {"active", "inactive"}:
        active_mask = working.get("status", "active").fillna("active").astype(str).eq("active")
        working = working[active_mask if active == "active" else ~active_mask]
    total = int(len(working))
    start = (page - 1) * page_size
    columns = [
        "source_api", "ts_code", "name", "category", "market", "publisher", "list_date",
        "exp_date", "status", "quote_source_api", "first_date", "latest_date", "rows",
        "coverage_status", "stale_days", "domestic_trade_day_coverage",
    ]
    for column in columns:
        if column not in working:
            working[column] = None
    items = working.sort_values(["source_api", "ts_code"]).iloc[start:start + page_size][columns]
    filters = {
        key: sorted(value for value in frame[key].dropna().astype(str).unique().tolist() if value)
        for key in ("source_api", "market", "category", "coverage_status")
        if key in frame
    }
    return {
        "schema_version": 1, "status": "complete", "page": page, "page_size": page_size,
        "total": total, "items": _records(items), "filters": filters,
    }

"""Bounded ETF OHLCV I/O over one activated snapshot; never substitute NAV.

Arrow performs projection/predicate pushdown. Necessary decode/sort copies happen
once at this boundary; existing fixed-signature NJIT kernels align and adjust
the arrays, which are frozen before being shared with the timing graph.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds

from backend.custom_indicators.errors import ValidationError
from backend.market_data import read_active_manifest
from backend.research_series.numba_kernels import adjusted_price_kernel, align_values_kernel
from .numeric import MISSING_DAY

MAX_BARS = 12_000
OHLC = ("open", "high", "low", "close")
CANDLE_FILE = "etf_daily_candle_df.parquet"
FACTOR_FILE = "fund_adj_factor_df.parquet"
INFO_FILE = "etf_info_df.parquet"
CALENDAR_FILE = "trade_day_df.parquet"


@dataclass(frozen=True)
class ETFBars:
    dates: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    available_days: np.ndarray
    raw_open: np.ndarray
    raw_high: np.ndarray
    raw_low: np.ndarray
    raw_close: np.ndarray
    lineage: dict[str, Any]
    warnings: tuple[str, ...]


def _error(code: str, message: str) -> ValidationError:
    return ValidationError(f"TIMING_{code}", message)


def _parse_bound(value: str, label: str) -> date:
    try:
        return date.fromisoformat(value)
    except (ValueError, TypeError):
        raise _error("DATE_INVALID", f"{label}需要使用 YYYY-MM-DD 格式。") from None


def _identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"file": path.name, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _active(base: Path) -> tuple[Path, dict[str, Any]]:
    manifest = read_active_manifest(base)
    if manifest is None:
        raise _error("SNAPSHOT_REQUIRED", "请先在数据管理中激活 ETF 行情快照。")
    validation = manifest.get("validation")
    if not isinstance(validation, dict) or validation.get("status") != "passed":
        raise _error("SNAPSHOT_UNVALIDATED", "当前数据快照尚未通过数据验收。")
    root = (base / str(manifest["snapshot_dir"])).resolve()
    if root != base and base not in root.parents:
        raise _error("SNAPSHOT_INVALID", "当前数据快照不在配置的数据目录内。")
    return root, manifest


def _required_file(root: Path, manifest: dict, name: str) -> Path:
    path = root / name
    if name not in manifest.get("files", {}) or not path.is_file():
        description = "ETF 复权因子" if name == FACTOR_FILE else name
        raise _error("DATA_MISSING", f"活跃快照缺少{description}，请在数据管理中补齐并激活；不会使用净值替代行情。")
    if path.resolve().parent != root:
        raise _error("DATA_PATH_INVALID", "行情文件必须属于当前活跃快照。")
    return path


def _day_array(values: pa.ChunkedArray | pa.Array) -> np.ndarray:
    """Date decoding is an I/O boundary, not a numerical pandas computation."""
    if pa.types.is_integer(values.type) or pa.types.is_floating(values.type):
        # Vendor compact dates are calendar encodings, never epoch nanoseconds.
        values = pc.cast(values, pa.string())
    if pa.types.is_string(values.type) or pa.types.is_large_string(values.type):
        compact = pc.strptime(values, format="%Y%m%d", unit="s", error_is_null=True)
        dashed = pc.strptime(values, format="%Y-%m-%d", unit="s", error_is_null=True)
        parsed = pc.coalesce(compact, dashed)
        if parsed.null_count:
            raise _error("DATE_INVALID", "行情或交易日历中存在缺失、无效日期。")
        return np.array(parsed.to_numpy(zero_copy_only=False), dtype="datetime64[D]").view(np.int64)
    if not (pa.types.is_timestamp(values.type) or pa.types.is_date(values.type)) or values.null_count:
        raise _error("DATE_INVALID", "行情或交易日历日期字段类型无效或存在缺失。")
    return np.array(values.to_numpy(zero_copy_only=False), dtype="datetime64[D]").view(np.int64)


def _date_predicate(schema: pa.Schema, name: str, lower: date, upper: date):
    if name not in schema.names:
        raise _error("SCHEMA_INVALID", f"数据缺少日期字段 {name}。")
    dtype = schema.field(name).type
    field = ds.field(name)
    if pa.types.is_timestamp(dtype):
        lo = pa.scalar(pd.Timestamp(lower), type=dtype)
        hi = pa.scalar(pd.Timestamp(upper) + pd.Timedelta(days=1), type=dtype)
        return (field >= lo) & (field < hi)
    if pa.types.is_date(dtype):
        return (field >= pa.scalar(lower, type=dtype)) & (field <= pa.scalar(upper, type=dtype))
    if pa.types.is_string(dtype) or pa.types.is_large_string(dtype):
        # Canonical feeds use compact dates. Accept ISO date fixtures as well.
        compact = (pc.match_substring_regex(field, r"^[0-9]{8}$")
                   & (field >= lower.strftime("%Y%m%d")) & (field <= upper.strftime("%Y%m%d")))
        dashed = (pc.match_substring_regex(field, r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")
                  & (field >= lower.isoformat()) & (field <= upper.isoformat()))
        return compact | dashed
    raise _error("DATE_INVALID", "行情日期字段类型不受支持。")


def _read_table(path: Path, columns: list[str], predicate, *, limit: int = MAX_BARS) -> pa.Table:
    source = ds.dataset(path, format="parquet")
    missing = set(columns) - set(source.schema.names)
    if missing:
        raise _error("SCHEMA_INVALID", f"{path.name} 缺少字段：{', '.join(sorted(missing))}。")
    batches, count = [], 0
    for batch in source.scanner(columns=columns, filter=predicate, batch_size=4096, use_threads=False).to_batches():
        count += batch.num_rows
        if count > limit:
            raise _error("BAR_LIMIT", f"单产品研究最多支持 {MAX_BARS} 个交易日，请缩小日期区间。")
        batches.append(batch)
    return pa.Table.from_batches(batches, schema=pa.schema([source.schema.field(c) for c in columns]))


def _dated_table(path: Path, code: str | None, lower: date, upper: date, fields: tuple[str, ...]):
    schema = ds.dataset(path, format="parquet").schema
    date_field = "date" if "date" in schema.names else "trade_date"
    if date_field not in schema.names:
        raise _error("SCHEMA_INVALID", f"{path.name} 缺少交易日期。")
    predicate = _date_predicate(schema, date_field, lower, upper)
    if code is not None:
        if "ts_code" not in schema.names:
            raise _error("SCHEMA_INVALID", f"{path.name} 缺少产品代码。")
        predicate &= ds.field("ts_code") == code
    columns = [date_field, *fields]
    if "available_at" in schema.names:
        columns.append("available_at")
    table = _read_table(path, columns, predicate)
    if not table.num_rows:
        return table, np.empty(0, dtype=np.int64), columns
    table = table.sort_by([(date_field, "ascending")])
    days = _day_array(table[date_field])
    if np.any(days[1:] <= days[:-1]):
        raise _error("DUPLICATE_DATE", f"{path.name} 同一产品同日有多条记录，请先处理版本冲突。")
    return table, days, columns


def _numbers(table: pa.Table, field: str) -> np.ndarray:
    try:
        values = pc.cast(table[field], pa.float64(), safe=True).to_numpy(zero_copy_only=False)
        # Existing shared kernels require owned mutable C arrays at this boundary.
        output = np.array(values, dtype=np.float64, order="C", copy=True)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, ValueError, TypeError):
        raise _error("VALUE_INVALID", f"{field} 中存在无法转换为数值的内容。") from None
    if np.isinf(output).any():
        raise _error("VALUE_INVALID", f"{field} 中存在无穷值。")
    return output


def load_etf_bars(base: Path, code: str, start: str | None, end: str, price_basis: str = "hfq") -> ETFBars:
    """Load actual ETF daily bars once, on the uncompressed SSE open-day axis."""
    if not re.fullmatch(r"\d{6}\.(SH|SZ)", code):
        raise _error("PRODUCT_INVALID", "请选择带交易所后缀的 ETF 代码，例如 510300.SH。")
    if price_basis not in {"raw", "hfq", "qfq"}:
        raise _error("PRICE_BASIS_INVALID", "价格口径需要是不复权、后复权或前复权。")
    upper = _parse_bound(end, "截止日期")
    lower = _parse_bound(start, "开始日期") if start else date(1990, 1, 1)
    if upper < lower:
        raise _error("DATE_RANGE_INVALID", "开始日期不能晚于截止日期。")
    base = Path(base).expanduser().resolve()
    root, manifest = _active(base)
    names = [INFO_FILE, CALENDAR_FILE, CANDLE_FILE] + ([FACTOR_FILE] if price_basis != "raw" else [])
    paths = {name: _required_file(root, manifest, name) for name in names}
    before = [_identity(paths[name]) for name in names]
    schema = ds.dataset(paths[INFO_FILE], format="parquet").schema
    info_fields = [f for f in ("ts_code", "name", "instrument_type", "list_date", "delist_date", "qdii_type") if f in schema.names]
    info = _read_table(paths[INFO_FILE], info_fields, ds.field("ts_code") == code, limit=2).to_pylist()
    if len(info) != 1:
        raise _error("PRODUCT_NOT_FOUND", "ETF 基础信息缺失或重复，请先核对产品数据。")
    item = info[0]
    if str(item.get("instrument_type", "etf")).lower() != "etf":
        raise _error("ETF_ONLY", "当前择时研究只支持有真实日频行情的 ETF。")
    if str(item.get("qdii_type", "")).upper().strip() == "QDII" or "QDII" in str(item.get("name", "")).upper():
        raise _error("CROSS_MARKET_UNSUPPORTED", "跨境 ETF 需配置海外交易日历与汇率规则，当前版本请先选择境内 ETF。")
    if not item.get("list_date"):
        raise _error("LIST_DATE_MISSING", "ETF 缺少上市日期，无法确定可交易历史范围。")
    try:
        listed_day = _day_array(pa.array([item["list_date"]]))[0]
        listed = date.fromisoformat(str(np.datetime64(int(listed_day), "D")))
        delisted = None
        if item.get("delist_date"):
            delisted_day = _day_array(pa.array([item["delist_date"]]))[0]
            delisted = date.fromisoformat(str(np.datetime64(int(delisted_day), "D")))
    except (ValueError, TypeError, OverflowError, ValidationError):
        raise _error("LIST_DATE_INVALID", "ETF 上市或退市日期无效，请先核对产品基础信息。") from None
    lower = max(lower, listed)
    if delisted is not None:
        upper = min(upper, delisted)
    if lower > upper:
        raise _error("PRODUCT_DATE_RANGE", "所选区间不在该 ETF 的上市交易范围内。")

    calendar_schema = ds.dataset(paths[CALENDAR_FILE], format="parquet").schema
    predicate = (_date_predicate(calendar_schema, "cal_date", lower, upper)
                 & (ds.field("exchange") == "SSE"))
    # Read both open and closed dates once, so a missing calendar tail cannot
    # silently shorten a requested study or be mistaken for a holiday.
    calendar = _read_table(paths[CALENDAR_FILE], ["cal_date", "is_open"], predicate,
                           limit=MAX_BARS * 4).sort_by([("cal_date", "ascending")])
    calendar_days = _day_array(calendar["cal_date"])
    if (not calendar_days.size or calendar_days[0] != np.datetime64(lower, "D").astype(np.int64)
            or calendar_days[-1] != np.datetime64(upper, "D").astype(np.int64)
            or np.any(calendar_days[1:] - calendar_days[:-1] != 1)):
        raise _error("CALENDAR_COVERAGE_INCOMPLETE", "交易日历未完整覆盖所选上市区间，无法确认缺少的日期是休市还是数据缺失，请先补齐日历或缩短区间。")
    calendar = calendar.filter(pc.equal(calendar["is_open"], 1))
    days = _day_array(calendar["cal_date"])
    if not days.size or np.any(days[1:] <= days[:-1]):
        raise _error("CALENDAR_INVALID", "研究区间没有有效、唯一的 SSE 交易日历。")
    if days.size > MAX_BARS:
        raise _error("BAR_LIMIT", f"单产品研究最多支持 {MAX_BARS} 个交易日，请缩小日期区间。")
    table, observed_days, projected = _dated_table(paths[CANDLE_FILE], code, lower, upper, (*OHLC, "vol"))
    if not observed_days.size:
        raise _error("NO_BARS", "所选产品和区间没有真实 OHLC 行情，请调整区间或补齐数据。")
    if not np.isin(observed_days, days).all():
        raise _error("CALENDAR_MISMATCH", "ETF 行情日期与交易日历不一致，请补齐日历后再研究。")
    available = _day_array(table["available_at"]) if "available_at" in table.column_names else observed_days
    if np.any(available < observed_days):
        raise _error("AVAILABILITY_INVALID", "行情可得日期不能早于观察日期。")
    if np.any(available > np.datetime64(end, "D").astype(np.int64)):
        raise _error("AVAILABILITY_AFTER_CUTOFF", "所选历史行情存在截止日后才可得的记录，当前快照不能重建该截止日的数据版本。")
    values = {field: _numbers(table, field) for field in (*OHLC, "vol")}
    if observed_days[-1] != days[-1] or any(not np.isfinite(values[field][-1]) for field in OHLC):
        raise _error("PRICE_COVERAGE_INCOMPLETE", "ETF 有效 OHLC 行情未覆盖截止日前最后一个交易日，请补齐行情或缩短区间。")
    for field in OHLC:
        if np.any(np.isfinite(values[field]) & (values[field] <= 0)):
            raise _error("PRICE_INVALID", f"{field} 中存在非正价格，请先核对行情。")
    if np.any(np.isfinite(values["vol"]) & (values["vol"] < 0)):
        raise _error("VOLUME_INVALID", "成交量不能为负数。")
    if (np.any(values["high"] < values["low"]) or np.any(values["high"] < values["open"])
            or np.any(values["high"] < values["close"]) or np.any(values["low"] > values["open"])
            or np.any(values["low"] > values["close"])):
        raise _error("OHLC_INVALID", "最高价、最低价与开收盘价不一致，请先核对行情。")
    digest = hashlib.sha256()
    digest.update(json.dumps({"product": item, "cutoff": end, "start": start, "basis": price_basis}, sort_keys=True, default=str).encode())
    for array in (days, observed_days, available, *values.values()):
        digest.update(memoryview(array).cast("B"))
    raw = {field: align_values_kernel(days, observed_days, values[field]) for field in OHLC}
    volume = align_values_kernel(days, observed_days, values["vol"])
    release_days = align_values_kernel(days, observed_days, np.array(available, dtype=np.float64))
    # Dates are bounded integer days; NaN is converted only to the missing sentinel.
    available_days = np.full(days.size, MISSING_DAY, dtype=np.int64)
    present = np.isfinite(release_days)
    available_days[present] = release_days[present].astype(np.int64)
    adjusted = raw
    warnings = ["当前快照未保证历史修订版本和历史产品池，结果用于研究，不构成已验证的历史时点回放。",
                "日线行情按收盘后可得处理；信号最早在下一交易日执行，缺失交易日保留为空。"]
    factor_fields: list[str] = []
    anchor = None
    if price_basis != "raw":
        factors, factor_days, factor_fields = _dated_table(paths[FACTOR_FILE], code, lower, upper, ("adj_factor",))
        factor_values = _numbers(factors, "adj_factor")
        for array in (factor_days, factor_values):
            digest.update(memoryview(array).cast("B"))
        if "available_at" in factors.column_names:
            factor_available = _day_array(factors["available_at"])
            if (np.any(factor_available < factor_days)
                    or np.any(factor_available > np.datetime64(end, "D").astype(np.int64))):
                raise _error("ADJUSTMENT_AVAILABILITY_INVALID", "复权因子可得日期无效或晚于截止日，当前快照无法重建该截止日的数据版本。")
            if np.any(factor_available > factor_days):
                raise _error("ADJUSTMENT_NOT_CAUSAL", "复权因子晚于对应行情日才可得，不能用于历史当日信号；请使用具有当日可得证据的因子版本。")
            digest.update(memoryview(factor_available).cast("B"))
        observed_factors = align_values_kernel(observed_days, factor_days, factor_values)
        try:
            adjusted = {field: align_values_kernel(days, observed_days, adjusted_price_kernel(values[field], observed_factors, np.int64(price_basis == "qfq"))) for field in OHLC}
        except ValueError:
            raise _error("ADJUSTMENT_INCOMPLETE", "所选产品或区间缺少有效复权因子，请补齐因子或缩小日期区间；不会用 1 或未来因子补齐。") from None
        if price_basis == "qfq":
            anchor = {"date": str(np.datetime64(int(observed_days[-1]), "D")), "factor": float(observed_factors[-1])}
            warnings.append("前复权基准绑定本次截止区间内最后一条行情；绝对价格阈值依赖该基准，不能据此宣称历史实时可得。")
        warnings.append("复权 OHLC 用于相对收益研究，价格数值不是当时实际报价或可成交股数；原始行情另行保留。")
    else:
        warnings.append("当前使用不复权行情，分红和份额拆合可能形成价格跳变；含这些事件的收益需要单独核对。")
    if np.any(~present):
        warnings.append("区间包含缺失行情交易日，已保留为空；不会将后续行情提前为下一交易日。")
    if [_identity(paths[name]) for name in names] != before or _active(base) != (root, manifest):
        raise _error("DATA_CHANGED", "读取期间数据快照发生变化，请在数据同步结束后重试。")
    for array in (days, available_days, volume, *raw.values(), *adjusted.values()):
        array.setflags(write=False)
    lineage = {"snapshot": root.name, "snapshot_dir": str(manifest["snapshot_dir"]),
               "product_code": code, "product_name": str(item.get("name") or code),
               "source": "active_tushare_snapshot", "source_api": "fund_daily", "files": before,
               "projected_fields": {CANDLE_FILE: projected, FACTOR_FILE: factor_fields},
               "source_hash": "sha256:" + digest.hexdigest(), "hash_scope": "selected normalized product data and calendar",
               "cutoff": end, "price_basis": price_basis, "qfq_anchor": anchor,
               "actual_start": str(np.datetime64(int(days[0]), "D")),
               "actual_end": str(np.datetime64(int(days[-1]), "D")),
               "volume_unit": "lot", "calendar": "SSE", "revision_history_guaranteed": False,
               "availability": "market_close_date_only", "adjustment_availability": "unverified_revision_history",
               "legacy_fallback": False, "observation_count": int(observed_days.size),
               "calendar_count": int(days.size), "maximum_bars": MAX_BARS}
    return ETFBars(days, *(adjusted[field] for field in OHLC), volume, available_days,
                   *(raw[field] for field in OHLC), lineage, tuple(warnings))

"""Local ETF prices and public-fund NAV: source contracts and projected I/O."""

from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from .numba_kernels import adjusted_price_kernel, research_series_numba_execution_audit

ADJUSTMENT_FILE = 'fund_adj_factor_df.parquet'
ETF_ADJUSTED_FIELDS = {f'{field}_{basis}': (field, basis) for basis in ('hfq', 'qfq') for field in ('close', 'open', 'high', 'low')}


PRODUCT_SOURCES = {
    "etf": {
        "label": "ETF行情", "source_api": "fund_daily",
        "filename": "etf_daily_candle_df.parquet", "info_file": "etf_info_df.parquet",
        "date_field": "trade_date", "available_at_field": "trade_date", "default_field": "close",
        "fields": {
            "adj_nav": ("复权净值（仅事后分析）", "source_unit"),
            "close": ("收盘价（不复权）", "CNY"), "open": ("开盘价（不复权）", "CNY"),
            "high": ("最高价（不复权）", "CNY"), "low": ("最低价（不复权）", "CNY"),
            "pre_close": ("前收盘价", "CNY"), "change": ("涨跌额", "CNY"),
            "pct_chg": ("涨跌幅（%）", "percent"), "vol": ("成交量（手）", "lot"),
            "amount": ("成交额（千元）", "CNY_thousand"),
        },
    },
    "fund": {
        "label": "公募基金行情", "source_api": "fund_nav",
        "filename": "fund_nav_df.parquet", "info_file": "fund_info_df.parquet",
        "date_field": "nav_date", "available_at_field": "ann_date", "default_field": "unit_nav",
        "fields": {
            "unit_nav": ("单位净值", "source_unit"), "accum_nav": ("累计净值", "source_unit"),
            "adj_nav": ("复权净值（仅事后分析）", "source_unit"),
            "accum_div": ("累计分红", "source_unit"), "net_asset": ("净资产（元）", "CNY"),
            "total_netasset": ("合计净资产（元）", "CNY"),
        },
    },
}
PRODUCT_SOURCES['etf']['fields'].update({
    name: (f"{dict(close='收盘价', open='开盘价', high='最高价', low='最低价')[field]}（{'前' if basis == 'qfq' else '后'}复权·事后）", 'CNY')
    for name, (field, basis) in ETF_ADJUSTED_FIELDS.items()
})


class ProductSourceError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def product_source_spec(kind: str, field: str | None = None) -> dict[str, Any]:
    spec = PRODUCT_SOURCES[kind]
    if kind == "etf" and field == "adj_nav":
        return {**spec, "source_api": "fund_nav", "filename": "etf_daily_df.parquet",
                "date_field": "nav_date", "available_at_field": "ann_date", "default_field": "adj_nav"}
    return spec


def product_fields(path: Path, kind: str) -> list[dict[str, Any]]:
    schema = pq.ParquetFile(path).schema_arrow
    fields = [
        {"name": name, "label": label, "unit": unit, "dtype": "float64", "nullable": True}
        for name, (label, unit) in PRODUCT_SOURCES[kind]["fields"].items()
        if name in schema.names and (
            pa.types.is_integer(schema.field(name).type)
            or pa.types.is_floating(schema.field(name).type)
            or pa.types.is_decimal(schema.field(name).type)
        )
    ]
    if kind == 'etf':
        factor = adjustment_path(path.parent)
        for name, (base, _) in ETF_ADJUSTED_FIELDS.items():
            if base in {item['name'] for item in fields}:
                fields.append({'name': name, 'label': PRODUCT_SOURCES[kind]['fields'][name][0], 'unit': 'CNY', 'dtype': 'float64', 'nullable': True,
                               'available': factor is not None, 'unavailable_reason': None if factor else '缺少复权因子'})
    return fields


def adjustment_path(root: Path) -> Path | None:
    path = root / ADJUSTMENT_FILE
    manifest_path = root.parent / 'tushare_active.json'
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get('snapshot_dir') == root.name and ADJUSTMENT_FILE not in manifest.get('files', {}):
            return None
    return path if path.is_file() else None


def apply_product_adjustment(frame: pd.DataFrame, kind: str, field: str) -> pd.DataFrame:
    """Apply only after revision/as-of/range selection; never use a future QFQ anchor."""
    if kind != 'etf' or field not in ETF_ADJUSTED_FIELDS:
        return frame
    try:
        values = adjusted_price_kernel(np.ascontiguousarray(frame['value'], dtype=np.float64), np.ascontiguousarray(frame['adj_factor'], dtype=np.float64), np.int64(ETF_ADJUSTED_FIELDS[field][1] == 'qfq'))
    except ValueError as exc:
        raise ProductSourceError('ADJUSTMENT_COVERAGE_INCOMPLETE', '所选区间缺少有效复权因子，无法计算复权市价；请补齐因子或缩小日期区间。') from exc
    frame = frame.copy()
    frame['value'] = values
    frame[field] = values
    if ETF_ADJUSTED_FIELDS[field][1] == 'qfq' and not frame.empty:
        frame['available_at'] = frame['available_at'].max()
    return frame


@lru_cache(maxsize=8)
def _present_codes(filename: str, size: int, mtime_ns: int) -> frozenset[str]:
    # A bounded, code-only scan. Never load the full multi-million-row NAV table
    # merely to build a picker; cache by the actual file identity.
    codes: set[str] = set()
    for batch in pq.ParquetFile(filename).iter_batches(columns=["ts_code"], batch_size=65536):
        codes.update(value for value in batch.column(0).unique().to_pylist() if value)
    return frozenset(codes)


def present_product_codes(path: Path) -> frozenset[str]:
    stat = path.stat()
    return _present_codes(str(path.resolve()), stat.st_size, stat.st_mtime_ns)


def product_pit(kind: str, field: str | None = None) -> dict[str, Any]:
    spec = product_source_spec(kind, field)
    return {
        "supported": True, "observation_field": spec["date_field"],
        "available_at_field": spec["available_at_field"],
        "availability_status": "date_only_announcement" if spec["source_api"] == "fund_nav" else "date_only_market_close",
        "revision_history_guaranteed": False,
    }


def read_product_observations(root: Path, kind: str, parameters: Mapping[str, Any], mode: str) -> pd.DataFrame:
    """Read one product/field. Preserve NAV observation and announcement dates."""
    spec = product_source_spec(kind, parameters.get("field"))
    code = str(parameters.get("ts_code") or "").strip()
    if not code:
        raise ProductSourceError("MISSING_PRODUCT_CODE", "请先按名称或代码选择产品。")
    if parameters.get("source_api", spec["source_api"]) != spec["source_api"]:
        raise ProductSourceError("PRODUCT_SOURCE_MISMATCH", "产品类型与行情来源不匹配，请重新选择产品。")
    field = str(parameters.get("field") or spec["default_field"])
    adjusted = kind == 'etf' and field in ETF_ADJUSTED_FIELDS
    if adjusted and mode == 'realtime':
        raise ProductSourceError('ADJUSTED_PRICE_REALTIME_BLOCKED', '当前复权因子没有历史发布版本保障，复权市价仅支持事后分析；实时分析请使用不复权价格。')
    if field == "adj_nav" and mode == "realtime":
        alternative = "不复权市价" if kind == "etf" else "单位净值"
        raise ProductSourceError("ADJUSTED_NAV_REALTIME_BLOCKED", f"复权净值可能回改历史，仅支持事后分析；实时分析请选择{alternative}。")
    path = root / spec["filename"]
    if not path.is_file():
        raise ProductSourceError("PRODUCT_DATA_NOT_FOUND", "该产品行情尚未下载到所选快照。")
    dataset = ds.dataset(path, format="parquet")
    names = set(dataset.schema.names)
    if not {"ts_code", spec["date_field"]}.issubset(names):
        raise ProductSourceError("PRODUCT_SCHEMA_MISMATCH", "行情文件缺少产品代码或观察日期。")
    if field not in {item["name"] for item in product_fields(path, kind)}:
        raise ProductSourceError("PRODUCT_FIELD_UNAVAILABLE", "所选数值字段不在该行情文件中，请重新选择字段。")
    value_field = ETF_ADJUSTED_FIELDS[field][0] if adjusted else field
    columns = list(dict.fromkeys([
        "ts_code", spec["date_field"], value_field,
        *[name for name in (spec["available_at_field"], "available_at", "revision", "vintage") if name in names],
    ]))
    raw = dataset.to_table(columns=columns, filter=ds.field("ts_code") == code).to_pandas()
    if raw.empty:
        raise ProductSourceError("PRODUCT_SERIES_NOT_FOUND", "所选产品在当前快照中没有行情数据。")
    raw["observation_date"] = pd.to_datetime(raw[spec["date_field"]], errors="coerce", format="mixed")
    if raw["observation_date"].isna().any():
        raise ProductSourceError("INVALID_OBSERVATION_DATE", "产品行情中存在无效观察日期。")
    # NAV availability must never silently fall back to its valuation date.
    release_field = "available_at" if "available_at" in raw else spec["available_at_field"]
    released = pd.to_datetime(raw[release_field], errors="coerce", format="mixed") if release_field in raw else pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns]")
    unknown = released.isna() | (released < raw["observation_date"])
    if unknown.any() and mode == "realtime":
        raise ProductSourceError("PRODUCT_AVAILABILITY_UNKNOWN", "行情中存在缺失或无效的公告日期，无法用于实时分析；请补齐日期或改用事后分析。")
    raw["available_at"] = released.where(~unknown, raw["observation_date"])
    raw["availability_unknown"] = unknown
    raw["value"] = pd.to_numeric(raw[value_field], errors="coerce")
    if adjusted:
        factor_path = adjustment_path(root)
        if factor_path is None:
            raise ProductSourceError('ADJUSTMENT_DATA_NOT_FOUND', '当前快照缺少 ETF 复权因子，请先下载并发布复权因子数据，再重新选择 ETF。')
        with factor_path.open('rb') as handle:
            checksum = 'sha256:' + hashlib.file_digest(handle, 'sha256').hexdigest()
        if parameters.get('adjustment_checksum') and parameters['adjustment_checksum'] != checksum:
            raise ProductSourceError('ADJUSTMENT_CHECKSUM_MISMATCH', '复权因子内容与绑定版本不一致，请重新选择 ETF。')
        if parameters.get('snapshot_id') and not parameters.get('adjustment_checksum'):
            raise ProductSourceError('ADJUSTMENT_BINDING_REQUIRED', '当前 ETF 尚未绑定复权因子版本，请重新选择该 ETF 后预览。')
        factor_data = ds.dataset(factor_path, format='parquet')
        if not {'ts_code', 'trade_date', 'adj_factor'}.issubset(factor_data.schema.names):
            raise ProductSourceError('ADJUSTMENT_SCHEMA_INVALID', '复权因子缺少代码、交易日或因子字段。')
        factors = factor_data.to_table(columns=['trade_date', 'adj_factor'], filter=ds.field('ts_code') == code).to_pandas()
        factors['observation_date'] = pd.to_datetime(factors['trade_date'], format='mixed', errors='coerce')
        if factors['observation_date'].isna().any() or factors['observation_date'].duplicated().any():
            raise ProductSourceError('ADJUSTMENT_DATES_INVALID', '复权因子日期无效或重复，无法确定唯一日值。')
        raw = raw.merge(factors[['observation_date', 'adj_factor']], on='observation_date', how='left', validate='many_to_one')
        raw['adj_factor'] = pd.to_numeric(raw['adj_factor'], errors='coerce')
        raw.attrs['adjustment'] = {'source_api': 'fund_adj', 'source_file': ADJUSTMENT_FILE, 'checksum': checksum, 'basis': ETF_ADJUSTED_FIELDS[field][1], 'retrospective_only': True, 'execution': research_series_numba_execution_audit()}
    return raw

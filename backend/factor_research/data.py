"""Provider-neutral research boundary over the project's active canonical snapshot."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from backend.market_data import resolve_tushare_data_dir
from backend.custom_indicators.series_provider import load_product_variable_series_batch
from backend.custom_indicators.errors import ValidationError
from backend.series_quality import load_sse_open_dates

MISSING_DAY = np.iinfo(np.int64).max


def date_values(values):
    text = values.astype(str).str.replace(r"\.0$", "", regex=True)
    compact = text.str.fullmatch(r"\d{8}")
    parsed = pd.to_datetime(text.where(~compact), errors="coerce", format="mixed")
    parsed.loc[compact] = pd.to_datetime(text.loc[compact], format="%Y%m%d", errors="coerce")
    return parsed


def file_record(path: Path):
    if not path.is_file():
        raise ValidationError("FACTOR_DATA_MISSING", f"缺少研究数据 {path.name}，请先在数据同步中补齐。")
    stat = path.stat()
    return {"dataset": path.name, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def records(root: Path, names):
    return [file_record(root / name) for name in sorted(set(names))]


def snapshot_directory(base: Path):
    return resolve_tushare_data_dir(base, strict=True)


def product_catalog(root: Path, kind: str, query: str = "", limit: int = 50):
    if kind == "stock":
        return []
    name = "etf_info_df.parquet" if kind == "etf" else "fund_info_df.parquet"
    path = root / name
    if not path.exists():
        return []
    schema = pq.ParquetFile(path).schema_arrow.names
    columns = [c for c in ("ts_code", "name", "fund_type", "list_date", "found_date", "delist_date", "benchmark") if c in schema]
    frame = pq.read_table(path, columns=columns).to_pandas()
    if query:
        frame = frame[frame["ts_code"].astype(str).str.contains(query, case=False, regex=False)
                      | frame["name"].astype(str).str.contains(query, case=False, regex=False)]
    frame = frame.drop_duplicates("ts_code").head(limit)
    return frame.astype(object).where(pd.notna(frame), None).to_dict("records")


def dataset_last_date(root: Path, filename="etf_daily_df.parquet"):
    path = root / filename
    if not path.exists():
        return None
    parquet = pq.ParquetFile(path)
    names = parquet.schema_arrow.names
    field = "date" if "date" in names else "nav_date"
    index = names.index(field)
    maxima = []
    for group in range(parquet.metadata.num_row_groups):
        stat = parquet.metadata.row_group(group).column(index).statistics
        if stat is not None and stat.has_min_max:
            maxima.append(str(stat.max))
    if not maxima:
        return None
    parsed = date_values(pd.Series(maxima)).dropna()
    return parsed.max().strftime("%Y-%m-%d") if len(parsed) else None


def load_index(root: Path, code: str, dates: pd.DatetimeIndex):
    path = root / "index_daily_df.parquet"
    frame = pq.read_table(path, columns=["ts_code", "trade_date", "close"],
                          filters=[("ts_code", "==", code)]).to_pandas()
    if frame.empty:
        raise ValidationError("FACTOR_INDEX_UNAVAILABLE", f"本地没有指数 {code} 的日线数据。")
    frame["date"] = date_values(frame["trade_date"])
    frame = frame[frame["date"].notna()]
    if frame["date"].duplicated().any():
        raise ValidationError("FACTOR_INDEX_AMBIGUOUS", f"指数 {code} 同日有重复记录，请先处理数据来源冲突。")
    return np.ascontiguousarray(frame.set_index("date")["close"].reindex(dates).to_numpy(dtype=np.float64))


def decision_indices(dates, calendar, start, frequency="monthly"):
    """Calendar scheduling only; incomplete weeks/months are not false period ends."""
    eligible = dates[dates >= pd.Timestamp(start)]
    if frequency == "daily":
        return np.ascontiguousarray(dates.get_indexer(eligible), dtype=np.int64)
    next_days = pd.Series(calendar, index=calendar).shift(-1).reindex(eligible)
    if frequency == "monthly":
        boundary = next_days.dt.to_period("M").to_numpy() != eligible.to_period("M").to_numpy()
    elif frequency == "weekly":
        boundary = next_days.dt.to_period("W-FRI").to_numpy() != eligible.to_period("W-FRI").to_numpy()
    else:
        raise ValidationError("FACTOR_SIGNAL_FREQUENCY", "信号频率必须为日、周或月。")
    selected = eligible[boundary & next_days.notna().to_numpy()]
    return np.ascontiguousarray(dates.get_indexer(selected), dtype=np.int64)


def load_panel(base: Path, kind: str, targets: list[str], start: str, end: str, benchmark=None, indices=(), signal_frequency="monthly"):
    if kind == "stock":
        raise ValidationError("STOCK_DATA_NOT_READY", "当前股票数据只有基础信息；需补齐行情、复权和财务时点数据。")
    root = snapshot_directory(base)
    nav_name = "etf_daily_df.parquet" if kind == "etf" else "fund_nav_df.parquet"
    info_name = "etf_info_df.parquet" if kind == "etf" else "fund_info_df.parquet"
    names = [nav_name, info_name, "trade_day_df.parquet"]
    if benchmark and benchmark["kind"] == "etf":
        names += ["etf_daily_df.parquet", "etf_info_df.parquet"]
    if indices or (benchmark and benchmark["kind"] == "index"):
        names += ["index_daily_df.parquet"]
    if benchmark and benchmark["kind"] == "index":
        catalog_path = root / "index_catalog_df.parquet"
        if not catalog_path.exists():
            raise ValidationError("FACTOR_BENCHMARK_BASIS", "缺少指数目录，无法核对价格/全收益口径。")
        names.append("index_catalog_df.parquet")
        catalog = pq.read_table(catalog_path, columns=["ts_code", "name"],
                                filters=[("ts_code", "==", benchmark["code"])]).to_pandas()
        if catalog.empty:
            raise ValidationError("FACTOR_BENCHMARK_BASIS", "基准代码不在本地指数目录中。")
        is_return = catalog["name"].astype(str).str.contains("全收益|全回报|净收益|净回报|total return|net return", case=False, regex=True).any()
        if bool(is_return) != (benchmark["return_basis"] == "total_return_index"):
            raise ValidationError("FACTOR_BENCHMARK_BASIS", "基准代码与所选价格/全收益口径不匹配，请核对指数名称。")
    before = records(root, names)
    calendar = load_sse_open_dates(root / "trade_day_df.parquet")
    lower = pd.Timestamp(start) - pd.Timedelta(days=1100)
    dates = calendar[(calendar >= lower) & (calendar <= pd.Timestamp(end))]
    if len(dates) < 3 or len(dates) > 6000:
        raise ValidationError("FACTOR_CALENDAR_RANGE", "研究区间需要 3–6000 个 SSE 交易日（含预热样本）。")
    days = np.ascontiguousarray(dates.to_numpy(dtype="datetime64[D]").astype(np.int64))
    loaded = load_product_variable_series_batch(kind, targets, ["adjusted_nav"], data_dir=root, as_of=end)
    prices = np.full((len(dates), len(targets)), np.nan)
    available = np.full((len(dates), len(targets)), MISSING_DAY, dtype=np.int64)
    identities, quality, lineage = [], [], []
    meta = pq.read_table(root / info_name).to_pandas().drop_duplicates("ts_code").set_index("ts_code")
    for a, code in enumerate(targets):
        product = loaded[code]
        if product.identity.ts_code != code:
            raise ValidationError("FACTOR_CANONICAL_CODE", "请使用带市场后缀的完整产品代码。")
        info = meta.loc[code] if code in meta.index else {}
        is_qdii = str(info.get("qdii_type", "")).strip().upper() == "QDII" or "QDII" in str(info.get("name", "")).upper()
        if is_qdii:
            raise ValidationError("FACTOR_CROSS_MARKET_NOT_READY", "当前研究使用中国交易日历与人民币口径；QDII 需先配置跨市场日历与汇率适配。")
        identities.append({"product_id": code, "code": code, "name": product.identity.name,
                           "kind": kind, "fund_type": str(info.get("fund_type", ""))})
        frame = product.frame
        if frame.empty or "_available_date_nav" not in frame.columns:
            quality.append({"code": code, "observations": 0, "reason": "缺少复权净值或公告日期"})
            continue
        frame = frame.set_index("date").reindex(dates)
        price = frame["adjusted_nav"].to_numpy(dtype=np.float64)
        availability = frame["_available_date_nav"].to_numpy(dtype="datetime64[D]").astype(np.int64)
        availability[availability == np.iinfo(np.int64).min] = MISSING_DAY
        # NAV before ETF listing / fund inception is not an eligible product history.
        if code in meta.index:
            row = meta.loc[code]
            birth = row.get("list_date") if kind == "etf" else row.get("found_date")
            if birth is not None and pd.notna(birth):
                parsed = date_values(pd.Series([birth])).iloc[0]
                if pd.notna(parsed):
                    price[dates < parsed] = np.nan
            death = row.get("delist_date")
            if death is not None and pd.notna(death):
                parsed = date_values(pd.Series([death])).iloc[0]
                if pd.notna(parsed):
                    price[dates > parsed] = np.nan
        prices[:, a] = price
        available[:, a] = availability
        quality.append({"code": code, "observations": len(product.frame),
                        "data_latest_date": product.data_latest_date, "warnings": product.warnings})
        lineage.extend(product.lineage)
    bm = np.full(len(dates), np.nan)
    if benchmark:
        if benchmark["kind"] == "index":
            bm = load_index(root, benchmark["code"], dates)
        elif kind == "etf" and benchmark["code"] in targets:
            bm = prices[:, targets.index(benchmark["code"])].copy()
        else:
            code = benchmark["code"]
            result = load_product_variable_series_batch("etf", [code], ["adjusted_nav"], data_dir=root, as_of=end)[code]
            if result.frame.empty:
                raise ValidationError("FACTOR_BENCHMARK_UNAVAILABLE", "所选 ETF 基准没有可用复权净值。")
            bm = np.ascontiguousarray(result.frame.set_index("date")["adjusted_nav"].reindex(dates).to_numpy(dtype=np.float64))
            lineage.extend(result.lineage)
    index_values = np.column_stack([load_index(root, code, dates) for code in indices]) if indices else np.empty((len(dates), 0))
    if records(root, names) != before or snapshot_directory(base) != root:
        raise ValidationError("FACTOR_DATA_CHANGED", "读取期间数据快照发生变化，请在同步结束后重试。")
    generation = hashlib.sha256(json.dumps(before, sort_keys=True).encode()).hexdigest()
    decisions = decision_indices(dates, calendar, start, signal_frequency)
    return {
        "dates": dates, "days": days, "prices": np.ascontiguousarray(prices),
        "available": np.ascontiguousarray(available), "benchmark": np.ascontiguousarray(bm),
        "indices": np.ascontiguousarray(index_values), "decisions": decisions,
        "identities": identities, "quality": quality,
        "lineage": {"snapshot": root.name, "generation": generation, "files": before, "sources": lineage,
                    "availability": "ann_date strictly before signal date; no vintage guarantee",
                    "calendar": "SSE", "price_basis": "adjusted_nav"},
    }

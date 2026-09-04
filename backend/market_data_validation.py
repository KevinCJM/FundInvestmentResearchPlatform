"""Read-only validation gates for a completed Tushare snapshot."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np
import pandas as pd
import pyarrow.parquet as parquet

try:
    from backend.instrument_analytics_numba import count_true_kernel, nav_metrics_kernel
    from backend.series_quality import (
        PeriodWindowQuality,
        adjusted_nav_anomaly_dates,
        assess_period_window,
        load_sse_open_dates,
    )
except ModuleNotFoundError:  # pragma: no cover - backend/ direct execution
    from instrument_analytics_numba import count_true_kernel, nav_metrics_kernel
    from series_quality import (
        PeriodWindowQuality,
        adjusted_nav_anomaly_dates,
        assess_period_window,
        load_sse_open_dates,
    )


INFO_FILES = {
    "etf": "etf_info_df.parquet",
    "fund": "fund_info_df.parquet",
}
HISTORY_FILES = {
    "etf_nav": ("etf_daily_df.parquet", "nav"),
    "fund_nav": ("fund_nav_df.parquet", "nav"),
    "etf_candle": ("etf_daily_candle_df.parquet", "candle"),
}
MIN_HISTORY_COVERAGE = {"etf_nav": 0.90, "fund_nav": 0.90, "etf_candle": 0.90}
MIN_METRICS_TO_NAV_COVERAGE = 0.99
MIN_BASELINE_ROW_RETENTION = 0.80
MAX_DATASET_LAG_DAYS = 10
MAX_FUTURE_DAYS = 1
REQUIRED_COLUMNS = {
    "info": {"ts_code", "name", "status_code"},
    "nav": {"ts_code", "date", "adj_nav"},
    "candle": {"ts_code", "date", "close", "vol", "amount"},
    "snapshot": {
        "instrument_type",
        "ts_code",
        "latest_date",
        "observation_count",
        "nav_source_fingerprint",
    },
}


class SnapshotValidationError(RuntimeError):
    """Raised when a candidate snapshot fails a data-integrity gate."""


def _count_mask(values: Any) -> int:
    mask = np.ascontiguousarray(np.asarray(values, dtype=np.uint8).reshape(-1))
    return int(count_true_kernel(mask))


def _source_fingerprint(path: Path) -> str:
    stat = path.stat()
    raw = f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def _schema_columns(path: Path) -> set[str]:
    return set(parquet.ParquetFile(path).schema.names)


def _require_columns(path: Path, contract: Literal["info", "nav", "candle", "snapshot"]) -> None:
    missing = REQUIRED_COLUMNS[contract] - _schema_columns(path)
    if missing:
        raise SnapshotValidationError(f"{path.name} 缺少必要列: {sorted(missing)}")


def _validate_info(path: Path, kind: str, sample_size: int) -> tuple[dict[str, Any], set[str]]:
    _require_columns(path, "info")
    frame = pd.read_parquet(path, columns=["ts_code", "name", "status_code"])
    codes = frame["ts_code"].fillna("").astype(str).str.strip()
    valid_codes = codes[codes.ne("")]
    duplicate_count = _count_mask(valid_codes.duplicated())
    if len(valid_codes) != len(frame) or duplicate_count:
        raise SnapshotValidationError(
            f"{path.name} 标的键异常: empty={len(frame) - len(valid_codes)}, duplicate={duplicate_count}"
        )
    return (
        {
            "kind": kind,
            "rows": int(len(frame)),
            "unique_codes": int(valid_codes.nunique()),
            "sample_codes": sorted(valid_codes.tolist())[:sample_size],
        },
        set(valid_codes.tolist()),
    )


def _batch_quality(frame: pd.DataFrame, contract: str) -> tuple[int, int, int]:
    if contract == "nav":
        values = pd.to_numeric(frame["adj_nav"], errors="coerce")
        missing_values = _count_mask(values.isna())
        invalid_values = _count_mask(values.notna() & (~np.isfinite(values) | values.le(0)))
        return invalid_values, 0, missing_values
    close = pd.to_numeric(frame["close"], errors="coerce")
    volume = pd.to_numeric(frame["vol"], errors="coerce")
    amount = pd.to_numeric(frame["amount"], errors="coerce")
    missing_values = _count_mask(close.isna())
    invalid_values = _count_mask(close.notna() & (~np.isfinite(close) | close.le(0)))
    invalid_activity = _count_mask(
        (np.isfinite(volume) & volume.lt(0)) | (np.isfinite(amount) & amount.lt(0))
    )
    return invalid_values, invalid_activity, missing_values


def _validate_history(
    path: Path,
    contract: Literal["nav", "candle"],
    sample_size: int,
    *,
    sample_codes: Iterable[str] = (),
) -> tuple[dict[str, Any], set[str], dict[str, pd.DataFrame]]:
    _require_columns(path, contract)
    source = parquet.ParquetFile(path)
    available_columns = set(source.schema.names)
    optional_columns = {"unit_nav", "accum_nav"} if contract == "nav" else set()
    columns = sorted(REQUIRED_COLUMNS[contract] | (optional_columns & available_columns))
    row_count = duplicate_count = out_of_order_count = invalid_key_count = 0
    invalid_value_count = invalid_activity_count = missing_value_count = 0
    last_key: tuple[str, pd.Timestamp] | None = None
    latest_date: pd.Timestamp | None = None
    earliest_date: pd.Timestamp | None = None
    codes: set[str] = set()
    selected = set(sample_codes)
    sample_parts: dict[str, list[pd.DataFrame]] = {code: [] for code in selected}

    for batch in source.iter_batches(batch_size=65_536, columns=columns, use_threads=False):
        frame = batch.to_pandas()
        frame["ts_code"] = frame["ts_code"].fillna("").astype(str).str.strip()
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        invalid_keys = frame["ts_code"].eq("") | frame["date"].isna()
        invalid_key_count += _count_mask(invalid_keys)
        valid = frame.loc[~invalid_keys]
        row_count += int(len(frame))
        invalid_values, invalid_activity, missing_values = _batch_quality(frame, contract)
        invalid_value_count += invalid_values
        invalid_activity_count += invalid_activity
        missing_value_count += missing_values
        if valid.empty:
            continue
        codes.update(valid["ts_code"].unique().tolist())
        if selected:
            sampled = valid[valid["ts_code"].isin(selected)]
            for code, group in sampled.groupby("ts_code", sort=False):
                sample_parts[str(code)].append(group.copy())
        batch_latest = valid["date"].max()
        batch_earliest = valid["date"].min()
        latest_date = batch_latest if latest_date is None or batch_latest > latest_date else latest_date
        earliest_date = (
            batch_earliest
            if earliest_date is None or batch_earliest < earliest_date
            else earliest_date
        )
        for code, date in zip(valid["ts_code"], valid["date"]):
            key = (str(code), pd.Timestamp(date))
            if last_key is not None:
                duplicate_count += int(key == last_key)
                out_of_order_count += int(key < last_key)
            last_key = key

    failures = {
        "invalid_keys": invalid_key_count,
        "duplicate_keys": duplicate_count,
        "out_of_order_keys": out_of_order_count,
        "invalid_values": invalid_value_count,
        "negative_activity": invalid_activity_count,
    }
    nonzero = {name: value for name, value in failures.items() if value}
    if nonzero:
        raise SnapshotValidationError(f"{path.name} 历史数据质量失败: {nonzero}")
    if row_count != source.metadata.num_rows:
        raise SnapshotValidationError(
            f"{path.name} 行数不一致: scanned={row_count}, metadata={source.metadata.num_rows}"
        )
    samples = {
        code: pd.concat(parts, ignore_index=True)
        for code, parts in sample_parts.items()
        if parts
    }
    return (
        {
            "rows": row_count,
            "unique_codes": len(codes),
            "latest_date": latest_date.strftime("%Y-%m-%d") if latest_date is not None else None,
            "earliest_date": earliest_date.strftime("%Y-%m-%d") if earliest_date is not None else None,
            "missing_values": missing_value_count,
            "sample_codes": sorted(codes)[:sample_size],
        },
        codes,
        samples,
    )


def _validate_metrics_snapshot(
    path: Path, info_codes: dict[str, set[str]], sample_size: int
) -> tuple[dict[str, Any], pd.DataFrame]:
    _require_columns(path, "snapshot")
    frame = pd.read_parquet(path)
    duplicate_count = _count_mask(
        frame.duplicated(subset=["instrument_type", "ts_code"])
    )
    if duplicate_count:
        raise SnapshotValidationError(f"{path.name} 存在 {duplicate_count} 个重复产品键。")
    unsupported_types = set(frame["instrument_type"].dropna().astype(str)) - {"etf", "fund"}
    if unsupported_types:
        raise SnapshotValidationError(f"{path.name} 包含未知 instrument_type: {sorted(unsupported_types)}")
    unexpected: dict[str, int] = {}
    for kind in ("etf", "fund"):
        codes = set(frame.loc[frame["instrument_type"].eq(kind), "ts_code"].astype(str))
        unexpected[kind] = len(codes - info_codes.get(kind, set()))
        required_sample = min(sample_size, len(info_codes.get(kind, set())))
        if len(codes) < required_sample:
            raise SnapshotValidationError(
                f"{path.name} 的 {kind} 指标行不足抽样门槛: rows={len(codes)}, required={required_sample}"
            )
    if any(unexpected.values()):
        raise SnapshotValidationError(f"{path.name} 包含基础信息外的代码: {unexpected}")
    numeric = frame.select_dtypes(include=[np.number])
    infinity_count = (
        _count_mask(np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan)))
        if not numeric.empty
        else 0
    )
    if infinity_count:
        raise SnapshotValidationError(f"{path.name} 包含 {infinity_count} 个 Infinity。")
    report = {
        "rows": int(len(frame)),
        "by_kind": {
            kind: _count_mask(frame["instrument_type"].eq(kind))
            for kind in ("etf", "fund")
        },
        "as_of": pd.to_datetime(frame.get("as_of"), errors="coerce").max().strftime("%Y-%m-%d")
        if "as_of" in frame and pd.to_datetime(frame["as_of"], errors="coerce").notna().any()
        else None,
    }
    return report, frame


def _spread_sample(values: Iterable[str], size: int) -> list[str]:
    ordered = sorted(set(values))
    if len(ordered) <= size:
        return ordered
    positions = np.linspace(0, len(ordered) - 1, num=size, dtype=int)
    return [ordered[int(position)] for position in positions]


def _independent_nav_metrics(
    frame: pd.DataFrame,
    open_dates: pd.DatetimeIndex | None = None,
) -> dict[str, Any]:
    working = frame.copy()
    working["date"] = pd.to_datetime(working["date"], errors="coerce")
    working["adj_nav"] = pd.to_numeric(working["adj_nav"], errors="coerce")
    working = (
        working.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["date", "adj_nav"])
        .loc[lambda item: item["adj_nav"].gt(0)]
        .sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
    )
    if working.empty:
        return {}
    latest_date = pd.Timestamp(working.iloc[-1]["date"])
    anomaly_dates = adjusted_nav_anomaly_dates(working)

    def window(months: int) -> tuple[pd.DataFrame, PeriodWindowQuality]:
        target = latest_date - pd.DateOffset(months=months)
        quality = assess_period_window(
            working["date"],
            target_date=target,
            effective_date=latest_date,
            open_dates=open_dates,
            anomaly_dates=anomaly_dates,
        )
        if quality.anchor_date is None:
            return working.iloc[0:0], quality
        selected = working[
            working["date"].between(quality.anchor_date, quality.effective_date)
        ]
        return selected, quality

    windows = {
        label: window(months)
        for label, months in {"1m": 1, "3m": 3, "1y": 12, "3y": 36}.items()
    }

    def nav_values(label: str) -> np.ndarray:
        selected, quality = windows[label]
        if not quality.complete:
            return np.empty(0, dtype=np.float64)
        return np.ascontiguousarray(
            selected["adj_nav"].to_numpy(dtype=np.float64), dtype=np.float64
        )

    three_year = windows["3y"][0] if windows["3y"][1].complete else working.iloc[0:0]
    three_year_elapsed_days = (
        max((three_year.iloc[-1]["date"] - three_year.iloc[0]["date"]).days, 1)
        if len(three_year) >= 2
        else 0
    )
    metrics = nav_metrics_kernel(
        nav_values("1m"),
        nav_values("3m"),
        nav_values("1y"),
        nav_values("3y"),
        three_year_elapsed_days,
    )

    def optional_metric(index: int) -> float | None:
        value = float(metrics[index])
        return value if np.isfinite(value) else None

    return {
        "first_date": pd.Timestamp(working.iloc[0]["date"]),
        "latest_date": latest_date,
        "observation_count": int(len(working)),
        "return_1m": optional_metric(0),
        "return_3m": optional_metric(1),
        "return_1y": optional_metric(2),
        "return_3y": optional_metric(3),
        "annual_volatility_1y": optional_metric(4),
        "max_drawdown_3y": optional_metric(5),
        "sharpe_1y": optional_metric(6),
        "calmar_3y": optional_metric(7),
    }


def _assert_metric_equal(code: str, metric: str, expected: Any, actual: Any) -> None:
    if metric.endswith("date"):
        expected_date = pd.to_datetime(expected, errors="coerce")
        actual_date = pd.to_datetime(actual, errors="coerce")
        if pd.isna(expected_date) != pd.isna(actual_date) or (
            pd.notna(expected_date) and pd.Timestamp(expected_date) != pd.Timestamp(actual_date)
        ):
            raise SnapshotValidationError(
                f"指标抽样复算不一致: {code} {metric}, expected={expected_date}, actual={actual_date}"
            )
        return
    expected_missing = expected is None or bool(pd.isna(expected))
    actual_missing = actual is None or bool(pd.isna(actual))
    if expected_missing or actual_missing:
        if expected_missing != actual_missing:
            raise SnapshotValidationError(
                f"指标抽样复算不一致: {code} {metric}, expected={expected}, actual={actual}"
            )
        return
    if metric == "observation_count":
        equal = int(expected) == int(actual)
    else:
        equal = bool(np.isclose(float(expected), float(actual), rtol=1e-8, atol=1e-10))
    if not equal:
        raise SnapshotValidationError(
            f"指标抽样复算不一致: {code} {metric}, expected={expected}, actual={actual}"
        )


def _verify_metric_samples(
    snapshot: pd.DataFrame,
    nav_samples: dict[str, dict[str, pd.DataFrame]],
    candle_samples: dict[str, pd.DataFrame],
    requested: dict[str, list[str]],
    open_dates: pd.DatetimeIndex | None = None,
) -> dict[str, int]:
    verified = {"etf": 0, "fund": 0, "premium_discount": 0}
    comparable = (
        "first_date",
        "latest_date",
        "observation_count",
        "return_1m",
        "return_3m",
        "return_1y",
        "return_3y",
        "annual_volatility_1y",
        "max_drawdown_3y",
        "sharpe_1y",
        "calmar_3y",
    )
    for kind in ("etf", "fund"):
        for code in requested[kind]:
            frame = nav_samples[kind].get(code)
            if frame is None:
                raise SnapshotValidationError(f"指标快照代码缺少净值历史: {kind}/{code}")
            actual = _independent_nav_metrics(frame, open_dates)
            row = snapshot[
                snapshot["instrument_type"].eq(kind) & snapshot["ts_code"].astype(str).eq(code)
            ].iloc[0]
            for metric in comparable:
                if metric in row.index:
                    _assert_metric_equal(code, metric, row.get(metric), actual.get(metric))
            verified[kind] += 1

            if kind != "etf" or code not in candle_samples or "premium_discount_latest" not in row.index:
                continue
            nav = frame.copy()
            nav["date"] = pd.to_datetime(nav["date"], errors="coerce")
            nav["unit_nav"] = pd.to_numeric(nav.get("unit_nav"), errors="coerce")
            candle = candle_samples[code].copy()
            candle["date"] = pd.to_datetime(candle["date"], errors="coerce")
            candle["close"] = pd.to_numeric(candle["close"], errors="coerce")
            common = candle.merge(nav[["date", "unit_nav"]], on="date", how="inner").dropna()
            common = common[(common["close"] > 0) & (common["unit_nav"] > 0)].sort_values("date")
            expected_premium = None if common.empty else float(
                common.iloc[-1]["close"] / common.iloc[-1]["unit_nav"] - 1.0
            )
            _assert_metric_equal(
                code, "premium_discount_latest", row.get("premium_discount_latest"), expected_premium
            )
            verified["premium_discount"] += 1
    return verified


def validate_tushare_snapshot(
    snapshot_dir: Path,
    *,
    sample_size: int = 30,
    strict: bool = False,
    baseline_dir: Path | None = None,
) -> dict[str, Any]:
    """Run the complete read-only promotion gate and return an audit report."""

    snapshot = snapshot_dir.expanduser().resolve()
    if not snapshot.is_dir():
        raise SnapshotValidationError(f"快照目录不存在: {snapshot}")
    datasets: dict[str, Any] = {}
    info_codes: dict[str, set[str]] = {}
    for kind, filename in INFO_FILES.items():
        path = snapshot / filename
        if not path.exists():
            raise SnapshotValidationError(f"缺少数据文件: {filename}")
        datasets[f"{kind}_info"], info_codes[kind] = _validate_info(path, kind, sample_size)
        if strict and len(info_codes[kind]) < sample_size:
            raise SnapshotValidationError(
                f"{filename} 代码数低于真实验收门槛: "
                f"codes={len(info_codes[kind])}, required={sample_size}"
            )
    metrics_path = snapshot / "instrument_metrics_snapshot.parquet"
    if not metrics_path.exists():
        raise SnapshotValidationError("缺少数据文件: instrument_metrics_snapshot.parquet")
    datasets["analytics_snapshot"], metrics_frame = _validate_metrics_snapshot(
        metrics_path, info_codes, sample_size
    )
    requested_samples = {
        kind: _spread_sample(
            metrics_frame.loc[metrics_frame["instrument_type"].eq(kind), "ts_code"].astype(str),
            sample_size,
        )
        for kind in ("etf", "fund")
    }
    nav_samples: dict[str, dict[str, pd.DataFrame]] = {"etf": {}, "fund": {}}
    candle_samples: dict[str, pd.DataFrame] = {}
    history_codes_by_dataset: dict[str, set[str]] = {}
    for dataset, (filename, contract) in HISTORY_FILES.items():
        path = snapshot / filename
        if not path.exists():
            raise SnapshotValidationError(f"缺少数据文件: {filename}")
        kind = "fund" if dataset == "fund_nav" else "etf"
        selected = requested_samples[kind] if dataset != "etf_candle" else requested_samples["etf"]
        datasets[dataset], history_codes, sampled_frames = _validate_history(
            path, contract, sample_size, sample_codes=selected
        )
        unknown_codes = history_codes - info_codes[kind]
        if unknown_codes:
            raise SnapshotValidationError(
                f"{filename} 包含 {len(unknown_codes)} 个 {kind} 基础信息外代码。"
            )
        history_codes_by_dataset[dataset] = history_codes
        denominator = len(info_codes[kind])
        coverage = len(history_codes) / denominator if denominator else 0.0
        datasets[dataset]["code_coverage"] = coverage
        required_coverage = MIN_HISTORY_COVERAGE[dataset]
        if coverage < required_coverage:
            raise SnapshotValidationError(
                f"{filename} 代码覆盖率过低: coverage={coverage:.4f}, "
                f"required={required_coverage:.4f}"
            )
        if dataset == "etf_nav":
            nav_samples["etf"] = sampled_frames
        elif dataset == "fund_nav":
            nav_samples["fund"] = sampled_frames
        else:
            candle_samples = sampled_frames

    latest_dates = {
        dataset: pd.to_datetime(datasets[dataset].get("latest_date"), errors="coerce")
        for dataset in HISTORY_FILES
    }
    valid_latest = [value for value in latest_dates.values() if not pd.isna(value)]
    if len(valid_latest) != len(HISTORY_FILES):
        raise SnapshotValidationError("历史数据集缺少有效最新日期。")
    newest = max(valid_latest)
    future_limit = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize() + pd.Timedelta(
        days=MAX_FUTURE_DAYS
    )
    for dataset, latest in latest_dates.items():
        if latest > future_limit:
            raise SnapshotValidationError(
                f"{HISTORY_FILES[dataset][0]} 包含未来最新日期: {latest.date()}"
            )
    for dataset, latest in latest_dates.items():
        lag_days = int((newest - latest).days)
        datasets[dataset]["lag_to_snapshot_latest_days"] = lag_days
        if lag_days > MAX_DATASET_LAG_DAYS:
            raise SnapshotValidationError(
                f"{HISTORY_FILES[dataset][0]} 最新日期落后其他数据集 {lag_days} 天。"
            )

    metrics_codes = {
        kind: set(
            metrics_frame.loc[
                metrics_frame["instrument_type"].eq(kind), "ts_code"
            ].dropna().astype(str)
        )
        for kind in ("etf", "fund")
    }
    for kind, nav_dataset in (("etf", "etf_nav"), ("fund", "fund_nav")):
        nav_codes = history_codes_by_dataset[nav_dataset]
        coverage = len(metrics_codes[kind] & nav_codes) / len(nav_codes) if nav_codes else 0.0
        datasets["analytics_snapshot"].setdefault("nav_code_coverage", {})[kind] = coverage
        if coverage < MIN_METRICS_TO_NAV_COVERAGE:
            raise SnapshotValidationError(
                f"instrument_metrics_snapshot.parquet 的 {kind} 净值代码覆盖率过低: "
                f"coverage={coverage:.4f}, required={MIN_METRICS_TO_NAV_COVERAGE:.4f}"
            )

    baseline_report: dict[str, Any] = {}
    if baseline_dir is not None:
        baseline = baseline_dir.expanduser().resolve()
        if baseline != snapshot and baseline.is_dir():
            files = {**INFO_FILES, **{name: value[0] for name, value in HISTORY_FILES.items()}}
            for dataset, filename in files.items():
                baseline_path = baseline / filename
                candidate_path = snapshot / filename
                if not baseline_path.exists() or not candidate_path.exists():
                    continue
                baseline_rows = parquet.ParquetFile(baseline_path).metadata.num_rows
                candidate_rows = parquet.ParquetFile(candidate_path).metadata.num_rows
                retained = candidate_rows / baseline_rows if baseline_rows else 1.0
                baseline_report[dataset] = {
                    "baseline_rows": baseline_rows,
                    "candidate_rows": candidate_rows,
                    "retained_ratio": retained,
                }
                if retained < MIN_BASELINE_ROW_RETENTION:
                    raise SnapshotValidationError(
                        f"{filename} 相对当前活跃数据行数退化过大: "
                        f"retained={retained:.4f}, required={MIN_BASELINE_ROW_RETENTION:.4f}"
                    )
    source_paths = {
        "etf": snapshot / "etf_daily_df.parquet",
        "fund": snapshot / "fund_nav_df.parquet",
    }
    for kind, source_path in source_paths.items():
        expected = _source_fingerprint(source_path)
        observed = set(
            metrics_frame.loc[
                metrics_frame["instrument_type"].eq(kind), "nav_source_fingerprint"
            ].dropna().astype(str)
        )
        if observed != {expected}:
            raise SnapshotValidationError(
                f"instrument_metrics_snapshot.parquet 的 {kind} 源文件指纹不一致。"
            )
    candle_fingerprint_column = metrics_frame.get("candle_source_fingerprint")
    if candle_fingerprint_column is not None:
        observed_candle = set(
            metrics_frame.loc[
                metrics_frame["instrument_type"].eq("etf"), "candle_source_fingerprint"
            ].dropna().astype(str)
        )
        if observed_candle and observed_candle != {
            _source_fingerprint(snapshot / "etf_daily_candle_df.parquet")
        }:
            raise SnapshotValidationError(
                "instrument_metrics_snapshot.parquet 的 ETF 行情源文件指纹不一致。"
            )
        if strict and not observed_candle:
            raise SnapshotValidationError(
                "instrument_metrics_snapshot.parquet 缺少 ETF 行情源文件指纹。"
            )
    elif strict:
        raise SnapshotValidationError(
            "instrument_metrics_snapshot.parquet 缺少 candle_source_fingerprint 列。"
        )
    datasets["analytics_snapshot"]["independent_metric_samples"] = _verify_metric_samples(
        metrics_frame,
        nav_samples,
        candle_samples,
        requested_samples,
        load_sse_open_dates(snapshot / "trade_day_df.parquet"),
    )
    inventory = {
        path.name: {"size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
        for path in snapshot.iterdir()
        if path.is_file() and path.suffix == ".parquet"
    }
    return {
        "status": "passed",
        "snapshot_dir": str(snapshot),
        "datasets": datasets,
        "baseline": baseline_report,
        "inventory": inventory,
    }

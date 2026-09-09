from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

try:
    from backend.backtest_numba import (
        annual_metrics_kernel,
        backtest_numba_execution_audit,
        cumulative_returns_kernel,
    )
    from backend.fit_numba import (
        assemble_weight_matrix_kernel,
        class_consistency_kernel,
        class_nav_corr_metrics_kernel,
        finite_matrix_mask_kernel,
        fit_numba_execution_audit,
        normalize_weight_matrix_kernel,
        returns_from_nav_kernel,
        rolling_correlation_kernel,
        weighted_returns_kernel,
    )
    from backend.market_data import resolve_market_data_file
except ModuleNotFoundError:  # pragma: no cover - backend/ direct execution
    from backtest_numba import (
        annual_metrics_kernel,
        backtest_numba_execution_audit,
        cumulative_returns_kernel,
    )
    from fit_numba import (
        assemble_weight_matrix_kernel,
        class_consistency_kernel,
        class_nav_corr_metrics_kernel,
        finite_matrix_mask_kernel,
        fit_numba_execution_audit,
        normalize_weight_matrix_kernel,
        returns_from_nav_kernel,
        rolling_correlation_kernel,
        weighted_returns_kernel,
    )
    from market_data import resolve_market_data_file


# The PIT package is imported bare-first, unlike the modules above. Startup
# warms `pit.audit`'s dataset cache; reaching the same code through
# `backend.pit.audit` would load a second copy with a cold cache and turn the
# first strict-mode run into a full 37M-row rescan.
try:
    from pit.catalog import RUN_MODE_RESEARCH, RUN_MODE_STRICT, RUN_MODES
    from pit.context import PitContextError, parse_as_of
except ModuleNotFoundError:  # pragma: no cover - imported as a backend.* module
    from backend.pit.catalog import RUN_MODE_RESEARCH, RUN_MODE_STRICT, RUN_MODES
    from backend.pit.context import PitContextError, parse_as_of


ANNUAL_METRIC_NAMES = (
    "cumulative",
    "volatility",
    "annualReturn",
    "annualVolatility",
    "sharpe",
    "maxDrawdown",
    "calmar",
)


@dataclass
class ETFSpec:
    code: str
    name: str
    weight: float


@dataclass
class ClassSpec:
    id: str
    name: str
    etfs: List[ETFSpec]


def _code_nosfx(value: str) -> str:
    normalized = (value or "").strip()
    return normalized.split(".")[0] if "." in normalized else normalized


# The column that says when a NAV row became public. Filtering on `nav_date`
# instead is look-ahead: 98% of rows in etf_daily_df are announced at least one
# day after the value date, and the tail reaches 25 days.
NAV_AVAILABILITY_FIELD = "ann_date"
NAV_EVENT_FIELD = "date"
_NAV_BASE_COLUMNS = ["ts_code", "name", NAV_EVENT_FIELD, "adj_nav"]


@dataclass(frozen=True)
class NavLoad:
    """NAV rows plus the audit trail of what the point-in-time cut removed."""

    frame: pd.DataFrame
    lineage: Dict[str, object]


def _read_nav_file(path: Path, requested_codes: list[str], requested_names: list[str]) -> tuple[pd.DataFrame, bool]:
    """Read one NAV parquet, taking `ann_date` along when the file has it."""

    # Read the footer only. Materialising the frame to inspect `.columns` would
    # pull 30M+ rows off disk before the column projection and row filters ever
    # apply, turning a sub-second load into eight seconds.
    try:
        available_columns = set(pq.read_schema(path).names)
    except Exception:  # noqa: BLE001 - schema probe must never break the read
        available_columns = set()
    has_availability = NAV_AVAILABILITY_FIELD in available_columns
    columns = list(_NAV_BASE_COLUMNS) + ([NAV_AVAILABILITY_FIELD] if has_availability else [])
    try:
        filters = []
        if requested_codes:
            filters.append([("ts_code", "in", requested_codes)])
        if requested_names:
            filters.append([("name", "in", requested_names)])
        frame = pd.read_parquet(path, columns=columns, engine="pyarrow", filters=filters or None)
    except Exception:  # noqa: BLE001 - fall back to a full read then mask
        frame = pd.read_parquet(path, columns=columns)
        if requested_codes or requested_names:
            code_mask = frame["ts_code"].astype(str).isin(requested_codes) if requested_codes else False
            name_mask = frame["name"].astype(str).isin(requested_names) if requested_names else False
            frame = frame[code_mask | name_mask]
    return frame, has_availability


def load_adj_nav_pit(
    data_dir: Path,
    codes: Iterable[str],
    names: Iterable[str],
    *,
    as_of: object = None,
    run_mode: str = RUN_MODE_RESEARCH,
) -> NavLoad:
    """Read NAV rows that were publicly known on `as_of`.

    `as_of=None` keeps the historical behaviour (everything on disk) but says so
    in the lineage rather than pretending a cut happened.  In STRICT_PIT a file
    without `ann_date`, or a row whose `ann_date` is missing, is refused instead
    of silently falling back to the value date.
    """

    mode = str(run_mode or RUN_MODE_RESEARCH).strip().upper() or RUN_MODE_RESEARCH
    if mode not in RUN_MODES:
        raise PitContextError(f"不支持的运行模式：{run_mode}")
    cutoff = parse_as_of(as_of)
    if mode == RUN_MODE_STRICT and cutoff is None:
        raise PitContextError("严格 PIT 模式必须指定研究日。")

    paths = [
        resolve_market_data_file("etf_daily_df.parquet", data_dir),
        resolve_market_data_file("fund_nav_df.parquet", data_dir),
    ]
    existing_paths = [path for path in paths if path.exists()]
    if not existing_paths:
        raise FileNotFoundError("data/etf_daily_df.parquet 与 data/fund_nav_df.parquet 均不存在")

    requested_codes = [code for code in set(codes) if code]
    requested_names = [name for name in set(names) if name]
    frames: list[pd.DataFrame] = []
    sources: list[dict[str, object]] = []
    for path in existing_paths:
        frame, has_availability = _read_nav_file(path, requested_codes, requested_names)
        sources.append({"file": path.name, "has_ann_date": has_availability, "rows": int(len(frame))})
        if mode == RUN_MODE_STRICT and not has_availability and not frame.empty:
            raise PitContextError(
                f"严格 PIT 模式要求净值数据带公告日：{path.name} 缺少 {NAV_AVAILABILITY_FIELD} 列。"
            )
        if not frame.empty:
            if not has_availability:
                frame = frame.copy()
                frame[NAV_AVAILABILITY_FIELD] = pd.NaT
            frames.append(frame)

    lineage: Dict[str, object] = {
        "as_of": cutoff.strftime("%Y-%m-%d") if cutoff is not None else None,
        "as_of_applied": cutoff is not None,
        "run_mode": mode,
        "availability_field": NAV_AVAILABILITY_FIELD,
        "sources": sources,
        "rows_before_cut": 0,
        "rows_after_cut": 0,
        "rows_dropped_by_as_of": 0,
        "rows_without_announcement": 0,
        "announcement_fallback": False,
        "warnings": [],
    }
    if not frames:
        return NavLoad(pd.DataFrame(columns=_NAV_BASE_COLUMNS), lineage)

    result = pd.concat(frames, ignore_index=True)
    result[NAV_EVENT_FIELD] = pd.to_datetime(result[NAV_EVENT_FIELD], errors="coerce")
    announced = pd.to_datetime(result[NAV_AVAILABILITY_FIELD], errors="coerce")
    if announced.isna().all() and result[NAV_AVAILABILITY_FIELD].notna().any():
        announced = pd.to_datetime(result[NAV_AVAILABILITY_FIELD], errors="coerce", format="%Y%m%d")
    result["available_date"] = announced.dt.normalize()
    result = result.dropna(subset=[NAV_EVENT_FIELD, "adj_nav"])

    missing = int(result["available_date"].isna().sum())
    lineage["rows_before_cut"] = int(len(result))
    lineage["rows_without_announcement"] = missing
    if missing:
        if mode == RUN_MODE_STRICT:
            result = result[result["available_date"].notna()].copy()
            lineage["warnings"].append(
                f"严格 PIT 模式下丢弃 {missing} 行没有公告日的净值。"
            )
        else:
            # Research mode keeps going, but the value date is an optimistic
            # stand-in for the announcement date and the caller must be told.
            lineage["announcement_fallback"] = True
            lineage["warnings"].append(
                f"{missing} 行净值缺少 {NAV_AVAILABILITY_FIELD}，已按净值日期近似可得时间；该部分不具备严格时点证明。"
            )
            result["available_date"] = result["available_date"].fillna(result[NAV_EVENT_FIELD])

    if cutoff is not None:
        before = int(len(result))
        result = result[result["available_date"] <= cutoff].copy()
        lineage["rows_dropped_by_as_of"] = before - int(len(result))

    # Sorting by availability before de-duplicating keeps the latest revision
    # that was actually knowable on as_of, not the latest one that exists today.
    result = result.sort_values(["ts_code", NAV_EVENT_FIELD, "available_date"]).drop_duplicates(
        subset=["ts_code", NAV_EVENT_FIELD], keep="last"
    )
    for column in _NAV_BASE_COLUMNS:
        if column not in result.columns:
            raise ValueError(f"parquet 缺少必要列：{column}")
    lineage["rows_after_cut"] = int(len(result))
    return NavLoad(result.sort_values(NAV_EVENT_FIELD), lineage)


def _load_adj_nav(
    data_dir: Path,
    codes: Iterable[str],
    names: Iterable[str],
    *,
    as_of: object = None,
    run_mode: str = RUN_MODE_RESEARCH,
) -> pd.DataFrame:
    """Frame-only view of :func:`load_adj_nav_pit` for callers that ignore lineage."""

    return load_adj_nav_pit(data_dir, codes, names, as_of=as_of, run_mode=run_mode).frame


_LAST_NAV_LINEAGE: Dict[str, object] = {}


def _remember_nav_lineage(lineage: Dict[str, object]) -> None:
    _LAST_NAV_LINEAGE.clear()
    _LAST_NAV_LINEAGE.update(lineage)


def last_nav_lineage() -> Dict[str, object]:
    """Audit trail of the most recent NAV load in this process.

    ponytail: process-local, so it is only meaningful immediately after the call
    that produced it. Thread NavLoad explicitly if a caller ever needs the
    lineage of an older load.
    """

    return dict(_LAST_NAV_LINEAGE)


def _returns_from_adj_nav(series: pd.Series) -> pd.Series:
    ordered = series.sort_index()
    nav = np.ascontiguousarray(ordered.to_numpy(dtype=np.float64).reshape((-1, 1)))
    values = returns_from_nav_kernel(nav)[:, 0]
    return pd.Series(values, index=ordered.index[1:], dtype=float)


def _pick_series(df: pd.DataFrame, code: str, name: str) -> Optional[pd.Series]:
    code_without_suffix = _code_nosfx(code)
    ts_codes = df["ts_code"].astype(str)
    names = df["name"].astype(str)
    subset = df[
        (ts_codes == code)
        | (ts_codes.str.split(".").str[0] == code_without_suffix)
        | (names == name)
    ]
    if subset.empty and code_without_suffix:
        subset = df[ts_codes.str.contains(code_without_suffix, na=False)]
    if subset.empty:
        return None
    result = subset.groupby("date")["adj_nav"].last().sort_index()
    return None if result.empty else result


def _returns_wide(df: pd.DataFrame, start_date: pd.Timestamp) -> pd.DataFrame:
    pivot = df.pivot_table(
        index="date", columns="ts_code", values="adj_nav", aggfunc="last"
    ).sort_index()
    if len(pivot.index) < 2:
        return pd.DataFrame(index=pivot.index[:0], columns=pivot.columns, dtype=float)
    values = returns_from_nav_kernel(
        np.ascontiguousarray(pivot.to_numpy(dtype=np.float64))
    )
    result = pd.DataFrame(values, index=pivot.index[1:], columns=pivot.columns)
    return result.loc[result.index >= start_date].dropna(axis=0, how="any")


def _map_to_ts(df: pd.DataFrame, available: list[str], code: str, name: str) -> Optional[str]:
    if code in available:
        return code
    code_without_suffix = _code_nosfx(code)
    if code_without_suffix in available:
        return code_without_suffix
    by_name = df[df["name"].astype(str) == name]
    if not by_name.empty:
        candidate = str(by_name.iloc[0]["ts_code"])
        if candidate in available:
            return candidate
    partial = [candidate for candidate in available if code_without_suffix and code_without_suffix in candidate]
    return partial[0] if partial else None


def _class_returns(
    df: pd.DataFrame,
    classes: List[ClassSpec],
    start_date: pd.Timestamp,
) -> pd.DataFrame:
    asset_returns = _returns_wide(df, start_date)
    available = list(asset_returns.columns.astype(str))
    asset_positions = {name: idx for idx, name in enumerate(available)}
    asset_indexes: list[int] = []
    class_indexes: list[int] = []
    raw_values: list[float] = []
    for class_idx, class_spec in enumerate(classes):
        for etf in class_spec.etfs:
            if etf.weight is None:
                continue
            column = _map_to_ts(df, available, str(etf.code), str(etf.name))
            if column is None:
                continue
            asset_indexes.append(asset_positions[column])
            class_indexes.append(class_idx)
            raw_values.append(float(etf.weight))
    raw_weights = assemble_weight_matrix_kernel(
        np.ascontiguousarray(np.asarray(asset_indexes, dtype=np.int64)),
        np.ascontiguousarray(np.asarray(class_indexes, dtype=np.int64)),
        np.ascontiguousarray(np.asarray(raw_values, dtype=np.float64)),
        len(available),
        len(classes),
    )
    normalized, valid = normalize_weight_matrix_kernel(np.ascontiguousarray(raw_weights))
    valid_indexes = np.flatnonzero(valid).astype(np.int64)
    if valid_indexes.size == 0:
        raise ValueError("没有可用的大类收益率：请检查权重或数据匹配。")
    result = weighted_returns_kernel(
        np.ascontiguousarray(asset_returns.to_numpy(dtype=np.float64)),
        np.ascontiguousarray(normalized[:, valid_indexes]),
    )
    class_names = [classes[int(idx)].name for idx in valid_indexes]
    return pd.DataFrame(result, index=asset_returns.index, columns=class_names)


def compute_classes_nav(
    data_dir: Path,
    classes: List[ClassSpec],
    start_date: pd.Timestamp,
    *,
    as_of: object = None,
    run_mode: str = RUN_MODE_RESEARCH,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    requested_codes = [etf.code for item in classes for etf in item.etfs if etf.weight is not None]
    requested_names = [etf.name for item in classes for etf in item.etfs if etf.weight is not None]
    loaded = load_adj_nav_pit(data_dir, requested_codes, requested_names, as_of=as_of, run_mode=run_mode)
    data = loaded.frame
    _remember_nav_lineage(loaded.lineage)
    class_returns = _class_returns(data, classes, start_date)
    if class_returns.empty:
        raise ValueError("没有可用的大类收益率：请检查权重或数据匹配。")
    nav, corr, metrics = class_nav_corr_metrics_kernel(
        np.ascontiguousarray(class_returns.to_numpy(dtype=np.float64)), 252.0
    )
    labels = class_returns.columns
    return (
        pd.DataFrame(nav, index=class_returns.index, columns=labels),
        pd.DataFrame(corr, index=labels, columns=labels),
        pd.DataFrame(
            metrics,
            index=labels,
            columns=["年化收益率", "年化波动率", "夏普比率", "99%VaR(日)", "99%ES(日)", "最大回撤", "卡玛比率"],
        ),
    )


def compute_nav_performance_payload(nav: pd.DataFrame) -> dict[str, object]:
    """Serialize NJIT-computed cumulative and calendar-year NAV metrics."""

    values = np.ascontiguousarray(nav.to_numpy(dtype=np.float64))
    cumulative = cumulative_returns_kernel(values)
    years, annual = annual_metrics_kernel(
        values,
        np.ascontiguousarray(nav.index.year.to_numpy(dtype=np.int64)),
        252.0,
    )

    def optional(value: float) -> float | None:
        parsed = float(value)
        return parsed if np.isfinite(parsed) else None

    labels = list(nav.columns.astype(str))
    annual_payload: dict[str, object] = {
        "years": [int(year) for year in years],
        "series": {},
    }
    annual_series = annual_payload["series"]
    assert isinstance(annual_series, dict)
    for column_index, label in enumerate(labels):
        by_year: dict[str, dict[str, float | None]] = {}
        for year_index, year in enumerate(years):
            base = column_index * len(ANNUAL_METRIC_NAMES)
            by_year[str(int(year))] = {
                metric_name: optional(annual[year_index, base + metric_index])
                for metric_index, metric_name in enumerate(ANNUAL_METRIC_NAMES)
            }
        annual_series[label] = by_year
    return {
        "cumulative_returns": {
            label: optional(cumulative[index]) for index, label in enumerate(labels)
        },
        "annual_metrics": annual_payload,
        "execution": {
            "fit_analytics": fit_numba_execution_audit(),
            "performance_metrics": backtest_numba_execution_audit(),
        },
    }


def compute_rolling_corr(
    data_dir: Path,
    etfs: List[ETFSpec],
    start_date: pd.Timestamp,
    window: int,
    target_code: str,
    target_name: str,
    *,
    as_of: object = None,
    run_mode: str = RUN_MODE_RESEARCH,
) -> Tuple[pd.DatetimeIndex, Dict[str, np.ndarray], List[Dict[str, float]]]:
    if window <= 1:
        raise ValueError("window 必须 > 1")
    data = _load_adj_nav(
        data_dir,
        [item.code for item in etfs],
        [item.name for item in etfs],
        as_of=as_of,
        run_mode=run_mode,
    )
    returns = _returns_wide(data, start_date)
    columns = list(returns.columns.astype(str))
    target = _map_to_ts(data, columns, target_code, target_name)
    if target is None or target not in returns.columns:
        raise ValueError("未能找到研究对象的收益序列")
    rolling, metrics = rolling_correlation_kernel(
        np.ascontiguousarray(returns.to_numpy(dtype=np.float64)), columns.index(target), int(window)
    )
    series_map: Dict[str, np.ndarray] = {}
    metrics_list: List[Dict[str, float]] = []
    for column_idx, column in enumerate(columns):
        if column == target:
            continue
        row = metrics[column_idx]
        series_map[column] = rolling[:, column_idx]
        metrics_list.append({
            "name": column, "sum": float(row[1]), "mean": float(row[2]),
            "median": float(row[3]), "std": float(row[4]), "skew": float(row[5]),
            "kurtosis": float(row[6]),
        })
    return returns.index, series_map, metrics_list


def compute_rolling_corr_classes(
    data_dir: Path,
    classes: List[ClassSpec],
    start_date: pd.Timestamp,
    window: int,
    target_class_name: str,
    *,
    as_of: object = None,
    run_mode: str = RUN_MODE_RESEARCH,
) -> Tuple[pd.DatetimeIndex, Dict[str, np.ndarray], List[Dict[str, float]]]:
    if window <= 1:
        raise ValueError("window 必须 > 1")
    data = _load_adj_nav(
        data_dir,
        [etf.code for item in classes for etf in item.etfs],
        [etf.name for item in classes for etf in item.etfs],
        as_of=as_of,
        run_mode=run_mode,
    )
    returns = _class_returns(data, classes, start_date)
    labels = list(returns.columns.astype(str))
    if target_class_name not in labels:
        raise ValueError("研究对象大类未找到")
    rolling, metrics = rolling_correlation_kernel(
        np.ascontiguousarray(returns.to_numpy(dtype=np.float64)), labels.index(target_class_name), int(window)
    )
    series_map: Dict[str, np.ndarray] = {}
    metrics_list: List[Dict[str, float]] = []
    for class_idx, label in enumerate(labels):
        if label == target_class_name:
            continue
        row = metrics[class_idx]
        series_map[label] = rolling[:, class_idx]
        metrics_list.append({
            "name": label, "overall": float(row[0]), "mean": float(row[2]),
            "median": float(row[3]), "std": float(row[4]), "skew": float(row[5]),
            "kurtosis": float(row[6]),
        })
    return returns.index, series_map, metrics_list


def serialize_rolling_correlation_payload(
    index: pd.DatetimeIndex,
    series_map: Dict[str, np.ndarray],
    metrics: List[Dict[str, float]],
) -> Dict[str, object]:
    """Serialize rolling results using NJIT finite masks and a policy audit."""

    series_names = list(series_map)
    series_values = np.empty((len(index), len(series_names)), dtype=np.float64)
    for column, name in enumerate(series_names):
        values = np.ascontiguousarray(series_map[name], dtype=np.float64)
        if values.size != len(index):
            raise ValueError("滚动相关序列长度与日期轴不一致")
        series_values[:, column] = values
    series_values = np.ascontiguousarray(series_values)
    series_finite = finite_matrix_mask_kernel(series_values)
    safe_series = {
        name: [
            float(series_values[row, column])
            if int(series_finite[row, column]) == 1
            else None
            for row in range(series_values.shape[0])
        ]
        for column, name in enumerate(series_names)
    }

    metric_names = list(
        dict.fromkeys(
            key
            for item in metrics
            for key in item
            if key != "name"
        )
    )
    metric_values = np.full(
        (len(metrics), len(metric_names)),
        np.nan,
        dtype=np.float64,
    )
    for row, item in enumerate(metrics):
        for column, name in enumerate(metric_names):
            try:
                metric_values[row, column] = float(item.get(name, np.nan))
            except (TypeError, ValueError, OverflowError):
                metric_values[row, column] = np.nan
    metric_finite = finite_matrix_mask_kernel(np.ascontiguousarray(metric_values))
    safe_metrics = [
        {
            "name": str(item.get("name") or ""),
            **{
                name: (
                    float(metric_values[row, column])
                    if int(metric_finite[row, column]) == 1
                    else None
                )
                for column, name in enumerate(metric_names)
            },
        }
        for row, item in enumerate(metrics)
    ]
    return {
        "dates": [value.strftime("%Y-%m-%d") for value in index],
        "series": safe_series,
        "metrics": safe_metrics,
        "execution": fit_numba_execution_audit(),
    }


def compute_class_consistency(
    data_dir: Path,
    classes: List[ClassSpec],
    start_date: pd.Timestamp,
    *,
    as_of: object = None,
    run_mode: str = RUN_MODE_RESEARCH,
) -> List[Dict[str, float]]:
    data = _load_adj_nav(
        data_dir,
        [etf.code for item in classes for etf in item.etfs],
        [etf.name for item in classes for etf in item.etfs],
        as_of=as_of,
        run_mode=run_mode,
    )
    asset_returns = _returns_wide(data, start_date)
    available = list(asset_returns.columns.astype(str))
    output: List[Dict[str, float]] = []
    for class_spec in classes:
        columns: list[str] = []
        for etf in class_spec.etfs:
            column = _map_to_ts(data, available, str(etf.code), str(etf.name))
            if column is not None and column not in columns:
                columns.append(column)
        if len(columns) < 2:
            output.append({"name": class_spec.name, "mean_corr": np.nan, "pca_evr1": np.nan, "max_te": np.nan})
            continue
        values = class_consistency_kernel(
            np.ascontiguousarray(asset_returns[columns].to_numpy(dtype=np.float64)), 252.0
        )
        output.append({
            "name": class_spec.name,
            "mean_corr": float(values[0]),
            "pca_evr1": float(values[1]),
            "max_te": float(values[2]),
        })
    return output

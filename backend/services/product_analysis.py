from __future__ import annotations

"""Python orchestration for the fixed-signature ProductDetail compute graph."""

import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

try:
    from backend.compute_policy import validate_execution_audit
    from backend.product_analysis_numba import (
        bollinger_kernel,
        box_plot_kernel,
        daily_returns_percent_kernel,
        distribution_interpretation_codes_kernel,
        histogram_normal_pdf_kernel,
        kdj_kernel,
        moving_average_kernel,
        normal_qq_kernel,
        parametric_monte_carlo_kernel,
        product_analysis_execution_audit,
        regime_performance_kernel,
        return_statistics_kernel,
        simulation_comparison_kernel,
        stationary_block_bootstrap_kernel,
        technical_input_availability_kernel,
        terminal_density_kernel,
    )
except ModuleNotFoundError:  # pragma: no cover - backend/ direct execution
    from compute_policy import validate_execution_audit
    from product_analysis_numba import (
        bollinger_kernel,
        box_plot_kernel,
        daily_returns_percent_kernel,
        distribution_interpretation_codes_kernel,
        histogram_normal_pdf_kernel,
        kdj_kernel,
        moving_average_kernel,
        normal_qq_kernel,
        parametric_monte_carlo_kernel,
        product_analysis_execution_audit,
        regime_performance_kernel,
        return_statistics_kernel,
        simulation_comparison_kernel,
        stationary_block_bootstrap_kernel,
        technical_input_availability_kernel,
        terminal_density_kernel,
    )


PERIOD_MONTHS: dict[str, int | None] = {
    "ALL": None,
    "1M": 1,
    "3M": 3,
    "6M": 6,
    "1Y": 12,
    "3Y": 36,
    "5Y": 60,
}
PERIOD_LABELS = {
    "ALL": "成立以来",
    "1M": "近 1 月",
    "3M": "近 3 月",
    "6M": "近 6 月",
    "1Y": "近 1 年",
    "3Y": "近 3 年",
    "5Y": "近 5 年",
}
MAX_BOUNDARY_GAP_DAYS = 10
MIN_SIMULATION_OBSERVATIONS = 20


def _float_array(frame: pd.DataFrame, column: str) -> np.ndarray:
    values = pd.to_numeric(frame.get(column), errors="coerce")
    return np.array(values, dtype=np.float64, copy=True, order="C")


def _int_array(values: Sequence[int]) -> np.ndarray:
    return np.array(values, dtype=np.int64, copy=True, order="C")


def _optional_float(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _optional_series(values: np.ndarray) -> list[float | None]:
    return [_optional_float(value) for value in values]


def _prepare_frame(points: Sequence[Mapping[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(points)
    if frame.empty or "date" not in frame.columns or "close" not in frame.columns:
        raise ValueError("产品行情数据缺少 date/close 字段")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = (
        frame.dropna(subset=["date"])
        .sort_values("date", kind="mergesort")
        .drop_duplicates(subset=["date"], keep="last")
        .reset_index(drop=True)
    )
    if frame.empty:
        raise ValueError("产品行情没有有效日期")
    for column in ("open", "high", "low", "close", "volume"):
        if column not in frame.columns:
            frame[column] = np.nan
    return frame


def _statistics_window(
    frame: pd.DataFrame,
    period: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    months = PERIOD_MONTHS[period]
    if months is None:
        return frame, {
            "complete": True,
            "requested_start_date": None,
            "message": None,
        }
    effective = pd.Timestamp(frame.iloc[-1]["date"]).normalize()
    requested = (effective - pd.DateOffset(months=months)).normalize()
    eligible = frame.index[frame["date"] <= requested]
    boundary_index = int(eligible[-1]) if len(eligible) else -1
    if boundary_index < 0:
        complete = False
    else:
        boundary = pd.Timestamp(frame.iloc[boundary_index]["date"]).normalize()
        complete = int((requested - boundary).days) <= MAX_BOUNDARY_GAP_DAYS
    if not complete:
        label = PERIOD_LABELS[period]
        return frame.iloc[0:0], {
            "complete": False,
            "requested_start_date": requested.strftime("%Y-%m-%d"),
            "message": f"{label}要求产品完整覆盖所选区间；当前历史数据不足，统计指标不计算。",
        }
    return frame.iloc[boundary_index:].copy(), {
        "complete": True,
        "requested_start_date": requested.strftime("%Y-%m-%d"),
        "message": None,
    }


def _distribution_interpretation(codes: np.ndarray) -> dict[str, object]:
    skew_code = int(codes[0])
    if skew_code == 99:
        skewness = {
            "label": "样本不足",
            "meaning": "至少需要 3 个有效收益观察值才能判断偏度。",
        }
    elif skew_code == 0:
        skewness = {
            "label": "近似对称",
            "meaning": "正负收益尾部大致均衡；这只描述分布方向，不代表波动或尾部风险较低。",
        }
    elif skew_code < 0:
        intensity = {-1: "轻度", -2: "中度", -3: "显著"}[skew_code]
        skewness = {
            "label": f"{intensity}左偏（负偏）",
            "meaning": "左侧负收益尾部更长，少数较大亏损可能拖累整体收益，需特别关注下行尾部风险。",
        }
    else:
        intensity = {1: "轻度", 2: "中度", 3: "显著"}[skew_code]
        skewness = {
            "label": f"{intensity}右偏（正偏）",
            "meaning": "右侧正收益尾部更长，少数较大盈利可能抬高平均收益，但不代表亏损风险较低。",
        }

    kurtosis_code = int(codes[1])
    if kurtosis_code == 99:
        kurtosis = {
            "label": "样本不足",
            "meaning": "至少需要 4 个有效收益观察值才能判断峰度。",
        }
    elif kurtosis_code == 0:
        kurtosis = {
            "label": "接近正态峰度",
            "meaning": "样本峰度接近正态分布；仍需结合偏度、正态性检验和极值共同判断风险。",
        }
    elif kurtosis_code > 0:
        intensity = {1: "轻度", 2: "中度", 3: "显著"}[kurtosis_code]
        kurtosis = {
            "label": f"{intensity}尖峰厚尾",
            "meaning": "收益更集中在中心且尾部更厚，极端涨跌出现概率高于正态分布，正态模型可能低估尾部风险。",
        }
    else:
        intensity = {-1: "轻度", -2: "中度", -3: "显著"}[kurtosis_code]
        kurtosis = {
            "label": f"{intensity}平峰薄尾",
            "meaning": "收益分布较平、样本尾部相对较薄，历史极端波动较少，但不能据此排除未来尾部事件。",
        }
    normality_code = int(codes[2])
    normality = (
        "样本不足，无法进行检验"
        if normality_code == 99
        else (
            "拒绝正态假设（5% 显著性水平）"
            if normality_code == 1
            else "无法拒绝正态假设（5% 显著性水平）"
        )
    )
    return {"skewness": skewness, "kurtosis": kurtosis, "normality": normality}


def _simulation_seed(product_id: str, payload: Mapping[str, object], lane: str) -> int:
    material = "|".join(
        [
            product_id,
            str(payload.get("statistics_period")),
            str(payload.get("simulation_horizon")),
            str(payload.get("simulation_path_count")),
            str(payload.get("simulation_run")),
            lane,
        ]
    )
    return int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:8], "big") & (
        (1 << 63) - 1
    )


def _simulation_payload(
    result: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    method: str,
    method_label: str,
    horizon_days: int,
) -> tuple[dict[str, object], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    sample_paths, percentiles, terminal_values, summary, packed = result
    assumption_values = packed[horizon_days + 1 :]
    status_code = _optional_float(assumption_values[8])
    status = (
        None
        if status_code is None
        else {0: "matched", 1: "approximate", 2: "normal_fallback"}[int(status_code)]
    )
    assumptions = {
        "sourceObservationCount": int(assumption_values[0]),
        "targetReturnPercent": _optional_float(assumption_values[1]),
        "meanDailyLogReturn": _optional_float(assumption_values[2]),
        "dailyLogVolatility": _optional_float(assumption_values[3]),
        "historicalLogSkewness": _optional_float(assumption_values[4]),
        "historicalLogExcessKurtosis": _optional_float(assumption_values[5]),
        "fittedLogSkewness": _optional_float(assumption_values[6]),
        "fittedLogExcessKurtosis": _optional_float(assumption_values[7]),
        "shapeCalibrationStatus": status,
        "shapeSkewParameter": _optional_float(assumption_values[9]),
        "tailWeightParameter": _optional_float(assumption_values[10]),
        "averageBlockLength": _optional_float(assumption_values[11]),
    }
    terminal = {
        "p05": float(summary[0]),
        "p25": float(summary[1]),
        "p50": float(summary[2]),
        "p75": float(summary[3]),
        "p95": float(summary[4]),
        "lossProbability": float(summary[5]),
        "valueAtRisk95": float(summary[6]),
        "conditionalValueAtRisk95": float(summary[7]),
        "targetHitProbability": float(summary[8]),
        "averageMaxDrawdown": float(summary[9]),
        "p05Return": float(summary[10]),
        "medianReturn": float(summary[11]),
    }
    payload = {
        "method": method,
        "methodLabel": method_label,
        "days": [int(value) for value in packed[: horizon_days + 1]],
        "samplePaths": [[float(value) for value in row] for row in sample_paths],
        "percentiles": {
            "p05": percentiles[0].tolist(),
            "p25": percentiles[1].tolist(),
            "p50": percentiles[2].tolist(),
            "p75": percentiles[3].tolist(),
            "p95": percentiles[4].tolist(),
        },
        "terminal": terminal,
        "assumptions": assumptions,
    }
    return payload, (terminal_values, percentiles, summary)


def _density_payload(
    terminal_values: np.ndarray,
    percentiles: np.ndarray,
) -> dict[str, object]:
    points, histogram, summary = terminal_density_kernel(
        terminal_values,
        percentiles,
        81,
        1.0,
    )
    return {
        "sampleSize": int(summary[9]),
        "points": [
            {
                "nav": float(row[0]),
                "density": float(row[1]),
                "estimatedCount": float(row[2]),
                "simulatedReturn": float(row[3]),
            }
            for row in points
        ],
        "histogram": [
            {
                "lowerNav": float(row[0]),
                "upperNav": float(row[1]),
                "density": float(row[2]),
                "count": int(row[3]),
                "frequency": float(row[4]),
            }
            for row in histogram
        ],
        "maxDensity": float(summary[0]),
        "modeNav": float(summary[1]),
        "minNav": float(summary[2]),
        "maxNav": float(summary[3]),
        "countAxisMax": int(summary[4]),
        "navAxisMin": float(summary[5]),
        "navAxisMax": float(summary[6]),
        "densityCountFactor": float(summary[7]),
        "histogramBinWidth": float(summary[8]),
    }


def _regime_payload(
    frame: pd.DataFrame,
    regime: Mapping[str, object] | None,
) -> list[dict[str, object]]:
    if not regime:
        return []
    states = list(regime.get("states") or [])
    segments = list(regime.get("segments") or [])
    if not states or not segments:
        return []
    state_codes = {str(state.get("id")): index for index, state in enumerate(states)}
    start_days: list[int] = []
    end_days: list[int] = []
    segment_codes: list[int] = []
    for segment in segments:
        start = pd.to_datetime(segment.get("start_date"), errors="coerce")
        end = pd.to_datetime(segment.get("end_date"), errors="coerce")
        code = state_codes.get(str(segment.get("state_id")))
        if pd.isna(start) or pd.isna(end) or code is None or start > end:
            continue
        start_days.append(int(pd.Timestamp(start).to_datetime64().astype("datetime64[D]").astype(np.int64)))
        end_days.append(int(pd.Timestamp(end).to_datetime64().astype("datetime64[D]").astype(np.int64)))
        segment_codes.append(code)
    if not start_days:
        return []
    date_days = np.array(
        frame["date"].to_numpy(dtype="datetime64[D]").view(np.int64),
        dtype=np.int64,
        copy=True,
        order="C",
    )
    values = regime_performance_kernel(
        date_days,
        _float_array(frame, "close"),
        _int_array(start_days),
        _int_array(end_days),
        _int_array(segment_codes),
        len(states),
    )
    output: list[dict[str, object]] = []
    for index, state in enumerate(states):
        if values[index, 0] <= 0:
            continue
        output.append(
            {
                "stateId": str(state.get("id")),
                "stateLabel": str(state.get("label") or state.get("id")),
                "color": str(state.get("color") or "#64748b"),
                "observations": int(values[index, 0]),
                "returnObservations": int(values[index, 1]),
                "cumulativeReturn": _optional_float(values[index, 2]),
                "annualizedVolatility": _optional_float(values[index, 3]),
                "maxDrawdown": _optional_float(values[index, 4]),
                "winRate": _optional_float(values[index, 5]),
            }
        )
    return output


def build_product_analysis_response(
    *,
    product_id: str,
    points: Sequence[Mapping[str, object]],
    parameters: Mapping[str, object],
) -> dict[str, object]:
    frame = _prepare_frame(points)
    open_values = _float_array(frame, "open")
    close = _float_array(frame, "close")
    high = _float_array(frame, "high")
    low = _float_array(frame, "low")
    volume = _float_array(frame, "volume")
    availability = technical_input_availability_kernel(
        open_values,
        high,
        low,
        close,
        volume,
    )
    price_periods = _int_array(parameters.get("price_ma_periods") or [])
    volume_periods = _int_array(parameters.get("volume_ma_periods") or [])
    price_ma = moving_average_kernel(close, price_periods)
    volume_ma = moving_average_kernel(volume, volume_periods)
    bollinger = bollinger_kernel(
        close,
        int(parameters["boll_period"]),
        float(parameters["boll_multiplier"]),
    )
    kdj = kdj_kernel(
        high,
        low,
        close,
        int(parameters["kdj_period"]),
        int(parameters["kdj_k_smoothing"]),
        int(parameters["kdj_d_smoothing"]),
    )

    statistics_frame, window = _statistics_window(
        frame,
        str(parameters["statistics_period"]),
    )
    statistics_close = _float_array(statistics_frame, "close")
    returns = daily_returns_percent_kernel(statistics_close)
    statistics = return_statistics_kernel(returns)
    interpretation = _distribution_interpretation(
        distribution_interpretation_codes_kernel(statistics)
    )
    histogram = histogram_normal_pdf_kernel(
        returns,
        float(parameters["histogram_bin_width"]),
    )
    box_statistics, outliers = box_plot_kernel(returns)
    qq_points, qq_key_points = normal_qq_kernel(returns)
    return_dates = (
        statistics_frame["date"].iloc[1:].dt.strftime("%Y-%m-%d").tolist()
        if len(statistics_frame) >= 2
        else []
    )
    daily_returns = [
        {"date": date, "return": float(value)}
        for date, value in zip(return_dates, returns)
        if np.isfinite(value)
    ]

    simulation: dict[str, object] | None = None
    if int(statistics[6]) >= MIN_SIMULATION_OBSERVATIONS:
        horizon = int(parameters["simulation_horizon"])
        path_count = int(parameters["simulation_path_count"])
        target_return = float(parameters["simulation_target_return"])
        parametric_result = parametric_monte_carlo_kernel(
            returns,
            1.0,
            horizon,
            path_count,
            _simulation_seed(product_id, parameters, "parametric"),
            target_return,
        )
        bootstrap_result = stationary_block_bootstrap_kernel(
            returns,
            1.0,
            horizon,
            path_count,
            _simulation_seed(product_id, parameters, "bootstrap"),
            target_return,
            int(parameters["bootstrap_block_length"]),
        )
        parametric_label = (
            "参数化蒙特卡洛（正态安全降级）"
            if int(parametric_result[4][horizon + 1 + 8]) == 2
            else "参数化蒙特卡洛（偏度/峰度校准）"
        )
        parametric, parametric_arrays = _simulation_payload(
            parametric_result,
            method="parametric",
            method_label=parametric_label,
            horizon_days=horizon,
        )
        bootstrap, bootstrap_arrays = _simulation_payload(
            bootstrap_result,
            method="block_bootstrap",
            method_label="历史区块 Bootstrap",
            horizon_days=horizon,
        )
        comparison_values = simulation_comparison_kernel(
            parametric_arrays[2], bootstrap_arrays[2], 1.0
        )
        level = {0: "low", 1: "medium", 2: "high"}[int(comparison_values[4])]
        message = {
            "high": "两种模型差异明显，结果对模型假设较敏感，决策时应采用更保守的尾部结果。",
            "medium": "两种模型存在一定差异，建议同时查看参数化假设与历史区块情景。",
            "low": "两种模型结果接近，但仍不代表对未来走势形成预测。",
        }[level]
        simulation = {
            "initialNav": 1.0,
            "parametric": parametric,
            "blockBootstrap": bootstrap,
            "comparison": {
                "p05ReturnGap": float(comparison_values[0]),
                "medianReturnGap": float(comparison_values[1]),
                "lossProbabilityGap": float(comparison_values[2]),
                "conditionalValueAtRiskGap": float(comparison_values[3]),
                "level": level,
                "message": message,
            },
            "densities": {
                "parametric": _density_payload(parametric_arrays[0], parametric_arrays[1]),
                "block_bootstrap": _density_payload(bootstrap_arrays[0], bootstrap_arrays[1]),
            },
        }

    qq_tail = {0: "lower", 1: "center", 2: "upper"}
    return {
        "schema_version": 1,
        "product_id": product_id,
        "execution": validate_execution_audit(product_analysis_execution_audit()),
        "window": window,
        "technical": {
            "availability": {
                "ohlc": bool(availability[0]),
                "volume": bool(availability[1]),
                "kdj": bool(availability[2]),
            },
            "priceMa": {
                str(period): _optional_series(price_ma[index])
                for index, period in enumerate(price_periods)
            },
            "volumeMa": {
                str(period): _optional_series(volume_ma[index])
                for index, period in enumerate(volume_periods)
            },
            "bollinger": {
                "upper": _optional_series(bollinger[0]),
                "middle": _optional_series(bollinger[1]),
                "lower": _optional_series(bollinger[2]),
            },
            "kdj": {
                "kValues": _optional_series(kdj[0]),
                "dValues": _optional_series(kdj[1]),
                "jValues": _optional_series(kdj[2]),
            },
        },
        "dailyReturns": daily_returns,
        "returnStatistics": {
            "mean": _optional_float(statistics[0]),
            "std": _optional_float(statistics[1]),
            "median": _optional_float(statistics[2]),
            "positiveRatio": _optional_float(statistics[3]),
            "best": _optional_float(statistics[4]),
            "worst": _optional_float(statistics[5]),
            "sampleSize": int(statistics[6]),
            "skewness": _optional_float(statistics[7]),
            "kurtosis": _optional_float(statistics[8]),
            "jbStatistic": _optional_float(statistics[9]),
            "normalityPValue": _optional_float(statistics[10]),
        },
        "interpretation": interpretation,
        "histogram": [
            {
                "start": float(row[0]),
                "end": float(row[1]),
                "count": int(row[2]),
                "normalPdfCount": _optional_float(row[3]),
                "frequency": float(row[4]),
                "center": float(row[5]),
            }
            for row in histogram
        ],
        "boxPlot": (
            None
            if box_statistics.size == 0
            else {
                "stats": box_statistics.tolist(),
                "outliers": outliers.tolist(),
                "quartiles": {
                    "q1": float(box_statistics[1]),
                    "median": float(box_statistics[2]),
                    "q3": float(box_statistics[3]),
                    "iqr": float(box_statistics[5]),
                },
                "whiskers": {
                    "lower": float(box_statistics[0]),
                    "upper": float(box_statistics[4]),
                },
            }
        ),
        "normalQq": (
            None
            if qq_points.size == 0
            else {
                "sampleSize": int(qq_points.shape[0]),
                "points": [
                    {
                        "percentile": float(row[0]),
                        "theoreticalQuantile": float(row[1]),
                        "observedReturn": float(row[2]),
                        "referenceReturn": float(row[3]),
                        "tail": qq_tail[int(row[4])],
                    }
                    for row in qq_points
                ],
                "keyPoints": [
                    {
                        "percentile": float(row[0]),
                        "theoreticalQuantile": float(row[1]),
                        "observedReturn": float(row[2]),
                        "referenceReturn": float(row[3]),
                        "tail": qq_tail[int(row[4])],
                    }
                    for row in qq_key_points
                    if np.isfinite(row[0])
                ],
            }
        ),
        "simulation": simulation,
        "regimeStatistics": _regime_payload(frame, parameters.get("regime")),
    }


__all__ = ["build_product_analysis_response"]

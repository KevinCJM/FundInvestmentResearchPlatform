from __future__ import annotations

"""Python orchestration for the fixed-signature ProductDetail compute graph."""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

try:
    from backend.compute_policy import validate_execution_audit
    from backend.product_analysis_numba import (
        DENSITY_CURVE_POINTS,
        DENSITY_FRAME_SLOTS,
        bollinger_kernel,
        box_plot_kernel,
        distribution_interpretation_codes_kernel,
        filtered_historical_simulation_kernel,
        histogram_normal_pdf_kernel,
        kdj_kernel,
        moving_average_kernel,
        normal_qq_kernel,
        parametric_monte_carlo_kernel,
        product_analysis_execution_audit,
        realized_path_comparison_kernel,
        regime_analysis_kernel,
        return_statistics_kernel,
        simulation_comparison_kernel,
        stationary_block_bootstrap_kernel,
        technical_input_availability_kernel,
    )
except ModuleNotFoundError:  # pragma: no cover - backend/ direct execution
    from compute_policy import validate_execution_audit
    from product_analysis_numba import (
        DENSITY_CURVE_POINTS,
        DENSITY_FRAME_SLOTS,
        bollinger_kernel,
        box_plot_kernel,
        distribution_interpretation_codes_kernel,
        filtered_historical_simulation_kernel,
        histogram_normal_pdf_kernel,
        kdj_kernel,
        moving_average_kernel,
        normal_qq_kernel,
        parametric_monte_carlo_kernel,
        product_analysis_execution_audit,
        realized_path_comparison_kernel,
        regime_analysis_kernel,
        return_statistics_kernel,
        simulation_comparison_kernel,
        stationary_block_bootstrap_kernel,
        technical_input_availability_kernel,
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
# Display order, and the order the comparison spread is computed over. The
# normal lane comes first on purpose: it is the textbook baseline every richer
# model has to earn its extra assumptions against.
SIMULATION_METHODS: tuple[str, ...] = (
    "gaussian",
    "parametric",
    "block_bootstrap",
    "fhs_ewma",
    "fhs_garch",
)
# Where the realised terminal value landed inside the simulated distribution.
# The band, not the exact rank, is what a reader can act on, and it is the one
# score that survives a horizon the data only partly covers.
REALIZED_BAND_LABELS = (
    "低于 5% 分位",
    "5% — 25% 分位",
    "25% — 50% 分位",
    "50% — 75% 分位",
    "75% — 95% 分位",
    "高于 95% 分位",
)
REALIZED_BAND_VERDICTS = (
    "实际走势跌破模拟分布最悲观的 5%：该模型低估了下行，按它做的风险预算偏松。",
    "实际走势落在偏悲观区间（5%—25% 分位）：该模型的中枢偏乐观。",
    "实际走势略低于模拟中位数：与该模型的中枢基本一致。",
    "实际走势略高于模拟中位数：与该模型的中枢基本一致。",
    "实际走势落在偏乐观区间（75%—95% 分位）：该模型的中枢偏保守。",
    "实际走势冲出模拟分布最乐观的 5%：该模型同样低估了上行幅度，不能因此认为它更安全。",
)


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
            str(payload.get("analysis_basis")),
            json.dumps(payload.get("regime"), sort_keys=True, ensure_ascii=False),
            lane,
        ]
    )
    return int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:8], "big") & (
        (1 << 63) - 1
    )


def _simulation_payload(
    result: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    method: str,
    method_label: str,
    horizon_days: int,
) -> tuple[dict[str, object], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    sample_paths, percentiles, terminal_values, summary, packed, frames = result
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
        # Filtered lanes only. `conditionalVolatilityStart` against
        # `dailyLogVolatility` is the whole difference between a conditional
        # model and an unconditional one, and `volatilityPersistence` says how
        # long that starting state keeps mattering.
        "conditionalVolatilityStart": _optional_float(assumption_values[12]),
        "volatilityPersistence": _optional_float(assumption_values[13]),
        "garchOmega": _optional_float(assumption_values[14]),
        "garchAlpha": _optional_float(assumption_values[15]),
        "garchBeta": _optional_float(assumption_values[16]),
        "ewmaLambda": _optional_float(assumption_values[17]),
        "residualSkewness": _optional_float(assumption_values[18]),
        "residualExcessKurtosis": _optional_float(assumption_values[19]),
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
    return payload, (terminal_values, percentiles, summary, frames)


def _density_payload(
    frames: np.ndarray,
    summary: np.ndarray,
    sample_size: int,
) -> dict[str, object]:
    """Every checkpoint day's landing distribution, plus the shared NAV axis.

    Counts only, on a grid the reader rebuilds from ``navLow``/``navHigh`` —
    naming each point's NAV would multiply the payload by four for numbers the
    client can compute exactly.
    """
    curve_base = DENSITY_FRAME_SLOTS
    bin_base = DENSITY_FRAME_SLOTS + DENSITY_CURVE_POINTS
    return {
        "sampleSize": sample_size,
        "navAxisMin": float(summary[12]),
        "navAxisMax": float(summary[13]),
        "frames": [
            {
                "day": int(row[0]),
                "navLow": float(row[1]),
                "navHigh": float(row[2]),
                "binWidth": float(row[3]),
                "countAxisMax": int(row[4]),
                "curve": [float(value) for value in row[curve_base:bin_base]],
                "bins": [int(value) for value in row[bin_base:]],
            }
            for row in frames
        ],
    }


def _last_valid_close(frame: pd.DataFrame) -> tuple[str, float] | None:
    """The observation the simulation is anchored on — its nav is 1.0 by definition."""

    closes = pd.to_numeric(frame.get("close"), errors="coerce")
    valid = frame.index[closes.notna() & (closes > 0)]
    if not len(valid):
        return None
    row = frame.loc[valid[-1]]
    return pd.Timestamp(row["date"]).strftime("%Y-%m-%d"), float(row["close"])


def _realized_payload(
    *,
    future_frame: pd.DataFrame,
    base_date: str,
    base_close: float,
    initial_nav: float,
    as_of: str | None,
    lanes: Mapping[str, tuple[str, np.ndarray, np.ndarray]],
) -> dict[str, object] | None:
    """Replay what actually happened after the research day, and score it.

    Loaded from a separate lane that never reaches the simulation inputs: the
    models are fitted on rows at or before the research day, so nothing here
    can leak backwards into them. Returns None when the future rows exist but
    hold no usable price, which is not the same thing as having no future.
    """

    future_closes = _float_array(future_frame, "close")
    dates = future_frame["date"].dt.strftime("%Y-%m-%d").tolist()
    shared: dict[str, object] = {}
    by_method: dict[str, object] = {}
    nav_values: list[float | None] = []
    for method, (label, percentiles, terminal_values) in lanes.items():
        path, summary = realized_path_comparison_kernel(
            future_closes,
            base_close,
            np.ascontiguousarray(percentiles),
            np.ascontiguousarray(terminal_values),
            initial_nav,
        )
        covered = int(summary[1])
        band = int(summary[6])
        by_method[method] = {
            "methodLabel": label,
            "percentileRank": _optional_float(summary[5]),
            "band": band if band >= 0 else None,
            "bandLabel": REALIZED_BAND_LABELS[band] if band >= 0 else None,
            "verdict": REALIZED_BAND_VERDICTS[band] if band >= 0 else None,
            "containmentRatio": _optional_float(summary[7]),
            "breachDays": int(summary[8]),
            "worstBreachGap": _optional_float(summary[9]),
            "worstBreachDay": int(summary[10]) if summary[10] >= 0 else None,
            "aboveMedianRatio": _optional_float(summary[12]),
            "simulatedP05": _optional_float(percentiles[0, covered]) if covered else None,
            "simulatedP50": _optional_float(percentiles[2, covered]) if covered else None,
            "simulatedP95": _optional_float(percentiles[4, covered]) if covered else None,
        }
        if shared:
            continue
        if covered <= 0:
            return None
        shared = {
            "requestedDays": int(summary[0]),
            "coveredDays": covered,
            "observationDays": int(summary[13]),
            "complete": bool(summary[2]),
            "terminalNav": _optional_float(summary[3]),
            "terminalReturn": _optional_float(summary[4]),
            "maxDrawdown": _optional_float(summary[11]),
        }
        nav_values = _optional_series(path)
    return {
        "asOf": as_of,
        "baseDate": base_date,
        "baseNav": initial_nav,
        "startDate": dates[0] if dates else None,
        "endDate": dates[int(shared["coveredDays"]) - 1],
        "dates": [base_date, *dates[: max(0, len(nav_values) - 1)]],
        "nav": nav_values,
        "byMethod": by_method,
        **shared,
    }


def _regime_analysis(
    frame: pd.DataFrame,
    regime: Mapping[str, object] | None,
) -> tuple[dict[str, object] | None, np.ndarray, np.ndarray, np.ndarray]:
    states = list(regime.get("states") or []) if regime else []
    segments = [dict(item) for item in regime.get("segments") or []] if regime else []
    state_codes = {str(state.get("id")): index for index, state in enumerate(states)}
    if len(state_codes) != len(states):
        raise ValueError("历史情景状态标识重复")
    selected_state_id = regime.get("state_id") if regime else None
    selected_segment_id = regime.get("segment_id") if regime else None
    if selected_state_id is not None and selected_state_id not in state_codes:
        raise ValueError("所选市场状态不属于该历史情景版本")
    starts: list[int] = []
    ends: list[int] = []
    codes: list[int] = []
    selected_segment = -1
    for index, segment in enumerate(segments):
        segment_id = str(segment.get("id") or f"segment-{index}")
        segment["id"] = segment_id
        start = pd.to_datetime(segment.get("start_date"), errors="coerce")
        end = pd.to_datetime(segment.get("end_date"), errors="coerce")
        code = state_codes.get(str(segment.get("state_id")))
        if pd.isna(start) or pd.isna(end) or code is None or start > end:
            raise ValueError("历史情景区间的日期或状态无效")
        starts.append(int(pd.Timestamp(start).to_datetime64().astype("datetime64[D]").astype(np.int64)))
        ends.append(int(pd.Timestamp(end).to_datetime64().astype("datetime64[D]").astype(np.int64)))
        codes.append(code)
        if segment_id == selected_segment_id:
            selected_segment = index
            if selected_state_id is not None and segment["state_id"] != selected_state_id:
                raise ValueError("所选区间不属于所选市场状态")
            selected_state_id = segment["state_id"]
    if selected_segment_id is not None and selected_segment < 0:
        raise ValueError("所选区间不属于该历史情景版本")
    values, segment_values, returns, return_segments, context = regime_analysis_kernel(
        np.array(frame["date"].to_numpy(dtype="datetime64[D]").view(np.int64), dtype=np.int64, copy=True, order="C"),
        _float_array(frame, "close"), _int_array(starts), _int_array(ends),
        _int_array(codes), len(states), state_codes.get(selected_state_id, -1), selected_segment,
    )
    if not regime:
        return None, returns, return_segments, context
    state_rows = []
    for index, state in enumerate(states):
        row = values[index]
        state_rows.append({
            "stateId": str(state["id"]),
            "stateLabel": str(state.get("label") or state["id"]),
            "color": str(state.get("color") or "#64748b"),
            "observations": int(row[0]), "returnObservations": int(row[1]),
            "segmentCount": int(row[2]), "medianSegmentObservations": _optional_float(row[3]),
            "eligibleSegmentCount": int(row[4]), "meanDailyReturn": _optional_float(row[5]),
            "annualizedVolatility": _optional_float(row[6]), "winRate": _optional_float(row[7]),
            "medianSegmentReturn": _optional_float(row[8]), "worstSegmentReturn": _optional_float(row[9]),
            "medianSegmentDrawdown": _optional_float(row[10]), "worstSegmentDrawdown": _optional_float(row[11]),
        })
    segment_rows = []
    for index, segment in enumerate(segments):
        row = segment_values[index]
        if row[0] <= 0:
            continue
        state = states[codes[index]]
        start_date = frame.iloc[int(row[5])]["date"].strftime("%Y-%m-%d")
        end_date = frame.iloc[int(row[6])]["date"].strftime("%Y-%m-%d")
        status_code = int(row[4])
        segment_rows.append({
            "id": segment["id"], "stateId": state["id"],
            "stateLabel": str(state.get("label") or state["id"]),
            "color": str(state.get("color") or "#64748b"),
            "startDate": start_date, "endDate": end_date,
            "observations": int(row[0]), "returnObservations": int(row[1]),
            "cumulativeReturn": _optional_float(row[2]), "maxDrawdown": _optional_float(row[3]),
            "status": {0: "complete", 1: "insufficient_sample", 2: "missing_data"}[status_code],
            "reason": {0: None, 1: "至少需要两个有效价格观察值。", 2: "区间存在缺失或无效价格，区间收益与回撤不计算。"}[status_code],
            "windowClipped": start_date != segment["start_date"] or end_date != segment["end_date"],
            "validObservations": int(row[7]),
        })
    return {
        "states": state_rows, "segments": segment_rows,
        "selectedStateId": selected_state_id, "selectedSegmentId": selected_segment_id,
    }, returns, return_segments, context


def build_product_analysis_response(
    *,
    product_id: str,
    points: Sequence[Mapping[str, object]],
    parameters: Mapping[str, object],
    research_points: Sequence[Mapping[str, object]] | None = None,
    future_points: Sequence[Mapping[str, object]] | None = None,
    as_of: str | None = None,
) -> dict[str, object]:
    if points:
        frame = _prepare_frame(points)
    else:
        frame = pd.DataFrame({name: pd.Series(dtype="float64") for name in ("open", "high", "low", "close", "volume")})
        frame["date"] = pd.Series(dtype="datetime64[ns]")
    research_frame = _prepare_frame(research_points) if research_points is not None else frame
    if research_frame.empty:
        raise ValueError("产品研究数据没有有效日期")
    open_values = _float_array(frame, "open")
    close = _float_array(frame, "close")
    high = _float_array(frame, "high")
    low = _float_array(frame, "low")
    volume = _float_array(frame, "volume")
    include_technical = bool(parameters.get("include_technical", True))
    if include_technical:
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
    else:
        availability = np.zeros(3, dtype=np.int64)
        price_periods = np.empty(0, dtype=np.int64)
        volume_periods = np.empty(0, dtype=np.int64)
        price_ma = np.empty((0, close.size), dtype=np.float64)
        volume_ma = np.empty((0, close.size), dtype=np.float64)
        bollinger = np.empty((3, 0), dtype=np.float64)
        kdj = np.empty((3, 0), dtype=np.float64)

    statistics_frame, window = _statistics_window(
        research_frame,
        str(parameters["statistics_period"]),
    )
    regime_analysis, returns, return_segments, context_counts = _regime_analysis(
        statistics_frame, parameters.get("regime")
    )
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
    eligible = int(statistics[6]) >= MIN_SIMULATION_OBSERVATIONS
    requested = bool(parameters.get("include_simulation", False))
    simulation_status = "not_requested" if not requested else "insufficient_sample"
    if requested and eligible:
        horizon = int(parameters["simulation_horizon"])
        path_count = int(parameters["simulation_path_count"])
        target_return = float(parameters["simulation_target_return"])
        ewma_lambda = float(parameters.get("fhs_ewma_lambda", 0.94))
        # Every lane runs on the same horizon, path budget, target and random
        # round: those belong to the experiment, not to a model. A lane allowed
        # its own horizon would make the comparison table and the realised
        # overlay meaningless at the same time.
        #
        # `returns` and `return_segments` are handed to all five lanes as the
        # same arrays — no per-lane copy, no `ascontiguousarray` round trip.
        # That is safe *and* the reason the lanes can run at once: every kernel
        # treats its inputs as read-only, which
        # `test_simulation_lanes_share_one_read_only_input` locks down.
        tasks = (
            ("gaussian", parametric_monte_carlo_kernel, (
                returns, 1.0, horizon, path_count,
                _simulation_seed(product_id, parameters, "gaussian"), target_return, 0,
            )),
            ("parametric", parametric_monte_carlo_kernel, (
                returns, 1.0, horizon, path_count,
                _simulation_seed(product_id, parameters, "parametric"), target_return, 1,
            )),
            ("block_bootstrap", stationary_block_bootstrap_kernel, (
                returns, return_segments, 1.0, horizon, path_count,
                _simulation_seed(product_id, parameters, "bootstrap"), target_return,
                int(parameters["bootstrap_block_length"]),
            )),
            ("fhs_ewma", filtered_historical_simulation_kernel, (
                returns, return_segments, 1.0, horizon, path_count,
                _simulation_seed(product_id, parameters, "fhs_ewma"), target_return,
                0, ewma_lambda,
            )),
            ("fhs_garch", filtered_historical_simulation_kernel, (
                returns, return_segments, 1.0, horizon, path_count,
                _simulation_seed(product_id, parameters, "fhs_garch"), target_return,
                1, ewma_lambda,
            )),
        )
        # Every kernel is compiled `nogil=True`, so these threads leave the GIL
        # and the five lanes really do overlap — five sequential lanes at the
        # 504-day / 1000-path ceiling cost about 190 ms, and the wall clock now
        # tracks the slowest lane instead of their sum. Results are collected by
        # name so the response order stays the declared one, not the finish
        # order, and any kernel ValueError surfaces from `.result()` exactly as
        # it would have from a direct call.
        with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
            pending = {
                method: pool.submit(kernel, *arguments)
                for method, kernel, arguments in tasks
            }
            raw_results = {method: future.result() for method, future in pending.items()}
        labels = {
            "gaussian": "标准正态蒙特卡洛",
            "parametric": (
                "参数化蒙特卡洛（正态安全降级）"
                if int(raw_results["parametric"][4][horizon + 1 + 8]) == 2
                else "参数化蒙特卡洛（偏度/峰度校准）"
            ),
            "block_bootstrap": "历史区块 Bootstrap",
            "fhs_ewma": "滤波历史模拟 · EWMA",
            "fhs_garch": "滤波历史模拟 · GARCH(1,1)",
        }
        payloads: dict[str, dict[str, object]] = {}
        arrays: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        for method in SIMULATION_METHODS:
            payloads[method], arrays[method] = _simulation_payload(
                raw_results[method],
                method=method,
                method_label=labels[method],
                horizon_days=horizon,
            )
        comparison_values = simulation_comparison_kernel(
            np.ascontiguousarray(
                np.vstack([arrays[method][2] for method in SIMULATION_METHODS])
            ),
            1.0,
        )
        level = {0: "low", 1: "medium", 2: "high"}[int(comparison_values[4])]
        message = {
            "high": f"{len(SIMULATION_METHODS)} 个模型之间差异明显，结果高度依赖模型假设；决策时应采用更保守的尾部结果。",
            "medium": "模型之间存在一定差异，建议对照条件模型（滤波历史模拟）与无条件模型的结论。",
            "low": "各模型结果接近，但一致并不等于正确，也不代表对未来走势形成预测。",
        }[level]
        # What actually happened after the research day, measured against both
        # models. Only reachable under a PIT研究日 with rows beyond it, which is
        # also the only situation in which the question can be asked honestly.
        realized: dict[str, object] | None = None
        realized_status = "off" if not as_of else "no_future_data"
        anchor = _last_valid_close(statistics_frame)
        if as_of and future_points and anchor is not None:
            future_frame = _prepare_frame(future_points)
            if not future_frame.empty:
                realized = _realized_payload(
                    future_frame=future_frame,
                    base_date=anchor[0],
                    base_close=anchor[1],
                    initial_nav=1.0,
                    as_of=as_of,
                    lanes={
                        method: (
                            payloads[method]["methodLabel"],
                            arrays[method][1],
                            arrays[method][0],
                        )
                        for method in SIMULATION_METHODS
                    },
                )
            if realized is not None:
                realized_status = "complete" if realized["complete"] else "partial"
        simulation_status = "complete"
        simulation = {
            "initialNav": 1.0,
            "realized": realized,
            "realizedStatus": realized_status,
            "methods": list(SIMULATION_METHODS),
            "byMethod": payloads,
            "comparison": {
                "p05ReturnGap": float(comparison_values[0]),
                "medianReturnGap": float(comparison_values[1]),
                "lossProbabilityGap": float(comparison_values[2]),
                "conditionalValueAtRiskGap": float(comparison_values[3]),
                "level": level,
                "message": message,
            },
            "densities": {
                method: _density_payload(
                    arrays[method][3], arrays[method][2], len(arrays[method][0])
                )
                for method in SIMULATION_METHODS
            },
        }

    qq_tail = {0: "lower", 1: "center", 2: "upper"}
    scope = "full"
    state_label = None
    if regime_analysis:
        scope = "segment" if regime_analysis["selectedSegmentId"] else ("state" if regime_analysis["selectedStateId"] else "full")
        state_label = next((item["stateLabel"] for item in regime_analysis["states"] if item["stateId"] == regime_analysis["selectedStateId"]), None)
    basis = str(parameters.get("analysis_basis") or "price")
    conditional_message = "假设未来持续处于所选状态；不估计状态切换或发生概率。" if scope != "full" else "从研究窗口的历史收益抽样；不代表未来预测。"
    simulation_message = conditional_message if eligible else f"至少需要 {MIN_SIMULATION_OBSERVATIONS} 个有效收益观察值，当前有 {int(statistics[6])} 个。"
    fingerprint = hashlib.sha256(statistics_frame[["date", "close"]].to_json(date_format="iso").encode()).hexdigest()
    return {
        "schema_version": 2,
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
        "simulationStatus": simulation_status,
        "regimeAnalysis": regime_analysis,
        "researchContext": {
            "startDate": statistics_frame.iloc[int(context_counts[3])]["date"].strftime("%Y-%m-%d") if context_counts[3] >= 0 else None,
            "endDate": statistics_frame.iloc[int(context_counts[4])]["date"].strftime("%Y-%m-%d") if context_counts[4] >= 0 else None,
            "windowStartDate": statistics_frame.iloc[0]["date"].strftime("%Y-%m-%d") if len(statistics_frame) else None,
            "windowEndDate": statistics_frame.iloc[-1]["date"].strftime("%Y-%m-%d") if len(statistics_frame) else None,
            "observations": int(context_counts[0]), "returnObservations": int(context_counts[1]),
            "segmentCount": int(context_counts[2]), "scope": scope, "stateLabel": state_label,
            "boundaryPolicy": "情景收益仅取同一连续区间内相邻有效观察值，跨状态边界与缺失价格不连接；多段覆盖范围不代表连续持有，区间路径指标仅统计完整有效路径。" if scope != "full" else "使用完整研究窗口内相邻有效观察值，缺失价格不连接。",
            "simulationEligible": eligible, "simulationMessage": simulation_message,
            "analysisBasis": basis, "basisLabel": "复权净值" if basis == "adjusted_nav" else "市场收盘价",
            "dataFingerprint": fingerprint,
        },
    }


__all__ = ["build_product_analysis_response"]

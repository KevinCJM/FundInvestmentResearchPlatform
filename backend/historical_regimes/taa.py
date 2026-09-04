"""Deterministic TAA backtest driven by immutable historical-regime runs."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
from numba import float64, njit, types, uint8

try:
    from backend.compute_policy import validate_execution_audit
except ModuleNotFoundError:  # pragma: no cover - backend/ direct execution
    from compute_policy import validate_execution_audit

from custom_indicators.errors import ValidationError


WEIGHT_TOLERANCE = 1e-6
MAX_ASSETS = 100
MAX_RETURN_ROWS = 20000
TAA_ENGINE_VERSION = "taa-njit-1.0.0"
TAA_KERNEL_VERSION = "1.1.0"


_TAA_CORE_SIGNATURE = types.Tuple(
    (float64[:, ::1], float64[:, ::1], float64[:, ::1], float64[::1])
)(
    float64[:, ::1],
    float64[:, ::1],
    uint8[::1],
    float64[:, ::1],
    float64[::1],
    float64,
    float64,
    float64,
    float64,
)


@njit(_TAA_CORE_SIGNATURE, cache=True, nogil=True)
def _taa_path_kernel(
    asset_returns: np.ndarray,
    probabilities: np.ndarray,
    use_signal: np.ndarray,
    state_tilts: np.ndarray,
    base_weights: np.ndarray,
    minimum_weight: float,
    maximum_weight: float,
    maximum_absolute_tilt: float,
    transaction_cost_bps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Advance both SAA and TAA with one fixed-signature nopython kernel."""

    period_count, asset_count = asset_returns.shape
    state_count = state_tilts.shape[0]
    # target[A], pretrade[A], scale/turnover/cost/returns/nav and baseline cost.
    path = np.empty((period_count, asset_count * 2 + 14), dtype=np.float64)
    contributions = np.zeros((period_count, state_count), dtype=np.float64)
    state_summary = np.zeros((state_count, 3), dtype=np.float64)
    totals = np.zeros(10, dtype=np.float64)
    taa_pretrade = base_weights.copy()
    baseline_pretrade = base_weights.copy()
    taa_nav = 1.0
    baseline_nav = 1.0

    for period in range(period_count):
        delta = np.zeros(asset_count, dtype=np.float64)
        tilt_scale = 0.0
        if use_signal[period] == 1:
            for state in range(state_count):
                probability = probabilities[period, state]
                for asset in range(asset_count):
                    delta[asset] += probability * state_tilts[state, asset]
            tilt_scale = 1.0
            for asset in range(asset_count):
                value = delta[asset]
                absolute_value = abs(value)
                if absolute_value > maximum_absolute_tilt and absolute_value > WEIGHT_TOLERANCE:
                    candidate = maximum_absolute_tilt / absolute_value
                    if candidate < tilt_scale:
                        tilt_scale = candidate
                if value > WEIGHT_TOLERANCE:
                    candidate = (maximum_weight - base_weights[asset]) / value
                    if candidate < tilt_scale:
                        tilt_scale = candidate
                elif value < -WEIGHT_TOLERANCE:
                    candidate = (minimum_weight - base_weights[asset]) / value
                    if candidate < tilt_scale:
                        tilt_scale = candidate
            if tilt_scale < 0.0:
                tilt_scale = 0.0
            elif tilt_scale > 1.0:
                tilt_scale = 1.0

        taa_turnover = 0.0
        baseline_turnover = 0.0
        for asset in range(asset_count):
            target = base_weights[asset] + tilt_scale * delta[asset]
            path[period, asset] = target
            path[period, asset_count + asset] = taa_pretrade[asset]
            taa_turnover += abs(target - taa_pretrade[asset])
            baseline_turnover += abs(base_weights[asset] - baseline_pretrade[asset])
        taa_turnover *= 0.5
        baseline_turnover *= 0.5
        taa_cost_rate = taa_turnover * transaction_cost_bps / 10000.0
        baseline_cost_rate = baseline_turnover * transaction_cost_bps / 10000.0

        gross_taa_return = 0.0
        gross_baseline_return = 0.0
        for asset in range(asset_count):
            gross_taa_return += path[period, asset] * asset_returns[period, asset]
            gross_baseline_return += base_weights[asset] * asset_returns[period, asset]
        net_taa_return = (1.0 - taa_cost_rate) * (1.0 + gross_taa_return) - 1.0
        net_baseline_return = (1.0 - baseline_cost_rate) * (1.0 + gross_baseline_return) - 1.0
        taa_cost_amount = taa_nav * taa_cost_rate
        baseline_cost_amount = baseline_nav * baseline_cost_rate
        taa_nav *= 1.0 + net_taa_return
        baseline_nav *= 1.0 + net_baseline_return

        offset = asset_count * 2
        path[period, offset] = tilt_scale
        path[period, offset + 1] = taa_turnover
        path[period, offset + 2] = taa_cost_rate
        path[period, offset + 3] = taa_cost_amount
        path[period, offset + 4] = gross_baseline_return
        path[period, offset + 5] = net_baseline_return
        path[period, offset + 6] = gross_taa_return
        path[period, offset + 7] = net_taa_return
        path[period, offset + 8] = baseline_nav
        path[period, offset + 9] = taa_nav
        path[period, offset + 10] = baseline_turnover
        path[period, offset + 11] = baseline_cost_rate
        path[period, offset + 12] = baseline_cost_amount
        path[period, offset + 13] = gross_taa_return - gross_baseline_return
        totals[0] += taa_turnover
        totals[1] += taa_cost_amount
        totals[2] += baseline_turnover
        totals[3] += baseline_cost_amount

        if use_signal[period] == 1:
            for state in range(state_count):
                state_active_return = 0.0
                for asset in range(asset_count):
                    state_active_return += state_tilts[state, asset] * asset_returns[period, asset]
                contributions[period, state] = (
                    tilt_scale * probabilities[period, state] * state_active_return
                )
                state_summary[state, 0] += probabilities[period, state]
                state_summary[state, 1] += contributions[period, state]
                totals[8] += contributions[period, state]
                if probabilities[period, state] > 0.0:
                    state_summary[state, 2] += 1.0
        else:
            totals[9] += 1.0

        for asset in range(asset_count):
            taa_pretrade[asset] = (
                path[period, asset]
                * (1.0 + asset_returns[period, asset])
                / (1.0 + gross_taa_return)
            )
            baseline_pretrade[asset] = (
                base_weights[asset]
                * (1.0 + asset_returns[period, asset])
                / (1.0 + gross_baseline_return)
            )
    totals[4] = totals[0] / period_count
    totals[5] = totals[2] / period_count
    totals[6] = taa_nav - baseline_nav
    totals[7] = taa_nav / baseline_nav - 1.0
    return path, contributions, state_summary, totals


_PERFORMANCE_SIGNATURE = float64[::1](float64[::1], float64[::1], float64)


@njit(_PERFORMANCE_SIGNATURE, cache=True, nogil=True)
def _performance_kernel(
    period_returns: np.ndarray,
    nav: np.ndarray,
    periods_per_year: float,
) -> np.ndarray:
    """Return total, annualized, volatility, Sharpe and max drawdown."""

    count = period_returns.size
    result = np.empty(5, dtype=np.float64)
    total_return = nav[-1] - 1.0
    result[0] = total_return
    result[1] = (1.0 + total_return) ** (periods_per_year / count) - 1.0
    if count > 1:
        mean = 0.0
        for index in range(count):
            mean += period_returns[index]
        mean /= count
        variance = 0.0
        for index in range(count):
            difference = period_returns[index] - mean
            variance += difference * difference
        variance /= count - 1
        standard_deviation = np.sqrt(variance)
        result[2] = standard_deviation * np.sqrt(periods_per_year)
        result[3] = mean / standard_deviation * np.sqrt(periods_per_year) if standard_deviation > 0.0 else np.nan
    else:
        result[2] = np.nan
        result[3] = np.nan
    peak = 1.0
    maximum_drawdown = 0.0
    for index in range(nav.size):
        if nav[index] > peak:
            peak = nav[index]
        drawdown = nav[index] / peak - 1.0
        if drawdown < maximum_drawdown:
            maximum_drawdown = drawdown
    result[4] = maximum_drawdown
    return result


_PROBABILITY_SIGNATURE = float64[::1](float64[::1])


@njit(_PROBABILITY_SIGNATURE, cache=True, nogil=True)
def _normalize_probability_kernel(values: np.ndarray) -> np.ndarray:
    total = 0.0
    for index in range(values.size):
        value = values[index]
        if not np.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError("INVALID_REGIME_PROBABILITIES")
        total += value
    if abs(total - 1.0) > WEIGHT_TOLERANCE:
        raise ValueError("INVALID_REGIME_PROBABILITIES")
    result = np.empty(values.size, dtype=np.float64)
    for index in range(values.size):
        result[index] = values[index] / total
    return result


_VECTOR_VALIDATION_SIGNATURE = types.Tuple((float64, types.int64))(
    float64[::1],
    float64,
    float64,
    float64,
    float64,
)


@njit(_VECTOR_VALIDATION_SIGNATURE, cache=True, nogil=True)
def _weight_vector_validation_kernel(
    values: np.ndarray,
    expected_sum: float,
    tolerance: float,
    minimum: float,
    maximum: float,
) -> tuple[float, int]:
    """Validate a weight/tilt vector without Python arithmetic.

    Status bits: 1=empty/invalid contract, 2=non-finite value,
    4=below minimum, 8=above maximum, 16=sum mismatch.
    """

    if (
        values.size == 0
        or not np.isfinite(expected_sum)
        or not np.isfinite(tolerance)
        or tolerance < 0.0
        or np.isnan(minimum)
        or np.isnan(maximum)
        or minimum > maximum
    ):
        return np.nan, 1
    total = 0.0
    status = 0
    for value in values:
        if not np.isfinite(value):
            status |= 2
            continue
        total += value
        if value < minimum - tolerance:
            status |= 4
        if value > maximum + tolerance:
            status |= 8
    if status & 2:
        return np.nan, status
    if abs(total - expected_sum) > tolerance:
        status |= 16
    return total, status


_PATH_VALIDATION_SIGNATURE = types.int64(float64[:, ::1], types.int64, float64)


@njit(_PATH_VALIDATION_SIGNATURE, cache=True, nogil=True)
def _taa_output_validation_kernel(
    path: np.ndarray,
    asset_count: int,
    tolerance: float,
) -> int:
    """Validate every NJIT path cell, return and weight sum in one pass.

    Status bits: 1=non-finite, 2=return at/below -100%,
    4=target/pre-trade weights do not sum to one, 8=invalid shape.
    """

    required_columns = asset_count * 2 + 14
    if asset_count <= 0 or path.shape[0] == 0 or path.shape[1] < required_columns:
        return 8
    status = 0
    offset = asset_count * 2
    for period in range(path.shape[0]):
        target_sum = 0.0
        pretrade_sum = 0.0
        for column in range(path.shape[1]):
            if not np.isfinite(path[period, column]):
                status |= 1
        for asset in range(asset_count):
            target_sum += path[period, asset]
            pretrade_sum += path[period, asset_count + asset]
        if (
            abs(target_sum - 1.0) > tolerance
            or abs(pretrade_sum - 1.0) > tolerance
        ):
            status |= 4
        if path[period, offset + 5] <= -1.0 or path[period, offset + 7] <= -1.0:
            status |= 2
    return status


_TAA_KERNELS = (
    _taa_path_kernel,
    _performance_kernel,
    _normalize_probability_kernel,
    _weight_vector_validation_kernel,
    _taa_output_validation_kernel,
)
_TAA_KERNEL_NAMES = (
    "taa_path_kernel",
    "performance_kernel",
    "normalize_probability_kernel",
    "weight_vector_validation_kernel",
    "taa_output_validation_kernel",
)

for _kernel in _TAA_KERNELS:
    _kernel.disable_compile()


def _finite_number(value: Any, code: str, message: str, field: str) -> float:
    if isinstance(value, bool):
        raise ValidationError(code, message, field)
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError(code, message, field) from exc
    if not np.isfinite(result):
        raise ValidationError(code, message, field)
    return result


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _iso_date(value: Any, field: str) -> str:
    try:
        parsed = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError("INVALID_TAA_DATE", f"{field} 必须是有效日期。", field) from exc
    if pd.isna(parsed):
        raise ValidationError("INVALID_TAA_DATE", f"{field} 必须是有效日期。", field)
    if parsed.tzinfo is not None:
        parsed = parsed.tz_localize(None)
    return parsed.normalize().date().isoformat()


def _validate_base_weights(raw: Any) -> tuple[list[str], dict[str, float]]:
    if not isinstance(raw, dict) or not 1 <= len(raw) <= MAX_ASSETS:
        raise ValidationError(
            "INVALID_BASE_WEIGHTS",
            f"base_weights 必须包含 1 至 {MAX_ASSETS} 个资产。",
            "base_weights",
        )
    weights: dict[str, float] = {}
    for key, value in raw.items():
        asset = str(key).strip()
        if not asset or asset in weights:
            raise ValidationError("INVALID_BASE_WEIGHTS", "资产代码不能为空或重复。", "base_weights")
        number = _finite_number(
            value,
            "INVALID_BASE_WEIGHTS",
            "基础权重必须是有限数值。",
            f"base_weights.{asset}",
        )
        weights[asset] = number
    _, validation_status = _weight_vector_validation_kernel(
        np.ascontiguousarray(list(weights.values()), dtype=np.float64),
        np.float64(1.0),
        np.float64(WEIGHT_TOLERANCE),
        np.float64(0.0),
        np.float64(1.0),
    )
    if validation_status & (2 | 4 | 8):
        raise ValidationError(
            "INVALID_BASE_WEIGHTS",
            "基础权重必须是位于 [0, 1] 的有限数值。",
            "base_weights",
        )
    if validation_status & 16:
        raise ValidationError("INVALID_BASE_WEIGHTS", "基础权重之和必须接近 1。", "base_weights")
    assets = sorted(weights)
    return assets, {asset: weights[asset] for asset in assets}


def _validate_state_tilts(
    raw: Any,
    states: list[str],
    assets: list[str],
) -> dict[str, dict[str, float]]:
    if not isinstance(raw, dict) or set(raw) != set(states):
        raise ValidationError(
            "INVALID_STATE_TILTS",
            "state_tilts 必须逐一覆盖运行中的全部状态。",
            "state_tilts",
        )
    result: dict[str, dict[str, float]] = {}
    for state in states:
        values = raw.get(state)
        if not isinstance(values, dict) or set(values) != set(assets):
            raise ValidationError(
                "INVALID_STATE_TILTS",
                f"状态 {state} 的倾斜必须逐一覆盖全部资产。",
                f"state_tilts.{state}",
            )
        tilt = {
            asset: _finite_number(
                values[asset],
                "INVALID_STATE_TILTS",
                "状态倾斜必须是有限数值。",
                f"state_tilts.{state}.{asset}",
            )
            for asset in assets
        }
        _, validation_status = _weight_vector_validation_kernel(
            np.ascontiguousarray(list(tilt.values()), dtype=np.float64),
            np.float64(0.0),
            np.float64(WEIGHT_TOLERANCE),
            np.float64(-np.inf),
            np.float64(np.inf),
        )
        if validation_status != 0:
            raise ValidationError(
                "INVALID_STATE_TILTS",
                f"状态 {state} 的资产倾斜之和必须接近 0。",
                f"state_tilts.{state}",
            )
        result[state] = tilt
    return result


def _validate_return_rows(raw: Any, assets: list[str]) -> list[dict[str, Any]]:
    if not isinstance(raw, list) or not 1 <= len(raw) <= MAX_RETURN_ROWS:
        raise ValidationError(
            "INVALID_ASSET_RETURNS",
            f"asset_returns 必须包含 1 至 {MAX_RETURN_ROWS} 条内联收益记录。",
            "asset_returns",
        )
    allowed = {"date", "observation_date", "period_start", *assets}
    rows: list[dict[str, Any]] = []
    seen_dates: set[str] = set()
    for index, source in enumerate(raw):
        if not isinstance(source, dict):
            raise ValidationError(
                "INVALID_ASSET_RETURNS",
                "每条资产收益必须是对象。",
                f"asset_returns.{index}",
            )
        unexpected = set(source) - allowed
        if unexpected:
            raise ValidationError(
                "UNEXPECTED_ASSET_RETURN_FIELD",
                f"资产收益包含未声明字段：{', '.join(sorted(unexpected))}。",
                f"asset_returns.{index}",
            )
        date_value = source.get("date") if source.get("date") is not None else source.get("observation_date")
        row_date = _iso_date(date_value, f"asset_returns.{index}.date")
        if source.get("date") is not None and source.get("observation_date") is not None:
            observation_date = _iso_date(source["observation_date"], f"asset_returns.{index}.observation_date")
            if observation_date != row_date:
                raise ValidationError(
                    "CONFLICTING_RETURN_DATE",
                    "date 与 observation_date 不一致。",
                    f"asset_returns.{index}",
                )
        if row_date in seen_dates:
            raise ValidationError(
                "DUPLICATE_RETURN_DATE",
                f"资产收益日期 {row_date} 重复。",
                f"asset_returns.{index}.date",
            )
        seen_dates.add(row_date)
        row: dict[str, Any] = {"date": row_date}
        if source.get("period_start") is not None:
            row["period_start"] = _iso_date(
                source["period_start"],
                f"asset_returns.{index}.period_start",
            )
        for asset in assets:
            if asset not in source or source[asset] is None:
                raise ValidationError(
                    "MISSING_ASSET_RETURN",
                    f"{row_date} 缺少资产 {asset} 的收益，不能按 0 填充。",
                    f"asset_returns.{index}.{asset}",
                )
            value = _finite_number(
                source[asset],
                "INVALID_ASSET_RETURN",
                f"{row_date} 的资产 {asset} 收益必须是有限数值。",
                f"asset_returns.{index}.{asset}",
            )
            if value <= -1.0:
                raise ValidationError(
                    "INVALID_ASSET_RETURN",
                    "单期资产收益必须大于 -100%。",
                    f"asset_returns.{index}.{asset}",
                )
            row[asset] = value
        rows.append(row)
    rows = sorted(rows, key=lambda item: item["date"])
    previous_end: str | None = None
    for index, row in enumerate(rows):
        explicit_start = row.get("period_start")
        period_start = explicit_start if explicit_start is not None else previous_end
        if period_start is not None and period_start >= row["date"]:
            raise ValidationError(
                "INVALID_RETURN_PERIOD",
                "period_start 必须严格早于收益期末 date。",
                f"asset_returns.{index}.period_start",
            )
        if previous_end is not None and period_start is not None and period_start < previous_end:
            raise ValidationError(
                "OVERLAPPING_RETURN_PERIOD",
                "资产收益区间不能与上一期重叠。",
                f"asset_returns.{index}.period_start",
            )
        row["period_start"] = period_start
        previous_end = row["date"]
    return rows


def _validate_limits(raw: Any, base: dict[str, float]) -> dict[str, float]:
    if raw is None:
        raw = {}
    if not isinstance(raw, dict) or set(raw) - {"min_weight", "max_weight", "max_abs_tilt"}:
        raise ValidationError(
            "INVALID_TAA_LIMITS",
            "limits 仅支持 min_weight、max_weight 和 max_abs_tilt。",
            "limits",
        )
    minimum = _finite_number(
        raw.get("min_weight", 0.0),
        "INVALID_TAA_LIMITS",
        "min_weight 必须是有限数值。",
        "limits.min_weight",
    )
    maximum = _finite_number(
        raw.get("max_weight", 1.0),
        "INVALID_TAA_LIMITS",
        "max_weight 必须是有限数值。",
        "limits.max_weight",
    )
    max_abs_tilt = _finite_number(
        raw.get("max_abs_tilt", 1.0),
        "INVALID_TAA_LIMITS",
        "max_abs_tilt 必须是有限数值。",
        "limits.max_abs_tilt",
    )
    if minimum < 0 or maximum > 1 or minimum > maximum or max_abs_tilt < 0 or max_abs_tilt > 1:
        raise ValidationError("INVALID_TAA_LIMITS", "权重上下限和最大倾斜必须位于 [0, 1]。", "limits")
    _, validation_status = _weight_vector_validation_kernel(
        np.ascontiguousarray(list(base.values()), dtype=np.float64),
        np.float64(1.0),
        np.float64(WEIGHT_TOLERANCE),
        np.float64(minimum),
        np.float64(maximum),
    )
    if validation_status & (2 | 4 | 8):
        raise ValidationError("INVALID_TAA_LIMITS", "基础权重不满足所设上下限。", "limits")
    return {"min_weight": minimum, "max_weight": maximum, "max_abs_tilt": max_abs_tilt}


def _regime_points(run: dict[str, Any]) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for index, source in enumerate(run.get("series") or []):
        effective = source.get("effective_date")
        if effective is None:
            continue
        points.append(
            {
                "effective_date": _iso_date(effective, f"run.series.{index}.effective_date"),
                "observation_date": _iso_date(source.get("observation_date"), f"run.series.{index}.observation_date"),
                "recognized_at": _iso_date(source.get("recognized_at"), f"run.series.{index}.recognized_at"),
                "probabilities": source.get("probabilities"),
                "probability_source": source.get("probability_source", "unspecified"),
                "confidence": source.get("confidence"),
            }
        )
    return sorted(points, key=lambda item: (item["effective_date"], item["observation_date"]))


def _validated_probabilities(raw: Any, states: list[str]) -> tuple[dict[str, float] | None, str | None]:
    if not isinstance(raw, dict) or not raw:
        return None, "missing_probabilities"
    if set(raw) - set(states):
        raise ValidationError(
            "INVALID_REGIME_PROBABILITIES",
            "情景概率包含未知状态。",
            "run.series.probabilities",
        )
    values = []
    for state in states:
        values.append(
            _finite_number(
                raw.get(state, 0.0),
                "INVALID_REGIME_PROBABILITIES",
                "情景概率必须是有限数值。",
                "run.series.probabilities",
            )
        )
    try:
        normalized = _normalize_probability_kernel(
            np.ascontiguousarray(values, dtype=np.float64)
        )
    except ValueError as exc:
        raise ValidationError(
            "INVALID_REGIME_PROBABILITIES",
            "情景概率必须位于 [0, 1] 且合计接近 1。",
            "run.series.probabilities",
        ) from exc
    return {
        state: float(normalized[index]) for index, state in enumerate(states)
    }, None


def _performance(
    period_returns: list[float],
    nav: list[float],
    periods_per_year: int,
) -> dict[str, float | None]:
    values = np.ascontiguousarray(period_returns, dtype=np.float64)
    nav_values = np.ascontiguousarray(nav, dtype=np.float64)
    computed = _performance_kernel(values, nav_values, float(periods_per_year))
    return {
        "total_return": float(computed[0]),
        "annualized_return": float(computed[1]) if np.isfinite(computed[1]) else None,
        "annualized_volatility": float(computed[2]) if np.isfinite(computed[2]) else None,
        "sharpe": float(computed[3]) if np.isfinite(computed[3]) else None,
        "max_drawdown": float(computed[4]),
    }


def _taa_kernel_signatures() -> dict[str, list[str]]:
    return {
        name: [str(item) for item in kernel.signatures]
        for name, kernel in zip(_TAA_KERNEL_NAMES, _TAA_KERNELS)
    }


def taa_numba_execution_audit() -> dict[str, Any]:
    signature_map = _taa_kernel_signatures()
    compiled_signatures = [
        signature for values in signature_map.values() for signature in values
    ]
    audit = validate_execution_audit(
        {
            "engine": TAA_ENGINE_VERSION,
            "backend": "numba_njit_fixed_signature",
            "kernel_version": TAA_KERNEL_VERSION,
            "kernel_names": list(_TAA_KERNEL_NAMES),
            "kernel_signatures": signature_map,
            "kernel_fingerprint": _stable_hash(
                {
                    "engine": TAA_ENGINE_VERSION,
                    "kernel_version": TAA_KERNEL_VERSION,
                    "kernel_signatures": signature_map,
                }
            ),
            "nopython": all(
                bool(kernel.nopython_signatures)
                and all(
                    not compilation.objectmode
                    for compilation in kernel.overloads.values()
                )
                for kernel in _TAA_KERNELS
            ),
            "object_mode": 0,
            "python_fallback": 0,
            "typed_indicator_dag": False,
            "fully_warmed": all(len(values) == 1 for values in signature_map.values()),
            "note": (
                "业务解析与序列化留在 Python；权重校验、收益、成本、归因、"
                "净值、路径完整性、回撤与绩效均由固定签名 NJIT 内核执行。"
            ),
        }
    )
    return {
        **audit,
        "complete": len(compiled_signatures) == len(_TAA_KERNELS),
        "compiled_signatures": compiled_signatures,
    }


def warm_taa_numba_kernels() -> dict[str, Any]:
    """Verify both fixed signatures before application readiness is exposed."""

    returns = np.ascontiguousarray([[0.0, 0.0], [0.01, -0.002]], dtype=np.float64)
    probabilities = np.ascontiguousarray([[0.5, 0.5], [0.7, 0.3]], dtype=np.float64)
    use_signal = np.ascontiguousarray([0, 1], dtype=np.uint8)
    tilts = np.ascontiguousarray([[0.1, -0.1], [-0.1, 0.1]], dtype=np.float64)
    base = np.ascontiguousarray([0.6, 0.4], dtype=np.float64)
    path, _, _, _ = _taa_path_kernel(
        returns,
        probabilities,
        use_signal,
        tilts,
        base,
        0.0,
        1.0,
        0.2,
        5.0,
    )
    _performance_kernel(
        np.ascontiguousarray(path[:, 2 * base.size + 7]),
        np.ascontiguousarray(path[:, 2 * base.size + 9]),
        252.0,
    )
    _normalize_probability_kernel(
        np.ascontiguousarray([0.5, 0.5], dtype=np.float64)
    )
    _weight_vector_validation_kernel(base, 1.0, WEIGHT_TOLERANCE, 0.0, 1.0)
    _taa_output_validation_kernel(path, base.size, WEIGHT_TOLERANCE)
    audit = taa_numba_execution_audit()
    if not audit["complete"]:
        raise RuntimeError("TAA NJIT fixed-signature warmup did not complete")
    return audit


def run_taa_backtest(
    run: dict[str, Any],
    request: dict[str, Any],
    gate: dict[str, Any],
) -> dict[str, Any]:
    """Run a probability-weighted TAA overlay without mutating the source run."""

    assets, base = _validate_base_weights(request.get("base_weights"))
    states = [str(item.get("id")) for item in run.get("states") or [] if item.get("id")]
    if not states:
        raise ValidationError(
            "MISSING_REGIME_STATES",
            "历史情景运行没有可用于 TAA 的状态字典。",
            "run_id",
        )
    tilts = _validate_state_tilts(request.get("state_tilts"), states, assets)
    return_rows = _validate_return_rows(request.get("asset_returns"), assets)
    limits = _validate_limits(request.get("limits"), base)
    transaction_cost_bps = _finite_number(
        request.get("transaction_cost_bps", 0.0),
        "INVALID_TRANSACTION_COST",
        "transaction_cost_bps 必须是有限数值。",
        "transaction_cost_bps",
    )
    confidence_floor = _finite_number(
        request.get("confidence_floor", 0.0),
        "INVALID_CONFIDENCE_FLOOR",
        "confidence_floor 必须是有限数值。",
        "confidence_floor",
    )
    if not 0 <= transaction_cost_bps <= 1000:
        raise ValidationError(
            "INVALID_TRANSACTION_COST",
            "transaction_cost_bps 必须位于 [0, 1000]。",
            "transaction_cost_bps",
        )
    if not 0 <= confidence_floor <= 1:
        raise ValidationError("INVALID_CONFIDENCE_FLOOR", "confidence_floor 必须位于 [0, 1]。", "confidence_floor")
    periods_per_year_value = _finite_number(
        request.get("periods_per_year", 252),
        "INVALID_PERIODS_PER_YEAR",
        "periods_per_year 必须是整数。",
        "periods_per_year",
    )
    periods_per_year = int(periods_per_year_value)
    if float(periods_per_year) != periods_per_year_value:
        raise ValidationError("INVALID_PERIODS_PER_YEAR", "periods_per_year 必须是整数。", "periods_per_year")
    if not 1 <= periods_per_year <= 3660:
        raise ValidationError("INVALID_PERIODS_PER_YEAR", "periods_per_year 必须位于 [1, 3660]。", "periods_per_year")
    max_signal_age_value = _finite_number(
        request.get("max_signal_age_days", 31),
        "INVALID_SIGNAL_AGE",
        "max_signal_age_days 必须是整数。",
        "max_signal_age_days",
    )
    max_signal_age_days = int(max_signal_age_value)
    if float(max_signal_age_days) != max_signal_age_value:
        raise ValidationError("INVALID_SIGNAL_AGE", "max_signal_age_days 必须是整数。", "max_signal_age_days")
    if not 1 <= max_signal_age_days <= 3650:
        raise ValidationError("INVALID_SIGNAL_AGE", "max_signal_age_days 必须位于 [1, 3650]。", "max_signal_age_days")

    regime_points = _regime_points(run)
    point_index = 0
    latest_point: dict[str, Any] | None = None
    fallback_reasons: Counter[str] = Counter()
    return_matrix = np.ascontiguousarray(
        [[float(row[asset]) for asset in assets] for row in return_rows],
        dtype=np.float64,
    )
    probability_matrix = np.zeros((len(return_rows), len(states)), dtype=np.float64)
    use_signal = np.zeros(len(return_rows), dtype=np.uint8)
    signal_contexts: list[dict[str, Any]] = []

    for period_index, row in enumerate(return_rows):
        period_date = row["date"]
        period_start = row.get("period_start")
        while (
            period_start is not None
            and point_index < len(regime_points)
            and regime_points[point_index]["effective_date"] <= period_start
        ):
            latest_point = regime_points[point_index]
            point_index += 1

        probabilities: dict[str, float] | None = None
        source_probabilities: dict[str, float] | None = None
        fallback_reason: str | None = None
        confidence: float | None = None
        if period_start is None:
            fallback_reason = "no_safe_period_start"
        elif latest_point is None:
            fallback_reason = "no_effective_regime"
        else:
            probabilities, fallback_reason = _validated_probabilities(latest_point.get("probabilities"), states)
            source_probabilities = dict(probabilities) if probabilities is not None else None
            raw_confidence = latest_point.get("confidence")
            if probabilities is not None:
                if raw_confidence is None:
                    fallback_reason = "missing_confidence"
                    probabilities = None
                else:
                    confidence = _finite_number(
                        raw_confidence,
                        "INVALID_REGIME_CONFIDENCE",
                        "情景置信度必须是有限数值。",
                        "run.series.confidence",
                    )
                    if confidence < 0 or confidence > 1:
                        raise ValidationError(
                            "INVALID_REGIME_CONFIDENCE",
                            "情景置信度必须位于 [0, 1]。",
                            "run.series.confidence",
                        )
                    if confidence < confidence_floor:
                        fallback_reason = "confidence_below_floor"
                        probabilities = None
            signal_age_days = (
                pd.Timestamp(period_start) - pd.Timestamp(latest_point["effective_date"])
            ).days
            if signal_age_days > max_signal_age_days:
                fallback_reason = "stale_regime_signal"
                probabilities = None
        if probabilities is not None:
            use_signal[period_index] = np.uint8(1)
            for state_index, state in enumerate(states):
                probability_matrix[period_index, state_index] = probabilities[state]
        else:
            fallback_reasons[fallback_reason or "no_valid_regime"] += 1
        signal_contexts.append(
            {
                "latest_point": latest_point,
                "source_probabilities": source_probabilities,
                "allocation_probabilities": probabilities,
                "fallback_reason": fallback_reason,
                "confidence": confidence,
            }
        )

    tilt_matrix = np.ascontiguousarray(
        [[tilts[state][asset] for asset in assets] for state in states],
        dtype=np.float64,
    )
    base_vector = np.ascontiguousarray([base[asset] for asset in assets], dtype=np.float64)
    path_matrix, contribution_matrix, state_summary_matrix, totals = _taa_path_kernel(
        return_matrix,
        np.ascontiguousarray(probability_matrix),
        np.ascontiguousarray(use_signal),
        tilt_matrix,
        base_vector,
        float(limits["min_weight"]),
        float(limits["max_weight"]),
        float(limits["max_abs_tilt"]),
        float(transaction_cost_bps),
    )
    offset = len(assets) * 2
    output_status = int(
        _taa_output_validation_kernel(
            path_matrix,
            np.int64(len(assets)),
            np.float64(WEIGHT_TOLERANCE),
        )
    )
    if output_status & (1 | 2 | 8):
        raise ValidationError(
            "NON_FINITE_TAA_RESULT",
            "NJIT 组合推进产生了非有限结果。",
            "asset_returns",
        )
    if output_status & 4:
        raise ValidationError(
            "INVALID_TAA_WEIGHTS",
            "NJIT 约束后的战术权重之和不为 1。",
            "state_tilts",
        )

    baseline_returns = path_matrix[:, offset + 5]
    taa_returns = path_matrix[:, offset + 7]
    baseline_nav_values = path_matrix[:, offset + 8]
    taa_nav_values = path_matrix[:, offset + 9]
    baseline_nav = [
        {"date": row["date"], "value": float(baseline_nav_values[index])}
        for index, row in enumerate(return_rows)
    ]
    taa_nav = [
        {"date": row["date"], "value": float(taa_nav_values[index])}
        for index, row in enumerate(return_rows)
    ]
    weight_path: list[dict[str, Any]] = []
    for period_index, row in enumerate(return_rows):
        context = signal_contexts[period_index]
        latest = context["latest_point"]
        allocation_probabilities = context["allocation_probabilities"]
        period_state_contributions = {
            state: float(contribution_matrix[period_index, state_index])
            for state_index, state in enumerate(states)
        }
        target_weights = {
            asset: float(path_matrix[period_index, asset_index])
            for asset_index, asset in enumerate(assets)
        }
        pretrade_weights = {
            asset: float(path_matrix[period_index, len(assets) + asset_index])
            for asset_index, asset in enumerate(assets)
        }
        weight_path.append(
            {
                "date": row["date"],
                "period_start": row.get("period_start"),
                "regime_observation_date": latest.get("observation_date") if latest else None,
                "regime_effective_date": latest.get("effective_date") if latest else None,
                "regime_recognized_at": latest.get("recognized_at") if latest else None,
                "probabilities": context["source_probabilities"] or {},
                "allocation_probabilities": allocation_probabilities or {},
                "probability_source": latest.get("probability_source") if latest else None,
                "confidence": context["confidence"],
                "fallback_to_base": allocation_probabilities is None,
                "fallback_reason": context["fallback_reason"],
                "tilt_scale": float(path_matrix[period_index, offset]),
                "pretrade_weights": pretrade_weights,
                "weights": target_weights,
                "turnover": float(path_matrix[period_index, offset + 1]),
                "transaction_cost_rate": float(path_matrix[period_index, offset + 2]),
                "transaction_cost_amount": float(path_matrix[period_index, offset + 3]),
                "baseline_turnover": float(path_matrix[period_index, offset + 10]),
                "baseline_transaction_cost_rate": float(path_matrix[period_index, offset + 11]),
                "baseline_transaction_cost_amount": float(path_matrix[period_index, offset + 12]),
                "gross_baseline_return": float(path_matrix[period_index, offset + 4]),
                "baseline_return": float(path_matrix[period_index, offset + 5]),
                "gross_taa_return": float(path_matrix[period_index, offset + 6]),
                "net_taa_return": float(path_matrix[period_index, offset + 7]),
                "gross_excess_return": float(path_matrix[period_index, offset + 13]),
                "state_contributions": period_state_contributions,
            }
        )
    state_contributions = [
        {
            "state_id": state,
            "probability_weight": float(state_summary_matrix[state_index, 0]),
            "gross_excess_return_contribution": float(state_summary_matrix[state_index, 1]),
            "active_periods": int(state_summary_matrix[state_index, 2]),
        }
        for state_index, state in enumerate(states)
    ]

    input_snapshot = {
        "regime_run_id": run.get("id"),
        "regime_run_content_hash": run.get("content_hash"),
        "asset_returns_hash": _stable_hash(return_rows),
        "parameters_hash": _stable_hash(
            {
                "base_weights": base,
                "state_tilts": tilts,
                "limits": limits,
                "transaction_cost_bps": transaction_cost_bps,
                "confidence_floor": confidence_floor,
                "periods_per_year": periods_per_year,
                "max_signal_age_days": max_signal_age_days,
            }
        ),
        "observations": len(return_rows),
        "start_date": return_rows[0]["date"],
        "end_date": return_rows[-1]["date"],
        "assets": assets,
    }
    baseline_metrics = _performance(
        baseline_returns,
        [item["value"] for item in baseline_nav],
        periods_per_year,
    )
    taa_metrics = _performance(
        taa_returns,
        [item["value"] for item in taa_nav],
        periods_per_year,
    )
    payload = {
        "schema_version": "1.0",
        "run_id": run.get("id"),
        "definition_id": run.get("definition_id"),
        "definition_revision": run.get("definition_revision"),
        "execution": taa_numba_execution_audit(),
        "gate": gate,
        "timing_policy": {
            "return_timestamp": "period_end",
            "period_start_policy": "explicit period_start, otherwise previous return period end; first row without period_start has no eligible signal",
            "signal_rule": "regime.effective_date <= asset_return.period_start",
            "same_period_end_signal_allowed": False,
            "same_day_signal_allowed": False,
            "max_signal_age_days": max_signal_age_days,
            "allocation_source": "probability_weighted_state_tilts",
        },
        "metrics_policy": {"annualization_periods": periods_per_year, "risk_free_rate": 0.0, "sharpe_method": "mean_period_return_over_sample_volatility"},
        "input_snapshot": input_snapshot,
        "parameters": {
            "base_weights": base,
            "state_tilts": tilts,
            "limits": limits,
            "transaction_cost_bps": transaction_cost_bps,
            "confidence_floor": confidence_floor,
            "periods_per_year": periods_per_year,
            "max_signal_age_days": max_signal_age_days,
        },
        "baseline": {"nav": baseline_nav, "metrics": baseline_metrics, "policy": "periodic_target_weight_with_same_cost_model"},
        "taa": {"nav": taa_nav, "metrics": taa_metrics, "policy": "periodic_target_weight_with_same_cost_model"},
        "weights": weight_path,
        "turnover_and_cost": {
            "total_turnover": float(totals[0]),
            "average_turnover": float(totals[4]),
            "total_transaction_cost": float(totals[1]),
            "baseline_total_turnover": float(totals[2]),
            "baseline_average_turnover": float(totals[5]),
            "baseline_total_transaction_cost": float(totals[3]),
        },
        "excess": {
            "total_return_difference": float(totals[6]),
            "relative_total_return": float(totals[7]),
            "gross_active_return_sum": float(totals[8]),
        },
        "state_contributions": state_contributions,
        "fallbacks": {
            "periods": int(totals[9]),
            "reasons": dict(sorted(fallback_reasons.items())),
        },
    }
    payload["snapshot_hash"] = _stable_hash(payload)
    return payload


__all__ = ["run_taa_backtest", "taa_numba_execution_audit", "warm_taa_numba_kernels"]

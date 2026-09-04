"""Numerical engines for deterministic and probabilistic scenario research."""

from __future__ import annotations

import copy
import math
from datetime import date
from typing import Any, Optional

import numpy as np

from custom_indicators.errors import ValidationError

from .contracts import PROBABILISTIC_METHODS
from .numba_kernels import (
    assert_scenario_stress_numba_ready,
    covariance_root_kernel,
    coverage_weights_kernel,
    deterministic_paths_kernel,
    empirical_regime_projection_kernel,
    expand_factor_path_kernel,
    factor_to_asset_kernel,
    fan_statistics_kernel,
    limit_evaluation_kernel,
    monte_carlo_projection_kernel,
    regime_paths_kernel,
    returns_to_nav_kernel,
    reverse_stress_kernel,
    scale_factor_path_kernel,
    seeded_factor_draws_kernel,
    seeded_uniform_draws_kernel,
    scenario_stress_numba_status,
    state_path_returns_kernel,
    state_weighted_contributions_kernel,
    transition_validation_kernel,
    transform_factor_draws_kernel,
)


MAX_HISTORICAL_ROWS = 20_000
MAX_PATHS = 50_000
MAX_PATH_CELLS = 2_000_000
MAX_FACTOR_DRAW_CELLS = 8_000_000
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ValidationError("INVALID_NUMBER", f"{field} 必须是有限数值。", field)
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError("INVALID_NUMBER", f"{field} 必须是有限数值。", field) from exc
    if not math.isfinite(result):
        raise ValidationError("INVALID_NUMBER", f"{field} 必须是有限数值。", field)
    return result


def _json_number(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _validate_return(value: float, field: str) -> float:
    if value <= -1.0:
        raise ValidationError(
            "RETURN_BELOW_NEGATIVE_ONE",
            "简单收益率不能小于或等于 -100%。",
            field,
            diagnostics=[{"value": value}],
        )
    return value


def _asset_ids(definition: dict[str, Any]) -> list[str]:
    return [str(item["id"]) for item in definition["assets"]]


def _factor_ids(definition: dict[str, Any]) -> list[str]:
    return [str(item["id"]) for item in definition["factors"]]


def _validate_factor_mapping(definition: dict[str, Any]) -> dict[str, dict[str, float]]:
    asset_ids = set(_asset_ids(definition))
    factor_ids = set(_factor_ids(definition))
    raw = definition["mapping"].get("factor_to_asset", {})
    if not isinstance(raw, dict):
        raise ValidationError("INVALID_FACTOR_MAPPING", "factor_to_asset 必须是对象。", "mapping.factor_to_asset")
    unknown_assets = sorted(set(raw) - asset_ids)
    if unknown_assets:
        raise ValidationError(
            "MAPPING_ASSET_DIMENSION_MISMATCH",
            "因子映射包含资产字典之外的资产。",
            "mapping.factor_to_asset",
            diagnostics=[{"unknown_assets": unknown_assets}],
        )
    result: dict[str, dict[str, float]] = {}
    for asset_id, row in raw.items():
        if not isinstance(row, dict):
            raise ValidationError("INVALID_FACTOR_MAPPING", "每个资产的因子映射必须是对象。", f"mapping.factor_to_asset.{asset_id}")
        unknown_factors = sorted(set(row) - factor_ids)
        if unknown_factors:
            raise ValidationError(
                "MAPPING_FACTOR_DIMENSION_MISMATCH",
                "因子映射包含因子字典之外的字段。",
                f"mapping.factor_to_asset.{asset_id}",
                diagnostics=[{"unknown_factors": unknown_factors}],
            )
        # JSON null is an explicit missing coefficient.  It remains absent
        # from the executable row so block/degrade coverage logic handles it;
        # it is never converted to a zero beta.
        result[asset_id] = {
            factor_id: _finite(value, f"mapping.factor_to_asset.{asset_id}.{factor_id}")
            for factor_id, value in row.items()
            if value is not None
        }
    return result


def _asset_intercepts(definition: dict[str, Any]) -> dict[str, float]:
    values = definition["mapping"].get("asset_intercepts", {})
    if not isinstance(values, dict):
        raise ValidationError("INVALID_ASSET_INTERCEPTS", "asset_intercepts 必须是对象。", "mapping.asset_intercepts")
    unknown = sorted(set(values) - set(_asset_ids(definition)))
    if unknown:
        raise ValidationError(
            "INTERCEPT_ASSET_DIMENSION_MISMATCH",
            "asset_intercepts 包含资产字典之外的资产。",
            "mapping.asset_intercepts",
            diagnostics=[{"unknown_assets": unknown}],
        )
    return {asset_id: _finite(value, f"mapping.asset_intercepts.{asset_id}") for asset_id, value in values.items()}


def _float64_1d(values: Any) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=np.float64)


def _float64_2d(values: Any) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=np.float64)


def _int64_1d(values: Any) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=np.int64)


def _weights_array(definition: dict[str, Any], portfolio: dict[str, Any]) -> np.ndarray:
    return _float64_1d([portfolio["weights"][asset_id] for asset_id in _asset_ids(definition)])


def _mapping_arrays(
    definition: dict[str, Any],
    mapping: dict[str, dict[str, float]],
    intercepts: dict[str, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    asset_ids = _asset_ids(definition)
    factor_ids = _factor_ids(definition)
    beta = np.zeros((len(asset_ids), len(factor_ids)), dtype=np.float64)
    available = np.zeros(len(asset_ids), dtype=np.uint8)
    intercept_array = np.zeros(len(asset_ids), dtype=np.float64)
    for asset_index, asset_id in enumerate(asset_ids):
        intercept_array[asset_index] = intercepts.get(asset_id, 0.0)
        row = mapping.get(asset_id)
        if row is None or not all(factor_id in row for factor_id in factor_ids):
            continue
        available[asset_index] = 1
        for factor_index, factor_id in enumerate(factor_ids):
            beta[asset_index, factor_index] = row[factor_id]
    return (
        np.ascontiguousarray(beta),
        np.ascontiguousarray(intercept_array),
        np.ascontiguousarray(available),
    )


def _policy_code(definition: dict[str, Any]) -> np.int64:
    return np.int64(0 if definition["mapping"]["missing_policy"] == "block" else 1)


def _asset_names_from_mask(asset_ids: list[str], mask: np.ndarray) -> list[str]:
    return [asset_id for asset_id, flag in zip(asset_ids, mask) if int(flag) == 1]


def _raise_coverage_status(
    definition: dict[str, Any],
    portfolio: dict[str, Any],
    status: int,
    ratio: float,
    missing_assets: list[str],
    field: str,
) -> None:
    if status == 1:
        raise ValidationError(
            "SCENARIO_MAPPING_INCOMPLETE",
            f"组合“{portfolio['label']}”存在未映射资产，阻断本次运行。",
            field,
            diagnostics=[{"portfolio_id": portfolio["id"], "coverage_ratio": ratio, "missing_assets": missing_assets}],
        )
    if status == 2:
        raise ValidationError(
            "SCENARIO_COVERAGE_BELOW_MINIMUM",
            f"组合“{portfolio['label']}”覆盖率低于允许的降级阈值。",
            field,
            diagnostics=[
                {
                    "portfolio_id": portfolio["id"],
                    "coverage_ratio": ratio,
                    "minimum_coverage": definition["mapping"]["minimum_coverage"],
                    "missing_assets": missing_assets,
                }
            ],
        )
    if status == 3:
        raise ValidationError(
            "DEGRADED_WEIGHT_NORMALIZATION_FAILED",
            "覆盖资产的净权重为 0，无法在不伪造缺失收益的前提下降级计算。",
            field,
        )


def _coverage_payload(
    definition: dict[str, Any],
    asset_ids: list[str],
    ratio: float,
    missing_mask: np.ndarray,
) -> dict[str, Any]:
    missing = _asset_names_from_mask(asset_ids, missing_mask)
    return {
        "status": "complete" if not missing else "degraded",
        "ratio": float(ratio),
        "covered_assets": sorted(set(asset_ids) - set(missing)),
        "missing_assets": missing,
        "policy": definition["mapping"]["missing_policy"],
        "minimum_coverage": definition["mapping"]["minimum_coverage"],
        "renormalized": bool(missing),
    }


_LIMIT_METRICS = (
    "terminal_return",
    "max_drawdown",
    "worst_step_return",
    "var_95",
    "es_95",
    "loss_probability",
    "target_hit_probability",
)
_LIMIT_OPERATORS = {"gt": 0, "gte": 1, "lt": 2, "lte": 3}


def _limit_breaches(
    definition: dict[str, Any],
    metrics: dict[str, Optional[float]],
    path: Optional[list[dict[str, Any]]] = None,
) -> tuple[list[dict[str, Any]], int, int]:
    metric_values = np.zeros(len(_LIMIT_METRICS), dtype=np.float64)
    metric_available = np.zeros(len(_LIMIT_METRICS), dtype=np.uint8)
    for metric_index, metric_name in enumerate(_LIMIT_METRICS):
        value = metrics.get(metric_name)
        if value is not None:
            metric_values[metric_index] = float(value)
            metric_available[metric_index] = 1
    metric_codes = _int64_1d([_LIMIT_METRICS.index(limit["metric"]) for limit in definition["limits"]])
    operator_codes = _int64_1d([_LIMIT_OPERATORS[limit["operator"]] for limit in definition["limits"]])
    thresholds = _float64_1d([limit["threshold"] for limit in definition["limits"]])
    path_returns = _float64_1d([point["return"] for point in path] if path else [])
    path_nav = _float64_1d([point["nav"] for point in path] if path else [])
    path_drawdown = _float64_1d([point["drawdown"] for point in path] if path else [])
    values, breached_values, first_indices, breach_count, overall_first = limit_evaluation_kernel(
        np.ascontiguousarray(metric_values),
        np.ascontiguousarray(metric_available),
        metric_codes,
        operator_codes,
        thresholds,
        np.float64(definition["initial_nav"]),
        path_returns,
        path_nav,
        path_drawdown,
        np.int64(1 if path else 0),
    )
    results: list[dict[str, Any]] = []
    for limit_index, limit in enumerate(definition["limits"]):
        breached_code = int(breached_values[limit_index])
        if breached_code < 0:
            results.append(
                {
                    **copy.deepcopy(limit),
                    "value": None,
                    "status": "not_evaluated",
                    "breached": None,
                    "first_breach_step": None,
                    "first_breach_date": None,
                }
            )
            continue
        first_index = int(first_indices[limit_index])
        first_point = path[first_index] if path and first_index >= 0 else None
        breached = breached_code == 1
        results.append(
            {
                **copy.deepcopy(limit),
                "value": float(values[limit_index]),
                "status": "breached" if breached else "passed",
                "breached": breached,
                "first_breach_step": first_point.get("step") if first_point else None,
                "first_breach_date": first_point.get("date") if first_point else None,
                "evaluation_scope": "path" if path else "terminal_distribution",
            }
        )
    return results, int(breach_count), int(overall_first)


def _deterministic_results(
    definition: dict[str, Any],
    steps: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    asset_ids = _asset_ids(definition)
    return_matrix = _float64_2d(
        [
            [np.nan if step["asset_returns"].get(asset_id) is None else step["asset_returns"][asset_id] for asset_id in asset_ids]
            for step in steps
        ]
    )
    for portfolio in definition["portfolios"]:
        output = deterministic_paths_kernel(
            return_matrix,
            _weights_array(definition, portfolio),
            _policy_code(definition),
            np.float64(definition["mapping"]["minimum_coverage"]),
            np.float64(definition["initial_nav"]),
        )
        (
            portfolio_returns,
            contribution_matrix,
            coverage_ratios,
            statuses,
            missing_by_step,
            nav_path,
            drawdown_path,
            summary_values,
            contribution_totals,
            aggregate_missing,
            minimum_coverage,
        ) = output
        for step_index, status_value in enumerate(statuses):
            status = int(status_value)
            if status == 0:
                continue
            field = f"scenario.steps.{step_index}"
            missing = _asset_names_from_mask(asset_ids, missing_by_step[step_index])
            if status in {1, 2, 3}:
                _raise_coverage_status(
                    definition,
                    portfolio,
                    status,
                    float(coverage_ratios[step_index]),
                    missing,
                    field,
                )
            if status in {4, 5}:
                raise ValidationError("RETURN_BELOW_NEGATIVE_ONE", "简单收益率不能小于或等于 -100%。", field)
            raise ValidationError("INVALID_SIMULATED_NAV", "情景净值出现非有限值或非正值。", field)
        path: list[dict[str, Any]] = []
        for step_index, step in enumerate(steps):
            contributions = {
                asset_id: (
                    None
                    if not math.isfinite(float(contribution_matrix[step_index, asset_index]))
                    else float(contribution_matrix[step_index, asset_index])
                )
                for asset_index, asset_id in enumerate(asset_ids)
            }
            path.append(
                {
                    "step": int(step.get("step", step_index + 1)),
                    "date": step.get("date"),
                    "return": float(portfolio_returns[step_index]),
                    "nav": float(nav_path[step_index]),
                    "drawdown": float(drawdown_path[step_index]),
                    "factor_shocks": copy.deepcopy(step.get("factor_shocks")),
                    "asset_returns": copy.deepcopy(step["asset_returns"]),
                    "contributions": contributions,
                    "coverage_ratio": float(coverage_ratios[step_index]),
                }
            )
        start_index = int(summary_values[4])
        trough_index = int(summary_values[5])
        recovery_index = int(summary_values[6])
        summary = {
            "initial_nav": float(definition["initial_nav"]),
            "terminal_nav": float(summary_values[0]),
            "terminal_return": float(summary_values[1]),
            "max_drawdown": float(summary_values[2]),
            "worst_step_return": float(summary_values[3]),
            "max_drawdown_start_step": (
                0 if trough_index >= 0 and start_index < 0 else int(path[start_index]["step"]) if start_index >= 0 else None
            ),
            "max_drawdown_trough_step": int(path[trough_index]["step"]) if trough_index >= 0 else None,
            "recovery_step": int(path[recovery_index]["step"]) if recovery_index >= 0 else None,
            "recovery_steps": None if not math.isfinite(float(summary_values[7])) else int(summary_values[7]),
            "recovered": bool(summary_values[8]),
        }
        limits, breach_count, overall_first = _limit_breaches(definition, summary, path)
        summary["breach_count"] = breach_count
        summary["first_breach_step"] = int(path[overall_first]["step"]) if overall_first >= 0 else None
        summary["first_breach_date"] = path[overall_first].get("date") if overall_first >= 0 else None
        coverage = _coverage_payload(definition, asset_ids, float(minimum_coverage), aggregate_missing)
        by_asset = {
            asset_id: None if not math.isfinite(float(contribution_totals[index])) else float(contribution_totals[index])
            for index, asset_id in enumerate(asset_ids)
        }
        results.append(
            {
                "portfolio_id": portfolio["id"],
                "name": portfolio["label"],
                "coverage": coverage,
                "summary": summary,
                "path": path,
                "contributions": {"by_asset": by_asset, "method": "sum_of_period_contributions"},
                "limits": limits,
            }
        )
    return results


def _factor_path_steps(definition: dict[str, Any]) -> list[dict[str, Any]]:
    scenario = definition["scenario"]
    factor_ids = _factor_ids(definition)
    factor_set = set(factor_ids)
    raw_path = scenario.get("factor_path")
    severity = _finite(scenario.get("severity", 1.0), "scenario.severity")
    if severity <= 0 or severity > 10:
        raise ValidationError("INVALID_SEVERITY", "severity 必须在 0（不含）至 10 之间。", "scenario.severity")
    if raw_path is not None:
        if not isinstance(raw_path, list) or not raw_path or len(raw_path) != definition["horizon"]:
            raise ValidationError("INVALID_FACTOR_PATH", "factor_path 必须是非空数组，且长度必须等于 horizon。", "scenario.factor_path")
        prior_step = 0
        prior_date: Optional[date] = None
        has_dates = [point.get("date") is not None for point in raw_path if isinstance(point, dict)]
        if has_dates and any(has_dates) and not all(has_dates):
            raise ValidationError("INCOMPLETE_FACTOR_PATH_DATES", "factor_path 的 date 必须全部提供或全部省略。", "scenario.factor_path")
        metadata: list[tuple[int, Optional[str]]] = []
        raw_values: list[list[float]] = []
        for index, point in enumerate(raw_path):
            if not isinstance(point, dict) or not isinstance(point.get("shocks"), dict):
                raise ValidationError("INVALID_FACTOR_PATH", "factor_path 每期必须包含 shocks 对象。", f"scenario.factor_path.{index}")
            shocks = point["shocks"]
            if set(shocks) != factor_set:
                raise ValidationError(
                    "FACTOR_PATH_DIMENSION_MISMATCH",
                    "每期 shocks 必须逐项覆盖因子字典；缺失冲击不会按 0 处理。",
                    f"scenario.factor_path.{index}.shocks",
                    diagnostics=[{"missing_factors": sorted(factor_set - set(shocks)), "unknown_factors": sorted(set(shocks) - factor_set)}],
                )
            try:
                step_number = int(point.get("step", index + 1))
            except (TypeError, ValueError) as exc:
                raise ValidationError("INVALID_FACTOR_PATH_STEP", "factor_path.step 必须是递增整数。", f"scenario.factor_path.{index}.step") from exc
            if step_number <= prior_step:
                raise ValidationError("INVALID_FACTOR_PATH_STEP", "factor_path.step 必须严格递增。", f"scenario.factor_path.{index}.step")
            prior_step = step_number
            date_text = point.get("date")
            if date_text is not None:
                try:
                    parsed_date = date.fromisoformat(str(date_text))
                except (TypeError, ValueError) as exc:
                    raise ValidationError("INVALID_FACTOR_PATH_DATE", "factor_path.date 必须是 YYYY-MM-DD。", f"scenario.factor_path.{index}.date") from exc
                if prior_date is not None and parsed_date <= prior_date:
                    raise ValidationError("INVALID_FACTOR_PATH_DATE_ORDER", "factor_path.date 必须严格递增。", f"scenario.factor_path.{index}.date")
                prior_date = parsed_date
                date_text = parsed_date.isoformat()
            metadata.append((step_number, date_text))
            raw_values.append(
                [
                    _finite(shocks[factor_id], f"scenario.factor_path.{index}.shocks.{factor_id}")
                    for factor_id in factor_ids
                ]
            )
        scaled = scale_factor_path_kernel(_float64_2d(raw_values), np.float64(severity))
        return [
            {
                "step": step_number,
                "date": date_text,
                "factor_shocks": {
                    factor_id: float(scaled[step_index, factor_index])
                    for factor_index, factor_id in enumerate(factor_ids)
                },
            }
            for step_index, (step_number, date_text) in enumerate(metadata)
        ]
    shocks = scenario.get("shocks")
    if not isinstance(shocks, dict) or set(shocks) != factor_set:
        raise ValidationError(
            "FACTOR_SHOCK_DIMENSION_MISMATCH",
            "scenario.shocks 必须逐项覆盖因子字典；缺失冲击不会按 0 处理。",
            "scenario.shocks",
            diagnostics=[{"missing_factors": sorted(factor_set - set(shocks or {})), "unknown_factors": sorted(set(shocks or {}) - factor_set)}],
        )
    values = _float64_1d([_finite(shocks[factor_id], f"scenario.shocks.{factor_id}") for factor_id in factor_ids])
    shape = str(scenario.get("path_shape") or "linear")
    if shape not in {"linear", "instant"}:
        raise ValidationError("INVALID_PATH_SHAPE", "path_shape 必须是 linear 或 instant。", "scenario.path_shape")
    horizon = int(definition["horizon"])
    expanded = expand_factor_path_kernel(
        values,
        np.int64(horizon),
        np.int64(0 if shape == "linear" else 1),
        np.float64(severity),
    )
    return [
        {
            "step": step_index + 1,
            "date": None,
            "factor_shocks": {
                factor_id: float(expanded[step_index, factor_index])
                for factor_index, factor_id in enumerate(factor_ids)
            },
        }
        for step_index in range(horizon)
    ]


def _map_factor_steps(definition: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    mapping = _validate_factor_mapping(definition)
    intercepts = _asset_intercepts(definition)
    factor_ids = _factor_ids(definition)
    available_assets = {
        asset_id for asset_id, row in mapping.items() if all(factor_id in row for factor_id in factor_ids)
    }
    steps = _factor_path_steps(definition)
    asset_ids = _asset_ids(definition)
    beta, intercept_array, available_array = _mapping_arrays(definition, mapping, intercepts)
    shock_matrix = _float64_2d(
        [[point["factor_shocks"][factor_id] for factor_id in factor_ids] for point in steps]
    )
    return_matrix, status, bad_step, bad_asset = factor_to_asset_kernel(
        shock_matrix,
        beta,
        intercept_array,
        available_array,
    )
    if int(status) != 0:
        field = f"mapping.factor_to_asset.{asset_ids[int(bad_asset)]}"
        if int(status) == 2:
            raise ValidationError(
                "RETURN_BELOW_NEGATIVE_ONE",
                "简单收益率不能小于或等于 -100%。",
                field,
                diagnostics=[{"step": int(bad_step) + 1}],
            )
        raise ValidationError("INVALID_NUMBER", "因子映射产生了非有限收益。", field)
    mapped = [
        {
            **point,
            "asset_returns": {
                asset_id: None if int(available_array[asset_index]) == 0 else float(return_matrix[step_index, asset_index])
                for asset_index, asset_id in enumerate(asset_ids)
            },
        }
        for step_index, point in enumerate(steps)
    ]
    diagnostics = {
        "orientation": "asset_by_factor",
        "factor_ids": factor_ids,
        "asset_ids": _asset_ids(definition),
        "mapped_assets": sorted(available_assets),
        "unmapped_assets": sorted(set(_asset_ids(definition)) - available_assets),
        "coefficient_count": sum(len(row) for row in mapping.values()),
        "intercept_assets": sorted(intercepts),
        "response_space": definition["mapping"]["response_space"],
    }
    return mapped, diagnostics


def _historical_steps(definition: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = definition["scenario"].get("historical_returns")
    if not isinstance(rows, list) or not rows or len(rows) > MAX_HISTORICAL_ROWS:
        raise ValidationError(
            "INVALID_HISTORICAL_RETURNS",
            f"historical_returns 必须包含 1 至 {MAX_HISTORICAL_ROWS} 行。",
            "scenario.historical_returns",
        )
    if len(rows) != int(definition["horizon"]):
        raise ValidationError(
            "HISTORICAL_HORIZON_MISMATCH",
            "historical_returns 行数必须等于 horizon。",
            "horizon",
            diagnostics=[{"horizon": int(definition["horizon"]), "historical_rows": len(rows)}],
        )
    asset_ids = set(_asset_ids(definition))
    normalized: list[dict[str, Any]] = []
    seen_dates: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not isinstance(row.get("returns"), dict):
            raise ValidationError("INVALID_HISTORICAL_ROW", "历史重演每行必须包含 date 与 returns。", f"scenario.historical_returns.{index}")
        try:
            parsed_date = date.fromisoformat(str(row.get("date")))
        except (TypeError, ValueError) as exc:
            raise ValidationError("INVALID_HISTORICAL_DATE", "历史重演日期必须是 YYYY-MM-DD。", f"scenario.historical_returns.{index}.date") from exc
        date_text = parsed_date.isoformat()
        if date_text in seen_dates:
            raise ValidationError("DUPLICATE_HISTORICAL_DATE", "历史重演日期不能重复。", f"scenario.historical_returns.{index}.date")
        seen_dates.add(date_text)
        supplied = row["returns"]
        unknown = sorted(set(supplied) - asset_ids)
        if unknown:
            raise ValidationError(
                "HISTORICAL_ASSET_DIMENSION_MISMATCH",
                "历史收益包含资产字典之外的字段。",
                f"scenario.historical_returns.{index}.returns",
                diagnostics=[{"unknown_assets": unknown}],
            )
        clean: dict[str, Optional[float]] = {}
        for asset_id in asset_ids:
            raw_value = supplied.get(asset_id)
            clean[asset_id] = None if raw_value is None else _validate_return(
                _finite(raw_value, f"scenario.historical_returns.{index}.returns.{asset_id}"),
                f"scenario.historical_returns.{index}.returns.{asset_id}",
            )
        normalized.append({"date": date_text, "asset_returns": clean})
    normalized.sort(key=lambda item: item["date"])
    for index, point in enumerate(normalized, start=1):
        point["step"] = index
    snapshot = {
        "source": "inline_historical_asset_returns",
        "rows": len(normalized),
        "start_date": normalized[0]["date"],
        "end_date": normalized[-1]["date"],
        "sorted_chronologically": True,
        "missing_values": sum(value is None for point in normalized for value in point["asset_returns"].values()),
    }
    return normalized, snapshot


def _validate_path_count(scenario: dict[str, Any], horizon: int, prefix: str = "scenario") -> tuple[int, int]:
    try:
        path_count = int(scenario.get("path_count", 2000))
        seed = int(scenario.get("seed", 0))
    except (TypeError, ValueError) as exc:
        raise ValidationError("INVALID_SIMULATION_SIZE", "path_count 与 seed 必须是整数。", prefix) from exc
    if path_count < 100 or path_count > MAX_PATHS:
        raise ValidationError("INVALID_PATH_COUNT", f"path_count 必须在 100 至 {MAX_PATHS} 之间。", f"{prefix}.path_count")
    if path_count * horizon > MAX_PATH_CELLS:
        raise ValidationError(
            "SIMULATION_SIZE_EXCEEDED",
            f"path_count × horizon 不能超过 {MAX_PATH_CELLS}。",
            prefix,
            diagnostics=[{"path_count": path_count, "horizon": horizon}],
        )
    if seed < 0 or seed > 2**32 - 1:
        raise ValidationError("INVALID_RANDOM_SEED", "seed 必须在 0 至 2^32-1 之间。", f"{prefix}.seed")
    return path_count, seed


def _validate_correlation(value: Any, dimension: int) -> np.ndarray:
    try:
        matrix = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValidationError("INVALID_CORRELATION", "correlation 必须是数值矩阵。", "scenario.correlation") from exc
    if matrix.shape != (dimension, dimension):
        raise ValidationError(
            "CORRELATION_DIMENSION_MISMATCH",
            "相关矩阵维度必须与因子数量一致。",
            "scenario.correlation",
            diagnostics=[{"expected": [dimension, dimension], "actual": list(matrix.shape)}],
        )
    return np.ascontiguousarray(matrix, dtype=np.float64)


def _fan_summary(
    nav_paths: np.ndarray,
    terminal_returns: np.ndarray,
    max_drawdowns: np.ndarray,
    *,
    path_count: int,
    seed: int,
    target_return: float,
) -> dict[str, Any]:
    fan_values, terminal_values = fan_statistics_kernel(
        np.ascontiguousarray(nav_paths),
        _float64_1d(terminal_returns),
        _float64_1d(max_drawdowns),
        np.float64(target_return),
    )
    quantile_names = ("p05", "p25", "p50", "p75", "p95")
    fan_quantiles = {
        name: [float(value) for value in fan_values[index]]
        for index, name in enumerate(quantile_names)
    }
    terminal = {
        "p05": float(terminal_values[0]),
        "p25": float(terminal_values[1]),
        "p50": float(terminal_values[2]),
        "p75": float(terminal_values[3]),
        "p95": float(terminal_values[4]),
        "return_p05": float(terminal_values[5]),
        "return_p50": float(terminal_values[6]),
        "return_p95": float(terminal_values[7]),
        "var_95": float(terminal_values[8]),
        "es_95": float(terminal_values[9]),
        "loss_probability": float(terminal_values[10]),
        "target_hit_probability": float(terminal_values[11]),
        "average_max_drawdown": float(terminal_values[12]),
        "p95_max_drawdown": float(terminal_values[13]),
    }
    return {
        "fan": {
            "steps": list(range(nav_paths.shape[1])),
            "quantiles": fan_quantiles,
            "nav_unit": "index",
        },
        "sample_paths": [[float(value) for value in row] for row in nav_paths[:12]],
        "terminal": terminal,
        "path_count": path_count,
        "seed": seed,
        "target_return": target_return,
}


def _monte_carlo_results(definition: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scenario = definition["scenario"]
    factor_ids = _factor_ids(definition)
    factor_set = set(factor_ids)
    mapping = _validate_factor_mapping(definition)
    intercepts = _asset_intercepts(definition)
    distribution = str(scenario.get("distribution") or "normal").lower()
    if distribution not in {"normal", "student_t"}:
        raise ValidationError("UNSUPPORTED_DISTRIBUTION", "distribution 必须是 normal 或 student_t。", "scenario.distribution")
    means_raw = scenario.get("factor_means")
    volatility_raw = scenario.get("factor_volatilities")
    if not isinstance(means_raw, dict) or set(means_raw) != factor_set:
        raise ValidationError("FACTOR_MEAN_DIMENSION_MISMATCH", "factor_means 必须逐项覆盖因子字典。", "scenario.factor_means")
    if not isinstance(volatility_raw, dict) or set(volatility_raw) != factor_set:
        raise ValidationError("FACTOR_VOLATILITY_DIMENSION_MISMATCH", "factor_volatilities 必须逐项覆盖因子字典。", "scenario.factor_volatilities")
    means = _float64_1d([_finite(means_raw[item], f"scenario.factor_means.{item}") for item in factor_ids])
    volatilities = _float64_1d(
        [_finite(volatility_raw[item], f"scenario.factor_volatilities.{item}") for item in factor_ids]
    )
    correlation = _validate_correlation(scenario.get("correlation"), len(factor_ids))
    covariance_root, minimum_correlation_eigenvalue, covariance_status = covariance_root_kernel(
        volatilities,
        correlation,
    )
    if int(covariance_status) == 1:
        raise ValidationError(
            "INVALID_VOLATILITY",
            "因子波动率必须非负，且至少一个大于 0。",
            "scenario.factor_volatilities",
        )
    if int(covariance_status) == 2:
        raise ValidationError(
            "INVALID_CORRELATION",
            "相关矩阵必须有限、对称，对角线为 1，且系数位于 [-1,1]。",
            "scenario.correlation",
        )
    if int(covariance_status) == 3:
        raise ValidationError(
            "CORRELATION_NOT_POSITIVE_SEMIDEFINITE",
            "相关矩阵不是半正定矩阵。",
            "scenario.correlation",
            diagnostics=[{"minimum_eigenvalue": float(minimum_correlation_eigenvalue)}],
        )
    horizon = int(definition["horizon"])
    path_count, seed = _validate_path_count(scenario, horizon)
    factor_draw_cells = path_count * horizon * len(factor_ids)
    if factor_draw_cells > MAX_FACTOR_DRAW_CELLS:
        raise ValidationError(
            "FACTOR_SIMULATION_SIZE_EXCEEDED",
            f"path_count × horizon × factor_count 不能超过 {MAX_FACTOR_DRAW_CELLS}。",
            "scenario",
            diagnostics=[
                {
                    "path_count": path_count,
                    "horizon": horizon,
                    "factor_count": len(factor_ids),
                    "factor_draw_cells": factor_draw_cells,
                }
            ],
        )
    target_return = _finite(scenario.get("target_return", 0.0), "scenario.target_return")
    degrees_of_freedom: Optional[float] = None
    if distribution == "student_t":
        degrees_of_freedom = _finite(scenario.get("df", 5.0), "scenario.df")
        if not 2.0 < degrees_of_freedom <= 100.0:
            raise ValidationError("INVALID_STUDENT_T_DF", "student_t 的 df 必须在 2（不含）至 100 之间。", "scenario.df")

    available_assets = {
        asset_id for asset_id, row in mapping.items() if all(factor_id in row for factor_id in factor_ids)
    }
    asset_ids = _asset_ids(definition)
    beta, intercept_array, available_array = _mapping_arrays(definition, mapping, intercepts)
    if distribution == "student_t" and degrees_of_freedom is not None:
        distribution_code = np.int64(1)
        df_value = np.float64(degrees_of_freedom)
    else:
        distribution_code = np.int64(0)
        df_value = np.float64(5.0)
    factor_draws, chi_square_draws = seeded_factor_draws_kernel(
        np.int64(horizon),
        np.int64(path_count),
        np.int64(len(factor_ids)),
        np.int64(seed),
        distribution_code,
        df_value,
    )
    transform_factor_draws_kernel(
        factor_draws,
        chi_square_draws,
        np.ascontiguousarray(covariance_root),
        means,
        distribution_code,
        df_value,
    )

    results: list[dict[str, Any]] = []
    for portfolio in definition["portfolios"]:
        effective_weights, coverage_ratio, coverage_status, missing_mask = coverage_weights_kernel(
            _weights_array(definition, portfolio),
            available_array,
            _policy_code(definition),
            np.float64(definition["mapping"]["minimum_coverage"]),
        )
        missing_assets = _asset_names_from_mask(asset_ids, missing_mask)
        _raise_coverage_status(
            definition,
            portfolio,
            int(coverage_status),
            float(coverage_ratio),
            missing_assets,
            "mapping.factor_to_asset",
        )
        period_matrix, mean_contributions, projection_status, bad_step, _, minimum_return = monte_carlo_projection_kernel(
            factor_draws,
            beta,
            intercept_array,
            np.ascontiguousarray(effective_weights),
            available_array,
        )
        if int(projection_status) != 0:
            raise ValidationError(
                "INVALID_SIMULATED_RETURN",
                "模拟产生了无法形成有效净值的收益；请降低冲击或检查映射与杠杆。",
                "mapping.factor_to_asset",
                diagnostics=[{"step": int(bad_step) + 1, "minimum_return": _json_number(minimum_return)}],
            )
        nav_paths, terminal_returns, max_drawdowns, nav_status, bad_step, _, minimum_return = returns_to_nav_kernel(
            np.ascontiguousarray(period_matrix),
            np.float64(definition["initial_nav"]),
        )
        if int(nav_status) in {1, 2}:
            raise ValidationError(
                "INVALID_SIMULATED_RETURN",
                "模拟产生了无法形成有效净值的收益；请降低冲击或检查映射与杠杆。",
                "mapping.factor_to_asset",
                diagnostics=[{"step": int(bad_step) + 1, "minimum_return": _json_number(minimum_return)}],
            )
        if int(nav_status) == 3:
            raise ValidationError("INVALID_SIMULATED_NAV", "模拟净值出现非有限值或非正值。", "scenario")
        distribution_result = _fan_summary(
            nav_paths,
            terminal_returns,
            max_drawdowns,
            path_count=path_count,
            seed=seed,
            target_return=target_return,
        )
        terminal = distribution_result["terminal"]
        summary = {
            "initial_nav": float(definition["initial_nav"]),
            "terminal_nav": terminal["p50"],
            "terminal_return": terminal["return_p50"],
            "max_drawdown": terminal["p95_max_drawdown"],
            "worst_step_return": None,
            "var_95": terminal["var_95"],
            "es_95": terminal["es_95"],
            "loss_probability": terminal["loss_probability"],
            "target_hit_probability": terminal["target_hit_probability"],
        }
        limits, breach_count, _ = _limit_breaches(definition, summary)
        summary["breach_count"] = breach_count
        contribution_map = {
            asset_id: None if not math.isfinite(float(mean_contributions[index])) else float(mean_contributions[index])
            for index, asset_id in enumerate(asset_ids)
        }
        results.append(
            {
                "portfolio_id": portfolio["id"],
                "name": portfolio["label"],
                "coverage": _coverage_payload(definition, asset_ids, float(coverage_ratio), missing_mask),
                "summary": summary,
                "contributions": {"by_asset": contribution_map, "method": "mean_sum_of_period_contributions"},
                "distribution": distribution_result,
                "limits": limits,
            }
        )
    diagnostics = {
        "orientation": "asset_by_factor",
        "factor_ids": factor_ids,
        "asset_ids": asset_ids,
        "mapped_assets": sorted(available_assets),
        "unmapped_assets": sorted(set(asset_ids) - available_assets),
        "coefficient_count": sum(len(row) for row in mapping.values()),
        "distribution": distribution,
        "degrees_of_freedom": degrees_of_freedom,
        "correlation_minimum_eigenvalue": float(minimum_correlation_eigenvalue),
        "return_mapping": "factor_to_asset_log_return",
        "response_space": definition["mapping"]["response_space"],
        "random_generation": "numba_njit_seeded_sampling",
    }
    return results, diagnostics


def _normalized_transition(
    definition: dict[str, Any],
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, str, dict[str, Any]]:
    scenario = definition["scenario"]
    transition = scenario.get("_resolved_transition") or scenario.get("transition")
    if not isinstance(transition, dict):
        raise ValidationError("INVALID_TRANSITION", "regime_conditioned 必须配置 transition。", "scenario.transition")
    states = transition.get("states")
    matrix_raw = transition.get("matrix")
    if not isinstance(states, list) or not 2 <= len(states) <= 12:
        raise ValidationError("INVALID_TRANSITION_STATES", "transition.states 必须包含 2 至 12 个状态。", "scenario.transition.states")
    state_ids: list[str] = []
    normalized_states: list[dict[str, Any]] = []
    asset_ids = set(_asset_ids(definition))
    for index, state in enumerate(states):
        if not isinstance(state, dict) or not str(state.get("id") or "").strip():
            raise ValidationError("INVALID_TRANSITION_STATE", "每个状态必须包含 id。", f"scenario.transition.states.{index}")
        state_id = str(state["id"]).strip()
        if state_id in state_ids:
            raise ValidationError("DUPLICATE_TRANSITION_STATE", "状态 id 不能重复。", f"scenario.transition.states.{index}.id")
        returns = state.get("asset_returns")
        if not isinstance(returns, dict):
            raise ValidationError("INVALID_STATE_RETURNS", "每个状态必须配置 asset_returns。", f"scenario.transition.states.{index}.asset_returns")
        unknown = sorted(set(returns) - asset_ids)
        if unknown:
            raise ValidationError(
                "STATE_RETURN_DIMENSION_MISMATCH",
                "状态收益包含资产字典之外的字段。",
                f"scenario.transition.states.{index}.asset_returns",
                diagnostics=[{"unknown_assets": unknown}],
            )
        clean_returns = {
            asset_id: (
                None if returns.get(asset_id) is None else _validate_return(
                    _finite(returns[asset_id], f"scenario.transition.states.{index}.asset_returns.{asset_id}"),
                    f"scenario.transition.states.{index}.asset_returns.{asset_id}",
                )
            )
            for asset_id in asset_ids
        }
        state_ids.append(state_id)
        normalized_states.append({**copy.deepcopy(state), "id": state_id, "asset_returns": clean_returns})
    try:
        matrix = _float64_2d(matrix_raw)
    except (TypeError, ValueError) as exc:
        raise ValidationError("INVALID_TRANSITION_MATRIX", "transition.matrix 必须是数值矩阵。", "scenario.transition.matrix") from exc
    dimension = len(states)
    if matrix.shape != (dimension, dimension):
        raise ValidationError("TRANSITION_DIMENSION_MISMATCH", "转移矩阵维度必须与状态数量一致。", "scenario.transition.matrix")
    cumulative, transition_status = transition_validation_kernel(matrix)
    if int(transition_status) == 1:
        raise ValidationError("INVALID_TRANSITION_MATRIX", "转移概率必须非负，且每行之和等于 1。", "scenario.transition.matrix")
    if int(transition_status) == 2:
        raise ValidationError("DEGENERATE_TRANSITION", "转移矩阵没有随机分支，不能伪装为概率模拟。", "scenario.transition.matrix")
    initial_state = str(transition.get("initial_state") or "")
    if initial_state not in state_ids:
        raise ValidationError("INVALID_INITIAL_STATE", "initial_state 必须属于状态字典。", "scenario.transition.initial_state")
    return (
        normalized_states,
        matrix,
        np.ascontiguousarray(cumulative),
        initial_state,
        copy.deepcopy(transition.get("source", {"kind": "inline"})),
    )


def _regime_results(definition: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    states, matrix, cumulative, initial_state, source = _normalized_transition(definition)
    transition_config = definition["scenario"].get("_resolved_transition") or definition["scenario"]["transition"]
    horizon = int(definition["horizon"])
    path_count, seed = _validate_path_count(transition_config, horizon, "scenario.transition")
    target_return = _finite(transition_config.get("target_return", 0.0), "scenario.transition.target_return")
    state_ids = [state["id"] for state in states]
    state_index = {state_id: index for index, state_id in enumerate(state_ids)}
    uniform_draws = seeded_uniform_draws_kernel(
        np.int64(path_count),
        np.int64(max(horizon - 1, 1)),
        np.int64(seed),
    )
    state_paths, probability_matrix = regime_paths_kernel(
        uniform_draws,
        cumulative,
        np.int64(state_index[initial_state]),
        np.int64(horizon),
    )
    probabilities_by_step = [
        {
            state_id: float(probability_matrix[step_index, index])
            for index, state_id in enumerate(state_ids)
        }
        for step_index in range(horizon)
    ]
    asset_ids = _asset_ids(definition)
    state_asset_returns = _float64_2d(
        [
            [
                np.nan if state["asset_returns"].get(asset_id) is None else state["asset_returns"][asset_id]
                for asset_id in asset_ids
            ]
            for state in states
        ]
    )
    historical_distribution = transition_config.get("_historical_asset_distribution")
    grouped_samples = None
    state_offsets = None
    return_uniform_draws = None
    distribution_seed = None
    if historical_distribution is not None:
        if not isinstance(historical_distribution, dict):
            raise ValidationError(
                "HISTORICAL_STATE_DISTRIBUTION_INVALID",
                "历史状态收益分布运行参数无效。",
                "scenario.transition.asset_return_source",
            )
        try:
            grouped_samples = np.ascontiguousarray(
                historical_distribution["grouped_samples"], dtype=np.float64
            )
            state_offsets = np.ascontiguousarray(
                historical_distribution["state_offsets"], dtype=np.int64
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValidationError(
                "HISTORICAL_STATE_DISTRIBUTION_INVALID",
                "历史状态收益分布运行参数无效。",
                "scenario.transition.asset_return_source",
            ) from exc
        if (
            grouped_samples.ndim != 2
            or grouped_samples.shape[1] != len(asset_ids)
            or state_offsets.ndim != 1
            or state_offsets.shape[0] != len(state_ids) + 1
            or int(state_offsets[0]) != 0
            or int(state_offsets[-1]) != grouped_samples.shape[0]
        ):
            raise ValidationError(
                "HISTORICAL_STATE_DISTRIBUTION_INVALID",
                "历史状态收益分布维度或状态切片无效。",
                "scenario.transition.asset_return_source",
            )
        distribution_seed = seed ^ 0x9E3779B9
        return_uniform_draws = seeded_uniform_draws_kernel(
            np.int64(path_count),
            np.int64(horizon),
            np.int64(distribution_seed),
        )

    results: list[dict[str, Any]] = []
    for portfolio in definition["portfolios"]:
        projection = deterministic_paths_kernel(
            state_asset_returns,
            _weights_array(definition, portfolio),
            _policy_code(definition),
            np.float64(definition["mapping"]["minimum_coverage"]),
            np.float64(definition["initial_nav"]),
        )
        (
            state_returns,
            state_contributions,
            coverage_ratios,
            coverage_statuses,
            missing_by_state,
            _,
            _,
            state_summary_values,
            _,
            aggregate_missing,
            minimum_coverage,
        ) = projection
        for state_position, status_value in enumerate(coverage_statuses):
            status = int(status_value)
            if status == 0:
                continue
            field = f"scenario.transition.states.{state_position}.asset_returns"
            missing = _asset_names_from_mask(asset_ids, missing_by_state[state_position])
            if status in {1, 2, 3}:
                _raise_coverage_status(
                    definition,
                    portfolio,
                    status,
                    float(coverage_ratios[state_position]),
                    missing,
                    field,
                )
            if status in {4, 5}:
                raise ValidationError("RETURN_BELOW_NEGATIVE_ONE", "简单收益率不能小于或等于 -100%。", field)
            raise ValidationError("INVALID_SIMULATED_NAV", "状态收益产生了无效净值。", field)
        contribution_values = None
        contribution_method = "state_frequency_weighted_period_contributions"
        if grouped_samples is not None and state_offsets is not None and return_uniform_draws is not None:
            (
                period_matrix,
                contribution_values,
                distribution_status,
                bad_step,
                _,
                minimum_return,
            ) = empirical_regime_projection_kernel(
                np.ascontiguousarray(state_paths),
                return_uniform_draws,
                grouped_samples,
                state_offsets,
                _weights_array(definition, portfolio),
            )
            if int(distribution_status) != 0:
                raise ValidationError(
                    "HISTORICAL_STATE_DISTRIBUTION_EXECUTION_FAILED",
                    "历史状态条件收益抽样产生了无效组合收益。",
                    "scenario.transition.asset_return_source",
                    diagnostics=[
                        {
                            "status": int(distribution_status),
                            "step": int(bad_step) + 1,
                            "minimum_return": _json_number(minimum_return),
                        }
                    ],
                )
            contribution_method = "historical_state_empirical_bootstrap_mean_sum_of_period_contributions"
        else:
            period_matrix = state_path_returns_kernel(
                np.ascontiguousarray(state_paths),
                np.ascontiguousarray(state_returns),
            )
        nav_paths, terminal_returns, max_drawdowns, nav_status, bad_step, _, minimum_return = returns_to_nav_kernel(
            np.ascontiguousarray(period_matrix),
            np.float64(definition["initial_nav"]),
        )
        if int(nav_status) in {1, 2}:
            raise ValidationError(
                "INVALID_SIMULATED_RETURN",
                "状态转移产生了无法形成有效净值的组合收益。",
                "scenario.transition.states",
                diagnostics=[{"step": int(bad_step) + 1, "minimum_return": _json_number(minimum_return)}],
            )
        if int(nav_status) == 3:
            raise ValidationError("INVALID_SIMULATED_NAV", "状态转移模拟净值出现非有限值或非正值。", "scenario")
        distribution_result = _fan_summary(
            nav_paths,
            terminal_returns,
            max_drawdowns,
            path_count=path_count,
            seed=seed,
            target_return=target_return,
        )
        distribution_result["state_probabilities"] = probabilities_by_step
        distribution_result["sample_state_paths"] = [
            [state_ids[int(state)] for state in row]
            for row in state_paths[:12]
        ]
        distribution_result["asset_return_sampling"] = (
            "historical_state_empirical_bootstrap"
            if grouped_samples is not None
            else "fixed_state_return"
        )
        distribution_result["asset_return_seed"] = distribution_seed
        terminal = distribution_result["terminal"]
        summary = {
            "initial_nav": float(definition["initial_nav"]),
            "terminal_nav": terminal["p50"],
            "terminal_return": terminal["return_p50"],
            "max_drawdown": terminal["p95_max_drawdown"],
            "worst_step_return": float(state_summary_values[3]),
            "var_95": terminal["var_95"],
            "es_95": terminal["es_95"],
            "loss_probability": terminal["loss_probability"],
            "target_hit_probability": terminal["target_hit_probability"],
        }
        limits, breach_count, _ = _limit_breaches(definition, summary)
        summary["breach_count"] = breach_count
        by_state = {
            state["id"]: _coverage_payload(
                definition,
                asset_ids,
                float(coverage_ratios[state_position]),
                missing_by_state[state_position],
            )
            for state_position, state in enumerate(states)
        }
        coverage = _coverage_payload(definition, asset_ids, float(minimum_coverage), aggregate_missing)
        coverage["by_state"] = by_state
        if contribution_values is None:
            contribution_values = state_weighted_contributions_kernel(
                np.ascontiguousarray(state_paths),
                np.ascontiguousarray(state_contributions),
            )
        mean_contributions = {
            asset_id: None if not math.isfinite(float(contribution_values[index])) else float(contribution_values[index])
            for index, asset_id in enumerate(asset_ids)
        }
        results.append(
            {
                "portfolio_id": portfolio["id"],
                "name": portfolio["label"],
                "coverage": coverage,
                "summary": summary,
                "contributions": {"by_asset": mean_contributions, "method": contribution_method},
                "distribution": distribution_result,
                "limits": limits,
            }
        )
    diagnostics = {
        "state_ids": state_ids,
        "transition_matrix": matrix.tolist(),
        "initial_state": initial_state,
        "source": source,
        "probability_semantics": "seeded_markov_state_paths",
        "initial_state_semantics": "initial_state_applies_to_step_1",
        "random_generation": "numba_njit_seeded_sampling",
        "asset_return_sampling": (
            "historical_state_empirical_bootstrap"
            if grouped_samples is not None
            else "fixed_state_return"
        ),
        "asset_return_seed": distribution_seed,
    }
    snapshot = {"source": source, "state_ids": state_ids, "transition_matrix": matrix.tolist()}
    return results, diagnostics, snapshot


def _reverse_results(definition: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    config = definition["scenario"].get("reverse_stress") or definition["scenario"]
    if not isinstance(config, dict):
        raise ValidationError("INVALID_REVERSE_STRESS", "必须配置 reverse_stress。", "scenario.reverse_stress")
    target_metric = str(config.get("target_metric") or "loss")
    if target_metric not in {"loss", "drawdown"}:
        raise ValidationError("INVALID_REVERSE_TARGET", "target_metric 必须是 loss 或 drawdown。", "scenario.reverse_stress.target_metric")
    threshold = _finite(config.get("threshold"), "scenario.reverse_stress.threshold")
    if not 0.0 < threshold < 1.0:
        raise ValidationError("INVALID_REVERSE_THRESHOLD", "反向压力阈值必须在 0 至 1 之间。", "scenario.reverse_stress.threshold")
    factor_ids = _factor_ids(definition)
    bounds = config.get("bounds")
    if not isinstance(bounds, dict) or set(bounds) != set(factor_ids):
        raise ValidationError("REVERSE_BOUND_DIMENSION_MISMATCH", "bounds 必须逐项覆盖因子字典。", "scenario.reverse_stress.bounds")
    ranges: dict[str, tuple[float, float]] = {}
    for factor_id in factor_ids:
        pair = bounds[factor_id]
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValidationError("INVALID_REVERSE_BOUND", "每个因子边界必须是 [下限, 上限]。", f"scenario.reverse_stress.bounds.{factor_id}")
        lower = _finite(pair[0], f"scenario.reverse_stress.bounds.{factor_id}.0")
        upper = _finite(pair[1], f"scenario.reverse_stress.bounds.{factor_id}.1")
        if lower >= upper or lower > 0 or upper < 0:
            raise ValidationError("INVALID_REVERSE_BOUND", "因子边界必须满足下限 < 0 < 上限。", f"scenario.reverse_stress.bounds.{factor_id}")
        ranges[factor_id] = (lower, upper)
    mapping = _validate_factor_mapping(definition)
    intercepts = _asset_intercepts(definition)
    asset_ids = _asset_ids(definition)
    beta, intercept_array, available = _mapping_arrays(definition, mapping, intercepts)
    bounds_array = _float64_2d([ranges[factor_id] for factor_id in factor_ids])
    available_assets = {
        asset_id for asset_id, row in mapping.items() if all(factor_id in row for factor_id in factor_ids)
    }
    results: list[dict[str, Any]] = []
    for portfolio in definition["portfolios"]:
        effective_weights, coverage_ratio, coverage_status, missing_mask = coverage_weights_kernel(
            _weights_array(definition, portfolio),
            available,
            _policy_code(definition),
            np.float64(definition["mapping"]["minimum_coverage"]),
        )
        missing_assets = _asset_names_from_mask(asset_ids, missing_mask)
        _raise_coverage_status(
            definition,
            portfolio,
            int(coverage_status),
            float(coverage_ratio),
            missing_assets,
            "mapping.factor_to_asset",
        )
        coverage = _coverage_payload(definition, asset_ids, float(coverage_ratio), missing_mask)
        (
            reverse_status,
            candidate_count,
            shock_matrix,
            asset_impacts,
            contribution_matrix,
            portfolio_returns,
            max_drawdowns,
            severities,
            meets_target,
            kind_codes,
            primary_indices,
            secondary_indices,
            order,
            worst_index,
            feasible_count,
            summary_values,
        ) = reverse_stress_kernel(
            beta,
            intercept_array,
            bounds_array,
            np.ascontiguousarray(effective_weights),
            available,
            np.float64(threshold),
            np.float64(definition["initial_nav"]),
        )
        if int(reverse_status) == 1:
            raise ValidationError(
                "REVERSE_STRESS_NO_SENSITIVITY",
                f"组合“{portfolio['label']}”对所选因子没有有效敏感度。",
                field="mapping.factor_to_asset",
            )
        if int(reverse_status) == 2:
            raise ValidationError(
                "RETURN_BELOW_NEGATIVE_ONE",
                "反向压力候选使简单收益率小于或等于 -100%，请收窄因子边界。",
                "scenario.reverse_stress.bounds",
            )
        candidates: list[dict[str, Any]] = []
        for candidate_position in range(int(candidate_count)):
            candidate_index = int(order[candidate_position])
            kind_code = int(kind_codes[candidate_index])
            primary_index = int(primary_indices[candidate_index])
            secondary_index = int(secondary_indices[candidate_index])
            if kind_code == 1:
                kind = "single_factor"
                label = f"仅冲击 {factor_ids[primary_index]}"
            elif kind_code == 2:
                kind = "multi_factor"
                label = "全因子最小二范数近似"
            else:
                kind = "factor_pair"
                label = f"{factor_ids[primary_index]} + {factor_ids[secondary_index]}"
            candidates.append(
                {
                    "kind": kind,
                    "label": label,
                    "factor_shocks": {
                        factor_id: float(shock_matrix[candidate_index, factor_index])
                        for factor_index, factor_id in enumerate(factor_ids)
                    },
                    "asset_impacts": {
                        asset_id: (
                            None
                            if not math.isfinite(float(asset_impacts[candidate_index, asset_index]))
                            else float(asset_impacts[candidate_index, asset_index])
                        )
                        for asset_index, asset_id in enumerate(asset_ids)
                    },
                    "contributions": {
                        asset_id: (
                            None
                            if not math.isfinite(float(contribution_matrix[candidate_index, asset_index]))
                            else float(contribution_matrix[candidate_index, asset_index])
                        )
                        for asset_index, asset_id in enumerate(asset_ids)
                    },
                    "portfolio_return": float(portfolio_returns[candidate_index]),
                    "max_drawdown": float(max_drawdowns[candidate_index]),
                    "severity": float(severities[candidate_index]),
                    "meets_target": bool(meets_target[candidate_index]),
                    "coverage": copy.deepcopy(coverage),
                }
            )
        if int(candidate_count) == 0 or int(worst_index) < 0:
            raise ValidationError(
                "REVERSE_STRESS_NO_SENSITIVITY",
                f"组合“{portfolio['label']}”对所选因子没有有效敏感度。",
                field="mapping.factor_to_asset",
            )
        summary = {
            "initial_nav": float(definition["initial_nav"]),
            "terminal_nav": float(summary_values[0]),
            "terminal_return": float(summary_values[1]),
            "max_drawdown": float(summary_values[2]),
            "worst_step_return": float(summary_values[3]),
        }
        limits, breach_count, _ = _limit_breaches(definition, summary)
        summary["breach_count"] = breach_count
        worst_contributions = {
            asset_id: (
                None
                if not math.isfinite(float(contribution_matrix[int(worst_index), asset_index]))
                else float(contribution_matrix[int(worst_index), asset_index])
            )
            for asset_index, asset_id in enumerate(asset_ids)
        }
        results.append(
            {
                "portfolio_id": portfolio["id"],
                "name": portfolio["label"],
                "coverage": coverage,
                "summary": summary,
                "contributions": {"by_asset": worst_contributions, "method": "worst_candidate_one_step"},
                "reverse_stress": {
                    "target_metric": target_metric,
                    "threshold": threshold,
                    "approximation": "linear_minimum_norm_with_box_bounds",
                    "candidates": candidates,
                    "feasible_candidate_count": int(feasible_count),
                },
                "limits": limits,
            }
        )
    diagnostics = {
        "orientation": "asset_by_factor",
        "factor_ids": factor_ids,
        "mapped_assets": sorted(available_assets),
        "unmapped_assets": sorted(set(_asset_ids(definition)) - available_assets),
        "solver": "closed_form_linear_minimum_norm_with_box_clipping",
        "exactness": "approximate_when_bounds_bind",
        "response_space": definition["mapping"]["response_space"],
    }
    return results, diagnostics


def execute(definition: dict[str, Any]) -> dict[str, Any]:
    """Execute one normalized definition and return JSON-safe analytical content."""

    # Explicit signatures are compiled before this call. This assertion fails
    # closed if a production dispatcher is missing or has specialized at runtime.
    assert_scenario_stress_numba_ready()
    method = definition["method"]
    probabilistic = method in PROBABILISTIC_METHODS
    diagnostics: list[dict[str, Any]] = []
    data_snapshot: dict[str, Any] = {"source": "definition_inline", "method": method}
    if method == "historical_replay":
        steps, replay_snapshot = _historical_steps(definition)
        results = _deterministic_results(definition, steps)
        mapping_diagnostics = {
            "orientation": "direct_asset_returns",
            "asset_ids": _asset_ids(definition),
            "missing_values": replay_snapshot["missing_values"],
        }
        data_snapshot = replay_snapshot
    elif method == "factor_path":
        steps, mapping_diagnostics = _map_factor_steps(definition)
        results = _deterministic_results(definition, steps)
        data_snapshot = {"source": "inline_factor_path", "steps": len(steps), "factor_ids": _factor_ids(definition)}
    elif method == "monte_carlo":
        results, mapping_diagnostics = _monte_carlo_results(definition)
        data_snapshot = {
            "source": "inline_distribution_parameters",
            "distribution": definition["scenario"].get("distribution", "normal"),
            "seed": int(definition["scenario"].get("seed", 0)),
        }
    elif method == "regime_conditioned":
        results, mapping_diagnostics, data_snapshot = _regime_results(definition)
    elif method == "reverse_stress":
        results, mapping_diagnostics = _reverse_results(definition)
        data_snapshot = {"source": "inline_reverse_stress_constraints", "factor_ids": _factor_ids(definition)}
    else:  # normalize_definition makes this unreachable.
        raise ValidationError("UNSUPPORTED_SCENARIO_METHOD", "不支持的情景模拟方法。", "method")

    degraded = [result["portfolio_id"] for result in results if result["coverage"]["status"] == "degraded"]
    if degraded:
        diagnostics.append(
            {
                "code": "COVERAGE_DEGRADED",
                "level": "warning",
                "message": "部分组合仅按覆盖资产重归一估算；缺失资产贡献保持为空。",
                "portfolio_ids": degraded,
            }
        )
    diagnostics.append(
        {
            "code": "PROBABILITY_SEMANTICS",
            "level": "info",
            "message": (
                "概率来自带固定种子的随机路径频率。"
                if probabilistic
                else "这是确定性情景，结果不包含概率、VaR 或 ES。"
            ),
        }
    )
    return {
        "method": method,
        "probabilistic": probabilistic,
        "results": results,
        "mapping_diagnostics": mapping_diagnostics,
        "data_snapshot": data_snapshot,
        "diagnostics": diagnostics,
        "compute_audit": scenario_stress_numba_status(),
    }

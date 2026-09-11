"""Stable contracts and validation for scenario simulation and stress testing."""

from __future__ import annotations

import copy
import math
import re
from typing import Any

import numpy as np

from custom_indicators.errors import ValidationError

from .numba_kernels import portfolio_weight_summary_kernel


SCHEMA_VERSION = "1.0"
METHOD_ALIASES = {
    "deterministic_path": "factor_path",
    "regime_transition": "regime_conditioned",
}
METHODS = (
    "historical_replay",
    "factor_path",
    "monte_carlo",
    "regime_conditioned",
    "reverse_stress",
)
APPLICATION_TARGETS = (
    "research_display",
    "product_research",
    "portfolio_backtest",
    "taa",
    "risk_monitoring",
)
DETERMINISTIC_METHODS = {"historical_replay", "factor_path", "reverse_stress"}
PROBABILISTIC_METHODS = {"monte_carlo", "regime_conditioned"}
IDENTIFIER_PATTERN = re.compile(r"^[\w.:-]{1,80}$", re.UNICODE)
WEIGHT_NET_TARGET = 1.0
WEIGHT_TOLERANCE = 1e-8
MAXIMUM_ABSOLUTE_WEIGHT = 2.0
MAXIMUM_GROSS_EXPOSURE = 3.0


def summarize_portfolio_weights(weights: dict[str, float]) -> dict[str, Any]:
    """Return one authoritative long/short weight summary from fixed NJIT."""

    values = np.ascontiguousarray(list(weights.values()), dtype=np.float64)
    net, gross, largest, status = portfolio_weight_summary_kernel(
        values,
        np.float64(WEIGHT_NET_TARGET),
        np.float64(WEIGHT_TOLERANCE),
        np.float64(MAXIMUM_ABSOLUTE_WEIGHT),
        np.float64(MAXIMUM_GROSS_EXPOSURE),
    )
    status_code = int(status)
    finite = status_code & 1 == 0
    nonempty = status_code & 16 == 0
    return {
        "weight_count": int(values.size),
        "total_weight": float(net),
        "net_exposure": float(net),
        "gross_exposure": float(gross),
        "largest_absolute_weight": float(largest),
        "finite": finite,
        "single_weight_valid": finite and nonempty and status_code & 2 == 0,
        "sums_to_one": finite and nonempty and status_code & 4 == 0,
        "gross_limit_valid": finite and nonempty and status_code & 8 == 0,
        "within_tolerance": status_code == 0,
        "status_code": status_code,
        "limits": {
            "expected_net_exposure": WEIGHT_NET_TARGET,
            "absolute_tolerance": WEIGHT_TOLERANCE,
            "maximum_absolute_weight": MAXIMUM_ABSOLUTE_WEIGHT,
            "maximum_gross_exposure": MAXIMUM_GROSS_EXPOSURE,
        },
    }


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ValidationError("INVALID_NUMBER", f"{field} 必须是有限数值。", field)
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError("INVALID_NUMBER", f"{field} 必须是有限数值。", field) from exc
    if not math.isfinite(result):
        raise ValidationError("INVALID_NUMBER", f"{field} 必须是有限数值。", field)
    return result


def _identifier(value: Any, field: str) -> str:
    result = str(value or "").strip()
    if not IDENTIFIER_PATTERN.fullmatch(result):
        raise ValidationError(
            "INVALID_IDENTIFIER",
            f"{field} 只能包含字母、数字、点、冒号、下划线或连字符，长度不超过 80。",
            field,
        )
    return result


def _entities(
    values: Any,
    *,
    field: str,
    minimum: int,
    maximum: int,
) -> list[dict[str, Any]]:
    if not isinstance(values, list) or not minimum <= len(values) <= maximum:
        raise ValidationError(
            f"INVALID_{field.upper()}",
            f"{field} 必须包含 {minimum} 至 {maximum} 项。",
            field,
        )
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(values):
        if not isinstance(item, dict):
            raise ValidationError(f"INVALID_{field.upper()}", f"{field} 每一项必须是对象。", f"{field}.{index}")
        item_id = _identifier(item.get("id"), f"{field}.{index}.id")
        label = str(item.get("label") or item.get("name") or item_id).strip()
        if not label or len(label) > 100 or item_id in seen:
            raise ValidationError(
                f"INVALID_{field.upper()}",
                f"{field} 的 id 必须唯一，名称长度不超过 100。",
                f"{field}.{index}",
            )
        seen.add(item_id)
        result.append({**copy.deepcopy(item), "id": item_id, "label": label})
    return result


def _normalize_portfolios(values: Any, asset_ids: list[str]) -> list[dict[str, Any]]:
    portfolios = _entities(values, field="portfolios", minimum=1, maximum=32)
    asset_set = set(asset_ids)
    normalized: list[dict[str, Any]] = []
    for index, portfolio in enumerate(portfolios):
        weights = portfolio.get("weights")
        if not isinstance(weights, dict):
            raise ValidationError("INVALID_WEIGHTS", "每个组合都必须配置 weights。", f"portfolios.{index}.weights")
        supplied = set(weights)
        if supplied != asset_set:
            missing = sorted(asset_set - supplied)
            unknown = sorted(supplied - asset_set)
            raise ValidationError(
                "WEIGHT_DIMENSION_MISMATCH",
                "组合权重必须逐项覆盖资产字典；缺失权重不会按 0 处理。",
                f"portfolios.{index}.weights",
                diagnostics=[{"missing_assets": missing, "unknown_assets": unknown}],
            )
        clean_weights = {
            asset_id: _number(weights[asset_id], f"portfolios.{index}.weights.{asset_id}")
            for asset_id in asset_ids
        }
        summary = summarize_portfolio_weights(clean_weights)
        if not summary["single_weight_valid"]:
            raise ValidationError("WEIGHT_OUT_OF_RANGE", "单项权重必须在 -2 至 2 之间。", f"portfolios.{index}.weights")
        if not summary["sums_to_one"]:
            raise ValidationError(
                "WEIGHTS_DO_NOT_SUM_TO_ONE",
                "组合权重之和必须等于 1（小数口径）。",
                f"portfolios.{index}.weights",
                diagnostics=[{"weight_sum": summary["total_weight"]}],
            )
        if not summary["gross_limit_valid"]:
            raise ValidationError("GROSS_EXPOSURE_EXCEEDED", "组合总敞口不能超过 3。", f"portfolios.{index}.weights")
        normalized.append({**portfolio, "weights": clean_weights})
    return normalized


def _normalize_mapping(value: Any, method: str) -> dict[str, Any]:
    mapping = copy.deepcopy(value) if isinstance(value, dict) else {}
    policy = str(mapping.get("missing_policy") or "block").strip().lower()
    if policy not in {"block", "degrade"}:
        raise ValidationError("INVALID_MISSING_POLICY", "missing_policy 必须是 block 或 degrade。", "mapping.missing_policy")
    minimum_coverage = _number(mapping.get("minimum_coverage", 0.8), "mapping.minimum_coverage")
    if not 0.5 <= minimum_coverage <= 1.0:
        raise ValidationError("INVALID_MINIMUM_COVERAGE", "minimum_coverage 必须在 0.5 至 1 之间。", "mapping.minimum_coverage")
    factor_to_asset = mapping.get("factor_to_asset", {})
    if method in {"factor_path", "monte_carlo", "reverse_stress"} and not isinstance(factor_to_asset, dict):
        raise ValidationError("INVALID_FACTOR_MAPPING", "必须配置 factor_to_asset 映射。", "mapping.factor_to_asset")
    if factor_to_asset and not isinstance(factor_to_asset, dict):
        raise ValidationError("INVALID_FACTOR_MAPPING", "factor_to_asset 必须是对象。", "mapping.factor_to_asset")
    expected_response_space = {
        "factor_path": "simple_return",
        "reverse_stress": "simple_return",
        "monte_carlo": "log_return",
    }.get(method, "direct_simple_return")
    response_space = mapping.get("response_space")
    if method in {"factor_path", "monte_carlo", "reverse_stress"} and response_space is None:
        raise ValidationError(
            "MAPPING_RESPONSE_SPACE_REQUIRED",
            "必须明确 mapping.response_space，避免在线性收益与对数收益之间无提示复用系数。",
            "mapping.response_space",
            diagnostics=[{"expected": expected_response_space, "method": method}],
        )
    response_space = str(response_space or expected_response_space)
    if response_space != expected_response_space:
        raise ValidationError(
            "MAPPING_RESPONSE_SPACE_MISMATCH",
            f"{method} 要求 mapping.response_space={expected_response_space}。",
            "mapping.response_space",
            diagnostics=[{"expected": expected_response_space, "actual": response_space}],
        )
    return {
        **mapping,
        "factor_to_asset": copy.deepcopy(factor_to_asset),
        "missing_policy": policy,
        "minimum_coverage": minimum_coverage,
        "response_space": response_space,
    }


def _normalize_limits(values: Any) -> list[dict[str, Any]]:
    if values is None:
        return []
    if not isinstance(values, list) or len(values) > 32:
        raise ValidationError("INVALID_LIMITS", "limits 必须是最多 32 项的数组。", "limits")
    allowed_metrics = {
        "terminal_return",
        "max_drawdown",
        "worst_step_return",
        "var_95",
        "es_95",
        "loss_probability",
        "target_hit_probability",
    }
    allowed_operators = {"gt", "gte", "lt", "lte"}
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(values):
        if not isinstance(item, dict):
            raise ValidationError("INVALID_LIMIT", "每条限额必须是对象。", f"limits.{index}")
        limit_id = _identifier(item.get("id") or f"limit-{index + 1}", f"limits.{index}.id")
        metric = str(item.get("metric") or "").strip()
        operator = str(item.get("operator") or "").strip()
        if limit_id in seen or metric not in allowed_metrics or operator not in allowed_operators:
            raise ValidationError(
                "INVALID_LIMIT",
                "限额 id 必须唯一，metric 与 operator 必须使用受支持值。",
                f"limits.{index}",
            )
        seen.add(limit_id)
        result.append(
            {
                **copy.deepcopy(item),
                "id": limit_id,
                "label": str(item.get("label") or limit_id).strip()[:100],
                "metric": metric,
                "operator": operator,
                "threshold": _number(item.get("threshold"), f"limits.{index}.threshold"),
            }
        )
    return result


def _normalize_historical_asset_return_source(
    scenario: dict[str, Any],
    method: str,
    asset_ids: list[str],
) -> dict[str, Any]:
    normalized = copy.deepcopy(scenario)
    transition = normalized.get("transition")
    if method != "regime_conditioned" or not isinstance(transition, dict):
        return normalized
    source = transition.get("asset_return_source")
    if source is None:
        return normalized
    if not isinstance(source, dict):
        raise ValidationError(
            "INVALID_HISTORICAL_ASSET_RETURN_SOURCE",
            "asset_return_source 必须是显式配置对象。",
            "scenario.transition.asset_return_source",
        )
    allowed_keys = {
        "kind",
        "run_content_hash",
        "evaluation_artifact_checksum",
        "asset_target_map",
        "sampling",
        "minimum_observations_per_state",
        "inline_policy",
    }
    unknown = sorted(set(source) - allowed_keys)
    if unknown:
        raise ValidationError(
            "UNKNOWN_HISTORICAL_ASSET_RETURN_SOURCE_FIELD",
            "asset_return_source 包含不支持的字段。",
            "scenario.transition.asset_return_source",
            diagnostics=[{"unknown_fields": unknown}],
        )
    if source.get("kind") != "historical_evaluation_targets":
        raise ValidationError(
            "INVALID_HISTORICAL_ASSET_RETURN_SOURCE",
            "asset_return_source.kind 必须是 historical_evaluation_targets。",
            "scenario.transition.asset_return_source.kind",
        )
    run_hash = str(source.get("run_content_hash") or "")
    if len(run_hash) != 64 or any(character not in "0123456789abcdef" for character in run_hash):
        raise ValidationError(
            "INVALID_CONTENT_LOCK",
            "run_content_hash 必须是 64 位 sha256 内容哈希。",
            "scenario.transition.asset_return_source.run_content_hash",
        )
    artifact_checksum = str(source.get("evaluation_artifact_checksum") or "")
    if (
        not artifact_checksum.startswith("sha256:")
        or len(artifact_checksum) != 71
        or any(character not in "0123456789abcdef" for character in artifact_checksum[7:])
    ):
        raise ValidationError(
            "INVALID_CONTENT_LOCK",
            "evaluation_artifact_checksum 必须是 sha256 内容锁。",
            "scenario.transition.asset_return_source.evaluation_artifact_checksum",
        )
    sampling = str(source.get("sampling") or "empirical_bootstrap")
    if sampling != "empirical_bootstrap":
        raise ValidationError(
            "UNSUPPORTED_HISTORICAL_DISTRIBUTION_SAMPLING",
            "当前仅支持 empirical_bootstrap 状态条件联合分布抽样。",
            "scenario.transition.asset_return_source.sampling",
        )
    inline_policy = str(source.get("inline_policy") or "forbid")
    if inline_policy not in {"forbid", "override"}:
        raise ValidationError(
            "INVALID_HISTORICAL_INLINE_POLICY",
            "inline_policy 必须是 forbid 或 override。",
            "scenario.transition.asset_return_source.inline_policy",
        )
    minimum = source.get("minimum_observations_per_state", 5)
    if isinstance(minimum, bool):
        raise ValidationError(
            "INVALID_HISTORICAL_MINIMUM_OBSERVATIONS",
            "每状态最少观测数必须是 2 至 20000 的整数。",
            "scenario.transition.asset_return_source.minimum_observations_per_state",
        )
    raw_minimum = minimum
    try:
        minimum = int(raw_minimum)
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            "INVALID_HISTORICAL_MINIMUM_OBSERVATIONS",
            "每状态最少观测数必须是 2 至 20000 的整数。",
            "scenario.transition.asset_return_source.minimum_observations_per_state",
        ) from exc
    if (
        isinstance(raw_minimum, (float, np.floating))
        and float(raw_minimum) != float(minimum)
    ) or minimum < 2 or minimum > 20_000:
        raise ValidationError(
            "INVALID_HISTORICAL_MINIMUM_OBSERVATIONS",
            "每状态最少观测数必须是 2 至 20000 的整数。",
            "scenario.transition.asset_return_source.minimum_observations_per_state",
        )
    mapping = source.get("asset_target_map")
    if not isinstance(mapping, dict) or not mapping:
        raise ValidationError(
            "INVALID_HISTORICAL_ASSET_TARGET_MAP",
            "asset_target_map 必须至少映射一项资产到评价目标。",
            "scenario.transition.asset_return_source.asset_target_map",
        )
    unknown_assets = sorted(set(mapping) - set(asset_ids))
    if unknown_assets:
        raise ValidationError(
            "HISTORICAL_ASSET_TARGET_DIMENSION_MISMATCH",
            "asset_target_map 包含资产字典之外的字段。",
            "scenario.transition.asset_return_source.asset_target_map",
            diagnostics=[{"unknown_assets": unknown_assets}],
        )
    clean_mapping: dict[str, dict[str, str]] = {}
    for asset_id, item in mapping.items():
        field = f"scenario.transition.asset_return_source.asset_target_map.{asset_id}"
        if not isinstance(item, dict) or set(item) - {"target_id", "return_transform"}:
            raise ValidationError(
                "INVALID_HISTORICAL_ASSET_TARGET",
                "每个资产映射只能包含 target_id 与 return_transform。",
                field,
            )
        target_id = _identifier(item.get("target_id"), f"{field}.target_id")
        transform = str(item.get("return_transform") or "")
        if transform not in {"simple_return", "forward_value"}:
            raise ValidationError(
                "INVALID_HISTORICAL_ASSET_TARGET",
                "return_transform 必须是 simple_return 或 forward_value。",
                f"{field}.return_transform",
            )
        clean_mapping[asset_id] = {"target_id": target_id, "return_transform": transform}
    transition["asset_return_source"] = {
        "kind": "historical_evaluation_targets",
        "run_content_hash": run_hash,
        "evaluation_artifact_checksum": artifact_checksum,
        "asset_target_map": clean_mapping,
        "sampling": sampling,
        "minimum_observations_per_state": minimum,
        "inline_policy": inline_policy,
    }
    return normalized


def normalize_definition(fields: dict[str, Any]) -> dict[str, Any]:
    """Validate common definition fields without fabricating missing observations."""

    if not isinstance(fields, dict):
        raise ValidationError("INVALID_DEFINITION", "情景定义必须是对象。", "definition")
    definition = copy.deepcopy(fields)
    name = str(definition.get("name") or "").strip()
    if not name or len(name) > 100:
        raise ValidationError("INVALID_SCENARIO_NAME", "情景名称长度应为 1 至 100 个字符。", "name")
    description = str(definition.get("description") or "").strip()
    if len(description) > 1000:
        raise ValidationError("INVALID_SCENARIO_DESCRIPTION", "情景说明不能超过 1000 个字符。", "description")
    raw_method = str(definition.get("method") or definition.get("type") or "").strip().lower()
    method = METHOD_ALIASES.get(raw_method, raw_method)
    if method not in METHODS:
        raise ValidationError("UNSUPPORTED_SCENARIO_METHOD", "不支持的情景模拟方法。", "method")
    try:
        horizon = int(definition.get("horizon", 1))
    except (TypeError, ValueError) as exc:
        raise ValidationError("INVALID_HORIZON", "horizon 必须是整数。", "horizon") from exc
    if horizon < 1 or horizon > 1200:
        raise ValidationError("INVALID_HORIZON", "horizon 必须在 1 至 1200 期之间。", "horizon")

    assets = _entities(definition.get("assets"), field="assets", minimum=1, maximum=64)
    asset_ids = [item["id"] for item in assets]
    portfolios = _normalize_portfolios(definition.get("portfolios"), asset_ids)
    factor_minimum = 1 if method in {"factor_path", "monte_carlo", "reverse_stress"} else 0
    factors = _entities(definition.get("factors", []), field="factors", minimum=factor_minimum, maximum=32)
    mapping = _normalize_mapping(definition.get("mapping"), method)
    scenario = definition.get("scenario")
    if not isinstance(scenario, dict):
        raise ValidationError("INVALID_SCENARIO_CONFIG", "scenario 必须是对象。", "scenario")
    scenario = _normalize_historical_asset_return_source(scenario, method, asset_ids)
    limits = _normalize_limits(definition.get("limits"))
    if method in DETERMINISTIC_METHODS:
        probabilistic_metrics = {"var_95", "es_95", "loss_probability", "target_hit_probability"}
        invalid = [item["metric"] for item in limits if item["metric"] in probabilistic_metrics]
        if invalid:
            raise ValidationError(
                "PSEUDO_PROBABILITY_LIMIT_BLOCKED",
                "确定性情景不能配置概率、VaR 或 ES 限额。",
                "limits",
                diagnostics=[{"invalid_metrics": invalid}],
            )
    elif method in PROBABILISTIC_METHODS:
        invalid = [item["metric"] for item in limits if item["metric"] == "worst_step_return"]
        if invalid:
            raise ValidationError(
                "UNAVAILABLE_LIMIT_METRIC",
                "概率路径暂不提供单一的 worst_step_return 口径，请使用 VaR、ES 或最大回撤限额。",
                "limits",
            )
    if method == "reverse_stress" and horizon != 1:
        raise ValidationError(
            "REVERSE_STRESS_HORIZON_MUST_BE_ONE",
            "当前反向压力求解器是单期线性近似，horizon 必须为 1。",
            "horizon",
        )
    usage_intent = str(definition.get("usage_intent") or "research_display")
    if usage_intent not in APPLICATION_TARGETS:
        raise ValidationError("INVALID_USAGE", "不支持的应用目标。", "usage_intent")
    initial_nav = _number(definition.get("initial_nav", 1.0), "initial_nav")
    if initial_nav <= 0:
        raise ValidationError("INVALID_INITIAL_NAV", "initial_nav 必须大于 0。", "initial_nav")
    return {
        **definition,
        "name": name,
        "description": description,
        "method": method,
        "horizon": horizon,
        "initial_nav": initial_nav,
        "factors": factors,
        "assets": assets,
        "portfolios": portfolios,
        "mapping": mapping,
        "scenario": scenario,
        "limits": limits,
        "usage_intent": usage_intent,
        "schema_version": SCHEMA_VERSION,
    }


def meta_contract() -> dict[str, Any]:
    """Return UI discovery metadata and executable example shapes."""

    assets = [
        {"id": "cn_equity", "label": "A 股权益"},
        {"id": "duration_bond", "label": "中长久期债券"},
        {"id": "gold", "label": "黄金"},
    ]
    factors = [
        {"id": "growth", "label": "增长动能", "unit": "sigma"},
        {"id": "inflation", "label": "通胀动能", "unit": "sigma"},
        {"id": "rate", "label": "利率变动", "unit": "bp"},
    ]
    common = {
        "name": "国内滞胀压力",
        "description": "增长下行、通胀与利率上行的多资产传导。",
        "horizon": 6,
        "initial_nav": 1.0,
        "factors": factors,
        "assets": assets,
        "portfolios": [
            {
                "id": "balanced",
                "name": "平衡组合",
                "weights": {"cn_equity": 0.4, "duration_bond": 0.45, "gold": 0.15},
            }
        ],
        "mapping": {
            "missing_policy": "block",
            "minimum_coverage": 1.0,
            "response_space": "simple_return",
            "factor_to_asset": {
                "cn_equity": {"growth": 0.035, "inflation": -0.01, "rate": -0.00025},
                "duration_bond": {"growth": -0.01, "inflation": -0.02, "rate": -0.00055},
                "gold": {"growth": -0.004, "inflation": 0.03, "rate": -0.0001},
            },
        },
        "limits": [
            {"id": "loss-10", "label": "累计亏损超过 10%", "metric": "terminal_return", "operator": "lt", "threshold": -0.1},
            {"id": "drawdown-12", "label": "最大回撤超过 12%", "metric": "max_drawdown", "operator": "gt", "threshold": 0.12},
        ],
        "usage_intent": "research_display",
    }
    templates = [
        {
            "id": "historical-liquidity-replay",
            "name": "流动性冲击历史重演",
            "phase": "P0",
            "definition": {
                **copy.deepcopy(common),
                "name": "历史流动性冲击重演",
                "method": "historical_replay",
                "horizon": 4,
                "factors": [],
                "mapping": {
                    "missing_policy": "block",
                    "minimum_coverage": 1.0,
                    "factor_to_asset": {},
                    "response_space": "direct_simple_return",
                },
                "scenario": {
                    "historical_returns": [
                        {"date": "2020-03-16", "returns": {"cn_equity": -0.035, "duration_bond": 0.004, "gold": -0.012}},
                        {"date": "2020-03-17", "returns": {"cn_equity": -0.018, "duration_bond": 0.002, "gold": 0.006}},
                        {"date": "2020-03-18", "returns": {"cn_equity": -0.012, "duration_bond": -0.001, "gold": 0.009}},
                        {"date": "2020-03-19", "returns": {"cn_equity": 0.021, "duration_bond": 0.001, "gold": 0.014}},
                    ]
                },
            },
        },
        {
            "id": "stagflation-factor-path",
            "name": "滞胀多因子路径",
            "phase": "P0",
            "definition": {
                **copy.deepcopy(common),
                "method": "factor_path",
                "scenario": {
                    "shocks": {"growth": -1.4, "inflation": 1.2, "rate": 55.0},
                    "path_shape": "linear",
                    "severity": 1.0,
                },
            },
        },
        {
            "id": "normal-monte-carlo",
            "name": "正态多因子蒙特卡罗",
            "phase": "P1",
            "definition": {
                **copy.deepcopy(common),
                "name": "多因子蒙特卡罗",
                "method": "monte_carlo",
                "mapping": {**copy.deepcopy(common["mapping"]), "response_space": "log_return"},
                "scenario": {
                    "distribution": "normal",
                    "factor_means": {"growth": 0.0, "inflation": 0.0, "rate": 0.0},
                    "factor_volatilities": {"growth": 0.12, "inflation": 0.1, "rate": 5.0},
                    "correlation": [[1.0, 0.2, -0.1], [0.2, 1.0, 0.1], [-0.1, 0.1, 1.0]],
                    "path_count": 2000,
                    "seed": 20260903,
                    "target_return": 0.03,
                },
            },
        },
        {
            "id": "regime-conditioned-cycle",
            "name": "经济周期状态转移",
            "phase": "P1",
            "definition": {
                **copy.deepcopy(common),
                "name": "经济周期状态条件模拟",
                "method": "regime_conditioned",
                "horizon": 12,
                "factors": [],
                "mapping": {
                    "missing_policy": "block",
                    "minimum_coverage": 1.0,
                    "factor_to_asset": {},
                    "response_space": "direct_simple_return",
                },
                "scenario": {
                    "transition": {
                        "states": [
                            {"id": "recovery", "label": "复苏", "asset_returns": {"cn_equity": 0.018, "duration_bond": -0.002, "gold": 0.003}},
                            {"id": "overheat", "label": "过热", "asset_returns": {"cn_equity": 0.006, "duration_bond": -0.009, "gold": 0.008}},
                            {"id": "recession", "label": "衰退", "asset_returns": {"cn_equity": -0.022, "duration_bond": 0.012, "gold": 0.009}},
                        ],
                        "matrix": [[0.65, 0.25, 0.1], [0.15, 0.55, 0.3], [0.35, 0.1, 0.55]],
                        "initial_state": "recovery",
                        "path_count": 2000,
                        "seed": 20260903,
                        "target_return": 0.03,
                    }
                },
            },
        },
        {
            "id": "reverse-loss-boundary",
            "name": "组合亏损边界反推",
            "phase": "P1",
            "definition": {
                **copy.deepcopy(common),
                "name": "组合亏损 10% 反向压力",
                "method": "reverse_stress",
                "horizon": 1,
                "scenario": {
                    "reverse_stress": {
                        "target_metric": "loss",
                        "threshold": 0.1,
                        "bounds": {"growth": [-5.0, 5.0], "inflation": [-5.0, 5.0], "rate": [-300.0, 300.0]},
                    }
                },
            },
        },
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "methods": [
            {"id": "historical_replay", "label": "历史重演", "phase": "P0", "probabilistic": False},
            {"id": "factor_path", "label": "单/多因子路径", "phase": "P0", "probabilistic": False},
            {"id": "monte_carlo", "label": "条件蒙特卡罗", "phase": "P1", "probabilistic": True},
            {"id": "regime_conditioned", "label": "状态转移模拟", "phase": "P1", "probabilistic": True},
            {"id": "reverse_stress", "label": "反向压力测试", "phase": "P1", "probabilistic": False},
        ],
        "types": list(METHODS),
        "templates": templates,
        "usages": [
            {"id": "research_display", "label": "研究展示"},
            {"id": "product_research", "label": "产品研究"},
            {"id": "portfolio_backtest", "label": "组合回测"},
            {"id": "taa", "label": "战术资产配置"},
            {"id": "risk_monitoring", "label": "风险监控"},
        ],
        "limits": {
            "max_assets": 64,
            "max_factors": 32,
            "max_portfolios": 32,
            "max_horizon": 1200,
            "max_historical_rows": 20000,
            "max_paths": 50000,
            "max_path_cells": 2000000,
            "max_factor_draw_cells": 8000000,
            "max_compare_runs": 8,
            "return_unit": "decimal",
            "weight_unit": "decimal",
        },
        "mapping_contract": {
            "orientation": "asset_by_factor",
            "missing_policy": ["block", "degrade"],
            "degrade_semantics": "仅对有映射资产的权重重新归一；缺失贡献保持 null，并披露覆盖率。",
            "optional_asset_intercept_default": 0.0,
            "monte_carlo_mapping_output": "asset_log_return",
            "deterministic_mapping_output": "asset_simple_return",
            "response_space_by_method": {
                "factor_path": "simple_return",
                "reverse_stress": "simple_return",
                "monte_carlo": "log_return",
                "historical_replay": "direct_simple_return",
                "regime_conditioned": "direct_simple_return",
            },
        },
        "historical_regime_distribution_contract": {
            "location": "scenario.transition.asset_return_source",
            "kind": "historical_evaluation_targets",
            "required_content_locks": ["run_content_hash", "evaluation_artifact_checksum"],
            "sampling": ["empirical_bootstrap"],
            "return_transforms": ["simple_return", "forward_value"],
            "inline_policies": ["forbid", "override"],
            "default_inline_policy": "forbid",
            "state_alignment": "state_at_period_start_to_forward_1_period_return",
            "joint_sampling": "complete_cross_asset_observation",
            "missing_value_policy": "exclude_incomplete_joint_observation_never_fill_zero",
        },
    }

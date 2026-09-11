"""Contracts and validation for historical regime research."""

from __future__ import annotations

import copy
from typing import Any

from custom_indicators.errors import ValidationError
from custom_indicators.periods import SUPPORTED_PERIODS, period_metadata

from .formula import formula_language_meta


SCHEMA_VERSION = "1.0"
RUN_MODES = ("realtime", "retrospective")
APPLICATION_TARGETS = (
    "research_display",
    "product_research",
    "formal_backtest",
    "taa",
)


DEFAULT_STATES: dict[str, list[dict[str, Any]]] = {
    "market": [
        {"id": "bull", "label": "牛市", "role": "positive", "color": "#16a34a", "order": 1},
        {"id": "sideways", "label": "震荡市", "role": "neutral", "color": "#64748b", "order": 2},
        {"id": "bear", "label": "熊市", "role": "negative", "color": "#dc2626", "order": 3},
    ],
    "clock": [
        {"id": "recovery", "label": "复苏", "role": "growth_up_inflation_down", "color": "#22c55e", "order": 1},
        {"id": "overheat", "label": "过热", "role": "growth_up_inflation_up", "color": "#f59e0b", "order": 2},
        {"id": "stagflation", "label": "滞胀", "role": "growth_down_inflation_up", "color": "#ef4444", "order": 3},
        {"id": "recession", "label": "衰退", "role": "growth_down_inflation_down", "color": "#3b82f6", "order": 4},
    ],
    "relative": [
        {"id": "numerator", "label": "分子占优", "role": "positive", "color": "#2563eb", "order": 1},
        {"id": "balanced", "label": "均衡", "role": "neutral", "color": "#64748b", "order": 2},
        {"id": "denominator", "label": "分母占优", "role": "negative", "color": "#d97706", "order": 3},
    ],
}


TEMPLATES: list[dict[str, Any]] = [
    {
        "id": "bull-bear-causal",
        "name": "指数牛熊震荡（实时可用）",
        "description": "单边滤波、趋势阈值、滞回与确认期，边界不会因未来数据重绘。",
        "definition": {
            "name": "沪深300牛熊震荡",
            "description": "以可得数据识别可交易的牛熊震荡状态。",
            "template_id": "bull-bear-causal",
            "target": {"kind": "index", "series_id": "000300.SH", "name": "沪深300", "frequency": "daily", "source_api": "index_daily", "ts_code": "000300.SH", "field": "close"},
            "features": {"transform": "log", "filter": "ema", "window": 20, "slope_window": 5, "volatility_window": 20},
            "algorithm": {"family": "causal_filter", "parameters": {"bull_enter": 0.0015, "bull_exit": 0.0002, "bear_enter": -0.0015, "bear_exit": -0.0002, "confirmation": 3, "min_duration": 5}},
            "states": DEFAULT_STATES["market"],
            "validation": {"walk_forward": True, "folds": 4, "stability_perturbation": 0.1},
            "usage_intent": "taa",
        },
    },
    {
        "id": "merrill-clock",
        "name": "美林时钟（宏观发布日期口径）",
        "description": "按增长与通胀方向划分复苏、过热、滞胀、衰退。",
        "definition": {
            "name": "美林时钟",
            "description": "以宏观指标的实际发布日期和方向变化识别经济周期。",
            "template_id": "merrill-clock",
            "target": {"kind": "inline", "series_id": "macro-clock", "name": "增长与通胀", "frequency": "monthly", "rows": []},
            "features": {"filter": "ema", "window": 3, "slope_window": 3},
            "algorithm": {"family": "merrill_clock", "parameters": {"growth_field": "growth", "inflation_field": "inflation", "confirmation": 2}},
            "states": DEFAULT_STATES["clock"],
            "validation": {"walk_forward": True, "folds": 4},
            "usage_intent": "research_display",
        },
    },
    {
        "id": "size-rotation",
        "name": "大小盘轮动",
        "description": "严格交集对齐中证1000与沪深300，以相对价格趋势识别大小盘占优区间。",
        "definition": {
            "name": "大盘小盘轮动",
            "description": "以中证1000相对沪深300的趋势识别大小盘占优状态。",
            "template_id": "size-rotation",
            "target": {
                "kind": "relative",
                "series_id": "size-rotation",
                "name": "中证1000 / 沪深300",
                "frequency": "daily",
                "numerator": {"kind": "index", "series_id": "000852.SH", "name": "中证1000", "frequency": "daily", "source_api": "index_daily", "ts_code": "000852.SH", "field": "close"},
                "denominator": {"kind": "index", "series_id": "000300.SH", "name": "沪深300", "frequency": "daily", "source_api": "index_daily", "ts_code": "000300.SH", "field": "close"},
                "transform": "log_ratio",
            },
            "features": {"filter": "ema", "window": 20, "slope_window": 5},
            "algorithm": {"family": "relative_strength", "parameters": {"upper": 0.001, "lower": -0.001, "confirmation": 3}},
            "states": [
                {"id": "small_cap", "label": "小盘占优", "role": "positive", "color": "#2563eb", "order": 1},
                {"id": "balanced", "label": "风格均衡", "role": "neutral", "color": "#64748b", "order": 2},
                {"id": "large_cap", "label": "大盘占优", "role": "negative", "color": "#d97706", "order": 3},
            ],
            "validation": {"walk_forward": True, "folds": 4},
            "usage_intent": "taa",
        },
    },
    {
        "id": "growth-value-rotation",
        "name": "成长价值轮动",
        "description": "严格交集对齐沪深300成长与沪深300价值，以相对价格趋势识别风格占优区间。",
        "definition": {
            "name": "成长价值轮动",
            "description": "以沪深300成长相对价值的趋势识别风格占优状态。",
            "template_id": "growth-value-rotation",
            "target": {
                "kind": "relative",
                "series_id": "growth-value-rotation",
                "name": "沪深300成长 / 沪深300价值",
                "frequency": "daily",
                "numerator": {"kind": "index", "series_id": "000918.CSI", "name": "沪深300成长", "frequency": "daily", "source_api": "index_daily", "ts_code": "000918.CSI", "field": "close"},
                "denominator": {"kind": "index", "series_id": "000919.CSI", "name": "沪深300价值", "frequency": "daily", "source_api": "index_daily", "ts_code": "000919.CSI", "field": "close"},
                "transform": "log_ratio",
            },
            "features": {"filter": "ema", "window": 20, "slope_window": 5},
            "algorithm": {"family": "relative_strength", "parameters": {"upper": 0.001, "lower": -0.001, "confirmation": 3}},
            "states": [
                {"id": "growth", "label": "成长占优", "role": "positive", "color": "#7c3aed", "order": 1},
                {"id": "balanced", "label": "风格均衡", "role": "neutral", "color": "#64748b", "order": 2},
                {"id": "value", "label": "价值占优", "role": "negative", "color": "#0f766e", "order": 3},
            ],
            "validation": {"walk_forward": True, "folds": 4},
            "usage_intent": "taa",
        },
    },
]


def meta_contract() -> dict[str, Any]:
    """Return the stable UI discovery contract."""

    return {
        "schema_version": SCHEMA_VERSION,
        "modes": [
            {"id": "realtime", "label": "实时识别", "description": "只使用当时已经可得的数据和单边算法。"},
            {"id": "retrospective", "label": "事后识别", "description": "允许全样本拟合或双边算法，仅用于研究解释。"},
        ],
        "data_sources": [
            {"id": "inline", "name": "粘贴/上传数据", "label": "粘贴/上传数据", "point_in_time": True, "supports_vintage": True},
            {"id": "index", "name": "指数行情", "label": "指数行情", "point_in_time": True, "supports_projection_filter": True},
            {"id": "relative", "name": "两序列相对强弱", "label": "两序列相对强弱", "point_in_time": True, "alignment": "strict_intersection"},
            {
                "id": "indicator",
                "name": "指标中心版本",
                "label": "指标中心版本",
                "point_in_time": True,
                "exact_revision_required": True,
                "execution_backend": "numba_njit_fixed_signature",
                "legacy_allowed": False,
            },
        ],
        "indicator_catalog": [],
        "indicator_periods": period_metadata(),
        "feature_catalog": [
            {"id": "ema", "label": "单边指数平滑", "causal": True},
            {"id": "kalman", "label": "单边卡尔曼滤波", "causal": True},
            {"id": "sma", "label": "单边移动平均", "causal": True},
            {"id": "zero_phase", "label": "零相位双边滤波", "causal": False, "repaints": True},
        ],
        "algorithm_families": [
            {"id": "causal_filter", "label": "趋势滤波 + 滞回确认", "supports_realtime": True, "phase": "P0"},
            {"id": "turning_point", "label": "峰谷周期划分", "supports_realtime": False, "phase": "P0"},
            {"id": "merrill_clock", "label": "美林时钟", "supports_realtime": True, "phase": "P0"},
            {"id": "relative_strength", "label": "相对强弱轮动", "supports_realtime": True, "phase": "P0"},
            {"id": "hmm", "label": "高斯隐马尔可夫", "supports_realtime": True, "phase": "P1", "realtime_policy": "initial_train_then_filter"},
            {"id": "markov", "label": "Markov 状态切换", "supports_realtime": True, "phase": "P1", "realtime_policy": "initial_train_then_filter"},
            {"id": "gmm", "label": "高斯混合聚类", "supports_realtime": True, "phase": "P1", "realtime_policy": "initial_train_then_predict"},
            {"id": "change_point", "label": "结构突变检测", "supports_realtime": True, "phase": "P1", "realtime_policy": "online_detector"},
            {"id": "ensemble", "label": "候选算法集成", "supports_realtime": True, "phase": "P1", "realtime_policy": "weighted_consensus_with_rejection"},
        ],
        "templates": copy.deepcopy(TEMPLATES),
        "application_targets": [
            {"id": "research_display", "label": "研究展示", "requires_causal": False},
            {"id": "product_research", "label": "产品研究", "requires_causal": False},
            {"id": "formal_backtest", "label": "正式回测", "requires_causal": True},
            {"id": "taa", "label": "战术资产配置", "requires_causal": True},
        ],
        "causality_classes": ["causal", "point_in_time_trained", "non_causal", "repainting"],
        "formula_language": formula_language_meta(),
        "taa_backtest": {
            "endpoint_template": "/api/historical-regimes/runs/{run_id}/taa-backtest",
            "required_run": {
                "immutable": True,
                "mode": "realtime",
                "published_usage_any_of": ["taa", "formal_backtest"],
            },
            "timing_rule": "仅使用 effective_date 不晚于收益期初 period_start 的情景概率；未显式填写时以紧邻的上一收益期末为期初，首期不使用情景信号。",
            "allocation_rule": "base_weights 加上按状态概率加权的 state_tilts；无有效状态回归基础权重。",
            "limits": ["min_weight", "max_weight", "max_abs_tilt"],
            "required_policies": ["periods_per_year", "max_signal_age_days"],
            "comparison_cost_policy": "SAA 与 TAA 使用相同交易成本模型。",
            "missing_return_policy": "reject",
        },
        "limits": {"max_inline_rows": 20000, "max_states": 12, "max_compare_runs": 8},
    }


def normalize_definition(fields: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize a user-authored regime definition."""

    if not isinstance(fields, dict):
        raise ValidationError("INVALID_DEFINITION", "情景定义必须是对象。", "definition")
    definition = copy.deepcopy(fields)
    name = str(definition.get("name") or "").strip()
    if not name or len(name) > 100:
        raise ValidationError("INVALID_REGIME_NAME", "情景定义名称长度应为 1 至 100 个字符。", "name")
    description = str(definition.get("description") or "").strip()
    if len(description) > 1000:
        raise ValidationError("INVALID_REGIME_DESCRIPTION", "情景定义说明不能超过 1000 个字符。", "description")
    target = definition.get("target") or definition.get("data")
    if not isinstance(target, dict) or target.get("kind") not in {"inline", "index", "relative", "indicator"}:
        raise ValidationError("INVALID_DATA_SOURCE", "target.kind 必须是 inline、index、relative 或 indicator。", "target.kind")
    target = copy.deepcopy(target)
    if target.get("kind") == "indicator":
        indicator_id = str(target.get("indicator_id") or "").strip()
        product_kind = str(target.get("product_kind") or "").strip().lower()
        product_id = str(target.get("product_id") or "").strip()
        period = str(target.get("period") or "").strip().upper()
        revision = target.get("indicator_revision")
        if not indicator_id:
            raise ValidationError("MISSING_INDICATOR_ID", "指标数据源必须选择指标。", "target.indicator_id")
        if isinstance(revision, bool):
            raise ValidationError("INVALID_INDICATOR_REVISION", "指标版本必须是正整数。", "target.indicator_revision")
        try:
            revision = int(revision)
        except (TypeError, ValueError) as exc:
            raise ValidationError("INVALID_INDICATOR_REVISION", "指标版本必须是正整数。", "target.indicator_revision") from exc
        if revision < 1:
            raise ValidationError("INVALID_INDICATOR_REVISION", "指标版本必须是正整数。", "target.indicator_revision")
        if product_kind not in {"etf", "fund"}:
            raise ValidationError("INVALID_INDICATOR_PRODUCT_KIND", "指标数据源的产品类型必须是 etf 或 fund。", "target.product_kind")
        if not product_id:
            raise ValidationError("MISSING_INDICATOR_PRODUCT", "指标数据源必须填写产品编号。", "target.product_id")
        if period not in SUPPORTED_PERIODS:
            raise ValidationError("INVALID_INDICATOR_PERIOD", "指标数据源必须选择受支持的评价周期。", "target.period")
        availability_mode = str(target.get("availability_mode") or "point_in_time")
        if availability_mode != "point_in_time":
            raise ValidationError(
                "INDICATOR_REQUIRES_POINT_IN_TIME",
                "版本化指标序列只支持 point_in_time 可得口径。",
                "target.availability_mode",
            )
        target.update(
            {
                "indicator_id": indicator_id,
                "indicator_revision": revision,
                "product_kind": product_kind,
                "product_id": product_id,
                "period": period,
                "frequency": "daily",
                "availability_mode": "point_in_time",
                "series_id": str(
                    target.get("series_id")
                    or f"{indicator_id}@{revision}:{product_kind}:{product_id}:{period}"
                ),
            }
        )
    algorithm = definition.get("algorithm")
    if not isinstance(algorithm, dict):
        raise ValidationError("INVALID_ALGORITHM", "必须配置 algorithm。", "algorithm")
    family = str(algorithm.get("family") or "").strip().lower()
    supported = {item["id"] for item in meta_contract()["algorithm_families"]}
    if family not in supported:
        raise ValidationError("UNSUPPORTED_ALGORITHM", "不支持的历史情景识别算法。", "algorithm.family")
    algorithm["family"] = family
    parameters = algorithm.get("parameters")
    if not isinstance(parameters, dict):
        parameters = algorithm.get("params") if isinstance(algorithm.get("params"), dict) else {}
    algorithm = {**algorithm, "parameters": parameters}
    algorithm.pop("params", None)

    default_group = "clock" if family == "merrill_clock" else "relative" if family == "relative_strength" else "market"
    states = definition.get("states") or copy.deepcopy(DEFAULT_STATES[default_group])
    if not isinstance(states, list) or len(states) < 2 or len(states) > 12:
        raise ValidationError("INVALID_STATES", "状态字典必须包含 2 至 12 个状态。", "states")
    seen: set[str] = set()
    normalized_states: list[dict[str, Any]] = []
    for order, state in enumerate(states, start=1):
        if not isinstance(state, dict):
            raise ValidationError("INVALID_STATE", "每个状态必须是对象。", "states")
        state_id = str(state.get("id") or "").strip()
        label = str(state.get("label") or "").strip()
        if not state_id or not label or state_id in seen:
            raise ValidationError("INVALID_STATE", "状态 id 与名称不能为空，且 id 不得重复。", "states")
        seen.add(state_id)
        normalized_states.append({**state, "id": state_id, "label": label, "order": int(state.get("order", order))})

    feature_definition = definition.get("features") if isinstance(definition.get("features"), dict) else {}
    if feature_definition.get("indicator_ref"):
        raise ValidationError(
            "UNSUPPORTED_INDICATOR_REFERENCE",
            "请使用 target.kind=indicator 并锁定指标、版本、产品与周期；features.indicator_ref 不会被静默执行。",
            "features.indicator_ref",
        )
    formula = feature_definition.get("formula")
    if formula is not None and not isinstance(formula, str):
        raise ValidationError("INVALID_FORMULA", "features.formula 必须是字符串。", "features.formula")
    validation = definition.get("validation") if isinstance(definition.get("validation"), dict) else {}
    try:
        folds = int(validation.get("folds", 4))
        stability_perturbation = float(validation.get("stability_perturbation", validation.get("sensitivity_pct", 10) / 100.0))
    except (TypeError, ValueError) as exc:
        raise ValidationError("INVALID_VALIDATION_CONFIG", "验证参数必须是有效数值。", "validation") from exc
    if folds < 2 or folds > 12:
        raise ValidationError("INVALID_FOLDS", "walk-forward 折数必须在 2 至 12 之间。", "validation.folds")
    if not 0.0 <= stability_perturbation <= 0.5:
        raise ValidationError("INVALID_STABILITY_PERTURBATION", "稳定性扰动比例必须在 0 至 0.5 之间。", "validation.stability_perturbation")
    usage_intent = str(definition.get("usage_intent") or "research_display")
    if usage_intent not in APPLICATION_TARGETS:
        raise ValidationError("INVALID_USAGE", "不支持的应用目标。", "usage_intent")

    definition.pop("data", None)
    definition.pop("template", None)
    return {
        **definition,
        "name": name,
        "description": description,
        "target": target,
        "features": feature_definition,
        "algorithm": algorithm,
        "states": normalized_states,
        "validation": {**validation, "walk_forward": bool(validation.get("walk_forward", True)), "folds": folds, "stability_perturbation": stability_perturbation},
        "usage_intent": usage_intent,
        "schema_version": SCHEMA_VERSION,
        "template_id": str(definition.get("template_id") or definition.get("template") or "custom"),
    }

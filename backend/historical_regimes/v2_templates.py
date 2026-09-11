"""Editable starting templates for Regime Graph v2."""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any


MARKET_STATES = [
    {"id": "bull", "label": "牛市", "role": "positive", "color": "#16a34a", "order": 1},
    {"id": "sideways", "label": "震荡市", "role": "neutral", "color": "#64748b", "order": 2},
    {"id": "bear", "label": "熊市", "role": "negative", "color": "#dc2626", "order": 3},
]

CLOCK_STATES = [
    {"id": "recovery", "label": "复苏", "role": "growth_up_inflation_down", "color": "#22c55e", "order": 1},
    {"id": "overheat", "label": "过热", "role": "growth_up_inflation_up", "color": "#f59e0b", "order": 2},
    {"id": "stagflation", "label": "滞胀", "role": "growth_down_inflation_up", "color": "#ef4444", "order": 3},
    {"id": "recession", "label": "衰退", "role": "growth_down_inflation_down", "color": "#3b82f6", "order": 4},
]


def _ref(node_id: str, port: str = "value") -> dict[str, str]:
    return {"node_id": node_id, "port": port}


def _rotation_template(
    template_id: str,
    name: str,
    numerator_code: str,
    numerator_name: str,
    denominator_code: str,
    denominator_name: str,
    positive_state: tuple[str, str, str],
    negative_state: tuple[str, str, str],
) -> dict[str, Any]:
    return {
        "schema_version": "2.0",
        "name": name,
        "description": f"以{numerator_name}相对{denominator_name}的单边趋势识别风格区间。",
        "template_id": template_id,
        "graph": {
            "nodes": [
                {"id": "numerator", "type": "source.index", "parameters": {"ts_code": numerator_code, "name": numerator_name, "source_api": "index_daily", "field": "close"}},
                {"id": "denominator", "type": "source.index", "parameters": {"ts_code": denominator_code, "name": denominator_name, "source_api": "index_daily", "field": "close"}},
                {"id": "aligned", "type": "align.strict_intersection", "inputs": {"left": _ref("numerator"), "right": _ref("denominator")}},
                {"id": "ratio", "type": "math.divide", "inputs": {"left": _ref("aligned", "left"), "right": _ref("aligned", "right")}},
                {"id": "trend", "type": "rolling.slope", "parameters": {"window": 20}, "inputs": {"value": _ref("ratio")}},
                {"id": "classifier", "type": "model.threshold", "parameters": {"upper": 0.001, "lower": -0.001}, "inputs": {"value": _ref("trend")}},
                {"id": "confirmed", "type": "post.confirmation", "parameters": {"confirmation": 3, "min_duration": 5}, "inputs": {"state": _ref("classifier", "state")}},
            ],
            "outputs": {"state": _ref("confirmed", "state")},
            "exposed_node_ids": ["aligned", "ratio", "trend", "classifier"],
        },
        "states": [
            {"id": positive_state[0], "label": positive_state[1], "role": "positive", "color": positive_state[2], "order": 1},
            {"id": "balanced", "label": "风格均衡", "role": "neutral", "color": "#64748b", "order": 2},
            {"id": negative_state[0], "label": negative_state[1], "role": "negative", "color": negative_state[2], "order": 3},
        ],
        "evaluation_targets": [],
        "validation": {"walk_forward": True, "folds": 4},
        "usage_intent": "taa",
    }


TEMPLATES_V2: list[dict[str, Any]] = [
    {
        "id": "peak-trough-ps-v2", "name": "峰谷定界法", "version": 2,
        "tags": ["事后识别", "峰谷定界", "牛熊震荡", "月频"],
        "description": "月频峰谷划分牛熊趋势，再按振幅、方向效率和持续长度合并震荡；所有规则可编辑，仅限事后研究。",
        "definition": {
            "schema_version": "2.0", "name": "沪深300峰谷定界法", "template_id": "peak-trough-ps-v2",
            "description": "月度原始指数峰谷定界。左右各 8 月、最短阶段 4 月、最短周期 16 月，超过 20% 的短阶段可例外保留。首尾不完整区间不外推；修改频率后须重新选择以观测期数计的参数。",
            "graph": {
                "nodes": [
                    {"id": "market", "type": "source.index", "parameters": {"ts_code": "000300.SH", "name": "沪深300", "source_api": "index_daily", "field": "close"}},
                    {"id": "monthly", "type": "align.resample", "parameters": {"frequency": "monthly", "aggregation": "last"}, "inputs": {"value": _ref("market")}},
                    {"id": "dating", "type": "model.peak_trough", "parameters": {"left_window": 8, "right_window": 8, "head_window": 6, "tail_window": 6, "min_phase": 4, "min_cycle": 16, "amplitude_exception": 0.2, "sideways_enabled": True, "small_swing_threshold": 0.03, "sideways_max_range": 0.06, "sideways_max_efficiency": 0.25, "sideways_min_duration": 3}, "inputs": {"value": _ref("monthly")}},
                ],
                "outputs": {"state": _ref("dating", "state")},
                "exposed_node_ids": ["market", "monthly", "dating"],
            },
            "states": copy.deepcopy(MARKET_STATES),
            "evaluation_targets": [], "validation": {"walk_forward": True, "folds": 4},
            "usage_intent": "research_display",
        },
    },
    {
        "id": "manual-historical-events-v1", "name": "人工历史事件区间", "version": 1,
        "tags": ["事后识别", "人工标注", "历史事件", "允许重叠"],
        "default_mode": "retrospective",
        "description": "由研究员手工定义历史事件的开始和结束日期；多个事件可重叠，仅用于事后研究。",
        "definition": {
            "schema_version": "2.0", "name": "人工历史事件区间", "template_id": "manual-historical-events-v1",
            "description": "选择一条观察序列，并由人类维护可重叠的历史事件区间。事件不是互斥市场状态，也不是当时可交易信号。",
            "graph": {
                "nodes": [
                    {"id": "market", "type": "source.index", "parameters": {"ts_code": "000300.SH", "name": "沪深300", "source_api": "index_daily", "field": "close", "frequency": "daily"}},
                    {"id": "events", "type": "annotation.manual_events", "parameters": {"events": []}, "inputs": {"value": _ref("market")}},
                ],
                "outputs": {"state": _ref("events", "state")},
                "exposed_node_ids": ["market", "events"],
            },
            "states": [
                {"id": "event", "label": "事件覆盖", "role": "event", "color": "#7c3aed", "order": 1},
                {"id": "normal", "label": "事件外", "role": "neutral", "color": "#cbd5e1", "order": 2},
            ],
            "evaluation_targets": [],
            "validation": {"walk_forward": False, "folds": 4},
            "usage_intent": "research_display",
        },
    },
    {
        "id": "blank-three-state",
        "name": "从零搭建三状态模型",
        "description": "最小可运行图谱；替换数据源、算子、阈值或继续添加节点。",
        "definition": {
            "schema_version": "2.0",
            "name": "我的历史情景模型",
            "description": "从空白链路开始搭建。",
            "template_id": "blank-three-state",
            "graph": {
                "nodes": [
                    {"id": "source", "type": "source.inline", "parameters": {"rows": [], "frequency": "daily", "name": "待选择序列"}},
                    {"id": "returns", "type": "transform.return", "parameters": {"window": 1}, "inputs": {"value": _ref("source")}},
                    {"id": "classifier", "type": "model.threshold", "parameters": {"upper": 0.001, "lower": -0.001}, "inputs": {"value": _ref("returns")}},
                ],
                "outputs": {"state": _ref("classifier", "state"), "probabilities": _ref("classifier", "probabilities"), "confidence": _ref("classifier", "confidence")},
                "exposed_node_ids": ["source", "returns", "classifier"],
            },
            "states": copy.deepcopy(MARKET_STATES),
            "evaluation_targets": [],
            "validation": {"walk_forward": True, "folds": 4},
            "usage_intent": "research_display",
        },
    },
    {
        "id": "bull-bear-causal-v2",
        "version": 2,
        "name": "指数牛熊震荡 · 可编辑图谱",
        "description": "指数＋单向滤波＋幅度门槛＋连续确认；震荡由平坦趋势与低方向效率共同识别。",
        "definition": {
            "schema_version": "2.0",
            "name": "沪深300牛熊震荡 · 趋势滤波",
            "description": "慢趋势区分牛熊，幅度与连续确认过滤短反向信号；平坦斜率与低方向效率识别震荡。参数为研究起点，需样本外验证；实时不回填历史。",
            "template_id": "bull-bear-causal-v2",
            "graph": {
                "nodes": [
                    {"id": "market", "type": "source.index", "parameters": {"ts_code": "000300.SH", "name": "沪深300", "source_api": "index_daily", "field": "close"}},
                    {"id": "log_price", "type": "transform.log", "inputs": {"value": _ref("market")}},
                    {"id": "trend", "type": "filter.super_smoother", "parameters": {"period": 126}, "inputs": {"value": _ref("log_price")}},
                    {"id": "metrics", "type": "feature.trend_metrics", "parameters": {"volatility_window": 60, "slope_window": 20, "efficiency_window": 60, "scale_floor": 0.0001, "shock_window": 5, "drawdown_alert": 0.2, "shock_alert": 0.08}, "inputs": {"log_price": _ref("log_price"), "trend": _ref("trend")}},
                    {"id": "classifier", "type": "model.trend_regime", "parameters": {"band": 1.0, "trend_enter": 0.1, "flat_threshold": 0.05, "efficiency_ceiling": 0.25, "confirmation": 3}, "inputs": {"distance": _ref("metrics", "distance"), "slope": _ref("metrics", "slope"), "efficiency": _ref("metrics", "efficiency")}},
                ],
                "outputs": {"state": _ref("classifier", "state")},
                "exposed_node_ids": ["market", "log_price", "trend", "metrics", "classifier"],
            },
            "states": copy.deepcopy(MARKET_STATES),
            "evaluation_targets": [{"id": "csi300", "name": "沪深300", "source": {"kind": "index", "ts_code": "000300.SH", "source_api": "index_daily", "field": "close"}, "primary": True}],
            "validation": {"walk_forward": True, "folds": 4},
            "usage_intent": "taa",
        },
    },
    {
        "id": "merrill-clock-v2",
        "name": "美林时钟 · 可编辑图谱",
        "description": "增长与通胀输入彼此独立，可替换成任意可得宏观序列或指标版本。",
        "definition": {
            "schema_version": "2.0",
            "name": "美林时钟 v2",
            "description": "以增长、通胀方向构造四象限历史区间。",
            "template_id": "merrill-clock-v2",
            "graph": {
                "nodes": [
                    {"id": "growth", "type": "source.inline", "parameters": {"rows": [], "frequency": "monthly", "name": "增长指标"}},
                    {"id": "inflation", "type": "source.inline", "parameters": {"rows": [], "frequency": "monthly", "name": "通胀指标"}},
                    {"id": "clock_aligned", "type": "align.strict_intersection", "inputs": {"left": _ref("growth"), "right": _ref("inflation")}},
                    {"id": "growth_trend", "type": "rolling.slope", "parameters": {"window": 3}, "inputs": {"value": _ref("clock_aligned", "left")}},
                    {"id": "inflation_trend", "type": "rolling.slope", "parameters": {"window": 3}, "inputs": {"value": _ref("clock_aligned", "right")}},
                    {"id": "quadrant", "type": "model.quadrant", "parameters": {"growth_threshold": 0.0, "inflation_threshold": 0.0}, "inputs": {"growth": _ref("growth_trend"), "inflation": _ref("inflation_trend")}},
                    {"id": "confirmed", "type": "post.confirmation", "parameters": {"confirmation": 2, "min_duration": 1}, "inputs": {"state": _ref("quadrant", "state")}},
                ],
                "outputs": {"state": _ref("confirmed", "state")},
                "exposed_node_ids": ["growth", "inflation", "clock_aligned", "growth_trend", "inflation_trend", "quadrant"],
            },
            "states": copy.deepcopy(CLOCK_STATES),
            "evaluation_targets": [],
            "validation": {"walk_forward": True, "folds": 4},
            "usage_intent": "research_display",
        },
    },
    {
        "id": "size-rotation-v2",
        "name": "大小盘轮动 · 可编辑图谱",
        "description": "两条指数、相对强弱、趋势与阈值均为可替换节点。",
        "definition": _rotation_template(
            "size-rotation-v2", "大小盘轮动 v2", "000852.SH", "中证1000", "000300.SH", "沪深300",
            ("small_cap", "小盘占优", "#2563eb"), ("large_cap", "大盘占优", "#d97706"),
        ),
    },
    {
        "id": "growth-value-rotation-v2",
        "name": "成长价值轮动 · 可编辑图谱",
        "description": "两条风格指数、相对强弱、趋势与阈值均为可替换节点。",
        "definition": _rotation_template(
            "growth-value-rotation-v2", "成长价值轮动 v2", "000918.CSI", "沪深300成长", "000919.CSI", "沪深300价值",
            ("growth", "成长占优", "#7c3aed"), ("value", "价值占优", "#0f766e"),
        ),
    },
]


def _legacy_daily_peak_template() -> dict[str, Any]:
    item = copy.deepcopy(TEMPLATES_V2[0])
    item.update(id="peak-trough-daily-legacy-v1", name="峰谷定界法 · 旧版日频兼容", version=1,
                tags=["事后识别", "峰谷定界", "牛熊震荡", "日频"],
                description="日频峰谷趋势＋小波段合并；以整段振幅、方向效率及至少 20 个观测间隔识别震荡。阈值为可调研究起点。")
    definition = item["definition"]
    definition.update(template_id=item["id"], name="沪深300日频峰谷牛熊震荡", description=item["description"])
    nodes = definition["graph"]["nodes"]
    definition["graph"]["nodes"] = [nodes[0], nodes[2]]
    nodes[2]["inputs"]["value"] = _ref("market")
    nodes[2]["parameters"]["sideways_min_duration"] = 20
    definition["graph"]["exposed_node_ids"] = ["market", "dating"]
    return item


def _daily_peak_template() -> dict[str, Any]:
    nodes = [
        {"id": "market", "type": "source.index", "parameters": {"ts_code": "000300.SH", "name": "沪深300", "source_api": "index_daily", "field": "close"}},
        {"id": "pivots", "type": "pivot.local_extrema", "parameters": {"left_window": 8, "right_window": 8, "head_window": 6, "tail_window": 6}, "inputs": {"value": _ref("market")}},
        {"id": "segments", "type": "segment.between_pivots", "inputs": {"pivot": _ref("pivots", "pivot")}},
        {"id": "change", "type": "segment.change", "inputs": {"value": _ref("market"), "start": _ref("segments", "start"), "end": _ref("segments", "end")}},
        {"id": "upper", "type": "source.constant", "label": "上涨门槛", "parameters": {"value": .03}, "inputs": {"anchor": _ref("market")}},
        {"id": "lower", "type": "source.constant", "label": "下跌门槛", "parameters": {"value": -.03}, "inputs": {"anchor": _ref("market")}},
        {"id": "classifier", "type": "model.range_threshold", "inputs": {"value": _ref("change"), "upper_bound": _ref("upper"), "lower_bound": _ref("lower")}},
    ]
    return {"id": "peak-trough-daily-v2", "name": "峰谷定界法 · 日频牛熊震荡", "version": 2,
            "tags": ["事后识别", "峰谷定界", "独立算子", "日频"],
            "description": "峰谷定位、区间涨跌幅、上下门槛独立配置；单个小幅完整波段即可判断震荡，不默认合并或过滤。",
            "definition": {"schema_version": "2.0", "name": "沪深300日频峰谷牛熊震荡", "template_id": "peak-trough-daily-v2",
                "description": "按四窗口定位峰谷，再计算相邻峰谷收益。涨幅超过3%为牛，跌幅超过3%为熊，其余为震荡；阈值为可调研究起点。未完成尾段不外推。",
                "graph": {"nodes": nodes, "outputs": {"state": _ref("classifier", "state")}, "exposed_node_ids": [node["id"] for node in nodes]},
                "states": copy.deepcopy(MARKET_STATES), "evaluation_targets": [], "validation": {"walk_forward": True, "folds": 4}, "usage_intent": "research_display"}}


TEMPLATES_V2.insert(1, _daily_peak_template())
TEMPLATES_V2.append(_legacy_daily_peak_template())


def _macro_clock_template() -> dict[str, Any]:
    """New default bindings; historical template definitions remain immutable."""
    item = copy.deepcopy(next(item for item in TEMPLATES_V2 if item["id"] == "merrill-clock-v2"))
    item.update(id="merrill-clock-macro-v3", name="美林时钟 · PMI与CPI", version=3,
                default_mode="retrospective", tags=["宏观周期", "月频", "事后研究"],
                description="制造业PMI与CPI同比，按共同月份计算3期趋势，再分四象限并连续2期确认。缺少发布日期时使用事后研究。")
    definition = item["definition"]
    definition.update(template_id=item["id"], name=item["name"], description=item["description"])
    for node_id, dataset, field, label in (
        ("growth", "cn_pmi", "pmi010000", "增长：制造业PMI"),
        ("inflation", "cn_cpi", "nt_yoy", "通胀：CPI同比"),
    ):
        node = next(node for node in definition["graph"]["nodes"] if node["id"] == node_id)
        node.update(type="source.macro", label=label, parameters={"dataset": f"macro_{dataset}_df.parquet", "source_api": dataset, "field": field, "frequency": "monthly", "name": label})
    definition["graph"]["outputs"].update(growth=_ref("clock_aligned", "left"), inflation=_ref("clock_aligned", "right"),
                                          growth_trend=_ref("growth_trend"), inflation_trend=_ref("inflation_trend"))
    definition["graph"]["channel_metadata"] = {key: {"label": label} for key, label in (
        ("growth", "制造业PMI"), ("inflation", "CPI同比（%）"),
        ("growth_trend", "增长趋势"), ("inflation_trend", "通胀趋势"))}
    return item


TEMPLATES_V2.append(_macro_clock_template())


def _versioned_template(item: dict[str, Any]) -> dict[str, Any]:
    from .v2_registry import NODE_REGISTRY
    result = copy.deepcopy(item)
    result["version"] = int(item.get("version", 1))
    from .v2_contracts import parse_definition_v2
    from .temporal_capability import analyze_temporal
    temporal = analyze_temporal(parse_definition_v2(result["definition"]), NODE_REGISTRY)
    causal = temporal["realtime_supported"]
    result["temporal_capability"] = temporal
    result["supported_modes"] = ["realtime", "retrospective"] if causal else ["retrospective"]
    result["default_mode"] = item.get("default_mode", "realtime" if causal else "retrospective")
    encoded = json.dumps(
        result["definition"],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    result["content_hash"] = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return result


def _step_templates() -> list[dict[str, Any]]:
    """New template identities; stored definitions and old hashes are untouched."""
    from .composite_expansion import expand_composite
    from .granular_registry import COMPOSITE_STEPS

    items = []
    for original in TEMPLATES_V2:
        composites = [node for node in original["definition"]["graph"]["nodes"] if node["type"] in COMPOSITE_STEPS]
        if not composites or "legacy" in original["id"]:
            items.append(original)
            continue
        item = copy.deepcopy(original)
        item.update(id=f"{original['id']}-steps-v1", name=f"{original['name']} · 分步", version=1)
        if original["id"] == "merrill-clock-v2":
            item["authoring_hidden"] = True
        item["tags"] = [*item.get("tags", []), "可编辑计算步骤"]
        for node in composites:
            item["definition"] = expand_composite(item["definition"], node["id"], "retrospective")["definition"]
        item["definition"]["template_id"] = item["id"]
        items.append(item)
        # Original IDs remain explicitly retrievable for compatibility, but
        # new authoring recommends the independent expanded definition.
        items.append({**original, "authoring_hidden": True})
    return items


def list_templates_v2() -> list[dict[str, Any]]:
    return [_versioned_template(item) for item in _step_templates()]


def get_template_v2(template_id: str) -> dict[str, Any] | None:
    # Exact historical lookups never depend on expanding a newer template.
    for item in TEMPLATES_V2:
        if item["id"] == template_id:
            return _versioned_template(item)
    for item in _step_templates():
        if item["id"] == template_id:
            return _versioned_template(item)
    return None


def instantiate_template_v2(template_id: str) -> dict[str, Any] | None:
    item = get_template_v2(template_id)
    return copy.deepcopy(item["definition"]) if item is not None else None


__all__ = ["CLOCK_STATES", "MARKET_STATES", "TEMPLATES_V2", "get_template_v2", "instantiate_template_v2", "list_templates_v2"]

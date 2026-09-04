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
        "name": "指数牛熊震荡 · 可编辑图谱",
        "description": "对数价格、滚动斜率、滞回和确认期均可替换或调参。",
        "definition": {
            "schema_version": "2.0",
            "name": "沪深300牛熊震荡 v2",
            "description": "以单边算子识别实时可用的牛熊震荡状态。",
            "template_id": "bull-bear-causal-v2",
            "graph": {
                "nodes": [
                    {"id": "market", "type": "source.index", "parameters": {"ts_code": "000300.SH", "name": "沪深300", "source_api": "index_daily", "field": "close"}},
                    {"id": "log_price", "type": "transform.log", "inputs": {"value": _ref("market")}},
                    {"id": "trend", "type": "rolling.slope", "parameters": {"window": 20}, "inputs": {"value": _ref("log_price")}},
                    {"id": "classifier", "type": "model.hysteresis", "parameters": {"upper_enter": 0.0015, "upper_exit": 0.0002, "lower_enter": -0.0015, "lower_exit": -0.0002}, "inputs": {"value": _ref("trend")}},
                    {"id": "confirmed", "type": "post.confirmation", "parameters": {"confirmation": 3, "min_duration": 5}, "inputs": {"state": _ref("classifier", "state")}},
                ],
                "outputs": {"state": _ref("confirmed", "state")},
                "exposed_node_ids": ["market", "log_price", "trend", "classifier"],
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


def _versioned_template(item: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(item)
    result["version"] = 1
    encoded = json.dumps(
        result["definition"],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    result["content_hash"] = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return result


def list_templates_v2() -> list[dict[str, Any]]:
    return [_versioned_template(item) for item in TEMPLATES_V2]


def get_template_v2(template_id: str) -> dict[str, Any] | None:
    for item in TEMPLATES_V2:
        if item["id"] == template_id:
            return _versioned_template(item)
    return None


def instantiate_template_v2(template_id: str) -> dict[str, Any] | None:
    item = get_template_v2(template_id)
    return copy.deepcopy(item["definition"]) if item is not None else None


__all__ = ["CLOCK_STATES", "MARKET_STATES", "TEMPLATES_V2", "get_template_v2", "instantiate_template_v2", "list_templates_v2"]

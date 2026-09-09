"""Stable node registry for user-authored historical-regime graphs."""

from __future__ import annotations

import copy
from typing import Any

from compute_policy import NJIT_BACKEND, THIRD_PARTY_BACKEND
from computation_graph.series_operators import register_series_operators
from research_series.product_sources import PRODUCT_SOURCES


REGISTRY_VERSION = "regime-graph-nodes/2.12.0"

SERIES = "series<float64>"
BOOL_SERIES = "series<bool>"
STATE_CODES = "state_codes<int64>"
REGIME_CANDIDATE = "regime_candidate<time,state>"
REGIME_OUTPUT = "regime_output<time,state>"
PROBABILITIES = "probabilities<time,state>"
CONFIDENCE = "confidence<time>"
MATRIX = "matrix<time,feature>"
INDEX_SERIES = "index<time>"
REASON_CODES = "reason_codes<int64>"


CATEGORY_LABELS = {
    "indicator_calculation": "指标计算",
    "indicator": "指标通用算子",
    "source": "数据源",
    "alignment": "数据对齐",
    "feature": "特征构建",
    "transform": "时序变换",
    "arithmetic": "数学运算",
    "filter": "滤波",
    "rolling": "滚动统计",
    "model": "状态识别模型",
    "postprocess": "状态稳定化",
    "output": "结果输出",
}

PORT_LABELS = {
    "value": "数值序列",
    "left": "左侧序列",
    "right": "右侧序列",
    "anchor": "基准时间轴",
    "feature": "待对齐特征",
    "feature_1": "特征一",
    "feature_2": "特征二",
    "feature_3": "特征三",
    "feature_4": "特征四",
    "features": "特征矩阵",
    "growth": "增长指标",
    "inflation": "通胀指标",
    "state": "状态序列",
    "state_1": "候选状态一",
    "state_2": "候选状态二",
    "state_3": "候选状态三",
    "state_4": "候选状态四",
    "primary": "优先状态",
    "secondary": "备选状态",
    "score": "状态得分",
    "confidence": "置信度",
    "probabilities": "状态概率矩阵",
    "recognition_index": "识别时点",
    "effective_index": "生效时点",
    "reason_code": "判定原因",
}

PORT_TYPE_LABELS = {
    SERIES: "数值时间序列",
    BOOL_SERIES: "布尔时间序列",
    MATRIX: "时间 × 特征矩阵",
    STATE_CODES: "状态编码序列",
    REGIME_CANDIDATE: "候选状态结果",
    REGIME_OUTPUT: "完整情景识别结果",
    PROBABILITIES: "时间 × 状态概率矩阵",
    CONFIDENCE: "置信度时间序列",
    INDEX_SERIES: "时点索引序列",
    REASON_CODES: "判定原因编码序列",
}

PARAMETER_LABELS = {
    "rows": "内联数据记录",
    "inline_rows": "内联数据记录",
    "value_field": "数值字段",
    "date_field": "观察日期字段",
    "available_at_field": "可得日期字段",
    "vintage_field": "数据版本字段",
    "revision_field": "修订号字段",
    "availability_mode": "数据可得性口径",
    "name": "显示名称",
    "artifact_id": "上传数据版本",
    "checksum": "文件校验值",
    "format": "文件格式",
    "ts_code": "指数代码",
    "source_api": "行情数据接口",
    "field": "数值字段",
    "frequency": "数据频率",
    "start_date": "开始日期",
    "end_date": "结束日期",
    "snapshot_id": "数据快照",
    "snapshot_generation": "快照代次",
    "source_file": "来源文件",
    "file_checksum": "来源文件校验值",
    "indicator_id": "指标编号",
    "indicator_revision": "指标修订版本",
    "product_kind": "研究对象类型",
    "product_id": "研究对象编号",
    "period": "计算区间",
    "data_fingerprint": "数据指纹",
    "indicator_data_snapshot": "指标数据快照",
    "numerator": "分子序列",
    "denominator": "分母序列",
    "transform": "相对强弱算法",
    "dataset": "宏观数据集",
    "series_id": "研究序列",
    "code": "序列代码",
    "value": "常量值",
    "max_age_days": "允许的最长陈旧天数",
    "aggregation": "聚合方法",
    "every": "采样间隔",
    "offset": "采样起点偏移",
    "expression": "计算公式",
    "variables": "公式变量映射",
    "window": "计算窗口",
    "periods": "比较期数",
    "lower": "下限",
    "upper": "上限",
    "process_variance": "过程噪声方差",
    "measurement_variance": "观测噪声方差",
    "upper_enter": "进入上方状态阈值",
    "upper_exit": "退出上方状态阈值",
    "lower_enter": "进入下方状态阈值",
    "lower_exit": "退出下方状态阈值",
    "growth_threshold": "增长分界线",
    "inflation_threshold": "通胀分界线",
    "min_move": "最小有效涨跌幅",
    "components": "状态数量",
    "initial_train_size": "初始训练样本数",
    "iterations": "最大迭代次数",
    "initialization_strategy": "初始参数生成方法",
    "random_seed": "随机种子",
    "initial_means": "显式初始中心",
    "threshold": "识别阈值",
    "confirmation": "确认期数",
    "weights": "候选模型权重",
    "consensus_threshold": "共识门槛",
    "min_duration": "最短持续期数",
    "mapping": "模型分量到状态的映射",
    "floor": "最低置信度",
    "declaration": "隔离执行声明",
    "adapter_id": "管理员模型适配器",
}

PARAMETER_DESCRIPTIONS = {
    "field": "选择本节点实际送入后续计算的数值列；界面显示中文名称，保存时仍记录稳定字段代码。",
    "window": "每个时点只使用截至当时最近若干期数据；窗口越大通常越平滑，但反应更慢。",
    "periods": "当前值与多少期之前的值比较。",
    "availability_mode": "时点可得只使用当时已公布的数据；最新口径可包含后续修订，仅适合事后研究。",
    "max_age_days": "宏观或低频数据超过该天数仍未更新时保留缺失，避免把过期信息当成实时信号。",
    "aggregation": "把一个低频区间内的多条记录合并为一条记录的方法。",
    "confirmation": "新状态连续出现达到该期数后才确认切换，用于抑制短暂噪声。",
    "min_duration": "状态一旦生效，至少保持的期数。",
    "upper": "得分高于该值时进入上方状态。",
    "lower": "得分低于该值时进入下方状态。",
    "weights": "多个候选状态模型参与共识判断时的相对权重。",
    "mapping": "按模型分量顺序填写业务状态编号；-1 表示该分量暂不分类。",
    "floor": "置信度低于该值的观察点不输出明确状态。",
    "usage_intent": "研究定义本身不绑定用途；发布或被业务引用时再按目标场景执行门禁。",
}

ENUM_LABELS = {
    "daily": "日频",
    "weekly": "周频",
    "monthly": "月频",
    "quarterly": "季频",
    "yearly": "年频",
    "annual": "年频",
    "irregular": "不定期",
    "first": "区间首值",
    "last": "区间末值",
    "mean": "区间平均值",
    "sum": "区间合计值",
    "point_in_time": "时点可得口径",
    "latest": "最新修订口径（仅事后研究）",
    "ratio": "直接比值",
    "log_ratio": "对数比值",
    "quantile": "按分位数初始化",
    "random": "按随机种子初始化",
    "explicit": "使用显式初始中心",
    "parquet": "Parquet 数据文件",
    "value": "数值",
    "observation_date": "观察日期",
    "available_at": "可得日期",
    "vintage": "数据版本",
    "revision": "修订号",
    "index_daily": "中证/上证指数日行情",
    "sw_daily": "申万指数日行情",
    "ci_daily": "中信指数日行情",
    "ths_daily": "同花顺指数日行情",
    "dc_daily": "东方财富指数日行情",
    "tdx_daily": "通达信指数日行情",
    "index_global": "全球指数日行情",
    "fut_index_daily": "股指期货日行情",
    "open": "开盘点位",
    "high": "最高点位",
    "low": "最低点位",
    "close": "收盘点位",
    "pre_close": "前收盘点位",
    "change": "涨跌点数",
    "pct_chg": "涨跌幅",
    "vol": "成交量",
    "amount": "成交额",
    "swing": "振幅",
    "turnover_rate": "换手率",
    "turnover_rate_f": "自由流通换手率",
    "pe": "市盈率",
    "pe_ttm": "滚动市盈率",
    "pb": "市净率",
}

INDEX_FIELD_OPTIONS = [
    "open", "high", "low", "close", "pre_close", "change", "pct_chg",
    "vol", "amount", "swing", "turnover_rate", "turnover_rate_f", "pe",
    "pe_ttm", "pb",
]
INDEX_SOURCE_OPTIONS = [
    "index_daily", "sw_daily", "ci_daily", "ths_daily", "dc_daily",
    "tdx_daily", "index_global", "fut_index_daily",
]


def _port(name: str, value_type: str, *, required: bool = True) -> dict[str, Any]:
    return {"name": name, "type": value_type, "required": required}


def _numeric_node(
    node_id: str,
    label: str,
    category: str,
    inputs: list[dict[str, Any]],
    outputs: list[dict[str, Any]],
    parameter_schema: dict[str, Any],
    *,
    causal: bool = True,
    phase: str = "P0",
) -> dict[str, Any]:
    kernel_id = (
        "typed_formula_plan"
        if node_id == "feature.formula"
        else node_id.replace(".", "_")
    )
    return {
        "id": node_id,
        "type_id": node_id,
        "version": 1,
        "type_version": 1,
        "label": label,
        "available": True,
        "status": "available",
        "unavailable_reason": None,
        "category": category,
        "phase": phase,
        "causal": causal,
        "repaints": not causal,
        "supports_realtime": causal,
        "minimum_samples": 2 if category not in {"model", "rolling"} else 5,
        "cost_estimate": {"class": "linear", "expression": "O(T)", "unit": "observations"},
        "kernel_id": kernel_id,
        "kernel_version": "typed-njit-v2.1" if node_id == "feature.formula" else "regime-graph-kernels/2.8.0",
        "model_version": "1" if category == "model" else None,
        "formula_language": (
            {
                "id": "historical-regime-typed-causal-series",
                "expression_parameter": "expression",
                "variables": [item["name"] for item in inputs],
                "prepare_required": True,
                "token_bound_to_ast": True,
            }
            if node_id == "feature.formula"
            else None
        ),
        "inputs": inputs,
        "outputs": outputs,
        "parameter_schema": parameter_schema,
        "njit_policy": {
            "execution_backend": NJIT_BACKEND,
            "njit_required": True,
            "fixed_signature": True,
            "request_time_compilation": 0,
            "python_fallback": 0,
            "execution_lane": "numeric_njit",
        },
    }


def _source_node(
    node_id: str,
    label: str,
    parameter_schema: dict[str, Any],
    *,
    available: bool = True,
    status: str = "available",
    unavailable_reason: str | None = None,
) -> dict[str, Any]:
    return {
        "id": node_id,
        "type_id": node_id,
        "version": 1,
        "type_version": 1,
        "label": label,
        "available": available,
        "status": status,
        "unavailable_reason": unavailable_reason,
        "category": "source",
        "phase": "P0",
        "causal": True,
        "repaints": False,
        "supports_realtime": True,
        "minimum_samples": 1,
        "cost_estimate": {"class": "io_bound", "expression": "O(T)", "unit": "observations"},
        "kernel_id": None,
        "kernel_version": None,
        "model_version": None,
        "formula_language": None,
        "inputs": [],
        "outputs": [_port("value", SERIES)],
        "parameter_schema": parameter_schema,
        "njit_policy": {
            "execution_backend": "io_boundary",
            "njit_required": False,
            "numeric_compute": False,
            "request_time_compilation": 0,
            "python_fallback": 0,
            "execution_lane": "data_boundary",
        },
    }


_EMPTY_OBJECT = {"type": "object", "properties": {}, "additionalProperties": False}
_WINDOW = {
    "type": "object",
    "properties": {"window": {"type": "integer", "minimum": 1, "maximum": 5000, "default": 20}},
    "additionalProperties": False,
}


def _latent_model_parameter_schema() -> dict[str, Any]:
    """Shared, versioned controls for NJIT latent-state model experiments."""

    return {
        "type": "object",
        "properties": {
            "components": {
                "type": "integer",
                "minimum": 2,
                "maximum": 12,
                "default": 3,
                "title": "状态数",
            },
            "initial_train_size": {
                "type": "integer",
                "minimum": 10,
                "maximum": 20000,
                "default": 60,
                "title": "初始训练样本",
            },
            "iterations": {
                "type": "integer",
                "minimum": 1,
                "maximum": 200,
                "default": 60,
                "title": "最大迭代次数",
            },
            "initialization_strategy": {
                "type": "string",
                "enum": ["quantile", "random", "explicit"],
                "default": "quantile",
                "title": "初始化策略",
                "description": "分位数初始化保持既有行为；随机初始化由 random_seed 驱动；显式初始化使用标准化特征空间的 initial_means。",
            },
            "random_seed": {
                "type": "integer",
                "minimum": 0,
                "maximum": 2147483646,
                "default": 0,
                "title": "随机种子",
            },
            "initial_means": {
                "type": "array",
                "maxItems": 12,
                "default": [],
                "title": "初始中心",
                "description": "仅 explicit 策略使用；形状必须为 状态数 x 特征数，数值位于标准化特征空间。",
                "items": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 64,
                    "items": {"type": "number"},
                },
            },
        },
        "additionalProperties": False,
    }


def _product_source_node(kind: str) -> dict[str, Any]:
    spec = PRODUCT_SOURCES[kind]
    return _source_node(f"source.{kind}", spec["label"], {
        "type": "object", "required": ["ts_code"], "additionalProperties": False,
        "properties": {
            "ts_code": {"type": "string", "minLength": 1, "title": "产品代码"},
            "source_api": ({"enum": ["fund_daily", "fund_nav"], "enum_labels": ["ETF交易行情", "ETF净值"]} if kind == "etf" else {"enum": [spec["source_api"]], "default": spec["source_api"], "enum_labels": [spec["label"]]}),
            **({"adjustment_checksum": {"type": "string"}} if kind == "etf" else {}),
            "field": {"enum": list(spec["fields"]), "default": spec["default_field"],
                      "enum_labels": [label for label, _ in spec["fields"].values()], "option_source": "research_series.fields"},
            "frequency": {"enum": ["daily"], "default": "daily"},
            "start_date": {"type": "string", "format": "date"},
            "end_date": {"type": "string", "format": "date"},
            **{name: {"type": "string"} for name in ("name", "snapshot_id", "snapshot_generation", "source_file", "file_checksum")},
        },
    })


NODE_REGISTRY: dict[str, dict[str, Any]] = {
    **{f"source.{kind}": _product_source_node(kind) for kind in PRODUCT_SOURCES},
    "source.inline": _source_node(
        "source.inline",
        "手工输入时序",
        {
            "type": "object",
            "properties": {
                "rows": {"type": "array", "maxItems": 20000},
                "inline_rows": {"type": "array", "maxItems": 20000},
                "value_field": {"type": "string", "default": "value"},
                "frequency": {"type": "string", "default": "daily"},
                "availability_mode": {"enum": ["point_in_time", "latest"]},
                "name": {"type": "string"},
            },
            "additionalProperties": True,
        },
    ),
    "source.upload": _source_node(
        "source.upload",
        "上传时序",
        {
            "type": "object",
            "required": ["artifact_id", "checksum"],
            "properties": {
                "artifact_id": {"type": "string"},
                "checksum": {"type": "string"},
                "format": {"enum": ["parquet"], "default": "parquet"},
                "value_field": {"enum": ["value"], "default": "value"},
                "date_field": {"enum": ["observation_date"], "default": "observation_date"},
                "available_at_field": {"enum": ["available_at"], "default": "available_at"},
                "vintage_field": {"enum": ["vintage"], "default": "vintage"},
                "revision_field": {"enum": ["revision"], "default": "revision"},
                "frequency": {"type": "string", "default": "daily"},
                "availability_mode": {"enum": ["point_in_time", "latest"], "default": "point_in_time"},
                "name": {"type": "string"},
            },
            "additionalProperties": False,
        },
    ),
    "source.index": _source_node(
        "source.index",
        "指数行情",
        {
            "type": "object",
            "required": ["ts_code"],
            "properties": {
                "ts_code": {"type": "string"},
                "source_api": {"type": "string", "default": "index_daily"},
                "field": {"type": "string", "default": "close"},
                "frequency": {"type": "string", "default": "daily"},
                "start_date": {"type": "string", "format": "date"},
                "end_date": {"type": "string", "format": "date"},
                "name": {"type": "string"},
                "snapshot_id": {"type": "string"},
                "snapshot_generation": {"type": "string"},
                "source_file": {"type": "string"},
                "file_checksum": {"type": "string"},
            },
            "additionalProperties": True,
        },
    ),
    "source.indicator": _source_node(
        "source.indicator",
        "引用指标",
        {
            "type": "object",
            "required": ["indicator_id", "indicator_revision", "product_kind", "product_id", "period"],
            "properties": {
                "indicator_id": {"type": "string"},
                "indicator_revision": {"type": "integer", "minimum": 1},
                "product_kind": {"type": "string"},
                "product_id": {"type": "string"},
                "period": {"type": "string"},
                "start_date": {"type": "string", "format": "date"},
                "end_date": {"type": "string", "format": "date"},
                "name": {"type": "string"},
                "data_fingerprint": {"type": "string"},
                "indicator_data_snapshot": {"type": "object"},
            },
            "additionalProperties": True,
        },
    ),
    "source.relative": _source_node(
        "source.relative",
        "相对强弱序列",
        {
            "type": "object",
            "required": ["numerator", "denominator"],
            "properties": {
                "numerator": {"type": "object"},
                "denominator": {"type": "object"},
                "transform": {"enum": ["ratio", "log_ratio"], "default": "log_ratio"},
                "frequency": {"type": "string", "default": "daily"},
                "name": {"type": "string"},
            },
            "additionalProperties": True,
        },
        available=False,
        status="deprecated_use_explicit_graph",
        unavailable_reason="请使用两个数据源、显式对齐，并通过“相除”或“自然对数”节点构造相对强弱序列。",
    ),
    "source.macro": _source_node(
        "source.macro",
        "宏观时序",
        {
            "type": "object",
            "required": ["dataset", "field"],
            "properties": {
                "dataset": {"type": "string"},
                "series_id": {"type": "string"},
                "field": {"type": "string"},
                "code": {"type": "string"},
                "date_field": {"type": "string"},
                "available_at_field": {"type": "string"},
                "frequency": {"type": "string", "default": "monthly"},
                "start_date": {"type": "string", "format": "date"},
                "end_date": {"type": "string", "format": "date"},
                "name": {"type": "string"},
                "snapshot_id": {"type": "string"},
                "snapshot_generation": {"type": "string"},
                "source_file": {"type": "string"},
                "file_checksum": {"type": "string"},
            },
            "additionalProperties": True,
        },
    ),
    "source.constant": _numeric_node(
        "source.constant",
        "常量序列",
        "source",
        [_port("anchor", SERIES)],
        [_port("value", SERIES)],
        {
            "type": "object",
            "properties": {"value": {"type": "number", "default": 0.0}},
            "additionalProperties": False,
        },
    ),
    "align.strict_intersection": _numeric_node(
        "align.strict_intersection",
        "严格交集对齐",
        "alignment",
        [_port("left", SERIES), _port("right", SERIES)],
        [_port("left", SERIES), _port("right", SERIES)],
        _EMPTY_OBJECT,
    ),
    "align.pit_asof": _numeric_node(
        "align.pit_asof",
        "时点可得日对齐",
        "alignment",
        [_port("anchor", SERIES), _port("feature", SERIES)],
        [_port("anchor", SERIES), _port("feature", SERIES)],
        {
            "type": "object",
            "properties": {"max_age_days": {"type": "integer", "minimum": 0, "maximum": 3650, "default": 400}},
            "additionalProperties": False,
        },
    ),
    "align.resample": _numeric_node(
        "align.resample",
        "日历频率转换",
        "alignment",
        [_port("value", SERIES)],
        [_port("value", SERIES)],
        {
            "type": "object",
            "properties": {
                "frequency": {
                    "type": "string",
                    "enum": ["daily", "weekly", "monthly", "quarterly", "yearly"],
                    "default": "weekly",
                },
                "aggregation": {
                    "type": "string",
                    "enum": ["first", "last", "mean", "sum"],
                    "default": "last",
                },
                "every": {"type": "integer", "minimum": 1, "maximum": 1000},
                "offset": {"type": "integer", "minimum": 0, "maximum": 999, "default": 0},
            },
            "additionalProperties": False,
        },
    ),
    "align.cross_section": _numeric_node(
        "align.cross_section",
        "多序列严格交集矩阵",
        "alignment",
        [_port("feature_1", SERIES), _port("feature_2", SERIES), _port("feature_3", SERIES, required=False), _port("feature_4", SERIES, required=False)],
        [_port("features", MATRIX)],
        _EMPTY_OBJECT,
    ),
    "feature.matrix": _numeric_node(
        "feature.matrix",
        "特征矩阵",
        "feature",
        [_port("feature_1", SERIES), _port("feature_2", SERIES, required=False), _port("feature_3", SERIES, required=False), _port("feature_4", SERIES, required=False)],
        [_port("features", MATRIX)],
        _EMPTY_OBJECT,
    ),
    "feature.formula": _numeric_node(
        "feature.formula",
        "受限公式特征",
        "feature",
        [_port("feature_1", SERIES), _port("feature_2", SERIES, required=False), _port("feature_3", SERIES, required=False), _port("feature_4", SERIES, required=False)],
        [_port("value", SERIES)],
        {
            "type": "object",
            "required": ["expression"],
            "properties": {
                "expression": {"type": "string", "minLength": 1, "maxLength": 1000},
                "variables": {
                    "type": "object",
                    "description": "公式变量名到输入端口名的显式映射。",
                },
            },
            "additionalProperties": False,
        },
    ),
    "transform.identity": _numeric_node(
        "transform.identity", "原值", "transform", [_port("value", SERIES)], [_port("value", SERIES)], _EMPTY_OBJECT
    ),
    "transform.log": _numeric_node(
        "transform.log", "自然对数", "transform", [_port("value", SERIES)], [_port("value", SERIES)], _EMPTY_OBJECT
    ),
    "transform.diff": _numeric_node(
        "transform.diff", "差分", "transform", [_port("value", SERIES)], [_port("value", SERIES)], _WINDOW
    ),
    "transform.lag": _numeric_node(
        "transform.lag", "滞后值", "transform", [_port("value", SERIES)], [_port("value", SERIES)], _WINDOW
    ),
    "transform.return": _numeric_node(
        "transform.return", "收益率", "transform", [_port("value", SERIES)], [_port("value", SERIES)], _WINDOW
    ),
    "transform.drawdown": _numeric_node(
        "transform.drawdown", "回撤序列", "transform", [_port("value", SERIES)], [_port("value", SERIES)], _EMPTY_OBJECT
    ),
    "transform.yoy": _numeric_node(
        "transform.yoy", "同比变化", "transform", [_port("value", SERIES)], [_port("value", SERIES)], {
            "type": "object", "properties": {"periods": {"type": "integer", "minimum": 1, "maximum": 120, "default": 12}}, "additionalProperties": False
        }
    ),
    "transform.mom": _numeric_node(
        "transform.mom", "环比变化", "transform", [_port("value", SERIES)], [_port("value", SERIES)], {
            "type": "object", "properties": {"periods": {"type": "integer", "minimum": 1, "maximum": 120, "default": 1}}, "additionalProperties": False
        }
    ),
    "transform.standardize": _numeric_node(
        "transform.standardize", "滚动标准化", "transform", [_port("value", SERIES)], [_port("value", SERIES)], _WINDOW
    ),
    "transform.clip": _numeric_node(
        "transform.clip", "截尾", "transform", [_port("value", SERIES)], [_port("value", SERIES)], {
            "type": "object", "properties": {"lower": {"type": "number", "default": -3.0}, "upper": {"type": "number", "default": 3.0}}, "additionalProperties": False
        }
    ),
    "math.add": _numeric_node(
        "math.add", "相加", "arithmetic", [_port("left", SERIES), _port("right", SERIES)], [_port("value", SERIES)], _EMPTY_OBJECT
    ),
    "math.subtract": _numeric_node(
        "math.subtract", "相减", "arithmetic", [_port("left", SERIES), _port("right", SERIES)], [_port("value", SERIES)], _EMPTY_OBJECT
    ),
    "math.multiply": _numeric_node(
        "math.multiply", "相乘", "arithmetic", [_port("left", SERIES), _port("right", SERIES)], [_port("value", SERIES)], _EMPTY_OBJECT
    ),
    "math.divide": _numeric_node(
        "math.divide", "相除", "arithmetic", [_port("left", SERIES), _port("right", SERIES)], [_port("value", SERIES)], _EMPTY_OBJECT
    ),
    "filter.ema": _numeric_node(
        "filter.ema", "单边 EMA", "filter", [_port("value", SERIES)], [_port("value", SERIES)], _WINDOW
    ),
    "filter.sma": _numeric_node(
        "filter.sma", "单边 SMA", "filter", [_port("value", SERIES)], [_port("value", SERIES)], _WINDOW
    ),
    "filter.kalman": _numeric_node(
        "filter.kalman", "单边卡尔曼滤波", "filter", [_port("value", SERIES)], [_port("value", SERIES)], {
            "type": "object", "properties": {"process_variance": {"type": "number", "minimum": 1e-12, "default": 1e-5}, "measurement_variance": {"type": "number", "minimum": 1e-12, "default": 1e-2}}, "additionalProperties": False
        }
    ),
    "rolling.mean": _numeric_node(
        "rolling.mean", "滚动均值", "rolling", [_port("value", SERIES)], [_port("value", SERIES)], _WINDOW
    ),
    "rolling.std": _numeric_node(
        "rolling.std", "滚动波动率", "rolling", [_port("value", SERIES)], [_port("value", SERIES)], _WINDOW
    ),
    "rolling.zscore": _numeric_node(
        "rolling.zscore", "滚动 Z 分数", "rolling", [_port("value", SERIES)], [_port("value", SERIES)], _WINDOW
    ),
    "rolling.slope": _numeric_node(
        "rolling.slope", "滚动斜率", "rolling", [_port("value", SERIES)], [_port("value", SERIES)], _WINDOW
    ),
    "rolling.min": _numeric_node(
        "rolling.min", "滚动最小值", "rolling", [_port("value", SERIES)], [_port("value", SERIES)], _WINDOW
    ),
    "rolling.max": _numeric_node(
        "rolling.max", "滚动最大值", "rolling", [_port("value", SERIES)], [_port("value", SERIES)], _WINDOW
    ),
    "model.threshold": _numeric_node(
        "model.threshold",
        "三状态阈值",
        "model",
        [_port("value", SERIES)],
        [_port("state", STATE_CODES), _port("score", SERIES), _port("confidence", CONFIDENCE), _port("probabilities", PROBABILITIES), _port("recognition_index", INDEX_SERIES), _port("reason_code", REASON_CODES)],
        {
            "type": "object",
            "properties": {
                "upper": {"type": "number", "default": 0.001},
                "lower": {"type": "number", "default": -0.001},
            },
            "additionalProperties": False,
        },
    ),
    "model.hysteresis": _numeric_node(
        "model.hysteresis",
        "滞回三状态",
        "model",
        [_port("value", SERIES)],
        [_port("state", STATE_CODES), _port("score", SERIES), _port("confidence", CONFIDENCE), _port("probabilities", PROBABILITIES)],
        {
            "type": "object",
            "properties": {
                "upper_enter": {"type": "number", "default": 0.0015},
                "upper_exit": {"type": "number", "default": 0.0002},
                "lower_enter": {"type": "number", "default": -0.0015},
                "lower_exit": {"type": "number", "default": -0.0002},
            },
            "additionalProperties": False,
        },
    ),
    "post.hysteresis": _numeric_node(
        "post.hysteresis",
        "分数滞回映射",
        "postprocess",
        [_port("score", SERIES)],
        [
            _port("state", STATE_CODES),
            _port("score", SERIES),
            _port("confidence", CONFIDENCE),
            _port("probabilities", PROBABILITIES),
        ],
        {
            "type": "object",
            "properties": {
                "upper_enter": {"type": "number", "default": 0.0015},
                "upper_exit": {"type": "number", "default": 0.0002},
                "lower_enter": {"type": "number", "default": -0.0015},
                "lower_exit": {"type": "number", "default": -0.0002},
            },
            "additionalProperties": False,
        },
    ),
    "model.quadrant": _numeric_node(
        "model.quadrant",
        "双指标四象限",
        "model",
        [_port("growth", SERIES), _port("inflation", SERIES)],
        [_port("state", STATE_CODES), _port("score", SERIES), _port("confidence", CONFIDENCE), _port("probabilities", PROBABILITIES)],
        {
            "type": "object",
            "properties": {
                "growth_threshold": {"type": "number", "default": 0.0},
                "inflation_threshold": {"type": "number", "default": 0.0},
            },
            "additionalProperties": False,
        },
    ),
    "model.turning_point": _numeric_node(
        "model.turning_point",
        "峰谷区间",
        "model",
        [_port("value", SERIES)],
        [_port("state", STATE_CODES), _port("score", SERIES), _port("confidence", CONFIDENCE), _port("probabilities", PROBABILITIES), _port("recognition_index", INDEX_SERIES), _port("reason_code", REASON_CODES)],
        {"type": "object", "properties": {"window": {"type": "integer", "minimum": 2, "maximum": 1000, "default": 20}, "min_move": {"type": "number", "minimum": 0.0, "default": 0.08}}, "additionalProperties": False},
        causal=False,
        phase="P1",
    ),
    "model.hmm": _numeric_node(
        "model.hmm", "高斯隐马尔可夫", "model", [_port("features", MATRIX)], [_port("state", STATE_CODES), _port("score", SERIES), _port("confidence", CONFIDENCE), _port("probabilities", PROBABILITIES)],
        _latent_model_parameter_schema(), phase="P1"
    ),
    "model.markov": _numeric_node(
        "model.markov", "Markov 状态切换", "model", [_port("features", MATRIX)], [_port("state", STATE_CODES), _port("score", SERIES), _port("confidence", CONFIDENCE), _port("probabilities", PROBABILITIES)],
        _latent_model_parameter_schema(), phase="P1"
    ),
    "model.gmm": _numeric_node(
        "model.gmm", "高斯混合聚类", "model", [_port("features", MATRIX)], [_port("state", STATE_CODES), _port("score", SERIES), _port("confidence", CONFIDENCE), _port("probabilities", PROBABILITIES)],
        _latent_model_parameter_schema(), phase="P1"
    ),
    "model.change_point": _numeric_node(
        "model.change_point", "结构突变", "model", [_port("value", SERIES)], [_port("state", STATE_CODES), _port("score", SERIES), _port("confidence", CONFIDENCE), _port("probabilities", PROBABILITIES), _port("recognition_index", INDEX_SERIES), _port("reason_code", REASON_CODES)],
        {"type": "object", "properties": {"window": {"type": "integer", "minimum": 4, "maximum": 1000, "default": 20}, "threshold": {"type": "number", "minimum": 0.0, "default": 1.5}, "confirmation": {"type": "integer", "minimum": 1, "maximum": 252, "default": 2}}, "additionalProperties": False}, phase="P1"
    ),
    "model.ensemble": _numeric_node(
        "model.ensemble", "加权共识", "model", [_port("state_1", STATE_CODES), _port("state_2", STATE_CODES), _port("state_3", STATE_CODES, required=False), _port("state_4", STATE_CODES, required=False)], [_port("state", STATE_CODES), _port("confidence", CONFIDENCE), _port("probabilities", PROBABILITIES)],
        {"type": "object", "properties": {"weights": {"type": "array", "minItems": 2, "maxItems": 4, "items": {"type": "number", "minimum": 0.0}, "default": [0.5, 0.5]}, "consensus_threshold": {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.6}}, "additionalProperties": False}, phase="P1"
    ),
    "post.confirmation": _numeric_node(
        "post.confirmation",
        "确认期与最短持续期",
        "postprocess",
        [_port("state", STATE_CODES)],
        [_port("state", STATE_CODES)],
        {
            "type": "object",
            "properties": {
                "confirmation": {"type": "integer", "minimum": 1, "maximum": 252, "default": 2},
                "min_duration": {"type": "integer", "minimum": 1, "maximum": 5000, "default": 1},
            },
            "additionalProperties": False,
        },
    ),
    "post.min_duration": _numeric_node(
        "post.min_duration", "最短持续期", "postprocess", [_port("state", STATE_CODES)], [_port("state", STATE_CODES)], {
            "type": "object", "properties": {"min_duration": {"type": "integer", "minimum": 1, "maximum": 5000, "default": 5}}, "additionalProperties": False
        }
    ),
    "post.component_map": _numeric_node(
        "post.component_map", "状态映射", "postprocess", [_port("state", STATE_CODES)], [_port("state", STATE_CODES)], {
            "type": "object", "required": ["mapping"], "properties": {"mapping": {"type": "array", "minItems": 1, "maxItems": 12, "items": {"type": "integer", "minimum": -1, "maximum": 11}}}, "additionalProperties": False
        }
    ),
    "post.priority": _numeric_node(
        "post.priority", "优先级合并", "postprocess", [_port("primary", STATE_CODES), _port("secondary", STATE_CODES)], [_port("state", STATE_CODES)], _EMPTY_OBJECT
    ),
    "post.conflict_reject": _numeric_node(
        "post.conflict_reject", "冲突拒识", "postprocess", [_port("left", STATE_CODES), _port("right", STATE_CODES)], [_port("state", STATE_CODES)], _EMPTY_OBJECT
    ),
    "post.confidence_gate": _numeric_node(
        "post.confidence_gate", "置信度门槛", "postprocess", [_port("state", STATE_CODES), _port("confidence", CONFIDENCE)], [_port("state", STATE_CODES)], {
            "type": "object", "properties": {"floor": {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.5}}, "additionalProperties": False
        }
    ),
    "output.state": _numeric_node(
        "output.state", "状态输出", "output", [_port("state", STATE_CODES)], [_port("state", STATE_CODES)], _EMPTY_OBJECT
    ),
    "output.probabilities": _numeric_node(
        "output.probabilities", "概率输出", "output", [_port("probabilities", PROBABILITIES)], [_port("probabilities", PROBABILITIES)], _EMPTY_OBJECT
    ),
    "output.confidence": _numeric_node(
        "output.confidence", "置信度输出", "output", [_port("confidence", CONFIDENCE)], [_port("confidence", CONFIDENCE)], _EMPTY_OBJECT
    ),
    "output.temporal": _numeric_node(
        "output.temporal", "识别与生效索引", "output", [_port("state", STATE_CODES)], [_port("recognition_index", INDEX_SERIES), _port("effective_index", INDEX_SERIES), _port("reason_code", REASON_CODES)], _EMPTY_OBJECT
    ),
    "model.external_optimized": {
        "id": "model.external_optimized",
        "type_id": "model.external_optimized",
        "version": 1,
        "type_version": 1,
        "label": "第三方优化模型",
        "available": False,
        "status": "admin_adapter_required",
        "unavailable_reason": "当前未安装管理员批准的隔离模型适配器。",
        "category": "model",
        "phase": "P1",
        "causal": True,
        "repaints": False,
        "supports_realtime": True,
        "minimum_samples": 5,
        "cost_estimate": {"class": "adapter_declared", "expression": "adapter-specific", "unit": "observations"},
        "kernel_id": None,
        "kernel_version": None,
        "model_version": "adapter-declared",
        "formula_language": None,
        "inputs": [_port("features", SERIES)],
        "outputs": [_port("state", STATE_CODES), _port("confidence", CONFIDENCE), _port("probabilities", PROBABILITIES)],
        "parameter_schema": {
            "type": "object",
            "required": ["declaration", "adapter_id"],
            "properties": {"declaration": {"type": "object"}, "adapter_id": {"type": "string"}},
            "additionalProperties": False,
        },
        "njit_policy": {
            "execution_backend": THIRD_PARTY_BACKEND,
            "njit_required": False,
            "execution_lane": "isolated_model_train_or_infer",
            "feature_pipeline_backend": NJIT_BACKEND,
            "postprocess_backend": NJIT_BACKEND,
            "python_callback": False,
            "python_fallback": 0,
        },
    },
}


def _trend_parameter(kind: str, default: int | float, minimum: int | float,
                     maximum: int | float, title: str, description: str) -> dict[str, Any]:
    return {"type": kind, "default": default, "minimum": minimum, "maximum": maximum,
            "title": title, "description": description}


def _trend_node(node_id: str, label: str, category: str, inputs: list[str],
                outputs: list[dict[str, Any]], properties: dict[str, Any],
                description: str, kernel_id: str, *, causal: bool = True) -> dict[str, Any]:
    item = _numeric_node(node_id, label, category,
                         [_port(name, STATE_CODES if name == "state" else SERIES) for name in inputs],
                         outputs, {"type": "object", "properties": properties, "additionalProperties": False},
                         causal=causal)
    item.update(description=description, kernel_id=kernel_id)
    if kernel_id in {"kama", "trend_features"}:
        item["cost_estimate"] = {"class": "windowed", "expression": "O(T × window)", "unit": "observations"}
    if kernel_id == "peak_trough":
        item["cost_estimate"] = {"class": "windowed", "expression": "O(T × window + T × turns)", "unit": "observations"}
    return item


NODE_REGISTRY.update({
    "model.peak_trough": _trend_node(
        "model.peak_trough", "峰谷定界法（PS · 事后）", "model", ["value"],
        [_port("state", STATE_CODES), _port("pivot", SERIES), _port("phase_start_index", INDEX_SERIES),
         _port("phase_end_index", INDEX_SERIES), _port("phase_return", SERIES), _port("boundary_line", SERIES),
         _port("sideways_range", SERIES), _port("sideways_efficiency", SERIES),
         _port("sideways_start_index", INDEX_SERIES), _port("sideways_end_index", INDEX_SERIES),
         _port("sideways_swing_count", INDEX_SERIES)],
        {"left_window": _trend_parameter("integer", 8, 1, 5000, "左窗口", "候选峰谷与此前这些观测比较；独立控制向前寻找范围。"),
         "right_window": _trend_parameter("integer", 8, 1, 5000, "右窗口", "候选峰谷还需与之后这些观测比较；减小可较早确认，但可能增加噪声，仍属于事后识别。"),
         "head_window": _trend_parameter("integer", 6, 0, 5000, "首窗口", "距连续有效样本开头不足这些观测的拐点排除；仍需满足左窗口。"),
         "tail_window": _trend_parameter("integer", 6, 0, 5000, "尾窗口", "距连续有效样本末尾不足这些观测的拐点排除；仍需满足右窗口。缩短不会强行填充未完成阶段。"),
         "window": {**_trend_parameter("integer", 8, 1, 5000, "峰谷左右窗口", "旧方案兼容项；仅当左窗口或右窗口未指定时作为对应默认值。"), "deprecated": True},
         "min_phase": _trend_parameter("integer", 4, 1, 10000, "最短牛熊阶段", "相邻峰谷至少相隔多少期；增大可过滤更短波段。大幅波动例外只放宽这项约束。"),
         "min_cycle": _trend_parameter("integer", 16, 2, 20000, "最短完整周期", "峰到峰或谷到谷至少相隔多少期；增大可保留更长周期。"),
         "endpoint_window": {**_trend_parameter("integer", 6, 0, 5000, "首尾排除窗口", "旧方案兼容项；仅当首窗口或尾窗口未指定时作为对应默认值。"), "deprecated": True},
         "amplitude_exception": _trend_parameter("number", 0.2, 0.0, 10.0, "大幅波动例外", "例如 0.2 为阶段涨跌幅绝对值严格超过 20% 时允许短阶段；不豁免最短完整周期。"),
         "sideways_enabled": {"type": "boolean", "default": False, "title": "启用震荡识别", "description": "合并至少两个小幅反向波段；需配置牛市、震荡、熊市三个状态。旧两态方案默认关闭。"},
         "small_swing_threshold": _trend_parameter("number", 0.03, 0.0001, 1.0, "小波段幅度门槛", "每个候选峰谷段的涨跌幅绝对值上限，0.03 表示 3%；大波段保留趋势。"),
         "sideways_max_range": _trend_parameter("number", 0.06, 0.0001, 1.0, "震荡最大振幅", "整段原始价格最高值 / 最低值 - 1 的上限，0.06 表示 6%；限制累计漂移和宽幅急跌反弹。"),
         "sideways_max_efficiency": _trend_parameter("number", 0.25, 0.0, 0.99, "震荡方向效率上限", "净价格变化绝对值 / 累计逐期价格变化绝对值；越低，要求越缺少单边方向。"),
         "sideways_min_duration": _trend_parameter("integer", 20, 2, 20000, "最短震荡长度", "以输入观测期数计；日频模板为 20，月频模板为 3。修改频率时须同时检查此参数。")},
        "PS 思路找峰谷并筛选完整牛熊区间；可按小波段幅度、整段振幅、方向效率和持续长度合并震荡。均使用原始价格，不跨缺失，不外推未完成尾部。事后分类不产生实时交易信号；长度单位为输入观测期数。",
        "peak_trough", causal=False),
    "filter.super_smoother": _trend_node(
        "filter.super_smoother", "Super Smoother 趋势滤波", "filter", ["value"], [_port("value", SERIES)],
        {"period": _trend_parameter("integer", 126, 3, 5000, "滤波周期", "控制低通响应；不是均线天数或牛熊区间长度。连续有效样本达到该周期后输出；缺失后重新预热。")},
        "二阶单向递推，压制短周期波动；默认中长期研究参数尚需样本外验证。", "super_smoother"),
    "filter.kama": _trend_node(
        "filter.kama", "KAMA 自适应趋势滤波", "filter", ["value"], [_port("value", SERIES)],
        {"window": _trend_parameter("integer", 60, 2, 5000, "效率计算期数", "最近净位移与累计绝对位移的比较窗口；完成窗口后才输出。"),
         "fast": _trend_parameter("integer", 2, 1, 5000, "快速响应期数", "趋势明确时的响应上限；必须小于慢速期数。"),
         "slow": _trend_parameter("integer", 126, 2, 5000, "慢速响应期数", "方向不明确时减慢响应；不能与固定滤波周期直接等同。")},
        "按方向效率调整增益的单向滤波；首值初始化，缺失后重新预热。", "kama"),
    "feature.trend_metrics": _trend_node(
        "feature.trend_metrics", "趋势偏离、斜率与方向效率", "feature", ["log_price", "trend"],
        [_port(name, SERIES) for name in ("distance", "slope", "efficiency", "scale", "drawdown", "risk", "index_value", "filtered_index")],
        {"volatility_window": _trend_parameter("integer", 60, 2, 5000, "波动尺度期数", "用上一期收益均方根缩放当期信号，避免急跌同时扩大自身门槛。"),
         "slope_window": _trend_parameter("integer", 20, 1, 5000, "趋势斜率跨度", "滤波线相隔这些观测的变化，除以跨度与上一期收益尺度。"),
         "efficiency_window": _trend_parameter("integer", 60, 2, 5000, "方向效率期数", "净位移除以累计绝对位移；0 表示无净方向，1 表示单向运动。"),
         "scale_floor": _trend_parameter("number", 0.0001, 1e-12, 1.0, "最小收益尺度", "平价或极低波动时的分母下限，不是缺失值替代。"),
         "shock_window": _trend_parameter("integer", 5, 1, 5000, "急跌观察期数", "原始指数短窗累计下跌观察跨度。"),
         "drawdown_alert": _trend_parameter("number", 0.2, 0.0001, 0.99, "回撤警报幅度", "例如 0.2 为从连续有效数据运行峰值回撤 20%；只标记风险，不强制改写主状态。"),
         "shock_alert": _trend_parameter("number", 0.08, 0.0001, 0.99, "急跌警报幅度", "例如 0.08 为短窗下跌 8%；风险标记独立于去噪。")},
        "输入对数价格及同轴滤波线，计算无量纲趋势证据与原始价格风险；缺失返回未知。", "trend_features"),
    "model.trend_regime": _trend_node(
        "model.trend_regime", "滤波牛熊震荡识别", "model", ["distance", "slope", "efficiency"],
        [_port("state", STATE_CODES), _port("candidate", STATE_CODES), _port("pending_count", SERIES), _port("phase", SERIES)],
        {"band": _trend_parameter("number", 1.0, 1e-8, 100.0, "价格偏离门槛", "价格偏离滤波线达到多少倍历史收益尺度，才产生牛熊候选。"),
         "trend_enter": _trend_parameter("number", 0.1, 1e-8, 100.0, "趋势进入门槛", "牛熊候选要求同方向标准化斜率达到此值；必须高于震荡斜率上限。"),
         "flat_threshold": _trend_parameter("number", 0.05, 0.0, 100.0, "震荡斜率上限", "绝对斜率低于此值且方向效率较低，才产生震荡候选。"),
         "efficiency_ceiling": _trend_parameter("number", 0.25, 0.0, 1.0, "震荡效率上限", "震荡必须同时满足低方向效率与平坦慢趋势。"),
         "confirmation": _trend_parameter("integer", 3, 1, 252, "连续确认期数", "同一新候选连续出现后，从确认日切换；未命中、回到旧状态或缺失均重置计数。")},
        "牛熊由幅度与斜率共同确认；震荡单独判断。门槛之间保留旧状态，回调/反弹不自动反转；不回填历史。阶段编码：0 上行、1 牛市回调、2 下行、3 熊市反弹、4 震荡。", "trend_regime"),
    "post.merge_short_regimes": _trend_node(
        "post.merge_short_regimes", "事后短反向区间合并", "postprocess", ["state", "price"], [_port("state", STATE_CODES)],
        {"max_duration": _trend_parameter("integer", 10, 1, 5000, "可合并最长反向期数", "只检查已结束、两侧同为牛或同为熊的反向区间。"),
         "max_move": _trend_parameter("number", 0.08, 0.0001, 0.99, "可合并幅度上限", "含前一主阶段峰谷的逆向幅度必须小于此值；保留暴跌与强反弹。"),
         "following_confirmation": _trend_parameter("integer", 3, 1, 252, "后续主状态确认期数", "必须已有足够的后续同向观测；使用未来信息，实时模式禁用。")},
        "仅事后研究：按原始区间从左到右单次合并短且浅的反向段，不跨未知区间，不合并未结束尾段。", "merge_short_regimes", causal=False),
})
PORT_LABELS.update({"log_price": "对数指数", "trend": "滤波趋势线", "distance": "标准化价格偏离",
                    "pivot": "保留拐点（1 峰，-1 谷）", "phase_start_index": "阶段起始位置",
                    "phase_end_index": "阶段结束位置", "phase_return": "完整阶段涨跌幅", "boundary_line": "峰谷连线",
                    "sideways_range": "震荡整段振幅", "sideways_efficiency": "震荡方向效率",
                    "sideways_start_index": "震荡起点", "sideways_end_index": "震荡终点", "sideways_swing_count": "合并波段数",
                    "index_value": "识别指数点位", "filtered_index": "滤波后指数点位",
                    "slope": "标准化趋势斜率", "efficiency": "方向效率", "scale": "历史收益尺度",
                    "drawdown": "原始指数回撤", "risk": "风险警报（1 为触发）", "price": "原始指数价格",
                    "candidate": "当期候选状态", "pending_count": "待确认连续期数", "phase": "行情内部阶段编码"})


"""Regime-only node definitions; independent of the Indicator Center registry."""
PIVOTS = "pivots<time>"
SEGMENT_START = "segment_start<time>"
SEGMENT_END = "segment_end<time>"
STATISTIC_IDS = {"segment.change": 0, "segment.amplitude": 1, "segment.volatility": 2,
                 "segment.duration": 3, "segment.efficiency": 4}


def register_segment_nodes(registry, numeric_node, port, parameter):
    series = "series<float64>"
    windows = {name: parameter("integer", default, minimum, 5000, label, description)
               for name, default, minimum, label, description in [
                   ("left_window", 8, 1, "左窗口", "与此前这些观测比较；平台取最早极值。"),
                   ("right_window", 8, 1, "右窗口", "与之后这些观测比较；因此仅支持事后研究。"),
                   ("head_window", 6, 0, "首窗口", "连续有效样本开头排除的观测数。"),
                   ("tail_window", 6, 0, "尾窗口", "连续有效样本末尾排除的观测数，不外推尾段。"),
               ]}

    def add(identifier, label, inputs, outputs, params, kernel, description, causal=False):
        node = numeric_node(identifier, label, "model" if identifier.startswith("model.") else "feature",
                            inputs, outputs, {"type": "object", "properties": params, "additionalProperties": False},
                            causal=causal)
        node.update(kernel_id=kernel, description=description)
        registry[identifier] = node
        return node

    detect = add("pivot.local_extrema", "峰谷定位", [port("value", series)],
                 [port("pivot", PIVOTS), port("pivot_price", series)], windows, "local_extrema",
                 "只定位交替峰谷，不计算区间涨跌幅或市场状态。四窗口以输入观测数计；缺失断开，尾部不外推。")
    detect["knowledge_scope"] = "full_input"
    detect["cost_estimate"] = {"class": "rolling", "expression": "O(T × (left + right))", "unit": "observations"}
    add("segment.between_pivots", "相邻峰谷分段", [port("pivot", PIVOTS)],
        [port("start", SEGMENT_START), port("end", SEGMENT_END)], {}, "between_pivots",
        "只输出完整相邻峰谷的边界；起点包含、终点不含。每段边界按原日期轴展开，未知与尾段不填充。")
    descriptions = [
        ("segment.change", "区间涨跌幅", "结束价格 / 起始价格 − 1；单个完整波段即可计算。"),
        ("segment.amplitude", "区间振幅", "含首尾点的最高价格 / 最低价格 − 1，与首尾净变化分开。"),
        ("segment.volatility", "区间收益波动率", "区间内逐期简单收益的标准差；默认样本标准差，不年化，不对价格水平求标准差。"),
        ("segment.duration", "区间长度", "结束索引减起始索引，按输入观测间隔计。"),
        ("segment.efficiency", "区间方向效率", "首尾价格差绝对值 / 累计逐期绝对价格变化；平价区间为 0。"),
    ]
    for identifier, label, description in descriptions:
        params = {"ddof": {"type": "integer", "enum": [0, 1], "enum_labels": ["总体标准差", "样本标准差"],
                           "default": 1, "title": "标准差口径"}} if identifier == "segment.volatility" else {}
        add(identifier, label, [port("value", series), port("start", SEGMENT_START), port("end", SEGMENT_END)],
            [port("value", series)], params, "interval_statistic", description)
    add("model.range_threshold", "区间阈值分类", [port("value", series), port("upper_bound", series, required=False),
        port("lower_bound", series, required=False)], [port("state", "state_codes<int64>")],
        {"upper": parameter("number", .03, -100, 100, "上涨门槛", "严格高于此值为正向；连接上界输入时使用输入值。"),
         "lower": parameter("number", -.03, -100, 100, "下跌门槛", "严格低于此值为负向；连接下界输入时使用输入值。")},
        "range_threshold", "上界之上为牛市，下界之下为熊市，两界之间（含等号）为震荡；缺失仍未分类。支持独立常量输入。", causal=True)


register_segment_nodes(NODE_REGISTRY, _numeric_node, _port, _trend_parameter)
register_series_operators(NODE_REGISTRY, _numeric_node, _port)
NODE_REGISTRY["model.peak_trough"]["knowledge_scope"] = "full_input"
PORT_TYPE_LABELS.update({PIVOTS: "峰谷事件序列", SEGMENT_START: "完整区间起点", SEGMENT_END: "完整区间终点"})
PORT_LABELS.update({"start": "区间起点", "end": "区间终点", "pivot_price": "拐点价格",
                    "upper_bound": "上涨门槛输入", "lower_bound": "下跌门槛输入"})


def _decorate_parameter_schema(node: dict[str, Any]) -> None:
    properties = node.get("parameter_schema", {}).get("properties", {})
    for name, schema in properties.items():
        if not isinstance(schema, dict):
            continue
        if name == "frequency" and "enum" not in schema:
            schema["enum"] = ["daily", "weekly", "monthly", "quarterly", "yearly"]
        schema.setdefault("title", PARAMETER_LABELS.get(name, "节点参数"))
        schema.setdefault(
            "description",
            PARAMETER_DESCRIPTIONS.get(
                name,
                f"用于配置“{node.get('label', '当前节点')}”的{PARAMETER_LABELS.get(name, '计算参数')}。",
            ),
        )
        options = schema.get("enum")
        if isinstance(options, list):
            schema.setdefault(
                "enum_labels",
                [ENUM_LABELS.get(str(option), "可选值") for option in options],
            )

    if node.get("id") == "source.index":
        properties["source_api"].update(
            {
                "enum": INDEX_SOURCE_OPTIONS,
                "enum_labels": [ENUM_LABELS[item] for item in INDEX_SOURCE_OPTIONS],
            }
        )
        properties["field"].update(
            {
                "enum": INDEX_FIELD_OPTIONS,
                "enum_labels": [ENUM_LABELS[item] for item in INDEX_FIELD_OPTIONS],
                "option_source": "research_series.fields",
            }
        )
    elif node.get("id") == "source.macro":
        properties["field"]["option_source"] = "research_series.fields"


def _catalog_item(node: dict[str, Any]) -> dict[str, Any]:
    item = copy.deepcopy(node)
    item.pop("_indicator_definition", None)
    if item["id"] in {"source.inline", "source.indicator", "source.relative"}:
        item["authoring_hidden"] = True
    category = str(item.get("category") or "")
    item["category_label"] = CATEGORY_LABELS.get(category, "计算节点")
    item.setdefault(
        "description",
        f"{item.get('label', '该节点')}用于{CATEGORY_LABELS.get(category, '情景识别计算')}。",
    )
    for port_group in ("inputs", "outputs"):
        for port in item.get(port_group, []):
            name = str(port.get("name") or "")
            value_type = str(port.get("type") or "")
            port.setdefault("label", PORT_LABELS.get(name, {"values": "输入序列", "lhs": "左侧输入", "rhs": "右侧输入"}.get(name, name)))
            port.setdefault("description", (
                f"{PORT_LABELS.get(name, '该端口')}，数据结构为"
                f"{PORT_TYPE_LABELS.get(value_type, '受类型系统约束的数据')}。"
            ))
            port["type_label"] = PORT_TYPE_LABELS.get(value_type, "受类型系统约束的数据")
    source_descriptions = {
        "source.inline": "直接录入日期和数值，作为计算输入。数据随当前计算定义保存。",
        "source.upload": "上传 CSV、Excel（.xlsx）或 JSON，或选择已上传的数据。确认列后保存为固定版本，可重复使用。",
        "source.etf": "按名称或代码选择 ETF，读取交易价格、成交量等字段。默认收盘价（不复权）。",
        "source.fund": "按名称或代码选择公募基金，读取单位净值、累计净值等字段。实时分析按公告日期使用数据。",
        "source.index": "按名称或代码选择指数，再选择所需数值字段；代码与行情来源自动绑定。",
        "source.macro": "选择宏观数据序列和数值字段；数据集、序列代码与来源自动绑定。",
        "source.indicator": "选择一个指标版本及基金或 ETF，生成该对象的指标历史时序。指标版本和计算对象各选一次。",
        "source.constant": "在上游序列的每个日期输出同一个数值。基准时间轴只决定日期，不改变常量值。",
    }
    if item.get("id") in source_descriptions:
        item["description"] = source_descriptions[item["id"]]
    if item.get("id") in {"source.inline", "source.upload"}:
        item["parameter_schema"]["properties"]["frequency"]["enum"] = ["daily", "weekly", "monthly", "quarterly", "annual", "irregular"]
    _decorate_parameter_schema(item)
    return item


def node_catalog() -> dict[str, Any]:
    return {
        "schema_version": "2.0",
        "registry_version": REGISTRY_VERSION,
        "port_types": [
            PIVOTS, SEGMENT_START, SEGMENT_END,
            SERIES,
            BOOL_SERIES,
            MATRIX,
            STATE_CODES,
            REGIME_CANDIDATE,
            REGIME_OUTPUT,
            PROBABILITIES,
            CONFIDENCE,
            INDEX_SERIES,
            REASON_CODES,
        ],
        "port_type_labels": copy.deepcopy(PORT_TYPE_LABELS),
        "items": [_catalog_item(item) for item in NODE_REGISTRY.values()],
        "limits": {"max_nodes": 128, "max_edges": 256, "max_observations": 20000},
        "execution_policy": {
            "ordinary_numeric": NJIT_BACKEND,
            "request_time_compilation": 0,
            "python_fallback": 0,
            "third_party_model_lane": THIRD_PARTY_BACKEND,
        },
    }


__all__ = [
    "BOOL_SERIES",
    "CONFIDENCE",
    "MATRIX",
    "INDEX_SERIES",
    "NODE_REGISTRY",
    "PROBABILITIES",
    "REGIME_CANDIDATE",
    "REGIME_OUTPUT",
    "REASON_CODES",
    "REGISTRY_VERSION",
    "SERIES",
    "STATE_CODES",
    "node_catalog",
]

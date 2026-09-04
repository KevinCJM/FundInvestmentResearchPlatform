"""Stable node registry for user-authored historical-regime graphs."""

from __future__ import annotations

import copy
from typing import Any

from compute_policy import NJIT_BACKEND, THIRD_PARTY_BACKEND


REGISTRY_VERSION = "regime-graph-nodes/2.2.0"

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
        "kernel_version": "typed-njit-v2.1" if node_id == "feature.formula" else "regime-graph-kernels/2.2.0",
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


NODE_REGISTRY: dict[str, dict[str, Any]] = {
    "source.inline": _source_node(
        "source.inline",
        "内联时序",
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
        "不可变上传时序",
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
        "指标中心版本",
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
            port["label"] = PORT_LABELS.get(name, "数据端口")
            port["description"] = (
                f"{PORT_LABELS.get(name, '该端口')}，数据结构为"
                f"{PORT_TYPE_LABELS.get(value_type, '受类型系统约束的数据')}。"
            )
            port["type_label"] = PORT_TYPE_LABELS.get(value_type, "受类型系统约束的数据")
    _decorate_parameter_schema(item)
    return item


def node_catalog() -> dict[str, Any]:
    return {
        "schema_version": "2.0",
        "registry_version": REGISTRY_VERSION,
        "port_types": [
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

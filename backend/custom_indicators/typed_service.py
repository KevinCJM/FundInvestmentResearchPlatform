"""Product-facing catalog, templates and safe composition for typed DSL v2."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Callable, Literal, Mapping, Optional

from cal_indicators.typed_dsl import (
    TypedDslError,
    get_typed_dsl_catalog,
    infer_typed_expression,
)
from cal_indicators.typed_latex import (
    MATH_NOTATION_VERSION,
    render_python_expression_latex,
)
from cal_indicators.typed_operators import (
    COMPAT_OPERATOR_REGISTRY_VERSION,
    COMPAT_TYPED_DSL_VERSION,
    LEGACY_OPERATOR_REGISTRY_VERSION,
    LEGACY_TYPED_DSL_VERSION,
    PREVIOUS_OPERATOR_REGISTRY_VERSION,
    PREVIOUS_TYPED_DSL_VERSION,
    TYPED_DSL_VERSION,
    TYPED_OPERATOR_REGISTRY_VERSION,
    get_typed_operator_registry,
)
from cal_indicators.typed_numba_plan import NumbaPlanCompileError, compile_numba_plan

from .errors import ValidationError
from .series_definitions import normalize_parameter_schema, parameter_variable_types
from .variable_registry import (
    CONTEXT_SCHEMA_VERSION,
    DATA_CONTRACT_VERSION,
    VARIABLE_REGISTRY_VERSION,
    allowed_variables,
    canonical_variable_id,
    get_variable,
    normalize_variable_latex,
    variable_catalog,
    variable_latex_symbols,
    variable_semantic_role,
    variable_types,
)


ContextKind = Literal["single_product", "portfolio"]


VARIABLE_METADATA: dict[str, dict[str, Any]] = {
    "returns": {
        "label": "普通收益率序列",
        "semantic_role": "ordinary_return",
        "source": "真实复权净值",
        "domains": ["single_product"],
    },
    "log_returns": {
        "label": "对数收益率序列",
        "semantic_role": "log_return",
        "source": "真实复权净值",
        "domains": ["single_product"],
    },
    "asset_returns": {
        "label": "多资产普通收益矩阵",
        "semantic_role": "ordinary_return_matrix",
        "source": "严格共同日期后的真实复权净值",
        "domains": ["portfolio"],
    },
    "asset_log_returns": {
        "label": "多资产对数收益矩阵",
        "semantic_role": "log_return_matrix",
        "source": "严格共同日期后的真实复权净值",
        "domains": ["portfolio"],
    },
    "portfolio_returns": {
        "label": "组合实际收益率序列",
        "semantic_role": "realized_portfolio_return",
        "source": "逐日生效权重与底层产品真实收益",
        "domains": ["portfolio"],
    },
    "asset_weights": {
        "label": "期末资产权重向量",
        "semantic_role": "weight",
        "source": "运行快照最后一个时点，仅用于当前截面估算",
        "domains": ["portfolio"],
    },
    "weight_path": {
        "label": "动态权重路径",
        "semantic_role": "weight_path",
        "source": "不可变运行快照",
        "domains": ["portfolio"],
    },
    "benchmark_returns": {
        "label": "基准收益率序列",
        "semantic_role": "benchmark_return",
        "source": "可选基准真实复权净值",
        "domains": ["portfolio"],
    },
    "annual_risk_free_rate_decimal": {
        "label": "年化无风险收益率",
        "semantic_role": "annual_risk_free_rate",
        "source": "指标配置",
        "domains": ["single_product", "portfolio"],
    },
    "risk_free_rate_per_observation": {
        "label": "单观察期无风险收益率",
        "semantic_role": "observation_risk_free_rate",
        "source": "由指标配置换算",
        "domains": ["single_product", "portfolio"],
    },
    "periods_per_year": {
        "label": "年化因子",
        "semantic_role": "annualization_factor",
        "source": "上下文约定",
        "domains": ["single_product", "portfolio"],
    },
}


OPERATOR_LABELS = {
    "add": ("逐元素加法", "对 A 与 B 执行逐元素加法；仅标量可广播。"),
    "subtract": ("逐元素减法", "对 A 与 B 执行逐元素减法；仅标量可广播。"),
    "multiply": ("逐元素乘法", "对同轴同 shape 输入逐元素相乘；仅标量可广播。"),
    "divide": ("逐元素安全除法", "逐元素计算 A 除以 B；分母为零或接近零时返回明确诊断。"),
    "power": ("逐元素幂", "逐元素计算 A 的 B 次幂。"),
    "minimum": ("逐元素最小值", "逐元素选择 A 与 B 的较小值。"),
    "maximum": ("逐元素最大值", "逐元素选择 A 与 B 的较大值。"),
    "negate": ("逐元素取负", "逐元素改变输入符号。"),
    "absolute": ("逐元素绝对值", "逐元素计算输入的绝对值。"),
    "sqrt": ("逐元素平方根", "逐元素计算非负输入的平方根。"),
    "clip": (
        "逐元素数值限幅",
        "按给定数值下界与上界逐元素限幅；不改变计算周期或样本窗口。",
    ),
    "sum": ("全元素求和", "将时间序列、向量或矩阵的全部元素相加并归约为标量。"),
    "product": ("全元素累乘", "将时间序列、向量或矩阵的全部元素相乘并归约为标量。"),
    "mean": ("全元素算术平均值", "计算时间序列、向量或矩阵全部元素的算术平均值并归约为标量。"),
    "variance": ("全元素方差", "计算时间序列、向量或矩阵全部元素的样本方差并归约为标量。"),
    "std": ("全元素标准差", "计算时间序列、向量或矩阵全部元素的样本标准差并归约为标量。"),
    "rolling_mean": ("滚动平均值", "沿时间轴按固定窗口计算算术平均值，窗口不足时返回缺失值。"),
    "rolling_std": ("滚动标准差", "沿时间轴按固定窗口计算标准差，可固定自由度修正和最少有效观察数。"),
    "rolling_min": ("滚动最小值", "沿时间轴按固定窗口取得最小值。"),
    "rolling_max": ("滚动最大值", "沿时间轴按固定窗口取得最大值。"),
    "recursive_smooth": ("递归平滑", "按固定平滑期数和初始值进行因果递归平滑。"),
    "divide_or_default": ("安全除法", "逐元素计算分子除以分母；分母为零或接近零时返回指定默认值。"),
    "min_value": ("全元素最小值", "归约得到时间序列、向量或矩阵全部元素中的最小值。"),
    "max_value": ("全元素最大值", "归约得到时间序列、向量或矩阵全部元素中的最大值。"),
    "cumulative_sum": ("一维累计和", "沿时间序列或向量的唯一名义轴生成累计和。"),
    "cumulative_product": ("一维累计乘积", "沿时间序列或向量的唯一名义轴生成累计乘积。"),
    "last": ("一维末值", "取得时间序列或向量的最后一个元素。"),
    "sum_time": ("时间轴求和", "沿矩阵时间轴求和，移除时间轴并保留资产轴。"),
    "sum_asset": ("资产轴求和", "沿矩阵资产轴求和，移除资产轴并保留时间轴。"),
    "product_time": ("时间轴累乘", "沿矩阵时间轴累乘，移除时间轴并保留资产轴。"),
    "product_asset": ("资产轴累乘", "沿矩阵资产轴累乘，移除资产轴并保留时间轴。"),
    "mean_time": ("时间轴算术平均值", "沿矩阵时间轴计算算术平均值，输出资产向量。"),
    "mean_asset": ("资产轴算术平均值", "沿矩阵资产轴计算算术平均值，输出时间序列。"),
    "variance_time": ("时间轴方差", "沿矩阵时间轴计算方差，输出资产向量。"),
    "variance_asset": ("资产轴方差", "沿矩阵资产轴计算方差，输出时间序列。"),
    "std_time": ("时间轴标准差", "沿矩阵时间轴计算标准差，输出资产向量。"),
    "std_asset": ("资产轴标准差", "沿矩阵资产轴计算标准差，输出时间序列。"),
    "min_time": ("时间轴最小值", "沿矩阵时间轴取最小值，输出资产向量。"),
    "min_asset": ("资产轴最小值", "沿矩阵资产轴取最小值，输出时间序列。"),
    "max_time": ("时间轴最大值", "沿矩阵时间轴取最大值，输出资产向量。"),
    "max_asset": ("资产轴最大值", "沿矩阵资产轴取最大值，输出时间序列。"),
    "transpose": ("转置", "交换矩阵的两个名义轴。"),
    "dot": ("点积", "对两个同轴一维输入执行点积并得到标量。"),
    "outer": ("外积", "对两个资产向量执行外积。"),
    "matmul": ("矩阵乘法", "按匹配的名义内轴执行矩阵 × 矩阵。"),
    "matvec": ("矩阵向量乘", "按匹配的名义内轴执行矩阵 × 向量。"),
    "diag": ("对角构造/提取", "从资产向量构造对角矩阵，或提取方阵对角线。"),
    "trace": ("迹", "对方阵主对角线求和。"),
    "solve": ("线性方程求解", "求解 A·x=b；不显式计算矩阵逆。"),
    "covariance": ("协方差", "按共同时间轴计算样本协方差或协方差矩阵。"),
    "correlation": ("相关系数", "按共同时间轴计算 Pearson 相关系数或矩阵。"),
    "quadratic_form": ("二次型", "计算 wᵀAw。"),
    "log": ("逐元素自然对数", "逐元素计算严格正、无量纲输入的自然对数。"),
    "exp": ("逐元素指数", "逐元素计算无量纲输入的自然指数。"),
    "reciprocal": ("逐元素倒数", "逐元素计算输入的倒数，并对零值返回诊断。"),
    "sign": ("逐元素符号", "逐元素返回负一、零或正一。"),
    "equal": ("逐元素等于比较", "比较同 shape 或标量广播输入，输出布尔 mask。"),
    "not_equal": ("逐元素不等比较", "比较同 shape 或标量广播输入，输出布尔 mask。"),
    "less_than": ("逐元素小于比较", "逐元素判断 A 是否小于 B，输出布尔 mask。"),
    "less_equal": ("逐元素小于等于比较", "逐元素判断 A 是否小于等于 B，输出布尔 mask。"),
    "greater_than": ("逐元素大于比较", "逐元素判断 A 是否大于 B，输出布尔 mask。"),
    "greater_equal": ("逐元素大于等于比较", "逐元素判断 A 是否大于等于 B，输出布尔 mask。"),
    "logical_and": ("逐元素逻辑与", "对两个同 shape 布尔 mask 执行逻辑与。"),
    "logical_or": ("逐元素逻辑或", "对两个同 shape 布尔 mask 执行逻辑或。"),
    "logical_not": ("逐元素逻辑非", "反转布尔 mask 的每一个元素。"),
    "where": ("按条件逐元素选择", "按布尔 mask 从两个兼容数值输入中逐元素选择。"),
    "first": ("一维首值", "取得时间序列或资产向量的第一个元素。"),
    "length": ("一维元素数量", "返回时间序列或资产向量的元素数量。"),
    "lag": ("一维滞后对齐", "按给定期数移除尾部元素，生成用于滞后比较的序列。"),
    "difference": ("一维差分", "计算一维输入相隔给定期数的差。"),
    "median": ("全元素中位数", "将数值数组全部元素归约为中位数。"),
    "skewness": ("全元素偏度", "计算数值数组的有限样本修正偏度。"),
    "excess_kurtosis": ("全元素超额峰度", "计算数值数组的有限样本修正超额峰度。"),
    "mean_absolute_deviation": ("平均绝对离差", "计算元素相对算术平均值的绝对偏差平均值。"),
    "root_mean_square": ("均方根", "计算元素平方平均值的平方根。"),
    "cumulative_max": ("一维累计最大值", "沿一维输入生成截至各位置的累计最大值。"),
    "cumulative_min": ("一维累计最小值", "沿一维输入生成截至各位置的累计最小值。"),
    "drawdown_series": (
        "回撤序列",
        "计算每个净值或价格相对截至当期历史峰值的有符号回撤；历史峰值处为零，峰值以下为负数。",
    ),
    "new_high_mask": (
        "严格创新高判断",
        "首个有效观察值计为创新高；后续仅严格超过此前历史峰值时为真，相同峰值不会重复计数。",
    ),
    "argmin": ("最小值位置", "返回全元素最小值的零基位置。"),
    "argmax": ("最大值位置", "返回全元素最大值的零基位置。"),
    "quantile": ("全元素分位数", "按受控概率参数计算全部元素的经验分位数。"),
    "sum_where": ("条件元素求和", "仅对布尔 mask 选中的元素求和。"),
    "mean_where": ("条件元素平均值", "仅对布尔 mask 选中的元素计算算术平均值。"),
    "variance_where": ("条件元素方差", "仅对布尔 mask 选中的元素计算样本方差。"),
    "std_where": ("条件元素标准差", "仅对布尔 mask 选中的元素计算样本标准差。"),
    "min_where": ("条件元素最小值", "仅在布尔 mask 选中的元素中取最小值。"),
    "max_where": ("条件元素最大值", "仅在布尔 mask 选中的元素中取最大值。"),
    "median_where": ("条件元素中位数", "仅对布尔 mask 选中的元素计算中位数。"),
    "quantile_where": ("条件元素分位数", "仅对布尔 mask 选中的元素计算经验分位数。"),
    "count_true": ("真值数量", "统计布尔 mask 中 True 元素的数量。"),
    "max_consecutive_true": ("最长连续真值数量", "计算一维布尔 mask 中最长连续 True 的长度。"),
    "linear_slope": ("线性回归斜率", "计算一元普通最小二乘回归斜率。"),
    "linear_intercept": ("线性回归截距", "计算一元普通最小二乘回归截距。"),
    "linear_r_squared": ("线性回归决定系数", "计算一元普通最小二乘回归 R²。"),
    "regression_standard_error": ("线性回归残差标准误", "计算一元普通最小二乘回归的残差标准误。"),
    "normal_pdf": ("标准正态密度", "逐元素计算标准正态分布概率密度。"),
    "normal_ppf": ("标准正态分位数", "逐元素计算开区间概率对应的标准正态分位数。"),
}


PARAMETER_NAMES: dict[str, tuple[str, ...]] = {
    "add": ("A", "B"),
    "subtract": ("A", "B"),
    "multiply": ("A", "B"),
    "divide": ("numerator", "denominator"),
    "power": ("base", "exponent"),
    "minimum": ("A", "B"),
    "maximum": ("A", "B"),
    "clip": ("values", "lower", "upper"),
    "dot": ("A", "B"),
    "outer": ("A", "B"),
    "matmul": ("A", "B"),
    "matvec": ("matrix", "vector"),
    "solve": ("matrix", "vector"),
    "quadratic_form": ("vector", "matrix"),
    "covariance": ("values",),
    "correlation": ("values",),
    "std": ("values", "ddof"),
    "variance": ("values", "ddof"),
    "rolling_mean": ("values", "window", "min_periods"),
    "rolling_std": ("values", "window", "ddof", "min_periods"),
    "rolling_min": ("values", "window", "min_periods"),
    "rolling_max": ("values", "window", "min_periods"),
    "recursive_smooth": ("values", "periods", "initial"),
    "divide_or_default": ("numerator", "denominator", "default"),
    "where": ("mask", "if_true", "if_false"),
    "lag": ("values", "periods"),
    "difference": ("values", "periods"),
    "drawdown_series": ("levels",),
    "new_high_mask": ("levels",),
    "quantile": ("values", "probability"),
    "sum_where": ("values", "mask"),
    "mean_where": ("values", "mask"),
    "variance_where": ("values", "mask"),
    "std_where": ("values", "mask"),
    "min_where": ("values", "mask"),
    "max_where": ("values", "mask"),
    "median_where": ("values", "mask"),
    "quantile_where": ("values", "mask", "probability"),
    "count_true": ("mask",),
}


_FIXED_CONSTANT_OPERATOR_PARAMETERS: dict[str, frozenset[str]] = {
    "rolling_mean": frozenset({"window", "min_periods"}),
    "rolling_std": frozenset({"window", "ddof", "min_periods"}),
    "rolling_min": frozenset({"window", "min_periods"}),
    "rolling_max": frozenset({"window", "min_periods"}),
    "recursive_smooth": frozenset({"periods", "initial"}),
    "divide_or_default": frozenset({"default"}),
    "lag": frozenset({"periods"}),
    "difference": frozenset({"periods"}),
    "variance": frozenset({"ddof"}),
    "std": frozenset({"ddof"}),
    "quantile": frozenset({"probability"}),
    "quantile_where": frozenset({"probability"}),
    "clip": frozenset({"lower", "upper"}),
    "power": frozenset({"exponent"}),
}

PARAMETER_LABELS = {
    "default": "分母无效时的默认值",
    "initial": "递推初始值",
    "values": "输入值",
    "levels": "净值或价格序列",
    "lhs": "输入 A",
    "rhs": "输入 B",
    "A": "输入 A",
    "B": "输入 B",
    "numerator": "分子",
    "denominator": "分母",
    "base": "底数",
    "exponent": "指数",
    "lower": "下界",
    "upper": "上界",
    "ddof": "自由度修正（ddof）",
    "matrix": "矩阵",
    "lhs_matrix": "左侧矩阵",
    "rhs_matrix": "右侧矩阵",
    "vector": "向量",
    "asset_returns": "多资产收益矩阵",
    "x": "自变量 X",
    "y": "因变量 Y",
    "mask": "布尔条件",
    "if_true": "条件成立值",
    "if_false": "条件不成立值",
    "periods": "间隔期数",
    "window": "窗口期数",
    "min_periods": "最少有效观察数",
    "initial": "递归初始值",
    "default": "分母为零时的默认值",
    "probability": "概率",
}


_FIXED_CONSTANT_PARAMETER_POLICIES: dict[tuple[str, str], dict[str, Any]] = {
    ("rolling_mean", "window"): {
        "source_policy": "fixed_constant",
        "constant_kind": "integer",
        "default": 20,
        "minimum": 1,
        "maximum": 20_000,
    },
    ("rolling_mean", "min_periods"): {
        "source_policy": "fixed_constant",
        "constant_kind": "integer",
        "default": 20,
        "minimum": 1,
        "maximum": 20_000,
    },
    ("rolling_std", "window"): {
        "source_policy": "fixed_constant",
        "constant_kind": "integer",
        "default": 20,
        "minimum": 1,
        "maximum": 20_000,
    },
    ("rolling_std", "ddof"): {
        "source_policy": "fixed_constant",
        "constant_kind": "integer",
        "default": 0,
        "minimum": 0,
        "maximum": 19_999,
    },
    ("rolling_std", "min_periods"): {
        "source_policy": "fixed_constant",
        "constant_kind": "integer",
        "default": 20,
        "minimum": 1,
        "maximum": 20_000,
    },
    ("rolling_min", "window"): {
        "source_policy": "fixed_constant",
        "constant_kind": "integer",
        "default": 20,
        "minimum": 1,
        "maximum": 20_000,
    },
    ("rolling_min", "min_periods"): {
        "source_policy": "fixed_constant",
        "constant_kind": "integer",
        "default": 20,
        "minimum": 1,
        "maximum": 20_000,
    },
    ("rolling_max", "window"): {
        "source_policy": "fixed_constant",
        "constant_kind": "integer",
        "default": 20,
        "minimum": 1,
        "maximum": 20_000,
    },
    ("rolling_max", "min_periods"): {
        "source_policy": "fixed_constant",
        "constant_kind": "integer",
        "default": 20,
        "minimum": 1,
        "maximum": 20_000,
    },
    ("recursive_smooth", "periods"): {
        "source_policy": "fixed_constant",
        "constant_kind": "integer",
        "default": 3,
        "minimum": 1,
        "maximum": 20_000,
    },
    ("recursive_smooth", "initial"): {
        "source_policy": "fixed_constant",
        "constant_kind": "number",
        "default": 50,
    },
    ("lag", "periods"): {
        "source_policy": "fixed_constant",
        "constant_kind": "integer",
        "default": 1,
        "minimum": 1,
        "maximum": 20_000,
    },
    ("difference", "periods"): {
        "source_policy": "fixed_constant",
        "constant_kind": "integer",
        "default": 1,
        "minimum": 1,
        "maximum": 20_000,
    },
    ("variance", "ddof"): {
        "source_policy": "fixed_constant",
        "constant_kind": "integer",
        "default": 1,
        "minimum": 0,
    },
    ("std", "ddof"): {
        "source_policy": "fixed_constant",
        "constant_kind": "integer",
        "default": 1,
        "minimum": 0,
    },
    ("quantile", "probability"): {
        "source_policy": "fixed_constant",
        "constant_kind": "number",
        "default": 0.5,
        "minimum": 0.0,
        "maximum": 1.0,
    },
    ("quantile_where", "probability"): {
        "source_policy": "fixed_constant",
        "constant_kind": "number",
        "default": 0.5,
        "minimum": 0.0,
        "maximum": 1.0,
    },
    ("clip", "lower"): {
        "source_policy": "fixed_constant",
        "constant_kind": "number",
        "default": 0.0,
    },
    ("clip", "upper"): {
        "source_policy": "fixed_constant",
        "constant_kind": "number",
        "default": 1.0,
    },
    ("divide_or_default", "default"): {
        "source_policy": "fixed_constant",
        "constant_kind": "number",
        "default": 0.0,
    },
}


@dataclass(frozen=True)
class TemplateSpec:
    template_id: str
    name: str
    description: str
    template_kind: Literal["indicator", "fragment"]
    domains: tuple[ContextKind, ...]
    parameters: tuple[dict[str, Any], ...]
    build: Callable[[dict[str, str]], str]
    output_contract: str
    display_format: str = "number"
    unit: str = ""
    precision: int = 3
    direction: str = "higher_better"

    def catalog_entry(self) -> dict[str, Any]:
        defaults = {
            parameter["name"]: str(parameter.get("default_expression") or "1")
            for parameter in self.parameters
        }
        return {
            "id": self.template_id,
            "label": self.name,
            "name": self.name,
            "description": self.description,
            "mathematical_essence": self.description,
            "semantic": self.description,
            "template_kind": self.template_kind,
            "domains": list(self.domains),
            "parameters": [copy.deepcopy(parameter) for parameter in self.parameters],
            "expression": self.build(defaults),
            "periods": ["1W", "1M", "3M", "6M", "1Y", "2Y", "3Y", "5Y", "10Y", "20Y", "30Y"],
            "unit": self.unit,
            "display_format": self.display_format,
            "precision": self.precision,
            "direction": self.direction,
            "annual_risk_free_rate_percent": 1.5,
            "output_shape": self.output_contract,
        }


TEMPLATES: tuple[TemplateSpec, ...] = (
    TemplateSpec(
        "cumulative-return",
        "累计收益率",
        "将普通收益率逐期加一后累乘，再减一。",
        "indicator",
        ("single_product",),
        (
            {
                "name": "values",
                "label": "普通收益率序列",
                "description": "允许普通收益率语义的时间序列或公式表达式；不能绑定对数收益率。",
                "shape": "series",
                "allowed_shapes": ["series"],
                "allowed_types": ["series<time>"],
                "allowed_semantic_roles": ["ordinary_return"],
                "default": "returns",
                "default_expression": r"\mathbf{r}",
            },
        ),
        lambda args: rf"\prod\left({args['values']}+1\right)-1",
        "scalar",
        "percent",
        "%",
        2,
    ),
    TemplateSpec(
        "portfolio-return-series",
        "组合收益序列",
        "使用矩阵向量乘，将多资产收益矩阵按权重合成为时间序列。",
        "fragment",
        ("portfolio",),
        (
            {
                "name": "returns_matrix",
                "label": "多资产普通收益矩阵",
                "description": "严格共同日期对齐的普通收益率矩阵。",
                "shape": "matrix",
                "allowed_shapes": ["matrix"],
                "allowed_semantic_roles": ["ordinary_return_matrix"],
                "default": "asset_returns",
                "default_expression": r"\mathbf{R}",
            },
            {
                "name": "weights",
                "label": "资产权重",
                "description": "与资产顺序一致且合计为 1 的向量。",
                "shape": "vector",
                "allowed_shapes": ["vector"],
                "allowed_semantic_roles": ["weight"],
                "default": "asset_weights",
                "default_expression": r"\mathbf{w}",
            },
        ),
        lambda args: rf"\operatorname{{matvec}}\left({args['returns_matrix']},{args['weights']}\right)",
        "series",
    ),
    TemplateSpec(
        "dynamic-portfolio-return-series",
        "动态权重组合收益序列",
        "将逐日资产收益与同日生效权重逐元素相乘，再沿资产轴求和。",
        "fragment",
        ("portfolio",),
        (
            {
                "name": "returns_matrix",
                "label": "多资产普通收益矩阵",
                "shape": "matrix",
                "allowed_shapes": ["matrix"],
                "allowed_semantic_roles": ["ordinary_return_matrix"],
                "default": "asset_returns",
                "default_expression": r"\mathbf{R}",
            },
            {
                "name": "weights_path",
                "label": "动态权重路径",
                "shape": "matrix",
                "allowed_shapes": ["matrix"],
                "allowed_semantic_roles": ["weight_path"],
                "default": "weight_path",
                "default_expression": r"\mathbf{W}",
            },
        ),
        lambda args: (
            rf"\operatorname{{sum_asset}}\left({args['returns_matrix']}\cdot {args['weights_path']}\right)"
        ),
        "series",
    ),
    TemplateSpec(
        "portfolio-cumulative-return",
        "组合累计收益率",
        "先以矩阵向量乘形成组合收益序列，再对增长因子累乘。",
        "indicator",
        ("portfolio",),
        (
            {
                "name": "returns_matrix",
                "label": "多资产普通收益矩阵",
                "shape": "matrix",
                "allowed_shapes": ["matrix"],
                "allowed_semantic_roles": ["ordinary_return_matrix"],
                "default": "asset_returns",
                "default_expression": r"\mathbf{R}",
            },
            {
                "name": "weights",
                "label": "资产权重",
                "shape": "vector",
                "allowed_shapes": ["vector"],
                "allowed_semantic_roles": ["weight"],
                "default": "asset_weights",
                "default_expression": r"\mathbf{w}",
            },
        ),
        lambda args: rf"\prod\left(\operatorname{{matvec}}\left({args['returns_matrix']},{args['weights']}\right)+1\right)-1",
        "scalar",
        "percent",
        "%",
        2,
    ),
    TemplateSpec(
        "portfolio-volatility",
        "组合波动率",
        "以协方差矩阵、矩阵向量乘、点积和平方根构造组合波动率。",
        "indicator",
        ("portfolio",),
        (
            {
                "name": "returns_matrix",
                "label": "多资产普通收益矩阵",
                "shape": "matrix",
                "allowed_shapes": ["matrix"],
                "allowed_semantic_roles": ["ordinary_return_matrix"],
                "default": "asset_returns",
                "default_expression": r"\mathbf{R}",
            },
            {
                "name": "weights",
                "label": "资产权重",
                "shape": "vector",
                "allowed_shapes": ["vector"],
                "allowed_semantic_roles": ["weight"],
                "default": "asset_weights",
                "default_expression": r"\mathbf{w}",
            },
        ),
        lambda args: (
            rf"\sqrt{{\operatorname{{dot}}\left({args['weights']},"
            rf"\operatorname{{matvec}}\left(\operatorname{{covariance}}\left({args['returns_matrix']}\right),{args['weights']}\right)\right)}}"
        ),
        "scalar",
        "percent",
        "%",
        2,
        "lower_better",
    ),
)

TEMPLATE_BY_ID = {item.template_id: item for item in TEMPLATES}


def _shape_from_type(value_type: dict[str, Any]) -> str:
    return str(value_type.get("kind") or "unknown")


def _operator_parameter_names(operator_id: str, arity: int) -> tuple[str, ...]:
    explicit = PARAMETER_NAMES.get(operator_id)
    if explicit and len(explicit) >= arity:
        return explicit[:arity]
    if arity == 1:
        return ("values",)
    if arity == 2:
        return ("A", "B")
    return tuple(f"arg{index + 1}" for index in range(arity))


def _operator_meta(entry: dict[str, Any]) -> dict[str, Any]:
    operator_id = str(entry["id"])
    category_id = str(entry.get("category") or "other")
    category_label = {
        "basic": "基础数学",
        "reduction": "归约计算",
        "sequence": "序列与路径",
        "path": "序列与路径",
        "statistics": "统计与回归",
        "mask": "比较与条件",
        "comparison": "比较与条件",
        "linear_algebra": "线性代数",
        "matrix": "矩阵统计",
        "portfolio": "线性代数",
        "rolling": "滚动与时序",
    }.get(category_id, "其他数学算子")
    label, essence = OPERATOR_LABELS.get(operator_id, (operator_id, str(entry.get("description") or "受控数学算子。")))
    signatures = list(entry.get("signatures") or [])

    def parameters_for_signature(signature: dict[str, Any]) -> list[dict[str, Any]]:
        inputs = [str(item) for item in signature.get("inputs") or []]
        declared_names = [str(item) for item in signature.get("parameters") or []]
        names = declared_names or list(_operator_parameter_names(operator_id, len(inputs)))
        result: list[dict[str, Any]] = []
        for index, contract in enumerate(inputs):
            name = names[index] if index < len(names) else f"arg{index + 1}"
            policy = copy.deepcopy(
                _FIXED_CONSTANT_PARAMETER_POLICIES.get((operator_id, name), {})
            )
            default = policy.get(
                "default",
                1
                if operator_id in {"std", "variance", "lag", "difference"}
                and name in {"ddof", "periods"}
                else None,
            )
            lowered = contract.lower()
            if "mask" in lowered:
                allowed_shapes = ["mask"]
            elif "numeric" in lowered:
                allowed_shapes = ["scalar", "series", "vector", "matrix"]
            elif "same(first)" in lowered and result:
                allowed_shapes = list(result[0]["allowed_shapes"])
            else:
                allowed_shapes = [
                    shape
                    for shape in ("scalar", "series", "vector", "matrix")
                    if shape in lowered
                ]
                if not allowed_shapes and "same(" in lowered:
                    allowed_shapes = ["scalar", "series", "vector", "matrix"]
            parameter_meta = {
                "name": name,
                "label": PARAMETER_LABELS.get(name, name),
                "description": (
                    "定义级固定常量；保存后成为指标版本的一部分。"
                    if policy.get("source_policy") == "fixed_constant"
                    else f"允许类型：{contract or '由算子签名约束'}。"
                ),
                "allowed_types": [contract],
                "allowed_shapes": allowed_shapes
                or ["scalar", "series", "vector", "matrix"],
                "optional": default is not None,
                "default": default,
                "requires_mask": "mask" in lowered,
            }
            parameter_meta.update(policy)
            parameter_meta["default"] = default
            result.append(parameter_meta)
        return result

    parameter_sets = [
        {
            "arity": len(signature.get("inputs") or []),
            "parameters": parameters_for_signature(signature),
            "output": signature.get("output"),
            "shape_rule": signature.get("shape_rule"),
        }
        for signature in signatures
    ]
    preferred_arity = (
        max((item["arity"] for item in parameter_sets), default=1)
        if operator_id in {"std", "variance", "lag", "difference"}
        else min((item["arity"] for item in parameter_sets), default=1)
    )
    parameters = next(
        (item["parameters"] for item in parameter_sets if item["arity"] == preferred_arity),
        [],
    )
    output = " | ".join(sorted({str(item.get("output")) for item in signatures}))
    output_shapes = [
        shape
        for shape in ("mask", "scalar", "series", "vector", "matrix")
        if shape in output
    ]
    output_shape = (
        output_shapes[0]
        if len(output_shapes) == 1
        and "same(" not in output
        and "one_dimensional" not in output
        else "unknown"
    )
    signature_inputs = [
        list(signature.get("inputs") or []) for signature in signatures
    ]

    def requires_portfolio(inputs: list[str]) -> bool:
        return any(
            ("matrix" in contract or "asset" in contract or "vector" in contract)
            and "series<" not in contract
            and "scalar" not in contract
            for contract in inputs
        )

    domains: list[ContextKind] = ["single_product", "portfolio"]
    if signature_inputs and all(requires_portfolio(inputs) for inputs in signature_inputs):
        domains = ["portfolio"]
    sample_contracts = {
        "variance": 2,
        "std": 2,
        "variance_where": 2,
        "std_where": 2,
        "skewness": 3,
        "excess_kurtosis": 4,
        "linear_slope": 2,
        "linear_intercept": 2,
        "linear_r_squared": 2,
        "regression_standard_error": 3,
        "covariance": 2,
        "correlation": 2,
    }
    fixed_ddof = 1 if operator_id in {
        "variance_where",
        "std_where",
        "covariance",
        "correlation",
    } else None
    source_latex = entry.get("latex_template") or rf"\operatorname{{{operator_id}}}(x)"
    latex_alias = entry.get("display_latex_template") or source_latex
    return {
        "name": operator_id,
        "id": operator_id,
        "family": category_id,
        "version": entry.get("version"),
        "label": label,
        "signature": " / ".join(
            f"{', '.join(item.get('inputs') or [])} → {item.get('output')}"
            for item in signatures
        ),
        "latex_template": source_latex,
        "latex_alias": latex_alias,
        "display_latex_template": latex_alias,
        "source_latex_template": source_latex,
        "return_type": output,
        "output_shape": output_shape,
        "mathematical_essence": essence,
        "semantic": essence,
        "category": category_id,
        "category_id": category_id,
        "category_label": category_label,
        "domains": domains,
        "parameters": parameters,
        "parameter_sets": parameter_sets,
        "signatures": signatures,
        "type_rules": signatures,
        "semantic_rules": {
            "description": essence,
            "enforcement": "compile_time",
        },
        "output_rule": {
            "type": output,
            "shape": output_shape,
        },
        "examples": [latex_alias],
        "cost_model": entry.get("cost_model"),
        "execution_backend": entry.get("execution_backend", "numpy"),
        "njit_supported": bool(entry.get("njit_supported")),
        "kernel_version": entry.get("kernel_version"),
        "execution_lane": entry.get("execution_lane"),
        "compiled_signatures": list(entry.get("compiled_signatures") or []),
        "warmup_status": entry.get("warmup_status", "pending"),
        "opcode": entry.get("opcode"),
        "kernel_input_signatures": list(entry.get("input_signatures") or []),
        "kernel_output_signature": entry.get("output_signature"),
        "parallel_policy": entry.get("parallel_policy"),
        "status_contract": entry.get("status_contract"),
        "cost": {"model": entry.get("cost_model")},
        "cost_estimate": entry.get("cost_model"),
        "minimum_samples": sample_contracts.get(operator_id, 1),
        "nan_policy": "reject_non_finite",
        "ddof": fixed_ddof,
    }


def typed_product_meta() -> dict[str, Any]:
    raw = get_typed_dsl_catalog()
    variables = variable_catalog()
    operators = [_operator_meta(entry) for entry in raw["operators"]]
    return {
        "dsl_version": raw["dsl_version"],
        "compiler_version": raw["compiler_version"],
        "operator_registry_version": raw["operator_registry_version"],
        "variable_registry_version": VARIABLE_REGISTRY_VERSION,
        "data_contract_version": DATA_CONTRACT_VERSION,
        "context_schema_version": CONTEXT_SCHEMA_VERSION,
        "math_notation_version": MATH_NOTATION_VERSION,
        "types": raw["type_system"]["types"],
        "type_system": raw["type_system"],
        "variables": variables,
        "operators": operators,
        # Formula templates/fragments are no longer a discoverable product
        # concept. Saved read-only indicators are the reusable examples. The
        # compose endpoint keeps accepting old template ids for compatibility.
        "predefined_calculations": [],
        "predefined_calculations_deprecated": True,
        "limits": raw["limits"],
    }


def _latex_for_variable(name: str) -> str:
    definition = get_variable(name)
    if definition is not None:
        return definition.latex
    raise ValidationError("UNKNOWN_VARIABLE", f"未知变量: {name}", "arguments")


def _allowed_variables(context: ContextKind) -> set[str]:
    return allowed_variables(context)


def _parameter_latex_symbol(parameter_id: str) -> str:
    escaped = parameter_id.replace("_", r"\_")
    return rf"\mathrm{{{escaped}}}"


def parameter_composition_context(
    raw_schema: Any,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, str],
    dict[str, str],
]:
    schema = normalize_parameter_schema(raw_schema or [])
    parameter_types = parameter_variable_types({"parameter_schema": schema})
    latex_symbols = {
        str(item["id"]): _parameter_latex_symbol(str(item["id"]))
        for item in schema
    }
    semantic_roles = {
        str(item["id"]): (
            "count" if item.get("type") == "integer" else "numeric_parameter"
        )
        for item in schema
    }
    return schema, parameter_types, latex_symbols, semantic_roles


def _argument_expressions(
    arguments: list[dict[str, Any]],
    context: ContextKind,
    *,
    additional_variable_types: Mapping[str, Any] | None = None,
    additional_semantics: Mapping[str, str] | None = None,
) -> tuple[dict[str, str], dict[str, Optional[str]]]:
    expressions: dict[str, str] = {}
    semantics: dict[str, Optional[str]] = {}
    extra_types = dict(additional_variable_types or {})
    extra_semantics = dict(additional_semantics or {})
    for argument in arguments:
        parameter = str(argument.get("parameter") or "")
        if not parameter or parameter in expressions:
            raise ValidationError("INVALID_TEMPLATE_ARGUMENT", "参数名不能为空且不能重复。", "arguments")
        source = str(argument.get("source") or "")
        value = argument.get("value")
        if source == "variable":
            requested_variable = str(value)
            if requested_variable in extra_types:
                expressions[parameter] = requested_variable
                semantics[parameter] = extra_semantics.get(requested_variable)
                continue
            variable = canonical_variable_id(requested_variable)
            if variable not in _allowed_variables(context):
                raise ValidationError(
                    "VARIABLE_CONTEXT_MISMATCH",
                    f"变量 {requested_variable} 不适用于 {context} 域。",
                    f"arguments.{parameter}",
                )
            expressions[parameter] = _latex_for_variable(variable)
            semantics[parameter] = variable_semantic_role(variable)
        elif source == "constant":
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise ValidationError("INVALID_CONSTANT", "常量必须是有限数值。", f"arguments.{parameter}") from exc
            if not math.isfinite(number):
                raise ValidationError("INVALID_CONSTANT", "常量必须是有限数值。", f"arguments.{parameter}")
            expressions[parameter] = repr(number)
            semantics[parameter] = "numeric_constant"
        elif source == "expression":
            expression = str(value or "").strip()
            if not expression:
                raise ValidationError("EMPTY_EXPRESSION_ARGUMENT", "公式表达式不能为空。", f"arguments.{parameter}")
            if len(expression) > 1000:
                raise ValidationError("FORMULA_TOO_LONG", "公式表达式不能超过 1000 个字符。", f"arguments.{parameter}")
            expressions[parameter] = expression
            semantics[parameter] = None
        else:
            raise ValidationError("INVALID_ARGUMENT_SOURCE", "参数来源必须为 variable、constant 或 expression。", f"arguments.{parameter}")
    return expressions, semantics


def _validate_fixed_constant_arguments(
    operator_id: str,
    arguments: list[dict[str, Any]],
    operator_registry_version: str,
) -> None:
    registry = get_typed_operator_registry(operator_registry_version)
    try:
        spec = registry[operator_id]
    except KeyError as exc:
        raise TypedDslError("UNKNOWN_OPERATOR", f"未知函数或算子: {operator_id}") from exc
    arity = len(arguments)
    if arity not in spec.arities:
        return
    canonical_names = spec.argument_names(arity)
    legacy_names = _operator_parameter_names(operator_id, arity)
    by_name = {str(item.get("parameter") or ""): item for item in arguments}
    for index, canonical_name in enumerate(canonical_names):
        legacy_name = legacy_names[index] if index < len(legacy_names) else canonical_name
        candidates = tuple(
            dict.fromkeys((canonical_name, legacy_name, f"input_{index + 1}"))
        )
        argument = next((by_name[name] for name in candidates if name in by_name), None)
        if argument is None:
            continue
        policy = _FIXED_CONSTANT_PARAMETER_POLICIES.get(
            (spec.operator_id, canonical_name)
        )
        if not policy:
            continue
        if str(argument.get("source") or "") != "constant":
            raise ValidationError(
                "SERIES_CONFIGURATION_MUST_BE_CONSTANT",
                f"{spec.operator_id} 的 {PARAMETER_LABELS.get(canonical_name, canonical_name)}必须是定义级固定常量。",
                f"arguments.{canonical_name}",
            )
        try:
            value = float(argument.get("value"))
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                "INVALID_CONSTANT",
                "固定配置必须是有限数值。",
                f"arguments.{canonical_name}",
            ) from exc
        if not math.isfinite(value):
            raise ValidationError(
                "INVALID_CONSTANT",
                "固定配置必须是有限数值。",
                f"arguments.{canonical_name}",
            )
        if policy.get("constant_kind") == "integer" and not value.is_integer():
            raise ValidationError(
                "SERIES_CONFIGURATION_MUST_BE_INTEGER",
                f"{PARAMETER_LABELS.get(canonical_name, canonical_name)}必须是整数常量。",
                f"arguments.{canonical_name}",
            )
        minimum = policy.get("minimum")
        maximum = policy.get("maximum")
        if minimum is not None and value < float(minimum):
            raise ValidationError(
                "SERIES_CONFIGURATION_OUT_OF_RANGE",
                f"{PARAMETER_LABELS.get(canonical_name, canonical_name)}不能小于 {minimum:g}。",
                f"arguments.{canonical_name}",
            )
        if maximum is not None and value > float(maximum):
            raise ValidationError(
                "SERIES_CONFIGURATION_OUT_OF_RANGE",
                f"{PARAMETER_LABELS.get(canonical_name, canonical_name)}不能大于 {maximum:g}。",
                f"arguments.{canonical_name}",
            )


def _operator_expression(
    operator_id: str,
    arguments: dict[str, str],
    operator_registry_version: str = TYPED_OPERATOR_REGISTRY_VERSION,
) -> str:
    registry = get_typed_operator_registry(operator_registry_version)
    try:
        spec = registry[operator_id]
    except KeyError as exc:
        raise TypedDslError("UNKNOWN_OPERATOR", f"未知函数或算子: {operator_id}") from exc
    arity = len(arguments)
    if arity not in spec.arities:
        raise ValidationError("ARITY_MISMATCH", f"算子 {operator_id} 参数数量无效。", "arguments")

    # The registry is the public parameter contract returned by /meta and must
    # therefore also own compose binding. PARAMETER_NAMES predates typed v2.1;
    # accept those positional names only as compatibility aliases.
    names = spec.argument_names(arity)
    legacy_names = _operator_parameter_names(operator_id, arity)
    resolved_names: list[str] = []
    missing: list[str] = []
    for index, name in enumerate(names):
        legacy_name = legacy_names[index] if index < len(legacy_names) else name
        candidates = tuple(
            dict.fromkeys((name, legacy_name, f"input_{index + 1}"))
        )
        resolved = next((candidate for candidate in candidates if candidate in arguments), None)
        if resolved is None:
            missing.append(name)
        else:
            resolved_names.append(resolved)
    if missing:
        raise ValidationError("MISSING_ARGUMENT", f"缺少参数: {', '.join(missing)}", "arguments")
    unknown = sorted(set(arguments) - set(resolved_names))
    if unknown:
        raise ValidationError(
            "UNKNOWN_ARGUMENT",
            f"未知或重复参数: {', '.join(unknown)}",
            "arguments",
        )
    values = [arguments[name] for name in resolved_names]
    if operator_id == "add":
        return rf"\left({values[0]}+{values[1]}\right)"
    if operator_id == "subtract":
        return rf"\left({values[0]}-{values[1]}\right)"
    if operator_id == "multiply":
        return rf"\left({values[0]}\cdot {values[1]}\right)"
    if operator_id == "divide":
        return rf"\frac{{{values[0]}}}{{{values[1]}}}"
    if operator_id == "power":
        return rf"\left({values[0]}\right)^{{{values[1]}}}"
    if operator_id == "negate":
        return rf"-\left({values[0]}\right)"
    if operator_id == "sqrt":
        return rf"\sqrt{{{values[0]}}}"
    if operator_id == "product":
        return rf"\prod\left({values[0]}\right)"
    if operator_id == "sum":
        return rf"\sum\left({values[0]}\right)"
    return rf"\operatorname{{{operator_id}}}\left({','.join(values)}\right)"


def _diagnostic_from_typed(error: TypedDslError, field: str = "expression") -> ValidationError:
    diagnostic = error.to_dict()
    details = diagnostic.get("details") or {}
    for key in ("expected", "actual"):
        if key in details:
            diagnostic[key] = details[key]
    diagnostic["field"] = field
    return ValidationError(error.code, error.message, field, [diagnostic])


def _ensure_context(
    plan: Any,
    context: ContextKind,
    additional_variables: Mapping[str, Any] | None = None,
) -> None:
    allowed = _allowed_variables(context) | set(additional_variables or {})
    unavailable = sorted(set(plan.context_requirements) - allowed)
    if unavailable:
        raise ValidationError(
            "CONTEXT_VARIABLE_UNAVAILABLE",
            f"{context} 域不提供变量: {', '.join(unavailable)}。",
            "expression",
        )


def infer_expression(
    expression: str,
    context: ContextKind,
    *,
    scalar_required: bool = False,
    dsl_version: str = TYPED_DSL_VERSION,
    operator_registry_version: str = TYPED_OPERATOR_REGISTRY_VERSION,
    additional_variable_types: Mapping[str, Any] | None = None,
    additional_latex_symbols: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    normalized_expression = normalize_variable_latex(expression)
    extended_variable_types = {
        **variable_types(context, dsl_version),
        **dict(additional_variable_types or {}),
    }
    try:
        plan = infer_typed_expression(
            normalized_expression,
            variable_types=extended_variable_types,
            allow_non_scalar_root=not scalar_required,
            dsl_version=dsl_version,
            operator_registry_version=operator_registry_version,
        )
        _ensure_context(plan, context, additional_variable_types)
        compiled = compile_numba_plan(plan)
    except NumbaPlanCompileError as exc:
        raise ValidationError(
            "NJIT_PLAN_COMPILE_FAILED",
            "公式无法编译为 NJIT 计算计划。",
            "expression",
            [
                {
                    "code": "NJIT_PLAN_COMPILE_FAILED",
                    "message": "公式无法编译为 NJIT 计算计划。",
                    "field": "expression",
                    "compiled_plan_id": exc.plan_id,
                    "operator": exc.operator_id,
                }
            ],
        ) from exc
    except TypedDslError as exc:
        raise _diagnostic_from_typed(exc) from exc
    output = plan.output_type.to_dict()
    latex_symbols = {
        **variable_latex_symbols(context),
        **dict(additional_latex_symbols or {}),
    }
    display_latex = render_python_expression_latex(
        plan.python_expression,
        latex_symbols,
    )
    dag = copy.deepcopy(plan.graph_payload())
    for node in dag.get("nodes", []):
        fragment = str(node.get("formula_fragment") or node.get("label") or "")
        try:
            node["latex_fragment"] = render_python_expression_latex(
                fragment,
                latex_symbols,
            )
        except (SyntaxError, ValueError):
            node["latex_fragment"] = None
    warnings: list[dict[str, Any]] = []
    if "log_returns" in plan.context_requirements and any(
        node.operator_id in {"product", "cumulative_product"} for node in plan.nodes
    ):
        warnings.append(
            {
                "code": "SEMANTIC_ROLE_WARNING",
                "message": "对数收益率不应直接用于增长因子累乘；请确认是否应改用普通收益率。",
            }
        )
    return {
        "expression": expression,
        # ``latex`` remains the executable, round-trippable surface for older
        # clients. New UIs render ``display_latex`` instead.
        "latex": expression,
        "display_latex": display_latex,
        "math_notation_version": MATH_NOTATION_VERSION,
        "normalized_expression": normalized_expression,
        "python_expression": plan.python_expression,
        "inferred_type": output["display"],
        "shape": output["kind"],
        "semantic_warnings": warnings,
        "dependencies": list(plan.context_requirements),
        "dag": dag,
        "estimated_cost": plan.estimated_cost,
        "dsl_version": plan.dsl_version,
        "operator_registry_version": plan.operator_registry_version,
        **compiled.metadata(),
    }


def compose_expression(request: dict[str, Any]) -> dict[str, Any]:
    context = str(request.get("context") or "single_product")
    if context not in {"single_product", "portfolio"}:
        raise ValidationError("INVALID_CONTEXT_KIND", "上下文域必须为 single_product 或 portfolio。", "context")
    (
        parameter_schema,
        parameter_types,
        parameter_latex_symbols,
        parameter_semantics,
    ) = parameter_composition_context(request.get("parameter_schema") or [])
    expressions, semantics = _argument_expressions(
        list(request.get("arguments") or []),
        context,  # type: ignore[arg-type]
        additional_variable_types=parameter_types,
        additional_semantics=parameter_semantics,
    )
    operator_id = request.get("operator_id")
    template_id = request.get("template_id")
    dsl_version = str(request.get("dsl_version") or TYPED_DSL_VERSION)
    operator_registry_version = str(
        request.get("operator_registry_version")
        or (
            LEGACY_OPERATOR_REGISTRY_VERSION
            if dsl_version == LEGACY_TYPED_DSL_VERSION
            else (
                COMPAT_OPERATOR_REGISTRY_VERSION
                if dsl_version == COMPAT_TYPED_DSL_VERSION
                else (
                    PREVIOUS_OPERATOR_REGISTRY_VERSION
                    if dsl_version == PREVIOUS_TYPED_DSL_VERSION
                    else TYPED_OPERATOR_REGISTRY_VERSION
                )
            )
        )
    )
    if bool(operator_id) == bool(template_id):
        raise ValidationError("INVALID_COMPOSE_TARGET", "必须且只能指定 operator_id 或 template_id。")
    if operator_id:
        try:
            _validate_fixed_constant_arguments(
                str(operator_id),
                list(request.get("arguments") or []),
                operator_registry_version,
            )
            expression = _operator_expression(
                str(operator_id),
                expressions,
                operator_registry_version,
            )
        except TypedDslError as exc:
            raise _diagnostic_from_typed(exc, "operator_id") from exc
        result = infer_expression(
            expression,
            context,  # type: ignore[arg-type]
            scalar_required=False,
            dsl_version=dsl_version,
            operator_registry_version=operator_registry_version,
            additional_variable_types=parameter_types,
            additional_latex_symbols=parameter_latex_symbols,
        )
        result["parameter_schema"] = copy.deepcopy(parameter_schema)
        result["template_origin"] = None
        return result

    template = TEMPLATE_BY_ID.get(str(template_id))
    if template is None:
        raise ValidationError("TEMPLATE_NOT_FOUND", "未找到指定公式模板。", "template_id")
    if context not in template.domains:
        raise ValidationError("TEMPLATE_CONTEXT_MISMATCH", "该公式模板不适用于当前域。", "context")
    expected = {parameter["name"] for parameter in template.parameters}
    if set(expressions) != expected:
        missing = sorted(expected - set(expressions))
        unknown = sorted(set(expressions) - expected)
        raise ValidationError(
            "TEMPLATE_ARGUMENT_MISMATCH",
            f"模板参数不完整。缺少: {missing or '无'}；未知: {unknown or '无'}。",
            "arguments",
        )
    for parameter in template.parameters:
        allowed_semantics = parameter.get("allowed_semantic_roles") or []
        semantic = semantics.get(parameter["name"])
        if semantic is not None and allowed_semantics and semantic not in allowed_semantics:
            raise ValidationError(
                "SEMANTIC_ROLE_MISMATCH",
                f"参数 {parameter['name']} 不允许绑定 {semantic} 语义。",
                f"arguments.{parameter['name']}",
            )
    expression = template.build(expressions)
    result = infer_expression(
        expression,
        context,  # type: ignore[arg-type]
        scalar_required=template.output_contract == "scalar",
        dsl_version=dsl_version,
        operator_registry_version=operator_registry_version,
        additional_variable_types=parameter_types,
        additional_latex_symbols=parameter_latex_symbols,
    )
    result["parameter_schema"] = copy.deepcopy(parameter_schema)
    result["template_origin"] = {
        "template_id": template.template_id,
        "template_version": 1,
        "bindings": copy.deepcopy(request.get("arguments") or []),
        "detached": False,
    }
    return result


__all__ = [
    "TEMPLATES",
    "compose_expression",
    "infer_expression",
    "parameter_composition_context",
    "typed_product_meta",
]

"""Definitions and validation for named multi-channel time-series indicators."""

from __future__ import annotations

import ast
import copy
import math
from typing import Any, Mapping

from cal_indicators.typed_dsl import TypedExpressionParser
from cal_indicators.typed_numba_kernels import NUMERIC_KERNEL_VERSION
from cal_indicators.typed_operators import (
    TYPED_DSL_VERSION,
    TYPED_OPERATOR_REGISTRY_VERSION,
)
from cal_indicators.typed_types import ValueType

from .errors import ValidationError
from .rolling_scalar import (
    lift_scalar_expression,
    rolling_source_metadata,
    validate_scalar_rolling_source,
)
from .periods import SUPPORTED_PERIODS
from .rolling_series import (
    derive_rolling_series_definition,
    normalize_rolling_source,
)
from .variable_registry import (
    CONTEXT_SCHEMA_VERSION,
    DATA_CONTRACT_VERSION,
    VARIABLE_REGISTRY_VERSION,
    get_variable,
)


TIME_SERIES_RESULT_KIND = "time_series"
TIME_SERIES_OUTPUT_CONTRACT = "series_bundle"
SERIES_HISTORY_POLICIES = frozenset({"lookback", "full_history"})
MAX_SERIES_CHANNELS = 8
MAX_SERIES_PARAMETERS = 16
MAX_SERIES_EXPRESSION_LENGTH = 4_000

# ``output_measure`` is presentation metadata, not part of the numerical type
# system.  Physical semantics continue to come from ``ValueType``; the profiles
# below add a stable display/range contract for dimensionless series.
_SERIES_OUTPUT_MEASURE_CATALOG: tuple[dict[str, Any], ...] = (
    {
        "id": "auto",
        "label": "自动推断",
        "description": "根据公式的类型量纲、价格基准和可证明的取值范围自动选择。",
        "semantic_dimensions": ["*"],
        "range": None,
        "default_unit": "",
        "default_display_format": "number",
    },
    {
        "id": "raw_market_price",
        "label": "原始市价",
        "description": "未复权开盘价、最高价、最低价、收盘价及其同量纲变换。",
        "semantic_dimensions": ["raw_market_price"],
        "range": None,
        "default_unit": "元",
        "default_display_format": "number",
    },
    {
        "id": "adjusted_nav",
        "label": "复权净值",
        "description": "复权后的单位净值或累计财富水平。",
        "semantic_dimensions": ["adjusted_nav"],
        "range": None,
        "default_unit": "净值",
        "default_display_format": "number",
    },
    {
        "id": "reported_nav",
        "label": "披露净值",
        "description": "基金披露的单位净值，不包含复权处理。",
        "semantic_dimensions": ["reported_nav"],
        "range": None,
        "default_unit": "净值",
        "default_display_format": "number",
    },
    {
        "id": "virtual_nav",
        "label": "虚拟净值（起点 1）",
        "description": "由收益率累计得到、通常以 1 为起点的无量纲财富指数。",
        "semantic_dimensions": ["dimensionless"],
        "range": None,
        "default_unit": "",
        "default_display_format": "number",
    },
    {
        "id": "normalized",
        "label": "归一化值",
        "description": "无物理单位的相对水平；不额外承诺固定上下界。",
        "semantic_dimensions": ["dimensionless"],
        "range": None,
        "default_unit": "",
        "default_display_format": "number",
    },
    {
        "id": "bounded_0_1",
        "label": "0～1 区间",
        "description": "明确限制在闭区间 [0, 1] 的无量纲值。",
        "semantic_dimensions": ["dimensionless"],
        "range": [0.0, 1.0],
        "default_unit": "",
        "default_display_format": "number",
    },
    {
        "id": "bounded_minus1_1",
        "label": "-1～1 区间",
        "description": "明确限制在闭区间 [-1, 1] 的无量纲值。",
        "semantic_dimensions": ["dimensionless"],
        "range": [-1.0, 1.0],
        "default_unit": "",
        "default_display_format": "number",
    },
    {
        "id": "oscillator_0_100",
        "label": "0～100 摆动值",
        "description": "技术指标常用的 0～100 无量纲刻度。",
        "semantic_dimensions": ["dimensionless"],
        "range": [0.0, 100.0],
        "default_unit": "",
        "default_display_format": "number",
    },
    {
        "id": "return_decimal",
        "label": "收益率",
        "description": "内部以小数保存，展示时通常使用百分比。",
        "semantic_dimensions": ["return_decimal"],
        "range": None,
        "default_unit": "%",
        "default_display_format": "percent",
    },
    {
        "id": "rate_decimal",
        "label": "利率",
        "description": "内部以小数保存的利率或费率。",
        "semantic_dimensions": ["rate_decimal"],
        "range": None,
        "default_unit": "%",
        "default_display_format": "percent",
    },
    {
        "id": "volume",
        "label": "成交量",
        "description": "份额、手数或其他成交数量口径。",
        "semantic_dimensions": ["volume"],
        "range": [0.0, None],
        "default_unit": "份",
        "default_display_format": "number",
    },
    {
        "id": "currency_amount",
        "label": "金额",
        "description": "成交额、资产规模或其他货币金额。",
        "semantic_dimensions": ["currency_amount"],
        "range": None,
        "default_unit": "元",
        "default_display_format": "number",
    },
    {
        "id": "count",
        "label": "计数",
        "description": "观察数、次数或其他离散计数。",
        "semantic_dimensions": ["count"],
        "range": [0.0, None],
        "default_unit": "",
        "default_display_format": "number",
    },
    {
        "id": "calendar_days",
        "label": "日历天数",
        "description": "以自然日为单位的期限或间隔。",
        "semantic_dimensions": ["calendar_days"],
        "range": [0.0, None],
        "default_unit": "天",
        "default_display_format": "number",
    },
    {
        "id": "dimensionless",
        "label": "其他无量纲值",
        "description": "没有物理单位，且不声明固定范围的数值。",
        "semantic_dimensions": ["dimensionless"],
        "range": None,
        "default_unit": "",
        "default_display_format": "number",
    },
    {
        "id": "derived",
        "label": "复合量纲",
        "description": "乘方、乘除等运算形成的派生量纲；由类型系统锁定。",
        "semantic_dimensions": ["derived:*", "squared:*", "inverse:*"],
        "range": None,
        "default_unit": "",
        "default_display_format": "number",
    },
)
_SERIES_OUTPUT_MEASURE_BY_ID = {
    str(item["id"]): item for item in _SERIES_OUTPUT_MEASURE_CATALOG
}
_SEMANTIC_DEFAULT_MEASURE = {
    "raw_market_price": "raw_market_price",
    "adjusted_nav": "adjusted_nav",
    "reported_nav": "reported_nav",
    "return_decimal": "return_decimal",
    "rate_decimal": "rate_decimal",
    "volume": "volume",
    "currency_amount": "currency_amount",
    "count": "count",
    "calendar_days": "calendar_days",
    "dimensionless": "dimensionless",
}



def _derived_semantic_dimension(semantic_dimension: str) -> bool:
    return semantic_dimension.startswith(("derived:", "squared:", "inverse:"))


def infer_series_output_measure(
    output_type: ValueType,
    *,
    proven_range: tuple[float, float] | None = None,
) -> str:
    """Infer the narrowest presentation profile justified by the typed DAG."""

    semantic_dimension = str(output_type.semantic_dimension)
    if semantic_dimension == "dimensionless" and proven_range is not None:
        lower, upper = proven_range
        if math.isclose(lower, 0.0) and math.isclose(upper, 1.0):
            return "bounded_0_1"
        if math.isclose(lower, -1.0) and math.isclose(upper, 1.0):
            return "bounded_minus1_1"
        if math.isclose(lower, 0.0) and math.isclose(upper, 100.0):
            return "oscillator_0_100"
    if _derived_semantic_dimension(semantic_dimension):
        return "derived"
    return _SEMANTIC_DEFAULT_MEASURE.get(semantic_dimension, "dimensionless")


def _measure_accepts_semantic_dimension(
    measure_id: str,
    semantic_dimension: str,
) -> bool:
    profile = _SERIES_OUTPUT_MEASURE_BY_ID[measure_id]
    allowed = tuple(str(item) for item in profile["semantic_dimensions"])
    if "*" in allowed or semantic_dimension in allowed:
        return True
    if any(item.endswith(":*") and semantic_dimension.startswith(item[:-1]) for item in allowed):
        return True
    return False


def resolve_series_output_measure(
    requested_measure: str | None,
    output_type: ValueType,
    *,
    proven_range: tuple[float, float] | None = None,
    field: str = "series_outputs.output_measure",
) -> dict[str, Any]:
    """Validate an explicit profile or resolve ``auto`` from typed semantics."""

    requested = str(requested_measure or "auto")
    if requested not in _SERIES_OUTPUT_MEASURE_BY_ID:
        raise ValidationError(
            "INVALID_SERIES_OUTPUT_MEASURE",
            f"未知时序输出量纲: {requested}。",
            field=field,
        )
    inferred = infer_series_output_measure(output_type, proven_range=proven_range)
    resolved = inferred if requested == "auto" else requested
    semantic_dimension = str(output_type.semantic_dimension)
    if not _measure_accepts_semantic_dimension(resolved, semantic_dimension):
        expected = _SERIES_OUTPUT_MEASURE_BY_ID[inferred]["label"]
        actual = _SERIES_OUTPUT_MEASURE_BY_ID[resolved]["label"]
        raise ValidationError(
            "SERIES_OUTPUT_MEASURE_MISMATCH",
            f"公式推断为“{expected}”，不能声明为“{actual}”。",
            field=field,
            diagnostics=[
                {
                    "code": "SERIES_OUTPUT_MEASURE_MISMATCH",
                    "message": f"公式语义量纲为 {semantic_dimension}。",
                    "expected": inferred,
                    "actual": resolved,
                }
            ],
        )
    profile = _SERIES_OUTPUT_MEASURE_BY_ID[resolved]
    return {
        "requested": requested,
        "inferred": inferred,
        "resolved": resolved,
        "semantic_dimension": semantic_dimension,
        "price_basis": output_type.price_basis,
        "range": copy.deepcopy(profile["range"]),
        "default_unit": str(profile["default_unit"]),
        "default_display_format": str(profile["default_display_format"]),
        "source": "inferred" if requested == "auto" else "explicit",
    }


def _channel(
    channel_id: str,
    label: str,
    expression: str,
    *,
    unit: str,
    output_measure: str = "auto",
    precision: int = 4,
    display_format: str = "number",
) -> dict[str, Any]:
    return {
        "id": channel_id,
        "label": label,
        "expression": expression,
        "unit": unit,
        "display_format": display_format,
        "precision": precision,
        "output_measure": output_measure,
    }


def _common(
    *,
    timestamp: str,
    indicator_id: str,
    name: str,
    description: str,
    channels: list[dict[str, Any]],
    required_variables: list[str],
    axis_anchor: str,
    history_policy: str,
    lookback_observations: int,
    minimum_observations: int,
    methodology: str,
    indicator_type: str = "technical",
    category_label: str = "技术与时序指标",
    annual_risk_free_rate_percent: float = 0.0,
    applicable_product_kinds: tuple[str, ...] = ("etf",),
    data_basis: str = "ETF 未复权日 K 行情；按日期对齐，缺失保留为空，不前向填充",
    fixed_parameters: list[dict[str, Any]] | None = None,
    rolling_source: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    primary = channels[0]
    return {
        "id": indicator_id,
        "revision": 1,
        "source": "built_in",
        "read_only": True,
        "created_at": timestamp,
        "updated_at": timestamp,
        "name": name,
        "description": description,
        "expression": primary["expression"],
        "series_outputs": channels,
        # Time-series definitions are immutable numerical contracts. Constants
        # such as 5, 10 or 20 belong in the formula rather than request payloads.
        "parameter_schema": [],
        "fixed_parameters": copy.deepcopy(fixed_parameters or []),
        "result_kind": TIME_SERIES_RESULT_KIND,
        "output_contract": TIME_SERIES_OUTPUT_CONTRACT,
        "context_kind": "single_product",
        "indicator_type": indicator_type,
        "category_id": indicator_type,
        "category_label": category_label,
        "direction": "higher_better",
        "unit": primary["unit"],
        "display_format": primary["display_format"],
        "precision": primary["precision"],
        "output_measure": "series_bundle",
        "annual_risk_free_rate_percent": float(annual_risk_free_rate_percent),
        "periods": list(SUPPORTED_PERIODS),
        "period_policy": "all_supported",
        "dsl_version": TYPED_DSL_VERSION,
        "operator_registry_version": TYPED_OPERATOR_REGISTRY_VERSION,
        "numeric_kernel_version": NUMERIC_KERNEL_VERSION,
        "variable_registry_version": VARIABLE_REGISTRY_VERSION,
        "data_contract_version": DATA_CONTRACT_VERSION,
        "context_schema_version": CONTEXT_SCHEMA_VERSION,
        "axis_anchor": axis_anchor,
        # These values are compiler-derived again during validation. Keeping
        # them on built-ins makes catalog reads useful before the first validate.
        "history_policy": history_policy,
        "lookback_parameter": None,
        "lookback_observations": int(lookback_observations),
        "required_variables": required_variables,
        "applicable_product_kinds": list(applicable_product_kinds),
        "minimum_observations": minimum_observations,
        "methodology": methodology,
        "data_basis": data_basis,
        "availability_status": "ready",
        "formula_version": TYPED_DSL_VERSION,
        "template_origin": None,
        "rolling_source": copy.deepcopy(rolling_source),
    }


def rolling_scalar_time_series_definition(
    *,
    timestamp: str,
    source_definition: Mapping[str, Any],
    window_observations: Any,
    min_periods: Any | None = None,
    name: str | None = None,
    description: str | None = None,
    channel_id: str = "value",
    channel_label: str | None = None,
    read_only: bool = False,
) -> dict[str, Any]:
    """Lift one locked scalar indicator into a normal time-series definition.

    This is a definition-time transform only. The generated formula is later
    compiled by the ordinary fixed-signature time-series NJIT path.
    """

    validate_scalar_rolling_source(source_definition)
    source_id = str(source_definition.get("id") or "").strip()
    source_revision = int(source_definition.get("revision") or 0)
    if not source_id or source_revision < 1:
        raise ValidationError(
            "INVALID_ROLLING_SOURCE",
            "来源标量指标必须是已保存的不可变版本。",
            field="rolling_source.indicator_id",
        )

    lifted = lift_scalar_expression(
        str(source_definition.get("expression") or ""),
        window_observations=window_observations,
        min_periods=min_periods,
    )
    window = int(lifted["window_observations"])
    minimum = int(lifted["min_periods"])
    source_name = str(source_definition.get("name") or source_id)
    resolved_name = str(name or f"{window} 日滚动{source_name}").strip()
    resolved_description = str(
        description
        or f"由“{source_name}”第 {source_revision} 版按最近 {window} 个有效观察值滚动计算。"
    ).strip()
    resolved_channel_id = str(channel_id or "value").strip()
    if not resolved_channel_id.isidentifier():
        raise ValidationError(
            "INVALID_SERIES_CHANNEL_ID",
            "时序输出通道 ID 必须是有效标识符。",
            field="series_outputs.0.id",
        )

    required_variables = list(source_definition.get("required_variables") or [])
    axis_anchor = next(
        (
            variable
            for variable in required_variables
            if (definition := get_variable(str(variable))) is not None
            and definition.kind == "series"
            and "single_product" in definition.domains
        ),
        None,
    )
    if axis_anchor is None:
        raise ValidationError(
            "INVALID_AXIS_ANCHOR",
            "来源标量指标没有可用于滚动时序日期轴的单产品序列变量。",
            field="axis_anchor",
        )

    unit = str(source_definition.get("unit") or "")
    display_format = str(source_definition.get("display_format") or "number")
    precision = int(source_definition.get("precision", 4))
    rolling_source = rolling_source_metadata(
        source_definition,
        window_observations=window,
        min_periods=minimum,
    )
    return {
        "source": "built_in" if read_only else "user_defined",
        "read_only": bool(read_only),
        "created_at": timestamp,
        "updated_at": timestamp,
        "name": resolved_name,
        "description": resolved_description,
        "expression": str(lifted["expression"]),
        "series_outputs": [
            _channel(
                resolved_channel_id,
                str(channel_label or resolved_name),
                str(lifted["expression"]),
                unit=unit,
                output_measure="auto",
                precision=precision,
                display_format=display_format,
            )
        ],
        "parameter_schema": [],
        "fixed_parameters": [
            {
                "id": "window_observations",
                "label": "滚动观察数",
                "type": "integer",
                "value": window,
                "source": "rolling_transform",
            },
            {
                "id": "min_periods",
                "label": "最少有效观察数",
                "type": "integer",
                "value": minimum,
                "source": "rolling_transform",
            },
        ],
        "result_kind": TIME_SERIES_RESULT_KIND,
        "output_contract": TIME_SERIES_OUTPUT_CONTRACT,
        "context_kind": "single_product",
        "indicator_type": str(source_definition.get("indicator_type") or "other"),
        "category_id": str(source_definition.get("category_id") or source_definition.get("indicator_type") or "other"),
        "category_label": str(source_definition.get("category_label") or "技术与时序指标"),
        "direction": str(source_definition.get("direction") or "higher_better"),
        "unit": unit,
        "display_format": display_format,
        "precision": precision,
        "output_measure": "series_bundle",
        "annual_risk_free_rate_percent": float(
            source_definition.get("annual_risk_free_rate_percent") or 0.0
        ),
        "periods": list(SUPPORTED_PERIODS),
        "period_policy": "all_supported",
        "dsl_version": TYPED_DSL_VERSION,
        "operator_registry_version": TYPED_OPERATOR_REGISTRY_VERSION,
        "numeric_kernel_version": NUMERIC_KERNEL_VERSION,
        "variable_registry_version": VARIABLE_REGISTRY_VERSION,
        "data_contract_version": DATA_CONTRACT_VERSION,
        "context_schema_version": CONTEXT_SCHEMA_VERSION,
        "axis_anchor": str(axis_anchor),
        "history_policy": "lookback",
        "history_inference_source": "rolling_scalar_transform",
        "lookback_parameter": None,
        "lookback_observations": window,
        "required_variables": required_variables,
        "applicable_product_kinds": list(
            source_definition.get("applicable_product_kinds") or ["etf", "fund"]
        ),
        "minimum_observations": minimum,
        "methodology": (
            f"将来源标量公式中的 {lifted['lifted_reductions']} 个归约步骤转换为"
            f"固定 {window} 个观察值、最少 {minimum} 个有效观察值的滚动计算。"
        ),
        "data_basis": str(
            source_definition.get("data_basis")
            or "沿用来源指标的真实单产品数据口径；日期对齐，缺失不填充"
        ),
        "availability_status": "ready",
        "formula_version": TYPED_DSL_VERSION,
        "template_origin": None,
        "rolling_source": rolling_source,
        "rolling_transform": {
            "version": str(lifted["transform_version"]),
            "window_observations": window,
            "min_periods": minimum,
            "source_expression": str(source_definition.get("expression") or ""),
            "generated_expression": str(lifted["expression"]),
            "lifted_reductions": int(lifted["lifted_reductions"]),
        },
    }


def time_series_builtin_indicators(
    timestamp: str,
    scalar_indicators: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Return immutable built-ins whose algorithm parameters are formula literals."""

    price_ma = _common(
        timestamp=timestamp,
        indicator_id="builtin-close-moving-average-series",
        name="20 日收盘价均线",
        description="对 ETF 原始收盘价计算固定 20 个交易日的简单移动平均。",
        channels=[
            _channel(
                "ma",
                "20 日收盘价均线",
                "rolling_mean(market_close, 20)",
                unit="",
                output_measure="auto",
            )
        ],
        required_variables=["market_close"],
        axis_anchor="market_close",
        history_policy="lookback",
        lookback_observations=20,
        minimum_observations=20,
        methodology="在每个时点使用截至当日最近 20 个有限收盘价求算术平均；样本不足 20 个时为空。",
    )

    middle = "rolling_mean(market_close, 20)"
    deviation = "rolling_std(market_close, 20)"
    bollinger = _common(
        timestamp=timestamp,
        indicator_id="builtin-bollinger-bands-series",
        name="20 日布林带",
        description="收盘价固定 20 日均值上下叠加 2 倍总体标准差。",
        channels=[
            _channel(
                "upper",
                "布林上轨",
                f"{middle} + 2 * {deviation}",
                unit="",
                output_measure="auto",
            ),
            _channel(
                "middle",
                "布林中轨",
                middle,
                unit="",
                output_measure="auto",
            ),
            _channel(
                "lower",
                "布林下轨",
                f"{middle} - 2 * {deviation}",
                unit="",
                output_measure="auto",
            ),
        ],
        required_variables=["market_close"],
        axis_anchor="market_close",
        history_policy="lookback",
        lookback_observations=20,
        minimum_observations=20,
        methodology="中轨为 20 日简单移动平均；上下轨为中轨加减 2 倍总体标准差（ddof=0）。",
    )

    volume_ma = _common(
        timestamp=timestamp,
        indicator_id="builtin-volume-moving-average-series",
        name="10 日成交量均线",
        description="对 ETF 日成交量计算固定 10 个交易日的简单移动平均。",
        channels=[
            _channel(
                "volume_ma",
                "10 日成交量均线",
                "rolling_mean(volume, 10)",
                unit="",
                output_measure="auto",
                precision=2,
            )
        ],
        required_variables=["market_close", "volume"],
        axis_anchor="market_close",
        history_policy="lookback",
        lookback_observations=10,
        minimum_observations=10,
        methodology="在每个时点使用截至当日最近 10 个有限成交量求算术平均；样本不足 10 个时为空。",
    )

    lowest = "rolling_min(market_low, 9, 1)"
    highest = "rolling_max(market_high, 9, 1)"
    rsv = (
        f"divide_or_default((market_close - {lowest}) * 100, "
        f"{highest} - {lowest}, 50)"
    )
    k_value = f"recursive_smooth({rsv}, 3, 50)"
    d_value = f"recursive_smooth({k_value}, 3, 50)"
    kdj = _common(
        timestamp=timestamp,
        indicator_id="builtin-kdj-series",
        name="KDJ（9, 3, 3）",
        description="以固定 9 日高低区间计算 RSV，并按固定 3 日参数递归平滑得到 K、D、J。",
        channels=[
            _channel(
                "k",
                "K 值",
                k_value,
                unit="",
                output_measure="oscillator_0_100",
                precision=2,
            ),
            _channel(
                "d",
                "D 值",
                d_value,
                unit="",
                output_measure="oscillator_0_100",
                precision=2,
            ),
            _channel(
                "j",
                "J 值",
                f"3 * ({k_value}) - 2 * ({d_value})",
                unit="",
                output_measure="auto",
                precision=2,
            ),
        ],
        required_variables=["market_close", "market_high", "market_low"],
        axis_anchor="market_close",
        history_policy="full_history",
        lookback_observations=9,
        minimum_observations=1,
        methodology="RSV 分母为零时取 50；K、D 初始值均为 50；J=3K-2D，J 不裁剪。",
    )

    scalar_by_id = {
        str(item.get("id") or ""): item for item in scalar_indicators or []
    }
    sharpe_source = scalar_by_id.get("builtin-annualized-sharpe-v2")
    if sharpe_source is None:
        raise RuntimeError("annualized Sharpe scalar built-in is required")
    rolling_sharpe = derive_rolling_series_definition(
        sharpe_source,
        5,
        name="5 日滚动年化夏普比率",
        description="对每个时点最近 5 个有效收益观察值计算样本标准差口径的年化夏普比率。",
    )
    rolling_sharpe.update(
        {
            "id": "builtin-rolling-5d-annualized-sharpe-series",
            "revision": 1,
            "source": "built_in",
            "read_only": True,
            "created_at": timestamp,
            "updated_at": timestamp,
        }
    )
    return [price_ma, bollinger, volume_ma, kdj, rolling_sharpe]

def parameter_variable_types(definition: Mapping[str, Any]) -> dict[str, ValueType]:
    """Compatibility support for old persisted parameterized definitions."""

    output: dict[str, ValueType] = {}
    for item in definition.get("parameter_schema") or []:
        parameter_id = str(item.get("id") or "")
        semantic_dimension = "count" if item.get("type") == "integer" else "dimensionless"
        output[parameter_id] = ValueType.scalar(
            semantic_dimension=semantic_dimension
        )
    return output


def series_expressions(definition: Mapping[str, Any]) -> dict[str, str]:
    return {
        str(item["id"]): str(item["expression"])
        for item in definition.get("series_outputs") or []
    }


def normalize_series_parameters(
    definition: Mapping[str, Any],
    supplied: Mapping[str, Any] | None,
) -> dict[str, float]:
    """Time-series algorithm parameters are immutable definition constants."""

    supplied_values = dict(supplied or {})
    if supplied_values:
        raise ValidationError(
            "SERIES_RUNTIME_PARAMETERS_NOT_SUPPORTED",
            "时序指标的窗口、平滑周期和阈值必须固定在指标公式中；请创建另一个指标或新版本。",
            field="parameters",
            diagnostics=[
                {
                    "code": "SERIES_RUNTIME_PARAMETERS_NOT_SUPPORTED",
                    "parameter": str(name),
                    "message": f"运行时参数 {name} 不允许覆盖已锁定的指标公式。",
                }
                for name in sorted(supplied_values)
            ],
        )
    return {}

def _normalized_parameter_schema(items: Any) -> list[dict[str, Any]]:
    """Validate legacy runtime parameters before freezing their defaults.

    New time-series definitions do not persist runtime algorithm parameters. This
    parser exists only so an older saved definition can be migrated to literal
    constants without accepting runtime overrides.
    """

    if not isinstance(items, list) or len(items) > MAX_SERIES_PARAMETERS:
        raise ValidationError(
            "INVALID_SERIES_PARAMETERS",
            f"时序指标最多定义 {MAX_SERIES_PARAMETERS} 个参数。",
            field="parameter_schema",
        )

    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(items):
        field = f"parameter_schema.{index}"
        if not isinstance(raw, Mapping):
            raise ValidationError(
                "INVALID_SERIES_PARAMETER",
                "时序参数定义必须是对象。",
                field=field,
            )

        parameter_id = str(raw.get("id") or "").strip()
        if (
            not parameter_id.isidentifier()
            or parameter_id in seen
            or get_variable(parameter_id)
        ):
            raise ValidationError(
                "INVALID_SERIES_PARAMETER_ID",
                "时序参数 ID 必须是唯一标识符，且不能覆盖数据变量。",
                field=f"{field}.id",
            )
        seen.add(parameter_id)

        parameter_type = str(raw.get("type") or "number")
        if parameter_type not in {"integer", "number"}:
            raise ValidationError(
                "INVALID_SERIES_PARAMETER_TYPE",
                "时序参数类型必须为 integer 或 number。",
                field=f"{field}.type",
            )

        def finite_number(key: str, fallback: Any = None) -> float:
            value = raw.get(key, fallback)
            if isinstance(value, bool):
                raise ValidationError(
                    "INVALID_SERIES_PARAMETER",
                    f"旧版时序参数 {parameter_id} 的 {key} 必须是有限数值。",
                    field=f"{field}.{key}",
                )
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise ValidationError(
                    "INVALID_SERIES_PARAMETER",
                    f"旧版时序参数 {parameter_id} 的 {key} 必须是有限数值。",
                    field=f"{field}.{key}",
                ) from exc
            if not math.isfinite(number):
                raise ValidationError(
                    "INVALID_SERIES_PARAMETER",
                    f"旧版时序参数 {parameter_id} 的 {key} 必须是有限数值。",
                    field=f"{field}.{key}",
                )
            return number

        default = finite_number("default")
        minimum = finite_number("minimum", default)
        maximum = finite_number("maximum", default)
        step = finite_number("step", 1)
        if minimum > maximum or not minimum <= default <= maximum:
            raise ValidationError(
                "INVALID_SERIES_PARAMETER_RANGE",
                "旧版时序参数的默认值必须位于有效上下界内。",
                field=field,
            )
        if step <= 0:
            raise ValidationError(
                "INVALID_SERIES_PARAMETER_STEP",
                "时序参数步长必须是有限正数。",
                field=f"{field}.step",
            )
        if parameter_type == "integer" and not all(
            value.is_integer() for value in (default, minimum, maximum, step)
        ):
            raise ValidationError(
                "INVALID_SERIES_PARAMETER_RANGE",
                "整数参数的默认值、上下界和步长都必须是整数。",
                field=field,
            )

        normalized.append(
            {
                "id": parameter_id,
                "label": str(raw.get("label") or parameter_id).strip()[:80],
                "type": parameter_type,
                "default": int(default) if parameter_type == "integer" else default,
                "minimum": int(minimum) if parameter_type == "integer" else minimum,
                "maximum": int(maximum) if parameter_type == "integer" else maximum,
                "step": int(step) if parameter_type == "integer" else step,
                "description": str(raw.get("description") or "").strip()[:300],
            }
        )
    return normalized


def normalize_parameter_schema(items: Any) -> list[dict[str, Any]]:
    """Validate legacy parameter metadata used by compose/infer compatibility."""

    return _normalized_parameter_schema(copy.deepcopy(items))


def _normalized_fixed_parameters(items: Any) -> list[dict[str, Any]]:
    if items is None or items == "":
        return []
    if not isinstance(items, list) or len(items) > MAX_SERIES_PARAMETERS:
        raise ValidationError(
            "INVALID_FIXED_SERIES_PARAMETERS",
            f"时序指标最多记录 {MAX_SERIES_PARAMETERS} 个固定参数。",
            field="fixed_parameters",
        )
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(items):
        if not isinstance(raw, Mapping):
            raise ValidationError(
                "INVALID_FIXED_SERIES_PARAMETER",
                "固定参数必须是对象。",
                field=f"fixed_parameters.{index}",
            )
        parameter_id = str(raw.get("id") or "").strip()
        parameter_type = str(raw.get("type") or "number")
        if (
            not parameter_id.isidentifier()
            or parameter_id in seen
            or parameter_type not in {"integer", "number"}
        ):
            raise ValidationError(
                "INVALID_FIXED_SERIES_PARAMETER",
                "固定参数 ID 必须唯一，类型必须为 integer 或 number。",
                field=f"fixed_parameters.{index}",
            )
        if isinstance(raw.get("value"), bool):
            raise ValidationError(
                "INVALID_FIXED_SERIES_PARAMETER",
                "固定参数必须是有限数值。",
                field=f"fixed_parameters.{index}.value",
            )
        try:
            value = float(raw.get("value"))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValidationError(
                "INVALID_FIXED_SERIES_PARAMETER",
                "固定参数必须是有限数值。",
                field=f"fixed_parameters.{index}.value",
            ) from exc
        if not math.isfinite(value) or (
            parameter_type == "integer" and not value.is_integer()
        ):
            raise ValidationError(
                "INVALID_FIXED_SERIES_PARAMETER",
                "固定参数必须符合声明类型并且是有限数值。",
                field=f"fixed_parameters.{index}.value",
            )
        seen.add(parameter_id)
        normalized.append(
            {
                "id": parameter_id,
                "label": str(raw.get("label") or parameter_id).strip()[:80],
                "type": parameter_type,
                "value": int(value) if parameter_type == "integer" else value,
                "source": str(raw.get("source") or "definition").strip()[:80],
            }
        )
    return normalized


def _normalized_outputs(items: Any) -> list[dict[str, Any]]:
    if not isinstance(items, list) or not 1 <= len(items) <= MAX_SERIES_CHANNELS:
        raise ValidationError(
            "INVALID_SERIES_OUTPUTS",
            f"时序指标需要 1 至 {MAX_SERIES_CHANNELS} 个输出通道。",
            field="series_outputs",
        )
    valid_measures = series_output_measure_ids()
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(items):
        if not isinstance(raw, Mapping):
            raise ValidationError(
                "INVALID_SERIES_OUTPUT",
                "时序输出通道必须是对象。",
                field=f"series_outputs.{index}",
            )
        channel_id = str(raw.get("id") or "").strip()
        expression = str(raw.get("expression") or "").strip()
        if not channel_id.isidentifier() or channel_id in seen:
            raise ValidationError(
                "INVALID_SERIES_OUTPUT_ID",
                "时序输出通道 ID 必须是唯一标识符。",
                field=f"series_outputs.{index}.id",
            )
        if not expression or len(expression) > MAX_SERIES_EXPRESSION_LENGTH:
            raise ValidationError(
                "INVALID_SERIES_EXPRESSION",
                f"每个时序输出公式长度必须为 1 至 {MAX_SERIES_EXPRESSION_LENGTH}。",
                field=f"series_outputs.{index}.expression",
            )
        display_format = str(raw.get("display_format") or "number")
        if display_format not in {"number", "percent"}:
            raise ValidationError(
                "INVALID_DISPLAY_FORMAT",
                "时序输出显示格式必须为 number 或 percent。",
                field=f"series_outputs.{index}.display_format",
            )
        precision = int(raw.get("precision", 4))
        if not 0 <= precision <= 8:
            raise ValidationError(
                "INVALID_PRECISION",
                "时序输出精度必须位于 0 至 8。",
                field=f"series_outputs.{index}.precision",
            )
        output_measure = str(raw.get("output_measure") or "auto").strip()
        if output_measure not in valid_measures:
            raise ValidationError(
                "INVALID_SERIES_OUTPUT_MEASURE",
                f"不支持的时序输出口径: {output_measure}。",
                field=f"series_outputs.{index}.output_measure",
            )
        seen.add(channel_id)
        item = {
            "id": channel_id,
            "label": str(raw.get("label") or channel_id).strip()[:80],
            "expression": expression,
            "unit": str(raw.get("unit") or "").strip()[:20],
            "display_format": display_format,
            "precision": precision,
            "output_measure": output_measure,
        }
        for metadata_key in (
            "inferred_output_measure",
            "resolved_output_measure",
            "output_measure_source",
            "semantic_dimension",
            "price_basis",
            "value_range",
        ):
            if metadata_key in raw:
                item[metadata_key] = copy.deepcopy(raw.get(metadata_key))
        normalized.append(item)
    return normalized


def _freeze_parameter_defaults(
    expression: str,
    parameters: list[dict[str, Any]],
) -> str:
    if not parameters:
        return expression
    values = {
        str(item["id"]): (
            int(item["default"])
            if item.get("type") == "integer"
            else float(item["default"])
        )
        for item in parameters
    }
    try:
        parsed = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValidationError(
            "SERIES_PARAMETER_FREEZE_FAILED",
            "旧版时序参数只能在可解析的受限 DSL 公式中固化。",
            field="series_outputs",
        ) from exc

    class Freezer(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.AST:  # noqa: N802
            if node.id not in values:
                return node
            return ast.copy_location(ast.Constant(value=values[node.id]), node)

        def visit_Call(self, node: ast.Call) -> ast.AST:  # noqa: N802
            node = self.generic_visit(node)
            if not isinstance(node.func, ast.Name):
                return node
            operator_id = node.func.id
            if operator_id in {"rolling_mean", "rolling_min", "rolling_max"} and len(node.args) == 3:
                if ast.dump(node.args[1], include_attributes=False) == ast.dump(node.args[2], include_attributes=False):
                    node.args = node.args[:2]
            elif operator_id == "rolling_std":
                if len(node.args) == 4:
                    ddof = node.args[2]
                    same_minimum = ast.dump(node.args[1], include_attributes=False) == ast.dump(node.args[3], include_attributes=False)
                    if isinstance(ddof, ast.Constant) and float(ddof.value) == 0.0 and same_minimum:
                        node.args = node.args[:2]
                elif len(node.args) == 3:
                    ddof = node.args[2]
                    if isinstance(ddof, ast.Constant) and float(ddof.value) == 0.0:
                        node.args = node.args[:2]
            return node

    frozen = Freezer().visit(parsed)
    ast.fix_missing_locations(frozen)
    return ast.unparse(frozen.body)

def normalize_time_series_definition(
    fields: Mapping[str, Any],
    protocol_defaults: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    defaults = dict(protocol_defaults or {})
    name = str(fields.get("name") or defaults.get("name") or "").strip()
    if not 1 <= len(name) <= 80:
        raise ValidationError("INVALID_NAME", "指标名称长度应为 1 至 80 个字符。", field="name")
    description = str(fields.get("description") or defaults.get("description") or "").strip()
    if len(description) > 500:
        raise ValidationError("INVALID_DESCRIPTION", "指标说明不能超过 500 个字符。", field="description")

    raw_parameters = copy.deepcopy(
        fields.get("parameter_schema")
        if "parameter_schema" in fields
        else defaults.get("parameter_schema", [])
    )
    parameters = _normalized_parameter_schema(raw_parameters or [])
    raw_outputs = copy.deepcopy(
        fields.get("series_outputs")
        if "series_outputs" in fields
        else defaults.get("series_outputs", [])
    )
    if parameters:
        if not isinstance(raw_outputs, list):
            raise ValidationError(
                "INVALID_SERIES_OUTPUTS",
                "时序输出通道必须是数组。",
                field="series_outputs",
            )
        for output in raw_outputs:
            if isinstance(output, Mapping):
                output["expression"] = _freeze_parameter_defaults(
                    str(output.get("expression") or ""), parameters
                )
    outputs = _normalized_outputs(raw_outputs)

    axis_anchor = str(fields.get("axis_anchor") or defaults.get("axis_anchor") or "").strip()
    anchor_definition = get_variable(axis_anchor)
    if (
        anchor_definition is None
        or anchor_definition.kind != "series"
        or "single_product" not in anchor_definition.domains
    ):
        raise ValidationError(
            "INVALID_AXIS_ANCHOR",
            "时序指标必须选择一个单产品时间序列变量作为日期轴。",
            field="axis_anchor",
        )
    dsl_version = str(fields.get("dsl_version") or defaults.get("dsl_version") or TYPED_DSL_VERSION)
    operator_registry_version = str(
        fields.get("operator_registry_version")
        or defaults.get("operator_registry_version")
        or TYPED_OPERATOR_REGISTRY_VERSION
    )
    if dsl_version != TYPED_DSL_VERSION or operator_registry_version != TYPED_OPERATOR_REGISTRY_VERSION:
        raise ValidationError(
            "SERIES_PROTOCOL_VERSION_MISMATCH",
            "新建时序指标必须使用当前 typed DSL 与算子注册表版本。",
            field="dsl_version",
        )
    history_hint = str(fields.get("history_policy") or defaults.get("history_policy") or "lookback")
    if history_hint not in SERIES_HISTORY_POLICIES:
        history_hint = "lookback"
    first = outputs[0]
    raw_fixed_parameters = copy.deepcopy(
        fields.get("fixed_parameters")
        if "fixed_parameters" in fields
        else defaults.get("fixed_parameters", [])
    )
    fixed_parameters = _normalized_fixed_parameters(raw_fixed_parameters or [])
    fixed_by_id = {str(item["id"]): item for item in fixed_parameters}
    for item in parameters:
        parameter_id = str(item["id"])
        fixed_by_id.setdefault(
            parameter_id,
            {
                "id": parameter_id,
                "label": str(item.get("label") or parameter_id),
                "type": str(item.get("type") or "number"),
                "value": copy.deepcopy(item["default"]),
                "source": "legacy_parameter_default",
            },
        )
    fixed_parameters = list(fixed_by_id.values())

    indicator_type = str(
        fields.get("indicator_type")
        or defaults.get("indicator_type")
        or "technical"
    )
    if indicator_type not in {
        "return",
        "risk",
        "risk_adjusted",
        "path",
        "market_liquidity",
        "technical",
        "other",
    }:
        raise ValidationError(
            "INVALID_INDICATOR_TYPE",
            "不支持的时序指标类型。",
            field="indicator_type",
        )
    direction = str(
        fields.get("direction") or defaults.get("direction") or "higher_better"
    )
    if direction not in {"higher_better", "lower_better"}:
        raise ValidationError(
            "INVALID_DIRECTION",
            "不支持的指标优劣方向。",
            field="direction",
        )
    try:
        risk_free_rate = float(
            fields.get("annual_risk_free_rate_percent")
            if "annual_risk_free_rate_percent" in fields
            else defaults.get("annual_risk_free_rate_percent", 0.0)
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError(
            "INVALID_RISK_FREE_RATE",
            "年化无风险利率必须是 -100 至 100 之间的有限数值。",
            field="annual_risk_free_rate_percent",
        ) from exc
    if not math.isfinite(risk_free_rate) or not -100.0 <= risk_free_rate <= 100.0:
        raise ValidationError(
            "INVALID_RISK_FREE_RATE",
            "年化无风险利率必须是 -100 至 100 之间的有限数值。",
            field="annual_risk_free_rate_percent",
        )
    rolling_source = normalize_rolling_source(
        fields.get("rolling_source")
        if "rolling_source" in fields
        else defaults.get("rolling_source")
    )
    rolling_transform = copy.deepcopy(
        fields.get("rolling_transform")
        if "rolling_transform" in fields
        else defaults.get("rolling_transform")
    )
    category_label = str(
        fields.get("category_label")
        or defaults.get("category_label")
        or (
            "风险调整指标"
            if indicator_type == "risk_adjusted"
            else "技术与时序指标"
        )
    )[:80]
    return {
        "name": name,
        "description": description,
        "expression": first["expression"],
        "series_outputs": outputs,
        # Algorithm parameters are constants embedded in each expression.
        "parameter_schema": [],
        "fixed_parameters": fixed_parameters,
        "result_kind": TIME_SERIES_RESULT_KIND,
        "output_contract": TIME_SERIES_OUTPUT_CONTRACT,
        "context_kind": "single_product",
        "indicator_type": indicator_type,
        "category_id": indicator_type,
        "category_label": category_label,
        "direction": direction,
        "unit": first["unit"],
        "display_format": first["display_format"],
        "precision": first["precision"],
        "output_measure": "series_bundle",
        "annual_risk_free_rate_percent": risk_free_rate,
        "periods": list(SUPPORTED_PERIODS),
        "period_policy": "all_supported",
        "dsl_version": dsl_version,
        "operator_registry_version": operator_registry_version,
        "numeric_kernel_version": NUMERIC_KERNEL_VERSION,
        "variable_registry_version": VARIABLE_REGISTRY_VERSION,
        "data_contract_version": DATA_CONTRACT_VERSION,
        "context_schema_version": CONTEXT_SCHEMA_VERSION,
        "axis_anchor": axis_anchor,
        # These are compiler-owned hints and are overwritten after typed-DAG validation.
        "history_policy": history_hint,
        "history_inference_source": "typed_dag",
        "lookback_parameter": None,
        "lookback_observations": max(
            1,
            int(fields.get("lookback_observations") or defaults.get("lookback_observations") or 1),
        ),
        "required_variables": list(
            fields.get("required_variables")
            if "required_variables" in fields
            else defaults.get("required_variables", [])
        ),
        "applicable_product_kinds": list(
            fields.get("applicable_product_kinds")
            if "applicable_product_kinds" in fields
            else defaults.get("applicable_product_kinds", ["etf", "fund"])
        ),
        "minimum_observations": max(
            1,
            int(fields.get("minimum_observations") or defaults.get("minimum_observations") or 1),
        ),
        "methodology": str(fields.get("methodology") or defaults.get("methodology") or description),
        "data_basis": str(
            fields.get("data_basis")
            or defaults.get("data_basis")
            or "真实数据、日期轴对齐、缺失不填充"
        ),
        "availability_status": "ready",
        "formula_version": dsl_version,
        "template_origin": copy.deepcopy(
            fields.get("template_origin")
            if "template_origin" in fields
            else defaults.get("template_origin")
        ),
        "rolling_source": rolling_source,
        "rolling_transform": rolling_transform,
    }


_PUBLIC_SERIES_OUTPUT_MEASURES: tuple[dict[str, Any], ...] = (
    {"id": "auto", "label": "自动推断", "description": "根据公式类型、价格基准和可证明数值范围推断。", "compatible_semantic_dimensions": []},
    {"id": "raw_market_price", "label": "原始市价", "description": "未复权开高低收及其同量纲变换。", "compatible_semantic_dimensions": ["raw_market_price"], "default_unit": "元", "default_display_format": "number"},
    {"id": "adjusted_nav", "label": "复权净值", "description": "包含复权处理的净值水平。", "compatible_semantic_dimensions": ["adjusted_nav"], "default_unit": "净值", "default_display_format": "number"},
    {"id": "reported_nav", "label": "披露净值", "description": "基金披露单位净值或累计净值。", "compatible_semantic_dimensions": ["reported_nav"], "default_unit": "净值", "default_display_format": "number"},
    {"id": "virtual_nav", "label": "虚拟净值（起点 1）", "description": "由收益累计形成、起点归一为 1 的财富路径。", "compatible_semantic_dimensions": ["dimensionless", "adjusted_nav"], "default_unit": "净值", "default_display_format": "number"},
    {"id": "normalized", "label": "归一化值", "description": "无单位相对水平，不承诺固定上下界。", "compatible_semantic_dimensions": ["dimensionless"], "default_unit": "", "default_display_format": "number"},
    {"id": "bounded_0_1", "label": "0～1 区间", "description": "概率或严格限制在 0 到 1 的归一化结果。", "compatible_semantic_dimensions": ["dimensionless"], "default_unit": "", "default_display_format": "number", "range": {"minimum": 0.0, "maximum": 1.0, "bounded": True}},
    {"id": "bounded_minus1_1", "label": "-1～1 区间", "description": "严格限制在 -1 到 1 的有符号结果。", "compatible_semantic_dimensions": ["dimensionless"], "default_unit": "", "default_display_format": "number", "range": {"minimum": -1.0, "maximum": 1.0, "bounded": True}},
    {"id": "oscillator_0_100", "label": "0～100 摆动值", "description": "K、D、RSI 等技术摆动指标。", "compatible_semantic_dimensions": ["dimensionless"], "default_unit": "", "default_display_format": "number", "range": {"minimum": 0.0, "maximum": 100.0, "bounded": True}},
    {"id": "return_decimal", "label": "收益率", "description": "内部使用小数，展示时通常转换为百分比。", "compatible_semantic_dimensions": ["return_decimal"], "default_unit": "%", "default_display_format": "percent"},
    {"id": "rate_decimal", "label": "利率或费率", "description": "内部使用小数，展示时通常转换为百分比。", "compatible_semantic_dimensions": ["rate_decimal"], "default_unit": "%", "default_display_format": "percent"},
    {"id": "volume", "label": "成交量", "description": "份额、手数或其他成交数量。", "compatible_semantic_dimensions": ["volume"], "default_unit": "份", "default_display_format": "number"},
    {"id": "currency_amount", "label": "金额", "description": "成交额、资产规模等货币金额。", "compatible_semantic_dimensions": ["currency_amount"], "default_unit": "元", "default_display_format": "number"},
    {"id": "count", "label": "计数", "description": "观察数、次数或期数。", "compatible_semantic_dimensions": ["count"], "default_unit": "", "default_display_format": "number"},
    {"id": "calendar_days", "label": "日历天数", "description": "期限、间隔或持续天数。", "compatible_semantic_dimensions": ["calendar_days"], "default_unit": "天", "default_display_format": "number"},
    {"id": "dimensionless", "label": "其他无量纲值", "description": "没有物理单位且没有可证明固定区间。", "compatible_semantic_dimensions": ["dimensionless"], "default_unit": "", "default_display_format": "number"},
    {"id": "derived", "label": "复合量纲", "description": "由乘方、乘法或除法形成的派生量纲。", "compatible_semantic_dimensions": ["derived", "squared", "inverse"], "default_unit": "", "default_display_format": "number"},
)


def series_output_measure_catalog() -> list[dict[str, Any]]:
    """Return the public dropdown catalog as a defensive copy."""

    return copy.deepcopy(list(_PUBLIC_SERIES_OUTPUT_MEASURES))


def series_output_measure_ids() -> frozenset[str]:
    return frozenset(str(item["id"]) for item in _PUBLIC_SERIES_OUTPUT_MEASURES)

__all__ = [
    "series_output_measure_ids",
    "MAX_SERIES_CHANNELS",
    "MAX_SERIES_EXPRESSION_LENGTH",
    "MAX_SERIES_PARAMETERS",
    "SERIES_HISTORY_POLICIES",
    "TIME_SERIES_OUTPUT_CONTRACT",
    "TIME_SERIES_RESULT_KIND",
    "infer_series_output_measure",
    "normalize_parameter_schema",
    "normalize_series_parameters",
    "normalize_time_series_definition",
    "parameter_variable_types",
    "resolve_series_output_measure",
    "rolling_scalar_time_series_definition",
    "series_expressions",
    "series_output_measure_catalog",
    "time_series_builtin_indicators",
]

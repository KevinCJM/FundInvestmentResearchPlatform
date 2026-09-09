"""Versioned variable catalog for custom-indicator execution contexts.

The typed compiler owns tensor mechanics.  This module owns product-facing
meaning: where a variable comes from, which products can provide it, its unit,
and the observation/availability rules that make it safe to use.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Literal

from cal_indicators.typed_types import ValueType


VARIABLE_REGISTRY_VERSION = "2.1.0"
DATA_CONTRACT_VERSION = "tushare-eod-v2"
CONTEXT_SCHEMA_VERSION = "typed-context-v2"

ContextKind = Literal["single_product", "portfolio"]
ProductKind = Literal["etf", "fund"]

SEMANTIC_ROLES = {
    "returns": "ordinary_return",
    "log_returns": "log_return",
    "adjusted_nav": "adjusted_nav_level",
    "market_open": "raw_open_price",
    "market_high": "raw_high_price",
    "market_low": "raw_low_price",
    "market_close": "raw_close_price",
    "previous_close": "raw_previous_close",
    "price_change": "raw_price_change",
    "price_return": "quote_return",
    "volume": "trading_volume",
    "turnover_amount": "turnover_amount",
    "unit_nav": "reported_unit_nav",
    "accumulated_nav": "reported_accumulated_nav",
    "accumulated_dividend": "reported_accumulated_dividend",
    "net_asset": "reported_net_asset",
    "total_net_asset": "reported_total_net_asset",
    "asset_returns": "ordinary_return_matrix",
    "asset_log_returns": "log_return_matrix",
    "portfolio_returns": "realized_portfolio_return",
    "asset_weights": "weight",
    "weight_path": "weight_path",
    "benchmark_returns": "benchmark_return",
    "annual_risk_free_rate_decimal": "annual_risk_free_rate",
    "risk_free_rate_per_observation": "observation_risk_free_rate",
    "risk_free_return_window": "window_risk_free_return",
    "periods_per_year": "annualization_factor",
}


@dataclass(frozen=True)
class VariableDefinition:
    variable_id: str
    label: str
    latex: str
    kind: str
    axes: tuple[str, ...]
    shape: tuple[str, ...]
    semantic_dimension: str
    price_basis: str | None
    unit: str
    description: str
    domains: tuple[ContextKind, ...]
    product_kinds: tuple[ProductKind, ...]
    source: str
    source_dataset: str | None = None
    source_field: str | None = None
    transform: str = "identity"
    availability_rule: str = "runtime_context"
    aliases: tuple[str, ...] = ()
    conditional: bool = False
    alternative_variables: tuple[str, ...] = ()

    def value_type(self) -> ValueType:
        kwargs = {
            "semantic_dimension": self.semantic_dimension,
            "price_basis": self.price_basis,
        }
        if self.kind == "scalar":
            return ValueType.scalar(**kwargs)
        if self.kind == "series":
            return ValueType.series(self.shape[0], **kwargs)
        if self.kind == "vector":
            return ValueType.vector(self.shape[0], **kwargs)
        if self.kind == "matrix":
            return ValueType.matrix(self.axes, self.shape, **kwargs)  # type: ignore[arg-type]
        raise ValueError(f"未知变量类型: {self.kind}")

    def to_catalog_entry(self) -> dict[str, Any]:
        value_type = self.value_type()
        semantic_role = SEMANTIC_ROLES.get(self.variable_id, self.semantic_dimension)
        availability_tier = "conditional" if self.conditional else "core"
        if "portfolio" in self.domains and "single_product" not in self.domains:
            category_id, category_label = "portfolio", "组合上下文"
        elif value_type.is_scalar:
            category_id, category_label = "configuration", "窗口与配置"
        elif self.semantic_dimension in {"volume", "currency_amount"}:
            category_id, category_label = "trading", "成交与资产"
        elif self.semantic_dimension in {"return_decimal", "rate_decimal"}:
            category_id, category_label = "returns", "收益与变化"
        else:
            category_id, category_label = "price", "净值与价格"
        source_bindings = {
            product_kind: {
                "configured": product_kind in self.product_kinds,
                "source_dataset": self.source_dataset if product_kind in self.product_kinds else None,
                "source_field": self.source_field if product_kind in self.product_kinds else None,
                "transform": self.transform if product_kind in self.product_kinds else None,
                "reason": (
                    None
                    if product_kind in self.product_kinds
                    else f"{self.label}当前没有为{'ETF' if product_kind == 'etf' else '场外公募基金'}配置真实数据源。"
                ),
            }
            for product_kind in ("etf", "fund")
        }
        return {
            "name": self.variable_id,
            "id": self.variable_id,
            "label": self.label,
            "latex": self.latex,
            "type": value_type.to_dict(),
            "structural_type": value_type.kind,
            "value_type": str(value_type),
            "dtype": value_type.dtype,
            "shape": value_type.kind,
            "symbolic_shape": list(value_type.shape),
            "axes": list(value_type.axes),
            "semantic": self.semantic_dimension,
            "semantic_role": semantic_role,
            "price_basis": self.price_basis,
            "unit": self.unit,
            "description": self.description,
            "category_id": category_id,
            "category_label": category_label,
            "domains": list(self.domains),
            "context_domains": list(self.domains),
            "context_kinds": list(self.domains),
            "product_kinds": list(self.product_kinds),
            "source": self.source,
            "source_dataset": self.source_dataset,
            "source_field": self.source_field,
            "canonical_field": self.variable_id,
            "source_bindings": source_bindings,
            "availability_policy": "runtime_required",
            "alternative_variables": list(self.alternative_variables),
            "transform": self.transform,
            "aliases": list(self.aliases),
            "availability_rule": self.availability_rule,
            "measure": self.semantic_dimension,
            "scale": self.unit,
            "timing": self.availability_rule,
            "availability_tier": availability_tier,
            "missing_policy": (
                "strict_inner_join_no_fill"
                if self.source_dataset
                else "derive_after_window_selection"
            ),
            "conditional": self.conditional,
            "availability": "conditional" if self.conditional else "available",
            "frequency": "daily" if value_type.kind in {"series", "matrix"} else "runtime",
            "data_basis": self.price_basis or self.semantic_dimension,
        }


def _series(
    variable_id: str,
    label: str,
    latex: str,
    semantic_dimension: str,
    price_basis: str | None,
    unit: str,
    description: str,
    *,
    products: tuple[ProductKind, ...] = ("etf", "fund"),
    dataset: str | None = None,
    field: str | None = None,
    transform: str = "identity",
    aliases: tuple[str, ...] = (),
    conditional: bool = False,
    alternatives: tuple[str, ...] = (),
    shape: str = "T",
) -> VariableDefinition:
    return VariableDefinition(
        variable_id,
        label,
        latex,
        "series",
        ("time",),
        (shape,),
        semantic_dimension,
        price_basis,
        unit,
        description,
        ("single_product",),
        products,
        "Tushare 本地 Parquet" if dataset else "派生上下文",
        dataset,
        field,
        transform,
        "ann_date <= as_of" if dataset == "nav" else "date <= as_of",
        aliases,
        conditional,
        alternatives,
    )


_VARIABLES = (
    VariableDefinition(
        "observation_dates", "净值观察日期", r"\mathbf{d}", "series", ("time",), ("L",),
        "date", None, "day", "与当前净值窗口逐项对齐的真实日期；不是观察序号。",
        ("single_product",), ("etf", "fund"), "当前净值窗口的日期轴",
        transform="aligned_window_dates",
    ),
    _series(
        "returns",
        "复权净值普通收益率",
        r"\mathbf{r}",
        "return_decimal",
        "adjusted_nav",
        "decimal",
        "由相邻复权净值计算的普通收益率序列，即 adjusted_nav[t] / adjusted_nav[t-1] - 1。",
        transform="adjusted_nav[t] / adjusted_nav[t-1] - 1",
        shape="T",
    ),
    _series(
        "log_returns",
        "复权净值对数收益率",
        r"\mathbf{\ell}",
        "return_decimal",
        "adjusted_nav",
        "decimal",
        "由相邻复权净值计算的对数收益率序列，即 log(adjusted_nav[t] / adjusted_nav[t-1])。",
        transform="log(adjusted_nav[t] / adjusted_nav[t-1])",
        shape="T",
    ),
    _series(
        "adjusted_nav",
        "复权净值",
        r"\mathbf{p}_{\mathrm{adj}}",
        "adjusted_nav",
        "adjusted_nav",
        "nav",
        "窗口内真实复权净值水平；用于事后研究，不宣称具备历史调整因子时点性。",
        dataset="nav",
        field="adj_nav",
        # The level path includes the boundary NAV used to derive T returns,
        # so its symbolic length is intentionally independent from returns.
        shape="L",
    ),
    _series(
        "market_open",
        "开盘价",
        r"\mathbf{o}",
        "raw_market_price",
        "raw_market",
        "price",
        "ETF 未复权日 K 开盘价；不得与复权净值做同口径价格运算。",
        products=("etf",),
        dataset="candle",
        field="open",
        aliases=("open_price",),
        alternatives=("adjusted_nav", "returns"),
    ),
    _series(
        "market_high",
        "最高价",
        r"\mathbf{h}",
        "raw_market_price",
        "raw_market",
        "price",
        "ETF 未复权日 K 最高价。",
        products=("etf",),
        dataset="candle",
        field="high",
        aliases=("high_price",),
        alternatives=("adjusted_nav", "returns"),
    ),
    _series(
        "market_low",
        "最低价",
        r"\mathbf{l}",
        "raw_market_price",
        "raw_market",
        "price",
        "ETF 未复权日 K 最低价。",
        products=("etf",),
        dataset="candle",
        field="low",
        aliases=("low_price",),
        alternatives=("adjusted_nav", "returns"),
    ),
    _series(
        "market_close",
        "收盘价",
        r"\mathbf{c}",
        "raw_market_price",
        "raw_market",
        "price",
        "ETF 未复权日 K 收盘价。",
        products=("etf",),
        dataset="candle",
        field="close",
        aliases=("close_price",),
        alternatives=("adjusted_nav", "returns"),
    ),
    _series(
        "previous_close",
        "前收盘价",
        r"\mathbf{c}_{\mathrm{prev}}",
        "raw_market_price",
        "raw_market",
        "price",
        "ETF 日 K 前收盘价。",
        products=("etf",),
        dataset="candle",
        field="pre_close",
        aliases=("pre_close_price",),
        alternatives=("adjusted_nav", "returns"),
    ),
    _series(
        "price_change",
        "价格变动额",
        r"\Delta\mathbf{c}",
        "raw_market_price",
        "raw_market",
        "price",
        "ETF 收盘价相对前收盘价的变动额。",
        products=("etf",),
        dataset="candle",
        field="change",
        alternatives=("returns",),
    ),
    _series(
        "price_return",
        "行情涨跌幅",
        r"\mathbf{r}_{\mathrm{quote}}",
        "return_decimal",
        "raw_market",
        "decimal",
        "Tushare pct_chg 除以 100 后的普通收益率。",
        products=("etf",),
        dataset="candle",
        field="pct_chg",
        transform="pct_chg / 100",
        aliases=("pct_change", "pct_change_decimal"),
        alternatives=("returns", "log_returns"),
    ),
    _series(
        "volume",
        "成交量",
        r"\mathbf{v}",
        "volume",
        None,
        "source_unit",
        "ETF 日成交量；原始单位由 Tushare 数据契约决定。",
        products=("etf",),
        dataset="candle",
        field="vol",
    ),
    _series(
        "turnover_amount",
        "成交额",
        r"\mathbf{a}",
        "currency_amount",
        None,
        "source_unit",
        "ETF 日成交额；原始单位由 Tushare 数据契约决定。",
        products=("etf",),
        dataset="candle",
        field="amount",
        aliases=("amount",),
    ),
    _series(
        "unit_nav",
        "单位净值",
        r"\mathbf{n}_{\mathrm{unit}}",
        "reported_nav",
        "unit_nav",
        "nav",
        "基金单位净值；仅在引用窗口内字段完整时可用。",
        dataset="nav",
        field="unit_nav",
        conditional=True,
    ),
    _series(
        "accumulated_nav",
        "累计净值",
        r"\mathbf{n}_{\mathrm{acc}}",
        "reported_nav",
        "accumulated_nav",
        "nav",
        "基金累计净值；仅在引用窗口内字段完整时可用。",
        dataset="nav",
        field="accum_nav",
        aliases=("accum_nav",),
        conditional=True,
    ),
    _series(
        "accumulated_dividend",
        "累计分红",
        r"\mathbf{d}_{\mathrm{acc}}",
        "currency_amount",
        None,
        "source_unit",
        "累计分红字段，数据稀疏时返回不可用，不进行填充。",
        dataset="nav",
        field="accum_div",
        aliases=("accum_div",),
        conditional=True,
    ),
    _series(
        "net_asset",
        "净资产",
        r"\mathbf{A}_{\mathrm{net}}",
        "currency_amount",
        None,
        "source_unit",
        "基金净资产；严格按公告可得日截断且不填充。",
        dataset="nav",
        field="net_asset",
        conditional=True,
    ),
    _series(
        "total_net_asset",
        "资产总额",
        r"\mathbf{A}_{\mathrm{total}}",
        "currency_amount",
        None,
        "source_unit",
        "基金资产总额；严格按公告可得日截断且不填充。",
        dataset="nav",
        field="total_netasset",
        aliases=("total_netasset",),
        conditional=True,
    ),
    VariableDefinition(
        "observation_count",
        "收益观察数",
        r"n_{\mathrm{obs}}",
        "scalar",
        (),
        (),
        "count",
        None,
        "count",
        "当前窗口实际收益观察数。",
        ("single_product",),
        ("etf", "fund"),
        "窗口派生",
        transform="len(returns)",
    ),
    VariableDefinition(
        "window_elapsed_days",
        "窗口实际自然日数",
        r"n_{\mathrm{day}}",
        "scalar",
        (),
        (),
        "calendar_days",
        None,
        "day",
        "当前窗口实际起止日之间的自然日数；不代表用户选择的周期标签。",
        ("single_product",),
        ("etf", "fund"),
        "窗口派生",
        transform="end_date - start_date",
        aliases=("window_calendar_days",),
    ),
    VariableDefinition(
        "risk_free_return_window",
        "窗口无风险累计收益",
        r"r_{f}^{\mathrm{window}}",
        "scalar",
        (),
        (),
        "rate_decimal",
        None,
        "decimal",
        "依据年化无风险利率和窗口实际自然日数换算的累计收益。",
        ("single_product", "portfolio"),
        ("etf", "fund"),
        "指标配置与窗口派生",
        transform="(1 + annual_rate) ** (elapsed_days / 365) - 1",
    ),
    VariableDefinition(
        "asset_returns", "多资产普通收益矩阵", r"\mathbf{R}", "matrix",
        ("time", "asset"), ("T", "N"), "return_decimal", "adjusted_nav", "decimal",
        "严格共同日期后的多资产普通收益矩阵。", ("portfolio",), ("etf", "fund"),
        "不可变组合运行快照",
    ),
    VariableDefinition(
        "asset_log_returns", "多资产对数收益矩阵", r"\mathbf{L}", "matrix",
        ("time", "asset"), ("T", "N"), "return_decimal", "adjusted_nav", "decimal",
        "严格共同日期后的多资产对数收益矩阵。", ("portfolio",), ("etf", "fund"),
        "不可变组合运行快照",
    ),
    VariableDefinition(
        "portfolio_returns", "组合实际收益率", r"\mathbf{r}_{\mathrm{portfolio}}", "series",
        ("time",), ("T",), "return_decimal", "adjusted_nav", "decimal",
        "按每日生效权重与当日各底层产品收益逐日汇总得到的组合收益率序列。",
        ("portfolio",), ("etf", "fund"), "不可变组合运行快照",
    ),
    VariableDefinition(
        "asset_weights", "期末资产权重向量", r"\mathbf{w}", "vector", ("asset",), ("N",),
        "dimensionless", None, "decimal",
        "当前快照末期资产权重，仅用于当前截面风险或情景估算，不代表整个历史区间权重。", ("portfolio",),
        ("etf", "fund"), "不可变组合运行快照",
    ),
    VariableDefinition(
        "weight_path", "动态权重路径", r"\mathbf{W}", "matrix", ("time", "asset"),
        ("T", "N"), "dimensionless", None, "decimal", "逐日实际生效权重路径。", ("portfolio",),
        ("etf", "fund"), "不可变组合运行快照",
    ),
    VariableDefinition(
        "benchmark_returns", "基准收益率", r"\mathbf{b}", "series", ("time",), ("T",),
        "return_decimal", "adjusted_nav", "decimal", "与研究收益使用共同日期的可选基准收益率。",
        ("portfolio",), ("etf", "fund"), "不可变组合运行快照", conditional=True,
    ),
    VariableDefinition(
        "annual_risk_free_rate_decimal", "年化无风险收益率", r"r_{f}^{\mathrm{annual}}", "scalar",
        (), (), "rate_decimal", None, "decimal", "指标配置中的年化无风险收益率。",
        ("single_product", "portfolio"), ("etf", "fund"), "指标配置",
    ),
    VariableDefinition(
        "risk_free_rate_per_observation", "单观察期无风险收益率", r"r_f", "scalar", (), (),
        "rate_decimal", None, "decimal", "按当前观察频率由年化无风险收益率换算的单期数值。",
        ("single_product", "portfolio"), ("etf", "fund"), "指标配置派生",
        aliases=("risk_free_rate_per_period",),
    ),
    VariableDefinition(
        "periods_per_year", "年化因子", r"p_{\mathrm{year}}", "scalar", (), (), "count", None,
        "count", "日频研究使用的每年观察数。", ("single_product", "portfolio"),
        ("etf", "fund"), "上下文约定",
    ),
)

VARIABLES_BY_ID = {item.variable_id: item for item in _VARIABLES}
VARIABLE_ALIASES = {
    alias: item.variable_id
    for item in _VARIABLES
    for alias in item.aliases
}

# Parse persisted formulas that predate the canonical text-style subscripts.
# These are input aliases only; catalog output always uses the forms above.
LEGACY_VARIABLE_LATEX = {
    r"r_{f}": "risk_free_rate_per_observation",
    r"\mathbf{p}_{adj}": "adjusted_nav",
    r"\mathbf{c}_{prev}": "previous_close",
    r"\mathbf{r}_{quote}": "price_return",
    r"\mathbf{n}_{unit}": "unit_nav",
    r"\mathbf{n}_{acc}": "accumulated_nav",
    r"\mathbf{d}_{acc}": "accumulated_dividend",
    r"\mathbf{A}_{net}": "net_asset",
    r"\mathbf{A}_{total}": "total_net_asset",
    r"n_{obs}": "observation_count",
    r"n_{day}": "window_elapsed_days",
    r"r_{f}^{window}": "risk_free_return_window",
    r"r_{f}^{annual}": "annual_risk_free_rate_decimal",
    r"p_{year}": "periods_per_year",
}
LEGACY_TYPED_VARIABLE_IDS = frozenset(
    {
        "returns",
        "log_returns",
        "asset_returns",
        "asset_log_returns",
        "asset_weights",
        "weight_path",
        "benchmark_returns",
        "annual_risk_free_rate_decimal",
        "risk_free_rate_per_observation",
        "periods_per_year",
    }
)


def canonical_variable_id(variable_id: str) -> str:
    return VARIABLE_ALIASES.get(variable_id, variable_id)


def get_variable(variable_id: str) -> VariableDefinition | None:
    return VARIABLES_BY_ID.get(canonical_variable_id(variable_id))


def variable_semantic_role(variable_id: str) -> str | None:
    canonical = canonical_variable_id(variable_id)
    definition = VARIABLES_BY_ID.get(canonical)
    return SEMANTIC_ROLES.get(canonical, definition.semantic_dimension if definition else None)


def variable_types(
    context: ContextKind | None = None,
    dsl_version: str = VARIABLE_REGISTRY_VERSION,
) -> dict[str, ValueType]:
    return {
        item.variable_id: item.value_type()
        for item in _VARIABLES
        if context is None or context in item.domains
        if dsl_version != "2.0.0" or item.variable_id in LEGACY_TYPED_VARIABLE_IDS
    }


def allowed_variables(context: ContextKind) -> set[str]:
    return {
        item.variable_id
        for item in _VARIABLES
        if context in item.domains
    }


def variable_catalog(context: ContextKind | None = None) -> list[dict[str, Any]]:
    return [
        item.to_catalog_entry()
        for item in _VARIABLES
        if context is None or context in item.domains
    ]


def variable_latex_symbols(context: ContextKind | None = None) -> dict[str, str]:
    """Return the canonical mathematical symbol for each executable variable."""

    return {
        item.variable_id: item.latex
        for item in _VARIABLES
        if context is None or context in item.domains
    }


def normalize_variable_latex(expression: str) -> str:
    normalized = expression
    # Stored typed-v2.0 formulas may still use an older variable identifier.
    # Canonicalize identifiers at runtime without rewriting persisted history.
    for alias, canonical in sorted(
        VARIABLE_ALIASES.items(), key=lambda item: len(item[0]), reverse=True
    ):
        normalized = re.sub(rf"\b{re.escape(alias)}\b", canonical, normalized)
    latex_aliases = {
        **LEGACY_VARIABLE_LATEX,
        **{item.latex: item.variable_id for item in _VARIABLES},
    }
    for latex, variable_id in sorted(
        latex_aliases.items(), key=lambda item: len(item[0]), reverse=True
    ):
        if latex.isidentifier():
            # Plain symbols such as r_f must not rewrite parts of unknown
            # identifiers (r_future / custom_r_f), hiding a real input error.
            normalized = re.sub(rf"\b{re.escape(latex)}\b", variable_id, normalized)
        else:
            normalized = normalized.replace(latex, variable_id)
    return normalized


def canonicalize_variables(variable_ids: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(canonical_variable_id(str(item)) for item in variable_ids))


__all__ = [
    "CONTEXT_SCHEMA_VERSION",
    "DATA_CONTRACT_VERSION",
    "LEGACY_TYPED_VARIABLE_IDS",
    "VARIABLE_REGISTRY_VERSION",
    "VariableDefinition",
    "allowed_variables",
    "canonical_variable_id",
    "canonicalize_variables",
    "get_variable",
    "normalize_variable_latex",
    "variable_latex_symbols",
    "variable_catalog",
    "variable_semantic_role",
    "variable_types",
]

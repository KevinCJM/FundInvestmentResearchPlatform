"""Application service for compiling, storing, and evaluating custom indicators."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import threading
import time
from collections import OrderedDict
from contextlib import ExitStack
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
import numba

try:
    from backend.market_data import resolve_tushare_data_dir
except ModuleNotFoundError:  # pragma: no cover - backend/ direct execution
    from market_data import resolve_tushare_data_dir

from cal_indicators.indicator_runtime import IndicatorRuntime
from cal_indicators.builtin_batch_kernel import (
    BUILTIN_METRIC_CODE,
    STATUS_DIVIDE_BY_ZERO,
    STATUS_DOMAIN_ERROR,
    STATUS_INSUFFICIENT_SAMPLE,
    STATUS_NON_FINITE_RESULT,
    STATUS_OK,
)
from cal_indicators.latex_excutor import DAGBuildError, LatexParseError
from cal_indicators.typed_dsl import (
    TypedDslError,
    TypedExpressionParser,
    TypedExpressionPlan,
    TypedIndicatorRuntime,
    compose_typed_expression,
)
from cal_indicators.typed_latex import (
    MATH_NOTATION_VERSION,
    render_python_expression_latex,
)
from cal_indicators.typed_numeric_backend import numeric_backend_status
from cal_indicators.typed_numba_kernels import (
    ENGINE_VERSION as NUMBA_ENGINE_VERSION,
    NUMERIC_KERNEL_VERSION,
    kernel_registry_status,
    warm_numba_kernel_registry,
)
from cal_indicators.typed_numba_plan import (
    NumbaPlanCompileError,
    compile_numba_batch_plan,
    compile_numba_plan,
    get_cached_numba_batch_plan,
    persist_numba_plan,
    plan_cache_status,
)
from cal_indicators.typed_operators import (
    COMPAT_OPERATOR_REGISTRY_VERSION,
    COMPAT_TYPED_DSL_VERSION,
    TYPED_DSL_VERSION,
    TYPED_OPERATOR_REGISTRY_VERSION,
)

from .errors import ConflictError, IndicatorDomainError, ValidationError
from .parallel_engine import AdaptiveComputeEngine, SharedArrayOwner, score_matrix
from .portfolio_repository import PortfolioRunRepository
from .presentation import (
    INDICATOR_TYPE_LABELS,
    catalog_status,
    indicator_type,
    metric_presentation,
    ui_exposed,
)
from .periods import SUPPORTED_PERIODS, period_cache_reference, period_metadata
from .repository import (
    IndicatorRepository,
    PlanRepository,
    SnapshotIndicatorConfigRepository,
)
from .snapshot_config import MAX_SNAPSHOT_INDICATORS, normalized_snapshot_item
from .run_result_repository import EvaluationRunResultRepository
from .series_provider import (
    DEFAULT_DATA_DIR,
    PeriodWindow,
    ProductSeries,
    ProductVariableSeries,
    VariablePeriodWindow,
    load_adjusted_product_series,
    load_product_series,
    load_product_variable_series,
    load_product_variable_series_batch,
    market_data_generation,
    prepare_variable_window_index,
    select_period_window,
    select_variable_window,
    select_variable_window_fast,
)
from .typed_service import compose_expression, infer_expression, typed_product_meta
from .variable_registry import (
    CONTEXT_SCHEMA_VERSION,
    DATA_CONTRACT_VERSION,
    VARIABLE_REGISTRY_VERSION,
    canonicalize_variables,
    get_variable,
    normalize_variable_latex,
    variable_catalog,
    variable_latex_symbols,
    variable_types,
)


ENGINE_VERSION = NUMBA_ENGINE_VERSION
LEGACY_DSL_VERSION = "1.0.0"
LEGACY_OPERATOR_REGISTRY_VERSION = "legacy-v1"
LEGACY_TYPED_DSL_VERSION = "2.0.0"
LEGACY_TYPED_OPERATOR_REGISTRY_VERSION = "2.0.0"
COMPAT_TYPED_OPERATOR_REGISTRY_VERSION = COMPAT_OPERATOR_REGISTRY_VERSION
NUMBA_V3_MIGRATION_MARKER = "typed-numba-3"
MAX_FORMULA_LENGTH = 1000
MAX_DAG_NODES = 128
MAX_DAG_DEPTH = 20
MAX_INDICATORS = 10
MAX_TARGETS = 50
MAX_COMBINATIONS = 500
MAX_PLAN_TARGETS = 50_000
MAX_ROLLING_COMBINATIONS = 10
MAX_SERIES_OBSERVATIONS = 5000
MAX_ROLLING_POINTS = 500
_WINDOW_NOT_SELECTED = object()
PLAN_PRODUCT_FILTER_KEYS = (
    "fund_type",
    "invest_type",
    "market",
    "status",
    "management",
    "custodian",
)
PLAN_PRODUCT_CONDITION_OPERATORS = {"gte", "lte", "gt", "lt", "eq"}


def _empty_plan_product_selection() -> dict[str, Any]:
    return {
        "query": "",
        "filters": {key: [] for key in PLAN_PRODUCT_FILTER_KEYS},
        "conditions": [],
        "selection_mode": "manual",
    }


@lru_cache(maxsize=512)
def _compile_typed_plan(
    expression: str,
    context_kind: str,
    dsl_version: str,
    operator_registry_version: str,
) -> TypedExpressionPlan:
    """Cache immutable typed plans; each request still gets its own trace runtime."""

    return compose_typed_expression(
        expression,
        variable_types=variable_types(context_kind, dsl_version),  # type: ignore[arg-type]
        output_contract="scalar",
        dsl_version=dsl_version,
        operator_registry_version=operator_registry_version,
        max_nodes=MAX_DAG_NODES,
        max_depth=MAX_DAG_DEPTH,
    )


@lru_cache(maxsize=1024)
def _typed_display_latex(expression: str, context_kind: str) -> str:
    """Render presentation LaTeX without compiling or evaluating the formula."""

    normalized = normalize_variable_latex(expression)
    symbols = variable_latex_symbols(context_kind)  # type: ignore[arg-type]
    parser = TypedExpressionParser(tuple(symbols))
    python_expression = parser.to_python(normalized)
    return render_python_expression_latex(python_expression, symbols)


def _built_in_indicators() -> list[dict[str, Any]]:
    timestamp = "2025-10-19T00:00:00+00:00"
    legacy_common = {
        "revision": 1,
        "source": "built_in",
        "read_only": True,
        "periods": list(SUPPORTED_PERIODS),
        "period_policy": "all_supported",
        "annual_risk_free_rate_percent": 1.5,
        "created_at": timestamp,
        "updated_at": timestamp,
        "dsl_version": LEGACY_DSL_VERSION,
        "operator_registry_version": LEGACY_OPERATOR_REGISTRY_VERSION,
        "context_kind": "single_product",
        "output_contract": "scalar",
        "output_measure": "dimensionless",
        "template_origin": None,
        "required_variables": ["returns"],
        "applicable_product_kinds": ["etf", "fund"],
        "availability_status": "ready",
    }
    portfolio_compatibility_common = {
        "revision": 1,
        "source": "built_in",
        "read_only": True,
        "periods": list(SUPPORTED_PERIODS),
        "period_policy": "all_supported",
        "annual_risk_free_rate_percent": 1.5,
        "created_at": timestamp,
        "updated_at": timestamp,
        "dsl_version": LEGACY_TYPED_DSL_VERSION,
        "operator_registry_version": LEGACY_TYPED_OPERATOR_REGISTRY_VERSION,
        "context_kind": "portfolio",
        "output_contract": "scalar",
        "output_measure": "dimensionless",
        "required_variables": ["asset_returns", "asset_weights"],
        "applicable_product_kinds": ["portfolio"],
        "availability_status": "ready",
        "catalog_status_override": "compatibility",
        "ui_exposed_override": False,
    }
    portfolio_common = {
        "revision": 1,
        "source": "built_in",
        "read_only": True,
        "periods": list(SUPPORTED_PERIODS),
        "period_policy": "all_supported",
        "annual_risk_free_rate_percent": 1.5,
        "created_at": timestamp,
        "updated_at": timestamp,
        "dsl_version": TYPED_DSL_VERSION,
        "operator_registry_version": TYPED_OPERATOR_REGISTRY_VERSION,
        "numeric_kernel_version": NUMERIC_KERNEL_VERSION,
        "variable_registry_version": VARIABLE_REGISTRY_VERSION,
        "data_contract_version": DATA_CONTRACT_VERSION,
        "context_schema_version": CONTEXT_SCHEMA_VERSION,
        "context_kind": "portfolio",
        "output_contract": "scalar",
        "output_measure": "dimensionless",
        "required_variables": ["portfolio_returns"],
        "applicable_product_kinds": ["portfolio"],
        "availability_status": "ready",
        "formula_version": TYPED_DSL_VERSION,
        "data_basis": "运行快照中的真实底层产品收益、每日生效权重与严格共同日期",
        "semantic_differences": [
            "组合历史收益使用每日生效权重，不将期末权重回填至历史",
            "非有限结果和样本不足返回诊断，不静默置零",
        ],
    }
    items = [
        {
            **legacy_common,
            "id": "builtin-cumulative-return",
            "indicator_type": "return",
            "name": "累计收益率",
            "description": "窗口内每日收益复合后的累计收益。",
            "expression": r"\left(\prod\left(\mathbf{r} + 1\right)\right) - 1",
            "unit": "%",
            "display_format": "percent",
            "precision": 2,
            "direction": "higher_better",
            "methodology": "逐期增长因子累乘后减一。",
            "output_measure": "return_decimal",
        },
        {
            **legacy_common,
            "id": "builtin-volatility",
            "indicator_type": "risk",
            "name": "收益波动率",
            "description": "窗口内日收益率的样本标准差。",
            "expression": r"\operatorname{std}\left(\mathbf{r}, 1\right)",
            "unit": "%",
            "display_format": "percent",
            "precision": 2,
            "direction": "lower_better",
            "methodology": "普通收益率序列的样本标准差（ddof=1）。",
            "output_measure": "return_decimal",
        },
        {
            **legacy_common,
            "id": "builtin-return-risk-ratio",
            "indicator_type": "risk_adjusted",
            "name": "收益风险比",
            "description": "累计收益率与窗口收益波动率之比。",
            "expression": (
                r"\frac{\left(\prod\left(\mathbf{r} + 1\right)\right) - 1}"
                r"{\operatorname{std}\left(\mathbf{r}, 1\right)}"
            ),
            "unit": "",
            "display_format": "number",
            "precision": 3,
            "direction": "higher_better",
            "methodology": "窗口累计收益除以普通收益率样本标准差。",
        },
        {
            **portfolio_compatibility_common,
            "id": "builtin-portfolio-cumulative-return",
            "indicator_type": "return",
            "name": "固定期末权重组合累计收益率（兼容）",
            "description": "兼容旧公式：将快照期末权重应用于整段历史收益矩阵。该口径不代表真实组合历史表现。",
            "expression": (
                r"\prod\left(\operatorname{matvec}\left(\mathbf{R},"
                r"\mathbf{w}\right)+1\right)-1"
            ),
            "unit": "%",
            "display_format": "percent",
            "precision": 2,
            "direction": "higher_better",
            "methodology": "旧版固定期末权重口径，仅用于复现历史结果。",
            "output_measure": "return_decimal",
            "template_origin": {
                "template_id": "portfolio-cumulative-return",
                "template_version": 1,
                "bindings": [
                    {
                        "parameter": "returns_matrix",
                        "source": "variable",
                        "value": "asset_returns",
                    },
                    {
                        "parameter": "weights",
                        "source": "variable",
                        "value": "asset_weights",
                    },
                ],
                "detached": False,
            },
        },
        {
            **portfolio_compatibility_common,
            "id": "builtin-portfolio-volatility",
            "indicator_type": "risk",
            "name": "固定期末权重组合波动率（兼容）",
            "description": "兼容旧公式：用快照期末权重和历史协方差估算当前截面波动，不代表真实组合历史波动。",
            "expression": (
                r"\sqrt{\operatorname{dot}\left(\mathbf{w},"
                r"\operatorname{matvec}\left(\operatorname{covariance}"
                r"\left(\mathbf{R}\right),\mathbf{w}\right)\right)}"
            ),
            "unit": "%",
            "display_format": "percent",
            "precision": 2,
            "direction": "lower_better",
            "methodology": "旧版期末权重协方差二次型口径，仅用于复现历史结果。",
            "output_measure": "return_decimal",
            "template_origin": {
                "template_id": "portfolio-volatility",
                "template_version": 1,
                "bindings": [
                    {
                        "parameter": "returns_matrix",
                        "source": "variable",
                        "value": "asset_returns",
                    },
                    {
                        "parameter": "weights",
                        "source": "variable",
                        "value": "asset_weights",
                    },
                ],
                "detached": False,
            },
        },
        {
            **portfolio_common,
            "id": "builtin-portfolio-realized-cumulative-return",
            "indicator_type": "return",
            "name": "组合累计收益率",
            "description": "根据每日生效权重与底层产品当日收益形成的真实组合收益序列计算累计收益率。",
            "expression": r"\prod\left(\mathbf{r}_{\mathrm{portfolio}}+1\right)-1",
            "unit": "%",
            "display_format": "percent",
            "precision": 2,
            "direction": "higher_better",
            "methodology": "每日先按当日生效权重汇总底层产品收益，再对逐日组合增长因子累乘。",
            "output_measure": "return_decimal",
            "minimum_observations": 1,
            "template_origin": None,
        },
        {
            **portfolio_common,
            "id": "builtin-portfolio-realized-volatility",
            "indicator_type": "risk",
            "name": "组合波动率",
            "description": "组合真实逐日收益率的样本标准差，已反映持仓市值变化和调仓后的每日权重。",
            "expression": r"\operatorname{std}\left(\mathbf{r}_{\mathrm{portfolio}},1\right)",
            "unit": "%",
            "display_format": "percent",
            "precision": 2,
            "direction": "lower_better",
            "methodology": "对运行快照中的真实逐日组合收益率计算样本标准差（ddof=1）。",
            "output_measure": "return_decimal",
            "minimum_observations": 2,
            "template_origin": None,
        },
    ]
    return items + _typed_builtin_indicators(timestamp)


def _typed_builtin_indicators(timestamp: str) -> list[dict[str, Any]]:
    """First scalar-only v2.1 research catalog.

    Names describe research results, while every formula is composed only from
    versioned mathematical primitives. Product applicability and required
    variables are explicit so clients can filter before attempting a run.
    """

    common = {
        "revision": 1,
        "source": "built_in",
        "read_only": True,
        "periods": list(SUPPORTED_PERIODS),
        "period_policy": "all_supported",
        "annual_risk_free_rate_percent": 1.5,
        "created_at": timestamp,
        "updated_at": timestamp,
        "dsl_version": TYPED_DSL_VERSION,
        "operator_registry_version": TYPED_OPERATOR_REGISTRY_VERSION,
        "numeric_kernel_version": NUMERIC_KERNEL_VERSION,
        "variable_registry_version": VARIABLE_REGISTRY_VERSION,
        "data_contract_version": DATA_CONTRACT_VERSION,
        "context_schema_version": CONTEXT_SCHEMA_VERSION,
        "context_kind": "single_product",
        "output_contract": "scalar",
        "template_origin": None,
        "availability_status": "ready",
    }
    drawdown = "drawdown_series(adjusted_nav)"
    minimum_drawdown = f"minimum(min_value({drawdown}), 0)"
    maximum_drawdown = f"absolute({minimum_drawdown})"
    annualized_return = "power(product(returns + 1), periods_per_year / observation_count) - 1"
    excess = "returns - risk_free_rate_per_observation"
    downside_deviation = f"sqrt(mean_where(power({excess}, 2), less_than({excess}, 0)))"
    upside_deviation = f"sqrt(mean_where(power({excess}, 2), greater_than({excess}, 0)))"
    locked_specs: tuple[tuple[Any, ...], ...] = (
        # 收益条件（10）
        ("total-return-v2", "累计收益率", "product(returns + 1) - 1", ["returns"], "逐期普通收益增长因子累乘后减一。", "%", "percent", 2, "higher_better", ["etf", "fund"]),
        ("annualized-return-v2", "年化收益率", annualized_return, ["returns", "periods_per_year", "observation_count"], "按实际收益观察数和年化因子折算复合收益。", "%", "percent", 2, "higher_better", ["etf", "fund"]),
        ("mean-return-v2", "平均收益率", "mean(returns)", ["returns"], "普通收益率序列的算术平均值。", "%", "percent", 3, "higher_better", ["etf", "fund"]),
        ("median-return-v2", "中位数收益率", "median(returns)", ["returns"], "普通收益率经验分布的中位数。", "%", "percent", 3, "higher_better", ["etf", "fund"]),
        ("maximum-gain-v2", "最大单期上涨", "max_value(returns)", ["returns"], "窗口内普通收益率最大值。", "%", "percent", 2, "higher_better", ["etf", "fund"]),
        ("maximum-loss-v2", "最大单期下跌", "absolute(minimum(min_value(returns), 0))", ["returns"], "窗口内最小普通收益率的非负损失幅度。", "%", "percent", 2, "lower_better", ["etf", "fund"]),
        ("positive-return-ratio-v2", "上涨期占比", "count_true(greater_than(returns, 0)) / observation_count", ["returns", "observation_count"], "正收益观察数除以窗口收益观察数。", "%", "percent", 2, "higher_better", ["etf", "fund"]),
        ("mean-positive-return-v2", "上涨期平均收益", "mean_where(returns, greater_than(returns, 0))", ["returns"], "仅对大于零的普通收益率求算术平均。", "%", "percent", 3, "higher_better", ["etf", "fund"]),
        ("mean-negative-return-v2", "下跌期平均收益", "mean_where(returns, less_than(returns, 0))", ["returns"], "仅对小于零的普通收益率求算术平均。", "%", "percent", 3, "higher_better", ["etf", "fund"]),
        ("payoff-ratio-v2", "平均盈亏比", "mean_where(returns, greater_than(returns, 0)) / absolute(mean_where(returns, less_than(returns, 0)))", ["returns"], "平均正收益除以平均负收益绝对值。", "", "number", 3, "higher_better", ["etf", "fund"]),
        # 分布风险（10）
        ("return-volatility-v2", "波动率", "std(returns, 1)", ["returns"], "普通收益率序列的样本标准差。", "%", "percent", 3, "lower_better", ["etf", "fund"]),
        ("annualized-volatility-v2", "年化波动率", "std(returns, 1) * sqrt(periods_per_year)", ["returns", "periods_per_year"], "样本标准差乘以年化因子平方根。", "%", "percent", 2, "lower_better", ["etf", "fund"]),
        ("return-mad-v2", "平均绝对偏差", "mean_absolute_deviation(returns)", ["returns"], "收益率相对其均值的绝对偏差平均值。", "%", "percent", 3, "lower_better", ["etf", "fund"]),
        ("return-range-v2", "收益区间", "max_value(returns) - min_value(returns)", ["returns"], "窗口内最大与最小普通收益率之差。", "%", "percent", 3, "lower_better", ["etf", "fund"]),
        ("return-skewness-v2", "偏度", "skewness(returns)", ["returns"], "经有限样本修正的普通收益率偏度。", "", "number", 3, "higher_better", ["etf", "fund"]),
        ("return-excess-kurtosis-v2", "超额峰度", "excess_kurtosis(returns)", ["returns"], "经有限样本修正的普通收益率超额峰度。", "", "number", 3, "lower_better", ["etf", "fund"]),
        ("historical-var-95-v2", "历史 VaR 95%", "negate(quantile(returns, 0.05))", ["returns"], "普通收益率 5% 分位数取负，按正损失展示。", "%", "percent", 3, "lower_better", ["etf", "fund"]),
        ("historical-cvar-95-v2", "历史 CVaR 95%", "negate(mean_where(returns, less_equal(returns, quantile(returns, 0.05))))", ["returns"], "不高于 5% 分位数的普通收益率均值取负。", "%", "percent", 3, "lower_better", ["etf", "fund"]),
        ("downside-deviation-v2", "下行偏差", downside_deviation, ["returns", "risk_free_rate_per_observation"], "低于单期无风险收益部分的均方根。", "%", "percent", 3, "lower_better", ["etf", "fund"]),
        ("upside-deviation-v2", "上行偏差", upside_deviation, ["returns", "risk_free_rate_per_observation"], "高于单期无风险收益部分的均方根。", "%", "percent", 3, "higher_better", ["etf", "fund"]),
        # 风险调整与路径（10）
        ("annualized-sharpe-v2", "年化夏普比率", "((mean(returns) - risk_free_rate_per_observation) / std(returns, 1)) * sqrt(periods_per_year)", ["returns", "risk_free_rate_per_observation", "periods_per_year"], "单期平均超额收益与波动率之比乘年化因子平方根。", "", "number", 3, "higher_better", ["etf", "fund"]),
        ("annualized-sortino-v2", "Sortino 比率", f"((mean(returns) - risk_free_rate_per_observation) / {downside_deviation}) * sqrt(periods_per_year)", ["returns", "risk_free_rate_per_observation", "periods_per_year"], "单期平均超额收益与下行偏差之比年化。", "", "number", 3, "higher_better", ["etf", "fund"]),
        ("omega-ratio-v2", "Omega 比率", f"sum_where({excess}, greater_than({excess}, 0)) / absolute(sum_where({excess}, less_than({excess}, 0)))", ["returns", "risk_free_rate_per_observation"], "阈值以上超额收益之和除以阈值以下超额收益绝对值之和。", "", "number", 3, "higher_better", ["etf", "fund"]),
        ("tail-ratio-95-v2", "尾部比率", "quantile(returns, 0.95) / absolute(quantile(returns, 0.05))", ["returns"], "95% 分位收益除以 5% 分位收益绝对值。", "", "number", 3, "higher_better", ["etf", "fund"]),
        ("maximum-drawdown-v2", "最大回撤", maximum_drawdown, ["adjusted_nav"], "复权净值回撤序列的最小负值先与零取较小值，再取绝对值。", "%", "percent", 2, "lower_better", ["etf", "fund"]),
        ("ulcer-index-v2", "Ulcer 指数", f"sqrt(mean(power(minimum({drawdown}, 0), 2)))", ["adjusted_nav"], "复权净值负回撤序列平方均值的平方根；正值统一截为零。", "%", "percent", 3, "lower_better", ["etf", "fund"]),
        ("calmar-ratio-v2", "Calmar 比率", f"({annualized_return}) / {maximum_drawdown}", ["returns", "periods_per_year", "observation_count", "adjusted_nav"], "年化复合收益除以基于复权净值路径计算的最大回撤。", "", "number", 3, "higher_better", ["etf", "fund"]),
        ("new-high-ratio-v2", "创新高比例", "count_true(new_high_mask(adjusted_nav)) / length(adjusted_nav)", ["adjusted_nav"], "严格超过此前历史峰值的观察数占净值观察数的比例；首个观察值计一次，相同峰值不重复计数。", "%", "percent", 2, "higher_better", ["etf", "fund"]),
        ("adjusted-nav-slope-v2", "净值线性斜率", "linear_slope(adjusted_nav)", ["adjusted_nav"], "复权净值对观察序号的一元线性回归斜率。", "", "number", 6, "higher_better", ["etf", "fund"]),
        ("adjusted-nav-r-squared-v2", "净值线性拟合度", "linear_r_squared(adjusted_nav)", ["adjusted_nav"], "复权净值对观察序号线性回归的决定系数。", "", "number", 3, "higher_better", ["etf", "fund"]),
        # ETF 行情与成交（5）
        ("average-volume-v2", "平均成交量", "mean(volume)", ["volume"], "ETF 窗口内日成交量算术平均值。", "", "number", 2, "higher_better", ["etf"]),
        ("volume-volatility-v2", "成交量波动率", "std(volume, 1)", ["volume"], "ETF 窗口内日成交量样本标准差。", "", "number", 2, "lower_better", ["etf"]),
        ("highest-market-price-v2", "最高价", "max_value(market_high)", ["market_high"], "ETF 窗口内原始日最高价的最大值。", "", "number", 4, "higher_better", ["etf"]),
        ("lowest-market-price-v2", "最低价", "min_value(market_low)", ["market_low"], "ETF 窗口内原始日最低价的最小值。", "", "number", 4, "higher_better", ["etf"]),
        ("market-high-low-range-v2", "高低价区间", "max_value(market_high) - min_value(market_low)", ["market_high", "market_low"], "ETF 窗口最高价与最低价之差。", "", "number", 4, "lower_better", ["etf"]),
    )
    measure_overrides = {
        "positive-return-ratio-v2": "dimensionless",
        "new-high-ratio-v2": "dimensionless",
        "average-volume-v2": "volume",
        "volume-volatility-v2": "volume",
        "highest-market-price-v2": "raw_market_price",
        "lowest-market-price-v2": "raw_market_price",
        "market-high-low-range-v2": "raw_market_price",
        "adjusted-nav-slope-v2": "derived:adjusted_nav/count",
    }
    minimum_observations = {
        "return-volatility-v2": 2,
        "annualized-volatility-v2": 2,
        "return-skewness-v2": 3,
        "return-excess-kurtosis-v2": 4,
        "annualized-sharpe-v2": 2,
        "annualized-sortino-v2": 2,
        "adjusted-nav-slope-v2": 2,
        "adjusted-nav-r-squared-v2": 2,
        "volume-volatility-v2": 2,
    }
    type_by_id = {
        "total-return-v2": "return",
        "annualized-return-v2": "return",
        "mean-return-v2": "return",
        "median-return-v2": "return",
        "maximum-gain-v2": "return",
        "maximum-loss-v2": "risk",
        "positive-return-ratio-v2": "return",
        "mean-positive-return-v2": "return",
        "mean-negative-return-v2": "risk",
        "payoff-ratio-v2": "risk_adjusted",
        "return-volatility-v2": "risk",
        "annualized-volatility-v2": "risk",
        "return-mad-v2": "risk",
        "return-range-v2": "risk",
        "return-skewness-v2": "risk",
        "return-excess-kurtosis-v2": "risk",
        "historical-var-95-v2": "risk",
        "historical-cvar-95-v2": "risk",
        "downside-deviation-v2": "risk",
        "upside-deviation-v2": "risk",
        "annualized-sharpe-v2": "risk_adjusted",
        "annualized-sortino-v2": "risk_adjusted",
        "omega-ratio-v2": "risk_adjusted",
        "tail-ratio-95-v2": "risk_adjusted",
        "maximum-drawdown-v2": "path",
        "ulcer-index-v2": "path",
        "calmar-ratio-v2": "risk_adjusted",
        "new-high-ratio-v2": "path",
        "adjusted-nav-slope-v2": "path",
        "adjusted-nav-r-squared-v2": "path",
        "average-volume-v2": "market_liquidity",
        "volume-volatility-v2": "market_liquidity",
        "highest-market-price-v2": "market_liquidity",
        "lowest-market-price-v2": "market_liquidity",
        "market-high-low-range-v2": "market_liquidity",
    }
    return [
        {
            **common,
            "id": f"builtin-{item[0]}",
            "name": item[1],
            "description": item[4],
            "expression": item[2],
            "required_variables": item[3],
            "methodology": item[4],
            "formula_version": TYPED_DSL_VERSION,
            "minimum_observations": minimum_observations.get(item[0], 1),
            "data_basis": "真实数据、严格窗口、缺失不填充",
            "metrics_factory_reference": "MetricsFactory 区间标量指标审计；公式已独立复核",
            "semantic_differences": [
                "非有限、除零和样本不足返回诊断，不静默置零",
                "最大回撤按非负损失幅度展示",
                "条件均值只对满足条件的样本归约",
            ],
            "indicator_type": type_by_id[item[0]],
            "category_id": type_by_id[item[0]],
            "category_label": INDICATOR_TYPE_LABELS[type_by_id[item[0]]],
            "unit": item[5],
            "display_format": item[6],
            "precision": item[7],
            "direction": item[8],
            "applicable_product_kinds": item[9],
            "output_measure": measure_overrides.get(
                item[0], "return_decimal" if item[6] == "percent" else "dimensionless"
            ),
        }
        for index, item in enumerate(locked_specs)
    ]


class BoundedTTLCache:
    def __init__(self, max_size: int = 512, ttl_seconds: int = 300) -> None:
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._items: OrderedDict[str, tuple[float, dict[str, Any]]] = OrderedDict()
        self._lock = threading.RLock()

    def get(self, key: str) -> Optional[dict[str, Any]]:
        now = time.monotonic()
        with self._lock:
            item = self._items.get(key)
            if item is None:
                return None
            expires_at, value = item
            if expires_at <= now:
                self._items.pop(key, None)
                return None
            self._items.move_to_end(key)
            return copy.deepcopy(value)

    def put(self, key: str, value: dict[str, Any]) -> None:
        with self._lock:
            self._items[key] = (time.monotonic() + self.ttl_seconds, copy.deepcopy(value))
            self._items.move_to_end(key)
            while len(self._items) > self.max_size:
                self._items.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()


class BoundedObjectTTLCache:
    """Byte-bounded cache for immutable batch arrays and window objects."""

    def __init__(self, max_bytes: int, ttl_seconds: int = 300) -> None:
        self.max_bytes = max(1, int(max_bytes))
        self.ttl_seconds = max(1, int(ttl_seconds))
        self._items: OrderedDict[str, tuple[float, int, Any]] = OrderedDict()
        self._bytes = 0
        self._lock = threading.RLock()

    def get(self, key: str) -> Any | None:
        now = time.monotonic()
        with self._lock:
            item = self._items.get(key)
            if item is None:
                return None
            expires_at, size, value = item
            if expires_at <= now:
                self._items.pop(key, None)
                self._bytes -= size
                return None
            self._items.move_to_end(key)
            return value

    def put(self, key: str, value: Any, size: int) -> None:
        item_size = max(1, int(size))
        if item_size > self.max_bytes:
            return
        with self._lock:
            previous = self._items.pop(key, None)
            if previous is not None:
                self._bytes -= previous[1]
            self._items[key] = (
                time.monotonic() + self.ttl_seconds,
                item_size,
                value,
            )
            self._bytes += item_size
            while self._items and self._bytes > self.max_bytes:
                _, (_, removed_size, _) = self._items.popitem(last=False)
                self._bytes -= removed_size

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self._bytes = 0

    def status(self) -> dict[str, int]:
        with self._lock:
            return {"entries": len(self._items), "bytes": self._bytes}


class CustomIndicatorService:
    def __init__(
        self,
        workspace_data_dir: Optional[Path] = None,
        market_data_dir: Optional[Path] = None,
        cache: Optional[BoundedTTLCache] = None,
    ) -> None:
        configured = os.getenv("CUSTOM_INDICATOR_DATA_DIR")
        self.workspace_data_dir = workspace_data_dir or (Path(configured) if configured else DEFAULT_DATA_DIR)
        self.market_data_dir = market_data_dir or DEFAULT_DATA_DIR
        self.indicators = IndicatorRepository(
            self.workspace_data_dir / "custom_indicators.json",
            _built_in_indicators(),
        )
        self.plans = PlanRepository(self.workspace_data_dir / "evaluation_plans.json")
        self.snapshot_config = SnapshotIndicatorConfigRepository(
            self.workspace_data_dir / "snapshot_indicator_config.json"
        )
        self.portfolio_runs = PortfolioRunRepository(self.workspace_data_dir / "portfolio_runs.json")
        self.cache = cache or BoundedTTLCache()
        self.plan_cache = BoundedTTLCache(
            max_size=max(1, int(os.getenv("INDICATOR_PLAN_CACHE_ENTRIES", "32"))),
            ttl_seconds=max(1, int(os.getenv("INDICATOR_CACHE_TTL_SECONDS", "300"))),
        )
        cache_ttl = max(1, int(os.getenv("INDICATOR_CACHE_TTL_SECONDS", "300")))
        self.data_cache = BoundedObjectTTLCache(
            int(os.getenv("INDICATOR_DATA_CACHE_BYTES", str(512 * 1024 * 1024))),
            cache_ttl,
        )
        self.window_cache = BoundedObjectTTLCache(
            int(os.getenv("INDICATOR_WINDOW_CACHE_BYTES", str(256 * 1024 * 1024))),
            cache_ttl,
        )
        self.compute_engine = AdaptiveComputeEngine(
            self.workspace_data_dir / ".indicator_runtime"
        )
        self._startup_warmup: dict[str, Any] = {
            "complete": False,
            "indicator_plans": 0,
            "single_metric_batch_plans": 0,
            "evaluation_batch_plans": 0,
        }
        self.run_results = EvaluationRunResultRepository(
            self.workspace_data_dir / ".evaluation_run_cache",
            ttl_seconds=max(
                1, int(os.getenv("INDICATOR_RESULT_TTL_SECONDS", "1800"))
            ),
            max_runs=max(1, int(os.getenv("INDICATOR_RESULT_MAX_RUNS", "20"))),
            max_bytes=max(
                1,
                int(
                    os.getenv(
                        "INDICATOR_RESULT_MAX_BYTES", str(2 * 1024 * 1024 * 1024)
                    )
                ),
            ),
        )
        self.run_results.cleanup()
        self.migration = self._apply_numba_v3_migration()

    def warm_numba_plans(self) -> dict[str, Any]:
        """Compile every persisted indicator plan without starting workers.

        Production calls this from ``start_compute_engine`` before the HTTP
        application becomes ready.  Keeping plan warmup separate also lets
        router-level tests honor the same no-request-compilation contract
        without creating a process pool for every isolated test app.
        """
        kernel_status = warm_numba_kernel_registry()
        failures: list[str] = []
        runtime_root = self.workspace_data_dir / ".indicator_runtime"
        indicator_plan_count = 0
        single_batch_count = 0
        evaluation_batch_count = 0
        for definition in self.indicators.list():
            dsl_version = str(definition.get("dsl_version") or LEGACY_DSL_VERSION)
            context_kind = str(definition.get("context_kind") or "single_product")
            try:
                adapted_dsl_version = (
                    dsl_version
                    if dsl_version.startswith("2.")
                    else LEGACY_TYPED_DSL_VERSION
                )
                registry_version = str(
                    (
                        definition.get("operator_registry_version")
                        or self._typed_registry_for_dsl(adapted_dsl_version)
                    )
                    if dsl_version.startswith("2.")
                    else LEGACY_TYPED_OPERATOR_REGISTRY_VERSION
                )
                plan = _compile_typed_plan(
                    normalize_variable_latex(
                        str(definition.get("expression") or "")
                    ),
                    context_kind,
                    adapted_dsl_version,
                    registry_version,
                )
                persist_numba_plan(compile_numba_plan(plan), runtime_root)
                indicator_plan_count += 1
                if context_kind == "single_product":
                    dependencies = self._physical_dependency_signature(
                        plan.context_requirements
                    )
                    physical_columns = tuple(
                        dict.fromkeys(
                            [
                                "adjusted_nav",
                                *[
                                    name
                                    for name in dependencies
                                    if name
                                    not in {
                                        "returns",
                                        "log_returns",
                                        "adjusted_nav",
                                    }
                                ],
                            ]
                        )
                    )
                    compile_numba_batch_plan(
                        (plan,),
                        (definition,),
                        physical_columns,
                    )
                    single_batch_count += 1
            except Exception:
                failures.append(
                    f"{definition.get('id')}@{definition.get('revision')}"
                )
        for saved_plan in self.plans.list():
            grouped: dict[
                tuple[tuple[str, ...], str],
                list[tuple[dict[str, Any], TypedIndicatorRuntime]],
            ] = {}
            try:
                for item in saved_plan.get("indicators", []):
                    definition = self.indicators.get(
                        str(item["indicator_id"]),
                        int(item["indicator_revision"]),
                    )
                    runtime = self._compile_runtime(
                        definition, str(item["period"])
                    )
                    if not isinstance(runtime, TypedIndicatorRuntime):
                        raise TypeError("evaluation plan contains a non-NJIT metric")
                    dependencies = self._physical_dependency_signature(
                        runtime.plan.context_requirements
                    )
                    grouped.setdefault(
                        (dependencies, str(item["period"])), []
                    ).append((definition, runtime))
                for (dependencies, _period), entries in grouped.items():
                    physical_columns = tuple(
                        dict.fromkeys(
                            [
                                "adjusted_nav",
                                *[
                                    name
                                    for name in dependencies
                                    if name
                                    not in {
                                        "returns",
                                        "log_returns",
                                        "adjusted_nav",
                                    }
                                ],
                            ]
                        )
                    )
                    compile_numba_batch_plan(
                        tuple(runtime.plan for _, runtime in entries),
                        tuple(definition for definition, _ in entries),
                        physical_columns,
                    )
                    evaluation_batch_count += 1
            except Exception:
                failures.append(
                    f"evaluation-plan:{saved_plan.get('id')}@{saved_plan.get('revision')}"
                )
        if failures:
            raise RuntimeError(
                "NJIT indicator warmup failed: " + ", ".join(failures)
            )
        return {
            "operator_coverage": kernel_status["operator_coverage"],
            "indicator_plans": indicator_plan_count,
            "single_metric_batch_plans": single_batch_count,
            "evaluation_batch_plans": evaluation_batch_count,
        }

    def start_compute_engine(self) -> None:
        plan_status = self.warm_numba_plans()
        self.compute_engine.start()
        worker_status = self.compute_engine.status()
        if not worker_status.get("fully_warmed"):
            raise RuntimeError("NJIT worker warmup did not complete")
        self._startup_warmup = {
            "complete": True,
            **plan_status,
            "workers": worker_status,
        }

    def close_compute_engine(self) -> None:
        self.compute_engine.close()

    @staticmethod
    def _typed_registry_for_dsl(dsl_version: str) -> str:
        if dsl_version == LEGACY_TYPED_DSL_VERSION:
            return LEGACY_TYPED_OPERATOR_REGISTRY_VERSION
        if dsl_version == COMPAT_TYPED_DSL_VERSION:
            return COMPAT_TYPED_OPERATOR_REGISTRY_VERSION
        return TYPED_OPERATOR_REGISTRY_VERSION

    def _apply_numba_v3_migration(self) -> dict[str, Any]:
        migration = self.plans.archive_and_reset(
            self.workspace_data_dir / "archive",
            NUMBA_V3_MIGRATION_MARKER,
        )
        if migration.get("applied"):
            self.run_results.clear()
        return migration

    def meta(self) -> dict[str, Any]:
        variable_path = Path(__file__).resolve().parents[1] / "cal_indicators" / "variable_latex.json"
        legacy_variables = json.loads(variable_path.read_text(encoding="utf-8")).get("variables", [])
        legacy_operators = [
            {
                "name": "sequence_mean",
                "label": "全元素算术平均值",
                "signature": "series<time> | vector<asset> -> scalar",
                "latex_template": r"\operatorname{mean}\left(x\right)",
                "return_type": "scalar",
                "parameters": [{"name": "values", "label": "输入值", "allowed_shapes": ["series", "vector"]}],
                "mathematical_essence": "计算时间序列或向量全部元素的算术平均值。",
            },
            {
                "name": "sequence_std",
                "label": "全元素标准差",
                "signature": "series<time> | vector<asset>, scalar ddof -> scalar",
                "latex_template": r"\operatorname{std}\left(x,1\right)",
                "return_type": "scalar",
                "parameters": [
                    {"name": "values", "label": "输入值", "allowed_shapes": ["series", "vector"]},
                    {"name": "ddof", "label": "自由度修正（ddof）", "shape": "scalar", "default": 1},
                ],
                "mathematical_essence": "计算时间序列或向量全部元素的样本标准差。",
            },
            {
                "name": "sequence_sum",
                "label": "全元素求和",
                "signature": "series<time> | vector<asset> -> scalar",
                "latex_template": r"\sum\left(x\right)",
                "return_type": "scalar",
                "parameters": [{"name": "values", "label": "输入值", "allowed_shapes": ["series", "vector"]}],
                "mathematical_essence": "将时间序列或向量的全部元素相加。",
            },
            {
                "name": "sequence_prod",
                "label": "全元素累乘",
                "signature": "series<time> | vector<asset> -> scalar",
                "latex_template": r"\prod\left(x\right)",
                "return_type": "scalar",
                "parameters": [{"name": "values", "label": "输入值", "allowed_shapes": ["series", "vector"]}],
                "mathematical_essence": "将时间序列或向量的全部元素相乘。",
            },
            {
                "name": "sqrt",
                "label": "逐元素平方根",
                "signature": "scalar | series<time> | vector<asset> -> same(input)",
                "latex_template": r"\sqrt{x}",
                "return_type": "scalar",
                "parameters": [{"name": "values", "label": "输入值", "allowed_shapes": ["scalar", "series", "vector"]}],
                "mathematical_essence": "逐元素计算非负输入的平方根。",
            },
            {
                "name": "sequence_geometric_mean",
                "label": "全元素几何平均值",
                "signature": "series<time> | vector<asset> -> scalar",
                "latex_template": r"\operatorname{gmean}\left(x\right)",
                "return_type": "scalar",
                "parameters": [{"name": "values", "label": "输入值", "allowed_shapes": ["series", "vector"]}],
                "mathematical_essence": "计算正值时间序列或向量全部元素的几何平均值。",
            },
        ]
        periods = period_metadata()
        legacy_templates = [
            self._draft_from_definition(item)
            for item in _built_in_indicators()
            if item.get("dsl_version") == LEGACY_DSL_VERSION
        ]
        typed = typed_product_meta()
        return {
            "engine_version": ENGINE_VERSION,
            "workspace_scope": "shared",
            "dsl_version": typed["dsl_version"],
            "compiler_version": typed["compiler_version"],
            "operator_registry_version": typed["operator_registry_version"],
            "variable_registry_version": typed["variable_registry_version"],
            "data_contract_version": typed["data_contract_version"],
            "context_schema_version": typed["context_schema_version"],
            "math_notation_version": typed["math_notation_version"],
            "types": typed["types"],
            "type_system": typed["type_system"],
            "context_kinds": ["single_product", "portfolio"],
            "numeric_backend": {
                **numeric_backend_status(),
                **kernel_registry_status(),
                "batch_engine": self.compute_engine.status(),
                "plan_cache": plan_cache_status(),
                "startup_warmup": copy.deepcopy(self._startup_warmup),
            },
            "numeric_kernel_version": NUMERIC_KERNEL_VERSION,
            "migration": {
                "marker": self.migration.get("marker"),
                "migrated_at": self.migration.get("migrated_at"),
                "archive_created": bool(self.migration.get("archived_file")),
                "applied": bool(self.migration.get("applied")),
            },
            "compile_cache": {
                "max_size": 512,
                "entries": _compile_typed_plan.cache_info().currsize,
            },
            "indicator_types": [
                {"id": type_id, "label": label}
                for type_id, label in INDICATOR_TYPE_LABELS.items()
            ],
            # Compatibility alias for existing clients. These are business
            # indicator types, not mathematical operator categories.
            "indicator_categories": [
                {"id": type_id, "label": label}
                for type_id, label in INDICATOR_TYPE_LABELS.items()
            ],
            "variables": typed["variables"],
            "operators": typed["operators"],
            "periods": periods,
            "period_policy": "all_supported",
            "templates": [],
            "predefined_calculations": [],
            "predefined_calculations_deprecated": True,
            "template_composition_deprecated": True,
            "legacy_variables": legacy_variables,
            "legacy_operators": legacy_operators,
            "legacy_templates": [],
            "limits": {
                "formula_length": MAX_FORMULA_LENGTH,
                "dag_nodes": MAX_DAG_NODES,
                "dag_depth": MAX_DAG_DEPTH,
                "indicators_per_request": MAX_INDICATORS,
                "targets_per_request": MAX_TARGETS,
                "combinations_per_request": MAX_COMBINATIONS,
                "rolling_combinations": MAX_ROLLING_COMBINATIONS,
                "series_observations": MAX_SERIES_OBSERVATIONS,
                "rolling_points": MAX_ROLLING_POINTS,
                **typed["limits"],
            },
        }

    @staticmethod
    def _draft_from_definition(definition: dict[str, Any]) -> dict[str, Any]:
        keys = (
            "name",
            "description",
            "expression",
            "periods",
            "unit",
            "display_format",
            "precision",
            "direction",
            "annual_risk_free_rate_percent",
        )
        draft = {key: copy.deepcopy(definition[key]) for key in keys}
        draft["indicator_type"] = indicator_type(definition)
        return draft

    @staticmethod
    def _decorate_definition(definition: dict[str, Any]) -> dict[str, Any]:
        """Expose legacy protocol defaults without mutating persisted history."""
        decorated = copy.deepcopy(definition)
        # Period is a runtime evaluation parameter, not a definition capability.
        # Historical records may contain a subset; expose the effective all-period
        # contract without rewriting the persisted version history.
        decorated["periods"] = list(SUPPORTED_PERIODS)
        decorated["period_policy"] = "all_supported"
        decorated.setdefault("dsl_version", LEGACY_DSL_VERSION)
        decorated.setdefault("operator_registry_version", LEGACY_OPERATOR_REGISTRY_VERSION)
        decorated.setdefault("context_kind", "single_product")
        decorated.setdefault("output_contract", "scalar")
        decorated.setdefault("output_measure", "dimensionless")
        decorated.setdefault("template_origin", None)
        if str(decorated["dsl_version"]).startswith("2."):
            is_modern = str(decorated["dsl_version"]) in {
                COMPAT_TYPED_DSL_VERSION,
                TYPED_DSL_VERSION,
            }
            decorated.setdefault(
                "variable_registry_version",
                VARIABLE_REGISTRY_VERSION if is_modern else "legacy-typed-v2.0",
            )
            decorated.setdefault(
                "data_contract_version",
                DATA_CONTRACT_VERSION if is_modern else "adjusted-nav-v1",
            )
            decorated.setdefault(
                "context_schema_version",
                CONTEXT_SCHEMA_VERSION if is_modern else "multi-asset-v1",
            )
            decorated.setdefault("numeric_kernel_version", NUMERIC_KERNEL_VERSION)
        decorated.setdefault("required_variables", [])
        decorated.setdefault("applicable_product_kinds", ["etf", "fund"])
        decorated["availability_policy"] = "runtime_required"
        decorated["availability_status"] = "runtime_check"
        if str(decorated["dsl_version"]).startswith("2."):
            try:
                decorated["display_latex"] = _typed_display_latex(
                    str(decorated.get("expression") or ""),
                    str(decorated.get("context_kind") or "single_product"),
                )
                decorated["math_notation_version"] = MATH_NOTATION_VERSION
            except (SyntaxError, ValueError):
                decorated["display_latex"] = None
        else:
            decorated["display_latex"] = decorated.get("expression")
        decorated["indicator_type"] = indicator_type(decorated)
        decorated["category_id"] = decorated["indicator_type"]
        decorated["category_label"] = INDICATOR_TYPE_LABELS[decorated["indicator_type"]]
        decorated["catalog_status"] = catalog_status(decorated)
        decorated["ui_exposed"] = ui_exposed(decorated)
        decorated["presentation"] = metric_presentation(decorated)
        return decorated

    @staticmethod
    def _is_typed_definition(definition: dict[str, Any]) -> bool:
        return str(definition.get("dsl_version") or LEGACY_DSL_VERSION).startswith("2.")

    @staticmethod
    def _normalize_definition(
        fields: dict[str, Any],
        protocol_defaults: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        protocol_defaults = protocol_defaults or {}
        name = str(fields.get("name", "")).strip()
        if not name or len(name) > 80:
            raise ValidationError("INVALID_NAME", "指标名称长度应为 1 至 80 个字符。", field="name")
        description = str(fields.get("description", "")).strip()
        if len(description) > 500:
            raise ValidationError("INVALID_DESCRIPTION", "指标说明不能超过 500 个字符。", field="description")
        expression = str(fields.get("expression", "")).strip()
        if not expression:
            raise ValidationError("EMPTY_EXPRESSION", "指标公式不能为空。", field="expression")
        if len(expression) > MAX_FORMULA_LENGTH:
            raise ValidationError(
                "FORMULA_TOO_LONG",
                f"指标公式不能超过 {MAX_FORMULA_LENGTH} 个字符。",
                field="expression",
            )
        # Keep accepting the legacy request field, but definitions always support
        # every engine period. The concrete period is selected only when running.
        periods = list(SUPPORTED_PERIODS)
        display_format = str(fields.get("display_format", "number"))
        if display_format not in {"number", "percent"}:
            raise ValidationError("INVALID_DISPLAY_FORMAT", "不支持的显示格式。", field="display_format")
        direction = str(fields.get("direction", "higher_better"))
        if direction not in {"higher_better", "lower_better"}:
            raise ValidationError("INVALID_DIRECTION", "不支持的优劣方向。", field="direction")
        requested_indicator_type = str(
            fields.get("indicator_type")
            or protocol_defaults.get("indicator_type")
            or "other"
        )
        if requested_indicator_type not in INDICATOR_TYPE_LABELS:
            raise ValidationError(
                "INVALID_INDICATOR_TYPE",
                "不支持的指标类型。",
                field="indicator_type",
            )
        precision = int(fields.get("precision", 2))
        if precision < 0 or precision > 8:
            raise ValidationError("INVALID_PRECISION", "显示精度必须在 0 到 8 之间。", field="precision")
        risk_free_rate = float(fields.get("annual_risk_free_rate_percent", 0.0))
        if not math.isfinite(risk_free_rate) or risk_free_rate < -100.0 or risk_free_rate > 100.0:
            raise ValidationError(
                "INVALID_RISK_FREE_RATE",
                "年化无风险利率必须是 -100 至 100 之间的有限数值。",
                field="annual_risk_free_rate_percent",
            )
        dsl_version = str(
            fields.get("dsl_version")
            or protocol_defaults.get("dsl_version")
            or TYPED_DSL_VERSION
        )
        if dsl_version not in {
            LEGACY_DSL_VERSION,
            LEGACY_TYPED_DSL_VERSION,
            COMPAT_TYPED_DSL_VERSION,
            TYPED_DSL_VERSION,
        }:
            raise ValidationError(
                "UNSUPPORTED_DSL_VERSION",
                f"不支持的 DSL 版本: {dsl_version}。",
                field="dsl_version",
            )
        context_kind = str(
            fields.get("context_kind")
            or protocol_defaults.get("context_kind")
            or "single_product"
        )
        if context_kind not in {"single_product", "portfolio"}:
            raise ValidationError(
                "INVALID_CONTEXT_KIND",
                "指标域必须为 single_product 或 portfolio。",
                field="context_kind",
            )
        if dsl_version == LEGACY_DSL_VERSION and context_kind != "single_product":
            raise ValidationError(
                "LEGACY_CONTEXT_UNSUPPORTED",
                "legacy v1 指标仅支持单产品域。",
                field="context_kind",
            )
        registry_default = (
            CustomIndicatorService._typed_registry_for_dsl(dsl_version)
            if dsl_version.startswith("2.")
            else LEGACY_OPERATOR_REGISTRY_VERSION
        )
        operator_registry_version = str(
            fields.get("operator_registry_version")
            or protocol_defaults.get("operator_registry_version")
            or registry_default
        )
        expected_registry = CustomIndicatorService._typed_registry_for_dsl(dsl_version)
        if dsl_version.startswith("2.") and operator_registry_version != expected_registry:
            raise ValidationError(
                "UNSUPPORTED_OPERATOR_REGISTRY_VERSION",
                f"typed {dsl_version} 仅支持算子注册表 {expected_registry}。",
                field="operator_registry_version",
            )
        requested_kernel_version = fields.get("numeric_kernel_version") or protocol_defaults.get(
            "numeric_kernel_version"
        )
        if dsl_version.startswith("2.") and requested_kernel_version not in {
            None,
            "",
            NUMERIC_KERNEL_VERSION,
        }:
            raise ValidationError(
                "UNSUPPORTED_NUMERIC_KERNEL_VERSION",
                f"typed 指标仅支持 numeric_kernel_version={NUMERIC_KERNEL_VERSION}。",
                field="numeric_kernel_version",
            )
        protocol_versions: dict[str, str | None] = {
            "variable_registry_version": None,
            "data_contract_version": None,
            "context_schema_version": None,
        }
        if dsl_version in {COMPAT_TYPED_DSL_VERSION, TYPED_DSL_VERSION}:
            supported_versions = {
                "variable_registry_version": VARIABLE_REGISTRY_VERSION,
                "data_contract_version": DATA_CONTRACT_VERSION,
                "context_schema_version": CONTEXT_SCHEMA_VERSION,
            }
            for field_name, supported_version in supported_versions.items():
                requested_version = str(
                    fields.get(field_name)
                    or protocol_defaults.get(field_name)
                    or supported_version
                )
                if requested_version != supported_version:
                    raise ValidationError(
                        f"UNSUPPORTED_{field_name.upper()}",
                        f"typed {dsl_version} 仅支持 {field_name}={supported_version}。",
                        field=field_name,
                    )
                protocol_versions[field_name] = requested_version
        elif dsl_version == LEGACY_TYPED_DSL_VERSION:
            # New requests that explicitly opt into v2.0 bind to the frozen
            # compatibility identifiers. Existing persisted JSON is never
            # rewritten; decoration applies these values only to API reads.
            protocol_versions = {
                "variable_registry_version": "legacy-typed-v2.0",
                "data_contract_version": "adjusted-nav-v1",
                "context_schema_version": "multi-asset-v1",
            }
        output_contract = str(
            fields.get("output_contract")
            or protocol_defaults.get("output_contract")
            or "scalar"
        )
        if output_contract != "scalar":
            raise ValidationError(
                "INVALID_OUTPUT_CONTRACT",
                "可保存指标的最终输出必须为有限标量。",
                field="output_contract",
            )
        origin = copy.deepcopy(
            fields.get("template_origin", protocol_defaults.get("template_origin"))
        )
        if isinstance(origin, str):
            origin = {
                "template_id": origin,
                "template_version": 1,
                "bindings": [],
                "detached": False,
            }
        if origin is not None and not isinstance(origin, dict):
            raise ValidationError(
                "INVALID_TEMPLATE_ORIGIN",
                "模板来源必须是对象、模板 ID 或 null。",
                field="template_origin",
            )
        previous_expression = protocol_defaults.get("expression")
        if origin and previous_expression and expression != previous_expression:
            origin["detached"] = True
        return {
            "name": name,
            "description": description,
            "expression": expression,
            "periods": periods,
            "period_policy": "all_supported",
            "unit": str(fields.get("unit", "")).strip()[:20],
            "display_format": display_format,
            "precision": precision,
            "direction": direction,
            "indicator_type": requested_indicator_type,
            "category_id": requested_indicator_type,
            "category_label": INDICATOR_TYPE_LABELS[requested_indicator_type],
            "annual_risk_free_rate_percent": risk_free_rate,
            "dsl_version": dsl_version,
            "operator_registry_version": operator_registry_version,
            "numeric_kernel_version": (
                NUMERIC_KERNEL_VERSION if dsl_version.startswith("2.") else None
            ),
            "context_kind": context_kind,
            "output_contract": output_contract,
            # The compiler is the authority. This placeholder is replaced by
            # ``_apply_compiled_contract`` before a definition is persisted.
            "output_measure": str(
                protocol_defaults.get("output_measure")
                or fields.get("output_measure")
                or "dimensionless"
            ),
            "template_origin": origin,
            "variable_registry_version": protocol_versions["variable_registry_version"],
            "data_contract_version": protocol_versions["data_contract_version"],
            "context_schema_version": protocol_versions["context_schema_version"],
        }

    @staticmethod
    def _diagnostic_code(message: str) -> str:
        if "未知变量" in message:
            return "UNKNOWN_VARIABLE"
        if "未知函数" in message:
            return "UNKNOWN_FUNCTION"
        if "个参数" in message and "需要" in message:
            return "ARITY_MISMATCH"
        if "实际为" in message or "最终结果必须是标量" in message:
            return "TYPE_MISMATCH"
        if "节点数" in message or "深度" in message:
            return "FORMULA_TOO_COMPLEX"
        return "INVALID_EXPRESSION"

    def _compose_indicator_reference(self, fields: dict[str, Any]) -> dict[str, Any]:
        indicator_id = str(fields.get("indicator_id") or "").strip()
        revision = fields.get("indicator_revision")
        definition = self._decorate_definition(
            self.indicators.get(indicator_id, int(revision) if revision is not None else None)
        )
        context_kind = str(fields.get("context") or "single_product")
        if definition.get("context_kind") != context_kind:
            raise ValidationError(
                "INDICATOR_CONTEXT_MISMATCH",
                "所选指标与当前计算域不一致。",
                field="indicator_id",
            )
        if definition.get("output_contract") != "scalar":
            raise ValidationError(
                "INDICATOR_OUTPUT_MISMATCH",
                "只有最终输出为标量的指标可以插入公式。",
                field="indicator_id",
            )

        protocol_fields = (
            "dsl_version",
            "operator_registry_version",
            "numeric_kernel_version",
            "variable_registry_version",
            "data_contract_version",
            "context_schema_version",
        )
        mismatches = [
            name
            for name in protocol_fields
            if fields.get(name) is not None
            and str(fields[name]) != str(definition.get(name) or "")
        ]
        if mismatches:
            raise ValidationError(
                "INDICATOR_PROTOCOL_MISMATCH",
                f"所选指标与当前草稿的版本协议不一致: {', '.join(mismatches)}。",
                field="indicator_id",
            )

        dsl_version = str(definition.get("dsl_version") or LEGACY_DSL_VERSION)
        if not dsl_version.startswith("2."):
            raise ValidationError(
                "INDICATOR_COMPOSITION_UNSUPPORTED",
                "兼容版指标不能直接插入类型化公式，请先迁移或复制为 typed 指标。",
                field="indicator_id",
            )
        registry_version = str(definition.get("operator_registry_version") or "")
        result = infer_expression(
            str(definition.get("expression") or ""),
            context_kind,  # type: ignore[arg-type]
            scalar_required=True,
            dsl_version=dsl_version,
            operator_registry_version=registry_version,
        )
        result["indicator_origin"] = {
            "indicator_id": definition["id"],
            "indicator_revision": int(definition["revision"]),
            "name": definition["name"],
            "source": definition["source"],
        }
        return result

    def compose(self, fields: dict[str, Any]) -> dict[str, Any]:
        indicator_id = fields.get("indicator_id")
        if indicator_id:
            if fields.get("operator_id") or fields.get("template_id"):
                raise ValidationError(
                    "INVALID_COMPOSE_TARGET",
                    "operator_id、template_id 与 indicator_id 必须且只能指定一个。",
                )
            return self._compose_indicator_reference(fields)
        return compose_expression(fields)

    def infer(self, fields: dict[str, Any]) -> dict[str, Any]:
        expression = str(fields.get("expression") or "").strip()
        context_kind = str(fields.get("context") or fields.get("context_kind") or "single_product")
        dsl_version = str(fields.get("dsl_version") or TYPED_DSL_VERSION)
        registry_version = str(
            fields.get("operator_registry_version")
            or self._typed_registry_for_dsl(dsl_version)
        )
        if len(expression) > MAX_FORMULA_LENGTH:
            raise ValidationError(
                "FORMULA_TOO_LONG",
                f"公式表达式不能超过 {MAX_FORMULA_LENGTH} 个字符。",
                field="expression",
            )
        return infer_expression(
            expression,
            context_kind,  # type: ignore[arg-type]
            scalar_required=False,
            dsl_version=dsl_version,
            operator_registry_version=registry_version,
        )

    def _validate_typed(self, fields: dict[str, Any]) -> dict[str, Any]:
        expression = str(fields.get("expression", "")).strip()
        periods = list(SUPPORTED_PERIODS)
        context_kind = str(fields.get("context_kind") or "single_product")
        dsl_version = str(fields.get("dsl_version") or TYPED_DSL_VERSION)
        registry_version = str(
            fields.get("operator_registry_version")
            or self._typed_registry_for_dsl(dsl_version)
        )
        if len(expression) > MAX_FORMULA_LENGTH:
            return self._invalid_validation(
                {
                    "code": "FORMULA_TOO_LONG",
                    "message": f"指标公式不能超过 {MAX_FORMULA_LENGTH} 个字符。",
                    "field": "expression",
                }
            )
        try:
            inferred = infer_expression(
                expression,
                context_kind,  # type: ignore[arg-type]
                scalar_required=True,
                dsl_version=dsl_version,
                operator_registry_version=registry_version,
            )
        except ValidationError as exc:
            diagnostics = exc.diagnostics or [
                {"code": exc.code, "message": exc.message, "field": exc.field or "expression"}
            ]
            for diagnostic in diagnostics:
                if diagnostic.get("code") == "UNKNOWN_OPERATOR":
                    diagnostic["code"] = "UNKNOWN_FUNCTION"
            return {
                **self._invalid_validation(diagnostics[0]),
                "diagnostics": diagnostics,
            }

        compiled_batch = None
        if context_kind == "single_product":
            try:
                plan = _compile_typed_plan(
                    normalize_variable_latex(expression),
                    context_kind,
                    dsl_version,
                    registry_version,
                )
                dependencies = self._physical_dependency_signature(
                    plan.context_requirements
                )
                physical_columns = tuple(
                    dict.fromkeys(
                        [
                            "adjusted_nav",
                            *[
                                name
                                for name in dependencies
                                if name
                                not in {
                                    "returns",
                                    "log_returns",
                                    "adjusted_nav",
                                }
                            ],
                        ]
                    )
                )
                compiled_batch = compile_numba_batch_plan(
                    (plan,),
                    (fields,),
                    physical_columns,
                )
            except (NumbaPlanCompileError, TypeError, ValueError) as exc:
                return self._invalid_validation(
                    {
                        "code": "NJIT_BATCH_COMPILE_FAILED",
                        "message": "指标公式无法编译为批量 NJIT 计算计划。",
                        "field": "expression",
                    }
                )

        dag = copy.deepcopy(inferred["dag"])
        root_id = dag.get("roots", {}).get("result")
        dag["roots"] = {"result": root_id}
        output_measure = "dimensionless"
        for node in dag.get("nodes", []):
            value_type = node.get("inferred_type") or {}
            if node.get("id") == root_id:
                output_measure = str(
                    value_type.get("semantic_dimension") or "dimensionless"
                )
            node["value_type"] = value_type.get("display")
            node["shape"] = value_type.get("kind")
            node["axes"] = value_type.get("axes", [])
            node["symbolic_shape"] = value_type.get("shape", [])
        return {
            "valid": True,
            "diagnostics": [],
            "dependencies": inferred["dependencies"],
            "python_expression": inferred.get("python_expression"),
            "latex": inferred.get("latex"),
            "display_latex": inferred.get("display_latex"),
            "math_notation_version": inferred.get("math_notation_version"),
            "dag": dag,
            "output_type": inferred["inferred_type"],
            "output_measure": output_measure,
            "context_requirements": inferred["dependencies"],
            "estimated_cost": inferred["estimated_cost"],
            "semantic_warnings": inferred["semantic_warnings"],
            "dsl_version": inferred["dsl_version"],
            "operator_registry_version": inferred["operator_registry_version"],
            "compiled_plan_id": inferred.get("compiled_plan_id"),
            "compiled_batch_plan_id": (
                compiled_batch.plan_id if compiled_batch is not None else None
            ),
            "compile_status": inferred.get("compile_status"),
            "compile_ms": inferred.get("compile_ms"),
            "batch_compile_ms": (
                compiled_batch.compile_ms if compiled_batch is not None else 0.0
            ),
            "kernel_version": inferred.get("kernel_version"),
            "engine_version": inferred.get("engine_version"),
            "required_workspace_bytes": inferred.get("required_workspace_bytes", 0),
            "python_fallback": 0,
            "python_operator_calls": 0,
        }

    def validate(self, fields: dict[str, Any]) -> dict[str, Any]:
        if str(fields.get("dsl_version") or LEGACY_DSL_VERSION).startswith("2."):
            return self._validate_typed(fields)
        expression = str(fields.get("expression", "")).strip()
        preview_period = str(fields.get("preview_period") or "1Y").upper()
        if preview_period not in SUPPORTED_PERIODS:
            preview_period = "1Y"
        name = str(fields.get("name") or "未保存指标")
        if len(expression) > MAX_FORMULA_LENGTH:
            diagnostic = {
                "code": "FORMULA_TOO_LONG",
                "message": f"指标公式不能超过 {MAX_FORMULA_LENGTH} 个字符。",
                "field": "expression",
            }
            return self._invalid_validation(diagnostic)
        try:
            runtime = IndicatorRuntime.from_definition(
                name,
                expression,
                [preview_period],
                metadata={"annual_risk_free_rate_percent": fields.get("annual_risk_free_rate_percent", 0.0)},
                max_nodes=MAX_DAG_NODES,
                max_depth=MAX_DAG_DEPTH,
            )
        except (DAGBuildError, LatexParseError, SyntaxError, ValueError, KeyError) as exc:
            message = str(exc) or "无法解析指标公式。"
            return self._invalid_validation(
                {"code": self._diagnostic_code(message), "message": message, "field": "expression"}
            )

        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, int]] = []
        roots: dict[str, int] = {}
        seen_nodes: set[int] = set()
        seen_edges: set[tuple[int, int]] = set()
        dependencies: list[str] = []
        for period in runtime.available_periods:
            payload = runtime.executor.graph_payload(period)
            root_map = payload["roots"]
            roots["result"] = int(root_map[name])
            for node in payload["nodes"]:
                if node["id"] not in seen_nodes:
                    nodes.append(node)
                    seen_nodes.add(node["id"])
                if node["kind"] == "variable" and node["label"] not in dependencies:
                    dependencies.append(node["label"])
            for edge in payload["edges"]:
                key = (edge["source"], edge["target"])
                if key not in seen_edges:
                    edges.append(edge)
                    seen_edges.add(key)
        return {
            "valid": True,
            "diagnostics": [],
            "dependencies": dependencies,
            "python_expression": runtime.executor.parser.to_python(expression),
            "dag": {"nodes": nodes, "edges": edges, "roots": roots},
        }

    @staticmethod
    def _invalid_validation(diagnostic: dict[str, Any]) -> dict[str, Any]:
        return {
            "valid": False,
            "diagnostics": [diagnostic],
            "dependencies": [],
            "python_expression": None,
            "dag": None,
        }

    def _compile_or_raise(self, fields: dict[str, Any]) -> dict[str, Any]:
        result = self.validate(fields)
        if not result["valid"]:
            diagnostic = result["diagnostics"][0]
            raise ValidationError(
                diagnostic["code"],
                diagnostic["message"],
                field="expression",
                diagnostics=result["diagnostics"],
            )
        return result

    @staticmethod
    def _apply_compiled_contract(
        definition: dict[str, Any], validation: dict[str, Any]
    ) -> None:
        dependencies = list(canonicalize_variables(validation.get("dependencies") or []))
        definition["required_variables"] = dependencies
        applicable = {"etf", "fund"}
        for variable_id in dependencies:
            variable = get_variable(variable_id)
            if variable is not None and "single_product" in variable.domains:
                applicable &= set(variable.product_kinds)
        if definition.get("context_kind") == "portfolio":
            definition["applicable_product_kinds"] = ["portfolio"]
        else:
            definition["applicable_product_kinds"] = sorted(applicable)
        definition["availability_status"] = "ready"
        definition["output_measure"] = str(
            validation.get("output_measure") or definition.get("output_measure") or "dimensionless"
        )
        definition["numeric_kernel_version"] = str(
            validation.get("kernel_version") or NUMERIC_KERNEL_VERSION
        )
        definition["compiled_plan_id"] = validation.get("compiled_plan_id")
        definition["compiled_batch_plan_id"] = validation.get(
            "compiled_batch_plan_id"
        )

    def list_indicators(
        self,
        *,
        context_kind: str | None = None,
        product_kind: str | None = None,
        source: str | None = None,
        category: str | None = None,
        include_compatibility: bool = False,
    ) -> dict[str, Any]:
        items = [self._decorate_definition(item) for item in self.indicators.list()]
        if not include_compatibility:
            items = [item for item in items if item.get("ui_exposed", True)]
        if context_kind:
            items = [item for item in items if item.get("context_kind") == context_kind]
        if product_kind:
            for item in items:
                likely_available = product_kind in item.get(
                    "applicable_product_kinds", []
                )
                item["product_kind_hint"] = {
                    "product_kind": product_kind,
                    "status": "likely_available" if likely_available else "runtime_check",
                    "message": (
                        "通常具备所需输入，仍将按具体产品和计算区间核对。"
                        if likely_available
                        else "不会按产品类型隐藏；运行时将检查具体产品是否具备所需字段。"
                    ),
                }
        if source:
            items = [item for item in items if item.get("source") == source]
        if category:
            items = [
                item
                for item in items
                if item.get("presentation", {}).get("category") == category
            ]
        return {"items": items, "total": len(items)}

    def get_indicator(self, indicator_id: str) -> dict[str, Any]:
        return self._decorate_definition(self.indicators.get(indicator_id))

    def create_indicator(self, fields: dict[str, Any]) -> dict[str, Any]:
        normalized = self._normalize_definition(fields)
        validation = self._compile_or_raise(normalized)
        self._apply_compiled_contract(normalized, validation)
        created = self.indicators.create(normalized)
        self.cache.clear()
        self.plan_cache.clear()
        return self._decorate_definition(created)

    def update_indicator(self, indicator_id: str, revision: int, fields: dict[str, Any]) -> dict[str, Any]:
        current = self._decorate_definition(self.indicators.get(indicator_id))
        normalized = self._normalize_definition(fields, current)
        validation = self._compile_or_raise(normalized)
        self._apply_compiled_contract(normalized, validation)
        updated = self.indicators.update(indicator_id, revision, normalized)
        self.cache.clear()
        self.plan_cache.clear()
        return self._decorate_definition(updated)

    def delete_indicator(self, indicator_id: str, revision: int) -> None:
        current = self.indicators.get(indicator_id)
        if int(current["revision"]) != revision:
            raise ConflictError(
                "REVISION_CONFLICT",
                "指标已被其他操作更新，请刷新后重试。",
                field="revision",
            )
        if self.plans.references_indicator(indicator_id):
            raise ConflictError(
                "INDICATOR_IN_USE",
                "该指标正在被评价方案引用，请先调整或删除相关方案。",
            )
        if self.snapshot_config.references_indicator(indicator_id):
            raise ConflictError(
                "INDICATOR_IN_SNAPSHOT_CONFIG",
                "该指标已配置为快照指标，请先从快照加速配置中移除。",
            )
        self.indicators.delete(indicator_id, revision)
        self.cache.clear()
        self.plan_cache.clear()

    def get_snapshot_config(self) -> dict[str, Any]:
        config = self.snapshot_config.get()
        items: list[dict[str, Any]] = []
        for item in config.get("items", []):
            resolved = dict(item)
            try:
                definition = self._decorate_definition(
                    self.indicators.get(
                        str(item["indicator_id"]),
                        int(item["indicator_revision"]),
                    )
                )
                resolved.update(
                    {
                        "name": definition["name"],
                        "source": definition["source"],
                        "presentation": definition.get("presentation"),
                        "status": "ready",
                        "status_message": "将在数据刷新后预计算并写入产品快照。",
                    }
                )
            except IndicatorDomainError as exc:
                resolved.update(
                    {
                        "name": str(item.get("indicator_id") or "未知指标"),
                        "status": "definition_missing",
                        "status_message": exc.message,
                    }
                )
            items.append(resolved)
        snapshot_data_dir = self._snapshot_data_dir()
        metadata_path = snapshot_data_dir / "instrument_metrics_snapshot.meta.json"
        snapshot: dict[str, Any] | None = None
        if metadata_path.exists():
            try:
                raw = json.loads(metadata_path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    snapshot = {
                        "generated_at": raw.get("generated_at"),
                        "config_revision": raw.get("config_revision"),
                        "configured_count": raw.get("configured_count"),
                        "data_generation": raw.get("data_generation"),
                    }
            except (OSError, json.JSONDecodeError):
                snapshot = None
        return {
            **config,
            "items": items,
            "max_items": MAX_SNAPSHOT_INDICATORS,
            "snapshot": snapshot,
            "snapshot_status": (
                "ready"
                if snapshot
                and int(snapshot.get("config_revision") or 0) == int(config.get("revision") or 0)
                and snapshot.get("data_generation") == market_data_generation(self.market_data_dir)
                else ("stale" if snapshot else "missing")
            ),
        }

    def update_snapshot_config(
        self,
        revision: int,
        items: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if len(items) > MAX_SNAPSHOT_INDICATORS:
            raise ValidationError(
                "SNAPSHOT_INDICATOR_LIMIT_EXCEEDED",
                f"快照指标最多配置 {MAX_SNAPSHOT_INDICATORS} 个指标与周期组合。",
                field="items",
            )
        normalized: list[dict[str, Any]] = []
        current_fields = {
            (
                str(item.get("indicator_id")),
                int(item.get("indicator_revision") or 0),
                str(item.get("period") or "").upper(),
            ): str(item.get("field") or "")
            for item in self.snapshot_config.get().get("items", [])
        }
        seen_keys: set[tuple[str, int, str]] = set()
        seen_fields: set[str] = set()
        revisions_by_indicator: dict[str, int] = {}
        for index, raw in enumerate(items):
            raw_key = (
                str(raw.get("indicator_id") or "").strip(),
                int(raw.get("indicator_revision") or 0),
                str(raw.get("period") or "").strip().upper(),
            )
            item = normalized_snapshot_item(
                {**raw, "field": raw.get("field") or current_fields.get(raw_key)}
            )
            if not item["indicator_id"] or int(item["indicator_revision"]) < 1:
                raise ValidationError(
                    "INVALID_SNAPSHOT_INDICATOR",
                    "快照指标必须指定有效的指标及版本。",
                    field=f"items.{index}",
                )
            if item["period"] not in SUPPORTED_PERIODS:
                raise ValidationError(
                    "INVALID_PERIOD",
                    f"不支持快照周期 {item['period']}。",
                    field=f"items.{index}.period",
                )
            key = (
                item["indicator_id"],
                int(item["indicator_revision"]),
                item["period"],
            )
            if key in seen_keys:
                raise ValidationError(
                    "DUPLICATE_SNAPSHOT_INDICATOR",
                    "同一指标版本和周期不能重复配置。",
                    field=f"items.{index}",
                )
            previous_revision = revisions_by_indicator.get(item["indicator_id"])
            if previous_revision is not None and previous_revision != int(item["indicator_revision"]):
                raise ValidationError(
                    "SNAPSHOT_VERSION_CONFLICT",
                    "同一个指标在快照配置中必须统一锁定到同一版本。",
                    field=f"items.{index}.indicator_revision",
                )
            if item["field"] in seen_fields:
                raise ValidationError(
                    "DUPLICATE_SNAPSHOT_FIELD",
                    "快照字段发生冲突，请重新选择指标或周期。",
                    field=f"items.{index}",
                )
            definition = self.indicators.get(
                item["indicator_id"], int(item["indicator_revision"])
            )
            if definition.get("context_kind", "single_product") != "single_product":
                raise ValidationError(
                    "SNAPSHOT_CONTEXT_MISMATCH",
                    "只有单产品标量指标可以配置为产品快照。",
                    field=f"items.{index}.indicator_id",
                )
            if definition.get("output_contract", "scalar") != "scalar":
                raise ValidationError(
                    "SNAPSHOT_OUTPUT_MISMATCH",
                    "快照指标的最终结果必须是单个数值。",
                    field=f"items.{index}.indicator_id",
                )
            self._compile_runtime(definition, item["period"])
            normalized.append(item)
            seen_keys.add(key)
            seen_fields.add(item["field"])
            revisions_by_indicator[item["indicator_id"]] = int(item["indicator_revision"])
        self.warm_snapshot_numba_plans(normalized)
        self.snapshot_config.update(revision, normalized)
        return self.get_snapshot_config()

    def warm_snapshot_numba_plans(
        self,
        items: Optional[list[dict[str, Any]]] = None,
    ) -> dict[str, int]:
        """Compile only the plans needed by the data-refresh snapshot job."""

        selected = items if items is not None else self.snapshot_config.get().get("items", [])
        groups: dict[
            tuple[tuple[str, ...], str],
            list[tuple[dict[str, Any], TypedIndicatorRuntime]],
        ] = {}
        singleton_count = 0
        for item in selected:
            definition = self.indicators.get(
                str(item["indicator_id"]), int(item["indicator_revision"])
            )
            runtime = self._compile_runtime(definition, str(item["period"]))
            if not isinstance(runtime, TypedIndicatorRuntime):
                raise ValidationError(
                    "NJIT_RUNTIME_REQUIRED",
                    "快照指标必须能够编译为 NJIT 计算计划。",
                )
            dependencies = self._physical_dependency_signature(
                runtime.plan.context_requirements
            )
            physical_columns = tuple(
                dict.fromkeys(
                    [
                        "adjusted_nav",
                        *[
                            name
                            for name in dependencies
                            if name not in {"returns", "log_returns", "adjusted_nav"}
                        ],
                    ]
                )
            )
            compile_numba_batch_plan(
                (runtime.plan,),
                (definition,),
                physical_columns,
            )
            singleton_count += 1
            groups.setdefault((dependencies, str(item["period"])), []).append(
                (definition, runtime)
            )
        for (dependencies, _period), entries in groups.items():
            physical_columns = tuple(
                dict.fromkeys(
                    [
                        "adjusted_nav",
                        *[
                            name
                            for name in dependencies
                            if name not in {"returns", "log_returns", "adjusted_nav"}
                        ],
                    ]
                )
            )
            for start in range(0, len(entries), MAX_INDICATORS):
                batch = entries[start : start + MAX_INDICATORS]
                compile_numba_batch_plan(
                    tuple(runtime.plan for _, runtime in batch),
                    tuple(definition for definition, _ in batch),
                    physical_columns,
                )
        return {"singletons": singleton_count, "batches": len(groups)}

    @staticmethod
    def _validate_targets(
        targets: Iterable[dict[str, Any]],
        max_targets: int = MAX_TARGETS,
    ) -> list[dict[str, str]]:
        normalized: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for raw in targets:
            kind = str(raw.get("kind", ""))
            product_id = str(raw.get("product_id", "")).strip()
            if kind not in {"etf", "fund"} or not product_id:
                raise ValidationError("INVALID_TARGET", "产品类型或产品编号无效。", field="targets")
            key = (kind, product_id)
            if key not in seen:
                normalized.append({"kind": kind, "product_id": product_id})
                seen.add(key)
        if not normalized or len(normalized) > max_targets:
            raise ValidationError(
                "TARGET_LIMIT_EXCEEDED",
                f"产品数量需要在 1 至 {max_targets} 之间。",
                field="targets",
            )
        return normalized

    def _resolve_evaluation_definitions(
        self,
        indicator_ids: list[str],
        inline_definition: Optional[dict[str, Any]],
        indicator_versions: Optional[dict[str, int]] = None,
    ) -> list[dict[str, Any]]:
        if bool(indicator_ids) == bool(inline_definition):
            raise ValidationError(
                "INDICATOR_SOURCE_CONFLICT",
                "indicator_ids 与 inline_definition 必须且只能提供一种。",
            )
        if inline_definition is not None:
            normalized = self._normalize_definition(inline_definition)
            validation = self._compile_or_raise(normalized)
            self._apply_compiled_contract(normalized, validation)
            return [{**normalized, "id": None, "revision": None, "source": "inline"}]
        unique_ids = list(dict.fromkeys(indicator_ids))
        if not unique_ids or len(unique_ids) > MAX_INDICATORS:
            raise ValidationError(
                "INDICATOR_LIMIT_EXCEEDED",
                f"每次计算需要 1 至 {MAX_INDICATORS} 个指标。",
                field="indicator_ids",
            )
        versions = indicator_versions or {}
        return [
            self._decorate_definition(self.indicators.get(indicator_id, versions.get(indicator_id)))
            for indicator_id in unique_ids
        ]

    @staticmethod
    def _definition_cache_key(definition: dict[str, Any]) -> str:
        payload = {
            "id": definition.get("id"),
            "revision": definition.get("revision"),
            "expression": definition["expression"],
            "risk_free": definition.get("annual_risk_free_rate_percent", 0.0),
            "dsl_version": definition.get("dsl_version", LEGACY_DSL_VERSION),
            "operator_registry_version": definition.get(
                "operator_registry_version", LEGACY_OPERATOR_REGISTRY_VERSION
            ),
            "context_kind": definition.get("context_kind", "single_product"),
            "output_contract": definition.get("output_contract", "scalar"),
            "output_measure": definition.get("output_measure", "dimensionless"),
            "variable_registry_version": definition.get("variable_registry_version"),
            "data_contract_version": definition.get("data_contract_version"),
            "context_schema_version": definition.get("context_schema_version"),
            "required_variables": definition.get("required_variables"),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    @staticmethod
    def _empty_window(
        product_series: Optional[ProductSeries | ProductVariableSeries],
        as_of: Optional[str],
    ) -> dict[str, Any]:
        return {
            "requested_as_of": as_of,
            "effective_as_of": None,
            "start_date": None,
            "end_date": None,
            "observation_count": 0,
            "data_latest_date": product_series.data_latest_date if product_series else None,
        }

    @staticmethod
    def _window_payload(window: PeriodWindow) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "requested_as_of": window.requested_as_of,
            "effective_as_of": window.effective_as_of,
            "start_date": window.start_date,
            "end_date": window.end_date,
            "observation_count": window.observation_count,
            "data_latest_date": window.data_latest_date,
        }
        if isinstance(window, VariablePeriodWindow):
            payload.update(
                {
                    "common_date_hash": window.common_date_hash,
                    "data_lineage": copy.deepcopy(window.lineage),
                    "coverage": copy.deepcopy(window.coverage),
                    "source_fingerprints": dict(window.fingerprints),
                }
            )
        return payload

    @staticmethod
    def _result_base(
        definition: dict[str, Any],
        target: dict[str, str],
        target_name: str,
        period: str,
    ) -> dict[str, Any]:
        return {
            "indicator_id": definition.get("id"),
            "indicator_revision": definition.get("revision"),
            "indicator_name": definition["name"],
            "target": {**target, "name": target_name},
            "period": period,
            "unit": definition.get("unit", ""),
            "display_format": definition.get("display_format", "number"),
            "presentation": metric_presentation(definition),
        }

    def _snapshot_data_dir(self) -> Path:
        candidate = Path(self.market_data_dir).expanduser().resolve()
        if candidate == DEFAULT_DATA_DIR.resolve():
            return resolve_tushare_data_dir(DEFAULT_DATA_DIR)
        return candidate

    def _evaluate_from_snapshot(
        self,
        definitions: list[dict[str, Any]],
        targets: list[dict[str, str]],
        period: str,
    ) -> dict[str, Any] | None:
        """Return an exact current snapshot hit, or ``None`` for live evaluation."""

        config = self.snapshot_config.get()
        configured = {
            (
                str(item.get("indicator_id")),
                int(item.get("indicator_revision") or 0),
                str(item.get("period") or "").upper(),
            ): str(item.get("field") or "")
            for item in config.get("items", [])
        }
        fields: dict[tuple[str, int], str] = {}
        for definition in definitions:
            key = (
                str(definition.get("id") or ""),
                int(definition.get("revision") or 0),
                period,
            )
            field = configured.get(key)
            if not field:
                return None
            fields[(key[0], key[1])] = field

        data_dir = self._snapshot_data_dir()
        snapshot_path = data_dir / "instrument_metrics_snapshot.parquet"
        metadata_path = data_dir / "instrument_metrics_snapshot.meta.json"
        if not snapshot_path.exists() or not metadata_path.exists():
            return None
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(metadata, dict):
            return None
        if int(metadata.get("config_revision") or 0) != int(config.get("revision") or 0):
            return None
        generation = market_data_generation(self.market_data_dir)
        if metadata.get("data_generation") != generation:
            return None

        suffixes = (
            "status",
            "observation_count",
            "start_date",
            "end_date",
            "effective_as_of",
            "warning_code",
            "warning_message",
        )
        columns = {"instrument_type", "ts_code", "latest_date"}
        for field in fields.values():
            columns.add(field)
            columns.update(f"{field}__{suffix}" for suffix in suffixes)
        try:
            snapshot = pd.read_parquet(snapshot_path, columns=sorted(columns))
        except (OSError, ValueError, KeyError):
            return None
        if snapshot.empty:
            return None
        snapshot["instrument_type"] = snapshot["instrument_type"].astype(str)
        snapshot["ts_code"] = snapshot["ts_code"].astype(str)
        snapshot = snapshot.set_index(["instrument_type", "ts_code"], drop=False)

        results: list[dict[str, Any]] = []
        for definition in definitions:
            definition_key = (
                str(definition.get("id") or ""),
                int(definition.get("revision") or 0),
            )
            field = fields[definition_key]
            required_count = len(definition.get("required_variables") or [])
            for target in targets:
                row_key = (target["kind"], target["product_id"])
                base = self._result_base(
                    definition,
                    target,
                    target["product_id"],
                    period,
                )
                if row_key not in snapshot.index:
                    results.append(
                        {
                            **base,
                            "value": None,
                            "status": "unavailable",
                            "warnings": [
                                {
                                    "code": "SNAPSHOT_TARGET_MISSING",
                                    "message": "当前数据快照中没有该产品的预计算结果。",
                                }
                            ],
                            "window": self._empty_window(None, None),
                            "input_requirements": {
                                "status": "blocked",
                                "required_count": required_count,
                                "available_count": 0,
                                "blocking_inputs": [],
                            },
                            "target_data": {
                                "available_datasets": [],
                                "available_variables": [],
                                "data_latest_date": None,
                            },
                        }
                    )
                    continue
                row = snapshot.loc[row_key]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[-1]
                raw_value = row.get(field)
                value = (
                    float(raw_value)
                    if raw_value is not None
                    and not pd.isna(raw_value)
                    and np.isfinite(float(raw_value))
                    else None
                )
                raw_status = row.get(f"{field}__status")
                status = (
                    ""
                    if raw_status is None or pd.isna(raw_status)
                    else str(raw_status).strip().lower()
                )
                if status not in {"ok", "warning", "unavailable", "error"}:
                    status = "ok" if value is not None else "unavailable"
                warning_code = row.get(f"{field}__warning_code")
                warning_message = row.get(f"{field}__warning_message")
                warnings = []
                if warning_message is not None and not pd.isna(warning_message):
                    warnings.append(
                        {
                            "code": str(warning_code or "SNAPSHOT_WARNING"),
                            "message": str(warning_message),
                        }
                    )
                data_latest = row.get("latest_date")
                data_latest_value = (
                    None
                    if data_latest is None or pd.isna(data_latest)
                    else pd.Timestamp(data_latest).strftime("%Y-%m-%d")
                )
                raw_observation_count = row.get(f"{field}__observation_count")
                observation_count = (
                    0
                    if raw_observation_count is None or pd.isna(raw_observation_count)
                    else int(raw_observation_count)
                )

                def date_value(suffix: str) -> str | None:
                    raw_date = row.get(f"{field}__{suffix}")
                    if raw_date is None or pd.isna(raw_date):
                        return None
                    return str(raw_date)[:10]
                results.append(
                    {
                        **base,
                        "value": value,
                        "status": status,
                        "warnings": warnings,
                        "window": {
                            "requested_as_of": None,
                            "effective_as_of": date_value("effective_as_of"),
                            "start_date": date_value("start_date"),
                            "end_date": date_value("end_date"),
                            "observation_count": observation_count,
                            "data_latest_date": data_latest_value,
                        },
                        "input_requirements": {
                            "status": "ready" if value is not None else "blocked",
                            "required_count": required_count,
                            "available_count": required_count if value is not None else 0,
                            "blocking_inputs": [],
                        },
                        "target_data": {
                            "available_datasets": ["指标预计算快照"],
                            "available_variables": list(
                                definition.get("required_variables") or []
                            ),
                            "data_latest_date": data_latest_value,
                        },
                    }
                )

        status_counts = {
            status: sum(item["status"] == status for item in results)
            for status in ("ok", "warning", "error", "unavailable")
        }
        return {
            "results": results,
            "summary": {"total": len(results), **status_counts},
            "cache": {"hits": len(results), "misses": 0},
            "execution": {
                **kernel_registry_status(),
                "engine_version": "indicator-snapshot-v1",
                "data_generation": generation,
                "snapshot_config_revision": int(config.get("revision") or 0),
                "snapshot_hits": len(results),
                "compile_cache_hits": 0,
                "compile_cache_misses": 0,
                "kernel_cache_hits": len(results),
                "kernel_cache_misses": 0,
                "compiled_plan_ids": [],
                "parallel_tasks": 0,
                "python_fallback": 0,
                "python_operator_calls": 0,
            },
        }

    @staticmethod
    def _requirement_status(reason_code: str) -> str:
        if reason_code in {
            "SOURCE_UNAVAILABLE_FOR_PRODUCT",
            "VARIABLE_NOT_APPLICABLE",
            "SOURCE_DATASET_MISSING",
        }:
            return "source_unavailable"
        if reason_code in {"SOURCE_FIELD_MISSING", "SOURCE_SCHEMA_MISMATCH"}:
            return "field_missing"
        if reason_code in {
            "PRODUCT_DATA_NOT_FOUND",
            "VARIABLE_NO_OBSERVATIONS",
            "DATA_NOT_FOUND",
        }:
            return "no_observations"
        if reason_code in {
            "INSUFFICIENT_SAMPLE",
            "INSUFFICIENT_COMMON_SAMPLE",
            "NO_DATA_AS_OF",
            "NO_DATA_FOR_PERIOD",
            "VARIABLE_NOT_IN_WINDOW",
            "ANN_DATE_UNAVAILABLE",
        }:
            return "insufficient_window"
        return "unavailable"

    @staticmethod
    def _target_data_payload(
        source: Optional[ProductSeries | ProductVariableSeries],
    ) -> dict[str, Any]:
        if source is None:
            return {
                "available_datasets": [],
                "available_variables": [],
                "data_latest_date": None,
            }
        if not isinstance(source, ProductVariableSeries):
            return {
                "available_datasets": ["真实净值"],
                "available_variables": ["adjusted_nav", "returns", "log_returns"],
                "data_latest_date": source.data_latest_date,
            }
        dataset_labels = {
            "nav": "基金净值" if source.identity.kind == "fund" else "ETF复权净值",
            "candle": "ETF日线行情",
        }
        datasets = [dataset_labels.get(name, name) for name in source.fingerprints]
        available_variables = [
            variable_id
            for variable_id, coverage in source.coverage.items()
            if int(coverage.get("non_null_rows", 0) or 0) > 0
        ]
        if "adjusted_nav" in available_variables:
            available_variables.extend(
                name
                for name in ("returns", "log_returns")
                if name not in available_variables
            )
        return {
            "available_datasets": datasets,
            "available_variables": available_variables,
            "data_latest_date": source.data_latest_date,
        }

    @classmethod
    def _input_requirements_payload(
        cls,
        definition: dict[str, Any],
        source: Optional[ProductSeries | ProductVariableSeries],
        *,
        window: Optional[PeriodWindow] = None,
        window_error: Optional[ValidationError] = None,
    ) -> dict[str, Any]:
        required = canonicalize_variables(definition.get("required_variables") or [])
        items: list[dict[str, Any]] = []
        unavailable = (
            source.unavailable_variables
            if isinstance(source, ProductVariableSeries)
            else {}
        )
        coverage_by_variable = (
            source.coverage if isinstance(source, ProductVariableSeries) else {}
        )
        product_kind = source.identity.kind if source is not None else None
        for variable_id in required:
            variable = get_variable(variable_id)
            label = variable.label if variable is not None else variable_id
            detail = unavailable.get(variable_id)
            coverage = copy.deepcopy(coverage_by_variable.get(variable_id, {}))
            ratio = float(coverage.get("coverage_ratio", 0.0) or 0.0)
            actual_shape: list[int] | None = None
            if isinstance(window, VariablePeriodWindow):
                runtime_value = window.context.get(variable_id)
                if runtime_value is not None:
                    actual_shape = list(np.asarray(runtime_value).shape)
            if source is None:
                detail = {
                    "code": "DATA_NOT_FOUND",
                    "message": f"未找到当前产品可供计算“{label}”的真实数据。",
                }
            if detail:
                status = cls._requirement_status(str(detail.get("code") or ""))
                reason_code = str(detail.get("code") or "VARIABLE_UNAVAILABLE")
                reason = str(detail.get("message") or f"“{label}”暂不可用。")
            elif window_error is not None and window is None:
                status = cls._requirement_status(window_error.code)
                reason_code = window_error.code
                reason = window_error.message
            elif variable is not None and variable.source_dataset and 0.0 < ratio < 1.0:
                status = "partial"
                reason_code = "INPUT_COVERAGE_PARTIAL"
                reason = f"“{label}”存在部分缺失，将只使用共同有效日期计算。"
            else:
                status = "available"
                reason_code = None
                reason = None
            alternatives = []
            if variable is not None:
                for alternative_id in variable.alternative_variables:
                    alternative = get_variable(alternative_id)
                    alternatives.append(
                        {
                            "variable_id": alternative_id,
                            "label": alternative.label if alternative is not None else alternative_id,
                        }
                    )
            items.append(
                {
                    "variable_id": variable_id,
                    "label": label,
                    "canonical_field": variable_id,
                    "status": status,
                    "reason_code": reason_code,
                    "reason": reason,
                    "source_configured": bool(
                        variable is not None
                        and product_kind is not None
                        and product_kind in variable.product_kinds
                    ),
                    "source_dataset": variable.source_dataset if variable is not None else None,
                    "source_field": variable.source_field if variable is not None else None,
                    "coverage_ratio": coverage.get("coverage_ratio"),
                    "non_null_count": coverage.get("non_null_rows"),
                    "first_date": coverage.get("first_date"),
                    "latest_date": coverage.get("latest_date"),
                    "actual_shape": actual_shape,
                    "alternative_variables": alternatives,
                }
            )
        blocking = [
            item
            for item in items
            if item["status"]
            in {"source_unavailable", "field_missing", "no_observations", "unavailable"}
        ]
        partial = [item for item in items if item["status"] == "partial"]
        if blocking:
            status = "blocked"
        elif window_error is not None and window is None:
            status = "insufficient"
        elif partial:
            status = "partial"
        else:
            status = "ready"
        return {
            "status": status,
            "required_count": len(items),
            "available_count": len(items) - len(blocking),
            "items": items,
            "blocking_inputs": blocking,
            "partial_inputs": partial,
            "reason": (
                {"code": window_error.code, "message": window_error.message}
                if window_error is not None and not blocking
                else None
            ),
        }

    @staticmethod
    def _input_requirement_warning(
        definition: dict[str, Any],
        requirements: dict[str, Any],
    ) -> dict[str, str]:
        blocking = list(requirements.get("blocking_inputs") or [])
        labels = "、".join(str(item.get("label") or item.get("variable_id")) for item in blocking)
        details = "；".join(str(item.get("reason") or "") for item in blocking if item.get("reason"))
        source_only = bool(blocking) and all(
            item.get("status") == "source_unavailable" for item in blocking
        )
        no_observations = bool(blocking) and all(
            item.get("status") == "no_observations" for item in blocking
        )
        return {
            "code": (
                "INDICATOR_NOT_APPLICABLE"
                if source_only
                else "DATA_NOT_FOUND"
                if no_observations
                else "VARIABLE_UNAVAILABLE"
            ),
            "message": (
                f"无法计算“{definition['name']}”：缺少{labels}。{details}"
                if labels
                else f"无法计算“{definition['name']}”：所需输入不可用。"
            ),
        }

    @staticmethod
    def _partial_input_warnings(requirements: dict[str, Any]) -> list[dict[str, str]]:
        partial = list(requirements.get("partial_inputs") or [])
        if not partial:
            return []
        labels = "、".join(str(item.get("label") or item.get("variable_id")) for item in partial)
        return [
            {
                "code": "INPUT_COVERAGE_PARTIAL",
                "message": f"{labels}存在部分缺失，本次仅使用共同有效日期计算。",
            }
        ]

    def _compile_runtime(
        self,
        definition: dict[str, Any],
        period: str,
    ) -> IndicatorRuntime | TypedIndicatorRuntime:
        if self._is_typed_definition(definition):
            if definition.get("context_kind") != "single_product":
                raise ValidationError(
                    "CONTEXT_KIND_MISMATCH",
                    "组合指标必须使用 evaluate-portfolio 和不可变组合运行快照。",
                    field="context_kind",
                )
            try:
                dsl_version = str(
                    definition.get("dsl_version", TYPED_DSL_VERSION)
                )
                registry_version = str(
                    definition.get(
                        "operator_registry_version", TYPED_OPERATOR_REGISTRY_VERSION
                    )
                )
                plan = _compile_typed_plan(
                    normalize_variable_latex(definition["expression"]),
                    "single_product",
                    dsl_version,
                    registry_version,
                )
                return TypedIndicatorRuntime.from_plan(plan)
            except TypedDslError as exc:
                diagnostic = exc.to_dict()
                diagnostic["field"] = "expression"
                raise ValidationError(
                    exc.code,
                    exc.message,
                    field="expression",
                    diagnostics=[diagnostic],
                ) from exc
        try:
            plan = _compile_typed_plan(
                normalize_variable_latex(definition["expression"]),
                "single_product",
                LEGACY_TYPED_DSL_VERSION,
                LEGACY_TYPED_OPERATOR_REGISTRY_VERSION,
            )
            return TypedIndicatorRuntime.from_plan(plan)
        except TypedDslError as exc:
            diagnostic = exc.to_dict()
            diagnostic["field"] = "expression"
            raise ValidationError(
                "LEGACY_NJIT_ADAPTER_UNSUPPORTED",
                "该兼容指标无法转换为 NJIT 计划，请复制并迁移公式。",
                field="expression",
                diagnostics=[diagnostic],
            ) from exc

    @staticmethod
    def _risk_free_context(
        definition: dict[str, Any], elapsed_days: float | None = None
    ) -> dict[str, float]:
        annual = float(definition.get("annual_risk_free_rate_percent", 0.0)) / 100.0
        per_period = math.pow(max(0.0, 1.0 + annual), 1.0 / 252.0) - 1.0
        elapsed = max(0.0, float(elapsed_days or 0.0))
        window_return = math.pow(max(0.0, 1.0 + annual), elapsed / 365.0) - 1.0
        return {
            "annual_risk_free_rate_decimal": annual,
            "risk_free_rate_per_observation": per_period,
            # Runtime-only alias for locked legacy definitions. New catalog
            # entries and formulas use risk_free_rate_per_observation.
            "risk_free_rate_per_period": per_period,
            "risk_free_return_window": window_return,
            "periods_per_year": 252.0,
        }

    @classmethod
    def _evaluate_runtime(
        cls,
        runtime: IndicatorRuntime | TypedIndicatorRuntime,
        definition: dict[str, Any],
        period: str,
        window: PeriodWindow,
    ) -> Optional[float]:
        elapsed_days = float(
            (window.frame.iloc[-1]["date"] - window.frame.iloc[0]["date"]).days
        )
        if isinstance(window, VariablePeriodWindow):
            context = dict(window.context)
        else:
            context = {
                "returns": window.returns,
                "log_returns": window.log_returns,
            }
        context.update(cls._risk_free_context(definition, elapsed_days))
        with np.errstate(all="ignore"):
            if isinstance(runtime, TypedIndicatorRuntime):
                raw = runtime.compute(context)
            else:
                raw = runtime.compute_period(period, context)[definition["name"]]
        if isinstance(raw, (bool, np.bool_)) or not np.isscalar(raw):
            return None
        value = float(raw)
        return value if math.isfinite(value) else None

    def _rolling_series(
        self,
        runtime: IndicatorRuntime | TypedIndicatorRuntime,
        definition: dict[str, Any],
        product_series: ProductSeries | ProductVariableSeries,
        period: str,
        as_of: Optional[str],
    ) -> list[dict[str, Any]]:
        frame = product_series.frame
        if as_of:
            try:
                cutoff = datetime.fromisoformat(as_of).date()
                frame = frame[frame["date"].dt.date <= cutoff]
            except ValueError:
                return []
        dates = frame["date"].iloc[-MAX_ROLLING_POINTS:]
        points: list[dict[str, Any]] = []
        for date in dates:
            date_token = date.strftime("%Y-%m-%d")
            try:
                if isinstance(product_series, ProductVariableSeries):
                    window = select_variable_window(
                        product_series,
                        period,
                        date_token,
                        max_observations=MAX_SERIES_OBSERVATIONS,
                    )
                else:
                    window = select_period_window(
                        product_series,
                        period,
                        date_token,
                        max_observations=MAX_SERIES_OBSERVATIONS,
                    )
                value = self._evaluate_runtime(runtime, definition, period, window)
            except (
                IndicatorDomainError,
                FloatingPointError,
                OverflowError,
                TypeError,
                ValueError,
                ZeroDivisionError,
            ):
                continue
            if value is not None:
                points.append({"date": date_token, "value": value})
        return points

    def _evaluate_one(
        self,
        definition: dict[str, Any],
        runtime: IndicatorRuntime | TypedIndicatorRuntime,
        target: dict[str, str],
        product_series: Optional[ProductSeries | ProductVariableSeries],
        period: str,
        as_of: Optional[str],
        include_series: bool,
        *,
        preselected_window: PeriodWindow | object = _WINDOW_NOT_SELECTED,
        preselection_error: ValidationError | None = None,
    ) -> dict[str, Any]:
        target_name = product_series.identity.name if product_series else target["product_id"]
        base = self._result_base(definition, target, target_name, period)
        initial_requirements = self._input_requirements_payload(
            definition,
            product_series,
            window_error=preselection_error,
        )
        target_data = self._target_data_payload(product_series)
        if initial_requirements["status"] == "blocked":
            return {
                **base,
                "value": None,
                "status": "unavailable",
                "warnings": [
                    self._input_requirement_warning(definition, initial_requirements)
                ],
                "window": self._empty_window(product_series, as_of),
                "input_requirements": initial_requirements,
                "target_data": target_data,
            }
        if product_series is None:
            return {
                **base,
                "value": None,
                "status": "error",
                "warnings": [{"code": "DATA_NOT_FOUND", "message": "未找到该产品的真实净值数据。"}],
                "window": self._empty_window(None, as_of),
                "input_requirements": initial_requirements,
                "target_data": target_data,
            }
        if preselection_error is not None:
            return {
                **base,
                "value": None,
                "status": "warning",
                "warnings": [
                    {
                        "code": preselection_error.code,
                        "message": preselection_error.message,
                    }
                ],
                "window": self._empty_window(product_series, as_of),
                "input_requirements": initial_requirements,
                "target_data": target_data,
            }
        if preselected_window is _WINDOW_NOT_SELECTED:
            try:
                if isinstance(product_series, ProductVariableSeries):
                    window = select_variable_window(
                        product_series,
                        period,
                        as_of,
                        max_observations=MAX_SERIES_OBSERVATIONS,
                    )
                else:
                    window = select_period_window(
                        product_series,
                        period,
                        as_of,
                        max_observations=MAX_SERIES_OBSERVATIONS,
                    )
            except ValidationError as exc:
                requirements = self._input_requirements_payload(
                    definition,
                    product_series,
                    window_error=exc,
                )
                return {
                    **base,
                    "value": None,
                    "status": "unavailable" if requirements["status"] == "blocked" else "warning",
                    "warnings": (
                        [self._input_requirement_warning(definition, requirements)]
                        if requirements["status"] == "blocked"
                        else [{"code": exc.code, "message": exc.message}]
                    ),
                    "window": self._empty_window(product_series, as_of),
                    "input_requirements": requirements,
                    "target_data": target_data,
                }
        else:
            if not isinstance(preselected_window, PeriodWindow):
                raise TypeError("preselected_window 必须是 PeriodWindow。")
            window = preselected_window
        runtime_warning: Optional[dict[str, Any]] = None
        try:
            value = self._evaluate_runtime(runtime, definition, period, window)
        except TypedDslError as exc:
            value = None
            runtime_warning = exc.to_dict()
        except (
            FloatingPointError,
            OverflowError,
            RuntimeError,
            TypeError,
            ValueError,
            ZeroDivisionError,
        ):
            value = None
        requirements = self._input_requirements_payload(
            definition,
            product_series,
            window=window,
        )
        warnings = [*window.warnings, *self._partial_input_warnings(requirements)]
        if runtime_warning is not None:
            warnings.append(runtime_warning)
        if value is None:
            if runtime_warning is None:
                warnings.append(
                    {
                        "code": "NON_FINITE_RESULT",
                        "message": "公式计算结果不是有限标量，请检查除零或数据范围。",
                    }
                )
        record: dict[str, Any] = {
            **base,
            "value": value,
            "status": "ok" if value is not None and not warnings else "warning",
            "warnings": warnings,
            "window": self._window_payload(window),
            "input_requirements": requirements,
            "target_data": target_data,
        }
        if include_series:
            record["series"] = self._rolling_series(runtime, definition, product_series, period, as_of)
        if isinstance(runtime, TypedIndicatorRuntime):
            record["runtime_trace"] = runtime.trace_payload()
        return record

    def evaluate(
        self,
        *,
        indicator_ids: list[str],
        inline_definition: Optional[dict[str, Any]],
        targets: list[dict[str, Any]],
        period: str,
        as_of: Optional[str] = None,
        include_series: bool = False,
        indicator_versions: Optional[dict[str, int]] = None,
        prefer_snapshot: bool = True,
    ) -> dict[str, Any]:
        period = period.upper()
        if period not in SUPPORTED_PERIODS:
            raise ValidationError("INVALID_PERIOD", "不支持的评价周期。", field="period")
        definitions = self._resolve_evaluation_definitions(
            indicator_ids,
            inline_definition,
            indicator_versions=indicator_versions,
        )
        normalized_targets = self._validate_targets(targets)
        combinations = len(definitions) * len(normalized_targets)
        if combinations > MAX_COMBINATIONS:
            raise ValidationError("COMBINATION_LIMIT_EXCEEDED", "指标与产品组合数不能超过 500。")
        if include_series and combinations > MAX_ROLLING_COMBINATIONS:
            raise ValidationError("ROLLING_LIMIT_EXCEEDED", "滚动曲线每次最多计算 10 个指标-产品组合。")
        unsupported = [item["name"] for item in definitions if period not in item["periods"]]
        if unsupported:
            raise ValidationError(
                "PERIOD_NOT_SUPPORTED",
                f"以下指标未启用 {period}：{'、'.join(unsupported)}",
                field="period",
            )
        wrong_domain = [
            item["name"] for item in definitions if item.get("context_kind", "single_product") != "single_product"
        ]
        if wrong_domain:
            raise ValidationError(
                "CONTEXT_KIND_MISMATCH",
                f"以下组合指标不能在产品评价中运行：{'、'.join(wrong_domain)}。",
                field="indicator_ids",
            )

        if (
            prefer_snapshot
            and inline_definition is None
            and as_of is None
            and not include_series
        ):
            snapshot_result = self._evaluate_from_snapshot(
                definitions,
                normalized_targets,
                period,
            )
            if snapshot_result is not None:
                return snapshot_result

        runtimes = {
            self._definition_cache_key(item): self._compile_runtime(item, period)
            for item in definitions
        }
        prepared: list[dict[str, Any]] = []
        prepared_by_index: dict[int, dict[str, Any]] = {}
        legacy_series_by_target: dict[
            tuple[str, str], Optional[ProductSeries]
        ] = {}
        for index, definition in enumerate(definitions):
            runtime = runtimes[self._definition_cache_key(definition)]
            if not isinstance(runtime, TypedIndicatorRuntime):
                raise ValidationError(
                    "NJIT_RUNTIME_REQUIRED",
                    "生产计算只允许执行已编译的 NJIT 指标计划。",
                )
            if not self._is_typed_definition(definition):
                continue
            dependencies = canonicalize_variables(runtime.plan.context_requirements)
            entry = {
                "index": index,
                "item": {"period": period},
                "definition": definition,
                "runtime": runtime,
                "dependencies": dependencies,
                "data_dependencies": self._physical_dependency_signature(
                    dependencies
                ),
                "typed": True,
            }
            prepared.append(entry)
            prepared_by_index[index] = entry
        if len(prepared) != len(definitions):
            for target in normalized_targets:
                key = (target["kind"], target["product_id"])
                legacy_series_by_target[key] = load_product_series(
                    target["kind"],
                    target["product_id"],
                    self.market_data_dir,
                )

        typed_series_by_target: dict[
            tuple[str, str, tuple[str, ...]], Optional[ProductVariableSeries]
        ] = {}
        selected_by_target: dict[
            tuple[str, str, tuple[str, ...]],
            tuple[VariablePeriodWindow | None, ValidationError | None],
        ] = {}
        batch_values: dict[tuple[int, str, str], tuple[float, int]] = {}
        batch_plan_ids: list[str] = []
        parallel_tasks = 0
        for kind in sorted({target["kind"] for target in normalized_targets}):
            kind_targets = [
                target for target in normalized_targets if target["kind"] == kind
            ]
            product_ids = [target["product_id"] for target in kind_targets]
            series_by_dependency: dict[
                tuple[str, ...], dict[str, ProductVariableSeries]
            ] = {}
            if not prepared:
                continue
            selected_windows: dict[
                tuple[tuple[str, ...], str, str],
                tuple[VariablePeriodWindow | None, ValidationError | None],
            ] = {}
            for dependencies in sorted(
                {entry["data_dependencies"] for entry in prepared}
            ):
                sources = load_product_variable_series_batch(
                    kind,  # type: ignore[arg-type]
                    product_ids,
                    dependencies,
                    self.market_data_dir,
                    as_of,
                )
                series_by_dependency[dependencies] = sources
                for product_id in product_ids:
                    source = sources.get(product_id)
                    key = (kind, product_id, dependencies)
                    typed_series_by_target[key] = source
                    try:
                        if source is None:
                            raise ValidationError(
                                "DATA_NOT_FOUND",
                                "未找到该产品的真实净值数据。",
                            )
                        window = select_variable_window_fast(
                            source,
                            period,
                            as_of,
                            max_observations=MAX_SERIES_OBSERVATIONS,
                            index=prepare_variable_window_index(source),
                        )
                        outcome: tuple[
                            VariablePeriodWindow | None, ValidationError | None
                        ] = (window, None)
                    except ValidationError as exc:
                        outcome = (None, exc)
                    selected_by_target[key] = outcome
                    selected_windows[(dependencies, period, product_id)] = outcome
            computed, batch_meta = self._run_fused_typed_groups(
                prepared=prepared,
                product_ids=product_ids,
                series_by_dependency=series_by_dependency,
                selected_windows=selected_windows,
                thread_budget=max(1, min(self.compute_engine.worker_count, os.cpu_count() or 1)),
            )
            batch_plan_ids.extend(batch_meta.get("compiled_plan_ids", []))
            parallel_tasks += int(batch_meta.get("parallel_tasks", 0))
            for (definition_index, row_index), outcome in computed.items():
                batch_values[
                    (definition_index, kind, product_ids[row_index])
                ] = outcome

        results: list[dict[str, Any]] = []
        hits = 0
        misses = 0
        for definition_index, definition in enumerate(definitions):
            definition_key = self._definition_cache_key(definition)
            runtime = runtimes[definition_key]
            entry = prepared_by_index.get(definition_index)
            dependencies = entry["data_dependencies"] if entry is not None else ()
            for target in normalized_targets:
                target_key = (target["kind"], target["product_id"])
                if entry is None:
                    product_series = legacy_series_by_target.get(target_key)
                    window = None
                    window_error = None
                else:
                    product_series = typed_series_by_target.get(
                        (*target_key, dependencies)
                    )
                    window, window_error = selected_by_target.get(
                        (*target_key, dependencies),
                        (
                            None,
                            ValidationError(
                                "DATA_NOT_FOUND", "未找到该产品的真实净值数据。"
                            ),
                        ),
                    )
                fingerprint = product_series.fingerprint if product_series else "missing"
                latest = product_series.data_latest_date if product_series else "missing"
                raw_key = (
                    definition_key,
                    target["kind"],
                    target["product_id"],
                    period,
                    as_of,
                    period_cache_reference(as_of),
                    fingerprint,
                    latest,
                    dependencies,
                    tuple(sorted(getattr(product_series, "fingerprints", {}).items())),
                    include_series,
                )
                cache_key = hashlib.sha256(repr(raw_key).encode("utf-8")).hexdigest()
                cached = self.cache.get(cache_key)
                if cached is not None:
                    results.append(cached)
                    hits += 1
                    continue
                fused = batch_values.get(
                    (definition_index, target["kind"], target["product_id"])
                )
                if (
                    fused is not None
                    and fused[1] == STATUS_OK
                    and product_series is not None
                    and window is not None
                    and window_error is None
                    and not include_series
                ):
                    record = self._precomputed_result(
                        definition,
                        target,
                        product_series,
                        period,
                        window,
                        fused[0],
                        fused[1],
                    )
                    record["runtime_trace"] = {
                        **runtime.compiled_plan.metadata(),
                        "batch_plan_ids": list(batch_plan_ids),
                        "nodes": [],
                        "runtime_window_shape": [
                            int(window.observation_count)
                        ],
                    }
                else:
                    record = self._evaluate_one(
                        definition,
                        runtime,
                        target,
                        product_series,
                        period,
                        as_of,
                        include_series,
                        preselected_window=(
                            window
                            if entry is not None and window is not None
                            else _WINDOW_NOT_SELECTED
                        ),
                        preselection_error=(
                            window_error if entry is not None else None
                        ),
                    )
                self.cache.put(cache_key, record)
                results.append(record)
                misses += 1
        statuses = {status: sum(item["status"] == status for item in results) for status in ("ok", "warning", "error")}
        unavailable_count = sum(item["status"] == "unavailable" for item in results)
        if unavailable_count:
            statuses["unavailable"] = unavailable_count
        return {
            "results": results,
            "summary": {"total": len(results), **statuses},
            "cache": {"hits": hits, "misses": misses},
            "execution": {
                **kernel_registry_status(),
                "compile_cache_hits": _compile_typed_plan.cache_info().hits,
                "compile_cache_misses": _compile_typed_plan.cache_info().misses,
                "kernel_cache_hits": 0,
                "kernel_cache_misses": 0,
                "compiled_plan_ids": list(dict.fromkeys(batch_plan_ids)),
                "parallel_tasks": parallel_tasks,
                "python_fallback": 0,
                "python_operator_calls": 0,
            },
        }

    @staticmethod
    def _portfolio_window(snapshot: dict[str, Any]) -> dict[str, Any]:
        return {
            "requested_as_of": snapshot.get("requested_as_of"),
            "effective_as_of": snapshot.get("effective_as_of"),
            "start_date": snapshot.get("actual_start_date"),
            "end_date": snapshot.get("actual_end_date"),
            "observation_count": int(snapshot.get("observation_count") or 0),
            "data_latest_date": snapshot.get("effective_as_of"),
        }

    @classmethod
    def _portfolio_context(
        cls,
        snapshot: dict[str, Any],
        definition: dict[str, Any],
    ) -> dict[str, Any]:
        asset_returns = np.asarray(snapshot.get("asset_returns"), dtype=np.float64)
        weight_path = np.asarray(snapshot.get("daily_weights"), dtype=np.float64)
        if asset_returns.ndim != 2 or not asset_returns.size:
            raise ValidationError("SNAPSHOT_DATA_INVALID", "组合运行快照中的收益矩阵无效。")
        if weight_path.shape != asset_returns.shape:
            raise ValidationError("SNAPSHOT_DATA_INVALID", "组合运行快照中的权重路径与收益矩阵不一致。")
        if not np.all(np.isfinite(asset_returns)) or not np.all(np.isfinite(weight_path)):
            raise ValidationError("SNAPSHOT_DATA_INVALID", "组合运行快照包含非有限收益或权重。")
        if not np.allclose(np.sum(weight_path, axis=1), 1.0, rtol=0.0, atol=1e-8):
            raise ValidationError("SNAPSHOT_DATA_INVALID", "组合运行快照中的每日生效权重合计必须为 1。")

        # daily_weights stores the beginning-of-day weights that are effective
        # for the matching asset_returns row.  This is the realized portfolio
        # path: after each return, holdings drift with market value until the
        # next scheduled rebalance.  A terminal weight vector must never be
        # applied retrospectively to the full history.
        derived_portfolio_returns = np.sum(weight_path * asset_returns, axis=1)
        stored_returns = snapshot.get("portfolio_returns")
        if stored_returns is None:
            portfolio_returns = derived_portfolio_returns
        else:
            portfolio_returns = np.asarray(stored_returns, dtype=np.float64)
            if portfolio_returns.shape != (asset_returns.shape[0],):
                raise ValidationError("SNAPSHOT_DATA_INVALID", "组合运行快照中的组合收益序列长度无效。")
            if not np.all(np.isfinite(portfolio_returns)) or not np.allclose(
                portfolio_returns,
                derived_portfolio_returns,
                rtol=1e-10,
                atol=1e-12,
            ):
                raise ValidationError(
                    "SNAPSHOT_DATA_INVALID",
                    "组合收益序列与每日生效权重及底层产品收益不一致。",
                )
        context: dict[str, Any] = {
            "asset_returns": asset_returns,
            "asset_log_returns": np.log1p(asset_returns),
            "portfolio_returns": np.ascontiguousarray(portfolio_returns),
            "asset_weights": weight_path[-1],
            "weight_path": weight_path,
            **cls._risk_free_context(
                definition,
                (
                    pd.Timestamp(snapshot.get("actual_end_date"))
                    - pd.Timestamp(snapshot.get("actual_start_date"))
                ).days
                if snapshot.get("actual_start_date") and snapshot.get("actual_end_date")
                else 0,
            ),
        }
        benchmark = snapshot.get("benchmark_returns")
        if benchmark is not None:
            benchmark_returns = np.asarray(benchmark, dtype=np.float64)
            if benchmark_returns.shape != (asset_returns.shape[0],):
                raise ValidationError("SNAPSHOT_DATA_INVALID", "组合运行快照中的基准收益序列长度无效。")
            context["benchmark_returns"] = benchmark_returns
        return context

    def evaluate_portfolio(
        self,
        *,
        run_id: str,
        indicator_ids: list[str],
        inline_definition: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        snapshot = self.portfolio_runs.get(run_id)
        return self.evaluate_portfolio_snapshot(
            indicator_ids,
            snapshot,
            inline_definition=inline_definition,
        )

    def evaluate_portfolio_snapshot(
        self,
        indicator_ids: list[str],
        snapshot: dict[str, Any],
        inline_definition: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        definitions = self._resolve_evaluation_definitions(indicator_ids, inline_definition)
        run_id = str(snapshot.get("id") or "unsaved-run")
        window = self._portfolio_window(snapshot)
        target = {
            "kind": "portfolio",
            "product_id": run_id,
            "name": str(snapshot.get("target_name") or run_id),
        }
        results: list[dict[str, Any]] = []
        hits = 0
        misses = 0
        for definition in definitions:
            base = {
                "indicator_id": definition.get("id"),
                "indicator_revision": definition.get("revision"),
                "indicator_name": definition["name"],
                "target": target,
                "period": "snapshot",
                "window": window,
                "unit": definition.get("unit", ""),
                "display_format": definition.get("display_format", "number"),
                "presentation": metric_presentation(definition),
            }
            if not self._is_typed_definition(definition) or definition.get("context_kind") != "portfolio":
                results.append(
                    {
                        **base,
                        "value": None,
                        "status": "error",
                        "warnings": [
                            {
                                "code": "CONTEXT_KIND_MISMATCH",
                                "message": "组合运行只能计算 typed v2 的 portfolio 指标。",
                            }
                        ],
                    }
                )
                misses += 1
                continue

            definition_key = self._definition_cache_key(definition)
            snapshot_portfolio_returns = snapshot.get("portfolio_returns")
            cache_payload = {
                "kind": "portfolio-indicator-v2",
                "definition": definition_key,
                "run_id": run_id,
                "dsl_version": definition.get("dsl_version"),
                "operator_registry_version": definition.get("operator_registry_version"),
                "context_schema": snapshot.get("context_schema"),
                "asset_order": snapshot.get("asset_order"),
                "common_date_hash": snapshot.get("common_date_hash"),
                "data_fingerprints": snapshot.get("data_fingerprints"),
                "weight_path": hashlib.sha256(
                    np.asarray(snapshot.get("daily_weights"), dtype=np.float64).tobytes()
                ).hexdigest(),
                "portfolio_returns": hashlib.sha256(
                    np.asarray(
                        [] if snapshot_portfolio_returns is None else snapshot_portfolio_returns,
                        dtype=np.float64,
                    ).tobytes()
                ).hexdigest(),
            }
            cache_key = hashlib.sha256(
                json.dumps(cache_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
            ).hexdigest()
            cached = self.cache.get(cache_key)
            if cached is not None:
                results.append(cached)
                hits += 1
                continue
            warnings = [copy.deepcopy(item) for item in (snapshot.get("warnings") or [])]
            try:
                dsl_version = str(
                    definition.get("dsl_version", TYPED_DSL_VERSION)
                )
                registry_version = str(
                    definition.get(
                        "operator_registry_version", TYPED_OPERATOR_REGISTRY_VERSION
                    )
                )
                plan = _compile_typed_plan(
                    normalize_variable_latex(definition["expression"]),
                    "portfolio",
                    dsl_version,
                    registry_version,
                )
                if {"asset_returns", "asset_weights"}.issubset(plan.context_requirements):
                    warnings.append(
                        {
                            "code": "STATIC_WEIGHT_HISTORY_ASSUMPTION",
                            "message": (
                                "该兼容公式会把期末权重应用于整段历史收益，仅适合当前截面估算；"
                                "历史组合表现应使用组合实际收益率序列。"
                            ),
                        }
                    )
                runtime = TypedIndicatorRuntime.from_plan(plan)
                context = self._portfolio_context(snapshot, definition)
                with np.errstate(all="ignore"):
                    raw = runtime.compute(context)
                value = float(raw) if np.isscalar(raw) and not isinstance(raw, (bool, np.bool_)) else None
                if value is None or not math.isfinite(value):
                    value = None
                    warnings.append(
                        {"code": "NON_FINITE_RESULT", "message": "组合指标计算结果不是有限标量。"}
                    )
                trace = runtime.trace_payload()
            except TypedDslError as exc:
                value = None
                warnings.append(exc.to_dict())
                trace = None
            except IndicatorDomainError as exc:
                value = None
                warnings.append({"code": exc.code, "message": exc.message})
                trace = None
            except (FloatingPointError, OverflowError, TypeError, ValueError, ZeroDivisionError):
                value = None
                warnings.append(
                    {"code": "NON_FINITE_RESULT", "message": "组合指标计算失败，请检查输入 shape 与数值范围。"}
                )
                trace = None
            record: dict[str, Any] = {
                **base,
                "value": value,
                "status": "ok" if value is not None and not warnings else "warning",
                "warnings": warnings,
            }
            if trace is not None:
                record["runtime_trace"] = trace
            self.cache.put(cache_key, record)
            results.append(record)
            misses += 1

        statuses = {
            status: sum(item["status"] == status for item in results)
            for status in ("ok", "warning", "error")
        }
        unavailable_count = sum(item["status"] == "unavailable" for item in results)
        if unavailable_count:
            statuses["unavailable"] = unavailable_count
        return {
            "run_id": run_id,
            "results": results,
            "summary": {"total": len(results), **statuses},
            "cache": {"hits": hits, "misses": misses},
        }

    def availability(
        self,
        *,
        kind: str | None = None,
        product_id: str | None = None,
        targets: Optional[list[dict[str, str]]] = None,
        variable_ids: Optional[list[str]] = None,
        period: str = "1Y",
        as_of: Optional[str] = None,
    ) -> dict[str, Any]:
        """Describe registry and target-specific input availability."""

        if targets:
            if kind is not None or product_id is not None:
                raise ValidationError(
                    "AVAILABILITY_TARGET_CONFLICT",
                    "targets 不能与 kind、product_id 同时提供。",
                    field="targets",
                )
            if len(targets) > 10:
                raise ValidationError(
                    "AVAILABILITY_TARGET_LIMIT",
                    "变量可用性一次最多核对 10 个产品。",
                    field="targets",
                )
            normalized_targets = self._validate_targets(targets)
            results = [
                self.availability(
                    kind=target["kind"],
                    product_id=target["product_id"],
                    variable_ids=variable_ids,
                    period=period,
                    as_of=as_of,
                )
                for target in normalized_targets
            ]
            aggregate_items: list[dict[str, Any]] = []
            requested = canonicalize_variables(
                variable_ids
                or [item["id"] for item in variable_catalog("single_product")]
            )
            for variable_id in requested:
                target_items = []
                for result in results:
                    item = next(
                        (
                            candidate
                            for candidate in result["items"]
                            if candidate["variable_id"] == variable_id
                        ),
                        None,
                    )
                    if item is not None:
                        target_items.append(
                            {
                                "target": copy.deepcopy(result["target"]),
                                "status": item["status"],
                                "reason": copy.deepcopy(item.get("reason")),
                                "coverage": copy.deepcopy(item.get("coverage", {})),
                                "actual_shape": copy.deepcopy(item.get("actual_shape")),
                            }
                        )
                available_count = sum(
                    item["status"] in {"available", "partial"}
                    for item in target_items
                )
                partial_count = sum(
                    item["status"] == "partial" for item in target_items
                )
                if available_count == len(target_items) and not partial_count:
                    aggregate_status = "available"
                elif available_count:
                    aggregate_status = "partial"
                else:
                    statuses = {item["status"] for item in target_items}
                    aggregate_status = (
                        next(iter(statuses)) if len(statuses) == 1 else "unavailable"
                    )
                definition = get_variable(variable_id)
                aggregate_items.append(
                    {
                        "variable_id": variable_id,
                        "label": definition.label if definition is not None else variable_id,
                        "status": aggregate_status,
                        "available_target_count": available_count,
                        "target_count": len(target_items),
                        "reason": {
                            "code": "VARIABLE_AVAILABILITY_SUMMARY",
                            "message": f"所选产品中 {available_count}/{len(target_items)} 个可使用该变量。",
                        },
                        "actual_shape": next(
                            (
                                item["actual_shape"]
                                for item in target_items
                                if item.get("actual_shape") is not None
                            ),
                            None,
                        ),
                        "target_statuses": target_items,
                    }
                )
            return {
                "target": None,
                "targets": results,
                "period": str(period or "1Y").upper(),
                "as_of": as_of,
                "variable_registry_version": VARIABLE_REGISTRY_VERSION,
                "data_contract_version": DATA_CONTRACT_VERSION,
                "summary": {
                    "target_count": len(results),
                    "variable_count": len(aggregate_items),
                },
                "items": aggregate_items,
            }

        if bool(kind) != bool(product_id):
            raise ValidationError(
                "AVAILABILITY_TARGET_INCOMPLETE",
                "kind 与 product_id 必须同时提供。",
                field="target",
            )
        if kind is not None and kind not in {"etf", "fund"}:
            raise ValidationError("INVALID_TARGET", "产品类型必须为 etf 或 fund。", field="kind")
        period = str(period or "1Y").upper()
        if period not in SUPPORTED_PERIODS:
            raise ValidationError(
                "INVALID_PERIOD",
                f"不支持的计算周期: {period}。",
                field="period",
            )
        requested = canonicalize_variables(
            variable_ids
            or [item["id"] for item in variable_catalog("single_product")]
        )
        unknown = [name for name in requested if get_variable(name) is None]
        if unknown:
            raise ValidationError(
                "UNKNOWN_VARIABLE",
                f"未知变量: {', '.join(unknown)}。",
                field="variable_ids",
            )
        if kind is None:
            items = []
            for variable_id in requested:
                definition = get_variable(variable_id)
                assert definition is not None
                items.append(
                    {
                        "variable_id": variable_id,
                        "label": definition.label,
                        "canonical_field": variable_id,
                        "status": "declared",
                        "product_kinds": list(definition.product_kinds),
                        "conditional": definition.conditional,
                        "source_dataset": definition.source_dataset,
                        "source_field": definition.source_field,
                        "source_bindings": definition.to_catalog_entry()[
                            "source_bindings"
                        ],
                        "availability_policy": "runtime_required",
                        "alternative_variables": [
                            {
                                "variable_id": alternative_id,
                                "label": (
                                    get_variable(alternative_id).label
                                    if get_variable(alternative_id) is not None
                                    else alternative_id
                                ),
                            }
                            for alternative_id in definition.alternative_variables
                        ],
                        "availability_rule": definition.availability_rule,
                    }
                )
            return {
                "target": None,
                "period": period,
                "as_of": as_of,
                "variable_registry_version": VARIABLE_REGISTRY_VERSION,
                "data_contract_version": DATA_CONTRACT_VERSION,
                "items": items,
            }

        source = load_product_variable_series(
            kind,  # type: ignore[arg-type]
            str(product_id),
            requested,
            self.market_data_dir,
            as_of,
        )
        runtime_windows: dict[str, VariablePeriodWindow] = {}
        window_errors: dict[str, dict[str, Any]] = {}
        # Availability is a per-variable contract. A sparse optional NAV field
        # must not shrink the reported shape of an unrelated OHLCV variable.
        # Load each dependency with the adjusted-NAV anchor used at runtime.
        for variable_id in requested:
            if source is not None and variable_id in source.unavailable_variables:
                continue
            single_source = load_product_variable_series(
                kind,  # type: ignore[arg-type]
                str(product_id),
                [variable_id],
                self.market_data_dir,
                as_of,
            )
            if single_source is None or single_source.unavailable_variables:
                continue
            try:
                runtime_windows[variable_id] = select_variable_window(
                    single_source,
                    period,
                    as_of,
                    MAX_SERIES_OBSERVATIONS,
                )
            except ValidationError as exc:
                window_errors[variable_id] = exc.detail()
        items = []
        for variable_id in requested:
            definition = get_variable(variable_id)
            assert definition is not None
            detail = source.unavailable_variables.get(variable_id) if source else None
            coverage = copy.deepcopy(source.coverage.get(variable_id, {})) if source else {}
            if detail:
                status_value = self._requirement_status(
                    str(detail.get("code") or "VARIABLE_UNAVAILABLE")
                )
            elif kind not in definition.product_kinds:
                status_value = "not_applicable"
            elif definition.source_dataset and 0 < float(coverage.get("coverage_ratio", 0.0)) < 1:
                status_value = "partial"
            else:
                status_value = "available"
            actual_shape: list[int] | None = None
            runtime_window = runtime_windows.get(variable_id)
            if runtime_window is not None and not detail:
                runtime_value = runtime_window.context.get(variable_id)
                if runtime_value is not None:
                    actual_shape = list(np.asarray(runtime_value).shape)
                elif definition.kind == "scalar":
                    actual_shape = []
                else:
                    status_value = "unavailable"
                    detail = {
                        "code": "VARIABLE_NOT_IN_WINDOW",
                        "message": f"变量 {variable_id} 未进入当前运行窗口。",
                    }
            elif variable_id in window_errors and not detail:
                if status_value == "available":
                    status_value = "insufficient_window"
                detail = copy.deepcopy(window_errors[variable_id])
            items.append(
                {
                    "variable_id": variable_id,
                    "label": definition.label,
                    "canonical_field": variable_id,
                    "status": status_value,
                    "conditional": definition.conditional,
                    "coverage": coverage,
                    "coverage_ratio": coverage.get("coverage_ratio"),
                    "non_null_count": coverage.get("non_null_rows"),
                    "first_date": coverage.get("first_date"),
                    "latest_date": coverage.get("latest_date"),
                    "actual_shape": actual_shape,
                    "window": self._window_payload(runtime_window) if runtime_window else None,
                    "reason": copy.deepcopy(detail),
                    "reason_code": detail.get("code") if detail else None,
                    "source_dataset": definition.source_dataset,
                    "source_field": definition.source_field,
                    "source_configured": kind in definition.product_kinds,
                    "source_bindings": definition.to_catalog_entry()[
                        "source_bindings"
                    ],
                    "alternative_variables": [
                        {
                            "variable_id": alternative_id,
                            "label": (
                                get_variable(alternative_id).label
                                if get_variable(alternative_id) is not None
                                else alternative_id
                            ),
                        }
                        for alternative_id in definition.alternative_variables
                    ],
                    "availability_rule": definition.availability_rule,
                }
            )
        return {
            "target": {
                "kind": kind,
                "product_id": product_id,
                "name": source.identity.name if source else product_id,
            },
            "as_of": as_of,
            "period": period,
            "data_latest_date": source.data_latest_date if source else None,
            "window": (
                self._window_payload(next(iter(runtime_windows.values())))
                if len(runtime_windows) == 1
                else None
            ),
            "variable_registry_version": VARIABLE_REGISTRY_VERSION,
            "data_contract_version": DATA_CONTRACT_VERSION,
            "source_fingerprints": dict(source.fingerprints) if source else {},
            "lineage": copy.deepcopy(source.lineage) if source else [],
            "warnings": copy.deepcopy(source.warnings) if source else [],
            "items": items,
        }

    @staticmethod
    def _with_plan_product_kind(plan: dict[str, Any]) -> dict[str, Any]:
        normalized = copy.deepcopy(plan)
        declared_kind = str(normalized.get("product_kind", ""))
        if declared_kind in {"etf", "fund"}:
            return normalized
        target_kinds = {
            str(target.get("kind", ""))
            for target in normalized.get("targets", [])
            if str(target.get("kind", "")) in {"etf", "fund"}
        }
        normalized["product_kind"] = next(iter(target_kinds)) if len(target_kinds) == 1 else "mixed"
        return normalized

    def list_plans(self, product_kind: Optional[str] = None) -> dict[str, Any]:
        if product_kind is not None and product_kind not in {"etf", "fund"}:
            raise ValidationError("INVALID_PRODUCT_KIND", "评价方案类型必须为 etf 或 fund。", field="kind")
        items = [self._with_plan_product_kind(item) for item in self.plans.list()]
        if product_kind is not None:
            items = [item for item in items if item["product_kind"] == product_kind]
        return {"items": items, "total": len(items)}

    def get_plan(self, plan_id: str) -> dict[str, Any]:
        return self._with_plan_product_kind(self.plans.get(plan_id))

    @staticmethod
    def _normalize_plan_product_selection(raw: Any) -> dict[str, Any]:
        if raw is None:
            return _empty_plan_product_selection()
        if not isinstance(raw, dict):
            raise ValidationError(
                "INVALID_PRODUCT_SELECTION",
                "产品筛选配置格式无效。",
                field="product_selection",
            )
        query = str(raw.get("query", "")).strip()
        if len(query) > 100:
            raise ValidationError(
                "INVALID_PRODUCT_SELECTION_QUERY",
                "产品搜索词不能超过 100 个字符。",
                field="product_selection.query",
            )
        raw_filters = raw.get("filters") or {}
        if not isinstance(raw_filters, dict):
            raise ValidationError(
                "INVALID_PRODUCT_SELECTION_FILTERS",
                "产品分类筛选格式无效。",
                field="product_selection.filters",
            )
        filters: dict[str, list[str]] = {}
        for key in PLAN_PRODUCT_FILTER_KEYS:
            raw_values = raw_filters.get(key) or []
            if not isinstance(raw_values, list) or len(raw_values) > 100:
                raise ValidationError(
                    "INVALID_PRODUCT_SELECTION_FILTERS",
                    "单个产品筛选项最多保存 100 个值。",
                    field=f"product_selection.filters.{key}",
                )
            values: list[str] = []
            for raw_value in raw_values:
                value = str(raw_value).strip()
                if not value or len(value) > 100:
                    raise ValidationError(
                        "INVALID_PRODUCT_SELECTION_FILTER_VALUE",
                        "产品筛选值长度应为 1 至 100 个字符。",
                        field=f"product_selection.filters.{key}",
                    )
                if value not in values:
                    values.append(value)
            filters[key] = values
        raw_conditions = raw.get("conditions") or []
        if not isinstance(raw_conditions, list) or len(raw_conditions) > 20:
            raise ValidationError(
                "INVALID_PRODUCT_SELECTION_CONDITIONS",
                "产品日期与指标筛选条件最多保存 20 项。",
                field="product_selection.conditions",
            )
        conditions: list[dict[str, str]] = []
        for index, raw_condition in enumerate(raw_conditions):
            if not isinstance(raw_condition, dict):
                raise ValidationError(
                    "INVALID_PRODUCT_SELECTION_CONDITION",
                    "产品筛选条件格式无效。",
                    field=f"product_selection.conditions.{index}",
                )
            field_name = str(raw_condition.get("field", "")).strip()
            operator = str(raw_condition.get("operator", "")).strip()
            value = str(raw_condition.get("value", "")).strip()
            valid_field = (
                0 < len(field_name) <= 80
                and field_name.isascii()
                and field_name.replace("_", "").isalnum()
            )
            if not valid_field or operator not in PLAN_PRODUCT_CONDITION_OPERATORS or not value or len(value) > 100:
                raise ValidationError(
                    "INVALID_PRODUCT_SELECTION_CONDITION",
                    "产品筛选条件字段、比较方式或值无效。",
                    field=f"product_selection.conditions.{index}",
                )
            conditions.append({"field": field_name, "operator": operator, "value": value})
        selection_mode = str(raw.get("selection_mode", "manual"))
        if selection_mode not in {"manual", "all_matching"}:
            raise ValidationError(
                "INVALID_PRODUCT_SELECTION_MODE",
                "产品选择模式必须为 manual 或 all_matching。",
                field="product_selection.selection_mode",
            )
        return {
            "query": query,
            "filters": filters,
            "conditions": conditions,
            "selection_mode": selection_mode,
        }

    def _normalize_plan(self, fields: dict[str, Any]) -> dict[str, Any]:
        name = str(fields.get("name", "")).strip()
        if not name or len(name) > 80:
            raise ValidationError("INVALID_PLAN_NAME", "方案名称长度应为 1 至 80 个字符。", field="name")
        targets = self._validate_targets(fields.get("targets") or [], MAX_PLAN_TARGETS)
        target_kinds = {target["kind"] for target in targets}
        if len(target_kinds) != 1:
            raise ValidationError(
                "MIXED_PRODUCT_KINDS",
                "一个评价方案只能包含 ETF 或场外公募基金中的一类产品。",
                field="targets",
            )
        target_kind = next(iter(target_kinds))
        requested_kind = str(fields.get("product_kind") or target_kind)
        if requested_kind not in {"etf", "fund"}:
            raise ValidationError(
                "INVALID_PRODUCT_KIND",
                "评价方案类型必须为 etf 或 fund。",
                field="product_kind",
            )
        if requested_kind != target_kind:
            raise ValidationError(
                "PLAN_KIND_MISMATCH",
                "评价方案类型与所选产品类型不一致。",
                field="targets",
            )
        raw_indicators = list(fields.get("indicators") or [])
        if not raw_indicators or len(raw_indicators) > MAX_INDICATORS:
            raise ValidationError("INVALID_PLAN_INDICATORS", "评价方案需要 1 至 10 个指标。", field="indicators")
        indicators: list[dict[str, Any]] = []
        compiled_entries: list[
            tuple[dict[str, Any], dict[str, Any], TypedIndicatorRuntime]
        ] = []
        indicator_keys: set[tuple[str, int, str]] = set()
        total_weight = 0.0
        for item in raw_indicators:
            indicator_id = str(item.get("indicator_id", ""))
            requested_revision = item.get("indicator_revision")
            definition = self._decorate_definition(
                self.indicators.get(
                    indicator_id,
                    int(requested_revision) if requested_revision is not None else None,
                )
            )
            if definition.get("context_kind") != "single_product":
                raise ValidationError(
                    "CONTEXT_KIND_MISMATCH",
                    f"组合指标 {definition['name']} 不能加入产品评价方案。",
                    field="indicators",
                )
            if not str(definition.get("dsl_version") or "").startswith("2."):
                raise ValidationError(
                    "PLAN_REQUIRES_NJIT_INDICATOR",
                    f"指标 {definition['name']} 是兼容指标，不能加入新的评价方案；请先迁移或复制为 typed v2.2 指标。",
                    field="indicators",
                )
            period = str(item.get("period", "")).upper()
            if period not in definition["periods"]:
                raise ValidationError("PERIOD_NOT_SUPPORTED", f"指标 {definition['name']} 未启用 {period}。")
            weight = float(item.get("weight", 0.0))
            if not math.isfinite(weight) or weight < 0:
                raise ValidationError("INVALID_WEIGHT", "指标权重必须是非负有限数值。", field="indicators")
            direction = str(item.get("direction") or definition["direction"])
            if direction not in {"higher_better", "lower_better"}:
                raise ValidationError("INVALID_DIRECTION", "不支持的优劣方向。", field="indicators")
            indicator_key = (indicator_id, int(definition["revision"]), period)
            if indicator_key in indicator_keys:
                raise ValidationError(
                    "DUPLICATE_PLAN_INDICATOR",
                    f"指标 {definition['name']} 的 {period} 配置重复。",
                    field="indicators",
                )
            indicator_keys.add(indicator_key)
            runtime = self._compile_runtime(definition, period)
            if not isinstance(runtime, TypedIndicatorRuntime):
                raise ValidationError(
                    "PLAN_REQUIRES_NJIT_INDICATOR",
                    f"指标 {definition['name']} 未生成 NJIT 计划。",
                    field="indicators",
                )
            normalized_item = {
                "indicator_id": indicator_id,
                "indicator_revision": int(definition["revision"]),
                "period": period,
                "weight": weight,
                "direction": direction,
                "compiled_plan_id": runtime.compiled_plan.plan_id,
            }
            indicators.append(normalized_item)
            compiled_entries.append((normalized_item, definition, runtime))
            total_weight += weight
        if total_weight <= 0:
            raise ValidationError("INVALID_WEIGHT", "至少一个指标权重必须大于 0。", field="indicators")
        if str(fields.get("missing_policy", "strict")) != "strict":
            raise ValidationError("INVALID_MISSING_POLICY", "首期只支持严格完整样本策略。")
        compiled_batches: list[dict[str, Any]] = []
        grouped: dict[
            tuple[tuple[str, ...], str],
            list[tuple[dict[str, Any], dict[str, Any], TypedIndicatorRuntime]],
        ] = {}
        for normalized_item, definition, runtime in compiled_entries:
            dependencies = self._physical_dependency_signature(
                runtime.plan.context_requirements
            )
            grouped.setdefault(
                (dependencies, str(normalized_item["period"])), []
            ).append((normalized_item, definition, runtime))
        try:
            for (dependencies, period), entries in grouped.items():
                physical_columns = tuple(
                    dict.fromkeys(
                        [
                            "adjusted_nav",
                            *[
                                name
                                for name in dependencies
                                if name
                                not in {"returns", "log_returns", "adjusted_nav"}
                            ],
                        ]
                    )
                )
                compiled = compile_numba_batch_plan(
                    tuple(entry[2].plan for entry in entries),
                    tuple(entry[1] for entry in entries),
                    physical_columns,
                )
                compiled_batches.append(
                    {
                        "compiled_plan_id": compiled.plan_id,
                        "period": period,
                        "dependencies": list(dependencies),
                        "metric_count": len(entries),
                    }
                )
        except (NumbaPlanCompileError, ValueError, TypeError) as exc:
            raise ValidationError(
                "NJIT_BATCH_COMPILE_FAILED",
                "评价方案无法编译为融合 NJIT 计算计划。",
                field="indicators",
            ) from exc
        return {
            "name": name,
            "description": str(fields.get("description", "")).strip()[:500],
            "product_kind": requested_kind,
            "indicators": indicators,
            "targets": targets,
            "product_selection": self._normalize_plan_product_selection(fields.get("product_selection")),
            "missing_policy": "strict",
            "engine_version": ENGINE_VERSION,
            "numeric_kernel_version": NUMERIC_KERNEL_VERSION,
            "compiled_batches": compiled_batches,
        }

    def create_plan(self, fields: dict[str, Any]) -> dict[str, Any]:
        created = self.plans.create(self._normalize_plan(fields))
        self.plan_cache.clear()
        return created

    def update_plan(self, plan_id: str, revision: int, fields: dict[str, Any]) -> dict[str, Any]:
        updated = self.plans.update(plan_id, revision, self._normalize_plan(fields))
        self.plan_cache.clear()
        return updated

    def delete_plan(self, plan_id: str, revision: int) -> None:
        self.plans.delete(plan_id, revision)
        self.plan_cache.clear()

    def _run_plan_compat(self, plan_id: str, as_of: Optional[str] = None) -> dict[str, Any]:
        plan = self.plans.get(plan_id)
        target_keys = [(target["kind"], target["product_id"]) for target in plan["targets"]]
        values_by_target: dict[tuple[str, str], list[dict[str, Any]]] = {key: [] for key in target_keys}
        target_names: dict[tuple[str, str], str] = {key: key[1] for key in target_keys}
        for item in plan["indicators"]:
            definition = self.indicators.get(item["indicator_id"], int(item["indicator_revision"]))
            for start in range(0, len(plan["targets"]), MAX_TARGETS):
                response = self.evaluate(
                    indicator_ids=[item["indicator_id"]],
                    inline_definition=None,
                    targets=plan["targets"][start : start + MAX_TARGETS],
                    period=item["period"],
                    as_of=as_of,
                    include_series=False,
                    indicator_versions={item["indicator_id"]: int(item["indicator_revision"])},
                )
                for result in response["results"]:
                    key = (result["target"]["kind"], result["target"]["product_id"])
                    target_names[key] = result["target"]["name"]
                    values_by_target[key].append(
                        {
                            "indicator_id": item["indicator_id"],
                            "indicator_revision": int(item["indicator_revision"]),
                            "indicator_name": definition["name"],
                            "period": item["period"],
                            "value": result["value"],
                            "status": result["status"],
                            "warnings": result["warnings"],
                            "window": copy.deepcopy(result.get("window")),
                            "input_requirements": copy.deepcopy(
                                result.get("input_requirements")
                            ),
                            "target_data": copy.deepcopy(result.get("target_data")),
                            "presentation": copy.deepcopy(result.get("presentation")),
                            "direction": item["direction"],
                            "definition_direction": definition.get("direction"),
                            "direction_overridden": item["direction"] != definition.get("direction"),
                            "configured_weight": float(item["weight"]),
                            "effective_weight": None,
                            "normalized_score": None,
                            "weighted_contribution": None,
                        }
                    )

        complete_keys = [
            key
            for key, values in values_by_target.items()
            if len(values) == len(plan["indicators"]) and all(value["value"] is not None for value in values)
        ]
        normalized_scores: dict[tuple[str, str], float] = {key: 0.0 for key in complete_keys}
        total_weight = sum(float(item["weight"]) for item in plan["indicators"])
        for index, item in enumerate(plan["indicators"]):
            effective_weight = float(item["weight"]) / total_weight
            for values in values_by_target.values():
                if index < len(values):
                    values[index]["effective_weight"] = effective_weight
            raw_values = [float(values_by_target[key][index]["value"]) for key in complete_keys]
            if not raw_values:
                continue
            lower = min(raw_values)
            upper = max(raw_values)
            for key, raw_value in zip(complete_keys, raw_values):
                component = 50.0 if upper == lower else (raw_value - lower) / (upper - lower) * 100.0
                if item["direction"] == "lower_better":
                    component = 100.0 - component
                contribution = component * effective_weight
                values_by_target[key][index]["normalized_score"] = component
                values_by_target[key][index]["weighted_contribution"] = contribution
                normalized_scores[key] += contribution

        ranked_keys = sorted(complete_keys, key=lambda key: normalized_scores[key], reverse=True)
        ranks = {key: index + 1 for index, key in enumerate(ranked_keys)}
        rows: list[dict[str, Any]] = []
        for target in plan["targets"]:
            key = (target["kind"], target["product_id"])
            complete = key in ranks
            values = values_by_target[key]
            rows.append(
                {
                    "rank": ranks.get(key),
                    "target": {**target, "name": target_names[key]},
                    "score": round(normalized_scores[key], 6) if complete else None,
                    "status": "ranked" if complete else "excluded",
                    "missing_indicators": [
                        value["indicator_name"] for value in values if value["value"] is None
                    ],
                    "exclusion_reasons": [
                        warning
                        for value in values
                        if value["value"] is None
                        for warning in value.get("warnings", [])
                    ],
                    "values": values,
                }
            )
        rows.sort(key=lambda row: (row["rank"] is None, row["rank"] or math.inf))
        return {
            "plan_id": plan_id,
            "plan_revision": int(plan["revision"]),
            "run_at": datetime.now(timezone.utc).isoformat(),
            "as_of": as_of,
            "rows": rows,
            "ranked_count": len(ranked_keys),
            "excluded_count": len(rows) - len(ranked_keys),
            "normalization": {
                "method": "min_max_0_100",
                "configured_weight_total": total_weight,
                "effective_weight_total": 1.0,
                "missing_policy": "strict",
            },
        }

    @staticmethod
    def _plan_value_payload(
        item: dict[str, Any],
        definition: dict[str, Any],
        result: dict[str, Any],
    ) -> dict[str, Any]:
        warnings = copy.deepcopy(result["warnings"])
        for warning in warnings:
            if warning.get("code") == "DATA_NOT_FOUND":
                warning["code"] = "PRODUCT_DATA_NOT_FOUND"
        return {
            "indicator_id": item["indicator_id"],
            "indicator_revision": int(item["indicator_revision"]),
            "indicator_name": definition["name"],
            "period": item["period"],
            "value": result["value"],
            "status": result["status"],
            "warnings": warnings,
            "window": copy.deepcopy(result.get("window")),
            "input_requirements": copy.deepcopy(result.get("input_requirements")),
            "target_data": copy.deepcopy(result.get("target_data")),
            "presentation": copy.deepcopy(result.get("presentation")),
            "direction": item["direction"],
            "definition_direction": definition.get("direction"),
            "direction_overridden": item["direction"] != definition.get("direction"),
            "configured_weight": float(item["weight"]),
            "effective_weight": None,
            "normalized_score": None,
            "weighted_contribution": None,
        }

    def _plan_run_cache_key(
        self,
        plan: dict[str, Any],
        as_of: Optional[str],
        data_generation: str,
    ) -> str:
        payload = {
            "engine": ENGINE_VERSION,
            "numeric_kernel_version": NUMERIC_KERNEL_VERSION,
            "plan_id": plan.get("id"),
            "plan_revision": plan.get("revision"),
            "indicators": plan.get("indicators"),
            "targets": plan.get("targets"),
            "as_of": as_of,
            "period_reference": period_cache_reference(as_of),
            "data_generation": data_generation,
        }
        return hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _batch_data_cache_key(
        *,
        generation: str,
        product_kind: str,
        product_ids: list[str],
        dependencies: tuple[str, ...],
        as_of: str | None,
    ) -> str:
        payload = (
            generation,
            product_kind,
            tuple(product_ids),
            dependencies,
            as_of,
            DATA_CONTRACT_VERSION,
        )
        return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()

    @staticmethod
    def _batch_series_size(
        sources: dict[str, ProductVariableSeries],
    ) -> int:
        return sum(
            int(source.frame.memory_usage(index=True, deep=True).sum()) + 1024
            for source in sources.values()
        )

    @staticmethod
    def _window_map_size(
        windows: dict[str, tuple[VariablePeriodWindow | None, ValidationError | None]],
    ) -> int:
        size = 0
        for window, _ in windows.values():
            if window is None:
                size += 256
                continue
            size += int(window.frame.memory_usage(index=True, deep=True).sum())
            size += int(window.returns.nbytes + window.log_returns.nbytes + 1024)
        return size

    @staticmethod
    def _physical_dependency_signature(
        dependencies: Iterable[str],
    ) -> tuple[str, ...]:
        """Collapse runtime scalars onto their underlying physical data inputs."""

        physical: list[str] = []
        for dependency in canonicalize_variables(dependencies):
            definition = get_variable(dependency)
            if dependency in {"returns", "log_returns", "adjusted_nav"}:
                physical.append(dependency)
            elif definition is not None and definition.kind != "scalar":
                physical.append(dependency)
        return canonicalize_variables(physical)

    @staticmethod
    def _fast_builtin_code(definition: dict[str, Any]) -> int | None:
        direct = BUILTIN_METRIC_CODE.get(str(definition.get("id") or ""))
        if direct is not None:
            return direct
        if not str(definition.get("dsl_version") or "").startswith("2."):
            return None
        expression = normalize_variable_latex(str(definition.get("expression") or ""))
        for built_in in _built_in_indicators():
            code = BUILTIN_METRIC_CODE.get(str(built_in.get("id") or ""))
            if code is not None and normalize_variable_latex(
                str(built_in.get("expression") or "")
            ) == expression:
                return code
        return None

    @classmethod
    def _precomputed_result(
        cls,
        definition: dict[str, Any],
        target: dict[str, Any],
        source: ProductVariableSeries,
        period: str,
        window: VariablePeriodWindow,
        value: float,
        status_code: int,
    ) -> dict[str, Any]:
        requirements = cls._input_requirements_payload(
            definition,
            source,
            window=window,
        )
        warnings = [*window.warnings, *cls._partial_input_warnings(requirements)]
        output_value: float | None = value if status_code == STATUS_OK else None
        warning_by_status = {
            STATUS_INSUFFICIENT_SAMPLE: {
                "code": "INSUFFICIENT_SAMPLE",
                "message": "指标所需的有效样本不足。",
            },
            STATUS_DIVIDE_BY_ZERO: {
                "code": "DIVIDE_BY_ZERO",
                "message": "指标计算发生除零。",
            },
            STATUS_DOMAIN_ERROR: {
                "code": "DOMAIN_ERROR",
                "message": "指标输入不满足数学定义域。",
            },
            STATUS_NON_FINITE_RESULT: {
                "code": "NON_FINITE_RESULT",
                "message": "公式计算结果不是有限标量，请检查除零或数据范围。",
            },
        }
        if status_code != STATUS_OK:
            warnings.append(
                warning_by_status.get(
                    status_code,
                    {
                        "code": "OPERATOR_EXECUTION_FAILED",
                        "message": "指标批量计算失败。",
                    },
                )
            )
        return {
            **cls._result_base(
                definition,
                target,
                source.identity.name,
                period,
            ),
            "value": output_value,
            "status": "ok" if output_value is not None and not warnings else "warning",
            "warnings": warnings,
            "window": cls._window_payload(window),
            "input_requirements": requirements,
            "target_data": cls._target_data_payload(source),
        }

    def _run_fused_builtin_groups(
        self,
        *,
        prepared: list[dict[str, Any]],
        product_ids: list[str],
        series_by_dependency: dict[
            tuple[str, ...], dict[str, ProductVariableSeries]
        ],
        selected_windows: dict[
            tuple[tuple[str, ...], str, str],
            tuple[VariablePeriodWindow | None, ValidationError | None],
        ],
        thread_budget: int,
    ) -> tuple[
        dict[tuple[int, int], tuple[float, int]],
        dict[str, Any],
    ]:
        """Execute supported scalar built-ins over shared arrays in workers."""

        empty_meta = {
            "metric_items": 0,
            "worker_pids": [],
            "shared_memory_bytes": 0,
            "mmap_bytes": 0,
            "parallel_tasks": 0,
        }
        # Production starts and warms the pool in FastAPI lifespan. Keeping this
        # fallback makes CLI/tests deterministic and avoids compiling at request time.
        if not self.compute_engine.status()["started"]:
            return {}, empty_meta

        grouped_entries: dict[
            tuple[str, ...], dict[str, list[tuple[dict[str, Any], int]]]
        ] = {}
        for entry in prepared:
            code = self._fast_builtin_code(entry["definition"])
            if not entry["typed"] or code is None:
                continue
            grouped_entries.setdefault(entry["data_dependencies"], {}).setdefault(
                str(entry["item"]["period"]), []
            ).append((entry, code))
        if not grouped_entries:
            return {}, empty_meta

        threshold = max(
            1,
            int(
                os.getenv(
                    "INDICATOR_SHM_THRESHOLD_BYTES",
                    str(512 * 1024 * 1024),
                )
            ),
        )
        precomputed: dict[tuple[int, int], tuple[float, int]] = {}
        task_specs: list[dict[str, Any]] = []
        shared_bytes = 0
        mmap_bytes = 0
        fast_metric_items: set[int] = set()

        with ExitStack() as stack:

            def own(array: np.ndarray) -> SharedArrayOwner:
                nonlocal shared_bytes, mmap_bytes
                owner = stack.enter_context(
                    SharedArrayOwner(
                        array,
                        self.compute_engine.runtime_dir,
                        shm_threshold_bytes=threshold,
                    )
                )
                if owner.descriptor.backend == "shm":
                    shared_bytes += owner.descriptor.nbytes
                else:
                    mmap_bytes += owner.descriptor.nbytes
                return owner

            for dependencies, entries_by_period in grouped_entries.items():
                sources = series_by_dependency[dependencies]
                physical_columns = list(
                    dict.fromkeys(
                        [
                            "adjusted_nav",
                            *[
                                name
                                for name in dependencies
                                if name
                                not in {"returns", "log_returns", "adjusted_nav"}
                            ],
                        ]
                    )
                )
                offsets = np.zeros(len(product_ids) + 1, dtype=np.int64)
                for row_index, product_id in enumerate(product_ids):
                    source = sources.get(product_id)
                    offsets[row_index + 1] = offsets[row_index] + (
                        len(source.frame) if source is not None else 0
                    )
                total_points = int(offsets[-1])
                if total_points == 0:
                    continue
                packed_values = np.empty(
                    (len(physical_columns), total_points), dtype=np.float64
                )
                for row_index, product_id in enumerate(product_ids):
                    source = sources.get(product_id)
                    start = int(offsets[row_index])
                    end = int(offsets[row_index + 1])
                    if source is None or start == end:
                        continue
                    for value_index, column in enumerate(physical_columns):
                        if column not in source.frame.columns:
                            packed_values[value_index, start:end] = np.nan
                        else:
                            packed_values[value_index, start:end] = source.frame[
                                column
                            ].to_numpy(dtype=np.float64, copy=False)
                values_owner = own(np.ascontiguousarray(packed_values))

                for period, period_entries in entries_by_period.items():
                    starts = np.full(len(product_ids), -1, dtype=np.int64)
                    ends = np.full(len(product_ids), -1, dtype=np.int64)
                    observation_total = 0
                    for row_index, product_id in enumerate(product_ids):
                        window, error = selected_windows[
                            (dependencies, period, product_id)
                        ]
                        if window is None or error is not None:
                            continue
                        local_start = int(window.frame.index[0])
                        local_end = int(window.frame.index[-1]) + 1
                        starts[row_index] = int(offsets[row_index]) + local_start
                        ends[row_index] = int(offsets[row_index]) + local_end
                        observation_total += max(0, local_end - local_start - 1)

                    codes = np.ascontiguousarray(
                        [code for _, code in period_entries], dtype=np.int64
                    )
                    primary = np.full(codes.size, -1, dtype=np.int64)
                    secondary = np.full(codes.size, -1, dtype=np.int64)
                    column_index = {
                        name: index for index, name in enumerate(physical_columns)
                    }
                    for metric_index, code in enumerate(codes):
                        if code in (30, 31):
                            primary[metric_index] = column_index.get("volume", -1)
                        elif code == 32:
                            primary[metric_index] = column_index.get("market_high", -1)
                        elif code == 33:
                            primary[metric_index] = column_index.get("market_low", -1)
                        elif code == 34:
                            primary[metric_index] = column_index.get("market_high", -1)
                            secondary[metric_index] = column_index.get("market_low", -1)
                    risk_free = np.ascontiguousarray(
                        [
                            self._risk_free_context(entry["definition"])[
                                "risk_free_rate_per_observation"
                            ]
                            for entry, _ in period_entries
                        ],
                        dtype=np.float64,
                    )
                    output_owner = own(
                        np.full(
                            (len(product_ids), len(period_entries)),
                            np.nan,
                            dtype=np.float64,
                        )
                    )
                    status_owner = own(
                        np.full(
                            (len(product_ids), len(period_entries)),
                            STATUS_INSUFFICIENT_SAMPLE,
                            dtype=np.int16,
                        )
                    )
                    task_specs.append(
                        {
                            "values": values_owner,
                            "starts": own(np.ascontiguousarray(starts)),
                            "ends": own(np.ascontiguousarray(ends)),
                            "codes": own(codes),
                            "primary": own(np.ascontiguousarray(primary)),
                            "secondary": own(np.ascontiguousarray(secondary)),
                            "risk_free": own(risk_free),
                            "output": output_owner,
                            "statuses": status_owner,
                            "entries": period_entries,
                            "observation_total": observation_total,
                        }
                    )
                    fast_metric_items.update(
                        int(entry["index"]) for entry, _ in period_entries
                    )

            if not task_specs:
                return {}, {
                    **empty_meta,
                    "shared_memory_bytes": shared_bytes,
                    "mmap_bytes": mmap_bytes,
                }

            use_inner_parallel = len(task_specs) == 1 and bool(
                len(product_ids) >= 32
                and task_specs[0]["observation_total"]
                >= self.compute_engine.prange_min_elements
                and thread_budget > 1
            )

            def submit_all() -> list[Any]:
                return [
                    self.compute_engine.submit_builtin_shared(
                        values=task["values"].descriptor,
                        starts=task["starts"].descriptor,
                        ends=task["ends"].descriptor,
                        codes=task["codes"].descriptor,
                        primary_indices=task["primary"].descriptor,
                        secondary_indices=task["secondary"].descriptor,
                        risk_free=task["risk_free"].descriptor,
                        output=task["output"].descriptor,
                        statuses=task["statuses"].descriptor,
                        parallel=use_inner_parallel,
                        thread_budget=(thread_budget if use_inner_parallel else 1),
                    )
                    for task in task_specs
                ]

            futures = submit_all()
            try:
                outcomes = [self.compute_engine.wait(future) for future in futures]
            except Exception:
                self.compute_engine.restart()
                outcomes = [
                    self.compute_engine.wait(future) for future in submit_all()
                ]
            worker_pids = {int(outcome["worker_pid"]) for outcome in outcomes}
            for task in task_specs:
                output = np.asarray(task["output"].view())
                statuses = np.asarray(task["statuses"].view())
                for metric_position, (entry, _) in enumerate(task["entries"]):
                    output_index = int(entry["index"])
                    dependencies = entry["data_dependencies"]
                    period = str(entry["item"]["period"])
                    for row_index, product_id in enumerate(product_ids):
                        window, error = selected_windows[
                            (dependencies, period, product_id)
                        ]
                        if window is None or error is not None:
                            continue
                        precomputed[(output_index, row_index)] = (
                            float(output[row_index, metric_position]),
                            int(statuses[row_index, metric_position]),
                        )

        return precomputed, {
            "metric_items": len(fast_metric_items),
            "worker_pids": sorted(worker_pids),
            "shared_memory_bytes": shared_bytes,
            "mmap_bytes": mmap_bytes,
            "parallel_tasks": int(use_inner_parallel),
        }

    def _run_fused_typed_groups(
        self,
        *,
        prepared: list[dict[str, Any]],
        product_ids: list[str],
        series_by_dependency: dict[
            tuple[str, ...], dict[str, ProductVariableSeries]
        ],
        selected_windows: dict[
            tuple[tuple[str, ...], str, str],
            tuple[VariablePeriodWindow | None, ValidationError | None],
        ],
        thread_budget: int,
    ) -> tuple[
        dict[tuple[int, int], tuple[float, int]],
        dict[str, Any],
    ]:
        """Run every typed metric through one compiled NJIT batch per data window.

        Python is limited to the Arrow/NumPy packing boundary and response
        assembly.  The product, metric and DAG-node loops are emitted by the
        trusted compiler and executed inside Numba.
        """

        grouped: dict[
            tuple[tuple[str, ...], str], list[dict[str, Any]]
        ] = {}
        for entry in prepared:
            if not entry["typed"]:
                continue
            grouped.setdefault(
                (entry["data_dependencies"], str(entry["item"]["period"])), []
            ).append(entry)

        precomputed: dict[tuple[int, int], tuple[float, int]] = {}
        compiled_plan_ids: list[str] = []
        compile_ms = 0.0
        parallel_tasks = 0
        metric_items: set[int] = set()
        previous_threads = numba.get_num_threads()
        try:
            for (dependencies, period), entries in grouped.items():
                sources = series_by_dependency[dependencies]
                physical_columns = tuple(
                    dict.fromkeys(
                        [
                            "adjusted_nav",
                            *[
                                name
                                for name in dependencies
                                if name
                                not in {"returns", "log_returns", "adjusted_nav"}
                            ],
                        ]
                    )
                )
                offsets = np.zeros(len(product_ids) + 1, dtype=np.int64)
                for row_index, product_id in enumerate(product_ids):
                    source = sources.get(product_id)
                    offsets[row_index + 1] = offsets[row_index] + (
                        len(source.frame) if source is not None else 0
                    )
                total_points = int(offsets[-1])
                if total_points == 0:
                    continue

                packed_values = np.full(
                    (len(physical_columns), total_points),
                    np.nan,
                    dtype=np.float64,
                )
                for row_index, product_id in enumerate(product_ids):
                    source = sources.get(product_id)
                    start = int(offsets[row_index])
                    end = int(offsets[row_index + 1])
                    if source is None or start == end:
                        continue
                    for value_index, column in enumerate(physical_columns):
                        if column in source.frame.columns:
                            packed_values[value_index, start:end] = source.frame[
                                column
                            ].to_numpy(dtype=np.float64, copy=False)

                starts = np.full(len(product_ids), -1, dtype=np.int64)
                ends = np.full(len(product_ids), -1, dtype=np.int64)
                elapsed_days = np.zeros(len(product_ids), dtype=np.float64)
                observation_total = 0
                for row_index, product_id in enumerate(product_ids):
                    window, error = selected_windows[
                        (dependencies, period, product_id)
                    ]
                    if window is None or error is not None:
                        continue
                    local_start = int(window.frame.index[0])
                    local_end = int(window.frame.index[-1]) + 1
                    starts[row_index] = int(offsets[row_index]) + local_start
                    ends[row_index] = int(offsets[row_index]) + local_end
                    elapsed_days[row_index] = float(
                        (
                            window.frame.iloc[-1]["date"]
                            - window.frame.iloc[0]["date"]
                        ).days
                    )
                    observation_total += max(0, local_end - local_start - 1)

                plans = tuple(entry["runtime"].plan for entry in entries)
                definitions = tuple(entry["definition"] for entry in entries)
                warmed_fused = get_cached_numba_batch_plan(
                    plans,
                    definitions,
                    physical_columns,
                )
                execution_groups: list[tuple[list[dict[str, Any]], Any]] = []
                if warmed_fused is not None:
                    execution_groups.append((entries, warmed_fused))
                else:
                    # Arbitrary preview selections have combinatorial batch
                    # shapes.  Reuse their startup-warmed singleton plans
                    # instead of compiling a new fused dispatcher in a request.
                    for entry in entries:
                        singleton = get_cached_numba_batch_plan(
                            (entry["runtime"].plan,),
                            (entry["definition"],),
                            physical_columns,
                        )
                        if singleton is None:
                            raise RuntimeError(
                                "NJIT batch plan was not prewarmed: "
                                f"{entry['definition'].get('id') or 'inline'}"
                            )
                        execution_groups.append(([entry], singleton))

                packed_contiguous = np.ascontiguousarray(packed_values)
                starts_contiguous = np.ascontiguousarray(starts)
                ends_contiguous = np.ascontiguousarray(ends)
                elapsed_contiguous = np.ascontiguousarray(elapsed_days)
                for execution_entries, batch_plan in execution_groups:
                    compiled_plan_ids.append(batch_plan.plan_id)
                    output = np.full(
                        (len(product_ids), len(execution_entries)),
                        np.nan,
                        dtype=np.float64,
                    )
                    statuses = np.full(
                        (len(product_ids), len(execution_entries)),
                        STATUS_INSUFFICIENT_SAMPLE,
                        dtype=np.int16,
                    )
                    use_parallel = bool(
                        len(product_ids) >= 32
                        and observation_total
                        >= self.compute_engine.prange_min_elements
                        and thread_budget > 1
                    )
                    if use_parallel:
                        numba.set_num_threads(
                            max(1, min(thread_budget, previous_threads))
                        )
                    try:
                        batch_plan.compute(
                            packed_contiguous,
                            starts_contiguous,
                            ends_contiguous,
                            elapsed_contiguous,
                            output,
                            statuses,
                            parallel=use_parallel,
                        )
                    except Exception:
                        # The retry remains inside the compiled NJIT lane.  It
                        # isolates row-level failures without Python operators.
                        output.fill(np.nan)
                        statuses.fill(STATUS_INSUFFICIENT_SAMPLE)
                        batch_plan.compute(
                            packed_contiguous,
                            starts_contiguous,
                            ends_contiguous,
                            elapsed_contiguous,
                            output,
                            statuses,
                            parallel=False,
                        )
                        use_parallel = False
                    parallel_tasks += int(use_parallel)
                    for metric_position, entry in enumerate(execution_entries):
                        output_index = int(entry["index"])
                        metric_items.add(output_index)
                        for row_index, product_id in enumerate(product_ids):
                            window, error = selected_windows[
                                (dependencies, period, product_id)
                            ]
                            if window is None or error is not None:
                                continue
                            precomputed[(output_index, row_index)] = (
                                float(output[row_index, metric_position]),
                                int(statuses[row_index, metric_position]),
                            )
        finally:
            if numba.get_num_threads() != previous_threads:
                numba.set_num_threads(previous_threads)

        return precomputed, {
            "metric_items": len(metric_items),
            "worker_pids": [os.getpid()] if metric_items else [],
            "shared_memory_bytes": 0,
            "mmap_bytes": 0,
            "parallel_tasks": parallel_tasks,
            "compiled_plan_ids": compiled_plan_ids,
            "compile_ms": round(compile_ms, 3),
            "python_fallback": 0,
            "python_operator_calls": 0,
        }

    def _run_plan_batch(
        self,
        plan_id: str,
        as_of: Optional[str] = None,
        *,
        thread_budget: int = 1,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        plan = self.plans.get(plan_id)
        generation = market_data_generation(self.market_data_dir)
        cache_key = self._plan_run_cache_key(plan, as_of, generation)
        if len(plan["targets"]) <= 2_000:
            cached = self.plan_cache.get(cache_key)
            if cached is not None:
                execution = cached.setdefault("execution", {})
                combinations = int(execution.get("combinations", 0))
                execution["cache"] = {
                    "plan_hits": 1,
                    "plan_misses": 0,
                    "data_hits": 0,
                    "data_misses": 0,
                    "window_hits": 0,
                    "window_misses": 0,
                    "cell_hits": combinations,
                    "cell_misses": 0,
                }
                cache_lookup_ms = round(
                    (time.perf_counter() - started) * 1000.0,
                    3,
                )
                execution["timings_ms"] = {
                    "planning": 0.0,
                    "data_load": 0.0,
                    "normalization": 0.0,
                    "windowing": 0.0,
                    "compute": 0.0,
                    "scoring": 0.0,
                    "result_assembly": 0.0,
                    "cache_lookup": cache_lookup_ms,
                    "total": cache_lookup_ms,
                }
                return cached

        planning_started = time.perf_counter()
        target_keys = [
            (target["kind"], target["product_id"]) for target in plan["targets"]
        ]
        target_index = {key: index for index, key in enumerate(target_keys)}
        values_by_target: list[list[dict[str, Any] | None]] = [
            [None] * len(plan["indicators"]) for _ in target_keys
        ]
        target_names = [key[1] for key in target_keys]
        prepared: list[dict[str, Any]] = []
        typed_dependencies: set[tuple[str, ...]] = set()
        for index, item in enumerate(plan["indicators"]):
            definition = self.indicators.get(
                item["indicator_id"], int(item["indicator_revision"])
            )
            runtime = self._compile_runtime(definition, item["period"])
            if not isinstance(runtime, TypedIndicatorRuntime):
                raise ValidationError(
                    "PLAN_REQUIRES_NJIT_INDICATOR",
                    f"评价方案中的指标 {definition['name']} 没有 NJIT 计算计划。",
                    field="indicators",
                )
            dependencies = (
                canonicalize_variables(runtime.plan.context_requirements)
                if isinstance(runtime, TypedIndicatorRuntime)
                else ()
            )
            data_dependencies = self._physical_dependency_signature(dependencies)
            if isinstance(runtime, TypedIndicatorRuntime):
                typed_dependencies.add(data_dependencies)
            prepared.append(
                {
                    "index": index,
                    "item": item,
                    "definition": definition,
                    "runtime": runtime,
                    "dependencies": dependencies,
                    "data_dependencies": data_dependencies,
                    "typed": isinstance(runtime, TypedIndicatorRuntime),
                }
            )
        snapshot_item_count = 0
        snapshot_cell_hits = 0
        if as_of is None:
            configured_keys = {
                (
                    str(item.get("indicator_id")),
                    int(item.get("indicator_revision") or 0),
                    str(item.get("period") or "").upper(),
                )
                for item in self.snapshot_config.get().get("items", [])
            }
            snapshot_groups: dict[str, list[dict[str, Any]]] = {}
            for entry in prepared:
                item = entry["item"]
                key = (
                    str(item["indicator_id"]),
                    int(item["indicator_revision"]),
                    str(item["period"]).upper(),
                )
                if key in configured_keys:
                    snapshot_groups.setdefault(key[2], []).append(entry)
            snapshot_indexes: set[int] = set()
            for snapshot_period, snapshot_entries in snapshot_groups.items():
                snapshot_response = self._evaluate_from_snapshot(
                    [entry["definition"] for entry in snapshot_entries],
                    plan["targets"],
                    snapshot_period,
                )
                if snapshot_response is None:
                    continue
                entries_by_definition = {
                    (
                        str(entry["item"]["indicator_id"]),
                        int(entry["item"]["indicator_revision"]),
                    ): entry
                    for entry in snapshot_entries
                }
                for snapshot_result in snapshot_response["results"]:
                    entry = entries_by_definition[
                        (
                            str(snapshot_result["indicator_id"]),
                            int(snapshot_result["indicator_revision"]),
                        )
                    ]
                    target = snapshot_result["target"]
                    row_index = target_index[
                        (str(target["kind"]), str(target["product_id"]))
                    ]
                    output_index = int(entry["index"])
                    values_by_target[row_index][output_index] = self._plan_value_payload(
                        entry["item"], entry["definition"], snapshot_result
                    )
                    target_names[row_index] = str(target.get("name") or target["product_id"])
                    snapshot_cell_hits += 1
                    snapshot_indexes.add(output_index)
                snapshot_item_count += len(snapshot_entries)
            prepared = [
                entry for entry in prepared if int(entry["index"]) not in snapshot_indexes
            ]
            typed_dependencies = {
                entry["data_dependencies"] for entry in prepared if entry["typed"]
            }
        planning_ms = (time.perf_counter() - planning_started) * 1000.0

        product_kind = str(plan.get("product_kind") or plan["targets"][0]["kind"])
        product_ids = [target["product_id"] for target in plan["targets"]]
        data_started = time.perf_counter()
        series_by_dependency: dict[
            tuple[str, ...], dict[str, ProductVariableSeries]
        ] = {}
        data_keys: dict[tuple[str, ...], str] = {}
        data_cache_hits = 0
        data_cache_misses = 0
        for dependencies in sorted(typed_dependencies):
            data_key = self._batch_data_cache_key(
                generation=generation,
                product_kind=product_kind,
                product_ids=product_ids,
                dependencies=dependencies,
                as_of=as_of,
            )
            data_keys[dependencies] = data_key
            cached_sources = self.data_cache.get(data_key)
            if cached_sources is not None:
                series_by_dependency[dependencies] = cached_sources
                data_cache_hits += 1
                continue
            sources = load_product_variable_series_batch(
                product_kind,  # type: ignore[arg-type]
                product_ids,
                dependencies,
                self.market_data_dir,
                as_of,
            )
            series_by_dependency[dependencies] = sources
            self.data_cache.put(
                data_key, sources, self._batch_series_size(sources)
            )
            data_cache_misses += 1
        data_load_ms = (time.perf_counter() - data_started) * 1000.0

        window_started = time.perf_counter()
        window_indices = {
            (dependencies, product_id): prepare_variable_window_index(source)
            for dependencies, sources in series_by_dependency.items()
            for product_id, source in sources.items()
        }
        selected_windows: dict[
            tuple[tuple[str, ...], str, str],
            tuple[VariablePeriodWindow | None, ValidationError | None],
        ] = {}
        dependency_periods = {
            (entry["data_dependencies"], str(entry["item"]["period"]))
            for entry in prepared
            if entry["typed"]
        }
        window_cache_hits = 0
        window_cache_misses = 0
        for dependencies, period in sorted(dependency_periods):
            window_cache_key = hashlib.sha256(
                repr(
                    (
                        data_keys[dependencies],
                        period,
                        as_of,
                        period_cache_reference(as_of),
                        MAX_SERIES_OBSERVATIONS,
                    )
                ).encode("utf-8")
            ).hexdigest()
            cached_windows = self.window_cache.get(window_cache_key)
            if cached_windows is not None:
                window_cache_hits += 1
                for product_id, outcome in cached_windows.items():
                    selected_windows[(dependencies, period, product_id)] = outcome
                continue
            period_windows: dict[
                str, tuple[VariablePeriodWindow | None, ValidationError | None]
            ] = {}
            for product_id in product_ids:
                source = series_by_dependency[dependencies].get(product_id)
                try:
                    if source is None:
                        raise ValidationError(
                            "DATA_NOT_FOUND", "未找到该产品的真实净值数据。"
                        )
                    period_windows[product_id] = (
                        select_variable_window_fast(
                            source,
                            period,
                            as_of,
                            max_observations=MAX_SERIES_OBSERVATIONS,
                            index=window_indices[(dependencies, product_id)],
                        ),
                        None,
                    )
                except ValidationError as exc:
                    period_windows[product_id] = (None, exc)
            self.window_cache.put(
                window_cache_key,
                period_windows,
                self._window_map_size(period_windows),
            )
            window_cache_misses += 1
            for product_id, outcome in period_windows.items():
                selected_windows[(dependencies, period, product_id)] = outcome
        window_ms = (time.perf_counter() - window_started) * 1000.0

        compute_started = time.perf_counter()
        precomputed, fused_meta = self._run_fused_typed_groups(
            prepared=prepared,
            product_ids=product_ids,
            series_by_dependency=series_by_dependency,
            selected_windows=selected_windows,
            thread_budget=thread_budget,
        )
        cell_hits = snapshot_cell_hits
        cell_misses = 0
        typed_item_count = 0
        compatibility_item_count = 0
        for entry in prepared:
            item = entry["item"]
            definition = entry["definition"]
            output_index = int(entry["index"])
            if entry["typed"]:
                typed_item_count += 1
                dependencies = entry["data_dependencies"]
                sources = series_by_dependency[dependencies]
                for target in plan["targets"]:
                    product_id = target["product_id"]
                    source = sources.get(product_id)
                    window, window_error = selected_windows[
                        (dependencies, item["period"], product_id)
                    ]
                    row_index = target_index[(target["kind"], product_id)]
                    fused = precomputed.get((output_index, row_index))
                    if (
                        fused is not None
                        and fused[1] == STATUS_OK
                        and source is not None
                        and window is not None
                        and window_error is None
                    ):
                        result = self._precomputed_result(
                            definition,
                            target,
                            source,
                            item["period"],
                            window,
                            fused[0],
                            fused[1],
                        )
                    else:
                        if (
                            fused is None
                            and source is not None
                            and window is not None
                            and window_error is None
                        ):
                            raise ValidationError(
                                "NJIT_BATCH_RESULT_MISSING",
                                "融合 NJIT 计算未返回完整结果，已拒绝兼容回退。",
                            )
                        result = self._evaluate_one(
                            definition,
                            entry["runtime"],
                            target,
                            source,
                            item["period"],
                            as_of,
                            False,
                            preselected_window=(
                                window if window is not None else _WINDOW_NOT_SELECTED
                            ),
                            preselection_error=window_error,
                        )
                    target_names[row_index] = result["target"]["name"]
                    values_by_target[row_index][output_index] = self._plan_value_payload(
                        item, definition, result
                    )
                    cell_misses += 1
                continue

            compatibility_item_count += 1
            for start_index in range(0, len(plan["targets"]), MAX_TARGETS):
                response = self.evaluate(
                    indicator_ids=[item["indicator_id"]],
                    inline_definition=None,
                    targets=plan["targets"][start_index : start_index + MAX_TARGETS],
                    period=item["period"],
                    as_of=as_of,
                    include_series=False,
                    indicator_versions={
                        item["indicator_id"]: int(item["indicator_revision"])
                    },
                )
                cell_hits += int(response.get("cache", {}).get("hits", 0))
                cell_misses += int(response.get("cache", {}).get("misses", 0))
                for result in response["results"]:
                    key = (result["target"]["kind"], result["target"]["product_id"])
                    row_index = target_index[key]
                    target_names[row_index] = result["target"]["name"]
                    values_by_target[row_index][output_index] = self._plan_value_payload(
                        item, definition, result
                    )
        compute_ms = (time.perf_counter() - compute_started) * 1000.0

        score_started = time.perf_counter()
        product_count = len(target_keys)
        metric_count = len(plan["indicators"])
        raw_matrix = np.full((product_count, metric_count), np.nan, dtype=np.float64)
        concrete_values: list[list[dict[str, Any]]] = []
        for row_index, value_slots in enumerate(values_by_target):
            row_values: list[dict[str, Any]] = []
            for metric_index, value in enumerate(value_slots):
                if value is None:
                    raise RuntimeError("评价方案批量执行未填充完整的指标结果。")
                row_values.append(value)
                if value["value"] is not None:
                    raw_matrix[row_index, metric_index] = float(value["value"])
            concrete_values.append(row_values)

        complete_mask = np.all(np.isfinite(raw_matrix), axis=1)
        weights = np.asarray(
            [float(item["weight"]) for item in plan["indicators"]],
            dtype=np.float64,
        )
        total_weight = float(np.sum(weights))
        effective_weights = weights / total_weight
        normalized = np.full_like(raw_matrix, np.nan)
        contributions = np.full_like(raw_matrix, np.nan)
        if np.any(complete_mask):
            complete_values = raw_matrix[complete_mask]
            lower = np.min(complete_values, axis=0)
            upper = np.max(complete_values, axis=0)
            spans = upper - lower
            lower_better = np.asarray(
                [item["direction"] == "lower_better" for item in plan["indicators"]],
                dtype=np.int8,
            )
            normalized, parallel_scoring = score_matrix(
                raw_matrix,
                effective_weights,
                lower_better,
                complete_mask,
                lower,
                upper,
                min_parallel_elements=self.compute_engine.prange_min_elements,
                thread_budget=thread_budget,
            )
            contributions[complete_mask] = (
                normalized[complete_mask] * effective_weights
            )
        else:
            parallel_scoring = False
        scores = np.full(product_count, np.nan, dtype=np.float64)
        scores[complete_mask] = np.sum(contributions[complete_mask], axis=1)
        ranked_indices = np.flatnonzero(complete_mask)
        if ranked_indices.size:
            order = np.argsort(-scores[ranked_indices], kind="stable")
            ranked_indices = ranked_indices[order]
        ranks = np.zeros(product_count, dtype=np.int64)
        ranks[ranked_indices] = np.arange(1, ranked_indices.size + 1, dtype=np.int64)
        for row_index, values in enumerate(concrete_values):
            for metric_index, value in enumerate(values):
                value["effective_weight"] = float(effective_weights[metric_index])
                if complete_mask[row_index]:
                    value["normalized_score"] = float(
                        normalized[row_index, metric_index]
                    )
                    value["weighted_contribution"] = float(
                        contributions[row_index, metric_index]
                    )
        scoring_ms = (time.perf_counter() - score_started) * 1000.0

        assembly_started = time.perf_counter()
        rows: list[dict[str, Any]] = []
        for row_index, target in enumerate(plan["targets"]):
            values = concrete_values[row_index]
            complete = bool(complete_mask[row_index])
            rows.append(
                {
                    "rank": int(ranks[row_index]) if complete else None,
                    "target": {**target, "name": target_names[row_index]},
                    "score": round(float(scores[row_index]), 6) if complete else None,
                    "status": "ranked" if complete else "excluded",
                    "missing_indicators": [
                        value["indicator_name"]
                        for value in values
                        if value["value"] is None
                    ],
                    "exclusion_reasons": [
                        warning
                        for value in values
                        if value["value"] is None
                        for warning in value.get("warnings", [])
                    ],
                    "values": values,
                }
            )
        rows.sort(key=lambda row: (row["rank"] is None, row["rank"] or math.inf))
        assembly_ms = (time.perf_counter() - assembly_started) * 1000.0
        total_ms = (time.perf_counter() - started) * 1000.0
        result = {
            "plan_id": plan_id,
            "plan_revision": int(plan["revision"]),
            "run_at": datetime.now(timezone.utc).isoformat(),
            "as_of": as_of,
            "rows": rows,
            "ranked_count": int(ranked_indices.size),
            "excluded_count": int(product_count - ranked_indices.size),
            "normalization": {
                "method": "min_max_0_100",
                "configured_weight_total": total_weight,
                "effective_weight_total": 1.0,
                "missing_policy": "strict",
            },
            "execution": {
                "engine_version": ENGINE_VERSION,
                "numeric_kernel_version": NUMERIC_KERNEL_VERSION,
                "operator_coverage": kernel_registry_status()[
                    "operator_coverage"
                ],
                "data_generation": generation,
                "execution_lanes": {
                    "numba_fused": fused_meta["metric_items"],
                    "numba_blas": 0,
                    **({"snapshot": snapshot_item_count} if snapshot_item_count else {}),
                    "python_fallback": 0,
                },
                "typed_batch_fallback": 0,
                "python_operator_calls": 0,
                "compiled_plan_ids": fused_meta.get("compiled_plan_ids", []),
                "compile_cache_hits": len(
                    fused_meta.get("compiled_plan_ids", [])
                ),
                "compile_cache_misses": 0,
                "kernel_cache_hits": fused_meta["metric_items"],
                "kernel_cache_misses": 0,
                "worker_processes": len(fused_meta["worker_pids"]),
                "worker_pids": fused_meta["worker_pids"],
                "numba_threads": (
                    thread_budget
                    if parallel_scoring or fused_meta["parallel_tasks"]
                    else 1
                ),
                "shared_memory_bytes": fused_meta["shared_memory_bytes"],
                "mmap_bytes": fused_meta["mmap_bytes"],
                "parallel_scoring": parallel_scoring,
                "combinations": product_count * metric_count,
                "cache": {
                    "plan_hits": 0,
                    "plan_misses": 1,
                    "data_hits": data_cache_hits,
                    "data_misses": data_cache_misses,
                    "window_hits": window_cache_hits,
                    "window_misses": window_cache_misses,
                    "cell_hits": cell_hits,
                    "cell_misses": cell_misses,
                },
                "timings_ms": {
                    "planning": round(planning_ms, 3),
                    "data_load": round(data_load_ms, 3),
                    "normalization": 0.0,
                    "windowing": round(window_ms, 3),
                    "compute": round(compute_ms, 3),
                    "scoring": round(scoring_ms, 3),
                    "result_assembly": round(assembly_ms, 3),
                    "total": round(total_ms, 3),
                },
            },
        }
        inline_limit = max(
            1, int(os.getenv("INDICATOR_INLINE_RESULT_LIMIT", "2000"))
        )
        if product_count > inline_limit:
            store_started = time.perf_counter()
            result_id = self.run_results.store(result)
            page = self.run_results.page(result_id, page=1, page_size=100)
            page["execution"]["timings_ms"]["result_store"] = round(
                (time.perf_counter() - store_started) * 1000.0, 3
            )
            return page
        if product_count <= 2_000:
            self.plan_cache.put(cache_key, result)
        return result

    def get_plan_run_result(
        self,
        result_id: str,
        *,
        page: int = 1,
        page_size: int = 100,
    ) -> dict[str, Any]:
        return self.run_results.page(result_id, page=page, page_size=page_size)

    @staticmethod
    def _compare_shadow_results(
        fast: dict[str, Any], compatibility: dict[str, Any]
    ) -> dict[str, Any]:
        fast_rows = {
            (row["target"]["kind"], row["target"]["product_id"]): row
            for row in fast.get("rows", [])
        }
        compatibility_rows = {
            (row["target"]["kind"], row["target"]["product_id"]): row
            for row in compatibility.get("rows", [])
        }
        mismatches = 0
        max_absolute_error = 0.0
        if fast_rows.keys() != compatibility_rows.keys():
            mismatches += len(fast_rows.keys() ^ compatibility_rows.keys())
        for key in fast_rows.keys() & compatibility_rows.keys():
            fast_row = fast_rows[key]
            compatibility_row = compatibility_rows[key]
            if (
                fast_row.get("rank") != compatibility_row.get("rank")
                or fast_row.get("status") != compatibility_row.get("status")
            ):
                mismatches += 1
            fast_values = fast_row.get("values", [])
            compatibility_values = compatibility_row.get("values", [])
            if len(fast_values) != len(compatibility_values):
                mismatches += abs(len(fast_values) - len(compatibility_values))
            for fast_value, compatibility_value in zip(
                fast_values, compatibility_values, strict=False
            ):
                left = fast_value.get("value")
                right = compatibility_value.get("value")
                if left is None or right is None:
                    if left is not right:
                        mismatches += 1
                else:
                    error = abs(float(left) - float(right))
                    max_absolute_error = max(max_absolute_error, error)
                    if not math.isclose(
                        float(left), float(right), rel_tol=1e-10, abs_tol=1e-12
                    ):
                        mismatches += 1
                if fast_value.get("window") != compatibility_value.get("window"):
                    mismatches += 1
                if fast_value.get("status") != compatibility_value.get("status"):
                    mismatches += 1
        return {
            "compared": True,
            "equivalent": mismatches == 0,
            "mismatch_count": mismatches,
            "max_absolute_error": max_absolute_error,
            "row_count_equal": len(fast_rows) == len(compatibility_rows),
            "ranked_count_equal": fast.get("ranked_count")
            == compatibility.get("ranked_count"),
        }

    def run_plan(self, plan_id: str, as_of: Optional[str] = None) -> dict[str, Any]:
        plan = self.plans.get(plan_id)
        period_count = max(
            1, len({str(item.get("period")) for item in plan.get("indicators", [])})
        )
        tokens = min(period_count, self.compute_engine.worker_count)
        with self.compute_engine.admission(tokens) as reserved:
            return self._run_plan_batch(
                plan_id, as_of, thread_budget=reserved
            )

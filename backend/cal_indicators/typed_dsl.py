"""Typed indicator DSL v2 compiler, inference graph and fixed-signature NJIT runtime.

This module is intentionally parallel to ``indicator_runtime.py``.  Existing
saved indicators continue to use the legacy scalar-only v1 runtime, while new
callers can opt in to nominal axes, multi-asset matrices and typed DAG output.
"""

from __future__ import annotations

import ast
import hashlib
import math
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
from numba import float64, njit, uint8

from cal_indicators.typed_operators import (
    COMPAT_OPERATOR_REGISTRY_VERSION,
    COMPAT_TYPED_COMPILER_VERSION,
    COMPAT_TYPED_DSL_VERSION,
    LEGACY_OPERATOR_REGISTRY_VERSION,
    LEGACY_TYPED_COMPILER_VERSION,
    LEGACY_TYPED_DSL_VERSION,
    SUPPORTED_TYPED_DSL_VERSIONS,
    TYPED_COMPILER_VERSION,
    TYPED_DSL_VERSION,
    TYPED_OPERATOR_REGISTRY_VERSION,
    TypedOperatorSpec,
    get_typed_operator_catalog,
    get_typed_operator_registry,
)
from cal_indicators.typed_numba_plan import (
    CompiledNumbaPlan,
    NumbaPlanCompileError,
    compile_numba_plan,
    get_cached_numba_plan,
    numba_plan_id,
)
from cal_indicators.typed_types import (
    ASSET_VECTOR,
    SCALAR,
    TIME_ASSET_MATRIX,
    TIME_SERIES,
    SUPPORTED_SEMANTIC_DIMENSIONS,
    ValueType,
    TypedDslError,
    user_type_label,
)
from compute_policy import NJIT_BACKEND, validate_execution_audit


DEFAULT_MAX_NODES = 128
DEFAULT_MAX_DEPTH = 20
DEFAULT_MAX_TIME = 5_000
DEFAULT_MAX_ASSETS = 50
DEFAULT_MAX_RUNTIME_COST = 200_000_000
DEFAULT_MAX_LIVE_ELEMENTS = 8_000_000


_F1 = float64[::1]
_F2 = float64[:, ::1]
_U1 = uint8[::1]
_U2 = uint8[:, ::1]


@njit(uint8(float64), cache=False, nogil=True)
def _finite_scalar_kernel(value: float) -> int:
    return 1 if math.isfinite(value) else 0


@njit(uint8(_F1), cache=False, nogil=True)
def _finite_1d_kernel(values: np.ndarray) -> int:
    for value in values:
        if not math.isfinite(value):
            return 0
    return 1


@njit(uint8(_F2), cache=False, nogil=True)
def _finite_2d_kernel(values: np.ndarray) -> int:
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            if not math.isfinite(values[row, column]):
                return 0
    return 1


@njit(uint8(_U1), cache=False, nogil=True)
def _binary_mask_1d_kernel(values: np.ndarray) -> int:
    for value in values:
        if value > 1:
            return 0
    return 1


@njit(uint8(_U2), cache=False, nogil=True)
def _binary_mask_2d_kernel(values: np.ndarray) -> int:
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            if values[row, column] > 1:
                return 0
    return 1


@njit(uint8(_F1, float64), cache=False, nogil=True)
def _weight_vector_sum_kernel(values: np.ndarray, tolerance: float) -> int:
    total = 0.0
    for value in values:
        total += value
    return 1 if abs(total - 1.0) <= tolerance else 0


@njit(uint8(_F2, float64), cache=False, nogil=True)
def _weight_path_sum_kernel(values: np.ndarray, tolerance: float) -> int:
    for row in range(values.shape[0]):
        total = 0.0
        for column in range(values.shape[1]):
            total += values[row, column]
        if abs(total - 1.0) > tolerance:
            return 0
    return 1


def runtime_validation_kernel_signatures() -> dict[str, list[str]]:
    """Fixed signatures used by the Python input-contract boundary."""

    dispatchers = (
        _finite_scalar_kernel,
        _finite_1d_kernel,
        _finite_2d_kernel,
        _binary_mask_1d_kernel,
        _binary_mask_2d_kernel,
        _weight_vector_sum_kernel,
        _weight_path_sum_kernel,
    )
    return {
        dispatcher.py_func.__name__: [
            str(signature) for signature in dispatcher.signatures
        ]
        for dispatcher in dispatchers
    }


def runtime_validation_execution_audit() -> dict[str, Any]:
    dispatchers = (
        _finite_scalar_kernel,
        _finite_1d_kernel,
        _finite_2d_kernel,
        _binary_mask_1d_kernel,
        _binary_mask_2d_kernel,
        _weight_vector_sum_kernel,
        _weight_path_sum_kernel,
    )
    return validate_execution_audit(
        {
            "execution_backend": NJIT_BACKEND,
            "nopython": all(bool(dispatcher.nopython_signatures) for dispatcher in dispatchers),
            "kernel_signatures": runtime_validation_kernel_signatures(),
            "python_fallback": 0,
            "python_operator_calls": 0,
        }
    )


def _literal_number(node: ast.AST) -> float | None:
    """Return a finite numeric literal without evaluating arbitrary AST."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        value = float(node.value)
        return value if math.isfinite(value) else None
    return None


@dataclass(frozen=True)
class VariableSpec:
    name: str
    latex: str
    value_type: ValueType
    label: str
    description: str
    context_domains: tuple[str, ...]
    semantic_role: str
    source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "latex": self.latex,
            "label": self.label,
            "type": self.value_type.to_dict(),
            "shape": list(self.value_type.shape),
            "description": self.description,
            "context_domains": list(self.context_domains),
            "semantic_role": self.semantic_role,
            "source": self.source,
        }


_VARIABLE_SPECS = (
    VariableSpec(
        "returns",
        r"\mathbf{r}",
        ValueType.series(semantic_dimension="return_decimal"),
        "普通收益率",
        "当前单产品的普通收益率序列。",
        ("single_asset",),
        "return_series",
        "series_provider",
    ),
    VariableSpec(
        "log_returns",
        r"\mathbf{\ell}",
        ValueType.series(semantic_dimension="return_decimal"),
        "对数收益率",
        "当前单产品的对数收益率序列。",
        ("single_asset",),
        "log_return_series",
        "series_provider",
    ),
    VariableSpec(
        "asset_returns",
        r"\mathbf{R}",
        ValueType.matrix(semantic_dimension="return_decimal"),
        "多资产收益矩阵",
        "按请求资产顺序对齐的多资产普通收益率矩阵。",
        ("multi_asset", "portfolio"),
        "return_matrix",
        "panel_provider",
    ),
    VariableSpec(
        "asset_log_returns",
        r"\mathbf{L}",
        ValueType.matrix(semantic_dimension="return_decimal"),
        "多资产对数收益矩阵",
        "按请求资产顺序对齐的多资产对数收益率矩阵。",
        ("multi_asset", "portfolio"),
        "log_return_matrix",
        "panel_provider",
    ),
    VariableSpec(
        "portfolio_returns",
        r"\mathbf{r}_{\mathrm{portfolio}}",
        ValueType.series(semantic_dimension="return_decimal"),
        "组合实际收益率",
        "按每日生效权重与当日各底层产品收益逐日汇总的组合收益率序列。",
        ("portfolio",),
        "realized_portfolio_return_series",
        "portfolio_run_snapshot",
    ),
    VariableSpec(
        "asset_weights",
        r"\mathbf{w}",
        ValueType.vector(semantic_dimension="dimensionless"),
        "静态资产权重",
        "组合运行最后一个时点的资产权重，只适用于当前截面估算。",
        ("portfolio",),
        "asset_weight_vector",
        "request",
    ),
    VariableSpec(
        "weight_path",
        r"\mathbf{W}",
        ValueType.matrix(semantic_dimension="dimensionless"),
        "动态权重路径",
        "与收益矩阵逐时点、逐资产对齐且每行合计为 1 的权重矩阵。",
        ("portfolio",),
        "asset_weight_path",
        "request",
    ),
    VariableSpec(
        "benchmark_returns",
        r"\mathbf{b}",
        ValueType.series(semantic_dimension="return_decimal"),
        "基准收益率",
        "与研究收益率使用共同日期的基准收益序列。",
        ("single_asset", "multi_asset", "portfolio"),
        "benchmark_return_series",
        "series_provider",
    ),
    VariableSpec(
        "annual_risk_free_rate_decimal",
        r"r_{f}^{\mathrm{annual}}",
        ValueType.scalar(semantic_dimension="rate_decimal"),
        "年化无风险利率",
        "年化无风险利率，小数表示。",
        ("single_asset", "multi_asset", "portfolio"),
        "annual_risk_free_rate",
        "request_or_default",
    ),
    VariableSpec(
        "risk_free_rate_per_period",
        r"r_{f}",
        ValueType.scalar(semantic_dimension="rate_decimal"),
        "单期无风险收益",
        "当前观察频率的无风险收益率。",
        ("single_asset", "multi_asset", "portfolio"),
        "period_risk_free_rate",
        "derived",
    ),
    VariableSpec(
        "periods_per_year",
        r"p_{\mathrm{year}}",
        ValueType.scalar(semantic_dimension="count"),
        "年观察数",
        "每年的观察期数量，例如日频为 252。",
        ("single_asset", "multi_asset", "portfolio"),
        "annualization_factor",
        "calendar",
    ),
    VariableSpec(
        "adjusted_nav",
        r"\mathbf{n}_{\mathrm{adj}}",
        ValueType.series(
            "L",
            semantic_dimension="adjusted_nav",
            price_basis="adjusted_nav",
        ),
        "复权净值",
        "按当前产品复权口径计算的净值时间序列。",
        ("single_asset",),
        "adjusted_nav_series",
        "series_provider",
    ),
    VariableSpec(
        "unit_nav",
        r"\mathbf{n}_{\mathrm{unit}}",
        ValueType.series(semantic_dimension="reported_nav", price_basis="unit_nav"),
        "单位净值",
        "基金披露的单位净值时间序列。",
        ("single_asset",),
        "reported_unit_nav_series",
        "series_provider",
    ),
    VariableSpec(
        "accumulated_nav",
        r"\mathbf{n}_{\mathrm{acc}}",
        ValueType.series(
            semantic_dimension="reported_nav", price_basis="accumulated_nav"
        ),
        "累计净值",
        "基金披露的累计净值时间序列。",
        ("single_asset",),
        "reported_accumulated_nav_series",
        "series_provider",
    ),
    *(
        VariableSpec(
            name,
            latex,
            ValueType.series(
                semantic_dimension="raw_market_price", price_basis="raw_market"
            ),
            label,
            description,
            ("single_asset",),
            role,
            "series_provider",
        )
        for name, latex, label, description, role in (
            (
                "market_open",
                r"\mathbf{o}",
                "开盘价",
                "未复权市场开盘价。",
                "market_open_series",
            ),
            (
                "market_high",
                r"\mathbf{h}",
                "最高价",
                "未复权市场最高价。",
                "market_high_series",
            ),
            (
                "market_low",
                r"\mathbf{l}",
                "最低价",
                "未复权市场最低价。",
                "market_low_series",
            ),
            (
                "market_close",
                r"\mathbf{c}",
                "收盘价",
                "未复权市场收盘价。",
                "market_close_series",
            ),
            (
                "previous_close",
                r"\mathbf{c}_{\mathrm{prev}}",
                "前收盘价",
                "未复权市场前收盘价。",
                "previous_close_series",
            ),
            (
                "price_change",
                r"\Delta\mathbf{c}",
                "价格变动",
                "未复权市场价格绝对变动。",
                "price_change_series",
            ),
        )
    ),
    VariableSpec(
        "price_return",
        r"\mathbf{r}_{\mathrm{price}}",
        ValueType.series(semantic_dimension="return_decimal"),
        "价格收益率",
        "由未复权市场收盘价计算的普通收益率。",
        ("single_asset",),
        "price_return_series",
        "series_provider",
    ),
    VariableSpec(
        "volume",
        r"\mathbf{v}",
        ValueType.series(semantic_dimension="volume"),
        "成交量",
        "当前产品逐期成交数量。",
        ("single_asset",),
        "trading_volume_series",
        "series_provider",
    ),
    VariableSpec(
        "turnover_amount",
        r"\mathbf{a}",
        ValueType.series(semantic_dimension="currency_amount"),
        "成交额",
        "当前产品逐期成交金额。",
        ("single_asset",),
        "turnover_amount_series",
        "series_provider",
    ),
)

_CANONICAL_VARIABLE_TYPES = {item.name: item.value_type for item in _VARIABLE_SPECS}
_LEGACY_PUBLIC_VARIABLE_NAMES = frozenset(
    {
        "returns",
        "log_returns",
        "asset_returns",
        "asset_log_returns",
        "asset_weights",
        "weight_path",
        "benchmark_returns",
        "annual_risk_free_rate_decimal",
        "risk_free_rate_per_period",
        "periods_per_year",
    }
)
_LEGACY_VARIABLE_TYPE_ALIASES = {
    "open_price": _CANONICAL_VARIABLE_TYPES["market_open"],
    "high_price": _CANONICAL_VARIABLE_TYPES["market_high"],
    "low_price": _CANONICAL_VARIABLE_TYPES["market_low"],
    "close_price": _CANONICAL_VARIABLE_TYPES["market_close"],
}
DEFAULT_VARIABLE_TYPES: Mapping[str, ValueType] = MappingProxyType(
    {**_CANONICAL_VARIABLE_TYPES, **_LEGACY_VARIABLE_TYPE_ALIASES}
)


def get_typed_variable_catalog(
    version: str = TYPED_OPERATOR_REGISTRY_VERSION,
) -> dict[str, Any]:
    legacy = version == LEGACY_OPERATOR_REGISTRY_VERSION
    return {
        "context_schema_version": "multi-asset-v1" if legacy else "typed-market-v2.1",
        "variables": [
            item.to_dict()
            for item in _VARIABLE_SPECS
            if not legacy or item.name in _LEGACY_PUBLIC_VARIABLE_NAMES
        ],
    }


def get_typed_dsl_catalog(
    version: str = TYPED_OPERATOR_REGISTRY_VERSION,
) -> dict[str, Any]:
    operator_catalog = get_typed_operator_catalog(version)
    legacy = version == LEGACY_OPERATOR_REGISTRY_VERSION
    numeric_types = [
        SCALAR.to_dict(),
        TIME_SERIES.to_dict(),
        ASSET_VECTOR.to_dict(),
        TIME_ASSET_MATRIX.to_dict(),
        ValueType.matrix(("asset", "asset"), ("N", "N")).to_dict(),
    ]
    return {
        **operator_catalog,
        **get_typed_variable_catalog(version),
        "type_system": {
            "dtype": ["float64"] if legacy else ["float64", "bool"],
            "semantic_dimensions": (
                ["dimensionless", "rate_decimal", "return_decimal"]
                if legacy
                else sorted(SUPPORTED_SEMANTIC_DIMENSIONS)
            ),
            "types": numeric_types
            if legacy
            else [
                *numeric_types,
                TIME_SERIES.as_mask().to_dict(),
                ASSET_VECTOR.as_mask().to_dict(),
                TIME_ASSET_MATRIX.as_mask().to_dict(),
            ],
            "broadcasting": "scalar_only",
            "multiplication": "elementwise; use matmul/matvec/dot for contraction",
        },
        "limits": {
            "max_nodes": DEFAULT_MAX_NODES,
            "max_depth": DEFAULT_MAX_DEPTH,
            "max_time_observations": DEFAULT_MAX_TIME,
            "max_assets": DEFAULT_MAX_ASSETS,
            "max_runtime_cost": DEFAULT_MAX_RUNTIME_COST,
            "max_live_elements": DEFAULT_MAX_LIVE_ELEMENTS,
        },
    }


def _extract_braced(expression: str, start: int) -> tuple[str, int]:
    if start >= len(expression) or expression[start] != "{":
        raise TypedDslError("LATEX_PARSE_ERROR", f"位置 {start} 处缺少 '{{'。")
    depth = 0
    for index in range(start, len(expression)):
        char = expression[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return expression[start + 1 : index], index + 1
    raise TypedDslError("LATEX_PARSE_ERROR", f"位置 {start} 处的 '{{' 没有闭合。")


def _replace_latex_fraction(expression: str) -> str:
    token = r"\frac"
    while token in expression:
        index = expression.index(token)
        numerator, after_numerator = _extract_braced(expression, index + len(token))
        denominator, after_denominator = _extract_braced(expression, after_numerator)
        replacement = f"({numerator})/({denominator})"
        expression = expression[:index] + replacement + expression[after_denominator:]
    return expression


def _replace_latex_sqrt(expression: str) -> str:
    token = r"\sqrt"
    while token in expression:
        index = expression.index(token)
        value, after_value = _extract_braced(expression, index + len(token))
        expression = expression[:index] + f"sqrt({value})" + expression[after_value:]
    return expression


class TypedExpressionParser:
    """Translate the supported LaTeX surface to a restricted Python AST."""

    _operator_name = re.compile(r"\\operatorname\s*\{([A-Za-z][A-Za-z0-9_]*)\}")

    def __init__(self, variable_names: Sequence[str]) -> None:
        self.variable_names = frozenset(variable_names)

    def to_python(self, expression: str) -> str:
        converted = expression.strip()
        converted = converted.replace(r"\left", "").replace(r"\right", "")
        for spacing in (r"\,", r"\;", r"\!", r"\ "):
            converted = converted.replace(spacing, "")
        converted = _replace_latex_fraction(converted)
        converted = _replace_latex_sqrt(converted)
        converted = self._operator_name.sub(lambda match: match.group(1), converted)
        converted = converted.replace(r"\prod", "product")
        converted = converted.replace(r"\sum", "sum")
        converted = converted.replace(r"\times", "*").replace(r"\cdot", "*")

        special = (
            (r"\overline{\mathbf{\ell}}", "mean(log_returns)"),
            (r"\overline{\mathbf{r}}", "mean(returns)"),
            (r"r_{f}^{\mathrm{annual}}", "annual_risk_free_rate_decimal"),
            (r"r_{f}^{annual}", "annual_risk_free_rate_decimal"),
            (r"p_{\mathrm{year}}", "periods_per_year"),
            (r"p_{year}", "periods_per_year"),
            (r"\mathbf{\ell}", "log_returns"),
            (r"\mathbf{R}", "asset_returns"),
            (r"\mathbf{L}", "asset_log_returns"),
            (r"\mathbf{r}_{\mathrm{portfolio}}", "portfolio_returns"),
            (r"\mathbf{W}", "weight_path"),
            (r"\mathbf{w}", "asset_weights"),
            (r"\mathbf{b}", "benchmark_returns"),
            (r"\mathbf{n}_{\mathrm{adj}}", "adjusted_nav"),
            (r"\mathbf{n}_{adj}", "adjusted_nav"),
            (r"\mathbf{n}_{\mathrm{unit}}", "unit_nav"),
            (r"\mathbf{n}_{unit}", "unit_nav"),
            (r"\mathbf{n}_{\mathrm{acc}}", "accumulated_nav"),
            (r"\mathbf{n}_{acc}", "accumulated_nav"),
            (r"\mathbf{o}", "market_open"),
            (r"\mathbf{h}", "market_high"),
            (r"\mathbf{l}", "market_low"),
            (r"\Delta\mathbf{c}", "price_change"),
            (r"\mathbf{c}_{\mathrm{prev}}", "previous_close"),
            (r"\mathbf{c}_{prev}", "previous_close"),
            (r"\mathbf{c}", "market_close"),
            (r"\mathbf{r}_{\mathrm{price}}", "price_return"),
            (r"\mathbf{r}_{price}", "price_return"),
            (r"\mathbf{v}", "volume"),
            (r"\mathbf{a}", "turnover_amount"),
            (r"\mathbf{r}", "returns"),
            (r"r_{f}", "risk_free_rate_per_period"),
        )
        for latex, variable in special:
            converted = converted.replace(latex, variable)

        converted = converted.replace("^", "**")
        converted = converted.replace("{", "(").replace("}", ")")
        converted = converted.replace("\\", "")
        return converted

    def parse(self, expression: str) -> tuple[str, ast.AST]:
        python_expression = self.to_python(expression)
        try:
            parsed = ast.parse(python_expression, mode="eval")
        except SyntaxError as exc:
            raise TypedDslError(
                "LATEX_PARSE_ERROR",
                "公式无法解析。",
                details={"expression": expression},
            ) from exc
        return python_expression, parsed.body


@dataclass(frozen=True)
class TypedDagNode:
    node_id: int
    kind: str
    label: str
    inputs: tuple[int, ...]
    arguments: tuple[tuple[str, int], ...]
    inferred_type: ValueType
    operator_id: str | None
    operator_version: str | None
    cost_model: str
    cost_expression: str
    formula_fragment: str
    raw: str

    def to_dict(self) -> dict[str, Any]:
        operator = None
        if self.operator_id is not None:
            operator = {
                "id": self.operator_id,
                "version": self.operator_version,
            }
        return {
            "id": self.node_id,
            "kind": self.kind,
            "label": self.label,
            "inputs": list(self.inputs),
            "arguments": [
                {"name": name, "input_node_id": input_node_id}
                for name, input_node_id in self.arguments
            ],
            "inferred_type": self.inferred_type.to_dict(),
            "operator": operator,
            "formula_fragment": self.formula_fragment,
            "cost": {
                "model": self.cost_model,
                "expression": self.cost_expression,
            },
        }


@dataclass(frozen=True)
class TypedExpressionPlan:
    expression: str
    python_expression: str
    expression_hash: str
    dsl_version: str
    compiler_version: str
    operator_registry_version: str
    nodes: tuple[TypedDagNode, ...]
    root_id: int
    output_type: ValueType
    output_contract: str
    context_requirements: Mapping[str, ValueType]
    estimated_cost: Mapping[str, Any]

    def graph_payload(self) -> dict[str, Any]:
        return {
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [
                {"source": input_id, "target": node.node_id}
                for node in self.nodes
                for input_id in node.inputs
            ],
            "roots": {"result": self.root_id},
            "output_type": self.output_type.to_dict(),
            "context_requirements": {
                name: value_type.to_dict()
                for name, value_type in self.context_requirements.items()
            },
            "estimated_cost": dict(self.estimated_cost),
            "compiler_version": self.compiler_version,
            "operator_registry_version": self.operator_registry_version,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "expression": self.expression,
            "python_expression": self.python_expression,
            "expression_hash": self.expression_hash,
            "dsl_version": self.dsl_version,
            "compiler_version": self.compiler_version,
            "operator_registry_version": self.operator_registry_version,
            "output_contract": self.output_contract,
            **self.graph_payload(),
        }


def _normalize_variable_types(
    variable_types: Mapping[str, ValueType | Mapping[str, Any]] | None,
) -> dict[str, ValueType]:
    normalized = dict(DEFAULT_VARIABLE_TYPES)
    if variable_types is None:
        return normalized
    for name, value_type in variable_types.items():
        if not name.isidentifier():
            raise TypedDslError("INVALID_VARIABLE", f"变量名不是合法标识符: {name}")
        if isinstance(value_type, ValueType):
            normalized[name] = value_type
        elif isinstance(value_type, Mapping):
            normalized[name] = ValueType.from_dict(value_type)
        else:
            raise TypedDslError(
                "INVALID_VARIABLE_TYPE", f"变量 {name} 的类型定义无效。"
            )
    return normalized


class _TypedDagBuilder:
    _binary_operators: Mapping[type[ast.operator], str] = {
        ast.Add: "add",
        ast.Sub: "subtract",
        ast.Mult: "multiply",
        ast.Div: "divide",
        ast.Pow: "power",
    }

    def __init__(
        self,
        variable_types: Mapping[str, ValueType],
        registry: Mapping[str, TypedOperatorSpec],
        *,
        max_nodes: int,
        max_depth: int,
    ) -> None:
        self.variable_types = variable_types
        self.registry = registry
        self.max_nodes = max_nodes
        self.max_depth = max_depth
        self.nodes: list[TypedDagNode] = []
        self.cache: dict[str, int] = {}

    def build(self, root: ast.AST) -> int:
        return self._build(root, depth=1)

    def _infer_operator(
        self,
        spec: TypedOperatorSpec,
        input_types: tuple[ValueType, ...],
    ) -> ValueType:
        try:
            return spec.infer_output(input_types)
        except TypedDslError as exc:
            if exc.node_id is None:
                # The failing operator is the next prospective DAG node. Its
                # children have already been appended and remain inspectable.
                exc.node_id = len(self.nodes)
            exc.details.setdefault(
                "expected",
                [signature.to_dict() for signature in spec.signatures],
            )
            exc.details.setdefault(
                "actual",
                [value_type.to_dict() for value_type in input_types],
            )
            exc.details.setdefault("operator", spec.operator_id)
            raise

    def _build(self, node: ast.AST, *, depth: int) -> int:
        if depth > self.max_depth:
            raise TypedDslError(
                "FORMULA_TOO_COMPLEX",
                f"表达式深度不能超过 {self.max_depth}。",
                details={"limit": self.max_depth, "dimension": "depth"},
            )
        cache_key = ast.dump(node, include_attributes=False)
        if cache_key in self.cache:
            return self.cache[cache_key]

        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise TypedDslError("INVALID_LITERAL", "仅允许有限数值常量。")
            try:
                value = float(node.value)
            except (OverflowError, TypeError, ValueError) as exc:
                raise TypedDslError("INVALID_LITERAL", "仅允许有限数值常量。") from exc
            if not math.isfinite(value):
                raise TypedDslError("INVALID_LITERAL", "仅允许有限数值常量。")
            return self._append(
                kind="constant",
                label=repr(node.value),
                inputs=(),
                inferred_type=SCALAR,
                spec=None,
                cost_model="constant",
                cost_expression="0",
                formula_fragment=ast.unparse(node),
                raw=cache_key,
            )

        if isinstance(node, ast.Name):
            try:
                value_type = self.variable_types[node.id]
            except KeyError as exc:
                raise TypedDslError("UNKNOWN_VARIABLE", f"未知变量: {node.id}") from exc
            return self._append(
                kind="variable",
                label=node.id,
                inputs=(),
                inferred_type=value_type,
                spec=None,
                cost_model="context_read",
                cost_expression="0",
                formula_fragment=ast.unparse(node),
                raw=cache_key,
            )

        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            input_id = self._build(node.operand, depth=depth + 1)
            spec = self.registry["negate"]
            input_types = (self.nodes[input_id].inferred_type,)
            output_type = self._infer_operator(spec, input_types)
            return self._append_operator(
                "unary",
                spec,
                (input_id,),
                input_types,
                output_type,
                ast.unparse(node),
                cache_key,
            )

        if isinstance(node, ast.BinOp):
            operator_name = self._binary_operators.get(type(node.op))
            if operator_name is None:
                raise TypedDslError(
                    "UNKNOWN_OPERATOR", f"不支持的二元算子: {type(node.op).__name__}"
                )
            left_id = self._build(node.left, depth=depth + 1)
            right_id = self._build(node.right, depth=depth + 1)
            spec = self.registry[operator_name]
            input_types = (
                self.nodes[left_id].inferred_type,
                self.nodes[right_id].inferred_type,
            )
            output_type = self._infer_operator(spec, input_types)
            return self._append_operator(
                "binary",
                spec,
                (left_id, right_id),
                input_types,
                output_type,
                ast.unparse(node),
                cache_key,
            )

        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise TypedDslError("ILLEGAL_AST", "函数只能通过白名单名称直接调用。")
            if node.keywords:
                raise TypedDslError("ILLEGAL_AST", "函数调用不允许关键字参数。")
            try:
                spec = self.registry[node.func.id]
            except KeyError as exc:
                raise TypedDslError(
                    "UNKNOWN_OPERATOR", f"未知函数或算子: {node.func.id}"
                ) from exc
            input_ids = tuple(
                self._build(argument, depth=depth + 1) for argument in node.args
            )
            input_types = tuple(
                self.nodes[input_id].inferred_type for input_id in input_ids
            )
            output_type = self._infer_operator(spec, input_types)
            if spec.version == TYPED_OPERATOR_REGISTRY_VERSION:
                probability_index = {
                    "quantile": 1,
                    "quantile_where": 2,
                }.get(spec.operator_id)
                if probability_index is not None and len(node.args) > probability_index:
                    probability = _literal_number(node.args[probability_index])
                    if probability is None or not 0.0 < probability < 1.0:
                        raise TypedDslError(
                            "INVALID_PARAMETER",
                            f"{spec.operator_id} 的 probability 必须是 0 与 1 之间的有限常数。",
                            node_id=len(self.nodes),
                            details={
                                "operator": spec.operator_id,
                                "parameter": "probability",
                                "expected": "finite constant in (0, 1)",
                                "actual": ast.unparse(node.args[probability_index]),
                            },
                        )
                if spec.operator_id in {"lag", "difference"} and len(node.args) == 2:
                    periods = _literal_number(node.args[1])
                    minimum = 0 if spec.operator_id == "lag" else 1
                    if periods is None or not periods.is_integer() or periods < minimum:
                        comparator = "非负" if minimum == 0 else "正"
                        raise TypedDslError(
                            "INVALID_PARAMETER",
                            f"{spec.operator_id} 的 periods 必须是{comparator}整数常数。",
                            node_id=len(self.nodes),
                            details={
                                "operator": spec.operator_id,
                                "parameter": "periods",
                                "expected": f"{comparator} integer constant",
                                "actual": ast.unparse(node.args[1]),
                            },
                        )
                if spec.operator_id in {"variance", "std"} and len(node.args) == 2:
                    ddof = _literal_number(node.args[1])
                    if ddof is None or not ddof.is_integer() or ddof < 0:
                        raise TypedDslError(
                            "INVALID_PARAMETER",
                            f"{spec.operator_id} 的 ddof 必须是非负整数常数。",
                            node_id=len(self.nodes),
                            details={
                                "operator": spec.operator_id,
                                "parameter": "ddof",
                                "expected": "non-negative integer constant",
                                "actual": ast.unparse(node.args[1]),
                            },
                        )
            return self._append_operator(
                "call",
                spec,
                input_ids,
                input_types,
                output_type,
                ast.unparse(node),
                cache_key,
            )

        raise TypedDslError(
            "ILLEGAL_AST",
            f"不允许的表达式节点: {type(node).__name__}",
            details={"ast": cache_key},
        )

    def _append_operator(
        self,
        kind: str,
        spec: TypedOperatorSpec,
        input_ids: tuple[int, ...],
        input_types: tuple[ValueType, ...],
        output_type: ValueType,
        formula_fragment: str,
        raw: str,
    ) -> int:
        return self._append(
            kind=kind,
            label=spec.operator_id,
            inputs=input_ids,
            inferred_type=output_type,
            spec=spec,
            cost_model=spec.cost_model,
            cost_expression=spec.cost_expression(input_types, output_type),
            formula_fragment=formula_fragment,
            raw=raw,
        )

    def _append(
        self,
        *,
        kind: str,
        label: str,
        inputs: tuple[int, ...],
        inferred_type: ValueType,
        spec: TypedOperatorSpec | None,
        cost_model: str,
        cost_expression: str,
        formula_fragment: str,
        raw: str,
    ) -> int:
        if len(self.nodes) >= self.max_nodes:
            raise TypedDslError(
                "FORMULA_TOO_COMPLEX",
                f"表达式节点数不能超过 {self.max_nodes}。",
                details={"limit": self.max_nodes, "dimension": "nodes"},
            )
        node_id = len(self.nodes)
        self.nodes.append(
            TypedDagNode(
                node_id=node_id,
                kind=kind,
                label=label,
                inputs=inputs,
                arguments=tuple(zip(spec.argument_names(len(inputs)), inputs))
                if spec
                else (),
                inferred_type=inferred_type,
                operator_id=spec.operator_id if spec else None,
                operator_version=spec.version if spec else None,
                cost_model=cost_model,
                cost_expression=cost_expression,
                formula_fragment=formula_fragment,
                raw=raw,
            )
        )
        self.cache[raw] = node_id
        return node_id


def _check_output_contract(output_type: ValueType, output_contract: str) -> None:
    allowed = {"any", "scalar", "tensor", "series", "vector", "matrix", "mask"}
    if output_contract not in allowed:
        raise TypedDslError(
            "INVALID_OUTPUT_CONTRACT",
            f"不支持的输出契约: {output_contract}",
            details={"allowed": sorted(allowed)},
        )
    if output_contract == "any":
        valid = True
    elif output_contract == "mask":
        valid = output_type.is_mask
    elif output_contract == "scalar":
        # Saved indicators remain finite numeric scalars. Boolean predicates are
        # formula fragments and must be reduced explicitly (for example count).
        valid = output_type.is_scalar and output_type.is_numeric
    elif output_contract == "tensor":
        valid = not output_type.is_scalar and output_type.is_numeric
    else:
        valid = output_contract == output_type.kind and output_type.is_numeric
    if not valid:
        expected_label = {
            "scalar": "有限标量",
            "tensor": "数值数组",
            "series": "时间序列",
            "vector": "资产向量",
            "matrix": "矩阵",
            "mask": "布尔掩码",
        }.get(output_contract, "指定的数据类型")
        raise TypedDslError(
            "OUTPUT_CONTRACT_MISMATCH",
            f"指标最终结果必须是{expected_label}，当前公式输出为{user_type_label(output_type)}。"
            "请继续使用求和、平均值、标准差等归约算子，将结果转换为单个数值。"
            if output_contract == "scalar"
            else f"输出数据要求为{expected_label}，当前公式输出为{user_type_label(output_type)}。",
            details={"contract": output_contract, "actual": output_type.to_dict()},
        )


def compose_typed_expression(
    expression: str,
    *,
    variable_types: Mapping[str, ValueType | Mapping[str, Any]] | None = None,
    output_contract: str = "scalar",
    dsl_version: str = TYPED_DSL_VERSION,
    operator_registry_version: str | None = None,
    max_nodes: int = DEFAULT_MAX_NODES,
    max_depth: int = DEFAULT_MAX_DEPTH,
) -> TypedExpressionPlan:
    """Compile an expression and enforce the requested public output contract."""

    if dsl_version not in SUPPORTED_TYPED_DSL_VERSIONS:
        raise TypedDslError(
            "UNSUPPORTED_DSL_VERSION",
            f"不支持 typed DSL 版本 {dsl_version}。",
            details={"available_versions": sorted(SUPPORTED_TYPED_DSL_VERSIONS)},
        )
    expected_registry_version = {
        LEGACY_TYPED_DSL_VERSION: LEGACY_OPERATOR_REGISTRY_VERSION,
        COMPAT_TYPED_DSL_VERSION: COMPAT_OPERATOR_REGISTRY_VERSION,
        TYPED_DSL_VERSION: TYPED_OPERATOR_REGISTRY_VERSION,
    }[dsl_version]
    if operator_registry_version is None:
        operator_registry_version = expected_registry_version
    elif operator_registry_version != expected_registry_version:
        raise TypedDslError(
            "OPERATOR_VERSION_MISMATCH",
            f"DSL {dsl_version} 必须使用算子注册表 {expected_registry_version}。",
            details={
                "expected": expected_registry_version,
                "actual": operator_registry_version,
            },
        )
    if not expression or not expression.strip():
        raise TypedDslError("EMPTY_EXPRESSION", "公式不能为空。")
    variables = _normalize_variable_types(variable_types)
    registry = get_typed_operator_registry(operator_registry_version)
    parser = TypedExpressionParser(tuple(variables))
    python_expression, ast_root = parser.parse(expression)
    builder = _TypedDagBuilder(
        variables,
        registry,
        max_nodes=max_nodes,
        max_depth=max_depth,
    )
    root_id = builder.build(ast_root)
    output_type = builder.nodes[root_id].inferred_type
    _check_output_contract(output_type, output_contract)

    used_variables = {
        node.label: node.inferred_type
        for node in builder.nodes
        if node.kind == "variable"
    }
    costs = [
        node.cost_expression for node in builder.nodes if node.cost_expression != "0"
    ]
    estimated_cost = {
        "unit": "primitive_ops",
        "symbolic": "+".join(costs) if costs else "0",
        "node_count": len(builder.nodes),
    }
    canonical_expression = ast.dump(ast_root, include_attributes=False)
    return TypedExpressionPlan(
        expression=expression,
        python_expression=python_expression,
        expression_hash=hashlib.sha256(
            canonical_expression.encode("utf-8")
        ).hexdigest(),
        dsl_version=dsl_version,
        compiler_version={
            LEGACY_TYPED_DSL_VERSION: LEGACY_TYPED_COMPILER_VERSION,
            COMPAT_TYPED_DSL_VERSION: COMPAT_TYPED_COMPILER_VERSION,
            TYPED_DSL_VERSION: TYPED_COMPILER_VERSION,
        }[dsl_version],
        operator_registry_version=operator_registry_version,
        nodes=tuple(builder.nodes),
        root_id=root_id,
        output_type=output_type,
        output_contract=output_contract,
        context_requirements=used_variables,
        estimated_cost=estimated_cost,
    )


def infer_typed_expression(
    expression: str,
    *,
    variable_types: Mapping[str, ValueType | Mapping[str, Any]] | None = None,
    allow_non_scalar_root: bool = True,
    **kwargs: Any,
) -> TypedExpressionPlan:
    """Infer a typed DAG; preview callers may opt into a non-scalar root."""

    output_contract = "any" if allow_non_scalar_root else "scalar"
    return compose_typed_expression(
        expression,
        variable_types=variable_types,
        output_contract=output_contract,
        **kwargs,
    )


def _runtime_element_count(value: Any) -> int:
    return int(np.asarray(value).size)


def _operator_runtime_cost(spec: TypedOperatorSpec, arguments: Sequence[Any]) -> int:
    arrays = [np.asarray(argument) for argument in arguments]
    if spec.operator_id == "matmul":
        lhs, rhs = arrays
        return int(lhs.shape[0] * lhs.shape[1] * rhs.shape[1])
    if spec.operator_id in {"covariance", "correlation"} and len(arrays) == 1:
        time_count, asset_count = arrays[0].shape
        return int(time_count * asset_count * asset_count)
    if spec.operator_id == "solve":
        return int(arrays[0].shape[0] ** 3)
    if spec.operator_id == "outer":
        return int(arrays[0].size * arrays[1].size)
    if spec.operator_id in {"matvec", "portfolio_returns"}:
        return int(arrays[0].shape[0] * arrays[0].shape[1])
    return max(1, sum(int(array.size) for array in arrays))


class TypedIndicatorRuntime:
    """Execute a compiled typed DAG with concrete shape and budget checks."""

    def __init__(
        self,
        plan: TypedExpressionPlan,
        *,
        max_time: int = DEFAULT_MAX_TIME,
        max_assets: int = DEFAULT_MAX_ASSETS,
        max_runtime_cost: int = DEFAULT_MAX_RUNTIME_COST,
        max_live_elements: int = DEFAULT_MAX_LIVE_ELEMENTS,
        _compiled_plan: CompiledNumbaPlan | None = None,
    ) -> None:
        self.plan = plan
        self.registry = get_typed_operator_registry(plan.operator_registry_version)
        self.max_time = max_time
        self.max_assets = max_assets
        self.max_runtime_cost = max_runtime_cost
        self.max_live_elements = max_live_elements
        self.last_trace: tuple[dict[str, Any], ...] = ()
        if _compiled_plan is not None:
            expected_plan_id = numba_plan_id(plan)
            if _compiled_plan.plan_id != expected_plan_id:
                raise TypedDslError(
                    "NJIT_PLAN_ID_MISMATCH",
                    "已预热 NJIT 计划与当前 typed DAG 不一致。",
                    details={
                        "expected_compiled_plan_id": expected_plan_id,
                        "actual_compiled_plan_id": _compiled_plan.plan_id,
                    },
                )
            self.compiled_plan = _compiled_plan
        else:
            try:
                self.compiled_plan = compile_numba_plan(plan)
            except NumbaPlanCompileError as exc:
                raise TypedDslError(
                    "NJIT_PLAN_COMPILE_FAILED",
                    "公式无法编译为 NJIT 计算计划。",
                    details={
                        "compiled_plan_id": exc.plan_id,
                        "operator": exc.operator_id,
                    },
                ) from exc

    @classmethod
    def from_expression(
        cls,
        expression: str,
        *,
        variable_types: Mapping[str, ValueType | Mapping[str, Any]] | None = None,
        output_contract: str = "scalar",
        **kwargs: Any,
    ) -> "TypedIndicatorRuntime":
        compile_keys = {
            "dsl_version",
            "operator_registry_version",
            "max_nodes",
            "max_depth",
        }
        compile_options = {
            key: kwargs.pop(key) for key in tuple(kwargs) if key in compile_keys
        }
        plan = compose_typed_expression(
            expression,
            variable_types=variable_types,
            output_contract=output_contract,
            **compile_options,
        )
        return cls(plan, **kwargs)

    @classmethod
    def from_plan(
        cls, plan: TypedExpressionPlan, **kwargs: Any
    ) -> "TypedIndicatorRuntime":
        return cls(plan, **kwargs)

    @classmethod
    def from_warmed_plan(
        cls, plan: TypedExpressionPlan, **kwargs: Any
    ) -> "TypedIndicatorRuntime":
        """Bind an immutable plan cache entry without compiling a signature."""

        compiled = get_cached_numba_plan(plan)
        if compiled is None:
            raise TypedDslError(
                "NJIT_PLAN_NOT_WARMED",
                "当前公式没有已预热的固定签名 NJIT 计划。",
                details={"compiled_plan_id": numba_plan_id(plan)},
            )
        return cls(plan, _compiled_plan=compiled, **kwargs)

    def compute(self, context: Mapping[str, Any]) -> Any:
        bindings: dict[str, int] = {}
        self.last_trace = ()
        variable_nodes = {
            node.label: node
            for node in self.plan.nodes
            if node.kind == "variable"
        }
        arguments: list[Any] = []
        for name in self.compiled_plan.context_names:
            node = variable_nodes[name]
            if name not in context:
                raise TypedDslError(
                    "CONTEXT_VARIABLE_UNAVAILABLE",
                    f"缺少上下文变量: {name}",
                    node_id=node.node_id,
                )
            value = self._validate_value(
                context[name],
                node.inferred_type,
                bindings,
                label=name,
                node_id=node.node_id,
                allow_bind=True,
                internal_mask=True,
            )
            if name == "asset_weights" and _weight_vector_sum_kernel(
                value, 1e-8
            ) != 1:
                raise TypedDslError(
                    "WEIGHT_SUM_INVALID", "asset_weights 必须合计为 1。", node_id=node.node_id
                )
            if name == "weight_path" and _weight_path_sum_kernel(
                value, 1e-8
            ) != 1:
                raise TypedDslError(
                    "WEIGHT_SUM_INVALID",
                    "weight_path 每个时间点的资产权重必须合计为 1。",
                    node_id=node.node_id,
                )
            arguments.append(value)

        trace = self._build_compiled_trace(bindings)
        total_cost = sum(item["runtime_cost"] for item in trace)
        if total_cost > self.max_runtime_cost:
            raise TypedDslError(
                "COMPUTE_BUDGET_EXCEEDED",
                f"计算预算超过 {self.max_runtime_cost}。",
                node_id=self.plan.root_id,
                details={"estimated_cost": total_cost},
            )
        live_elements = sum(max(1, math.prod(item["actual_shape"])) for item in trace)
        if live_elements > self.max_live_elements:
            raise TypedDslError(
                "COMPUTE_BUDGET_EXCEEDED",
                f"DAG 活跃元素数超过 {self.max_live_elements}。",
                node_id=self.plan.root_id,
                details={"live_elements": live_elements},
            )
        try:
            result = self.compiled_plan.compute(tuple(arguments))
        except (TypeError, ValueError, ZeroDivisionError, FloatingPointError) as exc:
            code = str(exc).strip()
            stable_codes = {
                "DIVIDE_BY_ZERO",
                "DOMAIN_ERROR",
                "INVALID_PARAMETER",
                "INSUFFICIENT_SAMPLE",
                "NON_FINITE_RESULT",
                "SINGULAR_MATRIX",
            }
            if isinstance(exc, TypeError):
                code = "NJIT_SIGNATURE_MISMATCH"
            if code not in stable_codes:
                code = (
                    "NJIT_SIGNATURE_MISMATCH"
                    if code == "NJIT_SIGNATURE_MISMATCH"
                    else "OPERATOR_EXECUTION_FAILED"
                )
            raise TypedDslError(
                code,
                "NJIT 计算计划执行失败。",
                node_id=self.plan.root_id,
            ) from exc
        result = self._validate_value(
            result,
            self.plan.output_type,
            bindings,
            label="result",
            node_id=self.plan.root_id,
            allow_bind=False,
            internal_mask=False,
        )
        self.last_trace = tuple(trace)
        if isinstance(result, np.ndarray):
            return result
        if isinstance(result, (bool, np.bool_)):
            return bool(result)
        return float(result)

    def trace_payload(self) -> dict[str, Any]:
        payload = {
            "nodes": [dict(item) for item in self.last_trace],
            "total_runtime_cost": sum(item["runtime_cost"] for item in self.last_trace),
            **self.compiled_plan.metadata(),
        }
        payload["runtime_validation_kernel_signatures"] = (
            runtime_validation_kernel_signatures()
        )
        payload["kernel_signatures"] = {
            **payload.get("kernel_signatures", {}),
            **payload["runtime_validation_kernel_signatures"],
        }
        return validate_execution_audit(payload)

    def _build_compiled_trace(self, bindings: Mapping[str, int]) -> list[dict[str, Any]]:
        trace: list[dict[str, Any]] = []
        shapes: dict[int, tuple[int, ...]] = {}
        for node in self.plan.nodes:
            shape: list[int] = []
            for dimension in node.inferred_type.shape:
                if isinstance(dimension, int):
                    shape.append(dimension)
                    continue
                direct = re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", dimension)
                derived = re.fullmatch(r"([A-Za-z][A-Za-z0-9_]*)-(\d+)", dimension)
                dynamic = re.fullmatch(r"([A-Za-z][A-Za-z0-9_]*)-n", dimension)
                if direct and dimension in bindings:
                    shape.append(bindings[dimension])
                elif derived and derived.group(1) in bindings:
                    shape.append(max(1, bindings[derived.group(1)] - int(derived.group(2))))
                elif dynamic and dynamic.group(1) in bindings:
                    shape.append(bindings[dynamic.group(1)])
                else:
                    shape.append(1)
            shapes[node.node_id] = tuple(shape)
            node_cost = 0
            if node.operator_id is not None:
                input_elements = [max(1, math.prod(shapes[input_id])) for input_id in node.inputs]
                if node.operator_id in {"covariance", "correlation"} and len(node.inputs) == 1 and len(shapes[node.inputs[0]]) == 2:
                    time_count, asset_count = shapes[node.inputs[0]]
                    node_cost = time_count * asset_count * asset_count
                elif node.operator_id == "solve" and shapes[node.inputs[0]]:
                    node_cost = shapes[node.inputs[0]][0] ** 3
                else:
                    node_cost = max(1, sum(input_elements))
            trace.append(
                {
                    "node_id": node.node_id,
                    "actual_shape": list(shape),
                    "runtime_cost": node_cost,
                }
            )
        return trace

    def _validate_value(
        self,
        value: Any,
        expected: ValueType,
        bindings: dict[str, int],
        *,
        label: str,
        node_id: int,
        allow_bind: bool,
        internal_mask: bool = False,
    ) -> Any:
        raw_array = np.asarray(value)
        if expected.is_mask:
            is_boolean = np.issubdtype(raw_array.dtype, np.bool_)
            if raw_array.dtype == np.uint8:
                mask_candidate = np.ascontiguousarray(raw_array, dtype=np.uint8)
                if not mask_candidate.flags.writeable:
                    mask_candidate = mask_candidate.copy()
                is_uint8_mask = (
                    _binary_mask_1d_kernel(mask_candidate) == 1
                    if mask_candidate.ndim == 1
                    else _binary_mask_2d_kernel(mask_candidate) == 1
                    if mask_candidate.ndim == 2
                    else bool(mask_candidate.ndim == 0 and int(mask_candidate) <= 1)
                )
            else:
                is_uint8_mask = False
            if not is_boolean and not is_uint8_mask:
                raise TypedDslError(
                    "RUNTIME_TYPE_MISMATCH",
                    f"{label} 需要布尔 mask。",
                    node_id=node_id,
                )
            array = np.asarray(value, dtype=np.uint8 if internal_mask else np.bool_)
        else:
            if np.issubdtype(raw_array.dtype, np.bool_):
                raise TypedDslError(
                    "RUNTIME_TYPE_MISMATCH",
                    f"{label} 不能是布尔值。",
                    node_id=node_id,
                )
            array = np.asarray(value, dtype=np.float64)
        if array.ndim != expected.rank:
            raise TypedDslError(
                "RUNTIME_SHAPE_MISMATCH",
                f"{label} 需要 rank={expected.rank}，实际 rank={array.ndim}。",
                node_id=node_id,
                details={
                    "expected": expected.to_dict(),
                    "actual_shape": list(array.shape),
                },
            )
        if expected.is_numeric:
            if array.ndim == 0:
                finite = _finite_scalar_kernel(float(array)) == 1
            else:
                numeric_array = np.ascontiguousarray(array, dtype=np.float64)
                if not numeric_array.flags.writeable:
                    numeric_array = numeric_array.copy()
                finite = (
                    _finite_1d_kernel(numeric_array) == 1
                    if numeric_array.ndim == 1
                    else _finite_2d_kernel(numeric_array) == 1
                )
                array = numeric_array
            if not finite:
                code = "NON_FINITE_INPUT" if allow_bind else "NON_FINITE_RESULT"
                raise TypedDslError(code, f"{label} 包含 NaN 或 Inf。", node_id=node_id)
        for axis, expected_dimension, actual_dimension in zip(
            expected.axes,
            expected.shape,
            array.shape,
        ):
            if actual_dimension <= 0:
                code = (
                    "INSUFFICIENT_SAMPLE"
                    if axis == "time"
                    else "RUNTIME_SHAPE_MISMATCH"
                )
                raise TypedDslError(
                    code,
                    f"{label} 的 {axis} 维不能为空。",
                    node_id=node_id,
                )
            if axis == "time" and actual_dimension > self.max_time:
                raise TypedDslError(
                    "SHAPE_LIMIT_EXCEEDED",
                    f"时间观察数不能超过 {self.max_time}。",
                    node_id=node_id,
                )
            if axis == "asset" and actual_dimension > self.max_assets:
                raise TypedDslError(
                    "SHAPE_LIMIT_EXCEEDED",
                    f"资产数不能超过 {self.max_assets}。",
                    node_id=node_id,
                )
            if isinstance(expected_dimension, int):
                if actual_dimension != expected_dimension:
                    raise TypedDslError(
                        "RUNTIME_SHAPE_MISMATCH",
                        f"{label} 的 {axis} 维应为 {expected_dimension}，实际为 {actual_dimension}。",
                        node_id=node_id,
                    )
                continue
            direct_symbol = re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", expected_dimension)
            derived_symbol = re.fullmatch(
                r"([A-Za-z][A-Za-z0-9_]*)-(\d+)", expected_dimension
            )
            dynamic_symbol = re.fullmatch(
                r"([A-Za-z][A-Za-z0-9_]*)-n", expected_dimension
            )
            if direct_symbol:
                bound = bindings.get(expected_dimension)
            elif derived_symbol:
                base, decrement_text = derived_symbol.groups()
                base_bound = bindings.get(base)
                bound = (
                    base_bound - int(decrement_text) if base_bound is not None else None
                )
            elif dynamic_symbol:
                base_bound = bindings.get(dynamic_symbol.group(1))
                if base_bound is None:
                    bound = None
                elif not 0 < actual_dimension <= base_bound:
                    raise TypedDslError(
                        "RUNTIME_SHAPE_MISMATCH",
                        f"{label} 的动态窗口输出长度无效。",
                        node_id=node_id,
                    )
                else:
                    bound = actual_dimension
            else:
                bound = None
            if direct_symbol and bound is None and allow_bind:
                bindings[expected_dimension] = actual_dimension
            elif bound is None:
                raise TypedDslError(
                    "UNRESOLVED_SHAPE",
                    f"无法解析 shape 变量 {expected_dimension}。",
                    node_id=node_id,
                )
            elif bound != actual_dimension:
                raise TypedDslError(
                    "RUNTIME_SHAPE_MISMATCH",
                    f"{label} 的 {axis} 维与 {expected_dimension}={bound} 不一致。",
                    node_id=node_id,
                    details={
                        "actual": actual_dimension,
                        "symbol": expected_dimension,
                        "bound": bound,
                    },
                )
        if expected.is_scalar:
            if expected.is_mask:
                return np.uint8(bool(array)) if internal_mask else bool(array)
            return float(array)
        dtype = np.uint8 if expected.is_mask and internal_mask else (
            np.bool_ if expected.is_mask else np.float64
        )
        contiguous = np.ascontiguousarray(array, dtype=dtype)
        if not contiguous.flags.writeable:
            contiguous = contiguous.copy()
        return contiguous


def evaluate_typed_expression(
    expression: str,
    context: Mapping[str, Any],
    *,
    output_contract: str = "scalar",
    variable_types: Mapping[str, ValueType | Mapping[str, Any]] | None = None,
    **kwargs: Any,
) -> Any:
    runtime = TypedIndicatorRuntime.from_expression(
        expression,
        variable_types=variable_types,
        output_contract=output_contract,
        **kwargs,
    )
    return runtime.compute(context)


# Short aliases are intentionally provided for service-layer composition.
catalog = get_typed_dsl_catalog
compose = compose_typed_expression
infer = infer_typed_expression


__all__ = [
    "DEFAULT_VARIABLE_TYPES",
    "TYPED_COMPILER_VERSION",
    "TYPED_DSL_VERSION",
    "TYPED_OPERATOR_REGISTRY_VERSION",
    "TypedDagNode",
    "TypedDslError",
    "TypedExpressionParser",
    "TypedExpressionPlan",
    "TypedIndicatorRuntime",
    "ValueType",
    "catalog",
    "compose",
    "compose_typed_expression",
    "evaluate_typed_expression",
    "get_typed_dsl_catalog",
    "get_typed_operator_catalog",
    "get_typed_variable_catalog",
    "infer",
    "infer_typed_expression",
    "runtime_validation_execution_audit",
    "runtime_validation_kernel_signatures",
]

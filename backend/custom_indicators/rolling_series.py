"""Compile a locked scalar indicator into a first-class rolling time series."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Mapping

from cal_indicators.typed_dsl import TypedDslError

from .errors import ValidationError
from .formula_source import canonical_formula_source
from .variable_registry import get_variable


ROLLING_SOURCE_KIND = "rolling_scalar"
ROLLING_TRANSFORM_VERSION = "1.0.0"
MIN_ROLLING_WINDOW = 1
MAX_ROLLING_WINDOW = 5_000

# A scalar indicator may use these reductions to collapse a one-dimensional
# series. The rolling compiler replaces only those reductions; all surrounding
# elementwise arithmetic remains unchanged and is type-checked by typed DSL.
_SUPPORTED_REDUCTIONS = frozenset(
    {"mean", "std", "variance", "min_value", "max_value"}
)

# These functions are safe to preserve around the transformed rolling series.
# Unsupported reducers fail closed instead of silently changing semantics.
_PASSTHROUGH_FUNCTIONS = frozenset(
    {
        "absolute",
        "add",
        "clip",
        "divide",
        "divide_or_default",
        "equal",
        "exp",
        "greater_equal",
        "greater_than",
        "less_equal",
        "less_than",
        "log",
        "logical_and",
        "logical_not",
        "logical_or",
        "maximum",
        "minimum",
        "multiply",
        "negate",
        "normal_pdf",
        "normal_ppf",
        "not_equal",
        "power",
        "reciprocal",
        "sign",
        "sqrt",
        "subtract",
        "where",
    }
)

# Scalar values supplied by the single-product runtime can be broadcast against
# every point of the generated rolling series.
_RUNTIME_SCALAR_NAMES = frozenset(
    {
        "observation_count",
        "window_elapsed_days",
        "risk_free_return_window",
        "annual_risk_free_rate_decimal",
        "risk_free_rate_per_observation",
        "periods_per_year",
    }
)

_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class RollingTransformResult:
    expression: str
    series_variables: tuple[str, ...]
    reductions: tuple[str, ...]

    @property
    def rewritten_reductions(self) -> tuple[str, ...]:
        """Compatibility alias used by Indicator Center metadata."""

        return self.reductions


def _validate_window(value: Any) -> int:
    if isinstance(value, bool):
        raise ValidationError(
            "INVALID_ROLLING_WINDOW",
            "滚动观察数必须是正整数。",
            field="window_observations",
        )
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            "INVALID_ROLLING_WINDOW",
            "滚动观察数必须是正整数。",
            field="window_observations",
        ) from exc
    if (
        not math.isfinite(number)
        or not number.is_integer()
        or not MIN_ROLLING_WINDOW <= number <= MAX_ROLLING_WINDOW
    ):
        raise ValidationError(
            "INVALID_ROLLING_WINDOW",
            f"滚动观察数必须是 {MIN_ROLLING_WINDOW} 至 {MAX_ROLLING_WINDOW} 的整数。",
            field="window_observations",
        )
    return int(number)


def _numeric_contract(definition: Mapping[str, Any]) -> dict[str, Any]:
    """Return the immutable scalar contract used by rolling provenance."""

    keys = (
        "id",
        "revision",
        "result_kind",
        "context_kind",
        "expression",
        "annual_risk_free_rate_percent",
        "dsl_version",
        "operator_registry_version",
        "numeric_kernel_version",
        "variable_registry_version",
        "data_contract_version",
        "context_schema_version",
        "output_contract",
        "output_measure",
        "unit",
        "display_format",
        "precision",
        "direction",
        "indicator_type",
        "required_variables",
        "applicable_product_kinds",
    )
    contract = {key: copy.deepcopy(definition.get(key)) for key in keys}
    # Catalog decoration materializes defaults that older persisted and built-in
    # definitions may omit. Hash the semantic defaults, not the serialization
    # shape, so a GET → validate round trip cannot invalidate its own source.
    contract["result_kind"] = str(definition.get("result_kind") or "scalar")
    contract["context_kind"] = str(
        definition.get("context_kind") or "single_product"
    )
    contract["output_contract"] = str(
        definition.get("output_contract") or "scalar"
    )
    return contract


def scalar_definition_hash(definition: Mapping[str, Any]) -> str:
    payload = json.dumps(
        _numeric_contract(definition),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class _RollingScalarTransformer(ast.NodeTransformer):
    def __init__(self, window: int) -> None:
        self.window = window
        self.reductions: list[str] = []
        self.series_variables: list[str] = []

    def visit_Name(self, node: ast.Name) -> ast.AST:  # noqa: N802
        variable = get_variable(node.id)
        if variable is not None and variable.kind == "series":
            if node.id not in self.series_variables:
                self.series_variables.append(node.id)
            return node
        if variable is not None and variable.kind == "scalar":
            if node.id not in _RUNTIME_SCALAR_NAMES:
                raise ValidationError(
                    "ROLLING_SCALAR_CONTEXT_UNSUPPORTED",
                    f"滚动转换暂不支持运行时标量 {node.id}。",
                    field="indicator_id",
                )
            return node
        # Function identifiers are visited as Call.func and validated there.
        return node

    @staticmethod
    def _literal_ddof(node: ast.AST | None) -> ast.Constant:
        if node is None:
            return ast.Constant(value=1)
        if (
            not isinstance(node, ast.Constant)
            or isinstance(node.value, bool)
            or not isinstance(node.value, (int, float))
        ):
            raise ValidationError(
                "ROLLING_REDUCTION_PARAMETER_UNSUPPORTED",
                "滚动标准差和方差的 ddof 必须是公式中的非负整数常量。",
                field="indicator_id",
            )
        value = float(node.value)
        if not math.isfinite(value) or not value.is_integer() or value < 0:
            raise ValidationError(
                "ROLLING_REDUCTION_PARAMETER_UNSUPPORTED",
                "滚动标准差和方差的 ddof 必须是公式中的非负整数常量。",
                field="indicator_id",
            )
        return ast.Constant(value=int(value))

    def visit_Call(self, node: ast.Call) -> ast.AST:  # noqa: N802
        if not isinstance(node.func, ast.Name) or node.keywords:
            raise ValidationError(
                "ROLLING_SCALAR_EXPRESSION_UNSUPPORTED",
                "标量指标包含滚动转换不支持的函数调用。",
                field="indicator_id",
            )
        operator_id = node.func.id
        if operator_id in _SUPPORTED_REDUCTIONS:
            if operator_id in {"std", "variance"}:
                if len(node.args) not in {1, 2}:
                    raise ValidationError(
                        "ROLLING_REDUCTION_PARAMETER_UNSUPPORTED",
                        f"{operator_id} 只能使用数值序列和可选 ddof。",
                        field="indicator_id",
                    )
                values = self.visit(node.args[0])
                ddof = self._literal_ddof(node.args[1] if len(node.args) == 2 else None)
                rolling_std = ast.Call(
                    func=ast.Name(id="rolling_std", ctx=ast.Load()),
                    args=[values, ast.Constant(value=self.window), ddof],
                    keywords=[],
                )
                replacement: ast.AST = rolling_std
                if operator_id == "variance":
                    replacement = ast.Call(
                        func=ast.Name(id="power", ctx=ast.Load()),
                        args=[rolling_std, ast.Constant(value=2)],
                        keywords=[],
                    )
            else:
                if len(node.args) != 1:
                    raise ValidationError(
                        "ROLLING_REDUCTION_PARAMETER_UNSUPPORTED",
                        f"{operator_id} 只能接收一个数值序列。",
                        field="indicator_id",
                    )
                values = self.visit(node.args[0])
                rolling_operator = {
                    "mean": "rolling_mean",
                    "min_value": "rolling_min",
                    "max_value": "rolling_max",
                }[operator_id]
                replacement = ast.Call(
                    func=ast.Name(id=rolling_operator, ctx=ast.Load()),
                    args=[values, ast.Constant(value=self.window)],
                    keywords=[],
                )
            self.reductions.append(operator_id)
            return ast.copy_location(replacement, node)

        if operator_id not in _PASSTHROUGH_FUNCTIONS:
            raise ValidationError(
                "ROLLING_SCALAR_OPERATOR_UNSUPPORTED",
                f"标量算子 {operator_id} 不能自动转换为滚动时序计算。",
                field="indicator_id",
                diagnostics=[
                    {
                        "code": "ROLLING_SCALAR_OPERATOR_UNSUPPORTED",
                        "operator": operator_id,
                        "supported_reductions": sorted(_SUPPORTED_REDUCTIONS),
                    }
                ],
            )
        return ast.copy_location(
            ast.Call(
                func=ast.Name(id=operator_id, ctx=ast.Load()),
                args=[self.visit(argument) for argument in node.args],
                keywords=[],
            ),
            node,
        )


def transform_scalar_expression(
    expression: str,
    window_observations: Any,
) -> RollingTransformResult:
    window = _validate_window(window_observations)
    try:
        parsed = ast.parse(str(expression or "").strip(), mode="eval")
    except SyntaxError as exc:
        raise ValidationError(
            "ROLLING_SCALAR_EXPRESSION_UNSUPPORTED",
            "标量指标公式无法转换为滚动时序公式。",
            field="indicator_id",
        ) from exc
    transformer = _RollingScalarTransformer(window)
    transformed = transformer.visit(parsed)
    ast.fix_missing_locations(transformed)
    if not transformer.reductions:
        raise ValidationError(
            "ROLLING_SCALAR_REDUCTION_REQUIRED",
            "该标量公式没有可转换的序列归约步骤。",
            field="indicator_id",
        )
    if not transformer.series_variables:
        raise ValidationError(
            "ROLLING_SCALAR_SERIES_REQUIRED",
            "该标量公式没有单产品时间序列输入。",
            field="indicator_id",
        )
    return RollingTransformResult(
        expression=ast.unparse(transformed.body),
        series_variables=tuple(transformer.series_variables),
        reductions=tuple(transformer.reductions),
    )


def _axis_anchor(series_variables: tuple[str, ...]) -> str:
    # Return and log-return observations are derived from adjacent adjusted NAV
    # points. Anchoring to adjusted NAV preserves the boundary observation and
    # represents the first unavailable return as a real null rather than
    # shifting all subsequent dates.
    if any(name in {"returns", "log_returns", "adjusted_nav"} for name in series_variables):
        return "adjusted_nav"
    for name in series_variables:
        variable = get_variable(name)
        if variable is not None and variable.kind == "series":
            return name
    raise ValidationError(
        "ROLLING_SCALAR_AXIS_UNRESOLVED",
        "无法确定滚动时序指标的日期轴。",
        field="indicator_id",
    )


def normalize_rolling_source(raw: Any) -> dict[str, Any] | None:
    """Normalize rolling provenance to one stable public contract.

    Older drafts used ``version`` / ``source_definition_hash`` while newer
    drafts use ``transform_version`` / ``definition_hash``. Both spellings are
    accepted at the API boundary, but all stored and returned definitions use
    the canonical fields below.
    """

    if raw is None or raw == "":
        return None
    if not isinstance(raw, Mapping):
        raise ValidationError(
            "INVALID_ROLLING_SOURCE",
            "滚动来源必须是对象。",
            field="rolling_source",
        )
    if raw.get("output_id") is not None:
        raise ValidationError(
            "REMOVED_SCALAR_OUTPUT_REFERENCE",
            "滚动来源必须引用独立指标及其版本，不能引用已取消的标量子结果。",
            field="rolling_source",
        )
    kind = str(raw.get("kind") or ROLLING_SOURCE_KIND)
    transform_version = str(
        raw.get("transform_version")
        or raw.get("version")
        or ROLLING_TRANSFORM_VERSION
    )
    indicator_id = str(raw.get("indicator_id") or "").strip()
    try:
        indicator_revision = int(raw.get("indicator_revision") or 0)
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            "INVALID_ROLLING_SOURCE",
            "滚动来源指标版本无效。",
            field="rolling_source.indicator_revision",
        ) from exc
    definition_hash = str(
        raw.get("definition_hash")
        or raw.get("source_definition_hash")
        or ""
    ).lower()
    window = _validate_window(raw.get("window_observations"))
    raw_minimum = raw.get("minimum_observations")
    if raw_minimum is None:
        raw_minimum = raw.get("min_periods")
    minimum = _validate_window(window if raw_minimum is None else raw_minimum)
    if minimum > window:
        raise ValidationError(
            "INVALID_ROLLING_SOURCE",
            "滚动指标的最少有效观察数不能大于窗口观察数。",
            field="rolling_source.minimum_observations",
        )
    detached = bool(raw.get("detached", False))
    if kind != ROLLING_SOURCE_KIND or transform_version != ROLLING_TRANSFORM_VERSION:
        raise ValidationError(
            "ROLLING_TRANSFORM_VERSION_UNSUPPORTED",
            "滚动来源使用了不支持的转换协议版本。",
            field="rolling_source.transform_version",
        )
    if not indicator_id or indicator_revision < 1 or not _HASH_PATTERN.fullmatch(definition_hash):
        raise ValidationError(
            "INVALID_ROLLING_SOURCE",
            "滚动来源缺少有效的指标 ID、版本或定义哈希。",
            field="rolling_source",
        )
    return {
        "kind": kind,
        "transform_version": transform_version,
        "indicator_id": indicator_id,
        "indicator_revision": indicator_revision,
        "indicator_name": str(raw.get("indicator_name") or indicator_id).strip()[:80],
        "definition_hash": definition_hash,
        "source_dsl_version": str(raw.get("source_dsl_version") or "").strip()[:40],
        "window_observations": window,
        "minimum_observations": minimum,
        "detached": detached,
    }


def derive_rolling_series_definition(
    source_definition: Mapping[str, Any],
    window_observations: Any,
    *,
    name: str | None = None,
    description: str | None = None,
) -> dict[str, Any]:
    """Materialize a scalar definition as an ordinary typed series formula."""

    if str(source_definition.get("result_kind") or "scalar") != "scalar":
        raise ValidationError(
            "ROLLING_SOURCE_MUST_BE_SCALAR",
            "只有标量指标可以转换为滚动时序指标。",
            field="indicator_id",
        )
    if str(source_definition.get("context_kind") or "single_product") != "single_product":
        raise ValidationError(
            "ROLLING_SOURCE_CONTEXT_UNSUPPORTED",
            "滚动转换当前只支持单产品标量指标。",
            field="indicator_id",
        )
    if not str(source_definition.get("dsl_version") or "").startswith("2."):
        raise ValidationError(
            "ROLLING_SOURCE_DSL_UNSUPPORTED",
            "滚动转换只支持 typed DSL 标量指标。",
            field="indicator_id",
        )
    source_id = str(source_definition.get("id") or "").strip()
    try:
        source_revision = int(source_definition.get("revision") or 0)
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            "INVALID_ROLLING_SOURCE",
            "来源指标必须是已保存的不可变版本。",
            field="indicator_id",
        ) from exc
    if not source_id or source_revision < 1:
        raise ValidationError(
            "INVALID_ROLLING_SOURCE",
            "来源指标必须是已保存的不可变版本。",
            field="indicator_id",
        )

    window = _validate_window(window_observations)
    transformed = transform_scalar_expression(
        str(source_definition.get("expression") or ""),
        window,
    )
    source_name = str(source_definition.get("name") or source_id)
    resolved_name = str(name or f"{window} 日滚动{source_name}").strip()
    resolved_description = str(
        description
        or f"由“{source_name}”第 {source_revision} 版按最近 {window} 个有效观察值滚动计算。"
    ).strip()
    rolling_source = {
        "kind": ROLLING_SOURCE_KIND,
        "transform_version": ROLLING_TRANSFORM_VERSION,
        "indicator_id": source_id,
        "indicator_revision": source_revision,
        "indicator_name": source_name[:80],
        "definition_hash": scalar_definition_hash(source_definition),
        "source_dsl_version": str(source_definition.get("dsl_version") or "")[:40],
        "window_observations": window,
        "minimum_observations": window,
        "detached": False,
    }
    return {
        "name": resolved_name,
        "description": resolved_description,
        "expression": transformed.expression,
        "series_outputs": [
            {
                "id": "value",
                "label": resolved_name,
                "expression": transformed.expression,
                "unit": str(source_definition.get("unit") or ""),
                "display_format": str(
                    source_definition.get("display_format") or "number"
                ),
                "precision": int(source_definition.get("precision", 4)),
                "output_measure": "auto",
            }
        ],
        "parameter_schema": [],
        "fixed_parameters": [
            {
                "id": "window_observations",
                "label": "滚动观察数",
                "type": "integer",
                "value": window,
                "source": "rolling_source",
            }
        ],
        "result_kind": "time_series",
        "output_contract": "series_bundle",
        "context_kind": "single_product",
        "indicator_type": str(source_definition.get("indicator_type") or "other"),
        "direction": str(source_definition.get("direction") or "higher_better"),
        "annual_risk_free_rate_percent": float(
            source_definition.get("annual_risk_free_rate_percent") or 0.0
        ),
        "axis_anchor": _axis_anchor(transformed.series_variables),
        "history_policy": "lookback",
        "lookback_parameter": None,
        "lookback_observations": window,
        "minimum_observations": window,
        "required_variables": list(transformed.series_variables),
        "applicable_product_kinds": list(
            source_definition.get("applicable_product_kinds") or ["etf", "fund"]
        ),
        "methodology": (
            f"把来源标量公式中的 {', '.join(transformed.reductions)} 归约替换为"
            f"固定 {window} 个观察值的滚动计算；其余算术与年化口径保持来源版本不变。"
        ),
        "data_basis": "沿用来源指标的真实单产品数据口径；日期对齐，缺失不填充",
        "rolling_source": rolling_source,
        "rolling_transform": {
            "version": ROLLING_TRANSFORM_VERSION,
            "window_observations": window,
            "source_expression": str(source_definition.get("expression") or ""),
            "generated_expression": transformed.expression,
            "series_variables": list(transformed.series_variables),
            "reduction_mappings": list(transformed.reductions),
        },
        "template_origin": None,
    }


def verify_rolling_series_definition(
    definition: Mapping[str, Any],
    source_definition: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Verify a generated rolling definition against its locked scalar source."""

    source = normalize_rolling_source(definition.get("rolling_source"))
    if source is None:
        return None
    if source["indicator_id"] != str(source_definition.get("id") or "") or source[
        "indicator_revision"
    ] != int(source_definition.get("revision") or 0):
        raise ValidationError(
            "ROLLING_SOURCE_REVISION_MISMATCH",
            "滚动时序指标引用的标量指标版本不一致。",
            field="rolling_source",
        )
    if source["definition_hash"] != scalar_definition_hash(source_definition):
        raise ValidationError(
            "ROLLING_SOURCE_REVISION_MISMATCH",
            "来源标量指标的计算契约与滚动定义记录不一致。",
            field="rolling_source",
        )
    if source["detached"]:
        return source

    expected = derive_rolling_series_definition(
        source_definition,
        source["window_observations"],
        name=str(definition.get("name") or ""),
        description=str(definition.get("description") or ""),
    )
    actual_outputs = definition.get("series_outputs") or []
    if len(actual_outputs) != 1:
        raise ValidationError(
            "ROLLING_SOURCE_FORMULA_MISMATCH",
            "锁定来源的滚动指标必须保留单一生成通道。",
            field="series_outputs",
        )
    actual_expression = str(actual_outputs[0].get("expression") or "").strip()
    expected_expression = str(expected["expression"]).strip()
    try:
        actual_ast = canonical_formula_source(actual_expression)
        expected_ast = canonical_formula_source(expected_expression)
    except (SyntaxError, TypedDslError) as exc:
        raise ValidationError(
            "ROLLING_SOURCE_FORMULA_MISMATCH",
            "滚动指标公式已不是有效的受控表达式。",
            field="series_outputs.0.expression",
        ) from exc
    if actual_ast != expected_ast:
        raise ValidationError(
            "ROLLING_SOURCE_FORMULA_MISMATCH",
            "滚动指标公式已偏离锁定的来源标量版本；请重新生成，或明确解除来源绑定。",
            field="series_outputs.0.expression",
        )
    expected_rate = float(
        source_definition.get("annual_risk_free_rate_percent") or 0.0
    )
    actual_rate = float(definition.get("annual_risk_free_rate_percent") or 0.0)
    if not math.isclose(actual_rate, expected_rate, rel_tol=0.0, abs_tol=1e-12):
        raise ValidationError(
            "ROLLING_SOURCE_PARAMETER_MISMATCH",
            "滚动指标必须沿用来源标量版本的无风险利率等计算参数。",
            field="annual_risk_free_rate_percent",
        )
    return source


__all__ = [
    "MAX_ROLLING_WINDOW",
    "MIN_ROLLING_WINDOW",
    "ROLLING_SOURCE_KIND",
    "ROLLING_TRANSFORM_VERSION",
    "RollingTransformResult",
    "derive_rolling_series_definition",
    "normalize_rolling_source",
    "scalar_definition_hash",
    "transform_scalar_expression",
    "verify_rolling_series_definition",
]

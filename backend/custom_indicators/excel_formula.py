"""Translate typed single-product indicator DAGs into transparent Excel formulas."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from xlsxwriter.utility import xl_range_abs, xl_rowcol_to_cell

from cal_indicators.typed_dsl import (
    TypedDagNode,
    TypedExpressionPlan,
    TypedSeriesBundlePlan,
)
from cal_indicators.typed_operators import get_typed_operator_registry

from .errors import ValidationError
from .variable_registry import get_variable


EXCEL_FORMULA_REGISTRY_VERSION = "1.3.0"

# The Excel export is a teaching and audit artifact. Formula cells must contain
# ordinary worksheet functions only: no VBA/UDF calls, no DSL operator names,
# and no hidden workbook-name indirection.
_EXCEL_FUNCTION_CALL = re.compile(
    r"(?<![A-Za-z0-9_.])([A-Za-z][A-Za-z0-9_.]*)\s*\("
)
_NATIVE_EXCEL_FUNCTIONS = frozenset(
    {
        "ABS",
        "AGGREGATE",
        "AND",
        "AVEDEV",
        "AVERAGE",
        "CORREL",
        "COUNT",
        "COUNTIF",
        "COVARIANCE.S",
        "DEVSQ",
        "EXP",
        "IF",
        "IFERROR",
        "INDEX",
        "INTERCEPT",
        "ISLOGICAL",
        "ISNUMBER",
        "KURT",
        "LN",
        "MATCH",
        "MAX",
        "MEDIAN",
        "MIN",
        "NA",
        "NORM.S.DIST",
        "NORM.S.INV",
        "NOT",
        "OR",
        "PERCENTILE.INC",
        "POWER",
        "PRODUCT",
        "ROWS",
        "RSQ",
        "SIGN",
        "SKEW",
        "SLOPE",
        "SQRT",
        "STDEV",
        "STDEVP",
        "STEYX",
        "SUM",
        "SUMPRODUCT",
        "SUMSQ",
    }
)
_SERIES_DSL_FUNCTION_NAMES = frozenset(
    {
        "rolling_mean",
        "rolling_std",
        "rolling_min",
        "rolling_max",
        "recursive_smooth",
        "divide_or_default",
    }
)
_SERIES_OPERATOR_LABELS = {
    "add": "逐行相加",
    "subtract": "逐行相减",
    "multiply": "逐行相乘",
    "divide": "逐行相除",
    "divide_or_default": "安全除法",
    "power": "逐行乘方",
    "maximum": "逐行取较大值",
    "minimum": "逐行取较小值",
    "rolling_mean": "移动平均",
    "rolling_std": "滚动标准差",
    "rolling_min": "滚动最小值",
    "rolling_max": "滚动最大值",
    "recursive_smooth": "递归平滑",
    "clip": "数值限幅",
    "where": "条件选择",
    "lag": "序列滞后",
    "difference": "序列差分",
    "cumulative_sum": "累计求和",
    "cumulative_product": "累计连乘",
    "cumulative_return": "累计收益",
    "cumulative_max": "累计最大值",
    "cumulative_min": "累计最小值",
    "drawdown_series": "回撤序列",
    "new_high_mask": "创新高判断",
}


def _assert_native_excel_formula(formula: str) -> None:
    """Fail closed if a generated formula would look like a custom function."""

    if not formula.startswith("="):
        raise ValidationError(
            "EXCEL_EXPORT_FORMULA_INVALID",
            "Excel 公式必须以等号开头。",
            field="expression",
        )
    upper = formula.upper()
    if "_XLFN." in upper or "_XLWS." in upper:
        raise ValidationError(
            "EXCEL_EXPORT_NON_NATIVE_FUNCTION",
            "Excel 导出不能依赖兼容前缀、加载项或自定义函数。",
            field="expression",
        )
    function_names = {
        match.group(1).upper() for match in _EXCEL_FUNCTION_CALL.finditer(formula)
    }
    unknown = sorted(function_names - _NATIVE_EXCEL_FUNCTIONS)
    if unknown:
        raise ValidationError(
            "EXCEL_EXPORT_NON_NATIVE_FUNCTION",
            f"Excel 公式包含非原生函数: {', '.join(unknown)}。",
            field="expression",
        )
    lowered = formula.lower()
    leaked = sorted(
        function_name
        for function_name in _SERIES_DSL_FUNCTION_NAMES
        if f"{function_name}(" in lowered
    )
    if leaked:
        raise ValidationError(
            "EXCEL_EXPORT_DSL_FUNCTION_LEAK",
            f"Excel 公式不能直接暴露 DSL 函数: {', '.join(leaked)}。",
            field="expression",
        )

_ELEMENTWISE_UNARY = frozenset(
    {
        "absolute",
        "exp",
        "log",
        "negate",
        "normal_pdf",
        "normal_ppf",
        "reciprocal",
        "sign",
        "sqrt",
        "logical_not",
    }
)
_ELEMENTWISE_BINARY = frozenset(
    {
        "add",
        "divide",
        "equal",
        "greater_equal",
        "greater_than",
        "less_equal",
        "less_than",
        "logical_and",
        "logical_or",
        "maximum",
        "minimum",
        "multiply",
        "not_equal",
        "power",
        "subtract",
    }
)
_MASKED_REDUCTIONS = frozenset(
    {
        "max_where",
        "mean_where",
        "median_where",
        "min_where",
        "quantile_where",
        "std_where",
        "sum_where",
        "variance_where",
    }
)
_CUMULATIVE_OPERATORS = frozenset(
    {
        "cumulative_max",
        "cumulative_min",
        "cumulative_product",
        "cumulative_sum",
        "cumulative_return",
    }
)
_SCALAR_REDUCTIONS = frozenset(
    {
        "argmax",
        "argmin",
        "count_true",
        "dot",
        "excess_kurtosis",
        "first",
        "last",
        "length",
        "linear_intercept",
        "linear_r_squared",
        "linear_slope",
        "max_consecutive_true",
        "max_value",
        "mean",
        "mean_absolute_deviation",
        "median",
        "min_value",
        "product",
        "quantile",
        "regression_standard_error",
        "root_mean_square",
        "skewness",
        "std",
        "sum",
        "variance",
        "correlation",
        "covariance",
        "total_return",
        "annualized_return",
    }
)
_SEQUENCE_OPERATORS = frozenset({"difference", "lag"})
_PATH_OPERATORS = frozenset({"drawdown_series", "new_high_mask"})
_ROLLING_OPERATORS = frozenset(
    {"rolling_mean", "rolling_std", "rolling_min", "rolling_max"}
)
_RECURSIVE_OPERATORS = frozenset({"recursive_smooth"})
_SPECIAL_ELEMENTWISE = frozenset(
    {"clip", "where", "active_returns", "divide_or_default"}
)

# Every operator currently exposed by IndicatorStudio in the single-product domain,
# plus three frozen typed compatibility operators used by historical definitions.
EXCEL_SINGLE_PRODUCT_OPERATOR_IDS = frozenset(
    _ELEMENTWISE_UNARY
    | _ELEMENTWISE_BINARY
    | _MASKED_REDUCTIONS
    | _CUMULATIVE_OPERATORS
    | _SCALAR_REDUCTIONS
    | _SEQUENCE_OPERATORS
    | _PATH_OPERATORS
    | _ROLLING_OPERATORS
    | _RECURSIVE_OPERATORS
    | _SPECIAL_ELEMENTWISE
    | {
        "max_consecutive_true",
    }
)

_MASKED_AGGREGATE_FUNCTION = {
    "mean_where": 1,
    "max_where": 4,
    "min_where": 5,
    "std_where": 7,
    "sum_where": 9,
    "variance_where": 10,
    "median_where": 12,
    "quantile_where": 16,
}

_A1_REFERENCE_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_!'])"
    r"(\$?[A-Z]{1,3}\$?\d+(?::\$?[A-Z]{1,3}\$?\d+)?)"
    r"(?![A-Za-z0-9_])"
)


@dataclass(frozen=True)
class NodePlacement:
    """One typed DAG node's output range in a product worksheet."""

    node_id: int
    kind: str
    is_mask: bool
    range_name: str
    title_row: int
    metadata_row: int
    header_row: int
    data_start_row: int
    data_start_col: int
    rows: int
    dates: tuple[datetime, ...]
    helper_name: str | None = None
    helper_start_col: int | None = None
    helper_rows: int = 0
    helper_kind: str | None = None

    @property
    def is_scalar(self) -> bool:
        return self.kind == "scalar"

    def cell(self, index: int = 0, *, absolute: bool = True) -> str:
        row = self.data_start_row if self.is_scalar else self.data_start_row + index
        return xl_rowcol_to_cell(row, self.data_start_col, absolute, absolute)

    def range_a1(self, *, absolute: bool = True) -> str:
        if absolute:
            return xl_range_abs(
                self.data_start_row,
                self.data_start_col,
                self.data_start_row + self.rows - 1,
                self.data_start_col,
            )
        start = xl_rowcol_to_cell(
            self.data_start_row,
            self.data_start_col,
            False,
            False,
        )
        end = xl_rowcol_to_cell(
            self.data_start_row + self.rows - 1,
            self.data_start_col,
            False,
            False,
        )
        return start if start == end else f"{start}:{end}"

    def segment_a1(
        self,
        start: int,
        end_inclusive: int,
        *,
        absolute: bool = True,
    ) -> str:
        if absolute:
            return xl_range_abs(
                self.data_start_row + start,
                self.data_start_col,
                self.data_start_row + end_inclusive,
                self.data_start_col,
            )
        start_cell = xl_rowcol_to_cell(
            self.data_start_row + start,
            self.data_start_col,
            False,
            False,
        )
        end_cell = xl_rowcol_to_cell(
            self.data_start_row + end_inclusive,
            self.data_start_col,
            False,
            False,
        )
        return start_cell if start_cell == end_cell else f"{start_cell}:{end_cell}"

    def helper_cell(self, index: int, *, absolute: bool = True) -> str:
        if self.helper_start_col is None:
            raise RuntimeError("helper range is not configured")
        return xl_rowcol_to_cell(
            self.data_start_row + index,
            self.helper_start_col,
            absolute,
            absolute,
        )

    def helper_range_a1(self, *, absolute: bool = True) -> str:
        if self.helper_start_col is None or self.helper_rows <= 0:
            raise RuntimeError("helper range is not configured")
        if absolute:
            return xl_range_abs(
                self.data_start_row,
                self.helper_start_col,
                self.data_start_row + self.helper_rows - 1,
                self.helper_start_col,
            )
        start = xl_rowcol_to_cell(
            self.data_start_row,
            self.helper_start_col,
            False,
            False,
        )
        end = xl_rowcol_to_cell(
            self.data_start_row + self.helper_rows - 1,
            self.helper_start_col,
            False,
            False,
        )
        return start if start == end else f"{start}:{end}"


@dataclass(frozen=True)
class FormulaFormats:
    section: Any
    metadata: Any
    header: Any
    text: Any
    formula_text: Any
    number: Any
    integer: Any
    boolean: Any
    date: Any
    warning: Any


class SingleProductExcelFormulaCompiler:
    """Plan and write a typed DAG as Excel formulas for one product.

    IndicatorStudio currently exposes only scalar and one-dimensional inputs in
    ``single_product`` context. Matrix-only operators are filtered out by the UI,
    so this compiler deliberately fails closed if a matrix reaches this path.
    """

    def __init__(
        self,
        *,
        plan: TypedExpressionPlan,
        context: Mapping[str, Any],
        dates_by_variable: Mapping[str, Sequence[Any]],
        sheet_name: str,
        prefix: str,
        first_block_row: int = 14,
    ) -> None:
        self.plan = plan
        self.context = context
        self.sheet_name = sheet_name
        self.prefix = self._safe_defined_name(prefix)
        self.dates_by_variable = {
            name: tuple(self._excel_datetime(value) for value in values)
            for name, values in dates_by_variable.items()
        }
        self.node_by_id = {node.node_id: node for node in plan.nodes}
        self.placements: dict[int, NodePlacement] = {}
        self.root_id = int(plan.root_id)
        self.transparent_series_export = False
        self.operator_registry = get_typed_operator_registry(plan.operator_registry_version)
        self._build_placements(first_block_row)

    @staticmethod
    def _safe_defined_name(value: str) -> str:
        normalized = re.sub(r"[^A-Za-z0-9_.]", "_", value)
        if not normalized or normalized[0].isdigit():
            normalized = f"N_{normalized}"
        return normalized[:120]

    @staticmethod
    def _excel_datetime(value: Any) -> datetime:
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is not None:
            timestamp = timestamp.tz_localize(None)
        return timestamp.to_pydatetime()

    @staticmethod
    def _constant_value(node: TypedDagNode) -> float:
        try:
            value = float(node.label)
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                "EXCEL_EXPORT_CONSTANT_INVALID",
                f"DAG 常量节点 {node.node_id} 无法转换为有限数值。",
                field="expression",
            ) from exc
        if not math.isfinite(value):
            raise ValidationError(
                "EXCEL_EXPORT_CONSTANT_INVALID",
                f"DAG 常量节点 {node.node_id} 不是有限数值。",
                field="expression",
            )
        return value

    @staticmethod
    def _excel_number_literal(value: float) -> str:
        if float(value).is_integer():
            return str(int(value))
        return format(float(value), ".15g")

    def _constant_input(
        self,
        node: TypedDagNode,
        index: int,
        default: float,
    ) -> float:
        if len(node.inputs) <= index:
            return float(default)
        input_node = self.node_by_id[int(node.inputs[index])]
        if input_node.kind != "constant":
            raise ValidationError(
                "EXCEL_EXPORT_LITERAL_REQUIRED",
                f"算子 {node.operator_id} 的配置参数必须是公式中的有限常量。",
                field="expression",
            )
        return self._constant_value(input_node)

    def _input_placements(self, node: TypedDagNode) -> list[NodePlacement]:
        return [self.placements[int(node_id)] for node_id in node.inputs]

    def _integer_constant_input(self, node: TypedDagNode, index: int, default: int) -> int:
        if len(node.inputs) <= index:
            return default
        input_node = self.node_by_id[int(node.inputs[index])]
        if input_node.kind != "constant":
            raise ValidationError(
                "EXCEL_EXPORT_LITERAL_REQUIRED",
                f"算子 {node.operator_id} 的长度参数必须是公式中的整数常量。",
                field="expression",
            )
        value = self._constant_value(input_node)
        if not value.is_integer():
            raise ValidationError(
                "EXCEL_EXPORT_LITERAL_REQUIRED",
                f"算子 {node.operator_id} 的长度参数必须是整数。",
                field="expression",
            )
        return int(value)

    def _node_rows(self, node: TypedDagNode) -> int:
        if node.kind == "variable":
            if node.label not in self.context:
                raise ValidationError(
                    "EXCEL_EXPORT_INPUT_MISSING",
                    f"Excel 导出缺少公式直接入参 {node.label}。",
                    field="expression",
                )
            value = np.asarray(self.context[node.label])
            if value.ndim == 0:
                return 1
            if value.ndim != 1:
                raise ValidationError(
                    "EXCEL_EXPORT_MATRIX_UNSUPPORTED",
                    "当前指标中心的 Excel 导出只支持单产品标量和一维序列。",
                    field="expression",
                )
            if value.size <= 0:
                raise ValidationError(
                    "EXCEL_EXPORT_INPUT_EMPTY",
                    f"公式直接入参 {node.label} 为空。",
                    field="expression",
                )
            return int(value.size)
        if node.kind == "constant":
            return 1

        output_type = node.inferred_type
        if output_type.kind == "matrix":
            raise ValidationError(
                "EXCEL_EXPORT_MATRIX_UNSUPPORTED",
                "当前指标中心的 Excel 导出只支持单产品标量和一维序列。",
                field="expression",
            )
        if output_type.is_scalar:
            return 1

        inputs = self._input_placements(node)
        operator_id = str(node.operator_id or "")
        if operator_id in _SEQUENCE_OPERATORS:
            periods = self._integer_constant_input(
                node,
                1,
                0 if operator_id == "lag" else 1,
            )
            rows = inputs[0].rows - periods
            if rows <= 0:
                raise ValidationError(
                    "INSUFFICIENT_SAMPLE",
                    f"{operator_id} 的 periods 必须小于观察值数量。",
                    field="expression",
                )
            return rows

        non_scalar_rows = [item.rows for item in inputs if not item.is_scalar]
        if not non_scalar_rows:
            raise ValidationError(
                "EXCEL_EXPORT_SHAPE_UNRESOLVED",
                f"无法确定算子 {operator_id} 的 Excel 输出行数。",
                field="expression",
            )
        if len(set(non_scalar_rows)) != 1:
            raise ValidationError(
                "EXCEL_EXPORT_SHAPE_MISMATCH",
                f"算子 {operator_id} 的非标量输入长度不一致。",
                field="expression",
            )
        return non_scalar_rows[0]

    def _helper_contract(self, node: TypedDagNode) -> tuple[str | None, int]:
        operator_id = str(node.operator_id or "")
        inputs = self._input_placements(node) if node.inputs else []
        if operator_id in _MASKED_REDUCTIONS:
            return "masked_values", inputs[0].rows
        if operator_id == "max_consecutive_true":
            return "streak", inputs[0].rows
        if operator_id in {
            "linear_slope",
            "linear_intercept",
            "linear_r_squared",
            "regression_standard_error",
        } and len(inputs) == 1:
            return "regression_index", inputs[0].rows
        if operator_id in {"total_return", "annualized_return"}:
            return "growth_factors", inputs[0].rows
        if operator_id == "recursive_smooth":
            # Column B remains the public result and is blank when the current
            # input is missing. Column C carries the recursive state so the next
            # valid row can continue from the previous finite value without a
            # hidden LOOKUP trick or a custom function.
            return "recursive_state", inputs[0].rows
        return None, 0

    def _derived_dates(self, node: TypedDagNode, rows: int) -> tuple[datetime, ...]:
        if node.kind == "variable":
            dates = self.dates_by_variable.get(node.label, ())
            return dates if len(dates) == rows else ()
        if node.kind == "constant" or node.inferred_type.is_scalar:
            return ()
        inputs = self._input_placements(node)
        operator_id = str(node.operator_id or "")
        dated_input = next((item for item in inputs if len(item.dates) == item.rows), None)
        if dated_input is None:
            return ()
        if operator_id == "difference":
            periods = self._integer_constant_input(node, 1, 1)
            dates = dated_input.dates[periods:]
        elif operator_id == "lag":
            periods = self._integer_constant_input(node, 1, 0)
            dates = dated_input.dates if periods == 0 else dated_input.dates[:-periods]
        else:
            dates = dated_input.dates
        return tuple(dates) if len(dates) == rows else ()

    def _build_placements(self, first_block_row: int) -> None:
        current_row = first_block_row
        for node in self.plan.nodes:
            operator_id = str(node.operator_id or "")
            if operator_id and operator_id not in EXCEL_SINGLE_PRODUCT_OPERATOR_IDS:
                raise ValidationError(
                    "EXCEL_EXPORT_OPERATOR_UNSUPPORTED",
                    f"当前 Excel 公式注册表暂不支持算子 {operator_id}。",
                    field="expression",
                )
            rows = self._node_rows(node)
            helper_kind, helper_rows = self._helper_contract(node)
            block_rows = max(1, rows, helper_rows)
            if current_row + block_rows + 3 >= 1_048_576:
                raise ValidationError(
                    "EXCEL_EXPORT_ROW_LIMIT_EXCEEDED",
                    "Excel 工作表行数超过上限，请缩短计算周期或简化公式。",
                )
            if node.kind == "variable":
                suffix = f"VAR_{self._safe_defined_name(node.label)}"
            else:
                suffix = f"NODE_{node.node_id:04d}"
            placement = NodePlacement(
                node_id=node.node_id,
                kind=node.inferred_type.kind,
                is_mask=bool(node.inferred_type.is_mask),
                range_name=f"{self.prefix}_{suffix}",
                title_row=current_row,
                metadata_row=current_row + 1,
                header_row=current_row + 2,
                data_start_row=current_row + 3,
                data_start_col=1,
                rows=rows,
                dates=(),
                helper_name=(
                    f"{self.prefix}_HELPER_{node.node_id:04d}"
                    if helper_kind
                    else None
                ),
                helper_start_col=2 if helper_kind else None,
                helper_rows=helper_rows,
                helper_kind=helper_kind,
            )
            self.placements[node.node_id] = placement
            dates = self._derived_dates(node, rows)
            self.placements[node.node_id] = NodePlacement(
                **{**placement.__dict__, "dates": dates}
            )
            current_row = placement.data_start_row + block_rows + 2

    @property
    def root(self) -> NodePlacement:
        return self.placements[self.root_id]

    @property
    def result_name(self) -> str:
        return f"{self.prefix}_RESULT"

    def result_formula(self, *, sheet_qualified: bool = False) -> str:
        """Return the visible final Excel formula using explicit A1 references."""

        root_node = self.node_by_id[self.root_id]
        if root_node.kind in {"variable", "constant"}:
            formula = f"={self.root.cell(absolute=False)}"
        else:
            formula = self.formula_for_node(root_node)
        if not sheet_qualified:
            return formula
        quoted_sheet = self.sheet_name.replace("'", "''")
        return _A1_REFERENCE_PATTERN.sub(
            lambda match: f"'{quoted_sheet}'!{match.group(1)}",
            formula,
        )

    @property
    def estimated_formula_cells(self) -> int:
        total = 0
        for node in self.plan.nodes:
            if node.kind not in {"variable", "constant"}:
                placement = self.placements[node.node_id]
                total += placement.rows + placement.helper_rows
        return total

    def define_names(self, workbook: Any) -> None:
        quoted_sheet = self.sheet_name.replace("'", "''")
        for placement in self.placements.values():
            workbook.define_name(
                placement.range_name,
                f"='{quoted_sheet}'!{placement.range_a1()}",
            )
            if placement.helper_name:
                workbook.define_name(
                    placement.helper_name,
                    f"='{quoted_sheet}'!{placement.helper_range_a1()}",
                )
        workbook.define_name(
            self.result_name,
            f"='{quoted_sheet}'!{self.root.range_a1()}",
        )

    def _element_ref(self, placement: NodePlacement, index: int) -> str:
        node = self.node_by_id[placement.node_id]
        if self.transparent_series_export and node.kind == "constant":
            return self._excel_number_literal(self._constant_value(node))
        return placement.cell(0 if placement.is_scalar else index, absolute=False)

    def _range_ref(self, placement: NodePlacement) -> str:
        node = self.node_by_id[placement.node_id]
        if self.transparent_series_export and node.kind == "constant":
            return self._excel_number_literal(self._constant_value(node))
        return placement.range_a1(absolute=False)

    def _transparent_elementwise_formula(
        self,
        node: TypedDagNode,
        inputs: list[NodePlacement],
        index: int,
    ) -> str:
        """Return readable row formulas for a time-series workbook.

        Formula-generated empty strings are intentionally treated as missing.
        Every arithmetic operator checks its numeric inputs first so an empty
        upstream value can never be coerced to zero by Excel.
        """

        operator_id = str(node.operator_id or "")
        refs = [self._element_ref(item, index) for item in inputs]

        def numeric_guard(arguments: Sequence[str]) -> str:
            checks = ",".join(f"NOT(ISNUMBER({argument}))" for argument in arguments)
            return f"OR({checks})" if len(arguments) > 1 else checks

        unary_expression = {
            "absolute": f"ABS({refs[0]})",
            "exp": f"EXP({refs[0]})",
            "negate": f"-({refs[0]})",
            "sign": f"SIGN({refs[0]})",
        }
        if operator_id in unary_expression:
            return (
                f'=IF({numeric_guard(refs[:1])},"",'
                f"IFERROR({unary_expression[operator_id]},\"\"))"
            )
        if operator_id == "log":
            return (
                f'=IF(OR({numeric_guard(refs[:1])},{refs[0]}<=0),"",'
                f"LN({refs[0]}))"
            )
        if operator_id == "reciprocal":
            return (
                f'=IF(OR({numeric_guard(refs[:1])},ABS({refs[0]})<1E-12),"",'
                f"1/({refs[0]}))"
            )
        if operator_id == "sqrt":
            return (
                f'=IF(OR({numeric_guard(refs[:1])},{refs[0]}<0),"",'
                f"SQRT({refs[0]}))"
            )
        if operator_id == "normal_pdf":
            return (
                f'=IF({numeric_guard(refs[:1])},"",'
                f"NORM.S.DIST({refs[0]},FALSE))"
            )
        if operator_id == "normal_ppf":
            return (
                f'=IF(OR({numeric_guard(refs[:1])},{refs[0]}<=0,{refs[0]}>=1),"",'
                f"NORM.S.INV({refs[0]}))"
            )
        if operator_id == "logical_not":
            return f'=IF(ISLOGICAL({refs[0]}),NOT({refs[0]}),"")'

        arithmetic = {
            "add": f"({refs[0]})+({refs[1]})",
            "subtract": f"({refs[0]})-({refs[1]})",
            "active_returns": f"({refs[0]})-({refs[1]})",
            "multiply": f"({refs[0]})*({refs[1]})",
        }
        if operator_id in arithmetic:
            return (
                f'=IF({numeric_guard(refs[:2])},"",'
                f"{arithmetic[operator_id]})"
            )
        if operator_id == "divide":
            return (
                f'=IF(OR({numeric_guard(refs[:2])},ABS({refs[1]})<1E-12),"",'
                f"({refs[0]})/({refs[1]}))"
            )
        if operator_id == "divide_or_default":
            return (
                f'=IF({numeric_guard(refs[:2])},"",'
                f"IF(ABS({refs[1]})<1E-12,{refs[2]},"
                f"({refs[0]})/({refs[1]})))"
            )
        if operator_id == "power":
            return (
                f'=IF({numeric_guard(refs[:2])},"",'
                f"IFERROR(POWER({refs[0]},{refs[1]}),\"\"))"
            )
        if operator_id == "maximum":
            return (
                f'=IF({numeric_guard(refs[:2])},"",MAX({refs[0]},{refs[1]}))'
            )
        if operator_id == "minimum":
            return (
                f'=IF({numeric_guard(refs[:2])},"",MIN({refs[0]},{refs[1]}))'
            )

        comparisons = {
            "equal": "=",
            "not_equal": "<>",
            "greater_than": ">",
            "greater_equal": ">=",
            "less_than": "<",
            "less_equal": "<=",
        }
        if operator_id in comparisons:
            return (
                f'=IF({numeric_guard(refs[:2])},"",'
                f"({refs[0]}){comparisons[operator_id]}({refs[1]}))"
            )
        if operator_id in {"logical_and", "logical_or"}:
            function = "AND" if operator_id == "logical_and" else "OR"
            return (
                f'=IF(AND(ISLOGICAL({refs[0]}),ISLOGICAL({refs[1]})),'
                f'{function}({refs[0]},{refs[1]}),"")'
            )
        if operator_id == "clip":
            return (
                f'=IF(OR({numeric_guard(refs[:3])},{refs[1]}>{refs[2]}),"",'
                f"MIN(MAX({refs[0]},{refs[1]}),{refs[2]}))"
            )
        if operator_id == "where":
            return (
                f'=IF(ISLOGICAL({refs[0]}),IF({refs[0]},{refs[1]},{refs[2]}),"")'
            )
        raise ValidationError(
            "EXCEL_EXPORT_OPERATOR_UNSUPPORTED",
            f"算子 {operator_id} 没有透明的原生 Excel 逐行公式实现。",
            field="expression",
        )

    def _elementwise_formula(
        self,
        node: TypedDagNode,
        inputs: list[NodePlacement],
        index: int,
    ) -> str:
        if self.transparent_series_export:
            return self._transparent_elementwise_formula(node, inputs, index)
        operator_id = str(node.operator_id or "")
        refs = [self._element_ref(item, index) for item in inputs]
        if operator_id == "absolute":
            return f"=ABS({refs[0]})"
        if operator_id == "exp":
            return f"=IFERROR(EXP({refs[0]}),NA())"
        if operator_id == "log":
            return f"=IF({refs[0]}<=0,NA(),LN({refs[0]}))"
        if operator_id == "negate":
            return f"=-({refs[0]})"
        if operator_id == "reciprocal":
            return f"=IF(ABS({refs[0]})<1E-12,NA(),1/({refs[0]}))"
        if operator_id == "sign":
            return f"=SIGN({refs[0]})"
        if operator_id == "sqrt":
            return f"=IF({refs[0]}<0,NA(),SQRT({refs[0]}))"
        if operator_id == "normal_pdf":
            return f"=NORM.S.DIST({refs[0]},FALSE)"
        if operator_id == "normal_ppf":
            return (
                f"=IF(OR({refs[0]}<=0,{refs[0]}>=1),NA(),"
                f"NORM.S.INV({refs[0]}))"
            )
        if operator_id == "logical_not":
            return f"=NOT({refs[0]})"

        if operator_id == "add":
            return f"=({refs[0]})+({refs[1]})"
        if operator_id == "subtract" or operator_id == "active_returns":
            return f"=({refs[0]})-({refs[1]})"
        if operator_id == "multiply":
            return f"=({refs[0]})*({refs[1]})"
        if operator_id == "divide":
            return (
                f"=IF(ABS({refs[1]})<1E-12,NA(),"
                f"({refs[0]})/({refs[1]}))"
            )
        if operator_id == "divide_or_default":
            return (
                f"=IF(OR(NOT(ISNUMBER({refs[0]})),NOT(ISNUMBER({refs[1]}))),"
                f"NA(),IF(ABS({refs[1]})<1E-12,{refs[2]},"
                f"({refs[0]})/({refs[1]})))"
            )
        if operator_id == "power":
            return f"=IFERROR(POWER({refs[0]},{refs[1]}),NA())"
        if operator_id == "maximum":
            return f"=MAX({refs[0]},{refs[1]})"
        if operator_id == "minimum":
            return f"=MIN({refs[0]},{refs[1]})"
        comparisons = {
            "equal": "=",
            "not_equal": "<>",
            "greater_than": ">",
            "greater_equal": ">=",
            "less_than": "<",
            "less_equal": "<=",
        }
        if operator_id in comparisons:
            return f"=({refs[0]}){comparisons[operator_id]}({refs[1]})"
        if operator_id == "logical_and":
            return f"=AND({refs[0]},{refs[1]})"
        if operator_id == "logical_or":
            return f"=OR({refs[0]},{refs[1]})"
        if operator_id == "clip":
            return (
                f"=IF({refs[1]}>{refs[2]},NA(),"
                f"MIN(MAX({refs[0]},{refs[1]}),{refs[2]}))"
            )
        if operator_id == "where":
            return f"=IF({refs[0]},{refs[1]},{refs[2]})"
        raise ValidationError(
            "EXCEL_EXPORT_OPERATOR_UNSUPPORTED",
            f"算子 {operator_id} 没有逐元素 Excel 公式实现。",
            field="expression",
        )

    @staticmethod
    def _rolling_segment_ref(
        values: NodePlacement,
        window: int,
        index: int,
    ) -> str:
        """Return an explicit trailing A1 range for a fixed formula window."""

        start_index = max(0, index - window + 1)
        return values.segment_a1(start_index, index, absolute=False)

    def _rolling_formula(
        self,
        node: TypedDagNode,
        index: int,
    ) -> str:
        """Translate rolling DSL nodes to familiar native Excel functions."""

        operator_id = str(node.operator_id or "")
        inputs = self._input_placements(node)
        values = inputs[0]
        window = self._integer_constant_input(node, 1, 1)
        segment = self._rolling_segment_ref(values, window, index)
        missing = '""' if self.transparent_series_export else "NA()"

        if operator_id == "rolling_std":
            ddof = self._integer_constant_input(node, 2, 0)
            minimum = self._integer_constant_input(node, 3, window)
            if ddof == 0:
                calculation = f"STDEVP({segment})"
            elif ddof == 1:
                calculation = f"STDEV({segment})"
            else:
                calculation = (
                    f"SQRT(DEVSQ({segment})/(COUNT({segment})-{ddof}))"
                )
            return (
                f"=IF(OR(COUNT({segment})<{minimum},"
                f"COUNT({segment})<={ddof}),{missing},"
                f"IFERROR({calculation},{missing}))"
            )

        minimum = self._integer_constant_input(node, 2, window)
        calculation = {
            "rolling_mean": f"AVERAGE({segment})",
            "rolling_max": f"MAX({segment})",
            "rolling_min": f"MIN({segment})",
        }[operator_id]
        return (
            f"=IF(COUNT({segment})<{minimum},{missing},"
            f"IFERROR({calculation},{missing}))"
        )

    def _recursive_formula(
        self,
        node: TypedDagNode,
        placement: NodePlacement,
        index: int,
    ) -> str:
        inputs = self._input_placements(node)
        current = self._element_ref(inputs[0], index)
        if placement.helper_start_col is None:
            raise RuntimeError("recursive-state helper column is missing")
        state = placement.helper_cell(index, absolute=False)
        missing = '""' if self.transparent_series_export else "NA()"
        return f"=IF(ISNUMBER({current}),{state},{missing})"

    def _array_formula(
        self,
        node: TypedDagNode,
        placement: NodePlacement,
        index: int,
    ) -> str:
        operator_id = str(node.operator_id or "")
        inputs = self._input_placements(node)
        if operator_id in _ELEMENTWISE_UNARY | _ELEMENTWISE_BINARY | _SPECIAL_ELEMENTWISE | {
            "logical_and",
            "logical_or",
        }:
            return self._elementwise_formula(node, inputs, index)
        if operator_id == "lag":
            return f"={self._element_ref(inputs[0], index)}"
        if operator_id == "difference":
            periods = self._integer_constant_input(node, 1, 1)
            current = self._element_ref(inputs[0], index + periods)
            previous = self._element_ref(inputs[0], index)
            return f"=({current})-({previous})"
        if operator_id in _ROLLING_OPERATORS:
            return self._rolling_formula(node, index)
        if operator_id in _RECURSIVE_OPERATORS:
            return self._recursive_formula(node, placement, index)
        if operator_id in _CUMULATIVE_OPERATORS:
            current = self._element_ref(inputs[0], index)
            if operator_id == "cumulative_return":
                return (
                    f"=({current})"
                    if index == 0
                    else f"=(1+{placement.cell(index - 1, absolute=False)})*(1+{current})-1"
                )
            if index == 0:
                return f"={current}"
            previous = placement.cell(index - 1, absolute=False)
            if operator_id == "cumulative_sum":
                return f"=({previous})+({current})"
            if operator_id == "cumulative_product":
                return f"=({previous})*({current})"
            if operator_id == "cumulative_max":
                return f"=MAX({previous},{current})"
            if operator_id == "cumulative_min":
                return f"=MIN({previous},{current})"
        if operator_id == "drawdown_series":
            current = self._element_ref(inputs[0], index)
            running = inputs[0].segment_a1(0, index, absolute=False)
            return f"=({current})/MAX({running})-1"
        if operator_id == "new_high_mask":
            current = self._element_ref(inputs[0], index)
            if index == 0:
                return "=TRUE"
            previous = inputs[0].segment_a1(0, index - 1, absolute=False)
            return f"=({current})>MAX({previous})"
        raise ValidationError(
            "EXCEL_EXPORT_OPERATOR_UNSUPPORTED",
            f"算子 {operator_id} 没有一维序列 Excel 公式实现。",
            field="expression",
        )

    def _masked_formula(
        self,
        node: TypedDagNode,
        placement: NodePlacement,
    ) -> str:
        operator_id = str(node.operator_id or "")
        inputs = self._input_placements(node)
        mask_range = self._range_ref(inputs[1])
        if placement.helper_start_col is None:
            raise RuntimeError("masked reduction helper range is missing")
        helper = placement.helper_range_a1(absolute=False)
        count = f"COUNTIF({mask_range},TRUE)"
        minimum = 2 if operator_id in {"std_where", "variance_where"} else 1
        aggregate_number = _MASKED_AGGREGATE_FUNCTION[operator_id]
        if operator_id == "quantile_where":
            probability = self._range_ref(inputs[2])
            aggregate = f"AGGREGATE({aggregate_number},6,{helper},{probability})"
        else:
            aggregate = f"AGGREGATE({aggregate_number},6,{helper})"
        return f"=IF({count}<{minimum},NA(),{aggregate})"

    def _linear_ranges(
        self,
        node: TypedDagNode,
        placement: NodePlacement,
    ) -> tuple[str, str]:
        inputs = self._input_placements(node)
        if len(inputs) == 1:
            if placement.helper_start_col is None:
                raise RuntimeError("regression index helper range is missing")
            return placement.helper_range_a1(absolute=False), self._range_ref(inputs[0])
        return self._range_ref(inputs[0]), self._range_ref(inputs[1])

    def _scalar_formula(self, node: TypedDagNode, placement: NodePlacement) -> str:
        operator_id = str(node.operator_id or "")
        inputs = self._input_placements(node)
        ranges = [self._range_ref(item) for item in inputs]
        if operator_id in _ELEMENTWISE_UNARY | _ELEMENTWISE_BINARY | _SPECIAL_ELEMENTWISE:
            return self._elementwise_formula(node, inputs, 0)
        if operator_id == "dot":
            return f"=IFERROR(SUMPRODUCT({ranges[0]},{ranges[1]}),NA())"
        if operator_id == "count_true":
            return f"=COUNTIF({ranges[0]},TRUE)"
        if operator_id == "max_consecutive_true":
            if placement.helper_start_col is None:
                raise RuntimeError("streak helper range is missing")
            return f"=MAX({placement.helper_range_a1(absolute=False)})"
        if operator_id in _MASKED_REDUCTIONS:
            return self._masked_formula(node, placement)
        if operator_id == "first":
            return f"=INDEX({ranges[0]},1)"
        if operator_id == "last":
            return f"=INDEX({ranges[0]},ROWS({ranges[0]}))"
        if operator_id == "length":
            return f"=ROWS({ranges[0]})"
        if operator_id == "max_value":
            return f"=MAX({ranges[0]})"
        if operator_id == "min_value":
            return f"=MIN({ranges[0]})"
        if operator_id == "mean":
            return f"=AVERAGE({ranges[0]})"
        if operator_id == "median":
            return f"=MEDIAN({ranges[0]})"
        if operator_id == "product":
            return f"=PRODUCT({ranges[0]})"
        if operator_id == "sum":
            return f"=SUM({ranges[0]})"
        if operator_id in {"variance", "std"}:
            ddof = ranges[1] if len(ranges) > 1 else "1"
            values = ranges[0]
            variance = f"DEVSQ({values})/(COUNT({values})-({ddof}))"
            result = f"SQRT({variance})" if operator_id == "std" else variance
            return f"=IF(COUNT({values})<=({ddof}),NA(),{result})"
        if operator_id == "argmax":
            values = ranges[0]
            return f"=MATCH(MAX({values}),{values},0)-1"
        if operator_id == "argmin":
            values = ranges[0]
            return f"=MATCH(MIN({values}),{values},0)-1"
        if operator_id == "quantile":
            values = ranges[0]
            probability = ranges[1]
            return (
                f"=IF(OR({probability}<=0,{probability}>=1),NA(),"
                f"PERCENTILE.INC({values},{probability}))"
            )
        if operator_id == "skewness":
            values = ranges[0]
            return f"=IF(COUNT({values})<3,NA(),SKEW({values}))"
        if operator_id == "excess_kurtosis":
            values = ranges[0]
            return f"=IF(COUNT({values})<4,NA(),KURT({values}))"
        if operator_id == "mean_absolute_deviation":
            return f"=AVEDEV({ranges[0]})"
        if operator_id == "root_mean_square":
            values = ranges[0]
            return f"=SQRT(SUMSQ({values})/COUNT({values}))"
        if operator_id in {"covariance", "correlation"}:
            if len(inputs) != 2:
                raise ValidationError(
                    "EXCEL_EXPORT_MATRIX_UNSUPPORTED",
                    f"单产品 Excel 导出只支持 {operator_id} 的双序列签名。",
                    field="expression",
                )
            function = "COVARIANCE.S" if operator_id == "covariance" else "CORREL"
            return f"=IFERROR({function}({ranges[0]},{ranges[1]}),NA())"
        if operator_id in {
            "linear_slope",
            "linear_intercept",
            "linear_r_squared",
            "regression_standard_error",
        }:
            x_values, y_values = self._linear_ranges(node, placement)
            function = {
                "linear_slope": "SLOPE",
                "linear_intercept": "INTERCEPT",
                "linear_r_squared": "RSQ",
                "regression_standard_error": "STEYX",
            }[operator_id]
            minimum = 3 if operator_id == "regression_standard_error" else 2
            return (
                f"=IF(COUNT({y_values})<{minimum},NA(),"
                f"IFERROR({function}({y_values},{x_values}),NA()))"
            )
        if operator_id == "total_return":
            if placement.helper_start_col is None:
                raise RuntimeError("total-return helper range is missing")
            return f"=PRODUCT({placement.helper_range_a1(absolute=False)})-1"
        if operator_id == "annualized_return":
            if placement.helper_start_col is None:
                raise RuntimeError("annualized-return helper range is missing")
            values = ranges[0]
            periods_per_year = ranges[1]
            helper = placement.helper_range_a1(absolute=False)
            return (
                f"=IF(COUNT({values})<1,NA(),"
                f"POWER(PRODUCT({helper}),"
                f"{periods_per_year}/COUNT({values}))-1)"
            )
        raise ValidationError(
            "EXCEL_EXPORT_OPERATOR_UNSUPPORTED",
            f"算子 {operator_id} 没有标量 Excel 公式实现。",
            field="expression",
        )

    def formula_for_node(self, node: TypedDagNode, index: int = 0) -> str:
        placement = self.placements[node.node_id]
        if node.kind in {"variable", "constant"}:
            raise ValueError("input nodes do not have generated Excel formulas")
        if placement.is_scalar:
            return self._scalar_formula(node, placement)
        return self._array_formula(node, placement, index)

    def _write_helper(
        self,
        worksheet: Any,
        node: TypedDagNode,
        placement: NodePlacement,
        formats: FormulaFormats,
    ) -> None:
        if not placement.helper_kind or placement.helper_start_col is None:
            return
        inputs = self._input_placements(node)
        for index in range(placement.helper_rows):
            row = placement.data_start_row + index
            formula: str | None = None
            value_format = formats.number
            if placement.helper_kind == "masked_values":
                value_ref = self._element_ref(inputs[0], index)
                mask_ref = self._element_ref(inputs[1], index)
                formula = f"=IF({mask_ref},{value_ref},NA())"
            elif placement.helper_kind == "streak":
                mask_ref = self._element_ref(inputs[0], index)
                formula = (
                    f"=IF({mask_ref},1,0)"
                    if index == 0
                    else f"=IF({mask_ref},{placement.helper_cell(index - 1, absolute=False)}+1,0)"
                )
                value_format = formats.integer
            elif placement.helper_kind == "regression_index":
                worksheet.write_number(
                    row,
                    placement.helper_start_col,
                    index,
                    formats.integer,
                )
                continue
            elif placement.helper_kind == "growth_factors":
                value_ref = self._element_ref(inputs[0], index)
                formula = f"=1+({value_ref})"
            elif placement.helper_kind == "recursive_state":
                current = self._element_ref(inputs[0], index)
                periods = self._excel_number_literal(
                    self._constant_input(node, 1, 1)
                )
                initial = self._excel_number_literal(
                    self._constant_input(node, 2, 0)
                )
                previous = (
                    initial
                    if index == 0
                    else placement.helper_cell(index - 1, absolute=False)
                )
                formula = (
                    f"=IF(ISNUMBER({current}),"
                    f"((({periods})-1)*({previous})+({current}))/({periods}),"
                    f"{previous})"
                )
            if formula is None:
                continue
            _assert_native_excel_formula(formula)
            worksheet.write_formula(
                row,
                placement.helper_start_col,
                formula,
                value_format,
            )

    @staticmethod
    def _input_value_format(
        placement: NodePlacement,
        formats: FormulaFormats,
    ) -> Any:
        if placement.is_mask:
            return formats.boolean
        return formats.number

    def write_nodes(
        self,
        worksheet: Any,
        *,
        formats: FormulaFormats,
        backend_value: float | None,
        cached_values_by_node: Mapping[int, Sequence[Any]] | None = None,
    ) -> None:
        for node in self.plan.nodes:
            placement = self.placements[node.node_id]
            if node.kind == "variable":
                variable = get_variable(node.label)
                title = variable.label if variable is not None else node.label
                description = variable.description if variable is not None else "公式直接入参"
                section_text = f"直接入参 · {title}（{node.label}）"
                metadata = (
                    f"{description}｜实际传入 {placement.rows} 个值｜"
                    f"类型 {node.inferred_type}"
                )
            elif node.kind == "constant":
                if self.transparent_series_export:
                    literal = self._excel_number_literal(self._constant_value(node))
                    section_text = f"固定常量 · {literal}"
                    metadata = "该数值已写入指标公式，不是运行时参数。"
                else:
                    section_text = f"常量节点 {node.node_id}"
                    metadata = f"DSL 常量：{node.formula_fragment}"
            else:
                operator_id = str(node.operator_id or node.label)
                spec = self.operator_registry.get(operator_id)
                description = spec.description if spec is not None else "Excel 公式计算步骤"
                if self.transparent_series_export:
                    operation_label = _SERIES_OPERATOR_LABELS.get(
                        operator_id,
                        "原生 Excel 计算",
                    )
                    section_text = f"计算步骤 {node.node_id} · {operation_label}"
                    metadata = (
                        f"{description}｜以下单元格使用原生 Excel 函数和普通四则运算｜"
                        f"输出类型 {node.inferred_type}"
                    )
                else:
                    section_text = f"计算步骤 {node.node_id} · {operator_id}"
                    metadata = (
                        f"{description}｜DSL：{node.formula_fragment}｜"
                        f"输出类型 {node.inferred_type}"
                    )
            worksheet.merge_range(
                placement.title_row,
                0,
                placement.title_row,
                3,
                section_text,
                formats.section,
            )
            worksheet.merge_range(
                placement.metadata_row,
                0,
                placement.metadata_row,
                3,
                metadata,
                formats.metadata,
            )

            if node.kind == "variable":
                worksheet.write(placement.header_row, 0, "日期 / 序号", formats.header)
                worksheet.write(placement.header_row, 1, "直接入参值", formats.header)
                value = np.asarray(self.context[node.label])
                values = [float(value)] if value.ndim == 0 else value.tolist()
                value_format = self._input_value_format(placement, formats)
                for index, raw_value in enumerate(values):
                    row = placement.data_start_row + index
                    if placement.dates:
                        worksheet.write_datetime(row, 0, placement.dates[index], formats.date)
                    else:
                        worksheet.write_number(row, 0, index + 1, formats.integer)
                    if isinstance(raw_value, (bool, np.bool_)):
                        worksheet.write_boolean(row, 1, bool(raw_value), value_format)
                    elif math.isfinite(float(raw_value)):
                        worksheet.write_number(row, 1, float(raw_value), value_format)
                    elif self.transparent_series_export:
                        # Blank input cells are the clearest Excel representation
                        # of missing observations. COUNT/AVERAGE/STDEV/MAX/MIN
                        # skip them while a genuine numeric zero remains intact.
                        worksheet.write_blank(row, 1, None, formats.warning)
                    else:
                        worksheet.write_formula(row, 1, "=NA()", formats.warning)
                continue

            if node.kind == "constant":
                worksheet.write(placement.header_row, 0, "常量", formats.header)
                worksheet.write(placement.header_row, 1, "值", formats.header)
                worksheet.write(
                    placement.data_start_row,
                    0,
                    node.formula_fragment,
                    formats.text,
                )
                worksheet.write_number(
                    placement.data_start_row,
                    1,
                    self._constant_value(node),
                    formats.number,
                )
                continue

            if placement.is_scalar:
                worksheet.write(placement.header_row, 0, "Excel 公式", formats.header)
                worksheet.write(placement.header_row, 1, "计算结果", formats.header)
                if placement.helper_kind == "masked_values":
                    worksheet.write(placement.header_row, 2, "Mask 选中值", formats.header)
                elif placement.helper_kind == "streak":
                    worksheet.write(placement.header_row, 2, "连续 TRUE 计数", formats.header)
                elif placement.helper_kind == "regression_index":
                    worksheet.write(placement.header_row, 2, "回归自变量 0..N-1", formats.header)
                elif placement.helper_kind == "growth_factors":
                    worksheet.write(placement.header_row, 2, "逐期增长因子 1+r", formats.header)
                formula = self.formula_for_node(node)
                _assert_native_excel_formula(formula)
                worksheet.write(
                    placement.data_start_row,
                    0,
                    formula,
                    formats.formula_text,
                )
                cached_value: Any = 0
                if node.node_id == self.root_id and backend_value is not None:
                    cached_value = float(backend_value)
                worksheet.write_formula(
                    placement.data_start_row,
                    1,
                    formula,
                    formats.number,
                    cached_value,
                )
                self._write_helper(worksheet, node, placement, formats)
                continue

            worksheet.write(placement.header_row, 0, "日期 / 序号", formats.header)
            worksheet.write(placement.header_row, 1, "Excel 公式结果", formats.header)
            if placement.helper_kind == "recursive_state":
                worksheet.write(
                    placement.header_row,
                    2,
                    "递归状态（沿用上一有效值）",
                    formats.header,
                )
            if self.transparent_series_export:
                worksheet.write(
                    placement.header_row,
                    3,
                    "原生 Excel 公式（可复制）",
                    formats.header,
                )
            for index in range(placement.rows):
                row = placement.data_start_row + index
                if placement.dates:
                    worksheet.write_datetime(row, 0, placement.dates[index], formats.date)
                else:
                    worksheet.write_number(row, 0, index + 1, formats.integer)
                formula = self.formula_for_node(node, index)
                _assert_native_excel_formula(formula)
                value_format = formats.boolean if placement.is_mask else formats.number
                cached_value: Any = 0
                cached_values = (
                    cached_values_by_node.get(node.node_id)
                    if cached_values_by_node is not None
                    else None
                )
                if cached_values is not None and index < len(cached_values):
                    raw_cached = cached_values[index]
                    if isinstance(raw_cached, (bool, np.bool_)):
                        cached_value = bool(raw_cached)
                    elif raw_cached is not None and math.isfinite(float(raw_cached)):
                        cached_value = float(raw_cached)
                    else:
                        cached_value = (
                            "" if self.transparent_series_export else "#N/A"
                        )
                worksheet.write_formula(
                    row,
                    1,
                    formula,
                    value_format,
                    cached_value,
                )
                if self.transparent_series_export:
                    worksheet.write(row, 3, formula, formats.formula_text)
            self._write_helper(worksheet, node, placement, formats)


class SeriesBundleExcelFormulaCompiler(SingleProductExcelFormulaCompiler):
    """Write one shared multi-root time-series DAG as ordinary Excel formulas."""

    def __init__(
        self,
        *,
        plan: TypedSeriesBundlePlan,
        context: Mapping[str, Any],
        dates_by_variable: Mapping[str, Sequence[Any]],
        sheet_name: str,
        prefix: str,
        first_block_row: int = 18,
    ) -> None:
        self.plan = plan
        self.context = context
        self.sheet_name = sheet_name
        self.prefix = self._safe_defined_name(prefix)
        self.dates_by_variable = {
            name: tuple(self._excel_datetime(value) for value in values)
            for name, values in dates_by_variable.items()
        }
        self.node_by_id = {node.node_id: node for node in plan.nodes}
        self.placements = {}
        self.transparent_series_export = True
        self.root_ids = {
            str(channel_id): int(root_id)
            for channel_id, root_id in plan.roots.items()
        }
        self.root_id = next(iter(self.root_ids.values()))
        self.operator_registry = get_typed_operator_registry(
            plan.operator_registry_version
        )
        self._build_placements(first_block_row)

    def channel_placement(self, channel_id: str) -> NodePlacement:
        try:
            return self.placements[self.root_ids[channel_id]]
        except KeyError as exc:
            raise ValidationError(
                "EXCEL_EXPORT_CHANNEL_NOT_FOUND",
                f"Excel 导出未找到时序通道 {channel_id}。",
                field="series_outputs",
            ) from exc

    def channel_formula(
        self,
        channel_id: str,
        index: int,
        *,
        sheet_qualified: bool = False,
    ) -> str:
        placement = self.channel_placement(channel_id)
        node = self.node_by_id[self.root_ids[channel_id]]
        if node.kind == "variable":
            formula = f"={placement.cell(index, absolute=False)}"
        elif node.kind == "constant":
            formula = f"={placement.cell(absolute=False)}"
        else:
            formula = self.formula_for_node(node, index)
        if not sheet_qualified:
            return formula
        quoted_sheet = self.sheet_name.replace("'", "''")
        return _A1_REFERENCE_PATTERN.sub(
            lambda match: f"'{quoted_sheet}'!{match.group(1)}",
            formula,
        )

    def define_names(self, workbook: Any) -> None:
        """Time-series workbooks use visible A1 references only.

        Named ranges such as ``S01_NODE_0004`` are valid Excel, but they look
        like private functions or black-box variables to end users. Keeping the
        method as an explicit no-op preserves the compiler interface while
        ensuring the downloaded workbook has no hidden calculation names.
        """

        del workbook


__all__ = [
    "EXCEL_FORMULA_REGISTRY_VERSION",
    "EXCEL_SINGLE_PRODUCT_OPERATOR_IDS",
    "FormulaFormats",
    "SeriesBundleExcelFormulaCompiler",
    "SingleProductExcelFormulaCompiler",
]

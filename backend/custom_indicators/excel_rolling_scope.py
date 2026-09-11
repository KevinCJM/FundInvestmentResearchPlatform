"""Native Excel evidence for complete interval graphs, with bounded expansion.

Inputs are written once. Every interval references the original cells, while
its actual mathematical nodes are expanded using the existing scalar compiler.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from cal_indicators.rolling_scope import analyze_interval, dependency_nodes, outside_nodes
from cal_indicators.typed_dsl import TypedExpressionPlan
from .errors import ValidationError
from .excel_formula import SingleProductExcelFormulaCompiler, SeriesBundleExcelFormulaCompiler

MAX_SCOPE_FORMULA_CELLS = 200_000
_LOCAL_CONTEXT = frozenset({"observation_count", "window_elapsed_days", "risk_free_return_window"})


class _IntervalExcelCompiler(SingleProductExcelFormulaCompiler):
    """Reuse scalar mathematical formulas with window-local input bindings."""

    def __init__(self, *, external_placements, scalar_formulas, **kwargs):
        self.external_placements = external_placements
        self.scalar_formulas = scalar_formulas
        super().__init__(**kwargs)
        self.transparent_series_export = True

    def _build_placements(self, first_block_row):
        self.placements.update(self.external_placements)
        self._owned_plan = replace(self.plan, nodes=tuple(
            node for node in self.plan.nodes if node.node_id not in self.external_placements
        ))
        original = self.plan
        self.plan = self._owned_plan
        try:
            super()._build_placements(first_block_row)
        finally:
            self.plan = original

    @property
    def estimated_formula_cells(self):
        return sum(self.placements[node.node_id].rows + self.placements[node.node_id].helper_rows
                   for node in self._owned_plan.nodes)  # Includes local scalar formulas.

    def _input_formula(self, node):
        # A streaming workbook cannot revisit a row after it has been flushed.
        return self.scalar_formulas.get(node.node_id)

    def write_nodes(self, worksheet, *, formats, backend_value, cached_values_by_node=None):
        original = self.plan
        self.plan = self._owned_plan
        try:
            super().write_nodes(worksheet, formats=formats, backend_value=backend_value)
        finally:
            self.plan = original


class ScopedSeriesExcelFormulaCompiler(SeriesBundleExcelFormulaCompiler):
    """Series compiler with transparent per-window scalar graph expansion."""

    def __init__(self, *, plan, **kwargs):
        self.full_plan = plan
        self.scope_windows: dict[tuple[int, int], _IntervalExcelCompiler] = {}
        self.window_guards: dict[tuple[int, int], str] = {}
        self.scopes = {node.node_id: node for node in plan.nodes if node.operator_id == "rolling_apply"}
        outer = replace(plan, nodes=outside_nodes(plan.nodes, tuple(plan.roots.values())))
        super().__init__(plan=outer, **kwargs)
        self._prepare_windows()

    def _node_rows(self, node):
        if node.operator_id == "rolling_apply":
            return len(self.context["observation_dates"])
        return super()._node_rows(node)

    def _helper_contract(self, node):
        if node.operator_id == "rolling_apply":
            return None, 0
        return super()._helper_contract(node)

    def _derived_dates(self, node, rows):
        if node.operator_id == "rolling_apply":
            return next((values for values in self.dates_by_variable.values() if len(values) == rows), ())
        return super()._derived_dates(node, rows)

    def _array_formula(self, node, placement, index):
        if node.operator_id != "rolling_apply":
            return super()._array_formula(node, placement, index)
        key = (node.node_id, index)
        compiler = self.scope_windows.get(key)
        if compiler is None:
            return '=""'
        root = compiler.root.cell(absolute=False)
        return f'=IF({self.window_guards[key]},IFERROR({root},""),"")'

    @property
    def estimated_formula_cells(self):
        return super().estimated_formula_cells + sum(compiler.estimated_formula_cells for compiler in self.scope_windows.values())

    def _prepare_windows(self):
        size = len(self.context["observation_dates"])
        by_id = {node.node_id: node for node in self.full_plan.nodes}
        estimate = 0
        declarations = []
        for scope in self.scopes.values():
            capability = analyze_interval(self.full_plan.nodes, scope.inputs[0], self.operator_registry)
            width = self._integer_constant_input(scope, 1, 1)
            if not 1 <= width <= 5000:
                raise ValidationError("INVALID_ROLLING_WINDOW", "滚动窗口必须为 1 至 5000 的整数。")
            minimum = self._integer_constant_input(scope, 4, width) if len(scope.inputs) == 5 else width
            if not 1 <= minimum <= width:
                raise ValidationError("INVALID_MIN_PERIODS", "最少有效观察数必须是1至窗口观察数的整数。")
            first = minimum if capability.needs_preceding_observation else minimum - 1
            body = dependency_nodes(self.full_plan.nodes, scope.inputs[0])
            estimate += max(0, size - first) * (width + 1) * max(1, len(body))
            if estimate > MAX_SCOPE_FORMULA_CELLS:
                raise ValidationError("EXCEL_EXPORT_TOO_LARGE", "完整区间计算图逐窗口展开超过 Excel 证据预算，请缩短导出区间或窗口。")
            declarations.append((scope, capability, width, first, body))
        cursor = max((p.data_start_row + max(p.rows, p.helper_rows) + 3 for p in self.placements.values()), default=12)
        for scope, capability, width, first, body in declarations:
            root = by_id[scope.inputs[0]]
            body_plan = TypedExpressionPlan(
                expression=root.formula_fragment, python_expression=root.formula_fragment,
                expression_hash=self.full_plan.expression_hash, dsl_version=self.full_plan.dsl_version,
                compiler_version=self.full_plan.compiler_version, operator_registry_version=self.full_plan.operator_registry_version,
                nodes=body, root_id=root.node_id, output_type=root.inferred_type, output_contract="scalar",
                context_requirements={node.label: node.inferred_type for node in capability.variables}, estimated_cost={},
            )
            for right in range(first, size):
                start, end = max(1 if capability.needs_preceding_observation else 0, right - width + 1), right + 1
                context, external, local_formulas, guards = self._window_bindings(scope, capability, start, end, width)
                compiler = _IntervalExcelCompiler(plan=body_plan, context=context, dates_by_variable={},
                    sheet_name=self.sheet_name, prefix=f"{self.prefix}_W{scope.node_id}_{right}",
                    first_block_row=cursor, external_placements=external, scalar_formulas=local_formulas)
                self.scope_windows[(scope.node_id, right)] = compiler
                self.window_guards[(scope.node_id, right)] = f"AND({','.join(guards)})"
                cursor = max(p.data_start_row + max(p.rows, p.helper_rows) + 3 for p in compiler.placements.values())
                if cursor >= 1_048_570:
                    raise ValidationError("EXCEL_EXPORT_ROW_LIMIT_EXCEEDED", "区间计算步骤超过工作表行数上限，请缩短导出区间。")

    def _window_bindings(self, scope, capability, start, end, width):
        context, external, formulas, guards = {}, {}, {}, []
        partial = len(scope.inputs) == 5
        minimum = self._integer_constant_input(scope, 4, width) if partial else width
        joint_ranges = []
        dates = self.placements[scope.inputs[2]]
        annual = self.placements[scope.inputs[3]].cell(absolute=False)
        left = start - 1 if capability.needs_preceding_observation else start
        elapsed = f"({dates.cell(end - 1, absolute=False)}-{dates.cell(left, absolute=False)})"
        for node in capability.variables:
            original = self.placements[node.node_id]
            if node.inferred_type.rank:
                offset = start - 1 if capability.has_returns and node.inferred_type.shape == ("L",) else start
                # NumPy input views here are only evidence metadata; numerical
                # outputs have already been computed by the prepared NJIT plan.
                context[node.label] = self.context[node.label][offset:end]
                placement = replace(original, data_start_row=original.data_start_row + offset,
                    rows=end - offset, dates=(), range_name=f"{original.range_name}_{offset}_{end}")
                external[node.node_id] = placement
                if partial:
                    joint_ranges.append(f"--ISNUMBER({original.segment_a1(start, end - 1, absolute=False)})")
                    if offset < start:
                        guards.append(f"ISNUMBER({original.cell(offset, absolute=False)})")
                else:
                    guards.append(f"COUNT({placement.range_a1(absolute=False)})={end - offset}")
            elif node.label in _LOCAL_CONTEXT:
                context[node.label] = 0.0  # Formula, not this placeholder, is the exported value.
                if node.label == "observation_count":
                    formulas[node.node_id] = f"={end - start if capability.has_returns else max(0, end - start - 1)}"
                    if partial and capability.has_returns:
                        returns = min((item for item in capability.variables if item.label in {"returns", "log_returns"}),
                                      key=lambda item: item.label != "returns")
                        source = self.placements[returns.node_id].segment_a1(start, end - 1, absolute=False)
                        formulas[node.node_id] = f"=COUNT({source})"
                elif node.label == "window_elapsed_days":
                    formulas[node.node_id] = f"={elapsed}"
                else:
                    formulas[node.node_id] = f"=MAX(0,1+{annual})^({elapsed}/365)-1"
            else:
                context[node.label] = self.context[node.label]
                external[node.node_id] = original
        if partial:
            guards.append(f"SUMPRODUCT({','.join(joint_ranges)})>={minimum}")
        return context, external, formulas, guards

    def write_nodes(self, worksheet, *, formats, backend_value, cached_values_by_node=None):
        super().write_nodes(worksheet, formats=formats, backend_value=backend_value,
                            cached_values_by_node=cached_values_by_node)
        for (scope_id, index), compiler in self.scope_windows.items():
            cached = (cached_values_by_node or {}).get(scope_id)
            value = cached[index] if cached is not None and index < len(cached) else None
            compiler.write_nodes(worksheet, formats=formats, backend_value=value)

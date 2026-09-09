"""Transparent native Excel lowering for a single multi-result drawdown scan."""
from __future__ import annotations

from xlsxwriter.utility import xl_rowcol_to_cell

from cal_indicators.multi_output import DRAWDOWN_PORTS


def drawdown_formulas(source, placement):
    """Build worksheet formulas, not numeric results; zero-based positions."""
    def cell(index, column):
        return xl_rowcol_to_cell(placement.data_start_row + index, column, True, True)

    rows = []
    for index in range(source.rows):
        value = cell(index, 3)
        peak = cell(index, 4)
        depth = cell(index, 6)
        if index == 0:
            formulas = [f"={source.cell(index)}", f"={value}", "=0", "=0", "=0", "=0", "=0", "=0", "=0"]
        else:
            previous_peak, previous_peak_pos = cell(index-1, 4), cell(index-1, 5)
            previous_worst = cell(index-1, 7)
            previous_worst_peak, previous_trough, previous_recovery = cell(index-1, 8), cell(index-1, 9), cell(index-1, 10)
            formulas = [
                f"={source.cell(index)}",
                f"=MAX({previous_peak},{value})",
                f"=IF({value}>={previous_peak},{index},{previous_peak_pos})",
                f"=1-{value}/{peak}",
                f"=MAX({previous_worst},{depth})",
                f"=IF({depth}>{previous_worst},{previous_peak_pos},{previous_worst_peak})",
                f"=IF({depth}>{previous_worst},{index},{previous_trough})",
                f"=IF({depth}>{previous_worst},0,IF(AND({previous_worst}>0,{previous_recovery}=0,{value}>=INDEX({source.range_a1()},{previous_worst_peak}+1)),{index},{previous_recovery}))",
                f"=MAX({cell(index-1,11)},IF({value}<{previous_peak},{index}-{previous_peak_pos},IF({index}>{previous_peak_pos}+1,{index}-{previous_peak_pos},0)))",
            ]
        rows.append(formulas)
    last = source.rows-1
    worst, peak_pos, trough, recovered, longest = [cell(last, column) for column in (7, 8, 9, 10, 11)]
    results = [
        worst,
        f"IF({worst}=0,0,{trough}-{peak_pos})",
        f"IF({worst}=0,0,IF({recovered}=0,NA(),{recovered}-{trough}))",
        longest,
    ]
    valid = f"AND(COUNT({source.range_a1()})={source.rows},MIN({source.range_a1()})>0)"
    return rows, [f"=IF({valid},{formula},NA())" for formula in results]


def write_drawdown_record(compiler, worksheet, node, placement, formats):
    from .excel_formula import _assert_native_excel_formula
    source = compiler.placements[node.inputs[0]]
    rows, results = drawdown_formulas(source, placement)
    worksheet.write(placement.header_row, 0, "命名结果", formats.header)
    worksheet.write(placement.header_row, 1, "Excel 公式结果", formats.header)
    headers = ["净值", "历史峰值", "峰值位置", "当前回撤", "最大回撤", "最大回撤峰值位置", "最大回撤谷底位置", "恢复位置（0为未恢复）", "最长水下期数"]
    worksheet.set_column(4, 11, 19)
    for column, label in enumerate(headers, start=3):
        worksheet.write(placement.header_row, column, label, formats.header)
    # Constant-memory worksheets require every row to be completed in order.
    for index in range(max(len(rows), len(results))):
        row = placement.data_start_row + index
        if index < len(results):
            worksheet.write(row, 0, DRAWDOWN_PORTS[index].label, formats.text)
            _assert_native_excel_formula(results[index])
            worksheet.write_formula(row, 1, results[index], formats.number, "")
        if index < len(rows):
            for column, formula in enumerate(rows[index], start=3):
                _assert_native_excel_formula(formula)
                worksheet.write_formula(row, column, formula, formats.number, "")

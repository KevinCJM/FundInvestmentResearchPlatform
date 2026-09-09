"""Native Excel cells for shared computational state and scalar projections."""
from __future__ import annotations

STATE_OPERATORS = frozenset({"last_drawdown_interval", "linear_fit"})
PROJECTIONS = {"interval_start": 0, "interval_trough": 1, "interval_recovery": 2,
               "fit_slope": 0, "fit_intercept": 1, "fit_residual_sum_squares": 2,
               "fit_total_sum_squares": 3, "fit_observation_count": 4}
SCALAR_OPERATORS = frozenset({*PROJECTIONS, "value_at", "days_between", "require_positive", "require_nonnegative"})


def state_formulas(compiler, node, placement):
    inputs = compiler._input_placements(node)
    ranges = [compiler._range_ref(item) for item in inputs]
    if node.operator_id == "last_drawdown_interval":
        source = ranges[0]
        peak, trough, recovery, status = (placement.cell(index) for index in range(4))
        positions = f"ROW({source})-ROW(INDEX({source},1))"
        # Largest matching index implements the same exact last-trough tie rule.
        return (
            f"=IF({status}=1,LOOKUP(2,1/(({source}=0)*({positions}<={trough})),{positions}),NA())",
            f"=IF({status}=1,LOOKUP(2,1/({source}=MIN({source})),{positions}),NA())",
            f"=IF({status}=1,IFERROR(AGGREGATE(15,6,{positions}/(({source}=0)*({positions}>{trough})),1),NA()),NA())",
            f"=IF(OR(COUNT({source})<>ROWS({source}),INDEX({source},1)<>0,MAX({source})>0,MIN({source})<-1),-1,IF(MIN({source})<0,1,0))",
        )
    if node.operator_id == "linear_fit":
        x, y = compiler._linear_ranges(node, placement)
        slope, intercept, sse, sst, count = (placement.cell(index) for index in range(5))
        invalid = f"OR(COUNT({x})<>ROWS({x}),COUNT({y})<>ROWS({y}),ROWS({x})<>ROWS({y}),COUNT({y})<2,DEVSQ({x})<=0)"
        return (
            f"=IF({invalid},NA(),SLOPE({y},{x}))",
            f"=AVERAGE({y})-{slope}*AVERAGE({x})",
            f"=SUMPRODUCT(({y}-({intercept}+{slope}*{x}))*({y}-({intercept}+{slope}*{x})))",
            f"=IF({invalid},NA(),DEVSQ({y}))",
            f"=IF({invalid},NA(),COUNT({y}))",
        )
    raise ValueError("Unknown computation state")


def scalar_formula(compiler, node):
    inputs = compiler._input_placements(node)
    refs = [compiler._range_ref(item) for item in inputs]
    name = node.operator_id
    if name in PROJECTIONS:
        return f"={inputs[0].cell(PROJECTIONS[name])}"
    if name == "value_at":
        values, position = refs
        return f"=IFERROR(IF(OR({position}<0,{position}<>INT({position}),{position}>=ROWS({values})),NA(),INDEX({values},{position}+1)),NA())"
    if name == "days_between":
        start, end = refs
        return f"=IFERROR(IF(OR({end}<{start},{start}<>INT({start}),{end}<>INT({end})),NA(),{end}-{start}),NA())"
    value = refs[0]
    comparison = ">0" if name == "require_positive" else ">=0"
    return f"=IFERROR(IF({value}{comparison},{value},NA()),NA())"


def write_state(compiler, worksheet, node, placement, formats):
    from .excel_formula import _assert_native_excel_formula
    worksheet.write(placement.header_row, 0, "计算状态字段", formats.header)
    worksheet.write(placement.header_row, 1, "共享计算值", formats.header)
    if placement.helper_kind == "regression_index":
        worksheet.write(placement.header_row, 2, "观察序号 0..N-1", formats.header)
    fields = node.inferred_type.fields
    formulas = state_formulas(compiler, node, placement)
    # Complete each row before advancing: the production writer streams rows.
    for index in range(max(len(fields), placement.helper_rows)):
        row = placement.data_start_row + index
        if index < len(fields):
            formula = formulas[index]
            _assert_native_excel_formula(formula)
            worksheet.write(row, 0, fields[index][0], formats.text)
            worksheet.write_formula(row, 1, formula, formats.number)
        if placement.helper_kind == "regression_index" and index < placement.helper_rows:
            worksheet.write_number(row, placement.helper_start_col, index, formats.integer)

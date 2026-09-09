"""Excel evidence for a named scalar bundle, using the existing formula compiler."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from xlsxwriter.utility import xl_rowcol_to_cell

from cal_indicators.typed_dsl import TypedExpressionPlan
from .errors import ValidationError
from .excel_export import ExcelExportArtifact, ExcelTargetEvidence, build_indicator_excel_workbook
from .scalar_outputs import project_scalar_output
from .series_provider import load_product_variable_series_batch, market_data_generation, select_variable_window


def export_scalar_bundle(
    owner: Any, definition: dict[str, Any], targets: list[dict[str, Any]],
    period: str, as_of: str | None,
) -> ExcelExportArtifact:
    """Freeze each result's exact input window and preserve unavailable siblings."""
    generation = market_data_generation(owner.market_data_dir)
    computed = owner.scalar_service.evaluate_definition(definition, targets, period, as_of)
    results = {
        (result["target"]["kind"], result["target"]["product_id"], output["output_id"]): output
        for result in computed for output in result["outputs"]
    }
    sources: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}
    windows: dict[tuple[str, str, tuple[str, ...]], Any] = {}
    output_entries = []
    for output in definition["scalar_outputs"]:
        projected = project_scalar_output(definition, output["id"])
        runtime = owner._compile_runtime(projected, period)
        dependencies = owner._physical_dependency_signature(runtime.plan.context_requirements)
        evidence = []
        for target in targets:
            kind, product_id = target["kind"], target["product_id"]
            result = results[(kind, product_id, output["id"])]
            direct_context, dates_by_variable = {}, {}
            if result["value"] is not None:
                source_key = (kind, dependencies)
                if source_key not in sources:
                    sources[source_key] = load_product_variable_series_batch(
                        kind, [item["product_id"] for item in targets if item["kind"] == kind],
                        dependencies, owner.market_data_dir, as_of,
                    )
                window_key = (kind, product_id, dependencies)
                if window_key not in windows:
                    windows[window_key] = select_variable_window(
                        sources[source_key][product_id], period, as_of, max_observations=5000,
                    )
                window = windows[window_key]
                if owner._window_payload(window) != result["window"]:
                    raise ValidationError("EXCEL_EXPORT_DATA_CHANGED", "导出窗口与已计算结果不一致，请重新运行。")
                elapsed = float((window.frame.iloc[-1]["date"] - window.frame.iloc[0]["date"]).days)
                context = {**window.context, **owner._risk_free_context(definition, elapsed)}
                direct_context = {name: context[name] for name in runtime.compiled_plan.context_names}
                dates_by_variable = owner._excel_dates_by_variable(window, direct_context)
            evidence.append(ExcelTargetEvidence(
                target=dict(target), name=result["target"]["name"], result=result,
                context=direct_context, dates_by_variable=dates_by_variable,
            ))
        output_entries.append((projected, runtime.plan, evidence))
    if generation != market_data_generation(owner.market_data_dir):
        raise ValidationError("EXCEL_EXPORT_DATA_CHANGED", "生成 Excel 期间市场数据发生变化，请重新导出。")
    return build_indicator_excel_workbook(
        output_dir=owner.workspace_data_dir / ".indicator_exports", definition=definition,
        plan=output_entries[0][1], targets=[], period=period, as_of=as_of,
        data_generation=generation, bundle_outputs=output_entries,
    )


def write_bundle_summary(
    workbook: Any, definition: Mapping[str, Any],
    entries: Sequence[tuple[Mapping[str, Any], TypedExpressionPlan, ExcelTargetEvidence]],
    compilers: Sequence[Any], formats: Mapping[str, Any], period: str,
    as_of: str | None, data_generation: str,
) -> None:
    """One row per product/output, with immutable identity and native Excel formulas."""
    sheet = workbook.add_worksheet("01_结果汇总")
    sheet.hide_gridlines(2)
    sheet.freeze_panes(10, 4)
    sheet.set_column(0, 1, 19)
    sheet.set_column(2, 2, 23)
    sheet.set_column(3, 3, 31)
    sheet.set_column(4, 4, 9)
    sheet.set_column(5, 5, 29)
    sheet.set_column(6, 11, 18)
    sheet.set_column(12, 12, 55)
    sheet.merge_range(0, 0, 0, 12, f"{definition.get('name')} · 多结果计算复现", formats["title"])
    sheet.set_row(0, 30)
    for row, (label, value) in enumerate([
        ("指标编码与版本", f"{definition.get('id') or '未保存草稿'} / {definition.get('revision') or '校验版本'}"),
        ("区间与截止日", f"{period} / {as_of or '最新可用数据'}"),
        ("数据版本", data_generation),
        ("DSL / 算子版本", f"{definition.get('dsl_version')} / {definition.get('operator_registry_version')}"),
        ("复现说明", "每个结果保留独立窗口、原始单位与公式。数值未按显示单位改写；缺失结果不填零。"),
        ("校验说明", "平台值由固定签名 NJIT 计算，Excel 原生公式将在打开文件时重新计算。一致性列不是独立 Excel 引擎验收记录。"),
    ], start=2):
        sheet.write(row, 0, label, formats["label"])
        sheet.merge_range(row, 1, row, 12, value, formats["value"])
    headers = ["产品代码", "产品名称", "结果名称", "结果编码", "版本", "实际窗口", "状态",
               "平台 NJIT 结果", "Excel 公式结果", "绝对差异", "一致性", "单位 / 方向", "不可计算原因"]
    for column, label in enumerate(headers):
        sheet.write(9, column, label, formats["header"])
    number_formats: dict[tuple[str, int], Any] = {}
    direction_labels = {"neutral": "仅展示", "higher_better": "越高越好", "lower_better": "越低越好"}
    for row, ((output, _plan, evidence), compiler) in enumerate(zip(entries, compilers, strict=True), start=10):
        result = evidence.result
        window = result["window"]
        raw = result.get("value")
        style_key = (str(output.get("display_format") or "number"), int(output.get("precision", 2)))
        if style_key not in number_formats:
            decimals = "." + "0" * style_key[1] if style_key[1] else ""
            number_formats[style_key] = workbook.add_format({"num_format": f"0{decimals}{'%' if style_key[0] == 'percent' else ''}"})
        value_format = number_formats[style_key]
        values = [evidence.target["product_id"], evidence.name, output.get("output_label"), output.get("output_id"),
                  output.get("revision"), f"{window.get('start_date') or '—'} 至 {window.get('end_date') or '—'}", result["status"]]
        for column, value in enumerate(values):
            sheet.write(row, column, value, formats["value"])
        if raw is not None:
            sheet.write_number(row, 7, float(raw), value_format)
        else:
            sheet.write_blank(row, 7, None, formats["warning"])
        if compiler is not None and raw is not None:
            sheet.write_formula(row, 8, compiler.result_formula(sheet_qualified=True), value_format, float(raw))
            backend_cell, excel_cell = xl_rowcol_to_cell(row, 7), xl_rowcol_to_cell(row, 8)
            difference_cell = xl_rowcol_to_cell(row, 9)
            sheet.write_formula(row, 9, f"=ABS({excel_cell}-{backend_cell})", formats["number"], 0)
            sheet.write_formula(row, 10, f'=IF({difference_cell}<=MAX(1E-12,ABS({backend_cell})*1E-10),"一致","不一致")', formats["value"], "打开 Excel 后重算")
        else:
            sheet.write_blank(row, 8, None, formats["warning"])
            sheet.write_blank(row, 9, None, formats["warning"])
            sheet.write(row, 10, "不可比较", formats["warning"])
        sheet.write(row, 11, f"{output.get('unit') or '无单位'} / {direction_labels.get(str(output.get('direction')), '仅展示')}", formats["value"])
        sheet.write(row, 12, "；".join(str(item.get("message") or item.get("code") or "") for item in result.get("warnings", [])), formats["warning"])
    if entries:
        sheet.autofilter(9, 0, 9 + len(entries), 12)

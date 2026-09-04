"""Build downloadable Excel workbooks that reproduce indicator calculations."""

from __future__ import annotations

import math
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import xlsxwriter
from xlsxwriter.utility import xl_rowcol_to_cell

from cal_indicators.typed_dsl import TypedExpressionPlan

from .errors import ValidationError
from .excel_formula import (
    EXCEL_FORMULA_REGISTRY_VERSION,
    FormulaFormats,
    SingleProductExcelFormulaCompiler,
)


EXCEL_MEDIA_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
DEFAULT_MAX_FORMULA_CELLS = 2_000_000


@dataclass(frozen=True)
class ExcelTargetEvidence:
    """Exact direct runtime inputs and official result for one product."""

    target: dict[str, str]
    name: str
    result: dict[str, Any]
    context: Mapping[str, Any]
    dates_by_variable: Mapping[str, Sequence[Any]]


@dataclass(frozen=True)
class ExcelExportArtifact:
    path: Path
    filename: str
    media_type: str = EXCEL_MEDIA_TYPE

    def cleanup(self) -> None:
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


def _safe_filename_component(value: str, fallback: str) -> str:
    cleaned = re.sub(r"[\\/:*?\"<>|\x00-\x1f]+", "_", str(value)).strip(" ._")
    return (cleaned or fallback)[:60]


def _safe_sheet_name(index: int, product_id: str) -> str:
    product = re.sub(r"[\[\]:*?/\\]+", "_", product_id).strip() or "product"
    return f"P{index:02d}_{product}"[:31]


def _warning_text(result: Mapping[str, Any]) -> str:
    warnings = result.get("warnings") or []
    return "；".join(
        str(item.get("message") or item.get("code") or "")
        for item in warnings
        if isinstance(item, Mapping)
    )


def _write_value(
    worksheet: Any,
    row: int,
    column: int,
    value: Any,
    cell_format: Any,
) -> None:
    if value is None:
        worksheet.write_blank(row, column, None, cell_format)
        return
    if isinstance(value, bool):
        worksheet.write_boolean(row, column, value, cell_format)
        return
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        worksheet.write_number(row, column, float(value), cell_format)
        return
    worksheet.write(row, column, str(value), cell_format)


def _formats(workbook: Any) -> tuple[FormulaFormats, dict[str, Any]]:
    title = workbook.add_format(
        {
            "bold": True,
            "font_size": 16,
            "font_color": "#FFFFFF",
            "bg_color": "#111827",
            "align": "left",
            "valign": "vcenter",
        }
    )
    label = workbook.add_format(
        {
            "bold": True,
            "font_color": "#334155",
            "bg_color": "#F1F5F9",
            "border": 1,
            "border_color": "#E2E8F0",
            "valign": "top",
        }
    )
    value = workbook.add_format(
        {
            "font_color": "#0F172A",
            "border": 1,
            "border_color": "#E2E8F0",
            "valign": "top",
            "text_wrap": True,
        }
    )
    section = workbook.add_format(
        {
            "bold": True,
            "font_color": "#FFFFFF",
            "bg_color": "#7C3AED",
            "border": 1,
            "border_color": "#6D28D9",
            "valign": "vcenter",
        }
    )
    metadata = workbook.add_format(
        {
            "font_color": "#475569",
            "bg_color": "#F8FAFC",
            "border": 1,
            "border_color": "#E2E8F0",
            "text_wrap": True,
            "valign": "top",
        }
    )
    header = workbook.add_format(
        {
            "bold": True,
            "font_color": "#FFFFFF",
            "bg_color": "#475569",
            "border": 1,
            "border_color": "#334155",
            "align": "center",
            "valign": "vcenter",
        }
    )
    text = workbook.add_format(
        {
            "border": 1,
            "border_color": "#CBD5E1",
            "text_wrap": True,
            "valign": "top",
        }
    )
    formula_text = workbook.add_format(
        {
            "font_name": "Consolas",
            "font_color": "#166534",
            "bg_color": "#F0FDF4",
            "border": 1,
            "border_color": "#BBF7D0",
            "text_wrap": True,
            "valign": "top",
        }
    )
    number = workbook.add_format(
        {
            "num_format": "0.000000000000",
            "border": 1,
            "border_color": "#CBD5E1",
        }
    )
    integer = workbook.add_format(
        {
            "num_format": "0",
            "border": 1,
            "border_color": "#CBD5E1",
        }
    )
    boolean = workbook.add_format(
        {
            "align": "center",
            "border": 1,
            "border_color": "#CBD5E1",
        }
    )
    date = workbook.add_format(
        {
            "num_format": "yyyy-mm-dd",
            "border": 1,
            "border_color": "#CBD5E1",
        }
    )
    warning = workbook.add_format(
        {
            "font_color": "#9A3412",
            "bg_color": "#FFF7ED",
            "border": 1,
            "border_color": "#FED7AA",
            "text_wrap": True,
        }
    )
    success = workbook.add_format(
        {
            "font_color": "#166534",
            "bg_color": "#F0FDF4",
            "border": 1,
            "border_color": "#BBF7D0",
        }
    )
    formula_formats = FormulaFormats(
        section=section,
        metadata=metadata,
        header=header,
        text=text,
        formula_text=formula_text,
        number=number,
        integer=integer,
        boolean=boolean,
        date=date,
        warning=warning,
    )
    return formula_formats, {
        "title": title,
        "label": label,
        "value": value,
        "header": header,
        "number": number,
        "integer": integer,
        "warning": warning,
        "success": success,
    }


def _summary_sheet(
    workbook: Any,
    *,
    definition: Mapping[str, Any],
    plan: TypedExpressionPlan,
    period: str,
    as_of: str | None,
    data_generation: str,
    targets: Sequence[ExcelTargetEvidence],
    compilers: Sequence[SingleProductExcelFormulaCompiler | None],
    formats: Mapping[str, Any],
) -> None:
    worksheet = workbook.add_worksheet("01_结果汇总")
    worksheet.hide_gridlines(2)
    worksheet.freeze_panes(11, 0)
    worksheet.set_column(0, 0, 8)
    worksheet.set_column(1, 1, 11)
    worksheet.set_column(2, 3, 22)
    worksheet.set_column(4, 5, 15)
    worksheet.set_column(6, 8, 20)
    worksheet.set_column(9, 9, 14)
    worksheet.set_column(10, 10, 55)
    worksheet.merge_range(0, 0, 0, 10, "自定义指标 Excel 计算复现", formats["title"])
    worksheet.set_row(0, 28)

    metadata = [
        ("指标名称", definition.get("name")),
        ("DSL 公式", definition.get("expression")),
        ("计算周期", period),
        ("历史截止日", as_of or "最新可用数据"),
        ("DSL / 算子版本", f"{plan.dsl_version} / {plan.operator_registry_version}"),
        ("Excel 公式注册表", EXCEL_FORMULA_REGISTRY_VERSION),
        ("数据版本", data_generation),
        ("说明", "每个产品 Sheet 的直接入参值与平台本次 NJIT 计算完全一致；修改入参后 Excel 会自动重新计算。"),
    ]
    for row, (label, value) in enumerate(metadata, start=2):
        worksheet.write(row, 0, label, formats["label"])
        worksheet.merge_range(row, 1, row, 10, value, formats["value"])

    header_row = 10
    headers = [
        "编号",
        "产品类型",
        "产品代码",
        "产品名称",
        "状态",
        "实际窗口",
        "Excel 公式结果",
        "平台 NJIT 结果",
        "绝对差异",
        "一致性",
        "说明",
    ]
    for column, header in enumerate(headers):
        worksheet.write(header_row, column, header, formats["header"])

    for index, (evidence, compiler) in enumerate(zip(targets, compilers, strict=True), start=1):
        row = header_row + index
        result = evidence.result
        window = result.get("window") or {}
        worksheet.write_number(row, 0, index, formats["integer"])
        worksheet.write(row, 1, evidence.target["kind"].upper(), formats["value"])
        worksheet.write(row, 2, evidence.target["product_id"], formats["value"])
        worksheet.write(row, 3, evidence.name, formats["value"])
        worksheet.write(row, 4, result.get("status"), formats["value"])
        worksheet.write(
            row,
            5,
            f"{window.get('start_date') or '—'} 至 {window.get('end_date') or '—'}",
            formats["value"],
        )
        backend_value = result.get("value")
        if compiler is not None:
            worksheet.write_formula(
                row,
                6,
                compiler.result_formula(sheet_qualified=True),
                formats["number"],
                float(backend_value) if backend_value is not None else 0,
            )
        else:
            worksheet.write_blank(row, 6, None, formats["warning"])
        _write_value(worksheet, row, 7, backend_value, formats["number"])
        excel_cell = xl_rowcol_to_cell(row, 6)
        backend_cell = xl_rowcol_to_cell(row, 7)
        difference_cell = xl_rowcol_to_cell(row, 8)
        if compiler is not None and backend_value is not None:
            worksheet.write_formula(
                row,
                8,
                f"=IF(AND(ISNUMBER({excel_cell}),ISNUMBER({backend_cell})),ABS({excel_cell}-{backend_cell}),NA())",
                formats["number"],
                0,
            )
            worksheet.write_formula(
                row,
                9,
                f'=IF({difference_cell}<=MAX(1E-12,ABS({backend_cell})*1E-10),"一致","不一致")',
                formats["success"],
                "打开 Excel 后自动重算",
            )
        else:
            worksheet.write(row, 8, "—", formats["warning"])
            worksheet.write(row, 9, "不可比较", formats["warning"])
        worksheet.write(row, 10, _warning_text(result), formats["warning"])

    if targets:
        worksheet.autofilter(header_row, 0, header_row + len(targets), len(headers) - 1)


def _write_product_summary(
    worksheet: Any,
    *,
    evidence: ExcelTargetEvidence,
    definition: Mapping[str, Any],
    period: str,
    as_of: str | None,
    compiler: SingleProductExcelFormulaCompiler | None,
    formats: Mapping[str, Any],
) -> None:
    result = evidence.result
    window = result.get("window") or {}
    worksheet.merge_range(
        0,
        0,
        0,
        3,
        f"{evidence.name} · {definition.get('name')}",
        formats["title"],
    )
    worksheet.set_row(0, 28)
    rows: list[tuple[str, Any]] = [
        ("产品", f"{evidence.name}（{evidence.target['product_id']}）"),
        ("产品类型", evidence.target["kind"].upper()),
        ("指标", definition.get("name")),
        ("DSL 公式", definition.get("expression")),
        ("计算周期", period),
        ("历史截止日", as_of or "最新可用数据"),
        (
            "实际窗口",
            f"{window.get('start_date') or '—'} 至 {window.get('end_date') or '—'}，{window.get('observation_count') or 0} 个收益观察值",
        ),
        ("计算状态", result.get("status")),
    ]
    for row, (label, value) in enumerate(rows, start=2):
        worksheet.write(row, 0, label, formats["label"])
        worksheet.merge_range(row, 1, row, 3, value, formats["value"])

    result_row = 10
    backend_value = result.get("value")

    # In XlsxWriter constant-memory mode each row must be completed before the
    # next row is written. Keep labels and values in strict row order.
    worksheet.write(result_row, 0, "Excel 公式结果", formats["label"])
    if compiler is not None:
        worksheet.write_formula(
            result_row,
            1,
            compiler.result_formula(),
            formats["number"],
            float(backend_value) if backend_value is not None else 0,
        )
    else:
        worksheet.merge_range(
            result_row,
            1,
            result_row,
            3,
            "本产品没有形成有效运行窗口，无法生成 Excel 公式。",
            formats["warning"],
        )

    worksheet.write(result_row + 1, 0, "平台 NJIT 结果", formats["label"])
    _write_value(worksheet, result_row + 1, 1, backend_value, formats["number"])

    worksheet.write(result_row + 2, 0, "绝对差异 / 一致性", formats["label"])
    if compiler is not None and backend_value is not None:
        excel_cell = xl_rowcol_to_cell(result_row, 1)
        backend_cell = xl_rowcol_to_cell(result_row + 1, 1)
        difference_cell = xl_rowcol_to_cell(result_row + 2, 1)
        worksheet.write_formula(
            result_row + 2,
            1,
            f"=ABS({excel_cell}-{backend_cell})",
            formats["number"],
            0,
        )
        worksheet.write_formula(
            result_row + 2,
            2,
            f'=IF({difference_cell}<=MAX(1E-12,ABS({backend_cell})*1E-10),"一致","不一致")',
            formats["success"],
            "打开 Excel 后自动重算",
        )
        worksheet.write(
            result_row + 2,
            3,
            "打开 Excel 后会自动重算全部公式。",
            formats["value"],
        )
    else:
        worksheet.merge_range(
            result_row + 2,
            1,
            result_row + 2,
            3,
            _warning_text(result) or "不可计算",
            formats["warning"],
        )


def build_indicator_excel_workbook(
    *,
    output_dir: Path,
    definition: Mapping[str, Any],
    plan: TypedExpressionPlan,
    targets: Sequence[ExcelTargetEvidence],
    period: str,
    as_of: str | None,
    data_generation: str,
) -> ExcelExportArtifact:
    """Generate a formula-driven workbook from exact runtime inputs."""

    output_dir.mkdir(parents=True, exist_ok=True)
    descriptor, path_text = tempfile.mkstemp(
        prefix="indicator-excel-",
        suffix=".xlsx",
        dir=str(output_dir),
    )
    os.close(descriptor)
    path = Path(path_text)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = (
        f"指标计算_{_safe_filename_component(str(definition.get('name') or ''), '未命名指标')}"
        f"_{period}_{timestamp}.xlsx"
    )

    compilers: list[SingleProductExcelFormulaCompiler | None] = []
    sheet_names: list[str] = []
    for index, evidence in enumerate(targets, start=1):
        sheet_name = _safe_sheet_name(index, evidence.target["product_id"])
        sheet_names.append(sheet_name)
        if not evidence.context:
            compilers.append(None)
            continue
        compilers.append(
            SingleProductExcelFormulaCompiler(
                plan=plan,
                context=evidence.context,
                dates_by_variable=evidence.dates_by_variable,
                sheet_name=sheet_name,
                prefix=f"P{index:02d}",
            )
        )

    formula_cells = sum(
        compiler.estimated_formula_cells
        for compiler in compilers
        if compiler is not None
    )
    max_formula_cells = max(
        1,
        int(os.getenv("INDICATOR_EXCEL_MAX_FORMULA_CELLS", str(DEFAULT_MAX_FORMULA_CELLS))),
    )
    if formula_cells > max_formula_cells:
        path.unlink(missing_ok=True)
        raise ValidationError(
            "EXCEL_EXPORT_TOO_LARGE",
            f"工作簿预计生成 {formula_cells} 个公式单元格，超过上限 {max_formula_cells}；请减少产品、缩短周期或简化指标。",
        )

    workbook: Any | None = None
    try:
        workbook = xlsxwriter.Workbook(
            path,
            {
                "constant_memory": True,
                "strings_to_formulas": False,
                "strings_to_urls": False,
                "nan_inf_to_errors": True,
            },
        )
        workbook.set_calc_mode("auto")
        workbook.set_properties(
            {
                "title": f"{definition.get('name')} Excel 计算复现",
                "subject": "自定义指标直接入参与 Excel 公式计算全流程",
                "author": "基金量化投研平台",
                "comments": "平台正式结果由固定签名 NJIT 计算；Excel 用同一批直接入参复现公式逻辑。",
            }
        )
        formula_formats, common_formats = _formats(workbook)

        _summary_sheet(
            workbook,
            definition=definition,
            plan=plan,
            period=period,
            as_of=as_of,
            data_generation=data_generation,
            targets=targets,
            compilers=compilers,
            formats=common_formats,
        )

        for evidence, sheet_name, compiler in zip(
            targets,
            sheet_names,
            compilers,
            strict=True,
        ):
            worksheet = workbook.add_worksheet(sheet_name)
            worksheet.hide_gridlines(2)
            worksheet.freeze_panes(14, 0)
            worksheet.set_column(0, 0, 24)
            worksheet.set_column(1, 1, 24)
            worksheet.set_column(2, 2, 24)
            worksheet.set_column(3, 3, 42)
            _write_product_summary(
                worksheet,
                evidence=evidence,
                definition=definition,
                period=period,
                as_of=as_of,
                compiler=compiler,
                formats={**common_formats, "formula_text": formula_formats.formula_text},
            )
            if compiler is not None:
                compiler.write_nodes(
                    worksheet,
                    formats=formula_formats,
                    backend_value=(
                        float(evidence.result["value"])
                        if evidence.result.get("value") is not None
                        else None
                    ),
                )
            else:
                worksheet.merge_range(
                    14,
                    0,
                    16,
                    3,
                    _warning_text(evidence.result)
                    or "本次没有可供导出的直接入参数据。",
                    common_formats["warning"],
                )

        workbook.close()
        workbook = None
    except Exception:
        if workbook is not None:
            try:
                workbook.close()
            except Exception:
                pass
        path.unlink(missing_ok=True)
        raise

    return ExcelExportArtifact(path=path, filename=filename)


__all__ = [
    "EXCEL_MEDIA_TYPE",
    "ExcelExportArtifact",
    "ExcelTargetEvidence",
    "build_indicator_excel_workbook",
]

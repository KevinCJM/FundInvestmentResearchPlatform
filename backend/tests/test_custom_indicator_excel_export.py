from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cal_indicators.typed_dsl import compose_typed_expression
from custom_indicators.excel_formula import (
    EXCEL_SINGLE_PRODUCT_OPERATOR_IDS,
    SingleProductExcelFormulaCompiler,
)
from custom_indicators.service import CustomIndicatorService
from custom_indicators.typed_service import typed_product_meta
from custom_indicators.variable_registry import variable_catalog, variable_types
from services import custom_indicator_routes


def _write_etf_data(data_dir: Path) -> None:
    dates = pd.bdate_range("2025-01-02", periods=320)
    pd.DataFrame(
        [{"ts_code": "510050.SH", "code": "510050", "name": "上证50ETF"}]
    ).to_parquet(data_dir / "etf_info_df.parquet", index=False)
    nav_rows = []
    candle_rows = []
    for index, date in enumerate(dates):
        adjusted_nav = 1.0 + index * 0.001 + (index % 7) * 0.0002
        unit_nav = 1.0 + index * 0.0008
        close = 100.0 + index * 0.03 + (index % 5) * 0.02
        previous_close = close - 0.03
        nav_rows.append(
            {
                "ts_code": "510050.SH",
                "name": "上证50ETF",
                "date": date,
                "ann_date": date,
                "adj_nav": adjusted_nav,
                "unit_nav": unit_nav,
                "accum_nav": unit_nav + 0.1,
                "accum_div": index * 0.0001,
                "net_asset": 10_000_000_000.0 + index * 1_000_000.0,
                "total_netasset": 10_500_000_000.0 + index * 1_000_000.0,
            }
        )
        candle_rows.append(
            {
                "ts_code": "510050.SH",
                "date": date,
                "open": close - 0.01,
                "high": close + 0.08,
                "low": close - 0.09,
                "close": close,
                "pre_close": previous_close,
                "change": close - previous_close,
                "pct_chg": (close / previous_close - 1.0) * 100.0,
                "vol": 1_000_000.0 + index * 1_000.0,
                "amount": 100_000_000.0 + index * 10_000.0,
            }
        )
    pd.DataFrame(nav_rows).to_parquet(
        data_dir / "etf_daily_df.parquet",
        index=False,
    )
    pd.DataFrame(candle_rows).to_parquet(
        data_dir / "etf_daily_candle_df.parquet",
        index=False,
    )
    # The current variable catalogue also exposes disclosed ETF share/size.
    pd.DataFrame({"ts_code": "510050.SH", "date": dates,
                  "total_share": 1000.0 + np.arange(len(dates)),
                  "total_size": 1500.0 + np.arange(len(dates))}).to_parquet(
        data_dir / "etf_share_size_df.parquet", index=False,
    )


def _inline_draft(name: str, expression: str) -> dict[str, object]:
    return {
        "name": name,
        "description": "Excel 导出测试",
        "expression": expression,
        "unit": "",
        "display_format": "number",
        "precision": 8,
        "direction": "higher_better",
        "indicator_type": "other",
        "annual_risk_free_rate_percent": 1.5,
        "dsl_version": "2.2.0",
        "operator_registry_version": "2.2.0",
        "variable_registry_version": "2.1.0",
        "data_contract_version": "tushare-eod-v2",
        "context_schema_version": "typed-context-v2",
        "period_policy": "all_supported",
        "context_kind": "single_product",
        "output_contract": "scalar",
        "output_measure": "dimensionless",
    }


def _compiler(expression: str) -> tuple[SingleProductExcelFormulaCompiler, object]:
    plan = compose_typed_expression(
        expression,
        variable_types=variable_types("single_product"),
    )
    dates = tuple(pd.bdate_range("2026-01-02", periods=6))
    context = {
        "returns": np.asarray([0.01, -0.02, 0.03, 0.015, -0.005], dtype=np.float64),
        "log_returns": np.asarray([0.00995, -0.02020, 0.02956, 0.01489, -0.00501], dtype=np.float64),
        "adjusted_nav": np.asarray([1.0, 1.01, 0.9898, 1.019494, 1.034786, 1.029612], dtype=np.float64),
        "observation_dates": np.asarray([20455., 20458., 20459., 20460., 20461., 20462.]),
        "observation_count": 5.0,
        "window_elapsed_days": 7.0,
        "risk_free_return_window": 0.0004,
        "annual_risk_free_rate_decimal": 0.015,
        "risk_free_rate_per_observation": 0.000059,
        "periods_per_year": 252.0,
    }
    required_context = {
        name: context[name]
        for name in plan.context_requirements
    }
    dates_by_variable = {
        "returns": dates[1:],
        "log_returns": dates[1:],
        "adjusted_nav": dates,
    }
    compiler = SingleProductExcelFormulaCompiler(
        plan=plan,
        context=required_context,
        dates_by_variable=dates_by_variable,
        sheet_name="P01_TEST",
        prefix="P01",
    )
    return compiler, plan


_OPERATOR_EXPRESSIONS = {
    "finite_mask": "count_true(finite_mask(returns))",
    "absolute": "absolute(-1.0)",
    "add": "mean(add(returns, returns))",
    "clip": "mean(clip(returns, -0.5, 0.5))",
    "divide": "mean(divide(returns, 2.0))",
    "exp": "exp(1.0)",
    "log": "log(2.0)",
    "maximum": "maximum(1.0, 2.0)",
    "minimum": "minimum(1.0, 2.0)",
    "multiply": "mean(multiply(returns, 2.0))",
    "negate": "negate(1.0)",
    "power": "power(2.0, 3.0)",
    "reciprocal": "reciprocal(2.0)",
    "sign": "sign(-1.0)",
    "sqrt": "sqrt(4.0)",
    "subtract": "mean(subtract(returns, log_returns))",
    "equal": "count_true(equal(returns, returns))",
    "greater_equal": "count_true(greater_equal(returns, 0.0))",
    "greater_than": "count_true(greater_than(returns, 0.0))",
    "less_equal": "count_true(less_equal(returns, 0.0))",
    "less_than": "count_true(less_than(returns, 0.0))",
    "not_equal": "count_true(not_equal(returns, log_returns))",
    "dot": "dot(returns, returns)",
    "count_true": "count_true(greater_than(returns, 0.0))",
    "logical_and": "count_true(logical_and(greater_than(returns, 0.0), less_than(returns, 1.0)))",
    "logical_not": "count_true(logical_not(greater_than(returns, 0.0)))",
    "logical_or": "count_true(logical_or(greater_than(returns, 0.0), less_than(returns, 0.0)))",
    "max_consecutive_true": "max_consecutive_true(greater_than(returns, 0.0))",
    "max_where": "max_where(returns, greater_than(returns, 0.0))",
    "mean_where": "mean_where(returns, greater_than(returns, 0.0))",
    "median_where": "median_where(returns, greater_than(returns, 0.0))",
    "min_where": "min_where(returns, greater_than(returns, 0.0))",
    "quantile_where": "quantile_where(returns, greater_than(returns, 0.0), 0.5)",
    "std_where": "std_where(returns, greater_than(returns, 0.0))",
    "sum_where": "sum_where(returns, greater_than(returns, 0.0))",
    "variance_where": "variance_where(returns, greater_than(returns, 0.0))",
    "where": "mean(where(greater_than(returns, 0.0), returns, negate(returns)))",
    "last_drawdown_interval": "interval_start(last_drawdown_interval(drawdown_series(adjusted_nav)))",
    "interval_start": "interval_start(last_drawdown_interval(drawdown_series(adjusted_nav)))",
    "interval_trough": "interval_trough(last_drawdown_interval(drawdown_series(adjusted_nav)))",
    "interval_recovery": "interval_recovery(last_drawdown_interval(drawdown_series(adjusted_nav)))",
    "value_at": "value_at(observation_dates, 1)",
    "days_between": "days_between(value_at(observation_dates, 0), value_at(observation_dates, 1))",
    "require_positive": "require_positive(1)",
    "require_nonnegative": "require_nonnegative(0)",
    "linear_fit": "fit_slope(linear_fit(returns))",
    "fit_slope": "fit_slope(linear_fit(returns))",
    "fit_intercept": "fit_intercept(linear_fit(returns))",
    "fit_residual_sum_squares": "fit_residual_sum_squares(linear_fit(returns))",
    "fit_total_sum_squares": "fit_total_sum_squares(linear_fit(returns))",
    "fit_observation_count": "fit_observation_count(linear_fit(returns))",
    "drawdown_series": "min_value(drawdown_series(adjusted_nav))",
    "new_high_mask": "count_true(new_high_mask(adjusted_nav))",
    "cumulative_max": "last(cumulative_max(returns))",
    "cumulative_min": "last(cumulative_min(returns))",
    "cumulative_product": "last(cumulative_product(add(returns, 1.0)))",
    "cumulative_sum": "last(cumulative_sum(returns))",
    "first": "first(returns)",
    "last": "last(returns)",
    "length": "length(returns)",
    "max_value": "max_value(returns)",
    "mean": "mean(returns)",
    "min_value": "min_value(returns)",
    "product": "product(add(returns, 1.0))",
    "std": "std(returns, 1.0)",
    "sum": "sum(returns)",
    "variance": "variance(returns, 1.0)",
    "difference": "mean(difference(returns, 1.0))",
    "lag": "mean(lag(returns, 1.0))",
    "argmax": "argmax(returns)",
    "argmin": "argmin(returns)",
    "correlation": "correlation(returns, log_returns)",
    "covariance": "covariance(returns, log_returns)",
    "excess_kurtosis": "excess_kurtosis(returns)",
    "mean_absolute_deviation": "mean_absolute_deviation(returns)",
    "median": "median(returns)",
    "normal_pdf": "normal_pdf(0.0)",
    "normal_ppf": "normal_ppf(0.5)",
    "quantile": "quantile(returns, 0.5)",
    "root_mean_square": "root_mean_square(returns)",
    "skewness": "skewness(returns)",
    "rolling_window": "mean(mean(rolling_window(returns, 3.0, 2.0)))",
    "recursive_smooth": "mean(recursive_smooth(returns, 3.0, 0.0))",
    "divide_or_default": "mean(divide_or_default(returns, returns, 0.0))",
}

_SCOPED_OPERATOR_EXPRESSIONS = {
    "rolling_apply": "rolling_apply(mean(adjusted_nav), 3)",
}

_COMPAT_OPERATOR_EXPRESSIONS = {
    "rolling_mean": "mean(rolling_mean(returns, 3.0, 2.0))",
    "rolling_std": "mean(rolling_std(returns, 3.0, 0.0, 2.0))",
    "rolling_min": "mean(rolling_min(returns, 3.0, 2.0))",
    "rolling_max": "mean(rolling_max(returns, 3.0, 2.0))",
    "linear_intercept": "linear_intercept(returns)",
    "linear_r_squared": "linear_r_squared(returns)",
    "linear_slope": "linear_slope(returns)",
    "regression_standard_error": "regression_standard_error(returns)",
    "active_returns": "mean(active_returns(returns, log_returns))",
    "annualized_return": "annualized_return(returns, periods_per_year)",
    "cumulative_return": "last(cumulative_return(returns))",
    "total_return": "total_return(returns)",
}


def test_excel_registry_covers_every_current_single_product_operator() -> None:
    public = {
        item["name"]
        for item in typed_product_meta()["operators"]
        if "single_product" in item.get("domains", [])
    }
    assert public == set(_OPERATOR_EXPRESSIONS) | set(_SCOPED_OPERATOR_EXPRESSIONS)
    assert public.issubset(EXCEL_SINGLE_PRODUCT_OPERATOR_IDS)


@pytest.mark.parametrize(
    "operator_id,expression",
    sorted({**_OPERATOR_EXPRESSIONS, **_COMPAT_OPERATOR_EXPRESSIONS}.items()),
)
def test_every_current_operator_generates_an_excel_formula(
    operator_id: str,
    expression: str,
) -> None:
    compiler, plan = _compiler(expression)
    node = next(item for item in plan.nodes if item.node_id == plan.root_id) if operator_id in _COMPAT_OPERATOR_EXPRESSIONS else next(item for item in plan.nodes if item.operator_id == operator_id)
    if operator_id in _COMPAT_OPERATOR_EXPRESSIONS:
        assert operator_id not in {item.operator_id for item in plan.nodes}
    placement = compiler.placements[node.node_id]
    formula = compiler.formula_for_node(node, 0)
    assert formula.startswith("=")
    assert "P01_VAR_" not in formula
    assert "P01_NODE_" not in formula
    assert "P01_RESULT" not in formula
    assert placement.rows >= 1


@pytest.mark.parametrize("expression", _SCOPED_OPERATOR_EXPRESSIONS.values())
def test_control_scope_generates_native_excel_from_its_interval_body(expression) -> None:
    from cal_indicators.typed_dsl import compose_typed_series_bundle
    from custom_indicators.excel_rolling_scope import ScopedSeriesExcelFormulaCompiler
    plan = compose_typed_series_bundle({"value": expression}, variable_types=variable_types("single_product"))
    dates = tuple(pd.bdate_range("2026-01-02", periods=6))
    compiler = ScopedSeriesExcelFormulaCompiler(
        plan=plan, context={"adjusted_nav": np.arange(1., 7.),
                           "observation_dates": np.arange(6.) + 20000.,
                           "annual_risk_free_rate_decimal": 0.0},
        dates_by_variable={"adjusted_nav": dates}, sheet_name="S01_SCOPE", prefix="S01",
    )
    assert "IFERROR(" in compiler.channel_formula("value", 3)
    assert len(compiler.scope_windows) == 4
    assert any("AVERAGE(" in item.result_formula() for item in compiler.scope_windows.values())


def test_visible_result_formula_uses_explicit_excel_range() -> None:
    compiler, _ = _compiler("product(returns)")

    assert compiler.result_formula() == "=PRODUCT(B18:B22)"
    assert compiler.result_formula(sheet_qualified=True) == (
        "=PRODUCT('P01_TEST'!B18:B22)"
    )
    assert "P01_RESULT" not in compiler.result_formula()
    assert "P01_VAR" not in compiler.result_formula()


def test_excel_compiler_accepts_every_current_single_product_variable() -> None:
    dates = tuple(pd.bdate_range("2026-01-02", periods=4))
    for variable in variable_catalog("single_product"):
        variable_id = variable["id"]
        expression = (
            variable_id
            if variable["structural_type"] == "scalar"
            else f"mean({variable_id})"
        )
        if variable.get("semantic") == "date":
            expression = f"value_at({variable_id}, 0)"
        plan = compose_typed_expression(
            expression,
            variable_types=variable_types("single_product"),
        )
        context = {
            variable_id: (
                1.0
                if variable["structural_type"] == "scalar"
                else np.asarray([1.0, 1.1, 1.2, 1.3], dtype=np.float64)
            )
        }
        compiler = SingleProductExcelFormulaCompiler(
            plan=plan,
            context=context,
            dates_by_variable={variable_id: dates},
            sheet_name="P01_VARIABLE",
            prefix="P01",
        )
        variable_node = next(item for item in plan.nodes if item.kind == "variable")
        assert compiler.placements[variable_node.node_id].rows in {1, 4}


def test_service_exports_every_current_single_product_variable(
    tmp_path: Path,
) -> None:
    _write_etf_data(tmp_path)
    # A fixed benchmark is a separate source, not the tested product's NAV.
    dates = pd.read_parquet(tmp_path / "etf_daily_df.parquet")["date"].drop_duplicates().sort_values()
    pd.DataFrame({"ts_code": "H00300.CSI", "trade_date": dates,
                  "close": 1000.0 + np.arange(len(dates))}).to_parquet(tmp_path / "index_daily_df.parquet", index=False)
    service = CustomIndicatorService(tmp_path, tmp_path)
    service.warm_numba_plans()

    for variable in variable_catalog("single_product"):
        variable_id = variable["id"]
        expression = (
            variable_id
            if variable["structural_type"] == "scalar"
            else f"mean({variable_id})"
        )
        if variable.get("semantic") == "date":
            expression = f"value_at({variable_id}, 0)"
        draft = _inline_draft(f"导出 {variable_id}", expression)
        validation = service.validate(draft)
        assert validation["valid"] is True, variable_id
        artifact = service.export_excel(
            indicator_ids=[],
            inline_definition=draft,
            compile_token=validation["compile_token"],
            targets=[{"kind": "etf", "product_id": "510050.SH"}],
            period="1Y",
        )
        try:
            with zipfile.ZipFile(artifact.path) as workbook:
                workbook_xml = workbook.read("xl/workbook.xml").decode("utf-8")
                worksheet_xml = "\n".join(
                    workbook.read(name).decode("utf-8")
                    for name in workbook.namelist()
                    if name.startswith("xl/worksheets/sheet")
                    and name.endswith(".xml")
                )
            assert f"P01_VAR_{variable_id}" not in workbook_xml
            assert "P01_RESULT" not in workbook_xml
            assert "<f>" in worksheet_xml, variable_id
        finally:
            artifact.cleanup()


def test_every_current_builtin_indicator_exports_with_excel_formulas(
    tmp_path: Path,
) -> None:
    _write_etf_data(tmp_path)
    service = CustomIndicatorService(tmp_path, tmp_path)
    service.warm_numba_plans()
    current_builtins = [
        item
        for item in service.list_indicators(context_kind="single_product")["items"]
        if item.get("source") == "built_in"
        and item.get("dsl_version") == "2.2.0"
    ]
    assert len(current_builtins) == 34  # max drawdown now has a native multi-output definition

    for definition in current_builtins:
        artifact = service.export_excel(
            indicator_ids=[definition["id"]],
            inline_definition=None,
            targets=[{"kind": "etf", "product_id": "510050.SH"}],
            period="1Y",
        )
        try:
            assert zipfile.is_zipfile(artifact.path), definition["id"]
        finally:
            artifact.cleanup()


def test_mean_return_export_contains_exact_input_range_and_average_formula(
    tmp_path: Path,
) -> None:
    _write_etf_data(tmp_path)
    service = CustomIndicatorService(tmp_path, tmp_path)
    service.warm_numba_plans()

    artifact = service.export_excel(
        indicator_ids=["builtin-mean-return-v2"],
        inline_definition=None,
        targets=[{"kind": "etf", "product_id": "510050.SH"}],
        period="1Y",
    )
    try:
        assert artifact.path.exists()
        assert zipfile.is_zipfile(artifact.path)
        with zipfile.ZipFile(artifact.path) as workbook:
            workbook_xml = workbook.read("xl/workbook.xml").decode("utf-8")
            worksheet_xml = "\n".join(
                workbook.read(name).decode("utf-8")
                for name in workbook.namelist()
                if name.startswith("xl/worksheets/sheet") and name.endswith(".xml")
            )
        assert "P01_VAR_returns" not in workbook_xml
        assert "P01_RESULT" not in workbook_xml
        assert "<f>AVERAGE(B18:B278)</f>" in worksheet_xml
        assert "<f>AVERAGE('P01_510050.SH'!B18:B278)</f>" in worksheet_xml
        assert "<f>ABS(B11-B12)</f>" in worksheet_xml
        assert "直接入参 · 复权净值普通收益率（returns）" in worksheet_xml
        assert "Excel 公式结果" in worksheet_xml
        assert "平台 NJIT 结果" in worksheet_xml
    finally:
        artifact.cleanup()
    assert not artifact.path.exists()


def test_inline_indicator_export_uses_validation_compile_token(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _write_etf_data(tmp_path)
    service = CustomIndicatorService(tmp_path, tmp_path)
    service.warm_numba_plans()
    monkeypatch.setattr(custom_indicator_routes, "indicator_service", service)
    app = FastAPI()
    app.include_router(custom_indicator_routes.router)
    client = TestClient(app)
    draft = {
        **_inline_draft("普通收益率均值", "mean(returns)"),
        "unit": "%",
        "display_format": "percent",
        "precision": 6,
        "indicator_type": "return",
        "output_measure": "return_decimal",
    }
    validation = client.post("/api/custom-indicators/validate", json=draft)
    assert validation.status_code == 200
    assert validation.json()["valid"] is True

    response = client.post(
        "/api/custom-indicators/export-excel",
        json={
            "inline_definition": draft,
            "compile_token": validation.json()["compile_token"],
            "targets": [{"kind": "etf", "product_id": "510050.SH"}],
            "period": "1Y",
        },
    )

    assert response.status_code == 200
    with zipfile.ZipFile(io.BytesIO(response.content)) as workbook:
        worksheet_xml = "\n".join(
            workbook.read(name).decode("utf-8")
            for name in workbook.namelist()
            if name.startswith("xl/worksheets/sheet") and name.endswith(".xml")
        )
    assert "AVERAGE(B18:B278)" in worksheet_xml
    assert "P01_VAR_returns" not in worksheet_xml
    assert "P01_RESULT" not in worksheet_xml


def test_excel_export_route_returns_downloadable_xlsx(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _write_etf_data(tmp_path)
    service = CustomIndicatorService(tmp_path, tmp_path)
    service.warm_numba_plans()
    monkeypatch.setattr(custom_indicator_routes, "indicator_service", service)
    app = FastAPI()
    app.include_router(custom_indicator_routes.router)

    response = TestClient(app).post(
        "/api/custom-indicators/export-excel",
        json={
            "indicator_ids": ["builtin-mean-return-v2"],
            "targets": [{"kind": "etf", "product_id": "510050.SH"}],
            "period": "1Y",
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith(
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    assert "attachment" in response.headers["content-disposition"]
    with zipfile.ZipFile(io.BytesIO(response.content)) as workbook:
        assert "xl/workbook.xml" in workbook.namelist()

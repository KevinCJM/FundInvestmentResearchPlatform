"""API, graph, parameter, history and native-Excel interval-scope contracts."""
from __future__ import annotations

import copy
from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from custom_indicators.service import CustomIndicatorService
from custom_indicators.rolling_series import derive_rolling_series_definition
from custom_indicators.series_parameters import inspect_parameter_inputs, bind_parameter_input
from services import custom_indicator_routes
from test_custom_indicator_time_series import _write_market_data

TARGET = {"kind": "etf", "product_id": "510300.SH"}
NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


@pytest.fixture(scope="module")
def service(tmp_path_factory):
    root = tmp_path_factory.mktemp("rolling-interval-service")
    _write_market_data(root)
    item = CustomIndicatorService(root, root)
    yield item
    item.close_compute_engine()


@pytest.fixture
def client(service, monkeypatch):
    monkeypatch.setattr(custom_indicator_routes, "indicator_service", service)
    app = FastAPI()
    app.include_router(custom_indicator_routes.router)
    return TestClient(app)


@pytest.mark.parametrize("indicator_id", [
    "builtin-mean-return-v2", "builtin-return-volatility-v2", "builtin-total-return-v2",
    "builtin-maximum-drawdown-v2", "builtin-annualized-sharpe-v2", "builtin-calmar-ratio-v2",
    "builtin-historical-var-95-v2", "builtin-historical-cvar-95-v2",
])
def test_whole_existing_scalar_metric_can_be_derived_saved_and_run(service, client, indicator_id):
    source = service.get_indicator(indicator_id)
    assert source["rolling_series_compatibility"]["supported"]
    before = copy.deepcopy(source)
    response = client.post("/api/custom-indicators/derive-rolling-series", json={
        "indicator_id": indicator_id, "indicator_revision": source["revision"], "window_observations": 5,
    })
    assert response.status_code == 200, response.text
    derived = response.json()
    assert derived["validation"]["valid"]
    draft = derived["definition"]
    assert draft["rolling_source"]["transform_version"] == "3.0.0"
    assert draft["series_outputs"][0]["expression"].startswith("rolling_apply(")
    created = client.post("/api/custom-indicators", json=draft)
    assert created.status_code in (200, 201), created.text
    saved = created.json()
    result = service.evaluate_series(indicator_instances=[{"indicator_id": saved["id"]}], target=TARGET, period="1M")
    assert result["results"][0]["status"] in {"ok", "warning"}, result
    assert result["results"][0]["history_policy"] == "lookback"
    # Calmar needs five returns plus the preceding NAV point for its path.
    assert result["results"][0]["lookback_observations"] == (6 if indicator_id == 'builtin-calmar-ratio-v2' else 5)
    assert result["execution"]["python_fallback"] == 0
    assert result["execution"]["request_time_compilation"] == 0
    assert service.get_indicator(indicator_id)["expression"] == before["expression"]


def test_custom_composite_and_graph_roundtrip_uses_actual_scope(service, client):
    custom = service.create_indicator({"name": "区间自定义组合", "expression": "std(returns,1) - min_value(drawdown_series(adjusted_nav))"})
    assert custom["rolling_series_compatibility"]["supported"]
    draft = service.build_rolling_scalar_draft(custom["id"], 1, 5)["definition"]
    first = client.post("/api/custom-indicators/graph/resolve", json={
        "source_kind": "formula", "result_kind": "time_series", "context_kind": "single_product",
        "dsl_version": "2.4.0", "operator_registry_version": "2.4.0",
        "expressions": {"value": draft["expression"]},
    })
    assert first.status_code == 200, first.text
    graph = first.json()
    assert graph["valid"], graph
    ops = [node.get("operator_id") for node in graph["graph"]["nodes"]]
    assert {"rolling_apply", "std", "drawdown_series", "min_value"}.issubset(ops)
    assert not {"rolling_std", "rolling_window"}.intersection(ops)
    second = client.post("/api/custom-indicators/graph/resolve", json={
        "source_kind": "graph", "result_kind": "time_series", "context_kind": "single_product",
        "dsl_version": "2.4.0", "operator_registry_version": "2.4.0", "graph": graph["graph"],
    }).json()
    assert second["valid"], second
    assert second["expressions"] == graph["expressions"]
    scope = next(node for node in second["dag"]["nodes"] if (node.get("operator") or {}).get("id") == "rolling_apply")
    assert scope["execution_scope"]["eager_body"] is False


def test_default_override_cache_history_revision_and_no_request_compilation(service, monkeypatch):
    from custom_indicators import series_service
    draft = service.build_rolling_scalar_draft("builtin-maximum-drawdown-v2", 1, 5)["definition"]
    candidate = next(item for item in inspect_parameter_inputs(draft)["candidates"] if item["operator_id"] == "rolling_apply")
    opened = bind_parameter_input(draft, candidate_id=candidate["id"])
    saved = service.create_indicator(opened)
    parameter = saved["parameter_schema"][0]["id"]
    instance = {"indicator_id": saved["id"], "indicator_revision": saved["revision"]}
    monkeypatch.setattr(series_service, "_compile_definition", lambda *a, **k: pytest.fail("request-time compilation"))
    results = service.evaluate_series(indicator_instances=[instance, {**instance, "parameters": {parameter: 10}}], target=TARGET, period="1M")
    a, b = results["results"]
    assert a["parameters"][parameter] == 5
    assert b["parameters"][parameter] == 10
    assert (a["lookback_observations"], b["lookback_observations"]) == (5, 10)
    assert a["parameter_hash"] != b["parameter_hash"]
    assert len(results["execution"]["compiled_plan_ids"]) == 1
    again = service.evaluate_series(indicator_instances=[instance], target=TARGET, period="1M")
    assert again["cache"]["hits"] == 1
    assert service.get_indicator(saved["id"])["parameter_schema"][0]["default"] == 5


def test_immutable_old_transform_versions_and_builtin_revision_two(service):
    source = service.get_indicator("builtin-annualized-sharpe-v2")
    for version, token in (("1.0.0", "rolling_mean("), ("2.0.0", "rolling_window("), ("3.0.0", "rolling_apply(")):
        draft = derive_rolling_series_definition(source, 5, transform_version=version)
        assert token in draft["expression"]
        saved = service.create_indicator(draft)
        assert saved["rolling_source"]["transform_version"] == version
    builtin = service.get_indicator("builtin-rolling-5d-annualized-sharpe-series", 2)
    assert builtin["rolling_source"]["transform_version"] == "2.0.0"
    assert "rolling_apply" not in builtin["expression"]
    assert "rolling_std" in service.get_indicator("builtin-rolling-5d-annualized-sharpe-series", 1)["expression"]


def test_ineligible_shape_or_history_is_explained_and_not_derived(service):
    item = service.create_indicator({"name": "窗口外状态测试", "expression": "mean(recursive_smooth(market_close,3,50))"})
    assert not item["rolling_series_compatibility"]["supported"]
    assert item["rolling_series_compatibility"]["code"] == "ROLLING_INTERVAL_POLICY_REQUIRED"
    from custom_indicators.errors import ValidationError
    with pytest.raises(ValidationError):
        service.build_rolling_scalar_draft(item["id"], 1, 5)


@pytest.mark.parametrize("indicator_id", ["builtin-maximum-drawdown-v2", "builtin-calmar-ratio-v2", "builtin-historical-cvar-95-v2"])
def test_excel_uses_complete_window_graph_and_writes_input_series_once(service, indicator_id):
    saved = service.create_indicator(service.build_rolling_scalar_draft(indicator_id, 1, 5)["definition"])
    artifact = service.export_excel(indicator_ids=[saved["id"]], inline_definition=None, targets=[TARGET], period="1W")
    try:
        with zipfile.ZipFile(artifact.path) as archive:
            assert archive.testzip() is None
            workbook_xml = ET.fromstring(archive.read("xl/workbook.xml"))
            assert all(item.attrib["name"].startswith("_xlnm.") for item in workbook_xml.findall(".//s:definedName", NS))
            # Streaming writers use inline strings; both forms are valid XLSX.
            strings = []
            if "xl/sharedStrings.xml" in archive.namelist():
                shared = ET.fromstring(archive.read("xl/sharedStrings.xml"))
                strings = [''.join(item.itertext()) for item in shared.findall('s:si', NS)]
            rendered_texts = []
            all_formulas = []
            for name in archive.namelist():
                if name.startswith("xl/worksheets/sheet") and name.endswith(".xml"):
                    root = ET.fromstring(archive.read(name))
                    all_formulas.extend(node.text or "" for node in root.findall(".//s:f", NS))
                    for cell in root.findall('.//s:c', NS):
                        if cell.attrib.get('t') == 's':
                            rendered_texts.append(strings[int(cell.find('s:v', NS).text)])
                        elif cell.attrib.get('t') == 'inlineStr':
                            rendered_texts.append(''.join(cell.find('s:is', NS).itertext()))
            for variable in ('returns', 'adjusted_nav'):
                titles = [text for text in rendered_texts if text.startswith('直接入参 ·') and f'（{variable}）' in text]
                assert len(titles) <= 1
            assert all_formulas
            assert not any("rolling_apply(" in formula or "_xlfn" in formula for formula in all_formulas)
            assert any("IFERROR(" in formula and "COUNT(" in formula for formula in all_formulas)
            if "drawdown" in indicator_id or "calmar" in indicator_id:
                assert any("MAX(" in formula and "-1" in formula for formula in all_formulas)
            if "cvar" in indicator_id:
                assert any("PERCENTILE" in formula for formula in all_formulas)
    finally:
        artifact.cleanup()


def test_scope_aware_causality_cannot_audit_body_as_full_history():
    from causality import audit_expression, Verdict
    for body in ("-min_value(drawdown_series(adjusted_nav))", "quantile(returns,0.05)"):
        result = audit_expression(f"rolling_apply({body}, 10)")
        assert result.verdict is Verdict.CAUSAL, result.as_dict()
        assert not result.window_consuming
    unscoped = audit_expression("mean(returns)")
    assert unscoped.verdict is Verdict.WINDOW_CONSUMING

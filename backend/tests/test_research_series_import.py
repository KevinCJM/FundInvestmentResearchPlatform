import io
import json
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
from openpyxl import Workbook
import pytest

from backend.research_series.file_import import parse_series_file
from backend.research_series.service import ResearchSeriesService, read_upload_artifact
from backend.research_series.numba_kernels import research_series_numba_execution_audit
from backend.services import research_series_routes as routes


@pytest.fixture
def client(tmp_path, monkeypatch):
    service = ResearchSeriesService(tmp_path)
    monkeypatch.setattr(routes, "research_series_service", service)
    app = FastAPI()
    app.include_router(routes.router)
    return TestClient(app), service


def workbook_bytes(formula=False):
    workbook = Workbook()
    workbook.active.title = "说明"
    workbook.active.append(["说明"])
    workbook.active.append(["请选择行情页"])
    sheet = workbook.create_sheet("行情")
    sheet.append(["日期", "收盘价"])
    sheet.append(["2024-01-02", "=1+2" if formula else 100.0])
    sheet.append(["2024-01-03", None])
    stream = io.BytesIO()
    workbook.save(stream)
    return stream.getvalue()


def test_excel_sheet_selection_and_preview_is_not_persisted(client):
    api, service = client
    response = api.post("/api/research-series/parse-file?filename=index.xlsx&sheet=行情", content=workbook_bytes())
    assert response.status_code == 200
    assert response.json() == {"columns": ["日期", "收盘价"], "sheets": ["说明", "行情"], "sheet": "行情", "rows": [{"日期": "2024-01-02", "收盘价": 100}, {"日期": "2024-01-03", "收盘价": None}]}
    assert service.uploaded_series()["total"] == 0


@pytest.mark.parametrize("content,filename", [
    (b"date,date\n2024-01-01,1", "x.csv"),
    (b"date,value\n2024-01-01,1,2", "x.csv"),
    (b"[]", "x.json"), (b"{}", "x.json"),
    (b'[{"date":"2024-01-01","value":NaN}]', "x.json"),
    (b"not excel", "x.xlsx"), (b"old excel", "x.xls"),
    (b"", "x.csv"),
])
def test_invalid_file_rejected_without_writes(client, content, filename):
    api, service = client
    response = api.post(f"/api/research-series/parse-file?filename={filename}", content=content)
    assert response.status_code == 400
    assert service.uploaded_series()["total"] == 0


def test_formula_and_size_and_row_limits(client):
    api, _ = client
    response = api.post("/api/research-series/parse-file?filename=x.xlsx&sheet=行情", content=workbook_bytes(True))
    assert response.status_code == 400 and "公式" in response.json()["detail"]
    assert api.post("/api/research-series/parse-file?filename=x.csv", content=b"x" * (8 * 1024 * 1024 + 1)).status_code == 413
    with pytest.raises(ValueError, match="20,000"):
        parse_series_file(b"date,value\n" + b"2024-01-01,1\n" * 20001, "x.csv")


def test_csv_encoding_and_null_are_preserved():
    parsed = parse_series_file('日期,数值\n2024-01-01,1.2\n2024-01-02,\n'.encode("gb18030"), "x.csv")
    assert parsed["rows"][1]["数值"] == ""


def test_named_uploads_reusable_without_market_snapshot_and_no_compilation(client):
    api, service = client
    request = {"name": "我的指数", "frequency": "monthly", "inline_rows": [{"date": "2024-01-01", "value": 1}, {"date": "2024-02-01", "value": None}, {"date": "2024-03-01", "value": 2}]}
    first = api.post("/api/research-series/profile", json=request)
    assert first.status_code == 200, first.text
    binding = first.json()["binding_parameters"]
    again = api.post("/api/research-series/profile", json=request)
    assert again.json()["binding_parameters"] == binding
    catalog = api.get("/api/research-series/uploads?q=我的").json()
    assert catalog["total"] == 1
    assert catalog["items"][0]["binding_parameters"] == binding
    assert catalog["items"][0]["coverage"]["observations"] == 3
    assert api.get("/api/research-series/uploads?q=没有").json()["total"] == 0
    assert api.get("/api/research-series/uploads?offset=1").json()["items"] == []
    original = read_upload_artifact(service.data_dir, binding["artifact_id"], binding["checksum"])
    assert original["value"].isna().sum() == 1
    changed = api.post("/api/research-series/profile", json={**request, "inline_rows": [{"date": "2024-01-01", "value": 3}]})
    assert changed.json()["binding_parameters"]["artifact_id"] != binding["artifact_id"]
    assert read_upload_artifact(service.data_dir, binding["artifact_id"], binding["checksum"]).equals(original)
    assert research_series_numba_execution_audit()["request_time_compilation"] == 0
    assert api.get("/api/research-series/catalog?kind=indicator").status_code == 200


def test_indicator_catalog_explains_incompatible_versions(client):
    api, service = client
    base = {"id": "indicator", "name": "收益", "expression": "1", "dsl_version": "2.0", "context_kind": "single_product", "applicable_product_kinds": ["fund"], "periods": ["1Y"]}
    (service.workspace_data_dir / "custom_indicators.json").write_text(json.dumps({"items": [{"current": {**base, "revision": 3}, "history": [{**base, "revision": 2, "context_kind": "portfolio"}, {**base, "revision": 1, "dsl_version": "1.0"}]}]}))
    items = api.get("/api/research-series/catalog?kind=indicator").json()["items"]
    assert len(items) == 3
    assert sum(item["binding_supported"] for item in items) == 1
    assert all(item["binding_reason"] for item in items if not item["binding_supported"])
    assert all(item["product_kinds"] == ["fund"] for item in items)


def test_canonical_registry_includes_builtins_and_exact_history(tmp_path, monkeypatch):
    versions = [{"id": "builtin", "revision": 1}, {"id": "custom", "revision": 2}, {"id": "custom", "revision": 1}]
    calls = []
    def get_indicator(identifier, revision):
        calls.append((identifier, revision))
        return {"id": identifier, "revision": revision, "name": identifier, "expression": "1", "dsl_version": "2.0", "context_kind": "single_product", "applicable_product_kinds": ["etf", "fund"], "periods": ["1Y"]}
    monkeypatch.setattr(routes, "indicator_service", SimpleNamespace(indicators=SimpleNamespace(list_all_versions=lambda: versions), get_indicator=get_indicator))
    service = ResearchSeriesService(tmp_path, indicator_versions=routes._indicator_versions)
    items = service.catalog(kind="indicator")["items"]
    assert {item["id"] for item in items} == {"indicator:builtin@1", "indicator:custom@1", "indicator:custom@2"}
    assert all(item["binding_supported"] for item in items)
    assert calls == [("builtin", 1), ("custom", 2), ("custom", 1)]

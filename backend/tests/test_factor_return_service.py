"""Offline API integration: real kernels, isolated files, no external services."""
import copy
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from backend.custom_indicators.errors import IndicatorDomainError
from backend.factor_research import return_kernels as kernels
from backend.factor_research.return_contracts import FF3SourceFields, ReturnDatasetFields
from backend.services.factor_research_routes import build_router
from backend.tests import test_factor_research_service as factor_fixtures

context = factor_fixtures.context

PREFIX = "/api/factor-research"


def source_fields():
    dates = pd.bdate_range("2024-06-28", periods=90).strftime("%Y-%m-%d").tolist()
    caps = [1., 2., 3., 10., 11., 12.]
    ratios = [1., 2., 3., 1., 2., 3.]
    return {
        "name": "离线FF3六组原始面板", "source_url": "https://example.org/offline-fixture",
        "market": "CN", "currency": "CNY", "construction": "仅用于离线测试，不是真实市场或官方因子数据。",
        "calendar": dates,
        "formations": [{"date": dates[0], "asset": f"STOCK{a}", "market_cap": cap,
                         "december_market_cap": 100., "book_equity": ratios[a] * 100.,
                         "fiscal_year_end": "2023-12-31", "announced_date": "2024-04-01",
                         "reference_member": True} for a, cap in enumerate(caps)],
        "returns": [{"date": day, "asset": f"STOCK{a}",
                     "return_value": .001 * np.sin(t / (3. + a)) + .0001 * a,
                     "lagged_market_cap": cap, "weight_date": dates[t - 1]}
                    for t, day in enumerate(dates) if t > 0 for a, cap in enumerate(caps)],
        "rf": [{"date": day, "RF": .0001} for day in dates[1:]],
    }


def create_run(service, fields):
    study = service.save_study(fields)
    return service.run_study(study["id"], study["revision"])


def client_for(service):
    app = FastAPI()
    app.include_router(build_router(service))
    return TestClient(app)


def test_ff3_full_api_create_revision_construct_export_and_kind_guards(context):
    service, _, _ = context
    with client_for(service) as client:
        catalog = client.get(PREFIX + "/return-catalog").json()
        assert catalog["ready"]
        assert not next(row for row in catalog["methods"] if row["id"] == "native_stock_ff3")["available"]
        response = client.post(PREFIX + "/return-sources", json=source_fields())
        assert response.status_code == 201, response.text
        source = response.json()
        fields = {"name": "构建FF3", "method": "ff3_2x3", "source_panel_id": source["id"]}
        response = client.post(PREFIX + "/return-plans", json=fields)
        assert response.status_code == 201, response.text
        plan = response.json()
        path = PREFIX + f"/return-plans/{plan['id']}"
        service._ready = False
        assert client.post(path + "/runs", json={"revision": 1}).status_code == 503
        service.warm()
        with service.computing(), service.computing():
            assert client.post(path + "/runs", json={"revision": 1}).status_code == 429
        response = client.post(path + "/runs", json={"revision": 1})
        assert response.status_code == 201, response.text
        dataset = response.json()
        assert dataset["kind"] == "dataset"
        assert dataset["factor_names"] == ["MKT_RF", "SMB", "HML"]
        assert dataset["dependent_return"] == "excess"
        assert len(dataset["rows"]) == 89
        assert dataset["source_checksum"] == source["checksum"]
        assert dataset["execution"]["python_fallback"] == 0
        assert "ff3_returns_kernel" in dataset["execution"]["kernel_signatures"]
        assert dataset["formation_evidence"][0]["counts"] == {key: 1. for key in ("SL", "SM", "SH", "BL", "BM", "BH")}
        get = PREFIX + f"/return-datasets/{dataset['id']}"
        assert client.get(get).json() == dataset
        export = client.get(get + "/export")
        assert export.status_code == 200 and "text/csv" in export.headers["content-type"]
        assert export.text.splitlines()[0] == "date,MKT_RF,SMB,HML,RF"
        assert client.get(PREFIX + f"/runs/{dataset['id']}").status_code == 422
        assert client.get(PREFIX + f"/return-datasets/{source['id']}").status_code == 422
        assert client.put(path, json={**fields, "name": "新修订", "revision": 1}).status_code == 200
        assert client.put(path, json={**fields, "revision": 1}).status_code == 409
        old = client.post(path + "/runs", json={"revision": 1}).json()
        assert old["name"] == "构建FF3" and old["rows"] == dataset["rows"]
        assert client.get(PREFIX + "/datasets").json()["items"][0]["source_method"] == "ff3_2x3"


@pytest.mark.parametrize("mutation", ["announcement", "fiscal", "weight_date", "rf_gap", "duplicate", "bad_calendar", "infinity"])
def test_source_rejects_unusable_or_non_pit_inputs(mutation):
    fields = source_fields()
    if mutation == "announcement":
        fields["formations"][0]["announced_date"] = "2024-06-28"
    elif mutation == "fiscal":
        fields["formations"][0]["fiscal_year_end"] = "2024-03-31"
    elif mutation == "weight_date":
        fields["returns"][0]["weight_date"] = fields["returns"][0]["date"]
    elif mutation == "rf_gap":
        fields["rf"].pop()
    elif mutation == "duplicate":
        fields["returns"].append(copy.deepcopy(fields["returns"][0]))
    elif mutation == "bad_calendar":
        fields["calendar"][0] = "2024-06-27"
    else:
        fields["returns"][0]["return_value"] = float("inf")
    with pytest.raises(ValidationError):
        FF3SourceFields.model_validate(fields)


def test_ff3_empty_groups_are_rejected_not_repaired(context):
    service, _, _ = context
    fields = source_fields()
    for row in fields["formations"]:
        row["book_equity"] = 100.
    source = service.return_research.import_source(fields)
    plan = service.return_research.save_plan({"name": "常数不可构建", "method": "ff3_2x3", "source_panel_id": source["id"]})
    with pytest.raises(IndicatorDomainError, match="分组为空"):
        service.return_research.run(plan["id"], 1)


def test_construct_spread_from_frozen_inputs_and_guard_model_compatibility(context, monkeypatch):
    service, fields, market = context
    run = create_run(service, fields)
    plan = service.return_research.save_plan({"name": "动量差额", "method": "characteristic_spread",
                                             "source_run_id": run["id"], "factor_key": run["factor_snapshots"][0]["id"],
                                             "output_factor": "MOM", "cost_bps": 5.})
    spy = Mock(wraps=kernels.spread_returns_kernel)
    monkeypatch.setattr(kernels, "spread_returns_kernel", spy)
    first = service.return_research.run(plan["id"], 1)
    spy.assert_called_once()
    assert spy.call_args.args[0].dtype == np.float64 and spy.call_args.args[0].flags.c_contiguous
    assert first["factor_names"] == ["MOM"] and first["dependent_return"] == "total"
    assert first["source_input_checksum"] == run["input_checksum"]
    assert any(row["MOM"] is not None for row in first["rows"])
    # Deliberately alter today's market file: a frozen parent must give identical returns.
    frame = pd.read_parquet(market / "etf_daily_df.parquet")
    frame["adj_nav"] *= 1. + np.linspace(0., .5, len(frame))
    frame.to_parquet(market / "etf_daily_df.parquet", index=False)
    second = service.return_research.run(plan["id"], 1)
    assert first["rows"] == second["rows"]
    assert first["input_checksum"] == second["input_checksum"]
    attribution = {key: fields[key] for key in ("name", "product_kind", "targets", "start_date", "end_date", "oos_date")}
    with pytest.raises(IndicatorDomainError, match="FF3 需要"):
        service.run_attribution({**attribution, "model": "ff3", "dataset_id": first["id"]})
    result = service.run_attribution({**attribution, "model": "factor_regression", "dataset_id": first["id"]})
    assert result["dependent_return"] == "total"
    assert result["factor_names"] == ["MOM"]
    assert result["dataset_snapshot"]["id"] == first["id"]
    assert "diagnostics" not in result["dataset_snapshot"] and "leg_returns" not in result["dataset_snapshot"]
    assert any(row["status"] == "ok" for row in result["results"])
    assert any("不是风险调整" in warning for warning in result["warnings"])
    with pytest.raises(IndicatorDomainError):
        service.publish({"run_id": first["id"], "name": "不能入池", "effective_from": pd.Timestamp.today().date(),
                         "effective_to": pd.Timestamp.today().date()})


def test_tampered_ff3_source_fails_closed(context, monkeypatch):
    service, _, _ = context
    source = service.return_research.import_source(source_fields())
    plan = service.return_research.save_plan({"name": "校验原始面板", "method": "ff3_2x3", "source_panel_id": source["id"]})
    tampered = copy.deepcopy(service.return_research.source(source["id"]))
    tampered["formations"][0]["book_equity"] *= 2
    monkeypatch.setattr(service.return_research, "source", lambda _: tampered)
    with pytest.raises(IndicatorDomainError, match="校验和"):
        service.return_research.run(plan["id"], 1)


def test_tampered_frozen_arrays_fail_closed(context):
    service, fields, _ = context
    run = create_run(service, fields)
    path = service.artifacts.root / (run["id"] + ".npz")
    with np.load(path, allow_pickle=False) as original:
        arrays = dict(original)
    arrays["prices"][0, 0] = 987.
    np.savez_compressed(path, **arrays)  # Only isolated temporary fixture files.
    with pytest.raises(IndicatorDomainError, match="校验和"):
        service.artifacts.load_arrays(run["id"])


def test_generic_import_schema_nulls_legacy_ff3_and_csv(context):
    service, _, _ = context
    fields = {"name": "通用收益", "source_url": "https://example.org/offline", "market": "CN", "currency": "CNY",
              "construction": "离线测试构造，明确小数日收益。", "factor_names": ["MOM"], "dependent_return": "total",
              "rows": [{"date": day.strftime("%Y-%m-%d"), "values": {"MOM": .001 if i != 5 else None}}
                       for i, day in enumerate(pd.bdate_range("2024-07-01", periods=40))]}
    dataset = service.return_research.import_dataset(fields)
    assert dataset["rows"][5]["MOM"] is None
    assert dataset["diagnostics"]["factors"][0]["observations"] == 39
    assert service.return_research.export_csv(dataset["id"]).splitlines()[6].endswith(",")
    bad = copy.deepcopy(fields)
    bad["factor_names"] = ["=1+1"]
    with pytest.raises(ValidationError):
        ReturnDatasetFields.model_validate(bad)
    bad = copy.deepcopy(fields)
    bad["dependent_return"] = "excess"
    with pytest.raises(ValidationError, match="RF"):
        ReturnDatasetFields.model_validate(bad)
    # Old flat FF3 artifact lacks v2 metadata; read adapter must not mutate it.
    legacy = service.artifacts.save("dataset", {"name": "历史FF3", "market": "CN", "currency": "CNY",
        "source_url": "https://example.org/offline", "rows": [{"date": row["date"], "MKT_RF": .01, "SMB": .02, "HML": .03, "RF": 0.}
                                                               for row in fields["rows"]]})
    adapted = service.return_research.dataset(legacy["id"])
    assert adapted["factor_names"] == ["MKT_RF", "SMB", "HML"]
    assert service.artifacts.get(legacy["id"]) == legacy


@pytest.mark.parametrize("frequency", ["daily", "weekly", "monthly"])
def test_signal_frequency_and_rolling_ic_label_maturity(context, frequency):
    service, fields, _ = context
    run = create_run(service, {**fields, "signal_frequency": frequency, "ic_window": 6, "ic_min_periods": 3})
    periods = run["periods"]
    expected_minimum = {"daily": 600, "weekly": 120, "monthly": 25}[frequency]
    assert len(periods) > expected_minimum
    rolling = run["rolling_diagnostics"]
    assert rolling["date_basis"] == "label_end"
    assert rolling["rows"]
    periods_by_signal = {row["date"]: row for row in periods}
    for row in rolling["rows"]:
        assert row["date"] == periods_by_signal[row["signal_date"]]["label_end"]
        assert row["date"] <= run["as_of"] and row["date"] > row["signal_date"]
        if row["sample"] == "in_sample":
            assert row["date"] < fields["oos_date"]
        assert row["sample"] != "purged"
    first_oos = next(row for row in rolling["rows"] if row["sample"] == "out_of_sample")
    assert first_oos["rank_ic"][-1]["mean"] is None

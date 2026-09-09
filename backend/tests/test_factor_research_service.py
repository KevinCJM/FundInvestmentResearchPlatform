from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.factor_research.service import FactorResearchService
from backend.factor_research.catalog import BUILTINS
from backend.custom_indicators.errors import IndicatorDomainError
from backend.product_pools.repository import ProductPoolRepository
from backend.product_pools.service import ProductPoolService
from backend.services.factor_research_routes import build_router


@pytest.fixture
def context(tmp_path):
    market = tmp_path / "market"
    market.mkdir()
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=900)
    codes = [f"51000{i}.SH" for i in range(6)]
    info = pd.DataFrame({"ts_code": codes, "code": [c[:6] for c in codes],
                         "name": [f"研究ETF{i}" for i in range(6)],
                         "fund_type": "股票型", "qdii_type": "非QDII", "list_date": "2000-01-01"})
    info.to_parquet(market / "etf_info_df.parquet", index=False)
    rows = []
    t = np.arange(len(dates))
    for a, code in enumerate(codes):
        prices = np.cumprod(1 + .00015 * (a + 1) + .009 * np.sin(t / (9 + a * 3) + a))
        rows.append(pd.DataFrame({"ts_code": code, "date": dates, "nav_date": dates.strftime("%Y%m%d"),
                                  "ann_date": dates.strftime("%Y%m%d"), "adj_nav": prices, "unit_nav": prices}))
    pd.concat(rows).to_parquet(market / "etf_daily_df.parquet", index=False)
    pd.DataFrame({"exchange": "SSE", "cal_date": dates.strftime("%Y%m%d"), "is_open": 1}).to_parquet(market / "trade_day_df.parquet", index=False)
    pd.concat([pd.DataFrame({"ts_code": f"INDEX{i}", "trade_date": dates, "close":
                            np.cumprod(1 + .001 * np.sin(t / (10 + i)))}) for i in range(3)]).to_parquet(market / "index_daily_df.parquet", index=False)
    service = FactorResearchService(tmp_path / "workspace", market)
    service.warm()
    fields = {
        "name": "真实结构测试", "product_kind": "etf", "targets": codes,
        "start_date": dates[180].date().isoformat(), "end_date": dates[-1].date().isoformat(),
        "oos_date": dates[-250].date().isoformat(),
        "benchmark": {"kind": "etf", "code": codes[0], "label": "ETF 净值基准", "return_basis": "adjusted_nav"},
        "factors": [{"factor_id": x["id"], "revision": 1, "weight": w} for x, w in zip(BUILTINS[:3], (.5, .3, .2))],
        "quantiles": 3, "top_n": 2,
    }
    return service, fields, market


def test_api_cold_fail_closed_validation_and_real_run(context):
    service, fields, _ = context
    app = FastAPI()
    app.include_router(build_router(service))
    with TestClient(app) as client:
        saved = client.post("/api/factor-research/studies", json=fields)
        assert saved.status_code == 201, saved.text
        study = saved.json()
        service._ready = False
        assert client.post(f"/api/factor-research/studies/{study['id']}/runs", json={"revision": 1}).status_code == 503
        service.warm()
        response = client.post(f"/api/factor-research/studies/{study['id']}/runs", json={"revision": 1})
        assert response.status_code == 201, response.text
        run = response.json()
        assert run["execution"]["python_fallback"] == 0
        assert run["execution"]["nopython"]
        assert run["summaries"]["out_of_sample"]["factors"][-1]["rank_ic"]["observations"] > 0
        assert run["summaries"]["out_of_sample"]["performance"]["days"] > 100
        assert len(run["input_checksum"]) == 64
        assert client.get(f"/api/factor-research/runs/{run['id']}").json() == run
        changed = {**fields, "revision": 1, "name": "新版本"}
        assert client.put(f"/api/factor-research/studies/{study['id']}", json=changed).status_code == 200
        assert client.put(f"/api/factor-research/studies/{study['id']}", json=changed).status_code == 409
        assert client.get(f"/api/factor-research/runs/{run['id']}").json()["study_snapshot"]["name"] == fields["name"]
        invalid = {**fields, "targets": [fields["targets"][0]] * 3}
        assert client.post("/api/factor-research/studies", json=invalid).status_code == 422
        assert client.post("/api/factor-research/studies", json={**fields, "product_kind": "stock"}).status_code == 422


def test_revision_locks_purging_and_input_snapshot(context):
    service, fields, _ = context
    custom = service.save_factor({"name": "动量", "operator": "momentum", "window": 21})
    fields["factors"][0] = {"factor_id": custom["id"], "revision": 1, "weight": .5}
    study = service.save_study(fields)
    service.save_factor({"name": "动量2", "operator": "momentum", "window": 63}, custom["id"], 1)
    run = service.run_study(study["id"], 1)
    assert run["factor_snapshots"][0]["window"] == 21
    for period in run["periods"]:
        if period["sample"] == "in_sample":
            assert period["label_end"] < fields["oos_date"]
        elif period["sample"] == "out_of_sample":
            assert period["date"] >= fields["oos_date"]
    arrays = np.load(service.artifacts.root / f"{run['id']}.npz", allow_pickle=False)
    assert arrays["prices"].dtype == np.float64
    assert arrays["available"].dtype == np.int64
    again = service.run_study(study["id"], 1)
    assert again["id"] != run["id"]
    assert again["input_checksum"] == run["input_checksum"]
    assert again["latest_scores"] == run["latest_scores"]


def test_missing_ann_date_is_never_silently_backfilled(context):
    service, fields, market = context
    frame = pd.read_parquet(market / "etf_daily_df.parquet").drop(columns="ann_date")
    frame.to_parquet(market / "etf_daily_df.parquet", index=False)
    study = service.save_study(fields)
    with pytest.raises(IndicatorDomainError, match="有效截面"):
        service.run_study(study["id"], 1)


def test_release_import_keeps_manual_gate_and_frozen_evidence(context):
    service, fields, _ = context
    study = service.save_study(fields)
    run = service.run_study(study["id"], 1)
    today = date.today()
    release = service.publish({"run_id": run["id"], "name": "研究发布", "effective_from": today.isoformat(),
                               "effective_to": (today + timedelta(days=30)).isoformat()})

    class Gateway:
        def get_plan(self, object_id):
            return service.evaluation_plan(object_id)
        def run_plan(self, object_id, as_of=None):
            return service.evaluation_run(object_id, as_of)
        def get_run_page(self, object_id, *, page, page_size):
            return service.evaluation_page(object_id, page, page_size)

    pools = ProductPoolService(ProductPoolRepository(service.workspace_data_dir / "product_pools.json"), Gateway())
    pool = pools.create_pool({"name": "候选池"})
    attached = pools.attach_evaluation_plan(pool["id"], pool["revision"],
                                           {"plan_id": release["id"], "selection_mode": "top_n", "selection_value": 2})
    assert len(attached["members"]) == 2
    assert all(x["research_status"] == "pending" for x in attached["members"])
    assert all(x["evidences"][0]["result_id"] == run["id"] for x in attached["members"])
    binding = service.bind({"release_id": release["id"], "context_type": "saa", "context_id": "test-research-1"})
    assert binding["run_id"] == run["id"]
    profile = service.portfolio_profile(release["id"], {"as_of": today.isoformat(),
        "holdings": [{"product_id": fields["targets"][0], "weight": .6}, {"product_id": "UNKNOWN", "weight": .4}]})
    assert all(x["covered_weight"] == .6 for x in profile["factors"])
    monitor = service.monitor(release["id"])
    assert monitor["comparable"] and not monitor["data_changed_since_run"]
    assert monitor["drift"]["mean_absolute_score_change"] == 0
    service.retire(release["id"])
    with pytest.raises(IndicatorDomainError):
        service.evaluation_run(release["id"])
    assert service.artifacts.get(run["id"]) == run


def test_attribution_actual_arrays_and_dataset_compatibility(context):
    service, fields, _ = context
    request = {key: fields[key] for key in ("name", "product_kind", "targets", "start_date", "end_date", "oos_date")}
    result = service.run_attribution({**request, "model": "rbsa", "indices": ["INDEX0", "INDEX1", "INDEX2"]})
    assert result["execution"]["python_fallback"] == 0
    assert any(x["status"] == "ok" for x in result["results"])
    for item in result["results"]:
        if item["status"] == "ok":
            assert sum(x["value"] for x in item["exposures"]) == pytest.approx(1)
            assert item["test_observations"] > 0
    dates = pd.bdate_range(start=fields["start_date"], periods=35)
    dataset = service.add_dataset({"name": "美国研究数据", "source_url": "https://example.org/research",
        "market": "US", "currency": "USD", "construction": "测试专用，不是真实官方 FF3 数据",
        "rows": [{"date": d.date().isoformat(), "MKT_RF": .001, "SMB": .002, "HML": .003, "RF": 0} for d in dates]})
    with pytest.raises(IndicatorDomainError, match="不匹配"):
        service.run_attribution({**request, "model": "ff3", "dataset_id": dataset["id"]})


def test_monitor_compares_same_score_definition_when_research_end_advances(context):
    service, fields, _ = context
    full_end = fields["end_date"]
    fields["end_date"] = (pd.Timestamp(full_end) - pd.offsets.BDay(1)).date().isoformat()
    study = service.save_study(fields)
    first = service.run_study(study["id"], 1)
    today = date.today()
    release = service.publish({"run_id": first["id"], "name": "监控测试", "effective_from": today.isoformat(),
                               "effective_to": (today + timedelta(days=30)).isoformat()})
    fields["end_date"] = full_end
    updated = service.save_study(fields, study["id"], 1)
    second = service.run_study(updated["id"], 2)
    result = service.monitor(release["id"])
    assert result["comparable"]
    assert result["latest_run_id"] == second["id"]
    assert result["drift"]["common_products"] == len(fields["targets"])
    assert not result["data_changed_since_run"]


def test_actual_service_call_path_reaches_njit_dispatchers(context, monkeypatch):
    from unittest.mock import Mock
    from backend.factor_research import numba_kernels as kernels
    service, fields, _ = context
    names = ["features_kernel", "normalize_kernel", "labels_kernel", "diagnostics_kernel", "backtest_kernel"]
    spies = {}
    for name in names:
        spy = Mock(wraps=getattr(kernels, name))
        monkeypatch.setattr(kernels, name, spy)
        spies[name] = spy
    study = service.save_study(fields)
    result = service.run_study(study["id"], 1)
    assert result["execution"]["python_fallback"] == 0
    for spy in spies.values():
        spy.assert_called_once()
        assert spy.call_args.args[0].flags.c_contiguous
        assert spy.call_args.args[0].dtype == np.float64


def test_index_basis_and_qdii_capability_are_explicit(context):
    service, fields, market = context
    pd.DataFrame({"ts_code": ["INDEX0"], "name": ["测试价格指数"]}).to_parquet(market / "index_catalog_df.parquet", index=False)
    index_fields = {**fields, "benchmark": {"kind": "index", "code": "INDEX0", "label": "测试价格指数", "return_basis": "total_return_index"}}
    study = service.save_study(index_fields)
    with pytest.raises(IndicatorDomainError, match="口径不匹配"):
        service.run_study(study["id"], 1)
    info = pd.read_parquet(market / "etf_info_df.parquet")
    info.loc[0, "qdii_type"] = "QDII"
    info.to_parquet(market / "etf_info_df.parquet", index=False)
    study = service.save_study(fields)
    with pytest.raises(IndicatorDomainError, match="跨市场"):
        service.run_study(study["id"], 1)

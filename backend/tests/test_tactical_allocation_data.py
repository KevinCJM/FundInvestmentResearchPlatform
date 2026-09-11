from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from backend.custom_indicators.errors import ValidationError
from backend.tactical_allocation.data import TacticalAllocationData, warm_tactical_data
from backend.tactical_allocation.repository import TacticalAllocationRepository


@pytest.fixture(scope="module", autouse=True)
def warmed():
    warm_tactical_data()
    from backend.tactical_allocation.numeric import warm_tactical_allocation_kernels
    warm_tactical_allocation_kernels()


@pytest.fixture
def data(tmp_path):
    info = [
        {"asset_alloc_name": "配置", "asset_name": name, "etf_code": code,
         "etf_name": name + "ETF", "etf_weight": 100.0,
         "creat_time": pd.Timestamp("2026-01-01"), "as_of": None,
         "universe_snapshot_id": None, "data_release_id": None}
        for name, code in [("股票", "510300.SH"), ("债券", "511010.SH")]
    ]
    pd.DataFrame(info).to_parquet(tmp_path / "asset_alloc_info.parquet", index=False)
    days = pd.date_range("2024-01-01", periods=6)
    nav = [{"asset_alloc_name": "配置", "asset_name": name, "date": day,
            "nv": values[index], "as_of": None, "available_at": day}
           for name, values in [("股票", [1.0, 1.1, 1.0, 1.2, 1.1, 1.3]),
                                ("债券", [1.0, 1.01, 1.02, 1.03, 1.04, 1.05])]
           for index, day in enumerate(days)]
    pd.DataFrame(nav).to_parquet(tmp_path / "asset_nv.parquet", index=False)
    return TacticalAllocationData(tmp_path)


def baseline(data):
    return data.create_baseline({"alloc_name": "配置", "name": "战略基线", "as_of": "2026-01-01",
                                 "weights": {"股票": 0.6, "债券": 0.4}})


def load(data, value=None, **kwargs):
    return data.load_data(value or baseline(data), "2024-01-01", "2024-01-06", kwargs.get("as_of", "2026-01-01"))


def test_real_saa_weights_and_returns_are_read_without_writes(data):
    before = sorted(p.name for p in data.data_dir.iterdir())
    catalog = data.catalog()
    assert catalog["allocations"][0]["alloc_name"] == "配置"
    assert catalog["allocations"][0]["coverage"] == {"start_date": "2024-01-01", "end_date": "2024-01-06"}
    value = baseline(data)
    assert value["assets"][0]["base_weight"] == 0.6
    assert value["assets"][0]["products"][0]["weight"] == 1.0
    result = load(data, value)
    np.testing.assert_allclose(result["returns"][:, 0], np.array([1.1, 1.0, 1.2, 1.1, 1.3]) / np.array([1.0, 1.1, 1.0, 1.2, 1.1]) - 1)
    assert result["period_starts"][0] == "2024-01-01"
    assert result["dates"][0] == "2024-01-02"
    assert result["returns"].flags.c_contiguous and not result["returns"].flags.writeable
    assert result["available_at"].dtype == np.int64 and not result["available_at"].flags.writeable
    assert result["pit"]["status"] == "research_only"
    assert value["apply_eligible"] is False
    assert any("全历史" in reason for reason in value["pit"]["reasons"])
    assert sorted(p.name for p in data.data_dir.iterdir()) == before


def test_strategy_weight_sum_and_class_identity_are_explicit(data):
    for weights, code in [({"股票": 0.7, "债券": 0.4}, "TAA_BASELINE_TOTAL"),
                          ({"股票": 1.0}, "TAA_BASELINE_ASSETS"),
                          ({"股票": True, "债券": 0.0}, "TAA_WEIGHT_INVALID")]:
        with pytest.raises(ValidationError) as error:
            data.create_baseline({"alloc_name": "配置", "name": "策略", "as_of": "2024-01-01", "weights": weights})
        assert error.value.code == code


def test_group_limits_preserve_saa_budgets(data):
    payload = {"alloc_name": "配置", "name": "基线", "as_of": "2024-01-01", "weights": {"股票": .6, "债券": .4},
               "group_limits": [{"id": "风险资产", "assets": ["股票"], "lo": .2, "hi": .7}]}
    assert data.create_baseline(payload)["group_limits"] == payload["group_limits"]
    for group in [{"id": "组", "assets": ["股票"], "hi": .5},
                  {"id": "组", "assets": ["股票", "股票"], "hi": 1},
                  {"id": "组", "assets": ["黄金"], "hi": 1}]:
        with pytest.raises(ValidationError):
            data.create_baseline({**payload, "group_limits": [group]})


def test_missing_knowledge_stays_unknown(data):
    path = data.data_dir / "asset_nv.parquet"
    frame = pd.read_parquet(path).drop(columns="available_at")
    frame.to_parquet(path, index=False)
    result = load(data)
    assert np.all(result["available_at"] == -1)
    assert result["lineage"]["missing_availability_rows"] == 12
    assert result["pit"]["status"] == "research_only"
    assert any("不能按 PIT" in reason for reason in result["reasons"])


def test_intraday_nav_cannot_be_disguised_as_daily_periods(data):
    path = data.data_dir / "asset_nv.parquet"
    frame = pd.read_parquet(path)
    frame.loc[0, "date"] = frame.loc[0, "date"] + pd.Timedelta(hours=12)
    frame.to_parquet(path, index=False)
    with pytest.raises(ValidationError, match="盘中时间"):
        load(data)


def test_endpoint_availability_preserves_publication_lag_and_unknown(data):
    path = data.data_dir / "asset_nv.parquet"
    frame = pd.read_parquet(path)
    frame.loc[0, "available_at"] = pd.Timestamp("2024-01-05")
    frame.loc[2, "available_at"] = pd.NaT
    frame.to_parquet(path, index=False)
    result = load(data)
    expected = int(np.datetime64("2024-01-05", "D").astype(np.int64))
    assert result["available_at"][0, 0] == expected
    assert result["available_at"][1, 0] == -1
    assert result["available_at"][2, 0] == -1


@pytest.mark.parametrize("invalid", ["not-a-date", "2023-12-31"])
def test_invalid_knowledge_is_not_silently_missing_or_backdated(data, invalid):
    path = data.data_dir / "asset_nv.parquet"
    frame = pd.read_parquet(path)
    frame["available_at"] = frame["available_at"].astype(str)
    frame.loc[0, "available_at"] = invalid
    frame.to_parquet(path, index=False)
    with pytest.raises(ValidationError) as error:
        load(data)
    assert error.value.code in {"TAA_KNOWLEDGE_INVALID", "TAA_KNOWLEDGE_DATE"}


def test_future_publications_are_excluded_and_intersection_is_reported(data):
    path = data.data_dir / "asset_nv.parquet"
    frame = pd.read_parquet(path)
    frame.loc[5, "available_at"] = pd.Timestamp("2024-02-01")
    frame.to_parquet(path, index=False)
    result = load(data, as_of="2024-01-06")
    assert result["dates"][-1] == "2024-01-05"
    assert result["lineage"]["excluded_not_yet_available_rows"] == 1
    assert result["lineage"]["excluded_incomplete_dates"] == 1


def test_duplicates_are_not_averaged(data):
    path = data.data_dir / "asset_nv.parquet"
    frame = pd.read_parquet(path)
    pd.concat([frame, frame.iloc[:1]], ignore_index=True).to_parquet(path, index=False)
    with pytest.raises(ValidationError, match="同一资产同一日期"):
        load(data)


def test_configuration_and_nav_changes_invalidate_baseline(data):
    value = baseline(data)
    path = data.data_dir / "asset_nv.parquet"
    frame = pd.read_parquet(path)
    frame.loc[1, "nv"] = 1.2
    frame.to_parquet(path, index=False)
    with pytest.raises(ValidationError) as error:
        load(data, value)
    assert error.value.code == "TAA_NAV_CHANGED"
    value = baseline(data)
    path = data.data_dir / "asset_alloc_info.parquet"
    frame = pd.read_parquet(path)
    frame.loc[0, "etf_name"] = "更名"
    frame.to_parquet(path, index=False)
    with pytest.raises(ValidationError) as error:
        load(data, value)
    assert error.value.code == "TAA_SAA_CHANGED"


def test_unrelated_allocation_append_does_not_change_baseline(data):
    value = baseline(data)
    for filename in ["asset_alloc_info.parquet", "asset_nv.parquet"]:
        path = data.data_dir / filename
        frame = pd.read_parquet(path)
        extra = frame.copy()
        extra["asset_alloc_name"] = "另一个配置"
        pd.concat([frame, extra], ignore_index=True).to_parquet(path, index=False)
    assert load(data, value)["returns"].shape == (5, 2)


def test_universe_membership_is_checked_without_creating_locks(data):
    path = data.data_dir / "asset_alloc_info.parquet"
    frame = pd.read_parquet(path)
    frame["universe_snapshot_id"] = "universe-frozen"
    frame.to_parquet(path, index=False)
    snapshot = {"id": "universe-frozen", "name": "域", "research_date": "2024-01-01",
                "created_at": "2026-01-01T00:00:00Z", "immutable": True,
                "members": [{"kind": "etf", "product_id": code, "name": code, "eligible": True}
                            for code in ["510300.SH", "511010.SH"]]}
    (data.data_dir / "product_pools.json").write_text(json.dumps({"pools": [], "versions": [], "universe_snapshots": [snapshot]}))
    value = baseline(data)
    assert value["apply_eligible"] is True
    assert value["lineage"]["universe"]["id"] == "universe-frozen"
    assert data.validate_application(value)["id"] == "universe-frozen"
    assert not list(data.data_dir.glob("*.lock"))
    snapshot["members"].pop()
    (data.data_dir / "product_pools.json").write_text(json.dumps({"pools": [], "versions": [], "universe_snapshots": [snapshot]}))
    with pytest.raises(ValidationError) as error:
        data.validate_application(value)
    assert error.value.code == "TAA_UNIVERSE_CHANGED"
    assert baseline(data)["apply_eligible"] is False


def test_application_never_trusts_stored_eligibility_flag(data):
    value = baseline(data)
    value["apply_eligible"] = True
    with pytest.raises(ValidationError) as error:
        data.validate_application(value)
    assert error.value.code == "TAA_APPLICATION_UNIVERSE"


def test_baseline_canonicalizes_only_verified_universe_aliases(data):
    path = data.data_dir / "asset_alloc_info.parquet"
    frame = pd.read_parquet(path)
    frame["universe_snapshot_id"] = "universe-frozen"
    frame["etf_code"] = ["510300", "511010"]
    frame.to_parquet(path, index=False)
    snapshot = {"id": "universe-frozen", "name": "域", "research_date": "2024-01-01", "immutable": True,
                "members": [{"kind": "etf", "product_id": code, "name": code, "eligible": True}
                            for code in ["510300.SH", "511010.SH"]]}
    (data.data_dir / "product_pools.json").write_text(json.dumps({"pools": [], "versions": [], "universe_snapshots": [snapshot]}))
    value = baseline(data)
    assert value["assets"][0]["products"][0]["product_id"] == "510300.SH"
    assert value["assets"][0]["products"][0]["source_product_id"] == "510300"
    assert "_resolved_products" not in value["lineage"]["universe"]
    assert data.validate_application(value)["id"] == "universe-frozen"
    # A second alias candidate makes the 6-digit name ambiguous: no guessed ID.
    snapshot["members"].append({"kind": "etf", "product_id": "510300.SZ", "eligible": True})
    (data.data_dir / "product_pools.json").write_text(json.dumps({"pools": [], "versions": [], "universe_snapshots": [snapshot]}))
    ambiguous = baseline(data)
    assert ambiguous["apply_eligible"] is False
    assert ambiguous["assets"][0]["products"][0]["product_id"] == "510300"


def test_immutable_repository_roundtrip_and_corruption_failure(data):
    repo = TacticalAllocationRepository(data.data_dir)
    assert repo.list_baselines() == [] and not repo.artifacts.root.exists()
    value = repo.save_baseline({**baseline(data), "id": "injected", "created_at": "1900-01-01", "content_hash": "injected"})
    assert value["id"] != "injected" and value["created_at"] != "1900-01-01"
    assert repo.get_baseline(value["id"])["content_hash"] == value["content_hash"]
    assert repo.list_baselines() == [value]
    result = load(data, value)
    decision = repo.save_decision({"name": "临时偏离", "baseline_id": value["id"], "source_hash": result["source_hash"]},
                                 {"returns": result["returns"], "available_at": result["available_at"]})
    assert repo.list_decisions()[0]["id"] == decision["id"]
    arrays = repo.decision_arrays(decision["id"])
    assert isinstance(arrays["returns"], np.memmap) and not arrays["returns"].flags.writeable
    np.testing.assert_array_equal(arrays["returns"], result["returns"])
    with pytest.raises(ValidationError):
        repo.get_baseline(decision["id"])
    with pytest.raises(ValidationError):
        repo.get_baseline("../../escape")
    path = repo.artifacts.root / value["id"] / "manifest.json"
    payload = json.loads(path.read_text())
    payload["name"] = "tampered"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValidationError) as error:
        repo.get_baseline(value["id"])
    assert error.value.code == "RESEARCH_ARTIFACT_CORRUPT"

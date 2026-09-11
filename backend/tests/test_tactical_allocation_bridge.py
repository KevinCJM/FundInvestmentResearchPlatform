from __future__ import annotations

import copy
import json
from datetime import date, timedelta

import pandas as pd
import numpy as np
import pytest

from backend.custom_indicators.errors import ValidationError
from backend.tactical_allocation.data import TacticalAllocationData
from backend.tactical_allocation.portfolio_bridge import validate_allocation_source, validate_decision_application
from backend.tactical_allocation.repository import TacticalAllocationRepository
from backend.tests import test_tactical_allocation_data as shared_fixtures

# Register the shared pytest fixtures without importing a name shadowed by
# fixture injection and local test variables.
baseline = shared_fixtures.baseline
data = shared_fixtures.data
warmed = shared_fixtures.warmed


@pytest.fixture
def application(data, monkeypatch):
    path = data.data_dir / "asset_alloc_info.parquet"
    frame = pd.read_parquet(path)
    frame["universe_snapshot_id"] = "universe-frozen"
    frame.to_parquet(path, index=False)
    snapshot = {"id": "universe-frozen", "name": "域", "research_date": "2024-01-01", "immutable": True,
                "members": [{"kind": "etf", "product_id": code, "name": code, "eligible": True}
                            for code in ["510300.SH", "511010.SH"]]}
    (data.data_dir / "product_pools.json").write_text(json.dumps({"pools": [], "versions": [], "universe_snapshots": [snapshot]}))
    repository = TacticalAllocationRepository(data.data_dir)
    frozen = repository.save_baseline(baseline(data))
    preview = {"baseline": frozen, "request": {"as_of": date.today().isoformat(), "max_turnover": .5},
               "weight_path": [{"date": "2024-01-02"}, {"date": "2024-01-03"}],
               "recommendation": {"weights": {"股票": .6, "债券": .4}, "turnover_from_current": .1,
                                  "expires_on": (date.today() + timedelta(days=30)).isoformat()},
               "selected_id": "scale-1", "candidates": [{"id": "scale-1", "strength": 1., "validation_feasible": True}]}
    components = [{"kind": "etf", "product_id": "510300.SH", "asset_class_id": "股票"},
                  {"kind": "etf", "product_id": "511010.SH", "asset_class_id": "债券"}]
    monkeypatch.setenv("TACTICAL_ALLOCATION_DATA_DIR", str(data.data_dir))
    return data, repository, preview, components


def apply(application, *, change=None, components=None):
    data, repository, original, original_components = application
    preview = copy.deepcopy(original)
    if change:
        change(preview)
    decision = repository.save_decision({"preview": preview}, arrays={"returns": np.zeros((2, 2))})
    result = validate_allocation_source({"kind": "taa", "decision_id": decision["id"]},
                                       components or original_components, {"type": "manual", "weights": [.6, .4]},
                                       "universe-frozen", data.data_dir)
    return result, decision


def test_direct_portfolio_request_keeps_server_verified_decision(application):
    result, decision = apply(application)
    assert result["decision_hash"] == decision["content_hash"]
    assert result["class_weights"] == {"股票": .6, "债券": .4}
    assert result["research_only"] is True


@pytest.mark.parametrize("change,code", [
    (lambda p: p["recommendation"].update(expires_on="2000-01-01"), "TAA_DECISION_EXPIRED"),
    (lambda p: p["recommendation"].update(turnover_from_current=.9), "TAA_CURRENT_TURNOVER_LIMIT"),
    (lambda p: p["candidates"][0].update(validation_feasible=False), "TAA_VALIDATION_LIMIT"),
    (lambda p: p.update(candidates=[]), "TAA_CANDIDATE_MISSING"),
    (lambda p: p["baseline"]["lineage"].update(universe=None), "TAA_APPLICATION_UNIVERSE"),
])
def test_direct_request_cannot_bypass_application_gates(application, change, code):
    with pytest.raises(ValidationError) as error:
        apply(application, change=change)
    assert error.value.code == code


def test_product_class_labels_cannot_override_frozen_saa_mapping(application):
    components = copy.deepcopy(application[3])
    components[0]["product_id"], components[1]["product_id"] = components[1]["product_id"], components[0]["product_id"]
    with pytest.raises(ValidationError) as error:
        apply(application, components=components)
    assert error.value.code == "TAA_PRODUCT_CLASS_MISMATCH"


def test_signal_expiry_is_checked_even_when_review_deadline_is_later(application):
    def stale(preview):
        preview["request"].update(signal_mode="momentum", max_signal_age_days=31)
        preview["recommendation"].update(signal_date=(date.today() - timedelta(days=40)).isoformat(), is_saa=False)
    with pytest.raises(ValidationError) as error:
        apply(application, change=stale)
    assert error.value.code == "TAA_SIGNAL_EXPIRED"


def test_unmapped_replacement_is_not_granted_a_budget(application):
    components = copy.deepcopy(application[3])
    components[0]["product_id"] = "159915.SZ"
    with pytest.raises(ValidationError) as error:
        apply(application, components=components)
    assert error.value.code == "TAA_PRODUCT_CLASS_MISMATCH"


def test_retaining_saa_is_allowed_despite_active_validation_failure(application):
    result, _ = apply(application, change=lambda p: p["candidates"][0].update(strength=0., validation_feasible=False))
    assert result["research_only"] is True


def test_universe_directory_is_independent_from_market_data(application, tmp_path):
    data, _, preview, _ = application
    empty_market = tmp_path / "market"
    reader = TacticalAllocationData(empty_market, universe_dir=data.data_dir)
    validate_decision_application({"preview": preview}, reader)
    assert not empty_market.exists()


def test_saved_input_quality_is_rechecked_even_for_old_versions(application):
    data, repository, preview, components = application
    decision = repository.save_decision({"preview": preview}, arrays={"returns": np.array([[0., 0.], [.01, -.99]])})
    with pytest.raises(ValidationError, match="尺度断点"):
        validate_allocation_source({"kind": "taa", "decision_id": decision["id"]}, components,
                                   {"type": "manual", "weights": [.6, .4]}, "universe-frozen", data.data_dir)


@pytest.mark.parametrize('missing', ['descriptor', 'file'])
def test_missing_frozen_returns_block_export_and_direct_portfolio_entry(application, missing):
    from backend.tactical_allocation.service import TacticalAllocationService
    data, repository, preview, components = application
    arrays = {} if missing == 'descriptor' else {'returns': np.zeros((2, 2))}
    decision = repository.save_decision({'preview': preview, 'name': '输入不完整的历史版本'}, arrays=arrays)
    if missing == 'file':
        (repository.artifacts.root / decision['id'] / 'returns.npy').unlink()
    expected_code = 'TAA_SNAPSHOT_MISSING' if missing == 'descriptor' else 'RESEARCH_ARRAY_CORRUPT'
    with pytest.raises(ValidationError) as read_error:
        repository.decision_arrays(decision['id'])
    assert read_error.value.code == expected_code
    service = TacticalAllocationService(data.data_dir, data.data_dir)
    with pytest.raises(ValidationError) as export_error:
        service.product_allocation(decision['id'])
    assert export_error.value.code == expected_code
    with pytest.raises(ValidationError) as direct_error:
        validate_allocation_source({'kind': 'taa', 'decision_id': decision['id']}, components,
                                   {'type': 'manual', 'weights': [.6, .4]}, 'universe-frozen', data.data_dir)
    assert direct_error.value.code == expected_code
    # No repair, replacement calculation, or mutation of the immutable version.
    assert repository.get_decision(decision['id'])['content_hash'] == decision['content_hash']

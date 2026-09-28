"""Scenario-center linkage, PIT rejection, calibration reuse and frozen evidence."""
import copy
from datetime import date, timedelta
import json

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.custom_indicators.errors import ValidationError
from backend.strategic_allocation.routes import build_router
from backend.historical_regimes.v2_service import _stored_run_snapshot_hash
from backend.tests.cma_scenario_fixtures import create_scenario_fixture
from backend.tests.test_strategic_allocation import workspace, warm
from backend.tests.test_ltcma_statistics import request, publish


@pytest.fixture
def scenario(workspace):
    service, days = workspace
    service.warm()
    historical, realtime, reference = create_scenario_fixture(service, days)
    return service, days, historical, realtime, reference


def payload(case, method="long_term_scenario"):
    _, _, historical, realtime, reference = case
    patch = {"run_ref": {"id": historical["id"], "content_hash": historical["content_hash"]},
             "historical_reference": reference}
    if method == "conditional_scenario":
        patch.update(realtime_ref={"id": realtime["id"], "content_hash": realtime["content_hash"]}, horizon_days=21)
    return request(method, **patch)


def test_gapped_longterm_freezes_contiguous_bootstrap_evidence(scenario):
    import pandas as pd
    from backend.strategic_allocation.cma_application import frozen_numeric_inputs
    service, *_ = scenario
    path = service.data.data_dir / "asset_nv.parquet"
    frame = pd.read_parquet(path)
    frame.drop(index=80).to_parquet(path, index=False)
    saved = publish(service, payload(scenario), "longterm-gapped-bootstrap")
    audit = saved["model_result"]["model_audit"]
    mask = service.artifacts.arrays(saved["id"], ["evidence_period_contiguous"])["evidence_period_contiguous"]
    assert not mask.flags.writeable and np.count_nonzero(mask[1:] == 0) == 1
    assert audit["bootstrap"]["gap_count"] == 1
    assert audit["bootstrap"]["sampling"] == "disjoint_contiguous_blocks"
    assert audit["model_policy_version"] == "scenario-cma/1.0.2"
    _, _, widths = frozen_numeric_inputs(saved, service.artifacts)
    np.testing.assert_array_equal(widths, saved["model_result"]["mean_uncertainty"])
    assert not widths.flags.writeable
    assert service.get_cma(saved["id"]) == saved


def test_history_accepts_frequency_but_explains_future_data_without_hiding_runs(scenario):
    service, _, historical, _, _ = scenario
    path = service.cma.evidence.regime_root / "historical_regime_runs.json"
    rows = json.loads(path.read_text())["items"]
    monthly = {**copy.deepcopy(historical), "id": "monthly", "frequency": "monthly", "publications": []}
    monthly["content_hash"] = _stored_run_snapshot_hash(monthly)
    future = {**copy.deepcopy(historical), "id": "future", "as_of": str(date.today() + timedelta(days=1)), "publications": []}
    future["content_hash"] = _stored_run_snapshot_hash(future)
    path.write_text(json.dumps({"items": [*rows, monthly, future]}))
    options = service.cma.study_options(date.today())["scenario_options"]
    assert options["default_historical_id"] is None
    refs = {row["id"]: row for row in options["historical_references"]}
    assert refs["monthly"]["available"] is True
    assert refs["monthly"]["reasons"] == []
    regime_options = {row["id"]: row for row in service.cma.study_options(date.today())["regime_runs"]}
    assert regime_options["monthly"]["available"] is True
    assert regime_options["future"]["available"] is False
    assert "研究日之后" in refs["future"]["reasons"][0]["message"]
    assert options["realtime_runs"][0]["available"] is True


def test_exact_reference_is_required_but_archive_date_does_not_block_research(scenario):
    service, _, _, _, _ = scenario
    req = payload(scenario)
    evidence = {"dates": [str(date.today())]}
    without = req.model.model_copy(update={"historical_reference": None})
    with pytest.raises(ValidationError, match="发布版本"):
        service.cma.evidence.scenario(without, evidence)
    raw = json.loads((service.cma.evidence.regime_root / "historical_regime_runs.json").read_text())
    raw["items"][0]["publications"][0]["published_at"] = str(date.today() + timedelta(days=1))
    (service.cma.evidence.regime_root / "historical_regime_runs.json").write_text(json.dumps(raw))
    service.cma.evidence.scenario(req.model, evidence)
    assert evidence["regime"][2]["reference"] == scenario[-1]


def test_every_publication_remains_reachable_for_exact_realtime_binding(scenario):
    from backend.historical_regimes.v2_contracts import parse_definition_v2, definition_content_hash
    from backend.historical_regimes.v2_service import _content_hash
    from backend.historical_regimes.reliability.execution import model_binding_hash

    service, days, historical, realtime, first_reference = scenario
    graph = service.cma.evidence.scenarios.graph
    second_publication = {**historical["publications"][0], "id": "pub-second-reference",
                          "usage": "product_research"}
    graph.runs.add_publications(historical["id"], [second_publication])
    second_reference = {**first_reference, "publication_id": second_publication["id"]}

    # Rebind the full frozen recognition/calibration chain, not just options metadata.
    stored_definition = graph.get_definition(realtime["definition_id"], 1)
    stored_definition["study"]["reference"] = second_reference
    definition = parse_definition_v2(stored_definition)
    changed_realtime = copy.deepcopy(realtime)
    changed_realtime["definition"] = definition.model_dump(mode="json")
    changed_realtime["definition_snapshot_hash"] = definition_content_hash(definition)
    changed_realtime["content_hash"] = _stored_run_snapshot_hash(changed_realtime)
    changed_realtime["publications"][0]["run_content_hash"] = changed_realtime["content_hash"]
    path = service.cma.evidence.regime_root / "historical_regime_runs.json"
    data = json.loads(path.read_text())
    data["items"] = [changed_realtime if row["id"] == realtime["id"] else row for row in data["items"]]
    path.write_text(json.dumps(data))
    store = graph.reliability._store(stored_definition["study"]["calibration_id"])
    with store.locked():
        data = store.read_unlocked()
        artifact = data["items"][0]
        artifact["request"]["reference"] = second_reference
        artifact["report"]["lineage"]["model_binding_hash"] = model_binding_hash(definition)
        artifact["content_hash"] = _content_hash({key: value for key, value in artifact.items() if key != "content_hash"})
        store.write_unlocked(data)

    options = service.cma.study_options(date.today())["scenario_options"]
    references = [row for row in options["historical_references"] if row["id"] == historical["id"]]
    assert [row["reference"] for row in references] == [first_reference, second_reference]
    assert all(row["available"] for row in references)
    assert len({row["name"] for row in references}) == 2
    assert all(row["reference"]["publication_id"] not in row["name"] for row in references)
    assert options["default_historical_id"] is None
    current = next(row for row in options["realtime_runs"] if row["id"] == realtime["id"])
    assert current["available"] is True
    assert current["reference"] == references[1]["reference"]

    model = payload(scenario, "conditional_scenario").model
    model = model.model_copy(update={
        "historical_reference": model.historical_reference.model_copy(update=second_reference),
        "realtime_ref": model.realtime_ref.model_copy(update={"content_hash": changed_realtime["content_hash"]}),
    })
    evidence = {"dates": [str(day.date()) for day in days[1:]]}
    service.cma.evidence.scenario(model, evidence)
    assert evidence["forecast_evidence"]["reference_ref"] == second_reference
    np.testing.assert_allclose(evidence["forecast_evidence"]["current_probabilities"], [.8, .2])


def test_late_publications_and_unpublished_research_remain_usable(scenario):
    service, _, historical, _, first_reference = scenario
    graph = service.cma.evidence.scenarios.graph
    future_publication = {**historical["publications"][0], "id": "pub-future-reference",
                          "published_at": str(date.today() + timedelta(days=1)) + "T00:00:00Z"}
    graph.runs.add_publications(historical["id"], [future_publication])
    old = {**copy.deepcopy(historical), "id": "unpublished-historical-run", "publications": []}
    old["content_hash"] = _stored_run_snapshot_hash(old)
    path = service.cma.evidence.regime_root / "historical_regime_runs.json"
    data = json.loads(path.read_text())
    data["items"].append(old)
    path.write_text(json.dumps(data))

    options = service.cma.study_options(date.today())["scenario_options"]
    references = [row for row in options["historical_references"] if row["id"] == historical["id"]]
    assert references[0]["reference"] == first_reference
    assert references[0]["available"] is True
    assert references[1]["reference"]["publication_id"] == future_publication["id"]
    assert references[1]["available"] is True
    assert references[1]["reasons"] == []
    saved = next(row for row in options["historical_references"] if row["id"] == old["id"])
    assert saved["source_kind"] == "saved_run"
    assert saved["reference"] is None
    assert saved["available"] is True


def test_calibrated_current_evidence_uses_shared_njit_and_matching_reference(scenario):
    service, days, _, _, reference = scenario
    evidence = {"dates": [str(day.date()) for day in days[1:]]}
    request_model = payload(scenario, "conditional_scenario").model
    service.cma.evidence.scenario(request_model, evidence)
    current = evidence["forecast_evidence"]
    np.testing.assert_allclose(current["current_probabilities"], [.8, .2])
    assert not current["current_probabilities"].flags.writeable
    assert current["reference_ref"] == reference
    assert current["forecast_validation"]["downstream_eligible"] is False
    assert current["calibration"]["method"] == "class_frequency"
    changed = request_model.model_copy(update={"historical_reference": request_model.historical_reference.model_copy(
        update={"publication_id": "wrong-publication"})})
    with pytest.raises(ValidationError, match="发布记录"):
        service.cma.evidence.scenario(changed, evidence)


def test_current_state_cannot_silently_reuse_stale_or_unavailable_dates(scenario):
    service, days, _, _, _ = scenario
    model = payload(scenario, "conditional_scenario").model
    evidence = {"dates": [str((days[-1] + timedelta(days=1)).date())]}
    with pytest.raises(ValidationError, match="尚未更新"):
        service.cma.evidence.scenario(model, evidence)


def test_calibration_is_verified_from_frozen_report_not_client_boolean(scenario):
    service, days, _, realtime, _ = scenario
    calibration_id = realtime["definition"]["study"]["calibration_id"]
    store = service.cma.evidence.scenarios.graph.reliability._store(calibration_id)
    with store.locked():
        data = store.read_unlocked()
        data["items"][0]["report"]["calibration"]["parameters"]["counts"][0] = [100., 0.]
        store.write_unlocked(data)
    with pytest.raises(ValidationError, match="验证或版本绑定"):
        service.cma.evidence.scenario(payload(scenario, "conditional_scenario").model,
                                     {"dates": [str(days[-1].date())]})


def test_unverified_probability_mass_cannot_enter_conditional_forecast(scenario):
    from backend.historical_regimes.v2_service import _content_hash
    service, days, _, realtime, _ = scenario
    store = service.cma.evidence.scenarios.graph.reliability._store(realtime["definition"]["study"]["calibration_id"])
    with store.locked():
        data = store.read_unlocked()
        item = data["items"][0]
        item["report"]["verification"]["verified_states"] = ["growth"]
        item["content_hash"] = _content_hash({key: value for key, value in item.items() if key != "content_hash"})
        store.write_unlocked(data)
    with pytest.raises(ValidationError, match="尚未通过独立验证"):
        service.cma.evidence.scenario(payload(scenario, "conditional_scenario").model,
                                     {"dates": [str(days[-1].date())]})


def test_options_http_filters_by_research_date_and_returns_human_reason(scenario):
    service, _, _, _, _ = scenario
    app = FastAPI(); app.include_router(build_router(service))
    client = TestClient(app)
    earlier = str(date.today() - timedelta(days=20))
    response = client.get("/api/strategic-allocation/cma/study-options", params={"as_of": earlier})
    assert response.status_code == 200, response.text
    options = response.json()["scenario_options"]
    assert options["default_historical_id"] is None
    assert "研究日之后" in options["historical_references"][0]["reasons"][0]["message"]
    assert client.get("/api/strategic-allocation/cma/study-options", params={"as_of": "bad"}).status_code == 422


def test_longterm_full_calculation_and_publish_freezes_exact_reference(scenario):
    service, _, _, _, reference = scenario
    saved = publish(service, payload(scenario), "scenario-longterm-validated-chain")
    assert saved["definition"]["model"]["historical_reference"] == reference
    assert service.get_cma(saved["id"]) == saved
    audit = saved["model_result"]["model_audit"]
    assert audit["regime"]["reference"] == reference


def test_conditional_full_calculation_is_saved_research_not_selectable_for_saa(scenario):
    service, _, _, _, _ = scenario
    saved = publish(service, payload(scenario, "conditional_scenario"), "scenario-conditional-validated-chain")
    assert saved["model_result"]["method"] == "conditional_scenario"
    with pytest.raises(ValidationError, match="条件|SAA|长期"):
        service.cma.require_selectable(saved["id"])
    row = next(row for row in service.cma.list()["items"] if row["id"] == saved["id"])
    assert row["downstream_eligible"] is False


def _save_current_fixture(service, current):
    """Change only an isolated fixture, keeping its analytical hash consistent."""
    current["content_hash"] = _stored_run_snapshot_hash(current)
    path = service.cma.evidence.regime_root / "historical_regime_runs.json"
    data = json.loads(path.read_text())
    data["items"] = [current if row["id"] == current["id"] else row for row in data["items"]]
    path.write_text(json.dumps(data))
    return path


@pytest.mark.parametrize("usage", [None, "research_display"])
def test_conditional_research_does_not_require_taa_publication(scenario, usage):
    from custom_indicators.errors import ValidationError as RegimeValidationError

    service, _, _, realtime, _ = scenario
    current = copy.deepcopy(realtime)
    current["publications"] = ([] if usage is None else [
        {**current["publications"][0], "usage": usage, "gate": "research_with_recorded_restrictions"}])
    path = _save_current_fixture(service, current)
    assert current["content_hash"] == realtime["content_hash"]
    before = path.read_bytes()
    app = FastAPI(); app.include_router(build_router(service))
    client = TestClient(app)
    response = client.get("/api/strategic-allocation/cma/study-options",
                          params={"section": "scenarios", "as_of": str(date.today())})
    assert response.status_code == 200, response.text
    option = next(row for row in response.json()["scenario_options"]["realtime_runs"]
                  if row["id"] == current["id"])
    assert option["available"] is True and option["reasons"] == []
    saved = publish(service, payload(scenario, "conditional_scenario"), f"conditional-without-taa-{usage}")
    assert saved["model_result"]["method"] == "conditional_scenario"
    assert saved["definition"]["model"]["realtime_ref"]["content_hash"] == current["content_hash"]
    with pytest.raises(ValidationError, match="条件|SAA"):
        service.cma.require_selectable(saved["id"])
    with pytest.raises(RegimeValidationError) as caught:
        service.cma.evidence.scenarios.graph.resolve_taa_run(current["id"])
    assert caught.value.code == "TAA_RUN_NOT_PUBLISHED"
    assert path.read_bytes() == before  # Reading/publishing CMA never republishes a source.


@pytest.mark.parametrize("issue, message", [
    ("is_causal", "因果识别检查"),
    ("uses_future_data", "因果识别检查"),
    ("repaints", "因果识别检查"),
    ("realtime_eligible", "因果识别检查"),
    ("future_cutoff", "研究日之后的数据"),
    ("monthly", "日频的当前市场研究"),
    ("calibration_missing", "没有概率校准结果"),
    ("tampered", "保存内容不完整"),
])
def test_unpublished_current_research_still_checks_its_own_evidence(scenario, issue, message):
    service, days, _, realtime, _ = scenario
    current = copy.deepcopy(realtime)
    current["publications"] = []
    if issue in {"is_causal", "uses_future_data", "repaints", "realtime_eligible"}:
        current["causality"][issue] = issue in {"uses_future_data", "repaints"}
    elif issue == "future_cutoff":
        current["as_of"] = str(date.today() + timedelta(days=1))
    elif issue == "monthly":
        current["frequency"] = "monthly"
    elif issue == "calibration_missing":
        current["definition"]["study"]["calibration_id"] = None
    path = _save_current_fixture(service, current)
    if issue == "tampered":
        data = json.loads(path.read_text())
        next(row for row in data["items"] if row["id"] == current["id"])["name"] = "tampered"
        path.write_text(json.dumps(data))
    options = service.cma.study_options(date.today(), "scenarios")["scenario_options"]
    option = next(row for row in options["realtime_runs"] if row["id"] == current["id"])
    assert option["available"] is False
    assert message in option["reasons"][0]["message"]
    assert "TAA" not in option["reasons"][0]["message"]
    model = payload(scenario, "conditional_scenario").model
    model = model.model_copy(update={"realtime_ref": model.realtime_ref.model_copy(
        update={"content_hash": current["content_hash"]})})
    with pytest.raises(ValidationError, match=message):
        service.cma.evidence.scenario(model, {"dates": [str(days[-1].date())]})


def test_options_read_each_run_store_once_and_recheck_next_request(scenario, monkeypatch):
    service, _, historical, _, _ = scenario
    expected = service.cma.evidence.scenario_options(date.today())
    store = service.cma.evidence.runs.store
    original = store.read_unlocked
    reads = []
    def counted():
        reads.append(1)
        return original()
    monkeypatch.setattr(store, 'read_unlocked', counted)
    assert service.cma.study_options(date.today(), 'scenarios') == {'scenario_options': expected}
    assert len(reads) == 1
    rows = json.loads(store.path.read_text())
    next(row for row in rows['items'] if row['id'] == historical['id'])['name'] = 'tampered'
    store.path.write_text(json.dumps(rows))
    changed = service.cma.study_options(date.today(), 'scenarios')['scenario_options']
    assert len(reads) == 2
    assert not next(row for row in changed['historical_references'] if row['id'] == historical['id'])['available']


def test_monthly_intervals_apply_to_daily_returns_without_guessing_boundaries(scenario):
    service, days, historical, _, _ = scenario
    raw = copy.deepcopy(historical)
    raw.update(id="monthly-intervals", frequency="monthly", publications=[])
    raw["series"] = [dict(raw["series"][i], state_id=state) for i, state in
                     [(0, "growth"), (20, "growth"), (40, "stress"), (60, "stress")]]
    raw["content_hash"] = _stored_run_snapshot_hash(raw)
    path = service.cma.evidence.regime_root / "historical_regime_runs.json"
    data = json.loads(path.read_text())
    data["items"].append(raw)
    path.write_text(json.dumps(data))
    model = payload(scenario).model
    model = model.model_copy(update={"run_ref": model.run_ref.model_copy(update={
        "id": raw["id"], "content_hash": raw["content_hash"]}), "historical_reference": None})
    evidence = {"dates": [str(day.date()) for day in days[1:71]]}
    service.cma.evidence.scenario(model, evidence)
    states, state_ids, audit = evidence["regime"]
    assert state_ids == ["growth", "stress"]
    np.testing.assert_array_equal(states, [0] * 20 + [-1] * 19 + [1] * 21 + [-1] * 10)
    assert audit["unknown_observations"] == 29
    assert audit["source_frequency"] == "monthly"
    assert audit["label_available_dates"][0] == str(days[20].date())
    assert audit["label_available_dates"][20] is None
    assert audit["historical_pit_proven"] is False
    assert not states.flags.writeable


@pytest.mark.parametrize("field", ["available_at", "recognized_at"])
def test_historical_interval_with_future_information_cannot_be_truncated_to_pit(scenario, field):
    service, _, historical, _, _ = scenario
    path = service.cma.evidence.regime_root / "historical_regime_runs.json"
    data = json.loads(path.read_text())
    raw = data["items"][0]
    raw["series"][-1][field] = str(date.today() + timedelta(days=1))
    raw["content_hash"] = _stored_run_snapshot_hash(raw)
    raw["publications"] = []
    path.write_text(json.dumps(data))
    options = service.cma.study_options(date.today())
    for rows in (options["regime_runs"], options["scenario_options"]["historical_references"]):
        entry = next(row for row in rows if row["id"] == historical["id"])
        assert entry["available"] is False
        assert "研究日之后的信息" in entry["reasons"][0]["message"]


def test_conditional_accepts_late_archive_but_rejects_future_calibration(scenario):
    from backend.historical_regimes.v2_service import _content_hash
    service, days, _, realtime, _ = scenario
    path = service.cma.evidence.regime_root / "historical_regime_runs.json"
    data = json.loads(path.read_text())
    for raw in data["items"]:
        for publication in raw["publications"]:
            publication["published_at"] = str(date.today() + timedelta(days=1))
    path.write_text(json.dumps(data))
    model = payload(scenario, "conditional_scenario").model
    evidence = {"dates": [str(days[-1].date())]}
    service.cma.evidence.scenario(model, evidence)
    np.testing.assert_allclose(evidence["forecast_evidence"]["current_probabilities"], [.8, .2])
    store = service.cma.evidence.scenarios.graph.reliability._store(realtime["definition"]["study"]["calibration_id"])
    with store.locked():
        data = store.read_unlocked()
        item = data["items"][0]
        item["report"]["calibration"]["available_from"] = str(date.today() + timedelta(days=1))
        item["content_hash"] = _content_hash({key: value for key, value in item.items() if key != "content_hash"})
        store.write_unlocked(data)
    with pytest.raises(ValidationError, match="校准结果在研究日尚不可用"):
        service.cma.evidence.scenario(model, evidence)

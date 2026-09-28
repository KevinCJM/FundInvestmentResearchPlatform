"""Boundary and handoff probes beyond the basic numerical reference suite."""
import copy
from datetime import date

import numpy as np
import pandas as pd
import pytest

from backend.custom_indicators.errors import ValidationError
from backend.strategic_allocation import cma_model_kernels, cma_statistical_kernels, reference_evidence_kernels
from backend.strategic_allocation.contracts import CmaRequest, PolicyRequest
from backend.strategic_allocation.cma_center_contracts import CmaCenterPublish
from backend.tests.test_strategic_allocation import workspace, warm, definition, saved_inputs
from backend.tests.test_ltcma_statistics import request, publish


@pytest.fixture(scope="module", autouse=True)
def warmed_statistics():
    cma_model_kernels.warm()
    reference_evidence_kernels.warm()
    cma_statistical_kernels.warm()


def modern_manual(**patch):
    return CmaRequest.model_validate({**definition().model_dump(mode="json"),
        "schema_version": "2.0", "moment_semantics": "annualized_periodic_arithmetic",
        "fee_basis": "source_embedded_no_additional_fee", "fx_hedging_basis": "same_currency_no_conversion", **patch})


def test_list_distinguishes_frozen_history_without_reloading_sources(workspace, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from backend.strategic_allocation.routes import build_router

    service, days = workspace
    whole = publish(service, request(), "list-full-history")
    recent = publish(service, request(window={"kind": "custom", "start_date": str(days[100].date()),
                                             "end_date": str(days[-1].date())}), "list-recent-history")
    monkeypatch.setattr(service.cma.evidence, "build", lambda *_: pytest.fail("list must use frozen evidence"))
    monkeypatch.setattr(service.data, "_configuration", lambda *_: pytest.fail("list must not reload product names"))
    app = FastAPI(); app.include_router(build_router(service))
    response = TestClient(app).get("/api/strategic-allocation/cma?method=historical_statistics")
    assert response.status_code == 200
    rows = {row["id"]: row for row in response.json()["items"]}
    # 同一资产范围下的两份研究靠名称区分，冻结历史仍各自独立。
    assert rows[whole["id"]]["scope_name"] == rows[recent["id"]]["scope_name"]
    assert rows[whole["id"]]["name"] != rows[recent["id"]]["name"]
    assert rows[whole["id"]]["history"]["observations"] == 160
    assert rows[recent["id"]]["history"]["observations"] == 60
    for saved in (whole, recent):
        summary = rows[saved["id"]]["history"]
        assert summary["window"] == saved["definition"]["model"]["window"]
        assert summary["start_date"] == saved["model_result"]["model_audit"]["evidence"]["actual_start"]
        assert summary["end_date"] == str(days[-1].date())
        assert summary["source_names"] == [p["name"] for a in saved["source_snapshot"]["assets"] for p in a["products"]]
        assert service.get_cma(saved["id"]) == saved


def dated_request(method, as_of, **params):
    payload = request(method, **params).model_dump(mode="json")
    payload["as_of"] = str(as_of)
    payload["model"]["as_of"] = str(as_of)
    return CmaRequest.model_validate(payload)


def test_prior_continuation_equals_one_update_and_rejects_overlap(workspace):
    service, days = workspace
    d0, d1, d2 = [str(days[i].date()) for i in (30, 100, 160)]
    prior = publish(service, modern_manual(as_of=d0), "prior-continuation-1")
    first = dated_request("bayesian_niw", d1,
        prior_ref={"id": prior["id"], "content_hash": prior["content_hash"]},
        window={"kind": "custom", "start_date": d0, "end_date": d1},
        mean_prior_observations=20., covariance_prior_observations=30.)
    first_saved = publish(service, first, "prior-continuation-2")
    second = dated_request("bayesian_niw", d2,
        prior_ref={"id": first_saved["id"], "content_hash": first_saved["content_hash"]}, prior_mode="continue",
        window={"kind": "custom", "start_date": d1, "end_date": d2})
    updated = service.preview_cma(second)
    single = dated_request("bayesian_niw", d2,
        prior_ref={"id": prior["id"], "content_hash": prior["content_hash"]},
        window={"kind": "custom", "start_date": d0, "end_date": d2},
        mean_prior_observations=20., covariance_prior_observations=30.)
    expected = service.preview_cma(single)
    np.testing.assert_allclose(updated["effective_returns"], expected["effective_returns"], rtol=1e-12)
    np.testing.assert_allclose(updated["effective_covariance"], expected["effective_covariance"], rtol=1e-12)
    invalid = second.model_dump(mode="json")
    invalid["model"]["window"]["start_date"] = str(days[99].date())
    with pytest.raises(ValidationError, match="新收益证据"):
        service.preview_cma(CmaRequest.model_validate(invalid))


def test_legacy_prior_requires_explicit_new_basis(workspace):
    service, _ = workspace
    prior = publish(service, definition(), "legacy-prior-probe")
    candidate = request("bayesian_niw", prior_ref={"id": prior["id"], "content_hash": prior["content_hash"]},
                        mean_prior_observations=20., covariance_prior_observations=20.)
    with pytest.raises(ValidationError, match="确认口径"):
        service.preview_cma(candidate)


def test_saa_uses_frozen_statistical_moments_without_retraining(workspace, monkeypatch):
    service, _ = workspace
    mandate, _, _ = saved_inputs(service)
    cma = publish(service, request(), "historical-for-saa")
    monkeypatch.setattr(service.cma, "calculation", lambda *_: pytest.fail("SAA must not refit LTCMA"))
    preview = service.preview_policy(PolicyRequest(mandate_id=mandate["id"], cma_id=cma["id"], candidate_count=300))
    np.testing.assert_allclose(preview["covariance"], cma["effective_covariance"])
    assert preview["cma_hash"] == cma["content_hash"]
    assert len(preview["candidates"]) == 6  # 四类代表组合 + 最大夏普 + 最小模拟回撤


def test_new_manual_cash_does_not_invent_variance(workspace):
    service, _ = workspace
    payload = modern_manual().model_dump(mode="json")
    payload["assets"][1].update(role="liquidity", annual_volatility=0.)
    payload["correlation"] = [[1., 0.], [0., 1.]]
    preview = service.preview_cma(CmaRequest.model_validate(payload))
    assert preview["covariance"][1] == [0., 0.]
    assert preview["covariance"][0][1] == 0.
    invalid = copy.deepcopy(payload)
    invalid["assets"][1]["role"] = "growth"
    with pytest.raises(ValueError, match="现金角色"):
        CmaRequest.model_validate(invalid)
    invalid = copy.deepcopy(payload); invalid["schema_version"] = "1.0"
    with pytest.raises(ValueError):
        CmaRequest.model_validate(invalid)


def test_new_confirmation_is_strict_and_cannot_use_legacy_service_write(workspace):
    service, _ = workspace
    req = modern_manual()
    preview = service.preview_cma(req)
    with pytest.raises(ValueError):
        CmaCenterPublish(request=req, preview_hash=preview["preview_hash"], confirm=1, idempotency_key="bad-confirm-value")
    from backend.strategic_allocation.contracts import PublishCmaRequest
    with pytest.raises(ValidationError, match="明确确认"):
        service.publish_cma(PublishCmaRequest(request=req, preview_hash=preview["preview_hash"]))


def test_stale_calendar_cannot_shorten_the_requested_window(workspace):
    service, days = workspace
    calendar_path = service.data.data_dir / "trade_day_df.parquet"
    calendar = pd.read_parquet(calendar_path)
    calendar.iloc[:-5].to_parquet(calendar_path, index=False)
    nav_path = service.data.data_dir / "asset_nv.parquet"
    nav = pd.read_parquet(nav_path)
    nav.loc[nav.date <= days[-6]].to_parquet(nav_path, index=False)
    with pytest.raises(ValidationError, match="日历.*覆盖"):
        service.preview_cma(request())
    assert not service.artifacts.root.exists()


def test_calendar_can_cover_a_closed_requested_end(workspace):
    from backend.strategic_allocation.cma_evidence import _calendar
    from datetime import timedelta
    service, days = workspace
    path = service.data.data_dir / "trade_day_df.parquet"
    calendar = pd.read_parquet(path)
    closed = days[-1].date() + timedelta(days=1)
    calendar = pd.concat([calendar, pd.DataFrame([{
        "exchange": "SSE", "cal_date": int(closed.strftime("%Y%m%d")), "is_open": 0,
    }])], ignore_index=True)
    calendar.to_parquet(path, index=False)
    observed = _calendar(service.data.data_dir, through=closed)
    assert observed[-1] == np.datetime64(days[-1].date(), "D").astype(np.int64)
    assert observed.size == len(days)


def test_model_axis_errors_are_validation_errors_not_key_errors():
    payload = request().model_dump(mode="json")
    payload["model"]["asset_ids"] = ["other", "债券"]
    with pytest.raises(ValueError):
        CmaRequest.model_validate(payload)


@pytest.mark.parametrize("method", ["manual", "historical_statistics"])
def test_optional_ltcma_notes_do_not_change_calculation_or_frozen_sources(workspace, method):
    service, _ = workspace
    original = modern_manual() if method == "manual" else request()
    before = service.preview_cma(original)
    raw = original.model_dump(mode="json")
    raw["source"] = ""
    for asset in raw["assets"]:
        asset["rationale"] = ""
    if raw["model"]:
        raw["model"]["source"] = ""
    candidate = CmaRequest.model_validate(raw)
    result = service.preview_cma(candidate)
    np.testing.assert_array_equal(result["covariance"], before["covariance"])
    if method != "manual":
        np.testing.assert_array_equal(result["effective_returns"], before["effective_returns"])
    saved = publish(service, candidate, f"optional-notes-{method}")
    restored = service.get_cma(saved["id"])
    assert restored["definition"]["source"] == ""
    assert all(a["rationale"] == "" for a in restored["definition"]["assets"])
    assert restored["source_snapshot"] == before["source_snapshot"]
    assert original.source


def test_legacy_cma_still_requires_original_annotations():
    from pydantic import ValidationError as ContractError
    raw = definition().model_dump(mode="json")
    raw["source"] = ""
    with pytest.raises(ContractError, match="旧版 CMA"):
        CmaRequest.model_validate(raw)


@pytest.mark.parametrize("method", ["manual", "historical_statistics"])
def test_retired_horizon_does_not_change_moments_or_new_saved_contract(workspace, method):
    service, _ = workspace
    original = modern_manual() if method == "manual" else request()
    raw = original.model_dump(mode="json")
    assert "horizon_years" not in raw
    expected = service.preview_cma(original)
    for years in (5, 10):
        candidate = CmaRequest.model_validate({**raw, "horizon_years": years})
        assert candidate.model_dump() == original.model_dump()
        actual = service.preview_cma(candidate)
        assert actual == expected
    saved = publish(service, original, f"without-horizon-{method}")
    assert "horizon_years" not in saved["definition"]
    assert "horizon_years" not in service.cma.list()["items"][0]
    assert "horizon_years" not in service.catalog()["assumptions"][0]
    assert "horizon_years" not in CmaRequest.model_json_schema()["properties"]


def test_old_horizon_snapshot_remains_readable_and_usable_for_saa_and_niw(workspace):
    service, _ = workspace
    mandate, _, _ = saved_inputs(service)
    payload, arrays = service.cma.calculation(modern_manual())
    payload["definition"]["horizon_years"] = 5
    old = service.artifacts.save("series", {**payload, "artifact_type": "capital_market_assumptions",
                                           "name": "旧五年标签", "research_only": True}, arrays)
    result = service.preview_policy(PolicyRequest(mandate_id=mandate["id"], cma_id=old["id"], candidate_count=300))
    np.testing.assert_array_equal(result["covariance"], old["covariance"])
    niw = request("bayesian_niw", prior_ref={"id": old["id"], "content_hash": old["content_hash"]},
                  mean_prior_observations=20., covariance_prior_observations=20.)
    assert service.preview_cma(niw)["model_result"]["method"] == "bayesian_niw"
    assert service.get_cma(old["id"]) == old
    assert "horizon_years" not in CmaRequest.model_validate(old["definition"]).model_dump()

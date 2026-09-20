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
        "fee_basis": "explicit_assumption", "fx_hedging_basis": "explicit_assumption", **patch})


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
    assert len(preview["candidates"]) == 4


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

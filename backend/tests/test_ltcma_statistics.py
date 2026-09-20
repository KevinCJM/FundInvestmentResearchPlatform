"""Independent numerical references and real service consumption for LTCMA methods."""
from datetime import date
import copy
import json

import numpy as np
import pandas as pd
import pytest
from scipy.stats import t as student_t

from backend.custom_indicators.errors import ValidationError
from backend.strategic_allocation import cma_model_kernels as old_numeric
from backend.strategic_allocation import cma_statistical_kernels as numeric
from backend.strategic_allocation import reference_evidence_kernels
from backend.strategic_allocation.cma_center_contracts import CmaCenterPublish
from backend.strategic_allocation.cma_application import frozen_assumptions, frozen_numeric_inputs
from backend.strategic_allocation.contracts import CmaRequest
from backend.tests.test_strategic_allocation import workspace, warm, definition


@pytest.fixture(scope="module", autouse=True)
def statistics_warm():
    old_numeric.warm()
    reference_evidence_kernels.warm()
    assert numeric.warm()["complete"]


def request(method="historical_statistics", **patch):
    value = definition().model_dump(mode="json")
    value.update(schema_version="2.0", moment_semantics="annualized_periodic_arithmetic",
                 fee_basis="source_embedded_no_additional_fee", fx_hedging_basis="same_currency_no_conversion")
    value["assets"] = [{**a, "annual_return": None, "annual_volatility": None, "mean_uncertainty": 0.}
                       for a in value["assets"]]
    value["correlation"] = None
    value["model"] = {"method": method, "asset_ids": [a["id"] for a in value["assets"]],
                      "as_of": value["as_of"], "currency": "CNY", "source": "固定历史统计证据",
                      "window": {"kind": "common_since_inception"}, **patch}
    return CmaRequest.model_validate(value)


def publish(service, req, key):
    preview = service.preview_cma(req)
    return service.publish_cma(CmaCenterPublish(request=req, preview_hash=preview["preview_hash"],
        confirm=True, idempotency_key=key))


@pytest.mark.parametrize("df", [2.01, 3., 23., 159., 5000., 100000.])
@pytest.mark.parametrize("p", [.5, .9, .975, .999])
def test_t_quantile_against_scipy(df, p):
    assert numeric.student_t_quantile(p, df) == pytest.approx(student_t.ppf(p, df), rel=2e-8, abs=2e-9)


def test_historical_iid_covariance_and_readonly_strides():
    base = np.random.default_rng(4).normal(.0002, .005, size=(160, 6))
    before = base.copy()
    panel = base[::2, ::2]; panel.flags.writeable = False
    assert np.shares_memory(panel, base)
    means, risk, uncertainty, half = numeric.historical_estimate(panel, .2)
    sample = np.cov(panel, rowvar=False, ddof=1)
    expected = 252 * (.8 * sample + .2 * np.diag(np.diag(sample)))
    np.testing.assert_allclose(means, panel.mean(0) * 252, rtol=1e-12)
    np.testing.assert_allclose(risk, expected, rtol=1e-12)
    np.testing.assert_allclose(uncertainty, expected * 252 / len(panel), rtol=1e-12)
    np.testing.assert_allclose(half, student_t.ppf(.975, len(panel) - 1) * np.sqrt(np.diag(uncertainty)), rtol=1e-10)
    np.testing.assert_array_equal(base, before)


def test_niw_analytic_update_and_annualization():
    returns = np.random.default_rng(5).normal(.0003, .01, (100, 3))
    prior_mean = np.array([.0002, .0001, .0004])
    psi = np.diag([.002, .003, .004]); kappa, nu = 10., 24.
    result = numeric.niw_update(returns, prior_mean, psi, kappa, nu)
    n = len(returns); sample = returns.mean(0); kt = kappa + n; nut = nu + n
    expected_mean = (kappa * prior_mean + n * sample) / kt
    delta = sample - prior_mean
    expected_psi = psi + (returns - sample).T @ (returns - sample) + kappa * n / kt * np.outer(delta, delta)
    base_risk = expected_psi / (nut - 3 - 1)
    np.testing.assert_allclose(result[0], 252 * expected_mean, rtol=1e-12)
    np.testing.assert_allclose(result[1], 252 * base_risk, rtol=1e-12)
    np.testing.assert_allclose(result[2], 252**2 * base_risk / kt, rtol=1e-12)
    df = nut - 3 + 1
    expected_half = 252 * student_t.ppf(.975, df) * np.sqrt(np.diag(expected_psi) / (kt * df))
    np.testing.assert_allclose(result[3], expected_half, rtol=1e-10)
    np.testing.assert_allclose(result[5], expected_psi, rtol=1e-12)
    assert result[6:] == (kt, nut)


def test_state_ml_identity_and_unknown_exclusion():
    returns = np.random.default_rng(9).normal(.0002, .009, (160, 3))
    states = np.arange(160, dtype=np.int64) % 3
    states[::17] = -1
    before = returns.copy()
    counts, means, covariances = numeric.conditional_state_moments(returns, states, 3, 0.)
    probabilities = numeric.occupancy_probabilities(counts)
    mean, covariance, within, between = old_numeric.mixture_moments_kernel(probabilities, means, covariances, False)
    reference = returns[states >= 0]
    np.testing.assert_allclose(mean, reference.mean(0), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(covariance, np.cov(reference, rowvar=False, ddof=0), rtol=1e-12, atol=1e-14)
    annual_mean, annual_cov = numeric.annualize_moments(mean, covariance, 252)
    np.testing.assert_allclose(annual_cov, 252 * (within + between), rtol=1e-12)
    np.testing.assert_array_equal(before, returns)
    assert not np.allclose(annual_cov, 252 * within + 252**2 * between, rtol=1e-8)


def test_cash_and_invalid_covariance():
    vol, corr, eig = numeric.statistical_covariance_diagnostics(np.diag([0., .04]))
    assert vol.tolist() == [0., .2]
    assert np.isnan(corr[0]).all()
    assert corr[1, 1] == pytest.approx(1.)
    for invalid in (np.array([[1., 2.], [2., 1.]]), np.array([[0., .1], [.1, .04]])):
        with pytest.raises(ValueError):
            numeric.statistical_covariance_diagnostics(invalid)


@pytest.mark.parametrize("shape", [(0, 2), (19, 2)])
def test_short_history_rejected(shape):
    with pytest.raises(ValueError):
        numeric.historical_estimate(np.zeros(shape), .1)


def test_nonfinite_and_no_request_compilation():
    panel = np.zeros((30, 2)); panel[2, 1] = np.nan
    with pytest.raises(ValueError):
        numeric.historical_estimate(panel, 0.)
    signatures = {k.__name__: list(k.signatures) for k in numeric.KERNELS}
    numeric.historical_estimate(np.random.default_rng(2).normal(0, .01, (50, 2))[::2], 0.)
    assert signatures == {k.__name__: list(k.signatures) for k in numeric.KERNELS}


def test_real_historical_preview_frozen_publish_and_consumption(workspace):
    service, _ = workspace
    req = request()
    preview = service.preview_cma(req)
    assert not service.artifacts.root.exists()
    assert preview["model_result"]["model_audit"]["evidence"]["observations"] == 160
    result = publish(service, req, "history-service-check")
    frozen = frozen_assumptions(result)
    means, risk, widths = frozen_numeric_inputs(result, service.artifacts)
    assert isinstance(risk, np.memmap) and not risk.flags.writeable
    np.testing.assert_allclose(means, [a["annual_return"] for a in frozen["assets"]])
    np.testing.assert_allclose(widths, [a["mean_uncertainty"] for a in frozen["assets"]])
    assert "evidence_returns" in result["arrays"]


def test_history_window_is_not_silently_shortened(workspace):
    service, _ = workspace
    with pytest.raises(ValidationError):
        service.preview_cma(request(window={"kind": "5Y"}))


def test_missing_whole_trading_day_rejected(workspace):
    service, days = workspace
    path = service.data.data_dir / "asset_nv.parquet"
    frame = pd.read_parquet(path)
    frame = frame.loc[frame.date != days[50]]
    frame.to_parquet(path, index=False)
    with pytest.raises(ValidationError, match="不连续"):
        service.preview_cma(request())


def test_niw_uses_frozen_prior_and_generated_uncertainty(workspace):
    service, _ = workspace
    prior_request = CmaRequest.model_validate({**definition().model_dump(mode="json"),
        "schema_version": "2.0", "moment_semantics": "annualized_periodic_arithmetic",
        "fee_basis": "explicit_assumption", "fx_hedging_basis": "explicit_assumption"})
    prior = publish(service, prior_request, "manual-prior-operation")
    req = request("bayesian_niw", prior_ref={"id": prior["id"], "content_hash": prior["content_hash"]},
                  mean_prior_observations=20., covariance_prior_observations=30.)
    cma = publish(service, req, "bayesian-operation")
    audit = cma["model_result"]["model_audit"]
    assert audit["niw_posterior"]["kappa"] == 180.
    assert audit["niw_posterior"]["nu"] == 193.
    assert audit["uncertainty_status"] == "estimated_under_niw"
    assert all(a["mean_uncertainty"] > 0 for a in frozen_assumptions(cma)["assets"])


def test_regime_reads_immutable_run_without_training_or_writes(workspace):
    service, days = workspace
    from backend.historical_regimes.v2_service import _stored_run_snapshot_hash
    service.cma.evidence.prepare_regime_reader()
    root = service.cma.evidence.regime_root
    root.mkdir(parents=True, exist_ok=True)
    run = {"id": "regime-run-cma-test", "name": "共同状态", "schema_version": "2.0", "immutable": True,
           "mode": "retrospective", "frequency": "daily", "as_of": str(date.today()),
           "states": [{"id": "up"}, {"id": "down"}], "application_bindings": [], "publications": [],
           "series": [{"observation_date": str(day.date()), "available_at": str(day.date()),
                       "recognized_at": str(day.date()), "state_id": "up" if i % 2 else "down"}
                      for i, day in enumerate(days)]}
    run["content_hash"] = _stored_run_snapshot_hash(run)
    path = root / "historical_regime_runs.json"
    path.write_text(json.dumps({"items": [run]}))
    before = path.read_bytes()
    req = request("historical_regime_occupancy", run_ref={"id": run["id"], "content_hash": run["content_hash"]})
    preview = service.preview_cma(req)
    assert path.read_bytes() == before
    assert not path.with_suffix(".json.lock").exists()
    audit = preview["model_result"]["model_audit"]
    assert audit["counts"] == [80, 80]
    assert audit["detected_probabilities"] == [.5, .5]
    assert audit["uncertainty_status"] == "not_estimated"


def test_statistics_fail_closed_when_not_warmed(workspace, monkeypatch):
    service, _ = workspace
    monkeypatch.setattr(numeric, "_WARMED_PID", None)
    with pytest.raises(RuntimeError, match="预热"):
        service.preview_cma(request())

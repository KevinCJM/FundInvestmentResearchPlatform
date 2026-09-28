"""Independent empirical-distribution, exact path and chronological-gate checks."""
from datetime import date, timedelta
from itertools import product

import numpy as np
import pytest
from pydantic import ValidationError as PydanticError

from backend.custom_indicators.errors import ValidationError
from backend.strategic_allocation import cma_scenario_kernels as numeric
from backend.strategic_allocation import cma_model_kernels, cma_statistical_kernels
from backend.strategic_allocation.cma_models import evaluate_cma_model
from backend.strategic_allocation.cma_model_contracts import CMA_MODEL_ADAPTER


@pytest.fixture(scope="module", autouse=True)
def scenario_warm():
    cma_model_kernels.warm()
    cma_statistical_kernels.warm()
    assert numeric.warm()["complete"]


def panel():
    generator = np.random.default_rng(582)
    states = (np.arange(300) // 15 % 2).astype(np.int64)
    returns = generator.normal(0, .008, (300, 3))
    returns[:, 0] += np.where(states == 0, .0006, -.0004)
    returns[:, 1] = returns[:, 0] * .3 + returns[:, 1] * .3
    returns[:, 2] = .01 / 252
    return returns, states


def request(method="long_term_scenario", **patch):
    return {"method": method, "asset_ids": ["equity", "bond", "cash"],
            "as_of": "2020-12-31", "currency": "CNY", "source": "",
            "run_ref": {"id": "historical", "content_hash": "a" * 64}, **patch}


def evidence():
    returns, states = panel()
    dates = [str(date(2019, 1, 1) + timedelta(days=t)) for t in range(len(states))]
    return {"returns": returns, "dates": dates, "period_contiguous": np.ones(len(states), dtype=np.int64), "metadata": {"warnings": []},
            "regime": (states, ["up", "down"], {"run_id": "historical", "state_labels": {"up": "上行", "down": "下行"},
                "label_available_dates": ["2020-12-31"] * len(states)}),
            "forecast_evidence": {"current_probabilities": np.array([.1, .9]),
                "calibration": {"id": "cal", "content_hash": "b" * 64},
                "forecast_validation": {"status": "not_validated"}}}


def test_new_contracts_are_minimal_and_reject_invented_probabilities():
    model = CMA_MODEL_ADAPTER.validate_python(request())
    assert model.window.kind == "common_since_inception"
    for patch in ({"probabilities": {"up": .9, "down": .1}}, {"shrinkage": .5}):
        with pytest.raises(PydanticError):
            CMA_MODEL_ADAPTER.validate_python(request(**patch))
    for horizon in (0, 2521, 2.5, True):
        with pytest.raises(PydanticError):
            CMA_MODEL_ADAPTER.validate_python(request("conditional_scenario", horizon_days=horizon,
                realtime_ref={"id": "now", "content_hash": "b" * 64}))
    with pytest.raises(PydanticError):
        CMA_MODEL_ADAPTER.validate_python(request(historical_reference={"run_id": "other", "publication_id": "publication", "content_hash": "a" * 64}))


def test_regularized_moments_equal_actual_joint_empirical_distribution():
    returns, states = panel()
    states[::19] = -1
    counts, means, risks = cma_statistical_kernels.conditional_state_moments(returns, states, 3, 0.)
    weights = np.array([.35, .7, 1.])
    adjusted_mean, adjusted_risk, _, _ = numeric.regularized_state_moments(counts, means, risks, weights)
    pooled = returns[states >= 0]
    for state in (0, 1):
        own = returns[states == state]
        atoms = np.vstack([own, pooled])
        probabilities = np.r_[np.full(len(own), (1 - weights[state]) / len(own)), np.full(len(pooled), weights[state] / len(pooled))]
        expected = probabilities @ atoms
        centered = atoms - expected
        covariance = centered.T @ (centered * probabilities[:, None])
        np.testing.assert_allclose(adjusted_mean[state], expected, atol=1e-14)
        np.testing.assert_allclose(adjusted_risk[state], covariance, atol=1e-14)
        assert adjusted_risk[state, 2, 2] == 0
    np.testing.assert_array_equal(adjusted_mean[2], np.zeros(3))
    np.testing.assert_array_equal(adjusted_risk[2], np.zeros((3, 3)))


def test_markov_compound_moments_match_complete_path_enumeration():
    initial = np.array([.2, .8])
    transition = np.array([[.95, .05], [.2, .8]])
    means = np.array([[.01, -.005, .001], [-.02, .008, .001]])
    covariances = np.zeros((2, 3, 3))
    expected, risk, path, average = numeric.markov_compound_moments(initial, transition, means, covariances, 3)
    returns, probabilities = [], []
    for sequence in product(range(2), repeat=4):
        p = initial[sequence[0]]
        for t in range(3):
            p *= transition[sequence[t], sequence[t + 1]]
        probabilities.append(p)
        returns.append(np.prod(1 + means[list(sequence[1:])], axis=0) - 1)
    returns, probabilities = np.asarray(returns), np.asarray(probabilities)
    ref_mean = probabilities @ returns
    ref_cov = (returns - ref_mean).T @ ((returns - ref_mean) * probabilities[:, None])
    np.testing.assert_allclose(expected, ref_mean, atol=1e-14)
    np.testing.assert_allclose(risk, ref_cov, atol=1e-14)
    np.testing.assert_allclose(path[-1], initial @ np.linalg.matrix_power(transition, 3), atol=1e-14)
    np.testing.assert_allclose(average, path.mean(0), atol=1e-14)
    assert not np.allclose(path[-1], average)
    np.testing.assert_array_equal(risk[2], np.zeros(3))


def test_joint_simulation_preserves_cross_asset_rows_and_readonly_views():
    returns, states = panel()
    returns[:, 1] = returns[:, 0] * 2
    before = returns.copy()
    view = returns[::-1]
    state_view = states[::-1]
    view.flags.writeable = state_view.flags.writeable = False
    assert np.shares_memory(view, returns)
    p, transition = np.array([.2, .8]), np.array([[.8, .2], [.3, .7]])
    first = numeric.simulate_joint_horizon(view, state_view, np.array([.3, .5]), p, transition, 1, 2000, 55)
    second = numeric.simulate_joint_horizon(view, state_view, np.array([.3, .5]), p, transition, 1, 2000, 55)
    np.testing.assert_allclose(first[:, 1], first[:, 0] * 2, atol=1e-15)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(returns, before)
    np.testing.assert_allclose(first[:, 2], .01 / 252, atol=1e-15)


def test_bootstrap_uncertainty_is_separate_psd_and_keeps_cash_zero():
    returns, states = panel()
    weights, support = numeric.empirical_regularization_weights(returns, states, 2)
    assert ((weights >= 0) & (weights <= 1)).all()
    assert (support > 0).all()
    covariance, width, block, _, _ = numeric.block_mean_uncertainty(returns, states, weights, 123, np.ones(len(states), dtype=np.int64))
    assert 1 <= block <= np.sqrt(len(returns))
    assert width[2] == 0
    np.testing.assert_array_equal(covariance[2], np.zeros(3))
    assert np.linalg.eigvalsh(covariance).min() >= -1e-12
    repeated = numeric.block_mean_uncertainty(returns, states, weights, 123, np.ones(len(states), dtype=np.int64))
    np.testing.assert_array_equal(covariance, repeated[0])


@pytest.mark.parametrize("gap, expected", [(0, [1, 1]), (2, [1, 0]), (3, [1, 0]),
                                          (5, [0, 0]), (7, [0, 1])])
def test_complete_episodes_require_observed_entry_exit_and_continuity(gap, expected):
    states = np.array([0, 0, 1, 1, 1, 0, 0, 1], dtype=np.int64)
    contiguous = np.ones(len(states), dtype=np.int64)
    contiguous[gap] = 0
    np.testing.assert_array_equal(numeric.complete_state_episodes(states, 2, contiguous), expected)
    states[3] = -1
    assert numeric.complete_state_episodes(states, 2, contiguous)[1] == 0


def test_contiguous_blocks_partition_every_row_once_and_reject_invalid_masks():
    owner = np.repeat([0, 1, 1, 0, 1, 1, 1, 1, 0], 2).astype(np.int64)
    mask = owner[::2]
    mask.flags.writeable = False
    ranges = numeric.contiguous_block_ranges(mask, 3)
    np.testing.assert_array_equal(ranges, [[0, 3], [3, 6], [6, 8], [8, 9]])
    np.testing.assert_array_equal(np.concatenate([np.arange(a, b) for a, b in ranges]), np.arange(9))
    assert np.shares_memory(mask, owner)
    for a, b in ranges:
        assert mask[a + 1:b].all()
    for invalid in (np.array([], dtype=np.int64), np.array([0, 2, 1]), np.array([0, -1, 1])):
        with pytest.raises(ValueError, match="STATE_AXIS"):
            numeric.contiguous_block_ranges(invalid, 1)


@pytest.mark.parametrize("lengths", [[1] * 60, [2] * 30, [1, 2, 4, 3, 1, 5] * 4])
def test_gapped_bootstrap_matches_independent_cluster_resampling(lengths):
    # All segments are shorter than the nominal block in these cases. The
    # reference samples explicit segments, not the production range helper.
    count = sum(lengths)
    states = np.zeros(count, dtype=np.int64)
    values = .004 * np.sin(np.arange(count) * .7)
    returns = np.column_stack((values, values * 2, np.full(count, .01 / 252)))
    mask = np.ones(count, dtype=np.int64)
    starts = np.r_[0, np.cumsum(lengths)[:-1]]
    mask[starts] = 0
    storage = np.repeat(returns, 2, axis=0)
    view = storage[::2]
    view.flags.writeable = states.flags.writeable = mask.flags.writeable = False
    signatures = {k.__name__: list(k.signatures) for k in numeric.KERNELS}
    cov, width, nominal, actual, blocks = numeric.block_mean_uncertainty(view, states, np.zeros(1), 123, mask)
    assert nominal >= max(lengths) and actual == max(lengths) and blocks == len(lengths)
    rng = np.random.RandomState(123)
    samples = []
    for _ in range(256):
        chosen = rng.randint(len(lengths), size=len(lengths))
        indices = np.concatenate([np.arange(starts[k], starts[k] + lengths[k]) for k in chosen])
        samples.append(returns[indices].mean(0) * 252)
    samples = np.array(samples)
    center = returns.mean(0) * 252
    expected_width = np.maximum(abs(np.percentile(samples, 97.5, axis=0) - center),
                                abs(np.percentile(samples, 2.5, axis=0) - center))
    np.testing.assert_allclose(cov[:2, :2], np.cov(samples[:, :2], rowvar=False), atol=1e-14)
    np.testing.assert_allclose(width[:2], expected_width[:2], atol=1e-14)
    assert width[2] == 0 and np.all(cov[2] == 0)
    assert np.shares_memory(view, storage)
    np.testing.assert_array_equal(view, returns)
    assert signatures == {k.__name__: list(k.signatures) for k in numeric.KERNELS}
    for kernel in (numeric.block_mean_uncertainty, numeric.contiguous_block_ranges, numeric.complete_state_episodes):
        assert len(kernel.nopython_signatures) == 1 and not kernel._can_compile


def test_isolated_observations_are_disclosed_without_changing_return_moments():
    before = evidence()
    original = evaluate_cma_model(request(), evidence=before)
    before["period_contiguous"][:] = 0
    result = evaluate_cma_model(request(), evidence=before)
    audit = result.model_audit
    assert audit["bootstrap"]["block_length"] == 1
    assert audit["bootstrap"]["fallback"] == "independent_observations"
    assert audit["bootstrap"]["gap_count"] == len(before["dates"]) - 1
    assert audit["bootstrap"]["sampling"] == "disjoint_contiguous_blocks"
    assert audit["complete_episodes"] == [0, 0]
    assert not np.array_equal(result.mean_estimation_covariance, original.mean_estimation_covariance)
    np.testing.assert_array_equal(result.effective_returns, original.effective_returns)
    np.testing.assert_array_equal(result.effective_covariance, original.effective_covariance)
    for invalid in (np.ones(1, dtype=np.int64), np.full(len(before["dates"]), 2, dtype=np.int64)):
        with pytest.raises(ValueError, match="STATE_AXIS"):
            numeric.block_mean_uncertainty(before["returns"], before["regime"][0], np.zeros(2), 123, invalid)
        with pytest.raises(ValueError, match="STATE_AXIS"):
            numeric.complete_state_episodes(before["regime"][0], 2, invalid)


def test_uneven_occupancy_cash_and_unused_states_do_not_create_fake_risk():
    returns, states = panel()
    states[80:133] = 0
    counts, means, covariance = cma_statistical_kernels.conditional_state_moments(returns, states, 3, 0.)
    weights = np.array([.2, .9, 1.])
    adjusted, risk, _, _ = numeric.regularized_state_moments(counts, means, covariance, weights)
    p = cma_statistical_kernels.occupancy_probabilities(counts)
    mean, cov, _, _ = numeric.scenario_mixture_moments(p, adjusted, risk)
    assert mean[2] == .01 / 252
    np.testing.assert_array_equal(cov[2], np.zeros(3))
    uncertainty, half_width, _, _, _ = numeric.block_mean_uncertainty(returns, states, weights, 123, np.ones(len(states), dtype=np.int64))
    assert half_width[2] == 0.0
    np.testing.assert_array_equal(uncertainty[2], np.zeros(3))


def test_future_label_maturity_blocks_claimed_validation_and_windows_do_not_overlap():
    _, states = panel()
    dates = np.arange(len(states), dtype=np.int64)
    scores, cutoff = numeric.transition_holdout_scores(states, dates, np.full(len(states), 299, np.int64), 2, 10, np.ones(len(states), dtype=np.int64))
    assert cutoff == 180
    assert scores[0, 0] == 11
    assert scores[1, 0] == 0
    assert np.isfinite(scores[0, 1:]).all()
    matured, _ = numeric.transition_holdout_scores(states, dates, dates, 2, 10, np.ones(len(states), dtype=np.int64))
    np.testing.assert_array_equal(matured[0], matured[1])
    assert matured[0, 0] <= (len(states) - cutoff) // 10


def test_unknown_breaks_transition_and_unobserved_rows_are_not_filled():
    with pytest.raises(ValueError, match="MISSING_TRANSITIONS"):
        numeric.estimated_transition_matrix(np.array([0, 0, -1, 1], np.int64), 2, np.ones(4, dtype=np.int64))
    with pytest.raises(ValueError, match="FORECAST_PROBABILITY"):
        numeric.markov_probability_path(np.array([.8, .8]), np.eye(2), 10)
    with pytest.raises(ValueError, match="FORECAST_TRANSITION"):
        numeric.markov_probability_path(np.array([.5, .5]), np.array([[1., 0], [np.nan, np.nan]]), 10)
    with pytest.raises(ValueError, match="STATE_SAMPLE"):
        numeric.empirical_regularization_weights(np.empty((0, 2)), np.empty(0, np.int64), 2)
    returns, states = panel()
    returns[0, 0] = np.inf
    with pytest.raises(ValueError, match="STATE_RETURN"):
        numeric.empirical_regularization_weights(returns, states, 2)


def test_long_term_result_frozen_semantics_and_conditional_research_only():
    before = evidence()
    result = evaluate_cma_model(request(), evidence=before)
    audit = result.model_audit
    assert audit["scenario_probabilities"]["state_labels"] == ["上行", "下行"]
    assert audit["model_validation"]["downstream_eligible"] is True
    assert result.effective_covariance[2, 2] == 0
    assert result.mean_estimation_covariance[2, 2] == 0
    np.testing.assert_allclose(result.effective_returns[2], .01)
    renamed = evaluate_cma_model(request(source="only a descriptive note"), evidence=before)
    np.testing.assert_array_equal(result.mean_uncertainty, renamed.mean_uncertainty)
    conditional = request("conditional_scenario", horizon_days=21, realtime_ref={"id": "now", "content_hash": "b" * 64})
    output = evaluate_cma_model(conditional, evidence=before)
    assert output.model_audit["model_validation"]["downstream_eligible"] is False
    assert output.mean_uncertainty is None
    assert output.model_audit["horizon_distribution"]["horizon_days"] == 21
    assert output.model_audit["forecast_validation"]["label_availability_checked"]["samples"] == 0
    longer = evaluate_cma_model({**conditional, "horizon_days": 126}, evidence=before)
    assert output.model_audit["horizon_distribution"]["expected_returns"] != longer.model_audit["horizon_distribution"]["expected_returns"]
    assert output.model_audit["scenario_probabilities"]["average"] != longer.model_audit["scenario_probabilities"]["average"]
    del before["forecast_evidence"]["calibration"]
    with pytest.raises(ValidationError, match="概率校准"):
        evaluate_cma_model(conditional, evidence=before)


def test_scenario_readiness_is_pid_bound_and_no_request_compilation(monkeypatch):
    signatures = {kernel.__name__: list(kernel.signatures) for kernel in numeric.KERNELS}
    evaluate_cma_model(request(), evidence=evidence())
    assert signatures == {kernel.__name__: list(kernel.signatures) for kernel in numeric.KERNELS}
    assert numeric.execution_audit()["python_fallback"] == 0
    monkeypatch.setattr(numeric, "_WARMED_PID", -1)
    with pytest.raises(RuntimeError, match="NOT_READY"):
        evaluate_cma_model(request(), evidence=evidence())


def test_adaptive_simulation_admits_total_work_not_only_last_batch(monkeypatch):
    sizes = []
    def sample(returns, states, weights, initial, transition, horizon, paths, seed):
        sizes.append(paths)
        return np.zeros((paths, returns.shape[1]))
    real_summary = numeric.horizon_sample_summary
    def noisy_summary(samples):
        summary = list(real_summary(samples))
        summary[2] = np.ones(samples.shape[1])
        return tuple(summary)
    monkeypatch.setattr(numeric, "simulate_joint_horizon", sample)
    monkeypatch.setattr(numeric, "horizon_sample_summary", noisy_summary)
    model = request("conditional_scenario", horizon_days=1000,
        realtime_ref={"id": "now", "content_hash": "b" * 64})
    result = evaluate_cma_model(model, evidence=evidence()).model_audit["horizon_distribution"]
    assert len(sizes) > 1
    assert sum(sizes) * 1000 * 3 == result["simulation_return_draws"]
    assert result["simulation_return_draws"] <= result["simulation_work_budget"]
    assert result["simulation_precision"] == "budget_reached"
    assert result["simulation_precision_scope"] == "mean_only_not_tail_or_model_uncertainty"


def test_transition_training_and_validation_never_connect_across_date_gaps():
    states = np.tile(np.array([0, 0, 1, 1], dtype=np.int64), 30)
    owner = np.repeat(np.tile([0, 1, 0, 1], 30), 2)
    contiguous = owner[::2]
    states.flags.writeable = contiguous.flags.writeable = False
    before = owner.copy()
    signatures = tuple(numeric.estimated_transition_matrix.signatures)
    transition, strength, pairs = numeric.estimated_transition_matrix(states, 2, contiguous)
    np.testing.assert_array_equal(transition, np.eye(2))
    assert strength == 0 and pairs == 36
    assert np.shares_memory(contiguous, owner)
    np.testing.assert_array_equal(owner, before)
    assert tuple(numeric.estimated_transition_matrix.signatures) == signatures
    assert not numeric.estimated_transition_matrix._can_compile
    with pytest.raises(ValueError, match='LTCMA_STATE_AXIS'):
        numeric.estimated_transition_matrix(states, 2, contiguous[:-1])


def test_transition_holdout_requires_an_unbroken_horizon():
    _, states = panel()
    dates = np.arange(states.size, dtype=np.int64)
    contiguous = np.ones(states.size, dtype=np.int64)
    complete, cutoff = numeric.transition_holdout_scores(states, dates, dates, 2, 2, contiguous)
    contiguous[cutoff + 1] = False
    gapped, _ = numeric.transition_holdout_scores(states, dates, dates, 2, 2, contiguous)
    np.testing.assert_array_equal(gapped[:, 0], complete[:, 0] - 1)
    contiguous[cutoff + 1:] = False
    unavailable, _ = numeric.transition_holdout_scores(states, dates, dates, 2, 2, contiguous)
    assert not unavailable[:, 0].any()
    assert np.isnan(unavailable[:, 1:]).all()
    with pytest.raises(ValueError, match='LTCMA_FORECAST_VALIDATION_AXIS'):
        numeric.transition_holdout_scores(states, dates, dates, 2, 2, contiguous[:-1])


def test_conditional_model_uses_continuity_evidence_without_changing_return_samples():
    from backend.strategic_allocation.cma_scenario_models import scenario_result
    model = CMA_MODEL_ADAPTER.validate_python(request('conditional_scenario', horizon_days=3,
        realtime_ref={'id': 'now', 'content_hash': 'b' * 64}))
    data = evidence()
    states = data['regime'][0]
    data['period_contiguous'][1:] = states[1:] == states[:-1]
    frozen_returns = data['returns'].copy()
    result = scenario_result(model, data)
    audit = result[3]
    np.testing.assert_array_equal(audit['transition_model']['matrix'], np.eye(2))
    assert audit['transition_model']['excluded_gap_transitions'] == 19
    assert audit['forecast_validation']['gapped_horizons_excluded']
    assert audit['observations'] == len(states)
    np.testing.assert_array_equal(data['returns'], frozen_returns)

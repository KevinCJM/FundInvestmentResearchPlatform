"""Research validity, shared-core parity and immutable array boundaries."""
from __future__ import annotations

import json

import numpy as np
import pytest

from tactical_allocation import numeric
from backend.historical_regimes.taa import _taa_path_kernel


@pytest.fixture(scope="module", autouse=True)
def warmed():
    audit = numeric.warm_tactical_allocation_kernels()
    assert audit["complete"] and audit["python_fallback"] == 0


def inputs(n=80):
    returns = np.zeros((n, 2))
    returns[: n // 2, 0] = 0.01
    returns[n // 2 :, 0] = -0.01
    return dict(returns=returns, probabilities=np.ones((n, 1)), use_signal=np.ones(n, dtype=np.uint8),
                base=np.array([0.5, 0.5]), state_tilts=np.array([[0.1, -0.1]]),
                min_weights=np.zeros(2), max_weights=np.ones(2), max_abs_tilts=np.full(2, 0.3),
                train_end_index=n // 2, cost=10.0, risk_penalty=1.0)


def test_training_selection_ignores_holdout_and_is_deterministic():
    request = inputs()
    result = numeric.evaluate_candidates(**request)
    assert result["selected_id"] == "scale-6"
    assert result["candidates"][6]["validation"]["excess_return"] < 0
    request["returns"][40:] = np.array([0.08, -0.04])
    different = numeric.evaluate_candidates(**request)
    assert different["selected_id"] == result["selected_id"]
    assert [c["train"] for c in result["candidates"]] == [c["train"] for c in different["candidates"]]
    assert result["selection_policy"]["holdout_used_for_selection"] is False
    assert result["selection_policy"]["future_optimality_claim"] is False


def test_zero_tilt_equals_costed_saa_and_holdout_resets_capital():
    result = numeric.evaluate_candidates(**inputs(), selected_candidate_id="scale-0")
    np.testing.assert_array_equal(result["baseline_nav"], result["selected_nav"])
    zero = result["candidates"][0]
    assert zero["train"]["excess_return"] == 0
    assert zero["validation"]["cost"] == zero["validation"]["baseline_cost"]
    assert result["selected_train_path"][0, 7] == 0  # First cost amount from SAA holdings.
    selected = numeric.evaluate_candidates(**inputs())
    path = selected["selected_path"]
    assert path[0, 2:4] == pytest.approx([0.5, 0.5])
    assert path[0, 9] == pytest.approx(-0.005)  # baseline first period net return


def test_selected_path_is_the_unique_existing_engine():
    request = inputs()
    result = numeric.evaluate_candidates(**request)
    reference, _, _, _ = _taa_path_kernel(request["returns"][40:], request["probabilities"][40:],
        request["use_signal"][40:], result["selected_state_tilts"], request["base"], 0.0, 1.0, 1.0, 10.0)
    np.testing.assert_array_equal(result["selected_path"], reference)


def test_per_asset_constraints_preserve_joint_budget_and_direction():
    request = inputs()
    request["max_weights"] = np.array([0.53, 1.0])
    result = numeric.evaluate_candidates(**request, selected_candidate_id="scale-6")
    assert result["selected_state_tilts"][0] == pytest.approx([0.03, -0.03])
    np.testing.assert_allclose(result["selected_weights"].sum(axis=1), 1.0)
    assert result["candidates"][6]["constraint_scales"] == pytest.approx([0.2])


def test_group_constraints_apply_after_each_candidate_strength():
    result = numeric.recommend_weights(np.array([1.0]), 1, np.array([0.3, 0.3, 0.4]),
        np.array([[0.1, 0.1, -0.2]]), np.zeros(3), np.ones(3), np.ones(3), 1.5,
        group_membership=np.array([[1, 1, 0]], dtype=np.uint8), group_min=np.array([0.4]), group_max=np.array([0.7]))
    assert result["weights"] == pytest.approx([0.35, 0.35, 0.3])
    assert result["constraint_scales"] == pytest.approx([1 / 3])
    with pytest.raises(ValueError, match="baseline violates"):
        numeric.recommend_weights(np.array([1.0]), 1, np.array([0.5, 0.5]), np.array([[0.1, -0.1]]),
            np.zeros(2), np.ones(2), np.ones(2), 1,
            group_membership=np.array([[1, 0]], dtype=np.uint8), group_min=np.array([0.6]), group_max=np.array([0.9]))


def test_active_risk_and_turnover_filter_only_training_candidates():
    request = inputs()
    result = numeric.evaluate_candidates(**request, max_turnover=0.01)
    assert result["selected_id"] == "scale-0"
    assert all(not x["feasible"] for x in result["candidates"][1:])
    with pytest.raises(ValueError, match="violates training"):
        numeric.evaluate_candidates(**request, max_turnover=0.01, selected_candidate_id="scale-6")
    request["returns"][:40, 0] = np.tile([0.02, -0.01], 20)
    result = numeric.evaluate_candidates(**request, max_tracking_error=0.0)
    assert result["selected_id"] == "scale-0"


def test_user_choice_is_distinct_from_training_winner():
    result = numeric.evaluate_candidates(**inputs(), selected_candidate_id="scale-1")
    assert result["selected_id"] == "scale-1" and result["auto_selected_id"] == "scale-6"
    result = numeric.evaluate_candidates(**inputs(), strengths=[0, 1], selected_candidate_id="scale-1", objective="excess_return")
    assert len(result["candidates"]) == 2
    assert result["candidates"][1]["strength"] == 1
    json.dumps(result["candidates"], allow_nan=False)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, -1.0])
def test_missing_invalid_or_bankrupt_asset_return_is_rejected(value):
    request = inputs()
    request["returns"][5, 1] = value
    with pytest.raises(ValueError, match="contract 5"):
        numeric.evaluate_candidates(**request)


@pytest.mark.parametrize("patch", [dict(train_end_index=19), dict(train_end_index=61), dict(train_end_index=20.5),
                                    dict(cost=-1), dict(risk_penalty=np.nan), dict(objective="future_best"),
                                    dict(strengths=[0, 10]), dict(periods_per_year=252.5)])
def test_invalid_training_and_parameter_boundaries(patch):
    request = inputs()
    request.update(patch)
    with pytest.raises(ValueError):
        numeric.evaluate_candidates(**request)


def test_dtype_layout_readonly_and_lifetime_are_shared_without_new_signatures(monkeypatch):
    request = inputs()
    backing = np.zeros((160, 4))
    backing[::2, ::2] = request["returns"]
    request["returns"] = backing[::2, ::2]
    request["returns"].setflags(write=False)
    signatures = {k.__name__: tuple(k.signatures) for k in (*numeric.KERNELS, _taa_path_kernel)}
    observed = []
    original = numeric._checked_path

    def check(values, *args):
        observed.append(np.shares_memory(values, backing))
        assert not values.flags.writeable
        return original(values, *args)

    monkeypatch.setattr(numeric, "_checked_path", check)
    result = numeric.evaluate_candidates(**request)
    assert all(observed)
    assert np.shares_memory(result["selected_nav"], result["selected_path"])
    assert np.shares_memory(result["selected_weights"], result["selected_path"])
    assert not result["selected_weights"].flags.writeable
    assert all(tuple(k.signatures) == signatures[k.__name__] for k in (*numeric.KERNELS, _taa_path_kernel))
    np.testing.assert_array_equal(backing[::2, ::2], inputs()["returns"])
    nav = result["selected_nav"]
    del result
    assert np.isfinite(nav).all()  # the view retains its owner


def test_float32_normalizes_only_at_boundary():
    request = inputs()
    request["returns"] = request["returns"].astype(np.float32)
    result = numeric.evaluate_candidates(**request)
    assert result["selected_path"].dtype == np.float64
    assert result["execution"]["request_time_compilation"] == 0


def test_active_metrics_match_independent_sample_variance_reference():
    request = inputs()
    request["returns"][:, 0] = np.tile([0.03, -0.02, 0.01, -0.005], 20)
    result = numeric.evaluate_candidates(**request, selected_candidate_id="scale-4")
    path = result["selected_path"]
    active = path[:, 11] - path[:, 9]
    expected_te = np.std(active, ddof=1) * np.sqrt(252)
    metrics = result["candidates"][4]["validation"]
    assert metrics["tracking_error"] == pytest.approx(expected_te)
    assert metrics["annual_volatility"] == pytest.approx(np.std(path[:, 11], ddof=1) * np.sqrt(252))
    assert metrics["score"] == pytest.approx(np.mean(active) * 252 - expected_te ** 2)


def test_negative_stride_input_and_invalid_flags():
    request = inputs()
    request["returns"] = request["returns"][::-1]
    assert numeric.evaluate_candidates(**request)["selected_id"] == "scale-0"
    request["use_signal"] = np.full(80, 256, dtype=np.int64)
    with pytest.raises(ValueError, match="zero or one"):
        numeric.evaluate_candidates(**request)
    request["use_signal"] = np.full(80, 0.5)
    with pytest.raises(ValueError, match="Integer metadata"):
        numeric.evaluate_candidates(**request)


def dated_momentum(values, lookback, tilt=.1, available=None, starts=None, as_of=None, max_age=31):
    starts = np.arange(100, 100 + len(values), dtype=np.int64) if starts is None else starts
    ends = starts + 1
    available = np.broadcast_to(ends[:, None], values.shape).copy() if available is None else available
    return numeric.build_momentum_signals(values, lookback, tilt, available, starts,
                                           int(ends[-1]) if as_of is None else as_of, ends, max_age)


def test_momentum_uses_only_prior_returns_with_neutral_warmup():
    values = np.tile([0.01, 0.0, -0.01], (8, 1))
    signal = dated_momentum(values, 3, .12)
    assert signal["use_signal"][:3].tolist() == [0, 0, 0]
    assert signal["probabilities"][3] == pytest.approx([1, 0, 0])
    assert signal["state_tilts"][0] == pytest.approx([0.12, -0.06, -0.06])
    altered = values.copy()
    altered[3:] = [-0.5, 0.1, 0.2]
    other = dated_momentum(altered, 3, .12)
    np.testing.assert_array_equal(signal["probabilities"][:4], other["probabilities"][:4])
    assert other["current_probabilities"] == pytest.approx([0, 0, 1])
    assert signal["current_momentum"] == pytest.approx(np.prod(1 + values[-3:], axis=0) - 1)


def test_momentum_t_plus_one_uses_previous_complete_window():
    values = np.tile([.01, 0.], (8, 1))
    available = np.broadcast_to(np.arange(102, 110)[:, None], values.shape).copy()
    signal = dated_momentum(values, 3, available=available)
    assert signal["use_signal"].tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    # At cutoff 104, the window ending 103 was published exactly at 104.
    assert signal["windows"][4].tolist() == [0, 3, 104, 1, 3]
    assert signal["current_knowledge_verified"] == 1
    assert signal["windows"][-1, 1] == 7
    changed = values.copy()
    changed[3:] = [-.5, .1]  # unavailable at cutoff 104, including the latest finished return
    other = dated_momentum(changed, 3, available=available)
    np.testing.assert_array_equal(signal["probabilities"][:5], other["probabilities"][:5])


def test_momentum_unknown_and_long_lags_never_skip_rows_or_extend_expiry():
    values = np.tile([.01, 0.], (8, 1))
    available = np.broadcast_to(np.arange(101, 109)[:, None], values.shape).copy()
    available[2, 0] = -1
    available[5:, 0] = 200
    signal = dated_momentum(values, 2, available=available, max_age=1)
    assert signal["windows"][3].tolist() == [0, 2, 102, 1, 3]
    assert signal["windows"][4].tolist() == [0, 2, 102, 2, 2]
    assert signal["use_signal"][4] == 0
    # New complete window [3:5] becomes valid; unknown row 2 is not compressed away.
    assert signal["windows"][5].tolist() == [3, 5, 105, 0, 3]
    assert signal["current_use_signal"] == 0 and signal["current_knowledge_verified"] == 0
    assert signal["windows"][-1].tolist() == [3, 5, 105, 3, 2]
    unknown = dated_momentum(values, 2, available=np.full(values.shape, -1, dtype=np.int64))
    assert not unknown["use_signal"].any() and unknown["current_use_signal"] == 0
    assert np.isnan(unknown["current_momentum"]).all()


def test_momentum_out_of_order_publications_match_exhaustive_reference():
    rng = np.random.default_rng(913)
    values = rng.normal(.001, .02, (120, 4))
    starts = np.arange(100, 340, 2, dtype=np.int64)
    ends = starts + 1
    available = ends[:, None] + rng.integers(0, 28, values.shape)
    available[::17, 1] = -1
    lookback, max_age = 7, 10
    signal = dated_momentum(values, lookback, available=available, starts=starts, as_of=int(ends[-1] + 3), max_age=max_age)
    for t, cutoff in enumerate([*starts, ends[-1] + 3]):
        candidates = [end for end in range(lookback, t + 1)
                      if ends[end - 1] <= cutoff and np.all(available[end-lookback:end] >= 0)
                      and np.all(available[end-lookback:end] <= cutoff)]
        if not candidates:
            assert signal["windows"][t, 1] == -1
            continue
        end = candidates[-1]
        assert signal["windows"][t, 1] == end
        assert signal["windows"][t, 3] == cutoff - ends[end - 1]
        reference = np.prod(1 + values[end-lookback:end], axis=0) - 1
        use = cutoff - ends[end - 1] <= max_age and np.ptp(reference) > 1e-12
        actual = signal["probabilities"][t] if t < len(values) else signal["current_probabilities"]
        expected = (reference == reference.max()).astype(float) if use else np.zeros(4)
        np.testing.assert_allclose(actual, expected)
        if t == len(values):
            np.testing.assert_allclose(signal["current_momentum"], reference, atol=1e-14)


def test_momentum_readonly_stride_inputs_share_memory_and_keep_fixed_signatures(monkeypatch):
    source = np.tile([.01, 0., -.01, 0.], (32, 1))
    values = source[::2, ::2]
    starts_owner = np.arange(100, 132, dtype=np.int64)
    starts = starts_owner[::2]
    ends_owner = starts_owner + 1
    ends = ends_owner[::2]
    availability_owner = np.broadcast_to(ends_owner[:, None], source.shape).copy()
    available = availability_owner[::2, ::2]
    for array in (values, starts, ends, available):
        array.setflags(write=False)
    signatures = {k.__name__: tuple(k.signatures) for k in numeric.KERNELS}
    original = numeric.momentum_signals_kernel
    def checked(v, lookback, tilt, a, s, e, cutoff, age):
        assert np.shares_memory(v, source) and np.shares_memory(a, availability_owner)
        assert np.shares_memory(s, starts_owner) and np.shares_memory(e, ends_owner)
        assert all(not arr.flags.writeable for arr in (v, a, s, e))
        return original(v, lookback, tilt, a, s, e, cutoff, age)
    monkeypatch.setattr(numeric, "momentum_signals_kernel", checked)
    result = numeric.build_momentum_signals(values, 3, .1, available, starts, 131, ends)
    assert result["current_use_signal"] == 1
    assert all(tuple(k.signatures) == signatures[k.__name__] for k in numeric.KERNELS)
    assert all(k.nopython_signatures and not k._can_compile for k in numeric.KERNELS)
    np.testing.assert_array_equal(source[:, 0], .01)


def test_momentum_empty_warmup_missing_dates_and_invalid_values():
    with pytest.raises(ValueError, match="window axes"):
        numeric.build_momentum_signals(np.empty((0, 2)), 2, .1, np.empty((0, 2), dtype=np.int64),
                                       np.empty(0, dtype=np.int64), 100, np.empty(0, dtype=np.int64))
    values = np.ones((2, 2)) * .01
    assert not dated_momentum(values, 3)["use_signal"].any()
    with pytest.raises(ValueError, match="explicit return dates"):
        numeric.build_momentum_signals(values, 2, .1)
    for invalid in (np.nan, np.inf, -1.):
        altered = values.copy(); altered[0, 0] = invalid
        with pytest.raises(ValueError, match="finite"):
            dated_momentum(altered, 2)


def test_momentum_ties_and_one_asset_are_neutral():
    equal = dated_momentum(np.zeros((5, 2)), 1)
    assert not equal["use_signal"].any() and equal["current_use_signal"] == 0
    assert equal["current_knowledge_verified"] == 1
    single = dated_momentum(np.full((5, 1), .1), 1)
    assert not single["use_signal"].any()
    tied = dated_momentum(np.tile([.1, .1, 0.], (5, 1)), 1)
    assert tied["current_probabilities"] == pytest.approx([.5, .5, 0])


def test_recommendation_uses_current_holdings_for_trade_delta():
    result = numeric.recommend_weights(np.array([1.0]), 1, np.array([0.5, 0.5]), np.array([[0.1, -0.1]]),
        np.zeros(2), np.ones(2), np.ones(2), 1.0, np.array([0.7, 0.3]))
    assert result["weights"] == pytest.approx([0.6, 0.4])
    assert result["tilts"] == pytest.approx([0.1, -0.1])
    assert result["trade_deltas"] == pytest.approx([-0.1, 0.1])
    assert result["turnover"] == pytest.approx(0.1)
    assert result["has_deviation"] is True and result["is_saa"] is False
    constrained = numeric.recommend_weights(np.array([1.0]), 1, np.array([0.5, 0.5]), np.array([[0.1, -0.1]]),
        np.zeros(2), np.ones(2), np.zeros(2), 1.0, np.array([0.7, 0.3]))
    assert constrained["is_saa"] is True
    assert constrained["turnover"] == pytest.approx(0.2)


def test_scenario_shocks_costs_and_linked_contributions_reconcile():
    result = numeric.stress_compare(np.array([[-0.2, 0.03], [0.08, -0.01]]),
                                    np.array([0.5, 0.5]), np.array([0.6, 0.4]), cost=50)
    assert sum(result["baseline_contributions"]) - result["baseline_cost"] == pytest.approx(result["baseline"]["total_return"])
    assert sum(result["target_contributions"]) - result["target_cost"] == pytest.approx(result["target"]["total_return"])
    assert sum(result["excess_contributions"]) - (result["target_cost"] - result["baseline_cost"]) == pytest.approx(result["total_return_difference"])
    assert result["target"]["cost"] > 0
    one = numeric.stress_compare(np.array([[-0.2, 0.03]]), np.array([0.5, 0.5]), np.array([0.6, 0.4]))
    assert one["target"]["total_return"] == pytest.approx(-0.108)
    assert one["target"]["annual_volatility"] is None
    assert "probability" not in one


def test_compose_and_aggregate_product_weights_conserve_class_budgets():
    classes = np.array([0.45, 0.55])
    indices = np.array([0, 0, 1], dtype=np.int64)
    products = numeric.compose_product_weights(classes, indices, np.array([0.6, 0.4, 1]))
    assert products == pytest.approx([0.27, 0.18, 0.55])
    assert numeric.aggregate_class_weights(products, indices, 2) == pytest.approx(classes)
    with pytest.raises(ValueError, match="complete within-class"):
        numeric.compose_product_weights(classes, indices, np.array([0.6, 0.3, 1]))
    with pytest.raises(ValueError, match="Unknown"):
        numeric.compose_product_weights(classes, np.array([0, 0, 3]), np.array([0.6, 0.4, 1]))
    with pytest.raises(ValueError, match="Integer metadata"):
        numeric.compose_product_weights(classes, np.array([0.1, 0.0, 1.0]), np.array([0.6, 0.4, 1]))
    with pytest.raises(ValueError, match="int64 range"):
        numeric.compose_product_weights(classes, np.array([0, 0, 2**64 - 1], dtype=np.uint64), np.array([0.6, 0.4, 1]))


def test_return_availability_preserves_missing_and_later_endpoint():
    values = np.array([[10, -1], [12, 13], [11, 15]], dtype=np.int64)
    values.setflags(write=False)
    result = numeric.returns_availability_kernel(values)
    np.testing.assert_array_equal(result, [[12, -1], [12, 15]])
    assert numeric.returns_availability_kernel(values[:1]).shape == (0, 2)
    with pytest.raises(ValueError):
        numeric.returns_availability_kernel(values[:0])


def test_immature_training_outcomes_and_unknown_outcomes_are_distinct():
    available = np.array([[10, 11], [12, 15], [-1, 14]], dtype=np.int64)
    assert numeric.knowledge_window_status(available, 12, 0, 2) == {"future_cells": 1, "unknown_cells": 0, "verified": False}
    assert numeric.knowledge_window_status(available, 15) == {"future_cells": 0, "unknown_cells": 1, "verified": False}
    assert numeric.knowledge_window_status(available, 15, 0, 2)["verified"] is True
    with pytest.raises(ValueError):
        numeric.knowledge_window_status(available, 15, 0, 4)


def test_numeric_requests_fail_closed_without_worker_warmup(monkeypatch):
    monkeypatch.setattr(numeric, "_WARMED_PID", None)
    assert numeric.execution_audit()["complete"] is False
    with pytest.raises(RuntimeError, match="not warmed"):
        numeric.evaluate_candidates(**inputs())
    with pytest.raises(RuntimeError, match="not warmed"):
        numeric.stress_compare(np.zeros((1, 2)), np.array([0.5, 0.5]), np.array([0.5, 0.5]))

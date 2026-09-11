"""Offline, independent references for the new return-producing numerical lane."""
import numpy as np
import pytest

from backend.factor_research import return_kernels as k


def ff3_arrays(assets=6, days=45):
    per_group = assets // 6
    size = np.r_[np.arange(1, assets // 2 + 1), np.arange(11, 11 + assets // 2)].astype(float)
    bm = np.tile(np.repeat([1., 2., 3.], per_group), 2)
    descriptors = np.empty((1, assets, 4))
    descriptors[0, :, 0] = size
    descriptors[0, :, 1] = 100.
    descriptors[0, :, 2] = bm * 100.
    descriptors[0, :, 3] = 1.
    returns = np.ascontiguousarray(.001 * np.arange(1, assets + 1)[None, :]
                                    + .0002 * np.arange(days)[:, None])
    caps = np.ascontiguousarray(np.tile(size, (days, 1)))
    return returns, caps, np.full(days, .0001), np.array([0], dtype=np.int64), descriptors


def test_fixed_signatures_warmed_and_dtype_refused():
    assert k.warm_return_kernels()["complete"]
    for kernel in k.KERNELS:
        assert len(kernel.signatures) == len(kernel.nopython_signatures) == 1
    with pytest.raises(TypeError):
        k.return_diagnostics_kernel(np.ones((4, 2), dtype=np.float32))


def test_ff3_six_groups_and_independent_formulas():
    returns, caps, rf, days, descriptors = ff3_arrays()
    values, counts, groups = k.ff3_returns_kernel(returns, caps, rf, days, descriptors)
    np.testing.assert_array_equal(groups, [[0, 1, 2, 3, 4, 5]])
    np.testing.assert_array_equal(counts[0, :6], np.ones(6))
    np.testing.assert_allclose(values[1:, 4:], returns[1:])
    np.testing.assert_allclose(values[1:, 0], np.sum(caps[1:] * returns[1:], axis=1) / caps[1:].sum(axis=1) - rf[1:])
    np.testing.assert_allclose(values[1:, 1], returns[1:, :3].mean(axis=1) - returns[1:, 3:].mean(axis=1))
    np.testing.assert_allclose(values[1:, 2], (returns[1:, 2] + returns[1:, 5] - returns[1:, 0] - returns[1:, 3]) / 2)
    assert np.isnan(values[0]).all()


def test_ff3_value_weights_drift_without_daily_reweighting():
    returns, caps, rf, days, descriptors = ff3_arrays(12)
    values, _, memberships = k.ff3_returns_kernel(returns, caps, rf, days, descriptors)
    for group in range(6):
        selected = memberships[0] == group
        dollars = descriptors[0, selected, 0].copy()
        for t in range(1, len(returns)):
            expected = np.dot(dollars / dollars.sum(), returns[t, selected])
            assert values[t, 4 + group] == pytest.approx(expected)
            dollars *= 1 + returns[t, selected]


def test_ff3_missing_held_asset_does_not_reweight_survivors():
    returns, caps, rf, days, descriptors = ff3_arrays(12)
    returns[5, 0] = np.nan
    values, _, _ = k.ff3_returns_kernel(returns, caps, rf, days, descriptors)
    assert np.isnan(values[5:, 4]).all()
    assert np.isnan(values[5:, 1:3]).all()
    assert np.isnan(values[5, 0])
    assert np.isfinite(values[6, 0])  # The independent daily market observation can resume.
    assert np.isfinite(values[5:, 5:]).all()


def test_ff3_formation_at_close_and_future_changes_do_not_change_history():
    returns, caps, rf, _, descriptors = ff3_arrays()
    second = descriptors.copy()
    second[0, :, 2] = second[0, ::-1, 2]
    dates = np.array([0, 20], dtype=np.int64)
    combined = np.ascontiguousarray(np.concatenate([descriptors, second]))
    baseline, _, groups = k.ff3_returns_kernel(returns, caps, rf, dates, combined)
    assert groups[1, 0] == 2
    assert baseline[20, 4] == pytest.approx(returns[20, 0])
    assert baseline[21, 4] == pytest.approx(returns[21, 2])
    altered = returns.copy()
    altered[30:] += .02
    result, _, _ = k.ff3_returns_kernel(altered, caps, rf, dates, combined)
    np.testing.assert_allclose(result[:30], baseline[:30], equal_nan=True)


def test_ff3_total_loss_is_not_treated_as_missing_or_dropped():
    returns, caps, rf, days, descriptors = ff3_arrays()
    returns[2, 0] = -1.
    values, _, _ = k.ff3_returns_kernel(returns, caps, rf, days, descriptors)
    assert values[2, 4] == pytest.approx(-1.)
    assert np.isnan(values[3, 4])


def test_equal_characteristics_do_not_arbitrarily_split_groups():
    returns, caps, rf, days, descriptors = ff3_arrays()
    descriptors[:, :, 0] = 1.
    descriptors[:, :, 2] = 1.
    _, counts, _ = k.ff3_returns_kernel(returns, caps, rf, days, descriptors)
    assert np.count_nonzero(counts[0, :6]) == 1
    np.testing.assert_array_equal(k.spread_targets_kernel(np.ones((2, 6)), 3), np.zeros((2, 2, 6)))


def test_spread_entry_delay_drift_two_sided_cost_and_future_independence():
    prices = np.ones((7, 6))
    prices[2] = [2, 3, 1, 1, 4, 5]  # Before entry; must earn none of these changes.
    prices[3] = prices[2] * np.array([1.1, 1.2, 1, 1, 1.3, 1.4])
    prices[4:] = prices[3] * np.array([1, 2, 1, 1, 2, 1])
    decisions = np.array([1], dtype=np.int64)
    scores = np.array([[0., 1., 2., 3., 4., 5.]])
    path, targets = k.spread_returns_kernel(prices, decisions, scores, 3, 10.)
    assert np.isnan(path[:2]).all()
    np.testing.assert_array_equal(targets[0, 0], [.5, .5, 0, 0, 0, 0])
    assert path[2, 2] == 0
    assert path[2, 4] == pytest.approx(2.)
    assert path[2, 3] == pytest.approx(-.002)
    assert path[3, 2] == pytest.approx(.35 - .15)
    assert path[4, 2] == pytest.approx(1.3 / 2.7 - 1.2 / 2.3)
    prices[5, 0] = np.nan
    broken, _ = k.spread_returns_kernel(prices, decisions, scores, 3, 10.)
    np.testing.assert_allclose(broken[:5], path[:5], equal_nan=True)
    assert np.isnan(broken[5:, 2]).all()


def test_spread_partial_initial_entry_does_not_invent_transition_returns():
    prices = np.ones((8, 6))
    prices[2, 0] = np.nan  # The low leg cannot enter initially; the high leg can.
    prices[5:, 4:] = 1.1
    decisions = np.array([1, 4], dtype=np.int64)
    scores = np.ascontiguousarray(np.tile(np.arange(6, dtype=np.float64), (2, 1)))
    path, _ = k.spread_returns_kernel(prices, decisions, scores, 3, 0.)
    assert np.isnan(path[2:6, 2]).all()
    assert path[6, 2] == pytest.approx(0.)


def test_rolling_ic_respects_window_missingness_and_sample_boundary():
    values = np.array([[.1], [.2], [np.nan], [.4], [.5], [.6], [.7]])
    samples = np.array([0, 0, -1, 1, 1, 1, 1], dtype=np.int64)
    out = k.rolling_ic_kernel(values, samples, 3, 3)
    assert np.isnan(out[4, 0, 1])
    assert out[5, 0, 1] == pytest.approx(.5)
    assert out[6, 0, 1] == pytest.approx(.6)
    assert out[5, 0, 2] == pytest.approx(np.std([.4, .5, .6], ddof=1))
    assert out[5, 0, 3] == pytest.approx(.5 / .1)
    assert out[5, 0, 4] == 1


def test_return_diagnostics_does_not_bridge_missing_cumulative_values():
    values = np.array([[np.nan, 1.], [.1, 1.], [.2, 1.], [np.nan, 1.], [.3, 1.]])
    stats, correlation, cumulative = k.return_diagnostics_kernel(values)
    assert stats[0, 0] == 3
    assert stats[0, 1] == pytest.approx(.2)
    assert cumulative[2, 0] == pytest.approx(.3)
    assert np.isnan(cumulative[3:, 0]).all()
    assert correlation[0, 0] == pytest.approx(1.)
    assert np.isnan(correlation[1]).all() and np.isnan(correlation[:, 1]).all()


def test_empty_arrays_and_determinism():
    stats, corr, cumulative = k.return_diagnostics_kernel(np.empty((0, 2)))
    assert stats[:, 0].tolist() == [0., 0.]
    assert cumulative.shape == (0, 2)
    assert np.isnan(corr).all()
    inputs = ff3_arrays()
    left = k.ff3_returns_kernel(*inputs)
    right = k.ff3_returns_kernel(*inputs)
    for one, two in zip(left, right):
        np.testing.assert_array_equal(one, two)

import numpy as np
import pytest
from backend.factor_research import numba_kernels as k


def test_warmup_has_only_fixed_nopython_signatures():
    audit = k.warm_factor_kernels()
    assert audit["complete"] and audit["python_fallback"] == 0
    assert audit["request_time_compilation"] == 0
    assert all(len(kernel.signatures) == 1 for kernel in k.KERNELS)


def test_average_ties_and_missing_spearman():
    values = np.array([3., 1., 1., np.nan, np.inf])
    np.testing.assert_allclose(k.ranks_kernel(values), [3, 1.5, 1.5, np.nan, np.nan], equal_nan=True)
    assert np.isnan(k.correlation_kernel(np.ones(4), np.arange(4.)))
    assert np.isnan(k.correlation_kernel(np.array([1., 2.]), np.array([2., 3.])))


@pytest.mark.parametrize("factor_count", [1, 3])
@pytest.mark.parametrize("missing_label", [None, np.nan, np.inf, -np.inf])
def test_diagnostics_preserves_signal_groups_and_inputs(factor_count, missing_label):
    scores = np.array([[1., 2., 3., 4., 5., 6.]])
    normalized = np.ascontiguousarray(np.stack(
        [scores if f % 2 == 0 else scores[:, ::-1] for f in range(factor_count)], axis=2))
    labels = np.array([[.01, .02, .03, .04, .05, .06]])
    if missing_label is not None:
        labels[0, 1] = missing_label
    inputs = (normalized, scores, labels)
    before = tuple(value.copy() for value in inputs)
    signatures = tuple(k.diagnostics_kernel.signatures)

    ic, rank_ic, groups, counts = k.diagnostics_kernel(*inputs, 3)

    # Group identities are fixed by scores, not by subsequent label coverage.
    expected_groups = [[.015 if missing_label is None else .01, .035, .055]]
    np.testing.assert_allclose(groups, expected_groups)
    np.testing.assert_array_equal(counts, np.full((1, factor_count + 1), 6 if missing_label is None else 5))
    signs = [1 if f % 2 == 0 else -1 for f in range(factor_count)] + [1]
    np.testing.assert_allclose(ic, [signs])
    np.testing.assert_allclose(rank_ic, [signs])
    for current, original in zip(inputs, before):
        np.testing.assert_array_equal(current, original)
    assert tuple(k.diagnostics_kernel.signatures) == signatures
    assert len(k.diagnostics_kernel.nopython_signatures) == len(signatures) == 1


def test_diagnostics_reuses_pair_buffers_without_leaking_missing_or_inf():
    scores = np.array([[1., 2., 3., 4., 5., 6.], [6., 5., 4., 3., 2., 1.]])
    normalized = np.ascontiguousarray(np.stack([scores, scores], axis=2))
    normalized[0, :3, 0] = [np.nan, np.inf, -np.inf]
    normalized[1, :, 0] = np.nan
    labels = np.array([[.01, np.nan, .03, .04, .05, .06], [.01, .02, .03, .04, .05, .06]])
    before = tuple(value.copy() for value in (normalized, scores, labels))

    ic, rank_ic, groups, counts = k.diagnostics_kernel(normalized, scores, labels, 3)

    np.testing.assert_allclose(groups, [[.01, .035, .055], [.055, .035, .015]])
    np.testing.assert_array_equal(counts, [[3, 5, 5], [0, 6, 6]])
    np.testing.assert_allclose(ic, [[1, 1, 1], [np.nan, -1, -1]], equal_nan=True)
    np.testing.assert_allclose(rank_ic, ic, equal_nan=True)
    for current, original in zip((normalized, scores, labels), before):
        np.testing.assert_array_equal(current, original)


@pytest.mark.parametrize("invalid_score", [np.nan, np.inf, -np.inf])
def test_diagnostics_excludes_only_invalid_signal_scores_from_group_assignment(invalid_score):
    scores = np.array([[1., 2., 3., 4., 5., invalid_score]])
    normalized = scores[:, :, None].copy()
    labels = np.array([[.01, np.nan, .03, .04, .05, .06]])
    before = scores.copy()
    _, _, groups, counts = k.diagnostics_kernel(normalized, scores, labels, 3)
    np.testing.assert_allclose(groups, [[.01, .035, .05]])
    np.testing.assert_array_equal(counts, [[4, 4]])
    np.testing.assert_array_equal(scores, before)
    np.testing.assert_array_equal(normalized[:, :, 0], before)


@pytest.mark.parametrize("shape", [(0, 6), (2, 0), (2, 6)])
def test_diagnostics_empty_or_all_missing_inputs(shape):
    scores = np.full(shape, np.nan)
    normalized = np.full((*shape, 2), np.nan)
    labels = np.full(shape, np.nan)
    ic, rank_ic, groups, counts = k.diagnostics_kernel(normalized, scores, labels, 3)
    assert np.isnan(ic).all() and np.isnan(rank_ic).all() and np.isnan(groups).all()
    assert not counts.any()
    assert np.isnan(normalized).all() and np.isnan(scores).all() and np.isnan(labels).all()


def test_point_in_time_features_do_not_see_same_day_or_future_announcements():
    p = np.ascontiguousarray(np.tile(np.arange(1., 11.)[:, None], (1, 3)))
    days = np.arange(10, dtype=np.int64)
    available = np.ascontiguousarray(np.tile(days[:, None], (1, 3)))
    params = np.array([[0, 2, 0, 1]], dtype=np.int64)
    d = np.array([5], dtype=np.int64)
    before = k.features_kernel(p, available, days, d, params)
    np.testing.assert_allclose(before[0, :, 0], 5 / 3 - 1)
    p[5:] = 999.
    np.testing.assert_equal(before, k.features_kernel(p, available, days, d, params))
    available[4, 0] = 9
    after = k.features_kernel(p, available, days, d, params)
    assert after[0, 0, 0] == pytest.approx(4 / 2 - 1)


def test_missing_window_does_not_compress_time():
    p = np.ascontiguousarray(np.tile(np.arange(1., 11.)[:, None], (1, 3)))
    p[3, 0] = np.nan
    days = np.arange(10, dtype=np.int64)
    availability = np.ascontiguousarray(np.tile(days[:, None], (1, 3)))
    result = k.features_kernel(p, availability, days, np.array([5], dtype=np.int64),
                               np.array([[0, 2, 0, 1], [1, 2, 0, -1]], dtype=np.int64))
    assert np.isnan(result[0, 0]).all()


def test_volatility_matches_independent_reference_and_dtype_guard():
    p = np.array([[1., 2., 3.], [1.1, 2.2, 3.3], [1.05, 2.1, 3.15],
                  [1.2, 2.4, 3.6], [1.3, 2.6, 3.9], [1.4, 2.8, 4.2]])
    days = np.arange(6, dtype=np.int64)
    available = np.ascontiguousarray(np.tile(days[:, None], (1, 3)))
    params = np.array([[1, 3, 0, -1]], dtype=np.int64)
    result = k.features_kernel(p, available, days, np.array([4], dtype=np.int64), params)
    expected = np.std(p[1:4, 0] / p[:3, 0] - 1, ddof=1) * np.sqrt(252)
    assert result[0, 0, 0] == pytest.approx(expected)
    with pytest.raises(TypeError):
        k.features_kernel(p.astype(np.float32), available, days, np.array([4], dtype=np.int64), params)


def test_labels_enter_next_session_and_reject_missing_path():
    p = np.ascontiguousarray(np.tile(np.arange(1., 11.)[:, None], (1, 3)))
    p[4, 1] = np.nan
    result = k.labels_kernel(p, np.array([2, 9], dtype=np.int64), 3)
    assert result[0, 0] == pytest.approx(7 / 4 - 1)
    assert np.isnan(result[0, 1])
    assert np.isnan(result[1]).all()


def test_direction_weights_and_constant_cross_section():
    raw = np.array([[[1., 3.], [2., 2.], [3., 1.]]])
    norm, score = k.normalize_kernel(raw, np.array([1, -1], dtype=np.int64), np.array([.5, .5]), 0)
    np.testing.assert_allclose(score, [[0, 50, 100]])
    assert np.isfinite(norm).all()
    _, constant = k.normalize_kernel(np.ones((1, 3, 2)), np.ones(2, dtype=np.int64), np.ones(2), 1)
    np.testing.assert_equal(constant, np.zeros((1, 3)))


def test_daily_backtest_delay_turnover_drift_cost_and_no_survivor_reweighting():
    p = np.array([[1., 1., 1.], [1., 1., 1.], [1., 1., 1.],
                  [2., 1., 1.], [2., 2., 1.], [2., 2., 1.]])
    d = np.array([1, 3], dtype=np.int64)
    scores = np.array([[3., 2., 1.], [1., 3., 2.]])
    path, targets = k.backtest_kernel(p, np.ones(6), d, scores, 1, 10.)
    assert np.isnan(path[1, 0])
    assert path[2, 0] == pytest.approx(.999)
    assert path[3, 0] == pytest.approx(1.998)
    assert path[4, 4] == 2  # Sell first asset and buy second at day-four close.
    assert path[4, 0] == pytest.approx(1.998 * .998)
    p[3, 0] = np.nan
    broken, _ = k.backtest_kernel(p, np.ones(6), d, scores, 1, 10.)
    assert np.isnan(broken[3:, 0]).all()
    assert np.isnan(k.performance_kernel(broken, 0, 6)[1])


def test_top_boundary_ties_get_equal_weights():
    targets = k.target_weights_kernel(np.array([[5., 4., 4., 1.]]), 2)
    np.testing.assert_allclose(targets, [[1/3, 1/3, 1/3, 0]])


def test_rbsa_and_ff3_recover_known_coefficients_and_intercept():
    rng = np.random.default_rng(23)
    x = np.ascontiguousarray(rng.normal(0, .01, (220, 3)))
    beta = np.array([.2, .3, .5])
    y = np.ascontiguousarray((x @ beta + .0002)[:, None])
    for model in [0, 1]:
        coefficients, stats = k.attribution_kernel(y, x, np.zeros(220), 150, model)
        np.testing.assert_allclose(coefficients[0, :3], beta, atol=1e-7)
        assert coefficients[0, 3] == pytest.approx(.0002, abs=1e-10)
        assert stats[0, 2] == pytest.approx(1.)
        assert stats[0, 3] == pytest.approx(1.)
        assert stats[0, 7] == 0
    duplicate = np.ascontiguousarray(np.column_stack([x[:, 0], x[:, 0], x[:, 0]]))
    _, stats = k.attribution_kernel(y, duplicate, np.zeros(220), 150, 1)
    assert stats[0, 7] == 3


def test_portfolio_profile_exposes_missing_weight():
    result = k.portfolio_profile_kernel(np.array([[.2, np.nan], [.8, .5]]), np.array([.6, .4]))
    np.testing.assert_allclose(result, [[.44, 1.], [.5, .4]])


def test_empty_inputs_and_determinism():
    assert k.ranks_kernel(np.array([], dtype=np.float64)).size == 0
    assert k.mean_stats_kernel(np.array([np.nan, np.inf]))[0] == 0
    p = np.ones((3, 3))
    args = (p, np.ones(3), np.array([], dtype=np.int64), np.empty((0, 3)), 1, 5.)
    one = k.backtest_kernel(*args)
    two = k.backtest_kernel(*args)
    np.testing.assert_equal(one[0], two[0])

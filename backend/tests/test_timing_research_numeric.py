"""Execution-reference, causality, missing-data and memory contracts."""
import numpy as np
import pytest

from backend.timing_research.numeric import (
    KERNELS, PATH_COLUMNS, SUMMARY_COLUMNS, QUALITY_COLUMNS,
    alpha_beta_kernel, condition_event_kernel, condition_values_kernel,
    simulate_kernel, analyze_kernel, signal_quality_kernel,
    warm_timing_kernels, timing_execution_audit,
    series_status_kernel, availability_status_kernel, MISSING_DAY,
    execution_volume_status_kernel,
    diagnostics_kernel, DIAGNOSTIC_COLUMNS,
)
from historical_regimes.condition_numba import (
    condition_compare_kernel, condition_logic_kernel, condition_valid_kernel, select_state_kernel,
)


def quote(n=12):
    return [np.full(n, 100.0) for _ in range(4)]


def run(prices, entry=None, exit_=None, **kwargs):
    n = len(prices[0])
    params = dict(start=0, end=n, max_holding=100, cooldown=0,
                  fee_bps=0.0, slippage_bps=0.0, take_profit=0.0, stop_loss=0.0)
    params.update(kwargs)
    return simulate_kernel(*prices, np.zeros(n, dtype=np.int64) if entry is None else entry,
                           np.zeros(n, dtype=np.int64) if exit_ is None else exit_, **params)


def test_alpha_beta_reference_and_reset():
    values = np.array([10.0, 11.0, 10.5, np.nan, 12.0, 13.0, np.inf, 14.0])
    expected = np.full((3, len(values)), np.nan)
    level, slope = None, 0.0
    for t, value in enumerate(values):
        if not np.isfinite(value):
            level = None
            continue
        if level is None:
            level, slope = value, 0.0
        else:
            error = value - (level + slope)
            level = level + slope + 0.4 * error
            slope = slope + 0.1 * error
            expected[2, t] = error
        expected[0, t], expected[1, t] = level, slope
    np.testing.assert_allclose(alpha_beta_kernel(values, 0.4, 0.1), expected, equal_nan=True)
    assert alpha_beta_kernel(np.empty(0), 0.4, 0.1)[0].size == 0
    with pytest.raises(ValueError):
        alpha_beta_kernel(values, 0.0, 0.1)


def test_series_and_availability_validation_preserves_warmup_and_rejects_future():
    assert series_status_kernel(np.array([np.nan, 1., 2.])) == 0
    assert series_status_kernel(np.array([np.nan, np.inf])) == 1
    dates = np.arange(5, dtype=np.int64)
    available = np.array([0, 1, MISSING_DAY, 4, 4], dtype=np.int64)
    assert availability_status_kernel(dates, available, 0, 3) == 0
    assert availability_status_kernel(dates, available, 0, 5) == 1
    with pytest.raises(ValueError):
        availability_status_kernel(dates, available, -1, 5)


def test_execution_volume_fails_closed_for_missing_or_nontrading_sessions():
    owner = np.array([100., -99., 200., -99., 0., -99., np.nan, -99.])
    volume = owner[::2]
    volume.setflags(write=False)
    signatures = list(execution_volume_status_kernel.signatures)
    assert execution_volume_status_kernel(volume, 0, 2) == 0
    assert execution_volume_status_kernel(volume, 0, 3) == 2
    assert execution_volume_status_kernel(volume, 0, 4) == 1
    assert execution_volume_status_kernel(volume, 3, 3) == 0
    assert signatures == list(execution_volume_status_kernel.signatures)
    assert np.shares_memory(volume, owner)
    with pytest.raises(ValueError):
        execution_volume_status_kernel(volume, 0, 5)


def test_diagnostics_counts_concentration_and_compounds_partial_months():
    # First month has +10%, then -10%: compounding is negative, not zero.
    path = np.zeros((6, len(PATH_COLUMNS)), dtype=np.float64)
    path[:, 9] = 1
    path[:, 2] = [.1, -.1, .02, .01, -.03, .05]
    path[:, 3] = [.1, -.2, .01, .01, .02, .01]
    months = np.array([202601, 202601, 202602, 202602, 202603, 202603], dtype=np.int64)
    entry = np.array([1, 1, -1, 0, 0, 1], dtype=np.int64)
    for array in (path, months, entry):
        array.setflags(write=False)
    result = dict(zip(DIAGNOSTIC_COLUMNS, diagnostics_kernel(entry, months, path, 0, 6)))
    assert result["raw_signal_count"] == 3
    assert result["known_condition_count"] == 5
    assert result["active_month_count"] == 2 and result["total_month_count"] == 3
    assert result["top_month_signal_share"] == pytest.approx(2 / 3)
    assert result["positive_month_fraction"] == pytest.approx(2 / 3)
    assert result["positive_excess_month_fraction"] == pytest.approx(2 / 3)
    partial = diagnostics_kernel(entry, months, path, 1, 5)
    assert partial[3] == 3 and partial[5] == pytest.approx(1 / 3)
    zero = diagnostics_kernel(np.zeros(6, dtype=np.int64), months, path, 0, 6)
    assert zero[0] == 0 and np.isnan(zero[4])
    empty = diagnostics_kernel(entry, months, path, 2, 2)
    assert empty[3] == 0 and np.isnan(empty[5])
    invalid = path.copy()
    invalid[2, 9] = 0
    assert np.isnan(diagnostics_kernel(entry, months, invalid, 0, 6)[5:]).all()


def test_alpha_beta_future_perturbation_does_not_change_history():
    base = np.linspace(100.0, 150.0, 100)
    changed = base.copy()
    changed[60:] = -300.0
    for actual, expected in zip(alpha_beta_kernel(changed, 0.5, 0.2), alpha_beta_kernel(base, 0.5, 0.2)):
        np.testing.assert_array_equal(actual[:60], expected[:60])


def test_conditions_preserve_unknown_and_explicit_events():
    values = np.array([-1, 1, 0, 1, 1, 1, -1, 1, 1, 0, 1], dtype=np.int64)
    np.testing.assert_array_equal(condition_event_kernel(values, 0, 1), [-1, -1, 0, 1, 0, 0, -1, -1, 0, 0, 1])
    np.testing.assert_array_equal(condition_event_kernel(values, 1, 2), [-1, 0, 0, 0, 1, 1, -1, 0, 1, 0, 0])
    np.testing.assert_array_equal(condition_event_kernel(values, 2, 2), [-1, 1, 0, 0, 1, 0, -1, 1, 0, 0, 1])
    np.testing.assert_allclose(condition_values_kernel(values), np.where(values < 0, np.nan, values), equal_nan=True)
    with pytest.raises(ValueError):
        condition_values_kernel(np.array([2], dtype=np.int64))


def test_next_open_t_plus_one_and_no_terminal_forced_exit():
    prices = quote(4)
    prices[1][1], prices[2][1] = 150, 50  # Entry day touches both; T+1 prohibits exit.
    entry = np.array([1, 0, 0, 0], dtype=np.int64)
    path, trades, count, status = run(prices, entry, take_profit=0.1, stop_loss=0.1)
    assert status == count == 0
    np.testing.assert_array_equal(path[:, 7], [0, 1, 0, 0])
    assert path[-1, 4] == 1
    assert trades[0, 0] == 0 and trades[0, 1] == 1 and trades[0, 2] == -1


@pytest.mark.parametrize("open_price,high,low,reason,exit_price", [
    (100, 120, 80, 2, 90), (80, 120, 70, 2, 80), (120, 125, 80, 3, 120),
])
def test_stop_priority_and_open_gap_execution(open_price, high, low, reason, exit_price):
    prices = quote(4)
    prices[0][2], prices[1][2], prices[2][2] = open_price, high, low
    path, trades, count, status = run(prices, np.array([1, 0, 0, 0], dtype=np.int64),
                                     take_profit=0.1, stop_loss=0.1)
    assert status == 0 and count == 1
    assert path[2, 8] == reason and trades[0, 4] == pytest.approx(exit_price)
    assert trades[0, 5] == pytest.approx(exit_price / 100 - 1)


def test_rule_exit_precedes_protective_exit_and_conflicting_entry():
    prices = quote(5)
    prices[1][2], prices[2][2] = 120, 80
    entry = np.array([1, 1, 0, 0, 0], dtype=np.int64)
    exit_ = np.array([0, 1, 0, 0, 0], dtype=np.int64)
    path, trades, count, status = run(prices, entry, exit_, take_profit=0.1, stop_loss=0.1)
    assert status == 0 and count == 1
    assert path[2, 7] == -1 and trades[0, 7] == 1 and trades[0, 4] == 100
    flat, _, count, _ = run(quote(3), np.ones(3, dtype=np.int64), np.ones(3, dtype=np.int64))
    assert count == 0 and np.all(flat[:, 4] == 0)


def test_round_trip_costs_and_slippage_match_cash_account():
    prices = quote(4)
    entry = np.array([1, 0, 0, 0], dtype=np.int64)
    exit_ = np.array([0, 1, 0, 0], dtype=np.int64)
    path, trades, count, status = run(prices, entry, exit_, fee_bps=10.0, slippage_bps=20.0)
    expected = (1 - 0.002) * (1 - 0.001) / ((1 + 0.002) * (1 + 0.001))
    assert status == 0 and count == 1
    assert path[2, 0] == pytest.approx(expected)
    assert trades[0, 5] == pytest.approx(expected - 1)
    assert path[1, 6] > 0 and path[2, 6] > 0


def test_maximum_holding_and_cooldown_are_counted_on_full_axis():
    path, trades, count, status = run(quote(9), np.ones(9, dtype=np.int64), max_holding=2, cooldown=2)
    assert status == 0 and count == 2
    np.testing.assert_array_equal(path[:, 7], [0, 1, 0, -1, 0, 0, 1, 0, -1])
    np.testing.assert_array_equal(trades[:count, 6], [2, 2])


def test_missing_market_data_fails_without_bridging_gap():
    prices = quote(7)
    prices[3][3] = np.nan
    path, trades, count, status = run(prices, np.ones(7, dtype=np.int64))
    assert status == 1 and count == 0
    assert np.isnan(path[3:, 0]).all() and np.all(path[3:, 9] == 0)
    summary = analyze_kernel(path, trades, count, 0, 7)
    assert np.isnan(summary[1]) and summary[0] == 3
    invalid = quote(3)
    invalid[2][1] = 101
    assert run(invalid)[3] == 2


def test_summary_period_rebases_and_no_trade_statistics_are_missing():
    prices = [np.array([100., 100., 110., 121.]) for _ in range(4)]
    path, trades, count, _ = run(prices, np.ones(4, dtype=np.int64))
    full = dict(zip(SUMMARY_COLUMNS, analyze_kernel(path, trades, count, 0, 4)))
    period = dict(zip(SUMMARY_COLUMNS, analyze_kernel(path, trades, count, 3, 4)))
    assert full["total_return"] == pytest.approx(0.21)
    assert period["total_return"] == pytest.approx(0.1)
    assert full["trade_count"] == 0 and np.isnan(full["win_rate"])
    assert np.isnan(full["mean_trade_return"]) and np.isnan(full["median_trade_return"])
    path, trades, count, status = run(quote(6), start=3, end=3)
    assert path.shape == (6, len(PATH_COLUMNS)) and count == status == 0
    assert analyze_kernel(path, trades, count, 3, 3)[0] == 0
    with pytest.raises(ValueError):
        run(quote(3), start=-1)


def test_signal_quality_horizons_are_contained_and_not_compressed():
    prices = [np.arange(100., 120.) for _ in range(4)]
    entry = np.zeros(20, dtype=np.int64)
    entry[[0, 10, 17]] = 1
    quality = signal_quality_kernel(*prices, entry, 0, 20)
    assert quality.shape == (3, len(QUALITY_COLUMNS))
    np.testing.assert_array_equal(quality[:, 2], [2, 1, 1])
    assert quality[0, 4] == pytest.approx(((105/101 - 1) + (115/111 - 1)) / 2)
    prices[3][3] = np.nan
    quality = signal_quality_kernel(*prices, entry, 0, 20)
    np.testing.assert_array_equal(quality[:, 2], [1, 0, 0])


def test_simulated_history_is_invariant_to_future_changes():
    base = quote(20)
    altered = [values.copy() for values in base]
    for values in altered:
        values[12:] = 200
    entry = np.ones(20, dtype=np.int64)
    np.testing.assert_array_equal(run(base, entry)[0][:12], run(altered, entry)[0][:12])


def test_readonly_strided_views_keep_owners_and_never_recompile_or_mutate():
    prices_owner = np.full((24, 4), 100.0)
    conditions_owner = np.zeros(24, dtype=np.int64)
    conditions_owner[0] = 1
    prices = [prices_owner[::2, column] for column in range(4)]
    conditions = conditions_owner[::2]
    for array in [*prices, conditions]:
        array.setflags(write=False)
    assert all(np.shares_memory(p, prices_owner) for p in prices)
    assert np.shares_memory(conditions, conditions_owner)
    before = {name: list(kernel.signatures) for name, kernel in KERNELS.items()}
    signatures = [list(kernel.signatures) for kernel in (condition_compare_kernel, condition_logic_kernel,
                                                       condition_valid_kernel, select_state_kernel)]
    run(prices, conditions)
    alpha_beta_kernel(prices[0], 0.4, 0.1)
    condition_values_kernel(conditions)
    condition_compare_kernel(prices[0], prices[1], 0.0, 45)
    condition_logic_kernel(conditions, conditions, 0)
    condition_valid_kernel(prices[0])
    select_state_kernel(conditions, conditions, conditions, 1, 0)
    np.testing.assert_array_equal(prices_owner, 100.0)
    np.testing.assert_array_equal(conditions_owner, [1] + [0] * 23)
    assert before == {name: list(kernel.signatures) for name, kernel in KERNELS.items()}
    assert signatures == [list(kernel.signatures) for kernel in (condition_compare_kernel, condition_logic_kernel,
                                                                condition_valid_kernel, select_state_kernel)]
    assert warm_timing_kernels()["complete"]
    assert timing_execution_audit()["python_fallback"] == 0
    with pytest.raises(TypeError):
        alpha_beta_kernel(np.ones(3, dtype=np.float32), 0.4, 0.1)
    with pytest.raises(TypeError):
        condition_values_kernel(np.ones(3, dtype=object))

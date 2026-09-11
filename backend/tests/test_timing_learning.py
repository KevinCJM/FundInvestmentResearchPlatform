"""ETF train/freeze contracts, numerical reference and zero-copy execution."""
import numpy as np
import pytest

from backend.timing_research import learning as lk
from backend.timing_research.numeric import TRADE_COLUMNS


def trades(outcomes, signals=None, exits=None, reasons=None):
    size = len(outcomes)
    signal = np.arange(size) * 3 if signals is None else np.asarray(signals)
    output = np.full((size, len(TRADE_COLUMNS)), np.nan)
    output[:, 0], output[:, 1] = signal, signal + 1
    output[:, 2] = signal + 2 if exits is None else exits
    output[:, 3:5] = 100.0
    output[:, 5] = outcomes
    output[:, 6] = output[:, 2] - output[:, 1]
    output[:, 7] = 1 if reasons is None else reasons
    return output


def score(rows, states=None, **kwargs):
    values = np.zeros(30, dtype=np.int64) if states is None else states
    params = dict(count=rows.shape[0], start=0, split=len(values), n_states=1,
                  min_trades=2, confidence=1.0, risk_penalty=0.1)
    params.update(kwargs)
    return lk.score_trades_kernel(rows, states=values, **params)


def readonly_strided(values):
    # This is test-boundary fixture allocation, not a production input copy.
    shape = tuple(size * 2 for size in values.shape)
    owner = np.empty(shape, dtype=values.dtype)
    view = owner[tuple(slice(None, None, 2) for _ in values.shape)]
    view[...] = values
    view.setflags(write=False)
    assert np.shares_memory(owner, view) or not view.size
    return owner, view


@pytest.fixture(autouse=True, scope="module")
def warmed():
    audit = lk.warm_learning_kernels()
    assert audit["complete"] and audit["nopython"]
    assert audit["python_fallback"] == audit["request_time_compilation"] == 0
    assert set(audit["kernel_signatures"]) == set(lk.KERNELS)


def test_state_encoding_has_explicit_order_unknowns_and_empty_axes():
    conditions = np.array([[0, 1, 0, 1, -1], [0, 0, 1, 1, 1], [1, 0, 1, 0, 0]], dtype=np.int64)
    np.testing.assert_array_equal(lk.encode_states_kernel(conditions), [4, 1, 6, 3, -1])
    np.testing.assert_array_equal(lk.encode_states_kernel(np.empty((0, 3), dtype=np.int64)), [0, 0, 0])
    assert lk.encode_states_kernel(np.empty((3, 0), dtype=np.int64)).size == 0
    with pytest.raises(ValueError, match="three"):
        lk.encode_states_kernel(np.zeros((4, 1), dtype=np.int64))


@pytest.mark.parametrize("invalid", [-2, 2, np.iinfo(np.int64).max])
def test_encoding_rejects_invalid_codes_even_after_unknown(invalid):
    with pytest.raises(ValueError, match="-1, 0 or 1"):
        lk.encode_states_kernel(np.array([[-1], [invalid]], dtype=np.int64))


def test_grouped_utility_matches_sample_std_reference_and_zero_is_not_win():
    rows = trades([0.1, -0.2, 0.0, 0.4, 0.2], reasons=[1, 2, 4, 3, 1])
    states = np.zeros(30, dtype=np.int64)
    states[9] = states[12] = 1
    result = score(rows, states, n_states=3, confidence=1.5, risk_penalty=0.2)
    for state, outcomes, stop_rate in ((0, [0.1, -0.2, 0.0], 1 / 3), (1, [0.4, 0.2], 0.0)):
        expected = [len(outcomes), np.mean(outcomes), np.mean(np.asarray(outcomes) > 0),
                    np.mean(outcomes) - 1.5 * np.std(outcomes, ddof=1) / np.sqrt(len(outcomes)) - 0.2 * stop_rate,
                    stop_rate]
        np.testing.assert_allclose(result[state], expected, rtol=1e-13, atol=1e-14)
    assert result[2, 0] == 0 and np.isnan(result[2, 1:]).all()
    assert tuple(lk.SCORE_COLUMNS) == ("count", "mean", "win_rate", "utility", "stop_rate")


def test_training_uses_signal_state_start_and_strict_mature_exit_not_entry_state():
    rows = trades([10.0, 0.1, 0.3, 50.0, 60.0, 70.0],
                  signals=[0, 2, 7, 8, 9, 5], exits=[2, 4, 9, 10, 11, -1])
    states = np.ones(16, dtype=np.int64)
    states[2] = states[7] = 0
    result = score(rows, states, start=2, split=10, n_states=2,
                   confidence=0.0, risk_penalty=0.0)
    np.testing.assert_allclose(result[0], [2, .2, 1, .2, 0])
    assert result[1, 0] == 0 and np.isnan(result[1, 1:]).all()
    altered = rows.copy()
    altered[3:, 5] = [-5000, 5000, -999]
    np.testing.assert_allclose(score(altered, states, start=2, split=10, n_states=2,
                                     confidence=0., risk_penalty=0.), result, equal_nan=True)


def test_unknown_state_and_nonfinite_required_fields_do_not_become_labels():
    rows = trades([.1, .2, .3, .4, .5])
    rows[1, 5] = np.nan
    rows[2, 4] = np.inf
    rows[3, 7] = np.nan
    states = np.zeros(30, dtype=np.int64)
    states[12] = -1
    result = score(rows, states)
    np.testing.assert_allclose(result[0, [0, 1, 2, 4]], [1, .1, 1, 0])
    assert np.isnan(result[0, 3])


def test_nan_excursions_on_mature_stop_trades_do_not_remove_risk_penalty():
    rows = trades([-.1, -.2], reasons=[2, 2])
    assert np.isnan(rows[:, 8:]).all()
    result = score(rows, confidence=0.0, risk_penalty=0.3)
    np.testing.assert_allclose(result[0], [2, -.15, 0, -.45, 1])


def test_empty_and_insufficient_samples_do_not_invent_zero_returns_or_utility():
    empty = score(trades([]))
    assert empty[0, 0] == 0 and np.isnan(empty[0, 1:]).all()
    one = score(trades([.1]), min_trades=1, confidence=0.0)
    assert one[0, 0] == 1 and one[0, 1] == .1 and np.isnan(one[0, 3])
    two = score(trades([.1, .1]), min_trades=3)
    assert two[0, 0] == 2 and np.isnan(two[0, 3])
    equal = score(trades([.1, .1]), min_trades=2)
    assert equal[0, 3] == pytest.approx(.1)
    # Count excludes unused capacity, whose NaN/garbage rows are not inspected.
    padded = np.concatenate((trades([.1, .1]), np.full((2, 10), np.nan)))
    np.testing.assert_allclose(score(padded, count=2), equal, equal_nan=True)


@pytest.mark.parametrize("kwargs", [dict(count=-1), dict(count=9), dict(start=-1), dict(split=31),
                                    dict(start=10, split=5), dict(n_states=0), dict(n_states=13),
                                    dict(min_trades=0), dict(confidence=-1.), dict(confidence=np.nan),
                                    dict(risk_penalty=-.1), dict(risk_penalty=np.inf)])
def test_invalid_training_contracts_are_rejected(kwargs):
    with pytest.raises(ValueError):
        score(trades([.1, .2]), **kwargs)


@pytest.mark.parametrize("column,value", [(0, .5), (1, .5), (2, 1.5), (3, 0.),
                                           (4, -1.), (6, 7.), (7, 2.5), (7, 5.)])
def test_mature_trade_structure_fails_closed(column, value):
    rows = trades([.1])
    rows[0, column] = value
    with pytest.raises(ValueError, match="Mature trade"):
        score(rows)


def test_training_axis_and_state_validation():
    with pytest.raises(ValueError):
        score(np.empty((0, 9)))
    with pytest.raises(ValueError, match="State code"):
        score(trades([.1]), np.full(30, 2, dtype=np.int64), n_states=2)


def test_selection_uses_strict_floor_finite_support_and_fixed_ties():
    utilities = np.array([[.1, .1, .0], [.0, -.1, np.nan], [np.inf, np.nan, -.1],
                          [.01, .2, .1], [-np.inf, np.nan, np.nan]])
    np.testing.assert_array_equal(lk.choose_actions_kernel(utilities, .0, -1), [0, -1, -1, 1, -1])
    np.testing.assert_array_equal(lk.choose_actions_kernel(utilities, .1, -1), [-1, -1, -1, 1, -1])
    # A designated cash column is ignored even if given an accidental high score.
    np.testing.assert_array_equal(lk.choose_actions_kernel(np.array([[99., .0], [99., .2]]), .0, 0), [0, 1])
    np.testing.assert_array_equal(lk.choose_actions_kernel(np.empty((2, 0)), .0, -1), [-1, -1])


@pytest.mark.parametrize("matrix,floor,cash", [(np.ones((13, 1)), .0, -1), (np.ones((0, 1)), .0, -1),
                                               (np.ones((2, 1)), np.nan, -1),
                                               (np.ones((2, 1)), .0, -2), (np.ones((2, 1)), .0, 1)])
def test_invalid_selection_contracts(matrix, floor, cash):
    with pytest.raises(ValueError):
        lk.choose_actions_kernel(matrix, floor, cash)


def test_routing_preserves_close_open_boundary_unknowns_and_cash():
    signals = np.array([[1, 0, 1, 1, 0, -1, 1, 1], [0, 1, 0, 1, 1, 0, 0, 1]], dtype=np.int64)
    states = np.array([0, 0, 0, 1, -1, 0, 2, 1], dtype=np.int64)
    selected = np.array([0, 1, -1], dtype=np.int64)
    np.testing.assert_array_equal(lk.route_actions_kernel(signals, states, selected, 4),
                                  [-1, -1, -1, 1, -1, -1, 0, 1])
    assert lk.route_actions_kernel(signals, states, selected, 8)[6] == -1
    assert lk.route_actions_kernel(signals, states, selected, 8)[7] == 1
    assert lk.route_actions_kernel(signals, states, selected, 0)[0] == 1


def test_future_features_do_not_change_a_frozen_routing_prefix():
    signals = np.ones((2, 20), dtype=np.int64)
    signals[1] = 0
    states = np.arange(20, dtype=np.int64) % 2
    selected = np.array([0, 1], dtype=np.int64)
    baseline = lk.route_actions_kernel(signals, states, selected, 6)
    signals[:, 12:] = -1
    states[12:] = -1
    altered = lk.route_actions_kernel(signals, states, selected, 6)
    np.testing.assert_array_equal(baseline[:12], altered[:12])
    encoded = lk.encode_states_kernel(np.vstack((np.arange(20) % 2, np.zeros(20, dtype=np.int64))))
    np.testing.assert_array_equal(encoded[:12], states[:12])


def test_invalid_routing_axes_codes_and_indices_are_rejected():
    signals = np.ones((2, 5), dtype=np.int64)
    states = np.zeros(5, dtype=np.int64)
    selected = np.array([0], dtype=np.int64)
    for args in [(signals, states[:4], selected, 2), (signals, states, selected, -1),
                 (signals, states, selected, 6), (signals, states, np.array([2], dtype=np.int64), 2),
                 (signals, states, np.array([-2], dtype=np.int64), 2),
                 (signals, states, np.empty(0, dtype=np.int64), 2)]:
        with pytest.raises(ValueError):
            lk.route_actions_kernel(*args)
    for invalid in (-2, 1):
        states[4] = invalid
        with pytest.raises(ValueError, match="State code"):
            lk.route_actions_kernel(signals, states, selected, 2)
    states[4] = 0
    signals[1, 0] = 2  # Even unused/early lanes cannot carry illegal conditions.
    with pytest.raises(ValueError, match="-1, 0 or 1"):
        lk.route_actions_kernel(signals, states, selected, 2)


def test_readonly_strided_inputs_are_not_copied_mutated_or_recompiled(monkeypatch):
    signatures = {name: tuple(kernel.signatures) for name, kernel in lk.KERNELS.items()}
    _, conditions = readonly_strided(np.array([[0, 1, 0, 1] * 5], dtype=np.int64))
    _, rows = readonly_strided(trades([.1, -.1, .2, .0], reasons=[1, 2, 3, 4]))
    _, states = readonly_strided(np.zeros(20, dtype=np.int64))
    _, utility = readonly_strided(np.array([[.1, .2]]))
    _, selected = readonly_strided(np.array([1], dtype=np.int64))
    _, signals = readonly_strided(np.vstack((np.zeros(20, dtype=np.int64), np.ones(20, dtype=np.int64))))
    snapshots = [array.copy() for array in (conditions, rows, states, utility, selected, signals)]
    def forbidden(*args, **kwargs):
        raise AssertionError("Python numerical fallback executed")
    for kernel in lk.KERNELS.values():
        monkeypatch.setattr(kernel, "py_func", forbidden)
    assert lk.encode_states_kernel(conditions).flags.c_contiguous
    assert score(rows, states)[0, 0] == 4
    np.testing.assert_array_equal(lk.choose_actions_kernel(utility, .0, -1), [1])
    assert lk.route_actions_kernel(signals, states, selected, 10)[9] == 1
    for array, snapshot in zip((conditions, rows, states, utility, selected, signals), snapshots):
        np.testing.assert_array_equal(array, snapshot)
        assert not array.flags.writeable
    assert signatures == {name: tuple(kernel.signatures) for name, kernel in lk.KERNELS.items()}
    assert all(not kernel._can_compile for kernel in lk.KERNELS.values())


def test_unsupported_array_dtypes_cannot_trigger_new_compilation():
    with pytest.raises(TypeError):
        lk.encode_states_kernel(np.zeros((1, 2), dtype=np.int32))
    with pytest.raises(TypeError):
        lk.encode_states_kernel(np.zeros((1, 2), dtype=object))
    with pytest.raises(TypeError):
        score(trades([.1]).astype(np.float32))
    with pytest.raises(TypeError):
        lk.choose_actions_kernel(np.ones((1, 2), dtype=np.float32), .0, -1)
    assert lk.learning_execution_audit()["kernel_signatures"]


def test_calendar_states_support_twelve_months_without_changing_bit_encoding():
    rows = trades([.1, .2])
    states = np.full(30, 11, dtype=np.int64)
    result = score(rows, states, n_states=12)
    assert result.shape == (12, 5) and result[11, 0] == 2
    selected = lk.choose_actions_kernel(result[:, 3:4], .0, -1)
    assert selected.shape == (12,) and selected[11] == 0
    routed = lk.route_actions_kernel(np.ones((1, 30), dtype=np.int64), states, selected, 10)
    assert routed[9] == 1 and routed[8] == -1


def test_priority_quota_core_wins_and_counts_only_emitted_supplements():
    core = np.array([0, 1, 0, 0, 1, 0, 0, 0], dtype=np.int64)
    supplement = np.ones(8, dtype=np.int64)
    months = np.full(8, 202601, dtype=np.int64)
    result = lk.priority_quota_kernel(core, supplement, months, 2, 2, 0)
    # First two supplements emit; then the supplement cap/core count both block.
    np.testing.assert_array_equal(result, [1, 1, 1, 0, 1, 0, 0, 0])
    # Core quota is an exclusive boundary, never a cap on the core lane itself.
    np.testing.assert_array_equal(lk.priority_quota_kernel(core, supplement, months, 1, 9, 0),
                                  [1, 1, 0, 0, 1, 0, 0, 0])
    np.testing.assert_array_equal(lk.priority_quota_kernel(core, supplement, months, 0, 9, 0), core)
    np.testing.assert_array_equal(lk.priority_quota_kernel(core, supplement, months, 9, 0, 0), core)


def test_priority_quota_month_reset_and_cooldown_span_month_boundary():
    core = np.zeros(8, dtype=np.int64)
    supplement = np.ones(8, dtype=np.int64)
    months = np.array([202601] * 3 + [202602] * 3 + [202603] * 2, dtype=np.int64)
    # Distance equal to cooldown remains blocked, including at a month boundary.
    np.testing.assert_array_equal(lk.priority_quota_kernel(core, supplement, months, 1, 1, 3),
                                  [1, 0, 0, 0, 1, 0, 0, 0])
    np.testing.assert_array_equal(lk.priority_quota_kernel(core, supplement, months, 1, 1, 0),
                                  [1, 0, 0, 1, 0, 0, 1, 0])


def test_priority_quota_unknown_is_not_false_and_does_not_claim_a_candidate():
    core = np.array([1, -1, 0, 0, 0], dtype=np.int64)
    supplement = np.array([-1, 1, -1, 1, 1], dtype=np.int64)
    months = np.zeros(5, dtype=np.int64)
    np.testing.assert_array_equal(lk.priority_quota_kernel(core, supplement, months, 2, 1, 0),
                                  [1, -1, -1, 1, 0])


def test_priority_quota_has_prefix_invariance_and_readonly_stride_contract():
    core = np.array([0, 0, 1, 0, 0, 0, 1, 0], dtype=np.int64)
    supplement = np.ones(8, dtype=np.int64)
    months = np.zeros(8, dtype=np.int64)
    owners_and_views = [readonly_strided(array) for array in (core, supplement, months)]
    views = [pair[1] for pair in owners_and_views]
    signatures = tuple(lk.priority_quota_kernel.signatures)
    result = lk.priority_quota_kernel(*views, 2, 2, 1)
    np.testing.assert_array_equal(lk.priority_quota_kernel(*(view[:4] for view in views), 2, 2, 1), result[:4])
    core[4:] = 1
    supplement[4:] = -1
    months[4:] = 202603
    np.testing.assert_array_equal(lk.priority_quota_kernel(core, supplement, months, 2, 2, 1)[:4], result[:4])
    assert signatures == tuple(lk.priority_quota_kernel.signatures)
    for owner, view in owners_and_views:
        assert np.shares_memory(owner, view) and not view.flags.writeable


def test_priority_quota_rejects_invalid_axes_codes_quotas_and_month_order():
    base = np.zeros(4, dtype=np.int64)
    assert lk.priority_quota_kernel(base[:0], base[:0], base[:0], 0, 0, 0).size == 0
    for args in [(base, base[:3], base, 1, 1, 0), (base, base, base, -1, 1, 0),
                 (base, base, base, 1, -1, 0), (base, base, base, 1, 1, -1),
                 (base, base, np.array([0, 1, 0, 0], dtype=np.int64), 1, 1, 0),
                 (base, base, np.array([0, 1, 1, -1], dtype=np.int64), 1, 1, 0),
                 (np.array([0, 0, 2, 0], dtype=np.int64), base, base, 1, 1, 0)]:
        with pytest.raises(ValueError):
            lk.priority_quota_kernel(*args)


def prepared_training_case(embargo=2, max_holding=2):
    from types import SimpleNamespace
    from backend.timing_research.catalog import build_catalog
    from backend.timing_research.contracts import Definition
    from backend.timing_research.graph import GraphRuntime
    from backend.timing_research.training_runtime import prepare_candidates
    runtime = GraphRuntime(build_catalog())
    definition = Definition(name="成熟标签边界", nodes=[
        {"id": "price", "label": "收盘价", "op": "source", "parameters": {"field": "close"}},
        {"id": "smooth", "label": "均线", "op": "formula", "inputs": {"a": "price.value"},
         "parameters": {"expression": "rolling_mean(a,2,2)"}},
        {"id": "entry", "label": "入场", "op": "compare", "inputs": {"left": "smooth.value"},
         "parameters": {"threshold": 0., "operator": "gt"}},
    ], entry="entry.value", training={"actions": [{"id": "long", "label": "持有", "entry": "entry.value"}],
        "embargo_bars": embargo, "min_trades": 2, "confidence": 0., "risk_penalty": 0.},
        execution={"take_profit": .5, "stop_loss": .5, "max_holding_bars": max_holding,
                   "fee_bps": 0., "slippage_bps": 0.})
    dates = np.arange(np.datetime64("2024-01-01"), np.datetime64("2024-01-17"), dtype="datetime64[D]").astype(np.int64)
    close = np.arange(16, dtype=np.float64) + 100.
    fields = dict(dates=dates, close=close, open=close, high=close + .1, low=close - .1)
    for array in fields.values():
        array.setflags(write=False)
    bars = SimpleNamespace(**fields)
    base = runtime.prepare(definition)
    candidates = prepare_candidates(definition, runtime)
    channels = runtime.evaluate(base, bars, {})
    return definition, runtime, candidates, bars, channels


def test_training_runtime_embargo_is_exclusive_and_unclosed_tail_is_not_a_label():
    from backend.timing_research.numeric import simulate_kernel
    from backend.timing_research.training_runtime import fit_and_route
    definition, runtime, candidates, bars, channels = prepared_training_case()
    observations = []
    def simulate(data, entry, exit_, start, end, execution):
        path, rows, count, status = simulate_kernel(data.open, data.high, data.low, data.close, entry, exit_,
            start, end, execution.max_holding_bars, execution.cooldown_bars, execution.fee_bps,
            execution.slippage_bps, execution.take_profit, execution.stop_loss)
        assert status == 0
        observations.append((start, end, rows[:count, 2].copy(), path[end - 1, 4], rows[count, 2]))
        return path, rows, count
    routed, audit = fit_and_route(definition.training, candidates, runtime, bars, {}, channels, 0, 11, simulate)
    assert len(observations) == 1
    start, end, exits, tail_position, tail_exit = observations[0]
    assert (start, end) == (0, 9)
    np.testing.assert_array_equal(exits, [4, 7])
    assert tail_position == 1. and tail_exit == -1.
    assert audit["candidates"][0]["states"][0]["sample_count"] == 2
    assert audit["freeze_date"] == "2024-01-11"
    assert audit["fit_end_date"] == "2024-01-09"
    assert routed[9] == -1 and routed[10] == 1 and not routed.flags.writeable
    # With no embargo the exit at index 10 becomes mature, rather than forcing
    # the open index-8 trade to close at the earlier fitting boundary.
    training = definition.training.model_copy(update={"embargo_bars": 0})
    _, all_train = fit_and_route(training, candidates, runtime, bars, {}, channels, 0, 11, simulate)
    assert all_train["candidates"][0]["states"][0]["sample_count"] == 3


def test_training_runtime_never_forces_a_long_open_trade_into_training_evidence():
    from backend.timing_research.service import TimingResearchService
    from backend.timing_research.training_runtime import fit_and_route
    definition, runtime, candidates, bars, channels = prepared_training_case(max_holding=100)
    routed, audit = fit_and_route(definition.training, candidates, runtime, bars, {}, channels,
                                  0, 11, TimingResearchService._simulate)
    assert audit["candidates"][0]["states"][0]["sample_count"] == 0
    assert audit["selection"][0]["action_id"] is None
    np.testing.assert_array_equal(routed[10:], np.zeros(6, dtype=np.int64))


def test_training_runtime_uses_prepared_candidate_and_fixed_dispatchers_only(monkeypatch):
    from backend.timing_research.service import TimingResearchService
    from backend.timing_research.training_runtime import fit_and_route
    import backend.timing_research.graph as graph
    definition, runtime, candidates, bars, channels = prepared_training_case()
    def forbidden(*args, **kwargs):
        pytest.fail("Training request attempted to prepare/compile a new plan")
    monkeypatch.setattr(runtime, "prepare", forbidden)
    monkeypatch.setattr(graph, "compile_numba_series_plan", forbidden)
    signatures = {key: tuple(kernel.signatures) for key, kernel in lk.KERNELS.items()}
    before = {key: tuple(plan.compiled_signatures) for key, plan in candidates[0].graph.formulas.items()}
    routed, audit = fit_and_route(definition.training, candidates, runtime, bars, {}, channels,
                                  0, 11, TimingResearchService._simulate)
    assert audit["selection"][0]["action_id"] == "long:0"
    assert routed[10] == 1
    assert signatures == {key: tuple(kernel.signatures) for key, kernel in lk.KERNELS.items()}
    assert before == {key: tuple(plan.compiled_signatures) for key, plan in candidates[0].graph.formulas.items()}


def test_training_runtime_rejects_embargo_exhaustion_before_any_evaluation(monkeypatch):
    from backend.timing_research.service import TimingResearchService
    from backend.timing_research.training_runtime import fit_and_route
    from custom_indicators.errors import ValidationError
    definition, runtime, candidates, bars, channels = prepared_training_case()
    monkeypatch.setattr(runtime, "evaluate", lambda *_: pytest.fail("An invalid training interval was evaluated"))
    for embargo in (11, 12):
        training = definition.training.model_copy(update={"embargo_bars": embargo})
        with pytest.raises(ValidationError, match="隔离期"):
            fit_and_route(training, candidates, runtime, bars, {}, channels, 0, 11, TimingResearchService._simulate)

"""Numerical, causal and service contracts for index/filter/noise regimes."""
import copy
import json
from datetime import date, timedelta

import numpy as np
import pytest

from historical_regimes.trend_numba import (
    TREND_KERNELS, kama_kernel, merge_short_regimes_kernel,
    super_smoother_kernel, trend_features_kernel, trend_regime_kernel,
)
from historical_regimes.v2_contracts import inspect_definition_v2, parse_definition_v2
from historical_regimes.v2_numba import confirmation_state_kernel, regime_graph_numba_status
from historical_regimes.v2_registry import NODE_REGISTRY
from historical_regimes.v2_service import RegimeGraphV2Service, _kernel_ids_for_definition
from historical_regimes.v2_templates import get_template_v2, instantiate_template_v2


def classify(d, s, er=None, confirmation=3):
    d, s = np.asarray(d, dtype=np.float64), np.asarray(s, dtype=np.float64)
    er = np.full(len(d), 0.8) if er is None else np.asarray(er, dtype=np.float64)
    return trend_regime_kernel(d, s, er, 1.0, 0.1, 0.05, 0.25, confirmation)


def features(x, f):
    return trend_features_kernel(x, f, 5, 3, 6, 0.0001, 3, 0.2, 0.08)


@pytest.mark.parametrize("period", [3, 20, 126])
def test_super_smoother_matches_independent_transfer_function(period):
    # scipy is a controlled reference in tests, never a production fallback.
    from scipy.signal import lfilter
    x = np.random.default_rng(12).normal(size=400).cumsum() + 100
    angle = np.sqrt(2) * np.pi / period
    a = np.exp(-angle)
    c2, c3 = 2 * a * np.cos(angle), -a * a
    c1 = 1 - c2 - c3
    # Recurrence begins at t=2 with F0=x0,F1=x1.
    zi = [c1 * x[1] / 2 + c2 * x[1] + c3 * x[0], c3 * x[1]]
    reference, _ = lfilter([c1 / 2, c1 / 2], [1, -c2, -c3], x[2:], zi=zi)
    actual = super_smoother_kernel(x, period)
    assert np.isnan(actual[:period - 1]).all()
    np.testing.assert_allclose(actual[period - 1:], reference[period - 3:], atol=1e-10)


def test_kama_reference_and_gap_reinitialization():
    x = np.random.default_rng(5).normal(size=100).cumsum() + 100
    reference = np.full(100, np.nan)
    previous = x[0]
    for t in range(10, 100):
        er = abs(x[t] - x[t - 10]) / np.abs(np.diff(x[t - 10:t + 1])).sum()
        previous += (er * (2 / 3 - 2 / 31) + 2 / 31) ** 2 * (x[t] - previous)
        reference[t] = previous
    np.testing.assert_allclose(kama_kernel(x, 10, 2, 30), reference, equal_nan=True)
    x[30] = np.inf
    actual = kama_kernel(x, 10, 2, 30)
    assert np.isnan(actual[30:41]).all()
    np.testing.assert_allclose(actual[31:], kama_kernel(x[31:].copy(), 10, 2, 30), equal_nan=True)


def test_prior_rms_efficiency_slope_and_raw_risk():
    x = np.log(np.array([100, 101, 102, 101, 102, 103, 104, 75, 76, 77], dtype=float))
    f = super_smoother_kernel(x, 3)
    d, s, er, scale, dd, risk, raw, filtered = features(x, f)
    for t in range(6, len(x)):
        prior = np.diff(x[:t]) ** 2
        v = prior[0]
        for square in prior[1:]:
            v = square / 3 + v * 2 / 3
        assert scale[t] == pytest.approx(max(0.0001, np.sqrt(v)))
        assert d[t] == pytest.approx((x[t] - f[t]) / scale[t])
        assert s[t] == pytest.approx((f[t] - f[t - 3]) / (3 * scale[t]))
        assert er[t] == pytest.approx(abs(x[t] - x[t - 6]) / np.abs(np.diff(x[t - 6:t + 1])).sum())
    assert dd[7] == pytest.approx(75 / 104 - 1)
    assert risk[7] == 1
    np.testing.assert_allclose(raw, np.exp(x))
    np.testing.assert_allclose(filtered, np.exp(f), equal_nan=True)


def test_short_bear_signals_removed_but_sustained_reversal_switches_on_confirmation():
    d = [2] * 4 + [-2] * 2 + [2] * 2 + [-2] * 5
    s = np.sign(d) * 0.2
    states, _, pending, phase = classify(d, s)
    assert states.tolist() == [-1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2]
    assert pending[5] == 2
    assert phase[4] == 1  # bull correction, not hidden raw-price risk


def test_sideways_needs_flatness_and_low_efficiency_and_missing_resets_count():
    d = np.array([2] * 3 + [0] * 3 + [0] * 3 + [-2, np.nan, -2, -2, -2])
    s = np.array([0.2] * 3 + [0] * 6 + [-0.2] * 5)
    er = [0.8] * 6 + [0.1] * 3 + [0.8] * 5
    states, _, pending, _ = classify(d, s, er)
    assert states.tolist() == [-1, -1, 0, 0, 0, 0, 0, 0, 1, 1, -1, 1, 1, 2]
    assert pending[10] == 0


@pytest.mark.parametrize("filter_name", ["super_smoother", "kama"])
def test_future_changes_never_revise_filter_features_or_states(filter_name):
    rng = np.random.default_rng(33)
    x = np.log(100) + np.cumsum(rng.normal(0.0003, 0.01, 300))
    x[125] = np.nan
    def run(values):
        f = (super_smoother_kernel(values, 20) if filter_name == "super_smoother"
             else kama_kernel(values, 10, 2, 30))
        metrics = features(values, f)
        states = classify(*metrics[:3])
        return (f, *metrics, *states)
    full = run(x)
    for end in (0, 1, 19, 20, 21, 90, 126, 150, 299):
        prefix = run(x[:end].copy())
        for actual, expected in zip(prefix, full):
            np.testing.assert_allclose(actual, expected[:end], equal_nan=True, rtol=0, atol=0)
    changed = x.copy()
    changed[200:] += 2
    for actual, expected in zip(run(changed), full):
        np.testing.assert_array_equal(actual[:200], expected[:200])


def test_constant_empty_and_nonfinite_inputs_are_not_bull_or_bear():
    for size in (0, 1, 2, 100):
        x = np.full(size, np.log(100.0))
        f = super_smoother_kernel(x, 20)
        metrics = features(x, f)
        states = classify(*metrics[:3])[0]
        assert not np.isin(states, [0, 2]).any()
        if size == 100:
            assert states[-1] == 1
            assert metrics[2][-1] == 0
    x = np.full(100, np.nan)
    x[20] = np.inf
    assert np.isnan(super_smoother_kernel(x, 20)).all()
    assert (classify(*features(x, x)[:3])[0] == -1).all()


def merge(states, prices, duration=2):
    return merge_short_regimes_kernel(np.asarray(states, dtype=np.int64),
                                     np.asarray(prices, dtype=np.float64), duration, 0.08, 3)


def test_retrospective_merge_preserves_crashes_unknowns_and_unfinished_tail():
    states = [0, 0, 2, 2, 0, 0, 0]
    assert merge(states, [100, 102, 101, 100, 103, 104, 105]).tolist() == [0] * 7
    # Even a recovered endpoint must not conceal the intraphase crash.
    assert merge(states, [100, 120, 90, 120, 121, 122, 123]).tolist() == states
    assert merge(states, [100, 102, 101, 100, np.nan, 104, 105]).tolist() == states
    assert merge(states[:-1], [100, 102, 101, 100, 103, 104]).tolist() == states[:-1]
    unknown = [0, 0, 2, -1, 0, 0, 0]
    assert merge(unknown, [100] * 7).tolist() == unknown
    assert merge([0, 0, 2], [100, 102, 101]).tolist() == [0, 0, 2]
    assert merge([2, 2, 0, 0, 2, 2, 2], [100, 98, 99, 100, 97, 96, 95]).tolist() == [2] * 7
    alternating = [0] * 3 + [2] * 3 + [0] * 3 + [2] * 3 + [0] * 3
    assert merge(alternating, [100] * 15, duration=3).tolist() == [0] * 15


def test_confirmation_does_not_freeze_when_reversal_starts_during_minimum_duration():
    states = np.array([0, 0, 2, 2, 2, 2, 2], dtype=np.int64)
    assert confirmation_state_kernel(states, 2, 4).tolist() == [-1, 0, 0, 0, 0, 2, 2]
    gap = np.array([0, 0, 2, -1, 2, 2], dtype=np.int64)
    assert confirmation_state_kernel(gap, 2, 1).tolist() == [-1, 0, 0, -1, 0, 2]


def definition_with_inline(count=420):
    definition = instantiate_template_v2("bull-bear-causal-v2")
    start = date(2020, 1, 1)
    # Uptrend, downtrend, then stationary range; all three states are attainable.
    x = np.r_[np.arange(160) * 0.005, 0.8 - np.arange(160) * 0.006,
              np.full(100, -0.16)][:count]
    definition["graph"]["nodes"][0] = {
        "id": "market", "type": "source.inline", "parameters": {"frequency": "daily", "rows": [
            {"observation_date": (start + timedelta(days=t)).isoformat(),
             "available_at": (start + timedelta(days=t + 1)).isoformat(), "value": float(100 * np.exp(value))}
            for t, value in enumerate(x)]}}
    definition["graph"]["nodes"][2]["parameters"] = {"period": 20}
    definition["graph"]["nodes"][3]["parameters"].update(volatility_window=10, slope_window=5, efficiency_window=10)
    definition["evaluation_targets"] = []
    return definition


def test_template_executes_actual_njit_chain_with_evidence_and_prefix_stability(tmp_path):
    service = RegimeGraphV2Service(tmp_path, tmp_path)
    raw = definition_with_inline()
    definition = parse_definition_v2(raw)
    assert get_template_v2("bull-bear-causal-v2")["version"] == 2
    assert {"super_smoother", "trend_features", "trend_regime"} <= set(_kernel_ids_for_definition(definition))
    before = {name: list(kernel.signatures) for name, kernel in TREND_KERNELS.items()}
    full = service._execute_graph(None, definition, "realtime", None)
    assert {row["state_id"] for row in full["series"]} == {"unclassified", "bull", "bear", "sideways"}
    assert full["result"]["diagnostics"]["python_fallback"] == 0
    assert full["result"]["diagnostics"]["request_time_compilation"] == 0
    evidence = full["series"][100]["features"]
    assert evidence["filtered_index"] is not None
    assert evidence["index_value"] == pytest.approx(raw["graph"]["nodes"][0]["parameters"]["rows"][100]["value"])
    assert {"distance", "slope", "efficiency", "risk", "phase", "pending_count"} <= evidence.keys()
    short = copy.deepcopy(raw)
    short["graph"]["nodes"][0]["parameters"]["rows"] = short["graph"]["nodes"][0]["parameters"]["rows"][:200]
    prefix = service._execute_graph(None, parse_definition_v2(short), "realtime", None)
    for previous, current in zip(prefix["series"], full["series"]):
        assert previous["state_code"] == current["state_code"]
        assert previous["features"] == current["features"]
    assert before == {name: list(kernel.signatures) for name, kernel in TREND_KERNELS.items()}
    assert regime_graph_numba_status()["complete"] is True
    assert all(kernel.nopython_signatures and not kernel._can_compile for kernel in TREND_KERNELS.values())
    with pytest.raises(TypeError):
        super_smoother_kernel(np.ones(30, dtype=np.float32), 20)


def test_retrospective_node_is_rejected_before_any_realtime_source_read(tmp_path, monkeypatch):
    service = RegimeGraphV2Service(tmp_path, tmp_path)
    raw = definition_with_inline()
    raw["graph"]["nodes"].append({"id": "merge", "type": "post.merge_short_regimes", "inputs": {
        "state": {"node_id": "classifier", "port": "state"}, "price": {"node_id": "market", "port": "value"}}})
    raw["graph"]["outputs"]["state"] = {"node_id": "merge", "port": "state"}
    definition = parse_definition_v2(raw)
    assert inspect_definition_v2(definition)["valid"]
    def unexpected(*args, **kwargs):
        pytest.fail("realtime must reject before reading sources")
    monkeypatch.setattr(service, "_resolve_sources", unexpected)
    with pytest.raises(Exception, match="实时"):
        service._execute_graph(None, definition, "realtime", None)
    assert NODE_REGISTRY["post.merge_short_regimes"]["supports_realtime"] is False


def test_kama_and_retrospective_service_execution_persist_evidence(tmp_path):
    import pyarrow.parquet as pq
    service = RegimeGraphV2Service(tmp_path, tmp_path)
    raw = definition_with_inline()
    raw["graph"]["nodes"][2].update(type="filter.kama", parameters={"window": 10, "fast": 2, "slow": 30})
    raw["graph"]["nodes"].append({"id": "merge", "type": "post.merge_short_regimes", "inputs": {
        "state": {"node_id": "classifier", "port": "state"}, "price": {"node_id": "market", "port": "value"}}})
    raw["graph"]["outputs"]["state"] = {"node_id": "merge", "port": "state"}
    definition = parse_definition_v2(raw)
    assert {"kama", "merge_short_regimes"} <= set(_kernel_ids_for_definition(definition))
    output = service._execute_graph(None, definition, "retrospective", None)
    assert {"bull", "bear"} <= {row["state_id"] for row in output["series"]}
    assert output["series"][100]["features"]["filtered_index"] is not None
    manifest = service._persist_series(output["series"])
    stored = pq.read_table(service.artifact_dir / (manifest["checksum"].split(":")[1] + ".parquet")).to_pylist()
    assert json.loads(stored[100]["features_json"]) == output["series"][100]["features"]


@pytest.mark.parametrize("node_id,parameters", [
    ("trend", {"period": 2}), ("trend", {"period": 3.5}),
    ("metrics", {"volatility_window": 0}), ("metrics", {"shock_alert": 1}),
    ("classifier", {"confirmation": 0}), ("classifier", {"flat_threshold": 0.2, "trend_enter": 0.1}),
])
def test_invalid_operator_parameters_fail_closed(node_id, parameters):
    raw = definition_with_inline(30)
    next(node for node in raw["graph"]["nodes"] if node["id"] == node_id)["parameters"].update(parameters)
    assert not inspect_definition_v2(parse_definition_v2(raw))["valid"]


def test_state_order_and_kama_parameter_relationship_are_validated():
    raw = definition_with_inline(30)
    raw["states"].reverse()
    assert not inspect_definition_v2(parse_definition_v2(raw))["valid"]
    raw = definition_with_inline(30)
    raw["graph"]["nodes"][2].update(type="filter.kama", parameters={"window": 10, "fast": 30, "slow": 2})
    assert not inspect_definition_v2(parse_definition_v2(raw))["valid"]

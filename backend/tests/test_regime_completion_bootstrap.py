"""Independent Python oracle for paired moving blocks, no production data."""

import numpy as np
import pytest
from historical_regimes.reliability import kernels
from historical_regimes.reliability.bootstrap import bootstrap_kernel, build_intervals
from historical_regimes.reliability.contracts import BootstrapPolicy, Policy
from historical_regimes.reliability.diagnostic_kernels import evidence_kernel


@pytest.fixture(autouse=True)
def warm():
    kernels.warm()


def fixture(n=180):
    y = (np.arange(n) // 10 % 3).astype(np.int64)
    pred = y.copy()
    pred[::7] = (pred[::7] + 1) % 3
    q = np.full((n, 3), 0.1)
    q[np.arange(n), pred] = 0.8
    return y, pred, q, np.full(3, 1 / 3)


def oracle(y, pred, q, base, floor, indices):
    y, pred, q = y[indices], pred[indices], q[indices]
    known = (y >= 0) & (y < len(base))
    valid = known & (pred >= 0) & (pred < len(base)) & np.isfinite(q).all(axis=1)
    valid &= (
        (q >= 0).all(axis=1)
        & (q <= 1).all(axis=1)
        & np.isclose(q.sum(axis=1), 1, atol=1e-6, rtol=0)
    )
    accepted = np.zeros(len(y), bool)
    accepted[valid] = q[np.flatnonzero(valid), pred[valid]] >= floor
    accuracy = np.mean(pred[known] == y[known]) if known.any() else np.nan
    coverage = accepted.sum() / known.sum() if known.any() else np.nan
    error = np.mean(pred[accepted] != y[accepted]) if accepted.any() else np.nan
    if valid.any():
        targets = np.eye(len(base))[y[valid]]
        brier = np.sum((q[valid] - targets) ** 2, axis=1)
        improvement = np.sum((base - targets) ** 2, axis=1) - brier
        return np.array([accuracy, coverage, error, brier.mean(), improvement.mean()])
    return np.array([accuracy, coverage, error, np.nan, np.nan])


def test_bootstrap_independent_oracle_missing_positions_pairing_and_percentiles():
    y, pred, q, base = fixture()
    y[23] = -1
    pred[31] = -1
    q[41] = np.nan
    q[50] = [np.inf, 0, 0]
    q[61] = [0, 0, 0]
    result, blocks, cycles = bootstrap_kernel(y, pred, q, base, 0.6, 10, 100, 0.95)
    expected = []
    rng = 1729
    starts = []
    for _ in range(100):
        indices = []
        while len(indices) < len(y):
            rng = rng * 48271 % 2147483647
            start = rng % (len(y) - 10 + 1)
            starts.append(start)
            indices.extend(range(start, start + min(10, len(y) - len(indices))))
        expected.append(oracle(y, pred, q, base, 0.6, np.array(indices)))
    expected = np.array(expected)
    np.testing.assert_allclose(
        result[:, 0], oracle(y, pred, q, base, 0.6, np.arange(len(y)))
    )
    np.testing.assert_allclose(
        result[:, 1:3], np.quantile(expected, [0.025, 0.975], axis=0).T, atol=1e-12
    )
    assert blocks[0] == 17
    assert blocks[3] < blocks[0]
    assert cycles > 0
    # A compressed axis gives a different experiment and must not be used.
    compressed = y >= 0
    other = bootstrap_kernel(
        y[compressed], pred[compressed], q[compressed], base, 0.6, 10, 100, 0.95
    )[0]
    assert not np.allclose(result[:, 1:3], other[:, 1:3])


def test_insufficient_blocks_cycles_and_invalid_replicates_are_null():
    y, pred, q, base = fixture()
    policy = BootstrapPolicy(
        replicates=100, minimum_valid_replicates=90, minimum_cycles=2
    )
    out = build_intervals(y, pred, q, base, policy, 0.6, "test")
    assert out["status"] == "available"
    assert out["conditional_on"] == "fixed_model_reference_and_calibrator"
    y[::9] = -1  # plenty of known days, zero whole 10-observation blocks
    out = build_intervals(y, pred, q, base, policy, 0.6, "holdout")
    assert out["full_blocks"] == 0
    assert out["status"] == "unavailable"
    assert all(m["lower"] is None and m["reason"] for m in out["metrics"].values())
    y, pred, q, base = fixture()
    out = build_intervals(np.zeros_like(y), pred, q, base, policy, 0.6, "test")
    assert out["complete_cycles"] == 0
    assert out["metrics"]["accuracy"]["reason"] == "insufficient_complete_cycles"
    q[:] = np.nan
    out = build_intervals(y, pred, q, base, policy, 0.6, "test")
    assert out["status"] == "partial"
    assert out["metrics"]["accepted_error"]["valid_replicates"] == 0
    assert out["metrics"]["brier"]["estimate"] is None
    assert out["metrics"]["brier"]["lower"] is None


def test_readonly_strides_sharing_determinism_and_budgets():
    y, pred, q, base = fixture(360)
    snapshots = [a.copy() for a in (y, pred, q, base)]
    view_y, view_p, view_q = y[::2], pred[::2], q[::2]
    for a in (view_y, view_p, view_q, base):
        a.flags.writeable = False
    before = kernels.audit()
    result = bootstrap_kernel(view_y, view_p, view_q, base, 0.6, 10, 100, 0.95)
    again = bootstrap_kernel(view_y, view_p, view_q, base, 0.6, 10, 100, 0.95)
    for a, b in zip(result[:2], again[:2]):
        np.testing.assert_array_equal(a, b)
    assert np.shares_memory(view_q, q)
    for a, b in zip((y, pred, q, base), snapshots):
        np.testing.assert_array_equal(a, b)
    assert kernels.audit() == before
    for n, block, reps in ((20001, 10, 100), (10, 1, 100), (10, 10, 501)):
        a, b, c, d = fixture(n)
        with pytest.raises(ValueError):
            bootstrap_kernel(a, b, c, d, 0.6, block, reps, 0.95)
    empty = bootstrap_kernel(y[:0], pred[:0], q[:0], base, 0.6, 10, 100, 0.95)[0]
    assert np.isnan(empty[:, :3]).all()
    assert (empty[:, 3:] == 0).all()
    with pytest.raises(ValueError):
        BootstrapPolicy(replicates=20, minimum_valid_replicates=100)


def test_verified_evidence_top_margin_entropy_oracle():
    q = np.array(
        [[0.2, 0.3, 0.5], [0.5, 0.5, 0.0], [np.nan, 0.0, 1.0], [0.2, 0.2, 0.2]]
    )
    pred = np.array([1, 0, 0, 1], np.int64)
    out = evidence_kernel(pred, q)
    np.testing.assert_allclose(
        out[0, :5], [0.3, 0.5, 0.3, 0.2, -np.sum(q[0] * np.log(q[0]))]
    )
    assert out[1, 3] == 0
    assert np.isnan(out[2:]).all()


def test_only_final_test_passed_to_estimator(monkeypatch):
    import copy
    from datetime import date, timedelta
    from historical_regimes.reliability.report import build_report
    from historical_regimes.v2_contracts import parse_definition_v2
    from test_historical_regime_v2 import _definition
    import historical_regimes.reliability.bootstrap as module

    payload = _definition()
    payload["study"] = {"purpose": "realtime_recognition", "family": "market_trend"}
    definition = parse_definition_v2(payload)
    states = [s.id for s in definition.states]
    points = [
        {
            "observation_date": (date(2020, 1, 1) + timedelta(days=i)).isoformat(),
            "data_available_at": (date(2020, 1, 1) + timedelta(days=i)).isoformat(),
            "recognized_at": (date(2020, 1, 1) + timedelta(days=i)).isoformat(),
            "state_id": states[i // 10 % 3],
        }
        for i in range(180)
    ]
    reference = {
        "series": points,
        "states": [s.model_dump() for s in definition.states],
        "content_hash": "a" * 64,
    }
    policy = Policy(
        calibration_end="2020-02-29",
        validation_end="2020-04-29",
        test_end="2020-06-28",
        minimum_samples=2,
        minimum_class_samples=1,
        minimum_segments=1,
    )
    original = module.build_intervals
    seen = []

    def capture(y, pred, q, base, settings, floor, scope):
        assert len(y) == 60 and scope == "test"
        assert y.base is not None and q.base is not None
        seen.append((y.copy(), q.copy()))
        return original(y, pred, q, base, settings, floor, scope)

    monkeypatch.setattr(module, "build_intervals", capture)
    lineage = {
        "probability_provenance": {
            "temperature_supported": False,
            "type": "deterministic_state",
        },
        "temporal_audit": {},
    }
    report = build_report(
        definition,
        reference,
        {"published_at": "2021-01-01"},
        copy.deepcopy(points),
        lineage,
        policy,
        "2020-06-28",
    )
    assert len(seen) == 1
    assert report["confidence_interval"]["samples"] == 60
    assert report["calibration"]["deployment_eligible"] is False


def test_kernel_memory_budget_is_linear_in_time_not_replicates():
    import tracemalloc

    y, pred, q, base = fixture(20000)
    for a in (y, pred, q, base):
        a.flags.writeable = False
    tracemalloc.start()
    bootstrap_kernel(y, pred, q, base, 0.6, 10, 500, 0.95)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    # 20001 x 10 float prefix + bounded replicate/quantile buffers. Input q is reused.
    assert peak < 3_000_000


def test_accepted_error_blocks_require_every_observation_to_be_accepted():
    y, pred, q, base = fixture(180)
    q[::10] = 1 / 3  # labels are complete, but each block contains one abstention
    result, blocks, _ = bootstrap_kernel(y, pred, q, base, .6, 10, 100, .95)
    assert blocks.tolist() == [18, 18, 0, 18, 18]
    assert result[2, 4] == 162
    out = build_intervals(y, pred, q, base, BootstrapPolicy(replicates=100), .6, 'holdout')
    assert out['metrics']['accepted_error']['reason'] == 'insufficient_full_blocks'
    assert out['metrics']['accepted_error']['lower'] is None
    assert out['metrics']['accuracy']['full_blocks'] == 18

"""Constrained segmentation against independent complete path enumeration."""
import itertools
import os

import numpy as np
import pytest

from backend import frontier_moments as fm
from backend.strategic_allocation import risk_scale_kernels as rk


@pytest.fixture(scope="module", autouse=True)
def warmed():
    fm.warm()
    rk.warm()


def geometry(n=31, line=False):
    x = np.linspace(.01, .25, n)
    y = x if line else np.sqrt(x)
    metrics = np.column_stack((x, y))
    indices = np.arange(n, dtype=np.int64)
    return metrics, rk.frontier_geometry_kernel(metrics, indices, 1e-10)


def brute_force(x, y, angles, lengths, mode, min_edges, min_span, tolerance):
    paths = []
    arc = np.r_[0., np.cumsum(lengths)]
    for cuts in itertools.combinations(range(1, len(x)-1), 4):
        nodes = (0, *cuts, len(x)-1)
        if any(r-l < min_edges or x[r]-x[l]+1e-14 < min_span for l, r in zip(nodes[:-1], nodes[1:])):
            continue
        if mode == 0:
            cost = sum(np.sum(lengths[l:r]*(angles[l:r]-np.average(angles[l:r], weights=lengths[l:r]))**2)
                       for l, r in zip(nodes[:-1], nodes[1:]))
        else:
            coordinate = x if mode == 1 else arc/arc[-1] if mode == 2 else y
            cost = sum((coordinate[j]-(k+1)/5)**2 for k, j in enumerate(cuts))
        paths.append((cost, cuts))
    optimum = min(c for c, _ in paths)
    return min(p for c, p in paths if c <= optimum+tolerance), optimum


@pytest.mark.parametrize("mode", [0, 1, 2, 3])
@pytest.mark.parametrize("tie_tolerance", [1e-12, .003, .03])
def test_dp_matches_brute_force_global_lexicographic_path(mode, tie_tolerance):
    _, (x, y, angles, lengths, prefix, status) = geometry(14)
    assert status == 0
    cuts, status, actual, optimum, fallback = rk.segment_frontier_kernel(x, y, angles, prefix, mode, 2, .05, tie_tolerance, 0.)
    expected, reference_optimum = brute_force(x, y, angles, lengths, mode, 2, .05, tie_tolerance)
    assert status == 0 and fallback == 0
    assert tuple(cuts) == expected
    assert optimum == pytest.approx(reference_optimum, abs=1e-14)
    assert actual <= optimum+tie_tolerance


def test_tie_is_full_path_not_last_predecessor():
    _, (x, y, angles, lengths, prefix, _) = geometry(16, line=True)
    cuts, status, cost, optimum, fallback = rk.segment_frontier_kernel(x, y, angles, prefix, 0, 2, .05, 1e-12, 0.)
    assert status == 0 and cuts.tolist() == [2, 4, 6, 8] and fallback == 0
    assert cost <= optimum+1e-12


def test_near_linearity_is_recorded_only_for_shape_algorithm():
    metrics, geo = geometry(101, line=True)
    status = np.zeros(101, dtype=np.int64)
    shape = rk.segment_frontier(metrics, status)
    arc = rk.segment_frontier(metrics, status, "equal_arclength_v1")
    vol = rk.segment_frontier(metrics, status, "equal_volatility_v1")
    ret = rk.segment_frontier(metrics, status, "equal_return_v1")
    assert shape["status"] == arc["status"] == vol["status"] == ret["status"] == 0
    assert shape["fallback_reason"] == "near_linear_frontier"
    assert shape["algorithm_id"] == "frontier_shape_dp_v2"
    assert arc["fallback_reason"] is None and vol["fallback_reason"] is None and ret["fallback_reason"] is None
    np.testing.assert_array_equal(shape["cut_node_indices"], arc["cut_node_indices"])
    np.testing.assert_array_equal(arc["cut_node_indices"], [20, 40, 60, 80])
    assert np.ptp(geo[2]) < .02


def test_angle_threshold_strict_and_bounded():
    _, (x, y, angles, _, prefix, _) = geometry(31)
    threshold = float(np.ptp(angles))
    assert rk.segment_frontier_kernel(x, y, angles, prefix, 0, 5, .05, 1e-12, threshold)[4] == 0
    assert rk.segment_frontier_kernel(x, y, angles, prefix, 0, 5, .05, 1e-12, threshold+1e-10)[4] == 1
    assert np.all(angles >= 0) and np.all(angles <= np.pi/2)


@pytest.mark.parametrize("n", [0, 1, 2, 24, 25])
def test_sparse_and_degenerate(n):
    metrics, _ = geometry(n)
    r = rk.segment_frontier(metrics, np.zeros(n, dtype=np.int64))
    assert r["status"] in (2, 3)
    assert np.all(r["representative_node_indices"] == -1)


def test_zero_risk_or_return_span_and_duplicate_risk():
    for metrics in [np.array([[0., .01], [0., .02]]), np.array([[.1, .02], [.2, .02]])]:
        assert rk.frontier_geometry_kernel(metrics, np.array([0, 1]), 1e-10)[5] == 2
    metrics = np.array([[.1, .02], [.1, .03], [.2, .04]])
    assert rk.frontier_geometry_kernel(metrics, np.arange(3), 1e-10)[5] == 1


def test_no_segmentation_across_failed_grid_point():
    metrics, _ = geometry(101)
    status = np.zeros(101, dtype=np.int64); status[50] = 1
    r = rk.segment_frontier(metrics, status)
    assert r["status"] == 4 and np.all(r["cut_node_indices"] == -1)
    manual = rk.segment_frontier(metrics, status, "manual_volatility_bands_v1", manual_caps=np.arange(1., 6.)/10)
    assert manual["status"] == 0 and np.all(manual["representative_node_indices"] == -1)


def test_manual_zero_missing_representatives_and_above_scale():
    metrics = np.array([[0., .01], [.15, .06], [.2, .08]])
    caps = np.array([0., .1, .2, .3, .4])
    r = rk.segment_frontier(metrics, np.zeros(3, dtype=np.int64), "manual_volatility_bands_v1", manual_caps=caps)
    assert r["status"] == 0
    assert r["representative_node_indices"].tolist() == [0, -1, 1, -1, -1]
    assert r["not_calibrated"].tolist() == [0, 0, 0, 1, 1]
    assert rk.classify_risk_kernel(.5, caps, 0., 1e-10)[0] == 6
    assert rk.classify_risk_kernel(0., caps, 0., 1e-10)[0] == 1


@pytest.mark.parametrize("caps", [[0., 0., .1, .2, .3], [-.1, .1, .2, .3, .4], [.1, .3, .2, .4, .5], [0., .1, .2, .3, np.nan]])
def test_manual_caps_are_not_repaired(caps):
    with pytest.raises(ValueError, match="CAPS_STRICT_ORDER"):
        rk.manual_volatility_bands_kernel(np.array([0., .1]), np.array(caps), 1e-10)


def test_boundary_and_authorized_cap_checks_agree():
    caps = np.arange(1., 6.)/10
    for i in range(5):
        for offset in [-2e-10, -5e-11, 0., 5e-11, 2e-10]:
            risk = caps[i]+offset
            level, flags = rk.classify_risk_kernel(risk, caps, .05, 1e-10)
            for band in range(1, 6):
                assert (1 <= level <= band) == (rk.risk_within_cap_kernel(risk, caps[band-1], 1e-10) == 1)
            if offset == 0:
                assert level == i+1 and flags & 2
    assert rk.classify_risk_kernel(.01, caps, .05, 1e-10) == (1, 1)
    for invalid in [np.nan, np.inf, -np.inf, -.01, -1e-14]:
        assert rk.classify_risk_kernel(invalid, caps, .05, 1e-10) == (0, 0)
        assert rk.risk_within_cap_kernel(invalid, caps[0], 1e-10) == -1


def test_representatives_do_not_reuse_lower_boundary_node():
    risks = np.linspace(0, 1., 101)
    caps = np.arange(1., 6.)/5
    representatives = rk.representative_nodes_kernel(risks, risks, caps, 1e-10)
    for i, j in enumerate(representatives):
        assert rk.classify_risk_kernel(risks[j], caps, 0., 1e-10)[0] == i+1
    assert representatives[1] > 20


@pytest.mark.parametrize("returns,alpha,es,var,q", [
    ([-.1, -.1, -.02, .01, .03], .5, .084, .02, 2.5),
    ([-.1, -.02, .03, .05], .5, .06, -.03, 2.),
    ([-.1, -.02, .03], .95, .1, .1, .15),
    ([.1, .2, .3], .5, -.4/3, -.2, 1.5),
])
def test_fractional_tail_ties_sign_and_thin_tail(returns, alpha, es, var, q):
    data = np.array(returns)
    original = data.copy(); data.flags.writeable = False
    actual_var, actual_es, mass, status = rk.historical_tail_kernel(data, alpha, 5.)
    assert actual_es == pytest.approx(es)
    assert actual_var == pytest.approx(var)
    assert mass == pytest.approx(q) and status == 1
    np.testing.assert_array_equal(data, original)


def test_drawdown_includes_initial_peak_and_no_external_cashflow():
    assert rk.historical_drawdown_kernel(np.array([-.1, .05])) == pytest.approx((.1, 0))
    assert rk.historical_drawdown_kernel(np.array([-.1, -.2, .5])) == pytest.approx((.28, 0))
    assert rk.historical_drawdown_kernel(np.array([-1., .5])) == (1., 0)
    for bad in [[], [np.nan], [np.inf], [-1.1]]:
        assert rk.historical_drawdown_kernel(np.array(bad, dtype=np.float64))[1] == 2
        assert rk.historical_tail_kernel(np.array(bad, dtype=np.float64), .95, 5.)[-1] == 2


def test_interval_roundoff_is_not_an_arbitrary_negative_cost_clamp():
    prefix = np.array([[0., 1.], [0., 1.], [0., 1.-1e-15]])
    assert rk.interval_shape_cost_kernel(prefix, 0, 1) == 0.
    prefix[2, 1] = .9
    with pytest.raises(ValueError, match="NEGATIVE_COST"):
        rk.interval_shape_cost_kernel(prefix, 0, 1)
    with pytest.raises(ValueError, match="AXIS"):
        rk.interval_shape_cost_kernel(prefix, -1, 1)


def test_101_200_stability_uses_frozen_common_endpoints():
    args = (np.array([.02, .08]), np.diag([0., .04]), np.tile([0., 1.], (2, 1)), np.zeros((0, 2)), np.empty(0), np.empty(0))
    r1, r2 = (fm.solve_frontier(*args, point_count=n) for n in [101, 200])
    s1, s2 = (rk.segment_frontier(r[2], r[3]) for r in [r1, r2])
    e1 = r1[7][[1, 3]].reshape(-1); e2 = r2[7][[1, 3]].reshape(-1)
    delta, status = rk.boundary_stability_kernel(s1["risk_caps"], s2["risk_caps"], e1, e2, 1e-8, .02)
    assert status == 0 and delta <= .02
    e2[0] += .1
    assert rk.boundary_stability_kernel(s1["risk_caps"], s2["risk_caps"], e1, e2, 1e-8, .02)[1] == 2


def test_readonly_noncontiguous_inputs_and_exact_fixed_abi(monkeypatch):
    metrics, _ = geometry(61)
    metrics = metrics[::2]; metrics.flags.writeable = False
    status = np.zeros(61, dtype=np.int64)[::2]; status.flags.writeable = False
    before = metrics.copy()
    signatures = [tuple(k.signatures) for k in rk.KERNELS]
    actual = rk.segment_frontier_kernel; calls = []
    def invoked(*args):
        calls.append(True)
        return actual(*args)
    monkeypatch.setattr(rk, "segment_frontier_kernel", invoked)
    assert rk.segment_frontier(metrics, status)["status"] == 0
    assert calls == [True]
    np.testing.assert_array_equal(metrics, before)
    assert signatures == [tuple(k.signatures) for k in rk.KERNELS]
    assert all(len(k.nopython_signatures) == 1 and not k._can_compile for k in rk.KERNELS)
    audit = rk.execution_audit()
    assert audit["complete"] and audit["python_fallback"] == audit["request_time_compilation"] == 0


def test_wrong_pid_and_invalid_index_fail_closed(monkeypatch):
    metrics, _ = geometry(31)
    for indices in [np.array([-1, 1]), np.array([0, 31]), np.array([1, 0])]:
        with pytest.raises(ValueError, match="INDICES"):
            rk.frontier_geometry_kernel(metrics, indices, 1e-10)
    monkeypatch.setattr(rk, "_WARMED_PID", os.getpid()+1)
    with pytest.raises(RuntimeError, match="NOT_READY"):
        rk.segment_frontier(metrics, np.zeros(31, dtype=np.int64))


def test_minimum_risk_span_is_not_only_an_edge_count_constraint():
    x = np.r_[np.linspace(0., .01, 25), 1.]
    metrics = np.column_stack((x, np.sqrt(x)))
    geo = rk.frontier_geometry_kernel(metrics, np.arange(26), 1e-10)
    for mode in [0, 1, 2, 3]:
        r = rk.segment_frontier_kernel(geo[0], geo[1], geo[2], geo[4], mode, 5, .05, 1e-12, .02)
        assert r[1] == 3


def test_representative_geometry_and_thresholds_do_not_copy_weights():
    metrics, _ = geometry(101)
    statuses = np.zeros(101, dtype=np.int64)
    a = rk.segment_frontier(metrics, statuses, "equal_volatility_v1")
    b = rk.segment_frontier(metrics, statuses, "equal_arclength_v1")
    assert a["status"] == b["status"] == 0
    assert a["cut_node_indices"].tolist() != b["cut_node_indices"].tolist()
    assert all(key not in a for key in ["weights", "covariance", "historical_returns"])


def test_integral_tail_mass_and_empirical_quantile_binary_roundoff():
    # .14*50 is 7.000000000000001 in binary floating point.
    returns = -np.arange(50, dtype=np.float64)/100
    var, es, q, status = rk.historical_tail_kernel(returns, .14, 5.)
    assert var == pytest.approx(.06)
    assert q == 43 and status == 0
    assert es == pytest.approx(np.arange(7, 50).mean()/100)


def test_manual_reference_with_one_real_node_retains_its_representative():
    result = rk.segment_frontier(np.array([[.15, .04]]), np.array([0]),
                                 "manual_volatility_bands_v1", manual_caps=np.arange(1., 6.)/10)
    assert result["status"] == 0
    assert result["representative_node_indices"].tolist() == [-1, 0, -1, -1, -1]


def test_segmentation_readiness_includes_frontier_dependency(monkeypatch):
    assert rk.execution_audit()["complete"]
    monkeypatch.setattr(fm, "_WARMED_PID", os.getpid()+1)
    assert not rk.execution_audit()["complete"]
    with pytest.raises(RuntimeError, match="NOT_READY"):
        rk.require_ready()

"""M1 numeric checks, offline analytic and independent SciPy test references."""
import os

import numpy as np
import pytest
from scipy.optimize import minimize

from backend import frontier_moments as fm
from backend import qp_numba as qp


@pytest.fixture(scope="module", autouse=True)
def warmed():
    fm.warm()


def inputs(n=2):
    return (np.linspace(.03, .09, n), np.diag(np.linspace(.01, .04, n)),
            np.tile([0., 1.], (n, 1)), np.zeros((0, n)), np.empty(0), np.empty(0))


def solve(*args, count=101, iterations=1000):
    return fm.solve_frontier(*(args or inputs()), point_count=count, max_iterations=iterations)


@pytest.mark.parametrize("count", [101, 200])
def test_two_asset_analytic(count):
    r = solve(count=count)
    assert r[11] == 0
    assert np.all(r[3] == 0), (np.unique(r[3], return_counts=True), r[8])
    np.testing.assert_allclose(r[0], np.linspace(.042, .09, count), atol=1e-10)
    w = (r[0]-.03)/.06
    np.testing.assert_allclose(r[1][:, 1], w, atol=1e-8)
    np.testing.assert_allclose(r[2][:, 0], np.sqrt(.01*(1-w)**2+.04*w*w), atol=1e-9)
    np.testing.assert_allclose(r[2][:, 1], r[0], atol=1e-10)
    assert np.isfinite(r[5]).all() and np.max(r[5][:, 0]) <= fm.PRIMAL_TOL
    assert fm.frontier_curve_indices_kernel(r[2], r[3], 1e-10)[1] == 0


@pytest.mark.parametrize("covariance,means,expected", [
    (np.diag([0., .04]), np.array([.02, .08]), [1., 0.]),
    (np.diag([0., 0., .04]), np.array([.01, .03, .08]), [0., 1., 0.]),
    (np.array([[.01, .01, 0.], [.01, .01, 0.], [0., 0., .04]]), np.array([.02, .04, .1]), [0., .8, .2]),
])
def test_psd_cash_and_return_maximal_gmv(covariance, means, expected):
    n = len(means)
    r = solve(means, covariance, np.tile([0., 1.], (n, 1)), np.zeros((0, n)), np.empty(0), np.empty(0))
    assert np.all(r[8] == 0), (r[8], r[10])
    assert np.all(r[3] == 0), np.unique(r[3], return_counts=True)
    np.testing.assert_allclose(r[6][1], expected, atol=2e-7)
    assert r[7][1, 0] == pytest.approx(r[7][0, 0], abs=1e-9)
    assert r[7][1, 1] >= r[7][0, 1]-1e-10


def test_minimum_variance_maximum_return_face():
    r = solve(np.array([.02, .1, .1]), np.diag([0., .01, .04]), np.tile([0., 1.], (3, 1)),
              np.zeros((0, 3)), np.empty(0), np.empty(0))
    assert np.all(r[8] == 0), r[8]
    np.testing.assert_allclose(r[6][3], [0., .8, .2], atol=1e-8)
    assert r[7][3, 0] < r[7][2, 0]


@pytest.mark.parametrize("n", [3, 5, 10, 30])
def test_random_matrix_frontier_matches_independent_qp(n):
    rng = np.random.default_rng(49+n)
    matrix = rng.normal(size=(n, n))
    sigma = matrix@matrix.T*.002 + np.eye(n)*.001
    mu = np.linspace(.015, .15, n)
    bounds = np.tile([0., 1.], (n, 1))
    groups = np.zeros((2, n)); groups[:, :2] = 1.
    lows, highs = np.array([.1, .1]), np.array([.6, .6])
    r = solve(mu, sigma, bounds, groups, lows, highs, count=31)
    assert r[11] == 0 and np.all(r[8] == 0), (r[11], r[8], r[10])
    assert np.all(r[3] == 0), np.unique(r[3], return_counts=True)
    for i in [0, 3, 15, 29, 30]:
        constraints = [{"type": "eq", "fun": lambda w: w.sum()-1},
                       {"type": "ineq", "fun": lambda w: w@mu-r[0][i]},
                       {"type": "ineq", "fun": lambda w: w[:2].sum()-.1},
                       {"type": "ineq", "fun": lambda w: .6-w[:2].sum()}]
        ref = minimize(lambda w: w@sigma@w, np.ones(n)/n, jac=lambda w: 2*sigma@w,
                       bounds=bounds, constraints=constraints, method="SLSQP",
                       options={"ftol": 1e-12, "maxiter": 1000})
        assert ref.success, ref.message
        assert r[1][i]@sigma@r[1][i] == pytest.approx(ref.fun, abs=2e-8)


def test_structure_infeasible_and_iteration_limit_are_distinct():
    mu, sigma, bounds, _, _, _ = inputs()
    groups = np.array([[1., 0.], [1., 0.]])
    r = solve(mu, sigma, bounds, groups, np.array([.8, .1]), np.array([.9, .2]))
    assert r[11] == 4 and np.all(r[3] == 6)
    exhausted = solve(iterations=1)
    assert exhausted[11] == 0
    assert 1 in exhausted[8] and 4 not in exhausted[8]
    assert np.all(exhausted[3] == 6) and np.isnan(exhausted[0]).all()


def test_phase_one_overlapping_infeasibility_is_not_false_certificate():
    # Every pair individually attainable, but all three pairs >= .8 is impossible.
    g = np.array([[1., 1., 0.], [0., 1., 1.], [1., 0., 1.]])
    args = (*inputs(3)[:3], g, np.full(3, .8), np.ones(3))
    r = solve(*args)
    assert r[11] == 5, (r[11], r[13])
    assert r[13][-1] > 0 and np.isfinite(r[13]).all()
    assert solve(*args, iterations=1)[11] == 1


def test_phase_one_finds_feasible_seed_under_joint_groups():
    mu, sigma, bounds, _, _, _ = inputs(3)
    g = np.array([[1., 1., 0.], [0., 1., 1.]])
    r = solve(mu, sigma, bounds, g, np.array([.8, .5]), np.array([.9, .6]))
    assert r[11] == 0 and np.all(r[3] == 0), (r[11], r[8], r[3])
    assert r[12] > 0
    assert np.all(r[1]@g.T >= np.array([.8, .5])-1e-7)
    assert np.all(r[1]@g.T <= np.array([.9, .6])+1e-7)


def test_multiple_equalities_rank_and_original_scale():
    e = np.array([[1., 1., 1.], [2., 2., 2.], [0., 0., 10.]])
    v = np.array([1., 2., 2.])
    g = np.eye(3); h = np.zeros(3)
    r = qp.matrix_qp_kernel(np.eye(3), np.zeros(3), e, v, g, h, np.array([.4, .4, .2]), 1000, 1e-9)
    assert r[1] == 0
    np.testing.assert_allclose(r[0], [.4, .4, .2], atol=1e-10)
    assert np.max(r[3]) < 1e-8
    v[1] = 2.1
    assert qp.matrix_qp_kernel(np.eye(3), np.zeros(3), e, v, g, h, np.ones(3)/3, 1000, 1e-9)[1] == 4


@pytest.mark.parametrize("h,f,x,it,status,used,residual,expected", [
    ([2., 4.], [0., 0.], [.5, .5], 1000, 0, 2, 2.2222224060897133e-11, [2/3, 1/3]),
    ([0., 0.], [-.02, -.08], [.5, .5], 1000, 0, 2, 0., [0., 1.]),
    ([2., 4.], [0., 0.], [.5, .5], 1, 1, 1, .16666666666111113, [2/3, 1/3]),
    ([1., 1.], [0., 0.], [2., -1.], 1000, 2, 0, np.inf, [2., -1.]),
    ([0., 0.], [0., 0.], [.5, .5], 1000, 0, 1, 0., [.5, .5]),
])
def test_old_abi_frozen_before_extension(h, f, x, it, status, used, residual, expected):
    a = np.array([[1., 1.], [1., 0.], [0., 1.], [-1., 0.], [0., -1.]])
    b = np.array([1., 0., 0., -1., -1.])
    r = qp.feasible_qp_kernel(np.diag(h), np.array(f), a, b, np.array(x), it, 1e-9)
    assert r[1:3] == (status, used)
    assert r[3] == pytest.approx(residual, abs=1e-16)
    np.testing.assert_allclose(r[0], expected, atol=1e-9)
    assert len(qp.feasible_qp_kernel.signatures) == 1
    assert len(qp.feasible_qp_kernel.signatures[0]) == 7


def test_readonly_strides_aliases_and_no_new_compilation(monkeypatch):
    args = inputs(5)
    owners, views = [], []
    for a in args:
        shape = tuple(2*s for s in a.shape)
        owner = np.zeros(shape)
        view = owner[tuple(slice(None, None, 2) for _ in a.shape)]
        view[...] = a
        view.flags.writeable = False
        owners.append(owner); views.append(view)
        assert np.shares_memory(owner, view) or a.size == 0
    before = [a.copy() for a in owners]
    signatures = [tuple(k.signatures) for k in fm.DISPATCHERS]
    actual = fm.frontier_moments_kernel
    calls = []
    def observed(*a):
        calls.append(True)
        return actual(*a)
    monkeypatch.setattr(fm, "frontier_moments_kernel", observed)
    r = solve(*views)
    assert calls == [True] and np.all(r[3] == 0)
    for a, b in zip(owners, before):
        np.testing.assert_array_equal(a, b)
    assert signatures == [tuple(k.signatures) for k in fm.DISPATCHERS]
    assert all(len(k.nopython_signatures) == 1 and not k._can_compile for k in fm.DISPATCHERS)


@pytest.mark.parametrize("which,value", [(0, np.nan), (0, np.inf), (1, np.nan), (1, np.inf), (2, np.inf)])
def test_invalid_numeric_input_fails_closed(which, value):
    args = list(inputs())
    args[which].flat[0] = value
    with pytest.raises(ValueError, match="NONFINITE"):
        solve(*args)


def test_invalid_psd_and_shapes():
    args = list(inputs())
    args[1] = np.array([[.01, .2], [.2, .04]])
    with pytest.raises(ValueError, match="NOT_PSD"):
        solve(*args)
    args[1] = np.array([[.01, .2], [.0, .04]])
    with pytest.raises(ValueError, match="ASYMMETRIC"):
        solve(*args)
    for count in (1, 201):
        with pytest.raises(ValueError, match="BUDGET"):
            solve(count=count)


def test_single_and_flat_frontiers_return_true_endpoints():
    for mu, sigma in [(np.array([.05]), np.array([[0.]])), (np.array([.05, .05]), np.zeros((2, 2)))]:
        n = len(mu)
        r = solve(mu, sigma, np.tile([0., 1.], (n, 1)), np.zeros((0, n)), np.empty(0), np.empty(0))
        assert np.all(r[8] == 0) and np.all(r[3] == 0)
        assert fm.frontier_curve_indices_kernel(r[2], r[3], 1e-10)[1] == 2


def test_gap_and_nondominated_indices_preserve_raw_evidence():
    metrics = np.array([[.1, .03], [.1, .04], [.2, .05], [.3, .06]])
    status = np.zeros(4, dtype=np.int64)
    indices, ok = fm.frontier_curve_indices_kernel(metrics, status, 1e-10)
    assert ok == 0 and indices.tolist() == [1, 2, 3]
    status[2] = 1
    indices, gap = fm.frontier_curve_indices_kernel(metrics, status, 1e-10)
    assert gap == 1 and indices.size == 0
    assert metrics[2, 1] == .05


def test_pid_bound_readiness_and_compilation_gate(monkeypatch):
    assert fm.execution_audit()["complete"]
    monkeypatch.setattr(fm, "_WARMED_PID", os.getpid()+1)
    with pytest.raises(RuntimeError, match="NOT_READY"):
        solve()


def test_nearly_dependent_equalities_do_not_certify_infeasibility():
    # This equality system HAS a (large) solution; treating numerical rank as
    # exact rank would produce a false infeasibility certificate.
    e = np.array([[1., 0.], [1., 1e-12]])
    v = np.array([0., 1.])
    r = qp.matrix_qp_kernel(np.eye(2), np.zeros(2), e, v, np.zeros((0, 2)), np.empty(0),
                            np.array([0., 1e12]), 1000, 1e-9)
    assert r[1] == 3


def test_scaled_equalities_and_zero_row_structural_proof():
    e = np.array([[1e8, 1e8], [0., 0.]])
    v = np.array([1e8, 0.])
    r = qp.matrix_qp_kernel(np.eye(2), np.zeros(2), e, v, np.eye(2), np.zeros(2),
                            np.array([.5, .5]), 1000, 1e-9)
    assert r[1] == 0 and r[3][0] <= 1e-7
    v[1] = 1.
    assert qp.matrix_qp_kernel(np.eye(2), np.zeros(2), e, v, np.eye(2), np.zeros(2),
                               np.array([.5, .5]), 1000, 1e-9)[1] == 4


def test_no_equalities_and_nonfinite_qp_inputs():
    e, v = np.zeros((0, 2)), np.empty(0)
    r = qp.matrix_qp_kernel(np.eye(2), np.array([-1., -2.]), e, v, np.eye(2), np.zeros(2),
                            np.array([.5, .5]), 1000, 1e-9)
    assert r[1] == 0
    np.testing.assert_allclose(r[0], [1., 2.], atol=1e-8)
    for bad in [np.nan, np.inf]:
        r = qp.matrix_qp_kernel(np.eye(2), np.array([bad, 0.]), e, v, np.eye(2), np.zeros(2),
                                np.array([.5, .5]), 1000, 1e-9)
        assert r[1] == 3


def test_200_repeated_group_constraints_supported():
    mu, sigma, bounds, _, _, _ = inputs()
    groups = np.tile(np.array([[1., 0.]]), (200, 1))
    r = solve(mu, sigma, bounds, groups, np.zeros(200), np.ones(200), count=200)
    assert np.all(r[8] == 0) and np.all(r[3] == 0)


def test_input_dtype_is_not_silently_specialized():
    args = list(inputs()); args[0] = args[0].astype(np.float32)
    with pytest.raises(TypeError):
        solve(*args)

"""Independent integration audit of matrix-frontier constraints and provenance.

The optional SciPy calculation is an offline test oracle, never a production
fallback. Inputs are deterministic and no application data directories are used.
"""
import numpy as np
import pytest

from backend import frontier_moments as frontier


@pytest.fixture(scope="module", autouse=True)
def ready():
    frontier.warm()


def test_right_endpoint_is_not_maximum_variance():
    means = np.array([.02, .07, .06])
    covariance = np.diag(np.array([.01, .04, .25]))
    bounds = np.tile([0., 1.], (3, 1))
    result = frontier.solve_frontier(means, covariance, bounds,
                                    np.empty((0, 3)), np.empty(0), np.empty(0))
    assert np.all(result[3] == 0)
    np.testing.assert_allclose(result[6][3], [0., 1., 0.], atol=1e-7)
    assert result[7][3, 0] == pytest.approx(.2, abs=1e-7)
    assert result[7][3, 0] < np.sqrt(np.max(np.diag(covariance)))


@pytest.mark.parametrize("case", range(12))
def test_overlapping_groups_against_independent_reference(case):
    optimize = pytest.importorskip("scipy.optimize")
    rng = np.random.default_rng(9016 + case)
    n = 5
    raw = rng.normal(size=(n, n))
    covariance = .015 * (raw @ raw.T / n + .2 * np.eye(n))
    means = rng.uniform(-.02, .16, n)
    bounds = np.column_stack((np.zeros(n), np.full(n, .6)))
    groups = np.array([[1., 1., 0., 0., 0.], [0., 1., 1., 0., 0.],
                       [0., 0., 0., 1., 1.]])
    lower = np.array([.2, .2, .1])
    upper = np.array([.7, .7, .6])
    if case % 3 == 0:
        covariance[0, :] = 0.
        covariance[:, 0] = 0.
    if case % 4 == 0:
        means[1] = means[2]
    inputs = means, covariance, bounds, groups, lower, upper
    original = [value.copy() for value in inputs]
    for value in inputs:
        value.flags.writeable = False
    signature_set = tuple(frontier.frontier_moments_kernel.signatures)
    result = frontier.solve_frontier(*inputs, point_count=31)
    assert np.all(result[3] == 0), result[3]
    for index in (0, 6, 15, 24, 30):
        target, weight = result[0][index], result[1][index]
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1,
             "jac": lambda w: np.ones(n)},
            {"type": "ineq", "fun": lambda w: groups @ w - lower,
             "jac": lambda w: groups},
            {"type": "ineq", "fun": lambda w: upper - groups @ w,
             "jac": lambda w: -groups},
            {"type": "ineq", "fun": lambda w: means @ w - target,
             "jac": lambda w: means},
        ]
        reference = optimize.minimize(
            lambda w: float(w @ covariance @ w), weight.copy(),
            jac=lambda w: 2 * covariance @ w,
            bounds=[(0., .6)] * n, constraints=constraints, method="SLSQP",
            options={"ftol": 1e-13, "maxiter": 1000},
        )
        assert reference.success, reference.message
        assert weight @ covariance @ weight - reference.fun <= 1e-8
        assert abs(weight.sum() - 1) <= 1e-7
        assert np.min(weight) >= -1e-7 and np.max(weight) <= .6 + 1e-7
        assert np.min(groups @ weight - lower) >= -1e-7
        assert np.min(upper - groups @ weight) >= -1e-7
        assert means @ weight >= target - 1e-7
    assert tuple(frontier.frontier_moments_kernel.signatures) == signature_set
    for actual, expected in zip(inputs, original):
        np.testing.assert_array_equal(actual, expected)
        assert not actual.flags.writeable

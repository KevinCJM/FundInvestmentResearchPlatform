"""Risk-budget comparison reuses one constrained finite search and its actual NJIT path."""
import numpy as np
import pytest

from backend.strategic_allocation import kernels


def inputs():
    return (
        np.array([0.06, 0.03]), np.diag([0.15 ** 2, 0.05 ** 2]), np.array([0.02, 0.005]),
        np.array([[0., 1.], [0., 1.]]), np.empty((0, 2), dtype=np.uint8), np.empty(0), np.empty(0),
        5., 1., 0., 1., np.empty(0), 1., 0., 2000, 42,
    )


def test_budget_uses_same_candidates_without_changing_existing_four():
    args = inputs()
    original = kernels.policy_candidates_kernel(*args)
    extended = kernels.policy_candidates_with_budget_kernel(*args, np.array([0.5, 0.5]))
    for before, after in zip(original[:3], extended[:3], strict=True):
        np.testing.assert_array_equal(before, after[:4])
    assert original[3] == extended[3]
    # With diagonal risks of 15%/5%, equal risk contributions require 25%/75% capital.
    np.testing.assert_allclose(extended[0][4], [.25, .75], atol=.002)
    np.testing.assert_allclose(extended[2][4], [.5, .5], atol=.004)
    assert kernels.risk_budget_error_kernel(extended[2][4], np.array([.5, .5])) < 1e-4


def test_budget_candidate_respects_same_cap_and_group_constraints():
    args = list(inputs())
    args[3] = np.array([[.6, .8], [.2, .4]])
    args[4] = np.array([[1, 0]], dtype=np.uint8)
    args[5], args[6] = np.array([.65]), np.array([.75])
    weights, metrics, _, accepted = kernels.policy_candidates_with_budget_kernel(*args, np.array([.5, .5]))
    assert accepted > 0
    assert .65 - 1e-8 <= weights[4, 0] <= .75 + 1e-8
    assert .2 - 1e-8 <= weights[4, 1] <= .4 + 1e-8
    assert np.isfinite(metrics[4]).all()
    np.testing.assert_allclose(weights.sum(axis=1), 1., atol=1e-8)


@pytest.mark.parametrize('budget', [np.array([.4, .4]), np.array([-.1, 1.1]),
                                   np.array([np.nan, .5]), np.array([np.inf, 0.]), np.array([1.])])
def test_invalid_budgets_fail_closed(budget):
    with pytest.raises(ValueError, match='POLICY_'):
        kernels.policy_candidates_with_budget_kernel(*inputs(), budget)


def test_zero_variance_cannot_fabricate_a_risk_budget_candidate():
    args = list(inputs())
    args[1] = np.zeros((2, 2))
    weights, metrics, contributions, accepted = kernels.policy_candidates_with_budget_kernel(*args, np.array([.5, .5]))
    assert accepted > 0  # Original mean/utility methods remain defined.
    assert np.isnan(metrics[4]).all()
    assert np.isnan(contributions[4]).all()
    assert weights[4].sum() == 0  # Serialization must expose unavailability, not a normalized result.


def test_signed_contributions_are_not_replaced_with_absolute_values():
    weights = np.array([.05, .95])
    covariance = np.array([[.04, -.005], [-.005, .01]])
    _, contributions = kernels.portfolio_moments_kernel(weights, np.array([.06, .03]), covariance,
                                                        np.zeros(2), 5., 1.)
    assert contributions[0] < 0
    np.testing.assert_allclose(contributions.sum(), 1.)
    target = np.array([.5, .5])
    assert kernels.risk_budget_error_kernel(contributions, target) == pytest.approx(np.sum((contributions - target) ** 2))


def test_readonly_strided_risk_inputs_and_budget_do_not_compile_or_mutate():
    kernels.warm_strategic_kernels()
    args = list(inputs())
    owner = np.array([[.0225, 7., 0., 7.], [7., 7., 7., 7.],
                      [0., 7., .0025, 7.], [7., 7., 7., 7.]])
    args[1] = owner[::2, ::2]
    args[1].flags.writeable = False
    budget_owner = np.array([.5, 77., .5, 77.])
    budget = budget_owner[::2]
    budget.flags.writeable = False
    assert np.shares_memory(args[1], owner) and np.shares_memory(budget, budget_owner)
    saved_owner, saved_budget = owner.copy(), budget_owner.copy()
    signatures = {k.__name__: tuple(k.signatures) for k in kernels.KERNELS}
    first = kernels.policy_candidates_with_budget_kernel(*args, budget)
    second = kernels.policy_candidates_with_budget_kernel(*args, budget)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(owner, saved_owner)
    np.testing.assert_array_equal(budget_owner, saved_budget)
    assert signatures == {k.__name__: tuple(k.signatures) for k in kernels.KERNELS}
    assert kernels.execution_audit()['python_fallback'] == 0
    assert kernels.execution_audit()['complete']

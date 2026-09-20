"""Monetary conservation and explicit cash limitations of product paths."""

import numpy as np
import pytest
from backend.pre_investment import path_kernels as paths


@pytest.fixture(scope="module", autouse=True)
def warm():
    paths.warm()


def run(
    *,
    months=3,
    holdings=(80.0, 20.0),
    inflows=None,
    outflows=None,
    rebalance=0,
    fee=0.0,
    rate=0.0,
    draws=None
):
    return paths.product_funding_paths_kernel(
        np.zeros((months, 200, 2)) if draws is None else draws,
        np.zeros(2),
        np.zeros((2, 2)),
        np.array(holdings),
        np.array([0.8, 0.2]),
        np.full(2, rate),
        np.full(2, rate),
        np.array([0, 1], dtype=np.int64),
        np.zeros(months) if inflows is None else inflows,
        np.zeros(months) if outflows is None else outflows,
        0.0,
        fee,
        rebalance,
    )


@pytest.mark.parametrize("months", [1, 13, 37])
def test_exact_conservation_without_returns_or_costs(months):
    metrics, terminal = run(
        months=months, inflows=np.full(months, 10.0), outflows=np.full(months, 5.0)
    )
    np.testing.assert_allclose(terminal, 100.0 + 5.0 * months)
    assert metrics[0] == 1 and metrics[3] == 0 and metrics[8] == 0


def test_total_nav_does_not_pay_cash_obligations_or_erase_past_failure():
    metrics, terminal = run(
        outflows=np.array([30.0, 0.0, 0.0]), inflows=np.array([0.0, 100.0, 0.0])
    )
    np.testing.assert_allclose(terminal, 180.0)
    assert metrics[0] == 0 and metrics[3] == 1 and metrics[7] == 10 and metrics[9] == 1


def test_monthly_rebalance_charges_each_trade_once():
    metrics, terminal = run(inflows=np.array([50.0, 0.0, 0.0]), rebalance=1, rate=0.01)
    np.testing.assert_allclose(terminal + metrics[8], 150.0)
    assert metrics[8] > 0 and metrics[3] == 0
    _, free = run(inflows=np.array([50.0, 0.0, 0.0]), rebalance=1)
    assert np.all(terminal < free)


def test_zero_principal_future_income_and_declared_wealth_fee():
    metrics, terminal = run(
        holdings=(0.0, 0.0), inflows=np.full(3, 10.0), outflows=np.full(3, 5.0)
    )
    np.testing.assert_allclose(terminal, 15.0)
    metrics, terminal = run(months=12, fee=0.01)
    np.testing.assert_allclose(terminal, 99.0)
    assert metrics[8] == pytest.approx(1.0)


def test_joint_lognormal_matches_annual_simple_moments_and_cash_axis():
    means = np.array([0.05, 0.0])
    cov = np.array([[0.04, 0.0], [0.0, 0.0]])
    drift, loading = paths.monthly_joint_lognormal_kernel(means, cov)
    log_cov = loading @ loading.T * 12
    expected = np.exp(drift * 12 + 0.5 * np.diag(log_cov)) - 1
    np.testing.assert_allclose(expected, means, atol=1e-14)
    np.testing.assert_allclose(
        np.expm1(log_cov) * np.outer(1 + means, 1 + means), cov, atol=1e-14
    )
    with pytest.raises(ValueError):
        paths.monthly_joint_lognormal_kernel(np.array([-1.0, 0.0]), cov)


def test_readonly_view_inputs_no_mutation_no_compilation_and_nonfinite_rejected():
    base = np.zeros((6, 200, 2))
    view = base[::2]
    view.flags.writeable = False
    assert np.shares_memory(base, view)
    run(draws=view)
    assert np.count_nonzero(base) == 0
    broken = base[:3].copy()
    broken[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="INPUT"):
        run(draws=broken)
    assert paths.audit()["complete"]
    assert all(
        len(k.nopython_signatures) == 1 and not k._can_compile for k in paths.KERNELS
    )

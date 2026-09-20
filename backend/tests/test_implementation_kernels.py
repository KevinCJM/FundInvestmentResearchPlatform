import numpy as np
import pytest
from backend.pre_investment import risk_kernels as risk, cost_kernels as cost


@pytest.fixture(scope="module", autouse=True)
def warm():
    risk.warm()
    cost.warm()


def test_joint_risk_retains_correlated_residuals_and_total_active_risk():
    # Two products have the same factor and perfectly correlated residual.
    beta = np.array([[1.2], [1.2]])
    residual = np.full((2, 2), 0.05**2)
    metrics, _, _, _ = risk.product_risk_kernel(
        beta,
        residual,
        np.array([[0.1**2]]),
        np.array([0.06]),
        np.array([0.5, 0.5]),
        np.array([1.0]),
        np.array([1.0]),
    )
    assert metrics[1] == pytest.approx(0.13)
    assert metrics[2] == pytest.approx(np.sqrt(0.02**2 + 0.05**2))
    assert risk.budget_kernel(
        np.array([0.5, 0.5]), np.array([0, 0]), np.array([1.0]), np.ones(2)
    )[1]
    assert metrics[1] > 0.12  # Exact class budget does not guarantee the risk cap.


def test_identity_and_axis_permutations_and_psd():
    cov = np.array([[0.04, 0.005], [0.005, 0.01]])
    w = np.array([0.6, 0.4])
    mu = np.array([0.08, 0.03])
    first = risk.product_risk_kernel(np.eye(2), np.zeros((2, 2)), cov, mu, w, w, w)[0]
    second = risk.product_risk_kernel(
        np.eye(2)[::-1], np.zeros((2, 2)), cov, mu, w[::-1], w, w
    )[0]
    np.testing.assert_allclose(first, second)
    assert first[1] == pytest.approx(np.sqrt(w @ cov @ w))
    assert first[2] == 0
    with pytest.raises(ValueError, match="PSD"):
        risk.validate_covariance_kernel(np.array([[1.0, 2.0], [2.0, 1.0]]))


def test_qp_phase_one_and_original_budgets():
    rng = np.random.default_rng(6)
    classes = rng.normal(0, 0.01, (120, 2))
    products = np.column_stack(
        [classes[:, 0], classes[:, 0] + rng.normal(0, 0.002, 120), classes[:, 1]]
    )
    target = np.array([0.6, 0.4])
    groups = np.array([0, 0, 1])
    result, status, _, diagnostics = risk.implementation_qp_kernel(
        products, classes, target, groups, np.ones(3), np.zeros(3)
    )
    assert status == 0
    assert result[0] + result[1] == pytest.approx(0.6)
    assert result[2] == pytest.approx(0.4)
    assert diagnostics[0] < 1e-7
    assert (
        risk.implementation_qp_kernel(
            products, classes, target, groups, np.array([0.1, 0.1, 1.0]), np.zeros(3)
        )[1]
        == 4
    )


def test_self_financing_each_side_not_half_turnover():
    after, trades, fees, stats = cost.self_financing_kernel(
        100000.0,
        np.array([50000.0, 50000.0]),
        np.array([0.6, 0.4]),
        np.full(2, 0.0005),
        np.full(2, 0.0005),
        np.zeros(2, dtype=np.int64),
    )
    assert stats[0] == pytest.approx(9.9990001, abs=1e-6)
    assert np.sum(after) + np.sum(fees) == pytest.approx(100000.0)
    assert np.sum(fees) == pytest.approx(np.abs(trades).sum() * 0.0005)
    np.testing.assert_allclose(after / after.sum(), [0.6, 0.4])


def test_initial_trade_asymmetric_cost_and_cash_no_fee():
    after, trades, fees, stats = cost.self_financing_kernel(
        100.0,
        np.array([0.0, 100.0]),
        np.array([0.8, 0.2]),
        np.array([0.01, 0.2]),
        np.array([0.02, 0.2]),
        np.array([0, 1]),
    )
    assert fees[0] > 0 and fees[1] == 0
    assert stats[0] == pytest.approx(0.8 / 1.008)
    assert stats[1] + stats[0] == pytest.approx(100.0)
    assert (
        cost.self_financing_kernel(
            100.0,
            np.array([80.0, 20.0]),
            np.array([0.8, 0.2]),
            np.zeros(2),
            np.zeros(2),
            np.array([0, 1]),
        )[3][0]
        < 1e-8
    )


def test_cash_dates_and_gross_net_reconciliation():
    balances, gaps = cost.cash_calendar_kernel(
        10.0, np.array([1, 2]), np.array([-20.0, 20.0])
    )
    assert (
        gaps[0] == 10 and balances[-1] == 10
    )  # Later receipt cannot erase a missed payment.
    replay = cost.net_replay_kernel(
        np.zeros((3, 2)),
        np.array([0.5, 0.5]),
        np.full(2, 0.01),
        np.full(2, 0.02),
        np.zeros(2, dtype=np.int64),
        np.ones(3, dtype=np.int64),
    )
    assert replay[-1, 0] == 1
    assert replay[-1, 1] + replay[:, 2].sum() == pytest.approx(1.0)


def test_readonly_strided_inputs_and_fixed_signatures():
    base = np.arange(120.0, dtype=np.float64).reshape(30, 4)
    view = base[::2, ::2]
    view.flags.writeable = False
    assert np.shares_memory(base, view)
    original = base.copy()
    np.testing.assert_allclose(
        risk.covariance_kernel(view, 0, 15, 1.0), np.cov(view, rowvar=False)
    )
    np.testing.assert_array_equal(base, original)
    for module in (risk, cost):
        assert module.audit()["complete"]
        for kernel in module.KERNELS:
            assert len(kernel.nopython_signatures) == 1 and not kernel._can_compile


def test_malformed_axes_and_nonfinite_risk_fail_closed():
    with pytest.raises(ValueError, match="AXIS"):
        risk.budget_kernel(np.ones(2), np.array([0]), np.ones(1), np.ones(2))
    with pytest.raises(ValueError, match="NONFINITE"):
        risk.product_risk_kernel(
            np.array([[np.nan]]),
            np.zeros((1, 1)),
            np.eye(1),
            np.ones(1),
            np.ones(1),
            np.ones(1),
            np.ones(1),
        )
    with pytest.raises(ValueError, match="QP_INPUT"):
        risk.implementation_qp_kernel(
            np.ones((10, 2)),
            np.ones((9, 1)),
            np.ones(1),
            np.array([0, 0]),
            np.ones(2),
            np.zeros(2),
        )


def test_training_rank_and_condition_reject_collinear_inputs():
    assert risk.design_diagnostics_kernel(np.eye(2)) == (2, 1.0)
    with pytest.raises(ValueError, match="ILL_CONDITIONED"):
        risk.design_diagnostics_kernel(np.array([[1.0, 1.0], [1.0, 1.0]]))
    rank, condition = risk.design_diagnostics_kernel(np.array([[1.0, 0.5], [0.5, 1.0]]))
    assert rank == 2 and condition == pytest.approx(np.sqrt(3.0))

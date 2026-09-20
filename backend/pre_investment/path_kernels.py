"""Coupled product holdings/cash path, for explicit same-month settlement research."""

import os
import numpy as np
from numba import njit, types
from backend.strategic_allocation.goal_kernels import (
    funding_payment_kernel,
    wilson_interval_kernel,
)
from backend.scenario_stress.numba_kernels import quantile_sorted_kernel
from .cost_kernels import self_financing_kernel
from .risk_kernels import validate_covariance_kernel

V = types.Array(types.float64, 1, "A", readonly=True)
M = types.Array(types.float64, 2, "A", readonly=True)
D = types.Array(types.float64, 3, "A", readonly=True)
INT_VECTOR = types.Array(types.int64, 1, "A", readonly=True)
F, N = types.float64, types.int64
_WARMED_PID = None


@njit((V, M), cache=True, nogil=True)
def monthly_joint_lognormal_kernel(means, covariance):
    """Exact annual simple-moment lognormal adapter; reject non-PSD log covariance."""
    n = means.size
    if covariance.shape != (n, n):
        raise ValueError("PRODUCT_PATH_AXIS")
    validate_covariance_kernel(covariance)
    log_cov = np.empty((n, n))
    drift = np.empty(n)
    for i in range(n):
        if not np.isfinite(means[i]) or means[i] <= -1:
            raise ValueError("PRODUCT_PATH_MEAN")
        for j in range(n):
            argument = 1.0 + covariance[i, j] / ((1.0 + means[i]) * (1.0 + means[j]))
            if argument <= 0:
                raise ValueError("PRODUCT_PATH_LOG_MOMENTS")
            log_cov[i, j] = np.log(argument) / 12.0
        drift[i] = np.log1p(means[i]) / 12.0 - 0.5 * log_cov[i, i]
    validate_covariance_kernel(log_cov)
    eigenvalues, eigenvectors = np.linalg.eigh(log_cov)
    loading = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            loading[i, j] = eigenvectors[i, j] * np.sqrt(max(0.0, eigenvalues[j]))
    return drift, loading


@njit((D, V, M, V, V, V, V, INT_VECTOR, V, V, F, F, N), cache=True, nogil=True)
def product_funding_paths_kernel(
    draws,
    drift,
    loading,
    initial_holdings,
    target,
    buy,
    sell,
    cash,
    inflows,
    outflows,
    terminal_target,
    fee,
    rebalance,
):
    months, paths, n = draws.shape
    if (
        n != target.size
        or months != inflows.size
        or months != outflows.size
        or initial_holdings.size != n
        or drift.size != n
        or loading.shape != (n, n)
        or buy.size != n
        or sell.size != n
        or cash.size != n
        or np.sum(cash) != 1
        or not 0 <= fee < 1
        or months < 1
        or paths < 1
        or rebalance not in (0, 1)
        or not np.isfinite(terminal_target)
        or terminal_target < 0
    ):
        raise ValueError("PRODUCT_FUNDING_AXIS")
    if (
        not np.all(np.isfinite(draws))
        or not np.all(np.isfinite(drift))
        or not np.all(np.isfinite(loading))
        or not np.all(np.isfinite(initial_holdings))
        or np.any(initial_holdings < 0)
        or not np.all(np.isfinite(target))
        or np.any(target < 0)
        or abs(np.sum(target) - 1.0) > 1e-8
        or not np.all(np.isfinite(inflows))
        or not np.all(np.isfinite(outflows))
        or np.any(inflows < 0)
        or np.any(outflows < 0)
        or not np.all(np.isfinite(buy))
        or not np.all(np.isfinite(sell))
        or np.any(buy < 0)
        or np.any(sell < 0)
        or np.any(buy > 0.2)
        or np.any(sell > 0.2)
        or np.any((cash != 0) & (cash != 1))
    ):
        raise ValueError("PRODUCT_FUNDING_INPUT")
    cash_index = 0
    for i in range(n):
        if cash[i]:
            cash_index = i
    terminals = np.empty(paths)
    costs = np.empty(paths)
    unpaid_values = np.empty(paths)
    success, payment_failures, earliest_gap = 0, 0, months + 1
    fee_factor = (1.0 - fee) ** (1.0 / 12.0)
    for p in range(paths):
        holdings = initial_holdings.copy()
        missed, unpaid, total_cost = False, 0.0, 0.0
        for month in range(months):
            for i in range(n):
                shock = drift[i]
                for j in range(n):
                    shock += loading[i, j] * draws[month, p, j]
                gross = holdings[i] * np.exp(shock)
                holdings[i] = gross * fee_factor
                total_cost += gross - holdings[i]
            holdings[cash_index] += inflows[month]
            if rebalance:
                proposed, _, _, stats = self_financing_kernel(
                    float(np.sum(holdings)), holdings, target, buy, sell, cash
                )
                # All non-cash legs settle at this model month-end under an
                # explicit adapter. There is no borrowing or negative cash.
                holdings = proposed
                total_cost += stats[0]
            cash_after, gap = funding_payment_kernel(
                holdings[cash_index], 1.0, 0.0, outflows[month]
            )
            holdings[cash_index] = cash_after
            if gap > 0:
                missed = True
                earliest_gap = min(earliest_gap, month + 1)
                unpaid += gap
            if not np.all(np.isfinite(holdings)) or np.any(holdings < 0):
                raise ValueError("PRODUCT_FUNDING_OVERFLOW")
        terminal = np.sum(holdings)
        terminals[p], costs[p], unpaid_values[p] = terminal, total_cost, unpaid
        success += int(not missed and terminal >= terminal_target - 1e-8)
        payment_failures += int(missed)
    lower, upper = wilson_interval_kernel(success, paths)
    ordered = np.sort(terminals)
    return (
        np.array(
            [
                success / paths,
                lower,
                upper,
                payment_failures / paths,
                quantile_sorted_kernel(ordered, 0.05),
                quantile_sorted_kernel(ordered, 0.5),
                quantile_sorted_kernel(ordered, 0.95),
                np.mean(unpaid_values),
                np.mean(costs),
                float(earliest_gap) if earliest_gap <= months else -1.0,
            ]
        ),
        terminals,
    )


KERNELS = (monthly_joint_lognormal_kernel, product_funding_paths_kernel)
for _kernel in KERNELS:
    _kernel.disable_compile()


def audit():
    return {
        "complete": _WARMED_PID == os.getpid()
        and all(
            len(k.nopython_signatures) == 1 and not k._can_compile for k in KERNELS
        ),
        "version": "product-monthly-cash-path/1.0",
        "python_fallback": 0,
        "request_time_compilation": 0,
    }


def warm():
    global _WARMED_PID
    _WARMED_PID = None
    drift, load = monthly_joint_lognormal_kernel(
        np.array([0.05, 0.0]), np.array([[0.04, 0.0], [0.0, 0.0]])
    )
    product_funding_paths_kernel(
        np.zeros((2, 4, 2)),
        drift,
        load,
        np.array([80.0, 20.0]),
        np.array([0.8, 0.2]),
        np.zeros(2),
        np.zeros(2),
        np.array([0, 1], dtype=np.int64),
        np.zeros(2),
        np.ones(2),
        0.0,
        0.0,
        1,
    )
    _WARMED_PID = os.getpid()
    return audit()

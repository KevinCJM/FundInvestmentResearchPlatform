"""Self-financing trades, historical drift and dated cash availability."""

import os
import numpy as np
from numba import njit, types

V = types.Array(types.float64, 1, "A", readonly=True)
M = types.Array(types.float64, 2, "A", readonly=True)
INT_VECTOR = types.Array(types.int64, 1, "A", readonly=True)
F, N = types.float64, types.int64
_WARMED_PID = None


@njit((V, F, F, F, F, F), cache=True, nogil=True)
def balance_kernel(
    product_values, settled, restricted, receivables, payables, confirmed
):
    calculated = np.sum(product_values) + settled + restricted + receivables - payables
    return calculated, abs(calculated - confirmed) <= max(1e-6, confirmed * 1e-10)


@njit((F, V, V, V, V, INT_VECTOR), cache=True, nogil=True)
def self_financing_kernel(wealth, holdings, target, buy, sell, cash):
    n = target.size
    if (
        holdings.size != n
        or buy.size != n
        or sell.size != n
        or cash.size != n
        or not np.isfinite(wealth)
        or wealth < 0
    ):
        raise ValueError("TRADE_AXIS")
    if abs(np.sum(target) - 1.0) > 1e-8 or np.sum(holdings) > wealth + 1e-6:
        raise ValueError("TRADE_BUDGET")
    for i in range(n):
        if (
            not np.isfinite(holdings[i])
            or holdings[i] < 0
            or not np.isfinite(target[i])
            or target[i] < 0
            or not 0 <= buy[i] <= 0.2
            or not 0 <= sell[i] <= 0.2
        ):
            raise ValueError("TRADE_INPUT")
    low, high = 0.0, wealth
    cost = 0.0
    for _ in range(100):
        cost = (low + high) * 0.5
        implied = 0.0
        for i in range(n):
            if cash[i]:
                continue
            delta = target[i] * (wealth - cost) - holdings[i]
            implied += delta * buy[i] if delta >= 0 else -delta * sell[i]
        if implied > cost:
            low = cost
        else:
            high = cost
        if high - low <= max(1e-9, wealth * 1e-13):
            break
    cost = (low + high) * 0.5
    after = np.empty(n)
    trades = np.empty(n)
    fees = np.zeros(n)
    gross = 0.0
    for i in range(n):
        after[i] = target[i] * (wealth - cost)
        trades[i] = after[i] - holdings[i]
        if not cash[i]:
            fees[i] = trades[i] * buy[i] if trades[i] >= 0 else -trades[i] * sell[i]
            gross += abs(trades[i])
    residual = abs(np.sum(fees) - cost)
    if residual > max(1e-7, wealth * 1e-10) or wealth - cost < 0:
        raise ValueError("TRADE_NOT_CONVERGED")
    return (
        after,
        trades,
        fees,
        np.array(
            [
                cost,
                wealth - cost,
                gross,
                gross / (2.0 * wealth) if wealth > 0 else 0.0,
                residual,
            ]
        ),
    )


@njit((F, INT_VECTOR, V), cache=True, nogil=True)
def cash_calendar_kernel(initial, days, signed_amounts):
    if initial < 0 or not np.isfinite(initial) or days.size != signed_amounts.size:
        raise ValueError("CASH_CALENDAR_INPUT")
    balances = np.empty(days.size)
    gaps = np.zeros(days.size)
    balance = initial
    # Inputs are ordered by date and then receipt before payment; end-of-day scope.
    for i in range(days.size):
        if not np.isfinite(signed_amounts[i]) or (i and days[i] < days[i - 1]):
            raise ValueError("CASH_CALENDAR_AXIS")
        balance += signed_amounts[i]
        balances[i] = balance
        gaps[i] = max(0.0, -balance)
    return balances, gaps


@njit((M, V, V, V, INT_VECTOR, INT_VECTOR), cache=True, nogil=True)
def net_replay_kernel(returns, target, buy, sell, cash, rebalance):
    """Initial funding and subsequent trades share the unique cost solve."""
    n = target.size
    if returns.shape[1] != n or rebalance.size != returns.shape[0]:
        raise ValueError("NET_REPLAY_AXIS")
    holdings, _, _, opening = self_financing_kernel(
        1.0, np.zeros(n), target, buy, sell, cash
    )
    gross_holdings = target.copy()
    result = np.empty((returns.shape[0] + 1, 4))
    result[0] = np.array([1.0, opening[1], opening[0], opening[2]])
    for t in range(returns.shape[0]):
        for i in range(n):
            r = returns[t, i]
            if not np.isfinite(r) or r <= -1:
                raise ValueError("NET_REPLAY_RETURN")
            holdings[i] *= 1.0 + r
            gross_holdings[i] *= 1.0 + r
        cost, traded = 0.0, 0.0
        if rebalance[t]:
            holdings, _, _, stats = self_financing_kernel(
                float(np.sum(holdings)), holdings, target, buy, sell, cash
            )
            cost, traded = stats[0], stats[2]
            gross_holdings = target * np.sum(gross_holdings)
        result[t + 1] = np.array(
            [np.sum(gross_holdings), np.sum(holdings), cost, traded]
        )
    return result


KERNELS = (
    balance_kernel,
    self_financing_kernel,
    cash_calendar_kernel,
    net_replay_kernel,
)
for _kernel in KERNELS:
    _kernel.disable_compile()


def audit():
    return {
        "version": "implementation-cost/1.0",
        "complete": _WARMED_PID == os.getpid()
        and all(
            len(k.nopython_signatures) == 1 and not k._can_compile for k in KERNELS
        ),
        "python_fallback": 0,
        "request_time_compilation": 0,
    }


def require_ready():
    if not audit()["complete"]:
        raise RuntimeError("实施成本计算尚未完成本进程预热。")


def warm():
    global _WARMED_PID
    _WARMED_PID = None
    balance_kernel(np.array([90.0]), 10.0, 0.0, 0.0, 0.0, 100.0)
    w = np.array([0.5, 0.5])
    r = np.array([0.001, 0.002])
    cash = np.zeros(2, dtype=np.int64)
    self_financing_kernel(100.0, np.zeros(2), w, r, r, cash)
    cash_calendar_kernel(10.0, np.array([1, 2], dtype=np.int64), np.array([2.0, -3.0]))
    net_replay_kernel(
        np.array([[0.01, -0.01], [0.02, 0.01]]),
        w,
        r,
        r,
        cash,
        np.ones(2, dtype=np.int64),
    )
    _WARMED_PID = os.getpid()
    require_ready()
    return audit()

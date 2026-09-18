"""Fixed-signature monthly funding calculations; inputs are read-only views."""
from __future__ import annotations

import os
import numpy as np
from numba import njit, types
from backend.scenario_stress.numba_kernels import seeded_factor_draws_kernel, quantile_sorted_kernel

V = types.Array(types.float64, 1, "A", readonly=True)
I = types.Array(types.int64, 2, "A", readonly=True)
D = types.Array(types.float64, 3, "A", readonly=True)
F = types.float64
N = types.int64
VERSION = "mandate-funding-monthly-lognormal/1.2.0"
_WARMED_PID = None


@njit((I, V, N, F, N), cache=True, nogil=True)
def funding_schedule_kernel(periods, signed_amounts, months, inflation, real_basis):
    if (months < 1 or months > 360 or periods.shape != (signed_amounts.size, 3)
            or not np.isfinite(inflation) or inflation <= -1 or real_basis not in (0, 1)):
        raise ValueError("FUNDING_SCHEDULE_AXIS")
    inflows = np.zeros(months, dtype=np.float64)
    outflows = np.zeros(months, dtype=np.float64)
    for row in range(signed_amounts.size):
        first, last, step = periods[row, 0], periods[row, 1], periods[row, 2]
        amount = signed_amounts[row]
        if first < 1 or last < first or last > months or step not in (1, 3, 12) or not np.isfinite(amount):
            raise ValueError("FUNDING_FLOW_INVALID")
        for month in range(first, last + 1, step):
            nominal = amount * ((1 + inflation) ** (month / 12.0) if real_basis else 1.0)
            if amount >= 0:
                inflows[month - 1] += nominal
            else:
                outflows[month - 1] -= nominal
    return inflows, outflows


@njit((F, F, V, V, F, F), cache=True, nogil=True)
def funding_pass_kernel(rate, initial, inflows, outflows, target, fee):
    growth = ((1.0 + rate) * (1.0 - fee)) ** (1.0 / 12.0)
    wealth = initial
    for month in range(inflows.size):
        wealth = wealth * growth + inflows[month] - outflows[month]
        if wealth < -1e-9:
            return False
    return wealth >= target


@njit((F, V, V, F, F), cache=True, nogil=True)
def required_return_kernel(initial, inflows, outflows, target, fee):
    if (inflows.size == 0 or inflows.size != outflows.size or initial <= 0 or target < 0
            or not np.isfinite(initial) or not np.isfinite(target) or not 0 <= fee < 1):
        raise ValueError("FUNDING_ROOT_INPUT")
    for i in range(inflows.size):
        if not np.isfinite(inflows[i]) or not np.isfinite(outflows[i]) or inflows[i] < 0 or outflows[i] < 0:
            raise ValueError("FUNDING_ROOT_INPUT")
    low, high = -0.99, 5.0
    if funding_pass_kernel(low, initial, inflows, outflows, target, fee):
        return low, 1
    if not funding_pass_kernel(high, initial, inflows, outflows, target, fee):
        return np.nan, 2
    for _ in range(80):
        mid = (low + high) * 0.5
        if funding_pass_kernel(mid, initial, inflows, outflows, target, fee):
            high = mid
        else:
            low = mid
    return high, 0


@njit((F, F, F, N, F, F, N, V, V, N, F), cache=True, nogil=True)
def funding_summary_kernel(total, reserve, target, months, inflation, fee, real_basis,
                           inflows, outflows, liquidity_months, contribution_ratio):
    initial = total - reserve
    if (initial <= 0 or reserve < 0 or inflows.size != months or outflows.size != months
            or liquidity_months < 1 or liquidity_months > months or not 0 <= contribution_ratio <= 1):
        raise ValueError("FUNDING_CAPITAL_INPUT")
    nominal_target = target * ((1 + inflation) ** (months / 12.0) if real_basis else 1.0)
    net_need, reserve_need = 0.0, 0.0
    contributions, withdrawals = 0.0, 0.0
    for i in range(months):
        contributions += inflows[i]
        withdrawals += outflows[i]
        if i < liquidity_months:
            net_need += outflows[i] - inflows[i] * contribution_ratio
            reserve_need = max(reserve_need, net_need)
    required, root_status = required_return_kernel(initial, inflows, outflows, nominal_target, fee)
    payment_buffer = max(initial - reserve_need, 0.0)
    return np.array([initial, nominal_target, contributions, withdrawals, reserve_need,
                     reserve_need / initial, required, payment_buffer,
                     payment_buffer / initial, max(reserve_need - initial, 0.0)]), root_status


@njit((N, N), cache=True, nogil=True)
def wilson_interval_kernel(successes, paths):
    if paths < 1 or successes < 0 or successes > paths:
        raise ValueError("FUNDING_PROBABILITY_COUNT")
    p = successes / paths
    z = 1.959963984540054
    denominator = 1.0 + z * z / paths
    middle = (p + z * z / (2 * paths)) / denominator
    half = z * np.sqrt(p * (1 - p) / paths + z * z / (4 * paths * paths)) / denominator
    return (0.0 if successes == 0 else max(0.0, middle - half),
            1.0 if successes == paths else min(1.0, middle + half))


@njit((V, F), cache=True, nogil=True)
def capital_gate_kernel(sorted_capitals, required_probability):
    """Minimum same-path capital meeting the Wilson lower-bound acceptance rule."""
    paths = sorted_capitals.size
    if paths < 1 or not np.isfinite(required_probability) or not 0 <= required_probability <= 1:
        raise ValueError("FUNDING_CAPITAL_GATE_INPUT")
    previous = -np.inf
    for value in sorted_capitals:
        if not np.isfinite(value) or value < 0 or value < previous:
            raise ValueError("FUNDING_CAPITAL_GATE_INPUT")
        previous = value
    if wilson_interval_kernel(paths, paths)[0] < required_probability:
        return np.nan, np.nan, np.nan, np.nan
    low, high = 0, paths
    while low < high:
        middle = (low + high) // 2
        if wilson_interval_kernel(middle, paths)[0] >= required_probability:
            high = middle
        else:
            low = middle + 1
    capital = sorted_capitals[max(low - 1, 0)] if low else 0.0
    # Include ties; a quantile interpolation can fall below the required rank.
    successes = 0
    for value in sorted_capitals:
        if value <= capital:
            successes += 1
    lower, upper = wilson_interval_kernel(successes, paths)
    return capital, successes / paths, lower, upper


@njit((F, F, F, F), cache=True, nogil=True)
def funding_payment_kernel(wealth, factor, contribution, payment):
    """One month of funding: growth, contribution, then required payment."""
    available = wealth * factor + contribution
    unpaid = payment - available if available < payment - 1e-8 else 0.0
    return max(0.0, available - payment), unpaid


@njit((D, F, F, F, V, V, F, F), cache=True, nogil=True)
def funding_capital_successes_kernel(draws, drift, scale, initial, inflows, outflows,
                                    target, contribution_ratio):
    """Verify capital with the same cash recurrence as the reported diagnosis."""
    successes = 0
    for path in range(draws.shape[1]):
        wealth, missed = initial, False
        for month in range(draws.shape[0]):
            factor = np.exp(drift + scale * draws[month, path, 0])
            wealth, unpaid = funding_payment_kernel(
                wealth, factor, inflows[month] * contribution_ratio, outflows[month])
            missed = missed or unpaid > 0.0
        if not np.isfinite(wealth):
            raise ValueError("FUNDING_PATH_OVERFLOW")
        successes += int(not missed and wealth >= target - 1e-8)
    return successes


@njit((F, F, F, N, N), cache=True, nogil=True)
def funding_monthly_parameters_kernel(mean, volatility, fee, method, periods):
    """Map declared moments to model-month growth; method 0 preserves the old ABI."""
    if (not np.isfinite(mean) or not np.isfinite(volatility) or volatility < 0
            or not 0 <= fee < 1 or method not in (0, 1) or periods < 1 or periods > 366):
        raise ValueError("FUNDING_DISTRIBUTION_INPUT")
    base_mean = mean / periods if method == 1 else mean
    base_volatility = volatility / np.sqrt(periods) if method == 1 else volatility
    if base_mean <= -1:
        raise ValueError("FUNDING_DISTRIBUTION_MEAN")
    log_variance = np.log1p((base_volatility / (1.0 + base_mean)) ** 2)
    frequency = periods if method == 1 else 1
    drift = (frequency * (np.log1p(base_mean) - 0.5 * log_variance) + np.log1p(-fee)) / 12.0
    scale = np.sqrt(frequency * log_variance / 12.0)
    if not np.isfinite(drift) or not np.isfinite(scale):
        raise ValueError("FUNDING_DISTRIBUTION_OVERFLOW")
    return drift, scale


@njit((D, F, F, F, V, V, F, F, F, F), cache=True, nogil=True)
def funding_paths_from_monthly_kernel(draws, drift, scale, initial, inflows,
                                      outflows, target, required_probability, drawdown_alert, contribution_ratio):
    months, paths, factors = draws.shape
    if (months < 1 or months % 12 or paths < 1 or factors != 1 or inflows.size != months
            or outflows.size != months or initial <= 0 or target < 0
            or not np.isfinite(initial) or not np.isfinite(target) or not 0 < drawdown_alert <= 1
            or not np.isfinite(drift) or not np.isfinite(scale) or scale < 0
            or not 0 <= required_probability <= 1 or not 0 <= contribution_ratio <= 1):
        raise ValueError("FUNDING_PATH_INPUT")
    for i in range(months):
        if (not np.isfinite(inflows[i]) or not np.isfinite(outflows[i])
                or inflows[i] < 0 or outflows[i] < 0):
            raise ValueError("FUNDING_PATH_INPUT")
    annual_balances = np.empty((paths, months // 12 + 1), dtype=np.float64)
    terminals = np.empty(paths, dtype=np.float64)
    drawdowns = np.empty(paths, dtype=np.float64)
    required_capitals = np.empty(paths, dtype=np.float64)
    success_count, missed_count, alert_count, extra_count, reduced_count = 0, 0, 0, 0, 0
    shortfall_sum, unpaid_sum = 0.0, 0.0
    for path in range(paths):
        wealth, growth, peak, drawdown = initial, 1.0, 1.0, 0.0
        discounted_flows, capital_floor, unpaid = 0.0, 0.0, 0.0
        missed = False
        annual_balances[path, 0] = initial
        for month in range(months):
            z = draws[month, path, 0]
            if not np.isfinite(z):
                raise ValueError("FUNDING_DRAW_NONFINITE")
            factor = np.exp(drift + scale * z)
            growth *= factor
            if not np.isfinite(growth) or growth <= 0:
                raise ValueError("FUNDING_PATH_OVERFLOW")
            peak = max(peak, growth)
            drawdown = max(drawdown, 1.0 - growth / peak)
            contribution = inflows[month] * contribution_ratio
            discounted_flows += (contribution - outflows[month]) / growth
            capital_floor = max(capital_floor, -discounted_flows)
            wealth, payment_gap = funding_payment_kernel(wealth, factor, contribution, outflows[month])
            missed = missed or payment_gap > 0.0
            unpaid += payment_gap
            if (month + 1) % 12 == 0:
                annual_balances[path, (month + 1) // 12] = wealth
        required_capital = max(capital_floor, target / growth - discounted_flows, 0.0)
        if not np.isfinite(required_capital) or not np.isfinite(wealth):
            raise ValueError("FUNDING_PATH_OVERFLOW")
        required_capitals[path] = required_capital
        terminals[path], drawdowns[path] = wealth, drawdown
        success_count += int(not missed and wealth >= target - 1e-8)
        missed_count += int(missed)
        alert_count += int(drawdown > drawdown_alert)
        extra_count += int(initial * 1.1 >= required_capital)
        reduced_count += int(not missed and wealth >= target * 0.9)
        shortfall_sum += max(0.0, target - wealth)
        unpaid_sum += unpaid
    low, high = wilson_interval_kernel(success_count, paths)
    sorted_terminal = np.sort(terminals)
    sorted_dd = np.sort(drawdowns)
    sorted_capital = np.sort(required_capitals)
    required_initial = quantile_sorted_kernel(sorted_capital, required_probability)
    gate_capital, gate_probability, gate_low, gate_high = capital_gate_kernel(sorted_capital, required_probability)
    if np.isfinite(gate_capital):
        # Match the displayed two-decimal amount, rounding upward. A rank from
        # discounted flows alone is not evidence that the cash recurrence passes.
        gate_capital = np.ceil(gate_capital * 100.0) / 100.0
        for attempt in range(8):
            verified_count = funding_capital_successes_kernel(
                draws, drift, scale, gate_capital, inflows, outflows, target, contribution_ratio)
            gate_probability = verified_count / paths
            gate_low, gate_high = wilson_interval_kernel(verified_count, paths)
            if gate_low >= required_probability:
                break
            correction = max(0.01, max(gate_capital, 1.0) * np.finfo(np.float64).eps * 64 * months)
            gate_capital = np.ceil((gate_capital + correction * (2 ** attempt)) * 100.0) / 100.0
        else:
            raise ValueError("FUNDING_CAPITAL_RECONCILIATION")
    metrics = np.array([success_count / paths, low, high, missed_count / paths,
        quantile_sorted_kernel(sorted_terminal, 0.05), quantile_sorted_kernel(sorted_terminal, 0.5),
        quantile_sorted_kernel(sorted_terminal, 0.95), shortfall_sum / paths, unpaid_sum / paths,
        quantile_sorted_kernel(sorted_dd, 0.95), alert_count / paths, required_initial,
        max(0.0, required_initial - initial), extra_count / paths, reduced_count / paths,
        gate_capital, max(0.0, gate_capital - initial) if np.isfinite(gate_capital) else np.nan,
        gate_probability, gate_low, gate_high])
    fan = np.empty((months // 12 + 1, 3), dtype=np.float64)
    for year in range(fan.shape[0]):
        sorted_values = np.sort(annual_balances[:, year])
        fan[year, 0] = quantile_sorted_kernel(sorted_values, 0.05)
        fan[year, 1] = quantile_sorted_kernel(sorted_values, 0.5)
        fan[year, 2] = quantile_sorted_kernel(sorted_values, 0.95)
    return metrics, fan


@njit((D, F, F, F, V, V, F, F, F, F, F), cache=True, nogil=True)
def funding_paths_kernel(draws, annual_mean, annual_volatility, initial, inflows,
                         outflows, target, fee, required_probability, drawdown_alert, contribution_ratio):
    """Supported annual-moment contract delegates to the unique monthly recurrence."""
    drift, scale = funding_monthly_parameters_kernel(annual_mean, annual_volatility, fee, 0, 1)
    return funding_paths_from_monthly_kernel(draws, drift, scale, initial, inflows, outflows,
                                            target, required_probability, drawdown_alert, contribution_ratio)


KERNELS = (funding_schedule_kernel, funding_pass_kernel, required_return_kernel,
           funding_summary_kernel, wilson_interval_kernel, capital_gate_kernel,
           funding_payment_kernel, funding_capital_successes_kernel,
           funding_monthly_parameters_kernel, funding_paths_from_monthly_kernel, funding_paths_kernel)
for dispatcher in KERNELS:
    dispatcher.disable_compile()


def execution_audit():
    complete = _WARMED_PID == os.getpid() and all(
        len(k.nopython_signatures) == 1 and not k._can_compile for k in KERNELS)
    return {"backend": "numba_njit_fixed_signature", "kernel_version": VERSION,
            "nopython": bool(complete), "fully_warmed": bool(complete), "complete": bool(complete),
            "object_mode": 0, "python_fallback": 0, "request_time_compilation": 0,
            "kernel_signatures": {k.__name__: [str(s) for s in k.signatures] for k in
                                  (*KERNELS, seeded_factor_draws_kernel, quantile_sorted_kernel)}}


def require_ready():
    if not execution_audit()["complete"]:
        raise RuntimeError("投资目标计算尚未完成本进程启动预热。")


def warm_goal_kernels():
    global _WARMED_PID
    _WARMED_PID = None
    inflows, outflows = funding_schedule_kernel(np.empty((0, 3), dtype=np.int64), np.empty(0), 12, 0., 0)
    funding_summary_kernel(100., 10., 120., 12, 0., 0., 0, inflows, outflows, 12, 0.5)
    draws, _ = seeded_factor_draws_kernel(12, 8, 1, 42, 0, 5.)
    funding_paths_kernel(draws, 0.06, 0.15, 90., inflows, outflows, 120., 0.01, 0.8, 0.2, 1.0)
    drift, scale = funding_monthly_parameters_kernel(.06, .15, .01, 1, 252)
    funding_paths_from_monthly_kernel(draws, drift, scale, 90., inflows, outflows, 120., .8, 1., 1.)
    _WARMED_PID = os.getpid()
    if not execution_audit()["complete"]:
        _WARMED_PID = None
        raise RuntimeError("投资目标固定签名预热失败。")
    return execution_audit()

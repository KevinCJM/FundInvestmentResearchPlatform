"""Causal timing primitives and daily execution, with fixed read-only signatures.

Inputs may be arbitrary-stride views. Only outputs/workspaces are allocated;
neither prices nor conditions are mutated. Fees are fractions of traded value,
slippage changes execution prices, and signal time is always prior-day close.
"""
from __future__ import annotations

import numpy as np
from numba import float64, int64, njit, types

from backend.compute_policy import validate_execution_audit

F = types.Array(float64, 1, "A", readonly=True)
I = types.Array(int64, 1, "A", readonly=True)
M = types.Array(float64, 2, "A", readonly=True)
FO = float64[::1]
IO = int64[::1]
MO = float64[:, ::1]

PATH_COLUMNS = ("nav", "buy_hold_nav", "daily_return", "buy_hold_daily_return",
                "position", "turnover", "cost", "action", "reason", "valid")
TRADE_COLUMNS = ("signal_index", "entry_index", "exit_index", "entry_price",
                 "exit_price", "net_return", "holding_days", "reason",
                 "max_adverse_excursion", "max_favorable_excursion")
SUMMARY_COLUMNS = ("observations", "total_return", "buy_hold_return", "excess_return",
                   "max_drawdown", "annualized_return", "annualized_volatility",
                   "exposure", "turnover", "cost", "trade_count", "win_rate",
                   "mean_trade_return", "median_trade_return", "mean_holding_days", "profit_factor")
QUALITY_COLUMNS = ("horizon", "signal_count", "eligible_count", "win_rate", "mean_return",
                   "median_return", "mean_max_adverse", "mean_max_favorable", "positive_close_fraction")
DIAGNOSTIC_COLUMNS = ("raw_signal_count", "known_condition_count", "active_month_count", "total_month_count",
                      "top_month_signal_share", "positive_month_fraction", "positive_excess_month_fraction")
STATUS_LABELS = {0: "complete", 1: "missing_market_data", 2: "invalid_market_data"}
REASON_LABELS = {0: "none", 1: "exit_rule", 2: "stop_loss", 3: "take_profit", 4: "max_holding"}
ENGINE_VERSION = "timing-njit/1.0.0"
REASON_NONE, REASON_RULE, REASON_STOP, REASON_TAKE, REASON_MAX_HOLDING = 0, 1, 2, 3, 4
MISSING_DAY = np.iinfo(np.int64).min


@njit(int64(F), cache=True, nogil=True)
def series_status_kernel(values):
    """Warm-up NaNs are allowed; infinities cannot enter a rule chain."""
    for value in values:
        if np.isinf(value):
            return 1
    return 0


@njit(int64(I, I, int64, int64), cache=True, nogil=True)
def availability_status_kernel(dates, available, start, end):
    """Reject observations known after their axis day; missing rows stay missing."""
    if dates.size != available.size or start < 0 or end < start or end > dates.size:
        raise ValueError("Invalid availability axis or interval.")
    for t in range(start, end):
        if available[t] != MISSING_DAY and available[t] > dates[t]:
            return 1
    return 0


@njit(int64(F, int64, int64), cache=True, nogil=True)
def execution_volume_status_kernel(volume, start, end):
    """No execution study over missing/nontrading sessions in this protocol."""
    if start < 0 or end < start or end > volume.size:
        raise ValueError("Invalid volume interval.")
    status = 0
    for t in range(start, end):
        if not np.isfinite(volume[t]):
            return 1
        if volume[t] <= 0:
            status = 2
    return status


@njit(types.Tuple((FO, FO, FO))(F, float64, float64), cache=True, nogil=True)
def alpha_beta_kernel(values, alpha, beta):
    """Joint level/slope recursion; a non-finite observation resets all state.

    Prediction = level + slope; innovation = observation - prediction;
    level = prediction + alpha * innovation; slope += beta * innovation.
    The first valid observation initializes level/zero slope, no innovation.
    """
    if not (np.isfinite(alpha) and np.isfinite(beta) and 0.0 < alpha <= 1.0 and 0.0 < beta <= 1.0):
        raise ValueError("alpha and beta must be finite in (0, 1].")
    level = np.full(values.size, np.nan)
    slope = np.full(values.size, np.nan)
    innovation = np.full(values.size, np.nan)
    active = False
    current_level = 0.0
    current_slope = 0.0
    for t in range(values.size):
        value = values[t]
        if not np.isfinite(value):
            active = False
            continue
        if not active:
            current_level = value
            current_slope = 0.0
            active = True
        else:
            predicted = current_level + current_slope
            error = value - predicted
            current_level = predicted + alpha * error
            current_slope += beta * error
            innovation[t] = error
        level[t] = current_level
        slope[t] = current_slope
    return level, slope, innovation


@njit(IO(I, int64, int64), cache=True, nogil=True)
def condition_event_kernel(condition, opcode, periods):
    """0 first false->true; 1 consecutive confirmation; 2 event cooldown.

    Conditions are -1 unknown, 0 false, 1 true. First occurrence after unknown
    is unknown: an edge cannot be inferred across a data gap. Cooldown skips
    the next ``periods`` axis positions after an emitted event.
    """
    if opcode < 0 or opcode > 2 or periods < 0 or (opcode == 1 and periods < 1):
        raise ValueError("Invalid event operation or period.")
    output = np.full(condition.size, -1, dtype=np.int64)
    streak = 0
    last_event = -periods - 1
    for t in range(condition.size):
        current = condition[t]
        if current < -1 or current > 1:
            raise ValueError("Conditions must use -1, 0, 1.")
        if current == -1:
            streak = 0
            continue
        if opcode == 0:
            if current == 0:
                output[t] = 0
            elif t > 0 and condition[t - 1] != -1:
                output[t] = 1 if condition[t - 1] == 0 else 0
        elif opcode == 1:
            streak = streak + 1 if current == 1 else 0
            output[t] = 1 if streak >= periods else 0
        else:
            output[t] = 0
            if current == 1 and t - last_event > periods:
                output[t] = 1
                last_event = t
    return output


@njit(FO(I), cache=True, nogil=True)
def condition_values_kernel(condition):
    """Explicit condition-to-numeric conversion, preserving unknown as NaN."""
    output = np.full(condition.size, np.nan)
    for t in range(condition.size):
        value = condition[t]
        if value < -1 or value > 1:
            raise ValueError("Conditions must use -1, 0, 1.")
        if value != -1:
            output[t] = float(value)
    return output


@njit(types.Tuple((MO, MO, int64, int64))(
    F, F, F, F, I, I, int64, int64, int64, int64,
    float64, float64, float64, float64), cache=True, nogil=True)
def simulate_kernel(open_, high, low, close, entry, exit_, start, end,
                    max_holding, cooldown, fee_bps, slippage_bps, take_profit, stop_loss):
    """Single-ETF, all-cash/all-invested daily account; no forced terminal exit.

    Ordinary/max-holding exits execute at the next open before protective exits.
    Intraday stop/take-profit can exit only a position held before this session.
    Stop wins when both levels are touched. Missing/invalid evaluation OHLC
    invalidates that row and all subsequent performance; no gap filling.
    """
    n = close.size
    if (open_.size != n or high.size != n or low.size != n or entry.size != n or exit_.size != n
            or start < 0 or end < start or end > n or max_holding < 1 or cooldown < 0):
        raise ValueError("Invalid price/condition axis or execution bounds.")
    if (not np.isfinite(fee_bps) or not np.isfinite(slippage_bps)
            or not np.isfinite(take_profit) or not np.isfinite(stop_loss)
            or fee_bps < 0 or fee_bps >= 10000 or slippage_bps < 0 or slippage_bps >= 10000
            or take_profit < 0 or stop_loss < 0 or stop_loss >= 1):
        raise ValueError("Invalid execution costs or exit thresholds.")
    path = np.full((n, len(PATH_COLUMNS)), np.nan)
    # One record slot per possible completed trade, plus any open record. No
    # per-trade growth/copy; the caller exposes only a basic slice of this owner.
    trades = np.full((end - start, len(TRADE_COLUMNS)), np.nan)
    fee = fee_bps / 10000.0
    slip = slippage_bps / 10000.0
    cash, units, previous_nav = 1.0, 0.0, 1.0
    previous_benchmark, benchmark_units = 1.0, 0.0
    entry_index, last_exit = -1, start - cooldown - 1
    entry_price, entry_wealth, adverse, favorable = 0.0, 1.0, 0.0, 0.0
    count, status = 0, 0
    for t in range(start, end):
        o, h, l, c = open_[t], high[t], low[t], close[t]
        if not (np.isfinite(o) and np.isfinite(h) and np.isfinite(l) and np.isfinite(c)):
            status = 1
        elif o <= 0 or l <= 0 or c <= 0 or h < max(o, c) or l > min(o, c) or h < l:
            status = 2
        if t > 0 and (entry[t - 1] < -1 or entry[t - 1] > 1 or exit_[t - 1] < -1 or exit_[t - 1] > 1):
            raise ValueError("Conditions must use -1, 0, 1.")
        if status != 0:
            for invalid in range(t, end):
                path[invalid, 9] = 0.0
            break
        action, reason, turnover, paid = 0.0, 0, 0.0, 0.0
        if t == start:
            benchmark_units = 1.0 / o
        benchmark = benchmark_units * c
        held_before = units > 0.0
        exit_price = 0.0
        intraday_exit = False
        if held_before:
            if t > 0 and exit_[t - 1] == 1:
                reason, exit_price = 1, o
            elif t - entry_index >= max_holding:
                reason, exit_price = 4, o
            else:
                stop = entry_price * (1.0 - stop_loss)
                take = entry_price * (1.0 + take_profit)
                if stop_loss > 0.0 and o <= stop:
                    reason, exit_price = 2, o
                elif take_profit > 0.0 and o >= take:
                    reason, exit_price = 3, o
                elif stop_loss > 0.0 and l <= stop:
                    reason, exit_price = 2, stop
                    intraday_exit = True
                elif take_profit > 0.0 and h >= take:
                    reason, exit_price = 3, take
                    intraday_exit = True
            # Full-day extrema after an exit are not realized path evidence.
            # With daily bars only mark close-held sessions and exit price.
            observed_low = exit_price if reason else l
            observed_high = exit_price if reason else h
            adverse = min(adverse, observed_low / entry_price - 1.0)
            favorable = max(favorable, observed_high / entry_price - 1.0)
            if reason:
                executed = exit_price * (1.0 - slip)
                gross = units * executed
                paid = gross * fee
                cash = gross - paid
                turnover = gross / previous_nav
                trades[count, 2] = t
                trades[count, 4] = executed
                trades[count, 5] = cash / entry_wealth - 1.0
                trades[count, 6] = t - entry_index
                trades[count, 7] = reason
                # Daily bars cannot tell whether either extremum preceded an
                # intraday protective exit. Do not invent precise trade MAE/MFE.
                trades[count, 8] = np.nan if intraday_exit else adverse
                trades[count, 9] = np.nan if intraday_exit else favorable
                count += 1
                units, action, last_exit = 0.0, -1.0, t
        # Exit wins when both rules fire. No same-session re-entry after exit.
        if (not held_before and t > 0 and entry[t - 1] == 1 and exit_[t - 1] == 0
                and t - last_exit > cooldown):
            entry_price = o * (1.0 + slip)
            entry_wealth = cash
            units = cash / (entry_price * (1.0 + fee))
            gross = units * entry_price
            paid = gross * fee
            turnover = gross / previous_nav
            cash, entry_index, action = 0.0, t, 1.0
            adverse = min(0.0, l / entry_price - 1.0)
            favorable = max(0.0, h / entry_price - 1.0)
            trades[count, 0] = t - 1
            trades[count, 1] = t
            trades[count, 2] = -1.0
            trades[count, 3] = entry_price
        nav = cash + units * c
        path[t, 0] = nav
        path[t, 1] = benchmark
        path[t, 2] = nav / previous_nav - 1.0
        path[t, 3] = benchmark / previous_benchmark - 1.0
        path[t, 4] = 1.0 if units > 0 else 0.0
        path[t, 5] = turnover
        path[t, 6] = paid / previous_nav
        path[t, 7] = action
        path[t, 8] = reason
        path[t, 9] = 1.0
        previous_nav, previous_benchmark = nav, benchmark
    return path, trades, count, status


@njit(FO(M, M, int64, int64, int64), cache=True, nogil=True)
def analyze_kernel(path, trades, count, start, end):
    """Interval account performance and exit-dated completed-trade statistics.

    Interval returns use the same account, rebased from the previous close;
    trade returns can span the interval boundary and are labelled exit-dated.
    """
    if (path.shape[1] != len(PATH_COLUMNS) or trades.shape[1] != len(TRADE_COLUMNS)
            or start < 0 or end < start or end > path.shape[0] or count < 0 or count > trades.shape[0]):
        raise ValueError("Invalid summary bounds or matrix schema.")
    out = np.full(len(SUMMARY_COLUMNS), np.nan)
    observations, exposure, turnover, costs, total, squared = 0, 0.0, 0.0, 0.0, 0.0, 0.0
    nav, benchmark, peak, drawdown = 1.0, 1.0, 1.0, 0.0
    invalid = False
    for t in range(start, end):
        if np.isnan(path[t, 9]):
            continue  # Outside the simulated account interval.
        if path[t, 9] != 1.0 or not np.isfinite(path[t, 2]):
            invalid = True
            continue
        ret = path[t, 2]
        observations += 1
        total += ret
        squared += ret * ret
        nav *= 1.0 + ret
        benchmark *= 1.0 + path[t, 3]
        peak = max(peak, nav)
        drawdown = min(drawdown, nav / peak - 1.0)
        exposure += path[t, 4]
        turnover += path[t, 5]
        costs += path[t, 6]
    out[0] = observations
    if observations and not invalid:
        out[1], out[2], out[3], out[4] = nav - 1.0, benchmark - 1.0, nav - benchmark, drawdown
        out[5] = nav ** (252.0 / observations) - 1.0
        if observations > 1:
            out[6] = np.sqrt(max(0.0, (squared - total * total / observations) / (observations - 1) * 252.0))
        out[7], out[8], out[9] = exposure / observations, turnover, costs
    returns = np.empty(count, dtype=np.float64)
    chosen, wins, gains, losses, total_return, holding = 0, 0, 0.0, 0.0, 0.0, 0.0
    for i in range(count):
        if start <= trades[i, 2] < end:
            value = trades[i, 5]
            returns[chosen] = value
            chosen += 1
            wins += value > 0.0
            gains += max(value, 0.0)
            losses -= min(value, 0.0)
            total_return += value
            holding += trades[i, 6]
    out[10] = chosen
    if chosen:
        ordered = np.sort(returns[:chosen])
        median = ordered[chosen // 2]
        if chosen % 2 == 0:
            median = (ordered[chosen // 2 - 1] + median) / 2.0
        out[11], out[12], out[13], out[14] = wins / chosen, total_return / chosen, median, holding / chosen
        if losses > 0:
            out[15] = gains / losses
    return out


@njit(MO(F, F, F, F, I, int64, int64), cache=True, nogil=True)
def signal_quality_kernel(open_, high, low, close, entry, start, end):
    """Forward 5/10/15-session labels from next open, never decision inputs.

    Fully contained observations only. These labels ignore execution filters,
    exits and costs and must not be presented as realized strategy returns.
    """
    n = close.size
    if (open_.size != n or high.size != n or low.size != n or entry.size != n
            or start < 0 or end < start or end > n):
        raise ValueError("Invalid signal quality axis or bounds.")
    output = np.full((3, len(QUALITY_COLUMNS)), np.nan)
    labels = np.empty(end - start, dtype=np.float64)
    for row in range(3):
        horizon = 5 * (row + 1)
        signals, eligible, positive = 0, 0, 0
        total, adverse_sum, favorable_sum, comfort_sum = 0.0, 0.0, 0.0, 0.0
        for t in range(start, end):
            if entry[t] < -1 or entry[t] > 1:
                raise ValueError("Conditions must use -1, 0, 1.")
            if entry[t] != 1:
                continue
            signals += 1
            last = t + horizon
            if last >= end:
                continue
            price = open_[t + 1]
            if not np.isfinite(price) or price <= 0:
                continue
            valid, adverse, favorable, comfortable = True, 0.0, 0.0, 0
            for j in range(t + 1, last + 1):
                o, h, l, c = open_[j], high[j], low[j], close[j]
                if (not np.isfinite(o) or not np.isfinite(h) or not np.isfinite(l) or not np.isfinite(c)
                        or l <= 0 or min(o, c) < l or max(o, c) > h or h < l):
                    valid = False
                    break
                adverse = min(adverse, l / price - 1.0)
                favorable = max(favorable, h / price - 1.0)
                comfortable += c > price
            if not valid:
                continue
            value = close[last] / price - 1.0
            labels[eligible] = value
            eligible += 1
            positive += value > 0.0
            total += value
            adverse_sum += adverse
            favorable_sum += favorable
            comfort_sum += comfortable / horizon
        output[row, 0], output[row, 1], output[row, 2] = horizon, signals, eligible
        if eligible:
            ordered = np.sort(labels[:eligible])
            median = ordered[eligible // 2]
            if eligible % 2 == 0:
                median = (ordered[eligible // 2 - 1] + median) / 2.0
            output[row, 3] = positive / eligible
            output[row, 4] = total / eligible
            output[row, 5] = median
            output[row, 6] = adverse_sum / eligible
            output[row, 7] = favorable_sum / eligible
            output[row, 8] = comfort_sum / eligible
    return output


@njit(FO(I, I, M, int64, int64), cache=True, nogil=True)
def diagnostics_kernel(entry, month_ids, path, start, end):
    """Signal coverage/concentration and account stability, including partial months.

    Month identifiers must be stable ordered calendar labels such as YYYYMM.
    Compound daily account returns independently within each month. Any missing
    performance day makes both positive-month fractions unavailable, never zero.
    """
    if (entry.size != month_ids.size or path.shape[0] != entry.size or path.shape[1] != len(PATH_COLUMNS)
            or start < 0 or end < start or end > entry.size):
        raise ValueError("Invalid diagnostics axis or bounds.")
    output = np.full(len(DIAGNOSTIC_COLUMNS), np.nan)
    signals, known, active_months, months, largest, month_signals = 0, 0, 0, 0, 0, 0
    positive, positive_excess = 0, 0
    nav, benchmark, invalid = 1.0, 1.0, False
    for t in range(start, end):
        if month_ids[t] < 0 or (t > start and month_ids[t] < month_ids[t - 1]):
            raise ValueError("Month labels must be nonnegative and ordered.")
        if t > start and month_ids[t] != month_ids[t - 1]:
            months += 1
            active_months += month_signals > 0
            largest = max(largest, month_signals)
            positive += nav > 1.0
            positive_excess += nav > benchmark
            nav, benchmark, month_signals = 1.0, 1.0, 0
        if entry[t] < -1 or entry[t] > 1:
            raise ValueError("Conditions must use -1, 0, 1.")
        known += entry[t] != -1
        signals += entry[t] == 1
        month_signals += entry[t] == 1
        if path[t, 9] != 1 or not np.isfinite(path[t, 2]) or not np.isfinite(path[t, 3]):
            invalid = True
        else:
            nav *= 1.0 + path[t, 2]
            benchmark *= 1.0 + path[t, 3]
    if end > start:
        months += 1
        active_months += month_signals > 0
        largest = max(largest, month_signals)
        positive += nav > 1.0
        positive_excess += nav > benchmark
    output[0], output[1], output[2], output[3] = signals, known, active_months, months
    if signals:
        output[4] = largest / signals
    if months and not invalid:
        output[5], output[6] = positive / months, positive_excess / months
    return output


KERNELS = {"alpha_beta": alpha_beta_kernel, "condition_event": condition_event_kernel,
           "condition_values": condition_values_kernel,
           "series_status": series_status_kernel, "availability_status": availability_status_kernel,
           "execution_volume_status": execution_volume_status_kernel,
           "diagnostics": diagnostics_kernel,
           "simulate": simulate_kernel, "analyze": analyze_kernel, "signal_quality": signal_quality_kernel}
for _kernel in KERNELS.values():
    _kernel.disable_compile()


def timing_execution_audit():
    """Audit eagerly compiled signatures; no lazy compilation."""
    for kernel in KERNELS.values():
        if (not kernel.nopython_signatures or len(kernel.signatures) != len(kernel.nopython_signatures)
                or kernel._can_compile):
            raise RuntimeError("择时计算内核尚未完成固定签名预热。")
    return {"complete": True, **validate_execution_audit({
        "execution_backend": "numba_njit_fixed_signature", "engine_version": ENGINE_VERSION,
        "nopython": True, "python_fallback": 0, "python_operator_calls": 0,
        "request_time_compilation": 0,
        "kernel_signatures": {name: [str(sig) for sig in kernel.signatures] for name, kernel in KERNELS.items()},
    })}


def warm_timing_kernels():
    """Startup verifies every production dispatcher is compiled and frozen."""
    return timing_execution_audit()

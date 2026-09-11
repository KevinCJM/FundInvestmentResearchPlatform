"""Fixed-signature return construction kernels; no pandas or request compilation."""
from __future__ import annotations

import numpy as np
from numba import float64, int64, njit, types

from .numba_kernels import F1, F2, F3, I1, I2, correlation_kernel, mean_stats_kernel, ranks_kernel


@njit(F3(F2, I1, int64, int64), cache=True, nogil=True)
def rolling_ic_kernel(values, samples, window, minimum):
    output = np.full((values.shape[0], values.shape[1], 5), np.nan)
    for d in range(values.shape[0]):
        if samples[d] < 0:
            continue
        start = max(0, d - window + 1)
        for f in range(values.shape[1]):
            history = np.full(d - start + 1, np.nan)
            for j in range(start, d + 1):
                if samples[j] == samples[d]:
                    history[j - start] = values[j, f]
            stats = mean_stats_kernel(history)
            output[d, f, 0] = stats[0]
            if stats[0] >= minimum:
                output[d, f] = stats
    return output


@njit(F3(F2, int64), cache=True, nogil=True)
def spread_targets_kernel(scores, quantiles):
    # Axis 1 is low/high. Average ranks never split identical scores by asset ID.
    output = np.zeros((scores.shape[0], 2, scores.shape[1]))
    for d in range(scores.shape[0]):
        ranks = ranks_kernel(scores[d].copy())
        count = 0
        for value in ranks:
            count += np.isfinite(value)
        if count < max(3, quantiles):
            continue
        counts = np.zeros(2)
        for a in range(scores.shape[1]):
            if not np.isfinite(ranks[a]):
                continue
            group = min(quantiles - 1, int((ranks[a] - 1.0) * quantiles / count))
            leg = 0 if group == 0 else 1 if group == quantiles - 1 else -1
            if leg >= 0:
                output[d, leg, a] = 1.0
                counts[leg] += 1.0
        if counts[0] == 0 or counts[1] == 0:
            output[d] = 0.0
        else:
            output[d, 0] /= counts[0]
            output[d, 1] /= counts[1]
    return output


@njit(types.Tuple((F2, F3))(F2, I1, F2, int64, float64), cache=True, nogil=True)
def spread_returns_kernel(prices, decisions, scores, quantiles, cost_bps):
    # low, high, gross spread, net spread diagnostic, two-sided turnover, fee.
    path = np.full((prices.shape[0], 6), np.nan)
    targets = spread_targets_kernel(scores, quantiles)
    if decisions.size == 0:
        return path, targets
    weights = np.zeros((2, prices.shape[1]))
    alive = np.zeros(2, dtype=np.int64)
    started = np.zeros(2, dtype=np.int64)
    decision = 0
    first = decisions[0] + 1
    for t in range(first, prices.shape[0]):
        gross = np.full(2, np.nan)
        for leg in range(2):
            if not alive[leg]:
                continue
            value = 0.0
            for a in range(prices.shape[1]):
                if weights[leg, a] <= 0:
                    continue
                if not (np.isfinite(prices[t, a]) and np.isfinite(prices[t - 1, a])
                        and prices[t, a] > 0 and prices[t - 1, a] > 0):
                    alive[leg] = 0
                    break
                value += weights[leg, a] * (prices[t, a] / prices[t - 1, a] - 1.0)
            if alive[leg]:
                gross[leg] = value
                for a in range(prices.shape[1]):
                    if weights[leg, a] > 0:
                        weights[leg, a] *= prices[t, a] / prices[t - 1, a] / (1.0 + value)
        turnover = 0.0
        had_history = started[0] != 0 or started[1] != 0
        if decision < decisions.size and t == decisions[decision] + 1:
            for leg in range(2):
                if started[leg] and not alive[leg]:
                    turnover = np.nan  # Unknown old weights cannot have invented turnover.
                selected = 0
                valid = True
                for a in range(prices.shape[1]):
                    target = targets[decision, leg, a]
                    if target > 0:
                        selected += 1
                        if not np.isfinite(prices[t, a]) or prices[t, a] <= 0:
                            valid = False
                    turnover += abs(target - weights[leg, a])
                    weights[leg, a] = target
                alive[leg] = int(valid and selected > 0)
                if not started[leg] and alive[leg]:
                    if not had_history:
                        gross[leg] = 0.0  # Only the joint initial entry can have a zero starting return.
                    started[leg] = 1
            decision += 1
        fee = turnover * cost_bps / 10000.0
        path[t, 0] = gross[0]
        path[t, 1] = gross[1]
        path[t, 2] = gross[1] - gross[0]
        path[t, 3] = path[t, 2] - fee
        path[t, 4] = turnover
        path[t, 5] = fee
    return path, targets


@njit(float64(F1, float64), cache=True, nogil=True)
def linear_quantile_kernel(values, probability):
    if values.size == 0:
        return np.nan
    ordered = np.sort(values)
    position = (ordered.size - 1) * probability
    lower = int(position)
    upper = min(lower + 1, ordered.size - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


@njit(types.Tuple((F2, F2, I2))(F2, F2, F1, I1, F3), cache=True, nogil=True)
def ff3_returns_kernel(returns, lagged_caps, rf, formation_days, descriptors):
    # MKT_RF, SMB, HML, RF, SL, SM, SH, BL, BM, BH.
    output = np.full((returns.shape[0], 10), np.nan)
    evidence = np.zeros((formation_days.size, 9))  # six counts, size / BM30 / BM70.
    memberships = np.full((formation_days.size, returns.shape[1]), -1, dtype=np.int64)
    weights = np.zeros((6, returns.shape[1]))
    alive = np.zeros(6, dtype=np.int64)
    formation = 0
    for t in range(returns.shape[0]):
        if t > 0:
            output[t, 3] = rf[t]
            total_cap = 0.0
            market_return = 0.0
            valid_market = True
            for a in range(returns.shape[1]):
                cap = lagged_caps[t, a]
                if np.isfinite(cap) and cap > 0:
                    total_cap += cap
                    if not np.isfinite(returns[t, a]):
                        valid_market = False
                    else:
                        market_return += cap * returns[t, a]
            if valid_market and total_cap > 0 and np.isfinite(rf[t]):
                output[t, 0] = market_return / total_cap - rf[t]
            for group in range(6):
                if not alive[group]:
                    continue
                value = 0.0
                for a in range(returns.shape[1]):
                    if weights[group, a] > 0:
                        if not np.isfinite(returns[t, a]):
                            alive[group] = 0
                            break
                        value += weights[group, a] * returns[t, a]
                if alive[group]:
                    output[t, 4 + group] = value
                    if value <= -1.0:
                        alive[group] = 0
                    else:
                        for a in range(returns.shape[1]):
                            if weights[group, a] > 0:
                                weights[group, a] *= (1.0 + returns[t, a]) / (1.0 + value)
            output[t, 1] = (output[t, 4] + output[t, 5] + output[t, 6]
                            - output[t, 7] - output[t, 8] - output[t, 9]) / 3.0
            output[t, 2] = (output[t, 6] + output[t, 9] - output[t, 4] - output[t, 7]) / 2.0
        # Formation is at the close, after today's old-portfolio return is earned.
        if formation < formation_days.size and t == formation_days[formation]:
            count = 0
            for a in range(returns.shape[1]):
                if np.isfinite(descriptors[formation, a, 0]) and descriptors[formation, a, 3] == 1.0:
                    count += 1
            weights[:] = 0.0
            alive[:] = 0
            if count < 6:
                evidence[formation, 6:] = np.nan
                formation += 1
                continue
            sizes = np.empty(count)
            ratios = np.empty(count)
            j = 0
            for a in range(returns.shape[1]):
                if np.isfinite(descriptors[formation, a, 0]) and descriptors[formation, a, 3] == 1.0:
                    sizes[j] = descriptors[formation, a, 0]
                    ratios[j] = descriptors[formation, a, 2] / descriptors[formation, a, 1]
                    j += 1
            size_cut = linear_quantile_kernel(sizes, 0.5)
            low_cut = linear_quantile_kernel(ratios, 0.3)
            high_cut = linear_quantile_kernel(ratios, 0.7)
            evidence[formation, 6] = size_cut
            evidence[formation, 7] = low_cut
            evidence[formation, 8] = high_cut
            totals = np.zeros(6)
            for a in range(returns.shape[1]):
                cap = descriptors[formation, a, 0]
                if not np.isfinite(cap):
                    continue
                ratio = descriptors[formation, a, 2] / descriptors[formation, a, 1]
                value_group = 0 if ratio <= low_cut else 1 if ratio <= high_cut else 2
                group = value_group + (0 if cap <= size_cut else 3)
                memberships[formation, a] = group
                weights[group, a] = cap
                totals[group] += cap
                evidence[formation, group] += 1.0
            for group in range(6):
                if totals[group] > 0:
                    weights[group] /= totals[group]
                    alive[group] = 1
            formation += 1
    return output, evidence, memberships


@njit(types.Tuple((F2, F2, F2))(F2), cache=True, nogil=True)
def return_diagnostics_kernel(values):
    stats = np.full((values.shape[1], 5), np.nan)
    correlation = np.full((values.shape[1], values.shape[1]), np.nan)
    cumulative = np.full(values.shape, np.nan)
    for f in range(values.shape[1]):
        column = values[:, f].copy()
        stats[f] = mean_stats_kernel(column)
        total = 0.0
        started = False
        broken = False
        for t in range(values.shape[0]):
            if not np.isfinite(values[t, f]):
                if started:
                    broken = True
                continue
            started = True
            if not broken:
                total += values[t, f]
                cumulative[t, f] = total
        for g in range(values.shape[1]):
            correlation[f, g] = correlation_kernel(column, values[:, g].copy())
    return stats, correlation, cumulative


KERNELS = (rolling_ic_kernel, spread_targets_kernel, spread_returns_kernel,
           linear_quantile_kernel, ff3_returns_kernel, return_diagnostics_kernel)


def warm_return_kernels():
    prices = np.ones((40, 6), dtype=np.float64)
    decisions = np.array([1, 20], dtype=np.int64)
    scores = np.ascontiguousarray(np.tile(np.arange(6, dtype=np.float64), (2, 1)))
    spread_returns_kernel(prices, decisions, scores, 3, 5.0)
    rolling_ic_kernel(scores, np.array([0, 1], dtype=np.int64), 3, 2)
    descriptors = np.empty((1, 6, 4), dtype=np.float64)
    descriptors[0, :, 0] = np.array([1., 2., 3., 10., 11., 12.])
    descriptors[0, :, 1] = 1.0
    descriptors[0, :, 2] = np.array([1., 2., 3., 1., 2., 3.])
    descriptors[0, :, 3] = 1.0
    ff3_returns_kernel(prices * 0.001, prices, np.zeros(40), np.array([0], dtype=np.int64), descriptors)
    return_diagnostics_kernel(scores)
    return {"complete": all(len(kernel.nopython_signatures) == 1 for kernel in KERNELS),
            "kernel_signatures": {kernel.py_func.__name__: [str(s) for s in kernel.nopython_signatures]
                                  for kernel in KERNELS}}

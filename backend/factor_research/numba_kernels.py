"""Eager fixed-signature numerical kernels. No request compilation or Python fallback."""
from __future__ import annotations

import numpy as np
from numba import float64, int64, njit, types
from backend.compute_policy import validate_execution_audit
from .catalog import ENGINE_VERSION

F1 = float64[::1]
F2 = float64[:, ::1]
F3 = float64[:, :, ::1]
I1 = int64[::1]
I2 = int64[:, ::1]


@njit(F1(F1), cache=True, nogil=True)
def ranks_kernel(values):
    result = np.full(values.size, np.nan)
    for i in range(values.size):
        if not np.isfinite(values[i]):
            continue
        lower = 0
        equal = 0
        for j in range(values.size):
            if np.isfinite(values[j]):
                lower += values[j] < values[i]
                equal += values[j] == values[i]
        result[i] = lower + (equal + 1.0) / 2.0
    return result


@njit(float64(F1, F1), cache=True, nogil=True)
def correlation_kernel(left, right):
    count = 0
    sx = 0.0
    sy = 0.0
    for i in range(left.size):
        if np.isfinite(left[i]) and np.isfinite(right[i]):
            count += 1
            sx += left[i]
            sy += right[i]
    if count < 3:
        return np.nan
    mx, my = sx / count, sy / count
    xx = 0.0
    yy = 0.0
    xy = 0.0
    for i in range(left.size):
        if np.isfinite(left[i]) and np.isfinite(right[i]):
            x, y = left[i] - mx, right[i] - my
            xx += x * x
            yy += y * y
            xy += x * y
    if xx <= 1e-24 or yy <= 1e-24:
        return np.nan
    return xy / np.sqrt(xx * yy)


@njit(F3(F2, I2, I1, I1, I2), cache=True, nogil=True)
def features_kernel(prices, available, days, decisions, parameters):
    output = np.full((decisions.size, prices.shape[1], parameters.shape[0]), np.nan)
    for d in range(decisions.size):
        t = decisions[d]
        if t < 1:
            continue
        cutoff = days[t - 1]  # Date-only disclosure becomes usable next session.
        for a in range(prices.shape[1]):
            last = t
            while last >= 0:
                if np.isfinite(prices[last, a]) and prices[last, a] > 0 and available[last, a] <= cutoff:
                    break
                last -= 1
            if last < 0 or t - last > 5:
                continue
            for f in range(parameters.shape[0]):
                op, window, skip = parameters[f, 0], parameters[f, 1], parameters[f, 2]
                end = last - skip
                start = end - window
                if start < 0:
                    continue
                valid = True
                for i in range(start, end + 1):
                    if not np.isfinite(prices[i, a]) or prices[i, a] <= 0 or available[i, a] > cutoff:
                        valid = False
                        break
                if not valid:
                    continue
                if op == 0 or op == 3:
                    value = prices[end, a] / prices[start, a] - 1.0
                    output[d, a, f] = -value if op == 3 else value
                elif op == 1:
                    total = 0.0
                    for i in range(start + 1, end + 1):
                        total += prices[i, a] / prices[i - 1, a] - 1.0
                    mean = total / window
                    squared = 0.0
                    for i in range(start + 1, end + 1):
                        delta = prices[i, a] / prices[i - 1, a] - 1.0 - mean
                        squared += delta * delta
                    output[d, a, f] = np.sqrt(squared / (window - 1.0) * 252.0)
                elif op == 2:
                    peak = prices[start, a]
                    drawdown = 0.0
                    for i in range(start, end + 1):
                        peak = max(peak, prices[i, a])
                        drawdown = min(drawdown, prices[i, a] / peak - 1.0)
                    output[d, a, f] = drawdown
    return output


@njit(types.Tuple((F3, F2))(F3, I1, F1, int64), cache=True, nogil=True)
def normalize_kernel(raw, directions, weights, method):
    normalized = np.full(raw.shape, np.nan)
    scores = np.full((raw.shape[0], raw.shape[1]), np.nan)
    weight_sum = 0.0
    for value in weights:
        weight_sum += value
    if weight_sum <= 0:
        return normalized, scores
    for d in range(raw.shape[0]):
        for f in range(raw.shape[2]):
            values = np.ascontiguousarray(raw[d, :, f] * directions[f])
            count = 0
            for value in values:
                count += np.isfinite(value)
            if count < 3:
                continue
            if method == 0:
                ranks = ranks_kernel(values)
                for a in range(values.size):
                    if np.isfinite(ranks[a]):
                        normalized[d, a, f] = (ranks[a] - 1.0) / (count - 1.0)
            else:
                compact = np.empty(count)
                j = 0
                for value in values:
                    if np.isfinite(value):
                        compact[j] = value
                        j += 1
                compact.sort()
                lo = compact[int(np.floor(0.05 * (count - 1)))]
                hi = compact[int(np.ceil(0.95 * (count - 1)))]
                total = 0.0
                for a in range(values.size):
                    if np.isfinite(values[a]):
                        values[a] = min(hi, max(lo, values[a]))
                        total += values[a]
                mean = total / count
                squared = 0.0
                for value in values:
                    if np.isfinite(value):
                        squared += (value - mean) ** 2
                std = np.sqrt(squared / (count - 1.0))
                for a in range(values.size):
                    if np.isfinite(values[a]):
                        normalized[d, a, f] = (values[a] - mean) / std if std > 1e-12 else 0.0
        for a in range(raw.shape[1]):
            total = 0.0
            valid = True
            for f in range(raw.shape[2]):
                if not np.isfinite(normalized[d, a, f]):
                    valid = False
                    break
                total += normalized[d, a, f] * weights[f] / weight_sum
            if valid:
                scores[d, a] = total * 100.0 if method == 0 else total
    return normalized, scores


@njit(F2(F2, I1, int64), cache=True, nogil=True)
def labels_kernel(prices, decisions, horizon):
    output = np.full((decisions.size, prices.shape[1]), np.nan)
    for d in range(decisions.size):
        entry = decisions[d] + 1
        end = entry + horizon
        if end >= prices.shape[0]:
            continue
        for a in range(prices.shape[1]):
            valid = True
            for t in range(entry, end + 1):
                if not np.isfinite(prices[t, a]) or prices[t, a] <= 0:
                    valid = False
                    break
            if valid:
                output[d, a] = prices[end, a] / prices[entry, a] - 1.0
    return output


@njit(types.Tuple((F2, F2, F2, F2))(F3, F2, F2, int64), cache=True, nogil=True)
def diagnostics_kernel(normalized, scores, labels, quantiles):
    ds, assets, factors = normalized.shape
    ic = np.full((ds, factors + 1), np.nan)
    rank_ic = np.full((ds, factors + 1), np.nan)
    groups = np.full((ds, quantiles), np.nan)
    counts = np.zeros((ds, factors + 1))
    for d in range(ds):
        for f in range(factors + 1):
            x = np.ascontiguousarray(scores[d] if f == factors else normalized[d, :, f])
            y = labels[d].copy()
            for a in range(assets):
                if not np.isfinite(x[a]) or not np.isfinite(y[a]):
                    x[a] = np.nan
                    y[a] = np.nan
                else:
                    counts[d, f] += 1
            ic[d, f] = correlation_kernel(x, y)
            rank_ic[d, f] = correlation_kernel(ranks_kernel(x), ranks_kernel(y))
        ranks = ranks_kernel(scores[d].copy())
        finite = 0
        for a in range(assets):
            finite += np.isfinite(scores[d, a])
        totals = np.zeros(quantiles)
        group_counts = np.zeros(quantiles)
        if finite < 3:
            continue
        for a in range(assets):
            if np.isfinite(ranks[a]) and np.isfinite(labels[d, a]):
                group = min(quantiles - 1, int((ranks[a] - 1.0) * quantiles / finite))
                totals[group] += labels[d, a]
                group_counts[group] += 1.0
        for q in range(quantiles):
            if group_counts[q]:
                groups[d, q] = totals[q] / group_counts[q]
    return ic, rank_ic, groups, counts


@njit(F1(F1), cache=True, nogil=True)
def mean_stats_kernel(values):
    out = np.full(5, np.nan)
    count = 0
    total = 0.0
    positive = 0
    for value in values:
        if np.isfinite(value):
            count += 1
            total += value
            positive += value > 0
    out[0] = count
    if count:
        mean = total / count
        out[1] = mean
        out[4] = positive / count
        if count > 1:
            squared = 0.0
            for value in values:
                if np.isfinite(value):
                    squared += (value - mean) ** 2
            out[2] = np.sqrt(squared / (count - 1.0))
            if out[2] > 1e-12:
                out[3] = mean / out[2]  # ICIR, deliberately not annualized.
    return out


@njit(F2(F3, I1), cache=True, nogil=True)
def factor_correlation_kernel(values, mask):
    factors = values.shape[2]
    out = np.full((factors, factors), np.nan)
    n = values.shape[0] * values.shape[1]
    for f in range(factors):
        for g in range(factors):
            x = np.full(n, np.nan)
            y = np.full(n, np.nan)
            for d in range(values.shape[0]):
                if mask[d] == 0:
                    continue
                for a in range(values.shape[1]):
                    i = d * values.shape[1] + a
                    x[i] = values[d, a, f]
                    y[i] = values[d, a, g]
            out[f, g] = correlation_kernel(x, y)
    return out


@njit(F2(F2, int64), cache=True, nogil=True)
def target_weights_kernel(scores, top_n):
    out = np.zeros(scores.shape)
    for d in range(scores.shape[0]):
        ranks = ranks_kernel(scores[d].copy())
        valid = 0
        for value in scores[d]:
            valid += np.isfinite(value)
        if valid < 3:
            continue
        finite = np.empty(valid)
        i = 0
        for value in scores[d]:
            if np.isfinite(value):
                finite[i] = value
                i += 1
        finite.sort()
        threshold = finite[max(0, valid - top_n)]
        selected = 0
        for a in range(scores.shape[1]):
            selected += np.isfinite(scores[d, a]) and scores[d, a] >= threshold
        for a in range(scores.shape[1]):
            if np.isfinite(scores[d, a]) and scores[d, a] >= threshold:
                out[d, a] = 1.0 / selected  # Include all ties at the boundary.
    return out


@njit(types.Tuple((F2, F2))(F2, F1, I1, F2, int64, float64), cache=True, nogil=True)
def backtest_kernel(prices, benchmark, decisions, scores, top_n, cost_bps):
    path = np.full((prices.shape[0], 8), np.nan)
    targets = target_weights_kernel(scores, top_n)
    if decisions.size == 0:
        return path, targets
    weights = np.zeros(prices.shape[1])
    nav = 1.0
    benchmark_nav = 1.0
    alive = True
    benchmark_alive = True
    decision = 0
    first = decisions[0] + 1
    for t in range(first, prices.shape[0]):
        gross = 0.0
        for a in range(prices.shape[1]):
            if weights[a] > 0.0:
                if not (np.isfinite(prices[t, a]) and np.isfinite(prices[t - 1, a])
                        and prices[t, a] > 0 and prices[t - 1, a] > 0):
                    alive = False
                elif alive:
                    gross += weights[a] * (prices[t, a] / prices[t - 1, a] - 1.0)
        if alive:
            for a in range(prices.shape[1]):
                if weights[a] > 0.0:
                    weights[a] *= prices[t, a] / prices[t - 1, a] / (1.0 + gross)
        turnover = 0.0
        if decision < decisions.size and t == decisions[decision] + 1:
            for a in range(prices.shape[1]):
                if targets[decision, a] > 0 and (not np.isfinite(prices[t, a]) or prices[t, a] <= 0):
                    alive = False
                turnover += abs(targets[decision, a] - weights[a])
                weights[a] = targets[decision, a]
            decision += 1
        fee = turnover * cost_bps / 10000.0
        net = (1.0 + gross) * (1.0 - fee) - 1.0 if alive else np.nan
        nav = nav * (1.0 + net) if alive else np.nan
        bm = 0.0
        if t > first:
            if not (np.isfinite(benchmark[t]) and np.isfinite(benchmark[t - 1])
                    and benchmark[t] > 0 and benchmark[t - 1] > 0):
                benchmark_alive = False
            if benchmark_alive:
                bm = benchmark[t] / benchmark[t - 1] - 1.0
        elif not np.isfinite(benchmark[t]) or benchmark[t] <= 0:
            benchmark_alive = False
        benchmark_nav = benchmark_nav * (1.0 + bm) if benchmark_alive else np.nan
        count = 0
        for a in range(weights.size):
            count += weights[a] > 0
        path[t, 0] = nav
        path[t, 1] = benchmark_nav
        path[t, 2] = net
        path[t, 3] = bm if benchmark_alive else np.nan
        path[t, 4] = turnover
        path[t, 5] = fee
        path[t, 6] = count
        path[t, 7] = 1.0 if alive else 0.0
    return path, targets


@njit(F1(F2, int64, int64), cache=True, nogil=True)
def performance_kernel(path, start, end):
    # days, total, annualized, vol, maxDD, benchmark total, excess total, turnover, fees
    out = np.full(9, np.nan)
    count = 0
    nav = 1.0
    benchmark = 1.0
    peak = 1.0
    dd = 0.0
    total = 0.0
    turnover = 0.0
    fees = 0.0
    invalid = False
    bm_invalid = False
    for t in range(start, end):
        if not np.isfinite(path[t, 7]):  # Warm-up, before first entry.
            continue
        if not np.isfinite(path[t, 2]):
            invalid = True
            continue
        value = path[t, 2]
        count += 1
        total += value
        nav *= 1.0 + value
        peak = max(peak, nav)
        dd = min(dd, nav / peak - 1.0)
        if np.isfinite(path[t, 3]):
            benchmark *= 1.0 + path[t, 3]
        else:
            bm_invalid = True
        turnover += path[t, 4]
        fees += path[t, 5]
    out[0] = count
    if count == 0 or invalid:
        return out
    out[1] = nav - 1.0
    out[2] = nav ** (252.0 / count) - 1.0
    out[4] = dd
    out[5] = benchmark - 1.0 if not bm_invalid else np.nan
    out[6] = nav - benchmark if not bm_invalid else np.nan
    out[7] = turnover
    out[8] = fees
    if count > 1:
        mean = total / count
        squared = 0.0
        for t in range(start, end):
            if np.isfinite(path[t, 2]):
                squared += (path[t, 2] - mean) ** 2
        out[3] = np.sqrt(squared / (count - 1.0) * 252.0)
    return out


@njit(F2(F2), cache=True, nogil=True)
def returns_kernel(prices):
    out = np.full(prices.shape, np.nan)
    for t in range(1, prices.shape[0]):
        for a in range(prices.shape[1]):
            if (np.isfinite(prices[t, a]) and np.isfinite(prices[t - 1, a])
                    and prices[t, a] > 0 and prices[t - 1, a] > 0):
                out[t, a] = prices[t, a] / prices[t - 1, a] - 1.0
    return out


@njit(F1(F1), cache=True, nogil=True)
def simplex_kernel(values):
    ordered = np.sort(values)[::-1]
    cumulative = 0.0
    theta = 0.0
    for j in range(values.size):
        cumulative += ordered[j]
        candidate = (cumulative - 1.0) / (j + 1)
        if ordered[j] > candidate:
            theta = candidate
    result = np.empty(values.size)
    for j in range(values.size):
        result[j] = max(0.0, values[j] - theta)
    return result


@njit(types.Tuple((F2, F2))(F2, F2, F1, int64, int64), cache=True, nogil=True)
def attribution_kernel(returns, x, rf, split, model):
    assets, k = returns.shape[1], x.shape[1]
    coefficients = np.full((assets, k + 1), np.nan)
    stats = np.full((assets, 8), np.nan)
    for a in range(assets):
        n = 0
        mx = np.zeros(k)
        my = 0.0
        valid = np.zeros(returns.shape[0], dtype=np.int64)
        for t in range(returns.shape[0]):
            okay = np.isfinite(returns[t, a]) and np.isfinite(rf[t])
            for j in range(k):
                okay = okay and np.isfinite(x[t, j])
            valid[t] = int(okay)
            if okay and t < split:
                n += 1
                my += returns[t, a] - rf[t]
                for j in range(k):
                    mx[j] += x[t, j]
        stats[a, 0] = n
        stats[a, 7] = 1.0
        if n < max(30, k * 5):
            continue
        my /= n
        mx /= n
        cov = np.zeros((k, k))
        cy = np.zeros(k)
        for t in range(split):
            if not valid[t]:
                continue
            y = returns[t, a] - rf[t] - my
            for i in range(k):
                xi = x[t, i] - mx[i]
                cy[i] += xi * y / n
                for j in range(k):
                    cov[i, j] += xi * (x[t, j] - mx[j]) / n
        trace = 0.0
        for j in range(k):
            trace += cov[j, j]
        if trace < 1e-20:
            stats[a, 7] = 3.0
            continue
        beta = np.zeros(k)
        if model == 0:
            beta[:] = 1.0 / k
            converged = False
            for iteration in range(20000):
                step = beta.copy()
                for i in range(k):
                    gradient = -cy[i]
                    for j in range(k):
                        gradient += cov[i, j] * beta[j]
                    step[i] -= gradient / trace
                updated = simplex_kernel(step)
                difference = 0.0
                for i in range(k):
                    difference = max(difference, abs(updated[i] - beta[i]))
                beta = updated
                if difference < 1e-10:
                    converged = True
                    break
            if not converged:
                stats[a, 7] = 2.0
                continue
        else:
            matrix = cov.copy()
            rhs = cy.copy()
            singular = False
            for j in range(k):
                pivot = j
                for i in range(j + 1, k):
                    if abs(matrix[i, j]) > abs(matrix[pivot, j]):
                        pivot = i
                if abs(matrix[pivot, j]) < trace * 1e-10:
                    singular = True
                    break
                if pivot != j:
                    for v in range(k):
                        tmp = matrix[j, v]
                        matrix[j, v] = matrix[pivot, v]
                        matrix[pivot, v] = tmp
                    tmp = rhs[j]
                    rhs[j] = rhs[pivot]
                    rhs[pivot] = tmp
                divisor = matrix[j, j]
                for v in range(j, k):
                    matrix[j, v] /= divisor
                rhs[j] /= divisor
                for i in range(k):
                    if i == j:
                        continue
                    scale = matrix[i, j]
                    for v in range(j, k):
                        matrix[i, v] -= scale * matrix[j, v]
                    rhs[i] -= scale * rhs[j]
            if singular:
                stats[a, 7] = 3.0
                continue
            beta = rhs
        alpha = my
        for j in range(k):
            alpha -= beta[j] * mx[j]
            coefficients[a, j] = beta[j]
        coefficients[a, k] = alpha
        stats[a, 4] = alpha * 252.0
        stats[a, 7] = 0.0
        for part in range(2):
            lo, hi = (0, split) if part == 0 else (split, returns.shape[0])
            count = 0
            sy = 0.0
            residual_sum = 0.0
            sse = 0.0
            for t in range(lo, hi):
                if valid[t]:
                    y = returns[t, a] - rf[t]
                    fitted = alpha
                    for j in range(k):
                        fitted += beta[j] * x[t, j]
                    residual = y - fitted
                    residual_sum += residual
                    sse += residual * residual
                    sy += y
                    count += 1
            stats[a, part] = count
            if count < 2:
                continue
            mean = sy / count
            sst = 0.0
            for t in range(lo, hi):
                if valid[t]:
                    sst += (returns[t, a] - rf[t] - mean) ** 2
            if sst > 1e-20:
                stats[a, 2 + part] = 1.0 - sse / sst
            variance = max(0.0, sse - residual_sum * residual_sum / count) / (count - 1)
            stats[a, 5 + part] = np.sqrt(variance * 252.0)
    return coefficients, stats


@njit(F2(F2, F1), cache=True, nogil=True)
def portfolio_profile_kernel(values, weights):
    out = np.full((values.shape[1], 2), np.nan)
    for f in range(values.shape[1]):
        covered = 0.0
        total = 0.0
        for a in range(values.shape[0]):
            if np.isfinite(values[a, f]):
                covered += weights[a]
                total += weights[a] * values[a, f]
        out[f, 1] = covered
        if covered > 0:
            out[f, 0] = total / covered
    return out



@njit(F2(F2, F2, F1, F1, int64), cache=True, nogil=True)
def score_table_kernel(raw, normalized, scores, weights, method):
    out = np.full((scores.size, 3 + raw.shape[1] * 3), np.nan)
    ranks = ranks_kernel(scores)
    count = 0
    total_weight = 0.0
    for value in scores:
        count += np.isfinite(value)
    for value in weights:
        total_weight += value
    for a in range(scores.size):
        out[a, 0] = scores[a]
        if np.isfinite(scores[a]):
            out[a, 1] = count + 1.0 - ranks[a]
            out[a, 2] = (ranks[a] - 1.0) / (count - 1.0) if count > 1 else 0.5
        for f in range(raw.shape[1]):
            out[a, 3 + f * 3] = raw[a, f]
            out[a, 4 + f * 3] = normalized[a, f]
            out[a, 5 + f * 3] = normalized[a, f] * weights[f] / total_weight * (100.0 if method == 0 else 1.0)
    return out


@njit(F1(F1, F1), cache=True, nogil=True)
def drift_kernel(old, new):
    total = 0.0
    common = 0
    current = 0
    for i in range(old.size):
        current += np.isfinite(new[i])
        if np.isfinite(old[i]) and np.isfinite(new[i]):
            common += 1
            total += abs(new[i] - old[i])
    return np.array([correlation_kernel(old, new), total / common if common else np.nan,
                     float(common), current / new.size if new.size else np.nan])

KERNELS = (ranks_kernel, correlation_kernel, features_kernel, normalize_kernel,
           labels_kernel, diagnostics_kernel, mean_stats_kernel, factor_correlation_kernel,
           target_weights_kernel, backtest_kernel, performance_kernel, returns_kernel,
           simplex_kernel, attribution_kernel, portfolio_profile_kernel, score_table_kernel, drift_kernel)


def execution_audit():
    signatures = {kernel.py_func.__name__: [str(s) for s in kernel.nopython_signatures] for kernel in KERNELS}
    if any(len(kernel.signatures) != len(kernel.nopython_signatures) for kernel in KERNELS):
        raise RuntimeError("因子 NJIT 签名存在非 nopython 通道")
    return validate_execution_audit({
        "execution_backend": "numba_njit_fixed_signature", "engine_version": ENGINE_VERSION,
        "nopython": True, "kernel_signatures": signatures, "python_fallback": 0,
        "python_operator_calls": 0, "request_time_compilation": 0,
    })


def warm_factor_kernels():
    prices = np.ascontiguousarray(np.ones((40, 3)), dtype=np.float64)
    days = np.arange(40, dtype=np.int64)
    available = np.ascontiguousarray(np.tile(days[:, None], (1, 3)))
    decisions = np.array([3, 20, 39], dtype=np.int64)
    params = np.array([[0, 2, 0, 1], [1, 2, 0, -1], [2, 2, 0, 1], [3, 2, 0, 1]], dtype=np.int64)
    raw = features_kernel(prices, available, days, decisions, params)
    norm, scores = normalize_kernel(raw, params[:, 3].copy(), np.ones(4), 0)
    normalize_kernel(raw, params[:, 3].copy(), np.ones(4), 1)
    labels = labels_kernel(prices, decisions, 2)
    diagnostics_kernel(norm, scores, labels, 3)
    factor_correlation_kernel(norm, np.ones(3, dtype=np.int64))
    mean_stats_kernel(scores[0].copy())
    path, _ = backtest_kernel(prices, prices[:, 0].copy(), decisions, scores, 2, 5.0)
    performance_kernel(path, 0, 40)
    ret = returns_kernel(prices)
    attribution_kernel(ret, ret, np.zeros(40), 32, 0)
    attribution_kernel(ret, ret, np.zeros(40), 32, 1)
    simplex_kernel(np.array([0.2, 0.3, 0.5]))
    portfolio_profile_kernel(scores, np.ones(3) / 3.0)
    score_table_kernel(raw[0].copy(), norm[0].copy(), scores[0].copy(), np.ones(4), 0)
    drift_kernel(scores[0].copy(), scores[0].copy())
    return {"complete": True, **execution_audit()}

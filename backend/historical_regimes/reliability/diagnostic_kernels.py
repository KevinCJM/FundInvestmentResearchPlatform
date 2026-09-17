"""Bounded diagnostics on readonly arbitrary-stride arrays; compiled at startup."""

import numpy as np
from numba import njit, types

I = types.Array(types.int64, 1, "A", readonly=True)
V = types.Array(types.float64, 1, "A", readonly=True)
F = types.Array(types.float64, 2, "A", readonly=True)


@njit(cache=True)
def quality_kernel(labels, prices, k):
    n = len(labels)
    if n > 20000 or len(prices) != n or not 2 <= k <= 12:
        raise ValueError("Invalid quality dimensions")
    summary = np.zeros(6, np.int64)
    stats = np.full((k, 8), np.nan)
    lengths = np.empty(n, np.float64)
    for c in range(k):
        count = 0
        observations = 0
        returns = 0.0
        return_count = 0
        start = 0
        while start < n:
            if labels[start] != c:
                start += 1
                continue
            end = start + 1
            while end < n and labels[end] == c:
                end += 1
            lengths[count] = end - start
            count += 1
            observations += end - start
            valid_price = end - start >= 2
            for j in range(start, end):
                if not np.isfinite(prices[j]) or prices[j] <= 0:
                    valid_price = False
            if valid_price:
                returns += prices[end - 1] / prices[start] - 1.0
                return_count += 1
            start = end
        stats[c, 0] = observations
        stats[c, 1] = count
        stats[c, 6] = return_count
        if count:
            stats[c, 2] = np.min(lengths[:count])
            stats[c, 3] = np.median(lengths[:count])
            stats[c, 4] = np.max(lengths[:count])
            stats[c, 5] = observations / count
        if return_count:
            stats[c, 7] = returns / return_count
        summary[0] += observations
        summary[4] += count
    summary[1] = n - summary[0]
    while summary[2] < n and not 0 <= labels[summary[2]] < k:
        summary[2] += 1
    while summary[3] < n and not 0 <= labels[n - 1 - summary[3]] < k:
        summary[3] += 1
    for j in range(1, n):
        if 0 <= labels[j - 1] < k and 0 <= labels[j] < k and labels[j] != labels[j - 1]:
            summary[5] += 1
    return summary, stats


@njit(cache=True)
def horizon_profile_kernel(labels, dates, k):
    """Empirical persistence of completed state episodes on the real date axis.

    Head/tail or unknown-bounded episodes are censored and excluded from duration
    quantiles. Horizon is therefore measured evidence, never a user-supplied tag.
    """
    n = len(labels)
    if n > 20000 or len(dates) != n or not 2 <= k <= 12:
        raise ValueError("Invalid horizon profile dimensions")
    per = np.full((k, 8), np.nan)
    obs_lengths = np.empty(n, np.float64)
    day_lengths = np.empty(n, np.float64)
    classified = 0
    observations = np.zeros(k, np.int64)
    transitions = 0
    for i in range(n):
        if i and dates[i] <= dates[i - 1]:
            raise ValueError("Date axis must increase")
        if 0 <= labels[i] < k:
            classified += 1
            observations[labels[i]] += 1
        if i and 0 <= labels[i - 1] < k and 0 <= labels[i] < k and labels[i] != labels[i - 1]:
            transitions += 1
    for state in range(k):
        complete = 0
        start = 0
        while start < n:
            end = start + 1
            while end < n and labels[end] == labels[start]:
                end += 1
            if (labels[start] == state and start > 0 and end < n
                    and 0 <= labels[start - 1] < k and 0 <= labels[end] < k):
                obs_lengths[complete] = end - start
                # State holds on [observation, next observation); the next
                # classified episode supplies the right boundary at any frequency.
                day_lengths[complete] = dates[end] - dates[start]
                complete += 1
            start = end
        per[state, 0] = complete
        per[state, 7] = observations[state] / classified if classified else np.nan
        if complete:
            per[state, 1] = np.percentile(obs_lengths[:complete], 25.0)
            per[state, 2] = np.median(obs_lengths[:complete])
            per[state, 3] = np.percentile(obs_lengths[:complete], 75.0)
            per[state, 4] = np.percentile(day_lengths[:complete], 25.0)
            per[state, 5] = np.median(day_lengths[:complete])
            per[state, 6] = np.percentile(day_lengths[:complete], 75.0)
    summary = np.full(4, np.nan)
    summary[0] = transitions
    if n > 1 and dates[-1] > dates[0]:
        years = (dates[-1] - dates[0]) / 365.2425
        if years > 0:
            summary[1] = transitions / years
        summary[2] = dates[-1] - dates[0]
    if n:
        summary[3] = classified / n
    return per, summary


@njit(cache=True)
def compare_kernel(base, candidate, k):
    n = len(base)
    if len(candidate) != n or n > 20000 or not 2 <= k <= 12:
        raise ValueError("Invalid comparison dimensions")
    out = np.full(5, np.nan)
    paired = hits = classified = 0
    for i in range(n):
        if 0 <= candidate[i] < k:
            classified += 1
            if 0 <= base[i] < k:
                paired += 1
                hits += base[i] == candidate[i]
    out[0] = paired
    if paired:
        out[1] = hits / paired
    if n:
        out[2] = classified / n
    # Symmetric nearest event with the SAME ordered states, no label permutation.
    total_distance = 0.0
    matched_events = 0
    for direction in range(2):
        left = base if direction == 0 else candidate
        right = candidate if direction == 0 else base
        nearest = np.full(n, n + 1, np.int64)
        last = np.full((k, k), -1, np.int64)
        for i in range(1, n):
            a, b = right[i - 1], right[i]
            if 0 <= a < k and 0 <= b < k and a != b:
                last[a, b] = i
            a, b = left[i - 1], left[i]
            if 0 <= a < k and 0 <= b < k and a != b and last[a, b] >= 0:
                nearest[i] = i - last[a, b]
        last[:, :] = -1
        for i in range(n - 1, 0, -1):
            a, b = right[i - 1], right[i]
            if 0 <= a < k and 0 <= b < k and a != b:
                last[a, b] = i
            a, b = left[i - 1], left[i]
            if 0 <= a < k and 0 <= b < k and a != b:
                if last[a, b] >= 0:
                    nearest[i] = min(nearest[i], last[a, b] - i)
                if nearest[i] <= n:
                    total_distance += nearest[i]
                    matched_events += 1
    if matched_events:
        out[3] = total_distance / matched_events
    out[4] = matched_events
    return out


@njit(cache=True)
def evidence_kernel(pred, q):
    if len(pred) != len(q) or len(pred) > 20000 or not 2 <= q.shape[1] <= 12:
        raise ValueError("Invalid probability evidence dimensions")
    result = np.full((len(pred), 5), np.nan)
    for i in range(len(pred)):
        if not 0 <= pred[i] < q.shape[1]:
            continue
        total = 0.0
        top = second = entropy = 0.0
        valid = True
        for c in range(q.shape[1]):
            p = q[i, c]
            if not np.isfinite(p) or p < 0 or p > 1:
                valid = False
                break
            total += p
            if p >= top:
                second, top = top, p
            elif p > second:
                second = p
            if p > 0:
                entropy -= p * np.log(p)
        if valid and abs(total - 1.0) <= 1e-6:
            result[i, 0] = q[i, pred[i]]
            result[i, 1] = top
            result[i, 2] = second
            result[i, 3] = top - second
            result[i, 4] = entropy
    return result


@njit(cache=True)
def integer_step_kernel(value, changed):
    return value + 1.0 if changed == value else changed


KERNELS = (
    (integer_step_kernel, (types.float64, types.float64)),
    (quality_kernel, (I, V, types.int64)),
    (horizon_profile_kernel, (I, I, types.int64)),
    (compare_kernel, (I, I, types.int64)),
    (evidence_kernel, (I, F)),
)


def smoke():
    labels = np.empty(0, np.int64)
    prices = np.empty(0, np.float64)
    q = np.empty((0, 2), np.float64)
    for a in (labels, prices, q):
        a.flags.writeable = False
    integer_step_kernel(2.0, 2.0)
    quality_kernel(labels, prices, 2)
    horizon_profile_kernel(labels, labels, 2)
    compare_kernel(labels, labels, 2)
    evidence_kernel(labels, q)

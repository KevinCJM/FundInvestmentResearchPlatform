"""Fixed-signature selection and integer feasibility for portfolio exploration."""
import numpy as np
from numba import njit, types

F1 = types.Array(types.float64, 1, "A", readonly=True)
F2 = types.Array(types.float64, 2, "A", readonly=True)
U2 = types.Array(types.uint8, 2, "A", readonly=True)


@njit((F1, F2, U2, F1, F1, types.float64, types.int64), cache=False, nogil=True)
def integer_weights_kernel(raw, bounds, groups, group_lows, group_highs, step, budget):
    """Find integer-feasible weights: status 0 found, 1 infeasible, 2 budget, 3 input."""
    n = raw.size
    g = groups.shape[0]
    output = np.full(n, np.nan)
    if (n == 0 or bounds.shape != (n, 2) or groups.shape[1] != n
            or group_lows.size != g or group_highs.size != g or budget < 0
            or not np.isfinite(step) or step <= 0.0 or step > 1.0):
        return output, 3
    inverse = 1.0 / step
    if inverse > 1e9 + 1e-6 or abs(np.round(inverse) * step - 1.0) > 1e-12:
        return output, 3
    units = int(np.round(inverse))
    lows = np.empty(n, dtype=np.int64)
    highs = np.empty(n, dtype=np.int64)
    target = np.empty(n)
    total_raw = 0.0
    for i in range(n):
        if (not np.isfinite(raw[i]) or not np.isfinite(bounds[i, 0])
                or not np.isfinite(bounds[i, 1]) or bounds[i, 0] < 0.0
                or bounds[i, 1] > 1.0 or bounds[i, 0] > bounds[i, 1]):
            return output, 3
        lows[i] = int(np.ceil(bounds[i, 0] * units - 1e-9))
        highs[i] = int(np.floor(bounds[i, 1] * units + 1e-9))
        if lows[i] > highs[i]:
            return output, 1
        total_raw += max(raw[i], 0.0)
    gl = np.empty(g, dtype=np.int64)
    gh = np.empty(g, dtype=np.int64)
    for j in range(g):
        if (not np.isfinite(group_lows[j]) or not np.isfinite(group_highs[j])
                or group_lows[j] < 0.0 or group_highs[j] > 1.0
                or group_lows[j] > group_highs[j]):
            return output, 3
        gl[j] = int(np.ceil(group_lows[j] * units - 1e-9))
        gh[j] = int(np.floor(group_highs[j] * units + 1e-9))
        if gl[j] > gh[j]:
            return output, 1
    values = np.empty(n, dtype=np.int64)
    if np.sum(lows) > units or np.sum(highs) < units:
        return output, 1
    for i in range(n):
        target[i] = max(raw[i], 0.0) * units / total_raw if total_raw > 0.0 else units / n
    # Bounded-simplex shift avoids a loop proportional to the number of units.
    shift_lo, shift_hi = -float(units), float(units)
    for _ in range(80):
        shift = (shift_lo + shift_hi) / 2.0
        amount = 0.0
        for i in range(n):
            amount += min(max(target[i] + shift, lows[i]), highs[i])
        if amount < units:
            shift_lo = shift
        else:
            shift_hi = shift
    for i in range(n):
        values[i] = min(max(int(np.round(target[i] + (shift_lo + shift_hi) / 2.0)), lows[i]), highs[i])
    delta = units - np.sum(values)
    # Balanced rounding is a fast path, never a substitute for group validation.
    while delta != 0:
        direction = 1 if delta > 0 else -1
        best = -1
        best_cost = np.inf
        for i in range(n):
            if lows[i] <= values[i] + direction <= highs[i]:
                cost = 2.0 * direction * (values[i] - target[i]) + 1.0
                if cost < best_cost:
                    best, best_cost = i, cost
        if best < 0:
            return output, 1
        values[best] += direction
        delta -= direction
    valid = True
    for j in range(g):
        amount = 0
        for i in range(n):
            if groups[j, i]:
                amount += values[i]
        if amount < gl[j] or amount > gh[j]:
            valid = False
    if valid:
        return values.astype(np.float64) / units, 0

    # Suffix limits permit overlapping groups without assuming a partition.
    suffix_lo = np.zeros(n + 1, dtype=np.int64)
    suffix_hi = np.zeros(n + 1, dtype=np.int64)
    group_lo = np.zeros((g, n + 1), dtype=np.int64)
    group_hi = np.zeros((g, n + 1), dtype=np.int64)
    for i in range(n - 1, -1, -1):
        suffix_lo[i] = suffix_lo[i + 1] + lows[i]
        suffix_hi[i] = suffix_hi[i + 1] + highs[i]
        for j in range(g):
            group_lo[j, i] = group_lo[j, i + 1] + (lows[i] if groups[j, i] else 0)
            group_hi[j, i] = group_hi[j, i + 1] + (highs[i] if groups[j, i] else 0)
    lower = np.empty(n, dtype=np.int64)
    upper = np.empty(n, dtype=np.int64)
    left = np.empty(n, dtype=np.int64)
    right = np.empty(n, dtype=np.int64)
    group_sum = np.zeros(g, dtype=np.int64)
    depth, total, visits = 0, 0, 0
    entering = True
    while depth >= 0:
        if entering:
            low = max(lows[depth], units - total - suffix_hi[depth + 1])
            high = min(highs[depth], units - total - suffix_lo[depth + 1])
            for j in range(g):
                if groups[j, depth]:
                    low = max(low, gl[j] - group_sum[j] - group_hi[j, depth + 1])
                    high = min(high, gh[j] - group_sum[j] - group_lo[j, depth + 1])
                elif (group_sum[j] + group_hi[j, depth + 1] < gl[j]
                      or group_sum[j] + group_lo[j, depth + 1] > gh[j]):
                    high = low - 1
            lower[depth], upper[depth] = low, high
            center = min(max(int(np.round(target[depth])), low), high)
            left[depth], right[depth] = center, center + 1
            entering = False
        if left[depth] < lower[depth] and right[depth] > upper[depth]:
            depth -= 1
            if depth >= 0:
                total -= values[depth]
                for j in range(g):
                    if groups[j, depth]:
                        group_sum[j] -= values[depth]
            continue
        if visits >= budget:
            return output, 2
        visits += 1
        if (left[depth] >= lower[depth]
                and (right[depth] > upper[depth]
                     or abs(left[depth] - target[depth]) <= abs(right[depth] - target[depth]))):
            value = left[depth]
            left[depth] -= 1
        else:
            value = right[depth]
            right[depth] += 1
        values[depth] = value
        total += value
        for j in range(g):
            if groups[j, depth]:
                group_sum[j] += value
        if depth == n - 1:
            return values.astype(np.float64) / units, 0
        depth += 1
        entering = True
    return output, 1


@njit((F1, F1, types.int64, types.int64, types.int64), cache=False, nogil=True)
def return_bucket_indices_kernel(risks, returns, start, end, buckets):
    """Choose minimum risk in each return bucket; rightmost endpoint is included."""
    output = np.full(max(end - start, 0), -1, dtype=np.int64)
    if start < 0 or end < start or end > risks.size or returns.size != risks.size or buckets <= 0:
        raise ValueError("BUCKET_AXIS")
    if end <= start:
        return output, 0
    for i in range(start, end):
        if not np.isfinite(risks[i]) or not np.isfinite(returns[i]):
            raise ValueError("BUCKET_NONFINITE")
    lo, hi = np.min(returns[start:end]), np.max(returns[start:end])
    if hi <= lo:
        count = min(buckets, end - start)
        used = np.zeros(end - start, dtype=np.bool_)
        for slot in range(count):
            best = -1
            for i in range(start, end):
                if not used[i - start] and (best < 0 or risks[i] < risks[best]):
                    best = i
            output[slot] = best
            used[best - start] = True
        return output, count
    best_by_bucket = np.full(buckets, -1, dtype=np.int64)
    for i in range(start, end):
        b = min(buckets - 1, max(0, int((returns[i] - lo) / (hi - lo) * buckets)))
        best = best_by_bucket[b]
        if best < 0 or risks[i] < risks[best]:
            best_by_bucket[b] = i
    count = 0
    for best in best_by_bucket:
        if best >= 0:
            output[count] = best
            count += 1
    return output, count


SAMPLING_KERNELS = (integer_weights_kernel, return_bucket_indices_kernel)
for _kernel in SAMPLING_KERNELS:
    _kernel.disable_compile()

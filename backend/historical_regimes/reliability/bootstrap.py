"""Paired moving blocks on an intact holdout axis, conditional on frozen calibration."""

import numpy as np
from numba import njit, types

I = types.Array(types.int64, 1, "A", readonly=True)
F = types.Array(types.float64, 2, "A", readonly=True)
V = types.Array(types.float64, 1, "A", readonly=True)
SEED = 1729
NAMES = (
    "accuracy",
    "accepted_coverage",
    "accepted_error",
    "brier",
    "paired_brier_improvement",
)


@njit(cache=True)
def bootstrap_kernel(y, pred, q, base, floor, block_length, replicates, level):
    n, k = q.shape
    if (
        n != len(y)
        or n != len(pred)
        or len(base) != k
        or n > 20000
        or not 2 <= k <= 12
        or not 2 <= block_length <= 250
        or not 20 <= replicates <= 500
        or not 0.8 <= level <= 0.99
        or not 0 <= floor <= 1
    ):
        raise ValueError("Invalid bootstrap budget or dimensions")
    # Paired additive numerators/denominators; invalid rows occupy real time slots.
    prefix = np.zeros((n + 1, 10), np.float64)
    for i in range(n):
        prefix[i + 1] = prefix[i]
        if not 0 <= y[i] < k:
            continue
        known = 0 <= pred[i] < k
        prefix[i + 1, 0] += known and pred[i] == y[i]
        prefix[i + 1, 1] += 1
        prefix[i + 1, 3] += 1
        total = 0.0
        valid = known
        score = baseline_score = 0.0
        base_valid = True
        base_total = 0.0
        for c in range(k):
            p = q[i, c]
            if not np.isfinite(p) or p < 0 or p > 1:
                valid = False
            total += p
            target = 1.0 if y[i] == c else 0.0
            score += (p - target) ** 2
            b = base[c]
            if not np.isfinite(b) or b < 0 or b > 1:
                base_valid = False
            base_total += b
            baseline_score += (b - target) ** 2
        valid = valid and abs(total - 1.0) <= 1e-6
        base_valid = base_valid and abs(base_total - 1.0) <= 1e-6
        if valid:
            accepted = q[i, pred[i]] >= floor
            prefix[i + 1, 2] += accepted
            prefix[i + 1, 4] += accepted and pred[i] != y[i]
            prefix[i + 1, 5] += accepted
            prefix[i + 1, 6] += score
            prefix[i + 1, 7] += 1
            if base_valid:
                prefix[i + 1, 8] += baseline_score - score
                prefix[i + 1, 9] += 1
    # Full nonoverlapping observed blocks, never n_valid // block_length.
    full_blocks = np.zeros(5, np.int64)
    for start in range(0, n - block_length + 1, block_length):
        for metric in range(5):
            denominator = metric * 2 + 1
            if (
                prefix[start + block_length, denominator] - prefix[start, denominator]
                == block_length
            ):
                full_blocks[metric] += 1
    # Complete state cycles: successive complete contiguous segments collectively
    # visit every state. Missing labels reset the cycle; edge segments are censored.
    seen = np.zeros(k, np.uint8)
    cycles = 0
    start = 0
    while start < n:
        end = start + 1
        while end < n and y[end] == y[start]:
            end += 1
        state = y[start]
        complete = (
            0 <= state < k
            and start > 0
            and end < n
            and 0 <= y[start - 1] < k
            and 0 <= y[end] < k
        )
        if complete:
            seen[state] = 1
            if seen.sum() == k:
                cycles += 1
                seen[:] = 0
        elif not 0 <= state < k:
            seen[:] = 0
        start = end
    values = np.full((replicates, 5), np.nan)
    rng = SEED
    totals = np.zeros(10, np.float64)
    if n >= block_length:
        for r in range(replicates):
            totals[:] = 0
            filled = 0
            while filled < n:
                rng = (rng * 48271) % 2147483647
                begin = rng % (n - block_length + 1)
                length = min(block_length, n - filled)
                for j in range(10):
                    totals[j] += prefix[begin + length, j] - prefix[begin, j]
                filled += length
            for metric in range(5):
                if totals[2 * metric + 1] > 0:
                    values[r, metric] = totals[2 * metric] / totals[2 * metric + 1]
    # estimate, lower, upper, valid replicates, effective sample denominator.
    result = np.full((5, 5), np.nan)
    scratch = np.empty(replicates, np.float64)
    for metric in range(5):
        result[metric, 4] = prefix[n, 2 * metric + 1]
        if result[metric, 4] > 0:
            result[metric, 0] = prefix[n, 2 * metric] / result[metric, 4]
        count = 0
        for r in range(replicates):
            if np.isfinite(values[r, metric]):
                scratch[count] = values[r, metric]
                count += 1
        result[metric, 3] = count
        if count:
            ordered = np.sort(scratch[:count])
            for output, probability in (
                (1, (1.0 - level) / 2.0),
                (2, (1.0 + level) / 2.0),
            ):
                position = (count - 1) * probability
                lo = int(position)
                hi = min(lo + 1, count - 1)
                result[metric, output] = ordered[lo] + (position - lo) * (
                    ordered[hi] - ordered[lo]
                )
    return result, full_blocks, cycles


KERNELS = (
    (
        bootstrap_kernel,
        (I, I, F, V, types.float64, types.int64, types.int64, types.float64),
    ),
)


def smoke():
    y = np.empty(0, np.int64)
    q = np.empty((0, 2), np.float64)
    base = np.array([0.5, 0.5])
    for a in (y, q, base):
        a.flags.writeable = False
    bootstrap_kernel(y, y, q, base, 0.6, 10, 20, 0.95)


def build_intervals(y, pred, q, base, policy, floor, scope):
    from .report import finite
    from .kernels import audit

    audit()
    result = {
        "status": "disabled",
        "reason": "disabled_by_policy",
        "method": "paired_moving_block",
        "scope": scope,
        "conditional_on": "fixed_model_reference_and_calibrator",
        "confidence_level": policy.confidence_level,
        "block_length": policy.block_length,
        "replicates": policy.replicates,
        "seed": SEED,
        "samples": len(y),
        "full_blocks": 0,
        "complete_cycles": 0,
        "metrics": {},
    }
    if not policy.enabled:
        return result
    numbers, blocks, cycles = bootstrap_kernel(
        y,
        pred,
        q,
        base,
        floor,
        policy.block_length,
        policy.replicates,
        policy.confidence_level,
    )
    result.update(full_blocks=int(blocks[0]), complete_cycles=int(cycles))
    available = 0
    for i, name in enumerate(NAMES):
        reason = (
            "insufficient_full_blocks"
            if blocks[i] < policy.minimum_blocks
            else (
                "insufficient_complete_cycles"
                if cycles < policy.minimum_cycles
                else (
                    "insufficient_valid_replicates"
                    if numbers[i, 3] < policy.minimum_valid_replicates
                    else None
                )
            )
        )
        available += reason is None
        result["metrics"][name] = {
            "estimate": finite(numbers[i, 0]),
            "lower": None if reason else finite(numbers[i, 1]),
            "upper": None if reason else finite(numbers[i, 2]),
            "valid_replicates": int(numbers[i, 3]),
            "samples": int(numbers[i, 4]),
            "full_blocks": int(blocks[i]),
            "reason": reason,
            "unit": "fraction" if i < 3 else "brier_score",
        }
    result["status"] = (
        "available" if available == 5 else "partial" if available else "unavailable"
    )
    result["reason"] = (
        None if available == 5 else "insufficient_dependency_aware_evidence"
    )
    return result

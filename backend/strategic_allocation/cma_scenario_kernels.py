"""Shared empirical scenario statistics and Markov reward calculations.

Only the empirical distribution regularizer, probability recursion and reward
recursion are coupled. They expose their intermediate state moments and paths.
All inputs use one readonly, arbitrary-stride ABI; no per-state return panels.
"""
from __future__ import annotations

import hashlib
import inspect
import os

import numpy as np
from numba import njit, float64, int64, types

from .cma_model_kernels import mixture_moments_kernel
from .cma_statistical_kernels import (conditional_state_moments, occupancy_probabilities,
                                      regime_transition_diagnostics_kernel)

V = types.Array(float64, 1, "A", readonly=True)
M = types.Array(float64, 2, "A", readonly=True)
D = types.Array(float64, 3, "A", readonly=True)
I = types.Array(int64, 1, "A", readonly=True)
VERSION = "scenario-cma/1.0.2"
_WARMED_PID = None
_WARMED_FINGERPRINT = None


@njit((M, int64, int64, V), cache=True, nogil=True)
def scaled_distance(returns, left, right, scale):
    total = 0.0
    for a in range(returns.shape[1]):
        if scale[a] > 0:
            delta = (returns[left, a] - returns[right, a]) / scale[a]
            total += delta * delta
    return np.sqrt(total)


@njit((M, I, int64), cache=True, nogil=True)
def empirical_regularization_weights(returns, states, state_count):
    """Chronological energy-score fitting of one joint mixture weight per state.

    Three expanding folds, at most 64 evenly spaced training atoms per group
    and 128 validation observations per state/fold bound the distance workload.
    This tunes retrospective conditional distributions, not a forecast skill
    certificate: reference-label model selection uncertainty is not covered.
    """
    rows, assets = returns.shape
    if not 20 <= rows <= 10000 or not 1 <= assets <= 30:
        raise ValueError("LTCMA_STATE_SAMPLE")
    conditional_state_moments(returns, states, state_count, 0.0)  # shared boundary
    linear = np.zeros(state_count)
    quadratic = np.zeros(state_count)
    support = np.zeros(state_count, np.int64)
    indices = np.empty((state_count + 1, rows), np.int64)
    for fold in range(3):
        end = rows * (fold + 2) // 5
        stop = rows * (fold + 3) // 5
        counts = np.zeros(state_count + 1, np.int64)
        mean = np.zeros(assets)
        scatter = np.zeros(assets)
        for t in range(end):
            s = states[t]
            if s < 0:
                continue
            indices[s, counts[s]] = t
            indices[state_count, counts[state_count]] = t
            counts[s] += 1
            counts[state_count] += 1
            for a in range(assets):
                delta = returns[t, a] - mean[a]
                mean[a] += delta / counts[state_count]
                scatter[a] += delta * (returns[t, a] - mean[a])
        if counts[state_count] < 20:
            continue
        scale = np.sqrt(scatter / counts[state_count])
        pool_n = min(64, counts[state_count])
        pool = np.empty(pool_n, np.int64)
        for i in range(pool_n):
            pool[i] = indices[state_count, i * counts[state_count] // pool_n]
        pool_pair = 0.0
        for i in range(pool_n):
            for j in range(pool_n):
                pool_pair += scaled_distance(returns, pool[i], pool[j], scale) / (pool_n * pool_n)
        for s in range(state_count):
            if counts[s] < 10:
                continue
            own_n = min(64, counts[s])
            own = np.empty(own_n, np.int64)
            for i in range(own_n):
                own[i] = indices[s, i * counts[s] // own_n]
            own_pair, cross_pair = 0.0, 0.0
            for i in range(own_n):
                for j in range(own_n):
                    own_pair += scaled_distance(returns, own[i], own[j], scale) / (own_n * own_n)
                for j in range(pool_n):
                    cross_pair += scaled_distance(returns, own[i], pool[j], scale) / (own_n * pool_n)
            valid_n = 0
            for t in range(end, stop):
                if states[t] == s:
                    indices[s, valid_n] = t
                    valid_n += 1
            used_n = min(128, valid_n)
            for v in range(used_n):
                t = indices[s, v * valid_n // used_n]
                own_y, pool_y = 0.0, 0.0
                for i in range(own_n):
                    own_y += scaled_distance(returns, own[i], t, scale) / own_n
                for i in range(pool_n):
                    pool_y += scaled_distance(returns, pool[i], t, scale) / pool_n
                linear[s] += pool_y - own_y + own_pair - cross_pair
                quadratic[s] += max(0.0, 2.0 * cross_pair - own_pair - pool_pair)
                support[s] += 1
    weights = np.ones(state_count)
    for s in range(state_count):
        if support[s] >= 10 and quadratic[s] > 1e-12:
            weights[s] = min(1.0, max(0.0, -linear[s] / quadratic[s]))
    return weights, support


@njit((V, M, D), cache=True, nogil=True)
def scenario_mixture_moments(probabilities, means, covariance):
    """Shared total-covariance calculation with exact deterministic coordinates."""
    mixed_mean, mixed_covariance, within, between = mixture_moments_kernel(probabilities, means, covariance, False)
    states, assets = means.shape
    for a in range(assets):
        deterministic = True
        first_value = 0.0
        found = False
        for s in range(states):
            if probabilities[s] > 0:
                if not found:
                    first_value, found = means[s, a], True
                if covariance[s, a, a] != 0.0 or means[s, a] != first_value:
                    deterministic = False
        if deterministic and found:
            mixed_mean[a] = first_value
            mixed_covariance[a, :] = 0.0
            mixed_covariance[:, a] = 0.0
            within[a, :], within[:, a] = 0.0, 0.0
            between[a, :], between[:, a] = 0.0, 0.0
    return mixed_mean, mixed_covariance, within, between


@njit((I, M, D, V), cache=True, nogil=True)
def regularized_state_moments(counts, means, covariance, weights):
    """Exact moments of (1-lambda) F_state + lambda F_pooled, including mean risk."""
    states, assets = means.shape
    if weights.size != states or counts.size != states or covariance.shape != (states, assets, assets):
        raise ValueError("LTCMA_SCENARIO_AXIS")
    probabilities = occupancy_probabilities(counts)
    pooled_mean, pooled_covariance, _, _ = scenario_mixture_moments(probabilities, means, covariance)
    result_mean, result_covariance = means.copy(), covariance.copy()
    for s in range(states):
        weight = weights[s]
        if not np.isfinite(weight) or not 0 <= weight <= 1:
            raise ValueError("LTCMA_SCENARIO_REGULARIZATION")
        if counts[s] == 0:
            continue  # Unobserved states never acquire invented distributions.
        for a in range(assets):
            result_mean[s, a] = (1.0 - weight) * means[s, a] + weight * pooled_mean[a]
            for b in range(assets):
                result_covariance[s, a, b] = ((1.0 - weight) * covariance[s, a, b]
                    + weight * pooled_covariance[a, b]
                    + weight * (1.0 - weight) * (means[s, a] - pooled_mean[a])
                    * (means[s, b] - pooled_mean[b]))
            if pooled_covariance[a, a] == 0.0:
                result_mean[s, a] = pooled_mean[a]
                result_covariance[s, a, :] = 0.0
                result_covariance[s, :, a] = 0.0
    return result_mean, result_covariance, pooled_mean, pooled_covariance


@njit((I, int64, I), cache=True, nogil=True)
def complete_state_episodes(states, state_count, contiguous):
    if (contiguous.size != states.size or (contiguous < 0).any() or (contiguous > 1).any()
            or not 1 <= state_count <= 60 or (states < -1).any() or (states >= state_count).any()):
        raise ValueError("LTCMA_STATE_AXIS")
    counts = np.zeros(state_count, np.int64)
    t = 0
    while t < states.size:
        end = t + 1
        while end < states.size and contiguous[end] and states[end] == states[t]:
            end += 1
        if (0 <= states[t] < state_count and t > 0 and end < states.size
                and contiguous[t] and contiguous[end]
                and states[t - 1] >= 0 and states[end] >= 0
                and states[t - 1] != states[t] and states[end] != states[t]):
            counts[states[t]] += 1
        t = end
    return counts


@njit((I, int64, I), cache=True, nogil=True)
def estimated_transition_matrix(states, state_count, contiguous):
    """Training-only choice of Dirichlet strength using next-state log loss.

    The training occupancy is the shrinkage target. Unknown labels split
    transitions. No absent state or missing outgoing row is manufactured.
    These historical folds tune a research model; they are not PIT validation.
    """
    if contiguous.size != states.size:
        raise ValueError("LTCMA_STATE_AXIS")
    candidates = np.array([0.0, .5, 1.0, 2.0, 5.0, 10.0])
    losses = np.zeros(candidates.size)
    pairs = 0
    for fold in range(3):
        end = states.size * (fold + 2) // 5
        stop = states.size * (fold + 3) // 5
        counts, _, _, _, _ = regime_transition_diagnostics_kernel(states[:end], state_count, contiguous[:end])
        base = np.zeros(state_count)
        for t in range(end):
            if states[t] >= 0:
                base[states[t]] += 1.0
        if base.sum() == 0:
            continue
        base /= base.sum()
        for t in range(max(1, end), stop):
            left, right = states[t - 1], states[t]
            if left < 0 or right < 0 or not contiguous[t]:
                continue
            total = counts[left].sum()
            if total == 0:
                continue
            pairs += 1
            for c in range(candidates.size):
                alpha = candidates[c]
                p = (counts[left, right] + alpha * base[right]) / (total + alpha)
                losses[c] -= np.log(max(1e-12, p))
    best = int(np.argmin(losses)) if pairs else 0
    alpha = candidates[best]
    counts, transition, _, _, _ = regime_transition_diagnostics_kernel(states, state_count, contiguous)
    base = np.zeros(state_count)
    for s in states:
        if s >= 0:
            base[s] += 1.0
    if base.sum() == 0:
        raise ValueError("LTCMA_FORECAST_NO_STATES")
    base /= base.sum()
    for i in range(state_count):
        total = counts[i].sum()
        if total == 0:
            raise ValueError("LTCMA_FORECAST_MISSING_TRANSITIONS")
        for j in range(state_count):
            transition[i, j] = (counts[i, j] + alpha * base[j]) / (total + alpha)
    return transition, alpha, pairs


@njit((V, M, int64), cache=True, nogil=True)
def markov_probability_path(initial, transition, horizon):
    """Forecast occupancy at each future period; unknown transitions fail closed."""
    count = initial.size
    if not 1 <= count <= 60 or transition.shape != (count, count) or not 1 <= horizon <= 2520:
        raise ValueError("LTCMA_FORECAST_AXIS")
    if not np.isfinite(initial).all() or (initial < 0).any() or abs(initial.sum() - 1.0) > 1e-8:
        raise ValueError("LTCMA_FORECAST_PROBABILITY")
    for s in range(count):
        if (not np.isfinite(transition[s]).all() or (transition[s] < 0).any()
                or abs(transition[s].sum() - 1.0) > 1e-8):
            raise ValueError("LTCMA_FORECAST_TRANSITION")
    path = np.empty((horizon, count))
    previous = initial.copy()
    average = np.zeros(count)
    for h in range(horizon):
        for j in range(count):
            value = 0.0
            for i in range(count):
                value += previous[i] * transition[i, j]
            path[h, j] = value
            average[j] += value / horizon
        previous[:] = path[h]
    return path, average


@njit((I, I, I, int64, int64, I), cache=True, nogil=True)
def transition_holdout_scores(states, dates, label_available_dates, state_count, horizon, contiguous):
    """Disjoint horizon endpoints after a frozen chronological training segment.

    The first row is retrospective, using an oracle historical start state.
    The second also enforces historical-label availability at each decision.
    Neither substitutes for evaluating a historical realtime probability stream.
    """
    n = states.size
    if dates.size != n or label_available_dates.size != n or contiguous.size != n or not 1 <= horizon <= 2520:
        raise ValueError("LTCMA_FORECAST_VALIDATION_AXIS")
    cutoff = n * 3 // 5
    scores = np.full((2, 7), np.nan)
    scores[:, 0] = 0.0
    for mode in range(2):
        training = states[:cutoff].copy()
        if mode == 1:
            for t in range(cutoff):
                if label_available_dates[t] == np.iinfo(np.int64).min or label_available_dates[t] > dates[cutoff - 1]:
                    training[t] = -1
        base = np.zeros(state_count)
        for s in training:
            if s >= 0:
                base[s] += 1.0
        if base.sum() < 20 or (base == 0).any():
            continue
        base /= base.sum()
        # Selecting transition strength uses only this training partition.
        try:
            transition, _, _ = estimated_transition_matrix(training, state_count, contiguous[:cutoff])
        except Exception:
            continue
        totals = np.zeros(6)
        tested = 0
        # No endpoint overlaps the fitting partition. Subsequent endpoints are
        # horizon-spaced so the comparison does not count overlapping windows.
        for origin in range(cutoff, n - horizon, horizon):
            current, actual = states[origin], states[origin + horizon]
            if current < 0 or actual < 0:
                continue
            if not np.all(contiguous[origin + 1:origin + horizon + 1]):
                continue  # H retained rows across a gap are not H daily steps.
            if mode == 1 and (label_available_dates[origin] == np.iinfo(np.int64).min
                              or label_available_dates[origin] > dates[origin]):
                continue
            q = np.zeros(state_count)
            q[current] = 1.0
            path, _ = markov_probability_path(q, transition, horizon)
            forecast = path[-1]
            persistence = q
            for method in range(3):
                probabilities = forecast if method == 0 else (base if method == 1 else persistence)
                for s in range(state_count):
                    totals[method] += (probabilities[s] - (1.0 if s == actual else 0.0)) ** 2
                totals[method + 3] -= np.log(max(1e-12, probabilities[actual]))
            tested += 1
        scores[mode, 0] = tested
        if tested:
            scores[mode, 1:] = totals / tested
    return scores, cutoff


@njit((V, M, M, D, int64), cache=True, nogil=True)
def markov_compound_moments(initial, transition, means, covariance, horizon):
    """Exact joint compounded moments under conditionally independent empirical draws.

    State-indexed first and second wealth moments retain temporal dependence
    caused by the chain. They are not H times a one-period covariance.
    """
    path, average = markov_probability_path(initial, transition, horizon)
    states, assets = means.shape
    if states != initial.size or covariance.shape != (states, assets, assets):
        raise ValueError("LTCMA_FORECAST_RETURN_AXIS")
    first = np.empty((states, assets))
    second = np.empty((states, assets, assets))
    for s in range(states):
        first[s] = initial[s]
        second[s] = initial[s]
    for h in range(horizon):
        next_first = np.zeros((states, assets))
        next_second = np.zeros((states, assets, assets))
        for j in range(states):
            for i in range(states):
                for a in range(assets):
                    next_first[j, a] += transition[i, j] * first[i, a] * (1.0 + means[j, a])
                    for b in range(assets):
                        cross = covariance[j, a, b] + (1.0 + means[j, a]) * (1.0 + means[j, b])
                        next_second[j, a, b] += transition[i, j] * second[i, a, b] * cross
        first, second = next_first, next_second
    expected_wealth = first.sum(axis=0)
    risk = second.sum(axis=0)
    for a in range(assets):
        for b in range(assets):
            risk[a, b] -= expected_wealth[a] * expected_wealth[b]
        if risk[a, a] < 0 and risk[a, a] > -1e-10:
            risk[a, a] = 0.0
    for a in range(assets):
        deterministic = True
        for s in range(states):
            if covariance[s, a, a] != 0.0 or means[s, a] != means[0, a]:
                deterministic = False
        if deterministic:
            expected_wealth[a] = (1.0 + means[0, a]) ** horizon
            risk[a, :] = 0.0
            risk[:, a] = 0.0
    if not np.isfinite(expected_wealth).all() or not np.isfinite(risk).all():
        raise ValueError("LTCMA_FORECAST_NONFINITE")
    return expected_wealth - 1.0, risk, path, average


@njit((V, float64), cache=True, nogil=True)
def categorical_index(probabilities, draw):
    cumulative = 0.0
    for i in range(probabilities.size):
        cumulative += probabilities[i]
        if draw < cumulative:
            return i
    return probabilities.size - 1


@njit((M, I, V, V, M, int64, int64, int64), cache=True, nogil=True)
def simulate_joint_horizon(returns, states, weights, initial, transition, horizon, paths, seed):
    """One sampled row supplies every asset; no independent marginal resampling."""
    rows, assets = returns.shape
    state_count = initial.size
    markov_probability_path(initial, transition, horizon)
    if (not 1 <= paths <= 16384 or weights.size != state_count or states.size != rows
            or not np.isfinite(weights).all() or (weights < 0).any() or (weights > 1).any()):
        raise ValueError("LTCMA_SIMULATION_BUDGET")
    counts = np.zeros(state_count + 1, np.int64)
    indices = np.empty((state_count + 1, rows), np.int64)
    for t in range(rows):
        s = states[t]
        if s < -1:
            raise ValueError("LTCMA_STATE_CODE")
        if s >= 0:
            if s >= state_count:
                raise ValueError("LTCMA_STATE_CODE")
            indices[s, counts[s]] = t
            for a in range(assets):
                if not np.isfinite(returns[t, a]) or returns[t, a] <= -1.0:
                    raise ValueError("LTCMA_STATE_RETURN")
            indices[state_count, counts[state_count]] = t
            counts[s] += 1
            counts[state_count] += 1
    if (counts[:state_count] < 2).any():
        raise ValueError("LTCMA_FORECAST_UNOBSERVED_STATE")
    np.random.seed(seed)
    result = np.zeros((paths, assets))
    for p in range(paths):
        state = categorical_index(initial, np.random.random())
        wealth = np.ones(assets)
        for h in range(horizon):
            state = categorical_index(transition[state], np.random.random())
            group = state_count if np.random.random() < weights[state] else state
            row = indices[group, np.random.randint(counts[group])]
            for a in range(assets):
                wealth[a] *= 1.0 + returns[row, a]
        result[p] = wealth - 1.0
    if not np.isfinite(result).all():
        raise ValueError("LTCMA_FORECAST_NONFINITE")
    return result


@njit((M,), cache=True, nogil=True)
def horizon_sample_summary(samples):
    paths, assets = samples.shape
    quantiles = np.empty((3, assets))
    loss = np.zeros(assets)
    mean_se = np.zeros(assets)
    loss_se = np.zeros(assets)
    quantile_bounds = np.empty((2, 3, assets))
    if paths < 2 or not np.isfinite(samples).all():
        raise ValueError("LTCMA_SIMULATION_SAMPLE")
    for a in range(assets):
        column = samples[:, a]
        quantiles[0, a] = np.percentile(column, 5.0)
        quantiles[1, a] = np.percentile(column, 50.0)
        quantiles[2, a] = np.percentile(column, 95.0)
        loss[a] = np.sum(column < 0.0) / paths
        mean_se[a] = np.std(column) / np.sqrt(paths)
        loss_se[a] = np.sqrt(loss[a] * (1.0 - loss[a]) / paths)
        for j in range(3):
            probability = .05 if j == 0 else (.5 if j == 1 else .95)
            error = 1.96 * np.sqrt(probability * (1.0 - probability) / paths)
            quantile_bounds[0, j, a] = np.percentile(column, 100.0 * max(0.0, probability - error))
            quantile_bounds[1, j, a] = np.percentile(column, 100.0 * min(1.0, probability + error))
    return quantiles, loss, mean_se, loss_se, quantile_bounds


@njit((I, int64), cache=True, nogil=True)
def contiguous_block_ranges(contiguous, max_length):
    """Partition retained intervals without crossing a gap or omitting short tails."""
    rows = contiguous.size
    if (not 1 <= max_length <= rows <= 10000
            or (contiguous < 0).any() or (contiguous > 1).any()):
        raise ValueError("LTCMA_STATE_AXIS")
    ranges = np.empty((rows, 2), np.int64)
    start, count = 0, 0
    for end in range(1, rows + 1):
        if end == rows or not contiguous[end] or end - start == max_length:
            ranges[count, 0], ranges[count, 1] = start, end
            count += 1
            start = end
    return ranges[:count]


@njit((M, I, V, int64, I), cache=True, nogil=True)
def block_mean_uncertainty(returns, states, weights, seed, contiguous):
    """Joint block bootstrap; fixed definition and fitted regularizer.

    Automatic block length is n^(1/3) inflated by positive lag-one dependence,
    bounded by sqrt(n). It is a declared estimation heuristic, not optimality.
    Continuous samples retain circular blocks. Gapped samples resample their
    disjoint contiguous blocks, including short tails, with replacement. Each
    draw has the same block count and normalizes by its actual known-row count.
    """
    rows, assets = returns.shape
    if contiguous.size != rows or (contiguous < 0).any() or (contiguous > 1).any():
        raise ValueError("LTCMA_STATE_AXIS")
    state_count = weights.size
    counts, means, covariance = conditional_state_moments(returns, states, state_count, 0.0)
    probabilities = occupancy_probabilities(counts)
    pooled, pooled_cov, _, _ = scenario_mixture_moments(probabilities, means, covariance)
    persistence = 0.0
    for a in range(assets):
        numerator, denominator = 0.0, 0.0
        for t in range(1, rows):
            if contiguous[t] and states[t] >= 0 and states[t - 1] >= 0:
                numerator += (returns[t, a] - pooled[a]) * (returns[t - 1, a] - pooled[a])
                denominator += (returns[t - 1, a] - pooled[a]) ** 2
        if denominator > 0:
            persistence = max(persistence, min(.95, max(0.0, numerator / denominator)))
    block_length = max(1, min(int(np.sqrt(rows)), int(np.ceil(rows ** (1.0 / 3.0) * (1.0 + persistence) / (1.0 - persistence)))))
    gapped = (contiguous[1:] == 0).any()
    ranges = contiguous_block_ranges(contiguous, block_length)
    max_sampled_length = int(np.max(ranges[:, 1] - ranges[:, 0])) if gapped else block_length
    block_count = ranges.shape[0] if gapped else (rows + block_length - 1) // block_length
    multiplicities = np.zeros(ranges.shape[0], np.int64)
    draws = 256
    simulations = np.empty((draws, assets))
    np.random.seed(seed)
    for b in range(draws):
        sums = np.zeros((state_count, assets))
        occurrences = np.zeros(state_count, np.int64)
        if gapped:
            multiplicities[:] = 0
            for block in range(block_count):
                multiplicities[np.random.randint(block_count)] += 1
        for block in range(block_count):
            if gapped:
                copies = multiplicities[block]
                if copies == 0:
                    continue
                start, stop = ranges[block]
                length = stop - start
            else:
                copies = 1
                start = np.random.randint(rows)
                length = min(block_length, rows - block * block_length)
            for offset in range(length):
                t = (start + offset) % rows
                s = states[t]
                if s >= 0:
                    occurrences[s] += copies
                    for a in range(assets):
                        sums[s, a] += copies * returns[t, a]
        total = occurrences.sum()
        if total < 2:
            raise ValueError("LTCMA_SCENARIO_BOOTSTRAP_SAMPLE")
        pooled_weight = 0.0
        for s in range(state_count):
            pooled_weight += occurrences[s] * weights[s] / total
        for a in range(assets):
            value = 0.0
            for s in range(state_count):
                value += ((1.0 - weights[s]) + pooled_weight) * sums[s, a] / total
            simulations[b, a] = value * 252.0
    center = np.zeros(assets)
    original = np.zeros(assets)
    for s in range(state_count):
        original += probabilities[s] * ((1.0 - weights[s]) * means[s] + weights[s] * pooled) * 252.0
    for b in range(draws):
        center += simulations[b] / draws
    result_covariance = np.zeros((assets, assets))
    half_width = np.zeros(assets)
    for a in range(assets):
        if pooled_cov[a, a] == 0:
            continue
        half_width[a] = max(abs(np.percentile(simulations[:, a], 97.5) - original[a]),
                            abs(np.percentile(simulations[:, a], 2.5) - original[a]))
        for c in range(assets):
            if pooled_cov[c, c] > 0:
                for b in range(draws):
                    result_covariance[a, c] += (simulations[b, a] - center[a]) * (simulations[b, c] - center[c]) / (draws - 1)
    return result_covariance, half_width, block_length, max_sampled_length, block_count


KERNELS = (scaled_distance, empirical_regularization_weights, scenario_mixture_moments, regularized_state_moments,
           complete_state_episodes, estimated_transition_matrix, markov_probability_path, transition_holdout_scores, markov_compound_moments,
           categorical_index, simulate_joint_horizon, horizon_sample_summary, contiguous_block_ranges, block_mean_uncertainty)
for kernel in KERNELS:
    kernel.disable_compile()


def execution_audit():
    dispatchers = (*KERNELS, conditional_state_moments, occupancy_probabilities, mixture_moments_kernel)
    fingerprint = hashlib.sha256(repr([(inspect.getsource(k.py_func), [str(s) for s in k.signatures])
                                      for k in dispatchers]).encode()).hexdigest()
    compiled = all(len(k.signatures) == len(k.nopython_signatures) == 1 and not k._can_compile
                   and not any(v.objectmode for v in k.overloads.values()) for k in dispatchers)
    ready = bool(compiled and _WARMED_PID == os.getpid() and fingerprint == _WARMED_FINGERPRINT)
    return {"backend": "numba_njit_fixed_signature", "kernel_version": VERSION,
            "fingerprint": fingerprint, "complete": ready, "fully_warmed": ready,
            "nopython": bool(compiled), "object_mode": 0, "python_fallback": 0,
            "request_time_compilation": 0,
            "kernel_signatures": {k.__name__: [str(s) for s in k.signatures] for k in dispatchers}}


def require_ready():
    if not execution_audit()["complete"]:
        raise RuntimeError("LTCMA_SCENARIO_NOT_READY: 情景计算尚未完成本进程预热。")


def warm():
    global _WARMED_PID, _WARMED_FINGERPRINT
    _WARMED_PID = None
    raw = np.column_stack((np.sin(np.arange(80)) * .01, np.full(80, .0001)))
    returns = raw[::-1]
    states = (np.arange(80, dtype=np.int64) // 8 % 2)[::-1]
    returns.flags.writeable = states.flags.writeable = False
    weights, _ = empirical_regularization_weights(returns, states, 2)
    counts, means, covariance = conditional_state_moments(returns, states, 2, 0.0)
    means, covariance, _, _ = regularized_state_moments(counts, means, covariance, weights)
    contiguous = np.ones(160, dtype=np.int64)[::2]
    contiguous[10] = contiguous[63] = False
    contiguous.flags.writeable = False
    complete_state_episodes(states, 2, contiguous)
    estimated_transition_matrix(states, 2, contiguous)
    initial = np.array([.4, .6])
    transition = np.array([[.9, .1], [.2, .8]])
    initial.flags.writeable = transition.flags.writeable = False
    transition_holdout_scores(states, np.arange(80, dtype=np.int64), np.arange(80, dtype=np.int64), 2, 3, contiguous)
    markov_compound_moments(initial, transition, means, covariance, 3)
    sample = simulate_joint_horizon(returns, states, weights, initial, transition, 3, 32, 123)
    horizon_sample_summary(sample)
    block_mean_uncertainty(returns, states, weights, 123, contiguous)
    block_mean_uncertainty(returns, states, weights, 123, np.ones(80, np.int64))
    block_mean_uncertainty(returns, states, weights, 123, np.zeros(80, np.int64))
    _WARMED_PID = os.getpid()
    _WARMED_FINGERPRINT = execution_audit()["fingerprint"]
    return execution_audit()

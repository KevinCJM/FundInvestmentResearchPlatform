"""PS-style retrospective dating with explicit, deterministic censoring rules.

Durations count input observations, not calendar days. Missing/nonpositive prices
split the sample. Only intervals bounded by two retained turns receive a state.
"""
import numpy as np
from numba import float64, int64, njit, types

F = float64[::1]
I = int64[::1]
RESULT = types.Tuple((I, F, I, I, F, F))
TIMING = types.Tuple((I, I, int64))
SIDEWAYS = types.Tuple((I, F, F, I, I, I))


@njit(int64(I, I, int64, int64), cache=True)
def _remove_turn(positions, kinds, count, index):
    for j in range(index, count - 1):
        positions[j], kinds[j] = positions[j + 1], kinds[j + 1]
    return count - 1


@njit(int64(F, I, I, int64), cache=True)
def _alternate_turns(prices, positions, kinds, count):
    kept = 0
    for j in range(count):
        position, kind = positions[j], kinds[j]
        if kept and kind == kinds[kept - 1]:
            if kind * (prices[position] - prices[positions[kept - 1]]) > 0:
                positions[kept - 1] = position
        elif kept == 0 or kind * (prices[position] - prices[positions[kept - 1]]) > 0:
            positions[kept], kinds[kept] = position, kind
            kept += 1
    return kept


@njit(RESULT(F, int64, int64, int64, int64, int64, int64, float64), cache=True)
def peak_trough_asymmetric_kernel(prices, left_window, right_window, min_phase, min_cycle,
                                 head_window, tail_window, amplitude_exception):
    n = prices.size
    states = np.full(n, -1, dtype=np.int64)
    pivots = np.zeros(n)
    phase_start = np.full(n, -1, dtype=np.int64)
    phase_end = np.full(n, -1, dtype=np.int64)
    phase_return = np.full(n, np.nan)
    boundary_line = np.full(n, np.nan)
    positions, kinds = np.empty(n, dtype=np.int64), np.empty(n, dtype=np.int64)
    start = 0
    while start < n:
        if not np.isfinite(prices[start]) or prices[start] <= 0:
            pivots[start] = np.nan
            start += 1
            continue
        stop = start + 1
        while stop < n and np.isfinite(prices[stop]) and prices[stop] > 0:
            stop += 1
        count = 0
        for t in range(start + left_window, stop - right_window):
            if t - start < head_window or stop - 1 - t < tail_window:
                continue
            peak, trough = True, True
            for j in range(t - left_window, t):
                # Earliest member of an equal-price plateau wins.
                peak = peak and prices[t] > prices[j]
                trough = trough and prices[t] < prices[j]
            for j in range(t + 1, t + right_window + 1):
                peak = peak and prices[t] >= prices[j]
                trough = trough and prices[t] <= prices[j]
            if peak or trough:
                positions[count], kinds[count] = t, 1 if peak else -1
                count += 1
        count = _alternate_turns(prices, positions, kinds, count)
        # Every changed pass removes at least one turn, so this loop is bounded.
        changed = True
        while changed and count:
            changed = False
            first, last = positions[0], positions[count - 1]
            for j in range(start, first):
                if kinds[0] * (prices[j] - prices[first]) > 0:
                    count = _remove_turn(positions, kinds, count, 0)
                    changed = True
                    break
            if changed:
                continue
            for j in range(last + 1, stop):
                if kinds[count - 1] * (prices[j] - prices[last]) > 0:
                    count = _remove_turn(positions, kinds, count, count - 1)
                    changed = True
                    break
            if changed:
                continue
            # Short full cycle: preserve the more extreme same-type endpoint,
            # removing its weaker counterpart and the intervening opposite turn.
            for j in range(count - 2):
                if positions[j + 2] - positions[j] < min_cycle:
                    later_stronger = kinds[j] * (prices[positions[j + 2]] - prices[positions[j]]) > 0
                    remove_at = j if later_stronger else j + 1
                    count = _remove_turn(positions, kinds, count, remove_at)
                    count = _remove_turn(positions, kinds, count, remove_at)
                    changed = True
                    break
            if changed:
                count = _alternate_turns(prices, positions, kinds, count)
                continue
            for j in range(count - 1):
                left, right = positions[j], positions[j + 1]
                movement = abs(prices[right] / prices[left] - 1.0)
                if right - left >= min_phase or movement > amplitude_exception:
                    continue
                # At a sample edge discard the edge turn. Internally discard
                # the endpoint with the weaker adjacent log-price excursion,
                # then re-enforce alternation and all duration constraints.
                remove_at = j
                if j + 1 == count - 1:
                    remove_at = j + 1
                elif j > 0:
                    before = abs(np.log(prices[left]) - np.log(prices[positions[j - 1]]))
                    after = abs(np.log(prices[positions[j + 2]]) - np.log(prices[right]))
                    if before >= after:
                        remove_at = j + 1
                count = _remove_turn(positions, kinds, count, remove_at)
                count = _alternate_turns(prices, positions, kinds, count)
                changed = True
                break
        for j in range(count):
            pivots[positions[j]] = kinds[j]
        for j in range(count - 1):
            left, right = positions[j], positions[j + 1]
            code = 0 if kinds[j] == -1 else 1
            move = prices[right] / prices[left] - 1.0
            for t in range(left, right):
                states[t] = code
                phase_start[t], phase_end[t], phase_return[t] = left, right, move
                boundary_line[t] = prices[left] + (prices[right] - prices[left]) * (t - left) / (right - left)
            boundary_line[right] = prices[right]
        start = stop
    return states, pivots, phase_start, phase_end, phase_return, boundary_line


@njit(RESULT(F, int64, int64, int64, int64, float64), cache=True)
def peak_trough_kernel(prices, window, min_phase, min_cycle, endpoint_window, amplitude_exception):
    """Retain the original symmetric signature for reproducible legacy callers."""
    return peak_trough_asymmetric_kernel(prices, window, window, min_phase, min_cycle,
                                        endpoint_window, endpoint_window, amplitude_exception)


@njit(TIMING(I, I), cache=True)
def retrospective_dating_timing_kernel(states, available):
    """Global censoring depends on the entire sample, never an online signal."""
    recognition = np.full(states.size, -1, dtype=np.int64)
    effective = np.full(states.size, -1, dtype=np.int64)
    knowledge_at = available[0] if available.size else np.int64(0)
    for t in range(available.size):
        knowledge_at = max(knowledge_at, available[t])
    for t in range(states.size):
        if states[t] >= 0:
            recognition[t] = states.size - 1
    return recognition, effective, knowledge_at


@njit(SIDEWAYS(F, I, I, I, int64, int64, int64, float64, float64, float64, int64), cache=True)
def peak_trough_sideways_kernel(prices, states, phase_start, phase_end, enabled,
                               bear_code, neutral_code, small_swing, max_range,
                               max_efficiency, min_duration):
    """Earliest-start, longest qualifying complete-wave range; no tail fill.

    Price extrema/path length include both boundary prices. State intervals are
    half-open. Adjacent ranges cannot silently form a wider drifting range:
    if the preceding accepted range cannot extend, retain a separating wave.
    """
    n = prices.size
    output = states.copy()
    ranges, efficiencies = np.full(n, np.nan), np.full(n, np.nan)
    starts, ends = np.full(n, -1, dtype=np.int64), np.full(n, -1, dtype=np.int64)
    counts = np.zeros(n, dtype=np.int64)
    for t in range(n):
        if states[t] == 1:
            output[t] = bear_code
    if enabled == 0:
        return output, ranges, efficiencies, starts, ends, counts
    left = 0
    while left < n:
        if states[left] < 0 or phase_start[left] != left or phase_end[left] <= left:
            left += 1
            continue
        if left > 0 and output[left - 1] == neutral_code:
            left = phase_end[left]
            continue
        cursor, scanned, waves = left, left, 0
        low, high, path = prices[left], prices[left], 0.0
        best_end, best_waves = -1, 0
        best_range, best_efficiency = np.nan, np.nan
        last_peak = prices[left] if states[left] == 1 else np.nan
        last_trough = prices[left] if states[left] == 0 else np.nan
        peak_steps, trough_steps = 0, 0
        peaks_up, peaks_down, troughs_up, troughs_down = True, True, True, True
        while cursor < n and states[cursor] >= 0 and phase_start[cursor] == cursor:
            right = phase_end[cursor]
            if right <= cursor or right >= n:
                break
            if not np.isfinite(prices[right]) or prices[right] <= 0:
                break
            if abs(prices[right] / prices[cursor] - 1.0) > small_swing:
                break
            valid = True
            for t in range(scanned + 1, right + 1):
                if not np.isfinite(prices[t]) or prices[t] <= 0 or states[t - 1] < 0:
                    valid = False
                    break
                low, high = min(low, prices[t]), max(high, prices[t])
                path += abs(prices[t] - prices[t - 1])
            if not valid:
                break
            width = high / low - 1.0
            if width > max_range:
                break  # A wider extension cannot restore this constraint.
            waves += 1
            if states[cursor] == 0:
                if np.isfinite(last_peak):
                    peak_steps += 1
                    peaks_up = peaks_up and prices[right] > last_peak
                    peaks_down = peaks_down and prices[right] < last_peak
                last_peak = prices[right]
            else:
                if np.isfinite(last_trough):
                    trough_steps += 1
                    troughs_up = troughs_up and prices[right] > last_trough
                    troughs_down = troughs_down and prices[right] < last_trough
                last_trough = prices[right]
            efficiency = abs(prices[right] - prices[left]) / path if path > 0 else 0.0
            directional = (peak_steps > 0 and trough_steps > 0
                           and ((peaks_up and troughs_up) or (peaks_down and troughs_down)))
            if waves >= 2 and right - left >= min_duration and efficiency <= max_efficiency and not directional:
                best_end, best_waves = right, waves
                best_range, best_efficiency = width, efficiency
            # ER may fail now and recover after another reversed wave.
            scanned, cursor = right, right
        if best_end >= 0:
            for t in range(left, best_end):
                output[t] = neutral_code
                ranges[t], efficiencies[t] = best_range, best_efficiency
                starts[t], ends[t], counts[t] = left, best_end, best_waves
            left = best_end
        else:
            left = phase_end[left]
    return output, ranges, efficiencies, starts, ends, counts


PEAK_TROUGH_KERNELS = {
    "peak_trough": peak_trough_kernel,
    "peak_trough_asymmetric": peak_trough_asymmetric_kernel,
    "retrospective_dating_timing": retrospective_dating_timing_kernel,
    "peak_trough_remove": _remove_turn,
    "peak_trough_alternate": _alternate_turns,
    "peak_trough_sideways": peak_trough_sideways_kernel,
}
for dispatcher in PEAK_TROUGH_KERNELS.values():
    dispatcher.disable_compile()

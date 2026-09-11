"""Fixed-signature NJIT helpers for retrospective user-defined event intervals."""
from __future__ import annotations

import numpy as np
from numba import float64, int64, njit, types

I = int64[::1]
F = float64[::1]
STATE_RESULT = types.Tuple((I, F))
SUMMARY_RESULT = types.Tuple((I, I, I, I))


@njit(int64(I, int64), cache=True, inline="always")
def _lower_bound(values, target):
    left, right = 0, values.size
    while left < right:
        middle = (left + right) // 2
        if values[middle] < target:
            left = middle + 1
        else:
            right = middle
    return left


@njit(int64(I, int64), cache=True, inline="always")
def _upper_bound(values, target):
    left, right = 0, values.size
    while left < right:
        middle = (left + right) // 2
        if values[middle] <= target:
            left = middle + 1
        else:
            right = middle
    return left


@njit(STATE_RESULT(I, I, I), cache=True)
def manual_event_state_kernel(dates, starts, ends):
    """Return union state and simultaneous event count without a T×E matrix.

    State 0 means at least one event covers the observation; state 1 means no
    event covers it. Event boundaries are inclusive calendar timestamps.
    """
    if starts.size != ends.size:
        raise ValueError("Event start/end arrays must have the same length.")
    size = dates.size
    delta = np.zeros(size + 1, dtype=np.int64)
    for event_index in range(starts.size):
        start, end = starts[event_index], ends[event_index]
        if end < start:
            raise ValueError("Event end must not be earlier than start.")
        left = _lower_bound(dates, start)
        right = _upper_bound(dates, end)
        if left < right:
            delta[left] += 1
            if right < size:
                delta[right] -= 1
    states = np.ones(size, dtype=np.int64)
    counts = np.zeros(size, dtype=np.float64)
    active = 0
    for index in range(size):
        active += delta[index]
        counts[index] = float(active)
        if active > 0:
            states[index] = 0
    return states, counts


@njit(SUMMARY_RESULT(I, F, I, I), cache=True)
def manual_event_summary_kernel(dates, event_count, starts, ends):
    """Summarize each event and overlap coverage in O(T + E log T)."""
    if dates.size != event_count.size or starts.size != ends.size:
        raise ValueError("Manual event summary axes do not match.")
    event_observations = np.zeros(starts.size, dtype=np.int64)
    first_indices = np.full(starts.size, -1, dtype=np.int64)
    last_indices = np.full(starts.size, -1, dtype=np.int64)
    for event_index in range(starts.size):
        left = _lower_bound(dates, starts[event_index])
        right = _upper_bound(dates, ends[event_index])
        if left < right:
            event_observations[event_index] = right - left
            first_indices[event_index] = left
            last_indices[event_index] = right - 1
    union_count = 0
    overlap_count = 0
    maximum_concurrent = 0
    for index in range(event_count.size):
        count = int(event_count[index])
        if count > 0:
            union_count += 1
        if count > 1:
            overlap_count += 1
        if count > maximum_concurrent:
            maximum_concurrent = count
    summary = np.asarray(
        [union_count, overlap_count, maximum_concurrent, starts.size],
        dtype=np.int64,
    )
    return event_observations, first_indices, last_indices, summary


MANUAL_EVENT_KERNELS = {
    "manual_event_state": manual_event_state_kernel,
    "manual_event_summary": manual_event_summary_kernel,
}
for _dispatcher in MANUAL_EVENT_KERNELS.values():
    _dispatcher.disable_compile()

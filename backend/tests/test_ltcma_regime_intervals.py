"""Interval projection, unchanged observed labels and precompiled array ownership."""
import numpy as np
import pytest

from backend.strategic_allocation import cma_statistical_kernels as kernels


def readonly(values):
    # Exercise non-contiguous input views without mutating caller storage.
    storage = np.repeat(np.asarray(values, dtype=np.int64), 2)
    result = storage[::2]
    result.flags.writeable = False
    return result


def test_closed_intervals_preserve_unknown_gaps_and_original_observations():
    observed = readonly([0, 2, 4, 6, 8, 10])
    states = readonly([0, 0, 1, -1, 1, 1])
    available = readonly([1, 3, 5, 7, 9, 11])
    targets = readonly(range(-1, 12))
    originals = [array.copy() for array in (observed, states, available, targets)]
    signatures = tuple(kernels.align_regime_intervals.signatures)
    result, known = kernels.align_regime_intervals(observed, states, available, targets)
    np.testing.assert_array_equal(result, [-1, 0, 0, 0, -1, 1, -1, -1, -1, 1, 1, 1, -1])
    np.testing.assert_array_equal(known, [-1, 1, 3, 3, -1, 5, -1, 7, -1, 9, 11, 11, -1])
    exact, exact_known = kernels.align_regime_intervals(observed, states, available, observed)
    np.testing.assert_array_equal(exact, states)
    np.testing.assert_array_equal(exact_known, available)
    assert tuple(kernels.align_regime_intervals.signatures) == signatures
    for current, before in zip((observed, states, available, targets), originals):
        np.testing.assert_array_equal(current, before)
        assert not current.flags.writeable


@pytest.mark.parametrize("observed,states,available,targets", [
    ([0, 0], [0, 0], [0, 0], [0]),
    ([1, 0], [0, 0], [1, 1], [0]),
    ([0], [0], [-1], [0]),
    ([0], [-2], [0], [0]),
    ([0], [], [0], [0]),
    ([0], [0], [0], [1, 0]),
])
def test_invalid_axes_are_rejected(observed, states, available, targets):
    with pytest.raises(ValueError, match="LTCMA_REGIME_AXIS"):
        kernels.align_regime_intervals(*(readonly(v) for v in (observed, states, available, targets)))


def test_empty_and_single_point_do_not_extrapolate():
    empty = readonly([])
    result, _ = kernels.align_regime_intervals(empty, empty, empty, readonly([1, 2]))
    np.testing.assert_array_equal(result, [-1, -1])
    result, _ = kernels.align_regime_intervals(readonly([1]), readonly([0]), readonly([1]), readonly([0, 1, 2]))
    np.testing.assert_array_equal(result, [-1, 0, -1])

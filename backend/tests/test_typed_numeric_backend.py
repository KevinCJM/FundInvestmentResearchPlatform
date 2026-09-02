from __future__ import annotations

import numpy as np
import pytest

from cal_indicators.typed_numeric_backend import (
    _flat_float64,
    numeric_backend_status,
    warm_typed_numeric_backend,
)
from cal_indicators.typed_operators import get_typed_operator_registry
from optimizer import (
    OPTIMIZER_NUMBA_KERNELS,
    warm_optimizer_numba_kernels,
)


def test_hot_typed_operators_use_fixed_signature_numba_kernels() -> None:
    status = warm_typed_numeric_backend()
    registry = get_typed_operator_registry("2.2.0")
    values = np.asarray([0.01, -0.02, 0.03, 0.005], dtype=np.float64)

    assert status["warmed"] is True
    assert status["policy"]["one_dimensional_reductions_and_scans"] == (
        "numba_njit_fixed_signature"
    )
    assert all(status["compiled_signatures"].values())
    assert registry["mean"].evaluate(values) == pytest.approx(np.mean(values))
    assert registry["std"].evaluate(values, 1) == pytest.approx(
        np.std(values, ddof=1)
    )
    assert np.allclose(
        registry["cumulative_product"].evaluate(values), np.cumprod(values)
    )
    assert numeric_backend_status()["warmed"] is True


def test_frozen_typed_v20_registry_keeps_legacy_numpy_execution() -> None:
    current = get_typed_operator_registry("2.2.0")
    legacy = get_typed_operator_registry("2.0.0")

    assert current["mean"].evaluate.__module__.endswith("typed_numeric_backend")
    assert not legacy["mean"].evaluate.__module__.endswith(
        "typed_numeric_backend"
    )


def test_float64_contiguous_input_is_reused_without_copy() -> None:
    values = np.arange(32, dtype=np.float64)

    flattened = _flat_float64(values)

    assert np.shares_memory(flattened, values)


def test_optimizer_njit_kernels_are_fixed_signature_and_startup_ready() -> None:
    status = warm_optimizer_numba_kernels()

    assert status["warmed"] is True
    assert status["kernel_coverage"] == "4/4"
    assert all(dispatcher.signatures for dispatcher in OPTIMIZER_NUMBA_KERNELS)

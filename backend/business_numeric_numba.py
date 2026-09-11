"""Fixed-signature NJIT controls for browser-facing business arithmetic.

The web layer may shape arrays and serialize results, but totals, balancing,
allocation amounts and normalized shares are all produced here.  Every
dispatcher has one eager, immutable signature and is disabled for further
compilation after startup warmup.
"""

from __future__ import annotations

import hashlib
import inspect
from functools import lru_cache
from typing import Any

import numba
import numpy as np
from numba import njit, types
from numba.core.registry import CPUDispatcher

from compute_policy import NJIT_BACKEND, validate_execution_audit


BUSINESS_NUMERIC_KERNEL_VERSION = "business-numeric-v1"

_F8_1D = types.Array(types.float64, 1, "C", readonly=False)
_I8_1D = types.Array(types.int64, 1, "C", readonly=False)
_U1_1D = types.Array(types.uint8, 1, "C", readonly=False)

_GROUPED_RESULT = types.Tuple(
    (_F8_1D, _F8_1D, _U1_1D, _U1_1D, _F8_1D, types.int64)
)
_ALLOCATION_RESULT = types.Tuple(
    (types.float64, _F8_1D, types.float64, types.float64, types.uint8, types.int64)
)
_LEDGER_RESULT = types.Tuple(
    (
        _F8_1D,
        _F8_1D,
        _F8_1D,
        _U1_1D,
        _F8_1D,
        _F8_1D,
        _F8_1D,
        _I8_1D,
        types.int64,
        types.int64,
        types.float64,
        types.uint8,
        types.int64,
        types.int64,
        types.int64,
    )
)


@njit(
    _GROUPED_RESULT(_F8_1D, _I8_1D, _F8_1D, _F8_1D),
    cache=False,
    nogil=True,
)
def grouped_numeric_controls_kernel(
    values: np.ndarray,
    offsets: np.ndarray,
    targets: np.ndarray,
    tolerances: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Return authoritative non-negative totals, target checks and shares."""

    group_count = targets.size
    totals = np.zeros(group_count, dtype=np.float64)
    differences = np.zeros(group_count, dtype=np.float64)
    within_tolerance = np.zeros(group_count, dtype=np.uint8)
    positive = np.zeros(group_count, dtype=np.uint8)
    shares = np.zeros(values.size, dtype=np.float64)
    if offsets.size != group_count + 1 or tolerances.size != group_count:
        return totals, differences, within_tolerance, positive, shares, 1
    if offsets[0] != 0 or offsets[group_count] != values.size:
        return totals, differences, within_tolerance, positive, shares, 1

    for group_index in range(group_count):
        start = offsets[group_index]
        end = offsets[group_index + 1]
        target = targets[group_index]
        tolerance = tolerances[group_index]
        if start < 0 or end < start or end > values.size:
            return totals, differences, within_tolerance, positive, shares, 1
        if not np.isfinite(target) or not np.isfinite(tolerance) or tolerance < 0.0:
            return totals, differences, within_tolerance, positive, shares, 2
        total = 0.0
        for value_index in range(start, end):
            value = values[value_index]
            if not np.isfinite(value) or value < 0.0:
                return totals, differences, within_tolerance, positive, shares, 2
            total += value
        totals[group_index] = total
        difference = target - total
        differences[group_index] = difference
        within_tolerance[group_index] = np.uint8(abs(difference) <= tolerance)
        positive[group_index] = np.uint8(total > 0.0)
        if total > 0.0:
            for value_index in range(start, end):
                shares[value_index] = values[value_index] / total
    return totals, differences, within_tolerance, positive, shares, 0


@njit(
    _ALLOCATION_RESULT(types.float64, types.float64, _F8_1D, types.float64),
    cache=False,
    nogil=True,
)
def trade_allocation_summary_kernel(
    source_quantity: float,
    unit_price: float,
    allocations: np.ndarray,
    tolerance: float,
) -> tuple[float, np.ndarray, float, float, int, int]:
    """Price a fill and validate that allocated quantities fully reconcile."""

    amounts = np.zeros(allocations.size, dtype=np.float64)
    if (
        not np.isfinite(source_quantity)
        or not np.isfinite(unit_price)
        or not np.isfinite(tolerance)
        or source_quantity < 0.0
        or unit_price < 0.0
        or tolerance < 0.0
    ):
        return 0.0, amounts, 0.0, 0.0, np.uint8(0), 1
    allocated_total = 0.0
    for index in range(allocations.size):
        quantity = allocations[index]
        if not np.isfinite(quantity) or quantity < 0.0:
            return 0.0, amounts, 0.0, 0.0, np.uint8(0), 1
        allocated_total += quantity
        amounts[index] = quantity * unit_price
    source_amount = source_quantity * unit_price
    residual = source_quantity - allocated_total
    balanced = np.uint8(abs(residual) <= tolerance)
    return source_amount, amounts, allocated_total, residual, balanced, 0


@njit(
    _LEDGER_RESULT(
        _F8_1D,
        _F8_1D,
        _I8_1D,
        _I8_1D,
        types.int64,
        _U1_1D,
        types.float64,
        types.float64,
        types.float64,
    ),
    cache=False,
    nogil=True,
)
def ledger_summary_kernel(
    debits: np.ndarray,
    credits: np.ndarray,
    offsets: np.ndarray,
    entity_codes: np.ndarray,
    entity_count: int,
    pending_flags: np.ndarray,
    trial_debit: float,
    trial_credit: float,
    tolerance: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int,
    int,
    float,
    int,
    int,
    int,
    int,
]:
    """Aggregate voucher and entity debits/credits and trial-balance checks."""

    voucher_count = entity_codes.size
    voucher_debits = np.zeros(voucher_count, dtype=np.float64)
    voucher_credits = np.zeros(voucher_count, dtype=np.float64)
    voucher_differences = np.zeros(voucher_count, dtype=np.float64)
    voucher_balanced = np.zeros(voucher_count, dtype=np.uint8)
    entity_debits = np.zeros(max(entity_count, 0), dtype=np.float64)
    entity_credits = np.zeros(max(entity_count, 0), dtype=np.float64)
    entity_differences = np.zeros(max(entity_count, 0), dtype=np.float64)
    entity_voucher_counts = np.zeros(max(entity_count, 0), dtype=np.int64)
    if (
        debits.size != credits.size
        or offsets.size != voucher_count + 1
        or offsets.size == 0
        or offsets[0] != 0
        or offsets[voucher_count] != debits.size
        or entity_count < 0
        or not np.isfinite(trial_debit)
        or not np.isfinite(trial_credit)
        or trial_debit < 0.0
        or trial_credit < 0.0
        or not np.isfinite(tolerance)
        or tolerance < 0.0
    ):
        return (
            voucher_debits,
            voucher_credits,
            voucher_differences,
            voucher_balanced,
            entity_debits,
            entity_credits,
            entity_differences,
            entity_voucher_counts,
            0,
            0,
            0.0,
            np.uint8(0),
            0,
            0,
            1,
        )

    balanced_count = 0
    for voucher_index in range(voucher_count):
        start = offsets[voucher_index]
        end = offsets[voucher_index + 1]
        entity_code = entity_codes[voucher_index]
        if start < 0 or end < start or end > debits.size or entity_code < 0 or entity_code >= entity_count:
            return (
                voucher_debits,
                voucher_credits,
                voucher_differences,
                voucher_balanced,
                entity_debits,
                entity_credits,
                entity_differences,
                entity_voucher_counts,
                0,
                0,
                0.0,
                np.uint8(0),
                0,
                0,
                1,
            )
        debit_total = 0.0
        credit_total = 0.0
        for line_index in range(start, end):
            debit_value = debits[line_index]
            credit_value = credits[line_index]
            if (
                not np.isfinite(debit_value)
                or not np.isfinite(credit_value)
                or debit_value < 0.0
                or credit_value < 0.0
            ):
                return (
                    voucher_debits,
                    voucher_credits,
                    voucher_differences,
                    voucher_balanced,
                    entity_debits,
                    entity_credits,
                    entity_differences,
                    entity_voucher_counts,
                    0,
                    0,
                    0.0,
                    np.uint8(0),
                    0,
                    0,
                    2,
                )
            debit_total += debit_value
            credit_total += credit_value
        difference = debit_total - credit_total
        is_balanced = np.uint8(abs(difference) <= tolerance)
        voucher_debits[voucher_index] = debit_total
        voucher_credits[voucher_index] = credit_total
        voucher_differences[voucher_index] = difference
        voucher_balanced[voucher_index] = is_balanced
        balanced_count += int(is_balanced)
        entity_debits[entity_code] += debit_total
        entity_credits[entity_code] += credit_total
        entity_voucher_counts[entity_code] += 1

    for entity_index in range(entity_count):
        entity_differences[entity_index] = entity_debits[entity_index] - entity_credits[entity_index]

    pending_count = 0
    for flag in pending_flags:
        pending_count += int(flag != 0)
    trial_difference = trial_debit - trial_credit
    trial_balanced = np.uint8(abs(trial_difference) <= tolerance)
    return (
        voucher_debits,
        voucher_credits,
        voucher_differences,
        voucher_balanced,
        entity_debits,
        entity_credits,
        entity_differences,
        entity_voucher_counts,
        balanced_count,
        pending_count,
        trial_difference,
        trial_balanced,
        pending_flags.size,
        voucher_count,
        0,
    )


BUSINESS_NUMERIC_KERNELS: tuple[CPUDispatcher, ...] = (
    grouped_numeric_controls_kernel,
    trade_allocation_summary_kernel,
    ledger_summary_kernel,
)
_BUSINESS_NUMERIC_WARMED = False


def _fingerprint(dispatcher: CPUDispatcher) -> str:
    return hashlib.sha256(inspect.getsource(dispatcher.py_func).encode("utf-8")).hexdigest()


def business_numeric_status(*, warmed: bool | None = None) -> dict[str, Any]:
    signatures = {
        dispatcher.py_func.__name__: [str(signature) for signature in dispatcher.nopython_signatures]
        for dispatcher in BUSINESS_NUMERIC_KERNELS
    }
    is_warmed = _BUSINESS_NUMERIC_WARMED if warmed is None else bool(warmed)
    fingerprints = {
        dispatcher.py_func.__name__: _fingerprint(dispatcher)
        for dispatcher in BUSINESS_NUMERIC_KERNELS
    }
    aggregate = hashlib.sha256(
        "|".join(f"{name}:{fingerprints[name]}" for name in sorted(fingerprints)).encode("utf-8")
    ).hexdigest()
    return {
        "version": BUSINESS_NUMERIC_KERNEL_VERSION,
        "kernel_version": BUSINESS_NUMERIC_KERNEL_VERSION,
        "numba_version": numba.__version__,
        "warmed": is_warmed,
        "fully_warmed": is_warmed and all(signatures.values()),
        "kernel_coverage": f"{sum(bool(value) for value in signatures.values())}/{len(signatures)}",
        "kernel_signatures": signatures,
        "compiled_signatures": signatures,
        "kernel_fingerprints": fingerprints,
        "fingerprint": aggregate,
        "backend": NJIT_BACKEND,
        "execution_backend": NJIT_BACKEND,
        "nopython": all(bool(dispatcher.nopython_signatures) for dispatcher in BUSINESS_NUMERIC_KERNELS),
        "object_mode": 0,
        "python_fallback": 0,
    }


@lru_cache(maxsize=1)
def warm_business_numeric_kernels() -> dict[str, Any]:
    global _BUSINESS_NUMERIC_WARMED

    values = np.ascontiguousarray([60.0, 40.0], dtype=np.float64)
    offsets = np.ascontiguousarray([0, 2], dtype=np.int64)
    targets = np.ascontiguousarray([100.0], dtype=np.float64)
    tolerances = np.ascontiguousarray([1.0e-8], dtype=np.float64)
    grouped = grouped_numeric_controls_kernel(values, offsets, targets, tolerances)
    if grouped[-1] != 0 or grouped[2][0] != 1:
        raise RuntimeError("business numeric grouped-control NJIT warmup failed")

    allocation = trade_allocation_summary_kernel(
        np.float64(100.0),
        np.float64(4.2),
        np.ascontiguousarray([60.0, 40.0], dtype=np.float64),
        np.float64(1.0e-8),
    )
    if allocation[-1] != 0 or allocation[4] != 1:
        raise RuntimeError("business numeric allocation NJIT warmup failed")

    ledger = ledger_summary_kernel(
        np.ascontiguousarray([100.0, 0.0], dtype=np.float64),
        np.ascontiguousarray([0.0, 100.0], dtype=np.float64),
        np.ascontiguousarray([0, 2], dtype=np.int64),
        np.ascontiguousarray([0], dtype=np.int64),
        np.int64(1),
        np.ascontiguousarray([1, 0], dtype=np.uint8),
        np.float64(100.0),
        np.float64(100.0),
        np.float64(0.005),
    )
    if ledger[-1] != 0 or ledger[3][0] != 1 or ledger[11] != 1:
        raise RuntimeError("business numeric ledger NJIT warmup failed")

    if any(len(dispatcher.nopython_signatures) != 1 for dispatcher in BUSINESS_NUMERIC_KERNELS):
        raise RuntimeError("business numeric NJIT kernels must each expose exactly one fixed signature")
    for dispatcher in BUSINESS_NUMERIC_KERNELS:
        dispatcher.disable_compile()
    _BUSINESS_NUMERIC_WARMED = True
    return validate_execution_audit(business_numeric_status())


def business_numeric_execution_audit() -> dict[str, Any]:
    if not _BUSINESS_NUMERIC_WARMED:
        raise RuntimeError("business numeric NJIT kernels are not warmed")
    return validate_execution_audit(business_numeric_status())


__all__ = [
    "BUSINESS_NUMERIC_KERNEL_VERSION",
    "BUSINESS_NUMERIC_KERNELS",
    "business_numeric_execution_audit",
    "business_numeric_status",
    "grouped_numeric_controls_kernel",
    "ledger_summary_kernel",
    "trade_allocation_summary_kernel",
    "warm_business_numeric_kernels",
]

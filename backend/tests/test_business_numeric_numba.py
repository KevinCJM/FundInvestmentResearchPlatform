from __future__ import annotations

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

from business_numeric_numba import (
    BUSINESS_NUMERIC_KERNELS,
    grouped_numeric_controls_kernel,
    ledger_summary_kernel,
    trade_allocation_summary_kernel,
    warm_business_numeric_kernels,
)
from services.business_numeric_routes import router


def test_business_numeric_kernels_are_fixed_signature_nopython_only():
    audit = warm_business_numeric_kernels()

    assert audit["execution_backend"] == "numba_njit_fixed_signature"
    assert audit["nopython"] is True
    assert audit["object_mode"] == 0
    assert audit["python_fallback"] == 0
    assert audit["request_time_compilation"] == 0
    assert audit["kernel_coverage"] == "3/3"
    assert all(len(dispatcher.nopython_signatures) == 1 for dispatcher in BUSINESS_NUMERIC_KERNELS)


def test_grouped_controls_return_total_target_check_and_normalized_shares():
    warm_business_numeric_kernels()
    result = grouped_numeric_controls_kernel(
        np.ascontiguousarray([60.0, 40.0, 25.0, 25.0], dtype=np.float64),
        np.ascontiguousarray([0, 2, 4], dtype=np.int64),
        np.ascontiguousarray([100.0, 100.0], dtype=np.float64),
        np.ascontiguousarray([1.0e-8, 1.0e-8], dtype=np.float64),
    )
    totals, differences, within, positive, shares, status = result

    assert status == 0
    np.testing.assert_allclose(totals, [100.0, 50.0])
    np.testing.assert_allclose(differences, [0.0, 50.0])
    np.testing.assert_array_equal(within, [1, 0])
    np.testing.assert_array_equal(positive, [1, 1])
    np.testing.assert_allclose(shares, [0.6, 0.4, 0.5, 0.5])


def test_grouped_controls_reject_negative_business_values_inside_njit():
    warm_business_numeric_kernels()
    result = grouped_numeric_controls_kernel(
        np.ascontiguousarray([110.0, -10.0], dtype=np.float64),
        np.ascontiguousarray([0, 2], dtype=np.int64),
        np.ascontiguousarray([100.0], dtype=np.float64),
        np.ascontiguousarray([1.0e-8], dtype=np.float64),
    )

    assert result[-1] == 2


def test_trade_and_ledger_kernels_keep_amounts_and_balances_in_njit():
    warm_business_numeric_kernels()
    source_amount, amounts, total, residual, balanced, status = trade_allocation_summary_kernel(
        np.float64(1_000_000),
        np.float64(4.2),
        np.ascontiguousarray([600_000.0, 400_000.0], dtype=np.float64),
        np.float64(1.0e-8),
    )
    assert status == 0
    assert source_amount == 4_200_000
    np.testing.assert_allclose(amounts, [2_520_000, 1_680_000])
    assert total == 1_000_000
    assert residual == 0
    assert balanced == 1

    ledger = ledger_summary_kernel(
        np.ascontiguousarray([490_000.0, 8.0, 0.0, 86_420.0, 0.0], dtype=np.float64),
        np.ascontiguousarray([0.0, 0.0, 490_008.0, 0.0, 86_420.0], dtype=np.float64),
        np.ascontiguousarray([0, 3, 5], dtype=np.int64),
        np.ascontiguousarray([0, 1], dtype=np.int64),
        np.int64(2),
        np.ascontiguousarray([0, 1, 1], dtype=np.uint8),
        np.float64(100_000),
        np.float64(99_800),
        np.float64(0.005),
    )
    assert ledger[-1] == 0
    np.testing.assert_allclose(ledger[0], [490_008, 86_420])
    np.testing.assert_allclose(ledger[1], [490_008, 86_420])
    np.testing.assert_array_equal(ledger[3], [1, 1])
    assert ledger[8] == 2
    assert ledger[9] == 2
    assert ledger[10] == 200
    assert ledger[11] == 0


def test_ledger_kernel_rejects_negative_amounts_inside_njit():
    warm_business_numeric_kernels()
    ledger = ledger_summary_kernel(
        np.ascontiguousarray([-1.0], dtype=np.float64),
        np.ascontiguousarray([0.0], dtype=np.float64),
        np.ascontiguousarray([0, 1], dtype=np.int64),
        np.ascontiguousarray([0], dtype=np.int64),
        np.int64(1),
        np.ascontiguousarray([], dtype=np.uint8),
        np.float64(0.0),
        np.float64(0.0),
        np.float64(0.005),
    )

    assert ledger[-1] == 2


def test_business_numeric_api_exposes_only_warmed_njit_results():
    warm_business_numeric_kernels()
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    controls = client.post(
        "/api/business-numeric/controls",
        json={
            "groups": [
                {"key": "weight", "values": [60, 40], "target": 100, "tolerance": 0.01}
            ]
        },
    )
    assert controls.status_code == 200
    assert controls.json()["items"][0] == {
        "key": "weight",
        "total": 100.0,
        "difference": 0.0,
        "within_tolerance": True,
        "positive": True,
        "normalized_shares": [0.6, 0.4],
    }
    assert controls.json()["execution"]["request_time_compilation"] == 0

    negative_controls = client.post(
        "/api/business-numeric/controls",
        json={
            "groups": [
                {"key": "weight", "values": [110, -10], "target": 100}
            ]
        },
    )
    assert negative_controls.status_code == 422
    assert "非负数" in negative_controls.json()["detail"]

    allocation = client.post(
        "/api/business-numeric/trade-allocation",
        json={
            "source_quantity": 100,
            "unit_price": 4.2,
            "allocations": [{"key": "a", "quantity": 60}, {"key": "b", "quantity": 40}],
        },
    )
    assert allocation.status_code == 200
    assert allocation.json()["source_amount"] == 420.0
    assert allocation.json()["residual"] == 0.0
    assert allocation.json()["balanced"] is True

    ledger = client.post(
        "/api/business-numeric/ledger-summary",
        json={
            "entities": ["fund", "manager"],
            "vouchers": [
                {
                    "id": "v1",
                    "entity": "fund",
                    "lines": [{"debit": 100, "credit": 0}, {"debit": 0, "credit": 100}],
                }
            ],
            "pending_flags": [False, True],
            "trial_debit": 100,
            "trial_credit": 99,
        },
    )
    assert ledger.status_code == 200
    assert ledger.json()["vouchers"][0]["balanced"] is True
    assert ledger.json()["metrics"]["pending_count"] == 1
    assert ledger.json()["trial"]["difference"] == 1.0
    assert ledger.json()["execution"]["python_fallback"] == 0

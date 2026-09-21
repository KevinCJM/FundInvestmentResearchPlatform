from datetime import date
import numpy as np
import pytest
from backend.pre_investment import funding
from backend.pre_investment.contracts import FundingState
from backend.custom_indicators.errors import ValidationError


@pytest.fixture(scope="module", autouse=True)
def warm():
    funding.warm()


def prepared(months):
    return {
        "remaining_months": months,
        "plan": {"required_probability": 0.8},
        "probability_required": True,
        "target": 0.0,
        "inflows": np.zeros(months),
        "outflows": np.full(months, 10000.0),
        "overdue_payment_amount": 0.0,
        "original_plan_missed_payment": False,
    }


@pytest.mark.parametrize("months", [1, 3, 13, 37])
def test_arbitrary_remaining_months_preserve_last_checkpoint(months):
    result = funding.diagnose(prepared(months), 0.0, 0.0, 800000.0, 0.0, 200, 42)
    assert result["metrics"]["terminal_median"] == pytest.approx(
        800000.0 - 10000.0 * months
    )
    assert result["fan"][-1]["month"] == months
    assert len(result["fan"]) == (months + 11) // 12 + 1
    assert result["status"] == "passed"


def test_zero_months_and_zero_initial_do_not_invent_capital():
    p = prepared(0)
    p["target"] = 100.0
    assert funding.diagnose(p, 0.0, 0.0, 100.0, 0.0, 200, 42)["status"] == "passed"
    assert funding.diagnose(p, 0.0, 0.0, 99.0, 0.0, 200, 42)["status"] == "failed"
    p = prepared(3)
    p["inflows"] = np.full(3, 20000.0)
    assert (
        funding.diagnose(p, 0.0, 0.0, 0.0, 0.0, 200, 42)["metrics"]["terminal_median"]
        == 30000.0
    )


def test_old_integer_year_abi_and_readonly_views():
    draws = np.zeros((12, 200, 1))
    flows = np.zeros(24)[::2]
    draws.flags.writeable = False
    flows.flags.writeable = False
    old = funding.goals.funding_paths_kernel(
        draws, 0.05, 0.1, 100.0, flows, flows, 100.0, 0.01, 0.8, 1.0, 1.0
    )
    d, s = funding.goals.funding_monthly_parameters_kernel(0.05, 0.1, 0.01, 0, 1)
    new = funding.goals.funding_paths_from_monthly_kernel(
        draws, d, s, 100.0, flows, flows, 100.0, 0.8, 1.0, 1.0
    )
    for left, right in zip(old, new, strict=True):
        np.testing.assert_allclose(left, right, equal_nan=True)
    with pytest.raises(ValueError):
        funding.goals.funding_paths_kernel(
            draws[:1],
            0.05,
            0.1,
            100.0,
            flows[:1],
            flows[:1],
            100.0,
            0.01,
            0.8,
            1.0,
            1.0,
        )


def mandate():
    return {
        "as_of": "2025-01-31",
        "horizon_years": 2,
        "funding_plan": {
            "total_capital": 1000000.0,
            "outside_reserve": 200000.0,
            "terminal_target": 100000.0,
            "amount_basis": "real",
            "inflation": 0.02,
            "annual_fee": 0.0,
            "required_probability": 0.8,
            "flows": [
                {
                    "name": "每月支付",
                    "kind": "withdrawal",
                    "amount": 10000.0,
                    "first_month": 1,
                    "last_month": 24,
                    "every_months": 1,
                }
            ],
        },
    }


def test_partial_overdue_and_original_real_price_base():
    m = mandate()
    rows, _ = funding.occurrences(m)
    state = FundingState(
        valuation_at=date(2025, 2, 28),
        knowledge_cutoff=date(2025, 2, 28),
        confirmed_investable_value=800000.0,
        settled_cash=800000.0,
        elapsed_months=1,
        evidence="已核对银行余额",
        reconciliation=[
            {
                "occurrence_id": rows[0]["occurrence_id"],
                "status": "partial",
                "paid_amount": 5000.0,
                "evidence": "实际付款五千元",
            }
        ],
    )
    p = funding.prepare(m, state)
    assert p["remaining_months"] == 23
    assert p["original_plan_missed_payment"]
    assert p["target"] == pytest.approx(100000.0 * 1.02**2)
    assert p["outflows"][0] == pytest.approx(
        10000.0 * 1.02 ** (1 / 12) - 5000.0 + 10000.0 * 1.02 ** (2 / 12)
    )
    result = funding.diagnose(p, 0.0, 0.0, 800000.0, 0.0, 200, 42)
    assert result[
        "original_plan_missed_payment"
    ]  # Future wealth never rewrites history.
    with pytest.raises(ValidationError, match="每笔"):
        funding.prepare(m, state.model_copy(update={"reconciliation": []}))
    with pytest.raises(ValidationError, match="模型月边界"):
        funding.prepare(m, state.model_copy(update={"valuation_at": date(2025, 2, 27)}))


def test_sequence_and_cost_cannot_erase_missed_payment():
    # Unique cash recurrence: a later contribution does not reset the missed flag.
    p = prepared(3)
    p["outflows"] = np.array([20.0, 0.0, 0.0])
    p["inflows"] = np.array([0.0, 100.0, 0.0])
    result = funding.diagnose(p, 0.0, 0.0, 10.0, 0.0, 200, 42)
    assert result["metrics"]["terminal_median"] == 100.0
    assert result["metrics"]["payment_failure_probability"] == 1.0
    assert result["future_conditional_success_probability"] == 0.0

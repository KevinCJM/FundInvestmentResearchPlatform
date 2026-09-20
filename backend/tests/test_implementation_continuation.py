"""Real saved policy → remaining 13 months → frozen validation integration."""

from datetime import date, timedelta
import json
import pytest
from backend.tests.test_implementation_service import implementation
from backend.tests.test_strategic_allocation import (
    workspace,
    warm,
    definition,
    confirmed_mandate,
)
from backend.strategic_allocation.contracts import (
    MandateRequest,
    PublishCmaRequest,
    PolicyRequest,
    PublishPolicyRequest,
)
from backend.pre_investment.contracts import (
    ImplementationCandidate,
    PackageWrite,
    PackageAction,
)
from backend.pre_investment.funding import month_date, occurrences


def test_saved_policy_thirteen_months_and_product_cash_paths(implementation):
    service, body = implementation
    today = date.today()
    origin = month_date(today, -11)
    pool_path = service.strategic.data.data_dir / "product_pools.json"
    pool = json.loads(pool_path.read_text())
    pool["universe_snapshots"][0]["research_date"] = str(origin)
    pool_path.write_text(json.dumps(pool))
    mandate = confirmed_mandate(
        service.strategic,
        MandateRequest(
            name="两年资金计划",
            as_of=origin,
            review_date=today + timedelta(days=180),
            horizon_years=2,
            objective_kind="funding_goal",
            max_volatility=0.3,
            max_tracking_error=0.2,
            target_return=0.0,
            funding_plan={
                "total_capital": 100000.0,
                "terminal_target": 1000.0,
                "required_probability": 0.5,
                "liquidity_months": 1,
                "flows": [
                    {
                        "name": "每月付款",
                        "kind": "withdrawal",
                        "amount": 100.0,
                        "first_month": 1,
                        "last_month": 24,
                        "every_months": 1,
                    }
                ],
            },
        ),
    )
    raw = definition().model_dump(mode="json")
    raw["horizon_years"] = 2
    raw["as_of"] = str(origin)
    raw["assets"][1]["role"] = "liquidity"
    from backend.strategic_allocation.contracts import CmaRequest

    req = CmaRequest.model_validate(raw)
    prev = service.strategic.preview_cma(req)
    cma = service.strategic.publish_cma(
        PublishCmaRequest(request=req, preview_hash=prev["preview_hash"])
    )
    req = PolicyRequest(
        mandate_id=mandate["id"],
        cma_id=cma["id"],
        candidate_count=300,
        constraints={
            "股票": {"min_weight": 0.5, "max_weight": 0.5},
            "债券": {"min_weight": 0.5, "max_weight": 0.5},
        },
    )
    prev = service.strategic.preview_policy(req)
    policy = service.strategic.publish_policy(
        PublishPolicyRequest(
            request=req,
            preview_hash=prev["preview_hash"],
            candidate_id="nominal-utility",
            name="两年现金政策",
            reason="固定资产预算用于产品现金验收",
        )
    )
    raw = body.model_dump(mode="json")
    raw["source"] = {
        "kind": "saa_policy",
        "id": policy["id"],
        "content_hash": policy["content_hash"],
    }
    raw["products"][0]["weight"] = 0.5
    raw["products"][1] = {
        "kind": "cash",
        "product_id": "CASH",
        "asset_class_id": "债券",
        "weight": 0.5,
        "current_value": 100000.0,
    }
    flows, _ = occurrences(policy["policy"]["mandate"])
    raw["state"].update(
        elapsed_months=11,
        reconciliation=[
            {
                "occurrence_id": r["occurrence_id"],
                "status": "paid",
                "paid_amount": 100.0,
                "evidence": "已确认历史付款",
            }
            for r in flows
            if r["month"] <= 11
        ],
    )
    candidate = ImplementationCandidate.model_validate(raw)
    preview = service.preview(candidate)
    assert preview["research_ready"], preview["checks"]
    assert preview["funding_results"][0]["remaining_months"] == 13
    assert not preview["independent_simulation"]
    package = service.save(
        PackageWrite(candidate=candidate, idempotency_key="continuation-create")
    )
    saved = service.validate(
        package["scheme_id"],
        PackageAction(
            expected_revision=package["revision"],
            candidate_hash=package["candidate_hash"],
            idempotency_key="continuation-validate",
        ),
    )
    report = service.repository.report(saved["report_id"])
    assert (
        report["independent_simulation"]
        and report["funding_results"][0]["seed"] == candidate.validation_seed
    )
    raw["future_weight_rule"] = "monthly_rebalance"
    raw["future_fee_assumption"] = "constant_declared_rates_sensitivity"
    raw["products"][0].update(
        same_month_settlement_confirmed=True,
        settlement_terms_source="离线月末同日结算约定",
    )
    product = service.preview(ImplementationCandidate.model_validate(raw))
    assert product["product_paths"]["status"] == "passed", product["checks"]
    assert (
        product["product_paths"]["results"][0]["metrics"]["payment_failure_probability"]
        == 0
    )


def test_optional_cash_budget_does_not_invent_success_threshold(implementation):
    import numpy as np
    from backend.pre_investment import paths

    service, body = implementation
    raw = body.model_dump(mode="json")
    raw["future_weight_rule"] = "buy_and_hold"
    raw["products"][0]["weight"] = 0.5
    raw["future_fee_assumption"] = "constant_declared_rates_sensitivity"
    raw["products"][0].update(
        same_month_settlement_confirmed=True, settlement_terms_source="月末测试结算"
    )
    raw["products"][1] = {
        "kind": "cash",
        "product_id": "CASH",
        "asset_class_id": "债券",
        "weight": 0.5,
    }
    candidate = ImplementationCandidate.model_validate(raw)
    prepared = {
        "remaining_months": 1,
        "plan": {},
        "probability_required": False,
        "target": 0.0,
        "inflows": np.zeros(1),
        "outflows": np.array([50.0]),
        "original_plan_missed_payment": False,
    }
    result = paths.diagnose(
        candidate,
        prepared,
        {"post_cost_holdings": [100.0, 0.0]},
        [{"model_id": "single", "name": "假设", "enforced": True}],
        {"product_means_0": np.zeros(2), "product_covariance_0": np.zeros((2, 2))},
        0.0,
        validation=True,
    )
    assert (
        result["status"] == "not_applicable"
        and result["results"][0]["threshold"] is None
    )
    assert result["results"][0]["metrics"]["payment_failure_probability"] == 1.0

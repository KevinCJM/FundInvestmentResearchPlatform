"""Exact SAA/TAA identities with optional, explicit later product mapping."""

import copy
from datetime import date
from backend.custom_indicators.errors import ConflictError, ValidationError
from backend.strategic_allocation.cma_application import frozen_policy_assumptions
from backend.strategic_allocation.policy_gate import check_policy
from backend.strategic_allocation.sources import strategic_source
from backend.tactical_allocation.portfolio_bridge import validate_decision_application
from .funding import occurrences


class AllocationSources:
    def __init__(self, strategic):
        self.strategic = strategic

    def resolve(self, reference, as_of, *, current=True):
        service = self.strategic
        decision = None
        if reference.kind == "saa_policy":
            saved = service.baselines.get_baseline(reference.id)
            baseline = copy.deepcopy(saved)
            target = {a["id"]: a["base_weight"] for a in baseline["assets"]}
        else:
            saved = decision = service.baselines.get_decision(reference.id)
            baseline = copy.deepcopy(saved["preview"]["baseline"])
            target = saved["preview"]["recommendation"]["weights"]
        if saved["content_hash"] != reference.content_hash:
            raise ConflictError(
                "IMPLEMENTATION_SOURCE_HASH", "来源指纹不匹配，请重新选择原始版本。"
            )
        policy = baseline.get("policy")
        if not policy:
            raise ValidationError(
                "IMPLEMENTATION_POLICY_REQUIRED",
                "来源缺少已确认的投资目标与长期政策，请先确认 SAA。",
            )
        assumptions = frozen_policy_assumptions(policy)
        if str(as_of) < baseline["as_of"]:
            raise ValidationError(
                "IMPLEMENTATION_SOURCE_DATE", "实施研究日不能早于来源。"
            )
        if reference.implementation_mapping_id:
            if decision:
                raise ValidationError(
                    "IMPLEMENTATION_TAA_MAPPING",
                    "TAA 应使用决策冻结的映射；更换映射须重新确认 TAA。",
                )
            if not baseline.get("strategic_universe_id"):
                raise ValidationError(
                    "IMPLEMENTATION_MAPPING_SCOPE",
                    "此政策未定义独立战略范围，不能添加不相关映射。",
                )
            mapping = service.scopes.get_mapping(reference.implementation_mapping_id)
            universe = service.scopes.get_universe(baseline["strategic_universe_id"])
            if (
                mapping["strategic_universe_hash"] != universe["content_hash"]
                or mapping["definition"]["strategic_universe_id"] != universe["id"]
            ):
                raise ValidationError(
                    "IMPLEMENTATION_MAPPING_SCOPE", "实施映射与政策的战略范围不一致。"
                )
            replacement = strategic_source(universe, mapping, str(as_of))
            for row, original in zip(
                replacement["assets"], baseline["assets"], strict=True
            ):
                for key in ("base_weight", "min_weight", "max_weight", "max_abs_tilt"):
                    row[key] = original[key]
            baseline.update(replacement)
        names = [x["id"] for x in baseline["assets"]]
        if set(target) != set(names):
            raise ValidationError(
                "IMPLEMENTATION_SOURCE_AXIS", "来源预算与资产轴不一致。"
            )
        if current:
            service._require_active_mandate(policy["mandate_id"])
            refs = policy.get("multi_cma", {}).get("refs") or [
                {"cma_id": policy["cma_id"], "content_hash": policy["cma_hash"]}
            ]
            for ref in refs:
                cma = service.cma.require_selectable(ref["cma_id"])
                if cma["content_hash"] != ref["content_hash"]:
                    raise ConflictError(
                        "IMPLEMENTATION_CMA_HASH", "长期假设冻结指纹不一致。"
                    )
            if decision:
                validate_decision_application(
                    decision,
                    service.data,
                    service.baselines.decision_arrays(decision["id"])["returns"],
                    strategic_root=service.artifacts.root.parents[1],
                )
            service.data.validate_application(baseline)
        gate = check_policy(
            baseline,
            target,
            float(policy["mandate"]["max_tracking_error"]),
            str(as_of),
            strategic_root=service.artifacts.root.parents[1],
            data_dir=service.data.data_dir,
        )
        if current and not gate["current_application_eligible"]:
            reasons = [
                *gate["violations"],
                *gate["implementation_blockers"],
                *gate["risk_scale_blockers"],
                *gate["manual_review_blockers"],
                *gate["current_manual_review_blockers"],
            ]
            raise ValidationError(
                "IMPLEMENTATION_SOURCE_INELIGIBLE",
                "；".join(reasons) or "政策已到复核日，请重新确认。",
            )
        return {
            "saved": saved,
            "baseline": baseline,
            "target": target,
            "assumptions": assumptions,
            "policy": policy,
            "gate": gate,
            "refs": {
                "source": {"id": saved["id"], "content_hash": saved["content_hash"]},
                "mandate": {
                    "id": policy["mandate_id"],
                    "content_hash": policy["mandate_hash"],
                },
                "cma": policy.get("multi_cma", {}).get("refs")
                or [{"cma_id": policy["cma_id"], "content_hash": policy["cma_hash"]}],
                "mapping": baseline.get("implementation_mapping_snapshot"),
            },
        }

    def catalog(self):
        service = self.strategic
        items = []
        for kind, rows in (
            ("saa_policy", service.baselines.list_baselines()),
            ("taa_decision", service.baselines.list_decisions()),
        ):
            for saved in rows[:100]:
                baseline = (
                    saved if kind == "saa_policy" else saved["preview"]["baseline"]
                )
                if not baseline.get("policy"):
                    continue
                policy = baseline["policy"]
                flows, context = occurrences(policy["mandate"])
                items.append(
                    {
                        "kind": kind,
                        "id": saved["id"],
                        "content_hash": saved["content_hash"],
                        "name": saved["name"],
                        "as_of": baseline["as_of"],
                        "expires_on": policy["expires_on"],
                        "mode": policy.get("mode", "single"),
                        "assets": [
                            {
                                **a,
                                "role": next(
                                    x["role"]
                                    for x in policy["assumptions"]["assets"]
                                    if x["id"] == a["id"]
                                ),
                            }
                            for a in baseline["assets"]
                        ],
                        "target": (
                            {a["id"]: a["base_weight"] for a in baseline["assets"]}
                            if kind == "saa_policy"
                            else saved["preview"]["recommendation"]["weights"]
                        ),
                        "strategic_universe_id": baseline.get("strategic_universe_id"),
                        "funding": (
                            {
                                "occurrences": flows,
                                "origin": context["origin"],
                                "plan": context["plan"],
                                "months": policy["mandate"]["horizon_years"] * 12,
                            }
                            if context
                            else None
                        ),
                    }
                )
        mappings = [
            service.scopes.get_mapping(x["id"])
            for x in service.artifacts.list("series")
            if x.get("artifact_type") == "implementation_mapping"
        ]
        return {
            "sources": items,
            "mappings": mappings,
            "today": str(date.today()),
            "capabilities": {
                "manual_products": True,
                "class_budget_qp": True,
                "linear_costs": True,
                "monthly_continuation": True,
                "authenticated_independent_review": False,
                "intraday_execution": False,
                "integer_orders": False,
            },
        }

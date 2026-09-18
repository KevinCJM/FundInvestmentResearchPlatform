"""Normalize compact objective inputs into frozen constraints consumed downstream."""
from __future__ import annotations

from copy import deepcopy

from backend.custom_indicators.errors import ConflictError, ValidationError
from backend.sensitivity.repository import digest_json

MODEL_POLICY_SOURCE = "investment_objectives_model_convention_v1"
DEFAULT_REQUIRED_PROBABILITY = 0.80
DEFAULT_LIQUIDITY_MONTHS = 12
DEFAULT_CONTRIBUTION_STRESS_RATIO = 0.50


def cash_success_required(definition: dict) -> bool:
    if definition.get("schema_version", "1.0") == "1.0":
        return definition.get("funding_plan") is not None
    budget = definition.get("cash_budget")
    if budget is None:
        return False
    return bool(definition.get("funding_target") or definition.get("cash_protection"))


def effective_boundary_policy(definition: dict) -> dict | None:
    """Freeze model conventions when the simplified UI supplies no governance form."""
    if definition.get("schema_version", "1.0") != "2.0":
        return definition.get("boundary_policy")
    existing = definition.get("boundary_policy")
    if existing is not None:
        return deepcopy(existing)
    budget = definition.get("cash_budget")
    months = min(DEFAULT_LIQUIDITY_MONTHS, int(definition["horizon_years"]) * 12) if budget else None
    return {
        "name": "investment-objectives-model-convention-v1",
        "source": MODEL_POLICY_SOURCE,
        "reviewed_on": definition["as_of"],
        "valid_until": definition.get("review_date"),
        "confirmed": True,
        "required_probability": DEFAULT_REQUIRED_PROBABILITY if cash_success_required(definition) else None,
        "liquidity_months": months,
        "contribution_stress_ratio": DEFAULT_CONTRIBUTION_STRESS_RATIO if budget else None,
        "cash_reserve_weight": 0.0,
    }


def effective_funding_plan(definition: dict) -> dict | None:
    """Adapt supported input contracts without mutating or duplicating cash flows."""
    if definition.get("schema_version", "1.0") == "1.0":
        return definition.get("funding_plan")
    budget = definition.get("cash_budget")
    if budget is None:
        return None
    policy = effective_boundary_policy(definition)
    protection = definition.get("cash_protection") or {}
    target = definition.get("funding_target") or protection.get("terminal_floor")
    return {**budget, "terminal_target": target["amount"] if target else 0.,
            "target_amount_basis": target["amount_basis"] if target else "nominal",
            "required_probability": policy.get("required_probability"),
            "liquidity_months": policy["liquidity_months"],
            "contribution_stress_ratio": policy["contribution_stress_ratio"],
            # No unrequested loss preference: only path MDD is reported.
            "drawdown_alert": 1., "drawdown_alert_enabled": False}


def has_cash_budget(definition: dict) -> bool:
    return effective_funding_plan(definition) is not None


def effective_cash_floor(definition: dict, funding_floor: float) -> float:
    """Compose cash-purpose floors without confusing tradability with cash."""
    institutional = (definition.get("institutional_context") or {}).get("cash_reserve_weight", 0.)
    if definition.get("schema_version", "1.0") != "2.0":
        return institutional
    policy = effective_boundary_policy(definition) or {}
    return max(float(definition.get("min_cash_weight", 0.)), institutional,
               float(policy.get("cash_reserve_weight", 0.)), funding_floor)


def effective_return_floor(definition: dict, cashflow_required_return: float | None) -> float | None:
    """现金流按期支付所要求的收益也是一条收益下限，与填写的预期收益取大者。

    None 表示本目标不使用算术收益下限（资金目标和相对收益目标各有自己的成功口径）。
    现金流要求的是扣费前固定年复合收益，与填写的算术预期收益口径不同，取大者是
    确定性筛选，不替代资金成功概率诊断。
    """
    if definition.get("objective_kind", "absolute_return") != "absolute_return":
        return None
    stated = float(definition.get("target_return") or 0.)
    if cashflow_required_return is None:
        return stated
    return max(stated, float(cashflow_required_return))


def _freeze_reference_benchmark(resolved: dict, version: dict, level: int) -> None:
    """Use the selected Risk Scale representative as the relative-return benchmark."""
    if resolved.get("objective_kind") != "benchmark_relative" or resolved.get("benchmark") is not None:
        return
    result = version["preview"]["result"]
    levels, ids = result.get("levels", []), result.get("ordered_asset_ids", [])
    if not 1 <= level <= len(levels):
        raise ValidationError("MANDATE_BENCHMARK_LEVEL", "所选风险等级不存在，无法形成参考基准。")
    weights = levels[level - 1].get("representative_weights")
    if (not isinstance(weights, list) or len(weights) != len(ids)
            or any(type(value) not in (int, float) for value in weights)
            or abs(sum(float(value) for value in weights) - 1.) > 1e-8):
        raise ValidationError("MANDATE_BENCHMARK_UNAVAILABLE", "该风险等级没有完整代表组合，无法作为相对收益基准。")
    resolved["benchmark"] = {
        "name": f"{version['name']} C{level} 参考组合",
        "alloc_name": "risk-scale-reference",
        "weights": {asset_id: float(weight) for asset_id, weight in zip(ids, weights, strict=True)},
        "target_excess_return": float(resolved.get("target_excess_return", 0.)),
        "max_tracking_error": 1.0,
        "source": "risk_scale_reference",
    }


def resolve_authorization(definition: dict, risk_scales) -> tuple[dict, dict | None]:
    """Resolve one immutable Risk Scale into a frozen numeric mandate."""
    resolved = deepcopy(definition)
    if definition.get("schema_version", "1.0") == "1.0":
        return resolved, None

    policy = effective_boundary_policy(resolved)
    resolved["boundary_policy"] = policy
    resolved["boundary_policy_hash"] = digest_json(policy)
    risk = resolved["risk_authorization"]
    decision = {"mode": risk["mode"], "risk_scale_ref": risk.get("risk_scale_ref"),
                "authorized_max_level": risk.get("authorized_max_level"),
                "selected_max_level": risk.get("selected_max_level"),
                "minimum_tested_feasible_level": None, "realized_model_risk_level": None,
                "authorized_volatility_cap": None, "selected_volatility_cap": None,
                "selection_pending": False, "current_application_blockers": []}
    if risk["mode"] == "explicit_numeric":
        decision.update(status="explicit_numeric", selected_volatility_cap=definition["max_volatility"],
                        authorized_volatility_cap=definition["max_volatility"])
        return resolved, decision
    if not risk.get("risk_scale_ref"):
        resolved["max_volatility"] = None
        decision.update(status="reference_pending", selection_pending=True)
        return resolved, decision

    ref = risk["risk_scale_ref"]
    version = risk_scales.get_version(ref["id"])
    if version["content_hash"] != ref["content_hash"]:
        raise ConflictError("MANDATE_RISK_SCALE_CHANGED", "风险标尺引用指纹不一致，请重新选择版本。")
    preview = version["preview"]
    scale = preview["request_echo"]["definition"]
    if not preview["publication_eligibility"]["eligible"]:
        raise ValidationError("MANDATE_RISK_SCALE_INVALID", "风险标尺未通过发布核验。")
    if scale["base_currency"] != definition["currency"] or scale["risk_basis_id"] != "annualized-periodic-volatility-v1":
        raise ValidationError("MANDATE_RISK_SCALE_BASIS", "目标计价币种须与所选风险标尺一致。")
    as_of = definition["as_of"]
    if scale["research_as_of"] > as_of or scale.get("valid_until") and scale["valid_until"] <= as_of:
        raise ValidationError("MANDATE_RISK_SCALE_DATE", "标尺晚于目标研究日或已超过硬失效日，请选择适用版本。")
    retirement = risk_scales.store.read()["retired"].get(version["id"])
    if retirement and retirement["retired_on"] <= as_of:
        raise ValidationError("MANDATE_RISK_SCALE_RETIRED", "目标研究日时此标尺已退休，不能新引用。")
    caps = preview["result"]["applied_boundaries"]
    if len(caps) != 5:
        raise ValidationError("MANDATE_RISK_SCALE_INVALID", "冻结标尺缺少完整的C1至C5边界。")
    authorized = risk["authorized_max_level"]
    selected = risk.get("selected_max_level")
    level = selected or authorized
    resolved["max_volatility"] = caps[level - 1]
    _freeze_reference_benchmark(resolved, version, level)
    decision.update(status="selected" if selected else "awaiting_recommendation",
                    selection_pending=selected is None,
                    authorized_volatility_cap=caps[authorized - 1],
                    selected_volatility_cap=caps[selected - 1] if selected else None,
                    scale_name=version["name"], scale_version=version["version_number"], applied_boundaries=caps,
                    current_application_blockers=version["current_eligibility"]["blockers"],
                    historical_pit_proven=False,
                    reference_valid_until=scale.get("valid_until"))
    return resolved, decision


def require_resolved_authorization(definition: dict) -> None:
    """Saved research inputs are not necessarily usable investment constraints."""
    if definition.get("max_volatility") is None:
        raise ValidationError("MANDATE_AUTHORIZATION_PENDING", "此目标尚无已确认数值风险上限，请补齐授权后再计算SAA。")
    if definition.get("schema_version", "1.0") == "2.0":
        policy = definition.get("boundary_policy")
        if not policy or definition.get("boundary_policy_hash") != digest_json(policy):
            raise ValidationError("MANDATE_POLICY_EVIDENCE", "冻结的模型与边界约定缺失或不一致，请重新确认目标。")

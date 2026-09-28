"""Downstream admission for frozen CMA results, separate from saving research."""
from __future__ import annotations

from backend.custom_indicators.errors import ValidationError


def downstream_eligibility(item: dict) -> dict:
    """A confirmed research result does not itself authorize a new SAA/prior use."""
    method = (item.get("definition", {}).get("model") or {}).get("method")
    validation = item.get("model_result", {}).get("model_audit", {}).get("model_validation") or {}
    if method == "conditional_scenario":
        # No approved horizon-to-SAA adapter exists. A metadata boolean must not
        # turn a finite-horizon experiment into long-term policy assumptions.
        return {"downstream_eligible": False, "downstream_reason":
            "条件情景用于研究指定期间的变化，暂不能作为 SAA 或贝叶斯先验；请使用长期情景或其他长期假设。"}
    if method == "long_term_scenario" and validation.get("downstream_eligible") is not True:
        return {"downstream_eligible": False, "downstream_reason":
            "此长期情景缺少完整的统计计算依据，请重新生成并确认后再用于 SAA。"}
    return {"downstream_eligible": True, "downstream_reason": None}


def require_downstream_eligible(item: dict) -> None:
    status = downstream_eligibility(item)
    if not status["downstream_eligible"]:
        raise ValidationError("LTCMA_DOWNSTREAM_UNAVAILABLE", status["downstream_reason"])

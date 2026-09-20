"""Reuse published scenario application; save results only inside explicit reports."""

from backend.custom_indicators.errors import IndicatorDomainError, ValidationError
from .evaluation import check, unavailable


class ScenarioAdapter:
    def __init__(self, published):
        self.published = published

    def options(self):
        return {
            "scenarios": [
                {"id": row["id"], "name": row["name"]}
                for row in self.published.releases()["items"]
                if row["status"] == "active"
            ],
            "exposures": [
                {"id": row["id"], "name": row["name"]}
                for row in self.published.risks.releases(include_unavailable=False)[
                    "items"
                ]
            ],
        }

    def __call__(self, candidate, source, candidate_hash):
        checks, results = [], []
        for identifier in candidate.scenario_release_ids:
            try:
                if not candidate.scenario_exposure_release_id:
                    raise ValidationError(
                        "SCENARIO_EXPOSURE_REQUIRED", "请选择已发布产品暴露模型。"
                    )
                if any(p.kind == "cash" for p in candidate.products):
                    raise ValidationError(
                        "SCENARIO_CASH_AXIS",
                        "现有发布情景尚未定义现金项，本次不自动删除现金并重分配风险持仓。",
                    )
                result = self.published.impact(
                    {
                        "scenario_release_id": identifier,
                        "exposure_release_id": candidate.scenario_exposure_release_id,
                        "as_of": str(candidate.as_of),
                        "target": {
                            "kind": "implementation_candidate",
                            "candidate_hash": candidate_hash,
                            "holdings": [
                                {
                                    "key": f"{p.kind}:{p.product_id.upper()}",
                                    "weight": p.weight,
                                }
                                for p in candidate.products
                            ],
                        },
                        "holding_policy": "buy_and_hold",
                        "hold_other_factors_constant": True,
                        "notional": candidate.state.confirmed_investable_value or 1.0,
                    }
                )
                results.append(
                    {
                        "impact": result,
                        "adoption": "frozen_only_when_explicit_package_validation_is_saved",
                    }
                )
                checks.append(
                    check(
                        "scenario:" + identifier,
                        result["name"],
                        "passed",
                        "确定性买入持有压力诊断；不赋予场景概率，未计入交易费或残差风险。",
                        scope="diagnostic",
                        enforcement="information",
                    )
                )
            except IndicatorDomainError as exc:
                checks.append(unavailable("scenario:" + identifier, "已发布情景", exc))
        return {
            "checks": checks,
            "results": results,
            "scope": "deterministic_buy_and_hold_published_model",
        }

    def current(self, candidate):
        for identifier in candidate.scenario_release_ids:
            self.published.resolve_release(identifier)
        if candidate.scenario_exposure_release_id:
            self.published.risks.resolve_release(candidate.scenario_exposure_release_id)

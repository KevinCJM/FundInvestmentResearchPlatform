"""Published scenario construction and on-demand, cacheable product applications."""
from datetime import date

from fastapi import APIRouter

from backend.scenario_stress.published import PublishedScenarioService
from backend.scenario_stress.published_contracts import ImpactRequest, ScenarioFields, ScenarioPublish
from backend.sensitivity.contracts import RetireRequest
from .risk_model_routes import call


def build_router(service: PublishedScenarioService):
    router = APIRouter(prefix="/api/published-scenarios", tags=["published-scenarios"])

    @router.post("/previews")
    def preview(request: ScenarioFields):
        return call(service.preview, request.model_dump(mode="json"))

    @router.get("/previews/{identifier}")
    def preview_detail(identifier: str):
        return call(service.artifacts.get, identifier, "preview")

    @router.get("/releases")
    def releases(as_of: date | None = None):
        return call(service.releases, as_of)

    @router.post("/releases", status_code=201)
    def publish(request: ScenarioPublish):
        return call(service.publish, request.model_dump(mode="json"))

    @router.post("/releases/{identifier}/retire")
    def retire(identifier: str, request: RetireRequest):
        return call(service.retire, identifier, request.note)

    @router.get("/portfolios")
    def portfolios():
        return call(service.portfolio_choices)

    @router.post("/impacts")
    def impact(request: ImpactRequest):
        return call(service.impact, request.model_dump(mode="json"))

    @router.get("/impacts")
    def impacts():
        return {"items": call(service.impacts.list, "impact")[:200]}

    @router.get("/impacts/{identifier}")
    def impact_detail(identifier: str):
        return call(service.impacts.get, identifier, "impact")

    return router


published_scenario_service = PublishedScenarioService()
router = build_router(published_scenario_service)

"""LTCMA catalog and lifecycle routes, installed before the generic ID route."""
from datetime import date
from typing import Literal
from fastapi import Query
from .cma_center_contracts import CmaDraftDelete, CmaDraftWrite, CmaRetire, CmaListResponse, CmaDraftView
from .cma_center_contracts import CmaSampleRequest, CmaSampleSummary, CmaCenterUpdate
from .common_contracts import Identifier


def install_cma_center_routes(router, service, call):
    @router.get("/cma", response_model=CmaListResponse)
    def list_cma(q: str = Query("", max_length=120), method: str = Query("", max_length=60),
                 include_retired: bool = False, offset: int = Query(0, ge=0),
                 limit: int = Query(50, ge=1, le=200)):
        return call(service.cma.list, q, method, include_retired, offset, limit)

    @router.get("/cma/capabilities")
    def capabilities():
        return call(service.cma.capabilities)

    @router.get("/cma/study-options")
    def study_options(as_of: date | None = None,
                      section: Literal["all", "base", "priors", "regimes", "scenarios"] = "all",
                      selected_prior_id: Identifier | None = None):
        return call(service.cma.study_options, as_of, section, selected_prior_id)

    @router.post("/cma/sample", response_model=CmaSampleSummary)
    def sample(body: CmaSampleRequest):
        return call(service.cma.sample, body)

    @router.get("/cma/drafts")
    def drafts():
        return {"items": call(service.cma.drafts.read)}

    @router.post("/cma/drafts", status_code=201, response_model=CmaDraftView)
    def create_draft(body: CmaDraftWrite):
        return call(service.cma.drafts.save, body)

    @router.get("/cma/drafts/{identifier}", response_model=CmaDraftView)
    def get_draft(identifier: str):
        return call(service.cma.drafts.get, identifier)

    @router.patch("/cma/drafts/{identifier}", response_model=CmaDraftView)
    def update_draft(identifier: str, body: CmaDraftWrite):
        return call(service.cma.drafts.save, body, identifier)

    @router.delete("/cma/drafts/{identifier}")
    def delete_draft(identifier: str, body: CmaDraftDelete):
        return call(service.cma.drafts.delete, identifier, body.expected_revision)

    @router.get("/cma/{identifier}/view")
    def view(identifier: str):
        return call(service.cma.view, identifier)

    @router.patch("/cma/{identifier}")
    def update_cma(identifier: str, body: CmaCenterUpdate):
        return call(service.cma.publish, body, identifier)

    @router.post("/cma/{identifier}/retire")
    def retire(identifier: str, body: CmaRetire):
        return call(service.cma.retire, identifier, body)

"""Dependency-injected implementation and research-package API."""

from fastapi import APIRouter, Response
from backend.strategic_allocation.routes import _call
from .contracts import (
    ImplementationCandidate,
    PackageWrite,
    PackageAction,
    FinalizePackage,
    RegisterAttempt,
)


def build_router(service):
    router = APIRouter(prefix="/api/pre-investment", tags=["pre-investment"])

    @router.get("/catalog")
    def catalog():
        return _call(service.catalog)

    @router.post("/preview")
    def preview(body: ImplementationCandidate):
        return _call(service.preview, body)

    @router.post("/optimize")
    def optimize(body: ImplementationCandidate):
        return _call(service.optimize, body)

    @router.get("/packages")
    def packages():
        return {"items": _call(service.repository.list)}

    @router.post("/packages", status_code=201)
    def create(body: PackageWrite):
        return _call(service.save, body)

    @router.get("/packages/{identifier}")
    def view(identifier: str):
        return _call(service.view, identifier)

    @router.put("/packages/{identifier}")
    def update(identifier: str, body: PackageWrite):
        return _call(service.save, body, identifier)

    @router.post("/packages/{identifier}/validate")
    def validate(identifier: str, body: PackageAction):
        return _call(service.validate, identifier, body)

    @router.post("/packages/{identifier}/finalize")
    def finalize(identifier: str, body: FinalizePackage):
        return _call(service.finalize, identifier, body)

    @router.post("/attempts", status_code=201)
    def attempt(body: RegisterAttempt):
        return _call(service.register_attempt, body)

    @router.get("/packages/{identifier}/export")
    def export(identifier: str):
        return Response(
            _call(service.export, identifier),
            media_type="application/zip",
            headers={
                "Content-Disposition": 'attachment; filename="research-package.zip"'
            },
        )

    return router

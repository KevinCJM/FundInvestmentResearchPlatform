"""Thin installation onto the historical-regime router."""

from .contracts import (
    PreviewRequest,
    ConfirmRequest,
    QualityRequest,
    QualityConfirmRequest,
)


def install(router, get_service, call):
    @router.get("/api/historical-regimes/references")
    def references():
        return call(get_service().references)

    @router.post("/api/historical-regimes/reliability/preview")
    def preview(request: PreviewRequest):
        return call(get_service().preview, request.model_dump(mode="json"))

    @router.post("/api/historical-regimes/reliability/confirm")
    def confirm(request: ConfirmRequest):
        return call(get_service().confirm, request.model_dump(mode="json"))

    @router.get("/api/historical-regimes/reliability/reports/{report_id}")
    def report(report_id: str):
        return call(get_service().get, report_id)

    @router.get("/api/historical-regimes/reliability/catalog")
    def catalog():
        return call(get_service().catalog)

    @router.get("/api/historical-regimes/reliability/reports/{report_id}/recognition-evidence")
    def recognition_evidence(report_id: str):
        return call(get_service().recognition_evidence, report_id)

    @router.get("/api/historical-regimes/reliability/reports/{report_id}/cma-evidence")
    def cma_evidence(report_id: str):
        return call(get_service().cma_evidence, report_id)


def install_quality(router, get_service, call):
    @router.post("/api/historical-regimes/reference-quality/preview")
    def quality_preview(request: QualityRequest):
        return call(get_service().preview, request.model_dump(mode="json"))

    @router.post("/api/historical-regimes/reference-quality/confirm")
    def quality_confirm(request: QualityConfirmRequest):
        return call(get_service().confirm, request.model_dump(mode="json"))

    @router.get("/api/historical-regimes/reference-quality/reports/{report_id}")
    def quality_report(report_id: str):
        return call(get_service().get, report_id)

    @router.get("/api/historical-regimes/reference-quality/catalog")
    def quality_catalog():
        return call(get_service().catalog)

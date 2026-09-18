"""Dependency-injected synchronous routes run in FastAPI's thread pool."""
from fastapi import APIRouter, HTTPException, Query
from fastapi.routing import APIRoute
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError as PydanticValidationError
from backend.custom_indicators.errors import IndicatorDomainError
from backend.data_storage import StorageError
from backend.research_series.service import ResearchSeriesError
from .risk_scale_contracts import (
    PreviewRequest, ConfirmRequest, DraftWrite, DraftUpdate, RevisionRequest, ActivateRequest, RetireRequest,
    CompareRequest, ClassifyRequest, PreviewResponse, VersionView, CatalogResponse, DefaultsResponse,
    DefaultBinding, DraftView, Capabilities, CompareResponse, ClassifyResponse, ReferencePreview, ReferenceVersion,
    SourceCatalog, Summary, StudyOptionsResponse, DeleteResponse, ErrorResponse,
)
from .reference_contracts import ReferenceInputRequest, ConfirmReferenceInput
from .reference_inputs import problem


class RiskScaleRoute(APIRoute):
    def get_route_handler(self):
        original = super().get_route_handler()
        async def handler(request):
            try:
                return await original(request)
            except RequestValidationError as exc:
                error = exc.errors()[0]
                field = '.'.join(str(x) for x in error['loc'] if x != 'body')
                raise HTTPException(422, detail=problem('INPUT_INVALID', '字段不符合契约，请检查必填项、类型及范围。', field)) from exc
        return handler


def call(function, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except IndicatorDomainError as exc:
        raise HTTPException(exc.status_code, detail=problem(exc.code, exc.message, exc.field)) from exc
    except (StorageError, OSError):
        raise HTTPException(503, detail=problem('STORAGE_UNAVAILABLE', '存储不可用、只读或容量不足，请检查受控数据磁盘。')) from None
    except ResearchSeriesError as exc:
        raise HTTPException(422, detail=problem(exc.code, '真实来源目前不可用，请在数据中心检查对应数据及覆盖区间。')) from exc
    except RuntimeError:
        raise HTTPException(503, detail=problem('COMPUTE_NOT_READY', '计算执行能力不可用，请等待本进程预热或检查启动日志。')) from None
    except (ValueError, PydanticValidationError):
        raise HTTPException(422, detail=problem('INPUT_NUMERIC_INVALID', '输入未通过数值或来源契约校验；不会自动修正。')) from None


def summaries(items):
    keys = Summary.model_fields
    return [{k: v for k,v in item.items() if k in keys} for item in items]


def build_router(service):
    router = APIRouter(prefix='/api/strategic-allocation', tags=['risk-scales'], route_class=RiskScaleRoute,
        responses={status: {'model': ErrorResponse} for status in (400, 404, 409, 422, 503)})
    prefix = '/risk-scales'

    @router.get(prefix + '/capabilities', response_model=Capabilities)
    def capabilities():
        return call(service.capabilities)

    @router.get(prefix, response_model=CatalogResponse)
    def catalog(offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=100),
                base_currency: str | None = None, risk_basis_id: str | None = None):
        result = call(service.catalog, offset, limit, base_currency, risk_basis_id)
        result['items'] = summaries(result['items'])
        return result

    @router.get(prefix + '/defaults', response_model=DefaultsResponse)
    def defaults():
        return call(service.defaults)

    @router.get(prefix + '/study-options', response_model=StudyOptionsResponse)
    def study_options(as_of: str = Query(..., pattern=r'^\d{4}-\d{2}-\d{2}$')):
        return call(service.study_options, as_of)

    @router.post(prefix + '/compare', response_model=CompareResponse)
    def compare(body: CompareRequest):
        return call(service.compare, body)

    @router.post(prefix + '/drafts', response_model=DraftView, status_code=201)
    def create_draft(body: DraftWrite):
        return call(service.store.create_draft, body)

    @router.get(prefix + '/drafts/{identifier}', response_model=DraftView)
    def get_draft(identifier: str):
        return call(service.store.get_draft, identifier)

    @router.patch(prefix + '/drafts/{identifier}', response_model=DraftView)
    def update_draft(identifier: str, body: DraftUpdate):
        return call(service.store.update_draft, identifier, body)

    @router.delete(prefix + '/drafts/{identifier}', response_model=DeleteResponse)
    def delete_draft(identifier: str, body: RevisionRequest):
        return call(service.store.delete_draft, identifier, body.expected_revision)

    @router.post(prefix + '/preview', response_model=PreviewResponse)
    def preview(body: PreviewRequest):
        return call(service.preview, body)

    @router.post(prefix + '/confirm', response_model=VersionView, status_code=201)
    def confirm(body: ConfirmRequest):
        return call(service.confirm, body)

    @router.get(prefix + '/{identifier}', response_model=VersionView)
    def get_version(identifier: str):
        return call(service.get_version, identifier)

    @router.post(prefix + '/{identifier}/activate', response_model=DefaultBinding)
    def activate(identifier: str, body: ActivateRequest):
        return call(service.activate, identifier, body)

    @router.post(prefix + '/{identifier}/retire', response_model=DefaultBinding)
    def retire(identifier: str, body: RetireRequest):
        return call(service.retire, identifier, body)

    @router.post(prefix + '/{identifier}/classify', response_model=ClassifyResponse)
    def classify(identifier: str, body: ClassifyRequest):
        return call(service.classify, identifier, body)

    shared = '/reference-inputs'

    @router.get(shared + '/catalog', response_model=SourceCatalog)
    def source_catalog(kind: str = Query('index', pattern='^(index|etf|fund)$'), q: str = Query('', max_length=120),
                       offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=200)):
        return call(service.references.sources.catalog, kind, q, offset, limit)

    @router.post(shared + '/preview', response_model=ReferencePreview)
    def preview_reference(body: ReferenceInputRequest):
        return call(service.reference_call, 'preview', body)

    @router.post(shared + '/confirm', response_model=ReferenceVersion, status_code=201)
    def confirm_reference(body: ConfirmReferenceInput):
        return call(service.reference_call, 'confirm', body)

    @router.get(shared + '/{identifier}', response_model=ReferenceVersion)
    def get_reference(identifier: str):
        return call(service.references.version, identifier)

    return router

"""Only authentication, context and business HTTP contracts for the external agent."""
from contextlib import asynccontextmanager
import json
import time

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from research_access import commit, data_policy, identity
from research_access.contracts import ResearchError
from services.custom_indicator_contracts import StableValidationRoute
from . import admission
from .contracts import AdoptionInput, AdmissionInput, AuthorityInput, AuthoringInput, BootstrapInput, CommitInput, ConfirmationInput, ContextInput, ToolInput
from .service import HostIntegration, TOOL_NAMES
from .catalog import capabilities


class IntegrationBodyLimit:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        path = scope.get('path', '')
        if scope['type'] != 'http' or not path.startswith(('/api/integrations/portable-agent/', '/internal/portable-agent/', '/api/custom-indicators/authorings')):
            return await self.app(scope, receive, send)
        limit, total = 4*1024*1024, 0
        async def bounded_receive():
            nonlocal total
            message = await receive()
            total += len(message.get('body', b''))
            if total > limit:
                raise HTTPException(413, '接入请求超过允许大小。')
            return message
        await self.app(scope, bounded_receive, send)


def install(app, *, indicator_service, page_services, store=None, authority=None):
    integration = HostIntegration(indicator_service, page_services, store=store, authority=authority)
    app.add_middleware(IntegrationBodyLimit)
    app.state.portable_agent_integration = integration

    @asynccontextmanager
    async def lifespan(_app):
        yield
        await integration.close()

    router = APIRouter(tags=['portable-agent-integration'], lifespan=lifespan, route_class=StableValidationRoute)

    @app.exception_handler(ResearchError)
    async def research_error(request, exc):
        return JSONResponse({'error': exc.detail()}, status_code=exc.status_code)

    @router.post('/api/integrations/portable-agent/local-session')
    async def local_session(request: Request):
        token = identity.local_session(request)
        response = JSONResponse({'authenticated': True})
        response.set_cookie('research_identity', token, httponly=True, secure=request.url.scheme == 'https',
                            samesite='strict', max_age=3600, path='/api/')
        return response

    @router.post('/api/integrations/portable-agent/contexts')
    async def register_context(body: ContextInput, request: Request):
        principal = identity.user(request)
        record = await integration.register(principal, body, pit_off=request.headers.get('x-pit-off', '').lower() in {'1', 'true', 'yes'}, view_headers=request.headers)
        return integration.public_context(record)

    @router.get('/api/integrations/portable-agent/capabilities')
    async def read_capabilities(request: Request):
        principal = identity.user(request)
        current = await integration.current_principal(principal['sub'], principal['workspace'])
        identity.permit(current, 'assistant:use')
        return {'items': capabilities()}

    @router.post('/api/integrations/portable-agent/bootstrap')
    async def bootstrap(body: BootstrapInput, request: Request):
        return await integration.bootstrap(identity.user(request), body.context_ref)

    @router.get('/api/integrations/portable-agent/contexts/{cid}')
    async def read_context(cid: str, request: Request):
        principal = identity.user(request)
        record = integration.store.context(cid, subject=principal['sub'], workspace=principal['workspace'])
        await integration.authorize(identity.application_id(), principal['sub'], integration.public_context(record), action='read')
        selected = record.get('requested_page_context', record['page_context'])
        return {'context': integration.public_context(record), 'value': selected,
                'identity': {k: selected[k] for k in ('page', 'page_instance_id', 'view_state', 'calculation')}}

    @router.post('/api/integrations/portable-agent/adoptions')
    async def adoption(body: AdoptionInput, request: Request):
        principal = identity.user(request)
        operation = integration.store.operation(body.operation_id)
        payload = operation['payload']
        if (operation['subject'] != principal['sub'] or operation['workspace'] != principal['workspace']
                or operation['status'] != 'succeeded' or operation.get('cancel_requested')
                or payload['session_id'] != body.session_id or payload['run_id'] != body.run_id or payload['context']['ref'] != body.context_ref):
            raise ResearchError('ADOPTION_STALE', '试算回执不属于当前任务。', status_code=409)
        record, current = await integration.authorize(payload['application'], principal['sub'], payload['context'])
        artifact = (operation.get('result') or {}).get('artifact') or {}
        if not artifact.get('preview_id'):
            raise ResearchError('ADOPTION_STALE', '该操作没有可采纳的试算结果。', status_code=409)
        preview = integration.store.preview(artifact['authoring_id'], artifact['preview_id'])
        if preview.get('run_id') != body.run_id or not any(row.get('status') in {'ok', 'warning'} for row in preview['result'].get('results', [])):
            raise ResearchError('ADOPTION_STALE', '尚无本轮成功试算可供采纳。', status_code=409)
        from research_access.contracts import PageContext
        versions = await run_in_threadpool(integration.versions, PageContext.model_validate(record['page_context']))
        if versions['catalog_version'] != record['catalog_version'] or versions['data_generation'] != record['data_generation']:
            raise ResearchError('ADOPTION_STALE', '数据或目录已经改变，不能自动采纳旧结果。', status_code=409)
        page = dict(record['page_context'])
        page['calculation'] = {**page['calculation'], 'targets': [preview['target']], 'period': preview['period'], 'as_of': preview.get('as_of')}
        candidate = ContextInput.model_validate({'page_context': page, 'authoring_id': artifact['authoring_id']})
        renewed = await integration.register(current, candidate, pit_off=page['view_state'] == 'off', derived_from=body.operation_id, frozen_pit=record.get('pit'))
        return {'context': integration.public_context(renewed), 'capture_identity': {k: renewed['page_context'][k] for k in ('page', 'page_instance_id', 'view_state', 'calculation')}}

    @router.post('/internal/portable-agent/authorize')
    async def authorize(body: AuthorityInput, request: Request):
        identity.service(request)
        if body.identity_only:
            if body.action not in {'metadata', 'settings_read', 'settings_write'} or body.application != identity.application_id():
                raise ResearchError('FORBIDDEN', '身份核验不能代替业务上下文授权。', status_code=403)
            record = integration.store.context(body.context.get('ref'), subject=body.subject, workspace=body.context.get('workspace'))
            if body.context != {key: integration.public_context(record)[key] for key in ('ref','hash','grant_id','grant_revision','workspace')} or record['revoked'] or record['expires'] <= time.time():
                raise ResearchError('FORBIDDEN', '身份授权已失效。', status_code=403)
            current = await integration.current_principal(body.subject, record['workspace'])
            identity.permit(current, 'assistant:use')
            if body.action == 'settings_write':
                identity.permit(current, 'settings:write')
            return {'allow': True}
        await integration.authorize(body.application, body.subject, body.context, action=body.action,
                                    operation_id=body.operation_id, session_id=body.session_id, run_id=body.run_id)
        if body.action == 'memory_source':
            with integration.store.db() as db:
                row = db.execute('SELECT authoring_id,body FROM confirmations WHERE id=?', (body.source_ref,)).fetchone()
            if not row:
                raise ResearchError('MEMORY_SOURCE_NOT_FOUND', '记忆来源不存在。', status_code=404)
            principal = await integration.current_principal(body.subject, body.context['workspace'])
            authoring = integration.store.read_authoring(row['authoring_id'], principal)
            frozen = json.loads(row['body'])
            return {'allow': True, 'memory': {'key': 'indicator_definition:'+frozen['definition_hash'], 'object_id':'scope',
                'value': f"记住指标「{frozen['definition'].get('name','')}」的定义引用；使用前应读取业务定义并重新校验。",
                'reference': f"research-authoring:{authoring['id']}@{frozen['definition_hash']}",
                'source_label': '指标保存影响预览', 'source_ref': body.source_ref}}
        return {'allow': True}

    @router.post('/internal/portable-agent/admission')
    async def check_admission(body: AdmissionInput, request: Request):
        identity.service(request)
        return await admission.admit(integration, body)

    @router.post('/internal/portable-agent/tools/{name}')
    async def execute_tool(name: str, body: ToolInput, request: Request):
        identity.service(request)
        if request.headers.get('idempotency-key') != body.operation_id:
            raise ResearchError('REQUEST_CONFLICT', '操作标识与HTTP幂等键不匹配。', status_code=409)
        try:
            result = await integration.submit(name, body)
        except ResearchError as exc:
            if name not in TOOL_NAMES:
                raise
            model = data_policy.seal({'ok': False, 'code': exc.code, 'message': exc.message}, TOOL_NAMES[name])
            return JSONResponse({'error': exc.detail(), 'model': model}, status_code=exc.status_code)
        if result['status'] == 'succeeded':
            return JSONResponse({k: result[k] for k in ('model', 'artifact', 'progress_token') if k in result})
        return JSONResponse(result, status_code=202)

    async def operation_for_service(oid, request, action):
        identity.service(request)
        operation = integration.store.operation(oid)
        payload = operation['payload']
        await integration.authorize(payload['application'], payload['subject'], payload['context'], action=action)
        return operation

    @router.get('/internal/portable-agent/operations/{oid}')
    async def read_operation(oid: str, request: Request):
        return integration.public_operation(await operation_for_service(oid, request, 'read'))

    @router.post('/internal/portable-agent/operations/{oid}/cancel')
    async def cancel_operation(oid: str, request: Request):
        await operation_for_service(oid, request, 'cancel')
        return integration.cancel(oid)

    @router.post('/api/custom-indicators/authorings')
    async def create_authoring(body: AuthoringInput, request: Request):
        principal = identity.user(request)
        record = integration.store.context(body.context_ref, subject=principal['sub'], workspace=principal['workspace'])
        _, current = await integration.authorize(identity.application_id(), principal['sub'], integration.public_context(record))
        identity.permit(current, 'indicator:draft')
        return integration.store.authoring(current, 'manual-'+body.request_id, scope=record['scope'], context_hash=record['hash'])

    @router.get('/api/custom-indicators/authorings/{aid}')
    async def read_authoring(aid: str, request: Request, revision: int | None = None):
        principal = identity.user(request)
        current = await integration.current_principal(principal['sub'], principal['workspace'])
        identity.permit(current, 'indicator:draft')
        await run_in_threadpool(commit.reconcile, integration.store, integration.indicators, current, aid)
        return integration.store.public_authoring(aid, current, revision)

    @router.get('/api/custom-indicators/authorings/{aid}/previews/{pid}')
    async def read_preview(aid: str, pid: str, request: Request):
        principal = identity.user(request)
        current = await integration.current_principal(principal['sub'], principal['workspace'])
        identity.permit(current, 'research:read')
        integration.store.read_authoring(aid, current)
        return integration.store.preview(aid, pid)

    @router.get('/api/research/scenario-artifacts/{oid}')
    async def read_scenario_artifact(oid: str, request: Request):
        principal = identity.user(request)
        current = await integration.current_principal(principal['sub'], principal['workspace'])
        identity.permit(current, 'scenario:research')
        return integration.store.scenario_artifact(oid, current)

    @router.post('/api/custom-indicators/authorings/{aid}/confirmations')
    async def confirmation(aid: str, body: ConfirmationInput, request: Request):
        principal = identity.user(request)
        record = integration.store.context(body.context_ref, subject=principal['sub'], workspace=principal['workspace'])
        _, current = await integration.authorize(identity.application_id(), principal['sub'], integration.public_context(record), action='save')
        identity.permit(current, 'indicator:save')
        return await run_in_threadpool(commit.preview, integration.store, integration.indicators, current, aid, record,
                                        body.expected_revision, body.definition_hash, body.target)

    @router.post('/api/custom-indicators/authorings/{aid}/commits')
    async def publish(aid: str, body: CommitInput, request: Request):
        principal = identity.user(request)
        current = await integration.current_principal(principal['sub'], principal['workspace'])
        identity.permit(current, 'indicator:save')
        integration.store.read_authoring(aid, current)
        with integration.store.db() as db:
            existing = db.execute('SELECT 1 FROM commits WHERE subject=? AND workspace=? AND request_id=? AND authoring_id=?',
                                  (current['sub'], current['workspace'], body.request_id, aid)).fetchone()
        if existing:
            return await run_in_threadpool(commit.publish, integration.store, integration.indicators, current, aid, body.model_dump())
        record = integration.store.context(body.context_ref, subject=principal['sub'], workspace=principal['workspace'])
        _, current = await integration.authorize(identity.application_id(), principal['sub'], integration.public_context(record), action='save')
        identity.permit(current, 'indicator:save')
        with integration.store.db() as db:
            row = db.execute('SELECT body FROM confirmations WHERE id=? AND authoring_id=?', (body.confirmation_id, aid)).fetchone()
            original = json.loads(row[0]) if row else {}
        if original.get('context_ref') != body.context_ref:
            raise ResearchError('CONFIRMATION_STALE', '保存上下文与确认不一致。', status_code=409)
        # The original repository atomically checks the preview's catalog revision with its creation write.
        return await run_in_threadpool(commit.publish, integration.store, integration.indicators, current, aid, body.model_dump())

    @router.get('/api/custom-indicators/authorings/{aid}/commits/{rid}')
    async def read_commit(aid: str, rid: str, request: Request):
        principal = identity.user(request)
        current = await integration.current_principal(principal['sub'], principal['workspace'])
        identity.permit(current, 'indicator:save')
        integration.store.read_authoring(aid, current)
        with integration.store.db() as db:
            row = db.execute('SELECT body FROM commits WHERE subject=? AND workspace=? AND request_id=? AND authoring_id=?',
                             (current['sub'], current['workspace'], rid, aid)).fetchone()
        if not row:
            raise ResearchError('COMMIT_NOT_FOUND', '尚无该保存回执，不能据此重复创建。', status_code=404)
        return json.loads(row[0])

    app.include_router(router)
    return integration

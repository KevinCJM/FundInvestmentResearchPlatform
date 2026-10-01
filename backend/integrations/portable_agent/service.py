"""External-agent transport to the existing research business services."""
import asyncio
import fcntl
import copy
import json
import os
import sys
import time
from contextlib import contextmanager
from datetime import date
from pathlib import Path

import httpx
from starlette.concurrency import run_in_threadpool

from research_access import data_policy, identity, tools, scenarios
from research_access.authoring import stable_hash, stable_json
from research_access.catalog import build_catalog
from research_access.contracts import PageContext, ResearchError
from research_access.scopes import allowed_tools, scope_for_page, validate_page_context
from research_access.store import ResearchStore
from .catalog import PAGE_PATHS
from .contracts import AgentRelease

TOOL_NAMES = {name.replace('.', '_'): name for name in tools.TOOL_REGISTRY}
REQUIRED_CAPABILITIES = ['profiles', 'turn-context', 'draft-tools', 'history-pagination', 'memory', 'compaction',
                         'handoff', 'host-admission', 'deferred-operations', 'artifact-slots', 'ag-ui']


def release_contract():
    path = os.environ.get('PORTABLE_AGENT_RELEASE_FILE')
    if not path:
        if os.environ.get('APP_ENV') == 'production':
            raise ResearchError('ASSISTANT_RELEASE_REQUIRED', '助手尚未配置已锁定的发布版本。', status_code=503)
        return {'required_capabilities': REQUIRED_CAPABILITIES, 'expected_release': None}
    try:
        release = AgentRelease.model_validate_json(Path(path).read_text()).model_dump()
        if not set(REQUIRED_CAPABILITIES).issubset(release['required_capabilities']):
            raise ValueError('Missing capabilities')
        if os.environ.get('PORTABLE_AGENT_IMAGE', release['image_digest']) != release['image_digest']:
            raise ValueError('Image differs from the release lock')
    except (OSError, ValueError):
        raise ResearchError('ASSISTANT_RELEASE_INVALID', '助手发布配置不完整或不兼容。', status_code=503) from None
    return {'required_capabilities': release['required_capabilities'],
            'expected_release': {key: release[key] for key in ('source_commit', 'protocol_major', 'widget_version', 'manifest_sha256')}}


def permission(name):
    if name.startswith('scenarios.'):
        return 'scenario:research'
    return 'indicator:draft' if tools.TOOL_REGISTRY[name].progress == 'draft' else 'research:read'


class HostIntegration:
    def __init__(self, indicator_service, page_services, *, store=None, authority=None):
        self.indicators, self.pages = indicator_service, page_services
        self._store, self.authority = store, authority
        self.tasks = {}
        self.slots = asyncio.Semaphore(4)
        self.started = False
        self.owner_lock = None

    @property
    def store(self):
        if self._store is None:
            self._store = ResearchStore()
        if not self.started:
            handle = (self._store.directory / 'operations.lock').open('a')
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                handle.close()
                raise ResearchError('ASSISTANT_SINGLE_EXECUTOR', '业务适配仅允许一个执行进程。', status_code=503) from None
            try:
                self._store.recover()
            except BaseException:
                handle.close()
                raise
            self.owner_lock = handle
            self.started = True
        return self._store

    async def current_principal(self, subject, workspace):
        if self.authority:
            value = self.authority(subject, workspace)
            if hasattr(value, '__await__'):
                value = await value
        elif os.environ.get('PORTABLE_AGENT_AUTH_MODE') == 'local':
            if subject != identity.required('PORTABLE_AGENT_LOCAL_OWNER') or workspace != os.environ.get('PORTABLE_AGENT_LOCAL_WORKSPACE', 'local'):
                raise ResearchError('FORBIDDEN', '本机主体或工作空间不匹配。', status_code=403)
            value = {'sub': subject, 'workspace': workspace, 'scopes': ['*']}
        else:
            endpoint = identity.required('PORTABLE_AGENT_AUTHORITY_URL')
            if not endpoint.startswith('https://'):
                raise ResearchError('ASSISTANT_NOT_CONFIGURED', '生产权限核验地址必须使用HTTPS。', status_code=503)
            async with httpx.AsyncClient(timeout=10, trust_env=False, follow_redirects=False) as client:
                response = await client.post(endpoint, json={'subject': subject, 'workspace': workspace},
                    headers={'Authorization': 'Bearer '+identity.required('PORTABLE_AGENT_AUTHORITY_TOKEN', 32)})
            if response.status_code != 200 or len(response.content) > 8192:
                raise ResearchError('FORBIDDEN', '当前业务权限无法核验。', status_code=403)
            value = response.json()
        if not isinstance(value, dict) or value.get('sub') != subject or value.get('workspace') != workspace or not isinstance(value.get('scopes'), list) or not all(isinstance(s, str) for s in value['scopes']):
            raise ResearchError('FORBIDDEN', '当前业务权限无效。', status_code=403)
        return value

    def versions(self, page):
        if page.context_kind == 'platform':
            return {'catalog_version': 'navigation-v1', 'data_generation': None}
        from custom_indicators.series_provider import market_data_generation
        catalog_version = build_catalog(self.indicators)['version']
        if page.context_kind == 'scenario':
            from historical_regimes.v2_registry import REGISTRY_VERSION
            from historical_regimes.v2_contracts import parse_definition_v2
            from market_data import MarketDataManifestError, read_active_manifest
            from custom_indicators.errors import IndicatorDomainError
            from backend.custom_indicators.errors import IndicatorDomainError as BackendDomainError
            from backend.data_storage import StorageError
            catalogs = {}
            # ponytail: hash current repository metadata; add durable generations if catalog size makes this costly.
            try:
                if graph := self.pages.get('graph'):
                    catalogs['definitions'] = graph.list_definitions()
                    for item in catalogs['definitions']:
                        parse_definition_v2(item)
                    catalogs['events'] = graph.event_library.list(archived=True, limit=sys.maxsize)
                if stress := self.pages.get('stress'):
                    from scenario_stress.contracts import normalize_definition
                    catalogs['stress'] = stress.list_definitions()
                    for item in catalogs['stress']:
                        normalize_definition(item)
                if published := self.pages.get('published'):
                    try:
                        as_of = date.fromisoformat(page.calculation.as_of) if page.calculation.as_of else None
                    except ValueError:
                        raise ResearchError('VALIDATION_ERROR', '研究日期格式无效，应为 YYYY-MM-DD。', status_code=422) from None
                    catalogs['releases'] = published.releases(as_of)
                if sources := self.pages.get('sources'):
                    catalogs['sources'] = {'snapshot': read_active_manifest(sources.data_dir),
                                           'indicators': sources._indicator_versions()}
            except (IndicatorDomainError, BackendDomainError) as exc:
                # Date is the only request input to these catalog reads; other errors describe stored evidence.
                if exc.code == 'FUTURE_RESEARCH_DATE':
                    raise ResearchError(exc.code, exc.message, status_code=exc.status_code) from exc
                raise ResearchError('RESEARCH_SERVICE_UNAVAILABLE', '情景目录暂不可读取，请检查业务数据后重试。', status_code=503) from exc
            except (MarketDataManifestError, StorageError, OSError, KeyError, TypeError, AttributeError, ValueError) as exc:
                raise ResearchError('RESEARCH_SERVICE_UNAVAILABLE', '情景目录暂不可读取，请检查业务数据后重试。', status_code=503) from exc
            catalog_version = stable_hash([catalog_version, REGISTRY_VERSION, page.calculation.workspace, catalogs])
        return {'catalog_version': catalog_version,
                'data_generation': market_data_generation(self.indicators.market_data_dir)}

    async def register(self, principal, body, *, pit_off=False, derived_from=None, view_headers=None, frozen_pit=None):
        current = await self.current_principal(principal['sub'], principal['workspace'])
        identity.permit(current, 'assistant:use')
        scope = validate_page_context(body.page_context, pit_off=pit_off, allow_authoring=True)
        page = body.page_context.model_copy(deep=True)
        if body.capability_id is not None and body.capability_id != page.page:
            raise ResearchError('CAPABILITY_CONTEXT_MISMATCH', '任务能力与实际页面不匹配。', status_code=403)
        pit = frozen_pit
        if page.context_kind in {'single_product','scenario'} and page.view_state != 'unknown':
            from pit.context import resolve_request_context, parse_view_override, set_view_override, reset_view_override, build_context
            override = build_context(**frozen_pit) if frozen_pit else parse_view_override(self.indicators.market_data_dir, view_headers or ({'x-pit-off':'1'} if pit_off else {}))
            token = set_view_override(override)
            try:
                resolved = await run_in_threadpool(resolve_request_context, self.indicators.market_data_dir, page.calculation.as_of)
            finally:
                reset_view_override(token)
            page.calculation.as_of = resolved.as_of
            pit = {'as_of': resolved.as_of, 'run_mode': resolved.run_mode, 'data_release_id': resolved.data_release_id}
        names = [name for name in allowed_tools(scope, page.context_kind) if '*' in current['scopes'] or permission(name) in current['scopes']]
        if page.context_kind == 'scenario' and page.calculation.purpose == 'event_library':
            names = [name for name in names if name in {'scenarios.catalog', 'page.read'}]
        if scope != 'platform' and not names:
            raise ResearchError('FORBIDDEN', '当前页面没有可用业务权限。', status_code=403)
        versions = await run_in_threadpool(self.versions, page)
        scope_key = stable_hash([principal['sub'], principal['workspace'], page.page, page.page_instance_id])
        authoring = None
        if page.context_kind == 'single_product':
            authoring = (self.store.read_authoring(body.authoring_id, current) if body.authoring_id else
                         self.store.authoring(current, 'page-'+scope_key, scope=scope, context_hash=''))
        record = {'page_context': page.model_dump(), 'page_snapshot': body.page_snapshot.model_dump() if body.page_snapshot else None,
            'requested_page_context': body.page_context.model_dump(), 'derived_from': derived_from,
            'pit': pit,
            'scope': scope, 'scope_key': scope_key, 'memory_scope': scope, 'tool_names': names,
            'authoring_id': authoring['id'] if authoring else None,
            'capability_id': page.page, 'intent_parameters': body.intent_parameters, **versions}
        return self.store.register_context(current, record)

    @staticmethod
    def public_context(record):
        return {'ref': record['id'], 'hash': record['hash'], 'grant_id': record['grant_id'], 'grant_revision': record['grant_revision'],
            'workspace': record['workspace'], 'scope_key': record['scope_key'], 'memory_scope': record['memory_scope'],
            'capability_id': record.get('capability_id'), 'intent_parameters': record.get('intent_parameters', {}),
            'authoring_id': record.get('authoring_id'),
            'derived_from': record.get('derived_from'),
            'memory_objects': [item['product_id'] for item in record['page_context']['calculation'].get('targets', [])],
            'page': record['page_context']['page'],
            'summary': {'calculation': record['page_context']['calculation'], 'view_state': record['page_context']['view_state'],
                        'catalog_version': record['catalog_version']}}

    async def authorize(self, application, subject, context, *, action='run', operation_id=None, session_id=None, run_id=None):
        if application != identity.application_id():
            raise ResearchError('APPLICATION_FORBIDDEN', '应用不匹配。', status_code=403)
        record = self.store.context(context.get('ref'), subject=subject, workspace=context.get('workspace'))
        if stable_json(context) != stable_json(self.public_context(record)):
            raise ResearchError('CONTEXT_FORBIDDEN', '授权上下文被修改。', status_code=403)
        current = await self.current_principal(subject, record['workspace'])
        identity.permit(current, 'assistant:use')
        if action not in {'read', 'cancel'} and (record['revoked'] or record['expires'] <= time.time()):
            raise ResearchError('CONTEXT_EXPIRED', '页面授权已失效，请重新登记。', status_code=403)
        # A frozen context also authorizes its signed history. Partial revocation invalidates the whole grant.
        if record['scope'] != 'platform' and not all('*' in current['scopes'] or permission(n) in current['scopes'] for n in record['tool_names']):
            raise ResearchError('FORBIDDEN', '当前权限已撤销。', status_code=403)
        if action == 'adopt':
            operation = self.store.operation(operation_id)
            if (record.get('derived_from') != operation_id or operation['status'] != 'succeeded' or operation.get('cancel_requested')
                    or operation['subject'] != subject or operation['workspace'] != record['workspace']
                    or operation['payload']['session_id'] != session_id or operation['payload']['run_id'] != run_id):
                raise ResearchError('ADOPTION_STALE', '程序回填没有本轮有效业务回执。', status_code=409)
        return record, current

    async def bootstrap(self, principal, cid):
        record = self.store.context(cid, subject=principal['sub'], workspace=principal['workspace'])
        public = self.public_context(record)
        _, current = await self.authorize(identity.application_id(), principal['sub'], public, action='read')
        record = self.store.renew_context(cid, current)
        scopes = ['agent:chat', *['tool:'+name.replace('.', '_') for name in record['tool_names'] if '*' in current['scopes'] or permission(name) in current['scopes']]]
        if '*' in current['scopes'] or 'settings:write' in current['scopes']:
            scopes.append('settings:write')
        return {'endpoint': '/assistant', 'module_url': '/assistant/widget/widget.js', 'app': identity.application_id(),
                **release_contract(),
                'principal_id': stable_hash([identity.application_id(), current['sub']])[:32],
                'protocol_major': 2, 'context': public, 'token': identity.agent_token(current, record, scopes), 'expires_in': 900}

    @contextmanager
    def business_context(self, record):
        from pit.context import build_context, set_view_override, reset_view_override
        page = PageContext.model_validate(record['page_context'])
        token = None
        try:
            if record.get('pit'):
                token = set_view_override(build_context(**record['pit']))
            yield page
        finally:
            if token is not None:
                reset_view_override(token)

    async def submit(self, name, body):
        if name not in TOOL_NAMES:
            raise ResearchError('TOOL_FORBIDDEN', '业务工具未注册。', status_code=403)
        canonical = TOOL_NAMES[name]
        record, principal = await self.authorize(body.application, body.subject, body.context, action='tool')
        identity.permit(principal, permission(canonical))
        if canonical not in record['tool_names']:
            raise ResearchError('TOOL_FORBIDDEN', '工具不属于当前页面。', status_code=403)
        tools.parse_arguments(canonical, body.arguments)
        data_policy.check(body.arguments)
        data_policy.check_text_fields(body.arguments)
        operation, accepted = self.store.start_operation(principal, body.model_dump(), canonical, record['scope_key'])
        if accepted:
            task = asyncio.create_task(self.execute(operation, record, principal))
            self.tasks[body.operation_id] = task
            task.add_done_callback(lambda done: self.tasks.pop(body.operation_id, None))
        return self.public_operation(operation)

    def calculate(self, operation, record, principal):
        payload = operation['payload']
        authoring = (self.store.read_authoring(record['authoring_id'], principal) if record.get('authoring_id') else
                     self.store.authoring(principal, 'page-'+record['scope_key'], scope=record['scope'], context_hash=record['hash']))
        authoring['context_hash'] = record['hash']
        expected = authoring['revision']
        before = copy.deepcopy(authoring)
        with self.business_context(record) as page:
            version = self.versions(page)
            tool = tools.get_tool(operation['name'])
            if version['catalog_version'] != record['catalog_version'] or (tool.uses_current_data(page, payload['arguments']) and version['data_generation'] != record['data_generation']):
                raise ResearchError('CONTEXT_CHANGED', '数据或目录已改变，请重新发送。', status_code=409)
            if operation['name'].startswith('scenarios.'):
                from custom_indicators.errors import IndicatorDomainError
                from backend.custom_indicators.errors import IndicatorDomainError as BackendDomainError
                from research_series.service import ResearchSeriesError
                from backend.research_series.service import ResearchSeriesError as BackendSeriesError
                try:
                    result = scenarios.execute(operation['name'], tools.parse_arguments(operation['name'], payload['arguments']), page, record['page_snapshot'], self.pages,
                        checkpoint=lambda **fields: self.store.update_operation(operation['operation_id'], **fields),
                        cancelled=lambda: self.store.operation(operation['operation_id']).get('cancel_requested', False))
                except (IndicatorDomainError, BackendDomainError, ResearchSeriesError, BackendSeriesError) as exc:
                    raise ResearchError(exc.code, exc.message, status_code=exc.status_code) from exc
            else:
                result = tools.execute_business(operation['name'], payload['arguments'], authoring=authoring, page_context=page,
                    service=self.indicators, page_snapshot=record['page_snapshot'], previews=self.store, page_services=self.pages)
            if tool.progress == 'draft' and authoring.get('draft'):
                authoring['draft']['source_run_id'] = payload['run_id']
            after = self.versions(page)
        changed = before != {k: v for k, v in authoring.items() if not k.startswith('_')}
        return authoring, expected, result, after, tool.uses_current_data(page, payload['arguments']), changed

    async def execute(self, operation, record, principal):
        oid = operation['operation_id']
        async with self.slots:
            operation = self.store.claim_operation(oid)
            if operation is None:
                return
            try:
                authoring, expected, result, versions, current_data, changed = await run_in_threadpool(self.calculate, operation, record, principal)
                with self.store.db() as db:
                    live = self.store.operation(oid, db=db)
                    stale = versions['catalog_version'] != record['catalog_version'] or (current_data and versions['data_generation'] != record['data_generation'])
                    if stale or live.get('cancel_requested'):
                        projection = {'ok': False, 'status': 'stale' if stale else 'cancelled', 'message': '操作已结束，旧结果不再采纳。'}
                        artifact = None
                    else:
                        scenario = result.get('_scenario_payload')
                        full = authoring.pop('_preview_payload', None)
                        if full:
                            full.update(context_hash=record['hash'], run_id=operation['payload']['run_id'], data_generation=record['data_generation'], catalog_version=record['catalog_version'])
                        saved = self.store.save_authoring(authoring, expected, preview=full, db=db) if changed or full else authoring
                        # The domain handler's _progress_payload can contain full market arrays. It never crosses the host boundary.
                        projection = {k: v for k, v in result.items() if not k.startswith('_')}
                        artifact = {'type': 'research.indicator', 'group': 'indicator:'+saved['id'], 'title': '查看指标草稿与试算', 'authoring_id': saved['id'],
                            'revision': saved['revision'], 'source_run_id': saved['draft'].get('source_run_id'),
                            'preview_id': (saved.get('preview') or {}).get('preview_id')} if (saved.get('draft') or {}).get('valid') and (full or tools.get_tool(operation['name']).progress == 'draft') else None
                        if scenario:
                            self.store.save_scenario_artifact(oid, principal, {**scenario, 'context_ref': record['id'], 'source_run_id': operation['payload']['run_id']}, db=db)
                            artifact = {'type': 'research.scenario', 'group': 'scenario:'+record['scope_key'], 'title': '查看情景候选与试算', 'artifact_id': oid}
                    marker = stable_hash({'name': operation['name'], 'arguments': operation['payload']['arguments'], 'result': projection})
                    projection = data_policy.seal({**projection, 'context_ref': 'op-'+oid, 'research_context_ref': record['id']}, operation['name'])
                    envelope = {'model': projection, 'progress_token': marker}
                    if artifact:
                        envelope['artifact'] = artifact
                    self.store.update_operation(oid, db=db, status='cancelled' if live.get('cancel_requested') else 'succeeded', result=envelope, current_data=current_data)
            except ResearchError as exc:
                # These errors prove no business side effect began; other post-dispatch failures stay uncertain.
                terminal = tools.get_tool(operation['name']).progress in {None, 'read'} or exc.code in {'AGENT_DRAFT_REQUIRED', 'AGENT_PREVIEW_TARGET_REQUIRED', 'AGENT_TOOL_DOMAIN_MISMATCH',
                    'AGENT_CONTEXT_CHANGED', 'CONTEXT_CHANGED', 'REVISION_CONFLICT', 'VALIDATION_ERROR',
                    'SCENARIO_PURPOSE_MISMATCH', 'SCENARIO_DEFINITION_REQUIRED', 'SCENARIO_AUTHORING_INVALID',
                    'RAW_DATA_FORBIDDEN', 'TOOL_DOMAIN_MISMATCH', 'RESEARCH_SERVICE_UNAVAILABLE',
                    'AGENT_PAGE_EVIDENCE_REQUIRED', 'AGENT_PAGE_SECTION_UNAVAILABLE',
                    'AGENT_PAGE_DEFINITION_UNAVAILABLE', 'AGENT_PAGE_TARGET_UNAVAILABLE',
                    'AGENT_PAGE_REQUEST_UNAVAILABLE', 'AGENT_PAGE_REQUEST_INVALID',
                    'AGENT_PAGE_PARAMETERS_INVALID', 'AGENT_PAGE_DEFINITION_INVALID',
                    'AGENT_PAGE_FEES_CHANGED', 'AGENT_PAGE_SERVICE_UNAVAILABLE',
                    'AGENT_PAGE_OPERATION_NOT_ALLOWED', 'AGENT_PORTFOLIO_RUN_INVALID', 'AGENT_SCENARIO_REQUEST_REQUIRED',
                    'AGENT_PRODUCT_UNAVAILABLE', 'AGENT_INDICATOR_VERSION_CHANGED',
                    'AGENT_DEMO_NOT_EVIDENCE', 'AGENT_ROLLING_WINDOW_AMBIGUOUS'}
                fields = {'status': 'failed' if terminal else 'unknown', 'error': exc.detail()}
                if terminal:
                    fields['result'] = {'model': data_policy.seal({'ok': False, 'code': exc.code, 'message': exc.message,
                        'context_ref': 'op-'+oid, 'research_context_ref': record['id']}, operation['name'])}
                self.store.update_operation(oid, **fields)
            except Exception:
                self.store.update_operation(oid, status='unknown', error={'code': 'OPERATION_UNCERTAIN', 'message': '业务结果尚待核对。'})

    @staticmethod
    def public_operation(operation):
        value = {k: operation[k] for k in ('operation_id', 'status')}
        if operation.get('result'):
            value.update(operation['result'])
        if operation.get('error'):
            value['error'] = operation['error']
        if operation['status'] in {'accepted', 'running', 'stop_requested'}:
            value['retry_after_ms'] = 250
        return value

    def cancel(self, oid):
        return self.public_operation(self.store.cancel_operation(oid))

    async def close(self):
        for oid in list(self.tasks):
            self.cancel(oid)
        if self.tasks:
            await asyncio.gather(*list(self.tasks.values()), return_exceptions=True)
        if self.owner_lock:
            self.owner_lock.close()
            self.owner_lock = None
            self.started = False

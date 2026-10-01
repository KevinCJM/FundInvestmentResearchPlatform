"""Scenario business tools: editable definitions and real services, never an agent executor."""
import copy
import time
from datetime import date
from typing import Literal

from pydantic import Field, ValidationError as SchemaError, field_validator

from .contracts import Contract, ResearchError
from . import data_policy, views
from .scenario_contracts import Definition, ReadArgs, authoring_definition, saved_definition_view, catalog_view, page_view, _search_text


class CatalogArgs(Contract):
    section: Literal['definitions', 'sources', 'nodes', 'templates', 'events', 'releases'] = 'definitions'
    query: str = Field(default='', max_length=100)
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=5, ge=1, le=10)


class TemplateArgs(Contract):
    template_id: str = Field(min_length=1, max_length=120)


class DefinitionArgs(Contract):
    definition: dict | None = None

    @field_validator('definition')
    @classmethod
    def graph_contract(cls, value):
        if isinstance(value, dict) and 'graph' in value:
            Definition.model_validate(value)
        return value


def required(services, name):
    value = (services or {}).get(name)
    if value is None:
        raise ResearchError('RESEARCH_SERVICE_UNAVAILABLE', '该情景业务服务尚未挂载。', status_code=503)
    return value


def safe_definition(value):
    data_policy.check(value)
    data_policy.check_text_fields(value)
    for node in (value.get('graph') or {}).get('nodes', []):
        if node.get('type') == 'source.inline':
            raise ResearchError('RAW_DATA_FORBIDDEN', '助手只可引用已登记数据，不能内联原始数值序列。', status_code=422)
    return value


def catalog(services, args, workspace, as_of=None):
    if args.section == 'sources':
        listing = required(services, 'sources').catalog(query=args.query, offset=args.offset, limit=args.limit)
        total = listing.get('total', len(listing['items']))
        result, dropped = catalog_view({'kind': 'sources', **listing,
            'next_offset': args.offset+args.limit if args.offset+args.limit < total else None}, views.Projection())
        return {'ok': True, 'result': result, 'truncated': bool(dropped)}
    if args.section == 'definitions' and workspace in {'graph', 'events'}:
        listing = {'items': required(services, 'graph').list_definitions()}
        names = {'id', 'revision', 'name', 'description', 'default_mode'}
    elif args.section == 'events':
        listing = required(services, 'graph').event_library.list(query=args.query, offset=args.offset, limit=args.limit)
        original_count = len(listing['items'])
        if as_of:
            listing['items'] = [item for item in listing['items'] if item.get('known_at') and str(item['known_at'])[:10] <= as_of]
        names = {'id','name','name_en','description','revision','verification','known_at','fact_start','fact_end','categories','regions','windows','sources'}
    elif args.section == 'releases':
        listing = {'items': required(services, 'published').releases(date.fromisoformat(as_of) if as_of else None).get('items', [])}
        names = {'id','name','entry','frequency','horizon','status','reason','created_at','expires_at'}
    elif workspace == 'stress':
        listing = {'items': required(services, 'stress').list_definitions()}
        names = {'id','revision','name','description','method','horizon','usage_intent'}
    else:
        listing = required(services, 'graph').templates() if args.section == 'templates' else required(services, 'graph').catalog()
        names = {'id','version','label','name','description','category','category_label','inputs','outputs','parameter_schema','temporal_contract','granularity','content_hash'}
    items = [{k: v for k, v in item.items() if k in names} for item in listing.get('items', [])]
    if args.section != 'events':
        items = [item for item in items if _search_text(args.query) in _search_text(str(item))]
        total = len(items); items = items[args.offset:args.offset+args.limit]
    else:
        total = listing['total']
    data_policy.check(items)
    next_offset = args.offset+(original_count if args.section == 'events' else len(items))
    return {'ok': True, 'result': {'items': items, 'total': total if args.section != 'events' else None,
        'as_of': as_of, 'next_offset': next_offset if next_offset<total else None}}


def execute(name, args, context, snapshot, services, *, checkpoint, cancelled):
    workspace = context.calculation.workspace
    def qualify(definition):
        current = (((snapshot or {}).get('sections') or {}).get('editing') or {}).get('definition') or {}
        if context.calculation.purpose in {'historical_reference', 'realtime_recognition'}:
            definition['study'] = {k: v for k, v in (current.get('study') or {
                'purpose': context.calculation.purpose, 'family': 'custom'}).items()
                if k not in {'calibration_id', 'qualification_id'}}
            if (definition.get('study') or {}).get('purpose') != context.calculation.purpose:
                raise ResearchError('SCENARIO_PURPOSE_MISMATCH', '候选定义用途与当前工作区不一致。', status_code=422)
        return definition
    if name == 'scenarios.catalog':
        return catalog(services, args, workspace, context.calculation.as_of)
    if name == 'scenarios.read':
        if workspace not in {'graph', 'events'}:
            raise ResearchError('TOOL_DOMAIN_MISMATCH', '该工作区不使用历史情景图。', status_code=409)
        definition = required(services, 'graph').get_definition(args.definition_id, revision=args.revision)
        graph = definition.get('graph') or {}
        nodes = graph.get('nodes', [])
        selected = nodes[args.offset:args.offset+args.limit]
        selected_ids = {node['id'] for node in selected}
        result, dropped = saved_definition_view({**definition, 'items': selected, 'total': len(nodes),
            'outputs': graph.get('outputs', {}),
            'edges': [e for e in graph.get('edges', []) if e.get('target', {}).get('node_id') in selected_ids],
            'next_offset': args.offset+args.limit if args.offset+args.limit < len(nodes) else None}, views.Projection())
        data_policy.check_text_fields(result)
        return {'ok': True, 'result': result, 'truncated': bool(dropped)}
    if name == 'scenarios.template':
        if workspace not in {'graph','events'}:
            raise ResearchError('TOOL_DOMAIN_MISMATCH', '该工作区不使用历史情景图模板。', status_code=409)
        value = required(services, 'graph').instantiate_template(args.template_id)
        try:
            args = DefinitionArgs(definition=authoring_definition(value['definition']))
        except (ValueError, TypeError):
            raise ResearchError('SCENARIO_AUTHORING_INVALID', '该模板含复合参数或人工事件，请在编辑器配置。', status_code=422) from None
    if args.definition is None:
        definition = copy.deepcopy((((snapshot or {}).get('sections') or {}).get('editing') or {}).get('definition'))
        if not isinstance(definition, dict):
            raise ResearchError('SCENARIO_DEFINITION_REQUIRED', '页面没有冻结定义，请先选择或构建情景。', status_code=422)
    else:
        definition = safe_definition(copy.deepcopy(args.definition))
        if workspace in {'graph', 'events'}:
            try:
                definition = Definition.model_validate(definition).model_dump(exclude_unset=True)
            except SchemaError:
                raise ResearchError('SCENARIO_AUTHORING_INVALID', '只能修改已登记图节点和单值参数；研究身份与资格由工作台管理。', status_code=422) from None
    definition = qualify(definition)
    if workspace in {'graph','events'}:
        service = required(services, 'graph')
        inference = service.infer(definition, context.calculation.mode)
        valid = bool(inference.get('valid'))
        if context.calculation.mode == 'realtime' and not (inference.get('temporal_capability') or {}).get('realtime_supported'):
            valid = False
        result = {'valid': valid, 'diagnostics': [*inference.get('errors', []), *inference.get('warnings', [])], 'validation_scope': 'graph',
                  'temporal_capability': inference.get('temporal_capability')}
        if name == 'scenarios.template':
            result.update(template_id=value['template_id'], template_version=value['template_version'], definition=authoring_definition(definition))
        artifact = {'definition': definition, **result, 'workspace': workspace}
        if name == 'scenarios.preview' and valid:
            prepared = service.prepare(definition)
            job = service.create_preview(definition, compile_token=prepared['compile_token'], mode=context.calculation.mode, as_of=context.calculation.as_of)
            checkpoint(external_job_id=job['id'], external_service='graph')
            stop_sent = False
            while True:
                if cancelled() and not stop_sent:
                    service.cancel_preview(job['id']); stop_sent = True
                current = service.get_preview(job['id'])
                if current.get('execution_finished'):
                    if current['status'] == 'completed':
                        result.update(status='completed', preview_id=job['id'])
                        artifact.update(result=current, preview_id=job['id'])
                    else:
                        result.update(status=current['status'], error=current.get('error'))
                    break
                time.sleep(.1)
    elif workspace == 'published':
        from backend.scenario_stress.published_contracts import ScenarioFields
        try:
            definition = ScenarioFields.model_validate(definition).model_dump(mode='json')
        except SchemaError:
            return {'ok': False, 'result': {'valid': False, 'code': 'SCENARIO_SCHEMA_INVALID', 'message': '情景参数结构不符合业务契约。'}}
        result = {'valid': True, 'validation_scope': 'parameter_schema', 'horizon': len(definition['rows'])}
        artifact = {'definition': definition, **result, 'workspace': workspace}
        if name == 'scenarios.preview':
            value = required(services, 'published').preview(definition)
            result.update(status='completed', preview_id=value['id'])
            artifact.update(result=value, preview_id=value['id'])
    elif workspace == 'stress':
        from scenario_stress.contracts import normalize_definition
        try:
            definition = normalize_definition(definition)
        except Exception as exc:
            if not isinstance(getattr(exc, 'code', None), str):
                raise
            return {'ok': False, 'result': {'valid': False, 'code': exc.code, 'message': '情景定义未通过业务校验。'}}
        result = {'valid': True, 'validation_scope': 'parameter_schema', 'method': definition['method'], 'horizon': definition['horizon']}
        artifact = {'definition': definition, **result, 'workspace': workspace}
        if name == 'scenarios.preview':
            value = required(services, 'stress').run(definition)
            result.update(status='completed', run_id=value['id'])
            artifact.update(result=value, run_id=value['id'])
    else:
        raise ResearchError('TOOL_DOMAIN_MISMATCH', '未知情景工作区。', status_code=409)
    data_policy.check(result)
    data_policy.check_text_fields(result)
    return {'ok': True, 'result': result, '_scenario_payload': artifact}

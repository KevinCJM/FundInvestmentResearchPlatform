"""Deterministic host data admission. This module never calls a model."""
import copy
import json
import time
import uuid

from research_access import data_policy, tools
from research_access.authoring import stable_hash, stable_json
from research_access.contracts import PageContext, ResearchError
from .service import TOOL_NAMES

INTERNAL_TOOLS = {'agent_task_read', 'agent_task_plan', 'agent_memory_propose', 'agent_intent_resolve', 'agent_capabilities', 'agent_history_read'}
STOP_NOTICE = 'Operation did not produce a confirmed result; consult the original receipt.'


def schema_contract(schema):
    """Compare local JSON-schema references after provider inlining, retaining actual constraints/defaults."""
    def normalize(value, seen=()):
        if isinstance(value, list):
            return [normalize(item, seen) for item in value]
        if not isinstance(value, dict):
            return value
        if '$ref' in value:
            reference = value['$ref']
            if not isinstance(reference, str) or not reference.startswith('#/') or reference in seen:
                raise ResearchError('MODEL_INPUT_REJECTED', '工具契约引用无效。', status_code=422)
            target = schema
            try:
                for key in reference[2:].split('/'):
                    target = target[key.replace('~1', '/').replace('~0', '~')]
            except (KeyError, TypeError):
                raise ResearchError('MODEL_INPUT_REJECTED', '工具契约引用不存在。', status_code=422) from None
            return normalize({**target, **{k: v for k, v in value.items() if k != '$ref'}}, (*seen, reference))
        return {key: normalize(item, seen) for key, item in value.items()
                if key not in {'$defs', 'definitions'} and not (key == 'title' and isinstance(item, str))}
    return normalize(schema)


def role_message(message):
    if 'role' in message:
        return message
    data = message.get('data', {})
    role = {'human': 'user', 'ai': 'assistant', 'system': 'system', 'tool': 'tool'}.get(message.get('type'))
    if role is None:
        raise ResearchError('MODEL_INPUT_REJECTED', '消息角色无效。', status_code=422)
    result = {'role': role, 'content': data.get('content', '')}
    if data.get('tool_calls'):
        result['tool_calls'] = [{'id': c['id'], 'type': 'function', 'function': {'name': c['name'], 'arguments': stable_json(c['args'])}} for c in data['tool_calls']]
    if role == 'tool':
        result['tool_call_id'] = data.get('tool_call_id')
    return result


def validate_messages(messages, record, *, store, versions, prepare=False, run_id=None):
    normalized = [role_message(m) for m in messages]
    non_system = [m for m in normalized if m['role'] != 'system']
    aliases = copy.deepcopy(non_system)
    for message in aliases:
        for call in message.get('tool_calls', []):
            function = call.get('function', {})
            function['name'] = TOOL_NAMES.get(function.get('name'), function.get('name'))
    sources, _ = data_policy.tool_call_sources(aliases)
    sources_by_index = dict(sources)
    output = copy.deepcopy(messages)
    non_system_index = -1
    for index, message in enumerate(normalized):
        role, content = message['role'], message.get('content', '')
        if role == 'assistant' and content is None:
            content = ''  # Chat Completions uses null for assistant tool-call messages.
        if 'role' in messages[index] and set(message) - {'role', 'content', 'name', 'tool_call_id', 'tool_calls', 'reasoning_content'}:
            raise ResearchError('MODEL_INPUT_REJECTED', '消息包含未登记字段。', status_code=422)
        if message.get('reasoning_content'):
            violation = data_policy.embedded_json_violation(message['reasoning_content'])
            if violation:
                raise violation
        if not isinstance(content, str):
            raise ResearchError('MODEL_INPUT_REJECTED', '此次接入仅允许文本消息。', status_code=422)
        if role != 'system':
            non_system_index += 1
        replacement = None
        if role == 'tool':
            name = sources_by_index.get(non_system_index)
            if content == STOP_NOTICE:
                continue
            try:
                payload = json.loads(content)
            except ValueError:
                raise ResearchError('MODEL_INPUT_REJECTED', '工具结果不可核验。', status_code=422) from None
            if name in tools.TOOL_REGISTRY:
                if not data_policy.verify(payload, name):
                    raise ResearchError('MODEL_INPUT_REJECTED', '业务结果缺少有效来源证明。', status_code=422)
                reference = payload.get('context_ref', '')
                if reference.startswith('op-'):
                    operation = store.operation(reference[3:])
                    if operation['subject'] != record['subject'] or operation['workspace'] != record['workspace']:
                        raise ResearchError('MODEL_INPUT_REJECTED', '业务结果不属于当前主体。', status_code=403)
                    source = store.context(payload.get('research_context_ref'), subject=record['subject'], workspace=record['workspace'])
                    stale = source['catalog_version'] != versions['catalog_version'] or (operation.get('current_data') and source['data_generation'] != versions['data_generation'])
                    if stale:
                        replacement = stable_json(data_policy.seal({'ok': False, 'status': 'historical_stale',
                            'message': '旧结果已失效，仅保留历史引用；需要实际结果时按新条件重新查询。'}, name))
            elif name not in INTERNAL_TOOLS:
                raise ResearchError('MODEL_INPUT_REJECTED', '未知工具不能进入模型上下文。', status_code=422)
            data_policy.check(payload)
            # page.read signs an already-projected, paginated JSON text field.
            # Its exact bytes were authenticated above; free text remains subject to admission.
            text_payload = payload
            if name == 'page.read' and isinstance(payload.get('result'), dict):
                text_payload = {**payload, 'result': {k: v for k, v in payload['result'].items() if k != 'content'}}
            data_policy.check_text_fields(text_payload)
        elif role == 'user':
            violation = data_policy.user_text_violation(content)
            if violation:
                message_id = (messages[index].get('data') or {}).get('id')
                if prepare and message_id and run_id and message_id != run_id:
                    replacement = data_policy.STRUCTURED_OMIT_NOTE
                else:
                    raise ResearchError('AGENT_DATA_ADMISSION_BLOCKED', '原始行情或结构化数值数据不能发送给模型。', status_code=422)
        elif role == 'system':
            from .service import HostIntegration
            data_policy.check_system_text(content, context=HostIntegration.public_context(record))
        elif role == 'assistant':
            violation = data_policy.embedded_json_violation(content)
            if violation:
                raise violation
            for call in message.get('tool_calls') or []:
                function = call.get('function', {})
                name = TOOL_NAMES.get(function.get('name'), function.get('name'))
                raw = function.get('arguments')
                if name in tools.TOOL_REGISTRY:
                    if not data_policy.enforce_arguments(name, raw):
                        raise ResearchError('MODEL_INPUT_REJECTED', '历史业务参数不符合注册契约。', status_code=422)
                elif name in INTERNAL_TOOLS:
                    data_policy.check_text_fields(json.loads(raw))
                else:
                    raise ResearchError('MODEL_INPUT_REJECTED', '未注册工具。', status_code=422)
        else:
            raise ResearchError('MODEL_INPUT_REJECTED', '消息角色不支持。', status_code=422)
        if replacement is not None:
            if not prepare:
                raise ResearchError('MODEL_INPUT_REJECTED', '过期证据尚未重新投影。', status_code=422)
            if 'role' in output[index]:
                output[index]['content'] = replacement
            else:
                output[index]['data']['content'] = replacement
    return output


async def admit(integration, body):
    record, _ = await integration.authorize(body.application, body.subject, body.context, action='model')
    from starlette.concurrency import run_in_threadpool
    versions = await run_in_threadpool(integration.versions, PageContext.model_validate(record['page_context']))
    if body.phase == 'after':
        with integration.store.db() as db:
            row = db.execute('SELECT body FROM admissions WHERE id=? AND subject=? AND context_id=?',
                             (body.receipt_id, body.subject, record['id'])).fetchone()
        receipt = json.loads(row[0]) if row else {}
        compared = versions | {'data_generation': versions['data_generation'] if receipt.get('current_data') else None}
        if receipt.get('payload_hash') != body.payload_hash or receipt.get('expires', 0) <= time.time() or receipt.get('versions') != compared:
            raise ResearchError('MODEL_RESULT_STALE', '模型等待期间的数据或目录已改变。', status_code=409)
        return {'allow': True, 'payload_hash': body.payload_hash}
    if body.phase in {'input', 'prepare'}:
        messages = validate_messages(body.messages, record, store=integration.store, versions=versions,
                                     prepare=body.phase == 'prepare', run_id=body.run_id)
        return {'allow': True, 'messages': messages}
    wire = body.wire
    if not isinstance(wire, dict) or stable_hash(wire) != body.payload_hash:
        raise ResearchError('MODEL_INPUT_REJECTED', '实际请求哈希不匹配。', status_code=422)
    allowed_controls = {'model', 'stream', 'stream_options', 'temperature', 'top_p', 'max_tokens', 'max_completion_tokens',
                        'reasoning_effort', 'thinking', 'parallel_tool_calls', 'tool_choice', 'stop', 'seed', 'frequency_penalty', 'presence_penalty', 'n'}
    if set(wire) - allowed_controls - {'messages', 'tools'}:
        raise ResearchError('MODEL_INPUT_REJECTED', '模型请求含未登记的传输字段。', status_code=422)
    controls = {k: v for k, v in wire.items() if k in allowed_controls}
    data_policy.check(controls); data_policy.check_text_fields(controls)
    validate_messages(wire.get('messages', []), record, store=integration.store, versions=versions)
    current_data = record['page_context']['calculation']['context_kind'] == 'single_product'
    for message in wire.get('messages', []):
        if message.get('role') == 'tool':
            try:
                payload = json.loads(message['content'])
            except (ValueError, KeyError):
                continue
            reference = payload.get('context_ref', '') if isinstance(payload, dict) else ''
            if reference.startswith('op-'):
                current_data = current_data or integration.store.operation(reference[3:]).get('current_data', False)
    if record['catalog_version'] != versions['catalog_version'] or (current_data and record['data_generation'] != versions['data_generation']):
        raise ResearchError('CONTEXT_CHANGED', '冻结研究条件已失效，请重新发送。', status_code=409)
    declared = wire.get('tools', [])
    for declaration in declared:
        function = declaration.get('function') or {}
        name = function.get('name')
        if name not in INTERNAL_TOOLS and TOOL_NAMES.get(name) not in record['tool_names']:
            raise ResearchError('MODEL_INPUT_REJECTED', '工具声明越出授权范围。', status_code=403)
        if name in INTERNAL_TOOLS:
            data_policy.check_text_fields(declaration)
        else:
            registered = tools.get_tool(TOOL_NAMES[name])
            if (function.get('description', '').strip() != registered.description.strip()
                    or schema_contract(function.get('parameters', {})) != schema_contract(registered.arguments.model_json_schema())):
                raise ResearchError('MODEL_INPUT_REJECTED', '工具声明不匹配已登记业务契约。', status_code=422)
    rid = uuid.uuid4().hex
    # Only hashes/versions are stored here. Conversation content belongs to the external service.
    receipt = {'payload_hash': body.payload_hash, 'versions': versions | {'data_generation': versions['data_generation'] if current_data else None},
               'current_data': current_data, 'expires': time.time()+1900}
    with integration.store.db() as db:
        db.execute("DELETE FROM admissions WHERE json_extract(body,'$.expires')<?", (time.time(),))
        db.execute('INSERT INTO admissions VALUES(?,?,?,?)', (rid, body.subject, record['id'], stable_json(receipt)))
    return {'allow': True, 'receipt_id': rid, 'payload_hash': body.payload_hash}

"""Offline context-window regressions, using synthetic evidence, never live APIs."""
import asyncio
import copy
import json

import pytest

from agent.context import (compact_if_needed, context_messages, estimate_tokens, input_budget,
                           measure, record_usage, request_tokens)
from agent.contracts import AgentError
from agent.llm import FixtureLLMClient, HttpLLMClient, LLMReply
from agent.sessions import stable_json
from agent import data_policy


def exchange(identifier='a', reasoning=0, content='contract'):
    return [{'role': 'assistant', 'content': '', 'reasoning_content': 'think ' * reasoning,
             'tool_calls': [{'id': identifier, 'type': 'function', 'function': {'name': 'metrics.lookup', 'arguments': '{}'}}]},
            {'role': 'tool', 'tool_call_id': identifier, 'content': stable_json(data_policy.seal({'ok': True, 'result': content, 'context_ref': 'op-'+'a'*32}, 'metrics.lookup'))}]


def signed_summary(text):
    return {'summary': text, 'summary_seal': data_policy.seal_text(text, 'summary')}


def compact(cp, llm=None, **kwargs):
    llm = llm or FixtureLLMClient([])
    llm.context_window_tokens = getattr(llm, 'context_window_tokens', 8192)
    events = []
    asyncio.run(compact_if_needed(cp, system=kwargs.pop('system', 'system'), llm=llm, tools=kwargs.pop('tools', []),
                                  on_compacted=events.append, **kwargs))
    return events


def assert_paired(messages):
    pending = set()
    for message in messages:
        if message.get('role') == 'assistant':
            assert not pending
            pending = {c['id'] for c in message.get('tool_calls', [])}
        elif message.get('role') == 'tool':
            assert message['tool_call_id'] in pending
            pending.remove(message['tool_call_id'])
        else:
            assert not pending
    assert not pending


def test_reused_call_sources_survive_projection_close_pending_and_compaction(tmp_path, monkeypatch):
    from agent.context import _projection, close_pending
    from test_agent_admission import assistant_call, primary_request, sealed, tool_message

    monkeypatch.setenv('CUSTOM_INDICATOR_DATA_DIR', str(tmp_path))
    first = [assistant_call('same', 'metrics.lookup'),
             tool_message(sealed({'ok': True, 'result': {'items': [{'id': 'returns'}]}}, 'metrics.lookup'), 'same')]
    second = [assistant_call('same', 'task.read'),
              tool_message(sealed({'ok': True, 'result': {'section': 'sources'}}, 'task.read'), 'same')]
    messages = first + second
    projected = _projection(messages)
    assert projected == messages
    pending = copy.deepcopy(first + [second[0]])
    close_pending(pending)
    assert pending[:len(first)] == first
    notice = json.loads(pending[-1]['content'])
    assert data_policy.verify(notice, 'task.read') and notice['status'] == 'not_executed'
    data_policy.enforce_request(**primary_request(pending))
    before = copy.deepcopy(pending)
    close_pending(pending)
    assert pending == before
    broken = copy.deepcopy([first[0], second[0]])
    close_pending(broken)
    assert broken == [first[0], second[0]]
    with pytest.raises(AgentError):
        data_policy.enforce_request(**primary_request(broken))

    checkpoint = {'messages': copy.deepcopy(messages)}
    checkpoint['messages'][0]['reasoning_content'] = 'reasoning ' * 9000
    assert compact(checkpoint)
    assert checkpoint['messages'] == second
    assert 'policy_reprojected' not in checkpoint['summary']
    assert 'returns' in checkpoint['summary']
    data_policy.enforce_request(**primary_request(context_messages(checkpoint)))


def test_duplicate_ids_inside_a_batch_cannot_be_hidden_by_compaction(tmp_path, monkeypatch):
    monkeypatch.setenv('CUSTOM_INDICATOR_DATA_DIR', str(tmp_path))
    checkpoint = {'messages': exchange(reasoning=16000)}
    checkpoint['messages'][0]['tool_calls'] *= 2
    before = copy.deepcopy(checkpoint)
    with pytest.raises(AgentError):
        compact(checkpoint)
    assert checkpoint == before


@pytest.mark.parametrize('boundary', ['completed', 'not_executed'])
def test_interrupted_batch_resumes_only_calls_without_committed_results(tmp_path, monkeypatch, boundary):
    from agent.harness import RunController
    from test_agent_runs import setup, request, DEFINITION

    monkeypatch.setenv('CUSTOM_INDICATOR_DATA_DIR', str(tmp_path))
    store, page, session, service = setup(tmp_path)
    sid = session['session_id']
    validations = []
    validate = service.validate
    service.validate = lambda definition: validations.append(definition['name']) or validate(definition)
    checkpoint = store.checkpoint
    stopped = []
    def interrupt_after_commit(run, **kwargs):
        applied = checkpoint(run, **kwargs)
        receipt = kwargs.get('tool_receipt') or {}
        if receipt.get('status') == boundary and receipt.get('call_id') == 'first' and not stopped:
            stopped.append(True)
            raise asyncio.CancelledError()
        return applied
    monkeypatch.setattr(store, 'checkpoint', interrupt_after_commit)
    lookup = {'name': 'metrics.lookup', 'arguments': {'kind': 'indicators', 'query': '演示'}}
    calls = [{'name': 'metrics.validate', 'arguments': {'definition': {**DEFINITION, 'name': name}}, 'call_id': identifier}
             for name, identifier in [('第一个工具', 'first'), ('剩余工具', 'remaining')]]
    replies = ([{'tool_calls': [lookup]}, {'tool_calls': [lookup]}, {'tool_calls': [lookup, *calls]}]
               if boundary == 'not_executed' else [{'tool_calls': calls}])

    async def run():
        owner = RunController(store)
        try:
            task, _ = owner.submit(sid, request(page), FixtureLLMClient(replies), service)
            with pytest.raises(asyncio.CancelledError):
                await owner.wait(task)
            interrupted = store.get_run(sid, task['run_id'])
            assert interrupted['status'] == 'interrupted'
            assert validations == ([] if boundary == 'not_executed' else ['第一个工具'])
            model = FixtureLLMClient([{'content': '未完成部分已继续。'}])
            resumed, _ = owner.submit(sid, request(page, message='continue', revision=store.read(sid)['session_revision'],
                text='继续', resume=task['run_id']), model, service)
            assert (await owner.wait(resumed))['status'] == 'completed'
            assert validations == (['剩余工具'] if boundary == 'not_executed' else ['第一个工具', '剩余工具'])
            assert len(model.requests) == 1
            assert 'pending_calls' not in store.get_run(sid, resumed['run_id'])['checkpoint']
            assert service.create_calls == []
        finally:
            await owner.close()

    asyncio.run(run())


@pytest.mark.parametrize('history', ['dangling_queue', 'changed_pending', 'changed_completed'])
def test_restored_dispatch_comes_from_unchanged_pending_transcript(tmp_path, monkeypatch, history):
    from agent.harness import RunController
    from agent.research_runtime import catalog_version, data_generation
    from test_agent_runs import setup, request, DEFINITION
    from test_agent_admission import assistant_call, sealed, tool_message

    monkeypatch.setenv('CUSTOM_INDICATOR_DATA_DIR', str(tmp_path))
    store, page, session, service = setup(tmp_path)
    sid = session['session_id']
    validated = []
    validate = service.validate
    service.validate = lambda definition: validated.append(definition['name']) or validate(definition)
    definition = {**DEFINITION, 'name': '待执行指标'}
    if history == 'changed_pending':
        definition['description'] = '[{"raw":987654.321}]'
    legacy_call = {'id': 'next', 'name': 'metrics.validate', 'arguments': {'definition': definition}}
    messages = [{'role': 'user', 'content': '原有研究要求'}]
    if history == 'changed_completed':
        messages += [assistant_call('earlier', 'metrics.lookup', '{"query":"[1,2]"}'),
                     tool_message(sealed({'ok': True, 'result': {'items': []}}, 'metrics.lookup'), 'earlier')]
    if history != 'dangling_queue':
        messages.append(assistant_call('next', legacy_call['name'], stable_json(legacy_call['arguments'])))
    parent, _ = store.accept(sid, request(page), {'owner_instance': 'legacy'})
    parent.update(status='interrupted', catalog_version=catalog_version(service), data_generation=data_generation(service),
                  checkpoint={'messages': messages, 'model_step': 1, 'pending_calls': [legacy_call]})
    with store.connection(write=True) as db:
        store._write_run(db, parent)
        state = store._read_state(db, sid)
        state['active_run_id'] = None
        store._write_state(db, state)

    async def run():
        owner = RunController(store)
        model = FixtureLLMClient([{'content': '根据当前计划继续。'}])
        try:
            resumed, _ = owner.submit(sid, request(page, message='resume', revision=store.read(sid)['session_revision'],
                text='继续', resume=parent['run_id']), model, service)
            done = await owner.wait(resumed)
            assert done['status'] == 'completed', done.get('error')
            assert validated == (['待执行指标'] if history == 'changed_completed' else [])
            assert len(model.requests) == 1
            assert '987654' not in stable_json(model.requests)
            assert 'pending_calls' not in done['checkpoint']
            assert service.create_calls == []
        finally:
            await owner.close()

    asyncio.run(run())


def test_automatic_window_is_endpoint_bound_and_override_wins():
    def client(url, **kw):
        return HttpLLMClient(base_url=url, model='deepseek-v4.1-flash', api_key='fixture', **kw)
    assert client('https://opencode.ai/zen/go/v1').context_window_tokens == 1_000_000
    assert client('https://proxy.example/v1').context_window_tokens == 32768
    assert client('http://opencode.ai/zen/go/v1').context_window_tokens == 32768
    assert client('https://opencode.ai/zen/go/v1', context_window_tokens=65536).context_window_tokens == 65536


def test_two_groups_with_huge_recent_reasoning_compact_and_preserve_user_quote():
    quote = '窗口20日，禁止未来数据，无风险利率1.5%，不要改为0。'
    cp = {'messages': [{'role': 'user', 'content': quote}, *exchange(reasoning=8000)]}
    source = copy.deepcopy(cp)
    events = compact(cp)
    assert events and events[0]['after_tokens'] < events[0]['before_tokens']
    assert any(m.get('content') == quote for m in context_messages(cp))
    assert 'think think' not in cp['summary']
    assert 'op-'+'a'*32 in cp['summary']
    assert_paired(cp['messages'])
    assert source['messages'][1]['reasoning_content'] == 'think '*8000


def test_one_giant_exchange_can_retire_without_any_recent_tool_group():
    cp = {'messages': exchange(reasoning=8000)}
    assert compact(cp)
    assert cp['messages'] == [] and cp['summary']


def test_retained_reasoning_is_byte_for_byte_unchanged_and_tools_count_towards_budget():
    small = exchange('b', reasoning=5)
    cp = {'messages': exchange(reasoning=9000) + small}
    assert compact(cp)
    assert cp['messages'] == small
    assert_paired(cp['messages'])
    a = request_tokens('sys', small, [])
    b = request_tokens('sys', small, [{'name': 'x', 'parameters': {'description': 'tool '*1000}}])
    assert b > a + 1000
    assert estimate_tokens('汉字'*1000) > estimate_tokens('ab'*1000)


def test_soft_threshold_without_retirable_history_is_not_a_hard_failure():
    # The newest user request fits; a soft trigger must not demand a shorter request.
    cp = {'messages': [{'role': 'user', 'content': 'a'*15900}]}
    before = copy.deepcopy(cp)
    assert compact(cp) == [] and cp == before


def test_fixed_prompt_or_user_cannot_fit_is_explicit_and_not_retried():
    for cp, system in [({'messages':[{'role':'user','content':'x'*20000}]}, 's'),
                       ({'messages':exchange(reasoning=9000)}, '系统'*14000)]:
        before = copy.deepcopy(cp); llm = FixtureLLMClient([])
        with pytest.raises(AgentError, match='核对'):
            compact(cp, llm, system=system)
        assert cp == before and not llm.requests


def test_summary_failure_preserves_a_deterministic_evidence_checkpoint():
    cp = {'messages': [{'role':'user','content':'仍然使用20日窗口'}, *exchange(reasoning=16000)],
          **signed_summary('已确认采用复权收盘价。'*550)}
    llm = FixtureLLMClient([]); llm.context_window_tokens = 16384
    events = compact(cp, llm)
    assert events and '复权收盘价' in cp['summary'] and 'context_ref' in cp['summary']
    assert len(llm.requests) == 1
    assert 'think think' not in stable_json(llm.requests)


def test_summary_is_history_not_system_and_repeated_compaction_keeps_latest_instructions():
    cp = {'messages':[{'role':'user','content':'禁止杠杆'}, *exchange(reasoning=15000)], **signed_summary('历史材料。'*1600)}
    llm = FixtureLLMClient([{'content':'目标：动量研究。未决：确认窗口。'}]); llm.context_window_tokens = 16384
    compact(cp, llm)
    cp['messages'] += [{'role':'user','content':'改为30日窗口'}, *exchange('c', reasoning=15000)]
    compact(cp)
    rendered = context_messages(cp)
    assert any(m.get('content') == '禁止杠杆' for m in rendered)
    assert any(m.get('content') == '改为30日窗口' for m in rendered)
    assert all(m['role'] != 'system' for m in rendered)
    assert cp['compaction_count'] == 2


def test_compaction_cancellation_never_installs_partial_checkpoint():
    class Cancel(FixtureLLMClient):
        async def complete(self, **kw):
            raise asyncio.CancelledError()
    cp = {'messages':exchange(reasoning=16000), **signed_summary('历史材料。'*1500)}
    before = copy.deepcopy(cp)
    llm = Cancel([]); llm.context_window_tokens = 16384
    with pytest.raises(asyncio.CancelledError): compact(cp, llm)
    assert cp == before


def test_pending_batch_and_orphan_results_are_never_compacted():
    cp = {'messages':exchange(reasoning=8000)[:1]}
    with pytest.raises(AgentError): compact(cp)
    # An obsolete queue cache cannot override a complete transcript.
    cp = {'messages':exchange(reasoning=8000), 'pending_calls':[{'id':'x'}]}
    assert compact(cp)
    cp = {'messages':[{'role':'tool','tool_call_id':'x','content':'x'*20000}]}
    with pytest.raises(AgentError, match='不完整'): compact(cp)


def test_provider_usage_calibrates_without_lowering_estimate():
    cp = {}; kwargs = {'system':'sys','messages':[{'role':'user','content':'hello'}], 'tools':[]}
    record_usage(cp, **kwargs, usage={'prompt_tokens':1000})
    first = cp['token_ratio']
    record_usage(cp, **kwargs, usage={'prompt_tokens':10})
    assert cp['token_ratio'] == first and first > 1
    record_usage(cp, **kwargs, usage={'prompt_tokens':None})
    assert cp['token_ratio'] == first


def test_legacy_sized_messages_fit_go_and_can_compact_in_a_small_window():
    for length in (41038,48561):
        cp={'messages':[{'role':'user','content':'评估ETF动量方向与强度'},*exchange()]}
        cp['messages'][1]['reasoning_content']='r'*length
        client=HttpLLMClient(base_url='https://opencode.ai/zen/go/v1', model='deepseek-v4.1-flash',api_key='fixture')
        before=copy.deepcopy(cp)
        assert compact(cp,client)==[] and cp==before  # Never contacts an upstream API.
        small=FixtureLLMClient([]);small.context_window_tokens=8192
        assert compact(cp,small)
        assert measure(cp,'system',[])[0] < input_budget(small,cp)


def test_retired_legacy_json_arrays_and_nested_evidence_references():
    cp={'messages':exchange(reasoning=9000)}
    cp['messages'][-1]['content']=stable_json(['x'*5000])
    assert compact(cp)
    cp={'messages':exchange(reasoning=9000)}
    ref='op-'+'b'*32
    cp['messages'][-1]['content']=stable_json({'ok':True,'result':{'context_ref':ref,'content':'x'*5000}})
    assert compact(cp) and ref in cp['summary']


@pytest.mark.parametrize('tool', ['memory.propose', 'task.plan'])
def test_edit_closes_old_tool_receipts_to_future_model_reads(tmp_path, monkeypatch, tool):
    from agent.harness import RunController
    from test_agent_runs import setup, stopped_turn, request

    monkeypatch.setenv('CUSTOM_INDICATOR_DATA_DIR', str(tmp_path))
    store, page, session, service = setup(tmp_path)
    sid = session['session_id']
    old_quote = '被替换的旧要求以后总是使用英文回答'
    arguments = ({'source_message_id': 'old', 'quote': old_quote, 'key': 'reply.language'}
                 if tool == 'memory.propose' else {'source_message_id': 'old', 'capabilities': ['explain'],
                     'constraints': [{'source_message_id': 'old', 'quote': old_quote}]})

    async def run():
        controller = RunController(store)
        try:
            first, waiting = stopped_turn(controller, sid, page, service, message='old', revision=0,
                text=old_quote, tool={'name': tool, 'arguments': arguments})
            await waiting.entered.wait()
            controller.cancel(sid, first['run_id'])
            assert (await controller.wait(first))['status'] == 'cancelled'
            with store.connection() as db:
                original = json.loads(db.execute('SELECT body FROM tool_calls WHERE run_id=?',
                    (first['run_id'],)).fetchone()[0])
            reference = original['operation_id']
            assert old_quote in store.read_context(sid, reference)['result']['content']

            model = FixtureLLMClient([
                {'tool_calls': [{'name': 'context.read', 'arguments': {'operation_id': reference}}]},
                {'content': '采用改写后的研究要求。'}])
            replacement, _ = controller.submit(sid,
                request(page, message='new', revision=1, edit='old', text='改写后的要求只使用中文'), model, service)
            assert (await controller.wait(replacement))['status'] == 'completed'
            with pytest.raises(AgentError) as denied:
                store.read_context(sid, reference)
            assert denied.value.code == 'AGENT_EVIDENCE_NOT_FOUND'
            replies = [json.loads(message['content']) for message in model.requests[-1]['messages']
                       if message['role'] == 'tool']
            assert replies[-1]['ok'] is False
            assert replies[-1]['error']['code'] == 'AGENT_EVIDENCE_NOT_FOUND'
            assert old_quote not in stable_json(model.requests)
            # Audit storage remains intact even though model readback is closed.
            with store.connection() as db:
                assert json.loads(db.execute('SELECT body FROM tool_calls WHERE run_id=?',
                    (first['run_id'],)).fetchone()[0]) == original
            assert service.create_calls == []
        finally:
            await controller.close()

    asyncio.run(run())


def test_legacy_edit_discards_large_plan_with_omitted_model_result(tmp_path, monkeypatch):
    from agent.harness import RunController
    from test_agent_runs import setup, stopped_turn, request

    monkeypatch.setenv('CUSTOM_INDICATOR_DATA_DIR', str(tmp_path))
    store, page, session, service = setup(tmp_path)
    sid = session['session_id']
    quote, question = '约束' * 250, '已被替换轮的旧问题' * 30

    async def run():
        controller = RunController(store)
        try:
            for revision, (message_id, text) in enumerate([('p' * 64, '此前约束'), ('s' * 64, quote)]):
                task, _ = controller.submit(sid, request(page, message=message_id, revision=revision, text=text),
                    FixtureLLMClient([{'content': '已核对原话。'}]), service)
                assert (await controller.wait(task))['status'] == 'completed'
            arguments = {'source_message_id': 's' * 64, 'questions': [question] * 5,
                         'constraints': [{'source_message_id': 's' * 64,
                             'supersedes_source_message_id': 'p' * 64, 'quote': quote}] * 10}
            stopped, waiting = stopped_turn(controller, sid, page, service, message='old', revision=2,
                text='待替换的旧研究问题', tool={'name': 'task.plan', 'arguments': arguments})
            await waiting.entered.wait()
            controller.cancel(sid, stopped['run_id'])
            assert (await controller.wait(stopped))['status'] == 'cancelled'
            with store.connection(write=True) as db:
                receipt = json.loads(db.execute('SELECT body FROM tool_calls WHERE run_id=?',
                    (stopped['run_id'],)).fetchone()[0])
                assert receipt['applied'] is True
                assert receipt['result']['result']['omitted']['code'] == 'result_too_large'
                old = store._read_run(db, sid, stopped['run_id'])
                old.pop('base_state', None)
                old.pop('parent_run_id', None)
                store._write_run(db, old)
            assert store.read(sid)['task_plans'][0]['questions'] == [question] * 5
            model = FixtureLLMClient([{'content': '只解释改写后的术语。'}])
            replacement, _ = controller.submit(sid,
                request(page, message='new', revision=3, edit='old', text='改写后只解释术语'), model, service)
            assert (await controller.wait(replacement))['status'] == 'completed'
            assert store.read(sid)['task_plans'] == []
            assert question not in stable_json(model.requests)
            assert quote in stable_json(model.requests)
            assert service.create_calls == []
        finally:
            await controller.close()

    asyncio.run(run())

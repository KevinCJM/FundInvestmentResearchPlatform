"""Run lifecycle, exactly-once receipts and stop/recovery races, fully offline."""
import asyncio
import copy
import json
import threading
import time
from pathlib import Path

import pytest
from agent import data_policy
from fastapi import FastAPI
from fastapi.testclient import TestClient
from agent.contracts import AgentError, AgentMessageRequest, PageContext, CommitPreviewRequest, CommitRequest
from agent.harness import RunController, stop_message
from agent.llm import FixtureLLMClient, LLMReply, LLMToolCall
from agent.sessions import AgentSessionStore, append_event, store_draft, stable_json
from test_agent_api import FakeIndicatorService, authoring_context, DEFINITION


def setup(tmp_path):
    store=AgentSessionStore(tmp_path/'sessions')
    page=PageContext.model_validate(authoring_context())
    session=store.create(page_context=page,scope='indicator_center')
    return store,page,session,FakeIndicatorService(tmp_path)


def request(page, *, message='m', revision=0, text='查询指标', resume=None, edit=None, snapshot=None):
    return AgentMessageRequest(message_id=message,expected_session_revision=revision,text=text,page_context=page,resume_from_run_id=resume,edit_of_message_id=edit,page_snapshot=snapshot)


def evidence_sections(sections, snapshot_id):
    return {'version':1,'snapshot_id':snapshot_id,'captured_at':'2026-09-21T02:00:00+00:00','page':'indicator-studio','sections':sections}


def evidence(marker, snapshot_id):
    # Registered page-contract fields only: unknown page keys are filtered by the strict
    # section view, and the resume/content assertions below use these declared fields.
    return evidence_sections({'editing':{'runtime_inputs':{'period':marker}},
                              'results':{'displayed_source':marker,'pending':[]}}, snapshot_id)


def tool_messages(llm, step):
    return [json.loads(message['content']) for message in llm.requests[step]['messages'] if message['role']=='tool']


def test_new_session_does_not_inherit_previous_conversation(tmp_path):
    store,page,old,service=setup(tmp_path)
    with store.locked(old['session_id']) as state:
        state['conversation'] = {'messages':[{'role':'user','content':'PRIVATE_OLD_MESSAGE'}],
            'summary':'PRIVATE_OLD_SUMMARY', 'pinned_user_messages':[{'role':'user','content':'PRIVATE_OLD_PIN'}]}
        state['memory'] = {'old':'PRIVATE_OLD_MEMORY'}
        store_draft(state, definition={**DEFINITION,'description':'PRIVATE_OLD_DRAFT'}, validation={'valid':True}, compile_token='a'*64)
        store.write(state)
    fresh = store.create(page_context=page,scope='indicator_center')
    assert fresh['session_id'] != old['session_id']
    assert fresh['draft'] is None and fresh['last_valid_draft'] is None and fresh['memory'] == {}
    async def run():
        controller=RunController(store);llm=FixtureLLMClient([{'content':'新的独立回复'}])
        try:
            task,_=controller.submit(fresh['session_id'],request(page,text='新研究需求'),llm,service)
            done=await controller.wait(task)
            assert done['status']=='completed'
            assert [m for m in llm.requests[0]['messages'] if not data_policy.verify_instruction(m.get('content'), 'task_state')]==[{'role':'user','content':'新研究需求'}]
            assert 'PRIVATE_OLD' not in stable_json(llm.requests)
            assert store.read(old['session_id'])['conversation']['summary']=='PRIVATE_OLD_SUMMARY'
        finally:await controller.close()
    asyncio.run(run())


async def until(predicate):
    async with asyncio.timeout(5):
        while not predicate():await asyncio.sleep(.005)


def test_over_100_productive_steps_and_durable_events(tmp_path, monkeypatch):
    store,page,session,service=setup(tmp_path)
    service.meta=lambda:{'engine_version':'t','dsl_version':'2.1.0','variables':[{'name':f'variable_{i:03d}','description':'研究所需变量'} for i in range(120)],'periods':[]}
    class Client(FixtureLLMClient):
        async def complete(self, *, system,messages,tools):
            if not tools:return LLMReply(content='用户要求查询变量，已取得先前目录证据。')
            return await super().complete(system=system,messages=messages,tools=tools)
    llm=Client([{'tool_calls':[{'name':'metrics.lookup','arguments':{'kind':'variables','query':f'variable_{i:03d}','limit':1}}]} for i in range(110)]+[{'content':'查找完成。'}])
    async def run():
        controller=RunController(store)
        try:
            task,_=controller.submit(session['session_id'],request(page,text='请逐一查询全部变量契约'),llm,service)
            done=await controller.wait(task)
            assert done['status']=='completed',done.get('error')
            assert done['usage']['tool_calls']==110
            assert len(llm.requests)==111
            events=[];cursor=0
            while True:
                result=store.events(session['session_id'],after_seq=cursor)
                events+=result['items'];cursor=result['last_seq']
                if not result['has_more']:break
            assert len(events)>400
            assert [e['seq'] for e in events]==list(range(1,len(events)+1))
            assert all('_model_message' not in e for e in events)
            assert len(stable_json(done['checkpoint']))<70000
        finally:await controller.close()
    asyncio.run(run())


def test_model_cancellation_and_duplicate_acceptance(tmp_path):
    store,page,session,service=setup(tmp_path)
    class Waiting(FixtureLLMClient):
        def __init__(self):super().__init__([]);self.entered=asyncio.Event();self.calls=0
        async def complete(self,**kwargs):self.calls+=1;self.entered.set();await asyncio.Event().wait()
    async def run():
        controller=RunController(store);llm=Waiting()
        try:
            task,_=controller.submit(session['session_id'],request(page),llm,service)
            same,replayed=controller.submit(session['session_id'],request(page),llm,service)
            assert replayed and same['run_id']==task['run_id']
            await llm.entered.wait()
            with pytest.raises(AgentError,match='处理'):controller.submit(session['session_id'],request(page,message='other',revision=1),llm,service)
            stopped=controller.cancel(session['session_id'],task['run_id'])
            assert stopped['status']=='stopping'
            done=await controller.wait(task)
            assert done['status']=='cancelled' and llm.calls==1
            assert store.read(session['session_id'])['active_run_id'] is None
        finally:await controller.close()
    asyncio.run(run())


def test_late_tool_result_after_cancel_is_discarded(tmp_path):
    store,page,session,service=setup(tmp_path);entered=threading.Event();release=threading.Event()
    def validate(definition):entered.set();release.wait(5);return {'valid':True,'compile_token':'a'*64}
    service.validate=validate
    async def run():
        controller=RunController(store)
        try:
            llm=FixtureLLMClient([{'tool_calls':[{'name':'metrics.validate','arguments':{'definition':DEFINITION}}]}])
            task,_=controller.submit(session['session_id'],request(page),llm,service)
            await until(entered.is_set)
            active = store.public(session['session_id'])
            assert active['execution_blocked_by'] is None and active['active_run']['execution_blocked_by'] is None
            controller.cancel(session['session_id'],task['run_id'])
            release.set();done=await controller.wait(task)
            assert done['status']=='cancelled'
            assert store.read(session['session_id'])['draft'] is None
            with store.connection() as db:
                receipt=json.loads(db.execute('SELECT body FROM tool_calls').fetchone()[0])
            assert receipt['status']=='discarded' and not receipt['applied']
            assert any(json.loads(m['content']).get('status')=='discarded' for m in done['checkpoint']['messages'] if m['role']=='tool')
            assert store.read(session['session_id'])['conversation']['messages'][0]['content']=='查询指标'
            assert len(llm.requests)==1
        finally:release.set();await controller.close()
    asyncio.run(run())


@pytest.mark.parametrize('source', ['indicator', 'registry'])
def test_catalog_change_during_tool_discards_receipt_and_draft(tmp_path, source):
    from agent.research_runtime import catalog_version
    store, page, session, service = setup(tmp_path)
    entered, release = threading.Event(), threading.Event()
    listing, metadata = service.list_indicators(), service.meta()
    service.list_indicators = lambda **kwargs: copy.deepcopy(listing)
    service.meta = lambda: dict(metadata)
    validate = service.validate

    def waiting_validate(definition):
        result = validate(definition)
        entered.set()
        assert release.wait(5)
        return result

    service.validate = waiting_validate
    async def run():
        controller = RunController(store)
        llm = FixtureLLMClient([{'tool_calls': [{'name': 'metrics.validate', 'arguments': {'definition': DEFINITION}}]},
                               {'content': '该指标已经校验完成。'}])
        try:
            task, _ = controller.submit(session['session_id'], request(page), llm, service)
            await until(entered.is_set)
            before = catalog_version(service)
            if source == 'indicator':
                listing['items'][0]['revision'] += 1
            else:
                metadata['operator_registry_version'] = 'changed-registry'
            assert catalog_version(service) != before
            release.set()
            done = await controller.wait(task)
            assert done['status'] == 'paused' and done['stop_reason'] == 'context_changed'
            assert done['context_change']['source'] == 'catalog'
            assert store.read(session['session_id'])['draft'] is None
            with store.connection() as db:
                receipt = json.loads(db.execute('SELECT body FROM tool_calls').fetchone()[0])
            assert receipt['status'] == 'discarded' and not receipt['applied']
            assert len(llm.requests) == 1
            assert all(json.loads(m['content']).get('status') == 'discarded'
                       for m in done['checkpoint']['messages'] if m['role'] == 'tool')
        finally:
            release.set()
            await controller.close()
    asyncio.run(run())


def test_catalog_change_during_model_reply_pauses_before_publishing(tmp_path):
    store, page, session, service = setup(tmp_path)
    listing = service.list_indicators()
    service.list_indicators = lambda **kwargs: copy.deepcopy(listing)

    async def run():
        entered, release = asyncio.Event(), asyncio.Event()
        class WaitingReply(FixtureLLMClient):
            async def complete(self, **kwargs):
                reply = await super().complete(**kwargs)
                if len(self.requests) == 2:
                    entered.set()
                    await release.wait()
                return reply

        llm = WaitingReply([{'tool_calls': [{'name': 'metrics.validate', 'arguments': {'definition': DEFINITION}}]},
                            {'content': 'SHOULD_NOT_PUBLISH_STALE_REPLY', 'usage': {'prompt_tokens': 19, 'completion_tokens': 7}}])
        controller = RunController(store)
        try:
            task, _ = controller.submit(session['session_id'], request(page), llm, service)
            await until(entered.is_set)
            listing['items'][0]['revision'] += 1
            release.set()
            done = await controller.wait(task)
            assert done['status'] == 'paused' and done['stop_reason'] == 'context_changed'
            assert done['context_change']['source'] == 'catalog'
            assert store.read(session['session_id'])['draft']['valid'], 'prior applied progress survives'
            assert done['usage']['prompt_tokens'] == 19 and done['usage']['completion_tokens'] == 7
            assert 'SHOULD_NOT_PUBLISH_STALE_REPLY' not in stable_json(store.public(session['session_id']))
            assert 'SHOULD_NOT_PUBLISH_STALE_REPLY' not in stable_json(done['checkpoint'])
        finally:
            release.set()
            await controller.close()
    asyncio.run(run())


def test_tool_timeout_keeps_slot_until_actual_completion(tmp_path,monkeypatch):
    monkeypatch.setenv('AGENT_TOOL_TIMEOUT_SECONDS','.03')
    store,page,session,service=setup(tmp_path);release=threading.Event()
    def validate(definition):release.wait(5);return {'valid':True}
    service.validate=validate
    async def run():
        controller=RunController(store)
        try:
            task,_=controller.submit(session['session_id'],request(page),FixtureLLMClient([{'tool_calls':[{'name':'metrics.validate','arguments':{'definition':DEFINITION}}]}]),service)
            done=await controller.wait(task)
            assert done['status']=='failed' and done['stop_reason']=='operation_timeout'
            assert done['error']['code'] == 'AGENT_OPERATION_TIMEOUT'
            assert done['response']['stop_reason'] == 'operation_timeout'
            assert done['response']['reply']['text']
            assert any(item.get('speaker') == 'assistant' and item.get('text') == done['response']['reply']['text']
                       for item in store.public(session['session_id'])['messages'])
            assert store.read(session['session_id'])['execution_blocked_by']
            with pytest.raises(AgentError):controller.submit(session['session_id'],request(page,message='new',revision=1),FixtureLLMClient([]),service)
            release.set()
            await until(lambda:not store.read(session['session_id']).get('execution_blocked_by'))
            assert not store.get_run(session['session_id'],task['run_id']).get('execution_blocked_by')
            assert store.read(session['session_id'])['draft'] is None
        finally:release.set();await controller.close()
    asyncio.run(run())


@pytest.mark.parametrize('failure_boundary', ['unknown_timeout', 'stop_phase', 'unknown_shutdown'])
def test_live_tool_remains_fenced_when_later_control_writes_fail(tmp_path, monkeypatch, failure_boundary):
    monkeypatch.setenv('CUSTOM_INDICATOR_DATA_DIR', str(tmp_path))
    monkeypatch.setenv('AGENT_TOOL_TIMEOUT_SECONDS', '5' if failure_boundary == 'unknown_shutdown' else '.04')
    store, page, session, service = setup(tmp_path)
    sid = session['session_id']
    entered, release = threading.Event(), threading.Event()
    calls, failures = [], []
    real_admit = store.admit
    def validate(definition):
        calls.append(definition)
        entered.set()
        assert release.wait(10)
        raise RuntimeError('线程在运行收尾后才返回的失败')
    def admit(run, **kwargs):
        unknown = (kwargs.get('tool_receipt') or {}).get('status') == 'unknown'
        stopping = any(e.get('type') == 'run.phase' and e.get('data', {}).get('status') == 'stopping'
                       for e in kwargs.get('events', ()))
        if not failures and (stopping if failure_boundary == 'stop_phase' else unknown):
            failures.append(failure_boundary)
            raise AgentError('AGENT_STORAGE_UNAVAILABLE', '本次控制状态写入失败。', status_code=503)
        return real_admit(run, **kwargs)
    service.validate = validate
    monkeypatch.setattr(store, 'admit', admit)
    async def run():
        controller = RunController(store)
        other = RunController(AgentSessionStore(store.root))
        unhandled = []
        loop = asyncio.get_running_loop()
        prior_handler = loop.get_exception_handler()
        loop.set_exception_handler(lambda _, context: unhandled.append(context))
        try:
            task, _ = controller.submit(sid, request(page), FixtureLLMClient([
                {'tool_calls': [{'name': 'metrics.validate', 'arguments': {'definition': DEFINITION}}]}]), service)
            await until(entered.is_set)
            if failure_boundary == 'stop_phase':
                controller.cancel(sid, task['run_id'])
            elif failure_boundary == 'unknown_shutdown':
                await controller.close()
            done = await controller.wait(task)
            assert failures == [failure_boundary]
            assert any(not f.done() for f in controller.futures)
            snapshot = store.public(sid)
            assert snapshot['execution_blocked_by'] and done['execution_blocked_by'] == snapshot['execution_blocked_by']
            with pytest.raises(AgentError) as busy:
                other.submit(sid, request(page, message='overlap', revision=snapshot['session_revision']), FixtureLLMClient([]), service)
            assert busy.value.code == 'AGENT_SESSION_BUSY'
            independent = store.create(page_context=page, scope=session['scope'])['session_id']
            parallel, _ = other.submit(independent, request(page), FixtureLLMClient([{'content': '独立研究可继续。'}]), service)
            assert (await other.wait(parallel))['status'] == 'completed'
            release.set()
            await until(lambda: not store.read(sid).get('execution_blocked_by'))
            following = FixtureLLMClient([{'content': '从已提交进度继续。'}])
            resumed, _ = other.submit(sid, request(page, message='after-exit', text='继续', resume=task['run_id'],
                revision=store.read(sid)['session_revision']), following, service)
            assert (await other.wait(resumed))['status'] == 'completed'
            assert len(calls) == 1 and service.create_calls == []
            await asyncio.sleep(0)
            assert not unhandled
        finally:
            release.set()
            await controller.close()
            await other.close()
            loop.set_exception_handler(prior_handler)
    asyncio.run(run())


def test_completed_receipt_releases_only_its_operation_before_delayed_callback(tmp_path, monkeypatch):
    monkeypatch.setenv('CUSTOM_INDICATOR_DATA_DIR', str(tmp_path))
    store, page, session, service = setup(tmp_path)
    sid = session['session_id']
    entered, release = threading.Event(), threading.Event()
    callbacks = []
    async def run():
        controller = RunController(store)
        monkeypatch.setattr(controller, '_dispatch_tool_callback', lambda callback, future: callbacks.append((callback, future)))
        try:
            first, _ = controller.submit(sid, request(page), FixtureLLMClient([
                {'tool_calls': [{'name': 'metrics.validate', 'arguments': {'definition': DEFINITION}}]},
                {'content': '第一轮完成。'}]), service)
            assert (await controller.wait(first))['status'] == 'completed'
            assert not store.read(sid).get('execution_blocked_by') and callbacks
            old_callbacks = list(callbacks)
            callbacks.clear()
            def validate(_):
                entered.set()
                assert release.wait(10)
                return {'valid': True}
            service.validate = validate
            second, _ = controller.submit(sid, request(page, message='second', revision=store.read(sid)['session_revision']),
                FixtureLLMClient([{'tool_calls': [{'name': 'metrics.validate', 'arguments': {'definition': DEFINITION}}]},
                                  {'content': '第二轮完成。'}]), service)
            await until(entered.is_set)
            operation = store.read(sid).get('execution_blocked_by')
            assert operation
            for callback, future in old_callbacks:
                callback(future)
            assert store.read(sid)['execution_blocked_by'] == operation
            release.set()
            assert (await controller.wait(second))['status'] == 'completed'
            assert not store.read(sid).get('execution_blocked_by')
        finally:
            release.set()
            for callback, future in callbacks:
                callback(future)
            await controller.close()
    asyncio.run(run())


@pytest.mark.parametrize('release_fails', [False, True])
def test_executor_rejection_releases_admitted_operation_without_running_tool(tmp_path, monkeypatch, release_fails):
    monkeypatch.setenv('CUSTOM_INDICATOR_DATA_DIR', str(tmp_path))
    store, page, session, service = setup(tmp_path)
    sid = session['session_id']
    real_release = store.release_quarantine
    async def run():
        controller = RunController(store)
        real_submit = controller.executor.submit
        def unavailable(*args, **kwargs):
            raise RuntimeError('executor unavailable')
        def unavailable_storage(*args, **kwargs):
            raise AgentError('AGENT_STORAGE_UNAVAILABLE', '清理暂时失败', status_code=503)
        monkeypatch.setattr(controller.executor, 'submit', unavailable)
        if release_fails:
            monkeypatch.setattr(store, 'release_quarantine', unavailable_storage)
        try:
            task, _ = controller.submit(sid, request(page), FixtureLLMClient([
                {'tool_calls': [{'name': 'metrics.validate', 'arguments': {'definition': DEFINITION}}]}]), service)
            done = await controller.wait(task)
            assert done['status'] == 'failed' and not service.validate_calls and not controller.futures
            assert bool(store.read(sid).get('execution_blocked_by')) is release_fails
            monkeypatch.setattr(store, 'release_quarantine', real_release)
            monkeypatch.setattr(controller.executor, 'submit', real_submit)
            controller.reconcile()
            assert not store.read(sid).get('execution_blocked_by')
            resumed, _ = controller.submit(sid, request(page, message='next', revision=store.read(sid)['session_revision']),
                FixtureLLMClient([{'content': '下一轮可继续。'}]), service)
            assert (await controller.wait(resumed))['status'] == 'completed'
        finally:
            await controller.close()
    asyncio.run(run())


def test_late_unknown_receipt_cannot_recreate_a_finished_operation_fence(tmp_path):
    store, page, session, _ = setup(tmp_path)
    sid = session['session_id']
    task, _ = store.accept(sid, request(page), {'owner_instance': 'owner', 'owner_workspace': 'test'})
    task['status'] = 'running'
    store.checkpoint(task, events=[{'type': 'run.started'}])
    receipt = {'model_step': 1, 'call_id': 'call', 'operation_id': 'operation', 'tool': 'metrics.lookup', 'status': 'started'}
    assert store.admit(task, tool_receipt=receipt)
    assert store.read(sid)['execution_blocked_by'] == 'operation'
    store.release_quarantine(sid, 'operation')
    # Timeout handling can have read Future.done() before its completion callback.
    assert store.admit(task, tool_receipt={**receipt, 'status': 'unknown'})
    assert not store.read(sid).get('execution_blocked_by')
    assert not store.get_run(sid, task['run_id']).get('execution_blocked_by')


def test_invalid_tool_timeout_configuration_fails_before_dispatch(tmp_path, monkeypatch):
    monkeypatch.setenv('CUSTOM_INDICATOR_DATA_DIR', str(tmp_path))
    monkeypatch.setenv('AGENT_TOOL_TIMEOUT_SECONDS', 'invalid')
    store, page, session, service = setup(tmp_path)
    async def run():
        controller = RunController(store)
        before_tasks = asyncio.all_tasks()
        try:
            task, _ = controller.submit(session['session_id'], request(page), FixtureLLMClient([
                {'tool_calls': [{'name': 'metrics.validate', 'arguments': {'definition': DEFINITION}}]}]), service)
            assert (await controller.wait(task))['status'] == 'failed'
            await asyncio.sleep(0)
            assert not service.validate_calls and not controller.futures
            assert not store.read(session['session_id']).get('execution_blocked_by')
            assert not (asyncio.all_tasks() - before_tasks)
        finally:
            await controller.close()
    asyncio.run(run())


@pytest.mark.parametrize('ending', ['timeout', 'cancel', 'close'])
def test_finished_quarantine_recovers_after_temporary_sqlite_write_lock(tmp_path, monkeypatch, ending):
    import sqlite3

    monkeypatch.setenv('AGENT_TOOL_TIMEOUT_SECONDS', '.03')
    monkeypatch.setenv('AGENT_ACTIVE_RUNS', '1')
    store, page, session, service = setup(tmp_path)
    sid = session['session_id']
    with store.locked(sid) as state:
        store_draft(state, definition={**DEFINITION, 'description': '此前已完成的草稿'},
                    validation={'valid': True}, compile_token=None)
        store.write(state)
    prior_draft = store.read(sid)['draft']
    entered, release, released = threading.Event(), threading.Event(), threading.Event()
    calls, release_errors, release_threads = [], [], []
    real_release = store.release_quarantine
    def validate(definition):
        calls.append(definition)
        entered.set()
        assert release.wait(10)
        return {'valid': True}
    def observe_release(*args):
        release_threads.append(threading.get_ident())
        try:
            return real_release(*args)
        except AgentError as exc:
            release_errors.append(exc.code)
            raise
        finally:
            released.set()
    service.validate = validate
    monkeypatch.setattr(store, 'release_quarantine', observe_release)

    async def run():
        controller = RunController(store)
        lock = None
        try:
            task, _ = controller.submit(sid, request(page), FixtureLLMClient([
                {'tool_calls': [{'name': 'metrics.validate', 'arguments': {'definition': DEFINITION}}]}]), service)
            await until(entered.is_set)
            if ending == 'cancel':
                controller.cancel(sid, task['run_id'])
            elif ending == 'close':
                await controller.close()
            done = await controller.wait(task)
            assert done['status'] == ('interrupted' if ending == 'close' else 'failed')
            blocked = store.read(sid)['execution_blocked_by']
            controller.reconcile()
            assert store.read(sid)['execution_blocked_by'] == blocked
            other_session = store.create(page_context=page, scope=session['scope'])['session_id']
            with pytest.raises(AgentError) as busy:
                controller.submit(other_session, request(page, message='overlap'), FixtureLLMClient([]), service)
            assert busy.value.code == 'AGENT_SERVICE_BUSY'
            # Even a second controller must not reclaim a still-executing owner's run.
            other = RunController(store)
            try:
                other.reconcile()
                assert store.read(sid)['execution_blocked_by'] == blocked
            finally:
                await other.close()
            lock = sqlite3.connect(store.path)
            lock.execute('BEGIN IMMEDIATE')
            release.set()
            await until(released.is_set)
            assert release_errors == ['AGENT_STORAGE_UNAVAILABLE']
            assert set(release_threads) == {threading.get_ident()}
            lock.rollback(); lock.close(); lock = None
            assert not controller.futures and not controller.future_owners
            controller.reconcile()
            repaired = store.get_run(sid, task['run_id'])
            assert not store.read(sid).get('execution_blocked_by')
            assert not repaired.get('execution_blocked_by')
            assert repaired['status'] == done['status'] and repaired.get('stop_reason') == done.get('stop_reason')
            assert repaired.get('response') == done.get('response')
            assert store.read(sid)['draft'] == prior_draft and len(calls) == 1
            # Recovery only releases the fence; a human's new message starts the next run.
            next_controller = RunController(store) if ending == 'close' else controller
            try:
                followup, _ = next_controller.submit(sid, request(page, message='after-recovery',
                    revision=store.read(sid)['session_revision']), FixtureLLMClient([{'content': '可继续研究。'}]), service)
                assert (await next_controller.wait(followup))['status'] == 'completed'
                assert len(calls) == 1 and service.create_calls == []
            finally:
                if next_controller is not controller:
                    await next_controller.close()
        finally:
            release.set()
            if lock is not None:
                lock.rollback(); lock.close()
            await controller.close()
    asyncio.run(run())


def test_legacy_migration_is_idempotent_and_sequences_do_not_repeat(tmp_path):
    store,page,session,service=setup(tmp_path)
    legacy_id='agent-'+'a'*32
    legacy={**session,'session_id':legacy_id,'schema_version':1,'events':[{'seq':201,'type':'tool'},{'seq':201,'type':'tool'}], 'history':[{'role':'user','content':'用户口径'}], 'turns':{}, 'commits':{}}
    (store.root/f'{legacy_id}.json').write_text(json.dumps(legacy))
    state=store.read(legacy_id)
    assert state['legacy_history_incomplete']
    for _ in range(4):store.read(legacy_id)
    with store.locked(legacy_id) as current:
        for i in range(501):append_event(current,{'type':'test','data':{'index':i}})
        store.write(current)
    page1=store.events(legacy_id)
    assert page1['has_more'] and len(page1['items'])==200
    with store.connection() as db:
        seq=[r[0] for r in db.execute('SELECT seq FROM events WHERE session_id=? ORDER BY seq',(legacy_id,))]
    assert seq==list(range(1,505))
    assert store.read(legacy_id)['conversation']['messages'][0]['content']=='用户口径'


def test_owner_lock_prevents_takeover_and_dead_owner_recovers(tmp_path):
    store,page,session,service=setup(tmp_path)
    async def check():
        first=RunController(store)
        run,_=store.accept(session['session_id'],request(page),{'owner_instance':first.instance_id,'owner_workspace':first.workspace})
        other=RunController(store)
        assert store.get_run(session['session_id'],run['run_id'])['status']=='queued'
        first.owner_file.close();first.executor.shutdown()
        other.recover_owners()
        assert store.get_run(session['session_id'],run['run_id'])['status']=='interrupted'
        assert store.read(session['session_id'])['active_run_id'] is None
        await other.close()
    asyncio.run(check())


@pytest.mark.parametrize('method', ['submit', 'cancel', 'reconcile', 'recover_owners', 'wait', 'close', '_event'])
@pytest.mark.parametrize('foreign_loop', [False, True])
def test_controller_rejects_foreign_execution_before_storage_changes(tmp_path, method, foreign_loop):
    store, page, session, service = setup(tmp_path)
    sid = session['session_id']
    async def check():
        controller = RunController(store)
        run, _ = store.accept(sid, request(page), {'owner_instance': controller.instance_id,
                                                 'owner_workspace': controller.workspace})
        target_sid = store.create(page_context=page, scope=session['scope'])['session_id'] if method == 'submit' else sid
        before = store.path.read_bytes()
        def invoke():
            args = {'submit': (target_sid, request(page, message='wrong'), FixtureLLMClient([]), service),
                    'cancel': (sid, run['run_id']), 'wait': (run,), '_event': (run, 'wrong-thread')}
            call = getattr(controller, method)
            async def under_loop():
                result = call(*args.get(method, ()))
                if asyncio.iscoroutine(result):
                    await result
            with pytest.raises(AgentError) as error:
                if foreign_loop or method in {'wait', 'close'}:
                    asyncio.run(under_loop())
                else:
                    call(*args.get(method, ()))
            assert error.value.code == 'AGENT_CONTROLLER_CONTEXT'
        try:
            await asyncio.to_thread(invoke)
            assert store.path.read_bytes() == before
            assert not controller.closing and controller.tasks == {} and controller.stops == {}
            controller.reconcile()
            assert store.get_run(sid, run['run_id'])['status'] == 'interrupted'
        finally:
            await controller.close()
    asyncio.run(check())


def test_controller_creation_without_event_loop_has_no_storage_side_effects(tmp_path, monkeypatch):
    monkeypatch.setenv('CUSTOM_INDICATOR_DATA_DIR', str(tmp_path))
    before = set(tmp_path.rglob('*'))
    with pytest.raises(AgentError) as error:
        RunController()
    assert error.value.code == 'AGENT_CONTROLLER_CONTEXT'
    assert set(tmp_path.rglob('*')) == before


@pytest.mark.parametrize('cancelled', [False, True])
def test_control_events_never_publish_tentative_content(tmp_path, monkeypatch, cancelled):
    monkeypatch.setenv('CUSTOM_INDICATOR_DATA_DIR', str(tmp_path))
    store, page, session, _ = setup(tmp_path)
    sid = session['session_id']
    async def run():
        controller = RunController(store)
        try:
            task, _ = store.accept(sid, request(page), {'owner_instance': controller.instance_id, 'owner_workspace': controller.workspace})
            task.update(status='running', checkpoint={'messages': [{'role': 'user', 'content': '已接纳输入'}]},
                        detector_state={'stagnant': 0}, tool_trace=[])
            assert store.checkpoint(task, events=[{'type': 'run.started'}])
            before = store.get_run(sid, task['run_id'])
            task['checkpoint']['messages'].append({'role': 'assistant', 'content': '尚未提交的候选正文'})
            task['detector_state']['stagnant'] = 99
            task['tool_trace'].append({'tool': 'metrics.validate', 'status': 'ok'})
            task['usage']['model_steps'] = 1
            if cancelled:
                AgentSessionStore(store.root).cancel(sid, task['run_id'])
            assert controller._event(task, 'run.phase', status='running', phase='thinking') is not cancelled
            after = store.get_run(sid, task['run_id'])
            for key in ('checkpoint', 'detector_state', 'tool_trace'):
                assert after[key] == before[key]
            assert after['usage']['model_steps'] == 1
        finally:
            await controller.close()
    asyncio.run(run())


def test_controller_rejects_a_different_loop_on_the_same_thread(tmp_path):
    store, _, _, _ = setup(tmp_path)
    async def create():
        controller = RunController(store)
        await controller.close()
        return controller
    controller = asyncio.run(create())
    before = store.path.read_bytes()
    async def another_loop():
        with pytest.raises(AgentError) as error:
            controller.reconcile()
        assert error.value.code == 'AGENT_CONTROLLER_CONTEXT'
    asyncio.run(another_loop())
    assert store.path.read_bytes() == before


@pytest.mark.parametrize('close_loop_first', [False, True])
def test_shutdown_tool_exit_after_owner_loop_stopped_uses_owner_recovery(tmp_path, monkeypatch, close_loop_first):
    monkeypatch.setenv('AGENT_TOOL_TIMEOUT_SECONDS', '10')
    store, page, session, service = setup(tmp_path)
    entered, release = threading.Event(), threading.Event()
    def validate(_):
        entered.set()
        assert release.wait(10)
        return {'valid': True}
    service.validate = validate
    async def start_and_close():
        controller = RunController(store)
        task, _ = controller.submit(session['session_id'], request(page), FixtureLLMClient([
            {'tool_calls': [{'name': 'metrics.validate', 'arguments': {'definition': DEFINITION}}]}]), service)
        await until(entered.is_set)
        await controller.close()
        assert not controller.owner_file.closed
        return controller, task
    loop = asyncio.new_event_loop()
    controller, task = loop.run_until_complete(start_and_close())
    if close_loop_first:
        loop.close()
    try:
        release.set()
        deadline = time.monotonic() + 5
        while not controller.owner_file.closed and time.monotonic() < deadline:
            time.sleep(.005)
        assert controller.owner_file.closed and not controller.futures
        async def recover():
            replacement = RunController(store)
            try:
                assert not store.read(session['session_id']).get('execution_blocked_by')
                assert store.get_run(session['session_id'], task['run_id'])['status'] == 'interrupted'
            finally:
                await replacement.close()
        asyncio.run(recover())
    finally:
        release.set()
        if not loop.is_closed():
            loop.close()


def test_uncertain_commit_never_repeats_business_write(tmp_path):
    from agent import commit
    store,page,session,service=setup(tmp_path)
    with store.locked(session['session_id']) as state:
        store_draft(state,definition=DEFINITION,validation={'valid':True},compile_token=None);store.write(state)
    prepared=commit.preview(store=store,session_id=session['session_id'],request=CommitPreviewRequest(draft_revision=1,definition=DEFINITION,page_context=page),service=service)
    calls=[]
    def write_then_fail(definition):calls.append(definition);raise OSError('simulated lost receipt')
    service.create_indicator=write_then_fail
    command=CommitRequest(request_id='c',confirmation_id=prepared['confirmation_id'],definition_hash=prepared['definition_hash'],draft_revision=1,confirmed=True)
    for _ in range(2):
        with pytest.raises(AgentError) as err:commit.commit(store=store,session_id=session['session_id'],request=command,service=service)
        assert err.value.code=='AGENT_COMMIT_UNCERTAIN'
    assert len(calls)==1


def test_async_api_status_event_stream_and_context_invalidation(tmp_path,monkeypatch):
    from agent import routes
    monkeypatch.setenv('CUSTOM_INDICATOR_DATA_DIR',str(tmp_path))
    monkeypatch.setattr(routes,'resolve_service',lambda:FakeIndicatorService(tmp_path))
    monkeypatch.setattr(routes,'_llm_client',lambda session_id:FixtureLLMClient([{'content':'已讨论需求。'}]))
    app=FastAPI();app.include_router(routes.router)
    with TestClient(app) as client:
        page=authoring_context()
        sid=client.post('/api/agent/sessions',json={'page_context':page}).json()['session_id']
        response=client.post(f'/api/agent/sessions/{sid}/messages?response_mode=async',json={'message_id':'m','expected_session_revision':0,'text':'讨论','page_context':page})
        assert response.status_code==202
        run_id=response.json()['run_id']
        import time
        for _ in range(100):
            run=client.get(f'/api/agent/sessions/{sid}/runs/{run_id}').json()
            if run['status']=='completed':break
            time.sleep(.01)
        assert run['status']=='completed'
        events=client.get(f'/api/agent/sessions/{sid}/events').json()['items']
        cursor=events[-2]['seq']
        stream=client.get(f'/api/agent/sessions/{sid}/events?stream=1',headers={'Last-Event-ID':str(cursor)})
        assert stream.status_code==200 and 'event: agent' in stream.text
        assert f'id: {events[-1]["seq"]}' in stream.text
        assert f'id: {cursor}\n' not in stream.text
        cancelled=client.post(f'/api/agent/sessions/{sid}/runs/{run_id}/cancel',json={'request_id':'cancel'})
        assert cancelled.json()['status']=='completed'


def test_context_invalidation_discards_result_and_stops_cleanly(tmp_path):
    store,page,session,service=setup(tmp_path)
    class Waiting(FixtureLLMClient):
        def __init__(self):super().__init__([]);self.entered=asyncio.Event()
        async def complete(self,**kwargs):self.entered.set();await asyncio.Event().wait()
    async def run():
        controller=RunController(store);llm=Waiting()
        try:
            task,_=controller.submit(session['session_id'],request(page),llm,service)
            await llm.entered.wait()
            # Canonically identical requests must not cancel an in-flight model.
            same = PageContext.model_validate(json.loads(json.dumps(page.model_dump(), sort_keys=True)))
            unchanged = controller.cancel(session['session_id'],task['run_id'],reason='context_changed',context=same)
            assert unchanged['status'] == 'running'
            assert not controller.stops[task['run_id']].is_set()
            assert not any(e['type'] == 'context.invalidated' for e in store.events(session['session_id'])['items'])
            changed = page.model_copy(update={'context_revision':page.context_revision+1,
                'calculation':page.calculation.model_copy(update={'period':'3Y'})})
            controller.cancel(session['session_id'],task['run_id'],reason='context_changed',context=changed)
            store.checkpoint(task)  # An older runner checkpoint cannot erase the cause.
            done=await controller.wait(task)
            assert done['status']=='paused' and done['stop_reason']=='context_changed'
            assert done['context_change'] == {'source':'page', 'fields':['context_revision','calculation.period']}
            assert done['response']['reply']['text'].startswith('计算周期已变化')
            assert '已保留草稿' not in done['response']['reply']['text']
            events = store.events(session['session_id'])['items']
            event = next(e for e in events if e['type'] == 'context.invalidated')
            assert event['data'] == done['context_change'] and '_page_context' not in event
            with store.connection() as db:
                recorded = json.loads(db.execute("SELECT body FROM events WHERE session_id=? AND seq=?", (session['session_id'],event['seq'])).fetchone()[0])
            assert recorded['_page_context']['calculation']['period'] == '3Y'
            assert not store.read(session['session_id'])['active_run_id']
            assert store.read(session['session_id'])['conversation']['messages'][0]['content']=='查询指标'
        finally:await controller.close()
    asyncio.run(run())


@pytest.mark.parametrize('source,label', [('catalog','指标或算子目录'), ('data','数据快照'), ('unknown','页面研究条件')])
def test_context_change_messages_distinguish_dependencies(source,label):
    assert stop_message({'stop_reason':'context_changed','context_change':{'source':source}}).startswith(label+'已变化')


def test_quarantine_release_cannot_be_undone_by_stale_checkpoint(tmp_path):
    store,page,session,service=setup(tmp_path)
    task,_=store.accept(session['session_id'],request(page),{'owner_instance':'x','owner_workspace':'x'})
    task.update(status='running',stop_reason='operation_timeout',execution_blocked_by='operation')
    store.checkpoint(task, events=[{'type': 'run.started'}])
    receipt = {'model_step': 1, 'call_id': 'a', 'operation_id': 'operation', 'tool': 'metrics.lookup', 'status': 'started'}
    store.admit(task, tool_receipt=receipt)
    store.admit(task,tool_receipt={**receipt, 'status':'unknown'})
    task.update(store.finish(task, '操作超时。', status='failed', reason='operation_timeout'))
    store.release_quarantine(session['session_id'],'operation')
    store.checkpoint(task)
    assert not store.read(session['session_id']).get('execution_blocked_by')
    assert not store.get_run(session['session_id'],task['run_id']).get('execution_blocked_by')


def test_interrupted_saved_batch_continues_without_reasking_or_replaying_completed_tool(tmp_path):
    from agent.catalog import build_catalog
    from custom_indicators.series_provider import market_data_generation
    store,page,session,service=setup(tmp_path)
    old,_=store.accept(session['session_id'],request(page),{'owner_instance':'dead','owner_workspace':'old'})
    old.update(status='running',data_generation=market_data_generation(service.market_data_dir),catalog_version=build_catalog(service)['version'])
    first={'id':'old-complete','name':'metrics.lookup','arguments':{}}
    pending={'id':'old-pending','name':'metrics.validate','arguments':{'definition':DEFINITION}}
    old['checkpoint']={'model_step':1,'messages':[{'role':'user','content':'生成指标'}, {'role':'assistant','content':'','tool_calls':[{'id':c['id'],'type':'function','function':{'name':c['name'],'arguments':json.dumps(c['arguments'])}} for c in [first,pending]]}, {'role':'tool','tool_call_id':'old-complete','content':'{"ok":true}'}], 'pending_calls':[pending]}
    store.checkpoint(old, events=[{'type': 'run.started'}])
    old.update(store.interrupt(old))
    async def run():
        controller=RunController(store);llm=FixtureLLMClient([{'content':'已从已保存的工具批次继续。'}])
        try:
            task,_=controller.submit(session['session_id'],request(page,message='resume',revision=1,text='继续',resume=old['run_id']),llm,service)
            done=await controller.wait(task)
            assert done['status']=='completed',done
            assert len(service.validate_calls)==1 and len(llm.requests)==1
            messages=llm.requests[0]['messages']
            assert [m.get('tool_call_id') for m in messages if m['role']=='tool']==['old-complete','old-pending']
        finally:await controller.close()
    asyncio.run(run())


def test_storage_checkpoint_failure_keeps_last_draft_and_can_be_reconciled(tmp_path,monkeypatch):
    store,page,session,service=setup(tmp_path)
    real_transition=store._save_transition
    def unavailable(*args,**kwargs):raise AgentError('AGENT_STORAGE_UNAVAILABLE','存储不可用',status_code=503)
    async def run():
        controller=RunController(store)
        try:
            monkeypatch.setattr(store,'_save_transition',unavailable)
            llm=FixtureLLMClient([{'content':'不应执行模型'}])
            task,_=controller.submit(session['session_id'],request(page),llm,service)
            await controller.wait(task)
            assert not llm.requests and store.read(session['session_id'])['draft'] is None
            monkeypatch.setattr(store,'_save_transition',real_transition)
            await asyncio.sleep(0)
            controller.reconcile()
            done=store.get_run(session['session_id'],task['run_id'])
            assert done['status']=='interrupted' and done['stop_reason']=='storage_unavailable'
            assert not store.read(session['session_id'])['active_run_id']
            next_run,_=controller.submit(session['session_id'],request(page,message='continue',revision=1,text='继续',resume=task['run_id']),FixtureLLMClient([{'content':'已接续原始要求'}]),service)
            done=await controller.wait(next_run)
            assert any(m.get('content')=='查询指标' for m in done['checkpoint']['messages'])
        finally:await controller.close()
    asyncio.run(run())


def test_shutdown_does_not_release_a_running_tool_early(tmp_path):
    store,page,session,service=setup(tmp_path);entered=threading.Event();release=threading.Event()
    def validate(definition):entered.set();release.wait(5);return {'valid':True}
    service.validate=validate
    async def run():
        controller=RunController(store)
        task,_=controller.submit(session['session_id'],request(page),FixtureLLMClient([{'tool_calls':[{'name':'metrics.validate','arguments':{'definition':DEFINITION}}]}]),service)
        try:
            await until(entered.is_set)
            await controller.close()
            assert controller.futures and not controller.owner_file.closed
            assert store.get_run(session['session_id'],task['run_id'])['status']=='interrupted'
            assert store.read(session['session_id'])['draft'] is None
            release.set();await until(lambda:not controller.futures)
            assert controller.owner_file.closed
        finally:release.set()
    asyncio.run(run())


@pytest.mark.parametrize('boundary', ['data.admitted', 'tool.started'])
def test_foreign_cancellation_before_dispatch_blocks_new_model_or_tool_work(tmp_path, monkeypatch, boundary):
    monkeypatch.setenv('CUSTOM_INDICATOR_DATA_DIR', str(tmp_path))
    store, page, session, service = setup(tmp_path)
    sid = session['session_id']
    foreign = AgentSessionStore(store.root)
    real_admit = store.admit
    cancelled = []
    def checkpoint(run, **kwargs):
        if not cancelled and any(event.get('type') == boundary for event in kwargs.get('events', ())):
            foreign.cancel(sid, run['run_id'])
            cancelled.append(run['run_id'])
        return real_admit(run, **kwargs)
    monkeypatch.setattr(store, 'admit', checkpoint)
    llm = FixtureLLMClient([
        {'tool_calls': [{'name': 'metrics.validate', 'arguments': {'definition': DEFINITION}}]},
        {'content': '这条回复不得发起。'}])
    async def run():
        controller = RunController(store)
        try:
            task, _ = controller.submit(sid, request(page), llm, service)
            done = await controller.wait(task)
            assert cancelled == [task['run_id']], done
            assert done['status'] == 'cancelled' and done['stop_reason'] == 'user_cancelled'
            assert len(llm.requests) == (0 if boundary == 'data.admitted' else 1)
            assert service.validate_calls == [] and service.create_calls == []
            assert store.read(sid)['draft'] is None and not controller.futures
            assert not store.read(sid).get('execution_blocked_by') and not done.get('execution_blocked_by')
        finally:
            await controller.close()
    asyncio.run(run())


@pytest.mark.parametrize('phase', ['primary', 'finalization'])
@pytest.mark.parametrize('outcome', ['reply', 'error'])
@pytest.mark.parametrize('accepted_first', [False, True])
def test_model_return_admission_orders_foreign_cancel_and_result(tmp_path, monkeypatch, phase, outcome, accepted_first):
    from agent.llm import LLMUnavailableError

    monkeypatch.setenv('CUSTOM_INDICATOR_DATA_DIR', str(tmp_path))
    store, page, session, service = setup(tmp_path)
    sid = session['session_id']
    foreign = AgentSessionStore(store.root)
    marker = '仅在取消前接纳的模型正文可沿用'
    admitted, cancelled_after_return = [], []
    real_admit = store.admit
    def checkpoint(run, **kwargs):
        result = real_admit(run, **kwargs)
        if any(e.get('type') == 'model.returned' for e in kwargs.get('events', ())):
            admitted.append(result)
            if accepted_first and not cancelled_after_return and run['phase'] == ('summarizing' if phase == 'finalization' else 'thinking'):
                foreign.cancel(sid, run['run_id'])
                cancelled_after_return.append(run['run_id'])
        return result
    monkeypatch.setattr(store, 'admit', checkpoint)

    async def run():
        entered, release = asyncio.Event(), asyncio.Event()
        class Delayed(FixtureLLMClient):
            async def complete(self, **kwargs):
                if phase == 'finalization' and kwargs.get('tools'):
                    return await super().complete(**kwargs)
                entered.set()
                await release.wait()
                if outcome == 'error':
                    raise LLMUnavailableError('迟到接口失败', code='AGENT_LLM_HTTP_ERROR')
                return LLMReply(content=marker, usage={'prompt_tokens': 17, 'completion_tokens': 9})
        model = Delayed([{'tool_calls': [{'name': 'metrics.lookup', 'arguments': {'query': '演示'}}]}] * 12)
        controller = RunController(store)
        try:
            task, _ = controller.submit(sid, request(page), model, service)
            await asyncio.wait_for(entered.wait(), 5)
            if not accepted_first:
                foreign.cancel(sid, task['run_id'])
                assert not controller.stops[task['run_id']].is_set()
            release.set()
            done = await controller.wait(task)
            assert admitted[-1] is accepted_first
            assert done['status'] == ('failed' if accepted_first and outcome == 'error' and phase == 'primary' else 'cancelled')
            assert marker not in stable_json(done['checkpoint'])
            if outcome == 'reply':
                assert done['usage']['prompt_tokens'] >= 17 and done['usage']['completion_tokens'] >= 9
            else:
                assert bool(done.get('error')) is (accepted_first and phase == 'primary')
            for index, resume in enumerate((None, done['run_id'])):
                followup = FixtureLLMClient([{'content': '新的分析'}])
                next_run, _ = controller.submit(sid, request(page, message=f'next-{index}', text='继续换个角度。',
                    revision=store.read(sid)['session_revision'], resume=resume), followup, service)
                assert (await controller.wait(next_run))['status'] == 'completed'
                assert marker not in stable_json(followup.requests)
            assert service.create_calls == []
        finally:
            release.set()
            await controller.close()
    asyncio.run(run())


@pytest.mark.parametrize('cancel_first', [False, True])
def test_finish_and_foreign_cancel_commit_in_one_order(tmp_path, monkeypatch, cancel_first):
    monkeypatch.setenv('CUSTOM_INDICATOR_DATA_DIR', str(tmp_path))
    store, page, session, service = setup(tmp_path)
    sid = session['session_id']
    foreign = AgentSessionStore(store.root)
    attempted, finished = threading.Event(), threading.Event()
    observations = []
    async def run():
        controller = RunController(store)
        worker = None
        def cancel(rid):
            attempted.set()
            observations.append(foreign.cancel(sid, rid)['status'])
            finished.set()
        real_finish, real_transition = controller._finish, store._save_transition
        async def finish(run, text, **kwargs):
            nonlocal worker
            if cancel_first:
                worker = threading.Thread(target=cancel, args=(run['run_id'],))
                worker.start()
                assert await asyncio.to_thread(finished.wait, 5)
            return await real_finish(run, text, **kwargs)
        def transition(db, current, run, state, events=()):
            nonlocal worker
            if not cancel_first and any(e.get('type') == 'assistant.message' for e in events):
                worker = threading.Thread(target=cancel, args=(run['run_id'],))
                worker.start()
                assert attempted.wait(5)
            return real_transition(db, current, run, state, events)
        monkeypatch.setattr(controller, '_finish', finish)
        monkeypatch.setattr(store, '_save_transition', transition)
        try:
            task, _ = controller.submit(sid, request(page), FixtureLLMClient([{'content': '已完成并提交的回复'}]), service)
            done = await controller.wait(task)
            assert await asyncio.to_thread(finished.wait, 5)
            assert observations == ['stopping' if cancel_first else 'completed']
            assert done['status'] == ('cancelled' if cancel_first else 'completed')
            assert ('已完成并提交的回复' in stable_json(done['checkpoint'])) is not cancel_first
            assert not store.read(sid)['active_run_id']
        finally:
            if worker is not None:
                worker.join(5)
            await controller.close()
    asyncio.run(run())


@pytest.mark.parametrize('ending', ['cancel_reply', 'cancel_error', 'epoch_reply'])
def test_legacy_compaction_return_uses_same_durable_admission(tmp_path, monkeypatch, ending):
    from agent.context import compact_if_needed
    from agent.harness import RunModel, RunStopped
    from agent.llm import LLMUnavailableError
    from agent.research_runtime import catalog_version
    from test_agent_context import exchange, signed_summary

    monkeypatch.setenv('CUSTOM_INDICATOR_DATA_DIR', str(tmp_path))
    monkeypatch.setenv('AGENT_CONTEXT_CHAR_LIMIT', '18000')
    store, page, session, service = setup(tmp_path)
    sid = session['session_id']
    foreign = AgentSessionStore(store.root)
    async def run():
        entered, release = asyncio.Event(), asyncio.Event()
        class Waiting(FixtureLLMClient):
            async def complete(self, **kwargs):
                entered.set()
                await release.wait()
                if ending == 'cancel_error':
                    raise LLMUnavailableError('迟到整理失败', code='AGENT_LLM_HTTP_ERROR')
                return await super().complete(**kwargs)
        controller = RunController(store)
        llm = Waiting([{'content': '不可安装的迟到摘要', 'usage': {'prompt_tokens': 17, 'completion_tokens': 5}}])
        llm.context_window_tokens = 16384
        checkpoint = {'messages': exchange(reasoning=16000), **signed_summary('历史材料。' * 1200)}
        before = copy.deepcopy(checkpoint)
        task, _ = store.accept(sid, request(page), {'owner_instance': controller.instance_id, 'owner_workspace': controller.workspace})
        task.update(status='running', phase='compacting', catalog_version=catalog_version(service), checkpoint=checkpoint)
        store.checkpoint(task, events=[{'type': 'run.started'}])
        model = RunModel(controller, task, llm, asyncio.Event(), service)
        work = asyncio.create_task(compact_if_needed(checkpoint, system='研究', llm=model, force=True,
            on_compacted=lambda evidence: store.checkpoint(task, events=[{'type': 'context.compacted', 'data': evidence}])))
        try:
            await asyncio.wait_for(entered.wait(), 5)
            if ending == 'epoch_reply':
                foreign.recover(foreign.get_run(sid, task['run_id']))
            else:
                foreign.cancel(sid, task['run_id'])
            release.set()
            with pytest.raises((RunStopped, AgentError)):
                await work
            assert checkpoint == before
            assert '不可安装的迟到摘要' not in stable_json(store.get_run(sid, task['run_id']))
            if ending != 'cancel_error':
                assert task['usage']['prompt_tokens'] == 17 and task['usage']['completion_tokens'] == 5
        finally:
            release.set()
            await asyncio.gather(work, return_exceptions=True)
            await controller.close()
    asyncio.run(run())


@pytest.mark.parametrize('cancelled', [False, True])
def test_prepared_context_is_committed_after_compaction_installs_its_candidate(tmp_path, monkeypatch, cancelled):
    from test_agent_context import exchange

    monkeypatch.setenv('CUSTOM_INDICATOR_DATA_DIR', str(tmp_path))
    monkeypatch.setenv('AGENT_CONTEXT_CHAR_LIMIT', '18000')
    store, page, session, _ = setup(tmp_path)
    sid = session['session_id']
    async def run():
        from agent.harness import RunStopped
        controller = RunController(store)
        task, _ = store.accept(sid, request(page), {'owner_instance': controller.instance_id, 'owner_workspace': controller.workspace})
        task.update(status='running', checkpoint={'messages': [{'role': 'user', 'content': '保留明确的研究口径'}, *exchange(reasoning=8000)]})
        assert store.checkpoint(task, events=[{'type': 'run.started'}])
        before = copy.deepcopy(task['checkpoint'])
        llm = FixtureLLMClient([])
        llm.context_window_tokens = 16384
        def during_compaction():
            if cancelled:
                AgentSessionStore(store.root).cancel(sid, task['run_id'])
        try:
            operation = controller._prepare_context(task, phase='thinking', system='研究', llm=llm,
                force=True, on_compacting=during_compaction)
            if cancelled:
                with pytest.raises(RunStopped):
                    await operation
            else:
                await operation
            persisted = store.get_run(sid, task['run_id'])['checkpoint']
            if cancelled:
                assert persisted == before
            else:
                assert persisted == task['checkpoint'] and persisted != before
                assert persisted['compaction_count'] == 1
            compacted = [e for e in store.events(sid)['items'] if e['type'] == 'context.compacted']
            assert len(compacted) == (0 if cancelled else 1)
        finally:
            await controller.close()
    asyncio.run(run())


def test_snapshot_returns_recent_chat_without_replaying_tool_history(tmp_path):
    store,page,session,service=setup(tmp_path)
    with store.locked(session['session_id']) as state:
        for i in range(250):
            append_event(state,{'type':'user.message','speaker':'user','text':f'问题{i}'})
            for j in range(3):append_event(state,{'type':'tool.completed','_model_message':{'content':'private process'}})
            append_event(state,{'type':'assistant.message','speaker':'assistant','text':f'回答{i}'})
        store.write(state)
    snapshot=store.public(session['session_id'])
    assert len(snapshot['messages'])==200 and snapshot['messages'][-1]['text']=='回答249'
    assert snapshot['next_event_seq']==1251
    earlier=store.message_page(session['session_id'],before_seq=snapshot['older_message_cursor'])
    assert len(earlier['items'])==200
    assert earlier['items'][-1]['seq']<snapshot['messages'][0]['seq']
    assert all('_model_message' not in item for item in snapshot['messages'])


def test_long_cycle_cannot_become_novel_when_recent_cache_is_evicted(tmp_path):
    from agent.progress import ProgressGuard
    store,page,session,service=setup(tmp_path)
    task,_=store.accept(session['session_id'],request(page),{'owner_instance':'test','owner_workspace':'test'})
    guard=ProgressGuard(seen=lambda kind,values:store.seen_facts(task['progress_space_id'],kind,values))
    for i in range(520):
        result={'ok':True,'result':{'items':[{'id':str(i)}]}}
        signal=guard.record('metrics.lookup',{'query':str(i)},result)
        assert signal['progress']
        store.checkpoint(task,new_facts=guard.pending_facts)
    for i in range(10):
        signal=guard.record('metrics.lookup',{'query':str(i)},{'ok':True,'result':{'items':[{'id':str(i)}]}})
        assert not signal['progress']
    assert signal['pause']
    assert len(guard.snapshot()['seen_facts'])<=512


def test_cancelled_preview_never_publishes_late_results(tmp_path):
    store, page, session, service = setup(tmp_path)
    entered, release = threading.Event(), threading.Event()
    with store.locked(session['session_id']) as state:
        store_draft(state, definition=DEFINITION, validation={'valid': True}, compile_token='a' * 64)
        state['product_candidates'] = [{'kind': 'etf', 'product_id': '510300.SH'}]
        store.write(state)
    def evaluate(**kwargs):
        entered.set(); release.wait(5)
        return {'results': [{'status': 'ok', 'value': .123}]}
    service.evaluate = evaluate
    async def run():
        controller = RunController(store)
        try:
            llm = FixtureLLMClient([{'tool_calls': [{'name': 'metrics.preview', 'arguments': {'target': {'kind': 'etf', 'product_id': '510300.SH'}}}]}])
            task, _ = controller.submit(session['session_id'], request(page), llm, service)
            await until(entered.is_set)
            controller.cancel(session['session_id'], task['run_id'])
            release.set()
            done = await controller.wait(task)
            assert done['status'] == 'cancelled'
            assert store.public(session['session_id'])['preview'] is None
            assert not any(e['type'] == 'preview.updated' for e in store.events(session['session_id'])['items'])
        finally:
            release.set(); await controller.close()
    asyncio.run(run())


def test_old_reply_artifacts_are_reconstructed_from_its_events_not_current_draft(tmp_path):
    store, page, session, service = setup(tmp_path)
    sid = session['session_id']
    original = {'valid': True, 'draft_revision': 1, 'definition_hash': 'old', 'definition': {'expression': 'mean(returns)'}}
    with store.locked(sid) as state:
        append_event(state, {'type': 'draft.updated', 'run_id': 'old-run', 'data': {'draft': original}})
        append_event(state, {'type': 'assistant.message', 'run_id': 'old-run', 'speaker': 'assistant', 'text': '旧回复'})
        append_event(state, {'type': 'assistant.message', 'run_id': 'new-run', 'speaker': 'assistant', 'text': '讨论新逻辑'})
        state['draft'] = {**original, 'definition_hash': 'current', 'definition': {'expression': 'std(returns)'}}
        store.write(state)
    messages = store.public(sid)['messages']
    assert messages[0]['artifacts'] == {'draft': original}
    assert messages[1]['artifacts'] == {}
    assert store.message_page(sid, before_seq=messages[1]['seq'])['items'][0]['artifacts'] == {'draft': original}


def test_reasoning_survives_tools_and_turns_but_not_api_switch_or_public_state(tmp_path):
    import httpx
    from agent.llm import HttpLLMClient
    from agent.sessions import public_run
    store, page, session, service = setup(tmp_path)
    sid = session['session_id']
    requests = []
    replies = [
        {'content': '', 'reasoning_content': 'private-tool-reasoning', 'tool_calls': [
            {'id': 'call-1', 'type': 'function', 'function': {'name': 'metrics_lookup', 'arguments': '{"query":"收益"}'}}]},
        {'content': '找到目录。', 'reasoning_content': 'private-final-reasoning'},
        {'content': '继续讨论。', 'reasoning_content': 'private-next-reasoning'},
        {'content': '已切换。'},
    ]
    def transport(req):
        requests.append(req)
        return httpx.Response(200, json={'choices': [{'message': replies.pop(0)}]})
    def client(base='https://opencode.ai/zen/go/v1', model='deepseek-v4.1-flash'):
        return HttpLLMClient(base_url=base, api_key='fixture-key', model=model, reasoning_effort='max',
                             session_id=sid, transport=httpx.MockTransport(transport))
    async def run():
        controller = RunController(store)
        try:
            for number in range(3):
                rev = store.public(sid)['session_revision']
                llm = client() if number < 2 else client('https://other.example/v1', 'another-model')
                task, _ = controller.submit(sid, request(page, message=f'turn-{number}', revision=rev), llm, service)
                done = await controller.wait(task)
                assert done['status'] == 'completed', done.get('error')
                assert 'private-' not in json.dumps(public_run(done))
            second = json.loads(requests[1].content)['messages']
            assert next(m for m in second if m['role'] == 'assistant')['reasoning_content'] == 'private-tool-reasoning'
            third = json.loads(requests[2].content)['messages']
            assert any(m.get('reasoning_content') == 'private-final-reasoning' for m in third)
            switched = json.loads(requests[3].content)['messages']
            assert all('reasoning_content' not in m for m in switched)
            assert all(r.headers['x-opencode-session'] == sid for r in requests[:3])
            assert 'x-opencode-session' not in requests[3].headers
            assert 'private-' not in json.dumps(store.public(sid))
            assert 'private-' not in json.dumps(store.events(sid))
            assert 'private-' not in json.dumps(store.message_page(sid))
        finally:
            await controller.close()
    asyncio.run(run())


@pytest.mark.parametrize('overflow_again', [False, True])
def test_context_overflow_compacts_then_retries_once_without_reexecuting_tools(tmp_path, overflow_again):
    import httpx
    from agent.llm import HttpLLMClient
    store,page,session,service=setup(tmp_path)
    requests=[]
    reasoning='private thought '*4000
    def handler(req):
        payload=json.loads(req.content); requests.append(payload)
        if len(requests)==1:
            return httpx.Response(200,json={'choices':[{'message':{'content':'','reasoning_content':reasoning,
                'tool_calls':[{'id':'lookup-once','function':{'name':'metrics_lookup','arguments':'{}'}}]}}]})
        if len(requests)==2 or overflow_again:
            return httpx.Response(400,json={'error':{'code':'context_length_exceeded','message':'private error'}})
        return httpx.Response(200,json={'choices':[{'message':{'content':'继续研究。','reasoning_content':'done'}}]})
    async def run():
        controller=RunController(store)
        llm=HttpLLMClient(base_url='https://opencode.ai/zen/go/v1',api_key='fixture',model='deepseek-v4.1-flash',
                          reasoning_effort='max',session_id=session['session_id'],transport=httpx.MockTransport(handler))
        try:
            started,_=controller.submit(session['session_id'],request(page,text='20日窗口，不用未来数据'),llm,service)
            done=await controller.wait(started)
            assert done['status']==('paused' if overflow_again else 'completed'),done.get('error')
            assert len(requests)==3 and done['usage']['tool_calls']==1
            assert next(m['reasoning_content'] for m in requests[1]['messages'] if m.get('reasoning_content'))==reasoning
            assert all(m.get('reasoning_content')!=reasoning for m in requests[2]['messages'])
            assert any(m.get('content')=='20日窗口，不用未来数据' for m in requests[2]['messages'])
            with store.connection() as db:
                assert db.execute('SELECT COUNT(*) FROM tool_calls').fetchone()[0]==1
            events=store.events(session['session_id'])['items']
            assert any(e['type']=='context.compacted' for e in events)
            assert any(e['type']=='run.phase' and e['data'].get('phase')=='compacting' for e in events)
            assert reasoning not in stable_json(events)
        finally: await controller.close()
    asyncio.run(run())


def test_context_evidence_read_pages_original_result_and_is_session_scoped(tmp_path):
    from agent.llm import LLMToolCall
    store,page,session,service=setup(tmp_path)
    other=store.create(page_context=page,scope='indicator_center')
    class Reader(FixtureLLMClient):
        def __init__(self): super().__init__([]);self.reference=None
        async def complete(self, **kw):
            self.requests.append(copy.deepcopy(kw))
            if len(self.requests)==1:
                return LLMReply(tool_calls=[LLMToolCall('metrics.lookup',{},'lookup')])
            if len(self.requests)==2:
                self.reference=json.loads(kw['messages'][-1]['content'])['context_ref']
                return LLMReply(tool_calls=[LLMToolCall('context.read',{'operation_id':self.reference,'limit':100},'read')])
            result=json.loads(kw['messages'][-1]['content'])
            assert result['ok'] and len(result['result']['content'])==100
            return LLMReply(content='已回读历史证据。')
    async def run():
        controller=RunController(store);llm=Reader()
        try:
            started,_=controller.submit(session['session_id'],request(page),llm,service)
            done=await controller.wait(started)
            assert done['status']=='completed',done.get('error')
            pieces=[];offset=0
            while True:
                result=store.read_context(session['session_id'],llm.reference,offset,100)['result'];pieces.append(result['content'])
                if result['next_offset'] is None: break
                offset=result['next_offset']
            original=json.loads(llm.requests[1]['messages'][-1]['content'])
            assert json.loads(''.join(pieces))==original
            for sid,ref in [(other['session_id'],llm.reference),(session['session_id'],'op-'+'0'*32)]:
                with pytest.raises(AgentError) as exc:store.read_context(sid,ref)
                assert exc.value.code=='AGENT_EVIDENCE_NOT_FOUND'
            with store.connection(write=True) as db:
                db.execute("UPDATE tool_calls SET body=json_set(body,'$.applied',0) WHERE json_extract(body,'$.operation_id')=?",(llm.reference,))
            with pytest.raises(AgentError): store.read_context(session['session_id'],llm.reference)
        finally: await controller.close()
    asyncio.run(run())


def test_old_compacted_checkpoint_restores_literal_goal_and_drops_stale_deferred_continue(tmp_path):
    store,page,session,service=setup(tmp_path)
    old,_=store.accept(session['session_id'],request(page,text='用20日窗口，禁止未来数据'),{'owner_instance':'dead','owner_workspace':'old'})
    old.update(status='running', checkpoint={'model_step':1,'summary':'旧版摘要，仅保留大意',
               'messages':[{'role':'assistant','content':'此前已查询目录'}], 'deferred_user':'继续'})
    store.checkpoint(old, events=[{'type': 'run.started'}])
    old.update(store.finish(old, '此前已查询目录', status='paused', reason='context_capacity'))
    async def run():
        controller=RunController(store);llm=FixtureLLMClient([{'content':'将按30日窗口继续。'}])
        try:
            task,_=controller.submit(session['session_id'],request(page,message='new',revision=1,text='改为30日窗口',resume=old['run_id']),llm,service)
            done=await controller.wait(task)
            assert done['status']=='completed',done.get('error')
            users=[m['content'] for m in llm.requests[0]['messages'] if m['role']=='user']
            assert '用20日窗口，禁止未来数据' in users
            assert users[-1]=='改为30日窗口' and '继续' not in users
            assert 'deferred_user' not in done['checkpoint']
        finally: await controller.close()
    asyncio.run(run())


def test_changed_window_releases_old_learned_ceiling_without_stripping_reasoning(tmp_path):
    store,page,session,service=setup(tmp_path)
    key=('https://example.com/v1','deepseek-v4.1-flash')
    from agent.sessions import stable_hash
    with store.locked(session['session_id']) as state:
        state['conversation']={'messages':[{'role':'assistant','content':'已确认20日窗口','reasoning_content':'original protocol context'}],
            'llm_context_key':stable_hash(key),'capacity_key':stable_hash([stable_hash(key),32768]),
            'observed_input_budget':100,'token_ratio':1,'model_step':0}
        store.write(state)
    async def run():
        controller=RunController(store);llm=FixtureLLMClient([{'content':'继续。'}])
        llm.context_key=key;llm.context_window_tokens=65536
        try:
            task,_=controller.submit(session['session_id'],request(page,text='继续'),llm,service)
            done=await controller.wait(task)
            assert done['status']=='completed',done.get('error')
            assert 'observed_input_budget' not in done['checkpoint']
            assert next(m['reasoning_content'] for m in llm.requests[0]['messages'] if m.get('reasoning_content'))=='original protocol context'
        finally:await controller.close()
    asyncio.run(run())


class StoppedTurnClient(FixtureLLMClient):
    """Answers the scripted prefix, then blocks and reports entry to the test thread."""

    def __init__(self, replies):
        super().__init__(replies)
        self.entered = asyncio.Event()

    async def complete(self, **kwargs):
        if self._replies:
            return await super().complete(**kwargs)
        self.entered.set()
        await asyncio.Event().wait()


def stopped_turn(controller, session_id, page, service, *, message, revision, text, tool=None, edit=None):
    """Run one exhausted-model turn, stop it, and keep whatever tool evidence it produced."""
    call = tool or {'name': 'metrics.lookup', 'arguments': {'kind': 'indicators', 'query': '演示'}}
    llm = StoppedTurnClient([{'tool_calls': [call]}])
    task, _ = controller.submit(session_id, request(page, message=message, revision=revision, text=text, edit=edit), llm, service)
    return task, llm


def test_edit_stopped_turn_drops_replaced_turn_and_keeps_earlier_memory(tmp_path):
    store,page,session,service=setup(tmp_path)
    sid=session['session_id']
    async def run():
        controller=RunController(store)
        try:
            first=FixtureLLMClient([{'content':'第一轮答复'}])
            task,_=controller.submit(sid,request(page,message='m1',text='第一轮需求'),first,service)
            done=await controller.wait(task)
            assert done['status']=='completed'
            with store.connection(write=True) as db:
                stored=store._read_run(db,sid,done['run_id'])
                stored['checkpoint']['summary']='历史摘要：此前只确认了第一轮需求。'
                # This test preserves an admitted prior summary across editing; unsigned
                # legacy-summary rejection is covered separately by admission tests.
                from agent import data_policy
                stored['checkpoint']['summary_seal']=data_policy.seal_text(stored['checkpoint']['summary'],'summary')
                store._write_run(db,stored)
            stopped,llm=stopped_turn(controller,sid,page,service,message='m2',revision=1,text='被停止的旧需求')
            await llm.entered.wait()
            controller.cancel(sid,stopped['run_id'])
            cancelled=await controller.wait(stopped)
            assert cancelled['status']=='cancelled'
            assert '被停止的旧需求' in stable_json(cancelled['checkpoint'])
            assert any(message['role']=='tool' for message in cancelled['checkpoint']['messages'])

            edited=FixtureLLMClient([{'content':'新的答复'}])
            replaced,_=controller.submit(sid,request(page,message='m3',revision=2,text='改写后的新需求',edit='m2'),edited,service)
            # Accepting the replacement already removed the stopped turn from the stored conversation.
            assert '被停止的旧需求' not in stable_json(store.read(sid).get('conversation'))
            finished=await controller.wait(replaced)
            assert finished['status']=='completed',finished.get('error')
            payload=stable_json(edited.requests[0]['messages'])
            assert '改写后的新需求' in payload and '历史摘要：此前只确认了第一轮需求。' in payload
            assert '被停止的旧需求' not in payload and 'metrics.lookup' not in payload
            assert '已停止自动处理' not in payload
            messages=store.message_page(sid)['items']
            assert [message['text'] for message in messages if message['speaker']=='user']==['第一轮需求','改写后的新需求']
            assert [message['text'] for message in messages if message['speaker']=='assistant']==['第一轮答复','新的答复']
            with store.connection() as db:
                audit=[json.loads(row[0]) for row in db.execute("SELECT body FROM events WHERE session_id=? AND json_extract(body,'$.type')='user.message.superseded'",(sid,))]
            assert [record['text'] for record in audit]==['被停止的旧需求'] and audit[0]['superseded_by']=='m3'
        finally: await controller.close()
    asyncio.run(run())


def test_edit_fallback_drops_summary_and_pins_of_the_replaced_turn(tmp_path):
    store,page,session,service=setup(tmp_path)
    sid=session['session_id']
    async def run():
        controller=RunController(store)
        try:
            first=FixtureLLMClient([{'content':'第一轮答复'}])
            task,_=controller.submit(sid,request(page,message='m1',text='第一轮需求'),first,service)
            assert (await controller.wait(task))['status']=='completed'
            stopped,llm=stopped_turn(controller,sid,page,service,message='m2',revision=1,text='被停止的旧需求')
            await llm.entered.wait()
            controller.cancel(sid,stopped['run_id'])
            assert (await controller.wait(stopped))['status']=='cancelled'
            # Legacy runs carry no parent snapshot; the stored conversation is the only base left.
            with store.connection(write=True) as db:
                stored=store._read_run(db,sid,stopped['run_id'])
                stored.pop('parent_run_id',None)
                store._write_run(db,stored)
            with store.locked(sid) as state:
                state['conversation']['summary']='摘要里含被停止的旧需求'
                state['conversation']['pinned_user_messages']=[{'role':'user','content':'被停止的旧需求'}]
                append_event(state,{'type':'context.compacted','run_id':stopped['run_id'],'data':{'method':'summary'}})
                store.write(state)

            edited=FixtureLLMClient([{'content':'新的答复'}])
            task,_=controller.submit(sid,request(page,message='m3',revision=2,text='改写后的新需求',edit='m2'),edited,service)
            done=await controller.wait(task)
            assert done['status']=='completed',done.get('error')
            payload=stable_json(edited.requests[0]['messages'])
            assert '改写后的新需求' in payload and '第一轮答复' in payload
            assert '被停止的旧需求' not in payload and '摘要里含' not in payload
        finally: await controller.close()
    asyncio.run(run())


def test_repeated_edit_keeps_ancestry_and_restores_invalid_draft_and_memory(tmp_path):
    store,page,session,service=setup(tmp_path)
    sid=session['session_id']
    invalid={'name':'AI 无效草稿','expression':'unknown(x)','periods':['1Y'],'dsl_version':'2.1.0','context_kind':'single_product','result_kind':'scalar'}
    async def run():
        controller=RunController(store)
        try:
            first=FixtureLLMClient([{'content':'第一轮答复'}])
            task,_=controller.submit(sid,request(page,message='m1',text='第一轮需求'),first,service)
            assert (await controller.wait(task))['status']=='completed'
            # A stopped turn whose only result is an invalid draft: never shown, so still replaceable.
            stopped,llm=stopped_turn(controller,sid,page,service,message='m2',revision=1,text='第一次被停止的需求',
                tool={'name':'metrics.validate','arguments':{'definition':invalid}})
            await llm.entered.wait()
            controller.cancel(sid,stopped['run_id'])
            cancelled=await controller.wait(stopped)
            assert cancelled['status']=='cancelled'
            assert cancelled['response']['artifacts']['draft']['valid'] is False
            assert store.read(sid)['draft']['valid'] is False

            # The first replacement is stopped as well, again with an invalid draft and compacted memory.
            stopped2,llm2=stopped_turn(controller,sid,page,service,message='m3',revision=2,text='第一次改写',edit='m2',
                tool={'name':'metrics.validate','arguments':{'definition':invalid}})
            await llm2.entered.wait()
            controller.cancel(sid,stopped2['run_id'])
            assert (await controller.wait(stopped2))['status']=='cancelled'
            with store.connection(write=True) as db:
                stored=store._read_run(db,sid,stopped2['run_id'])
                stored['checkpoint']['summary']='压缩记忆：第一次改写与 unknown(x)'
                store._write_run(db,stored)
                db.execute("INSERT INTO events VALUES (?,?,?)",(sid,9000,stable_json({'seq':9000,'type':'context.compacted','run_id':stopped2['run_id'],'data':{'method':'summary'}})))
            assert store.read(sid)['draft']['valid'] is False

            # Replacing the replacement must start from the state before the first stop, not from it.
            final=FixtureLLMClient([{'content':'第二次改写答复'}])
            task,_=controller.submit(sid,request(page,message='m4',revision=3,text='第二次改写',edit='m3'),final,service)
            assert store.read(sid).get('draft') is None
            assert '第一次改写' not in stable_json(store.read(sid).get('conversation'))
            done=await controller.wait(task)
            assert done['status']=='completed',done.get('error')
            payload=stable_json(final.requests[0])
            assert '第二次改写' in payload and '第一轮答复' in payload
            assert '第一次被停止的需求' not in payload and '第一次改写' not in payload
            assert 'unknown(x)' not in payload and '压缩记忆' not in payload and '已停止自动处理' not in payload
            messages=store.message_page(sid)['items']
            assert [message['text'] for message in messages if message['speaker']=='user']==['第一轮需求','第二次改写']
            assert [message['text'] for message in messages if message['speaker']=='assistant']==['第一轮答复','第二次改写答复']
            with store.connection() as db:
                audit=[json.loads(row[0]) for row in db.execute("SELECT body FROM events WHERE session_id=? AND json_extract(body,'$.type')='user.message.superseded' ORDER BY seq",(sid,))]
            assert [record['text'] for record in audit]==['第一次被停止的需求','第一次改写']
            assert [record['superseded_by'] for record in audit]==['m3','m4']
        finally: await controller.close()
    asyncio.run(run())


def test_legacy_edit_infers_previous_turn_and_keeps_earlier_artifacts(tmp_path):
    store,page,session,service=setup(tmp_path)
    sid=session['session_id']
    invalid={'name':'AI 无效草稿','expression':'unknown(x)','periods':['1Y'],'dsl_version':'2.1.0','context_kind':'single_product','result_kind':'scalar'}
    async def run():
        controller=RunController(store)
        try:
            first=FixtureLLMClient([{'content':'第一轮答复'}])
            task,_=controller.submit(sid,request(page,message='m1',text='第一轮需求'),first,service)
            assert (await controller.wait(task))['status']=='completed'
            with store.locked(sid) as state:
                store_draft(state, definition={**DEFINITION,'name':'早前有效草稿'}, validation={'valid':True}, compile_token='a'*64)
                store.write(state)
            stopped,llm=stopped_turn(controller,sid,page,service,message='m2',revision=1,text='被停止的旧需求',
                tool={'name':'metrics.validate','arguments':{'definition':invalid}})
            await llm.entered.wait()
            controller.cancel(sid,stopped['run_id'])
            assert (await controller.wait(stopped))['status']=='cancelled'
            # A legacy session: the run has neither ancestry nor per-turn snapshot, its state holds the
            # stopped turn's invalid draft, and the earlier valid draft/preview and compacted memory remain.
            with store.connection(write=True) as db:
                stored=store._read_run(db,sid,stopped['run_id'])
                stored.pop('parent_run_id',None)
                stored.pop('base_state',None)
                store._write_run(db,stored)
            with store.locked(sid) as state:
                assert state['draft']['valid'] is False and state['last_valid_draft']['valid'] is True
                state['preview']={'preview_id':'preview-old','definition_hash':state['last_valid_draft']['definition_hash']}
                state['conversation']['summary']='压缩记忆：被停止的旧需求与 unknown(x)'
                store.write(state)

            edited=FixtureLLMClient([{'content':'新的答复'}])
            task,_=controller.submit(sid,request(page,message='m3',revision=2,text='改写后的新需求',edit='m2'),edited,service)
            current=store.read(sid)
            assert current['draft']['valid'] is True and current['draft']['definition']['name']=='早前有效草稿'
            assert current['preview']=={'preview_id':'preview-old','definition_hash':current['draft']['definition_hash']}
            done=await controller.wait(task)
            assert done['status']=='completed',done.get('error')
            payload=stable_json(edited.requests[0])
            assert '改写后的新需求' in payload and '第一轮答复' in payload
            assert '被停止的旧需求' not in payload and 'unknown(x)' not in payload and '压缩记忆' not in payload
            assert store.get_run(sid,stopped['run_id'])['superseded_by']=='m3'
        finally: await controller.close()
    asyncio.run(run())


def interrupted_page_read_run(store, service, session, page, *, message, revision, snapshot, arguments=None):
    """A run interrupted after the model planned page reads that never executed."""
    from agent.catalog import build_catalog
    from custom_indicators.series_provider import market_data_generation
    run,_=store.accept(session['session_id'], request(page, message=message, revision=revision, text='读出页面证据', snapshot=snapshot), {'owner_instance':'dead','owner_workspace':'old'})
    run.update(status='running', data_generation=market_data_generation(service.market_data_dir), catalog_version=build_catalog(service)['version'])
    pending={'id':'old-pending','name':'page.read','arguments':arguments or {'section':'results','offset':4000,'limit':4000}}
    run['checkpoint']={'model_step':1,'messages':[
        {'role':'user','content':'读出页面证据'},
        {'role':'assistant','content':'','tool_calls':[{'id':'old-pending','type':'function','function':{'name':'page.read','arguments':json.dumps(pending['arguments'])}}]}],
        'pending_calls':[pending]}
    store.checkpoint(run, events=[{'type': 'run.started'}])
    run.update(store.interrupt(run))
    return run


def test_resume_replays_pending_page_reads_only_for_identical_snapshot_content(tmp_path):
    store,page,session,service=setup(tmp_path)
    snapshot={'version':1,'snapshot_id':'snap-'+'b'*32,'captured_at':'2026-09-21T02:00:00+00:00','page':'indicator-studio',
              'sections':{'editing':{'runtime_inputs':{'period':'S1'}},'results':{'displayed_source':'S1','pending':[]}}}
    old=interrupted_page_read_run(store, service, session, page, message='parent', revision=0, snapshot=snapshot)
    async def run():
        controller=RunController(store);llm=FixtureLLMClient([{'content':'已按新页面证据重新规划。'}])
        try:
            # Different snapshot content: the pending call must be closed, never replayed under S2.
            changed=FixtureLLMClient([{'content':'已按新页面证据重新规划。'}])
            task,_=controller.submit(session['session_id'], request(page, message='resume-changed', revision=1, text='继续', resume=old['run_id'],
                                                                   snapshot=evidence('S2', 'snap-'+'e'*32)), changed, service)
            done=await controller.wait(task)
            assert done['status']=='completed',done
            pending_result=[json.loads(message['content']) for message in changed.requests[0]['messages'] if message['role']=='tool' and message['tool_call_id']=='old-pending']
            assert pending_result and pending_result[0]['status']=='not_executed', '不同快照不得复用旧批次'
            executed=[item for item in (json.loads(message['content']) for message in changed.requests[0]['messages'] if message['role']=='tool')
                      if isinstance(item.get('result'),dict) and item['result'].get('available')]
            assert executed==[], '旧批次的 page.read 不得在新快照下执行'
        finally:await controller.close()
    asyncio.run(run())


def test_resume_keeps_safe_contract_for_identical_snapshot_content(tmp_path):
    store,page,session,service=setup(tmp_path)
    first=evidence('S1', 'snap-'+'c'*32)
    old=interrupted_page_read_run(store, service, session, page, message='parent', revision=0, snapshot=first, arguments={'section':'results'})
    async def run():
        controller=RunController(store);llm=FixtureLLMClient([{'content':'已继续读取。'}])
        try:
            same=evidence('S1', 'snap-'+'d'*32)  # same content, new random id and capture time
            task,_=controller.submit(session['session_id'], request(page, message='resume-same', revision=1, text='继续', resume=old['run_id'], snapshot=same), llm, service)
            done=await controller.wait(task)
            assert done['status']=='completed',done
            assert len(llm.requests)==1
            payload=[json.loads(message['content']) for message in llm.requests[0]['messages'] if message['role']=='tool' and message['tool_call_id']=='old-pending'][0]
            assert payload['ok'] is True and payload['result']['available'] is True
            assert payload['result']['snapshot_id']=='snap-'+'d'*32, '续跑读取的是本次运行自己的快照'
            assert '"displayed_source":"S1"' in payload['result']['content'], '相同内容的快照仍可安全续跑'
        finally:await controller.close()
    asyncio.run(run())


def test_page_read_no_progress_depends_on_section_content_not_snapshot_id(tmp_path):
    store,page,session,service=setup(tmp_path)
    repeated={'name':'page.read','arguments':{'section':'results'}}
    async def run():
        controller=RunController(store)
        try:
            parent_llm=FixtureLLMClient([{'tool_calls':[repeated]} for _ in range(5)]+[{'content':'没有新证据，先暂停。'}])
            task,_=controller.submit(session['session_id'], request(page, message='parent', snapshot=evidence('A','snap-'+'f'*32)), parent_llm, service)
            paused=await controller.wait(task)
            assert paused['status']=='paused' and paused['stop_reason']=='no_progress', paused

            # Unchanged content with a new random id and timestamp must stay blocked across continue.
            unchanged=FixtureLLMClient([{'tool_calls':[repeated]},{'content':'按已有页面证据回答。'}])
            next_run,_=controller.submit(session['session_id'], request(page, message='continue-1', revision=1, text='继续', resume=paused['run_id'],
                                                                       snapshot=evidence('A','snap-'+'a'*31+'1')), unchanged, service)
            blocked=await controller.wait(next_run)
            assert blocked['status']=='completed',blocked
            assert [(entry['tool'],entry['status'],entry['error_code']) for entry in blocked['tool_trace']]==[('page.read','error','AGENT_NO_PROGRESS')], '未变化的分区不得重新执行'
            denied=tool_messages(unchanged,1)[-1]
            assert denied['status']=='blocked' and denied['error']['code']=='AGENT_NO_PROGRESS'

            # Changed content unblocks the same call and reports real progress.
            changed_llm=FixtureLLMClient([{'tool_calls':[repeated]},{'content':'新证据已读到。'}])
            last_run,_=controller.submit(session['session_id'], request(page, message='continue-2', revision=2, text='继续', resume=blocked['run_id'],
                                                                       snapshot=evidence('B','snap-'+'a'*31+'2')), changed_llm, service)
            moved=await controller.wait(last_run)
            assert moved['status']=='completed',moved
            assert [(entry['tool'],entry['status'],entry['error_code']) for entry in moved['tool_trace']]==[('page.read','ok',None)]
            assert '"displayed_source":"B"' in tool_messages(changed_llm,1)[-1]['result']['content']
        finally:await controller.close()
    asyncio.run(run())


def test_page_read_section_identity_ignores_other_sections(tmp_path):
    from agent.sessions import page_snapshot_identity
    base={'editing':{'runtime_inputs':{'period':'E1'}},'results':{'displayed_source':'R','pending':[]}}
    same_results=evidence_sections({**base, 'editing':{'runtime_inputs':{'period':'E2'}}}, 'snap-'+'1'*32)
    changed_results=evidence_sections({**base, 'results':{'displayed_source':'R2','pending':[]}}, 'snap-'+'2'*32)
    original=evidence_sections(base, 'snap-'+'3'*32)
    # 只改 editing 不得改变 results 的读取身份；改 results 必须改变；随机 id/时间不参与身份。
    assert page_snapshot_identity(original,'results') == page_snapshot_identity(same_results,'results')
    assert page_snapshot_identity(original,'results') == page_snapshot_identity(evidence_sections(base, 'snap-'+'4'*32),'results')
    assert page_snapshot_identity(original,'results') != page_snapshot_identity(changed_results,'results')
    # 整份内容身份仍用于比较续跑快照：editing 变化也必须能区分。
    assert page_snapshot_identity(original) != page_snapshot_identity(same_results)
    assert page_snapshot_identity(None) is None


def test_page_read_blocked_by_results_content_ignores_editing_changes(tmp_path):
    store,page,session,service=setup(tmp_path)
    repeated={'name':'page.read','arguments':{'section':'results'}}
    async def run():
        controller=RunController(store)
        try:
            parent_llm=FixtureLLMClient([{'tool_calls':[repeated]} for _ in range(5)]+[{'content':'没有新证据，先暂停。'}])
            task,_=controller.submit(session['session_id'], request(page, message='parent', snapshot=evidence('A','snap-'+'f'*32)), parent_llm, service)
            paused=await controller.wait(task)
            assert paused['status']=='paused' and paused['stop_reason']=='no_progress',paused
            # 只有 editing 变化：results 读起来完全一样，不得解除阻断。
            unchanged_results=FixtureLLMClient([{'tool_calls':[repeated]},{'content':'按已有结果证据回答。'}])
            next_run,_=controller.submit(session['session_id'], request(page, message='continue-1', revision=1, text='继续', resume=paused['run_id'],
                snapshot=evidence_sections({'editing':{'runtime_inputs':{'period':'EDITED'}},'results':{'displayed_source':'A','pending':[]}}, 'snap-'+'a'*31+'3')), unchanged_results, service)
            blocked=await controller.wait(next_run)
            assert blocked['status']=='completed',blocked
            assert [(entry['tool'],entry['status'],entry['error_code']) for entry in blocked['tool_trace']]==[('page.read','error','AGENT_NO_PROGRESS')], '其他分区变化不得解除阻断'
            # results 分区真的变了才放行。
            changed_results=FixtureLLMClient([{'tool_calls':[repeated]},{'content':'新结果已读到。'}])
            last_run,_=controller.submit(session['session_id'], request(page, message='continue-2', revision=2, text='继续', resume=blocked['run_id'],
                snapshot=evidence_sections({'editing':{'runtime_inputs':{'period':'EDITED'}},'results':{'displayed_source':'B','pending':[]}}, 'snap-'+'a'*31+'4')), changed_results, service)
            moved=await controller.wait(last_run)
            assert moved['status']=='completed',moved
            assert [(entry['tool'],entry['status'],entry['error_code']) for entry in moved['tool_trace']]==[('page.read','ok',None)]
            assert '"displayed_source":"B"' in tool_messages(changed_results,1)[-1]['result']['content']
        finally:await controller.close()
    asyncio.run(run())


def test_commit_recovery_deduplicates_new_confirmations_and_restores_saved_state(tmp_path, monkeypatch):
    from agent import commit
    monkeypatch.setenv('CUSTOM_INDICATOR_DATA_DIR', str(tmp_path))
    store, page, session, service = setup(tmp_path)
    sid = session['session_id']
    with store.locked(sid) as state:
        store_draft(state, definition=DEFINITION, validation={'valid': True}, compile_token=None)
        store.write(state)

    def save(request_id, definition):
        prepared = commit.preview(store=store, session_id=sid, service=service,
            request=CommitPreviewRequest(draft_revision=1, definition=definition, page_context=page))
        command = CommitRequest(request_id=request_id, confirmation_id=prepared['confirmation_id'],
            definition_hash=prepared['definition_hash'], draft_revision=1, confirmed=True)
        result = commit.commit(store=store, session_id=sid, request=command, service=service)
        assert result['session_revision'] == prepared['session_revision'] + 1
        return result, command

    first, _ = save('first', DEFINITION)
    # The response was lost. Reopen the actual store, like a fresh page/process.
    store = AgentSessionStore(store.root)
    restored = store.public(sid)['saved_commit']
    assert restored['indicator_id'] == first['indicator_id']
    again, retry = save('new-id-after-reload', DEFINITION)
    assert again['replayed'] and again['indicator_id'] == first['indicator_id']
    assert len(service.create_calls) == 1
    assert commit.commit(store=store, session_id=sid, request=retry, service=service)['replayed']
    with pytest.raises(AgentError) as error:
        commit.commit(store=store, session_id=sid, request=retry.model_copy(update={'confirmed': False}), service=service)
    assert error.value.code == 'REVISION_CONFLICT'
    different, _ = save('different-definition', {**DEFINITION, 'name': '另一个指标'})
    assert different['indicator_id'] != first['indicator_id'] and len(service.create_calls) == 2
    # Old completed receipts are recoverable too, without an invented migration.
    with store.connection(write=True) as db:
        body = store.commit_receipt(db, sid, 'first'); body.pop('definition_hash')
        store.put_commit(db, sid, 'first', body)
    assert store.public(sid)['saved_commit']['indicator_id'] == first['indicator_id']
    other = store.create(page_context=page, scope='indicator_center')['session_id']
    assert store.public(other)['saved_commit'] is None


@pytest.mark.parametrize('when', ['before', 'during'])
@pytest.mark.parametrize('tool,operation', [('page.analyze', 'scenario'), ('page.analyze', 'diagnosis'), ('portfolios.eval', None)])
def test_portfolio_current_data_changes_are_fenced_without_invalidating_frozen_reads(tmp_path, monkeypatch, when, tool, operation):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from agent.research_runtime import data_generation
    from test_agent_research_pages import page, request_for, snapshot, RUN_ID
    monkeypatch.setenv('CUSTOM_INDICATOR_DATA_DIR', str(tmp_path))
    store = AgentSessionStore(tmp_path / 'sessions')
    context = page('holding-diagnosis')
    sid = store.create(page_context=context, scope='product_research')['session_id']
    service = FakeIndicatorService(tmp_path)
    before = data_generation(service)
    called = []
    def change_data():
        pq.write_table(pa.table({'close': [1.0, 1.1]}), tmp_path / 'etf_daily_df.parquet')
    def business(**kwargs):
        called.append(tool)
        if when == 'during': change_data()
        return {'run_id': RUN_ID, 'source_run_id': RUN_ID, 'locked_target_revision': 4,
                'observation_count': 20, 'summary': {'cumulative_return': .25}, 'results': []}
    service.evaluate_portfolio = business
    callbacks = {'run': lambda _: {'id': RUN_ID, 'immutable': True, 'target_revision': 4},
                 'scenario': lambda *a, **kw: business(**kw), 'diagnosis': lambda *a: business()}
    class Client(FixtureLLMClient):
        async def complete(self, **kwargs):
            if when == 'before' and not self.requests: change_data()
            return await super().complete(**kwargs)
    async def run():
        controller = RunController(store, page_services=callbacks)
        llm = Client([{'tool_calls': [{'name': tool, 'arguments': {'operation': operation} if operation else {}}]}, {'content': '已完成读取。'}])
        try:
            task, _ = controller.submit(sid, request(context, text='分析当前情景', snapshot=snapshot('holding-diagnosis', request_for('holding-diagnosis'))), llm, service)
            done = await controller.wait(task)
            assert data_generation(service) != before
            if operation == 'scenario':
                assert done['status'] == 'paused' and done['stop_reason'] == 'context_changed'
                assert len(called) == int(when == 'during')
                assert len(llm.requests) == 1, 'changed-generation output must never reach the next model call'
                assert not done['checkpoint']['task_state']['milestones']
                with store.connection() as db:
                    receipts = [json.loads(row[0]) for row in db.execute('SELECT body FROM tool_calls WHERE run_id=?', (task['run_id'],))]
                assert not any(item.get('applied') for item in receipts)
            else:
                assert done['status'] == 'completed', done.get('error')
                assert called == [tool]
        finally:
            await controller.close()
    asyncio.run(run())


def test_failed_save_cannot_be_repeated_with_a_new_confirmation(tmp_path, monkeypatch):
    from agent import commit
    from custom_indicators.errors import IndicatorDomainError
    monkeypatch.setenv('CUSTOM_INDICATOR_DATA_DIR', str(tmp_path))
    store, page, session, service = setup(tmp_path)
    sid = session['session_id']
    with store.locked(sid) as state:
        store_draft(state, definition=DEFINITION, validation={'valid': True}, compile_token=None)
        store.write(state)
    calls = []
    def fail_after_write(definition):
        calls.append(definition)
        raise IndicatorDomainError('DECORATION_FAILED', '保存后响应处理失败', status_code=500)
    service.create_indicator = fail_after_write
    for index in range(2):
        prepared = commit.preview(store=store, session_id=sid, service=service,
            request=CommitPreviewRequest(draft_revision=1, definition=DEFINITION, page_context=page))
        command = CommitRequest(request_id=f'new-attempt-{index}', confirmation_id=prepared['confirmation_id'],
            definition_hash=prepared['definition_hash'], draft_revision=1, confirmed=True)
        with pytest.raises((AgentError, IndicatorDomainError)) as error:
            commit.commit(store=store, session_id=sid, request=command, service=service)
        assert error.value.code == ('DECORATION_FAILED' if index == 0 else 'AGENT_COMMIT_UNCERTAIN')
    assert len(calls) == 1 and store.public(sid)['saved_commit'] is None

"""Sourced state, confirmed memory, compression invariants; offline stores only."""
import asyncio
import copy
import json

import pytest

from agent import data_policy, ledger, memory
from agent.context import compact_if_needed, context_messages
from agent.contracts import AgentError, MemoryRequest, MemoryRevokeRequest
from agent.harness import RunController
from agent.llm import FixtureLLMClient
from agent.progress import ProgressGuard
from agent.sessions import stable_hash, stable_json
from test_agent_runs import setup, request


def start_operation(store, run, *, operation_id='operation', call_id='call', model_step=1):
    if store.get_run(run['session_id'], run['run_id'])['status'] == 'queued':
        store.checkpoint(run, events=[{'type': 'run.started'}])
    receipt = {'model_step': model_step, 'call_id': call_id, 'operation_id': operation_id,
               'tool': 'metrics.lookup', 'status': 'started'}
    store.admit(run, tool_receipt=receipt)
    return receipt


@pytest.mark.parametrize('entry', ['admit', 'checkpoint'])
@pytest.mark.parametrize('candidate_status', ['running', 'stopping'])
def test_candidate_status_cannot_replace_explicit_control_transitions(tmp_path, entry, candidate_status):
    store, page, session, _ = setup(tmp_path)
    sid = session['session_id']
    run, _ = store.accept(sid, request(page), {'owner_instance': 'owner'})
    run['status'] = candidate_status
    assert getattr(store, entry)(run)
    assert store.get_run(sid, run['run_id'])['status'] == 'queued'
    assert store.checkpoint(run, events=[{'type': 'run.started', 'data': {'status': 'running'}}])
    run['status'] = 'stopping'
    assert getattr(store, entry)(run)
    current = store.get_run(sid, run['run_id'])
    assert current['status'] == 'running' and not current.get('cancel_requested')
    store.cancel(sid, run['run_id'])
    run['status'] = 'running'
    assert not getattr(store, entry)(run)
    assert store.get_run(sid, run['run_id'])['status'] == 'stopping'


@pytest.mark.parametrize('status', ['completed', 'unknown'])
def test_operation_completion_requires_a_started_receipt(tmp_path, status):
    store, page, session, _ = setup(tmp_path)
    sid = session['session_id']
    run, _ = store.accept(sid, request(page), {'owner_instance': 'owner'})
    store.checkpoint(run, events=[{'type': 'run.started'}])
    before = store.public(sid)
    receipt = {'model_step': 1, 'call_id': 'call', 'operation_id': 'operation', 'tool': 'metrics.lookup', 'status': status}
    with pytest.raises(AgentError) as error:
        (store.admit if status == 'unknown' else store.checkpoint)(run, tool_receipt=receipt)
    assert error.value.code == 'AGENT_RUN_TRANSITION_INVALID'
    assert store.public(sid) == before


@pytest.mark.parametrize('mutation', ['operation_id', 'tool', 'arguments', 'restart', 'rewrite_completed', 'blocked_after_start', 'skipped_after_start'])
def test_operation_identity_and_terminal_receipt_are_not_rewritable(tmp_path, mutation):
    store, page, session, _ = setup(tmp_path)
    sid = session['session_id']
    run, _ = store.accept(sid, request(page), {'owner_instance': 'owner'})
    store.checkpoint(run, events=[{'type': 'run.started'}])
    receipt = {'model_step': 1, 'call_id': 'call', 'operation_id': 'operation', 'tool': 'metrics.lookup', 'status': 'started'}
    assert store.admit(run, tool_receipt=receipt)
    candidate = {**receipt, 'status': 'completed'}
    entry = store.checkpoint
    if mutation == 'rewrite_completed':
        assert store.checkpoint(run, tool_receipt=candidate)
    elif mutation in {'operation_id', 'tool', 'arguments'}:
        candidate[mutation] = {'query': 'different'} if mutation == 'arguments' else 'different'
    elif mutation == 'restart':
        candidate['operation_id'] = 'replacement-operation'
        candidate['status'] = 'started'
        entry = store.admit
    else:
        candidate = {key: value for key, value in receipt.items() if key != 'operation_id'}
        candidate['status'] = 'blocked' if mutation == 'blocked_after_start' else 'not_executed'
    before = store.public(sid)
    with store.connection() as db:
        prior = db.execute('SELECT body FROM tool_calls WHERE run_id=?', (run['run_id'],)).fetchone()[0]
    with pytest.raises(AgentError) as error:
        entry(run, tool_receipt=candidate)
    assert error.value.code == 'AGENT_RUN_TRANSITION_INVALID'
    assert store.public(sid) == before
    with store.connection() as db:
        assert db.execute('SELECT body FROM tool_calls WHERE run_id=?', (run['run_id'],)).fetchone()[0] == prior


def test_cancelled_operation_completion_does_not_depend_on_its_released_fence(tmp_path):
    store, page, session, _ = setup(tmp_path)
    sid = session['session_id']
    run, _ = store.accept(sid, request(page), {'owner_instance': 'owner'})
    receipt = start_operation(store, run)
    store.cancel(sid, run['run_id'])
    assert not store.admit(run, tool_receipt={**receipt, 'status': 'unknown'})
    store.release_quarantine(sid, receipt['operation_id'])
    assert not store.checkpoint(run, tool_receipt={**receipt, 'status': 'completed'})
    with store.connection() as db:
        final = json.loads(db.execute('SELECT body FROM tool_calls WHERE run_id=?', (run['run_id'],)).fetchone()[0])
    assert final['status'] == 'discarded' and final['execution_status'] == 'completed' and not final['applied']
    assert not store.read(sid).get('execution_blocked_by')
    with pytest.raises(AgentError) as error:
        store.checkpoint(run, tool_receipt={**receipt, 'status': 'completed'})
    assert error.value.code == 'AGENT_RUN_TRANSITION_INVALID'


def test_rejected_start_cannot_become_an_unknown_or_completed_operation(tmp_path):
    store, page, session, _ = setup(tmp_path)
    sid = session['session_id']
    run, _ = store.accept(sid, request(page), {'owner_instance': 'owner'})
    store.checkpoint(run, events=[{'type': 'run.started'}])
    store.cancel(sid, run['run_id'])
    receipt = {'model_step': 1, 'call_id': 'call', 'operation_id': 'operation', 'tool': 'metrics.lookup', 'status': 'started'}
    assert not store.admit(run, tool_receipt=receipt)
    for status in ('unknown', 'completed'):
        with pytest.raises(AgentError) as error:
            (store.admit if status == 'unknown' else store.checkpoint)(run, tool_receipt={**receipt, 'status': status})
        assert error.value.code == 'AGENT_RUN_TRANSITION_INVALID'
    assert not store.read(sid).get('execution_blocked_by')


@pytest.mark.parametrize('event', ['data.admitted', 'tool.started'])
def test_queued_run_cannot_dispatch_before_explicit_start(tmp_path, event):
    store, page, session, _ = setup(tmp_path)
    sid = session['session_id']
    run, _ = store.accept(sid, request(page), {'owner_instance': 'owner'})
    receipt = {'model_step': 1, 'call_id': 'call', 'operation_id': 'operation', 'tool': 'metrics.lookup', 'status': 'started'}
    with pytest.raises(AgentError) as error:
        store.admit(run, events=[{'type': event}], tool_receipt=receipt if event == 'tool.started' else None)
    assert error.value.code == 'AGENT_RUN_TRANSITION_INVALID'
    assert store.get_run(sid, run['run_id'])['status'] == 'queued'


def test_start_is_one_content_transition_and_cancellation_cannot_restart(tmp_path):
    store, page, session, _ = setup(tmp_path)
    sid = session['session_id']
    run, _ = store.accept(sid, request(page), {'owner_instance': 'owner'})
    with pytest.raises(AgentError):
        store.admit(run, events=[{'type': 'run.started'}])
    with pytest.raises(AgentError):
        store.checkpoint(run, events=[{'type': 'run.started'}, {'type': 'run.started'}])
    assert store.checkpoint(run, events=[{'type': 'run.started', 'data': {'status': 'failed'}}])
    assert store.checkpoint(run, events=[{'type': 'run.phase', 'data': {'status': 'cancelled', 'phase': 'tool'}},
                                         {'type': 'run.recovering', 'data': {'status': 'stopping', 'phase': 'recovery'}}])
    lifecycle = [e for e in store.events(sid)['items'] if e['type'] in {'run.started', 'run.phase', 'run.recovering'}]
    assert all(e['data']['status'] == 'running' for e in lifecycle)
    assert lifecycle[-1]['data']['phase'] == 'recovery'
    with pytest.raises(AgentError):
        store.checkpoint(run, events=[{'type': 'run.started'}])
    store.cancel(sid, run['run_id'])
    before = store.public(sid)
    assert not store.checkpoint(run, events=[{'type': 'run.started'}])
    assert len([e for e in store.events(sid)['items'] if e['type'] == 'run.started']) == 1
    assert store.public(sid)['active_run']['status'] == before['active_run']['status'] == 'stopping'


@pytest.mark.parametrize('entry', ['admit', 'checkpoint'])
@pytest.mark.parametrize('cancelled', [False, True])
def test_execution_facts_cannot_publish_candidates_or_change_identity(tmp_path, entry, cancelled):
    store, page, session, _ = setup(tmp_path)
    sid = session['session_id']
    run, _ = store.accept(sid, request(page), {'owner_instance': 'owner'})
    run.update(status='running', checkpoint={'messages': [{'role': 'user', 'content': '已提交原话'}]},
               detector_state={'committed': True}, tool_trace=[{'tool': 'committed'}])
    assert store.checkpoint(run, events=[{'type': 'run.started'}])
    before = store.get_run(sid, run['run_id'])
    if cancelled:
        type(store)(store.root).cancel(sid, run['run_id'])
    run.update(checkpoint={'messages': [{'role': 'assistant', 'content': '尚未提交的候选'}]},
               detector_state={'candidate': True}, tool_trace=[{'tool': 'candidate'}])
    run['request']['text'] = '不得改写已接受消息'
    run['session_revision'] = 999
    run['usage']['prompt_tokens'] = 17
    assert getattr(store, entry)(run) is not cancelled
    current = store.get_run(sid, run['run_id'])
    assert current['request'] == before['request']
    assert current['session_revision'] == before['session_revision']
    assert current['usage']['prompt_tokens'] == 17
    for key in ('checkpoint', 'detector_state', 'tool_trace'):
        assert current[key] == (run[key] if entry == 'checkpoint' and not cancelled else before[key])


@pytest.mark.parametrize('entry', ['admit', 'checkpoint'])
def test_terminal_state_requires_finalization_entry(tmp_path, entry):
    store, page, session, _ = setup(tmp_path)
    run, _ = store.accept(session['session_id'], request(page), {'owner_instance': 'owner'})
    before = store.public(session['session_id'])
    run.update(status='completed', response={'reply': {'text': '绕过收尾'}})
    with pytest.raises(AgentError) as error:
        getattr(store, entry)(run)
    assert error.value.code == 'AGENT_RUN_TRANSITION_INVALID'
    assert store.public(session['session_id']) == before


def test_interruption_keeps_durable_checkpoint_instead_of_pending_candidate(tmp_path):
    store, page, session, _ = setup(tmp_path)
    run, _ = store.accept(session['session_id'], request(page), {'owner_instance': 'owner'})
    run['checkpoint'] = {'messages': [{'role': 'user', 'content': '已提交原话'}]}
    store.checkpoint(run)
    run['checkpoint']['messages'].append({'role': 'assistant', 'content': '未提交结果'})
    interrupted = store.interrupt(run)
    assert interrupted['status'] == 'interrupted'
    assert '未提交结果' not in stable_json(interrupted)
    assert 'response' not in interrupted


@pytest.mark.parametrize('ending', ['completed', 'paused', 'failed', 'cancelled'])
def test_finish_can_only_add_explicit_reply_to_durable_content(tmp_path, ending):
    store, page, session, _ = setup(tmp_path)
    sid = session['session_id']
    run, _ = store.accept(sid, request(page), {'owner_instance': 'owner'})
    run['checkpoint'] = {'messages': [{'role': 'user', 'content': '已提交原话'}]}
    store.checkpoint(run)
    run['checkpoint']['summary'] = '未提交的候选摘要'
    run['checkpoint']['messages'].append({'role': 'assistant', 'content': '未提交的候选正文'})
    run['detector_state'] = {'uncommitted': True}
    run['tool_trace'] = [{'tool': 'uncommitted'}]
    run['message_id'] = 'forged-local-id'
    if ending == 'cancelled':
        store.cancel(sid, run['run_id'])
    if ending == 'failed':
        run['error'] = {'code': 'AGENT_STORAGE_UNAVAILABLE', 'message': '写入失败'}
    done = store.finish(run, '明确的最终答复', status=ending, reasoning_content='本次模型协议内容')
    assert '未提交' not in stable_json(done)
    assert 'uncommitted' not in stable_json(done)
    assert done['response']['message_id'] == 'm'
    if ending in {'completed', 'paused'}:
        assert done['checkpoint']['messages'][-1]['reasoning_content'] == '本次模型协议内容'
    else:
        assert 'reasoning_content' not in done['checkpoint']['messages'][-1]


@pytest.mark.parametrize('private_payload', [False, True])
def test_control_entry_rejects_content_events(tmp_path, private_payload):
    store, page, session, _ = setup(tmp_path)
    run, _ = store.accept(session['session_id'], request(page), {'owner_instance': 'owner'})
    before = store.public(session['session_id'])
    event = {'type': 'context.compacted'} if not private_payload else {'type': 'data.admitted', '_model_message': {'content': '候选正文'}}
    with pytest.raises(AgentError) as error:
        store.admit(run, events=[event])
    assert error.value.code == 'AGENT_RUN_TRANSITION_INVALID'
    assert store.public(session['session_id']) == before


def test_discarded_tool_message_is_bound_to_latest_plan_with_reused_call_id(tmp_path):
    store, page, session, _ = setup(tmp_path)
    sid = session['session_id']
    run, _ = store.accept(sid, request(page), {'owner_instance': 'owner'})
    call = {'role': 'assistant', 'content': '', 'tool_calls': [
        {'id': 'reused', 'type': 'function', 'function': {'name': 'metrics.lookup', 'arguments': '{}'}}]}
    prior_result = {'role': 'tool', 'tool_call_id': 'reused', 'content': stable_json(
        data_policy.seal({'ok': True, 'result': {'items': [{'id': 'committed-prior'}]}}, 'metrics.lookup'))}
    run['checkpoint'] = {'messages': [copy.deepcopy(call), prior_result, copy.deepcopy(call)], 'model_step': 2}
    store.checkpoint(run, events=[{'type': 'run.started'}])
    receipt = start_operation(store, run, model_step=2, call_id='reused')
    store.cancel(sid, run['run_id'])
    assert not store.checkpoint(run, tool_receipt={**receipt, 'status': 'completed'})
    done = store.finish(run, '停止')
    tool_messages = [message for message in done['checkpoint']['messages'] if message['role'] == 'tool']
    assert tool_messages[0] == prior_result
    assert json.loads(tool_messages[1]['content'])['status'] == 'discarded'


@pytest.mark.parametrize('rollback', [False, True])
def test_final_checkpoint_shares_the_locked_session_transaction(tmp_path, rollback):
    store, page, session, _ = setup(tmp_path)
    sid = session['session_id']
    run, _ = store.accept(sid, request(page), {'owner_instance': 'owner'})
    observer = type(store)(store.root)
    before = observer.public(sid)

    def finish():
        with store.locked(sid):
            run.update(store.finish(run, '完整答复'))
            assert observer.public(sid) == before
            if rollback:
                raise ValueError('abort finalization')

    if rollback:
        with pytest.raises(ValueError, match='abort finalization'):
            finish()
        assert observer.public(sid) == before
    else:
        finish()
        state = observer.public(sid)
        assert state['active_run']['status'] == 'completed'
        assert state['messages'][-1]['text'] == '完整答复'
        assert observer.read(sid)['active_run_id'] is None
        assert observer.read(sid)['conversation']['messages'][-1]['content'] == '完整答复'


@pytest.mark.parametrize('terminal', ['completed', 'paused', 'cancelled', 'failed', 'interrupted'])
@pytest.mark.parametrize('late_write', ['active', 'terminal', 'artifact'])
def test_terminal_checkpoint_rejects_all_late_effects(tmp_path, terminal, late_write):
    store, page, session, service = setup(tmp_path)
    sid = session['session_id']
    run, _ = store.accept(sid, request(page), {'owner_instance': 'owner'})
    late = copy.deepcopy(run)
    run.update(store.finish(run, '已保存的答复', status=terminal, reason='original'))
    before, persisted = store.public(sid), store.get_run(sid, run['run_id'])
    late['status'] = 'running' if late_write == 'active' else 'failed'
    late['response'] = {'reply': {'text': '迟到覆盖'}}
    receipt = {'model_step': 1, 'call_id': 'late', 'status': 'completed'} if late_write == 'artifact' else None
    assert store.checkpoint(late, events=[{'type': 'assistant.message', 'text': '迟到事件'}],
                            patch={'draft': {'valid': True}, 'conversation': {'messages': ['迟到内容']}},
                            tool_receipt=receipt, new_facts=[('late', 'value')]) is False
    assert store.public(sid) == before
    assert store.get_run(sid, run['run_id']) == persisted
    with store.connection() as db:
        assert db.execute('SELECT count(*) FROM tool_calls').fetchone()[0] == 0
        assert db.execute('SELECT count(*) FROM progress_facts').fetchone()[0] == 0
    assert service.create_calls == []


def test_checkpoint_requires_current_session_run_ownership(tmp_path):
    store, page, session, _ = setup(tmp_path)
    sid = session['session_id']
    run, _ = store.accept(sid, request(page), {'owner_instance': 'owner'})
    with store.locked(sid) as state:
        state['active_run_id'] = 'another-run'
        store.write(state)
    before = store.read(sid)
    with pytest.raises(AgentError) as error:
        store.checkpoint(run, patch={'draft': {'valid': True}})
    assert error.value.code == 'AGENT_RUN_STALE'
    assert store.read(sid) == before


def test_running_checkpoint_cannot_return_to_queued(tmp_path):
    store, page, session, _ = setup(tmp_path)
    sid = session['session_id']
    run, _ = store.accept(sid, request(page), {'owner_instance': 'owner'})
    queued = copy.deepcopy(run)
    run['status'] = 'running'
    store.checkpoint(run, events=[{'type': 'run.started'}])
    before = store.public(sid)
    assert store.checkpoint(queued, patch={'draft': {'valid': True}}) is False
    assert store.public(sid) == before


def test_delayed_owner_recovery_cannot_clear_a_new_runs_quarantine(tmp_path):
    store, page, session, _ = setup(tmp_path)
    sid = session['session_id']
    old, _ = store.accept(sid, request(page), {'owner_instance': 'dead-owner'})
    candidate = store.unfinished_runs()[0]
    other_worker = type(store)(store.root)
    other_worker.recover(candidate)
    new, _ = other_worker.accept(sid, request(page, message='new', revision=1), {'owner_instance': 'new-owner'})
    receipt = start_operation(other_worker, new, operation_id='new-operation', call_id='new-call')
    other_worker.admit(new, tool_receipt={**receipt, 'status': 'unknown'})
    new.update(other_worker.finish(new, '超时', status='failed', reason='operation_timeout'))
    before, old_before = store.public(sid), store.get_run(sid, old['run_id'])
    store.recover(candidate)
    assert store.public(sid) == before
    assert store.get_run(sid, old['run_id']) == old_before
    assert store.read(sid)['execution_blocked_by'] == 'new-operation'


@pytest.mark.parametrize('change', ['checkpoint', 'cancel', 'release'])
def test_recovery_revalidates_the_observed_version_inside_its_transaction(tmp_path, change):
    store, page, session, _ = setup(tmp_path)
    sid = session['session_id']
    run, _ = store.accept(sid, request(page), {'owner_instance': 'owner'})
    if change == 'release':
        receipt = start_operation(store, run)
        store.admit(run, tool_receipt={**receipt, 'status': 'unknown'})
        run.update(store.finish(run, '超时', status='failed', reason='operation_timeout'))
    candidate = store.unfinished_runs()[0]
    other_worker = type(store)(store.root)
    if change == 'checkpoint':
        run['status'] = 'running'
        other_worker.checkpoint(run, events=[{'type': 'run.started'}], patch={'draft': {'valid': True, 'definition': {'name': '已完成草稿'}}})
    elif change == 'cancel':
        other_worker.cancel(sid, run['run_id'])
    else:
        other_worker.release_quarantine(sid, 'operation')
    before, current = store.public(sid), store.get_run(sid, run['run_id'])
    store.recover(candidate)
    assert store.public(sid) == before
    assert store.get_run(sid, run['run_id']) == current
    if change != 'release':
        store.recover(current)
        assert store.get_run(sid, run['run_id'])['status'] == 'interrupted'
        assert store.read(sid).get('draft') == before.get('draft')


@pytest.mark.parametrize('terminal', ['failed', 'cancelled', 'interrupted'])
def test_dead_owner_quarantine_cleanup_keeps_the_terminal_result(tmp_path, terminal):
    store, page, session, _ = setup(tmp_path)
    sid = session['session_id']
    run, _ = store.accept(sid, request(page), {'owner_instance': 'dead-owner'})
    receipt = start_operation(store, run)
    store.admit(run, tool_receipt={**receipt, 'status': 'unknown'})
    run.update(store.finish(run, '原答复', status=terminal, reason='original'))
    candidate = store.unfinished_runs()[0]
    store.recover(candidate)
    current = store.get_run(sid, run['run_id'])
    assert current['status'] == terminal and current['stop_reason'] == 'original'
    assert current['response'] == run['response'] and not current['execution_blocked_by']
    assert not store.read(sid).get('execution_blocked_by')
    assert current['run_revision'] > candidate['run_revision']
    before = store.public(sid)
    store.recover(candidate)
    assert store.public(sid) == before


def test_quarantine_release_advances_run_version_and_is_idempotent(tmp_path):
    store, page, session, _ = setup(tmp_path)
    sid = session['session_id']
    run, _ = store.accept(sid, request(page), {'owner_instance': 'owner'})
    receipt = start_operation(store, run)
    store.admit(run, tool_receipt={**receipt, 'status': 'unknown'})
    run.update(store.finish(run, '失败', status='failed'))
    before = store.get_run(sid, run['run_id'])
    store.release_quarantine(sid, 'wrong-operation')
    assert store.get_run(sid, run['run_id']) == before
    store.release_quarantine(sid, 'operation')
    current = store.get_run(sid, run['run_id'])
    assert current['run_revision'] == before['run_revision'] + 1
    assert current['status'] == 'failed' and current['execution_blocked_by'] is None
    snapshot = store.public(sid)
    store.release_quarantine(sid, 'operation')
    assert store.public(sid) == snapshot


@pytest.fixture(autouse=True)
def isolate(tmp_path, monkeypatch):
    monkeypatch.setenv('CUSTOM_INDICATOR_DATA_DIR', str(tmp_path))


def turn(store, page, session_id, service, *, message, text, replies, revision=None):
    async def run():
        controller = RunController(store); llm = FixtureLLMClient(replies)
        try:
            run, _ = controller.submit(session_id, request(page, message=message, text=text,
                revision=store.read(session_id)['session_revision'] if revision is None else revision), llm, service)
            done = await controller.wait(run)
            assert done['status'] == 'completed', done.get('error')
            return done, llm
        finally:
            await controller.close()
    return asyncio.run(run())


def test_ledger_sources_and_proposals_cannot_certify_work(tmp_path):
    store, page, session, service = setup(tmp_path)
    sid = session['session_id']
    done, _ = turn(store, page, sid, service, message='goal', text='请解释波动率，禁止未来数据；窗口20日。', replies=[
        {'tool_calls': [{'name': 'task.plan', 'arguments': {'source_message_id': 'goal',
                            'capabilities': ['explain'], 'questions': ['需要哪种年化口径？']}}]},
        {'content': '请确认年化口径。'}])
    state = done['checkpoint']['task_state']
    assert state['sources'][0]['text'] == '请解释波动率，禁止未来数据；窗口20日。'
    assert state['goal_status'] == 'open' and state['milestones'] == []
    assert state['plans'][0]['status'] == 'proposed'
    assert state['pending_questions'] == ['需要哪种年化口径？']
    rebuilt, _ = ledger.rebuild(type(store)(store.root), sid)
    assert rebuilt['sources'][0]['id'] == 'goal' and rebuilt['plans'][0]['source_message_id'] == 'goal'
    done, llm = turn(store, page, sid, service, message='amendment', text='改为30日，仍然禁止未来数据。', replies=[{'content': '按30日继续。'}])
    assert done['checkpoint']['task_state']['latest_source_id'] == 'amendment'
    assert [item['id'] for item in done['checkpoint']['task_state']['sources']] == ['goal', 'amendment']
    assert '30日' in stable_json(llm.requests) and '禁止未来数据' in stable_json(llm.requests)
    assert service.create_calls == []


def proposal_turn(store, page, sid, service, message, quote):
    done, _ = turn(store, page, sid, service, message=message, text=quote, replies=[
        {'tool_calls': [{'name': 'memory.propose', 'arguments': {
            'source_message_id': message, 'quote': quote, 'key': 'reply.language'}}]}, {'content': '请确认是否记住。'}])
    return next(item for item in store.read(sid)['memory_proposals'] if item['source_message_id'] == message)


def test_memory_confirmation_scope_replace_revoke_and_new_session_isolation(tmp_path):
    store, page, session, service = setup(tmp_path); sid = session['session_id']
    p = proposal_turn(store, page, sid, service, 'remember', '以后请用中文回答。')
    assert memory.recall(store=store, session_id=sid) == []
    accepted = memory.resolve(store=store, session_id=sid, request=MemoryRequest(proposal_id=p['proposal_id'], decision='accept'))
    replay = memory.resolve(store=store, session_id=sid, request=MemoryRequest(proposal_id=p['proposal_id'], decision='accept'))
    assert replay['replayed'] and replay['memory_id'] == accepted['memory_id']
    fresh = store.create(page_context=page, scope='indicator_center')['session_id']
    done, llm = turn(store, page, fresh, service, message='new', text='本轮请用英文回答。', replies=[{'content': 'In English.'}])
    sources = done['checkpoint']['task_state']['sources']
    assert [item['id'] for item in sources] == ['new']
    preference = done['checkpoint']['memory_sources'][0]
    assert preference['source_session_id'] == sid and preference['text'] == '以后请用中文回答。'
    assert preference['source_message_id'] == 'remember'
    assert done['checkpoint']['task_state']['precedence'].startswith('current_user_over_preferences')
    assert sources[0]['text'] == '本轮请用英文回答。'
    other = store.create(page_context=page, scope='product_research')['session_id']
    assert memory.recall(store=store, session_id=other) == []
    p2 = proposal_turn(store, page, sid, service, 'change', '以后请用英文回答。')
    with pytest.raises(AgentError, match='替换'):
        memory.resolve(store=store, session_id=sid, request=MemoryRequest(proposal_id=p2['proposal_id'], decision='accept'))
    replaced = memory.resolve(store=store, session_id=sid, request=MemoryRequest(proposal_id=p2['proposal_id'], decision='accept',
        replace_memory_id=accepted['memory_id'], expected_version=1))
    recalled = memory.recall(store=store, session_id=fresh)
    assert len(recalled) == 1 and recalled[0]['text'] == '以后请用英文回答。'
    revoke = MemoryRevokeRequest(request_id='revoke-one', memory_id=replaced['memory_id'], expected_version=1)
    assert memory.revoke(store=store, session_id=fresh, request=revoke)['status'] == 'revoked'
    assert memory.revoke(store=store, session_id=fresh, request=revoke)['replayed']
    assert memory.recall(store=store, session_id=fresh) == [] and service.create_calls == []


@pytest.mark.parametrize('scope', ['indicator_center', 'product_research'])
@pytest.mark.parametrize('legacy', [False, True])
def test_indicator_memory_is_listed_recalled_replaced_and_revoked(tmp_path, monkeypatch, scope, legacy):
    from agent import commit, routes
    from agent.contracts import CommitPreviewRequest, PageContext
    from agent.sessions import store_draft
    from test_agent_api import DEFINITION, single_context

    store, page, session, service = setup(tmp_path)
    if scope == 'product_research':
        page = PageContext.model_validate(single_context())
        session = store.create(page_context=page, scope=scope)
    sid = session['session_id']
    monkeypatch.setattr(routes, 'session_store', lambda: store)
    with store.locked(sid) as state:
        store_draft(state, definition=DEFINITION, validation={'valid': True}, compile_token=None)
        store.write(state)

    def proposal():
        prepared = commit.preview(store=store, session_id=sid, service=service,
            request=CommitPreviewRequest(draft_revision=1, definition=DEFINITION, page_context=page))
        if legacy:
            with store.locked(sid) as state:
                state['memory_proposals'][-1].pop('key', None)
                state['memory_proposals'][-1].pop('object_id', None)
                store.write(state)
        public = store.public(sid)['memory_proposals'][-1]
        assert public['key'] == 'indicator_definition:' + prepared['definition_hash']
        assert public['object_id'] == 'scope'
        return prepared

    first = proposal()
    accepted = memory.resolve(store=store, session_id=sid,
        request=MemoryRequest(proposal_id=first['memory_proposal_id'], decision='accept'))
    with store.connection(write=True) as db:
        record = json.loads(db.execute('SELECT body FROM memory_records WHERE id=?', (accepted['memory_id'],)).fetchone()[0])
        if legacy:
            record = data_policy.seal({**record, 'object_id': first['definition_hash']}, 'memory.record')
            db.execute('UPDATE memory_records SET body=? WHERE id=?', (stable_json(record), record['memory_id']))
        persisted = db.execute('SELECT body FROM memory_records WHERE id=?', (accepted['memory_id'],)).fetchone()[0]
    visible = routes.agent_memory_sources(sid)['items']
    assert len(visible) == 1 and visible[0]['memory_id'] == accepted['memory_id']
    assert visible[0]['object_id'] == 'scope' and visible[0]['key'] == 'indicator_definition:' + first['definition_hash']
    with store.connection() as db:
        assert db.execute('SELECT body FROM memory_records WHERE id=?', (accepted['memory_id'],)).fetchone()[0] == persisted

    fresh = store.create(page_context=page, scope=scope)['session_id']
    done, llm = turn(store, page, fresh, service, message='recall-definition', text='请核对已确认记忆。',
                     replies=[{'content': '已核对记忆。'}])
    assert done['checkpoint']['memory_sources'][0]['memory_id'] == accepted['memory_id']
    assert record['text'] in stable_json(llm.requests)
    other = store.create(page_context=page, scope='product_research' if scope == 'indicator_center' else 'indicator_center')['session_id']
    assert memory.recall(store=store, session_id=other) == []

    second = proposal()
    with pytest.raises(AgentError) as error:
        memory.resolve(store=store, session_id=sid,
            request=MemoryRequest(proposal_id=second['memory_proposal_id'], decision='accept'))
    assert error.value.code == 'AGENT_MEMORY_CONFLICT'
    replacement = memory.resolve(store=store, session_id=sid,
        request=MemoryRequest(proposal_id=second['memory_proposal_id'], decision='accept',
                              replace_memory_id=visible[0]['memory_id'], expected_version=visible[0]['version']))
    current = routes.agent_memory_sources(sid)['items']
    assert len(current) == 1 and current[0]['memory_id'] == replacement['memory_id']
    with store.connection() as db:
        prior = json.loads(db.execute('SELECT body FROM memory_records WHERE id=?', (accepted['memory_id'],)).fetchone()[0])
        assert prior['status'] == 'replaced' and data_policy.verify(prior, 'memory.record')
    action = MemoryRevokeRequest(request_id='revoke-definition', memory_id=current[0]['memory_id'], expected_version=current[0]['version'])
    assert memory.revoke(store=store, session_id=sid, request=action)['status'] == 'revoked'
    assert memory.revoke(store=store, session_id=sid, request=action)['replayed']
    assert routes.agent_memory_sources(sid)['items'] == [] and memory.recall(store=store, session_id=fresh) == []
    assert service.create_calls == []


def test_memory_bad_source_raw_corrupt_and_pending_are_never_recalled(tmp_path):
    store, page, session, service = setup(tmp_path); sid = session['session_id']
    done, _ = turn(store, page, sid, service, message='goal', text='解释收益率。', replies=[{'content': '收益率是比值变化。'}])
    state = store.read(sid)
    with pytest.raises(AgentError):
        memory.propose(store=store, session_id=sid, state=state, source_message_id='missing', quote='任意偏好', key='test')
    with pytest.raises(AgentError):
        memory.propose(store=store, session_id=sid, state=state, source_message_id='goal', quote='[{"x":987654.321}]', key='test')
    with store.connection(write=True) as db:
        db.execute('INSERT INTO memory_records VALUES (?,?)', ('broken', 'not JSON'))
        db.execute('INSERT INTO memory_records VALUES (?,?)', ('forged', json.dumps({'status': 'accepted', 'text': '秘密'})))
    (tmp_path/'agent_memory').mkdir(exist_ok=True)
    (tmp_path/'agent_memory'/'indicator-center.md').write_text('旧的无来源偏好')
    assert memory.recall(store=store, session_id=sid) == []


def test_old_definition_memory_projection_preserves_admission_and_object_boundaries(tmp_path):
    store, page, session, _ = setup(tmp_path)
    sid = session['session_id']
    digest = 'a' * 64
    record = {'memory_id': 'memory-' + '1' * 32, 'version': 1, 'status': 'accepted',
              'scope': session['scope'], 'object_id': digest, 'key': 'indicator_definition:' + digest,
              'text': '记住指标草稿的定义与上下文版本。', 'source_session_id': sid,
              'source_message_id': None, 'source_verified': True, 'accepted_at': '2026-09-22'}
    variants = [record,
        {**record, 'source_message_id': 'ordinary-product-preference'},
        {**record, 'key': 'reply.style'},
        {**record, 'object_id': 'b' * 64},
        {**record, 'scope': 'product_research'},
        {**record, 'text': '[{"x":987654.321}]'},
        {**record, 'source_verified': False},
        {**record, 'status': 'revoked'},
        {**record, 'object_id': 'short', 'key': 'indicator_definition:short'}]
    with store.connection(write=True) as db:
        for index, variant in enumerate(variants, 1):
            value = data_policy.seal({**variant, 'memory_id': 'memory-' + f'{index:032x}'}, 'memory.record')
            db.execute('INSERT INTO memory_records VALUES (?,?)', (value['memory_id'], stable_json(value)))
        forged = data_policy.seal({**record, 'memory_id': 'memory-' + 'f' * 32}, 'memory.record')
        forged['text'] = '改写了签名正文'
        db.execute('INSERT INTO memory_records VALUES (?,?)', (forged['memory_id'], stable_json(forged)))
    recalled = memory.recall(store=store, session_id=sid)
    assert len(recalled) == 1 and recalled[0]['memory_id'] == 'memory-' + f'{1:032x}'
    assert recalled[0]['object_id'] == 'scope'
    action = MemoryRevokeRequest(request_id='revoke-old-definition', memory_id=recalled[0]['memory_id'], expected_version=1)
    assert memory.revoke(store=store, session_id=sid, request=action)['status'] == 'revoked'
    assert memory.recall(store=store, session_id=sid) == []


def test_three_compactions_preserve_exact_state_negative_questions_and_refs(tmp_path, monkeypatch):
    monkeypatch.setenv('AGENT_CONTEXT_CHAR_LIMIT', '18000')
    source = {'schema_version': 1, 'policy_version': data_policy.POLICY_VERSION, 'session_id': 's',
              'revision': 'state1', 'sources': [{'id': 'goal', 'text': '只用复权净值，禁止未来数据。'},
                                                {'id': 'amend', 'text': '窗口改为30日，仍禁止杠杆。'}],
              'pending_questions': ['年化口径待确认'], 'rejected_strategies': [{'ref': 'op-'+'a'*32}],
              'evidence_refs': ['op-'+'a'*32]}
    state = data_policy.seal(source, 'task.state')
    checkpoint = {'task_state': state, 'valid_evidence_refs': ['op-'+'a'*32], 'session_id': 's',
                  'llm_context_key': 'fixed-model', 'messages': []}
    llm = FixtureLLMClient([{'content': '未决年化口径；仍禁止未来数据。'}]*3); llm.context_window_tokens = 32768
    for cycle in range(3):
        for i in range(5):
            call_id = f'{cycle}-{i}'
            checkpoint['messages'] += [{'role': 'assistant', 'content': '', 'reasoning_content': 'r'*12000,
                'tool_calls': [{'id': call_id, 'type': 'function', 'function': {'name': 'metrics.lookup', 'arguments': '{}'}}]},
                {'role': 'tool', 'tool_call_id': call_id, 'content': stable_json(data_policy.seal(
                    {'ok': True, 'context_ref': 'op-'+'a'*32, 'result': {'items': [{'id': 'returns', 'description': '文字'*200}]}}, 'metrics.lookup'))}]
        asyncio.run(compact_if_needed(checkpoint, system='研究', llm=llm, on_compacted=lambda _: None, force=True))
        assert checkpoint['task_state'] == state
        assert '禁止未来数据' in stable_json(context_messages(checkpoint))
        assert '年化口径待确认' in stable_json(context_messages(checkpoint))
    assert len(checkpoint['segments']) >= 3
    for segment in checkpoint['segments']:
        assert data_policy.verify(segment, 'context.segment') and segment['state_revision'] == 'state1'
        assert segment['source_hash'] and segment['source_message_range'] == ['goal', 'amend']
        assert segment['token_after'] < segment['token_before']


def test_irrelevant_novelty_does_not_reset_progress_but_long_requested_plan_does():
    task = {'sources': [{'text': '解释波动率'}], 'read_evidence': [{'tool': 'metrics.lookup', 'kind': 'indicators'}]}
    guard = ProgressGuard(relevant=lambda tool, args: ledger.relevant(tool, args, task))
    for i in range(10):
        result = guard.record('metrics.lookup', {'query': f'unrelated{i}', 'kind': 'indicators'},
                             {'ok': True, 'result': {'items': [{'id': f'unrelated{i}'}]}})
    assert result['pause']
    task['sources'] = [{'text': '依次查询 ' + ' '.join(f'item{i}' for i in range(120))}]
    guard = ProgressGuard(relevant=lambda tool, args: ledger.relevant(tool, args, task))
    for i in range(120):
        result = guard.record('metrics.lookup', {'query': f'item{i}', 'kind': 'indicators'},
                             {'ok': True, 'result': {'items': [{'id': f'item{i}'}]}})
        assert result['progress'] and not result['pause']


def test_bad_reference_preserves_original_and_masking_avoids_summary_model(monkeypatch):
    monkeypatch.setenv('AGENT_CONTEXT_CHAR_LIMIT', '40000')
    state = data_policy.seal({'revision': 'unchanged', 'sources': [{'id': 'm', 'text': '禁止杠杆'}],
                              'evidence_refs': []}, 'task.state')
    checkpoint = {'task_state': state, 'valid_evidence_refs': [], 'messages': [
        {'role': 'user', 'content': '禁止杠杆'}, {'role': 'assistant', 'content': '已有分析'*1800, 'reasoning_content': 'r'*60000}]}
    class Cancel(FixtureLLMClient):
        async def complete(self, **kw): raise AssertionError('canonical checkpoint must not call a summary model')
    llm = Cancel([]); llm.context_window_tokens = 65536
    asyncio.run(compact_if_needed(checkpoint, system='研究', llm=llm, on_compacted=lambda _: None, force=True))
    assert checkpoint['task_state'] == state and checkpoint['segments'][-1]['method'] == 'evidence_checkpoint'
    checkpoint['messages'] += [{'role': 'assistant', 'content': '', 'reasoning_content': 'r'*60000,
        'tool_calls': [{'id': 'bad', 'type': 'function', 'function': {'name': 'metrics.lookup', 'arguments': '{}'}}]},
        {'role': 'tool', 'tool_call_id': 'bad', 'content': stable_json(data_policy.seal(
            {'ok': True, 'context_ref': 'op-'+'f'*32}, 'metrics.lookup'))}]
    before = copy.deepcopy(checkpoint)
    with pytest.raises(AgentError):
        asyncio.run(compact_if_needed(checkpoint, system='研究', llm=llm, on_compacted=lambda _: None, force=True))
    assert checkpoint == before


def test_stopped_edited_turn_removes_plan_memory_sources_and_keeps_prior_state(tmp_path):
    store, page, session, service = setup(tmp_path); sid = session['session_id']
    turn(store, page, sid, service, message='prior', text='保留原先禁止杠杆条件', replies=[{'content': '已记录'}])
    async def run():
        from agent.llm import LLMReply
        class Blocking(FixtureLLMClient):
            def __init__(self):
                super().__init__([{'tool_calls': [{'name': 'memory.propose', 'arguments': {
                    'source_message_id': 'stopped', 'quote': '以后用中文回答', 'key': 'reply.language'}}]}])
                self.entered = asyncio.Event()
            async def complete(self, **kwargs):
                if self._replies:
                    return await super().complete(**kwargs)
                self.entered.set(); await asyncio.Event().wait()
                return LLMReply(content='unreachable')
        controller = RunController(store); llm = Blocking()
        try:
            task, _ = controller.submit(sid, request(page, message='stopped', revision=1, text='以后用中文回答'), llm, service)
            await asyncio.wait_for(llm.entered.wait(), 5)
            controller.cancel(sid, task['run_id']); await controller.wait(task)
            assert any(p['source_message_id'] == 'stopped' for p in store.read(sid)['memory_proposals'])
            replacement = FixtureLLMClient([{'content': '只解释，不记忆'}])
            task, _ = controller.submit(sid, request(page, message='edited', revision=2, text='只解释，不记忆', edit='stopped'), replacement, service)
            done = await controller.wait(task)
            assert done['status'] == 'completed', done.get('error')
            assert [s['id'] for s in done['checkpoint']['task_state']['sources']] == ['prior', 'edited']
            assert store.read(sid).get('memory_proposals') == []
            assert '以后用中文回答' not in stable_json(replacement.requests)
        finally:
            await controller.close()
    asyncio.run(run())


@pytest.mark.parametrize('legacy', [False, True])
@pytest.mark.parametrize('decision', ['accept', 'reject', 'replace', 'revoke'])
def test_repeated_edits_preserve_independent_memory_decisions(tmp_path, legacy, decision):
    store, page, session, service = setup(tmp_path)
    sid = session['session_id']
    prior = proposal_turn(store, page, sid, service, 'prior-memory', '以后回答先给结论。')
    old = None
    if decision in {'replace', 'revoke'}:
        old = memory.resolve(store=store, session_id=sid,
            request=MemoryRequest(proposal_id=prior['proposal_id'], decision='accept'))
    if decision == 'replace':
        prior = proposal_turn(store, page, sid, service, 'replace-memory', '以后回答先说明依据。')

    async def run():
        class Blocking(FixtureLLMClient):
            def __init__(self, message):
                self.entered = asyncio.Event()
                super().__init__([{'tool_calls': [{'name': 'memory.propose', 'arguments': {
                    'source_message_id': message, 'quote': '临时未确认偏好', 'key': 'reply.temporary'}}]}])
            async def complete(self, **kwargs):
                if self._replies:
                    return await super().complete(**kwargs)
                self.entered.set()
                await asyncio.Event().wait()

        controller = RunController(store)
        try:
            previous = None
            for index in range(2):
                message = f'stopped-{index}'
                model = Blocking(message)
                task, _ = controller.submit(sid, request(page, message=message, text='临时未确认偏好',
                    edit=previous, revision=store.read(sid)['session_revision']), model, service)
                await asyncio.wait_for(model.entered.wait(), 5)
                controller.cancel(sid, task['run_id'])
                assert (await controller.wait(task))['status'] == 'cancelled'
                if legacy:
                    with store.connection(write=True) as db:
                        stored = store._read_run(db, sid, task['run_id'])
                        stored.pop('base_state', None)
                        store._write_run(db, stored)
                if index == 0:
                    if decision == 'revoke':
                        memory.revoke(store=store, session_id=sid, request=MemoryRevokeRequest(
                            request_id='independent-revoke', memory_id=old['memory_id'], expected_version=1))
                    else:
                        memory.resolve(store=store, session_id=sid, request=MemoryRequest(
                            proposal_id=prior['proposal_id'], decision='reject' if decision == 'reject' else 'accept',
                            replace_memory_id=old['memory_id'] if decision == 'replace' else None,
                            expected_version=1 if decision == 'replace' else None))
                    with store.connection() as db:
                        records = list(db.execute('SELECT body FROM memory_records ORDER BY id'))
                        records = [row[0] for row in records]
                expected = 'rejected' if decision == 'reject' else 'accepted'
                assert next(p for p in store.public(sid)['memory_proposals'] if p['proposal_id'] == prior['proposal_id'])['status'] == expected
                previous = message
            final, _ = controller.submit(sid, request(page, message='final-edit', text='只解释，不新增偏好。',
                edit=previous, revision=store.read(sid)['session_revision']), FixtureLLMClient([{'content': '已解释。'}]), service)
            done = await controller.wait(final)
            assert done['status'] == 'completed'
            visible = store.public(sid)['memory_proposals']
            assert all(p['source_message_id'] not in {'stopped-0', 'stopped-1'} for p in visible)
            assert next(p for p in visible if p['proposal_id'] == prior['proposal_id'])['status'] == expected
            with store.connection() as db:
                assert [row[0] for row in db.execute('SELECT body FROM memory_records ORDER BY id')] == records
            recalled = memory.recall(store=store, session_id=sid)
            assert len(recalled) == (1 if decision in {'accept', 'replace'} else 0)
            assert done['checkpoint']['memory_sources'] == recalled
            replay = memory.resolve(store=store, session_id=sid, request=MemoryRequest(
                proposal_id=prior['proposal_id'], decision='reject' if decision == 'reject' else 'accept',
                replace_memory_id=old['memory_id'] if decision == 'replace' else None,
                expected_version=1 if decision == 'replace' else None))
            assert replay['replayed'] and memory.recall(store=store, session_id=sid) == recalled
            assert service.create_calls == []
        finally:
            await controller.close()
    asyncio.run(run())


@pytest.mark.parametrize('decision', ['accept', 'reject'])
def test_memory_receipts_reconcile_old_pending_state_and_survive_proposal_removal(tmp_path, decision):
    store, page, session, service = setup(tmp_path)
    sid = session['session_id']
    proposal = proposal_turn(store, page, sid, service, 'prior-memory', '以后回答先给结论。')
    command = MemoryRequest(proposal_id=proposal['proposal_id'], decision=decision)
    receipt = memory.resolve(store=store, session_id=sid, request=command)
    # Reproduce data persisted by the old edit implementation, bypassing today's writer.
    with store.connection(write=True) as db:
        raw = json.loads(db.execute('SELECT body FROM sessions WHERE id=?', (sid,)).fetchone()[0])
        raw['memory_proposals'][0]['status'] = 'pending'
        db.execute('UPDATE sessions SET body=? WHERE id=?', (stable_json(raw), sid))
    expected = 'accepted' if decision == 'accept' else 'rejected'
    assert store.read(sid)['memory_proposals'][0]['status'] == expected
    assert store.public(sid)['memory_proposals'][0]['status'] == expected
    with store.locked(sid) as state:
        state['memory_proposals'] = []
        store.write(state)
    replay = memory.resolve(store=store, session_id=sid, request=command)
    assert replay == {**receipt, 'replayed': True}
    with pytest.raises(AgentError) as error:
        memory.resolve(store=store, session_id=sid, request=command.model_copy(update={
            'decision': 'reject' if decision == 'accept' else 'accept'}))
    assert error.value.code == 'AGENT_MEMORY_DECISION_CONFLICT'


@pytest.mark.parametrize('legacy', [False, True])
def test_edit_removes_unconfirmed_turn_outputs_but_keeps_human_memory_actions(tmp_path, legacy):
    from agent import commit
    from agent.contracts import CommitPreviewRequest
    from agent.sessions import store_draft
    from test_agent_api import DEFINITION

    store, page, session, service = setup(tmp_path)
    sid = session['session_id']
    with store.locked(sid) as state:
        store_draft(state, definition=DEFINITION, validation={'valid': True}, compile_token=None)
        store.write(state)

    async def run():
        class Blocking(FixtureLLMClient):
            def __init__(self):
                self.entered = asyncio.Event()
                super().__init__([{'tool_calls': [
                    {'name': 'memory.propose', 'arguments': {'source_message_id': 'stopped',
                        'quote': quote, 'key': key}} for quote, key in [
                            ('以后简洁回答', 'reply.brief'), ('保留详细依据', 'reply.detail'), ('其他偏好', 'reply.other')]
                ]}])
            async def complete(self, **kwargs):
                if self._replies:
                    return await super().complete(**kwargs)
                self.entered.set()
                await asyncio.Event().wait()
        controller = RunController(store)
        model = Blocking()
        try:
            task, _ = controller.submit(sid, request(page, message='stopped',
                text='以后简洁回答，同时保留详细依据，暂不记其他偏好。'), model, service)
            await asyncio.wait_for(model.entered.wait(), 5)
            controller.cancel(sid, task['run_id'])
            await controller.wait(task)
            if legacy:
                with store.connection(write=True) as db:
                    stored = store._read_run(db, sid, task['run_id'])
                    stored.pop('base_state', None)
                    store._write_run(db, stored)
            proposed = store.read(sid)['memory_proposals']
            commands = [MemoryRequest(proposal_id=proposed[index]['proposal_id'], decision=decision)
                        for index, decision in enumerate(('accept', 'reject'))]
            decisions = [memory.resolve(store=store, session_id=sid, request=command) for command in commands]
            manual = commit.preview(store=store, session_id=sid, service=service,
                request=CommitPreviewRequest(draft_revision=1, definition=DEFINITION, page_context=page))
            replacement = FixtureLLMClient([{'content': '只解释。'}])
            edited, _ = controller.submit(sid, request(page, message='edited', text='只解释。', edit='stopped',
                revision=store.read(sid)['session_revision']), replacement, service)
            done = await controller.wait(edited)
            assert done['status'] == 'completed'
            visible = {item['proposal_id']: item['status'] for item in store.public(sid)['memory_proposals']}
            assert visible == {proposed[0]['proposal_id']: 'accepted', proposed[1]['proposal_id']: 'rejected',
                               manual['memory_proposal_id']: 'pending'}
            assert [item['id'] for item in done['checkpoint']['task_state']['sources']] == ['edited']
            assert [item['text'] for item in memory.recall(store=store, session_id=sid)] == ['以后简洁回答']
            for command, decision in zip(commands, decisions):
                assert memory.resolve(store=store, session_id=sid, request=command) == {**decision, 'replayed': True}
            assert service.create_calls == []
        finally:
            await controller.close()
    asyncio.run(run())


@pytest.mark.parametrize('legacy', [False, True])
@pytest.mark.parametrize('tool', ['memory.propose', 'task.plan'])
def test_edit_discards_new_tool_proposals_quoting_prior_messages(tmp_path, legacy, tool):
    store, page, session, service = setup(tmp_path)
    sid = session['session_id']
    if tool == 'memory.propose':
        prior_args = {'source_message_id': 'prior', 'quote': '以后简洁回答', 'key': 'reply.before'}
        new_args = {**prior_args, 'key': 'reply.discard'}
        field, identity = 'memory_proposals', 'proposal_id'
    else:
        prior_args = {'source_message_id': 'prior', 'questions': ['此前保留的问题']}
        new_args = {**prior_args, 'questions': ['本轮应撤销的问题']}
        field, identity = 'task_plans', 'id'
    turn(store, page, sid, service, message='prior', text='以后简洁回答', replies=[
        {'tool_calls': [{'name': tool, 'arguments': prior_args}]}, {'content': '已记录提案。'}])
    original = store.read(sid)[field][0][identity]

    async def run():
        class Blocking(FixtureLLMClient):
            def __init__(self):
                self.entered = asyncio.Event()
                super().__init__([{'tool_calls': [{'name': tool, 'arguments': args} for args in (prior_args, new_args)]}])
            async def complete(self, **kwargs):
                if self._replies:
                    return await super().complete(**kwargs)
                self.entered.set()
                await asyncio.Event().wait()
        controller = RunController(store)
        model = Blocking()
        try:
            task, _ = controller.submit(sid, request(page, message='stopped', text='临时尝试另一个方案。',
                revision=store.read(sid)['session_revision']), model, service)
            await asyncio.wait_for(model.entered.wait(), 5)
            controller.cancel(sid, task['run_id'])
            await controller.wait(task)
            assert len(store.read(sid)[field]) == 2
            if legacy:
                with store.connection(write=True) as db:
                    stored = store._read_run(db, sid, task['run_id'])
                    stored.pop('base_state', None)
                    store._write_run(db, stored)
            task, _ = controller.submit(sid, request(page, message='edited', text='只保留原提案。', edit='stopped',
                revision=store.read(sid)['session_revision']), FixtureLLMClient([{'content': '已保留。'}]), service)
            assert (await controller.wait(task))['status'] == 'completed'
            assert [item[identity] for item in store.read(sid)[field]] == [original]
            assert service.create_calls == []
        finally:
            await controller.close()
    asyncio.run(run())


def test_null_frozen_date_resolves_existing_default_and_future_is_rejected(tmp_path):
    from agent.tools import execute_tool
    from agent.contracts import PageContext
    from pit.context import ResearchContext, set_view_override, reset_view_override
    from test_agent_admission import APPROVED_DEFINITION
    store, page, session, service = setup(tmp_path)
    page = page.model_copy(deep=True); page.calculation.as_of = '2019-12-31'
    snapshot = {'sections': {'results': {'frozen_request': {'definition': APPROVED_DEFINITION,
        'targets': [{'kind': 'etf', 'product_id': '510300.SH'}], 'period': '1Y', 'as_of': None, 'parameters': {}}}}}
    token = set_view_override(ResearchContext(as_of='2019-12-31'))
    try:
        result = execute_tool('page.recompute', {}, session={'scope': 'indicator_center'}, page_context=page,
                              service=service, page_snapshot=snapshot)
        assert service.evaluate_calls[-1]['as_of'] == '2019-12-31'
        assert result['frozen_inputs']['as_of'] is None
        assert result['frozen_inputs']['effective_as_of'] == '2019-12-31'
        snapshot['sections']['results']['frozen_request']['as_of'] = '2030-01-01'
        with pytest.raises(AgentError) as err:
            execute_tool('page.recompute', {}, session={'scope': 'indicator_center'}, page_context=page,
                          service=service, page_snapshot=snapshot)
        assert err.value.code == 'AGENT_CONTEXT_CHANGED' and len(service.evaluate_calls) == 1
    finally:
        reset_view_override(token)


def test_working_sources_are_bounded_and_full_source_pages_remain_readable(tmp_path):
    from agent.sessions import append_event
    from agent.tools import execute_tool
    store, page, session, service = setup(tmp_path); sid = session['session_id']
    with store.locked(sid) as state:
        for i in range(230):
            append_event(state, {'type': 'user.message', 'id': f'm{i}', 'text': f'第{i}条原话：禁止未来数据。'})
        store.write(state)
    state, _ = ledger.rebuild(store, sid)
    working = ledger.working_view(state)
    assert working['source_count'] == 230 and len(working['sources']) == 6
    assert working['omitted_source_count'] == 224 and working['sources'][0]['id'] == 'm0'
    result = execute_tool('task.read', {'section': 'sources', 'offset': 100, 'limit': 20},
                          session=store.read(sid), page_context=page, service=service, store=store, session_id=sid)
    assert result['result']['total'] == 230 and result['result']['sources'][0]['id'] == 'm100'
    assert result['result']['next_offset'] == 120


def test_memory_rejection_and_busy_guard_are_independent_of_business_save(tmp_path):
    store, page, session, service = setup(tmp_path); sid = session['session_id']
    p = proposal_turn(store, page, sid, service, 'propose', '希望说明更简洁。')
    rejected = memory.resolve(store=store, session_id=sid, request=MemoryRequest(proposal_id=p['proposal_id'], decision='reject'))
    assert rejected['decision'] == 'reject' and memory.recall(store=store, session_id=sid) == []
    with store.locked(sid) as state:
        state['active_run_id'] = 'busy'
        store.write(state)
    with pytest.raises(AgentError) as err:
        memory.resolve(store=store, session_id=sid, request=MemoryRequest(proposal_id=p['proposal_id'], decision='reject'))
    assert err.value.code == 'AGENT_SESSION_BUSY' and service.create_calls == []


def test_middle_negative_quote_and_later_correction_survive_twenty_turns_and_three_compactions(tmp_path, monkeypatch):
    from agent.sessions import append_event
    from agent.tools import execute_tool
    from agent.views import VIEW_TASK_STATE, Projection
    store, page, session, service = setup(tmp_path); sid = session['session_id']
    with store.locked(sid) as state:
        for i in range(25):
            text = '禁止复权，使用单位净值。' if i == 7 else '改为复权净值，但仍禁止未来数据。' if i == 19 else f'继续第{i}轮研究'
            append_event(state, {'type': 'user.message', 'id': f'm{i}', 'text': text})
        store.write(state)
    local = store.read(sid)
    execute_tool('task.plan', {'source_message_id': 'm19', 'capabilities': ['calculate'],
        'questions': ['尚未确认年化口径'], 'constraints': [
            {'source_message_id': 'm7', 'quote': '禁止复权，使用单位净值。'},
            {'source_message_id': 'm19', 'quote': '改为复权净值，但仍禁止未来数据。', 'supersedes_source_message_id': 'm7'}]},
        session=local, page_context=page, service=service, store=store, session_id=sid)
    with store.locked(sid) as state:
        state['task_plans'] = local['task_plans']; store.write(state)
    canonical, _ = ledger.rebuild(store, sid)
    work, _ = VIEW_TASK_STATE(ledger.working_view(canonical), Projection())
    assert 'm7' not in [item['id'] for item in work['sources']]
    assert work['quoted_constraints'][0]['quote'] == '禁止复权，使用单位净值。'
    assert work['quoted_constraints'][1]['supersedes_source_message_id'] == 'm7'
    checkpoint = {'task_state': data_policy.seal(work, 'task.state'), 'valid_evidence_refs': [], 'messages': []}
    llm = FixtureLLMClient([]); llm.context_window_tokens = 32768
    for i in range(3):
        checkpoint['messages'] += [{'role': 'assistant', 'content': '研究说明', 'reasoning_content': 'x'*70000}]
        asyncio.run(compact_if_needed(checkpoint, system='研究', llm=llm, on_compacted=lambda _: None, force=True))
        assert checkpoint['task_state']['quoted_constraints'] == work['quoted_constraints']
        assert '禁止复权' in stable_json(context_messages(checkpoint))
        assert '改为复权净值' in stable_json(context_messages(checkpoint))
        assert '尚未确认年化口径' in stable_json(context_messages(checkpoint))
    assert len(checkpoint['segments']) == 3
    with pytest.raises(AgentError):
        execute_tool('task.plan', {'source_message_id': 'm19', 'constraints': [
            {'source_message_id': 'm7', 'quote': '允许未来数据'}]}, session=local, page_context=page,
            service=service, store=store, session_id=sid)


def test_null_date_uses_persisted_pit_default_without_a_new_date(tmp_path):
    from pit.settings import PitSettingsRepository
    from agent.research_runtime import research_context
    from agent.tools import execute_tool
    from test_agent_admission import APPROVED_DEFINITION
    store, page, session, service = setup(tmp_path)
    PitSettingsRepository(tmp_path).update(active_release_id=None, run_mode='RESEARCH', as_of='2019-12-31')
    run = {}; snapshot = {'sections': {'results': {'frozen_request': {'definition': APPROVED_DEFINITION,
        'targets': [{'kind': 'etf', 'product_id': '510300.SH'}], 'period': '1Y', 'as_of': None, 'parameters': {}}}}}
    with research_context(run, page, service) as effective_page:
        assert page.calculation.as_of is None and effective_page.calculation.as_of == '2019-12-31'
        result = execute_tool('page.recompute', {}, session={'scope': 'indicator_center'}, page_context=effective_page,
                              service=service, page_snapshot=snapshot)
    assert service.evaluate_calls[-1]['as_of'] == '2019-12-31'
    assert result['frozen_inputs']['as_of'] is None and result['frozen_inputs']['effective_as_of'] == '2019-12-31'


@pytest.mark.parametrize('page_name,operation', [
    ('product-research', 'catalog'), ('product-compare', 'comparison'), ('holding-diagnosis', 'scenario'),
])
def test_page_request_changes_expire_selected_evidence_and_task_read(tmp_path, page_name, operation):
    from agent.sessions import AgentSessionStore
    from test_agent_api import FakeIndicatorService
    from test_agent_research_pages import page, snapshot, request_for, RUN_ID

    frozen = request_for(page_name)
    targets = [{key: target[key] for key in ('kind', 'product_id')} for target in frozen.get('targets', [])]
    context = page(page_name, targets)
    store = AgentSessionStore(tmp_path / 'sessions')
    sid = store.create(page_context=context, scope='product_research')['session_id']
    callbacks = {
        'catalog': lambda **_: {'items': [], 'total': 0},
        'comparison': lambda target, _: {'product_id': target['product_id'], 'ranges': {}},
        'run': lambda _: {'id': RUN_ID, 'immutable': True, 'target_revision': 4},
        'scenario': lambda id, **kw: {'source_run_id': id, 'locked_target_revision': 4,
            'observation_count': 30, 'actual_start_date': kw['start_date'], 'actual_end_date': kw['end_date'],
            'summary': {'cumulative_return': .12}},
    }

    async def run():
        controller = RunController(store, page_services=callbacks)
        try:
            original_refs = set()
            for index in range(3):
                if index == 2:
                    if page_name == 'product-research':
                        frozen['q'] = '红利'
                    elif page_name == 'product-compare':
                        frozen['ranges']['risk']['start_date'] = '2021-06-01'
                    else:
                        frozen['scenario']['start_date'] = '2015-06-01'
                evidence = snapshot(page_name, frozen)
                evidence['snapshot_id'] = 'snap-' + str(index) * 32
                evidence['captured_at'] = f'2026-09-22T01:00:0{index}Z'
                calls = ([{'name': 'page.read', 'arguments': {'section': 'request'}},
                          {'name': 'page.analyze', 'arguments': {'operation': operation}}] if index == 0
                         else [{'name': 'task.read', 'arguments': {'section': 'milestones'}},
                               {'name': 'context.read', 'arguments': {'operation_id': sorted(original_refs)[0]}}] if index == 1
                         else [{'name': 'task.read', 'arguments': {'section': 'milestones'}}])
                for number, call in enumerate(calls):
                    call['call_id'] = f'page-{index}-call-{number}'
                model = FixtureLLMClient(([{'tool_calls': calls}] if calls else []) + [{'content': '已核对口径。'}])
                task, _ = controller.submit(sid, request(context.model_copy(update={'context_revision': index + 1}),
                    message=f'page-{index}', revision=store.read(sid)['session_revision'],
                    text='解释当前页面', snapshot=evidence), model, FakeIndicatorService(tmp_path))
                done = await controller.wait(task)
                assert done['status'] == 'completed', done.get('error')
                if index == 0:
                    original_refs = {item['ref'] for item in done['checkpoint']['selected_evidence']}
                    assert original_refs
                elif index == 1:
                    assert original_refs <= {item['ref'] for item in done['checkpoint']['selected_evidence']}
                else:
                    assert not original_refs & {item['ref'] for item in done['checkpoint']['selected_evidence']}
                    assert all(item['source_message_id'] == 'page-2' for item in done['checkpoint']['selected_evidence'])
                    results = [json.loads(m['content']) for m in model.requests[-1]['messages']
                               if m['role'] == 'tool' and json.loads(m['content'])['admission']['source'] == 'task.read']
                    assert results[-1]['result']['milestones']
                    assert all(not item['current'] for item in results[-1]['result']['milestones'])
                    assert store.read_context(sid, next(iter(original_refs)))['result']['historical'] is True
        finally:
            await controller.close()
    asyncio.run(run())


def test_changing_draft_expires_previous_preview_evidence(tmp_path):
    from test_agent_api import DEFINITION
    from agent.contracts import EvaluationTarget
    store, page, session, service = setup(tmp_path)
    page = page.model_copy(deep=True)
    page.view_state = 'inherit'
    page.calculation.targets = [EvaluationTarget(kind='etf', product_id='510300.SH')]
    sid = session['session_id']
    first, _ = turn(store, page, sid, service, message='original', text='创建并试算指标', replies=[
        {'tool_calls': [{'name': 'metrics.validate', 'call_id': 'draft-original', 'arguments': {'definition': DEFINITION}}]},
        {'tool_calls': [{'name': 'metrics.preview', 'call_id': 'preview-original', 'arguments': {}}]},
        {'content': '已试算。'}])
    preview_refs = {item['ref'] for item in first['checkpoint']['task_state']['milestones'] if item['tool'] == 'metrics.preview'}
    assert preview_refs
    changed, _ = turn(store, page, sid, service, message='changed', text='修改定义后再核对', replies=[
        {'tool_calls': [{'name': 'metrics.validate', 'call_id': 'draft-changed',
                         'arguments': {'definition': {**DEFINITION, 'expression': 'mean(returns)'}}}]},
        {'content': '已修改定义，尚未重新试算。'}])
    assert not preview_refs & {item['ref'] for item in changed['checkpoint']['selected_evidence']}
    assert all(item['ref'] not in preview_refs for item in changed['checkpoint']['task_state']['milestones'])


@pytest.mark.parametrize('foreign_cancel', [False, True])
def test_tool_completion_write_failure_never_enters_future_context(tmp_path, monkeypatch, foreign_cancel):
    from test_agent_api import DEFINITION

    monkeypatch.setenv('CUSTOM_INDICATOR_DATA_DIR', str(tmp_path))
    store, page, session, service = setup(tmp_path)
    sid = session['session_id']
    real_checkpoint = store.checkpoint
    rejected_call = 'rejected-validation'
    rejected = []
    durable_before_failure = []

    def fail_completion(run, **kwargs):
        receipt = kwargs.get('tool_receipt')
        if not rejected and receipt and receipt.get('call_id') == rejected_call and receipt.get('status') == 'completed':
            rejected.append(rejected_call)
            durable_before_failure.append(store.get_run(sid, run['run_id']))
            if foreign_cancel:
                type(store)(store.root).cancel(sid, run['run_id'])
            raise AgentError('AGENT_STORAGE_UNAVAILABLE', '本次工具回执写入失败', status_code=503)
        return real_checkpoint(run, **kwargs)

    async def run():
        controller = RunController(store)
        try:
            earlier = FixtureLLMClient([
                {'tool_calls': [{'name': 'metrics.validate', 'call_id': 'accepted-validation', 'arguments': {'definition': DEFINITION}}]},
                {'content': '此前的定义已完成校验。'},
            ])
            first, _ = controller.submit(sid, request(page, message='first'), earlier, service)
            first_done = await controller.wait(first)
            assert first_done['status'] == 'completed'
            prior_draft = copy.deepcopy(store.read(sid)['draft'])
            assert prior_draft and prior_draft['valid']
            monkeypatch.setattr(store, 'checkpoint', fail_completion)
            changing = FixtureLLMClient([{'tool_calls': [{'name': 'metrics.validate', 'call_id': rejected_call,
                'arguments': {'definition': {**DEFINITION, 'expression': 'mean(returns)'}}}]}])
            second, _ = controller.submit(sid, request(page, message='second', revision=store.read(sid)['session_revision'],
                text='修改定义后再核验'), changing, service)
            done = await controller.wait(second)
            assert rejected == [rejected_call]
            assert done['status'] in {'failed', 'cancelled'}
            assert done.get('detector_state', {}) == durable_before_failure[0].get('detector_state', {})
            assert done.get('tool_trace', []) == durable_before_failure[0].get('tool_trace', [])
            assert done['response']['tool_trace'] == durable_before_failure[0].get('tool_trace', [])
            state = store.read(sid)
            assert state['draft'] == prior_draft
            assert store.get_run(sid, first['run_id']) == first_done
            with store.connection() as db:
                receipt = json.loads(db.execute('SELECT body FROM tool_calls WHERE run_id=? AND call_id=?',
                    (second['run_id'], rejected_call)).fetchone()[0])
            assert receipt.get('applied') is False
            assert receipt['status'] in {'started', 'discarded'}

            following = FixtureLLMClient([{'content': '继续核验已提交的定义。'}])
            third, _ = controller.submit(sid, request(page, message='third', revision=state['session_revision'],
                text='继续核验已提交进度'), following, service)
            assert (await controller.wait(third))['status'] == 'completed'
            messages = [message for sent in following.requests for message in sent['messages']]
            assert any(message.get('role') == 'assistant' and message.get('content') == '此前的定义已完成校验。' for message in messages)
            prior_results = [json.loads(message['content']) for message in messages
                if message.get('role') == 'tool' and message.get('tool_call_id') == 'accepted-validation']
            assert any(value.get('result', {}).get('valid') is True for value in prior_results)
            rejected_results = [json.loads(message['content']) for message in messages
                if message.get('role') == 'tool' and message.get('tool_call_id') == rejected_call]
            assert not any(value.get('result', {}).get('valid') is True for value in rejected_results)
            assert store.read(sid)['draft'] == prior_draft
        finally:
            await controller.close()

    asyncio.run(run())

"""Meaningful tool trajectories: productive work, repeats, repair and compaction."""
from agent.sessions import stable_json
from agent import data_policy
import asyncio
import copy

import pytest
from agent.progress import ProgressGuard, call_key
from agent.context import compact_if_needed, context_messages
from agent.contracts import AgentError
from agent.llm import FixtureLLMClient


def record(guard, tool='metrics.lookup', args=None, result=None, draft=None, dependencies=None):
    return guard.record(tool, args or {}, result or {'ok':True,'result':{'items':[{'id':'x','revision':1}]}}, draft=draft, dependencies=dependencies or {})


def test_productive_long_trajectory_has_no_total_limit():
    guard=ProgressGuard()
    for i in range(160):
        decision=record(guard,args={'query':str(i)},result={'ok':True,'result':{'items':[{'id':str(i),'revision':1}]}})
        assert decision['progress'] and not decision['pause'] and not decision['recovery_started']
    assert len(guard.snapshot()['history'])==32


def test_metadata_changes_do_not_mask_a_repeat_and_recovery_is_finite():
    guard=ProgressGuard()
    for i in range(3):
        decision=record(guard,result={'ok':True,'result':{'items':[{'id':'x','updated_at':str(i),'compile_token':str(i)}]}})
    assert decision['recovery_started'] and guard.check_call('metrics.lookup',{}, {})
    assert not guard.record_blocked('metrics.lookup',{}, {})['pause']
    assert guard.record_blocked('metrics.lookup',{}, {})['pause']
    restored=ProgressGuard(guard.snapshot(),continued=True)
    assert restored.check_call('metrics.lookup',{}, {})
    assert not restored.state['pause']


def test_same_failure_repeats_but_different_dependencies_are_allowed():
    guard=ProgressGuard();failure={'ok':False,'error':{'code':'MISSING','field':'target'}}
    record(guard,result=failure,dependencies={'generation':'v1'})
    decision=record(guard,result=failure,dependencies={'generation':'v1'})
    assert decision['reason']=='repeat_failure'
    assert guard.check_call('metrics.lookup',{}, {'generation':'v1'})
    assert not guard.check_call('metrics.lookup',{}, {'generation':'v2'})


def test_alternating_cycle_detected():
    guard=ProgressGuard()
    for i in range(6):
        v=i%2
        decision=record(guard,args={'q':v},result={'ok':True,'result':{'items':[{'id':str(v)}]}})
    assert decision['reason']=='cycle_no_progress'


def test_novel_invalid_formulas_and_cosmetic_revisions_are_not_progress():
    guard=ProgressGuard()
    for i in range(4):
        draft={'valid':False,'draft_revision':i+1,'definition':{'name':str(i),'expression':f'unknown_{i}(x)'},'diagnostics':[{'code':'UNKNOWN','field':'expression'}]}
        decision=record(guard,'metrics.validate',{'definition':draft['definition']},{'ok':True,'result':{'valid':False}},draft)
        assert not decision['progress']
    assert decision['reason']=='repair_stall'
    assert call_key('metrics.validate',{'definition':{'name':'a','expression':'x'}},{})==call_key('metrics.draft_save',{'definition':{'name':'b','expression':'x'}},{})


def test_real_validation_improvement_exits_recovery_without_erasing_rejections():
    guard=ProgressGuard()
    for _ in range(2): record(guard,result={'ok':False,'error':{'code':'WRONG'}})
    blocked=list(guard.state['blocked'])
    draft={'valid':True,'definition':{'expression':'mean(returns)'},'diagnostics':[]}
    decision=record(guard,'metrics.validate',{'definition':draft['definition']},{'ok':True,'result':{'valid':True}},draft)
    assert decision['progress'] and not guard.state['recovering']
    assert guard.state['blocked']==blocked


def test_new_directory_trivia_does_not_reset_failed_draft():
    guard=ProgressGuard();draft={'valid':False,'definition':{'expression':'unknown(x)'},'diagnostics':[{'code':'UNKNOWN'}]}
    for i in range(8):
        decision=record(guard,args={'q':str(i)},result={'ok':True,'result':{'items':[{'id':str(i)}]}},draft=draft)
    assert decision['reason']=='stagnation'


def test_compaction_keeps_complete_groups_and_guard_state(monkeypatch):
    monkeypatch.setenv('AGENT_CONTEXT_CHAR_LIMIT','8000')
    messages=[{'role':'user','content':'保留无风险利率1.5%和20日窗口'}]
    # Signed receipts add envelope bytes: keep the compaction source inside the
    # explicit 8k input bound while still crossing its soft trigger.
    for i in range(12):
        messages.extend([{'role':'assistant','content':'','tool_calls':[{'id':str(i),'type':'function','function':{'name':'metrics.lookup','arguments':'{}'}}]}, {'role':'tool','tool_call_id':str(i),'content':stable_json(data_policy.seal({'ok':True,'result':{'items':[{'id':f'item-{i}','description':'x'*200}]}},'metrics.lookup'))}])
    checkpoint={'messages':messages};guard=ProgressGuard();record(guard);before=guard.snapshot();events=[]
    asyncio.run(compact_if_needed(checkpoint,system='system',llm=FixtureLLMClient([{'content':'用户确认无风险利率1.5%，窗口20日。'}]),on_compacted=events.append))
    assert guard.snapshot()==before and events
    assert any(m.get('content') == '保留无风险利率1.5%和20日窗口' for m in context_messages(checkpoint))
    call_ids={c['id'] for m in checkpoint['messages'] for c in m.get('tool_calls',[])}
    assert call_ids=={m['tool_call_id'] for m in checkpoint['messages'] if m['role']=='tool'}


def test_impossible_context_pauses_without_recursive_compaction(monkeypatch):
    monkeypatch.setenv('AGENT_CONTEXT_CHAR_LIMIT','8000')
    checkpoint={'messages':[{'role':'user','content':'a'*9000}]}
    llm=FixtureLLMClient([])
    with pytest.raises(AgentError) as err:
        asyncio.run(compact_if_needed(checkpoint,system='s',llm=llm,on_compacted=lambda x:None))
    assert err.value.code=='AGENT_CONTEXT_CAPACITY' and not llm.requests


def test_failed_tool_cannot_claim_the_previous_valid_draft_as_progress():
    guard=ProgressGuard()
    draft={'valid':True,'definition':{'expression':'mean(returns)'},'diagnostics':[]}
    failure={'ok':False,'error':{'code':'ARGUMENTS_INVALID','field':'definition'}}
    decision=record(guard,'metrics.validate',{'wrong':'x'},failure,draft)
    assert not decision['progress'] and not guard.state['best_valid']


def test_evidence_pages_are_progress_but_reading_same_page_stalls():
    guard=ProgressGuard()
    for offset in (0,100,200):
        assert record(guard,'context.read',{'operation_id':'op-x','offset':offset},
                      {'ok':True,'result':{'offset':offset,'content':str(offset)}})['progress']
    for _ in range(3):
        decision=record(guard,'context.read',{'operation_id':'op-x','offset':200},
                        {'ok':True,'result':{'offset':200,'content':'200'}})
    assert decision['reason']=='repeat_no_change'

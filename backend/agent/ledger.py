"""Sourced task state rebuilt from active SQLite messages and applied receipts.

Whole user statements are authoritative quotes, not an NLP extraction. Proposed
plans/questions remain model suggestions; only applied service receipts certify work.
"""
from __future__ import annotations

import copy
import json
import re

from . import data_policy
from .sessions import page_snapshot_identity, stable_hash

SCHEMA_VERSION = 1


def tokens(text):
    return set(re.findall(r'[a-z0-9_]+|[\u4e00-\u9fff]{2,}', str(text).casefold()))


def permitted_quote(text):
    return text if isinstance(text, str) and not data_policy.user_text_violation(text) else data_policy.STRUCTURED_OMIT_NOTE


def rebuild(store, session_id, *, page=None, page_snapshot=None, dependencies=None, draft=None, best_draft=None):
    from .tools import TOOL_REGISTRY

    with store.connection() as db:
        state = store._read_state(db, session_id)
        users = [json.loads(row[0]) for row in db.execute(
            "SELECT body FROM events WHERE session_id=? AND json_extract(body,'$.type')='user.message' ORDER BY seq", (session_id,))]
        receipts = list(db.execute("""SELECT t.body,r.body AS run_body FROM tool_calls t JOIN runs r ON r.id=t.run_id
            WHERE r.session_id=? AND json_extract(r.body,'$.superseded_by') IS NULL
            AND json_extract(t.body,'$.applied')=1 AND json_extract(t.body,'$.status')='completed'
            ORDER BY json_extract(r.body,'$.created_at'),t.model_step,t.call_id""", (session_id,)))
        plans = state.get('task_plans') or []
    sources = [{"id": item.get("id") or f"event-{item['seq']}", "run_id": item.get("run_id"), "seq": item["seq"],
                "text": permitted_quote(item.get("text")), "kind": "user_statement",
                "edit_of": item.get("edit_of")} for item in users]
    source_ids = {item['id'] for item in sources}
    dependencies = dependencies or {}
    milestones, rejected, evidence, reads = [], [], [], []
    for entry in receipts:
        receipt, run = json.loads(entry['body']), json.loads(entry['run_body'])
        result, tool = receipt.get('result'), receipt.get('tool')
        if run['message_id'] not in source_ids or not data_policy.verify(result, tool):
            continue
        ref = receipt.get('operation_id')
        if not ref:
            continue
        same = all(run.get(key) == value for key, value in dependencies.items())
        definition = TOOL_REGISTRY.get(tool)
        stamp = receipt.get('dependency_stamp') or {}
        if definition and 'page_evidence' in definition.dependencies:
            section = (receipt.get('validated_arguments') or {}).get('section')
            same = (same and 'page_evidence' in stamp
                    and stamp['page_evidence'] == page_snapshot_identity(page_snapshot, section))
        if definition and 'draft' in definition.dependencies:
            current_draft = (draft if draft is not None else state.get('draft')) or {}
            same = (same and bool(current_draft.get('valid')) and not current_draft.get('stale')
                    and 'definition' in stamp and stamp['definition'] == current_draft.get('definition_hash'))
        payload = result.get('result') or {}
        item = {'ref': ref, 'tool': tool, 'source_message_id': run['message_id'],
                'dependencies': {key: run.get(key) for key in dependencies}, 'current': same}
        failed = result.get('ok') is False or (isinstance(payload, dict) and payload.get('valid') is False)
        if failed:
            item['status'] = 'rejected'
            item['code'] = str((result.get('error') or {}).get('code') or 'VALIDATION_FAILED')
            rejected.append(item)
        elif tool in {'metrics.validate', 'metrics.draft_save', 'metrics.rolling_draft'} and payload.get('valid'):
            milestones.append({**item, 'status': 'definition_validated', 'definition_hash': payload.get('definition_hash')})
        elif tool in {'metrics.preview', 'page.recompute', 'products.eval', 'products.series', 'portfolios.eval'}:
            rows = payload.get('results', []) if isinstance(payload, dict) else []
            statuses = [row.get('status') for row in rows if isinstance(row, dict)]
            milestones.append({**item, 'status': 'calculated' if statuses and all(s == 'ok' for s in statuses)
                               else 'calculation_incomplete', 'result_statuses': statuses})
        elif tool == 'page.analyze':
            milestones.append({**item, 'status': 'page_analysis_read', 'operation': payload.get('operation')})
        evidence.append({**item, 'result': result})
        if tool in {'metrics.lookup', 'products.search'} and same:
            reads.append({'tool': tool, 'kind': (receipt.get('validated_arguments') or {}).get('kind')})
    def draft_ref(value):
        if not isinstance(value, dict):
            return None
        return {key: value.get(key) for key in ('draft_revision', 'definition_hash', 'valid', 'stale')}
    active_plans = [plan for plan in plans if plan.get('source_message_id') in source_ids]
    constraints = []
    by_id = {item['id']: item for item in sources}
    for plan in active_plans:
        for quote in plan.get('constraints', []):
            source = by_id.get(quote.get('source_message_id'))
            if source and quote.get('quote') and quote['quote'] in source['text']:
                # The quotation is verified, its selection/replacement interpretation
                # remains a proposal. Both original and later quote stay visible.
                constraints.append({**quote, 'source_verified': True, 'selection_status': 'proposed'})
    constraints = list({stable_hash(item): item for item in constraints}.values())
    ledger = {'schema_version': SCHEMA_VERSION, 'policy_version': data_policy.POLICY_VERSION,
              'session_id': session_id, 'scope': state['scope'], 'sources': sources,
              'latest_source_id': sources[-1]['id'] if sources else None,
              'precedence': 'current_user_over_preferences; later_user_statements_over_earlier_conflicts',
              'page_context': page if page is not None else state.get('page_context'),
              'dependencies': dependencies, 'plans': active_plans,
              'quoted_constraints': constraints,
              'pending_questions': [question for plan in active_plans for question in plan.get('questions', [])],
              'current_draft': draft_ref(draft if draft is not None else state.get('draft')),
              'best_valid_draft': draft_ref(best_draft if best_draft is not None else state.get('last_valid_draft')),
              'milestones': milestones, 'rejected_strategies': rejected,
              'read_evidence': reads,
              'goal_status': 'open'}
    ledger['revision'] = stable_hash(ledger)
    return ledger, evidence


def relevant(tool, arguments, ledger, draft=None):
    """Read novelty is useful only when linked to a user goal or a chosen formula."""
    if tool in {'task.read', 'task.plan', 'memory.propose'}:
        return False
    if tool not in {'metrics.lookup', 'products.search'}:
        return True
    query = str(arguments.get('query') or '').strip()
    if not query:
        return True  # registered directory discovery, repeats are already detected
    text = ' '.join(source['text'] for source in ledger.get('sources', []))
    text += ' ' + json.dumps((draft or {}).get('definition') or {}, ensure_ascii=False)
    kind = arguments.get('kind')
    if kind in {'variables', 'operators', 'indicators'} and any(word in text.casefold() for word in ('全部', '逐一', '所有', 'every', 'all ')):
        labels = {'variables': ('变量', 'variable'), 'operators': ('算子', 'operator'), 'indicators': ('指标', 'indicator')}
        if any(label in text.casefold() for label in labels[kind]):
            return True
    if query.casefold() in text.casefold() or tokens(query) & tokens(text):
        return True
    # Referenced operator/variable vocabulary may require translation: one
    # exploratory lookup per kind is permitted, then require a concrete link.
    return not any(item.get('tool') == tool for item in ledger.get('read_evidence', [])
                   if item.get('kind') == arguments.get('kind'))


def select_evidence(evidence, ledger, *, limit=6):
    query = tokens(' '.join(item['text'] for item in ledger.get('sources', [])[-3:]))
    # State and historical-read receipts can contain old "current" claims. Rebuild
    # state from source receipts; never promote a cached wrapper as fresh evidence.
    candidates = [item for item in evidence if item['current']
                  and item['tool'] not in {'task.read', 'task.plan', 'memory.propose', 'context.read'}
                  and len(json.dumps(item['result'], ensure_ascii=False)) < 2400]
    candidates.sort(key=lambda item: (bool(query & tokens(json.dumps(item['result'], ensure_ascii=False))),
                                     item['source_message_id'] == ledger.get('latest_source_id')), reverse=True)
    return copy.deepcopy(candidates[:limit])


def public_view(ledger):
    return {key: copy.deepcopy(ledger.get(key)) for key in ('schema_version', 'policy_version', 'revision',
            'session_id', 'scope', 'precedence',
            'quoted_constraints',
            'sources', 'latest_source_id', 'page_context', 'dependencies', 'plans', 'pending_questions',
            'current_draft', 'best_valid_draft', 'milestones', 'rejected_strategies', 'goal_status')}


def page_view(state, section, offset, limit):
    rows = state.get(section) or []
    return {'revision': state['revision'], section: rows[offset:offset+limit], 'section': section,
            'offset': offset, 'total': len(rows), 'next_offset': offset+limit if offset+limit < len(rows) else None}


def working_view(state):
    """Bounded verbatim sources plus a complete task.read index, never a lossy authority."""
    result = public_view(state)
    sources = state.get('sources', [])
    selected = sources[:1] + sources[-5:] if len(sources) > 6 else sources
    result['sources'] = list({item['id']: item for item in selected}.values())
    result['source_count'] = len(sources)
    result['omitted_source_count'] = len(sources) - len(result['sources'])
    result['source_read_tool'] = 'task.read'
    result['source_note'] = ('较早用户原话未全部进入本轮工作集；不能据缺失推断约束取消。需要旧口径时按task.read分页核对。'
                             if result['omitted_source_count'] else '当前来源原话完整。')
    for key in ('milestones', 'rejected_strategies'):
        rows = state.get(key, [])
        result[key] = [item for item in rows if item.get('current')][-10:]
        result[key+'_count'] = len(rows)
    result['plans'] = state.get('plans', [])[-5:]
    result['pending_questions'] = [question for plan in result['plans'] for question in plan.get('questions', [])][-10:]
    return result

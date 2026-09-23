"""Regression checks for P1's concrete data-boundary and chart-preservation failures."""
import copy
import hashlib
import hmac
import json
import math

import numpy as np
import pytest

from agent import data_policy, derivation, views
from agent.contracts import AgentError, PageContext
from agent.series_summary import series_summary, summarize
from agent.tools import execute_tool, _handle_page_read, PageReadArgs
from custom_indicators.service import CustomIndicatorService
from test_agent_admission import APPROVED_DEFINITION, SENTINEL, primary_request
from test_agent_api import authoring_context


@pytest.fixture(autouse=True)
def isolated_key(tmp_path, monkeypatch):
    monkeypatch.setenv('CUSTOM_INDICATOR_DATA_DIR', str(tmp_path))


@pytest.mark.parametrize('text', [
    '解释 [{"date":"2026-01-02","x":987654.321}]',
    'date,x\n2026-01-02,987654.321',
    '| date | x |\n| --- | --- |\n| 2026-01-02 | 987654.321 |',
    'x\n987654.321',
    '说明 ' + json.dumps(json.dumps({'x': SENTINEL})),
    '"{\\"x\\":987654.321}"',
])
def test_small_structured_text_never_reaches_any_model_role(text):
    for role in ('user', 'assistant'):
        with pytest.raises(AgentError):
            data_policy.enforce_request(**primary_request([{'role': role, 'content': text}]))
    checkpoint = {'messages': [{'role': 'assistant', 'content': text}],
                  'pinned_user_messages': [{'role': 'user', 'content': text}]}
    data_policy.scrub_checkpoint(checkpoint)
    assert '987654' not in json.dumps(checkpoint)
    data_policy.enforce_request(**primary_request(checkpoint['messages'] + checkpoint['pinned_user_messages']))


def test_arguments_and_system_use_contracts_not_container_names():
    assert not data_policy.enforce_arguments('metrics.lookup', '{"x":987654.321}')
    assert data_policy.enforce_arguments('metrics.lookup', '{"query":"returns","limit":5}')
    for definition in ({'x': SENTINEL}, {'description': '说明 [{"x":987654.321}]'}):
        with pytest.raises(AgentError):
            data_policy.enforce_request(**primary_request([], system=json.dumps({'definition': definition})))


def test_client_parameters_and_preclaimed_statistics_do_not_gain_trust():
    definition = {**APPROVED_DEFINITION, 'parameter_contract_version': '1.0',
                  'parameter_schema': [{'id': 'window', 'label': '窗口', 'type': 'integer',
                                        'default': 20.0, 'minimum': 2.0, 'maximum': 252.0}]}
    editing, _ = views.project_editing_section({'definition': definition,
        'runtime_inputs': {'runtime_parameters': {'window': 60, 'x': SENTINEL}}}, views.Projection())
    assert editing['runtime_inputs']['runtime_parameters'] == {'window': 60}
    series, _ = views.project_series_section({'groups': [{'channels': [{
        'id': 'nav', 'values': [SENTINEL], 'mean': SENTINEL, 'minimum': SENTINEL,
        'maximum': SENTINEL, 'precision': 987654}]}]}, views.Projection())
    assert '987654' not in json.dumps(series)
    assert series['groups'][0]['channels'][0]['finite_count'] == 1


@pytest.mark.parametrize('tool', ['page.read', 'context.read', 'metrics.lookup'])
def test_legacy_receipt_cannot_be_resigned_through_a_view(tool):
    legacy = {'ok': True, 'result': {'content': '[{"x":987654.321}]'},
              'admission': {'policy_version': data_policy.POLICY_VERSION, 'source': tool, 'signature': '0'*64}}
    rebuilt = data_policy.reproject_evidence(tool, legacy, view=views.VIEW_PAGE_READ)
    assert data_policy.verify(rebuilt, tool)
    assert rebuilt['status'] == 'policy_reprojected'
    assert '987654' not in json.dumps(rebuilt)


def test_previous_page_receipts_and_summaries_expire_without_losing_confirmed_memory():
    # Reproduce the persisted signature format from before the page-view fix.
    def previous_signature(body, purpose, source):
        payload = {'policy_version': data_policy.POLICY_VERSION, 'purpose': purpose, 'source': source, 'body': body}
        return hmac.new(data_policy._signing_key(), data_policy.stable_json(payload).encode(), hashlib.sha256).hexdigest()

    for tool in ('page.read', 'context.read', 'task.state'):
        body = {'ok': True, 'result': {'content': json.dumps({'management_fee': SENTINEL})}}
        old = {**body, 'admission': {'policy_version': data_policy.POLICY_VERSION, 'source': tool,
                                   'signature': previous_signature(body, 'tool_result', tool)}}
        assert not data_policy.verify(old, tool)
        rebuilt = data_policy.reproject_evidence(tool, old)
        assert data_policy.verify(rebuilt, tool) and '987654' not in json.dumps(rebuilt)
    summary = f'旧页面声明的费率为{SENTINEL}'
    checkpoint = {'messages': [], 'summary': summary, 'summary_seal': previous_signature(summary, 'summary', 'summary')}
    assert data_policy.scrub_checkpoint(checkpoint)['summary_dropped'] == 1
    assert not checkpoint['summary']
    preference = {'status': 'accepted', 'text': '以后用中文回答', 'source_verified': True}
    old_memory = {**preference, 'admission': {'policy_version': data_policy.POLICY_VERSION, 'source': 'memory.record',
        'signature': previous_signature(preference, 'tool_result', 'memory.record')}}
    assert data_policy.verify(old_memory, 'memory.record')


def test_compiled_summary_is_correct_fixed_and_readonly():
    before = tuple(series_summary.signatures)
    for values in ([], [math.nan, math.inf], [0.0], [0.0, 2.0], [1e8+1, 1e8+2, 1e8+3]):
        result = summarize(values)
        finite = np.array([v for v in values if math.isfinite(v)], dtype=np.float64)
        assert result['finite_count'] == len(finite)
        if len(finite):
            assert result['mean'] == pytest.approx(float(np.mean(finite)))
            expected = float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0
            assert result['std'] == pytest.approx(expected)
        else:
            assert result['mean'] is None and result['std'] is None
    assert tuple(series_summary.signatures) == before and len(before) == 1
    assert len(series_summary.nopython_signatures) == 1 and not series_summary._can_compile
    array = np.array([0.0, 2.0]); array.flags.writeable = False
    assert series_summary(array)[5] == pytest.approx(math.sqrt(2))
    assert np.array_equal(array, [0.0, 2.0])


def test_scalar_sample_guard_and_full_ui_result_preservation(tmp_path):
    service = CustomIndicatorService(tmp_path, tmp_path)
    proof = derivation.definition_proof(service, {**APPROVED_DEFINITION, 'expression': 'mean(returns)'},
                                        context_kind='single_product')
    for count in (0, 1, 2):
        original = {'results': [{'value': SENTINEL, 'status': 'ok', 'window': {'observation_count': count},
                                 'series': [{'date': '2026-01-02', 'value': SENTINEL}]}]}
        before = copy.deepcopy(original)
        output, _ = views.project_evaluation_result(original, views.Projection(proof_map={(None, None): proof}),
                                                    default_kind='scalar')
        assert ('value' in output['results'][0]) == (count >= 2)
        assert original == before
    masked = derivation.definition_proof(service, {**APPROVED_DEFINITION,
        'expression': 'sum(adjusted_nav * greater_than(adjusted_nav, mean(adjusted_nav)))'}, context_kind='single_product')
    assert masked['status'] == 'unproven'
    last = derivation.definition_proof(service, {**APPROVED_DEFINITION, 'expression': 'last(adjusted_nav)'},
                                       context_kind='single_product')
    assert last['status'] == 'unproven'


def test_derived_series_is_useful_and_raw_or_one_point_is_not(tmp_path):
    service = CustomIndicatorService(tmp_path, tmp_path)
    for expr, permitted in [('returns', True), ('drawdown_series(adjusted_nav)', True),
                            ('rolling_apply(mean(returns), 3)', True),
                            ('rolling_apply(mean(adjusted_nav), 3)', False),
                            ('adjusted_nav * 2', False), ('rolling_apply(mean(adjusted_nav), 3, 1)', False),
                            ('rolling_apply(last(adjusted_nav), 3)', False)]:
        definition = {**APPROVED_DEFINITION, 'result_kind': 'time_series', 'expression': expr,
                      'series_outputs': [{'id': 'x', 'expression': expr}]}
        proof = derivation.definition_proof(service, definition, context_kind='single_product')
        for values in ([0.0, 2.0], [SENTINEL], [SENTINEL, math.nan], [None, SENTINEL, math.inf]):
            row = {'window': {'observation_count': len(values)}, 'channels': [{'id': 'x', 'values': values}]}
            output = views.project_result_row(row, proof, kind='time_series', counter=views.Counter())
            channel = output['channels'][0]
            assert ('mean' in channel) == (permitted and sum(v is not None and math.isfinite(v) for v in values) >= 2), (expr, output)
            if 'mean' in channel:
                assert channel['mean'] == 1 and channel['std'] == pytest.approx(math.sqrt(2))
            assert row['channels'][0]['values'] == values
    for indicator in ('builtin-annualized-sharpe-v2', 'builtin-maximum-drawdown-v2', 'builtin-total-return-v2'):
        assert derivation.saved_definition_proof(service, indicator, 1)['status'] == 'approved'


def test_preview_source_must_match_the_frozen_page(tmp_path):
    service = CustomIndicatorService(tmp_path, tmp_path)
    target = {'kind': 'etf', 'product_id': '510300.SH'}
    artifact = {'preview_id': 'p', 'run_id': 'r', 'definition_hash': 'd', 'context_hash': 'c',
                'definition': APPROVED_DEFINITION, 'target': target, 'period': '1Y', 'as_of': None,
                'result': {'results': [{'status': 'ok', 'value': 0.25, 'window': {'observation_count': 10}}]}}
    class Store:
        def preview(self, *args, **kwargs): return artifact
    section = {'provenance': {key: artifact[key] for key in ('preview_id', 'run_id', 'definition_hash', 'context_hash')},
               'frozen_definition': APPROVED_DEFINITION,
               'frozen_request': {'targets': [target], 'period': '1Y', 'as_of': None}, 'groups': [{'target': target}]}
    snapshot = {'page': 'indicator-studio', 'sections': {'results': section}}
    def read():
        receipt = _handle_page_read(PageReadArgs(section='results', limit=8000), snapshot,
                                   service=service, session_id='s', store=Store())
        return json.loads(receipt['result']['content'])['groups'][0]
    assert read()['server_verified']['value'] == 0.25
    section['frozen_request']['period'] = '3M'
    assert 'server_verified' not in read()


def test_raw_reasoning_and_extra_message_fields_are_blocked():
    for message in (
        {'role': 'assistant', 'content': '', 'reasoning_content': '表 [{"x":987654.321}]'},
        {'role': 'assistant', 'content': '', 'extra': {'x': SENTINEL}},
    ):
        with pytest.raises(AgentError):
            data_policy.enforce_request(**primary_request([message]))
    assert data_policy.enforce_arguments('metrics.infer', '{"expression":"0"}')


def test_compaction_does_not_accept_a_generated_raw_summary():
    import asyncio
    from agent.context import compact_if_needed
    from agent.llm import FixtureLLMClient
    safe_summary = '已经确认复权口径；窗口二十日。' * 400
    checkpoint = {'summary': safe_summary, 'summary_seal': data_policy.seal_text(safe_summary, 'summary'),
                  'messages': [{'role': 'user', 'content': '保持口径'},
                               {'role': 'assistant', 'content': '继续研究', 'reasoning_content': 'r' * 60000}]}
    llm = FixtureLLMClient([{'content': '[{"x":987654.321}]'}]); llm.context_window_tokens = 20000
    asyncio.run(compact_if_needed(checkpoint, system='整理研究', llm=llm, on_compacted=lambda _: None))
    assert llm.requests and '987654' not in json.dumps(checkpoint)
    assert data_policy.verify_text(checkpoint['summary'], 'summary', checkpoint['summary_seal'])


def test_key_generation_is_consistent_across_concurrent_first_requests():
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=4) as pool:
        signatures = list(pool.map(lambda _: data_policy.seal_text('same', 'primary'), range(12)))
    assert len(set(signatures)) == 1


@pytest.mark.parametrize('expression', ['mean(volume)', 'sum(volume)', 'mean(unit_nav)', 'mean(adjusted_nav)'])
def test_axis_count_is_not_finite_raw_sample_proof(tmp_path, expression):
    service = CustomIndicatorService(tmp_path, tmp_path)
    proof = derivation.definition_proof(service, {**APPROVED_DEFINITION, 'expression': expression},
                                        context_kind='single_product')
    # Actual compiled summary proves this array has two positions but only one
    # finite observation. Existing service receipts do not carry this evidence.
    observations = [SENTINEL, math.nan]
    finite = summarize(observations)
    assert finite['point_count'] == 2 and finite['finite_count'] == 1
    assert finite['mean'] == SENTINEL
    row = {'status': 'ok', 'value': finite['mean'],
           'window': {'observation_count': 2, 'coverage': {'volume': {'window_rows': 2}}}}
    projected = views.project_result_row(row, proof, kind='scalar', counter=views.Counter())
    assert proof['status'] == 'unproven' and 'value' not in projected
    assert '987654' not in json.dumps(projected)
    count = derivation.definition_proof(service, {**APPROVED_DEFINITION, 'expression': 'length(volume)'},
                                        context_kind='single_product')
    assert count['status'] == 'approved'


@pytest.mark.parametrize('frozen,declared,effective,allowed', [
    ('2027-01-01', '2026-09-18', '2026-09-18', False),
    ('2027-01-01', '2027-01-01', '2026-09-18', False),
    ('2026-09-17', '2026-09-18', '2026-09-18', False),
    (None, '2026-09-18', '2026-09-18', True),
    ('2026-09-18', '2026-09-18', '2026-09-18', True),
])
def test_page_recompute_cannot_override_effective_pit(tmp_path, frozen, declared, effective, allowed):
    from pit.context import ResearchContext, set_view_override, reset_view_override
    service = CustomIndicatorService(tmp_path, tmp_path)
    called = []
    service.validate = lambda _: {'valid': True, 'compile_token': 'test'}
    service.evaluate = lambda **kw: called.append(kw) or {'results': []}
    page = authoring_context(); page['calculation']['as_of'] = declared
    context = PageContext.model_validate(page)
    snapshot = {'sections': {'results': {'frozen_request': {
        'definition': APPROVED_DEFINITION, 'parameters': {}, 'period': context.calculation.period,
        'targets': [{'kind': 'etf', 'product_id': '510300.SH'}], 'as_of': frozen}}}}
    token = set_view_override(ResearchContext(as_of=effective))
    try:
        if allowed:
            execute_tool('page.recompute', {}, session={'scope': 'indicator_center'}, page_context=context,
                         service=service, page_snapshot=snapshot)
            assert called[0]['as_of'] == effective
        else:
            with pytest.raises(AgentError) as error:
                execute_tool('page.recompute', {}, session={'scope': 'indicator_center'}, page_context=context,
                             service=service, page_snapshot=snapshot)
            assert error.value.code == 'AGENT_CONTEXT_CHANGED' and called == []
    finally:
        reset_view_override(token)

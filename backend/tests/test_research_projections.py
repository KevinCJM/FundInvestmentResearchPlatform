"""Regression checks for P1's concrete data-boundary and chart-preservation failures."""
import copy
import hashlib
import hmac
import json
import math

import numpy as np
import pytest

from research_access import data_policy, derivation, views
from research_access.contracts import ResearchError, PageContext
from research_access.series_summary import series_summary, summarize
from research_access.tools import execute_business, _handle_page_read, PageReadArgs
from custom_indicators.service import CustomIndicatorService
from research_fixtures import APPROVED_DEFINITION, SENTINEL
from research_fixtures import authoring_context


@pytest.fixture(autouse=True)
def isolated_key(tmp_path, monkeypatch):
    monkeypatch.setenv('CUSTOM_INDICATOR_DATA_DIR', str(tmp_path))






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






def test_key_generation_is_consistent_across_concurrent_first_requests():
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=4) as pool:
        signatures = list(pool.map(lambda _: data_policy.seal({'message': 'same'}, 'page.read')['admission']['signature'], range(12)))
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
            execute_business('page.recompute', {}, authoring={'scope': 'indicator_center'}, page_context=context,
                         service=service, page_snapshot=snapshot)
            assert called[0]['as_of'] == effective
        else:
            with pytest.raises(ResearchError) as error:
                execute_business('page.recompute', {}, authoring={'scope': 'indicator_center'}, page_context=context,
                             service=service, page_snapshot=snapshot)
            assert error.value.code == 'AGENT_CONTEXT_CHANGED' and called == []
    finally:
        reset_view_override(token)

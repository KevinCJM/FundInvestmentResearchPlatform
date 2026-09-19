"""New realtime studies must bind a verified reference before research publication."""
import copy
import pytest
from custom_indicators.errors import ValidationError
from test_regime_reliability_service import case


def test_unbound_study_saves_and_previews_but_cannot_activate_or_publish(case):
    graph, model, _, _ = case
    payload = copy.deepcopy(model)
    payload['study'].pop('reference')
    saved = graph.create_definition(payload)
    ref = dict(schema_version='2.0', id=saved['id'], revision=1)
    plan = graph.prepare(saved)
    job = graph.create_preview(saved, mode='realtime', compile_token=plan['compile_token'])
    assert job['id']
    with pytest.raises(ValidationError, match='先绑定'):
        graph.enable_research_version(ref, 'realtime', compile_token=plan['compile_token'])
    run = graph.run_saved(ref, 'realtime', compile_token=plan['compile_token'])
    with pytest.raises(ValidationError, match='先绑定'):
        graph.publish(run['id'], 'research_display')
    assert not graph.runs.get(run['id'])['publications']


def test_bound_study_can_enable_and_invalid_reference_or_mapping_is_rejected(case):
    graph, model, reference, _ = case
    ref = dict(schema_version='2.0', id=model['id'], revision=1)
    before = graph.runs.get(reference['run_id'])
    version = graph.enable_research_version(ref, 'realtime', compile_token=graph.prepare(model)['compile_token'])
    assert version['available_for'] == ['product_research', 'research_display']
    assert graph.runs.get(reference['run_id']) == before
    graph.publish(reference['run_id'], 'product_research')
    references = graph.reliability.references()['items']
    historical = [item for item in references if item['run_id'] == reference['run_id']]
    assert {item['publication_usage'] for item in historical} == {'research_display', 'product_research'}
    assert next(item for item in historical if item['publication_usage'] == 'research_display')['publication_id'] == reference['publication_id']
    for patch in [dict(reference={**reference, 'content_hash': '0'*64}), dict(state_mapping={s['id']: 'missing' for s in model['states']})]:
        candidate = copy.deepcopy(model)
        candidate['study'].update(patch)
        saved = graph.create_definition(candidate)
        with pytest.raises(ValidationError):
            graph.enable_research_version(dict(schema_version='2.0', id=saved['id'], revision=1), 'realtime', compile_token=graph.prepare(saved)['compile_token'])

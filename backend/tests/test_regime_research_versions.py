"""One-action research availability, without weakening formal-publication gates."""
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from custom_indicators.errors import ValidationError
from historical_regimes.v2_service import RegimeGraphV2Service
from historical_regimes.v2_contracts import definition_content_hash, parse_definition_v2
from historical_regimes.v2_templates import get_template_v2
from services import historical_regime_routes, instrument_routes
from test_merrill_clock import write_macro


@pytest.fixture
def scenario(tmp_path):
    write_macro(tmp_path)
    service = RegimeGraphV2Service(tmp_path, tmp_path)
    saved = service.create_definition(get_template_v2('merrill-clock-macro-v3-steps-v1')['definition'])
    reference = {'schema_version': '2.0', 'id': saved['id'], 'revision': 1}
    token = service.prepare(saved)['compile_token']
    return service, saved, reference, token


def test_merrill_save_is_reusable_in_product_research_and_not_taa(scenario, monkeypatch):
    service, saved, reference, token = scenario
    version = service.enable_research_version(reference, 'retrospective', compile_token=token)
    assert version['revision'] == 1
    assert version['series_summary']['row_count'] == 18
    assert version['available_for'] == ['product_research', 'research_display']
    monkeypatch.setattr(instrument_routes, '_historical_regime_workspace_dir', lambda: service.workspace_data_dir)
    resolved, lineage = instrument_routes._resolve_product_regime_reference(
        instrument_routes.ProductAnalysisRegime(run_id=version['run_id'], publication_id=version['publication_id']))
    assert resolved['segments'] and {state['label'] for state in resolved['states']} == {'复苏', '过热', '滞胀', '衰退'}
    assert lineage['definition_revision'] == saved['revision']
    run = service.get_run(version['run_id'])
    assert run['calculation_audit']['python_fallback'] == 0
    assert run['calculation_audit']['request_time_compilation'] == 0
    with pytest.raises(ValidationError, match='发布门禁'):
        service.publish(version['run_id'], 'taa')
    with pytest.raises(ValidationError, match='发布日期未知'):
        service.enable_research_version(reference, 'realtime', compile_token=token)


def test_repeated_and_concurrent_activation_reuses_version(scenario):
    service, _, reference, token = scenario
    def activate():
        return service.enable_research_version(reference, 'retrospective', compile_token=token)
    first = activate()
    with patch.object(service, 'run_saved', side_effect=AssertionError('must not recompute')):
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _: activate(), range(2)))
    assert results == [first, first]
    assert len(service.runs.list()) == 1
    assert len(service.runs.get(first['run_id'])['publications']) == 2


def test_failed_publish_retry_reuses_calculation_and_new_revision_keeps_old(scenario):
    service, saved, reference, token = scenario
    with patch.object(service, 'publish', side_effect=ValidationError('FAIL', '测试失败')):
        with pytest.raises(ValidationError, match='测试失败'):
            service.enable_research_version(reference, 'retrospective', compile_token=token)
    assert len(service.runs.list()) == 1 and not service.runs.list()[0]['publications']
    with patch.object(service, 'run_saved', side_effect=AssertionError('must reuse completed run')):
        first = service.enable_research_version(reference, 'retrospective', compile_token=token)
    snapshot = service.runs.get(first['run_id'])
    revised = service.update_definition(saved['id'], 1, {**saved, 'name': '美林新版本'})
    second = service.enable_research_version({**reference, 'revision': 2}, 'retrospective',
                                            compile_token=service.prepare(revised)['compile_token'])
    assert second['revision'] == 2 and second['run_id'] != first['run_id']
    assert service.runs.get(first['run_id']) == snapshot
    assert service.get_definition(saved['id'], 1)['name'] == saved['name']


def test_tampered_results_are_not_enabled(scenario):
    service, _, reference, token = scenario
    version = service.enable_research_version(reference, 'retrospective', compile_token=token)
    with service.runs.store.locked():
        payload = service.runs.store.read_unlocked()
        payload['items'][0]['name'] = 'tampered'
        service.runs.store.write_unlocked(payload)
    with pytest.raises(ValidationError, match='完整性'):
        service.enable_research_version(reference, 'retrospective', compile_token=token)
    assert version['revision'] == 1


def test_empty_classification_is_not_made_available(scenario):
    service, saved, reference, _ = scenario
    next(node for node in saved['graph']['nodes'] if node['id'] == 'confirmed')['parameters']['confirmation'] = 60
    updated = service.update_definition(saved['id'], 1, saved)
    token = service.prepare(updated)['compile_token']
    with pytest.raises(ValidationError, match='尚无有效情景区间'):
        service.enable_research_version({**reference, 'revision': 2}, 'retrospective', compile_token=token)
    assert all(not run['publications'] for run in service.runs.list())


def test_default_mode_roundtrip_preserves_old_definition_hash(scenario):
    service, saved, _, _ = scenario
    template = get_template_v2('merrill-clock-v2-steps-v1')
    historical = template['definition']
    # Template hashes raw authoring data; saved definitions hash normalized models.
    assert template['content_hash'] == 'bc1820f59128e8ae885744f81d0247609ced29ef78468851824d0f16b180f961'
    assert definition_content_hash(parse_definition_v2(historical)) == definition_content_hash(parse_definition_v2({**historical, 'default_mode': None}))
    assert 'default_mode' not in parse_definition_v2(historical).model_dump(mode='json')
    revised = service.update_definition(saved['id'], 1, {**saved, 'default_mode': 'retrospective'})
    assert revised['default_mode'] == service.get_definition(saved['id'], 2)['default_mode'] == 'retrospective'


def test_route_requires_saved_reference_and_never_accepts_formal_usage(scenario, monkeypatch):
    service, saved, reference, token = scenario
    monkeypatch.setattr(historical_regime_routes, 'regime_graph_v2_service', service)
    app = FastAPI(); app.include_router(historical_regime_routes.router)
    with TestClient(app) as client:
        payload = {'definition': reference, 'mode': 'retrospective', 'compile_token': token}
        assert client.post('/api/historical-regimes/research-versions', json={**payload, 'usage': 'taa'}).status_code == 422
        assert client.post('/api/historical-regimes/research-versions', json={**payload, 'definition': saved}).status_code == 422
        assert client.post('/api/historical-regimes/research-versions', json={**payload, 'compile_token': 'wrong'}).status_code == 422
        response = client.post('/api/historical-regimes/research-versions', json=payload)
        assert response.status_code == 200
        assert response.json()['definition_id'] == saved['id']

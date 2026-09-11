import copy
from types import SimpleNamespace
import pytest
from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.testclient import TestClient
from custom_indicators.errors import IndicatorDomainError, ValidationError, ConflictError
from historical_regimes.event_library import EventLibraryService
from historical_regimes.event_routes import install_event_routes
from historical_regimes.v2_service import RegimeGraphV2Service
from historical_regimes.v2_templates import instantiate_template_v2


def draft():
    return {'name': '供应链冲击', 'categories': ['geopolitical', 'supply_chain'], 'regions': ['中东'],
            'windows': [{'id': 'acute', 'label': '急性冲击窗口', 'start_date': '2020-01-01', 'end_date': '2020-02-01', 'rationale': '选取最初市场压力阶段；并非事件全部持续时间。'}]}


@pytest.fixture
def library(tmp_path):
    return EventLibraryService(tmp_path)


@pytest.fixture
def api(library):
    app = FastAPI(); router = APIRouter()
    def call(method, *args, **kwargs):
        try:
            return method(*args, **kwargs)
        except IndicatorDomainError as exc:
            raise HTTPException(exc.status_code, detail=exc.detail()) from exc
    install_event_routes(router, lambda: SimpleNamespace(event_library=library), call)
    app.include_router(router)
    return TestClient(app)


def test_versions_windows_and_archive_do_not_change_old_references(library):
    event = library.create(draft())
    selection = {'event_id': event['id'], 'revision': 1, 'window_id': 'acute'}
    frozen = library.resolve([selection])[0]
    fields = draft(); fields['windows'][0]['end_date'] = '2020-03-01'
    updated = library.update(event['id'], 1, fields)
    assert updated['revision'] == 2
    assert library.get(event['id'], 1)['windows'][0]['end_date'] == '2020-02-01'
    assert library.resolve([selection])[0] == frozen
    fields['archived'] = True
    library.update(event['id'], 2, fields)
    with pytest.raises(ConflictError): library.resolve([selection])
    assert library.resolve([selection], allow_archived=True)[0] == frozen
    assert not library.list()['items']
    assert library.list(archived=True)['total'] == 1


def test_conflicts_and_forged_references_are_rejected(library):
    event = library.create(draft())
    with pytest.raises(ConflictError): library.update(event['id'], 2, draft())
    manual = library.resolve([{'event_id': event['id'], 'revision': 1, 'window_id': 'acute'}])[0]
    definition = {'graph': {'nodes': [{'id': 'events', 'type': 'annotation.manual_events', 'parameters': {'events': [manual]}}]}}
    library.verify_definition(definition)
    manual['start_date'] = '2019-01-01'
    with pytest.raises(ValidationError, match='锁定'): library.verify_definition(definition)


@pytest.mark.parametrize('change', [
    {'name': '  '}, {'verification': 'verified'}, {'status': 'ongoing', 'fact_end': '2021-01-01'},
    {'categories': ['made_up']}, {'fact_start': '2022-01-01', 'fact_end': '2020-01-01'},
    {'known_at': '2020-01-01T09:00:00'},
    {'sources': [{'title': 'x', 'url': 'javascript:alert(1)'}]},
    {'sources': [{'title': 'x', 'url': 'https://user:password@example.com'}]},
    {'regions': ['x', 'x']},
])
def test_invalid_metadata_is_not_persisted(library, change):
    with pytest.raises(ValidationError): library.create({**draft(), **change})
    assert library.list()['total'] == 0


def test_import_is_idempotent_unreviewed_and_keeps_original(library):
    definition = instantiate_template_v2('manual-historical-events-v1')
    definition.update(id='regime-demo', revision=2)
    definition['graph']['nodes'][1]['parameters']['events'] = [
        {'id': 'event_a', 'label': '冲击A', 'start_date': '2000-01-01', 'end_date': '2001-01-01', 'color': '#112233'},
        {'id': 'event_b', 'label': '冲击B', 'start_date': '2000-06-01', 'end_date': '2001-06-01', 'color': '#223344'},
    ]
    original = copy.deepcopy(definition)
    assert library.import_definition(definition)['imported'] == 2
    assert library.import_definition(definition)['skipped'] == 2
    assert definition == original
    for event in library.list()['items']:
        assert event['verification'] == 'unreviewed'
        assert event['fact_start'] is None and event['fact_end'] is None
        assert event['provenance']['revision'] == 2


def test_packs_deduplicate_and_dates_filter_research_not_fact(library):
    event = library.create(draft())
    selection = library.pack('supply_chain')['selections']
    assert len(library.resolve(selection + selection)) == 1
    assert library.list(start='2020-01-15', end='2020-01-16')['total'] == 1
    assert library.list(start='2022-01-01')['total'] == 0
    assert library.list(category='geopolitical', region='中东', query='供应')['total'] == 1
    with pytest.raises(ValidationError): library.resolve(selection * 101)


def test_real_routes_crud_contract(api):
    path = '/api/historical-regimes/event-library'
    result = api.post(path + '/events', json={'event': draft()})
    assert result.status_code == 201, result.text
    saved = result.json()
    assert api.get(path + '/events').json()['total'] == 1
    assert api.get(path + '/events/' + saved['id'] + '/history').json()['items'][0]['revision'] == 1
    assert api.put(path + '/events/' + saved['id'], json={'revision': 10, 'event': draft()}).status_code == 409
    assert api.get(path + '/events?start=not-a-date').status_code == 422
    assert api.post(path + '/resolve', json={'selections': [{'event_id': saved['id'], 'revision': 1, 'window_id': 'acute'}]}).json()['events'][0]['library_reference']['content_hash'] == saved['content_hash']


def test_graph_save_and_prepare_verify_library_snapshots(tmp_path):
    service = RegimeGraphV2Service(workspace_data_dir=tmp_path, market_data_dir=tmp_path)
    event = service.event_library.create(draft())
    selected = service.event_library.resolve([{'event_id': event['id'], 'revision': 1, 'window_id': 'acute'}])
    definition = instantiate_template_v2('manual-historical-events-v1')
    definition['graph']['nodes'][0] = {'id': 'market', 'type': 'source.inline', 'parameters': {'rows': [
        {'observation_date': '2020-01-%02d' % day, 'available_at': '2020-01-%02d' % day, 'value': float(day)} for day in range(1, 20)]}}
    definition['graph']['nodes'][1]['parameters']['events'] = selected
    saved = service.create_definition(definition)
    service.prepare(saved)
    changed = copy.deepcopy(saved); changed['graph']['nodes'][1]['parameters']['events'][0]['label'] = '伪造来源标题'
    with pytest.raises(ValidationError): service.create_definition(changed)
    with pytest.raises(ValidationError): service.prepare(changed)

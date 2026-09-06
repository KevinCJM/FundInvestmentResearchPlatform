"""Exercise the real resolution API with temporary local candidate storage."""
from __future__ import annotations
import sys
from pathlib import Path
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from backend.data_sources.batches import capture_batch
from backend.data_sources.presets import default_interfaces
from backend.data_sources.store import SourceStore
from backend.services import data_source_routes as routes


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv('DATA_SOURCE_CENTER_ENABLED', 'true')
    store = SourceStore(tmp_path); store.seed()
    monkeypatch.setattr(routes, 'get_store', lambda: store)
    app = FastAPI(); app.include_router(routes.router)
    return TestClient(app), store


def test_config_revision_and_readonly_boundary(client, monkeypatch):
    http, _store = client
    response = http.get('/api/data-sources/resolution/config')
    assert response.status_code == 200
    saved = response.json()
    body = {'config': saved['config'], 'expected_revision': saved['revision']}
    assert http.put('/api/data-sources/resolution/config', json=body).status_code == 200
    assert http.put('/api/data-sources/resolution/config', json=body).status_code == 409
    monkeypatch.setenv('DATA_SOURCE_CENTER_ENABLED', 'false')
    assert http.put('/api/data-sources/resolution/config', json=body).status_code == 403
    response = http.post('/api/data-sources/resolution/preview', json={'table_id':'market.quote_daily', 'config': saved['config'], 'rows':[]})
    assert response.status_code == 200
    assert response.json()['published'] is False


def test_downloaded_candidates_can_be_resolved_through_api(client):
    http, store = client
    config = next(i.model_copy(deep=True) for i in default_interfaces() if i.api_name == 'fund_daily')
    for source, price in [('tushare', 11), ('akshare', 12)]:
        config.source_id = source; config.id = source + '.test_quotes'
        capture_batch(store, config, [dict(ts_code='510300.SH', trade_date='20240102', open=10, high=12, low=9, close=price)], {}, source)
    policy = http.get('/api/data-sources/resolution/config').json()
    response = http.post('/api/data-sources/resolution/run', json={'table_id':'market.quote_daily', 'expected_revision': policy['revision']})
    assert response.status_code == 200, response.text
    result = response.json()
    assert result['status'] == 'NEEDS_REVIEW'
    assert result['summary']['CONFLICT'] == 1
    assert result['summary']['selected_rows'] == 0
    assert result['published'] is False
    assert (store.root / result['artifact']).is_file()
    assert not (store.root/'tushare_active.json').exists()


def test_invalid_resolution_target_and_large_preview_fail_closed(client):
    http, _store = client
    config = http.get('/api/data-sources/resolution/config').json()['config']
    response = http.post('/api/data-sources/resolution/preview', json={'table_id':'master.instrument_identifier', 'config': config, 'rows':[]})
    assert response.status_code == 400
    response = http.post('/api/data-sources/resolution/preview', json={'table_id':'market.quote_daily', 'config': config, 'rows':[{}]*1001})
    assert response.status_code == 400
    assert response.json()['detail']['code'] == 'PREVIEW_ROWS_INVALID'


def test_start_download_requires_explicit_confirmation(client):
    http, _store = client
    response = http.post('/api/data-sources/interfaces/akshare.fund_nav/sync', json={'expected_revision':1, 'mode':'full', 'params':{}})
    assert response.status_code == 422
    assert http.get('/api/data-sources/sync/jobs').json() == []

"""Socket-peer authorization is independent of optional browser/proxy headers."""
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
import pytest

from backend.services import data_source_routes, etl_routes


@pytest.fixture
def app(monkeypatch):
    application = FastAPI()
    application.include_router(data_source_routes.router)
    application.include_router(etl_routes.router)
    def forbidden_store():
        pytest.fail('Unauthorized requests must not initialize or touch the store')
    monkeypatch.setattr(data_source_routes, 'get_store', forbidden_store)
    monkeypatch.setattr(etl_routes, 'get_store', forbidden_store)

    @application.post('/guard')
    async def guard(request: Request):
        return await data_source_routes.body(request)

    return application


@pytest.mark.parametrize('method,path', [
    ('PUT', '/api/data-sources/credentials/tushare'),
    ('PUT', '/api/data-sources/config/source'),
    ('DELETE', '/api/data-sources/config/source/tushare'),
    ('POST', '/api/data-sources/interfaces/tushare.fund_daily/sample'),
    ('POST', '/api/data-sources/interfaces/tushare.fund_daily/sync'),
    ('POST', '/api/data-sources/resolution/run'),
    ('PUT', '/api/data-sources/resolution/config'),
    ('PUT', '/api/data-sources/etl/workflows/example'),
    ('POST', '/api/data-sources/etl/runs'),
    ('POST', '/api/data-sources/etl/runs/example/recovery'),
    ('POST', '/api/data-sources/etl/runs/example/cancel'),
    ('POST', '/api/data-sources/etl/runs/example/resume'),
])
@pytest.mark.parametrize('headers', [{}, {
    'Origin': 'http://127.0.0.1', 'Sec-Fetch-Site': 'same-origin',
    'X-Forwarded-For': '127.0.0.1', 'X-Real-IP': '::1', 'Forwarded': 'for=127.0.0.1',
}])
def test_remote_source_writes_fail_before_any_store_access(app, method, path, headers):
    with TestClient(app, client=('192.0.2.1', 4321), base_url='http://127.0.0.1') as client:
        response = client.request(method, path, json={'confirm': True}, headers=headers)
    assert response.status_code == 403
    assert response.json()['detail']['code'] == 'SOURCE_LOCAL_ONLY'


@pytest.mark.parametrize('peer', ['127.0.0.1', '127.0.0.2', '::1', '::ffff:127.0.0.1'])
@pytest.mark.parametrize('origin', [None, 'http://127.0.0.1'])
def test_local_peer_can_submit_without_origin_or_with_same_origin(app, peer, origin):
    with TestClient(app, client=(peer, 4321), base_url='http://127.0.0.1') as client:
        response = client.post('/guard', json={'accepted': True}, headers={'Origin': origin} if origin else {})
    assert response.status_code == 200 and response.json() == {'accepted': True}


def test_local_dev_origin_different_port_remains_supported(app):
    with TestClient(app, client=('127.0.0.1', 4321), base_url='http://127.0.0.1:8000') as client:
        assert client.post('/guard', json={}, headers={'Origin': 'http://127.0.0.1:5173'}).status_code == 200
        assert client.post('/guard', json={}, headers={'Origin': 'https://evil.invalid'}).status_code == 403


@pytest.mark.parametrize('peer,host', [('testclient', '127.0.0.1'), ('unreadable', '127.0.0.1'),
                                    ('127.0.0.1', 'attacker.invalid'), ('2001:db8::1', '127.0.0.1')])
def test_unknown_peer_and_rebinding_host_fail_closed(app, peer, host):
    with TestClient(app, client=(peer, 4321), base_url='http://' + host) as client:
        assert client.post('/guard', json={}).status_code == 403

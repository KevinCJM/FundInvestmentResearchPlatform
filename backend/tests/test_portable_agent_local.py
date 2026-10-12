"""Offline local integration regressions; no user services, disk or provider calls."""
import importlib.util
import json
from pathlib import Path
import sys

import httpx
import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('portable_agent_local', ROOT/'scripts/portable_agent_local.py')
local = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(local)


@pytest.fixture
def config(tmp_path):
    value = dict(schema_version=1, python=sys.executable, owner='owner', workspace='local',
                 identity_key='i'*64, service_token='s'*64, issuer_key='a'*64,
                 backend_url='http://127.0.0.1:18080', browser_origin='http://127.0.0.1:14173',
                 agent_url='http://127.0.0.1:18787', model_endpoints=['https://provider.example/v1'])
    local.private_json(tmp_path/local.CONFIG_NAME, value)
    return tmp_path, value


@pytest.mark.parametrize('url', ['http://0.0.0.0:8787', 'https://127.0.0.1', 'http://example.com',
                                'http://user:pass@localhost', 'http://localhost/a', 'http://localhost:bad'])
def test_local_identity_is_never_enabled_for_public_or_ambiguous_urls(url):
    with pytest.raises(local.LocalAgentError):
        local.local_url(url)


def test_secret_config_permissions_and_no_link_overwrite(config, tmp_path):
    root, value = config
    path = root/local.CONFIG_NAME
    assert path.stat().st_mode & 0o077 == 0
    path.chmod(0o644)
    with pytest.raises(local.LocalAgentError, match='600'):
        local.read_config(root)
    target = tmp_path/'external.json'
    target.write_text('unchanged')
    link = tmp_path/'linked.json'
    link.symlink_to(target)
    with pytest.raises(local.LocalAgentError):
        local.private_json(link, value)
    assert target.read_text() == 'unchanged'


def test_separate_agent_process_receives_only_service_auth_not_source_path(config, monkeypatch):
    root, value = config
    monkeypatch.setenv('PYTHONPATH', '/unrelated/source')
    monkeypatch.setenv('HTTPS_PROXY', 'socks5://127.0.0.1:9999')
    env = local.service_environment(root, 'agent')
    assert 'PYTHONPATH' not in env and 'HTTPS_PROXY' not in env
    assert env['PORTABLE_AGENT_HOST_AUTHORIZATION'] == 'Bearer '+value['service_token']
    host = local.service_environment(root, 'backend')
    assert host['PORTABLE_AGENT_BROWSER_ORIGIN'] == value['browser_origin']
    assert host['PORTABLE_AGENT_AUTH_MODE'] == 'local'
    assert local.service_environment(root, 'frontend')['VITE_PORTABLE_AGENT_TARGET'] == value['agent_url']
    command = local.agent_command(root)
    assert command[:4] == [sys.executable, '-m', 'portable_agent', 'serve']
    assert str(root/'.data/portable-agent') in command


def transport_client(monkeypatch, handler):
    original = httpx.Client
    monkeypatch.setattr(local.httpx, 'Client', lambda **kw: original(transport=httpx.MockTransport(handler), **kw))


def handler(request):
    path = request.url.path
    if path.endswith('/local-session'):
        return httpx.Response(200, json={'authenticated': True})
    if path.endswith('/contexts'):
        return httpx.Response(200, json={'ref': 'ctx-'+'a'*32})
    if path.endswith('/bootstrap'):
        return httpx.Response(200, json={'token': 'private', 'required_capabilities': ['turn-context']})
    if path.endswith('/widget/widget.js'):
        return httpx.Response(200, text='export {}', headers={'Content-Type': 'application/javascript'})
    if path.endswith('/model-profiles'):
        return httpx.Response(200, json={'items': [{'id': 'p', 'has_key': True}], 'active_profile_id': 'p', 'revision': 1})
    return httpx.Response(200, json={'protocol_major': 2, 'capabilities': ['turn-context']})


def test_browser_readiness_uses_real_same_origin_routes(config, monkeypatch):
    root, value = config
    requests = []
    def record(request):
        requests.append(request)
        return handler(request)
    transport_client(monkeypatch, record)
    local.check(root, via_frontend=True)
    assert all(str(r.url).startswith(value['browser_origin']) for r in requests)
    assert any('/assistant/v2/' in r.url.path for r in requests)
    assert all(r.headers['Origin'] == value['browser_origin'] for r in requests)


@pytest.mark.parametrize('failure', ['auth', 'protocol', 'widget', 'model'])
def test_failed_integration_cannot_be_reported_as_ready(config, monkeypatch, failure):
    root, _ = config
    def broken(request):
        if failure == 'auth' and request.url.path.endswith('/local-session'):
            return httpx.Response(403, json={'secret': 'must-not-be-shown'})
        if failure == 'protocol' and request.url.path.endswith('/fund-research'):
            return httpx.Response(200, json={'protocol_major': 1, 'capabilities': []})
        if failure == 'widget' and '/widget/' in request.url.path:
            return httpx.Response(500)
        if failure == 'model' and request.url.path.endswith('/model-profiles'):
            return httpx.Response(200, json={'items': [], 'active_profile_id': None})
        return handler(request)
    transport_client(monkeypatch, broken)
    with pytest.raises(local.LocalAgentError) as error:
        local.check(root)
    assert 'must-not-be-shown' not in str(error.value)
    if failure == 'model':
        local.check(root, require_model=False)  # Keep settings UI available to repair it.


def test_profile_initialization_does_not_overwrite_later_user_settings(config, monkeypatch):
    root, value = config
    value['profiles_initialized'] = True
    local.private_json(root/local.CONFIG_NAME, value)
    monkeypatch.setattr(local.httpx, 'Client', lambda **_: pytest.fail('No repeated import/network allowed'))
    assert local.import_profiles(root) == 0


def test_missing_original_master_key_never_generates_replacement(config, monkeypatch):
    root, value = config
    monkeypatch.setattr(local.subprocess, 'run', lambda *a, **kw: pytest.fail('Must not run issuer/key generation'))
    with pytest.raises(local.LocalAgentError, match='主密钥缺失'):
        local.prepare(value['backend_url'], value['browser_origin'], value['agent_url'], root)
    assert not (root/'.data/portable-agent/master.key').exists()


def test_partial_profile_import_is_retryable_and_selects_original_active(config, monkeypatch):
    root, value = config
    source = [{'id': 'first', 'name': 'One', 'base_url': value['model_endpoints'][0], 'model': 'm', 'api_key': 'private'},
              {'id': 'second', 'name': 'Two', 'base_url': value['model_endpoints'][0], 'model': 'n', 'api_key': 'private'}]
    (root/'data').mkdir()
    local.private_json(root/'data/llm_settings.json', {'profiles': source, 'active_profile_id': 'second'})
    first = 'legacy-'+local.hashlib.sha256(b'first').hexdigest()[:32]
    writes = []
    def wire(request):
        if request.url.path.endswith('/model-profiles'):
            if request.method == 'GET':
                return httpx.Response(200, json={'items': [{'id': first}], 'active_profile_id': None, 'revision': 3})
            body = json.loads(request.content)
            writes.append(body)
            return httpx.Response(201, json={'id': body['request_id']})
        if request.url.path.endswith('/active'):
            body = json.loads(request.content)
            assert body['expected_revision'] == 3
            writes.append(body)
            return httpx.Response(200, json={'revision': 4})
        return handler(request)
    transport_client(monkeypatch, wire)
    assert local.import_profiles(root) == 2
    assert len(writes) == 2  # Only the missing profile and active selection.
    assert local.read_config(root)['profiles_initialized'] is True
    assert json.loads((root/'data/llm_settings.json').read_text())['profiles'] == source

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
              {'id': 'second', 'name': 'Two', 'base_url': value['model_endpoints'][0], 'model': 'n', 'api_key': 'private', 'context_window_tokens': 131072}]
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
    assert writes[0]['context_window_tokens'] == 131072
    assert local.read_config(root)['profiles_initialized'] is True
    assert json.loads((root/'data/llm_settings.json').read_text())['profiles'] == source


@pytest.mark.parametrize('files', [('master.key',), ('agent.sqlite3',), ('archive/session.json',)])
def test_existing_data_without_paired_config_cannot_create_new_identity(tmp_path, monkeypatch, files):
    directory = tmp_path/'.data/portable-agent'
    originals = {}
    for name in files:
        path = directory/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'preserve-existing-identity-data')
        originals[path] = path.read_bytes()
    monkeypatch.setenv('PORTABLE_AGENT_PYTHON', sys.executable)
    monkeypatch.setattr(local.subprocess, 'run', lambda *a, **kw: pytest.fail('Must not derive/regenerate any key'))
    with pytest.raises(local.LocalAgentError, match='原本机身份配置缺失'):
        local.prepare('http://127.0.0.1:8000', 'http://127.0.0.1:5173', 'http://127.0.0.1:8787', tmp_path)
    assert not (tmp_path/local.CONFIG_NAME).exists()
    assert {path: path.read_bytes() for path in originals} == originals


def test_first_install_checks_model_config_before_creating_a_key(tmp_path, monkeypatch):
    monkeypatch.setenv('PORTABLE_AGENT_PYTHON', sys.executable)
    monkeypatch.delenv('PORTABLE_AGENT_MODEL_ENDPOINTS', raising=False)
    monkeypatch.setattr(local.subprocess, 'run', lambda *a, **kw: pytest.fail('No key or runtime side effects before configuration'))
    with pytest.raises(local.LocalAgentError, match='模型 API 地址'):
        local.prepare('http://127.0.0.1:8000', 'http://127.0.0.1:5173', 'http://127.0.0.1:8787', tmp_path)
    assert not (tmp_path/'.data/portable-agent').exists()


@pytest.mark.parametrize('endpoint', ['not-a-url', 'ftp://provider.example/v1', 'https:///v1',
    'https://user:private@provider.example/v1', 'https://provider.example/v1?api_key=private',
    'https://provider.example/v1#private', 'https://provider.example:bad/v1', 'https://[invalid/v1'])
@pytest.mark.parametrize('source', ['environment', 'saved', 'legacy'])
def test_invalid_model_endpoint_has_no_key_or_config_side_effects(config, monkeypatch, endpoint, source):
    root, value = config
    directory = root/'.data/portable-agent'
    monkeypatch.setenv('PORTABLE_AGENT_PYTHON', sys.executable)
    monkeypatch.delenv('PORTABLE_AGENT_MODEL_ENDPOINTS', raising=False)
    if source == 'saved':
        directory.mkdir(parents=True)
        (directory/'master.key').write_bytes(b'preserve-master-key')
        value['model_endpoints'] = [endpoint]
        local.private_json(root/local.CONFIG_NAME, value)
    else:
        (root/local.CONFIG_NAME).unlink()
        if source == 'environment':
            monkeypatch.setenv('PORTABLE_AGENT_MODEL_ENDPOINTS', endpoint)
        else:
            (root/'data').mkdir()
            local.private_json(root/'data/llm_settings.json', {'settings': {'base_url': endpoint}})
    before = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    monkeypatch.setattr(local.subprocess, 'run', lambda *a, **kw: pytest.fail('Invalid endpoint must not reach runtime/key generation'))
    with pytest.raises(local.LocalAgentError, match='模型 API 地址无效') as error:
        local.prepare(value['backend_url'], value['browser_origin'], value['agent_url'], root)
    assert 'private' not in str(error.value)
    assert {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()} == before
    assert directory.exists() is (source == 'saved')


@pytest.mark.parametrize('endpoint', ['https://provider.example/v1/', 'http://127.0.0.1:19000/v1', 'https://[::1]:9443/v1'])
def test_valid_model_endpoints_keep_the_agent_contract(endpoint):
    assert local.model_endpoint(endpoint) == endpoint.rstrip('/')


def test_empty_data_directory_can_initialize_and_keep_its_identity(tmp_path, monkeypatch):
    from types import SimpleNamespace
    directory = tmp_path/'.data/portable-agent'
    directory.mkdir(parents=True)
    monkeypatch.setitem(sys.modules, 'export_portable_agent_config',
                        SimpleNamespace(configuration=lambda *args: {'apps': []}))
    monkeypatch.setenv('PORTABLE_AGENT_PYTHON', sys.executable)
    monkeypatch.setenv('PORTABLE_AGENT_MODEL_ENDPOINTS', 'https://provider.example/v1')
    def installed_wheel(args, **kwargs):
        if 'issuer-key' in args:
            (directory/'master.key').write_bytes(b'original-key')
            return SimpleNamespace(returncode=0, stdout='a'*64)
        return SimpleNamespace(returncode=0, stdout='{"protocol":2,"version":"0.2.0"}')
    monkeypatch.setattr(local.subprocess, 'run', installed_wheel)
    first = local.prepare('http://127.0.0.1:8000', 'http://127.0.0.1:5173', 'http://127.0.0.1:8787', tmp_path)
    second = local.prepare('http://127.0.0.1:8000', 'http://127.0.0.1:5173', 'http://127.0.0.1:8787', tmp_path)
    assert first == second
    assert (directory/'master.key').read_bytes() == b'original-key'


@pytest.mark.parametrize('mismatch', [None, 'disabled', 'identity', 'issuer', 'owner', 'service'])
def test_host_reuse_checks_the_current_credentials_without_an_agent(config, monkeypatch, mismatch):
    import time
    root, value = config
    seen = []
    now = int(time.time())
    def sign(audience, issuer, key, **extra):
        return local.jwt.encode(dict(sub='other' if mismatch == 'owner' else value['owner'],
            workspace=value['workspace'], iat=now, exp=now+600, aud=audience, iss=issuer, **extra), key, algorithm='HS256')
    identity = sign('fund-research-platform', 'fund-research',
                    'x'*64 if mismatch == 'identity' else value['identity_key'])
    grant = sign('portable-agent', 'portable-agent-host',
                 b'x'*32 if mismatch == 'issuer' else bytes.fromhex(value['issuer_key']), app='fund-research')
    def host(request):
        seen.append(request)
        if request.url.path.endswith('/local-session'):
            if mismatch == 'disabled':
                return httpx.Response(404, json={'error': 'local login disabled'})
            return httpx.Response(200, json={'authenticated': True},
                headers={'Set-Cookie': f'research_identity={identity}; Path=/api/; HttpOnly'})
        if request.url.path.endswith('/bootstrap'):
            return httpx.Response(200, json={'token': grant, 'context': {
                'ref': 'ctx', 'hash': 'hash', 'grant_id': 'grant', 'grant_revision': 1, 'workspace': value['workspace']}})
        if request.url.path.endswith('/authorize'):
            assert request.headers['Authorization'] == 'Bearer '+value['service_token']
            assert json.loads(request.content)['identity_only'] is True
            return httpx.Response(401, json={}) if mismatch == 'service' else httpx.Response(200, json={'allow': True})
        return handler(request)
    transport_client(monkeypatch, host)
    if mismatch:
        with pytest.raises(local.LocalAgentError):
            local.check_host(root)
    else:
        local.check_host(root)
        assert seen[-1].url.path == '/internal/portable-agent/authorize'
    assert all(str(r.url).startswith(value['backend_url']) for r in seen)

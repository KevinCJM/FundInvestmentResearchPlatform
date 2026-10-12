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
    secrets = {'TUSHARE_TOKEN', 'AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'OPENAI_API_KEY',
               'PORTABLE_AGENT_IDENTITY_KEY', 'PORTABLE_AGENT_ISSUER_KEY', 'PORTABLE_AGENT_SERVICE_TOKEN',
               'PORTABLE_AGENT_HOST_AUTHORIZATION', 'PYTHONHOME'}
    for key in secrets:
        monkeypatch.setenv(key, 'platform-private-value')
    probe_env = local.agent_environment()
    assert secrets.isdisjoint(probe_env)
    assert probe_env['PATH'] and probe_env['NO_PROXY'] == '127.0.0.1,localhost,::1'
    env = local.service_environment(root, 'agent')
    assert 'PYTHONPATH' not in env and 'HTTPS_PROXY' not in env
    assert (secrets - {'PORTABLE_AGENT_HOST_AUTHORIZATION'}).isdisjoint(env)
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
    'https://provider.example/v1#private', 'https://provider.example:bad/v1', 'https://[invalid/v1',
    'https://provider.example:0/v1', 'https://provider.example:65536/v1',
    'https://provider.exa mple/v1', 'https://provider.example\t/v1', 'https://provider.example\n/v1',
    'https://provider.example\u00a0/v1', 'https://provider.exa%20mple/v1',
    'https://provider.exa\\mple/v1'])
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


@pytest.mark.parametrize('endpoint', ['https://provider.example/v1/', 'http://127.0.0.1:19000/v1', 'https://[::1]:9443/v1',
                                'https://provider.example:1/v1', 'https://provider.example:65535/v1'])
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
            (Path(args[args.index('--data-dir')+1])/'master.key').write_bytes(b'original-key')
            return SimpleNamespace(returncode=0, stdout='a'*64)
        return SimpleNamespace(returncode=0, stdout='{"protocol":2,"version":"0.2.0"}')
    monkeypatch.setattr(local.subprocess, 'run', installed_wheel)
    first = local.prepare('http://127.0.0.1:8000', 'http://127.0.0.1:5173', 'http://127.0.0.1:8787', tmp_path)
    second = local.prepare('http://127.0.0.1:8000', 'http://127.0.0.1:5173', 'http://127.0.0.1:8787', tmp_path)
    assert first == second
    assert (directory/'master.key').read_bytes() == b'original-key'


@pytest.mark.parametrize('relative', [True, False])
@pytest.mark.parametrize('outside_cwd', [True, False])
def test_interpreter_is_persisted_absolute_without_resolving_venv_link(tmp_path, monkeypatch, relative, outside_cwd):
    from types import SimpleNamespace
    interpreter = tmp_path/'.venv/agent/bin/python'
    interpreter.parent.mkdir(parents=True)
    interpreter.symlink_to(sys.executable)
    other = tmp_path/'another-working-directory'
    other.mkdir()
    monkeypatch.chdir(other if outside_cwd else tmp_path)
    monkeypatch.setenv('PORTABLE_AGENT_PYTHON', '.venv/agent/bin/python' if relative else str(interpreter))
    monkeypatch.setenv('PORTABLE_AGENT_MODEL_ENDPOINTS', 'https://provider.example/v1')
    monkeypatch.setitem(sys.modules, 'export_portable_agent_config',
                        SimpleNamespace(configuration=lambda *args: {'apps': []}))
    def installed_wheel(args, **kwargs):
        assert args[0] == str(interpreter)
        if 'issuer-key' in args:
            key = Path(args[args.index('--data-dir')+1])/'master.key'
            if not key.exists():
                key.write_bytes(b'original-master')
            return SimpleNamespace(returncode=0, stdout='a'*64)
        return SimpleNamespace(returncode=0, stdout='{"protocol":2,"version":"0.2.0"}')
    monkeypatch.setattr(local.subprocess, 'run', installed_wheel)
    urls = ('http://127.0.0.1:8000', 'http://127.0.0.1:5173', 'http://127.0.0.1:8787')
    first = local.prepare(*urls, tmp_path)
    assert local.read_config(tmp_path) == first
    assert first['python'] == str(interpreter)
    assert local.agent_command(tmp_path)[0] == str(interpreter)
    assert local.prepare(*urls, tmp_path) == first


def test_missing_relative_interpreter_fails_before_initialization(tmp_path, monkeypatch):
    monkeypatch.setenv('PORTABLE_AGENT_PYTHON', '.venv/missing/bin/python')
    monkeypatch.setattr(local.subprocess, 'run', lambda *a, **kw: pytest.fail('Missing interpreter must have no runtime effects'))
    with pytest.raises(local.LocalAgentError, match='缺少独立助手运行环境'):
        local.prepare('http://127.0.0.1:8000', 'http://127.0.0.1:5173', 'http://127.0.0.1:8787', tmp_path)
    assert not (tmp_path/local.CONFIG_NAME).exists()
    assert not (tmp_path/'.data').exists()


@pytest.mark.parametrize('failure', ['key-command', 'interrupt', 'apps-write', 'config-commit'])
def test_first_initialization_can_retry_without_changing_paired_identity(tmp_path, monkeypatch, failure):
    from types import SimpleNamespace
    monkeypatch.setitem(sys.modules, 'export_portable_agent_config',
                        SimpleNamespace(configuration=lambda *args: {'apps': []}))
    monkeypatch.setenv('PORTABLE_AGENT_PYTHON', sys.executable)
    monkeypatch.setenv('PORTABLE_AGENT_MODEL_ENDPOINTS', 'https://provider.example/v1')
    directory = tmp_path/'.data/portable-agent'
    private_write = local.private_json
    attempt = {'fail': True}
    def installed_wheel(args, **kwargs):
        if 'issuer-key' not in args:
            return SimpleNamespace(returncode=0, stdout='{"protocol":2,"version":"0.2.0"}')
        assert local.read_config(tmp_path, allow_initializing=True)['initializing'] is True
        key_dir = Path(args[args.index('--data-dir')+1])
        master = key_dir/'master.key'
        if not master.exists():
            master.write_bytes(b'original-master-key')
        if attempt['fail'] and failure == 'interrupt':
            raise KeyboardInterrupt()
        return SimpleNamespace(returncode=1 if attempt['fail'] and failure == 'key-command' else 0, stdout='a'*64)
    def guarded_write(path, value):
        if attempt['fail'] and (failure == 'apps-write' and path.name == 'portable-agent-apps.json'
                or failure == 'config-commit' and path.name == 'portable-agent-local.json' and 'issuer_key' in value):
            raise OSError('injected disk failure')
        return private_write(path, value)
    monkeypatch.setattr(local.subprocess, 'run', installed_wheel)
    monkeypatch.setattr(local, 'private_json', guarded_write)
    with pytest.raises((local.LocalAgentError, OSError, KeyboardInterrupt)):
        local.prepare('http://127.0.0.1:8000', 'http://127.0.0.1:5173', 'http://127.0.0.1:8787', tmp_path)
    pending = local.read_config(tmp_path, allow_initializing=True)
    with pytest.raises(local.LocalAgentError, match='初始化未完成'):
        local.read_config(tmp_path)
    if (directory/'master.key').exists():
        assert (directory/'master.key').read_bytes() == b'original-master-key'
    attempt['fail'] = False
    monkeypatch.setattr(local.getpass, 'getuser', lambda: pytest.fail('Must reuse persisted owner'))
    final = local.prepare('http://127.0.0.1:8000', 'http://127.0.0.1:5173', 'http://127.0.0.1:8787', tmp_path)
    for field in ('owner', 'workspace', 'identity_key', 'service_token'):
        assert final[field] == pending[field]
    assert final['issuer_key'] == 'a'*64 and 'initializing' not in final
    assert (directory/'master.key').read_bytes() == b'original-master-key'
    assert local.read_config(tmp_path) == final


@pytest.mark.parametrize('failure', ['identity-directory', 'master-file', 'master-directory'])
def test_durability_failure_stops_setup_and_retry_keeps_identity(tmp_path, monkeypatch, failure):
    from types import SimpleNamespace
    monkeypatch.setitem(sys.modules, 'export_portable_agent_config',
                        SimpleNamespace(configuration=lambda *args: {'apps': []}))
    monkeypatch.setenv('PORTABLE_AGENT_PYTHON', sys.executable)
    monkeypatch.setenv('PORTABLE_AGENT_MODEL_ENDPOINTS', 'https://provider.example/v1')
    data = tmp_path/'.data/portable-agent'
    original_directory_sync = local.sync_directory
    original_file_sync = local.sync_file
    attempt = {'fail': True, 'keys': 0}
    def directory_sync(path):
        if attempt['fail'] and (failure == 'identity-directory' and path == tmp_path/'.run'
                or failure == 'master-directory' and path == data and (data/'master.key').exists()):
            raise OSError('injected directory fsync failure')
        original_directory_sync(path)
    def file_sync(path):
        if attempt['fail'] and failure == 'master-file':
            raise OSError('injected key fsync failure')
        original_file_sync(path)
    def installed_wheel(args, **kwargs):
        if 'issuer-key' in args:
            attempt['keys'] += 1
            pending = local.read_config(tmp_path, allow_initializing=True)
            assert pending['initializing'] is True
            key = Path(args[args.index('--data-dir')+1])/'master.key'
            if not key.exists():
                key.write_bytes(b'preserved-master')
            return SimpleNamespace(returncode=0, stdout='a'*64)
        return SimpleNamespace(returncode=0, stdout='{"protocol":2,"version":"0.2.0"}')
    monkeypatch.setattr(local, 'sync_directory', directory_sync)
    monkeypatch.setattr(local, 'sync_file', file_sync)
    monkeypatch.setattr(local.subprocess, 'run', installed_wheel)
    with pytest.raises(OSError):
        local.prepare('http://127.0.0.1:8000', 'http://127.0.0.1:5173', 'http://127.0.0.1:8787', tmp_path)
    pending = local.read_config(tmp_path, allow_initializing=True)
    if failure == 'identity-directory':
        assert attempt['keys'] == 0
    if failure != 'master-directory':
        assert not (data/'master.key').exists()
    attempt['fail'] = False
    final = local.prepare('http://127.0.0.1:8000', 'http://127.0.0.1:5173', 'http://127.0.0.1:8787', tmp_path)
    assert final['identity_key'] == pending['identity_key']
    assert final['service_token'] == pending['service_token']
    assert (data/'master.key').read_bytes() == b'preserved-master'
    assert local.read_config(tmp_path) == final


@pytest.mark.parametrize('flag', ['true', 1, None])
def test_initialization_marker_cannot_bypass_normal_config_validation(config, flag):
    root, value = config
    value['initializing'] = flag
    local.private_json(root/local.CONFIG_NAME, value)
    with pytest.raises(local.LocalAgentError, match='配置不完整'):
        local.read_config(root, allow_initializing=True)


def test_retry_ignores_incomplete_scratch_key_without_replacing_data(config, monkeypatch):
    from types import SimpleNamespace
    root, value = config
    value.pop('issuer_key')
    value['initializing'] = True
    local.private_json(root/local.CONFIG_NAME, value)
    directory = root/'.data/portable-agent'
    orphan = directory/'.initial-key-interrupted/master.key'
    orphan.parent.mkdir(parents=True)
    orphan.write_bytes(b'partial-key-from-killed-process')
    monkeypatch.setitem(sys.modules, 'export_portable_agent_config',
                        SimpleNamespace(configuration=lambda *args: {'apps': []}))
    def installed_wheel(args, **kwargs):
        if 'issuer-key' in args:
            master = Path(args[args.index('--data-dir')+1])/'master.key'
            assert master != orphan and master.parent != directory
            master.write_bytes(b'valid-new-initial-key')
            return SimpleNamespace(returncode=0, stdout='a'*64)
        return SimpleNamespace(returncode=0, stdout='{"protocol":2,"version":"0.2.0"}')
    monkeypatch.setattr(local.subprocess, 'run', installed_wheel)
    final = local.prepare(value['backend_url'], value['browser_origin'], value['agent_url'], root)
    assert final['identity_key'] == value['identity_key']
    assert orphan.read_bytes() == b'partial-key-from-killed-process'
    assert (directory/'master.key').read_bytes() == b'valid-new-initial-key'


def test_initial_identity_write_failure_does_not_create_an_orphan_master(tmp_path, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setitem(sys.modules, 'export_portable_agent_config',
                        SimpleNamespace(configuration=lambda *args: {'apps': []}))
    monkeypatch.setenv('PORTABLE_AGENT_PYTHON', sys.executable)
    monkeypatch.setenv('PORTABLE_AGENT_MODEL_ENDPOINTS', 'https://provider.example/v1')
    def installed_wheel(args, **kwargs):
        assert 'issuer-key' not in args
        return SimpleNamespace(returncode=0, stdout='{"protocol":2,"version":"0.2.0"}')
    def unwritable(*args):
        raise OSError('cannot persist initial identity')
    monkeypatch.setattr(local.subprocess, 'run', installed_wheel)
    monkeypatch.setattr(local, 'private_json', unwritable)
    with pytest.raises(OSError):
        local.prepare('http://127.0.0.1:8000', 'http://127.0.0.1:5173', 'http://127.0.0.1:8787', tmp_path)
    assert not (tmp_path/'.data/portable-agent/master.key').exists()
    assert not (tmp_path/local.CONFIG_NAME).exists()


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

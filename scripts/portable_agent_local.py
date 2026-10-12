"""Local lifecycle configuration; the agent remains an independently installed wheel.

Never print credentials or import the agent into the platform interpreter.
"""
from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import re
from pathlib import Path
import secrets
import subprocess
import sys
import tempfile
from urllib.parse import urlsplit

import httpx
import jwt

ROOT = Path(__file__).resolve().parents[1]
CONFIG_NAME = '.run/portable-agent-local.json'


class LocalAgentError(RuntimeError):
    pass


class LocalAgentAuthMismatch(LocalAgentError):
    """Positive evidence that the owned host needs the current paired config."""
    pass


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def sync_file(path):
    with path.open('rb') as handle:
        os.fsync(handle.fileno())


def private_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise LocalAgentError('本地助手配置不能是软链接。')
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix='.agent-')
    try:
        with os.fdopen(fd, 'w') as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        sync_directory(path.parent)
        sync_directory(path.parent.parent)
    finally:
        Path(temporary).unlink(missing_ok=True)


def local_url(value):
    if not isinstance(value, str) or any(c.isspace() or ord(c) < 32 or ord(c) == 127 for c in value):
        raise LocalAgentError('本地服务地址不能包含空白或控制字符，请检查端口设置。')
    parsed = urlsplit(value)
    if (parsed.scheme != 'http' or parsed.hostname not in {'127.0.0.1', 'localhost', '::1'}
            or parsed.username is not None or parsed.password is not None or parsed.path not in {'', '/'}
            or '?' in value or '#' in value):
        raise LocalAgentError('本地助手仅支持明确的本机 HTTP 地址；公共部署请使用独立部署配置。')
    try:
        port = parsed.port
        if port is not None and not 1 <= port <= 65535:
            raise ValueError()
    except ValueError as exc:
        raise LocalAgentError('本地服务端口必须在 1–65535 之间。') from exc
    host = f'[{parsed.hostname}]' if ':' in parsed.hostname else parsed.hostname
    authority = host if port in {None, 80} else f'{host}:{port}'
    return 'http://'+authority


def model_endpoint(value):
    # Mirror the independently installed agent's HTTP(S) endpoint contract.
    try:
        if not isinstance(value, str) or any(c.isspace() or ord(c) < 32 or ord(c) == 127 for c in value):
            raise ValueError()
        parsed = urlsplit(value)
        if (parsed.scheme not in {'http', 'https'} or not parsed.hostname
                or parsed.username or parsed.password or parsed.query or parsed.fragment):
            raise ValueError()
        port = parsed.port
        if port is not None and not 1 <= port <= 65535:
            raise ValueError()
        if ':' not in parsed.hostname and not re.fullmatch(r'[A-Za-z0-9_.-]+', parsed.hostname.encode('idna').decode('ascii')):
            raise ValueError()
    except (ValueError, TypeError, AttributeError, UnicodeError):
        raise LocalAgentError('模型 API 地址无效；请填写完整的 HTTP(S) 地址，不含账号、密码、查询参数或片段。') from None
    return value.rstrip('/')


def read_config(root=ROOT, *, allow_initializing=False):
    path = Path(root) / CONFIG_NAME
    if path.is_symlink() or not path.is_file():
        raise LocalAgentError('助手尚未初始化，请运行 ./start_services.sh start。')
    if path.stat().st_mode & 0o077:
        raise LocalAgentError('助手配置包含服务凭证，请将权限设置为 600。')
    try:
        value = json.loads(path.read_text())
        if value['schema_version'] != 1:
            raise ValueError()
        for key in ('backend_url', 'browser_origin', 'agent_url'):
            local_url(value[key])
        if any(not isinstance(value[k], str) or len(value[k]) < 32
               for k in ('identity_key', 'service_token')):
            raise ValueError()
        if not value['owner'] or not value['workspace'] or not Path(value['python']).is_absolute():
            raise ValueError()
        if 'initializing' in value and type(value['initializing']) is not bool:
            raise ValueError()
        if value.get('initializing') is True:
            if 'issuer_key' in value:
                raise ValueError()
            if not allow_initializing:
                raise LocalAgentError('助手初始化未完成，请重新运行 ./start_services.sh start；会保留已记录的本机身份。')
        elif not isinstance(value.get('issuer_key'), str) or len(value['issuer_key']) < 32:
            raise ValueError()
        return value
    except (ValueError, KeyError, TypeError) as exc:
        raise LocalAgentError('助手本地配置不完整；保留原密钥与数据，检查配置后重试。') from exc


def agent_environment():
    # Only process/locale/TLS basics cross the independent-runtime boundary.
    allowed = {'PATH', 'HOME', 'USER', 'LANG', 'LC_ALL', 'LC_CTYPE', 'TZ',
               'TMPDIR', 'TMP', 'TEMP', 'SYSTEMROOT', 'SSL_CERT_FILE', 'SSL_CERT_DIR'}
    env = {key: value for key, value in os.environ.items() if key in allowed}
    env.setdefault('PATH', os.defpath)
    env['NO_PROXY'] = '127.0.0.1,localhost,::1'
    return env


def prepare(backend_url, browser_origin, agent_url, root=ROOT):
    root = Path(os.path.abspath(root))
    urls = {key: local_url(value) for key, value in
            [('backend_url', backend_url), ('browser_origin', browser_origin), ('agent_url', agent_url)]}
    path = root / CONFIG_NAME
    previous = read_config(root, allow_initializing=True) if path.exists() or path.is_symlink() else None
    python = (previous or {}).get('python') or os.environ.get('PORTABLE_AGENT_PYTHON') or str(root/'.venv/portable-agent/bin/python')
    # Keep the venv's executable link; resolving it can select system Python.
    python = os.path.abspath(root / python)
    if not Path(python).is_file():
        raise LocalAgentError('缺少独立助手运行环境。请先安装 v2 wheel，设置 PORTABLE_AGENT_PYTHON 后重试。')
    # Probe from the private runtime directory, never from a source checkout.
    directory = root/'.data/portable-agent'
    if directory.is_symlink() or (directory/'master.key').is_symlink():
        raise LocalAgentError('助手持久目录及主密钥不能是软链接。')
    if not previous and directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
        raise LocalAgentError('已有助手数据，但原本机身份配置缺失；请恢复配对的 .run/portable-agent-local.json，不会创建替代身份。')
    if previous and previous.get('initializing') is not True and not (directory/'master.key').is_file():
        raise LocalAgentError('原助手主密钥缺失；请恢复备份，启动不会生成替代密钥。')
    endpoints = list((previous or {}).get('model_endpoints', []))
    environment_endpoints = os.environ.get('PORTABLE_AGENT_MODEL_ENDPOINTS', '')
    if environment_endpoints:
        endpoints += environment_endpoints.split(',')
    legacy = root/'data/llm_settings.json'
    if legacy.is_file():
        old = json.loads(legacy.read_text())
        profiles = old.get('profiles') or ([old['settings']] if old.get('settings') else [])
        endpoints += [p['base_url'].rstrip('/') for p in profiles if p.get('base_url')]
    endpoints = list(dict.fromkeys(model_endpoint(item) for item in endpoints))
    if not endpoints:
        raise LocalAgentError('尚未指定模型 API 地址，请在本地配置的 model_endpoints 中填写允许接入的地址。')
    sys.path.insert(0, str(ROOT/'scripts'))
    from export_portable_agent_config import configuration
    apps = configuration(urls['backend_url'], urls['browser_origin'], endpoints)
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    sync_directory(directory.parent)
    sync_directory(root)
    probe = subprocess.run([python, '-c',
        'import json, importlib.metadata; from pathlib import Path; import portable_agent; '
        'p=Path(portable_agent.__file__).parent/"web/manifest.json"; '
        'm=json.loads(p.read_text()); print(json.dumps({"protocol":m["protocol_major"],'
        '"version":importlib.metadata.version("portable-web-agent")}))'],
        cwd=directory, env=agent_environment(), capture_output=True, text=True)
    try:
        if probe.returncode or json.loads(probe.stdout)['protocol'] != 2:
            raise ValueError()
    except (ValueError, KeyError):
        raise LocalAgentError('助手运行环境不是完整的 v2 安装包（需包含网页 manifest）。') from None
    value = previous or dict(schema_version=1, python=python, owner=getpass.getuser(), workspace='local',
                             identity_key=secrets.token_hex(32), service_token=secrets.token_hex(32), initializing=True)
    value.update(urls)
    value['model_endpoints'] = endpoints
    if value.get('initializing') is True:
        # Persist the paired identity before any key can exist; retries reuse it.
        private_json(path, value)
    if value.get('initializing') is True and not (directory/'master.key').exists():
        # A killed key command can only leave an incomplete scratch key. Publish
        # the validated key atomically without ever replacing an existing key.
        with tempfile.TemporaryDirectory(dir=directory, prefix='.initial-key-') as staging:
            key = issuer_key(python, Path(staging))
            sync_file(Path(staging)/'master.key')
            os.link(Path(staging)/'master.key', directory/'master.key')
    else:
        key = issuer_key(python, directory)
        sync_file(directory/'master.key')
    sync_directory(directory)
    if value.get('initializing') is not True and value['issuer_key'] != key:
        raise LocalAgentError('助手数据目录与原签名密钥不一致；请恢复原目录，不要重置密钥。')
    value['issuer_key'] = key
    value.pop('initializing', None)
    private_json(root/'.run/portable-agent-apps.json', apps)
    private_json(path, value)
    return value


def issuer_key(python, directory):
    issuer = subprocess.run([python, '-m', 'portable_agent', 'issuer-key', '--data-dir', str(directory),
                             '--app', 'fund-research'], cwd=directory, env=agent_environment(),
                            capture_output=True, text=True)
    key = issuer.stdout.strip()
    if issuer.returncode or len(key) != 64 or any(c not in '0123456789abcdef' for c in key):
        raise LocalAgentError('无法读取助手应用签名密钥；不会重置现有数据。')
    return key


def service_environment(root, service):
    value = read_config(root)
    env = dict(os.environ)
    if service == 'backend':
        env.update(PORTABLE_AGENT_AUTH_MODE='local', PORTABLE_AGENT_BROWSER_ORIGIN=value['browser_origin'],
                   PORTABLE_AGENT_LOCAL_OWNER=value['owner'], PORTABLE_AGENT_WORKSPACE=value['workspace'],
                   PORTABLE_AGENT_LOCAL_WORKSPACE=value['workspace'], PORTABLE_AGENT_APP='fund-research',
                   PORTABLE_AGENT_IDENTITY_ISSUER='fund-research',
                   PORTABLE_AGENT_IDENTITY_KEY=value['identity_key'], PORTABLE_AGENT_ISSUER_KEY=value['issuer_key'],
                   PORTABLE_AGENT_SERVICE_TOKEN=value['service_token'])
    elif service == 'agent':
        env = agent_environment()
        env['PORTABLE_AGENT_HOST_AUTHORIZATION'] = 'Bearer '+value['service_token']
    elif service == 'frontend':
        env['VITE_API_TARGET'] = value['backend_url']
        env['VITE_PORTABLE_AGENT_TARGET'] = value['agent_url']
    return env


def agent_command(root=ROOT):
    root = Path(root)
    value = read_config(root)
    parsed = urlsplit(value['agent_url'])
    return [value['python'], '-m', 'portable_agent', 'serve', '--config', str(root/'.run/portable-agent-apps.json'),
            '--data-dir', str(root/'.data/portable-agent'), '--host', parsed.hostname, '--port', str(parsed.port or 80)]


def checked(response, label, *, auth_probe=False):
    if not response.is_success:
        if auth_probe:
            try:
                code = response.json()['error']['code']
            except (ValueError, KeyError, TypeError):
                code = None
            # Only stable host errors prove configuration incompatibility.
            if ((label == '本机登录' and response.status_code == 404 and code == 'NOT_FOUND')
                    or (label == '后端服务凭证' and response.status_code == 401 and code == 'SERVICE_UNAUTHORIZED')
                    or (response.status_code == 403 and code == 'ORIGIN_FORBIDDEN')
                    or (response.status_code == 503 and code == 'ASSISTANT_NOT_CONFIGURED')):
                raise LocalAgentAuthMismatch('运行中的后端鉴权配置与当前助手接入不兼容。')
        # Only expose the status/step, never a provider body that may contain credentials.
        raise LocalAgentError(f'{label}失败（HTTP {response.status_code}）；请检查数据卷、接入配置与服务日志。')
    return response.json()


def connect(client, value, host_url, *, auth_probe=False):
    base = host_url+'/api/integrations/portable-agent'
    checked(client.post(base+'/local-session', headers={'X-Portable-Local': '1'}), '本机登录', auth_probe=auth_probe)
    try:
        context = checked(client.post(base+'/contexts', json={'page_context': {
            'page': 'platform-agent', 'page_instance_id': 'service-readiness', 'context_revision': 0,
            'view_state': 'inherit', 'calculation': {'context_kind': 'platform'}}}), '助手上下文', auth_probe=auth_probe)
        bootstrap = checked(client.post(base+'/bootstrap', json={'context_ref': context['ref']}), '助手授权', auth_probe=auth_probe)
        return {'Authorization': 'Bearer '+bootstrap['token']}, bootstrap
    except (ValueError, KeyError, TypeError):
        raise LocalAgentError('后端助手授权响应尚未完整，请重试接入检查。') from None


def check_host(root=ROOT):
    """Check the running host against the current paired credentials before reuse."""
    value = read_config(root)
    with httpx.Client(trust_env=False, timeout=10, headers={'Origin': value['browser_origin']}) as client:
        _, bootstrap = connect(client, value, value['backend_url'], auth_probe=True)
        try:
            identity = jwt.decode(client.cookies.get('research_identity'), value['identity_key'],
                algorithms=['HS256'], audience='fund-research-platform', issuer='fund-research',
                options={'require': ['sub', 'workspace', 'iat', 'exp']})
            grant = jwt.decode(bootstrap['token'], bytes.fromhex(value['issuer_key']),
                algorithms=['HS256'], audience='portable-agent', issuer='portable-agent-host',
                options={'require': ['sub', 'workspace', 'app', 'iat', 'exp']})
            if any(claims['sub'] != value['owner'] or claims['workspace'] != value['workspace']
                   for claims in (identity, grant)) or grant['app'] != 'fund-research':
                raise LocalAgentAuthMismatch('运行中的后端主体、工作空间或应用与当前助手配置不一致。')
            context = {key: bootstrap['context'][key]
                       for key in ('ref', 'hash', 'grant_id', 'grant_revision', 'workspace')}
        except (jwt.InvalidSignatureError, jwt.InvalidIssuerError, jwt.InvalidAudienceError):
            raise LocalAgentAuthMismatch('运行中的后端身份或应用签名与当前助手配置不一致。') from None
        except (jwt.PyJWTError, ValueError, KeyError, TypeError):
            raise LocalAgentError('尚未取得有效的后端助手身份，请重试接入检查。') from None
        allowed = checked(client.post(value['backend_url']+'/internal/portable-agent/authorize',
            headers={'Authorization': 'Bearer '+value['service_token']}, json={
                'application': 'fund-research', 'subject': value['owner'], 'context': context,
                'action': 'metadata', 'identity_only': True}), '后端服务凭证', auth_probe=True)
        if not isinstance(allowed, dict) or allowed.get('allow') is not True:
            raise LocalAgentError('运行中的后端尚未允许助手接入。')

def check(root=ROOT, *, via_frontend=False, require_model=True):
    value = read_config(root)
    host_url = value['browser_origin'] if via_frontend else value['backend_url']
    agent_url = host_url+'/assistant' if via_frontend else value['agent_url']
    with httpx.Client(trust_env=False, timeout=10, headers={'Origin': value['browser_origin']}) as client:
        headers, bootstrap = connect(client, value, host_url)
        meta = checked(client.get(agent_url+'/v2/apps/fund-research', headers=headers), '助手 v2 接口')
        if meta.get('protocol_major') != 2 or not set(bootstrap['required_capabilities']) <= set(meta.get('capabilities', [])):
            raise LocalAgentError('助手协议或能力与网页不兼容。')
        widget = client.get(agent_url+'/widget/widget.js')
        if widget.status_code != 200 or 'javascript' not in widget.headers.get('content-type', ''):
            raise LocalAgentError('助手网页组件不可访问。')
        profiles = checked(client.get(agent_url+'/v2/apps/fund-research/model-profiles', headers=headers), '模型配置')
        selected = next((p for p in profiles.get('items', []) if p['id'] == profiles.get('active_profile_id')), None)
        if require_model and (not selected or not selected.get('has_key')):
            raise LocalAgentError('助手已连接，但尚未选择模型；请在网页模型设置中完成配置。')
    return meta


def import_profiles(root=ROOT):
    """Explicit one-time copy over the authenticated API; leave legacy data intact."""
    root = Path(root)
    value = read_config(root)
    if value.get('profiles_initialized'):
        return 0
    source_path = root/'data/llm_settings.json'
    old = json.loads(source_path.read_text()) if source_path.is_file() else {}
    source = old.get('profiles') or ([{'id': 'default', 'name': 'Default', **old['settings']}] if old.get('settings') else [])
    with httpx.Client(trust_env=False, timeout=15, headers={'Origin': value['browser_origin']}) as client:
        headers, _ = connect(client, value, value['backend_url'])
        base = value['agent_url']+'/v2/apps/fund-research/model-profiles'
        existing = checked(client.get(base, headers=headers), '读取模型配置')
        import_ids = {'legacy-'+hashlib.sha256(p['id'].encode()).hexdigest()[:32] for p in source}
        if any(item['id'] not in import_ids for item in existing.get('items', [])):
            value['profiles_initialized'] = True
            private_json(root/CONFIG_NAME, value)
            return 0
        active = None
        for profile in source:
            pid = 'legacy-'+hashlib.sha256(profile['id'].encode()).hexdigest()[:32]
            record = next((item for item in existing.get('items', []) if item['id'] == pid), None)
            if record is None:
                record = checked(client.post(base, headers=headers, json={
                    'name': profile.get('name', profile['id']), 'base_url': profile['base_url'], 'model': profile['model'],
                    'api_key': profile.get('api_key'), 'provider': profile.get('provider', 'openai-compatible'),
                    'timeout': profile.get('timeout_seconds') or 60, 'reasoning_effort': profile.get('reasoning_effort') or 'default',
                    'context_window_tokens': profile.get('context_window_tokens') or 0,
                    'expected_revision': 0, 'request_id': pid}), '导入模型配置')
            if profile['id'] == old.get('active_profile_id', 'default'):
                active = record['id']
        if active and not existing.get('active_profile_id'):
            checked(client.put(base+'/active', headers=headers, json={'profile_id': active, 'expected_revision': existing.get('revision', 0)}), '选择原模型')
    value['profiles_initialized'] = True
    private_json(root/CONFIG_NAME, value)
    return len(source)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'check', 'check-host', 'import-profiles'])
    parser.add_argument('--backend-url', default='http://127.0.0.1:8000')
    parser.add_argument('--browser-origin', default='http://127.0.0.1:5173')
    parser.add_argument('--agent-url', default='http://127.0.0.1:8787')
    parser.add_argument('--via-frontend', action='store_true')
    parser.add_argument('--allow-unconfigured-model', action='store_true')
    args = parser.parse_args()
    try:
        if args.action == 'prepare':
            prepare(args.backend_url, args.browser_origin, args.agent_url)
            print('[配置] 独立助手 v2、本机身份与服务凭证已就绪。')
        elif args.action == 'check-host':
            check_host()
            print('[就绪] 后端助手身份、签名与服务凭证匹配。')
        elif args.action == 'import-profiles':
            count = import_profiles()
            print(f'[配置] 模型接入已初始化（本次导入 {count} 项）；保留既有助手设置。')
        else:
            check(via_frontend=args.via_frontend, require_model=not args.allow_unconfigured_model)
            print('[就绪] 助手鉴权、v2 接口和网页组件正常。'+('模型选择正常。' if not args.allow_unconfigured_model else ''))
    except (LocalAgentError, OSError, ValueError, httpx.HTTPError) as exc:
        message = str(exc) if isinstance(exc, LocalAgentError) else '助手接入检查失败，请检查私有配置和服务日志。'
        print('[助手未就绪] '+message, file=sys.stderr)
        return 3 if args.action == 'check-host' and isinstance(exc, LocalAgentAuthMismatch) else 1
    return 0


if __name__ == '__main__':
    sys.exit(main())

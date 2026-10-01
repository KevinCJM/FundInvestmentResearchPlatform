"""Authenticate a platform user or the configured agent service; issue scoped grants only."""
import hmac
import ipaddress
import os
import time
from urllib.parse import urlsplit

import jwt

from .contracts import ResearchError


def required(name, minimum=1):
    value = os.environ.get(name, '')
    if len(value) < minimum:
        raise ResearchError('ASSISTANT_NOT_CONFIGURED', f'接入服务尚未配置 {name}。', status_code=503)
    return value


def application_id():
    return os.environ.get('PORTABLE_AGENT_APP', 'fund-research')


def allowed_origin(request):
    origin = request.headers.get('origin')
    allowed = os.environ.get('PORTABLE_AGENT_BROWSER_ORIGIN', str(request.base_url).rstrip('/'))
    if origin and origin != allowed:
        raise ResearchError('ORIGIN_FORBIDDEN', '该网页来源未获准接入。', status_code=403)
    if request.headers.get('sec-fetch-site') == 'cross-site':
        raise ResearchError('ORIGIN_FORBIDDEN', '不接受跨站身份请求。', status_code=403)


def user(request):
    allowed_origin(request)
    mode = os.environ.get('PORTABLE_AGENT_AUTH_MODE', 'disabled')
    if mode not in {'local', 'jwt'}:
        raise ResearchError('ASSISTANT_NOT_CONFIGURED', '请先配置智能体接入身份。', status_code=503)
    authorization = request.headers.get('authorization', '')
    token = authorization[7:] if authorization.startswith('Bearer ') else request.cookies.get('research_identity', '')
    if not token or len(token) > 8192:
        raise ResearchError('UNAUTHORIZED', '请先建立投研平台登录会话。', status_code=401)
    try:
        value = jwt.decode(token, required('PORTABLE_AGENT_IDENTITY_KEY', 32), algorithms=['HS256'],
            audience='fund-research-platform', issuer=os.environ.get('PORTABLE_AGENT_IDENTITY_ISSUER', 'fund-research'),
            options={'require': ['sub', 'workspace', 'scopes', 'iat', 'exp']})
        if (not isinstance(value['sub'], str) or not 1 <= len(value['sub']) <= 200
                or not isinstance(value['workspace'], str) or not 1 <= len(value['workspace']) <= 100
                or not isinstance(value['scopes'], list) or len(value['scopes']) > 100
                or not all(isinstance(s, str) and 0 < len(s) <= 100 for s in value['scopes'])
                or type(value['exp']) is not int or type(value['iat']) is not int or value['exp']-value['iat'] > 86400):
            raise ValueError()
        workspace = os.environ.get('PORTABLE_AGENT_LOCAL_WORKSPACE', 'local') if mode == 'local' else required('PORTABLE_AGENT_WORKSPACE')
        if value['workspace'] != workspace:
            raise ValueError()
        return value
    except (jwt.PyJWTError, ValueError, TypeError, KeyError):
        raise ResearchError('UNAUTHORIZED', '登录会话无效或已过期。', status_code=401) from None


def local_session(request):
    if os.environ.get('PORTABLE_AGENT_AUTH_MODE') != 'local':
        raise ResearchError('NOT_FOUND', '本机登录未启用。', status_code=404)
    allowed_origin(request)
    try:
        local = ipaddress.ip_address(request.client.host).is_loopback
    except (ValueError, AttributeError):
        local = False
    host = urlsplit(str(request.base_url)).hostname
    origin_host = urlsplit(request.headers.get('origin', '')).hostname
    if not local or host not in {'127.0.0.1', 'localhost', '::1'} or origin_host not in {'127.0.0.1', 'localhost', '::1'} or request.headers.get('x-portable-local') != '1':
        raise ResearchError('LOCAL_ONLY', '本机会话只允许同源回环页面建立。', status_code=403)
    now = int(time.time())
    claims = {'sub': required('PORTABLE_AGENT_LOCAL_OWNER'), 'workspace': os.environ.get('PORTABLE_AGENT_LOCAL_WORKSPACE', 'local'),
              'scopes': ['*'], 'iat': now, 'exp': now+3600, 'aud': 'fund-research-platform',
              'iss': os.environ.get('PORTABLE_AGENT_IDENTITY_ISSUER', 'fund-research')}
    return jwt.encode(claims, required('PORTABLE_AGENT_IDENTITY_KEY', 32), algorithm='HS256')


def service(request):
    expected = required('PORTABLE_AGENT_SERVICE_TOKEN', 32)
    if not hmac.compare_digest(request.headers.get('authorization', ''), 'Bearer '+expected):
        raise ResearchError('SERVICE_UNAUTHORIZED', '服务凭证无效。', status_code=401)


def permit(principal, scope):
    if '*' not in principal['scopes'] and scope not in principal['scopes']:
        raise ResearchError('FORBIDDEN', '当前身份没有此业务权限。', status_code=403)


def agent_token(principal, context, scopes):
    now = int(time.time())
    try:
        key = bytes.fromhex(required('PORTABLE_AGENT_ISSUER_KEY', 64))
        if len(key) != 32:
            raise ValueError()
    except ValueError:
        raise ResearchError('ASSISTANT_NOT_CONFIGURED', '应用签名配置无效。', status_code=503) from None
    return jwt.encode({'app': application_id(), 'sub': principal['sub'], 'scopes': scopes,
        'workspace': context['workspace'], 'context_ref': context['id'], 'context_hash': context['hash'],
        'grant_id': context['grant_id'], 'grant_revision': context['grant_revision'], 'iat': now, 'exp': now+900,
        'aud': 'portable-agent', 'iss': 'portable-agent-host'}, key, algorithm='HS256')

"""Run isolated host and agent HTTP processes for the portable browser contract.

PORTABLE_AGENT_PYTHON selects an independently installed framework environment.
PORTABLE_AGENT_SOURCE optionally selects an explicit development source tree.
No framework package is imported by this process or by the platform.
"""
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import tempfile
import time

import httpx

from export_portable_agent_config import configuration

ROOT = Path(__file__).resolve().parents[1]


def main():
    agent_python = os.environ.get('PORTABLE_AGENT_PYTHON')
    if not agent_python:
        raise SystemExit('Set PORTABLE_AGENT_PYTHON to the independent agent environment.')
    host_port = int(os.environ.get('PORTABLE_TEST_HOST_PORT', '18080'))
    agent_port = int(os.environ.get('PORTABLE_TEST_AGENT_PORT', '18787'))
    origin = os.environ.get('PORTABLE_TEST_ORIGIN', 'http://127.0.0.1:14173')
    host_url, agent_url = f'http://127.0.0.1:{host_port}', f'http://127.0.0.1:{agent_port}'
    children = []
    def stop(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    with tempfile.TemporaryDirectory(prefix='portable-integration-') as folder:
        directory = Path(folder)
        common = {**os.environ, 'ALL_PROXY': '', 'HTTP_PROXY': '', 'HTTPS_PROXY': '',
                  'NO_PROXY': '127.0.0.1,localhost', 'PYTHONDONTWRITEBYTECODE': '1'}
        agent_env = {**common, 'PYTHONPATH': os.environ.get('PORTABLE_AGENT_SOURCE', '')}
        issuer = subprocess.check_output([agent_python, '-m', 'portable_agent', 'issuer-key',
            '--data-dir', str(directory/'agent'), '--app', 'fund-research'], env=agent_env, text=True).strip()
        service_token = secrets.token_hex(32)
        host_env = {**common, 'PYTHONPATH': os.pathsep.join(map(str, [ROOT, ROOT/'backend', ROOT/'backend/tests'])),
            'CUSTOM_INDICATOR_DATA_DIR': str(directory/'business'), 'INDICATOR_PROCESS_WORKERS': '1',
            'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'NUMBA_NUM_THREADS': '1',
            'PORTABLE_AGENT_AUTH_MODE': 'local', 'PORTABLE_AGENT_LOCAL_OWNER': 'offline-browser',
            'PORTABLE_AGENT_IDENTITY_KEY': secrets.token_hex(32), 'PORTABLE_AGENT_SERVICE_TOKEN': service_token,
            'PORTABLE_AGENT_ISSUER_KEY': issuer, 'PORTABLE_AGENT_BROWSER_ORIGIN': origin,
            'PORTABLE_TEST_READY_FILE': str(directory/'ready')}
        agent_env['PORTABLE_AGENT_HOST_AUTHORIZATION'] = 'Bearer '+service_token
        config = directory/'apps.json'
        config.write_text(json.dumps(configuration(host_url, origin, [host_url+'/fixture/v1'])))
        try:
            children.append(subprocess.Popen([sys.executable, '-m', 'uvicorn', 'portable_integration_app:app',
                '--host', '127.0.0.1', '--port', str(host_port), '--no-access-log', '--no-proxy-headers'], cwd=ROOT, env=host_env))
            children.append(subprocess.Popen([agent_python, '-m', 'portable_agent', 'serve', '--config', str(config),
                '--data-dir', str(directory/'agent'), '--port', str(agent_port)], cwd=directory, env=agent_env))
            with httpx.Client(trust_env=False, timeout=3, headers={'Origin': origin}) as client:
                for url in [host_url+'/ready', agent_url+'/health']:
                    deadline = time.monotonic()+120
                    while True:
                        if any(p.poll() is not None for p in children):
                            raise RuntimeError('Integration process exited before readiness.')
                        try:
                            if client.get(url).status_code == 200:
                                break
                        except httpx.HTTPError:
                            pass
                        if time.monotonic() >= deadline:
                            raise RuntimeError('Integration readiness timed out.')
                        time.sleep(.2)
                def checked(response):
                    response.raise_for_status()
                    return response.json()
                checked(client.post(host_url+'/api/integrations/portable-agent/local-session', headers={'X-Portable-Local': '1'}))
                context = checked(client.post(host_url+'/api/integrations/portable-agent/contexts', json={'page_context': {
                    'page': 'platform-agent', 'page_instance_id': 'fixture-config', 'context_revision': 0, 'view_state': 'unknown',
                    'calculation': {'context_kind': 'platform'}}}))
                token = checked(client.post(host_url+'/api/integrations/portable-agent/bootstrap', json={'context_ref': context['ref']}))['token']
                base = agent_url+'/v2/apps/fund-research'
                headers = {'Authorization': 'Bearer '+token}
                profile = checked(client.post(base+'/model-profiles', headers=headers, json={
                    'name': '本地固定模型', 'base_url': host_url+'/fixture/v1', 'model': 'offline-fixture',
                    'api_key': 'offline-fixture', 'expected_revision': 0, 'request_id': 'fixture-model'}))
                checked(client.put(base+'/model-profiles/active', headers=headers, json={'profile_id': profile['id'], 'expected_revision': 0}))
            (directory/'ready').touch()
            print('Portable integration ready (isolated data and local model).', flush=True)
            while all(p.poll() is None for p in children):
                time.sleep(.5)
            raise RuntimeError('Integration process exited.')
        except KeyboardInterrupt:
            pass
        finally:
            for child in reversed(children):
                if child.poll() is None:
                    child.terminate()
            for child in children:
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    child.kill(); child.wait()


if __name__ == '__main__':
    main()

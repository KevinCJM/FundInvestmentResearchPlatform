"""Service management tests never inspect or signal the user's running services."""
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('service_process', ROOT / 'scripts/service_process.py')
processes = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(processes)


@pytest.fixture
def owned(tmp_path, monkeypatch):
    (tmp_path / '.run').mkdir()
    (tmp_path / 'backend').mkdir()
    (tmp_path / 'frontend').mkdir()
    record = dict(protocol=1, pid=45678, birth='original-start', project=str(tmp_path),
                  service='backend', cwd=str(tmp_path / 'backend'),
                  argv=[sys.executable, '-m', 'uvicorn', 'app:app'])
    pid_path, identity_path = processes.paths(tmp_path, 'backend')
    pid_path.write_text(str(record['pid']))
    identity_path.write_text(json.dumps(record))
    monkeypatch.setattr(processes, 'process_birth', lambda _: record['birth'])
    monkeypatch.setattr(processes.os, 'getpgid', lambda pid: pid)
    monkeypatch.setattr(processes.os, 'getsid', lambda pid: pid)
    monkeypatch.setattr(processes, 'process_command_and_cwd', lambda _: (record['argv'], record['cwd']))
    signals = []
    monkeypatch.setattr(processes.os, 'kill', lambda pid, sig: signals.append(('pid', pid, sig)))
    monkeypatch.setattr(processes.os, 'killpg', lambda pid, sig: signals.append(('group', pid, sig)))
    monkeypatch.setattr(processes, 'group_alive', lambda _: False)
    return tmp_path, record, signals


def test_stop_owned_group_and_clear_identity(owned):
    root, record, signals = owned
    processes.stop(root, 'backend')
    assert signals == [('group', record['pid'], signal.SIGTERM)]
    assert all(not path.exists() for path in processes.paths(root, 'backend'))


def test_stop_manages_workers_but_not_independent_etl(owned, monkeypatch):
    root, record, signals = owned
    groups = {record['pid']: [record['pid'], 45679, 45680], 56789: [56789, 56790]}
    terminated = []
    monkeypatch.setattr(processes.os, 'killpg', lambda pgid, sig: terminated.extend(groups[pgid]))
    processes.stop(root, 'backend')
    assert terminated == [45678, 45679, 45680]
    assert 56789 not in terminated and 56790 not in terminated
    assert signals == []


def test_stale_identity_is_removed_without_signal(owned, monkeypatch):
    root, _, signals = owned
    monkeypatch.setattr(processes, 'process_birth', lambda _: None)
    monkeypatch.setattr(processes, 'alive', lambda _: False)
    processes.stop(root, 'backend')
    assert signals == []
    assert all(not path.exists() for path in processes.paths(root, 'backend'))


@pytest.mark.parametrize('case', ['reused', 'unreadable_birth', 'wrong_group', 'wrong_session',
                                  'wrong_command', 'wrong_cwd', 'unreadable_command', 'orphan_group'])
def test_uncertain_or_foreign_identity_fails_closed(owned, monkeypatch, case):
    root, record, signals = owned
    if case == 'reused':
        monkeypatch.setattr(processes, 'process_birth', lambda _: 'different-start')
    elif case == 'unreadable_birth':
        monkeypatch.setattr(processes, 'process_birth', lambda _: None)
        monkeypatch.setattr(processes, 'alive', lambda _: True)
    elif case == 'wrong_group':
        monkeypatch.setattr(processes.os, 'getpgid', lambda _: 98765)
    elif case == 'wrong_session':
        monkeypatch.setattr(processes.os, 'getsid', lambda _: 98765)
    elif case == 'wrong_command':
        monkeypatch.setattr(processes, 'process_command_and_cwd', lambda _: (['sleep', '10'], record['cwd']))
    elif case == 'wrong_cwd':
        monkeypatch.setattr(processes, 'process_command_and_cwd', lambda _: (record['argv'], '/unrelated/backend'))
    elif case == 'unreadable_command':
        def denied(_):
            raise processes.OwnershipError('unreadable')
        monkeypatch.setattr(processes, 'process_command_and_cwd', denied)
    else:
        monkeypatch.setattr(processes, 'process_birth', lambda _: None)
        monkeypatch.setattr(processes, 'alive', lambda _: False)
        monkeypatch.setattr(processes, 'group_alive', lambda _: True)
    with pytest.raises(processes.OwnershipError):
        processes.stop(root, 'backend')
    assert signals == []
    assert all(path.exists() for path in processes.paths(root, 'backend'))


@pytest.mark.parametrize('live', [False, True])
def test_legacy_pid_only_never_adopts_a_live_process(owned, monkeypatch, live):
    root, _, signals = owned
    pid_path, identity_path = processes.paths(root, 'backend')
    identity_path.unlink()
    monkeypatch.setattr(processes, 'alive', lambda _: live)
    if live:
        with pytest.raises(processes.OwnershipError, match='旧版'):
            processes.stop(root, 'backend')
        assert pid_path.exists()
    else:
        processes.stop(root, 'backend')
        assert not pid_path.exists()
    assert signals == []


@pytest.mark.parametrize('value', ['123garbage', '-1', '0', '', '1\n2'])
def test_malformed_pid_is_not_coerced_to_another_pid(owned, value):
    root, _, signals = owned
    processes.paths(root, 'backend')[0].write_text(value)
    with pytest.raises(processes.OwnershipError):
        processes.stop(root, 'backend')
    assert signals == []


@pytest.mark.parametrize('field,value', [('project', '/other'), ('service', 'frontend'),
                                       ('pid', 98765), ('birth', None)])
def test_mismatched_record_is_preserved(owned, field, value):
    root, record, signals = owned
    record[field] = value
    processes.paths(root, 'backend')[1].write_text(json.dumps(record))
    with pytest.raises(processes.OwnershipError):
        processes.stop(root, 'backend')
    assert signals == []


def test_identity_is_rechecked_before_signal(owned, monkeypatch):
    root, record, signals = owned
    births = iter([record['birth'], 'reused-between-checks'])
    monkeypatch.setattr(processes, 'process_birth', lambda _: next(births))
    with pytest.raises(processes.OwnershipError):
        processes.stop(root, 'backend')
    assert signals == []


def test_stop_timeout_preserves_record_without_kill_escalation(owned, monkeypatch):
    root, record, signals = owned
    monkeypatch.setattr(processes, 'group_alive', lambda _: True)
    with pytest.raises(processes.OwnershipError, match='超时'):
        processes.stop(root, 'backend', timeout=0)
    assert signals == [('group', record['pid'], signal.SIGTERM)]
    assert all(path.exists() for path in processes.paths(root, 'backend'))


def test_group_observation_failure_is_not_reported_as_stopped(owned, monkeypatch):
    root, _, _ = owned
    def unreadable(_):
        raise processes.OwnershipError('unreadable')
    monkeypatch.setattr(processes, 'group_alive', unreadable)
    with pytest.raises(processes.OwnershipError):
        processes.stop(root, 'backend')
    assert all(path.exists() for path in processes.paths(root, 'backend'))


@pytest.mark.parametrize('command', [['npm', 'run', 'dev', '--port', '5173'],
                                    ['/usr/bin/node', '/usr/local/bin/npm', 'run', 'dev', '--', '--port', '5173'],
                                    ['/usr/bin/node', '/usr/local/lib/npm-cli.js', 'run', 'dev']])
def test_frontend_live_command_survives_npm_process_title_change(command):
    assert processes.command_matches('frontend', command)


@pytest.mark.parametrize('command', [['python', '-m', 'backend.data_sources.etl_runner'],
                                    ['npm', 'install'], ['node', '/other/dev.js']])
def test_other_process_commands_do_not_establish_service_ownership(command):
    assert not processes.command_matches('backend', command)
    assert not processes.command_matches('frontend', command)


@pytest.mark.parametrize('output,expected', [('12 12 S\n13 12 S\n20 20 S\n', True),
                                          ('12 12 Z\n20 20 S\n', False)])
def test_group_scan_ignores_zombies_and_other_sessions(monkeypatch, output, expected):
    monkeypatch.setattr(processes.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=0, stdout=output))
    assert processes.group_alive(12) is expected


@pytest.mark.parametrize('status,output', [(1, ''), (0, ''), (0, 'malformed')])
def test_unreadable_process_table_fails_closed(monkeypatch, status, output):
    monkeypatch.setattr(processes.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=status, stdout=output))
    with pytest.raises(processes.OwnershipError):
        processes.group_alive(12)


def shell_functions(*names):
    source = (ROOT / 'start_services.sh').read_text()
    return '\n'.join(re.search(rf'^{name}\(\) \{{\n.*?^\}}', source, re.M | re.S).group() for name in names)


@pytest.mark.parametrize('service', ['backend', 'frontend'])
def test_foreign_healthy_port_is_not_adopted_or_killed(service):
    script = shell_functions(f'start_{service}') + '''
BACKEND_PORT=8000
FRONTEND_PORT=5173
is_port_listening() { return 0; }
service_is_owned() { return 1; }
backend_is_ready() { echo unexpected-health-check; return 0; }
frontend_is_ready() { echo unexpected-health-check; return 0; }
kill() { echo unexpected-kill; }
launch_detached() { echo unexpected-launch; }
''' + f'\nstart_{service}\n'
    result = subprocess.run(['/bin/bash', '-c', script], capture_output=True, text=True, check=False)
    assert result.returncode == 1
    assert '未验证归属' in result.stdout
    assert 'unexpected' not in result.stdout


@pytest.mark.parametrize('stop_failed,port_busy', [(True, False), (False, True)])
def test_restart_stops_on_uncertain_ownership_or_foreign_listener(stop_failed, port_busy):
    script = shell_functions('stop_services', 'restart_services') + f'''
BACKEND_PID_FILE=backend.pid
FRONTEND_PID_FILE=frontend.pid
BACKEND_PORT=8000
FRONTEND_PORT=5173
stop_process_by_pid_file() {{ return {int(stop_failed)}; }}
is_port_listening() {{ return {int(not port_busy)}; }}
start_services() {{ echo unexpected-start; }}
kill() {{ echo unexpected-kill; }}
sleep() {{ :; }}
restart_services
'''
    result = subprocess.run(['/bin/bash', '-c', script], capture_output=True, text=True, check=False)
    assert result.returncode == 1
    assert '不会继续自动重启' in result.stdout
    assert 'unexpected' not in result.stdout


def test_successful_restart_proceeds_only_after_safe_stop():
    script = shell_functions('stop_services', 'restart_services') + '''
BACKEND_PID_FILE=backend.pid
FRONTEND_PID_FILE=frontend.pid
BACKEND_PORT=8000
FRONTEND_PORT=5173
stop_process_by_pid_file() { echo "stopped $1"; }
is_port_listening() { return 1; }
start_services() { echo restarted; }
sleep() { :; }
restart_services
'''
    result = subprocess.run(['/bin/bash', '-c', script], capture_output=True, text=True, check=False)
    assert result.returncode == 0
    assert result.stdout.index('stopped backend.pid') < result.stdout.index('restarted')
    assert result.stdout.index('stopped frontend.pid') < result.stdout.index('restarted')


def test_launch_records_identity_and_supports_second_launch_after_stop(owned, monkeypatch):
    root, record, signals = owned
    processes.clear(root, 'backend')
    child = SimpleNamespace(pid=record['pid'], poll=lambda: None)
    launches = []
    def popen(argv, **kwargs):
        launches.append((argv, kwargs))
        return child
    monkeypatch.setattr(processes.subprocess, 'Popen', popen)
    for _ in range(2):
        assert processes.launch(root, 'backend', root / '.run/backend.log', record['argv']) == record['pid']
        assert processes.inspect(root, 'backend')['birth'] == record['birth']
        processes.stop(root, 'backend')
    assert len(launches) == 2
    assert all(kwargs['start_new_session'] and kwargs['cwd'] == root / 'backend' for _, kwargs in launches)
    assert signals == [('group', record['pid'], signal.SIGTERM)] * 2


def test_shell_has_no_port_based_signal_path():
    source = (ROOT / 'start_services.sh').read_text()
    assert 'stop_ports_if_busy' not in source
    assert 'kill -TERM' not in source
    assert 'rm -f "$BACKEND_PID_FILE"' not in source
    assert 'rm -f "$FRONTEND_PID_FILE"' not in source


def test_alive_permission_error_is_not_a_stale_pid(monkeypatch):
    def denied(*_):
        raise PermissionError('denied')
    monkeypatch.setattr(os, 'kill', denied)
    with pytest.raises(processes.OwnershipError):
        processes.alive(45678)


@pytest.mark.parametrize('foreign', [False, True])
def test_listener_must_belong_to_verified_service_session(owned, monkeypatch, foreign):
    _, record, signals = owned
    monkeypatch.setattr(processes.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=0, stdout='45679\n'))
    monkeypatch.setattr(processes.os, 'getpgid', lambda pid: 88888 if foreign and pid == 45679 else record['pid'])
    monkeypatch.setattr(processes.os, 'getsid', lambda _: record['pid'])
    if foreign:
        with pytest.raises(processes.OwnershipError, match='其他 session'):
            processes.verify_listener(record, 12345)
    else:
        processes.verify_listener(record, 12345)
    assert signals == []


def test_launchers_do_not_trust_forwarded_headers():
    source = (ROOT / 'start_services.sh').read_text()
    assert '--no-proxy-headers' in source
    assert 'BACKEND_HOST:-127.0.0.1' in source
    assert 'FRONTEND_HOST:-127.0.0.1' in source
    for name in ('backend/run.py', 'backend/app.py'):
        assert 'proxy_headers=False' in (ROOT / name).read_text()

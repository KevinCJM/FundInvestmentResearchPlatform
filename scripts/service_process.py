"""Own only services launched here; ports and bare PID files are never ownership."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import shlex
import subprocess
import sys
import tempfile
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.data_sources.etl_executor import process_birth  # noqa: E402


class OwnershipError(RuntimeError):
    pass


def paths(root, service):
    directory = Path(root) / '.run'
    return directory / f'{service}.pid', directory / f'{service}.identity.json'


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError as exc:
        raise OwnershipError('无法读取进程身份，未发送停止信号。') from exc


def group_alive(pgid):
    """Do not mistake an unreaped zombie for a running service/worker."""
    try:
        result = subprocess.run(
            ['/bin/ps', '-axo', 'pid=,pgid=,stat='],
            capture_output=True, text=True, timeout=2, check=False,
        )
        if result.returncode or not result.stdout.strip():
            raise OwnershipError('无法核验服务进程组，未继续操作。')
        rows = [line.split() for line in result.stdout.splitlines()]
        return any(int(row[1]) == pgid and not row[2].startswith('Z') for row in rows)
    except (OSError, subprocess.SubprocessError, ValueError, IndexError) as exc:
        raise OwnershipError('无法核验服务进程组，未继续操作。') from exc


def process_command_and_cwd(pid):
    try:
        command = subprocess.run(
            ['/bin/ps', '-p', str(pid), '-o', 'command='],
            capture_output=True, text=True, timeout=2, check=False,
        )
        if command.returncode or not command.stdout.strip():
            raise OwnershipError('服务实际命令不可读，未继续操作。')
        proc_cwd = Path(f'/proc/{pid}/cwd')
        if proc_cwd.exists():
            cwd = str(proc_cwd.resolve(strict=True))
        else:
            location = subprocess.run(
                ['lsof', '-a', '-p', str(pid), '-d', 'cwd', '-Fn'],
                capture_output=True, text=True, timeout=2, check=False,
            )
            directories = [line[1:] for line in location.stdout.splitlines() if line.startswith('n')]
            if location.returncode or len(directories) != 1:
                raise OwnershipError('服务实际工作目录不可读，未继续操作。')
            cwd = str(Path(directories[0]).resolve())
        return shlex.split(command.stdout.strip()), cwd
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        raise OwnershipError('服务命令或工作目录不可读，未继续操作。') from exc


def command_matches(service, command):
    if service == 'backend':
        return len(command) >= 4 and command[1:4] == ['-m', 'uvicorn', 'app:app']
    # npm changes its process title after its Node entrypoint has loaded.
    return (command[:3] == ['npm', 'run', 'dev']
            or (len(command) >= 4 and Path(command[1]).name in {'npm', 'npm-cli.js'}
                and command[2:4] == ['run', 'dev']))


def verify(record):
    pid = record['pid']
    birth = process_birth(pid)
    if birth is None:
        if not alive(pid):
            return False
        raise OwnershipError('服务出生标识不可读，未发送停止信号。')
    try:
        matched = (birth == record['birth'] and os.getpgid(pid) == pid
                   and os.getsid(pid) == pid)
    except OSError as exc:
        raise OwnershipError('服务 session 身份不可读，未发送停止信号。') from exc
    if not matched:
        raise OwnershipError('PID 已被复用或不属于已记录的独立服务 session，未发送停止信号。')
    command, cwd = process_command_and_cwd(pid)
    if cwd != record['cwd'] or not command_matches(record['service'], command):
        raise OwnershipError('活进程命令或工作目录不属于本项目服务，未发送停止信号。')
    return True


def verify_listener(record, port):
    try:
        result = subprocess.run(
            ['lsof', '-nP', f'-iTCP:{int(port)}', '-sTCP:LISTEN', '-t'],
            capture_output=True, text=True, timeout=2, check=False,
        )
        if result.returncode or not result.stdout.strip():
            raise OwnershipError('监听进程归属不可读，不能判定服务就绪。')
        listeners = {int(value) for value in result.stdout.split()}
        if any(os.getpgid(pid) != record['pid'] or os.getsid(pid) != record['pid'] for pid in listeners):
            raise OwnershipError('端口由其他 session 监听，不能判定为本项目服务。')
        if not verify(record):
            raise OwnershipError('监听核验期间服务已退出。')
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        raise OwnershipError('监听进程归属不可读，不能判定服务就绪。') from exc


def inspect(root, service):
    pid_path, identity_path = paths(root, service)
    try:
        if pid_path.is_symlink() or identity_path.is_symlink():
            raise OwnershipError('服务身份文件不能是软链接。')
        if not pid_path.exists() and not identity_path.exists():
            return None
        raw = pid_path.read_text().strip()
        if not raw.isdecimal() or int(raw) <= 0:
            raise OwnershipError('PID 文件格式无效，未继续操作。')
        pid = int(raw)
        if not identity_path.exists():
            if alive(pid):
                raise OwnershipError('旧版 PID 文件没有出生标识；请先人工确认并停止旧服务，再重试。')
            return None
        record = json.loads(identity_path.read_text())
        if (not isinstance(record, dict) or record.get('protocol') != 1
                or type(record.get('pid')) is not int or record['pid'] != pid
                or not isinstance(record.get('birth'), str) or not record['birth']
                or record.get('project') != str(Path(root).resolve())
                or record.get('service') != service
                or record.get('cwd') != str((Path(root) / service).resolve())
                or not isinstance(record.get('argv'), list) or not record['argv']):
            raise OwnershipError('服务身份记录不匹配当前项目，未继续操作。')
        if verify(record):
            return record
        if group_alive(pid):
            raise OwnershipError('服务主进程已退出但进程组仍在；归属不能完整核验，请人工检查。')
        return None
    except (OSError, ValueError) as exc:
        raise OwnershipError('服务身份记录不可读，未继续操作。') from exc


def clear(root, service):
    for path in paths(root, service):
        path.unlink(missing_ok=True)


def atomic_write(path, content):
    fd, temporary = tempfile.mkstemp(prefix='.service-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def launch(root, service, log_path, argv):
    if inspect(root, service) is not None:
        raise OwnershipError('已记录的服务仍在运行；请等待就绪或先安全停止。')
    clear(root, service)
    pid_path, identity_path = paths(root, service)
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    process = None
    try:
        with open(log_path, 'ab', buffering=0) as log:
            process = subprocess.Popen(
                argv, cwd=Path(root) / service, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                close_fds=True,
            )
        birth = process_birth(process.pid)
        if birth is None:
            raise OwnershipError('新服务出生标识不可读，启动已撤销。')
        record = dict(protocol=1, pid=process.pid, birth=birth,
                      project=str(Path(root).resolve()), service=service,
                      cwd=str((Path(root) / service).resolve()), argv=list(argv))
        if not verify(record):
            raise OwnershipError('新服务已退出，未记录为运行中。')
        atomic_write(identity_path, json.dumps(record) + '\n')
        # Retain the integer file consumed by the offline storage migration guard.
        atomic_write(pid_path, str(process.pid) + '\n')
        return process.pid
    except BaseException:
        if process is not None and process.poll() is None:
            # This unreaped child cannot have a reused PID. Its session was
            # created by this Popen, not inferred from a port or an old file.
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                raise OwnershipError('新服务停止超时，请人工检查启动日志。') from None
        raise


def stop(root, service, timeout=15):
    record = inspect(root, service)
    if record is None:
        clear(root, service)
        return
    # Verify again immediately before signalling the dedicated group. Workers
    # inherit it; independent ETL executors start a different session/group.
    if not verify(record):
        raise OwnershipError('服务身份在停止前变化，未发送停止信号。')
    os.killpg(record['pid'], signal.SIGTERM)
    deadline = time.monotonic() + timeout
    while group_alive(record['pid']):
        if time.monotonic() >= deadline:
            raise OwnershipError('服务进程组停止超时；身份记录已保留，不会强杀未知进程。')
        time.sleep(0.1)
    clear(root, service)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('launch', 'check', 'stop'))
    parser.add_argument('service', choices=('backend', 'frontend'))
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    try:
        if args.action == 'launch':
            if len(args.arguments) < 2:
                parser.error('launch requires a log path and command')
            signal.signal(signal.SIGHUP, signal.SIG_IGN)
            print(launch(PROJECT_ROOT, args.service, args.arguments[0], args.arguments[1:]))
        elif args.action == 'stop':
            stop(PROJECT_ROOT, args.service)
        else:
            record = inspect(PROJECT_ROOT, args.service)
            if record and args.arguments:
                verify_listener(record, args.arguments[0])
            return 0 if record else 1
    except (OwnershipError, OSError) as exc:
        print(f'[服务保护] {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())

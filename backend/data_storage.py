"""Managed data-directory relocation. No market-data or numerical dependencies."""
from __future__ import annotations

import fcntl
import json
import os
import shutil
import stat
import tempfile
import uuid
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from backend.data_sources.models import CenterError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MARKER = '.fund-storage-identity.json'
RESERVE_BYTES = 2 * 1024**3


class StorageError(CenterError):
    def __init__(self, code, message, status=409):
        super().__init__(code, message, status)


def read_json(path):
    try:
        if path.is_symlink() or path.stat().st_size > 131072:
            raise ValueError
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise ValueError
        return value
    except (OSError, ValueError, TypeError) as exc:
        raise StorageError('STORAGE_CONFIG_INVALID', '存储配置或身份文件不可读取，请检查磁盘；不会回退到本机。', 503) from exc


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, name = tempfile.mkstemp(prefix='.storage-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as out:
            json.dump(value, out, ensure_ascii=False)
            out.flush()
            os.fsync(out.fileno())
        os.replace(name, path)
        fsync_dir(path.parent)
    finally:
        Path(name).unlink(missing_ok=True)


def fsync_dir(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@contextmanager
def file_lease(path, *, shared=False):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open('a+') as handle:
        try:
            fcntl.flock(handle, (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise StorageError('STORAGE_BUSY', '仍有服务、下载或存储操作运行。请停止服务及下载后重试；不会强制中止或删除锁。') from exc
        try:
            yield handle
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def mount_anchor(path):
    """Find the filesystem mount without parsing localized shell output."""
    path = path.resolve(strict=True)
    while path.parent != path and path.parent.stat().st_dev == path.stat().st_dev:
        path = path.parent
    return path


class StorageManager:
    def __init__(self, project=PROJECT_ROOT):
        self.project = Path(project).resolve()
        self.logical = self.project / 'data'
        self.control = self.project / '.storage'
        self.config_path = self.control / 'config.json'

    def config(self):
        if self.control.is_symlink():
            raise StorageError('STORAGE_CONFIG_INVALID', '存储控制目录不能是软链接。', 503)
        if not self.config_path.exists() and not self.config_path.is_symlink():
            if self.logical.is_symlink():
                raise StorageError('STORAGE_UNMANAGED_LINK', 'data 是未登记的目录链接，请先核验存储配置，禁止自动写入。', 503)
            return {'revision': 0, 'active': None, 'pending': None}
        value = read_json(self.config_path)
        if type(value.get('revision')) is not int or 'active' not in value or 'pending' not in value:
            raise StorageError('STORAGE_CONFIG_INVALID', '存储配置格式无效。', 503)
        return value

    def guard(self, *, write=False):
        config = self.config()
        active = config['active']
        if not active:
            pending = config.get('pending') or {}
            if pending.get('phase') in {'SWITCHING', 'LINKED'} or (write and pending.get('phase') not in {None, 'PLANNED'}):
                raise StorageError('STORAGE_SWITCH_PENDING', '数据目录切换尚未完成，请运行 start_services.sh start 恢复。', 503)
            return self.logical
        target = Path(active['target'])
        anchor = Path(active['mount'])
        if (not self.logical.is_symlink() or self.logical.resolve() != target
                or not target.is_dir() or target.is_symlink() or not anchor.is_dir()
                or mount_anchor(target) != anchor):
            raise StorageError('STORAGE_OFFLINE', '数据磁盘未挂载或目录入口已改变，请接回原磁盘；不会改写本机目录。', 503)
        identity = read_json(target / MARKER)
        if identity != {'id': active['id'], 'project': str(self.project)}:
            raise StorageError('STORAGE_IDENTITY_CHANGED', '存储身份不一致，拒绝使用同名但不同的数据目录。', 503)
        if write and shutil.disk_usage(target).free < RESERVE_BYTES:
            raise StorageError('STORAGE_LOW_SPACE', '数据磁盘剩余空间不足 2 GiB，已阻止继续写入；请先释放空间。', 503)
        return target

    def probe(self, text):
        if not isinstance(text, str) or not text.strip() or '\x00' in text:
            raise StorageError('STORAGE_PATH_INVALID', '请输入后端本机的绝对目录路径。', 422)
        raw = Path(text.strip())
        if not raw.is_absolute() or '..' in raw.parts or raw.is_symlink():
            raise StorageError('STORAGE_PATH_INVALID', '必须使用绝对路径，不能包含上级跳转或目录链接。', 422)
        # Never mkdir an absent /Volumes/Disk mount point.
        if not raw.parent.is_dir():
            raise StorageError('STORAGE_PARENT_MISSING', '父目录不存在，请先挂载磁盘并创建父目录；不会自动创建挂载点。')
        target = raw.parent.resolve(strict=True) / raw.name
        source = self.logical.resolve()
        if (target == Path('/') or target == Path.home().resolve()
                or target == self.project or self.project in target.parents
                or target == source or source in target.parents or target in source.parents):
            raise StorageError('STORAGE_UNSAFE_PATH', '请选择项目外的专用空目录，不能使用系统根目录、主目录或数据区的父子目录。')
        if target.exists() and (not target.is_dir() or next(target.iterdir(), None) is not None):
            raise StorageError('STORAGE_TARGET_NOT_EMPTY', '目标必须为不存在或为空的专用目录；不会合并或覆盖其他文件。')
        anchor = mount_anchor(target.parent)
        if raw.parts[1:2] == ('Volumes',) and len(raw.parts) >= 3:
            volume = Path('/Volumes') / raw.parts[2]
            if not volume.is_mount():
                raise StorageError('STORAGE_VOLUME_NOT_MOUNTED', '指定的外接磁盘没有真正挂载，拒绝写入同名本机目录。')
        try:
            with tempfile.TemporaryDirectory(prefix='.fund-storage-probe-', dir=target.parent) as name:
                directory = Path(name)
                test = directory / 'probe'
                with test.open('xb') as out:
                    out.write(b'fund-storage-probe')
                    out.flush()
                    os.fsync(out.fileno())
                test.chmod(0o600)
                if stat.S_IMODE(test.stat().st_mode) != 0o600:
                    raise StorageError('STORAGE_PERMISSIONS', '目标盘不能保存私有文件权限，不适合迁移含凭据的数据区。请使用支持 POSIX 权限的文件系统。')
                with file_lease(test):
                    try:
                        with file_lease(test):
                            raise StorageError('STORAGE_LOCK_UNSUPPORTED', '目标盘不支持互斥文件锁。')
                    except StorageError as exc:
                        if exc.code != 'STORAGE_BUSY':
                            raise
                os.replace(test, directory / 'renamed')
                fsync_dir(directory)
        except OSError as exc:
            raise StorageError('STORAGE_NOT_WRITABLE', '目标盘不可写，或不支持同步写入及原子替换；请检查权限与文件系统。') from exc
        usage = shutil.disk_usage(target.parent)
        if usage.free < RESERVE_BYTES:
            raise StorageError('STORAGE_LOW_SPACE', '目标剩余空间不足 2 GiB，请选择其他磁盘。')
        return {'target': str(target), 'mount': str(anchor), 'free_bytes': usage.free,
                'total_bytes': usage.total, 'reserve_bytes': RESERVE_BYTES,
                'same_device': target.parent.stat().st_dev == source.stat().st_dev,
                'message': '目录能力检查通过；完整数据大小与容量将在离线迁移时再次核验。'}

    def save_plan(self, text, revision):
        with file_lease(self.control / 'config.lock'):
            config = self.config()
            if type(revision) is not int or revision != config['revision']:
                raise StorageError('STORAGE_REVISION_CONFLICT', '存储配置已变化，请刷新后重新检查。')
            if config['active']:
                raise StorageError('STORAGE_ALREADY_EXTERNAL', '当前已使用指定磁盘。本期不支持再次跨盘搬迁，以免改变历史任务的物理路径引用。')
            if config['pending']:
                raise StorageError('STORAGE_PLAN_EXISTS', '已有迁移计划，请先完成或取消它。')
            result = self.probe(text)
            identifier = uuid.uuid4().hex
            config.update(revision=revision + 1, pending={
                'id': identifier, 'target': result['target'], 'mount': result['mount'],
                'phase': 'PLANNED', 'files': 0, 'bytes': 0,
                'message': '计划已保存。停止下载后重启服务，在启动前离线迁移。',
            })
            atomic_json(self.config_path, config)
            return self.status()

    def cancel_plan(self, revision):
        with file_lease(self.control / 'config.lock'):
            config = self.config()
            if type(revision) is not int or revision != config['revision']:
                raise StorageError('STORAGE_REVISION_CONFLICT', '配置已变化，请刷新。')
            if config['pending'] and config['pending']['phase'] in {'SWITCHING', 'LINKED'}:
                raise StorageError('STORAGE_MIGRATION_STARTED', '目录切换已开始，请离线重试完成；不会删除暂存或原数据。')
            config.update(revision=revision + 1, pending=None)
            atomic_json(self.config_path, config)
            return self.status()

    def status(self):
        result = {'logical_path': str(self.logical), 'online': False, 'error': None,
                  'volumes': [], 'free_bytes': None, 'total_bytes': None}
        try:
            config = self.config()
            result.update(config)
            target = self.guard()
            usage = shutil.disk_usage(target if target.exists() else self.project)
            result.update(online=True, actual_path=str(target), free_bytes=usage.free, total_bytes=usage.total)
        except (StorageError, OSError) as exc:
            result['error'] = exc.message if isinstance(exc, StorageError) else '存储目录无法访问，请检查磁盘。'
        volumes = Path('/Volumes')
        if volumes.is_dir():
            for volume in sorted(volumes.iterdir()):
                if volume.is_symlink() or not volume.is_mount():
                    continue
                try:
                    usage = shutil.disk_usage(volume)
                    result['volumes'].append({'name': volume.name, 'path': str(volume), 'free_bytes': usage.free})
                except OSError:
                    continue
        return result


_manager = StorageManager()


def guard_path(path, *, write=False):
    """Check both logical and worker-resolved paths; isolated test roots stay isolated."""
    candidate = Path(os.path.abspath(path))
    config = _manager.config()
    roots = [_manager.logical]
    if config['active']:
        roots.append(Path(config['active']['target']))
    if any(candidate == root or root in candidate.parents for root in roots):
        _manager.guard(write=write)


def storage_lifespan(inner):
    @asynccontextmanager
    async def wrapped(app):
        with file_lease(_manager.control / 'service.lock', shared=True):
            _manager.guard()
            async with inner(app):
                yield
    return wrapped

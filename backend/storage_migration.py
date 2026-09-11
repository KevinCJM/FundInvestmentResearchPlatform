"""Offline copy/verify/switch transaction. Never download or rewrite data lineage."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
import time
from pathlib import Path

from .data_storage import (MARKER, RESERVE_BYTES, StorageError, StorageManager,
                           atomic_json, file_lease, fsync_dir, mount_anchor, read_json)


def files(root, *, allow_marker=False):
    for directory, dirs, names in os.walk(root, followlinks=False):
        dirs.sort()
        for name in sorted([*dirs, *names]):
            path = Path(directory) / name
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode) or not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
                raise StorageError('STORAGE_SPECIAL_FILE', '数据区包含软链接或特殊文件，需人工核验后迁移。')
        for name in sorted(names):
            if Path(directory) == root and name == MARKER:
                if allow_marker:
                    continue
                raise StorageError('STORAGE_SOURCE_MARKER', '原数据区已有未登记的存储标记，请人工核验。')
            yield Path(directory) / name


def digest(path, notify=lambda: None):
    h = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(4 * 1024**2), b''):
            h.update(block)
            notify()
    return h.hexdigest()


def inventory(root, notify=lambda *_: None, *, allow_marker=False):
    h, count, size, largest = hashlib.sha256(), 0, 0, 0
    for path in files(root, allow_marker=allow_marker):
        st = path.stat()
        record = [path.relative_to(root).as_posix(), st.st_size, st.st_mtime_ns, stat.S_IMODE(st.st_mode)]
        h.update(json.dumps(record, ensure_ascii=False).encode() + b'\n')
        count += 1
        size += st.st_size
        largest = max(largest, st.st_size)
        notify(count, size)
    return {'files': count, 'bytes': size, 'largest_file': largest, 'fingerprint': h.hexdigest()}


def copy_verified(source, target, *, temporary_dir=None, notify=lambda: None):
    """Bounded memory; resume only bytes verified against the current frozen source."""
    before = source.stat()
    expected = digest(source, notify)
    if target.exists() and not target.is_symlink() and target.stat().st_size == before.st_size and digest(target, notify) == expected:
        shutil.copystat(source, target)
        return expected
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if shutil.disk_usage(target.parent).free < before.st_size + RESERVE_BYTES:
        raise StorageError('STORAGE_LOW_SPACE', '复制时目标磁盘空间不足；原数据和已校验暂存文件保留。')
    fd, name = tempfile.mkstemp(prefix='.copy-', dir=temporary_dir or target.parent)
    try:
        with os.fdopen(fd, 'wb') as out, source.open('rb') as src:
            for block in iter(lambda: src.read(4 * 1024**2), b''):
                out.write(block)
                notify()
            out.flush()
            os.fsync(out.fileno())
        copied = Path(name)
        shutil.copystat(source, copied)
        after = source.stat()
        if ((before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns)
                or digest(copied, notify) != expected):
            raise StorageError('STORAGE_COPY_MISMATCH', '复制期间源文件变化或校验和不一致，未切换活动目录。')
        os.replace(copied, target)
    finally:
        Path(name).unlink(missing_ok=True)
    return expected


class StorageMigration:
    def __init__(self, manager: StorageManager, progress=print):
        self.manager, self.progress = manager, progress
        self.last_report = 0.0

    def report(self, config, phase, message, **values):
        previous = config['pending']['phase']
        config['pending'].update(phase=phase, message=message, **values)
        if phase != previous or time.monotonic() - self.last_report >= 1:
            atomic_json(self.manager.config_path, config)
            self.progress(message)
            self.last_report = time.monotonic()

    def startup(self):
        manager = self.manager
        config = manager.config()
        if not config['pending']:
            manager.guard()
            return manager.status()
        # The API may not yet support the lease when upgrading this feature.
        pid_file = manager.project / '.run/backend.pid'
        if pid_file.exists():
            try:
                pid = int(pid_file.read_text().strip())
                os.kill(pid, 0)
            except (ValueError, ProcessLookupError):
                pass
            else:
                raise StorageError('STORAGE_API_RUNNING', '后端仍在运行，请使用 start_services.sh restart，在启动前迁移。')
        with file_lease(manager.control / 'config.lock'), file_lease(manager.control / 'service.lock'):
            config = manager.config()
            pending = config['pending']
            if config['active'] or not re.fullmatch('[a-f0-9]{32}', pending['id']):
                raise StorageError('STORAGE_PLAN_INVALID', '迁移计划无效，拒绝修改数据入口。')
            try:
                if pending['phase'] in {'SWITCHING', 'LINKED'}:
                    self.finish_switch(config)
                else:
                    manager.guard()
                    manager.logical.mkdir(exist_ok=True)
                    with file_lease(manager.logical / '.tushare_refresh.lock'):
                        self.copy(config)
                        self.finish_switch(config)
            except Exception as exc:
                pending['message'] = (exc.message if isinstance(exc, StorageError)
                                      else f'迁移未完成（{type(exc).__name__}），原数据及暂存文件保留。')
                atomic_json(manager.config_path, config)
                raise
        return manager.status()

    def copy(self, config):
        manager, plan = self.manager, config['pending']
        # A data/ symlink must not hide Git-tracked fixtures/source from Git.
        if (manager.project / '.git').exists():
            tracked = subprocess.run(['git', 'ls-files', '--', 'data'], cwd=manager.project,
                                     capture_output=True, check=True, timeout=10).stdout
            if tracked.strip():
                raise StorageError('STORAGE_TRACKED_DATA', 'data/ 包含 Git 跟踪文件，请先把版本化夹具与运行数据分离；不会移动这些文件。')
        probe = manager.probe(plan['target'])
        if probe['mount'] != plan['mount']:
            raise StorageError('STORAGE_MOUNT_CHANGED', '目标挂载位置发生变化，请重新核验。')
        target = Path(plan['target'])
        stage = target.parent / ('.fund-storage-' + plan['id'])
        if stage.is_symlink():
            raise StorageError('STORAGE_STAGE_CHANGED', '迁移暂存目录不能是软链接。')
        identity = {'id': plan['id'], 'project': str(manager.project)}
        if stage.exists():
            if read_json(stage / MARKER) != identity:
                raise StorageError('STORAGE_STAGE_CHANGED', '迁移暂存目录不属于当前计划。')
        else:
            stage.mkdir(mode=0o700)
            atomic_json(stage / MARKER, identity)
        temporary = target.parent / ('.fund-storage-' + plan['id'] + '-temporary')
        if temporary.exists():
            if temporary.is_symlink() or read_json(temporary / MARKER) != identity:
                raise StorageError('STORAGE_STAGE_CHANGED', '临时复制目录不属于当前迁移。')
        else:
            temporary.mkdir(mode=0o700)
            atomic_json(temporary / MARKER, identity)
        # Only this private, identity-bound directory contains interrupted copy buffers.
        for leftover in temporary.iterdir():
            if leftover.name != MARKER:
                if leftover.is_symlink() or not leftover.is_file() or not leftover.name.startswith('.copy-'):
                    raise StorageError('STORAGE_STAGE_CHANGED', '临时复制目录存在未知内容，拒绝清理。')
                leftover.unlink()
        self.report(config, 'SCANNING', '正在扫描原数据区，统计文件与所需容量…')
        source_inventory = inventory(manager.logical, lambda count, size: self.report(
            config, 'SCANNING', f'正在扫描：{count:,} 个文件，{size:,} 字节', files=count, bytes=size))
        if plan.get('inventory') and plan['inventory'] != source_inventory:
            raise StorageError('STORAGE_SOURCE_CHANGED', '上次迁移后原数据已变化，请取消旧计划后重新检查；旧暂存文件保留，不会混用。')
        # Conservative first-copy budget. On resume, per-file checks below govern space.
        if plan.get('inventory') is None and probe['free_bytes'] < source_inventory['bytes'] + RESERVE_BYTES:
            raise StorageError('STORAGE_LOW_SPACE', '目标盘容量不足以容纳完整数据和 2 GiB 安全余量；原目录未变。')
        plan['inventory'] = source_inventory
        atomic_json(manager.config_path, config)
        self.report(config, 'COPYING', '正在逐文件复制并校验 SHA-256；已校验文件可断点复用。', files=0, bytes=0)
        receipts = manager.control / 'migrations'
        receipts.mkdir(parents=True, exist_ok=True)
        listing = receipts / (plan['id'] + '.files.jsonl')
        done, copied_bytes = 0, 0
        with listing.open('w') as output:
            for source in files(manager.logical):
                relative = source.relative_to(manager.logical)
                if stage.resolve() not in (stage / relative).resolve().parents or (stage / relative).is_symlink():
                    raise StorageError('STORAGE_STAGE_CHANGED', '暂存路径出现目录逃逸，拒绝写入。')
                checksum = copy_verified(source, stage / relative, temporary_dir=temporary, notify=lambda: self.report(
                    config, 'COPYING', f'复制并校验：{done:,}/{source_inventory["files"]:,} 个文件；当前 {source.name}'))
                size = source.stat().st_size
                output.write(json.dumps({'path': relative.as_posix(), 'size': size, 'checksum': checksum}) + '\n')
                done += 1
                copied_bytes += size
                self.report(config, 'COPYING', f'复制并校验：{done:,}/{source_inventory["files"]:,} 个文件，{copied_bytes:,}/{source_inventory["bytes"]:,} 字节', files=done, bytes=copied_bytes)
            output.flush()
            os.fsync(output.fileno())
        # Preserve empty directories as well; never synthesize missing downloaded rows.
        for directory, dirs, _ in os.walk(manager.logical):
            for name in dirs:
                destination = stage / (Path(directory) / name).relative_to(manager.logical)
                destination.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.report(config, 'VERIFYING', '正在复核原数据清单，未切换活动目录…')
        if inventory(manager.logical, lambda n, b: self.report(config, 'VERIFYING', f'正在复核：{n:,} 个文件')) != source_inventory:
            raise StorageError('STORAGE_SOURCE_CHANGED', '原数据在迁移期间发生变化，已拒绝切换。请确认所有写入进程已停止。')
        receipt = {'id': plan['id'], 'source': str(manager.logical), 'target': str(target),
                   'inventory': source_inventory, 'listing_checksum': digest(listing), 'verified': True}
        atomic_json(receipts / (plan['id'] + '.json'), receipt)
        # Persist the switch intent before any source rename (crash-recoverable).
        self.report(config, 'SWITCHING', '全部内容校验通过，正在切换数据入口；原数据保留为本机副本。')
        atomic_json(manager.config_path, config)
        # finish_switch performs the rename so a crash at either rename is resumable.

    def finish_switch(self, config):
        manager, plan = self.manager, config['pending']
        target = Path(plan['target'])
        identity = {'id': plan['id'], 'project': str(manager.project)}
        stage = target.parent / ('.fund-storage-' + plan['id'])
        receipt = read_json(manager.control / 'migrations' / (plan['id'] + '.json'))
        if receipt.get('target') != str(target) or receipt.get('verified') is not True:
            raise StorageError('STORAGE_RECEIPT_INVALID', '缺少匹配的复制校验收据，拒绝切换。')
        if not target.exists() or (target.is_dir() and next(target.iterdir(), None) is None):
            if read_json(stage / MARKER) != identity:
                raise StorageError('STORAGE_STAGE_CHANGED', '暂存身份已改变。')
            if target.exists():
                target.rmdir()
            os.replace(stage, target)
            fsync_dir(target.parent)
        if target.is_symlink() or read_json(target / MARKER) != identity or mount_anchor(target) != Path(plan['mount']):
            raise StorageError('STORAGE_IDENTITY_CHANGED', '目标盘身份或挂载发生变化，未改写原数据。')
        listing = manager.control / 'migrations' / (plan['id'] + '.files.jsonl')
        if digest(listing) != receipt['listing_checksum']:
            raise StorageError('STORAGE_RECEIPT_INVALID', '文件校验清单发生变化，拒绝切换。')
        # Recheck after a crashed switch too: an identity marker alone proves no content.
        if inventory(target, allow_marker=True) != receipt['inventory']:
            raise StorageError('STORAGE_TARGET_CHANGED', '目标文件清单或时间戳发生变化，拒绝切换。')
        with listing.open() as records:
            for line in records:
                row = json.loads(line)
                relative = Path(row['path'])
                if relative.is_absolute() or '..' in relative.parts or digest(target / relative, lambda: self.report(
                        config, 'SWITCHING', f'切换前复核：{relative.name}')) != row['checksum']:
                    raise StorageError('STORAGE_TARGET_CHANGED', '目标文件校验和不符，原数据副本保留。')
        backup = manager.control / 'backups' / plan['id'] / 'data'
        if not manager.logical.is_symlink():
            if manager.logical.exists():
                if backup.exists():
                    raise StorageError('STORAGE_BACKUP_EXISTS', '备份路径已存在，拒绝覆盖原数据。')
                backup.parent.mkdir(parents=True, exist_ok=True)
                os.replace(manager.logical, backup)
                fsync_dir(manager.project)
            if not backup.is_dir():
                raise StorageError('STORAGE_SOURCE_MISSING', '找不到本次原数据副本，拒绝完成切换。')
            manager.logical.symlink_to(target, target_is_directory=True)
            fsync_dir(manager.project)
        elif manager.logical.resolve() != target:
            raise StorageError('STORAGE_LINK_CHANGED', 'data 入口被其他操作改变，拒绝覆盖。')
        config.update(active={'id': plan['id'], 'target': str(target), 'mount': plan['mount'],
                              'backup': str(backup), 'backup_removed': False},
                      pending=None, revision=config['revision'] + 1)
        atomic_json(manager.config_path, config)
        manager.guard()
        self.progress('存储迁移成功 / SUCCESS：数据已使用指定目录。原本机副本仍保留，确认后可清理释放空间。')

    def cleanup_backup(self, confirmation):
        manager = self.manager
        with file_lease(manager.control / 'config.lock'), file_lease(manager.control / 'service.lock'):
            manager.guard()
            config = manager.config()
            active = config['active']
            if not active or active['id'] != confirmation or not re.fullmatch('[a-f0-9]{32}', confirmation):
                raise StorageError('STORAGE_CONFIRMATION_REQUIRED', '请提供已成功迁移的准确 ID；不接受任意删除路径。')
            backup = manager.control / 'backups' / confirmation / 'data'
            receipt = read_json(manager.control / 'migrations' / (confirmation + '.json'))
            if (active.get('backup') != str(backup) or backup.is_symlink()
                    or backup.parent.is_symlink() or backup.parent.parent.is_symlink()
                    or receipt.get('verified') is not True or receipt.get('target') != active['target']):
                raise StorageError('STORAGE_BACKUP_INVALID', '副本登记或迁移收据不一致，拒绝清理。')
            if active.get('backup_removed'):
                return manager.status()
            if not backup.is_dir():
                raise StorageError('STORAGE_BACKUP_MISSING', '副本已被外部操作移动或删除，请人工核验。')
            if inventory(backup) != receipt['inventory']:
                raise StorageError('STORAGE_BACKUP_CHANGED', '原本机副本已被修改，拒绝删除；请人工核验新增或变化的文件。')
            with file_lease(manager.logical / '.tushare_refresh.lock'):
                self.progress(f'正在删除已确认的原本机副本（不可恢复）：{backup}')
                shutil.rmtree(backup)
                active['backup_removed'] = True
                config['revision'] += 1
                atomic_json(manager.config_path, config)
        self.progress('原本机数据副本已清理，空间已释放；指定磁盘上的活动数据未删除。')
        return manager.status()

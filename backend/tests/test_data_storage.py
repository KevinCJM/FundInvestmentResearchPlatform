"""No network, no real mounts, no production data changes."""
import json
import os
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend import data_storage as storage
from backend import storage_migration as migration
from backend.data_storage import StorageError, StorageManager, atomic_json, file_lease
from backend.storage_migration import StorageMigration


@pytest.fixture
def setup(tmp_path, monkeypatch):
    project = tmp_path / '项目'
    project.mkdir()
    (project / 'data').mkdir()
    disk = tmp_path / '外接 磁盘'
    disk.mkdir()
    manager = StorageManager(project)
    monkeypatch.setattr(storage, '_manager', manager)
    monkeypatch.setattr(storage, 'RESERVE_BYTES', 0)
    monkeypatch.setattr(migration, 'RESERVE_BYTES', 0)
    return manager, disk / '研究 数据'


def seed(manager):
    path = manager.logical / 'versions' / 'one'
    path.mkdir(parents=True)
    (path / 'fund_nav_df.parquet').write_bytes(b'original-file-content')
    os.utime(path / 'fund_nav_df.parquet', ns=(123456789, 123456789))
    (manager.logical / '.tushare_token').write_bytes(b'private-test-placeholder')
    (manager.logical / '.tushare_token').chmod(0o600)
    (manager.logical / 'empty').mkdir()
    (manager.logical / 'tushare_active.json').write_text(json.dumps({
        'schema_version': 1, 'snapshot_dir': 'versions/one', 'files': {'fund_nav_df.parquet': 21}}))


def activate(manager, target):
    manager.save_plan(str(target), 0)
    return StorageMigration(manager, lambda *_: None).startup()


def test_probe_and_save_are_not_migration(setup):
    manager, target = setup
    seed(manager)
    probe = manager.probe(str(target))
    assert probe['target'] == str(target) and not target.exists()
    assert not list(target.parent.glob('.fund-storage-probe-*'))
    result = manager.save_plan(str(target), 0)
    assert result['pending']['phase'] == 'PLANNED'
    assert not manager.logical.is_symlink() and not target.exists()
    assert manager.cancel_plan(1)['pending'] is None


@pytest.mark.parametrize('kind', ['relative', 'root', 'parent', 'inside', 'nonempty', 'file', 'missing_parent', 'link', 'dotdot'])
def test_unsafe_targets_fail(setup, kind):
    manager, target = setup
    choices = {'relative': Path('relative'), 'root': Path('/'), 'parent': manager.project.parent,
               'inside': manager.logical / 'nested', 'missing_parent': target / 'missing' / 'data',
               'dotdot': target.parent / '..' / 'data'}
    if kind == 'nonempty':
        target.mkdir(); (target / 'user-file').write_text('keep')
    elif kind == 'file':
        target.write_text('keep')
    elif kind == 'link':
        target.symlink_to(manager.logical)
    with pytest.raises(StorageError):
        manager.probe(str(choices.get(kind, target)))
    assert manager.config()['revision'] == 0


def test_source_content_timestamps_receipts_and_new_writes(setup):
    manager, target = setup
    seed(manager)
    status = activate(manager, target)
    assert status['online'] and manager.logical.is_symlink()
    assert manager.logical.resolve() == target
    backup = Path(status['active']['backup'])
    assert backup.is_dir()
    assert (target / 'versions/one/fund_nav_df.parquet').stat().st_mtime_ns == 123456789
    assert (target / '.tushare_token').stat().st_mode & 0o777 == 0o600
    assert (target / 'empty').is_dir()
    assert (target / 'tushare_active.json').read_bytes() == (backup / 'tushare_active.json').read_bytes()
    from backend.market_data import resolve_tushare_data_dir
    assert resolve_tushare_data_dir(manager.logical, strict=True) == target / 'versions/one'
    from backend.data_sources.store import SourceStore
    from backend.data_sources.etl_store import EtlStore
    store = SourceStore(manager.logical)
    journal = EtlStore(store)
    artifact = journal.write_json(manager.logical / 'etl_runs/new/receipt.json', {'done': True})
    assert journal.checked_path(artifact) == target / 'etl_runs/new/receipt.json'
    assert not (backup / 'etl_runs/new').exists()
    assert not (backup / 'data_sources.sqlite3').exists()
    assert (target / 'data_sources.sqlite3').is_file()
    from T01_get_data import save_dataframe
    import pandas as pd
    save_dataframe(pd.DataFrame({'x': [1, 2]}), manager.logical / 'new.parquet', quiet=True)
    assert (target / 'new.parquet').is_file() and not (backup / 'new.parquet').exists()


@pytest.mark.parametrize('lock', ['service', 'download'])
def test_api_and_download_lock_block_switch(setup, lock):
    manager, target = setup
    manager.save_plan(str(target), 0)
    path = manager.control / 'service.lock' if lock == 'service' else manager.logical / '.tushare_refresh.lock'
    with file_lease(path, shared=lock == 'service'):
        with pytest.raises(StorageError, match='仍有服务'):
            StorageMigration(manager).startup()
    assert not manager.logical.is_symlink() and not target.exists()


def test_revision_and_second_move_rejected(setup):
    manager, target = setup
    with pytest.raises(StorageError, match='配置已变化'):
        manager.save_plan(str(target), 9)
    activate(manager, target)
    with pytest.raises(StorageError, match='再次跨盘'):
        manager.save_plan(str(target.parent / 'other'), manager.config()['revision'])


def test_internal_symlink_not_followed(setup):
    manager, target = setup
    (manager.logical / 'escape').symlink_to(target.parent)
    manager.save_plan(str(target), 0)
    with pytest.raises(StorageError, match='软链接'):
        StorageMigration(manager).startup()
    assert not manager.logical.is_symlink() and not target.exists()


def test_unplug_no_recreation_fallback_or_network(setup, monkeypatch):
    manager, target = setup
    seed(manager)
    activate(manager, target)
    target.rename(target.with_name('detached'))
    assert not manager.status()['online']
    from backend.market_data import resolve_tushare_data_dir
    from backend.data_sources.store import SourceStore
    from backend.data_sources.batches import atomic_bytes
    from backend.services.refresh_runtime import InterProcessFileLock
    from T01_get_data import ensure_output_dir
    for operation in (lambda: manager.guard(), lambda: SourceStore(manager.logical),
                      lambda: SourceStore(target), lambda: resolve_tushare_data_dir(manager.logical),
                      lambda: atomic_bytes(target / 'new/receipt.json', b'{}'),
                      lambda: InterProcessFileLock(target / '.lock').acquire(),
                      lambda: ensure_output_dir(target / 'new')):
        with pytest.raises(StorageError):
            operation()
    assert not target.exists()


def test_wrong_identity_fails_closed(setup):
    manager, target = setup
    activate(manager, target)
    atomic_json(target / storage.MARKER, {'id': 'wrong', 'project': str(manager.project)})
    with pytest.raises(StorageError, match='身份不一致'):
        manager.guard()


def test_remaining_space_stops_write_not_diagnostics(setup, monkeypatch):
    manager, target = setup
    activate(manager, target)
    monkeypatch.setattr(storage, 'RESERVE_BYTES', 10**18)
    assert manager.status()['online']
    with pytest.raises(StorageError, match='剩余空间不足'):
        storage.guard_path(target / 'new-file', write=True)


def test_switch_rechecks_target_checksum_after_crash(setup, monkeypatch):
    manager, target = setup
    seed(manager)
    manager.save_plan(str(target), 0)
    worker = StorageMigration(manager, lambda *_: None)
    monkeypatch.setattr(worker, 'finish_switch', lambda _: (_ for _ in ()).throw(OSError('crash')))
    with pytest.raises(OSError):
        worker.startup()
    plan = manager.config()['pending']
    stage = target.parent / ('.fund-storage-' + plan['id'])
    file = stage / 'versions/one/fund_nav_df.parquet'
    stamp = file.stat().st_mtime_ns
    file.write_bytes(b'x' * file.stat().st_size)
    os.utime(file, ns=(stamp, stamp))
    with pytest.raises(StorageError, match='校验和不符'):
        StorageMigration(manager, lambda *_: None).startup()
    assert not manager.logical.is_symlink()


def test_cleanup_rejects_added_backup_files(setup):
    manager, target = setup
    activate(manager, target)
    active = manager.config()['active']
    (Path(active['backup']) / 'new-user-file').write_text('do not delete')
    with pytest.raises(StorageError, match='副本已被修改'):
        StorageMigration(manager).cleanup_backup(active['id'])


def test_no_git_tracked_fixtures_are_moved(setup, monkeypatch):
    manager, target = setup
    (manager.project / '.git').mkdir()
    manager.save_plan(str(target), 0)
    from types import SimpleNamespace
    monkeypatch.setattr(migration.subprocess, 'run', lambda *_, **__: SimpleNamespace(stdout=b'data/fixtures/test.json'))
    with pytest.raises(StorageError, match='Git 跟踪文件'):
        StorageMigration(manager).startup()
    assert not manager.logical.is_symlink()


def test_copy_failure_preserves_source_and_can_resume(setup, monkeypatch):
    manager, target = setup
    seed(manager)
    manager.save_plan(str(target), 0)
    original = migration.copy_verified
    counter = []
    def fail_after_one(src, dst, **kwargs):
        counter.append(src)
        if len(counter) == 2:
            raise OSError('simulated unplug')
        return original(src, dst, **kwargs)
    monkeypatch.setattr(migration, 'copy_verified', fail_after_one)
    with pytest.raises(OSError):
        StorageMigration(manager, lambda *_: None).startup()
    assert not manager.logical.is_symlink()
    assert (manager.logical / '.tushare_token').exists()
    monkeypatch.setattr(migration, 'copy_verified', original)
    StorageMigration(manager, lambda *_: None).startup()
    assert manager.logical.resolve() == target


def test_source_changed_after_partial_copy_rejected(setup, monkeypatch):
    manager, target = setup
    seed(manager)
    manager.save_plan(str(target), 0)
    original = migration.copy_verified
    monkeypatch.setattr(migration, 'copy_verified', lambda *_, **__: (_ for _ in ()).throw(OSError()))
    with pytest.raises(OSError):
        StorageMigration(manager, lambda *_: None).startup()
    (manager.logical / 'new-data').write_bytes(b'more data')
    monkeypatch.setattr(migration, 'copy_verified', original)
    with pytest.raises(StorageError, match='上次迁移后'):
        StorageMigration(manager, lambda *_: None).startup()
    manager.cancel_plan(manager.config()['revision'])
    assert (manager.logical / 'new-data').exists()


def test_switch_crash_recovers_after_backup_rename(setup, monkeypatch):
    manager, target = setup
    seed(manager)
    manager.save_plan(str(target), 0)
    original = Path.symlink_to
    monkeypatch.setattr(Path, 'symlink_to', lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError('crash')))
    with pytest.raises(OSError):
        StorageMigration(manager, lambda *_: None).startup()
    assert not manager.logical.exists()
    assert manager.config()['pending']['phase'] == 'SWITCHING'
    monkeypatch.setattr(Path, 'symlink_to', original)
    StorageMigration(manager, lambda *_: None).startup()
    assert manager.guard() == target


def test_cleanup_requires_exact_confirmation_and_success(setup):
    manager, target = setup
    seed(manager)
    activate(manager, target)
    worker = StorageMigration(manager, lambda *_: None)
    with pytest.raises(StorageError):
        worker.cleanup_backup('../data')
    active = manager.config()['active']
    with file_lease(manager.control / 'service.lock', shared=True):
        with pytest.raises(StorageError):
            worker.cleanup_backup(active['id'])
    worker.cleanup_backup(active['id'])
    assert not Path(active['backup']).exists()
    assert (target / 'versions/one/fund_nav_df.parquet').exists()
    assert manager.config()['active']['backup_removed']


def test_low_space_fail_before_copy(setup, monkeypatch):
    manager, target = setup
    seed(manager)
    manager.save_plan(str(target), 0)
    usage = storage.shutil.disk_usage(target.parent)
    monkeypatch.setattr(storage.shutil, 'disk_usage', lambda _: usage._replace(free=1))
    with pytest.raises(StorageError, match='容量不足'):
        StorageMigration(manager, lambda *_: None).startup()
    assert not manager.logical.is_symlink()


def test_readonly_probe_fails_without_residue(setup, monkeypatch):
    manager, target = setup
    monkeypatch.setattr(storage.tempfile, 'TemporaryDirectory', lambda **_: (_ for _ in ()).throw(PermissionError()))
    with pytest.raises(StorageError, match='不可写'):
        manager.probe(str(target))
    assert not target.exists()


def test_corrupt_config_fails_closed(setup):
    manager, target = setup
    manager.control.mkdir()
    manager.config_path.write_text('not-json')
    assert manager.status()['online'] is False
    with pytest.raises(StorageError):
        manager.guard()


def test_restart_tolerates_listener_exiting_between_checks():
    import re
    import subprocess
    script = (Path(__file__).resolve().parents[2] / 'start_services.sh').read_text()
    function = '\n'.join(re.search(rf'^{name}\(\) \{{\n.*?^\}}', script, re.M | re.S).group()
                         for name in ('stop_services', 'restart_services'))
    harness = ('set -euo pipefail\n' + function + '\n'
               'BACKEND_PID_FILE=backend.pid; FRONTEND_PID_FILE=frontend.pid\n'
               'BACKEND_PORT=8000; FRONTEND_PORT=5173\n'
               'stop_process_by_pid_file() { :; }\n'
               'is_port_listening() { return 1; }\n'
               'kill() { echo unexpected-kill; return 1; }\n'
               'sleep() { :; }\n'
               'start_services() { printf "restart-continues"; }\n'
               'restart_services\n')
    result = subprocess.run(['bash', '-c', harness], capture_output=True, text=True)
    assert result.returncode == 0 and result.stdout.endswith('restart-continues')
    assert 'unexpected-kill' not in result.stdout


def test_local_api_probe_save_cancel_and_remote_denial(setup, monkeypatch):
    manager, target = setup
    from backend.services import storage_routes
    monkeypatch.setattr(storage_routes, 'manager', manager)
    monkeypatch.setattr(storage_routes, 'enabled', lambda: True)
    app = FastAPI()
    app.include_router(storage_routes.router)
    with TestClient(app, client=('127.0.0.1', 1234), base_url='http://127.0.0.1') as client:
        assert client.get('/api/data-storage').json()['online']
        assert client.post('/api/data-storage/probe', json={'path': str(target)}).status_code == 200
        assert client.put('/api/data-storage/plan', json={'path': str(target), 'expected_revision': 0}).status_code == 422
        response = client.put('/api/data-storage/plan', json={'path': str(target), 'expected_revision': 0, 'confirm': True})
        assert response.status_code == 200 and response.json()['pending']
        assert client.request('DELETE', '/api/data-storage/plan', json={'expected_revision': 1}).status_code == 200
        assert client.post('/api/data-storage/probe', json={'path': str(target)}, headers={'origin': 'https://evil.example'}).status_code == 403
    with TestClient(app, client=('192.0.2.3', 1234)) as client:
        assert client.get('/api/data-storage').status_code == 403
    monkeypatch.setattr(storage_routes, 'enabled', lambda: False)
    with TestClient(app) as client:
        assert client.post('/api/data-storage/probe', json={'path': str(target)}).status_code == 403

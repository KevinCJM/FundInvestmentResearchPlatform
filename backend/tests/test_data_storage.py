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
    from backend.custom_indicators.repository import AtomicJsonStore
    store = AtomicJsonStore(manager.logical / 'example.json')
    store.write_unlocked({'items': []})
    monkeypatch.setattr(storage, 'RESERVE_BYTES', 10**18)
    assert manager.status()['online']
    with store.locked():
        assert store.read_unlocked() == {'items': []}
        with pytest.raises(StorageError, match='剩余空间不足'):
            store.write_unlocked({'items': ['blocked']})
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


def another_project(manager, name='another checkout'):
    project = manager.project.parent / name
    project.mkdir()
    (project / 'data').mkdir()
    return StorageManager(project)


def attach(manager, target):
    checked = manager.probe_existing(str(target))
    manager.save_attachment(str(target), manager.config()['revision'], checked['id'])
    return StorageMigration(manager, lambda *_: None).startup()


def _storage_process(project, operation, connection):
    """Real independent interpreters, not threads sharing a mocked singleton."""
    from backend import data_storage
    manager = data_storage.StorageManager(Path(project))
    data_storage._manager = manager
    try:
        if operation in {'read', 'exclusive'}:
            with manager.data_lease(shared=operation == 'read'):
                connection.send('acquired')
                if not connection.poll(15):
                    raise TimeoutError('test parent did not release lease')
                connection.recv()
            connection.send('released')
        elif operation == 'download':
            from backend.services.refresh_runtime import InterProcessFileLock
            lock = InterProcessFileLock(manager.logical / '.tushare_refresh.lock')
            acquired = lock.acquire()
            connection.send(acquired)
            if acquired:
                try:
                    if not connection.poll(15):
                        raise TimeoutError('test parent did not release downloader')
                    connection.recv()
                finally:
                    lock.release()
        elif operation == 'increment':
            from backend.custom_indicators.repository import AtomicJsonStore
            store = AtomicJsonStore(manager.logical / 'shared.json')
            for _ in range(25):
                with store.locked():
                    payload = store.read_unlocked()
                    payload['count'] = payload.get('count', 0) + 1
                    store.write_unlocked(payload)
            connection.send('done')
    except Exception as exc:
        connection.send(getattr(exc, 'code', type(exc).__name__))
    finally:
        connection.close()


@pytest.fixture
def storage_children():
    import multiprocessing
    context = multiprocessing.get_context('spawn')
    children = []
    def start(manager, operation):
        parent, child = context.Pipe()
        process = context.Process(target=_storage_process, args=(str(manager.project), operation, child))
        process.start()
        child.close()
        children.append((process, parent))
        return parent
    yield start
    for process, connection in children:
        if process.is_alive():
            try:
                connection.send('release')
            except (BrokenPipeError, EOFError):
                pass
        process.join(20)
        if process.is_alive():
            process.terminate()  # Only the exact test-owned child, never an external PID.
            process.join(5)
        assert process.exitcode == 0
        connection.close()


def received(connection):
    assert connection.poll(15), 'child did not report within the test deadline'
    return connection.recv()


@pytest.mark.parametrize('legacy', [False, True])
def test_two_projects_read_same_data_without_ownership_or_copy(setup, monkeypatch, legacy):
    manager, target = setup
    seed(manager)
    activate(manager, target)
    if legacy:
        atomic_json(target / storage.MARKER, {'id': manager.config()['active']['id'], 'project': '/retired/checkout'})
    marker = (target / storage.MARKER).read_bytes()
    manifest = (target / 'tushare_active.json').read_bytes()
    other = another_project(manager)
    (other.logical / 'local-only.txt').write_text('preserve my data')
    result = attach(other, target)
    assert manager.guard() == other.guard() == target
    assert manager.logical.samefile(other.logical)
    assert (target / storage.MARKER).read_bytes() == marker
    assert (target / 'tushare_active.json').read_bytes() == manifest
    assert not (target / 'local-only.txt').exists()
    assert (Path(result['active']['backup']) / 'local-only.txt').read_text() == 'preserve my data'
    from backend.market_data import resolve_tushare_data_dir
    for owner in (manager, other):
        monkeypatch.setattr(storage, '_manager', owner)
        assert resolve_tushare_data_dir(owner.logical, strict=True) == target / 'versions/one'
    with pytest.raises(StorageError, match='不是迁移校验副本'):
        StorageMigration(other).cleanup_backup(result['active']['id'])


def test_cross_project_shared_reads_and_exclusive_storage_lease(setup, storage_children):
    manager, target = setup
    activate(manager, target)
    other = another_project(manager)
    attach(other, target)
    with manager.data_lease():
        reader = storage_children(other, 'read')
        assert received(reader) == 'acquired'
        assert received(storage_children(other, 'exclusive')) == 'STORAGE_BUSY'
        reader.send('release')
        assert received(reader) == 'released'
    with manager.data_lease(shared=False):
        assert received(storage_children(other, 'read')) == 'STORAGE_BUSY'


def test_cross_project_download_lock_is_one_physical_lock(setup, storage_children):
    manager, target = setup
    activate(manager, target)
    other = another_project(manager)
    attach(other, target)
    first = storage_children(manager, 'download')
    assert received(first) is True
    assert received(storage_children(other, 'download')) is False
    first.send('release')


def test_cross_project_json_transactions_do_not_lose_updates(setup, storage_children):
    manager, target = setup
    activate(manager, target)
    other = another_project(manager)
    attach(other, target)
    writers = [storage_children(owner, 'increment') for owner in (manager, other)]
    assert [received(writer) for writer in writers] == ['done', 'done']
    assert json.loads((target / 'shared.json').read_text())['count'] == 50


def test_shared_directory_unplug_rejects_read_write_and_lock_without_recreation(setup, monkeypatch):
    manager, target = setup
    activate(manager, target)
    other = another_project(manager)
    attach(other, target)
    target.rename(target.with_name('disconnected'))
    from backend.custom_indicators.repository import AtomicJsonStore
    from backend.data_sources.store import SourceStore
    for owner in (manager, other):
        monkeypatch.setattr(storage, '_manager', owner)
        assert not owner.status()['online']
        store = AtomicJsonStore(owner.logical / 'nested/new.json')
        for operation in (owner.guard, store.read_unlocked, lambda: store.write_unlocked({'items': []}),
                          lambda: SourceStore(owner.logical)):
            with pytest.raises(StorageError):
                operation()
        with pytest.raises(StorageError):
            with owner.data_lease():
                pytest.fail('disconnected data must not acquire a lease')
    assert not target.exists()


def test_legacy_configuration_and_inflight_migration_remain_usable(setup):
    manager, target = setup
    seed(manager)
    manager.save_plan(str(target), 0)
    config = manager.config()
    del config['pending']['identity_version']
    atomic_json(manager.config_path, config)
    StorageMigration(manager, lambda *_: None).startup()
    assert json.loads((target / storage.MARKER).read_text())['project'] == str(manager.project)
    original_config = manager.config_path.read_bytes()
    attach(another_project(manager), target)
    assert manager.guard() == target
    assert manager.config_path.read_bytes() == original_config


@pytest.mark.parametrize('change', ['identity', 'unplug', 'version', 'copying'])
def test_attach_revalidates_directory_before_switch(setup, change):
    manager, target = setup
    activate(manager, target)
    other = another_project(manager)
    original_id = manager.config()['active']['id']
    other.save_attachment(str(target), 0, original_id)
    if change == 'unplug':
        target.rename(target.with_name('disconnected'))
    else:
        marker = json.loads((target / storage.MARKER).read_text())
        marker.update({'identity': {'id': 'a' * 32}, 'version': {'schema_version': 999},
                       'copying': {'state': 'copying'}}[change])
        atomic_json(target / storage.MARKER, marker)
    with pytest.raises(StorageError):
        StorageMigration(other, lambda *_: None).startup()
    assert not other.logical.is_symlink()
    assert other.config()['active'] is None


def test_attach_crash_after_local_backup_rename_is_recoverable(setup, monkeypatch):
    manager, target = setup
    activate(manager, target)
    other = another_project(manager)
    (other.logical / 'important.txt').write_text('keep')
    other.save_attachment(str(target), 0, manager.config()['active']['id'])
    original = Path.symlink_to
    monkeypatch.setattr(Path, 'symlink_to', lambda *_args, **_kw: (_ for _ in ()).throw(OSError('crash')))
    with pytest.raises(OSError):
        StorageMigration(other, lambda *_: None).startup()
    assert other.config()['pending']['phase'] == 'SWITCHING'
    with pytest.raises(StorageError):
        other.cancel_plan(other.config()['revision'])
    monkeypatch.setattr(Path, 'symlink_to', original)
    result = StorageMigration(other, lambda *_: None).startup()
    assert other.guard() == target
    assert (Path(result['active']['backup']) / 'important.txt').read_text() == 'keep'


def test_attach_api_checks_confirmation_identity_revision_and_no_online_switch(setup, monkeypatch):
    manager, target = setup
    activate(manager, target)
    other = another_project(manager)
    from backend.services import storage_routes
    monkeypatch.setattr(storage_routes, 'manager', other)
    monkeypatch.setattr(storage_routes, 'enabled', lambda: True)
    app = FastAPI()
    app.include_router(storage_routes.router)
    with TestClient(app, client=('127.0.0.1', 1234), base_url='http://127.0.0.1') as client:
        checked = client.post('/api/data-storage/existing/probe', json={'path': str(target)})
        assert checked.status_code == 200
        payload = {'path': str(target), 'expected_revision': 0, 'expected_id': checked.json()['id']}
        assert client.put('/api/data-storage/existing/plan', json=payload).status_code == 422
        assert client.put('/api/data-storage/existing/plan', json={**payload, 'confirm': True, 'expected_id': 'bad'}).status_code == 409
        assert client.put('/api/data-storage/existing/plan', json={**payload, 'confirm': True, 'expected_revision': 3}).status_code == 409
        assert client.put('/api/data-storage/existing/plan', json={**payload, 'confirm': True}).status_code == 200
        assert not other.logical.is_symlink()
        with file_lease(other.control / 'service.lock', shared=True):
            with pytest.raises(StorageError):
                StorageMigration(other).startup()
    with TestClient(app, client=('192.0.2.3', 1234), base_url='http://127.0.0.1') as client:
        assert client.put('/api/data-storage/existing/plan', json={**payload, 'confirm': True}).status_code == 403


def test_local_lifespan_holds_shared_data_lease(setup, monkeypatch):
    from contextlib import asynccontextmanager
    manager, target = setup
    activate(manager, target)
    other = another_project(manager)
    attach(other, target)
    monkeypatch.setattr(storage, '_manager', manager)
    @asynccontextmanager
    async def lifespan(_app):
        yield
    app = FastAPI(lifespan=storage.storage_lifespan(lifespan))
    with TestClient(app):
        with other.data_lease():
            assert other.guard() == target
        with pytest.raises(StorageError, match='仍有服务'):
            with other.data_lease(shared=False):
                pytest.fail('the application must own a shared physical lease')
    with other.data_lease(shared=False):
        pass


def test_cannot_attach_a_substituted_lock_file(setup):
    manager, target = setup
    activate(manager, target)
    other = another_project(manager)
    external = other.project / 'keep.txt'
    external.write_text('unchanged')
    (target / storage.USE_LOCK).unlink()
    (target / storage.USE_LOCK).symlink_to(external)
    with pytest.raises(OSError):
        attach(other, target)
    assert external.read_text() == 'unchanged'
    assert not other.logical.is_symlink()


@pytest.mark.parametrize('active', [[], {'id': 'bad'}, {'id': 'a' * 32, 'target': '/data', 'mount': '/', 'operation': []}])
def test_malformed_active_config_is_diagnostic_not_an_uncaught_error(setup, active):
    manager, _target = setup
    atomic_json(manager.config_path, {'revision': 1, 'active': active, 'pending': None})
    assert manager.status()['online'] is False
    with pytest.raises(StorageError, match='配置格式'):
        manager.guard()


@pytest.mark.parametrize('busy_area', ['local-download', 'target-maintenance'])
def test_attach_respects_local_downloader_and_shared_directory_maintenance(setup, busy_area):
    manager, target = setup
    activate(manager, target)
    other = another_project(manager)
    other.save_attachment(str(target), 0, manager.config()['active']['id'])
    path = other.logical / '.tushare_refresh.lock' if busy_area == 'local-download' else target / storage.USE_LOCK
    with file_lease(path):
        with pytest.raises(StorageError, match='仍有服务'):
            StorageMigration(other, lambda *_: None).startup()
    assert other.config()['pending']['phase'] == 'PLANNED'
    assert other.logical.is_dir() and not other.logical.is_symlink()
    assert StorageMigration(other, lambda *_: None).startup()['online']


def test_attach_cli_requires_data_id_then_switches_without_download(setup, monkeypatch, capsys):
    from scripts import manage_data_storage as cli
    manager, target = setup
    activate(manager, target)
    other = another_project(manager)
    identifier = manager.config()['active']['id']
    monkeypatch.setattr(cli, 'ROOT', other.project)
    arguments = ['manage_data_storage.py', 'attach', '--path', str(target)]
    monkeypatch.setattr(cli.sys, 'argv', arguments)
    assert cli.main() == 1
    assert identifier in capsys.readouterr().err
    assert other.config()['pending'] is None
    monkeypatch.setattr(cli.sys, 'argv', [*arguments, '--confirm', identifier])
    assert cli.main() == 0
    assert other.guard() == target
    assert 'SUCCESS' in capsys.readouterr().out

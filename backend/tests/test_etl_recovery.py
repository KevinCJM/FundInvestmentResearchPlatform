"""Recovery diagnostics never bypass locks or rewrite historical fingerprints."""
import copy
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.data_sources import etl_service as etl, task_runtime
from backend.data_sources.etl_store import EtlStore
from backend.data_sources.models import CenterError
from backend.services import etl_routes
from backend.services.refresh_runtime import InterProcessFileLock
from backend.tests.test_etl_dataset_tasks import store, small_plan, request, finish, fake_worker


def interrupted_run(store, monkeypatch):
    monkeypatch.setattr(task_runtime, 'run_worker', fake_worker)
    run = finish(store, etl.start(store, request(small_plan())))
    run['status'] = 'INTERRUPTED'
    run['error'] = '服务已退出。已完成步骤保留，可继续未完成步骤。'
    run['steps'][-1]['status'] = 'INTERRUPTED'
    EtlStore(store).save_run(run)
    return run


def test_lock_and_changed_code_reported_together_without_mutation(store, monkeypatch):
    from backend.data_sources import etl_collection
    run=interrupted_run(store, monkeypatch)
    for step in run['steps']:
        step.update(started_at='2026-09-07T01:00:00Z', finished_at='2026-09-07T01:01:00Z')
    EtlStore(store).save_run(run)
    monkeypatch.setattr(etl_collection, 'utc_now', lambda: '2026-09-08T01:00:00Z')
    saved=copy.deepcopy(run['frozen'])
    monkeypatch.setattr(etl, 'execution_fingerprint', lambda:'changed-code')
    lock=InterProcessFileLock(store.root/'.tushare_refresh.lock')
    assert lock.acquire(owner='orphan-worker')
    try:
        status=etl.recovery_status(store,run)
        assert not status['can_resume']
        assert status['warnings'][0]['code'] == 'ETL_CROSS_DATE_RESUME'
        assert [b['code'] for b in status['blockers']] == ['DATA_TASK_RUNNING','ETL_IMPLEMENTATION_CHANGED']
        monkeypatch.setattr(etl, '_launch', lambda *a: pytest.fail('must not launch'))
        with pytest.raises(CenterError) as error:
            etl.resume(store,run['run_id'],True)
        assert '任务锁' in error.value.message and '执行程序已更新' in error.value.message
        assert EtlStore(store).get_run(run['run_id'])['frozen'] == saved
    finally:
        lock.release()


def test_unlocked_unchanged_run_can_resume_and_reuses_successes(store,monkeypatch):
    run=interrupted_run(store,monkeypatch)
    status=etl.recovery_status(store,run)
    assert status == {'can_resume':True,'artifact_check_pending':True,'blockers':[], 'warnings':[]}
    result=finish(store,etl.resume(store,run['run_id'],True))
    assert result['status']=='SUCCEEDED'
    assert result['steps'][0]['attempt']==1
    assert result['steps'][1]['attempt']==2


def test_cross_day_resume_allowed_and_warning_persisted_without_rewriting_frozen(store, monkeypatch):
    from backend.data_sources import etl_collection
    run = interrupted_run(store, monkeypatch)
    for step in run['steps']:
        step.update(started_at='2026-09-07T01:00:00Z', finished_at='2026-09-07T01:01:00Z')
    EtlStore(store).save_run(run)
    frozen = copy.deepcopy(run['frozen'])
    monkeypatch.setattr(etl_collection, 'utc_now', lambda: '2026-09-08T01:00:00Z')
    monkeypatch.setattr(etl, 'utc_now', lambda: '2026-09-08T01:00:00Z')
    status = etl.recovery_status(store, run)
    assert status['can_resume'] and not status['blockers']
    assert status['warnings'][0]['code'] == 'ETL_CROSS_DATE_RESUME'
    result = finish(store, etl.resume(store, run['run_id'], True))
    assert result['status'] == 'SUCCEEDED'
    assert result['frozen'] == frozen and result['options'] == run['options']
    assert result['steps'][0]['attempt'] == 1 and result['steps'][1]['attempt'] == 2
    assert result['resume_events'][0]['warnings'][0]['code'] == 'ETL_CROSS_DATE_RESUME'
    assert result['collection_history'][1]['first_at'].startswith('2026-09-07')
    assert etl_collection.collection_timing(result)['cross_date']


def test_get_api_adds_recovery_conditions_but_does_not_hash_large_artifacts(store,monkeypatch):
    run=interrupted_run(store,monkeypatch)
    before=copy.deepcopy(run)
    monkeypatch.setattr(etl_routes,'get_store',lambda:store)
    monkeypatch.setattr(EtlStore,'checked_path',lambda *a:pytest.fail('GET must not scan large data'))
    monkeypatch.setattr(etl,'execution_fingerprint',lambda:'changed-code')
    app=FastAPI(); app.include_router(etl_routes.router)
    client=TestClient(app)
    for url in ['/api/data-sources/etl/runs',f'/api/data-sources/etl/runs/{run["run_id"]}']:
        response=client.get(url)
        assert response.status_code==200
        value=response.json(); value=value[0] if isinstance(value,list) else value
        assert value['recovery']['can_resume'] is False
        assert '恢复检查' in value['error']
        assert 'frozen' not in value
    assert EtlStore(store).get_run(run['run_id'])['frozen']==before['frozen']


def test_lock_race_is_still_checked_at_write_boundary(store,monkeypatch):
    run=interrupted_run(store,monkeypatch)
    monkeypatch.setattr(etl,'recovery_status',lambda *a:{'can_resume':True})
    lock=InterProcessFileLock(store.root/'.tushare_refresh.lock')
    assert lock.acquire(owner='competing-worker')
    try:
        with pytest.raises(CenterError) as error:
            etl.resume(store,run['run_id'],True)
        assert error.value.code=='DATA_TASK_RUNNING'
        assert EtlStore(store).get_run(run['run_id'])['attempt']==1
    finally:
        lock.release()

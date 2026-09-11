"""Offline collection-clock warnings never certify PIT or weaken recovery gates."""
import copy

import pytest

from backend.data_sources.etl_collection import collection_timing, record_resume, resume_warnings
from backend.data_sources.etl_store import public_run
from backend.data_sources.task_progress import TaskProgressLog


def run_at(first='2026-09-07T15:59:00+00:00', last='2026-09-07T15:59:59+00:00'):
    return {'run_id': 'old', 'attempt': 1, 'status': 'CANCELLED', 'steps': [
        {'id': 'nav', 'name': '净值', 'kind': 'download', 'status': 'CANCELLED', 'attempt': 1,
         'started_at': first, 'finished_at': last,
         'progress': {'collection_window': {'first_at': first, 'last_at': last}}}]}


def test_beijing_midnight_warns_without_modifying_run_or_business_dates():
    run = run_at()
    run['options'] = {'parameters': {'end_date': '20260906'}}
    before = copy.deepcopy(run)
    assert not collection_timing(run)['cross_date']
    assert resume_warnings(run, '2026-09-07T15:59:59Z') == []
    warning = resume_warnings(run, '2026-09-07T16:00:00Z')[0]
    assert warning['code'] == 'ETL_CROSS_DATE_RESUME'
    assert '2026-09-07' in warning['message'] and '2026-09-08' in warning['message']
    assert run == before


def test_continuous_download_crosses_midnight_without_resume():
    value = collection_timing(run_at(last='2026-09-07T16:00:01Z'))
    assert value['cross_date']
    assert value['first_date'] == '2026-09-07' and value['last_date'] == '2026-09-08'
    assert [w['code'] for w in value['warnings']] == ['ETL_CROSS_DATE_COLLECTION']
    assert '同日采集也不代表' in value['boundary']


def test_waiting_and_local_processing_do_not_extend_collection_dates():
    run = run_at()
    run['steps'][0]['finished_at'] = '2026-09-10T18:00:00Z'
    run['steps'][0]['progress']['activity_at'] = '2026-09-10T17:00:00Z'
    # Actual receipts take precedence over a later local merge/cancel timestamp.
    assert not collection_timing(run)['cross_date']
    run['steps'][0]['status'] = 'SUCCEEDED'
    run['steps'].append({'id': 'local', 'name': '映射', 'kind': 'map', 'status': 'FAILED'})
    assert resume_warnings(run, '2026-09-10T18:00:00Z') == []


def test_registered_local_dataset_is_not_a_new_download():
    run = run_at()
    run['steps'][0]['status'] = 'SUCCEEDED'
    run['steps'].append({'id': 'scale', 'name': '本地规模', 'kind': 'task', 'status': 'FAILED',
                         'attempt': 1, 'started_at': '2026-09-08T01:00:00Z'})
    run['frozen'] = {'tasks': {'scale': {'spec': {'network': False}}}}
    assert not collection_timing(run)['cross_date']
    assert resume_warnings(run, '2026-09-08T02:00:00Z') == []


@pytest.mark.parametrize('stamp', [None, 'invalid', '2026-09-07T23:59:00'])
def test_missing_or_naive_old_times_are_not_invented(stamp):
    run = run_at(first=stamp, last=stamp)
    result = collection_timing(run)
    assert result['first_date'] is None
    assert result['windows'][0]['basis'] == 'unknown'
    assert resume_warnings(run)[0]['code'] == 'ETL_COLLECTION_TIME_UNKNOWN'


def test_old_step_times_are_explicit_estimates_and_get_is_read_only():
    run = run_at()
    del run['steps'][0]['progress']
    before = copy.deepcopy(run)
    result = public_run(run, detail=False)['collection_timing']
    assert result['windows'][0]['basis'] == 'execution_window'
    assert result['warnings'][0]['code'] == 'ETL_COLLECTION_TIME_ESTIMATED'
    assert run == before


def test_attempt_and_migration_lineage_survives_repeated_resume():
    run = run_at()
    record_resume(run, '2026-09-08T01:00:00Z')
    original = copy.deepcopy(run['collection_history'])
    run['attempt'] = 2
    step = run['steps'][0]
    step.update(attempt=2, started_at='2026-09-08T01:00:00Z', finished_at='2026-09-08T01:01:00Z', progress={})
    record_resume(run, '2026-09-09T01:00:00Z')
    assert run['collection_history'][:1] == original
    assert len(run['collection_history']) == 2
    assert [e['attempt'] for e in run['resume_events']] == [2, 3]
    assert all(e['warnings'][0]['code'] == 'ETL_CROSS_DATE_RESUME' for e in run['resume_events'])
    # A staged new run retains both stopped partial and completed old attempts.
    migrated = {'run_id': 'new', 'attempt': 0, 'status': 'INTERRUPTED', 'recovered_from': 'old',
                'collection_history': copy.deepcopy(run['collection_history']),
                'steps': [{'id': 'nav', 'name': '净值', 'kind': 'download', 'status': 'PENDING', 'attempt': 0}]}
    assert collection_timing(migrated)['cross_date']
    assert collection_timing(migrated)['windows'][0]['run_id'] == 'old'
    assert len(collection_timing(run)['windows']) == 2  # No duplicate archived attempt.


def test_batch_receipt_clock_is_not_advanced_by_logs_or_local_processing(monkeypatch):
    from backend.data_sources import task_progress
    monkeypatch.setattr(task_progress, 'utc_now', lambda: '2026-09-07T15:59:00Z')
    log = TaskProgressLog()
    log.batch(1)
    monkeypatch.setattr(task_progress, 'utc_now', lambda: '2026-09-07T16:01:00Z')
    log.write('[STAGE] 合并历史数据\n')
    assert log.state['collection_window']['last_at'] == '2026-09-07T15:59:00Z'
    log.batch(0)  # Empty successful responses also have a collection time.
    assert log.state['collection_window'] == {'first_at': '2026-09-07T15:59:00Z', 'last_at': '2026-09-07T16:01:00Z'}

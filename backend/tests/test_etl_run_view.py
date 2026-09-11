import copy
import uuid

from backend.services.etl_run_view import current_run_views


def record(identifier, parent=None, status='FAILED'):
    return {'run_id': identifier, 'recovered_from': parent, 'name': '全数据同步' + ('（恢复）' if parent else ''),
            'status': status, 'attempt': 1, 'created_at': identifier, 'updated_at': identifier,
            'error': 'original failure' if status == 'FAILED' else None,
            'steps': [{'name': '成分与权重', 'status': status}]}


def test_long_chain_is_one_current_card_without_changing_old_records():
    rows = [record(f'{i:03}', f'{i-1:03}' if i else None) for i in range(45)]
    rows[-1]['status'] = 'RUNNING'
    original = copy.deepcopy(rows)
    groups = current_run_views(rows)
    assert len(groups) == 1
    run, history, _ = groups[0]
    assert run['run_id'] == '044'
    assert history['display_name'] == '全数据同步'
    assert history['root_run_id'] == '000'
    assert len(history['records']) == 44 and history['resume_count'] == 44
    assert rows == original


def test_same_names_do_not_merge_independent_tasks_and_branches_remain_visible():
    groups = current_run_views([record('a'), record('b'), record('c', 'a'), record('d', 'a')])
    assert {run['run_id'] for run, _, _ in groups} == {'b', 'c', 'd'}


def test_missing_ancestor_and_cycles_are_visible_and_finite():
    groups = current_run_views([record('a', 'b'), record('b', 'a'), record('c', 'missing')])
    assert len(groups) == 2
    assert all(history['lineage_warning'] for _, history, _ in groups)


def test_active_ancestor_not_hidden_and_same_run_retries_counted():
    rows = [record('a', status='RUNNING'), record('b', 'a')]
    rows[1]['attempt'] = 3
    groups = current_run_views(rows)
    assert {run['run_id'] for run, _, _ in groups} == {'a', 'b'}
    assert next(h for r, h, _ in groups if r['run_id'] == 'b')['resume_count'] == 3


def test_current_route_groups_before_limit_preserves_cancel_and_pending_recovery(tmp_path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from backend.data_sources.store import SourceStore
    from backend.data_sources.etl_store import EtlStore
    from backend.services import etl_routes
    source = SourceStore(tmp_path)
    journal = EtlStore(source)
    rows = [record(uuid.uuid4().hex) for _ in range(35)]
    for index, row in enumerate(rows):
        row['recovered_from'] = rows[index - 1]['run_id'] if index else None
        row['request_hash'] = row['run_id']
        journal.save_run(row, create=True)
    journal.request_cancel(rows[-1]['run_id'])
    monkeypatch.setattr(etl_routes, 'get_store', lambda:source)
    monkeypatch.setattr(etl_routes, '_run_with_recovery', lambda _store, run, **_:copy.deepcopy(run))
    monkeypatch.setattr(etl_routes.etl_recovery, 'job_status', lambda _store, identifier:
                        {'status':'RUNNING', 'id':'migration'} if identifier == rows[-2]['run_id'] else None)
    app = FastAPI(); app.include_router(etl_routes.router)
    client = TestClient(app)
    current = client.get('/api/data-sources/etl/runs?view=current')
    assert current.status_code == 200
    [card] = current.json()
    assert card['run_id'] == rows[-1]['run_id'] and card['cancel_requested'] is True
    assert len(card['history']['records']) == 34
    assert card['recovery']['job']['id'] == 'migration'
    original = client.get('/api/data-sources/etl/runs').json()
    assert len(original) == 30 and all('history' not in r for r in original)
    assert client.get('/api/data-sources/etl/runs?view=unknown').status_code == 422
    assert all('history' not in journal.get_run(r['run_id']) for r in rows)


def test_record_cache_skips_unchanged_bodies_but_refreshes_live_state(tmp_path, monkeypatch):
    from contextlib import contextmanager
    from backend.services.etl_run_view import read_run_records
    from backend.data_sources.store import SourceStore
    from backend.data_sources.etl_store import EtlStore
    source = SourceStore(tmp_path); journal = EtlStore(source)
    old = record('old'); old['request_hash'] = 'old'
    journal.save_run(old, create=True)
    original = source.connection; statements = []
    @contextmanager
    def trace():
        with original() as db:
            db.set_trace_callback(statements.append); yield db
    monkeypatch.setattr(source, 'connection', trace)
    assert read_run_records(source)[0]['status'] == 'FAILED'
    with original() as db:
        plan = db.execute('EXPLAIN QUERY PLAN SELECT id,updated_at,cancel_requested FROM etl_run').fetchall()
        assert any('COVERING INDEX etl_run_view_revision' in r[3] for r in plan)
    statements.clear()
    read_run_records(source)
    assert not any('SELECT body' in s for s in statements)
    journal.request_cancel('old')
    assert read_run_records(source)[0]['cancel_requested'] is True
    old['status'] = 'SUCCEEDED'; journal.save_run(old)
    assert read_run_records(source)[0]['status'] == 'SUCCEEDED'
    new = record('new', 'old'); new['request_hash'] = 'new'; journal.save_run(new, create=True)
    assert len(read_run_records(source)) == 2
    with original() as db: db.execute("DELETE FROM etl_run WHERE id='old'")
    assert [r['run_id'] for r in read_run_records(source)] == ['new']

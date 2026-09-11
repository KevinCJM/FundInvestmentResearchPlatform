"""Offline graph failure propagation, explicit workspaces and safe retries."""
import copy
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from backend.data_sources import etl_service as etl, task_runtime
from backend.data_sources.etl_models import EtlDefinition
from backend.data_sources.etl_store import EtlStore
from backend.data_sources.models import CenterError
from backend.data_sources.task_workspace import merge_inventories, inventory, read_inventory
from backend.data_sources.etl_dependencies import plan_dependencies
from backend.tests.test_etl_dataset_tasks import small_plan, request, finish, fake_worker
from backend.tests import test_etl_dataset_tasks as dataset_fixtures

store = dataset_fixtures.store


def branching():
    definition = small_plan()
    first, second = definition['steps']
    dependent = {**copy.deepcopy(second), 'id': 'dependent', 'name': '硬依赖'}
    second.update(inputs=[], after=['calendar'])
    tail = {**copy.deepcopy(second), 'id': 'tail', 'name': '阻断节点之后', 'after': ['dependent']}
    definition.update(graph_version=1, steps=[first, dependent, second, tail])
    return definition


def test_failed_data_blocks_descendants_but_order_only_continues_and_resume_reuses_success(store, monkeypatch):
    calls = []
    fail = [True]
    def worker(payload, check, lock):
        calls.append(payload['task_id'])
        if payload['task_id'] == 'tushare.calendar' and fail[0]:
            raise CenterError('SOURCE_TIMEOUT', '测试网络超时')
        return fake_worker(payload, check, lock)
    monkeypatch.setattr(task_runtime, 'run_worker', worker)
    run = finish(store, etl.start(store, request(branching())))
    assert run['status'] == 'FAILED' and run['code'] == 'ETL_PARTIAL_FAILURE'
    assert [s['status'] for s in run['steps']] == ['FAILED', 'SKIPPED', 'SUCCEEDED', 'SUCCEEDED']
    assert run['steps'][1]['blocked_by'] == ['calendar']
    assert len(calls) == 3
    fail[0] = False
    resumed = finish(store, etl.resume(store, run['run_id'], True))
    assert resumed['status'] == 'SUCCEEDED', resumed
    assert len(calls) == 5  # Two independent successful nodes are not repeated.
    assert 'failure_summary' not in resumed
    assert 'blocked_by' not in resumed['steps'][1]
    assert not (store.root / 'tushare_active.json').exists()


@pytest.mark.parametrize('code', ['ETL_CANCELLED', 'ETL_TIMEOUT', 'ETL_CONFIG_CHANGED', 'SOURCE_DB_FULL', 'STORAGE_OFFLINE', 'STORAGE_LOW_SPACE', 'STORAGE_IDENTITY_CHANGED'])
def test_global_safety_failure_never_continues(store, monkeypatch, code):
    calls = []
    def worker(*args):
        calls.append(1)
        raise CenterError(code, '全局停止')
    monkeypatch.setattr(task_runtime, 'run_worker', worker)
    run = finish(store, etl.start(store, request(branching())))
    assert len(calls) == 1 and run['status'] != 'SUCCEEDED'
    assert all(s['status'] == 'SKIPPED' for s in run['steps'][1:])


def test_multiple_inputs_materialize_and_incremental_keeps_each_branch(store, monkeypatch):
    definition = small_plan()
    first, second = definition['steps']
    second.update(inputs=[], after=[first['id']])
    third = {**copy.deepcopy(second), 'id':'third', 'name':'合并', 'inputs':[first['id'], second['id']], 'after':[]}
    definition.update(graph_version=1, steps=[first, second, third])
    payloads = []
    def worker(payload, check, lock):
        payloads.append(payload)
        if Path(payload['directory']).parent.name == 'third':
            assert (Path(payload['directory']) / 'calendar.parquet').exists()
            assert (Path(payload['directory']) / 'fund_company.parquet').exists()
        return fake_worker(payload, check, lock)
    monkeypatch.setattr(task_runtime, 'run_worker', worker)
    result = finish(store, etl.start(store, request(definition)))
    assert result['status'] == 'SUCCEEDED', result
    for step in definition['steps']:
        step['params']['end_date'] = '20240110'
    later = finish(store, etl.start(store, request(definition, 'incremental')))
    assert later['status'] == 'SUCCEEDED', later
    assert len(later['frozen']['task_baseline']['workspaces']) == 3
    assert all(p['has_baseline'] for p in payloads[3:])


def test_merge_conflicts_fail_closed_and_descendant_wins_independent_of_order(store):
    journal = EtlStore(store)
    def output(name, value, before=None):
        path = store.root / name; path.mkdir()
        pq.write_table(pa.table({'v':[value]}), path / 'values.parquet')
        return inventory(journal, path, path / 'workspace.json', [], 'tushare', before=before)
    base = output('base', 1)
    updated = output('updated', 2, read_inventory(journal, base))
    conflict = output('conflict', 3)
    assert merge_inventories(journal, [base, updated]) == merge_inventories(journal, [updated, base])
    with pytest.raises(CenterError, match='不同版本'):
        merge_inventories(journal, [updated, conflict])


def test_dependency_planner_separates_sequence_and_required_capabilities(store):
    from backend.data_sources.etl_templates import tushare_all_data_workflow
    result = tushare_all_data_workflow(store)
    steps = {s.task_id:s for s in result.steps}
    assert result.graph_version == 1
    assert not steps['tushare.macro_cycle'].inputs
    assert steps['tushare.macro_cycle'].after == ['dataset_index_coverage']
    assert 'dataset_index_constituents' not in steps['tushare.index_coverage'].inputs
    assert 'dataset_index_domestic' in steps['tushare.index_coverage'].inputs
    assert set(steps['tushare.nav'].inputs) == {'dataset_etf_info', 'dataset_calendar'}
    assert etl.validate(store, result.model_dump(mode='json'))['valid']
    # Explicit conversion, not mutation of the old persisted document.
    old = EtlDefinition.model_validate(small_plan())
    newer = plan_dependencies(old)
    assert old.steps[1].inputs == ['calendar'] and not newer.steps[1].inputs


def test_required_data_cannot_be_replaced_by_order_only(store):
    definition = small_plan()
    definition.update(graph_version=1)
    definition['steps'][1].update(task_id='tushare.nav', inputs=[], after=['calendar'])
    assert not etl.validate(store, definition)['valid']


def test_dependency_plan_endpoint_validates_without_saving_or_starting(store, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from backend.services import etl_routes
    monkeypatch.setattr(etl_routes, 'get_store', lambda: store)
    app = FastAPI(); app.include_router(etl_routes.router)
    client = TestClient(app)
    path = '/api/data-sources/etl/dependencies/plan'
    definition = small_plan()
    response = client.post(path, json={'definition':definition})
    assert response.status_code == 200, response.text
    assert response.json()['definition']['steps'][1]['inputs'] == []
    assert response.json()['definition']['steps'][1]['after'] == ['calendar']
    assert response.json()['published'] is False
    assert not EtlStore(store).runs() and not EtlStore(store).workflows()
    assert client.post(path, json={'definition':definition}, headers={'Origin':'https://bad.invalid'}).status_code == 403

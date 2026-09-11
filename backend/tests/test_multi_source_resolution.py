"""Offline canonical arbitration, integrity and immutable replay regressions."""
from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import sys

import pytest
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from backend.data_sources.batches import capture_batch
from backend.data_sources.mapping import map_table
from backend.data_sources.models import CenterError
from backend.data_sources.presets import default_interfaces
from backend.data_sources.resolution import resolve_records
from backend.data_sources.resolution_models import ResolutionConfig
from backend.data_sources.resolution_store import get_policy, resolve_saved, save_policy
from backend.data_sources.store import SourceStore

PRESET = next(i for i in default_interfaces() if i.api_name == 'fund_daily')


def quote(source='tushare', **changes):
    raw = dict(ts_code='510300.SH', trade_date='20240102', open=10., high=12., low=9., close=11., pre_close=10., vol=2., amount=2., pct_chg=10.)
    table, errors = map_table([raw], PRESET.mappings[0], source, 'batch-' + source)
    assert not errors
    row = table.to_pylist()[0]
    row.update(ingested_at=datetime(2024, 1, 3, tzinfo=timezone.utc))
    row.update(changes)
    return row


def configuration(**rule):
    payload = ResolutionConfig().model_dump()
    payload['tables'][0].update(rule)
    return ResolutionConfig.model_validate(payload)


def resolve(rows, config=None, **kwargs):
    return resolve_records('market.quote_daily', rows, config or configuration(), **kwargs)


def test_priority_agreement_and_deterministic_order():
    rows = [quote('akshare'), quote()]
    result = resolve(rows)
    assert result['rows'][0]['source_id'] == 'tushare'
    assert result['decisions'][0]['status'] == 'SELECTED'
    assert resolve(rows[::-1])['decisions'] == result['decisions']
    assert result['execution'] == dict(complete=True, python_fallback=0, request_time_compilation=0)


@pytest.mark.parametrize('allowed', [False, True])
def test_missing_source_switch(allowed):
    result = resolve([quote('akshare')], configuration(fallback_on_missing=allowed))
    assert len(result['rows']) == int(allowed)
    assert result['decisions'][0]['status'] == ('FALLBACK' if allowed else 'BLOCKED')


@pytest.mark.parametrize('invalid', [0, -1, float('inf'), float('nan'), True, 'bad'])
@pytest.mark.parametrize('allowed', [False, True])
def test_invalid_source_switch(invalid, allowed):
    result = resolve([quote(close=invalid), quote('akshare')], configuration(fallback_on_invalid=allowed))
    assert len(result['rows']) == int(allowed)
    if allowed:
        assert result['rows'][0]['source_id'] == 'akshare'
        assert result['decisions'][0]['skipped'][0]['reasons']


def test_missing_value_is_not_invalid_value_policy():
    result = resolve([quote(close=None), quote('akshare')], configuration(fallback_on_missing=True, fallback_on_invalid=False))
    assert result['decisions'][0]['status'] == 'FALLBACK'


@pytest.mark.parametrize('policy,status', [('quarantine','CONFLICT'), ('prefer_priority','WARNING')])
def test_legal_conflict_never_averages(policy, status):
    result = resolve([quote(), quote('akshare', close=11.5)], configuration(conflict_action=policy))
    assert result['decisions'][0]['status'] == status
    assert result['decisions'][0]['conflicts'][0]['fields'] == ['close']
    if result['rows']:
        assert result['rows'][0]['close'] == 11.


def test_per_field_tolerance_and_range():
    config = configuration(field_rules=[dict(field='close', absolute_tolerance=.6, relative_tolerance=0)])
    assert resolve([quote(), quote('akshare', close=11.5)], config)['decisions'][0]['status'] == 'SELECTED'
    config = configuration(fallback_on_invalid=True, field_rules=[dict(field='close', maximum=11.2)])
    assert resolve([quote(close=11.5), quote('akshare')], config)['rows'][0]['source_id'] == 'akshare'


@pytest.mark.parametrize('changes', [dict(high=8), dict(low=13), dict(volume=-1), dict(currency='INVALID'), dict(quote_type='WRONG')])
def test_quality_and_schema_fail_closed(changes):
    result = resolve([quote(**changes), quote('akshare')], configuration(fallback_on_invalid=True))
    assert not any(r['source_id'] == 'tushare' and r.get('quote_type') == 'WRONG' for r in result['rows'])
    if 'currency' not in changes:
        assert result['rows'][0]['source_id'] == 'akshare'


def test_different_dates_currency_and_adjustment_are_not_spliced():
    rows = [quote(close=None), quote('akshare', adjustment_basis='FORWARD'), quote('akshare', currency='USD'), quote('akshare', trade_date='2024-01-03')]
    result = resolve(rows)
    assert not any(r['currency']=='CNY' and r['adjustment_basis']=='RAW' and str(r['trade_date'])=='2024-01-02' for r in result['rows'])


def test_table_override_and_source_exclusion():
    config = configuration(source_priority=['akshare'])
    result = resolve([quote(), quote('akshare')], config)
    assert result['rows'][0]['source_id'] == 'akshare'
    assert result['excluded'][0]['reason'] == 'SOURCE_NOT_SELECTED'


def test_same_source_same_revision_conflict_and_latest_revision():
    first = quote()
    other = quote(close=11.5)
    result = resolve([first, other, quote('akshare')], configuration(fallback_on_invalid=True))
    assert result['rows'][0]['source_id'] == 'akshare'
    assert 'SAME_SOURCE_CONFLICT' in result['decisions'][0]['skipped'][0]['reasons']
    other['revision'] = 2
    assert resolve([first, other])['rows'][0]['close'] == 11.5


def test_pit_filters_before_latest_and_rejects_backdated_new_capture():
    future = quote(close=11.5, revision=2, ingested_at=datetime(2024,2,1,tzinfo=timezone.utc), available_at=datetime(2024,1,2,tzinfo=timezone.utc), availability_status='EXACT')
    result = resolve([quote(), future], as_of='2024-01-10T00:00:00Z')
    assert result['rows'][0]['close'] == 11.
    assert result['excluded'][0]['reason'] == 'NOT_KNOWN_AS_OF'
    assert resolve([quote()], as_of='2024-01-01T00:00:00Z')['rows'] == []


def test_jump_flag_is_not_silently_overridden():
    result = resolve([quote(previous_close=1), quote('akshare')], configuration(max_relative_jump=.5, fallback_on_invalid=True))
    assert result['rows'][0]['source_id'] == 'akshare'
    assert 'SUSPICIOUS_JUMP' in result['decisions'][0]['skipped'][0]['reasons']


def test_all_bad_and_empty_inputs_never_fill_zero():
    assert resolve([])['rows'] == []
    assert resolve([quote(close=0), quote('akshare', close=0)], configuration(fallback_on_invalid=True))['rows'] == []


def test_internal_targets_and_unknown_fields_refused():
    with pytest.raises(CenterError):
        resolve_records('master.instrument_identifier', [], configuration())
    with pytest.raises(CenterError):
        resolve([quote(injected_field=1)])


def test_policy_cas_and_immutable_replay(tmp_path):
    store = SourceStore(tmp_path)
    store.seed()
    saved = get_policy(store)
    new = save_policy(store, saved['config'], saved['revision'])
    with pytest.raises(CenterError, match='修改'):
        save_policy(store, saved['config'], saved['revision'])
    for source in ('tushare', 'akshare'):
        interface = PRESET.model_copy(deep=True)
        interface.source_id = source
        interface.id = source + '.quotes'
        capture_batch(store, interface, [dict(ts_code='510300.SH',trade_date='20240102',open=10,high=12,low=9,close=11,vol=2,amount=2)], {}, source)
    result = resolve_saved(store, 'market.quote_daily', new['revision'])
    assert result['status'] == 'CANDIDATE_READY'
    assert result['published'] is False
    path = tmp_path / result['artifact']
    before = path.read_bytes()
    assert pq.read_table(path).num_rows == 1
    assert resolve_saved(store, 'market.quote_daily', new['revision']) == result
    assert path.read_bytes() == before
    manifest = json.loads((path.parent/'manifest.json').read_text())
    assert len(manifest['snapshot']['inputs']) == 2
    assert not (tmp_path/'tushare_active.json').exists()


def test_legacy_candidates_are_rejection_markers_not_trusted_values():
    old = quote(source_batch_id='old', close=12.)
    secondary = quote('akshare')
    blocked = resolve_records('market.quote_daily', [old, secondary], configuration(), unverified_batches=frozenset({'old'}))
    assert blocked['rows'] == []
    fallback = resolve_records('market.quote_daily', [old, secondary], configuration(fallback_on_invalid=True), unverified_batches=frozenset({'old'}))
    assert fallback['rows'][0]['source_id'] == 'akshare'
    assert 'LEGACY_CANDIDATE_UNVERIFIED' in fallback['decisions'][0]['skipped'][0]['reasons']
    refreshed = quote(source_batch_id='new')
    repaired = resolve_records('market.quote_daily', [old, refreshed, secondary], configuration(), unverified_batches=frozenset({'old'}))
    assert repaired['rows'][0]['source_batch_id'] == 'new'


def test_old_batch_without_checksum_does_not_prevent_verified_refresh(tmp_path):
    store = SourceStore(tmp_path); store.seed()
    raw = [dict(ts_code='510300.SH', trade_date='20240102', close=11.)]
    batch = capture_batch(store, PRESET, raw, {}, 'legacy')
    batch['tables'][0].pop('checksum')
    with store.connection() as db:
        db.execute('UPDATE source_run SET result=? WHERE id=?', (json.dumps(batch), batch['batch_id']))
    revision = get_policy(store)['revision']
    rejected = resolve_saved(store, 'market.quote_daily', revision)
    assert rejected['status'] == 'NEEDS_REVIEW'
    assert rejected['summary']['selected_rows'] == 0
    capture_batch(store, PRESET, raw, {}, 'refreshed')
    repaired = resolve_saved(store, 'market.quote_daily', revision)
    assert repaired['status'] == 'CANDIDATE_READY'
    assert repaired['summary']['selected_rows'] == 1
    assert repaired['summary']['unverified_input_batches'] == 1


def test_candidate_tampering_is_detected(tmp_path):
    store = SourceStore(tmp_path); store.seed()
    batch = capture_batch(store, PRESET, [dict(ts_code='510300.SH',trade_date='20240102',close=11)], {}, 'test')
    path = tmp_path / batch['tables'][0]['artifact']
    values = pq.read_table(path).to_pydict(); values['close'] = [9.]
    import pyarrow as pa
    pq.write_table(pa.table(values), path)
    with pytest.raises(CenterError, match='校验'):
        resolve_saved(store, 'market.quote_daily', get_policy(store)['revision'])

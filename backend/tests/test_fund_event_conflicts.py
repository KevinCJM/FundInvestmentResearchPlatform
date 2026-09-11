"""Offline conflict provenance, downstream gates and restart contracts."""
import json

import pandas as pd
import pytest

import T01_get_data as script
from backend.data_sources import fund_event_conflicts as conflicts
from backend.data_sources.models import CenterError
from backend.data_sources.task_workspace import inventory, materialize, merge_inventories
from backend.data_sources.auto_baseline import candidate_inventory
from backend.market_data import validate_snapshot_directory, MarketDataManifestError
from backend.tests.test_fund_event_recovery import fixture
from backend.data_sources.fund_event_recovery import import_history_receipts
from backend.data_sources.fund_events import FundEventDownload


def conflict_checkpoint(tmp_path):
    journal, parts, dest, universe, old, frame, step = fixture(tmp_path)
    rows = pd.DataFrame([dict(ts_code='A', ann_date='20240102', end_date='20231231',
                             symbol='600000.SH', mkv=v, amount=v) for v in (1., 2.)])
    def prepare(value):
        return script._prepare_fund_event_rows(value, fields=old.fields,
                    source_api='fund_portfolio', observation_column='end_date')
    frame = old._prepare(rows, '20240101-20240102', 'A', prepare)
    path = old._paths('20240101-20240102', 'A')[0]
    script.save_dataframe(frame, path, quiet=True)
    old._record('20240101-20240102', 'A', 'COMPLETE', rows=len(frame), sha256=journal.artifact(path)['checksum'])
    return journal, parts, dest, universe, old, frame, step


def test_conflict_receipt_migrates_with_raw_variants_and_no_new_requests(tmp_path):
    journal, parts, dest, universe, old, frame, step = conflict_checkpoint(tmp_path)
    from backend.data_sources.etl_partial_recovery import _check_event_layout
    (old.directory / 'unfinished.parquet.tmp').unlink()
    _check_event_layout(parts)
    result = import_history_receipts(journal, parts, dest, step)
    assert result['complete'] == 1
    new = FundEventDownload(directory=dest, dates=old.dates, universe=universe,
        api_name='fund_portfolio', fields=old.fields, smoke=False, strategy='fund_announcement_history')
    record = new._cached('20240101-20240102', 'A')
    assert record['quality_status'] == 'CONFLICTED'
    assert len(new.conflicts) == 1
    evidence = conflicts.checked_part(new.directory, record['conflict_evidence'])
    assert set(evidence.mkv) == {1., 2.}
    for item in result['files']:
        assert journal.checked_path(item['source']).read_bytes() == journal.checked_path(item['imported']).read_bytes()


@pytest.mark.parametrize('change', ['missing_evidence', 'altered_evidence', 'invented_value', 'hidden_status'])
def test_invalid_conflict_evidence_blocks_recovery(tmp_path, change):
    journal, parts, dest, universe, old, frame, step = conflict_checkpoint(tmp_path)
    evidence = next(iter(old.conflicts.values()))
    if change == 'missing_evidence':
        (old.directory / evidence['path']).unlink()
    elif change == 'altered_evidence':
        (old.directory / evidence['path']).write_bytes(b'corrupt')
    else:
        if change == 'invented_value': frame['mkv'] = 1.
        else: frame['availability_status'] = 'announced_date'
        path = old._paths('20240101-20240102', 'A')[0]
        script.save_dataframe(frame, path, quiet=True)
        old._record('20240101-20240102', 'A', 'COMPLETE', rows=1, sha256=journal.artifact(path)['checksum'])
    with pytest.raises(CenterError):
        import_history_receipts(journal, parts, dest, step)


def test_quarantine_inventory_prevents_materialization_and_activation(tmp_path):
    journal, _, _, _, session, frame, _ = conflict_checkpoint(tmp_path)
    work = tmp_path / 'candidate'
    work.mkdir()
    path = work / 'fund_portfolio_df.parquet'
    script.save_dataframe(frame, path, quiet=True)
    issue = conflicts.finish(path, session)
    artifact = inventory(journal, work, tmp_path / 'inventory.json', ['fund_portfolio'], 'tushare')
    merged = merge_inventories(journal, [artifact])
    assert merged['quality_issues'][0]['conflicting_keys'] == 1
    with pytest.raises(CenterError, match='数值冲突'):
        materialize(journal, artifact, tmp_path / 'consumer')
    assert not list((tmp_path / 'consumer').glob('*.parquet'))
    with pytest.raises(MarketDataManifestError, match='冲突'):
        validate_snapshot_directory(work, required_files=['fund_portfolio_df.parquet'])
    run = {'status': 'SUCCEEDED', 'steps': [{'output': {'data_quality': issue}}]}
    with pytest.raises(CenterError, match='冲突'):
        candidate_inventory(None, run)
    metadata = path.with_name(path.name + '.quality.meta.json')
    value = json.loads(metadata.read_text()); value['data_checksum'] = '0' * 64
    metadata.write_text(json.dumps(value))
    with pytest.raises(CenterError, match='不匹配'):
        inventory(journal, work, tmp_path / 'invalid.json', ['fund_portfolio'], 'tushare')


def test_normal_unpaged_values_unchanged_but_conflicts_need_stable_recheck(tmp_path):
    _, _, _, _, session, _, _ = conflict_checkpoint(tmp_path)
    raw = pd.DataFrame([dict(ts_code='A', ann_date='20240102', end_date='20231231', symbol='X', mkv=v) for v in (1., 2.)])
    session._confirm_unpaged_conflicts(raw, lambda: raw.copy())
    with pytest.raises(CenterError, match='发生变化'):
        session._confirm_unpaged_conflicts(raw, lambda: raw.iloc[:1])
    session._confirm_unpaged_conflicts(raw.iloc[:1], lambda: pytest.fail('No needless verification'))

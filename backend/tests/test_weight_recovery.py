"""Offline proof of old successful weights; no supplier calls or active writes."""
import json
from types import SimpleNamespace

import pandas as pd
import pytest

import T01_get_data as script
from backend.data_sources.etl_store import EtlStore
from backend.data_sources.models import CenterError
from backend.data_sources.store import SourceStore
from backend.data_sources.weight_recovery import import_weights


@pytest.mark.parametrize('case', ['valid', 'empty', 'wrong_value', 'checksum', 'orphan', 'gap', 'unknown', 'contradiction'])
def test_weight_import_requires_verified_raw_and_complete_latest_window(tmp_path, monkeypatch, case):
    store = SourceStore(tmp_path); store.seed(); journal = EtlStore(store)
    source, target = tmp_path / 'old', tmp_path / 'new'
    source.mkdir()
    options = SimpleNamespace(start_date='20260901', end_date='20260906', resume=True)
    rows = [] if case in {'empty', 'contradiction'} else [{'index_code':'A.SH', 'con_code':'600000.SH', 'trade_date':'20260904', 'weight':1.0}]
    monkeypatch.setattr(script, 'fetch_constituent_pages', lambda *a, **kw: pd.DataFrame(rows))
    raw = script.cached_constituent_request(object(), 'index_weight', None, options, source / 'queries',
            index_code='A.SH', start_date='20260901', end_date='20260904' if case == 'gap' else '20260906')
    if case == 'empty':
        script.mark_empty_checkpoint(source / 'A.SH.empty')
    else:
        frame = script._normalise_member_frame(raw, 'index_weight', 'A.SH')
        if case in {'wrong_value', 'contradiction'}: frame['weight'] = 99.0
        frame.to_parquet(source / 'A.SH.parquet')
    if case == 'checksum': next((source / 'queries').glob('*.parquet')).write_bytes(b'corrupt')
    if case == 'orphan': next((source / 'queries').glob('*.json')).unlink()
    if case == 'unknown': (source / 'other.txt').write_text('unknown')
    step = SimpleNamespace(params={'start_date':'20260901', 'end_date':'20260906'})
    catalog = pd.DataFrame([{'ts_code':'A.SH', 'quote_source_api':'index_daily', 'status':'active'}])
    if case not in {'valid', 'empty'}:
        with pytest.raises(CenterError): import_weights(journal, source, target, step, catalog, lambda _: None)
    else:
        result = import_weights(journal, source, target, step, catalog, lambda _: None)
        assert result['copied'] == result['queries'] == 1 and len(result['files']) == 3
        for evidence in result['files']:
            assert evidence['source']['checksum'] == evidence['imported']['checksum']
        assert json.loads(next((target / 'queries').glob('*.json')).read_text())['request']['params']['end_date'] == '20260906'

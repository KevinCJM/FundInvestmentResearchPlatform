"""Actual NJIT-backed snapshot integration over isolated canonical fixtures."""
from __future__ import annotations

import json
import sys
from pathlib import Path
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
if str(ROOT / 'backend') not in sys.path: sys.path.insert(0, str(ROOT / 'backend'))
from backend.data_sources.etl_snapshot import build_snapshot
from backend.data_sources.models import CenterError


def inputs(tmp_path):
    directory = tmp_path / 'canonical'; directory.mkdir()
    master = pd.DataFrame([{'instrument_id':'internal-etf-001','canonical_name':'测试ETF','instrument_type':'ETF','valid_from':'2024-01-01'}])
    nav = pd.DataFrame({'instrument_id':['internal-etf-001']*8, 'valuation_date':pd.date_range('2024-01-01',periods=8), 'adjusted_nav':[1.,1.01,1.02,1.03,1.04,1.05,1.06,1.07], 'unit_nav':[.8]*8, 'accumulated_nav':[None]*8, 'announced_at':pd.date_range('2024-01-02',periods=8,tz='UTC'), 'available_at':pd.date_range('2024-01-02',periods=8,tz='UTC'), 'availability_status':['EXACT']*8, 'instrument_type':['ETF']*8})
    master.to_parquet(directory/'master.parquet'); nav.to_parquet(directory/'nav.parquet')
    config=tmp_path/'config'; config.mkdir()
    (config/'snapshot_indicator_config.json').write_text(json.dumps({'schema_version':1,'revision':7,'items':[{'indicator_id':'builtin-total-return-v2','indicator_revision':1,'period':'ALL','field':'research_return'}]}))
    return {'master.instrument':directory/'master.parquet','market.nav_daily':directory/'nav.parquet'}, config


def test_real_snapshot_uses_canonical_nav_and_locked_metric_config(tmp_path):
    paths, config = inputs(tmp_path)
    result = build_snapshot(tmp_path/'private_snapshot', paths, config)
    assert result['rows'] == 1
    frame = pd.read_parquet(result['path'])
    assert frame.iloc[0]['latest_adj_nav'] == pytest.approx(1.07)
    assert frame.iloc[0]['ts_code'] == 'internal-etf-001'
    assert frame.iloc[0]['research_return'] == pytest.approx(.07)
    assert result['snapshot_indicators']['config_revision'] == 7
    assert result['execution']['python_fallback'] == 0
    assert not (tmp_path/'tushare_active.json').exists()


def test_snapshot_worker_preserves_frozen_config_and_ignores_active_override(tmp_path, monkeypatch):
    import shutil
    from backend.data_sources.etl_service import _snapshot_process
    from backend.data_sources.etl_store import EtlStore
    from backend.data_sources.store import SourceStore
    paths, config = inputs(tmp_path)
    journal = EtlStore(SourceStore(tmp_path))
    frozen = tmp_path / 'etl_runs' / 'run1' / 'config'
    shutil.copytree(config, frozen)
    artifact = journal.artifact(frozen / 'snapshot_indicator_config.json')
    run = {'run_id':'run1','frozen':{'config_artifacts':[artifact]}}
    output_dir = tmp_path / 'etl_runs' / 'run1' / 'snapshot'; output_dir.mkdir()
    unrelated = tmp_path / 'wrong_market'; unrelated.mkdir()
    monkeypatch.setenv('TUSHARE_DATA_DIR', str(unrelated))
    result = _snapshot_process(journal, run, output_dir, paths, lambda: None)
    assert result['rows'] == 1
    assert pd.read_parquet(result['path']).iloc[0]['research_return'] == pytest.approx(.07)
    assert journal.checked_path(artifact).is_file()
    assert not (frozen / 'custom_indicators.json').exists()
    assert (output_dir / 'runtime_config' / 'snapshot_indicator_config.json').exists()
    assert result['snapshot_indicators']['config_revision'] == 7


def test_snapshot_rejects_unknown_product_instead_of_using_active_data(tmp_path):
    paths, config = inputs(tmp_path)
    master = pd.read_parquet(paths['master.instrument']); master['instrument_id']='other'; master.to_parquet(paths['master.instrument'])
    with pytest.raises(CenterError, match='主数据'):
        build_snapshot(tmp_path/'snapshot', paths, config)


def test_unit_nav_never_becomes_adjusted_nav(tmp_path):
    paths, config = inputs(tmp_path)
    nav = pd.read_parquet(paths['market.nav_daily']); nav['adjusted_nav']=None; nav.to_parquet(paths['market.nav_daily'])
    with pytest.raises(CenterError, match='复权净值'):
        build_snapshot(tmp_path/'snapshot', paths, config)

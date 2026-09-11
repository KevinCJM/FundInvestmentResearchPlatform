"""Merrill template, macro acquisition timing and complete preview regression."""
import copy
import time

import numpy as np
import pandas as pd
import pytest

from custom_indicators.errors import ValidationError
from historical_regimes.v2_contracts import parse_definition_v2
from historical_regimes.v2_numba import KERNELS
from historical_regimes.v2_service import RegimeGraphV2Service, _macro_bundle
from historical_regimes.v2_templates import get_template_v2


def write_macro(root, *, known=False):
    dates = pd.date_range('2020-01-31', periods=18, freq='ME')
    for filename, field, values in (
        ('macro_cn_pmi_df.parquet', 'pmi010000', [50, 51, 52, 53, 54, 55, 54, 53, 52, 51, 50, 49, 50, 51, 52, 53, 54, 55]),
        ('macro_cn_cpi_df.parquet', 'nt_yoy', [6, 5, 4, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 2, 3, 4, 5]),
    ):
        pd.DataFrame({'observation_date': dates, field: values,
                      'available_at': dates + pd.Timedelta(days=10) if known else None,
                      'availability_status': 'release_date_known' if known else 'release_date_unknown',
                      'vintage': '2022-01-01T12:00:00+00:00', 'revision': 1}).to_parquet(root / filename)


def test_catalog_filename_and_unknown_release_contract(tmp_path):
    write_macro(tmp_path)
    spec = {'dataset': 'macro_cn_pmi_df.parquet', 'field': 'pmi010000'}
    source_bytes = (tmp_path / spec['dataset']).read_bytes()
    bundle = _macro_bundle(spec, 'retrospective', None, tmp_path)
    assert bundle.snapshot['release_dates_verified'] is False
    assert set(bundle.frame.available_at) == {pd.Timestamp('2022-01-01')}
    assert len(bundle.frame) == 18
    with pytest.raises(ValidationError, match='发布日期未知'):
        _macro_bundle(spec, 'realtime', None, tmp_path)
    with pytest.raises(ValidationError, match='尚无可得数据'):
        _macro_bundle(spec, 'retrospective', '2021-12-31', tmp_path)
    assert source_bytes == (tmp_path / spec['dataset']).read_bytes()


def test_macro_revisions_obey_cutoff_and_keep_missing_values(tmp_path):
    filename = 'macro_cn_cpi_df.parquet'
    pd.DataFrame({'observation_date': ['2020-01-31'] * 2 + ['2020-02-29'],
                  'available_at': [None] * 3, 'nt_yoy': [1., 2., np.nan],
                  'vintage': ['2020-02-10', '2020-04-10', '2020-03-10'],
                  'revision': [1, 2, 1]}).to_parquet(tmp_path / filename)
    spec = {'dataset': filename, 'field': 'nt_yoy'}
    before = _macro_bundle(spec, 'retrospective', '2020-03-31', tmp_path)
    after = _macro_bundle(spec, 'retrospective', None, tmp_path)
    assert before.frame.value.iloc[0] == 1.
    assert after.frame.value.iloc[0] == 2.
    assert np.isnan(before.frame.value.iloc[1])


@pytest.mark.parametrize('mode', ['retrospective', 'realtime'])
def test_merrill_step_template_executes_all_quadrants_with_njit(tmp_path, mode):
    write_macro(tmp_path, known=mode == 'realtime')
    service = RegimeGraphV2Service(tmp_path, tmp_path)
    item = get_template_v2('merrill-clock-macro-v3-steps-v1')
    assert item['default_mode'] == 'retrospective'
    definition = item['definition']
    assert not any(node['type'] in {'source.inline', 'model.quadrant'} for node in definition['graph']['nodes'])
    signatures = {key: list(kernel.signatures) for key, kernel in KERNELS.items()}
    prepared = service.prepare(definition)
    started = service.create_preview(definition, compile_token=prepared['compile_token'], mode=mode)
    for _ in range(200):
        job = service.get_preview(started['id'])
        if job['status'] in {'completed', 'failed'}:
            break
        time.sleep(.01)
    assert job['status'] == 'completed', job
    rows = service.preview_series(started['id'], limit=100)['items']
    assert len(rows) == 18
    assert {row['state_id'] for row in rows if row['state_code'] >= 0} == {'recovery', 'overheat', 'stagflation', 'recession'}
    assert rows[0]['state_id'] == rows[1]['state_id'] == 'unclassified'
    assert signatures == {key: list(kernel.signatures) for key, kernel in KERNELS.items()}
    if mode == 'retrospective':
        assert not any(row['executable'] for row in rows)
    # Prefixes cannot alter an already confirmed state with known release dates.
    if mode == 'realtime':
        full = service._execute_graph(None, parse_definition_v2(definition), mode, None)
        prefix = service._execute_graph(None, parse_definition_v2(definition), mode, '2020-12-15')
        assert [x['state_id'] for x in prefix['series']] == [x['state_id'] for x in full['series'][:11]]


def test_historical_placeholder_template_is_preserved_and_actionable(tmp_path):
    service = RegimeGraphV2Service(tmp_path, tmp_path)
    old = get_template_v2('merrill-clock-v2-steps-v1')
    frozen = copy.deepcopy(old)
    with pytest.raises(ValidationError) as caught:
        service.prepare(old['definition'])
    assert caught.value.code == 'SOURCE_INPUT_REQUIRED'
    assert {item['node_id'] for item in caught.value.diagnostics} == {'growth', 'inflation'}
    assert old == frozen


def test_missing_macro_file_names_failed_input_and_etl_task(tmp_path):
    service = RegimeGraphV2Service(tmp_path, tmp_path)
    definition = get_template_v2('merrill-clock-macro-v3-steps-v1')['definition']
    with pytest.raises(ValidationError) as caught:
        service._execute_graph(None, parse_definition_v2(definition), 'retrospective', None)
    assert caught.value.field == 'graph.nodes.growth.parameters'
    assert '制造业PMI' in caught.value.message and '通用ETL' in caught.value.message

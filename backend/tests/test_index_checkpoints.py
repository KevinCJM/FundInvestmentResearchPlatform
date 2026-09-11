"""No network: validate empty provenance before allowing checkpoint reuse."""
import copy
import json
import uuid

import pytest

from backend.data_sources.index_checkpoints import read_empty_evidence, write_empty_evidence
from backend.data_sources.models import CenterError


def confirmations():
    return [{'id': uuid.uuid4().hex, 'started_at': f'2026-01-01T00:00:0{i}+00:00',
             'finished_at': f'2026-01-01T00:00:0{i}+00:00', 'rows': 0} for i in (1, 2)]


def test_empty_proof_and_old_marker_are_distinct(tmp_path):
    path = tmp_path / 'A.empty'
    args = (path, 'ths_daily', 'A.TI', '20200101', '20200102', 'checkpoint')
    write_empty_evidence(*args, confirmations())
    assert read_empty_evidence(*args)['format'] == 'index_empty_v1'
    assert len(list(tmp_path.iterdir())) == 1
    path.write_bytes(b'no data\n')
    assert read_empty_evidence(*args) is None


@pytest.mark.parametrize('case', ['api', 'code', 'start', 'end', 'checkpoint', 'format',
                                 'single', 'duplicate', 'rows', 'backward', 'naive', 'future',
                                 'extra', 'broken', 'huge', 'symlink'])
def test_bad_empty_receipts_are_rejected(tmp_path, case):
    path = tmp_path / 'A.empty'
    args = (path, 'ths_daily', 'A.TI', '20200101', '20200102', 'checkpoint')
    write_empty_evidence(*args, confirmations())
    value = json.loads(path.read_text())
    if case in {'api', 'code', 'start', 'end', 'checkpoint', 'format'}: value[case] = 'wrong'
    if case == 'single': value['confirmations'].pop()
    if case == 'duplicate': value['confirmations'][1] = copy.deepcopy(value['confirmations'][0])
    if case == 'rows': value['confirmations'][1]['rows'] = 1
    if case == 'backward': value['confirmations'].reverse()
    if case == 'naive': value['confirmations'][0]['started_at'] = '2026-01-01T00:00:01'
    if case == 'future': value['confirmations'][1]['finished_at'] = '9999-01-01T00:00:00+00:00'
    if case == 'extra': value['untrusted'] = True
    path.write_text(json.dumps(value))
    if case == 'broken': path.write_bytes(b'broken')
    if case == 'huge': path.write_bytes(b' ' * 8193)
    if case == 'symlink':
        saved = path.rename(tmp_path / 'saved')
        path.symlink_to(saved)
    with pytest.raises(CenterError) as error:
        read_empty_evidence(*args)
    assert error.value.code == 'INDEX_EMPTY_EVIDENCE_INVALID'

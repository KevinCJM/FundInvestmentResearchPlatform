from types import SimpleNamespace

import numpy as np
import pytest

from historical_regimes.v2_numba import binary_math_kernel
from test_historical_regime_v2 import client, classic_service, v2_service, _wait_for_preview
from test_regime_node_preview import start, unfinished


def preview(client):
    response, _ = start(client, unfinished())
    job = _wait_for_preview(client, response.json()['id'])
    assert job['status'] == 'completed'
    return job['id']


def test_rebases_frozen_output_with_warmed_arithmetic_and_preserves_raw_data(client, v2_service, monkeypatch):
    job_id = preview(client)
    path = f'/api/historical-regimes/preview-runs/{job_id}'
    raw = client.get(path + '/series').json()
    signatures = list(binary_math_kernel.signatures)
    calls = []

    def audited(left, right, opcode):
        assert left.dtype == right.dtype == np.float64
        assert left.flags.c_contiguous and right.flags.c_contiguous
        assert binary_math_kernel._can_compile is False
        calls.append(opcode)
        return binary_math_kernel(left, right, opcode)

    monkeypatch.setattr('historical_regimes.v2_service.binary_math_kernel', audited)
    for base_index in [0, 20, 79]:
        response = client.get(path + '/normalized-chart', params={'node_id': 'source', 'base_index': base_index})
        assert response.status_code == 200, response.text
        result = response.json()
        base = raw['items'][base_index]['value']
        np.testing.assert_allclose(result['values'], [row['value'] / base for row in raw['items']])
        np.testing.assert_allclose(result['change_pct'], [(row['value'] / base - 1) * 100 for row in raw['items']])
        assert result['values'][base_index] == 1
        assert result['base_date'] == raw['items'][base_index]['observation_date']
        assert result['execution']['python_fallback'] == result['execution']['request_time_compilation'] == 0
    assert calls == [3, 1, 2] * 3
    assert binary_math_kernel.signatures == signatures
    assert client.get(path + '/series').json() == raw


@pytest.mark.parametrize('values,base,valid', [
    ([2, 3, 1], 0, True), ([2], 0, True),
    ([2, np.nan, np.inf, -np.inf, 0, -2], 0, True),
    ([1e-308, 1e308], 0, True), ([], 0, False),
    ([np.nan, 2], 0, False), ([np.inf, 2], 0, False),
    ([0, 2], 0, False), ([-2, 2], 0, False), ([2, 3], 2, False),
])
def test_normalization_missing_zero_boundary_and_determinism(client, v2_service, values, base, valid):
    job_id = preview(client)
    arr = np.ascontiguousarray(values, dtype=np.float64)
    dates = np.ascontiguousarray(np.arange(len(values)), dtype=np.int64)
    v2_service._jobs[job_id]['_node_outputs']['source']['value'] = SimpleNamespace(values=arr, dates=dates)
    original = arr.copy()
    path = f'/api/historical-regimes/preview-runs/{job_id}/normalized-chart'
    params = {'node_id': 'source', 'base_index': base}
    response = client.get(path, params=params)
    assert response.status_code == (200 if valid else 422), response.text
    if valid:
        result = response.json()
        assert result == client.get(path, params=params).json()
        with np.errstate(over='ignore', invalid='ignore'):
            expected = arr / arr[base]
        assert result['values'] == [float(x) if np.isfinite(x) else None for x in expected]
    np.testing.assert_array_equal(arr, original)


def test_normalization_rejects_enum_missing_job_and_port(client):
    response, _ = start(client, unfinished(), 'classifier', 'state')
    job = _wait_for_preview(client, response.json()['id'])
    assert job['status'] == 'completed', job
    path = f"/api/historical-regimes/preview-runs/{job['id']}/normalized-chart"
    response = client.get(path, params={'node_id': 'classifier', 'port': 'state'})
    assert response.status_code == 422
    assert response.json()['detail']['code'] == 'NON_NUMERIC_NORMALIZATION'
    assert client.get(path, params={'node_id': 'source', 'port': 'missing'}).status_code == 404
    assert client.get(path, params={'node_id': 'source', 'base_index': -1}).status_code == 422
    assert client.get('/api/historical-regimes/preview-runs/missing/normalized-chart', params={'node_id': 'source'}).status_code == 404

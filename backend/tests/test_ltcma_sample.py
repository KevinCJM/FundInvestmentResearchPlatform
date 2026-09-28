"""Sample inspection uses actual evidence before NIW prior inputs are complete."""
import copy

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from backend.custom_indicators.errors import ValidationError

from backend.strategic_allocation.cma_center_contracts import CmaSampleRequest
from backend.strategic_allocation.cma_model_contracts import StatisticalCmaContext
from backend.strategic_allocation.routes import build_router
from backend.strategic_allocation.contracts import CmaRequest
from backend.tests.test_cma_scope_facts import scope_case
from backend.tests.test_strategic_allocation import workspace, warm
from backend.tests.test_ltcma_statistics import request, statistics_warm


def sample_request(**patch):
    original = request(**patch)
    model = original.model.model_dump(mode="json")
    return CmaSampleRequest(alloc_name=original.alloc_name, model={
        key: model[key] for key in StatisticalCmaContext.model_fields})


def client_for(service):
    app = FastAPI()
    app.include_router(build_router(service))
    return TestClient(app)


@pytest.mark.parametrize("start_index, count", [(0, 160), (100, 60)])
def test_sample_matches_calculation_without_prior_or_artifact(workspace, monkeypatch, start_index, count):
    service, days = workspace
    window = {"kind": "custom", "start_date": str(days[start_index].date()), "end_date": str(days[-1].date())}
    expected = service.preview_cma(request(window=window))["model_result"]["model_audit"]["evidence"]
    monkeypatch.setattr(service.cma, "calculation", lambda *_: pytest.fail("sample must not fit a model"))
    monkeypatch.setattr(service.artifacts, "save", lambda *_: pytest.fail("sample must not save artifacts"))
    monkeypatch.setattr(service.cma, "get", lambda *_: pytest.fail("sample must not require a prior"))
    response = client_for(service).post("/api/strategic-allocation/cma/sample", json=sample_request(window=window).model_dump(mode="json"))
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["observations"] == count
    assert result["actual_start"] == str(days[start_index].date())
    assert result["actual_end"] == str(days[-1].date())
    assert result == {key: expected[key] for key in result}


def test_sample_excludes_cross_day_periods_instead_of_counting_a_truncated_window(workspace):
    service, _ = workspace
    path = service.data.data_dir / "asset_nv.parquet"
    frame = pd.read_parquet(path)
    complete = client_for(service).post("/api/strategic-allocation/cma/sample", json=sample_request().model_dump(mode="json"))
    assert complete.status_code == 200, complete.text
    assert complete.json()["excluded_return_periods"] == 0
    frame.drop(index=50).to_parquet(path, index=False)
    response = client_for(service).post("/api/strategic-allocation/cma/sample", json=sample_request().model_dump(mode="json"))
    # 某个资产缺一天不再整体拒绝：该日退出共同样本，跨过它的收益期整段排除并如实计数。
    assert response.status_code == 200, response.text
    assert response.json()["excluded_return_periods"] == 1
    assert response.json()["observations"] == complete.json()["observations"] - 2


def test_sample_uses_the_same_strategic_proxy_panel(scope_case):
    service, _, raw = scope_case
    model = {key: raw["model"][key] for key in StatisticalCmaContext.model_fields if key in raw["model"]}
    client = client_for(service)
    body = {"strategic_universe_id": raw["strategic_universe_id"], "model": model}
    response = client.post("/api/strategic-allocation/cma/sample", json=body)
    assert response.status_code == 200, response.text
    expected = service.preview_cma(CmaRequest.model_validate(raw))["model_result"]["model_audit"]["evidence"]
    assert response.json() == {key: expected[key] for key in response.json()}
    body["model"]["proxy_inputs"] = None
    assert client.post("/api/strategic-allocation/cma/sample", json=body).status_code == 422


@pytest.mark.parametrize("case", ["wrong_axis", "both_scopes", "missing_scope", "incomplete_window", "unsupported_currency", "prior_input"])
def test_sample_rejects_invalid_scope_or_context(workspace, case):
    service, _ = workspace
    body = copy.deepcopy(sample_request().model_dump(mode="json"))
    if case == "wrong_axis":
        body["model"]["asset_ids"].reverse()
    elif case == "both_scopes":
        body["strategic_universe_id"] = "another-scope"
    elif case == "missing_scope":
        body["alloc_name"] = None
    elif case == "incomplete_window":
        body["model"]["window"] = {"kind": "custom"}
    elif case == "unsupported_currency":
        body["model"]["currency"] = "USD"
    else:
        body["model"]["mean_prior_observations"] = 900000
    response = client_for(service).post("/api/strategic-allocation/cma/sample", json=body)
    assert response.status_code == 422, response.text


@pytest.mark.parametrize('edge', ['start', 'end'])
def test_explicit_proxy_window_rejects_truncated_source_boundaries(scope_case, monkeypatch, edge):
    service, _, raw = scope_case
    load = service.cma.evidence.sources.load
    model = CmaRequest.model_validate(raw).model
    component = model.proxy_inputs.assets[0].components[0]
    complete = load(component, model)
    days = complete['dates']
    raw['model']['window'] = {'kind': 'custom', 'start_date': days[0], 'end_date': days[-1]}

    def incomplete(component, request):
        result = load(component, request)
        if '000012' in component.series_id:
            retained = slice(60, None) if edge == 'start' else slice(None, -60)
            result = {**result, **{key: result[key][retained] for key in ('dates', 'values', 'available_at')}}
        return result

    monkeypatch.setattr(service.cma.evidence.sources, 'load', incomplete)
    definition = CmaRequest.model_validate(raw)
    with pytest.raises(ValidationError, match='不会自动截短') as error:
        service.preview_cma(definition)
    assert error.value.code == 'LTCMA_PROXY_WINDOW_COVERAGE'
    # The evidence-only endpoint must enforce the same boundary before NIW setup.
    body = {'strategic_universe_id': raw['strategic_universe_id'], 'model': {
        key: raw['model'][key] for key in StatisticalCmaContext.model_fields if key in raw['model']}}
    response = client_for(service).post('/api/strategic-allocation/cma/sample', json=body)
    assert response.status_code == 422 and 'LTCMA_PROXY_WINDOW_COVERAGE' in response.text
    # Explicitly choosing the covered interval makes it usable again.
    raw['model']['window'].update(start_date=days[60] if edge == 'start' else days[0],
                                  end_date=days[-1] if edge == 'start' else days[-61])
    evidence = service.preview_cma(CmaRequest.model_validate(raw))['model_result']['model_audit']['evidence']
    assert evidence['observations'] == 100


def test_common_inception_window_and_known_closed_end_remain_usable(scope_case, monkeypatch):
    service, _, raw = scope_case
    load = service.cma.evidence.sources.load

    def later_inception(component, request):
        result = load(component, request)
        if '000012' in component.series_id:
            result = {**result, **{key: result[key][60:] for key in ('dates', 'values', 'available_at')}}
        return result

    monkeypatch.setattr(service.cma.evidence.sources, 'load', later_inception)
    # The fixture calendar covers weekends through as_of, even when prices stop Friday.
    evidence = service.preview_cma(CmaRequest.model_validate(raw))['model_result']['model_audit']['evidence']
    assert evidence['observations'] == 100
    assert evidence['requested_start'] == evidence['actual_start']


@pytest.mark.parametrize('source_kind', ['product', 'strategic'])
def test_saved_evidence_preserves_interval_breaks_after_date_intersection(scope_case, monkeypatch, source_kind):
    from backend.tests.test_ltcma_statistics import publish
    from backend.strategic_allocation.cma_evidence import _period_contiguity
    from backend.strategic_allocation.reference_inputs import array_digest
    service, _, raw = scope_case
    if source_kind == 'product':
        nav_path = service.data.data_dir / 'asset_nv.parquet'
        frame = pd.read_parquet(nav_path)
        frame.drop(index=80).to_parquet(nav_path, index=False)
        definition = request()
    else:
        load = service.cma.evidence.sources.load
        def missing(component, model):
            import numpy as np
            data = load(component, model)
            if '000012' in component.series_id:
                for key in ('dates', 'values', 'available_at'):
                    data[key] = np.delete(data[key], 80).tolist() if key != 'values' else np.delete(data[key], 80)
            return data
        monkeypatch.setattr(service.cma.evidence.sources, 'load', missing)
        definition = CmaRequest.model_validate(raw)
    saved = publish(service, definition, f'continuity-{source_kind}')
    frozen = service.artifacts.arrays(saved['id'], ['evidence_period_contiguous', 'evidence_dates'])
    contiguous = frozen['evidence_period_contiguous']
    assert contiguous.dtype.name == "int64" and not contiguous.flags.writeable
    assert len(contiguous) == len(frozen['evidence_dates']) == 158
    assert not contiguous[0] and sum(contiguous[1:] == 0) == 1
    assert not contiguous[79] and contiguous[80]  # restart at the gap; keep the next valid transition
    audit = saved['model_result']['model_audit']['evidence']
    assert audit['transition_gap_count'] == 1 and audit['period_contiguous_hash'] == array_digest(contiguous)
    # A weekend preserves the shared endpoint; skipped returns do not.
    assert _period_contiguity(['2026-06-04', '2026-06-05'], ['2026-06-05', '2026-06-08']).tolist() == [False, True]

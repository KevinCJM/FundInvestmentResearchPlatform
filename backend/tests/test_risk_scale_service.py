"""Offline Risk Scale lifecycle, frozen historical-reference consumption and numerical gating."""
from copy import deepcopy
from datetime import date, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from backend.tests.risk_scale_app import make_service
from backend.strategic_allocation.risk_scale_routes import build_router
from backend.strategic_allocation.risk_scale_contracts import PreviewRequest, ConfirmRequest, ActivateRequest, RetireRequest, CompareRequest, RiskScaleDefinition
from backend.strategic_allocation.reference_contracts import ReferenceInputRequest, ConfirmReferenceInput
from backend.strategic_allocation import risk_scale_kernels as numeric
from backend import frontier_moments
from backend.custom_indicators.errors import IndicatorDomainError


@pytest.fixture(scope='session')
def warmed(tmp_path_factory):
    svc, _ = make_service(tmp_path_factory.mktemp('risk-warm'))
    assert svc.warm()['complete']


@pytest.fixture
def setup(tmp_path, warmed):
    service, source_request = make_service(tmp_path)
    app = FastAPI()
    app.include_router(build_router(service))
    with TestClient(app) as client:
        yield service, client, source_request


def freeze_reference(service, source_request):
    req = ReferenceInputRequest.model_validate(source_request)
    rp = service.reference_call('preview', req)
    ref = service.reference_call('confirm', ConfirmReferenceInput(request=req, preview_hash=rp['preview_hash'], confirm=True,
        idempotency_key='reference-one', acknowledged_warnings=[x['code'] for x in rp['warnings']]))
    return ref, ref


def definition(reference):
    return {'name':'Synthetic risk scale', 'scheme_id':'scheme-test', 'research_as_of':str(date.today()),
        'review_due_at':str(date.today()+timedelta(days=90)), 'reference_input_ref':{k:reference[k] for k in ('id','content_hash')},
        'purpose':'Strictly synthetic offline fixture'}


def publish(service, payload, key='scale-one'):
    req = PreviewRequest.model_validate({'definition':payload})
    preview = service.preview(req)
    assert preview['publication_eligibility']['eligible'], preview['publication_eligibility']
    body = ConfirmRequest(request=req,confirm=True,preview_hash=preview['preview_hash'],idempotency_key=key,
        acknowledged_warnings=[x['code'] for x in preview['warnings']])
    return service.confirm(body), body, preview


def test_optional_text_and_review_date_contract(setup):
    _, _, source = setup
    raw = deepcopy(source)
    for asset in raw['assets']:
        asset['rationale'] = ''
    reference = ReferenceInputRequest.model_validate(raw)
    assert all(asset.rationale == '' for asset in reference.assets)

    frozen = {'id': 'reference-test', 'content_hash': 'a' * 64}
    scale = RiskScaleDefinition.model_validate({
        'name': 'No text explanation required', 'scheme_id': 'scheme-optional-text',
        'research_as_of': raw['as_of'], 'reference_input_ref': frozen,
    })
    assert scale.purpose == '' and scale.description == '' and scale.review_due_at is None


P = '/api/strategic-allocation/risk-scales'
R = '/api/strategic-allocation/reference-inputs'


def test_cold_start_full_flow_and_schema(setup):
    svc,client,source_request=setup
    assert client.get(P).json()['items']==[]
    assert client.get(P+'/defaults').json()['items']==[]
    caps=client.get(P+'/capabilities').json()
    assert caps['ready'] and caps['trust_mode']=='single_local_trusted_workspace'
    assert len(caps['algorithms'])==len(numeric.ALGORITHMS) and caps['templates'][0]['published_default'] is False
    catalog=client.get(R+'/catalog?kind=etf').json()
    assert len(catalog['items'])==3
    reference,_=freeze_reference(svc,source_request)
    version,body,preview=publish(svc,definition(reference))
    assert len(preview['result']['frontier'])==101 and len(preview['result']['levels'])==5
    assert svc.confirm(body)['id']==version['id']
    assert client.get(P+'/'+version['id']).status_code==200
    assert client.get(P).json()['total']==1
    schema=client.get('/openapi.json').json()
    assert schema['paths'][P+'/preview']['post']['responses']['200']['content']['application/json']['schema']['$ref'].endswith('PreviewResponse')
    for path, methods in schema['paths'].items():
        if path.startswith((P, R)):
            for operation in methods.values():
                for status in ('400', '404', '409', '422', '503'):
                    assert operation['responses'][status]['content']['application/json']['schema']['$ref'].endswith('ErrorResponse')
    malformed = client.post(P+'/preview', json={})
    from backend.strategic_allocation.risk_scale_contracts import ErrorResponse
    assert ErrorResponse.model_validate(malformed.json()).detail.code == 'INPUT_INVALID'
    assert preview['execution_audit']['python_fallback']==0
    original_hash=version['content_hash']
    active=client.post(P+'/'+version['id']+'/activate',json={'confirm':True,'expected_revision':0})
    assert active.status_code==200,active.json()
    assert client.post(P+'/'+version['id']+'/activate',json={'confirm':True,'expected_revision':0}).status_code==409
    assert client.post(P+'/'+version['id']+'/retire',json={'confirm':True,'expected_revision':1,'reason':'Test retirement'}).status_code==422
    retired=client.post(P+'/'+version['id']+'/retire',json={'confirm':True,'expected_revision':1,'reason':'Test retirement','clear_default':True})
    assert retired.status_code==200 and retired.json()['version_id'] is None
    old=client.get(P+'/'+version['id']).json()
    assert old['content_hash']==original_hash and old['retired'] and not old['current_eligibility']['eligible']
    assert client.get(P).json()['total']==0  # Deleted/retired versions leave the normal configuration list.
    historical = client.get(P+'?include_retired=true&limit=1').json()
    assert historical['total'] == 1 and historical['items'][0]['id'] == version['id']
    assert historical['items'][0]['retired'] is True
    assert historical['next_offset'] is None


def test_study_options_only_returns_versions_usable_on_requested_research_day(setup):
    svc, client, source_request = setup
    reference, _ = freeze_reference(svc, source_request)
    payload = definition(reference)
    payload['valid_until'] = str(date.today() + timedelta(days=30))
    version, _, _ = publish(svc, payload, key='study-option-scale')
    current = client.get(P + '/study-options', params={'as_of': str(date.today())})
    assert current.status_code == 200, current.json()
    assert [item['id'] for item in current.json()['items']] == [version['id']]
    assert current.json()['items'][0]['base_currency'] == 'CNY'
    before_publication = client.get(P + '/study-options', params={'as_of': str(date.today() - timedelta(days=1))})
    assert before_publication.status_code == 200 and before_publication.json()['items'] == []
    future = client.get(P + '/study-options', params={'as_of': str(date.today() + timedelta(days=1))})
    assert future.status_code == 422
    retired = client.post(P + '/' + version['id'] + '/retire',
        json={'confirm': True, 'expected_revision': 0, 'reason': 'Test study option retirement'})
    assert retired.status_code == 200, retired.json()
    assert client.get(P + '/study-options', params={'as_of': str(date.today())}).json()['items'] == []


def test_registered_algorithms_reuse_exact_frontier_and_hash(setup, monkeypatch):
    svc,_,source_request=setup
    reference,_=freeze_reference(svc,source_request)
    d=definition(reference)
    first=svc.preview(PreviewRequest(definition=d))
    def forbidden(*a,**k): raise AssertionError('frontier reacquired or refit')
    monkeypatch.setattr(frontier_moments,'solve_frontier',forbidden)
    hashes=set()
    for algorithm in numeric.ALGORITHMS:
        d['segmentation']={'algorithm_id':algorithm}
        if algorithm.startswith('manual'):
            d['segmentation'].update(manual_caps=[0.,.03,.06,.2,.8],rationale='Explicit manual policy')
        result=svc.preview(PreviewRequest(definition=d))
        assert result['result']['frontier']==first['result']['frontier']
        assert result['publication_eligibility']['eligible']
        hashes.add(result['preview_hash'])
        if algorithm.startswith('manual'):
            assert result['result']['levels'][0]['representative_node_id']==0
            assert result['result']['levels'][0]['volatility']['value']<=numeric.BOUNDARY_TOL
            assert result['result']['levels'][-1]['calibration_status']=='not_calibrated'
    assert len(hashes)==len(numeric.ALGORITHMS) and svc.cache.hits>=len(numeric.ALGORITHMS)


def test_automatic_boundaries_can_be_fine_tuned_without_rebuilding_frontier(setup, monkeypatch):
    svc,_,source_request=setup
    reference,_=freeze_reference(svc,source_request)
    d=definition(reference)
    base=svc.preview(PreviewRequest(definition=d))
    original=np.asarray(base['result']['applied_boundaries'], dtype=np.float64)
    adjusted=original.copy()
    for i in range(4):
        adjusted[i]=original[i]+.2*(original[i+1]-original[i])
    def forbidden(*a,**k): raise AssertionError('boundary fine-tuning rebuilt the frontier')
    monkeypatch.setattr(frontier_moments,'solve_frontier',forbidden)
    d['segmentation']={'algorithm_id':'frontier_shape_dp_v2','adjusted_caps':adjusted.tolist()}
    result=svc.preview(PreviewRequest(definition=d))
    assert result['result']['frontier']==base['result']['frontier']
    np.testing.assert_allclose(result['result']['applied_boundaries'],adjusted)
    assert result['result']['levels'][0]['lower_bound']==0
    for i in range(1,5):
        assert result['result']['levels'][i]['lower_bound']==pytest.approx(adjusted[i-1])
        assert result['result']['levels'][i-1]['upper_bound']==pytest.approx(adjusted[i-1])
        assert result['result']['levels'][i]['lower_inclusive'] is False
    metrics=np.asarray([[point['volatility'],point['expected_return']] for point in result['result']['frontier']],dtype=np.float64)
    statuses=np.zeros(metrics.shape[0],dtype=np.int64)
    expected=numeric.segment_frontier(metrics,statuses,'manual_volatility_bands_v1',manual_caps=adjusted)
    assert [level['representative_node_id'] if level['representative_node_id'] is not None else -1 for level in result['result']['levels']]==expected['representative_node_indices'].tolist()
    assert result['result']['stability']['status']=='manual_adjustment'
    assert result['result']['diagnostics']['algorithm_boundaries']==base['result']['applied_boundaries']
    assert 'BOUNDARY_ADJUSTED' in [item['code'] for item in result['warnings']]


def test_cached_and_fresh_hash_identical_no_input_modification(setup):
    svc,_,source_request=setup
    reference,_=freeze_reference(svc,source_request)
    request=PreviewRequest(definition=definition(reference))
    before=svc.artifacts.arrays(reference['id'])['covariance']
    checksum=before.tobytes()
    a=svc.preview(request)
    svc.cache.items.clear();svc.cache.bytes=0
    b=svc.preview(request)
    assert a['preview_hash']==b['preview_hash'] and a['result']==b['result']
    assert before.tobytes()==checksum and not before.flags.writeable


def test_preview_writes_nothing(setup,monkeypatch):
    svc,_,source_request=setup
    reference,_=freeze_reference(svc,source_request)
    req=PreviewRequest(definition=definition(reference))
    before={p: p.read_bytes() for p in svc.artifacts.root.parent.rglob('*') if p.is_file()}
    def forbidden(*a,**kw): raise AssertionError('preview writes')
    from backend.custom_indicators.repository import AtomicJsonStore
    monkeypatch.setattr(svc.artifacts,'save',forbidden)
    monkeypatch.setattr(AtomicJsonStore,'locked',forbidden)
    monkeypatch.setattr(Path,'mkdir',forbidden)
    assert svc.preview(req)['publication_eligibility']['eligible']
    assert before=={p:p.read_bytes() for p in svc.artifacts.root.parent.rglob('*') if p.is_file()}


def test_confirmation_hash_warnings_and_idempotency_conflict(setup):
    svc,client,sources=setup
    reference,_=freeze_reference(svc,sources)
    req={'definition':definition(reference)}
    pre=client.post(P+'/preview',json=req).json()
    body={'request':req,'confirm':True,'preview_hash':'0'*64,'idempotency_key':'test-confirm','acknowledged_warnings':[]}
    assert client.post(P+'/confirm',json=body).status_code==409
    body['preview_hash']=pre['preview_hash']
    assert client.post(P+'/confirm',json=body).status_code==422
    body['acknowledged_warnings']=[x['code'] for x in pre['warnings']]
    one=client.post(P+'/confirm',json=body);assert one.status_code==201,one.json()
    assert client.post(P+'/confirm',json=body).json()['id']==one.json()['id']
    body['request']['definition']['name']='Changed'
    assert client.post(P+'/confirm',json=body).status_code==409


def test_numeric_failure_is_finite_blocked_result(setup):
    svc,client,sources=setup
    reference,_=freeze_reference(svc,sources)
    d=definition(reference);d['constraint_profile']={'asset_limits':{x:{'min_weight':.8,'max_weight':1.0} for x in ['cash','bond','equity']}}
    response=client.post(P+'/preview',json={'definition':d})
    assert response.status_code==200,response.json()
    value=response.json();assert not value['publication_eligibility']['eligible']
    assert 'NaN' not in response.text and 'Infinity' not in response.text


def test_classification_lower_at_equality_and_above_scale(setup):
    svc,client,sources=setup
    reference,_=freeze_reference(svc,sources);version,_,_=publish(svc,definition(reference))
    caps=version['preview']['result']['applied_boundaries']
    for i,cap in enumerate(caps):
        r=client.post(P+'/'+version['id']+'/classify',json={'volatility':cap,'authorized_level':i+1}).json()
        assert r['level_code']==f'C{i+1}' and r['cap_satisfied']
    r=client.post(P+'/'+version['id']+'/classify',json={'volatility':caps[-1]+.1}).json()
    assert r['status']=='above_scale' and r['level_code'] is None
    assert client.post(P+'/'+version['id']+'/classify',json={'volatility':None}).json()['status']=='unavailable'


def test_busy_and_pid_not_ready_do_not_block_lists(setup,monkeypatch):
    svc,client,sources=setup
    reference,_=freeze_reference(svc,sources)
    req={'definition':definition(reference)}
    svc._slots.acquire()
    try:
        assert client.get(P).status_code==200
        assert client.post(P+'/preview',json=req).status_code==503
    finally: svc._slots.release()
    monkeypatch.setattr(numeric,'_WARMED_PID',-1)
    assert client.post(P+'/preview',json=req).status_code==503
    assert client.get(P).status_code==200


@pytest.mark.parametrize('change',[
    {'research_as_of':None},{'reference_input_ref':{'id':'../secret','content_hash':'0'*64}},
    {'review_due_at':'2000-01-01'},{'segmentation':{'algorithm_id':'not-an-algorithm'}},
    {'segmentation':{'algorithm_id':'manual_volatility_bands_v1','manual_caps':[.1]*5,'rationale':'reason'}},
    {'constraint_profile':{'allow_short':True}},
])
def test_malformed_inputs_structured_errors(setup,change):
    _,client,_=setup
    d={'name':'Test','scheme_id':'scheme-test','research_as_of':str(date.today()),'review_due_at':str(date.today()+timedelta(days=5)),
       'reference_input_ref':{'id':'research-series-test','content_hash':'0'*64},'purpose':'test case',**change}
    r=client.post(P+'/preview',json={'definition':d})
    assert r.status_code==422 and set(r.json()['detail'])=={'code','message','field','suggested_action'}


def test_drafts_revision_crud(setup):
    _,client,_=setup
    body={'name':'Draft','scheme_id':'scheme-test','editable_definition':{'incomplete':'-'}}
    r=client.post(P+'/drafts',json=body);assert r.status_code==201
    draft=r.json();assert draft['revision']==1
    route=P+'/drafts/'+draft['id']
    patch={**body,'expected_revision':1,'name':'Edited'}
    assert client.patch(route,json=patch).json()['revision']==2
    assert client.patch(route,json=patch).status_code==409
    assert client.request('DELETE',route,json={'expected_revision':1}).status_code==409
    assert client.request('DELETE',route,json={'expected_revision':2}).status_code==200
    assert client.get(route).status_code==404


def test_default_cas_race_and_compare(setup):
    svc,_,sources=setup
    reference,_=freeze_reference(svc,sources);d=definition(reference)
    first,_,_=publish(svc,d)
    d['name']='Second';d['segmentation']={'algorithm_id':'equal_arclength_v1'}
    second,_,_=publish(svc,d,'scale-two')
    def activate(identifier):
        try: return svc.activate(identifier,ActivateRequest(confirm=True,expected_revision=0))['revision']
        except IndicatorDomainError as exc:return exc.code
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(activate,[first['id'],second['id']]))
    assert sorted(map(str,results))==['1','DEFAULT_CHANGED']
    compared=svc.compare(CompareRequest(left_id=first['id'],right_id=second['id']))
    assert compared['compatible'] and len(compared['boundary_differences'])==5
    assert 'segmentation' in compared['differences']


def test_risk_scale_consumes_frozen_reference_parameters_without_refit(setup, monkeypatch):
    svc,_,sources=setup
    reference,_=freeze_reference(svc,sources)
    monkeypatch.setattr(svc.references, '_input_calculation', lambda *a: (_ for _ in ()).throw(AssertionError('historical refit')))
    result = svc.preview(PreviewRequest(definition=definition(reference)))
    assert result['result']['parameter_evidence']['method_identity']['id'] == 'historical_common_intersection'
    assert result['resolved_refs']['reference_inputs']['id'] == reference['id']


def test_generated_types_are_deterministic_and_registry_matches():
    from scripts.export_risk_scale_contracts import generate,TARGET
    from backend.strategic_allocation.risk_scale_contracts import Segmentation
    assert generate()==TARGET.read_text()
    assert set(Segmentation.model_json_schema()['properties']['algorithm_id']['enum'])==set(numeric.ALGORITHMS)


def test_runtime_observations_do_not_change_preview_hash(setup,monkeypatch):
    svc,_,source=setup
    reference,_=freeze_reference(svc,source);request=PreviewRequest(definition=definition(reference))
    first=svc.preview(request)
    original=svc.execution
    def changed():
        result=original();result['test_elapsed_ms']=999.;result['test_pid']=123;return result
    monkeypatch.setattr(svc,'execution',changed)
    def forbidden(*a,**kw):raise AssertionError('segmentation change reacquires data or refits history')
    monkeypatch.setattr(svc.references.sources,'load',forbidden)
    monkeypatch.setattr('backend.strategic_allocation.reference_evidence_kernels.annual_moments',forbidden)
    second=svc.preview(request)
    assert first['preview_hash']==second['preview_hash']


def test_review_due_is_reminder_only_and_read_never_recomputes(setup,monkeypatch):
    svc,client,source=setup
    reference,_=freeze_reference(svc,source); payload=definition(reference)
    payload['review_due_at']=str(date.today()+timedelta(days=10))
    version,_,_=publish(svc,payload)
    monkeypatch.setattr(frontier_moments,'solve_frontier',lambda *a,**kw: (_ for _ in ()).throw(AssertionError('historical recompute')))
    r=client.get(P+'/'+version['id'])
    assert r.status_code==200 and r.json()['current_eligibility']['eligible']
    assert r.json()['review_status']=='upcoming'
    assert r.json()['content_hash']==version['content_hash']
    listed=next(item for item in client.get(P).json()['items'] if item['id']==version['id'])
    assert listed['review_status']=='upcoming' and listed['review_due_at']==payload['review_due_at']
    # Simulate a pre-review-metadata index row: catalog must recover from the immutable manifest.
    index=svc.artifacts.index.read_unlocked()
    for item in index['items']:
        if item['id']==version['id']: item.pop('review_due_at',None)
    with svc.artifacts.index.locked(): svc.artifacts.index.write_unlocked(index)
    legacy=next(item for item in client.get(P).json()['items'] if item['id']==version['id'])
    assert legacy['review_status']=='upcoming' and legacy['review_due_at']==payload['review_due_at']

    import backend.strategic_allocation.risk_scale_service as service_module
    real_today=date.today()
    class Later(date):
        @classmethod
        def today(cls): return real_today+timedelta(days=20)
    monkeypatch.setattr(service_module,'date',Later)
    due=client.get(P+'/'+version['id'])
    assert due.status_code==200 and due.json()['current_eligibility']['eligible']
    assert due.json()['review_status']=='due'


def test_group_and_nonempty_asset_constraints_resolve_exactly(setup):
    svc,client,source=setup
    reference,_=freeze_reference(svc,source);d=definition(reference)
    d['constraint_profile']={'asset_limits':{'equity':{'min_weight':.1,'max_weight':.6}},
        'group_limits':[{'id':'defensive','assets':['cash','bond'],'lo':.4,'hi':.9}]}
    r=client.post(P+'/preview',json={'definition':d})
    assert r.status_code==200,r.json()
    for p in r.json()['result']['frontier']:
        if p['weights'] is not None:
            assert .1-1e-7<=p['weights'][2]<=.6+1e-7
            assert .4-1e-7<=sum(p['weights'][:2])<=.9+1e-7


def test_constraints_cannot_exclude_low_risk_space_and_keep_default_qualification(setup):
    svc,client,source=setup
    reference,_=freeze_reference(svc,source);d=definition(reference)
    d['constraint_profile']={'asset_limits':{'equity':{'min_weight':.8,'max_weight':1.0}}}
    pre=client.post(P+'/preview',json={'definition':d}).json()
    assert not pre['default_eligibility']['eligible']
    assert 'CASH_CONSTRAINT_COVERAGE' in [x['code'] for x in pre['default_eligibility']['blockers']]


@pytest.mark.parametrize('adjusted', [False, True])
def test_unstable_automatic_calibration_cannot_be_default(setup, monkeypatch, adjusted):
    svc, _, inputs = setup
    reference, _ = freeze_reference(svc, inputs)
    d = definition(reference)
    if adjusted:
        stable = svc.preview(PreviewRequest(definition=d))
        d['segmentation'] = {'algorithm_id': 'frontier_shape_dp_v2',
                             'adjusted_caps': stable['result']['applied_boundaries']}
    monkeypatch.setattr(numeric, 'boundary_stability_kernel', lambda *args: (.2, 1))
    version, _, preview = publish(svc, d)
    assert preview['publication_eligibility']['eligible']
    assert not preview['default_eligibility']['eligible']
    assert 'UNSTABLE_CALIBRATION' in [x['code'] for x in preview['default_eligibility']['blockers']]
    with pytest.raises(IndicatorDomainError) as error:
        svc.activate(version['id'], ActivateRequest(confirm=True, expected_revision=0))
    assert error.value.code == 'DEFAULT_INELIGIBLE'
    assert svc.get_version(version['id'])['content_hash'] == version['content_hash']
    old = svc._item(version['id'])
    old['preview']['default_eligibility'] = {'eligible': True, 'blockers': []}
    assert not svc._current(old, default=True)['eligible']
    monkeypatch.setattr(svc, '_item', lambda identifier: old)
    view = svc.get_version(version['id'])
    assert view['preview']['default_eligibility']['eligible']
    assert not view['current_default_eligibility']['eligible']
    assert 'UNSTABLE_CALIBRATION' in [x['code'] for x in view['current_default_eligibility']['blockers']]


@pytest.mark.parametrize('action', ['update', 'delete'])
def test_draft_revision_is_rechecked_on_preview_and_confirm(setup, action):
    svc, client, inputs = setup
    reference, _ = freeze_reference(svc, inputs)
    d = definition(reference)
    saved = client.post(P+'/drafts', json={'name': d['name'], 'scheme_id': d['scheme_id'],
        'editable_definition': {'schema_version': 1, 'definition': d}}).json()
    # In-memory edits remain supported; the saved revision is the concurrency token.
    request = {'definition': {**d, 'name': 'Unsaved local edit'}, 'draft_id': saved['id'], 'draft_revision': saved['revision']}
    response = client.post(P+'/preview', json=request)
    assert response.status_code == 200, response.json()
    preview = response.json()
    if action == 'update':
        changed = client.patch(P+'/drafts/'+saved['id'], json={'name': d['name'], 'scheme_id': d['scheme_id'],
            'editable_definition': {'definition': d}, 'expected_revision': saved['revision']})
    else:
        changed = client.request('DELETE', P+'/drafts/'+saved['id'], json={'expected_revision': saved['revision']})
    assert changed.status_code == 200, changed.json()
    stale = client.post(P+'/preview', json=request)
    assert stale.status_code == 409 and stale.json()['detail']['code'] == 'REVISION_CONFLICT'
    before = svc.catalog()['total']
    stale = client.post(P+'/confirm', json={'request': request, 'preview_hash': preview['preview_hash'],
        'confirm': True, 'idempotency_key': 'stale-draft-'+action,
        'acknowledged_warnings': [item['code'] for item in preview['warnings']]})
    assert stale.status_code == 409 and stale.json()['detail']['code'] == 'REVISION_CONFLICT'
    assert svc.catalog()['total'] == before

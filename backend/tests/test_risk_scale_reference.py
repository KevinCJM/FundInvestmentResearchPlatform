"""Strict real-resolver reference boundaries and shared evidence kernels."""
from copy import deepcopy
from datetime import date, timedelta
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from backend.tests.test_risk_scale_service import setup, warmed, freeze_reference, definition, P, R
from backend.strategic_allocation.reference_contracts import ReferenceInputRequest, ConfirmReferenceInput
from backend.strategic_allocation.common_contracts import FrozenRef
from backend.strategic_allocation.reference_inputs import _rebalance_reset_flags
from backend.strategic_allocation import reference_evidence_kernels as evidence
from backend.strategic_allocation.risk_scale_contracts import PreviewRequest
from backend.custom_indicators.errors import IndicatorDomainError


def test_reference_preview_is_read_only_and_selected_index_level_is_accepted(setup,monkeypatch):
    svc,client,source=setup
    index=deepcopy(source)
    for asset in index['assets']:
        for component in asset['components']:
            component.update(kind='index',series_id=component['series_id'].replace('etf:fund_daily:','index:index_daily:'),field='close')
    def forbidden(*a,**kw): raise AssertionError('reference preview creates directories/locks/artifacts')
    from backend.custom_indicators.repository import AtomicJsonStore
    monkeypatch.setattr(Path,'mkdir',forbidden);monkeypatch.setattr(AtomicJsonStore,'locked',forbidden)
    r=client.post(R+'/preview',json=index)
    assert r.status_code==200,r.json()
    assert 'INDEX_SERIES_SEMANTICS' in [x['code'] for x in r.json()['warnings']]
    assert r.json()['provenance']['assets'][0]['sources']==[]
    frozen_source=r.json()['provenance']['assets'][1]['sources'][0]
    assert frozen_source['kind']=='index'
    assert frozen_source['name'] and frozen_source['code']=='900002.SH'


def test_reference_day_is_automatic_from_pit_or_today(setup):
    svc,client,source=setup
    requested=deepcopy(source);requested['as_of']=str(date.today()-timedelta(days=90))
    without_pit=client.post(R+'/preview',json=requested)
    assert without_pit.status_code==200,without_pit.json()
    assert without_pit.json()['definition']['as_of']==str(date.today())

    pit_day=date.today()-timedelta(days=30)
    data_dir=svc.references.sources.data_dir
    (data_dir/'pit_settings.json').write_text(json.dumps({'active_release_id':'release-risk-test','as_of':None}))
    (data_dir/'data_releases.json').write_text(json.dumps({'releases':[{'id':'release-risk-test','as_of':str(pit_day),'summary':{'available_through':str(pit_day)}}]}))
    with_pit=client.post(R+'/preview',json=source)
    assert with_pit.status_code==200,with_pit.json()
    assert with_pit.json()['definition']['as_of']==str(pit_day)
    assert with_pit.json()['quality']['intersection_end']<=str(pit_day)


def test_missing_date_shrinks_intersection_and_unknown_announcement_rejected(setup):
    svc,client,source=setup
    path=svc.references.sources.data_dir/'synthetic-test-only'/'etf_daily_df.parquet'
    frame=pd.read_parquet(path)
    original=frame.copy()
    target=frame.index[frame['ts_code']=='900002.SH'][12]
    frame.drop(index=[target]).to_parquet(path,index=False)
    r=client.post(R+'/preview',json=source)
    assert r.status_code==422,r.json()
    assert r.json()['detail']['code']=='REFERENCE_SSE_CALENDAR_GAP'
    # Restore a complete market series, then make one common-date information clock unknown.
    target=original.index[original['ts_code']=='900002.SH'][12]
    original.loc[target,'ann_date']=pd.NaT
    original.to_parquet(path,index=False)
    r=client.post(R+'/preview',json=source)
    assert r.status_code==422 and r.json()['detail']['code']=='REFERENCE_INFORMATION_CLOCK'


def test_sample_window_is_not_user_input_and_missing_source_is_rejected(setup):
    _,client,source=setup
    with_window=deepcopy(source);with_window['start_date']='2000-01-01'
    assert client.post(R+'/preview',json=with_window).status_code==422
    source['assets'][1]['components'][0]['series_id']='index:index_daily:missing'
    assert client.post(R+'/preview',json=source).status_code==422


def test_source_change_on_confirmation_is_rejected(setup):
    svc,client,source=setup
    r=client.post(R+'/preview',json=source);assert r.status_code==200
    body={'request':source,'preview_hash':r.json()['preview_hash'],'confirm':True,'idempotency_key':'source-confirm',
          'acknowledged_warnings':[x['code'] for x in r.json()['warnings']]}
    path=svc.references.sources.data_dir/'synthetic-test-only'/'etf_daily_df.parquet'
    frame=pd.read_parquet(path);frame.loc[20,'adj_nav']*=1.01;frame.to_parquet(path,index=False)
    assert client.post(R+'/confirm',json=body).status_code==409


def test_shared_moments_matches_reference_and_readonly_strides(warmed):
    rng=np.random.default_rng(123)
    values=rng.normal(.0002,.01,(120,6))[::2,::2]
    values.flags.writeable=False
    before=values.tobytes()
    mean,cov,vol,corr=evidence.annual_moments(values,.2,252)
    expected=np.cov(values,rowvar=False,ddof=1)*252
    expected=expected*.8+np.diag(np.diag(expected))*.2
    np.testing.assert_allclose(mean,values.mean(axis=0)*252)
    np.testing.assert_allclose(cov,expected)
    assert before==values.tobytes() and evidence.audit()['complete']
    zero=np.zeros((30,2));_,cov,vol,_=evidence.annual_moments(zero,0.,252)
    assert np.all(vol==0) and np.all(cov==0)
    with pytest.raises(ValueError):evidence.annual_moments(values[:2],.1,252)


def test_cash_is_proxy_free_and_zero_risk_in_direct_history_parameters(setup):
    svc,client,source=setup
    cash=source['assets'][0]
    assert cash['asset_type']=='cash' and cash['components']==[] and cash['cash_return']==.01
    preview=client.post(R+'/preview',json=source)
    assert preview.status_code==200,preview.json()
    assert preview.json()['quality']['observed_annual_volatility'][0]==pytest.approx(0.)
    assert preview.json()['provenance']['assets'][0]['sources']==[]
    reference,_=freeze_reference(svc,source)
    assert reference['moments']['annual_returns'][0]==pytest.approx(.01)
    assert reference['moments']['annual_volatilities'][0]==pytest.approx(0.)
    covariance=np.asarray(reference['moments']['covariance'])
    np.testing.assert_allclose(covariance[0],0.,atol=1e-15)
    np.testing.assert_allclose(covariance[:,0],0.,atol=1e-15)


def test_cash_and_market_contracts_fail_closed(setup):
    _,client,source=setup
    cash_with_proxy=deepcopy(source)
    cash_with_proxy['assets'][0]['components']=[deepcopy(source['assets'][1]['components'][0])]
    assert client.post(R+'/preview',json=cash_with_proxy).status_code==422
    market_without_proxy=deepcopy(source)
    market_without_proxy['assets'][1]['components']=[]
    assert client.post(R+'/preview',json=market_without_proxy).status_code==422
    two_cash=deepcopy(source)
    two_cash['assets'][1].update(asset_type='cash',cash_return=0.,components=[],rebalance=None)
    assert client.post(R+'/preview',json=two_cash).status_code==422


def test_rebalance_period_boundaries_use_last_common_value_date():
    days=['2025-03-28','2025-03-31','2025-04-01','2025-06-30','2025-07-01','2025-12-31','2026-01-02']
    np.testing.assert_array_equal(_rebalance_reset_flags(days,'daily'),np.ones(6,dtype=np.int64))
    np.testing.assert_array_equal(_rebalance_reset_flags(days,'monthly'),[0,1,1,1,1,1])
    np.testing.assert_array_equal(_rebalance_reset_flags(days,'quarterly'),[0,1,0,1,1,1])
    np.testing.assert_array_equal(_rebalance_reset_flags(days,'yearly'),[0,0,0,0,0,1])
    np.testing.assert_array_equal(_rebalance_reset_flags(days,'buy_and_hold'),np.zeros(6,dtype=np.int64))


def test_proxy_drift_missing_and_rebalance_semantics(warmed):
    r=np.array([[.1,0.],[0.,.1],[.1,0.]])
    w=np.array([.5,.5]);r.flags.writeable=w.flags.writeable=False
    got=evidence.proxy_returns(r,w,np.zeros(3,dtype=np.int64))
    nav=np.cumprod(1+r,axis=0)@w
    np.testing.assert_allclose(got,np.diff(np.r_[1.,nav])/np.r_[1.,nav[:-1]])
    np.testing.assert_allclose(evidence.proxy_returns(r,w,np.ones(3,dtype=np.int64)),[.05,.05,.05])
    missing=np.array([[.1,np.nan],[.2,.1]])
    assert np.isnan(evidence.proxy_returns(missing,w,np.zeros(2,dtype=np.int64))).all()
    levels=np.array([[1.,1.],[np.nan,1.1],[1.2,1.2]])
    assert np.isnan(evidence.adjacent_returns(levels)[:,0]).all()


def test_missing_cash_and_incompatible_basis_cannot_default(setup):
    svc,client,source=setup
    source['assets']=[asset for asset in source['assets'] if asset['asset_type']!='cash']
    reference,_=freeze_reference(svc,source)
    d=definition(reference)
    pre=client.post(P+'/preview',json={'definition':d}).json()
    assert pre['publication_eligibility']['eligible'] and not pre['default_eligibility']['eligible']
    assert 'CASH_ANCHOR_REQUIRED' not in [x['code'] for x in pre['default_eligibility']['blockers']]
    assert 'CASH_ANCHOR_INVALID' in [x['code'] for x in pre['default_eligibility']['blockers']]
    d['base_currency']='USD'
    assert client.post(P+'/preview',json={'definition':d}).json()['detail']['code']=='REFERENCE_BASIS_MISMATCH'


def test_reference_freezes_direct_historical_parameters_with_explicit_identity(setup):
    svc,_,source=setup
    reference,_=freeze_reference(svc,source)
    assert reference['artifact_type']=='reference_inputs'
    assert reference['quality']['historical_pit_proven'] is False
    assert reference['quality']['intersection_start'] and reference['quality']['intersection_end']
    frozen=svc.references.get(FrozenRef(id=reference['id'],content_hash=reference['content_hash']))
    assert frozen['artifact_type']=='reference_inputs'
    preview=svc.preview(PreviewRequest(definition=definition(reference)))
    assert preview['result']['parameter_evidence']['method_identity']['id']=='historical_common_intersection'


def test_confirmation_requires_literal_true(setup):
    _,client,source=setup
    for confirm in [False,1,'true',None]:
        r=client.post(R+'/confirm',json={'request':source,'preview_hash':'0'*64,'confirm':confirm,
             'idempotency_key':'invalid-confirm','acknowledged_warnings':[]})
        assert r.status_code==422


def test_reference_load_uses_pinned_catalog_snapshot(setup,monkeypatch):
    svc,_,source=setup
    request=ReferenceInputRequest.model_validate(source)
    component=request.assets[1].components[0]
    snapshot, manifest=svc.references.sources.active_snapshot_context()
    monkeypatch.setattr(svc.references.sources.series, '_active_snapshot',
                        lambda: (_ for _ in ()).throw(AssertionError('active snapshot changed during load')))
    loaded=svc.references.sources.load(component,request,snapshot=snapshot,manifest=manifest)
    assert loaded['identity']['series_id']==component.series_id


def test_raw_resolver_used_without_display_profile_or_numeric_json_roundtrip(setup,monkeypatch):
    svc,client,source=setup
    monkeypatch.setattr(svc.references.sources.series,'profile',lambda **kwargs: (_ for _ in ()).throw(AssertionError('unused display statistics')))
    r=client.post(R+'/preview',json=source)
    assert r.status_code==200,r.json()
    assert len(r.json()['quality']['observed_annual_volatility'])==3
    loaded=svc.references.sources.load(ReferenceInputRequest.model_validate(source).assets[1].components[0],ReferenceInputRequest.model_validate(source))
    assert isinstance(loaded['values'],np.ndarray) and not loaded['values'].flags.writeable


def test_composite_requires_explicit_weights_and_preserves_names(setup):
    svc,client,source=setup
    composite=deepcopy(source)
    first=composite['assets'][1]
    first['components']=[{**first['components'][0],'weight':.4},{**composite['assets'][2]['components'][0],'weight':.6}]
    assert client.post(R+'/preview',json=composite).status_code==200
    first['components'][1]['weight']=.5
    assert client.post(R+'/preview',json=composite).status_code==422
    reference,_=freeze_reference(svc,source)
    assert [a['name'] for a in reference['definition']['assets']]==[a['name'] for a in source['assets']]


def test_index_catalog_accepts_non_sh_sz_code_and_products_need_no_pool(setup):
    _,client,source=setup
    index=client.get(R+'/catalog?kind=index&q=H00985.CSI').json()
    match=next(item for item in index['items'] if item['code']=='H00985.CSI')
    assert match['reference_capability']['available'] is True
    assert match['reference_capability']['supported_fields']==['close']
    component=source['assets'][1]['components'][0]
    assert 'product_pool_ref' not in component
    assert client.post(R+'/preview',json=source).status_code==200


def test_source_catalog_readiness_never_uses_compute_slot(setup,monkeypatch):
    svc,client,_=setup
    svc._slots.acquire()
    try:
        assert client.get(R+'/catalog?kind=index').status_code==200
    finally:svc._slots.release()
    import backend.strategic_allocation.reference_sources as sources_module
    monkeypatch.setattr(sources_module,'_WARMED_PID',-1)
    assert client.get(R+'/catalog?kind=index').status_code==503
    assert client.get(P).status_code==200

"""Immutable artifacts, crash recovery and generic storage compatibility."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import numpy as np
import pytest
from backend.sensitivity.repository import ArtifactRepository, digest_json
from backend.custom_indicators.errors import IndicatorDomainError
from backend.data_storage import StorageError
from backend.tests.test_risk_scale_service import setup, warmed, freeze_reference, definition, publish, P
from backend.strategic_allocation.risk_scale_contracts import PreviewRequest


def test_generic_save_unchanged_and_idempotent_recovery_after_rename(tmp_path,monkeypatch):
    repo=ArtifactRepository(tmp_path/'artifacts')
    old=repo.save('series',{'name':'Existing caller'})
    assert repo.get(old['id'])['name']=='Existing caller'
    fields={'name':'Risk','artifact_type':'risk_scale','scheme_id':'scheme-a','version_number':1}
    arrays={'values':np.array([.1,.2])}
    request_hash=digest_json({'input':'a'})
    register=repo._register
    def crash(item): raise OSError('simulated after rename before index')
    monkeypatch.setattr(repo,'_register',crash)
    with pytest.raises(OSError): repo.save('series',fields,arrays,idempotency_key='operation-a',request_hash=request_hash)
    operation=json.loads((repo.root/'operations.json').read_text())['items'][0]
    assert (repo.root/operation['id']/'manifest.json').is_file()
    monkeypatch.setattr(repo,'_register',register)
    result=repo.save('series',fields,arrays,idempotency_key='operation-a',request_hash=request_hash)
    assert result['id']==operation['id']
    assert len(repo.list())==2
    assert repo.idempotent_result('operation-a',request_hash)['id']==result['id']
    assert len(list(repo.root.glob('research-series-*')))==2
    with pytest.raises(IndicatorDomainError,match='不同输入'):
        repo.save('series',fields,arrays,idempotency_key='operation-a',request_hash='0'*64)


def test_same_key_race_no_duplicate_artifacts(tmp_path):
    root=tmp_path/'artifacts'
    def save(_):
        return ArtifactRepository(root).save('series',{'name':'Concurrent'}, {'values':np.ones(3)},
                idempotency_key='concurrent-key',request_hash=digest_json({'x':1}))['id']
    with ThreadPoolExecutor(max_workers=4) as pool:
        ids=list(pool.map(save,range(8)))
    assert len(set(ids))==1 and len(ArtifactRepository(root).list())==1


@pytest.mark.parametrize('corrupt',['manifest','array'])
def test_published_corruption_rejected(setup,corrupt):
    svc,client,sources=setup
    reference,_=freeze_reference(svc,sources);version,_,_=publish(svc,definition(reference))
    folder=svc.artifacts.root/version['id']
    target=folder/('manifest.json' if corrupt=='manifest' else 'frontier_weights.npy')
    target.write_bytes(b'broken')
    r=client.get(P+'/'+version['id'])
    assert r.status_code==422 and 'CORRUPT' in r.json()['detail']['code']
    assert str(folder) not in r.text


def test_corrupt_source_blocks_new_preview_old_version_remains_readable(setup):
    svc,client,sources=setup
    reference,_=freeze_reference(svc,sources);d=definition(reference);version,_,_=publish(svc,d)
    (svc.artifacts.root/reference['id']/'covariance.npy').write_bytes(b'bad')
    assert client.post(P+'/preview',json={'definition':d}).status_code==422
    old=client.get(P+'/'+version['id'])
    assert old.status_code==200 and not old.json()['current_eligibility']['eligible']
    assert old.json()['content_hash']==version['content_hash']


def test_offline_readonly_and_capacity_errors_are_safe(setup,monkeypatch):
    svc,client,_=setup
    import backend.strategic_allocation.risk_scale_store as stores
    def offline(*a,**kw): raise StorageError('STORAGE_OFFLINE','/secret/path credentials',503)
    monkeypatch.setattr(stores,'guard_path',offline)
    r=client.post(P+'/drafts',json={'name':'Draft','scheme_id':'scheme-test','editable_definition':{}})
    assert r.status_code==503 and '/secret' not in r.text
    monkeypatch.undo()
    def readonly(*a,**kw): raise PermissionError('/secret/readonly')
    monkeypatch.setattr(svc.store.document,'write_unlocked',readonly)
    r=client.post(P+'/drafts',json={'name':'Draft','scheme_id':'scheme-test','editable_definition':{}})
    assert r.status_code==503 and '/secret' not in r.text
    def full(*a,**kw): raise OSError(28,'No space left on device /secret')
    monkeypatch.setattr(svc.store.document,'write_unlocked',full)
    assert client.post(P+'/drafts',json={'name':'Draft','scheme_id':'scheme-test','editable_definition':{}}).status_code==503


def test_list_read_does_not_scan_artifact_directories(setup,monkeypatch):
    svc,client,sources=setup
    reference,_=freeze_reference(svc,sources);publish(svc,definition(reference))
    def forbidden(*a,**kw): raise AssertionError('ordinary listing scans complete artifacts')
    monkeypatch.setattr(svc.artifacts,'get',forbidden)
    monkeypatch.setattr(Path,'iterdir',forbidden)
    assert client.get(P).status_code==200


def test_bounded_cache_eviction_and_ttl(setup):
    svc,_,_=setup
    from backend.strategic_allocation.risk_scale_service import FrontierCache
    cache=FrontierCache(max_bytes=64,ttl=-1)
    grid=((np.zeros(4),),)
    cache.put('a',grid)
    assert cache.bytes==32 and cache.get('a') is None and cache.bytes==0
    cache.ttl=60
    cache.put('a',grid);cache.put('b',grid);cache.put('c',grid)
    assert cache.bytes==64 and list(cache.items)==['b','c']


def test_publication_version_reservation_survives_index_crash(setup,monkeypatch):
    svc,_,sources=setup
    reference,_=freeze_reference(svc,sources)
    from backend.strategic_allocation.risk_scale_contracts import ConfirmRequest
    d=definition(reference);req=PreviewRequest(definition=d);pre=svc.preview(req)
    body=ConfirmRequest(request=req,confirm=True,preview_hash=pre['preview_hash'],idempotency_key='crash-first',
        acknowledged_warnings=[x['code'] for x in pre['warnings']])
    register=svc.artifacts._register
    monkeypatch.setattr(svc.artifacts,'_register',lambda *a: (_ for _ in ()).throw(OSError('after rename')))
    with pytest.raises(OSError):svc.confirm(body)
    monkeypatch.setattr(svc.artifacts,'_register',register)
    other={**d,'name':'Another publication'}
    second,_,_=publish(svc,other,'second-publish')
    first=svc.confirm(body)
    assert first['version_number']==1 and second['version_number']==2
    assert len(svc.catalog()['items'])==2
    assert svc.confirm(body)['id']==first['id']


@pytest.mark.parametrize('value',[{'drafts':{}},{'defaults':[]},{'defaults':{'CNY:x':None}},{'drafts':[None]}])
def test_corrupt_governance_is_structured(setup,value):
    svc,client,_=setup
    svc.store.document.path.parent.mkdir(parents=True,exist_ok=True)
    svc.store.document.path.write_text(json.dumps({'schema_version':1,'items':[],**value}))
    r=client.get(P)
    assert r.status_code==422 and r.json()['detail']['code']=='RISK_SCALE_STATE_CORRUPT'


def test_corrupt_operations_shape_and_symlink(tmp_path):
    repo=ArtifactRepository(tmp_path/'artifacts');repo.root.mkdir()
    operations=repo.root/'operations.json'
    operations.write_text(json.dumps({'items':[{'bad':'record'}]}))
    with pytest.raises(IndicatorDomainError):repo.idempotent_result('test-operation','0'*64)
    operations.unlink();external=tmp_path/'outside.json';external.write_text('{"items":[]}');operations.symlink_to(external)
    with pytest.raises(IndicatorDomainError):repo.idempotent_result('test-operation','0'*64)


def test_retired_frozen_reference_disables_default(setup):
    svc,client,source=setup
    reference,_=freeze_reference(svc,source);version,_,_=publish(svc,definition(reference))
    svc.artifacts.save('retirement',{'release_id':reference['id'],'note':'test reference retirement'})
    assert client.post(P+'/preview',json={'definition':definition(reference)}).status_code==422
    r=client.post(P+'/'+version['id']+'/activate',json={'confirm':True,'expected_revision':0})
    assert r.status_code==422
    assert not client.get(P+'/'+version['id']).json()['current_eligibility']['eligible']

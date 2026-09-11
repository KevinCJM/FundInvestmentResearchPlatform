"""SDK safety and canonical mapping checks; all HTTP is replaced by fixtures."""
from __future__ import annotations
import json
import sys
import subprocess
from pathlib import Path
from contextlib import nullcontext

import pandas as pd
import pytest

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from backend.data_sources import akshare_adapter as adapter, akshare_worker as worker
from backend.data_sources.akshare_presets import akshare_interfaces, akshare_source
from backend.data_sources.mapping import map_table, validate_mapping
from backend.data_sources.models import CenterError
from backend.data_sources.presets import default_interfaces
from backend.data_sources.store import SourceStore


def test_akshare_codes_share_internal_identity_with_tushare():
    ak_quote, ak_nav=akshare_interfaces()
    ts_quote=next(i for i in default_interfaces() if i.api_name=='fund_daily')
    ts_nav=next(i for i in default_interfaces() if i.api_name=='fund_nav')
    for left,right,raw,ts in [
        (ak_quote,ts_quote,{'symbol':'510300','日期':'2024-01-02','收盘':2.5,'成交量':10,'成交额':500,'涨跌幅':2,'_adjustment_basis':'RAW'},{'ts_code':'510300.SH','trade_date':'20240102','close':2.5}),
        (ak_nav,ts_nav,{'symbol':'000001','净值日期':'2024-01-02','单位净值':1.25},{'ts_code':'000001.OF','nav_date':'20240102','ann_date':'20240103','unit_nav':1.25}),
    ]:
        assert validate_mapping(left)['ready']
        a,errors=map_table([raw],left.mappings[0],'akshare','test'); assert not errors
        b,errors=map_table([ts],right.mappings[0],'tushare','test'); assert not errors
        assert a.to_pylist()[0]['instrument_id']==b.to_pylist()[0]['instrument_id']
        if left is ak_quote:
            assert a.to_pylist()[0]['volume']==1000
            assert a.to_pylist()[0]['return_decimal']==.02
        else:
            assert a.to_pylist()[0]['adjusted_nav'] is None
            assert a.to_pylist()[0]['available_at'] is None


@pytest.mark.parametrize('params', [{'symbol':'1'},{'symbol':510300},{'symbol':'510300','period':'weekly'},{'symbol':'510300','start_date':'20240231'},{'symbol':'510300','unexpected':1}])
def test_invalid_sdk_parameters_fail_before_network(params):
    with pytest.raises(CenterError): adapter.validate_sdk_params('fund_etf_hist_em',params)


def test_unregistered_sdk_function_is_not_invoked():
    with pytest.raises(CenterError): adapter.validate_sdk_params('unregistered_function',{})


def test_worker_enforces_request_cap_and_restores_session(tmp_path,monkeypatch):
    ak=pytest.importorskip('akshare')
    import requests
    original=requests.sessions.Session.request
    calls=[]
    monkeypatch.setattr(worker.SharedQuota,'acquire',lambda *a,**k:nullcontext())
    monkeypatch.setattr(worker,'request',lambda *a: calls.append(a) or '{}')
    def fake_api(**params):
        requests.get('https://fund.eastmoney.com/fixture')
        requests.get('https://fund.eastmoney.com/fixture')
        return pd.DataFrame()
    monkeypatch.setattr(ak,'fund_etf_hist_em',fake_api)
    payload={'root':str(tmp_path),'source':akshare_source().model_dump(mode='json'),'interface':akshare_interfaces()[0].model_dump(mode='json'),'params':{'symbol':'510300'},'max_http_requests':1}
    with pytest.raises(CenterError,match='请求上限'): worker.execute(payload)
    assert len(calls)==1
    assert requests.sessions.Session.request is original


def test_subprocess_timeout_fails_closed(tmp_path,monkeypatch):
    store=SourceStore(tmp_path); store.seed()
    def timeout(*args,**kwargs): raise subprocess.TimeoutExpired(args[0],1)
    monkeypatch.setattr(adapter.subprocess,'run',timeout)
    with pytest.raises(CenterError,match='超时'):
        adapter.fetch_sdk(store,akshare_source(),akshare_interfaces()[0],{'symbol':'510300'},sample=True)

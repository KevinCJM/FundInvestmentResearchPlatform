"""One explicit Tushare request through ETL; artifacts remain temporary."""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.data_sources import acquisition, etl_service, runtime
from backend.data_sources.credentials import read_credential, save_credential
from backend.data_sources.etl_store import EtlStore
from backend.data_sources.models import CenterError, InterfaceConfig, SourceConfig
from backend.data_sources.store import SourceStore
from backend.services.refresh_runtime import InterProcessFileLock


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--confirm-network', action='store_true')
    args = parser.parse_args()
    if not args.confirm_network:
        parser.error('--confirm-network is required; no request was made')
    lock = InterProcessFileLock(ROOT / 'data' / '.tushare_refresh.lock')
    if not lock.acquire(owner='etl-single-request-smoke'):
        print(json.dumps({'status':'skipped','reason':'another data task is active'})); return 2
    original = acquisition.fetch_with_retry
    try:
        live = SourceStore(ROOT / 'data')
        source = SourceConfig.model_validate(live.get('source', 'tushare')['config'])
        interface = InterfaceConfig.model_validate(live.get('interface', 'tushare.trade_cal')['config'])
        if source.base_url != 'https://api.tushare.pro' or interface.api_name != 'trade_cal' or interface.path or interface.method != 'POST':
            raise CenterError('SMOKE_DESTINATION_CHANGED', '单请求验证仅允许已核实的官方交易日历接口。')
        secret = read_credential(live, 'tushare')
        calls = []
        def once(_temporary_store, frozen_source, frozen_interface, params):
            if calls:
                raise CenterError('SMOKE_REQUEST_CAP', '本次测试只允许一个真实请求。')
            calls.append(1)
            return runtime.fetch_with_retry(live, frozen_source, frozen_interface, params, sample=True)
        acquisition.fetch_with_retry = once
        with tempfile.TemporaryDirectory(prefix='fund-etl-smoke-') as path:
            store = SourceStore(Path(path)); store.seed()
            store.save(source, store.get('source', 'tushare')['revision'])
            interface.pagination.mode = 'none'
            store.save(interface, store.get('interface', interface.id)['revision'])
            save_credential(store, 'tushare', secret)
            definition = {'name':'单请求 ETL 验证','max_runtime_seconds':120,'steps':[
                {'id':'download','name':'下载交易日历','kind':'download','source_id':'tushare','interface_id':interface.id,'interface_revision':store.get('interface',interface.id)['revision'],'mode':'full','params':{'exchange':'SSE','start_date':'20240102','end_date':'20240105'}},
                {'id':'mapping','name':'字段映射','kind':'map','inputs':['download']},
                {'id':'resolve','name':'取值检查','kind':'resolve','inputs':['mapping'],'table_id':'master.trading_calendar','include_history':False},
            ]}
            run = etl_service.start(store, {'definition':definition,'confirm':True,'request_id':str(uuid.uuid4())})
            journal = EtlStore(store)
            while run['status'] == 'RUNNING':
                time.sleep(.1)
                run = journal.get_run(run['run_id'])
            ok = run['status'] == 'SUCCEEDED' and len(calls) == 1
            print(json.dumps({'status':'passed' if ok else 'failed','real_requests':len(calls),'steps':[{'name':s['name'],'status':s['status'],'rows':s.get('rows')} for s in run['steps']], 'error':run.get('error'),'temporary_only':True,'published':False,'execution_hash':run['frozen']['execution_fingerprint']},ensure_ascii=False))
            return 0 if ok else 1
    except CenterError as exc:
        print(json.dumps({'status':'failed','code':exc.code,'message':exc.message,'temporary_only':True},ensure_ascii=False)); return 1
    finally:
        acquisition.fetch_with_retry = original
        lock.release()


if __name__ == '__main__':
    raise SystemExit(main())

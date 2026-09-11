"""Explicit one-request smoke; stores all artifacts in a temporary directory."""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.data_sources.akshare_presets import akshare_interfaces, akshare_source
from backend.data_sources.batches import capture_batch
from backend.data_sources.credentials import save_credential
from backend.data_sources.models import CenterError
from backend.data_sources.presets import default_interfaces, default_source
from backend.data_sources.runtime import fetch_with_retry
from backend.data_sources.store import SourceStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--interface', required=True, choices=['akshare.etf_daily', 'akshare.fund_nav', 'tushare.trade_cal'])
    parser.add_argument('--confirm-network', action='store_true')
    args = parser.parse_args()
    if not args.confirm_network:
        parser.error('--confirm-network is required; no network request was made')
    source = akshare_source() if args.interface.startswith('akshare.') else default_source()
    configs = akshare_interfaces() if source.id == 'akshare' else default_interfaces()
    interface = next(i.model_copy(deep=True) for i in configs if i.id == args.interface)
    interface.params.update({'start_date': '20240102', 'end_date': '20240105'})
    if source.id == 'tushare':
        interface.params.update(exchange='SSE')
    fingerprint = hashlib.sha256(json.dumps(interface.model_dump(mode='json'), sort_keys=True).encode()).hexdigest()
    with tempfile.TemporaryDirectory(prefix='fund-source-smoke-') as directory:
        store = SourceStore(Path(directory)); store.seed()
        if source.id == 'tushare':
            token = ROOT / 'data' / '.tushare_token'
            if not token.is_file() or token.is_symlink():
                print(json.dumps({'status':'skipped', 'reason':'local credential unavailable'})); return 2
            save_credential(store, source.id, token.read_text().strip())
        try:
            _, rows = fetch_with_retry(store, source, interface, sample=True)
            batch = capture_batch(store, interface, rows, interface.params, fingerprint)
            success = batch['status'] == 'VALIDATED_CANDIDATE' and len(rows) > 0
            print(json.dumps({'interface': interface.id, 'status':'passed' if success else 'failed',
                              'rows': len(rows), 'request_limit': 1, 'configuration_hash': fingerprint,
                              'tables': [{k:r.get(k) for k in ('table_id','status','rows','rejected_rows')} for r in batch['tables']],
                              'temporary_only': True}, ensure_ascii=False))
            return 0 if success else 1
        except CenterError as exc:
            print(json.dumps({'interface':interface.id, 'status':'failed','code':exc.code,'message':exc.message,'request_limit':1,'temporary_only':True}, ensure_ascii=False))
            return 1


if __name__ == '__main__':
    raise SystemExit(main())

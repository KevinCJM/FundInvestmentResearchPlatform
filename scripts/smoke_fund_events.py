"""Explicit bounded live announcement/range smoke through the collector, temp only."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import T01_get_data as script
from backend.data_sources.credentials import read_credential
from backend.data_sources.models import CenterError
from backend.data_sources.runtime import ConfiguredTushareClient
from backend.data_sources.store import SourceStore
from backend.services.refresh_runtime import InterProcessFileLock


def hashes():
    names = ['T01_get_data.py', 'backend/data_sources/fund_events.py',
             'backend/data_sources/fund_event_merge.py',
             'backend/data_sources/runtime.py', 'backend/data_sources/quota.py',
             'backend/data_sources/task_worker.py', 'scripts/smoke_fund_events.py',
             'backend/data_storage.py', 'backend/data_sources/store.py',
             'backend/data_sources/batches.py', 'backend/services/refresh_runtime.py']
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--confirm-network', action='store_true')
    parser.add_argument('--smoke', action='store_true', required=True)
    parser.add_argument('--api', choices=['fund_portfolio', 'fund_div', 'fund_adj'], default='fund_portfolio')
    parser.add_argument('--max-requests', type=int, default=1, help='Default one; increase only with explicit user authorization.')
    parser.add_argument('--announcement-date')
    parser.add_argument('--fund-code', help='Exercise the full-history range path for one fund.')
    parser.add_argument('--start-date')
    parser.add_argument('--end-date')
    parser.add_argument('--universe-file', type=Path, required=True)
    options = parser.parse_args()
    if options.max_requests < 1 or options.max_requests > 100:
        parser.error('Smoke request budget must be between 1 and 100.')
    if not options.confirm_network:
        parser.error('Explicit --confirm-network required; no request performed.')
    history = bool(options.fund_code)
    if history and options.api == 'fund_div':
        parser.error('Dividend smoke uses a single announcement date.')
    if options.api == 'fund_adj' and not history:
        parser.error('Factor smoke requires one fund-code/start-date/end-date.')
    if history:
        if options.announcement_date or not options.start_date or not options.end_date:
            parser.error('Range smoke requires fund-code/start-date/end-date, not announcement-date.')
        start, end = options.start_date, options.end_date
    else:
        if not options.announcement_date or options.start_date or options.end_date:
            parser.error('Specify announcement-date OR fund-code/start-date/end-date.')
        start = end = options.announcement_date
    for value in (start, end):
        pd.to_datetime(value, format='%Y%m%d', errors='raise')
    if start > end:
        parser.error('start-date must not exceed end-date.')
    universe = pd.read_parquet(options.universe_file, columns=['ts_code', 'name', 'found_date'])
    if history:
        universe = universe[universe.ts_code.eq(options.fund_code)].drop_duplicates('ts_code')
        if len(universe) != 1:
            parser.error('Fund not found in local universe; no request performed.')
    store = SourceStore(ROOT / 'data')
    lock = InterProcessFileLock(store.root / '.tushare_refresh.lock')
    if not lock.acquire(owner='bounded-fund-event-smoke'):
        raise CenterError('DATA_TASK_RUNNING', '已有任务运行，未发起请求。')
    before = hashes()
    try:
        # Reuse the actual account-wide quota store; no live batch capture or
        # snapshot writes. A private quota DB would bypass other account calls.
        client = ConfiguredTushareClient(read_credential(store, 'tushare'), root=store.root, capture=False)
        calls = 0
        original_call = client._call
        def counted_call(*args, **kwargs):
            nonlocal calls
            if calls >= options.max_requests:
                raise CenterError('SMOKE_BUDGET', '达到真实请求预算。')
            calls += 1
            return original_call(*args, **kwargs)
        client._call = counted_call
        with tempfile.TemporaryDirectory(prefix='fund-event-smoke-') as temporary:
            output = Path(temporary)
            # Full-history mode must exercise the real range dispatch, which
            # T01's --smoke replaces with a daily request. Its independent
            # request budget=1 still forbids splitting, retries and empty rechecks.
            args = script.parse_args(([] if history else ['--smoke']) + ['--output-dir', temporary,
                                      '--end-date', end, '--start-date', start,
                                      '--max-retries', '1', '--max-workers', '1',
                                      '--fund-event-max-requests', str(options.max_requests),
                                      '--fund-event-max-runtime', '90'])
            args.limit = None  # Only one market-announcement request, no universe request.
            args.source_configuration_hash = client.configuration_hash
            download, filename = {
                'fund_portfolio': (script.save_fund_portfolio, 'fund_portfolio_df.parquet'),
                'fund_div': (script.save_fund_dividend, 'fund_dividend_df.parquet'),
                'fund_adj': (script.save_fund_adjustment, 'fund_adj_factor_df.parquet'),
            }[options.api]
            info = {'etf_info' if options.api == 'fund_adj' else 'fund_info': universe}
            with contextlib.redirect_stdout(io.StringIO()):
                download(client, output, script.RateLimiter(20), args, **info)
            frame = pd.read_parquet(output / filename)
            if frame.empty:
                raise CenterError('SMOKE_EMPTY_INCONCLUSIVE', '单次空响应不能证明在线下载正确。')
            if (not frame.available_at.between(pd.Timestamp(start), pd.Timestamp(end)).all()
                    or not frame.source_api.eq(options.api).all()):
                raise CenterError('SMOKE_LINEAGE', '单次响应时点或来源不符。')
            if history and not frame.ts_code.eq(options.fund_code).all():
                raise CenterError('SMOKE_IDENTITY', '单基金响应身份不符。')
            if hashes() != before:
                raise CenterError('SMOKE_CODE_CHANGED', '测试期间计算链路代码改变，不能采纳验证结果。')
            print(json.dumps(dict(status='passed', api=options.api, requests=calls, rows=len(frame),
                                  strategy='etf_price_factor_history' if options.api == 'fund_adj' else 'fund_announcement_history' if history else 'announcement',
                                  start_date=start, end_date=end, fund_code=options.fund_code,
                                  returned_available_min=str(frame.available_at.min()),
                                  returned_available_max=str(frame.available_at.max()),
                                  returned_observation_min=str(frame.observation_date.min()),
                                  temporary_only=True,
                                  published=False, collector_hashes=before), ensure_ascii=False))
    finally:
        lock.release()


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(json.dumps({'status':'failed', 'code':getattr(exc, 'code', type(exc).__name__),
                          'message':exc.message if isinstance(exc, CenterError) else 'Smoke failed; raw exception omitted.',
                          'temporary_only':True, 'published':False}))
        sys.exit(1)

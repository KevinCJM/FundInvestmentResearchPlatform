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
             'backend/data_sources/fund_event_conflicts.py',
             'backend/data_sources/event_coverage.py',
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
    parser.add_argument('--verify-market-pages', action='store_true', help='Explicit bounded full-day pagination test, including duplicate-page verification; temporary output only.')
    parser.add_argument('--trade-date', help='Exercise bounded full-market ETF factor pages for one trading day.')
    parser.add_argument('--test-offset-pagination', action='store_true', help='Test a private none-to-offset configuration; never save it.')
    parser.add_argument('--fund-code', help='Exercise the full-history range path for one fund.')
    parser.add_argument('--start-date')
    parser.add_argument('--end-date')
    parser.add_argument('--universe-file', type=Path, required=True)
    options = parser.parse_args()
    request_ceiling = 1000 if options.verify_market_pages else 100
    if options.max_requests < 1 or options.max_requests > request_ceiling:
        parser.error(f'Smoke request budget must be between 1 and {request_ceiling}.')
    if not options.confirm_network:
        parser.error('Explicit --confirm-network required; no request performed.')
    history = bool(options.fund_code)
    factor_day = bool(options.trade_date)
    if options.verify_market_pages and (options.api != 'fund_portfolio' or not options.announcement_date or history or factor_day):
        parser.error('Market pagination verification requires fund_portfolio/announcement-date only.')
    if factor_day and (options.api != 'fund_adj' or history or options.announcement_date or options.start_date or options.end_date):
        parser.error('trade-date is only for ETF factor daily paging, without a fund/range.')
    if options.test_offset_pagination and not factor_day:
        parser.error('Temporary paging configuration is only allowed with fund_adj/trade-date.')
    if history and options.api == 'fund_div':
        parser.error('Dividend smoke uses a single announcement date.')
    if options.api == 'fund_adj' and not history and not factor_day:
        parser.error('Factor smoke requires one fund-code/start-date/end-date.')
    if factor_day:
        start = end = options.trade_date
    elif history:
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
        if options.test_offset_pagination:
            candidate = client.interfaces['fund_adj'].model_copy(deep=True)
            if candidate.pagination.mode not in {'none', 'offset'}:
                raise CenterError('SMOKE_CONFIGURATION', '只允许测试已核定的 offset 分页启用。')
            candidate.pagination.mode = 'offset'
            client.interfaces['fund_adj'] = candidate
        calls = 0
        original_call = client._call
        def counted_call(*args, **kwargs):
            nonlocal calls
            if calls >= options.max_requests:
                raise CenterError('SMOKE_BUDGET', '达到真实请求预算。')
            calls += 1
            if options.verify_market_pages and calls % 25 == 0:
                print(f'[SMOKE] 已执行 {calls}/{options.max_requests} 次临时分页验证请求。', file=sys.stderr, flush=True)
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
                                      '--fund-event-max-runtime', '600' if options.verify_market_pages else '90'])
            args.limit = None  # Only one market-announcement request, no universe request.
            args.source_configuration_hash = client.configuration_hash
            if factor_day:
                args.latest = True
                args.smoke = False
                args.automatic_start_date = start
                pd.DataFrame([{'exchange': 'SSE', 'cal_date': start, 'is_open': 1}]).to_parquet(output / 'trade_day_df.parquet')
            if options.verify_market_pages:
                args.latest, args.smoke = True, False
                args.automatic_start_date = start
            download, filename = {
                'fund_portfolio': (script.save_fund_portfolio, 'fund_portfolio_df.parquet'),
                'fund_div': (script.save_fund_dividend, 'fund_dividend_df.parquet'),
                'fund_adj': (script.save_fund_adjustment, 'fund_adj_factor_df.parquet'),
            }[options.api]
            info = {'etf_info' if options.api == 'fund_adj' else 'fund_info': universe}
            policy = getattr(client, options.api).download_policy
            with contextlib.redirect_stdout(io.StringIO()):
                download(client, output, script.RateLimiter(policy.requests_per_minute if options.verify_market_pages else 20), args, **info)
            frame = pd.read_parquet(output / filename)
            if frame.empty:
                raise CenterError('SMOKE_EMPTY_INCONCLUSIVE', '单次空响应不能证明在线下载正确。')
            if (not frame.available_at.between(pd.Timestamp(start), pd.Timestamp(end)).all()
                    or not frame.source_api.eq(options.api).all()):
                raise CenterError('SMOKE_LINEAGE', '单次响应时点或来源不符。')
            if history and not frame.ts_code.eq(options.fund_code).all():
                raise CenterError('SMOKE_IDENTITY', '单基金响应身份不符。')
            quality_file = output / 'fund_portfolio_df.parquet.quality.meta.json'
            quality = json.loads(quality_file.read_text()) if quality_file.exists() else None
            if quality:
                from backend.data_sources.fund_event_conflicts import VALUES
                quarantined = frame[frame.availability_status.eq('source_conflict')]
                if len(quarantined) != quality['conflicting_keys'] or not quarantined[VALUES].isna().all().all():
                    raise CenterError('SMOKE_QUARANTINE', '冲突隔离证据或空值占位未通过验证。')
            if hashes() != before:
                raise CenterError('SMOKE_CODE_CHANGED', '测试期间计算链路代码改变，不能采纳验证结果。')
            print(json.dumps(dict(status='passed', api=options.api, requests=calls, rows=len(frame),
                                  strategy='etf_factor_date_pages' if factor_day else 'etf_price_factor_history' if options.api == 'fund_adj' else 'fund_announcement_history' if history else 'announcement',
                                  test_offset_pagination=options.test_offset_pagination,
                                  verified_market_pages=options.verify_market_pages,
                                  data_quality=quality,
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

"""Offset protocol checks use offline vendor responses, never live requests."""
from types import SimpleNamespace

import pandas as pd
import pytest

import T01_get_data as script
from backend.data_sources.models import CenterError, DownloadPolicy, Pagination


@pytest.mark.parametrize('case', ['overlap', 'oversized', 'wrong_day', 'code', 'negative', 'missing', 'limit', 'disabled'])
def test_invalid_factor_pages_fail_before_filtering_or_writing(tmp_path, monkeypatch, case):
    calls = []
    def fetch(**params):
        calls.append(params)
        start = params['offset']
        if case == 'overlap' and start: start = 1
        rows = [{'ts_code': f'51000{i}.SH', 'trade_date': '20260904', 'adj_factor': 1.5}
                for i in range(start, start + (3 if case == 'oversized' else 2))]
        if case == 'wrong_day': rows[0]['trade_date'] = '20260903'
        if case == 'code': rows[0]['ts_code'] = 'bad'
        if case == 'negative': rows[1]['adj_factor'] = -1  # Outside the selected ETF universe, still invalid.
        if case == 'missing': del rows[1]['adj_factor']
        return pd.DataFrame(rows)
    fetch.pagination_config = Pagination(mode='none' if case == 'disabled' else 'offset', page_size=2, max_pages=2)
    fetch.download_policy = DownloadPolicy(max_rows_per_request=3)
    args = script.parse_args(['--latest', '--end-date', '20260904', '--max-retries', '1', '--max-workers', '1'])
    monkeypatch.setattr(script, 'load_open_trade_dates', lambda *a, **kw: ['20260904'])
    with pytest.raises((CenterError, ValueError)):
        script.save_fund_adjustment(SimpleNamespace(fund_adj=fetch), tmp_path, script.RateLimiter(100000), args,
                                   etf_info=pd.DataFrame([{'ts_code': '510000.SH', 'name': 'ETF'}]))
    assert len(calls) <= 2
    if case == 'disabled': assert not calls
    assert not (tmp_path / 'fund_adj_factor_df.parquet').exists()

import { act, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter, useLocation } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import IndexData from './IndexData';

function LocationProbe() {
  const location = useLocation();
  return <output data-testid="location">{location.pathname}{location.search}</output>;
}

describe('IndexData', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn().mockImplementation(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url === '/api/data/refresh/status') {
        return { ok: true, json: async () => ({
          source: 'tushare', enabled: true, full_refresh_enabled: true,
          available_modules: ['base', 'etf', 'fund', 'index'], token_configured: true,
          token_configuration_enabled: true, token_editable: true,
          job: { status: 'idle', message: '尚未启动更新' }, datasets: {},
        }) };
      }
      if (url === '/api/indices/summary') {
        return { ok: true, json: async () => ({
          schema_version: 1, status: 'complete', catalog_count: 2, source_count: 2,
          covered_count: 1, latest_date: '2026-08-31', missing_count: 1, stale_count: 0,
          datasets: [{ key: 'index_domestic', scope: 'domestic', file: 'index_daily_df.parquet', exists: true, rows: 100 }],
        }) };
      }
      if (url.startsWith('/api/indices?')) {
        return { ok: true, json: async () => ({
          schema_version: 1, status: 'complete', page: 1, page_size: 20, total: 1,
          filters: { source_api: ['index_basic'], category: ['规模指数'], market: ['CSI'], coverage_status: ['ready'] },
          items: [{ source_api: 'index_basic', quote_source_api: 'index_daily', ts_code: '000300.SH', name: '沪深300', category: '规模指数', market: 'CSI', publisher: '中证指数公司', list_date: '2005-04-08', exp_date: null, first_date: '2005-04-08', latest_date: '2026-08-31', rows: 5000, coverage_status: 'ready' }],
        }) };
      }
      return { ok: false, status: 404, json: async () => ({}) };
    }));
  });

  afterEach(() => vi.unstubAllGlobals());

  it('展示 KPI、健康卡和可访问目录表', async () => {
    render(<MemoryRouter><IndexData /></MemoryRouter>);

    expect(await screen.findByRole('heading', { name: '指数数据中心' })).toBeInTheDocument();
    expect(await screen.findByText('沪深300')).toBeInTheDocument();
    expect(screen.getAllByText('2', { selector: 'p' })).toHaveLength(2);
    expect(screen.getByText('境内')).toBeInTheDocument();
    expect(screen.getByRole('table', { name: '指数目录、来源与行情覆盖状态' })).toBeInTheDocument();
    expect(screen.getAllByText('可用').length).toBeGreaterThanOrEqual(2);
  });

  it('筛选条件同步 URL', async () => {
    const user = userEvent.setup();
    render(<MemoryRouter initialEntries={['/index-data']}><LocationProbe /><IndexData /></MemoryRouter>);
    await screen.findByText('沪深300');

    await act(async () => { await user.selectOptions(screen.getByLabelText('来源'), 'index_basic'); });

    await waitFor(() => expect(screen.getByTestId('location')).toHaveTextContent('source=index_basic'));
    expect(screen.getByTestId('location')).toHaveTextContent('page=1');
    await waitFor(() => expect(vi.mocked(fetch).mock.calls.filter(([input]) => String(input).includes('source=index_basic')).length).toBeGreaterThan(0));
  });
});

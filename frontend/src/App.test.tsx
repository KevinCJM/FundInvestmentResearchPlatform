import { act, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import App from './App';

vi.mock('echarts-for-react', () => ({ default: () => <div data-testid="chart" /> }));

const fixedExecution = {
  execution_backend: 'numba_njit_fixed_signature',
  nopython: true,
  object_mode: 0,
  python_fallback: 0,
  request_time_compilation: 0,
  kernel_signatures: { dashboard_kernel: ['fixed'] },
};

const summary = {
  share_code_count: 1,
  active_count: 1,
  issuing_count: 0,
  inactive_count: 0,
  unknown_status_count: 0,
  unique_managements: 1,
  nav_covered_count: 1,
  nav_coverage_rate: 1,
  latest_nav_date: '2026-08-28',
};

const segment = (kind: 'etf' | 'fund') => ({
  availability: 'ready',
  summary,
  distributions: { fund_type: [], invest_type: [], market: [], status: [], management: [] },
  event_trend: {
    date_field: kind === 'etf' ? 'list_date' : 'found_date',
    label: kind === 'etf' ? 'ETF上市趋势' : '场外公募基金成立趋势',
    points: [],
  },
});

const analyticsResponse = {
  schema_version: 1,
  kind: 'all',
  status: 'complete',
  as_of: '2026-08-28',
  summary: { all: { ...summary, share_code_count: 2 }, etf: summary, fund: summary },
  segments: { etf: segment('etf'), fund: segment('fund') },
  available_filters: { fund_type: [], invest_type: [], status: [], management: [], market: [] },
  data_quality: {
    snapshot: { exists: true, rows: 2, updated_at: '2026-08-28', as_of: '2026-08-28' },
    segments: {
      etf: { info_file_exists: true, info_rows: 1, snapshot_rows: 1, nav_coverage_rate: 1, warnings: [] },
      fund: { info_file_exists: true, info_rows: 1, snapshot_rows: 1, nav_coverage_rate: 1, warnings: [] },
    },
    warnings: [],
  },
  metric_definitions: { return_1y: { label: '近1年收益率', unit: 'ratio', source: 'adj_nav' } },
  execution: fixedExecution,
};

const refreshResponse = {
  source: 'tushare',
  enabled: true,
  full_refresh_enabled: true,
  available_modules: ['base', 'etf', 'fund'],
  token_configured: true,
  token_source: 'frontend_local',
  token_configuration_enabled: true,
  token_editable: true,
  job: { job_id: null, status: 'idle', message: '尚未启动更新' },
  datasets: {
    etf_candle: { file: 'etf_daily_candle_df.parquet', exists: true, rows: 1, latest_date: '2026-08-28' },
  },
};

describe('App', () => {
  let candidateAvailable = false;
  let tokenConfigured = true;

  beforeEach(() => {
    window.history.replaceState({}, '', '/settings/data-sources');
    candidateAvailable = false;
    tokenConfigured = true;
    vi.stubGlobal('confirm', vi.fn(() => true));
    vi.stubGlobal('fetch', vi.fn().mockImplementation(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url === '/api/data/refresh/status') {
        const currentRefreshResponse = { ...refreshResponse, token_configured: tokenConfigured };
        return {
          ok: true,
          json: async () => candidateAvailable
            ? {
                ...currentRefreshResponse,
                job: {
                  job_id: 'job-candidate',
                  status: 'failed',
                  message: '数据抓取完成，分析快照构建失败',
                  fetch_complete: true,
                  staging_data_dir: '/tmp/tushare-candidate',
                  analytics_snapshot: { status: 'failed', error: 'snapshot failed' },
                },
              }
            : currentRefreshResponse,
        };
      }
      if (url === '/api/data/token' && init?.method === 'PUT') {
        tokenConfigured = true;
        return { ok: true, json: async () => ({ token_configured: true, token_source: 'frontend_local' }) };
      }
      if (url === '/api/data/token' && init?.method === 'DELETE') {
        tokenConfigured = false;
        return { ok: true, json: async () => ({ token_configured: false, token_source: 'none' }) };
      }
      if (url === '/api/data/refresh' && init?.method === 'POST') {
        return {
          ok: true,
          json: async () => ({
            ...refreshResponse,
            job: { job_id: 'job-1', status: 'succeeded', mode: 'incremental', started_at: new Date(Date.now() - 60_000).toISOString(), finished_at: new Date().toISOString(), message: 'Tushare 数据更新完成' },
          }),
        };
      }
      if (url === '/api/data/analytics/rebuild' && init?.method === 'POST') {
        return { ok: true, json: async () => ({ status: 'succeeded', rows: 2 }) };
      }
      if (url.startsWith('/api/instruments/analytics/trend?')) {
        return { ok: true, json: async () => ({ schema_version: 1, kind: 'all', status: 'complete', series: { etf: segment('etf').event_trend, fund: segment('fund').event_trend }, available_values: [], data_quality: { warnings: [] }, execution: fixedExecution }) };
      }
      if (url.startsWith('/api/instruments/analytics/rankings?')) {
        const kind = url.includes('kind=fund') ? 'fund' : 'etf';
        return { ok: true, json: async () => ({ schema_version: 1, kind, status: 'unavailable', metric: 'return_1y', metric_definition: { label: '近1年收益率', unit: 'ratio', source: 'adj_nav' }, sort_dir: 'desc', page: 1, page_size: 10, total: 0, as_of: null, items: [], data_quality: { warnings: [] }, execution: fixedExecution }) };
      }
      if (url.startsWith('/api/instruments/analytics?')) {
        return { ok: true, json: async () => analyticsResponse };
      }
      return { ok: false, status: 404, json: async () => ({}) };
    }));
  });

  afterEach(() => {
    window.history.replaceState({}, '', '/');
    vi.unstubAllGlobals();
  });

  it('在产品研究流程中渲染基金市场概览', async () => {
    window.history.replaceState({}, '', '/product-research/panorama');
    render(<App />);
    expect(screen.getByRole('link', { name: '产品研究' })).toBeInTheDocument();
    await waitFor(() => expect(screen.getByRole('heading', { name: '基金投研全景驾驶舱' })).toBeInTheDocument());
    expect(screen.getByRole('tab', { name: '全市场' })).toHaveAttribute('aria-selected', 'true');
    expect(await screen.findByText('场外公募基金市场镜头')).toBeInTheDocument();
    expect(screen.queryByRole('heading', { name: '数据下载与更新' })).not.toBeInTheDocument();
    expect(vi.mocked(fetch).mock.calls.some(([input]) => String(input).startsWith('/api/data/'))).toBe(false);
  });

  it('从设置的数据管理页启动 Tushare 增量更新', async () => {
    const user = userEvent.setup();
    render(<App />);
    await screen.findByText(/尚未启动更新/);
    expect(screen.getByRole('region', { name: '数据下载与更新' })).toBeVisible();
    expect(screen.queryByRole('button', { name: /数据管理：查看明细并拉取最新数据/ })).not.toBeInTheDocument();
    const button = await screen.findByRole('button', { name: '开始数据更新' });
    await waitFor(() => expect(button).toBeEnabled());

    await act(async () => { await user.click(button); });

    await waitFor(() => expect(fetch).toHaveBeenCalledWith('/api/data/refresh', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        modules: ['base', 'etf', 'fund'],
        mode: 'incremental',
        module_scopes: {
          base: ['calendar', 'stock_basic', 'fund_company'],
          etf: ['info', 'nav', 'share', 'candle'],
          fund: ['info', 'nav', 'manager', 'scale', 'benchmark'],
        },
      }),
    }));
    expect(confirm).not.toHaveBeenCalled();
    await waitFor(() => expect(screen.getByText(/刚刚完成 · 增量更新/)).toBeInTheDocument());
    const analyticsCalls = vi.mocked(fetch).mock.calls.filter(([input]) => String(input).startsWith('/api/instruments/analytics?'));
    expect(analyticsCalls).toHaveLength(0);
  });

  it('可在设置的数据管理页保存 Token 后启用数据下载', async () => {
    tokenConfigured = false;
    const user = userEvent.setup();
    render(<App />);
    await screen.findByText(/尚未启动更新/);
    const refreshButton = await screen.findByRole('button', { name: '开始数据更新' });
    expect(refreshButton).toBeDisabled();

    const tokenInput = screen.getByLabelText('输入 Token');
    await user.type(tokenInput, 'frontend-token-1234567890');
    await user.click(screen.getByRole('button', { name: '保存 Token' }));

    await waitFor(() => expect(fetch).toHaveBeenCalledWith('/api/data/token', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ token: 'frontend-token-1234567890' }),
    }));
    await waitFor(() => expect(screen.getByText('已配置')).toBeInTheDocument());
    expect(tokenInput).toHaveValue('');
    expect(refreshButton).toBeEnabled();
  });

  it('清除 Token 后保留本地数据但禁用新的下载任务', async () => {
    const user = userEvent.setup();
    render(<App />);
    await screen.findByText(/尚未启动更新/);
    await user.click(screen.getByRole('button', { name: '清除 Token' }));

    expect(confirm).toHaveBeenCalledWith(expect.stringContaining('已下载的本地数据不会删除'));
    await waitFor(() => expect(fetch).toHaveBeenCalledWith('/api/data/token', { method: 'DELETE' }));
    await waitFor(() => expect(screen.getByText('未配置')).toBeInTheDocument());
    expect(screen.getByRole('button', { name: '开始数据更新' })).toBeDisabled();
  });

  it('下载页不承载数据质量明细并提供独立入口', async () => {
    render(<App />);
    await screen.findByText(/尚未启动更新/);
    expect(screen.getByRole('link', { name: '查看数据质量 →' })).toHaveAttribute('href', '/settings/data-quality');
    expect(screen.queryByRole('heading', { name: '数据健康' })).not.toBeInTheDocument();
    expect(screen.queryByText('信息行数')).not.toBeInTheDocument();
    expect(screen.queryByText('净值覆盖')).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: '仅重建分析快照' })).not.toBeInTheDocument();
    expect(vi.mocked(fetch).mock.calls.some(([input]) => String(input).startsWith('/api/instruments/analytics?'))).toBe(false);
  });

  it('候选数据可重建并接入候选快照且不会再次请求 Tushare', async () => {
    candidateAvailable = true;
    const user = userEvent.setup();
    render(<App />);
    await screen.findByText(/数据抓取完成，分析快照构建失败/);
    await user.click(await screen.findByRole('button', { name: '重建并接入候选快照' }));

    await waitFor(() => expect(fetch).toHaveBeenCalledWith(
      '/api/data/analytics/rebuild',
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ candidate: true }),
      },
    ));
    const refreshCalls = vi.mocked(fetch).mock.calls.filter(([input]) => String(input) === '/api/data/refresh');
    expect(refreshCalls).toHaveLength(0);
  });

  it('数据管理区域默认常驻展示且没有展开收起入口', async () => {
    render(<App />);
    await screen.findByText(/尚未启动更新/);
    expect(screen.getByRole('region', { name: '数据下载与更新' })).toBeVisible();
    expect(screen.getByLabelText('输入 Token')).toBeVisible();
    expect(screen.getByRole('button', { name: '开始数据更新' })).toBeVisible();
    expect(screen.queryByRole('button', { name: /数据管理：查看明细并拉取最新数据/ })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /下载内容/ })).not.toBeInTheDocument();
  });

  it('全量更新必须二次确认', async () => {
    const user = userEvent.setup();
    render(<App />);
    await screen.findByText(/尚未启动更新/);
    await user.click(screen.getByRole('radio', { name: /^全量更新/ }));
    await user.click(screen.getByRole('button', { name: '开始数据更新' }));

    expect(confirm).toHaveBeenCalledTimes(1);
    expect(confirm).toHaveBeenCalledWith(expect.stringContaining('全量重建可能运行数小时'));
  });
});

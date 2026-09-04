import { act, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter, Route, Routes, useLocation, useNavigate } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { formatMetricValue } from '../components/dashboard/DashboardRankingTable';
import Dashboard from './Dashboard';

vi.mock('echarts-for-react', () => ({ default: () => <div data-testid="chart" /> }));

const fixedExecution = {
  execution_backend: 'numba_njit_fixed_signature',
  nopython: true,
  object_mode: 0,
  python_fallback: 0,
  request_time_compilation: 0,
  kernel_signatures: { dashboard_kernel: ['fixed'] },
};

const baseSummary = {
  share_code_count: 2,
  active_count: 1,
  issuing_count: 0,
  inactive_count: 1,
  unknown_status_count: 0,
  unique_managements: 1,
  nav_covered_count: 1,
  nav_coverage_rate: 0.5,
  issue_amount_total: 100,
  issue_amount_coverage_rate: 1,
  latest_nav_date: '2026-08-29',
  latest_candle_date: '2026-08-30',
  index_covered_count: 1,
  index_coverage_rate: 0.5,
  liquidity_covered_count: 1,
  liquidity_coverage_rate: 0.5,
  purchase_redemption_covered_count: 1,
  purchase_redemption_coverage_rate: 0.5,
};

const createSegment = (kind: 'etf' | 'fund', missing = false) => ({
  availability: missing ? 'missing' : 'ready',
  summary: baseSummary,
  distributions: {
    fund_type: [{ name: kind === 'etf' ? '股票型ETF' : '混合型', value: 2 }],
    invest_type: [{ name: '被动指数型', value: 1 }],
    market: [{ name: kind === 'etf' ? 'SSE' : '场外', value: 2 }],
    status: [{ name: kind === 'etf' ? '上市交易' : '存续', value: 1 }],
    management: [{ name: '示例基金公司', value: 2 }],
    index_name: kind === 'etf' ? [{ name: '沪深300', value: 1 }] : [],
    m_fee: [{ name: '0.25%–0.50%', value: 1 }],
    c_fee: [{ name: '≤0.25%', value: 1 }],
  },
  event_trend: {
    date_field: kind === 'etf' ? 'list_date' : 'found_date',
    label: kind === 'etf' ? 'ETF上市趋势' : '场外公募基金成立趋势',
    points: [{ year: 2026, count: 2, total_issue_amount: 100 }],
  },
  latest_products: [{
    ts_code: kind === 'etf' ? '510300.SH' : '000001.OF',
    name: kind === 'etf' ? '沪深300ETF' : '示例场外基金',
    market: kind === 'etf' ? '上交所' : '场外',
    index_name: kind === 'etf' ? '沪深300' : null,
    fund_type: '混合型',
    invest_type: '被动指数型',
    list_date: kind === 'etf' ? '2026-08-20' : null,
    found_date: kind === 'fund' ? '2026-08-18' : null,
    purc_startdate: kind === 'fund' ? '2026-08-25' : null,
    redm_startdate: kind === 'fund' ? '2026-08-26' : null,
  }],
});

const createAnalytics = (kind: 'all' | 'etf' | 'fund', fundMissing = false) => {
  const etf = createSegment('etf');
  const fund = createSegment('fund', fundMissing);
  return {
    schema_version: 1,
    kind,
    status: fundMissing ? 'partial' : 'complete',
    as_of: '2026-08-29',
    availability: { etf_info: 'ready', fund_info: fundMissing ? 'missing' : 'ready', analysis_snapshot: 'ready' },
    summary: kind === 'all'
      ? { all: { ...baseSummary, share_code_count: 4 }, etf: baseSummary, fund: baseSummary }
      : { [kind]: baseSummary },
    segments: kind === 'all' ? { etf, fund } : { [kind]: kind === 'etf' ? etf : fund },
    available_filters: {
      fund_type: [{ value: '混合型', label: '混合型', count: 2 }],
      invest_type: [], status: [], management: [], market: [],
    },
    data_quality: {
      snapshot: { exists: true, rows: 2, updated_at: '2026-08-29', as_of: '2026-08-29' },
      segments: {
        etf: { info_file_exists: true, info_rows: 2, snapshot_rows: 1, nav_coverage_rate: 0.5, warnings: [] },
        fund: { info_file_exists: !fundMissing, info_rows: fundMissing ? 0 : 2, snapshot_rows: fundMissing ? 0 : 1, nav_coverage_rate: fundMissing ? null : 0.5, warnings: [] },
      },
      warnings: fundMissing ? [{ code: 'FUND_INFO_MISSING', kind: 'fund', message: '场外基金基础信息缺失' }] : [],
    },
    metric_definitions: { return_1y: { label: '近1年收益率', unit: 'ratio', source: 'adj_nav' } },
    units: { return_1y: 'ratio' },
    execution: fixedExecution,
  };
};

function CurrentLocation() {
  const location = useLocation();
  return <output data-testid="location">{location.pathname}{location.search}</output>;
}

function HistoryControls() {
  const navigate = useNavigate();
  return <button type="button" onClick={() => navigate(-1)}>后退测试</button>;
}

describe('Dashboard', () => {
  let fundMissing = false;

  beforeEach(() => {
    fundMissing = false;
    vi.stubGlobal('confirm', vi.fn(() => true));
    vi.stubGlobal('fetch', vi.fn().mockImplementation(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.startsWith('/api/instruments/analytics/trend?')) {
        const kind = new URL(`http://test${url}`).searchParams.get('kind') as 'all' | 'etf' | 'fund';
        const series = kind === 'all'
          ? { etf: createSegment('etf').event_trend, fund: createSegment('fund').event_trend }
          : { [kind]: createSegment(kind).event_trend };
        return { ok: true, json: async () => ({ schema_version: 1, kind, status: 'complete', series, available_values: [], data_quality: { warnings: [] }, execution: fixedExecution }) };
      }
      if (url.startsWith('/api/instruments/analytics/rankings?')) {
        const params = new URL(`http://test${url}`).searchParams;
        const kind = params.get('kind') as 'etf' | 'fund';
        return { ok: true, json: async () => ({
          schema_version: 1, kind, status: fundMissing && kind === 'fund' ? 'unavailable' : 'complete', metric: params.get('metric'),
          metric_definition: { label: '近1年收益率', unit: 'ratio', source: 'adj_nav' }, sort_dir: 'desc', page: 1, page_size: 10,
          total: fundMissing && kind === 'fund' ? 0 : 1, as_of: '2026-08-29',
          items: fundMissing && kind === 'fund' ? [] : [{ instrument_type: kind, ts_code: kind === 'etf' ? '510300.SH' : '000001.OF', name: kind === 'etf' ? '沪深300ETF' : '示例场外基金', management: '示例基金公司', fund_type: '混合型', invest_type: '被动指数型', status: '存续', latest_date: '2026-08-29', observation_count: 250, value: 0.12 }],
          data_quality: { warnings: [] },
          execution: fixedExecution,
        }) };
      }
      if (url.startsWith('/api/instruments/analytics?')) {
        const kind = new URL(`http://test${url}`).searchParams.get('kind') as 'all' | 'etf' | 'fund';
        return { ok: true, json: async () => createAnalytics(kind, fundMissing) };
      }
      return { ok: false, status: 404, json: async () => ({}) };
    }));
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('区分百分比指标与无量纲 Sharpe/Calmar', () => {
    expect(formatMetricValue(0.12, { label: '收益率', unit: 'ratio' }, 'return_1y')).toBe('12.00%');
    expect(formatMetricValue(0.8, { label: '夏普比率', unit: 'ratio' }, 'sharpe_1y')).toBe('0.80');
  });

  it('默认展示双品类、正确日期口径和可访问数据表', async () => {
    render(<MemoryRouter initialEntries={['/']}><Dashboard /></MemoryRouter>);

    expect(await screen.findByText('ETF市场镜头')).toBeInTheDocument();
    expect(screen.getByText('场外公募基金市场镜头')).toBeInTheDocument();
    expect(screen.getByText(/严格按交易所上市日期统计/)).toBeInTheDocument();
    expect(screen.getByText(/严格按成立日期统计；场外基金不使用上市日期/)).toBeInTheDocument();
    expect(screen.getByText('指数标的覆盖率')).toBeInTheDocument();
    expect(screen.getByText('申赎起始日覆盖率')).toBeInTheDocument();
    expect(screen.getByText(/不代表当前开放状态/)).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /数据管理：查看明细并拉取最新数据/ })).not.toBeInTheDocument();
    expect(screen.getAllByText('查看数据表').length).toBeGreaterThan(0);
    expect(screen.getAllByRole('link', { name: '沪深300ETF' }).length).toBeGreaterThan(0);
    expect(screen.getAllByRole('link', { name: '示例场外基金' }).length).toBeGreaterThan(0);

    await waitFor(() => {
      const calls = vi.mocked(fetch).mock.calls.map(([input]) => String(input));
      expect(calls.some((url) => url.startsWith('/api/data/'))).toBe(false);
      expect(calls.some((url) => url.includes('/rankings?') && url.includes('kind=etf') && url.includes('active_only=true'))).toBe(true);
      expect(calls.some((url) => url.includes('/rankings?') && url.includes('kind=fund') && url.includes('active_only=true'))).toBe(true);
    });
  });

  it('通过 URL 页签切换到场外基金并携带筛选进入产品研究', async () => {
    const user = userEvent.setup();
    render(
      <MemoryRouter initialEntries={['/']}>
        <CurrentLocation />
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/product-research/products" element={<div>产品研究占位</div>} />
        </Routes>
      </MemoryRouter>,
    );

    await screen.findByText('ETF市场镜头');
    await act(async () => user.click(await screen.findByRole('tab', { name: '场外公募基金' })));
    await waitFor(() => expect(screen.getByTestId('location')).toHaveTextContent('/?kind=fund'));
    expect(await screen.findByText('场外公募基金市场镜头')).toBeInTheDocument();
    expect(screen.getByText('申赎信息覆盖率')).toBeInTheDocument();
    expect(screen.queryByText('ETF市场镜头')).not.toBeInTheDocument();
    expect((await screen.findAllByRole('link', { name: '示例场外基金' })).length).toBeGreaterThan(0);

    await act(async () => user.click(screen.getByRole('button', { name: '混合型' })));
    await waitFor(() => expect(screen.getByTestId('location')).toHaveTextContent('/product-research/products?kind=fund'));
    expect(screen.getByTestId('location')).toHaveTextContent('fund_type=%E6%B7%B7%E5%90%88%E5%9E%8B');
    expect(screen.getByTestId('location')).toHaveTextContent('sort_by=found_date');
  });

  it('标题区搜索入口携带类别与关键词进入产品研究', async () => {
    const user = userEvent.setup();
    render(
      <MemoryRouter initialEntries={['/?kind=all']}>
        <CurrentLocation />
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/product-research/products" element={<div>产品研究占位</div>} />
        </Routes>
      </MemoryRouter>,
    );

    await user.type(await screen.findByRole('searchbox', { name: '基金代码、名称或管理人' }), '华夏成长');
    await user.click(screen.getByRole('button', { name: '搜场外基金' }));

    await waitFor(() => expect(screen.getByTestId('location')).toHaveTextContent('/product-research/products?kind=fund'));
    expect(screen.getByTestId('location')).toHaveTextContent('q=%E5%8D%8E%E5%A4%8F%E6%88%90%E9%95%BF');
  });

  it('基金数据缺失时保留 ETF 并局部降级', async () => {
    fundMissing = true;
    render(<MemoryRouter initialEntries={['/']}><Dashboard /></MemoryRouter>);

    expect(await screen.findByText('ETF市场镜头')).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: '场外公募基金数据尚未就绪' })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: '设置中的“数据源与下载”' })).toHaveAttribute('href', '/settings/data-sources');
    expect(screen.getByText('部分数据集未就绪，页面已保留可用区块；缺失值以 “--” 展示，不按 0 参与统计。')).toBeInTheDocument();
  });

  it('默认写入 kind=all 且浏览器后退恢复上一范围', async () => {
    const user = userEvent.setup();
    render(
      <MemoryRouter initialEntries={['/']}>
        <CurrentLocation />
        <HistoryControls />
        <Dashboard />
      </MemoryRouter>,
    );
    await waitFor(() => expect(screen.getByTestId('location')).toHaveTextContent('/?kind=all'));
    await user.click(screen.getByRole('tab', { name: 'ETF' }));
    await waitFor(() => expect(screen.getByTestId('location')).toHaveTextContent('/?kind=etf'));
    await user.click(screen.getByRole('tab', { name: '场外公募基金' }));
    await waitFor(() => expect(screen.getByTestId('location')).toHaveTextContent('/?kind=fund'));
    await user.click(screen.getByRole('button', { name: '后退测试' }));
    await waitFor(() => expect(screen.getByRole('tab', { name: 'ETF' })).toHaveAttribute('aria-selected', 'true'));
  });
});

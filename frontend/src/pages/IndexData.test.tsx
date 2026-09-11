import { act, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import DataQuality from './IndexData';

const fixedExecution = {
  execution_backend: 'numba_njit_fixed_signature',
  nopython: true,
  object_mode: 0,
  python_fallback: 0,
  request_time_compilation: 0,
  kernel_signatures: { coverage_ratio_kernel: ['fixed'] },
};

const analyticsPayload = {
  schema_version: 1,
  kind: 'all',
  status: 'partial',
  as_of: '2026-08-31',
  availability: {},
  summary: {
    etf: { index_coverage_rate: 0.92 },
    fund: {},
  },
  segments: {},
  available_filters: {},
  data_quality: {
    snapshot: { exists: true, status: 'ready', rows: 3000, as_of: '2026-08-31' },
    segments: {
      etf: { info_file_exists: true, info_rows: 100, snapshot_rows: 90, nav_coverage_rate: 0.9, warnings: [] },
      fund: { info_file_exists: true, info_rows: 200, snapshot_rows: 160, nav_coverage_rate: 0.8, warnings: [] },
    },
    warnings: [{ code: 'NAV_PARTIAL', message: '部分产品净值历史不足。', kind: 'fund' }],
  },
  metric_definitions: {},
  execution: fixedExecution,
};

const refreshStatusPayload = {
  source: 'tushare', enabled: true, full_refresh_enabled: true,
  available_modules: ['base', 'etf', 'fund', 'index'], token_configured: true,
  token_configuration_enabled: true, token_editable: true,
  job: { status: 'idle', message: '尚未启动更新' },
  datasets: {
    calendar: { file: 'trade_day_df.parquet', exists: true, status: 'ready', rows: 6000, earliest_date: '2010-01-01', latest_date: '2026-08-31', updated_at: '2026-08-31T08:00:00Z' },
    etf_nav: { file: 'etf_daily_df.parquet', exists: true, status: 'ready', rows: 10000, earliest_date: '2010-01-04', latest_date: '2026-08-31', updated_at: '2026-08-31T08:00:00Z' },
    fund_nav: { file: 'fund_nav_df.parquet', exists: false, status: 'missing', rows: 0, earliest_date: null, latest_date: null },
    index_ci: { file: 'index_ci_daily_df.parquet', exists: true, status: 'ready', rows: 0, earliest_date: null, latest_date: null },
    instrument_metrics: { file: 'instrument_metrics_snapshot.parquet', exists: true, status: 'ready', rows: 3000, earliest_date: '2024-01-01', latest_date: '2026-08-31', updated_at: '2026-08-31T08:30:00Z' },
  },
};

const qualityPayload = {
  schema_version: 1,
  status: 'attention',
  generated_at: '2026-09-01T08:30:00Z',
  activated_at: '2026-09-01T08:45:00Z',
  as_of: '2026-08-31',
  summary: {
    checks_total: 9, checks_passed: 5, checks_warning: 4, checks_failed: 0, checks_unavailable: 0,
    total_products: 300, affected_products: 18, affected_rate: 0.06, issue_count: 4,
    critical_issue_count: 0, high_issue_count: 3, medium_issue_count: 1,
    nav_anomaly_products: 3, nav_anomaly_events: 4, stale_active_products: 8,
  },
  checks: [
    { key: 'schema', label: '结构契约', dimension: 'validity', status: 'passed', summary: '必要字段齐全', detail: '检查固定字段契约。', threshold: '必要字段必须 100% 存在' },
    { key: 'primary_key', label: '主键唯一性', dimension: 'integrity', status: 'passed', summary: '产品键无重复', detail: '检查产品和日期复合键。', threshold: '重复键 = 0' },
    { key: 'nav_discontinuity', label: '净值突变', dimension: 'continuity', status: 'warning', summary: '3 个产品 / 4 个异常点', detail: '用参考净值交叉确认跳变。', threshold: '复权变动 > 20% 且参考净值稳定；或单日变动 > 100%' },
    { key: 'series_density', label: '序列连续性', dimension: 'continuity', status: 'warning', summary: '7 个产品需补数', detail: '检查交易日覆盖率。', threshold: '交易日覆盖 ≥ 90%；连续缺失 ≤ 5 个交易日' },
  ],
  issues: [
    {
      id: 'nav-discontinuity', code: 'NAV_DISCONTINUITY', severity: 'high', dimension: 'continuity', scope: 'ETF 与场外基金净值',
      title: '发现净值突变或复权断点', description: '复权净值发生大幅跳变，但参考净值没有同步变化。',
      evidence: '3 个产品共发现 4 个异常点。', impact: '相关区间的收益、波动率和回撤会被判为不可用。',
      affected_count: 3, affected_rate: 0.01, record_count: 4, action: 'inspect_source',
      samples: [{ kind: 'fund', ts_code: '000001.OF', name: '测试基金', latest_date: '2026-08-31', observed: '2 个异常点' }],
    },
    {
      id: 'series-gap', code: 'SERIES_INTERNAL_GAP', severity: 'high', dimension: 'continuity', scope: '近一年产品净值',
      title: '净值序列存在密度不足或连续缺口', description: '近一年窗口未通过连续性检查。',
      evidence: '7 个产品需要补数。', impact: '相关指标不会进入正式排行。',
      affected_count: 7, affected_rate: 0.023, record_count: 0, action: 'refresh', samples: [],
    },
  ],
  validation: { status: 'passed', manifest: 'tushare_active.json' },
  execution: fixedExecution,
};

describe('DataQuality', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn().mockImplementation(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url === '/api/data/refresh/status') return { ok: true, json: async () => refreshStatusPayload };
      if (url === '/api/data/quality') return { ok: true, json: async () => qualityPayload };
      if (url === '/api/instruments/analytics?kind=all') return { ok: true, json: async () => analyticsPayload };
      if (url === '/api/indices/summary') return { ok: true, json: async () => ({
        status: 'partial', catalog_count: 2000, covered_count: 1800, latest_date: '2026-08-31',
        coverage_rate: 0.9, missing_count: 150, stale_count: 50, execution: fixedExecution,
      }) };
      return { ok: false, status: 404, json: async () => ({}) };
    }));
  });

  afterEach(() => { vi.unstubAllGlobals(); });

  it('展示真实深度检查、净值突变证据与检查口径，不再用待接入项冒充质量结论', async () => {
    render(<MemoryRouter><DataQuality /></MemoryRouter>);

    expect(await screen.findByRole('heading', { name: '数据质量监控与治理' })).toBeInTheDocument();
    expect(await screen.findByText('部分指数缺少行情覆盖')).toBeInTheDocument();
    expect(screen.getByText('指数目录覆盖 90%。')).toBeInTheDocument();
    expect(screen.getByText('发现净值突变或复权断点')).toBeInTheDocument();
    expect(screen.getByText('000001.OF')).toBeInTheDocument();
    expect(screen.getByText('相关区间的收益、波动率和回撤会被判为不可用。')).toBeInTheDocument();
    expect(screen.getByText('场外基金净值缺失')).toBeInTheDocument();
    expect(screen.getByText('中信行业行情没有有效记录')).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: '深度检查矩阵' })).toBeInTheDocument();
    expect(screen.getByText('产品键无重复')).toBeInTheDocument();
    expect(screen.getByText('交易日覆盖 ≥ 90%；连续缺失 ≤ 5 个交易日')).toBeInTheDocument();
    expect(screen.queryByText('重复记录规则待接入')).not.toBeInTheDocument();
    const overview = screen.getByRole('region', { name: '数据质量概览' });
    for (const metricLabel of ['深度检查规则', '受影响产品', '净值突变', '待处理问题']) {
      const metric = within(overview).getByText(metricLabel).closest('article');
      expect(metric).not.toBeNull();
      expect(within(metric!).getByText('需关注')).toBeInTheDocument();
    }
    expect(screen.getByRole('heading', { name: '问题工作台' })).toBeInTheDocument();
    expect(screen.getByText('待接入 · 不计入当前通过结论')).toBeInTheDocument();
    expect(screen.getByRole('table', { name: '核心数据集质量明细' })).toBeInTheDocument();
    expect(screen.queryByText('指数数据中心')).not.toBeInTheDocument();
    expect(screen.queryByText(/全景驾驶舱/)).not.toBeInTheDocument();
    expect(vi.mocked(fetch).mock.calls.some(([input]) => String(input).startsWith('/api/indices?'))).toBe(false);
  });

  it('支持按风险等级筛选质量问题', async () => {
    const user = userEvent.setup();
    render(<MemoryRouter><DataQuality /></MemoryRouter>);
    expect(await screen.findByText('发现净值突变或复权断点')).toBeInTheDocument();

    await act(async () => { await user.click(screen.getByRole('button', { name: /关注/ })); });

    await waitFor(() => expect(screen.queryByText('发现净值突变或复权断点')).not.toBeInTheDocument());
    expect(screen.getByText('部分指数缺少行情覆盖')).toBeInTheDocument();
  });

  it('支持按数据域筛选并只查看需处理的数据集', async () => {
    const user = userEvent.setup();
    render(<MemoryRouter><DataQuality /></MemoryRouter>);
    const table = await screen.findByRole('table', { name: '核心数据集质量明细' });
    expect(within(table).getByText('ETF 净值')).toBeInTheDocument();

    await act(async () => { await user.click(screen.getByRole('button', { name: '只看需处理' })); });
    expect(screen.getByRole('button', { name: '只看需处理' })).toHaveAttribute('aria-pressed', 'true');
    await waitFor(() => expect(within(table).queryByText('ETF 净值')).not.toBeInTheDocument());
    expect(within(table).getByText('场外基金净值')).toBeInTheDocument();

    await act(async () => { await user.selectOptions(screen.getByLabelText('数据域'), 'index'); });
    await waitFor(() => expect(within(table).queryByText('场外基金净值')).not.toBeInTheDocument());
    expect(within(table).getByText('中信行业行情')).toBeInTheDocument();
  });

  it('深度质量统计缺少固定签名 NJIT 证明时不纳入质量结论', async () => {
    vi.mocked(fetch).mockImplementation(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url === '/api/data/refresh/status') return { ok: true, json: async () => refreshStatusPayload } as Response;
      if (url === '/api/data/quality') return { ok: true, json: async () => ({ ...qualityPayload, execution: undefined }) } as Response;
      if (url === '/api/instruments/analytics?kind=all') return { ok: true, json: async () => analyticsPayload } as Response;
      if (url === '/api/indices/summary') return { ok: true, json: async () => ({
        status: 'partial', catalog_count: 2000, covered_count: 1800, latest_date: '2026-08-31',
        coverage_rate: 0.9, missing_count: 150, stale_count: 50, execution: fixedExecution,
      }) } as Response;
      return { ok: false, status: 404, json: async () => ({}) } as Response;
    });

    render(<MemoryRouter><DataQuality /></MemoryRouter>);

    expect(await screen.findByText('深度数据质量统计未提供有效的固定签名 NJIT 执行证明')).toBeInTheDocument();
    expect(screen.queryByText('产品键无重复')).not.toBeInTheDocument();
    for (const label of ['受影响产品', '净值突变']) {
      const card = within(screen.getByRole('region', { name: '数据质量概览' })).getByText(label).closest('article')!;
      expect(within(card).getByText('未检查')).toBeInTheDocument();
      expect(within(card).queryByText('通过')).not.toBeInTheDocument();
    }
  });

  it('质量接口失败显示未检查，重新检查成功后真实零值可以通过', async () => {
    const originalFetch = vi.mocked(fetch).getMockImplementation()!;
    let failed = true;
    vi.mocked(fetch).mockImplementation(async (input, init) => String(input) === '/api/data/quality'
      ? { ok: !failed, status: failed ? 503 : 200, json: async () => ({ ...qualityPayload, status: 'healthy', issues: [],
        summary: { ...qualityPayload.summary, checks_total: 4, checks_passed: 4, checks_warning: 0, affected_products: 0, nav_anomaly_products: 0, nav_anomaly_events: 0 },
        checks: qualityPayload.checks.map(check => ({ ...check, status: 'passed' })),
      }) } as Response : originalFetch(input, init));
    render(<MemoryRouter><DataQuality /></MemoryRouter>);
    await screen.findByText('深度质量检查结果暂不可用。');
    const overview = screen.getByRole('region', { name: '数据质量概览' });
    for (const label of ['受影响产品', '净值突变']) {
      const card = within(overview).getByText(label).closest('article')!;
      expect(within(card).getByText('--')).toBeInTheDocument();
      expect(within(card).getByText('未检查')).toBeInTheDocument();
    }
    failed = false;
    await userEvent.click(within(screen.getByRole('heading', { name: '数据质量监控与治理' }).closest('header')!).getByRole('button', { name: '重新检查' }));
    await waitFor(() => expect(screen.queryByText('深度质量检查结果暂不可用。')).not.toBeInTheDocument());
    for (const label of ['受影响产品', '净值突变']) {
      const card = within(overview).getByText(label).closest('article')!;
      expect(within(card).getByText('0')).toBeInTheDocument();
      expect(within(card).getByText('通过')).toBeInTheDocument();
    }
  });

  it.each(['unavailable', 'missing', 'passed-in-unavailable-report'] as const)('汇总零值但净值规则 %s 不能通过', async (ruleState) => {
    const originalFetch = vi.mocked(fetch).getMockImplementation()!;
    vi.mocked(fetch).mockImplementation(async (input, init) => String(input) === '/api/data/quality'
      ? { ok: true, json: async () => ({ ...qualityPayload, status: 'unavailable', issues: [],
        summary: { ...qualityPayload.summary, affected_products: 0, nav_anomaly_products: 0, checks_total: 4, checks_passed: 4, checks_unavailable: ruleState === 'passed-in-unavailable-report' ? 0 : 1 },
        checks: ruleState === 'missing' ? [] : qualityPayload.checks.map(check => ({ ...check, status: ruleState === 'passed-in-unavailable-report' ? 'passed' : 'unavailable' })),
      }) } as Response : originalFetch(input, init));
    render(<MemoryRouter><DataQuality /></MemoryRouter>);
    await screen.findByText('部分指数缺少行情覆盖');
    for (const label of ['受影响产品', '净值突变']) {
      const card = within(screen.getByRole('region', { name: '数据质量概览' })).getByText(label).closest('article')!;
      expect(within(card).getByText('未检查')).toBeInTheDocument();
      expect(within(card).getByText('--')).toBeInTheDocument();
    }
    expect(screen.queryByText('已接入规则全部通过')).not.toBeInTheDocument();
  });
});

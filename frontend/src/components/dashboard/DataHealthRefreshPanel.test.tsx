import { act, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, describe, expect, it, vi } from 'vitest';
import DataHealthRefreshPanel from './DataHealthRefreshPanel';

describe('DataHealthRefreshPanel module scopes', () => {
  afterEach(() => vi.unstubAllGlobals());

  it('默认选择情景核心范围并提交扩展范围', async () => {
    const fetchMock = vi.fn().mockImplementation(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url === '/api/data/refresh/status') {
        return { ok: true, json: async () => ({
          source: 'tushare', enabled: true, full_refresh_enabled: true,
          available_modules: ['base', 'etf', 'fund', 'index'],
          available_module_scopes: {
            base: ['calendar', 'stock_basic', 'fund_company'], etf: ['info', 'nav', 'share', 'candle'],
            fund: ['info', 'nav'], index: ['catalog', 'domestic', 'industry', 'concept', 'global', 'futures', 'valuation', 'constituents'],
          },
          default_module_scopes: {
            base: ['calendar', 'stock_basic', 'fund_company'], etf: ['info', 'nav', 'share', 'candle'],
            fund: ['info', 'nav'], index: ['catalog', 'domestic', 'industry', 'global'],
          },
          available_index_scopes: ['catalog', 'domestic', 'industry', 'concept', 'global', 'futures', 'valuation', 'constituents'],
          default_index_scopes: ['catalog', 'domestic', 'industry', 'global'],
          token_configured: true, token_configuration_enabled: true, token_editable: true,
          job: { status: 'idle', message: '尚未启动更新' }, datasets: {},
        }) };
      }
      if (url === '/api/data/refresh') {
        return { ok: true, json: async () => ({
          source: 'tushare', enabled: true, full_refresh_enabled: true,
          available_modules: ['base', 'etf', 'fund', 'index'], token_configured: true,
          token_configuration_enabled: true, token_editable: true,
          job: { job_id: '1', status: 'running', message: '运行中' }, datasets: {},
        }) };
      }
      return { ok: false, status: 404, json: async () => ({}) };
    });
    vi.stubGlobal('fetch', fetchMock);
    const user = userEvent.setup();
    render(<MemoryRouter><DataHealthRefreshPanel analyticsStatus="complete" onRefreshCompleted={vi.fn()} /></MemoryRouter>);
    await screen.findByText(/尚未启动更新/);
    await act(async () => { await user.click(screen.getByRole('button', { name: /数据管理：查看明细并拉取最新数据/ })); });

    const etfModule = within(screen.getByRole('region', { name: 'ETF模块' }));
    await act(async () => { await user.click(etfModule.getByRole('button', { name: /下载内容/ })); });
    const etfScopes = within(screen.getByRole('group', { name: 'ETF下载内容' }));
    expect(etfScopes.getByRole('checkbox', { name: /产品基础信息/ })).toBeDisabled();
    await act(async () => { await user.click(etfScopes.getByRole('checkbox', { name: /交易行情/ })); });

    const indexModule = within(screen.getByRole('region', { name: '指数模块' }));
    await act(async () => { await user.click(indexModule.getByRole('button', { name: /下载内容/ })); });
    const indexScopes = within(screen.getByRole('group', { name: '指数下载内容' }));
    expect(indexScopes.getByRole('checkbox', { name: /指数目录/ })).toBeChecked();
    expect(indexScopes.getByRole('checkbox', { name: /境内指数/ })).toBeChecked();
    expect(indexScopes.getByRole('checkbox', { name: /行业指数/ })).toBeChecked();
    expect(indexScopes.getByRole('checkbox', { name: /国际指数/ })).toBeChecked();
    expect(indexScopes.getByRole('checkbox', { name: /概念板块/ })).not.toBeChecked();
    expect(screen.getByRole('link', { name: /查看指数数据中心/ })).toHaveAttribute('href', '/index-data');

    await act(async () => { await user.click(indexScopes.getByRole('checkbox', { name: /概念板块/ })); });
    await act(async () => { await user.click(screen.getByRole('button', { name: '开始数据更新' })); });

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith('/api/data/refresh', expect.objectContaining({
      body: JSON.stringify({
        modules: ['base', 'etf', 'fund', 'index'], mode: 'incremental',
        module_scopes: {
          base: ['calendar', 'stock_basic', 'fund_company'],
          etf: ['info', 'nav', 'share'],
          fund: ['info', 'nav'],
          index: ['catalog', 'domestic', 'industry', 'concept', 'global'],
        },
      }),
    })));
    await waitFor(() => expect(screen.getByRole('status')).toHaveTextContent('运行中'));
  });

  it('节点拉取达到 100% 后立即展示合并阶段，避免误认为整个任务完成', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => ({
      source: 'tushare', enabled: true, full_refresh_enabled: true,
      available_modules: ['index'], token_configured: true,
      token_configuration_enabled: true, token_editable: true,
      job: {
        job_id: 'progress-1', status: 'running',
        message: '50/9741，异常 0。\n[INFO] index_daily 分段进度 7000/9741，异常 0。\n[INFO]',
        log_tail: '[INFO] fund_nav 增量进度 7/7，日期 20260826，23118 行。\n[STAGE] 场外公募基金净值已完成本批数据拉取，正在合并本地数据（批次 1/1）。\n[INFO]',
      },
      datasets: {},
    }) }));

    render(<MemoryRouter><DataHealthRefreshPanel analyticsStatus="complete" onRefreshCompleted={vi.fn()} /></MemoryRouter>);

    const status = await screen.findByRole('status');
    expect(status).toHaveTextContent('场外公募基金净值已完成本批数据拉取，正在合并本地数据（批次 1/1）。');
    expect(status).not.toHaveTextContent('100.0%');
  });

  it('任务刚完成时短暂显示完成反馈及具体时间', async () => {
    const finishedAt = new Date().toISOString();
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => ({
      source: 'tushare', enabled: true, full_refresh_enabled: true,
      available_modules: ['fund'], token_configured: true,
      token_configuration_enabled: true, token_editable: true,
      job: {
        job_id: 'completed-1', status: 'succeeded',
        mode: 'incremental', finished_at: finishedAt,
        message: '下载完成：增量数据已更新；分析快照重建完成',
      },
      datasets: {},
    }) }));

    render(<MemoryRouter><DataHealthRefreshPanel analyticsStatus="complete" onRefreshCompleted={vi.fn()} /></MemoryRouter>);

    const status = await screen.findByRole('status');
    expect(status).toHaveTextContent('刚刚完成 · 增量更新');
    expect(status).toHaveTextContent('增量数据已更新；分析快照重建完成');
  });

  it('历史成功任务只显示上次更新时间，不再显示为刚刚下载完成', async () => {
    const finishedAt = new Date(Date.now() - 2 * 24 * 60 * 60 * 1000).toISOString();
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => ({
      source: 'tushare', enabled: true, full_refresh_enabled: true,
      available_modules: ['fund'], token_configured: true,
      token_configuration_enabled: true, token_editable: true,
      job: {
        job_id: 'completed-2', status: 'succeeded', mode: 'full', finished_at: finishedAt,
        message: '下载完成：Tushare 数据更新及分析快照重建完成',
      },
      datasets: {},
    }) }));

    render(<MemoryRouter><DataHealthRefreshPanel analyticsStatus="complete" onRefreshCompleted={vi.fn()} /></MemoryRouter>);

    const status = await screen.findByRole('status');
    expect(status).toHaveTextContent('上次更新于');
    expect(status).toHaveTextContent('2 天前');
    expect(status).toHaveTextContent('全量更新');
    expect(status).not.toHaveTextContent('下载完成');
    expect(status).not.toHaveTextContent('刚刚完成');
  });

  it('后台任务运行时提示用户并锁定全部数据更新控件', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => ({
      source: 'tushare', execution_mode: 'background', refresh_locked: true,
      enabled: true, full_refresh_enabled: true,
      available_modules: ['base', 'etf', 'fund', 'index'],
      available_index_scopes: ['catalog', 'domestic', 'industry', 'concept', 'global', 'futures', 'valuation', 'constituents'],
      token_configured: true, token_configuration_enabled: true, token_editable: true,
      job: {
        status: 'idle', message: '尚未启动更新',
      },
      datasets: {},
    }) }));

    const user = userEvent.setup();
    render(<MemoryRouter><DataHealthRefreshPanel analyticsStatus="complete" onRefreshCompleted={vi.fn()} /></MemoryRouter>);
    await screen.findByText(/检测到后台数据任务正在运行/);
    await act(async () => { await user.click(screen.getByRole('button', { name: /数据管理：查看明细并拉取最新数据/ })); });

    expect(screen.getByText('数据更新正在后台运行，下载入口已锁定。')).toBeInTheDocument();
    expect(screen.getByText(/可以收起数据管理并继续使用系统其他功能/)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: '后台更新中（已锁定）' })).toBeDisabled();
    expect(screen.getByRole('button', { name: '仅重建分析快照' })).toBeDisabled();
    expect(screen.getByRole('radio', { name: '增量更新' })).toBeDisabled();
    expect(screen.getByRole('radio', { name: '全量更新' })).toBeDisabled();
    expect(screen.getByRole('checkbox', { name: /^基础信息/ })).toBeDisabled();
    const indexModule = within(screen.getByRole('region', { name: '指数模块' }));
    await act(async () => { await user.click(indexModule.getByRole('button', { name: /下载内容/ })); });
    expect(within(screen.getByRole('group', { name: '指数下载内容' })).getByRole('checkbox', { name: /境内指数/ })).toBeDisabled();
    expect(screen.getByLabelText('输入 Token')).toBeDisabled();
  });
});

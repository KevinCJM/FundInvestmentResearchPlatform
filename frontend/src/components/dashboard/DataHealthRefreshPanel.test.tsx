import { act, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, describe, expect, it, vi } from 'vitest';
import DataHealthRefreshPanel from './DataHealthRefreshPanel';

describe('DataHealthRefreshPanel index scopes', () => {
  afterEach(() => vi.unstubAllGlobals());

  it('默认选择情景核心范围并提交扩展范围', async () => {
    const fetchMock = vi.fn().mockImplementation(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url === '/api/data/refresh/status') {
        return { ok: true, json: async () => ({
          source: 'tushare', enabled: true, full_refresh_enabled: true,
          available_modules: ['base', 'etf', 'fund', 'index'],
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

    const indexScopes = within(screen.getByRole('group', { name: '指数下载范围' }));
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
        index_scopes: ['catalog', 'domestic', 'industry', 'global', 'concept'],
      }),
    })));
    await waitFor(() => expect(screen.getByRole('status')).toHaveTextContent('运行中'));
  });

  it('运行时只展示日志尾部最后一条完整进度并计算百分比', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => ({
      source: 'tushare', enabled: true, full_refresh_enabled: true,
      available_modules: ['index'], token_configured: true,
      token_configuration_enabled: true, token_editable: true,
      job: {
        job_id: 'progress-1', status: 'running',
        message: '50/9741，异常 0。\n[INFO] index_daily 分段进度 7000/9741，异常 0。\n[INFO]',
        log_tail: '[INFO] index_daily 分段进度 7150/9741，异常 0。\n[INFO] index_daily 分段进度 7200/9741，异常 0。\n[INFO]',
      },
      datasets: {},
    }) }));

    render(<MemoryRouter><DataHealthRefreshPanel analyticsStatus="complete" onRefreshCompleted={vi.fn()} /></MemoryRouter>);

    const status = await screen.findByRole('status');
    expect(status).toHaveTextContent('index_daily：7,200 / 9,741（73.9%），异常 0。');
    expect(status).not.toHaveTextContent('7,150');
    expect(status).not.toHaveTextContent('7,000');
  });
});

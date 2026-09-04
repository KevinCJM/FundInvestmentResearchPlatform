import { render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, describe, expect, it, vi } from 'vitest';
import DataManagement from './DataManagement';

describe('DataManagement', () => {
  afterEach(() => vi.unstubAllGlobals());

  it('聚焦数据下载流程，并把质量检查交给独立页面', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        source: 'tushare',
        enabled: true,
        full_refresh_enabled: true,
        available_modules: ['base', 'etf', 'fund', 'index'],
        token_configured: true,
        token_configuration_enabled: true,
        token_editable: true,
        job: { status: 'idle', message: '尚未启动更新' },
        datasets: {},
      }),
    });
    vi.stubGlobal('fetch', fetchMock);

    render(<MemoryRouter><DataManagement /></MemoryRouter>);
    await screen.findByText(/尚未启动更新/);

    expect(screen.getByRole('heading', { name: '数据下载与更新' })).toBeInTheDocument();
    expect(screen.getByRole('region', { name: '数据下载与更新' })).toBeVisible();
    expect(screen.getByRole('link', { name: '查看数据质量 →' })).toHaveAttribute('href', '/settings/data-quality');
    expect(screen.getByRole('heading', { name: '连接数据源' })).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: '选择更新方式' })).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: '选择下载范围' })).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: '启动更新' })).toBeInTheDocument();
    expect(screen.queryByRole('heading', { name: '数据健康' })).not.toBeInTheDocument();
    expect(screen.queryByText('净值覆盖')).not.toBeInTheDocument();
    expect(screen.queryByRole('table')).not.toBeInTheDocument();
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith('/api/data/refresh/status', { cache: 'no-store' }));
    expect(fetchMock.mock.calls.some(([input]) => String(input).startsWith('/api/instruments/analytics?'))).toBe(false);
  });
});

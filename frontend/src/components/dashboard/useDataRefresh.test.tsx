import { act, render } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { useDataRefresh } from './useDataRefresh';

function Probe() {
  useDataRefresh(vi.fn());
  return null;
}

const runningStatus = (startedAt: string) => ({
  source: 'tushare',
  enabled: true,
  full_refresh_enabled: true,
  available_modules: ['base', 'etf', 'fund'],
  token_configured: true,
  token_source: 'frontend_local',
  token_configuration_enabled: true,
  token_editable: true,
  job: { job_id: 'job-1', status: 'running', started_at: startedAt, message: '正在更新' },
  datasets: {},
});

describe('useDataRefresh polling', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-09-01T00:00:00Z'));
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.useRealTimers();
  });

  it('运行前一分钟每 10 秒检查一次', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => runningStatus('2026-09-01T00:00:00Z') }));
    render(<Probe />);
    await act(async () => { await Promise.resolve(); });
    expect(fetch).toHaveBeenCalledTimes(1);
    expect(fetch).toHaveBeenLastCalledWith('/api/data/refresh/status', { cache: 'no-store' });

    await act(async () => { await vi.advanceTimersByTimeAsync(9_999); });
    expect(fetch).toHaveBeenCalledTimes(1);
    await act(async () => { await vi.advanceTimersByTimeAsync(1); });
    expect(fetch).toHaveBeenCalledTimes(2);
  });

  it('长任务降频为每 60 秒检查一次', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => runningStatus('2026-08-31T23:58:00Z') }));
    render(<Probe />);
    await act(async () => { await Promise.resolve(); });
    expect(fetch).toHaveBeenCalledTimes(1);

    await act(async () => { await vi.advanceTimersByTimeAsync(59_999); });
    expect(fetch).toHaveBeenCalledTimes(1);
    await act(async () => { await vi.advanceTimersByTimeAsync(1); });
    expect(fetch).toHaveBeenCalledTimes(2);
  });

  it('页面恢复焦点时立即检查运行中任务', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => runningStatus('2026-09-01T00:00:00Z') }));
    render(<Probe />);
    await act(async () => { await Promise.resolve(); });
    expect(fetch).toHaveBeenCalledTimes(1);

    await act(async () => { window.dispatchEvent(new Event('focus')); });

    expect(fetch).toHaveBeenCalledTimes(2);
  });
});

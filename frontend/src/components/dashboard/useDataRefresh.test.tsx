import { act, render } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { useDataRefresh } from './useDataRefresh';

function Probe({ onCompleted = vi.fn() }: { onCompleted?: () => void }) {
  useDataRefresh(onCompleted);
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

  it('运行中每 2 秒读取一次轻量进度状态', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => runningStatus('2026-09-01T00:00:00Z') }));
    render(<Probe />);
    await act(async () => { await Promise.resolve(); });
    expect(fetch).toHaveBeenCalledTimes(1);
    expect(fetch).toHaveBeenLastCalledWith('/api/data/refresh/status', { cache: 'no-store' });

    await act(async () => { await vi.advanceTimersByTimeAsync(1_999); });
    expect(fetch).toHaveBeenCalledTimes(1);
    await act(async () => { await vi.advanceTimersByTimeAsync(1); });
    expect(fetch).toHaveBeenCalledTimes(2);
    expect(fetch).toHaveBeenLastCalledWith('/api/data/refresh/status?progress_only=true', { cache: 'no-store' });
  });

  it('轻量轮询发现完成后立即刷新完整状态并触发完成回调', async () => {
    const completed = vi.fn();
    const fetchMock = vi.fn()
      .mockResolvedValueOnce({ ok: true, json: async () => runningStatus('2026-09-01T00:00:00Z') })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          ...runningStatus('2026-09-01T00:00:00Z'),
          refresh_locked: false,
          job: { job_id: 'job-1', status: 'succeeded', message: '下载完成' },
        }),
      })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          ...runningStatus('2026-09-01T00:00:00Z'),
          refresh_locked: false,
          job: { job_id: 'job-1', status: 'succeeded', message: '下载完成' },
          datasets: { fund_nav: { exists: true, rows: 10 } },
        }),
      });
    vi.stubGlobal('fetch', fetchMock);

    render(<Probe onCompleted={completed} />);
    await act(async () => { await Promise.resolve(); });
    await act(async () => { await vi.advanceTimersByTimeAsync(2_000); });

    expect(fetchMock).toHaveBeenNthCalledWith(2, '/api/data/refresh/status?progress_only=true', { cache: 'no-store' });
    expect(fetchMock).toHaveBeenNthCalledWith(3, '/api/data/refresh/status', { cache: 'no-store' });
    expect(completed).toHaveBeenCalledTimes(1);
  });

  it('页面恢复焦点时立即检查运行中任务', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => runningStatus('2026-09-01T00:00:00Z') }));
    render(<Probe />);
    await act(async () => { await Promise.resolve(); });
    expect(fetch).toHaveBeenCalledTimes(1);

    await act(async () => { window.dispatchEvent(new Event('focus')); });

    expect(fetch).toHaveBeenCalledTimes(2);
    expect(fetch).toHaveBeenLastCalledWith('/api/data/refresh/status?progress_only=true', { cache: 'no-store' });
  });
});

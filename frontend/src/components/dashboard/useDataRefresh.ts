import { useCallback, useEffect, useRef, useState } from 'react';
import type { DataRefreshStatus, IndexScope, RefreshMode, RefreshModule } from './types';

const FAST_POLL_INTERVAL_MS = 10_000;
const SLOW_POLL_INTERVAL_MS = 60_000;
const FAST_POLL_WINDOW_MS = 60_000;

const completionKey = (status: DataRefreshStatus) => {
  const job = status.job;
  return job.job_id ?? `${job.started_at ?? 'unknown'}:${job.finished_at ?? job.status}`;
};

export function useDataRefresh(onCompleted: () => void) {
  const [status, setStatus] = useState<DataRefreshStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [rebuilding, setRebuilding] = useState(false);
  const [savingToken, setSavingToken] = useState(false);
  const handledCompletion = useRef<string | null>(null);
  const runningSeenAt = useRef<number | null>(null);

  const fetchStatus = useCallback(async () => {
    try {
      const response = await fetch('/api/data/refresh/status', { cache: 'no-store' });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const payload = (await response.json()) as DataRefreshStatus;
      setStatus(payload);
      setError(null);
      return payload;
    } catch (requestError) {
      setError('无法获取 Tushare 数据更新状态。');
      return null;
    }
  }, []);

  useEffect(() => {
    fetchStatus();
  }, [fetchStatus]);

  useEffect(() => {
    if (status?.job.status !== 'running') {
      runningSeenAt.current = null;
      return;
    }
    if (runningSeenAt.current === null) {
      runningSeenAt.current = Date.now();
    }

    const startedAt = status.job.started_at ? Date.parse(status.job.started_at) : Number.NaN;
    const effectiveStartedAt = Number.isFinite(startedAt) ? startedAt : runningSeenAt.current;
    let cancelled = false;
    let timer: number | undefined;
    const schedule = () => {
      const elapsed = Date.now() - effectiveStartedAt;
      const interval = elapsed < FAST_POLL_WINDOW_MS ? FAST_POLL_INTERVAL_MS : SLOW_POLL_INTERVAL_MS;
      timer = window.setTimeout(async () => {
        if (!document.hidden) {
          await fetchStatus();
        }
        if (!cancelled) {
          schedule();
        }
      }, interval);
    };
    schedule();
    return () => {
      cancelled = true;
      if (timer !== undefined) {
        window.clearTimeout(timer);
      }
    };
  }, [fetchStatus, status?.job.started_at, status?.job.status]);

  useEffect(() => {
    const checkOnResume = () => {
      if (!document.hidden && status?.job.status === 'running') {
        fetchStatus();
      }
    };
    document.addEventListener('visibilitychange', checkOnResume);
    window.addEventListener('focus', checkOnResume);
    return () => {
      document.removeEventListener('visibilitychange', checkOnResume);
      window.removeEventListener('focus', checkOnResume);
    };
  }, [fetchStatus, status?.job.status]);

  useEffect(() => {
    if (!status || status.job.status !== 'succeeded') {
      return;
    }
    const key = completionKey(status);
    if (handledCompletion.current !== key) {
      handledCompletion.current = key;
      onCompleted();
    }
  }, [onCompleted, status]);

  const startRefresh = useCallback(async (
    modules: RefreshModule[],
    mode: RefreshMode,
    indexScopes: IndexScope[] = [],
  ) => {
    if (modules.length === 0) {
      setError('请至少选择一个数据模块。');
      return false;
    }
    const modeText = mode === 'full'
      ? '全量重建可能运行数小时，并会逐只基金循环抓取历史净值'
      : '增量更新会从本地最新日期开始补抓';
    const moduleLabels: Record<RefreshModule, string> = { base: '基础信息', etf: 'ETF', fund: '场外公募基金', index: '指数' };
    const scopeLabels: Record<IndexScope, string> = {
      catalog: '指数目录', domestic: '境内指数', industry: '行业指数', concept: '概念板块',
      global: '国际指数', futures: '商品期货指数', valuation: '指数估值', constituents: '成分与权重',
    };
    const moduleText = modules.map((item) => moduleLabels[item]).join('、');
    const scopeText = modules.includes('index')
      ? `\n指数范围：${indexScopes.map((item) => scopeLabels[item]).join('、')}`
      : '';
    if (mode === 'full' && !window.confirm(`${modeText}。\n模块：${moduleText}${scopeText}\n是否继续？`)) {
      return false;
    }
    try {
      setSubmitting(true);
      setError(null);
      const response = await fetch('/api/data/refresh', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          modules,
          mode,
          ...(modules.includes('index') ? { index_scopes: indexScopes } : {}),
        }),
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || '无法启动数据更新');
      }
      setStatus(payload as DataRefreshStatus);
      return true;
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : '无法启动数据更新。');
      return false;
    } finally {
      setSubmitting(false);
    }
  }, []);

  const saveToken = useCallback(async (token: string) => {
    if (!token.trim()) {
      setError('请输入 Tushare Token。');
      return false;
    }
    try {
      setSavingToken(true);
      setError(null);
      const response = await fetch('/api/data/token', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ token: token.trim() }),
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || '无法保存 Tushare Token');
      }
      await fetchStatus();
      return true;
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : '无法保存 Tushare Token。');
      return false;
    } finally {
      setSavingToken(false);
    }
  }, [fetchStatus]);

  const removeToken = useCallback(async () => {
    try {
      setSavingToken(true);
      setError(null);
      const response = await fetch('/api/data/token', { method: 'DELETE' });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || '无法清除 Tushare Token');
      }
      await fetchStatus();
      return true;
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : '无法清除 Tushare Token。');
      return false;
    } finally {
      setSavingToken(false);
    }
  }, [fetchStatus]);

  const rebuildAnalytics = useCallback(async (candidate = false) => {
    try {
      setRebuilding(true);
      setError(null);
      const response = await fetch('/api/data/analytics/rebuild', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ candidate }),
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || '无法重建分析快照');
      }
      await fetchStatus();
      onCompleted();
      return true;
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : '无法重建分析快照。');
      return false;
    } finally {
      setRebuilding(false);
    }
  }, [fetchStatus, onCompleted]);

  return {
    status,
    error,
    submitting,
    rebuilding,
    savingToken,
    fetchStatus,
    startRefresh,
    rebuildAnalytics,
    saveToken,
    removeToken,
  };
}

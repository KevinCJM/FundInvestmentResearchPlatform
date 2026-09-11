import { useCallback, useEffect, useRef, useState } from 'react';
import type { DataRefreshStatus, RefreshMode, RefreshModule, RefreshModuleScopes } from './types';

const RUNNING_POLL_INTERVAL_MS = 2_000;
const REFRESH_STATUS_URL = '/api/data/refresh/status';

const responseMessage = (payload: unknown, fallback: string): string => {
  const value = payload as { detail?: unknown; message?: unknown } | null;
  if (typeof value?.detail === 'string') return value.detail;
  const detail = value?.detail as { message?: unknown } | undefined;
  if (typeof detail?.message === 'string') return detail.message;
  return typeof value?.message === 'string' ? value.message : fallback;
};

const isRunning = (status: DataRefreshStatus) => (
  status.job.status === 'running' || status.refresh_locked === true
);

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
  const [statusError, setStatusError] = useState<string | null>(null);
  const [checking, setChecking] = useState(false);
  const [lastCheckedAt, setLastCheckedAt] = useState<string | null>(null);
  const requestSequence = useRef(0);
  const mounted = useRef(true);
  const handledCompletion = useRef<string | null>(null);
  const refreshSubmissionInFlight = useRef(false);

  const fetchStatus = useCallback(async (progressOnly = false) => {
    const sequence = ++requestSequence.current;
    setChecking(true);
    try {
      const statusUrl = progressOnly ? `${REFRESH_STATUS_URL}?progress_only=true` : REFRESH_STATUS_URL;
      let response = await fetch(statusUrl, { cache: 'no-store' });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      let payload = (await response.json()) as DataRefreshStatus;
      if (progressOnly && !isRunning(payload)) {
        response = await fetch(REFRESH_STATUS_URL, { cache: 'no-store' });
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
        payload = (await response.json()) as DataRefreshStatus;
        progressOnly = false;
      }
      if (!mounted.current || sequence !== requestSequence.current) return null;
      setStatus((current) => (
        progressOnly && current
          ? { ...current, ...payload, datasets: current.datasets }
          : payload
      ));
      setStatusError(null);
      setLastCheckedAt(new Date().toISOString());
      return payload;
    } catch {
      if (mounted.current && sequence === requestSequence.current) {
        setStatusError('无法获取 Tushare 数据更新状态。请重新检查，暂勿重复启动任务。');
      }
      return null;
    } finally {
      if (mounted.current && sequence === requestSequence.current) setChecking(false);
    }
  }, []);

  useEffect(() => {
    mounted.current = true;
    void fetchStatus();
    return () => { mounted.current = false; requestSequence.current += 1; };
  }, [fetchStatus]);

  useEffect(() => {
    const refreshRunning = status?.job.status === 'running' || status?.refresh_locked === true;
    if (!refreshRunning) {
      return;
    }
    let cancelled = false;
    let timer: number | undefined;
    const schedule = () => {
      timer = window.setTimeout(async () => {
        if (!document.hidden) {
          await fetchStatus(true);
        }
        if (!cancelled) {
          schedule();
        }
      }, RUNNING_POLL_INTERVAL_MS);
    };
    schedule();
    return () => {
      cancelled = true;
      if (timer !== undefined) {
        window.clearTimeout(timer);
      }
    };
  }, [fetchStatus, status?.job.started_at, status?.job.status, status?.refresh_locked]);

  useEffect(() => {
    const checkOnResume = () => {
      if (!document.hidden && (status?.job.status === 'running' || status?.refresh_locked === true)) {
        fetchStatus(true);
      }
    };
    document.addEventListener('visibilitychange', checkOnResume);
    window.addEventListener('focus', checkOnResume);
    return () => {
      document.removeEventListener('visibilitychange', checkOnResume);
      window.removeEventListener('focus', checkOnResume);
    };
  }, [fetchStatus, status?.job.status, status?.refresh_locked]);

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
    moduleScopes: RefreshModuleScopes,
  ) => {
    if (
      refreshSubmissionInFlight.current
      || status?.job.status === 'running'
      || status?.refresh_locked === true
    ) {
      setError('已有数据更新任务正在后台运行，请等待其完成后再启动新的下载。');
      return false;
    }
    if (statusError) {
      setError(null);
      return false;
    }
    if (modules.length === 0) {
      setError('请至少选择一个数据模块。');
      return false;
    }
    const modeText = mode === 'full'
      ? '全量重建可能运行数小时；基金净值与复权因子按基金抓取，持仓和分红按公告日抓取'
      : '增量更新会从本地最新日期开始补抓';
    const moduleLabels: Record<RefreshModule, string> = { base: '基础信息', etf: 'ETF', fund: '场外公募基金', index: '指数', macro: '宏观数据' };
    const scopeLabels: Record<RefreshModule, Record<string, string>> = {
      base: { calendar: '交易日历', stock_basic: '股票目录', fund_company: '基金公司' },
      etf: { info: '产品基础信息', nav: '净值', share: '份额与规模', candle: '交易行情' },
      fund: {
        info: '产品基础信息', nav: '复权净值', manager: '基金经理', scale: '资产规模',
        portfolio: '股票持仓披露', dividend: '分红记录', adjustment: '复权因子', benchmark: '业绩基准库',
      },
      index: {
        catalog: '指数目录', domestic: '境内指数', industry: '行业指数', concept: '概念板块',
        global: '国际指数', futures: '商品期货指数', valuation: '指数估值', constituents: '成分与权重',
      },
      macro: {
        cycle: '增长、通胀与景气', money_credit: '货币与社会融资',
        rates: '利率与回购', release_calendar: '发布日历',
      },
    };
    const moduleText = modules.map((item) => moduleLabels[item]).join('、');
    const scopeText = modules.map((module) => {
      const selected = moduleScopes[module] ?? [];
      return `\n${moduleLabels[module]}下载内容：${selected.map((scope) => scopeLabels[module][scope] ?? scope).join('、')}`;
    }).join('');
    if (mode === 'full' && !window.confirm(`${modeText}。\n模块：${moduleText}${scopeText}\n是否继续？`)) {
      return false;
    }
    refreshSubmissionInFlight.current = true;
    try {
      setSubmitting(true);
      setError(null);
      const response = await fetch('/api/data/refresh', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          modules,
          mode,
          module_scopes: modules.reduce<RefreshModuleScopes>((selected, module) => {
            selected[module] = moduleScopes[module] ?? [];
            return selected;
          }, {}),
        }),
      });
      const payload = await response.json();
      if (!response.ok) {
        if (response.status === 409) {
          await fetchStatus();
        }
        throw new Error(responseMessage(payload, '无法启动数据更新'));
      }
      requestSequence.current += 1;
      setChecking(false);
      setStatusError(null);
      setLastCheckedAt(new Date().toISOString());
      setStatus(payload as DataRefreshStatus);
      return true;
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : '无法启动数据更新。');
      return false;
    } finally {
      refreshSubmissionInFlight.current = false;
      setSubmitting(false);
    }
  }, [fetchStatus, status?.job.status, status?.refresh_locked, statusError]);

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
        throw new Error(responseMessage(payload, '无法保存 Tushare Token'));
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
        throw new Error(responseMessage(payload, '无法清除 Tushare Token'));
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
        throw new Error(responseMessage(payload, '无法重建分析快照'));
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
    error: statusError ?? error,
    statusError,
    checking,
    lastCheckedAt,
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

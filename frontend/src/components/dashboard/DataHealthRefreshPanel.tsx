import { type FormEvent, useEffect, useMemo, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import type {
  DashboardDataQuality,
  DashboardStatus,
  RefreshMode,
  RefreshModule,
  IndexScope,
  SegmentDataQuality,
  SegmentKind,
} from './types';
import { useDataRefresh } from './useDataRefresh';

const integerFormatter = new Intl.NumberFormat('zh-CN', { maximumFractionDigits: 0 });
const percentFormatter = new Intl.NumberFormat('zh-CN', { maximumFractionDigits: 1 });

const moduleOptions: { value: RefreshModule; label: string; description: string }[] = [
  { value: 'base', label: '基础信息', description: '交易日历、股票、基金公司等目录' },
  { value: 'etf', label: 'ETF', description: 'ETF 基础信息、净值与交易行情' },
  { value: 'fund', label: '场外公募基金', description: '基金基础信息与复权净值' },
  { value: 'index', label: '指数', description: '情景模拟所需的指数目录、行情、估值与成分' },
];

const defaultIndexScopes: IndexScope[] = ['catalog', 'domestic', 'industry', 'global'];
const indexScopeOptions: { value: IndexScope; label: string; description: string }[] = [
  { value: 'catalog', label: '指数目录', description: 'index_basic、etf_index 及行业/概念目录' },
  { value: 'domestic', label: '境内指数', description: 'index_daily' },
  { value: 'industry', label: '行业指数', description: 'sw_daily、ci_daily' },
  { value: 'concept', label: '概念板块', description: 'ths_daily、dc_daily、tdx_daily' },
  { value: 'global', label: '国际指数', description: 'index_global' },
  { value: 'futures', label: '商品期货指数', description: 'fut_index_daily' },
  { value: 'valuation', label: '指数估值', description: 'index_dailybasic' },
  { value: 'constituents', label: '成分与权重', description: '成分目录与最新可用权重' },
];

const datasetLabels: Record<string, string> = {
  etf_info: 'ETF 基础信息',
  etf_nav: 'ETF 净值',
  etf_candle: 'ETF 交易行情',
  fund_info: '场外基金基础信息',
  fund_nav: '场外基金净值',
  fund_nav_manifest: '场外基金净值分片',
  fund_company: '基金公司',
  trade_calendar: '交易日历',
  calendar: '交易日历',
  instrument_metrics: '业绩与风险指标快照',
  index_catalog: '指数统一目录（多接口）',
  index_domestic: '境内指数行情（index_daily）',
  index_sw: '申万行业行情（sw_daily）',
  index_ci: '中信行业行情（ci_daily）',
  index_ths: '同花顺概念行情（ths_daily）',
  index_dc: '东财概念行情（dc_daily）',
  index_tdx: '通达信概念行情（tdx_daily）',
  index_global: '国际指数行情（index_global）',
  index_futures: '商品期货指数（fut_index_daily）',
  index_valuation: '指数估值（index_dailybasic）',
  index_members: '指数成分（多接口）',
  index_weights: '最新指数权重（index_weight）',
  index_coverage: '指数覆盖快照',
};

const healthDatasetKeys = ['etf_info', 'etf_nav', 'etf_candle', 'fund_info', 'fund_nav', 'instrument_metrics'] as const;

const statusTone: Record<DashboardStatus, string> = {
  complete: 'bg-emerald-100 text-emerald-800',
  partial: 'bg-amber-100 text-amber-800',
  unavailable: 'bg-rose-100 text-rose-800',
};

const statusLabel: Record<DashboardStatus, string> = {
  complete: '数据完整',
  partial: '部分可用',
  unavailable: '数据未就绪',
};

const completeStatusLines = (value: string) => {
  const lines = value.split(/\r?\n/);
  if (value && !value.endsWith('\n') && !value.endsWith('\r')) {
    lines.pop();
  }
  return lines.map((line) => line.trim()).filter(Boolean);
};

export function refreshStatusSummary(message?: string | null, logTail?: string | null) {
  const logCandidates = completeStatusLines(logTail ?? '');
  const messageCandidates = (message ?? '').split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
  const candidates = [...messageCandidates, ...logCandidates];
  const latest = [...candidates].reverse().find((line) => (
    (line.includes('进度') || line.startsWith('[OK]'))
    && !/^\[(?:INFO|OK|WARN|ERROR)\]$/.test(line)
  ));
  const selected = latest ?? messageCandidates.at(-1) ?? '';
  const cleaned = selected.replace(/^\[(?:INFO|OK)\]\s*/, '');
  const progress = cleaned.match(/^(.*?)\s+(?:分段|增量)?进度\s+(\d+)\/(\d+)(.*)$/);
  if (!progress) {
    return cleaned;
  }
  const [, label, completedText, totalText, detail] = progress;
  const completed = Number(completedText);
  const total = Number(totalText);
  const percentage = total > 0 ? `（${(completed / total * 100).toFixed(1)}%）` : '';
  return `${label}：${integerFormatter.format(completed)} / ${integerFormatter.format(total)}${percentage}${detail}`;
}

interface SegmentHealthCardProps {
  kind: SegmentKind;
  quality?: SegmentDataQuality;
}

function SegmentHealthCard({ kind, quality }: SegmentHealthCardProps) {
  const label = kind === 'etf' ? 'ETF' : '场外公募基金';
  const coverage = quality?.nav_coverage_rate;
  return (
    <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
      <div className="flex items-center justify-between gap-3">
        <h3 className="text-sm font-semibold text-slate-800">{label}</h3>
        <span className={`rounded-full px-2.5 py-1 text-xs font-semibold ${quality?.info_file_exists ? 'bg-emerald-100 text-emerald-700' : 'bg-rose-100 text-rose-700'}`}>
          {quality?.info_file_exists ? '基础信息就绪' : '基础信息缺失'}
        </span>
      </div>
      <dl className="mt-3 grid grid-cols-3 gap-3 text-xs">
        <div><dt className="text-slate-400">信息行数</dt><dd className="mt-1 font-semibold tabular-nums text-slate-700">{quality ? integerFormatter.format(quality.info_rows) : '--'}</dd></div>
        <div><dt className="text-slate-400">指标快照</dt><dd className="mt-1 font-semibold tabular-nums text-slate-700">{quality ? integerFormatter.format(quality.snapshot_rows) : '--'}</dd></div>
        <div><dt className="text-slate-400">净值覆盖</dt><dd className="mt-1 font-semibold tabular-nums text-slate-700">{coverage === null || coverage === undefined ? '--' : `${percentFormatter.format(coverage * 100)}%`}</dd></div>
      </dl>
    </div>
  );
}

interface DataHealthRefreshPanelProps {
  analyticsStatus: DashboardStatus;
  asOf?: string | null;
  dataQuality?: DashboardDataQuality | null;
  onRefreshCompleted: () => void;
}

export default function DataHealthRefreshPanel({
  analyticsStatus,
  asOf,
  dataQuality,
  onRefreshCompleted,
}: DataHealthRefreshPanelProps) {
  const {
    status,
    error,
    submitting,
    rebuilding,
    savingToken,
    startRefresh,
    rebuildAnalytics,
    saveToken,
    removeToken,
  } = useDataRefresh(onRefreshCompleted);
  const [modules, setModules] = useState<RefreshModule[]>(['base', 'etf', 'fund', 'index']);
  const [indexScopes, setIndexScopes] = useState<IndexScope[]>(defaultIndexScopes);
  const [mode, setMode] = useState<RefreshMode>('incremental');
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [tokenInput, setTokenInput] = useState('');
  const drawerButtonRef = useRef<HTMLButtonElement>(null);
  const running = status?.job.status === 'running';
  const recoverableCandidate = Boolean(
    status?.job.fetch_complete
    && status.job.staging_data_dir
    && status.job.analytics_snapshot?.status === 'failed',
  );

  useEffect(() => {
    if (!status?.available_modules?.length) {
      return;
    }
    setModules((current) => current.filter((item) => status.available_modules.includes(item)));
  }, [status?.available_modules]);

  useEffect(() => {
    if (!status?.default_index_scopes?.length) {
      return;
    }
    setIndexScopes((current) => current.length ? current : status.default_index_scopes ?? defaultIndexScopes);
  }, [status?.default_index_scopes]);

  useEffect(() => {
    if (!drawerOpen) {
      setTokenInput('');
      return undefined;
    }
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.preventDefault();
        setDrawerOpen(false);
        drawerButtonRef.current?.focus();
      }
    };
    document.addEventListener('keydown', closeOnEscape);
    return () => document.removeEventListener('keydown', closeOnEscape);
  }, [drawerOpen]);

  const warningCount = dataQuality?.warnings?.length ?? 0;
  const snapshotText = dataQuality?.snapshot?.exists
    ? `指标快照 ${integerFormatter.format(dataQuality.snapshot.rows)} 行 · 截至 ${dataQuality.snapshot.as_of ?? asOf ?? '--'}`
    : '业绩指标快照尚未生成';
  const refreshDisabled = submitting
    || running
    || modules.length === 0
    || !status?.enabled
    || !status?.token_configured
    || (mode === 'full' && !status?.full_refresh_enabled);
  const rebuildDisabled = rebuilding || running || !status?.enabled;
  const tokenControlsDisabled = savingToken
    || running
    || !status?.token_configuration_enabled
    || !status?.token_editable;
  const datasets = useMemo(() => Object.entries(status?.datasets ?? {}), [status?.datasets]);
  const healthDatasets = useMemo(
    () => healthDatasetKeys.flatMap((key) => {
      const dataset = status?.datasets?.[key];
      return dataset ? [[key, dataset] as const] : [];
    }),
    [status?.datasets],
  );
  const refreshMessage = error ?? (
    status?.job.status === 'running'
      ? refreshStatusSummary(status.job.message, status.job.log_tail)
      : status?.job.message
  ) ?? '正在检查数据源状态...';

  const submitToken = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (await saveToken(tokenInput)) {
      setTokenInput('');
    }
  };

  const clearToken = async () => {
    if (!window.confirm('清除后将无法继续从 Tushare 下载数据，已下载的本地数据不会删除。是否继续？')) {
      return;
    }
    if (await removeToken()) {
      setTokenInput('');
    }
  };

  return (
    <section aria-labelledby="data-health-heading" className="min-w-0 overflow-hidden rounded-2xl bg-white shadow-sm ring-1 ring-slate-100">
      <div className="flex flex-col gap-4 px-5 py-4 lg:flex-row lg:items-center lg:justify-between">
        <div>
          <div className="flex flex-wrap items-center gap-2">
            <h2 id="data-health-heading" className="text-base font-semibold text-slate-900">数据健康</h2>
            <span className={`rounded-full px-2.5 py-1 text-xs font-semibold ${statusTone[analyticsStatus]}`}>{statusLabel[analyticsStatus]}</span>
            {warningCount > 0 && <span className="rounded-full bg-amber-50 px-2.5 py-1 text-xs font-semibold text-amber-700">{warningCount} 条提示</span>}
          </div>
          <p className="mt-1 text-sm text-slate-500">{snapshotText}</p>
        </div>
        <div role="status" aria-live="polite" className="min-w-0 rounded-xl bg-slate-50 px-4 py-3 text-sm text-slate-600 lg:max-w-[70%]">
          <span className="font-semibold text-slate-800">Tushare：</span>
          <span className="break-words">{refreshMessage}</span>
        </div>
      </div>

      {dataQuality?.warnings?.length ? (
        <ul className="border-t border-amber-100 bg-amber-50 px-5 py-3 text-sm text-amber-800">
          {dataQuality.warnings.slice(0, 4).map((warning, index) => <li key={`${warning.code ?? 'warning'}-${index}`}>• {warning.message}</li>)}
        </ul>
      ) : null}

      {healthDatasets.length > 0 && (
        <div role="list" aria-label="核心数据集健康状态" className="grid gap-px border-t border-slate-100 bg-slate-100 sm:grid-cols-2 xl:grid-cols-6">
          {healthDatasets.map(([key, dataset]) => {
            const datasetStatus = dataset.status ?? (dataset.error ? 'error' : dataset.exists ? 'ready' : 'missing');
            return (
              <div role="listitem" key={key} className="min-w-0 bg-white px-4 py-3">
                <div className="flex items-center justify-between gap-2">
                  <span className="truncate text-xs font-semibold text-slate-700">{datasetLabels[key] ?? key}</span>
                  <span className={`h-2.5 w-2.5 shrink-0 rounded-full ${datasetStatus === 'ready' ? 'bg-emerald-500' : datasetStatus === 'error' ? 'bg-rose-500' : 'bg-amber-400'}`} aria-label={datasetStatus === 'ready' ? '已就绪' : datasetStatus === 'error' ? '异常' : '缺失'} />
                </div>
                <div className="mt-1 truncate text-xs tabular-nums text-slate-500">
                  {dataset.rows === null || dataset.rows === undefined ? '--' : `${integerFormatter.format(dataset.rows)} 行`} · {dataset.latest_date ?? '无日期'}
                </div>
              </div>
            );
          })}
        </div>
      )}

      <div className="border-t border-slate-100">
        <button
          ref={drawerButtonRef}
          type="button"
          aria-expanded={drawerOpen}
          aria-controls="dashboard-data-management"
          onClick={() => setDrawerOpen((open) => !open)}
          className="w-full px-5 py-4 text-left text-sm font-semibold text-indigo-700 hover:bg-indigo-50 focus:outline-none focus:ring-2 focus:ring-inset focus:ring-indigo-500"
        >
          数据管理：查看明细并拉取最新数据 {drawerOpen ? '收起' : '展开'}
        </button>
        <div
          id="dashboard-data-management"
          role="region"
          aria-label="Tushare 数据管理"
          hidden={!drawerOpen}
          className="space-y-6 border-t border-slate-100 bg-white px-5 py-5"
        >
          <div className="grid gap-4 md:grid-cols-2">
            <SegmentHealthCard kind="etf" quality={dataQuality?.segments?.etf} />
            <SegmentHealthCard kind="fund" quality={dataQuality?.segments?.fund} />
          </div>

          <form onSubmit={submitToken} className="rounded-xl border border-slate-200 bg-slate-50 p-4">
            <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
              <div className="min-w-0 flex-1">
                <div className="flex flex-wrap items-center gap-2">
                  <h3 className="text-sm font-semibold text-slate-900">Tushare Token</h3>
                  <span className={`rounded-full px-2.5 py-1 text-xs font-semibold ${status?.token_configured ? 'bg-emerald-100 text-emerald-700' : 'bg-amber-100 text-amber-800'}`}>
                    {status?.token_configured ? '已配置' : '未配置'}
                  </span>
                </div>
                <p id="tushare-token-help" className="mt-1 text-xs leading-5 text-slate-500">
                  Token 只保存在运行本系统的本机凭据文件中，页面不会回显明文，也不会写入任务日志。保存新 Token 会覆盖旧值。
                </p>
                <label htmlFor="tushare-token-input" className="mt-3 block text-xs font-semibold text-slate-700">输入 Token</label>
                <input
                  id="tushare-token-input"
                  name="tushare-token"
                  type="password"
                  autoComplete="new-password"
                  spellCheck={false}
                  aria-describedby="tushare-token-help"
                  value={tokenInput}
                  onChange={(event) => setTokenInput(event.target.value)}
                  disabled={tokenControlsDisabled}
                  placeholder={status?.token_configured ? '输入新 Token 以更新' : '粘贴 Tushare Token'}
                  className="mt-1 w-full rounded-xl border border-slate-300 bg-white px-3 py-2.5 text-sm text-slate-800 shadow-sm focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-500 disabled:cursor-not-allowed disabled:bg-slate-100"
                />
              </div>
              <div className="flex shrink-0 flex-col gap-2 sm:flex-row">
                {status?.token_configured && (
                  <button
                    type="button"
                    onClick={clearToken}
                    disabled={tokenControlsDisabled}
                    className="rounded-xl border border-rose-200 bg-white px-4 py-2.5 text-sm font-semibold text-rose-700 hover:bg-rose-50 focus:outline-none focus:ring-2 focus:ring-rose-500 focus:ring-offset-2 disabled:cursor-not-allowed disabled:border-slate-200 disabled:text-slate-400"
                  >
                    清除 Token
                  </button>
                )}
                <button
                  type="submit"
                  disabled={tokenControlsDisabled || !tokenInput.trim()}
                  className="rounded-xl bg-slate-900 px-4 py-2.5 text-sm font-semibold text-white hover:bg-slate-700 focus:outline-none focus:ring-2 focus:ring-slate-600 focus:ring-offset-2 disabled:cursor-not-allowed disabled:bg-slate-300"
                >
                  {savingToken ? '正在保存...' : status?.token_configured ? '更新 Token' : '保存 Token'}
                </button>
              </div>
            </div>
            {status && !status.token_configuration_enabled && (
              <p className="mt-3 rounded-lg bg-amber-50 px-3 py-2 text-xs text-amber-800">当前环境未开启前端 Token 配置。</p>
            )}
          </form>

          {datasets.length > 0 && (
            <div className="overflow-x-auto rounded-xl border border-slate-200">
              <table className="min-w-[680px] divide-y divide-slate-200 text-left text-sm">
                <caption className="sr-only">Tushare 本地数据集状态</caption>
                <thead className="bg-slate-50"><tr><th scope="col" className="px-3 py-2 text-xs text-slate-500">数据集 / 接口</th><th scope="col" className="px-3 py-2 text-xs text-slate-500">状态</th><th scope="col" className="px-3 py-2 text-right text-xs text-slate-500">行数</th><th scope="col" className="px-3 py-2 text-xs text-slate-500">首个日期</th><th scope="col" className="px-3 py-2 text-xs text-slate-500">最新日期</th><th scope="col" className="px-3 py-2 text-xs text-slate-500">更新时间</th></tr></thead>
                <tbody className="divide-y divide-slate-100">
                  {datasets.map(([key, dataset]) => (
                    <tr key={key}><td className="px-3 py-2 font-medium text-slate-700">{datasetLabels[key] ?? key}</td><td className="px-3 py-2 text-slate-600">{dataset.error ? `异常：${dataset.error}` : dataset.exists ? '已就绪' : '缺失'}</td><td className="px-3 py-2 text-right tabular-nums text-slate-600">{dataset.rows === null || dataset.rows === undefined ? '--' : integerFormatter.format(dataset.rows)}</td><td className="px-3 py-2 text-slate-600">{dataset.earliest_date ?? '--'}</td><td className="px-3 py-2 text-slate-600">{dataset.latest_date ?? '--'}</td><td className="px-3 py-2 text-slate-500">{dataset.updated_at ?? '--'}</td></tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          <fieldset>
            <legend className="text-sm font-semibold text-slate-800">更新模式</legend>
            <div className="mt-3 flex flex-wrap gap-3">
              {(['incremental', 'full'] as RefreshMode[]).map((option) => (
                <label key={option} className={`flex cursor-pointer items-center gap-2 rounded-xl border px-4 py-3 text-sm ${mode === option ? 'border-indigo-300 bg-indigo-50 text-indigo-800' : 'border-slate-200 text-slate-600'}`}>
                  <input type="radio" name="refresh-mode" value={option} checked={mode === option} disabled={running || (option === 'full' && !status?.full_refresh_enabled)} onChange={() => setMode(option)} className="text-indigo-600 focus:ring-indigo-500" />
                  <span className="font-semibold">{option === 'incremental' ? '增量更新' : '全量更新'}</span>
                </label>
              ))}
            </div>
            <p className="mt-2 text-xs leading-5 text-slate-500">{mode === 'incremental' ? '从本地最新日期继续补抓。' : '从配置起始日重建；基金会逐只循环并遵守请求间隔，可能运行数小时。'}</p>
          </fieldset>

          <fieldset>
            <legend className="text-sm font-semibold text-slate-800">数据模块</legend>
            <div className="mt-3 grid gap-3 md:grid-cols-2 xl:grid-cols-4">
              {moduleOptions.map((option) => {
                const checked = modules.includes(option.value);
                const unavailable = Boolean(status?.available_modules && !status.available_modules.includes(option.value));
                return (
                  <label key={option.value} className={`cursor-pointer rounded-xl border p-4 ${checked ? 'border-indigo-300 bg-indigo-50' : 'border-slate-200 bg-white'} ${unavailable ? 'cursor-not-allowed opacity-50' : ''}`}>
                    <span className="flex items-center gap-2 text-sm font-semibold text-slate-800"><input type="checkbox" checked={checked} disabled={running || unavailable} onChange={() => setModules((current) => checked ? current.filter((item) => item !== option.value) : [...current, option.value])} className="rounded border-slate-300 text-indigo-600 focus:ring-indigo-500" />{option.label}</span>
                    <span className="mt-1 block text-xs leading-5 text-slate-500">{option.description}</span>
                  </label>
                );
              })}
            </div>
          </fieldset>

          {modules.includes('index') && (
            <fieldset className="rounded-xl border border-indigo-100 bg-indigo-50/40 p-4">
              <legend className="px-1 text-sm font-semibold text-slate-800">指数下载范围</legend>
              <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
                <span className="text-xs font-semibold text-indigo-700">情景模拟数据底座</span>
                <Link to="/index-data" className="text-sm font-semibold text-indigo-700 hover:text-indigo-900 focus:outline-none focus:ring-2 focus:ring-indigo-500">查看指数数据中心 →</Link>
              </div>
              <p className="mt-1 text-xs leading-5 text-slate-500">默认勾选“情景模拟核心”。选择任一范围都会自动包含指数目录。</p>
              <div className="mt-3 grid gap-2 sm:grid-cols-2 xl:grid-cols-4">
                {indexScopeOptions.map((option) => {
                  const checked = indexScopes.includes(option.value);
                  const unavailable = Boolean(status?.available_index_scopes && !status.available_index_scopes.includes(option.value));
                  const catalogLocked = option.value === 'catalog';
                  return (
                    <label key={option.value} className={`rounded-lg border bg-white p-3 ${checked ? 'border-indigo-300' : 'border-slate-200'} ${unavailable ? 'opacity-50' : ''}`}>
                      <span className="flex items-center gap-2 text-sm font-semibold text-slate-800">
                        <input
                          type="checkbox"
                          checked={checked}
                          disabled={running || unavailable || catalogLocked}
                          onChange={() => setIndexScopes((current) => checked
                            ? current.filter((item) => item !== option.value)
                            : Array.from(new Set<IndexScope>(['catalog', ...current, option.value])))}
                          className="rounded border-slate-300 text-indigo-600 focus:ring-indigo-500"
                        />
                        {option.label}
                      </span>
                      <span className="mt-1 block text-xs text-slate-500">{option.description}</span>
                    </label>
                  );
                })}
              </div>
            </fieldset>
          )}

          {error && <div role="alert" className="rounded-xl bg-rose-50 p-4 text-sm text-rose-700">{error}</div>}
          {status && (!status.enabled || !status.token_configured) && (
            <div className="rounded-xl bg-amber-50 p-4 text-sm text-amber-800">{!status.enabled ? '当前环境未开启网页数据刷新。' : '请先在上方保存 Tushare Token，再启动数据更新。'}</div>
          )}
          {status?.job.log_tail && (
            <details className="rounded-xl border border-slate-200 bg-slate-950 p-4 text-xs text-slate-200"><summary className="cursor-pointer font-semibold">查看任务日志尾部</summary><pre className="mt-3 max-h-64 overflow-auto whitespace-pre-wrap">{status.job.log_tail}</pre></details>
          )}

          {status?.job.analytics_snapshot?.status === 'failed' && (
            <div role="status" className="rounded-xl border border-amber-200 bg-amber-50 p-4 text-sm text-amber-800">
              Tushare 数据已经更新成功，但分析快照重建失败。{recoverableCandidate ? '可直接重建、验收并接入已保留候选，' : '可单独重建当前快照，'}不会再次请求 Tushare。
              {status.job.analytics_snapshot.message ? <div className="mt-1 text-xs">{status.job.analytics_snapshot.message}</div> : null}
            </div>
          )}

          <div className="flex flex-col gap-3 border-t border-slate-100 pt-4 sm:flex-row sm:items-center sm:justify-between">
            <p className="text-xs leading-5 text-slate-500">任务运行前 1 分钟每 10 秒检查一次，之后每 60 秒；页面隐藏时暂停轮询。单独重建分析快照只读取本地文件，不会调用 Tushare。</p>
            <div className="flex flex-col gap-2 sm:flex-row">
              <button type="button" onClick={() => rebuildAnalytics(recoverableCandidate)} disabled={rebuildDisabled} className="inline-flex min-w-44 items-center justify-center rounded-xl border border-indigo-200 bg-white px-5 py-3 text-sm font-semibold text-indigo-700 hover:bg-indigo-50 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 disabled:cursor-not-allowed disabled:border-slate-200 disabled:text-slate-400">
                {rebuilding ? '正在重建...' : recoverableCandidate ? '重建并接入候选快照' : '仅重建分析快照'}
              </button>
              <button type="button" onClick={() => startRefresh(modules, mode, indexScopes)} disabled={refreshDisabled} className="inline-flex min-w-44 items-center justify-center rounded-xl bg-indigo-600 px-5 py-3 text-sm font-semibold text-white shadow-sm hover:bg-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 disabled:cursor-not-allowed disabled:bg-slate-300">
                {submitting ? '正在启动...' : running ? '数据更新运行中...' : '开始数据更新'}
              </button>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

import { type FormEvent, useEffect, useMemo, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import type {
  DataRefreshStatus,
  RefreshModuleScopes,
  RefreshMode,
  RefreshModule,
  RefreshScope,
} from './types';
import { useDataRefresh } from './useDataRefresh';
import RefreshRecoveryActions from './RefreshRecoveryActions';

const integerFormatter = new Intl.NumberFormat('zh-CN', { maximumFractionDigits: 0 });
const RECENT_REFRESH_COMPLETION_MS = 10 * 60 * 1000;
const noop = () => undefined;

const padDatePart = (value: number) => String(value).padStart(2, '0');

const localDateTime = (timestamp: number) => {
  const date = new Date(timestamp);
  return `${date.getFullYear()}-${padDatePart(date.getMonth() + 1)}-${padDatePart(date.getDate())} ${padDatePart(date.getHours())}:${padDatePart(date.getMinutes())}`;
};

const relativeFinishedTime = (ageMs: number) => {
  const safeAge = Math.max(0, ageMs);
  if (safeAge < 60_000) return '刚刚';
  if (safeAge < 3_600_000) return `${Math.floor(safeAge / 60_000)} 分钟前`;
  if (safeAge < 86_400_000) return `${Math.floor(safeAge / 3_600_000)} 小时前`;
  return `${Math.floor(safeAge / 86_400_000)} 天前`;
};

const cleanCompletionMessage = (message: string) => message
  .replace(/^下载完成\s*[：:·-]?\s*/, '')
  .replace(/^Tushare\s*/, '')
  .trim();

export function completedRefreshSummary(
  job: DataRefreshStatus['job'],
  nowMs = Date.now(),
) {
  const modeLabel = job.mode === 'full' ? '全量更新' : job.mode === 'incremental' ? '增量更新' : '数据更新';
  const finishedAt = Date.parse(job.finished_at ?? '');
  if (!Number.isFinite(finishedAt)) {
    return `最近一次${modeLabel}已结束 · 完成时间未知`;
  }
  const ageMs = Math.max(0, nowMs - finishedAt);
  const completedAt = localDateTime(finishedAt);
  if (ageMs <= RECENT_REFRESH_COMPLETION_MS) {
    const detail = cleanCompletionMessage(job.message);
    return `刚刚完成 · ${modeLabel} · ${completedAt}${detail ? ` · ${detail}` : ''}`;
  }
  return `上次更新于 ${completedAt}（${relativeFinishedTime(ageMs)}）· ${modeLabel}`;
}

interface RefreshScopeOption {
  value: RefreshScope;
  label: string;
  description: string;
}

interface RefreshModuleOption {
  value: RefreshModule;
  label: string;
  description: string;
  scopes: RefreshScopeOption[];
}

const moduleOptions: RefreshModuleOption[] = [
  {
    value: 'base',
    label: '基础信息',
    description: '市场与机构公共目录',
    scopes: [
      { value: 'calendar', label: '交易日历', description: '沪深交易日与开闭市状态' },
      { value: 'stock_basic', label: '股票目录', description: '股票代码、名称与上市状态' },
      { value: 'fund_company', label: '基金公司', description: '基金管理人基础目录' },
    ],
  },
  {
    value: 'etf',
    label: 'ETF',
    description: 'ETF 产品与交易数据',
    scopes: [
      { value: 'info', label: '产品基础信息', description: 'ETF 代码、类型、管理人与上市信息' },
      { value: 'nav', label: '净值', description: '单位净值与复权净值历史' },
      { value: 'share', label: '份额与规模', description: '总份额、单位净值；当前规模按两者相乘计算' },
      { value: 'candle', label: '交易行情', description: '日线价格、成交量与成交额' },
    ],
  },
  {
    value: 'fund',
    label: '场外公募基金',
    description: '产品、净值、人员、规模与定期披露',
    scopes: [
      { value: 'info', label: '产品基础信息', description: '基金代码、类型、管理人与存续状态' },
      { value: 'nav', label: '复权净值', description: '单位净值、累计净值与复权净值历史' },
      { value: 'manager', label: '基金经理', description: '任职区间、公告日与公开履历' },
      { value: 'scale', label: '资产规模', description: '从净值接口提取份额类别与基金总资产规模' },
      { value: 'portfolio', label: '股票持仓披露', description: '季报公开股票持仓；不代表完整资产配置' },
      { value: 'dividend', label: '分红记录', description: '公告日、除息日、派息日与分红金额' },
      { value: 'adjustment', label: '复权因子', description: '用于净值复权结果的独立校验' },
      { value: 'benchmark', label: '业绩基准库', description: '官方 ETF/公募市场基准目录' },
    ],
  },
  {
    value: 'index',
    label: '指数',
    description: '情景模拟所需的指数数据底座',
    scopes: [
      { value: 'catalog', label: '指数目录', description: 'index_basic、etf_index 及行业/概念目录' },
      { value: 'domestic', label: '境内指数', description: 'index_daily' },
      { value: 'industry', label: '行业指数', description: 'sw_daily、ci_daily' },
      { value: 'concept', label: '概念板块', description: 'ths_daily、dc_daily、tdx_daily' },
      { value: 'global', label: '国际指数', description: 'index_global' },
      { value: 'futures', label: '商品期货指数', description: 'fut_index_daily' },
      { value: 'valuation', label: '指数估值', description: 'index_dailybasic' },
      { value: 'constituents', label: '成分与权重', description: '成分目录与最新可用权重' },
    ],
  },
  {
    value: 'macro',
    label: '宏观数据',
    description: '经济周期、货币信用与利率环境',
    scopes: [
      { value: 'cycle', label: '增长、通胀与景气', description: 'GDP、CPI、PPI 与 PMI' },
      { value: 'money_credit', label: '货币与社会融资', description: 'M0/M1/M2 与月度社会融资' },
      { value: 'rates', label: '利率与回购', description: 'Shibor、LPR 与债券回购日行情' },
      { value: 'release_calendar', label: '发布日历', description: '统计局、央行等宏观数据发布日期' },
    ],
  },
];

const defaultModuleScopes: Record<RefreshModule, RefreshScope[]> = {
  base: ['calendar', 'stock_basic', 'fund_company'],
  etf: ['info', 'nav', 'share', 'candle'],
  fund: ['info', 'nav', 'manager', 'scale', 'benchmark'],
  index: ['catalog', 'domestic', 'industry', 'global'],
  macro: ['cycle', 'money_credit', 'rates', 'release_calendar'],
};

const normaliseScopeSelection = (module: RefreshModule, values: RefreshScope[]) => {
  const selected = new Set(values);
  if (module === 'etf' && (selected.has('nav') || selected.has('share') || selected.has('candle'))) selected.add('info');
  if (module === 'fund') {
    if (selected.has('nav') || selected.has('manager') || selected.has('portfolio') || selected.has('dividend') || selected.has('adjustment')) selected.add('info');
    if (selected.has('scale')) {
      selected.add('info');
      selected.add('nav');
    }
  }
  if (module === 'index' && selected.size > 0) selected.add('catalog');
  const option = moduleOptions.find((item) => item.value === module);
  return option?.scopes.map((scope) => scope.value).filter((scope) => selected.has(scope)) ?? [];
};

const scopeIsDependency = (module: RefreshModule, scope: RefreshScope, values: RefreshScope[]) => (
  (module === 'etf' && scope === 'info' && (values.includes('nav') || values.includes('share') || values.includes('candle')))
  || (module === 'fund' && scope === 'info' && values.some((item) => ['nav', 'manager', 'scale', 'portfolio', 'dividend', 'adjustment'].includes(item)))
  || (module === 'fund' && scope === 'nav' && values.includes('scale'))
  || (module === 'index' && scope === 'catalog' && values.some((item) => item !== 'catalog'))
);

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
    (
      line.includes('进度')
      || line.startsWith('[STAGE]')
      || line.startsWith('[DONE]')
      || line.startsWith('[OK]')
    )
    && !/^\[(?:INFO|STAGE|DONE|OK|WARN|ERROR)\]$/.test(line)
  ));
  const selected = latest ?? messageCandidates[messageCandidates.length - 1] ?? '';
  const cleaned = selected.replace(/^\[(?:INFO|STAGE|DONE|OK)\]\s*/, '');
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

interface DataHealthRefreshPanelProps {
  onRefreshCompleted?: () => void;
}

export default function DataHealthRefreshPanel({
  onRefreshCompleted = noop,
}: DataHealthRefreshPanelProps) {
  const {
    status,
    error,
    statusError,
    checking,
    lastCheckedAt,
    fetchStatus,
    submitting,
    rebuilding,
    savingToken,
    startRefresh,
    rebuildAnalytics,
    saveToken,
    removeToken,
  } = useDataRefresh(onRefreshCompleted);
  const [modules, setModules] = useState<RefreshModule[]>(['base', 'etf', 'fund']);
  const [preset, setPreset] = useState<'daily' | 'research' | 'custom'>('daily');
  const [showScopes, setShowScopes] = useState(false);
  const [showToken, setShowToken] = useState(false);
  const taskHeading = useRef<HTMLHeadingElement>(null);
  const [moduleScopes, setModuleScopes] = useState<Record<RefreshModule, RefreshScope[]>>(() => ({
    base: [...defaultModuleScopes.base],
    etf: [...defaultModuleScopes.etf],
    fund: [...defaultModuleScopes.fund],
    index: [...defaultModuleScopes.index],
    macro: [...defaultModuleScopes.macro],
  }));
  const [mode, setMode] = useState<RefreshMode>('incremental');
  const [tokenInput, setTokenInput] = useState('');
  const [statusClock, setStatusClock] = useState(() => Date.now());
  const serverDefaultsApplied = useRef(false);
  const running = status?.job.status === 'running' || status?.refresh_locked === true;
  const refreshControlsLocked = submitting || running || rebuilding || savingToken || Boolean(statusError);
  const recoverableCandidate = Boolean(
    status?.job.fetch_complete
    && status.job.staging_data_dir
    && status.job.analytics_snapshot?.status === 'failed',
  );
  const resumePreviousAvailable = Boolean(
    status?.job.status === 'failed'
    && !status.job.fetch_complete
    && status.job.analytics_snapshot?.status !== 'failed'
    && status.job.resume_available
    && status.job.mode
    && status.job.modules?.length,
  );

  useEffect(() => {
    if (!status?.available_modules?.length) {
      return;
    }
    setModules((current) => current.filter((item) => status.available_modules.includes(item)));
  }, [status?.available_modules]);

  useEffect(() => {
    if (serverDefaultsApplied.current || !status?.default_module_scopes) {
      return;
    }
    setModuleScopes((current) => {
      const next = { ...current };
      moduleOptions.forEach(({ value }) => {
        const serverDefault = status.default_module_scopes?.[value];
        if (serverDefault?.length) next[value] = normaliseScopeSelection(value, serverDefault);
      });
      return next;
    });
    serverDefaultsApplied.current = true;
  }, [status?.default_module_scopes]);

  useEffect(() => {
    if (status?.job.status !== 'succeeded' || !status.job.finished_at) return undefined;
    const timer = window.setInterval(() => setStatusClock(Date.now()), 60_000);
    return () => window.clearInterval(timer);
  }, [status?.job.finished_at, status?.job.status]);

  const refreshDisabled = refreshControlsLocked
    || modules.length === 0
    || !status?.enabled
    || !status?.token_configured
    || (mode === 'full' && !status?.full_refresh_enabled);
  const tokenControlsDisabled = savingToken
    || refreshControlsLocked
    || !status?.token_configuration_enabled
    || !status?.token_editable;
  const selectedScopeCount = useMemo(
    () => modules.reduce((total, module) => total + moduleScopes[module].length, 0),
    [moduleScopes, modules],
  );
  const selectedModuleLabels = useMemo(
    () => moduleOptions.filter((option) => modules.includes(option.value)).map((option) => option.label),
    [modules],
  );
  const refreshMessage = error ?? (
    status?.job.status === 'running'
      ? `后台更新中（下载入口已锁定）· ${refreshStatusSummary(status.job.message, status.job.log_tail)}`
      : status?.refresh_locked
        ? '检测到后台数据任务正在运行，新的下载入口已锁定。'
        : status?.job.analytics_snapshot?.status === 'failed'
          ? '数据已拉取，分析处理尚未完成。请使用下方恢复操作，无需重新下载。'
        : status?.job.status === 'succeeded'
          ? completedRefreshSummary(status.job, statusClock)
          : status?.job.status === 'failed'
            ? `下载失败 · ${status.job.message}`
            : status?.job.message
  ) ?? '正在检查数据源状态...';
  const needsAttention = Boolean(error || status?.job.analytics_snapshot?.status === 'failed' || status?.job.status === 'failed');
  const jobTone = needsAttention ? 'bg-amber-100 text-amber-900' : status?.job.status === 'running' || status?.refresh_locked
    ? 'bg-accent-100 text-accent-800'
    : status?.job.status === 'succeeded'
      ? 'bg-emerald-100 text-emerald-800'
      : status?.job.status === 'failed' || error
        ? 'bg-rose-100 text-rose-800'
        : 'bg-slate-100 text-slate-700';
  const jobLabel = statusError ? '状态待确认' : needsAttention ? '需要处理' : status?.job.status === 'running' || status?.refresh_locked
    ? '更新中'
    : status?.job.status === 'succeeded'
      ? '最近成功'
      : status?.job.status === 'failed' || error
        ? '需要处理'
        : '待执行';
  const refreshDisabledReason = statusError ? '状态失联，请先点击“重新检查状态”。' : !status
    ? '正在读取更新配置…'
    : refreshControlsLocked
      ? '当前任务结束后可再次启动。'
      : !status.enabled
        ? '当前环境未开启网页数据更新。'
        : !status.token_configured
          ? '请先保存 Tushare Token。'
          : mode === 'full' && !status.full_refresh_enabled
            ? '当前环境未开放全量更新。'
            : modules.length === 0
              ? '请至少选择一个数据模块。'
              : '配置已就绪，任务会在后台运行。';

  const submitToken = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (await saveToken(tokenInput)) {
      setTokenInput('');
      setShowToken(false);
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

  const resumePreviousRefresh = async () => {
    const previousModules = status?.job.modules ?? [];
    const previousMode = status?.job.mode;
    if (!previousMode || previousModules.length === 0) return;
    const previousScopes = previousModules.reduce<RefreshModuleScopes>((result, module) => {
      const saved = status?.job.module_scopes?.[module];
      result[module] = saved?.length ? [...saved] : [...defaultModuleScopes[module]];
      return result;
    }, {});
    setModules(previousModules);
    setPreset('custom');
    setMode(previousMode);
    setModuleScopes((current) => ({ ...current, ...previousScopes }));
    await startRefresh(previousModules, previousMode, previousScopes);
  };

  const applyPreset = (choice: 'daily' | 'research') => {
    const requested: RefreshModule[] = choice === 'daily' ? ['base', 'etf', 'fund'] : ['base', 'index', 'macro'];
    const supported = requested.filter(module => !status?.available_modules || status.available_modules.includes(module));
    setModules(supported);
    setModuleScopes(current => {
      const next = { ...current };
      moduleOptions.forEach(({ value }) => {
        const defaults = status?.default_module_scopes?.[value] ?? defaultModuleScopes[value];
        const available = status?.available_module_scopes?.[value];
        next[value] = supported.includes(value)
          ? normaliseScopeSelection(value, defaults.filter(scope => !available || available.includes(scope)))
          : [];
      });
      return next;
    });
    setPreset(choice);
  };

  const toggleModule = (module: RefreshModule, checked: boolean) => {
    setPreset('custom');
    if (checked) {
      setModules((current) => current.includes(module) ? current : [...current, module]);
      setModuleScopes((current) => ({
        ...current,
        [module]: current[module].length ? current[module] : [...defaultModuleScopes[module]],
      }));
      return;
    }
    setModules((current) => current.filter((item) => item !== module));
    setModuleScopes((current) => ({ ...current, [module]: [] }));
  };

  const toggleScope = (module: RefreshModule, scope: RefreshScope, checked: boolean) => {
    setPreset('custom');
    const previous = modules.includes(module) ? moduleScopes[module] : [];
    const next = normaliseScopeSelection(
      module,
      checked
        ? [...previous, scope]
        : previous.filter((item) => item !== scope),
    );
    setModuleScopes((current) => ({ ...current, [module]: next }));
    setModules((current) => {
      if (!next.length) return current.filter((item) => item !== module);
      return current.includes(module) ? current : [...current, module];
    });
  };

  return (
    <section
      id="dashboard-data-management"
      role="region"
      aria-label="数据下载与更新"
      className="min-w-0 space-y-5"
    >
      <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm sm:p-6">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-2">
              <h2 ref={taskHeading} tabIndex={-1} id="data-refresh-heading" className="text-lg font-semibold text-slate-950">当前任务</h2>
              <span className={`rounded-full px-2.5 py-1 text-xs font-semibold ${jobTone}`}>{jobLabel}</span>
            </div>
            <p className="mt-1 text-sm text-slate-600">任务在后台运行，离开页面不会中断下载。</p>
          </div>
          <dl className="grid grid-cols-2 gap-x-8 gap-y-2 text-xs sm:grid-cols-3">
            <div><dt className="text-slate-600">数据源</dt><dd className="mt-1 font-semibold text-slate-700">Tushare</dd></div>
            <div><dt className="text-slate-600">凭证</dt><dd className="mt-1 font-semibold text-slate-700">{status?.token_configured ? '已配置' : '未配置'}</dd></div>
            <div className="col-span-2 sm:col-span-1"><dt className="text-slate-600">执行方式</dt><dd className="mt-1 font-semibold text-slate-700">后台任务</dd></div>
          </dl>
        </div>
        <div role="status" aria-live="polite" className={`mt-4 rounded-xl px-4 py-3 text-sm ${error ? 'bg-rose-50 text-rose-700' : running ? 'bg-accent-50 text-accent-800' : 'bg-slate-50 text-slate-700'}`}>
          <span className="font-semibold">{error ? '操作失败：' : running ? '正在执行：' : '任务状态：'}</span>
          <span className="break-words">{refreshMessage}</span>
        </div>
        <div className="mt-3 flex flex-wrap items-center gap-3 text-xs">
          <button type="button" onClick={() => void fetchStatus()} disabled={checking || submitting || rebuilding} className="min-h-10 rounded-lg border border-slate-300 px-3 font-semibold text-slate-700 disabled:opacity-50">{checking ? '正在检查…' : '重新检查状态'}</button>
          <span className="text-slate-600">{lastCheckedAt ? `状态检查于 ${localDateTime(Date.parse(lastCheckedAt))}` : '尚未取得任务状态'}</span>
        </div>
        {status?.job.status === 'succeeded' && !needsAttention ? <div className="mt-4 space-y-2 border-t border-slate-100 pt-4 text-sm">
          <p className="font-semibold text-slate-800">下载任务已结束，接下来检查数据质量。</p>
          <p className="text-xs leading-5 text-slate-600">现有研究数据与新标准表候选是两条独立链路。下载成功不代表新标准表已经发布。</p>
          <div className="flex flex-wrap gap-4 font-semibold text-accent-700"><Link to="/settings/data-quality">检查数据质量 →</Link><Link to="/settings/source-center?view=results">查看标准表映射结果 →</Link></div>
        </div> : null}
      </div>

      <RefreshRecoveryActions status={status} locked={refreshControlsLocked} rebuilding={rebuilding} recoverableCandidate={recoverableCandidate} canResume={resumePreviousAvailable} onResume={() => void resumePreviousRefresh()} onRebuild={() => void rebuildAnalytics(recoverableCandidate)} />

      <form onSubmit={submitToken} className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm sm:p-6">
        <h2 className="text-base font-semibold text-slate-950">连接数据源</h2>
        <p className="mt-1 text-sm text-slate-600">Tushare · {status?.token_configured ? '凭据已保存，日常更新无需重复填写。' : '首次使用，请填写访问凭据。'} 保存凭据不代表接口权限已验证。</p>
        <details open={!status?.token_configured || showToken} onToggle={event => { if (status?.token_configured) setShowToken(event.currentTarget.open); }} className="mt-3">
          <summary className="cursor-pointer py-2 text-sm font-semibold text-accent-700">{status?.token_configured ? '管理连接凭据' : '填写连接凭据'}</summary>
        <div className="flex flex-col gap-5 lg:flex-row lg:items-end lg:justify-between">
          <div className="min-w-0 flex-1">
            <div className="flex flex-wrap items-center gap-3">
              <span className="inline-flex h-7 w-7 items-center justify-center rounded-full bg-accent-100 text-xs font-bold text-accent-700">1</span>
              <div>
                <h3 className="text-base font-semibold text-slate-950">Tushare 访问凭据</h3>
                <p className="mt-0.5 text-xs text-slate-600">用于本机直接下载 Tushare 数据。</p>
              </div>
              <span className={`rounded-full px-2.5 py-1 text-xs font-semibold ${status?.token_configured ? 'bg-emerald-100 text-emerald-700' : 'bg-amber-100 text-amber-800'}`}>
                {status?.token_configured ? 'Token 已配置' : 'Token 未配置'}
              </span>
            </div>
            <label htmlFor="tushare-token-input" className="mt-4 block text-xs font-semibold text-slate-700">输入 Token</label>
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
              placeholder={status?.token_configured ? '输入新 Token 以替换现有凭证' : '粘贴 Tushare Token'}
              className="mt-1 w-full rounded-xl border border-slate-300 bg-white px-3 py-2.5 text-sm text-slate-800 shadow-sm focus:border-accent-500 focus:outline-none focus:ring-2 focus:ring-accent-500 disabled:cursor-not-allowed disabled:bg-slate-100"
            />
            <p id="tushare-token-help" className="mt-2 text-xs leading-5 text-slate-600">
              Token 只保存在运行本系统的本机凭据文件中；页面不回显明文，任务日志也不会记录。
            </p>
          </div>
          <div className="flex shrink-0 flex-col gap-2 sm:flex-row">
            {status?.token_configured && (
              <button
                type="button"
                onClick={clearToken}
                disabled={tokenControlsDisabled}
                className="rounded-xl border border-rose-200 bg-white px-4 py-2.5 text-sm font-semibold text-rose-700 hover:bg-rose-50 focus:outline-none focus:ring-2 focus:ring-accent-500 focus:ring-offset-2 disabled:cursor-not-allowed disabled:border-slate-200 disabled:text-slate-600"
              >
                清除 Token
              </button>
            )}
            <button
              type="submit"
              disabled={tokenControlsDisabled || !tokenInput.trim()}
              className="rounded-xl bg-slate-900 px-4 py-2.5 text-sm font-semibold text-white hover:bg-slate-700 focus:outline-none focus:ring-2 focus:ring-accent-500 focus:ring-offset-2 disabled:cursor-not-allowed disabled:bg-slate-300"
            >
              {savingToken ? '正在保存...' : status?.token_configured ? '更新 Token' : '保存 Token'}
            </button>
          </div>
        </div>
        {status && !status.token_configuration_enabled && (
          <p className="mt-3 rounded-lg bg-amber-50 px-3 py-2 text-xs text-amber-800">当前环境未开启前端 Token 配置。</p>
        )}
        </details>
      </form>

      <fieldset className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm sm:p-6">
        <legend className="sr-only">更新方式</legend>
        <div className="flex items-start gap-3">
          <span className="inline-flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-accent-100 text-xs font-bold text-accent-700">2</span>
          <div>
            <h2 className="text-base font-semibold text-slate-950">选择更新方式</h2>
            <p className="mt-0.5 text-xs leading-5 text-slate-600">日常维护使用增量更新；仅在首次建库或需要彻底重建时使用全量更新。</p>
          </div>
        </div>
        <div className="mt-4 grid gap-3 md:grid-cols-2">
          {(['incremental', 'full'] as RefreshMode[]).map((option) => {
            const selected = mode === option;
            const fullDisabled = option === 'full' && !status?.full_refresh_enabled;
            return (
              <label key={option} className={`flex items-start gap-3 rounded-xl border p-4 ${selected ? 'border-accent-400 bg-accent-50 ring-1 ring-accent-100' : 'border-slate-200 bg-white'} ${refreshControlsLocked || fullDisabled ? 'cursor-not-allowed opacity-60' : 'cursor-pointer hover:border-accent-200'}`}>
                <input type="radio" name="refresh-mode" value={option} checked={selected} disabled={refreshControlsLocked || fullDisabled} onChange={() => setMode(option)} className="mt-1 text-accent-600 focus:ring-accent-500" />
                <span>
                  <span className="flex flex-wrap items-center gap-2 text-sm font-semibold text-slate-900">
                    {option === 'incremental' ? '增量更新' : '全量更新'}
                    {option === 'incremental' && <span className="rounded-full bg-emerald-100 px-2 py-0.5 text-xs text-emerald-700">日常推荐</span>}
                    {fullDisabled && <span className="rounded-full bg-slate-100 px-2 py-0.5 text-xs text-slate-600">当前不可用</span>}
                  </span>
                  <span className="mt-1 block text-xs leading-5 text-slate-600">
                    {option === 'incremental' ? '从本地最新日期继续补抓，耗时更短。' : '从配置起始日重新构建，可能运行数小时。'}
                  </span>
                </span>
              </label>
            );
          })}
        </div>
      </fieldset>

      <fieldset className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm sm:p-6">
        <legend className="sr-only">下载范围</legend>
        <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
          <div className="flex items-start gap-3">
            <span className="inline-flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-accent-100 text-xs font-bold text-accent-700">3</span>
            <div>
              <h2 className="text-base font-semibold text-slate-950">选择下载范围</h2>
              <p className="mt-0.5 text-xs leading-5 text-slate-600">先选常用方案，需要时再展开具体内容；必要依赖会自动加入。</p>
            </div>
          </div>
          <p className="text-xs font-semibold tabular-nums text-accent-700">已选 {modules.length} 个模块 · {selectedScopeCount} 项内容</p>
        </div>
        <div className="mt-4 flex flex-wrap gap-2" aria-label="同步方案">
          <button type="button" aria-pressed={preset === 'daily'} disabled={refreshControlsLocked} onClick={() => applyPreset('daily')} className={`min-h-11 rounded-xl border px-4 text-sm font-semibold disabled:opacity-50 ${preset === 'daily' ? 'border-accent-300 bg-accent-50 text-accent-900' : 'border-slate-200'}`}>日常基金更新</button>
          <button type="button" aria-pressed={preset === 'research'} disabled={refreshControlsLocked} onClick={() => applyPreset('research')} className={`min-h-11 rounded-xl border px-4 text-sm font-semibold disabled:opacity-50 ${preset === 'research' ? 'border-accent-300 bg-accent-50 text-accent-900' : 'border-slate-200'}`}>指数与宏观研究</button>
          <button type="button" aria-pressed={preset === 'custom'} disabled={refreshControlsLocked} onClick={() => { setPreset('custom'); setShowScopes(true); }} className="min-h-11 rounded-xl border border-slate-200 px-4 text-sm font-semibold disabled:opacity-50">自定义范围</button>
        </div>
        <p className="mt-2 text-xs leading-5 text-slate-600">方案只调整下载内容，不会自动启动，也不会切换为全量更新。</p>
        <details open={showScopes} onToggle={event => setShowScopes(event.currentTarget.open)} className="mt-4">
          <summary className="cursor-pointer py-2 text-sm font-semibold text-accent-700">调整具体下载内容 · {selectedScopeCount} 项已选</summary>
        <div className="mt-4 grid gap-4 md:grid-cols-2">
          {moduleOptions.map((option) => {
            const checked = modules.includes(option.value);
            const unavailable = Boolean(status?.available_modules && !status.available_modules.includes(option.value));
            const selectedScopes = checked ? moduleScopes[option.value] : [];
            const availableScopes = status?.available_module_scopes?.[option.value]
              ?? (option.value === 'index' ? status?.available_index_scopes : undefined);
            return (
              <section aria-label={`${option.label}模块`} key={option.value} className={`min-w-0 rounded-xl border ${checked ? 'border-accent-300 bg-accent-50/50' : 'border-slate-200 bg-slate-50/50'} ${unavailable ? 'opacity-50' : ''} ${option.value === 'index' ? 'md:col-span-2' : ''}`}>
                <label className={`flex items-start justify-between gap-3 p-4 ${refreshControlsLocked || unavailable ? 'cursor-not-allowed' : 'cursor-pointer'}`}>
                  <span className="flex min-w-0 items-start gap-3">
                    <input
                      type="checkbox"
                      checked={checked}
                      disabled={refreshControlsLocked || unavailable}
                      onChange={(event) => toggleModule(option.value, event.target.checked)}
                      className="mt-0.5 rounded-lg border-slate-300 text-accent-600 focus:ring-accent-500"
                    />
                    <span className="min-w-0">
                      <span className="block text-sm font-semibold text-slate-900">{option.label}</span>
                      <span className="mt-1 block text-xs leading-5 text-slate-600">{option.description}</span>
                    </span>
                  </span>
                  <span className="shrink-0 rounded-full bg-white px-2.5 py-1 text-xs font-semibold tabular-nums text-slate-600 ring-1 ring-slate-200">{selectedScopes.length}/{option.scopes.length}</span>
                </label>
                <fieldset aria-label={`${option.label}下载内容`} className={`grid gap-2 border-t border-slate-200/80 p-3 ${option.value === 'index' ? 'sm:grid-cols-2 xl:grid-cols-4' : 'sm:grid-cols-2'}`}>
                  <legend className="sr-only">{option.label}下载内容</legend>
                  {option.scopes.map((scope) => {
                    const scopeChecked = selectedScopes.includes(scope.value);
                    const dependency = scopeIsDependency(option.value, scope.value, selectedScopes);
                    const scopeUnavailable = Boolean(availableScopes && !availableScopes.includes(scope.value));
                    const scopeDisabled = refreshControlsLocked || unavailable || scopeUnavailable || dependency;
                    return (
                      <label key={scope.value} className={`block rounded-lg border p-2.5 sm:p-3 ${scopeChecked ? 'border-accent-200 bg-white' : 'border-slate-200 bg-white/70'} ${scopeDisabled ? 'cursor-not-allowed opacity-60' : 'cursor-pointer hover:border-accent-200'}`}>
                        <span className="flex items-center gap-2 text-sm font-semibold text-slate-800">
                          <input
                            type="checkbox"
                            checked={scopeChecked}
                            disabled={scopeDisabled}
                            onChange={(event) => toggleScope(option.value, scope.value, event.target.checked)}
                            className="rounded-lg border-slate-300 text-accent-600 focus:ring-accent-500"
                          />
                          {scope.label}
                          {dependency && <span className="rounded-full bg-slate-100 px-2 py-0.5 text-xs font-medium text-slate-600">自动依赖</span>}
                        </span>
                        <span className="mt-1 hidden text-xs leading-5 text-slate-600 sm:block">{scope.description}</span>
                      </label>
                    );
                  })}
                </fieldset>
              </section>
            );
          })}
        </div>
        </details>
        <p className="mt-4 border-t border-slate-100 pt-4 text-xs leading-5 text-slate-600">增量更新历史数据时会自动同步交易日历，避免日期断层。</p>
      </fieldset>

      {status && (!status.enabled || !status.token_configured) && (
        <div className="rounded-xl bg-amber-50 p-4 text-sm text-amber-800">{!status.enabled ? '当前环境未开启网页数据更新。' : '请先保存 Tushare Token，再启动数据更新。'}</div>
      )}
      {(submitting || running || rebuilding) && (
        <div
          id="refresh-background-notice"
          aria-live="polite"
          className="rounded-xl border border-accent-200 bg-accent-50 p-4 text-sm text-accent-900"
        >
          <div className="font-semibold">
            {submitting ? '正在启动后台数据更新…' : rebuilding ? '正在整理分析数据，暂时不能启动新的下载。' : '数据更新正在后台运行，下载入口已锁定。'}
          </div>
          <p className="mt-1 text-xs leading-5 text-accent-700">
            您可以继续使用系统其他功能；当前任务结束前，不能启动新的增量更新、全量更新或分析快照重建。
          </p>
        </div>
      )}

      {status?.job.log_tail && (
        <details className="rounded-xl border border-slate-200 bg-slate-950 p-4 text-xs text-slate-200"><summary className="cursor-pointer font-semibold">查看任务日志</summary><pre className="mt-3 max-h-64 overflow-auto whitespace-pre-wrap">{status.job.log_tail}</pre></details>
      )}

      <section aria-label="本次同步清单" className="rounded-xl border border-slate-200 bg-white p-5">
        <h2 className="text-base font-semibold text-slate-900">确认本次同步内容</h2>
        <dl className="mt-3 space-y-2 text-sm">{moduleOptions.filter(option => modules.includes(option.value)).map(option => <div key={option.value} className="grid gap-1 sm:grid-cols-[130px_minmax(0,1fr)]">
          <dt className="font-semibold text-slate-700">{option.label}</dt><dd className="text-slate-600">{option.scopes.filter(scope => moduleScopes[option.value].includes(scope.value)).map(scope => scope.label).join('、') || '未选择内容'}</dd>
        </div>)}</dl>
        {!modules.length ? <p className="mt-2 text-sm text-amber-800">尚未选择数据，请选择方案或勾选具体内容。</p> : null}
        <p className="mt-3 text-xs leading-5 text-slate-600">{mode === 'full' ? '全量会按系统配置的历史范围重建，启动前还需确认。' : '增量用于补充最近数据；首次使用需要完整历史时，请选择全量。'} 标准表映射结果单独生成候选，不自动替换正式研究数据。</p>
      </section>
      <div className="rounded-xl bg-slate-950 p-5 text-white shadow-sm sm:p-6">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
          <div className="min-w-0">
            <div className="flex items-center gap-3">
              <span className="inline-flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-white/10 text-xs font-bold text-white">4</span>
              <div className="min-w-0 flex-1">
                <h2 className="text-base font-semibold">启动更新</h2>
                <p className="mt-1 text-sm text-slate-200">
                  {mode === 'incremental' ? '增量更新' : '全量更新'} · {selectedModuleLabels.length ? selectedModuleLabels.join('、') : '未选择模块'} · {selectedScopeCount} 项内容
                </p>
              </div>
            </div>
            <p className={`mt-2 text-xs ${refreshDisabled ? 'text-amber-300' : 'text-slate-200'}`}>{refreshDisabledReason}</p>
          </div>
          <button type="button" aria-describedby={refreshControlsLocked ? 'refresh-background-notice' : undefined} onClick={async () => { await startRefresh(modules, mode, moduleScopes); taskHeading.current?.focus(); taskHeading.current?.scrollIntoView?.({ behavior: 'smooth', block: 'start' }); }} disabled={refreshDisabled} className="inline-flex min-h-12 w-full shrink-0 items-center justify-center rounded-xl bg-accent-600 px-6 py-3 text-sm font-semibold text-white shadow-sm hover:bg-accent-700 focus:outline-none focus:ring-2 focus:ring-accent-500 focus:ring-offset-2 focus:ring-offset-slate-950 disabled:cursor-not-allowed disabled:bg-slate-700 disabled:text-slate-400 lg:w-auto lg:min-w-52">
            {submitting ? '正在启动后台任务...' : running ? '后台更新中（已锁定）' : '开始数据更新'}
          </button>
        </div>
      </div>
    </section>
  );
}

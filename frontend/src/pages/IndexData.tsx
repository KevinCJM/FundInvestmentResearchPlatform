import { useCallback, useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import DataWorkspaceNav from '../components/data-sources/DataWorkspaceNav';
import { useDataRefresh } from '../components/dashboard/useDataRefresh';
import type { DataQualityWarning, DataRefreshStatus, InstrumentAnalyticsResponse } from '../components/dashboard/types';
import {
  assertFixedNjitExecution,
  type FixedNjitExecutionAudit,
} from '../utils/fixedNjitExecution';

interface IndexSummary {
  status: 'complete' | 'partial' | 'unavailable';
  catalog_count: number;
  covered_count: number;
  coverage_rate: number | null;
  latest_date?: string | null;
  missing_count: number;
  stale_count: number;
  execution: FixedNjitExecutionAudit;
}

type QualityTone = 'good' | 'warning' | 'danger' | 'planned';
type QualitySeverity = 'critical' | 'high' | 'medium';
type CheckStatus = 'passed' | 'warning' | 'failed' | 'unavailable';
type DataDomain = 'base' | 'etf' | 'fund' | 'index' | 'snapshot';

interface DatasetView {
  key: string;
  label: string;
  domain: DataDomain;
  file: string;
  tone: Exclude<QualityTone, 'planned'>;
  statusLabel: string;
  rows?: number | null;
  earliestDate?: string | null;
  latestDate?: string | null;
  updatedAt?: string | null;
}

interface QualitySample {
  kind: string;
  ts_code: string;
  name?: string;
  latest_date?: string;
  observed?: string;
}

interface DeepQualityIssue {
  id: string;
  code: string;
  severity: QualitySeverity;
  dimension: string;
  scope: string;
  title: string;
  description: string;
  evidence: string;
  impact: string;
  affected_count: number;
  affected_rate?: number | null;
  record_count: number;
  samples: QualitySample[];
  action: 'refresh' | 'rebuild_analytics' | 'inspect_source' | 'inspect' | string;
}

interface QualityCheck {
  key: string;
  label: string;
  dimension: string;
  status: CheckStatus;
  summary: string;
  detail: string;
  threshold?: string | null;
}

interface DeepQualityReport {
  schema_version: number;
  status: 'healthy' | 'attention' | 'blocked' | 'unavailable';
  generated_at?: string | null;
  activated_at?: string | null;
  as_of?: string | null;
  summary: {
    checks_total: number;
    checks_passed: number;
    checks_warning: number;
    checks_failed: number;
    checks_unavailable: number;
    total_products: number;
    affected_products: number;
    affected_rate?: number | null;
    issue_count: number;
    critical_issue_count: number;
    high_issue_count: number;
    medium_issue_count: number;
    nav_anomaly_products: number;
    nav_anomaly_events: number;
    stale_active_products: number;
  };
  checks: QualityCheck[];
  issues: DeepQualityIssue[];
  validation: { status: 'passed' | 'unavailable'; manifest?: string | null };
  execution: FixedNjitExecutionAudit;
}

interface QualityIssue extends DeepQualityIssue {
  actionTo?: string;
  actionLabel?: string;
}

const integerFormatter = new Intl.NumberFormat('zh-CN', { maximumFractionDigits: 0 });
const percentFormatter = new Intl.NumberFormat('zh-CN', { maximumFractionDigits: 1 });
const dateTimeFormatter = new Intl.DateTimeFormat('zh-CN', {
  month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', hour12: false,
});

const datasetLabels: Record<string, string> = {
  etf_info: 'ETF 基础信息', etf_nav: 'ETF 净值', etf_share: 'ETF 份额与规模', etf_candle: 'ETF 交易行情',
  fund_info: '场外基金基础信息', fund_nav: '场外基金净值', fund_manager: '基金经理', fund_scale: '基金规模',
  fund_portfolio: '基金持仓', fund_dividend: '基金分红', fund_adjustment: '基金复权因子', fund_benchmark: '基金业绩基准',
  fund_company: '基金公司', calendar: '交易日历', instrument_metrics: '业绩与风险指标快照',
  index_catalog: '指数统一目录', index_domestic: '境内指数行情', index_sw: '申万行业行情',
  index_ci: '中信行业行情', index_ths: '同花顺概念行情', index_dc: '东财概念行情',
  index_tdx: '通达信概念行情', index_global: '国际指数行情', index_futures: '商品期货指数',
  index_valuation: '指数估值', index_members: '指数成分', index_weights: '指数权重', index_coverage: '指数覆盖快照',
};

const domainLabels: Record<DataDomain, string> = {
  base: '基础与机构数据', etf: 'ETF 产品数据', fund: '场外公募基金', index: '指数与情景数据', snapshot: '研究指标快照',
};

const dimensionLabels: Record<string, string> = {
  validity: '合法性', integrity: '唯一性与完整性', consistency: '一致性', continuity: '连续性',
  completeness: '覆盖完整性', timeliness: '及时性', availability: '可用性',
};

const toneStyles: Record<QualityTone, { badge: string; dot: string; label: string }> = {
  good: { badge: 'bg-emerald-50 text-emerald-700', dot: 'bg-emerald-500', label: '通过' },
  warning: { badge: 'bg-amber-50 text-amber-800', dot: 'bg-amber-400', label: '需关注' },
  danger: { badge: 'bg-rose-50 text-rose-700', dot: 'bg-rose-500', label: '阻断' },
  planned: { badge: 'bg-slate-100 text-slate-600', dot: 'bg-slate-300', label: '未检查' },
};

const severityStyles: Record<QualitySeverity, { label: string; badge: string; border: string }> = {
  critical: { label: '阻断', badge: 'bg-rose-100 text-rose-800', border: 'border-l-rose-500' },
  high: { label: '高风险', badge: 'bg-orange-100 text-orange-800', border: 'border-l-orange-400' },
  medium: { label: '关注', badge: 'bg-amber-100 text-amber-800', border: 'border-l-amber-400' },
};

const plannedChecks = [
  ['多源交叉核验', '将 Tushare 与托管、交易所或其他供应商的同日净值、规模和成分做差异对账。'],
  ['分红拆分与复权链', '把分红、拆分、份额折算和复权因子连成事件链，区分真实跳变与供应商断点。'],
  ['横截面分布漂移', '按产品类型监控收益、波动率、规模和缺失率分布，识别批量异常与变点。'],
  ['PIT 历史改写', '比较相邻数据版本的历史分区，记录回填、撤回和无公告的历史数据改写。'],
];

const domainForDataset = (key: string): DataDomain => {
  if (key === 'instrument_metrics') return 'snapshot';
  if (key.startsWith('etf_')) return 'etf';
  if (key.startsWith('fund_') && key !== 'fund_company') return 'fund';
  if (key.startsWith('index_')) return 'index';
  return 'base';
};

const datasetTone = (dataset: DataRefreshStatus['datasets'][string]): Exclude<QualityTone, 'planned'> => {
  if (dataset.status === 'error' || dataset.error) return 'danger';
  if (!dataset.exists || dataset.rows === 0) return 'warning';
  return 'good';
};

const checkTone = (status: CheckStatus): QualityTone => (
  status === 'passed' ? 'good' : status === 'failed' ? 'danger' : status === 'warning' ? 'warning' : 'planned'
);

const severityWeight: Record<QualitySeverity, number> = { critical: 3, high: 2, medium: 1 };

const formatPercent = (value?: number | null) => (
  value === null || value === undefined ? '--' : `${percentFormatter.format(value * 100)}%`
);

const formatUpdatedAt = (value?: string | null) => {
  if (!value) return '--';
  const parsed = Date.parse(value);
  return Number.isFinite(parsed) ? dateTimeFormatter.format(parsed) : value;
};

const issueAction = (action: string) => ({
  actionTo: '/settings/data-sources',
  actionLabel: action === 'inspect' ? '检查数据范围' : '前往数据更新',
});

function StatusBadge({ tone, label }: { tone: QualityTone; label?: string }) {
  const style = toneStyles[tone];
  return <span className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-semibold ${style.badge}`}><span aria-hidden="true" className={`h-2 w-2 rounded-full ${style.dot}`} />{label ?? style.label}</span>;
}

function QualityMetric({ label, value, detail, tone, statusLabel }: { label: string; value: string; detail: string; tone: QualityTone; statusLabel?: string }) {
  return (
    <article className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
      <div className="flex items-start justify-between gap-3"><p className="text-sm font-semibold text-slate-600">{label}</p><StatusBadge tone={tone} label={statusLabel} /></div>
      <p className="mt-3 text-2xl font-bold tabular-nums text-slate-950">{value}</p>
      <p className="mt-1 text-xs leading-5 text-slate-500">{detail}</p>
    </article>
  );
}

function CheckCard({ check }: { check: QualityCheck }) {
  return (
    <article className="rounded-xl border border-slate-200 bg-slate-50/70 p-4">
      <div className="flex items-start justify-between gap-3"><div><p className="text-[11px] font-semibold uppercase tracking-wide text-slate-400">{dimensionLabels[check.dimension] ?? check.dimension}</p><h3 className="mt-1 font-semibold text-slate-900">{check.label}</h3></div><StatusBadge tone={checkTone(check.status)} /></div>
      <p className="mt-3 text-sm font-semibold text-slate-800">{check.summary}</p>
      <p className="mt-1 text-xs leading-5 text-slate-500">{check.detail}</p>
      {check.threshold && <p className="mt-3 border-t border-slate-200 pt-3 text-[11px] leading-5 text-slate-500"><span className="font-semibold text-slate-700">规则：</span>{check.threshold}</p>}
    </article>
  );
}

const warningIssue = (warning: DataQualityWarning, index: number): QualityIssue => ({
  id: `analytics-warning-${warning.code ?? index}-${index}`, code: warning.code ?? 'ANALYTICS_WARNING', severity: 'medium',
  dimension: 'completeness', scope: warning.kind === 'etf' ? 'ETF 产品数据' : warning.kind === 'fund' ? '场外公募基金' : '研究数据',
  title: warning.message, description: '该提示来自当前已生效的分析快照，请确认是否影响本次研究范围。',
  evidence: '分析快照返回了覆盖或可用性提示。', impact: '相关产品的部分指标可能为空或被排除。',
  affected_count: 0, affected_rate: null, record_count: 0, samples: [], action: 'inspect', ...issueAction('inspect'),
});

function IssueCard({ issue, onReload }: { issue: QualityIssue; onReload: () => void }) {
  const style = severityStyles[issue.severity];
  return (
    <li className={`border-l-4 p-4 sm:p-5 ${style.border}`}>
      <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2"><span className={`rounded-full px-2 py-0.5 text-[11px] font-semibold ${style.badge}`}>{style.label}</span><span className="rounded bg-slate-100 px-2 py-0.5 text-[11px] font-medium text-slate-500">{dimensionLabels[issue.dimension] ?? issue.dimension}</span><span className="text-xs text-slate-400">{issue.scope}</span></div>
          <h3 className="mt-2 font-semibold text-slate-950">{issue.title}</h3>
          <p className="mt-1 text-sm leading-6 text-slate-600">{issue.description}</p>
        </div>
        {issue.actionTo ? <Link to={issue.actionTo} className="shrink-0 self-start text-sm font-semibold text-indigo-700 hover:text-indigo-900">{issue.actionLabel} →</Link> : <button type="button" onClick={onReload} className="shrink-0 self-start text-sm font-semibold text-indigo-700 hover:text-indigo-900 focus:outline-none focus:ring-2 focus:ring-indigo-500">重新检查</button>}
      </div>
      <dl className="mt-4 grid gap-3 rounded-xl bg-slate-50 p-3 text-xs sm:grid-cols-2"><div><dt className="font-semibold text-slate-700">检查证据</dt><dd className="mt-1 leading-5 text-slate-600">{issue.evidence}</dd></div><div><dt className="font-semibold text-slate-700">研究影响</dt><dd className="mt-1 leading-5 text-slate-600">{issue.impact}</dd></div></dl>
      {issue.samples.length > 0 && <div className="mt-3"><p className="text-xs font-semibold text-slate-500">异常样本（最多展示 5 个）</p><ul className="mt-2 flex flex-wrap gap-2">{issue.samples.map((sample) => <li key={`${issue.id}-${sample.kind}-${sample.ts_code}`} className="rounded-lg border border-slate-200 bg-white px-3 py-2 text-xs"><span className="font-semibold text-slate-800">{sample.name || sample.ts_code}</span>{sample.name && <span className="ml-1 text-slate-400">{sample.ts_code}</span>}<span className="mt-0.5 block text-slate-500">{[sample.observed, sample.latest_date].filter(Boolean).join(' · ')}</span></li>)}</ul></div>}
    </li>
  );
}

export default function DataQuality() {
  const [analytics, setAnalytics] = useState<InstrumentAnalyticsResponse | null>(null);
  const [indexSummary, setIndexSummary] = useState<IndexSummary | null>(null);
  const [deepQuality, setDeepQuality] = useState<DeepQualityReport | null>(null);
  const [requestErrors, setRequestErrors] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [reloadVersion, setReloadVersion] = useState(0);
  const [domainFilter, setDomainFilter] = useState<'all' | DataDomain>('all');
  const [issuesOnly, setIssuesOnly] = useState(false);
  const [severityFilter, setSeverityFilter] = useState<'all' | QualitySeverity>('all');

  const handleRefreshCompleted = useCallback(() => setReloadVersion((version) => version + 1), []);
  const { status: refreshStatus, error: refreshStatusError, fetchStatus } = useDataRefresh(handleRefreshCompleted);

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setRequestErrors([]);
    Promise.allSettled([
      fetch('/api/data/quality', { signal: controller.signal }).then(async (response) => {
        if (!response.ok) throw new Error('深度质量检查结果暂不可用。');
        const payload = await response.json() as DeepQualityReport;
        assertFixedNjitExecution(payload.execution, '深度数据质量统计');
        return payload;
      }),
      fetch('/api/instruments/analytics?kind=all', { signal: controller.signal }).then(async (response) => {
        if (!response.ok) throw new Error('产品覆盖质量结果暂不可用。');
        const payload = await response.json() as InstrumentAnalyticsResponse;
        assertFixedNjitExecution(payload.execution, '产品覆盖质量统计');
        return payload;
      }),
      fetch('/api/indices/summary', { signal: controller.signal }).then(async (response) => {
        if (!response.ok) throw new Error('指数覆盖质量结果暂不可用。');
        const payload = await response.json() as IndexSummary;
        assertFixedNjitExecution(payload.execution, '指数覆盖质量统计');
        return payload;
      }),
    ]).then(([qualityResult, analyticsResult, indexResult]) => {
      if (controller.signal.aborted) return;
      const errors: string[] = [];
      if (qualityResult.status === 'fulfilled') setDeepQuality(qualityResult.value);
      else if ((qualityResult.reason as Error)?.name !== 'AbortError') { setDeepQuality(null); errors.push((qualityResult.reason as Error)?.message ?? '深度质量检查结果暂不可用。'); }
      if (analyticsResult.status === 'fulfilled') setAnalytics(analyticsResult.value);
      else if ((analyticsResult.reason as Error)?.name !== 'AbortError') { setAnalytics(null); errors.push((analyticsResult.reason as Error)?.message ?? '产品覆盖质量结果暂不可用。'); }
      if (indexResult.status === 'fulfilled') setIndexSummary(indexResult.value);
      else if ((indexResult.reason as Error)?.name !== 'AbortError') { setIndexSummary(null); errors.push((indexResult.reason as Error)?.message ?? '指数覆盖质量结果暂不可用。'); }
      setRequestErrors(errors);
      setLoading(false);
    });
    return () => controller.abort();
  }, [reloadVersion]);

  const reload = useCallback(() => {
    setReloadVersion((version) => version + 1);
    void fetchStatus();
  }, [fetchStatus]);

  const datasets = useMemo<DatasetView[]>(() => Object.entries(refreshStatus?.datasets ?? {}).map(([key, dataset]) => {
    const tone = datasetTone(dataset);
    return {
      key, label: datasetLabels[key] ?? key, domain: domainForDataset(key), file: dataset.file, tone,
      statusLabel: tone === 'danger' ? '无法读取' : tone === 'warning' ? (dataset.exists ? '空数据集' : '缺失') : '正常',
      rows: dataset.rows, earliestDate: dataset.earliest_date, latestDate: dataset.latest_date, updatedAt: dataset.updated_at,
    };
  }), [refreshStatus?.datasets]);

  const issues = useMemo<QualityIssue[]>(() => {
    const next: QualityIssue[] = (deepQuality?.issues ?? []).map((issue) => ({ ...issue, ...issueAction(issue.action) }));
    requestErrors.forEach((message, index) => next.push({
      id: `request-${index}`, code: 'QUALITY_SERVICE_UNAVAILABLE', severity: 'high', dimension: 'availability', scope: '质量检查服务',
      title: '部分质量结果无法读取', description: message, evidence: '本次检查请求未成功返回。', impact: '未返回的检查不能纳入通过结论。',
      affected_count: 0, affected_rate: null, record_count: 0, samples: [], action: 'retry',
    }));
    if (refreshStatusError) next.push({
      id: 'refresh-status', code: 'DATASET_STATUS_UNAVAILABLE', severity: 'high', dimension: 'availability', scope: '数据集状态',
      title: '无法读取本地数据集状态', description: '暂时无法确认文件是否完整，请稍后重新检查。', evidence: '数据集状态接口未成功返回。',
      impact: '文件存在性和更新时间不能纳入通过结论。', affected_count: 0, affected_rate: null, record_count: 0, samples: [], action: 'retry',
    });
    if (refreshStatus?.job.status === 'failed') next.push({
      id: 'refresh-failed', code: 'LAST_REFRESH_FAILED', severity: 'high', dimension: 'timeliness', scope: '数据更新任务',
      title: '最近一次数据更新失败', description: '任务未完整结束；增量任务已完成的部分可能已经写入，请检查日期覆盖和数据一致性。', evidence: refreshStatus.job.message,
      impact: '不能把部分更新当作整批完成；请先恢复任务，再复核质量。', affected_count: 0, affected_rate: null, record_count: 0, samples: [], action: 'refresh', ...issueAction('refresh'),
    });
    datasets.filter((dataset) => dataset.tone !== 'good').forEach((dataset) => next.push({
      id: `dataset-${dataset.key}`, code: dataset.tone === 'danger' ? 'DATASET_UNREADABLE' : 'DATASET_EMPTY_OR_MISSING',
      severity: dataset.tone === 'danger' ? 'critical' : 'high', dimension: 'availability', scope: domainLabels[dataset.domain],
      title: `${dataset.label}${dataset.statusLabel === '缺失' ? '缺失' : dataset.statusLabel === '空数据集' ? '没有有效记录' : '无法读取'}`,
      description: dataset.tone === 'danger' ? '文件存在但无法完成结构检查。' : '该数据集暂不能为研究计算提供有效记录。',
      evidence: `${dataset.file}：${dataset.statusLabel}。`, impact: '依赖该数据集的研究能力不可用或降级。',
      affected_count: 0, affected_rate: null, record_count: dataset.rows ?? 0, samples: [], action: 'refresh', ...issueAction('refresh'),
    }));
    (analytics?.data_quality.warnings ?? []).forEach((warning, index) => next.push(warningIssue(warning, index)));
    if ((indexSummary?.missing_count ?? 0) > 0) next.push({
      id: 'index-missing', code: 'INDEX_COVERAGE_MISSING', severity: 'medium', dimension: 'completeness', scope: '指数与情景数据', title: '部分指数缺少行情覆盖',
      description: `${integerFormatter.format(indexSummary?.missing_count ?? 0)} 个指数代码没有可用行情。`, evidence: `指数目录覆盖 ${formatPercent(indexSummary?.coverage_rate)}。`,
      impact: '相关市场状态识别和情景模拟不可用。', affected_count: indexSummary?.missing_count ?? 0, affected_rate: null, record_count: 0, samples: [], action: 'refresh', ...issueAction('refresh'),
    });
    if ((indexSummary?.stale_count ?? 0) > 0) next.push({
      id: 'index-stale', code: 'INDEX_SERIES_STALE', severity: 'medium', dimension: 'timeliness', scope: '指数与情景数据', title: '部分指数行情未更新到最新交易日',
      description: '应结合指数是否终止、休市或供应商确实缺数判断。', evidence: `${integerFormatter.format(indexSummary?.stale_count ?? 0)} 个指数代码被识别为陈旧数据。`,
      impact: '相关情景参数可能停留在旧时点。', affected_count: indexSummary?.stale_count ?? 0, affected_rate: null, record_count: 0, samples: [], action: 'refresh', ...issueAction('refresh'),
    });
    return next.sort((left, right) => severityWeight[right.severity] - severityWeight[left.severity]);
  }, [analytics?.data_quality.warnings, datasets, deepQuality?.issues, indexSummary, refreshStatus?.job.message, refreshStatus?.job.status, refreshStatusError, requestErrors]);

  const filteredDatasets = datasets.filter((dataset) => (
    (domainFilter === 'all' || dataset.domain === domainFilter) && (!issuesOnly || dataset.tone !== 'good')
  ));
  const filteredIssues = issues.filter((issue) => severityFilter === 'all' || issue.severity === severityFilter);
  const severityCounts = (['critical', 'high', 'medium'] as const).reduce((result, severity) => ({ ...result, [severity]: issues.filter((issue) => issue.severity === severity).length }), {} as Record<QualitySeverity, number>);
  const criticalIssueCount = severityCounts.critical;
  const refreshRunning = refreshStatus?.job.status === 'running' || refreshStatus?.refresh_locked === true;
  const summary = deepQuality?.summary;
  const qualityAvailable = Boolean(summary && deepQuality?.status !== 'unavailable');
  const checksComplete = Boolean(qualityAvailable && summary && summary.checks_total > 0 && summary.checks_unavailable === 0
    && deepQuality?.checks.length === summary.checks_total && deepQuality.checks.every((check) => check.status !== 'unavailable'));
  const checksPassed = checksComplete && summary?.checks_passed === summary?.checks_total;
  const navCheck = deepQuality?.checks.find((check) => check.key === 'nav_discontinuity');
  const navChecked = Boolean(qualityAvailable && navCheck && navCheck.status !== 'unavailable');
  const affectedKnown = Boolean(qualityAvailable && summary && (summary.affected_products > 0 || checksComplete));
  const attentionTone: QualityTone = loading ? 'planned' : criticalIssueCount > 0 || deepQuality?.status === 'blocked' ? 'danger' : issues.length > 0 ? 'warning' : checksPassed ? 'good' : 'planned';

  return (
    <div className="mx-auto min-w-0 max-w-7xl space-y-6">
      <DataWorkspaceNav />
      <header className="overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm">
        <div className="flex flex-col gap-6 p-5 sm:p-6 lg:flex-row lg:items-start lg:justify-between">
          <div className="max-w-3xl"><p className="text-xs font-semibold uppercase tracking-[0.22em] text-indigo-600">Data governance</p><h1 className="mt-2 text-2xl font-bold text-slate-950 sm:text-3xl">数据质量监控与治理</h1><p className="mt-3 text-sm leading-6 text-slate-600">从“文件能否读取”深入到产品级时间序列：检查净值突变、连续缺口、窗口覆盖、存续产品陈旧、主键、值域和源快照一致性，并明确对研究结果的影响。</p></div>
          <div className="flex shrink-0 flex-col items-stretch gap-3 sm:flex-row sm:items-center lg:flex-col lg:items-end"><div role="status" aria-live="polite"><StatusBadge tone={attentionTone} label={loading ? '正在执行质量检查' : criticalIssueCount ? `${criticalIssueCount} 项阻断问题` : issues.length ? `${issues.length} 项待处理` : checksPassed ? '已接入规则全部通过' : '检查结果尚未完整'} /></div><div className="flex flex-wrap gap-2"><button type="button" onClick={reload} disabled={loading} className="min-h-11 rounded-xl border border-slate-300 bg-white px-4 text-sm font-semibold text-slate-700 hover:bg-slate-50 focus:outline-none focus:ring-2 focus:ring-indigo-500 disabled:cursor-wait disabled:opacity-50">{loading ? '检查中…' : '重新检查'}</button><Link to="/settings/data-sources" className="inline-flex min-h-11 items-center rounded-xl bg-slate-900 px-4 text-sm font-semibold text-white hover:bg-slate-700 focus:outline-none focus:ring-2 focus:ring-slate-600 focus:ring-offset-2">数据下载与更新 →</Link></div></div>
        </div>
        <div className="grid gap-px border-t border-slate-200 bg-slate-200 sm:grid-cols-3"><div className="bg-slate-50 px-5 py-3 text-xs leading-5 text-slate-600"><span className="font-semibold text-slate-800">当前研究时点：</span>{deepQuality?.as_of ?? analytics?.as_of ?? '--'}</div><div className="bg-indigo-50 px-5 py-3 text-xs leading-5 text-indigo-800"><span className="font-semibold">质量快照生成：</span>{formatUpdatedAt(deepQuality?.generated_at)}</div><div className="bg-slate-50 px-5 py-3 text-xs leading-5 text-slate-600"><span className="font-semibold text-slate-800">数据版本激活：</span>{formatUpdatedAt(deepQuality?.activated_at)}</div></div>
      </header>

      {refreshRunning && <div role="status" className="flex flex-col gap-2 rounded-xl border border-indigo-200 bg-indigo-50 px-4 py-3 text-sm text-indigo-900 sm:flex-row sm:items-center sm:justify-between"><span><span className="font-semibold">数据更新正在后台运行。</span> 本页展示当前本地数据，更新期间结果可能变化；任务完成后会自动重新检查。</span><Link to="/settings/data-sources" className="shrink-0 font-semibold text-indigo-700 hover:text-indigo-900">查看更新进度 →</Link></div>}

      <section aria-label="数据质量概览" className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <QualityMetric label="深度检查规则" value={summary ? `${summary.checks_passed} / ${summary.checks_total}` : '--'} detail={summary ? `${summary.checks_warning} 项关注 · ${summary.checks_failed} 项未通过 · ${summary.checks_unavailable} 项未检查` : '等待有效质量快照'} tone={loading || !summary || deepQuality?.status === 'unavailable' ? 'planned' : checksPassed ? 'good' : summary.checks_failed ? 'danger' : 'warning'} statusLabel={loading ? '检查中' : undefined} />
        <QualityMetric label="受影响产品" value={affectedKnown && summary ? integerFormatter.format(summary.affected_products) : '--'} detail={affectedKnown && summary ? `占已检查 ${integerFormatter.format(summary.total_products)} 个产品的 ${formatPercent(summary.affected_rate)}${checksComplete ? '' : '；仍有规则未完成，影响范围尚未完整'}` : '产品级规则尚未完成，不能判断影响范围'} tone={loading || !affectedKnown ? 'planned' : summary?.affected_products || !checksPassed ? 'warning' : 'good'} statusLabel={loading ? '检查中' : undefined} />
        <QualityMetric label="净值突变" value={navChecked && summary ? integerFormatter.format(summary.nav_anomaly_products) : '--'} detail={navChecked && summary ? `共 ${integerFormatter.format(summary.nav_anomaly_events)} 个异常点，使用参考净值交叉确认` : '净值突变规则尚未完成，暂无检查结论'} tone={loading || !navChecked ? 'planned' : summary?.nav_anomaly_products ? 'warning' : checkTone(navCheck!.status)} statusLabel={loading ? '检查中' : undefined} />
        <QualityMetric label="待处理问题" value={loading ? '--' : integerFormatter.format(issues.length)} detail={`${criticalIssueCount} 项阻断 · ${severityCounts.high} 项高风险 · ${severityCounts.medium} 项关注`} tone={attentionTone} statusLabel={loading ? '检查中' : undefined} />
      </section>

      <section aria-labelledby="quality-checks-title" className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm sm:p-6">
        <div className="flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between"><div><h2 id="quality-checks-title" className="text-lg font-semibold text-slate-950">深度检查矩阵</h2><p className="mt-1 text-sm text-slate-500">每项结果均来自当前已生效数据版本；未检查项不会被算作通过。</p></div><span className="text-xs text-slate-400">{deepQuality?.validation.status === 'passed' ? '全量验收记录已绑定' : '缺少全量验收记录'}</span></div>
        {loading && !deepQuality && <div className="mt-5 grid gap-3 md:grid-cols-2 xl:grid-cols-3">{Array.from({ length: 6 }, (_, index) => <div key={index} className="h-40 animate-pulse rounded-xl bg-slate-100" />)}</div>}
        {deepQuality && <div className="mt-5 grid gap-3 md:grid-cols-2 xl:grid-cols-3">{deepQuality.checks.map((check) => <CheckCard key={check.key} check={check} />)}</div>}
      </section>

      <section aria-labelledby="quality-issues-title" className="rounded-2xl border border-slate-200 bg-white shadow-sm">
        <div className="flex flex-col gap-4 border-b border-slate-200 p-5 sm:p-6 lg:flex-row lg:items-end lg:justify-between"><div><h2 id="quality-issues-title" className="text-lg font-semibold text-slate-950">问题工作台</h2><p className="mt-1 text-sm text-slate-500">按严重级别处理；每条问题都说明证据、研究影响和异常样本。</p></div><div className="flex flex-wrap gap-2" aria-label="问题风险筛选">{([['all', '全部问题'], ['critical', '阻断'], ['high', '高风险'], ['medium', '关注']] as const).map(([value, label]) => { const count = value === 'all' ? issues.length : severityCounts[value]; const selected = severityFilter === value; return <button key={value} type="button" aria-pressed={selected} onClick={() => setSeverityFilter(value)} className={`min-h-10 rounded-lg border px-3 text-sm font-semibold focus:outline-none focus:ring-2 focus:ring-indigo-500 ${selected ? 'border-indigo-300 bg-indigo-50 text-indigo-800' : 'border-slate-200 bg-white text-slate-600 hover:bg-slate-50'}`}>{label} <span className="tabular-nums">{count}</span></button>; })}</div></div>
        {loading && <div className="m-5 h-28 animate-pulse rounded-xl bg-slate-100 sm:m-6" aria-label="正在加载质量问题" />}
        {!loading && filteredIssues.length === 0 && <div className="m-5 rounded-xl border border-emerald-200 bg-emerald-50 p-5 text-sm text-emerald-800 sm:m-6"><p className="font-semibold">当前筛选级别没有待处理问题。</p><p className="mt-1 text-xs leading-5 text-emerald-700">该结论仅覆盖上方已接入且有结果的检查规则。</p></div>}
        {!loading && filteredIssues.length > 0 && <ul className="divide-y divide-slate-100">{filteredIssues.map((issue) => <IssueCard key={issue.id} issue={issue} onReload={reload} />)}</ul>}
      </section>

      <section aria-labelledby="dataset-details-title" className="min-w-0 rounded-2xl border border-slate-200 bg-white p-5 shadow-sm sm:p-6">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between"><div><h2 id="dataset-details-title" className="text-lg font-semibold text-slate-950">底层数据集明细</h2><p className="mt-1 text-sm text-slate-500">用于定位缺失、空文件和读取异常；深度规则结果以上方质量快照为准。</p></div><div className="flex flex-col gap-3 sm:flex-row sm:items-end"><label className="text-xs font-semibold text-slate-600">数据域<select aria-label="数据域" value={domainFilter} onChange={(event) => setDomainFilter(event.target.value as typeof domainFilter)} className="mt-1 min-h-11 w-full rounded-lg border border-slate-300 bg-white px-3 text-sm font-medium text-slate-700 focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-500 sm:w-44"><option value="all">全部数据域</option><option value="base">基础与机构数据</option><option value="etf">ETF 产品数据</option><option value="fund">场外公募基金</option><option value="index">指数与情景数据</option><option value="snapshot">研究指标快照</option></select></label><button type="button" aria-pressed={issuesOnly} onClick={() => setIssuesOnly((current) => !current)} className={`min-h-11 rounded-lg border px-4 text-sm font-semibold focus:outline-none focus:ring-2 focus:ring-indigo-500 ${issuesOnly ? 'border-amber-300 bg-amber-50 text-amber-800' : 'border-slate-300 bg-white text-slate-700 hover:bg-slate-50'}`}>只看需处理</button></div></div>
        <div className="mt-5 max-w-full overflow-x-auto rounded-xl border border-slate-200" tabIndex={0} aria-label="核心数据集质量明细滚动区域"><table className="min-w-[900px] divide-y divide-slate-200 text-left text-sm"><caption className="sr-only">核心数据集质量明细</caption><thead className="bg-slate-50"><tr>{['数据集', '数据域', '检查结果', '行数', '日期覆盖', '文件更新时间'].map((label) => <th key={label} scope="col" className="whitespace-nowrap px-3 py-3 text-xs font-semibold text-slate-600">{label}</th>)}</tr></thead><tbody className="divide-y divide-slate-100">{filteredDatasets.map((dataset) => <tr key={dataset.key} className="hover:bg-slate-50"><td className="px-3 py-3"><div className="font-medium text-slate-800">{dataset.label}</div><div className="mt-0.5 text-xs text-slate-400">{dataset.file}</div></td><td className="px-3 py-3 text-slate-600">{domainLabels[dataset.domain]}</td><td className="px-3 py-3"><StatusBadge tone={dataset.tone} label={dataset.statusLabel} /></td><td className="px-3 py-3 text-right tabular-nums text-slate-600">{dataset.rows === null || dataset.rows === undefined ? '--' : integerFormatter.format(dataset.rows)}</td><td className="whitespace-nowrap px-3 py-3 tabular-nums text-slate-600">{dataset.earliestDate || dataset.latestDate ? `${dataset.earliestDate ?? '--'} ～ ${dataset.latestDate ?? '--'}` : '不适用'}</td><td className="whitespace-nowrap px-3 py-3 tabular-nums text-slate-500">{formatUpdatedAt(dataset.updatedAt)}</td></tr>)}{!loading && filteredDatasets.length === 0 && <tr><td colSpan={6} className="px-4 py-12 text-center text-slate-500">当前筛选范围没有需要展示的数据集。</td></tr>}{loading && datasets.length === 0 && <tr><td colSpan={6} className="px-4 py-12 text-center text-slate-500">正在检查底层数据集...</td></tr>}</tbody></table></div>
      </section>

      <section aria-labelledby="quality-roadmap-title" className="rounded-2xl border border-slate-200 bg-slate-900 p-5 text-white sm:p-6">
        <div className="flex flex-col gap-3 lg:flex-row lg:items-end lg:justify-between"><div><p className="text-xs font-semibold uppercase tracking-[0.18em] text-indigo-300">Quality roadmap</p><h2 id="quality-roadmap-title" className="mt-2 text-lg font-semibold">下一阶段的高阶检查</h2><p className="mt-1 max-w-3xl text-sm leading-6 text-slate-300">这些问题值得检查，但当前缺少独立数据源、事件链或历史版本证据，因此不会出现在“通过”数量里。</p></div><span className="self-start rounded-full bg-slate-700 px-3 py-1 text-xs font-semibold text-slate-200 lg:self-auto">待接入 · 不计入当前通过结论</span></div>
        <div className="mt-5 grid gap-3 md:grid-cols-2">{plannedChecks.map(([title, description]) => <article key={title} className="rounded-xl border border-slate-700 bg-slate-800/70 p-4"><h3 className="font-semibold text-white">{title}</h3><p className="mt-1 text-xs leading-5 text-slate-300">{description}</p></article>)}</div>
        <div className="mt-5 flex flex-col gap-3 border-t border-slate-700 pt-5 sm:flex-row sm:items-center sm:justify-between"><p className="text-xs leading-5 text-slate-400">质量检查回答“数据能不能用”；PIT 快照回答“当时用了哪一版”。两者不能互相替代。</p><Link to="/settings/pit-snapshots" className="inline-flex min-h-10 shrink-0 items-center justify-center rounded-xl border border-slate-600 px-4 text-sm font-semibold hover:bg-slate-800 focus:outline-none focus:ring-2 focus:ring-white">查看 PIT 时点快照 →</Link></div>
      </section>
    </div>
  );
}

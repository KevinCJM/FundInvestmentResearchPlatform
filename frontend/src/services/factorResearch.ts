import { assertFixedNjitExecution, type FixedNjitExecutionAudit } from '../utils/fixedNjitExecution'

export type NumberValue = number | null
export type FactorKind = 'etf' | 'fund' | 'stock'
export type ContextType = 'product_research' | 'saa' | 'taa' | 'allocation' | 'portfolio' | 'post_investment' | 'regime'
export interface FactorFields {
  name: string; description: string; operator: 'momentum' | 'volatility' | 'drawdown' | 'reversal'
  window: number; skip: number; direction: -1 | 1; product_kinds: FactorKind[]
}
export interface FactorDefinition extends FactorFields { id: string; revision: number; read_only: boolean }
export interface FactorRef { factor_id: string; revision: number; weight: number }
export interface StudyDraft {
  name: string; product_kind: FactorKind; asset_class: 'equity' | 'bond' | 'commodity' | 'multi_asset'
  market: 'CN'; currency: 'CNY'; targets: string[]; universe_source: string
  start_date: string; end_date: string; oos_date: string
  benchmark: { kind: 'etf' | 'index'; code: string; label: string; return_basis: 'adjusted_nav' | 'price_index' | 'total_return_index' }
  factors: FactorRef[]; normalization: 'rank' | 'zscore'; horizon: number; quantiles: number; top_n: number; cost_bps: number
  signal_frequency?: 'daily' | 'weekly' | 'monthly'; ic_window?: number; ic_min_periods?: number
  model: 'characteristic_composite'; dataset: 'active_adjusted_nav'
}
export interface Study extends StudyDraft { id: string; revision: number; created_at: string; updated_at: string }
export interface ResearchCatalog {
  factors: FactorDefinition[]
  models: Array<{ id: string; name: string; purpose: string; available: boolean }>
  contexts: Array<{ id: ContextType; name: string; path: string }>
  snapshot: { id: string; latest_date: string | null }
  capabilities: Array<{ kind: FactorKind; available: boolean; reason?: string }>
  default_study: StudyDraft; ready: boolean
}
export interface RunRecord { id: string; name: string; created_at: string; study_id?: string; study_revision?: number }
export interface FactorScore {
  product_id: string; code: string; name: string; kind: FactorKind; score: NumberValue; rank: NumberValue
  percentile: NumberValue; status: 'ranked' | 'excluded'; exclusion_reason: string | null
  factors: Array<{ factor_id: string; revision: number; name: string; raw_value: NumberValue; normalized_value: NumberValue; contribution: NumberValue }>
}
export interface FactorStats { observations: number; mean: NumberValue; std: NumberValue; icir: NumberValue; positive_rate: NumberValue }
export interface RunSummary {
  factors: Array<{ name: string; factor_id: string; ic: FactorStats; rank_ic: FactorStats }>
  performance: Record<'days' | 'total_return' | 'annualized_return' | 'annualized_volatility' | 'max_drawdown' | 'benchmark_return' | 'excess_return' | 'turnover' | 'fee_sum', NumberValue>
  group_returns: FactorStats[]; factor_correlation: NumberValue[][]
}
export interface FactorRun extends RunRecord {
  kind: 'run'; study_snapshot: Study; factor_snapshots: FactorDefinition[]; as_of: string
  input_checksum: string; engine_version: string; execution: FixedNjitExecutionAudit
  summaries: { in_sample: RunSummary; out_of_sample: RunSummary }
  periods: Array<{ date: string; entry_date: string | null; label_end: string | null; sample: string; ic: NumberValue[]; rank_ic: NumberValue[]; pair_counts: number[]; group_returns: NumberValue[] }>
  curves: Array<{ date: string; nav: NumberValue; benchmark_nav: NumberValue; turnover: NumberValue; cost: NumberValue }>
  latest_scores: FactorScore[]; warnings: string[]; data_quality: Array<{ code: string; observations: number; reason?: string }>
  data_lineage: { snapshot: string; generation: string; [key: string]: unknown }
  rolling_diagnostics?: { window: number; min_periods: number; date_basis: 'label_end'; rows: Array<{ date: string; signal_date: string; sample: string; ic: FactorStats[]; rank_ic: FactorStats[] }> }
}
export interface FactorRelease {
  id: string; name: string; run_id: string; as_of: string; effective_from: string; effective_to: string
  state: 'active' | 'expired' | 'retired' | 'scheduled' | 'stale'; usage: 'research_only'; note: string
  product_score?: FactorScore
}
export interface FactorBinding { release_id: string; run_id: string; context_type: ContextType; context_id: string; note: string; created_at: string }
export interface FactorDataset {
  id: string; name: string; market: string; currency: string; observations: number; source_url: string; start_date: string; end_date: string
  factor_names?: string[]; dependent_return?: 'total' | 'excess'; source_method?: string; return_plan_id?: string; return_plan_revision?: number
}
export interface ReturnPlanDraft {
  name: string; method: 'characteristic_spread' | 'ff3_2x3'; source_run_id: string | null; source_panel_id: string | null
  factor_key: string; quantiles: number; cost_bps: number; output_factor: string
}
export interface ReturnPlan extends ReturnPlanDraft { id: string; revision: number }
export interface ReturnSource { id: string; name: string; market: string; currency: string; start_date: string; end_date: string; observations: number; checksum: string }
export interface ReturnCatalog {
  ready: boolean
  methods: Array<{ id: string; name: string; available: boolean; input?: string; output?: string; reason?: string }>
  source_template: Record<string, unknown>; dataset_template: Record<string, unknown>; source_fields: Record<string, string>
}
export interface ReturnDataset extends Omit<FactorDataset, 'observations' | 'start_date' | 'end_date'> {
  kind: 'dataset'; created_at: string; frequency: 'daily'; units: 'decimal_return'; construction: string
  factor_names: string[]; dependent_return: 'total' | 'excess'; rows: Array<{ date: string } & Record<string, string | NumberValue>>
  warnings: string[]; input_checksum?: string; checksum?: string; execution?: FixedNjitExecutionAudit
  plan_snapshot?: ReturnPlan; source_run_id?: string; source_panel_id?: string
  diagnostics?: {
    factors: Array<{ factor: string; observations: number; mean: NumberValue; std: NumberValue; positive_rate: NumberValue }>
    correlation: NumberValue[][]; cumulative: Array<{ date: string; values: NumberValue[] }>; cumulative_meaning: string
  }
  leg_returns?: Array<{ date: string } & Record<string, string | NumberValue>>
  formation_evidence?: Array<{ date: string; counts: Record<string, number>; size_break: number; bm30: number; bm70: number }>
}
export interface AttributionRequest {
  name: string; product_kind: 'etf' | 'fund'; targets: string[]; model: 'rbsa' | 'ff3' | 'factor_regression'; indices: string[]
  dataset_id?: string; market: 'CN'; currency: 'CNY'; start_date: string; end_date: string; oos_date: string
  exposure_mode?: 'fixed' | 'rolling'; rolling_window?: number; min_observations?: number; refit_step?: number
}
export interface ContributionComponent { id: string; label: string; kind: 'factor' | 'risk_free' | 'intercept' | 'residual' }
export interface ContributionDay {
  date: string; sample: 'in_sample' | 'out_of_sample'; status: 'ok' | 'warmup' | 'unavailable'; reason: string | null
  actual_return: NumberValue; exposures: NumberValue[]; factor_returns: NumberValue[]; contributions: NumberValue[]
  contribution_sum: NumberValue; reconciliation_error: NumberValue; fit_start: string | null; fit_end: string | null
  fit_observations: NumberValue; fit_r2: NumberValue; exposure_status: 'ok' | 'unavailable'
  exposure_basis: 'prior_window' | 'retrospective_fit' | 'fixed_training_fit'
}
export interface ContributionSummary {
  start_date: string | null; end_date: string | null; days: number; valid_days: number; actual_days: number
  total_return: NumberValue; contributions: NumberValue[]; contribution_sum: NumberValue; reconciliation_error: NumberValue
  model_r2: NumberValue; residual_volatility: NumberValue; coverage: NumberValue
  status: 'complete' | 'incomplete' | 'empty' | 'numerical_error'
}
export interface ContributionCurve { date: string; contributions: NumberValue[]; total_return: NumberValue; contribution_sum: NumberValue; reconciliation_error: NumberValue }
export interface ContributionProduct {
  code: string; name: string; daily: ContributionDay[]; summaries: Record<string, ContributionSummary>
  curves: Record<'all' | 'in_sample' | 'out_of_sample', ContributionCurve[]>
}
export interface ContributionAnalysis {
  schema_version: 1; engine_version: string; mode: 'fixed' | 'rolling'; dependent_return: 'total' | 'excess'
  linking_method: 'beginning_wealth_weighted'; units: 'decimal_return_contribution'; warmup_days: number; evaluation_start: string
  summary_basis: string; components: ContributionComponent[]; products: ContributionProduct[]; notes: string[]
}
export interface AttributionRun extends RunRecord {
  attribution?: ContributionAnalysis
  request: AttributionRequest; as_of: string; execution: FixedNjitExecutionAudit; warnings: string[]
  results: Array<{ name: string; code: string; status: string; reason: string | null; exposures: Array<{ factor: string; value: NumberValue }>
    train_r2: NumberValue; test_r2: NumberValue; train_observations: number; test_observations: number
    annualized_intercept: NumberValue; train_residual_volatility: NumberValue; test_residual_volatility: NumberValue }>
}
export interface ReleaseMonitor {
  release: FactorRelease; latest_run_id: string; latest_run_at: string; data_changed_since_run: boolean; comparable: boolean
  drift: { score_correlation: NumberValue; mean_absolute_score_change: NumberValue; common_products: number; coverage: NumberValue } | null
  bindings: FactorBinding[]; note: string
}
export interface PortfolioFactorProfile { id: string; meaning: string; factors: Array<{ name: string; value: NumberValue; covered_weight: NumberValue }> }

const prefix = '/api/factor-research'
async function request<T>(path: string, method = 'GET', body?: unknown, signal?: AbortSignal): Promise<T> {
  const response = await fetch(prefix + path, { method, signal, headers: body === undefined ? undefined : { 'Content-Type': 'application/json' }, body: body === undefined ? undefined : JSON.stringify(body) })
  const value = await response.json().catch(() => null)
  if (!response.ok) {
    const detail = value?.detail
    const message = typeof detail === 'string' ? detail : Array.isArray(detail) ? detail.map((item: { msg: string }) => item.msg).join('；') : detail?.message
    throw new Error(message || `请求失败（${response.status}）`)
  }
  if (value?.execution) assertFixedNjitExecution(value.execution, '因子研究')
  return value as T
}
async function numerical<T>(path: string, method = 'GET', body?: unknown): Promise<T> {
  const value = await request<T & { execution?: unknown }>(path, method, body)
  assertFixedNjitExecution(value.execution, '因子研究')
  return value
}
export const factorApi = {
  catalog: () => request<ResearchCatalog>('/catalog'),
  factors: () => request<{ items: FactorDefinition[] }>('/factors'),
  saveFactor: (fields: FactorFields, current?: FactorDefinition) => request<FactorDefinition>(current ? `/factors/${current.id}` : '/factors', current ? 'PUT' : 'POST', current ? { ...fields, revision: current.revision } : fields),
  products: (kind: FactorKind, query: string, signal?: AbortSignal) => request<{ items: Array<{ ts_code: string; name: string; fund_type?: string }> }>(`/products?kind=${kind}&query=${encodeURIComponent(query)}`, 'GET', undefined, signal),
  studies: () => request<{ items: Study[] }>('/studies'),
  saveStudy: (fields: StudyDraft, current?: Study) => request<Study>(current ? `/studies/${current.id}` : '/studies', current ? 'PUT' : 'POST', current ? { ...fields, revision: current.revision } : fields),
  run: (study: Study) => numerical<FactorRun>(`/studies/${study.id}/runs`, 'POST', { revision: study.revision }),
  runs: () => request<{ items: RunRecord[] }>('/runs'),
  getRun: (id: string) => numerical<FactorRun>(`/runs/${encodeURIComponent(id)}`),
  datasets: () => request<{ items: FactorDataset[] }>('/datasets'),
  returnCatalog: () => request<ReturnCatalog>('/return-catalog'),
  returnPlans: () => request<{ items: ReturnPlan[] }>('/return-plans'),
  saveReturnPlan: (fields: ReturnPlanDraft, current?: ReturnPlan) => request<ReturnPlan>(current ? `/return-plans/${current.id}` : '/return-plans', current ? 'PUT' : 'POST', current ? { ...fields, revision: current.revision } : fields),
  runReturnPlan: (plan: ReturnPlan) => numerical<ReturnDataset>(`/return-plans/${plan.id}/runs`, 'POST', { revision: plan.revision }),
  returnSources: () => request<{ items: ReturnSource[] }>('/return-sources'),
  importReturnSource: (value: unknown) => request<ReturnSource>('/return-sources', 'POST', value),
  importReturnDataset: (value: unknown) => numerical<ReturnDataset>('/return-datasets', 'POST', value),
  getReturnDataset: (id: string) => request<ReturnDataset>(`/return-datasets/${encodeURIComponent(id)}`),
  returnCsvUrl: (id: string) => `${prefix}/return-datasets/${encodeURIComponent(id)}/export`,
  exportReturnCsv: async (id: string): Promise<Blob> => {
    const response = await fetch(`${prefix}/return-datasets/${encodeURIComponent(id)}/export`)
    if (!response.ok) {
      const value = await response.json().catch(() => null)
      throw new Error(value?.detail?.message || `导出失败（${response.status}）`)
    }
    if (!response.headers.get('content-type')?.includes('text/csv')) throw new Error('导出响应不是 CSV，已停止下载，请检查服务版本。')
    return response.blob()
  },
  importDataset: (value: unknown) => request<FactorDataset>('/datasets', 'POST', value),
  attribution: (value: AttributionRequest) => numerical<AttributionRun>('/attributions', 'POST', value),
  attributions: () => request<{ items: RunRecord[] }>('/attributions'),
  getAttribution: (id: string) => numerical<AttributionRun>(`/runs/${encodeURIComponent(id)}`),
  releases: (productId?: string) => request<{ items: FactorRelease[] }>(`/releases${productId ? '?product_id=' + encodeURIComponent(productId) : ''}`),
  publish: (value: { run_id: string; name: string; effective_from: string; effective_to: string; note: string }) => request<FactorRelease>('/releases', 'POST', value),
  retire: (id: string) => request<FactorRelease>(`/releases/${id}/retire`, 'POST'),
  monitor: (id: string) => numerical<ReleaseMonitor>(`/releases/${id}/monitor`),
  bindings: (contextType?: string, contextId?: string) => request<{ items: FactorBinding[] }>(`/bindings?context_type=${encodeURIComponent(contextType || '')}&context_id=${encodeURIComponent(contextId || '')}`),
  bind: (value: Omit<FactorBinding, 'created_at' | 'run_id'>) => request<FactorBinding>('/bindings', 'POST', value),
  profile: (id: string, holdings: Array<{ product_id: string; weight: number }>, asOf: string) => numerical<PortfolioFactorProfile>(`/releases/${id}/portfolio-profile`, 'POST', { holdings, as_of: asOf }),
}
export const numberText = (value: NumberValue | undefined, digits = 3) => value == null || !Number.isFinite(value) ? '—' : value.toLocaleString('zh-CN', { maximumFractionDigits: digits })
export const percentText = (value: NumberValue | undefined) => value == null || !Number.isFinite(value) ? '—' : new Intl.NumberFormat('zh-CN', { style: 'percent', maximumFractionDigits: 2 }).format(value)
export const parseCodes = (value: string) => [...new Set(value.split(/[\s,，;；]+/).map(v => v.trim()).filter(Boolean))]
export function studyDraft(value: StudyDraft): StudyDraft {
  const { name, product_kind, asset_class, market, currency, targets, universe_source, start_date, end_date, oos_date, benchmark, factors, normalization, horizon, quantiles, top_n, cost_bps, model, dataset, signal_frequency = 'monthly', ic_window = 12, ic_min_periods = 6 } = value
  return { name, product_kind, asset_class, market, currency, targets, universe_source, start_date, end_date, oos_date, benchmark, factors, normalization, horizon, quantiles, top_n, cost_bps, model, dataset, signal_frequency, ic_window, ic_min_periods }
}

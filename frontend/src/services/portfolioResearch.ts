import type { IndicatorDefinition, MetricPresentation } from './customIndicators'
import { assertFixedNjitExecution, type FixedNjitExecutionAudit } from '../utils/fixedNjitExecution'
import type { HistoricalRegimeBacktestReference, RegimeConditioningResult } from './portfolioRegime'

export type PortfolioProductKind = 'etf' | 'fund'
export type PortfolioMethod = 'equal_weight' | 'manual' | 'risk_budget' | 'target_optimization'
export type PortfolioExportTable = 'summary' | 'components' | 'daily-contributions' | 'weight-path' | 'covariance' | 'correlation' | 'risk-contributions' | 'scenario-summary' | 'scenario-series'

export interface PortfolioInstrument {
  product_id: string
  kind: PortfolioProductKind
  name: string
  code?: string | null
}

export interface PortfolioConstituent extends PortfolioInstrument {
  weight?: number
  risk_budget?: number
  asset_class_id?: string
  asset_class_name?: string
}

export interface PortfolioConstraint {
  min_weight: number
  max_weight: number
  max_turnover?: number | null
}

export interface PortfolioRunRequest {
  name: string
  universe_snapshot_id?: string | null
  constituents: PortfolioConstituent[]
  method: PortfolioMethod
  constraints: PortfolioConstraint
  window: { mode: 'all' | 'rolling'; observations?: number; start_date?: string; end_date?: string }
  rebalance: { frequency: 'fixed' | 'weekly' | 'monthly' | 'yearly'; transaction_cost_bps: number }
  benchmark?: { kind: PortfolioProductKind; product_id: string; name?: string } | null
  objective?: 'max_sharpe' | 'min_volatility' | 'target_return' | null
  target_return?: number | null
}

export interface PortfolioSeriesPoint { date: string; value: number }
export interface PortfolioMetric {
  metric_id?: string
  name: string
  value: number | null
  unit?: string
  direction?: 'higher_better' | 'lower_better'
  status?: 'ok' | 'warning' | 'unavailable' | 'error'
  warnings?: string[]
  presentation?: MetricPresentation
}
export type PortfolioIndicatorDefinition = IndicatorDefinition
export interface PortfolioWeightPoint { date: string; weights: Record<string, number> }
export interface PortfolioContribution { product_id: string; name: string; contribution: number | null; risk_contribution?: number | null }
export interface PortfolioRun {
  id: string
  target_id?: string | null
  name: string
  request?: PortfolioRunRequest
  nav: PortfolioSeriesPoint[]
  drawdown: PortfolioSeriesPoint[]
  metrics: PortfolioMetric[]
  weights: PortfolioWeightPoint[]
  contributions: PortfolioContribution[]
  correlation?: { labels: string[]; values: number[][] } | null
  covariance?: { labels: string[]; values: number[][] } | null
  custom_indicators?: PortfolioMetric[]
  rebalances?: Array<{ date: string; turnover?: number | null; message?: string | null }>
  regime_conditioning?: RegimeConditioningResult | null
  warnings: string[]
}

export interface PortfolioWarning { code?: string; message: string }

export interface PortfolioDiagnosis {
  summary: PortfolioMetric[] | Record<string, number | null>
  components: Array<{ product_id: string; name: string; kind?: PortfolioProductKind; weight?: number | null; warnings?: string[] }>
  custom_indicators?: PortfolioMetric[]
  contributions?: PortfolioContribution[]
  contribution_series?: Array<{ date: string; values: Record<string, number> }>
  covariance?: { labels: string[]; values: number[][] } | null
  correlation?: { labels: string[]; values: number[][] } | null
  risk_contributions?: PortfolioContribution[]
  concentration?: PortfolioMetric[]
  weight_path?: PortfolioWeightPoint[]
  rebalances?: Array<{ date: string; turnover?: number | null; message?: string | null }>
  warnings: string[]
}

export interface ResearchTarget {
  id: string
  name: string
  kind: 'portfolio'
  revision: number
  created_at?: string
  updated_at?: string
  definition: Record<string, unknown>
}

export class PortfolioResearchApiError extends Error {
  constructor(readonly status: number, message: string) {
    super(message)
    this.name = 'PortfolioResearchApiError'
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
  })
  if (!response.ok) {
    let message = `请求失败（${response.status}）`
    try {
      const body = await response.json()
      message = body?.detail?.message ?? body?.detail ?? body?.message ?? message
    } catch { /* keep stable fallback */ }
    throw new PortfolioResearchApiError(response.status, String(message))
  }
  return response.json() as Promise<T>
}

function seriesPoints(dates: unknown, values: unknown): PortfolioSeriesPoint[] {
  if (!Array.isArray(values)) return []
  if (values.every((item) => typeof item === 'object' && item !== null && 'date' in item && 'value' in item)) {
    return values.flatMap((item): PortfolioSeriesPoint[] => {
      const point = item as { date?: unknown; value?: unknown }
      return typeof point.date === 'string' && typeof point.value === 'number' ? [{ date: point.date, value: point.value }] : []
    })
  }
  return values.flatMap((value, index): PortfolioSeriesPoint[] => typeof value === 'number' && Array.isArray(dates) && typeof dates[index] === 'string' ? [{ date: dates[index], value }] : [])
}

const SUMMARY_PRESENTATION: Record<string, { name: string; displayFormat: 'number' | 'percent'; precision: number; direction: 'higher_better' | 'lower_better' }> = {
  cumulative_return: { name: '累计收益率', displayFormat: 'percent', precision: 2, direction: 'higher_better' },
  annual_return: { name: '年化收益率', displayFormat: 'percent', precision: 2, direction: 'higher_better' },
  annual_volatility: { name: '年化波动率', displayFormat: 'percent', precision: 2, direction: 'lower_better' },
  sharpe_ratio: { name: '夏普比率', displayFormat: 'number', precision: 3, direction: 'higher_better' },
  max_drawdown: { name: '最大回撤', displayFormat: 'percent', precision: 2, direction: 'lower_better' },
  var_99: { name: '历史 VaR 99%', displayFormat: 'percent', precision: 3, direction: 'lower_better' },
  es_99: { name: '历史 CVaR 99%', displayFormat: 'percent', precision: 3, direction: 'lower_better' },
}

function fallbackMetricPresentation(id: string, name: string, unit?: string): MetricPresentation {
  const summary = SUMMARY_PRESENTATION[id]
  const displayFormat = summary?.displayFormat ?? (unit === 'percent' ? 'percent' : 'number')
  return {
    indicator_id: `portfolio-summary-${id}`, revision: 1, name: summary?.name ?? name,
    source: 'built_in', category: 'portfolio_summary', category_label: '组合汇总', context_kind: 'portfolio',
    catalog_status: 'current', display_format: displayFormat, precision: summary?.precision ?? 3,
    unit: displayFormat === 'percent' ? '%' : unit ?? '', notation: 'standard', value_scale: displayFormat === 'percent' ? 100 : 1,
    output_measure: displayFormat === 'percent' ? 'return_decimal' : 'dimensionless', direction: summary?.direction ?? 'higher_better',
    description: '', methodology: '', data_basis: '锁定运行快照', minimum_observations: 1, applicable_product_kinds: ['portfolio'],
  }
}

function metricList(value: unknown): PortfolioMetric[] {
  if (Array.isArray(value)) return value.flatMap((item): PortfolioMetric[] => {
    if (!item || typeof item !== 'object') return []
    const record = item as Record<string, unknown>
    const id = String(record.metric_id ?? record.indicator_id ?? record.name ?? '')
    const name = String(record.name ?? id)
    const unit = typeof record.unit === 'string' ? record.unit : undefined
    const raw = record.value
    return [{
      metric_id: id,
      name,
      value: typeof raw === 'number' && Number.isFinite(raw) ? raw : null,
      unit,
      status: record.status === 'ok' || record.status === 'warning' || record.status === 'unavailable' || record.status === 'error' ? record.status : undefined,
      warnings: warningMessages(record.warnings),
      presentation: record.presentation && typeof record.presentation === 'object' ? record.presentation as unknown as MetricPresentation : fallbackMetricPresentation(id, name, unit),
    }]
  })
  if (!value || typeof value !== 'object') return []
  return Object.entries(value as Record<string, unknown>).flatMap(([name, raw]): PortfolioMetric[] => {
    const spec = SUMMARY_PRESENTATION[name]
    if (typeof raw === 'number' || raw === null) return [{ name: spec?.name ?? name, metric_id: name, value: raw, presentation: fallbackMetricPresentation(name, spec?.name ?? name) }]
    return []
  })
}

function warningMessages(value: unknown): string[] {
  if (!Array.isArray(value)) return []
  return value.flatMap((warning): string[] => {
    if (typeof warning === 'string') return warning ? [warning] : []
    if (warning && typeof warning === 'object') {
      const { message, code } = warning as PortfolioWarning
      return typeof message === 'string' && message ? [message] : typeof code === 'string' && code ? [code] : []
    }
    return []
  })
}

function portfolioMetricList(value: unknown): PortfolioMetric[] {
  const source = value && typeof value === 'object' && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null
  const items = Array.isArray(value) ? value : Array.isArray(source?.results) ? source.results : []
  return items.flatMap((item): PortfolioMetric[] => {
    if (!item || typeof item !== 'object') return []
    const record = item as Record<string, unknown>
    const name = record.indicator_name ?? record.name ?? record.indicator_id
    if (typeof name !== 'string' || !name) return []
    const rawValue = record.value
    return [{
      name,
      value: typeof rawValue === 'number' && Number.isFinite(rawValue) ? rawValue : null,
      unit: typeof record.unit === 'string' ? record.unit : undefined,
      status: record.status === 'ok' || record.status === 'warning' || record.status === 'unavailable' || record.status === 'error' ? record.status : undefined,
      warnings: warningMessages(record.warnings),
      presentation: record.presentation && typeof record.presentation === 'object' ? record.presentation as unknown as MetricPresentation : fallbackMetricPresentation(String(record.indicator_id ?? name), name, typeof record.unit === 'string' ? record.unit : undefined),
    }]
  })
}

function assetKeys(assets: unknown): string[] {
  if (!Array.isArray(assets)) return []
  return assets.map((asset, index) => {
    if (!asset || typeof asset !== 'object') return `asset_${index + 1}`
    const source = asset as Record<string, unknown>
    for (const field of ['asset_key', 'product_id', 'code', 'name']) {
      if (typeof source[field] === 'string' && source[field]) return source[field] as string
    }
    return `asset_${index + 1}`
  })
}

function normalizeWeights(value: unknown, assets: unknown): Record<string, number> {
  if (value && typeof value === 'object' && !Array.isArray(value)) {
    return Object.entries(value as Record<string, unknown>).reduce<Record<string, number>>((result, [key, weight]) => {
      if (typeof weight === 'number' && Number.isFinite(weight)) result[key] = weight
      return result
    }, {})
  }
  if (!Array.isArray(value)) return {}
  const keys = assetKeys(assets)
  return value.reduce<Record<string, number>>((result, weight, index) => {
    if (typeof weight === 'number' && Number.isFinite(weight)) result[keys[index] ?? `asset_${index + 1}`] = weight
    else if (weight && typeof weight === 'object') {
      const entry = weight as Record<string, unknown>
      const key = ['asset_key', 'product_id', 'code', 'name'].map((field) => entry[field]).find((item): item is string => typeof item === 'string')
      const numeric = entry.weight ?? entry.value
      if (key && typeof numeric === 'number' && Number.isFinite(numeric)) result[key] = numeric
    }
    return result
  }, {})
}

function normalizeWeightPath(value: unknown, assets: unknown, dates: unknown): PortfolioWeightPoint[] {
  if (!Array.isArray(value)) return []
  return value.flatMap((item, index): PortfolioWeightPoint[] => {
    const source: Record<string, unknown> = item && typeof item === 'object'
      ? item as Record<string, unknown>
      : { weights: item }
    const date = source.effective_date ?? source.date ?? source.decision_date ?? (Array.isArray(dates) ? dates[index] : undefined)
    const weights = normalizeWeights(source.weights ?? source.weight ?? item, assets)
    return typeof date === 'string' && Object.keys(weights).length ? [{ date, weights }] : []
  })
}

function normalizeRun(raw: unknown): PortfolioRun {
  const source = (raw ?? {}) as Record<string, any>
  assertFixedNjitExecution(source.execution, '组合研究运行')
  const regimeConditioning = source.regime_conditioning && typeof source.regime_conditioning === 'object'
    ? source.regime_conditioning as RegimeConditioningResult
    : null
  if (regimeConditioning) assertFixedNjitExecution(regimeConditioning.execution, '组合历史情景条件统计')
  const dates = source.dates ?? []
  const rawWeights = Array.isArray(source.weight_path) ? source.weight_path : Array.isArray(source.weights) ? source.weights : []
  return {
    id: String(source.id ?? ''), target_id: source.target_id ?? null, name: String(source.name ?? source.target_name ?? '组合运行'),
    request: source.request,
    nav: seriesPoints(dates, source.portfolio_nav ?? source.nav),
    drawdown: seriesPoints(dates, source.drawdown),
    metrics: metricList(source.summary_metrics ?? source.metrics ?? source.summary),
    weights: normalizeWeightPath(rawWeights, source.assets ?? source.components, dates),
    contributions: Array.isArray(source.contributions) ? source.contributions : [],
    correlation: source.correlation ?? null, covariance: source.covariance ?? null,
    custom_indicators: portfolioMetricList(source.custom_indicators), rebalances: Array.isArray(source.rebalances) ? source.rebalances : [],
    regime_conditioning: regimeConditioning,
    warnings: warningMessages(source.warnings),
  }
}

function normalizeDiagnosis(raw: unknown): PortfolioDiagnosis {
  const source = (raw ?? {}) as Record<string, any>
  assertFixedNjitExecution(source.execution, '组合研究诊断')
  return {
    summary: metricList(source.summary_metrics ?? source.metrics ?? source.summary), components: Array.isArray(source.components) ? source.components : [],
    custom_indicators: portfolioMetricList(source.custom_indicators), contributions: Array.isArray(source.contributions) ? source.contributions : [],
    contribution_series: Array.isArray(source.contribution_series) ? source.contribution_series : [],
    covariance: source.covariance ?? null, correlation: source.correlation ?? null,
    risk_contributions: Array.isArray(source.risk_contributions) ? source.risk_contributions : [],
    concentration: metricList(source.concentration),
    weight_path: normalizeWeightPath(source.weight_path, source.components, source.dates), rebalances: Array.isArray(source.rebalances) ? source.rebalances : [],
    warnings: warningMessages(source.warnings),
  }
}

export async function searchPortfolioInstruments(query: string, signal?: AbortSignal) {
  const params = new URLSearchParams({ q: query, kind: 'all', page: '1', page_size: '20' })
  const response = await request<{ items: Array<{ code?: string | null; ts_code?: string | null; name?: string | null; instrument_type?: string | null }>; total: number }>(`/api/instruments/search?${params}`, { signal })
  return {
    ...response,
    items: response.items.flatMap((item): PortfolioInstrument[] => {
      const kind = item.instrument_type === 'etf' || item.instrument_type === 'fund' ? item.instrument_type : null
      const productId = item.ts_code ?? item.code
      return kind && productId && item.name ? [{ product_id: productId, kind, name: item.name, code: item.ts_code ?? item.code }] : []
    }),
  }
}

export const listResearchTargets = () => request<{ items: ResearchTarget[] }>('/api/research-targets?kind=portfolio')
export const getResearchTarget = (id: string) => request<ResearchTarget>(`/api/research-targets/${encodeURIComponent(id)}`)
export async function listPortfolioIndicators() {
  const response = await request<{ items: PortfolioIndicatorDefinition[] }>('/api/custom-indicators?context_kind=portfolio&product_kind=portfolio')
  return (response.items ?? []).filter((item) => item.context_kind === 'portfolio')
}
function targetDefinition(input: PortfolioRunRequest) {
  return {
    universe_snapshot_id: input.universe_snapshot_id ?? undefined,
    components: input.constituents.map(({
      kind,
      product_id,
      name,
      asset_class_id,
      asset_class_name,
    }) => ({ kind, product_id, name, asset_class_id, asset_class_name })),
    strategy: {
      type: input.method,
      weights: input.method === 'manual' ? input.constituents.map((item) => (item.weight ?? 0) / 100) : undefined,
      budgets: input.method === 'risk_budget' ? input.constituents.map((item) => (item.risk_budget ?? 0) / 100) : undefined,
      target: input.objective ?? undefined,
      target_return: input.target_return ?? undefined,
      lookback_observations: input.window.mode === 'rolling' ? input.window.observations : undefined,
    },
    constraints: input.constraints,
    rebalance: {
      enabled: input.rebalance.frequency !== 'fixed',
      mode: input.rebalance.frequency,
      transaction_cost_bps: input.rebalance.transaction_cost_bps,
    },
    benchmark: input.benchmark ? { kind: input.benchmark.kind, product_id: input.benchmark.product_id } : undefined,
    alignment: 'strict_intersection',
  }
}

export const createResearchTarget = (input: { name: string; kind?: 'portfolio'; description?: string; definition: PortfolioRunRequest }) =>
  request<ResearchTarget>('/api/research-targets', {
    method: 'POST',
    body: JSON.stringify({ name: input.name, description: input.description ?? '', definition: targetDefinition(input.definition) }),
  })

export async function runPortfolio(targetId: string, input: { as_of?: string | null; start_date?: string | null; historical_regime?: HistoricalRegimeBacktestReference | null } = {}) {
  return normalizeRun(await request<unknown>(`/api/research-targets/${encodeURIComponent(targetId)}/run`, { method: 'POST', body: JSON.stringify(input) }))
}
export async function getPortfolioRun(id: string) { return normalizeRun(await request<unknown>(`/api/portfolio-runs/${encodeURIComponent(id)}`)) }
export async function listPortfolioRuns() {
  const response = await request<{ items: unknown[] }>('/api/portfolio-runs')
  return { items: (response.items ?? []).map(normalizeRun) }
}
export async function diagnosePortfolioRun(id: string, indicatorIds: string[] = []) {
  return normalizeDiagnosis(await request<unknown>(`/api/portfolio-runs/${encodeURIComponent(id)}/diagnose`, {
    method: 'POST',
    body: JSON.stringify({ indicator_ids: indicatorIds }),
  }))
}
export async function runPortfolioScenario(id: string, input: { name: string; start_date: string; end_date: string }) {
  const response = await request<{ name: string; metrics?: PortfolioMetric[]; warnings?: unknown; execution: FixedNjitExecutionAudit }>(`/api/portfolio-runs/${encodeURIComponent(id)}/scenario`, { method: 'POST', body: JSON.stringify(input) })
  assertFixedNjitExecution(response.execution, '组合历史情景')
  return { name: response.name, metrics: metricList(response.metrics), warnings: warningMessages(response.warnings) }
}

export async function downloadPortfolioExport(
  id: string,
  format: 'csv' | 'zip',
  options: { table?: PortfolioExportTable; scenario_start?: string; scenario_end?: string } = {},
) {
  const params = new URLSearchParams({ format })
  if (options.table) params.set('table', options.table)
  if (options.scenario_start) params.set('scenario_start', options.scenario_start)
  if (options.scenario_end) params.set('scenario_end', options.scenario_end)
  const response = await fetch(`/api/portfolio-runs/${encodeURIComponent(id)}/export?${params}`)
  if (!response.ok) throw new PortfolioResearchApiError(response.status, `导出失败（${response.status}）`)
  const blob = await response.blob()
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = `组合研究_${id}.${format}`
  document.body.appendChild(link)
  link.click()
  link.remove()
  URL.revokeObjectURL(url)
}

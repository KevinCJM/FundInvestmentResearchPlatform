import { assertFixedNjitExecution, type FixedNjitExecutionAudit } from '../utils/fixedNjitExecution'

export type TimingValueType = 'series' | 'condition' | 'panel' | 'condition_panel'
export interface TimingPort { name: string; label: string; type: TimingValueType; required?: boolean }
export interface TimingParameter {
  name: string; label: string; type: 'number' | 'integer' | 'string'; default: number | string
  minimum?: number; maximum?: number; options?: Array<{ value: string | number; label: string }>
}
export interface TimingOperator {
  id: string; label: string; description: string; category: string; granularity: string
  inputs: TimingPort[]; outputs: TimingPort[]; parameters: TimingParameter[]
}
export interface TimingNode { id: string; label: string; op: string; inputs: Record<string, string>; parameters: Record<string, number | string> }
export interface TimingExecution {
  take_profit: number; stop_loss: number; max_holding_bars: number; cooldown_bars: number; fee_bps: number; slippage_bps: number
}
export interface TimingTraining {
  mode: 'global' | 'state' | 'month' | 'quarter'; state_refs: string[]
  actions: Array<{ id: string; label: string; entry: string | null }>
  search_space: Array<{ label: string; choices: Array<Array<{ node: string; parameter: string; value: number | string }>> }>
  min_trades: number; confidence: number; risk_penalty: number; min_utility: number; embargo_bars: number
}
export interface TimingAdaptation { source_experiments: string[]; version: 'etf-v1'; preserved: string[]; changed: string[] }
export interface TimingBaskets { market: string[]; category: string[] }
export interface TimingTrainingAudit {
  mode: TimingTraining['mode']; freeze_date: string; fit_end_date: string; embargo_bars: number; min_trades: number
  selection: Array<{ state: string; action_id: string | null; action_label: string; sample_count: number; utility: number | null }>
  candidates: Array<{ id: string; label: string; parameters?: Array<{ node: string; parameter: string; value: number | string }>; states: Array<{ state: string; sample_count: number; mean_return: number | null; win_rate: number | null; utility: number | null; stop_rate: number | null }> }>
  warnings: string[]
}
export interface TimingDefinition {
  name: string; description: string; nodes: TimingNode[]; entry: string; exit: string | null; execution: TimingExecution
  adaptation?: TimingAdaptation; training?: TimingTraining
}
export interface SavedTimingDefinition extends TimingDefinition { id: string; revision: number }
export interface TimingCatalog {
  operators: TimingOperator[]; templates: Array<{ id: string; label: string; description: string; definition: TimingDefinition }>
  limits?: Record<string, number>
}
export interface TimingRequest {
  definition: TimingDefinition; compile_token: string; targets: Array<{ kind: 'etf'; product_id: string }>
  start_date: string; end_date: string; holdout_start: string; walk_forward_splits: number; price_basis: 'hfq' | 'qfq' | 'raw'
  context_baskets?: Partial<TimingBaskets>
}
export type TimingMetrics = Partial<Record<'observations' | 'total_return' | 'buy_hold_return' | 'excess_return' | 'max_drawdown' | 'annualized_return' | 'annualized_volatility' | 'exposure' | 'turnover' | 'cost' | 'trade_count' | 'win_rate' | 'mean_trade_return' | 'median_trade_return' | 'mean_holding_days' | 'profit_factor', number | null>>
export interface TimingPeriod extends TimingMetrics { period: string }
export interface TimingPoint { date: string; close: number | null; nav: number | null; buy_hold_nav: number | null; position: number; action: number | string; reason: string }
export interface TimingTrade {
  signal_date: string; entry_date: string; exit_date: string; entry_price: number; exit_price: number
  net_return: number; holding_days: number; reason: string; max_adverse_excursion: number; max_favorable_excursion: number
}
export interface TimingDiagnostics {
  raw_signal_count: number; known_condition_count: number; active_month_count: number; total_month_count: number
  top_month_signal_share: number | null; positive_month_fraction: number | null; positive_excess_month_fraction: number | null
}
export interface TimingProductResult {
  product_id: string; status: 'ok' | 'error'; error?: string; detail_loaded?: boolean
  summary?: { all: TimingMetrics; in_sample: TimingMetrics; out_of_sample: TimingMetrics }
  yearly?: TimingPeriod[]; monthly?: TimingPeriod[]; walk_forward?: TimingPeriod[]
  diagnostics?: { all: TimingDiagnostics; out_of_sample: TimingDiagnostics }
  training?: TimingTrainingAudit
  signal_quality?: Array<{ horizon: number; signal_count: number; eligible_count: number; win_rate: number | null; mean_return: number | null; median_return: number | null; mean_max_adverse: number | null; mean_max_favorable: number | null; positive_close_fraction: number | null }>
  curve?: TimingPoint[]; trades?: TimingTrade[]; channels?: Array<{ id: string; label: string; type: TimingValueType; values: Array<number | null> }>; warnings?: string[]
}
export interface TimingRunSummary { id: string; name: string; created_at: string; status?: string }
export interface TimingRun extends TimingRunSummary {
  definition_snapshot: TimingDefinition; request_snapshot: Omit<TimingRequest, 'compile_token'>
  execution: FixedNjitExecutionAudit; restrictions: string[]; products: TimingProductResult[]
}
export interface TimingJob { id: string; status: 'queued' | 'running' | 'completed' | 'done' | 'failed'; progress: number; total: number; run_id?: string; error?: string }
export interface TimingRelease { id: string; run_id: string; name?: string; note?: string; usage?: string; created_at?: string; products?: Array<{ product_id: string; status?: string }> }
export interface TimingComparison { items: Array<{ run_id: string; name: string; products: TimingProductResult[] }> }

const prefix = '/api/timing-research'
async function request<T>(path: string, method = 'GET', body?: unknown, signal?: AbortSignal): Promise<T> {
  const response = await fetch(prefix + path, { method, signal, headers: body === undefined ? undefined : { 'Content-Type': 'application/json' }, body: body === undefined ? undefined : JSON.stringify(body) })
  const value = await response.json().catch(() => null)
  if (!response.ok) {
    const detail = value?.detail
    const message = typeof detail === 'string' ? detail : Array.isArray(detail) ? detail.map((item: { msg?: string }) => item.msg).join('；') : detail?.message
    throw new Error(message || `请求失败（${response.status}），请稍后重试。`)
  }
  return value as T
}
async function numerical<T>(path: string, method = 'GET', body?: unknown, signal?: AbortSignal): Promise<T> {
  const value = await request<T & { execution?: unknown }>(path, method, body, signal)
  assertFixedNjitExecution(value.execution, '择时研究')
  return value
}
export const timingApi = {
  catalog: (signal?: AbortSignal) => request<TimingCatalog>('/catalog', 'GET', undefined, signal),
  definitions: (signal?: AbortSignal) => request<{ items: SavedTimingDefinition[] }>('/definitions', 'GET', undefined, signal),
  save: (definition: TimingDefinition, current?: SavedTimingDefinition) => request<SavedTimingDefinition>(current ? `/definitions/${encodeURIComponent(current.id)}` : '/definitions', current ? 'PUT' : 'POST', current ? { ...definition, revision: current.revision } : definition),
  prepare: (definition: TimingDefinition, signal?: AbortSignal) => numerical<{ compile_token: string; definition_hash: string; execution: FixedNjitExecutionAudit }>('/prepare', 'POST', { definition }, signal),
  run: (body: TimingRequest, signal?: AbortSignal) => request<TimingJob>('/runs', 'POST', body, signal),
  job: (id: string, signal?: AbortSignal) => request<TimingJob>(`/jobs/${encodeURIComponent(id)}`, 'GET', undefined, signal),
  runs: (signal?: AbortSignal) => request<{ items: TimingRunSummary[] }>('/runs', 'GET', undefined, signal),
  getRun: (id: string, signal?: AbortSignal) => numerical<TimingRun>(`/runs/${encodeURIComponent(id)}`, 'GET', undefined, signal),
  product: (runId: string, productId: string, signal?: AbortSignal) => numerical<TimingProductResult>(`/runs/${encodeURIComponent(runId)}/products/${encodeURIComponent(productId)}?offset=0&limit=12000`, 'GET', undefined, signal),
  compare: (runIds: string[]) => request<TimingComparison>('/compare', 'POST', { run_ids: runIds }),
  release: (runId: string, note: string) => request<TimingRelease>('/releases', 'POST', { run_id: runId, note }),
  releases: (signal?: AbortSignal) => request<{ items: TimingRelease[] }>('/releases', 'GET', undefined, signal),
  bind: (releaseId: string, context: 'product_research' | 'pre_investment', note: string) => request<{ id: string }>('/bindings', 'POST', { release_id: releaseId, context, note }),
}

export const timingNumber = (value: number | null | undefined, digits = 2) => value == null || !Number.isFinite(value) ? '—' : value.toLocaleString('zh-CN', { maximumFractionDigits: digits })
export const timingPercent = (value: number | null | undefined) => value == null || !Number.isFinite(value) ? '—' : `${(value * 100).toFixed(2)}%`
export const timingReason = (reason: string) => ({ take_profit: '止盈', stop_loss: '止损', max_holding: '达到最长持有期', max_holding_bars: '达到最长持有期', exit_signal: '退出条件触发', exit_rule: '退出条件触发', entry_signal: '入场条件触发', end_of_data: '区间结束，仍持仓', hold: '继续持有', none: '无操作', entry: '买入', exit: '卖出' }[reason] || reason || '—')

export function editableTimingDefinition(definition: TimingDefinition): TimingDefinition {
  return { name: definition.name, description: definition.description, entry: definition.entry, exit: definition.exit, execution: { ...definition.execution }, nodes: definition.nodes.map(node => ({ ...node, inputs: { ...node.inputs }, parameters: { ...node.parameters } })), ...(definition.adaptation ? { adaptation: structuredClone(definition.adaptation) } : {}), ...(definition.training ? { training: structuredClone(definition.training) } : {}) }
}

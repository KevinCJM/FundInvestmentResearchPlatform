import {
  assertFixedNjitExecution,
  type FixedNjitExecutionAudit,
} from '../utils/fixedNjitExecution'

export interface TaaBacktestRequest {
  asset_returns: Array<Record<string, string | number>>
  base_weights: Record<string, number>
  state_tilts: Record<string, Record<string, number>>
  limits: {
    min_weight?: number
    max_weight?: number
    max_abs_tilt?: number
  }
  transaction_cost_bps: number
  confidence_floor: number
  periods_per_year: number
  max_signal_age_days: number
}

export interface TaaPerformanceMetrics {
  total_return: number
  annualized_return: number | null
  annualized_volatility: number | null
  sharpe: number | null
  max_drawdown: number
}

export interface TaaNavPoint {
  date: string
  value: number
}

export interface TaaWeightPoint {
  date: string
  period_start?: string | null
  regime_observation_date: string | null
  regime_effective_date: string | null
  regime_recognized_at?: string | null
  probabilities: Record<string, number>
  allocation_probabilities: Record<string, number>
  confidence: number | null
  probability_source?: string | null
  fallback_to_base: boolean
  fallback_reason: string | null
  tilt_scale: number
  weights: Record<string, number>
  pretrade_weights?: Record<string, number>
  turnover: number
  transaction_cost_amount: number
  transaction_cost_rate?: number
  baseline_turnover?: number
  baseline_transaction_cost_rate?: number
  baseline_transaction_cost_amount?: number
  gross_baseline_return?: number
  baseline_return: number
  gross_taa_return: number
  net_taa_return: number
  gross_excess_return?: number
  state_contributions?: Record<string, number>
}

export interface TaaBacktestResult {
  schema_version: string
  run_id: string
  definition_id: string | null
  definition_revision: number | null
  execution: FixedNjitExecutionAudit & {
    engine?: string
    typed_indicator_dag: boolean
    kernel_version?: string
    kernel_names?: string[]
    compiled_signatures?: string[]
    kernel_fingerprint?: string
    njit_required?: boolean
    note?: string
  }
  gate: {
    passed: boolean
    immutable: boolean
    mode: string
    causal: boolean
    publication_usages: string[]
    publication_ids: string[]
    run_content_hash: string
  }
  timing_policy: {
    return_timestamp: string
    signal_rule: string
    same_day_signal_allowed: boolean
    allocation_source: string
    period_start_policy?: string
    max_signal_age_days?: number
  }
  metrics_policy?: { annualization_periods: number; risk_free_rate: number }
  input_snapshot: {
    observations: number
    start_date: string
    end_date: string
    assets: string[]
    regime_run_content_hash: string
    asset_returns_hash: string
    parameters_hash: string
  }
  baseline: { nav: TaaNavPoint[]; metrics: TaaPerformanceMetrics; policy: string }
  taa: { nav: TaaNavPoint[]; metrics: TaaPerformanceMetrics; policy: string }
  weights: TaaWeightPoint[]
  turnover_and_cost: {
    total_turnover: number
    average_turnover: number
    total_transaction_cost: number
    baseline_total_turnover?: number
    baseline_average_turnover?: number
    baseline_total_transaction_cost?: number
  }
  excess: {
    total_return_difference: number
    relative_total_return: number
    gross_active_return_sum: number
  }
  state_contributions: Array<{
    state_id: string
    probability_weight: number
    gross_excess_return_contribution: number
    active_periods: number
  }>
  fallbacks: { periods: number; reasons: Record<string, number> }
  snapshot_hash: string
}

export class TaaBacktestApiError extends Error {
  constructor(readonly status: number, message: string) {
    super(message)
    this.name = 'TaaBacktestApiError'
  }
}

export async function backtestHistoricalRegimeTaa(
  runId: string,
  payload: TaaBacktestRequest,
): Promise<TaaBacktestResult> {
  const response = await fetch(`/api/historical-regimes/runs/${encodeURIComponent(runId)}/taa-backtest`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  if (!response.ok) {
    let message = `TAA 回测请求失败（${response.status}）`
    try {
      const body = await response.json()
      if (typeof body?.detail?.message === 'string') message = body.detail.message
      else if (typeof body?.detail === 'string') message = body.detail
      else if (Array.isArray(body?.detail)) message = '回测参数未通过校验，请检查权重、状态偏移和收益数据。'
    } catch { /* keep stable fallback */ }
    throw new TaaBacktestApiError(response.status, message)
  }
  const result = await response.json() as TaaBacktestResult
  assertFixedNjitExecution(result.execution, 'TAA 回测')
  return result
}

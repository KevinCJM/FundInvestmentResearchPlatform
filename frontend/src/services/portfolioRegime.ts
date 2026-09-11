import type { FixedNjitExecutionAudit } from '../utils/fixedNjitExecution'

export interface HistoricalRegimeBacktestReference {
  run_id: string
  publication_id: string
}

export interface RegimeConditioningBinding extends HistoricalRegimeBacktestReference {
  schema_version?: string
  publication_usage?: 'formal_backtest' | string
  published_at?: string | null
  definition_id?: string
  definition_revision?: number
  definition_snapshot_hash?: string
  run_content_hash?: string
  mode?: 'realtime' | string
  usage_intent?: string | null
}

export interface RegimeConditionalPerformanceRow {
  state_id: string
  state_label: string
  observations: number
  return_observations: number
  mean_period_return?: number | null
  annualized_return?: number | null
  volatility?: number | null
  max_drawdown?: number | null
  sharpe?: number | null
  positive_rate?: number | null
  return_alignment?: string
}

export interface RegimeConditioningResult {
  schema_version?: string
  binding: RegimeConditioningBinding
  alignment?: {
    rule?: string
    same_period_end_signal_allowed?: boolean
    implicit_latest_version?: boolean
  }
  states?: Array<{ id: string; label: string; color?: string }>
  period_states?: Array<{
    date: string
    period_start: string
    state_id: string
    state_label: string
    regime_observation_date?: string | null
    regime_recognized_at?: string | null
    regime_effective_date?: string | null
    confidence?: number | null
  }>
  period_states_included?: boolean
  period_states_count?: number
  coverage: {
    periods: number
    classified_periods: number
    classified_ratio?: number | null
    state_counts?: Record<string, number>
    state_transitions?: number
  }
  conditional_performance: Record<string, RegimeConditionalPerformanceRow[]>
  execution?: FixedNjitExecutionAudit
}

export type EvidenceStatus = 'passed' | 'failed' | 'unavailable' | 'not_applicable'
export interface AllocationSourceRef { kind: 'saa_policy' | 'taa_decision'; id: string; content_hash: string; implementation_mapping_id?: string | null }
export interface ImplementationProduct {
  kind: 'etf' | 'fund' | 'cash'; product_id: string; asset_class_id: string; weight: number; max_weight: number; current_value: number
  buy_rate: number | null; sell_rate: number | null; fee_source: string; fee_valid_until: string | null
  fee_basis: 'each_side_notional' | 'unknown'; holding_days: number | null; channel: string
  nav_includes_management_fee: boolean | null; settlement_date: string | null
  same_month_settlement_confirmed?: boolean; settlement_terms_source?: string
}
export interface FundingState {
  valuation_at: string; knowledge_cutoff: string; confirmed_investable_value: number; settled_cash: number
  restricted_cash: number; receivables: number; payables: number; elapsed_months: number
  cutoff_phase: 'after_model_month_end' | 'calendar_adapter_required'; transition_cost_in_balance: boolean; evidence: string
  reconciliation: Array<{ occurrence_id: string; status: 'paid' | 'partial' | 'unpaid' | 'unknown'; paid_amount: number; evidence: string }>
  cash_events: Array<{ id: string; date: string | null; kind: 'receipt' | 'payment'; amount: number; evidence: string }>
}
export interface ImplementationCandidate {
  name: string; source: AllocationSourceRef; as_of: string; start_date: string; products: ImplementationProduct[]; state: FundingState
  annual_additional_fee: number; annual_fee_source: string; return_basis_confirmed: boolean; horizon_stationarity_acknowledged: boolean
  future_weight_rule: 'frozen_scalar_proxy' | 'buy_and_hold' | 'monthly_rebalance'; paths: number; search_seed: number; validation_seed: number
  min_validation_r2: number; scenario_release_ids: string[]
  scenario_exposure_release_id?: string | null; future_fee_assumption?: 'constant_declared_rates_sensitivity' | 'unknown'
}
export interface SourceOption extends AllocationSourceRef {
  name: string; as_of: string; expires_on: string; mode: string; strategic_universe_id?: string | null
  assets: Array<{ id: string; role?: string; products: Array<{ kind: 'etf' | 'fund'; product_id: string; name?: string }> }>
  target: Record<string, number>
  funding: null | { origin: string; months: number; plan: { total_capital: number; outside_reserve: number; inflation: number; amount_basis: string }
    occurrences: Array<{ occurrence_id: string; name: string; kind: string; month: number; amount: number; nominal_amount?: number; due_at: string }> }
}
export interface ImplementationCatalog {
  today: string; sources: SourceOption[]
  scenario_options?: {scenarios: Array<{id:string;name:string}>; exposures: Array<{id:string;name:string}>}
  mappings: Array<{ id: string; name: string; definition: { strategic_universe_id: string; valid_until: string; assignments: Array<{strategic_asset_id: string; proxy_asset_id: string}> }; source_snapshot: { assets: SourceOption['assets'] } }>
}
export interface ValidationCheck { check_id: string; title: string; status: EvidenceStatus; reason: string; scope: string; enforcement: string }
export interface ImplementationReport {
  id?: string; content_hash?: string; candidate_hash: string; checks: ValidationCheck[]; research_ready: boolean
  implementation_eligibility: string; independent_simulation: boolean; validation_mode: 'preview' | 'frozen_candidate_validation'; limitations: string[]; checked_at: string
  models: Array<{ model_id: string; name: string; enforced: boolean; expected_return: number | null; volatility: number; total_active_risk: number; implementation_tracking_error: number; status: EvidenceStatus }>
  transition?: { cost: number; post_cost_value: number; gross_traded_notional: number; half_turnover: number }
  historical_replay?: { gross_terminal: number; net_terminal: number; weight_rule: string }
  funding_results?: Array<{model_id: string; name: string; status: EvidenceStatus; remaining_months: number; original_plan_missed_payment: boolean; future_conditional_success_probability: number | null; metrics?: { probability_lower: number; terminal_median: number; payment_failure_probability: number }}>
  cash_calendar?: { status: EvidenceStatus; events: Array<{id: string; date: string; kind: string; amount: number; available_balance: number; cash_gap: number}> }
  product_paths?: {status: EvidenceStatus; assumptions:string[]; results:Array<{model_id:string;name:string;enforced:boolean;status:EvidenceStatus;original_plan_missed_payment:boolean;metrics:{success_probability:number;probability_lower:number;payment_failure_probability:number;terminal_median:number;expected_path_cost:number;first_payment_gap_month:number}}>}
  scenarios?: {results:Array<{impact:{id:string;name:string;summary:{terminal_return:number;max_drawdown:number;pnl_amount:number}}}>}
}
export interface ResearchPackage { id: string; scheme_id: string; name: string; revision: number; stage: string; candidate: ImplementationCandidate; candidate_hash: string; report_id: string | null; validation_report_hash?: string; copied_from_id?: string | null }
export interface PackageView { package: ResearchPackage; report: ImplementationReport | null; history: ResearchPackage[]; current_eligibility: {status: string; reasons: string[]} }
const base = '/api/pre-investment'
async function request<T>(path: string, body?: unknown, signal?: AbortSignal, method = body === undefined ? 'GET' : 'POST'): Promise<T> {
  const response = await fetch(base+path, {method, signal, headers: {'Content-Type':'application/json'}, ...(body === undefined ? {} : {body: JSON.stringify(body)})})
  const payload = await response.json()
  if (!response.ok) {
    const detail = payload.detail
    throw new Error(Array.isArray(detail) ? detail.map(item => `${item.loc?.slice(1).join('.')}: ${item.msg}`).join('；') : detail?.message || detail || response.statusText)
  }
  return payload
}
export const operationKey = () => crypto.randomUUID()
const path = (id: string) => `/packages/${encodeURIComponent(id)}`
export const implementation = {
  catalog: (signal?: AbortSignal) => request<ImplementationCatalog>('/catalog', undefined, signal),
  list: (signal?: AbortSignal) => request<{items: ResearchPackage[]}>('/packages', undefined, signal),
  view: (id: string, signal?: AbortSignal) => request<PackageView>(path(id), undefined, signal),
  preview: (candidate: ImplementationCandidate, signal?: AbortSignal) => request<ImplementationReport>('/preview', candidate, signal),
  optimize: (candidate: ImplementationCandidate, signal?: AbortSignal) => request<{candidate: ImplementationCandidate; validation: ImplementationReport}>('/optimize', candidate, signal),
  save: (candidate: ImplementationCandidate, current: ResearchPackage | null, key: string, copiedFrom: string | null, signal?: AbortSignal) => request<ResearchPackage>(current ? path(current.scheme_id) : '/packages',
    {candidate, expected_revision: current?.revision ?? 0, idempotency_key: key, copied_from_id: copiedFrom}, signal, current ? 'PUT' : 'POST'),
  validate: (item: ResearchPackage, key: string, signal?: AbortSignal) => request<ResearchPackage>(path(item.scheme_id)+'/validate',
    {expected_revision:item.revision, candidate_hash:item.candidate_hash, idempotency_key:key}, signal),
  finalize: (item: ResearchPackage, report: ImplementationReport, review: {reviewer: string; reason: string; review_due_at: string}, key: string, signal?: AbortSignal) => request<ResearchPackage>(path(item.scheme_id)+'/finalize',
    {expected_revision:item.revision, candidate_hash:item.candidate_hash, validation_report_hash:report.content_hash, idempotency_key:key, ...review, accept_research_limits:true}, signal),
  exportUrl: (item: ResearchPackage) => base+path(item.scheme_id)+'/export',
}

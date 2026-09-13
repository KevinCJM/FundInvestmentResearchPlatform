import { assertFixedNjitExecution, type FixedNjitExecutionAudit } from '../utils/fixedNjitExecution'
import type { TaaBaseline, TaaCatalog } from './tacticalAllocation'

export type ObjectiveKind = 'absolute_return' | 'funding_goal' | 'benchmark_relative'
export interface FundingFlow {
  name: string; kind: 'contribution' | 'withdrawal'; amount: number
  first_month: number; last_month: number; every_months: 1 | 3 | 12
}
export interface FundingPlan {
  total_capital: number; outside_reserve: number; terminal_target: number
  amount_basis: 'nominal' | 'real'; inflation: number; annual_fee: number
  required_probability: number; liquidity_months: number; contribution_stress_ratio: number
  drawdown_alert: number; flows: FundingFlow[]
}
export interface BenchmarkPolicy {
  name: string; alloc_name: string; weights: Record<string, number>
  target_excess_return: number; max_tracking_error: number
}
export interface MandateDefinition {
  name: string; as_of: string; review_date: string; currency: string; horizon_years: number
  target_return: number; max_volatility: number; min_liquid_weight: number; max_illiquid_weight: number
  max_tracking_error: number; risk_aversion: number
  rebalance_policy: 'monthly' | 'quarterly' | 'annually' | 'threshold'; rebalance_note: string; note: string
  objective_kind?: ObjectiveKind; funding_plan?: FundingPlan | null; benchmark?: BenchmarkPolicy | null
  boundary_reason?: string; allocation_scope?: string | null
  asset_limits?: PolicyRequest['constraints']; group_limits?: PolicyRequest['group_limits']
}
export interface MandateVersion {
  id: string; name: string; created_at: string; content_hash: string; definition: MandateDefinition
  assessment?: MandateAssessment; assessment_status?: MandateAssessment['status']
  planning_settings?: { simulation_paths: number; seed: number; uncertainty_penalty: number }
}
export type EconomicRole = 'growth' | 'rates' | 'inflation' | 'credit' | 'liquidity' | 'diversifier'
export interface AssetAssumption {
  id: string; role: EconomicRole; liquidity: 'liquid' | 'illiquid'; rationale: string
  annual_return: number; annual_volatility: number; mean_uncertainty: number
}
export interface RiskReferenceRequest {
  alloc_name: string; as_of: string; start_date: string; end_date: string
  shrinkage: number; periods_per_year: number
}
export interface CmaDefinition {
  name: string; alloc_name: string; as_of: string; currency: string; horizon_years: number
  return_basis: 'annual_arithmetic_total_return'; source: string; basis_confirmed: boolean
  assets: AssetAssumption[]; correlation: number[][]
  risk_origin: 'manual' | 'historical_reference'
  risk_reference: RiskReferenceRequest | null; risk_reference_hash: string | null
}
export type CmaDraft = Omit<CmaDefinition, 'assets'> & {
  assets: Array<Omit<AssetAssumption, 'role' | 'liquidity'> & { role: EconomicRole | ''; liquidity: 'liquid' | 'illiquid' | '' }>
}
export function completeCma(value: CmaDraft): value is CmaDefinition {
  return Boolean(value.name.trim() && value.source.trim().length >= 3 && value.as_of && value.basis_confirmed
    && value.assets.length && value.assets.every(asset => asset.role && asset.liquidity && asset.rationale.trim().length >= 3
      && [asset.annual_return, asset.annual_volatility, asset.mean_uncertainty].every(Number.isFinite)
      && asset.annual_volatility > 0 && asset.mean_uncertainty >= 0)
    && value.correlation.length === value.assets.length && value.correlation.every(row => row.length === value.assets.length && row.every(Number.isFinite)))
}
export interface RiskReference {
  preview_hash: string; request: RiskReferenceRequest; assets: string[]; volatility: number[]
  correlation: number[][]; historical_mean: number[]; observations: number; source_hash: string
  lineage: { start_date: string; end_date: string }; warnings: string[]; execution: FixedNjitExecutionAudit
}
export interface CmaPreview {
  preview_hash: string; definition: CmaDefinition; source_snapshot: Omit<TaaBaseline, 'id' | 'created_at' | 'content_hash'>
  covariance: number[][]; warnings: string[]; execution: FixedNjitExecutionAudit
}
export interface CmaVersion extends CmaPreview {
  id: string; name: string; created_at: string; content_hash: string
}
export interface PolicyRequest {
  mandate_id: string; cma_id: string
  constraints: Record<string, { min_weight: number; max_weight: number; max_abs_tilt: number }>
  group_limits: Array<{ id: string; assets: string[]; lo: number; hi: number }>
  uncertainty_penalty: number; candidate_count: number; seed: number
}
export interface FundingMetrics {
  success_probability: number; probability_lower: number; probability_upper: number
  payment_failure_probability: number; terminal_p05: number; terminal_median: number; terminal_p95: number
  expected_terminal_shortfall: number; expected_unpaid_payments: number
  market_drawdown_p95: number; drawdown_alert_probability: number
  required_initial_capital: number; additional_initial_capital: number
  success_with_10pct_more_capital: number; success_with_10pct_lower_target: number
  capital_gate_status?: 'solved' | 'insufficient_paths'
  gate_required_initial_capital?: number | null; gate_additional_initial_capital?: number | null
  gate_success_probability?: number | null; gate_probability_lower?: number | null; gate_probability_upper?: number | null
  annual_fan?: Array<{ year: number; p05: number; median: number; p95: number }>
}
export interface FundingSummary {
  investable_capital: number; nominal_terminal_target: number; total_contributions: number; total_withdrawals: number
  required_liquid_capital: number; required_liquid_weight: number; required_effective_return: number | null
  root_status: 'solved' | 'at_lower_bound' | 'above_search_bound'; currency: string; liquidity_months: number
  liquidity_payment_buffer?: number; liquidity_payment_buffer_ratio?: number; liquidity_shortfall_capital?: number
  monthly_cashflows: Array<{ month: number; contribution: number; withdrawal: number }>
}
export interface MandateStudyRequest {
  definition: MandateDefinition; cma_id: string | null; simulation_paths: number; seed: number; uncertainty_penalty: number
}
export interface MandateAssessment {
  preview_hash: string; request: MandateStudyRequest; definition: MandateDefinition
  funding: FundingSummary | null; candidates: PolicyCandidate[]; status: 'inputs_only' | 'diagnosed' | 'needs_revision'
  cma: { id: string; name: string; as_of: string; content_hash: string } | null
  blockers: string[]; warnings: string[]; execution: FixedNjitExecutionAudit
  funding_model?: { version: string; paths: number; seed: number; frequency: string }
  funding_execution?: FixedNjitExecutionAudit
}
export interface PolicyCandidate {
  id: 'minimum-risk' | 'nominal-utility' | 'robust-utility' | 'maximum-return'
  name: string; weights: Record<string, number>; risk_contributions: Record<string, number | null>
  goal_check?: { within_limits: boolean; threshold: number; gate_basis: string; central: FundingMetrics
    conservative: FundingMetrics | null; stress_contribution_ratio: number }
  benchmark_check?: { name: string; expected_excess_return: number; tracking_error: number
    target_excess_return: number; max_tracking_error: number }
  metrics: { expected_return: number; volatility: number; conservative_return: number; nominal_utility: number; robust_utility: number }
}
export interface PolicyPreview {
  preview_hash: string; request: PolicyRequest; mandate: MandateDefinition; assumptions: CmaDefinition
  source_snapshot: Omit<TaaBaseline, 'id' | 'created_at' | 'content_hash'>; candidates: PolicyCandidate[]; accepted_candidates: number
  warnings: string[]; execution: FixedNjitExecutionAudit
  funding?: FundingSummary | null; current_application_eligible?: boolean
  funding_model?: MandateAssessment['funding_model'] | null
  funding_execution?: FixedNjitExecutionAudit
}
export interface StrategicCatalog {
  allocations: TaaCatalog['allocations']; mandates: MandateVersion[]
  assumptions: Array<{ id: string; name: string; alloc_name: string; as_of: string; currency: string; horizon_years: number }>
  policies: Array<{ id: string; name: string; as_of: string; alloc_name: string }>
}

/** Display-only percent conversion; the saved fractional assumption is unchanged. */
export const percentInputValue = (value: number): number => Number.isFinite(value) ? Number((value * 100).toPrecision(12)) : NaN

const root = '/api/strategic-allocation'
async function request<T>(path: string, body?: unknown, signal?: AbortSignal): Promise<T> {
  const response = await fetch(`${root}${path}`, {
    method: body === undefined ? 'GET' : 'POST', signal,
    headers: { 'Content-Type': 'application/json' }, ...(body === undefined ? {} : { body: JSON.stringify(body) }),
  })
  const result = await response.json().catch(() => { throw new Error('资产配置服务没有返回可读取的数据，请重试。') })
  if (!response.ok) {
    const detail = result?.detail
    throw new Error(typeof detail === 'string' ? detail : Array.isArray(detail)
      ? detail.map(item => `${item.loc?.slice(1).join('.') ?? ''}：${item.msg ?? '输入无效'}`).join('；')
      : detail?.message ?? '资产配置请求失败，请检查输入后重试。')
  }
  return result as T
}
const verified = <T extends { execution: FixedNjitExecutionAudit }>(value: T): T => {
  assertFixedNjitExecution(value.execution, '长期配置研究')
  return value
}
const probability = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value) && value >= 0 && value <= 1
const missingDiagnosis = () => new Error('资金目标诊断缺失或口径不一致，已停止展示通过状态，请重新运行。')

function checkFundingMetrics(metrics: FundingMetrics, threshold: number) {
  if (!metrics) throw missingDiagnosis()
  const probabilities = ['success_probability', 'probability_lower', 'probability_upper', 'payment_failure_probability',
    'market_drawdown_p95', 'drawdown_alert_probability', 'success_with_10pct_more_capital', 'success_with_10pct_lower_target'] as const
  const amounts = ['terminal_p05', 'terminal_median', 'terminal_p95', 'expected_terminal_shortfall', 'expected_unpaid_payments',
    'required_initial_capital', 'additional_initial_capital'] as const
  if (probabilities.some(key => !probability(metrics[key])) || amounts.some(key => !Number.isFinite(metrics[key]) || metrics[key] < 0)
    || metrics.probability_lower > metrics.success_probability + 1e-12 || metrics.success_probability > metrics.probability_upper + 1e-12
    || metrics.terminal_p05 > metrics.terminal_median || metrics.terminal_median > metrics.terminal_p95) throw missingDiagnosis()
  if (metrics.annual_fan && (!Array.isArray(metrics.annual_fan) || metrics.annual_fan.some((row, index, rows) =>
    !Number.isInteger(row.year) || row.year < 0 || (index > 0 && row.year <= rows[index - 1].year)
    || ![row.p05, row.median, row.p95].every(Number.isFinite) || row.p05 < 0 || row.p05 > row.median || row.median > row.p95))) throw missingDiagnosis()
  if (metrics.capital_gate_status === 'solved') {
    if (!Number.isFinite(metrics.gate_required_initial_capital) || !Number.isFinite(metrics.gate_additional_initial_capital)
      || metrics.gate_required_initial_capital! < 0 || metrics.gate_additional_initial_capital! < 0
      || !probability(metrics.gate_success_probability) || !probability(metrics.gate_probability_lower) || !probability(metrics.gate_probability_upper)
      || metrics.gate_probability_lower < threshold || metrics.gate_probability_lower > metrics.gate_success_probability + 1e-12
      || metrics.gate_success_probability > metrics.gate_probability_upper + 1e-12) throw missingDiagnosis()
  } else if (metrics.capital_gate_status === 'insufficient_paths') {
    if ([metrics.gate_required_initial_capital, metrics.gate_additional_initial_capital, metrics.gate_success_probability,
      metrics.gate_probability_lower, metrics.gate_probability_upper].some(value => value !== null)) throw missingDiagnosis()
  } else if (metrics.capital_gate_status !== undefined) throw missingDiagnosis()
}

function checkGoalCandidates(definition: MandateDefinition, candidates: PolicyCandidate[]) {
  if (!Array.isArray(candidates)) throw missingDiagnosis()
  if (!definition.funding_plan) return
  const threshold = definition.funding_plan.required_probability
  for (const candidate of candidates) {
    const goal = candidate.goal_check
    if (!goal || typeof goal.within_limits !== 'boolean' || goal.threshold !== threshold || goal.gate_basis !== 'wilson_95pct_lower_bound') throw missingDiagnosis()
    checkFundingMetrics(goal.central, threshold)
    if (goal.conservative !== null) checkFundingMetrics(goal.conservative, threshold)
    if (goal.within_limits !== (goal.central.probability_lower >= threshold)) throw missingDiagnosis()
  }
}

function checkedAssessment(value: MandateAssessment): MandateAssessment {
  verified(value)
  if (!value.definition || !value.request || !Array.isArray(value.blockers) || !Array.isArray(value.warnings)
    || !['inputs_only', 'diagnosed', 'needs_revision'].includes(value.status)) throw missingDiagnosis()
  checkGoalCandidates(value.definition, value.candidates)
  if ((value.status === 'diagnosed' && (!value.cma || !value.candidates.length))
    || (value.status === 'inputs_only' && (value.cma || value.candidates.length))
    || (value.cma && value.cma.id !== value.request.cma_id)) throw missingDiagnosis()
  if (value.definition.funding_plan && (!value.funding || !Number.isFinite(value.funding.investable_capital) || value.funding.investable_capital <= 0)) throw missingDiagnosis()
  if (value.definition.funding_plan && value.candidates.length) {
    if (!value.funding_execution || !value.funding_model) throw missingDiagnosis()
    assertFixedNjitExecution(value.funding_execution, '资金目标诊断')
    const passing = value.candidates.some(candidate => candidate.goal_check?.within_limits === true)
    if ((value.status === 'diagnosed') !== passing) throw missingDiagnosis()
  }
  return value
}

export const getStrategicCatalog = (signal?: AbortSignal) => request<StrategicCatalog>('/catalog', undefined, signal)
export const previewMandate = async (body: MandateStudyRequest, signal?: AbortSignal) => checkedAssessment(await request<MandateAssessment>('/mandates/preview', body, signal))
export const confirmMandate = async (body: MandateStudyRequest, previewHash: string, signal?: AbortSignal) => {
  const value = await request<MandateVersion>('/mandates/confirm', { request: body, preview_hash: previewHash, acknowledge_limits: true }, signal)
  if (!value.assessment?.execution) throw new Error('目标版本缺少对应的诊断记录，已停止交接。')
  checkedAssessment(value.assessment)
  return value
}
export const getMandate = async (id: string, signal?: AbortSignal) => {
  const value = await request<MandateVersion>(`/mandates/${encodeURIComponent(id)}`, undefined, signal)
  if (value.assessment?.preview_hash) checkedAssessment(value.assessment)
  return value
}
export const riskReference = async (body: RiskReferenceRequest, signal?: AbortSignal) => verified(await request<RiskReference>('/risk-reference', body, signal))
export const previewCma = async (body: CmaDefinition, signal?: AbortSignal) => verified(await request<CmaPreview>('/cma/preview', body, signal))
export const publishCma = async (body: CmaDefinition, previewHash: string, signal?: AbortSignal) => verified(await request<CmaVersion>('/cma', { request: body, preview_hash: previewHash }, signal))
export const getCma = async (id: string, signal?: AbortSignal) => verified(await request<CmaVersion>(`/cma/${encodeURIComponent(id)}`, undefined, signal))
export const previewPolicy = async (body: PolicyRequest, signal?: AbortSignal) => {
  const value = verified(await request<PolicyPreview>('/policy/preview', body, signal))
  checkGoalCandidates(value.mandate, value.candidates)
  if (value.mandate.funding_plan) {
    if (!value.funding_execution) throw missingDiagnosis()
    assertFixedNjitExecution(value.funding_execution, '政策资金目标诊断')
  }
  return value
}
export const publishPolicy = async (body: PolicyRequest, hash: string, candidate: PolicyCandidate['id'], name: string, reason: string, signal?: AbortSignal) => {
  const result = await request<TaaBaseline>('/policies', { request: body, preview_hash: hash, candidate_id: candidate, name, reason }, signal)
  if (!result.policy) throw new Error('返回的版本缺少政策与目标引用，已停止交接。')
  assertFixedNjitExecution(result.policy.execution, '政策采纳')
  return result
}

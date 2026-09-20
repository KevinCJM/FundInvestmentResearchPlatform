import { cmaModelInputError, type CmaModelRequest } from './cmaModelTypes'
import type { CmaRequest as WireCmaRequest, PolicyRequest as WirePolicyRequest, PolicyCmaRef } from './ltcmaContract.generated'
import { cashSuccessRequired, hasCashBudget, fundingThreshold, checkReferenceAssessment, type CashBudget, type BoundaryPolicy, type CapitalTarget, type CashProtection, type RiskAuthorization, type RiskDecision, type ReferenceDiagnosis } from './mandateTypes'
export type { CashBudget, BoundaryPolicy, CapitalTarget, CashProtection, RiskAuthorization } from './mandateTypes'
import type { InstitutionalContext, InstitutionalDiagnostics } from "./institutionalContext"
import type { UniverseVersion, MappingVersion } from "./strategicScope"
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
  target_excess_return: number; max_tracking_error: number; source?: 'explicit' | 'risk_scale_reference'
}
export interface MandateDefinition {
  schema_version?: '1.0' | '2.0'
  cash_budget?: CashBudget | null; funding_target?: CapitalTarget | null; cash_protection?: CashProtection | null
  boundary_policy?: BoundaryPolicy | null; risk_authorization?: RiskAuthorization | null
  name: string; as_of: string; review_date: string | null; currency: string; horizon_years: number
  target_return: number; target_excess_return: number; min_cash_weight: number
  max_volatility: number | null; min_liquid_weight: number; max_illiquid_weight: number
  max_tracking_error: number; risk_aversion: number
  rebalance_policy: 'monthly' | 'quarterly' | 'annually' | 'threshold'; rebalance_note: string; note: string
  objective_kind?: ObjectiveKind; funding_plan?: FundingPlan | null; benchmark?: BenchmarkPolicy | null
  stated_benchmark?: string
  institutional_context?: InstitutionalContext | null; strategic_universe_id?: string | null
  boundary_reason?: string; allocation_scope?: string | null
  boundary_policy_hash?: string; effective_cash_reserve_weight?: number; risk_reference_valid_until?: string | null
  effective_target_return?: number | null
  asset_limits?: PolicyRequest['constraints']; group_limits?: PolicyRequest['group_limits']
}
export interface MandateVersion {
  id: string; name: string; created_at: string; content_hash: string; definition: MandateDefinition
  assessment?: MandateAssessment; assessment_status?: MandateAssessment['status']
  planning_settings?: { simulation_paths: number; seed: number; validation_seed?: number; uncertainty_penalty: number }
}
export type EconomicRole = 'growth' | 'rates' | 'inflation' | 'credit' | 'liquidity' | 'diversifier'
export interface AssetAssumption {
  id: string; role: EconomicRole; liquidity: 'liquid' | 'illiquid'; rationale: string
  annual_return?: number | null; annual_volatility?: number | null; mean_uncertainty: number
}
export interface RiskReferenceRequest {
  alloc_name: string; as_of: string; start_date: string; end_date: string
  shrinkage: number; periods_per_year: number
}
export interface CmaDefinition {
  schema_version?: WireCmaRequest['schema_version']; moment_semantics?: WireCmaRequest['moment_semantics']
  fee_basis?: WireCmaRequest['fee_basis']; fx_hedging_basis?: WireCmaRequest['fx_hedging_basis']
  name: string; alloc_name: string | null; strategic_universe_id?: string | null; implementation_mapping_id?: string | null
  as_of: string; currency: string; horizon_years: number
  return_basis: 'annual_arithmetic_total_return'; source: string; basis_confirmed: boolean
  assets: AssetAssumption[]; correlation?: number[][] | null; model?: CmaModelRequest | null
  risk_origin: 'manual' | 'historical_reference'
  risk_reference: RiskReferenceRequest | null; risk_reference_hash: string | null
}
export type CmaDraft = Omit<CmaDefinition, 'assets' | 'correlation'> & {
  correlation: number[][]
  assets: Array<Omit<AssetAssumption, 'role' | 'liquidity' | 'annual_return' | 'annual_volatility'> & {
    role: EconomicRole | ''; liquidity: 'liquid' | 'illiquid' | ''; annual_return: number; annual_volatility: number
  }>
}
export function cmaDraftFromDefinition(value: CmaDefinition): CmaDraft {
  return { ...value, assets: value.assets.map(a => ({ ...a, annual_return: a.annual_return ?? NaN, annual_volatility: a.annual_volatility ?? NaN })),
    correlation: value.correlation ?? value.assets.map((_, i) => value.assets.map((__, j) => i === j ? 1 : NaN)) }
}
export function completeCma(value: CmaDraft | CmaDefinition): value is CmaDraft & CmaDefinition {
  const model = value.model
  const validModel = !model || !cmaModelInputError(model) && model.as_of === value.as_of && model.currency === value.currency
    && JSON.stringify(model.asset_ids) === JSON.stringify(value.assets.map(a => a.id))
    && (model.return_basis ?? 'annual_arithmetic_total_return') === value.return_basis
    && value.risk_origin === 'manual' && !value.risk_reference && !value.risk_reference_hash
  return Boolean(value.name.trim() && value.source.trim().length >= 3 && value.as_of && value.basis_confirmed && validModel
    && value.assets.length && value.assets.every(asset => asset.role && asset.liquidity && asset.rationale.trim().length >= 3
      && Number.isFinite(asset.mean_uncertainty) && asset.mean_uncertainty >= 0 && asset.mean_uncertainty <= 1
      && (model || Number.isFinite(asset.annual_return) && Number.isFinite(asset.annual_volatility)
        && typeof asset.annual_return === 'number' && typeof asset.annual_volatility === 'number' && asset.annual_return >= -.5 && asset.annual_return <= 2 && (asset.annual_volatility > 0 || value.schema_version === '2.0' && asset.annual_volatility === 0 && asset.role === 'liquidity' && asset.liquidity === 'liquid') && asset.annual_volatility <= 3))
    && (model || value.correlation && value.correlation.length === value.assets.length && value.correlation.every(row => row.length === value.assets.length && row.every(Number.isFinite))))
}
/** Model inputs supply the numbers; no fake manual values are sent to the API. */
export function cmaRequest(value: CmaDefinition): CmaDefinition {
  return !value.model ? value : { ...value, correlation: undefined,
    assets: value.assets.map(({ annual_return: _mean, annual_volatility: _risk, ...asset }) => asset) }
}
export interface CmaModelResult {
  asset_ids: string[]; method: CmaModelRequest['method']; definition: CmaModelRequest
  effective_returns: number[]; effective_covariance: number[][]; posterior_mean_covariance: number[][] | null
  mean_uncertainty?: number[]
  mean_estimation_covariance?: number[][]
  content_hash: string; execution: FixedNjitExecutionAudit
  model_audit: { limitations: string[]; [key: string]: unknown }
}
export interface RiskReference {
  preview_hash: string; request: RiskReferenceRequest; assets: string[]; volatility: number[]
  correlation: number[][]; historical_mean: number[]; observations: number; source_hash: string
  lineage: { start_date: string; end_date: string }; warnings: string[]; execution: FixedNjitExecutionAudit
}
export type StrategicBaseline = Omit<TaaBaseline, 'alloc_name'> & {
  alloc_name: string | null
  implementation_status?: 'complete' | 'incomplete'; implementation_gaps?: string[]
}
export type StrategicSourceSnapshot = Omit<StrategicBaseline, 'id' | 'created_at' | 'content_hash'>
export interface CmaPreview {
  semantics?: Record<string, unknown>
  preview_hash: string; definition: CmaDefinition; source_snapshot: StrategicSourceSnapshot
  covariance: number[][]; warnings: string[]; execution: FixedNjitExecutionAudit
  effective_assumptions?: CmaDefinition; effective_returns?: number[]; effective_covariance?: number[][]; model_result?: CmaModelResult
}
export interface CmaVersion extends CmaPreview {
  id: string; name: string; created_at: string; content_hash: string
}
export type PolicyRequest = Omit<WirePolicyRequest, 'constraints' | 'group_limits' | 'uncertainty_penalty' | 'candidate_count' | 'seed'> & {
  constraints: Record<string, { min_weight: number; max_weight: number; max_abs_tilt: number }>
  group_limits: Array<{ id: string; assets: string[]; lo: number; hi: number }>
  uncertainty_penalty: number; candidate_count: number; seed: number
}
export type CmaReference = PolicyCmaRef
export interface MultiCmaEvidence {
  mode: 'parameter_average' | 'compatible_all_models'; aggregation_semantics: 'parameter_average' | 'all_models_required'; refs: CmaReference[]
  primary_evaluation_spec?: 'each_frozen_source_model'; effective_moments_role?: 'display_reference_only'
  sources: Array<CmaReference & { name: string; as_of: string }>
  effective_returns: number[]; effective_covariance: number[][]; effective_mean_uncertainty: number[]
  model_disagreement: number[][]; uncertainty_status: 'not_jointly_calibrated'; content_hash: string
}
export interface CrossModelResult {
  cma_id: string; cma_hash: string; name: string; weight?: number | null
  metrics: PolicyCandidate['metrics']; risk_contributions: PolicyCandidate['risk_contributions']
  goal_check?: PolicyCandidate['goal_check'] | null; benchmark_check?: PolicyCandidate['benchmark_check'] | null
  within_limits: boolean; violations: string[]
  expected_tracking_error?: number | null; diagnostic_scope?: 'strategic_moments_and_funding' | 'tactical_moments_only'
}
export interface FundingMetrics {
  success_probability: number; probability_lower: number; probability_upper: number
  payment_failure_probability: number; terminal_p05: number; terminal_median: number; terminal_p95: number
  expected_terminal_shortfall: number; expected_unpaid_payments: number
  market_drawdown_p95: number; drawdown_alert_probability: number | null
  required_initial_capital: number; additional_initial_capital: number
  success_with_10pct_more_capital: number; success_with_10pct_lower_target: number
  capital_gate_status?: 'solved' | 'insufficient_paths'
  gate_required_initial_capital?: number | null; gate_additional_initial_capital?: number | null
  gate_success_probability?: number | null; gate_probability_lower?: number | null; gate_probability_upper?: number | null
  annual_fan?: Array<{ year: number; p05: number; median: number; p95: number }>
}
export interface FundingSummary {
  investable_capital: number; nominal_terminal_target: number | null; total_contributions: number; total_withdrawals: number
  required_liquid_capital: number; required_liquid_weight: number; required_effective_return: number | null
  root_status: 'solved' | 'at_lower_bound' | 'above_search_bound' | 'not_applicable'; currency: string; liquidity_months: number
  cashflow_required_return?: number | null
  cashflow_required_return_status?: 'solved' | 'at_lower_bound' | 'above_search_bound'
  liquidity_payment_buffer?: number; liquidity_payment_buffer_ratio?: number; liquidity_shortfall_capital?: number
  monthly_cashflows: Array<{ month: number; contribution: number; withdrawal: number }>
}
export interface MandateStudyRequest {
  definition: MandateDefinition; cma_id: string | null; simulation_paths: number; seed: number; uncertainty_penalty: number
  validation_seed?: number
}
export interface MandateAssessment {
  risk_decision?: RiskDecision; reference_diagnosis?: ReferenceDiagnosis
  diagnosis_scope?: 'universal_reference' | 'actual_cma'
  preview_hash: string; request: MandateStudyRequest; definition: MandateDefinition
  funding: FundingSummary | null; candidates: PolicyCandidate[]; status: 'inputs_only' | 'diagnosed' | 'needs_revision'
  cma: { id: string; name: string; as_of: string; content_hash: string } | null
  institutional_diagnostics?: InstitutionalDiagnostics | null
  blockers: string[]; warnings: string[]; execution: FixedNjitExecutionAudit
  funding_model?: { version: string; paths: number; seed: number; frequency: string }
  funding_execution?: FixedNjitExecutionAudit
}
/** Deterministic cash-flow arithmetic for the input page; no CMA, no simulation. */
export interface MandateFundingEcho {
  funding: FundingSummary | null; effective_target_return: number | null; execution: FixedNjitExecutionAudit
}
export interface PolicyCandidate {
  cross_model_results?: CrossModelResult[]
  id: 'minimum-risk' | 'nominal-utility' | 'robust-utility' | 'maximum-return' | 'risk-budget' | 'compatible'
  all_models_pass?: boolean; metric_basis?: 'worst_per_metric_not_one_distribution'; solver?: CompatibilitySolver
  available?: boolean; unavailable_reason?: string | null; risk_budget?: Record<string, number>; risk_budget_distance?: number; distance_basis?: string
  name: string; weights: Record<string, number>; risk_contributions: Record<string, number | null>
  goal_check?: { within_limits: boolean; threshold: number; gate_basis: string; central: FundingMetrics
    conservative: FundingMetrics | null; stress_contribution_ratio: number }
  benchmark_check?: { name: string; expected_excess_return: number; tracking_error: number
    target_excess_return: number; max_tracking_error: number }
  metrics: { expected_return: number; volatility: number; conservative_return: number; nominal_utility: number; robust_utility: number }
}
export interface UnavailableRiskBudgetCandidate {
  id: 'risk-budget'; name: string; available: false; unavailable_reason: string
  weights: Record<string, never>; risk_contributions: Record<string, never>
  risk_budget: Record<string, number>; risk_budget_distance: null
  metrics: Record<keyof PolicyCandidate['metrics'], null>
}
export type UnavailablePolicyCandidate = UnavailableRiskBudgetCandidate | (PolicyCandidate & { id: 'compatible'; available: false; unavailable_reason: string })
export interface CompatibilitySolver {
  status: 'converged' | 'verified_feasible' | 'infeasible' | 'phase_one_unresolved' | 'time_budget' | 'iteration_limit' | 'master_iteration_limit' | 'numerical_failure' | 'cut_budget'
  objective_value: number | null; lower_bound: number | null; objective_gap: number | null; phase_one_lower_bound: number | null
  iterations: number; support_cuts: number; blocked_by_anchor?: string
}
export interface CompatibilityEvidence {
  objective: 'minimax_regret' | 'maximin_return'; regret_basis: 'bounded_continuous_anchor_optima' | 'approximate_regret'
  gate: 'all_frozen_models'; joint_solver: CompatibilitySolver
  funding_search_domain: 'one_joint_candidate_with_anchor_cross_diagnostics'
  anchors: Array<{ cma_id: string; name: string; solver: CompatibilitySolver; weights: Record<string, number> | null;
    reference_value: number | null; reference_upper_bound: number | null; cross_model_results: CrossModelResult[] }>
  limitations: string[]
}
export interface MeanUncertaintyEvidence {
  set: 'ellipsoidal'; confidence: '68' | '90' | '95'; kappa: number; dimension: number
  calibration: string; warnings: string[]; content_hash: string
}
export interface PolicyPreview {
  uncertainty_model?: MeanUncertaintyEvidence
  multi_cma?: MultiCmaEvidence
  compatibility?: CompatibilityEvidence
  unavailable_candidates?: UnavailablePolicyCandidate[]
  preview_hash: string; request: PolicyRequest; mandate: MandateDefinition; assumptions: CmaDefinition
  source_snapshot: StrategicSourceSnapshot; candidates: PolicyCandidate[]; accepted_candidates: number
  warnings: string[]; execution: FixedNjitExecutionAudit
  funding?: FundingSummary | null; current_application_eligible?: boolean; application_blockers?: string[]
  funding_model?: MandateAssessment['funding_model'] | null
  funding_execution?: FixedNjitExecutionAudit
}
export interface StrategicCatalog {
  allocations: TaaCatalog['allocations']; mandates: MandateVersion[]
  strategic_universes?: Pick<UniverseVersion, 'id' | 'name' | 'created_at' | 'content_hash' | 'definition'>[]
  implementation_maps?: Pick<MappingVersion, 'id' | 'name' | 'created_at' | 'content_hash' | 'definition' | 'implementation_status' | 'implementation_gaps'>[]
  assumptions: Array<{ id: string; name: string; schema_version?: '1.0' | '2.0'; retired?: boolean; alloc_name: string | null; strategic_universe_id?: string | null; implementation_mapping_id?: string | null; as_of: string; currency: string; horizon_years: number }>
  policies: Array<{ id: string; name: string; as_of: string; alloc_name: string | null }>
}

/** Display-only percent conversion; the saved fractional assumption is unchanged. */
export const percentInputValue = (value: number | null | undefined): number => typeof value === 'number' && Number.isFinite(value) ? Number((value * 100).toPrecision(12)) : NaN

const root = '/api/strategic-allocation'
async function request<T>(path: string, body?: unknown, signal?: AbortSignal, method?: 'GET' | 'POST' | 'PATCH' | 'DELETE'): Promise<T> {
  const resolvedMethod = method ?? (body === undefined ? 'GET' : 'POST')
  const response = await fetch(`${root}${path}`, {
    method: resolvedMethod, signal,
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

function checkFundingMetrics(metrics: FundingMetrics, threshold: number, allowMissingAlert = false) {
  if (!metrics) throw missingDiagnosis()
  const probabilities = ['success_probability', 'probability_lower', 'probability_upper', 'payment_failure_probability',
    'market_drawdown_p95', 'drawdown_alert_probability', 'success_with_10pct_more_capital', 'success_with_10pct_lower_target'] as const
  const amounts = ['terminal_p05', 'terminal_median', 'terminal_p95', 'expected_terminal_shortfall', 'expected_unpaid_payments',
    'required_initial_capital', 'additional_initial_capital'] as const
  if (probabilities.some(key => !(allowMissingAlert && key === 'drawdown_alert_probability' && metrics[key] === null) && !probability(metrics[key])) || amounts.some(key => !Number.isFinite(metrics[key]) || metrics[key] < 0)
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
  if (!cashSuccessRequired(definition)) return
  const threshold = fundingThreshold(definition)
  if (typeof threshold !== 'number') throw missingDiagnosis()
  for (const candidate of candidates) {
    const goal = candidate.goal_check
    if (!goal || typeof goal.within_limits !== 'boolean' || goal.threshold !== threshold || goal.gate_basis !== 'wilson_95pct_lower_bound') throw missingDiagnosis()
    checkFundingMetrics(goal.central, threshold, definition.schema_version === '2.0')
    if (goal.conservative !== null) checkFundingMetrics(goal.conservative, threshold, definition.schema_version === '2.0')
    if (goal.within_limits !== (goal.central.probability_lower >= threshold)) throw missingDiagnosis()
  }
}

function checkedAssessment(value: MandateAssessment): MandateAssessment {
  verified(value)
  if (value.definition?.institutional_context) {
    const institutional = value.institutional_diagnostics
    if (!institutional || institutional.automated_compliance !== 'not_modelled' || institutional.independent_approval !== false) throw new Error('机构诊断缺少明确的人工核验边界。')
    assertFixedNjitExecution(institutional.execution, '经济状况诊断')
  }
  if (!value.definition || !value.request || !Array.isArray(value.blockers) || !Array.isArray(value.warnings)
    || !['inputs_only', 'diagnosed', 'needs_revision'].includes(value.status)) throw missingDiagnosis()
  checkGoalCandidates(value.definition, value.candidates)
  checkReferenceAssessment(value, checkFundingMetrics)
  if ((value.status === 'diagnosed' && (!value.cma || !value.candidates.length) && value.reference_diagnosis?.status !== 'validated')
    || (value.status === 'inputs_only' && (value.cma || value.candidates.length))
    || (value.cma && value.cma.id !== value.request.cma_id)) throw missingDiagnosis()
  if (hasCashBudget(value.definition) && (!value.funding || !Number.isFinite(value.funding.investable_capital) || value.funding.investable_capital <= 0)) throw missingDiagnosis()
  if (cashSuccessRequired(value.definition) && value.candidates.length) {
    if (!value.funding_execution || !value.funding_model) throw missingDiagnosis()
    assertFixedNjitExecution(value.funding_execution, '资金目标诊断')
    const passing = value.candidates.some(candidate => candidate.goal_check?.within_limits === true)
    if ((value.status === 'diagnosed') !== passing) throw missingDiagnosis()
  }
  return value
}

export function checkedCma<T extends CmaPreview>(value: T): T {
  verified(value)
  if (value.definition.model) {
    if (!value.model_result || !value.effective_assumptions || !value.effective_returns || !value.effective_covariance) throw new Error('模型版本缺少冻结的有效假设，不能用于政策研究。')
    assertFixedNjitExecution(value.model_result.execution, '长期假设模型')
  }
  return value
}

export const getStrategicCatalog = (signal?: AbortSignal) => request<StrategicCatalog>('/catalog', undefined, signal)
export const previewMandate = async (body: MandateStudyRequest, signal?: AbortSignal) => checkedAssessment(await request<MandateAssessment>('/mandates/preview', body, signal))
export const previewMandateFunding = async (body: MandateStudyRequest, signal?: AbortSignal) =>
  verified(await request<MandateFundingEcho>('/mandates/funding', body, signal))
export const confirmMandate = async (body: MandateStudyRequest, previewHash: string, signal?: AbortSignal, replacesMandateId?: string | null) => {
  const value = await request<MandateVersion>('/mandates/confirm', {
    request: body, preview_hash: previewHash, acknowledge_limits: true, replaces_mandate_id: replacesMandateId ?? null,
  }, signal)
  if (!value.assessment?.execution) throw new Error('目标版本缺少对应的诊断记录，已停止交接。')
  checkedAssessment(value.assessment)
  return value
}
export const getMandate = async (id: string, signal?: AbortSignal) => {
  const value = await request<MandateVersion>(`/mandates/${encodeURIComponent(id)}`, undefined, signal)
  if (value.assessment?.preview_hash) checkedAssessment(value.assessment)
  return value
}
export const deleteMandate = (id: string, signal?: AbortSignal) =>
  request<{ deleted: true; id: string }>(`/mandates/${encodeURIComponent(id)}`, undefined, signal, 'DELETE')
export const riskReference = async (body: RiskReferenceRequest, signal?: AbortSignal) => verified(await request<RiskReference>('/risk-reference', body, signal))
export const previewCma = async (body: CmaDefinition, signal?: AbortSignal) => checkedCma(await request<CmaPreview>('/cma/preview', cmaRequest(body), signal))
export const publishCma = async (body: CmaDefinition, previewHash: string, signal?: AbortSignal) => checkedCma(await request<CmaVersion>('/cma', { request: cmaRequest(body), preview_hash: previewHash }, signal))
export const getCma = async (id: string, signal?: AbortSignal) => checkedCma(await request<CmaVersion>(`/cma/${encodeURIComponent(id)}`, undefined, signal))
export const previewPolicy = async (body: PolicyRequest, signal?: AbortSignal) => {
  const wire = verified(await request<Omit<PolicyPreview, 'candidates'> & { candidates: Array<PolicyCandidate | UnavailablePolicyCandidate> }>('/policy/preview', body, signal))
  const unavailable = wire.candidates.filter((c): c is UnavailablePolicyCandidate => c.available === false)
  if (unavailable.some(c => !c.unavailable_reason || (c.id === 'risk-budget' ? Object.keys(c.weights).length > 0 : c.id !== 'compatible' || c.all_models_pass !== false))) throw new Error('不可用候选的状态不完整，请重新比较。')
  const value: PolicyPreview = { ...wire, candidates: wire.candidates.filter((c): c is PolicyCandidate => c.available !== false), unavailable_candidates: unavailable }
  checkMultiCmaEvidence(body, value.multi_cma)
  if (body.mode === 'parameter_average' || body.mode === 'compatible_all_models') {
    for (const candidate of [...value.candidates, ...unavailable.filter(c => c.id === 'compatible')]) {
      const rows = candidate.cross_model_results
      if (!rows || rows.length !== body.cma_refs!.length || body.cma_refs!.some(ref => !rows.some(row =>
        row.cma_id === ref.cma_id && row.cma_hash === ref.content_hash && (row.weight ?? null) === (ref.weight ?? null) &&
        typeof row.within_limits === 'boolean' && Array.isArray(row.violations) && (row.within_limits || row.violations.length > 0)))) throw new Error('原 CMA 交叉评估不完整，请重新比较。')
    }
  }
  if (body.mode === 'compatible_all_models') {
    if (!value.compatibility || value.compatibility.gate !== 'all_frozen_models' ||
      value.compatibility.objective !== (body.compatibility_objective ?? 'minimax_regret') || !value.compatibility.joint_solver ||
      value.candidates.some(c => c.id !== 'compatible' || c.all_models_pass !== true || !c.cross_model_results?.every(row => row.within_limits))) {
      throw new Error('共同配置缺少全模型约束证据，请重新比较。')
    }
  }
  checkGoalCandidates(value.mandate, value.candidates)
  if (cashSuccessRequired(value.mandate)) {
    if (!value.funding_execution) throw missingDiagnosis()
    assertFixedNjitExecution(value.funding_execution, '政策资金目标诊断')
  }
  return value
}
export const publishPolicy = async (body: PolicyRequest, hash: string, candidate: PolicyCandidate['id'], name: string, reason: string, signal?: AbortSignal) => {
  const result = await request<StrategicBaseline>('/policies', { request: body, preview_hash: hash, candidate_id: candidate, name, reason }, signal)
  if (!result.policy) throw new Error('返回的版本缺少政策与目标引用，已停止交接。')
  checkMultiCmaEvidence(body, result.policy.multi_cma)
  if (body.mode === 'compatible_all_models' && (result.policy.mode !== body.mode || result.policy.compatibility?.gate !== 'all_frozen_models')) throw new Error('共同配置的冻结模式不一致，已停止交接。')
  assertFixedNjitExecution(result.policy.execution, '政策采纳')
  return result
}

function checkMultiCmaEvidence(body: PolicyRequest, evidence?: MultiCmaEvidence) {
  if (body.mode !== 'parameter_average' && body.mode !== 'compatible_all_models') return
  const common = body.mode === 'compatible_all_models'
  if (!evidence || evidence.mode !== body.mode || evidence.aggregation_semantics !== (common ? 'all_models_required' : 'parameter_average') ||
    (common && (evidence.primary_evaluation_spec !== 'each_frozen_source_model' || evidence.effective_moments_role !== 'display_reference_only')) ||
    !Array.isArray(evidence.refs) || !Array.isArray(evidence.sources) || evidence.refs.length !== body.cma_refs?.length ||
    evidence.sources.length !== body.cma_refs?.length || body.cma_refs.some(ref => !evidence.refs.some(item =>
      item.cma_id === ref.cma_id && item.content_hash === ref.content_hash && (item.weight ?? null) === (ref.weight ?? null)) || !evidence.sources.some(source =>
        source.cma_id === ref.cma_id && source.content_hash === ref.content_hash && (source.weight ?? null) === (ref.weight ?? null) && source.name && source.as_of))) {
    throw new Error('返回的融合证据与所选 CMA 不一致，已停止交接。')
  }
}

export { request as strategicRequest }

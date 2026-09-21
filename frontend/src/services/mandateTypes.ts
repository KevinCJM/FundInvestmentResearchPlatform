import type { CapitalTarget, CashBudget as WireBudget, CashProtection, MandatePolicy as WirePolicy, RiskAuthorization } from './mandateContract.generated'
import type { FundingMetrics, FundingFlow, MandateDefinition, MandateAssessment } from './strategicAllocation'
import type { NativeNumericalExecutionAudit } from '../utils/fixedNjitExecution'
import { assertNativeNumericalExecution } from '../utils/fixedNjitExecution'

export type CashBudget = Omit<Required<WireBudget>, 'flows'> & { flows: FundingFlow[] }
export type BoundaryPolicy = Omit<Required<WirePolicy>, 'confirmed'> & { confirmed: boolean }
export type { CapitalTarget, CashProtection, RiskAuthorization }
export interface RiskDecision {
  mode: RiskAuthorization['mode']; status: string; risk_scale_ref: RiskAuthorization['risk_scale_ref']
  authorized_max_level: number | null; selected_max_level: number | null
  minimum_tested_feasible_level: number | null; realized_model_risk_level: number | null
  authorized_volatility_cap: number | null; selected_volatility_cap: number | null; selection_pending: boolean
  scale_name?: string; scale_version?: number; applied_boundaries?: number[]
  current_application_blockers: Array<{ code: string; message: string }>
}
export interface ReferenceCandidate {
  id: string; solver_status: string; hard_constraints_pass: boolean; risk_level: number | null
  expected_return: number | null; volatility: number | null
  search_probability: number | null; search_probability_lower: number | null
}
export interface FundingValidation {
  seed: number; paths: number; within_limits: boolean; threshold: number; gate_basis: string
  central: FundingMetrics; candidate_frozen_before_validation: boolean
}
export interface ReferenceFrontierPoint {
  node_id: number; solver_status?: string; status?: string; expected_return: number | null; volatility: number | null
}
/** Which stated bound actually binds, read off the frozen reference frontier. */
export interface Reachability {
  volatility_cap: number; max_return_under_cap: number | null; max_return_candidate_id: string | null
  max_return_risk_level: number | null; target_return: number | null
  min_volatility_for_target: number | null; required_risk_level: number | null
  binding: 'none' | 'volatility_cap' | 'unreachable_at_any_level'; basis: string
}
export interface ReferenceDiagnosis {
  status: 'reference_pending' | 'awaiting_actual_scope' | 'constraint_conflict'
    | 'solver_failed' | 'no_validated_candidate_in_search' | 'validation_failed' | 'validated'
  minimum_tested_feasible_level: number | null
  selected_candidate: (ReferenceCandidate & { weights: Record<string, number>; content_hash: string }) | null
  validation: FundingValidation | null
  adjustment_diagnosis?: { candidate: ReferenceCandidate & { weights: Record<string, number>; content_hash: string }
    validation: FundingValidation; purpose: 'fixed_candidate_capital_diagnostic_only'; selection_rule: string }
  candidates: ReferenceCandidate[]; limitations: string[]; blockers: string[]
  reference_frontier?: ReferenceFrontierPoint[]; constrained_frontier?: ReferenceFrontierPoint[]; risk_boundaries?: number[]
  cash_constraint?: { cash_asset_ids: string[]; requested_min_cash_weight: number; cashflow_derived_weight: number; effective_min_cash_weight: number }
  reachability?: Reachability
  search_seed: number; validation_seed: number; paths: number; execution: NativeNumericalExecutionAudit
  distribution?: { engine: string; version: string; frequency: string }
  reference_input_ref?: { id: string; content_hash: string }; risk_scale_ref?: { id: string; content_hash: string }
}

export const cashSuccessRequired = (d: MandateDefinition) => d.schema_version === '2.0'
  ? Boolean(d.cash_budget && (d.funding_target || d.cash_protection)) : Boolean(d.funding_plan)
export const hasCashBudget = (d: MandateDefinition) => Boolean(d.cash_budget || d.funding_plan)
export const fundingThreshold = (d: MandateDefinition) => d.schema_version === '2.0'
  ? d.boundary_policy?.required_probability : d.funding_plan?.required_probability

/** Fail closed on corrupt reference advice; never render a missing check as success. */
export function checkReferenceAssessment(value: MandateAssessment, checkMetrics: (m: FundingMetrics, threshold: number, allowMissingAlert?: boolean) => void) {
  if (value.definition.schema_version !== '2.0') return
  const fail = () => { throw new Error('风险授权或独立验证记录不完整，请重新诊断。') }
  const d = value.risk_decision, r = value.reference_diagnosis
  if (!d || typeof d.selection_pending !== 'boolean') return fail()
  if (d.selection_pending ? value.definition.max_volatility !== null :
    value.definition.max_volatility !== d.selected_volatility_cap) return fail()
  if (d.mode !== 'explicit_numeric' && !d.selection_pending) {
    if (!d.applied_boundaries || d.applied_boundaries.length !== 5 || !d.selected_max_level || !d.authorized_max_level
      || d.selected_max_level > d.authorized_max_level || d.selected_volatility_cap !== d.applied_boundaries[d.selected_max_level - 1]) return fail()
  }
  if (!r) {
    if (d.risk_scale_ref) return fail()
    return
  }
  assertNativeNumericalExecution(r.execution, '目标参考诊断')
  if (!Array.isArray(r.candidates) || !Array.isArray(r.blockers) || !Array.isArray(r.limitations)
    || r.search_seed !== value.request.seed || r.validation_seed !== value.request.validation_seed
    || r.search_seed === r.validation_seed) return fail()
  if (r.adjustment_diagnosis && (r.adjustment_diagnosis.purpose !== 'fixed_candidate_capital_diagnostic_only'
    || r.status !== 'no_validated_candidate_in_search' || !r.adjustment_diagnosis.candidate.hard_constraints_pass
    || !r.candidates.some(c => c.id === r.adjustment_diagnosis!.candidate.id && c.hard_constraints_pass))) return fail()
  for (const v of [r.validation, r.adjustment_diagnosis?.validation].filter((item): item is FundingValidation => item != null)) {
    const threshold = fundingThreshold(value.definition)
    if (typeof threshold !== 'number' || typeof v.within_limits !== 'boolean' || v.threshold !== threshold
      || v.seed !== r.validation_seed || v.paths !== r.paths || !v.candidate_frozen_before_validation
      || v.gate_basis !== 'wilson_95pct_lower_bound') return fail()
    checkMetrics(v.central, threshold, true)
    if (v.within_limits !== (v.central.probability_lower >= threshold)) return fail()
  }
  if (r.status === 'validated') {
    const c = r.selected_candidate
    if (!c || !c.hard_constraints_pass || !r.candidates.some(x => x.id === c.id && x.hard_constraints_pass)
      || !Number.isFinite(c.volatility) || !c.risk_level || c.risk_level !== r.minimum_tested_feasible_level
      || d.minimum_tested_feasible_level !== r.minimum_tested_feasible_level
      || cashSuccessRequired(value.definition) && !r.validation?.within_limits) return fail()
  } else if (r.minimum_tested_feasible_level !== null) return fail()
}

// UI-only fixtures. Numerical verification uses independent backend tests.
import type { FundingMetrics, MandateAssessment, MandateStudyRequest } from '../services/strategicAllocation'
import { mandateDefinition, policyPreview } from './strategicAllocationFixtures'
import { taaExecution } from './tacticalAllocationFixtures'

export const fundingStudy: MandateStudyRequest = {
  definition: { ...mandateDefinition, name: '资金目标方案', objective_kind: 'funding_goal', target_return: 0,
    boundary_reason: '按必要支付与风险承受能力制定边界', benchmark: null, max_tracking_error: 0,
    funding_plan: { total_capital: 1_000_000, outside_reserve: 100_000, terminal_target: 1_500_000,
      amount_basis: 'nominal', inflation: .02, annual_fee: .005, required_probability: .8,
      liquidity_months: 12, contribution_stress_ratio: .5, drawdown_alert: .2, flows: [] } },
  cma_id: null, simulation_paths: 2000, seed: 42, uncertainty_penalty: 1,
}
export const fundingMetrics: FundingMetrics = {
  success_probability: .82, probability_lower: .79, probability_upper: .85, payment_failure_probability: .01,
  terminal_p05: 800_000, terminal_median: 1_700_000, terminal_p95: 2_800_000,
  expected_terminal_shortfall: 40_000, expected_unpaid_payments: 100, market_drawdown_p95: .3,
  drawdown_alert_probability: .2, required_initial_capital: 950_000, additional_initial_capital: 50_000,
  success_with_10pct_more_capital: .88, success_with_10pct_lower_target: .9,
  capital_gate_status: 'solved', gate_required_initial_capital: 975_000, gate_additional_initial_capital: 75_000,
  gate_success_probability: .825, gate_probability_lower: .808, gate_probability_upper: .841,
  annual_fan: [{ year: 0, p05: 900_000, median: 900_000, p95: 900_000 }, { year: 10, p05: 800_000, median: 1_700_000, p95: 2_800_000 }],
}
export function assessment(request: MandateStudyRequest = fundingStudy): MandateAssessment {
  return { request, definition: request.definition, preview_hash: 'd'.repeat(64), execution: taaExecution,
    status: request.cma_id ? 'needs_revision' : 'inputs_only', blockers: [], warnings: ['模拟采样区间不覆盖模型误差。'],
    ...(request.cma_id ? { funding_execution: taaExecution, funding_model: { version: 'mandate-funding-monthly-lognormal/1.1.0', paths: request.simulation_paths, seed: request.seed, frequency: 'monthly' } } : {}),
    cma: request.cma_id ? { id: request.cma_id, name: '十年人民币假设', as_of: request.definition.as_of, content_hash: 'c'.repeat(64) } : null,
    funding: request.definition.funding_plan ? { investable_capital: 900_000, nominal_terminal_target: 1_500_000,
      total_contributions: 0, total_withdrawals: 0, required_liquid_capital: 0, required_liquid_weight: 0,
      required_effective_return: .0577, root_status: 'solved', currency: 'CNY', liquidity_months: 12,
      liquidity_payment_buffer: 900_000, liquidity_payment_buffer_ratio: 1, liquidity_shortfall_capital: 0,
      monthly_cashflows: [{ month: 1, contribution: 0, withdrawal: 0 }] } : null,
    candidates: request.cma_id ? [{ ...policyPreview.candidates[0], goal_check: { within_limits: false, threshold: .8,
      gate_basis: 'wilson_95pct_lower_bound', central: fundingMetrics, conservative: { ...fundingMetrics, success_probability: .65, probability_lower: .62, probability_upper: .68 }, stress_contribution_ratio: .5 } }] : [],
  }
}

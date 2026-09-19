// Synthetic presentation fixtures only. Backend numerical tests establish the actual calculations.
import type { MandateAssessment, MandateStudyRequest } from '../services/strategicAllocation'
import { newBoundaryMandate, newCashBudget } from '../components/investment-mandate/model'
import { fundingMetrics } from './mandateFixtures'
import { riskVersion } from './riskScaleFixtures'
import { taaExecution } from './tacticalAllocationFixtures'

export function boundaryStudy(): MandateStudyRequest {
  const day = '2026-09-17', definition = newBoundaryMandate(day)
  return { definition: { ...definition, name: '量化边界测试', objective_kind: 'funding_goal', target_return: 0,
    min_cash_weight: .10,
    cash_budget: { ...newCashBudget(day), total_capital: 1_000_000, flows: [] },
    funding_target: { amount: 1_500_000, amount_basis: 'nominal' },
    risk_authorization: { mode: 'manual_level', source: 'risk_scale_selection', authorized_max_level: 3, selected_max_level: 3,
      risk_scale_ref: { id: riskVersion.id, content_hash: riskVersion.content_hash } } },
    cma_id: null, simulation_paths: 2000, seed: 42, validation_seed: 104729, uncertainty_penalty: 1 }
}

export function boundaryAssessment(request = boundaryStudy()): MandateAssessment {
  const level = request.definition.risk_authorization?.selected_max_level ?? 3
  const cap = riskVersion.preview.result.applied_boundaries[level - 1]
  const policy = { name: 'investment-objectives-model-convention-v1', source: 'investment_objectives_model_convention_v1',
    reviewed_on: request.definition.as_of, valid_until: request.definition.review_date ?? null, confirmed: true as const,
    required_probability: .8, liquidity_months: 12, contribution_stress_ratio: .5, cash_reserve_weight: 0 }
  const definition = { ...request.definition, max_volatility: cap, boundary_policy: policy,
    boundary_policy_hash: 'f'.repeat(64), effective_cash_reserve_weight: .10,
    risk_authorization: { ...request.definition.risk_authorization!, selected_max_level: level } }
  const chosen = { id: 'frontier-10', solver_status: 'optimal_to_tolerance', hard_constraints_pass: true,
    risk_level: 2, expected_return: .03, volatility: .03, search_probability: .9, search_probability_lower: .88 }
  const central = { ...fundingMetrics, success_probability: .9, probability_lower: .88, probability_upper: .92,
    drawdown_alert_probability: null }
  return { request, definition, preview_hash: 'd'.repeat(64), execution: taaExecution,
    status: 'diagnosed', diagnosis_scope: 'universal_reference', blockers: [], warnings: ['Synthetic reference only'],
    cma: null, funding: { investable_capital: 1_000_000, nominal_terminal_target: 1_500_000,
      total_contributions: 0, total_withdrawals: 0, required_liquid_capital: 50_000, required_liquid_weight: .05,
      required_effective_return: .042, root_status: 'solved', currency: 'CNY', liquidity_months: 12,
      liquidity_payment_buffer: 50_000, liquidity_payment_buffer_ratio: .05, liquidity_shortfall_capital: 0,
      monthly_cashflows: [{ month: 1, contribution: 0, withdrawal: 0 }] }, candidates: [],
    risk_decision: { mode: 'manual_level', status: 'selected',
      risk_scale_ref: { id: riskVersion.id, content_hash: riskVersion.content_hash }, authorized_max_level: level,
      selected_max_level: level, minimum_tested_feasible_level: 2, realized_model_risk_level: null,
      authorized_volatility_cap: cap, selected_volatility_cap: cap, selection_pending: false,
      applied_boundaries: riskVersion.preview.result.applied_boundaries, scale_name: riskVersion.name,
      scale_version: 1, current_application_blockers: [] },
    reference_diagnosis: { status: 'validated', minimum_tested_feasible_level: 2,
      selected_candidate: { ...chosen, weights: { cash: .8, equity: .2 }, content_hash: 'e'.repeat(64) },
      candidates: [chosen], blockers: [], limitations: ['Synthetic reference only'], search_seed: request.seed,
      validation_seed: request.validation_seed!, paths: request.simulation_paths, execution: taaExecution,
      distribution: { engine: 'annual_moment_proxy_approximation', version: 'test', frequency: 'equal_model_month' },
      reference_frontier: riskVersion.preview.result.frontier.map(point => ({ node_id: point.node_id,
        expected_return: point.expected_return, volatility: point.volatility, status: point.status })),
      constrained_frontier: riskVersion.preview.result.frontier.slice(1).map(point => ({ node_id: point.node_id,
        expected_return: point.expected_return == null ? null : point.expected_return - .002,
        volatility: point.volatility, status: point.status })),
      risk_boundaries: riskVersion.preview.result.applied_boundaries,
      cash_constraint: { cash_asset_ids: ['cash'], requested_min_cash_weight: .10, cashflow_derived_weight: .05,
        effective_min_cash_weight: .10 },
      validation: { seed: request.validation_seed!, paths: request.simulation_paths, within_limits: true,
        candidate_frozen_before_validation: true, threshold: .8, gate_basis: 'wilson_95pct_lower_bound', central } } }
}

// Offline interaction fixtures only. No fixture is imported by a production page.
import type { CmaDefinition, CmaPreview, CmaVersion, MandateDefinition, MandateVersion, PolicyPreview, StrategicCatalog } from '../services/strategicAllocation'
import { taaBaseline, taaCatalog, taaExecution } from './tacticalAllocationFixtures'

export const mandateDefinition: MandateDefinition = {
  name: '长期配置目标', as_of: '2026-09-12', review_date: '2099-09-12', currency: 'CNY', horizon_years: 10,
  target_return: .03, target_excess_return: 0, min_cash_weight: 0, max_volatility: .15, min_liquid_weight: .2, max_illiquid_weight: 0,
  max_tracking_error: .04, risk_aversion: 5, rebalance_policy: 'quarterly', rebalance_note: '', note: '',
}
export const mandateVersion: MandateVersion = { id: 'mandate-1', name: mandateDefinition.name, created_at: '2026-09-12T00:00:00Z', content_hash: 'm'.repeat(64), definition: mandateDefinition }
export const cmaDefinition: CmaDefinition = {
  name: '十年人民币假设', alloc_name: '股债分类', as_of: '2026-09-12', currency: 'CNY', horizon_years: 10,
  return_basis: 'annual_arithmetic_total_return', basis_confirmed: true, source: '离线研究员假设，用于界面测试',
  assets: [
    { id: 'equity', role: 'growth', liquidity: 'liquid', rationale: '广泛权益代理', annual_return: .07, annual_volatility: .18, mean_uncertainty: .02 },
    { id: 'bond', role: 'rates', liquidity: 'liquid', rationale: '利率债防御代理', annual_return: .03, annual_volatility: .05, mean_uncertainty: .005 },
  ], correlation: [[1, -.1], [-.1, 1]], risk_origin: 'manual', risk_reference: null, risk_reference_hash: null,
}
export const cmaPreview: CmaPreview = { preview_hash: 'a'.repeat(64), definition: cmaDefinition, source_snapshot: taaBaseline,
  covariance: [[.0324, -.0009], [-.0009, .0025]], warnings: ['显式假设不是收益保证。'], execution: taaExecution }
export const cmaVersion: CmaVersion = { ...cmaPreview, id: 'cma-1', name: cmaDefinition.name, created_at: '2026-09-12T00:00:00Z', content_hash: 'c'.repeat(64) }
export const strategicCatalog: StrategicCatalog = { allocations: taaCatalog.allocations, mandates: [mandateVersion],
  assumptions: [{ id: cmaVersion.id, name: cmaVersion.name, alloc_name: cmaDefinition.alloc_name, as_of: cmaDefinition.as_of, currency: 'CNY', horizon_years: 10 }], policies: [] }
export const policyPreview: PolicyPreview = {
  preview_hash: 'p'.repeat(64), request: { mandate_id: 'mandate-1', cma_id: 'cma-1', constraints: {}, group_limits: [], uncertainty_penalty: 1, candidate_count: 2000, seed: 42 },
  mandate: mandateDefinition, assumptions: cmaDefinition, source_snapshot: taaBaseline, accepted_candidates: 1500,
  candidates: [{ id: 'robust-utility', name: '区间稳健效用', weights: { equity: .4, bond: .6 }, risk_contributions: { equity: .8, bond: .2 },
    metrics: { expected_return: .046, volatility: .076, conservative_return: .035, nominal_utility: .03, robust_utility: .02 } }],
  warnings: ['候选比较不保证全局最优。'], execution: taaExecution,
}
export const policyBaseline = { ...taaBaseline, id: 'POLICY-1', as_of: '2026-09-12', policy: {
  mandate_id: 'mandate-1', cma_id: 'cma-1', expires_on: '2099-09-12', reason: '采纳保守假设下的配置',
  mandate: mandateDefinition, independent_approval: false, execution: taaExecution,
} }

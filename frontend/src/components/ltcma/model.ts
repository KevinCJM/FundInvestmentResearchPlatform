import { cmaDraftFromDefinition, type CmaDraft, type CmaVersion, type EconomicRole } from '../../services/strategicAllocation'
import { isStatisticalCma, type CmaModelRequest } from '../../services/cmaModelTypes'
import type { CmaMethodId, LtcmaOptions } from '../../services/ltcma'
import type { ReferenceInputRequest } from '../../services/riskScales'

export const blankMatrix = (n: number) => Array.from({ length: n }, (_, i) => Array.from({ length: n }, (__, j) => i === j ? 1 : NaN))
export const methodOf = (draft: CmaDraft): CmaMethodId => draft.model?.method ?? 'manual'
export const scopeKey = (draft: CmaDraft) => draft.strategic_universe_id ? `universe:${draft.strategic_universe_id}` : draft.alloc_name ? `allocation:${draft.alloc_name}` : ''
export function newDraft(day: string): CmaDraft {
  return { schema_version: '2.0', moment_semantics: 'annualized_periodic_arithmetic', fee_basis: 'source_embedded_no_additional_fee',
    fx_hedging_basis: 'same_currency_no_conversion', name: '', alloc_name: null, strategic_universe_id: null,
    implementation_mapping_id: null, as_of: day, currency: 'CNY', horizon_years: 10,
    return_basis: 'annual_arithmetic_total_return', source: '', basis_confirmed: false, assets: [], correlation: [],
    risk_origin: 'manual', risk_reference: null, risk_reference_hash: null, model: null }
}
export function modelFor(method: CmaMethodId, draft: CmaDraft): CmaModelRequest | null {
  if (method === 'manual') return null
  const context = { asset_ids: draft.assets.map(a => a.id), as_of: draft.as_of, currency: draft.currency,
    source: draft.source, return_basis: 'annual_arithmetic_total_return' as const }
  if (method === 'black_litterman') return { ...context, method, covariance: draft.assets.map(() => draft.assets.map(() => NaN)),
    risk_covariance_basis: 'input_covariance', market_weights: Object.fromEntries(draft.assets.map(a => [a.id, NaN])),
    market_weight_source: '', delta: NaN, tau: .05, risk_free_rate: NaN, views: [] }
  if (method === 'scenario_mixture') return { ...context, method, risk_mode: 'shared',
    shared_covariance: draft.assets.map(() => draft.assets.map(() => NaN)), scenarios: [] }
  const window = isStatisticalCma(draft.model) ? draft.model.window : { kind: '5Y' as const }
  const common = { ...context, window, observation_frequency: 'daily' as const, periods_per_year: 252 as const,
    proxy_inputs: isStatisticalCma(draft.model) ? draft.model.proxy_inputs : undefined }
  if (method === 'historical_statistics') return { ...common, method, shrinkage: .1 }
  if (method === 'bayesian_niw') return { ...common, method, prior_ref: { id: '', content_hash: '' },
    prior_mode: 'recenter', mean_prior_observations: null, covariance_prior_observations: null, data_reuse_acknowledged: false }
  return { ...common, method: 'historical_regime_occupancy', run_ref: { id: '', content_hash: '' }, shrinkage: 0,
    probabilities: null, probability_reason: '' }
}
export function changeMethod(draft: CmaDraft, method: CmaMethodId): CmaDraft {
  const next = { ...draft, schema_version: '2.0' as const, moment_semantics: method === 'scenario_mixture' ? 'one_year_simple' as const : 'annualized_periodic_arithmetic' as const,
    basis_confirmed: false, risk_origin: 'manual' as const, risk_reference: null, risk_reference_hash: null,
    assets: draft.assets.map(a => ({ ...a, mean_uncertainty: method === 'manual' ? a.mean_uncertainty : 0,
      annual_return: NaN, annual_volatility: NaN })), correlation: blankMatrix(draft.assets.length) }
  const model = modelFor(method, next)
  if (next.strategic_universe_id && isStatisticalCma(model) && !model.proxy_inputs) model.proxy_inputs = proxyFor(next)
  return { ...next, model }
}
export function applyScope(draft: CmaDraft, key: string, options: LtcmaOptions): CmaDraft {
  const universe = key.startsWith('universe:') ? options.strategic_universes.find(x => x.id === key.slice(9)) : null
  const allocation = key.startsWith('allocation:') ? options.allocations.find(x => x.alloc_name === key.slice(11)) : null
  const metadata = universe?.definition.assets ?? allocation?.assets ?? []
  const assets: CmaDraft['assets'] = metadata.map(asset => ({ id: asset.id,
    role: ('role' in asset ? asset.role : '') as EconomicRole | '',
    liquidity: ('liquidity' in asset ? asset.liquidity : '') as 'liquid' | 'illiquid' | '',
    rationale: ('rationale' in asset ? asset.rationale : '') as string,
    annual_return: NaN, annual_volatility: NaN, mean_uncertainty: 0 }))
  const next = { ...draft, assets, currency: universe?.definition.currency ?? 'CNY',
    alloc_name: allocation?.alloc_name ?? null, strategic_universe_id: universe?.id ?? null,
    implementation_mapping_id: null, correlation: blankMatrix(assets.length) }
  const result = changeMethod(next, methodOf(draft))
  // A product allocation owns its return series; do not retain another scope's proxies.
  if (isStatisticalCma(result.model)) result.model = { ...result.model,
    proxy_inputs: universe ? proxyFor(result, metadata.map(x => x.name || x.id)) : undefined }
  return result
}
export function proxyFor(draft: CmaDraft, names?: string[]): ReferenceInputRequest {
  return { name: draft.name || 'LTCMA', as_of: draft.as_of, currency: 'CNY', calendar: 'SSE', frequency: 'daily', periods_per_year: 252,
    return_basis: 'selected_index_and_adjusted_product_total_return', fee_basis: 'source_embedded_no_additional_fee', fx_basis: 'same_currency_no_conversion',
    assets: draft.assets.map((a, i) => ({ id: a.id, name: names?.[i] ?? a.id, asset_type: 'market',
      rationale: a.rationale, cash_return: null, components: [], rebalance: 'daily' })) }
}
export function copyVersion(value: CmaVersion): CmaDraft {
  const raw = cmaDraftFromDefinition(JSON.parse(JSON.stringify(value.definition)))
  const draft: CmaDraft = { ...raw, schema_version: '2.0', implementation_mapping_id: null, basis_confirmed: false,
    moment_semantics: raw.model?.method === 'scenario_mixture' ? 'one_year_simple' : raw.moment_semantics ?? 'annualized_periodic_arithmetic',
    fee_basis: raw.fee_basis ?? 'explicit_assumption', fx_hedging_basis: raw.fx_hedging_basis ?? 'explicit_assumption' }
  return draft
}
export function updateContext(draft: CmaDraft, patch: Partial<CmaDraft>): CmaDraft {
  const next = { ...draft, ...patch }
  if (next.as_of !== draft.as_of) {
    next.risk_origin = 'manual'; next.risk_reference = null; next.risk_reference_hash = null
  }
  if (next.model) {
    next.model = { ...next.model, as_of: next.as_of, currency: next.currency, source: next.source }
    if (isStatisticalCma(next.model) && next.model.proxy_inputs) next.model = {
      ...next.model, proxy_inputs: { ...next.model.proxy_inputs, name: next.name || 'LTCMA', as_of: next.as_of } }
  }
  return next
}

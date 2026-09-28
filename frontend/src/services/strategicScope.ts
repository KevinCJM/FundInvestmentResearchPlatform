import { systemText } from '../i18n/runtime'
import { strategicRequest, type EconomicRole } from './strategicAllocation'
import type { ReferenceAsset } from './riskScales'

export type ResearchProxy = Pick<ReferenceAsset, 'asset_type' | 'cash_return' | 'components' | 'rebalance'> & { source_labels: Record<string, string> }

export interface StrategicAsset {
  id: string; name: string; currency: string; role: EconomicRole
  liquidity: 'liquid' | 'illiquid'; rationale: string; source: string
  research_proxy?: ResearchProxy | null
  /** 范围层大类权重边界；未设置时不写入，保持旧版本哈希。 */
  weight_limits?: WeightLimits | null
}
export interface WeightLimits { min_weight: number; max_weight: number }
export interface UniverseDefinition {
  name: string; as_of: string; currency: string; source: string; assets: StrategicAsset[]
}
export interface UniversePreview {
  definition: UniverseDefinition; preview_hash: string; implementation_status: 'unmapped'
  implementation_gaps: string[]; research_only: true
}
export interface UniverseVersion extends UniversePreview { id: string; name: string; content_hash: string; created_at: string; mandate_id?: string; mandate_hash?: string }

export type ScopeFeasibilityInput = { mandate_id: string; as_of: string } & (
  | { strategic_definition: UniverseDefinition }
  | { product_version_ids: string[]; product_excluded_keys?: string[] }
  | { universe_snapshot_id: string }
)
export type ScopeHistoryWindow = '1Y' | '2Y' | '3Y' | '5Y' | '10Y' | 'common_since_inception'
export interface ScopeFeasibilityPoint {
  volatility: number | null; expected_return: number | null; weights: Record<string, number>; status: string
}
export type ScopeReferenceComparison = { status: 'not_linked' | 'unavailable' | 'incompatible' } | {
  status: 'available'; risk_scale_ref: { id: string; content_hash: string }; name: string; as_of: string; currency: string
  sample_start: string | null; sample_end: string | null
  points: Omit<ScopeFeasibilityPoint, 'weights'>[]; constrained_points: Omit<ScopeFeasibilityPoint, 'weights'>[]
}
export interface ScopeFeasibilityResult {
  status: 'feasible' | 'infeasible' | 'undetermined'
  reason_code: string | null
  reasons: Array<{ code: string; message: string }>
  research_only: true
  mandate: { id: string; content_hash: string; target_return: number | null; volatility_cap: number | null; min_cash_weight: number
    target_curve?: Array<{ volatility: number; expected_return: number | null }>
    return_requirements?: import('./strategicAllocation').ReturnRequirements
    funding_requirement?: { required_return: number | null; status: 'solved' | 'at_lower_bound' | 'above_search_bound' | 'funding_return_unresolved' | 'benchmark_required' | 'benchmark_moments_required'; basis: 'annual_effective_gross_of_model_fee' } }
  scope?: { kind: string; asset_ids: string[] } | null
  sample?: { requested_start: string | null; requested_end: string; actual_start: string; actual_end: string; observations: number
    common_days: number; excluded_return_periods: number; missing_trading_days: number } | null
  frontier?: { points: ScopeFeasibilityPoint[]; complete: boolean; status: string; constraints_applied: boolean } | null
  reference_comparison?: ScopeReferenceComparison
  funding_comparison?: {
    basis: 'annual_compound_median_gross_of_model_fee'
    status: 'passed' | 'no_candidate' | 'unavailable'
    target_return: number | null
    points: ScopeFeasibilityPoint[]
    reference_points: Omit<ScopeFeasibilityPoint, 'weights'>[]
    constrained_points: Omit<ScopeFeasibilityPoint, 'weights'>[]
    candidate: { volatility: number; expected_return: number; weights: Record<string, number> } | null
    probability_validated: false
  } | null
  target_check?: {
    target_return: number | null; volatility_cap: number | null; max_return_under_cap: number | null; status: string
    candidate: { volatility: number; expected_return: number; weights: Record<string, number> } | null
  } | null
  additional_checks?: { funding: boolean; benchmark: boolean }
}
export async function getScopeFeasibility(body: ScopeFeasibilityInput & { window: { kind: ScopeHistoryWindow } }, signal?: AbortSignal) {
  const value = await strategicRequest<ScopeFeasibilityResult>('/scope-feasibility', body, signal)
  if (value.research_only !== true || !['feasible', 'infeasible', 'undetermined'].includes(value.status)
    || value.mandate?.id !== body.mandate_id || !Array.isArray(value.reasons)
    || (value.frontier != null && !Array.isArray(value.frontier.points))) {
    throw new Error(systemText('preInvestment.strategicScope.scopeScreeningResultsAreIncompleteOrDo'))
  }
  return value
}
/** Human summary of a saved research proxy, for reuse suggestions elsewhere. */
export function proxySummaryText(proxy: ResearchProxy | null | undefined): string {
  if (!proxy) return ''
  if (proxy.asset_type === 'cash') return systemText('preInvestment.strategicScope.expectedAnnualReturn', { p0: ((proxy.cash_return ?? 0) * 100).toFixed(2) })
  if (!proxy.components.length) return ''
  return proxy.components.map(item => `${proxy.source_labels[item.series_id] ?? item.series_id.split(':').pop()} ${(item.weight * 100).toFixed(0)}%`).join(' + ')
}
export function researchProxyIssue(definition: UniverseDefinition): string {
  for (const asset of definition.assets) {
    const proxy = asset.research_proxy
    if (!proxy) continue
    const label = asset.name || systemText('preInvestment.strategicScope.unnamedAsset')
    if (definition.currency !== 'CNY' && (proxy.asset_type === 'cash' || proxy.components.length)) return systemText('preInvestment.strategicScope.researchProxiesCurrentlySupportCnyOnlySwitch')
    if (proxy.asset_type === 'cash' && (proxy.cash_return == null || !Number.isFinite(proxy.cash_return) || proxy.cash_return < -.5 || proxy.cash_return > 1)) return systemText('preInvestment.strategicScope.enterAnExpectedAnnualCashReturnBetween', { p0: label })
    if (proxy.components.length && (proxy.components.some(item => !Number.isFinite(item.weight) || item.weight < 0 || item.weight > 1) || Math.abs(proxy.components.reduce((sum, item) => sum + item.weight, 0) - 1) > 1e-10)) return systemText('preInvestment.strategicScope.proxyWeightsMustTotal100', { p0: label })
  }
  return ''
}
export interface MappingDefinition {
  name: string; strategic_universe_id: string; universe_snapshot_id: string; alloc_name: string
  as_of: string; valid_until: string
  assignments: Array<{ strategic_asset_id: string; proxy_asset_id: string; rationale: string }>
}
export interface MappingPreview {
  definition: MappingDefinition; preview_hash: string; implementation_status: 'complete' | 'incomplete'
  implementation_gaps: string[]
  coverage: Array<{ strategic_asset_id: string; proxy_asset_id: string | null; status: 'mapped' | 'missing_products' }>
}
export interface MappingVersion extends MappingPreview { id: string; name: string; content_hash: string; created_at: string }
type ValidationIssue = { type?: string; loc?: Array<string | number>; msg?: string; ctx?: Record<string, unknown> }

/** Translate the scope contract at the request boundary; never ask users to edit API fields. */
function universeErrorMessage(detail: unknown, definition: UniverseDefinition): string {
  const fallback = systemText('preInvestment.strategicScope.unableToSaveTheResearchScopeYour')
  if (!Array.isArray(detail)) {
    const message = typeof detail === 'string' ? detail : detail && typeof detail === 'object' && 'message' in detail ? detail.message : null
    return typeof message === 'string' && /[\u3400-\u9fff]/.test(message) ? message : fallback
  }
  const issues = detail.filter((value): value is ValidationIssue => value !== null && typeof value === 'object')
  if (issues.some(issue => issue.type === 'extra_forbidden')) {
    const proxyUnsupported = issues.some(issue => issue.type === 'extra_forbidden' && Array.isArray(issue.loc) && issue.loc.includes('research_proxy'))
    return systemText('preInvestment.strategicScope.thePageAndServiceVersionsDoNot', { p0: proxyUnsupported ? systemText('preInvestment.strategicScope.indexProductOrCashReturnSettings') : '' })
  }
  const messages = issues.map(issue => {
    const path = Array.isArray(issue.loc) ? issue.loc.filter(part => part !== 'body' && part !== 'request') : []
    const assetIndex = path[0] === 'assets' && typeof path[1] === 'number' ? path[1] : null
    const asset = assetIndex === null ? null : definition.assets[assetIndex]
    const prefix = assetIndex === null ? '' : `「${asset?.name.trim() || systemText('preInvestment.strategicScope.assetClass', { p0: assetIndex + 1 })}」：`
    const field = path[path.length - 1]
    const labels: Record<string, string> = {
      name: assetIndex === null ? systemText('preInvestment.strategicScope.strategicScopeName') : systemText('preInvestment.strategicScope.assetName'), as_of: systemText('preInvestment.strategicScope.strategicResearchDate'), currency: systemText('preInvestment.strategicScope.baseCurrency'),
      source: assetIndex === null ? systemText('preInvestment.strategicScope.scopeNotes') : systemText('preInvestment.strategicScope.references'), rationale: systemText('preInvestment.strategicScope.notes'), assets: systemText('preInvestment.strategicScope.assetClasses'),
      research_proxy: systemText('preInvestment.strategicScope.researchProxies'), asset_type: systemText('preInvestment.strategicScope.assetType'), cash_return: systemText('preInvestment.strategicScope.expectedAnnualCashReturn'),
      components: systemText('preInvestment.strategicScope.proxySelection'), rebalance: systemText('preInvestment.strategicScope.proxyRebalancing'), weight: systemText('preInvestment.strategicScope.proxyWeight'),
      kind: systemText('preInvestment.strategicScope.proxyType'), series_id: systemText('preInvestment.strategicScope.proxySelection'), field: systemText('preInvestment.strategicScope.proxyDataConvention'), source_labels: systemText('preInvestment.strategicScope.proxyName'),
    }
    const label = typeof field === 'string' ? labels[field] : undefined
    // These fields are generated by the app, not editable form inputs.
    if (field === 'id' || field === 'preview_hash' || field === 'replaces_universe_id') return systemText('preInvestment.strategicScope.scopeInformationCouldNotBeSynchronizedRefresh')
    if (field === 'cash_return' && ['greater_than_equal', 'less_than_equal', 'float_type', 'float_parsing', 'finite_number'].includes(issue.type ?? '')) return systemText('preInvestment.strategicScope.enterAnExpectedAnnualCashReturnBetween2', { p0: prefix })
    if (field === 'weight' && ['greater_than_equal', 'less_than_equal', 'float_type', 'float_parsing', 'finite_number'].includes(issue.type ?? '')) return systemText('preInvestment.strategicScope.enterEachProxyWeightBetween0And', { p0: prefix })
    if (label) {
      if (issue.type === 'missing') return systemText('preInvestment.strategicScope.enter', { p0: prefix, p1: label })
      if (issue.type === 'string_too_short') return systemText('preInvestment.strategicScope.cannotBeBlank', { p0: prefix, p1: label })
      const maxLength = issue.ctx?.max_length
      if (issue.type === 'string_too_long' && typeof maxLength === 'number') return systemText('preInvestment.strategicScope.mustNotExceedCharacters', { p0: prefix, p1: label, p2: maxLength })
      if (['date_type', 'date_parsing', 'date_from_datetime_parsing', 'date_from_datetime_inexact'].includes(issue.type ?? '')) return systemText('preInvestment.strategicScope.selectAValid', { p0: prefix, p1: label })
      if (issue.type === 'literal_error') return systemText('preInvestment.strategicScope.selectAgain', { p0: prefix, p1: label })
    }
    const message = typeof issue.msg === 'string' ? issue.msg.replace(/^Value error,\s*/, '') : ''
    if (issue.type === 'value_error' && /[\u3400-\u9fff]/.test(message)) return `${prefix}${message}`
    return `${prefix}${label ? systemText('preInvestment.strategicScope.isInvalidCheckAndRetry', { p0: label }) : systemText('preInvestment.strategicScope.unableToSaveThisConfigurationCheckAnd')}`
  })
  return [...new Set(messages)].join('；') || fallback
}

export const previewUniverse = (body: UniverseDefinition, signal?: AbortSignal) => strategicRequest<UniversePreview>('/universes/preview', body, signal, 'POST', detail => universeErrorMessage(detail, body))
export const confirmUniverse = (body: UniverseDefinition, hash: string, signal?: AbortSignal, replacesUniverseId?: string, mandateId?: string) => strategicRequest<UniverseVersion>('/universes/confirm', { request: body, preview_hash: hash, ...(replacesUniverseId ? { replaces_universe_id: replacesUniverseId } : {}), ...(mandateId ? { mandate_id: mandateId } : {}) }, signal, 'POST', detail => universeErrorMessage(detail, body))
export const bindUniverseMandate = (id: string, mandate_id: string, signal?: AbortSignal) => strategicRequest<{ mandate_id: string }>(`/universes/${encodeURIComponent(id)}/mandate`, { mandate_id }, signal)
export const retireUniverse = (id: string, signal?: AbortSignal) => strategicRequest<{ deleted: true; id: string }>(`/universes/${encodeURIComponent(id)}`, undefined, signal, 'DELETE')
export const getStrategicUniverse = (id: string, signal?: AbortSignal) => strategicRequest<UniverseVersion>(`/universes/${encodeURIComponent(id)}`, undefined, signal)
export const previewImplementationMap = (body: MappingDefinition, signal?: AbortSignal) => strategicRequest<MappingPreview>('/implementation-maps/preview', body, signal)
export const confirmImplementationMap = (body: MappingDefinition, hash: string, signal?: AbortSignal) => strategicRequest<MappingVersion>('/implementation-maps/confirm', { request: body, preview_hash: hash }, signal)
export const getImplementationMap = (id: string, signal?: AbortSignal) => strategicRequest<MappingVersion>(`/implementation-maps/${encodeURIComponent(id)}`, undefined, signal)
export const economicRoles: Array<[EconomicRole, string]> = [['growth', systemText('preInvestment.strategicScope.growth')], ['rates', systemText('preInvestment.strategicScope.rates')], ['inflation', systemText('preInvestment.strategicScope.inflation')], ['credit', systemText('preInvestment.strategicScope.credit')], ['liquidity', systemText('preInvestment.strategicScope.cashAndLiquidityReserves')], ['diversifier', systemText('preInvestment.strategicScope.diversifier')]]

const percent = (value: number) => `${(value * 100).toFixed(2)}%`
const isCashAsset = (asset: StrategicAsset) => asset.role === 'liquidity' && asset.liquidity === 'liquid'

/**
 * 范围现金下限偏离投资目标时的提醒；只提示不阻断，SAA 仍按目标下限执行。
 * 未显式设置边界的旧范围沿用目标，不提示。
 */
export function cashFloorIssue(assets: StrategicAsset[], floor: number): string {
  if (!(floor > 0)) return ''
  const cash = assets.find(isCashAsset)
  if (!cash) return systemText('preInvestment.scopeCashFloor.missing', { p0: percent(floor) })
  const min = cash.weight_limits?.min_weight
  return typeof min === 'number' && min < floor - 1e-12 ? systemText('preInvestment.scopeCashFloor.below', { p0: percent(min), p1: percent(floor) }) : ''
}

/** 按投资目标补齐现金大类及其下限：已有现金且显式设过边界时尊重用户设置。 */
export function withCashFloor(definition: UniverseDefinition, floor: number): UniverseDefinition {
  if (!(floor > 0)) return definition
  const limits = { min_weight: floor, max_weight: 1 }
  const index = definition.assets.findIndex(isCashAsset)
  if (index >= 0) return definition.assets[index].weight_limits ? definition
    : { ...definition, assets: definition.assets.map((asset, i) => i === index ? { ...asset, weight_limits: limits } : asset) }
  if (definition.assets.length >= 30) return definition
  const ids = new Set(definition.assets.map(asset => asset.id))
  const id = ids.has('cash') ? `cash-${crypto.randomUUID().slice(0, 8)}` : 'cash'
  return { ...definition, assets: [{ id, name: systemText('riskScales.template.cash'), currency: definition.currency, role: 'liquidity', liquidity: 'liquid',
    rationale: '', source: '', research_proxy: { asset_type: 'cash', cash_return: 0, components: [], rebalance: null, source_labels: {} }, weight_limits: limits }, ...definition.assets] }
}

/** 与后端契约一致：每类 0–100%、下限不高于上限、下限合计不超过 100%。 */
export function weightLimitsIssue(definition: UniverseDefinition): string {
  const limits = definition.assets.map(asset => asset.weight_limits).filter((item): item is WeightLimits => Boolean(item))
  if (limits.some(item => ![item.min_weight, item.max_weight].every(v => Number.isFinite(v) && v >= 0 && v <= 1) || item.min_weight > item.max_weight))
    return systemText('preInvestment.scopeCashFloor.invalidLimits')
  return limits.reduce((sum, item) => sum + item.min_weight, 0) > 1 + 1e-12 ? systemText('preInvestment.scopeCashFloor.minSumExceeds') : ''
}

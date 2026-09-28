import type { ScopeFacts, CmaListItem } from './ltcmaContract.generated'
import type { CmaDraft, CmaVersion } from './strategicAllocation'
import type { UniverseDefinition, ResearchProxy } from './strategicScope'
import { isStatisticalCma } from './cmaModelTypes'

const equal = (a: unknown, b: unknown) => JSON.stringify(a) === JSON.stringify(b)
export function proxyFacts(proxy?: Omit<ResearchProxy, 'source_labels'> | null) {
  return proxy ? [proxy.asset_type, proxy.cash_return ?? null, proxy.rebalance ?? null,
    proxy.components.map(c => [c.kind, c.series_id, c.field, c.weight]).sort((a, b) => {
      for (let i = 0; i < a.length; i++) { if (a[i] !== b[i]) return a[i]! < b[i]! ? -1 : 1 }
      return 0
    })] : null
}
export function scopeFacts(definition?: UniverseDefinition): ScopeFacts | null {
  return definition ? { currency: definition.currency, asset_ids: definition.assets.map(a => a.id),
    asset_currencies: definition.assets.map(a => a.currency), roles: definition.assets.map(a => a.role),
    liquidities: definition.assets.map(a => a.liquidity), proxies: definition.assets.map(a => proxyFacts(a.research_proxy)) } : null
}
export function researchProxyFacts(model: CmaDraft['model']) {
  return isStatisticalCma(model) && model.proxy_inputs ? model.proxy_inputs.assets.map(a => [a.id, proxyFacts(a)]) : null
}
export const cmaScopeFacts = (version: CmaVersion) => scopeFacts(version.source_snapshot.strategic_universe_snapshot?.definition)
export function proxyDifference(a: unknown[] | null, b: unknown[] | null): string | null {
  if (equal(a, b)) return null
  if (!a || !b || a[0] !== b[0]) return 'scopeProxyType'
  if (a[1] !== b[1]) return 'scopeCashReturn'
  if (a[2] !== b[2]) return 'scopeRebalance'
  if (!equal((a[3] as unknown[][]).map(c => c.slice(0, 3)), (b[3] as unknown[][]).map(c => c.slice(0, 3)))) return 'scopeProxySource'
  return 'scopeProxyWeights'
}
export function researchProxyDifference(a?: unknown[] | null, b?: unknown[] | null): string | null {
  if (!a || !b) return null
  const left = a as Array<[string, unknown[] | null]>, right = b as Array<[string, unknown[] | null]>
  if (!equal(left.map(x => x[0]), right.map(x => x[0]))) return 'scopeAssets'
  return left.map((x, i) => proxyDifference(x[1], right[i][1])).find(Boolean) ?? null
}
export function scopeDifference(a?: ScopeFacts | null, b?: ScopeFacts | null): string | null {
  if (!a || !b) return 'scopeFactsMissing'
  const rules: Array<[Array<keyof ScopeFacts>, string]> = [
    [['currency', 'asset_currencies'], 'scopeCurrency'], [['asset_ids'], 'scopeAssets'],
    [['roles'], 'scopeRoles'], [['liquidities'], 'scopeLiquidity'],
  ]
  return rules.find(([keys]) => keys.some(key => !equal(a[key], b[key])))?.[1]
    ?? (a.proxies.length !== b.proxies.length ? 'scopeProxyType' : a.proxies.map((p, i) => proxyDifference(p, b.proxies[i])).find(Boolean) ?? null)
}
type ScopeChoice = { strategic_universe_id?: string | null; alloc_name?: string | null; scope_facts?: ScopeFacts | null }
export function cmaScopeDifference(a: ScopeChoice, b: ScopeChoice): string | null {
  if (Boolean(a.strategic_universe_id) !== Boolean(b.strategic_universe_id)) return 'scopeAssets'
  if (!a.strategic_universe_id) return a.alloc_name === b.alloc_name ? null : 'scopeAssets'
  if (!a.scope_facts || !b.scope_facts) return a.strategic_universe_id === b.strategic_universe_id ? null : 'scopeFactsMissing'
  return scopeDifference(a.scope_facts, b.scope_facts)
}
export function priorReason(item: CmaListItem, value: CmaDraft, universe?: UniverseDefinition): string | null {
  if (item.retired || item.usable?.status === 'blocked') return 'retired'
  if (item.usable?.status === 'stale') return 'priorNotCurrent'
  if (item.method === 'conditional_scenario' || item.downstream_eligible === false) return 'scenarioHandoffBlocked'
  if (item.as_of > value.as_of) return 'priorFuture'
  if (item.currency !== value.currency) return 'scopeCurrency'
  if (!equal(item.asset_ids, value.assets.map(a => a.id))) return 'scopeAssets'
  const issue = cmaScopeDifference(item, { ...value, scope_facts: scopeFacts(universe) })
  if (issue) return issue
  if (item.schema_version !== '2.0' || item.moment_semantics !== 'annualized_periodic_arithmetic') return 'priorBasis'
  if (['return_basis', 'fee_basis', 'fx_hedging_basis'].some(key => {
    const field = key as 'return_basis' | 'fee_basis' | 'fx_hedging_basis'
    return item[field] == null || value[field] == null || item[field] !== value[field]
  })) return 'priorBasis'
  return researchProxyDifference(item.research_proxy_facts, researchProxyFacts(value.model))
}

export function cmaPairReason(a: Partial<CmaListItem>, b: Partial<CmaListItem>): string | null {
  if (a.schema_version !== '2.0' || b.schema_version !== '2.0') return 'handoffLegacy'
  const scope = cmaScopeDifference(a, b)
  if (scope) return scope
  if (a.as_of !== b.as_of) return 'handoffDate'
  if (a.currency !== b.currency) return 'handoffCurrency'
  if (['return_basis', 'moment_semantics', 'fee_basis', 'fx_hedging_basis'].some(key => a[key as keyof CmaListItem] == null || b[key as keyof CmaListItem] == null || a[key as keyof CmaListItem] !== b[key as keyof CmaListItem])) return 'handoffBasis'
  if (!equal(a.asset_ids, b.asset_ids)) return 'handoffAssets'
  return researchProxyDifference(a.research_proxy_facts, b.research_proxy_facts)
}
export function cmaChoice(version: CmaVersion) {
  return { ...version.definition, id: version.id, method: version.definition.model?.method ?? 'manual',
    scope_facts: cmaScopeFacts(version), asset_ids: version.definition.assets.map(a => a.id),
    research_proxy_facts: researchProxyFacts(version.definition.model) }
}

import { ltcma, ltcmaSaaIssue, type CmaListItem } from './ltcma'
import { getStrategicCatalog } from './strategicAllocation'
import { getInvestableUniverseSnapshot } from './productPools'
import type { AllocationJourney } from '../app/allocationJourney'
import { cmaPairReason, cmaChoice } from './cmaCompatibility'
import { getStrategicUniverse } from './strategicScope'

/** Same objectives and constraints, research path and scope. */
export const cmaGroupKey = (item: CmaListItem) => JSON.stringify([item.research_path ?? null, ...item.upstream.map(ref => ref.id)])

/** Fast selection feedback only; SAA still validates frozen sources before calculating. */
export function cmaHandoffIssue(items: CmaListItem[], cutoff: string | undefined): string | null {
  if (!items.length) return 'selectCma'
  if (cutoff === undefined) return 'handoffClock'
  if (items.length > 20 || new Set(items.map(item => item.id)).size !== items.length) return 'handoffLimit'
  if (items.some(item => item.retired)) return 'handoffRetired'
  if (items.some(item => item.usable.status !== 'ready')) return 'handoffNotCurrent'
  if (items.some(item => item.method === 'conditional_scenario' || item.downstream_eligible === false)) return 'scenarioHandoffBlocked'
  if (items.some(item => item.as_of > cutoff)) return 'handoffFuture'
  const first = items[0]
  if (items.some(item => !item.alloc_name && !item.strategic_universe_id)) return 'handoffScopeMissing'
  if (items.some(item => cmaGroupKey(item) !== cmaGroupKey(first))) return 'handoffGroupMismatch'
  if (items.length === 1) return null
  return items.slice(1).map(item => cmaPairReason(first, item)).find(Boolean) ?? null
}

export async function prepareLtcmaHandoff(items: CmaListItem[], cutoff: string | undefined, text: (key: string) => string, signal: AbortSignal) {
  const initialIssue = cmaHandoffIssue(items, cutoff)
  if (initialIssue) throw new Error(text(initialIssue))
  const [views, catalog] = await Promise.all([
    Promise.all(items.map(item => ltcma.view(item.id, signal))), getStrategicCatalog(signal, items[0].strategic_universe_id),
  ])
  if (views.some((view, index) => view.version.id !== items[index].id || view.version.content_hash !== items[index].content_hash)) throw new Error(text('handoffChanged'))
  if (views.some(view => view.retired)) throw new Error(text('handoffRetired'))
  const researchIssue = views.map(view => ltcmaSaaIssue(view.version)).find(Boolean)
  if (researchIssue) throw new Error(researchIssue)
  const first = views[0].version, definition = first.definition
  if (items.length > 1) {
    const pairIssue = views.slice(1).map(({ version }) => cmaPairReason(cmaChoice(first), cmaChoice(version))).find(Boolean)
    if (pairIssue) throw new Error(text(pairIssue))
    if (views.some(({ version }) => JSON.stringify(version.definition.assets.map(asset => [asset.id, asset.role, asset.liquidity])) !==
      JSON.stringify(definition.assets.map(asset => [asset.id, asset.role, asset.liquidity])))) throw new Error(text('handoffAssets'))
  }
  // Resolve the scope's persisted binding. A browser bookmark is not a mandate association.
  const scope = catalog.strategic_universes?.find(item => item.id === definition.strategic_universe_id)
  const allocation = catalog.allocations.find(item => item.alloc_name === definition.alloc_name)
  if (definition.strategic_universe_id ? !scope : !allocation) throw new Error(text('handoffScopeMissing'))
  if (scope) {
    const ids = [...new Set(views.map(view => view.version.definition.strategic_universe_id!))]
    const scopes = await Promise.all(ids.map(id => id === scope.id ? scope : getStrategicUniverse(id, signal)))
    for (const { version } of views) {
      const original = scopes.find(item => item.id === version.definition.strategic_universe_id)
      const hash = version.source_snapshot.lineage.strategic_universe_hash
      if (!original || hash != null && hash !== original.content_hash) throw new Error(text('handoffChanged'))
      if (original.mandate_id !== scope.mandate_id) throw new Error(text('handoffGoalMismatch'))
    }
  }
  const snapshot = !scope && allocation?.universe_snapshot_id
    ? await getInvestableUniverseSnapshot(allocation.universe_snapshot_id, signal) : null
  if (snapshot && snapshot.id !== allocation?.universe_snapshot_id) throw new Error(text('handoffScopeMissing'))
  const mandateId = scope?.mandate_id || snapshot?.mandate_id || undefined
  const mandate = catalog.mandates.find(item => item.id === mandateId)?.definition
  if (mandateId && !mandate) throw new Error(text('handoffMandateUnavailable'))
  if (mandate) {
    if (mandate.currency !== definition.currency) throw new Error(text('handoffGoalCurrency'))
    if (definition.as_of < mandate.as_of || mandate.review_date && definition.as_of >= mandate.review_date) throw new Error(text('handoffGoalDate'))
    if ((mandate.cash_budget || mandate.funding_plan) && definition.as_of !== mandate.as_of) throw new Error(text('handoffCashDate'))
  }
  const query = new URLSearchParams()
  items.forEach(item => query.append('cma', item.id))
  if (scope) query.set('strategic_universe', scope.id)
  else query.set('alloc', definition.alloc_name!)
  if (definition.implementation_mapping_id) query.set('mapping', definition.implementation_mapping_id)
  if (snapshot) query.set('universe', snapshot.id)
  if (mandateId) query.set('mandate', mandateId)
  const journey: Partial<AllocationJourney> = {
    name: scope?.name ?? snapshot?.name ?? definition.alloc_name ?? undefined,
    mandateId, strategicUniverseId: scope?.id, allocationName: scope ? undefined : definition.alloc_name!,
    universeId: snapshot?.id, implementationMappingId: definition.implementation_mapping_id ?? undefined,
    researchDate: definition.as_of, ltcmaId: first.id, ltcmaIds: items.map(item => item.id), baselineId: undefined, taaRunId: undefined,
  }
  return { path: `/pre-investment/saa/policy?${query}`, journey }
}

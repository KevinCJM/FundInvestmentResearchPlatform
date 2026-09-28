import { describe, expect, it } from 'vitest'
import { cmaScopeDifference, priorReason, scopeFacts, scopeDifference } from './cmaCompatibility'
import { ltcmaDefinition, ltcmaItem } from '../test/ltcmaFixtures'
import { cmaSelectionReason } from '../components/strategic-allocation/LtcmaSelection'
import { cmaHandoffIssue } from './ltcmaHandoff'
import { mandateVersion } from '../test/strategicAllocationFixtures'
import type { UniverseDefinition } from './strategicScope'
import { cmaDraftFromDefinition } from './strategicAllocation'

const definition: UniverseDefinition = { name: '股债范围', as_of: ltcmaDefinition.as_of, currency: 'CNY', source: '',
  assets: ltcmaDefinition.assets.map(a => ({ ...a, name: a.id, currency: 'CNY', source: '', research_proxy: {
    asset_type: 'market', cash_return: null, rebalance: 'daily', source_labels: {}, components: [
      { kind: 'index', series_id: 'index:index_daily:000300.SH', field: 'close', weight: .4 },
      { kind: 'index', series_id: 'index:index_daily:000012.SH', field: 'close', weight: .6 },
    ] } })) }
const scope = { strategic_universe_id: 'original-scope', alloc_name: null, scope_facts: scopeFacts(definition) }
const prior = { ...ltcmaItem, ...scope }
const current = cmaDraftFromDefinition({ ...ltcmaDefinition, strategic_universe_id: 'another-record', alloc_name: null })

describe('configuration compatibility shared by NIW, SAA and center handoff', () => {
  it('accepts another record with identical facts, including different labels and component order', () => {
    const edited = structuredClone(definition)
    edited.name = '改名'; edited.source = '补充说明'; edited.assets[0].research_proxy!.components.reverse()
    const next = { ...scope, strategic_universe_id: 'another-record', scope_facts: scopeFacts(edited) }
    expect(cmaScopeDifference(scope, next)).toBeNull()
    expect(priorReason(prior, current, edited)).toBeNull()
    expect(cmaSelectionReason(prior, { allocationName: '', strategicUniverseId: next.strategic_universe_id,
      scopeFacts: next.scope_facts, mandate: mandateVersion.definition, cutoff: current.as_of })).toBeNull()
    expect(cmaHandoffIssue([prior, { ...prior, ...next, id: 'second-cma' }], current.as_of)).toBeNull()
  })
  it.each([
    ['currency', 'USD', 'scopeCurrency'], ['asset_ids', ['other'], 'scopeAssets'],
    ['roles', ['credit'], 'scopeRoles'], ['liquidities', ['illiquid'], 'scopeLiquidity'],
    ['proxies', [], 'scopeProxyType'],
  ])('rejects actual %s changes even if the record IDs agree', (key, value, reason) => {
    expect(scopeDifference(scope.scope_facts, { ...scope.scope_facts!, [key]: value })).toBe(reason)
    expect(cmaScopeDifference(scope, { ...scope, scope_facts: { ...scope.scope_facts!, [key]: value } })).toBe(reason)
  })
  it('keeps date, basis, retirement and evidence completeness independent of scope equality', () => {
    expect(priorReason({ ...prior, retired: true }, current, definition)).toBe('retired')
    expect(priorReason({ ...prior, usable: { status: 'stale', reasons: [] } }, current, definition)).toBe('priorNotCurrent')
    expect(priorReason({ ...prior, usable: { status: 'blocked', reasons: [] } }, current, definition)).toBe('retired')
    expect(priorReason({ ...prior, as_of: '2099-01-01' }, current, definition)).toBe('priorFuture')
    expect(priorReason({ ...prior, moment_semantics: 'one_year_simple' }, current, definition)).toBe('priorBasis')
    expect(priorReason({ ...prior, fee_basis: undefined }, current, definition)).toBe('priorBasis')
    expect(priorReason({ ...prior, method: 'conditional_scenario' }, current, definition)).toBe('scenarioHandoffBlocked')
    expect(cmaScopeDifference(scope, { ...scope, strategic_universe_id: 'other', scope_facts: null })).toBe('scopeFactsMissing')
  })
})

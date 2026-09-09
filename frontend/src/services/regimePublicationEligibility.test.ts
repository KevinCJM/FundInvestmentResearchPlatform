import { describe, expect, it } from 'vitest'
import type { HistoricalRegimeRun, RegimePublication } from './historicalRegimes'
import { eligibleRegimePublications, isRegimeRunEligibleForTaa } from './regimePublicationEligibility'

const hash = 'a'.repeat(64)
const publication: RegimePublication = {
  id: 'publication-1', usage: 'taa', published_at: '2026-09-06',
  run_id: 'run-1', definition_revision: 3, run_content_hash: hash,
  gate: 'comprehensive_formal_gate_passed',
}

function candidate(overrides: Partial<HistoricalRegimeRun> = {}): HistoricalRegimeRun {
  return {
    id: 'run-1', schema_version: '2.0', definition_id: 'definition-1',
    definition_revision: 3, definition_snapshot_hash: 'b'.repeat(64),
    immutable: true, mode: 'realtime', content_hash: hash,
    name: '市场状态', created_at: '2026-09-06', states: [], series: [], segments: [],
    conditional_stats: [], transition: { states: [], counts: [], probabilities: [] },
    stability: {}, walk_forward: {}, diagnostics: [],
    causality: {
      classification: 'causal', is_causal: true, uses_future_data: false, repaints: false,
      realtime_eligible: true, publish_eligible_usages: ['taa', 'formal_backtest'],
      blockers: [], warnings: [],
    },
    governance: { formal_gate_passed: true, publish_eligible_usages: ['taa', 'formal_backtest'] },
    publications: [publication],
    ...overrides,
  }
}

describe('versioned regime publication eligibility', () => {
  it('accepts a frozen V2 TAA publication and an explicitly published formal-backtest alternative', () => {
    expect(isRegimeRunEligibleForTaa(candidate())).toBe(true)
    const formal = candidate({ publications: [{ ...publication, usage: 'formal_backtest' }] })
    expect(isRegimeRunEligibleForTaa(formal)).toBe(true)
    expect(eligibleRegimePublications(formal, 'formal_backtest')).toHaveLength(1)
    expect(eligibleRegimePublications(candidate(), 'formal_backtest')).toHaveLength(0)
  })

  it.each([undefined, '1.0'])('preserves legacy TAA with schema %s and the legacy gate', (schema_version) => {
    const legacy = candidate({
      schema_version, content_hash: 'legacy-hash', definition_snapshot_hash: undefined,
      governance: undefined,
      publications: [{ ...publication, run_content_hash: 'legacy-hash', gate: 'causality_passed' }],
    })
    expect(isRegimeRunEligibleForTaa(legacy)).toBe(true)
    expect(eligibleRegimePublications(legacy, 'formal_backtest')).toHaveLength(0)
  })

  it('does not accept either generation under the other generation publication gate', () => {
    expect(isRegimeRunEligibleForTaa(candidate({
      publications: [{ ...publication, gate: 'causality_passed' }],
    }))).toBe(false)
    expect(isRegimeRunEligibleForTaa(candidate({ schema_version: '1.0' }))).toBe(false)
  })

  it.each(['3.0', '2.1', ''])('fails closed for unsupported schema %s', (schema_version) => {
    expect(isRegimeRunEligibleForTaa(candidate({ schema_version }))).toBe(false)
  })

  it.each([
    ['mutable', { immutable: false }],
    ['retrospective', { mode: 'retrospective' }],
    ['no saved definition', { definition_id: null }],
    ['no revision', { definition_revision: null }],
    ['invalid revision', { definition_revision: 0 }],
    ['missing run snapshot', { content_hash: undefined }],
    ['malformed run snapshot', { content_hash: 'short-hash' }],
    ['missing definition snapshot', { definition_snapshot_hash: undefined }],
    ['missing governance', { governance: undefined }],
    ['failed comprehensive gate', { governance: { formal_gate_passed: false, publish_eligible_usages: ['taa'] } }],
    ['ineligible usage', { governance: { formal_gate_passed: true, publish_eligible_usages: ['research_display'] } }],
    ['no publication', { publications: [] }],
  ] satisfies Array<[string, Partial<HistoricalRegimeRun>]>)('rejects %s', (_label, overrides) => {
    expect(isRegimeRunEligibleForTaa(candidate(overrides))).toBe(false)
  })

  it.each([
    ['uses_future_data', true], ['repaints', true],
    ['is_causal', false], ['realtime_eligible', false],
  ] as const)('rejects unsafe causality flag %s', (flag, value) => {
    const run = candidate()
    run.causality = { ...run.causality, [flag]: value }
    expect(isRegimeRunEligibleForTaa(run)).toBe(false)
  })

  it.each([
    ['run mismatch', { run_id: 'other-run' }],
    ['revision mismatch', { definition_revision: 2 }],
    ['hash mismatch', { run_content_hash: 'c'.repeat(64) }],
    ['missing publication identity', { id: '' }],
    ['research-only publication', { usage: 'research_display' }],
    ['missing gate', { gate: undefined }],
  ] satisfies Array<[string, Partial<RegimePublication>]>)('rejects publication %s', (_label, overrides) => {
    expect(isRegimeRunEligibleForTaa(candidate({
      publications: [{ ...publication, ...overrides }],
    }))).toBe(false)
  })
})

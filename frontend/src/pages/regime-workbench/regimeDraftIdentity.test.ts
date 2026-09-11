import { describe, expect, it } from 'vitest'
import { createBlankRegimeDefinition, type RegimeGraphDefinition } from '../../services/regimeGraph'
import { regimeCalculationFingerprint, regimePresentationFingerprint } from './regimeDraftIdentity'

function fixture(): RegimeGraphDefinition {
  return {
    ...createBlankRegimeDefinition(),
    name: '趋势研究',
    states: [{ id: 'bull', label: '牛市', color: '#ef4444', role: 'neutral', order: 0 }, { id: 'bear', label: '熊市', color: '#22c55e', role: 'neutral', order: 1 }],
    graph: { nodes: [{ id: 'trend', type: 'feature.rolling_mean', label: '均线', parameters: { window: 20, snapshot_id: 'v1' }, inputs: {}, position: { x: 0, y: 0 } }], outputs: { state: { node_id: 'trend', port: 'state' } } },
  }
}

describe('regime draft identities', () => {
  it('keeps numeric results current for layout and display edits but retains a distinct presentation snapshot', () => {
    const before = fixture()
    const after = { ...before, name: '新研究名', states: before.states.map((state) => ({ ...state, label: `新${state.label}`, color: '#64748b' })), graph: { ...before.graph, nodes: before.graph.nodes.map((node) => ({ ...node, label: '新节点名', position: { x: 300, y: 40 } })) } }
    expect(regimeCalculationFingerprint(after, 'realtime', '')).toBe(regimeCalculationFingerprint(before, 'realtime', ''))
    expect(regimePresentationFingerprint(after)).not.toBe(regimePresentationFingerprint(before))
  })

  it('separates mode, as-of, parameter, snapshot and economically meaningful state changes', () => {
    const before = fixture()
    const original = regimeCalculationFingerprint(before, 'realtime', '')
    expect(regimeCalculationFingerprint(before, 'retrospective', '')).not.toBe(original)
    expect(regimeCalculationFingerprint(before, 'realtime', '2026-01-01')).not.toBe(original)
    for (const patch of [{ window: 30 }, { snapshot_id: 'v2' }]) {
      const after = { ...before, graph: { ...before.graph, nodes: before.graph.nodes.map((node) => ({ ...node, parameters: { ...node.parameters, ...patch } })) } }
      expect(regimeCalculationFingerprint(after, 'realtime', '')).not.toBe(original)
    }
    expect(regimeCalculationFingerprint({ ...before, states: before.states.map((state) => ({ ...state, order: 1 - (state.order ?? 0) })) }, 'realtime', '')).not.toBe(original)
  })

  it('does not make a frozen preview stale merely by saving its definition revision', () => {
    const before = fixture()
    expect(regimeCalculationFingerprint({ ...before, id: 'saved-id', revision: 2 }, 'realtime', '')).toBe(regimeCalculationFingerprint(before, 'realtime', ''))
  })
})

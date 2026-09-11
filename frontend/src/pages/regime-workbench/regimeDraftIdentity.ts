import { definitionForRequest, type RegimeGraphDefinition, type RegimeMode } from '../../services/regimeGraph'

function stable(value: unknown): string {
  return JSON.stringify(value, (_key, item) => item && typeof item === 'object' && !Array.isArray(item)
    ? Object.fromEntries(Object.entries(item).sort(([left], [right]) => left.localeCompare(right)))
    : item)
}

// UI freshness only. Never substitute this fingerprint for server hashes or compile tokens.
export function regimeCalculationFingerprint(definition: RegimeGraphDefinition, mode: RegimeMode, asOf: string) {
  const payload = definitionForRequest(definition)
  const { id: _id, revision: _revision, created_at: _created, updated_at: _updated, name: _name, description: _description, template_id: _template, default_mode: _defaultMode, evaluation_targets: _evaluation, ...calculation } = payload
  const { channel_metadata: _channels, ...graph } = payload.graph
  return stable({
    ...calculation,
    graph: { ...graph, nodes: payload.graph.nodes.map(({ label: _label, position: _position, ...node }) => node) },
    // Role, order and identifiers may affect model state mapping and must remain computational.
    states: payload.states.map(({ label: _label, color: _color, ...state }) => state),
    mode,
    as_of: asOf || null,
  })
}

export function regimePresentationFingerprint(definition: RegimeGraphDefinition) {
  return stable({
    name: definition.name,
    description: definition.description,
    channels: definition.graph.channel_metadata,
    nodes: definition.graph.nodes.map(({ id, label }) => ({ id, label })),
    states: definition.states.map(({ id, label, color }) => ({ id, label, color })),
  })
}

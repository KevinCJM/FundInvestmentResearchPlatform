import type { PageEvidenceSnapshot } from './agent'
import { newPageSnapshotId } from './agentPageEvidence'
import { cloneRegimeGraphDefinition, createBlankRegimeDefinition, definitionForRequest, type RegimeGraphDefinition, type RegimeMode } from './regimeGraph'
import { invalidateStudyQualification } from '../pages/regime-workbench/regimeStudy'

export function regimeAgentSnapshot(definition: RegimeGraphDefinition | undefined, mode: RegimeMode, asOf: string, pending: boolean, selectedNodeId = ''): PageEvidenceSnapshot {
  return { version: 1, snapshot_id: newPageSnapshotId(), captured_at: new Date().toISOString(), page: 'regime-workbench',
    sections: { editing: { definition: definition?.graph.nodes.length ? definitionForRequest(definition) : null,
      mode, as_of: asOf || null, editor_pending: pending, selected_node_id: selectedNodeId } } }
}

/** AI owns only authored graph fields; saved identity, reference and validation stay with the editor. */
export function mergeRegimeAgentDraft(current: RegimeGraphDefinition | undefined, proposal: Record<string, unknown>): RegimeGraphDefinition {
  const base = current || createBlankRegimeDefinition()
  const authored = proposal as unknown as Pick<RegimeGraphDefinition, 'name' | 'description' | 'graph' | 'states'>
  const positions = new Map(base.graph.nodes.map(node => [node.id, node.position]))
  return invalidateStudyQualification(cloneRegimeGraphDefinition({ ...base,
    name: authored.name, description: authored.description || '', states: authored.states,
    graph: { ...authored.graph,
      nodes: authored.graph.nodes.map(node => ({ ...node, position: positions.get(node.id) })),
      channel_metadata: Object.fromEntries(Object.entries(base.graph.channel_metadata || {}).filter(([key]) => key in authored.graph.outputs)),
    },
  }))
}

import { cloneRegimeGraphDefinition, createBlankRegimeDefinition, type RegimeGraphDefinition } from './regimeGraph'
import { invalidateStudyQualification } from '../pages/regime-workbench/regimeStudy'

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

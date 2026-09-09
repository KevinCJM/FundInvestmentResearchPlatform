import type { AuthoringGraph, AuthoringNode, GraphBinding, GraphDocument, GraphValueType } from '../../services/indicatorGraph'

export function boundConstant(graph: AuthoringGraph, binding?: GraphBinding) {
  if (binding?.source !== 'node') return undefined
  const node = graph.nodes.find(item => item.id === binding.node_id)
  return node?.kind === 'constant' ? node : undefined
}

export function constantType(node: AuthoringNode | undefined): GraphValueType | undefined {
  if (node?.kind !== 'constant') return undefined
  return { kind: 'scalar', dtype: typeof node.value === 'boolean' ? 'bool' : 'float64', display: 'scalar', axes: [] }
}

/** Convert legacy inline bindings and newly entered values into real editable nodes.
 * Values are never deduplicated here: sharing is an explicit user connection.
 */
export function materializeConstantNodes(document: GraphDocument): GraphDocument {
  const used = new Set([...document.graph.nodes.map(node => node.id), ...document.graph.outputs.map(output => `output_${output.id}`)])
  const constants: AuthoringNode[] = []
  const positions = { ...document.positions }
  const nodes = document.graph.nodes.map(node => {
    if (node.kind !== 'operator') return node
    let changed = false
    const bindings = Object.entries(node.arguments).map(([parameter, binding], index) => {
      if (binding.source !== 'constant') return [parameter, binding] as const
      changed = true
      const base = `c_${node.id.slice(0, 40)}_${parameter.slice(0, 24)}`
      let id = base, suffix = 1
      while (used.has(id)) id = `${base}_${suffix++}`
      used.add(id)
      constants.push({ id, kind: 'constant', value: binding.value })
      const owner = positions[node.id] || { x: 280, y: 0 }
      positions[id] = { x: owner.x - 250, y: owner.y + 150 + index * 150 }
      return [parameter, { source: 'node' as const, node_id: id }] as const
    })
    return changed ? { ...node, arguments: Object.fromEntries(bindings) } : node
  })
  return constants.length ? { ...document, positions, graph: { ...document.graph, nodes: [...nodes, ...constants] } } : document
}

/** Replacing a parameter removes only its now-unreferenced constant, not a shared input. */
export function replaceGraphNode(graph: AuthoringGraph, replacement: AuthoringNode): AuthoringGraph {
  const previous = graph.nodes.find(node => node.id === replacement.id)
  const removable = new Set(previous?.kind === 'operator'
    ? Object.values(previous.arguments).flatMap(binding => {
      const constant = boundConstant(graph, binding)
      return constant ? [constant.id] : []
    }) : [])
  const nodes = graph.nodes.map(node => node.id === replacement.id ? replacement : node)
  for (const node of nodes) {
    if (node.kind === 'operator') for (const binding of Object.values(node.arguments)) {
      if (binding.source === 'node') removable.delete(binding.node_id)
    }
  }
  graph.outputs.forEach(output => { if (output.node_id) removable.delete(output.node_id) })
  return { ...graph, nodes: nodes.filter(node => !removable.has(node.id)) }
}

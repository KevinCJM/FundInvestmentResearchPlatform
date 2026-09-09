import { cloneRegimeGraphDefinition, type RegimeGraphDefinition, type RegimeGraphNode, type RegimeNodeSchema } from '../../services/regimeGraph'
import type { CanvasConnection, CanvasEdge } from '../../components/computation-graph/types'

export function regimeConnectionIssue(nodes: RegimeGraphNode[], schemas: RegimeNodeSchema[], connection: CanvasConnection): string | null {
  if (connection.source === connection.target) return '不能连接节点自身'
  const source = nodes.find((node) => node.id === connection.source)
  const target = nodes.find((node) => node.id === connection.target)
  const output = schemas.find((schema) => schema.id === source?.type || schema.type === source?.type)?.outputs.find((port) => port.id === connection.sourcePort)
  const input = schemas.find((schema) => schema.id === target?.type || schema.type === target?.type)?.inputs.find((port) => port.id === connection.targetPort)
  if (!source || !target || !input || !output) return '找不到对应输入或输出端口'
  if (output.value_type !== input.value_type) return '数据类型不兼容'
  const seen = new Set<string>()
  const pending = [connection.source]
  while (pending.length) {
    const id = pending.pop()!
    if (id === connection.target) return '会形成循环依赖'
    if (seen.has(id)) continue
    seen.add(id)
    const node = nodes.find((entry) => entry.id === id)
    for (const upstream of Object.values(node?.inputs || {})) pending.push(upstream.node_id)
  }
  return null
}

export function layoutRegimeGraph(nodes: RegimeGraphNode[], edges: CanvasEdge[]) {
  const levels = new Map<string, number>()
  const unresolved = new Set(nodes.map((node) => node.id))
  for (let pass = 0; pass < nodes.length && unresolved.size; pass += 1) {
    for (const node of nodes) {
      if (!unresolved.has(node.id)) continue
      const parents = edges.filter((edge) => edge.target === node.id).map((edge) => edge.source)
      if (parents.some((id) => unresolved.has(id))) continue
      levels.set(node.id, parents.length ? Math.max(...parents.map((id) => levels.get(id) ?? 0)) + 1 : 0)
      unresolved.delete(node.id)
    }
  }
  const rows = new Map<number, number>()
  return nodes.map((node) => {
    const level = levels.get(node.id)
    if (level == null) return { type: 'position' as const, id: node.id, position: node.position || { x: 0, y: 0 } }
    const row = rows.get(level) || 0
    rows.set(level, row + 1)
    return { type: 'position' as const, id: node.id, position: { x: level * 280, y: row * 180 } }
  })
}

/** A user-approved graph edit: keep date alignment visible, serializable and undoable. */
export function alignRegimeBinaryInputs(definition: RegimeGraphDefinition, nodeIds: string[], version = 1): RegimeGraphDefinition {
  const next = cloneRegimeGraphDefinition(definition)
  const ids = new Set(next.graph.nodes.map(node => node.id))
  for (const id of new Set(nodeIds)) {
    const node = next.graph.nodes.find(item => item.id === id)
    if (!node || !['math.add', 'math.subtract', 'math.multiply', 'math.divide'].includes(node.type)) continue
    const { left, right } = node.inputs
    if (!left || !right || !ids.has(left.node_id) || !ids.has(right.node_id)) continue
    if (left.node_id === right.node_id && next.graph.nodes.find(item => item.id === left.node_id)?.type === 'align.strict_intersection') continue
    if (next.graph.nodes.length >= 128) throw new Error('计算图已达到 128 个节点，请先删除不用的节点再添加日期对齐。')
    let sequence = 1
    while (ids.has(`align_dates_${sequence}`)) sequence++
    const alignmentId = `align_dates_${sequence}`
    ids.add(alignmentId)
    next.graph.nodes.splice(next.graph.nodes.indexOf(node), 0, {
      id: alignmentId, type: 'align.strict_intersection', type_version: version,
      label: `${node.label || '运算'} · 共同日期对齐`, parameters: {}, inputs: { left, right },
      position: { x: (node.position?.x ?? 280) - 280, y: (node.position?.y ?? 0) + 180 },
    })
    node.inputs = { ...node.inputs, left: { node_id: alignmentId, port: 'left' }, right: { node_id: alignmentId, port: 'right' } }
  }
  next.graph.edges = next.graph.nodes.flatMap(node => Object.entries(node.inputs).map(([port, source]) => ({ source: { ...source }, target: { node_id: node.id, port } })))
  return next
}

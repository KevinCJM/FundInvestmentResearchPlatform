import type { EtlDefinition, EtlStep } from '../../services/etl'
import { blankStep } from '../../services/etl'
import type { CanvasConnection, CanvasEdge, GraphNodeSchema } from '../computation-graph/types'
import { cloneDocument } from '../computation-graph/history'
import { graphOrder, layoutGraph } from '../computation-graph/graph'

/** Show legacy sequence as explicit control edges, without changing saved data. */
export function asGraph(definition: EtlDefinition): EtlDefinition {
  if (definition.graph_version === 1) return definition
  return { ...definition, graph_version: 1, steps: definition.steps.map((step, index) => ({ ...step,
    after: index && !step.inputs.includes(definition.steps[index - 1].id) ? [...new Set([...(step.after ?? []), definition.steps[index - 1].id])] : step.after ?? [],
  })) }
}
export function etlEdges(definition: EtlDefinition): CanvasEdge[] {
  return definition.steps.flatMap(step => [
    ...step.inputs.map(source => ({ id: `data:${source}>${step.id}`, source, sourcePort: 'data', target: step.id, targetPort: 'data', kind: 'data' as const })),
    ...(step.after ?? []).map(source => ({ id: `control:${source}>${step.id}`, source, sourcePort: 'done', target: step.id, targetPort: 'after', kind: 'control' as const })),
  ])
}
export function etlPositions(definition: EtlDefinition) {
  return { ...layoutGraph(definition.steps.map(step => step.id), etlEdges(definition)), ...definition.canvas?.positions }
}
export function connectionProblem(definition: EtlDefinition, connection: CanvasConnection, schemas: GraphNodeSchema[]): string | null {
  if (connection.source === connection.target) return '节点不能连接自身。'
  const source = definition.steps.find(step => step.id === connection.source)
  const target = definition.steps.find(step => step.id === connection.target)
  if (!source || !target) return '连线节点不存在。'
  const output = schemas.find(s => s.id === source.kind)?.outputs.find(p => p.id === connection.sourcePort)
  const input = schemas.find(s => s.id === target.kind)?.inputs.find(p => p.id === connection.targetPort)
  if (!output || !input || input.value_type !== output.value_type) return '端口类型不同，不能传递这种数据。'
  const references = connection.targetPort === 'after' ? target.after ?? [] : target.inputs
  if (references.includes(source.id)) return '该连线已经存在。'
  if (!input.multiple && references.length) return '该输入只能连接一个上游，请先断开原连线。'
  try { graphOrder(definition.steps.map(s => s.id), [...etlEdges(definition), connection]) }
  catch (error) { return error instanceof Error ? error.message : '依赖无效。' }
  return null
}
export function connectEtl(definition: EtlDefinition, connection: CanvasConnection, schemas: GraphNodeSchema[]): EtlDefinition {
  const problem = connectionProblem(definition, connection, schemas)
  if (problem) throw new Error(problem)
  const key = connection.targetPort === 'after' ? 'after' : 'inputs'
  return { ...definition, graph_version: 1, steps: definition.steps.map(step => step.id === connection.target ? { ...step, [key]: [...(step[key] ?? []), connection.source] } : step) }
}
export function removeEtlEdges(definition: EtlDefinition, ids: string[]): EtlDefinition {
  const removed = new Set(ids)
  return { ...definition, steps: definition.steps.map(step => ({ ...step,
    inputs: step.inputs.filter(source => !removed.has(`data:${source}>${step.id}`)),
    after: (step.after ?? []).filter(source => !removed.has(`control:${source}>${step.id}`)),
  })) }
}
export function removeEtlNodes(definition: EtlDefinition, ids: string[]): EtlDefinition {
  const removed = new Set(ids)
  return { ...definition, steps: definition.steps.filter(s => !removed.has(s.id)).map(s => ({ ...s, inputs: s.inputs.filter(id => !removed.has(id)), after: (s.after ?? []).filter(id => !removed.has(id)) })),
    canvas: definition.canvas ? { ...definition.canvas, positions: Object.fromEntries(Object.entries(definition.canvas.positions).filter(([id]) => !removed.has(id))) } : null }
}
export function duplicateEtlNodes(definition: EtlDefinition, ids: string[]): EtlDefinition {
  const selected = definition.steps.filter(step => ids.includes(step.id))
  if (definition.steps.length + selected.length > 40) throw new Error('流程最多 40 个节点，请减少复制数量。')
  const remap = new Map(selected.map(step => [step.id, blankStep(step.kind).id]))
  const positions = etlPositions(definition)
  const copies: EtlStep[] = selected.map(step => {
    const id = remap.get(step.id)!
    positions[id] = { x: positions[step.id].x + 44, y: positions[step.id].y + 44 }
    return { ...cloneDocument(step), id, name: `${step.name.slice(0, 94)} 副本`, inputs: step.inputs.map(key => remap.get(key) ?? key), after: (step.after ?? []).map(key => remap.get(key) ?? key) }
  })
  return { ...definition, steps: [...definition.steps, ...copies], canvas: { version: 1, ...definition.canvas, positions } }
}

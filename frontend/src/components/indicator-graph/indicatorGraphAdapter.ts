import type { IndicatorDraft, IndicatorOperator, IndicatorOperatorParameter, IndicatorVariable } from '../../services/customIndicators'
import type { AuthoringGraph, AuthoringNode, GraphBinding, GraphDiagnostic, GraphDocument, GraphOutput, GraphValueType, OperatorNode } from '../../services/indicatorGraph'
import type { CanvasConnection, CanvasEdge, CanvasNode, GraphNodeSchema } from '../computation-graph/types'
import { graphOrder } from '../computation-graph/graph'
import { businessText, systemText } from '../../i18n/runtime'
import { boundConstant, constantType, replaceGraphNode } from './indicatorGraphConstants'
import { graphConstantLabel, graphOperatorLabel, graphParameterLabel, graphTypeLabel, graphVariableLabel } from './indicatorGraphPresentation'

export const outputNodeId = (id: string) => `output_${id}`
export function portsForNode(node: AuthoringNode, operators: IndicatorOperator[], types: Record<string, GraphValueType> = {}): Array<{ id: string; label: string; type?: GraphValueType }> {
  return [{ id: 'value', label: systemText('graph.result'), type: types[node.id] || constantType(node) }]
}
export const nodePortKey = (nodeId: string, portId = 'value') => portId === 'value' ? nodeId : `${nodeId}::${portId}`
export function nodePortRef(key: string): { node_id: string; port_id?: string } {
  const [node_id, port_id] = key.split('::')
  return port_id ? { node_id, port_id } : { node_id }
}

export const newNodeId = () => `n_${crypto.randomUUID().replace(/-/g, '')}`
export const graphSignature = (graph: AuthoringGraph) => JSON.stringify({ ...graph, nodes: graph.nodes.map(({ label: _label, ...node }) => node) })
export const emptyOutput = (id: string, label: string): GraphOutput => ({ id, label, node_id: null, unit: '', display_format: 'number', precision: 4, output_measure: 'auto' })
export const outputPresentation = (output: Partial<GraphOutput> & Pick<GraphOutput, 'id' | 'label'>): Omit<GraphOutput, 'node_id'> => ({ id: output.id, label: output.label, unit: output.unit ?? '', display_format: output.display_format ?? 'number', precision: output.precision ?? 4, output_measure: output.output_measure ?? 'auto', ...(output.direction ? { direction: output.direction } : {}), ...(output.description !== undefined ? { description: output.description } : {}) })
export const emptyDocument = (draft: IndicatorDraft): GraphDocument => ({
  graph: { graph_version: 1, nodes: [], outputs: draft.result_kind === 'time_series'
      ? (draft.series_outputs?.length ? draft.series_outputs : [{ id: 'value', label: '输出通道' }]).map(output => ({ ...emptyOutput(output.id, output.label), ...outputPresentation(output) }))
      : [emptyOutput('result', '最终结果')] }, positions: {},
})
export function parametersForNode(node: OperatorNode, operators: IndicatorOperator[]): IndicatorOperatorParameter[] {
  const spec = operators.find(operator => operator.name === node.operator_id)
  if (!spec) return Object.keys(node.arguments).map(name => ({ name, label: name }))
  const arity = node.arity ?? Object.keys(node.arguments).length
  const sets = (spec.parameter_sets ?? []).filter(set => set.arity === arity)
  if (sets.length) return sets[0].parameters.map((parameter, index) => ({
    ...parameter,
    allowed_shapes: [...new Set(sets.flatMap(set => set.parameters[index]?.allowed_shapes ?? (set.parameters[index]?.shape ? [set.parameters[index].shape!] : [])))],
  }))
  return spec.parameters?.slice(0, arity || undefined) ?? (spec.input_shapes ?? []).map((shape, index) => ({ name: `input_${index + 1}`, label: `输入 ${index + 1}`, shape }))
}
export function makeOperator(operator: IndicatorOperator): OperatorNode {
  const minSet = [...(operator.parameter_sets ?? [])].sort((a, b) => a.arity - b.arity)[0]
  const parameters = minSet?.parameters ?? operator.parameters ?? []
  return { id: newNodeId(), kind: 'operator', operator_id: operator.name, arity: minSet?.arity ?? parameters.length, arguments: Object.fromEntries(parameters.flatMap(parameter =>
    typeof parameter.default === 'number' ? [[parameter.name, { source: 'constant', value: parameter.default }]] : [],
  )) }
}
export const nodeLabel = (node: AuthoringNode, variables: IndicatorVariable[], operators: IndicatorOperator[]) => node.label || (
  node.kind === 'variable' ? graphVariableLabel(node.variable_id, variables)
    : node.kind === 'operator' ? graphOperatorLabel(node.operator_id, operators)
      : node.kind === 'parameter' ? node.parameter_id
      : node.value === null ? systemText('graph.constantRequired', {}, '待填写常量') : systemText('graph.constantNode', { value: graphConstantLabel(node.value) }, '常量 {{value}}')
)
export function graphEdges(graph: AuthoringGraph): CanvasEdge[] {
  return [
    ...graph.nodes.flatMap(node => node.kind === 'operator' ? Object.entries(node.arguments).flatMap(([name, binding]) => binding.source === 'node' ? [{
      id: `${node.id}:${name}`, source: binding.node_id, sourcePort: binding.port_id || 'value', target: node.id, targetPort: name,
    }] : []) : []),
    ...graph.outputs.flatMap(output => output.node_id ? [{ id: `${outputNodeId(output.id)}:value`, source: output.node_id, sourcePort: output.port_id || 'value', target: outputNodeId(output.id), targetPort: 'value' }] : []),
  ]
}
export function graphConnectionIssue(graph: AuthoringGraph, connection: CanvasConnection, operators: IndicatorOperator[], types: Record<string, GraphValueType> = {}): string | null {
  const source = graph.nodes.find(node => node.id === connection.source)
  const selectedPort = source ? portsForNode(source, operators, types).find(port => port.id === connection.sourcePort) : undefined
  if (!selectedPort) return '请选择有效的上游计算结果。'
  const output = graph.outputs.find(item => outputNodeId(item.id) === connection.target)
  const target = graph.nodes.find(node => node.id === connection.target)
  if (output) return source?.kind === 'parameter' ? systemText('indicatorParameters.connectionError') : connection.targetPort === 'value' ? null : '最终结果只有一个输入。'
  if (!target || target.kind !== 'operator') return '只能连接到计算节点的输入。'
  const parameter = parametersForNode(target, operators).find(item => item.name === connection.targetPort)
  if (!parameter) return '此输入不属于当前算子签名。'
  const sourceNode = graph.nodes.find(node => node.id === connection.source)
  if (sourceNode?.kind === 'parameter' && !parameter.parameterizable) return systemText('indicatorParameters.connectionError')
  if (parameter.source_policy === 'fixed_constant' && sourceNode?.kind !== 'constant' && !(sourceNode?.kind === 'parameter' && parameter.parameterizable)) return '此参数只能连接固定常量节点或已开放参数，不能连接行情变量或计算结果。'
  const sourceType = selectedPort.type?.kind
  const allowed = parameter.allowed_shapes?.length ? parameter.allowed_shapes : parameter.shape ? [parameter.shape] : []
  if (sourceType && allowed.length && !allowed.includes(sourceType as never) && !allowed.includes('unknown')) return '该上游结果的形状与输入要求不兼容。'
  try {
    graphOrder(graph.nodes.map(node => node.id), [
      ...graphEdges(graph).filter(edge => !edge.target.startsWith('output_') && !(edge.target === connection.target && edge.targetPort === connection.targetPort)), connection,
    ])
  } catch (failure) { return failure instanceof Error ? failure.message : '连接不能形成循环。' }
  return null
}
export function connectGraph(graph: AuthoringGraph, connection: CanvasConnection): AuthoringGraph {
  const target = graph.nodes.find(node => node.id === connection.target)
  if (target?.kind === 'operator') return replaceGraphNode(graph, { ...target, arguments: { ...target.arguments, [connection.targetPort]: { source: 'node', node_id: connection.source, ...(connection.sourcePort !== 'value' ? { port_id: connection.sourcePort } : {}) } } })
  return { ...graph, outputs: graph.outputs.map(output => outputNodeId(output.id) === connection.target ? { ...output, node_id: connection.source, port_id: connection.sourcePort } : output) }
}
export function removeGraphNodes(graph: AuthoringGraph, ids: string[]): AuthoringGraph {
  const removed = new Set(ids)
  return { ...graph,
    nodes: graph.nodes.filter(node => !removed.has(node.id)).map(node => node.kind === 'operator' ? { ...node, arguments: Object.fromEntries(Object.entries(node.arguments).filter(([, binding]) => binding.source !== 'node' || !removed.has(binding.node_id))) } : node),
    outputs: graph.outputs.map(output => removed.has(output.node_id ?? '') ? { ...output, node_id: null } : output),
  }
}
export function removeGraphEdges(graph: AuthoringGraph, ids: string[]): AuthoringGraph {
  const removed = new Set(ids)
  return { ...graph,
    nodes: graph.nodes.map(node => node.kind === 'operator' ? { ...node, arguments: Object.fromEntries(Object.entries(node.arguments).filter(([name]) => !removed.has(`${node.id}:${name}`))) } : node),
    outputs: graph.outputs.map(output => removed.has(`${outputNodeId(output.id)}:value`) ? { ...output, node_id: null } : output),
  }
}
export function duplicateGraphNodes(document: GraphDocument, ids: string[]): GraphDocument {
  const selectedIds = new Set(ids)
  for (const node of document.graph.nodes) {
    if (selectedIds.has(node.id) && node.kind === 'operator') for (const binding of Object.values(node.arguments)) {
      const constant = boundConstant(document.graph, binding)
      if (constant) selectedIds.add(constant.id)
    }
  }
  const selected = document.graph.nodes.filter(node => selectedIds.has(node.id))
  const mapping = new Map(selected.map(node => [node.id, newNodeId()]))
  const positions = { ...document.positions }
  const copies = selected.map(node => {
    const id = mapping.get(node.id)!
    const position = positions[node.id] ?? { x: 0, y: 0 }
    positions[id] = { x: position.x + 40, y: position.y + 60 }
    return node.kind === 'operator' ? { ...node, id, arguments: Object.fromEntries(Object.entries(node.arguments).map(([name, binding]) => [name, binding.source === 'node' ? { ...binding, node_id: mapping.get(binding.node_id) ?? binding.node_id } : { ...binding }])) } : { ...node, id }
  })
  return { ...document, positions, graph: { ...document.graph, nodes: [...document.graph.nodes, ...copies] } }
}
export function layoutDocument(document: GraphDocument): GraphDocument {
  const ids = [...document.graph.nodes.map(node => node.id), ...document.graph.outputs.map(output => outputNodeId(output.id))]
  const edges = graphEdges(document.graph)
  const incoming = new Map(ids.map(id => [id, edges.filter(edge => edge.target === id).map(edge => edge.source)]))
  let order: string[]
  try { order = graphOrder(ids, edges) } catch { order = ids }
  const depth = new Map<string, number>()
  const layers: string[][] = []
  const outputs = new Set(document.graph.outputs.map(output => outputNodeId(output.id)))
  for (const id of order) {
    const level = outputs.has(id) ? 1 + Math.max(0, ...depth.values()) : Math.max(-1, ...(incoming.get(id) || []).map(parent => depth.get(parent) ?? -1)) + 1
    depth.set(id, level)
    if (!outputs.has(id)) (layers[level] ??= []).push(id)
  }
  const outputLevel = 1 + Math.max(-1, ...document.graph.nodes.map(node => depth.get(node.id) ?? 0))
  layers[outputLevel] = [...outputs]
  const nodesById = new Map(document.graph.nodes.map(node => [node.id, node]))
  const heightOf = (id: string) => {
    const node = nodesById.get(id)
    return node?.kind === 'constant' ? 76 : node?.kind === 'operator' ? 112 + Math.max(Object.keys(node.arguments).length, new Set(edges.filter(edge => edge.source === id).map(edge => edge.sourcePort)).size) * 24 : 132
  }
  const heights = layers.map(layer => layer?.reduce((height, id) => height + heightOf(id) + 28, 0) || 0)
  const tallest = Math.max(1, ...heights)
  const positions: GraphDocument['positions'] = {}
  layers.forEach((layer, column) => {
    let y = (tallest - heights[column]) / 2
    layer?.forEach(id => {
      positions[id] = { x: column * 285 + (nodesById.get(id)?.kind === 'constant' ? 78 : 0), y }
      y += heightOf(id) + 28
    })
  })
  return { ...document, viewport: null, positions }
}
export function localGraphIssues(graph: AuthoringGraph, operators: IndicatorOperator[]): GraphDiagnostic[] {
  const issues: GraphDiagnostic[] = []
  const invalidConstant = (binding: GraphBinding | undefined) => binding?.source === 'constant' && (binding.value === null || (typeof binding.value === 'number' && !Number.isFinite(binding.value)))
  for (const node of graph.nodes) {
    if (node.kind === 'constant' && invalidConstant({ source: 'constant', value: node.value })) issues.push({ code: 'INVALID_CONSTANT', message: '请填写有限数值。', editor_node_id: node.id })
    if (node.kind !== 'operator') continue
    for (const parameter of parametersForNode(node, operators)) {
      const binding = node.arguments[parameter.name]
      const constant = boundConstant(graph, binding)
      if (!binding || invalidConstant(binding) || (constant && invalidConstant({ source: 'constant', value: constant.value }))) issues.push({ code: 'MISSING_ARGUMENT', message: `请配置“${graphParameterLabel(parameter)}”。`, editor_node_id: node.id, parameter_id: parameter.name })
    }
  }
  const outputIds = new Set<string>()
  for (const output of graph.outputs) {
    if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(output.id) || outputIds.has(output.id)) issues.push({ code: 'INVALID_OUTPUT_ID', message: '通道 ID 必须唯一，且只包含字母、数字和下划线。', editor_node_id: outputNodeId(output.id) })
    outputIds.add(output.id)
    if (!output.node_id) issues.push({ code: 'OUTPUT_NOT_CONNECTED', message: `请为“${output.label || output.id}”选择最终结果。`, editor_node_id: outputNodeId(output.id) })
  }
  return issues
}
export function canvasModel(document: GraphDocument, variables: IndicatorVariable[], operators: IndicatorOperator[], types: Record<string, GraphValueType> = {}, issues: GraphDiagnostic[] = []): { nodes: CanvasNode[]; edges: CanvasEdge[]; schemas: GraphNodeSchema[] } {
  const badNodes = new Set(issues.map(issue => issue.editor_node_id))
  const nodes: CanvasNode[] = document.graph.nodes.map(node => ({
    id: node.id, type: node.id, appearance: node.kind, label: nodeLabel(node, variables, operators), position: document.positions[node.id],
    inputs: node.kind === 'operator' ? node.arguments : {}, ready: !badNodes.has(node.id),
    statusLabel: graphTypeLabel(types[node.id] || constantType(node)),
  }))
  const schemas: GraphNodeSchema[] = document.graph.nodes.map(node => ({
    id: node.id, label: nodeLabel(node, variables, operators), category_label: businessText(`nodeKinds.${node.kind}`),
    inputs: node.kind === 'operator' ? parametersForNode(node, operators).map((parameter, index) => ({ id: parameter.name, label: graphParameterLabel(parameter, index, node.operator_id), required: true, value_type: parameter.shape || 'unknown', type_label: graphTypeLabel(parameter.allowed_types?.join('|') || parameter.shape) })) : [],
    outputs: portsForNode(node, operators, types).map(port => ({ id: port.id, label: port.label, value_type: port.type?.kind || 'unknown', type_label: graphTypeLabel(port.type) })),
  }))
  for (const output of document.graph.outputs) {
    const id = outputNodeId(output.id)
    const label = output.id === 'result' && (!output.label || output.label === '最终结果') ? businessText('nodeKinds.final') : output.label || systemText('graph.unnamedOutput', {}, '未命名输出')
    nodes.push({ id, type: id, appearance: 'output', label, position: document.positions[id], inputs: output.node_id ? { value: output.node_id } : {}, ready: Boolean(output.node_id) && !badNodes.has(id), statusLabel: businessText('nodeKinds.output') })
    schemas.push({ id, label, category_label: businessText('nodeKinds.output'), inputs: [{ id: 'value', label: systemText('graph.outputSource'), required: true, value_type: 'unknown', type_label: businessText('nodeKinds.final') }], outputs: [] })
  }
  return { nodes, edges: graphEdges(document.graph), schemas }
}

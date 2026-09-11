import { describe, expect, it } from 'vitest'
import type { IndicatorOperator } from '../../services/customIndicators'
import type { AuthoringGraph, GraphDocument, OperatorNode } from '../../services/indicatorGraph'
import { canvasModel, connectGraph, duplicateGraphNodes, emptyOutput, graphConnectionIssue, graphEdges, graphSignature, layoutDocument, localGraphIssues, makeOperator, parametersForNode, removeGraphEdges, removeGraphNodes } from './indicatorGraphAdapter'

const operators: IndicatorOperator[] = [
  { name: 'mean', label: '平均值', signature: 'mean(x)', latex_template: 'mean(x)', return_type: 'scalar', parameters: [{ name: 'x', label: '输入值', shape: 'series' }], parameter_sets: [{ arity: 1, parameters: [{ name: 'x', label: '输入值', shape: 'series' }] }] },
  { name: 'divide', label: '相除', signature: 'divide(left, right)', latex_template: 'left/right', return_type: 'scalar', parameter_sets: [{ arity: 2, parameters: [{ name: 'left', label: '分子', shape: 'scalar' }, { name: 'right', label: '分母', shape: 'scalar' }] }] },
  { name: 'rolling_std', label: '滚动标准差', signature: 'rolling_std(x, window, ddof)', latex_template: 'rolling_std(x, window, ddof)', return_type: 'series', parameter_sets: [
    { arity: 2, parameters: [{ name: 'x', shape: 'series' }, { name: 'window', label: '窗口', source_policy: 'fixed_constant', constant_kind: 'integer', minimum: 2, default: 20 }] },
    { arity: 3, parameters: [{ name: 'x', shape: 'series' }, { name: 'window', label: '窗口', source_policy: 'fixed_constant', constant_kind: 'integer', minimum: 2, default: 20 }, { name: 'ddof', source_policy: 'fixed_constant', default: 1 }] },
  ] },
]
const graph = (): AuthoringGraph => ({ graph_version: 1, nodes: [
  { id: 'r', kind: 'variable', variable_id: 'returns' },
  { id: 'm', kind: 'operator', operator_id: 'mean', arity: 1, arguments: { x: { source: 'node', node_id: 'r' } } },
  { id: 'd', kind: 'operator', operator_id: 'divide', arity: 2, arguments: { left: { source: 'node', node_id: 'm' }, right: { source: 'node', node_id: 'm' } } },
], outputs: [{ ...emptyOutput('result', '最终结果'), node_id: 'd' }] })

describe('indicator authoring graph adapter', () => {
  it('preserves two named operands from one shared upstream', () => {
    const edges = graphEdges(graph()).filter(edge => edge.target === 'd')
    expect(edges.map(edge => edge.targetPort)).toEqual(['left', 'right'])
    expect(new Set(edges.map(edge => edge.id)).size).toBe(2)
    expect(edges.every(edge => edge.source === 'm')).toBe(true)
  })
  it('replacing one operand never changes the other', () => {
    const changed = connectGraph(graph(), { source: 'r', sourcePort: 'value', target: 'd', targetPort: 'left' })
    const divide = changed.nodes[2] as OperatorNode
    expect(divide.arguments.left).toEqual({ source: 'node', node_id: 'r' })
    expect(divide.arguments.right).toEqual({ source: 'node', node_id: 'm' })
  })
  it('rejects cycles and accepts shared acyclic inputs', () => {
    expect(graphConnectionIssue(graph(), { source: 'd', sourcePort: 'value', target: 'm', targetPort: 'x' }, operators)).toBeTruthy()
    expect(graphConnectionIssue(graph(), { source: 'm', sourcePort: 'value', target: 'd', targetPort: 'right' }, operators)).toBeNull()
  })
  it('removing one named edge leaves the parallel operand intact', () => {
    const changed = removeGraphEdges(graph(), ['d:left'])
    expect((changed.nodes[2] as OperatorNode).arguments).toEqual({ right: { source: 'node', node_id: 'm' } })
  })
  it('deleting a shared node disconnects every consumer and undo can retain the original', () => {
    const original = graph()
    const removed = removeGraphNodes(original, ['m'])
    expect((removed.nodes[1] as OperatorNode).arguments).toEqual({})
    expect(original.nodes).toHaveLength(3)
    expect(removeGraphNodes(original, ['d']).outputs[0].node_id).toBeNull()
  })
  it('copies selected internal references but keeps external upstream references', () => {
    const original: GraphDocument = { graph: graph(), positions: {} }
    const copied = duplicateGraphNodes(original, ['m', 'd'])
    const mean = copied.graph.nodes[3] as OperatorNode
    const divide = copied.graph.nodes[4] as OperatorNode
    expect(mean.arguments.x).toEqual({ source: 'node', node_id: 'r' })
    expect(divide.arguments.left).toEqual({ source: 'node', node_id: mean.id })
    expect(divide.arguments.right).toEqual(divide.arguments.left)
    expect(copied.graph.outputs[0].node_id).toBe('d')
  })
  it('inlines fixed constants and changes optional arity by name', () => {
    const node = makeOperator(operators[2])
    expect(node.arity).toBe(2)
    expect(node.arguments.window).toEqual({ source: 'constant', value: 20 })
    expect(parametersForNode({ ...node, arity: 3 }, operators).map(parameter => parameter.name)).toEqual(['x', 'window', 'ddof'])
    const candidate = { ...graph(), nodes: [...graph().nodes, node] }
    expect(graphConnectionIssue(candidate, { source: 'm', sourcePort: 'value', target: node.id, targetPort: 'window' }, operators)).toContain('固定常量')
  })
  it('layout and node notes do not change semantic signature', () => {
    const original = graph()
    const signature = graphSignature(original)
    const changed = layoutDocument({ graph: original, positions: {} })
    changed.graph = { ...original, nodes: original.nodes.map(node => ({ ...node, label: '注释' })) }
    expect(graphSignature(changed.graph)).toBe(signature)
    changed.graph.outputs[0] = { ...changed.graph.outputs[0], precision: 6 }
    expect(graphSignature(changed.graph)).not.toBe(signature)
  })
  it('reports blank constants and roots locally, without an API call', () => {
    const invalid = { ...graph(), nodes: [...graph().nodes, { id: 'empty', kind: 'constant' as const, value: null }], outputs: [emptyOutput('result', '最终结果')] }
    expect(localGraphIssues(invalid, operators).map(issue => issue.code)).toEqual(['INVALID_CONSTANT', 'OUTPUT_NOT_CONNECTED'])
  })
  it('builds accessible ports by parameter name and synthetic final output', () => {
    const model = canvasModel({ graph: graph(), positions: {} }, [], operators)
    expect(model.schemas.find(schema => schema.id === 'd')?.inputs.map(port => port.label)).toEqual(['分子', '分母'])
    expect(model.schemas.find(schema => schema.id === 'output_result')?.outputs).toEqual([])
  })
})

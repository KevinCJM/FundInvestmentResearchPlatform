import { describe, expect, it } from 'vitest'
import type { IndicatorOperator } from '../../services/customIndicators'
import type { GraphDocument, OperatorNode } from '../../services/indicatorGraph'
import { boundConstant, materializeConstantNodes, replaceGraphNode } from './indicatorGraphConstants'
import { canvasModel, connectGraph, duplicateGraphNodes, emptyOutput, graphConnectionIssue, graphEdges } from './indicatorGraphAdapter'

const operators: IndicatorOperator[] = [{ name: 'rolling_mean', label: '滚动平均', signature: '', latex_template: '', return_type: 'series', parameters: [
  { name: 'x', shape: 'series', label: '输入值' },
  { name: 'window', shape: 'scalar', label: '窗口', source_policy: 'fixed_constant', minimum: 1, constant_kind: 'integer' },
] }]
const document = (): GraphDocument => ({ graph: { graph_version: 1, nodes: [
  { id: 'r', kind: 'variable', variable_id: 'returns' },
  { id: 'm', kind: 'operator', operator_id: 'rolling_mean', arity: 2, arguments: { x: { source: 'node', node_id: 'r' }, window: { source: 'constant', value: 5 } } },
], outputs: [{ ...emptyOutput('result', '最终结果'), node_id: 'm' }] }, positions: { m: { x: 280, y: 200 } } })

describe('independently editable constant nodes', () => {
  it('migrates inline parameters without mutating old drafts; migration is idempotent', () => {
    const old = document()
    const next = materializeConstantNodes(old)
    const operator = next.graph.nodes[1] as OperatorNode
    expect(operator.arguments.window.source).toBe('node')
    expect(boundConstant(next.graph, operator.arguments.window)?.value).toBe(5)
    expect((old.graph.nodes[1] as OperatorNode).arguments.window.source).toBe('constant')
    expect(materializeConstantNodes(next)).toBe(next)
    expect(graphEdges(next.graph).some(edge => edge.target === 'm' && edge.targetPort === 'window')).toBe(true)
  })
  it('equal parameter values are not automatically linked', () => {
    const old = document()
    old.graph.nodes.push({ ...(old.graph.nodes[1] as OperatorNode), id: 'm2' })
    const next = materializeConstantNodes(old)
    const first = next.graph.nodes[1] as OperatorNode, second = next.graph.nodes[2] as OperatorNode
    expect(first.arguments.window).not.toEqual(second.arguments.window)
  })
  it('new constant fields can be blank in an editable draft', () => {
    const old = document()
    ;(old.graph.nodes[1] as OperatorNode).arguments.window = { source: 'constant', value: null }
    const next = materializeConstantNodes(old)
    expect(next.graph.nodes.find(node => node.kind === 'constant')).toMatchObject({ value: null })
  })
  it('fixed inputs accept constant nodes but not variables or operator results', () => {
    const next = materializeConstantNodes(document())
    const constant = next.graph.nodes.find(node => node.kind === 'constant')!
    const connection = { source: constant.id, sourcePort: 'value', target: 'm', targetPort: 'window' }
    expect(graphConnectionIssue(next.graph, connection, operators)).toBeNull()
    expect(graphConnectionIssue(next.graph, { ...connection, source: 'r' }, operators)).toContain('固定常量节点')
  })
  it('copying an operator also copies its constants, while preserving variable dependencies', () => {
    const next = materializeConstantNodes(document())
    const copied = duplicateGraphNodes(next, ['m'])
    expect(copied.graph.nodes.length).toBe(next.graph.nodes.length + 2)
    const copy = copied.graph.nodes.slice(next.graph.nodes.length).find(node => node.kind === 'operator') as OperatorNode
    expect(copy.arguments.x).toEqual((next.graph.nodes[1] as OperatorNode).arguments.x)
    expect(copy.arguments.window).not.toEqual((next.graph.nodes[1] as OperatorNode).arguments.window)
    expect(boundConstant(copied.graph, copy.arguments.window)?.value).toBe(5)
  })
  it('replacing an input removes only a no-longer-referenced constant', () => {
    const next = materializeConstantNodes(document())
    const oldConstant = next.graph.nodes.find(node => node.kind === 'constant')!
    const replacement = { ...(next.graph.nodes[1] as OperatorNode), arguments: { x: { source: 'node' as const, node_id: 'r' }, window: { source: 'constant' as const, value: 10 } } }
    const changed = replaceGraphNode(next.graph, replacement)
    expect(changed.nodes.some(node => node.id === oldConstant.id)).toBe(false)
    next.graph.outputs.push({ ...emptyOutput('shared', '共享'), node_id: oldConstant.id })
    expect(replaceGraphNode(next.graph, replacement).nodes.some(node => node.id === oldConstant.id)).toBe(true)
  })
  it('rewiring to the same constant preserves it', () => {
    const next = materializeConstantNodes(document())
    const constant = next.graph.nodes.find(node => node.kind === 'constant')!
    const changed = connectGraph(next.graph, { source: constant.id, sourcePort: 'value', target: 'm', targetPort: 'window' })
    expect(changed.nodes).toEqual(next.graph.nodes)
  })
  it('marks all four presentation kinds and keeps fixed parameter ports visible', () => {
    const model = canvasModel(materializeConstantNodes(document()), [], operators)
    expect(model.nodes.map(node => node.appearance)).toEqual(['variable', 'operator', 'constant', 'output'])
    expect(model.schemas.find(schema => schema.id === 'm')?.inputs.map(port => port.id)).toEqual(['x', 'window'])
  })
})

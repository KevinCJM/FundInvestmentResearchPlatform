import { fireEvent, render, screen } from '@testing-library/react'
import type { ReactNode } from 'react'
import { describe, expect, it, vi } from 'vitest'
import RegimeGraphCanvas, { type RegimeCanvasEdge } from './RegimeGraphCanvas'

vi.mock('@xyflow/react', () => ({
  ReactFlow: ({ children, edges, nodes, onNodeDragStop, onNodesChange }: {
    children?: ReactNode
    edges: unknown[]
    nodes: Array<{ id: string; position: { x: number; y: number } }>
    onNodeDragStop?: (event: unknown, node: unknown, nodes: unknown[]) => void
    onNodesChange?: (changes: unknown[]) => void
  }) => {
    const moved = { ...nodes[0], position: { x: 80, y: 40 } }
    return <div>
      <span data-testid="rendered-edge-count">{edges.length}</span>
      <span data-testid="rendered-node-position">{`${nodes[0].position.x},${nodes[0].position.y}`}</span>
      <button type="button" onClick={() => onNodesChange?.([{ type: 'position', id: moved.id, position: moved.position, dragging: true }])}>模拟拖动中</button>
      <button type="button" onClick={() => onNodeDragStop?.({}, moved, [moved])}>模拟拖动结束</button>
      {children}
    </div>
  },
  Background: () => null,
  Controls: () => null,
  MiniMap: () => null,
  Handle: () => null,
  Position: { Left: 'left', Right: 'right' },
  applyNodeChanges: (changes: Array<{ id: string; type: string; position?: { x: number; y: number } }>, nodes: Array<{ id: string; position: { x: number; y: number } }>) => nodes.map((node) => {
    const position = changes.find((change) => change.type === 'position' && change.id === node.id)?.position
    return position ? { ...node, position } : node
  }),
  useUpdateNodeInternals: () => () => undefined,
}))

const nodes = [
  { id: 'source', type: 'source.series', label: '数据源', position: { x: 0, y: 0 }, parameters: {}, inputs: {} },
  { id: 'model', type: 'decoder.threshold', label: '状态模型', position: { x: 260, y: 0 }, parameters: {}, inputs: { value: { node_id: 'source', port: 'value' } } },
]

const edges: RegimeCanvasEdge[] = [
  { id: 'source:value->model:value', source: 'source', sourcePort: 'value', target: 'model', targetPort: 'value' },
]

const schemas = [
  { id: 'source.series', label: '数据源', category_label: '数据源', inputs: [], outputs: [{ id: 'value', label: '数值序列', value_type: 'numeric_series' }] },
  { id: 'decoder.threshold', label: '状态模型', category_label: '状态识别模型', inputs: [{ id: 'value', label: '数值序列', value_type: 'numeric_series', required: true }], outputs: [{ id: 'state', label: '状态序列', value_type: 'state_codes' }] },
]

describe('RegimeGraphCanvas', () => {
  it('拖动期间保持本地节点与连线，只在结束时固化位置', () => {
    const onNodesChange = vi.fn()
    render(<RegimeGraphCanvas nodes={nodes} edges={edges} schemas={schemas} selectedNodeId="source" onNodesChange={onNodesChange} onConnect={vi.fn()} onDuplicate={vi.fn()} />)

    fireEvent.click(screen.getByRole('button', { name: '模拟拖动中' }))
    expect(screen.getByTestId('rendered-node-position')).toHaveTextContent('80,40')
    expect(screen.getByTestId('rendered-edge-count')).toHaveTextContent('1')
    expect(onNodesChange).not.toHaveBeenCalledWith(expect.arrayContaining([expect.objectContaining({ type: 'position' })]))

    fireEvent.click(screen.getByRole('button', { name: '模拟拖动结束' }))
    expect(onNodesChange).toHaveBeenCalledWith([{ type: 'position', id: 'source', position: { x: 80, y: 40 } }])
  })
})

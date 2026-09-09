import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import type { AuthoringGraph } from '../../services/indicatorGraph'
import { drawdownOperator, drawdownVariables } from '../../test/drawdownFixtures'
import { canvasModel, connectGraph, graphConnectionIssue, graphEdges } from './indicatorGraphAdapter'
import IndicatorNodeInspector from './IndicatorNodeInspector'

const graph: AuthoringGraph = {
  graph_version: 1,
  nodes: [
    { id: 'nav', kind: 'variable', variable_id: 'adjusted_nav' },
    { id: 'dd', kind: 'operator', operator_id: 'drawdown_analysis', arity: 1, arguments: { values: { source: 'node', node_id: 'nav' } } },
  ],
  outputs: [{ id: 'mdd', label: '最大回撤', node_id: 'dd', port_id: 'max_drawdown', unit: '%', precision: 2, display_format: 'percent', output_measure: 'auto' }],
}

describe('Named output graph ports', () => {
  it('shows one operation with four result ports and preserves selected connections', () => {
    const model = canvasModel({ graph, positions: {} }, drawdownVariables, [drawdownOperator])
    expect(model.schemas.find(item => item.id === 'dd')?.outputs).toHaveLength(4)
    expect(graphEdges(graph).find(edge => edge.target === 'output_mdd')?.sourcePort).toBe('max_drawdown')
    const connection = { source: 'dd', sourcePort: 'recovery_periods', target: 'output_mdd', targetPort: 'value' }
    expect(graphConnectionIssue(graph, connection, [drawdownOperator])).toBeNull()
    expect(connectGraph(graph, connection).outputs[0].port_id).toBe('recovery_periods')
    expect(graphConnectionIssue(graph, { ...connection, sourcePort: 'value' }, [drawdownOperator])).not.toBeNull()
  })
  it('allows keyboard-accessible selection of an individual output in the inspector', () => {
    const update = vi.fn()
    render(<IndicatorNodeInspector graph={graph} selectedId="output_mdd" variables={drawdownVariables} operators={[drawdownOperator]} types={{}} isTimeSeries={false} isScalarBundle onNodeChange={() => undefined} onOutputChange={update} onRemove={() => undefined} onDuplicate={() => undefined} />)
    const selector = screen.getByRole('combobox', { name: '结果来源' })
    expect(selector).toHaveValue('dd::max_drawdown')
    fireEvent.change(selector, { target: { value: 'dd::decline_periods' } })
    expect(update).toHaveBeenCalledWith('mdd', { node_id: 'dd', port_id: 'decline_periods' })
  })
})

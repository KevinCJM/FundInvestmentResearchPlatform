import { useState } from 'react'
import { fireEvent, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'
import type { RegimeGraphDefinition, RegimeNodeSchema } from '../../services/regimeGraph'
import RegimeSeriesOutputs from './RegimeSeriesOutputs'
import { regimeCalculationFingerprint } from './regimeDraftIdentity'

const initial: RegimeGraphDefinition = { schema_version: '2.0', name: '输出测试', description: '',
  graph: { nodes: [{ id: 'price', type: 'source.inline', parameters: {}, inputs: {}, label: '指数' },
    { id: 'classifier', type: 'model.range_threshold', parameters: {}, inputs: {}, label: '分类' }],
    outputs: { state: { node_id: 'classifier', port: 'state' } } },
  states: [{ id: 'bull', label: '牛市', color: '#ff0000', role: 'positive', order: 0 }, { id: 'bear', label: '熊市', color: '#00ff00', role: 'negative', order: 1 }],
  validation: {}, evaluation_targets: [], usage_intent: 'research_display' }
const schemas = [
  { id: 'source.inline', category: 'source', label: '价格', inputs: [], outputs: [{ id: 'value', value_type: 'series<float64>' }] },
  { id: 'model.range_threshold', category: 'model', label: '分类', inputs: [], outputs: [{ id: 'state', value_type: 'state_codes<int64>' }] },
] as RegimeNodeSchema[]

function Harness() {
  const [definition, setDefinition] = useState(initial)
  return <><RegimeSeriesOutputs definition={definition} schemas={schemas} onChange={setDefinition} /><output data-testid="definition">{JSON.stringify(definition)}</output><output data-testid="calculation">{regimeCalculationFingerprint(definition, 'realtime', '')}</output></>
}
const current = () => JSON.parse(screen.getByTestId('definition').textContent || '{}') as RegimeGraphDefinition

describe('时序输出构建', () => {
  it('枚举不显示收益单位，颜色与名称不改变计算身份', () => {
    render(<Harness />)
    expect(screen.getByLabelText('输出值类型')).toHaveValue('枚举时序')
    expect(screen.queryByLabelText('输出通道 1 小数位')).not.toBeInTheDocument()
    const identity = screen.getByTestId('calculation').textContent
    fireEvent.change(screen.getByLabelText('状态1颜色'), { target: { value: '#112233' } })
    fireEvent.change(screen.getByLabelText('输出通道 1 名称'), { target: { value: '市场阶段' } })
    expect(current().states[0].color).toBe('#112233')
    expect(current().graph.channel_metadata?.state.label).toBe('市场阶段')
    expect(screen.getByTestId('calculation').textContent).toBe(identity)
  })

  it('新增数值通道、修改稳定编号和格式，仍使用同一计算来源', async () => {
    render(<Harness />)
    const user = userEvent.setup()
    await user.click(screen.getByRole('button', { name: '新增数值通道' }))
    expect(current().graph.outputs.channel_1).toEqual({ node_id: 'price', port: 'value' })
    expect(screen.queryByLabelText('输出值类型')).not.toBeInTheDocument()
    const identifier = screen.getByLabelText('输出通道 1 ID')
    await user.clear(identifier); await user.type(identifier, 'trend'); await user.tab()
    expect(current().graph.outputs.trend).toEqual({ node_id: 'price', port: 'value' })
    expect(current().graph.outputs.channel_1).toBeUndefined()
    fireEvent.change(screen.getByLabelText('输出通道 1 名称'), { target: { value: '滤波线' } })
    await user.selectOptions(screen.getByLabelText('输出通道 1 展示格式'), 'percent')
    expect(current().graph.channel_metadata?.trend).toMatchObject({ label: '滤波线', display_format: 'percent' })
    const choice = screen.getByLabelText('滤波线来源')
    expect(choice).not.toHaveTextContent('分类')
    await user.click(screen.getByRole('button', { name: '移除当前通道' }))
    expect(current().graph.outputs.trend).toBeUndefined()
    expect(current().graph.channel_metadata?.trend).toBeUndefined()
    expect(current().graph.outputs.state).toEqual(initial.graph.outputs.state)
  })

  it('重复通道编号可读地报错并保留原始定义', async () => {
    render(<Harness />)
    const user = userEvent.setup()
    await user.click(screen.getByRole('button', { name: '新增数值通道' }))
    const identifier = screen.getByLabelText('输出通道 1 ID')
    await user.clear(identifier); await user.type(identifier, 'state'); await user.tab()
    expect(screen.getByRole('alert')).toHaveTextContent('不能重复')
    expect(identifier).toHaveValue('channel_1')
    expect(current().graph.outputs.state).toEqual(initial.graph.outputs.state)
  })
})

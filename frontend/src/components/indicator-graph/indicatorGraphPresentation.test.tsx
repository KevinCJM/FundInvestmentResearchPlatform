import { cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import type { IndicatorOperator, IndicatorVariable } from '../../services/customIndicators'
import type { AuthoringGraph, GraphValueType } from '../../services/indicatorGraph'
import IndicatorNodeInspector from './IndicatorNodeInspector'
import { canvasModel, emptyOutput, graphSignature, nodeLabel } from './indicatorGraphAdapter'
import { createGraphTextFormatter, graphAxesLabel, graphConstantLabel, graphParameterLabel, graphTypeLabel, graphVariableLabel } from './indicatorGraphPresentation'

const variables: IndicatorVariable[] = [
  { name: 'returns', label: '复权净值普通收益率', value_type: 'series<time>[T]', dtype: 'float64', latex: String.raw`\mathbf{r}` },
  { name: 'periods_per_year', label: '年化因子', value_type: 'scalar', dtype: 'float64', latex: 'p' },
  { name: 'risk_free_rate_per_observation', label: '单观察期无风险收益率', value_type: 'scalar', dtype: 'float64', latex: 'r_f' },
]
const operators: IndicatorOperator[] = [
  {
    name: 'less', label: '逐元素小于比较', signature: 'less(a,b)', latex_template: 'a<b', return_type: 'mask',
    mathematical_essence: '逐元素判断 A 是否小于 B，输出布尔 mask。',
    parameters: [
      { name: 'a', label: '输入 A', description: '允许类型： scalar | series<T> | vector<N> | matrix<A,B>。', allowed_types: ['scalar', 'series<T>', 'vector<N>', 'matrix<A,B>'] },
      { name: 'b', label: '输入 B', description: '允许类型： scalar | series<T> | vector<N> | matrix<A,B>。', allowed_types: ['scalar', 'series<T>', 'vector<N>', 'matrix<A,B>'] },
    ],
  },
]
const graph: AuthoringGraph = {
  graph_version: 1,
  nodes: [
    { id: 'r', kind: 'variable', variable_id: 'returns' },
    { id: 'lt', kind: 'operator', operator_id: 'less', arguments: { a: { source: 'node', node_id: 'r' }, b: { source: 'constant', value: 0 } } },
  ],
  outputs: [{ ...emptyOutput('result', '最终结果'), node_id: 'lt' }],
}
const types: Record<string, GraphValueType> = {
  r: { kind: 'series', display: 'series<time>[T]', axes: ['time'], dtype: 'float64' },
  lt: { kind: 'mask', display: 'mask<time>[T]', axes: ['time'], dtype: 'bool' },
}
const text = createGraphTextFormatter(variables, operators)
afterEach(cleanup)

describe('画布中文展示，不改变计算协议', () => {
  it.each([
    ['scalar', '单个数值'], ['series<Time>[T]', '时间序列'], ['series<T>', '时间序列'],
    ['vector<N>', '一维数组'], ['vector<asset>[N]', '资产向量'], ['matrix<A,B>', '矩阵（二维数据）'],
    ['matrix<time,asset>[T,N]', '时间—资产矩阵'], ['matrix<asset,time>[N,T]', '资产—时间矩阵'],
    ['matrix<asset,asset>[N,N]', '资产方阵'], ['mask<time>[T]', '时间条件序列（是／否）'],
    ['mask<asset>[N]', '资产条件序列（是／否）'], ['mask<time,asset>[T,N]', '条件矩阵（是／否）'],
    ['mask<A,B>[M,N]', '条件矩阵（是／否）'], ['bool', '单个判断值（是／否）'],
    ['tuple', '结构化值（兼容）'], ['unknown', '类型待检查'],
  ])('类型 %s 显示为 %s', (input, expected) => {
    expect(graphTypeLabel(input)).toBe(expected)
    expect(text(input)).toBe(expected)
  })

  it('使用结构化类型，保留布尔和名义轴的区别', () => {
    expect(graphTypeLabel({ kind: 'scalar', dtype: 'bool', display: 'scalar' })).toBe('单个判断值（是／否）')
    expect(graphTypeLabel(types.lt)).toBe('时间条件序列（是／否）')
    expect(graphAxesLabel(types.lt)).toBe('时间')
    expect(graphAxesLabel({ kind: 'matrix', axes: ['asset', 'time'] })).toBe('资产、时间')
    expect(graphAxesLabel({ kind: 'matrix', axes: ['A', 'B'] })).toBe('第 1 维、第 2 维')
    expect(graphAxesLabel({ kind: 'scalar' })).toBe('')
  })

  it('说明中的类型并集、变量和缺失值均使用中文', () => {
    expect(text(operators[0].parameters![0].description)).toBe('允许类型： 单个数值、时间序列、一维数组、矩阵（二维数据）。')
    expect(text('数据依赖：returns、risk_free_rate_per_observation、periods_per_year')).toBe('数据依赖：复权净值普通收益率、单观察期无风险收益率、年化因子')
    expect(text('NaN / Inf')).toBe('缺失值 / 无穷值')
    expect(text('same(series<T>)')).toBe('与输入相同的数据类型')
    expect(text('[OUTPUT_CONTRACT_MISMATCH] 最终输出应为 series<time>[T]，当前为 scalar。')).toBe('最终输出应为 时间序列，当前为 单个数值。')
  })

  it('未知元数据不回显代码名，也不改变用户自己填写的步骤备注', () => {
    expect(graphVariableLabel('future_variable', variables)).toBe('输入数据（名称待补充）')
    expect(graphParameterLabel({ name: 'ddof', label: '自由度修正（ddof）' })).toBe('自由度修正')
    expect(graphParameterLabel({ name: 'future_parameter' }, 2)).toBe('输入 3')
    expect(nodeLabel({ ...graph.nodes[0], label: 'My returns 研究' }, variables, operators)).toBe('My returns 研究')
    expect(graphConstantLabel(true)).toBe('是')
    expect(graphConstantLabel(false)).toBe('否')
  })

  it('节点摘要、端口提示中文化；类型、参数名、连线及图签名保持不变', () => {
    const before = JSON.stringify(graph)
    const signature = graphSignature(graph)
    const model = canvasModel({ graph, positions: {} }, variables, operators, types)
    expect(model.nodes[0]).toMatchObject({ id: 'r', label: '复权净值普通收益率', statusLabel: '时间序列' })
    expect(model.nodes[1].statusLabel).toBe('时间条件序列（是／否）')
    expect(model.schemas[0].outputs[0]).toMatchObject({ id: 'value', value_type: 'series', type_label: '时间序列' })
    expect(model.schemas[1].inputs[0]).toMatchObject({ id: 'a', label: '输入 A', type_label: '单个数值、时间序列、一维数组、矩阵（二维数据）' })
    expect(model.edges[0]).toMatchObject({ source: 'r', target: 'lt', targetPort: 'a' })
    expect(JSON.stringify(graph)).toBe(before)
    expect(graphSignature(graph)).toBe(signature)
  })

  it('截图中的参数面板、上游选项和输出类型不再显示原始代码', () => {
    const onNodeChange = vi.fn()
    const { container } = render(<IndicatorNodeInspector graph={graph} selectedId="lt" variables={variables} operators={operators} types={types} isTimeSeries={false} onNodeChange={onNodeChange} onOutputChange={vi.fn()} onRemove={vi.fn()} onDuplicate={vi.fn()} />)
    expect(container.textContent).not.toMatch(/\b(?:returns|scalar|series|mask|time)\b|<T>|<A,B>/)
    expect(screen.getByText('已检查输出：时间条件序列（是／否） · 对齐维度：时间')).toBeInTheDocument()
    expect(screen.getByLabelText('输入 A当前输入')).toHaveTextContent('已连接：步骤 1 · 复权净值普通收益率 · 时间序列')
    const selector = screen.getByRole('combobox', { name: '输入 A输入来源' })
    expect(within(selector).getByRole('option', { name: '步骤 1 · 复权净值普通收益率 · 时间序列' })).toHaveValue('r')
    fireEvent.change(screen.getByRole('spinbutton', { name: '输入 B常量' }), { target: { value: '2' } })
    expect(onNodeChange).toHaveBeenCalledWith({ ...graph.nodes[1], arguments: { a: { source: 'node', node_id: 'r' }, b: { source: 'constant', value: 2 } } })
  })
})

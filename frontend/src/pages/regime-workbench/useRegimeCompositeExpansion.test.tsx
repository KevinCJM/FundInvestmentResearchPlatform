import { act, render, renderHook, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, expect, it, vi } from 'vitest'
import { expandRegimeComposite, type RegimeCompositeExpansion, type RegimeGraphDefinition, type RegimeGraphNode, type RegimeNodeSchema } from '../../services/regimeGraph'
import useRegimeCompositeExpansion, { positionExpandedDefinition } from './useRegimeCompositeExpansion'
import RegimeResourceLibrary from './RegimeResourceLibrary'
import RegimeGranularityInfo from './RegimeGranularityInfo'
import { ParameterInput, regimeParameterIsActive } from './RegimeNodeInspector'

vi.mock('../../services/regimeGraph', async importOriginal => ({
  ...await importOriginal<typeof import('../../services/regimeGraph')>(), expandRegimeComposite: vi.fn(),
}))

function original(): RegimeGraphDefinition {
  return { schema_version: '2.0', name: '测试方案', description: '', usage_intent: 'research_display', evaluation_targets: [], validation: {},
    states: [{ id: 'bull', label: '牛市', order: 0 }, { id: 'flat', label: '震荡', order: 1 }, { id: 'bear', label: '熊市', order: 2 }],
    graph: { nodes: [
      { id: 'source', type: 'source.index', inputs: {}, parameters: {}, position: { x: 10, y: 20 } },
      { id: 'old', type: 'model.threshold', inputs: { value: { node_id: 'source', port: 'value' } }, parameters: {}, position: { x: 100, y: 200 } },
      { id: 'tail', type: 'post.confirmation', inputs: { state: { node_id: 'old', port: 'state' } }, parameters: {}, position: { x: 900, y: 500 } },
    ], outputs: { state: { node_id: 'tail', port: 'state' } } },
  }
}

function expanded(): RegimeCompositeExpansion {
  const definition = original()
  const replacements: RegimeGraphNode[] = [
    { id: 'comparison', type: 'condition.compare', inputs: { value: { node_id: 'source', port: 'value' } }, parameters: {} },
    { id: 'selection', type: 'state.select', inputs: { condition: { node_id: 'comparison', port: 'condition' } }, parameters: {} },
  ]
  definition.graph.nodes = definition.graph.nodes.flatMap<RegimeGraphNode>(node => node.id === 'old' ? replacements
    : [{ ...node, position: undefined, inputs: node.id === 'tail' ? { state: { node_id: 'selection', port: 'state' } } : node.inputs }])
  return { contract_version: 1, definition, replaced_node_id: 'old', inserted_node_ids: ['comparison', 'selection'], primary_node_id: 'selection',
    output_map: { state: { node_id: 'selection', port: 'state' } }, steps: ['比较', '选态'] }
}

function options() {
  return { definition: original(), mode: 'realtime' as const, blocked: false,
    onChange: vi.fn(), onSelected: vi.fn(), onError: vi.fn(), onNotice: vi.fn() }
}

function deferred() {
  let resolve!: (value: RegimeCompositeExpansion) => void
  const promise = new Promise<RegimeCompositeExpansion>(done => { resolve = done })
  vi.mocked(expandRegimeComposite).mockReturnValue(promise)
  return { resolve, promise }
}

beforeEach(() => { vi.clearAllMocks() })

it('展开保留已有位置与下游连线，新增步骤按依赖排列且不修改原对象', () => {
  const source = original(), response = expanded()
  const frozenSource = JSON.stringify(source), frozenResponse = JSON.stringify(response)
  const next = positionExpandedDefinition(source, response)
  expect(next.graph.nodes.find(node => node.id === 'tail')?.position).toEqual({ x: 900, y: 500 })
  expect(next.graph.nodes.find(node => node.id === 'source')?.position).toEqual({ x: 10, y: 20 })
  expect(next.graph.nodes.find(node => node.id === 'comparison')?.position).toEqual({ x: 100, y: 200 })
  expect(next.graph.nodes.find(node => node.id === 'selection')?.position?.x).toBeGreaterThan(100)
  expect(next.graph.nodes.find(node => node.id === 'tail')?.inputs.state.node_id).toBe('selection')
  expect(JSON.stringify(source)).toBe(frozenSource)
  expect(JSON.stringify(response)).toBe(frozenResponse)
})

it('拒绝节点冲突或不支持的展开契约，不静默覆盖现有节点', () => {
  const response = expanded()
  response.inserted_node_ids.push('source')
  expect(() => positionExpandedDefinition(original(), response)).toThrow('不匹配')
  expect(() => positionExpandedDefinition(original(), { ...expanded(), contract_version: 99 })).toThrow('不匹配')
})

it('展开只应用一次编辑，支持工作台单次撤销且不保存或计算', async () => {
  const pending = deferred(), callbacks = options()
  const { result } = renderHook(() => useRegimeCompositeExpansion(callbacks))
  let operation!: Promise<void>
  act(() => { operation = result.current.expand(callbacks.definition, 'old') })
  expect(result.current.expanding).toBe(true)
  expect(callbacks.onChange).not.toHaveBeenCalled()
  await act(async () => { pending.resolve(expanded()); await operation })
  expect(callbacks.onChange).toHaveBeenCalledTimes(1)
  expect(callbacks.onSelected).toHaveBeenCalledWith('selection')
  expect(callbacks.onNotice).toHaveBeenCalledWith(expect.stringContaining('撤销'))
  expect(expandRegimeComposite).toHaveBeenCalledWith(callbacks.definition, 'old', 'realtime', expect.any(AbortSignal))
  expect(result.current.expanding).toBe(false)
})

it.each(['definition', 'layout', 'mode', 'formula'] as const)('拒绝在 %s 修改后到达的过期展开结果', async kind => {
  const pending = deferred()
  const callbacks: Parameters<typeof useRegimeCompositeExpansion>[0] = options()
  const { result, rerender } = renderHook(props => useRegimeCompositeExpansion(props), { initialProps: callbacks })
  let operation!: Promise<void>
  act(() => { operation = result.current.expand(callbacks.definition, 'old') })
  const edited = original()
  if (kind === 'definition') edited.name = '用户的新名称'
  if (kind === 'layout') edited.graph.nodes[0].position = { x: 333, y: 44 }
  rerender({ ...callbacks, definition: edited, mode: kind === 'mode' ? 'retrospective' : 'realtime', blocked: kind === 'formula' })
  await act(async () => { pending.resolve(expanded()); await operation })
  expect(callbacks.onChange).not.toHaveBeenCalled()
  expect(callbacks.onNotice).toHaveBeenCalledWith(expect.stringContaining('过期'))
})

it('公式未应用时不展开，连续点击不产生重复请求', async () => {
  const pending = deferred(), callbacks = options()
  const { result, rerender } = renderHook(props => useRegimeCompositeExpansion(props), { initialProps: { ...callbacks, blocked: true } })
  await act(async () => { await result.current.expand(callbacks.definition, 'old') })
  expect(expandRegimeComposite).not.toHaveBeenCalled()
  rerender({ ...callbacks, blocked: false })
  let operation!: Promise<void>
  act(() => { operation = result.current.expand(callbacks.definition, 'old'); void result.current.expand(callbacks.definition, 'old') })
  expect(expandRegimeComposite).toHaveBeenCalledTimes(1)
  await act(async () => { pending.resolve(expanded()); await operation })
})

it('卸载时取消请求，返回后不改草稿', async () => {
  const pending = deferred(), callbacks = options()
  const { result, unmount } = renderHook(() => useRegimeCompositeExpansion(callbacks))
  let operation!: Promise<void>
  act(() => { operation = result.current.expand(callbacks.definition, 'old') })
  const signal = vi.mocked(expandRegimeComposite).mock.calls[0][3]
  unmount()
  expect(signal?.aborted).toBe(true)
  await act(async () => { pending.resolve(expanded()); await operation })
  expect(callbacks.onChange).not.toHaveBeenCalled()
})

it('接口失败时保留草稿并显示原因', async () => {
  const callbacks = options()
  vi.mocked(expandRegimeComposite).mockRejectedValue(new Error('超过节点预算'))
  const { result } = renderHook(() => useRegimeCompositeExpansion(callbacks))
  await act(async () => { await result.current.expand(callbacks.definition, 'old') })
  expect(callbacks.onChange).not.toHaveBeenCalled()
  expect(callbacks.onError).toHaveBeenLastCalledWith('超过节点预算')
  expect(result.current.expanding).toBe(false)
})

const composite: RegimeNodeSchema = { id: 'model.threshold', label: '三状态阈值', category: 'model', inputs: [], outputs: [],
  granularity: { kind: 'composite', label: '组合模板', reason: '展开为真实计算步骤', expandable: true, steps: ['上界比较', '下界比较', '选态'], contract_version: 1 } }

it('节点库区分基础、组合与内核，组合明确显示添加并展开', async () => {
  const add = vi.fn()
  render(<RegimeResourceLibrary schemas={[composite,
    { ...composite, id: 'condition.compare', label: '数值比较', granularity: { ...composite.granularity!, kind: 'primitive', label: '基础算子', expandable: false } },
    { ...composite, id: 'pivot.ps_filter', label: 'PS 联合筛选', granularity: { ...composite.granularity!, kind: 'coupled', label: '耦合内核', expandable: false } },
  ]} onAdd={add} onOpenDataLab={vi.fn()} />)
  expect(screen.getAllByRole('combobox')).toHaveLength(1)
  await userEvent.selectOptions(screen.getByLabelText('节点类型'), 'granularity:composite')
  expect(screen.queryByRole('button', { name: '添加数值比较' })).not.toBeInTheDocument()
  await userEvent.click(screen.getByRole('button', { name: '添加并展开三状态阈值' }))
  expect(add).toHaveBeenCalledWith(composite)
  expect(screen.getByText(/上界比较 → 下界比较 → 选态/)).toBeVisible()
})

it('实时不允许的组合与请求处理中禁止添加', () => {
  const { rerender } = render(<RegimeResourceLibrary schemas={[{ ...composite, available: false, unavailable_reason: '仅限事后' }]} onAdd={vi.fn()} onOpenDataLab={vi.fn()} />)
  expect(screen.getByRole('button', { name: '添加并展开三状态阈值' })).toBeDisabled()
  expect(screen.getByText('仅限事后')).toBeVisible()
  rerender(<RegimeResourceLibrary schemas={[composite]} busy onAdd={vi.fn()} onOpenDataLab={vi.fn()} />)
  expect(screen.getByRole('button', { name: '添加并展开三状态阈值' })).toBeDisabled()
})

it('已有组合提供显式展开动作，耦合内核只解释保留原因', async () => {
  const expand = vi.fn()
  const { rerender } = render(<RegimeGranularityInfo schema={composite} onExpand={expand} />)
  await userEvent.click(screen.getByRole('button', { name: '展开为计算步骤' }))
  expect(expand).toHaveBeenCalledTimes(1)
  rerender(<RegimeGranularityInfo schema={{ ...composite, granularity: { ...composite.granularity!, kind: 'coupled', label: '耦合内核', expandable: false, reason: '删点后必须重新检查相邻约束' } }} onExpand={expand} />)
  expect(screen.queryByRole('button')).not.toBeInTheDocument()
  expect(screen.getByText('删点后必须重新检查相邻约束')).toBeVisible()
})

it('状态参数显示业务名称并提交整数编码，已连接分支隐藏常量参数', async () => {
  const change = vi.fn()
  render(<ParameterInput name="true_code" schema={{ type: 'integer', state_code: true, title: '条件满足时的状态', default: 0 }} value={0} states={original().states} onChange={change} />)
  const select = screen.getByRole('combobox')
  expect(screen.getByRole('option', { name: '牛市' })).toBeInTheDocument()
  expect(screen.getByRole('option', { name: '未分类' })).toBeInTheDocument()
  await userEvent.selectOptions(select, '2')
  expect(change).toHaveBeenCalledWith(2)
  const node = { id: 'select', type: 'state.select', parameters: {}, inputs: { when_true: { node_id: 'other', port: 'state' } } }
  expect(regimeParameterIsActive(node, 'true_code')).toBe(false)
  expect(regimeParameterIsActive(node, 'false_code')).toBe(true)
})

import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, expect, it, vi } from 'vitest'
import RegimeNodePreviewPanel from './RegimeNodePreviewPanel'
import { comparisonOptions } from './RegimeComparisonPicker'
import type { RegimeGraphDefinition, RegimeNodeSchema } from '../../services/regimeGraph'

vi.mock('echarts-for-react', () => ({ default: ({ option }: { option: unknown }) => <div data-testid="node-chart">{JSON.stringify(option)}</div> }))
const definition: RegimeGraphDefinition = { schema_version: '2.0', description: '', usage_intent: 'research_display', name: '尚未写完', graph: { nodes: [{ id: 'market', type: 'source.index', label: '沪深300', parameters: { ts_code: '000300.SH' }, inputs: {} }, { id: 'unfinished', type: 'model.threshold', parameters: {}, inputs: {} }], outputs: {} }, states: [], evaluation_targets: [], validation: {} }
const schemas: RegimeNodeSchema[] = [{ id: 'source.index', label: '指数行情', category: 'source', inputs: [], outputs: [{ id: 'value', label: '指数点位' }] }, { id: 'model.threshold', label: '分类器', category: 'model', inputs: [{ id: 'value' }], outputs: [{ id: 'state' }] }]
const audit = { execution_backend: 'numba_njit_fixed_signature', nopython: true, object_mode: 0, python_fallback: 0, request_time_compilation: 0, kernel_signatures: { graph: ['fixed'] } }
const ok = (body: unknown) => ({ ok: true, json: async () => body } as Response)
function setup() {
  const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = new URL(String(input), 'http://localhost')
    if (url.pathname.endsWith('/prepare')) return ok({ compile_token: 'test-token', runtime_audit: audit })
    if (init?.method === 'DELETE') return ok({ id: 'node-run', status: 'cancelled' })
    if (url.pathname.endsWith('/preview-runs')) return ok({ id: 'node-run', status: 'queued' })
    if (url.pathname.endsWith('/node-run')) return ok({ id: 'node-run', status: 'completed', execution: audit })
    const offset = Number(url.searchParams.get('offset'))
    return ok({ id: 'node-run', node_id: 'market', port: 'value', offset, total: 3, limit: 2, items: offset ? [{ observation_date: '2026-09-07', value: 4200 }] : [{ observation_date: '2010-01-04', value: 3500 }, { observation_date: '2020-01-02', value: 4000 }] })
  })
  vi.stubGlobal('fetch', fetchMock)
  const view = render(<RegimeNodePreviewPanel definition={definition} schemas={schemas} initialNodeId="market" initialMode="realtime" initialAsOf="" />)
  return { fetchMock, ...view }
}
afterEach(() => { vi.unstubAllGlobals(); vi.restoreAllMocks() })

it('未连接最终输出仍可预览，提交独立目标并完整加载到最新日期', async () => {
  const { fetchMock } = setup()
  await userEvent.click(screen.getByRole('button', { name: '预览节点数据' }))
  expect(await screen.findByText('已返回 3 / 3 条节点结果')).toBeInTheDocument()
  expect(screen.getByTestId('node-chart')).toHaveTextContent('2026-09-07')
  const prepare = fetchMock.mock.calls.find(([url]) => String(url).endsWith('/prepare'))
  expect(JSON.parse(String(prepare?.[1]?.body))).toMatchObject({ definition: { graph: { outputs: {} } }, preview_target: { node_id: 'market', port: 'value' } })
  expect(JSON.parse(String(prepare?.[1]?.body))).not.toHaveProperty('comparison_targets')
  const submitted = fetchMock.mock.calls.find(([url]) => String(url).endsWith('/preview-runs'))
  expect(JSON.parse(String(submitted?.[1]?.body))).toMatchObject({ preview_target: { node_id: 'market', port: 'value' }, mode: 'realtime' })
  expect(JSON.parse(String(submitted?.[1]?.body))).not.toHaveProperty('comparison_targets')
  expect(fetchMock.mock.calls.some(([url]) => String(url).includes('offset=2'))).toBe(true)
  expect(definition.graph.outputs).toEqual({})
})

it('旧后端不支持新预览参数时说明更新办法，不静默丢弃对比节点或重试', async () => {
  const { fetchMock } = setup()
  fetchMock.mockImplementation(async () => ({ ok: false, status: 422, json: async () => ({ detail: {
    code: 'REQUEST_VALIDATION_ERROR', message: '请求参数无效。', field: 'comparison_targets',
    diagnostics: [{ code: 'extra_forbidden', field: 'comparison_targets', message: 'Extra inputs are not permitted' }],
  } }) } as Response))
  await userEvent.click(screen.getByRole('button', { name: '预览节点数据' }))
  expect(await screen.findAllByText(/当前后端版本不支持多节点对比，请更新并重启后端服务后重试/)).not.toHaveLength(0)
  expect(screen.queryByText(/Extra inputs are not permitted/)).not.toBeInTheDocument()
  expect(fetchMock).toHaveBeenCalledTimes(1)
})

it('模式、截至日或目标改变后旧结果标记过期，不自动覆盖原算法', async () => {
  const { fetchMock } = setup()
  fireEvent.change(screen.getByLabelText('节点预览截至日'), { target: { value: '2024-01-01' } })
  await userEvent.click(screen.getByRole('button', { name: '预览节点数据' }))
  await screen.findByText('已返回 3 / 3 条节点结果')
  const submitted = fetchMock.mock.calls.find(([url]) => String(url).endsWith('/preview-runs'))
  expect(JSON.parse(String(submitted?.[1]?.body)).as_of).toBe('2024-01-01')
  await userEvent.selectOptions(screen.getByLabelText('节点预览分析方式'), 'retrospective')
  expect(screen.getByRole('status')).toHaveTextContent('配置已变化')
  await userEvent.selectOptions(screen.getByLabelText('待预览节点'), 'unfinished')
  expect(screen.getByLabelText('预览输出端口')).toHaveValue('state')
})

it('取消预览会取消后台任务，停止等待', async () => {
  const { fetchMock } = setup()
  fetchMock.mockImplementation(async (input, init) => {
    if (String(input).endsWith('/prepare')) return ok({ compile_token: 'test-token', runtime_audit: audit })
    return ok({ id: 'node-run', status: init?.method === 'DELETE' ? 'cancelled' : 'running' })
  })
  await userEvent.click(screen.getByRole('button', { name: '预览节点数据' }))
  await waitFor(() => expect(fetchMock.mock.calls.some(([url]) => String(url).endsWith('/preview-runs'))).toBe(true))
  await userEvent.click(screen.getByRole('button', { name: '取消预览' }))
  await waitFor(() => expect(fetchMock.mock.calls.some(([, init]) => init?.method === 'DELETE')).toBe(true))
  expect(screen.getByRole('button', { name: '预览节点数据' })).toBeEnabled()
})

it('显示具体日期问题，一次点击添加可保存的对齐节点后重新预览，保持左右次序和截止日', async () => {
  const { useState } = await import('react')
  const draft = { ...definition, graph: { ...definition.graph, nodes: [
    ...definition.graph.nodes,
    { id: 'other', type: 'source.index', label: '标普500', parameters: {}, inputs: {} },
    { id: 'difference', type: 'math.subtract', label: 'ETF价差', parameters: {}, inputs: { left: { node_id: 'other', port: 'value' }, right: { node_id: 'market', port: 'value' } } },
  ] } }
  const catalog = [...schemas,
    { id: 'math.subtract', label: '相减', category: 'arithmetic', inputs: [{ id: 'left' }, { id: 'right' }], outputs: [{ id: 'value' }] },
    { id: 'align.strict_intersection', label: '严格交集对齐', category: 'alignment', inputs: [{ id: 'left' }, { id: 'right' }], outputs: [{ id: 'left' }, { id: 'right' }] },
  ]
  let edited: RegimeGraphDefinition | undefined
  function Harness() {
    const [value, setValue] = useState<RegimeGraphDefinition>(draft)
    return <RegimeNodePreviewPanel definition={value} schemas={catalog} initialNodeId="difference" initialMode="realtime" initialAsOf="2024-01-11" onChange={next => { edited = next; setValue(next) }} />
  }
  const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const path = new URL(String(input), 'http://localhost').pathname
    if (path.endsWith('/prepare')) {
      const data = JSON.parse(String(init?.body))
      if (!data.definition.graph.nodes.some((node: { type: string }) => node.type === 'align.strict_intersection')) return { ok: false, status: 422, json: async () => ({ detail: { message: '历史情景图谱未通过语义校验。', diagnostics: [{ code: 'EXPLICIT_ALIGNMENT_REQUIRED', path: 'graph.nodes.difference.inputs', message: 'ETF价差需要先按共同日期对齐。' }] } }) } as Response
      return ok({ compile_token: 'aligned-token', runtime_audit: audit })
    }
    if (path.endsWith('/preview-runs')) return ok({ id: 'node-run', status: 'completed', execution: audit })
    return ok({ node_id: 'difference', port: 'value', offset: 0, total: 1, limit: 5000, items: [{ observation_date: '2024-01-03', value: 92 }] })
  })
  vi.stubGlobal('fetch', fetchMock)
  render(<Harness />)
  await userEvent.click(screen.getByRole('button', { name: '预览节点数据' }))
  expect(await screen.findByRole('alert')).toHaveTextContent('ETF价差需要先按共同日期对齐')
  expect(edited).toBeUndefined()
  await userEvent.click(screen.getByRole('button', { name: '按共同日期对齐并预览' }))
  await screen.findByText('已返回 1 / 1 条节点结果')
  const align = edited!.graph.nodes.find(node => node.type === 'align.strict_intersection')!
  expect(align.inputs).toEqual({ left: { node_id: 'other', port: 'value' }, right: { node_id: 'market', port: 'value' } })
  expect(edited!.graph.nodes.find(node => node.id === 'difference')!.inputs).toEqual({ left: { node_id: align.id, port: 'left' }, right: { node_id: align.id, port: 'right' } })
  expect(edited!.graph.outputs).toEqual({})
  expect(edited!.graph.nodes.find(node => node.id === 'unfinished')).toEqual(draft.graph.nodes[1])
  expect(draft.graph.nodes).toHaveLength(4)
  const submitted = fetchMock.mock.calls.find(([url]) => String(url).endsWith('/preview-runs'))!
  expect(JSON.parse(String(submitted[1]?.body))).toMatchObject({ mode: 'realtime', as_of: '2024-01-11', preview_target: { node_id: 'difference', port: 'value' } })
  expect(screen.queryByText(/配置已变化/)).not.toBeInTheDocument()
  expect(screen.queryByRole('button', { name: '按共同日期对齐并预览' })).not.toBeInTheDocument()
})

it('搜索并选择非上游分支，同次提交准备和试算，切换模式会阻止已选事后分支', async () => {
  const draft = { ...definition, graph: { ...definition.graph, nodes: [...definition.graph.nodes,
    { id: 'smooth', type: 'filter.ema', label: '单边 EMA', parameters: {}, inputs: { value: { node_id: 'market', port: 'value' } } },
    { id: 'future', type: 'retrospective', label: '事后数值', parameters: {}, inputs: { value: { node_id: 'market', port: 'value' } } },
  ] } }
  const catalog = [...schemas,
    { id: 'filter.ema', label: 'EMA', category: 'filter', inputs: [{ id: 'value' }], outputs: [{ id: 'value', value_type: 'series<float64>' }] },
    { id: 'retrospective', label: '事后', category: 'filter', causal: false, inputs: [{ id: 'value' }], outputs: [{ id: 'value', value_type: 'series<float64>' }] },
  ]
  const fetchMock = vi.fn(async (input: RequestInfo | URL, _init?: RequestInit) => {
    const path = new URL(String(input), 'http://localhost').pathname
    if (path.endsWith('/prepare')) return ok({ compile_token: 'test-token', runtime_audit: audit })
    if (path.endsWith('/preview-runs')) return ok({ id: 'node-run', status: 'completed', execution: audit })
    return ok({ id: 'node-run', node_id: 'market', port: 'value', offset: 0, total: 1, limit: 5000, items: [{ date: '2020-01-01', value: 100 }] })
  })
  vi.stubGlobal('fetch', fetchMock)
  render(<RegimeNodePreviewPanel definition={draft} schemas={catalog} initialNodeId="market" initialMode="realtime" initialAsOf="2024-01-11" />)
  await userEvent.click(screen.getByText('叠加对比节点（可选）'))
  expect(screen.getByRole('checkbox', { name: /事后数值/ })).toBeDisabled()
  await userEvent.type(screen.getByLabelText('搜索对比节点'), 'EMA')
  expect(screen.getAllByRole('checkbox')).toHaveLength(1)
  await userEvent.click(screen.getByRole('checkbox', { name: /单边 EMA/ }))
  await userEvent.click(screen.getByRole('button', { name: '预览节点数据' }))
  await screen.findByText('已返回 1 / 1 条节点结果')
  const calls = fetchMock.mock.calls.filter(([url]) => /\/(prepare|preview-runs)$/.test(String(url)))
  expect(calls).toHaveLength(2)
  for (const call of calls) expect(JSON.parse(String(call[1]?.body))).toMatchObject({ preview_target: { node_id: 'market', port: 'value' }, comparison_targets: [{ node_id: 'smooth', port: 'value' }] })
  await userEvent.clear(screen.getByLabelText('搜索对比节点'))
  await userEvent.selectOptions(screen.getByLabelText('节点预览分析方式'), 'retrospective')
  await userEvent.click(screen.getByRole('checkbox', { name: /事后数值/ }))
  await userEvent.selectOptions(screen.getByLabelText('节点预览分析方式'), 'realtime')
  expect(screen.getByRole('button', { name: '预览节点数据' })).toBeDisabled()
  expect(screen.getByRole('alert')).toHaveTextContent('此节点或其上游需要事后分析')
  await userEvent.click(screen.getByRole('checkbox', { name: /事后数值/ }))
  expect(screen.getByRole('button', { name: '预览节点数据' })).toBeEnabled()
  expect(screen.queryByText(/配置已变化/)).not.toBeInTheDocument()
})

it('候选按输出端口区分，实时限制沿输入及显式连线传播，循环不会卡住选择器', () => {
  const float = { id: 'value', value_type: 'series<float64>' }
  const catalog: RegimeNodeSchema[] = [
    { id: 'source.index', label: '指数', category: 'source', inputs: [], outputs: [float] },
    { id: 'future', label: '峰谷', category: 'filter', causal: false, inputs: [], outputs: [float] },
    { id: 'filter', label: '滤波', category: 'filter', causal: true, inputs: [{ id: 'value' }], outputs: [float, { id: 'confidence', value_type: 'confidence<time>' }, { id: 'state', value_type: 'state_codes<int64>' }] },
  ]
  const draft: RegimeGraphDefinition = { ...definition, graph: { outputs: {}, nodes: [definition.graph.nodes[0],
    { id: 'future', type: 'future', inputs: {}, parameters: {} },
    { id: 'after', type: 'filter', inputs: {}, parameters: {} },
    { id: 'cycle', type: 'filter', inputs: { value: { node_id: 'cycle', port: 'value' } }, parameters: {} },
  ], edges: [{ source: { node_id: 'future', port: 'value' }, target: { node_id: 'after', port: 'value' } }] } }
  const choices = comparisonOptions(draft, catalog, { node_id: 'market', port: 'value' }, 'realtime')
  expect(choices.filter(item => item.node_id === 'after' && item.port !== 'state').every(item => item.reason.includes('事后分析'))).toBe(true)
  expect(choices.find(item => item.port === 'state')?.reason).toContain('不是连续数值序列')
  expect(choices.filter(item => item.node_id === 'cycle')).toHaveLength(3)
})

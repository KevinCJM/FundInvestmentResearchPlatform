import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { StrictMode, type ReactNode } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import HistoricalRegimeWorkbench from './HistoricalRegimeWorkbench'

vi.mock('echarts-for-react', () => ({ default: () => <div data-testid="regime-v2-chart" /> }))
vi.mock('@xyflow/react', () => ({
  ReactFlow: ({ children }: { children?: ReactNode }) => <div data-testid="react-flow-test-surface">{children}</div>,
  Background: () => null,
  Controls: () => null,
  MiniMap: (props: Record<string, unknown>) => <div data-testid={String(props['data-testid'] || 'mini-map')} className={String(props.className || '')} aria-label={String(props.ariaLabel || '')} />,
  Handle: () => null,
  Position: { Left: 'left', Right: 'right' },
  applyNodeChanges: (_changes: unknown[], nodes: unknown[]) => nodes,
  addEdge: (connection: Record<string, unknown>, edges: unknown[]) => [...edges, connection],
}))

const fixedExecution = {
  execution_backend: 'numba_njit_fixed_signature',
  nopython: true,
  object_mode: 0,
  python_fallback: 0,
  request_time_compilation: 0,
  kernel_signatures: { regime_graph: ['float64[:]->int8[:]'] },
}

const schemas = [
  {
    id: 'source.series', label: '研究序列', description: '来自数据实验室的时序', category: 'source', category_label: '数据源', inputs: [], outputs: [{ id: 'value', label: '时序值' }],
    parameter_schema: { type: 'object', properties: { series_id: { type: 'string', label: '序列 ID', default: '' }, field: { type: 'string', label: '字段', default: 'close' } }, required: ['series_id'] },
    execution_policy: { backend: 'numba_njit_fixed_signature', njit_required: true, request_time_compilation: 0 },
  },
  {
    id: 'transform.ema', label: '因果 EMA', description: '单边指数滤波', category: 'transform', category_label: '变换', inputs: [{ id: 'series', label: '输入序列', required: true }], outputs: [{ id: 'value', label: '滤波序列' }],
    parameter_schema: { type: 'object', properties: { window: { type: 'integer', label: '窗口', default: 20, minimum: 2, maximum: 500 } }, required: ['window'] },
    execution_policy: { backend: 'numba_njit_fixed_signature', njit_required: true, request_time_compilation: 0 },
  },
  {
    id: 'decoder.threshold', label: '阈值状态解码', description: '将信号转换为状态', category: 'decoder', category_label: '状态规则', inputs: [{ id: 'score', label: '状态得分', required: true }], outputs: [{ id: 'state', label: '状态序列' }, { id: 'confidence', label: '置信度' }],
    parameter_schema: { type: 'object', properties: { upper: { type: 'number', label: '上阈值', default: 0.01 }, lower: { type: 'number', label: '下阈值', default: -0.01 } }, required: ['upper', 'lower'] },
    execution_policy: { backend: 'numba_njit_fixed_signature', njit_required: true, request_time_compilation: 0 },
  },
  {
    id: 'model.external_optimized', label: '第三方优化模型', description: '管理员安装适配器后可用', category: 'model', category_label: '模型', inputs: [{ id: 'features' }], outputs: [{ id: 'state' }],
    parameter_schema: { type: 'object', properties: {} },
    execution_policy: { backend: 'third_party_optimized_isolated', njit_required: false, third_party_exempt: true, request_time_compilation: 0 },
    available: false, status: 'admin_adapter_required', unavailable_reason: '当前未安装管理员批准的隔离模型适配器',
  },
]

const templateDefinition = {
  schema_version: '2.0',
  name: '沪深300自由牛熊研究',
  description: '模板实例化草稿',
  graph: {
    nodes: [
      { id: 'source-1', type: 'source.series', label: '沪深300', parameters: { series_id: 'index:index_daily:000300.SH', field: 'close' }, inputs: {} },
      { id: 'ema-1', type: 'transform.ema', label: '趋势滤波', parameters: { window: 20 }, inputs: { series: { node_id: 'source-1', port: 'value' } } },
      { id: 'decoder-1', type: 'decoder.threshold', label: '牛熊震荡', parameters: { upper: 0.01, lower: -0.01 }, inputs: { score: { node_id: 'ema-1', port: 'value' } } },
    ],
    outputs: { state: { node_id: 'decoder-1', port: 'state' }, confidence: { node_id: 'decoder-1', port: 'confidence' } },
  },
  states: [{ id: 'bull', label: '牛市', color: '#16a34a' }, { id: 'bear', label: '熊市', color: '#dc2626' }],
  evaluation_targets: [],
  validation: { walk_forward: true },
  usage_intent: 'taa',
}

const ok = (body: unknown, status = 200) => ({ ok: true, status, json: async () => body } as Response)
const settle = (milliseconds: number) => act(async () => { await new Promise((resolve) => window.setTimeout(resolve, milliseconds)) })

function makeFetch(options?: { keepRunning?: boolean }) {
  let pollCount = 0
  return vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const path = String(input)
    if (path.includes('/api/research-series/catalog')) return ok({ items: [{ id: 'index:index_daily:000300.SH', name: '沪深300', kind: 'index', status: 'available', regime_node_type: 'source.series', fields: [{ name: 'close', label: '收盘点位' }, { name: 'pct_chg', label: '涨跌幅' }], binding_parameters: { series_id: 'index:index_daily:000300.SH', field: 'close' } }], total: 1, offset: 0, limit: 500 })
    if (path.endsWith('/nodes')) return ok({ items: schemas })
    if (path.endsWith('/templates/v2')) return ok({ items: [{ id: 'bull-bear-v2', name: '牛熊震荡计算图' }] })
    if (path.endsWith('/v2/definitions')) return ok({ items: [] })
    if (path.includes('/v2/definitions/saved-query?revision=4')) return ok({ ...templateDefinition, id: 'saved-query', revision: 4, name: '精确修订研究' })
    if (path.endsWith('/v2/graph-assets')) return ok({ items: [] })
    if (path.includes('/v2/experiments?definition_id=')) return ok({ items: [] })
    if (path.includes('/templates/bull-bear-v2/instantiate')) return ok({ definition: templateDefinition })
    if (path.endsWith('/infer')) {
      const body = JSON.parse(String(init?.body))
      const nodes = body.definition.graph.nodes
      return ok({ valid: nodes.length > 0, graph_hash: `hash-${nodes.length}`, errors: nodes.length ? [] : [{ message: '至少添加一个节点。' }], warnings: [], inferred: { nodes: Object.fromEntries(nodes.map((node: { id: string }) => [node.id, { causal: true, execution_backend: 'numba_njit_fixed_signature' }])), causal: true, realtime_eligible: true } })
    }
    if (path.endsWith('/prepare')) return ok({ plan_id: 'PLAN-1', compile_token: 'TOKEN-1', graph_hash: 'hash-3', prepared_at: '2026-09-04', runtime_audit: fixedExecution })
    if (path.endsWith('/preview-runs') && init?.method === 'POST') return ok({ id: 'RUN-V2', status: 'queued', stage: 'queued', progress: 0.05 }, 202)
    if (path.endsWith('/preview-runs/RUN-V2') && init?.method === 'DELETE') return ok({ id: 'RUN-V2', status: 'cancelled', stage: 'cancelled', progress: 0.1 })
    if (path.endsWith('/preview-runs/RUN-V2')) {
      pollCount += 1
      if (options?.keepRunning || pollCount === 1) return ok({ id: 'RUN-V2', status: 'running', stage: '识别历史状态', progress: 0.55 })
      return ok({ id: 'RUN-V2', status: 'completed', stage: 'completed', progress: 1, execution: fixedExecution, result: { observations: 3 } })
    }
    if (path.includes('/preview-runs/RUN-V2/series?')) return ok({ run_id: 'RUN-V2', node_id: 'ema-1', port: 'value', items: [{ date: '2024-01-02', value: 3500 }, { date: '2024-01-03', value: 3510 }], total: 2, offset: 0, limit: 500, execution: fixedExecution })
    throw new Error(`Unexpected request: ${path} ${init?.method || 'GET'}`)
  })
}

describe('HistoricalRegimeWorkbench', () => {
  afterEach(() => { window.history.replaceState({}, '', '/'); vi.unstubAllGlobals(); vi.restoreAllMocks() })

  it('从 URL 查询参数载入已保存的精确 revision', async () => {
    const fetchMock = makeFetch()
    vi.stubGlobal('fetch', fetchMock)
    window.history.replaceState({}, '', '/settings/scenario-algorithms/workbench?definition=saved-query&revision=4')
    render(<StrictMode><HistoricalRegimeWorkbench /></StrictMode>)

    expect(await screen.findByDisplayValue('精确修订研究')).toBeInTheDocument()
    expect(await screen.findByText(/已从目录载入 精确修订研究 · r4/)).toBeInTheDocument()
    expect(screen.getByText('与服务端版本一致')).toBeInTheDocument()
    expect(fetchMock.mock.calls.some(([path]) => String(path).includes('/v2/definitions/saved-query?revision=4'))).toBe(true)
  })

  it('从 URL 查询参数实例化模板为独立未保存草稿', async () => {
    vi.stubGlobal('fetch', makeFetch())
    window.history.replaceState({}, '', '/settings/scenario-algorithms/workbench?template=bull-bear-v2')
    render(<StrictMode><HistoricalRegimeWorkbench /></StrictMode>)

    expect(await screen.findByDisplayValue('沪深300自由牛熊研究')).toBeInTheDocument()
    expect(await screen.findByText(/已从目录实例化模板为独立草稿/)).toBeInTheDocument()
    expect(screen.getByText('未保存草稿')).toBeInTheDocument()
  })

  it('从空白图添加动态 schema 节点，并支持撤销、重做与 300ms 检查', async () => {
    const fetchMock = makeFetch()
    vi.stubGlobal('fetch', fetchMock)
    const user = userEvent.setup()
    render(<HistoricalRegimeWorkbench />)
    await settle(340)

    expect(await screen.findByRole('heading', { name: /节点资源库/ })).toBeInTheDocument()
    expect(screen.getByText('空白计算图')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: '添加研究序列' }))
    await settle(340)

    const canvas = screen.getByTestId('regime-graph-canvas')
    expect(within(canvas).getAllByText('研究序列').length).toBeGreaterThan(0)
    expect(screen.getByLabelText('序列 ID *')).toBeInTheDocument()
    await waitFor(() => expect(screen.getByRole('status')).toHaveTextContent(/图谱有效/), { timeout: 2000 })

    await user.click(screen.getByRole('button', { name: '撤销' }))
    expect(screen.getByText('空白计算图')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: '重做' }))
    await settle(340)
    expect(within(canvas).getAllByText('研究序列').length).toBeGreaterThan(0)

    const inferCalls = fetchMock.mock.calls.filter(([path]) => String(path).endsWith('/infer'))
    expect(inferCalls.length).toBeGreaterThan(0)
  })

  it('按服务端目录禁用尚未安装适配器的第三方模型', async () => {
    vi.stubGlobal('fetch', makeFetch())
    render(<HistoricalRegimeWorkbench />)
    expect(await screen.findByText('当前未安装管理员批准的隔离模型适配器')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '添加第三方优化模型' })).toBeDisabled()
  })

  it('把 V1 转换返回的已持久化 V2 定义视为干净版本', async () => {
    vi.stubGlobal('fetch', makeFetch())
    render(<HistoricalRegimeWorkbench initialDefinition={{ ...templateDefinition, id: 'converted-v2', revision: 1 }} />)
    await settle(340)
    await waitFor(() => expect(screen.getByRole('button', { name: '显式预热计划' })).toBeEnabled())
    expect(screen.getByText('与服务端版本一致')).toBeInTheDocument()
  })

  it('模板被实例化为可编辑图，动态参数修改可撤销', async () => {
    vi.stubGlobal('fetch', makeFetch())
    const user = userEvent.setup()
    render(<HistoricalRegimeWorkbench />)
    await settle(340)
    await screen.findByRole('heading', { name: /节点资源库/ })

    await user.selectOptions(screen.getByLabelText('图谱模板'), 'bull-bear-v2')
    await user.click(screen.getByRole('button', { name: '载入模板' }))
    await settle(340)
    expect(await screen.findByText(/每个节点均可继续修改/)).toBeInTheDocument()
    expect(screen.getByDisplayValue('沪深300自由牛熊研究')).toBeInTheDocument()

    const canvas = screen.getByTestId('regime-graph-canvas')
    await user.click(within(canvas).getAllByText('趋势滤波')[0])
    const windowInput = screen.getByLabelText('窗口 *')
    fireEvent.change(windowInput, { target: { value: '60' } })
    await settle(340)
    expect(windowInput).toHaveValue(60)
    await user.click(screen.getByRole('button', { name: '撤销' }))
    await settle(340)
    expect(screen.getByLabelText('窗口 *')).toHaveValue(20)
  })

  it('数据源字段使用中文下拉并提供悬停帮助，研究阶段不绑定使用意图', async () => {
    vi.stubGlobal('fetch', makeFetch())
    const user = userEvent.setup()
    render(<HistoricalRegimeWorkbench />)
    await settle(340)
    await user.selectOptions(screen.getByLabelText('图谱模板'), 'bull-bear-v2')
    await user.click(screen.getByRole('button', { name: '载入模板' }))
    await settle(340)
    await user.click(within(screen.getByTestId('regime-graph-canvas')).getAllByText('沪深300')[0])

    const field = screen.getByRole('combobox', { name: '字段' })
    expect(field).toHaveValue('close')
    expect(within(field).getByRole('option', { name: '收盘点位' })).toBeInTheDocument()
    expect(within(field).getByRole('option', { name: '涨跌幅' })).toBeInTheDocument()
    expect(screen.getByLabelText('字段说明')).toBeInTheDocument()
    expect(screen.queryByLabelText('历史情景使用意图')).not.toBeInTheDocument()
    expect(screen.getByText('图谱不绑定使用意图')).toBeInTheDocument()
    expect(screen.getByTestId('regime-canvas-minimap')).toHaveClass('!bg-indigo-50')
    expect(screen.getByTestId('regime-canvas-minimap')).toHaveAttribute('aria-label', expect.stringContaining('缩略导航'))
  })

  it('复制粘贴所选节点，并保留副本之间的内部连线', async () => {
    vi.stubGlobal('fetch', makeFetch())
    const user = userEvent.setup()
    render(<HistoricalRegimeWorkbench />)
    await settle(340)
    await user.selectOptions(screen.getByLabelText('图谱模板'), 'bull-bear-v2')
    await user.click(screen.getByRole('button', { name: '载入模板' }))
    await settle(340)

    const canvas = screen.getByTestId('regime-graph-canvas')
    await user.click(within(canvas).getAllByText('趋势滤波')[0])
    await user.click(within(canvas).getByRole('button', { name: '复制所选' }))
    await user.click(within(canvas).getByRole('button', { name: '粘贴副本' }))
    expect(await screen.findByText(/已复制 1 个节点/)).toBeInTheDocument()
    expect(within(canvas).getAllByText(/趋势滤波 副本/).length).toBeGreaterThan(0)
  })

  it('按预热、异步轮询执行试算，并读取任意节点的真实序列', async () => {
    const fetchMock = makeFetch()
    vi.stubGlobal('fetch', fetchMock)
    const user = userEvent.setup()
    render(<HistoricalRegimeWorkbench />)
    await settle(340)
    await screen.findByRole('heading', { name: /节点资源库/ })
    await user.selectOptions(screen.getByLabelText('图谱模板'), 'bull-bear-v2')
    await user.click(screen.getByRole('button', { name: '载入模板' }))
    await settle(340)
    await waitFor(() => expect(screen.getByRole('button', { name: '预热并试算' })).toBeEnabled(), { timeout: 2000 })

    await user.click(screen.getByRole('button', { name: '预热并试算' }))
    await settle(720)
    expect(await screen.findByText(/试算完成。可选择任意节点/ , {}, { timeout: 2500 })).toBeInTheDocument()
    expect(fetchMock.mock.calls.some(([path, init]) => String(path).endsWith('/preview-runs/RUN-V2') && !init?.method)).toBe(true)

    await user.selectOptions(screen.getByLabelText('预览节点'), 'ema-1')
    await user.click(screen.getByRole('button', { name: '查看节点结果' }))
    expect(await screen.findByRole('table', { name: '节点预览抽样数据' })).toHaveTextContent('3,510')
    expect(screen.getByTestId('regime-result-mobile-list')).toHaveTextContent('2024-01-03')
    expect(screen.getByText(/试算的固定签名 NJIT 执行链已通过/)).toBeInTheDocument()

    const prepareBody = JSON.parse(String(fetchMock.mock.calls.find(([path]) => String(path).endsWith('/prepare'))?.[1]?.body))
    expect(prepareBody.definition.schema_version).toBe('2.0')
    expect(prepareBody.definition.graph.nodes[0]).not.toHaveProperty('position')
    expect(prepareBody.definition.graph.edges).toContainEqual({ source: { node_id: 'source-1', port: 'value' }, target: { node_id: 'ema-1', port: 'series' } })
  })

  it('运行中可以取消，并向后端发送 DELETE', async () => {
    const fetchMock = makeFetch({ keepRunning: true })
    vi.stubGlobal('fetch', fetchMock)
    const user = userEvent.setup()
    render(<HistoricalRegimeWorkbench />)
    await settle(340)
    await screen.findByRole('heading', { name: /节点资源库/ })
    await user.selectOptions(screen.getByLabelText('图谱模板'), 'bull-bear-v2')
    await user.click(screen.getByRole('button', { name: '载入模板' }))
    await settle(340)
    await waitFor(() => expect(screen.getByRole('button', { name: '预热并试算' })).toBeEnabled(), { timeout: 2000 })
    await user.click(screen.getByRole('button', { name: '预热并试算' }))
    const cancel = await screen.findByRole('button', { name: '取消试算' })
    await act(async () => { await user.click(cancel) })
    expect(await screen.findByText('试算已取消。')).toBeInTheDocument()
    expect(fetchMock.mock.calls.some(([path, init]) => String(path).endsWith('/preview-runs/RUN-V2') && init?.method === 'DELETE')).toBe(true)
  })
})

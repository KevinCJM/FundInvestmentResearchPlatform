import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { StrictMode, type ReactNode } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import HistoricalRegimeWorkbench from './HistoricalRegimeWorkbench'
import type { RegimeGraphDefinition } from '../services/regimeGraph'

vi.mock('./regime-workbench/RegimeResultView', () => ({ default: ({ runId }: { runId: string }) => <div data-testid="final-regime-result">{runId}</div> }))

vi.mock('echarts-for-react', () => ({ default: () => <div data-testid="regime-v2-chart" /> }))
vi.mock('@xyflow/react', () => ({
  ReactFlowProvider: ({ children }: { children?: ReactNode }) => <>{children}</>,
  useReactFlow: () => ({ fitView: vi.fn() }),
  useUpdateNodeInternals: () => () => undefined,
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
    id: 'source.index', label: '研究序列', description: '来自数据实验室的时序', category: 'source', category_label: '数据源', inputs: [], outputs: [{ id: 'value', label: '时序值' }],
    parameter_schema: { type: 'object', properties: { ts_code: { type: 'string', label: '指数代码', default: '' }, field: { type: 'string', label: '字段', default: 'close' } }, required: ['ts_code'] },
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
].map((schema) => ({ ...schema, causal: true, repaints: false, supports_realtime: true }))

const retrospectiveSchema = { ...schemas[2], id: 'model.turning_point', label: '峰谷区间', causal: false, repaints: true, supports_realtime: false }

const templateDefinition: RegimeGraphDefinition = {
  schema_version: '2.0',
  name: '沪深300自由牛熊研究',
  description: '模板实例化草稿',
  graph: {
    nodes: [
      { id: 'source-1', type: 'source.index', label: '沪深300', parameters: { ts_code: '000300.SH', source_api: 'index_daily', field: 'close' }, inputs: {} },
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
    if (path.includes('/api/research-series/catalog')) return ok({ items: [{ id: 'index:index_daily:000300.SH', name: '沪深300', kind: 'index', status: 'available', regime_node_type: 'source.index', fields: [{ name: 'close', label: '收盘点位' }, { name: 'pct_chg', label: '涨跌幅' }], binding_parameters: { ts_code: '000300.SH', source_api: 'index_daily', field: 'close' } }], total: 1, offset: 0, limit: 500 })
    if (path.endsWith('/nodes')) return ok({ items: [...schemas, retrospectiveSchema] })
    if (path.endsWith('/templates/v2')) return ok({ items: [{ id: 'bull-bear-v2', name: '牛熊震荡计算图' }] })
    if (path.endsWith('/v2/definitions')) return ok({ items: [] })
    if (path.includes('/v2/definitions/saved-query?revision=4')) return ok({ ...templateDefinition, id: 'saved-query', revision: 4, name: '精确修订研究' })
    if (path.endsWith('/v2/graph-assets')) return ok({ items: [] })
    if (path.includes('/v2/experiments?definition_id=')) return ok({ items: [] })
    if (path.includes('/templates/bull-bear-v2/instantiate')) return ok({ definition: templateDefinition })
    if (path.endsWith('/authoring/resolve')) return ok({ valid: true, definition: JSON.parse(String(init?.body)).definition, source: '', diagnostics: [], compile_status: 'not_requested', display_latex: { state: 'S_t=1' } })
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

const workspaceTab = (name: string) => within(screen.getByRole('tablist', { name: '历史情景工作区' })).getByRole('tab', { name: new RegExp(`^${name}`) })
const toPreview = (user: ReturnType<typeof userEvent.setup>) => user.click(workspaceTab('校验与预览'))

describe('HistoricalRegimeWorkbench', () => {
  afterEach(() => { window.history.replaceState({}, '', '/'); vi.unstubAllGlobals(); vi.restoreAllMocks() })

  async function ready(fetchMock = makeFetch(), initialDefinition = templateDefinition) {
    vi.stubGlobal('fetch', fetchMock)
    const user = userEvent.setup()
    render(<HistoricalRegimeWorkbench initialDefinition={initialDefinition} />)
    await waitFor(() => expect(screen.getByRole('button', { name: '运行识别', hidden: true })).toBeEnabled())
    return { user, fetchMock }
  }

  async function complete(user: ReturnType<typeof userEvent.setup>) {
    await toPreview(user)
    await user.click(screen.getByRole('button', { name: '运行识别' }))
    await screen.findByTestId('final-regime-result', {}, { timeout: 2500 })
  }

  it('高级公式草稿阻止旧图保存运行，应用后同步到向导', async () => {
    const baseFetch = makeFetch()
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      if (String(input).endsWith('/authoring/resolve')) {
        const body = JSON.parse(String(init?.body))
        const next = JSON.parse(JSON.stringify(body.definition))
        if (body.source_kind === 'formula') next.graph.nodes[1].parameters.window = 60
        return ok({ valid: true, source: body.source_kind === 'formula' ? 'ema = transform_ema(window=60)' : 'ema = transform_ema(window=20)', definition: next, diagnostics: [], compile_status: 'not_requested' })
      }
      return baseFetch(input, init)
    })
    const { user } = await ready(fetchMock)
    await user.click(screen.getByRole('tab', { name: '高级公式' }))
    const input = await screen.findByDisplayValue('ema = transform_ema(window=20)')
    fireEvent.change(input, { target: { value: 'ema = transform_ema(window=60)' } })
    await waitFor(() => expect(screen.getByRole('button', { name: '运行识别', hidden: true })).toBeDisabled())
    expect(screen.getByRole('button', { name: '保存' })).toBeDisabled()
    await user.click(screen.getByRole('tab', { name: '构建向导' }))
    expect(screen.getByLabelText('情景计算公式')).toHaveValue('ema = transform_ema(window=60)')
    await user.click(screen.getByRole('button', { name: '检查并应用公式' }))
    await waitFor(() => expect(screen.getByRole('button', { name: '运行识别', hidden: true })).toBeEnabled())
    await user.click(screen.getByRole('tab', { name: '构建向导' }))
    await user.click(screen.getByRole('button', { name: /2\. 趋势滤波/ }))
    expect(screen.getByLabelText('窗口')).toHaveValue(60)
    expect(fetchMock.mock.calls.filter(([input]) => String(input).endsWith('/preview-runs'))).toHaveLength(0)
  })

  it('默认释放完整画板，资源和参数按需打开，关闭后焦点返回', async () => {
    const { user } = await ready()
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
    expect(screen.getByTestId('regime-graph-desktop-flow')).not.toHaveClass('min-w-[680px]')
    const add = screen.getByRole('button', { name: '添加节点' })
    await user.click(add)
    expect(screen.getByRole('dialog', { name: '添加节点' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '添加第三方优化模型' })).toBeDisabled()
    await user.keyboard('{Escape}')
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
    expect(add).toHaveFocus()
    await user.click(within(screen.getByTestId('regime-graph-canvas')).getAllByText('趋势滤波')[0])
    expect(screen.getByRole('dialog', { name: '节点参数' })).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: '关闭节点参数' }))
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
  })

  it('从 URL 载入精确修订，并在版本抽屉显示已保存身份', async () => {
    const fetchMock = makeFetch()
    vi.stubGlobal('fetch', fetchMock)
    window.history.replaceState({}, '', '/settings/scenario-algorithms/workbench?definition=saved-query&revision=4')
    render(<StrictMode><HistoricalRegimeWorkbench /></StrictMode>)
    expect(await screen.findByDisplayValue('精确修订研究')).toBeInTheDocument()
    fireEvent.click(workspaceTab('校验与预览'))
    fireEvent.click(screen.getByRole('button', { name: '版本与发布' }))
    expect(screen.getByText('与服务端版本一致')).toBeInTheDocument()
    expect(fetchMock.mock.calls.some(([path]) => String(path).includes('/v2/definitions/saved-query?revision=4'))).toBe(true)
  })

  it('从模板开始，并通过引导和画板无损修改同一份计算图', async () => {
    vi.stubGlobal('fetch', makeFetch())
    const user = userEvent.setup()
    render(<HistoricalRegimeWorkbench />)
    await user.click(await screen.findByRole('button', { name: '选择算法：牛熊震荡计算图' }))
    await user.click(screen.getByRole('tab', { name: '构建向导' }))
    expect(await screen.findByDisplayValue('沪深300自由牛熊研究')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: /2\. 趋势滤波/ }))
    fireEvent.change(screen.getByLabelText('窗口'), { target: { value: '60' } })
    await user.click(screen.getByRole('button', { name: '关闭公式构建向导' }))
    await user.click(screen.getByRole('tab', { name: '画布构建' }))
    await user.click(within(screen.getByTestId('regime-graph-canvas')).getAllByText('趋势滤波')[0])
    expect(screen.getByLabelText('窗口 *')).toHaveValue(60)
    await user.click(screen.getByRole('button', { name: '关闭节点参数' }))
    await user.click(screen.getByRole('button', { name: '撤销' }))
    await user.click(screen.getByRole('tab', { name: '构建向导' }))
    await user.click(screen.getByRole('button', { name: /2\. 趋势滤波/ }))
    expect(screen.getByLabelText('窗口')).toHaveValue(20)
  })

  it('算法库迟到的载入结果不能覆盖等待期间的编辑', async () => {
    const fallback = makeFetch()
    let resolveTemplate!: (response: Response) => void
    const pending = new Promise<Response>(resolve => { resolveTemplate = resolve })
    const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => String(input).includes('/templates/bull-bear-v2/instantiate') ? pending : fallback(input, init))
    const { user } = await ready(fetchMock)
    await user.click(screen.getByRole('tab', { name: '构建向导' }))
    await user.click(screen.getByRole('button', { name: '选择算法：牛熊震荡计算图' }))
    fireEvent.change(screen.getByLabelText('研究名称'), { target: { value: '保留我的编辑' } })
    await act(async () => { resolveTemplate(ok({ definition: templateDefinition })); await pending })
    expect(screen.getByLabelText('研究名称')).toHaveValue('保留我的编辑')
    expect(screen.getByText('当前草稿已修改，未覆盖你的编辑。')).toBeInTheDocument()
  })

  it('数据绑定保留中文字段选择，支持节点复制且保留内部连线', async () => {
    const { user, fetchMock } = await ready()
    const canvas = screen.getByTestId('regime-graph-canvas')
    await user.click(within(canvas).getAllByText('沪深300')[0])
    const field = await within(screen.getByRole('dialog')).findByRole('combobox', { name: '字段' })
    expect(await within(field).findByRole('option', { name: '收盘点位' })).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: '关闭节点参数' }))
    await user.click(within(canvas).getAllByText('趋势滤波')[0])
    await user.click(screen.getByRole('button', { name: '关闭节点参数' }))
    await user.click(within(canvas).getByRole('button', { name: '复制所选' }))
    await user.click(within(canvas).getByRole('button', { name: '粘贴副本' }))
    expect(within(canvas).getAllByText('趋势滤波 副本').length).toBeGreaterThan(0)
    await settle(340)
    const body = JSON.parse(String(fetchMock.mock.calls.filter(([path]) => String(path).endsWith('/infer')).slice(-1)[0]?.[1]?.body))
    const copy = body.definition.graph.nodes.find((node: { label: string }) => node.label === '趋势滤波 副本')
    expect(copy.inputs.series).toEqual({ node_id: 'source-1', port: 'value' })
  })

  it('运行完成默认打开最终结果，节点调试使用冻结节点与完整请求身份', async () => {
    const definition = structuredClone(templateDefinition)
    delete definition.graph.nodes[2].label
    const { user, fetchMock } = await ready(makeFetch(), definition)
    await complete(user)
    expect(workspaceTab('校验与预览')).toHaveAttribute('aria-selected', 'true')
    await user.click(workspaceTab('算法定义'))
    await user.click(within(screen.getByTestId('regime-graph-canvas')).getAllByText('趋势滤波')[0])
    fireEvent.change(screen.getByLabelText('节点名称'), { target: { value: '新草稿滤波名称' } })
    await user.click(screen.getByRole('button', { name: '关闭节点参数' }))
    expect(screen.queryByText(/结果已过期/)).not.toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: '节点调试' }))
    const selector = screen.getByLabelText('预览节点')
    expect(within(selector).getByRole('option', { name: '趋势滤波' })).toBeInTheDocument()
    expect(within(selector).getByRole('option', { name: '阈值状态解码' })).toHaveValue('decoder-1')
    expect(within(selector).queryByRole('option', { name: '未命名节点' })).not.toBeInTheDocument()
    expect(within(selector).queryByRole('option', { name: '新草稿滤波名称' })).not.toBeInTheDocument()
    await user.selectOptions(selector, 'ema-1')
    await user.click(screen.getByRole('button', { name: '查看节点结果' }))
    expect(await screen.findByRole('table', { name: '节点预览抽样数据' })).toHaveTextContent('3,510')
    const prepareBody = JSON.parse(String(fetchMock.mock.calls.find(([path]) => String(path).endsWith('/prepare'))?.[1]?.body))
    expect(prepareBody.definition.graph.nodes[0]).not.toHaveProperty('position')
    expect(prepareBody.definition.graph.edges).toContainEqual({ source: { node_id: 'source-1', port: 'value' }, target: { node_id: 'ema-1', port: 'series' } })
  })

  it('模式与截至日让旧结果过期，自动布局和颜色编辑不影响识别结果身份', async () => {
    const { user } = await ready()
    await complete(user)
    await user.click(workspaceTab('算法定义'))
    await user.click(screen.getByRole('button', { name: '自动布局' }))
    await user.click(screen.getByRole('button', { name: '输出通道' }))
    fireEvent.change(screen.getByLabelText('状态1颜色'), { target: { value: '#64748b' } })
    expect(screen.queryByText(/结果已过期/)).not.toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: '关闭输出通道' }))
    await toPreview(user)
    await toPreview(user)
  await user.click(screen.getByRole('radio', { name: '事后研究' }))
    expect(screen.getByText(/结果已过期/)).toBeInTheDocument()
    await toPreview(user)
  await user.click(screen.getByRole('radio', { name: '实时识别' }))
    expect(screen.queryByText(/结果已过期/)).not.toBeInTheDocument()
    fireEvent.change(screen.getByLabelText('V2 截至日'), { target: { value: '2024-01-02' } })
    expect(screen.getByText(/结果已过期/)).toBeInTheDocument()
  })

  it('运行期间主动继续编辑，完成后不强制跳走', async () => {
    const { user } = await ready()
    await toPreview(user)
    await user.click(screen.getByRole('button', { name: '运行识别' }))
    await user.click(workspaceTab('算法定义'))
    await screen.findByRole('button', { name: '查看结果' }, { timeout: 2500 })
    expect(workspaceTab('算法定义')).toHaveAttribute('aria-selected', 'true')
    expect(screen.queryByTestId('final-regime-result')).not.toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: '查看结果' }))
    expect(screen.getByTestId('final-regime-result')).toBeInTheDocument()
  })

  it('取消运行后，迟到的准备请求不能创建旧运行', async () => {
    const original = makeFetch()
    let resolvePrepare!: (response: Response) => void
    const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => String(input).endsWith('/prepare') ? new Promise<Response>((resolve) => { resolvePrepare = resolve }) : original(input, init))
    const { user } = await ready(fetchMock)
    await toPreview(user)
    await user.click(screen.getByRole('button', { name: '运行识别' }))
    await user.click(await screen.findByRole('button', { name: '取消识别' }))
    await act(async () => { resolvePrepare(ok({ plan_id: 'OLD', compile_token: 'OLD', graph_hash: 'hash-3', runtime_audit: fixedExecution })) })
    expect(screen.getByText('识别已取消。')).toBeInTheDocument()
    expect(fetchMock.mock.calls.some(([path, init]) => String(path).endsWith('/preview-runs') && init?.method === 'POST')).toBe(false)
  })

  it('切换预览节点时丢弃迟到响应，并取消前一个请求', async () => {
    const original = makeFetch()
    let resolveOld!: (response: Response) => void
    const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => String(input).includes('/series?') && String(input).includes('node_id=source-1') ? new Promise<Response>((resolve) => { resolveOld = resolve }) : original(input, init))
    const { user } = await ready(fetchMock)
    await complete(user)
    await user.click(workspaceTab('算法定义'))
    await user.click(screen.getByRole('button', { name: '节点调试' }))
    await user.selectOptions(screen.getByLabelText('预览节点'), 'source-1')
    await user.click(screen.getByRole('button', { name: '查看节点结果' }))
    await user.selectOptions(screen.getByLabelText('预览节点'), 'ema-1')
    await user.click(screen.getByRole('button', { name: '查看节点结果' }))
    expect(await screen.findByRole('table', { name: '节点预览抽样数据' })).toHaveTextContent('3,510')
    const oldRequest = fetchMock.mock.calls.find(([path]) => String(path).includes('node_id=source-1'))
    expect(oldRequest?.[1]?.signal?.aborted).toBe(true)
    await act(async () => { resolveOld(ok({ run_id: 'RUN-V2', node_id: 'source-1', items: [{ date: '2024-01-02', value: 99999 }], total: 1, offset: 0, limit: 500, execution: fixedExecution })) })
    expect(screen.getByRole('table', { name: '节点预览抽样数据' })).not.toHaveTextContent('99,999')
  })

  it('运行中取消向服务端发送 DELETE', async () => {
    const { user, fetchMock } = await ready(makeFetch({ keepRunning: true }))
    await toPreview(user)
    await user.click(screen.getByRole('button', { name: '运行识别' }))
    await waitFor(() => expect(fetchMock.mock.calls.some(([path]) => String(path).endsWith('/preview-runs/RUN-V2'))).toBe(true))
    await user.click(screen.getByRole('button', { name: '取消识别' }))
    expect(await screen.findByText('识别已取消。')).toBeInTheDocument()
    expect(fetchMock.mock.calls.some(([path, init]) => String(path).endsWith('/preview-runs/RUN-V2') && init?.method === 'DELETE')).toBe(true)
  })
})


it('实时禁用事后节点，切换模式后恢复可选，已有事后节点阻止运行', async () => {
  const fetchMock = makeFetch()
  vi.stubGlobal('fetch', fetchMock)
  const user = userEvent.setup()
  const definition = structuredClone(templateDefinition)
  definition.graph.nodes.push({ id: 'offline', type: 'model.turning_point', label: '峰谷区间', inputs: {}, parameters: {} })
  render(<HistoricalRegimeWorkbench initialDefinition={definition} />)
  await screen.findByText(/实时识别已禁用事后分析算法/)
  await settle(340)
  expect(screen.getByRole('button', { name: '运行识别', hidden: true })).toBeDisabled()
  await user.click(screen.getByRole('button', { name: '添加节点' }))
  expect(screen.getByRole('button', { name: '添加峰谷区间' })).toBeDisabled()
  await user.click(screen.getByRole('button', { name: '关闭添加节点' }))
  await toPreview(user)
  await user.click(screen.getByRole('radio', { name: '事后研究' }))
  expect(screen.getByRole('button', { name: '运行识别', hidden: true })).toBeEnabled()
  await user.click(workspaceTab('算法定义'))
  await user.click(screen.getByRole('button', { name: '添加节点' }))
  expect(screen.getByRole('button', { name: '添加峰谷区间' })).toBeEnabled()
  await user.click(screen.getByRole('button', { name: '关闭添加节点' }))
  await toPreview(user)
  await user.click(screen.getByRole('radio', { name: '实时识别' }))
  expect(screen.getByRole('button', { name: '运行识别', hidden: true })).toBeDisabled()
  expect(fetchMock.mock.calls.some(([path]) => String(path).endsWith('/preview-runs'))).toBe(false)
})


it('峰谷模板自动进入事后模式，参数可编辑，切回实时后禁用运行', async () => {
  const offline = { ...retrospectiveSchema, id: 'model.peak_trough', label: '峰谷定界法',
    inputs: [{ id: 'value', label: '原始指数', required: true }], outputs: [{ id: 'state', label: '状态' }],
    parameter_schema: { type: 'object', properties: {
      window: { type: 'integer', title: '峰谷左右窗口', default: 8, minimum: 1, deprecated: true },
      left_window: { type: 'integer', title: '左窗口', default: 8, minimum: 1 },
      right_window: { type: 'integer', title: '右窗口', default: 8, minimum: 1 },
      head_window: { type: 'integer', title: '首窗口', default: 6, minimum: 0 },
      tail_window: { type: 'integer', title: '尾窗口', default: 6, minimum: 0 },
      sideways_enabled: { type: 'boolean', title: '启用震荡识别', default: false },
      small_swing_threshold: { type: 'number', title: '小波段幅度门槛', default: .03, minimum: .0001 },
      sideways_max_range: { type: 'number', title: '震荡最大振幅', default: .06, minimum: .0001 },
      sideways_max_efficiency: { type: 'number', title: '震荡方向效率上限', default: .25, minimum: 0 },
      sideways_min_duration: { type: 'integer', title: '最短震荡长度', default: 20, minimum: 2 },
      min_phase: { type: 'integer', title: '最短牛熊阶段', default: 4, minimum: 1 },
      min_cycle: { type: 'integer', title: '最短完整周期', default: 16, minimum: 2 },
      endpoint_window: { type: 'integer', title: '首尾排除窗口', default: 6, minimum: 0, deprecated: true },
      amplitude_exception: { type: 'number', title: '大幅波动例外', default: 0.2, minimum: 0 },
    } },
  }
  const peakDefinition = structuredClone(templateDefinition)
  peakDefinition.name = '峰谷研究'
  peakDefinition.graph.nodes = [peakDefinition.graph.nodes[0], { id: 'dating', type: 'model.peak_trough', parameters: { window: 5, min_phase: 4, min_cycle: 16, endpoint_window: 6, amplitude_exception: 0.2 }, inputs: { value: { node_id: 'source-1', port: 'value' } } }]
  peakDefinition.graph.outputs = { state: { node_id: 'dating', port: 'state' } }
  const fallback = makeFetch()
  const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const path = String(input)
    if (path.endsWith('/nodes')) return ok({ items: [...schemas, offline] })
    if (path.endsWith('/templates/v2')) return ok({ items: [{ id: 'peak-trough-ps-v2', name: '峰谷定界法', default_mode: 'retrospective' }] })
    if (path.endsWith('/templates/peak-trough-ps-v2/instantiate')) return ok({ definition: peakDefinition })
    return fallback(input, init)
  })
  vi.stubGlobal('fetch', fetchMock)
  window.history.replaceState({}, '', '/settings/scenario-algorithms/workbench?template=peak-trough-ps-v2')
  const user = userEvent.setup()
  render(<HistoricalRegimeWorkbench />)
  await screen.findByDisplayValue('峰谷研究')
  await toPreview(user)
  expect(screen.getByRole('radio', { name: '事后研究' })).toHaveAttribute('aria-checked', 'true')
  await user.click(workspaceTab('算法定义'))
  await user.click(screen.getByRole('tab', { name: '构建向导' }))
  await user.click(screen.getByRole('button', { name: /2\. 峰谷定界法/ }))
  fireEvent.change(screen.getByRole('spinbutton', { name: '最短完整周期' }), { target: { value: '24' } })
  fireEvent.change(screen.getByRole('spinbutton', { name: '大幅波动例外' }), { target: { value: '0.3' } })
  expect(screen.getByRole('spinbutton', { name: '左窗口' })).toHaveValue(5)
  expect(screen.getByRole('spinbutton', { name: '右窗口' })).toHaveValue(5)
  expect(screen.queryByRole('spinbutton', { name: '峰谷左右窗口' })).not.toBeInTheDocument()
  expect(screen.queryByRole('spinbutton', { name: '首尾排除窗口' })).not.toBeInTheDocument()
  for (const [name, value] of [['左窗口', '10'], ['右窗口', '2'], ['首窗口', '4'], ['尾窗口', '1'], ['小波段幅度门槛', '.04'], ['震荡最大振幅', '.08'], ['震荡方向效率上限', '.2'], ['最短震荡长度', '30']]) {
    fireEvent.change(screen.getByRole('spinbutton', { name }), { target: { value } })
  }
  await user.click(screen.getByRole('checkbox', { name: '启用震荡识别' }))
  await settle(400)
  const requests = fetchMock.mock.calls.filter(([path]) => String(path).endsWith('/infer'))
  const last = JSON.parse(String(requests.slice(-1)[0]?.[1]?.body))
  await user.click(screen.getByRole('button', { name: '关闭公式构建向导' }))
  expect(last.definition.graph.nodes[1].parameters).toMatchObject({ min_cycle: 24, amplitude_exception: 0.3, left_window: 10, right_window: 2, head_window: 4, tail_window: 1, sideways_enabled: true, small_swing_threshold: .04, sideways_max_range: .08, sideways_max_efficiency: .2, sideways_min_duration: 30 })
  await toPreview(user)
  await user.click(screen.getByRole('radio', { name: '实时识别' }))
  expect(screen.getByRole('button', { name: '运行识别', hidden: true })).toBeDisabled()
  expect(screen.getByText(/实时识别已禁用事后分析算法/)).toBeInTheDocument()
})

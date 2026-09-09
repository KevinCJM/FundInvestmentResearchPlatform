import { test, expect } from '@playwright/test'
import { resultFixture } from '../src/pages/regime-workbench/regimeResultFixtures'

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
].map(schema => ({ ...schema, causal: true, supports_realtime: true, repaints: false, inputs: schema.inputs.map(port => ({ ...port, value_type: 'series<float64>' })), outputs: schema.outputs.map(port => ({ ...port, value_type: port.id === 'state' ? 'state_codes<int64>' : port.id === 'confidence' ? 'confidence<time>' : 'series<float64>' })) }))

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


test('历史情景工作台保留完整画板、按需配置并自动展示完整牛熊结果', async ({ page }, testInfo) => {
  const errors: string[] = []
  page.on('pageerror', error => errors.push(error.message))
  const { overview, rows } = resultFixture('RUN-E2E', 3000)
  overview.mode = 'realtime'
  const requests: Array<{ offset: number; limit: number }> = []
  let runDefinition: typeof templateDefinition | undefined
  await page.route('**/api/**', async route => {
    const url = new URL(route.request().url())
    const path = url.pathname
    const post = route.request().method() === 'POST'
    const body = post ? route.request().postDataJSON() : undefined
    let value: unknown
    if (path === '/api/research-series/catalog') value = { items: [{ id: 'index:index_daily:000300.SH', name: '沪深300', kind: 'index', status: 'available', regime_node_type: 'source.series', fields: [{ name: 'close', label: '收盘点位' }], binding_parameters: { series_id: 'index:index_daily:000300.SH', field: 'close' } }], total: 1, offset: 0, limit: 500 }
    else if (path.endsWith('/nodes')) value = { items: schemas }
    else if (path.endsWith('/templates/v2')) value = { items: [{ id: 'bull-bear-v2', name: '牛熊趋势情景', description: '用因果趋势和上下阈值识别牛熊状态。' }] }
    else if (path.endsWith('/templates/bull-bear-v2/instantiate')) value = { definition: templateDefinition }
    else if (path.endsWith('/authoring/resolve')) value = { valid: true, definition: body.definition, source: '', diagnostics: [], compile_status: 'not_requested', display_latex: { state: String.raw`S_t=\begin{cases}\text{牛市},&r_t>0.01\\\text{熊市},&r_t<-0.01\\\text{震荡},&|r_t|\le 0.01\end{cases}` } }
    else if (path.endsWith('/v2/definitions') || path.endsWith('/v2/graph-assets') || path.endsWith('/v2/experiments')) value = { items: [] }
    else if (path.endsWith('/infer')) value = { valid: true, graph_hash: 'graph-e2e', errors: [], warnings: [], inferred: { nodes: Object.fromEntries(body.definition.graph.nodes.map((node: { id: string }) => [node.id, { causal: true, execution_backend: 'numba_njit_fixed_signature' }])), causal: true, realtime_eligible: true } }
    else if (path.endsWith('/prepare')) value = { plan_id: 'PLAN-E2E', compile_token: 'TOKEN-E2E', graph_hash: 'graph-e2e', prepared_at: '2026-09-06', runtime_audit: fixedExecution }
    else if (path.endsWith('/preview-runs') && post) { runDefinition = body.definition; overview.mode = body.mode; overview.as_of = body.as_of || null; value = { id: 'RUN-E2E', status: 'queued', progress: 0.1 } }
    else if (path.endsWith('/preview-runs/RUN-E2E')) value = { id: 'RUN-E2E', status: 'completed', stage: 'completed', progress: 1, execution: fixedExecution }
    else if (path.endsWith('/preview-runs/RUN-E2E/overview')) value = overview
    else if (path.endsWith('/preview-runs/RUN-E2E/series')) {
      const offset = Number(url.searchParams.get('offset') || 0)
      const limit = Number(url.searchParams.get('limit') || 1000)
      requests.push({ offset, limit })
      value = { run_id: 'RUN-E2E', items: rows.slice(offset, offset + limit), total: rows.length, offset, limit }
    } else return route.fulfill({ status: 404, json: { detail: 'Offline historical regime fixture' } })
    return route.fulfill({ json: value })
  })
  await page.goto('/settings/scenario-algorithms/workbench?template=bull-bear-v2')
  await expect(page.getByLabel('研究名称')).toHaveValue(templateDefinition.name)
  await page.getByRole('tab', { name: '画布构建', exact: true }).click()
  const workspace = page.getByTestId('historical-regime-workbench')
  const canvas = page.getByTestId('regime-canvas-workspace')
  await expect(canvas).toBeVisible()
  await expect(page.getByLabel('当前输出的数学公式').locator('.katex')).toBeVisible()
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1)).toBeTruthy()
  const initial = await canvas.boundingBox()
  const outer = await workspace.boundingBox()
  expect(initial && outer && initial.width / outer.width).toBeGreaterThan(0.5)
  const flow = page.getByTestId('regime-graph-desktop-flow')
  if (testInfo.project.name !== 'mobile-320') {
    await expect(flow).toBeVisible()
    expect((await flow.boundingBox())!.height).toBeGreaterThan(250)
  } else await expect(page.getByTestId('regime-graph-mobile-list')).toBeVisible()

  const add = page.getByRole('button', { name: '添加节点', exact: true })
  await add.click()
  const drawer = page.getByRole('dialog', { name: '添加节点', exact: true })
  await expect(drawer).toBeVisible()
  expect((await canvas.boundingBox())!.width).toBeCloseTo(initial!.width, 0)
  await page.keyboard.press('Escape')
  await expect(drawer).toHaveCount(0)
  await expect(add).toBeFocused()

  await page.getByRole('tab', { name: '构建向导', exact: true }).click()
  const guide = page.getByLabel('情景公式构建向导', { exact: true })
  await expect(guide).toBeVisible()
  await expect(page.getByLabel('当前输出的数学公式').locator('.katex')).toBeVisible()
  await guide.getByRole('button', { name: /2\. 趋势滤波/ }).click()
  const builder = page.getByRole('dialog', { name: '公式构建向导', exact: true })
  await builder.getByRole('spinbutton', { name: '窗口', exact: true }).fill('60')
  await page.getByRole('button', { name: '关闭公式构建向导' }).click()
  await page.getByRole('tab', { name: '画布构建', exact: true }).click()
  await page.getByRole('tab', { name: '构建向导', exact: true }).click()
  await guide.getByRole('button', { name: /2\. 趋势滤波/ }).click()
  await expect(builder.getByRole('spinbutton', { name: '窗口', exact: true })).toHaveValue('60')
  await page.getByRole('button', { name: '关闭公式构建向导' }).click()
  await page.getByRole('tab', { name: '画布构建', exact: true }).click()
  await page.getByRole('button', { name: '专注模式', exact: true }).click()
  const focused = await workspace.boundingBox()
  expect(focused!.width).toBeCloseTo(page.viewportSize()!.width, 0)
  await page.screenshot({ path: testInfo.outputPath('historical-regime-canvas.png') })
  await page.getByRole('button', { name: '退出专注', exact: true }).click()

  await page.getByRole('button', { name: '前往校验与预览' }).click()
  await expect(page.getByRole('button', { name: '运行识别', exact: true })).toBeEnabled()
  await page.getByRole('button', { name: '运行识别', exact: true }).click()
  const result = page.getByLabel('完整历史情景结果')
  await expect(result).toBeVisible()
  await expect(result).toContainText('共 3000 个观测日')
  await expect(result.getByLabel('情景图例')).toContainText('牛市')
  await expect(result.getByLabel('情景图例')).toContainText('熊市')
  await expect(result.locator('canvas').first()).toBeVisible()
  const intervals = result.getByRole('table', { name: '完整情景区间明细' })
  await expect(intervals).toContainText(rows[2999].observation_date)
  await expect(result.getByLabel('状态概率')).toHaveCount(0)
  expect(requests.some(request => request.limit > 500)).toBeTruthy()
  expect(runDefinition!.graph.nodes.map(node => node.id)).toEqual(templateDefinition.graph.nodes.map(node => node.id))
  expect(runDefinition!.graph.nodes.find(node => node.id === 'ema-1')!.parameters).toEqual({ window: 60 })
  expect(runDefinition!.graph.outputs).toEqual(templateDefinition.graph.outputs)
  await page.screenshot({ path: testInfo.outputPath('historical-regime-results.png') })
  await page.getByLabel('V2 截至日').fill('2026-09-05')
  await expect(page.getByText(/结果与当前配置不同（结果已过期）/)).toBeVisible()
  await expect(result).toHaveAttribute('data-run-id', 'RUN-E2E')
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1)).toBeTruthy()
  if (testInfo.project.name === 'desktop-1440') {
    await page.setViewportSize({ width: 1366, height: 768 })
    await page.getByRole('tablist', { name: '历史情景工作区' }).getByRole('tab', { name: /^算法定义/ }).click()
    await expect(flow).toBeVisible()
    expect((await canvas.boundingBox())!.width / (await workspace.boundingBox())!.width).toBeGreaterThan(0.5)
    expect((await flow.boundingBox())!.height).toBeGreaterThan(200)
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1)).toBeTruthy()
  }
  expect(errors).toEqual([])
})

test('正式运行从版本抽屉打开冻结情景图，不混用当前草稿的颜色与截至日', async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== 'desktop-1440', '响应式布局由前一用例覆盖')
  const errors: string[] = []
  page.on('pageerror', error => errors.push(error.message))
  const definition = { ...templateDefinition, id: 'saved-e2e', revision: 2 }
  const { overview, rows } = resultFixture('RUN-FORMAL-E2E', 3000)
  overview.run_kind = 'saved'
  overview.definition_id = definition.id
  overview.definition_revision = 2
  overview.mode = 'realtime'
  overview.as_of = '2025-12-31'
  const formal = {
    id: overview.run_id, schema_version: '2.0', definition_id: definition.id,
    definition_revision: 2, name: '牛熊正式运行', mode: 'realtime',
    created_at: '2026-09-06T08:00:00Z', immutable: true,
    causality: { publish_eligible_usages: ['research_display'] },
    calculation_audits: [fixedExecution], series: rows, overview, publications: [],
  }
  let detailReads = 0
  let previewReads = 0
  await page.route('**/api/**', async route => {
    const url = new URL(route.request().url())
    const path = url.pathname
    let value: unknown
    if (path.endsWith('/nodes')) value = { items: schemas }
    else if (path.endsWith('/templates/v2')) value = { items: [] }
    else if (path.endsWith('/v2/definitions')) value = { items: [definition] }
    else if (path.endsWith('/v2/definitions/saved-e2e')) value = definition
    else if (path.endsWith('/research-series/catalog')) value = { items: [], total: 0, offset: 0, limit: 500 }
    else if (path.endsWith('/infer')) value = { valid: true, graph_hash: 'formal-e2e-graph', errors: [], warnings: [], inferred: { nodes: {}, causal: true, realtime_eligible: true } }
    else if (path.endsWith('/historical-regimes/runs')) value = { items: [{ ...formal, series: undefined, calculation_audits: undefined, series_included: false }] }
    else if (path.endsWith('/historical-regimes/runs/RUN-FORMAL-E2E')) { detailReads += 1; value = formal }
    else if (path.includes('/preview-runs')) { previewReads += 1; return route.fulfill({ status: 400, json: { detail: '正式结果不能读取试算接口' } }) }
    else if (path.endsWith('/v2/graph-assets') || path.endsWith('/v2/experiments')) value = { items: [] }
    else return route.fulfill({ status: 404, json: { detail: 'Offline formal result fixture' } })
    return route.fulfill({ json: value })
  })
  await page.goto('/settings/scenario-algorithms/workbench?definition=saved-e2e&revision=2')
  await expect(page.getByLabel('研究名称')).toHaveValue(definition.name)
  await page.getByRole('button', { name: '前往校验与预览' }).click()
  await page.getByLabel('V2 截至日').fill('2026-09-05')
  await page.getByRole('button', { name: '版本与发布', exact: true }).click()
  const drawer = page.getByRole('dialog', { name: '版本与发布', exact: true })
  await expect(drawer.getByRole('button', { name: /牛熊正式运行/ })).toBeVisible()
  await drawer.getByRole('button', { name: /牛熊正式运行/ }).click()
  await drawer.getByRole('button', { name: '查看完整情景结果', exact: true }).click()
  const result = page.getByLabel('完整历史情景结果')
  await expect(result).toBeVisible()
  await expect(drawer).toHaveCount(0)
  await expect(result).toHaveAttribute('data-run-id', formal.id)
  await expect(page.getByText('正在查看正式运行的冻结版本；其模式、截至日和修订以结果详情为准。')).toBeVisible()
  await expect(result).toContainText('截至 2025-12-31')
  const bull = result.getByLabel('情景图例').locator('span').filter({ hasText: /^牛市$/ }).locator('[aria-hidden="true"]')
  await expect(bull).toHaveCSS('background-color', 'rgb(239, 68, 68)')
  await expect(result.locator('canvas').first()).toBeVisible()
  await expect(result.getByRole('table', { name: '完整情景区间明细' })).toContainText(rows[2999].observation_date)
  expect(detailReads).toBeGreaterThanOrEqual(2)
  expect(previewReads).toBe(0)
  expect(errors).toEqual([])
})

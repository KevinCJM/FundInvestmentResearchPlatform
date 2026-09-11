import { test, expect } from '@playwright/test'

const fixedExecution = {
  execution_backend: 'numba_njit_fixed_signature', nopython: true, object_mode: 0,
  python_fallback: 0, request_time_compilation: 0,
  kernel_signatures: { manual_event_state: ['int64[:],int64[:],int64[:]'] },
}

const schemas = [
  {
    id: 'source.index', label: '指数行情', description: '指数观察序列', category: 'source', category_label: '数据源',
    causal: true, repaints: false, supports_realtime: true, inputs: [], outputs: [{ id: 'value', label: '指数点位', value_type: 'series<float64>' }],
    parameter_schema: { type: 'object', properties: { ts_code: { type: 'string' }, source_api: { type: 'string', default: 'index_daily' }, field: { type: 'string', default: 'close' }, frequency: { type: 'string', default: 'daily', enum: ['daily', 'weekly', 'monthly'], enum_labels: ['日频', '周频', '月频'] }, name: { type: 'string' } }, required: ['ts_code'] },
    execution_policy: { backend: 'io_boundary', njit_required: false, request_time_compilation: 0 },
  },
  {
    id: 'annotation.manual_events', label: '人工历史事件区间', description: '人类定义可重叠历史事件区间', category: 'annotation', category_label: '人工标注',
    causal: false, repaints: false, supports_realtime: false,
    inputs: [{ id: 'value', label: '观察序列', required: true, value_type: 'series<float64>' }],
    outputs: [{ id: 'state', label: '内部事件覆盖状态', value_type: 'state_codes<int64>' }, { id: 'event_count', label: '同时事件数量', value_type: 'series<float64>' }],
    parameter_schema: { type: 'object', properties: { events: { type: 'array', default: [], title: '历史事件区间' } } },
    execution_policy: { backend: 'numba_njit_fixed_signature', njit_required: true, request_time_compilation: 0 },
    granularity: { kind: 'primitive', label: '领域基础能力', expandable: false, reason: '人工事件集合', contract_version: 1 },
  },
]

const manualDefinition = {
  schema_version: '2.0', name: '人工历史事件区间', description: '人类维护可重叠历史事件',
  graph: { nodes: [
    { id: 'market', type: 'source.index', label: '观察指数', parameters: { ts_code: '000300.SH', source_api: 'index_daily', field: 'close', frequency: 'daily', name: '沪深300' }, inputs: {} },
    { id: 'events', type: 'annotation.manual_events', label: '人工历史事件区间', parameters: { events: [] }, inputs: { value: { node_id: 'market', port: 'value' } } },
  ], outputs: { state: { node_id: 'events', port: 'state' } }, exposed_node_ids: ['market', 'events'] },
  states: [
    { id: 'event', label: '事件覆盖', role: 'event', color: '#7c3aed', order: 1 },
    { id: 'normal', label: '事件外', role: 'neutral', color: '#cbd5e1', order: 2 },
  ],
  evaluation_targets: [], validation: { walk_forward: false, folds: 4 }, usage_intent: 'research_display',
}

const rows = Array.from({ length: 10 }, (_, index) => {
  const day = String(index + 1).padStart(2, '0')
  const event = index >= 2 && index <= 7
  return {
    index, observation_date: `2020-01-${day}`, date: `2020-01-${day}`, available_at: `2020-01-${day}`,
    recognized_at: '2020-01-10', effective_date: null, state_id: event ? 'event' : 'normal', state_label: event ? '事件覆盖' : '事件外',
    state_code: event ? 0 : 1, value: 100 + index, confidence: 1, probabilities: {}, features: { event_count: index >= 4 && index <= 5 ? 2 : event ? 1 : 0 }, reasons: [],
  }
})

const manualEvents = [
  { id: 'event_a', label: '次贷危机', start_date: '2020-01-03', end_date: '2020-01-06', color: '#7c3aed', description: '第一段', covered_observations: 4, first_observation_index: 2, last_observation_index: 5, first_observation_date: '2020-01-03', last_observation_date: '2020-01-06' },
  { id: 'event_b', label: '流动性冲击', start_date: '2020-01-05', end_date: '2020-01-08', color: '#dc2626', description: '与前一事件重叠', covered_observations: 4, first_observation_index: 4, last_observation_index: 7, first_observation_date: '2020-01-05', last_observation_date: '2020-01-08' },
]

const overview = {
  schema_version: '2.0', result_kind: 'manual_events', manual_events: manualEvents,
  manual_event_summary: { event_count: 2, covered_observations: 6, overlap_observations: 2, max_concurrent_events: 2 },
  run_kind: 'preview', run_id: 'RUN-MANUAL', definition_id: null, definition_revision: null,
  definition_hash: 'manual-definition', graph_hash: 'manual-graph', mode: 'retrospective', as_of: null,
  data_snapshots: { market: { selected_observations: 10 } }, frequency: 'daily', calendar: 'SSE', time_basis: 'observation', complete: true,
  date_range: { start: '2020-01-01', end: '2020-01-10' }, states: manualDefinition.states,
  segments: [
    { id: 'normal-1', state_id: 'normal', label: '事件外', start_date: '2020-01-01', end_date: '2020-01-02', start_index: 0, end_index: 1, observations: 2, confirmed_at: '2020-01-10', effective_start: null },
    { id: 'event-union', state_id: 'event', label: '事件覆盖', start_date: '2020-01-03', end_date: '2020-01-08', start_index: 2, end_index: 7, observations: 6, confirmed_at: '2020-01-10', effective_start: null },
    { id: 'normal-2', state_id: 'normal', label: '事件外', start_date: '2020-01-09', end_date: '2020-01-10', start_index: 8, end_index: 9, observations: 2, confirmed_at: '2020-01-10', effective_start: null },
  ], unknown_intervals: [],
  summary: { total: 10, classified: 10, unknown: 0, state_counts: { event: 6, normal: 4 }, switch_count: 2, denominator: 'all_observations' },
  primary_series: { source_kind: 'final_series', run_id: 'RUN-MANUAL', node_id: null, port: 'state', value_column: 'value', date_column: 'observation_date', unit: '点', endpoint: '/api/historical-regimes/preview-runs/RUN-MANUAL/series', label: '沪深300', total: 10 },
  capabilities: { observation: { available: true }, effective: { available: false, reason: '人工事件仅用于事后研究。' }, probabilities: { available: false }, confidence: { available: false }, evidence: { available: true } },
}

test('人工历史事件可重叠编辑，并在事后预览中显示独立事件轨道', async ({ page }, testInfo) => {
  const errors: string[] = []
  let submittedDefinition: typeof manualDefinition | undefined
  page.on('pageerror', error => errors.push(error.message))
  await page.route('**/api/**', async route => {
    const url = new URL(route.request().url())
    const path = url.pathname
    const method = route.request().method()
    const body = method === 'POST' || method === 'PUT' ? route.request().postDataJSON() : undefined
    let value: unknown
    if (path.endsWith('/nodes')) value = { items: schemas }
    else if (path.endsWith('/templates/v2')) value = { items: [{ id: 'manual-historical-events-v1', name: '人工历史事件区间', description: '人类定义可重叠历史事件', default_mode: 'retrospective', supported_modes: ['retrospective'] }] }
    else if (path.endsWith('/templates/manual-historical-events-v1/instantiate')) value = { definition: manualDefinition }
    else if (path.endsWith('/v2/definitions')) value = { items: [] }
    else if (path.endsWith('/v2/graph-assets') || path.endsWith('/v2/experiments')) value = { items: [] }
    else if (path.endsWith('/research-series/catalog')) value = { items: [], total: 0, offset: 0, limit: 100 }
    else if (path.endsWith('/authoring/resolve')) value = { valid: true, definition: body?.definition, source: '', diagnostics: [], display_latex: { state: '人工历史事件区间' } }
    else if (path.endsWith('/infer')) value = { valid: true, graph_hash: 'manual-graph', definition_hash: 'manual-definition', errors: [], warnings: [], inferred: { nodes: { market: { causal: true }, events: { causal: false } } } }
    else if (path.endsWith('/prepare')) value = { plan_id: 'PLAN-MANUAL', compile_token: 'TOKEN-MANUAL', graph_hash: 'manual-graph', prepared_at: '2026-09-10', runtime_audit: fixedExecution }
    else if (path.endsWith('/preview-runs') && method === 'POST') { submittedDefinition = body.definition; value = { id: 'RUN-MANUAL', status: 'queued', stage: 'queued', progress: 0.1 } }
    else if (path.endsWith('/preview-runs/RUN-MANUAL')) value = { id: 'RUN-MANUAL', status: 'completed', stage: 'completed', progress: 1, execution: fixedExecution }
    else if (path.endsWith('/preview-runs/RUN-MANUAL/overview')) value = overview
    else if (path.endsWith('/preview-runs/RUN-MANUAL/series')) value = { run_id: 'RUN-MANUAL', items: rows, total: rows.length, offset: Number(url.searchParams.get('offset') || 0), limit: Number(url.searchParams.get('limit') || 5000) }
    else return route.fulfill({ status: 404, json: { detail: `offline fixture: ${path}` } })
    return route.fulfill({ json: value })
  })

  await page.goto('/settings/scenario-algorithms/workbench?template=manual-historical-events-v1')
  await expect(page.getByLabel('研究名称')).toHaveValue('人工历史事件区间')
  await page.getByRole('tab', { name: '构建向导', exact: true }).click()
  await expect(page.getByLabel('结果类型')).toHaveValue('历史事件区间')
  await expect(page.getByLabel('主输出值类型')).toHaveValue('多标签区间（允许重叠）')
  const guide = page.getByLabel('情景公式构建向导')
  await guide.getByRole('button', { name: /2\. 人工历史事件区间/ }).click()
  const drawer = page.getByRole('dialog', { name: '公式构建向导' })
  await drawer.getByRole('button', { name: '添加事件' }).click()
  await drawer.getByLabel('事件1名称').fill('次贷危机')
  await drawer.getByLabel('事件1开始日期').fill('2020-01-03')
  await drawer.getByLabel('事件1结束日期').fill('2020-01-06')
  await drawer.getByRole('button', { name: '添加事件' }).click()
  await drawer.getByLabel('事件2名称').fill('流动性冲击')
  await drawer.getByLabel('事件2开始日期').fill('2020-01-05')
  await drawer.getByLabel('事件2结束日期').fill('2020-01-08')
  await expect(drawer.getByText(/允许重叠/).first()).toBeVisible()
  await page.getByRole('button', { name: '关闭公式构建向导' }).click()

  await page.getByRole('button', { name: '前往校验与预览' }).click()
  await expect(page.getByRole('radio', { name: '事后研究' })).toHaveAttribute('aria-checked', 'true')
  await expect(page.getByRole('button', { name: '运行识别' })).toBeEnabled()
  await page.getByRole('button', { name: '运行识别' }).click()
  const result = page.getByLabel('人工历史事件结果')
  await expect(result).toBeVisible()
  await expect(result).toContainText('重叠观测')
  await expect(result).toContainText('最大同时事件')
  await expect(result.getByRole('table', { name: '人工历史事件明细' })).toContainText('次贷危机')
  await expect(result.getByRole('table', { name: '人工历史事件明细' })).toContainText('流动性冲击')
  expect((submittedDefinition?.graph.nodes.find(node => node.id === 'events')?.parameters.events as Array<{ label: string; start_date: string; end_date: string }>).map(event => [event.label, event.start_date, event.end_date])).toEqual([
    ['次贷危机', '2020-01-03', '2020-01-06'], ['流动性冲击', '2020-01-05', '2020-01-08'],
  ])
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1)).toBeTruthy()
  await page.screenshot({ path: testInfo.outputPath('manual-events.png'), fullPage: true })
  expect(errors).toEqual([])
})

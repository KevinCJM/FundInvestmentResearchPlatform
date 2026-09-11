import { test, expect } from '@playwright/test'
import { libraryEventFixture, temporalFixture } from '../src/test/eventLibraryFixtures'
import { resultFixture } from '../src/pages/regime-workbench/regimeResultFixtures'

const execution = { execution_backend: 'numba_njit_fixed_signature', nopython: true, object_mode: 0, python_fallback: 0, request_time_compilation: 0, kernel_signatures: { temporal_compare: ['fixed'] } }
const definition = { schema_version: '2.0', name: '时点审计测试方案', description: '', graph: {
  nodes: [{ id: 'market', type: 'source.index', inputs: {}, parameters: { ts_code: '000300.SH', frequency: 'daily' } },
    { id: 'classifier', type: 'model.threshold', inputs: { value: { node_id: 'market', port: 'value' } }, parameters: { upper: 101, lower: 99 } }],
  outputs: { state: { node_id: 'classifier', port: 'state' } },
}, states: [{ id: 'bull', label: '牛市', role: 'positive', color: '#16a34a', order: 0 }, { id: 'flat', label: '震荡', role: 'neutral', color: '#64748b', order: 1 }, { id: 'bear', label: '熊市', role: 'negative', color: '#dc2626', order: 2 }], evaluation_targets: [], validation: {}, usage_intent: 'research_display' }
const schemas = [{ id: 'source.index', label: '观察指数', category: 'source', causal: true, repaints: false, supports_realtime: true, temporal_contract: { rule: 'source_availability', reason: '公布时间待核对' }, inputs: [], outputs: [{ id: 'value', value_type: 'series<float64>' }], parameter_schema: { properties: {} } },
  { id: 'model.threshold', label: '阈值分类', category: 'model', causal: true, repaints: false, supports_realtime: true, temporal_contract: { rule: 'history', reason: '历史输入' }, inputs: [{ id: 'value', value_type: 'series<float64>' }], outputs: [{ id: 'state', value_type: 'state_codes<int64>' }], parameter_schema: { properties: {} } }]

test('事件库能选择窗口创建冻结引用，事实未核验清晰可见', async ({ page }, info) => {
  let saved: any
  const errors: string[] = []; page.on('pageerror', error => errors.push(error.message))
  await page.route('**/api/**', async route => {
    const url = new URL(route.request().url()); const path = url.pathname; const method = route.request().method()
    let data: unknown = { items: [] }
    if (path.endsWith('/event-library/events')) data = { items: [libraryEventFixture], total: 1, offset: 0, limit: 20 }
    else if (path.endsWith('/event-library/events/event-demo/history')) data = { items: [libraryEventFixture] }
    else if (path.endsWith('/event-library/packs')) data = { items: [] }
    else if (path.endsWith('/event-library/resolve')) data = { events: [{ id: 'library_demo', label: libraryEventFixture.name, start_date: '2020-01-01', end_date: '2020-01-20', color: '#7c3aed', description: '窗口', library_reference: { event_id: 'event-demo', revision: 1, window_id: 'acute', content_hash: 'a'.repeat(64) } }] }
    else if (path.endsWith('/templates/manual-historical-events-v1/instantiate')) data = { definition: { ...definition, graph: { ...definition.graph, nodes: [definition.graph.nodes[0], { id: 'events', type: 'annotation.manual_events', parameters: { events: [] }, inputs: { value: { node_id: 'market', port: 'value' } } }], outputs: { state: { node_id: 'events', port: 'state' } } } } }
    else if (path.endsWith('/v2/definitions') && method === 'POST') { saved = route.request().postDataJSON().definition; data = { ...saved, id: 'saved-events', revision: 1 } }
    else if (path.endsWith('/infer')) data = { valid: false, errors: [], warnings: [], temporal_capability: temporalFixture }
    return route.fulfill({ json: data })
  })
  await page.goto('/settings/scenario-algorithms')
  await page.getByRole('tab', { name: '全球历史事件库' }).click()
  await page.getByRole('button', { name: /全球供应链事件/ }).click()
  const detail = page.getByRole('article', { name: '事件详情' })
  await expect(detail).toContainText('尚未核实')
  await expect(detail).toContainText('待核验，不是官方标准区间')
  await detail.getByRole('button', { name: '选择此窗口' }).first().click()
  await page.getByRole('button', { name: '创建事后情景' }).click()
  await expect(page.getByRole('link', { name: '打开新建的事后情景' })).toHaveAttribute('href', /definition=saved-events.*mode=retrospective/)
  expect(saved.graph.nodes[1].parameters.events[0].library_reference.revision).toBe(1)
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBe(true)
  expect(errors).toEqual([])
  await page.screenshot({ path: info.outputPath('event-library.png'), fullPage: true })
})

test('时点审计使用后端报告，参数改变后旧证据失效', async ({ page }, info) => {
  const { overview, rows } = resultFixture('audit-preview', 10)
  overview.mode = 'realtime'
  const report = { ...temporalFixture, status: 'realtime_verified', label: '本次时点审计通过', verified: true, numerical_verdict: 'causal', runtime_audit: 'completed', definition_hash: overview.definition_hash, data_checks: { passed: true, sources: [] }, coverage: { executions: 12, comparisons: 36, cutoffs: 3, ports: 3, untested: [] } }
  let requested = false
  const errors: string[] = []; page.on('pageerror', error => errors.push(error.message))
  await page.route('**/api/**', async route => {
    const path = new URL(route.request().url()).pathname
    let data: unknown = { items: [] }
    if (path.endsWith('/nodes')) data = { items: schemas }
    else if (path.endsWith('/templates/v2')) data = { items: [{ id: 'temporal-test', name: definition.name, default_mode: 'realtime' }] }
    else if (path.endsWith('/templates/temporal-test/instantiate')) data = { definition }
    else if (path.endsWith('/infer')) data = { valid: true, errors: [], warnings: [], temporal_capability: temporalFixture }
    else if (path.endsWith('/authoring/resolve')) data = { valid: true, definition, source: '', diagnostics: [], display_latex: { state: 'S_t' } }
    else if (path.endsWith('/prepare')) data = { plan_id: 'p', compile_token: 't', graph_hash: 'g', runtime_audit: execution }
    else if (path.endsWith('/preview-runs')) { requested = route.request().postDataJSON().audit_temporal; data = { id: 'audit-preview', status: 'queued' } }
    else if (path.endsWith('/preview-runs/audit-preview')) data = { id: 'audit-preview', status: 'completed', execution, result: { temporal_capability: report } }
    else if (path.endsWith('/preview-runs/audit-preview/overview')) data = { ...overview, temporal_capability: report }
    else if (path.endsWith('/preview-runs/audit-preview/series')) data = { run_id: 'audit-preview', items: rows, total: rows.length, offset: 0, limit: 5000 }
    return route.fulfill({ json: data })
  })
  await page.goto('/settings/scenario-algorithms/workbench?template=temporal-test')
  await expect(page.getByLabel('研究名称')).toHaveValue(definition.name)
  await page.getByRole('button', { name: '前往校验与预览' }).click()
  const settings = page.getByLabel('情景校验与预览设置')
  await expect(settings.getByRole('button', { name: '因果性审计' })).toBeEnabled()
  await settings.getByRole('button', { name: '因果性审计' }).click()
  await expect(settings.getByRole('heading', { name: '本次时点审计通过' })).toBeVisible()
  expect(requested).toBe(true)
  await page.getByLabel('V2 截至日').fill('2020-02-01')
  await expect(settings.getByRole('heading', { name: '配置已变更，旧审计不再适用' })).toBeVisible()
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBe(true)
  expect(errors).toEqual([])
  await page.screenshot({ path: info.outputPath('temporal-audit.png'), fullPage: true })
})

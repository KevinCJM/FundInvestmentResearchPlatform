import { test, expect } from '@playwright/test'
import type { RegimeGraphDefinition } from '../src/services/regimeGraph'

for (const [kind, label, code, name, queryText, api, defaultField, secondField, defaultLabel, secondLabel] of [
  ['index', '指数行情', '000300.SH', '沪深300指数', '沪深300', 'index_daily', 'close', 'open', '收盘点位', '开盘点位'],
  ['etf', 'ETF行情', '510300.SH', '沪深300ETF', '沪深300', 'fund_daily', 'close', 'adj_nav', '收盘价（不复权）', '复权净值（仅事后分析）'],
  ['fund', '公募基金行情', '000001.OF', '华夏成长', '华夏', 'fund_nav', 'unit_nav', 'accum_nav', '单位净值', '累计净值'],
]) test(`${label}在画布与向导中可搜索且只配置一次`, async ({ page }, testInfo) => {
  const errors: string[] = []
  page.on('pageerror', error => errors.push(error.message))
  const sourceSchema = { id: `source.${kind}`, label, category: 'source', inputs: [], outputs: [{ id: 'value', value_type: 'series<float64>' }], causal: true, supports_realtime: true, parameter_schema: { required: ['ts_code'], properties: {
    ts_code: { type: 'string' }, source_api: { type: 'string', default: api, enum: [api, 'sw_daily'], enum_labels: ['指数日行情', '申万指数日行情'] },
    name: { type: 'string' }, field: { type: 'string', default: defaultField, enum: [defaultField, secondField] }, start_date: { type: 'string' }, snapshot_id: { type: 'string' },
  } } }
  const definition = { schema_version: '2.0', name: '数据源搜索验证', graph: { nodes: [
    { id: 'market', type: `source.${kind}`, label: '市场基准', parameters: {}, inputs: {} },
    { id: 'classifier', type: 'model.threshold', parameters: { upper: 1, lower: -1 }, inputs: { value: { node_id: 'market', port: 'value' } } },
  ], outputs: { state: { node_id: 'classifier', port: 'state' } } }, states: [{ id: 'bull', label: '牛市' }, { id: 'sideways', label: '震荡' }, { id: 'bear', label: '熊市' }], evaluation_targets: [], validation: {} }
  const defaultBinding = { ts_code: code, source_api: api, name, field: defaultField, snapshot_id: 'snapshot-fixture', snapshot_generation: 'fixture', ...(kind === 'etf' ? { source_file: 'etf_daily_candle_df.parquet', file_checksum: 'price-checksum' } : {}) }
  const selectedBinding = { ...defaultBinding, field: secondField, ...(kind === 'etf' ? { source_api: 'fund_nav', source_file: 'etf_daily_df.parquet', file_checksum: 'nav-checksum' } : {}) }
  const series = { id: `${kind}:${api}:${code}`, name, code, kind, regime_node_type: `source.${kind}`, status: 'available', fields: [{ name: defaultField, label: defaultLabel, binding_parameters: defaultBinding }, { name: secondField, label: secondLabel, binding_parameters: selectedBinding }], binding_parameters: defaultBinding }
  const searches: string[] = []
  let lastDefinition: RegimeGraphDefinition | undefined
  await page.route('**/api/**', async route => {
    const url = new URL(route.request().url()), path = url.pathname
    const body = route.request().method() === 'POST' ? route.request().postDataJSON() : undefined
    let value: unknown
    if (path.endsWith('/nodes')) value = { items: [sourceSchema, { id: 'model.threshold', label: '阈值分类', category: 'model', inputs: [{ id: 'value', value_type: 'series<float64>' }], outputs: [{ id: 'state', value_type: 'state_codes<int64>' }], causal: true, supports_realtime: true }] }
    else if (path.endsWith('/templates/v2')) value = { items: [{ id: 'search-test', name: '搜索验证' }] }
    else if (path.endsWith('/templates/search-test/instantiate')) value = { definition }
    else if (path.endsWith('/research-series/catalog')) {
      const query = url.searchParams.get('q') || ''; searches.push(query)
      expect(url.searchParams.get('kind')).toBe(kind)
      value = { items: query ? [series] : [], total: query ? 1 : 14503, offset: 0, limit: 100 }
    } else if (path.endsWith('/infer')) { lastDefinition = body.definition; value = { valid: true, errors: [], warnings: [], inferred: { nodes: {}, realtime_eligible: true } } }
    else if (path.endsWith('/authoring/resolve')) value = { valid: true, definition: body.definition, source: '', diagnostics: [], compile_status: 'not_requested' }
    else value = { items: [] }
    await route.fulfill({ json: value })
  })
  await page.goto('/settings/scenario-algorithms/workbench?template=search-test')
  await expect(page.getByLabel('研究名称')).toHaveValue(definition.name)
  await page.getByRole('tab', { name: '画布构建', exact: true }).click()
  const surface = page.getByTestId(testInfo.project.name === 'mobile-320' ? 'regime-graph-mobile-list' : 'regime-graph-desktop-flow')
  await surface.getByText('市场基准', { exact: true }).click()
  const inspector = page.getByRole('dialog', { name: '节点参数', exact: true })
  await expect(inspector.getByRole('textbox', { name: /指数代码/ })).toHaveCount(0)
  await expect(inspector.getByRole('combobox', { name: '行情数据接口' })).toHaveCount(0)
  await inspector.getByRole('searchbox', { name: '搜索研究数据' }).fill(queryText)
  await expect(inspector.getByLabel('节点研究数据序列').locator(`option[value="${series.id}"]`)).toHaveText(`${name} · ${code}`)
  await inspector.getByLabel('节点研究数据序列').selectOption(series.id)
  await expect(inspector.getByLabel('节点研究数据序列')).toHaveValue(series.id)
  await expect(inspector.getByLabel('数值字段', { exact: true }).locator(`option[value="${secondField}"]`)).toHaveText(secondLabel)
  await inspector.getByLabel('数值字段', { exact: true }).selectOption(secondField)
  await expect.poll(() => lastDefinition?.graph.nodes[0].parameters).toEqual(selectedBinding)
  expect(searches).toContain(queryText)
  expect(searches).toContain(code)
  await inspector.screenshot({ path: testInfo.outputPath(`searchable-${kind}-source.png`) })
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1)).toBeTruthy()
  await page.getByRole('button', { name: '关闭节点参数' }).click()
  await page.getByRole('tab', { name: '构建向导', exact: true }).click()
  await page.getByLabel('情景公式构建向导', { exact: true }).getByRole('button', { name: '1. 市场基准', exact: true }).click()
  const guide = page.getByRole('dialog', { name: '公式构建向导', exact: true })
  await expect(guide.getByRole('searchbox', { name: '搜索研究数据' })).toBeVisible()
  await expect(guide.getByLabel('节点研究数据序列')).toHaveValue(series.id)
  await expect(guide.getByLabel('数值字段', { exact: true })).toHaveValue(secondField)
  await expect(guide.getByRole('textbox', { name: /指数代码/ })).toHaveCount(0)
  await guide.getByLabel('数值字段', { exact: true }).selectOption(defaultField)
  await expect.poll(() => lastDefinition?.graph.nodes[0].parameters).toEqual(defaultBinding)
  await guide.getByLabel('数值字段', { exact: true }).selectOption(secondField)
  await expect.poll(() => lastDefinition?.graph.nodes[0].parameters).toEqual(selectedBinding)
  expect(errors).toEqual([])
})

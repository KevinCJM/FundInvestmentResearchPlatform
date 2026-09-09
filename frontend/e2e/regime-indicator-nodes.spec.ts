import { test, expect } from '@playwright/test'

test('指标计算使用上游输入、多输出，资源分类只有一个下拉', async ({ page }, testInfo) => {
  const errors: string[] = []
  page.on('pageerror', error => errors.push(error.message))
  const series = 'series<float64>'
  const schemas = [
    { id: 'source.index', label: '指数行情', category: 'source', inputs: [], outputs: [{ id: 'value', value_type: series }] },
    { id: 'source.inline', label: '手工输入时序', category: 'source', inputs: [], outputs: [] },
    { id: 'source.indicator', label: '引用指标', category: 'source', inputs: [], outputs: [] },
    { id: 'indicator.calc_bands', label: '20 日布林带', category: 'indicator_calculation', category_label: '指标计算', indicator_reference: { id: 'bands', revision: 2, result_kind: 'time_series', definition_hash: 'hash' },
      inputs: [{ id: 'market_close', label: '收盘价', value_type: series, required: true }], outputs: ['upper', 'middle', 'lower'].map((id, index) => ({ id, label: ['布林上轨', '布林中轨', '布林下轨'][index], value_type: series })) },
  ]
  const definition = { schema_version: '2.0', name: '指数接入指标', graph: { nodes: [{ id: 'market', type: 'source.index', label: '沪深300', parameters: {}, inputs: {} }], outputs: {} }, states: [], evaluation_targets: [], validation: {} }
  let latest: { graph: { nodes: Array<{ type: string; inputs: Record<string, unknown> }> } } = definition
  await page.route('**/api/**', async route => {
    const path = new URL(route.request().url()).pathname
    const body = route.request().method() === 'POST' ? route.request().postDataJSON() : undefined
    let data: unknown = { items: [] }
    if (path.endsWith('/nodes')) data = { items: schemas }
    else if (path.endsWith('/templates/v2')) data = { items: [{ id: 'indicators', name: definition.name }] }
    else if (path.endsWith('/templates/indicators/instantiate')) data = { definition }
    else if (path.endsWith('/infer')) { latest = body.definition; data = { valid: false, errors: [], warnings: [] } }
    else if (path.endsWith('/authoring/resolve')) data = { valid: false, definition: body.definition, source: '', diagnostics: [], compile_status: 'not_requested' }
    await route.fulfill({ json: data })
  })
  await page.goto('/settings/scenario-algorithms/workbench?template=indicators')
  await expect(page.getByLabel('研究名称')).toHaveValue(definition.name)
  await page.getByRole('tab', { name: '画布构建', exact: true }).click()
  await page.getByRole('button', { name: '添加节点', exact: true }).click()
  const library = page.getByRole('dialog', { name: '添加节点', exact: true })
  await expect(library.getByRole('combobox')).toHaveCount(1)
  await expect(library.getByText('手工输入时序', { exact: true })).toHaveCount(0)
  await expect(library.getByText('引用指标', { exact: true })).toHaveCount(0)
  await library.getByLabel('节点类型').selectOption('indicator_calculation')
  await library.screenshot({ path: testInfo.outputPath('indicator-library.png') })
  await library.getByRole('button', { name: '添加20 日布林带 第2版' }).click()
  const inspector = page.getByRole('dialog', { name: '节点参数', exact: true })
  await inspector.getByLabel('收盘价 *上游节点').selectOption('market')
  await expect.poll(() => latest.graph.nodes.find(node => node.type === 'indicator.calc_bands')?.inputs).toEqual({ market_close: { node_id: 'market', port: 'value' } })
  await expect(inspector.getByLabel('节点研究数据序列')).toHaveCount(0)
  await expect(inspector.getByLabel('指标计算对象', { exact: true })).toHaveCount(0)
  await expect(inspector.getByText(/提供 3 个输出/)).toBeVisible()
  await inspector.screenshot({ path: testInfo.outputPath('indicator-upstream.png') })
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBeTruthy()
  expect(errors).toEqual([])
})

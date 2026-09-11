import { test, expect } from '@playwright/test'

test('已有数据兼容与上传文件确认', async ({ page }, testInfo) => {
  const errors: string[] = []
  page.on('pageerror', error => errors.push(error.message))
  const names = { inline: '手工输入时序', upload: '上传时序', indicator: '引用指标' }
  const schemas = Object.entries(names).map(([kind, label]) => ({ id: `source.${kind}`, label, category: 'source', inputs: [], outputs: [{ id: 'value', value_type: 'series<float64>' }], parameter_schema: { properties: {
    frequency: { type: 'string', enum: ['daily', 'monthly'], enum_labels: ['日频', '月频'], default: 'daily' },
    ...(kind === 'inline' ? { rows: { type: 'array' }, inline_rows: { type: 'array' } } : kind === 'upload' ? { artifact_id: { type: 'string' }, checksum: { type: 'string' } } : { indicator_id: { type: 'string' }, product_kind: { type: 'string' }, product_id: { type: 'string' }, period: { type: 'string' } }),
  } } }))
  const definition = { schema_version: '2.0', name: '数据源表单验证', graph: { nodes: Object.entries(names).map(([id, label]) => ({ id, label, type: `source.${id}`, parameters: id === 'inline' ? { frequency: 'daily', rows: [{ observation_date: '2024-01-02', available_at: '2024-01-02', value: 100 }, { observation_date: '2024-01-03', available_at: '2024-01-03', value: 101 }] } : {}, inputs: {} })), outputs: {} }, states: [], evaluation_targets: [], validation: {} }
  const indicator = { id: 'indicator:return@2', name: '年度收益率', kind: 'indicator', status: 'available', regime_node_type: 'source.indicator', product_kinds: ['fund'], periods: ['1Y'], indicator_version: { indicator_id: 'return', revision: 2 }, binding_parameters: { indicator_id: 'return', indicator_revision: 2, name: '年度收益率', period: '1Y', product_kind: '', product_id: '' } }
  const binding = { artifact_id: 'upload-sha256-test', checksum: 'sha256:test', name: '指数行情', frequency: 'monthly' }
  let saved = false
  let latest = definition
  await page.route('**/api/**', async route => {
    const path = new URL(route.request().url()).pathname
    let data: unknown
    if (path.endsWith('/parse-file')) data = { columns: ['日期', '收盘价'], rows: [{ 日期: '2024-01-02', 收盘价: 100 }, { 日期: '2024-01-03', 收盘价: 101 }], sheets: [], sheet: null }
    else {
      const body = route.request().method() === 'POST' ? route.request().postDataJSON() : undefined
      if (path.endsWith('/nodes')) data = { items: schemas }
      else if (path.endsWith('/templates/v2')) data = { items: [{ id: 'forms', name: '表单验证' }] }
      else if (path.endsWith('/templates/forms/instantiate')) data = { definition }
      else if (path.endsWith('/infer')) { latest = body.definition; data = { valid: true, errors: [], warnings: [], inferred: { nodes: {}, realtime_eligible: true } } }
      else if (path.endsWith('/authoring/resolve')) data = { valid: true, definition: body.definition, source: '', diagnostics: [], compile_status: 'not_requested' }
      else if (path.endsWith('/research-series/catalog')) data = { items: [indicator], total: 1, offset: 0, limit: 100 }
      else if (path.endsWith('/research-series/uploads')) data = { items: saved ? [{ id: 'upload:test', kind: 'upload', name: '指数行情', status: 'available', regime_node_type: 'source.upload', binding_parameters: binding }] : [], total: saved ? 1 : 0, offset: 0, limit: 100 }
      else if (path.endsWith('/instruments/products')) data = { items: [{ ts_code: '000001.OF', name: '示例基金' }], total: 1 }
      else if (path.endsWith('/research-series/profile')) {
        expect(body.frequency).toBe('monthly')
        expect(body.inline_rows).toEqual([{ date: '2024-01-02', value: 100 }, { date: '2024-01-03', value: 101 }])
        saved = true
        data = { binding_parameters: binding, dates: [], values: {}, execution: { execution_backend: 'numba_njit_fixed_signature', nopython: true, object_mode: 0, python_fallback: 0, request_time_compilation: 0, kernel_signatures: { profile: ['float64[:]'] } } }
      } else data = { items: [] }
    }
    await route.fulfill({ json: data })
  })
  await page.goto('/settings/scenario-algorithms/workbench?template=forms')
  await expect(page.getByLabel('研究名称')).toHaveValue(definition.name)
  await page.getByRole('tab', { name: '画布构建', exact: true }).click()
  const surface = page.getByTestId(testInfo.project.name === 'mobile-320' ? 'regime-graph-mobile-list' : 'regime-graph-desktop-flow')
  const inspector = page.getByRole('dialog', { name: '节点参数', exact: true })
  await surface.getByText('手工输入时序', { exact: true }).click()
  await expect(inspector.getByText(/若要更换数据/)).toBeVisible()
  await page.getByRole('button', { name: '关闭节点参数' }).click()
  await surface.getByText('上传时序', { exact: true }).click()
  await inspector.getByLabel('选择时序文件').setInputFiles({ name: '指数行情.csv', mimeType: 'text/csv', buffer: Buffer.from('日期,收盘价\n2024-01-02,100\n2024-01-03,101') })
  await expect(inspector.getByRole('table', { name: '文件列预览' })).toBeVisible()
  await inspector.getByLabel('数据频率', { exact: true }).selectOption('monthly')
  await expect(inspector.getByRole('table', { name: '文件列预览' })).toBeVisible()
  await inspector.screenshot({ path: testInfo.outputPath('upload-columns.png') })
  await inspector.getByRole('button', { name: '保存并使用' }).click()
  await expect.poll(() => latest.graph.nodes.find(node => node.id === 'upload')?.parameters).toEqual(binding)
  await expect(inspector.getByLabel('节点研究数据序列')).toHaveValue('upload:test')
  await page.getByRole('button', { name: '关闭节点参数' }).click()
  await surface.getByText('引用指标', { exact: true }).click()
  await inspector.getByLabel('节点研究数据序列').selectOption(indicator.id)
  await expect(inspector.getByLabel('指标对象类型').locator('option[value="fund"]')).toHaveCount(1)
  await inspector.getByLabel('指标对象类型').selectOption('fund')
  await inspector.getByLabel('搜索计算对象').fill('示例')
  await expect(inspector.locator('select[aria-label="指标计算对象"] option[value="000001.OF"]')).toHaveCount(1)
  await inspector.locator('select[aria-label="指标计算对象"]').selectOption('000001.OF')
  await expect(inspector.getByLabel('指标计算窗口')).toHaveValue('1Y')
  await inspector.screenshot({ path: testInfo.outputPath('indicator-target.png') })
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1)).toBeTruthy()
  expect(errors).toEqual([])
})

import { test, expect } from '@playwright/test'

test('未完成公式可从画布与向导预览单个指数', async ({ page }, testInfo) => {
  const errors: string[] = []
  page.on('pageerror', error => errors.push(error.message))
  const definition = { schema_version: '2.0', name: '仅有沪深300', graph: { nodes: [
    { id: 'market', type: 'source.index', label: '沪深300', parameters: { ts_code: '000300.SH', source_api: 'index_daily' }, inputs: {} },
  ], outputs: {} }, states: [], evaluation_targets: [], validation: {} }
  const audit = { execution_backend: 'numba_njit_fixed_signature', nopython: true, object_mode: 0, python_fallback: 0, request_time_compilation: 0, kernel_signatures: { graph: ['fixed'] } }
  let previewRequests = 0
  const normalizationBases: number[] = []
  await page.route('**/api/**', async route => {
    const request = route.request(), path = new URL(request.url()).pathname
    const body = request.method() === 'POST' ? request.postDataJSON() : undefined
    let value: unknown
    if (path.endsWith('/nodes')) value = { items: [{ id: 'source.index', label: '指数行情', category: 'source', inputs: [], outputs: [{ id: 'value', value_type: 'series<float64>' }], causal: true, supports_realtime: true, parameter_schema: { properties: { ts_code: { type: 'string' } } } }] }
    else if (path.endsWith('/templates/v2')) value = { items: [{ id: 'partial', name: definition.name }] }
    else if (path.endsWith('/templates/partial/instantiate')) value = { definition }
    else if (path.endsWith('/infer')) value = { valid: false, errors: [{ code: 'MISSING_STATE_OUTPUT', message: '缺少最终输出' }], warnings: [] }
    else if (path.endsWith('/authoring/resolve')) value = { valid: false, definition: body.definition, source: '', diagnostics: [], compile_status: 'not_requested' }
    else if (path.endsWith('/prepare') || path.endsWith('/preview-runs')) {
      expect(body.definition.graph.outputs).toEqual({})
      expect(body.definition.graph.nodes).toHaveLength(1)
      expect(body.preview_target).toEqual({ node_id: 'market', port: 'value' })
      if (path.endsWith('/prepare')) value = { compile_token: 'node-token', runtime_audit: audit }
      else { previewRequests++; value = { id: 'node-run', status: 'queued' } }
    } else if (path.endsWith('/preview-runs/node-run')) value = { id: 'node-run', status: 'completed', execution: audit }
    else if (path.endsWith('/preview-runs/node-run/normalized-chart')) {
      const base = Number(new URL(request.url()).searchParams.get('base_index')); normalizationBases.push(base)
      const values = [3500, 4000, 4200], dates = ['2010-01-04', '2020-01-02', '2026-09-07']
      value = { run_id: 'node-run', node_id: 'market', port: 'value', base_index: base, base_date: dates[base], base_value: values[base], values: values.map(value => value / values[base]), change_pct: values.map(value => (value / values[base] - 1) * 100), execution: audit }
    }
    else if (path.endsWith('/preview-runs/node-run/series')) value = { value_type: 'series<float64>', id: 'node-run', node_id: 'market', port: 'value', offset: 0, limit: 5000, total: 3, items: [{ observation_date: '2010-01-04', value: 3500 }, { observation_date: '2020-01-02', value: 4000 }, { observation_date: '2026-09-07', value: 4200 }] }
    else { expect(path).not.toContain('/overview'); value = { items: [] } }
    await route.fulfill({ json: value })
  })
  await page.goto('/settings/scenario-algorithms/workbench?template=partial')
  await expect(page.getByLabel('研究名称')).toHaveValue(definition.name)
  await page.getByRole('tab', { name: '画布构建', exact: true }).click()
  const surface = page.getByTestId(testInfo.project.name === 'mobile-320' ? 'regime-graph-mobile-list' : 'regime-graph-desktop-flow')
  await surface.getByText('沪深300', { exact: true }).click()
  await page.getByRole('dialog', { name: '节点参数', exact: true }).getByRole('button', { name: '预览此节点', exact: true }).click()
  const preview = page.getByRole('dialog', { name: '节点预览', exact: true })
  await expect(preview.getByLabel('待预览节点')).toHaveValue('market')
  await preview.getByRole('button', { name: '预览节点数据' }).click()
  await expect(preview.getByText('已返回 3 / 3 条节点结果')).toBeVisible()
  await expect(preview.locator('canvas')).toHaveCount(1)
  if (testInfo.project.name === 'desktop-1440') {
    const viewport = page.viewportSize()!
    await page.setViewportSize({ ...viewport, height: 1600 })
    await expect.poll(async () => (await preview.getByRole('region', { name: '节点预览结果区' }).boundingBox())!.y + (await preview.getByRole('region', { name: '节点预览结果区' }).boundingBox())!.height).toBeGreaterThanOrEqual(1576)
    await expect.poll(async () => (await preview.locator('canvas').boundingBox())!.height).toBeGreaterThan(700)
    await preview.screenshot({ path: testInfo.outputPath('node-preview-fill-height.png') })
    await page.setViewportSize(viewport)
  }
  await preview.getByText('查看节点数据（原值）', { exact: true }).click()
  const table = testInfo.project.name === 'mobile-320' ? preview.getByTestId('regime-result-mobile-list') : preview.getByRole('table')
  await expect(table.getByText('2026-09-07', { exact: true })).toBeVisible()
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1)).toBeTruthy()
  const toggle = preview.getByRole('switch', { name: '区间归一化（首日＝1）' })
  await toggle.click()
  await expect(preview.getByText(/基准日 2010-01-04/)).toContainText('涨跌幅 20%')
  await preview.locator('canvas').scrollIntoViewIfNeeded()
  const box = (await preview.locator('canvas').boundingBox())!
  await page.mouse.move(box.x + 61, box.y + box.height - 8)
  await page.mouse.down()
  await page.mouse.move(box.x + box.width * 0.68, box.y + box.height - 8, { steps: 8 })
  await page.mouse.up()
  await expect.poll(() => normalizationBases[normalizationBases.length - 1]).toBeGreaterThan(0)
  await expect(preview.getByText(/基准日 2020-01-02|基准日 2026-09-07/)).toBeVisible()
  await preview.screenshot({ path: testInfo.outputPath('normalized-index-preview.png') })
  await toggle.click()
  await expect(toggle).toHaveAttribute('aria-checked', 'false')
  await page.getByRole('button', { name: '关闭节点预览' }).click()
  await page.getByRole('tab', { name: '构建向导', exact: true }).click()
  await page.getByLabel('情景公式构建向导', { exact: true }).getByRole('button', { name: '修改计算逻辑', exact: true }).click()
  await page.getByRole('dialog', { name: '公式构建向导', exact: true }).getByRole('button', { name: /预览/ }).click()
  await expect(preview.getByRole('button', { name: '预览节点数据' })).toBeEnabled()
  expect(previewRequests).toBe(1)
  expect(errors).toEqual([])
})

import { test, expect } from '@playwright/test'

test('卡尔曼节点可叠加上游指数并联动区间归一化', async ({ page }, testInfo) => {
  const errors: string[] = []
  page.on('pageerror', error => errors.push(error.message))
  const definition = { schema_version: '2.0', name: '指数与卡尔曼对比', graph: { nodes: [
    { id: 'market', type: 'source.index', label: '沪深300', parameters: { ts_code: '000300.SH', source_api: 'index_daily' }, inputs: {} },
    { id: 'filtered', type: 'filter.kalman', label: '卡尔曼滤波', inputs: { value: { node_id: 'market', port: 'value' } }, parameters: {} },
  ], outputs: {} }, states: [], evaluation_targets: [], validation: {} }
  const audit = { execution_backend: 'numba_njit_fixed_signature', nopython: true, object_mode: 0, python_fallback: 0, request_time_compilation: 0, kernel_signatures: { graph: ['fixed'] } }
  let previewRequests = 0
  const normalizationBases: Array<{node: string; base: number}> = []
  let seriesRequests = 0
  await page.route('**/api/**', async route => {
    const request = route.request(), path = new URL(request.url()).pathname
    const body = request.method() === 'POST' ? request.postDataJSON() : undefined
    let value: unknown
    if (path.endsWith('/nodes')) value = { items: [{ id: 'source.index', label: '指数行情', category: 'source', inputs: [], outputs: [{ id: 'value', value_type: 'series<float64>' }], causal: true, supports_realtime: true, parameter_schema: { properties: { ts_code: { type: 'string' } } } }, { id: 'filter.kalman', label: '卡尔曼滤波', category: 'filter', inputs: [{ id: 'value', value_type: 'series<float64>' }], outputs: [{ id: 'value', value_type: 'series<float64>' }], causal: true, supports_realtime: true, parameter_schema: { properties: { process_variance: { type: 'number', label: '过程噪声方差', default: 0.00001, description: '用于配置单边卡尔曼滤波的过程噪声方差，说明需要完整显示，不能被侧栏边缘裁切。' } } } }] }
    else if (path.endsWith('/templates/v2')) value = { items: [{ id: 'partial', name: definition.name }] }
    else if (path.endsWith('/templates/partial/instantiate')) value = { definition }
    else if (path.endsWith('/infer')) value = { valid: false, errors: [{ code: 'MISSING_STATE_OUTPUT', message: '缺少最终输出' }], warnings: [] }
    else if (path.endsWith('/authoring/resolve')) value = { valid: false, definition: body.definition, source: '', diagnostics: [], compile_status: 'not_requested' }
    else if (path.endsWith('/prepare') || path.endsWith('/preview-runs')) {
      expect(body.definition.graph.outputs).toEqual({})
      expect(body.definition.graph.nodes).toHaveLength(2)
      expect(body.preview_target).toEqual({ node_id: 'filtered', port: 'value' })
      if (path.endsWith('/prepare')) value = { compile_token: 'node-token', runtime_audit: audit }
      else { previewRequests++; value = { id: 'node-run', status: 'queued' } }
    } else if (path.endsWith('/preview-runs/node-run')) value = { id: 'node-run', status: 'completed', execution: audit }
    else if (path.endsWith('/preview-runs/node-run/normalized-chart')) {
      const query = new URL(request.url()).searchParams; const base = Number(query.get('base_index')); const node = query.get('node_id')!; normalizationBases.push({node, base})
      const values = node === 'market' ? [3500, 4000, 4200] : [3500, 3900, 4100], dates = ['2010-01-04', '2020-01-02', '2026-09-07']
      value = { run_id: 'node-run', node_id: node, port: 'value', base_index: base, base_date: dates[base], base_value: values[base], values: values.map(value => value / values[base]), change_pct: values.map(value => (value / values[base] - 1) * 100), execution: audit }
    }
    else if (path.endsWith('/preview-runs/node-run/series')) {
      seriesRequests++
      const node = new URL(request.url()).searchParams.get('node_id')!
      const values = node === 'market' ? [3500, 4000, 4200] : [3500, 3900, 4100]
      value = { value_type: 'series<float64>', id: 'node-run', node_id: node, node_label: node === 'market' ? '沪深300' : '卡尔曼滤波', port: 'value', offset: 0, limit: 5000, total: 3, items: ['2010-01-04', '2020-01-02', '2026-09-07'].map((date, index) => ({ observation_date: date, value: values[index] })), upstream_outputs: node === 'market' ? [] : [{ node_id: 'market', node_label: '沪深300', port: 'value', port_label: '数值序列', value_type: 'series<float64>', distance: 1, plottable: true }] }
    }
    else { expect(path).not.toContain('/overview'); value = { items: [] } }
    await route.fulfill({ json: value })
  })
  await page.goto('/settings/scenario-algorithms/workbench?template=partial')
  await expect(page.getByLabel('研究名称')).toHaveValue(definition.name)
  await page.getByRole('tab', { name: '画布构建', exact: true }).click()
  const surface = page.getByTestId(testInfo.project.name === 'mobile-320' ? 'regime-graph-mobile-list' : 'regime-graph-desktop-flow')
  await surface.getByText('卡尔曼滤波', { exact: true }).click()
  const inspector = page.getByRole('dialog', { name: '节点参数', exact: true })
  const help = inspector.getByRole('button', { name: '过程噪声方差说明' })
  await help.hover()
  const tooltip = page.getByRole('tooltip')
  await expect(tooltip).toContainText('不能被侧栏边缘裁切')
  await tooltip.hover()
  expect(await tooltip.evaluate(element => {
    const box = element.getBoundingClientRect(), cx = box.x + box.width / 2, cy = box.y + box.height / 2
    return [[box.left + 4, cy], [box.right - 4, cy], [cx, box.top + 4], [cx, box.bottom - 4]].every(([x, y]) => element.contains(document.elementFromPoint(x, y)))
  })).toBe(true)
  expect(await tooltip.evaluate(element => element.parentElement === document.body)).toBe(true)
  const tipBox = (await tooltip.boundingBox())!
  expect(tipBox.x).toBeGreaterThanOrEqual(8)
  expect(tipBox.y).toBeGreaterThanOrEqual(8)
  expect(tipBox.x + tipBox.width).toBeLessThanOrEqual(page.viewportSize()!.width - 7)
  expect(tipBox.y + tipBox.height).toBeLessThanOrEqual(page.viewportSize()!.height - 7)
  await page.screenshot({ path: testInfo.outputPath('parameter-help-unclipped.png') })
  await page.keyboard.press('Escape')
  await expect(tooltip).toHaveCount(0)
  await expect(inspector).toBeVisible()
  await inspector.getByRole('button', { name: '预览此节点', exact: true }).click()
  const preview = page.getByRole('dialog', { name: '节点预览', exact: true })
  await expect(preview.getByLabel('待预览节点')).toHaveValue('filtered')
  await preview.getByRole('button', { name: '预览节点数据' }).click()
  await expect(preview.getByText('已返回 3 / 3 条节点结果')).toBeVisible()
  await expect(preview.locator('canvas')).toHaveCount(1)
  await expect(preview.getByRole('button', { name: '紧凑', exact: true })).toHaveCount(0)
  await expect(preview.getByRole('button', { name: '标准', exact: true })).toHaveCount(0)
  await expect(preview.getByRole('separator', { name: '调整结果区高度' })).toHaveCount(0)
  expect(await preview.getByTestId('regime-result-content').evaluate(element => getComputedStyle(element).overflowY)).toBe('visible')
  expect(await preview.getByRole('region', { name: '节点预览结果区' }).evaluate(element => element.getBoundingClientRect().bottom >= window.innerHeight - 24)).toBe(true)
  await preview.getByText('查看节点数据（原值）', { exact: true }).click()
  const table = testInfo.project.name === 'mobile-320' ? preview.getByTestId('regime-result-mobile-list') : preview.getByRole('table')
  await expect(table.getByText('2026-09-07', { exact: true })).toBeVisible()
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1)).toBeTruthy()
  await preview.getByText('查看节点数据（原值）', { exact: true }).click()
  await preview.getByText('叠加已计算节点', { exact: true }).click()
  await preview.getByRole('checkbox', { name: /沪深300/ }).check()
  const placement = preview.getByRole('combobox', { name: /沪深300.*展示位置/ })
  await expect(placement).toHaveValue('subplot')
  await expect.poll(async () => (await preview.locator('canvas').boundingBox())!.height).toBeGreaterThanOrEqual(520)
  await expect(preview.getByText('正在读取对比数据…')).toHaveCount(0)
  const toggle = preview.getByRole('switch', { name: '区间归一化（首日＝1）' })
  await toggle.click()
  await expect(preview.getByText(/沪深300.*基准日 2010-01-04/)).toBeVisible()
  await preview.locator('canvas').evaluate(element => element.scrollIntoView({ block: 'end' }))
  const box = (await preview.locator('canvas').boundingBox())!
  await page.mouse.move(box.x + 61, box.y + box.height - 8)
  await page.mouse.down()
  await page.mouse.move(box.x + box.width * 0.68, box.y + box.height - 8, { steps: 8 })
  await page.mouse.up()
  await expect.poll(() => normalizationBases.filter(item => item.base > 0).length).toBe(2)
  const last = normalizationBases.slice(-2)
  expect(last[0].base).toBe(last[1].base)
  await preview.screenshot({ path: testInfo.outputPath('upstream-subplots.png') })
  const requestsBeforeFullscreen = { series: seriesRequests, normalization: normalizationBases.length, previews: previewRequests }
  const normalizedSummary = await preview.getByRole('status').textContent()
  await preview.getByRole('button', { name: '全屏展开', exact: true }).click()
  const fullscreen = page.getByRole('dialog', { name: '节点图表全屏', exact: true })
  await expect(fullscreen).toBeVisible()
  await expect(fullscreen.getByRole('button', { name: '退出全屏' })).toBeFocused()
  expect(await fullscreen.evaluate(element => element.matches(':modal'))).toBe(true)
  expect(await fullscreen.boundingBox()).toMatchObject({ x: 0, y: 0, ...page.viewportSize()! })
  await expect(fullscreen.getByRole('switch')).toHaveAttribute('aria-checked', 'true')
  await expect(fullscreen.getByRole('status')).toHaveText(normalizedSummary!)
  const fullCanvas = fullscreen.locator('canvas')
  await expect.poll(async () => (await fullCanvas.boundingBox())!.height).toBeGreaterThanOrEqual(440)
  await expect.poll(async () => Math.abs((await fullCanvas.boundingBox())!.height - (await fullscreen.getByTestId('regime-node-plot').boundingBox())!.height)).toBeLessThan(2)
  const beforeResize = (await fullCanvas.boundingBox())!.height
  const viewport = page.viewportSize()!
  await page.setViewportSize({ ...viewport, height: viewport.height + 200 })
  await expect.poll(async () => (await fullCanvas.boundingBox())!.height).toBeGreaterThan(beforeResize + 50)
  await page.setViewportSize(viewport)
  await expect.poll(async () => (await fullCanvas.boundingBox())!.height).toBeCloseTo(beforeResize, 0)
  await fullscreen.screenshot({ path: testInfo.outputPath('upstream-fullscreen.png') })
  await page.keyboard.press('Shift+Tab')
  expect(await fullscreen.evaluate(element => element.contains(document.activeElement))).toBe(true)
  await page.keyboard.press('Escape')
  await expect(fullscreen).toHaveCount(0)
  await expect(preview).toBeVisible()
  await expect(preview.getByRole('button', { name: '全屏展开' })).toBeFocused()
  await expect(toggle).toHaveAttribute('aria-checked', 'true')
  await expect(placement).toHaveValue('subplot')
  await expect(preview.getByRole('status')).toHaveText(normalizedSummary!)
  expect({ series: seriesRequests, normalization: normalizationBases.length, previews: previewRequests }).toEqual(requestsBeforeFullscreen)
  await userSelect('right')
  await expect.poll(async () => (await preview.locator('canvas').boundingBox())!.height).toBeGreaterThanOrEqual(320)
  await placement.selectOption({ label: '同图同轴' })
  await preview.screenshot({ path: testInfo.outputPath('upstream-overlay.png') })
  await toggle.click()
  await expect(toggle).toHaveAttribute('aria-checked', 'false')
  await preview.getByText(/查看对比数据/).click()
  await expect(preview.getByRole('table', { name: '节点对比数据' })).toContainText('4,200')
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1)).toBeTruthy()
  expect(previewRequests).toBe(1)
  expect(errors).toEqual([])
  async function userSelect(value: string) { await placement.selectOption(value); await expect(placement).toHaveValue(value) }
})


test('递归平滑可搜索并叠加平行EMA分支，保留三种轴布局和全屏', async ({ page }, testInfo) => {
  const errors: string[] = []
  page.on('pageerror', error => errors.push(error.message))
  const reference = { node_id: 'market', port: 'value' }
  const definition = { schema_version: '2.0', name: '跨分支滤波对比', graph: { nodes: [
    { id: 'market', type: 'source.index', label: '沪深300', parameters: { ts_code: '000300.SH' }, inputs: {} },
    { id: 'recursive', type: 'indicator.recursive_smooth', label: '递归平滑', parameters: { periods: 3, initial: 100 }, inputs: { values: reference } },
    { id: 'ema', type: 'filter.ema', label: '单边 EMA', parameters: { window: 3 }, inputs: { value: reference } },
    { id: 'future', type: 'pivot.local_extrema', label: '事后峰谷', parameters: {}, inputs: { value: reference } },
  ], outputs: {} }, states: [], evaluation_targets: [], validation: {} }
  const dates = ['2020-01-01', '2020-01-02', '2020-01-03'], values: Record<string, number[]> = { market: [100, 120, 110], recursive: [100, 106, 108], ema: [100, 110, 110] }
  const audit = { execution_backend: 'numba_njit_fixed_signature', nopython: true, object_mode: 0, python_fallback: 0, request_time_compilation: 0, kernel_signatures: { graph: ['fixed'] } }
  let runCount = 0
  const comparisons = [{ node_id: 'ema', port: 'value' }]
  const curveRequests: string[] = []
  await page.route('**/api/**', async route => {
    const request = route.request(), url = new URL(request.url()), path = url.pathname
    const body = request.method() === 'POST' ? request.postDataJSON() : undefined
    let value: unknown = { items: [] }
    if (path.endsWith('/nodes')) value = { items: definition.graph.nodes.map(node => ({
      id: node.type, label: node.label, category: node.id === 'market' ? 'source' : 'filter',
      causal: node.id !== 'future', repaints: node.id === 'future', supports_realtime: node.id !== 'future',
      inputs: Object.keys(node.inputs).map(id => ({ id, value_type: 'series<float64>' })),
      outputs: [{ id: 'value', value_type: 'series<float64>' }], parameter_schema: { properties: {} },
    })) }
    else if (path.endsWith('/templates/v2')) value = { items: [{ id: 'parallel', name: definition.name }] }
    else if (path.endsWith('/templates/parallel/instantiate')) value = { definition }
    else if (path.endsWith('/infer')) value = { valid: false, errors: [], warnings: [] }
    else if (path.endsWith('/authoring/resolve')) value = { valid: false, definition: body.definition, source: '', diagnostics: [], compile_status: 'not_requested' }
    else if (path.endsWith('/prepare') || path.endsWith('/preview-runs')) {
      expect(body.preview_target).toEqual({ node_id: 'recursive', port: 'value' })
      expect(body.comparison_targets).toEqual(comparisons)
      expect(body.mode || 'realtime').toBe('realtime')
      expect(body.definition.graph.outputs).toEqual({})
      if (path.endsWith('/prepare')) value = { compile_token: 'parallel-token', runtime_audit: audit }
      else { runCount++; value = { id: 'parallel-run', status: 'queued' } }
    } else if (path.endsWith('/preview-runs/parallel-run')) value = { id: 'parallel-run', status: 'completed', execution: audit }
    else if (path.endsWith('/series')) {
      const node = url.searchParams.get('node_id')!
      curveRequests.push(node)
      value = { id: 'parallel-run', node_id: node, node_label: node === 'recursive' ? '递归平滑' : '单边 EMA', port: 'value', value_type: 'series<float64>', offset: 0, limit: 5000, total: 3,
        items: dates.map((date, index) => ({ observation_date: date, value: values[node][index] })), comparison_targets: comparisons,
        overlay_outputs: [{ ...reference, node_label: '沪深300', port_label: '数值序列', value_type: 'series<float64>', distance: 1, relationship: 'upstream', plottable: true },
          { node_id: 'ema', port: 'value', node_label: '单边 EMA', port_label: '数值序列', value_type: 'series<float64>', distance: null, relationship: 'other', plottable: true }],
      }
    } else if (path.endsWith('/normalized-chart')) {
      const node = url.searchParams.get('node_id')!, base = Number(url.searchParams.get('base_index'))
      value = { run_id: 'parallel-run', node_id: node, port: 'value', base_index: base, base_date: dates[base], base_value: values[node][base], values: values[node].map(v => v / values[node][base]), change_pct: values[node].map(v => (v / values[node][base] - 1) * 100), execution: audit }
    }
    await route.fulfill({ json: value })
  })
  await page.goto('/settings/scenario-algorithms/workbench?template=parallel')
  await expect(page.getByLabel('研究名称')).toHaveValue(definition.name)
  await page.getByRole('tab', { name: '画布构建', exact: true }).click()
  const surface = page.getByTestId(testInfo.project.name === 'mobile-320' ? 'regime-graph-mobile-list' : 'regime-graph-desktop-flow')
  await surface.getByText('递归平滑', { exact: true }).click()
  await page.getByRole('dialog', { name: '节点参数', exact: true }).getByRole('button', { name: '预览此节点', exact: true }).click()
  const preview = page.getByRole('dialog', { name: '节点预览', exact: true })
  await preview.getByLabel('节点预览分析方式').selectOption('realtime')
  await preview.getByText('叠加对比节点（可选）').click()
  await expect(preview.getByRole('checkbox', { name: /事后峰谷/ })).toBeDisabled()
  await preview.getByLabel('搜索对比节点').fill('EMA')
  await expect(preview.getByRole('checkbox')).toHaveCount(1)
  await preview.getByRole('checkbox', { name: /单边 EMA/ }).check()
  await preview.getByText('叠加对比节点（已选 1）').click()
  await preview.getByRole('button', { name: '预览节点数据' }).click()
  await expect(preview.getByText('已返回 3 / 3 条节点结果')).toBeVisible()
  const placement = preview.getByRole('combobox', { name: /单边 EMA.*展示位置/ })
  await expect(placement).toHaveValue('subplot')
  await expect.poll(() => [...new Set(curveRequests)]).toEqual(['recursive', 'ema'])
  await expect(preview.getByText('正在读取对比数据…')).toHaveCount(0)
  await placement.selectOption('left')
  await expect(placement).toHaveValue('left')
  await preview.getByRole('switch').click()
  await expect(preview.getByText(/单边 EMA.*基准日 2020-01-01/)).toBeVisible()
  await preview.getByRole('button', { name: '全屏展开' }).click()
  const fullscreen = page.getByRole('dialog', { name: '节点图表全屏' })
  await expect(fullscreen).toBeVisible()
  await expect(fullscreen.locator('canvas')).toBeVisible()
  await fullscreen.screenshot({ path: testInfo.outputPath('parallel-filter-comparison.png') })
  await page.keyboard.press('Escape')
  await expect(placement).toHaveValue('left')
  await placement.selectOption('right')
  await expect(placement).toHaveValue('right')
  await placement.selectOption('subplot')
  await preview.getByText(/查看对比数据/).click()
  await expect(preview.getByRole('table', { name: '节点对比数据' })).toContainText('单边 EMA')
  await expect(preview.getByRole('table', { name: '节点对比数据' })).toContainText('110')
  await preview.getByRole('button', { name: '添加其他节点对比' }).click()
  await expect(preview.getByLabel('搜索对比节点')).toBeFocused()
  expect(runCount).toBe(1)
  expect(errors).toEqual([])
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1)).toBe(true)
})

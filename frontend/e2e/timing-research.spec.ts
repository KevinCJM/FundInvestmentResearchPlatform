import { test, expect } from '@playwright/test'
import { timingCatalogFixture, timingExecutionFixture, timingRunFixture, timingTrainingCatalogFixture } from '../src/components/timing-research/timingFixtures'

test('ETF 改编训练可编辑，篮子显式指定且结果解释冻结选择', async ({ page }, testInfo) => {
  const errors: string[] = [], writes: Array<{ path: string; body: any }> = []
  const result = structuredClone(timingRunFixture)
  result.products[0].training = { mode: 'month', freeze_date: '2024-01-01', fit_end_date: '2023-12-29', embargo_bars: 0, min_trades: 5, selection: [{ state: '1', action_id: 'trend', action_label: '趋势候选', sample_count: 14, utility: .01 }], candidates: [{ id: 'trend', label: '趋势候选', states: [{ state: '1', sample_count: 14, mean_return: .02, win_rate: .6, utility: .01, stop_rate: .05 }] }], warnings: ['离线交互夹具，不是实际研究成绩。'] }
  result.products[0].training.candidates[0].parameters = [{ node: 'signal', parameter: 'threshold', value: 1.04 }]
  page.on('pageerror', error => errors.push(error.message))
  await page.route('**/api/**', route => {
    const path = new URL(route.request().url()).pathname
    const body = route.request().method() === 'POST' ? route.request().postDataJSON() : undefined
    if (body) writes.push({ path, body })
    const value = path.endsWith('/catalog') ? timingTrainingCatalogFixture
      : path.endsWith('/definitions') ? { items: [] }
      : path.endsWith('/prepare') ? { compile_token: 'test', execution: timingExecutionFixture }
      : path.endsWith('/runs') ? (body ? { id: 'job', status: 'completed', run_id: result.id } : { items: [] })
      : path.endsWith(`/runs/${result.id}`) ? result : null
    return route.fulfill({ status: value ? 200 : 404, json: value || {} })
  })
  await page.goto('/product-research/timing')
  await expect(page.getByLabel('算法名称')).toHaveValue('ETF 弱月选择改编')
  await expect(page.getByRole('button', { name: '运行研究', exact: true })).toBeDisabled()
  await page.getByLabel('市场参考篮子代码').fill('510300.SH, 510500.SH')
  await page.getByText('ETF 改编说明 · A2074', { exact: true }).click()
  await expect(page.getByText(/不继承原实验评级或收益/)).toBeVisible()
  await page.getByText('编辑训练规则与候选动作', { exact: true }).click()
  await expect(page.getByText(/不是每月 \/ 每季重新训练/)).toBeVisible()
  await page.getByText('参数搜索空间 · 1 个维度', { exact: true }).click()
  await page.getByLabel('搜索 1 方案 2 参数 1 值', { exact: true }).fill('1.04')
  await expect(page.getByLabel('动作 1 买入条件', { exact: true })).toHaveValue('signal.condition')
  expect(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1)).toBe(false)
  await page.evaluate(() => window.scrollTo(0, 0))
  await page.screenshot({ path: testInfo.outputPath('timing-training-editor.png'), fullPage: true })
  await page.getByRole('button', { name: '运行研究', exact: true }).click()
  await expect(page.getByRole('region', { name: '择时研究结果' })).toBeVisible()
  const request = writes.find(item => item.path.endsWith('/runs'))!.body
  expect(request.context_baskets).toEqual({ market: ['510300.SH', '510500.SH'], category: [] })
  expect(request.definition.training.search_space[0].choices[1][0].value).toBe(1.04)
  await page.getByText('训练选择与冻结审计', { exact: true }).click()
  await expect(page.getByText(/冻结日 2024-01-01，训练截止 2023-12-29/)).toBeVisible()
  await page.getByText('已选参数', { exact: true }).click()
  await expect(page.locator('dd').filter({ hasText: /^1\.04$/ })).toBeVisible()
  await page.getByText('查看候选方案训练对比', { exact: true }).click()
  await expect(page.getByRole('table', { name: '候选训练统计' })).toContainText('趋势候选')
  expect(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1)).toBe(false)
  await page.evaluate(() => window.scrollTo(0, 0))
  await page.screenshot({ path: testInfo.outputPath('timing-training-audit.png'), fullPage: true })
  expect(errors).toEqual([])
})

test('密集买卖点使用 B/S，缩放保留交易且点击定位日期', async ({ page }, testInfo) => {
  const result = structuredClone(timingRunFixture)
  const curve = Array.from({ length: 360 }, (_, index) => ({
    ...result.products[0].curve![0], date: new Date(Date.UTC(2024, 0, index + 1)).toISOString().slice(0, 10),
    close: 5 + Math.sin(index / 25) * .5 + Math.sin(index / 5) * .06,
  }))
  const trades = Array.from({ length: 90 }, (_, index) => ({
    ...result.products[0].trades![0],
    entry_date: curve[index * 4].date, exit_date: curve[index * 4 + 2].date,
    entry_price: curve[index * 4].close, exit_price: curve[index * 4 + 2].close,
  }))
  result.products[0] = { ...result.products[0], curve, trades }
  const errors: string[] = []
  page.on('pageerror', error => errors.push(error.message))
  await page.route('**/api/**', route => {
    const path = new URL(route.request().url()).pathname
    const value = path.endsWith('/catalog') ? timingCatalogFixture
      : path.endsWith('/definitions') ? { items: [] }
      : path.endsWith('/prepare') ? { compile_token: 'test', execution: timingExecutionFixture }
      : path.endsWith('/runs') ? (route.request().method() === 'POST' ? { id: 'job', status: 'completed', run_id: result.id } : { items: [] })
      : path.endsWith(`/runs/${result.id}`) ? result : null
    return route.fulfill({ status: value ? 200 : 404, json: value || {} })
  })
  await page.goto('/product-research/timing?product_id=510300.SH&kind=etf')
  await expect(page.getByLabel('算法名称')).toHaveValue('均线趋势研究')
  await page.getByRole('button', { name: '运行研究', exact: true }).click()
  const chart = page.getByRole('group', { name: '买卖点图表' })
  await expect(chart.getByText('B', { exact: true })).toBeVisible()
  await expect(chart.locator('canvas')).toHaveCount(1)
  await chart.scrollIntoViewIfNeeded()
  await chart.screenshot({ path: testInfo.outputPath('timing-bs-full.png') })
  const zoomed = await chart.locator('.echarts-for-react').evaluate(async element => {
    // Reuse the wrapper's exact ECharts module; a second import of the raw
    // package would own a different instance registry in Vite's dev server.
    const modulePath = performance.getEntriesByType('resource').map(item => item.name).find(name => name.includes('/echarts-for-react.js?v='))
    if (!modulePath) throw new Error('Rendered chart module was not loaded')
    const { default: Chart } = await import(modulePath)
    const echarts = new Chart({ option: {} }).echarts
    const instance = echarts.getInstanceByDom(element)
    instance.dispatchAction({ type: 'dataZoom', start: 75, end: 100 })
    const option = instance.getOption()
    return { labels: option.series.slice(1).map((series: any) => series.label.formatter), counts: option.series.slice(1).map((series: any) => series.data.length), start: option.dataZoom[0].start }
  })
  expect(zoomed).toEqual({ labels: ['B', 'S'], counts: [90, 90], start: 75 })
  const target = trades[85]
  const point = await chart.locator('.echarts-for-react').evaluate(async (element, value) => {
    const modulePath = performance.getEntriesByType('resource').map(item => item.name).find(name => name.includes('/echarts-for-react.js?v='))
    if (!modulePath) throw new Error('Rendered chart module was not loaded')
    const { default: Chart } = await import(modulePath)
    const echarts = new Chart({ option: {} }).echarts
    const pixel = echarts.getInstanceByDom(element).convertToPixel({ seriesIndex: 1 }, [value.entry_date, value.entry_price])
    return { x: pixel[0], y: pixel[1] }
  }, target)
  await chart.locator('canvas').click({ position: point })
  await expect(page.getByLabel('查看某日信号')).toHaveValue(target.entry_date)
  await chart.screenshot({ path: testInfo.outputPath('timing-bs-zoomed.png') })
  expect(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1)).toBe(false)
  expect(errors).toEqual([])
})

test('择时算法可编辑、检验并解释结果，窄屏保持可用', async ({ page }, testInfo) => {
  const errors: string[] = []
  const bodies: Array<{ path: string; body: any }> = []
  page.on('pageerror', error => errors.push(error.message))
  await page.route('**/api/**', async route => {
    const path = new URL(route.request().url()).pathname
    const body = ['POST', 'PUT'].includes(route.request().method()) ? route.request().postDataJSON() : undefined
    if (body) bodies.push({ path, body })
    let value: unknown
    if (path.endsWith('/timing-research/catalog')) value = timingCatalogFixture
    else if (path.endsWith('/timing-research/definitions')) value = body ? { ...body, id: 'saved-definition', revision: 1 } : { items: [] }
    else if (path.endsWith('/timing-research/prepare')) value = { compile_token: 'test-compile-token', definition_hash: 'test-hash', execution: timingExecutionFixture }
    else if (path.endsWith('/timing-research/runs') && body) value = { id: 'job-test', status: 'completed', progress: 1, total: 1, run_id: timingRunFixture.id }
    else if (path.endsWith('/timing-research/runs')) value = { items: [] }
    else if (path.endsWith(`/timing-research/runs/${timingRunFixture.id}`)) value = timingRunFixture
    else return route.fulfill({ status: 404, json: { detail: `Offline fixture: ${path}` } })
    return route.fulfill({ json: value })
  })
  await page.goto('/product-research/timing?product_id=510300.SH&kind=etf')
  await expect(page.getByLabel('算法名称')).toHaveValue('均线趋势研究')
  await expect(page.getByRole('heading', { name: '产品择时研究', exact: true })).toBeVisible()
  await page.getByRole('button', { name: /买入信号/ }).click()
  await page.getByLabel('阈值', { exact: true }).fill('1.02')
  await expect(page.getByLabel('买入信号 比较序列')).toHaveValue('price.value')
  await page.getByRole('button', { name: '连线图', exact: true }).click()
  await expect(page.locator('.react-flow__node')).toHaveCount(2)
  await expect(page.locator('.react-flow__edge')).toHaveCount(1)
  await page.getByRole('button', { name: '步骤编辑', exact: true }).click()
  await page.getByRole('button', { name: '运行研究', exact: true }).click()
  const results = page.getByRole('region', { name: '择时研究结果' })
  await expect(results).toBeVisible()
  expect(bodies.find(item => item.path.endsWith('/runs'))?.body.definition.nodes[1].parameters.threshold).toBe(1.02)
  await expect(results.locator('canvas')).toHaveCount(2)
  await results.getByText('查看图表数据表', { exact: true }).click()
  await expect(results.getByRole('table', { name: '净值与信号数据' })).toContainText('2024-01-03')
  await results.getByRole('tab', { name: '逐笔交易', exact: true }).click()
  await expect(results.getByRole('table', { name: '逐笔交易' })).toContainText('退出条件触发')
  await results.getByRole('tab', { name: '步骤预览', exact: true }).click()
  await results.getByLabel('选择计算步骤').selectOption('signal.condition')
  await expect(results.getByText(/−1 表示数据不足/)).toBeVisible()
  await page.getByLabel('算法名称').fill('修改后的假设')
  await expect(page.getByText(/配置已修改/)).toBeVisible()
  await expect(page.getByRole('button', { name: '保存研究版本', exact: true })).toBeDisabled()
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth + 1)
  expect(overflow).toBe(false)
  await page.evaluate(() => window.scrollTo(0, 0))
  await page.screenshot({ path: testInfo.outputPath('timing-research.png'), fullPage: true })
  expect(errors).toEqual([])
})

test('投前优先选择已有研究版本，直接引用不重复发布', async ({ page }, testInfo) => {
  const writes: Array<{ path: string; body: any }> = []
  await page.route('**/api/**', async route => {
    const path = new URL(route.request().url()).pathname
    const body = ['POST', 'PUT'].includes(route.request().method()) ? route.request().postDataJSON() : undefined
    if (body) writes.push({ path, body })
    let value: unknown
    if (path.endsWith('/timing-research/catalog')) value = timingCatalogFixture
    else if (path.endsWith('/timing-research/definitions') || path.endsWith('/timing-research/runs')) value = { items: [] }
    else if (path.endsWith('/timing-research/releases')) value = { items: [
      { id: 'release-trend', run_id: timingRunFixture.id, name: '均线研究版本', created_at: '2026-09-10T09:00:00', usage: 'research_only', products: [{ product_id: '510300.SH' }], note: '用于宽基 ETF 研究对照' },
      { id: 'release-repair', run_id: timingRunFixture.id, name: '修复研究版本', created_at: '2026-09-09T09:00:00', usage: 'research_only', products: [{ product_id: '510300.SH' }, { product_id: '510500.SH' }] },
    ] }
    else if (path.endsWith('/timing-research/bindings')) value = { id: 'binding-existing', ...body }
    else return route.fulfill({ status: 404, json: { detail: `Offline fixture: ${path}` } })
    return route.fulfill({ json: value })
  })
  await page.goto('/pre-investment/product-allocation-timing/timing')
  await expect(page.getByRole('tab', { name: '已有研究版本', exact: true })).toHaveAttribute('aria-selected', 'true')
  await expect(page.getByLabel('算法名称')).toHaveCount(0)
  await page.getByRole('radio', { name: /修复研究版本/ }).check()
  await page.getByLabel('本次引用说明').fill('投前固定研究对照')
  await page.getByRole('button', { name: '引用此版本到投前' }).click()
  await expect(page.getByRole('status')).toContainText('已引用“修复研究版本”')
  expect(writes).toEqual([{ path: '/api/timing-research/bindings', body: { release_id: 'release-repair', context: 'pre_investment', note: '投前固定研究对照' } }])
  expect(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1)).toBe(false)
  await page.evaluate(() => window.scrollTo(0, 0))
  await page.screenshot({ path: testInfo.outputPath('timing-application.png'), fullPage: true })
})

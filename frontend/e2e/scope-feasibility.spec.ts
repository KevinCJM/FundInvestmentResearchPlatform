import { expect, test, type Locator, type Page, type TestInfo } from '@playwright/test'
import { auditTextContrast } from './helpers/contrast'

const server = process.env.LTCMA_TEST_API || 'http://127.0.0.1:8128'
const api = `${server}/api/strategic-allocation`
async function chartState(chart: Locator) {
  return chart.locator('.echarts-for-react').evaluate(async element => {
    // Use the wrapper's module registry, as in timing-research.spec.ts.
    const modulePath = performance.getEntriesByType('resource').map(item => item.name).find(name => name.includes('/echarts-for-react.js?v='))
    if (!modulePath) throw new Error('Rendered chart module was not loaded')
    const { default: Chart } = await import(modulePath)
    const instance = new Chart({ option: {} }).echarts.getInstanceByDom(element)
    const option = instance.getOption(), visible: string[] = []
    instance.getModel().eachSeries((series: any) => {
      if (series.option.data.length || series.option.markLine?.data?.length || series.option.markArea?.data?.length) visible.push(series.name)
    })
    const coordinate = instance.getModel().getComponent('grid').coordinateSystem
    const rect = coordinate.getRect()
    return { visible, targetLines: option.series[1]?.markLine?.data, targetArea: option.series[1]?.markArea?.data, x: option.xAxis, y: option.yAxis, data: option.series.map((series: any) => series.data),
      zoom: option.dataZoom, extent: coordinate.getCartesians()[0].getAxis('x').scale.getExtent().concat(coordinate.getCartesians()[0].getAxis('y').scale.getExtent()),
      rect: { x: rect.x, y: rect.y, width: rect.width, height: rect.height } }
  })
}

async function checkZoom(page: Page, panel: Locator, chart: Locator, info: TestInfo) {
  const original = await chartState(chart)
  const select = panel.getByRole('button', { name: '框选放大', exact: true })
  await select.focus(); await page.keyboard.press('Enter')
  await expect(select).toHaveAttribute('aria-pressed', 'true')
  await page.keyboard.press('Escape')
  await expect(select).toHaveAttribute('aria-pressed', 'false')
  let previous = original.extent
  for (let attempt = 0; attempt < 2; attempt++) {
    await select.click()
    await chart.scrollIntoViewIfNeeded()
    const box = (await chart.boundingBox())!, { rect } = await chartState(chart)
    await page.mouse.move(box.x + rect.x + rect.width * .2, box.y + rect.y + rect.height * .2)
    await page.mouse.down()
    await page.mouse.move(box.x + rect.x + rect.width * .8, box.y + rect.y + rect.height * .8, { steps: 12 })
    await page.mouse.up()
    await expect(select).toHaveAttribute('aria-pressed', 'false')
    await expect.poll(async () => (await chartState(chart)).extent[1] - (await chartState(chart)).extent[0]).toBeLessThan(previous[1] - previous[0])
    const current = await chartState(chart)
    expect(current.extent[3] - current.extent[2]).toBeLessThan(previous[3] - previous[2])
    expect(current.data).toEqual(original.data)
    previous = current.extent
  }
  const goal = panel.getByRole('checkbox', { name: '目标与约束', exact: true })
  await goal.uncheck()
  expect((await chartState(chart)).extent).toEqual(previous)
  await chart.screenshot({ path: info.outputPath('frontier-zoomed.png') })
  const reset = panel.getByRole('button', { name: '恢复全图', exact: true })
  await reset.focus(); await page.keyboard.press('Enter')
  await expect.poll(async () => (await chartState(chart)).extent).toEqual(original.extent)
  await expect(goal).not.toBeChecked()
  expect((await chartState(chart)).visible).not.toContain('目标与约束')
  await goal.check()
  await chart.screenshot({ path: info.outputPath('frontier-zoom-reset.png') })
}

async function checkLegend(page: Page, panel: Locator, testId: string, info: TestInfo) {
  const chart = panel.getByTestId(testId), legend = panel.getByRole('group', { name: '图例 · 显示内容' })
  await expect(chart.locator('canvas')).toHaveCount(1)
  const before = await chartState(chart), requests: string[] = []
  const track = (request: import('@playwright/test').Request) => { if (request.url().includes('/api/')) requests.push(request.url()) }
  page.on('request', track)
  await legend.getByRole('button', { name: '全部隐藏', exact: true }).focus()
  await page.keyboard.press('Tab')
  const boxes = legend.getByRole('checkbox')
  await expect(boxes.first()).toBeFocused()
  for (let index = 0; index < await boxes.count(); index++) {
    const box = boxes.nth(index), name = await box.getAttribute('aria-label')
    await expect(box).toBeChecked()
    if (index === 0) await page.keyboard.press('Space')
    else await box.locator('..').click()
    await expect(box).not.toBeChecked()
    await expect.poll(async () => (await chartState(chart)).visible).toEqual(before.visible.filter(value => value !== name))
    const hidden = await chartState(chart)
    expect(hidden.x).toEqual(before.x); expect(hidden.y).toEqual(before.y); expect(hidden.data).toEqual(before.data)
    if (name === '目标与约束') await panel.screenshot({ path: info.outputPath(`${testId}-goal-hidden.png`) })
    await box.check()
  }
  await legend.getByRole('button', { name: '全部隐藏', exact: true }).click()
  await expect.poll(async () => (await chartState(chart)).visible).toEqual([])
  await expect(legend.getByRole('status')).toContainText('图中内容已全部隐藏')
  expect(await page.evaluate(auditTextContrast)).toEqual([])
  await legend.getByRole('button', { name: '全部显示', exact: true }).click()
  await expect.poll(async () => (await chartState(chart)).visible).toEqual(before.visible)
  await expect(legend.getByRole('status')).toHaveCount(0)
  await checkZoom(page, panel, chart, info)
  expect(requests).toEqual([])
  page.off('request', track)
  await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth - innerWidth)).toBeLessThanOrEqual(1)
  await expect.poll(() => page.evaluate(auditTextContrast)).toEqual([])
  await panel.screenshot({ path: info.outputPath(`${testId}-interactive-legend.png`) })
}
test.beforeEach(async ({ page }) => {
  await page.route('**/api/**', async route => {
    const url = new URL(route.request().url())
    const response = await route.fetch({ url: `${server}${url.pathname}${url.search}`, timeout: 60000 })
    await route.fulfill({ response })
  })
})
test.afterEach(async ({ page }) => { await page.unrouteAll({ behavior: 'wait' }) })
test('shows the saved quarterly funding hurdle in the collapsed objective summary', async ({ page, request }, info) => {
  const fixture = await (await request.get(`${server}/fixture/ltcma`)).json()
  const scope = await (await request.get(`${api}/universes/${fixture.scope_feasibility_universe_id}`)).json()
  const day = scope.definition.as_of
  const study = { definition: {
    schema_version: '2.0', name: `季度资金目标 ${info.project.name}`, as_of: day, horizon_years: 5,
    currency: 'CNY', objective_kind: 'funding_goal', max_volatility: .06, min_cash_weight: .1,
    cash_budget: { total_capital: 100000, outside_reserve: 0, balance_as_of: day,
      source: 'offline_browser_fixture', amount_basis: 'nominal', inflation: 0, annual_fee: 0,
      flows: [{ name: '季度支出', kind: 'withdrawal', amount: 500, first_month: 1, last_month: 58, every_months: 3 }] },
    funding_target: { amount: 100000, amount_basis: 'nominal' },
    boundary_policy: { name: '离线测试', source: 'offline_browser_fixture', reviewed_on: day, confirmed: true,
      required_probability: .8, liquidity_months: 12, contribution_stress_ratio: .5, cash_reserve_weight: 0 },
    risk_authorization: { mode: 'explicit_numeric', source: 'offline_browser_fixture' },
  } }
  const previewResponse = await request.post(`${api}/mandates/preview`, { data: study })
  expect(previewResponse.ok()).toBeTruthy()
  const preview = await previewResponse.json()
  const savedResponse = await request.post(`${api}/mandates/confirm`, { data: {
    request: study, preview_hash: preview.preview_hash, acknowledge_limits: true,
  } })
  expect(savedResponse.ok()).toBeTruthy()
  const saved = await savedResponse.json()
  expect(saved.assessment.funding.cashflow_required_return).toBeCloseTo(.020218343484683543, 10)
  await page.goto('/pre-investment/objectives')
  const row = page.getByRole('row').filter({ has: page.getByRole('link', { name: study.definition.name, exact: true }) })
  await expect(row.getByText('2.02%', { exact: true })).toBeVisible()
  await row.getByRole('link', { name: study.definition.name, exact: true }).click()
  const impact = page.getByRole('region', { name: '本目标将约束后续研究' })
  await expect(impact.getByText('计划所需年化收益率', { exact: true })).toBeVisible()
  await expect(impact.getByText('2.02%', { exact: true })).toBeVisible()
  await expect.poll(() => page.evaluate(auditTextContrast)).toEqual([])
  await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth - innerWidth)).toBeLessThanOrEqual(1)
  await impact.screenshot({ path: info.outputPath('objective-return.png') })
  const definition = { ...scope.definition, name: `资金初筛范围 ${info.project.name}`, assets: [
    ...scope.definition.assets, { id: 'cash', name: '现金', currency: 'CNY', role: 'liquidity', liquidity: 'liquid',
      research_proxy: { asset_type: 'cash', cash_return: .01, components: [], rebalance: null, source_labels: {} } },
  ] }
  const scopePreview = await (await request.post(`${api}/universes/preview`, { data: definition })).json()
  const scopeResponse = await request.post(`${api}/universes/confirm`, { data: { request: definition, preview_hash: scopePreview.preview_hash, mandate_id: saved.id } })
  expect(scopeResponse.ok(), await scopeResponse.text()).toBeTruthy()
  const fundingScope = await scopeResponse.json()
  await page.goto(`/pre-investment/product-pool/new?scope=strategic&strategic_universe=${fundingScope.id}`)
  const summary = page.getByRole('region', { name: '当前投资目标与约束', exact: true })
  await expect(summary.getByText('2.02%', { exact: true })).toBeVisible()
  await expect(summary.getByText('计划所需年化收益率', { exact: true })).toBeVisible()
  await expect(summary.getByText(/不是市场收益预测/)).toBeVisible()
  await expect(summary.locator('details')).not.toHaveAttribute('open', '')
  const toggle = summary.locator('summary')
  await toggle.focus(); await page.keyboard.press('Enter')
  await expect(summary.getByText(/季度支出：支出 500 CNY/)).toBeVisible()
  await expect.poll(() => page.evaluate(auditTextContrast)).toEqual([])
  await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth - innerWidth)).toBeLessThanOrEqual(1)
  await summary.screenshot({ path: info.outputPath('funding-summary-expanded.png') })
  await toggle.focus(); await page.keyboard.press('Enter')
  await expect(summary.getByText('2.02%', { exact: true })).toBeVisible()
  await summary.screenshot({ path: info.outputPath('funding-summary-collapsed.png') })
  const panel = page.getByRole('region', { name: '所选范围能否达到目标？' })
  const screeningResponse = page.waitForResponse(response => response.url().includes('/scope-feasibility') && response.request().postDataJSON()?.window?.kind === 'common_since_inception')
  await panel.getByLabel('历史测算窗口').selectOption('common_since_inception')
  const screening = await (await screeningResponse).json()
  expect(screening.funding_comparison.target_return).toBeCloseTo(.020218343484683543, 10)
  expect(screening.funding_comparison.status).toBe('passed')
  await expect(panel.getByText('收益门槛与风险约束初筛通过', { exact: true })).toBeVisible()
  await expect(panel.getByText('资金路径成功率待验证', { exact: true })).toHaveCount(0)
  const chart = panel.getByTestId('scope-frontier-chart')
  await expect.poll(async () => (await chartState(chart)).targetLines?.length).toBe(2)
  const state = await chartState(chart)
  expect(state.targetLines[0].yAxis).toBeCloseTo(2.0218343484683543, 8)
  expect(state.targetLines[1].xAxis).toBe(6)
  expect(state.targetArea).toHaveLength(1)
  await expect(panel.getByText('模型年化复利收益（中位数）', { exact: true }).first()).toBeVisible()
  await expect.poll(() => page.evaluate(auditTextContrast)).toEqual([])
  await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth - innerWidth)).toBeLessThanOrEqual(1)
  await panel.screenshot({ path: info.outputPath('funding-scope-compound.png') })
})
test('distinguishes passed historical screening from funding validation and retains all three frontiers', async ({ page, request }, info) => {
  const fixture = await (await request.get(`${server}/fixture/ltcma`)).json()
  await page.route('**/api/strategic-allocation/scope-feasibility', async route => {
    const response = await route.fetch({ url: `${api}/scope-feasibility`, timeout: 60000 })
    const result = await response.json()
    // Synthetic response tests three-curve rendering; the preceding case exercises real funding calculations.
    if (result.target_check?.status === 'feasible') {
      result.status = 'undetermined'; result.reason_code = 'SCOPE_FUNDING_CHECK_REQUIRED'
      result.mandate.target_return = null
      result.mandate.funding_requirement = { required_return: .0772, status: 'solved', basis: 'annual_effective_gross_of_model_fee' }
      result.reasons = [{ code: result.reason_code, message: '历史收益与风险约束存在可行点；资金支付及期末目标成功率仍需后续验证。' }]
      result.additional_checks = { funding: true, benchmark: false }
      result.reference_comparison = { status: 'available', risk_scale_ref: { id: 'fixture-scale', content_hash: 'fixture-hash' },
        name: '原目标参考标尺', as_of: '2024-01-01', currency: 'CNY', sample_start: '2010-01-04', sample_end: '2023-12-29',
        points: [{ volatility: .01, expected_return: .02, status: 'optimal_to_tolerance' }, { volatility: .1, expected_return: .08, status: 'optimal_to_tolerance' }],
        constrained_points: [{ volatility: .01, expected_return: .015, status: 'optimal_to_tolerance' }, { volatility: .09, expected_return: .06, status: 'optimal_to_tolerance' }] }
      result.funding_comparison = { basis: 'annual_compound_median_gross_of_model_fee', status: 'passed',
        target_return: .0772, probability_validated: false, points: result.frontier.points,
        candidate: result.target_check.candidate, reference_points: result.reference_comparison.points,
        constrained_points: result.reference_comparison.constrained_points }
    }
    await route.fulfill({ response, json: result })
  })
  await page.goto(`/pre-investment/product-pool/new?scope=strategic&strategic_universe=${fixture.scope_feasibility_universe_id}`)
  const panel = page.getByRole('region', { name: '所选范围能否达到目标？' })
  await panel.getByLabel('历史测算窗口').selectOption('common_since_inception')
  await expect(panel.getByText('收益门槛与风险约束初筛通过', { exact: true })).toBeVisible()
  await expect(panel.getByText('资金路径成功率待验证', { exact: true })).toHaveCount(0)
  await expect(panel.getByText('暂无法判断', { exact: true })).toHaveCount(0)
  await expect(panel.getByText(/下一步在 LTCMA／SAA 中/)).toBeVisible()
  await expect(panel.getByText('资金计划所需年化收益率', { exact: true })).toBeVisible()
  await expect(panel.locator('dd').filter({ hasText: /^7\.72%$/ })).toBeVisible()
  await expect(panel.getByText(/沿用 01 资金计算口径/)).toBeVisible()
  const chart = panel.getByTestId('scope-frontier-chart')
  await expect.poll(async () => (await chartState(chart)).visible).toEqual(expect.arrayContaining([
    '当前范围的历史有效前沿', '风险标尺原始前沿', '原目标约束下的参考前沿',
  ]))
  const current = panel.getByRole('checkbox', { name: '当前范围的历史有效前沿', exact: true })
  await current.focus(); await page.keyboard.press('Space')
  await expect.poll(async () => (await chartState(chart)).visible).not.toContain('当前范围的历史有效前沿')
  await expect(panel.getByText('资金路径成功率待验证', { exact: true })).toHaveCount(0)
  await current.check()
  await expect.poll(() => page.evaluate(auditTextContrast)).toEqual([])
  await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth - innerWidth)).toBeLessThanOrEqual(1)
  await panel.screenshot({ path: info.outputPath('scope-funding-compound-comparison.png') })
})
test('compares the original scale and constrained reference without replacing current-scope results', async ({ page, request }, info) => {
  const fixture = await (await request.get(`${server}/fixture/ltcma`)).json()
  let referenceAvailable = true, currentAvailable = true
  await page.route('**/api/strategic-allocation/scope-feasibility', async route => {
    const response = await route.fetch({ url: `${api}/scope-feasibility`, timeout: 60000 })
    const result = await response.json()
    // Fixed display evidence; actual projection/hash checks are exercised by the backend regression.
    result.reference_comparison = referenceAvailable ? { status: 'available', risk_scale_ref: { id: 'fixture-scale', content_hash: 'fixture-hash' },
      name: '原目标参考标尺', as_of: '2024-01-01', currency: 'CNY', sample_start: '2010-01-04', sample_end: '2023-12-29',
      points: [{ volatility: 0, expected_return: .0005, status: 'optimal_to_tolerance' }, { volatility: .1, expected_return: .08, status: 'optimal_to_tolerance' }, { volatility: .25, expected_return: .13, status: 'optimal_to_tolerance' }],
      constrained_points: [{ volatility: 0, expected_return: .0005, status: 'optimal_to_tolerance' }, { volatility: .08, expected_return: .06, status: 'optimal_to_tolerance' }, { volatility: .2, expected_return: .1, status: 'optimal_to_tolerance' }] } : { status: 'unavailable' }
    if (!currentAvailable) {
      Object.assign(result, { status: 'undetermined', frontier: null, sample: null, target_check: null })
      result.mandate.target_return = null
    }
    await route.fulfill({ response, json: result })
  })
  await page.goto(`/pre-investment/product-pool/new?scope=strategic&strategic_universe=${fixture.scope_feasibility_universe_id}`)
  const panel = page.getByRole('region', { name: '所选范围能否达到目标？' })
  await panel.getByLabel('历史测算窗口').selectOption('common_since_inception')
  await expect(panel.getByText('历史测算可达', { exact: true })).toBeVisible()
  await expect(panel.getByText('风险标尺原始前沿', { exact: true })).toBeVisible()
  await expect(panel.getByText('原目标约束下的参考前沿', { exact: true })).toBeVisible()
  await expect(panel.getByText('原始参考样本：2010-01-04 至 2023-12-29。')).toBeVisible()
  await checkLegend(page, panel, 'scope-frontier-chart', info)
  await expect(panel.getByText('历史测算可达', { exact: true })).toBeVisible()
  await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth - innerWidth)).toBeLessThanOrEqual(1)
  await expect.poll(() => page.evaluate(auditTextContrast)).toEqual([])
  await panel.screenshot({ path: info.outputPath('scope-original-reference.png') })
  await panel.locator('summary').click()
  await expect(panel.getByRole('table')).toContainText('风险标尺原始前沿 3')
  await expect(panel.getByRole('table')).toContainText('25.00%')
  referenceAvailable = false
  await panel.getByLabel('历史测算窗口').selectOption('2Y')
  await expect(panel.getByText(/原始参考前沿暂不可读/)).toBeVisible()
  await expect(panel.getByText('风险标尺原始前沿', { exact: true })).toHaveCount(0)
  referenceAvailable = true; currentAvailable = false
  await panel.getByLabel('历史测算窗口').selectOption('1Y')
  await expect(panel.getByText(/当前范围尚无可用前沿/)).toBeVisible()
  await expect(panel.getByText('暂无法判断', { exact: true })).toBeVisible()
  await expect(panel.getByTestId('scope-frontier-chart')).toBeVisible()
  const goal = panel.getByRole('checkbox', { name: '目标与约束', exact: true })
  await expect(goal.locator('..')).toContainText('风险上限')
  await expect(goal.locator('..')).not.toContainText('收益下限')
  await goal.uncheck()
  await expect.poll(async () => (await chartState(panel.getByTestId('scope-frontier-chart'))).visible).toEqual(['风险标尺原始前沿', '原目标约束下的参考前沿'])
  await goal.check()
  await expect.poll(async () => (await chartState(panel.getByTestId('scope-frontier-chart'))).visible).toContain('目标与约束')
})

test('SAA frontier legends toggle the goal group and individual candidates without API requests', async ({ page, request }, info) => {
  const fixture = await (await request.get(`${server}/fixture/ltcma`)).json()
  const errors: string[] = []
  page.on('pageerror', error => errors.push(error.message))
  await page.goto(`/pre-investment/saa/policy?alloc=${encodeURIComponent(fixture.allocation)}&mandate=${fixture.mandate_id}&cma=${fixture.manual_id}`)
  const panel = page.getByRole('region', { name: '目标与有效前沿', exact: true })
  await expect(panel.getByTestId('saa-frontier-chart')).toBeVisible()
  const pending = page.waitForResponse(response => response.url().endsWith('/policy/preview') && response.request().method() === 'POST')
  await page.getByRole('button', { name: '比较符合目标的政策候选', exact: true }).click()
  const response = await pending
  expect(response.status(), await response.text()).toBe(200)
  const result = await response.json()
  expect(result.candidates.filter((candidate: any) => candidate.available !== false).length).toBeGreaterThan(1)
  await expect(panel.getByRole('checkbox', { name: /^候选 1/ })).toBeVisible()
  const conclusion = await panel.getByRole('status').innerText()
  await checkLegend(page, panel, 'saa-frontier-chart', info)
  await expect(panel.getByRole('status')).toHaveText(conclusion)
  expect(errors).toEqual([])
})

test('step 02 calculates the real historical frontier before an LTCMA is selected', async ({ page, request }, info) => {
  const fixture = await (await request.get(`${server}/fixture/ltcma`)).json()
  const errors: string[] = []
  page.on('pageerror', error => errors.push(error.message))
  await page.goto(`/pre-investment/product-pool/new?scope=strategic&strategic_universe=${fixture.scope_feasibility_universe_id}`)
  const panel = page.getByRole('region', { name: '所选范围能否达到目标？' })
  await expect(panel.getByText('暂无法判断', { exact: true })).toBeVisible()
  const pending = page.waitForResponse(r => r.url().endsWith('/scope-feasibility') && r.request().postDataJSON().window.kind === 'common_since_inception')
  await panel.getByLabel('历史测算窗口').selectOption('common_since_inception')
  const response = await pending
  expect(response.status()).toBe(200)
  const value = await response.json()
  expect(value.status, JSON.stringify(value.reasons)).toBe('feasible')
  expect(value.frontier.constraints_applied).toBe(true)
  expect(value.frontier.points.some((p: { status: string }) => p.status === 'optimal_to_tolerance')).toBe(true)
  expect(value.target_check.candidate.volatility).toBeLessThanOrEqual(value.mandate.volatility_cap + 1e-8)
  await expect(panel.getByText('历史测算可达', { exact: true })).toBeVisible()
  await expect(panel.getByTestId('scope-frontier-chart')).toBeVisible()
  await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth - innerWidth)).toBeLessThanOrEqual(1)
  await expect.poll(() => page.evaluate(auditTextContrast)).toEqual([])
  await panel.screenshot({ path: info.outputPath('scope-frontier-reachable.png') })
  await panel.locator('summary').click()
  await expect(panel.getByRole('table')).toBeVisible()
  expect(errors).toEqual([])
})

test('a target outside the continuous frontier is warned without preventing scope editing', async ({ page, request }, info) => {
  const fixture = await (await request.get(`${server}/fixture/ltcma`)).json()
  const source = await (await request.get(`${api}/mandates/${fixture.scope_feasibility_mandate_id}`)).json()
  const definition = { ...source.definition, name: `初筛高目标 ${info.project.name}`, target_return: .8 }
  const preview = await request.post(`${api}/mandates/preview`, { data: { definition } })
  expect(preview.status(), await preview.text()).toBe(200)
  const checked = await preview.json()
  const confirmed = await request.post(`${api}/mandates/confirm`, { data: { request: { definition }, preview_hash: checked.preview_hash, acknowledge_limits: true } })
  expect(confirmed.status(), await confirmed.text()).toBe(201)
  const goal = await confirmed.json()
  await page.goto(`/pre-investment/product-pool/new?scope=strategic&strategic_universe=${fixture.scope_feasibility_universe_id}&copy=1&mandate=${goal.id}`)
  const panel = page.getByRole('region', { name: '所选范围能否达到目标？' })
  await panel.getByLabel('历史测算窗口').selectOption('common_since_inception')
  await expect(panel.getByText('历史测算不可达', { exact: true })).toBeVisible()
  await expect(panel.getByText(/仍可保存当前范围/)).toBeVisible()
  await expect(page.getByLabel('战略范围名称')).toBeEnabled()
  await expect(panel.getByTestId('scope-frontier-chart')).toBeVisible()
  await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth - innerWidth)).toBeLessThanOrEqual(1)
  await expect.poll(() => page.evaluate(auditTextContrast)).toEqual([])
  await panel.screenshot({ path: info.outputPath('scope-frontier-unreachable.png') })
  // Invalidating a proxy must remove the previous answer and chart immediately.
  await page.getByTestId('risk-reference-asset').first().getByLabel('大类名称').fill('已修改的大类')
  await expect(panel.getByText('正在测算历史有效前沿与目标可达性…')).toBeVisible()
})

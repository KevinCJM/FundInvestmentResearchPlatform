import { test, expect } from '@playwright/test'
import { taaBaseline, taaCatalog, taaExecution, taaPreview, taaPreflight } from '../src/test/tacticalAllocationFixtures'

test('TAA真实界面在桌面和手机完成观点、候选、情景与保存交接', async ({ page }, testInfo) => {
  const errors: string[] = []; const writes: Array<{ path: string; body: any }> = []
  let preview = structuredClone(taaPreview)
  let saved: any = null
  page.on('pageerror', error => errors.push(error.message))
  await page.route('**/api/**', async route => {
    const path = new URL(route.request().url()).pathname
    const post = route.request().method() === 'POST'; const body = post ? route.request().postDataJSON() : undefined
    if (post) writes.push({ path, body })
    let value: unknown
    if (path === '/api/tactical-allocation/catalog') value = { ...taaCatalog, decisions: saved ? [saved] : [] }
    else if (path === '/api/tactical-allocation/baselines/SAA-1') value = taaBaseline
    else if (path === '/api/tactical-allocation/preflight') value = taaPreflight
    else if (path === '/api/tactical-allocation/preview') { preview = { ...taaPreview, request: body }; value = preview }
    else if (path === '/api/tactical-allocation/scenarios') value = { name: body.scenario.name, kind: body.scenario.kind, baseline_return: -.0601, taa_return: -.0651, excess_return: -.005, contributions: [{ asset_id: 'equity', baseline: -.06, taa: -.065, excess: -.005 }, { asset_id: 'bond', baseline: 0, taa: 0, excess: 0 }], cost: { baseline: .0001, taa: .0001 }, warnings: ['固定浏览器夹具：情景是研究假设，不含发生概率。'], execution: taaExecution }
    else if (path === '/api/tactical-allocation/decisions') { saved = { id: 'DECISION-1', name: body.name, created_at: '2026-09-11', preview, note: body.note, scenarios: body.scenarios.map((scenario: any) => ({ scenario, result: { name: scenario.name, kind: scenario.kind, baseline_return: -.06, taa_return: -.065, excess_return: -.005, contributions: [], cost: { baseline: 0, taa: 0 }, warnings: [], execution: taaExecution } })) }; value = saved }
    else if (path === '/api/tactical-allocation/decisions/DECISION-1') value = saved
    else if (path === '/api/tactical-allocation/decisions/DECISION-1/product-allocation') value = { name: saved.name, method: 'manual', universe_snapshot_id: 'UNIVERSE-1', allocation_source: { kind: 'taa', decision_id: saved.id, baseline_id: taaBaseline.id, class_weights: { equity: .65, bond: .35 } }, constituents: [{ kind: 'etf', product_id: '510300.SH', name: '沪深300ETF', asset_class_id: 'equity', asset_class_name: '权益', weight: 65 }, { kind: 'etf', product_id: '511010.SH', name: '国债ETF', asset_class_id: 'bond', asset_class_name: '债券', weight: 35 }] }
    else if (path === '/api/portfolios/targets') value = { items: [] }
    else return route.fulfill({ status: 404, json: { detail: 'Offline browser fixture: no real requests or writes.' } })
    await route.fulfill({ json: value })
  })
  await page.goto('/pre-investment/taa?baseline=SAA-1')
  await expect(page.getByRole('heading', { name: '本次准备怎么配？' })).toBeVisible()
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBeTruthy()
  await page.evaluate(() => window.scrollTo(0, 0))
  await page.screenshot({ path: testInfo.outputPath('taa-initial.png'), fullPage: true })
  await page.getByRole('radio', { name: /研究员观点/ }).check()
  const equity = page.getByRole('spinbutton', { name: /权益偏离/ }); const bond = page.getByRole('spinbutton', { name: /债券偏离/ })
  await equity.fill('5'); await expect(page.getByRole('button', { name: '计算并比较方案' })).toBeDisabled()
  await bond.fill(''); await bond.pressSequentially('-5'); await expect(bond).toHaveValue('-5')
  await page.getByRole('button', { name: '计算并比较方案' }).click()
  await expect(page.getByRole('region', { name: 'SAA 与战术方案对照' })).toBeVisible()
  expect(writes.find(item => item.path.endsWith('/preview'))?.body.manual_tilts).toEqual({ equity: .05, bond: -.05 })
  await expect(page.getByText('训练与验证分别以 1 为起点；两段独立展示，不连接成一条持续投资净值。')).toBeVisible()
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBeTruthy()
  await page.evaluate(() => window.scrollTo(0, 0))
  await page.screenshot({ path: testInfo.outputPath('taa-backtest.png'), fullPage: true })
  await page.getByRole('tab', { name: '情景模拟' }).click()
  await page.getByRole('spinbutton', { name: '权益假设涨跌（%）' }).fill('-10')
  await page.getByRole('button', { name: '计算情景影响', exact: true }).click()
  await expect(page.getByRole('region', { name: '情景模拟结果' })).toContainText('扣费净收益合计')
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBeTruthy()
  await page.evaluate(() => window.scrollTo(0, 0))
  await page.screenshot({ path: testInfo.outputPath('taa-scenario.png'), fullPage: true })
  await page.getByLabel('情景方式').selectOption('historical')
  await expect(page.getByLabel('历史情景结束')).toHaveValue('2026-09-10')
  await page.getByRole('button', { name: '计算情景影响', exact: true }).click()
  await expect(page.getByRole('region', { name: '情景实验对照' })).toContainText('2/12')
  await page.getByRole('tab', { name: '版本与审计' }).click()
  await expect(page.getByRole('button', { name: '带入产品配置' })).toBeDisabled()
  await page.getByRole('button', { name: '保存研究版本', exact: true }).click()
  await expect(page.getByRole('button', { name: '带入产品配置' })).toBeEnabled()
  await page.evaluate(() => window.scrollTo(0, 0))
  await page.screenshot({ path: testInfo.outputPath('taa-saved.png'), fullPage: true })
  expect(saved.scenarios).toHaveLength(2)
  await page.reload()
  await expect(page.getByRole('button', { name: '当前研究版本已保存' })).toBeVisible()
  await page.getByRole('tab', { name: '情景模拟' }).click()
  await expect(page.getByRole('region', { name: '情景实验对照' })).toContainText('2/12')
  await page.getByRole('tab', { name: '版本与审计' }).click()
  await page.getByRole('button', { name: '带入产品配置' }).click()
  await expect(page).toHaveURL(/product-allocation-timing\/construction/)
  expect(writes.filter(item => item.path.endsWith('/product-allocation'))).toHaveLength(1)
  expect(errors).toEqual([])
})

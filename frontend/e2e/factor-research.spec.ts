import { test, expect } from '@playwright/test'
import { fixtureCatalog, fixtureRun, fixtureStudy } from '../src/test/factorFixtures'
import { fixtureReturnCatalog } from '../src/test/factorReturnFixtures'

test('factor research saves, inspects, publishes and supplies workflow evidence', async ({ page }, testInfo) => {
  const errors: string[] = []
  page.on('pageerror', error => errors.push(error.message))
  let studies: unknown[] = []
  let runs: unknown[] = []
  let releases: any[] = []
  let pools: any[] = []
  let bindings: any[] = []
  let published = false
  await page.route('**/api/**', async route => {
    const path = new URL(route.request().url()).pathname
    const post = route.request().method() === 'POST'
    const body = post ? route.request().postDataJSON() : undefined
    let value: unknown
    if (path === '/api/factor-research/catalog') value = fixtureCatalog
    else if (path === '/api/factor-research/return-catalog') value = fixtureReturnCatalog
    else if (path.endsWith('/return-plans') || path.endsWith('/return-sources')) value = { items: [] }
    else if (path === '/api/factor-research/studies') {
      if (post) { studies = [{ ...fixtureStudy, ...body }]; value = studies[0] } else value = { items: studies }
    } else if (path === '/api/factor-research/studies/factor-study-test/runs') { runs = [fixtureRun]; value = fixtureRun }
    else if (path === '/api/factor-research/runs') value = { items: runs }
    else if (path === '/api/factor-research/runs/factor-run-test') value = fixtureRun
    else if (path === '/api/factor-research/release/factor-release-test') value = releases[0]
    else if (path === '/api/factor-research/releases') {
      if (post) { releases = [{ ...body, id: 'factor-release-test', state: 'active', as_of: fixtureRun.as_of, usage: 'research_only' }]; published = true; value = releases[0] } else value = { items: releases }
    } else if (path === '/api/product-pools') {
      if (post) { pools = [{ ...body, id: 'pool-test', revision: 1, state: 'draft', members: [] }]; value = pools[0] } else value = { items: pools }
    } else if (path === '/api/product-pools/pool-test/evaluation-plans') { value = { ...pools[0], revision: 2, members: [{ research_status: 'pending' }] } }
    else if (path === '/api/factor-research/bindings') {
      if (post) { bindings.push({ ...body, run_id: fixtureRun.id, created_at: '2026-09-06' }); value = bindings[bindings.length - 1] } else value = { items: bindings }
    } else if (path.endsWith('/datasets') || path.endsWith('/attributions')) value = { items: [] }
    else return route.fulfill({ status: 404, json: { detail: 'Offline UI fixture' } })
    return route.fulfill({ json: value })
  })
  await page.goto('/settings/factor-research')
  await expect(page.getByRole('heading', { name: '因子研究中心', exact: true })).toBeVisible()
  await expect(page.getByRole('combobox', { name: '因子模型', exact: true })).toHaveValue('characteristic_composite')
  await expect(page.getByLabel('因子输入数据集')).toHaveValue('active_adjusted_nav')
  await page.getByRole('button', { name: '保存并运行检验' }).click()
  await expect(page.getByLabel('因子检验结果')).toBeVisible()
  await expect(page.getByLabel('最新因子得分')).toContainText('测试ETF0')
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1)).toBeTruthy()
  await page.getByRole('heading', { name: /检验结果/ }).scrollIntoViewIfNeeded()
  await page.screenshot({ path: testInfo.outputPath('factor-results.png'), fullPage: false })
  await page.getByRole('button', { name: '发布研究版', exact: true }).click()
  await expect(page.getByRole('button', { name: '确认发布研究版' })).toBeEnabled()
  await page.getByRole('button', { name: '确认发布研究版' }).click()
  await expect(page.getByLabel('选择因子发布')).toHaveValue('factor-release-test')
  expect(published).toBeTruthy()
  await page.getByRole('button', { name: '导入待审核候选' }).click()
  await expect(page.getByRole('status').filter({ hasText: '保留人工准入审核' })).toBeVisible()
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1)).toBeTruthy()
  await page.getByRole('tab', { name: '因子收益率', exact: true }).click()
  await page.getByRole('tab', { name: '收益归因', exact: true }).click()
  await page.getByLabel('归因模型').selectOption('ff3')
  await expect(page.getByLabel('FF3 因子收益数据集')).toHaveValue('')
  await expect(page.getByRole('button', { name: '运行归因研究' })).toBeDisabled()
  await page.goto('/pre-investment/saa')
  await page.getByText('参考：因子证据', { exact: true }).click()
  await page.getByText('因子研究证据', { exact: true }).click()
  await page.getByLabel('引用因子发布').selectOption('factor-release-test')
  await expect(page.getByLabel('投研因子证据')).toContainText('测试ETF0')
  await page.getByLabel('研究对象 / 组合版本 ID').fill('acceptance-saa-study')
  await page.getByLabel('使用依据', { exact: true }).fill('验证版本化研究证据引用')
  await page.getByRole('button', { name: '登记本环节引用' }).click()
  await expect(page.getByText('已登记发布版本及来源运行。')).toBeVisible()
  expect(bindings[0].run_id).toBe(fixtureRun.id)
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1)).toBeTruthy()
  expect(errors).toEqual([])
})

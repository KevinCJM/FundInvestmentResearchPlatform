import { test, expect } from '@playwright/test'
import { readFile } from 'node:fs/promises'
import { fixtureCatalog } from '../src/test/factorFixtures'
import { fixtureAttributionRun } from '../src/test/factorAttributionFixtures'

for (const mode of ['fixed', 'rolling'] as const) test(`factor contribution reconciliation and export: ${mode}`, async ({ page }, testInfo) => {
  const errors: string[] = []
  const posted: any[] = []
  const old = { ...fixtureAttributionRun, id: 'factor-attribution-legacy', name: '旧版归因', attribution: undefined }
  let latest = structuredClone(fixtureAttributionRun)
  page.on('pageerror', error => errors.push(error.message))
  await page.route('**/api/**', async route => {
    const path = new URL(route.request().url()).pathname
    const post = route.request().method() === 'POST'
    let value: unknown = { items: [] }
    if (path.endsWith('/catalog')) value = fixtureCatalog
    else if (path.endsWith('/attributions')) {
      if (post) {
        const body = route.request().postDataJSON()
        posted.push(body)
        latest = structuredClone(fixtureAttributionRun)
        latest.request = body
        latest.attribution!.mode = mode
        latest.attribution!.warmup_days = mode === 'rolling' ? 126 : 0
        value = latest
      } else value = { items: [old, latest] }
    } else if (path.endsWith('/runs/factor-attribution-legacy')) value = old
    else if (path.endsWith('/runs/' + latest.id)) value = latest
    await route.fulfill({ json: value })
  })
  await page.goto('/settings/factor-research?module=returns&tab=attribution')
  await expect(page.getByLabel('暴露估计方式')).toBeVisible()
  await page.getByLabel('暴露估计方式').selectOption(mode)
  await page.getByLabel('归因产品代码', { exact: true }).fill('000001.OF')
  if (mode === 'rolling') await expect(page.getByLabel('滚动窗口（交易日）')).toHaveValue('126')
  await page.getByRole('button', { name: '运行归因研究' }).click()
  await expect(page.getByLabel('累计因子贡献')).toContainText('3.6 个百分点')
  expect(posted[0].exposure_mode).toBe(mode)
  await expect(page.getByRole('status').filter({ hasText: '完整对账' })).toContainText('2 / 2 日')
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBeTruthy()
  await page.getByRole('heading', { name: '累计贡献与实际收益对账' }).scrollIntoViewIfNeeded()
  await page.screenshot({ path: testInfo.outputPath('contribution-reconciliation.png'), fullPage: false })
  await page.getByText('逐日暴露、收益贡献与估计证据', { exact: true }).click()
  await expect(page.getByLabel('逐日收益贡献')).toContainText('2023-12-29')
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBeTruthy()
  const downloadEvent = page.waitForEvent('download')
  await page.getByRole('button', { name: '导出逐日贡献 CSV' }).click()
  const download = await downloadEvent
  expect(download.suggestedFilename()).toBe('attribution-000001.OF.csv')
  const content = await readFile((await download.path())!, 'utf8')
  expect(content).toContain('reconciliation_error')
  expect(content).toContain('"-0.04"')
  expect(content).toContain('decimal_return_contribution')
  await page.getByLabel('贡献评价区间').selectOption('in_sample')
  await expect(page.getByRole('status').filter({ hasText: '无可评价日期' })).toBeVisible()
  await page.getByLabel('历史归因运行').selectOption(old.id)
  await expect(page.getByText(/旧运行未保存逐日贡献/)).toBeVisible()
  await expect(page.getByLabel('累计因子贡献')).toHaveCount(0)
  expect(errors).toEqual([])
})

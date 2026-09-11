import { test, expect } from '@playwright/test'

const baseStatus = () => ({
  source: 'tushare', enabled: true, full_refresh_enabled: true,
  available_modules: ['base', 'etf', 'fund', 'index', 'macro'],
  token_configured: true, token_configuration_enabled: true, token_editable: true,
  job: { status: 'idle', message: '尚未启动更新' }, datasets: {},
})

test('everyday sync uses explicit selections and links to verified results', async ({ page }, testInfo) => {
  const errors: string[] = []
  const started: Record<string, unknown>[] = []
  page.on('pageerror', error => errors.push(error.message))
  await page.route('**/api/**', async route => {
    const pathname = new URL(route.request().url()).pathname
    if (pathname === '/api/data/refresh/status') return route.fulfill({ json: baseStatus() })
    if (pathname === '/api/data/refresh' && route.request().method() === 'POST') {
      started.push(route.request().postDataJSON())
      return route.fulfill({ json: { ...baseStatus(), job: {
        status: 'succeeded', job_id: 'browser-test', mode: 'incremental',
        finished_at: new Date().toISOString(), message: '下载及分析快照处理完成',
      } } })
    }
    return route.fulfill({ status: 404, json: { detail: 'Offline browser fixture' } })
  })
  await page.goto('/settings/data-sources')
  await page.getByText('原 Tushare 全市场任务与旧快照维护', { exact: true }).click()
  await expect(page.getByRole('button', { name: '开始数据更新' })).toBeEnabled()
  await expect(page.getByLabel('输入 Token')).not.toBeVisible()
  const summary = page.getByRole('region', { name: '本次同步清单' })
  await expect(summary.getByText('ETF', { exact: true })).toBeVisible()
  await expect(summary.getByText('指数', { exact: true })).toHaveCount(0)
  await expect(page.getByRole('group', { name: '指数下载内容' })).not.toBeVisible()
  expect(started).toHaveLength(0)
  const reviewBounds = await summary.boundingBox()
  const startBounds = await page.getByRole('button', { name: '开始数据更新' }).boundingBox()
  expect(reviewBounds).not.toBeNull()
  expect(startBounds).not.toBeNull()
  expect(startBounds!.y).toBeGreaterThanOrEqual(reviewBounds!.y + reviewBounds!.height)
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1)).toBeTruthy()
  await page.screenshot({ path: testInfo.outputPath('daily-sync.png'), fullPage: true })

  await page.getByRole('button', { name: '指数与宏观研究' }).click()
  await expect(summary.getByText('指数', { exact: true })).toBeVisible()
  await expect(summary.getByText('ETF', { exact: true })).toHaveCount(0)
  expect(started).toHaveLength(0)
  await page.getByRole('button', { name: '自定义范围' }).click()
  const scopes = page.getByRole('group', { name: '指数下载内容' })
  await expect(scopes).toBeVisible()
  await expect(scopes.getByRole('checkbox', { name: /指数目录/ })).toBeDisabled()
  await scopes.getByRole('checkbox', { name: /概念板块/ }).check()
  await page.getByRole('button', { name: '开始数据更新' }).click()
  await expect(page.getByText('下载任务已结束，接下来检查数据质量。')).toBeVisible()
  expect(started).toHaveLength(1)
  expect(started[0].modules).toEqual(['base', 'index', 'macro'])
  expect(started[0].mode).toBe('incremental')
  await expect(page.getByRole('link', { name: '查看标准表映射结果 →' })).toHaveAttribute('href', '/settings/source-center?view=results')
  await expect(page.getByRole('link', { name: '检查数据质量 →' })).toHaveAttribute('href', '/settings/data-quality')
  expect(errors).toEqual([])
})

test('failed post-processing offers repair ahead of settings without refetching', async ({ page }, testInfo) => {
  const calls: string[] = []
  await page.route('**/api/**', async route => {
    const pathname = new URL(route.request().url()).pathname
    if (pathname === '/api/data/refresh/status') return route.fulfill({ json: {
      ...baseStatus(), job: { status: 'failed', message: '分析失败', mode: 'full',
        resume_available: true, modules: ['fund'], fetch_complete: true,
        staging_data_dir: 'candidate-fixture', analytics_snapshot: { status: 'failed' } },
    } })
    if (route.request().method() === 'POST') {
      calls.push(pathname)
      return route.fulfill({ json: { status: 'succeeded' } })
    }
    return route.fulfill({ status: 404, json: { detail: 'Offline browser fixture' } })
  })
  await page.goto('/settings/data-sources')
  await page.getByText('原 Tushare 全市场任务与旧快照维护', { exact: true }).click()
  const recovery = page.getByRole('region', { name: '恢复数据处理' })
  await expect(recovery).toBeVisible()
  await expect(page.getByRole('button', { name: '按原配置继续上次更新' })).toHaveCount(0)
  expect(await page.evaluate(() => {
    const recovery = document.querySelector('[aria-label="恢复数据处理"]')!
    const form = document.querySelector('#tushare-token-input')!.closest('form')!
    return Boolean(recovery.compareDocumentPosition(form) & Node.DOCUMENT_POSITION_FOLLOWING)
  })).toBeTruthy()
  await recovery.scrollIntoViewIfNeeded()
  await page.screenshot({ path: testInfo.outputPath('sync-recovery.png') })
  await recovery.getByRole('button', { name: '重建并接入候选快照' }).click()
  await expect.poll(() => calls).toEqual(['/api/data/analytics/rebuild'])
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1)).toBeTruthy()
})

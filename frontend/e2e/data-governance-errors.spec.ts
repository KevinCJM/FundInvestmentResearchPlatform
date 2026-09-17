import { expect, test } from '@playwright/test'
import { auditTextContrast } from './helpers/contrast'

const execution = { execution_backend: 'numba_njit_fixed_signature', nopython: true, object_mode: 0, python_fallback: 0, request_time_compilation: 0, kernel_signatures: { coverage_ratio_kernel: ['fixed'] } }
const quality = {
  schema_version: 1, status: 'healthy', generated_at: '2026-09-01T08:00:00Z', activated_at: null, as_of: '2026-08-31',
  summary: { checks_total: 1, checks_passed: 1, checks_warning: 0, checks_failed: 0, checks_unavailable: 0, total_products: 10, affected_products: 0, affected_rate: 0, issue_count: 0, critical_issue_count: 0, high_issue_count: 0, medium_issue_count: 0, nav_anomaly_products: 0, nav_anomaly_events: 0, stale_active_products: 0 },
  checks: [{ key: 'nav_discontinuity', label: '净值突变', dimension: 'continuity', status: 'passed', summary: '未发现净值突变', detail: '固定离线检查结果。' }],
  issues: [], validation: { status: 'passed', manifest: 'offline-fixture' }, execution,
}
const strictSettings = {
  settings: { active_release_id: null, as_of: '2026-09-01', run_mode: 'STRICT_PIT', updated_at: null, note: '' },
  effective: { as_of: '2026-09-01', as_of_source: 'explicit', run_mode: 'STRICT_PIT', run_mode_label: '严格 PIT', data_release_id: null, no_pit: false, label: '站在 2026-09-01 · 严格 PIT' },
  release: null, release_error: null, available_releases: [], can_apply: true,
}

test('PIT 关闭状态在桌面和平板手机深色导航中保持可读', async ({ page }) => {
  await page.route('**/api/**', route => new URL(route.request().url()).pathname === '/api/pit/settings'
    ? route.fulfill({ json: { ...strictSettings, effective: {
      ...strictSettings.effective, no_pit: true, as_of: null, run_mode: 'RESEARCH', label: '无 PIT 口径 · 使用全部磁盘数据',
    } } })
    : route.fulfill({ status: 503, json: { detail: '离线环境' } }))
  await page.goto('/product-research')
  if (page.viewportSize()!.width < 1280) await page.locator('button[aria-controls="mobile-navigation"]').click()
  const badge = page.locator('[data-testid="pit-badge"]:visible')
  await expect(badge).toHaveText('PIT 关闭')
  await expect.poll(async () => (await page.evaluate(auditTextContrast)).filter(item => item.text.includes('PIT 关闭'))).toEqual([])
})

test('质量接口失败不会把未知卡片标绿，重试后真实零值可通过', async ({ page }, testInfo) => {
  let unavailable = true
  const writes: string[] = []
  await page.route('**/api/**', route => {
    const request = route.request()
    if (request.method() !== 'GET') writes.push(request.url())
    if (new URL(request.url()).pathname === '/api/data/quality' && !unavailable) return route.fulfill({ json: quality })
    return route.fulfill({ status: 503, json: { detail: '离线故障注入：服务暂不可用' } })
  })
  await page.goto('/settings/data-quality')
  const retry = page.locator('header').filter({ hasText: 'Data governance' }).getByRole('button', { name: '重新检查', exact: true })
  await expect(retry).toBeEnabled()
  const overview = page.getByRole('region', { name: '数据质量概览' })
  for (const label of ['受影响产品', '净值突变']) {
    const card = overview.locator('article').filter({ has: page.getByText(label, { exact: true }) })
    await expect(card).toContainText('--')
    await expect(card.getByText('未检查', { exact: true })).toBeVisible()
    await expect(card.getByText('通过', { exact: true })).toHaveCount(0)
  }
  await page.screenshot({ path: testInfo.outputPath('quality-read-failure.png'), fullPage: true })
  unavailable = false
  await retry.click()
  for (const label of ['受影响产品', '净值突变']) {
    const card = overview.locator('article').filter({ has: page.getByText(label, { exact: true }) })
    await expect(card.getByText('0', { exact: true })).toBeVisible()
    await expect(card.getByText('通过', { exact: true })).toBeVisible()
  }
  expect(writes).toEqual([])
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth)).toBe(true)
})

test('汇总零值但规则未检查时保持未知', async ({ page }) => {
  await page.route('**/api/**', route => new URL(route.request().url()).pathname === '/api/data/quality'
    ? route.fulfill({ json: { ...quality, status: 'unavailable', summary: { ...quality.summary, checks_passed: 0, checks_unavailable: 1 }, checks: quality.checks.map(check => ({ ...check, status: 'unavailable' })) } })
    : route.fulfill({ status: 503, json: { detail: '离线环境' } }))
  await page.goto('/settings/data-quality')
  const overview = page.getByRole('region', { name: '数据质量概览' })
  for (const label of ['受影响产品', '净值突变']) {
    const card = overview.locator('article').filter({ has: page.getByText(label, { exact: true }) })
    await expect(card.getByText('未检查', { exact: true })).toBeVisible()
    await expect(card.getByText('通过', { exact: true })).toHaveCount(0)
  }
})

test('PIT 设置失败显示未知并可重试恢复严格模式，不推断关闭', async ({ page }, testInfo) => {
  let unavailable = true
  const writes: string[] = []
  await page.route('**/api/**', route => {
    if (route.request().method() !== 'GET') writes.push(route.request().url())
    if (new URL(route.request().url()).pathname === '/api/pit/settings' && !unavailable) return route.fulfill({ json: strictSettings })
    return route.fulfill({ status: 503, json: { detail: '离线故障注入：PIT 设置暂不可读' } })
  })
  // PIT belongs to research workspaces; the landing page has no data context.
  await page.goto('/product-research')
  if (page.viewportSize()!.width < 1280) await page.locator('button[aria-controls="mobile-navigation"]').click()
  const badge = page.locator('[data-testid="pit-badge"]:visible')
  await expect(badge).toHaveText('PIT 口径未知')
  await badge.click()
  const switcher = page.getByTestId('pit-switcher')
  await expect(switcher).toContainText('系统默认：未知（读取失败）')
  await expect(switcher.getByRole('alert')).toContainText('PIT 设置暂不可读')
  await page.screenshot({ path: testInfo.outputPath('pit-settings-unknown.png'), fullPage: true })
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth)).toBe(true)
  unavailable = false
  await switcher.getByRole('button', { name: '重试读取 PIT 口径' }).click()
  await expect(badge).toHaveText('PIT 打开：2026-09-01')
  await expect(switcher).toContainText('严格 PIT')
  await expect(switcher.getByRole('alert')).toHaveCount(0)
  expect(writes).toEqual([])
})

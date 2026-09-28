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
  const retry = page.getByRole('alert').getByRole('button', { name: '重试', exact: true })
  await expect(retry).toBeEnabled()
  await expect(page.locator('img[src*="mascot-error"]')).toBeVisible()
  const overview = page.getByRole('region', { name: '数据质量概览' })
  await expect(overview).toHaveCount(0)
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


const routes = [
  'data-sources', 'source-center', 'source-center?view=resolution', 'data-model', 'data-quality',
  'research-data-lab', 'pit-snapshots', 'risk-scales', 'risk-scales/new', 'risk-scales/drafts/unavailable',
  'risk-scales/versions/unavailable', 'risk-scales/compare?left=a&right=b', 'indicators-models',
  'factor-research', 'factor-research?module=returns&tab=datasets', 'risk-models', 'timing-algorithms',
  'scenario-algorithms', 'scenario-algorithms?center=market-state&stage=realtime',
  'scenario-algorithms?center=market-state&stage=validation',
  'scenario-algorithms?center=market-state&stage=historical&new=1',
  'scenario-algorithms?center=market-state&stage=realtime&new=1',
  'scenario-algorithms?center=market-state&definition=unavailable&revision=1',
  'scenario-algorithms?center=events', 'scenario-algorithms?center=events&event_view=manual',
  'scenario-algorithms?center=simulation', 'scenario-algorithms/apply', 'language-terminology',
]

for (const route of routes) test(`设置读取失败可重试：${route}`, async ({ page }, info) => {
  let reads = 0
  const writes: string[] = []
  await page.route('**/api/**', request => {
    reads++
    if (request.request().method() !== 'GET') writes.push(request.request().method() + ' ' + new URL(request.request().url()).pathname)
    return request.fulfill({ status: 503, json: { detail: 'INTERNAL_DATABASE_ERROR: sensitive-query-must-not-render' } })
  })
  await page.goto(`/settings/${route}`)
  const mascot = page.locator('img[src*="mascot-error"]:visible')
  await expect(mascot).toHaveCount(1)
  const panel = page.getByRole('alert').filter({ has: mascot })
  await expect(panel).toContainText(/重试|重新加载/)
  const retry = panel.getByRole('button', { name: /重试|重新加载/ })
  await expect(retry).toBeEnabled()
  const before = reads
  await retry.click()
  await expect.poll(() => reads).toBeGreaterThan(before)
  await expect(mascot).toHaveCount(1)
  await expect(panel).not.toContainText('sensitive-query')
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth)).toBe(true)
  const contrast = (await page.evaluate(auditTextContrast)).filter(item => /暂时无法读取|重试/.test(item.text))
  expect(contrast).toEqual([])
  // Initial graph inference is an existing read-only POST; no business record is saved by retry.
  expect(writes.filter(item => !item.endsWith('/infer') && !item.endsWith('/risk-scales/compare'))).toEqual([])
  if (route === 'risk-scales/new' || route === 'scenario-algorithms') {
    await mascot.scrollIntoViewIfNeeded()
    await page.screenshot({ path: info.outputPath('read-failure.png') })
  }
})

test('断网且插画不可用时仍可理解错误，恢复后保留表单与保存失败提示', async ({ page }) => {
  let offline = true
  const { riskCapabilities } = await import('../src/test/riskScaleFixtures')
  await page.route('**/homepage/images/mascot-error-240.webp', route => route.abort())
  await page.route('**/api/**', route => {
    const path = new URL(route.request().url()).pathname
    if (path === '/api/pit/settings') return route.fulfill({ json: {
      settings: { active_release_id: null, as_of: '2019-12-31', run_mode: 'STRICT_PIT', updated_at: null, note: '' },
      effective: { as_of: '2019-12-31', as_of_source: 'explicit', run_mode: 'STRICT_PIT', run_mode_label: '严格 PIT', data_release_id: null, no_pit: false, label: '站在 2019-12-31 · 严格 PIT' },
      release: null, release_error: null, available_releases: [], can_apply: true,
    } })
    if (path.endsWith('/risk-scales/capabilities')) {
      return offline ? route.abort() : route.fulfill({ json: riskCapabilities })
    }
    return route.fulfill({ status: 503, json: { detail: '服务暂不可用' } })
  })
  await page.goto('/settings/risk-scales/new')
  await expect(page.getByRole('alert')).toContainText('暂时无法读取数据，请重试。')
  const retry = page.getByRole('alert').getByRole('button', { name: '重试' })
  await retry.focus()
  await expect(retry).toBeFocused()
  offline = false
  await retry.press('Enter')
  const name = page.getByLabel('标尺名称', { exact: true })
  await expect(name).toBeVisible()
  await expect(page.locator('img[src*="mascot-error"]')).toHaveCount(0)
  await name.fill('保留这份未完成的研究')
  await page.getByRole('button', { name: '保存草稿', exact: true }).click()
  await expect(page.getByRole('alert')).toBeVisible()
  await expect(name).toHaveValue('保留这份未完成的研究')
  await expect(page.locator('img[src*="mascot-error"]')).toHaveCount(0)
})

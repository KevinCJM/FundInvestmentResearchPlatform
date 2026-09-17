import { expect, test } from '@playwright/test'
import { auditTextContrast } from './helpers/contrast'

async function offlineScenarioApi(page: import('@playwright/test').Page) {
  await page.route('**/api/**', async route => {
    const url = new URL(route.request().url())
    const path = url.pathname
    const method = route.request().method()
    const body = method === 'POST' ? route.request().postDataJSON() : undefined

    if (path.endsWith('/nodes')) return route.fulfill({ json: { items: [] } })
    if (path.endsWith('/templates/v2')) return route.fulfill({ json: { items: [] } })
    if (path.endsWith('/v2/definitions')) return route.fulfill({ json: { items: [] } })
    if (path.endsWith('/v2/graph-assets') || path.endsWith('/v2/experiments')) return route.fulfill({ json: { items: [] } })
    if (path.endsWith('/reference-quality/catalog')) return route.fulfill({ json: { items: [] } })
    if (path.endsWith('/reliability/catalog')) return route.fulfill({ json: { items: [] } })
    if (path.endsWith('/references')) return route.fulfill({ json: { items: [] } })
    if (path.includes('/prospective/') && method === 'GET') return route.fulfill({ json: { items: [] } })
    if (path.endsWith('/research-series/catalog')) return route.fulfill({ json: { items: [], total: 0, offset: 0, limit: 500 } })
    if (path.endsWith('/infer') && method === 'POST') return route.fulfill({ json: { valid: false, errors: [], warnings: [], inferred: { nodes: {} } } })
    if (path.endsWith('/authoring/resolve') && method === 'POST') return route.fulfill({ json: { valid: false, definition: body?.definition, diagnostics: [], formula_text: '', output_id: 'state' } })

    // The shell can request unrelated read-only settings. Keep this acceptance
    // focused on the market-state workflow without inventing domain data.
    if (method === 'GET') return route.fulfill({ json: { items: [] } })
    return route.fulfill({ status: 400, json: { detail: 'Offline market-state acceptance fixture' } })
  })
}

async function audit(page: import('@playwright/test').Page) {
  await expect.poll(() => page.evaluate(auditTextContrast)).toEqual([])
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1)).toBeTruthy()
}

test('市场状态研究把历史参考、实时识别和有效性验证串成一个三步流程', async ({ page }) => {
  await offlineScenarioApi(page)
  await page.goto('/settings/scenario-algorithms?center=market-state&stage=historical')

  const outer = page.getByRole('tablist', { name: '情景研究类型' })
  await expect(outer.getByRole('tab', { name: /市场状态研究/ })).toHaveAttribute('aria-selected', 'true')
  const steps = page.getByRole('tablist', { name: '市场状态研究步骤' })
  await expect(steps.getByRole('tab', { name: /定义历史参考/ })).toHaveAttribute('aria-selected', 'true')
  await expect(page.getByRole('heading', { name: '定义历史参考', exact: true })).toBeVisible()

  await steps.getByRole('tab', { name: /建立实时识别/ }).click()
  await expect(page).toHaveURL(/center=market-state.*stage=realtime|stage=realtime.*center=market-state/)
  await expect(page.getByRole('heading', { name: '建立实时识别', exact: true })).toBeVisible()
  await expect(page.getByText('还没有已确认的历史参考。先生成历史区间，再保存为历史参考。')).toBeVisible()

  await steps.getByRole('tab', { name: /验证识别能力/ }).click()
  await expect(page).toHaveURL(/stage=validation/)
  await expect(page.locator('h1').filter({ hasText: /^验证识别能力$/ })).toBeVisible()
  await audit(page)
})

test('旧 historical/realtime 深链接继续落到新的市场状态三步流程', async ({ page }) => {
  await offlineScenarioApi(page)
  await page.goto('/settings/scenario-algorithms?center=historical&mode=realtime')
  const steps = page.getByRole('tablist', { name: '市场状态研究步骤' })
  await expect(steps.getByRole('tab', { name: /建立实时识别/ })).toHaveAttribute('aria-selected', 'true')
  await expect(page.getByRole('heading', { name: '建立实时识别', exact: true })).toBeVisible()
  await audit(page)
})

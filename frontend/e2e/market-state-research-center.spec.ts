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

test('已保存算法展示当前版本 PIT，缺失与读取失败明确区分', async ({ page }, info) => {
  await offlineScenarioApi(page)
  const definitions = ['有截止日', '未设截止日', '旧记录', '新修订'].map((name, index) => ({
    id: `pit-${index}`, revision: 3, schema_version: '2.0', name,
    description: '固定测试数据：保存时间与运行的 PIT 截止日不同。',
    study: { purpose: 'historical_reference', family: 'custom' },
    states: [], graph: { nodes: [], outputs: {} }, updated_at: '2026-09-23',
  }))
  const run = (id: string, revision: number, created_at: string, cutoff?: string | null) => ({
    id: `${id}-${created_at}`, name: '正式运行', definition_id: id, definition_revision: revision,
    mode: 'retrospective', created_at, ...(cutoff !== undefined ? { as_of: cutoff } : {}),
  })
  const runs = [
    run('pit-0', 3, '2026-09-20', '2019-12-31'),
    run('pit-0', 2, '2026-09-22', '2025-01-01'),
    run('pit-0', 3, '2026-09-19', '2018-12-31'),
    run('pit-1', 3, '2026-09-20', null),
    run('pit-2', 3, '2026-09-20'),
    run('pit-3', 2, '2026-09-20', '2019-12-31'),
  ]
  await page.route('**/api/historical-regimes/v2/definitions', route => route.fulfill({ json: { items: definitions } }))
  let failed = true
  await page.route('**/api/historical-regimes/runs?**', route => route.fulfill(failed
    ? { status: 503, json: { detail: '运行目录不可用' } }
    : { json: { items: runs } }))
  await page.goto('/settings/scenario-algorithms?center=market-state&stage=historical')
  const list = page.getByTestId('regime-study-list')
  await expect(list.getByRole('alert')).toContainText('正式运行记录读取失败')
  await expect(list.getByText('读取失败', { exact: true })).toHaveCount(4)
  failed = false
  await list.getByRole('button', { name: '重试读取 PIT 日期' }).click()
  await expect(list.locator('time[datetime="2019-12-31"]')).toBeVisible()
  await expect(list.getByRole('columnheader', { name: 'PIT 日期' })).toBeVisible()
  await expect(list.getByText('未设置', { exact: true })).toBeVisible()
  await expect(list.getByText('未记录', { exact: true })).toBeVisible()
  await expect(list.getByText('当前版本尚未运行', { exact: true })).toBeVisible()
  await expect(list.locator('time[datetime="2025-01-01"]')).toHaveCount(0)
  await expect(list.getByRole('alert')).toHaveCount(0)
  await audit(page)
  await page.screenshot({ path: info.outputPath('saved-algorithm-pit.png'), fullPage: true })
  const scrollArea = list.getByLabel('定义历史参考的已保存研究', { exact: true })
  await scrollArea.focus()
  await expect(scrollArea).toBeFocused()
  const pitDate = list.locator('time[datetime="2019-12-31"]')
  await pitDate.scrollIntoViewIfNeeded()
  await expect(pitDate).toBeInViewport()
  await expect(pitDate).toHaveText('2019/12/31')
  await page.screenshot({ path: info.outputPath('pit-column-in-view.png') })
  await expect(list.getByRole('link', { name: '继续研究' }).first()).toHaveAttribute('href', /definition=pit-0.*revision=3/)
})

test('市场状态研究每一步先给已保存清单，点新建才进工作台', async ({ page }) => {
  await offlineScenarioApi(page)
  await page.goto('/settings/scenario-algorithms?center=market-state&stage=historical')

  const outer = page.getByRole('tablist', { name: '情景研究类型' })
  await expect(outer.getByRole('tab', { name: /市场状态研究/ })).toHaveAttribute('aria-selected', 'true')
  const steps = page.getByRole('tablist', { name: '市场状态研究步骤' })
  await expect(steps.getByRole('tab', { name: /定义历史参考/ })).toHaveAttribute('aria-selected', 'true')
  await expect(page.getByRole('heading', { name: '定义历史参考', exact: true })).toBeVisible()
  await expect(page.getByText('还没有历史参考算法')).toBeVisible()

  await steps.getByRole('tab', { name: /建立实时识别/ }).click()
  await expect(page).toHaveURL(/center=market-state.*stage=realtime|stage=realtime.*center=market-state/)
  await expect(page.getByRole('heading', { name: '建立实时识别', exact: true })).toBeVisible()
  await expect(page.getByText('还没有实时识别模型')).toBeVisible()

  await steps.getByRole('tab', { name: /验证识别能力/ }).click()
  await expect(page).toHaveURL(/stage=validation/)
  await expect(page.getByRole('heading', { name: '验证识别能力与应用', exact: true })).toBeVisible()
  await expect(page.getByText('还没有可验证的实时识别模型')).toBeVisible()
  await audit(page)

  // 新建才进工作台；工作台可以退回本步骤清单，地址栏不再带研究身份。
  await steps.getByRole('tab', { name: /定义历史参考/ }).click()
  await page.getByRole('link', { name: '新建历史参考算法' }).first().click()
  await expect(page).toHaveURL(/new=1/)
  await expect(page.locator('h1').filter({ hasText: /^定义历史参考$/ })).toBeVisible()
  await page.getByRole('link', { name: '返回历史参考清单' }).click()
  await expect(page).not.toHaveURL(/new=1/)
  await expect(page.getByText('还没有历史参考算法')).toBeVisible()
})

test('旧 historical/realtime 深链接继续落到新的市场状态三步流程', async ({ page }) => {
  await offlineScenarioApi(page)
  await page.goto('/settings/scenario-algorithms?center=historical&mode=realtime')
  const steps = page.getByRole('tablist', { name: '市场状态研究步骤' })
  await expect(steps.getByRole('tab', { name: /建立实时识别/ })).toHaveAttribute('aria-selected', 'true')
  await expect(page.getByRole('heading', { name: '建立实时识别', exact: true })).toBeVisible()
  await audit(page)
})

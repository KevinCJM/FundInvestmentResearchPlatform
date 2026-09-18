import { test, expect, type Page } from '@playwright/test'
import path from 'node:path'
import fs from 'node:fs/promises'

const screenshots = path.resolve('../.tmp_risk_scales_20260916/screenshots')
async function noOverflow(page: Page) {
  try {
    // ECharts resizes after the viewport event; verify the settled layout, not one frame.
    await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1), { timeout: 5000 }).toBe(true)
  } catch (error) {
    console.log('OVERFLOW', await page.evaluate(() => ({
      viewport: window.innerWidth, document: document.documentElement.scrollWidth,
      elements: Array.from(document.querySelectorAll('body *')).filter(element => {
        const box = element.getBoundingClientRect()
        return box.width > 0 && box.right > window.innerWidth + 1
      }).slice(0, 16).map(element => ({ tag: element.tagName, class: element.getAttribute('class'), width: element.getBoundingClientRect().width, text: element.textContent?.slice(0, 100) })),
    })))
    await shot(page, `overflow-${page.viewportSize()?.width}`)
    throw error
  }
}
async function shot(page: Page, name: string) {
  await fs.mkdir(screenshots, { recursive: true })
  await page.evaluate(() => window.scrollTo(0, 0))
  await page.screenshot({ path: path.join(screenshots, `${name}.png`), fullPage: true })
}

test('historical intersection to risk scale, five segmentation methods, publish and immutable version', async ({ page }) => {
  const browserErrors: string[] = []
  page.on('pageerror', error => browserErrors.push(error.message))
  const fixtureResponse = await page.request.get('/api/risk-scale-fixture')
  expect(fixtureResponse.ok()).toBe(true)
  const fixture = await fixtureResponse.json(); expect(fixture.test_only).toBe(true)
  const reference = fixture.reference_request
  await page.goto('/settings/risk-scales')
  await expect(page.getByText('尚未配置风险等级', { exact: true })).toBeVisible()
  await expect(page.getByLabel('本位币')).toHaveCount(0)
  await expect(page.getByLabel('风险口径')).toHaveCount(0)
  await page.getByRole('link', { name: '新建标尺', exact: true }).click()
  await page.getByLabel('标尺名称', { exact: true }).fill('浏览器合成测试风险标尺')
  await expect(page.getByLabel(/参考研究日/)).toHaveAttribute('readonly')
  await expect(page.getByLabel('预计复核日（可选）')).toBeVisible()
  await expect(page.getByLabel('预测期限（年）')).toHaveCount(0)
  await expect(page.getByLabel('风险口径')).toHaveCount(0)
  await page.getByLabel(/适用范围与用途/).fill('仅用于离线浏览器完整链路测试，不用于真实投资')
  await page.getByRole('button', { name: '下一步', exact: true }).click()
  for (const asset of reference.assets) {
    await page.getByRole('button', { name: '添加参考大类', exact: true }).click()
    const named = page.getByTestId('risk-reference-asset').last()
    await named.getByLabel('大类名称').fill(asset.name)
    await named.getByLabel('资产类型').selectOption(asset.asset_type)
    await named.getByLabel(/经济定义与依据/).fill(asset.rationale)
    if (asset.asset_type === 'cash') {
      await named.getByLabel('现金预期年化收益率（%）').fill(String(asset.cash_return * 100))
      await expect(named.getByRole('button', { name: '添加或更换代理' })).toHaveCount(0)
    } else {
      await expect(named.getByLabel('经济角色')).toHaveCount(0)
      await expect(named.getByLabel('代理再平衡')).toHaveValue('daily')
      await named.getByRole('button', { name: '添加或更换代理' }).click()
      await named.getByLabel('来源类型').selectOption('etf')
      await named.getByLabel('搜索名称或代码').fill(asset.components[0].series_id.split(':').at(-1))
      await expect(named.getByLabel('冻结产品池')).toHaveCount(0)
      await named.getByRole('button', { name: '选择', exact: true }).click()
    }
  }
  await page.getByRole('button', { name: '检查参考数据', exact: true }).click()
  await expect(page.getByText(/共同历史区间/).first()).toBeVisible()
  await page.getByLabel(/我已核对来源与警告，确认保存参考资产/).check()
  await page.getByRole('button', { name: '确认保存参考资产与代理', exact: true }).click()
  await expect(page.getByText('历史参数与约束', { exact: true })).toBeVisible()
  await expect(page.getByText(/共同历史区间/).first()).toBeVisible()
  const parameterTable = page.getByRole('table', { name: '历史参数摘要' })
  for (const asset of reference.assets) await expect(parameterTable.getByText(asset.name, { exact: true })).toBeVisible()
  const assetBounds = page.locator('details').filter({ hasText: '单大类约束' })
  await expect(assetBounds).not.toHaveAttribute('open', '')
  await expect(page.getByText(/CMA/)).toHaveCount(0)
  await page.getByRole('button', { name: '计算前沿与五档', exact: true }).click()
  await expect(page.getByRole('table', { name: 'C1–C5 风险等级', exact: true })).toBeVisible()
  await expect(page.getByTestId('risk-frontier-chart').locator('canvas')).toBeVisible()
  for (const method of ['equal_volatility_v1', 'equal_arclength_v1', 'equal_return_v1', 'manual_volatility_bands_v1', 'frontier_shape_dp_v2']) {
    await page.getByLabel('分档方式').selectOption(method)
    await expect(page.getByRole('button', { name: '更新前沿与分档预览' })).toHaveCount(0)
    await expect(page.getByRole('button', { name: '核对发布内容' })).toBeDisabled()
    await expect(page.getByRole('button', { name: '核对发布内容' })).toBeEnabled()
  }
  const c1 = page.getByLabel('C1 上限（%）'), c2 = page.getByLabel('C2 上限（%）')
  const c1Value = Number(await c1.inputValue()), c2Value = Number(await c2.inputValue())
  await c1.fill(String((c1Value + c2Value) / 2))
  await expect(page.getByRole('button', { name: '核对发布内容' })).toBeDisabled()
  await expect(page.getByRole('button', { name: '核对发布内容' })).toBeEnabled()
  await expect(page.getByText(/分档边界已在算法结果上人工微调/)).toBeVisible()
  await page.getByRole('button', { name: 'C3', exact: true }).click()
  await expect(page.getByText('C3 代表权重与风险画像', { exact: true })).toBeVisible()
  for (const width of [320, 768, 1279, 1280, 1440]) {
    await page.setViewportSize({ width, height: 1000 }); await noOverflow(page); await shot(page, `frontier-zh-${width}`)
  }
  const draftSaved = page.waitForResponse(r => r.url().endsWith('/risk-scales/drafts') && r.request().method() === 'POST')
  await page.getByRole('button', { name: '保存草稿', exact: true }).click()
  await expect(page.getByText(/已保存修订/)).toBeVisible()
  const savedDraft = await (await draftSaved).json()
  expect(savedDraft.editable_definition.step).toBe(2)
  await page.goto(`/settings/risk-scales/drafts/${savedDraft.id}`)
  await page.getByRole('button', { name: '计算前沿与五档', exact: true }).click()
  await page.getByRole('button', { name: '核对发布内容', exact: true }).click()
  const publication = page.locator('#risk-step-title').locator('..')
  const checkboxes = publication.getByRole('checkbox')
  for (let i = 0; i < await checkboxes.count(); i++) await checkboxes.nth(i).check()
  await page.getByRole('button', { name: '发布新版本', exact: true }).click()
  await expect(page).toHaveURL(/\/versions\//)
  const versionUrl = page.url()
  await page.getByRole('button', { name: '设为系统默认', exact: true }).click()
  await expect(page.getByRole('button', { name: '确认执行', exact: true })).toBeDisabled()
  await page.getByLabel('我已核对版本及本次操作的影响。').check()
  await page.getByRole('button', { name: '确认执行', exact: true }).click()
  await expect(page.getByText('默认已更新；已有引用保持原版本。')).toBeVisible()
  const previews: string[] = []; page.on('request', request => { if (request.url().endsWith('/risk-scales/preview')) previews.push(request.url()) })
  await page.reload(); await expect(page.getByText('只读版本', { exact: true })).toBeVisible(); expect(previews).toEqual([])
  await shot(page, 'published-default-zh-1440')
  await page.evaluate(() => localStorage.setItem('fund-research.i18n.locale', 'en-US'))
  await page.reload(); await expect(page.getByText('Read-only version', { exact: true })).toBeVisible()
  for (const width of [320, 768, 1440]) { await page.setViewportSize({ width, height: 1000 }); await noOverflow(page); await shot(page, `published-en-${width}`) }
  // 200% zoom on a 1440px screen has a 720px CSS viewport. Root CSS zoom is
  // not browser zoom: it leaves desktop media queries active and is not a valid reflow test.
  const cdp = await page.context().newCDPSession(page)
  await cdp.send('Emulation.setDeviceMetricsOverride', { width: 720, height: 500, deviceScaleFactor: 2, mobile: false })
  expect(await page.evaluate(() => window.innerWidth)).toBe(720)
  await noOverflow(page)
  await page.keyboard.press('Tab'); expect(await page.evaluate(() => document.activeElement !== document.body)).toBe(true)
  await shot(page, 'published-en-200-percent-equivalent')
  await cdp.send('Emulation.clearDeviceMetricsOverride'); await cdp.detach()
  expect(page.url()).toBe(versionUrl)
  await page.goto('/settings/risk-scales')
  await expect(page.getByRole('table', { name: 'Risk scale configurations' })).toBeVisible()
  await expect(page.getByRole('link', { name: 'View details' })).toBeVisible()
  await expect(page.getByRole('link', { name: 'Edit' }).first()).toBeVisible()
  await expect(page.getByRole('button', { name: 'Delete' }).first()).toBeVisible()
  await expect(page.getByText('System default', { exact: true })).toBeVisible()

  const scaleRoot = '/api/strategic-allocation/risk-scales'
  const originalId = new URL(versionUrl).pathname.split('/').at(-1)!
  const original = await (await page.request.get(`${scaleRoot}/${originalId}`)).json()
  const secondRequest = { ...original.preview.request_echo,
    definition: { ...original.preview.request_echo.definition, name: 'Browser comparison version' } }
  const secondPreviewResponse = await page.request.post(`${scaleRoot}/preview`, { data: secondRequest })
  expect(secondPreviewResponse.status()).toBe(200)
  const secondPreview = await secondPreviewResponse.json()
  const secondResponse = await page.request.post(`${scaleRoot}/confirm`, { data: { request: secondRequest,
    preview_hash: secondPreview.preview_hash, confirm: true, idempotency_key: 'browser-comparison-version',
    acknowledged_warnings: secondPreview.warnings.map((warning: any) => warning.code) } })
  expect(secondResponse.status()).toBe(201)
  // Use the real paginated API with one row per page to exercise navigation compactly.
  await page.route('**/api/strategic-allocation/risk-scales?*', async route => {
    const url = new URL(route.request().url()); url.searchParams.set('limit', '1')
    await route.fulfill({ response: await route.fetch({ url: url.toString() }) })
  })
  await page.reload()
  await page.getByRole('checkbox', { name: /Browser comparison version/ }).check()
  await page.getByRole('button', { name: 'Load more versions', exact: true }).click()
  await page.getByRole('checkbox', { name: /浏览器合成测试风险标尺/ }).check()
  await page.getByRole('link', { name: 'Compare selected versions', exact: true }).click()
  await expect(page).toHaveURL(/risk-scales\/compare\?left=.+&right=.+/)
  await expect(page.getByRole('heading', { name: 'Browser comparison version · v2', exact: true })).toBeVisible()
  for (const width of [320, 768, 1440]) {
    await page.setViewportSize({ width, height: 1000 }); await noOverflow(page); await shot(page, `comparison-en-${width}`)
  }
  await page.getByRole('link', { name: 'Back to risk scales', exact: true }).click()
  await page.getByRole('button', { name: 'Load more versions', exact: true }).click()
  const defaultRow = page.getByRole('row').filter({ hasText: 'System default' })
  await defaultRow.getByRole('button', { name: 'Delete', exact: true }).click()
  await expect(page.getByRole('button', { name: 'Confirm delete', exact: true })).toBeDisabled()
  await page.getByRole('checkbox', { name: /I explicitly confirm clearing/ }).check()
  await page.getByRole('button', { name: 'Confirm delete', exact: true }).click()
  await expect(page.getByText('System default', { exact: true })).toHaveCount(0)
  const retired = await (await page.request.get(`${scaleRoot}/${originalId}`)).json()
  expect(retired.retired).toBe(true)
  await page.getByRole('checkbox', { name: 'Include retired historical versions', exact: true }).check()
  await page.getByRole('button', { name: 'Load more versions', exact: true }).click()
  const historicalRow = page.getByRole('row').filter({ hasText: '浏览器合成测试风险标尺' })
  await historicalRow.getByRole('link', { name: 'View details', exact: true }).click()
  await expect(page).toHaveURL(versionUrl)
  await expect(page.getByRole('button', { name: 'Set as system default', exact: true })).toBeDisabled()
  expect(browserErrors).toEqual([])
})

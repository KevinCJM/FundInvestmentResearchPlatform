import { expect, test, type Page } from '@playwright/test'
import { auditTextContrast } from './helpers/contrast'

const api = 'http://127.0.0.1:8769'
test.afterEach(async ({ page }) => {
  // The newly mounted realtime workbench may still be loading its catalog.
  // Finish real API forwards before Playwright disposes the request context.
  await page.unrouteAll({ behavior: 'wait' })
})
async function connect(page: Page) {
  await page.route('**/api/**', async route => {
    const url = new URL(route.request().url())
    if (!/^\/api\/(historical-regimes|scenario-stress|published-scenarios|risk-models|scenario-transmission)(\/|$)/.test(url.pathname)) {
      return route.fulfill({ status: 404, json: { detail: 'Unrelated service outside this isolated fixture.' } })
    }
    const response = await route.fetch({ url: api + url.pathname + url.search, timeout: 120_000 })
    await route.fulfill({ response })
  })
  const ready = await page.request.get(api + '/ready')
  expect(ready.ok()).toBeTruthy()
  return ready.json() as Promise<{ factor_name: string; definition_a: string; definition_b: string; historical_id: string }>
}
async function visual(page: Page, filename: string) {
  await expect.poll(() => page.evaluate(auditTextContrast)).toEqual([])
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBeTruthy()
  await page.evaluate(() => window.scrollTo(0, 0))
  await page.screenshot({ path: filename, fullPage: true })
}

test('事件库空筛选及字段错误连接真实 API', async ({ page }, info) => {
  await connect(page)
  const loaded = page.waitForResponse(r => new URL(r.url()).pathname.endsWith('/event-library/events'))
  await page.goto('/settings/scenario-algorithms?center=events')
  const response = await loaded
  expect(response.status(), await response.text()).toBe(200)
  expect(new URL(response.url()).searchParams.has('start')).toBe(false)
  expect(new URL(response.url()).searchParams.has('end')).toBe(false)
  await expect(page.getByRole('heading', { name: '全球历史事件库', exact: true })).toBeVisible()
  await page.getByRole('button', { name: '新建事件', exact: true }).click()
  await page.getByLabel('库事件名称').fill('验证结束日期错误')
  await page.getByLabel('窗口1开始', { exact: true }).fill('2020-03-01')
  await page.getByLabel('窗口1结束', { exact: true }).fill('2020-02-01')
  await page.getByLabel('窗口1理由').fill('用于验证错误字段能够被用户识别。')
  const rejected = page.waitForResponse(r => r.request().method() === 'POST' && r.url().endsWith('/event-library/events'))
  await page.getByRole('button', { name: '保存事件', exact: true }).click()
  expect((await rejected).status()).toBe(422)
  await expect(page.getByRole('alert')).toContainText(/windows|窗口|结束/)
  await expect(page.getByRole('alert')).not.toContainText('事件库操作失败')
  await expect(page.getByLabel('库事件名称')).toHaveValue('验证结束日期错误')
  await visual(page, info.outputPath('event-error.png'))
})

test('确认历史参考后下一步自动带入精确发布版本', async ({ page }, info) => {
  const identity = await connect(page)
  await page.goto(`/settings/scenario-algorithms?center=historical&definition=${identity.historical_id}&revision=1`)
  await expect(page.getByLabel('研究名称')).toHaveValue('浏览器验收·历史定义（合成数据）')
  await page.getByRole('button', { name: '保存', exact: true }).click()
  const dialog = page.getByRole('dialog')
  const savedResponse = page.waitForResponse(r => r.request().method() === 'POST' && r.url().includes('/research-versions'))
  await dialog.getByRole('button', { name: '保存为历史参考', exact: true }).click()
  const response = await savedResponse
  expect(response.ok(), await response.text()).toBeTruthy()
  const version = await response.json()
  expect(version.historical_reference?.content_hash).toBeTruthy()
  await page.keyboard.press('Escape')
  await page.getByRole('button', { name: '下一步：建立实时识别', exact: true }).click()
  const reference = version.historical_reference
  await expect(page.getByLabel('历史参考版本')).toHaveValue(JSON.stringify([reference.run_id, reference.publication_id, reference.content_hash]))
  await visual(page, info.outputPath('historical-reference-handoff.png'))
})

test('旧版无 Horizon 情景仍能查看路径', async ({ page }, info) => {
  await connect(page)
  const errors: string[] = []
  page.on('pageerror', e => errors.push(e.message))
  await page.goto('/settings/scenario-algorithms?center=simulation')
  const row = page.getByRole('row').filter({ hasText: '旧版无 Horizon 证据' })
  await row.getByRole('button', { name: '查看路径' }).click()
  await expect(page.getByText('此版本未提供 Horizon 证据。', { exact: true })).toBeVisible()
  await expect(page.getByRole('table', { name: '最终市场冲击' })).toBeVisible()
  expect(errors).toEqual([])
  await visual(page, info.outputPath('legacy-scenario.png'))
})

test('真实预览分开说明增量与水平，停用后同定义同有效期可再发布', async ({ page }, info) => {
  const identity = await connect(page)
  await page.goto('/settings/scenario-algorithms?center=simulation')
  await page.getByRole('button', { name: '新建情景', exact: true }).click()
  await page.getByLabel('情景名称', { exact: true }).fill('恢复口径验证 ' + info.project.name)
  await page.getByRole('button', { name: '下一步：设置变化' }).click()
  await page.getByRole('checkbox', { name: new RegExp(identity.factor_name) }).check()
  await page.getByLabel('每期代表什么').selectOption('monthly')
  await page.getByLabel('设置多少期').fill('2')
  await page.getByLabel(`第1期${identity.factor_name}变化`, { exact: true }).fill('-10')
  await page.getByRole('button', { name: '预览冲击路径' }).click()
  await expect(page.getByText('末期增量：归零。', { exact: true })).toBeVisible()
  await expect(page.getByText(/累计市场水平：尚未恢复至起点/)).toBeVisible()
  await page.getByLabel('发布说明', { exact: true }).fill('第一次明确发布')
  await page.getByRole('checkbox', { name: /我已核对单位/ }).check()
  const firstResponse = page.waitForResponse(r => r.request().method() === 'POST' && r.url().endsWith('/published-scenarios/releases'))
  await page.getByRole('button', { name: '确认发布情景', exact: true }).click()
  const first = await firstResponse
  expect(first.status(), await first.text()).toBe(201)
  const original = await first.json()
  const retired = await page.request.post(`${api}/api/published-scenarios/releases/${original.id}/retire`, { data: { note: '测试停止本轮发布' } })
  expect(retired.ok(), await retired.text()).toBeTruthy()
  await page.getByRole('button', { name: '返回设置变化' }).click()
  await page.getByRole('button', { name: '预览冲击路径' }).click()
  await expect(page.getByRole('button', { name: '确认发布情景', exact: true })).toBeVisible()
  await page.getByLabel('发布说明', { exact: true }).fill('同内容第二次明确发布')
  await page.getByRole('checkbox', { name: /我已核对单位/ }).check()
  const secondResponse = page.waitForResponse(r => r.request().method() === 'POST' && r.url().endsWith('/published-scenarios/releases'))
  await page.getByRole('button', { name: '确认发布情景', exact: true }).click()
  const second = await secondResponse
  expect(second.status(), await second.text()).toBe(201)
  const replacement = await second.json()
  expect(replacement.id).not.toBe(original.id)
  expect(replacement.preview_id).toBe(original.preview_id)
  expect(replacement.valid_days).toBe(original.valid_days)
  expect(replacement.note).toBe('同内容第二次明确发布')
  await visual(page, info.outputPath('scenario-republished.png'))
})

test('乱序真实定义响应不覆盖新选择，未应用 JSON 保留并存入新修订', async ({ page }, info) => {
  const identity = await connect(page)
  await page.goto('/settings/scenario-algorithms?center=simulation')
  await page.getByRole('button', { name: '高级计算实验', exact: true }).click()
  const select = page.getByLabel('已保存情景定义')
  await expect(select).toBeVisible()
  // Fetch real definitions, then release their responses in the opposite order.
  let releaseA!: () => void
  const holdA = new Promise<void>(resolve => { releaseA = resolve })
  await page.route(`**/api/scenario-stress/definitions/${identity.definition_a}`, async route => {
    const response = await route.fetch({ url: `${api}/api/scenario-stress/definitions/${identity.definition_a}` })
    await holdA
    await route.fulfill({ response })
  })
  const aRequested = page.waitForRequest(r => r.url().endsWith(`/definitions/${identity.definition_a}`))
  await select.selectOption(identity.definition_a)
  await aRequested
  await select.selectOption(identity.definition_b)
  await expect(page.getByLabel('定义名称')).toHaveValue('并发载入乙')
  const aResponded = page.waitForResponse(r => r.url().endsWith(`/definitions/${identity.definition_a}`))
  releaseA()
  await aResponded
  await expect(page.getByLabel('定义名称')).toHaveValue('并发载入乙')
  await page.getByRole('button', { name: /期限与路径/ }).click()
  const text = '[{"step":1,"shocks":{"growth":-2,"rate":20}}]'
  const editor = page.getByLabel('可选逐期冲击路径 JSON')
  await editor.fill(text)
  const severity = info.project.name === 'mobile-320' ? 2 : info.project.name === 'tablet-768' ? 3 : 4
  await page.getByLabel('情景严重度').fill(String(severity))
  await expect(editor).toHaveValue(text)
  const revisionSaved = page.waitForResponse(r => r.request().method() === 'PUT' && r.url().endsWith(`/definitions/${identity.definition_b}`))
  await page.getByRole('button', { name: '保存新修订', exact: true }).click()
  const revisionResponse = await revisionSaved
  expect(revisionResponse.ok(), await revisionResponse.text()).toBeTruthy()
  await expect(editor).toHaveValue(text)
  await page.getByRole('button', { name: /对象与映射/ }).click()
  await page.getByRole('button', { name: /添加组合/ }).click()
  const selectedPortfolio = await page.getByLabel('组合名称', { exact: true }).inputValue()
  await page.getByRole('button', { name: /约束与阈值/ }).click()
  await page.getByRole('button', { name: /对象与映射/ }).click()
  await expect(page.getByLabel('组合名称', { exact: true })).toHaveValue(selectedPortfolio)
  await page.getByRole('tab', { name: /结果与归因/ }).click()
  await page.getByRole('tab', { name: /情景定义/ }).click()
  await expect(page.getByLabel('组合名称', { exact: true })).toHaveValue(selectedPortfolio)
  await page.getByRole('button', { name: '移除组合', exact: true }).click()
  await page.getByRole('button', { name: /期限与路径/ }).click()
  await expect(editor).toHaveValue(text)
  await expect(page.getByRole('tab')).not.toHaveCount(0)
  expect(await page.locator('[aria-controls]').evaluateAll(elements => elements.filter(element => !document.getElementById(element.getAttribute('aria-controls')!)).map(element => element.getAttribute('aria-controls')))).toEqual([])
  const menu = page.locator('header button[aria-expanded]').first()
  if (await menu.isVisible()) {
    await menu.click()
    await expect(page.locator('#mobile-navigation')).toBeVisible()
    await expect(menu).toHaveAttribute('aria-controls', 'mobile-navigation')
    await menu.click()
    await expect(page.locator('#mobile-navigation')).toHaveCount(0)
  }
  await page.getByRole('button', { name: '解析并应用', exact: true }).click()
  const savedResponse = page.waitForResponse(r => r.request().method() === 'PUT' && r.url().endsWith(`/definitions/${identity.definition_b}`))
  await page.getByRole('button', { name: '保存新修订', exact: true }).click()
  const response = await savedResponse
  expect(response.ok(), await response.text()).toBeTruthy()
  const saved = await response.json()
  expect(saved.scenario.factor_path).toEqual(JSON.parse(text))
  expect(saved.scenario.severity).toBe(severity)
  await visual(page, info.outputPath('saved-json.png'))
})

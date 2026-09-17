import { expect, test, type Page } from '@playwright/test'
import { auditTextContrast } from './helpers/contrast'

const api = 'http://127.0.0.1:8769'
async function connectRealRegimeAPI(page: Page) {
  // Only unrelated application services are unavailable. Regime responses below
  // come from the real isolated Python graph/metrics/persistence implementation.
  await page.route('**/api/**', route => route.fulfill({ status: 404, json: { detail: 'Unrelated service is outside this isolated test.' } }))
  await page.route('**/api/historical-regimes/**', async route => {
    const url = new URL(route.request().url())
    const response = await route.fetch({ url: api + url.pathname + url.search, timeout: 120_000 })
    await route.fulfill({ response })
  })
  const ready = await page.request.get(`${api}/ready`)
  expect(ready.ok()).toBeTruthy()
  return ready.json() as Promise<{ historical_id: string; realtime_id: string; cutoff: string }>
}

test('历史定义的质量检查调用真实计算及保存接口', async ({ page }, info) => {
  const identity = await connectRealRegimeAPI(page)
  await page.goto(`/settings/scenario-algorithms?center=historical&definition=${identity.historical_id}&revision=1`)
  await expect(page.getByLabel('研究名称')).toHaveValue('浏览器验收·历史定义（合成数据）')
  const check = page.getByRole('button', { name: '检查划分质量', exact: true })
  await expect(check).toBeEnabled()
  const responsePromise = page.waitForResponse(response => response.url().endsWith('/reference-quality/preview'))
  await check.click()
  const response = await responsePromise
  expect(response.status(), await response.text()).toBe(200)
  const preview = await response.json()
  expect(preview.report.sample.input).toBe(900)
  expect(preview.report.stability.variants.some((variant: { status: string }) => variant.status === 'completed')).toBeTruthy()
  expect(preview.report.execution.python_fallback).toBe(0)
  const save = page.getByRole('button', { name: '确认保存质量报告', exact: true })
  await expect(save).toBeEnabled()
  const confirmedResponse = page.waitForResponse(response => response.url().endsWith('/reference-quality/confirm'))
  await save.click()
  const confirmed = await confirmedResponse
  expect(confirmed.status(), await confirmed.text()).toBe(200)
  const saved = await confirmed.json()
  expect(saved.immutable).toBe(true)
  const read = await page.request.get(`${api}/api/historical-regimes/reference-quality/reports/${saved.id}`)
  expect(await read.json()).toEqual(saved)
  await expect.poll(() => page.evaluate(auditTextContrast)).toEqual([])
  await page.screenshot({ path: info.outputPath('real-reference-quality.png'), fullPage: true })
})

test('实时模型的校准稳定性和统计区间来自真实后端', async ({ page }, info) => {
  const identity = await connectRealRegimeAPI(page)
  await page.goto(`/settings/scenario-algorithms?center=realtime&definition=${identity.realtime_id}&revision=1`)
  await expect(page.getByLabel('研究名称')).toHaveValue('浏览器验收·实时模型（合成数据）')
  await page.getByLabel('校准截止日', { exact: true }).fill('2020-10-31')
  await page.getByLabel('验证截止日（可选）', { exact: true }).fill('2021-04-30')
  await page.getByLabel('测试截止日（可选）', { exact: true }).fill(identity.cutoff)
  const responsePromise = page.waitForResponse(response => response.url().endsWith('/reliability/preview'), { timeout: 120_000 })
  await page.getByRole('button', { name: '验证识别能力', exact: true }).click()
  const response = await responsePromise
  expect(response.status(), await response.text()).toBe(200)
  const preview = await response.json()
  expect(preview.report.status).toBe('retrospective_only')
  expect(preview.report.calibration.deployment_eligible).toBe(false)
  expect(preview.report.stability.parameter_sensitivity.variants.length).toBeGreaterThan(0)
  expect(['available', 'partial']).toContain(preview.report.confidence_interval.status)
  expect(preview.report.confidence_interval.scope).toBe('test')
  expect(preview.report.execution.python_fallback).toBe(0)
  await expect(page.getByLabel('识别能力验证报告')).toBeVisible()
  const confirmedResponse = page.waitForResponse(response => response.url().endsWith('/reliability/confirm'))
  await page.getByRole('button', { name: '保存验证报告', exact: true }).click()
  const confirmed = await confirmedResponse
  expect(confirmed.status(), await confirmed.text()).toBe(200)
  const saved = await confirmed.json()
  expect(saved.immutable).toBe(true)
  await expect.poll(() => page.evaluate(auditTextContrast)).toEqual([])
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBeTruthy()
  await page.screenshot({ path: info.outputPath('real-recognition-report.png'), fullPage: true })
})

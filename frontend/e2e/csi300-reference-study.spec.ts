import { readFileSync } from 'node:fs'
import path from 'node:path'
import { expect, test } from '@playwright/test'
import { auditTextContrast } from './helpers/contrast'

// Explicit read-only real-data acceptance, never an ordinary offline unit test.
test.skip(!process.env.CSI300_STUDY_ARTIFACTS, 'Run only after the explicit CSI300 research study has been saved.')
test.afterEach(async ({ page }) => { await page.unrouteAll({ behavior: 'wait' }) })

function actualStudy() {
  const root = process.env.CSI300_STUDY_ARTIFACTS!
  return JSON.parse(readFileSync(path.join(root, 'state-evidence-summary.json'), 'utf8'))
}
async function forwardRealApi(page: import('@playwright/test').Page) {
  const api = process.env.CSI300_READBACK_API || 'http://127.0.0.1:8774'
  await page.route('**/api/**', async route => {
    const url = new URL(route.request().url())
    const response = await route.fetch({ url: api + url.pathname + url.search })
    await route.fulfill({ response })
  })
}
async function audit(page: import('@playwright/test').Page) {
  await expect.poll(() => page.evaluate(auditTextContrast)).toEqual([])
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBeTruthy()
}

test('真实沪深300事后参考显示独立区间与 LTCMA 条件估计准备度', async ({ page }, testInfo) => {
  const actual = actualStudy()
  await forwardRealApi(page)
  await page.goto(`/settings/scenario-algorithms?center=historical&definition=${actual.historical_definition.id}&revision=${actual.historical_definition.revision}`)
  await expect(page.getByLabel('研究名称')).toHaveValue(actual.historical_definition.name)
  await page.getByRole('button', { name: '前往校验与预览' }).click()
  await page.getByLabel('V2 截至日').fill('2026-09-03')
  await page.getByText('已保存质量报告', { exact: true }).click()
  await expect(page.getByLabel('已保存质量报告')).toBeEnabled()
  await page.getByLabel('已保存质量报告').selectOption(actual.quality_report_id)
  await page.getByRole('button', { name: '载入质量报告', exact: true }).click()
  const report = page.getByLabel('历史划分质量报告')
  await expect(report).toBeVisible()
  await expect(report).toContainText('LTCMA条件估计：全部状态样本可用')
  await report.getByText('状态支持、持续时间与区间收益', { exact: true }).click()
  const table = report.getByRole('table', { name: '历史状态区间统计' })
  await expect(table.getByRole('row', { name: /牛市/ })).toContainText('6')
  await expect(table.getByRole('row', { name: /震荡市/ })).toContainText('3')
  await expect(table.getByRole('row', { name: /熊市/ })).toContainText('5')
  await expect(table).toContainText('可估计')
  await audit(page)
  await page.screenshot({ path: testInfo.outputPath('csi300-historical-reference-quality.png'), fullPage: true })
})

test('真实沪深300实时模型把最终区间不足判为证据不足而不是失败', async ({ page }, testInfo) => {
  const actual = actualStudy()
  await forwardRealApi(page)
  await page.goto(`/settings/scenario-algorithms?center=realtime&definition=${actual.model.definition_id}&revision=${actual.model.revision}`)
  await expect(page.getByLabel('研究名称')).toHaveValue(actual.model.name)
  await expect(page.getByRole('heading', { name: '建立实时识别', exact: true })).toBeVisible()
  await expect(page.getByLabel('历史参考版本').locator('option:checked')).toContainText('沪深300主趋势牛熊震荡')
  await page.getByText('已保存报告', { exact: true }).click()
  await expect(page.getByLabel('已保存验证报告')).toBeEnabled()
  await page.getByLabel('已保存验证报告').selectOption(actual.report_id)
  await page.getByRole('button', { name: '载入报告', exact: true }).click()
  const report = page.getByLabel('识别能力验证报告')
  await expect(report).toBeVisible()
  await expect(report).toContainText('仅回顾性评分')
  await expect(report).toContainText('不可部署')
  const stateSection = report.getByLabel('状态级验证')
  await expect(stateSection).toContainText('证据不足')
  await expect(stateSection).toContainText('识别验证暂不可用')
  const table = stateSection.getByRole('table', { name: '逐状态验证结果' })
  await expect(table.getByRole('row', { name: /牛市/ })).toContainText('1')
  await expect(table.getByRole('row', { name: /震荡市/ })).toContainText('1')
  await expect(table.getByRole('row', { name: /熊市/ })).toContainText('0')
  await expect(stateSection).toContainText('回退状态：牛市、震荡市、熊市')
  await audit(page)
  await page.screenshot({ path: testInfo.outputPath('csi300-realtime-state-evidence.png'), fullPage: true })
})

import { expect, type Page } from '@playwright/test'

/** Shared browser operations for the only active CMA editor. No API-side form bypass. */
export async function fillLtcmaAsset(page: Page, asset: string, values: {
  role?: string; rationale?: string; annualReturn?: string; volatility?: string; uncertainty?: string
}) {
  if (values.role) {
    await page.getByRole('combobox', { name: `${asset} · 经济用途`, exact: true }).selectOption(values.role)
    await page.getByRole('combobox', { name: `${asset} · 流动性`, exact: true }).selectOption('liquid')
  }
  if (values.rationale) await page.getByLabel(`${asset} · 分类与代理依据`, { exact: true }).fill(values.rationale)
  if (values.annualReturn !== undefined) await page.getByLabel(`${asset} · 预期年收益（%）`, { exact: true }).fill(values.annualReturn)
  if (values.volatility !== undefined) await page.getByLabel(`${asset} · 年化波动（%）`, { exact: true }).fill(values.volatility)
  if (values.uncertainty !== undefined) {
    const input = page.getByLabel(`${asset} · 均值不确定半宽（百分点）`, { exact: false })
    const group = page.getByRole('group', { name: asset, exact: true })
    if (!await input.isVisible()) await group.locator('summary').filter({ hasText: '均值不确定半宽' }).click()
    await input.fill(values.uncertainty)
  }
}

export async function previewCurrentLtcma(page: Page) {
  await page.getByRole('checkbox', { name: /我已核对资产范围/ }).check()
  const pending = page.waitForResponse(response => new URL(response.url()).pathname.endsWith('/cma/preview') && response.request().method() === 'POST')
  await page.getByRole('button', { name: '计算预览', exact: true }).click()
  const response = await pending
  expect(response.status(), await response.text()).toBe(200)
  const value = await response.json()
  expect(value.execution.python_fallback).toBe(0)
  await expect(page.getByRole('table', { name: '收益与风险假设', exact: true })).toBeVisible()
  return value
}

export async function publishCurrentLtcma(page: Page) {
  await page.getByRole('checkbox', { name: /我已阅读结果和限制/ }).check()
  const pending = page.waitForResponse(response => new URL(response.url()).pathname.endsWith('/cma') && response.request().method() === 'POST')
  await page.getByRole('button', { name: '确认保存版本', exact: true }).click()
  const response = await pending
  expect(response.status(), await response.text()).toBe(201)
  const version = await response.json()
  await expect(page).toHaveURL(new RegExp(`/ltcma/${version.id}`))
  await expect(page.getByRole('button', { name: '用于 SAA', exact: true })).toBeVisible()
  return version
}

import { test, expect } from '@playwright/test'
import { execFileSync } from 'node:child_process'
import { fileURLToPath } from 'node:url'

const root = fileURLToPath(new URL('../../', import.meta.url))
const python = process.env.TEST_PYTHON || '/Users/chenjunming/Desktop/myenv_312/bin/python3.12'
// Use the actual backend definitions with an isolated configuration database.
const catalog = JSON.parse(execFileSync(python, ['-c', 'import json,tempfile; from pathlib import Path; from backend.data_sources.store import SourceStore; from backend.data_sources.service import catalog; tmp=tempfile.TemporaryDirectory(); print(json.dumps(catalog(SourceStore(Path(tmp.name))),ensure_ascii=False)); tmp.cleanup()'], { cwd: root, encoding: 'utf8' }))

test('source center exposes import mappings and stays within the viewport', async ({ page }, testInfo) => {
  const errors: string[] = []
  page.on('pageerror', error => errors.push(error.message))
  await page.route('**/api/**', async route => {
    if (new URL(route.request().url()).pathname === '/api/data-sources/catalog') {
      return route.fulfill({ json: catalog })
    }
    return route.fulfill({ status: 404, json: { detail: 'Offline browser fixture' } })
  })
  await page.goto('/settings/source-center')
  await expect(page.getByRole('heading', { name: '数据源与接口映射' })).toBeVisible()
  await page.getByLabel('搜索接口').fill('fund_daily')
  await expect(page.getByRole('button', { name: /ETF 日行情/ })).toBeVisible()
  await page.screenshot({ path: testInfo.outputPath('source-center-overview.png'), fullPage: true })
  await page.getByRole('button', { name: /ETF 日行情/ }).click()
  await expect(page.getByRole('button', { name: '保存接口配置' })).toBeVisible()
  await expect(page.getByLabel('接口 API 名称')).toHaveValue('fund_daily')
  await expect(page.getByLabel('close 转换方式', { exact: true })).toHaveValue('copy')
  await expect(page.getByLabel('return_decimal 来源字段')).toHaveValue('pct_chg')
  const targets = await page.getByLabel('目标标准表', { exact: true }).locator('option').allTextContents()
  expect(targets.some(value => value.includes('market.quote_daily'))).toBeTruthy()
  expect(targets.some(value => value.includes('governance.'))).toBeFalsy()
  expect(targets.some(value => value.includes('mart.'))).toBeFalsy()
  for (const tableId of [
    'master.organization_identifier', 'master.person_identifier',
    'master.instrument_identifier', 'portfolio.external_account_identifier',
  ]) {
    expect(targets.some(value => value.includes(tableId))).toBeFalsy()
  }
  expect(targets.some(value => value.includes('master.organization'))).toBeTruthy()
  expect(targets.some(value => value.includes('index.membership'))).toBeTruthy()
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1)).toBeTruthy()
  await expect(page.getByLabel('接口 API 名称')).not.toBeVisible()
  await page.getByLabel('日行情 字段筛选').selectOption('all')
  await page.getByLabel('日行情 搜索字段').fill('close')
  await expect(page.getByLabel('close 转换方式', { exact: true })).toBeVisible()
  await page.getByLabel('日行情 搜索字段').fill('')
  await page.getByRole('button', { name: '1. 数据与接口' }).click()
  await expect(page.getByLabel('接口 API 名称')).toBeVisible()
  await expect(page.getByLabel('接口 API 名称')).toBeEnabled()
  await expect(page.getByLabel('响应数据格式')).toBeEnabled()
  await page.getByLabel('接口 API 名称').scrollIntoViewIfNeeded()
  await page.screenshot({ path: testInfo.outputPath('source-center.png'), fullPage: false })
  await page.getByRole('button', { name: '添加默认请求参数' }).click()
  await page.getByRole('button', { name: '3. 验证与使用' }).click()
  await page.getByText('真实接口采样（消耗配额）', { exact: true }).click()
  await expect(page.getByRole('button', { name: '确认并采样一次' })).toBeDisabled()
  await page.getByRole('button', { name: '保存接口配置' }).click()
  await expect(page.getByRole('button', { name: '1. 数据与接口' })).toHaveAttribute('aria-current', 'step')
  await expect(page.getByRole('alert')).toContainText('请修正标出的输入')
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1)).toBeTruthy()
  expect(errors).toEqual([])
})

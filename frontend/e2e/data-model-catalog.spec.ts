import { test, expect } from '@playwright/test'
import { execFileSync } from 'node:child_process'
import { fileURLToPath } from 'node:url'

const root = fileURLToPath(new URL('../../', import.meta.url))
const python = process.env.TEST_PYTHON || '/Users/chenjunming/Desktop/myenv_312/bin/python3.12'
const externalCatalog = JSON.parse(execFileSync(python, ['-c', 'import json; from backend.data_model.catalog import get_data_model_catalog; print(json.dumps(get_data_model_catalog(), ensure_ascii=False))'], { cwd: root, encoding: 'utf8' }))

test('external import tables are grouped by business category and searchable by field', async ({ page }) => {
  const errors: string[] = []
  page.on('pageerror', error => errors.push(error.message))
  await page.route('**/api/data-model/catalog*', async route => route.fulfill({ json: externalCatalog }))

  await page.goto('/settings/data-model')
  await expect(page.getByRole('heading', { name: '外部数据导入标准' })).toBeVisible()
  await expect(page.getByRole('button', { name: /产品与参与方主数据/ })).toBeVisible()
  await expect(page.getByRole('button', { name: /行情、净值与估值/ })).toBeVisible()
  await expect(page.getByRole('region', { name: '产品与参与方主数据表' })).toBeVisible()
  await expect(page.getByRole('region', { name: '行情、净值与估值表' })).toBeVisible()

  const search = page.getByLabel(/搜索业务表或字段/)
  await search.fill('adjusted_nav')
  await expect(page.getByText('复权净值 · adjusted_nav')).toBeVisible()
  await expect(page.getByRole('heading', { name: '基金日净值' })).toBeVisible()

  await page.getByRole('button', { name: '高级：查看系统维护字段' }).click()
  const systemRow = page.locator('tr[data-field-kind="system"]').first()
  const sourceRow = page.locator('tr[data-field-kind="source"]').first()
  await expect(systemRow.getByText('系统维护', { exact: true })).toBeVisible()
  expect(await systemRow.evaluate(row => getComputedStyle(row).backgroundColor)).not.toBe(await sourceRow.evaluate(row => getComputedStyle(row).backgroundColor))
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1)).toBeTruthy()
  expect(errors).toEqual([])
})

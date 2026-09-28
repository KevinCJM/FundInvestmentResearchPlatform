import { expect, test } from '@playwright/test'
import { auditTextContrast } from './helpers/contrast'

test('HTTP browser without native randomUUID renders objectives and distinct new-draft links', async ({ page }) => {
  const errors: string[] = []
  page.on('pageerror', error => errors.push(error.message))
  await page.addInitScript(() => {
    Object.defineProperty(crypto, 'randomUUID', { value: undefined, configurable: true, writable: true })
  })
  await page.route('**/api/**', route => {
    const path = new URL(route.request().url()).pathname
    return path === '/api/strategic-allocation/catalog'
      ? route.fulfill({ json: { mandates: [] } })
      : route.fulfill({ status: 404, json: { detail: 'Offline browser compatibility fixture' } })
  })
  await page.goto('/pre-investment/objectives')
  await expect(page.getByRole('heading', { name: '投资目标与约束', exact: true })).toBeVisible()
  const add = page.locator('header').getByRole('link', { name: '添加投资目标与约束', exact: true })
  await expect(add).toBeVisible()
  const first = await add.getAttribute('href')
  expect(first).toMatch(/\/objectives\/new\?fresh=[0-9a-f]{8}$/)
  await page.reload()
  await expect(add).toBeVisible()
  expect(await add.getAttribute('href')).not.toBe(first)
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true)
  expect(await page.evaluate(auditTextContrast)).toEqual([])
  await add.focus()
  await page.keyboard.press('Enter')
  await expect(page).toHaveURL(/\/objectives\/new\?fresh=[0-9a-f]{8}$/)
  await expect(page.locator('#root')).not.toBeEmpty()
  expect(errors).toEqual([])
})

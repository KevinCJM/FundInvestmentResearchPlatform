import { expect, test, type Page } from '@playwright/test'
import { strategicCatalog } from '../src/test/strategicAllocationFixtures'
import { ltcmaCapabilities, ltcmaOptions } from '../src/test/ltcmaFixtures'
import { taaBaseline, taaCatalog, taaPreflight, taaPreview } from '../src/test/tacticalAllocationFixtures'
import { auditTextContrast } from './helpers/contrast'

// These names are saved researcher input, deliberately Chinese in both languages.
const userText = ['长期配置目标', '股债分类', '权益', '债券', '稳健长期组合', '十年人民币假设']
async function untranslated(page: Page) {
  return page.evaluate((names) => {
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT)
    const found: string[] = []
    while (walker.nextNode()) {
      const node = walker.currentNode, parent = node.parentElement
      if (!parent || parent.closest('script,style') || !parent.getClientRects().length) continue
      const value = names.reduce((text, name) => text.replaceAll(name, ''), node.textContent || '').trim()
      if (/[\u3400-\u9fff]/u.test(value)) found.push(value)
    }
    for (const el of document.querySelectorAll<HTMLInputElement>('input[placeholder],textarea[placeholder],[aria-label],[title]')) {
      if (!el.getClientRects().length) continue
      for (const attr of ['placeholder', 'aria-label', 'title']) {
        const value = names.reduce((text, name) => text.replaceAll(name, ''), el.getAttribute(attr) || '')
        if (/[\u3400-\u9fff]/u.test(value)) found.push(`${attr}: ${value}`)
      }
    }
    return [...new Set(found)]
  }, userText)
}

test.setTimeout(120_000)

test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => localStorage.setItem('fund-research.i18n.locale', 'en-US'))
  await page.route('**/api/**', async route => {
    const path = new URL(route.request().url()).pathname
    let value: unknown
    if (path === '/api/strategic-allocation/catalog') value = { ...strategicCatalog, strategic_universes: [], implementations: [] }
    else if (path.endsWith('/cma/capabilities')) value = ltcmaCapabilities
    else if (path.endsWith('/cma/study-options')) value = ltcmaOptions
    else if (path.endsWith('/cma/drafts') || path.endsWith('/cma')) value = { items: [], total: 0 }
    else if (path.endsWith('/tactical-allocation/catalog')) value = taaCatalog
    else if (path.endsWith('/baselines/SAA-1')) value = taaBaseline
    else if (path.endsWith('/preflight')) value = taaPreflight
    else if (path.endsWith('/tactical-allocation/preview')) value = { ...taaPreview, request: route.request().postDataJSON() }
    else if (path.endsWith('/investable-universes')) value = { items: [] }
    else if (path.includes('/pre-investment/') && path.endsWith('/catalog')) value = { today: '2026-09-23', sources: [], products: [], scenarios: [], risk_models: [] }
    else if (path.includes('/pre-investment/') && path.endsWith('/packages')) value = { items: [] }
    else return route.fulfill({ status: 503, json: { detail: { code: 'OFFLINE_TEST', message: 'Offline test: service unavailable.' } } })
    return route.fulfill({ json: value })
  })
})

test('English covers every pre-investment entry and retains saved researcher names', async ({ page }, info) => {
  for (const path of [
    '', '/objectives', '/objectives/new', '/product-pool', '/product-pool?scope=strategic',
    '/product-pool/new?scope=strategic', '/ltcma', '/ltcma/new', '/saa', '/saa/policy',
    '/taa?baseline=SAA-1', '/product-allocation-timing', '/portfolio-synthesis', '/validation', '/approval',
    '/saa/asset-classes', '/saa/auto-classification', '/saa/allocation-lab',
    '/product-allocation-timing/construction', '/product-allocation-timing/timing',
  ]) {
    await page.goto(`/pre-investment${path}`)
    await expect(page.getByRole('combobox', { name: 'Language' })).toHaveValue('en-US')
    await expect.poll(() => untranslated(page)).toEqual([])
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), path).toBeTruthy()
    if (path === '/objectives') await expect(page.getByText('长期配置目标', { exact: true })).toBeVisible()
    if (['', '/product-pool', '/taa?baseline=SAA-1'].includes(path)) await page.screenshot({ path: info.outputPath(`english-${path.includes('taa') ? 'taa' : path ? 'scope' : 'overview'}.png`), fullPage: true })
  }
})

test('TAA language switches preserve numeric input and translate every tab and chart', async ({ page }, info) => {
  await page.goto('/pre-investment/taa?baseline=SAA-1')
  await expect(page.getByText('What allocation are you considering?')).toBeVisible()
  await page.getByLabel('Rebalancing convention').selectOption('daily_target')
  const input = page.getByRole('spinbutton', { name: 'Trend observation window (trading periods)' })
  await input.fill('90')
  await page.getByRole('combobox', { name: 'Language' }).selectOption('zh-CN')
  await expect(page.getByRole('tab', { name: '观点与规则' })).toBeVisible()
  await expect(page.getByRole('spinbutton', { name: '趋势观察窗口（交易期）' })).toHaveValue('90')
  await page.getByRole('combobox', { name: '语言' }).selectOption('en-US')
  await expect(input).toHaveValue('90')
  for (const tab of ['Views and rules', 'Backtest and selection', 'Scenario simulation', 'Versions and audit']) {
    await page.getByRole('tab', { name: tab }).click()
    await expect.poll(() => untranslated(page)).toEqual([])
  }
  await page.getByRole('tab', { name: 'Views and rules' }).click()
  await page.getByRole('button', { name: 'Calculate and compare plans' }).click()
  await expect(page.getByRole('region', { name: 'SAA and tactical plan comparison' })).toBeVisible()
  await expect.poll(() => untranslated(page)).toEqual([])
  await expect.poll(() => page.evaluate(auditTextContrast)).toEqual([])
  await page.screenshot({ path: info.outputPath('english-taa-results.png'), fullPage: true })
})

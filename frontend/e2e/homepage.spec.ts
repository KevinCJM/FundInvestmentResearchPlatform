import { expect, test } from '@playwright/test'
import { homeModules, researchExamples } from '../src/homepage/catalog'

test.beforeEach(async ({ page }) => {
  await page.route('**/api/**', route => {
    const url = new URL(route.request().url())
    if (url.pathname === '/api/i18n/settings') return route.fulfill({ json: {
      revision: 0, preferences_revision: 0, default_locale: 'zh-CN', catalog_version: 'test', overrides: {},
      locales: [{ id: 'zh-CN', label: '中文' }, { id: 'en-US', label: 'English' }],
    } })
    if (url.pathname === '/api/i18n/bundle') return route.fulfill({ json: {
      locale: url.searchParams.get('locale'), revision: 0, resources: { system: {}, business: {} },
    } })
    // Deliberately unavailable PIT settings must remain visibly unknown in workspaces.
    return route.fulfill({ status: 503, json: { detail: 'Isolated navigation fixture' } })
  })
})

test('homepage imagery, destinations and responsive layout', async ({ page }, info) => {
  const errors: string[] = [], writes: string[] = []
  page.on('pageerror', error => errors.push(error.message))
  page.on('console', message => { if (message.type() === 'error' && !message.text().includes('503 (Service Unavailable)')) errors.push(message.text()) })
  page.on('request', request => { if (!['GET', 'HEAD'].includes(request.method())) writes.push(request.url()) })
  await page.goto('/')
  await expect(page).toHaveTitle('基金量化投研平台')
  await expect(page.getByRole('heading', { level: 1 })).toContainText('更好的投资机会')
  await expect(page.locator('header')).toHaveCount(1)
  await expect(page.getByTestId('pit-badge')).toHaveCount(0)
  await page.locator('.home-footer').scrollIntoViewIfNeeded()
  await expect.poll(() => page.locator('.quant-homepage img').evaluateAll(images => images.every(image => (image as HTMLImageElement).complete && (image as HTMLImageElement).naturalWidth > 0))).toBe(true)
  expect(await page.locator('.home-workflow-card').count()).toBe(5)
  expect(await page.locator('.home-core-card').count()).toBe(4)
  const links = await page.locator('.quant-homepage a').evaluateAll(items => items.map(item => item.getAttribute('href')))
  const allowed = new Set(['/', '#home-content', '/settings/research-data-lab', ...homeModules.map(item => item.path), ...researchExamples.map(item => item.path)])
  expect(links.every(link => allowed.has(link!))).toBe(true)
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true)
  await page.evaluate(() => scrollTo(0, 0))
  await page.screenshot({ path: info.outputPath('homepage.png'), fullPage: true })
  await page.getByRole('link', { name: '开始研究', exact: true }).click()
  await expect(page).toHaveURL(/\/product-research$/)
  await expect(page.getByTestId('quant-homepage')).toHaveCount(0)
  await expect(page.getByTestId('pit-badge')).toContainText('PIT 口径未知')
  await page.goBack()
  await page.getByRole('tab', { name: '最近打开' }).click()
  await expect(page.getByRole('tabpanel').getByRole('link', { name: '产品研究' })).toHaveAttribute('href', '/product-research')
  await page.getByRole('button', { name: '清空访问记录' }).click()
  await expect(page.getByRole('tabpanel')).toContainText('还没有访问记录')
  expect(errors).toEqual([])
  expect(writes).toEqual([])
})

test('native modal focus, search, tour and examples are interactive', async ({ page }, info) => {
  await page.goto('/')
  const trigger = page.getByRole('button', { name: '搜索平台功能', exact: true })
  await trigger.click()
  const dialog = page.getByRole('dialog'), search = page.getByRole('searchbox')
  await expect(search).toBeFocused()
  await search.fill('指标')
  await expect(dialog.getByRole('link')).toHaveCount(1)
  await expect(dialog.getByRole('link')).toHaveAttribute('href', '/settings/indicators-models')
  for (let i = 0; i < 5; i++) { await page.keyboard.press('Tab'); expect(await dialog.evaluate(element => element.contains(document.activeElement))).toBe(true) }
  await search.fill('no-such-feature')
  await expect(dialog.getByRole('status')).toContainText('没有找到匹配内容')
  await page.screenshot({ path: info.outputPath('search-empty.png') })
  await page.keyboard.press('Escape')
  await expect(dialog).toHaveCount(0)
  await expect(trigger).toBeFocused()
  await page.keyboard.press('Control+k')
  await expect(search).toBeVisible()
  await dialog.getByRole('button', { name: '关闭', exact: true }).click()
  await page.getByRole('button', { name: '查看平台导览' }).click()
  await expect(dialog.getByRole('button', { name: '上一步' })).toBeDisabled()
  await dialog.getByRole('button', { name: '下一步' }).click()
  await expect(dialog).toContainText('2 / 3')
  await dialog.getByRole('button', { name: '上一步' }).click()
  await dialog.getByRole('button', { name: '下一步' }).click()
  await dialog.getByRole('button', { name: '下一步' }).click()
  await dialog.getByRole('button', { name: '开始探索' }).click()
  await expect(dialog).toHaveCount(0)
  await page.getByRole('button', { name: '沪深 300 历史情景识别 示例' }).click()
  await expect(dialog).toContainText('不是已完成的研究结果')
  await expect(dialog.getByRole('link')).toHaveAttribute('href', '/settings/scenario-algorithms/workbench')
  await page.keyboard.press('Escape')
  await page.getByRole('button', { name: '关于平台', exact: true }).click()
  await expect(dialog).toContainText('不承诺投资收益')
  await page.keyboard.press('Escape')
  await page.getByRole('button', { name: '使用帮助', exact: true }).click()
  await expect(dialog).toContainText('不会删除任何研究数据')
})

test('shared language, navigation menus and reduced motion', async ({ page }, info) => {
  await page.emulateMedia({ reducedMotion: 'reduce' })
  await page.goto('/')
  if ((page.viewportSize()?.width ?? 0) < 1250) {
    const menu = page.getByRole('button', { name: '菜单', exact: true })
    await menu.click()
    await expect(page.getByRole('navigation', { name: '移动端主导航' }).getByRole('link')).toHaveCount(10)
    await page.keyboard.press('Escape'); await expect(menu).toBeFocused()
  } else {
    await page.getByRole('button', { name: '更多', exact: true }).click()
    await expect(page.getByRole('link', { name: '基金会计', exact: true })).toHaveAttribute('href', '/fund-accounting')
    await page.keyboard.press('Escape')
  }
  await page.getByRole('combobox').selectOption('en-US')
  await expect(page.getByRole('link', { name: 'Start research', exact: true })).toBeVisible()
  const originalViewport = page.viewportSize()!
  if (info.project.name === 'desktop-1440') {
    // Long English navigation must also fit immediately above the mobile breakpoint.
    for (const width of [1251, 1366]) {
      await page.setViewportSize({ width, height: originalViewport.height })
      expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true)
    }
    await page.setViewportSize(originalViewport)
  }
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true)
  await page.screenshot({ path: info.outputPath('homepage-en.png'), fullPage: true })
  await page.getByRole('link', { name: 'Start research', exact: true }).click()
  await expect(page.getByRole('combobox')).toHaveValue('en-US')
  await page.goBack(); await page.reload()
  await expect(page.getByRole('link', { name: 'Start research', exact: true })).toBeVisible()
})

test('reference-size visual evidence', async ({ page }, info) => {
  test.skip(info.project.name !== 'desktop-1440', 'One shared reference-size capture')
  // Reference is 2994x4198 image pixels. Compare at 1497 CSS px and normalize scale in QA.
  await page.setViewportSize({ width: 1497, height: 2099 })
  await page.goto('/')
  await page.locator('.home-footer').scrollIntoViewIfNeeded()
  await expect.poll(() => page.locator('.quant-homepage img').evaluateAll(images => images.every(image => (image as HTMLImageElement).complete))).toBe(true)
  await page.evaluate(() => scrollTo(0, 0))
  await page.screenshot({ path: info.outputPath('homepage-reference-size.png'), fullPage: true })
  await page.locator('.home-hero').screenshot({ path: info.outputPath('hero.png') })
  await page.locator('.home-data-panel').screenshot({ path: info.outputPath('data-panel.png') })
})

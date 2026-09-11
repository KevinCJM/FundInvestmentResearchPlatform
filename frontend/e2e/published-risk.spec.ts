import { test, expect } from '@playwright/test'
import { impactFixture, riskCatalogFixture, riskFactor, riskPreviewFixture, riskReleaseFixture, riskRunFixture, scenarioPreviewFixture, scenarioReleaseFixture } from '../src/test/publishedRiskFixtures'

test('研究发布、三入口情景与产品应用在各屏幕保持完整闭环', async ({ page }, testInfo) => {
  const errors: string[] = []
  const posts: Array<{ path: string; body: any }> = []
  const reads: string[] = []
  const riskRun = structuredClone(riskRunFixture)
  let riskPreview = structuredClone(riskPreviewFixture)
  let riskPublished = false
  let scenarioPublished = false
  let preview = structuredClone(scenarioPreviewFixture)
  page.on('pageerror', error => errors.push(error.message))
  await page.route('**/api/**', async route => {
    const request = route.request(), path = new URL(request.url()).pathname
    const post = request.method() === 'POST', body = post ? request.postDataJSON() : undefined
    if (post) posts.push({ path, body }); else reads.push(path)
    let value: unknown
    if (path === '/api/risk-models/catalog' || path === '/api/scenario-transmission/catalog') value = riskCatalogFixture
    else if (path === '/api/risk-models/previews') { riskPreview = { ...riskPreview, name: body.name, model: body }; value = riskPreview }
    else if (path === `/api/risk-models/runs/${riskRun.id}`) value = riskRun
    else if (path === '/api/risk-models/releases') { if (post) riskPublished = true; value = post ? riskReleaseFixture : { items: riskPublished ? [riskReleaseFixture] : [] } }
    else if (path === '/api/scenario-transmission/releases') value = { items: [] }
    else if (path === '/api/published-scenarios/previews') { preview = { ...preview, name: body.name, definition: body }; value = preview }
    else if (path === '/api/published-scenarios/releases') { if (post) scenarioPublished = true; value = post ? scenarioReleaseFixture : { items: scenarioPublished ? [scenarioReleaseFixture] : [] } }
    else if (path === '/api/published-scenarios/portfolios') value = { items: [{ id: 'run-a', name: '原配置', as_of: '2026-08-31', created_at: '2026-09-01' }, { id: 'run-b', name: '调整配置', as_of: '2026-08-31', created_at: '2026-09-01' }] }
    else if (path === '/api/published-scenarios/impacts') value = !post ? { items: [impactFixture] } : body.target.kind === 'portfolio_run' ? { ...impactFixture, id: `impact-${body.target.portfolio_run_id}`, request: body, target: { kind: 'portfolio_run', name: body.target.portfolio_run_id === 'run-a' ? '原配置' : '调整配置', holdings_date: '2026-08-31' } } : impactFixture
    else if (path === `/api/published-scenarios/impacts/${impactFixture.id}`) value = impactFixture
    else if (path === '/api/instruments/products/000001.OF') value = { product_id: '000001.OF', name: '测试基金甲', base_info: { ts_code: '000001.OF' }, metrics: {}, timeseries: [] }
    else return route.fulfill({ status: 404, json: { detail: 'Explicit offline browser fixture; no real data writes.' } })
    return route.fulfill({ json: value })
  })
  await page.goto('/settings/risk-models?product_key=fund%3A000001.OF&product_name=测试基金甲')
  await expect(page.getByRole('heading', { name: '风险模型中心', exact: true })).toBeVisible()
  await page.getByRole('textbox', { name: '研究名称', exact: true }).fill('基金敏感度验收')
  await page.getByRole('checkbox', { name: new RegExp(riskFactor.name) }).check()
  await page.getByRole('button', { name: '计算敏感度' }).click()
  await expect(page.getByLabel('敏感性研究结果')).toContainText('0.7%')
  await expect(page.getByText('当前预览 · 未保存')).toBeVisible()
  expect(posts.filter(item => item.path === '/api/risk-models/releases')).toHaveLength(0)
  await expect(page.getByRole('button', { name: '确认并发布' })).toBeEnabled()
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBeTruthy()
  await page.screenshot({ path: testInfo.outputPath('risk-research.png'), fullPage: true })
  await page.getByRole('button', { name: '确认并发布' }).click()
  await page.getByRole('checkbox', { name: /我已查看验证及数据来源/ }).check()
  await page.getByRole('button', { name: '确认发布成果', exact: true }).click()
  await expect(page.getByRole('button', { name: '已发布', exact: true })).toBeDisabled()
  expect(posts.filter(item => item.path === '/api/risk-models/releases')).toHaveLength(1)
  expect(posts.find(item => item.path === '/api/risk-models/releases')?.body).toEqual(expect.objectContaining({ preview_hash: riskPreviewFixture.preview_hash, definition: expect.objectContaining({ name: '基金敏感度验收' }) }))

  await page.goto('/settings/scenario-algorithms?center=simulation')
  await expect(page.getByRole('heading', { name: '情景模拟与压测', exact: true })).toBeVisible()
  expect(reads.some(path => path.startsWith('/api/historical-regimes'))).toBeFalsy()
  await page.getByRole('button', { name: '新建情景', exact: true }).click()
  await page.getByRole('textbox', { name: '情景名称' }).fill('股市压力验收')
  await expect(page.getByRole('radio')).toHaveCount(3)
  await page.getByRole('button', { name: '下一步：设置变化' }).click()
  await page.getByRole('checkbox', { name: new RegExp(riskFactor.name) }).check()
  const shock = page.getByRole('spinbutton', { name: `第1期${riskFactor.name}变化` })
  await shock.fill(''); await shock.pressSequentially('-10.5')
  await expect(shock).toHaveValue('-10.5')
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBeTruthy()
  await page.getByRole('button', { name: '预览冲击路径', exact: true }).click()
  await expect(page.getByText('股市压力验收 · 市场风险因子路径')).toBeVisible()
  expect(posts.find(item => item.path === '/api/published-scenarios/previews')?.body.rows).toEqual([[-10.5]])
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBeTruthy()
  await page.screenshot({ path: testInfo.outputPath('scenario-preview.png'), fullPage: true })
  await page.getByRole('checkbox', { name: /我已核对单位/ }).check()
  expect(posts.filter(item => item.path === '/api/published-scenarios/releases')).toHaveLength(0)
  await page.getByRole('button', { name: '确认发布情景' }).click()
  expect(posts.find(item => item.path === '/api/published-scenarios/releases')?.body).toEqual(expect.objectContaining({ preview_hash: scenarioPreviewFixture.preview_hash, definition: expect.objectContaining({ name: '股市压力验收' }) }))
  await page.getByRole('link', { name: '应用到产品或组合', exact: true }).click()
  await page.getByRole('combobox', { name: '选择产品', exact: true }).selectOption('fund:000001.OF')
  await expect(page.getByRole('combobox', { name: '选择已发布情景' })).toHaveValue(scenarioReleaseFixture.id)
  await page.getByRole('checkbox', { name: /未设置冲击的已建模因子保持不变/ }).check()
  await page.getByRole('button', { name: '计算情景影响', exact: true }).click()
  await expect(page.getByLabel('本次压测结果')).toContainText('-7.00%')
  expect(posts.filter(item => item.path.endsWith('/runs'))).toHaveLength(0)
  expect(posts.find(item => item.path === '/api/published-scenarios/impacts')?.body.target).toEqual({ kind: 'product', product_key: 'fund:000001.OF' })
  const layout = await page.evaluate(() => ({ width: document.documentElement.scrollWidth, viewport: innerWidth,
    overflow: [...document.querySelectorAll('section,div,a,p,table')].filter(element => { const box = element.getBoundingClientRect(); return box.width > 0 && box.right > innerWidth + 1 }).slice(-12).map(element => element.outerHTML.slice(0, 250)) }))
  expect(layout.width, JSON.stringify(layout.overflow)).toBeLessThanOrEqual(layout.viewport + 1)
  await page.getByLabel('本次压测结果').scrollIntoViewIfNeeded()
  await page.screenshot({ path: testInfo.outputPath('published-impact.png'), fullPage: true })
  await expect(page.getByText(/本次计算结果未写入磁盘/)).toBeVisible()
  await page.goto('/settings/risk-models')
  await page.getByRole('button', { name: '查看成果', exact: true }).click()
  const productLink = page.getByRole('link', { name: '到测试基金甲查看', exact: true })
  await expect(productLink).toHaveAttribute('href', new RegExp(`exposure_release_id=${riskReleaseFixture.id}`))
  await productLink.click()
  await expect(page.getByRole('tab', { name: '风险与压测', exact: true })).toHaveAttribute('aria-selected', 'true')
  await expect(page.getByLabel('敏感性研究结果')).toContainText('0.7%')
  await expect(page.getByRole('combobox', { name: '选择已发布的敏感度成果' })).toHaveValue(riskReleaseFixture.id)
  await expect(page.getByLabel('分析样本')).toHaveCount(0)
  expect(posts.filter(item => item.path.endsWith('/runs'))).toHaveLength(0)
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBeTruthy()
  await page.screenshot({ path: testInfo.outputPath('product-published-risk.png'), fullPage: true })
  await page.goto('/post-investment/scenarios')
  await page.getByRole('link', { name: /使用已发布情景做压测/ }).click()
  await page.getByRole('combobox', { name: /选择已保存的产品组合快照/ }).selectOption('run-a')
  await page.getByRole('checkbox', { name: '对比另一个已保存组合' }).check()
  await page.getByRole('combobox', { name: /选择对照组合快照/ }).selectOption('run-b')
  await page.getByRole('combobox', { name: '选择已发布情景' }).selectOption(scenarioReleaseFixture.id)
  await page.getByRole('checkbox', { name: /未设置冲击的已建模因子保持不变/ }).check()
  await page.getByRole('button', { name: '计算并对比' }).click()
  await expect(page.getByLabel('组合情景对比结果')).toContainText('原配置')
  await expect(page.getByLabel('组合情景对比结果')).toContainText('调整配置')
  expect(posts.filter(item => item.path.endsWith('/runs'))).toHaveLength(0)
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBeTruthy()
  await page.screenshot({ path: testInfo.outputPath('portfolio-comparison.png'), fullPage: true })
  expect(errors).toEqual([])
})

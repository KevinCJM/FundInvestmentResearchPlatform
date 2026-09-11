/** Explicit user-authorized acceptance: persists a real research suite via the application UI/API.
 * Default test runs skip it. Run with FACTOR_REAL_DATA=1 and playwright.factor-live.config.ts.
 */
import { test, expect } from '@playwright/test'

test.skip(process.env.FACTOR_REAL_DATA !== '1', 'Real workspace mutation requires explicit acceptance invocation.')

test('create the real ETF suite, run it, publish research evidence and import candidates', async ({ page, request }, testInfo) => {
  const errors: string[] = []
  page.on('pageerror', error => errors.push(error.message))
  const health = await (await request.get('/api/health')).json()
  expect(health.numba_warmup.factor_research.complete).toBe(true)
  const catalog = await (await request.get('/api/factor-research/catalog')).json()
  expect(catalog.ready).toBe(true)
  const definitions = [
    { builtin: 'factor-momentum-126-21', name: 'ETF 中期动量（126日·跳过21日）', weight: .5 },
    { builtin: 'factor-low-volatility-63', name: 'ETF 低波动（63日）', weight: .3 },
    { builtin: 'factor-drawdown-126', name: 'ETF 回撤控制（126日）', weight: .2 },
  ]
  const chosen: any[] = []
  await page.goto('/settings/factor-research')
  await expect(page.getByRole('tab', { name: '因子库', exact: true })).toBeEnabled()
  await page.getByRole('tab', { name: '因子库', exact: true }).click()
  for (const definition of definitions) {
    const existing = catalog.factors.find((item: any) => item.name === definition.name && !item.read_only)
    if (existing) { chosen.push(existing); continue }
    const original = catalog.factors.find((item: any) => item.id === definition.builtin)
    const article = page.locator('article').filter({ has: page.getByRole('heading', { name: original.name, exact: true }) })
    await article.getByRole('button', { name: '复制构建' }).click()
    await page.getByLabel('因子名称', { exact: true }).fill(definition.name)
    const response = page.waitForResponse(value => value.url().endsWith('/api/factor-research/factors') && value.request().method() === 'POST')
    await page.getByRole('button', { name: '保存因子', exact: true }).click()
    const saved = await response
    expect(saved.status(), await saved.text()).toBe(201)
    chosen.push(await saved.json())
    await expect(page.getByRole('button', { name: '保存因子', exact: true })).toBeEnabled()
  }
  const suiteName = 'ETF 趋势与风险三因子 · ' + catalog.snapshot.latest_date
  const studies = await (await request.get('/api/factor-research/studies')).json()
  const existingStudy = studies.items.find((item: any) => item.name === suiteName)
  await page.getByRole('tab', { name: '研究工作台', exact: true }).click()
  if (existingStudy) await page.getByLabel('选择研究方案').selectOption(existingStudy.id)
  await page.getByLabel('研究方案名称', { exact: true }).fill(suiteName)
  const checkboxes = page.getByRole('checkbox')
  for (let i = 0; i < await checkboxes.count(); i++) await checkboxes.nth(i).uncheck()
  for (let i = 0; i < chosen.length; i++) {
    await page.getByRole('checkbox', { name: chosen[i].name + 'v' + chosen[i].revision, exact: true }).check()
    await page.getByLabel(chosen[i].name + '权重', { exact: true }).fill(String(definitions[i].weight))
  }
  const runResponse = page.waitForResponse(value => /\/api\/factor-research\/studies\/[^/]+\/runs$/.test(value.url()) && value.request().method() === 'POST', { timeout: 120_000 })
  await page.getByRole('button', { name: '保存并运行检验', exact: true }).click()
  const calculated = await runResponse
  expect(calculated.status(), await calculated.text()).toBe(201)
  const run = await calculated.json()
  expect(run.execution.python_fallback).toBe(0)
  expect(run.execution.request_time_compilation).toBe(0)
  expect(run.latest_scores.filter((row: any) => row.status === 'ranked').length).toBeGreaterThanOrEqual(8)
  expect(run.summaries.out_of_sample.factors[3].rank_ic.observations).toBeGreaterThanOrEqual(12)
  expect(run.summaries.out_of_sample.performance.total_return).not.toBeNull()
  await expect(page.getByLabel('因子检验结果')).toBeVisible()
  await page.getByRole('heading', { name: /检验结果/ }).scrollIntoViewIfNeeded()
  await page.screenshot({ path: testInfo.outputPath('real-factor-results.png'), fullPage: false })
  await page.getByRole('button', { name: '发布研究版', exact: true }).click()
  await expect(page.getByRole('button', { name: '确认发布研究版' })).toBeEnabled()
  await page.getByLabel('研究结论与适用范围').fill('固定12只国内股票ETF；动量/低波动/回撤权重50%/30%/20%，未经样本外调参。采用公告日滞后、净值口径、月度Top4及单边5bp费用，仅供研究与候选审核；保留样本选择、历史修订及成交差异限制。')
  const publishedResponse = page.waitForResponse(value => value.url().endsWith('/api/factor-research/releases') && value.request().method() === 'POST')
  await page.getByRole('button', { name: '确认发布研究版' }).click()
  const publication = await publishedResponse
  expect(publication.status(), await publication.text()).toBe(201)
  const release = await publication.json()
  await expect(page.getByLabel('选择因子发布')).toHaveValue(release.id)
  const pools = await (await request.get('/api/product-pools')).json()
  const poolName = 'ETF 三因子研究候选池 · ' + catalog.snapshot.latest_date
  const existingPool = pools.items.find((item: any) => item.name === poolName)
  if (existingPool) await page.getByLabel('导入产品池', { exact: true }).selectOption(existingPool.id)
  else await page.getByLabel('新候选池名称', { exact: true }).fill(poolName)
  const attachedResponse = page.waitForResponse(value => /\/api\/product-pools\/[^/]+\/evaluation-plans$/.test(value.url()) && value.request().method() === 'POST')
  await page.getByRole('button', { name: '导入待审核候选' }).click()
  const attachment = await attachedResponse
  expect(attachment.status(), await attachment.text()).toBe(200)
  const pool = await attachment.json()
  const added = pool.members.filter((item: any) => item.evidences.some((evidence: any) => evidence.plan_id === release.id))
  expect(added).toHaveLength(4)
  expect(added.every((item: any) => item.research_status === 'pending')).toBe(true)

  await page.goto('/product-research')
  await page.getByText('因子研究证据', { exact: true }).click()
  await page.getByLabel('引用因子发布').selectOption(release.id)
  await expect(page.getByLabel('投研因子证据')).toBeVisible()
  await page.getByLabel('研究对象 / 组合版本 ID').fill(run.study_id)
  await page.getByLabel('使用依据', { exact: true }).fill('本三因子研究方案的产品筛选证据；候选产品继续接受产品池审核。')
  await page.getByRole('button', { name: '登记本环节引用' }).click()
  await expect(page.getByText('已登记发布版本及来源运行。')).toBeVisible()

  const attributionResponse = await request.post('/api/factor-research/attributions', { timeout: 90_000, data: {
    name: '公募基金收益风格研究 · ' + catalog.snapshot.latest_date, product_kind: 'fund',
    targets: ['000001.OF', '110022.OF', '260108.OF'], model: 'rbsa',
    indices: ['000300.SH', '000905.SH', '000852.SH'], market: 'CN', currency: 'CNY',
    start_date: '2020-01-01', end_date: catalog.snapshot.latest_date, oos_date: '2024-01-01',
  } })
  expect(attributionResponse.status(), await attributionResponse.text()).toBe(201)
  const attribution = await attributionResponse.json()
  expect(attribution.results.some((item: any) => item.status === 'ok')).toBe(true)
  const result = { run_id: run.id, study_id: run.study_id, release_id: release.id, pool_id: pool.id,
    data_as_of: run.as_of, input_checksum: run.input_checksum, factor_definitions: chosen,
    in_sample: run.summaries.in_sample, out_of_sample: run.summaries.out_of_sample,
    latest_candidates: added.map((item: any) => ({ code: item.code, name: item.name, status: item.research_status })),
    attribution_id: attribution.id, attribution: attribution.results, warnings: run.warnings }
  await testInfo.attach('real-factor-evidence', { body: JSON.stringify(result, null, 2), contentType: 'application/json' })
  console.log('FACTOR_REAL_RESULT ' + JSON.stringify(result))
  expect(errors).toEqual([])
})

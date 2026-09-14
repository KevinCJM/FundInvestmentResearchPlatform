import { test, expect, type Page, type APIRequestContext } from '@playwright/test'
import { auditTextContrast } from './helpers/contrast'

const api = 'http://127.0.0.1:8118/api/strategic-allocation'

async function realApi(page: Page) {
  const errors: string[] = []
  page.on('pageerror', error => errors.push(error.message))
  await page.route('**/api/**', async route => {
    const url = new URL(route.request().url())
    const response = await route.fetch({ url: `http://127.0.0.1:8118${url.pathname}${url.search}`, timeout: 60000 })
    await route.fulfill({ response })
  })
  return errors
}

async function createMandate(request: APIRequestContext, name: string) {
  const day = new Date().toISOString().slice(0, 10)
  const review = new Date(Date.now() + 180 * 86400000).toISOString().slice(0, 10)
  const study = { definition: { name, as_of: day, review_date: review, currency: 'CNY', horizon_years: 10,
    target_return: 0, max_volatility: .3, max_tracking_error: .04,
    boundary_reason: '浏览器离线验收的明确资金与风险边界，不是投资建议。' } }
  const preview = await request.post(`${api}/mandates/preview`, { data: study })
  expect(preview.status()).toBe(200)
  const saved = await request.post(`${api}/mandates/confirm`, { data: {
    request: study, preview_hash: (await preview.json()).preview_hash, acknowledge_limits: true,
  } })
  expect(saved.status()).toBe(201)
  return saved.json()
}

async function verifyLayout(page: Page) {
  await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth - innerWidth)).toBeLessThanOrEqual(1)
  await expect.poll(() => page.evaluate(auditTextContrast), { timeout: 5000 }).toEqual([])
}

test('strategic-first journey: edit, preview, confirm, real SAA, visible implementation gap', async ({ page, request }, info) => {
  const errors = await realApi(page)
  const mandate = await createMandate(request, `无产品战略研究-${info.project.name}`)
  const savedScopes: any[] = []
  page.on('response', async response => {
    if (response.url().endsWith('/universes/confirm') && response.status() === 201) savedScopes.push(await response.json())
  })
  await page.goto(`/pre-investment/product-pool?scope=strategic&mandate=${encodeURIComponent(mandate.id)}`)
  const scope = page.getByRole('region', { name: '独立战略范围' })
  await expect(scope.getByRole('heading', { name: '先确定需要的战略资产' })).toBeVisible()
  await expect(scope.getByRole('button', { name: '确认保存战略范围', exact: true })).toBeDisabled()
  await scope.getByLabel('战略范围名称', { exact: true }).fill(`不依赖产品的战略范围-${info.project.name}`)
  await scope.getByLabel('战略范围来源', { exact: true }).fill('研究员明确经济风险；浏览器隔离验收，无产品伪造。')
  for (const [index, id, name, role] of [[1, 'growth', '经济增长', 'growth'], [2, 'reserve', '现金储备', 'liquidity']] as const) {
    await scope.getByRole('button', { name: '增加战略资产', exact: true }).click()
    await scope.getByLabel(`资产${index}稳定ID`, { exact: false }).fill(id)
    await scope.getByLabel(`资产${index}展示名称`, { exact: true }).fill(name)
    await scope.getByRole('combobox', { name: `资产${index}经济角色`, exact: true }).selectOption(role)
    await scope.getByLabel(`资产${index}定义理由`, { exact: true }).fill(`${name}明确经济角色与长期用途`)
    await scope.getByLabel(`资产${index}来源`, { exact: true }).fill('离线研究定义')
  }
  await scope.getByRole('button', { name: '预览战略范围', exact: true }).click()
  await expect(scope.getByText(/定义已通过校验/)).toBeVisible()
  expect(savedScopes).toHaveLength(0)
  await verifyLayout(page)
  await page.screenshot({ path: info.outputPath('scope-preview.png'), fullPage: true })
  await scope.getByRole('button', { name: '确认保存战略范围', exact: true }).click()
  await expect(scope.getByText(/只读战略范围/)).toBeVisible()
  await expect(scope.getByLabel('战略范围名称', { exact: true })).toBeDisabled()
  const journey = page.getByRole('navigation', { name: '配置研究流程' })
  await expect(journey.getByRole('link', { name: '2. 产品映射', exact: true })).toBeVisible()
  await expect(journey.getByRole('link', { name: '4. 战术研究 TAA', exact: true })).toHaveCount(0)
  await journey.getByRole('link', { name: '3. 长期配置 SAA', exact: true }).click()
  await expect(page.getByRole('heading', { name: '长期政策配置', exact: true })).toBeVisible()
  await page.getByRole('combobox', { name: '投资目标版本', exact: true }).selectOption(mandate.id)
  await expect(page.getByText(/尚有实施缺口/)).toBeVisible()
  await page.getByRole('button', { name: '填写长期假设', exact: true }).click()
  await page.getByLabel('预测来源与主要假设', { exact: false }).fill('明确人工前瞻输入，未使用任何虚构代理净值。')
  for (const [asset, mean, volatility, uncertainty] of [['growth', '7', '18', '2'], ['reserve', '2', '1', '.1']]) {
    await page.getByLabel(`${asset}预期年收益（%）`, { exact: true }).fill(mean)
    await page.getByLabel(`${asset}年化波动（%）`, { exact: false }).fill(volatility)
    await page.getByLabel(`${asset}均值不确定半宽（百分点）`, { exact: false }).fill(uncertainty)
  }
  // This is a manual risk assumption, not an inferred product proxy.
  const correlation = page.getByLabel(/growth.*reserve.*相关/)
  if (await correlation.count()) await correlation.first().fill('0')
  else await page.getByLabel(/相关.*growth.*reserve/).first().fill('0')
  await page.getByRole('checkbox', { name: /我已确认.*同币种/ }).check()
  await page.getByRole('button', { name: '验证长期假设', exact: true }).click()
  await expect(page.getByText(/资产轴和风险矩阵已通过校验/)).toBeVisible()
  await page.getByRole('button', { name: '确认保存假设版本', exact: true }).click()
  await page.getByRole('button', { name: '比较符合目标的政策候选', exact: true }).click()
  const table = page.getByRole('table', { name: '长期政策候选比较' })
  await expect(table).toBeVisible()
  await verifyLayout(page)
  await table.getByRole('button', { name: '复核此候选', exact: true }).first().click()
  await page.getByLabel('采纳理由与复核关注点', { exact: false }).fill('先保留战略需求，再补实施映射；没有产品不等于不存在该风险。')
  await page.getByRole('button', { name: '确认采用此长期政策', exact: true }).click()
  await expect(page.getByRole('button', { name: '长期政策已确认', exact: true })).toBeVisible()
  await expect(page.getByRole('button', { name: /进入 TAA，研究是否需要偏离/ })).toBeDisabled()
  await expect(page.getByText(/已保存纯前瞻政策/)).toBeVisible()
  await verifyLayout(page)
  await page.screenshot({ path: info.outputPath('saa-implementation-gap.png'), fullPage: true })
  expect(errors).toEqual([])
})

for (const method of ['black_litterman', 'scenario_mixture'] as const) {
  test(`real ${method}: human inputs → frozen effective CMA → risk budget → SAA → TAA`, async ({ page, request }, info) => {
    const errors = await realApi(page)
    const mandate = await createMandate(request, `${method}-${info.project.name}`)
    await page.goto(`/pre-investment/saa/policy?mandate=${mandate.id}&alloc=${encodeURIComponent('浏览器离线股债')}`)
    await page.getByRole('button', { name: '填写长期假设', exact: true }).click()
    await page.getByRole('combobox', { name: '预期生成方法', exact: true }).selectOption(method)
    await page.getByRole('textbox', { name: '模型与风险依据', exact: true }).fill('离线验收明确提供的模型假设；不是下载的机构预测。')
    await page.getByLabel('预测来源与主要假设', { exact: true }).fill('同日同币种的隔离研究输入；不作为真实投资建议。')
    for (const [asset, role] of [['股票', 'growth'], ['债券', 'rates']]) {
      await page.getByRole('combobox', { name: `${asset}经济角色`, exact: true }).selectOption(role)
      await page.getByRole('combobox', { name: `${asset}流动性`, exact: true }).selectOption('liquid')
      await page.getByLabel(`${asset}分类与代理理由`, { exact: true }).fill(`${asset}明确代理与经济用途`)
      await page.getByLabel(`${asset}均值不确定半宽（百分点）`, { exact: true }).fill('1')
    }
    const covarianceLabel = method === 'black_litterman' ? '资产风险协方差' : '共用风险协方差'
    await page.locator('summary').filter({ hasText: new RegExp(`^${covarianceLabel}$`) }).click()
    await page.getByLabel(`${covarianceLabel}：股票 / 股票`, { exact: true }).fill('0.04')
    await page.getByLabel(`${covarianceLabel}：债券 / 债券`, { exact: true }).fill('0.01')
    await page.getByLabel(new RegExp(`^${covarianceLabel}：(股票 / 债券|债券 / 股票)$`)).fill('0.001')
    if (method === 'black_litterman') {
      await page.getByLabel('股票市场权重（%）', { exact: true }).fill('60')
      await page.getByLabel('债券市场权重（%）', { exact: true }).fill('40')
      await page.getByLabel('市场权重来源', { exact: true }).fill('验收者明确提供的基准权重')
      await page.getByLabel('市场风险厌恶系数 δ', { exact: true }).fill('3')
      await page.getByLabel('年化无风险收益（%）', { exact: true }).fill('2')
      await page.getByRole('button', { name: '添加观点', exact: true }).click()
      await page.getByRole('combobox', { name: '观点1类型', exact: true }).selectOption('relative')
      await page.getByRole('combobox', { name: '观点1资产', exact: true }).selectOption('股票')
      await page.getByRole('combobox', { name: '观点1比较资产', exact: true }).selectOption('债券')
      await page.getByLabel('观点1年化收益差（百分点）', { exact: true }).fill('2')
      await page.getByLabel('观点1标准差（百分点）', { exact: true }).fill('1')
      const day = new Date().toISOString().slice(0, 10)
      await page.getByLabel('观点1观察日', { exact: true }).fill(day)
      await page.getByLabel('观点1可得日', { exact: true }).fill(day)
      await page.getByLabel('观点1依据', { exact: true }).fill('同研究日可得的明确相对观点')
    } else {
      for (const [index, name, probability, equity, bond] of [[1, '增长', '60', '8', '2'], [2, '收缩', '40', '-4', '5']]) {
        await page.getByRole('button', { name: '添加情景', exact: true }).click()
        await page.getByLabel(`情景${index}名称`, { exact: true }).fill(String(name))
        await page.getByLabel(`情景${index}概率（%）`, { exact: true }).fill(String(probability))
        await page.getByLabel(`情景${index}股票年化收益（%）`, { exact: true }).fill(String(equity))
        await page.getByLabel(`情景${index}债券年化收益（%）`, { exact: true }).fill(String(bond))
        await page.getByLabel(`情景${index}依据`, { exact: true }).fill('明确概率与条件收益，非事件发生预测')
      }
    }
    await page.getByRole('checkbox', { name: /我已确认同币种/ }).check()
    const previewResponse = page.waitForResponse(response => response.url().endsWith('/cma/preview') && response.request().method() === 'POST')
    await page.getByRole('button', { name: '验证长期假设', exact: true }).click()
    const preview = await previewResponse
    expect(preview.status()).toBe(200)
    const effective = await preview.json()
    expect(effective.model_result.method).toBe(method)
    expect(effective.execution.cma_models.python_fallback).toBe(0)
    expect(effective.definition.assets.every((asset: { annual_return: unknown }) => asset.annual_return === null)).toBe(true)
    await expect(page.getByText(/资产轴和风险矩阵已通过校验/)).toBeVisible()
    await verifyLayout(page)
    await page.screenshot({ path: info.outputPath(`${method}-effective-cma.png`), fullPage: true })
    // Editing an input must invalidate the old preview, not certify stale numbers.
    await page.getByRole('textbox', { name: '模型与风险依据', exact: true }).fill('更新来源后必须重新验证的明确研究假设')
    await expect(page.getByRole('button', { name: '确认保存假设版本', exact: true })).toBeDisabled()
    await page.getByRole('button', { name: '验证长期假设', exact: true }).click()
    await expect(page.getByRole('button', { name: '确认保存假设版本', exact: true })).toBeEnabled()
    const savedResponse = page.waitForResponse(response => response.url().endsWith('/cma') && response.request().method() === 'POST')
    await page.getByRole('button', { name: '确认保存假设版本', exact: true }).click()
    const savedCma = await (await savedResponse).json()
    await page.getByRole('checkbox', { name: '增加风险预算候选', exact: true }).check()
    await expect(page.getByRole('button', { name: '比较符合目标的政策候选', exact: true })).toBeDisabled()
    await page.getByLabel('股票风险预算（%）', { exact: true }).fill('50')
    await page.getByLabel('债券风险预算（%）', { exact: true }).fill('50')
    await page.getByRole('button', { name: '比较符合目标的政策候选', exact: true }).click()
    const budget = page.getByRole('table', { name: '长期政策候选比较' }).getByRole('row').filter({ hasText: '风险预算匹配（有限搜索）' })
    await expect(budget).toBeVisible()
    await budget.getByRole('button', { name: '复核此候选', exact: true }).click()
    await page.getByLabel('采纳理由与复核关注点', { exact: false }).fill('选用冻结模型与明确风险预算；模型改变须重新研究。')
    const policyResponse = page.waitForResponse(response => response.url().endsWith('/policies') && response.request().method() === 'POST')
    await page.getByRole('button', { name: '确认采用此长期政策', exact: true }).click()
    const baseline = await (await policyResponse).json()
    expect(baseline.policy.covariance).toEqual(savedCma.effective_covariance)
    expect(baseline.policy.assumptions).toEqual(savedCma.effective_assumptions)
    expect(baseline.policy.selection.id).toBe('risk-budget')
    await verifyLayout(page)
    await page.getByRole('button', { name: /进入 TAA，研究是否需要偏离/ }).click()
    await expect(page).toHaveURL(new RegExp(`baseline=${baseline.id}`))
    await expect(page.getByRole('heading', { name: '战术资产配置', exact: true })).toBeVisible()
    const calculate = page.getByRole('button', { name: '计算并比较方案', exact: true })
    await expect(calculate).toBeEnabled({ timeout: 15000 })
    const tacticalResponse = page.waitForResponse(response => response.url().endsWith('/tactical-allocation/preview') && response.request().method() === 'POST')
    await calculate.click()
    const tactical = await tacticalResponse
    expect(tactical.status()).toBe(200)
    const result = await tactical.json()
    expect(result.baseline.policy.covariance).toEqual(savedCma.effective_covariance)
    expect(result.request.decision_policy).not.toBeNull()
    expect(result.application.eligible).toBe(false)
    expect(result.execution.python_fallback).toBe(0)
    await expect(page.getByRole('heading', { name: '不可交接', exact: true })).toBeVisible()
    await verifyLayout(page)
    await page.screenshot({ path: info.outputPath(`${method}-taa-real-result.png`), fullPage: true })
    expect(errors).toEqual([])
  })
}

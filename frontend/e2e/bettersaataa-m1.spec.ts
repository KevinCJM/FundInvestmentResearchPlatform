import { test, expect } from '@playwright/test'
import { saveMandateFixture } from './helpers/mandates'
import { fillLtcmaAsset, previewCurrentLtcma, publishCurrentLtcma } from './helpers/ltcma'
import { auditTextContrast } from './helpers/contrast'

test('real M1 objective → no-product scope → LTCMA → SAA → explicit mapping; responsive and readonly', async ({ page, request }, info) => {
  const errors: string[] = [], posts: string[] = []
  page.on('pageerror', e => errors.push(e.message))
  await page.route('**/api/**', async route => {
    const url = new URL(route.request().url())
    if (route.request().method() === 'POST') posts.push(url.pathname)
    const response = await route.fetch({ url: `http://127.0.0.1:8129${url.pathname}${url.search}` })
    await route.fulfill({ response })
  })
  const day = new Date().toISOString().slice(0, 10)
  // 同名范围现在按规则拒绝重复；夹具多次运行必须使用唯一名称。
  const runTag = Date.now().toString(36)
  const review = new Date(Date.now() + 180 * 86400000).toISOString().slice(0, 10)
  const saved = await saveMandateFixture(request, 'http://127.0.0.1:8129/api/strategic-allocation', {
    definition: { name: `M1企业现金研究-${runTag}`, as_of: day, review_date: review, target_return: 0, max_volatility: .2,
      boundary_reason: '明确现金支付与损失承受能力边界',
      institutional_context: { investor_type: 'corporate_treasury', purpose: '经营储备与长期风险资金分开研究',
        cash_reserve_weight: .2, balance_sheet: { as_of: day, currency: 'CNY', source: '离线浏览器测试经济快照',
          investable_assets: 1000000, outside_assets: 200000, confirmed_liabilities: 300000 } } },
  })
  const diagnosis = saved.preview.institutional_diagnostics
  expect(diagnosis.balance_sheet.net_assets_after_confirmed_liabilities).toBe(900000)
  expect(diagnosis.balance_sheet.uncalled_commitments).toBeNull()
  expect(diagnosis.review_blockers).toHaveLength(5)
  await page.goto(`/pre-investment/product-pool?mandate=${saved.version.id}`)
  await page.getByRole('link', { name: '先做战略研究：独立资产范围' }).click()
  // 范围库已有记录时编辑器需显式新建，夹具重复运行也走同一条用户路径。
  await page.getByRole('link', { name: '新建研究范围' }).click()
  await page.getByLabel('战略范围名称').fill(`M1独立战略-${runTag}`)
  await page.getByText('补充说明（非必填）').click()
  await page.getByLabel('范围备注').fill('明确的离线风险与流动性研究')
  const scopeAssetIds: Record<string, string> = {}
  for (const [i, name, role, key] of [[1, '增长资产', 'growth', 'growth'], [2, '储备现金', 'liquidity', 'cash']] as const) {
    await page.getByRole('button', { name: '添加参考大类' }).click()
    const row = page.getByTestId('risk-reference-asset').nth(i - 1)
    await row.getByLabel('大类名称').fill(name)
    await row.getByLabel('资产类型').selectOption(role === 'liquidity' ? 'cash' : 'market')
  }
  const scopeRequest = page.waitForRequest(request => request.url().endsWith('/universes/preview') && request.method() === 'POST')
  expect(posts).not.toContain('/api/strategic-allocation/universes/confirm')
  await page.getByRole('button', { name: '保存战略范围' }).click()
  const scopeAssets = (await scopeRequest).postDataJSON().assets
  scopeAssetIds.growth = scopeAssets[0].id; scopeAssetIds.cash = scopeAssets[1].id
  expect(scopeAssetIds.growth).toMatch(/^asset-[0-9a-f]{8}$/)
  await expect(page.getByRole('table', { name: '只读战略资产摘要' })).toBeVisible()
  const scopeId = await page.getByRole('link', { name: '下一步', exact: true }).getAttribute('href').then(url => new URL(url!, 'http://test').searchParams.get('strategic_universe'))
  expect(scopeId).toBeTruthy()
  const scopeRead = await request.get(`http://127.0.0.1:8129/api/strategic-allocation/universes/${scopeId}`)
  expect(await scopeRead.json()).toMatchObject({ mandate_id: saved.version.id, mandate_hash: saved.version.content_hash })
  await page.evaluate(() => { localStorage.clear(); sessionStorage.clear() })
  await page.goto(`/pre-investment/product-pool/new?scope=strategic&strategic_universe=${scopeId}&edit=1`)
  const goalPanel = page.getByRole('region', { name: '当前投资目标与约束' })
  await expect(goalPanel.getByRole('heading', { name: saved.version.name })).toBeVisible()
  await expect(page.getByRole('button', { name: '保存修改' })).toBeEnabled()
  await page.reload()
  await expect(goalPanel.getByRole('heading', { name: saved.version.name })).toBeVisible()
  await page.getByRole('button', { name: '取消编辑' }).click()
  await expect(page.getByRole('table', { name: '只读战略资产摘要' })).toBeVisible()
  for (const width of [320, 768, 1440]) {
    await page.setViewportSize({ width, height: 1000 })
    await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth - innerWidth)).toBeLessThanOrEqual(1)
    // 颜色过渡结束后再断言稳定状态，避免把中间帧的透明度当成最终对比度。
    await expect.poll(() => page.evaluate(() => {
      const button = Array.from(document.querySelectorAll('button')).find(node => node.textContent?.includes('重试读取范围目录'))
      return button ? getComputedStyle(button).opacity : '1'
    })).toBe('1')
    expect(await page.evaluate(auditTextContrast)).toEqual([])
    await page.screenshot({ path: info.outputPath(`scope-${width}.png`), fullPage: true })
  }
  await expect(page.getByRole('region', { name: '独立战略范围' }).locator('a[href*="/saa/"]')).toHaveCount(0)
  await page.getByRole('link', { name: '下一步', exact: true }).click()
  await expect(page).toHaveURL(/\/pre-investment\/ltcma\/new\?/)
  await page.getByLabel('名称', { exact: true }).fill(`M1 独立战略 LTCMA-${runTag}`)
  await expect(page.getByRole('button', { name: '读取历史风险参考' })).toHaveCount(0)
  await page.getByLabel('假设依据', { exact: false }).fill('离线测试手工前瞻预期，非投资建议')
  for (const [id, ret, vol] of [['增长资产', '6', '15'], ['储备现金', '2', '1']] as const) {
    await fillLtcmaAsset(page, id, { rationale: '明确分类与前瞻假设依据，仅用于离线验收', annualReturn: ret, volatility: vol, uncertainty: '1' })
  }
  await page.getByLabel('相关矩阵: 增长资产 / 储备现金', { exact: true }).fill('0')
  await previewCurrentLtcma(page)
  await publishCurrentLtcma(page)
  await page.getByRole('button', { name: '用于 SAA', exact: true }).click()
  await page.getByRole('button', { name: '比较符合目标的政策候选' }).click()
  await expect(page.getByRole('table', { name: '长期政策候选比较' })).toBeVisible()
  await page.getByRole('button', { name: '复核此候选' }).first().click()
  await page.getByLabel(/采纳理由与复核关注点/).fill('保留未映射资产，明确只进行前瞻研究')
  await page.getByRole('button', { name: '确认采用此长期政策' }).click()
  await expect(page.getByRole('button', { name: /进入 TAA，研究是否需要偏离/ })).toBeDisabled()
  await expect(page.getByText(/已保存纯前瞻政策/)).toBeVisible()
  await page.goto(`/pre-investment/product-pool/new?scope=strategic&strategic_universe=${scopeId}`)
  await expect(page.getByText(/只读战略范围/)).toBeVisible()
  // 实施映射默认折叠，先展开再填写。
  await page.getByText('匹配真实代理产品（可稍后完成）', { exact: true }).click()
  await page.getByLabel('锁定的产品域').selectOption('m1-browser-domain')
  await page.getByLabel('实际代理大类方案').selectOption('浏览器离线股债')
  const later = new Date(Date.now() + 60 * 86400000).toISOString().slice(0, 10)
  await page.getByLabel('映射复核日').fill(later)
  await page.getByLabel('增长资产的真实代理').selectOption('股票')
  await page.getByLabel('增长资产匹配理由').fill('明确选择真实权益代理，不按名称匹配')
  await page.getByRole('button', { name: '检查映射覆盖' }).click()
  await expect(page.getByText(/存在实施缺口，暂不能进入TAA/)).toBeVisible()
  await page.getByLabel('储备现金的真实代理').selectOption('债券')
  await page.getByLabel('储备现金匹配理由').fill('只验证显式映射技术，不认定债券符合真实经营现金要求')
  await page.getByRole('button', { name: '检查映射覆盖' }).click()
  await expect(page.getByText(/已覆盖全部战略资产/)).toBeVisible()
  await page.getByRole('button', { name: '确认保存实施映射' }).click()
  await expect(page.getByText(/只读映射/)).toBeVisible()
  for (const width of [320, 768, 1440]) {
    await page.setViewportSize({ width, height: 1000 })
    await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth - innerWidth)).toBeLessThanOrEqual(1)
    expect(await page.evaluate(auditTextContrast)).toEqual([])
    await page.screenshot({ path: info.outputPath(`mapping-${width}.png`), fullPage: true })
  }
  await page.getByRole('button', { name: '复制映射为新研究' }).focus()
  await expect(page.getByRole('button', { name: '复制映射为新研究' })).toBeFocused()
  await page.keyboard.press('Enter')
  await expect(page.getByLabel('映射名称')).toBeEnabled()
  expect(errors).toEqual([])
  await page.unrouteAll({ behavior: 'wait' })
})

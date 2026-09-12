import { test, expect } from '@playwright/test'

test('real isolated API: historical frontier stays discoverable and renders real ECharts', async ({ page }, info) => {
  const errors: string[] = []
  let frontierPayload: any = null
  page.on('pageerror', error => errors.push(error.message))
  await page.route('**/api/**', async route => {
    const source = new URL(route.request().url())
    if (source.pathname === '/api/efficient-frontier' && route.request().method() === 'POST') {
      frontierPayload = route.request().postDataJSON()
    }
    const result = await route.fetch({ url: `http://127.0.0.1:8118${source.pathname}${source.search}`, timeout: 60000 })
    await route.fulfill({ response: result })
  })
  await page.goto('/pre-investment/saa/allocation-lab')
  await expect(page.getByRole('heading', { name: '大类资产配置' })).toBeVisible()
  await page.getByRole('button', { name: '选择该方案' }).click()
  await expect(page).toHaveURL(/\/pre-investment\/saa\/allocation-lab\?alloc=/)
  await expect(page.getByText('当前大类：浏览器离线股债 · 2 类')).toBeVisible()
  await page.getByText('高级设置：指标口径与计算精度').click()
  await page.getByLabel('第0轮样本点').fill('1200')
  await page.getByLabel('权重量化').selectOption('0.002')
  await page.getByRole('checkbox', { name: '使用受约束局部精炼' }).check()
  await page.getByLabel('局部精炼最大迭代次数').fill('25')
  await page.getByRole('button', { name: '生成可配置空间与有效前沿' }).click()
  await expect(page.getByRole('table', { name: '长期配置候选' })).toBeVisible({ timeout: 60000 })
  await expect(page.getByText(/实际样本：.*采样候选 [1-9][0-9]* 个.*当前候选合计 [1-9][0-9]* 个.*有效前沿 [1-9][0-9]* 个/)).toBeVisible()
  await expect(page.getByText(/局部精炼仅处理最大夏普、最小风险、最大收益 3 个代表候选/)).toBeVisible()
  await expect(page.getByText(/严格改善的结果会加入候选集合并重新构造前沿/)).toBeVisible()
  expect(frontierPayload?.exploration?.rounds?.[0]?.samples).toBe(1200)
  expect(frontierPayload?.quantization?.step).toBe(.002)
  expect(frontierPayload?.refine).toEqual({ enabled: true, method: 'bounded_pairwise_pattern_search_njit', iterations: 25 })
  await expect(page.getByText('有效前沿图（默认展开，可收起）')).toBeVisible()
  await expect(page.locator('canvas').first()).toBeVisible()
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBeTruthy()
  await page.screenshot({ path: info.outputPath('historical-frontier-real-echarts.png'), fullPage: true })
  expect(errors).toEqual([])
})

test('real target grid: 20 and 200 solves change the actual frontier and preserve failures', async ({ page }, info) => {
  const results: Array<{ input: any; output: any }> = []
  const errors: string[] = []
  page.on('pageerror', error => errors.push(error.message))
  await page.route('**/api/**', async route => {
    const source = new URL(route.request().url())
    const response = await route.fetch({ url: `http://127.0.0.1:8118${source.pathname}${source.search}`, timeout: 60000 })
    if (source.pathname === '/api/efficient-frontier') {
      results.push({ input: route.request().postDataJSON(), output: await response.json() })
    }
    await route.fulfill({ response })
  })
  await page.goto('/pre-investment/saa/policy')
  await page.getByRole('link', { name: /历史有效前沿与策略回测/ }).click()
  await page.getByRole('button', { name: '选择该方案' }).click()
  await expect(page.getByText('当前大类：浏览器离线股债 · 2 类')).toBeVisible()
  await page.getByRole('checkbox', { name: '按目标网格加密整条前沿' }).check()
  await page.getByText('高级设置：指标口径与计算精度').click()
  await page.getByLabel('权重量化').selectOption('0.005')
  await expect(page.getByRole('button', { name: '生成可配置空间与有效前沿' })).toBeDisabled()
  await page.getByRole('checkbox', { name: /我确认网格采用连续权重/ }).check()
  let firstCanvas: Buffer | undefined
  for (const count of [20, 200]) {
    await page.getByLabel('前沿目标点数', { exact: true }).fill(String(count))
    await page.getByLabel('单点最大迭代次数', { exact: true }).fill('300')
    await page.getByRole('button', { name: '生成可配置空间与有效前沿' }).click()
    const panel = page.getByRole('region', { name: '逐目标前沿求解结果' })
    await expect(panel.getByRole('status')).toContainText(`目标 ${count} 个 · 成功 ${count} 个`, { timeout: 60000 })
    const { input, output } = results[results.length - 1]
    expect(input.frontier_grid).toEqual({ point_count: count, max_iterations: 300, weight_domain: 'continuous', accept_continuous_weights: true })
    expect(output.frontier_grid.solver_calls).toBe(count)
    expect(output.frontier_grid.points).toHaveLength(count)
    expect(output.frontier_grid.curve.filter(Boolean).length).toBeGreaterThan(count * .9)
    expect(output.accepted_candidates).toBe(output.sampled_candidates + output.refined_candidates + output.grid_candidates)
    for (const point of output.frontier_grid.points) {
      expect(point.value[1]).toBeGreaterThanOrEqual(point.target - 1e-7)
      expect(point.weights.reduce((sum: number, value: number) => sum + value, 0)).toBeCloseTo(1, 7)
      expect(point.constraint_violation).toBeLessThanOrEqual(1e-7)
    }
    const canvas = page.locator('canvas').first()
    await expect(canvas).toBeVisible()
    const image = await canvas.screenshot({ path: info.outputPath(`target-grid-${count}.png`) })
    if (firstCanvas) expect(image.equals(firstCanvas)).toBe(false)
    firstCanvas = image
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBeTruthy()
  }
  // Exhaustion is an actual backend result, not a mock or hidden omission.
  await page.getByLabel('单点最大迭代次数', { exact: true }).fill('1')
  await page.getByRole('button', { name: '生成可配置空间与有效前沿' }).click()
  await expect(page.getByText(/前沿端点未完成/)).toBeVisible({ timeout: 60000 })
  expect(results[results.length - 1].output.frontier_grid.unattempted_points).toBe(200)
  expect(errors).toEqual([])
})

test('real isolated API: goal, CMA, policy adoption and TAA on desktop/mobile', async ({ page, request }, info) => {
  const errors: string[] = []
  const writes: string[] = []
  page.on('pageerror', error => errors.push(error.message))
  await page.route('**/api/**', async route => {
    const path = new URL(route.request().url()).pathname
    if (!path.startsWith('/api/strategic-allocation/') && !path.startsWith('/api/tactical-allocation/')) {
      return route.fulfill({ status: 404, json: { detail: 'Isolated browser acceptance: unrelated service not provided.' } })
    }
    if (route.request().method() === 'POST') writes.push(path)
    const result = await route.fetch({ url: `http://127.0.0.1:8118${path}`, timeout: 60000 })
    await route.fulfill({ response: result })
  })
  await page.goto('/pre-investment/objectives')
  await expect(page.getByRole('heading', { name: '投资目标与边界' })).toBeVisible()
  await page.getByLabel('目标名称', { exact: true }).fill(`浏览器目标-${info.project.name}`)
  await page.getByLabel('最低预期年收益（%）', { exact: false }).fill('3')
  await page.getByRole('button', { name: '保存新目标版本', exact: true }).click()
  await page.getByRole('link', { name: /使用此目标进入长期配置/ }).click()
  await page.getByRole('combobox', { name: '已保存的大类配置', exact: true }).selectOption('浏览器离线股债')
  await page.getByRole('button', { name: '填写长期假设', exact: true }).click()
  await expect(page.getByLabel('股票预期年收益（%）')).toHaveValue('')
  await page.getByLabel('预测来源与主要假设', { exact: false }).fill('离线验收显式假设：人民币十年算术总收益，无投资推荐含义。')
  for (const [asset, role, annualReturn, uncertainty] of [['股票', 'growth', '7', '2'], ['债券', 'rates', '2.5', '.5']]) {
    await page.getByRole('combobox', { name: `${asset}经济角色`, exact: true }).selectOption(role)
    await page.getByRole('combobox', { name: `${asset}流动性`, exact: true }).selectOption('liquid')
    await page.getByLabel(`${asset}分类与代理理由`, { exact: true }).fill(`${asset}经济风险代理，仅供离线验收`)
    await page.getByLabel(`${asset}预期年收益（%）`, { exact: true }).fill(annualReturn)
    await page.getByLabel(`${asset}均值不确定半宽（百分点）`, { exact: false }).fill(uncertainty)
  }
  await page.getByRole('button', { name: '读取历史风险参考', exact: true }).click()
  await expect(page.getByText(/已读取 300 个共同收益观察期/)).toBeVisible()
  await expect(page.getByLabel('股票预期年收益（%）')).toHaveValue('7')
  await page.getByRole('checkbox', { name: /我已确认：所有假设/ }).check()
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBeTruthy()
  await page.screenshot({ path: info.outputPath('cma-inputs.png'), fullPage: true })
  await page.getByRole('button', { name: '验证长期假设', exact: true }).click()
  await expect(page.getByText(/资产轴和风险矩阵已通过校验/)).toBeVisible()
  expect(writes.filter(path => path === '/api/strategic-allocation/cma')).toHaveLength(0)
  await page.getByRole('button', { name: '确认保存假设版本', exact: true }).click()
  await page.getByRole('button', { name: '比较符合目标的政策候选', exact: true }).click()
  await expect(page.getByRole('table', { name: '长期政策候选比较' })).toBeVisible()
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBeTruthy()
  await page.screenshot({ path: info.outputPath('policy-comparison.png'), fullPage: true })
  await page.getByRole('row').filter({ hasText: '区间稳健效用' }).getByRole('button', { name: '复核此候选' }).click()
  await page.getByLabel('采纳理由与复核关注点', { exact: false }).fill('采用稳健候选，定期复核前瞻假设与风险预算。')
  await page.getByRole('button', { name: '确认采用此长期政策', exact: true }).click()
  await page.getByRole('button', { name: /进入 TAA，研究是否需要偏离/ }).click()
  await expect(page.getByRole('heading', { name: '本次准备怎么配？' })).toBeVisible()
  await page.getByRole('tab', { name: '回测与选优' }).click()
  await page.getByRole('checkbox', { name: '同时运行多段样本外检验' }).check()
  await page.getByLabel('初始训练期数', { exact: false }).fill('100')
  await page.getByLabel('每段验证期数', { exact: false }).fill('40')
  await page.getByRole('button', { name: '计算并比较方案', exact: true }).click()
  await expect(page.getByText(/政策前瞻风险门禁：/)).toBeVisible({ timeout: 60000 })
  await expect(page.getByRole('table', { name: '分段样本外结果' })).toBeVisible({ timeout: 60000 })
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBeTruthy()
  await page.screenshot({ path: info.outputPath('taa-validation.png'), fullPage: true })
  const catalog = await (await request.get('http://127.0.0.1:8118/api/strategic-allocation/catalog')).json()
  expect(catalog.policies.length).toBeGreaterThan(0)
  expect(writes.filter(path => path === '/api/strategic-allocation/policies')).toHaveLength(1)
  expect(errors).toEqual([])
})

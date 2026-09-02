import { expect, test, type Page } from '@playwright/test'

const presentation = {
  indicator_id: 'builtin-total-return-v2', revision: 1, name: '累计收益率', source: 'built_in',
  category: 'return_statistics', category_label: '收益与条件统计', context_kind: 'single_product', catalog_status: 'current',
  display_format: 'percent', precision: 2, unit: '%', notation: 'standard', value_scale: 100,
  output_measure: 'return_decimal', direction: 'higher_better', description: '逐期普通收益增长因子累乘后减一。',
  methodology: '使用真实复权净值计算。', data_basis: '真实数据、严格窗口、缺失不填充',
  minimum_observations: 2, applicable_product_kinds: ['etf', 'fund'],
}

const indicator = {
  id: 'builtin-total-return-v2', revision: 1, source: 'built_in', read_only: true,
  name: '累计收益率', description: presentation.description, expression: 'product(returns + 1) - 1',
  display_latex: '\\prod\\left(\\mathbf{r}+1\\right)-1',
  periods: ['1W', '1M', '1Y'], period_policy: 'all_supported', unit: '%', display_format: 'percent', precision: 2,
  direction: 'higher_better', annual_risk_free_rate_percent: 1.5, context_kind: 'single_product',
  category_id: 'return_statistics', category_label: '收益与条件统计', applicable_product_kinds: ['etf', 'fund'],
  catalog_status: 'current', ui_exposed: true, presentation,
  created_at: '2026-01-01T00:00:00Z', updated_at: '2026-01-01T00:00:00Z',
}

async function mockMetricDisplayApi(page: Page) {
  await page.route('**/api/**', async (route) => {
    const url = new URL(route.request().url())
    if (url.pathname === '/api/instruments/products/510300.SH') {
      return route.fulfill({ json: {
        product_id: '510300.SH', name: '沪深300ETF', management: '示例管理人', status: '上市',
        base_info: {
          ts_code: '510300.SH', fund_type: 'ETF', list_date: '2012-05-28', delist_date: '2026-12-31',
        },
        metrics: { issue_amount: 100, m_fee: 0.5, c_fee: 0.1 },
        timeseries: Array.from({ length: 25 }, (_, index) => {
          const close = 3 + index * 0.002 + ((index % 5) - 2) * 0.005
          return {
            date: new Date(Date.UTC(2026, 0, 2 + index)).toISOString().slice(0, 10),
            open: close - 0.002, high: close + 0.01, low: close - 0.01, close,
            volume: 1000 + index * 10,
          }
        }),
      } })
    }
    if (url.pathname === '/api/custom-indicators/meta') {
      return route.fulfill({ json: { periods: [{ value: '1M', label: '近 1 月', description: '自然月窗口' }, { value: '1Y', label: '近 1 年', description: '自然年窗口' }] } })
    }
    if (url.pathname === '/api/custom-indicators' && route.request().method() === 'GET') {
      return route.fulfill({ json: { items: [indicator], total: 1 } })
    }
    if (url.pathname === '/api/custom-indicators/evaluate') {
      const period = (route.request().postDataJSON() as { period?: string } | null)?.period ?? '1Y'
      return route.fulfill({ json: {
        results: [{
          indicator_id: indicator.id, indicator_revision: 1, indicator_name: indicator.name,
          target: { kind: 'etf', product_id: '510300.SH', name: '沪深300ETF' }, period,
          value: 0.0183, status: 'ok', warnings: [], presentation,
          window: { requested_as_of: null, effective_as_of: '2026-01-06', start_date: period === '1M' ? '2025-12-06' : '2025-01-06', end_date: '2026-01-06', observation_count: period === '1M' ? 21 : 250, data_latest_date: '2026-01-06' },
        }],
        summary: { total: 1, ok: 1, warning: 0, error: 0 }, cache: { hits: 0, misses: 1 },
      } })
    }
    return route.fulfill({ status: 404, json: {} })
  })
}

test('详情页在三档宽度统一展示指标值、窗口和定义抽屉', async ({ page }) => {
  await mockMetricDisplayApi(page)
  await page.goto('/product/510300.SH?kind=etf')

  await expect(page.getByRole('heading', { name: '自定义研究指标' })).toBeVisible()
  await expect(page.getByLabel('上市日期：2012-05-28')).toBeVisible()
  await expect(page.getByLabel('退市日期：2026-12-31')).toBeVisible()
  await expect(page.getByLabel('统计区间')).toHaveValue('ALL')
  await expect(page.getByRole('heading', { name: '未来虚拟净值模拟' })).toBeVisible()
  await expect(page.getByText(/所有路径统一从虚拟净值 1\.0000 出发/)).toBeVisible()
  await expect(page.getByLabel('模拟未来区间')).toHaveValue('252')
  await expect(page.getByLabel('模拟路径数')).toHaveValue('500')
  await expect(page.getByLabel('Bootstrap 平均区块长度')).toHaveValue('20')
  await expect(page.getByLabel('目标期末收益率')).toHaveValue('5')
  await expect(page.getByRole('radio', { name: '参数化蒙特卡洛' })).toBeChecked()
  const combinedMonteCarloChart = page.getByLabel('参数化蒙特卡洛（偏度/峰度校准）：路径与期末净值概率分布组合图')
  await expect(combinedMonteCarloChart).toBeVisible()
  await expect(combinedMonteCarloChart.locator('canvas')).toHaveCount(1)
  await expect(page.getByText(/横向柱状图按期末净值区间展示实际路径数，共计 500 条/)).toBeVisible()
  await page.getByLabel('模拟路径数').selectOption('200')
  await expect(page.getByText(/共计 200 条/)).toBeVisible()
  await expect(page.getByRole('heading', { name: '双模型结果对比' })).toBeVisible()
  await page.getByText('区块 Bootstrap', { exact: true }).click()
  await expect(page.getByRole('radio', { name: '区块 Bootstrap' })).toBeChecked()
  await expect(page.getByLabel('历史区块 Bootstrap：路径与期末净值概率分布组合图')).toBeVisible()
  await expect(page.getByRole('heading', { name: '箱形图' })).toBeVisible()
  await expect(page.getByRole('heading', { name: '正态 Q-Q 图' })).toBeVisible()
  await expect(page.getByText('查看关键分位点数据')).toBeVisible()
  const diagnosticGrid = page.getByTestId('distribution-diagnostics-grid')
  const diagnosticCards = diagnosticGrid.locator(':scope > div')
  await expect(diagnosticCards).toHaveCount(2)
  const boxPlotBounds = await diagnosticCards.nth(0).boundingBox()
  const normalQqBounds = await diagnosticCards.nth(1).boundingBox()
  expect(boxPlotBounds).not.toBeNull()
  expect(normalQqBounds).not.toBeNull()
  if ((page.viewportSize()?.width ?? 0) >= 1024) {
    expect(Math.abs(boxPlotBounds!.y - normalQqBounds!.y)).toBeLessThanOrEqual(2)
    expect(boxPlotBounds!.x + boxPlotBounds!.width).toBeLessThan(normalQqBounds!.x)
  } else {
    expect(normalQqBounds!.y).toBeGreaterThan(boxPlotBounds!.y + boxPlotBounds!.height)
  }
  await expect(page.getByText('1.83%')).toBeVisible()
  await expect(page.getByText(/1Y · 2025-01-06 至 2026-01-06 · 250 个观察值/)).toBeVisible()
  await expect(page.getByLabel('累计收益率计算区间')).toHaveValue('1Y')
  await page.getByLabel('累计收益率计算区间').selectOption('1M')
  await expect(page.getByLabel('累计收益率计算区间')).toHaveValue('1M')
  await expect(page.getByText(/1M · 2025-12-06 至 2026-01-06 · 21 个观察值/)).toBeVisible()
  await page.getByRole('button', { name: /选择研究指标/ }).click()
  const selectorPanel = page.getByRole('dialog', { name: '选择研究指标面板' })
  await expect(selectorPanel).toBeVisible()
  await expect(page.getByLabel('按指标来源筛选')).toHaveValue('all')
  await expect(page.getByLabel('按指标来源筛选').getByRole('option')).toHaveText([
    '全部',
    '内置指标',
    '工作区指标',
  ])
  const panelBox = await selectorPanel.boundingBox()
  const viewportWidth = page.viewportSize()?.width ?? 0
  expect(panelBox).not.toBeNull()
  expect(panelBox!.x).toBeGreaterThanOrEqual(15)
  expect(panelBox!.x + panelBox!.width).toBeLessThanOrEqual(viewportWidth - 15)
  await page.getByRole('button', { name: '完成' }).click()
  await page.getByRole('button', { name: '查看定义与口径' }).click()
  await expect(page.getByRole('dialog', { name: '累计收益率' })).toBeVisible()
  await expect(page.getByText('真实数据、严格窗口、缺失不填充')).toBeVisible()
  await expect(page.getByTestId('metric-formula-latex').locator('.katex')).toBeVisible()
  await page.getByRole('button', { name: '关闭' }).click()

  await page.getByRole('button', { name: '移除指标 累计收益率' }).click()
  await expect(page.getByText('1.83%')).not.toBeVisible()
  await expect(page.getByRole('button', { name: /选择研究指标/ })).toContainText('已选 0/8')

  await page.getByLabel('模拟未来区间').selectOption('21')
  await page.getByLabel('模拟路径数').selectOption('200')
  await page.getByRole('button', { name: '重新模拟' }).click()
  await expect(page.getByText('200 条虚拟路径中的样本比例')).toBeVisible()
  await page.getByLabel('统计区间').selectOption('1M')
  await expect(page.getByText(/要求产品完整覆盖所选区间/)).toBeVisible()

  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth)
  expect(overflow).toBeLessThanOrEqual(1)
})

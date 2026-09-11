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

type AnalysisRequest = {
  statistics_period: string
  analysis_basis?: 'adjusted_nav' | 'price'
  include_simulation?: boolean
  price_ma_periods: number[]
  volume_ma_periods: number[]
  simulation_horizon: number
  simulation_path_count: number
  bootstrap_block_length: number
  fhs_ewma_lambda: number
  simulation_target_return: number
}

const SIMULATION_METHODS = ['gaussian', 'parametric', 'block_bootstrap', 'fhs_ewma', 'fhs_garch'] as const
type SpecSimulationMethod = (typeof SIMULATION_METHODS)[number]
const SIMULATION_METHOD_LABELS: Record<SpecSimulationMethod, string> = {
  gaussian: '标准正态蒙特卡洛',
  parametric: '参数化蒙特卡洛（偏度/峰度校准）',
  block_bootstrap: '历史区块 Bootstrap',
  fhs_ewma: '滤波历史模拟 · EWMA',
  fhs_garch: '滤波历史模拟 · GARCH(1,1)',
}

const makeSimulation = (method: SpecSimulationMethod, request: AnalysisRequest) => ({
  method,
  methodLabel: SIMULATION_METHOD_LABELS[method],
  days: [0, request.simulation_horizon],
  samplePaths: [[1, 1.03], [1, 0.98]],
  percentiles: {
    p05: [1, 0.94], p25: [1, 0.98], p50: [1, 1.03], p75: [1, 1.07], p95: [1, 1.12],
  },
  terminal: {
    p05: 0.94, p25: 0.98, p50: 1.03, p75: 1.07, p95: 1.12,
    lossProbability: 0.25, valueAtRisk95: 0.06, conditionalValueAtRisk95: 0.08,
    targetHitProbability: 0.42, averageMaxDrawdown: 0.07, p05Return: -0.06, medianReturn: 0.03,
  },
  assumptions: {
    sourceObservationCount: 24,
    targetReturnPercent: request.simulation_target_return,
    meanDailyLogReturn: method === 'parametric' ? 0.0002 : null,
    dailyLogVolatility: method === 'parametric' ? 0.008 : null,
    historicalLogSkewness: method === 'parametric' ? -0.2 : null,
    historicalLogExcessKurtosis: method === 'parametric' ? 0.8 : null,
    fittedLogSkewness: method === 'parametric' ? -0.18 : null,
    fittedLogExcessKurtosis: method === 'parametric' ? 0.76 : null,
    shapeCalibrationStatus: method === 'parametric' ? 'matched' : null,
    shapeSkewParameter: method === 'parametric' ? 0.1 : null,
    tailWeightParameter: method === 'parametric' ? 1.1 : null,
    averageBlockLength: method === 'block_bootstrap' ? request.bootstrap_block_length : null,
    conditionalVolatilityStart: method.startsWith('fhs') ? 0.011 : null,
    volatilityPersistence: method.startsWith('fhs') ? 0.97 : null,
    garchOmega: method === 'fhs_garch' ? 2e-6 : method === 'fhs_ewma' ? 0 : null,
    garchAlpha: method.startsWith('fhs') ? 0.08 : null,
    garchBeta: method.startsWith('fhs') ? 0.89 : null,
    ewmaLambda: method === 'fhs_ewma' ? request.fhs_ewma_lambda : null,
    residualSkewness: method.startsWith('fhs') ? -0.2 : null,
    residualExcessKurtosis: method.startsWith('fhs') ? 1.1 : null,
  },
})

const makeDensity = (pathCount: number, horizon: number) => ({
  sampleSize: pathCount,
  navAxisMin: 0.88,
  navAxisMax: 1.12,
  frames: [1, 2, 3, 4].map((step) => ({
    day: Math.round((horizon * step) / 4),
    navLow: 1 - 0.05 * step,
    navHigh: 1 + 0.05 * step,
    binWidth: (0.1 * step) / 4,
    countAxisMax: pathCount,
    curve: [1, 10, 25, 10, 1],
    bins: [pathCount * 0.1, pathCount * 0.4, pathCount * 0.4, pathCount * 0.1],
  })),
})

const makeAnalysis = (request: AnalysisRequest) => {
  const incomplete = request.statistics_period === '1M'
  const returns = incomplete
    ? []
    : Array.from({ length: 24 }, (_, index) => ({ date: `2026-01-${String(index + 2).padStart(2, '0')}`, return: index % 2 ? 0.2 : -0.1 }))
  const qqPoints = [
    { percentile: 0.05, theoreticalQuantile: -1.64, observedReturn: -0.3, referenceReturn: -0.25, tail: 'lower' },
    { percentile: 0.5, theoreticalQuantile: 0, observedReturn: 0.05, referenceReturn: 0.05, tail: 'center' },
    { percentile: 0.95, theoreticalQuantile: 1.64, observedReturn: 0.4, referenceReturn: 0.35, tail: 'upper' },
  ]
  const byMethod = Object.fromEntries(SIMULATION_METHODS.map((method) => [method, makeSimulation(method, request)]))
  return {
    schema_version: 1,
    product_id: '510300.SH',
    execution: {
      execution_backend: 'numba_njit_fixed_signature',
      engine: 'product-analysis-njit-1.0.0',
      kernel_version: 'product-chart-statistics-simulation-2',
      kernel_coverage: '21/21',
      kernel_fingerprint: '1234567890abcdef',
      kernel_signatures: { product_analysis_kernel: ['fixed'] },
      nopython: true,
      njit_required: true,
      object_mode: 0,
      python_fallback: 0,
      request_time_compilation: 0,
    },
    window: {
      complete: !incomplete,
      requested_start_date: incomplete ? '2025-12-25' : null,
      message: incomplete ? '近 1 月要求产品完整覆盖所选区间；当前历史数据不足，统计指标不计算。' : null,
    },
    technical: {
      availability: { ohlc: true, volume: true, kdj: true },
      priceMa: Object.fromEntries(request.price_ma_periods.map((period) => [String(period), Array(25).fill(3.02)])),
      volumeMa: Object.fromEntries(request.volume_ma_periods.map((period) => [String(period), Array(25).fill(1100)])),
      bollinger: { upper: Array(25).fill(3.1), middle: Array(25).fill(3.02), lower: Array(25).fill(2.94) },
      kdj: { kValues: Array(25).fill(55), dValues: Array(25).fill(52), jValues: Array(25).fill(61) },
    },
    dailyReturns: returns,
    returnStatistics: {
      mean: incomplete ? null : 0.05, std: incomplete ? null : 0.15, median: incomplete ? null : 0.05,
      positiveRatio: incomplete ? null : 0.5, best: incomplete ? null : 0.2, worst: incomplete ? null : -0.1,
      sampleSize: returns.length, skewness: incomplete ? null : -0.2, kurtosis: incomplete ? null : 0.8,
      jbStatistic: incomplete ? null : 1.2, normalityPValue: incomplete ? null : 0.55,
    },
    interpretation: {
      skewness: { label: '轻度左偏（负偏）', meaning: '测试偏度解释' },
      kurtosis: { label: '轻度尖峰厚尾', meaning: '测试峰度解释' },
      normality: incomplete ? '样本不足，无法进行检验' : '无法拒绝正态假设（5% 显著性水平）',
    },
    histogram: incomplete ? [] : [{ start: -0.2, end: 0.2, count: 24, normalPdfCount: 23.5, frequency: 1, center: 0 }],
    boxPlot: incomplete ? null : {
      stats: [-0.1, -0.05, 0.05, 0.15, 0.2, 0.2], outliers: [],
      quartiles: { q1: -0.05, median: 0.05, q3: 0.15, iqr: 0.2 }, whiskers: { lower: -0.1, upper: 0.2 },
    },
    normalQq: incomplete ? null : { sampleSize: returns.length, points: qqPoints, keyPoints: qqPoints },
    simulationStatus: incomplete ? 'insufficient_sample' : request.include_simulation ? 'complete' : 'not_requested',
    simulation: incomplete || !request.include_simulation ? null : {
      initialNav: 1,
      methods: [...SIMULATION_METHODS],
      byMethod,
      realized: null,
      realizedStatus: 'off',
      comparison: {
        p05ReturnGap: 0.01, medianReturnGap: 0.01, lossProbabilityGap: 0.02,
        conditionalValueAtRiskGap: 0.01, level: 'low', message: '各模型结果接近。',
      },
      densities: Object.fromEntries(SIMULATION_METHODS.map((method) => [method, makeDensity(request.simulation_path_count, request.simulation_horizon)])),
    },
    regimeAnalysis: null,
    researchContext: {
      analysisBasis: request.analysis_basis ?? 'adjusted_nav', basisLabel: request.analysis_basis === 'price' ? '交易价格' : '复权净值',
      startDate: '2026-01-02', endDate: '2026-01-26', observations: incomplete ? 0 : 25,
      returnObservations: returns.length, segmentCount: incomplete ? 0 : 1, scope: 'full',
      boundaryPolicy: '仅使用相邻有效观察值，缺失价格不连接。', simulationEligible: !incomplete,
      simulationMessage: incomplete ? '有效收益样本不足，无法模拟。' : null,
    },
  }
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
    if (url.pathname === '/api/historical-regimes/runs') {
      return route.fulfill({ json: { items: [] } })
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
        execution: {
          execution_backend: 'numba_njit_fixed_signature',
          nopython: true,
          object_mode: 0,
          python_fallback: 0,
          request_time_compilation: 0,
          kernel_signatures: { typed_indicator_plan: ['fixed'] },
        },
      } })
    }
    if (url.pathname === '/api/custom-indicators/evaluate-series') {
      return route.fulfill({ json: { results: [], execution: {
        execution_backend: 'numba_njit_fixed_signature', nopython: true,
        object_mode: 0, python_fallback: 0, request_time_compilation: 0,
        kernel_signatures: { ui_contract_fixture: ['fixed'] },
      } } })
    }
    if (url.pathname === '/api/instruments/products/510300.SH/analysis') {
      return route.fulfill({ json: makeAnalysis(route.request().postDataJSON() as AnalysisRequest) })
    }
    return route.fulfill({ status: 404, json: {} })
  })
}

test('详情页在三档宽度统一展示指标值、窗口、定义抽屉与显式模拟', async ({ page }, testInfo) => {
  await mockMetricDisplayApi(page)
  await page.goto('/product-research/products/510300.SH?kind=etf')

  await expect(page.getByRole('heading', { name: '自定义研究指标' })).toBeVisible()
  await page.getByText('产品资料与规模口径', { exact: true }).click()
  await expect(page.getByLabel('上市日期：2012-05-28')).toBeVisible()
  await expect(page.getByLabel('退市日期：2026-12-31')).toBeVisible()
  await expect(page.getByLabel('分析样本区间')).not.toBeVisible()
  await expect(page.getByText('暂无可用情景', { exact: true })).not.toBeVisible()
  await page.getByRole('tab', { name: '情景表现', exact: true }).click()
  await expect(page.getByRole('heading', { name: '暂无可用情景', exact: true })).toBeVisible()
  await page.evaluate(() => window.scrollTo(0, 0))
  await page.waitForTimeout(1100)
  await page.screenshot({ path: `../output/product-controls/${testInfo.project.name}-empty-scenario.png`, fullPage: true })
  await page.getByRole('tab', { name: '收益统计', exact: true }).click()
  await expect(page.getByLabel('分析样本区间')).toHaveValue('ALL')
  await expect(page.getByText('暂无可用情景', { exact: true })).not.toBeVisible()
  await expect(page.getByLabel('收益数据口径')).not.toBeVisible()
  await page.evaluate(() => window.scrollTo(0, 0))
  await page.waitForTimeout(1100)
  await page.screenshot({ path: `../output/product-controls/${testInfo.project.name}-toolbar.png`, fullPage: true })
  await page.getByRole('button', { name: '分析设置', exact: true }).click()
  await expect(page.getByLabel('收益数据口径')).toBeVisible()
  expect(await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth)).toBeLessThanOrEqual(1)
  await page.evaluate(() => window.scrollTo(0, 0))
  await page.waitForTimeout(1100)
  await page.screenshot({ path: `../output/product-controls/${testInfo.project.name}-settings.png`, fullPage: true })
  await page.getByRole('region', { name: '分析样本', exact: true }).screenshot({ path: `../output/product-controls/${testInfo.project.name}-settings-detail.png` })
  await page.getByRole('button', { name: '分析设置', exact: true }).click()
  await expect(page.getByLabel('收益数据口径')).not.toBeVisible()
  await page.getByRole('tab', { name: '走势与指标', exact: true }).click()
  await expect(page.getByText('1.83%')).toBeVisible()
  await expect(page.getByText(/1Y · 2025-01-06 至 2026-01-06 · 250 个观察值/)).toBeVisible()
  await page.getByLabel('累计收益率计算区间').selectOption('1M')
  await expect(page.getByLabel('累计收益率计算区间')).toHaveValue('1M')
  await expect(page.getByText(/1M · 2025-12-06 至 2026-01-06 · 21 个观察值/)).toBeVisible()
  await page.getByRole('button', { name: /选择研究指标/ }).click()
  const selectorPanel = page.getByRole('dialog', { name: '选择研究指标面板' })
  await expect(selectorPanel).toBeVisible()
  await expect(page.getByLabel('按指标来源筛选')).toHaveValue('all')
  await expect(page.getByLabel('按指标来源筛选').getByRole('option')).toHaveText(['全部', '内置指标', '工作区指标'])
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
  await page.getByRole('button', { name: '关闭', exact: true }).click()
  await page.getByRole('button', { name: '移除指标 累计收益率' }).click()
  await expect(page.getByText('1.83%')).not.toBeVisible()
  await expect(page.getByRole('button', { name: /选择研究指标/ })).toContainText('已选 0/8')

  await page.getByRole('tab', { name: '收益统计', exact: true }).click()
  await expect(page.getByRole('heading', { name: '箱形图' })).toBeVisible()
  await expect(page.getByRole('heading', { name: '正态 Q-Q 图' })).toBeVisible()
  await expect(page.getByText('查看关键分位点数据')).toBeVisible()
  const diagnosticCards = page.getByTestId('distribution-diagnostics-grid').locator(':scope > div')
  await expect(diagnosticCards).toHaveCount(2)
  const boxPlotBounds = await diagnosticCards.nth(0).boundingBox()
  const normalQqBounds = await diagnosticCards.nth(1).boundingBox()
  expect(boxPlotBounds).not.toBeNull()
  expect(normalQqBounds).not.toBeNull()
  if (viewportWidth >= 1024) {
    expect(Math.abs(boxPlotBounds!.y - normalQqBounds!.y)).toBeLessThanOrEqual(2)
    expect(boxPlotBounds!.x + boxPlotBounds!.width).toBeLessThan(normalQqBounds!.x)
  } else {
    expect(normalQqBounds!.y).toBeGreaterThan(boxPlotBounds!.y + boxPlotBounds!.height)
  }

  await page.getByRole('tab', { name: '未来模拟', exact: true }).click()
  await expect(page.getByRole('heading', { name: '未来虚拟净值模拟' })).toBeVisible()
  await expect(page.getByLabel('模拟未来区间')).toHaveValue('252')
  await expect(page.getByLabel('模拟路径数')).toHaveValue('500')
  // Model-specific inputs appear only after selecting that model; shared
  // controls stay separate and changing tabs must not start a calculation.
  await expect(page.getByLabel('Bootstrap 平均区块长度')).toHaveCount(0)
  await expect(page.getByLabel('目标期末收益率')).toHaveValue('5')
  const combinedMonteCarloChart = page.getByLabel('参数化蒙特卡洛（偏度/峰度校准）：路径与期末净值概率分布组合图')
  await expect(combinedMonteCarloChart).not.toBeVisible()
  await page.getByRole('button', { name: '运行模拟', exact: true }).click()
  await expect(combinedMonteCarloChart).toBeVisible()
  await expect(combinedMonteCarloChart.locator('canvas')).toHaveCount(1)
  await expect(page.getByRole('heading', { name: /\d+ 个模型结果对比/ })).toBeVisible()
  await page.getByText('区块 Bootstrap', { exact: true }).click()
  await expect(page.getByRole('radio', { name: '区块 Bootstrap' })).toBeChecked()
  await expect(page.getByLabel('Bootstrap 平均区块长度')).toHaveValue('20')
  await expect(page.getByLabel('历史区块 Bootstrap：路径与期末净值概率分布组合图')).toBeVisible()
  await page.getByLabel('模拟路径数').selectOption('200')
  await expect(page.getByLabel('历史区块 Bootstrap：路径与期末净值概率分布组合图')).not.toBeVisible()
  await page.getByLabel('模拟未来区间').selectOption('21')
  const recalculation = page.waitForRequest(request => new URL(request.url()).pathname.endsWith('/analysis') && request.method() === 'POST' && request.postDataJSON().include_simulation === true)
  await page.getByRole('button', { name: '运行模拟', exact: true }).click()
  expect((await recalculation).postDataJSON()).toMatchObject({ simulation_horizon: 21, simulation_path_count: 200 })
  await expect(page.getByText(/200 条模拟路径当天的净值落点/)).toBeVisible()
  await page.getByLabel('分析样本区间').selectOption('1M')
  await expect(page.getByRole('button', { name: '运行模拟', exact: true })).toBeDisabled()
  await page.getByRole('tab', { name: '收益统计', exact: true }).click()
  await expect(page.getByText(/要求产品完整覆盖所选区间/).first()).toBeVisible()

  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth)
  expect(overflow).toBeLessThanOrEqual(1)
})

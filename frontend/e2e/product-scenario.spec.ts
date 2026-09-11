import { expect, test, type Page } from '@playwright/test'

// Explicit UI contract fixtures. Numerical correctness is verified in backend
// kernel tests; these examples are not market observations or investment results.
type AnalysisRequest = {
  statistics_period: string
  include_simulation?: boolean
  simulation_horizon: number
  simulation_path_count: number
  bootstrap_block_length: number
  fhs_ewma_lambda: number
  simulation_target_return: number
  regime?: { run_id: string; publication_id: string; state_id?: string; segment_id?: string }
}

const execution = {
  execution_backend: 'numba_njit_fixed_signature', nopython: true, njit_required: true,
  object_mode: 0, python_fallback: 0, request_time_compilation: 0,
  kernel_signatures: { ui_contract_fixture: ['fixed'] }, engine: 'ui-contract-fixture',
  kernel_version: 'ui-contract-fixture', kernel_coverage: '1/1', kernel_fingerprint: 'fixture-only',
}
const dates = Array.from({ length: 60 }, (_, index) => new Date(Date.UTC(2026, 0, 2 + index)).toISOString().slice(0, 10))
const stateDefinitions = [
  { id: 'bull', label: '上涨', color: '#059669' },
  { id: 'bear', label: '下跌', color: '#e11d48' },
  { id: 'range', label: '震荡', color: '#d97706' },
]
const segments = [
  { id: 'segment-0', stateId: 'bull', start: 0, end: 29, cumulativeReturn: 0.065, maxDrawdown: -0.012 },
  { id: 'segment-1', stateId: 'bear', start: 30, end: 44, cumulativeReturn: -0.042, maxDrawdown: -0.051 },
  { id: 'segment-2', stateId: 'bull', start: 45, end: 57, cumulativeReturn: 0.021, maxDrawdown: -0.008 },
  { id: 'segment-3', stateId: 'bear', start: 58, end: 58, cumulativeReturn: null, maxDrawdown: null },
  { id: 'segment-4', stateId: 'range', start: 59, end: 59, cumulativeReturn: null, maxDrawdown: null },
].map(({ start, end, ...segment }) => {
  const state = stateDefinitions.find((item) => item.id === segment.stateId)!
  return {
    ...segment, stateLabel: state.label, color: state.color, startDate: dates[start], endDate: dates[end],
    observations: end - start + 1, returnObservations: end - start,
    status: end === start ? 'insufficient_sample' : 'complete',
    reason: end === start ? '仅有 1 个观测值，区间收益与回撤样本不足。' : null,
    windowClipped: false,
  }
})
const states = stateDefinitions.map((state) => {
  const matching = segments.filter((segment) => segment.stateId === state.id)
  const valid = matching.filter((segment) => segment.returnObservations > 0)
  return {
    stateId: state.id, stateLabel: state.label, color: state.color,
    observations: matching.reduce((sum, segment) => sum + segment.observations, 0),
    returnObservations: matching.reduce((sum, segment) => sum + segment.returnObservations, 0),
    segmentCount: matching.length, medianSegmentObservations: state.id === 'bull' ? 21.5 : state.id === 'bear' ? 8 : 1,
    eligibleSegmentCount: valid.length, meanDailyReturn: valid.length ? 0.0005 : null,
    annualizedVolatility: valid.length ? 0.14 : null, winRate: valid.length ? 0.55 : null,
    medianSegmentReturn: state.id === 'bull' ? 0.043 : state.id === 'bear' ? -0.042 : null,
    worstSegmentReturn: state.id === 'bull' ? 0.021 : state.id === 'bear' ? -0.042 : null,
    medianSegmentDrawdown: state.id === 'bull' ? -0.01 : state.id === 'bear' ? -0.051 : null,
    worstSegmentDrawdown: state.id === 'bull' ? -0.012 : state.id === 'bear' ? -0.051 : null,
  }
})
const run = {
  id: 'run-published', name: '沪深300市场周期 · 验收样本', definition_id: 'market-cycle', definition_revision: 3,
  immutable: true, content_hash: 'fixture-hash', definition_source: 'saved_version', mode: 'retrospective',
  created_at: '2026-03-03T00:00:00Z', states: stateDefinitions, series: [],
  segments: segments.map((segment) => ({
    state_id: segment.stateId, start_date: segment.startDate, end_date: segment.endDate,
    duration_observations: segment.observations, return: 0, confidence: 1, reasons: [],
  })),
  conditional_stats: [], transition: { states: [], counts: [], probabilities: [] },
  causality: { is_causal: false, uses_future_data: true, repaints: true, realtime_eligible: false, warnings: [], blockers: [] },
  stability: {}, walk_forward: {}, diagnostics: [], calculation_audit: execution,
  publications: [{ id: 'publication-product', run_id: 'run-published', definition_revision: 3,
    run_content_hash: 'fixture-hash', usage: 'product_research', published_at: '2026-03-03T00:00:00Z' }],
}

function simulation(request: AnalysisRequest, sampleCount: number) {
  const methods = ['gaussian', 'parametric', 'block_bootstrap', 'fhs_ewma', 'fhs_garch'] as const
  const make = (method: (typeof methods)[number]) => ({
    method, methodLabel: method === 'parametric' ? '参数化蒙特卡洛（偏度/峰度校准）' : '历史区块 Bootstrap',
    days: [0, request.simulation_horizon], samplePaths: [[1, 0.96], [1, 1.08]],
    percentiles: { p05: [1, 0.88], p25: [1, 0.97], p50: [1, 1.04], p75: [1, 1.1], p95: [1, 1.2] },
    terminal: { p05: 0.88, p25: 0.97, p50: 1.04, p75: 1.1, p95: 1.2, lossProbability: 0.35,
      valueAtRisk95: 0.12, conditionalValueAtRisk95: 0.15, targetHitProbability: 0.4,
      averageMaxDrawdown: 0.08, p05Return: -0.12, medianReturn: 0.04 },
    assumptions: { sourceObservationCount: sampleCount, targetReturnPercent: request.simulation_target_return,
      meanDailyLogReturn: 0.0005, dailyLogVolatility: 0.008, historicalLogSkewness: -0.4,
      historicalLogExcessKurtosis: 1.2, fittedLogSkewness: -0.38, fittedLogExcessKurtosis: 1.1,
      shapeCalibrationStatus: 'matched', shapeSkewParameter: 0.1, tailWeightParameter: 0.9,
      averageBlockLength: method === 'block_bootstrap' ? request.bootstrap_block_length : null,
      conditionalVolatilityStart: method.startsWith('fhs') ? 0.011 : null,
      volatilityPersistence: method.startsWith('fhs') ? 0.97 : null,
      garchOmega: method === 'fhs_garch' ? 2e-6 : method === 'fhs_ewma' ? 0 : null,
      garchAlpha: method.startsWith('fhs') ? 0.08 : null,
      garchBeta: method.startsWith('fhs') ? 0.89 : null,
      ewmaLambda: method === 'fhs_ewma' ? request.fhs_ewma_lambda : null,
      residualSkewness: method.startsWith('fhs') ? -0.2 : null,
      residualExcessKurtosis: method.startsWith('fhs') ? 1.1 : null },
  })
  const pathCount = request.simulation_path_count
  const horizon = request.simulation_horizon
  const density = {
    sampleSize: pathCount,
    navAxisMin: 0.85,
    navAxisMax: 1.2,
  frames: [1, 2, 3, 4].map((step) => ({
    day: Math.round((horizon * step) / 4),
    navLow: 1 - 0.05 * step,
    navHigh: 1 + 0.05 * step,
    binWidth: (0.1 * step) / 4,
    countAxisMax: pathCount,
    curve: [1, 10, 25, 10, 1],
    bins: [pathCount * 0.1, pathCount * 0.4, pathCount * 0.4, pathCount * 0.1],
  })),
  }
  return { initialNav: 1, methods: [...methods],
    byMethod: Object.fromEntries(methods.map((method) => [method, make(method)])),
    realized: null, realizedStatus: 'off',
    comparison: { p05ReturnGap: 0, medianReturnGap: 0, lossProbabilityGap: 0, conditionalValueAtRiskGap: 0, level: 'low', message: '固定验收样本：各模型差异较小。' },
    densities: Object.fromEntries(methods.map((method) => [method, density])) }
}

function analysis(request: AnalysisRequest) {
  const selectedStateId = request.regime?.state_id ?? null
  const selectedSegmentId = request.regime?.segment_id ?? null
  const selected = segments.filter((segment) => (!selectedStateId || segment.stateId === selectedStateId) && (!selectedSegmentId || segment.id === selectedSegmentId))
  const scoped = Boolean(selectedStateId || selectedSegmentId)
  const observationCount = scoped ? selected.reduce((sum, item) => sum + item.observations, 0) : 60
  const returnCount = scoped ? selected.reduce((sum, item) => sum + item.returnObservations, 0) : 59
  const eligible = returnCount >= 20
  const returnDates = scoped
    ? selected.flatMap(segment => dates.filter(date => date > segment.startDate && date <= segment.endDate))
    : dates.slice(1)
  const dailyReturns = returnDates.map((date, index) => ({ date, return: index % 3 ? 0.2 : -0.25 }))
  const stats = { mean: returnCount ? 0.05 : null, std: returnCount ? 0.2 : null, median: returnCount ? 0.2 : null,
    positiveRatio: returnCount ? 0.65 : null, best: returnCount ? 0.2 : null, worst: returnCount ? -0.25 : null,
    sampleSize: returnCount, skewness: returnCount > 2 ? -0.4 : null, kurtosis: returnCount > 3 ? 1.2 : null,
    jbStatistic: returnCount > 3 ? 2.4 : null, normalityPValue: returnCount > 3 ? 0.3 : null }
  return {
    schema_version: 2, product_id: '510300.SH', execution,
    window: { complete: true, requested_start_date: null, message: null },
    technical: { availability: { ohlc: true, volume: true, kdj: true }, priceMa: {}, volumeMa: {},
      bollinger: { upper: [], middle: [], lower: [] }, kdj: { kValues: [], dValues: [], jValues: [] } },
    dailyReturns, returnStatistics: stats,
    interpretation: { skewness: { label: '轻度左偏', meaning: '固定验收样本，观察收益分布左尾。' },
      kurtosis: { label: '尖峰厚尾', meaning: '固定验收样本，观察尾部损失。' }, normality: '固定验收样本，无实际市场结论。' },
    histogram: returnCount ? [{ start: -0.4, end: 0, center: -0.2, count: Math.ceil(returnCount / 3), normalPdfCount: returnCount * 0.3, frequency: 0.34 }, { start: 0, end: 0.4, center: 0.2, count: returnCount - Math.ceil(returnCount / 3), normalPdfCount: returnCount * 0.65, frequency: 0.66 }] : [],
    boxPlot: returnCount < 5 ? null : {
      stats: [-0.25, -0.25, 0.2, 0.2, 0.2, 0.45], outliers: [],
      quartiles: { q1: -0.25, median: 0.2, q3: 0.2, iqr: 0.45 }, whiskers: { lower: -0.25, upper: 0.2 },
    },
    normalQq: returnCount < 3 ? null : {
      sampleSize: returnCount,
      points: [
        { percentile: 0.05, theoreticalQuantile: -1.64, observedReturn: -0.25, referenceReturn: -0.3, tail: 'lower' },
        { percentile: 0.5, theoreticalQuantile: 0, observedReturn: 0.2, referenceReturn: 0.05, tail: 'center' },
        { percentile: 0.95, theoreticalQuantile: 1.64, observedReturn: 0.2, referenceReturn: 0.4, tail: 'upper' },
      ], keyPoints: [],
    },
    simulation: request.include_simulation && eligible ? simulation(request, returnCount) : null,
    simulationStatus: !request.include_simulation ? 'not_requested' : eligible ? 'complete' : 'insufficient_sample',
    researchContext: { startDate: scoped ? selected[0]?.startDate : dates[0], endDate: scoped ? selected.at(-1)?.endDate : dates.at(-1),
      windowStartDate: dates[0], windowEndDate: dates.at(-1), observations: observationCount, returnObservations: returnCount, segmentCount: scoped ? selected.length : 1,
      scope: selectedSegmentId ? 'segment' : selectedStateId ? 'state' : 'full',
      stateLabel: stateDefinitions.find((state) => state.id === selectedStateId)?.label,
      boundaryPolicy: '仅使用同一连续区间内相邻有效观察值，跨状态边界与缺失价格不连接。', simulationEligible: eligible, analysisBasis: 'adjusted_nav', basisLabel: '复权净值', observationFrequency: 'nav_observations', warnings: [],
      simulationMessage: eligible ? null : '有效收益样本不足 20 个，无法运行条件模拟。' },
    regimeAnalysis: request.regime ? { states, segments, selectedStateId, selectedSegmentId } : null,
  }
}

async function mockApi(page: Page) {
  const requests: AnalysisRequest[] = []
  const pageErrors: string[] = []
  page.on('pageerror', (error) => pageErrors.push(error.message))
  await page.route('**/api/**', async (route) => {
    const path = new URL(route.request().url()).pathname
    if (path === '/api/instruments/products/510300.SH') return route.fulfill({ json: {
      product_id: '510300.SH', name: '沪深300ETF · 固定验收样本', management: '演示管理人', status: '上市',
      base_info: { ts_code: '510300.SH', fund_type: 'ETF', list_date: '2012-05-28' },
      metrics: { issue_amount: 100, m_fee: 0.5, c_fee: 0.1 },
      timeseries: dates.map((date, index) => ({ date, open: 3 + index * 0.001, close: 3 + index * 0.002,
        high: 3.02 + index * 0.002, low: 2.98 + index * 0.001, volume: 1000 + index * 10 })),
    } })
    if (path === '/api/historical-regimes/runs') return route.fulfill({ json: { items: [
      { ...run, segments: undefined, series: undefined, series_included: false },
      { ...run, id: 'run-draft', name: '未发布草稿', immutable: false, publications: [] },
      { ...run, id: 'run-wrong-usage', name: '仅战术配置可用', publications: [{ ...run.publications[0], run_id: 'run-wrong-usage', usage: 'taa' }] },
    ] } })
    if (path === '/api/historical-regimes/runs/run-published') return route.fulfill({ json: run })
    if (path === '/api/instruments/products/510300.SH/analysis') {
      const request = route.request().postDataJSON() as AnalysisRequest
      requests.push(request)
      return route.fulfill({ json: analysis(request) })
    }
    if (path === '/api/custom-indicators/meta') return route.fulfill({ json: { periods: [{ value: '1Y', label: '近 1 年' }] } })
    if (path === '/api/custom-indicators') return route.fulfill({ json: { items: [], total: 0 } })
    if (path === '/api/custom-indicators/evaluate-series') return route.fulfill({ json: { results: [], execution } })
    return route.fulfill({ status: 404, json: {} })
  })
  return { requests, pageErrors }
}

async function noPageOverflow(page: Page) {
  expect(await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth)).toBeLessThanOrEqual(1)
}

test('情景上下文、碎片明细与显式模拟在三档宽度保持一致', async ({ page }, testInfo) => {
  const { requests, pageErrors } = await mockApi(page)
  await page.goto('/product-research/products/510300.SH?kind=etf')
  await expect(page.getByRole('heading', { name: '沪深300ETF · 固定验收样本', exact: true })).toBeVisible()
  await expect(page.getByRole('tab', { name: '走势与指标', exact: true })).toHaveAttribute('aria-selected', 'true')
  await expect.poll(() => requests.length).toBeGreaterThan(0)
  expect(requests.every((request) => request.include_simulation === false)).toBe(true)
  await noPageOverflow(page)
  await page.evaluate(() => window.scrollTo(0, 0))
  // ECharts globalDefault uses a 1000 ms entrance animation; capture its completed frame.
  await page.waitForTimeout(1100)
  await page.screenshot({ path: `../output/product-scenario/${testInfo.project.name}-overview.png`, fullPage: true })

  await expect(page.getByLabel('分析样本区间')).not.toBeVisible()
  await page.getByRole('tab', { name: '情景表现', exact: true }).click()
  const scheme = page.getByLabel('历史情景背景')
  await expect(scheme).toBeEnabled()
  await expect(scheme.getByRole('option', { name: /未发布草稿|仅战术配置可用/ })).toHaveCount(0)
  await scheme.selectOption('run-published')
  await expect(page.getByLabel('市场状态')).toBeEnabled()
  await page.getByRole('tab', { name: '情景表现', exact: true }).click()
  await expect(page.getByRole('table', { name: '产品历史情景表现' })).toBeVisible()
  await expect.poll(() => requests.at(-1)?.regime?.publication_id).toBe('publication-product')
  expect(requests.at(-1)?.regime?.state_id).toBeFalsy()
  await noPageOverflow(page)
  await page.evaluate(() => window.scrollTo(0, 0))
  // ECharts globalDefault uses a 1000 ms entrance animation; capture its completed frame.
  await page.waitForTimeout(1100)
  await page.screenshot({ path: `../output/product-scenario/${testInfo.project.name}-scenario.png`, fullPage: true })

  await page.getByRole('table', { name: '产品历史情景表现' }).getByRole('button', { name: '上涨', exact: true }).click()
  await expect(page.getByLabel('市场状态')).toHaveValue('bull')
  await expect(page.getByText(/上涨 · 全部连续区间 · 41 个有效收益 · 2 段/)).toBeVisible()
  await expect.poll(() => requests.at(-1)?.regime?.state_id).toBe('bull')
  await page.getByRole('tab', { name: '收益统计', exact: true }).click()
  await noPageOverflow(page)
  await page.evaluate(() => window.scrollTo(0, 0))
  // ECharts globalDefault uses a 1000 ms entrance animation; capture its completed frame.
  await page.waitForTimeout(1100)
  await page.screenshot({ path: `../output/product-scenario/${testInfo.project.name}-statistics.png`, fullPage: true })
  await page.getByRole('tab', { name: '未来模拟', exact: true }).click()
  const chart = page.getByLabel('参数化蒙特卡洛（偏度/峰度校准）：路径与期末净值概率分布组合图')
  await expect(chart).not.toBeVisible()
  expect(requests.filter((request) => request.include_simulation)).toHaveLength(0)
  await page.getByRole('button', { name: '运行模拟', exact: true }).click()
  await expect(chart).toBeVisible()
  const runRequests = requests.filter((request) => request.include_simulation)
  expect(runRequests).toHaveLength(1)
  expect(runRequests[0].regime).toEqual({ run_id: 'run-published', publication_id: 'publication-product', state_id: 'bull' })
  await noPageOverflow(page)
  await page.evaluate(() => window.scrollTo(0, 0))
  // ECharts globalDefault uses a 1000 ms entrance animation; capture its completed frame.
  await page.waitForTimeout(1100)
  await page.screenshot({ path: `../output/product-scenario/${testInfo.project.name}-simulation.png`, fullPage: true })

  await page.getByLabel('模拟路径数').selectOption('200')
  await expect(chart).not.toBeVisible()
  expect(requests.filter((request) => request.include_simulation)).toHaveLength(1)
  await page.getByRole('button', { name: '运行模拟', exact: true }).click()
  await expect(chart).toBeVisible()
  expect(requests.filter((request) => request.include_simulation).at(-1)?.simulation_path_count).toBe(200)

  await page.getByLabel('市场状态').selectOption('bear')
  await expect(chart).not.toBeVisible()
  await expect(page.getByRole('button', { name: '运行模拟', exact: true })).toBeDisabled()
  await expect(page.getByText(/有效收益样本不足 20 个/).first()).toBeVisible()
  await page.getByRole('tab', { name: '情景表现', exact: true }).click()
  await expect(page.getByRole('table', { name: '情景连续区间明细' })).toBeVisible()
  await page.getByRole('button', { name: `查看区间 ${dates[58]} 至 ${dates[58]}`, exact: true }).click()
  await expect(page.getByLabel('连续区间', { exact: true })).toHaveValue('segment-3')
  await expect.poll(() => requests.at(-1)?.regime?.segment_id).toBe('segment-3')
  await expect(page.getByText(/仅有 1 个观测值/).first()).toBeVisible()
  await page.getByRole('tab', { name: '未来模拟', exact: true }).click()
  await expect(page.getByRole('button', { name: '运行模拟', exact: true })).toBeDisabled()
  expect(requests.filter((request) => request.include_simulation)).toHaveLength(2)
  expect(pageErrors).toEqual([])
  await noPageOverflow(page)
})

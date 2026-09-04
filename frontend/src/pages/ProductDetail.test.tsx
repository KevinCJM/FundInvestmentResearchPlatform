import { act, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import ProductDetail from './ProductDetail'
import { evaluateCustomIndicators, getCustomIndicatorMeta, listCustomIndicators } from '../services/customIndicators'
import type { IndicatorDefinition } from '../services/customIndicators'
import { getHistoricalRegimeRun, listHistoricalRegimeRuns } from '../services/historicalRegimes'
import type { HistoricalRegimeRun } from '../services/historicalRegimes'
import { analyzeProduct } from '../services/productAnalysis'
import type {
  FuturePathSimulation,
  ProductAnalysisRequest,
  ProductAnalysisResponse,
  SimulationMethod,
  TerminalNavDensity,
} from '../services/productAnalysis'

vi.mock('echarts-for-react', () => ({
  default: ({ option }: {
    option?: {
      grid?: unknown | Array<{ left?: string }>;
      xAxis?: { type?: string; name?: string } | Array<{ type?: string; name?: string }>;
      yAxis?: { type?: string } | Array<{ type?: string }>;
      series?: Array<{ name?: string; data?: unknown[]; markArea?: { data?: unknown[] } }>;
    };
  }) => {
    const xAxis = Array.isArray(option?.xAxis) ? option.xAxis[0] : option?.xAxis
    const yAxis = Array.isArray(option?.yAxis) ? option.yAxis[0] : option?.yAxis
    const terminalXAxis = Array.isArray(option?.xAxis) ? option.xAxis[1] : undefined
    const grids = Array.isArray(option?.grid) ? option.grid : option?.grid ? [option.grid] : []
    const terminalHistogram = (option?.series ?? []).find((series) => series.name === '期末净值直方图')
    const medianPath = (option?.series ?? []).find((series) => series.name === '中位路径')
    const priceSeries = (option?.series ?? []).find((series) => series.name === '价格')
    const terminalHistogramTotal = ((terminalHistogram?.data ?? []) as unknown[]).reduce<number>((sum, item) => {
      const count = Array.isArray(item) ? Number(item[0]) : 0
      return sum + (Number.isFinite(count) ? count : 0)
    }, 0)
    return <div
      data-testid="chart"
      data-series={(option?.series ?? []).map((series) => series.name).filter(Boolean).join(',')}
      data-x-axis-type={xAxis?.type}
      data-y-axis-type={yAxis?.type}
      data-grid-count={grids.length}
      data-terminal-grid-left={(grids[1] as { left?: string } | undefined)?.left}
      data-terminal-axis-name={terminalXAxis?.name}
      data-terminal-histogram-total={terminalHistogramTotal}
      data-simulation-start={medianPath ? String(Number(medianPath.data?.[0])) : undefined}
      data-regime-mark-area={JSON.stringify(priceSeries?.markArea?.data ?? [])}
    />
  },
}))
vi.mock('../services/customIndicators', () => ({
  evaluateCustomIndicators: vi.fn(),
  getCustomIndicatorMeta: vi.fn(),
  indicatorPeriodLabel: (period: string) => period,
  indicatorsForContext: (items: IndicatorDefinition[], context: string) => items.filter((item) => (item.context_kind ?? 'single_product') === context),
  listCustomIndicators: vi.fn(),
}))
vi.mock('../services/historicalRegimes', () => ({
  getHistoricalRegimeRun: vi.fn(),
  listHistoricalRegimeRuns: vi.fn(),
}))
vi.mock('../services/productAnalysis', async () => {
  const actual = await vi.importActual<typeof import('../services/productAnalysis')>('../services/productAnalysis')
  return { ...actual, analyzeProduct: vi.fn() }
})

const fixedIndicatorExecution = {
  execution_backend: 'numba_njit_fixed_signature',
  nopython: true,
  object_mode: 0,
  python_fallback: 0,
  request_time_compilation: 0,
  kernel_signatures: { typed_indicator_plan: ['fixed'] },
}

const makeSimulation = (
  method: SimulationMethod,
  pathCount: number,
  horizon: number,
  targetReturn: number,
  blockLength: number,
): FuturePathSimulation => ({
  method,
  methodLabel: method === 'parametric' ? '参数化蒙特卡洛（偏度/峰度校准）' : '历史区块 Bootstrap',
  days: [0, horizon],
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
    targetReturnPercent: targetReturn,
    meanDailyLogReturn: method === 'parametric' ? 0.0002 : null,
    dailyLogVolatility: method === 'parametric' ? 0.008 : null,
    historicalLogSkewness: method === 'parametric' ? -0.2 : null,
    historicalLogExcessKurtosis: method === 'parametric' ? 0.8 : null,
    fittedLogSkewness: method === 'parametric' ? -0.18 : null,
    fittedLogExcessKurtosis: method === 'parametric' ? 0.76 : null,
    shapeCalibrationStatus: method === 'parametric' ? 'matched' : null,
    shapeSkewParameter: method === 'parametric' ? 0.1 : null,
    tailWeightParameter: method === 'parametric' ? 1.1 : null,
    averageBlockLength: method === 'block_bootstrap' ? blockLength : null,
  },
})

const makeDensity = (pathCount: number): TerminalNavDensity => ({
  sampleSize: pathCount,
  points: [
    { nav: 0.9, density: 0.2, estimatedCount: 10, simulatedReturn: -0.1 },
    { nav: 1.0, density: 0.5, estimatedCount: 25, simulatedReturn: 0 },
    { nav: 1.1, density: 0.2, estimatedCount: 10, simulatedReturn: 0.1 },
  ],
  histogram: [{ lowerNav: 0.9, upperNav: 1.1, density: 1, count: pathCount, frequency: 1 }],
  maxDensity: 1,
  modeNav: 1,
  minNav: 0.9,
  maxNav: 1.1,
  countAxisMax: pathCount,
  navAxisMin: 0.88,
  navAxisMax: 1.12,
  densityCountFactor: pathCount * 0.2,
  histogramBinWidth: 0.2,
})

const makeAnalysisResponse = (request: ProductAnalysisRequest): ProductAnalysisResponse => {
  const incomplete = request.statistics_period === '1M'
  const returns = incomplete
    ? []
    : Array.from({ length: 24 }, (_, index) => ({ date: `2026-01-${String(index + 2).padStart(2, '0')}`, return: index % 2 ? 0.2 : -0.1 }))
  const parametric = makeSimulation('parametric', request.simulation_path_count, request.simulation_horizon, request.simulation_target_return, request.bootstrap_block_length)
  const bootstrap = makeSimulation('block_bootstrap', request.simulation_path_count, request.simulation_horizon, request.simulation_target_return, request.bootstrap_block_length)
  const qqPoints = [
    { percentile: 0.05, theoreticalQuantile: -1.64, observedReturn: -0.3, referenceReturn: -0.25, tail: 'lower' as const },
    { percentile: 0.5, theoreticalQuantile: 0, observedReturn: 0.05, referenceReturn: 0.05, tail: 'center' as const },
    { percentile: 0.95, theoreticalQuantile: 1.64, observedReturn: 0.4, referenceReturn: 0.35, tail: 'upper' as const },
  ]
  return {
    schema_version: 1,
    product_id: '510300.SH',
    execution: {
      execution_backend: 'numba_njit_fixed_signature', engine: 'product-analysis-njit-1.0.0',
      kernel_version: 'product-chart-statistics-simulation-2', kernel_coverage: '21/21', kernel_fingerprint: '1234567890abcdef',
      kernel_signatures: { product_analysis_kernel: ['fixed'] }, nopython: true, njit_required: true, object_mode: 0, python_fallback: 0,
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
    simulation: incomplete ? null : {
      initialNav: 1, parametric, blockBootstrap: bootstrap,
      comparison: { p05ReturnGap: 0.01, medianReturnGap: 0.01, lossProbabilityGap: 0.02, conditionalValueAtRiskGap: 0.01, level: 'low', message: '两种模型结果接近。' },
      densities: { parametric: makeDensity(request.simulation_path_count), block_bootstrap: makeDensity(request.simulation_path_count) },
    },
    regimeStatistics: request.regime ? [
      { stateId: 'bull', stateLabel: '牛市', color: '#16a34a', observations: 5, returnObservations: 4, cumulativeReturn: 0.02, annualizedVolatility: 0.12, maxDrawdown: -0.03, winRate: 0.5 },
      { stateId: 'range', stateLabel: '震荡', color: '#f59e0b', observations: 5, returnObservations: 4, cumulativeReturn: 0.01, annualizedVolatility: 0.08, maxDrawdown: -0.01, winRate: 0.5 },
    ] : [],
  }
}

const percentIndicator: IndicatorDefinition = {
  id: 'total-return', revision: 2, source: 'custom', read_only: false,
  name: '区间累计收益', description: '基于真实净值的累计收益。', expression: '\\left(\\prod\\left(\\mathbf{r}+1\\right)\\right)-1',
  periods: ['1Y'], unit: '%', display_format: 'percent', precision: 2,
  direction: 'higher_better', annual_risk_free_rate_percent: 1.5,
  created_at: '2026-01-01T00:00:00Z', updated_at: '2026-01-01T00:00:00Z',
}

const portfolioIndicator: IndicatorDefinition = {
  ...percentIndicator,
  id: 'portfolio-volatility',
  name: '组合波动率',
  context_kind: 'portfolio',
}

const makeHistoricalRegimeRun = (
  id: string,
  name: string,
  options: {
    immutable?: boolean;
    usage?: HistoricalRegimeRun['publications'][number]['usage'];
    publicationRunId?: string;
    mode?: HistoricalRegimeRun['mode'];
  } = {},
): HistoricalRegimeRun => ({
  id,
  definition_id: 'definition-market-cycle',
  definition_revision: 3,
  definition_source: 'saved_version',
  name,
  mode: options.mode ?? 'realtime',
  created_at: '2026-02-01T08:00:00Z',
  immutable: options.immutable ?? true,
  states: [
    { id: 'bull', label: '牛市', color: '#16a34a' },
    { id: 'range', label: '震荡', color: '#f59e0b' },
  ],
  series: [],
  segments: [
    {
      state_id: 'bull', state_label: '旧标签不应使用', start_date: '2025-12-20', end_date: '2026-01-05',
      duration_observations: 10, return: 0.08, confidence: 0.88, reasons: [],
    },
    {
      state_id: 'range', state_label: '旧标签不应使用', start_date: '2026-01-06', end_date: '2026-01-10',
      duration_observations: 5, return: 0.01, confidence: 0.7, reasons: [],
    },
  ],
  conditional_stats: [],
  transition: { states: [], counts: [], probabilities: [] },
  causality: {
    classification: 'causal', is_causal: true, uses_future_data: false, repaints: false,
    realtime_eligible: true, publish_eligible_usages: ['product_research'], blockers: [], warnings: [],
  },
  stability: {},
  walk_forward: {},
  diagnostics: [],
  publications: [{
    id: `publication-${id}`,
    usage: options.usage ?? 'product_research',
    published_at: '2026-02-02T08:00:00Z',
    definition_revision: 3,
    run_id: options.publicationRunId ?? id,
  }],
})

describe('ProductDetail custom indicators', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    window.localStorage.clear()
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        product_id: '510300.SH', name: '沪深300ETF', management: '测试管理人', status: '上市',
        base_info: { ts_code: '510300.SH', fund_type: 'ETF', list_date: '2012-05-28', delist_date: '2026-12-31' },
        metrics: {
          issue_amount: 100,
          current_size: 123456.78,
          current_size_as_of: '2026-06-30',
          current_size_source: 'instrument_metrics_snapshot',
          current_share: 100000,
          current_unit_nav: 1.2345678,
          m_fee: 0.5,
          c_fee: 0.1,
        },
        timeseries: Array.from({ length: 25 }, (_, index) => {
          const close = 3 + index * 0.002 + ((index % 5) - 2) * 0.005
          return {
            date: new Date(Date.UTC(2026, 0, 2 + index)).toISOString().slice(0, 10),
            open: close - 0.002,
            high: close + 0.01,
            low: close - 0.01,
            close,
            volume: 1000 + index * 10,
          }
        }),
      }),
    }))
    vi.mocked(listCustomIndicators).mockResolvedValue({ items: [percentIndicator, portfolioIndicator], total: 2 })
    vi.mocked(getCustomIndicatorMeta).mockResolvedValue({ periods: [{ value: '1M', label: '近 1 月', description: '运行周期' }, { value: '1Y', label: '近 1 年', description: '运行周期' }] } as any)
    vi.mocked(evaluateCustomIndicators).mockResolvedValue({
      results: [{
        indicator_id: percentIndicator.id, indicator_revision: percentIndicator.revision, indicator_name: percentIndicator.name,
        target: { kind: 'etf', product_id: '510300.SH', name: '沪深300ETF' }, period: '1Y', value: 0.1234, status: 'ok', warnings: [],
        window: { requested_as_of: null, effective_as_of: '2026-01-06', start_date: '2025-01-06', end_date: '2026-01-06', observation_count: 250, data_latest_date: '2026-01-06' },
        presentation: {
          indicator_id: percentIndicator.id, revision: percentIndicator.revision, name: percentIndicator.name,
          source: 'custom', category: 'return', category_label: '收益', context_kind: 'single_product',
          catalog_status: 'current', display_format: 'percent', precision: 2, unit: '%', notation: 'standard',
          value_scale: 100, output_measure: 'return', direction: 'higher_better', description: percentIndicator.description,
          methodology: '测试', data_basis: '真实净值', minimum_observations: 2, applicable_product_kinds: ['etf', 'fund'],
        },
      }], summary: { total: 1, ok: 1, warning: 0, error: 0 }, cache: { hits: 0, misses: 1 },
      execution: fixedIndicatorExecution,
    })
    vi.mocked(listHistoricalRegimeRuns).mockResolvedValue([])
    vi.mocked(getHistoricalRegimeRun).mockReset()
    vi.mocked(analyzeProduct).mockImplementation(async (_productId, _kind, request) => makeAnalysisResponse(request))
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('展示已保存指标的当前值、百分比格式并可进入指标中心', async () => {
    const user = userEvent.setup()
    render(<MemoryRouter initialEntries={['/product/510300.SH?kind=etf']}><Routes><Route path="/product/:productId" element={<ProductDetail />} /></Routes></MemoryRouter>)

    expect(await screen.findByText('自定义研究指标')).toBeInTheDocument()
    expect(await screen.findByText('高性能计算已验证')).toBeInTheDocument()
    expect(screen.getByTestId('product-analysis-execution')).toHaveTextContent('内核覆盖 21/21')
    expect(screen.getByTestId('product-analysis-execution')).toHaveTextContent('Python 回退 0')
    await waitFor(() => expect(fetch).toHaveBeenCalledWith(
      '/api/instruments/products/510300.SH?kind=etf',
      expect.objectContaining({ signal: expect.anything() }),
    ))
    expect(await screen.findByText('12.34%')).toBeInTheDocument()
    expect(screen.getByLabelText('上市日期：2012-05-28')).toBeInTheDocument()
    expect(screen.getByLabelText('退市日期：2026-12-31')).toBeInTheDocument()
    expect(screen.getByText('当前规模')).toBeInTheDocument()
    expect(screen.getByText('12.35 亿')).toBeInTheDocument()
    expect(screen.getByText('快照截至 2026-06-30 · 100,000 万份 × 1.23 元/份')).toBeInTheDocument()
    expect(screen.getByLabelText('统计区间')).toHaveValue('ALL')
    expect(screen.getByRole('heading', { name: '未来虚拟净值模拟' })).toBeInTheDocument()
    expect(screen.getByText(/所有路径统一从虚拟净值 1\.0000 出发/)).toBeInTheDocument()
    expect(screen.getByLabelText('模拟未来区间')).toHaveValue('252')
    expect(screen.getByLabelText('模拟路径数')).toHaveValue('500')
    expect(screen.getByLabelText('Bootstrap 平均区块长度')).toHaveValue('20')
    expect(screen.getByLabelText('目标期末收益率')).toHaveValue(5)
    expect(screen.getByRole('radio', { name: '参数化蒙特卡洛' })).toBeChecked()
    expect(screen.getByRole('radio', { name: '区块 Bootstrap' })).not.toBeChecked()
    const combinedChart = screen.getByTestId('monte-carlo-combined-chart')
    const combinedChartRenderer = combinedChart.querySelector('[data-testid="chart"]')
    expect(combinedChart).toHaveAccessibleName('参数化蒙特卡洛（偏度/峰度校准）：路径与期末净值概率分布组合图')
    expect(screen.getByText(/历史对数收益偏度/)).toBeInTheDocument()
    expect(screen.getByText(/四矩校准已匹配|四矩近似校准/)).toBeInTheDocument()
    expect(combinedChartRenderer).toHaveAttribute('data-grid-count', '2')
    expect(combinedChartRenderer).toHaveAttribute('data-terminal-grid-left', '83%')
    expect(combinedChartRenderer).toHaveAttribute('data-terminal-axis-name', '路径数')
    expect(combinedChartRenderer).toHaveAttribute('data-terminal-histogram-total', '500')
    expect(combinedChartRenderer).toHaveAttribute('data-simulation-start', '1')
    expect(combinedChartRenderer).toHaveAttribute('data-series', expect.stringContaining('期末净值直方图'))
    expect(combinedChartRenderer).toHaveAttribute('data-series', expect.stringContaining('期末净值概率密度'))
    expect(screen.getByText(/共计 500 条/)).toBeInTheDocument()
    expect(screen.getByText('双模型结果对比')).toBeInTheDocument()
    expect(screen.getByRole('table', { name: '参数化蒙特卡洛与区块 Bootstrap 模拟结果对比' })).toBeInTheDocument()
    expect(screen.getByText('95% CVaR（预期短缺）')).toBeInTheDocument()
    expect(screen.getAllByText('平均最大回撤')).toHaveLength(2)
    expect(screen.getByText('达到 5% 概率')).toBeInTheDocument()
    await user.selectOptions(screen.getByLabelText('模拟路径数'), '200')
    await waitFor(() => expect(combinedChartRenderer).toHaveAttribute('data-terminal-histogram-total', '200'))
    expect(screen.getByText(/共计 200 条/)).toBeInTheDocument()
    await user.click(screen.getByRole('radio', { name: '区块 Bootstrap' }))
    expect(screen.getByRole('radio', { name: '区块 Bootstrap' })).toBeChecked()
    expect(combinedChart).toHaveAccessibleName('历史区块 Bootstrap：路径与期末净值概率分布组合图')
    expect(screen.getByText(/平均区块长度 20 个交易日/)).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: '正态 Q-Q 图' })).toBeInTheDocument()
    expect(screen.getByText('查看关键分位点数据')).toBeInTheDocument()
    expect(screen.getByTestId('distribution-diagnostics-grid')).toHaveClass('lg:grid-cols-2')
    const diagnosticCharts = screen.getAllByTestId('chart')
    const boxPlotChart = diagnosticCharts.find((chart) => chart.dataset.series?.includes('箱形图'))
    const normalQqChart = diagnosticCharts.find((chart) => chart.dataset.series?.includes('正态参考线'))
    expect(boxPlotChart).toHaveAttribute('data-x-axis-type', 'value')
    expect(boxPlotChart).toHaveAttribute('data-y-axis-type', 'category')
    expect(normalQqChart).toHaveAttribute('data-x-axis-type', 'value')
    expect(normalQqChart).toHaveAttribute('data-y-axis-type', 'value')
    expect(screen.queryByText('组合波动率')).not.toBeInTheDocument()
    await waitFor(() => expect(evaluateCustomIndicators).toHaveBeenCalledWith({
      indicator_ids: ['total-return'], targets: [{ kind: 'etf', product_id: '510300.SH' }], period: '1Y',
      as_of: undefined,
    }))
    expect(screen.getByText(/1Y · 2025-01-06 至 2026-01-06 · 250 个观察值/)).toBeInTheDocument()
    expect(screen.getByLabelText('区间累计收益计算区间')).toHaveValue('1Y')
    await user.selectOptions(screen.getByLabelText('区间累计收益计算区间'), '1M')
    await waitFor(() => expect(evaluateCustomIndicators).toHaveBeenCalledWith({
      indicator_ids: ['total-return'], targets: [{ kind: 'etf', product_id: '510300.SH' }], period: '1M',
      as_of: undefined,
    }))
    expect(screen.getByLabelText('区间累计收益计算区间')).toHaveValue('1M')
    expect(screen.getByRole('link', { name: '在指标中心分析' })).toHaveAttribute('href', '/settings/indicators-models?kind=etf&ids=510300.SH')

    await user.click(screen.getByRole('button', { name: '移除指标 区间累计收益' }))
    expect(screen.queryByText('12.34%')).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: /选择研究指标/ })).toHaveTextContent('已选 0/8')
    await waitFor(() => expect(JSON.parse(window.localStorage.getItem('indicator-display:v2:product-detail:single_product') ?? '{}')).toEqual({
      indicatorIds: [],
      periodsByIndicator: {},
    }))

    await user.selectOptions(screen.getByLabelText('模拟未来区间'), '21')
    await user.selectOptions(screen.getByLabelText('模拟路径数'), '200')
    await user.selectOptions(screen.getByLabelText('Bootstrap 平均区块长度'), '10')
    await user.clear(screen.getByLabelText('目标期末收益率'))
    await user.type(screen.getByLabelText('目标期末收益率'), '8')
    await user.click(screen.getByRole('button', { name: '重新模拟' }))
    expect(screen.getByText('200 条虚拟路径中的样本比例')).toBeInTheDocument()
    expect(screen.getByText('达到 8% 概率')).toBeInTheDocument()

    await user.selectOptions(screen.getByLabelText('统计区间'), '1M')
    expect(screen.getByText(/要求产品完整覆盖所选区间/)).toBeInTheDocument()
    expect(screen.queryByRole('heading', { name: '未来虚拟净值模拟' })).not.toBeInTheDocument()
  })

  it('场外公募基金展示成立日期，并仅在披露时展示到期日期', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        product_id: '000001.OF', name: '示例场外基金', management: '测试管理人', status: '存续',
        base_info: { ts_code: '000001.OF', found_date: '20010101', due_date: '2030-12-31' },
        metrics: { issue_amount: 100, m_fee: 0.5, c_fee: 0.1 },
        timeseries: [
          { date: '2026-01-05', open: 1, high: 1, low: 1, close: 1, volume: 0 },
          { date: '2026-01-06', open: 1.01, high: 1.01, low: 1.01, close: 1.01, volume: 0 },
        ],
      }),
    }))

    render(<MemoryRouter initialEntries={['/product/000001.OF?kind=fund']}><Routes><Route path="/product/:productId" element={<ProductDetail />} /></Routes></MemoryRouter>)

    expect(await screen.findByLabelText('成立日期：2001-01-01')).toBeInTheDocument()
    expect(screen.getByLabelText('到期日期：2030-12-31')).toBeInTheDocument()
    expect(screen.queryByText(/上市日期/)).not.toBeInTheDocument()
  })

  it('分析接口失败时显式提示且不执行浏览器本地回退', async () => {
    vi.mocked(analyzeProduct).mockRejectedValue(new Error('NJIT 预热未完成'))

    render(<MemoryRouter initialEntries={['/product/510300.SH?kind=etf']}><Routes><Route path="/product/:productId" element={<ProductDetail />} /></Routes></MemoryRouter>)

    expect(await screen.findByText('产品数值分析未完成')).toBeInTheDocument()
    expect(screen.getByTestId('product-analysis-execution')).toHaveTextContent('NJIT 预热未完成')
    expect(screen.getByTestId('product-analysis-execution')).toHaveTextContent('不会退回浏览器本地计算')
    expect(screen.queryByRole('heading', { name: '未来虚拟净值模拟' })).not.toBeInTheDocument()
  })

  it('原始 OHLC 与成交量缺失时只展示真实净值且不伪造字段', async () => {
    vi.mocked(analyzeProduct).mockImplementation(async (_productId, _kind, request) => {
      const response = makeAnalysisResponse(request)
      response.technical.availability = { ohlc: false, volume: false, kdj: false }
      response.technical.volumeMa = {}
      response.technical.kdj = { kValues: [], dValues: [], jValues: [] }
      return response
    })

    render(<MemoryRouter initialEntries={['/product/510300.SH?kind=etf']}><Routes><Route path="/product/:productId" element={<ProductDetail />} /></Routes></MemoryRouter>)

    expect(await screen.findByText('真实收盘价 / 净值')).toBeInTheDocument()
    expect(screen.getByText('成交量未披露')).toBeInTheDocument()
    const missingFieldNotice = screen.getAllByRole('status').find((element) => (
      element.textContent?.includes('不使用 close 或 0 伪造缺失字段')
    ))
    expect(missingFieldNotice).toHaveTextContent('不使用 close 或 0 伪造缺失字段')
    expect(missingFieldNotice).toHaveTextContent('KDJ、成交量均线会保持不可用')
  })

  it('只展示已发布到产品研究的不可变版本，并将所选区间映射为主价格图背景', async () => {
    const user = userEvent.setup()
    const eligibleDetail = makeHistoricalRegimeRun('run-eligible', '沪深300牛熊震荡')
    const eligibleSummary = { ...eligibleDetail, series: undefined, segments: undefined, series_included: false, series_detail_endpoint: '/api/historical-regimes/runs/run-eligible' } as unknown as HistoricalRegimeRun
    vi.mocked(getHistoricalRegimeRun).mockResolvedValue(eligibleDetail)
    vi.mocked(listHistoricalRegimeRuns).mockResolvedValue([
      eligibleSummary,
      makeHistoricalRegimeRun('run-mutable', '可变试算', { immutable: false }),
      makeHistoricalRegimeRun('run-display-only', '仅研究展示', { usage: 'research_display' }),
      makeHistoricalRegimeRun('run-mismatched-publication', '发布记录错配', { publicationRunId: 'another-run' }),
    ])

    render(<MemoryRouter initialEntries={['/product/510300.SH?kind=etf']}><Routes><Route path="/product/:productId" element={<ProductDetail />} /></Routes></MemoryRouter>)

    const selector = await screen.findByRole('combobox', { name: '历史情景背景' })
    await waitFor(() => expect(selector).toBeEnabled())
    expect(within(selector).getByRole('option', { name: '关闭历史情景背景' })).toBeInTheDocument()
    expect(within(selector).getByRole('option', { name: /沪深300牛熊震荡 · v3 · 实时识别/ })).toBeInTheDocument()
    expect(within(selector).queryByRole('option', { name: /可变试算/ })).not.toBeInTheDocument()
    expect(within(selector).queryByRole('option', { name: /仅研究展示/ })).not.toBeInTheDocument()
    expect(within(selector).queryByRole('option', { name: /发布记录错配/ })).not.toBeInTheDocument()

    await user.selectOptions(selector, 'run-eligible')
    expect(await screen.findByTestId('historical-regime-selection-meta')).toHaveTextContent('不可变运行 · 定义版本 v3')
    expect(getHistoricalRegimeRun).toHaveBeenCalledWith('run-eligible')
    expect(screen.getByTestId('historical-regime-selection-meta')).toHaveTextContent('实时识别：按当时可得信息生成')
    expect(within(screen.getByLabelText('历史情景图例')).getByText('牛市')).toBeInTheDocument()
    const regimePerformance = await screen.findByRole('table', { name: '产品历史情景表现' })
    expect(within(regimePerformance).getByRole('row', { name: /牛市/ })).toHaveTextContent('%')
    expect(within(regimePerformance).getByRole('row', { name: /震荡/ })).toHaveTextContent('%')
    expect(screen.getByText(/跨情景边界收益不会归入任一状态/)).toBeInTheDocument()

    const getMainPriceChart = () => screen.getAllByTestId('chart')
      .find((chart) => chart.dataset.series?.split(',').includes('价格'))
    await waitFor(() => expect(getMainPriceChart()).not.toHaveAttribute('data-regime-mark-area', '[]'))
    expect(JSON.parse(getMainPriceChart()?.getAttribute('data-regime-mark-area') ?? '[]')).toEqual([
      [
        { name: '牛市', xAxis: '2026-01-02', itemStyle: { color: '#16a34a', opacity: 0.12 } },
        { xAxis: '2026-01-05' },
      ],
      [
        { name: '震荡', xAxis: '2026-01-06', itemStyle: { color: '#f59e0b', opacity: 0.12 } },
        { xAxis: '2026-01-10' },
      ],
    ])

    await user.selectOptions(selector, '')
    await waitFor(() => expect(getMainPriceChart()).toHaveAttribute('data-regime-mark-area', '[]'))
    expect(screen.getByText('当前未叠加历史情景背景。')).toBeInTheDocument()
    expect(screen.queryByRole('table', { name: '产品历史情景表现' })).not.toBeInTheDocument()
    expect(listHistoricalRegimeRuns).toHaveBeenCalledTimes(1)
    const regimeAnalysisCall = vi.mocked(analyzeProduct).mock.calls.find(([, , request]) => request.regime?.run_id === 'run-eligible')
    expect(regimeAnalysisCall?.[2].regime).toEqual({ run_id: 'run-eligible', publication_id: 'publication-run-eligible' })
  })

  it('没有合格发布版本时显示明确空态且不伪造图表背景', async () => {
    vi.mocked(listHistoricalRegimeRuns).mockResolvedValue([
      makeHistoricalRegimeRun('run-display-only', '仅研究展示', { usage: 'research_display' }),
      makeHistoricalRegimeRun('run-mutable', '可变试算', { immutable: false }),
    ])

    render(<MemoryRouter initialEntries={['/product/510300.SH?kind=etf']}><Routes><Route path="/product/:productId" element={<ProductDetail />} /></Routes></MemoryRouter>)

    expect(await screen.findByText(/暂无已发布到“产品研究”的不可变历史情景版本/)).toBeInTheDocument()
    expect(screen.getByRole('combobox', { name: '历史情景背景' })).toBeDisabled()
    const mainPriceChart = screen.getAllByTestId('chart')
      .find((chart) => chart.dataset.series?.split(',').includes('价格'))
    expect(mainPriceChart).toHaveAttribute('data-regime-mark-area', '[]')
  })

  it('从手动构建大类进入后返回原来源页', async () => {
    const user = userEvent.setup()
    render(
      <MemoryRouter
        initialEntries={[
          '/manual-construction',
          {
            pathname: '/product/510300.SH',
            search: '?kind=etf',
            state: { returnTo: '/manual-construction', returnLabel: '返回手动构建大类' },
          },
        ]}
        initialIndex={1}
      >
        <Routes>
          <Route path="/manual-construction" element={<div>手动构建大类来源页</div>} />
          <Route path="/product/:productId" element={<ProductDetail />} />
        </Routes>
      </MemoryRouter>,
    )

    await act(async () => {
      await user.click(screen.getByRole('button', { name: /返回手动构建大类/ }))
    })
    expect(await screen.findByText('手动构建大类来源页')).toBeInTheDocument()
  })
})

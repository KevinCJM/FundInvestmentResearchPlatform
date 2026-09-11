import { act, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import ProductDetail from './ProductDetail'
import {
  evaluateCustomIndicators,
  evaluateTimeSeriesIndicators,
  getCustomIndicatorMeta,
  listCustomIndicators,
} from '../services/customIndicators'
import type {
  EvaluateTimeSeriesIndicatorsRequest,
  EvaluateTimeSeriesIndicatorsResponse,
  TimeSeriesIndicatorResult,
  IndicatorDefinition,
} from '../services/customIndicators'
import { getHistoricalRegimeRun, listHistoricalRegimeRuns } from '../services/historicalRegimes'
import type { HistoricalRegimeRun } from '../services/historicalRegimes'
import { analyzeProduct } from '../services/productAnalysis'
import type {
  FuturePathSimulation,
  ProductAnalysisRequest,
  ProductAnalysisResponse,
  RealizedFuturePath,
  RealizedMethodScore,
  SimulationMethod,
  NavDensity,
} from '../services/productAnalysis'

vi.mock('echarts-for-react', () => ({
  default: ({ option, onEvents }: {
    option?: {
      grid?: unknown | Array<{ left?: string }>;
      xAxis?: { type?: string; name?: string } | Array<{ type?: string; name?: string }>;
      yAxis?: { type?: string; min?: number; max?: number } | Array<{ type?: string; min?: number; max?: number }>;
      series?: Array<{ name?: string; data?: unknown[]; markArea?: { data?: unknown[] } }>;
      graphic?: Array<{ style?: { text?: string } }>;
    };
    onEvents?: Record<string, (params: unknown) => void>;
  }) => {
    const xAxis = Array.isArray(option?.xAxis) ? option.xAxis[0] : option?.xAxis
    const yAxis = Array.isArray(option?.yAxis) ? option.yAxis[0] : option?.yAxis
    const terminalXAxis = Array.isArray(option?.xAxis) ? option.xAxis[1] : undefined
    const grids = Array.isArray(option?.grid) ? option.grid : option?.grid ? [option.grid] : []
    const terminalHistogram = (option?.series ?? []).find((series) => series.name === '期末净值直方图')
    const medianPath = (option?.series ?? []).find((series) => series.name === '中位路径')
    const realizedPath = (option?.series ?? []).find((series) => series.name === '实际走势')
    const priceSeries = (option?.series ?? []).find((series) => series.name === '价格')
    const terminalHistogramTotal = ((terminalHistogram?.data ?? []) as unknown[]).reduce<number>((sum, item) => {
      const count = Array.isArray(item) ? Number(item[0]) : 0
      return sum + (Number.isFinite(count) ? count : 0)
    }, 0)
    const densityTitle = (option?.graphic ?? [])[0]?.style?.text
    return <div
      data-testid="chart"
      data-density-title={densityTitle}
      data-series={(option?.series ?? []).map((series) => series.name).filter(Boolean).join(',')}
      data-x-axis-type={xAxis?.type}
      data-y-axis-type={yAxis?.type}
      data-grid-count={grids.length}
      data-terminal-grid-left={(grids[1] as { left?: string } | undefined)?.left}
      data-terminal-axis-name={terminalXAxis?.name}
      data-terminal-histogram-total={terminalHistogramTotal}
      data-simulation-start={medianPath ? String(Number(medianPath.data?.[0])) : undefined}
      data-regime-mark-area={JSON.stringify(priceSeries?.markArea?.data ?? [])}
      data-y-axis-min={yAxis?.min === undefined ? undefined : String(yAxis.min)}
      data-y-axis-max={yAxis?.max === undefined ? undefined : String(yAxis.max)}
      data-realized-length={realizedPath ? String((realizedPath.data ?? []).length) : undefined}
    >
      {onEvents?.datazoom ? (
        <button
          type="button"
          data-testid="drag-zoom-to-half"
          onClick={() => onEvents.datazoom?.({ batch: [{ start: 0, end: 50 }] })}
        />
      ) : null}
    </div>
  },
}))
vi.mock('../services/customIndicators', () => ({
  evaluateCustomIndicators: vi.fn(),
  evaluateTimeSeriesIndicators: vi.fn(),
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

const makeTimeSeriesResponse = (request: EvaluateTimeSeriesIndicatorsRequest): EvaluateTimeSeriesIndicatorsResponse => {
  const dates = Array.from({ length: 25 }, (_, index) => (
    new Date(Date.UTC(2026, 0, 2 + index)).toISOString().slice(0, 10)
  ))
  const outputById: Record<string, Array<{ id: string; label: string }>> = {
    'builtin-close-moving-average-series': [{ id: 'ma', label: '收盘价均线' }],
    'builtin-volume-moving-average-series': [{ id: 'volume_ma', label: '成交量均线' }],
    'builtin-bollinger-bands-series': [
      { id: 'upper', label: '布林上轨' },
      { id: 'middle', label: '布林中轨' },
      { id: 'lower', label: '布林下轨' },
    ],
    'builtin-kdj-series': [
      { id: 'k', label: 'K 值' },
      { id: 'd', label: 'D 值' },
      { id: 'j', label: 'J 值' },
    ],
  }
  const results = request.indicator_instances.map<TimeSeriesIndicatorResult>((instance, instanceIndex) => {
    const indicatorId = instance.indicator_id ?? 'inline-series'
    const channels = (outputById[indicatorId] ?? [{ id: 'value', label: '数值' }]).map(
      (channel, channelIndex) => ({
        ...channel,
        unit: '',
        display_format: 'number' as const,
        precision: 2,
        output_measure: 'dimensionless',
        values: dates.map((_date, index) => (
          index < 2 ? null : instanceIndex * 10 + channelIndex + index / 10
        )),
      }),
    )
    return {
      indicator_id: indicatorId,
      indicator_revision: 1,
      indicator_name: indicatorId,
      result_kind: 'time_series' as const,
      target: { ...request.target, name: '沪深300ETF' },
      period: request.period,
      parameters: instance.parameters ?? {},
      axis_anchor: 'market_close',
      history_policy: indicatorId === 'builtin-kdj-series' ? 'full_history' as const : 'lookback' as const,
      status: 'ok' as const,
      warnings: [],
      window: {
        requested_as_of: request.as_of ?? null,
        effective_as_of: dates[dates.length - 1] ?? null,
        start_date: dates[0],
        end_date: dates[dates.length - 1] ?? null,
        observation_count: dates.length,
        data_latest_date: dates[dates.length - 1] ?? null,
      },
      dates,
      channels,
      presentation: {
        indicator_id: indicatorId,
        revision: 1,
        name: indicatorId,
        source: 'built_in',
        category: 'technical',
        category_label: '技术与时序指标',
        context_kind: 'single_product',
        result_kind: 'time_series',
        catalog_status: 'current',
        display_format: 'number',
        precision: 2,
        unit: '',
        notation: 'standard',
        value_scale: 1,
        output_measure: 'series_bundle',
        direction: 'higher_better',
        description: '',
        methodology: '',
        data_basis: '真实行情',
        minimum_observations: 1,
        applicable_product_kinds: ['etf'],
      },
      execution: fixedIndicatorExecution,
    }
  })
  return {
    results,
    summary: {
      total: results.length,
      ok: results.length,
      warning: 0,
      unavailable: 0,
      error: 0,
    },
    cache: { hits: 0, misses: results.length },
    execution: { ...fixedIndicatorExecution, compiled_plan_ids: ['series-plan'] },
  }
}

const SIMULATION_METHOD_LABELS: Record<SimulationMethod, string> = {
  gaussian: '标准正态蒙特卡洛',
  parametric: '参数化蒙特卡洛（偏度/峰度校准）',
  block_bootstrap: '历史区块 Bootstrap',
  fhs_ewma: '滤波历史模拟 · EWMA',
  fhs_garch: '滤波历史模拟 · GARCH(1,1)',
}

const makeSimulation = (
  method: SimulationMethod,
  pathCount: number,
  horizon: number,
  targetReturn: number,
  blockLength: number,
  ewmaLambda = 0.94,
): FuturePathSimulation => {
  const moments = method === 'gaussian' || method === 'parametric'
  const filtered = method === 'fhs_ewma' || method === 'fhs_garch'
  return {
    method,
    methodLabel: SIMULATION_METHOD_LABELS[method],
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
      meanDailyLogReturn: moments || filtered ? 0.0002 : null,
      dailyLogVolatility: moments || filtered ? 0.008 : null,
      historicalLogSkewness: moments || filtered ? -0.2 : null,
      historicalLogExcessKurtosis: moments || filtered ? 0.8 : null,
      fittedLogSkewness: method === 'parametric' ? -0.18 : null,
      fittedLogExcessKurtosis: method === 'parametric' ? 0.76 : null,
      shapeCalibrationStatus: method === 'parametric' ? 'matched' : null,
      shapeSkewParameter: method === 'parametric' ? 0.1 : null,
      tailWeightParameter: method === 'parametric' ? 1.1 : null,
      averageBlockLength: method === 'block_bootstrap' ? blockLength : null,
      conditionalVolatilityStart: filtered ? 0.012 : null,
      volatilityPersistence: filtered ? (method === 'fhs_ewma' ? 1 : 0.972) : null,
      garchOmega: filtered ? (method === 'fhs_ewma' ? 0 : 1.8e-6) : null,
      garchAlpha: filtered ? (method === 'fhs_ewma' ? 1 - ewmaLambda : 0.083) : null,
      garchBeta: filtered ? (method === 'fhs_ewma' ? ewmaLambda : 0.889) : null,
      ewmaLambda: method === 'fhs_ewma' ? ewmaLambda : null,
      residualSkewness: filtered ? -0.11 : null,
      residualExcessKurtosis: filtered ? 1.42 : null,
    },
  }
}

/** Four strided frames, fanning out with the horizon like the kernel's do. */
const makeDensity = (pathCount: number, horizon: number): NavDensity => ({
  sampleSize: pathCount,
  navAxisMin: 0.88,
  navAxisMax: 1.12,
  frames: [1, 2, 3, 4].map((step) => {
    const halfWidth = 0.05 * step
    const navLow = 1 - halfWidth
    const navHigh = 1 + halfWidth
    return {
      day: Math.round((horizon * step) / 4),
      navLow,
      navHigh,
      binWidth: (navHigh - navLow) / 4,
      countAxisMax: pathCount,
      curve: [1, 10, 25, 10, 1],
      bins: [pathCount * 0.1, pathCount * 0.4, pathCount * 0.4, pathCount * 0.1],
    }
  }),
})

/** Opt-in realised future, so the默认 fixture keeps testing the PIT-off path. */
let realizedFuture: RealizedFuturePath | null = null

const makeRealized = (
  overrides: Partial<RealizedFuturePath> = {},
  score: Partial<RealizedMethodScore> = {},
): RealizedFuturePath => {
  const covered = overrides.coveredDays ?? 21
  const methodScore = (methodLabel: string): RealizedMethodScore => ({
    methodLabel,
    percentileRank: 0.18,
    band: 1,
    bandLabel: '5% — 25% 分位',
    verdict: '实际走势落在偏悲观区间（5%—25% 分位）：该模型的中枢偏乐观。',
    containmentRatio: 0.9,
    breachDays: 2,
    worstBreachGap: -0.031,
    worstBreachDay: 14,
    aboveMedianRatio: 0.2,
    simulatedP05: 0.9,
    simulatedP50: 1.01,
    simulatedP95: 1.12,
    ...score,
  })
  return {
    asOf: '2026-01-25',
    baseDate: '2026-01-25',
    baseNav: 1,
    startDate: '2026-01-26',
    endDate: '2026-02-24',
    requestedDays: 21,
    coveredDays: covered,
    observationDays: covered,
    complete: true,
    terminalNav: 0.964,
    terminalReturn: -0.036,
    maxDrawdown: 0.052,
    nav: Array.from({ length: covered + 1 }, (_, index) => 1 - index * 0.0018),
    dates: Array.from({ length: covered + 1 }, (_, index) => `2026-01-${String(25 + index).padStart(2, '0')}`),
    byMethod: Object.fromEntries(
      (Object.keys(SIMULATION_METHOD_LABELS) as SimulationMethod[]).map((method) => [
        method,
        methodScore(SIMULATION_METHOD_LABELS[method]),
      ]),
    ) as Record<SimulationMethod, RealizedMethodScore>,
    ...overrides,
  }
}

const makeAnalysisResponse = (request: ProductAnalysisRequest): ProductAnalysisResponse => {
  const incomplete = request.statistics_period === '1M'
  const returns = incomplete
    ? []
    : Array.from({ length: 24 }, (_, index) => ({ date: `2026-01-${String(index + 2).padStart(2, '0')}`, return: index % 2 ? 0.2 : -0.1 }))
  const methods = Object.keys(SIMULATION_METHOD_LABELS) as SimulationMethod[]
  const byMethod = Object.fromEntries(methods.map((method) => [
    method,
    makeSimulation(method, request.simulation_path_count, request.simulation_horizon, request.simulation_target_return, request.bootstrap_block_length, request.fhs_ewma_lambda),
  ])) as Record<SimulationMethod, FuturePathSimulation>
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
    simulation: incomplete || !request.include_simulation ? null : {
      initialNav: 1, methods, byMethod,
      realized: realizedFuture,
      realizedStatus: realizedFuture ? (realizedFuture.complete ? 'complete' : 'partial') : 'off',
      comparison: { p05ReturnGap: 0.01, medianReturnGap: 0.01, lossProbabilityGap: 0.02, conditionalValueAtRiskGap: 0.01, level: 'low', message: '各模型结果接近。' },
      densities: Object.fromEntries(methods.map((method) => [method, makeDensity(request.simulation_path_count, request.simulation_horizon)])) as Record<SimulationMethod, NavDensity>,
    },
    simulationStatus: !request.include_simulation ? 'not_requested' : incomplete ? 'insufficient_sample' : 'complete',
    researchContext: {
      startDate: '2026-01-02', endDate: '2026-01-25', observations: 25, returnObservations: returns.length,
      segmentCount: request.regime?.state_id ? 1 : 2, scope: request.regime?.segment_id ? 'segment' : request.regime?.state_id ? 'state' : 'full',
      stateLabel: request.regime?.state_id === 'bull' ? '牛市' : request.regime?.state_id === 'range' ? '震荡' : undefined,
      boundaryPolicy: 'within_continuous_segment', simulationEligible: !incomplete, simulationMessage: incomplete ? '有效收益样本不足。' : null,
      analysisBasis: request.analysis_basis ?? 'adjusted_nav', basisLabel: '复权净值', dataFingerprint: 'fixture',
    },
    regimeAnalysis: request.regime ? {
      selectedStateId: request.regime.state_id ?? null, selectedSegmentId: request.regime.segment_id ?? null,
      states: [
        { stateId: 'bull', stateLabel: '牛市', color: '#16a34a', observations: 5, returnObservations: 4, segmentCount: 1, medianSegmentObservations: 5, eligibleSegmentCount: 1, meanDailyReturn: 0.005, annualizedVolatility: 0.12, winRate: 0.5, medianSegmentReturn: 0.02, worstSegmentReturn: 0.02, medianSegmentDrawdown: -0.03, worstSegmentDrawdown: -0.03 },
        { stateId: 'range', stateLabel: '震荡', color: '#f59e0b', observations: 5, returnObservations: 4, segmentCount: 1, medianSegmentObservations: 5, eligibleSegmentCount: 1, meanDailyReturn: 0.001, annualizedVolatility: 0.08, winRate: 0.5, medianSegmentReturn: 0.01, worstSegmentReturn: 0.01, medianSegmentDrawdown: -0.01, worstSegmentDrawdown: -0.01 },
      ],
      segments: [
        { id: 'bull-1', stateId: 'bull', stateLabel: '牛市', color: '#16a34a', startDate: '2026-01-02', endDate: '2026-01-05', observations: 4, returnObservations: 3, cumulativeReturn: 0.02, maxDrawdown: -0.03, status: 'complete', reason: null },
        { id: 'range-1', stateId: 'range', stateLabel: '震荡', color: '#f59e0b', startDate: '2026-01-06', endDate: '2026-01-10', observations: 5, returnObservations: 4, cumulativeReturn: 0.01, maxDrawdown: -0.01, status: 'complete', reason: null },
      ],
    } : null,
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
    realizedFuture = null
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
    vi.mocked(evaluateTimeSeriesIndicators).mockImplementation(async (request) => (
      makeTimeSeriesResponse(request)
    ))
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
    expect(screen.queryByLabelText('分析样本区间')).not.toBeInTheDocument()
    expect(vi.mocked(analyzeProduct).mock.calls.every(([, , request]) => request.include_simulation === false)).toBe(true)
    await user.click(screen.getByRole('tab', { name: '未来模拟' }))
    expect(screen.getByLabelText('分析样本区间')).toHaveValue('ALL')
    expect(screen.queryByLabelText('收益数据口径')).not.toBeInTheDocument()
    expect(screen.getByRole('heading', { name: '未来虚拟净值模拟' })).toBeInTheDocument()
    expect(screen.getByText(/所有路径统一从虚拟净值 1\.0000 出发/)).toBeInTheDocument()
    expect(screen.getByLabelText('模拟未来区间')).toHaveValue('252')
    expect(screen.getByLabelText('模拟路径数')).toHaveValue('500')
    expect(screen.getByLabelText('目标期末收益率')).toHaveValue(5)
    // A model's own parameter belongs under that model. Sitting in the shared
    // row it read as though the block length applied to all five lanes.
    expect(screen.getByTestId('simulation-experiment-settings')).not.toHaveTextContent('区块')
    expect(screen.getByRole('radio', { name: '四矩校准' })).toBeChecked()
    expect(screen.getByRole('radio', { name: '区块 Bootstrap' })).not.toBeChecked()
    expect(screen.getByTestId('simulation-model-parameter')).toHaveTextContent('本模型无可调参数')
    expect(screen.queryByLabelText('Bootstrap 平均区块长度')).not.toBeInTheDocument()
    expect(screen.queryByTestId('monte-carlo-combined-chart')).not.toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: '运行模拟' }))
    const combinedChart = await screen.findByTestId('monte-carlo-combined-chart')
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
    expect(screen.getByText(/500 条模拟路径当天的净值落点/)).toBeInTheDocument()
    expect(screen.getByText('5 个模型结果对比')).toBeInTheDocument()
    expect(screen.getByRole('table', { name: '全部模拟模型的结果对比' })).toBeInTheDocument()
    expect(screen.getByText('95% CVaR（预期短缺）')).toBeInTheDocument()
    expect(screen.getAllByText('平均最大回撤')).toHaveLength(2)
    expect(screen.getByText('达到 5% 概率')).toBeInTheDocument()
    await user.selectOptions(screen.getByLabelText('模拟路径数'), '200')
    expect(screen.queryByTestId('monte-carlo-combined-chart')).not.toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: '运行模拟' }))
    await waitFor(() => expect(screen.getByTestId('monte-carlo-combined-chart').querySelector('[data-testid="chart"]')).toHaveAttribute('data-terminal-histogram-total', '200'))
    expect(screen.getByText(/200 条模拟路径当天的净值落点/)).toBeInTheDocument()
    await user.click(screen.getByRole('radio', { name: '区块 Bootstrap' }))
    expect(screen.getByRole('radio', { name: '区块 Bootstrap' })).toBeChecked()
    expect(screen.getByTestId('monte-carlo-combined-chart')).toHaveAccessibleName('历史区块 Bootstrap：路径与期末净值概率分布组合图')
    expect(screen.getByText(/平均区块长度 20 个收益观察值/)).toBeInTheDocument()
    expect(screen.getByLabelText('Bootstrap 平均区块长度')).toHaveValue('20')
    await user.click(screen.getByRole('tab', { name: '收益统计' }))
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
    await user.click(screen.getByRole('tab', { name: '走势与指标' }))
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

    await user.click(screen.getByRole('tab', { name: '未来模拟' }))
    await user.selectOptions(screen.getByLabelText('模拟未来区间'), '21')
    await user.selectOptions(screen.getByLabelText('模拟路径数'), '200')
    await user.selectOptions(screen.getByLabelText('Bootstrap 平均区块长度'), '10')
    await user.clear(screen.getByLabelText('目标期末收益率'))
    await user.type(screen.getByLabelText('目标期末收益率'), '8')
    await user.click(screen.getByRole('button', { name: '运行模拟' }))
    expect(screen.getByText('200 条虚拟路径中的样本比例')).toBeInTheDocument()
    expect(screen.getByText('达到 8% 概率')).toBeInTheDocument()

    await user.selectOptions(screen.getByLabelText('分析样本区间'), '1M')
    expect(screen.queryByTestId('monte-carlo-combined-chart')).not.toBeInTheDocument()
    await user.click(screen.getByRole('tab', { name: '收益统计' }))
    expect(await screen.findByText(/要求产品完整覆盖所选区间/)).toBeInTheDocument()
    expect(screen.queryByRole('heading', { name: '未来虚拟净值模拟' })).not.toBeInTheDocument()
  })

  const openSimulation = async (user: ReturnType<typeof userEvent.setup>) => {
    render(<MemoryRouter initialEntries={['/product/510300.SH?kind=etf']}><Routes><Route path="/product/:productId" element={<ProductDetail />} /></Routes></MemoryRouter>)
    await screen.findByRole('tab', { name: '未来模拟' })
    await user.click(screen.getByRole('tab', { name: '未来模拟' }))
    await user.selectOptions(screen.getByLabelText('模拟未来区间'), '21')
    await user.click(screen.getByRole('button', { name: '运行模拟' }))
    return screen.findByTestId('monte-carlo-combined-chart')
  }

  it('研究日之后的数据够长时，把实际走势叠加到模拟图上并给出事后评分', async () => {
    const user = userEvent.setup()
    realizedFuture = makeRealized()

    const chart = await openSimulation(user)

    const strip = screen.getByTestId('realized-future-strip')
    expect(strip).toHaveTextContent('完整覆盖 21/21 个交易日')
    expect(strip).toHaveTextContent('0.9640')
    expect(strip).toHaveTextContent('-3.6%')
    expect(strip).toHaveTextContent('18.0% 分位')
    expect(strip).toHaveTextContent('该模型的中枢偏乐观')
    // The drawdown only means something next to the model's own average.
    expect(strip).toHaveTextContent('模拟路径平均')
    // The overlay is a fact and the quantiles are scenarios; the caveat travels
    // with the numbers rather than sitting in a page footnote.
    expect(strip).toHaveTextContent('在研究日 2026-01-25 当天不可得')

    const renderer = chart.querySelector('[data-testid="chart"]')
    expect(renderer).toHaveAttribute('data-series', expect.stringContaining('实际走势'))
    // Anchor day plus one point per covered future day.
    expect(renderer).toHaveAttribute('data-realized-length', '22')

    // Same realised path, one score per model — that is what makes 双模型对比
    // answer "which one was closer" instead of only "how far apart are they".
    expect(screen.getByRole('columnheader', { name: '实际所处分位' })).toBeInTheDocument()
    expect(screen.getByRole('columnheader', { name: '实际留在区间' })).toBeInTheDocument()
  })

  it('研究日之后数据只走完一部分时，给区间不给分位', async () => {
    const user = userEvent.setup()
    realizedFuture = makeRealized(
      { coveredDays: 8, observationDays: 8, complete: false, endDate: '2026-02-05' },
      { percentileRank: null },
    )

    await openSimulation(user)

    const strip = screen.getByTestId('realized-future-strip')
    expect(strip).toHaveTextContent('仅覆盖 8/21 个交易日')
    expect(strip).toHaveTextContent('区间未走完，只给所处区间')
    expect(strip).toHaveTextContent('5% — 25% 分位')
  })

  it('没有 PIT 研究日时说明为什么看不到实际走势，而不是留白', async () => {
    const user = userEvent.setup()

    const chart = await openSimulation(user)

    expect(screen.queryByTestId('realized-future-strip')).not.toBeInTheDocument()
    expect(screen.getByText(/未启用 PIT 研究日/)).toBeInTheDocument()
    expect(chart.querySelector('[data-testid="chart"]')).not.toHaveAttribute('data-realized-length')
    expect(screen.queryByRole('columnheader', { name: '实际所处分位' })).not.toBeInTheDocument()
  })

  it('实际走势跌出模拟区间时，纵轴跟着放开，不把证据裁掉', async () => {
    const user = userEvent.setup()
    // The density fixture spans 0.88 — 1.12; this path ends well below it.
    realizedFuture = makeRealized({ nav: [1, 0.9, 0.72, 0.61], coveredDays: 3, observationDays: 3, terminalNav: 0.61 })

    const chart = await openSimulation(user)

    const renderer = chart.querySelector('[data-testid="chart"]')
    expect(Number(renderer?.getAttribute('data-y-axis-min'))).toBeCloseTo(0.61, 5)
    expect(Number(renderer?.getAttribute('data-y-axis-max'))).toBeCloseTo(1.12, 5)
  })

  it('缩放到区间中段时，右侧分布换成那一天的落点，而不是期末的', async () => {
    const user = userEvent.setup()

    const chart = await openSimulation(user)
    const renderer = chart.querySelector('[data-testid="chart"]')

    // Nothing zoomed: the panel answers for the horizon itself.
    expect(renderer).toHaveAttribute('data-density-title', '期末净值分布')
    expect(screen.getByText(/第 21 个未来交易日/)).toBeInTheDocument()

    await user.click(within(chart).getByTestId('drag-zoom-to-half'))

    // Half of 21 days is day 11, which the fixture has a frame for. Debounced,
    // so the assertion has to wait for the drag to settle.
    await waitFor(() => {
      expect(chart.querySelector('[data-testid="chart"]')).toHaveAttribute('data-density-title', '第 11 日净值分布')
    })
    expect(screen.getByText(/第 11 个未来交易日/)).toBeInTheDocument()
  })

  it('每个模型的参数只出现在它自己的面板里，并且只发给后端一次', async () => {
    const user = userEvent.setup()

    await openSimulation(user)

    // Bootstrap's block length and FHS's decay used to share one row with the
    // horizon and the path budget, which said they applied to every lane.
    await user.click(screen.getByRole('radio', { name: '区块 Bootstrap' }))
    expect(screen.getByLabelText('Bootstrap 平均区块长度')).toBeInTheDocument()
    expect(screen.queryByLabelText('EWMA 衰减系数')).not.toBeInTheDocument()

    await user.click(screen.getByRole('radio', { name: 'FHS · EWMA' }))
    expect(screen.queryByLabelText('Bootstrap 平均区块长度')).not.toBeInTheDocument()
    await user.selectOptions(screen.getByLabelText('EWMA 衰减系数'), '0.97')

    // Changing one model's parameter invalidates the whole batch: five lanes
    // are only comparable when they come from the same run.
    expect(screen.queryByTestId('monte-carlo-combined-chart')).not.toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: '运行模拟' }))
    await screen.findByTestId('monte-carlo-combined-chart')

    const calls = vi.mocked(analyzeProduct).mock.calls
    const [, , request] = calls[calls.length - 1]
    expect(request.fhs_ewma_lambda).toBe(0.97)
    expect(request.bootstrap_block_length).toBe(20)

    // A conditional model has to say so — that is the whole reason it exists.
    expect(screen.getByTestId('simulation-model-picker')).toHaveTextContent('条件模型 · 从当前波动状态出发')
    expect(screen.getByText(/起始条件日波动/)).toBeInTheDocument()
    expect(screen.getByText(/持续性恒为 1/)).toBeInTheDocument()

    await user.click(screen.getByRole('radio', { name: '标准正态' }))
    expect(screen.getByTestId('simulation-model-picker')).toHaveTextContent('无条件模型 · 忽略当前波动状态')
    expect(screen.getByText(/本模型不使用这两个形状参数/)).toBeInTheDocument()
    expect(screen.getByTestId('simulation-model-parameter')).toHaveTextContent('本模型无可调参数')
  })

  it('实际走势走完后，把中枢最接近的模型标出来', async () => {
    const user = userEvent.setup()
    // Only the GARCH lane's median sat near what happened; the rest were off.
    realizedFuture = makeRealized({
      byMethod: {
        gaussian: { percentileRank: 0.02 },
        parametric: { percentileRank: 0.05 },
        block_bootstrap: { percentileRank: 0.09 },
        fhs_ewma: { percentileRank: 0.2 },
        fhs_garch: { percentileRank: 0.46 },
      } as RealizedFuturePath['byMethod'],
    })

    await openSimulation(user)

    const marks = screen.getAllByText('最接近')
    // One on the model tab, one on that model's row in the comparison table.
    expect(marks).toHaveLength(2)
    expect(screen.getByRole('radio', { name: /FHS · GARCH/ }).closest('label')).toHaveTextContent('最接近')
    const garchRow = screen.getByRole('rowheader', { name: /滤波历史模拟 · GARCH/ })
    expect(garchRow).toHaveTextContent('最接近')
  })

  it('场外公募基金展示成立日期，并仅在披露时展示到期日期', async () => {
    const user = userEvent.setup()
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
    await user.click(screen.getByRole('tab', { name: '收益统计' }))
    expect(screen.getByLabelText('当前收益口径')).toHaveTextContent('复权净值')
    expect(screen.queryByRole('button', { name: '分析设置' })).not.toBeInTheDocument()
    expect(screen.queryByLabelText('收益数据口径')).not.toBeInTheDocument()
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
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        product_id: '510300.SH', name: '沪深300ETF', management: '测试管理人', status: '上市',
        base_info: { ts_code: '510300.SH', fund_type: 'ETF', list_date: '2012-05-28' },
        metrics: {},
        timeseries: Array.from({ length: 25 }, (_, index) => ({
          date: new Date(Date.UTC(2026, 0, 2 + index)).toISOString().slice(0, 10),
          open: null,
          high: null,
          low: null,
          close: 3 + index * 0.002,
          volume: null,
        })),
      }),
    }))

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

    await user.click(await screen.findByRole('tab', { name: '情景表现' }))
    const selector = await screen.findByRole('combobox', { name: '历史情景背景' })
    await waitFor(() => expect(selector).toBeEnabled())
    expect(within(selector).getByRole('option', { name: '不使用情景 · 普通研究' })).toBeInTheDocument()
    expect(within(selector).getByRole('option', { name: /沪深300牛熊震荡 · v3 · 实时识别/ })).toBeInTheDocument()
    expect(within(selector).queryByRole('option', { name: /可变试算/ })).not.toBeInTheDocument()
    expect(within(selector).queryByRole('option', { name: /仅研究展示/ })).not.toBeInTheDocument()
    expect(within(selector).queryByRole('option', { name: /发布记录错配/ })).not.toBeInTheDocument()

    await user.selectOptions(selector, 'run-eligible')
    expect(await screen.findByTestId('historical-regime-selection-meta')).toHaveTextContent('不可变运行 · 定义版本 v3')
    expect(getHistoricalRegimeRun).toHaveBeenCalledWith('run-eligible')
    expect(screen.getByTestId('historical-regime-selection-meta')).toHaveTextContent('实时模式：历史研究结果，可得性以版本证据为准')
    expect(within(screen.getByLabelText('市场状态')).getByRole('option', { name: '牛市' })).toBeInTheDocument()
    await user.click(screen.getByRole('tab', { name: '情景表现' }))
    const regimePerformance = await screen.findByRole('table', { name: '产品历史情景表现' })
    expect(within(regimePerformance).getByRole('row', { name: /牛市/ })).toHaveTextContent('%')
    expect(within(regimePerformance).getByRole('row', { name: /震荡/ })).toHaveTextContent('%')
    expect(screen.getByText(/跨情景边界收益不会归入任一状态/)).toBeInTheDocument()

    await user.click(screen.getByRole('tab', { name: '走势与指标' }))
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

    await user.click(screen.getByRole('button', { name: '调整情景' }))
    await user.click(screen.getByRole('button', { name: '清除情景' }))
    await user.click(screen.getByRole('tab', { name: '走势与指标' }))
    await waitFor(() => expect(getMainPriceChart()).toHaveAttribute('data-regime-mark-area', '[]'))
    expect(screen.queryByRole('button', { name: '调整情景' })).not.toBeInTheDocument()
    expect(screen.queryByRole('table', { name: '产品历史情景表现' })).not.toBeInTheDocument()
    expect(listHistoricalRegimeRuns).toHaveBeenCalledTimes(1)
    const regimeAnalysisCall = vi.mocked(analyzeProduct).mock.calls.find(([, , request]) => request.regime?.run_id === 'run-eligible')
    expect(regimeAnalysisCall?.[2].regime).toEqual({ run_id: 'run-eligible', publication_id: 'publication-run-eligible' })
  })

  it('没有合格发布版本时仅在情景页显示空态且不伪造图表背景', async () => {
    const user = userEvent.setup()
    vi.mocked(listHistoricalRegimeRuns).mockResolvedValue([
      makeHistoricalRegimeRun('run-display-only', '仅研究展示', { usage: 'research_display' }),
      makeHistoricalRegimeRun('run-mutable', '可变试算', { immutable: false }),
    ])

    render(<MemoryRouter initialEntries={['/product/510300.SH?kind=etf']}><Routes><Route path="/product/:productId" element={<ProductDetail />} /></Routes></MemoryRouter>)

    await screen.findByRole('tab', { name: '情景表现' })
    expect(screen.queryByText('暂无可用情景')).not.toBeInTheDocument()
    expect(screen.queryByLabelText('分析样本区间')).not.toBeInTheDocument()
    await user.click(screen.getByRole('tab', { name: '情景表现' }))
    expect(await screen.findByRole('heading', { name: '暂无可用情景' })).toBeInTheDocument()
    expect(screen.getByRole('link', { name: '前往情景中心' })).toHaveAttribute('href', '/settings/scenario-algorithms')
    expect(screen.queryByRole('combobox', { name: '历史情景背景' })).not.toBeInTheDocument()
    await user.click(screen.getByRole('tab', { name: '收益统计' }))
    expect(screen.queryByText('暂无可用情景')).not.toBeInTheDocument()
    const mainPriceChart = screen.getAllByTestId('chart')
      .find((chart) => chart.dataset.series?.split(',').includes('价格'))
    expect(mainPriceChart).toHaveAttribute('data-regime-mark-area', '[]')
  })

  it('保存新情景后可就地刷新并选用，选项不展示运行ID', async () => {
    const user = userEvent.setup()
    const run = makeHistoricalRegimeRun('run-clock', '美林时钟')
    vi.mocked(listHistoricalRegimeRuns).mockResolvedValueOnce([]).mockResolvedValue([run])
    vi.mocked(getHistoricalRegimeRun).mockResolvedValue(run)
    render(<MemoryRouter initialEntries={['/product/510300.SH?kind=etf']}><Routes><Route path="/product/:productId" element={<ProductDetail />} /></Routes></MemoryRouter>)
    await user.click(await screen.findByRole('tab', { name: '情景表现' }))
    await screen.findByRole('heading', { name: '暂无可用情景' })
    await user.click(screen.getByRole('button', { name: '刷新情景' }))
    const choice = await screen.findByRole('combobox', { name: '历史情景背景' })
    expect(within(choice).getByRole('option', { name: /美林时钟/ })).not.toHaveTextContent('run-clock')
    await user.selectOptions(choice, 'run-clock')
    await waitFor(() => expect(analyzeProduct).toHaveBeenCalledWith(expect.anything(), expect.anything(), expect.objectContaining({ regime: expect.objectContaining({ run_id: 'run-clock' }) }), expect.anything()))
    expect(listHistoricalRegimeRuns).toHaveBeenCalledTimes(2)
  })

  it('情景与单段联动研究条件，切换后忽略迟到的模拟响应', async () => {
    const user = userEvent.setup()
    const run = makeHistoricalRegimeRun('run-eligible', '沪深300牛熊震荡')
    vi.mocked(listHistoricalRegimeRuns).mockResolvedValue([run])
    vi.mocked(getHistoricalRegimeRun).mockResolvedValue(run)
    let finishSimulation: (() => void) | undefined
    let simulationSignal: AbortSignal | undefined
    vi.mocked(analyzeProduct).mockImplementation(async (_id, _kind, request, signal) => {
      if (!request.include_simulation) return makeAnalysisResponse(request)
      simulationSignal = signal
      return new Promise(resolve => { finishSimulation = () => resolve(makeAnalysisResponse(request)) })
    })
    render(<MemoryRouter initialEntries={['/product/510300.SH?kind=etf']}><Routes><Route path="/product/:productId" element={<ProductDetail />} /></Routes></MemoryRouter>)
    await user.click(await screen.findByRole('tab', { name: '情景表现' }))
    const scheme = await screen.findByLabelText('历史情景背景')
    await waitFor(() => expect(scheme).toBeEnabled())
    await user.selectOptions(scheme, 'run-eligible')
    await user.click(screen.getByRole('tab', { name: '情景表现' }))
    expect(await screen.findByRole('table', { name: '产品历史情景表现' })).toHaveTextContent('可计算 1/1 段')
    await user.click(screen.getByRole('button', { name: '牛市' }))
    expect(screen.getByLabelText('市场状态')).toHaveValue('bull')
    await user.click(await screen.findByRole('button', { name: '查看区间 2026-01-02 至 2026-01-05' }))
    await waitFor(() => expect(screen.getByLabelText('连续区间')).toHaveValue('bull-1'))
    expect(vi.mocked(analyzeProduct).mock.lastCall?.[2].regime).toEqual({ run_id: 'run-eligible', publication_id: 'publication-run-eligible', state_id: 'bull', segment_id: 'bull-1' })
    const settings = screen.getByRole('button', { name: '分析设置' })
    expect(settings).toHaveAttribute('aria-expanded', 'false')
    expect(screen.queryByLabelText('收益数据口径')).not.toBeInTheDocument()
    await user.click(settings)
    await user.selectOptions(screen.getByLabelText('收益数据口径'), 'price')
    await user.click(settings)
    expect(screen.getByLabelText('当前收益口径')).toHaveTextContent('交易价格')
    await waitFor(() => expect(vi.mocked(analyzeProduct).mock.lastCall?.[2]).toMatchObject({
      analysis_basis: 'price', regime: { run_id: 'run-eligible', publication_id: 'publication-run-eligible', state_id: 'bull' },
    }))
    expect(vi.mocked(analyzeProduct).mock.lastCall?.[2].regime).not.toHaveProperty('segment_id')
    expect(screen.getByLabelText('市场状态')).toHaveValue('bull')
    expect(screen.getByLabelText('连续区间')).toHaveValue('')
    expect(vi.mocked(analyzeProduct).mock.calls.every(([, , request]) => !request.include_simulation)).toBe(true)
    await user.click(screen.getByRole('tab', { name: '未来模拟' }))
    await user.click(screen.getByRole('button', { name: '运行模拟' }))
    expect(screen.getByRole('button', { name: '正在模拟…' })).toBeDisabled()
    await user.selectOptions(screen.getByLabelText('市场状态'), 'range')
    expect(simulationSignal?.aborted).toBe(true)
    expect(screen.getByLabelText('连续区间')).toHaveValue('')
    await act(async () => { finishSimulation?.() })
    expect(screen.queryByTestId('monte-carlo-combined-chart')).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: '运行模拟' })).toBeEnabled()
    await user.click(screen.getByRole('button', { name: '清除情景' }))
    expect(screen.queryByLabelText('市场状态')).not.toBeInTheDocument()
    expect(screen.getByTestId('product-research-scope')).toHaveTextContent('完整样本')
    expect(screen.getByTestId('product-research-scope')).not.toHaveTextContent('段')
  })

  it('行情为空时保留产品资料与研究入口', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => ({
      product_id: '510300.SH', name: '暂无行情的产品', management: '示例管理人',
      base_info: { ts_code: '510300.SH', list_date: '20120528' }, metrics: {}, timeseries: [],
    }) }))
    render(<MemoryRouter initialEntries={['/product/510300.SH?kind=etf']}><Routes><Route path="/product/:productId" element={<ProductDetail />} /></Routes></MemoryRouter>)
    expect(await screen.findByRole('heading', { name: '暂无行情的产品' })).toBeInTheDocument()
    expect(screen.queryByLabelText('分析样本')).not.toBeInTheDocument()
    expect(screen.getByText('暂无可视化数据')).toBeInTheDocument()
    expect(screen.getByRole('tab', { name: '情景表现' })).toBeInTheDocument()
  })

  it('工作区支持键盘切换，切换标签不触发重新模拟', async () => {
    const user = userEvent.setup()
    render(<MemoryRouter initialEntries={['/product/510300.SH?kind=etf']}><Routes><Route path="/product/:productId" element={<ProductDetail />} /></Routes></MemoryRouter>)
    const firstTab = await screen.findByRole('tab', { name: '走势与指标' })
    firstTab.focus()
    await user.keyboard('{ArrowRight}')
    expect(screen.getByRole('tab', { name: '情景表现' })).toHaveFocus()
    expect(screen.getByRole('tab', { name: '情景表现' })).toHaveAttribute('aria-selected', 'true')
    await user.keyboard('{End}')
    expect(screen.getByRole('tab', { name: '风险与压测' })).toHaveFocus()
    expect(screen.queryByLabelText('分析样本')).not.toBeInTheDocument()
    await user.keyboard('{ArrowLeft}')
    expect(screen.getByRole('tab', { name: '未来模拟' })).toHaveFocus()
    expect(vi.mocked(analyzeProduct).mock.calls.every(([, , request]) => !request.include_simulation)).toBe(true)
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

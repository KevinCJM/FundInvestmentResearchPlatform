import {
  assertFixedNjitExecution,
  type FixedNjitExecutionAudit,
} from '../utils/fixedNjitExecution'

export type StatisticsPeriod = 'ALL' | '1M' | '3M' | '6M' | '1Y' | '3Y' | '5Y'

export const STATISTICS_PERIOD_OPTIONS: Array<{
  value: StatisticsPeriod
  label: string
}> = [
  { value: 'ALL', label: '成立以来' },
  { value: '1M', label: '近 1 月' },
  { value: '3M', label: '近 3 月' },
  { value: '6M', label: '近 6 月' },
  { value: '1Y', label: '近 1 年' },
  { value: '3Y', label: '近 3 年' },
  { value: '5Y', label: '近 5 年' },
]

export const MONTE_CARLO_HORIZON_OPTIONS = [
  { value: 21, label: '未来 1 个月（21 个交易日）' },
  { value: 63, label: '未来 3 个月（63 个交易日）' },
  { value: 126, label: '未来 6 个月（126 个交易日）' },
  { value: 252, label: '未来 1 年（252 个交易日）' },
  { value: 504, label: '未来 2 年（504 个交易日）' },
]

export const MONTE_CARLO_PATH_OPTIONS = [200, 500, 1000]
export const BOOTSTRAP_BLOCK_LENGTH_OPTIONS = [5, 10, 20, 60]
export const FHS_EWMA_LAMBDA_OPTIONS = [
  { value: 0.9, label: '0.90 · 更快跟随（半衰期约 7 天）' },
  { value: 0.94, label: '0.94 · RiskMetrics 日频默认（约 11 天）' },
  { value: 0.97, label: '0.97 · 更平滑（约 23 天）' },
]
export const MIN_SIMULATION_OBSERVATIONS = 20

export type SimulationMethod =
  | 'gaussian'
  | 'parametric'
  | 'block_bootstrap'
  | 'fhs_ewma'
  | 'fhs_garch'
export type ShapeCalibrationStatus = 'matched' | 'approximate' | 'normal_fallback'

/** Which request field a model owns, so the UI can put the control under it. */
export type SimulationMethodParameter = 'none' | 'bootstrap_block_length' | 'fhs_ewma_lambda'

/**
 * Display order and the one-line description each model gets in its own panel.
 *
 * `conditional` is the property that actually separates these five: the first
 * three fit one distribution to the whole window and start every path from it,
 * so a研究日 in a volatility spike gets the same fan as one in a calm stretch.
 */
export const SIMULATION_METHOD_META: Record<SimulationMethod, {
  tab: string
  conditional: boolean
  summary: string
  parameter: SimulationMethodParameter
}> = {
  gaussian: {
    tab: '标准正态',
    conditional: false,
    summary: '教科书基准：日对数收益服从正态分布，只用到样本均值与波动率。它低估肥尾，作用是给其他四个模型一条“多出来的假设值不值”的对照线。',
    parameter: 'none',
  },
  parametric: {
    tab: '四矩校准',
    conditional: false,
    summary: '用 sinh-arcsinh 变换同时校准样本的均值、波动率、偏度与超额峰度，单日分布形状比正态准确得多；仍假设每天独立同分布，没有波动聚集。',
    parameter: 'none',
  },
  block_bootstrap: {
    tab: '区块 Bootstrap',
    conditional: false,
    summary: '成段抽取真实历史收益，保留段内的序列依赖，不假设任何分布形状；代价是永远画不出比历史更坏的单日走势。',
    parameter: 'bootstrap_block_length',
  },
  fhs_ewma: {
    tab: 'FHS · EWMA',
    conditional: true,
    summary: '先用 EWMA 把历史收益除以当时的条件波动率，抽取标准化残差，再用“今天”的波动率放大回去。从当前波动状态出发，且平静期的大残差能落到高波动上——这样才可能生成比历史更坏的一天。EWMA 的持续性恒为 1，波动不回归长期均值。',
    parameter: 'fhs_ewma_lambda',
  },
  fhs_garch: {
    tab: 'FHS · GARCH(1,1)',
    conditional: true,
    summary: '同样是滤波历史模拟，但波动率的三个系数由该产品自身历史用准极大似然估计（方差目标化），因此波动会按估计出的持续性向长期均值回归。系数自动拟合，无需手工设置。',
    parameter: 'none',
  },
}

export const SIMULATION_METHOD_ORDER = Object.keys(
  SIMULATION_METHOD_META,
) as SimulationMethod[]

export interface DailyReturnPoint {
  date: string
  return: number
}

export interface DistributionInterpretation {
  label: string
  meaning: string
}

export interface ReturnStatistics {
  mean: number | null
  std: number | null
  median: number | null
  positiveRatio: number | null
  best: number | null
  worst: number | null
  sampleSize: number
  skewness: number | null
  kurtosis: number | null
  jbStatistic: number | null
  normalityPValue: number | null
}

export interface ReturnHistogramBin {
  start: number
  end: number
  count: number
  normalPdfCount: number | null
  frequency: number
  center: number
}

export interface BoxPlotResult {
  stats: number[]
  outliers: number[]
  quartiles: { q1: number; q3: number; median: number; iqr: number }
  whiskers: { lower: number; upper: number }
}

export interface NormalQqPoint {
  percentile: number
  theoreticalQuantile: number
  observedReturn: number
  referenceReturn: number
  tail: 'lower' | 'center' | 'upper'
}

export interface NormalQqData {
  sampleSize: number
  points: NormalQqPoint[]
  keyPoints: NormalQqPoint[]
}

export interface FuturePathSimulation {
  method: SimulationMethod
  methodLabel: string
  days: number[]
  samplePaths: number[][]
  percentiles: { p05: number[]; p25: number[]; p50: number[]; p75: number[]; p95: number[] }
  terminal: {
    p05: number
    p25: number
    p50: number
    p75: number
    p95: number
    lossProbability: number
    valueAtRisk95: number
    conditionalValueAtRisk95: number
    targetHitProbability: number
    averageMaxDrawdown: number
    p05Return: number
    medianReturn: number
  }
  assumptions: {
    sourceObservationCount: number
    targetReturnPercent: number | null
    meanDailyLogReturn: number | null
    dailyLogVolatility: number | null
    historicalLogSkewness: number | null
    historicalLogExcessKurtosis: number | null
    fittedLogSkewness: number | null
    fittedLogExcessKurtosis: number | null
    shapeCalibrationStatus: ShapeCalibrationStatus | null
    shapeSkewParameter: number | null
    tailWeightParameter: number | null
    averageBlockLength: number | null
    /** Filtered lanes: the volatility the first simulated day starts from. */
    conditionalVolatilityStart: number | null
    /** alpha + beta. Exactly 1 for EWMA, so its volatility never mean-reverts. */
    volatilityPersistence: number | null
    garchOmega: number | null
    garchAlpha: number | null
    garchBeta: number | null
    ewmaLambda: number | null
    /** Shape left in the residuals after filtering — what the bootstrap draws. */
    residualSkewness: number | null
    residualExcessKurtosis: number | null
  }
}

/** One model's score against the realised future, from the same research day. */
export interface RealizedMethodScore {
  methodLabel: string
  /** Share of simulated terminal values at or below the realised one. Null until the horizon is fully covered. */
  percentileRank: number | null
  band: number | null
  bandLabel: string | null
  verdict: string | null
  /** Days the realised path stayed inside 5%—95%. Descriptive, not a score: one path's ratio is noisy. */
  containmentRatio: number | null
  breachDays: number
  /** Widest signed distance outside the band; negative below 5%, positive above 95%. */
  worstBreachGap: number | null
  worstBreachDay: number | null
  aboveMedianRatio: number | null
  simulatedP05: number | null
  simulatedP50: number | null
  simulatedP95: number | null
}

/**
 * What actually happened after the研究日, on the simulation's own day axis.
 *
 * Present only under a PIT研究日 with rows beyond it. The prices reach the
 * response through a lane that never touches the simulation inputs, so the
 * overlay grades the model rather than joining it.
 */
export interface RealizedFuturePath {
  asOf: string | null
  /** The observation the path is normalised on — nav 1.0 by definition. */
  baseDate: string
  baseNav: number
  startDate: string | null
  endDate: string
  requestedDays: number
  coveredDays: number
  /** Days with a usable price; below coveredDays when the source has gaps. */
  observationDays: number
  complete: boolean
  terminalNav: number | null
  terminalReturn: number | null
  maxDrawdown: number | null
  /** Index 0 is the anchor day; nulls are未披露 observations. */
  nav: Array<number | null>
  dates: string[]
  byMethod: Record<SimulationMethod, RealizedMethodScore>
}

export type RealizedStatus = 'off' | 'no_future_data' | 'partial' | 'complete'

/** Spread — not a pairwise difference — of each headline figure across models. */
export interface SimulationComparison {
  p05ReturnGap: number
  medianReturnGap: number
  lossProbabilityGap: number
  conditionalValueAtRiskGap: number
  level: 'low' | 'medium' | 'high'
  message: string
}

/**
 * One day's landing distribution. `curve` and `bins` are path counts on
 * uniform grids over [navLow, navHigh] — the NAV each entry sits at is
 * derivable, and not naming it is what keeps ~40 frames per model affordable.
 */
export interface NavDensityFrame {
  day: number
  navLow: number
  navHigh: number
  binWidth: number
  countAxisMax: number
  curve: number[]
  bins: number[]
}

export interface NavDensity {
  sampleSize: number
  /** Shared by every frame and by the path chart, so frames stay comparable. */
  navAxisMin: number
  navAxisMax: number
  /** Ascending, strided; the last one is the horizon's own distribution. */
  frames: NavDensityFrame[]
}

/** Evenly spaced NAV points of a frame's curve, paired with its path counts. */
export function navDensityCurve(frame: NavDensityFrame, initialNav: number) {
  const span = frame.navHigh - frame.navLow
  const steps = Math.max(1, frame.curve.length - 1)
  return frame.curve.map((count, index) => {
    const nav = frame.navLow + (span * index) / steps
    return { nav, count, simulatedReturn: nav / initialNav - 1 }
  })
}

/** The frame whose day is closest to `day` — frames are strided, not daily. */
export function nearestNavDensityFrame(frames: NavDensityFrame[], day: number): NavDensityFrame | null {
  let best: NavDensityFrame | null = null
  let bestDistance = Number.POSITIVE_INFINITY
  for (const frame of frames) {
    const distance = Math.abs(frame.day - day)
    if (distance < bestDistance) {
      bestDistance = distance
      best = frame
    }
  }
  return best
}

export interface ProductRegimeState {
  stateId: string
  stateLabel: string
  color: string
  observations: number
  returnObservations: number
  segmentCount: number
  medianSegmentObservations: number | null
  eligibleSegmentCount: number
  meanDailyReturn: number | null
  annualizedVolatility: number | null
  winRate: number | null
  medianSegmentReturn: number | null
  worstSegmentReturn: number | null
  medianSegmentDrawdown: number | null
  worstSegmentDrawdown: number | null
}

export interface ProductRegimeSegment {
  id: string
  stateId: string
  stateLabel: string
  color: string
  startDate: string
  endDate: string
  observations: number
  returnObservations: number
  cumulativeReturn: number | null
  maxDrawdown: number | null
  status: string
  reason: string | null
  windowClipped?: boolean
  validObservations?: number
}

export interface ProductRegimeAnalysis {
  states: ProductRegimeState[]
  segments: ProductRegimeSegment[]
  selectedStateId: string | null
  selectedSegmentId: string | null
}

export interface ProductResearchContext {
  windowStartDate?: string | null
  windowEndDate?: string | null
  observationFrequency?: string
  warnings?: string[]
  asOf?: string | null
  startDate: string | null
  endDate: string | null
  observations: number
  returnObservations: number
  segmentCount: number
  scope: 'full' | 'state' | 'segment'
  stateLabel?: string | null
  analysisBasis: 'adjusted_nav' | 'price'
  basisLabel: string
  dataFingerprint: string
  boundaryPolicy: string
  simulationEligible: boolean
  simulationMessage: string | null
}

export interface ProductAnalysisExecution extends FixedNjitExecutionAudit {
  execution_backend: 'numba_njit_fixed_signature'
  engine: string
  kernel_version: string
  kernel_coverage: string
  kernel_fingerprint: string
  kernel_signatures: Record<string, string[]>
  nopython: true
  njit_required: true
  object_mode: 0
  python_fallback: 0
  request_time_compilation: 0
}

export interface ProductAnalysisResponse {
  schema_version: number
  product_id: string
  execution: ProductAnalysisExecution
  window: {
    complete: boolean
    requested_start_date: string | null
    message: string | null
  }
  technical: {
    availability: {
      ohlc: boolean
      volume: boolean
      kdj: boolean
    }
    priceMa: Record<string, Array<number | null>>
    volumeMa: Record<string, Array<number | null>>
    bollinger: {
      upper: Array<number | null>
      middle: Array<number | null>
      lower: Array<number | null>
    }
    kdj: {
      kValues: Array<number | null>
      dValues: Array<number | null>
      jValues: Array<number | null>
    }
  }
  dailyReturns: DailyReturnPoint[]
  returnStatistics: ReturnStatistics
  interpretation: {
    skewness: DistributionInterpretation
    kurtosis: DistributionInterpretation
    normality: string
  }
  histogram: ReturnHistogramBin[]
  boxPlot: BoxPlotResult | null
  normalQq: NormalQqData | null
  simulation: {
    initialNav: number
    realized: RealizedFuturePath | null
    realizedStatus: RealizedStatus
    /** Display order, server-owned so a new lane needs no frontend release. */
    methods: SimulationMethod[]
    byMethod: Record<SimulationMethod, FuturePathSimulation>
    comparison: SimulationComparison
    densities: Record<SimulationMethod, NavDensity>
  } | null
  regimeAnalysis: ProductRegimeAnalysis | null
  researchContext: ProductResearchContext
  simulationStatus: 'not_requested' | 'insufficient_sample' | 'complete'
}

export interface ProductAnalysisRequest {
  statistics_period: StatisticsPeriod
  analysis_basis?: 'adjusted_nav' | 'price'
  include_simulation?: boolean
  include_technical?: boolean
  price_ma_periods: number[]
  volume_ma_periods: number[]
  boll_period: number
  boll_multiplier: number
  kdj_period: number
  kdj_k_smoothing: number
  kdj_d_smoothing: number
  histogram_bin_width: number
  // The experiment's own settings: shared by every model so the comparison
  // table and the realised overlay stay on one axis.
  simulation_horizon: number
  simulation_path_count: number
  simulation_target_return: number
  simulation_run: number
  // Per-model settings, one field per model that has one.
  bootstrap_block_length: number
  fhs_ewma_lambda: number
  regime: {
    run_id: string
    publication_id: string
    state_id?: string
    segment_id?: string
  } | null
}

export class ProductAnalysisApiError extends Error {
  constructor(readonly status: number, message: string) {
    super(message)
    this.name = 'ProductAnalysisApiError'
  }
}

export async function analyzeProduct(
  productId: string,
  kind: 'etf' | 'fund',
  request: ProductAnalysisRequest,
  signal?: AbortSignal,
): Promise<ProductAnalysisResponse> {
  const response = await fetch(
    `/api/instruments/products/${encodeURIComponent(productId)}/analysis?kind=${kind}`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
      signal,
    },
  )
  if (!response.ok) {
    let message = `产品分析失败（${response.status}）`
    try {
      const payload = await response.json()
      if (typeof payload?.detail === 'string') message = payload.detail
    } catch { /* retain stable fallback */ }
    throw new ProductAnalysisApiError(response.status, message)
  }
  let result: ProductAnalysisResponse
  try {
    result = await response.json() as ProductAnalysisResponse
  } catch {
    throw new ProductAnalysisApiError(response.status, '产品分析服务返回了无效响应')
  }
  try {
    assertFixedNjitExecution(result.execution, '产品分析')
  } catch (failure) {
    throw new ProductAnalysisApiError(
      response.status,
      failure instanceof Error ? failure.message : '产品分析未通过固定签名 NJIT 执行校验',
    )
  }
  return result
}

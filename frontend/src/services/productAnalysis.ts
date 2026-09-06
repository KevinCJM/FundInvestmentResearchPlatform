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
export const MIN_SIMULATION_OBSERVATIONS = 20

export type SimulationMethod = 'parametric' | 'block_bootstrap'
export type ShapeCalibrationStatus = 'matched' | 'approximate' | 'normal_fallback'

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
  }
}

export interface SimulationComparison {
  p05ReturnGap: number
  medianReturnGap: number
  lossProbabilityGap: number
  conditionalValueAtRiskGap: number
  level: 'low' | 'medium' | 'high'
  message: string
}

export interface TerminalNavDensity {
  sampleSize: number
  points: Array<{ nav: number; density: number; estimatedCount: number; simulatedReturn: number }>
  histogram: Array<{
    lowerNav: number
    upperNav: number
    density: number
    count: number
    frequency: number
  }>
  maxDensity: number
  modeNav: number
  minNav: number
  maxNav: number
  countAxisMax: number
  navAxisMin: number
  navAxisMax: number
  densityCountFactor: number
  histogramBinWidth: number
}

export interface ProductRegimeStatistic {
  stateId: string
  stateLabel: string
  color: string
  observations: number
  returnObservations: number
  cumulativeReturn: number | null
  annualizedVolatility: number | null
  maxDrawdown: number | null
  winRate: number | null
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
    parametric: FuturePathSimulation
    blockBootstrap: FuturePathSimulation
    comparison: SimulationComparison
    densities: {
      parametric: TerminalNavDensity
      block_bootstrap: TerminalNavDensity
    }
  } | null
  regimeStatistics: ProductRegimeStatistic[]
}

export interface ProductAnalysisRequest {
  statistics_period: StatisticsPeriod
  include_technical?: boolean
  price_ma_periods: number[]
  volume_ma_periods: number[]
  boll_period: number
  boll_multiplier: number
  kdj_period: number
  kdj_k_smoothing: number
  kdj_d_smoothing: number
  histogram_bin_width: number
  simulation_horizon: number
  simulation_path_count: number
  bootstrap_block_length: number
  simulation_target_return: number
  simulation_run: number
  regime: {
    run_id: string
    publication_id: string
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

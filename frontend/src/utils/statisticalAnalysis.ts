export type StatisticsPeriod = 'ALL' | '1M' | '3M' | '6M' | '1Y' | '3Y' | '5Y'

export const STATISTICS_PERIOD_OPTIONS: Array<{
  value: StatisticsPeriod
  label: string
  months: number | null
}> = [
  { value: 'ALL', label: '成立以来', months: null },
  { value: '1M', label: '近 1 月', months: 1 },
  { value: '3M', label: '近 3 月', months: 3 },
  { value: '6M', label: '近 6 月', months: 6 },
  { value: '1Y', label: '近 1 年', months: 12 },
  { value: '3Y', label: '近 3 年', months: 36 },
  { value: '5Y', label: '近 5 年', months: 60 },
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

interface DatedPoint {
  date: string
}

export interface StatisticsWindow<T> {
  series: T[]
  complete: boolean
  requestedStartDate: string | null
  message: string | null
}

export interface DistributionInterpretation {
  label: string
  meaning: string
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
}

export interface FuturePathSimulation {
  method: SimulationMethod
  methodLabel: string
  days: number[]
  samplePaths: number[][]
  terminalValues: number[]
  percentiles: {
    p05: number[]
    p25: number[]
    p50: number[]
    p75: number[]
    p95: number[]
  }
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
  }
  assumptions: {
    sourceObservationCount: number
    targetReturnPercent: number
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

// Kept as a source-compatible alias for callers that used the old name.
export type MonteCarloSimulation = FuturePathSimulation

export interface SimulationComparison {
  p05ReturnGap: number
  medianReturnGap: number
  lossProbabilityGap: number
  conditionalValueAtRiskGap: number
  level: 'low' | 'medium' | 'high'
  message: string
}

export interface TerminalNavDensityPoint {
  nav: number
  density: number
}

export interface TerminalNavHistogramBin {
  lowerNav: number
  upperNav: number
  density: number
  count: number
}

export interface TerminalNavDensity {
  sampleSize: number
  points: TerminalNavDensityPoint[]
  histogram: TerminalNavHistogramBin[]
  maxDensity: number
  modeNav: number
  minNav: number
  maxNav: number
}

const DAY_MS = 24 * 60 * 60 * 1000
const MAX_BOUNDARY_GAP_DAYS = 10

const parseDate = (value: string) => {
  const timestamp = Date.parse(`${value.slice(0, 10)}T00:00:00Z`)
  return Number.isFinite(timestamp) ? timestamp : null
}

const formatUtcDate = (timestamp: number) => new Date(timestamp).toISOString().slice(0, 10)

const subtractMonths = (timestamp: number, months: number) => {
  const source = new Date(timestamp)
  const day = source.getUTCDate()
  source.setUTCDate(1)
  source.setUTCMonth(source.getUTCMonth() - months)
  const lastDay = new Date(Date.UTC(source.getUTCFullYear(), source.getUTCMonth() + 1, 0)).getUTCDate()
  source.setUTCDate(Math.min(day, lastDay))
  return source.getTime()
}

export function selectStatisticsWindow<T extends DatedPoint>(
  input: T[],
  period: StatisticsPeriod,
): StatisticsWindow<T> {
  const series = input
    .filter((point) => parseDate(point.date) !== null)
    .slice()
    .sort((left, right) => left.date.localeCompare(right.date))
  if (period === 'ALL' || series.length === 0) {
    return { series, complete: true, requestedStartDate: null, message: null }
  }

  const periodOption = STATISTICS_PERIOD_OPTIONS.find((option) => option.value === period)
  const latestTimestamp = parseDate(series[series.length - 1].date)
  if (!periodOption?.months || latestTimestamp === null) {
    return { series: [], complete: false, requestedStartDate: null, message: '无法识别统计区间。' }
  }
  const requestedStart = subtractMonths(latestTimestamp, periodOption.months)
  let boundaryIndex = -1
  for (let index = series.length - 1; index >= 0; index -= 1) {
    const timestamp = parseDate(series[index].date)
    if (timestamp !== null && timestamp <= requestedStart) {
      boundaryIndex = index
      break
    }
  }
  const boundaryTimestamp = boundaryIndex >= 0 ? parseDate(series[boundaryIndex].date) : null
  const gapDays = boundaryTimestamp === null ? Number.POSITIVE_INFINITY : (requestedStart - boundaryTimestamp) / DAY_MS
  if (boundaryIndex < 0 || gapDays > MAX_BOUNDARY_GAP_DAYS) {
    return {
      series: [],
      complete: false,
      requestedStartDate: formatUtcDate(requestedStart),
      message: `${periodOption.label}要求产品完整覆盖所选区间；当前历史数据不足，统计指标不计算。`,
    }
  }
  return {
    series: series.slice(boundaryIndex),
    complete: true,
    requestedStartDate: formatUtcDate(requestedStart),
    message: null,
  }
}

export const interpretSkewness = (value: number | null): DistributionInterpretation => {
  if (value === null || !Number.isFinite(value)) {
    return { label: '样本不足', meaning: '至少需要 3 个有效收益观察值才能判断偏度。' }
  }
  if (value <= -0.1) {
    const intensity = value <= -1 ? '显著' : value <= -0.5 ? '中度' : '轻度'
    return {
      label: `${intensity}左偏（负偏）`,
      meaning: '左侧负收益尾部更长，少数较大亏损可能拖累整体收益，需特别关注下行尾部风险。',
    }
  }
  if (value >= 0.1) {
    const intensity = value >= 1 ? '显著' : value >= 0.5 ? '中度' : '轻度'
    return {
      label: `${intensity}右偏（正偏）`,
      meaning: '右侧正收益尾部更长，少数较大盈利可能抬高平均收益，但不代表亏损风险较低。',
    }
  }
  return {
    label: '近似对称',
    meaning: '正负收益尾部大致均衡；这只描述分布方向，不代表波动或尾部风险较低。',
  }
}

export const interpretExcessKurtosis = (value: number | null): DistributionInterpretation => {
  if (value === null || !Number.isFinite(value)) {
    return { label: '样本不足', meaning: '至少需要 4 个有效收益观察值才能判断峰度。' }
  }
  if (value >= 0.1) {
    const intensity = value >= 3 ? '显著' : value >= 1 ? '中度' : '轻度'
    return {
      label: `${intensity}尖峰厚尾`,
      meaning: '收益更集中在中心且尾部更厚，极端涨跌出现概率高于正态分布，正态模型可能低估尾部风险。',
    }
  }
  if (value <= -0.1) {
    const intensity = value <= -1 ? '显著' : value <= -0.5 ? '中度' : '轻度'
    return {
      label: `${intensity}平峰薄尾`,
      meaning: '收益分布较平、样本尾部相对较薄，历史极端波动较少，但不能据此排除未来尾部事件。',
    }
  }
  return {
    label: '接近正态峰度',
    meaning: '样本峰度接近正态分布；仍需结合偏度、正态性检验和极值共同判断风险。',
  }
}

const seededRandom = (seedText: string) => {
  let seed = 2166136261
  for (let index = 0; index < seedText.length; index += 1) {
    seed ^= seedText.charCodeAt(index)
    seed = Math.imul(seed, 16777619)
  }
  return () => {
    seed += 0x6D2B79F5
    let value = seed
    value = Math.imul(value ^ (value >>> 15), value | 1)
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61)
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296
  }
}

const quantile = (sorted: number[], probability: number) => {
  const position = (sorted.length - 1) * probability
  const base = Math.floor(position)
  const remainder = position - base
  const lower = sorted[base]
  const upper = sorted[Math.min(sorted.length - 1, base + 1)]
  return lower + (upper - lower) * remainder
}

// Peter J. Acklam's rational approximation keeps Q-Q construction dependency-free.
const inverseStandardNormal = (probability: number) => {
  if (probability <= 0 || probability >= 1) return Number.NaN
  const a = [-39.6968302866538, 220.946098424521, -275.928510446969, 138.357751867269, -30.6647980661472, 2.50662827745924]
  const b = [-54.4760987982241, 161.585836858041, -155.698979859887, 66.8013118877197, -13.2806815528857]
  const c = [-0.00778489400243029, -0.322396458041136, -2.40075827716184, -2.54973253934373, 4.37466414146497, 2.93816398269878]
  const d = [0.00778469570904146, 0.32246712907004, 2.445134137143, 3.75440866190742]
  const lowerBoundary = 0.02425
  const upperBoundary = 1 - lowerBoundary

  if (probability < lowerBoundary) {
    const q = Math.sqrt(-2 * Math.log(probability))
    return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
      / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
  }
  if (probability > upperBoundary) {
    const q = Math.sqrt(-2 * Math.log(1 - probability))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
      / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
  }
  const q = probability - 0.5
  const r = q * q
  return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
    / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
}

export const buildNormalQqData = (returnsPercent: number[]): NormalQqData | null => {
  const sorted = returnsPercent
    .filter((value) => Number.isFinite(value))
    .slice()
    .sort((left, right) => left - right)
  if (sorted.length < 3) return null

  const theoreticalQ1 = inverseStandardNormal(0.25)
  const theoreticalQ3 = inverseStandardNormal(0.75)
  const observedQ1 = quantile(sorted, 0.25)
  const observedQ3 = quantile(sorted, 0.75)
  const referenceSlope = (observedQ3 - observedQ1) / (theoreticalQ3 - theoreticalQ1)
  const referenceIntercept = observedQ1 - referenceSlope * theoreticalQ1

  return {
    sampleSize: sorted.length,
    points: sorted.map((observedReturn, index) => {
      const percentile = (index + 0.5) / sorted.length
      const theoreticalQuantile = inverseStandardNormal(percentile)
      return {
        percentile,
        theoreticalQuantile,
        observedReturn,
        referenceReturn: referenceIntercept + referenceSlope * theoreticalQuantile,
        tail: percentile <= 0.1 ? 'lower' : percentile >= 0.9 ? 'upper' : 'center',
      }
    }),
  }
}

export const buildTerminalNavDensity = (
  terminalValues: number[],
  pointCount = 81,
): TerminalNavDensity | null => {
  const sorted = terminalValues
    .filter((value) => Number.isFinite(value) && value > 0)
    .slice()
    .sort((left, right) => left - right)
  if (sorted.length < 2) return null

  const mean = sorted.reduce((sum, value) => sum + value, 0) / sorted.length
  const variance = sorted.reduce((sum, value) => sum + (value - mean) ** 2, 0) / Math.max(1, sorted.length - 1)
  const standardDeviation = Math.sqrt(Math.max(0, variance))
  const interquartileRange = quantile(sorted, 0.75) - quantile(sorted, 0.25)
  const robustScaleCandidates = [standardDeviation, interquartileRange / 1.34]
    .filter((value) => Number.isFinite(value) && value > 0)
  const robustScale = robustScaleCandidates.length > 0 ? Math.min(...robustScaleCandidates) : 0
  const minimumBandwidth = Math.max(Math.abs(mean) * 0.0005, 1e-6)
  const bandwidth = Math.max(minimumBandwidth, 0.9 * robustScale * sorted.length ** (-0.2))
  const lowerQuantile = quantile(sorted, 0.01)
  const upperQuantile = quantile(sorted, 0.99)
  const minNav = Math.max(0, lowerQuantile - bandwidth * 2)
  const maxNav = Math.max(minNav + minimumBandwidth, upperQuantile + bandwidth * 2)
  const safePointCount = Math.max(21, Math.min(161, Math.round(pointCount)))
  const normalizer = sorted.length * bandwidth * Math.sqrt(2 * Math.PI)
  const points = Array.from({ length: safePointCount }, (_, index) => {
    const nav = minNav + ((maxNav - minNav) * index) / (safePointCount - 1)
    const kernelSum = sorted.reduce((sum, value) => {
      const standardized = (nav - value) / bandwidth
      return sum + Math.exp(-0.5 * standardized ** 2)
    }, 0)
    return { nav, density: kernelSum / normalizer }
  })
  const modePoint = points.reduce((highest, point) => (point.density > highest.density ? point : highest))
  const binCount = Math.max(8, Math.min(28, Math.round(Math.sqrt(sorted.length))))
  const binWidth = (maxNav - minNav) / binCount
  const histogram = Array.from({ length: binCount }, (_, index) => ({
    lowerNav: minNav + index * binWidth,
    upperNav: minNav + (index + 1) * binWidth,
    count: 0,
    density: 0,
  }))
  sorted.forEach((value) => {
    const rawIndex = Math.floor((value - minNav) / binWidth)
    const index = Math.max(0, Math.min(binCount - 1, rawIndex))
    histogram[index].count += 1
  })
  histogram.forEach((bin) => {
    bin.density = bin.count / (sorted.length * binWidth)
  })
  const maxDensity = Math.max(modePoint.density, ...histogram.map((bin) => bin.density))
  return {
    sampleSize: sorted.length,
    points,
    histogram,
    maxDensity,
    modeNav: modePoint.nav,
    minNav,
    maxNav,
  }
}

interface SimulationInput {
  returnsPercent: number[]
  initialNav: number
  horizonDays: number
  pathCount: number
  seed: string
  targetReturnPercent?: number
}

interface SimulationEngineInput extends SimulationInput {
  method: SimulationMethod
  methodLabel: string
  sourceObservationCount: number
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
  createDailyReturn: (random: () => number) => () => number
}

const standardNormalSampler = (random: () => number) => {
  let spare: number | null = null
  return () => {
    if (spare !== null) {
      const value = spare
      spare = null
      return value
    }
    const first = Math.max(random(), Number.EPSILON)
    const second = random()
    const magnitude = Math.sqrt(-2 * Math.log(first))
    spare = magnitude * Math.sin(2 * Math.PI * second)
    return magnitude * Math.cos(2 * Math.PI * second)
  }
}

interface DistributionShape {
  mean: number
  standardDeviation: number
  skewness: number
  excessKurtosis: number
}

interface ShapeCalibration extends DistributionShape {
  skewParameter: number
  tailWeightParameter: number
  status: ShapeCalibrationStatus
}

const calculateDistributionShape = (values: number[]): DistributionShape | null => {
  if (values.length < 2) return null
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length
  const secondMoment = values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length
  if (!Number.isFinite(secondMoment) || secondMoment <= Number.EPSILON) return null
  const standardDeviation = Math.sqrt(secondMoment)
  const standardized = values.map((value) => (value - mean) / standardDeviation)
  const skewness = standardized.reduce((sum, value) => sum + value ** 3, 0) / values.length
  const excessKurtosis = standardized.reduce((sum, value) => sum + value ** 4, 0) / values.length - 3
  if (![mean, standardDeviation, skewness, excessKurtosis].every(Number.isFinite)) return null
  return { mean, standardDeviation, skewness, excessKurtosis }
}

const calculateAdjustedSampleShape = (values: number[]) => {
  const shape = calculateDistributionShape(values)
  if (!shape || values.length < 4) return null
  const sampleSize = values.length
  const skewness = (Math.sqrt(sampleSize * (sampleSize - 1)) / (sampleSize - 2)) * shape.skewness
  const excessKurtosis = (
    ((sampleSize - 1) / ((sampleSize - 2) * (sampleSize - 3)))
    * ((sampleSize + 1) * shape.excessKurtosis + 6)
  )
  if (![skewness, excessKurtosis].every(Number.isFinite)) return null
  return { skewness, excessKurtosis }
}

const sinhArcsinhValue = (normalValue: number, skewParameter: number, tailWeightParameter: number) => (
  Math.sinh((Math.asinh(normalValue) + skewParameter) / tailWeightParameter)
)

const SAS_CALIBRATION_SAMPLE = Array.from({ length: 801 }, (_, index) => (
  inverseStandardNormal((index + 0.5) / 801)
))

const evaluateShapeCandidate = (
  skewParameter: number,
  tailWeightParameter: number,
): DistributionShape | null => {
  const values = SAS_CALIBRATION_SAMPLE.map((normalValue) => (
    sinhArcsinhValue(normalValue, skewParameter, tailWeightParameter)
  ))
  return values.every(Number.isFinite) ? calculateDistributionShape(values) : null
}

const calibrateSinhArcsinhShape = (
  targetSkewness: number,
  targetExcessKurtosis: number,
): ShapeCalibration | null => {
  if (![targetSkewness, targetExcessKurtosis].every(Number.isFinite)) return null
  const skewScale = Math.max(0.5, Math.abs(targetSkewness))
  const kurtosisScale = Math.max(1, Math.abs(targetExcessKurtosis))
  const scoreFor = (shape: DistributionShape) => (
    ((shape.skewness - targetSkewness) / skewScale) ** 2
    + ((shape.excessKurtosis - targetExcessKurtosis) / kurtosisScale) ** 2
  )
  const normalScore = (targetSkewness / skewScale) ** 2 + (targetExcessKurtosis / kurtosisScale) ** 2
  let best: (DistributionShape & { skewParameter: number; tailWeightParameter: number; score: number }) | null = null
  const consider = (skewParameter: number, tailWeightParameter: number) => {
    if (Math.abs(skewParameter) > 2 || tailWeightParameter < 0.4 || tailWeightParameter > 3) return
    const shape = evaluateShapeCandidate(skewParameter, tailWeightParameter)
    if (!shape) return
    const score = scoreFor(shape)
    if (!best || score < best.score) {
      best = { ...shape, skewParameter, tailWeightParameter, score }
    }
  }

  const coarseSkewParameters = [-1.5, -1.2, -0.9, -0.6, -0.3, 0, 0.3, 0.6, 0.9, 1.2, 1.5]
  const coarseTailWeights = [0.45, 0.55, 0.65, 0.75, 0.85, 1, 1.15, 1.35, 1.6, 2, 2.5]
  coarseSkewParameters.forEach((skewParameter) => {
    coarseTailWeights.forEach((tailWeightParameter) => consider(skewParameter, tailWeightParameter))
  })
  if (!best) return null

  let skewStep = 0.15
  let tailStep = 0.12
  for (let iteration = 0; iteration < 18; iteration += 1) {
    const anchor = best
    ;[-1, 0, 1].forEach((skewDirection) => {
      ;[-1, 0, 1].forEach((tailDirection) => {
        if (skewDirection === 0 && tailDirection === 0) return
        consider(
          anchor.skewParameter + skewDirection * skewStep,
          anchor.tailWeightParameter + tailDirection * tailStep,
        )
      })
    })
    skewStep *= 0.72
    tailStep *= 0.72
  }

  const skewnessGap = Math.abs(best.skewness - targetSkewness)
  const kurtosisGap = Math.abs(best.excessKurtosis - targetExcessKurtosis)
  const matched = skewnessGap <= Math.max(0.08, Math.abs(targetSkewness) * 0.15)
    && kurtosisGap <= Math.max(0.25, Math.abs(targetExcessKurtosis) * 0.15)
  const status: ShapeCalibrationStatus = matched
    ? 'matched'
    : best.score < normalScore * 0.8
      ? 'approximate'
      : 'normal_fallback'
  return { ...best, status }
}

const summarizeSimulation = ({
  initialNav,
  horizonDays,
  pathCount,
  seed,
  targetReturnPercent = 0,
  method,
  methodLabel,
  sourceObservationCount,
  meanDailyLogReturn,
  dailyLogVolatility,
  historicalLogSkewness,
  historicalLogExcessKurtosis,
  fittedLogSkewness,
  fittedLogExcessKurtosis,
  shapeCalibrationStatus,
  shapeSkewParameter,
  tailWeightParameter,
  averageBlockLength,
  createDailyReturn,
}: SimulationEngineInput): FuturePathSimulation | null => {
  const safeHorizon = Math.floor(horizonDays)
  const safePathCount = Math.floor(pathCount)
  const safeTargetReturn = Number.isFinite(targetReturnPercent) ? targetReturnPercent : 0
  if (!Number.isFinite(initialNav) || initialNav <= 0 || safeHorizon < 1 || safePathCount < 1) {
    return null
  }
  const random = seededRandom(seed)
  const valuesByDay: number[][] = Array.from({ length: safeHorizon + 1 }, () => [])
  const samplePaths: number[][] = []
  const terminalValues: number[] = []
  const maxDrawdowns: number[] = []
  for (let pathIndex = 0; pathIndex < safePathCount; pathIndex += 1) {
    let nav = initialNav
    let peakNav = initialNav
    let maxDrawdown = 0
    const path = [nav]
    const drawDailyReturn = createDailyReturn(random)
    valuesByDay[0].push(nav)
    for (let day = 1; day <= safeHorizon; day += 1) {
      const sampledReturn = drawDailyReturn()
      if (!Number.isFinite(sampledReturn) || sampledReturn <= -1) return null
      nav *= 1 + sampledReturn
      if (!Number.isFinite(nav) || nav <= 0) return null
      peakNav = Math.max(peakNav, nav)
      maxDrawdown = Math.max(maxDrawdown, (peakNav - nav) / peakNav)
      valuesByDay[day].push(nav)
      if (pathIndex < 12) path.push(nav)
    }
    if (pathIndex < 12) samplePaths.push(path)
    terminalValues.push(nav)
    maxDrawdowns.push(maxDrawdown)
  }

  const percentiles = {
    p05: [] as number[],
    p25: [] as number[],
    p50: [] as number[],
    p75: [] as number[],
    p95: [] as number[],
  }
  valuesByDay.forEach((values) => {
    const sorted = values.slice().sort((left, right) => left - right)
    percentiles.p05.push(quantile(sorted, 0.05))
    percentiles.p25.push(quantile(sorted, 0.25))
    percentiles.p50.push(quantile(sorted, 0.5))
    percentiles.p75.push(quantile(sorted, 0.75))
    percentiles.p95.push(quantile(sorted, 0.95))
  })
  const terminalSorted = terminalValues.slice().sort((left, right) => left - right)
  const terminalReturns = terminalValues.map((value) => value / initialNav - 1)
  const terminalReturnsSorted = terminalReturns.slice().sort((left, right) => left - right)
  const tailBoundary = quantile(terminalReturnsSorted, 0.05)
  const tailReturns = terminalReturnsSorted.filter((value) => value <= tailBoundary)
  const tailAverage = tailReturns.reduce((sum, value) => sum + value, 0) / Math.max(1, tailReturns.length)
  const targetReturn = safeTargetReturn / 100
  return {
    method,
    methodLabel,
    days: Array.from({ length: safeHorizon + 1 }, (_, index) => index),
    samplePaths,
    terminalValues: terminalSorted,
    percentiles,
    terminal: {
      p05: quantile(terminalSorted, 0.05),
      p25: quantile(terminalSorted, 0.25),
      p50: quantile(terminalSorted, 0.5),
      p75: quantile(terminalSorted, 0.75),
      p95: quantile(terminalSorted, 0.95),
      lossProbability: terminalReturns.filter((value) => value < 0).length / safePathCount,
      valueAtRisk95: Math.max(0, -tailBoundary),
      conditionalValueAtRisk95: Math.max(0, -tailAverage),
      targetHitProbability: terminalReturns.filter((value) => value >= targetReturn).length / safePathCount,
      averageMaxDrawdown: maxDrawdowns.reduce((sum, value) => sum + value, 0) / safePathCount,
    },
    assumptions: {
      sourceObservationCount,
      targetReturnPercent: safeTargetReturn,
      meanDailyLogReturn,
      dailyLogVolatility,
      historicalLogSkewness,
      historicalLogExcessKurtosis,
      fittedLogSkewness,
      fittedLogExcessKurtosis,
      shapeCalibrationStatus,
      shapeSkewParameter,
      tailWeightParameter,
      averageBlockLength,
    },
  }
}

export const simulateParametricMonteCarlo = ({
  returnsPercent,
  ...input
}: SimulationInput): FuturePathSimulation | null => {
  const logReturns = returnsPercent
    .filter((value) => Number.isFinite(value) && value > -100)
    .map((value) => Math.log1p(value / 100))
  if (logReturns.length < MIN_SIMULATION_OBSERVATIONS) return null
  const mean = logReturns.reduce((sum, value) => sum + value, 0) / logReturns.length
  const variance = logReturns.reduce((sum, value) => sum + (value - mean) ** 2, 0) / (logReturns.length - 1)
  const volatility = Math.sqrt(Math.max(0, variance))
  const historicalShape = calculateAdjustedSampleShape(logReturns)
  const calibration = historicalShape && volatility > Number.EPSILON
    ? calibrateSinhArcsinhShape(historicalShape.skewness, historicalShape.excessKurtosis)
    : null
  const activeCalibration = calibration?.status === 'matched' || calibration?.status === 'approximate'
    ? calibration
    : null
  return summarizeSimulation({
    returnsPercent,
    ...input,
    method: 'parametric',
    methodLabel: activeCalibration
      ? '参数化蒙特卡洛（偏度/峰度校准）'
      : '参数化蒙特卡洛（正态安全降级）',
    sourceObservationCount: logReturns.length,
    meanDailyLogReturn: mean,
    dailyLogVolatility: volatility,
    historicalLogSkewness: historicalShape?.skewness ?? null,
    historicalLogExcessKurtosis: historicalShape?.excessKurtosis ?? null,
    fittedLogSkewness: activeCalibration?.skewness ?? 0,
    fittedLogExcessKurtosis: activeCalibration?.excessKurtosis ?? 0,
    shapeCalibrationStatus: activeCalibration?.status ?? 'normal_fallback',
    shapeSkewParameter: activeCalibration?.skewParameter ?? null,
    tailWeightParameter: activeCalibration?.tailWeightParameter ?? null,
    averageBlockLength: null,
    createDailyReturn: (random) => {
      const drawNormal = standardNormalSampler(random)
      return () => {
        const normalValue = drawNormal()
        const standardizedInnovation = activeCalibration
          ? (
            sinhArcsinhValue(
              normalValue,
              activeCalibration.skewParameter,
              activeCalibration.tailWeightParameter,
            ) - activeCalibration.mean
          ) / activeCalibration.standardDeviation
          : normalValue
        return Math.expm1(mean + volatility * standardizedInnovation)
      }
    },
  })
}

export const simulateStationaryBlockBootstrap = ({
  returnsPercent,
  averageBlockLength = 20,
  ...input
}: SimulationInput & { averageBlockLength?: number }): FuturePathSimulation | null => {
  const returns = returnsPercent.filter((value) => Number.isFinite(value) && value > -100)
  if (returns.length < MIN_SIMULATION_OBSERVATIONS) return null
  const blockLength = Math.max(1, Math.min(returns.length, Math.floor(averageBlockLength)))
  return summarizeSimulation({
    returnsPercent,
    ...input,
    method: 'block_bootstrap',
    methodLabel: '历史区块 Bootstrap',
    sourceObservationCount: returns.length,
    meanDailyLogReturn: null,
    dailyLogVolatility: null,
    historicalLogSkewness: null,
    historicalLogExcessKurtosis: null,
    fittedLogSkewness: null,
    fittedLogExcessKurtosis: null,
    shapeCalibrationStatus: null,
    shapeSkewParameter: null,
    tailWeightParameter: null,
    averageBlockLength: blockLength,
    createDailyReturn: (random) => {
      let sourceIndex = Math.floor(random() * returns.length)
      let firstDraw = true
      return () => {
        if (!firstDraw && random() >= 1 / blockLength) {
          sourceIndex = (sourceIndex + 1) % returns.length
        } else if (!firstDraw) {
          sourceIndex = Math.floor(random() * returns.length)
        }
        firstDraw = false
        return returns[sourceIndex] / 100
      }
    },
  })
}

export const compareSimulations = (
  parametric: FuturePathSimulation,
  bootstrap: FuturePathSimulation,
): SimulationComparison => {
  const initialNav = parametric.percentiles.p50[0]
  const returnFor = (value: number) => value / initialNav - 1
  const p05ReturnGap = Math.abs(returnFor(parametric.terminal.p05) - returnFor(bootstrap.terminal.p05))
  const medianReturnGap = Math.abs(returnFor(parametric.terminal.p50) - returnFor(bootstrap.terminal.p50))
  const lossProbabilityGap = Math.abs(parametric.terminal.lossProbability - bootstrap.terminal.lossProbability)
  const conditionalValueAtRiskGap = Math.abs(
    parametric.terminal.conditionalValueAtRisk95 - bootstrap.terminal.conditionalValueAtRisk95,
  )
  const largestGap = Math.max(p05ReturnGap, medianReturnGap, lossProbabilityGap, conditionalValueAtRiskGap)
  const level = largestGap >= 0.1 ? 'high' : largestGap >= 0.03 ? 'medium' : 'low'
  const message = level === 'high'
    ? '两种模型差异明显，结果对模型假设较敏感，决策时应采用更保守的尾部结果。'
    : level === 'medium'
      ? '两种模型存在一定差异，建议同时查看参数化假设与历史区块情景。'
      : '两种模型结果接近，但仍不代表对未来走势形成预测。'
  return { p05ReturnGap, medianReturnGap, lossProbabilityGap, conditionalValueAtRiskGap, level, message }
}

export const simulateHistoricalBootstrap = (input: SimulationInput): MonteCarloSimulation | null => (
  simulateStationaryBlockBootstrap({ ...input, averageBlockLength: 1 })
)

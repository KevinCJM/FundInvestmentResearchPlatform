import { describe, expect, it } from 'vitest'
import {
  buildNormalQqData,
  buildTerminalNavDensity,
  compareSimulations,
  interpretExcessKurtosis,
  interpretSkewness,
  selectStatisticsWindow,
  simulateHistoricalBootstrap,
  simulateParametricMonteCarlo,
  simulateStationaryBlockBootstrap,
} from './statisticalAnalysis'

describe('statistical analysis helpers', () => {
  const series = [
    { date: '2024-01-05', close: 1 },
    { date: '2025-01-03', close: 1.1 },
    { date: '2025-06-30', close: 1.2 },
    { date: '2026-01-05', close: 1.3 },
  ]

  it('默认成立以来保留全部样本，固定区间要求完整覆盖', () => {
    expect(selectStatisticsWindow(series, 'ALL').series).toEqual(series)
    const oneYear = selectStatisticsWindow(series, '1Y')
    expect(oneYear.complete).toBe(true)
    expect(oneYear.series.map((point) => point.date)).toEqual(['2025-01-03', '2025-06-30', '2026-01-05'])

    const insufficient = selectStatisticsWindow(series.slice(2), '1Y')
    expect(insufficient.complete).toBe(false)
    expect(insufficient.series).toEqual([])
    expect(insufficient.message).toContain('完整覆盖')
  })

  it('按偏度方向解释左偏、右偏与近似对称的金融含义', () => {
    expect(interpretSkewness(-0.25).label).toContain('左偏')
    expect(interpretSkewness(-0.25).meaning).toContain('下行尾部风险')
    expect(interpretSkewness(0.4).label).toContain('右偏')
    expect(interpretSkewness(0.4).meaning).toContain('较大盈利')
    expect(interpretSkewness(0.05).label).toBe('近似对称')
  })

  it('按超额峰度解释尖峰厚尾与平峰薄尾', () => {
    expect(interpretExcessKurtosis(7.07).label).toContain('尖峰厚尾')
    expect(interpretExcessKurtosis(7.07).meaning).toContain('低估尾部风险')
    expect(interpretExcessKurtosis(-0.8).label).toContain('平峰薄尾')
    expect(interpretExcessKurtosis(0.02).label).toBe('接近正态峰度')
  })

  it('构建排序稳定、尾部可识别的正态 Q-Q 数据', () => {
    const qqData = buildNormalQqData([3, Number.NaN, -4, 0, 1, -1, 8])

    expect(qqData?.sampleSize).toBe(6)
    expect(qqData?.points).toHaveLength(6)
    expect(qqData?.points.map((point) => point.observedReturn)).toEqual([-4, -1, 0, 1, 3, 8])
    expect(qqData?.points[0].theoreticalQuantile).toBeLessThan(0)
    expect(qqData?.points[5].theoreticalQuantile).toBeGreaterThan(0)
    expect(qqData?.points[0].tail).toBe('lower')
    expect(qqData?.points[5].tail).toBe('upper')
    expect(qqData?.points[2].tail).toBe('center')
    expect(qqData?.points.every((point) => Number.isFinite(point.referenceReturn))).toBe(true)
  })

  it('Q-Q 图样本不足时返回空，常数样本仍生成水平参考线', () => {
    expect(buildNormalQqData([0, 1])).toBeNull()

    const flat = buildNormalQqData([0.2, 0.2, 0.2, 0.2])
    expect(flat?.points.every((point) => point.referenceReturn === 0.2)).toBe(true)
  })

  it('将模拟期末净值构建为平滑概率密度并保留净值方向', () => {
    const density = buildTerminalNavDensity([0.94, 0.98, 1, 1.01, 1.02, 1.08, Number.NaN])

    expect(density?.sampleSize).toBe(6)
    expect(density?.points).toHaveLength(81)
    expect(density?.points[0].nav).toBeLessThan(density?.points[80].nav ?? 0)
    expect(density?.points.every((point) => point.density >= 0 && Number.isFinite(point.density))).toBe(true)
    expect(density?.histogram.length).toBeGreaterThanOrEqual(8)
    expect(density?.histogram.reduce((sum, bin) => sum + bin.count, 0)).toBe(6)
    expect(density?.histogram.every((bin) => bin.density >= 0 && bin.upperNav > bin.lowerNav)).toBe(true)
    expect(density?.maxDensity).toBeGreaterThan(0)
    expect(density?.modeNav).toBeGreaterThanOrEqual(0.94)
    expect(density?.modeNav).toBeLessThanOrEqual(1.08)
    expect(buildTerminalNavDensity([1])).toBeNull()
  })

  it('历史重采样模拟可复现并返回完整分位路径与终值概率', () => {
    const input = {
      returnsPercent: Array.from({ length: 40 }, (_, index) => [-2, -1, 0.5, 1, 3][index % 5]),
      initialNav: 1.25,
      horizonDays: 21,
      pathCount: 200,
      seed: '510300.SH-ALL-1',
    }
    const first = simulateHistoricalBootstrap(input)
    const second = simulateHistoricalBootstrap(input)

    expect(first).toEqual(second)
    expect(first?.days).toHaveLength(22)
    expect(first?.samplePaths).toHaveLength(12)
    expect(first?.terminalValues).toHaveLength(200)
    expect(first?.terminalValues).toEqual([...(first?.terminalValues ?? [])].sort((left, right) => left - right))
    expect(first?.percentiles.p50).toHaveLength(22)
    expect(first?.percentiles.p05[0]).toBeCloseTo(1.25)
    expect(first?.terminal.p05).toBeLessThanOrEqual(first?.terminal.p50 ?? 0)
    expect(first?.terminal.p50).toBeLessThanOrEqual(first?.terminal.p95 ?? 0)
    expect(first?.terminal.lossProbability).toBeGreaterThanOrEqual(0)
    expect(first?.terminal.lossProbability).toBeLessThanOrEqual(1)
    expect(first?.terminal.valueAtRisk95).toBeGreaterThanOrEqual(0)
    expect(first?.terminal.conditionalValueAtRisk95).toBeGreaterThanOrEqual(first?.terminal.valueAtRisk95 ?? 0)
    expect(first?.terminal.averageMaxDrawdown).toBeGreaterThanOrEqual(0)
    expect(first?.terminal.targetHitProbability).toBeGreaterThanOrEqual(0)
    expect(first?.assumptions.averageBlockLength).toBe(1)
  })

  it('参数化蒙特卡洛校准对数收益四矩且固定种子可复现', () => {
    const input = {
      returnsPercent: Array.from({ length: 40 }, (_, index) => [-2, -1, 0.5, 1, 3][index % 5]),
      initialNav: 1.25,
      horizonDays: 63,
      pathCount: 200,
      targetReturnPercent: 5,
      seed: 'parametric-1',
    }
    const first = simulateParametricMonteCarlo(input)
    const second = simulateParametricMonteCarlo(input)

    expect(first).toEqual(second)
    expect(first?.method).toBe('parametric')
    expect(first?.assumptions.sourceObservationCount).toBe(40)
    expect(first?.assumptions.meanDailyLogReturn).not.toBeNull()
    expect(first?.assumptions.dailyLogVolatility).toBeGreaterThan(0)
    expect(first?.assumptions.historicalLogSkewness).not.toBeNull()
    expect(first?.assumptions.historicalLogExcessKurtosis).not.toBeNull()
    expect(['matched', 'approximate']).toContain(first?.assumptions.shapeCalibrationStatus)
    expect(first?.assumptions.fittedLogSkewness).not.toBe(0)
    expect(first?.assumptions.shapeSkewParameter).not.toBeNull()
    expect(first?.assumptions.tailWeightParameter).not.toBeNull()
    expect(first?.assumptions.averageBlockLength).toBeNull()
    expect(first?.assumptions.targetReturnPercent).toBe(5)
    expect(first?.terminal.targetHitProbability).toBeGreaterThanOrEqual(0)
    expect(first?.terminal.targetHitProbability).toBeLessThanOrEqual(1)
  })

  it('参数化蒙特卡洛保留左偏与尖峰厚尾，而不是强制改成正态分布', () => {
    const returnsPercent = Array.from({ length: 80 }, (_, index) => {
      if (index % 20 === 0) return -8
      if (index % 13 === 0) return 3.5
      return [0.1, 0.2, 0.3, -0.1, 0.15][index % 5]
    })
    const simulation = simulateParametricMonteCarlo({
      returnsPercent,
      initialNav: 1,
      horizonDays: 21,
      pathCount: 500,
      seed: 'four-moment-left-tail',
    })

    expect(simulation?.assumptions.historicalLogSkewness).toBeLessThan(0)
    expect(simulation?.assumptions.historicalLogExcessKurtosis).toBeGreaterThan(0)
    expect(['matched', 'approximate']).toContain(simulation?.assumptions.shapeCalibrationStatus)
    expect(simulation?.assumptions.fittedLogSkewness).toBeLessThan(0)
    expect(simulation?.assumptions.fittedLogExcessKurtosis).toBeGreaterThan(0)
    expect(Math.abs(
      (simulation?.assumptions.fittedLogSkewness ?? 0)
      - (simulation?.assumptions.historicalLogSkewness ?? 0),
    )).toBeLessThan(0.5)
    expect(Math.abs(
      (simulation?.assumptions.fittedLogExcessKurtosis ?? 0)
      - (simulation?.assumptions.historicalLogExcessKurtosis ?? 0),
    )).toBeLessThan(2)
  })

  it('少于 20 个历史收益观察值时拒绝生成容易误导的未来路径', () => {
    const shortInput = {
      returnsPercent: Array.from({ length: 19 }, () => 0.1),
      initialNav: 1,
      horizonDays: 21,
      pathCount: 200,
      seed: 'short',
    }
    expect(simulateParametricMonteCarlo(shortInput)).toBeNull()
    expect(simulateStationaryBlockBootstrap(shortInput)).toBeNull()
  })

  it('零波动样本无法识别偏度峰度时明确执行正态安全降级', () => {
    const simulation = simulateParametricMonteCarlo({
      returnsPercent: Array.from({ length: 30 }, () => 0.1),
      initialNav: 1,
      horizonDays: 21,
      pathCount: 200,
      seed: 'flat-shape-fallback',
    })

    expect(simulation?.methodLabel).toContain('正态安全降级')
    expect(simulation?.assumptions.shapeCalibrationStatus).toBe('normal_fallback')
    expect(simulation?.assumptions.historicalLogSkewness).toBeNull()
    expect(simulation?.assumptions.shapeSkewParameter).toBeNull()
    expect(simulation?.terminal.p05).toBeCloseTo(simulation?.terminal.p95 ?? 0)
  })

  it('区块 Bootstrap 保留连续历史片段并公开实际区块长度', () => {
    const block = simulateStationaryBlockBootstrap({
      returnsPercent: Array.from({ length: 36 }, (_, index) => [-2, -1, 0, 1, 2, 3][index % 6]),
      initialNav: 1,
      horizonDays: 21,
      pathCount: 200,
      averageBlockLength: 20,
      targetReturnPercent: 3,
      seed: 'block-1',
    })
    const iid = simulateHistoricalBootstrap({
      returnsPercent: Array.from({ length: 36 }, (_, index) => [-2, -1, 0, 1, 2, 3][index % 6]),
      initialNav: 1,
      horizonDays: 21,
      pathCount: 200,
      targetReturnPercent: 3,
      seed: 'block-1',
    })

    expect(block?.method).toBe('block_bootstrap')
    expect(block?.assumptions.averageBlockLength).toBe(20)
    expect(block?.terminalValues).not.toEqual(iid?.terminalValues)
    expect(block?.terminal.conditionalValueAtRisk95).toBeGreaterThanOrEqual(block?.terminal.valueAtRisk95 ?? 0)
  })

  it('双模型对比会把显著差异识别为模型风险', () => {
    const common = {
      returnsPercent: Array.from({ length: 45 }, (_, index) => [-8, -3, -1, 0, 0.2, 0.5, 1, 2, 7][index % 9]),
      initialNav: 1,
      horizonDays: 252,
      pathCount: 500,
      targetReturnPercent: 5,
    }
    const parametric = simulateParametricMonteCarlo({ ...common, seed: 'compare-parametric' })
    const bootstrap = simulateStationaryBlockBootstrap({ ...common, seed: 'compare-bootstrap', averageBlockLength: 5 })

    expect(parametric).not.toBeNull()
    expect(bootstrap).not.toBeNull()
    const comparison = compareSimulations(parametric!, bootstrap!)
    expect(comparison.p05ReturnGap).toBeGreaterThanOrEqual(0)
    expect(comparison.lossProbabilityGap).toBeGreaterThanOrEqual(0)
    expect(['low', 'medium', 'high']).toContain(comparison.level)
    expect(comparison.message).toContain('模型')
  })
})

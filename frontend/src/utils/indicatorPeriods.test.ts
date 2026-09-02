import { describe, expect, it } from 'vitest'
import { indicatorPeriodLabel, indicatorPeriodOptionLabel } from './indicatorPeriods'

describe('indicatorPeriodLabel', () => {
  it('区分滚动、完整自然区间和成立以来', () => {
    expect(indicatorPeriodLabel('1Y')).toBe('近 1 年')
    expect(indicatorPeriodLabel('Y1')).toBe('去年')
    expect(indicatorPeriodLabel('Y2')).toBe('前年')
    expect(indicatorPeriodLabel('W1')).toBe('上周')
    expect(indicatorPeriodLabel('W2')).toBe('上上周')
    expect(indicatorPeriodLabel('ALL')).toBe('成立以来')
    expect(indicatorPeriodOptionLabel('Y1')).toBe('去年（Y1）')
  })
})

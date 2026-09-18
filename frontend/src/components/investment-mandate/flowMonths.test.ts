import { describe, expect, it } from 'vitest'
import { lastPaymentMonth, levelReaches } from './model'

describe('derived cash flow end month', () => {
  it('runs to the last payment that still fits the horizon', () => {
    expect(lastPaymentMonth(1, 1, 12)).toBe(12)
    expect(lastPaymentMonth(2, 3, 12)).toBe(11)
    expect(lastPaymentMonth(1, 12, 36)).toBe(25)
    expect(lastPaymentMonth(12, 3, 12)).toBe(12)
  })

  it('never lands before the first month', () => {
    expect(lastPaymentMonth(1, 1, 0)).toBe(1)
    expect(lastPaymentMonth(10, 12, 12)).toBe(10)
  })
})

describe('level reference return against the stated target', () => {
  it('compares only when both numbers exist', () => {
    expect(levelReaches(.052, null)).toBeNull()
    expect(levelReaches(null, .06)).toBeNull()
    expect(levelReaches(NaN, .06)).toBeNull()
  })

  it('reads equal as reached, lower as short', () => {
    expect(levelReaches(.06, .06)).toBe(true)
    expect(levelReaches(.061, .06)).toBe(true)
    expect(levelReaches(.052, .06)).toBe(false)
  })
})

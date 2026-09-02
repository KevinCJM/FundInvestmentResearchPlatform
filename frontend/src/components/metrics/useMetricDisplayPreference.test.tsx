import { act, renderHook, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it } from 'vitest'
import { groupIndicatorsByPeriod, useMetricDisplayPreference } from './useMetricDisplayPreference'

describe('指标展示浏览器记忆', () => {
  beforeEach(() => window.localStorage.clear())

  it('剔除目录中已失效的指标并回退到系统默认项', async () => {
    window.localStorage.setItem('indicator-display:v1:detail:single_product', JSON.stringify({
      indicatorIds: ['removed-metric'],
      period: '3M',
      products: ['不应读取'],
    }))
    const { result } = renderHook(() => useMetricDisplayPreference(
      'detail', 'single_product', ['default-metric'], '1Y', ['default-metric', 'other-metric'],
    ))

    await waitFor(() => expect(result.current[0]).toEqual({
      indicatorIds: ['default-metric'],
      periodsByIndicator: { 'default-metric': '3M' },
    }))
    expect(JSON.parse(window.localStorage.getItem('indicator-display:v2:detail:single_product') ?? '{}')).toEqual({
      indicatorIds: ['default-metric'],
      periodsByIndicator: { 'default-metric': '3M' },
    })
  })

  it('按各指标自己的区间分组，同一区间只生成一组请求参数', () => {
    expect(groupIndicatorsByPeriod({
      indicatorIds: ['return', 'volatility', 'drawdown'],
      periodsByIndicator: { return: '1Y', volatility: '3M', drawdown: '1Y' },
    }, '1Y')).toEqual([
      { period: '1Y', indicatorIds: ['return', 'drawdown'] },
      { period: '3M', indicatorIds: ['volatility'] },
    ])
  })

  it('允许用户主动移除最后一个指标并持久化为空选择', async () => {
    const { result } = renderHook(() => useMetricDisplayPreference(
      'detail', 'single_product', ['default-metric'], '1Y', ['default-metric'],
    ))

    act(() => result.current[1]({ indicatorIds: [], periodsByIndicator: {} }))

    await waitFor(() => expect(result.current[0]).toEqual({ indicatorIds: [], periodsByIndicator: {} }))
    expect(JSON.parse(window.localStorage.getItem('indicator-display:v2:detail:single_product') ?? '{}')).toEqual({
      indicatorIds: [],
      periodsByIndicator: {},
    })
  })
})

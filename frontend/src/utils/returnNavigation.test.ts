import { describe, expect, it, vi } from 'vitest'
import { buildReturnNavigationState, readReturnNavigationState, returnToOrigin } from './returnNavigation'

describe('returnNavigation', () => {
  it('保存完整来源地址并拒绝站外返回地址', () => {
    expect(buildReturnNavigationState(
      { pathname: '/manual-construction', search: '?draft=1', hash: '#class-a' },
      '返回手动构建大类',
    )).toEqual({
      returnTo: '/manual-construction?draft=1#class-a',
      returnLabel: '返回手动构建大类',
    })
    expect(readReturnNavigationState({ returnTo: '//example.com', returnLabel: '站外' })).toBeNull()
    expect(readReturnNavigationState({ returnTo: 'https://example.com', returnLabel: '站外' })).toBeNull()
  })

  it('有历史记录时后退，无历史记录时使用安全来源或兜底页', () => {
    const navigate = vi.fn()
    returnToOrigin(navigate, {
      key: 'detail-entry',
      state: { returnTo: '/manual-construction', returnLabel: '返回手动构建大类' },
    }, '/research?kind=etf')
    expect(navigate).toHaveBeenLastCalledWith(-1)

    navigate.mockClear()
    returnToOrigin(navigate, {
      key: 'default',
      state: { returnTo: '/manual-construction', returnLabel: '返回手动构建大类' },
    }, '/research?kind=etf')
    expect(navigate).toHaveBeenLastCalledWith('/manual-construction', { replace: true })

    navigate.mockClear()
    returnToOrigin(navigate, { key: 'default', state: null }, '/research?kind=etf')
    expect(navigate).toHaveBeenLastCalledWith('/research?kind=etf', { replace: true })
  })
})

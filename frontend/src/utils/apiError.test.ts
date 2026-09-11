import { describe, expect, it } from 'vitest'
import { apiErrorMessage } from './apiError'

describe('apiErrorMessage', () => {
  it('passes a plain string detail through', () => {
    expect(apiErrorMessage({ detail: '不支持的算法：nope' }, 'x')).toBe('不支持的算法：nope')
  })

  it('renders a FastAPI 422 body instead of [object Object]', () => {
    const body = {
      detail: [
        { type: 'literal_error', loc: ['body', 'algorithm'], msg: "Input should be 'rule' or 'gmm'" },
        { type: 'missing', loc: ['body', 'products'], msg: 'Field required' },
      ],
    }
    const message = apiErrorMessage(body, 'fallback')
    expect(message).toBe("algorithm: Input should be 'rule' or 'gmm'；products: Field required")
    expect(message).not.toContain('[object Object]')
  })

  it('falls back when there is no detail at all', () => {
    expect(apiErrorMessage({}, '后端错误 500')).toBe('后端错误 500')
    expect(apiErrorMessage(null, '后端错误 500')).toBe('后端错误 500')
    expect(apiErrorMessage({ detail: '   ' }, '后端错误 500')).toBe('后端错误 500')
  })

  it('never returns [object Object] for an unrecognised shape', () => {
    expect(apiErrorMessage({ detail: { code: 42 } }, 'x')).toBe('{"code":42}')
  })
})

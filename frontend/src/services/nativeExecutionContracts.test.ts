import { afterEach, describe, expect, it, vi } from 'vitest'
import { analyzeProductComparison } from './productCompare'
import { requestEqualWeights } from './strategyWeights'
import { cppAotAudit } from '../test/cppAotFixture'

const njit = {
  backend: 'numba_njit_fixed_signature', execution_backend: 'numba_njit_fixed_signature',
  nopython: true, njit_required: true, object_mode: 0, python_fallback: 0,
  request_time_compilation: 0, kernel_signatures: { kernel: ['float64[:]'] },
}

afterEach(() => { vi.unstubAllGlobals() })

describe.each([
  ['equal weights', () => requestEqualWeights(2)],
  ['product comparison', () => analyzeProductComparison('510300.SH', 'etf', {
    rolling_window_days: 20, management_fee: null, custody_fee: null,
    ranges: {
      performance: { start_date: null, end_date: null },
      risk: { start_date: null, end_date: null },
      efficiency: { start_date: null, end_date: null },
    },
  })],
] as const)('%s execution contract', (_name, request) => {
  function response(execution: Record<string, unknown>) {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: true, status: 200,
      json: async () => ({
        weights: [0.5, 0.5], execution,
        ranges: { performance: {}, risk: {}, efficiency: {} },
      }),
    }))
  }

  it.each(['backend', 'execution_backend', 'njit_required'])('rejects incomplete NJIT proof missing %s', async key => {
    const audit: Record<string, unknown> = { ...njit }
    delete audit[key]
    response(audit)
    await expect(request()).rejects.toThrow('固定签名 NJIT 执行校验')
  })

  it.each([njit, cppAotAudit])('accepts a complete supported proof', async audit => {
    response(audit)
    await expect(request()).resolves.toHaveProperty('execution', audit)
  })
})

import { afterEach, describe, expect, it, vi } from 'vitest'
import { evaluateCustomIndicators } from './customIndicators'

const execution = {
  execution_backend: 'numba_njit_fixed_signature',
  nopython: true,
  object_mode: 0,
  python_fallback: 0,
  request_time_compilation: 0,
  kernel_signatures: { typed_indicator_plan: ['fixed'] },
}

const payload = {
  results: [],
  summary: { total: 0, ok: 0, warning: 0, error: 0 },
  cache: { hits: 0, misses: 0 },
  execution,
}

describe('custom indicator calculation client', () => {
  afterEach(() => vi.unstubAllGlobals())

  it('只接受带固定签名 NJIT 执行证明的指标结果', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => payload,
    }))

    await expect(evaluateCustomIndicators({
      targets: [{ kind: 'etf', product_id: '510300.SH' }],
      period: '1Y',
    })).resolves.toEqual(payload)
  })

  it('执行证明缺失或存在回退时失败关闭', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({
        ...payload,
        execution: { ...execution, kernel_signatures: {} },
      }),
    }))

    await expect(evaluateCustomIndicators({
      targets: [{ kind: 'etf', product_id: '510300.SH' }],
      period: '1Y',
    })).rejects.toThrow('指标与评价计算未提供有效的固定签名 NJIT 执行证明')
  })
})

import { afterEach, describe, expect, it, vi } from 'vitest'
import { evaluateCustomIndicators } from './customIndicators'

const execution = {
  execution_backend: 'numba_njit_fixed_signature', nopython: true,
  object_mode: 0, python_fallback: 0, request_time_compilation: 0,
  kernel_signatures: { typed_indicator_plan: ['fixed'] },
}
const payload = {
  results: [], summary: { total: 0, ok: 0, warning: 0, error: 0 },
  cache: { hits: 0, misses: 0 }, execution,
}
const input = {
  indicator_ids: ['metric-a', 'metric-b'],
  targets: [{ kind: 'etf' as const, product_id: '510300.SH' }], period: '1Y',
}
const refs = [{ indicator_id: 'metric-a', indicator_revision: 2 }, { indicator_id: 'metric-b', indicator_revision: 4 }]
const response = (value: unknown) => ({ ok: true, status: 200, json: async () => value })

describe('independent indicator preparation and execution client', () => {
  afterEach(() => { vi.unstubAllGlobals() })

  it('先准备共享计划，再使用服务器锁定版本计算；不发送子结果', async () => {
    const fetcher = vi.fn()
      .mockResolvedValueOnce(response({ prepared: true, plans: [], indicator_refs: refs }))
      .mockResolvedValueOnce(response(payload))
    vi.stubGlobal('fetch', fetcher)
    await expect(evaluateCustomIndicators(input)).resolves.toEqual(payload)
    expect(fetcher).toHaveBeenCalledTimes(2)
    expect(fetcher.mock.calls[0][0]).toBe('/api/custom-indicators/prepare')
    expect(JSON.parse(fetcher.mock.calls[0][1].body)).toEqual({ indicator_ids: input.indicator_ids, indicator_refs: [] })
    expect(fetcher.mock.calls[1][0]).toBe('/api/custom-indicators/evaluate')
    expect(JSON.parse(fetcher.mock.calls[1][1].body)).toEqual({ ...input, indicator_ids: [], indicator_refs: refs })
  })

  it.each([
    { ...execution, kernel_signatures: {} },
    { ...execution, python_fallback: 1 },
    { ...execution, request_time_compilation: 1 },
  ])('缺失固定签名执行证明或出现回退时失败关闭', async invalidExecution => {
    vi.stubGlobal('fetch', vi.fn()
      .mockResolvedValueOnce(response({ prepared: true, plans: [], indicator_refs: refs }))
      .mockResolvedValueOnce(response({ ...payload, execution: invalidExecution })))
    await expect(evaluateCustomIndicators(input)).rejects.toThrow('指标与评价计算未提供有效的固定签名 NJIT 执行证明')
  })

  it('准备失败不进入正式计算，也不回退逐指标计算', async () => {
    const fetcher = vi.fn().mockResolvedValue(response({ prepared: false }))
    vi.stubGlobal('fetch', fetcher)
    await expect(evaluateCustomIndicators(input)).rejects.toThrow('计算计划尚未准备完成')
    expect(fetcher).toHaveBeenCalledTimes(1)
  })
})

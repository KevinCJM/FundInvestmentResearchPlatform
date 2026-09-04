import { afterEach, describe, expect, it, vi } from 'vitest'
import { evaluateNumericControls } from './businessNumeric'

const execution = {
  execution_backend: 'numba_njit_fixed_signature' as const,
  nopython: true as const,
  object_mode: 0 as const,
  python_fallback: 0 as const,
  request_time_compilation: 0 as const,
  kernel_signatures: { numeric_control_kernel: ['fixed'] },
}

describe('businessNumeric client', () => {
  afterEach(() => vi.unstubAllGlobals())

  it('发送数值分组并只接受固定签名 NJIT 返回', async () => {
    const fetchMock = vi.fn(async () => ({
      ok: true,
      json: async () => ({
        items: [{ key: 'weights', total: 100, difference: 0, within_tolerance: true, positive: true, normalized_shares: [0.6, 0.4] }],
        execution,
      }),
    }))
    vi.stubGlobal('fetch', fetchMock)

    const result = await evaluateNumericControls([
      { key: 'weights', values: [60, 40], target: 100, tolerance: 0.01 },
    ])

    expect(result.items[0].normalized_shares).toEqual([0.6, 0.4])
    expect(fetchMock).toHaveBeenCalledWith('/api/business-numeric/controls', expect.objectContaining({
      method: 'POST',
      body: JSON.stringify({ groups: [{ key: 'weights', values: [60, 40], target: 100, tolerance: 0.01 }] }),
    }))
  })

  it('拒绝 Python 回退或请求期编译的结果', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => ({
      ok: true,
      json: async () => ({ items: [], execution: { ...execution, python_fallback: 1 } }),
    })))

    await expect(evaluateNumericControls([{ key: 'weights', values: [100] }]))
      .rejects.toThrow('固定签名 NJIT 执行证明')
  })
})

import { afterEach, describe, expect, it, vi } from 'vitest'
import {
  ProductAnalysisApiError,
  analyzeProduct,
  type ProductAnalysisRequest,
  type ProductAnalysisResponse,
} from './productAnalysis'

const request: ProductAnalysisRequest = {
  statistics_period: 'ALL',
  include_simulation: false,
  analysis_basis: 'adjusted_nav',
  price_ma_periods: [5, 20],
  volume_ma_periods: [5],
  boll_period: 20,
  boll_multiplier: 2,
  kdj_period: 9,
  kdj_k_smoothing: 3,
  kdj_d_smoothing: 3,
  histogram_bin_width: 0.2,
  simulation_horizon: 252,
  simulation_path_count: 500,
  bootstrap_block_length: 20,
  fhs_ewma_lambda: 0.94,
  simulation_target_return: 5,
  simulation_run: 0,
  regime: null,
}

const response = {
  schema_version: 1,
  product_id: '510300.SH',
  execution: {
    execution_backend: 'numba_njit_fixed_signature',
    engine: 'product-analysis-njit-1.0.0',
    kernel_version: 'product-chart-statistics-simulation-2',
    kernel_coverage: '21/21',
    kernel_fingerprint: 'fingerprint',
    kernel_signatures: { product_analysis_kernel: ['fixed'] },
    nopython: true,
    njit_required: true,
    object_mode: 0,
    python_fallback: 0,
    request_time_compilation: 0,
  },
} as unknown as ProductAnalysisResponse

describe('product analysis API', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('将全部计算参数提交给后端并返回 NJIT 执行审计', async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => response })
    vi.stubGlobal('fetch', fetchMock)

    const result = await analyzeProduct('510300.SH', 'etf', request)

    expect(fetchMock).toHaveBeenCalledWith(
      '/api/instruments/products/510300.SH/analysis?kind=etf',
      expect.objectContaining({ method: 'POST', body: JSON.stringify(request) }),
    )
    expect(result.execution).toMatchObject({
      execution_backend: 'numba_njit_fixed_signature',
      nopython: true,
      python_fallback: 0,
    })
  })

  it('历史情景只提交已发布运行引用，不上传 states/segments', async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, status: 200, json: async () => response })
    vi.stubGlobal('fetch', fetchMock)

    await analyzeProduct('510300.SH', 'etf', { ...request, regime: { run_id: 'run-1', publication_id: 'publication-1', state_id: 'bear', segment_id: 'segment-2' } })

    const body = JSON.parse(String(fetchMock.mock.calls[0]?.[1]?.body))
    expect(body.regime).toEqual({ run_id: 'run-1', publication_id: 'publication-1', state_id: 'bear', segment_id: 'segment-2' })
    expect(body.include_simulation).toBe(false)
    expect(body.analysis_basis).toBe('adjusted_nav')
    expect(body.regime).not.toHaveProperty('states')
    expect(body.regime).not.toHaveProperty('segments')
  })

  it('后端失败时显式抛错，不提供浏览器本地计算回退', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: false,
      status: 422,
      json: async () => ({ detail: '预热内核不可用' }),
    }))

    await expect(analyzeProduct('510300.SH', 'etf', request)).rejects.toEqual(
      new ProductAnalysisApiError(422, '预热内核不可用'),
    )
  })

  it('拒绝缺少固定签名或请求期编译的成功响应', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({
        ...response,
        execution: { ...response.execution, request_time_compilation: 1 },
      }),
    }))

    await expect(analyzeProduct('510300.SH', 'etf', request))
      .rejects.toThrow('产品分析未提供有效的固定签名 NJIT 执行证明')
  })
})

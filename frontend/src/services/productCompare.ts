export interface ProductCompareRangeRequest {
  start_date: string | null
  end_date: string | null
}

export interface ProductCompareAnalysisRequest {
  ranges: {
    performance: ProductCompareRangeRequest
    risk: ProductCompareRangeRequest
    efficiency: ProductCompareRangeRequest
  }
  rolling_window_days: number
  management_fee: number | null
  custody_fee: number | null
}

export interface ProductCompareMetrics {
  cumulativeReturn: number | null
  annualizedReturn: number | null
  volatility: number | null
  maxDrawdown: number | null
  totalFee: number | null
  returnToFee: number | null
  sharpeRatio: number | null
  calmarRatio: number | null
}

export interface ProductCompareSeriesPoint {
  date: string
  value: number | null
}

export interface ProductCompareRangeResult {
  window: {
    start_date: string
    end_date: string
    observation_count: number
  }
  metrics: ProductCompareMetrics
  normalized_nav: ProductCompareSeriesPoint[]
  drawdown: ProductCompareSeriesPoint[]
  rolling_volatility: ProductCompareSeriesPoint[]
}

export interface ProductCompareExecutionAudit {
  backend: 'numba_njit_fixed_signature'
  execution_backend: 'numba_njit_fixed_signature'
  engine: string
  kernel_version: string
  kernel_coverage: string
  kernel_signatures: Record<string, string[]>
  kernel_fingerprint: string
  nopython: true
  object_mode: 0
  njit_required: true
  python_fallback: 0
  request_time_compilation: 0
}

export interface ProductCompareAnalysisResponse {
  schema_version: number
  product_id: string
  ranges: {
    performance: ProductCompareRangeResult
    risk: ProductCompareRangeResult
    efficiency: ProductCompareRangeResult
  }
  execution: ProductCompareExecutionAudit
}

export class ProductCompareApiError extends Error {
  constructor(readonly status: number, message: string) {
    super(message)
    this.name = 'ProductCompareApiError'
  }
}

export async function analyzeProductComparison(
  productId: string,
  kind: 'etf' | 'fund',
  request: ProductCompareAnalysisRequest,
  signal?: AbortSignal,
): Promise<ProductCompareAnalysisResponse> {
  const response = await fetch(
    `/api/instruments/products/${encodeURIComponent(productId)}/compare-analysis?kind=${kind}`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
      signal,
    },
  )
  if (!response.ok) {
    let message = `产品比较指标计算失败（${response.status}）`
    try {
      const payload = await response.json()
      if (typeof payload?.detail === 'string') message = payload.detail
    } catch { /* retain stable Chinese fallback */ }
    throw new ProductCompareApiError(response.status, message)
  }
  let payload: unknown
  try {
    payload = await response.json()
  } catch {
    throw new ProductCompareApiError(response.status, '产品比较服务返回了无效响应')
  }
  const result = payload as Partial<ProductCompareAnalysisResponse>
  const execution = result.execution
  if (
    !execution
    || execution.backend !== 'numba_njit_fixed_signature'
    || execution.execution_backend !== 'numba_njit_fixed_signature'
    || execution.nopython !== true
    || execution.njit_required !== true
    || execution.object_mode !== 0
    || execution.python_fallback !== 0
    || execution.request_time_compilation !== 0
    || !execution.kernel_signatures
    || typeof execution.kernel_signatures !== 'object'
    || Object.keys(execution.kernel_signatures).length === 0
    || Object.values(execution.kernel_signatures).some(
      (signatures) => !Array.isArray(signatures) || signatures.length === 0,
    )
  ) {
    throw new ProductCompareApiError(
      response.status,
      '产品比较服务未通过固定签名 NJIT 执行校验',
    )
  }
  if (!result.ranges?.performance || !result.ranges.risk || !result.ranges.efficiency) {
    throw new ProductCompareApiError(response.status, '产品比较服务返回的区间结果不完整')
  }
  return result as ProductCompareAnalysisResponse
}

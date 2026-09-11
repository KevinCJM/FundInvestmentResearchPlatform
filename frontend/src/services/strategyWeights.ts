export interface StrategyExecutionAudit {
  backend: 'numba_njit_fixed_signature'
  execution_backend: 'numba_njit_fixed_signature'
  kernel_version: string
  kernel_coverage: string
  kernel_signatures: Record<string, string[]>
  fingerprint: string
  nopython: true
  njit_required: true
  object_mode: 0
  python_fallback: 0
  request_time_compilation: 0
}

export interface EqualWeightsResponse {
  weights: number[]
  execution: StrategyExecutionAudit
}

export async function requestEqualWeights(
  assetCount: number,
  signal?: AbortSignal,
): Promise<EqualWeightsResponse> {
  const response = await fetch('/api/strategy/equal-weights', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ asset_count: assetCount, max_leverage: 0 }),
    signal,
  })
  let payload: unknown
  try {
    payload = await response.json()
  } catch {
    throw new Error(`等权计算服务返回了无效响应（${response.status}）`)
  }
  if (!response.ok) {
    const detail = (payload as { detail?: unknown })?.detail
    throw new Error(typeof detail === 'string' ? detail : `等权计算失败（${response.status}）`)
  }
  const result = payload as Partial<EqualWeightsResponse>
  if (
    !Array.isArray(result.weights)
    || result.weights.length !== assetCount
    || result.weights.some((weight) => typeof weight !== 'number' || !Number.isFinite(weight) || weight < 0)
  ) {
    throw new Error('等权计算服务返回的权重向量无效或资产数量不一致')
  }
  if (
    !result.execution
    || result.execution.backend !== 'numba_njit_fixed_signature'
    || result.execution.execution_backend !== 'numba_njit_fixed_signature'
    || result.execution.nopython !== true
    || result.execution.njit_required !== true
    || result.execution.object_mode !== 0
    || result.execution.python_fallback !== 0
    || result.execution.request_time_compilation !== 0
    || !result.execution.kernel_signatures
    || typeof result.execution.kernel_signatures !== 'object'
    || Object.keys(result.execution.kernel_signatures).length === 0
    || Object.values(result.execution.kernel_signatures).some((signatures) => !Array.isArray(signatures) || signatures.length === 0)
  ) {
    throw new Error('等权计算服务未通过固定签名 NJIT 执行校验')
  }
  return result as EqualWeightsResponse
}

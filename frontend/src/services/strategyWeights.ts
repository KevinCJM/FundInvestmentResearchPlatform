import { assertNativeNumericalExecution, type CppAotExecutionAudit } from '../utils/fixedNjitExecution'
interface NjitStrategyExecutionAudit {
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

export type StrategyExecutionAudit = NjitStrategyExecutionAudit | CppAotExecutionAudit

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
  try {
    assertNativeNumericalExecution(result.execution)
    if ((result.execution?.execution_backend ?? result.execution?.backend) !== 'cpp_aot'
        && (result.execution?.backend !== 'numba_njit_fixed_signature'
          || result.execution.execution_backend !== 'numba_njit_fixed_signature'
          || (result.execution as NjitStrategyExecutionAudit).njit_required !== true)) {
      throw new Error('Incomplete NJIT proof')
    }
  } catch {
    throw new Error((result.execution?.execution_backend ?? result.execution?.backend) === 'cpp_aot' ? '等权计算服务未通过 C++ AOT 执行校验' : '等权计算服务未通过固定签名 NJIT 执行校验')
  }
  return result as EqualWeightsResponse
}

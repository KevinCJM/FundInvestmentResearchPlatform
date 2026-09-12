import { assertFixedNjitExecution, type FixedNjitExecutionAudit } from '../utils/fixedNjitExecution'
import type { TaaBaseline, TaaCatalog } from './tacticalAllocation'

export interface MandateDefinition {
  name: string; as_of: string; review_date: string; currency: string; horizon_years: number
  target_return: number; max_volatility: number; min_liquid_weight: number; max_illiquid_weight: number
  max_tracking_error: number; risk_aversion: number
  rebalance_policy: 'monthly' | 'quarterly' | 'annually' | 'threshold'; rebalance_note: string; note: string
}
export interface MandateVersion {
  id: string; name: string; created_at: string; content_hash: string; definition: MandateDefinition
}
export type EconomicRole = 'growth' | 'rates' | 'inflation' | 'credit' | 'liquidity' | 'diversifier'
export interface AssetAssumption {
  id: string; role: EconomicRole; liquidity: 'liquid' | 'illiquid'; rationale: string
  annual_return: number; annual_volatility: number; mean_uncertainty: number
}
export interface RiskReferenceRequest {
  alloc_name: string; as_of: string; start_date: string; end_date: string
  shrinkage: number; periods_per_year: number
}
export interface CmaDefinition {
  name: string; alloc_name: string; as_of: string; currency: string; horizon_years: number
  return_basis: 'annual_arithmetic_total_return'; source: string; basis_confirmed: boolean
  assets: AssetAssumption[]; correlation: number[][]
  risk_origin: 'manual' | 'historical_reference'
  risk_reference: RiskReferenceRequest | null; risk_reference_hash: string | null
}
export type CmaDraft = Omit<CmaDefinition, 'assets'> & {
  assets: Array<Omit<AssetAssumption, 'role' | 'liquidity'> & { role: EconomicRole | ''; liquidity: 'liquid' | 'illiquid' | '' }>
}
export function completeCma(value: CmaDraft): value is CmaDefinition {
  return Boolean(value.name.trim() && value.source.trim().length >= 3 && value.as_of && value.basis_confirmed
    && value.assets.length && value.assets.every(asset => asset.role && asset.liquidity && asset.rationale.trim().length >= 3
      && [asset.annual_return, asset.annual_volatility, asset.mean_uncertainty].every(Number.isFinite)
      && asset.annual_volatility > 0 && asset.mean_uncertainty >= 0)
    && value.correlation.length === value.assets.length && value.correlation.every(row => row.length === value.assets.length && row.every(Number.isFinite)))
}
export interface RiskReference {
  preview_hash: string; request: RiskReferenceRequest; assets: string[]; volatility: number[]
  correlation: number[][]; historical_mean: number[]; observations: number; source_hash: string
  lineage: { start_date: string; end_date: string }; warnings: string[]; execution: FixedNjitExecutionAudit
}
export interface CmaPreview {
  preview_hash: string; definition: CmaDefinition; source_snapshot: Omit<TaaBaseline, 'id' | 'created_at' | 'content_hash'>
  covariance: number[][]; warnings: string[]; execution: FixedNjitExecutionAudit
}
export interface CmaVersion extends CmaPreview {
  id: string; name: string; created_at: string; content_hash: string
}
export interface PolicyRequest {
  mandate_id: string; cma_id: string
  constraints: Record<string, { min_weight: number; max_weight: number; max_abs_tilt: number }>
  group_limits: Array<{ id: string; assets: string[]; lo: number; hi: number }>
  uncertainty_penalty: number; candidate_count: number; seed: number
}
export interface PolicyCandidate {
  id: 'minimum-risk' | 'nominal-utility' | 'robust-utility' | 'maximum-return'
  name: string; weights: Record<string, number>; risk_contributions: Record<string, number | null>
  metrics: { expected_return: number; volatility: number; conservative_return: number; nominal_utility: number; robust_utility: number }
}
export interface PolicyPreview {
  preview_hash: string; request: PolicyRequest; mandate: MandateDefinition; assumptions: CmaDefinition
  source_snapshot: Omit<TaaBaseline, 'id' | 'created_at' | 'content_hash'>; candidates: PolicyCandidate[]; accepted_candidates: number
  warnings: string[]; execution: FixedNjitExecutionAudit
}
export interface StrategicCatalog {
  allocations: TaaCatalog['allocations']; mandates: MandateVersion[]
  assumptions: Array<{ id: string; name: string; alloc_name: string; as_of: string; currency: string; horizon_years: number }>
  policies: Array<{ id: string; name: string; as_of: string; alloc_name: string }>
}

/** Display-only percent conversion; the saved fractional assumption is unchanged. */
export const percentInputValue = (value: number): number => Number.isFinite(value) ? Number((value * 100).toPrecision(12)) : NaN

const root = '/api/strategic-allocation'
async function request<T>(path: string, body?: unknown, signal?: AbortSignal): Promise<T> {
  const response = await fetch(`${root}${path}`, {
    method: body === undefined ? 'GET' : 'POST', signal,
    headers: { 'Content-Type': 'application/json' }, ...(body === undefined ? {} : { body: JSON.stringify(body) }),
  })
  const result = await response.json().catch(() => { throw new Error('资产配置服务没有返回可读取的数据，请重试。') })
  if (!response.ok) {
    const detail = result?.detail
    throw new Error(typeof detail === 'string' ? detail : Array.isArray(detail)
      ? detail.map(item => `${item.loc?.slice(1).join('.') ?? ''}：${item.msg ?? '输入无效'}`).join('；')
      : detail?.message ?? '资产配置请求失败，请检查输入后重试。')
  }
  return result as T
}
const verified = <T extends { execution: FixedNjitExecutionAudit }>(value: T): T => {
  assertFixedNjitExecution(value.execution, '长期配置研究')
  return value
}
export const getStrategicCatalog = (signal?: AbortSignal) => request<StrategicCatalog>('/catalog', undefined, signal)
export const saveMandate = (body: MandateDefinition, signal?: AbortSignal) => request<MandateVersion>('/mandates', body, signal)
export const getMandate = (id: string, signal?: AbortSignal) => request<MandateVersion>(`/mandates/${encodeURIComponent(id)}`, undefined, signal)
export const riskReference = async (body: RiskReferenceRequest, signal?: AbortSignal) => verified(await request<RiskReference>('/risk-reference', body, signal))
export const previewCma = async (body: CmaDefinition, signal?: AbortSignal) => verified(await request<CmaPreview>('/cma/preview', body, signal))
export const publishCma = async (body: CmaDefinition, previewHash: string, signal?: AbortSignal) => verified(await request<CmaVersion>('/cma', { request: body, preview_hash: previewHash }, signal))
export const getCma = async (id: string, signal?: AbortSignal) => verified(await request<CmaVersion>(`/cma/${encodeURIComponent(id)}`, undefined, signal))
export const previewPolicy = async (body: PolicyRequest, signal?: AbortSignal) => verified(await request<PolicyPreview>('/policy/preview', body, signal))
export const publishPolicy = async (body: PolicyRequest, hash: string, candidate: PolicyCandidate['id'], name: string, reason: string, signal?: AbortSignal) => {
  const result = await request<TaaBaseline>('/policies', { request: body, preview_hash: hash, candidate_id: candidate, name, reason }, signal)
  if (!result.policy) throw new Error('返回的版本缺少政策与目标引用，已停止交接。')
  assertFixedNjitExecution(result.policy.execution, '政策采纳')
  return result
}

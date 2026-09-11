import { assertFixedNjitExecution, type FixedNjitExecutionAudit } from '../utils/fixedNjitExecution'

export interface TaaAsset {
  id: string
  name: string
  base_weight: number
  min_weight: number
  max_weight: number
  max_abs_tilt: number
  products: Array<{ kind: 'etf' | 'fund'; product_id: string; name: string; weight: number }>
}

export interface TaaBaseline {
  id: string
  name: string
  created_at: string
  content_hash: string
  alloc_name: string
  as_of: string
  universe_snapshot_id?: string | null
  data_release_id?: string | null
  assets: TaaAsset[]
  group_limits?: Array<{ id: string; assets: string[]; lo: number; hi: number }>
  apply_eligible?: boolean
  apply_reasons?: string[]
  pit: { status: string; reasons: string[] }
  lineage: Record<string, unknown>
}

export interface TaaBaselineInput {
  alloc_name: string
  name: string
  as_of: string
  weights: Record<string, number>
  constraints?: Record<string, { min_weight: number; max_weight: number; max_abs_tilt: number }>
  group_limits?: Array<{ id: string; assets: string[]; lo: number; hi: number }>
}

export interface TaaCatalog {
  allocations: Array<{ alloc_name: string; assets: Array<{ id: string; name: string }>; as_of?: string; universe_snapshot_id?: string; data_release_id?: string; coverage?: { start_date: string; end_date: string } | null }>
  baselines: TaaBaseline[]
  decisions: TaaDecision[]
}

export interface TaaPreviewRequest {
  baseline_id: string
  start_date: string
  end_date: string
  as_of: string
  train_end_date: string
  signal_mode: 'momentum' | 'manual' | 'regime'
  lookback: number
  manual_tilts: Record<string, number>
  regime_run_id?: string
  state_tilts?: Record<string, Record<string, number>>
  max_abs_tilt: number
  transaction_cost_bps: number
  risk_penalty: number
  max_tracking_error: number
  max_turnover: number
  confidence_floor: number
  max_signal_age_days: number
  search: boolean
  objective: 'active_utility' | 'excess_return' | 'min_drawdown'
  current_weights?: Record<string, number>
  review_days: number
  note: string
  selected_candidate_id?: string
}

export interface TaaMetrics {
  total_return: number
  baseline_return: number
  excess_return: number
  annual_volatility: number | null
  max_drawdown: number
  tracking_error: number | null
  turnover: number
  cost: number
  score: number | null
}

export interface TaaCandidate {
  id: string
  strength: number
  feasible: boolean
  validation_feasible?: boolean
  train: TaaMetrics
  validation: TaaMetrics
}

export interface TaaPreview {
  preview_hash: string
  request: TaaPreviewRequest
  baseline: TaaBaseline
  data: {
    start_date: string
    end_date: string
    observations: number
    train_observations: number
    validation_observations: number
    training?: TaaPreflight['training']
    lineage: Record<string, unknown>
    pit: { status: string; reasons: string[] }
  }
  candidates: TaaCandidate[]
  selected_id: string
  recommendation: {
    weights: Record<string, number>
    tilts: Record<string, number>
    trade_deltas: Record<string, number> | null
    reason: string
    signal_date: string | null
    expires_on: string
    confidence: number | null
    fallback_reason: string | null
    is_saa: boolean
    turnover_from_current?: number | null
    signal_details?: Array<{ asset_id: string; value: number | null; signal_date: string | null; window?: { window_start: string | null; window_end: string | null; available_at: string | null; lag_days: number | null; status: string }; direction: string; raw_tilt: number; applied_tilt: number; constraint_reason: string | null }>
  }
  chart: Array<{ date: string; baseline: number; taa: number; segment: 'train' | 'validation' }>
  weight_path: Array<{ date: string; weights: Record<string, number>; turnover: number }>
  warnings: string[]
  execution: FixedNjitExecutionAudit
  audit: Record<string, unknown>
}

export interface TaaScenarioRequest {
  preview_request: TaaPreviewRequest
  candidate_id?: string
  scenario: { kind: 'shock' | 'historical'; name: string; shocks?: Record<string, number>; start_date?: string; end_date?: string }
}

export type TaaScenario = TaaScenarioRequest['scenario']
export interface TaaScenarioExperiment { scenario: TaaScenario; result?: TaaScenarioResult }
export interface TaaPreflight {
  coverage: { start_date: string; end_date: string }
  dates: Pick<TaaPreviewRequest, 'start_date' | 'end_date' | 'as_of' | 'train_end_date'>
  quality: { status: 'clear' | 'blocked'; issues: Array<{ asset_id: string; date: string; value: number; code: string; message: string }> }
  training: { train_signal_observations?: number; validation_signal_observations?: number; train_known_observations?: number; validation_known_observations?: number; eligible: boolean; train_observations: number; validation_observations: number; unavailable_count: number; unknown_count: number; earliest_available_date: string | null; reasons: string[] }
  guidance: Array<{ code: string; message: string; action: 'adjust_dates' | 'fixed_comparison' | 'review_data'; patch?: Partial<TaaPreviewRequest> }>
  pit: { status: string; reasons: string[] }
  can_calculate: boolean
}

export interface TaaScenarioResult {
  name: string
  kind: 'shock' | 'historical'
  baseline_return: number
  taa_return: number
  excess_return: number
  contributions: Array<{ asset_id: string; baseline: number; taa: number; excess: number }>
  cost?: { baseline: number; taa: number }
  warnings: string[]
  execution: FixedNjitExecutionAudit
}

export interface TaaDecision {
  id: string
  name: string
  created_at: string
  preview: TaaPreview
  note?: string
  content_hash?: string
  scenarios?: Array<{ scenario: TaaScenario; result: TaaScenarioResult }>
}

export interface TaaProductAllocation {
  name: string
  method: 'manual'
  universe_snapshot_id?: string
  allocation_source: { kind: 'taa'; decision_id: string; baseline_id: string; class_weights: Record<string, number> }
  constituents: Array<{ kind: 'etf' | 'fund'; product_id: string; name: string; asset_class_id: string; asset_class_name: string; weight: number }>
}

const root = '/api/tactical-allocation'

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(`${root}${path}`, { ...init, headers: { 'Content-Type': 'application/json', ...init.headers } })
  let payload: any
  try { payload = await response.json() } catch { throw new Error('服务没有返回可读取的资产配置数据，请检查后端连接后重试。') }
  if (!response.ok) {
    const detail = payload?.detail
    const message = typeof detail === 'string' ? detail : Array.isArray(detail)
      ? detail.map((item: { msg?: string }) => item.msg ?? '输入无效').join('；')
      : detail?.message ?? payload?.message ?? `资产配置操作未完成（${response.status}）。`
    throw new Error(message)
  }
  return payload as T
}

const post = <T,>(path: string, body: unknown) => request<T>(path, { method: 'POST', body: JSON.stringify(body) })
const audited = <T extends { execution: FixedNjitExecutionAudit }>(result: T, name: string): T => {
  if (!result || typeof result !== 'object') throw new Error(`${name}返回的结果不完整，请重新读取。`)
  assertFixedNjitExecution(result.execution, name)
  return result
}

export const getTaaCatalog = async (signal?: AbortSignal) => {
  const value = await request<TaaCatalog>('/catalog', { signal })
  if (!value || !Array.isArray(value.baselines) || !Array.isArray(value.allocations) || !Array.isArray(value.decisions)
    || value.allocations.some(item => !item || typeof item.alloc_name !== 'string' || !Array.isArray(item.assets))) {
    throw new Error('资产配置目录格式不完整，请刷新或检查后端版本。')
  }
  return value
}
export const getTaaBaseline = async (id: string, signal?: AbortSignal) => {
  const value = await request<TaaBaseline>(`/baselines/${encodeURIComponent(id)}`, { signal })
  if (!value || typeof value.id !== 'string' || typeof value.as_of !== 'string' || !Array.isArray(value.assets)
    || !Array.isArray(value.pit?.reasons) || value.assets.some(asset => !asset || typeof asset.id !== 'string' || typeof asset.base_weight !== 'number')) {
    throw new Error('SAA 基准信息不完整，请重新选择有效的已保存版本。')
  }
  return value
}
export const createTaaBaseline = (input: TaaBaselineInput) => post<TaaBaseline>('/baselines', input)
export const previewTaa = async (input: TaaPreviewRequest) => audited(await post<TaaPreview>('/preview', input), '战术配置研究')
export const preflightTaa = async (input: TaaPreviewRequest, signal?: AbortSignal) => {
  const value = await request<TaaPreflight>('/preflight', { method: 'POST', body: JSON.stringify(input), signal })
  if (!value || typeof value.can_calculate !== 'boolean' || !['clear', 'blocked'].includes(value.quality?.status)
    || !Array.isArray(value.quality?.issues) || !Array.isArray(value.guidance) || !Array.isArray(value.pit?.reasons)
    || !Array.isArray(value.training?.reasons) || typeof value.training?.eligible !== 'boolean'
    || typeof value.coverage?.start_date !== 'string' || typeof value.coverage?.end_date !== 'string' || !value.dates) {
    throw new Error('研究条件检查结果不完整，请确认后端版本后重试。')
  }
  return value
}
export const simulateTaaScenario = async (input: TaaScenarioRequest) => audited(await post<TaaScenarioResult>('/scenarios', input), '战术配置情景模拟')
export const saveTaaDecision = async (input: { request: TaaPreviewRequest; preview_hash: string; name: string; note: string; scenarios?: TaaScenario[] }) => checkedDecision(await post<TaaDecision>('/decisions', input))
function checkedDecision(result: TaaDecision): TaaDecision {
  audited(result.preview, '已保存战术配置研究')
  if (result.scenarios && !Array.isArray(result.scenarios)) throw new Error('已保存情景格式不完整，请重新读取。')
  result.scenarios?.forEach(item => audited(item.result, '已保存情景实验'))
  return result
}
export const getTaaDecision = async (id: string, signal?: AbortSignal) => {
  const result = await request<TaaDecision>(`/decisions/${encodeURIComponent(id)}`, { signal })
  return checkedDecision(result)
}
export const taaProductAllocation = (id: string) => post<TaaProductAllocation>(`/decisions/${encodeURIComponent(id)}/product-allocation`, {})

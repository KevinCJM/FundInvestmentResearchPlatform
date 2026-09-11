import { assertFixedNjitExecution, type FixedNjitExecutionAudit } from '../utils/fixedNjitExecution'

export type ResearchDomain = 'product' | 'transmission'
export type Frequency = 'daily' | 'weekly' | 'monthly' | 'quarterly'
export type VariableUnit = 'return' | 'bp' | 'pp' | 'points'
export type ModelStage = 'product' | 'event_macro' | 'macro_market'
export interface RiskVariable {
  id: string; name: string; roles: Array<'driver' | 'macro' | 'market'>; unit: VariableUnit
  unit_label: string; frequency: Frequency; transform: string; reference_move: string
  contract_hash: string; basis: string; market: string; currency: string; revision: number
  availability?: { available: boolean; reason: string }
}
export interface RiskProduct { kind: 'etf' | 'fund'; product_id: string; name: string }
export interface RiskModelFields {
  name: string; stage: ModelStage; method: 'ols'; inputs: string[]; outputs: string[]
  targets: RiskProduct[]; frequency: Frequency; start_date: string; end_date: string
  validation_start: string; as_of: string; lags: number; min_train: number; min_validation: number
  minimum_validation_r2: number; refit_after_validation: boolean
}
export interface RiskRow {
  target_id: string; name: string; unit: VariableUnit; train_observations: number; validation_observations: number
  training_r2: number | null; validation_r2: number | null; status: number; status_label: string; data_as_of?: string
  betas: Array<number | null>; reference_responses: Array<number | null>
}
export interface ResearchTarget { kind: string; product_id: string; name: string; key: string }
export interface RiskRun {
  id: string; name: string; stage: ModelStage; method: 'ols' | 'cashflow'; model: RiskModelFields | CashflowStudy
  model_id?: string; model_revision?: number; as_of: string; frequency: Frequency | 'single_shock'
  inputs: RiskVariable[]; outputs: RiskVariable[]; targets: ResearchTarget[]; target_keys: string[]
  data_as_of?: string
  rows: RiskRow[]; coefficients?: Array<Array<number | null>>; publishable: boolean; blockers: string[]
  metrics?: { price: number; modified_duration: number; convexity: number }
  execution: FixedNjitExecutionAudit; limitations: string[]; created_at: string; content_hash: string
  preview_hash?: string; transient?: boolean; deployment_fit?: string
}
export interface RiskRelease {
  id: string; name: string; stage: ModelStage; run_id: string; run_hash: string; method: 'ols' | 'cashflow'
  model_id?: string; model_revision?: number; as_of: string; frequency: Frequency | 'single_shock'
  inputs: RiskVariable[]; outputs: RiskVariable[]; targets: ResearchTarget[]; target_keys: string[]
  effective_at: string; expires_at: string; created_at: string; status: string; usage: 'research_only'; note: string; data_as_of?: string
}
export interface RiskCatalog {
  domain: ResearchDomain; variables: RiskVariable[]
  frequencies: Array<{ id: Frequency; name: string }>
  event_templates: Array<{ id: string; name: string; description: string }>
  storage: { logical_path: string; managed_data_disk: boolean }
  capabilities?: { preview_persistence?: boolean; persist_only_on_publish?: boolean; auto_training_on_read?: boolean }
}
export interface CashflowStudy {
  name: string; product_id: string; as_of: string; yield_factor_id: string; yield_percent: number
  compounding: 1 | 2 | 4 | 12; cashflows: Array<{ years: number; amount: number }>; source_label: string
}
export interface SeriesImport {
  name: string; roles: RiskVariable['roles']; unit: VariableUnit; frequency: Frequency
  transform: 'price_return' | 'percent_rate_change' | 'difference' | 'identity'
  source_label: string; csv_text: string
  category: 'activity' | 'inflation' | 'policy' | 'energy' | 'credit' | 'equity' | 'currency' | 'other'
}
export interface ScenarioDraft {
  name: string; description: string; entry: 'event' | 'macro' | 'market'; frequency: Frequency
  event_template: 'energy' | 'policy' | 'credit' | 'custom'
  event_model_release_id: string | null; macro_model_release_id: string | null
  input_ids: string[]; rows: number[][]; shock_basis: 'period_change'
}
export interface TransmissionStep {
  release_id: string; name: string; stage: ModelStage; inputs: RiskVariable[]; outputs: RiskVariable[]; path: number[][]
}
export interface ScenarioPreview {
  id: string; name: string; entry: ScenarioDraft['entry']; definition: ScenarioDraft; definition_hash: string
  frequency: Frequency; input_variables: RiskVariable[]; factors: RiskVariable[]; path: number[][]
  horizon: number; lineage: TransmissionStep[]; limitations: string[]; execution: FixedNjitExecutionAudit
  preview_hash?: string; transient?: boolean
}
export interface ScenarioRelease {
  id: string; name: string; entry: ScenarioDraft['entry']; preview_id: string; frequency: Frequency
  factors: RiskVariable[]; horizon: number; lineage: TransmissionStep[]; status: string; reason?: string
  effective_at: string; expires_at: string; created_at: string; note: string
}
export interface ImpactRequest {
  scenario_release_id: string; exposure_release_id: string
  target: { kind: 'product'; product_key: string } | { kind: 'portfolio_run'; portfolio_run_id: string }
  as_of: string; holding_policy: 'buy_and_hold' | 'constant_weights_zero_cost'
  hold_other_factors_constant: true; notional: number; usage: 'research'
}
export interface RiskImpact {
  id: string; name: string; request: ImpactRequest; as_of: string; created_at: string; content_hash: string
  target: { kind: string; name: string; holdings_date?: string; portfolio_run_id?: string }
  summary: { terminal_return: number; terminal_nav: number; max_drawdown: number; pnl_amount: number; asset_reconciliation_error: number; factor_reconciliation_error: number }
  path: Array<{ step: number; return: number; nav: number; drawdown: number }>
  by_asset: Array<ResearchTarget & { weight: number; contribution: number }>
  by_factor: Array<{ id: string; name: string; contribution: number }>
  lineage: TransmissionStep[]; limitations: string[]; execution: FixedNjitExecutionAudit
  assumed_unchanged_factors: string[]; frequency: Frequency; transient?: boolean
}
export interface PortfolioChoice { id: string; name: string; as_of: string; created_at: string }

const root = (domain: ResearchDomain) => domain === 'product' ? '/api/risk-models' : '/api/scenario-transmission'

export async function riskRequest<T>(url: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(url, { ...options, headers: { 'Content-Type': 'application/json', ...options.headers } })
  const contentType = response.headers.get('content-type') ?? ''
  if (!contentType.includes('application/json')) throw new Error('服务没有返回可读取的研究数据，请检查后端连接。')
  const payload = await response.json()
  if (!response.ok) {
    const detail = payload?.detail
    const message = Array.isArray(detail)
      ? detail.map((item: { msg?: string; loc?: string[] }) => item.msg ?? '输入格式无效').join('；')
      : detail?.message ?? payload?.message ?? '研究操作未完成，请检查输入与数据。'
    throw new Error(message)
  }
  return payload as T
}
const post = <T,>(url: string, body: unknown, signal?: AbortSignal) => riskRequest<T>(url, { method: 'POST', body: JSON.stringify(body), signal })
const items = async <T,>(url: string, signal?: AbortSignal) => (await riskRequest<{ items: T[] }>(url, { signal })).items
const audit = <T extends { execution: FixedNjitExecutionAudit }>(value: T, name: string) => {
  assertFixedNjitExecution(value.execution, name)
  return value
}
export const riskCatalog = (domain: ResearchDomain, signal?: AbortSignal) => riskRequest<RiskCatalog>(`${root(domain)}/catalog`, { signal })
export const getRiskRun = async (domain: ResearchDomain, id: string, signal?: AbortSignal) => audit(await riskRequest<RiskRun>(`${root(domain)}/runs/${encodeURIComponent(id)}`, { signal }), '已发布敏感性研究')
export const searchRiskProducts = (kind: 'etf' | 'fund', query: string, signal?: AbortSignal) => items<{ ts_code: string; name: string }>(`${root('product')}/products?${new URLSearchParams({ kind, query })}`, signal)
export const previewRiskModel = async (domain: ResearchDomain, fields: RiskModelFields) => audit(await post<RiskRun>(`${root(domain)}/previews`, fields), '敏感性研究预览')
export const previewCashflow = async (fields: CashflowStudy) => audit(await post<RiskRun>('/api/risk-models/cashflow-previews', fields), '现金流估值预览')
export const riskReleases = (domain: ResearchDomain, options: { as_of?: string; product_key?: string } = {}, signal?: AbortSignal) => {
  const query = new URLSearchParams(Object.entries(options).filter((entry): entry is [string, string] => Boolean(entry[1])))
  return items<RiskRelease>(`${root(domain)}/releases?${query}`, signal)
}
export const publishRiskPreview = (domain: ResearchDomain, definition: RiskModelFields, preview_hash: string, valid_days: number, note: string) => post<RiskRelease>(`${root(domain)}/releases`, { definition, preview_hash, valid_days, note, acknowledge_limitations: true })
export const publishCashflowPreview = (study: CashflowStudy, preview_hash: string, valid_days: number, note: string) => post<RiskRelease>('/api/risk-models/cashflow-releases', { study, preview_hash, valid_days, note, acknowledge_limitations: true })
export const retireRiskRelease = (domain: ResearchDomain, id: string, note: string) => post(`${root(domain)}/releases/${encodeURIComponent(id)}/retire`, { note })
export const importRiskSeries = (domain: ResearchDomain, fields: SeriesImport) => post<RiskVariable>(`${root(domain)}/variables`, fields)
export const scenarioReleases = (as_of?: string, signal?: AbortSignal) => items<ScenarioRelease>(`/api/published-scenarios/releases${as_of ? `?${new URLSearchParams({ as_of })}` : ''}`, signal)
export const previewScenario = async (fields: ScenarioDraft) => audit(await post<ScenarioPreview>('/api/published-scenarios/previews', fields), '情景传导预览')
export const getScenarioPreview = async (id: string, signal?: AbortSignal) => audit(await riskRequest<ScenarioPreview>(`/api/published-scenarios/previews/${encodeURIComponent(id)}`, { signal }), '已保存情景路径')
export const publishScenario = (definition: ScenarioDraft, preview_hash: string, valid_days: number, note: string) => post<ScenarioRelease>('/api/published-scenarios/releases', { definition, preview_hash, valid_days, note, acknowledge_limitations: true })
export const retireScenario = (id: string, note: string) => post(`/api/published-scenarios/releases/${encodeURIComponent(id)}/retire`, { note })
export const riskPortfolios = (signal?: AbortSignal) => items<PortfolioChoice>('/api/published-scenarios/portfolios', signal)
export const runRiskImpact = async (fields: ImpactRequest) => audit(await post<RiskImpact>('/api/published-scenarios/impacts', fields), '已发布模型情景压测')
export const getRiskImpact = async (id: string, signal?: AbortSignal) => audit(await riskRequest<RiskImpact>(`/api/published-scenarios/impacts/${encodeURIComponent(id)}`, { signal }), '已保存情景压测')

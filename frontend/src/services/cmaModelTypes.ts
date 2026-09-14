/** Opt-in model payloads. Rates/weights use decimals; covariance is decimal return squared. */
export interface CmaModelContext {
  asset_ids: string[]
  as_of: string
  currency: string
  return_basis?: 'annual_arithmetic_total_return'
  source: string
}

export interface BlackLittermanView {
  kind: 'absolute' | 'relative'
  asset_id: string
  relative_to?: string | null
  annual_return: number
  view_std: number
  observed_on: string
  available_on: string
  source: string
}

export interface BlackLittermanRequest extends CmaModelContext {
  method: 'black_litterman'
  covariance: number[][]
  risk_covariance_basis: 'input_covariance'
  market_weights: Record<string, number>
  market_weight_source: string
  delta: number
  tau: number
  risk_free_rate: number
  views: BlackLittermanView[]
}

export interface CmaScenario {
  id: string
  probability: number
  annual_returns: Record<string, number>
  covariance?: number[][] | null
  source: string
}

export interface ScenarioMixtureRequest extends CmaModelContext {
  method: 'scenario_mixture'
  risk_mode: 'shared' | 'scenario_specific'
  shared_covariance?: number[][] | null
  scenarios: CmaScenario[]
}

export type CmaModelRequest = BlackLittermanRequest | ScenarioMixtureRequest

/** NaN represents unfilled draft numbers locally and must never be submitted. */
export function cmaModelInputError(value: CmaModelRequest): string | null {
  const axis = value.asset_ids
  const between = (n: number, low: number, high: number) => Number.isFinite(n) && n >= low && n <= high
  const source = (text: string) => text.trim().length >= 3 && text.trim().length <= 2000
  const date = (text: string) => /^\d{4}-\d{2}-\d{2}$/.test(text) && Number.isFinite(Date.parse(text)) && new Date(`${text}T00:00:00Z`).toISOString().slice(0, 10) === text
  const complete = (items: Record<string, number>) => Object.keys(items).length === axis.length && axis.every(a => Object.prototype.hasOwnProperty.call(items, a))
  const matrix = (m: number[][] | null | undefined) => !!m && m.length === axis.length && m.every((row, i) => row.length === axis.length && row.every((v, j) => Number.isFinite(v) && (i !== j || v > 0 && v <= 9) && Math.abs(v - m[j]?.[i]) <= 1e-10))
  if (!axis.length || axis.length > 30 || new Set(axis).size !== axis.length || axis.some(a => !a.trim() || a.length > 120)) return '请先载入完整、不重复的资产范围。'
  if (!date(value.as_of) || !/^[A-Z]{3}$/.test(value.currency)) return '请确认研究日期与三位计价币种。'
  if (!source(value.source)) return '请填写至少3字的模型与风险依据。'
  if (value.method === 'black_litterman') {
    if (!complete(value.market_weights) || axis.some(a => !between(value.market_weights[a], 0, 1)) || Math.abs(Object.values(value.market_weights).reduce((a, b) => a + b, 0) - 1) > 1e-8) return '市场权重须完整、非负并合计100%；请填写明确的市场组合。'
    if (!source(value.market_weight_source)) return '请填写市场权重来源，不能把默认等权当作市场组合。'
    if (!matrix(value.covariance)) return '请填写完整对称且对角为正的风险协方差；服务端将检查半正定性。'
    if (!Number.isFinite(value.delta) || value.delta <= 0 || !Number.isFinite(value.tau) || value.tau <= 0 || !between(value.risk_free_rate, -0.5, 2)) return '请检查风险厌恶系数、τ（均须为有限正数）与无风险收益。'
    if (value.views.length > 60) return '观点最多60条。'
    for (const view of value.views) {
      if (!axis.includes(view.asset_id) || (view.kind === 'relative' && (!view.relative_to || !axis.includes(view.relative_to) || view.relative_to === view.asset_id)) || (view.kind === 'absolute' && view.relative_to != null)) return '每条观点须指定有效资产；相对观点须选择另一资产。'
      if (!between(view.annual_return, view.kind === 'absolute' ? -0.5 : -2.5, view.kind === 'absolute' ? 2 : 2.5) || !Number.isFinite(view.view_std) || view.view_std <= 0 || !Number.isFinite(view.view_std ** 2) || view.view_std ** 2 === 0) return '请填写有效观点收益与严格为正、可计算的观点标准差。'
      if (!source(view.source) || !date(view.observed_on) || !date(view.available_on) || view.observed_on > view.available_on || view.available_on > value.as_of) return '观点须有依据，且满足观察日 ≤ 可得日 ≤ 研究日。'
    }
  } else {
    if (!value.scenarios.length || value.scenarios.length > 60) return '请添加1至60个明确情景，再填写概率和假设。'
    if (new Set(value.scenarios.map(s => s.id.trim())).size !== value.scenarios.length) return '情景名称不能重复。'
    if (value.scenarios.some(s => !between(s.probability, 0, 1)) || Math.abs(value.scenarios.reduce((sum, s) => sum + s.probability, 0) - 1) > 1e-8) return '情景概率须明确、非负并合计100%；不会自动归一化。'
    if (value.risk_mode === 'shared' ? !matrix(value.shared_covariance) || value.scenarios.some(s => s.covariance != null) : value.shared_covariance != null || value.scenarios.some(s => !matrix(s.covariance))) return '请选择共用或逐情景风险，并完整填写对应协方差。'
    for (const scenario of value.scenarios) {
      if (!scenario.id.trim() || scenario.id.length > 120 || !source(scenario.source)) return '每个情景须有名称与至少3字的依据。'
      if (!complete(scenario.annual_returns) || axis.some(a => !between(scenario.annual_returns[a], -0.5, 2))) return '每个情景须覆盖所有资产，年化收益范围为 -50% 至 200%。'
    }
  }
  return null
}

import type { BlackLittermanRequest as WireBlackLittermanRequest, ScenarioMixtureRequest, HistoricalCmaRequest, BayesianCmaRequest, RegimeCmaRequest } from './ltcmaContract.generated'
export type { CmaModelContext, BlackLittermanView, CmaScenario, ScenarioMixtureRequest, CmaWindow, HistoricalCmaRequest, BayesianCmaRequest, RegimeCmaRequest } from './ltcmaContract.generated'
// The editor materializes this server default rather than repeatedly treating it as missing.
export type BlackLittermanRequest = Omit<WireBlackLittermanRequest, 'views'> & { views: NonNullable<WireBlackLittermanRequest['views']> }

export type CoreCmaModelRequest = BlackLittermanRequest | ScenarioMixtureRequest
export type StatisticalCmaRequest = HistoricalCmaRequest | BayesianCmaRequest | RegimeCmaRequest
export type CmaModelRequest = CoreCmaModelRequest | StatisticalCmaRequest
export const isStatisticalCma = (value: CmaModelRequest | null | undefined): value is StatisticalCmaRequest => Boolean(value && ['historical_statistics', 'bayesian_niw', 'historical_regime_occupancy'].includes(value.method))
export const isCoreCma = (value: CmaModelRequest): value is CoreCmaModelRequest => value.method === 'black_litterman' || value.method === 'scenario_mixture'

/** Draft NaN means unfinished input; it is never accepted as a model estimate. */
export function cmaModelInputError(value: CmaModelRequest): string | null {
  const axis = value.asset_ids
  const between = (n: number, low: number, high: number) => Number.isFinite(n) && n >= low && n <= high
  const source = (text: string) => typeof text === 'string' && text.trim().length >= 3 && text.trim().length <= 2000
  const date = (text: string) => /^\d{4}-\d{2}-\d{2}$/.test(text) && Number.isFinite(Date.parse(text)) && new Date(`${text}T00:00:00Z`).toISOString().slice(0, 10) === text
  const complete = (items: Record<string, number>) => Object.keys(items).length === axis.length && axis.every(a => Object.prototype.hasOwnProperty.call(items, a))
  const matrix = (m: number[][] | null | undefined) => !!m && m.length === axis.length && m.every((row, i) => row.length === axis.length && row.every((v, j) => Number.isFinite(v) && (i !== j || v > 0 && v <= 9) && Math.abs(v - m[j]?.[i]) <= 1e-10))
  if (!axis.length || axis.length > 30 || new Set(axis).size !== axis.length || axis.some(a => !a.trim() || a.length > 120)) return '请先载入完整、不重复的资产范围。'
  if (!date(value.as_of) || !/^[A-Z]{3}$/.test(value.currency)) return '请确认研究日期与三位计价币种。'
  if (!source(value.source)) return '请填写至少3字的模型与风险依据。'
  if (isStatisticalCma(value)) {
    const window = value.window ?? { kind: '5Y' }
    if (value.currency !== 'CNY') return '统计证据目前支持 CNY/SSE 日频；不会自动转换币种。'
    if (window.kind === 'custom' && (!window.start_date || !window.end_date || !date(window.start_date) || !date(window.end_date) || window.start_date >= window.end_date || window.end_date > value.as_of)) return '请填写研究日之前的有效历史窗口。'
    if (value.proxy_inputs) {
      const proxy = value.proxy_inputs
      if (proxy.assets.map(a => a.id).join('\u0000') !== axis.join('\u0000')) return '研究代理与战略资产轴不一致。'
      for (const asset of proxy.assets) {
        if (asset.asset_type === 'cash') {
          if (asset.cash_return == null || !between(asset.cash_return, -.5, 1)) return '请填写明确的现金收益假设。'
        } else if (!asset.components.length || asset.components.some(c => !c.series_id || !Number.isFinite(c.weight) || c.weight < 0)
          || Math.abs(asset.components.reduce((sum, c) => sum + c.weight, 0) - 1) > 1e-8) return '请为每个大类选择研究代理，且类内权重合计100%。'
      }
    }
    if (value.method !== 'bayesian_niw' && !between(value.shrinkage ?? 0, 0, 1)) return '对角收缩强度须在0%至100%之间。'
    const ref = (item: { id: string; content_hash: string } | undefined) => Boolean(item?.id && /^[0-9a-f]{64}$/.test(item.content_hash))
    if (value.method === 'bayesian_niw') {
      if (!ref(value.prior_ref)) return '请选择已确认且口径一致的先验 LTCMA。'
      if (value.prior_mode !== 'continue' && (!Number.isFinite(value.mean_prior_observations) || !Number.isFinite(value.covariance_prior_observations)
        || (value.mean_prior_observations ?? 0) <= 0 || (value.covariance_prior_observations ?? 0) <= 0)) return '请明确均值和风险先验的日频等效观察数。'
    }
    if (value.method === 'historical_regime_occupancy') {
      if (!ref(value.run_ref)) return '请选择已保存的日频事后状态运行。'
      if (value.probabilities && (Object.values(value.probabilities).some(p => !between(p, 0, 1))
        || Math.abs(Object.values(value.probabilities).reduce((sum, p) => sum + p, 0) - 1) > 1e-8
        || (value.probability_reason?.trim().length ?? 0) < 5)) return '应用概率须合计100%，并填写覆盖历史占用率的原因。'
    }
    return null
  }
  if (value.method === 'black_litterman') {
    if (!complete(value.market_weights) || axis.some(a => !between(value.market_weights[a], 0, 1)) || Math.abs(Object.values(value.market_weights).reduce((a, b) => a + b, 0) - 1) > 1e-8) return '市场权重须完整、非负并合计100%；请填写明确的市场组合。'
    if (!source(value.market_weight_source)) return '请填写市场权重来源，不能把默认等权当作市场组合。'
    if (!matrix(value.covariance)) return '请填写完整对称且对角为正的风险协方差；服务端将检查半正定性。'
    if (!Number.isFinite(value.delta) || value.delta <= 0 || !Number.isFinite(value.tau) || value.tau <= 0 || !between(value.risk_free_rate, -.5, 2)) return '请检查风险厌恶系数、τ（均须为有限正数）与无风险收益。'
    if ((value.views?.length ?? 0) > 60) return '观点最多60条。'
    for (const view of value.views ?? []) {
      if (view.kind === 'basket') {
        if (!view.legs.length || view.legs.length > 30 || new Set(view.legs.map(v => v.asset_id)).size !== view.legs.length
          || view.legs.some(v => !axis.includes(v.asset_id) || !Number.isFinite(v.coefficient))) return '篮子观点须使用不重复、有效的资产和有限系数。'
        const gross = view.legs.reduce((sum, v) => sum + Math.abs(v.coefficient), 0)
        const total = view.legs.reduce((sum, v) => sum + v.coefficient, 0)
        if (gross < 1e-12 || gross > 4 || Math.abs(total - (view.basis === 'absolute' ? 1 : 0)) > 1e-10) return '篮子绝对观点系数合计为1，相对观点为0；不能全零，绝对值合计不超过4。'
      } else if (!axis.includes(view.asset_id) || (view.kind === 'relative' && (!view.relative_to || !axis.includes(view.relative_to) || view.relative_to === view.asset_id)) || (view.kind === 'absolute' && view.relative_to != null)) return '每条观点须指定有效资产；相对观点须选择另一资产。'
      const basis = view.kind === 'basket' ? view.basis : view.kind
      if (!between(view.annual_return, basis === 'absolute' ? -.5 : -2.5, basis === 'absolute' ? 2 : 2.5) || !Number.isFinite(view.view_std) || view.view_std <= 0 || !Number.isFinite(view.view_std ** 2) || view.view_std ** 2 === 0) return '请填写有效观点收益与严格为正、可计算的观点标准差。'
      if (!source(view.source) || !date(view.observed_on) || !date(view.available_on) || view.observed_on > view.available_on || view.available_on > value.as_of) return '观点须有依据，且满足观察日 ≤ 可得日 ≤ 研究日。'
    }
  } else {
    if (!value.scenarios.length || value.scenarios.length > 60) return '请添加1至60个明确情景，再填写概率和假设。'
    if (new Set(value.scenarios.map(s => s.id.trim())).size !== value.scenarios.length) return '情景名称不能重复。'
    if (value.scenarios.some(s => !between(s.probability, 0, 1)) || Math.abs(value.scenarios.reduce((sum, s) => sum + s.probability, 0) - 1) > 1e-8) return '情景概率须明确、非负并合计100%；不会自动归一化。'
    if (value.risk_mode === 'shared' ? !matrix(value.shared_covariance) || value.scenarios.some(s => s.covariance != null) : value.shared_covariance != null || value.scenarios.some(s => !matrix(s.covariance))) return '请选择共用或逐情景风险，并完整填写对应协方差。'
    for (const scenario of value.scenarios) {
      if (!scenario.id.trim() || scenario.id.length > 120 || !source(scenario.source)) return '每个情景须有名称与至少3字的依据。'
      if (!complete(scenario.annual_returns) || axis.some(a => !between(scenario.annual_returns[a], -.5, 2))) return '每个情景须覆盖所有资产，年化收益范围为 -50% 至 200%。'
    }
  }
  return null
}

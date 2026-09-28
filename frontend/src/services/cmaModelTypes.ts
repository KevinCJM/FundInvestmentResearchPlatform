import { systemText } from '../i18n/runtime'
import type { BlackLittermanRequest as WireBlackLittermanRequest, ScenarioMixtureRequest, HistoricalCmaRequest, BayesianCmaRequest, RegimeCmaRequest, LongTermScenarioCmaRequest, ConditionalScenarioCmaRequest } from './ltcmaContract.generated'
import { niwPriorObservationBounds } from './ltcmaContract.generated'
export type { CmaModelContext, BlackLittermanView, CmaScenario, ScenarioMixtureRequest, CmaWindow, HistoricalCmaRequest, BayesianCmaRequest, RegimeCmaRequest } from './ltcmaContract.generated'
// The editor materializes this server default rather than repeatedly treating it as missing.
export type BlackLittermanRequest = Omit<WireBlackLittermanRequest, 'views'> & { views: NonNullable<WireBlackLittermanRequest['views']> }

export type CoreCmaModelRequest = BlackLittermanRequest | ScenarioMixtureRequest
export type ScenarioCmaRequest = LongTermScenarioCmaRequest | ConditionalScenarioCmaRequest
export type StatisticalCmaRequest = HistoricalCmaRequest | BayesianCmaRequest | RegimeCmaRequest | ScenarioCmaRequest
export type CmaModelRequest = CoreCmaModelRequest | StatisticalCmaRequest
export const isScenarioCma = (value: CmaModelRequest | null | undefined): value is ScenarioCmaRequest => Boolean(value && ['long_term_scenario', 'conditional_scenario'].includes(value.method))
export const isStatisticalCma = (value: CmaModelRequest | null | undefined): value is StatisticalCmaRequest => Boolean(value && ['historical_statistics', 'bayesian_niw', 'historical_regime_occupancy', 'long_term_scenario', 'conditional_scenario'].includes(value.method))
export const isCoreCma = (value: CmaModelRequest): value is CoreCmaModelRequest => value.method === 'black_litterman' || value.method === 'scenario_mixture'

export const validNiwPriorObservations = (value: number | null | undefined): value is number => typeof value === 'number'
  && Number.isFinite(value) && value > niwPriorObservationBounds.exclusiveMinimum && value <= niwPriorObservationBounds.maximum

/** Draft NaN means unfinished input; it is never accepted as a model estimate. */
export function cmaModelInputError(value: CmaModelRequest): string | null {
  const axis = value.asset_ids
  const between = (n: number, low: number, high: number) => Number.isFinite(n) && n >= low && n <= high
  const source = (text: string) => typeof text === 'string' && text.trim().length >= 3 && text.trim().length <= 2000
  const date = (text: string) => /^\d{4}-\d{2}-\d{2}$/.test(text) && Number.isFinite(Date.parse(text)) && new Date(`${text}T00:00:00Z`).toISOString().slice(0, 10) === text
  const complete = (items: Record<string, number>) => Object.keys(items).length === axis.length && axis.every(a => Object.prototype.hasOwnProperty.call(items, a))
  const matrix = (m: number[][] | null | undefined) => !!m && m.length === axis.length && m.every((row, i) => row.length === axis.length && row.every((v, j) => Number.isFinite(v) && (i !== j || v > 0 && v <= 9) && Math.abs(v - m[j]?.[i]) <= 1e-10))
  if (!axis.length || axis.length > 30 || new Set(axis).size !== axis.length || axis.some(a => !a.trim() || a.length > 120)) return systemText('preInvestment.cmaModelTypes.loadACompleteAssetScopeWithoutDuplicates')
  if (!date(value.as_of) || !/^[A-Z]{3}$/.test(value.currency)) return systemText('preInvestment.cmaModelTypes.confirmTheResearchDateAndThreeLetter')
  if (typeof value.source !== 'string' || value.source.length > 2000) return systemText('preInvestment.cmaModelTypes.keepResearchNotesWithin2000Characters')
  if (isStatisticalCma(value)) {
    const window = value.window ?? { kind: '5Y' }
    if (value.currency !== 'CNY') return systemText('preInvestment.cmaModelTypes.statisticalEvidenceCurrentlySupportsCnySseDaily')
    if (window.kind === 'custom' && (!window.start_date || !window.end_date || !date(window.start_date) || !date(window.end_date) || window.start_date >= window.end_date || window.end_date > value.as_of)) return systemText('preInvestment.cmaModelTypes.enterAValidHistoricalWindowBeforeThe')
    if (value.proxy_inputs) {
      const proxy = value.proxy_inputs
      if (proxy.assets.map(a => a.id).join('\u0000') !== axis.join('\u0000')) return systemText('preInvestment.cmaModelTypes.researchProxiesDoNotMatchTheStrategic')
      for (const asset of proxy.assets) {
        if (asset.asset_type === 'cash') {
          if (asset.cash_return == null || !between(asset.cash_return, -.5, 1)) return systemText('preInvestment.cmaModelTypes.enterAnExplicitCashReturnAssumption')
        } else if (!asset.components.length || asset.components.some(c => !c.series_id || !Number.isFinite(c.weight) || c.weight < 0)
          || Math.abs(asset.components.reduce((sum, c) => sum + c.weight, 0) - 1) > 1e-8) return systemText('preInvestment.cmaModelTypes.selectResearchProxiesForEveryClassWith')
      }
    }
    if ((value.method === 'historical_statistics' || value.method === 'historical_regime_occupancy') && !between(value.shrinkage ?? 0, 0, 1)) return systemText('preInvestment.cmaModelTypes.diagonalShrinkageStrengthMustBeBetween0')
    const ref = (item: { id: string; content_hash: string } | undefined) => Boolean(item?.id && /^[0-9a-f]{64}$/.test(item.content_hash))
    if (isScenarioCma(value)) {
      if (!ref(value.run_ref)) return systemText('preInvestment.cmaModelTypes.selectAHistoricalRegimeStudyAvailableOn')
      if (value.method === 'conditional_scenario') {
        if (!ref(value.realtime_ref)) return systemText('preInvestment.cmaModelTypes.selectAProbabilityCalibratedRealTimeStudy')
        if (!Number.isInteger(value.horizon_days) || value.horizon_days < 1 || value.horizon_days > 2520) return systemText('preInvestment.cmaModelTypes.selectAForecastHorizonFrom1To')
      }
    }
    if (value.method === 'bayesian_niw') {
      if (!ref(value.prior_ref)) return systemText('preInvestment.cmaModelTypes.selectAConfirmedPriorLtcmaWithCompatible')
      if (value.prior_mode !== 'continue' && (!Number.isFinite(value.mean_prior_observations) || !Number.isFinite(value.covariance_prior_observations)
        || (value.mean_prior_observations ?? 0) <= 0 || (value.covariance_prior_observations ?? 0) <= 0)) return systemText('preInvestment.cmaModelTypes.specifyDailyEquivalentObservationCountsForThe')
      if (value.prior_mode !== 'continue' && (!validNiwPriorObservations(value.mean_prior_observations) || !validNiwPriorObservations(value.covariance_prior_observations))) return systemText('preInvestment.cmaModelTypes.meanAndRiskPriorEquivalentDailyObservations', { p0: niwPriorObservationBounds.maximum })
    }
    if (value.method === 'historical_regime_occupancy') {
      if (!ref(value.run_ref)) return systemText('preInvestment.cmaModelTypes.selectASavedDailyRetrospectiveStateRun')
      if (value.probabilities && (Object.values(value.probabilities).some(p => !between(p, 0, 1))
        || Math.abs(Object.values(value.probabilities).reduce((sum, p) => sum + p, 0) - 1) > 1e-8
        || (value.probability_reason?.trim().length ?? 0) < 5)) return systemText('preInvestment.cmaModelTypes.applicationProbabilitiesMustTotal100ExplainWhy')
    }
    return null
  }
  if (value.method === 'black_litterman') {
    if (!complete(value.market_weights) || axis.some(a => !between(value.market_weights[a], 0, 1)) || Math.abs(Object.values(value.market_weights).reduce((a, b) => a + b, 0) - 1) > 1e-8) return systemText('preInvestment.cmaModelTypes.marketWeightsMustBeCompleteNonnegativeAnd')
    if (!source(value.market_weight_source)) return systemText('preInvestment.cmaModelTypes.specifyTheMarketWeightSourceDefaultEqual')
    if (!matrix(value.covariance)) return systemText('preInvestment.cmaModelTypes.enterACompleteSymmetricRiskCovarianceMatrix')
    if (!Number.isFinite(value.delta) || value.delta <= 0 || !Number.isFinite(value.tau) || value.tau <= 0 || !between(value.risk_free_rate, -.5, 2)) return systemText('preInvestment.cmaModelTypes.checkRiskAversionAndFiniteAndPositive')
    if ((value.views?.length ?? 0) > 60) return systemText('preInvestment.cmaModelTypes.upTo60ViewsAreAllowed')
    for (const view of value.views ?? []) {
      if (view.kind === 'basket') {
        if (!view.legs.length || view.legs.length > 30 || new Set(view.legs.map(v => v.asset_id)).size !== view.legs.length
          || view.legs.some(v => !axis.includes(v.asset_id) || !Number.isFinite(v.coefficient))) return systemText('preInvestment.cmaModelTypes.basketViewsNeedUniqueValidAssetsAnd')
        const gross = view.legs.reduce((sum, v) => sum + Math.abs(v.coefficient), 0)
        const total = view.legs.reduce((sum, v) => sum + v.coefficient, 0)
        if (gross < 1e-12 || gross > 4 || Math.abs(total - (view.basis === 'absolute' ? 1 : 0)) > 1e-10) return systemText('preInvestment.cmaModelTypes.absoluteBasketViewCoefficientsMustSumTo')
      } else if (!axis.includes(view.asset_id) || (view.kind === 'relative' && (!view.relative_to || !axis.includes(view.relative_to) || view.relative_to === view.asset_id)) || (view.kind === 'absolute' && view.relative_to != null)) return systemText('preInvestment.cmaModelTypes.eachViewNeedsValidAssetsRelativeViews')
      const basis = view.kind === 'basket' ? view.basis : view.kind
      if (!between(view.annual_return, basis === 'absolute' ? -.5 : -2.5, basis === 'absolute' ? 2 : 2.5) || !Number.isFinite(view.view_std) || view.view_std <= 0 || !Number.isFinite(view.view_std ** 2) || view.view_std ** 2 === 0) return systemText('preInvestment.cmaModelTypes.enterValidViewReturnsAndAStrictly')
      if (!source(view.source) || !date(view.observed_on) || !date(view.available_on) || view.observed_on > view.available_on || view.available_on > value.as_of) return systemText('preInvestment.cmaModelTypes.viewsNeedEvidenceAndMustSatisfyObservation')
    }
  } else {
    if (!value.scenarios.length || value.scenarios.length > 60) return systemText('preInvestment.cmaModelTypes.add160ExplicitScenariosThenEnter')
    if (new Set(value.scenarios.map(s => s.id.trim())).size !== value.scenarios.length) return systemText('preInvestment.cmaModelTypes.scenarioNamesMustBeUnique')
    if (value.scenarios.some(s => !between(s.probability, 0, 1)) || Math.abs(value.scenarios.reduce((sum, s) => sum + s.probability, 0) - 1) > 1e-8) return systemText('preInvestment.cmaModelTypes.scenarioProbabilitiesMustBeExplicitNonnegativeAnd')
    if (value.risk_mode === 'shared' ? !matrix(value.shared_covariance) || value.scenarios.some(s => s.covariance != null) : value.shared_covariance != null || value.scenarios.some(s => !matrix(s.covariance))) return systemText('preInvestment.cmaModelTypes.chooseSharedOrPerScenarioRiskAnd')
    for (const scenario of value.scenarios) {
      if (!scenario.id.trim() || scenario.id.length > 120 || !source(scenario.source)) return systemText('preInvestment.cmaModelTypes.everyScenarioNeedsANameAndRationale')
      if (!complete(scenario.annual_returns) || axis.some(a => !between(scenario.annual_returns[a], -.5, 2))) return systemText('preInvestment.cmaModelTypes.everyScenarioMustCoverAllAssetsWith')
    }
  }
  return null
}

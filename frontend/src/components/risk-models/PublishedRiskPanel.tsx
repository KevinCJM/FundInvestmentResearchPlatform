import { ErrorPanel } from '../ui'
import { systemText, useI18n, i18n } from '../../i18n/runtime'
import { useEffect, useMemo, useRef, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import ReactECharts from 'echarts-for-react'
import type { EChartsOption } from 'echarts'
import {
  getRiskRun, riskPortfolios, riskReleases, runRiskImpact, scenarioReleases,
  type PortfolioChoice, type RiskImpact, type RiskRelease, type RiskRun, type ScenarioRelease,
} from '../../services/riskModels'
import RiskRunView from './RiskRunView'
import { buttonClass, Empty, Feedback, Field, frequencyLabels, inputClass, numberText, percentText, primaryClass, sectionClass, statusLabels, today } from './ResearchUI'

function incompatibility(exposure: RiskRelease | undefined, scenario: ScenarioRelease | undefined): string {
  if (!exposure || !scenario) return ''
  if (exposure.status !== 'active') return systemText('preInvestment.publishedRiskPanel.riskResult', { p0: statusLabels[exposure.status] ?? systemText('preInvestment.publishedRiskPanel.unavailable') })
  if (scenario.status !== 'active') return systemText('preInvestment.publishedRiskPanel.scenario', { p0: statusLabels[scenario.status] ?? systemText('preInvestment.publishedRiskPanel.unavailable') })
  if (exposure.method === 'cashflow' ? scenario.horizon !== 1 : exposure.frequency !== scenario.frequency) return systemText('preInvestment.publishedRiskPanel.periodMismatchCashFlowValuationSupportsOne')
  const missing = scenario.factors.filter(factor => !exposure.inputs.some(input => input.id === factor.id && input.contract_hash === factor.contract_hash))
  return missing.length ? systemText('preInvestment.publishedRiskPanel.thisRiskModelDoesNotCoverFactors', { p0: missing.map(item => item.name).join('、') }) : ''
}

export function RiskImpactResult({ result }: { result: RiskImpact }) {
  useI18n()
  const option = useMemo<EChartsOption>(() => ({
    aria: { enabled: true, description: systemText('preInvestment.publishedRiskPanel.modelExplainedImpactPathUnderThePublished') },
    tooltip: { trigger: 'axis' }, grid: { left: 55, right: 20, top: 20, bottom: 36 },
    xAxis: { type: 'category', data: [0, ...result.path.map(item => item.step)], name: systemText('preInvestment.publishedRiskPanel.periods') },
    yAxis: { type: 'value', scale: true },
    series: [{ name: systemText('preInvestment.publishedRiskPanel.modelImpactPath'), type: 'line', showSymbol: result.path.length < 12, data: [1, ...result.path.map(item => item.nav)] }],
  }), [result, i18n.language])
  return <section className={`${sectionClass} space-y-5`} aria-label={systemText('preInvestment.publishedRiskPanel.currentStressTestResult')}>
    <div><h3 className="font-semibold">{result.name}</h3><p className="mt-1 text-xs text-slate-600">{systemText('preInvestment.publishedRiskPanel.researchDate') + " "}{result.as_of} · {frequencyLabels[result.frequency]} · {result.transient ? systemText('preInvestment.publishedRiskPanel.thisResultHasNotBeenSavedTo') : systemText('preInvestment.publishedRiskPanel.historicalResultSaved')}</p></div>
    <div className="grid gap-5 sm:grid-cols-3"><div><p className="text-xs text-slate-600">{systemText('preInvestment.publishedRiskPanel.modelExplainedScenarioImpact')}</p><p className={`mt-1 text-2xl font-semibold tabular-nums ${result.summary.terminal_return < 0 ? 'text-rose-700' : 'text-accent-800'}`}>{percentText(result.summary.terminal_return)}</p></div><div><p className="text-xs text-slate-600">{systemText('preInvestment.publishedRiskPanel.profitLossOnHypotheticalPrincipal')}</p><p className="mt-1 text-xl font-semibold tabular-nums">{numberText(result.summary.pnl_amount, 2)} {" " + systemText('preInvestment.publishedRiskPanel.cny')}</p></div><div><p className="text-xs text-slate-600">{result.path.length > 1 ? systemText('preInvestment.publishedRiskPanel.maximumDrawdownWithinTheSpecifiedPath') : systemText('preInvestment.publishedRiskPanel.relativeValueAfterShocks')}</p><p className="mt-1 text-xl font-semibold tabular-nums">{result.path.length > 1 ? percentText(result.summary.max_drawdown) : numberText(result.summary.terminal_nav, 4)}</p></div></div>
    {result.path.length > 1 && <div><p className="mb-2 text-xs text-slate-600">{systemText('preInvestment.publishedRiskPanel.startsAt1AndShowsOnlyShock')}</p><ReactECharts option={option} style={{ height: 250 }} notMerge /></div>}
    <div className="grid gap-5 lg:grid-cols-2"><div className="overflow-auto"><table className="w-full text-sm"><caption className="mb-2 text-left font-medium">{systemText('preInvestment.publishedRiskPanel.whichProductsContributedToProfitLoss')}</caption><thead className="text-xs text-slate-600"><tr><th scope="col" className="p-2 text-left">{systemText('preInvestment.publishedRiskPanel.products')}</th><th scope="col" className="p-2 text-right">{systemText('preInvestment.publishedRiskPanel.initialWeight')}</th><th scope="col" className="p-2 text-right">{systemText('preInvestment.publishedRiskPanel.terminalValueContribution')}</th></tr></thead><tbody className="divide-y divide-slate-100">{result.by_asset.map(item => <tr key={item.key}><td className="p-2">{item.name}</td><td className="p-2 text-right tabular-nums">{percentText(item.weight)}</td><td className="p-2 text-right tabular-nums">{percentText(item.contribution)}</td></tr>)}</tbody></table></div><div className="overflow-auto"><table className="w-full text-sm"><caption className="mb-2 text-left font-medium">{systemText('preInvestment.publishedRiskPanel.whichRiskSourcesContributedToProfitLoss')}</caption><thead className="text-xs text-slate-600"><tr><th scope="col" className="p-2 text-left">{systemText('preInvestment.publishedRiskPanel.riskSource')}</th><th scope="col" className="p-2 text-right">{systemText('preInvestment.publishedRiskPanel.terminalValueContribution')}</th></tr></thead><tbody className="divide-y divide-slate-100">{result.by_factor.map(item => <tr key={item.id}><td className="p-2">{item.name}</td><td className="p-2 text-right tabular-nums">{percentText(item.contribution)}</td></tr>)}</tbody></table></div></div>
    <details className="border-t border-slate-100 pt-3"><summary className="cursor-pointer text-sm font-medium text-slate-700">{systemText('preInvestment.publishedRiskPanel.pathDataLimitationsAndSources')}</summary><div className="mt-3 space-y-2 text-xs leading-5 text-slate-600">{result.limitations.map(item => <p key={item}>{item}</p>)}{result.target.holdings_date && <p>{systemText('preInvestment.publishedRiskPanel.endingHoldingsUsed')}{result.target.holdings_date}{systemText('preInvestment.publishedRiskPanel.thePortfolioStrategyWasNotRerun')}</p>}<p>{systemText('preInvestment.publishedRiskPanel.theServerReconciledBothAssetAndFactor')}</p><p className="break-all">{systemText('preInvestment.publishedRiskPanel.scenario2')}{result.request.scenario_release_id}</p><p className="break-all">{systemText('preInvestment.publishedRiskPanel.riskModelResult')}{result.request.exposure_release_id}</p><p className="break-all">{systemText('preInvestment.publishedRiskPanel.result')}{result.id}</p></div><div className="mt-3 max-h-64 overflow-auto"><table className="w-full text-xs"><thead className="text-slate-600"><tr><th scope="col" className="p-2 text-left">{systemText('preInvestment.publishedRiskPanel.periods2')}</th><th scope="col" className="p-2 text-right">{systemText('preInvestment.publishedRiskPanel.periodImpact')}</th><th scope="col" className="p-2 text-right">{systemText('preInvestment.publishedRiskPanel.relativeValue')}</th><th scope="col" className="p-2 text-right">{systemText('preInvestment.publishedRiskPanel.drawdown')}</th></tr></thead><tbody>{result.path.map(item => <tr key={item.step}><td className="p-2">{item.step}</td><td className="p-2 text-right">{percentText(item.return)}</td><td className="p-2 text-right">{numberText(item.nav, 5)}</td><td className="p-2 text-right">{percentText(item.drawdown)}</td></tr>)}</tbody></table></div></details>
  </section>
}

export default function PublishedRiskPanel({ productKey, productName, portfolioRunId, portfolioOnly = false, showReadErrorMascot = false }: { productKey?: string; productName?: string; portfolioRunId?: string; portfolioOnly?: boolean; showReadErrorMascot?: boolean }) {
  useI18n()
  const [params] = useSearchParams()
  const [mode, setMode] = useState<'product' | 'portfolio_run'>(portfolioRunId || portfolioOnly ? 'portfolio_run' : 'product')
  const [asOf, setAsOf] = useState(today)
  const [exposures, setExposures] = useState<RiskRelease[]>([])
  const [scenarios, setScenarios] = useState<ScenarioRelease[]>([])
  const [portfolios, setPortfolios] = useState<PortfolioChoice[]>([])
  const [exposureId, setExposureId] = useState(params.get('exposure_release_id') ?? '')
  const [scenarioId, setScenarioId] = useState(params.get('scenario_release_id') ?? '')
  const [selectedProduct, setSelectedProduct] = useState(productKey ?? params.get('product_key') ?? '')
  const [selectedPortfolio, setSelectedPortfolio] = useState(portfolioRunId ?? params.get('portfolio_run_id') ?? '')
  const [exposureRun, setExposureRun] = useState<RiskRun | null>(null)
  const [result, setResult] = useState<RiskImpact | null>(null)
  const [compareEnabled, setCompareEnabled] = useState(false)
  const [comparisonPortfolio, setComparisonPortfolio] = useState('')
  const [comparison, setComparison] = useState<RiskImpact | null>(null)
  const [notional, setNotional] = useState('1000000')
  const [policy, setPolicy] = useState<'buy_and_hold' | 'constant_weights_zero_cost'>('buy_and_hold')
  const [acknowledged, setAcknowledged] = useState(false)
  const [loading, setLoading] = useState(true)
  const [catalogFailed, setCatalogFailed] = useState(false)
  const [readingRun, setReadingRun] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [refresh, setRefresh] = useState(0)
  const alive = useRef(true); const current = useRef('')
  const exposure = exposures.find(item => item.id === exposureId)
  const scenario = scenarios.find(item => item.id === scenarioId)
  const mismatch = incompatibility(exposure, scenario)
  const comparing = mode === 'portfolio_run' && compareEnabled
  const comparisonMissing = comparing && (!comparisonPortfolio || comparisonPortfolio === selectedPortfolio)
  const signature = JSON.stringify([productKey, portfolioRunId, mode, asOf, exposureId, scenarioId, selectedProduct, selectedPortfolio, notional, policy, comparing, comparisonPortfolio])
  current.current = signature
  useEffect(() => { alive.current = true; return () => { alive.current = false } }, [])
  useEffect(() => { setResult(null); setComparison(null); setAcknowledged(false); setError('') }, [signature])
  useEffect(() => { if (productKey) { setSelectedProduct(productKey); setMode('product') } if (portfolioRunId) { setSelectedPortfolio(portfolioRunId); setMode('portfolio_run') } }, [productKey, portfolioRunId])
  useEffect(() => {
    const controller = new AbortController(); setLoading(true); setCatalogFailed(false); setError(''); setExposures([]); setScenarios([]); setExposureRun(null)
    Promise.all([riskReleases('product', { as_of: asOf, ...(productKey ? { product_key: productKey } : {}) }, controller.signal), scenarioReleases(asOf, controller.signal)])
      .then(([nextExposures, nextScenarios]) => { if (!controller.signal.aborted) { setExposures(nextExposures); setScenarios(nextScenarios); setExposureId(old => old || nextExposures.find(item => item.status === 'active')?.id || '') } })
      .catch(caught => { if (!controller.signal.aborted) { setCatalogFailed(true); setError(caught instanceof Error ? caught.message : systemText('preInvestment.publishedRiskPanel.unableToLoadPublishedResults')) } })
      .finally(() => { if (!controller.signal.aborted) setLoading(false) })
    return () => controller.abort()
  }, [asOf, productKey, refresh])
  useEffect(() => {
    if (mode !== 'portfolio_run' || (portfolioRunId && !comparing)) return
    const controller = new AbortController()
    riskPortfolios(controller.signal).then(items => { if (!controller.signal.aborted) setPortfolios(items) }).catch(caught => { if (!controller.signal.aborted) setError(caught instanceof Error ? caught.message : systemText('preInvestment.publishedRiskPanel.unableToLoadPortfolioSnapshots')) })
    return () => controller.abort()
  }, [mode, portfolioRunId, comparing, refresh])
  useEffect(() => {
    const controller = new AbortController(); setExposureRun(null)
    if (!exposure) { setReadingRun(false); return () => controller.abort() }
    setReadingRun(true)
    getRiskRun('product', exposure.run_id, controller.signal).then(run => { if (!controller.signal.aborted) setExposureRun(run) }).catch(caught => { if (!controller.signal.aborted) setError(caught instanceof Error ? caught.message : systemText('preInvestment.publishedRiskPanel.unableToLoadSensitivityResults')) }).finally(() => { if (!controller.signal.aborted) setReadingRun(false) })
    return () => controller.abort()
  }, [exposure?.run_id])
  async function calculate() {
    if (!exposure || !scenario || mismatch || !acknowledged || loading || busy || comparisonMissing) return
    const requested = signature; setBusy(true); setError(''); setResult(null); setComparison(null)
    try {
      if (!Number.isFinite(Number(notional)) || Number(notional) <= 0 || (mode === 'product' ? !selectedProduct : !selectedPortfolio)) throw new Error(systemText('preInvestment.publishedRiskPanel.selectAResearchTargetAndEnterPositive'))
      const shared = { scenario_release_id: scenario.id, exposure_release_id: exposure.id, as_of: asOf, holding_policy: policy, hold_other_factors_constant: true as const, notional: Number(notional), usage: 'research' as const }
      const primary = runRiskImpact({ ...shared, target: mode === 'product' ? { kind: 'product', product_key: selectedProduct } : { kind: 'portfolio_run', portfolio_run_id: selectedPortfolio } })
      const other = comparing ? runRiskImpact({ ...shared, target: { kind: 'portfolio_run', portfolio_run_id: comparisonPortfolio } }) : Promise.resolve(null)
      const [value, compared] = await Promise.all([primary, other])
      if (alive.current && current.current === requested) { setResult(value); setComparison(compared) }
    } catch (caught) { if (alive.current && current.current === requested) setError(caught instanceof Error ? caught.message : systemText('preInvestment.publishedRiskPanel.scenarioStressTestDidNotComplete')) }
    finally { if (alive.current) setBusy(false) }
  }
  const riskLink = `/settings/risk-models${productKey ? `?${new URLSearchParams({ product_key: productKey, product_name: productName ?? productKey })}` : ''}`
  const displayRun = exposureRun && mode === 'product' && selectedProduct ? { ...exposureRun, rows: exposureRun.rows.filter(row => row.target_id === selectedProduct) } : exposureRun
  if (!loading && catalogFailed && !result) return <ErrorPanel mascot={showReadErrorMascot} onRetry={() => setRefresh(old => old + 1)} />
  return <div className="min-w-0 space-y-4" data-testid="published-risk-panel"><section className={`${sectionClass} space-y-4`}>
    <div><h3 className="text-lg font-semibold">{productKey ? systemText('preInvestment.publishedRiskPanel.whatRisksAffectThisProduct') : systemText('preInvestment.publishedRiskPanel.stressTestWithPublishedResults')}</h3><p className="mt-1 text-sm leading-6 text-slate-600">{systemText('preInvestment.publishedRiskPanel.onlyPublishedSensitivitiesAndScenariosAreLoaded')}</p></div>
    <Feedback error={error} />
    <fieldset disabled={busy} className="min-w-0 space-y-4"><div className="grid gap-4 sm:grid-cols-2"><Field label={systemText('preInvestment.publishedRiskPanel.riskResearchDate')} hint={systemText('preInvestment.publishedRiskPanel.independentOfTheHistoricalPerformanceIntervalModels')}><input className={inputClass} type="date" max={today()} value={asOf} onChange={event => setAsOf(event.target.value)} /></Field>{!productKey && !portfolioRunId && !portfolioOnly && <Field label={systemText('preInvestment.publishedRiskPanel.testTarget')}><select className={inputClass} value={mode} onChange={event => setMode(event.target.value as typeof mode)}><option value="product">{systemText('preInvestment.publishedRiskPanel.singleProduct')}</option><option value="portfolio_run">{systemText('preInvestment.publishedRiskPanel.savedProductPortfolio')}</option></select></Field>}</div>
      {loading ? <p role="status" className="py-3 text-sm text-slate-600">{systemText('preInvestment.publishedRiskPanel.loadingLocalPublishedResultsWithoutRecalculatingSensitivities')}</p> : !exposures.length ? <Empty title={productKey ? systemText('preInvestment.publishedRiskPanel.noPublishedRiskModelForThisProduct') : systemText('preInvestment.publishedRiskPanel.noPublishedRiskModelsYet')}><p>{systemText('preInvestment.publishedRiskPanel.selectATargetCalculateAndValidateIn')}</p><Link className={`${primaryClass} mt-3`} to={riskLink}>{systemText('preInvestment.publishedRiskPanel.researchInTheRiskModelCenter')}</Link></Empty> : <>
        <Field label={systemText('preInvestment.publishedRiskPanel.selectAPublishedSensitivityResult')}><select className={inputClass} value={exposureId} onChange={event => setExposureId(event.target.value)}><option value="">{systemText('preInvestment.publishedRiskPanel.selectARiskResult')}</option>{exposures.map(item => <option key={item.id} value={item.id}>{item.name} · {frequencyLabels[item.frequency]} · {item.as_of} · {statusLabels[item.status] ?? item.status}</option>)}</select></Field>
        {exposureId && !exposure && <p role="status" className="text-sm text-amber-800">{systemText('preInvestment.publishedRiskPanel.theSpecifiedRiskResultIsUnavailableFor')}</p>}
        {scenarioId && !scenario && <p role="status" className="text-sm text-amber-800">{systemText('preInvestment.publishedRiskPanel.theSpecifiedScenarioIsNotInThe')}</p>}
        {exposure && <p className={`rounded-lg p-3 text-xs leading-5 ${exposure.status === 'active' ? 'bg-slate-50 text-slate-600' : 'bg-amber-50 text-amber-900'}`}>{systemText('preInvestment.publishedRiskPanel.status')}{statusLabels[exposure.status] ?? exposure.status}{systemText('preInvestment.publishedRiskPanel.effective') + " "}{exposure.effective_at.slice(0, 10)}{systemText('preInvestment.publishedRiskPanel.expires') + " "}{exposure.expires_at.slice(0, 10)}{systemText('preInvestment.publishedRiskPanel.onlyThisResultIsReferencedNoOther')}<Link className="ml-2 underline" to={riskLink}>{systemText('preInvestment.publishedRiskPanel.viewOrResearchAgain')}</Link></p>}
        {displayRun && <RiskRunView run={displayRun} published />}
        {mode === 'product' && !productKey && <Field label={systemText('preInvestment.publishedRiskPanel.selectProduct')}><select className={inputClass} value={selectedProduct} onChange={event => setSelectedProduct(event.target.value)}><option value="">{systemText('preInvestment.publishedRiskPanel.selectAProductCoveredByThisResult')}</option>{exposure?.targets.map(item => <option key={item.key} value={item.key}>{item.name} · {item.product_id}</option>)}</select></Field>}
        {mode === 'portfolio_run' && (portfolioRunId ? <p className="rounded-lg bg-slate-50 p-3 text-xs text-slate-600">{systemText('preInvestment.publishedRiskPanel.usesEndingHoldingsFromTheCurrentImmutable')}</p> : <Field label={systemText('preInvestment.publishedRiskPanel.selectASavedProductPortfolioSnapshot')} hint={systemText('preInvestment.publishedRiskPanel.testsAnExplicitlySavedProductPortfolioWithout')}><select className={inputClass} value={selectedPortfolio} onChange={event => setSelectedPortfolio(event.target.value)}><option value="">{systemText('preInvestment.publishedRiskPanel.selectAnImmutablePortfolioRun')}</option>{portfolios.map(item => <option key={item.id} value={item.id}>{item.name} {" " + systemText('preInvestment.publishedRiskPanel.holdingsAsOf') + " "}{item.as_of}</option>)}</select>{!portfolios.length && <Link className="mt-2 block text-xs text-accent-800 underline" to="/pre-investment/product-allocation-timing/construction">{systemText('preInvestment.publishedRiskPanel.saveProductPortfolioResearchFirst')}</Link>}</Field>)}
        {mode === 'portfolio_run' && <div className="space-y-3"><label className="flex min-h-11 items-center gap-2 text-sm text-slate-700"><input type="checkbox" checked={compareEnabled} onChange={event => setCompareEnabled(event.target.checked)} />{systemText('preInvestment.publishedRiskPanel.compareAnotherSavedPortfolio')}</label>{comparing && <Field label={systemText('preInvestment.publishedRiskPanel.selectComparisonPortfolioSnapshot')} hint={systemText('preInvestment.publishedRiskPanel.bothPlansUseTheSameScenarioSensitivities')}><select className={inputClass} value={comparisonPortfolio} onChange={event => setComparisonPortfolio(event.target.value)}><option value="">{systemText('preInvestment.publishedRiskPanel.selectADifferentPortfolioSnapshot')}</option>{portfolios.filter(item => item.id !== selectedPortfolio).map(item => <option key={item.id} value={item.id}>{item.name} {" " + systemText('preInvestment.publishedRiskPanel.holdingsAsOf') + " "}{item.as_of}</option>)}</select></Field>}</div>}
        <div className="border-t border-slate-100 pt-4"><Field label={systemText('preInvestment.publishedRiskPanel.selectPublishedScenario')}><select className={inputClass} value={scenarioId} onChange={event => setScenarioId(event.target.value)}><option value="">{systemText('preInvestment.publishedRiskPanel.selectAScenarioToTest')}</option>{scenarios.map(item => <option key={item.id} value={item.id}>{item.name} · {frequencyLabels[item.frequency]} · {item.horizon} {" " + systemText('preInvestment.publishedRiskPanel.periods3') + " "}{statusLabels[item.status] ?? item.status}</option>)}</select></Field>{!scenarios.length && <p className="mt-2 text-sm text-amber-800">{systemText('preInvestment.publishedRiskPanel.noPublishedScenariosYet')}<Link className="ml-2 underline" to="/settings/scenario-algorithms?center=simulation">{systemText('preInvestment.publishedRiskPanel.buildAndPublishAScenario')}</Link></p>}</div>
        {mismatch && <p role="status" className="rounded-lg bg-amber-50 p-3 text-sm text-amber-900">{mismatch}</p>}
        <div className="grid gap-4 sm:grid-cols-2"><Field label={systemText('preInvestment.publishedRiskPanel.hypotheticalPrincipalCny')} hint={systemText('preInvestment.publishedRiskPanel.usedOnlyToConvertReturnsToMonetary')}><input type="number" min="1" className={inputClass} value={notional} onChange={event => setNotional(event.target.value)} /></Field><Field label={systemText('preInvestment.publishedRiskPanel.holdingRuleDuringShocks')}><select className={inputClass} value={policy} onChange={event => setPolicy(event.target.value as typeof policy)}><option value="buy_and_hold">{systemText('preInvestment.publishedRiskPanel.noRebalancingWeightsDriftNaturally')}</option><option value="constant_weights_zero_cost">{systemText('preInvestment.publishedRiskPanel.restoreOriginalWeightsEachPeriodAssumesZero')}</option></select></Field></div>
        <label className="flex items-start gap-2 text-sm leading-6 text-slate-600"><input className="mt-1" type="checkbox" checked={acknowledged} onChange={event => setAcknowledged(event.target.checked)} />{systemText('preInvestment.publishedRiskPanel.modeledFactorsWithoutShocksRemainUnchangedUnmodeled')}</label>
        <div className="flex flex-wrap gap-3"><button type="button" className={primaryClass} disabled={busy || loading || readingRun || !exposureRun || !exposure || !scenario || Boolean(mismatch) || !acknowledged || comparisonMissing || (mode === 'product' ? !selectedProduct : !selectedPortfolio)} onClick={() => void calculate()}>{busy ? systemText('preInvestment.publishedRiskPanel.applyingPublishedResults') : comparing ? systemText('preInvestment.publishedRiskPanel.calculateAndCompare') : systemText('preInvestment.publishedRiskPanel.calculateScenarioImpact')}</button></div>
      </>}
    </fieldset>
    <button type="button" className={buttonClass} disabled={busy || loading} onClick={() => { setError(''); setRefresh(old => old + 1) }}>{systemText('preInvestment.publishedRiskPanel.refreshResultsCatalog')}</button>
  </section>{result && (comparison ? <RiskImpactComparison primary={result} comparison={comparison} /> : <RiskImpactResult result={result} />)}</div>
}

export function RiskImpactComparison({ primary, comparison }: { primary: RiskImpact; comparison: RiskImpact }) {
  useI18n()
  return <section className={`${sectionClass} space-y-4`} aria-label={systemText('preInvestment.publishedRiskPanel.portfolioScenarioComparison')}>
    <h3 className="font-semibold">{systemText('preInvestment.publishedRiskPanel.whichPlanIsMoreAffectedByThe')}</h3>
    <p className="text-sm text-slate-600">{systemText('preInvestment.publishedRiskPanel.usesTheSameScenarioVersionAndRisk')}</p>
    <div className="overflow-x-auto"><table className="w-full min-w-[420px] text-sm"><thead className="text-left text-slate-600"><tr><th scope="col" className="p-2">{systemText('preInvestment.publishedRiskPanel.comparisonMetric')}</th><th scope="col" className="p-2">{systemText('preInvestment.publishedRiskPanel.currentPlan')}</th><th scope="col" className="p-2">{systemText('preInvestment.publishedRiskPanel.comparisonPlan')}</th></tr></thead><tbody className="divide-y divide-slate-100">
      <tr><th scope="row" className="p-2 text-left font-medium">{systemText('preInvestment.publishedRiskPanel.portfolio')}</th>{[primary, comparison].map(value => <td className="p-2" key={value.id}>{value.target.name}</td>)}</tr>
      <tr><th scope="row" className="p-2 text-left font-medium">{systemText('preInvestment.publishedRiskPanel.holdingsAsOf2')}</th>{[primary, comparison].map(value => <td className="p-2" key={value.id}>{value.target.holdings_date ?? systemText('preInvestment.publishedRiskPanel.notProvided')}</td>)}</tr>
      <tr><th scope="row" className="p-2 text-left font-medium">{systemText('preInvestment.publishedRiskPanel.scenarioImpact')}</th>{[primary, comparison].map(value => <td className="p-2 font-semibold tabular-nums" key={value.id}>{percentText(value.summary.terminal_return)}</td>)}</tr>
      <tr><th scope="row" className="p-2 text-left font-medium">{systemText('preInvestment.publishedRiskPanel.hypotheticalPrincipalProfitLoss')}</th>{[primary, comparison].map(value => <td className="p-2 tabular-nums" key={value.id}>{numberText(value.summary.pnl_amount, 2)} {" " + systemText('preInvestment.publishedRiskPanel.cny')}</td>)}</tr>
    </tbody></table></div>
  </section>
}

export function PortfolioRiskSection({ portfolioRunId, context }: { portfolioRunId?: string; context?: string }) {
  useI18n()
  const [open, setOpen] = useState(false)
  return <details className="mt-5 min-w-0 rounded-xl border border-slate-200 bg-white p-4" onToggle={event => setOpen(event.currentTarget.open)}><summary className="cursor-pointer text-sm font-semibold text-slate-800">{systemText('preInvestment.publishedRiskPanel.portfolioStressTestWithPublishedModelsAnd')}</summary>{context && <p className="mt-3 text-sm leading-6 text-slate-600">{context}</p>}{open && <div className="mt-4"><PublishedRiskPanel portfolioRunId={portfolioRunId} portfolioOnly /></div>}</details>
}

import { systemText, useI18n } from '../i18n/runtime'
import { useEffect, useMemo, useRef, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import {
  createResearchTarget,
  PortfolioConstituent,
  PortfolioInstrument,
  PortfolioMethod,
  PortfolioRun,
  PortfolioRunRequest,
  runPortfolio,
} from '../services/portfolioResearch'
import { MetricValue } from '../components/metrics/MetricDisplay'
import type { MetricPresentation } from '../services/customIndicators'
import { humanizeIndicatorMessage } from '../utils/indicatorDiagnostics'
import { evaluateNumericControls, type NumericControlResult } from '../services/businessNumeric'
import {
  getInvestableUniverse,
  investableUniverseEligibleCount,
  searchInvestableUniverseProducts,
  type InvestableUniverseSnapshot,
} from '../services/productPools'
import {
  HistoricalRegimeBacktestSelector,
  RegimeConditioningPanel,
} from '../components/HistoricalRegimeBacktest'
import type { HistoricalRegimeBacktestReference } from '../services/portfolioRegime'
import { PortfolioRiskSection } from '../components/risk-models/PublishedRiskPanel'
import { readAllocationDraft, updateAllocationJourney, writeAllocationDraft } from '../app/allocationJourney'



const percent = (value: number | null | undefined, digits = 2) =>
  value === null || value === undefined || !Number.isFinite(value) ? '—' : `${(value * 100).toFixed(digits)}%`

const resultMetricPresentation = (metric: PortfolioRun['metrics'][number]): MetricPresentation => metric.presentation ?? {
  indicator_id: metric.metric_id ?? metric.name, revision: 1, name: metric.name, source: 'built_in', category: 'portfolio_summary', category_label: systemText('preInvestment.portfolioConstruction.portfolioSummary'), context_kind: 'portfolio', catalog_status: 'current',
  display_format: metric.unit === 'percent' ? 'percent' : 'number', precision: 3, unit: metric.unit === 'percent' ? '%' : metric.unit ?? '', notation: 'standard', value_scale: metric.unit === 'percent' ? 100 : 1,
  output_measure: 'dimensionless', direction: metric.direction ?? 'higher_better', description: '', methodology: '', data_basis: systemText('preInvestment.portfolioConstruction.lockRunSnapshot'), minimum_observations: 1, applicable_product_kinds: ['portfolio'],
}

type PortfolioDraft = {
  name: string; universeId: string; constituents: PortfolioConstituent[]; method: PortfolioMethod;
  minWeight: number; maxWeight: number; windowMode: 'all' | 'rolling'; observations: number;
  rebalance: PortfolioRunRequest['rebalance']['frequency']; benchmark: PortfolioInstrument | null;
  objective: NonNullable<PortfolioRunRequest['objective']>; targetReturn: number;
  allocationSource?: PortfolioRunRequest['allocation_source']; historicalRegime: HistoricalRegimeBacktestReference | null;
}

export default function PortfolioConstruction() {

  useI18n()
  const [searchParams] = useSearchParams()
  return <PortfolioConstructionEditor key={searchParams.toString()} />
}

function PortfolioConstructionEditor() {
  const METHODS: Array<{ value: PortfolioMethod; label: string; help: string }> = [
  { value: 'equal_weight', label: systemText('preInvestment.portfolioConstruction.equalWeight'), help: systemText('preInvestment.portfolioConstruction.allocateCapitalEquallyAmongSelectedProducts') },
  { value: 'manual', label: systemText('preInvestment.portfolioConstruction.manualWeights'), help: systemText('preInvestment.portfolioConstruction.setProductWeightsDirectlyFromResearchAssumptions') },
  { value: 'risk_budget', label: systemText('preInvestment.portfolioConstruction.riskBudget'), help: systemText('preInvestment.portfolioConstruction.solveFundingWeightsForTargetRiskContributions') },
  { value: 'target_optimization', label: systemText('preInvestment.portfolioConstruction.targetOptimization'), help: systemText('preInvestment.portfolioConstruction.useHistoricalWindowsToSolveForMaximum') },
]
  useI18n()
  const [searchParams] = useSearchParams()
  const [initialImport] = useState(() => {
    try { return JSON.parse(sessionStorage.getItem('portfolioResearchImport') ?? 'null') } catch { return null }
  })
  // 地址栏与显式交接负载决定草稿身份；裸路径进来落在 local 草稿，不认领上次研究的可投资域。
  const draftScope = `products:${searchParams.get('universe') ?? initialImport?.universe_snapshot_id ?? 'local'}:${searchParams.get('decision') ?? initialImport?.allocation_source?.decision_id ?? 'manual'}`
  const [draft] = useState(() => readAllocationDraft<PortfolioDraft>(draftScope))
  const [inputsChanged, setInputsChanged] = useState(false)
  const [universeId, setUniverseId] = useState(() => searchParams.get('universe') ?? initialImport?.universe_snapshot_id ?? draft?.universeId ?? '')
  const [universe, setUniverse] = useState<InvestableUniverseSnapshot | null>(null)
  const [universeLoading, setUniverseLoading] = useState(false)
  const [step, setStep] = useState(1)
  const [name, setName] = useState(draft?.name ?? systemText('preInvestment.portfolioConstruction.unnamedPortfolioStudy'))
  const [query, setQuery] = useState('')
  const [searching, setSearching] = useState(false)
  const [searchError, setSearchError] = useState('')
  const [candidates, setCandidates] = useState<PortfolioInstrument[]>([])
  const [constituents, setConstituents] = useState<PortfolioConstituent[]>(draft?.constituents ?? [])
  const [method, setMethod] = useState<PortfolioMethod>(draft?.method ?? 'equal_weight')
  const [minWeight, setMinWeight] = useState(draft?.minWeight ?? 0)
  const [maxWeight, setMaxWeight] = useState(draft?.maxWeight ?? 100)
  const [windowMode, setWindowMode] = useState<'all' | 'rolling'>(draft?.windowMode ?? 'all')
  const [observations, setObservations] = useState(draft?.observations ?? 252)
  const [rebalance, setRebalance] = useState<PortfolioRunRequest['rebalance']['frequency']>(draft?.rebalance ?? 'monthly')
  const [benchmark, setBenchmark] = useState<PortfolioInstrument | null>(draft?.benchmark ?? null)
  const [objective, setObjective] = useState<NonNullable<PortfolioRunRequest['objective']>>(draft?.objective ?? 'max_sharpe')
  const [targetReturn, setTargetReturn] = useState(draft?.targetReturn ?? 8)
  const [running, setRunning] = useState(false)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')
  const [result, setResult] = useState<PortfolioRun | null>(null)
  const [historicalRegime, setHistoricalRegime] = useState<HistoricalRegimeBacktestReference | null>(draft?.historicalRegime ?? null)
  const [numericControls, setNumericControls] = useState<Record<string, NumericControlResult>>({})
  const [numericControlError, setNumericControlError] = useState('')
  const [selectedAssetClassId, setSelectedAssetClassId] = useState('')
  const [allocationSource, setAllocationSource] = useState<PortfolioRunRequest['allocation_source']>(draft?.allocationSource)

  useEffect(() => {
    const raw = sessionStorage.getItem('portfolioResearchImport')
    if (!raw) return
    sessionStorage.removeItem('portfolioResearchImport')
    try {
      const imported = JSON.parse(raw) as {
        name?: string
        method?: PortfolioMethod
        universe_snapshot_id?: string
        constituents?: PortfolioConstituent[]
        allocation_source?: PortfolioRunRequest['allocation_source']
      }
      if (imported.name) setName(imported.name)
      if (imported.allocation_source) { setAllocationSource(imported.allocation_source); setMethod('manual'); updateAllocationJourney({ taaRunId: imported.allocation_source.decision_id, universeId: imported.universe_snapshot_id }) }
      if (imported.universe_snapshot_id) setUniverseId(imported.universe_snapshot_id)
      if (Array.isArray(imported.constituents)) setConstituents(imported.constituents)
      if (!imported.allocation_source && imported.method && METHODS.some((item) => item.value === imported.method)) setMethod(imported.method)
    } catch {
      setError(systemText('preInvestment.portfolioConstruction.theImportedPortfolioConfigurationCannotBeRead'))
    }
  }, [])

  const inputState: PortfolioDraft = { name, universeId, constituents, method, minWeight, maxWeight, windowMode, observations, rebalance, benchmark, objective, targetReturn, allocationSource, historicalRegime }
  const inputKey = JSON.stringify(inputState)
  const latestInput = useRef(inputKey)
  latestInput.current = inputKey
  useEffect(() => {
    if (universeId && constituents.length) writeAllocationDraft(draftScope, inputState)
    setResult(null)
    setInputsChanged(true)
  }, [inputKey, draftScope])

  useEffect(() => {
    if (!universeId) {
      setUniverse(null)
      return
    }
    let active = true
    setUniverseLoading(true)
    getInvestableUniverse(universeId)
      .then((snapshot) => { if (active) setUniverse(snapshot) })
      .catch((caught) => {
        if (active) {
          setUniverse(null)
          setError(caught instanceof Error ? caught.message : systemText('preInvestment.portfolioConstruction.unableToLoadTheInvestableUniverseSnapshot'))
        }
      })
      .finally(() => { if (active) setUniverseLoading(false) })
    return () => { active = false }
  }, [universeId])

  useEffect(() => {
    const controller = new AbortController()
    const trimmed = query.trim()
    if (!trimmed || !universeId || !universe) {
      setCandidates([])
      return () => controller.abort()
    }
    setSearching(true)
    setSearchError('')
    searchInvestableUniverseProducts(universeId, {
      query: trimmed,
      eligibleOnly: true,
      pageSize: 20,
      signal: controller.signal,
    })
      .then((response) => setCandidates(response.items.map((item) => ({
        product_id: item.product_id,
        kind: item.kind,
        name: item.name,
        code: item.code,
      }))))
      .catch((caught) => {
        if ((caught as DOMException)?.name !== 'AbortError') {
          setSearchError(caught instanceof Error ? caught.message : systemText('preInvestment.portfolioConstruction.productSearchFailed'))
        }
      })
      .finally(() => { if (!controller.signal.aborted) setSearching(false) })
    return () => controller.abort()
  }, [query, universe, universeId])

  const assetClasses = useMemo(() => {
    const items = new Map<string, string>()
    constituents.forEach((item) => {
      if (item.asset_class_id && item.asset_class_name) {
        items.set(item.asset_class_id, item.asset_class_name)
      }
    })
    return [...items.entries()].map(([id, name]) => ({ id, name }))
  }, [constituents])

  useEffect(() => {
    if (assetClasses.some((item) => item.id === selectedAssetClassId)) return
    setSelectedAssetClassId(assetClasses[0]?.id ?? '')
  }, [assetClasses, selectedAssetClassId])

  useEffect(() => {
    const controller = new AbortController()
    setNumericControls({})
    setNumericControlError('')
    evaluateNumericControls([
      { key: 'portfolio-weights', values: constituents.map((item) => item.weight ?? 0), target: 100, tolerance: 0.01 },
      { key: 'portfolio-risk-budgets', values: constituents.map((item) => item.risk_budget ?? 0), target: 100, tolerance: 0.01 },
      ...Object.entries(allocationSource?.class_weights ?? {}).map(([name, weight]) => ({ key: `class-budget:${name}`, values: constituents.filter(item => item.asset_class_name === name).map(item => item.weight ?? 0), target: weight * 100, tolerance: 0.01 })),
    ], controller.signal)
      .then((response) => setNumericControls(Object.fromEntries(response.items.map((item) => [item.key, item]))))
      .catch((reason) => {
        if ((reason as DOMException)?.name !== 'AbortError') setNumericControlError(systemText('preInvestment.portfolioConstruction.weightValidationIsTemporarilyUnavailableRetryLater'))
      })
    return () => controller.abort()
  }, [constituents, allocationSource])

  const weightControl = numericControls['portfolio-weights']
  const budgetControl = numericControls['portfolio-risk-budgets']
  const classBudgetsValid = !allocationSource || (method === 'manual' && Object.keys(allocationSource.class_weights).every(name => numericControls[`class-budget:${name}`]?.within_tolerance === true))
  const readyToRun = classBudgetsValid && Boolean(universe) && constituents.length >= 2 && minWeight <= maxWeight &&
    constituents.every((item) => item.asset_class_id && item.asset_class_name) &&
    (method !== 'manual' || weightControl?.within_tolerance === true) &&
    (method !== 'risk_budget' || budgetControl?.within_tolerance === true)

  function addInstrument(item: PortfolioInstrument) {
    const assetClass = assetClasses.find((entry) => entry.id === selectedAssetClassId)
    if (!assetClass) {
      setError(systemText('preInvestment.portfolioConstruction.importAtLeastOneAssetClassFrom'))
      return
    }
    setConstituents((current) => current.some((entry) => entry.product_id === item.product_id && entry.kind === item.kind)
      ? current
      : [...current, {
        ...item,
        weight: 0,
        risk_budget: 0,
        asset_class_id: assetClass.id,
        asset_class_name: assetClass.name,
      }])
  }
  function updateNumber(index: number, field: 'weight' | 'risk_budget', raw: string) {
    const value = Math.max(0, Number(raw) || 0)
    setConstituents((current) => current.map((item, currentIndex) => currentIndex === index ? { ...item, [field]: value } : item))
  }
  function buildRequest(): PortfolioRunRequest {
    return {
      name: name.trim() || systemText('preInvestment.portfolioConstruction.unnamedPortfolioStudy'),
      universe_snapshot_id: universe?.id ?? (universeId || null),
      allocation_source: allocationSource,
      constituents,
      method,
      constraints: { min_weight: minWeight / 100, max_weight: maxWeight / 100 },
      window: windowMode === 'all' ? { mode: 'all' } : { mode: 'rolling', observations },
      rebalance: { frequency: rebalance, transaction_cost_bps: 0 },
      benchmark: benchmark ? { kind: benchmark.kind, product_id: benchmark.product_id, name: benchmark.name } : null,
      objective: method === 'target_optimization' ? objective : null,
      target_return: method === 'target_optimization' && objective === 'target_return' ? targetReturn / 100 : null,
    }
  }
  async function handleRun() {
    if (!readyToRun) { setError(systemText('preInvestment.portfolioConstruction.lockAnInvestableUniverseImportProductsFrom')); return }
    const submittedInput = inputKey
    setRunning(true); setError('')
    try {
      const target = await createResearchTarget({ name: name.trim() || systemText('preInvestment.portfolioConstruction.unnamedPortfolioStudy'), kind: 'portfolio', definition: buildRequest() })
      const run = await runPortfolio(target.id, { historical_regime: historicalRegime })
      if (latestInput.current !== submittedInput) { setError(systemText('preInvestment.portfolioConstruction.researchInputsChangedRunTheCurrentPlan')); return }
      setResult({ ...run, target_id: target.id })
      setInputsChanged(false)
      setStep(4)
    } catch (caught: any) { setError(humanizeIndicatorMessage(caught?.message, systemText('preInvestment.portfolioConstruction.portfolioRunFailed'))) } finally { setRunning(false) }
  }
  async function handleSave() {
    if (!result) return
    setSaving(true); setError('')
    try {
      const target = await createResearchTarget({ name: name.trim() || result.name, kind: 'portfolio', definition: buildRequest() })
      setResult({ ...result, target_id: target.id })
    } catch (caught: any) { setError(humanizeIndicatorMessage(caught?.message, systemText('preInvestment.portfolioConstruction.unableToSaveResearchTarget'))) } finally { setSaving(false) }
  }

  return <div className="mx-auto max-w-7xl space-y-6 p-4 sm:p-6" aria-busy={running || saving || universeLoading}>
    <header className="rounded-xl bg-slate-900 px-5 py-6 text-white sm:px-7">
      <p className="text-sm text-emerald-300">{systemText('preInvestment.portfolioConstruction.workspaceSharedRealHistoricalData')}</p><h1 className="mt-1 text-2xl font-semibold">{systemText('preInvestment.portfolioConstruction.productPortfolioConstruction')}</h1>
      <p className="mt-2 max-w-3xl text-sm text-slate-200">{systemText('preInvestment.portfolioConstruction.allocateAssetClassBudgetsToProductsCheck')}</p>
    </header>
    {universe ? <div className="rounded-xl border border-emerald-200 bg-emerald-50 px-4 py-3 text-sm text-emerald-900">{systemText('preInvestment.portfolioConstruction.investableUniverse')}<b>{universe.name}</b> · {investableUniverseEligibleCount(universe)} {" " + systemText('preInvestment.portfolioConstruction.availableProductsResearchDate') + " "}{universe.research_date}</div> : <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-amber-300 bg-amber-50 px-4 py-3 text-sm text-amber-900"><span>{universeLoading ? systemText('preInvestment.portfolioConstruction.loadingInvestableUniverse') : systemText('preInvestment.portfolioConstruction.enterThroughProductPoolsAndAssetConstruction')}</span><Link to="/pre-investment/product-pool" className="rounded-lg bg-amber-800 px-3 py-2 font-medium text-white">{systemText('preInvestment.portfolioConstruction.selectProductPoolVersions')}</Link></div>}
    <ol className="grid grid-cols-2 gap-2 text-sm sm:grid-cols-4" aria-label={systemText('preInvestment.portfolioConstruction.portfolioConstructionSteps')}>
      {[systemText('preInvestment.portfolioConstruction.selectProduct'), systemText('preInvestment.portfolioConstruction.weightingMethod'), systemText('preInvestment.portfolioConstruction.constraintsAndBacktest'), systemText('preInvestment.portfolioConstruction.runAndSave')].map((label, index) => <li key={label}><button type="button" onClick={() => { setStep(index + 1); document.getElementById(`portfolio-step-${index + 1}`)?.scrollIntoView({ behavior: 'smooth', block: 'start' }) }} className={`w-full rounded-lg border px-3 py-2 text-left ${step === index + 1 ? 'border-emerald-500 bg-emerald-50 font-semibold text-emerald-800' : 'border-slate-200 bg-white text-slate-600'}`}>{index + 1}. {label}</button></li>)}
    </ol>
    {error && <div role="alert" className="rounded-lg border border-rose-300 bg-rose-50 p-3 text-sm text-rose-800">{error}</div>}
    {allocationSource && <aside className="rounded-xl border border-emerald-200 bg-emerald-50 p-4 text-sm text-emerald-950">
      <p className="font-semibold">{systemText('preInvestment.portfolioConstruction.taaAssetClassBudgetReceived')}</p>
      {allocationSource.expires_on && <p className="mt-1 text-xs">{systemText('preInvestment.portfolioConstruction.planValidThrough') + " "}{allocationSource.expires_on}{systemText('preInvestment.portfolioConstruction.validityAndRiskChecksStillApplyAt')}</p>}
      <div className="mt-2 grid gap-2 sm:grid-cols-3">{Object.entries(allocationSource.class_weights).map(([name, weight]) => { const control = numericControls[`class-budget:${name}`]; return <div key={name} className="rounded-lg bg-white p-2"><b>{name}</b> {" " + systemText('preInvestment.portfolioConstruction.budget') + " "}{(weight * 100).toFixed(2)}%<p className={control?.within_tolerance ? 'text-emerald-700' : 'text-amber-800'}>{control ? systemText('preInvestment.portfolioConstruction.allocated', { p0: control.total.toFixed(2), p1: control.within_tolerance ? systemText('preInvestment.portfolioConstruction.withinBudget') : systemText('preInvestment.portfolioConstruction.adjustWithinClassWeights') }) : systemText('preInvestment.portfolioConstruction.checkingBudgets')}</p></div> })}</div>
      <p className="mt-2">{systemText('preInvestment.portfolioConstruction.adjustProductsWithinEachClassWhilePreserving')}</p>
      <Link className="mt-2 inline-block underline" to={`/pre-investment/taa?decision=${encodeURIComponent(allocationSource.decision_id)}`}>{systemText('preInvestment.portfolioConstruction.returnToTaaToViewOrResearch')}</Link>
    </aside>}
    {(draft || inputsChanged) && !result && <p className="text-xs text-slate-600">{systemText('preInvestment.portfolioConstruction.inputsAreRetainedAutomaticallyRerunAfterChanges')}</p>}
    <section id="portfolio-step-1" className="grid gap-6 lg:grid-cols-[1.1fr_.9fr]">
      <div className="space-y-5 rounded-xl border border-slate-200 bg-white p-4 shadow-sm sm:p-6">
        <div><label className="block text-sm font-medium" htmlFor="portfolio-name">{systemText('preInvestment.portfolioConstruction.researchName')}</label><input id="portfolio-name" value={name} onChange={(event) => setName(event.target.value)} className="mt-1 w-full rounded-lg border border-slate-300 px-3 py-2" /></div>
        <div className="space-y-2"><div className="grid gap-3 sm:grid-cols-[1fr_180px]"><label className="block text-sm font-medium" htmlFor="portfolio-search">{systemText('preInvestment.portfolioConstruction.searchProductsInTheInvestableUniverse')}<input id="portfolio-search" disabled={!universe || !assetClasses.length} value={query} onChange={(event) => setQuery(event.target.value)} placeholder={systemText('preInvestment.portfolioConstruction.enterACodeOrName')} className="mt-1 w-full rounded-lg border border-slate-300 px-3 py-2 disabled:bg-slate-100" /></label><label className="block text-sm font-medium">{systemText('preInvestment.portfolioConstruction.addToAssetClass')}<select aria-label={systemText('preInvestment.portfolioConstruction.candidateProductAssetClass')} disabled={!assetClasses.length} value={selectedAssetClassId} onChange={(event) => setSelectedAssetClassId(event.target.value)} className="mt-1 w-full rounded-xl border border-slate-300 bg-white px-3 py-2 disabled:bg-slate-100"><option value="">{systemText('preInvestment.portfolioConstruction.select')}</option>{assetClasses.map((item) => <option key={item.id} value={item.id}>{item.name}</option>)}</select></label></div>
          {!assetClasses.length && <p className="rounded-lg bg-amber-50 px-3 py-2 text-xs text-amber-800">{systemText('preInvestment.portfolioConstruction.firstCreateAssetClassesIn')}<Link to={universe ? `/pre-investment/saa/asset-classes?universe=${encodeURIComponent(universe.id)}` : '/pre-investment/product-pool'} className="mx-1 font-medium underline">{systemText('preInvestment.portfolioConstruction.assetConstruction')}</Link>{systemText('preInvestment.portfolioConstruction.andImportProducts')}</p>}
          <p aria-live="polite" className="text-xs text-slate-600">{searching ? systemText('preInvestment.portfolioConstruction.searchingInvestableUniverseProducts') : searchError || (query && !candidates.length ? systemText('preInvestment.portfolioConstruction.noMatchingProducts') : '')}</p>
          <ul className="divide-y rounded-lg border border-slate-200" aria-label={systemText('preInvestment.portfolioConstruction.productSearchResults')}>{candidates.map((item) => <li key={`${item.kind}-${item.product_id}`} className="flex items-center justify-between gap-3 px-3 py-2 text-sm"><span><b>{item.name}</b> <span className="text-slate-600">{item.code ?? item.product_id} · {item.kind === 'etf' ? 'ETF' : systemText('preInvestment.portfolioConstruction.mutualFund')}</span></span><button type="button" disabled={!selectedAssetClassId} onClick={() => addInstrument(item)} className="rounded-lg border border-emerald-600 px-2 py-1 text-emerald-700 disabled:border-slate-300 disabled:text-slate-600">{systemText('preInvestment.portfolioConstruction.add')}</button></li>)}</ul>
        </div>
        <div><h2 className="text-base font-semibold">{systemText('preInvestment.portfolioConstruction.selectedProductsAtLeast2')}</h2>{!constituents.length ? <p className="mt-2 text-sm text-slate-600">{systemText('preInvestment.portfolioConstruction.searchAndAddProductsToStartConstruction')}</p> : <ul className="mt-2 space-y-2">{constituents.map((item, index) => <li key={`${item.kind}-${item.product_id}`} className="grid grid-cols-[1fr_auto] gap-2 rounded-lg border border-slate-200 p-2"><span className="text-sm"><b>{item.name}</b><br /><span className="text-xs text-slate-600">{item.code ?? item.product_id} · {item.asset_class_name || systemText('preInvestment.portfolioConstruction.noAssetClassAssigned')}</span></span><button type="button" onClick={() => setConstituents((current) => current.filter((_, currentIndex) => currentIndex !== index))} className="text-sm text-rose-700 underline">{systemText('preInvestment.portfolioConstruction.remove')}</button>{method === 'manual' && <label className="col-span-2 text-sm">{systemText('preInvestment.portfolioConstruction.weight')}<input aria-label={systemText('preInvestment.portfolioConstruction.weight2', { p0: item.name })} type="number" min="0" max="100" value={item.weight ?? 0} onChange={(event) => updateNumber(index, 'weight', event.target.value)} className="ml-2 w-24 rounded-lg border border-slate-300 px-2 py-1" /></label>}{method === 'risk_budget' && <label className="col-span-2 text-sm">{systemText('preInvestment.portfolioConstruction.riskBudget2')}<input aria-label={systemText('preInvestment.portfolioConstruction.riskBudget3', { p0: item.name })} type="number" min="0" max="100" value={item.risk_budget ?? 0} onChange={(event) => updateNumber(index, 'risk_budget', event.target.value)} className="ml-2 w-24 rounded-lg border border-slate-300 px-2 py-1" /></label>}</li>)}</ul>}
          {(method === 'manual' || method === 'risk_budget') && (() => { const control = method === 'manual' ? weightControl : budgetControl; return <p className={`mt-2 text-sm ${control?.within_tolerance ? 'text-emerald-700' : 'text-amber-700'}`}>{method === 'manual' ? systemText('preInvestment.portfolioConstruction.weight3') : systemText('preInvestment.portfolioConstruction.riskBudget')}{systemText('preInvestment.portfolioConstruction.total')}{control ? systemText('preInvestment.portfolioConstruction.mustEqual100', { p0: control.total.toFixed(2) }) : numericControlError || systemText('preInvestment.portfolioConstruction.validating')}</p> })()}</div>
      </div>
      <div className="space-y-5 rounded-xl border border-slate-200 bg-white p-4 shadow-sm sm:p-6">
        <fieldset id="portfolio-step-2"><legend className="text-base font-semibold">{systemText('preInvestment.portfolioConstruction.weightingMethod')}</legend><div className="mt-2 grid gap-2">{METHODS.map((item) => <label key={item.value} className={`cursor-pointer rounded-lg border p-3 ${method === item.value ? 'border-emerald-500 bg-emerald-50' : 'border-slate-200'}`}><input type="radio" name="method" disabled={Boolean(allocationSource) && item.value !== 'manual'} checked={method === item.value} onChange={() => setMethod(item.value)} /> <b className="ml-1">{item.label}</b><span className="block pl-5 text-xs text-slate-600">{item.help}</span></label>)}</div></fieldset>
        {method === 'target_optimization' && <div className="space-y-2"><label className="block text-sm">{systemText('preInvestment.portfolioConstruction.optimizationObjective')}<select value={objective} onChange={(event) => setObjective(event.target.value as typeof objective)} className="ml-2 rounded-lg border border-slate-300 p-1"><option value="max_sharpe">{systemText('preInvestment.portfolioConstruction.maximumSharpe')}</option><option value="min_volatility">{systemText('preInvestment.portfolioConstruction.minimumVolatility')}</option><option value="target_return">{systemText('preInvestment.portfolioConstruction.targetReturn')}</option></select></label>{objective === 'target_return' && <label className="block text-sm">{systemText('preInvestment.portfolioConstruction.targetAnnualReturn')}<input type="number" value={targetReturn} onChange={(event) => setTargetReturn(Number(event.target.value))} className="ml-2 w-24 rounded-lg border border-slate-300 p-1" /></label>}</div>}
        {allocationSource && <p className="text-xs text-slate-600">{systemText('preInvestment.portfolioConstruction.aTaaBudgetIsActiveUseManual')}</p>}
        <fieldset id="portfolio-step-3" className="space-y-2"><legend className="text-base font-semibold">{systemText('preInvestment.portfolioConstruction.constraintsWindowsAndRebalancing')}</legend><div className="grid grid-cols-2 gap-3 text-sm"><label>{systemText('preInvestment.portfolioConstruction.minimumWeight')}<input type="number" value={minWeight} onChange={(event) => setMinWeight(Number(event.target.value))} className="mt-1 w-full rounded-lg border border-slate-300 p-2" /></label><label>{systemText('preInvestment.portfolioConstruction.maximumWeight')}<input type="number" value={maxWeight} onChange={(event) => setMaxWeight(Number(event.target.value))} className="mt-1 w-full rounded-lg border border-slate-300 p-2" /></label><label>{systemText('preInvestment.portfolioConstruction.sampleWindow')}<select value={windowMode} onChange={(event) => setWindowMode(event.target.value as 'all' | 'rolling')} className="mt-1 w-full rounded-lg border border-slate-300 p-2"><option value="all">{systemText('preInvestment.portfolioConstruction.fullHistory')}</option><option value="rolling">{systemText('preInvestment.portfolioConstruction.rollingObservations')}</option></select></label><label>{systemText('preInvestment.portfolioConstruction.rebalancing')}<select value={rebalance} onChange={(event) => setRebalance(event.target.value as typeof rebalance)} className="mt-1 w-full rounded-lg border border-slate-300 p-2"><option value="fixed">{systemText('preInvestment.portfolioConstruction.fixedWeights')}</option><option value="weekly">{systemText('preInvestment.portfolioConstruction.weekly')}</option><option value="monthly">{systemText('preInvestment.portfolioConstruction.monthly')}</option><option value="yearly">{systemText('preInvestment.portfolioConstruction.annually')}</option></select></label>{windowMode === 'rolling' && <label>{systemText('preInvestment.portfolioConstruction.observations')}<input type="number" min="2" value={observations} onChange={(event) => setObservations(Number(event.target.value))} className="mt-1 w-full rounded-lg border border-slate-300 p-2" /></label>}</div><p className="text-xs text-slate-600">{systemText('preInvestment.portfolioConstruction.transactionCostsAreFixedAtZeroIn')}</p></fieldset>
        <HistoricalRegimeBacktestSelector
          value={historicalRegime}
          disabled={running}
          onChange={(value) => {
            setHistoricalRegime(value)
            setResult(null)
          }}
        />
        <div><label className="block text-sm font-medium" htmlFor="benchmark">{systemText('preInvestment.portfolioConstruction.benchmarkOptionalSearchByCodeOrName')}</label><select id="benchmark" value={benchmark ? `${benchmark.kind}:${benchmark.product_id}` : ''} onChange={(event) => { const picked = candidates.find((item) => `${item.kind}:${item.product_id}` === event.target.value) ?? null; setBenchmark(picked) }} className="mt-1 w-full rounded-lg border border-slate-300 p-2"><option value="">{systemText('preInvestment.portfolioConstruction.noBenchmark')}</option>{candidates.map((item) => <option key={`${item.kind}:${item.product_id}`} value={`${item.kind}:${item.product_id}`}>{item.name}</option>)}</select></div>
        {!classBudgetsValid && <p role="status" className="text-sm text-amber-800">{systemText('preInvestment.portfolioConstruction.productWeightsInEachAssetClassMust')}</p>}
        <button id="portfolio-step-4" type="button" disabled={!readyToRun || running} onClick={handleRun} className="w-full rounded-lg bg-emerald-700 px-4 py-2 font-medium text-white disabled:cursor-not-allowed disabled:bg-slate-400">{running ? systemText('preInvestment.portfolioConstruction.runningOnRealHistoricalData') : systemText('preInvestment.portfolioConstruction.runPortfolioResearch')}</button>
      </div>
    </section>
    {result && <section className="rounded-xl border border-emerald-200 bg-white p-4 shadow-sm sm:p-6" aria-live="polite"><div className="flex flex-wrap items-start justify-between gap-3"><div><h2 className="text-lg font-semibold">{systemText('preInvestment.portfolioConstruction.runResults')}</h2><p className="text-sm text-slate-600">{systemText('preInvestment.portfolioConstruction.runId')}{result.id}</p></div>{result.target_id ? <Link to={`/post-investment/research-diagnosis?target=${encodeURIComponent(result.target_id)}&run=${encodeURIComponent(result.id)}`} className="rounded-lg bg-slate-900 px-3 py-2 text-sm text-white">{systemText('preInvestment.portfolioConstruction.openHoldingsDiagnostics')}</Link> : <button type="button" onClick={handleSave} disabled={saving} className="rounded-lg bg-slate-900 px-3 py-2 text-sm text-white">{saving ? systemText('preInvestment.portfolioConstruction.saving') : systemText('preInvestment.portfolioConstruction.saveAsResearchTarget')}</button>}</div><div className="mt-4 grid grid-cols-2 gap-3 md:grid-cols-4">{result.metrics.slice(0, 8).map((metric) => <div key={metric.name} className="rounded-lg bg-slate-50 p-3"><p className="text-xs text-slate-500">{metric.name}</p><p className="mt-1 font-semibold"><MetricValue value={metric.value} presentation={resultMetricPresentation(metric)} /></p></div>)}</div>{result.warnings.length > 0 && <ul className="mt-4 list-disc rounded-lg bg-amber-50 px-6 py-3 text-sm text-amber-900">{result.warnings.map((warning) => <li key={warning}>{humanizeIndicatorMessage(warning)}</li>)}</ul>}<div className="mt-5"><RegimeConditioningPanel result={result.regime_conditioning} /></div></section>}
    {result?.id && <PortfolioRiskSection key={result.id} portfolioRunId={result.id} context={systemText('preInvestment.portfolioConstruction.testsEndingHoldingsFromTheSavedRun')} />}
  </div>
}

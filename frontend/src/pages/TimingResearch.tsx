import { systemText, useI18n } from '../i18n/runtime'
import { useEffect, useMemo, useRef, useState } from 'react'
import { Button, EmptyState, ErrorPanel, LoadingPanel } from '../components/ui'
import { useSearchParams } from 'react-router-dom'
import { searchInstruments, type InstrumentSearchItem } from '../services/customIndicators'
import { editableTimingDefinition, timingApi, timingNumber, timingPercent, type SavedTimingDefinition, type TimingCatalog, type TimingComparison, type TimingDefinition, type TimingJob, type TimingRequest, type TimingRun, type TimingRunSummary } from '../services/timingResearch'
import TimingRuleEditor, { timingField } from '../components/timing-research/TimingRuleEditor'
import TimingResults from '../components/timing-research/TimingResults'
import TimingReleaseLibrary from '../components/timing-research/TimingReleaseLibrary'
import TimingTrainingEditor, { timingTrainingError } from '../components/timing-research/TimingTrainingEditor'
import { TimingAdaptationNote, TimingBasketEditor, basketsFromDraft, basketValidation, type BasketDraft } from '../components/timing-research/TimingStudyContext'

const button = 'min-h-10 rounded-xl border border-slate-300 bg-white px-4 py-2 text-sm font-medium text-slate-700 hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-40'
const primary = 'min-h-10 rounded-lg bg-accent-600 px-4 py-2 text-sm font-semibold text-white hover:bg-accent-700 disabled:cursor-not-allowed disabled:opacity-40'
const emptyDefinition = (): TimingDefinition => ({ name: systemText('preInvestment.timingResearch.myTimingAlgorithm'), description: '', nodes: [], entry: '', exit: null, execution: { take_profit: .15, stop_loss: .15, max_holding_bars: 15, cooldown_bars: 0, fee_bps: 3, slippage_bps: 2 } })
type Mode = 'library' | 'research' | 'application'
type Product = { product_id: string; name: string }
const errorText = (error: unknown) => error instanceof Error ? error.message : systemText('preInvestment.timingResearch.theOperationDidNotCompleteRetryLater')
const currentDefinition = (definition: TimingDefinition) => editableTimingDefinition(definition)
function delay(milliseconds: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    const aborted = () => { window.clearTimeout(timer); reject(new DOMException('Aborted', 'AbortError')) }
    const timer = window.setTimeout(() => { signal.removeEventListener('abort', aborted); resolve() }, milliseconds)
    if (signal.aborted) aborted(); else signal.addEventListener('abort', aborted, { once: true })
  })
}

export default function TimingResearch({ mode = 'research' }: { mode?: Mode }) {
  useI18n()
  const [params] = useSearchParams()
  const [catalog, setCatalog] = useState<TimingCatalog | null>(null)
  const [definitions, setDefinitions] = useState<SavedTimingDefinition[]>([])
  const [definition, setDefinition] = useState<TimingDefinition>(emptyDefinition)
  const [saved, setSaved] = useState<SavedTimingDefinition | undefined>()
  const [templateId, setTemplateId] = useState('')
  const [targets, setTargets] = useState<Product[]>(() => (params.get('product_id') || params.get('ids') || '510300.SH').split(',').filter(Boolean).slice(0, 12).map(product_id => ({ product_id, name: product_id })))
  const [search, setSearch] = useState('')
  const [products, setProducts] = useState<InstrumentSearchItem[]>([])
  const [searchError, setSearchError] = useState('')
  const [startDate, setStartDate] = useState('2020-01-01')
  const [endDate, setEndDate] = useState(params.get('as_of') || '2026-09-03')
  const [holdout, setHoldout] = useState('2024-01-01')
  const [priceBasis, setPriceBasis] = useState<TimingRequest['price_basis']>('hfq')
  const [basketDraft, setBasketDraft] = useState<BasketDraft>({ market: '', category: '' })
  const [run, setRun] = useState<TimingRun | null>(null)
  const [runKey, setRunKey] = useState('')
  const [job, setJob] = useState<TimingJob | null>(null)
  const [busy, setBusy] = useState<string | null>(null)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [loadAttempt, setLoadAttempt] = useState(0)
  const [tab, setTab] = useState<'editor' | 'results' | 'history' | 'releases'>(mode === 'application' ? 'releases' : 'editor')
  const [settingsOpen, setSettingsOpen] = useState(true)
  const [history, setHistory] = useState<TimingRunSummary[]>([])
  const [comparisonIds, setComparisonIds] = useState<string[]>([])
  const [comparison, setComparison] = useState<TimingComparison | null>(null)
  const [note, setNote] = useState('')
  const activeOperation = useRef<AbortController | null>(null)
  const mounted = useRef(true)
  const version = useRef(0)
  const study = useMemo(() => ({ definition, targets: targets.map(target => ({ kind: 'etf' as const, product_id: target.product_id })), start_date: startDate, end_date: endDate, holdout_start: holdout, walk_forward_splits: 3, price_basis: priceBasis, context_baskets: basketsFromDraft(basketDraft) }), [definition, targets, startDate, endDate, holdout, priceBasis, basketDraft])
  const studyKey = JSON.stringify(study)
  const stale = !!run && runKey !== studyKey
  const unsupportedKind = !!params.get('kind') && params.get('kind') !== 'etf'
  const trainingError = catalog ? timingTrainingError(definition, catalog) : ''
  const validationError = unsupportedKind ? systemText('preInvestment.timingResearch.currentlySupportsExchangeTradedEtfsOtcFunds') : !targets.length ? systemText('preInvestment.timingResearch.addAnEtfFirst') : !startDate || !endDate || startDate >= endDate ? systemText('preInvestment.timingResearch.theResearchEndDateMustBeLater') : !holdout || holdout <= startDate || holdout > endDate ? systemText('preInvestment.timingResearch.outOfSampleStartMustBeAfter') : !definition.name.trim() ? systemText('preInvestment.timingResearch.enterAnAlgorithmName') : !definition.nodes.length || !definition.entry ? systemText('preInvestment.timingResearch.addCalculationStepsAndSelectABuy') : trainingError || basketValidation(definition, basketDraft)
  const title = ({ library: systemText('preInvestment.timingResearch.timingAlgorithmCenter'), research: systemText('preInvestment.timingResearch.productTimingResearch'), application: systemText('preInvestment.timingResearch.productAllocationAndTiming') })[mode]
  const change = (next: TimingDefinition) => { version.current += 1; setDefinition(next); setNotice(''); setError('') }
  useEffect(() => {
    mounted.current = true
    return () => { mounted.current = false; activeOperation.current?.abort() }
  }, [])
  useEffect(() => {
    const controller = new AbortController()
    setError('')
    void Promise.all([timingApi.catalog(controller.signal), timingApi.definitions(controller.signal), timingApi.runs(controller.signal)]).then(([nextCatalog, nextDefinitions, nextHistory]) => {
      if (controller.signal.aborted) return
      setCatalog(nextCatalog); setDefinitions(nextDefinitions.items); setHistory(nextHistory.items)
      if (nextCatalog.templates[0]) { setDefinition(currentDefinition(nextCatalog.templates[0].definition)); setTemplateId(nextCatalog.templates[0].id) }
    }).catch(reason => { if (!controller.signal.aborted) setError(errorText(reason)) })
    return () => controller.abort()
  }, [loadAttempt])
  useEffect(() => {
    const controller = new AbortController()
    setProducts([]); setSearchError('')
    if (!search.trim()) return () => controller.abort()
    const timer = window.setTimeout(() => {
      void searchInstruments({ kind: 'etf', query: search, pageSize: 8, signal: controller.signal }).then(response => {
        if (!controller.signal.aborted) setProducts(response.items)
      }).catch(reason => { if (!controller.signal.aborted) setSearchError(errorText(reason)) })
    }, 250)
    return () => { window.clearTimeout(timer); controller.abort() }
  }, [search])
  const addTarget = (productId: string, name = productId) => {
    if (targets.length >= 12 || targets.some(target => target.product_id === productId)) return
    version.current += 1; setTargets(previous => [...previous, { product_id: productId, name }]); setSearch('')
  }
  const applyTemplate = (id: string) => {
    const template = catalog?.templates.find(item => item.id === id)
    if (template) { change(currentDefinition(template.definition)); setTemplateId(id); setSaved(undefined); setTab('editor') }
  }
  const save = async () => {
    setBusy('save'); setError(''); setNotice('')
    const captured = definition, capturedVersion = version.current
    try {
      const next = await timingApi.save(captured, saved)
      if (!mounted.current) return
      setDefinitions(previous => [next, ...previous.filter(item => item.id !== next.id)])
      if (version.current === capturedVersion) setSaved(next)
      setNotice(systemText('preInvestment.timingResearch.algorithmSavedFurtherEditsCreateNewVersions'))
    } catch (reason) { if (mounted.current) setError(errorText(reason)) }
    finally { if (mounted.current) setBusy(null) }
  }
  const execute = async () => {
    if (validationError) { setError(validationError); return }
    activeOperation.current?.abort()
    const controller = new AbortController(); activeOperation.current = controller
    const snapshot = study, snapshotKey = studyKey
    setBusy('run'); setJob(null); setError(''); setNotice('')
    try {
      const prepared = await timingApi.prepare(snapshot.definition, controller.signal)
      let next = await timingApi.run({ ...snapshot, compile_token: prepared.compile_token }, controller.signal)
      if (controller.signal.aborted) return
      setJob(next)
      while (next.status === 'queued' || next.status === 'running') {
        await delay(1000, controller.signal)
        next = await timingApi.job(next.id, controller.signal)
        if (controller.signal.aborted) return
        setJob(next)
      }
      if (next.status === 'failed') throw new Error(next.error || systemText('preInvestment.timingResearch.researchCalculationDidNotCompleteCheckData'))
      if (!next.run_id) throw new Error(systemText('preInvestment.timingResearch.theJobReturnedNoResearchResultsRefresh'))
      const result = await timingApi.getRun(next.run_id, controller.signal)
      if (controller.signal.aborted) return
      setRun(result); setRunKey(snapshotKey); setTab('results'); setJob(null); setSettingsOpen(false)
      setHistory(previous => [{ id: result.id, name: result.name, created_at: result.created_at }, ...previous.filter(item => item.id !== result.id)])
    } catch (reason) { if (!controller.signal.aborted) setError(errorText(reason)) }
    finally { if (!controller.signal.aborted && mounted.current) { setBusy(null); setJob(null) } }
  }
  const loadRun = async (id: string) => {
    activeOperation.current?.abort()
    const controller = new AbortController(); activeOperation.current = controller
    const capturedVersion = version.current
    setBusy('load'); setError('')
    try {
      const result = await timingApi.getRun(id, controller.signal)
      if (controller.signal.aborted) return
      if (capturedVersion !== version.current) { setNotice(systemText('preInvestment.timingResearch.theDraftChangedWhileLoadingSelectThe')); return }
      const request = result.request_snapshot
      const next = currentDefinition(result.definition_snapshot)
      setDefinition(next); setSaved(undefined); setTemplateId(''); setTargets(request.targets.map(target => ({ product_id: target.product_id, name: target.product_id })))
      setStartDate(request.start_date); setEndDate(request.end_date); setHoldout(request.holdout_start); setPriceBasis(request.price_basis)
      const baskets = { market: request.context_baskets?.market || [], category: request.context_baskets?.category || [] }
      setBasketDraft({ market: baskets.market.join(', '), category: baskets.category.join(', ') })
      setRun(result); setRunKey(JSON.stringify({ definition: next, targets: request.targets, start_date: request.start_date, end_date: request.end_date, holdout_start: request.holdout_start, walk_forward_splits: 3, price_basis: request.price_basis, context_baskets: baskets })); setTab('results'); setSettingsOpen(false)
    } catch (reason) { if (!controller.signal.aborted) setError(errorText(reason)) }
    finally { if (!controller.signal.aborted && mounted.current) setBusy(null) }
  }
  const compare = async () => {
    setBusy('compare'); setError('')
    try { const result = await timingApi.compare(comparisonIds); if (mounted.current) setComparison(result) }
    catch (reason) { if (mounted.current) setError(errorText(reason)) }
    finally { if (mounted.current) setBusy(null) }
  }
  const publish = async (bind = false) => {
    if (!run || stale) return
    setBusy('release'); setError(''); setNotice('')
    try {
      const release = await timingApi.release(run.id, note)
      if (bind) await timingApi.bind(release.id, 'pre_investment', note)
      if (mounted.current) setNotice(bind ? systemText('preInvestment.timingResearch.referencedInPreInvestmentResearchWithoutChanging') : systemText('preInvestment.timingResearch.researchVersionSavedThisFixedResultCan'))
    } catch (reason) { if (mounted.current) setError(errorText(reason)) }
    finally { if (mounted.current) setBusy(null) }
  }
  return <main className="mx-auto w-full min-w-0 max-w-[1600px] space-y-5 p-3 sm:p-5 lg:p-6">
    <header className="flex flex-wrap items-start justify-between gap-4"><div className="min-w-0"><h1 className="text-2xl font-semibold tracking-tight text-slate-950">{title}</h1><p className="mt-2 max-w-3xl text-sm leading-6 text-slate-600">{systemText('preInvestment.timingResearch.selectEtfsCombineCalculationsWithTradingRules')}</p>{mode === 'application' && <p className="mt-1 text-xs leading-5 text-slate-600">{systemText('preInvestment.timingResearch.referencingResearchDoesNotTradeOrChange')}</p>}</div><span className="rounded-full bg-accent-50 px-3 py-1 text-xs font-medium text-accent-700">{systemText('preInvestment.timingResearch.dailyEtfsResearchUse')}</span></header>
    {/* 目录没读出来时整页没有研究内容，换成公共错误态；运行失败这类操作错误旁边已有研究内容，仍是纯文字。 */}
    {error && catalog && <div role="alert" className="flex flex-wrap items-center justify-between gap-2 rounded-xl border border-rose-200 bg-rose-50 p-3 text-sm text-rose-700"><p>{error}</p></div>}
    {error && !catalog && <ErrorPanel onRetry={() => setLoadAttempt(value => value + 1)} />}
    {notice && <p role="status" className="rounded-xl bg-emerald-50 p-3 text-sm text-emerald-800">{notice}</p>}
    {!catalog && !error && <LoadingPanel text={systemText('preInvestment.timingResearch.loadingAlgorithmCatalog')} />}
    {catalog && tab !== 'releases' && <button type="button" aria-expanded={settingsOpen} aria-controls="timing-study-settings" className="flex w-full items-center justify-between gap-3 rounded-xl border border-slate-200 bg-white p-3 text-left lg:hidden" onClick={() => setSettingsOpen(value => !value)}><span className="min-w-0"><strong className="block text-sm text-slate-800">{systemText('preInvestment.timingResearch.researchSettings') + " "}{targets.length} {" " + systemText('preInvestment.timingResearch.etfs')}</strong><span className="text-xs text-slate-600">{startDate} {" " + systemText('preInvestment.timingResearch.to') + " "}{endDate}</span></span><span className="shrink-0 text-xs text-accent-700">{settingsOpen ? systemText('preInvestment.timingResearch.collapse') : systemText('preInvestment.timingResearch.expandToEdit')}</span></button>}
    {catalog && <div className={`grid min-w-0 items-start gap-5 ${tab === 'releases' ? '' : 'lg:grid-cols-[280px_minmax(0,1fr)]'}`}>
      {tab !== 'releases' && <aside id="timing-study-settings" className={`${settingsOpen ? 'block' : 'hidden'} min-w-0 space-y-5 rounded-xl border border-slate-200 bg-white p-4 lg:block`}>
        <section className="space-y-3"><h2 className="text-sm font-semibold text-slate-900">{systemText('preInvestment.timingResearch.1SelectResearchTargets')}</h2><div className="flex flex-wrap gap-2">{targets.map(target => <span key={target.product_id} className="inline-flex max-w-full items-center gap-1 rounded-lg bg-slate-100 py-1 pl-2 text-xs text-slate-700"><span className="truncate" title={target.name}>{target.name}</span><button type="button" aria-label={systemText('preInvestment.timingResearch.remove', { p0: target.product_id })} className="min-h-8 min-w-8 rounded-lg hover:bg-slate-200" onClick={() => { version.current += 1; setTargets(targets.filter(item => item.product_id !== target.product_id)) }}>×</button></span>)}</div>
          <label className="block text-xs text-slate-600">{systemText('preInvestment.timingResearch.searchEtfNameOrCode')}<input className={timingField} value={search} onChange={event => setSearch(event.target.value)} placeholder={systemText('preInvestment.timingResearch.forExample510300')} disabled={targets.length >= 12} /></label>
          {!!products.length && <ul aria-label={systemText('preInvestment.timingResearch.etfSearchResults')} className="max-h-56 overflow-auto rounded-lg border border-slate-200">{products.map(item => { const code = item.code || item.ts_code || ''; return <li key={code}><button type="button" className="min-h-11 w-full px-3 py-2 text-left text-xs hover:bg-accent-50 disabled:opacity-40" disabled={!code || targets.some(target => target.product_id === code)} onClick={() => addTarget(code, item.name || code)}>{item.name || code}<span className="ml-2 text-slate-600">{code}</span></button></li> })}</ul>}
          {searchError && <p className="text-xs text-amber-700">{searchError}</p>}
          {/^[0-9]{6}\.(SH|SZ)$/i.test(search.trim()) && !targets.some(target => target.product_id === search.trim().toUpperCase()) && <button type="button" className="min-h-9 text-xs text-accent-700" onClick={() => addTarget(search.trim().toUpperCase())}>{systemText('preInvestment.timingResearch.addCode') + " "}{search.trim().toUpperCase()}</button>}
          <p className="text-xs text-slate-600">{systemText('preInvestment.timingResearch.upTo12EtfsTestedSeparatelyRather')}</p>
          <label className="block text-xs text-slate-600">{systemText('preInvestment.timingResearch.researchStartDate')}<input type="date" className={timingField} value={startDate} onChange={event => { version.current += 1; setStartDate(event.target.value) }} /></label><label className="block text-xs text-slate-600">{systemText('preInvestment.timingResearch.researchEndDate')}<input type="date" className={timingField} value={endDate} onChange={event => { version.current += 1; setEndDate(event.target.value) }} /></label><label className="block text-xs text-slate-600">{systemText('preInvestment.timingResearch.outOfSampleStartDate')}<input type="date" className={timingField} value={holdout} onChange={event => { version.current += 1; setHoldout(event.target.value) }} /></label><p className="text-xs leading-5 text-slate-600">{systemText('preInvestment.timingResearch.buildRulesOnTheEarlierPeriodAnd')}</p>
        </section>
        <section className="space-y-3 border-t border-slate-100 pt-4"><h2 className="text-sm font-semibold text-slate-900">{systemText('preInvestment.timingResearch.2ChooseAStartingPoint')}</h2><label className="block text-xs text-slate-600">{systemText('preInvestment.timingResearch.builtInTemplates')}<select className={timingField} value={templateId} onChange={event => applyTemplate(event.target.value)}><option value="">{systemText('preInvestment.timingResearch.customAlgorithm')}</option>{catalog.templates.map(template => <option key={template.id} value={template.id}>{template.label}</option>)}</select></label>{templateId && <p className="text-xs leading-5 text-slate-600">{catalog.templates.find(template => template.id === templateId)?.description}</p>}
          <label className="block text-xs text-slate-600">{systemText('preInvestment.timingResearch.savedAlgorithms')}<select className={timingField} value={saved?.id || ''} onChange={event => { const item = definitions.find(value => value.id === event.target.value); if (item) { change(currentDefinition(item)); setSaved(item); setTemplateId(''); setTab('editor') } }}><option value="">{systemText('preInvestment.timingResearch.selectASavedAlgorithm')}</option>{definitions.map(item => <option key={item.id} value={item.id}>{item.name} · v{item.revision}</option>)}</select></label><button type="button" className="min-h-9 text-xs font-medium text-accent-700" onClick={() => { change(emptyDefinition()); setSaved(undefined); setTemplateId(''); setTab('editor') }}>{systemText('preInvestment.timingResearch.startFromBlank')}</button>
        </section>
        <TimingBasketEditor value={basketDraft} definition={definition} onChange={next => { version.current += 1; setBasketDraft(next) }} />
        <details className="border-t border-slate-100 pt-3"><summary className="cursor-pointer py-1 text-sm font-medium text-slate-700">{systemText('preInvestment.timingResearch.tradingAndPriceSettings')}</summary><p className="mt-2 text-xs leading-5 text-slate-600">{systemText('preInvestment.timingResearch.executeAtTheNextTradingDayS')}</p><div className="mt-3 space-y-3"><label className="block text-xs text-slate-600">{systemText('preInvestment.timingResearch.priceConvention')}<select className={timingField} value={priceBasis} onChange={event => { version.current += 1; setPriceBasis(event.target.value as TimingRequest['price_basis']) }}><option value="hfq">{systemText('preInvestment.timingResearch.backwardAdjustedResearchReturns')}</option><option value="qfq">{systemText('preInvestment.timingResearch.forwardAdjustedCutoffAnchored')}</option><option value="raw">{systemText('preInvestment.timingResearch.unadjustedTradingPrices')}</option></select></label>{([
          ['take_profit', systemText('preInvestment.timingResearch.takeProfit'), 100], ['stop_loss', systemText('preInvestment.timingResearch.stopLoss'), 100], ['max_holding_bars', systemText('preInvestment.timingResearch.maximumHoldingTradingDays'), 1], ['cooldown_bars', systemText('preInvestment.timingResearch.cooldownTradingDaysAfterExit'), 1], ['fee_bps', systemText('preInvestment.timingResearch.oneWayFeeBasisPoints'), 1], ['slippage_bps', systemText('preInvestment.timingResearch.oneWaySlippageBasisPoints'), 1],
        ] as const).map(([key, label, scale]) => <label key={key} className="block text-xs text-slate-600">{label}<input type="number" className={timingField} min={key === 'max_holding_bars' ? 1 : 0} step={key.includes('bars') ? 1 : 'any'} value={definition.execution[key] * scale} onChange={event => change({ ...definition, execution: { ...definition.execution, [key]: Number(event.target.value) / scale } })} /></label>)}<p className="text-xs text-slate-600">{systemText('preInvestment.timingResearch.1BasisPoint001FeesAnd')}</p></div></details>
      </aside>}
      <div className="min-w-0 space-y-4">
        <div className="min-w-0 rounded-xl border border-slate-200 bg-white p-4 sm:p-5">
          {tab !== 'releases' && <div className="grid grid-cols-2 items-end gap-3 sm:flex sm:flex-wrap"><label className="col-span-2 min-w-0 flex-1 text-xs text-slate-600">{systemText('preInvestment.timingResearch.algorithmName')}<input className={timingField} value={definition.name} maxLength={100} onChange={event => change({ ...definition, name: event.target.value })} /></label><button type="button" className={button} disabled={!!busy || !definition.nodes.length} onClick={save}>{busy === 'save' ? systemText('preInvestment.timingResearch.saving') : systemText('preInvestment.timingResearch.saveAlgorithm')}</button><button type="button" className={primary} disabled={!!busy || !!validationError} onClick={execute}>{busy === 'run' ? job ? systemText('preInvestment.timingResearch.researching', { p0: job.progress, p1: job.total }) : systemText('preInvestment.timingResearch.preparingCalculation') : systemText('preInvestment.timingResearch.runResearch')}</button></div>}
          {tab !== 'releases' && !!validationError && <p className="mt-2 text-xs text-amber-700">{validationError}</p>}
          {busy === 'run' && <div role="status" className="mt-4 rounded-lg bg-accent-50 p-3 text-sm text-accent-700">{job ? systemText('preInvestment.timingResearch.researchingProduct', { p0: Math.min(job.progress + 1, job.total), p1: job.total }) : systemText('preInvestment.timingResearch.validatingCalculationStepsAndPreparingExecution')}<p className="mt-1 text-xs">{systemText('preInvestment.timingResearch.youCanKeepEditingTheDraftThis')}</p></div>}
          <div role="tablist" aria-label={systemText('preInvestment.timingResearch.timingWorkspace')} className="mt-5 flex flex-wrap gap-4 border-b border-slate-200">{(mode === 'application' ? ['releases', 'editor', 'results', 'history'] as const : ['editor', 'results', 'history', 'releases'] as const).map(value => <button type="button" key={value} role="tab" aria-selected={tab === value} aria-controls="timing-workspace-panel" onClick={() => setTab(value)} className={`min-h-11 border-b-2 text-sm ${tab === value ? 'border-accent-600 font-semibold text-accent-700' : 'border-transparent text-slate-600'}`}>{({ editor: systemText('preInvestment.timingResearch.3AlgorithmSteps'), results: systemText('preInvestment.timingResearch.4ResearchResults'), history: systemText('preInvestment.timingResearch.historyAndComparison'), releases: systemText('preInvestment.timingResearch.existingResearchVersions') })[value]}</button>)}</div>
          <div id="timing-workspace-panel" role="tabpanel" className="mt-5 min-w-0">
            {tab === 'releases' && <TimingReleaseLibrary onView={loadRun} disabled={!!busy} />}
            {tab === 'editor' && <div className="space-y-4"><TimingAdaptationNote adaptation={definition.adaptation} /><details><summary className="cursor-pointer text-xs text-slate-600">{systemText('preInvestment.timingResearch.algorithmDescription')}</summary><textarea className={timingField} rows={2} maxLength={2000} value={definition.description} onChange={event => change({ ...definition, description: event.target.value })} placeholder={systemText('preInvestment.timingResearch.recordResearchAssumptionsAndScope')} /></details><TimingTrainingEditor definition={definition} catalog={catalog} onChange={change} /><TimingRuleEditor definition={definition} catalog={catalog} onChange={change} /></div>}
            {tab === 'results' && (run ? <>{stale && <p role="status" className="mb-4 rounded-lg border border-amber-200 bg-amber-50 p-3 text-sm text-amber-800">{systemText('preInvestment.timingResearch.configurationChangedResultsBelowBelongToThe')}</p>}<TimingResults key={run.id} run={run} /><div className="mt-5 space-y-3 border-t border-slate-200 pt-4"><label className="block text-xs text-slate-600">{systemText('preInvestment.timingResearch.researchReferenceNotes')}<input className={timingField} value={note} onChange={event => setNote(event.target.value)} placeholder={systemText('preInvestment.timingResearch.forExampleResearchComparisonForBroadMarket')} /></label><div className="flex flex-wrap gap-2"><button type="button" className={button} disabled={!!busy || stale} onClick={() => publish()}>{systemText('preInvestment.timingResearch.saveResearchVersion')}</button><button type="button" className={button} disabled={!!busy || stale} onClick={() => publish(true)}>{systemText('preInvestment.timingResearch.referenceInPreInvestmentResearch')}</button></div><p className="text-xs text-slate-600">{systemText('preInvestment.timingResearch.researchReferencesLockTheResultAndScope')}</p></div></> : <EmptyState title={systemText('preInvestment.timingResearch.noResearchResultsYet')} hint={systemText('preInvestment.timingResearch.checkProductsAndDatesOnTheLeft')} />)}
            {tab === 'history' && <div className="space-y-4"><div className="flex flex-wrap items-center justify-between gap-3"><p className="text-sm text-slate-600">{systemText('preInvestment.timingResearch.selectAHistoricalResultToViewOr')}</p><button type="button" className={button} disabled={!!busy || comparisonIds.length < 2} onClick={compare}>{systemText('preInvestment.timingResearch.compareSelectedStudies')}</button></div>{!history.length && <p className="rounded-xl bg-slate-50 p-6 text-sm text-slate-600">{systemText('preInvestment.timingResearch.researchRunsWillBeSavedHereIn')}</p>}<div className="space-y-2">{history.map(item => <div key={item.id} className="flex items-center gap-3 rounded-xl border border-slate-200 p-3"><input type="checkbox" aria-label={systemText('preInvestment.timingResearch.compare', { p0: item.name, p1: item.id })} className="h-4 w-4 shrink-0" checked={comparisonIds.includes(item.id)} disabled={!comparisonIds.includes(item.id) && comparisonIds.length >= 4} onChange={event => setComparisonIds(previous => event.target.checked ? [...previous, item.id] : previous.filter(id => id !== item.id))} /><button type="button" className="min-w-0 flex-1 text-left" disabled={!!busy} onClick={() => loadRun(item.id)}><strong className="block truncate text-sm text-slate-800">{item.name}</strong><span className="text-xs text-slate-600">{item.created_at?.replace('T', ' ').slice(0, 19)}</span></button><button type="button" className="min-h-9 shrink-0 px-2 text-xs text-accent-700" disabled={!!busy} onClick={() => loadRun(item.id)}>{systemText('preInvestment.timingResearch.view')}</button></div>)}</div>{comparison && <div className="overflow-auto rounded-lg border border-slate-200"><table aria-label={systemText('preInvestment.timingResearch.researchComparison')} className="w-full text-left text-xs [&_td]:whitespace-nowrap [&_td]:border-t [&_td]:p-3 [&_th]:whitespace-nowrap [&_th]:bg-slate-50 [&_th]:p-3"><thead><tr><th scope="col">{systemText('preInvestment.timingResearch.study')}</th><th scope="col">{systemText('preInvestment.timingResearch.products')}</th><th scope="col">{systemText('preInvestment.timingResearch.outOfSampleReturn')}</th><th scope="col">{systemText('preInvestment.timingResearch.maximumDrawdown')}</th><th scope="col">{systemText('preInvestment.timingResearch.closedTrades')}</th></tr></thead><tbody>{comparison.items.flatMap(item => item.products.map(product => <tr key={`${item.run_id}:${product.product_id}`}><td>{item.name}</td><td>{product.product_id}</td><td>{timingPercent(product.summary?.out_of_sample.total_return)}</td><td>{timingPercent(product.summary?.out_of_sample.max_drawdown)}</td><td>{timingNumber(product.summary?.out_of_sample.trade_count, 0)}</td></tr>))}</tbody></table></div>}</div>}
          </div>
        </div>
      </div>
    </div>}
  </main>
}

import { researchMessage } from '../i18n/researchMessages'
import { TaaPolicySignals, scheduledPolicy, policySignalIssue } from '../components/tactical-allocation/TaaPolicySignals'
import { useEffect, useRef, useState } from 'react'
import { Link, useNavigate, useSearchParams } from 'react-router-dom'
import BaselineSetup from '../components/tactical-allocation/BaselineSetup'
import { TaaPerformance } from '../components/tactical-allocation/TaaResults'
import TaaWalkForward from '../components/tactical-allocation/TaaWalkForward'
import TaaResearchContext from '../components/tactical-allocation/TaaResearchContext'
import TaaScenarioExperiments from '../components/tactical-allocation/TaaScenarioExperiments'
import { useResearchDay } from '../app/ResearchContext'
import { Button, ErrorPanel, LoadingPanel } from '../components/ui'
import { useI18n, systemText } from '../i18n/runtime'
import { allocationJourneyPath, readAllocationDraft, updateAllocationJourney, writeAllocationDraft } from '../app/allocationJourney'
import { buttonClass, Empty, Feedback, Field, inputClass, NumberInput, percentText, primaryClass, sectionClass, today } from '../components/risk-models/ResearchUI'
import { getHistoricalRegimeRun, listHistoricalRegimeRuns, type HistoricalRegimeRun } from '../services/historicalRegimes'
import { isRegimeRunEligibleForTaa } from '../services/regimePublicationEligibility'
import { getTaaBaseline, getTaaCatalog, getTaaDecision, previewTaa, preflightTaa, saveTaaDecision, taaProductAllocation, type TaaBaseline, type TaaCatalog, type TaaDecision, type TaaPreview, type TaaPreviewRequest, type TaaScenario, type TaaScenarioExperiment, type TaaPreflight } from '../services/tacticalAllocation'
import { UpstreamLink, UsabilityNote, upstreamOf } from '../components/versioning'

type Tab = 'views' | 'backtest' | 'scenarios' | 'versions'

const emptyCatalog: TaaCatalog = { allocations: [], baselines: [], decisions: [] }
const zeroes = (baseline: TaaBaseline) => Object.fromEntries(baseline.assets.map(asset => [asset.id, 0]))
const points = (value: number | undefined) => value == null || !Number.isFinite(value) ? '—' : systemText('preInvestment.tacticalAllocationWorkspace.percentagePoints', { p0: value > 0 ? '+' : '', p1: (value * 100).toFixed(2) })
const fail = (error: unknown, message: string) => error instanceof Error ? error.message : message
const eligible = (run: HistoricalRegimeRun) => run.schema_version === '2.0' && isRegimeRunEligibleForTaa(run)

function baselineJourney(value: TaaBaseline) {
  return {
    mandateId: value.policy?.mandate_id,
    strategicUniverseId: value.strategic_universe_id ?? undefined,
    implementationMappingId: value.implementation_mapping_id ?? undefined,
    universeId: value.universe_snapshot_id ?? undefined,
    allocationName: value.alloc_name ?? undefined, baselineId: value.id,
  }
}

function validDraftRequest(value: unknown, baselineId: string): value is TaaPreviewRequest {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return false
  const row = value as Record<string, unknown>
  const record = (item: unknown) => Boolean(item) && typeof item === 'object' && !Array.isArray(item)
  return row.baseline_id === baselineId && ['start_date', 'end_date', 'as_of', 'train_end_date', 'note'].every(key => typeof row[key] === 'string')
    && ['momentum', 'manual', 'regime', 'composite'].includes(String(row.signal_mode)) && typeof row.search === 'boolean'
    && record(row.manual_tilts) && (row.current_weights == null || record(row.current_weights))
    && (row.state_tilts == null || (record(row.state_tilts) && Object.values(row.state_tilts).every(record)))
}

function initialRequest(baseline: TaaBaseline, catalog: TaaCatalog | null, platformDay: string | null = null): TaaPreviewRequest {
  const coverage = catalog?.allocations.find(item => item.alloc_name === baseline.alloc_name)?.coverage
  // Decision/knowledge time is not the same thing as the last market observation.
  // Weekends, holidays and T+1 NAV publication may legitimately leave market data
  // one or more days behind an otherwise valid policy decision date.
  const asOf = [platformDay || undefined, today()].filter(Boolean).sort()[0]!
  const end = [coverage?.end_date, asOf, coverage?.end_date ? undefined : baseline.as_of.slice(0, 10)].filter(Boolean).sort()[0]!
  const start = coverage?.start_date ?? `${Number(end.slice(0, 4)) - 3}${end.slice(4)}`
  const split = new Date(Date.parse(start) + (Date.parse(end) - Date.parse(start)) * .67).toISOString().slice(0, 10)
  return { baseline_id: baseline.id, start_date: start, end_date: end, as_of: asOf, train_end_date: split, signal_mode: 'momentum', decision_policy: { ...scheduledPolicy }, lookback: 60, manual_tilts: zeroes(baseline), max_abs_tilt: .1, transaction_cost_bps: 10, risk_penalty: 3, max_tracking_error: baseline.policy?.mandate.max_tracking_error ?? .1, max_turnover: 1, confidence_floor: .6, max_signal_age_days: 31, search: true, objective: 'active_utility', review_days: 30, note: '' }
}

export default function TacticalAllocationWorkspace() {
  const tabs: Array<[Tab, string]> = [['views', systemText('preInvestment.tacticalAllocationWorkspace.viewsAndRules')], ['backtest', systemText('preInvestment.tacticalAllocationWorkspace.backtestAndSelection')], ['scenarios', systemText('preInvestment.tacticalAllocationWorkspace.scenarioSimulation')], ['versions', systemText('preInvestment.tacticalAllocationWorkspace.versionsAndAudit')]]
  const { s } = useI18n()
  const navigate = useNavigate()
  const platformDay = useResearchDay()
  const platformRef = useRef(platformDay); platformRef.current = platformDay
  const previousPlatform = useRef(platformDay)
  const [params, setParams] = useSearchParams()
  const [catalog, setCatalog] = useState<TaaCatalog | null>(null)
  // 地址栏是唯一事实来源：裸路径进来不自动套用上次的决策或基准，续接由续接条显式发起。
  const decisionParam = params.get('decision') ?? params.get('run') ?? undefined
  const [baselineId, setBaselineId] = useState(decisionParam ? '' : params.get('baseline') ?? '')
  const [baseline, setBaseline] = useState<TaaBaseline | null>(null)
  const [request, setRequest] = useState<TaaPreviewRequest | null>(null)
  const [tab, setTab] = useState<Tab>('views')
  const [loading, setLoading] = useState(true)
  const [loadingBaseline, setLoadingBaseline] = useState(false)
  // 目录或基准读取失败后要能重来一次：两个读取都跟着这个计数重跑。
  const [reload, setReload] = useState(0)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [preview, setPreview] = useState<TaaPreview | null>(null)
  const [decision, setDecision] = useState<TaaDecision | null>(null)
  const [decisionName, setDecisionName] = useState('')
  const [saving, setSaving] = useState(false)
  const [runs, setRuns] = useState<HistoricalRegimeRun[]>([])
  const [regime, setRegime] = useState<HistoricalRegimeRun | null>(null)
  const [regimeStates, setRegimeStates] = useState<HistoricalRegimeRun['states']>([])
  const [loadingRegime, setLoadingRegime] = useState(false)
  const [regimeError, setRegimeError] = useState('')
  const [experiments, setExperiments] = useState<TaaScenarioExperiment[]>([])
  const [decisionNote, setDecisionNote] = useState('')
  const [preflight, setPreflight] = useState<TaaPreflight | null>(null)
  const [checking, setChecking] = useState(false)
  const [checkError, setCheckError] = useState('')
  const [checkRevision, setCheckRevision] = useState(0)
  const version = useRef(0)
  const mounted = useRef(true)
  const openingDecision = useRef<TaaDecision | null>(null)
  const restore = useRef<TaaPreviewRequest | null>(null)
  const catalogRef = useRef(catalog); catalogRef.current = catalog
  const fingerprint = JSON.stringify(request)
  const latest = useRef(fingerprint); latest.current = fingerprint
  const saveFingerprint = JSON.stringify({ preview: preview?.preview_hash, name: decisionName, note: decisionNote, scenarios: experiments.filter(item => item.result).map(item => item.scenario) })
  const latestSave = useRef(saveFingerprint); latestSave.current = saveFingerprint

  useEffect(() => { mounted.current = true; return () => { mounted.current = false; version.current += 1 } }, [])
  useEffect(() => {
    const controller = new AbortController(); setLoading(true)
    getTaaCatalog(controller.signal).then(value => {
      if (!controller.signal.aborted) { setCatalog(value); if (!decisionParam) setBaselineId(current => current || value.baselines[0]?.id || '') }
    }).catch(error => { if (!controller.signal.aborted) setError(fail(error, systemText('preInvestment.tacticalAllocationWorkspace.unableToLoadTheAllocationCatalog'))) }).finally(() => { if (!controller.signal.aborted) setLoading(false) })
    return () => controller.abort()
  }, [reload])

  function invalidate() {
    version.current += 1
    setPreview(null); setDecision(null); setExperiments(previous => previous.map(item => ({ scenario: item.scenario }))); setPreflight(null); setBusy(false); setError(''); setNotice('')
  }
  function showDecision(value: TaaDecision) {
    setBaseline(value.preview.baseline); setRequest(value.preview.request); setPreview(value.preview); setDecision(value)
    setDecisionName(value.name); setDecisionNote(value.note ?? ''); setExperiments(value.scenarios ?? []); setTab('versions')
    updateAllocationJourney({ ...baselineJourney(value.preview.baseline), taaRunId: value.id })
  }
  useEffect(() => {
    if (loading || !catalogRef.current) return
    const controller = new AbortController()
    if (!baselineId) { setLoadingBaseline(false); return () => controller.abort() }
    invalidate(); setBaseline(null); setRequest(null)
    setLoadingBaseline(true)
    getTaaBaseline(baselineId, controller.signal).then(value => {
      if (controller.signal.aborted) return
      if (value.id !== baselineId) throw new Error(systemText('preInvestment.tacticalAllocationWorkspace.theLoadedSaaVersionDiffersFromThe'))
      if (openingDecision.current?.preview.baseline.id === value.id) { showDecision(openingDecision.current); openingDecision.current = null; return }
      const copied = restore.current?.baseline_id === value.id ? restore.current : null
      const draft = readAllocationDraft<{ request: TaaPreviewRequest; tab: Tab; name: string; note: string; scenarios: TaaScenario[] }>(`taa:${value.id}`)
      const restored = validDraftRequest(draft?.request, value.id) && draft && ['views', 'backtest', 'scenarios', 'versions'].includes(draft.tab) && Array.isArray(draft.scenarios) && draft.scenarios.every(item => item && typeof item.name === 'string' && ['shock', 'historical'].includes(item.kind)) ? draft : null
      restore.current = null; setBaseline(value); setRequest(copied ?? restored?.request ?? initialRequest(value, catalogRef.current, platformRef.current))
      setExperiments((restored?.scenarios ?? []).map(scenario => ({ scenario }))); setDecisionName(restored?.name ?? systemText('preInvestment.tacticalAllocationWorkspace.tacticalPlan', { p0: value.name })); setDecisionNote(restored?.note ?? '')
      if (restored) { setTab(restored.tab); setNotice(systemText('preInvestment.tacticalAllocationWorkspace.researchDraftRestoredInputsAndScenarioAssumptions')) }
      updateAllocationJourney(baselineJourney(value))
    }).catch(error => { if (!controller.signal.aborted) setError(fail(error, systemText('preInvestment.tacticalAllocationWorkspace.unableToLoadTheBaseline'))) }).finally(() => { if (!controller.signal.aborted) setLoadingBaseline(false) })
    return () => controller.abort()
  }, [baselineId, loading, reload])
  useEffect(() => {
    const id = decisionParam; if (!id || decision?.id === id) return
    const controller = new AbortController(); const token = version.current
    getTaaDecision(id, controller.signal).then(value => {
      if (controller.signal.aborted || token !== version.current) return
      if (value.id !== id) throw new Error(systemText('preInvestment.tacticalAllocationWorkspace.theSavedVersionDoesNotMatchThe'))
      openingDecision.current = value
      if (baselineId === value.preview.baseline.id) { showDecision(value); openingDecision.current = null }
      else setBaselineId(value.preview.baseline.id)
    }).catch(error => { if (!controller.signal.aborted && token === version.current) setError(fail(error, systemText('preInvestment.tacticalAllocationWorkspace.unableToLoadTheSavedVersion'))) })
    return () => controller.abort()
  }, [decisionParam])
  useEffect(() => {
    if (request?.signal_mode !== 'regime') return
    let active = true; setRegimeError('')
    listHistoricalRegimeRuns().then(value => { if (active) setRuns(value.filter(eligible)) }).catch(error => { if (active) setRegimeError(fail(error, systemText('preInvestment.tacticalAllocationWorkspace.unableToLoadScenarioVersions'))) })
    return () => { active = false }
  }, [request?.signal_mode])
  useEffect(() => {
    let active = true; setRegime(null); setRegimeStates([]); setRegimeError('')
    if (!request?.regime_run_id || request.signal_mode !== 'regime') { setLoadingRegime(false); return () => { active = false } }
    setLoadingRegime(true); const id = request.regime_run_id
    getHistoricalRegimeRun(id).then(async value => {
      if (!active) return
      const summary = runs.find(item => item.id === id)
      if (value.id !== id || !eligible(value) || (summary && (value.content_hash !== summary.content_hash || value.definition_revision !== summary.definition_revision))) throw new Error(systemText('preInvestment.tacticalAllocationWorkspace.scenarioDetailsDifferFromThePublishedVersion'))
      const binding = value.definition?.study?.reference
      let states = value.states
      if (binding) {
        const reference = await getHistoricalRegimeRun(binding.run_id)
        if (!active) return
        if (reference.id !== binding.run_id || reference.content_hash !== binding.content_hash || reference.immutable !== true || reference.mode !== 'retrospective' || !reference.publications?.some(item =>
          item.id === binding.publication_id && item.run_id === reference.id && item.definition_revision === reference.definition_revision &&
          ['research_display', 'product_research', 'formal_backtest', 'taa'].includes(item.usage) && item.run_content_hash === binding.content_hash
        )) throw new Error(systemText('preInvestment.tacticalAllocationWorkspace.theHistoricalReferenceStateAxisDiffersFrom'))
        states = reference.states
      }
      setRegime(value); setRegimeStates(states)
      setRequest(previous => previous?.regime_run_id === id ? { ...previous, state_tilts: previous.state_tilts ?? Object.fromEntries(states.map(state => [state.id, Object.fromEntries(Object.keys(previous.manual_tilts).map(asset => [asset, 0]))])) } : previous)
    }).catch(error => { if (active) setRegimeError(fail(error, systemText('preInvestment.tacticalAllocationWorkspace.unableToLoadScenarioDetails'))) }).finally(() => { if (active) setLoadingRegime(false) })
    return () => { active = false }
  }, [request?.regime_run_id, request?.signal_mode, runs])

  const pitConflict = request && platformDay && request.as_of > platformDay ? systemText('preInvestment.tacticalAllocationWorkspace.researchDateIsLaterThanTheCurrent', { p0: request.as_of, p1: platformDay }) : ''
  const allocationCoverage = catalog?.allocations.find(item => item.alloc_name === baseline?.alloc_name)?.coverage
  const canAlignPit = Boolean(pitConflict && allocationCoverage && platformDay && allocationCoverage.start_date < platformDay)
  const expired = Boolean(preview && preview.recommendation.expires_on < today())
  const selectedCandidate = preview?.candidates.find(item => item.id === preview.selected_id)
  const missingProducts = baseline?.strategic_universe_id && baseline.implementation_status !== 'complete'
    ? systemText('preInvestment.tacticalAllocationWorkspace.actualTradingProductsAreNotConfiguredTaa', { p0: baseline.assets.filter(asset => !asset.products.length).map(asset => asset.name).join('、') }) : ''
  const applyBlocked = missingProducts || (preview?.request.decision_policy && !preview.application ? systemText('preInvestment.tacticalAllocationWorkspace.executionEligibilityEvidenceIsMissingRecalculateBefore') : '') || (preview?.request.decision_policy && preview.request.current_weights_as_of !== today() ? systemText('preInvestment.tacticalAllocationWorkspace.theActualHoldingsSnapshotIsNotFrom') : '') || (preview?.application?.eligible === false ? preview.application.reasons.join('；') : '') || pitConflict || (preview?.policy_check?.current_application_eligible === false ? systemText('preInvestment.tacticalAllocationWorkspace.thePolicyIsDueForReviewOr') : '') || (preview?.policy_check?.within_limits === false ? preview.policy_check.violations.join('；') : '') || (preflight?.quality.status === 'blocked' ? systemText('preInvestment.tacticalAllocationWorkspace.dataContainOrderOfMagnitudeJumpsReturn') : selectedCandidate?.validation_feasible === false ? systemText('preInvestment.tacticalAllocationWorkspace.theSelectedCandidateBreachesConstraintsInThe') : preview?.recommendation.turnover_from_current != null && preview.recommendation.turnover_from_current > preview.request.max_turnover + 1e-8 ? systemText('preInvestment.tacticalAllocationWorkspace.turnoverFromReferenceHoldingsToThisPlan') : '')
  const manualSum = Object.values(request?.manual_tilts ?? {}).reduce((sum, value) => sum + value, 0)
  const invalidNumeric = request ? [request.lookback, request.max_abs_tilt, request.transaction_cost_bps, request.risk_penalty, request.max_tracking_error, request.max_turnover, request.confidence_floor, request.max_signal_age_days, request.review_days, ...(request.walk_forward ? [request.walk_forward.training_periods, request.walk_forward.validation_periods] : []), ...Object.values(request.manual_tilts), ...Object.values(request.current_weights ?? {}), ...Object.values(request.state_tilts ?? {}).flatMap(Object.values)].some(value => !Number.isFinite(value)) : true
  const extendedIssue = request ? policySignalIssue(request, baseline?.assets.map(a => a.id) ?? []) : ''
  const formIssue = !request ? '' : extendedIssue ? extendedIssue : pitConflict ? pitConflict : invalidNumeric ? systemText('preInvestment.tacticalAllocationWorkspace.completeAllNumericFieldsUseNegativeValues') : !(request.start_date < request.train_end_date && request.train_end_date < request.end_date && request.end_date <= request.as_of) ? systemText('preInvestment.tacticalAllocationWorkspace.datesMustSatisfyBacktestStartTrainingEnd') : request.signal_mode === 'manual' && Math.abs(manualSum) > 1e-8 ? systemText('preInvestment.tacticalAllocationWorkspace.manualDeviationsMustSumToZeroOverweights') : request.current_weights && Math.abs(Object.values(request.current_weights).reduce((sum, value) => sum + value, 0) - 1) > 1e-8 ? systemText('preInvestment.tacticalAllocationWorkspace.referenceHoldingWeightsMustSumTo100') : request.signal_mode === 'regime' && (!regime || loadingRegime) ? systemText('preInvestment.tacticalAllocationWorkspace.selectAndVerifyAPublishedRealTime') : ''
  useEffect(() => {
    if (previousPlatform.current === platformDay) return
    previousPlatform.current = platformDay
    if (!request && !decision) return
    if (decision) { version.current += 1; setPreflight(null); setNotice(systemText('preInvestment.tacticalAllocationWorkspace.thePlatformDataContextChangedThisHistorical')) }
    else { invalidate(); setNotice(systemText('preInvestment.tacticalAllocationWorkspace.thePlatformDataContextChangedDraftInputs')) }
  }, [platformDay])
  useEffect(() => {
    if (!request || !baseline || formIssue) { setChecking(false); setPreflight(null); return }
    const controller = new AbortController()
    setChecking(true); setPreflight(null); setCheckError('')
    const timeout = window.setTimeout(() => {
      preflightTaa(request, controller.signal).then(value => { if (!controller.signal.aborted) setPreflight(value) })
        .catch(caught => { if (!controller.signal.aborted) setCheckError(fail(caught, systemText('preInvestment.tacticalAllocationWorkspace.unableToCheckResearchConditionsPleaseRetry'))) })
        .finally(() => { if (!controller.signal.aborted) setChecking(false) })
    }, 200)
    return () => { window.clearTimeout(timeout); controller.abort() }
  }, [fingerprint, baseline?.id, formIssue, checkRevision])
  useEffect(() => {
    if (!request || request.baseline_id !== baselineId) return
    writeAllocationDraft(`taa:${baselineId}`, { request, tab, name: decisionName, note: decisionNote, scenarios: experiments.map(item => item.scenario) })
  }, [fingerprint, baselineId, tab, decisionName, decisionNote, experiments])
  function markDraftChanged() { version.current += 1; setDecision(null); updateAllocationJourney({ taaRunId: undefined }); if (decisionParam) setParams({ baseline: baselineId }, { replace: true }) }
  function update(patch: Partial<TaaPreviewRequest>) {
    if (!request) return
    const next = { ...request, ...patch, selected_candidate_id: undefined }
    if (JSON.stringify(next) === fingerprint) return
    markDraftChanged(); invalidate(); setRequest(next)
    if (patch.as_of) updateAllocationJourney({ taaRunId: undefined })
  }
  function alignPitDates() {
    if (!request || !platformDay || !allocationCoverage || !canAlignPit) return
    const end = [request.end_date, allocationCoverage.end_date, platformDay].sort()[0]
    const start = request.start_date < end ? request.start_date : allocationCoverage.start_date
    if (start >= end) return
    const split = request.train_end_date > start && request.train_end_date < end ? request.train_end_date : new Date(Date.parse(start) + (Date.parse(end) - Date.parse(start)) * .67).toISOString().slice(0, 10)
    update({ as_of: platformDay, start_date: start, end_date: end, train_end_date: split })
  }
  function selectBaseline(id: string) {
    if (id === baselineId) return
    invalidate(); if (!id) { setBaseline(null); setRequest(null) }
    setBaselineId(id); setParams(id ? { baseline: id } : {}, { replace: true })
  }

  function baselineCreated(value: TaaBaseline) { setCatalog(previous => ({ ...(previous ?? emptyCatalog), baselines: [value, ...(previous?.baselines ?? [])] })); selectBaseline(value.id) }

  async function calculate(candidateId?: string) {
    if (!request || !baseline || formIssue || busy || checking || !preflight?.can_calculate) return
    const token = ++version.current; const requested = fingerprint
    setBusy(true); setError(''); setNotice(''); setPreview(null); setDecision(null); setExperiments(previous => previous.map(item => ({ scenario: item.scenario })))
    try {
      const result = await previewTaa({ ...request, selected_candidate_id: candidateId })
      if (mounted.current && token === version.current && requested === latest.current) {
        if (result.baseline.id !== baseline.id) throw new Error(systemText('preInvestment.tacticalAllocationWorkspace.theCalculationReferencedADifferentSaaVersion'))
        setPreview(result); setTab('backtest')
      }
    } catch (error) { if (mounted.current && token === version.current) setError(fail(error, systemText('preInvestment.tacticalAllocationWorkspace.tacticalAllocationResearchDidNotComplete'))) }
    finally { if (mounted.current && token === version.current) setBusy(false) }
  }
  async function saveDecision() {
    if (!preview || !decisionName.trim() || saving || formIssue || checking || !preflight) return
    const token = version.current; const requestedSave = saveFingerprint; setSaving(true); setError('')
    try {
      const value = await saveTaaDecision({ request: preview.request, preview_hash: preview.preview_hash, name: decisionName.trim(), note: decisionNote, scenarios: experiments.filter(item => item.result).map(item => item.scenario) })
      if (mounted.current && (token !== version.current || requestedSave !== latestSave.current)) { setCatalog(previous => ({ ...(previous ?? emptyCatalog), decisions: [value, ...(previous?.decisions ?? []).filter(item => item.id !== value.id)] })); setNotice(systemText('preInvestment.tacticalAllocationWorkspace.theVersionUnderPreviousConditionsWasSaved')) }
      if (mounted.current && token === version.current && requestedSave === latestSave.current) { setDecision(value); setExperiments(value.scenarios ?? []); updateAllocationJourney({ baselineId: value.preview.baseline.id, taaRunId: value.id }); setCatalog(previous => ({ ...(previous ?? emptyCatalog), decisions: [value, ...(previous?.decisions ?? []).filter(item => item.id !== value.id)] })); setParams({ decision: value.id }, { replace: true }); setNotice(systemText('preInvestment.tacticalAllocationWorkspace.researchVersionSavedSubsequentChangesWillNot')) }
    } catch (error) { if (mounted.current && token === version.current) setError(fail(error, systemText('preInvestment.tacticalAllocationWorkspace.unableToSaveTheVersion'))) }
    finally { if (mounted.current) setSaving(false) }
  }
  async function applyDecision() {
    if (!decision || expired || saving || checking || !preflight || applyBlocked || baseline?.apply_eligible === false) return
    const token = version.current; setSaving(true); setError('')
    try {
      const value = await taaProductAllocation(decision.id)
      if (mounted.current && token === version.current) { sessionStorage.setItem('portfolioResearchImport', JSON.stringify(value)); navigate(`/pre-investment/product-allocation-timing/construction?${new URLSearchParams({ ...(value.universe_snapshot_id ? { universe: value.universe_snapshot_id } : {}), decision: decision.id })}`) }
    } catch (error) { if (mounted.current && token === version.current) setError(fail(error, systemText('preInvestment.tacticalAllocationWorkspace.productAllocationHandoffDidNotComplete'))) }
    finally { if (mounted.current) setSaving(false) }
  }
  async function readDecision(id: string, copy: boolean) {
    const token = ++version.current; setError(''); setSaving(true)
    try {
      const value = await getTaaDecision(id)
      if (!mounted.current || token !== version.current) return
      if (!copy) { if (baselineId === value.preview.baseline.id) showDecision(value); else { openingDecision.current = value; setBaselineId(value.preview.baseline.id) } setParams({ decision: id }, { replace: true }); return }
      restore.current = value.preview.request
      if (baselineId !== value.preview.baseline.id) selectBaseline(value.preview.baseline.id)
      else { invalidate(); setRequest({ ...value.preview.request }); restore.current = null; setParams({ baseline: baselineId }, { replace: true }) }
      setExperiments((value.scenarios ?? []).map(item => ({ scenario: item.scenario }))); setDecisionNote(value.note ?? ''); setNotice(systemText('preInvestment.tacticalAllocationWorkspace.researchConditionsAndScenarioAssumptionsCopiedRecalculate')); setTab('views')
    } catch (error) { if (mounted.current && token === version.current) setError(fail(error, systemText('preInvestment.tacticalAllocationWorkspace.unableToLoadTheHistoricalVersion'))) }
    finally { if (mounted.current) setSaving(false) }
  }

  const loadFailed = Boolean(error) && !loading && !loadingBaseline && !baseline
  return <div className="min-h-screen min-w-0 max-w-full break-words bg-slate-50 pb-12 text-slate-800"><main className="mx-auto max-w-[1500px] min-w-0 space-y-4 px-3 py-4 sm:px-7">
    <header className="flex flex-col justify-between gap-3 sm:flex-row sm:items-start"><div className="min-w-0 max-w-5xl"><h1 className="mt-1 text-2xl font-semibold text-slate-950">{systemText('preInvestment.tacticalAllocationWorkspace.tacticalAssetAllocation')}</h1><p className="mt-2 text-sm leading-6 text-slate-600">{s('taaIntro.description')}</p><p className="mt-1 text-sm leading-6 text-slate-600">{s('taaIntro.workflow')}</p></div><span className="w-fit shrink-0 rounded-full border border-slate-300 bg-white px-3 py-1.5 text-xs text-slate-600">{preview ? systemText('preInvestment.tacticalAllocationWorkspace.calculatedReadyForReview') : systemText('preInvestment.tacticalAllocationWorkspace.researchWorkspace')}</span></header>
    {/* 什么都没读出来时屏幕上没有业务数值，整块换成公共错误态；研究计算、保存这类操作失败时结果还在，仍是纯文字。 */}
    <Feedback error={loadFailed ? '' : error} notice={notice} />
    <BaselineSetup catalog={catalog} selectedId={baselineId} loading={loading} onSelect={selectBaseline} onCreated={baselineCreated} />
    {loadingBaseline && <LoadingPanel text={systemText('preInvestment.tacticalAllocationWorkspace.loadingTheLockedBaselineAndAssetScope')} className="rounded-xl bg-white" />}
    {loadFailed && <ErrorPanel className="rounded-xl bg-white" message={error} action={<Button onClick={() => { setError(''); setReload(value => value + 1) }}>{systemText('preInvestment.tacticalAllocationWorkspace.retryLoading')}</Button>} />}
    {baseline && request && <>
      <div role="tablist" aria-label={systemText('preInvestment.tacticalAllocationWorkspace.tacticalAllocationResearch')} className="grid grid-cols-2 gap-1 sm:grid-cols-4 rounded-xl border border-slate-200 bg-white p-1">{tabs.map(([id, label]) => <button key={id} type="button" role="tab" id={`taa-tab-${id}`} aria-controls={`taa-panel-${id}`} aria-selected={tab === id} tabIndex={tab === id ? 0 : -1} onKeyDown={event => { if (event.key === 'ArrowRight' || event.key === 'ArrowLeft') { event.preventDefault(); const next = (tabs.findIndex(item => item[0] === tab) + (event.key === 'ArrowRight' ? 1 : -1) + tabs.length) % tabs.length; setTab(tabs[next][0]); document.getElementById(`taa-tab-${tabs[next][0]}`)?.focus() } }} onClick={() => setTab(id)} className={`min-h-11 min-w-0 rounded-lg px-2 py-2.5 text-sm focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent-700 ${tab === id ? 'bg-accent-50 font-semibold text-accent-900' : 'text-slate-600 hover:bg-slate-50'}`}>{label}</button>)}</div>
      {pitConflict && <section className="rounded-xl border border-amber-200 bg-amber-50 p-4 text-sm text-amber-900" aria-label={systemText('preInvestment.tacticalAllocationWorkspace.platformPitDateConflict')}><p>{pitConflict}</p><div className="mt-3 flex flex-wrap gap-2">{canAlignPit ? <button type="button" className={buttonClass} onClick={alignPitDates}>{systemText('preInvestment.tacticalAllocationWorkspace.alignDatesWithCurrentPit')}</button> : <p className="text-xs leading-5">{systemText('preInvestment.tacticalAllocationWorkspace.currentAssetClassCoverageStartsOn') + " "}{allocationCoverage?.start_date ?? systemText('preInvestment.tacticalAllocationWorkspace.anUnknownDate')} {" " + systemText('preInvestment.tacticalAllocationWorkspace.andHasNoDirectlyUsableResearchInterval')}</p>}<Link className={buttonClass} to={allocationJourneyPath('classes')}>{systemText('preInvestment.tacticalAllocationWorkspace.returnToAssetClassesToSelectData')}</Link><Link className={buttonClass} to="/settings/pit-snapshots">{systemText('preInvestment.tacticalAllocationWorkspace.checkPlatformPitContext')}</Link></div></section>}
      <TaaResearchContext baseline={baseline} request={request} preview={preview} preflight={preflight} checking={checking} error={checkError} contextIssue={pitConflict} onRetry={() => setCheckRevision(value => value + 1)} onChange={update} onEdit={() => setTab('backtest')} />
      <div className="grid min-w-0 gap-5 xl:grid-cols-[minmax(0,1fr)_300px]">
        <section className={`${sectionClass} space-y-4`} aria-label={systemText('preInvestment.tacticalAllocationWorkspace.currentAllocation')}><div className="flex flex-wrap items-start justify-between gap-3"><div><h2 className="text-lg font-semibold text-slate-950">{systemText('preInvestment.tacticalAllocationWorkspace.whatAllocationAreYouConsidering')}</h2><p className="mt-1 text-xs leading-5 text-slate-600">{baseline.name} {" " + systemText('preInvestment.tacticalAllocationWorkspace.policyDate') + " "}{baseline.as_of.slice(0, 10)}</p></div><span className="rounded-lg bg-amber-50 px-2.5 py-1.5 text-xs text-amber-900">{preview?.data.pit.status === 'verified' ? systemText('preInvestment.tacticalAllocationWorkspace.viewPointInTimeValidationScope') : systemText('preInvestment.tacticalAllocationWorkspace.researchResultHistoricalPitPendingVerification')}</span></div>
          <div className="space-y-3 sm:hidden">{baseline.assets.map(asset => <article key={asset.id} className="rounded-lg border border-slate-200 p-3"><h3 className="text-sm font-semibold">{asset.name}</h3><dl className="mt-3 grid grid-cols-3 gap-2 text-xs"><div><dt className="text-slate-600">SAA</dt><dd className="mt-1 font-medium tabular-nums">{percentText(asset.base_weight)}</dd></div><div><dt className="text-slate-600">{systemText('preInvestment.tacticalAllocationWorkspace.proposedWeight')}</dt><dd className="mt-1 font-semibold tabular-nums text-accent-900">{preview ? percentText(preview.recommendation.weights[asset.id]) : systemText('preInvestment.tacticalAllocationWorkspace.pendingCalculation')}</dd></div><div><dt className="text-slate-600">{systemText('preInvestment.tacticalAllocationWorkspace.deviationPercentagePoints')}</dt><dd className="mt-1 font-medium tabular-nums">{preview ? `${(preview.recommendation.tilts[asset.id] * 100).toFixed(2)}` : '—'}</dd></div></dl><p className="mt-3 text-xs leading-5 text-slate-600">{systemText('preInvestment.tacticalAllocationWorkspace.allowed') + " "}{percentText(asset.min_weight)}–{percentText(asset.max_weight)}；{preview?.recommendation.trade_deltas ? systemText('preInvestment.tacticalAllocationWorkspace.holdingsDifference', { p0: points(preview.recommendation.trade_deltas[asset.id]) }) : systemText('preInvestment.tacticalAllocationWorkspace.noReferenceHoldingsProvided')}</p></article>)}</div>
          <div className="hidden overflow-auto sm:block"><table className="w-full min-w-[540px] text-sm"><caption className="sr-only">{systemText('preInvestment.tacticalAllocationWorkspace.saaAndProposedTacticalWeightsDeviationsIn')}</caption><thead className="text-xs text-slate-600"><tr><th scope="col" className="px-2 py-3 text-left">{systemText('preInvestment.tacticalAllocationWorkspace.asset')}</th><th scope="col" className="px-2 py-3 text-right">SAA</th><th scope="col" className="px-2 py-3 text-right">{systemText('preInvestment.tacticalAllocationWorkspace.proposedWeight')}</th><th scope="col" className="px-2 py-3 text-right">{systemText('preInvestment.tacticalAllocationWorkspace.relativeToSaa')}</th><th scope="col" className="px-2 py-3 text-right">{systemText('preInvestment.tacticalAllocationWorkspace.differenceFromReferenceHoldings')}</th></tr></thead><tbody className="divide-y divide-slate-100">{baseline.assets.map(asset => <tr key={asset.id}><th scope="row" className="px-2 py-3 text-left font-medium">{asset.name}<span className="mt-1 block text-xs font-normal text-slate-600">{systemText('preInvestment.tacticalAllocationWorkspace.allowed') + " "}{percentText(asset.min_weight)}–{percentText(asset.max_weight)}</span></th><td className="px-2 py-3 text-right tabular-nums">{percentText(asset.base_weight)}</td><td className="px-2 py-3 text-right font-semibold tabular-nums text-accent-900">{preview ? percentText(preview.recommendation.weights[asset.id]) : systemText('preInvestment.tacticalAllocationWorkspace.pendingCalculation')}</td><td className="px-2 py-3 text-right tabular-nums">{preview ? points(preview.recommendation.tilts[asset.id]) : '—'}</td><td className="px-2 py-3 text-right text-xs tabular-nums">{preview?.recommendation.trade_deltas ? points(preview.recommendation.trade_deltas[asset.id]) : systemText('preInvestment.tacticalAllocationWorkspace.noReferenceHoldingsProvided')}</td></tr>)}</tbody></table></div><p className="text-xs leading-5 text-slate-600">{systemText('preInvestment.tacticalAllocationWorkspace.deviationIsTheDifferenceFromSaaDifferences')}</p>
        </section>
        <aside className={`${sectionClass} space-y-4`} aria-label={systemText('preInvestment.tacticalAllocationWorkspace.decisionSummary')}><p className="text-xs font-semibold text-accent-800">{systemText('preInvestment.tacticalAllocationWorkspace.currentDecision')}</p><h2 className="text-lg font-semibold text-slate-950">{preview?.application ? ({ maintain: systemText('preInvestment.tacticalAllocationWorkspace.keepHoldings'), waiting_execution: systemText('preInvestment.tacticalAllocationWorkspace.awaitExecution'), adjustment_proposal: systemText('preInvestment.tacticalAllocationWorkspace.adjustmentProposed'), ineligible: systemText('preInvestment.tacticalAllocationWorkspace.handoffUnavailable') }[preview.application.state]) : preview ? preview.recommendation.is_saa ? systemText('preInvestment.tacticalAllocationWorkspace.keepLongTermAllocation') : systemText('preInvestment.tacticalAllocationWorkspace.deviationPlanGenerated') : systemText('preInvestment.tacticalAllocationWorkspace.validateFirstThenDecide')}</h2><p className="text-sm leading-6 text-slate-600">{(preview ? researchMessage(preview.recommendation.reason) : undefined) ?? systemText('preInvestment.tacticalAllocationWorkspace.chooseSignalsAndDeviationLimitsThenCheck')}</p>{preview?.application && <div role="status" className="space-y-1 text-sm text-amber-800">{preview.application.reasons.map(reason => <p key={reason}>{researchMessage(reason)}</p>)}<p>{systemText('preInvestment.tacticalAllocationWorkspace.decisionDateForTheExecutableTarget')}{preview.application.decision_date ?? systemText('preInvestment.tacticalAllocationWorkspace.notYetAvailable')}{systemText('preInvestment.tacticalAllocationWorkspace.executionOpportunityThisPeriod')}{preview.application.execution_opportunity ? systemText('preInvestment.tacticalAllocationWorkspace.yes') : systemText('preInvestment.tacticalAllocationWorkspace.no')}{systemText('preInvestment.tacticalAllocationWorkspace.researchProposalOnly')}</p>{preview.application.pending_decision && <p>{systemText('preInvestment.tacticalAllocationWorkspace.latestDecision') + " "}{preview.application.latest_decision_date} {" " + systemText('preInvestment.tacticalAllocationWorkspace.isStillWithinTheLagPeriodAnd')}</p>}</div>}{preview?.recommendation.signal_details && <details className="text-xs leading-5"><summary className="cursor-pointer font-medium text-accent-800">{systemText('preInvestment.tacticalAllocationWorkspace.whyAdjustThisWay')}</summary><div className="mt-2 space-y-2">{preview.recommendation.signal_details.map(item => <div key={item.asset_id}><p className="font-medium">{baseline.assets.find(asset => asset.id === item.asset_id)?.name ?? item.asset_id} · {researchMessage(item.direction)}</p><p>{systemText('preInvestment.tacticalAllocationWorkspace.signal') + " "}{item.value == null ? systemText('preInvestment.tacticalAllocationWorkspace.unavailable') : percentText(item.value)} · {item.signal_date ?? systemText('preInvestment.tacticalAllocationWorkspace.noAvailableDate')}</p><p>{item.window?.window_start && <>{systemText('preInvestment.tacticalAllocationWorkspace.observed') + " "}{item.window.window_start} — {item.window.window_end}{systemText('preInvestment.tacticalAllocationWorkspace.latestPublication') + " "}{item.window.available_at}{systemText('preInvestment.tacticalAllocationWorkspace.ageAtDecision') + " "}{item.window.lag_days} {" " + systemText('preInvestment.tacticalAllocationWorkspace.days')}</>}</p><p>{systemText('preInvestment.tacticalAllocationWorkspace.rawDeviation') + " "}{points(item.raw_tilt)} {" " + systemText('preInvestment.tacticalAllocationWorkspace.afterConstraints') + " "}{points(item.applied_tilt)}</p>{item.constraint_reason && <p className="text-amber-800">{researchMessage(item.constraint_reason)}</p>}</div>)}</div></details>}{preview && <dl className="space-y-2 text-xs"><div className="flex justify-between gap-2"><dt className="text-slate-600">{systemText('preInvestment.tacticalAllocationWorkspace.decisionDate')}</dt><dd>{preview.request.as_of}</dd></div><div className="flex justify-between gap-2"><dt className="text-slate-600">{systemText('preInvestment.tacticalAllocationWorkspace.signalDate')}</dt><dd>{preview.recommendation.signal_date ?? systemText('preInvestment.tacticalAllocationWorkspace.noAvailableSignals')}</dd></div><div className="flex justify-between gap-2"><dt className="text-slate-600">{systemText('preInvestment.tacticalAllocationWorkspace.reviewDue')}</dt><dd className={expired ? 'text-amber-800' : ''}>{preview.recommendation.expires_on}{expired ? " " + systemText('preInvestment.tacticalAllocationWorkspace.expired') : ''}</dd></div></dl>}{formIssue && <p role="status" className="rounded-lg bg-amber-50 p-3 text-xs leading-5 text-amber-900">{formIssue}</p>}<button type="button" className={`${primaryClass} w-full`} disabled={busy || checking || !preflight?.can_calculate || Boolean(formIssue)} onClick={() => void calculate()}>{busy ? systemText('preInvestment.tacticalAllocationWorkspace.calculatingAndValidating') : checking ? systemText('preInvestment.tacticalAllocationWorkspace.checkingResearchConditions') : preview ? systemText('preInvestment.tacticalAllocationWorkspace.recalculateResearch') : systemText('preInvestment.tacticalAllocationWorkspace.calculateAndComparePlans')}</button><p className="text-xs leading-5 text-slate-600">{systemText('preInvestment.tacticalAllocationWorkspace.usesRealLocalDataSavingResearchVersions')}</p></aside>
      </div>

      {tabs.filter(([id]) => id !== tab).map(([id]) => <div key={id} role="tabpanel" id={`taa-panel-${id}`} aria-labelledby={`taa-tab-${id}`} hidden />)}
      <div role="tabpanel" id={`taa-panel-${tab}`} aria-labelledby={`taa-tab-${tab}`} className="min-w-0 space-y-5">
        {tab === 'views' && <>
          <section className={`${sectionClass} space-y-5`}><div><h2 className="text-lg font-semibold">{systemText('preInvestment.tacticalAllocationWorkspace.whyDeviate')}</h2><p className="mt-1 text-sm text-slate-600">{systemText('preInvestment.tacticalAllocationWorkspace.selectYourEvidenceThenSetAdjustmentSizes')}</p></div><div className="grid gap-3 md:grid-cols-3">{[['momentum', systemText('preInvestment.tacticalAllocationWorkspace.trendRules'), systemText('preInvestment.tacticalAllocationWorkspace.generateSignalsFromPriorWindowAssetPerformance')], ['manual', systemText('preInvestment.tacticalAllocationWorkspace.researcherViews'), systemText('preInvestment.tacticalAllocationWorkspace.specifyOverweightsAndUnderweightsTestSensitivityWith')], ['regime', systemText('preInvestment.tacticalAllocationWorkspace.publishedMarketStates'), systemText('preInvestment.tacticalAllocationWorkspace.useRealTimeScenariosAndProbabilitiesThat')], ['composite', systemText('preInvestment.tacticalAllocationWorkspace.compositeSignals'), systemText('preInvestment.tacticalAllocationWorkspace.combineMomentumWindowsOrResearchScoresWith')]].map(([id, label, help]) => <label key={id} className={`flex cursor-pointer items-start gap-3 rounded-xl border p-4 ${request.signal_mode === id ? 'border-accent-600 bg-accent-50/50' : 'border-slate-200'}`}><input type="radio" className="mt-1" name="taa-signal" value={id} checked={request.signal_mode === id} onChange={() => update({ signal_mode: id as TaaPreviewRequest['signal_mode'] })} /><span><span className="block text-sm font-semibold">{label}</span><span className="mt-1 block text-xs leading-5 text-slate-600">{help}</span></span></label>)}</div>
            {request.signal_mode === 'momentum' && <div className="grid gap-4 sm:grid-cols-[240px_1fr]"><Field label={systemText('preInvestment.tacticalAllocationWorkspace.trendObservationWindowTradingPeriods')}><NumberInput className={inputClass} value={request.lookback} min={2} max={756} onValueChange={lookback => update({ lookback })} /></Field><p className="self-center rounded-lg bg-slate-50 p-4 text-sm leading-6 text-slate-600">{systemText('preInvestment.tacticalAllocationWorkspace.eachPeriodUsesTheMostRecentComplete')}</p></div>}
            {request.signal_mode === 'manual' && <><p className="rounded-lg bg-amber-50 p-3 text-sm leading-6 text-amber-900">{systemText('preInvestment.tacticalAllocationWorkspace.viewsEnteredTodayWereNotNecessarilyKnown')}</p><div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">{baseline.assets.map(asset => <Field key={asset.id} label={systemText('preInvestment.tacticalAllocationWorkspace.deviationPercentagePoints2', { p0: asset.name })} hint={systemText('preInvestment.tacticalAllocationWorkspace.saaIs5Adds5PercentagePoints', { p0: percentText(asset.base_weight) })}><NumberInput className={inputClass} value={request.manual_tilts[asset.id] * 100} onValueChange={value => update({ manual_tilts: { ...request.manual_tilts, [asset.id]: value / 100 } })} /></Field>)}</div><p className="text-xs text-slate-600">{systemText('preInvestment.tacticalAllocationWorkspace.totalDeviation')}{points(manualSum)}{systemText('preInvestment.tacticalAllocationWorkspace.everyOverweightNeedsACorrespondingFundingSource')}</p></>}
            {request.signal_mode === 'regime' && <div className="space-y-4"><Feedback error={regimeError} /><Field label={systemText('preInvestment.tacticalAllocationWorkspace.publishedRealTimeScenarioVersion')}><select className={inputClass} value={request.regime_run_id ?? ''} onChange={event => update({ regime_run_id: event.target.value, state_tilts: undefined })}><option value="">{systemText('preInvestment.tacticalAllocationWorkspace.selectAScenarioVersion')}</option>{runs.map(run => <option key={run.id} value={run.id}>{run.name} · R{run.definition_revision}</option>)}</select></Field>{!runs.length && <p className="text-sm leading-6 text-amber-800">{systemText('preInvestment.tacticalAllocationWorkspace.noEligibleStateVersionsRealTimeAvailability')}<Link className="ml-1 underline" to={`/settings/scenario-algorithms?return_to=${encodeURIComponent(allocationJourneyPath('taa'))}`}>{systemText('preInvestment.tacticalAllocationWorkspace.validateAndPublishInTheScenarioAlgorithm')}</Link></p>}{loadingRegime && <p role="status" className="text-sm text-slate-600">{systemText('preInvestment.tacticalAllocationWorkspace.checkingScenarioDetailsAndPublicationRecords')}</p>}{regime && <><p className="text-xs leading-5 text-slate-600">{systemText('preInvestment.tacticalAllocationWorkspace.fillEveryAssetForEachStateDeviations')}</p><div className="overflow-auto"><table className="w-full min-w-[480px] text-sm"><thead><tr><th scope="col" className="p-2 text-left">{systemText('preInvestment.tacticalAllocationWorkspace.marketState')}</th>{baseline.assets.map(asset => <th scope="col" className="p-2 text-left" key={asset.id}>{asset.name}{systemText('preInvestment.tacticalAllocationWorkspace.percentagePoints2')}</th>)}</tr></thead><tbody>{regimeStates.map(state => <tr key={state.id}><th scope="row" className="p-2 text-left font-medium">{state.label}</th>{baseline.assets.map(asset => <td className="p-2" key={asset.id}><NumberInput aria-label={systemText('preInvestment.tacticalAllocationWorkspace.deviationPercentagePoints3', { p0: state.label, p1: asset.name })} className={inputClass} value={(request.state_tilts?.[state.id]?.[asset.id] ?? 0) * 100} onValueChange={value => update({ state_tilts: { ...request.state_tilts, [state.id]: { ...request.state_tilts?.[state.id], [asset.id]: value / 100 } } })} /></td>)}</tr>)}</tbody></table></div></>}</div>}
            <Field label={systemText('preInvestment.tacticalAllocationWorkspace.currentViewsAndReviewConditions')} hint={systemText('preInvestment.tacticalAllocationWorkspace.recordEvidenceRationaleAndChangesThatWould')}><textarea rows={3} className={`${inputClass} placeholder:text-slate-600 placeholder:opacity-100`} value={request.note} placeholder={systemText('preInvestment.tacticalAllocationWorkspace.forExampleImprovingTrendsSupportASmall')} onChange={event => update({ note: event.target.value })} /></Field>
          </section>
          <TaaPolicySignals request={request} assets={baseline.assets.map(a => a.id)} update={update} />
          <section className={`${sectionClass} space-y-4`}><h2 className="text-lg font-semibold">{systemText('preInvestment.tacticalAllocationWorkspace.howFarToDeviateAndWhenTo')}</h2><div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4"><Field label={systemText('preInvestment.tacticalAllocationWorkspace.perAssetDeviationLimitPercentagePoints')}><NumberInput className={inputClass} value={request.max_abs_tilt * 100} min={0} max={100} onValueChange={value => update({ max_abs_tilt: value / 100 })} /></Field><Field label={systemText('preInvestment.tacticalAllocationWorkspace.activeRiskLimitYear')}><NumberInput className={inputClass} value={request.max_tracking_error * 100} min={0} max={100} onValueChange={value => update({ max_tracking_error: value / 100 })} /></Field><Field label={systemText('preInvestment.tacticalAllocationWorkspace.onePeriodTurnoverLimit')}><NumberInput className={inputClass} value={request.max_turnover * 100} min={0} max={200} onValueChange={value => update({ max_turnover: value / 100 })} /></Field><Field label={systemText('preInvestment.tacticalAllocationWorkspace.daysUntilReview')}><NumberInput className={inputClass} value={request.review_days} min={1} max={365} onValueChange={review_days => update({ review_days })} /></Field></div><p className="text-xs leading-5 text-slate-600">{systemText('preInvestment.tacticalAllocationWorkspace.saaAssetBoundsAlsoApplyInfeasibleDeviations')}</p><details className="border-t border-slate-100 pt-3"><summary className="cursor-pointer text-sm font-medium">{systemText('preInvestment.tacticalAllocationWorkspace.referenceHoldingsAndSignalValidity')}</summary><div className="mt-4 space-y-4"><label className="flex min-h-11 items-center gap-2 text-sm"><input type="checkbox" checked={Boolean(request.current_weights)} onChange={event => update({ current_weights: event.target.checked ? zeroes(baseline) : undefined })} />{systemText('preInvestment.tacticalAllocationWorkspace.enterReferenceHoldingsToCompareRequiredAdjustments')}</label>{request.current_weights && <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">{baseline.assets.map(asset => <Field key={asset.id} label={systemText('preInvestment.tacticalAllocationWorkspace.referenceHolding', { p0: asset.name })}><NumberInput className={inputClass} value={request.current_weights![asset.id] * 100} onValueChange={value => update({ current_weights: { ...request.current_weights, [asset.id]: value / 100 } })} /></Field>)}</div>}<div className="grid gap-4 sm:grid-cols-2"><Field label={systemText('preInvestment.tacticalAllocationWorkspace.stateConfidenceThreshold')} hint={systemText('preInvestment.tacticalAllocationWorkspace.appliesOnlyToPublishedMarketStatesNo')}><NumberInput className={inputClass} value={request.confidence_floor * 100} onValueChange={value => update({ confidence_floor: value / 100 })} /></Field><Field label={systemText('preInvestment.tacticalAllocationWorkspace.maximumStateSignalAgeDays')}><NumberInput className={inputClass} value={request.max_signal_age_days} onValueChange={max_signal_age_days => update({ max_signal_age_days })} /></Field></div></div></details></section>
          <button type="button" className={primaryClass} onClick={() => setTab('backtest')}>{systemText('preInvestment.tacticalAllocationWorkspace.nextBacktestAndCosts')}</button>
        </>}
        {tab === 'backtest' && <><TaaWalkForward value={request.walk_forward} result={preview?.walk_forward} disabled={busy} onChange={walk_forward => update({ walk_forward })} /><section className={`${sectionClass} space-y-4`}><h2 className="text-lg font-semibold">{systemText('preInvestment.tacticalAllocationWorkspace.selectOnTrainingDataThenValidateOn')}</h2><div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4"><Field label={systemText('preInvestment.tacticalAllocationWorkspace.backtestStart')}><input type="date" min={preflight?.coverage.start_date} max={request.train_end_date} className={inputClass} value={request.start_date} onChange={event => update({ start_date: event.target.value })} /></Field><Field label={systemText('preInvestment.tacticalAllocationWorkspace.trainingEnd')}><input type="date" min={request.start_date} max={request.end_date} className={inputClass} value={request.train_end_date} onChange={event => update({ train_end_date: event.target.value })} /></Field><Field label={systemText('preInvestment.tacticalAllocationWorkspace.backtestEnd')}><input type="date" max={request.as_of} min={request.train_end_date} className={inputClass} value={request.end_date} onChange={event => update({ end_date: event.target.value })} /></Field><Field label={systemText('preInvestment.tacticalAllocationWorkspace.decisionDate')} hint={systemText('preInvestment.tacticalAllocationWorkspace.onlyDataKnownAndSignalsEffectiveOn')}><input type="date" max={today()} className={inputClass} value={request.as_of} onChange={event => update({ as_of: event.target.value })} /></Field></div><div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4"><Field label={systemText('preInvestment.tacticalAllocationWorkspace.candidateComparisonObjective')}><select className={inputClass} value={request.objective} onChange={event => update({ objective: event.target.value as TaaPreviewRequest['objective'] })}><option value="active_utility">{systemText('preInvestment.tacticalAllocationWorkspace.balanceNetExcessReturnAndActiveRisk')}</option><option value="excess_return">{systemText('preInvestment.tacticalAllocationWorkspace.trainingPeriodNetExcessReturn')}</option><option value="min_drawdown">{systemText('preInvestment.tacticalAllocationWorkspace.trainingPeriodMaximumDrawdown')}</option></select></Field><Field label={systemText('preInvestment.tacticalAllocationWorkspace.oneWayCostsBasisPoints')} hint={systemText('preInvestment.tacticalAllocationWorkspace.10BasisPoints010SaaAnd')}><NumberInput className={inputClass} value={request.transaction_cost_bps} min={0} onValueChange={transaction_cost_bps => update({ transaction_cost_bps })} /></Field><Field label={systemText('preInvestment.tacticalAllocationWorkspace.activeRiskPenalty')}><NumberInput className={inputClass} value={request.risk_penalty} min={0} onValueChange={risk_penalty => update({ risk_penalty })} /></Field><Field label={systemText('preInvestment.tacticalAllocationWorkspace.searchDeviationStrength')}><select className={inputClass} value={request.search ? 'search' : 'fixed'} onChange={event => update({ search: event.target.value === 'search' })}><option value="search">{systemText('preInvestment.tacticalAllocationWorkspace.trainingSearchMultipleStrengthsAndZeroDeviation')}</option><option value="fixed">{systemText('preInvestment.tacticalAllocationWorkspace.fixedHypothesisComparisonDefinedStrengthAndZero')}</option></select></Field></div><p className="text-xs leading-5 text-slate-600">{request.decision_policy ? systemText('preInvestment.tacticalAllocationWorkspace.holdingsAdvanceOnCommonObservationDatesAdjustments') : systemText('preInvestment.tacticalAllocationWorkspace.originalResearchConventionRestoreTargetWeightsDaily')}{systemText('preInvestment.tacticalAllocationWorkspace.postTrainingDataAreForIndependentValidation')}</p><button type="button" className={primaryClass} disabled={busy || checking || !preflight?.can_calculate || Boolean(formIssue)} onClick={() => void calculate()}>{busy ? systemText('preInvestment.tacticalAllocationWorkspace.calculatingAndValidating') : systemText('preInvestment.tacticalAllocationWorkspace.runBacktestAndCompareCandidates')}</button></section>{preview ? <TaaPerformance result={preview} busy={busy} onSelect={id => void calculate(id)} /> : <Empty title={systemText('preInvestment.tacticalAllocationWorkspace.resultsWillAppearHere')}><p>{systemText('preInvestment.tacticalAllocationWorkspace.setTrainingAndValidationDatesThenReview')}</p></Empty>}</>}
        {tab === 'scenarios' && <TaaScenarioExperiments key={baseline.id} baselineId={baseline.id} preview={preview} experiments={experiments} onChange={value => { setExperiments(value); markDraftChanged() }} onBacktest={() => setTab('backtest')} />}
        {tab === 'versions' && <><section className={`${sectionClass} space-y-4`}><h2 className="text-lg font-semibold">{systemText('preInvestment.tacticalAllocationWorkspace.saveYourJudgmentAndHandOffTo')}</h2><p className="text-sm leading-6 text-slate-600">{systemText('preInvestment.tacticalAllocationWorkspace.saveSaaViewsBacktestResultsCompletedScenarios')}</p>{preview ? <><p className="text-sm text-slate-600">{systemText('preInvestment.tacticalAllocationWorkspace.completedScenarios') + " "}{experiments.filter(item => item.result).length} {" " + systemText('preInvestment.tacticalAllocationWorkspace.awaitingRecalculation') + " "}{experiments.filter(item => !item.result).length} {" " + systemText('preInvestment.tacticalAllocationWorkspace.assumptionsAwaitingRecalculationRemainInTheDraft')}</p><Field label={systemText('preInvestment.tacticalAllocationWorkspace.researchConclusionAndReviewPlan')} hint={systemText('preInvestment.tacticalAllocationWorkspace.explainAdoptionKeepingSaaOrDeferralAnd')}><textarea rows={3} className={inputClass} value={decisionNote} onChange={event => { setDecisionNote(event.target.value); markDraftChanged() }} /></Field><Field label={systemText('preInvestment.tacticalAllocationWorkspace.researchVersionName')}><input className={inputClass} value={decisionName} onChange={event => { setDecisionName(event.target.value); markDraftChanged() }} /></Field><div className="flex flex-wrap gap-3"><button type="button" className={primaryClass} disabled={saving || Boolean(formIssue) || checking || !preflight || !decisionName.trim() || Boolean(decision) || preflight?.quality.status === 'blocked'} onClick={() => void saveDecision()}>{saving ? systemText('preInvestment.tacticalAllocationWorkspace.processing') : decision ? systemText('preInvestment.tacticalAllocationWorkspace.currentResearchVersionSaved') : systemText('preInvestment.tacticalAllocationWorkspace.saveResearchVersion')}</button><button type="button" className={buttonClass} disabled={!decision || expired || saving || checking || !preflight || Boolean(applyBlocked) || baseline.apply_eligible === false} onClick={() => void applyDecision()}>{systemText('preInvestment.tacticalAllocationWorkspace.handOffToProductAllocation')}</button>{decision && <Link className={buttonClass} to={`/pre-investment/product-allocation-timing?source=${encodeURIComponent(decision.id)}`}>{systemText('preInvestment.tacticalAllocationWorkspace.checkProductsCostsAndFunding')}</Link>}</div>{expired && <p role="status" className="text-sm text-amber-800">{systemText('preInvestment.tacticalAllocationWorkspace.thisDecisionIsDueForReviewHistorical')}</p>}{applyBlocked && <p role="status" className="text-sm text-amber-800">{researchMessage(applyBlocked)}</p>}{!missingProducts && baseline.apply_reasons?.map((reason, index) => <p key={index} className="text-xs text-amber-800">{researchMessage(reason)}</p>)}{!decision && <p className="text-xs text-slate-600">{systemText('preInvestment.tacticalAllocationWorkspace.saveTheCurrentResultBeforePassingExplicit')}</p>}</> : <Empty title={systemText('preInvestment.tacticalAllocationWorkspace.noResultsAvailableToSave')}><p>{systemText('preInvestment.tacticalAllocationWorkspace.calculateAndReviewAPlanFirstRecalculate')}</p></Empty>}</section>
          <section className={`${sectionClass} space-y-4`}><h2 className="text-lg font-semibold">{systemText('preInvestment.tacticalAllocationWorkspace.historicalResearchVersions')}</h2>{!catalog?.decisions.length ? <p className="text-sm text-slate-600">{systemText('preInvestment.tacticalAllocationWorkspace.noTacticalAllocationResearchSavedYet')}</p> : <div className="divide-y divide-slate-100">{catalog.decisions.map(item => <div key={item.id} className="flex flex-col justify-between gap-3 py-3 sm:flex-row sm:items-center"><div><p className="text-sm font-medium">{item.name}</p><p className="mt-1 flex flex-wrap items-center gap-x-1 text-xs text-slate-600">SAA：<UpstreamLink item={upstreamOf(item, 'saa_policy')} /></p><UsabilityNote usable={item.usable} /><p className="mt-1 text-xs text-slate-600">{systemText('preInvestment.tacticalAllocationWorkspace.savedOn') + " "}{item.created_at.slice(0, 10)}{item.preview?.recommendation?.expires_on ? systemText('preInvestment.tacticalAllocationWorkspace.review', { p0: item.preview.recommendation.expires_on, p1: item.preview.recommendation.expires_on < today() ? systemText('preInvestment.tacticalAllocationWorkspace.expired2') : '' }) : ''}</p></div><div className="flex gap-2"><button type="button" className={buttonClass} disabled={saving} onClick={() => void readDecision(item.id, false)}>{systemText('preInvestment.tacticalAllocationWorkspace.viewVersion')}</button><button type="button" className={buttonClass} disabled={saving} onClick={() => void readDecision(item.id, true)}>{systemText('preInvestment.tacticalAllocationWorkspace.copyAsNewResearch')}</button></div></div>)}</div>}</section>
          <section className={`${sectionClass} space-y-4`}><h2 className="text-lg font-semibold">{systemText('preInvestment.tacticalAllocationWorkspace.wasThisInformationActuallyKnownAtThe')}</h2><ol className="space-y-3 text-sm leading-6 text-slate-600"><li><strong className="text-slate-900">{systemText('preInvestment.tacticalAllocationWorkspace.dataAvailability')}</strong>{systemText('preInvestment.tacticalAllocationWorkspace.observationDatesDifferFromPublicationDatesCurrent')}</li><li><strong className="text-slate-900">{systemText('preInvestment.tacticalAllocationWorkspace.signalEffectiveness')}</strong>{systemText('preInvestment.tacticalAllocationWorkspace.useCompletedWindowsStatesMustBeAvailable')}</li><li><strong className="text-slate-900">{systemText('preInvestment.tacticalAllocationWorkspace.trainingBoundary')}</strong>{systemText('preInvestment.tacticalAllocationWorkspace.onlyTrainingDataSelectDeviationStrengthIndependent')}</li><li><strong className="text-slate-900">{systemText('preInvestment.tacticalAllocationWorkspace.decisionAuditTrail')}</strong>{systemText('preInvestment.tacticalAllocationWorkspace.savingFreezesInputsAndResultsWithoutRewriting')}</li></ol>{[...new Set([...(preview?.data.pit.reasons ?? baseline.pit.reasons), ...(preview?.warnings ?? [])])].map(reason => <p key={reason} className="rounded-lg bg-amber-50 p-3 text-xs leading-5 text-amber-900">{researchMessage(reason)}</p>)}<details className="border-t border-slate-100 pt-3"><summary className="cursor-pointer text-sm font-medium">{systemText('preInvestment.tacticalAllocationWorkspace.dataSourcesDailyWeightsAndCalculationAudit')}</summary><div className="mt-3 space-y-3 text-xs leading-5 text-slate-600"><p className="break-all">{systemText('preInvestment.tacticalAllocationWorkspace.saaVersion')}{baseline.id} {" " + systemText('preInvestment.tacticalAllocationWorkspace.verification')}{baseline.content_hash}</p><p>{systemText('preInvestment.tacticalAllocationWorkspace.baselineFrozenAt')}{baseline.created_at}</p>{preview && <><p className="break-all">{systemText('preInvestment.tacticalAllocationWorkspace.resultVerification')}{preview.preview_hash}</p><pre className="max-h-64 overflow-auto whitespace-pre-wrap break-all rounded-lg bg-slate-50 p-3">{JSON.stringify({ data: preview.data.lineage, timing_and_selection: preview.audit, execution: preview.execution }, null, 2)}</pre><div className="max-h-64 overflow-auto"><table className="w-full min-w-[450px] text-xs"><thead><tr><th scope="col" className="p-2 text-left">{systemText('preInvestment.tacticalAllocationWorkspace.date')}</th>{baseline.assets.map(asset => <th scope="col" key={asset.id} className="p-2 text-right">{asset.name}</th>)}<th scope="col" className="p-2 text-right">{systemText('preInvestment.tacticalAllocationWorkspace.turnover')}</th></tr></thead><tbody>{preview.weight_path.map(row => <tr key={row.date}><td className="p-2">{row.date}</td>{baseline.assets.map(asset => <td className="p-2 text-right" key={asset.id}>{percentText(row.weights[asset.id])}</td>)}<td className="p-2 text-right">{percentText(row.turnover)}</td></tr>)}</tbody></table></div></>}</div></details></section>
        </>}
      </div>
      {preview && tab !== 'versions' && <div className="flex flex-col items-start justify-between gap-3 rounded-xl border border-accent-200 bg-accent-50 px-5 py-4 sm:flex-row sm:items-center"><p className="text-sm text-accent-900">{systemText('preInvestment.tacticalAllocationWorkspace.afterReviewingRiskSaveYourJudgmentAnd')}</p><button type="button" className={buttonClass} onClick={() => setTab('versions')}>{systemText('preInvestment.tacticalAllocationWorkspace.saveAndHandOff')}</button></div>}
    </>}
  </main></div>
}

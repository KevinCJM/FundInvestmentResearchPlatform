import { useEffect, useRef, useState } from 'react'
import { Link, useNavigate, useSearchParams } from 'react-router-dom'
import BaselineSetup from '../components/tactical-allocation/BaselineSetup'
import { TaaPerformance } from '../components/tactical-allocation/TaaResults'
import TaaResearchContext from '../components/tactical-allocation/TaaResearchContext'
import TaaScenarioExperiments from '../components/tactical-allocation/TaaScenarioExperiments'
import { useResearchDay } from '../app/ResearchContext'
import { allocationJourneyPath, readAllocationDraft, readAllocationJourney, updateAllocationJourney, writeAllocationDraft } from '../app/allocationJourney'
import { buttonClass, Empty, Feedback, Field, inputClass, NumberInput, percentText, primaryClass, sectionClass, today } from '../components/risk-models/ResearchUI'
import { getHistoricalRegimeRun, listHistoricalRegimeRuns, type HistoricalRegimeRun } from '../services/historicalRegimes'
import { isRegimeRunEligibleForTaa } from '../services/regimePublicationEligibility'
import { getTaaBaseline, getTaaCatalog, getTaaDecision, previewTaa, preflightTaa, saveTaaDecision, taaProductAllocation, type TaaBaseline, type TaaCatalog, type TaaDecision, type TaaPreview, type TaaPreviewRequest, type TaaScenario, type TaaScenarioExperiment, type TaaPreflight } from '../services/tacticalAllocation'

type Tab = 'views' | 'backtest' | 'scenarios' | 'versions'
const tabs: Array<[Tab, string]> = [['views', '观点与规则'], ['backtest', '回测与选优'], ['scenarios', '情景模拟'], ['versions', '版本与审计']]
const emptyCatalog: TaaCatalog = { allocations: [], baselines: [], decisions: [] }
const zeroes = (baseline: TaaBaseline) => Object.fromEntries(baseline.assets.map(asset => [asset.id, 0]))
const points = (value: number | undefined) => value == null || !Number.isFinite(value) ? '—' : `${value > 0 ? '+' : ''}${(value * 100).toFixed(2)} 个百分点`
const fail = (error: unknown, message: string) => error instanceof Error ? error.message : message
const eligible = (run: HistoricalRegimeRun) => run.schema_version === '2.0' && isRegimeRunEligibleForTaa(run)

function validDraftRequest(value: unknown, baselineId: string): value is TaaPreviewRequest {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return false
  const row = value as Record<string, unknown>
  const record = (item: unknown) => Boolean(item) && typeof item === 'object' && !Array.isArray(item)
  return row.baseline_id === baselineId && ['start_date', 'end_date', 'as_of', 'train_end_date', 'note'].every(key => typeof row[key] === 'string')
    && ['momentum', 'manual', 'regime'].includes(String(row.signal_mode)) && typeof row.search === 'boolean'
    && record(row.manual_tilts) && (row.current_weights == null || record(row.current_weights))
    && (row.state_tilts == null || (record(row.state_tilts) && Object.values(row.state_tilts).every(record)))
}

function initialRequest(baseline: TaaBaseline, catalog: TaaCatalog | null, platformDay: string | null = null): TaaPreviewRequest {
  const coverage = catalog?.allocations.find(item => item.alloc_name === baseline.alloc_name)?.coverage
  const journey = readAllocationJourney()
  const end = [baseline.as_of.slice(0, 10), coverage?.end_date, today(), platformDay && (!coverage?.start_date || coverage.start_date < platformDay) ? platformDay : undefined, journey.baselineId === baseline.id ? journey.researchDate : undefined].filter(Boolean).sort()[0]!
  const start = coverage?.start_date ?? `${Number(end.slice(0, 4)) - 3}${end.slice(4)}`
  const split = new Date(Date.parse(start) + (Date.parse(end) - Date.parse(start)) * .67).toISOString().slice(0, 10)
  return { baseline_id: baseline.id, start_date: start, end_date: end, as_of: end, train_end_date: split, signal_mode: 'momentum', lookback: 60, manual_tilts: zeroes(baseline), max_abs_tilt: .1, transaction_cost_bps: 10, risk_penalty: 3, max_tracking_error: .1, max_turnover: 1, confidence_floor: .6, max_signal_age_days: 31, search: true, objective: 'active_utility', review_days: 30, note: '' }
}

export default function TacticalAllocationWorkspace() {
  const navigate = useNavigate()
  const platformDay = useResearchDay()
  const platformRef = useRef(platformDay); platformRef.current = platformDay
  const previousPlatform = useRef(platformDay)
  const [params, setParams] = useSearchParams()
  const [catalog, setCatalog] = useState<TaaCatalog | null>(null)
  const decisionParam = params.get('decision') ?? params.get('run') ?? (!params.size ? readAllocationJourney().taaRunId : undefined)
  const [baselineId, setBaselineId] = useState(decisionParam ? '' : params.get('baseline') ?? readAllocationJourney().baselineId ?? '')
  const [baseline, setBaseline] = useState<TaaBaseline | null>(null)
  const [request, setRequest] = useState<TaaPreviewRequest | null>(null)
  const [tab, setTab] = useState<Tab>('views')
  const [loading, setLoading] = useState(true)
  const [loadingBaseline, setLoadingBaseline] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [preview, setPreview] = useState<TaaPreview | null>(null)
  const [decision, setDecision] = useState<TaaDecision | null>(null)
  const [decisionName, setDecisionName] = useState('')
  const [saving, setSaving] = useState(false)
  const [runs, setRuns] = useState<HistoricalRegimeRun[]>([])
  const [regime, setRegime] = useState<HistoricalRegimeRun | null>(null)
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
    }).catch(error => { if (!controller.signal.aborted) setError(fail(error, '资产配置目录读取失败。')) }).finally(() => { if (!controller.signal.aborted) setLoading(false) })
    return () => controller.abort()
  }, [])

  function invalidate() {
    version.current += 1
    setPreview(null); setDecision(null); setExperiments(previous => previous.map(item => ({ scenario: item.scenario }))); setPreflight(null); setBusy(false); setError(''); setNotice('')
  }
  function showDecision(value: TaaDecision) {
    setBaseline(value.preview.baseline); setRequest(value.preview.request); setPreview(value.preview); setDecision(value)
    setDecisionName(value.name); setDecisionNote(value.note ?? ''); setExperiments(value.scenarios ?? []); setTab('versions')
    updateAllocationJourney({ universeId: value.preview.baseline.universe_snapshot_id ?? undefined, allocationName: value.preview.baseline.alloc_name, baselineId: value.preview.baseline.id, taaRunId: value.id })
  }
  useEffect(() => {
    if (loading || !catalogRef.current) return
    const controller = new AbortController()
    if (!baselineId) { setLoadingBaseline(false); return () => controller.abort() }
    invalidate(); setBaseline(null); setRequest(null)
    setLoadingBaseline(true)
    getTaaBaseline(baselineId, controller.signal).then(value => {
      if (controller.signal.aborted) return
      if (value.id !== baselineId) throw new Error('读取的 SAA 版本与所选版本不一致，请重新选择。')
      if (openingDecision.current?.preview.baseline.id === value.id) { showDecision(openingDecision.current); openingDecision.current = null; return }
      const copied = restore.current?.baseline_id === value.id ? restore.current : null
      const draft = readAllocationDraft<{ request: TaaPreviewRequest; tab: Tab; name: string; note: string; scenarios: TaaScenario[] }>(`taa:${value.id}`)
      const restored = validDraftRequest(draft?.request, value.id) && draft && ['views', 'backtest', 'scenarios', 'versions'].includes(draft.tab) && Array.isArray(draft.scenarios) && draft.scenarios.every(item => item && typeof item.name === 'string' && ['shock', 'historical'].includes(item.kind)) ? draft : null
      restore.current = null; setBaseline(value); setRequest(copied ?? restored?.request ?? initialRequest(value, catalogRef.current, platformRef.current))
      setExperiments((restored?.scenarios ?? []).map(scenario => ({ scenario }))); setDecisionName(restored?.name ?? `${value.name} · 战术方案`); setDecisionNote(restored?.note ?? '')
      if (restored) { setTab(restored.tab); setNotice('已恢复本次研究草稿。输入和情景假设已保留，请重新计算确认结果。') }
      updateAllocationJourney({ universeId: value.universe_snapshot_id ?? undefined, allocationName: value.alloc_name, baselineId: value.id })
    }).catch(error => { if (!controller.signal.aborted) setError(fail(error, '基准读取失败。')) }).finally(() => { if (!controller.signal.aborted) setLoadingBaseline(false) })
    return () => controller.abort()
  }, [baselineId, loading])
  useEffect(() => {
    const id = decisionParam; if (!id || decision?.id === id) return
    const controller = new AbortController(); const token = version.current
    getTaaDecision(id, controller.signal).then(value => {
      if (controller.signal.aborted || token !== version.current) return
      if (value.id !== id) throw new Error('保存版本与链接不一致。')
      openingDecision.current = value
      if (baselineId === value.preview.baseline.id) { showDecision(value); openingDecision.current = null }
      else setBaselineId(value.preview.baseline.id)
    }).catch(error => { if (!controller.signal.aborted && token === version.current) setError(fail(error, '保存版本读取失败。')) })
    return () => controller.abort()
  }, [decisionParam])
  useEffect(() => {
    if (request?.signal_mode !== 'regime') return
    let active = true; setRegimeError('')
    listHistoricalRegimeRuns().then(value => { if (active) setRuns(value.filter(eligible)) }).catch(error => { if (active) setRegimeError(fail(error, '情景版本读取失败。')) })
    return () => { active = false }
  }, [request?.signal_mode])
  useEffect(() => {
    let active = true; setRegime(null); setRegimeError('')
    if (!request?.regime_run_id || request.signal_mode !== 'regime') { setLoadingRegime(false); return () => { active = false } }
    setLoadingRegime(true); const id = request.regime_run_id
    getHistoricalRegimeRun(id).then(value => {
      if (!active) return
      const summary = runs.find(item => item.id === id)
      if (value.id !== id || !eligible(value) || (summary && (value.content_hash !== summary.content_hash || value.definition_revision !== summary.definition_revision))) throw new Error('情景详情与发布版本不一致，或不满足实时使用条件。')
      setRegime(value)
      setRequest(previous => previous?.regime_run_id === id ? { ...previous, state_tilts: previous.state_tilts ?? Object.fromEntries(value.states.map(state => [state.id, Object.fromEntries(Object.keys(previous.manual_tilts).map(asset => [asset, 0]))])) } : previous)
    }).catch(error => { if (active) setRegimeError(fail(error, '情景详情读取失败。')) }).finally(() => { if (active) setLoadingRegime(false) })
    return () => { active = false }
  }, [request?.regime_run_id, request?.signal_mode, runs])

  const pitConflict = request && platformDay && request.as_of > platformDay ? `本次研究日 ${request.as_of} 晚于当前 PIT 截止 ${platformDay}。已保存版本仅供查看；调整日期后才能重新计算或交接。` : ''
  const allocationCoverage = catalog?.allocations.find(item => item.alloc_name === baseline?.alloc_name)?.coverage
  const canAlignPit = Boolean(pitConflict && allocationCoverage && platformDay && allocationCoverage.start_date < platformDay)
  const expired = Boolean(preview && preview.recommendation.expires_on < today())
  const selectedCandidate = preview?.candidates.find(item => item.id === preview.selected_id)
  const applyBlocked = pitConflict || (preflight?.quality.status === 'blocked' ? '数据存在数量级跳变，请先返回大类检查产品。' : selectedCandidate?.validation_feasible === false ? '所选候选在独立验证区间超出约束，请先复核。' : preview?.recommendation.turnover_from_current != null && preview.recommendation.turnover_from_current > preview.request.max_turnover + 1e-8 ? '从参考持仓调整到当前方案的换手超过上限。' : '')
  const manualSum = Object.values(request?.manual_tilts ?? {}).reduce((sum, value) => sum + value, 0)
  const invalidNumeric = request ? [request.lookback, request.max_abs_tilt, request.transaction_cost_bps, request.risk_penalty, request.max_tracking_error, request.max_turnover, request.confidence_floor, request.max_signal_age_days, request.review_days, ...Object.values(request.manual_tilts), ...Object.values(request.current_weights ?? {}), ...Object.values(request.state_tilts ?? {}).flatMap(Object.values)].some(value => !Number.isFinite(value)) : true
  const formIssue = !request ? '' : pitConflict ? pitConflict : invalidNumeric ? '请填写完整的数值，负偏离用负数表示。' : !(request.start_date < request.train_end_date && request.train_end_date < request.end_date && request.end_date <= request.as_of) ? '日期应满足：回测开始 < 训练结束 < 回测结束 ≤ 决策日期。' : request.signal_mode === 'manual' && Math.abs(manualSum) > 1e-8 ? '手动偏离合计须为 0；增配资金需要来自减配资产。' : request.current_weights && Math.abs(Object.values(request.current_weights).reduce((sum, value) => sum + value, 0) - 1) > 1e-8 ? '参考持仓权重合计须为 100%。' : request.signal_mode === 'regime' && (!regime || loadingRegime) ? '请先选择并核验一个已发布的实时情景版本。' : ''
  useEffect(() => {
    if (previousPlatform.current === platformDay) return
    previousPlatform.current = platformDay
    if (!request && !decision) return
    if (decision) { version.current += 1; setPreflight(null); setNotice('当前平台数据口径已变化；此处保留历史保存版本，按原研究日解释结果。') }
    else { invalidate(); setNotice('平台数据口径已变化。草稿输入和情景假设仍保留，请检查日期并重新计算。') }
  }, [platformDay])
  useEffect(() => {
    if (!request || !baseline || formIssue) { setChecking(false); setPreflight(null); return }
    const controller = new AbortController()
    setChecking(true); setPreflight(null); setCheckError('')
    const timeout = window.setTimeout(() => {
      preflightTaa(request, controller.signal).then(value => { if (!controller.signal.aborted) setPreflight(value) })
        .catch(caught => { if (!controller.signal.aborted) setCheckError(fail(caught, '研究条件检查失败，请重试。')) })
        .finally(() => { if (!controller.signal.aborted) setChecking(false) })
    }, 200)
    return () => { window.clearTimeout(timeout); controller.abort() }
  }, [fingerprint, baseline?.id, formIssue, checkRevision])
  useEffect(() => {
    if (!request || request.baseline_id !== baselineId) return
    writeAllocationDraft(`taa:${baselineId}`, { request, tab, name: decisionName, note: decisionNote, scenarios: experiments.map(item => item.scenario) })
  }, [fingerprint, baselineId, tab, decisionName, decisionNote, experiments])
  useEffect(() => {
    if (params.size) return
    const journey = readAllocationJourney()
    if (journey.taaRunId) setParams({ decision: journey.taaRunId }, { replace: true })
  }, [])
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
        if (result.baseline.id !== baseline.id) throw new Error('计算结果引用了不同 SAA 版本，已停止展示。')
        setPreview(result); setTab('backtest')
      }
    } catch (error) { if (mounted.current && token === version.current) setError(fail(error, '战术配置研究未完成。')) }
    finally { if (mounted.current && token === version.current) setBusy(false) }
  }
  async function saveDecision() {
    if (!preview || !decisionName.trim() || saving || formIssue || checking || !preflight) return
    const token = version.current; const requestedSave = saveFingerprint; setSaving(true); setError('')
    try {
      const value = await saveTaaDecision({ request: preview.request, preview_hash: preview.preview_hash, name: decisionName.trim(), note: decisionNote, scenarios: experiments.filter(item => item.result).map(item => item.scenario) })
      if (mounted.current && (token !== version.current || requestedSave !== latestSave.current)) { setCatalog(previous => ({ ...(previous ?? emptyCatalog), decisions: [value, ...(previous?.decisions ?? []).filter(item => item.id !== value.id)] })); setNotice('此前条件的版本已保存到历史列表；当前新草稿仍保留，请另行保存。') }
      if (mounted.current && token === version.current && requestedSave === latestSave.current) { setDecision(value); setExperiments(value.scenarios ?? []); updateAllocationJourney({ baselineId: value.preview.baseline.id, taaRunId: value.id }); setCatalog(previous => ({ ...(previous ?? emptyCatalog), decisions: [value, ...(previous?.decisions ?? []).filter(item => item.id !== value.id)] })); setParams({ decision: value.id }, { replace: true }); setNotice('研究版本已保存，后续更改不会覆盖此版本。') }
    } catch (error) { if (mounted.current && token === version.current) setError(fail(error, '版本保存失败。')) }
    finally { if (mounted.current) setSaving(false) }
  }
  async function applyDecision() {
    if (!decision || expired || saving || checking || !preflight || applyBlocked || baseline?.apply_eligible === false) return
    const token = version.current; setSaving(true); setError('')
    try {
      const value = await taaProductAllocation(decision.id)
      if (mounted.current && token === version.current) { sessionStorage.setItem('portfolioResearchImport', JSON.stringify(value)); navigate(`/pre-investment/product-allocation-timing/construction?${new URLSearchParams({ ...(value.universe_snapshot_id ? { universe: value.universe_snapshot_id } : {}), decision: decision.id })}`) }
    } catch (error) { if (mounted.current && token === version.current) setError(fail(error, '产品配置交接未完成。')) }
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
      setExperiments((value.scenarios ?? []).map(item => ({ scenario: item.scenario }))); setDecisionNote(value.note ?? ''); setNotice('已复制研究条件和情景假设，请重新计算；原版本保留。'); setTab('views')
    } catch (error) { if (mounted.current && token === version.current) setError(fail(error, '历史版本读取失败。')) }
    finally { if (mounted.current) setSaving(false) }
  }

  return <div className="min-h-screen min-w-0 max-w-full break-words bg-slate-50 pb-12 text-slate-800"><main className="mx-auto max-w-[1500px] min-w-0 space-y-4 px-3 py-4 sm:px-7">
    <header className="flex flex-col justify-between gap-3 sm:flex-row sm:items-center"><div><h1 className="mt-1 text-2xl font-semibold text-slate-950">战术资产配置</h1><p className="mt-1 text-sm leading-6 text-slate-600">判断是否偏离长期配置、偏离多少，并验证何时回归。</p></div><span className="w-fit rounded-full border border-slate-300 bg-white px-3 py-1.5 text-xs text-slate-600">{preview ? '已计算 · 可复核' : '研究工作台'}</span></header>
    <Feedback error={error} notice={notice} />
    <BaselineSetup catalog={catalog} selectedId={baselineId} loading={loading} onSelect={selectBaseline} onCreated={baselineCreated} />
    {loadingBaseline && <p role="status" className="rounded-xl bg-white p-5 text-sm text-slate-500">正在加载锁定的基准与资产范围…</p>}
    {baseline && request && <>
      <div role="tablist" aria-label="战术配置研究内容" className="grid grid-cols-2 gap-1 sm:grid-cols-4 rounded-xl border border-slate-200 bg-white p-1">{tabs.map(([id, label]) => <button key={id} type="button" role="tab" id={`taa-tab-${id}`} aria-controls={`taa-panel-${id}`} aria-selected={tab === id} tabIndex={tab === id ? 0 : -1} onKeyDown={event => { if (event.key === 'ArrowRight' || event.key === 'ArrowLeft') { event.preventDefault(); const next = (tabs.findIndex(item => item[0] === tab) + (event.key === 'ArrowRight' ? 1 : -1) + tabs.length) % tabs.length; setTab(tabs[next][0]); document.getElementById(`taa-tab-${tabs[next][0]}`)?.focus() } }} onClick={() => setTab(id)} className={`min-h-11 min-w-0 rounded-lg px-2 py-2.5 text-sm focus-visible:outline focus-visible:outline-2 focus-visible:outline-teal-700 ${tab === id ? 'bg-teal-50 font-semibold text-teal-900' : 'text-slate-600 hover:bg-slate-50'}`}>{label}</button>)}</div>
      {pitConflict && <section className="rounded-xl border border-amber-200 bg-amber-50 p-4 text-sm text-amber-900" aria-label="平台 PIT 日期冲突"><p>{pitConflict}</p><div className="mt-3 flex flex-wrap gap-2">{canAlignPit ? <button type="button" className={buttonClass} onClick={alignPitDates}>按当前 PIT 调整日期</button> : <p className="text-xs leading-5">当前大类覆盖从 {allocationCoverage?.start_date ?? '未知日期'} 开始，与当前 PIT 没有可直接采用的共同研究范围。</p>}<Link className={buttonClass} to={allocationJourneyPath('classes')}>返回大类选择数据</Link><Link className={buttonClass} to="/settings/pit-snapshots">检查平台 PIT 口径</Link></div></section>}
      <TaaResearchContext baseline={baseline} request={request} preview={preview} preflight={preflight} checking={checking} error={checkError} contextIssue={pitConflict} onRetry={() => setCheckRevision(value => value + 1)} onChange={update} onEdit={() => setTab('backtest')} />
      <div className="grid min-w-0 gap-5 xl:grid-cols-[minmax(0,1fr)_300px]">
        <section className={`${sectionClass} space-y-4`} aria-label="本次资产配置"><div className="flex flex-wrap items-start justify-between gap-3"><div><h2 className="text-lg font-semibold text-slate-950">本次准备怎么配？</h2><p className="mt-1 text-xs leading-5 text-slate-500">{baseline.name} · 方案日期 {baseline.as_of.slice(0, 10)}</p></div><span className="rounded-md bg-amber-50 px-2.5 py-1.5 text-xs text-amber-900">{preview?.data.pit.status === 'verified' ? '查看逐时点验证范围' : '研究结果 · 历史 PIT 待核验'}</span></div>
          <div className="space-y-3 sm:hidden">{baseline.assets.map(asset => <article key={asset.id} className="rounded-lg border border-slate-200 p-3"><h3 className="text-sm font-semibold">{asset.name}</h3><dl className="mt-3 grid grid-cols-3 gap-2 text-xs"><div><dt className="text-slate-500">SAA</dt><dd className="mt-1 font-medium tabular-nums">{percentText(asset.base_weight)}</dd></div><div><dt className="text-slate-500">拟议权重</dt><dd className="mt-1 font-semibold tabular-nums text-teal-900">{preview ? percentText(preview.recommendation.weights[asset.id]) : '待计算'}</dd></div><div><dt className="text-slate-500">偏离百分点</dt><dd className="mt-1 font-medium tabular-nums">{preview ? `${(preview.recommendation.tilts[asset.id] * 100).toFixed(2)}` : '—'}</dd></div></dl><p className="mt-3 text-xs leading-5 text-slate-500">允许 {percentText(asset.min_weight)}–{percentText(asset.max_weight)}；{preview?.recommendation.trade_deltas ? `持仓差额 ${points(preview.recommendation.trade_deltas[asset.id])}` : '未提供参考持仓'}</p></article>)}</div>
          <div className="hidden overflow-auto sm:block"><table className="w-full min-w-[540px] text-sm"><caption className="sr-only">SAA 与拟议战术权重，偏离以百分点表示</caption><thead className="text-xs text-slate-500"><tr><th className="px-2 py-3 text-left">资产</th><th className="px-2 py-3 text-right">SAA</th><th className="px-2 py-3 text-right">拟议权重</th><th className="px-2 py-3 text-right">相对 SAA</th><th className="px-2 py-3 text-right">参考持仓差额</th></tr></thead><tbody className="divide-y divide-slate-100">{baseline.assets.map(asset => <tr key={asset.id}><th className="px-2 py-3 text-left font-medium">{asset.name}<span className="mt-1 block text-xs font-normal text-slate-500">允许 {percentText(asset.min_weight)}–{percentText(asset.max_weight)}</span></th><td className="px-2 py-3 text-right tabular-nums">{percentText(asset.base_weight)}</td><td className="px-2 py-3 text-right font-semibold tabular-nums text-teal-900">{preview ? percentText(preview.recommendation.weights[asset.id]) : '待计算'}</td><td className="px-2 py-3 text-right tabular-nums">{preview ? points(preview.recommendation.tilts[asset.id]) : '—'}</td><td className="px-2 py-3 text-right text-xs tabular-nums">{preview?.recommendation.trade_deltas ? points(preview.recommendation.trade_deltas[asset.id]) : '未提供参考持仓'}</td></tr>)}</tbody></table></div><p className="text-xs leading-5 text-slate-500">偏离是与 SAA 的差值；参考持仓差额用于比较拟议目标，不代表实际订单。拟议权重由后端约束计算后展示。</p>
        </section>
        <aside className={`${sectionClass} space-y-4`} aria-label="本次决策摘要"><p className="text-xs font-semibold text-teal-800">本次决策</p><h2 className="text-lg font-semibold text-slate-950">{preview ? preview.recommendation.is_saa ? '维持长期配置' : '已生成偏离方案' : '先验证，再决定'}</h2><p className="text-sm leading-6 text-slate-600">{preview?.recommendation.reason ?? '选择信号与偏离边界，查看扣费后是否仍有改善。保持 SAA 始终参与比较。'}</p>{preview?.recommendation.signal_details && <details className="text-xs leading-5"><summary className="cursor-pointer font-medium text-teal-800">为什么这样调整？</summary><div className="mt-2 space-y-2">{preview.recommendation.signal_details.map(item => <div key={item.asset_id}><p className="font-medium">{baseline.assets.find(asset => asset.id === item.asset_id)?.name ?? item.asset_id} · {item.direction}</p><p>信号 {item.value == null ? '不可用' : percentText(item.value)} · {item.signal_date ?? '无可用日期'}</p><p>{item.window?.window_start && <>观察 {item.window.window_start} — {item.window.window_end}；最晚公布 {item.window.available_at}；距决策 {item.window.lag_days} 天。</>}</p><p>原始偏离 {points(item.raw_tilt)} → 约束后 {points(item.applied_tilt)}</p>{item.constraint_reason && <p className="text-amber-800">{item.constraint_reason}</p>}</div>)}</div></details>}{preview && <dl className="space-y-2 text-xs"><div className="flex justify-between gap-2"><dt className="text-slate-500">决策日期</dt><dd>{preview.request.as_of}</dd></div><div className="flex justify-between gap-2"><dt className="text-slate-500">信号日期</dt><dd>{preview.recommendation.signal_date ?? '暂无可用信号'}</dd></div><div className="flex justify-between gap-2"><dt className="text-slate-500">复核到期</dt><dd className={expired ? 'text-amber-800' : ''}>{preview.recommendation.expires_on}{expired ? ' · 已过期' : ''}</dd></div></dl>}{formIssue && <p role="status" className="rounded-lg bg-amber-50 p-3 text-xs leading-5 text-amber-900">{formIssue}</p>}<button type="button" className={`${primaryClass} w-full`} disabled={busy || checking || !preflight?.can_calculate || Boolean(formIssue)} onClick={() => void calculate()}>{busy ? '正在计算与验证…' : checking ? '正在检查研究条件…' : preview ? '重新计算研究' : '计算并比较方案'}</button><p className="text-xs leading-5 text-slate-500">使用本地真实数据。保存研究版本、进入产品配置均需显式操作。</p></aside>
      </div>

      <div role="tabpanel" id={`taa-panel-${tab}`} aria-labelledby={`taa-tab-${tab}`} className="min-w-0 space-y-5">
        {tab === 'views' && <>
          <section className={`${sectionClass} space-y-5`}><div><h2 className="text-lg font-semibold">为什么要偏离？</h2><p className="mt-1 text-sm text-slate-500">先选择判断依据，再设定调整幅度。系统不会从“牛市、熊市”等名称猜测买卖方向。</p></div><div className="grid gap-3 md:grid-cols-3">{[['momentum', '趋势规则', '用此前窗口的资产表现产生信号，逐期更新。'], ['manual', '研究员观点', '明确填写增减配，用历史回放检验敏感性。'], ['regime', '已发布市场状态', '使用已通过发布门禁的实时情景与概率。']].map(([id, label, help]) => <label key={id} className={`flex cursor-pointer items-start gap-3 rounded-xl border p-4 ${request.signal_mode === id ? 'border-teal-600 bg-teal-50/50' : 'border-slate-200'}`}><input type="radio" className="mt-1" name="taa-signal" value={id} checked={request.signal_mode === id} onChange={() => update({ signal_mode: id as TaaPreviewRequest['signal_mode'] })} /><span><span className="block text-sm font-semibold">{label}</span><span className="mt-1 block text-xs leading-5 text-slate-500">{help}</span></span></label>)}</div>
            {request.signal_mode === 'momentum' && <div className="grid gap-4 sm:grid-cols-[240px_1fr]"><Field label="趋势观察窗口（交易期）"><NumberInput className={inputClass} value={request.lookback} min={2} max={756} onValueChange={lookback => update({ lookback })} /></Field><p className="self-center rounded-lg bg-slate-50 p-4 text-sm leading-6 text-slate-600">逐期使用当时最近、完整且已公布的窗口；最新净值尚未公布时沿用最近已知窗口，超过信号有效期则回归 SAA。窗口与上限属于本次策略定义，不按验证期结果反复挑选。</p></div>}
            {request.signal_mode === 'manual' && <><p className="rounded-lg bg-amber-50 p-3 text-sm leading-6 text-amber-900">今天填写的观点并非过去已知的观点。历史结果属于假设回放，不是可证明当时能执行的 PIT 业绩。</p><div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">{baseline.assets.map(asset => <Field key={asset.id} label={`${asset.name}偏离（百分点）`} hint={`SAA 为 ${percentText(asset.base_weight)}；+5 表示增加 5 个百分点。`}><NumberInput className={inputClass} value={request.manual_tilts[asset.id] * 100} onValueChange={value => update({ manual_tilts: { ...request.manual_tilts, [asset.id]: value / 100 } })} /></Field>)}</div><p className="text-xs text-slate-500">偏离合计：{points(manualSum)}。每一份增配都要有对应资金来源。</p></>}
            {request.signal_mode === 'regime' && <div className="space-y-4"><Feedback error={regimeError} /><Field label="已发布的实时情景版本"><select className={inputClass} value={request.regime_run_id ?? ''} onChange={event => update({ regime_run_id: event.target.value, state_tilts: undefined })}><option value="">选择情景版本</option>{runs.map(run => <option key={run.id} value={run.id}>{run.name} · R{run.definition_revision}</option>)}</select></Field>{!runs.length && <p className="text-sm leading-6 text-amber-800">暂无可用状态版本：需要先通过实时可用性验证并正式发布，事后分类和未发布结果不能用于此处。<Link className="ml-1 underline" to={`/settings/scenario-algorithms?return_to=${encodeURIComponent(allocationJourneyPath('taa'))}`}>去情景算法中心验证并发布</Link></p>}{loadingRegime && <p role="status" className="text-sm text-slate-500">正在核验情景详情与发布记录…</p>}{regime && <><p className="text-xs leading-5 text-slate-500">每个状态填满全部资产，偏离合计为 0。状态概率表示分类不确定性，不是赚钱概率。</p><div className="overflow-auto"><table className="w-full min-w-[480px] text-sm"><thead><tr><th className="p-2 text-left">市场状态</th>{baseline.assets.map(asset => <th className="p-2 text-left" key={asset.id}>{asset.name}（百分点）</th>)}</tr></thead><tbody>{regime.states.map(state => <tr key={state.id}><th className="p-2 text-left font-medium">{state.label}</th>{baseline.assets.map(asset => <td className="p-2" key={asset.id}><NumberInput aria-label={`${state.label} · ${asset.name}偏离（百分点）`} className={inputClass} value={(request.state_tilts?.[state.id]?.[asset.id] ?? 0) * 100} onValueChange={value => update({ state_tilts: { ...request.state_tilts, [state.id]: { ...request.state_tilts?.[state.id], [asset.id]: value / 100 } } })} /></td>)}</tr>)}</tbody></table></div></>}</div>}
            <Field label="本次观点与复核条件" hint="记录证据、理由，以及哪些变化会让你撤回观点；随保存版本保留。"><textarea rows={3} className={inputClass} value={request.note} placeholder="例如：趋势改善支持小幅增配；若信号消失或风险超限则回到 SAA。" onChange={event => update({ note: event.target.value })} /></Field>
          </section>
          <section className={`${sectionClass} space-y-4`}><h2 className="text-lg font-semibold">偏离多大、何时回归？</h2><div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4"><Field label="单资产偏离上限（百分点）"><NumberInput className={inputClass} value={request.max_abs_tilt * 100} min={0} max={100} onValueChange={value => update({ max_abs_tilt: value / 100 })} /></Field><Field label="主动风险上限（% / 年）"><NumberInput className={inputClass} value={request.max_tracking_error * 100} min={0} max={100} onValueChange={value => update({ max_tracking_error: value / 100 })} /></Field><Field label="单期换手上限（%）"><NumberInput className={inputClass} value={request.max_turnover * 100} min={0} max={200} onValueChange={value => update({ max_turnover: value / 100 })} /></Field><Field label="多少天后复核"><NumberInput className={inputClass} value={request.review_days} min={1} max={365} onValueChange={review_days => update({ review_days })} /></Field></div><p className="text-xs leading-5 text-slate-500">同时遵守 SAA 各资产上下界。约束不满足会缩小偏离或拒绝候选；信号失效时回归 SAA。</p><details className="border-t border-slate-100 pt-3"><summary className="cursor-pointer text-sm font-medium">参考持仓与信号有效期</summary><div className="mt-4 space-y-4"><label className="flex min-h-11 items-center gap-2 text-sm"><input type="checkbox" checked={Boolean(request.current_weights)} onChange={event => update({ current_weights: event.target.checked ? zeroes(baseline) : undefined })} />填写参考持仓，比较实际需要调整的幅度</label>{request.current_weights && <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">{baseline.assets.map(asset => <Field key={asset.id} label={`${asset.name}参考持仓（%）`}><NumberInput className={inputClass} value={request.current_weights![asset.id] * 100} onValueChange={value => update({ current_weights: { ...request.current_weights, [asset.id]: value / 100 } })} /></Field>)}</div>}<div className="grid gap-4 sm:grid-cols-2"><Field label="状态置信度门槛（%）" hint="仅用于已发布市场状态；低于门槛不进行偏离。"><NumberInput className={inputClass} value={request.confidence_floor * 100} onValueChange={value => update({ confidence_floor: value / 100 })} /></Field><Field label="状态信号最长有效天数"><NumberInput className={inputClass} value={request.max_signal_age_days} onValueChange={max_signal_age_days => update({ max_signal_age_days })} /></Field></div></div></details></section>
          <button type="button" className={primaryClass} onClick={() => setTab('backtest')}>下一步：设置回测与费用 →</button>
        </>}
        {tab === 'backtest' && <><section className={`${sectionClass} space-y-4`}><h2 className="text-lg font-semibold">先用训练区间选，再到未参与选择的数据验证</h2><div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4"><Field label="回测开始"><input type="date" min={preflight?.coverage.start_date} max={request.train_end_date} className={inputClass} value={request.start_date} onChange={event => update({ start_date: event.target.value })} /></Field><Field label="训练结束"><input type="date" min={request.start_date} max={request.end_date} className={inputClass} value={request.train_end_date} onChange={event => update({ train_end_date: event.target.value })} /></Field><Field label="回测结束"><input type="date" max={request.as_of} min={request.train_end_date} className={inputClass} value={request.end_date} onChange={event => update({ end_date: event.target.value })} /></Field><Field label="决策日期" hint="只能使用此日已知的数据与已生效信号。"><input type="date" max={today()} className={inputClass} value={request.as_of} onChange={event => update({ as_of: event.target.value })} /></Field></div><div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4"><Field label="候选比较目标"><select className={inputClass} value={request.objective} onChange={event => update({ objective: event.target.value as TaaPreviewRequest['objective'] })}><option value="active_utility">兼顾净超额与主动风险</option><option value="excess_return">训练期相对净超额</option><option value="min_drawdown">训练期最大回撤</option></select></Field><Field label="单边成本（基点）" hint="10 基点 = 0.10%；SAA 与 TAA 使用相同口径。"><NumberInput className={inputClass} value={request.transaction_cost_bps} min={0} onValueChange={transaction_cost_bps => update({ transaction_cost_bps })} /></Field><Field label="主动风险惩罚系数"><NumberInput className={inputClass} value={request.risk_penalty} min={0} onValueChange={risk_penalty => update({ risk_penalty })} /></Field><Field label="是否搜索偏离强度"><select className={inputClass} value={request.search ? 'search' : 'fixed'} onChange={event => update({ search: event.target.value === 'search' })}><option value="search">训练期搜索（多档强度与零偏离）</option><option value="fixed">固定假设比较（定义强度与零偏离）</option></select></Field></div><p className="text-xs leading-5 text-slate-500">按真实共同交易日逐日回测，每日恢复目标权重并计入成本。训练结束后的数据仅做独立验证；当前数据版本与当前观点仍可能缺少完整历史 PIT 证明。</p><button type="button" className={primaryClass} disabled={busy || checking || !preflight?.can_calculate || Boolean(formIssue)} onClick={() => void calculate()}>{busy ? '正在计算与验证…' : '运行回测与候选比较'}</button></section>{preview ? <TaaPerformance result={preview} busy={busy} onSelect={id => void calculate(id)} /> : <Empty title="结果将在这里出现"><p>明确训练与验证日期，运行后查看净值、净超额、风险、换手与成本；没有数据时不会生成示例业绩。</p></Empty>}</>}
        {tab === 'scenarios' && <TaaScenarioExperiments key={baseline.id} baselineId={baseline.id} preview={preview} experiments={experiments} onChange={value => { setExperiments(value); markDraftChanged() }} onBacktest={() => setTab('backtest')} />}
        {tab === 'versions' && <><section className={`${sectionClass} space-y-4`}><h2 className="text-lg font-semibold">保存判断，并交给产品配置</h2><p className="text-sm leading-6 text-slate-500">保存 SAA、观点、回测结果、已完成情景和复核条件。交接后由产品配置确认类内产品；此处不会执行交易。</p>{preview ? <><p className="text-sm text-slate-600">已完成情景 {experiments.filter(item => item.result).length} 个；待重算 {experiments.filter(item => !item.result).length} 个。待重算假设只保留在草稿，不冒充已验证证据。</p><Field label="研究结论与复核计划" hint="说明采纳、维持 SAA 或暂不采纳的原因，以及何时重新评估。"><textarea rows={3} className={inputClass} value={decisionNote} onChange={event => { setDecisionNote(event.target.value); markDraftChanged() }} /></Field><Field label="研究版本名称"><input className={inputClass} value={decisionName} onChange={event => { setDecisionName(event.target.value); markDraftChanged() }} /></Field><div className="flex flex-wrap gap-3"><button type="button" className={primaryClass} disabled={saving || Boolean(formIssue) || checking || !preflight || !decisionName.trim() || Boolean(decision) || preflight?.quality.status === 'blocked'} onClick={() => void saveDecision()}>{saving ? '正在处理…' : decision ? '当前研究版本已保存' : '保存研究版本'}</button><button type="button" className={buttonClass} disabled={!decision || expired || saving || checking || !preflight || Boolean(applyBlocked) || baseline.apply_eligible === false} onClick={() => void applyDecision()}>带入产品配置</button></div>{expired && <p role="status" className="text-sm text-amber-800">这个决策已到复核日期。可以保存历史研究，请更新研究日期、重新计算后再带入产品配置。</p>}{applyBlocked && <p role="status" className="text-sm text-amber-800">{applyBlocked}</p>}{baseline.apply_reasons?.map(reason => <p key={reason} className="text-xs text-amber-800">{reason}</p>)}{!decision && <p className="text-xs text-slate-500">先保存当前结果，再将明确的资产权重与产品映射带入下游。</p>}</> : <Empty title="当前尚无可保存结果"><p>先计算并复核一份方案。参数变更后需要重新计算，防止保存过期结果。</p></Empty>}</section>
          <section className={`${sectionClass} space-y-4`}><h2 className="text-lg font-semibold">历史研究版本</h2>{!catalog?.decisions.length ? <p className="text-sm text-slate-500">尚未保存战术配置研究。</p> : <div className="divide-y divide-slate-100">{catalog.decisions.map(item => <div key={item.id} className="flex flex-col justify-between gap-3 py-3 sm:flex-row sm:items-center"><div><p className="text-sm font-medium">{item.name}</p><p className="mt-1 text-xs text-slate-500">保存于 {item.created_at.slice(0, 10)}{item.preview?.recommendation?.expires_on ? ` · 复核 ${item.preview.recommendation.expires_on}${item.preview.recommendation.expires_on < today() ? '（已过期）' : ''}` : ''}</p></div><div className="flex gap-2"><button type="button" className={buttonClass} disabled={saving} onClick={() => void readDecision(item.id, false)}>查看版本</button><button type="button" className={buttonClass} disabled={saving} onClick={() => void readDecision(item.id, true)}>复制为新研究</button></div></div>)}</div>}</section>
          <section className={`${sectionClass} space-y-4`}><h2 className="text-lg font-semibold">当时真的知道这些信息吗？</h2><ol className="space-y-3 text-sm leading-6 text-slate-600"><li><strong className="text-slate-900">数据可得：</strong>观测日期不等于发布日期，当前修订数据不自动代表当时版本。</li><li><strong className="text-slate-900">信号生效：</strong>使用已结束窗口，状态必须在收益区间开始前已可用；缺失、过期或低置信度时返回 SAA。</li><li><strong className="text-slate-900">训练分界：</strong>只用训练区间挑偏离强度，独立验证不参与选择；人工观点回放始终带研究限制。</li><li><strong className="text-slate-900">决策留痕：</strong>保存时锁定输入与结果，不会回写过去版本或倒签发布时点。</li></ol>{[...new Set([...(preview?.data.pit.reasons ?? baseline.pit.reasons), ...(preview?.warnings ?? [])])].map(reason => <p key={reason} className="rounded-lg bg-amber-50 p-3 text-xs leading-5 text-amber-900">{reason}</p>)}<details className="border-t border-slate-100 pt-3"><summary className="cursor-pointer text-sm font-medium">数据来源、逐日权重与计算审计</summary><div className="mt-3 space-y-3 text-xs leading-5 text-slate-600"><p className="break-all">SAA 版本：{baseline.id} · 校验：{baseline.content_hash}</p><p>基准冻结时间：{baseline.created_at}</p>{preview && <><p className="break-all">结果校验：{preview.preview_hash}</p><pre className="max-h-64 overflow-auto whitespace-pre-wrap break-all rounded-lg bg-slate-50 p-3">{JSON.stringify({ data: preview.data.lineage, timing_and_selection: preview.audit, execution: preview.execution }, null, 2)}</pre><div className="max-h-64 overflow-auto"><table className="w-full min-w-[450px] text-xs"><thead><tr><th className="p-2 text-left">日期</th>{baseline.assets.map(asset => <th key={asset.id} className="p-2 text-right">{asset.name}</th>)}<th className="p-2 text-right">换手</th></tr></thead><tbody>{preview.weight_path.map(row => <tr key={row.date}><td className="p-2">{row.date}</td>{baseline.assets.map(asset => <td className="p-2 text-right" key={asset.id}>{percentText(row.weights[asset.id])}</td>)}<td className="p-2 text-right">{percentText(row.turnover)}</td></tr>)}</tbody></table></div></>}</div></details></section>
        </>}
      </div>
      {preview && tab !== 'versions' && <div className="flex flex-col items-start justify-between gap-3 rounded-xl border border-teal-200 bg-teal-50 px-5 py-4 sm:flex-row sm:items-center"><p className="text-sm text-teal-900">完成风险复核后，保存本次判断与回归条件。</p><button type="button" className={buttonClass} onClick={() => setTab('versions')}>保存与交接 →</button></div>}
    </>}
  </main></div>
}

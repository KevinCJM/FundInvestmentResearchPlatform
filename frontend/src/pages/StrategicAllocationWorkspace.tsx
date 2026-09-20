import { useEffect, useRef, useState } from 'react'
import { Link, useNavigate, useSearchParams } from 'react-router-dom'
import { allocationJourneyPath, readAllocationJourney, updateAllocationJourney, useAllocationDraft } from '../app/allocationJourney'
import { useResearchDay } from '../app/ResearchContext'
import { Button } from '../components/ui'
import { Empty, Feedback, Field, inputClass, percentText, sectionClass, today } from '../components/risk-models/ResearchUI'
import PolicyCandidates from '../components/strategic-allocation/PolicyCandidates'
import { MeanUncertaintySummary } from '../components/strategic-allocation/MeanUncertaintyFields'
import { riskBudgetError } from '../components/strategic-allocation/RiskBudgetEditor'
import { GoalCandidateSummary } from '../components/investment-mandate/MandateResults'
import { amountText, objectiveLabels } from '../components/investment-mandate/model'
import {
  getCma, getStrategicCatalog, previewPolicy, publishPolicy,
  type CmaVersion, type PolicyCandidate, type PolicyPreview,
  type PolicyRequest, type StrategicCatalog, type StrategicBaseline,
} from '../services/strategicAllocation'
import { getTaaBaseline } from '../services/tacticalAllocation'
import LtcmaSelection, { cmaSelectionReason } from '../components/strategic-allocation/LtcmaSelection'
import { useLtcmaText } from '../components/ltcma/shared'
import { useI18n } from '../i18n/runtime'
import MultiCmaSelection, { multiCmaWeightIssue } from '../components/strategic-allocation/MultiCmaSelection'
import CompatibilityResults from '../components/strategic-allocation/CompatibilityResults'
import CrossModelResults from '../components/strategic-allocation/CrossModelResults'
import type { CmaReference } from '../services/strategicAllocation'

interface Draft {
  mandateId: string; allocationName: string; strategicUniverseId?: string; implementationMappingId?: string
  settings: Omit<PolicyRequest, 'mandate_id' | 'cma_id' | 'mode' | 'cma_refs'>
  policyName: string; reason: string; savedCmaId?: string | null
  mode?: 'single' | 'parameter_average' | 'compatible_all_models'; cmaRefs?: CmaReference[]
}
const steps = ['研究范围', '选择 LTCMA', '政策比较', '确认与交接']
const historicalLabPath = (allocationName: string) => {
  const journey = readAllocationJourney()
  const query = new URLSearchParams()
  if (allocationName) query.set('alloc', allocationName)
  if (journey.universeId) query.set('universe', journey.universeId)
  return `/pre-investment/saa/allocation-lab${query.size ? `?${query}` : ''}`
}

export default function StrategicAllocationWorkspace() {
  const [params] = useSearchParams()
  const baselineId = params.get('baseline')
  const allocationName = params.get('strategic_universe') ? '' : params.get('alloc') ?? readAllocationJourney().allocationName ?? ''
  const mandateId = params.get('mandate') ?? readAllocationJourney().mandateId ?? ''
  const strategicUniverseId = params.get('strategic_universe') ?? (!params.get('alloc') ? readAllocationJourney().strategicUniverseId ?? '' : '')
  const mappingId = params.get('mapping') ?? ''
  const cmaId = params.get('cma') ?? ''
  if (baselineId) return <SavedPolicy key={baselineId} id={baselineId} />
  return <StrategicEditor key={`${allocationName}:${mandateId}:${strategicUniverseId}:${mappingId}:${cmaId}`} initialAllocation={allocationName} initialMandate={mandateId} initialUniverse={strategicUniverseId} initialMapping={mappingId} initialCma={cmaId} />
}

/** A return from TAA reads its exact immutable baseline, never a browser draft. */
function SavedPolicy({ id }: { id: string }) {
  const { s } = useI18n()
  const [baseline, setBaseline] = useState<StrategicBaseline | null>(null)
  const [error, setError] = useState('')
  const [retry, setRetry] = useState(0)
  useEffect(() => {
    const controller = new AbortController()
    setError(''); setBaseline(null)
    getTaaBaseline(id, controller.signal).then(value => {
      if (controller.signal.aborted) return
      if (value.id !== id) throw new Error('读取的政策与所选版本不一致，请重新读取。')
      setBaseline(value)
    }).catch(reason => { if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : '政策读取失败。') })
    return () => controller.abort()
  }, [id, retry])
  const query = new URLSearchParams()
  if (baseline) {
    if (baseline.policy?.assumptions?.strategic_universe_id) {
      query.set('strategic_universe', baseline.policy.assumptions.strategic_universe_id)
      const frozenMapping = baseline.implementation_mapping_id ?? baseline.policy.assumptions?.implementation_mapping_id
      if (frozenMapping) query.set('mapping', frozenMapping)
    } else if (baseline.alloc_name) query.set('alloc', baseline.alloc_name)
    if (baseline.policy) query.set('mandate', baseline.policy.mandate_id)
    if (baseline.universe_snapshot_id) query.set('universe', baseline.universe_snapshot_id)
  }
  return <section className="mx-auto max-w-6xl space-y-5 p-4 sm:p-6" aria-label="已保存的 SAA 政策">
    <h1 className="text-2xl font-semibold text-slate-900">已保存的 SAA 政策</h1>
    <Feedback error={error} />
    {error ? <Button onClick={() => setRetry(value => value + 1)}>重新读取政策</Button> : !baseline ? <p role="status">正在读取政策版本…</p> : <>
      <h2 className="text-lg font-semibold">{baseline.name}</h2>
      <p className="text-sm text-slate-600">研究日：{baseline.as_of}。以下为不可变历史记录，当前应用资格须在 TAA 另行核验。</p>
      <dl className="grid gap-3 sm:grid-cols-3">{baseline.assets.map(asset => <div key={asset.id} className="rounded-lg bg-slate-50 p-3"><dt className="text-sm text-slate-600">{asset.name}</dt><dd className="mt-1 text-lg font-semibold tabular-nums">{percentText(asset.base_weight)}</dd></div>)}</dl>
      {baseline.policy && <><p className="text-sm leading-6 text-slate-700">采纳理由：{baseline.policy.reason}</p><details className="text-sm"><summary className="cursor-pointer">目标与假设版本</summary><p className="break-all">目标：{baseline.policy.mandate_id}；CMA：{baseline.policy.cma_id ?? s(baseline.policy.mode === 'compatible_all_models' ? 'multiCma.common' : 'multiCma.average')}</p></details></>}
      {baseline.policy?.uncertainty_model && <MeanUncertaintySummary value={baseline.policy.uncertainty_model} />}
      {baseline.policy?.multi_cma && <section className="space-y-3" aria-label={s(baseline.policy?.mode === 'compatible_all_models' ? 'multiCma.commonFrozen' : 'multiCma.frozen')}><h3 className="font-semibold">{s(baseline.policy?.mode === 'compatible_all_models' ? 'multiCma.commonFrozen' : 'multiCma.frozen')}</h3><p className="text-sm text-slate-600">{s(baseline.policy.mode === 'compatible_all_models' ? 'multiCma.commonHint' : 'multiCma.hint')}</p><div className="divide-y divide-slate-200">{baseline.policy.multi_cma.sources.map(source => <p key={source.cma_id} className="break-words py-2 text-sm tabular-nums"><Link className="text-accent-700 underline" to={`/pre-investment/ltcma/${encodeURIComponent(source.cma_id)}`}>{source.name}</Link> · {source.as_of} · {baseline.policy?.mode === 'compatible_all_models' ? s('multiCma.required') : percentText(source.weight)}</p>)}</div></section>}
      {baseline.policy?.compatibility && <CompatibilityResults evidence={baseline.policy.compatibility} />}
      {baseline.policy?.selection?.cross_model_results && <CrossModelResults common={baseline.policy.mode === 'compatible_all_models'} rows={baseline.policy.selection.cross_model_results} />}
      <div className="flex flex-wrap gap-4 text-sm"><Link className="inline-flex min-h-10 items-center text-accent-800 underline" to={`/pre-investment/product-allocation-timing?source=${encodeURIComponent(id)}`}>{s('implementation.handoff')}</Link><Link className="inline-flex min-h-10 items-center text-accent-800 underline" to={`/pre-investment/saa/policy?${query}`}>使用此目标建立新政策</Link>{baseline.implementation_status === 'incomplete' ? <p className="text-sm text-amber-800">映射尚未完整，不能进入TAA；请补齐映射后建立新政策。</p> : <Link className="inline-flex min-h-10 items-center text-accent-800 underline" to={`/pre-investment/taa?baseline=${encodeURIComponent(id)}`}>返回此政策的 TAA 研究</Link>}</div>
    </>}
  </section>
}

function StrategicEditor({ initialAllocation, initialMandate, initialUniverse, initialMapping, initialCma }: { initialAllocation: string; initialMandate: string; initialUniverse: string; initialMapping: string; initialCma: string }) {
  const { t } = useLtcmaText()
  const { s } = useI18n()
  const navigate = useNavigate()
  const platformDay = useResearchDay()
  const [editor, setEditor] = useAllocationDraft<Draft>(`strategic-policy:${initialAllocation}:${initialMandate}${initialUniverse ? `:universe:${initialUniverse}:${initialMapping}` : ''}`, () => ({
    mandateId: initialMandate, allocationName: initialAllocation, strategicUniverseId: initialUniverse, implementationMappingId: initialMapping, savedCmaId: initialCma || null,
    settings: { constraints: {}, group_limits: [], uncertainty_penalty: 1, candidate_count: 2000, seed: 42 }, policyName: '', reason: '',
  }))
  const [catalog, setCatalog] = useState<StrategicCatalog | null>(null)
  const [reload, setReload] = useState(0)
  const [loading, setLoading] = useState(true)
  const [step, setStep] = useState(0)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [cmaVersion, setCmaVersion] = useState<CmaVersion | null>(null)
  const [cmaVersions, setCmaVersions] = useState<CmaVersion[]>([])
  const [policyPreview, setPolicyPreview] = useState<PolicyPreview | null>(null)
  const [candidate, setCandidate] = useState<PolicyCandidate | null>(null)
  const [savedPolicy, setSavedPolicy] = useState<StrategicBaseline | null>(null)
  const generation = useRef(0), operation = useRef<AbortController | null>(null)
  const attemptedCma = useRef(''), initializedScope = useRef('')
  const attemptedMulti = useRef('')
  const previousClock = useRef(platformDay)
  const heading = useRef<HTMLHeadingElement>(null)
  const mandate = catalog?.mandates.find(value => value.id === editor.mandateId)
  const universe = catalog?.strategic_universes?.find(value => value.id === editor.strategicUniverseId)
  const mapping = catalog?.implementation_maps?.find(value => value.id === editor.implementationMappingId && value.definition.strategic_universe_id === universe?.id)
  const allocation = catalog?.allocations.find(value => value.alloc_name === editor.allocationName)
  const mode = editor.mode ?? 'single'
  const refs = editor.cmaRefs ?? []
  const multiComplete = refs.length > 0 && refs.every(ref => cmaVersions.some(version => version.id === ref.cma_id && version.content_hash === ref.content_hash))
  const selectedVersions = mode !== 'single' ? cmaVersions.filter(version => refs.some(ref => ref.cma_id === version.id)) : cmaVersion ? [cmaVersion] : []
  const assumptionsReady = mode !== 'single' ? multiComplete && !multiCmaWeightIssue(refs, mode === 'compatible_all_models') : Boolean(cmaVersion)
  const primaryCma = selectedVersions[0]
  const draft = primaryCma?.effective_assumptions ?? primaryCma?.definition ?? null
  const selectionContext = { allocationName: editor.allocationName, strategicUniverseId: editor.strategicUniverseId,
    implementationMappingId: editor.implementationMappingId, mandate: mandate?.definition,
    cutoff: platformDay === undefined ? undefined : platformDay && platformDay < today() ? platformDay : today() }
  const policyRequest: PolicyRequest = { ...editor.settings, mandate_id: editor.mandateId,
    ...(mode !== 'single' ? { mode, cma_id: null, cma_refs: refs } : { cma_id: cmaVersion?.id ?? '' }),
    implementation_mapping_id: primaryCma?.definition.schema_version === '2.0' && editor.strategicUniverseId ? editor.implementationMappingId || null : undefined }
  const issue = selectedVersions.map(version => cmaSelectionReason(version.definition, selectionContext)).find(Boolean)
  const dateIssue = platformDay === undefined ? '平台知识截止日尚未确认，请先恢复 PIT 设置；暂不能计算或保存。' : issue ? t(issue) : ''

  useEffect(() => {
    const controller = new AbortController()
    setLoading(true)
    getStrategicCatalog(controller.signal).then(value => { if (!controller.signal.aborted) setCatalog(value) })
      .catch(reason => { if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : '长期配置目录读取失败。') })
      .finally(() => { if (!controller.signal.aborted) setLoading(false) })
    return () => controller.abort()
  }, [reload])
  useEffect(() => () => { generation.current += 1; operation.current?.abort() }, [])
  useEffect(() => { heading.current?.focus() }, [step])
  useEffect(() => {
    if (previousClock.current === platformDay) return
    previousClock.current = platformDay
    generation.current += 1
    setBusy(false)
    setError('')
    if (savedPolicy) {
      setNotice('当前展示已保存的历史政策；知识截止日已变化，不代表该政策在新截止日下可用。')
      return
    }
    setPolicyPreview(null)
    setCandidate(null)
    setStep(current => current > 2 ? 2 : current)
    setNotice('知识截止日已变化，未保存的政策候选已失效，请核对日期后重新比较。')
  }, [platformDay, savedPolicy, cmaVersion])
  useEffect(() => {
    if ((!allocation && !universe) || !mandate) return
    const names = (universe?.definition.assets ?? allocation!.assets).map(asset => asset.id)
    const scope = `${universe?.id ?? allocation?.alloc_name}:${mandate.id}`
    if (initializedScope.current === scope) return
    const changed = Boolean(initializedScope.current)
    initializedScope.current = scope
    setEditor(current => ({ ...current,
      policyName: current.policyName || `${universe?.name ?? allocation!.alloc_name} · 长期政策`.slice(0, 120),
      settings: !changed && Object.keys(current.settings.constraints).length && Object.keys(current.settings.constraints).every(id => names.includes(id)) ? current.settings : {
        ...current.settings, risk_budget: null, group_limits: [],
        constraints: Object.fromEntries(names.map(id => [id, { min_weight: mandate.definition.asset_limits?.[id]?.min_weight ?? 0,
          max_weight: mandate.definition.asset_limits?.[id]?.max_weight ?? 1,
          max_abs_tilt: mandate.definition.max_tracking_error === 0 ? 0 : mandate.definition.asset_limits?.[id]?.max_abs_tilt ?? .1 }])),
      } }))
  }, [allocation, universe, mandate])
  useEffect(() => {
    const wanted = initialCma || editor.savedCmaId
    if (mode !== 'single' || !catalog || !mandate || !wanted || cmaVersion || attemptedCma.current === wanted) return
    attemptedCma.current = wanted
    loadAssumptions(wanted)
  }, [catalog, mandate, initialCma, editor.savedCmaId, cmaVersion, mode])
  useEffect(() => {
    const signature = JSON.stringify(refs.map(ref => [ref.cma_id, ref.content_hash]))
    if (mode === 'single' || !catalog || !mandate || !refs.length || multiComplete || attemptedMulti.current === signature) return
    loadMultiple(refs)
  }, [mode, catalog, mandate, refs, multiComplete])

  function invalidate(assumptions = false) {
    generation.current += 1; operation.current?.abort()
    setBusy(false); setError(''); setNotice(''); setPolicyPreview(null); setCandidate(null); setSavedPolicy(null)
    if (assumptions) { setCmaVersion(null); setCmaVersions([]); attemptedMulti.current = ''; setEditor(current => ({ ...current, savedCmaId: null, cmaRefs: [] })) }
  }
  function changeScope(patch: Partial<Pick<Draft, 'mandateId' | 'allocationName' | 'strategicUniverseId' | 'implementationMappingId'>>) {
    invalidate('allocationName' in patch || 'strategicUniverseId' in patch)
    updateAllocationJourney({
      ...('mandateId' in patch ? { mandateId: patch.mandateId || undefined } : {}),
      ...('allocationName' in patch ? { allocationName: patch.allocationName || undefined } : {}),
      ...('strategicUniverseId' in patch ? { strategicUniverseId: patch.strategicUniverseId || undefined } : {}),
      ...('implementationMappingId' in patch ? { implementationMappingId: patch.implementationMappingId || undefined } : {}),
    })
    setEditor(current => ({ ...current, ...patch }))
  }
  async function run<T,>(work: (signal: AbortSignal) => Promise<T>, consume: (result: T) => void) {
    operation.current?.abort(); const controller = new AbortController(); operation.current = controller
    const token = ++generation.current
    setBusy(true); setError(''); setNotice('')
    try { const result = await work(controller.signal); if (!controller.signal.aborted && generation.current === token) consume(result) }
    catch (reason) { if (!controller.signal.aborted && generation.current === token) setError(reason instanceof Error ? reason.message : '计算失败，请检查输入。') }
    finally { if (generation.current === token) setBusy(false) }
  }
  function loadAssumptions(id: string) {
    invalidate(true)
    if (!id) return
    attemptedCma.current = id
    void run(signal => getCma(id, signal), value => {
      const reason = cmaSelectionReason(value.definition, selectionContext)
      if (value.id !== id || reason) throw new Error(reason ? t(reason) : '读取的 LTCMA 版本不一致。')
      setEditor(current => ({ ...current, savedCmaId: value.id }))
      setCmaVersion(value); setStep(2)
      setNotice('已引用保存的假设；政策比较时服务端会重新检查来源及引用资格。')
    })
  }
  function loadMultiple(wanted: CmaReference[]) {
    attemptedMulti.current = JSON.stringify(wanted.map(ref => [ref.cma_id, ref.content_hash]))
    void run(signal => Promise.all(wanted.map(ref => getCma(ref.cma_id, signal))), versions => {
      versions.forEach((version, index) => {
        const reason = cmaSelectionReason(version.definition, selectionContext)
        if (version.id !== wanted[index].cma_id || version.content_hash !== wanted[index].content_hash || reason) throw new Error(reason ? t(reason) : s('multiCma.reloadMismatch'))
      })
      setCmaVersions(versions)
      setNotice(s('multiCma.ready'))
    })
  }
  function changeMode(next: NonNullable<Draft['mode']>) {
    invalidate(); attemptedMulti.current = ''
    const selected = refs.length ? refs : cmaVersion ? [{ cma_id: cmaVersion.id, content_hash: cmaVersion.content_hash, weight: 1 }] : []
    if (!refs.length && cmaVersion) setCmaVersions([cmaVersion])
    setEditor(current => ({ ...current, mode: next,
      cmaRefs: selected.map(ref => ({ ...ref, weight: next === 'compatible_all_models' ? null : ref.weight ?? (selected.length === 1 ? 1 : 0) })),
      settings: { ...current.settings, ...(next === 'compatible_all_models' ? { risk_budget: null } : {}),
        compatibility_objective: undefined, solver_max_iterations: undefined,
        uncertainty_set: 'box', uncertainty_confidence: null, uncertainty_approximation_acknowledged: false } }))
    setStep(1)
  }
  function changeReferences(next: CmaReference[]) {
    invalidate()
    setCmaVersions(current => current.filter(version => next.some(ref => ref.cma_id === version.id)))
    setEditor(current => ({ ...current, cmaRefs: next }))
  }
  function addReference(id: string) {
    if (refs.some(ref => ref.cma_id === id) || refs.length >= 20) return
    invalidate()
    void run(signal => getCma(id, signal), version => {
      const reason = cmaSelectionReason(version.definition, selectionContext)
      if (version.id !== id || reason) throw new Error(reason ? t(reason) : s('multiCma.reloadMismatch'))
      setCmaVersions(current => [...current, version])
      setEditor(current => ({ ...current, cmaRefs: [...(current.cmaRefs ?? []), { cma_id: version.id, content_hash: version.content_hash, weight: mode === 'compatible_all_models' ? null : current.cmaRefs?.length ? 0 : 1 }] }))
    })
  }
  function adopt() {
    if (dateIssue || !assumptionsReady || !candidate || candidate.available === false || mode === 'compatible_all_models' && candidate.all_models_pass !== true || !policyPreview || candidate.goal_check?.within_limits === false || !editor.reason.trim() || !editor.policyName.trim()) return
    void run(signal => publishPolicy(policyRequest, policyPreview.preview_hash, candidate.id, editor.policyName, editor.reason, signal), value => {
      setSavedPolicy(value)
      updateAllocationJourney({ mandateId: editor.mandateId, strategicUniverseId: editor.strategicUniverseId || undefined, implementationMappingId: editor.implementationMappingId || undefined, allocationName: value.alloc_name ?? undefined, universeId: value.universe_snapshot_id ?? undefined, baselineId: value.id, taaRunId: undefined })
      setCatalog(current => current && ({ ...current, policies: [value, ...current.policies] }))
      setNotice('长期政策已确认。目标、假设、产品映射和预算已锁定；这是研究采纳，不是外部审批。')
    })
  }

  const selection = <div className="min-w-0 space-y-4">
    <Field label={s('multiCma.mode')}><select className={inputClass} value={mode} onChange={event => changeMode(event.target.value as 'single' | 'parameter_average' | 'compatible_all_models')}>
      <option value="single">{s('multiCma.single')}</option><option value="parameter_average">{s('multiCma.average')}</option><option value="compatible_all_models">{s('multiCma.common')}</option>
    </select></Field>
    {mode === 'single' ? <LtcmaSelection items={catalog?.assumptions ?? []} selected={cmaVersion} context={selectionContext} onSelect={loadAssumptions} busy={busy} mandateId={editor.mandateId} />
      : <MultiCmaSelection common={mode === 'compatible_all_models'} items={catalog?.assumptions ?? []} refs={refs} versions={cmaVersions} context={selectionContext} busy={busy} onAdd={addReference} onChange={changeReferences} onContinue={() => setStep(2)} onRetry={() => loadMultiple(refs)} />}
  </div>

  return <div className="mx-auto max-w-6xl space-y-5 p-4 sm:p-6">
    <header><h1 className="text-2xl font-semibold text-slate-900">长期政策配置</h1><p className="mt-2 text-sm leading-6 text-slate-600">从投资目标与前瞻假设出发，选定长期资金比例，再研究是否需要战术偏离。历史有效前沿属于独立实验工具，不替代前瞻 CMA。</p><Link className="mt-2 inline-flex min-h-10 items-center text-sm font-medium text-accent-800 underline" to={historicalLabPath(editor.allocationName)}>打开历史有效前沿与策略回测</Link></header>
    <nav aria-label="长期配置步骤" className="grid gap-2 border-b border-slate-200 pb-4 sm:grid-cols-4">{steps.map((label, i) => <button key={label} type="button" aria-current={step === i ? 'step' : undefined} disabled={(i > 0 && ((!allocation && !universe) || !mandate)) || (i > 1 && !assumptionsReady) || (i > 2 && !candidate)} onClick={() => setStep(i)} className={`min-h-11 rounded-lg px-3 py-2 text-left text-sm focus-visible:outline focus-visible:outline-accent-600 disabled:cursor-not-allowed disabled:opacity-50 ${step === i ? 'bg-accent-50 font-semibold text-accent-900' : 'text-slate-600 hover:bg-slate-50'}`}>{i + 1}. {label}</button>)}</nav>
    <h2 ref={heading} tabIndex={-1} className="text-lg font-semibold outline-none">{steps[step]}</h2>
    <Feedback error={error || dateIssue} notice={notice} />
    {busy && <p role="status" className="text-sm text-slate-600">正在核验当前输入…</p>}
    {loading ? <p role="status" className="text-sm text-slate-600">正在读取目标与分类…</p> : !catalog ? <Empty title="目录暂时不可用"><Button onClick={() => setReload(value => value + 1)}>重试读取</Button></Empty> : <>
      {step === 0 && <section className={`${sectionClass} space-y-5`} aria-label="目标与大类来源">
        <div className="grid gap-4 sm:grid-cols-2">
          <Field label="投资目标版本"><select className={inputClass} value={editor.mandateId} onChange={e => changeScope({ mandateId: e.target.value })}><option value="">选择已保存目标</option>{catalog.mandates.map(value => <option key={value.id} value={value.id}>{value.name} · {value.definition.currency} · {value.definition.horizon_years} 年</option>)}</select></Field>
          <Field label="已保存的大类配置"><select className={inputClass} value={editor.allocationName} onChange={e => changeScope({ allocationName: e.target.value, strategicUniverseId: '', implementationMappingId: '' })}><option value="">选择大类与代理产品</option>{catalog.allocations.map(value => <option key={value.alloc_name} value={value.alloc_name}>{value.alloc_name}</option>)}</select></Field>
        </div>
        <div className="grid gap-4 sm:grid-cols-2"><Field label="或选择独立战略范围"><select className={inputClass} value={editor.strategicUniverseId ?? ''} onChange={e => changeScope({ strategicUniverseId: e.target.value, implementationMappingId: '', allocationName: '' })}><option value="">使用原产品大类路径</option>{catalog.strategic_universes?.map(v => <option key={v.id} value={v.id}>{v.name} · {v.definition.as_of}</option>)}</select></Field>
          {universe && <Field label="实施映射（可稍后完成）"><select className={inputClass} value={editor.implementationMappingId ?? ''} onChange={e => changeScope({ implementationMappingId: e.target.value })}><option value="">暂无映射，先做纯前瞻研究</option>{catalog.implementation_maps?.filter(m => m.definition.strategic_universe_id === universe.id).map(m => <option key={m.id} value={m.id}>{m.name} · {m.implementation_status === 'complete' ? '完整覆盖' : '存在缺口'}</option>)}</select></Field>}</div>
        {universe && <p className="text-sm text-amber-800">战略资产：{universe.definition.assets.map(a => a.name).join('、')}。{mapping?.implementation_status === 'complete' ? '已选择完整映射，交接时会复核有效期与真实数据。' : '尚有实施缺口，可以进行完整CMA/SAA研究；补齐映射前不能进入TAA或产品应用。'}</p>}
        <Link className="inline-flex min-h-10 items-center text-sm text-accent-800 underline" to="/pre-investment/product-pool?scope=strategic">定义战略范围或补齐实施映射</Link>
        <div className="flex flex-wrap gap-4 text-sm"><Link className="inline-flex min-h-10 items-center text-accent-800 underline" to="/pre-investment/objectives">建立或复核投资目标</Link><Link className="inline-flex min-h-10 items-center text-accent-800 underline" to="/pre-investment/saa/asset-classes">构建与检查大类</Link></div>
        {mandate && <p className="rounded-lg bg-slate-50 p-3 text-sm leading-6 text-slate-700">目标：{objectiveLabels[mandate.definition.objective_kind ?? 'absolute_return']}{mandate.definition.funding_target ? `，期末保有${amountText(mandate.definition.funding_target.amount)} ${mandate.definition.currency}` : mandate.definition.funding_plan ? `，期末保有${amountText(mandate.definition.funding_plan.terminal_target)} ${mandate.definition.currency}` : mandate.definition.benchmark ? `，相对${mandate.definition.benchmark.name}预期超额至少${percentText(mandate.definition.benchmark.target_excess_return)}` : `，预期年收益至少${percentText(mandate.definition.target_return)}`}；预期波动不超过 {percentText(mandate.definition.max_volatility)}，现金底线 {percentText(mandate.definition.effective_cash_reserve_weight ?? mandate.definition.min_cash_weight ?? 0)}；{mandate.definition.review_date ? `计划于 ${mandate.definition.review_date} 复核。` : '未设置强制复核日期。'}</p>}
        {allocation && <p className="text-sm leading-6 text-slate-600">大类：{allocation.assets.map(asset => asset.name).join('、')}。后续保留原产品映射，不自动从名称推断经济角色。</p>}
        <Button tone="primary" disabled={!mandate || (!allocation && !universe)} onClick={() => setStep(1)}>选择已确认 LTCMA</Button>
        {selection}
      </section>}
      {step === 1 && selection}
      {step === 2 && draft && assumptionsReady && <PolicyCandidates value={policyRequest} assets={draft.assets.map(asset => asset.id)} result={policyPreview} busy={busy} compareDisabled={Boolean(dateIssue)}
        meanCovarianceAvailable={Boolean(cmaVersion?.model_result?.mean_estimation_covariance || cmaVersion?.model_result?.posterior_mean_covariance)}
        onChange={value => { invalidate(); const { mandate_id: _mandate, cma_id: _cma, mode: _mode, cma_refs: _refs, ...settings } = value; setEditor(current => ({ ...current, settings })) }}
        onCompare={() => { if (assumptionsReady && !dateIssue && !riskBudgetError(draft.assets.map(a => a.id), policyRequest.risk_budget)) void run(signal => previewPolicy(policyRequest, signal), value => { setPolicyPreview(value); setCandidate(null) }) }}
        onSelect={value => { setCandidate(value); setStep(3) }} />}
      {step === 3 && candidate && policyPreview && <section className={`${sectionClass} space-y-5`} aria-label="政策采纳确认">
        <h2 className="text-lg font-semibold">这份长期政策是否符合你的判断？</h2><p className="text-sm leading-6 text-slate-600">{candidate.name} · 预期年收益 {percentText(candidate.metrics.expected_return)} · 预期年波动 {percentText(candidate.metrics.volatility)}。这些是冻结假设下的计算，不是业绩预测保证。</p>
        <GoalCandidateSummary candidate={candidate} />
        {candidate.cross_model_results && <CrossModelResults common={mode === 'compatible_all_models'} rows={candidate.cross_model_results} />}
        {policyPreview.current_application_eligible === false && <div className="space-y-2 text-sm text-amber-800"><p>此政策可以保存研究，当前产品应用仍有未满足的条件。</p>{policyPreview.application_blockers?.length ? policyPreview.application_blockers.map(reason => <p key={reason}>{reason}</p>) : <p>请复核政策与映射的适用日期，已到复核日的版本不能直接应用。</p>}</div>}
        <dl className="grid gap-3 sm:grid-cols-3">{Object.entries(candidate.weights).map(([asset, weight]) => <div key={asset} className="rounded-lg bg-slate-50 p-3"><dt className="text-sm text-slate-600">{asset}</dt><dd className="mt-1 text-lg font-semibold tabular-nums">{percentText(weight)}</dd></div>)}</dl>
        <Field label="政策版本名称"><input className={inputClass} maxLength={120} value={editor.policyName} disabled={busy} onChange={e => { setSavedPolicy(null); setEditor(current => ({ ...current, policyName: e.target.value })) }} /></Field>
        <Field label="采纳理由与复核关注点" hint="记录为何采用、哪些假设变化会触发复核。"><textarea rows={3} className={inputClass} value={editor.reason} disabled={busy} onChange={e => { setSavedPolicy(null); setEditor(current => ({ ...current, reason: e.target.value })) }} /></Field>
        <p className="text-xs leading-5 text-slate-600">确认后生成不可变 SAA 基线，TAA 继承目标、资产映射、权重及约束。政策再平衡约定与 TAA 日频研究回测并非同一执行规则。</p>
        <div className="flex flex-wrap gap-3"><Button tone="primary" disabled={busy || Boolean(dateIssue) || Boolean(savedPolicy) || candidate.goal_check?.within_limits === false || editor.reason.trim().length < 5 || !editor.policyName.trim()} onClick={adopt}>{savedPolicy ? '长期政策已确认' : '确认采用此长期政策'}</Button>
          {savedPolicy && <Link className="inline-flex min-h-10 items-center text-accent-800 underline" to={`/pre-investment/product-allocation-timing?source=${encodeURIComponent(savedPolicy.id)}`}>{s('implementation.handoff')}</Link>}
          {savedPolicy && <Button tone="primary" disabled={Boolean(editor.strategicUniverseId && mapping?.implementation_status !== 'complete')} onClick={() => navigate(allocationJourneyPath('taa', { ...readAllocationJourney(), baselineId: savedPolicy.id, taaRunId: undefined }))}>进入 TAA，研究是否需要偏离 →</Button>}</div>
        {savedPolicy && editor.strategicUniverseId && mapping?.implementation_status !== 'complete' && <p className="text-sm text-amber-800">已保存纯前瞻政策；尚缺完整映射。回到投资范围补齐映射，重新确认适用政策；2.0 LTCMA 无需因实施映射变化重新生成，旧版本保持只读。</p>}
      </section>}
      <details className={`${sectionClass} text-sm`}><summary className="cursor-pointer font-medium">已保存政策与历史研究工具</summary><div className="mt-3 space-y-2">{catalog.policies.length ? catalog.policies.map(value => <Link key={value.id} className="flex min-h-10 items-center text-accent-800 underline" to={`/pre-investment/saa/policy?baseline=${encodeURIComponent(value.id)}`}>{value.name} · {value.as_of}</Link>) : <p className="text-slate-600">尚无已确认政策。</p>}<Link className="inline-flex min-h-10 items-center text-accent-800 underline" to={`/pre-investment/saa/allocation-lab?alloc=${encodeURIComponent(editor.allocationName)}`}>历史有效前沿、风险预算与策略回测</Link><p className="text-xs leading-5 text-slate-600">历史实验保留原功能；不能将历史最优权重直接当成长期前瞻政策。</p></div></details>
    </>}
  </div>
}

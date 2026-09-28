import { researchMessage } from '../i18n/researchMessages'
import { cmaScopeFacts, scopeFacts, cmaPairReason, cmaChoice } from '../services/cmaCompatibility'
import { useEffect, useId, useRef, useState } from 'react'
import { Link, useNavigate, useSearchParams } from 'react-router-dom'
import { allocationJourneyPath, readAllocationJourney, updateAllocationJourney, useAllocationDraft } from '../app/allocationJourney'
import { useResearchDay } from '../app/ResearchContext'
import { Button, ErrorPanel, LoadingPanel } from '../components/ui'
import { Feedback, Field, inputClass, percentText, sectionClass, today } from '../components/risk-models/ResearchUI'
import SavedPolicyDetails from '../components/strategic-allocation/SavedPolicyDetails'
import PolicyCandidates from '../components/strategic-allocation/PolicyCandidates'
import PolicyFrontier from '../components/strategic-allocation/PolicyFrontier'
import { MeanUncertaintySummary } from '../components/strategic-allocation/MeanUncertaintyFields'
import { riskBudgetError } from '../components/strategic-allocation/RiskBudgetEditor'
import { GoalCandidateSummary } from '../components/investment-mandate/MandateResults'
import ScopeMandateSummary from '../components/strategic-scope/ScopeMandateSummary'
import {
  getCma, getStrategicCatalog, previewPolicy, publishPolicy,
  type CmaVersion, type PolicyCandidate, type PolicyPreview,
  type PolicyRequest, type StrategicCatalog, type StrategicBaseline,
} from '../services/strategicAllocation'
import { getTaaBaseline } from '../services/tacticalAllocation'
import LtcmaSelection, { cmaSelectionReason } from '../components/strategic-allocation/LtcmaSelection'
import { cmaMethodText, useLtcmaText } from '../components/ltcma/shared'
import { isStatisticalCma } from '../services/cmaModelTypes'
import { useI18n, systemText } from '../i18n/runtime'
import MultiCmaSelection, { multiCmaWeightIssue } from '../components/strategic-allocation/MultiCmaSelection'
import CompatibilityResults from '../components/strategic-allocation/CompatibilityResults'
import CrossModelResults from '../components/strategic-allocation/CrossModelResults'
import ResearchScopeSelection from '../components/strategic-allocation/ResearchScopeSelection'
import type { CmaReference } from '../services/strategicAllocation'
import { ltcma, ltcmaSaaIssue } from '../services/ltcma'

interface Draft {
  mandateId: string; allocationName: string; strategicUniverseId?: string; implementationMappingId?: string
  settings: Omit<PolicyRequest, 'mandate_id' | 'cma_id' | 'mode' | 'cma_refs'>
  policyName: string; reason: string; savedCmaId?: string | null
  mode?: 'single' | 'parameter_average' | 'compatible_all_models'; cmaRefs?: CmaReference[]
  needsModeChoice?: boolean
}
const POLICY_REASON_MIN = 5
const POLICY_REASON_MAX = 2000
const historicalLabPath = (allocationName: string) => {
  const journey = readAllocationJourney()
  const query = new URLSearchParams()
  if (allocationName) query.set('alloc', allocationName)
  if (journey.universeId) query.set('universe', journey.universeId)
  return `/pre-investment/saa/allocation-lab${query.size ? `?${query}` : ''}`
}

export default function StrategicAllocationWorkspace() {
  useI18n()
  const [params] = useSearchParams()
  const baselineId = params.get('baseline')
  const newToken = params.get('new') ?? ''
  // 地址栏是唯一事实来源；裸路径进来就按未选择渲染，续接由续接条显式发起。
  const allocationName = params.get('strategic_universe') ? '' : params.get('alloc') ?? ''
  const mandateId = params.get('mandate') ?? ''
  const strategicUniverseId = params.get('strategic_universe') ?? ''
  const mappingId = params.get('mapping') ?? ''
  const cmaIds = params.getAll('cma').filter(Boolean)
  if (baselineId) return <SavedPolicy key={baselineId} id={baselineId} />
  return <StrategicEditor key={`${newToken}:${allocationName}:${mandateId}:${strategicUniverseId}:${mappingId}:${JSON.stringify(cmaIds)}`} newToken={newToken} initialAllocation={allocationName} initialMandate={mandateId} initialUniverse={strategicUniverseId} initialMapping={mappingId} initialCmas={cmaIds} />
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
      if (value.id !== id) throw new Error(systemText('preInvestment.strategicAllocationWorkspace.theLoadedPolicyDiffersFromTheSelected'))
      setBaseline(value)
    }).catch(reason => { if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : systemText('preInvestment.strategicAllocationWorkspace.unableToLoadThePolicy')) })
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
  return <section className="min-w-0 space-y-5 text-slate-900" aria-label={systemText('preInvestment.strategicAllocationWorkspace.savedSaaPolicy')}>
    <Link className="inline-flex min-h-10 items-center text-sm font-medium text-accent-700 underline" to="/pre-investment/saa">{s('saaCenter.back')}</Link>
    {error ? <ErrorPanel message={error} action={<Button onClick={() => setRetry(value => value + 1)}>{systemText('preInvestment.strategicAllocationWorkspace.reloadPolicy')}</Button>} /> : !baseline ? <LoadingPanel text={systemText('preInvestment.strategicAllocationWorkspace.loadingPolicyVersion')} /> : <SavedPolicyDetails baseline={baseline} newPolicyHref={`/pre-investment/saa/policy?${query}`} />}

  </section>
}

function StrategicEditor({ initialAllocation, initialMandate, initialUniverse, initialMapping, initialCmas, newToken }: { newToken: string; initialAllocation: string; initialMandate: string; initialUniverse: string; initialMapping: string; initialCmas: string[] }) {
const steps = [systemText('preInvestment.strategicAllocationWorkspace.researchScope'), systemText('preInvestment.strategicAllocationWorkspace.selectLtcma'), systemText('preInvestment.strategicAllocationWorkspace.comparePolicies'), systemText('preInvestment.strategicAllocationWorkspace.confirmAndHandOff')]
  const { t } = useLtcmaText()
  const { s } = useI18n()
  const navigate = useNavigate()
  const formId = useId()
  const platformDay = useResearchDay()
  const initialCma = initialCmas.length === 1 ? initialCmas[0] : ''
  const incomingMultiple = initialCmas.length > 1
  const [editor, setEditor] = useAllocationDraft<Draft>(`strategic-policy:${initialAllocation}:${initialMandate}${initialUniverse ? `:universe:${initialUniverse}:${initialMapping}` : ''}${initialCmas.length ? `:cmas:${JSON.stringify(initialCmas)}` : ''}${newToken ? `:new:${newToken}` : ''}`, () => ({
    mandateId: initialMandate, allocationName: initialAllocation, strategicUniverseId: initialUniverse, implementationMappingId: initialMapping, savedCmaId: initialCma || null, ...(incomingMultiple ? { needsModeChoice: true } : {}),
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
  // Candidate comparison is its own page inside step 3: one frontier chart per page.
  const [showCandidates, setShowCandidates] = useState(false)
  const [candidate, setCandidate] = useState<PolicyCandidate | null>(null)
  const [savedPolicy, setSavedPolicy] = useState<StrategicBaseline | null>(null)
  const generation = useRef(0), operation = useRef<AbortController | null>(null)
  const initialCmaHandled = useRef(false), initializedScope = useRef('')
  const attemptedMulti = useRef('')
  const previousClock = useRef(platformDay)
  const heading = useRef<HTMLHeadingElement>(null)
  const mandate = catalog?.mandates.find(value => value.id === editor.mandateId)
  const universe = catalog?.strategic_universes?.find(value => value.id === editor.strategicUniverseId)
  const allocation = catalog?.allocations.find(value => value.alloc_name === editor.allocationName)
  const mode = editor.mode ?? 'single'
  const refs = editor.cmaRefs ?? []
  const multiComplete = refs.length > 0 && refs.every(ref => cmaVersions.some(version => version.id === ref.cma_id && version.content_hash === ref.content_hash))
  const selectedVersions = mode !== 'single' ? cmaVersions.filter(version => refs.some(ref => ref.cma_id === version.id)) : cmaVersion ? [cmaVersion] : []
  const assumptionsReady = !editor.needsModeChoice && (mode !== 'single' ? multiComplete && !multiCmaWeightIssue(refs, mode === 'compatible_all_models') : Boolean(cmaVersion))
  const primaryCma = selectedVersions[0]
  const draft = primaryCma?.effective_assumptions ?? primaryCma?.definition ?? null
  const assetLabels = Object.fromEntries(primaryCma?.source_snapshot.assets.map(asset => [asset.id, asset.name || asset.id]) ?? [])
  const selectionContext = { allocationName: editor.allocationName, strategicUniverseId: editor.strategicUniverseId,
    implementationMappingId: editor.implementationMappingId, scopeFacts: scopeFacts(universe?.definition), mandate: mandate?.definition,
    cutoff: platformDay === undefined ? undefined : platformDay && platformDay < today() ? platformDay : today() }
  const policyRequest: PolicyRequest = { ...editor.settings, mandate_id: editor.mandateId,
    ...(mode !== 'single' ? { mode, cma_id: null, cma_refs: refs } : { cma_id: cmaVersion?.id ?? '' }),
    implementation_mapping_id: primaryCma?.definition.schema_version === '2.0' && editor.strategicUniverseId ? editor.implementationMappingId || null : undefined }
  const issue = selectedVersions.map(version => cmaSelectionReason({ ...version.definition, scope_facts: cmaScopeFacts(version) }, selectionContext)).find(Boolean)
    || selectedVersions.slice(1).map(version => cmaPairReason(cmaChoice(selectedVersions[0]), cmaChoice(version))).find(Boolean)
  const dateIssue = platformDay === undefined ? systemText('preInvestment.strategicAllocationWorkspace.thePlatformKnowledgeCutoffIsUnconfirmedRestore') : issue ? t(issue) : ''
  const requiredText = (key: string, values: Record<string, string | number> = {}) => s(`saaRequired.${key}`, values)
  const scopeIssue = [!mandate && requiredText('objective'), !allocation && !universe && requiredText('scope')].filter(Boolean).join(' ')
  const reasonLength = Array.from(editor.reason.trim()).length
  const fieldIssues = [
    !editor.policyName.trim() && requiredText('name'),
    reasonLength < POLICY_REASON_MIN && requiredText('reasonShort', { min: POLICY_REASON_MIN, remaining: POLICY_REASON_MIN - reasonLength }),
    reasonLength > POLICY_REASON_MAX && requiredText('reasonLong', { max: POLICY_REASON_MAX }),
  ].filter(Boolean).join(' ')
  // The visible explanation, disabled state and click guard share one decision.
  const adoptionIssue = busy ? requiredText('saving') : savedPolicy ? requiredText('saved') : dateIssue
    || (!assumptionsReady ? requiredText('cma') : '')
    || (!candidate || !policyPreview ? requiredText('candidate') : '')
    || (candidate?.available === false || mode === 'compatible_all_models' && candidate?.all_models_pass !== true ? requiredText('candidateUnavailable') : '')
    || (candidate?.goal_check?.within_limits === false ? requiredText('goalFailed') : '')
    || (candidate?.return_check?.within_limits === false ? requiredText('returnFailed') : '')
    || fieldIssues
  const stepIssue = (index: number) => index > 0 && scopeIssue || index > 1 && !assumptionsReady && requiredText('cma') || index > 2 && !candidate && requiredText('candidate') || ''

  useEffect(() => {
    const controller = new AbortController()
    setLoading(true); setError(''); setCatalog(null)
    getStrategicCatalog(controller.signal, editor.strategicUniverseId).then(value => { if (!controller.signal.aborted) setCatalog(value) })
      .catch(reason => { if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : systemText('preInvestment.strategicAllocationWorkspace.unableToLoadTheLongTermAllocation')) })
      .finally(() => { if (!controller.signal.aborted) setLoading(false) })
    return () => controller.abort()
  }, [reload])
  useEffect(() => () => { generation.current += 1; operation.current?.abort() }, [])
  useEffect(() => { heading.current?.focus() }, [step, showCandidates])
  useEffect(() => {
    if (previousClock.current === platformDay) return
    previousClock.current = platformDay
    generation.current += 1
    operation.current?.abort()
    if (incomingMultiple && editor.needsModeChoice && !refs.length) initialCmaHandled.current = false
    setBusy(false)
    setError('')
    if (savedPolicy) {
      setNotice(systemText('preInvestment.strategicAllocationWorkspace.thisIsASavedHistoricalPolicyA'))
      return
    }
    setPolicyPreview(null)
    setCandidate(null)
    setStep(current => current > 2 ? 2 : current)
    setNotice(systemText('preInvestment.strategicAllocationWorkspace.theKnowledgeCutoffChangedUnsavedPolicyCandidates'))
  }, [platformDay, savedPolicy, cmaVersion])
  useEffect(() => {
    if ((!allocation && !universe) || !mandate) return
    const names = (universe?.definition.assets ?? allocation!.assets).map(asset => asset.id)
    const scope = `${universe?.id ?? allocation?.alloc_name}:${mandate.id}`
    if (initializedScope.current === scope) return
    const changed = Boolean(initializedScope.current)
    initializedScope.current = scope
    setEditor(current => ({ ...current,
      policyName: current.policyName || systemText('preInvestment.strategicAllocationWorkspace.longTermPolicy', { p0: universe?.name ?? allocation!.alloc_name }).slice(0, 120),
      settings: !changed && Object.keys(current.settings.constraints).length && Object.keys(current.settings.constraints).every(id => names.includes(id)) ? current.settings : {
        ...current.settings, risk_budget: null, group_limits: [],
        // 目标授权与范围大类边界取更严者，与后端 _constraints 合并口径一致。
        constraints: Object.fromEntries(names.map(id => { const scoped = universe?.definition.assets.find(asset => asset.id === id)?.weight_limits; return [id, {
          min_weight: Math.max(mandate.definition.asset_limits?.[id]?.min_weight ?? 0, scoped?.min_weight ?? 0),
          max_weight: Math.min(mandate.definition.asset_limits?.[id]?.max_weight ?? 1, scoped?.max_weight ?? 1),
          max_abs_tilt: mandate.definition.max_tracking_error === 0 ? 0 : mandate.definition.asset_limits?.[id]?.max_abs_tilt ?? .1 }] })),
      } }))
  }, [allocation, universe, mandate])
  useEffect(() => {
    if (!incomingMultiple || !catalog || !mandate || platformDay === undefined || initialCmaHandled.current) return
    // A restored user choice supersedes the entry URL, including switching to single or clearing all.
    if (!editor.needsModeChoice && (mode === 'single' || !refs.length)) return
    initialCmaHandled.current = true
    if (refs.length) { setStep(1); return }
    loadIncomingCmas()
  }, [catalog, mandate, incomingMultiple, platformDay])
  useEffect(() => {
    const wanted = editor.savedCmaId
    if (editor.needsModeChoice || mode !== 'single' || !catalog || !mandate || !wanted || cmaVersion || initialCmaHandled.current) return
    loadAssumptions(wanted)
  }, [catalog, mandate, initialCma, editor.savedCmaId, cmaVersion, mode])
  useEffect(() => {
    const signature = JSON.stringify(refs.map(ref => [ref.cma_id, ref.content_hash]))
    if (mode === 'single' && !editor.needsModeChoice || !catalog || !mandate || !refs.length || multiComplete || attemptedMulti.current === signature) return
    loadMultiple(refs)
  }, [mode, catalog, mandate, refs, multiComplete])

  function invalidate(assumptions = false) {
    generation.current += 1; operation.current?.abort()
    setBusy(false); setError(''); setNotice(''); setPolicyPreview(null); setCandidate(null); setSavedPolicy(null)
    if (assumptions) {
      initialCmaHandled.current = true
      setCmaVersion(null); setCmaVersions([]); attemptedMulti.current = ''
      setEditor(current => ({ ...current, savedCmaId: null, cmaRefs: [], needsModeChoice: false }))
    }
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
    catch (reason) { if (!controller.signal.aborted && generation.current === token) setError(reason instanceof Error ? reason.message : systemText('preInvestment.strategicAllocationWorkspace.calculationFailedCheckTheInputs')) }
    finally { if (generation.current === token) setBusy(false) }
  }
  function loadAssumptions(id: string) {
    invalidate(true)
    // 选定的假设是本次研究的上游状态：顶部流程条据此点亮 03，回到 SAA 时带 ?cma= 自动载入。
    if (!id) { updateAllocationJourney({ ltcmaId: undefined }); return }
    void run(signal => getCma(id, signal), value => {
      const reason = cmaSelectionReason({ ...value.definition, scope_facts: cmaScopeFacts(value) }, selectionContext)
      if (value.id !== id || reason) throw new Error(reason ? t(reason) : systemText('preInvestment.strategicAllocationWorkspace.theLoadedLtcmaVersionDoesNotMatch'))
      setEditor(current => ({ ...current, savedCmaId: value.id }))
      updateAllocationJourney({ ltcmaId: value.id })
      setCmaVersion(value); setStep(2)
      setNotice(systemText('preInvestment.strategicAllocationWorkspace.savedAssumptionsAreReferencedTheServerWill'))
    })
  }
  function loadIncomingCmas() {
    if (initialCmas.length > 20 || new Set(initialCmas).size !== initialCmas.length) { setError(t('handoffLimit')); return }
    void run(signal => Promise.all(initialCmas.map(id => ltcma.view(id, signal))), views => {
      views.forEach(({ version, retired }, index) => {
        const researchIssue = ltcmaSaaIssue(version)
        if (researchIssue) throw new Error(researchIssue)
        const reason = cmaSelectionReason({ ...version.definition, retired, scope_facts: cmaScopeFacts(version) }, selectionContext)
        if (version.id !== initialCmas[index] || reason) throw new Error(reason ? t(reason) : s('multiCma.reloadMismatch'))
      })
      const versions = views.map(view => view.version)
      setCmaVersions(versions)
      setEditor(current => ({ ...current, needsModeChoice: true, cmaRefs: versions.map(version => ({ cma_id: version.id, content_hash: version.content_hash, weight: null })) }))
      updateAllocationJourney({ ltcmaId: versions[0].id, ltcmaIds: versions.map(version => version.id) })
      setStep(1)
    })
  }
  function loadMultiple(wanted: CmaReference[]) {
    attemptedMulti.current = JSON.stringify(wanted.map(ref => [ref.cma_id, ref.content_hash]))
    void run(signal => Promise.all(wanted.map(ref => getCma(ref.cma_id, signal))), versions => {
      versions.forEach((version, index) => {
        const reason = cmaSelectionReason({ ...version.definition, scope_facts: cmaScopeFacts(version) }, selectionContext)
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
    setEditor(current => ({ ...current, mode: next, needsModeChoice: false,
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
    updateAllocationJourney({ ltcmaId: next[0]?.cma_id, ltcmaIds: next.map(ref => ref.cma_id) })
  }
  function addReference(id: string) {
    if (refs.some(ref => ref.cma_id === id) || refs.length >= 20) return
    invalidate()
    void run(signal => getCma(id, signal), version => {
      const reason = cmaSelectionReason({ ...version.definition, scope_facts: cmaScopeFacts(version) }, selectionContext)
      if (version.id !== id || reason) throw new Error(reason ? t(reason) : s('multiCma.reloadMismatch'))
      setCmaVersions(current => [...current, version])
      updateAllocationJourney({ ltcmaId: refs[0]?.cma_id ?? version.id, ltcmaIds: [...refs.map(ref => ref.cma_id), version.id] })
      setEditor(current => ({ ...current, cmaRefs: [...(current.cmaRefs ?? []), { cma_id: version.id, content_hash: version.content_hash, weight: mode === 'compatible_all_models' ? null : current.cmaRefs?.length ? 0 : 1 }] }))
    })
  }
  function adopt() {
    if (adoptionIssue || !candidate || !policyPreview) return
    void run(signal => publishPolicy(policyRequest, policyPreview.preview_hash, candidate.id, editor.policyName, editor.reason, signal), value => {
      setSavedPolicy(value)
      updateAllocationJourney({ mandateId: editor.mandateId, strategicUniverseId: editor.strategicUniverseId || undefined, implementationMappingId: editor.implementationMappingId || undefined, allocationName: value.alloc_name ?? undefined, universeId: value.universe_snapshot_id ?? undefined, ltcmaId: primaryCma?.id, ltcmaIds: selectedVersions.map(version => version.id), baselineId: value.id, taaRunId: undefined })
      setCatalog(current => current && ({ ...current, policies: [value, ...current.policies] }))
      setNotice(systemText('preInvestment.strategicAllocationWorkspace.saaIsConfirmedContinueResearchingTacticalAsset'))
    })
  }

  const selection = <div className="min-w-0 space-y-4">
    {editor.needsModeChoice && <div className="space-y-2 rounded-lg border border-accent-200 bg-accent-50 p-4">
      <p className="font-semibold">{t('selectedCmas', { count: refs.length || initialCmas.length })}</p>
      <p className="text-sm leading-6 text-slate-700">{t('chooseCmaUse')}</p>
      {cmaVersions.map(version => <p key={version.id} className="break-words text-sm text-slate-700">{version.name} · {cmaMethodText(t, version.definition.model?.method ?? 'manual', isStatisticalCma(version.definition.model) ? version.definition.model.window?.kind : undefined)}</p>)}
    </div>}
    <Field required label={s('multiCma.mode')}><select required className={inputClass} value={editor.needsModeChoice ? '' : mode} disabled={editor.needsModeChoice && !multiComplete} onChange={event => changeMode(event.target.value as 'single' | 'parameter_average' | 'compatible_all_models')}>
      {editor.needsModeChoice && <option value="" disabled>{t('chooseCmaMode')}</option>}
      {!editor.needsModeChoice && <option value="single">{s('multiCma.single')}</option>}<option value="parameter_average">{s('multiCma.average')}</option><option value="compatible_all_models">{s('multiCma.common')}</option>
    </select></Field>
    {editor.needsModeChoice ? <>
      <p className="text-sm leading-6 text-slate-600">{t('averageUseHint')}</p><p className="text-sm leading-6 text-slate-600">{t('commonUseHint')}</p>
      {error && !refs.length && mandate && <Button disabled={busy} onClick={loadIncomingCmas}>{t('retry')}</Button>}
      {refs.length > 0 && !multiComplete && <Button disabled={busy} onClick={() => loadMultiple(refs)}>{t('retry')}</Button>}
      <Link className="inline-flex min-h-10 items-center text-sm text-accent-700 underline" to="/pre-investment/ltcma">{t('back')}</Link>
    </> : mode === 'single' ? <LtcmaSelection items={catalog?.assumptions ?? []} selected={cmaVersion} context={selectionContext} onSelect={loadAssumptions} busy={busy} mandateId={editor.mandateId} />
      : <MultiCmaSelection common={mode === 'compatible_all_models'} items={catalog?.assumptions ?? []} refs={refs} versions={cmaVersions} context={selectionContext} busy={busy} onAdd={addReference} onChange={changeReferences} onContinue={() => setStep(2)} onRetry={() => loadMultiple(refs)} />}
  </div>

  const catalogFailed = !loading && !catalog
  return <div className="mx-auto max-w-6xl space-y-5 p-4 sm:p-6">
    <Link className="inline-flex min-h-10 items-center text-sm font-medium text-accent-700 underline" to="/pre-investment/saa">{s('saaCenter.back')}</Link>
    <header><h1 className="text-2xl font-semibold text-slate-900">{s('saaScope.title')}</h1><p className="mt-2 max-w-5xl text-sm leading-6 text-slate-600">{s('saaScope.description')}</p><p className="mt-1 max-w-5xl text-sm leading-6 text-slate-600">{s('saaScope.workflow')}</p><Link className="mt-2 inline-flex min-h-10 items-center text-sm font-medium text-accent-800 underline" to={historicalLabPath(editor.allocationName)}>{systemText('preInvestment.strategicAllocationWorkspace.openHistoricalEfficientFrontiersAndStrategyBacktests')}</Link></header>
    {mandate && <ScopeMandateSummary mandate={mandate} loading={false} blockedReason="" backHref="/pre-investment/objectives" researchDay={primaryCma?.definition.as_of ?? platformDay} />}
    <nav aria-label={systemText('preInvestment.strategicAllocationWorkspace.longTermAllocationSteps')} className="grid gap-2 border-b border-slate-200 pb-4 sm:grid-cols-4">{steps.map((label, i) => <button key={label} type="button" aria-label={`${i + 1}. ${label}`} aria-current={step === i ? 'step' : undefined} aria-describedby={stepIssue(i) ? `${formId}-step-${i}` : undefined} disabled={Boolean(stepIssue(i))} onClick={() => setStep(i)} className={`min-h-11 rounded-lg px-3 py-2 text-left text-sm focus-visible:outline focus-visible:outline-accent-600 disabled:cursor-not-allowed disabled:bg-slate-100 ${step === i ? 'bg-accent-50 font-semibold text-accent-900' : 'text-slate-600 hover:bg-slate-50'}`}>{i + 1}. {label}{stepIssue(i) && <span id={`${formId}-step-${i}`} className="mt-1 block text-xs font-normal leading-5">{stepIssue(i)}</span>}</button>)}</nav>
    <h2 ref={heading} tabIndex={-1} className="text-lg font-semibold outline-none">{steps[step]}</h2>
    {/* 目录没读出来时下面整块都出不来，失败原因交给中间的错误态，这里只留研究日这类仍然成立的提示。 */}
    <Feedback error={catalogFailed ? dateIssue : error || dateIssue} notice={notice} />
    {initialCmas.length > 0 && !editor.mandateId && <p className="text-sm leading-6 text-amber-800">{t('handoffNeedsMandate')}</p>}
    {busy && <p role="status" className="text-sm text-slate-600">{systemText('preInvestment.strategicAllocationWorkspace.checkingCurrentInputs')}</p>}
    {loading ? <LoadingPanel text={systemText('preInvestment.strategicAllocationWorkspace.loadingObjectivesAndClassifications')} /> : !catalog ? <ErrorPanel message={error || systemText('preInvestment.strategicAllocationWorkspace.catalogTemporarilyUnavailable')} action={<Button onClick={() => setReload(value => value + 1)}>{systemText('preInvestment.strategicAllocationWorkspace.retryLoading')}</Button>} /> : <>
      {step === 0 && <section className={`${sectionClass} space-y-5`} aria-label={systemText('preInvestment.strategicAllocationWorkspace.objectiveAndAssetClassSources')}>
        <ResearchScopeSelection catalog={catalog} allocationName={editor.allocationName} strategicUniverseId={editor.strategicUniverseId ?? ''} onChange={changeScope}>
          <Field required label={systemText('preInvestment.strategicAllocationWorkspace.investmentObjectiveVersion')}><select required className={inputClass} value={editor.mandateId} onChange={e => changeScope({ mandateId: e.target.value })}><option value="">{systemText('preInvestment.strategicAllocationWorkspace.selectASavedObjective')}</option>{catalog.mandates.map(value => <option key={value.id} value={value.id}>{value.name} · {value.definition.currency} · {value.definition.horizon_years} {" " + systemText('preInvestment.strategicAllocationWorkspace.years')}</option>)}</select></Field>
        </ResearchScopeSelection>
        <Link className="inline-flex min-h-10 items-center text-sm text-accent-800 underline" to={`/pre-investment/product-pool?${new URLSearchParams(editor.mandateId ? { mandate: editor.mandateId } : {})}`}>{s('saaScope.manage')}</Link>
        <div className="flex flex-wrap gap-4 text-sm"><Link className="inline-flex min-h-10 items-center text-accent-800 underline" to="/pre-investment/objectives">{systemText('preInvestment.strategicAllocationWorkspace.createOrReviewInvestmentObjectives')}</Link><Link className="inline-flex min-h-10 items-center text-accent-800 underline" to="/pre-investment/saa/asset-classes">{systemText('preInvestment.strategicAllocationWorkspace.buildAndCheckAssetClasses')}</Link></div>
        <div className="flex flex-col gap-3 sm:flex-row sm:items-center"><Button tone="primary" disabled={Boolean(scopeIssue)} aria-describedby={scopeIssue ? `${formId}-scope` : undefined} className="shrink-0 disabled:bg-slate-100 disabled:text-slate-600 disabled:opacity-100" onClick={() => setStep(1)}>{systemText('preInvestment.strategicAllocationWorkspace.selectConfirmedLtcma')}</Button>{scopeIssue && <p id={`${formId}-scope`} className="text-sm leading-6 text-amber-800">{scopeIssue}</p>}</div>
        {selection}
      </section>}
      {step === 1 && selection}
      {step === 2 && draft && assumptionsReady && <><PolicyFrontier request={policyRequest} disabled={Boolean(dateIssue)} clock={platformDay} result={showCandidates ? policyPreview : null}>{gate => <PolicyCandidates view={showCandidates && policyPreview ? 'results' : 'settings'} value={policyRequest} assets={draft.assets.map(asset => asset.id)} assetLabels={assetLabels} result={policyPreview} busy={busy} compareDisabled={gate.disabled} compareReason={dateIssue || gate.reason}
        meanCovarianceAvailable={Boolean(cmaVersion?.model_result?.mean_estimation_covariance || cmaVersion?.model_result?.posterior_mean_covariance)}
        onChange={value => { invalidate(); const { mandate_id: _mandate, cma_id: _cma, mode: _mode, cma_refs: _refs, ...settings } = value; setEditor(current => ({ ...current, settings })) }}
        onCompare={() => { if (assumptionsReady && !dateIssue && !gate.disabled && !riskBudgetError(draft.assets.map(a => a.id), policyRequest.risk_budget)) void run(signal => previewPolicy(policyRequest, signal), value => { setPolicyPreview(value); setCandidate(null); setShowCandidates(true) }) }}
        onBack={() => setShowCandidates(false)} onShowResults={() => setShowCandidates(true)}
        onSelect={value => { setCandidate(value); setStep(3) }} />}</PolicyFrontier></>}
      {step === 3 && candidate && policyPreview && <section className={`${sectionClass} space-y-5`} aria-label={systemText('preInvestment.strategicAllocationWorkspace.confirmPolicyAdoption')}>
        <h2 className="text-lg font-semibold">{systemText('preInvestment.strategicAllocationWorkspace.doesThisLongTermPolicyReflectYour')}</h2><p className="text-sm leading-6 text-slate-600">{researchMessage(candidate.name)} {" " + systemText('preInvestment.strategicAllocationWorkspace.expectedAnnualReturn') + " "}{percentText(candidate.metrics.expected_return)} {" " + systemText('preInvestment.strategicAllocationWorkspace.expectedAnnualVolatility') + " "}{percentText(candidate.metrics.volatility)}{systemText('preInvestment.strategicAllocationWorkspace.theseCalculationsUseFrozenAssumptionsAndDo')}</p>
        <GoalCandidateSummary candidate={candidate} />
        {candidate.cross_model_results && <CrossModelResults common={mode === 'compatible_all_models'} rows={candidate.cross_model_results} />}
        <dl className="grid gap-3 sm:grid-cols-3">{Object.entries(candidate.weights).map(([asset, weight]) => <div key={asset} className="rounded-lg bg-slate-50 p-3"><dt className="text-sm text-slate-600">{assetLabels[asset] ?? asset}</dt><dd className="mt-1 text-lg font-semibold tabular-nums">{percentText(weight)}</dd></div>)}</dl>
        <p className="text-xs text-slate-600">{requiredText('legend')}</p>
        <Field required label={systemText('preInvestment.strategicAllocationWorkspace.policyVersionName')}><input required className={inputClass} maxLength={120} value={editor.policyName} disabled={busy} onChange={e => { setSavedPolicy(null); setEditor(current => ({ ...current, policyName: e.target.value })) }} /></Field>
        <div>
          <Field required label={systemText('preInvestment.strategicAllocationWorkspace.adoptionRationaleAndReviewPriorities')}><textarea required minLength={POLICY_REASON_MIN} maxLength={POLICY_REASON_MAX} rows={3} aria-describedby={`${formId}-reason-hint ${formId}-reason-count`} className={inputClass} value={editor.reason} disabled={busy} onChange={e => { setSavedPolicy(null); setEditor(current => ({ ...current, reason: e.target.value })) }} /></Field>
          <div className="mt-1 flex flex-wrap items-start justify-between gap-x-4 gap-y-1 text-xs leading-5 text-slate-600">
            <p id={`${formId}-reason-hint`}>{requiredText('reasonHint', { min: POLICY_REASON_MIN, max: POLICY_REASON_MAX })}</p>
            <p id={`${formId}-reason-count`} className="tabular-nums">{requiredText('reasonCount', { count: reasonLength, max: POLICY_REASON_MAX })}</p>
          </div>
        </div>
        <p className="text-xs leading-5 text-slate-600">{systemText('preInvestment.strategicAllocationWorkspace.confirmationCreatesAnImmutableSaaBaselineTaa')}</p>
        <div className="flex flex-col gap-3 sm:flex-row sm:items-center"><Button tone="primary" className="shrink-0 disabled:bg-slate-100 disabled:text-slate-600 disabled:opacity-100" aria-describedby={adoptionIssue ? `${formId}-adoption` : undefined} disabled={Boolean(adoptionIssue)} onClick={adopt}>{savedPolicy ? systemText('preInvestment.strategicAllocationWorkspace.longTermPolicyConfirmed') : systemText('preInvestment.strategicAllocationWorkspace.confirmThisLongTermPolicy')}</Button>
          <p id={`${formId}-adoption`} role="status" className={`min-w-0 text-sm leading-6 ${busy || savedPolicy ? 'text-slate-600' : 'text-amber-800'}`}>{adoptionIssue}</p>
        </div>
        <div className="flex flex-wrap gap-3">
          {savedPolicy && <Link className="inline-flex min-h-10 items-center text-accent-800 underline" to={`/pre-investment/product-allocation-timing?source=${encodeURIComponent(savedPolicy.id)}`}>{s('implementation.handoff')}</Link>}
          {savedPolicy && <Button tone="primary" onClick={() => navigate(allocationJourneyPath('taa', { ...readAllocationJourney(), baselineId: savedPolicy.id, taaRunId: undefined }))}>{systemText('preInvestment.strategicAllocationWorkspace.continueToTaaAssessTacticalDeviations')}</Button>}</div>
      </section>}
      <details className={`${sectionClass} text-sm`}><summary className="cursor-pointer font-medium">{systemText('preInvestment.strategicAllocationWorkspace.savedPoliciesAndHistoricalResearchTools')}</summary><div className="mt-3 space-y-2">{catalog.policies.length ? catalog.policies.map(value => <Link key={value.id} className="flex min-h-10 items-center text-accent-800 underline" to={`/pre-investment/saa/policy?baseline=${encodeURIComponent(value.id)}`}>{value.name} · {value.as_of}</Link>) : <p className="text-slate-600">{systemText('preInvestment.strategicAllocationWorkspace.noConfirmedPoliciesYet')}</p>}<Link className="inline-flex min-h-10 items-center text-accent-800 underline" to={`/pre-investment/saa/allocation-lab?alloc=${encodeURIComponent(editor.allocationName)}`}>{systemText('preInvestment.strategicAllocationWorkspace.historicalEfficientFrontiersRiskBudgetsAndStrategy')}</Link><p className="text-xs leading-5 text-slate-600">{systemText('preInvestment.strategicAllocationWorkspace.historicalExperimentsRetainTheirFunctionalityHistoricallyOptimal')}</p></div></details>
    </>}
  </div>
}

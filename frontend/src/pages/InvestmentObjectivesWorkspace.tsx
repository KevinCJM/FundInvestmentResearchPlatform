import { useEffect, useRef, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { readAllocationDraft, useAllocationDraft, updateAllocationJourney, allocationJourneyPath } from '../app/allocationJourney'
import { useResearchDay } from '../app/ResearchContext'
import { Badge, Button, Card } from '../components/ui'
import { Feedback, today } from '../components/risk-models/ResearchUI'
import { TaskFields, GoalFields, ReturnCheck } from '../components/investment-mandate/MandateFields'
import CashBudgetFields from '../components/investment-mandate/CashBudgetFields'
import MandateRiskFields, { ScaleFields } from '../components/investment-mandate/MandateRiskFields'
import MandateReferenceResults from '../components/investment-mandate/MandateReferenceResults'
import MandateImpactSummary from '../components/investment-mandate/MandateImpactSummary'
import { FundingOverview } from '../components/investment-mandate/MandateResults'
import { compactMandateDefinition, mandateStepIssues, newBoundaryMandate, studyIssue } from '../components/investment-mandate/model'
import { useMandateText } from '../components/investment-mandate/text'
import type { VersionView } from '../services/riskScales'
import { confirmMandate, getMandate, previewMandate, previewMandateFunding,
  type MandateAssessment, type MandateDefinition, type MandateFundingEcho, type MandateStudyRequest, type MandateVersion } from '../services/strategicAllocation'

const stepKeys = ['mandateStep', 'resultStep'] as const
const newStudy = (day: string): MandateStudyRequest => ({ definition: newBoundaryMandate(day), cma_id: null,
  simulation_paths: 2000, seed: 42, validation_seed: 104729, uncertainty_penalty: 1 })
const listLinkClass = 'inline-flex min-h-10 items-center rounded-lg px-2 text-sm font-medium text-accent-700 focus-visible:ring-2 focus-visible:ring-accent-500'

function alignPitDate(value: MandateDefinition, day: string, locked: boolean): MandateDefinition {
  if (!locked || value.as_of === day) return value
  return { ...value, as_of: day,
    risk_authorization: { ...value.risk_authorization!, risk_scale_ref: null, authorized_max_level: null, selected_max_level: null },
    benchmark: null, max_volatility: null }
}

export default function InvestmentObjectivesWorkspace() {
  const { t } = useMandateText()
  const [params, setParams] = useSearchParams()
  const editFrom = params.get('editFrom'), viewId = params.get('view'), fresh = params.get('fresh') === '1'
  const platformDay = useResearchDay()
  const pitUnknown = platformDay === undefined
  const pitLocked = typeof platformDay === 'string'
  const researchDay = pitLocked ? platformDay : today()
  const cutoff = researchDay
  const clockIssue = pitUnknown ? t('clockUnknown') : ''
  const researchLabel = pitUnknown ? t('clockUnknown') : pitLocked ? t('pitActive', { date: platformDay }) : t('pitOff')
  const [draft, setDraft] = useAllocationDraft<MandateStudyRequest>('mandate-study:editor', () => {
    if (!editFrom && !fresh) {
      const current = readAllocationDraft<MandateStudyRequest>('mandate-study:editor')
      if (current?.definition) return { ...newStudy(researchDay), ...current,
        definition: alignPitDate(compactMandateDefinition(current.definition, researchDay), researchDay, pitLocked), cma_id: null,
        simulation_paths: 2000, seed: 42, validation_seed: 104729, uncertainty_penalty: 1 }
      const prior = readAllocationDraft<MandateDefinition>('strategic-mandate:editor')
      if (prior) return { ...newStudy(researchDay), definition: alignPitDate(compactMandateDefinition(prior, researchDay), researchDay, pitLocked) }
    }
    return newStudy(researchDay)
  })
  const [editSource, setEditSource] = useState<MandateVersion | null>(null)
  const [selected, setSelected] = useState<MandateVersion | null>(null)
  const [preview, setPreview] = useState<MandateAssessment | null>(null)
  const [scaleVersion, setScaleVersion] = useState<VersionView | null>(null)
  const [liveFunding, setLiveFunding] = useState<MandateFundingEcho | null>(null)
  const [step, setStep] = useState(0), [acknowledged, setAcknowledged] = useState(false)
  const [busy, setBusy] = useState(false), [initializing, setInitializing] = useState(Boolean(editFrom || viewId))
  const [error, setError] = useState(''), [notice, setNotice] = useState('')
  const [reload, setReload] = useState(0)
  const generation = useRef(0), operation = useRef<AbortController | null>(null)
  const heading = useRef<HTMLHeadingElement>(null), previousResearchDay = useRef(platformDay)
  const definition = draft.definition
  const versionUnavailable = viewId ? selected?.id !== viewId : Boolean(editFrom && editSource?.id !== editFrom)
  const issues = mandateStepIssues(definition, cutoff), numericIssue = studyIssue(draft)
  // Objective and risk decide each other, so they are filled and reported on one page.
  const inputIssue = issues.find(Boolean) ?? ''
  const activeIssue = clockIssue || inputIssue || (step === 0 ? '' : numericIssue)
  const invalid = Boolean(versionUnavailable || clockIssue || issues.some(Boolean) || numericIssue)
  const statusLabel = (status?: string) => t(status === 'diagnosed' ? 'diagnosed' : status === 'needs_revision' ? 'needsRevision' : 'inputsOnly')

  useEffect(() => {
    if (!fresh) return
    const next = new URLSearchParams(params); next.delete('fresh'); setParams(next, { replace: true })
  }, [])
  useEffect(() => { heading.current?.focus() }, [step])
  useEffect(() => () => { generation.current += 1; operation.current?.abort() }, [])
  useEffect(() => {
    const identifier = viewId ?? editFrom
    if (!identifier) { setEditSource(null); setSelected(null); setInitializing(false); return }
    // Confirmation already returned this immutable version; keep it visible.
    if (viewId && selected?.id === viewId) { setInitializing(false); return }
    const controller = new AbortController(); setInitializing(true); setError(''); setNotice('')
    setSelected(null); setEditSource(null); setPreview(null)
    getMandate(identifier, controller.signal).then(version => {
      if (controller.signal.aborted) return
      const assessment = version.assessment?.preview_hash ? version.assessment : null
      setAcknowledged(false)
      if (viewId) {
        setEditSource(null); setSelected(version); setPreview(assessment); setStep(1)
        setDraft(assessment?.request ?? { ...newStudy(researchDay), definition: version.definition })
      } else {
        setEditSource(version); setSelected(null); setPreview(null); setStep(0)
        setDraft({ ...newStudy(researchDay), definition: alignPitDate(compactMandateDefinition(version.definition, researchDay), researchDay, pitLocked) })
      }
    }).catch(reason => { if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : t('operationFailed')) })
      .finally(() => { if (!controller.signal.aborted) setInitializing(false) })
    return () => controller.abort()
  }, [editFrom, viewId, reload])
  useEffect(() => {
    const changed = platformDay !== previousResearchDay.current
    const needsAlignment = pitLocked && definition.as_of !== platformDay
    if (!changed && !needsAlignment) return
    previousResearchDay.current = platformDay
    // A viewing clock does not replace the inputs or diagnosis of a saved version.
    if (viewId || selected) return
    generation.current += 1; operation.current?.abort(); setBusy(false); setAcknowledged(false); setPreview(null); setSelected(null)
    if (pitLocked) setDraft(current => ({ ...current, definition: alignPitDate(current.definition, platformDay, true) }))
    setNotice(pitUnknown ? '' : t('pitChanged'))
  }, [platformDay, pitLocked, pitUnknown, viewId, selected, definition.as_of])

  // 现金流改变真正需要的收益，所以填写页一边填一边回显服务端的确定性资金算术。
  useEffect(() => {
    if (versionUnavailable || viewId || !definition.cash_budget || clockIssue || inputIssue) { setLiveFunding(null); return }
    const controller = new AbortController()
    const timer = setTimeout(() => { previewMandateFunding({ ...draft, cma_id: null }, controller.signal)
      .then(result => { if (!controller.signal.aborted) setLiveFunding(result) })
      .catch(() => { if (!controller.signal.aborted) setLiveFunding(null) }) }, 400)
    return () => { clearTimeout(timer); controller.abort() }
  }, [draft, clockIssue, inputIssue, versionUnavailable, viewId])

  function invalidate() {
    generation.current += 1; operation.current?.abort(); setBusy(false)
    setPreview(null); setSelected(null); setAcknowledged(false); setError(''); setNotice(t('inputChanged'))
  }
  function update(patch: Partial<MandateDefinition>) {
    invalidate()
    setDraft(current => {
      const dateChanged = typeof patch.as_of === 'string' && patch.as_of !== current.definition.as_of
      const next = { ...current.definition, ...patch }
      if (dateChanged) {
        next.risk_authorization = { ...next.risk_authorization!, risk_scale_ref: null, authorized_max_level: null, selected_max_level: null }
        next.max_volatility = null; next.benchmark = null
      }
      return { ...current, definition: next, cma_id: null }
    })
  }
  async function run<T,>(work: (signal: AbortSignal) => Promise<T>, consume: (value: T) => void) {
    operation.current?.abort(); const controller = new AbortController(); operation.current = controller
    const token = ++generation.current; setBusy(true); setError(''); setNotice('')
    try { const result = await work(controller.signal); if (generation.current === token && !controller.signal.aborted) consume(result) }
    catch (reason) { if (generation.current === token && !controller.signal.aborted) setError(reason instanceof Error ? reason.message : t('operationFailed')) }
    finally { if (generation.current === token) setBusy(false) }
  }
  function diagnose() {
    if (invalid || selected || initializing) return
    void run(signal => previewMandate({ ...draft, cma_id: null }, signal), result => { setPreview(result); setAcknowledged(false) })
  }
  function save() {
    if (invalid || !preview || !acknowledged || selected || initializing) return
    void run(signal => confirmMandate({ ...draft, cma_id: null }, preview.preview_hash, signal, editSource?.id), version => {
      setSelected(version); setEditSource(null); setPreview(version.assessment ?? null); updateAllocationJourney({ mandateId: version.id })
      setParams({ view: version.id }, { replace: true }); setNotice(t('savedNotice'))
    })
  }

  const resolved = preview?.definition ?? definition
  return <div className="min-w-0 space-y-5 text-slate-900">
    <header className="space-y-2"><Link className={listLinkClass} to="/pre-investment/objectives">{t('backObjectiveList')}</Link>
      <h1 className="text-2xl font-bold">{selected ? t('savedObjectiveTitle') : editSource ? t('editObjectiveTitle') : t('addObjectiveTitle')}</h1><p className="text-sm leading-6 text-slate-600">{t('simpleDescription')}</p></header>
    <Feedback error={error || clockIssue} notice={notice} />
    {initializing ? <div role="status" aria-live="polite" className="space-y-2"><p className="text-sm text-slate-600">{t('loadingObjective')}</p><div className="h-12 animate-pulse rounded-lg bg-slate-200 motion-reduce:animate-none" /><div className="h-24 animate-pulse rounded-lg bg-slate-100 motion-reduce:animate-none" /></div>
      : versionUnavailable ? <Button onClick={() => setReload(value => value + 1)}>{t('retry')}</Button> : <>
      <nav aria-label={t('stepNavigation')} className="grid grid-cols-2 gap-2 border-b border-slate-200 pb-4">{stepKeys.map((key, index) => {
        const disabled = !selected && (Boolean(clockIssue) || index >= 1 && Boolean(inputIssue || numericIssue))
        return <button key={key} type="button" aria-current={index === step ? 'step' : undefined} disabled={disabled} onClick={() => setStep(index)}
          className={`min-h-11 rounded-lg p-3 text-left text-sm disabled:cursor-not-allowed disabled:opacity-50 ${index === step ? 'bg-accent-50 font-semibold text-accent-900' : 'text-slate-600 hover:bg-slate-50'}`}>{index + 1}. {t(key)}</button>
      })}</nav>
      <MandateImpactSummary value={resolved} decision={preview?.risk_decision} funding={preview?.funding}
        effectiveCash={preview?.reference_diagnosis?.cash_constraint?.effective_min_cash_weight ?? resolved.effective_cash_reserve_weight} />
      {editSource && !selected && <p className="rounded-lg bg-slate-50 p-3 text-sm leading-6 text-slate-700">{t('editingPublishedHint', { name: editSource.name })}</p>}
      {selected && <div className="flex flex-wrap items-center gap-3 rounded-lg bg-slate-50 p-3"><Badge>{t('readonly')}</Badge><p className="text-sm">{selected.name} · {statusLabel(selected.assessment?.status ?? selected.assessment_status)}</p>
        <Link className="inline-flex min-h-10 items-center text-sm font-semibold text-accent-800 underline" to={allocationJourneyPath('pool')}>{t('nextScope')}</Link>
      </div>}
      <h2 ref={heading} tabIndex={-1} className="text-lg font-semibold">{t(stepKeys[step])}</h2>
      <Card className="min-w-0 space-y-5" aria-label={t('editor')}>
        {step === 0 && <fieldset disabled={Boolean(selected)} className="min-w-0 space-y-6">
          <section className="min-w-0 space-y-5"><h3 className="text-base font-semibold">{t('sectionStudy')}</h3>
            <TaskFields value={definition} onChange={update} cutoff={cutoff} pitLocked={pitLocked} pitLabel={researchLabel} /></section>
          <section className="min-w-0 space-y-5 border-t border-slate-200 pt-5"><h3 className="text-base font-semibold">{t('sectionScale')}</h3>
            <ScaleFields value={definition} onChange={update} version={scaleVersion} onVersion={setScaleVersion} readonly={Boolean(selected)} /></section>
          <section className="min-w-0 space-y-5 border-t border-slate-200 pt-5"><h3 className="text-base font-semibold">{t('sectionRisk')}</h3>
            <MandateRiskFields value={definition} onChange={update} version={scaleVersion} readonly={Boolean(selected)} /></section>
          <section className="min-w-0 space-y-5 border-t border-slate-200 pt-5"><h3 className="text-base font-semibold">{t('sectionGoal')}</h3>
            <GoalFields value={definition} onChange={update} version={scaleVersion} />
            <CashBudgetFields value={definition} onChange={update} funding={liveFunding?.funding ?? preview?.funding} />
            <ReturnCheck value={definition} version={scaleVersion} funding={liveFunding} pending={Boolean(clockIssue || inputIssue)} /></section>
        </fieldset>}
        {step === 1 && <div className="space-y-5">
          {!selected && <><p className="text-sm leading-6 text-slate-600">{t('simpleDiagnosisHint')}</p><Button tone="primary" disabled={busy || invalid} onClick={diagnose}>{busy ? t('computing') : preview ? t('recalculate') : t('runDiagnosis')}</Button></>}
          {busy && <div role="status" className="space-y-2"><p className="text-sm text-slate-600">{t('computingHint')}</p><div className="h-24 animate-pulse rounded-lg bg-slate-100 motion-reduce:animate-none" /></div>}
          {preview ? <>
            <MandateReferenceResults value={preview} />
            {preview.funding && <details className="border-t border-slate-200 pt-4"><summary className="min-h-10 cursor-pointer text-sm font-medium">{t('fundingDetails')}</summary><div className="mt-4"><FundingOverview value={preview.funding} /></div></details>}
            <div className="border-t border-slate-200 pt-4">
              <dl className="grid gap-4 sm:grid-cols-2"><div><dt className="text-xs text-slate-600">{t('goalSummary')}</dt><dd className="mt-1 text-sm">{t(definition.objective_kind ?? 'absolute_return')} · {definition.horizon_years} {t('years')}</dd></div>
                <div><dt className="text-xs text-slate-600">{t('riskSummary')}</dt><dd className="mt-1 text-sm tabular-nums">C{resolved.risk_authorization?.selected_max_level ?? '—'} · {t('cap')} {resolved.max_volatility == null ? '—' : `${(resolved.max_volatility * 100).toFixed(2)}%`}</dd></div></dl>
              {preview.blockers.map((reason, index) => <p key={index} className="mt-3 text-sm leading-6 text-amber-800">{reason}</p>)}
              {!selected && <label className="mt-4 flex min-h-11 items-start gap-2 text-sm leading-6"><input className="mt-1.5" type="checkbox" checked={acknowledged} onChange={event => setAcknowledged(event.target.checked)} />{t('finalAcknowledgement')}</label>}
              {!selected && <Button className="mt-3" tone="primary" disabled={busy || !acknowledged || invalid} onClick={save}>{busy ? t('saving') : editSource ? t('saveModifiedVersion') : t('saveVersion')}</Button>}
            </div>
          </> : !busy && <p className="text-sm text-slate-600">{t('diagnosisEmptySimple')}</p>}
        </div>}
        {!selected && activeIssue && <p role="status" className="text-sm text-amber-800">{activeIssue}</p>}
        <div className="flex flex-wrap justify-between gap-3 border-t border-slate-200 pt-4"><Button disabled={step === 0 || busy} onClick={() => setStep(current => current - 1)}>{t('previous')}</Button>
          {step < 1 && <Button tone="primary" disabled={Boolean(activeIssue) || busy} onClick={() => setStep(current => current + 1)}>{t('next')}{t(stepKeys[step + 1])}</Button>}</div>
      </Card>
    </>}
  </div>
}

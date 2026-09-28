import { priorReason } from '../services/cmaCompatibility'
import { useEffect, useRef, useState } from 'react'
import { Link, useNavigate, useSearchParams } from 'react-router-dom'
import { readAllocationJourney, updateAllocationJourney } from '../app/allocationJourney'
import { useResearchDay } from '../app/ResearchContext'
import { Button, Card, ErrorPanel, LoadingPanel } from '../components/ui'
import { Feedback, today } from '../components/risk-models/ResearchUI'
import LtcmaInputFields from '../components/ltcma/LtcmaInputFields'
import LtcmaResults from '../components/ltcma/LtcmaResults'
import { applyScope, copyVersion, methodOf, newDraft, scopeKey, updateContext } from '../components/ltcma/model'
import { linkClass, useLtcmaTask, useLtcmaText } from '../components/ltcma/shared'
import { cmaDraftFromDefinition, completeCma, getMandate, type MandateVersion, type CmaDefinition, type CmaDraft, type CmaPreview } from '../services/strategicAllocation'
import { cmaModelInputError, isScenarioCma } from '../services/cmaModelTypes'
import { scenarioSelectionIssue } from '../components/ltcma/LtcmaScenarioFields'
import ScopeMandateSummary from '../components/strategic-scope/ScopeMandateSummary'
import { ltcma, ltcmaSaaIssue, type CmaDraftView, type LtcmaCapabilities, type LtcmaOptions, type LtcmaOptionSection } from '../services/ltcma'
import type { CmaVersionRef } from '../services/ltcmaContract.generated'

export default function LtcmaWorkspace() {
  const { t } = useLtcmaText(), task = useLtcmaTask(), [params] = useSearchParams(), navigate = useNavigate()
  const platformDay = useResearchDay(), clock = useRef(platformDay), previousClock = useRef(platformDay)
  clock.current = platformDay
  const editId = params.get('edit'), copyId = params.get('copy'), draftId = params.get('draft')
  const initialScope = params.get('strategic_universe') ? `universe:${params.get('strategic_universe')}` : params.get('alloc') ? `allocation:${params.get('alloc')}` : ''
  const [value, setValue] = useState<CmaDraft>(() => newDraft(platformDay || today()))
  const [options, setOptions] = useState<LtcmaOptions | null>(null), [capabilities, setCapabilities] = useState<LtcmaCapabilities | null>(null)
  const [savedDraft, setSavedDraft] = useState<CmaDraftView | null>(null), [copiedFrom, setCopiedFrom] = useState<string | null>(copyId)
  const [editingRef, setEditingRef] = useState<CmaVersionRef | null>(null), [editingName, setEditingName] = useState('')
  const [sourceLabels, setSourceLabels] = useState<Record<string, string>>({})
  const [initializing, setInitializing] = useState(true), [revision, setRevision] = useState(0), [step, setStep] = useState(0)
  const [preview, setPreview] = useState<CmaPreview | null>(null), [acknowledged, setAcknowledged] = useState(false), [operationKey, setOperationKey] = useState('')
  const [notice, setNotice] = useState(''), heading = useRef<HTMLHeadingElement>(null)
  const [optionsState, setOptionsState] = useState({ key: '', error: '' }), [optionsRevision, setOptionsRevision] = useState(0)
  const [mandate, setMandate] = useState<MandateVersion | null>(null), [mandateError, setMandateError] = useState('')
  const inheritConditionalHorizon = useRef<string | null>(null)
  const initializedKey = useRef('')
  const initializationKey = JSON.stringify([editId, copyId, draftId, initialScope, revision])
  const mandateId = options?.strategic_universes.find(item => item.id === value.strategic_universe_id)?.mandate_id || params.get('mandate')
  const cutoff = platformDay && platformDay < today() ? platformDay : today()
  const clockIssue = platformDay === undefined ? t('clockUnknown') : (value.as_of > cutoff || platformDay && value.as_of !== platformDay) ? t('clockInvalid') : ''
  const defaultName = [
    options?.strategic_universes.find(item => item.id === value.strategic_universe_id)?.name || value.alloc_name || 'LTCMA',
    t(methodOf(value)), value.as_of,
  ].join(' · ').slice(0, 120)
  const prepared = updateContext(value, { name: value.name.trim() || defaultName })
  // 名称在保存时后端强制唯一；先在输入步骤拦下，重名改名会作废已算出的候选。
  const nameError = options?.existing_names.some(existing => existing.trim().toLowerCase() === prepared.name.trim().toLowerCase()
    && (!editingRef || existing.trim().toLowerCase() !== editingName.trim().toLowerCase())) ? t('nameConflict') : ''
  const modelIssue = value.model ? cmaModelInputError(value.model) : null
  const method = capabilities?.methods.find(item => item.id === methodOf(value))
  const optionSection: LtcmaOptionSection | null = value.model?.method === 'bayesian_niw' ? 'priors'
    : value.model?.method === 'historical_regime_occupancy' ? 'regimes' : isScenarioCma(value.model) ? 'scenarios' : null
  const priorId = value.model?.method === 'bayesian_niw' ? value.model.prior_ref.id : ''
  const optionsKey = JSON.stringify([initializationKey, optionSection, value.as_of, platformDay, optionsRevision, priorId])
  const optionsError = optionSection && optionsState.key === optionsKey ? optionsState.error : ''
  const optionsReady = !optionSection || optionsState.key === optionsKey && !optionsError
  const names = Object.fromEntries(options?.strategic_universes.find(item => item.id === value.strategic_universe_id)?.definition.assets.map(asset => [asset.id, asset.name]) ?? [])
  const missingAsset = !value.model && value.assets.find(asset => !Number.isFinite(asset.annual_return) || !Number.isFinite(asset.annual_volatility))
  const missingInputs = !value.assets.length ? t('selectScopeFirst') : missingAsset ? t('missingAssetInputs', { asset: names[missingAsset.id] ?? missingAsset.id })
    : !value.model && value.correlation.some(row => row.some(n => !Number.isFinite(n))) ? t('missingCorrelation')
    : !value.basis_confirmed ? t('needBasis') : ''
  const prior = options?.assumptions.find(item => item.id === priorId)
  const priorIssue = optionsReady && value.model?.method === 'bayesian_niw' && value.model.prior_ref.id && options
    ? prior ? priorReason(prior, value, options.strategic_universes.find(item => item.id === value.strategic_universe_id)?.definition) : 'priorUnavailable' : null
  const inputIssue = (priorIssue ? t(priorIssue) : '') || nameError || clockIssue || (method && !method.available ? method.reason ?? t('unavailable') : '')
    || (value.model?.method === 'conditional_scenario' && mandateId && mandate?.id !== mandateId ? mandateError || t('loading') : '')
    || (optionSection && !optionsReady ? optionsError || t('methodOptionsLoading') : '')
    || ((isScenarioCma(value.model) || value.model?.method === 'historical_regime_occupancy') && optionsReady ? scenarioSelectionIssue(value, options, t) : '')
    || modelIssue || missingInputs || (!completeCma(prepared) ? t('needInputs') : '')

  useEffect(() => {
    if (platformDay === undefined || initializedKey.current === initializationKey) return
    setInitializing(true); setPreview(null); setSavedDraft(null); setStep(0); setSourceLabels({})
    setEditingRef(null); setEditingName('')
    inheritConditionalHorizon.current = null
    void task.run(async signal => {
      const requestedDay = clock.current || today()
      const [available, supported, stored, copied] = await Promise.all([
        ltcma.options(signal, requestedDay), ltcma.capabilities(signal), draftId ? ltcma.draft(draftId, signal) : Promise.resolve(null),
        !draftId && (editId || copyId) ? ltcma.get((editId || copyId)!, signal) : Promise.resolve(null),
      ])
      const editing = stored?.editing_ref ? await ltcma.get(stored.editing_ref.id, signal) : editId ? copied : null
      return { available, supported, stored, copied, editing }
    }, ({ available, supported, stored, copied, editing }) => {
      setOptions(available); setCapabilities(supported)
      setEditingRef(stored?.editing_ref ?? (editing ? { id: editing.id, content_hash: editing.content_hash } : null))
      setEditingName(editing?.name ?? '')
      const alignDay = (draft: CmaDraft) => clock.current && clock.current !== draft.as_of
        ? updateContext(draft, { as_of: clock.current, basis_confirmed: false }) : draft
      if (stored) {
        const envelope = stored.editable_definition, raw = (envelope.definition ?? envelope) as CmaDefinition
        if (!Array.isArray(raw.assets) || !raw.as_of) throw new Error('LTCMA_DRAFT_INVALID')
        setValue(alignDay(cmaDraftFromDefinition(raw))); setSavedDraft(stored); setCopiedFrom(stored.copied_from_id)
        const labels = envelope.source_labels
        if (labels && typeof labels === 'object' && !Array.isArray(labels)) setSourceLabels(Object.fromEntries(Object.entries(labels).filter((entry): entry is [string, string] => typeof entry[1] === 'string')))
      } else if (copied) { setValue(alignDay(copyVersion(copied))); setCopiedFrom(editing ? null : copied.id) }
      else { const draft = newDraft(clock.current || today()); setValue(initialScope ? applyScope(draft, initialScope, available) : draft); setCopiedFrom(null) }
      initializedKey.current = initializationKey
      previousClock.current = clock.current
      setInitializing(false)
    })
    return task.invalidate
  }, [initializationKey, platformDay])
  useEffect(() => {
    if (initializing || !optionSection || platformDay === undefined || clockIssue) return
    const controller = new AbortController()
    setOptionsState({ key: '', error: '' })
    ltcma.methodOptions(optionSection, value.as_of, controller.signal, priorId).then(result => {
      if (!controller.signal.aborted) {
        setOptions(current => current && { ...current, ...result })
        setOptionsState({ key: optionsKey, error: '' })
      }
    }).catch(error => { if (!controller.signal.aborted) setOptionsState({ key: optionsKey, error: error instanceof Error ? error.message : t('methodOptionsFailed') }) })
    return () => controller.abort()
  }, [initializing, optionsKey, clockIssue])
  useEffect(() => {
    setMandate(null); setMandateError('')
    if (!mandateId) return
    const controller = new AbortController()
    getMandate(mandateId, controller.signal).then(result => {
      if (result.id !== mandateId) throw new Error('LTCMA_MANDATE_MISMATCH')
      if (!controller.signal.aborted) setMandate(result)
    })
      .catch(() => { if (!controller.signal.aborted) setMandateError(t('scenarioMandateFailed')) })
    return () => controller.abort()
  }, [mandateId])
  useEffect(() => {
    if (value.model?.method !== 'conditional_scenario' || inheritConditionalHorizon.current !== scopeKey(value)) return
    if (!mandateId) { inheritConditionalHorizon.current = null; return }
    if (mandate?.id !== mandateId) return
    inheritConditionalHorizon.current = null
    const days = Math.round(mandate.definition.horizon_years * 252)
    if (days >= 1 && days <= 2520 && value.model.horizon_days !== days) {
      setValue(current => current.model?.method === 'conditional_scenario' ? { ...current, model: { ...current.model, horizon_days: days } } : current)
    }
  }, [mandate, mandateId, value])
  useEffect(() => { heading.current?.focus() }, [step])
  useEffect(() => {
    if (initializing || platformDay === previousClock.current) return
    previousClock.current = platformDay
    task.invalidate(); setPreview(null); setAcknowledged(false); setOperationKey(''); setStep(0); setNotice(t('clockChanged'))
    setValue(current => updateContext(current, { ...(platformDay ? { as_of: platformDay } : {}), basis_confirmed: false }))
  }, [platformDay, initializing])
  const change = (next: CmaDraft) => {
    if (next.model?.method === 'conditional_scenario') {
      if (value.model?.method !== 'conditional_scenario' || scopeKey(next) !== scopeKey(value)) inheritConditionalHorizon.current = scopeKey(next)
      else if (value.model.horizon_days !== next.model.horizon_days) inheritConditionalHorizon.current = null
    } else inheritConditionalHorizon.current = null
    task.invalidate(); setValue(next); setPreview(null); setAcknowledged(false); setOperationKey(''); setNotice(preview ? t('inputChanged') : '')
  }
  const calculate = () => {
    if (inputIssue || initializing || task.busy || !completeCma(prepared)) return
    setValue(prepared)
    void task.run(signal => ltcma.preview(prepared, signal), result => {
      setPreview(result); setAcknowledged(false); setOperationKey(crypto.randomUUID()); setNotice(''); setStep(1)
    })
  }
  const saveDraft = () => {
    if (initializing || task.busy || clockIssue) return
    setValue(prepared)
    const editable = JSON.parse(JSON.stringify({ definition: prepared, source_labels: sourceLabels })) as Record<string, unknown>
    void task.run(signal => ltcma.saveDraft({ name: prepared.name, editable_definition: editable, copied_from_id: copiedFrom, editing_ref: editingRef,
      expected_revision: savedDraft?.revision ?? null }, savedDraft?.id, signal), result => { setSavedDraft(result); setNotice(t('draftSaved')) })
  }
  const publish = () => {
    if (!preview || !acknowledged || inputIssue || !operationKey || task.busy || !completeCma(prepared)) return
    void task.run(signal => editingRef
      ? ltcma.update(editingRef, prepared, preview.preview_hash, operationKey, signal)
      : ltcma.publish(prepared, preview.preview_hash, operationKey, copiedFrom, signal), result => {
      const mandate = params.get('mandate')
      // 从 SAA 的「新建 / 复制 LTCMA」过来的，发布即是本次研究选定的假设；换个目标的版本不抢占当前研究。
      if (mandate && mandate === readAllocationJourney().mandateId && !ltcmaSaaIssue(result)) updateAllocationJourney({ ltcmaId: result.id })
      navigate(`/pre-investment/ltcma/${encodeURIComponent(result.id)}${mandate ? `?mandate=${encodeURIComponent(mandate)}` : ''}`)
    })
  }
  const loadFailed = Boolean(task.error) && initializing
  return <div className="min-w-0 space-y-4 text-slate-900">
    <Link className={linkClass} to="/pre-investment/ltcma">{t('back')}</Link>
    <header className="space-y-2"><h1 className="text-2xl font-bold">{t(editId || editingRef ? 'editVersion' : copyId || copiedFrom ? 'copy' : 'new')}</h1><p className="text-sm leading-6 text-slate-600">{t(value.model?.method === 'conditional_scenario' ? 'scenarioConditionalDescription' : 'description')}</p>{(editingRef || copiedFrom) && <p className="text-sm leading-6 text-slate-600">{t(editingRef ? 'editVersionHint' : 'copyVersionHint')}</p>}</header>
    {/* 初始化就失败时编辑器整块出不来，失败原因交给中间的错误态，顶部只留 PIT 这类仍然成立的提示。 */}
    <Feedback error={loadFailed ? clockIssue : task.error || clockIssue} notice={notice} />
    {mandateId && <ScopeMandateSummary mandate={mandate} loading={!mandate && !mandateError} blockedReason={mandateError || (!mandate ? t('loading') : '')} backHref={`/pre-investment/objectives/new?view=${encodeURIComponent(mandateId)}`} researchDay={platformDay} />}
    {optionsError && <div className="flex flex-wrap items-center gap-3"><Feedback error={optionsError} /><Button onClick={() => setOptionsRevision(n => n + 1)}>{t('retry')}</Button></div>}
    {loadFailed && <ErrorPanel message={task.error} action={<Button onClick={() => setRevision(value => value + 1)}>{t('retry')}</Button>} />}
    {initializing ? !task.error && <LoadingPanel text={t('loading')} /> : options && capabilities && <>
      <nav className="grid grid-cols-2 gap-2 border-b border-slate-200 pb-3" aria-label={t('steps')}>{(['inputStep', 'resultStep'] as const).map((key, index) => <button key={key} type="button" className={`min-h-11 rounded-lg p-3 text-left text-sm focus-visible:ring-2 focus-visible:ring-accent-500 disabled:cursor-not-allowed ${step === index ? 'bg-accent-50 font-semibold text-accent-900' : 'text-slate-600'}`} disabled={task.busy || index === 1 && !preview} aria-current={step === index ? 'step' : undefined} onClick={() => setStep(index)}>{index + 1}. {t(key)}</button>)}</nav>
      <h2 ref={heading} tabIndex={-1} className="text-lg font-semibold">{t(step === 0 ? 'inputStep' : 'resultStep')}</h2>
      <Card>{step === 0 ? <fieldset disabled={task.busy} className="min-w-0"><LtcmaInputFields value={value} options={options} capabilities={capabilities} cutoff={cutoff} platformDay={platformDay} defaultName={defaultName} onChange={change} sourceLabels={sourceLabels} onLabels={setSourceLabels} mandateHorizonDays={mandate ? Math.round(mandate.definition.horizon_years * 252) : undefined} nameError={nameError} methodOptionsReady={optionsReady} methodOptionsError={optionsError} /></fieldset>
        : preview ? <LtcmaResults value={preview} /> : <p className="text-sm text-slate-600">{t('needPreview')}</p>}
        <div className="mt-5 space-y-3 border-t border-slate-200 pt-4">
          {/* 结果步骤上已经渲染出预览数值，形象不能挨着它们（15.2 第 1 条），此时只留文字。 */}
          {task.busy && (step === 0 ? <LoadingPanel text={t('computing')} /> : <p role="status" className="text-sm text-slate-600">{t('computing')}</p>)}
          {step === 1 && preview && <label className="flex min-h-10 items-start gap-2 text-sm leading-6"><input className="mt-1.5" type="checkbox" disabled={task.busy} checked={acknowledged} onChange={event => setAcknowledged(event.target.checked)} />{t(value.model?.method === 'conditional_scenario' ? 'scenarioPublishConfirm' : 'publishConfirm')}</label>}
          <div className="flex flex-wrap gap-3"><Button disabled={task.busy || Boolean(clockIssue)} onClick={saveDraft}>{t('saveDraft')}</Button>
            {step === 0 ? <Button tone="primary" disabled={task.busy || Boolean(inputIssue)} onClick={calculate}>{t(preview ? 'recalculate' : 'preview')}</Button> : <><Button disabled={task.busy} onClick={() => setStep(0)}>{t('editInputs')}</Button><Button tone="primary" disabled={task.busy || !preview || !acknowledged || Boolean(inputIssue)} onClick={publish}>{t(editingRef ? 'saveChanges' : value.model?.method === 'conditional_scenario' ? 'scenarioSaveResearch' : 'publish')}</Button></>}
          </div>
          {inputIssue && <p role="status" className="text-sm leading-6 text-amber-800">{inputIssue}</p>}
          {!preview && <p className="text-xs text-slate-600">{t('needPreview')}</p>}
        </div>
      </Card>
    </>}
  </div>
}

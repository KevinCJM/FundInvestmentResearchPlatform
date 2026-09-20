import { useEffect, useRef, useState } from 'react'
import { Link, useNavigate, useSearchParams } from 'react-router-dom'
import { useResearchDay } from '../app/ResearchContext'
import { Button, Card } from '../components/ui'
import { Feedback, today } from '../components/risk-models/ResearchUI'
import LtcmaInputFields from '../components/ltcma/LtcmaInputFields'
import LtcmaResults from '../components/ltcma/LtcmaResults'
import { applyScope, copyVersion, methodOf, newDraft } from '../components/ltcma/model'
import { linkClass, useLtcmaTask, useLtcmaText } from '../components/ltcma/shared'
import { cmaDraftFromDefinition, completeCma, type CmaDefinition, type CmaDraft, type CmaPreview } from '../services/strategicAllocation'
import { cmaModelInputError } from '../services/cmaModelTypes'
import { ltcma, type CmaDraftView, type LtcmaCapabilities, type LtcmaOptions } from '../services/ltcma'

export default function LtcmaWorkspace() {
  const { t } = useLtcmaText(), task = useLtcmaTask(), [params] = useSearchParams(), navigate = useNavigate()
  const platformDay = useResearchDay(), clock = useRef(platformDay), previousClock = useRef(platformDay)
  clock.current = platformDay
  const copyId = params.get('copy'), draftId = params.get('draft')
  const initialScope = params.get('strategic_universe') ? `universe:${params.get('strategic_universe')}` : params.get('alloc') ? `allocation:${params.get('alloc')}` : ''
  const [value, setValue] = useState<CmaDraft>(() => newDraft(platformDay || today()))
  const [options, setOptions] = useState<LtcmaOptions | null>(null), [capabilities, setCapabilities] = useState<LtcmaCapabilities | null>(null)
  const [savedDraft, setSavedDraft] = useState<CmaDraftView | null>(null), [copiedFrom, setCopiedFrom] = useState<string | null>(copyId)
  const [sourceLabels, setSourceLabels] = useState<Record<string, string>>({})
  const [initializing, setInitializing] = useState(true), [revision, setRevision] = useState(0), [step, setStep] = useState(0)
  const [preview, setPreview] = useState<CmaPreview | null>(null), [acknowledged, setAcknowledged] = useState(false), [operationKey, setOperationKey] = useState('')
  const [notice, setNotice] = useState(''), heading = useRef<HTMLHeadingElement>(null)
  const cutoff = platformDay && platformDay < today() ? platformDay : today()
  const clockIssue = platformDay === undefined ? t('clockUnknown') : value.as_of > cutoff ? t('clockInvalid') : ''
  const modelIssue = value.model ? cmaModelInputError(value.model) : null
  const method = capabilities?.methods.find(item => item.id === methodOf(value))
  const inputIssue = clockIssue || (method && !method.available ? method.reason ?? t('unavailable') : '')
    || modelIssue || (!completeCma(value) || !Number.isInteger(value.horizon_years) || value.horizon_years < 1 || value.horizon_years > 30 ? t('needInputs') : '')

  useEffect(() => {
    setInitializing(true); setPreview(null); setSavedDraft(null); setStep(0); setSourceLabels({})
    void task.run(async signal => {
      const [available, supported, stored, copied] = await Promise.all([
        ltcma.options(signal), ltcma.capabilities(signal), draftId ? ltcma.draft(draftId, signal) : Promise.resolve(null),
        !draftId && copyId ? ltcma.get(copyId, signal) : Promise.resolve(null),
      ])
      return { available, supported, stored, copied }
    }, ({ available, supported, stored, copied }) => {
      setOptions(available); setCapabilities(supported)
      if (stored) {
        const envelope = stored.editable_definition, raw = (envelope.definition ?? envelope) as CmaDefinition
        if (!Array.isArray(raw.assets) || !raw.as_of) throw new Error('LTCMA_DRAFT_INVALID')
        setValue(cmaDraftFromDefinition(raw)); setSavedDraft(stored); setCopiedFrom(stored.copied_from_id)
        const labels = envelope.source_labels
        if (labels && typeof labels === 'object' && !Array.isArray(labels)) setSourceLabels(Object.fromEntries(Object.entries(labels).filter((entry): entry is [string, string] => typeof entry[1] === 'string')))
      } else if (copied) { setValue(copyVersion(copied)); setCopiedFrom(copied.id) }
      else { const draft = newDraft(clock.current || today()); setValue(initialScope ? applyScope(draft, initialScope, available) : draft); setCopiedFrom(null) }
      setInitializing(false)
    })
    return task.invalidate
  }, [copyId, draftId, initialScope, revision])
  useEffect(() => { heading.current?.focus() }, [step])
  useEffect(() => {
    if (platformDay === previousClock.current) return
    previousClock.current = platformDay
    if (initializing) return
    task.invalidate(); setPreview(null); setAcknowledged(false); setOperationKey(''); setNotice(t('clockChanged'))
  }, [platformDay, initializing])
  const change = (next: CmaDraft) => {
    task.invalidate(); setValue(next); setPreview(null); setAcknowledged(false); setOperationKey(''); setNotice(preview ? t('inputChanged') : '')
  }
  const calculate = () => {
    if (inputIssue || initializing || task.busy || !completeCma(value)) return
    void task.run(signal => ltcma.preview(value, signal), result => {
      setPreview(result); setAcknowledged(false); setOperationKey(crypto.randomUUID()); setNotice(''); setStep(1)
    })
  }
  const saveDraft = () => {
    if (!value.name.trim() || initializing || task.busy) return
    const editable = JSON.parse(JSON.stringify({ definition: value, source_labels: sourceLabels })) as Record<string, unknown>
    void task.run(signal => ltcma.saveDraft({ name: value.name, editable_definition: editable, copied_from_id: copiedFrom,
      expected_revision: savedDraft?.revision ?? null }, savedDraft?.id, signal), result => { setSavedDraft(result); setNotice(t('draftSaved')) })
  }
  const publish = () => {
    if (!preview || !acknowledged || inputIssue || !operationKey || task.busy || !completeCma(value)) return
    void task.run(signal => ltcma.publish(value, preview.preview_hash, operationKey, copiedFrom, signal), result => {
      const mandate = params.get('mandate')
      navigate(`/pre-investment/ltcma/${encodeURIComponent(result.id)}${mandate ? `?mandate=${encodeURIComponent(mandate)}` : ''}`)
    })
  }
  return <div className="min-w-0 space-y-4 text-slate-900">
    <Link className={linkClass} to="/pre-investment/ltcma">{t('back')}</Link>
    <header className="space-y-2"><h1 className="text-2xl font-bold">{t(copyId ? 'copy' : 'new')}</h1><p className="text-sm leading-6 text-slate-600">{t('description')}</p></header>
    <Feedback error={task.error || clockIssue} notice={notice} />
    {task.error && initializing && <Button onClick={() => setRevision(value => value + 1)}>{t('retry')}</Button>}
    {initializing ? !task.error && <div role="status" className="space-y-2"><p className="text-sm text-slate-600">{t('loading')}</p><div className="h-12 animate-pulse rounded-lg bg-slate-200 motion-reduce:animate-none" /><div className="h-24 animate-pulse rounded-lg bg-slate-100 motion-reduce:animate-none" /></div> : options && capabilities && <>
      <nav className="grid grid-cols-2 gap-2 border-b border-slate-200 pb-3" aria-label={t('steps')}>{(['inputStep', 'resultStep'] as const).map((key, index) => <button key={key} type="button" className={`min-h-11 rounded-lg p-3 text-left text-sm focus-visible:ring-2 focus-visible:ring-accent-500 disabled:cursor-not-allowed ${step === index ? 'bg-accent-50 font-semibold text-accent-900' : 'text-slate-600'}`} disabled={task.busy || index === 1 && !preview} aria-current={step === index ? 'step' : undefined} onClick={() => setStep(index)}>{index + 1}. {t(key)}</button>)}</nav>
      <h2 ref={heading} tabIndex={-1} className="text-lg font-semibold">{t(step === 0 ? 'inputStep' : 'resultStep')}</h2>
      <Card>{step === 0 ? <fieldset disabled={task.busy} className="min-w-0"><LtcmaInputFields value={value} options={options} capabilities={capabilities} cutoff={cutoff} onChange={change} sourceLabels={sourceLabels} onLabels={setSourceLabels} /></fieldset>
        : preview ? <LtcmaResults value={preview} /> : <p className="text-sm text-slate-600">{t('needPreview')}</p>}
        <div className="mt-5 space-y-3 border-t border-slate-200 pt-4">
          {task.busy && <p role="status" className="text-sm text-slate-600">{t('computing')}</p>}
          {step === 1 && preview && <label className="flex min-h-10 items-start gap-2 text-sm leading-6"><input className="mt-1.5" type="checkbox" disabled={task.busy} checked={acknowledged} onChange={event => setAcknowledged(event.target.checked)} />{t('publishConfirm')}</label>}
          <div className="flex flex-wrap gap-3"><Button disabled={task.busy || !value.name.trim()} onClick={saveDraft}>{t('saveDraft')}</Button>
            {step === 0 ? <Button tone="primary" disabled={task.busy || Boolean(inputIssue)} onClick={calculate}>{t(preview ? 'recalculate' : 'preview')}</Button> : <><Button disabled={task.busy} onClick={() => setStep(0)}>{t('editInputs')}</Button><Button tone="primary" disabled={task.busy || !preview || !acknowledged || Boolean(inputIssue)} onClick={publish}>{t('publish')}</Button></>}
          </div>
          {inputIssue && <p role="status" className="text-sm leading-6 text-amber-800">{inputIssue}</p>}
          {!value.name.trim() && <p className="text-xs text-slate-600">{t('needName')}</p>}
          {!preview && <p className="text-xs text-slate-600">{t('needPreview')}</p>}
        </div>
      </Card>
    </>}
  </div>
}

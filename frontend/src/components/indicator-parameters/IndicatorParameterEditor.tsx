import { useEffect, useRef, useState } from 'react'
import { useI18n } from '../../i18n/runtime'
import type { IndicatorDraft, SeriesParameterDefinition } from '../../services/customIndicators'
import { bindIndicatorParameter, inspectIndicatorParameters, parameterDefinitionKey, parameterInputIssue, type ParameterCandidate } from '../../services/indicatorParameters'

const field = 'mt-1 min-h-10 w-full rounded-xl border border-slate-300 bg-white px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-accent-500'
const button = 'min-h-10 rounded-lg border border-accent-200 bg-white px-3 py-2 text-sm font-semibold text-accent-700 disabled:opacity-50 focus:ring-2 focus:ring-accent-500'

function ParameterSettings({ parameter, disabled, onSave, onFix, onDirty }: {
  parameter: SeriesParameterDefinition; disabled: boolean
  onSave: (value: SeriesParameterDefinition) => void; onFix: () => void; onDirty: (dirty: boolean) => void
}) {
  const { s } = useI18n()
  const initial = () => ({ label: parameter.label, default: String(parameter.default), minimum: String(parameter.minimum), maximum: String(parameter.maximum), step: String(parameter.step), description: parameter.description ?? '' })
  const [text, setText] = useState(initial)
  const [error, setError] = useState(false)
  const signature = JSON.stringify(parameter)
  useEffect(() => { setText(initial()); setError(false) }, [signature])
  const dirty = JSON.stringify(text) !== JSON.stringify(initial())
  useEffect(() => { onDirty(dirty) }, [dirty, onDirty])
  const save = () => {
    const numericKeys = ['default', 'minimum', 'maximum', 'step'] as const
    const next = { ...parameter, label: text.label.trim(), description: text.description.trim(), ...Object.fromEntries(numericKeys.map(key => [key, Number(text[key])])) } as SeriesParameterDefinition
    const invalid = !next.label || numericKeys.some(key => !text[key].trim() || !Number.isFinite(next[key]))
      || next.minimum > next.maximum || next.step <= 0
      || (next.type === 'integer' && numericKeys.some(key => !Number.isInteger(next[key])))
      || Boolean(parameterInputIssue(next, text.default))
    setError(invalid)
    if (!invalid) onSave(next)
  }
  return <article className="min-w-0 rounded-xl border border-accent-100 bg-white p-3">
    <div className="grid gap-3 sm:grid-cols-2">
      <label className="text-xs font-semibold text-slate-600">{s('indicatorParameters.label')}<input className={field} maxLength={80} value={text.label} onChange={event => setText({ ...text, label: event.target.value })} /></label>
      <label className="text-xs font-semibold text-slate-600">{s('indicatorParameters.defaultValue')}<input className={field} type="number" step={parameter.type === 'integer' ? 1 : 'any'} value={text.default} onChange={event => setText({ ...text, default: event.target.value })} /></label>
    </div>
    <p className="mt-2 text-xs text-slate-600">{s('indicatorParameters.code')}: <code>{parameter.id}</code> · {s(`indicatorParameters.type.${parameter.type}`)}</p>
    <details className="mt-2"><summary className="cursor-pointer text-xs font-medium text-slate-600">{s('indicatorParameters.advanced')}</summary>
      <div className="mt-2 grid gap-3 sm:grid-cols-3">{(['minimum', 'maximum', 'step'] as const).map(key => <label key={key} className="text-xs text-slate-600">{s(`indicatorParameters.${key}`)}<input className={field} type="number" step={parameter.type === 'integer' ? 1 : 'any'} value={text[key]} onChange={event => setText({ ...text, [key]: event.target.value })} /></label>)}</div>
      <label className="mt-2 block text-xs text-slate-600">{s('indicatorParameters.description')}<input className={field} maxLength={300} value={text.description} onChange={event => setText({ ...text, description: event.target.value })} /></label>
    </details>
    {error && <p role="alert" className="mt-2 text-xs text-rose-700">{s('indicatorParameters.definitionError')}</p>}
    <div className="mt-3 flex flex-wrap gap-2"><button type="button" disabled={disabled || !dirty} className={button} onClick={save}>{s('indicatorParameters.applySettings')}</button><button type="button" disabled={disabled || dirty} className={button} onClick={onFix}>{s('indicatorParameters.fix')}</button></div>
    {dirty && <p className="mt-2 text-xs text-amber-800">{s('indicatorParameters.pending')}</p>}
  </article>
}

export default function IndicatorParameterEditor({ draft, disabled = false, onPatch, onPendingChange }: {
  draft: IndicatorDraft; disabled?: boolean; onPatch: (patch: Partial<IndicatorDraft>) => void
  onPendingChange?: (pending: boolean) => void
}) {
  const { s, b } = useI18n()
  const [inspection, setInspection] = useState<{ key: string; candidates: ParameterCandidate[] } | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [shared, setShared] = useState<Record<string, string>>({})
  const [pending, setPending] = useState<Record<string, boolean>>({})
  const key = parameterDefinitionKey(draft)
  const latest = useRef(key)
  const request = useRef(0)
  latest.current = key
  const schema = draft.parameter_contract_version === '1.0' ? draft.parameter_schema ?? [] : []
  const hasPending = schema.some(item => pending[item.id])
  useEffect(() => { onPendingChange?.(hasPending || busy) }, [hasPending, busy, onPendingChange])
  useEffect(() => () => { request.current += 1; onPendingChange?.(false) }, [])
  const candidates = inspection?.key === key ? inspection.candidates : []
  const run = async (action?: { candidate_id?: string; parameter_id?: string; fixed_parameter_id?: string }) => {
    const sequence = ++request.current
    const sourceKey = key
    setBusy(true); setError('')
    try {
      if (!action) {
        const result = await inspectIndicatorParameters(draft)
        if (sequence === request.current && latest.current === sourceKey) setInspection({ key: sourceKey, candidates: result.candidates })
      } else {
        const result = await bindIndicatorParameter(draft, action)
        if (sequence !== request.current || latest.current !== sourceKey) return
        const patch: Partial<IndicatorDraft> = {
          parameter_contract_version: '1.0', parameter_schema: result.definition.parameter_schema,
          series_outputs: result.definition.series_outputs, expression: result.definition.expression,
          rolling_source: result.definition.rolling_source ?? null, rolling_transform: result.definition.rolling_transform ?? null,
        }
        setInspection({ key: parameterDefinitionKey({ ...draft, ...patch }), candidates: result.candidates })
        setShared({}); onPatch(patch)
      }
    } catch (failure) {
      if (sequence === request.current && latest.current === sourceKey) setError(failure instanceof Error ? failure.message : s('indicatorParameters.requestError'))
    } finally { if (sequence === request.current) setBusy(false) }
  }
  return <section aria-label={s('indicatorParameters.editorTitle')} className="mt-4 min-w-0 rounded-xl border border-accent-200 bg-accent-50/30 p-4">
    <div className="flex flex-wrap items-center justify-between gap-3"><h3 className="text-sm font-semibold text-slate-800">{s('indicatorParameters.editorTitle')}</h3><button type="button" className={button} disabled={disabled || busy || hasPending} onClick={() => void run()}>{s(busy ? 'indicatorParameters.processing' : 'indicatorParameters.inspect')}</button></div>
    <p className="mt-2 text-xs leading-5 text-slate-600">{s('indicatorParameters.editorHint')}</p>
    {disabled && <p className="mt-2 text-xs text-amber-800">{s('indicatorParameters.applyCanvasFirst')}</p>}
    {error && <p role="alert" className="mt-2 text-sm text-rose-700">{error}</p>}
    <fieldset disabled={disabled || busy} className="mt-3 min-w-0 space-y-3">
      {schema.map(parameter => <ParameterSettings key={parameter.id} parameter={parameter} disabled={disabled || busy}
        onDirty={dirty => setPending(current => current[parameter.id] === dirty ? current : { ...current, [parameter.id]: dirty })}
        onSave={value => onPatch({ parameter_schema: schema.map(item => item.id === parameter.id ? value : item) })}
        onFix={() => { if (window.confirm(s('indicatorParameters.confirmFix', { name: parameter.label }))) void run({ fixed_parameter_id: parameter.id }) }} />)}
      {candidates.filter(item => !item.parameter_id).map(candidate => <div key={candidate.id} className="flex flex-wrap items-center gap-3 rounded-xl border border-slate-200 bg-white p-3">
        <div className="min-w-0 flex-1"><p className="text-sm font-medium text-slate-700">{candidate.output_label} · {b(`parameters.${candidate.argument}.label`, candidate.label)}{candidate.position ? ` · #${candidate.position}` : ''}</p><p className="mt-1 break-words text-xs text-slate-600">{b(`operators.${candidate.operator_id}.label`, candidate.operator_id)} · {s('indicatorParameters.current')} {candidate.value}</p>{candidate.source_expression && <details className="mt-1 text-xs text-slate-600"><summary className="cursor-pointer">{s('indicatorParameters.showUsage')}</summary><code className="mt-1 block break-all">{candidate.source_expression}</code></details>}</div>
        {schema.length > 0 && <select aria-label={s('indicatorParameters.shareWith')} className={`${field} mt-0 !w-auto max-w-full`} value={shared[candidate.id] ?? ''} onChange={event => setShared(current => ({ ...current, [candidate.id]: event.target.value }))}><option value="">{s('indicatorParameters.newParameter')}</option>{schema.filter(item => item.type === candidate.type).map(item => <option key={item.id} value={item.id}>{item.label} ({item.id})</option>)}</select>}
        <button type="button" className={button} disabled={hasPending} onClick={() => void run({ candidate_id: candidate.id, parameter_id: shared[candidate.id] || undefined })}>{s(shared[candidate.id] ? 'indicatorParameters.link' : 'indicatorParameters.open')}</button>
      </div>)}
      {inspection?.key === key && !candidates.some(item => !item.parameter_id) && <p className="text-xs text-slate-600">{s('indicatorParameters.noCandidates')}</p>}
    </fieldset>
  </section>
}

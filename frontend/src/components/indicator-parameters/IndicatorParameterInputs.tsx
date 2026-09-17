import { useEffect, useId, useState } from 'react'
import { useI18n } from '../../i18n/runtime'
import type { SeriesParameterDefinition } from '../../services/customIndicators'
import { parameterInputIssue, parameterRangeLabel } from '../../services/indicatorParameters'

interface Props {
  schema: SeriesParameterDefinition[]
  values: Record<string, number>
  onApply: (values: Record<string, number>) => void
  disabled?: boolean
  /** What the server actually computed with, when a result has come back. */
  effective?: Record<string, number> | null
}

/**
 * Always visible, never a nested card: the number on the card is meaningless
 * without the values behind it, so folding these away hides the question the
 * reader is actually asking.
 */
export default function IndicatorParameterInputs({ schema, values, onApply, disabled = false, effective = null }: Props) {
  const { s } = useI18n()
  const id = useId()
  const toText = (overrides: Record<string, number>) => Object.fromEntries(schema.map(item => [item.id, String(overrides[item.id] ?? item.default)]))
  const [text, setText] = useState(() => toText(values))
  const [submitted, setSubmitted] = useState(false)
  const identity = JSON.stringify([schema, values])
  useEffect(() => { setText(toText(values)); setSubmitted(false) }, [identity])
  if (!schema.length) return null
  const issues = Object.fromEntries(schema.map(item => [item.id, parameterInputIssue(item, text[item.id] ?? '')]))
  const apply = () => {
    setSubmitted(true)
    if (Object.values(issues).some(Boolean)) return
    onApply(Object.fromEntries(schema.flatMap(item => {
      const value = Number(text[item.id])
      return value === item.default ? [] : [[item.id, value]]
    })))
  }
  const pending = schema.some(item => Number(text[item.id]) !== (values[item.id] ?? item.default))
  return <div className="mt-4 border-t border-slate-100 pt-4">
    <p className="text-xs font-semibold text-slate-700">{s('indicatorParameters.runtimeTitle')}</p>
    <fieldset disabled={disabled} className="mt-2 grid min-w-0 grid-cols-[repeat(auto-fit,minmax(min(100%,11rem),1fr))] gap-3">
      {schema.map(item => <div key={item.id} className="min-w-0">
        <label htmlFor={`${id}-${item.id}`} className="block text-xs font-medium text-slate-700">{item.label}</label>
        <input id={`${id}-${item.id}`} type="number" min={item.minimum} max={item.maximum} step={item.step}
          value={text[item.id] ?? ''} onChange={event => setText(current => ({ ...current, [item.id]: event.target.value }))}
          aria-invalid={submitted && Boolean(issues[item.id])} aria-describedby={`${id}-${item.id}-help`}
          className="mt-1 min-h-11 w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm tabular-nums focus:border-accent-500 focus:outline-none focus-visible:ring-2 focus-visible:ring-accent-500" />
        <p id={`${id}-${item.id}-help`} className="mt-1 text-xs text-slate-600">{parameterRangeLabel(item)} · {s('indicatorParameters.defaultValue')} {item.default} · {s('indicatorParameters.step')} {item.step}</p>
        {submitted && issues[item.id] && <p role="alert" className="mt-1 text-xs text-rose-700">{s(`indicatorParameters.error.${issues[item.id]}`)}</p>}
      </div>)}
    </fieldset>
    <div className="mt-3 flex flex-wrap items-center gap-2">
      <button type="button" disabled={disabled} onClick={apply} className="inline-flex min-h-10 items-center rounded-lg bg-accent-600 px-4 text-sm font-semibold text-white transition hover:bg-accent-700 focus:outline-none focus-visible:ring-2 focus-visible:ring-accent-500 disabled:cursor-not-allowed disabled:opacity-50">{s('indicatorParameters.apply')}</button>
      <button type="button" disabled={disabled} onClick={() => { setText(toText({})); setSubmitted(false); onApply({}) }} className="inline-flex min-h-10 items-center rounded-lg border border-slate-300 bg-white px-4 text-sm font-medium text-slate-700 transition hover:border-slate-400 hover:bg-slate-50 focus:outline-none focus-visible:ring-2 focus-visible:ring-accent-500 disabled:cursor-not-allowed disabled:opacity-50">{s('indicatorParameters.reset')}</button>
      {pending
        ? <span role="status" className="text-xs text-amber-800">{s('indicatorParameters.unapplied')}</span>
        : effective && <span className="min-w-0 break-words text-xs text-slate-600">{s('indicatorParameters.actual')}: {schema.map(item => `${item.label}=${effective[item.id] ?? item.default}`).join(' · ')}</span>}
    </div>
  </div>
}

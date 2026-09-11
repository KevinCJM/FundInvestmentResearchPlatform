import { useEffect, useId, useState } from 'react'
import { useI18n } from '../../i18n/runtime'
import type { SeriesParameterDefinition } from '../../services/customIndicators'
import { parameterInputIssue } from '../../services/indicatorParameters'

interface Props {
  schema: SeriesParameterDefinition[]
  values: Record<string, number>
  onApply: (values: Record<string, number>) => void
  disabled?: boolean
}

export default function IndicatorParameterInputs({ schema, values, onApply, disabled = false }: Props) {
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
  const summary = schema.map(item => `${item.label}=${values[item.id] ?? item.default}`).join(' · ')
  return <details className="mt-3 rounded-xl border border-sky-200 bg-sky-50/40 p-4">
    <summary className="cursor-pointer text-sm font-semibold text-slate-800">
      {s('indicatorParameters.runtimeTitle')}<span className="ml-3 break-words text-xs font-normal text-slate-600">{summary}</span>
    </summary>
    <p className="mt-2 text-xs leading-5 text-slate-600">{s('indicatorParameters.runtimeHint')}</p>
    <fieldset disabled={disabled} className="mt-3 grid min-w-0 gap-3 sm:grid-cols-2 lg:grid-cols-3">
      {schema.map(item => <div key={item.id}>
        <label htmlFor={`${id}-${item.id}`} className="block text-sm font-medium text-slate-700">{item.label}</label>
        <input id={`${id}-${item.id}`} type="number" min={item.minimum} max={item.maximum} step={item.step}
          value={text[item.id] ?? ''} onChange={event => setText(current => ({ ...current, [item.id]: event.target.value }))}
          aria-invalid={submitted && Boolean(issues[item.id])} aria-describedby={`${id}-${item.id}-help`}
          className="mt-1 min-h-11 w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-sky-300" />
        <p id={`${id}-${item.id}-help`} className="mt-1 text-xs text-slate-500">{item.minimum}–{item.maximum} · {s('indicatorParameters.defaultValue')} {item.default} · {s('indicatorParameters.step')} {item.step}</p>
        {submitted && issues[item.id] && <p role="alert" className="mt-1 text-xs text-rose-700">{s(`indicatorParameters.error.${issues[item.id]}`)}</p>}
      </div>)}
    </fieldset>
    <div className="mt-3 flex flex-wrap gap-2">
      <button type="button" disabled={disabled} onClick={apply} className="min-h-10 rounded-lg bg-sky-700 px-4 py-2 text-sm font-semibold text-white disabled:opacity-50 focus:ring-2 focus:ring-sky-300">{s('indicatorParameters.apply')}</button>
      <button type="button" disabled={disabled} onClick={() => { setText(toText({})); setSubmitted(false); onApply({}) }} className="min-h-10 rounded-lg border border-slate-300 bg-white px-4 py-2 text-sm disabled:opacity-50 focus:ring-2 focus:ring-sky-300">{s('indicatorParameters.reset')}</button>
    </div>
  </details>
}

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
 * One line: label, value, range, and only what currently says something —
 * the buttons that would do something, and the server's own values only while
 * they contradict the box. An unapplied edit then reads as the disagreement it
 * is, with no sentence needed, and lists of indicators stay scannable.
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
  const customized = schema.some(item => Number(text[item.id]) !== item.default)
  // Only worth printing when it contradicts the box: otherwise the box already
  // says what was computed, and a second copy of every number is just noise.
  const diverged = effective !== null && schema.some(item => Number(text[item.id]) !== (effective[item.id] ?? item.default))
  return <div className="flex min-w-0 flex-wrap items-center gap-x-3 gap-y-2 text-xs">
    <span className="font-medium text-slate-700">{s('indicatorParameters.runtimeTitle')}</span>
    {schema.map(item => <span key={item.id} className="inline-flex items-center gap-1.5">
      <label htmlFor={`${id}-${item.id}`} className="text-slate-700">{item.label}</label>
      <input id={`${id}-${item.id}`} type="number" disabled={disabled} min={item.minimum} max={item.maximum} step={item.step}
        value={text[item.id] ?? ''} onChange={event => setText(current => ({ ...current, [item.id]: event.target.value }))}
        onKeyDown={event => { if (event.key === 'Enter') apply() }}
        aria-invalid={submitted && Boolean(issues[item.id])} aria-describedby={`${id}-${item.id}-help`}
        className="min-h-10 w-24 rounded-lg border border-slate-300 bg-white px-2 text-sm tabular-nums focus:border-accent-500 focus:outline-none focus-visible:ring-2 focus-visible:ring-accent-500 disabled:bg-slate-50" />
      <span id={`${id}-${item.id}-help`} className="text-slate-600">{parameterRangeLabel(item)}</span>
    </span>)}
    {/* Both buttons appear only when they would change something. */}
    {pending && <button type="button" disabled={disabled} onClick={apply}
      className="inline-flex min-h-10 items-center rounded-lg bg-accent-600 px-3 font-semibold text-white transition hover:bg-accent-700 focus:outline-none focus-visible:ring-2 focus-visible:ring-accent-500 disabled:cursor-not-allowed disabled:opacity-50">{s('indicatorParameters.apply')}</button>}
    {customized && <button type="button" disabled={disabled} onClick={() => { setText(toText({})); setSubmitted(false); onApply({}) }}
      className="inline-flex min-h-10 items-center rounded-lg border border-slate-300 bg-white px-3 font-medium text-slate-700 transition hover:border-slate-400 hover:bg-slate-50 focus:outline-none focus-visible:ring-2 focus-visible:ring-accent-500 disabled:cursor-not-allowed disabled:opacity-50">{s('indicatorParameters.reset')}</button>}
    {diverged && effective && <span role="status" className="min-w-0 break-words text-slate-600">{s('indicatorParameters.actual')}: {schema.map(item => `${item.label}=${effective[item.id] ?? item.default}`).join(' · ')}</span>}
    {submitted && schema.filter(item => issues[item.id]).map(item => (
      <p key={item.id} role="alert" className="w-full text-rose-700">{item.label}：{s(`indicatorParameters.error.${issues[item.id]}`)}</p>
    ))}
  </div>
}

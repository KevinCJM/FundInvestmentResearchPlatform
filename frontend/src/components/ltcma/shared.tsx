import { useCallback, useEffect, useRef, useState } from 'react'
import { useI18n } from '../../i18n/runtime'
import { inputClass, NumberInput } from '../risk-models/ResearchUI'
import { percentInputValue } from '../../services/strategicAllocation'

export const control = `${inputClass} placeholder:text-slate-600 placeholder:opacity-100 focus-visible:ring-2 focus-visible:ring-accent-500`
export const linkClass = 'inline-flex min-h-10 items-center rounded-lg px-2 text-sm font-medium text-accent-700 focus-visible:ring-2 focus-visible:ring-accent-500'
export function useLtcmaText() {
  const { s, locale } = useI18n()
  return { locale, t: (key: string, values?: Record<string, string | number>) => s(`ltcma.${key}`, values) }
}
export function RateInput({ value, onChange, label }: { value: number | null | undefined; onChange: (value: number) => void; label?: string }) {
  return <NumberInput aria-label={label} className={`${control} tabular-nums`} value={percentInputValue(value)} onValueChange={number => onChange(number / 100)} />
}
export function useLtcmaTask() {
  const request = useRef<AbortController | null>(null), generation = useRef(0)
  const [busy, setBusy] = useState(false), [error, setError] = useState('')
  useEffect(() => () => { generation.current += 1; request.current?.abort() }, [])
  const invalidate = useCallback(() => { generation.current += 1; request.current?.abort(); setBusy(false); setError('') }, [])
  const run = useCallback(async <T,>(work: (signal: AbortSignal) => Promise<T>, consume: (value: T) => void) => {
    request.current?.abort(); const controller = new AbortController(); request.current = controller
    const token = ++generation.current; setBusy(true); setError('')
    try { const value = await work(controller.signal); if (!controller.signal.aborted && generation.current === token) consume(value) }
    catch (reason) { if (!controller.signal.aborted && generation.current === token) setError(reason instanceof Error ? reason.message : String(reason)) }
    finally { if (generation.current === token) setBusy(false) }
  }, [])
  return { run, invalidate, busy, error, setError }
}

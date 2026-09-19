import { useCallback, useEffect, useRef, useState, type ReactNode } from 'react'
import { Link } from 'react-router-dom'
import { Button } from '../ui'
import { Field, NumberInput } from '../risk-models/ResearchUI'
import { formatNumber, systemText, useI18n } from '../../i18n/runtime'
import { RiskScaleError, type Problem } from '../../services/riskScales'

export const controlClass = 'mt-1 min-h-10 w-full min-w-0 rounded-lg border border-slate-300 bg-white px-2.5 py-1.5 text-sm text-slate-900 placeholder:text-slate-600 placeholder:opacity-100 focus-visible:ring-2 focus-visible:ring-accent-500'
export const linkClass = 'inline-flex min-h-10 items-center rounded-lg px-2 text-sm font-medium text-accent-700 focus-visible:ring-2 focus-visible:ring-accent-500'
export function useRiskText() { const { s, locale } = useI18n(); return { t: (key: string, values?: Record<string, string | number>) => s(`riskScales.${key}`, values), locale } }
export const pct = (value: number | null | undefined) => formatNumber(value, { style: 'percent', minimumFractionDigits: 2, maximumFractionDigits: 2 })
export const problemMessage = (problem: Pick<Problem, 'code' | 'message'>) => systemText(`riskScales.problem.${problem.code}`, {}, problem.message)
export function PercentField({ label, value, onChange, error, required = false }: { label: string; value: number; onChange: (value: number) => void; error?: string; required?: boolean }) {
  const { t } = useRiskText()
  const issue = error || (!Number.isFinite(value) ? t('numberRequired') : '')
  return <Field label={label} required={required}><NumberInput className={controlClass} aria-label={label} aria-required={required || undefined} value={typeof value === 'number' ? value * 100 : NaN} onValueChange={next => onChange(next / 100)} />{issue && <span role="alert" className="mt-1 block text-xs text-rose-800">{issue}</span>}</Field>
}
export function Loading() { const { t } = useRiskText(); return <div role="status" aria-live="polite" className="space-y-2"><span className="text-sm text-slate-600">{t('loading')}</span>{[0, 1].map(row => <div key={row} className="h-10 animate-pulse rounded-lg bg-slate-200 motion-reduce:animate-none" />)}</div> }
export function ErrorNotice({ error, retry, retryLabel }: { error: unknown; retry?: () => void; retryLabel?: string }) {
  const { t } = useRiskText()
  if (!error) return null
  const problem = error instanceof RiskScaleError ? error : new RiskScaleError('REQUEST_FAILED', '')
  return <div role="alert" className="space-y-1 rounded-lg border border-rose-200 bg-rose-50 p-3 text-sm text-rose-900"><p>{problem.status === 409 ? t('conflict') : t('requestError')} {problemMessage(problem) || t('networkError')}</p>{retry && <Button onClick={retry}>{retryLabel ?? t('retry')}</Button>}</div>
}
export function Problems({ items = [] }: { items?: Problem[] }) { return <div className="space-y-1">{items.map((item, index) => <p role="status" key={`${item.code}-${index}`} className="break-words text-sm text-amber-900">{problemMessage(item)}</p>)}</div> }
export function TaskHeader({ title, children }: { title: string; children?: ReactNode }) { const { t } = useRiskText(); return <header className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between"><div className="min-w-0"><Link className={linkClass} to="/settings/risk-scales">{t('backList')}</Link><h1 className="break-words text-2xl font-bold text-slate-900">{title}</h1></div><div className="flex flex-wrap gap-2">{children}</div></header> }

/** All editor requests share a revision. Aborting alone cannot reject a late response. */
export function useRiskTask() {
  const generation = useRef(0)
  const controller = useRef<AbortController | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<unknown>(null)
  const invalidate = useCallback(() => { generation.current++; controller.current?.abort(); setBusy(false); setError(null) }, [])
  useEffect(() => () => { generation.current++; controller.current?.abort() }, [])
  const run = async <T,>(operation: (signal: AbortSignal) => Promise<T>, apply: (value: T) => void) => {
    controller.current?.abort(); const token = ++generation.current; const next = new AbortController(); controller.current = next; setBusy(true); setError(null)
    try { const result = await operation(next.signal); if (token === generation.current && !next.signal.aborted) apply(result) }
    catch (failure) { if (token === generation.current && !next.signal.aborted) setError(failure) }
    finally { if (token === generation.current) setBusy(false) }
  }
  return { run, invalidate, busy, error, setError }
}

import { cloneElement, isValidElement, useId, type ReactNode } from 'react'
export const inputClass = 'min-h-10 w-full min-w-0 rounded-xl border border-slate-300 bg-white px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-accent-500 disabled:bg-slate-100'
export const buttonClass = 'min-h-10 rounded-lg bg-accent-600 px-4 py-2 text-sm font-semibold text-white hover:bg-accent-700 focus:outline-none focus:ring-2 focus:ring-accent-500 disabled:cursor-wait disabled:opacity-50'
export const secondaryClass = 'min-h-10 rounded-xl border border-slate-300 bg-white px-3 py-2 text-sm font-semibold text-slate-700 hover:bg-slate-50 focus:outline-none focus:ring-2 focus:ring-accent-500 disabled:opacity-50'
export type Action = <T>(label: string, work: () => Promise<T>) => Promise<T | undefined>
export function Field({ label, children, hint }: { label: string; children: ReactNode; hint?: string }) {
  const generatedId = useId()
  const hintId = generatedId + '-hint'
  const element = isValidElement<{ id?: string; 'aria-describedby'?: string }>(children) ? children : undefined
  const controlId = element?.props.id || generatedId
  const control = element ? cloneElement(element, { id: controlId, ...(hint ? { 'aria-describedby': [element.props['aria-describedby'], hintId].filter(Boolean).join(' ') } : {}) }) : children
  return <div className="min-w-0 text-sm font-medium text-slate-700"><label htmlFor={controlId} className="mb-1.5 block">{label}</label>{control}{hint && <p id={hintId} className="mt-1 text-xs font-normal leading-5 text-slate-600">{hint}</p>}</div>
}
export function Card({ title, children }: { title: string; children: ReactNode }) {
  return <section className="min-w-0 rounded-xl border border-slate-200 bg-white p-4 sm:p-5"><h3 className="mb-4 text-base font-bold text-slate-900">{title}</h3>{children}</section>
}

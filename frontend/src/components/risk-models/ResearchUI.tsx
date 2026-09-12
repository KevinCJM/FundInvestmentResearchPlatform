import { useEffect, useRef, useState, type InputHTMLAttributes, type ReactNode } from 'react'

export const inputClass = 'mt-1 min-h-11 w-full min-w-0 rounded-xl border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 outline-none focus:border-accent-600 focus:ring-2 focus:ring-accent-500 disabled:bg-slate-100'
export const buttonClass = 'inline-flex min-h-11 items-center justify-center rounded-xl border border-slate-300 bg-white px-4 py-2 text-sm font-medium text-slate-700 hover:border-accent-600 focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent-600 disabled:cursor-not-allowed disabled:opacity-50'
export const primaryClass = 'inline-flex min-h-11 items-center justify-center rounded-lg bg-accent-700 px-5 py-2 text-sm font-semibold text-white hover:bg-accent-800 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent-700 disabled:cursor-not-allowed disabled:bg-slate-400'
export const sectionClass = 'min-w-0 rounded-xl border border-slate-200 bg-white p-4 sm:p-5'
export const today = () => new Date().toISOString().slice(0, 10)
export const numberText = (value: number | null | undefined, digits = 3) => value == null || !Number.isFinite(value) ? '—' : new Intl.NumberFormat('zh-CN', { maximumFractionDigits: digits }).format(value)
export const percentText = (value: number | null | undefined) => value == null || !Number.isFinite(value) ? '—' : new Intl.NumberFormat('zh-CN', { style: 'percent', minimumFractionDigits: 2, maximumFractionDigits: 2 }).format(value)
export const statusLabels: Record<string, string> = { active: '可使用', not_yet_available: '当时尚不可用', retired: '已停用', expired: '已过期', dependency_unavailable: '上游成果不可用' }
export const frequencyLabels: Record<string, string> = { daily: '日频', weekly: '周频', monthly: '月频', quarterly: '季频', single_shock: '单次估值' }

export function NumberInput({ value, onValueChange, ...props }: Omit<InputHTMLAttributes<HTMLInputElement>, 'value' | 'onChange' | 'type'> & { value: number; onValueChange: (value: number) => void }) {
  const [text, setText] = useState(Number.isFinite(value) ? String(value) : '')
  const emitted = useRef(value)
  useEffect(() => {
    if (!Object.is(value, emitted.current)) { emitted.current = value; setText(Number.isFinite(value) ? String(value) : '') }
  }, [value])
  return <input {...props} type="text" inputMode="decimal" role="spinbutton" value={text}
    aria-valuenow={Number.isFinite(value) ? value : undefined} aria-invalid={!Number.isFinite(value)}
    onChange={event => {
      const raw = event.target.value
      const parsed = /^[+-]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?$/i.test(raw.trim()) ? Number(raw) : NaN
      setText(raw); emitted.current = parsed; onValueChange(parsed)
    }} />
}

export function Field({ label, hint, children }: { label: string; hint?: string; children: ReactNode }) {
  return <label className="block min-w-0 text-sm font-medium text-slate-800">{label}{children}{hint && <span className="mt-1 block text-xs font-normal leading-5 text-slate-600">{hint}</span>}</label>
}
export function Feedback({ error, notice }: { error?: string; notice?: string }) {
  return <>{error && <p role="alert" className="rounded-lg border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-900">{error}</p>}{notice && <p role="status" className="rounded-lg border border-accent-200 bg-accent-50 px-4 py-3 text-sm text-accent-900">{notice}</p>}</>
}
export function Steps({ labels, active, onChange, disabled = false }: { labels: string[]; active: number; onChange: (index: number) => void; disabled?: boolean }) {
  return <nav aria-label="操作步骤" className="grid grid-cols-1 gap-2 border-b border-slate-200 pb-4 sm:grid-cols-3">{labels.map((label, index) => <button type="button" key={label} disabled={disabled} aria-current={active === index ? 'step' : undefined} onClick={() => onChange(index)} className={`flex min-h-11 items-center gap-2 rounded-lg px-3 py-2 text-left text-sm ${active === index ? 'bg-accent-50 font-semibold text-accent-900' : 'text-slate-600 hover:bg-slate-50'}`}><span className={`flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-xs ${active === index ? 'bg-accent-700 text-white' : 'border border-slate-300'}`}>{index + 1}</span>{label}</button>)}</nav>
}
export function Empty({ title, children }: { title: string; children: ReactNode }) {
  return <div className="rounded-xl border border-dashed border-slate-300 bg-slate-50 p-6"><h3 className="font-semibold text-slate-900">{title}</h3><div className="mt-2 text-sm leading-6 text-slate-600">{children}</div></div>
}

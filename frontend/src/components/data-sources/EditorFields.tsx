import { useEffect, useRef, useState } from 'react'
import type { DownloadPolicy } from '../../services/dataSources'

export const inputClass = 'mt-1 min-h-10 w-full min-w-0 rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 disabled:bg-slate-100 disabled:text-slate-500'
export const buttonClass = 'min-h-10 rounded-lg border border-slate-300 px-3 py-2 text-sm font-semibold text-slate-700 hover:bg-slate-50 focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 disabled:opacity-40'
export const primaryClass = 'min-h-10 rounded-lg bg-indigo-700 px-4 py-2 text-sm font-semibold text-white hover:bg-indigo-600 focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 disabled:opacity-40'

export function TextField({ label, value, onChange, disabled = false, required = false, placeholder }: {
  label: string; value: string; onChange: (value: string) => void; disabled?: boolean; required?: boolean; placeholder?: string
}) {
  return <label className="block min-w-0 text-xs font-semibold text-slate-600">{label}<input className={inputClass} value={value} onChange={event => onChange(event.target.value)} disabled={disabled} required={required} placeholder={placeholder} /></label>
}

export function JsonField({ label, value, onChange, objectOnly = true, disabled = false }: {
  label: string; value: unknown; onChange: (value: unknown) => void; objectOnly?: boolean; disabled?: boolean
}) {
  const serialized = JSON.stringify(value, null, 2)
  const [text, setText] = useState(serialized)
  const [error, setError] = useState('')
  const ref = useRef<HTMLTextAreaElement>(null)
  useEffect(() => { setText(serialized); setError(''); ref.current?.setCustomValidity('') }, [serialized])
  return <label className="block min-w-0 text-xs font-semibold text-slate-600">{label}
    <textarea ref={ref} className={`${inputClass} min-h-24 font-mono text-xs`} value={text} disabled={disabled} spellCheck={false} onChange={event => {
      const next = event.target.value
      setText(next)
      try {
        const parsed: unknown = JSON.parse(next)
        if (objectOnly && (!parsed || Array.isArray(parsed) || typeof parsed !== 'object')) throw new Error()
        event.target.setCustomValidity(''); setError(''); onChange(parsed)
      } catch {
        const message = objectOnly ? '请填写有效的 JSON 对象。' : '请填写有效 JSON；文本需加双引号。'
        event.target.setCustomValidity(message); setError(message)
      }
    }} />
    {error ? <span className="mt-1 block text-rose-700">{error}</span> : null}
  </label>
}

export function validateEditor(form: HTMLFormElement | null): boolean {
  if (!form) return false
  if (form.checkValidity()) return true
  // Reveal invalid controls inside collapsed advanced sections before focusing.
  form.querySelectorAll<HTMLDetailsElement>('details').forEach(item => { item.open = true })
  form.reportValidity()
  return false
}

const policyFields: { key: keyof DownloadPolicy; label: string; min: number; max: number; step?: number; optional?: boolean }[] = [
  { key: 'requests_per_minute', label: '每分钟请求次数', min: 1, max: 10000 },
  { key: 'rows_per_minute', label: '每分钟数据行数（留空不限）', min: 1, max: 10000000, optional: true },
  { key: 'max_rows_per_request', label: '单次返回行数上限', min: 1, max: 100000 },
  { key: 'min_interval_seconds', label: '最小请求间隔（秒）', min: 0, max: 3600, step: 0.01 },
  { key: 'max_concurrency', label: '最大并发请求数', min: 1, max: 32 },
  { key: 'max_attempts', label: '最多尝试次数（含首次）', min: 1, max: 6 },
  { key: 'connect_timeout_seconds', label: '连接超时（秒）', min: 0.1, max: 60, step: 0.1 },
  { key: 'read_timeout_seconds', label: '读取超时（秒）', min: 0.1, max: 300, step: 0.1 },
  { key: 'backoff_seconds', label: '首次重试等待（秒）', min: 0, max: 60, step: 0.1 },
  { key: 'rate_limit_wait_seconds', label: '限流后至少等待（秒）', min: 1, max: 600 },
  { key: 'max_response_bytes', label: '单次响应字节上限', min: 1024, max: 33554432 },
  { key: 'max_runtime_seconds', label: '任务运行上限（秒）', min: 1, max: 86400 },
]

export function PolicyEditor({ value, onChange, effective }: {
  value: DownloadPolicy; onChange: (value: DownloadPolicy) => void; effective?: DownloadPolicy
}) {
  return <section className="space-y-3" aria-label="下载限制">
    <p className="text-xs leading-6 text-slate-500">数据源与接口限制共同生效，取更严格值。所有工作线程和进程共享配额；行数配额按单次上限预留。</p>
    {effective ? <p className="rounded-lg bg-indigo-50 p-3 text-xs text-indigo-900">上次保存的有效限制：每分钟 {effective.requests_per_minute} 次请求；每次最多 {effective.max_rows_per_request} 行；最多 {effective.max_concurrency} 个并发。修改后保存以重新计算。</p> : null}
    <p className="text-xs text-slate-500">优先核对下面三项额度。连接、重试、并发等技术参数一般不需修改。</p>
    <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">{policyFields.slice(0, 3).map(item => <label key={item.key} className="text-xs font-semibold text-slate-600">{item.label}
      <input className={inputClass} type="number" min={item.min} max={item.max} step={item.step ?? 1} required={!item.optional} value={value[item.key] ?? ''} onChange={event => onChange({ ...value, [item.key]: event.target.value === '' && item.optional ? null : Number(event.target.value) })} />
    </label>)}</div>
    <details className="rounded-lg border border-slate-200 p-3"><summary className="cursor-pointer text-sm font-semibold">高级：并发、超时与重试</summary>
      <div className="mt-3 grid gap-3 sm:grid-cols-2 xl:grid-cols-3">{policyFields.slice(3).map(item => <label key={item.key} className="text-xs font-semibold text-slate-600">{item.label}
        <input className={inputClass} type="number" min={item.min} max={item.max} step={item.step ?? 1} required={!item.optional} value={value[item.key] ?? ''} onChange={event => onChange({ ...value, [item.key]: event.target.value === '' && item.optional ? null : Number(event.target.value) })} />
      </label>)}</div>
    </details>
  </section>
}

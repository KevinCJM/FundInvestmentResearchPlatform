import type { RequestField } from '../../services/dataSources'
import { inputClass } from './EditorFields'

export default function RequestParameterFields({ fields, values, bindings = {}, onChange }: {
  fields: RequestField[]; values: Record<string, unknown>; bindings?: Record<string, string>
  onChange: (values: Record<string, unknown>) => void
}) {
  return <div className="grid gap-3 sm:grid-cols-2">{fields.map(field => {
    const raw = String(values[field.name] ?? field.default ?? '')
    const value = field.data_type === 'date' && /^\d{8}$/.test(raw) ? `${raw.slice(0, 4)}-${raw.slice(4, 6)}-${raw.slice(6)}` : raw
    return <label key={field.name} className="text-xs font-semibold">{field.label}
      {bindings[field.name] ? <span className="mt-2 block rounded-lg bg-accent-50 p-3 text-accent-800">运行时填写：{bindings[field.name]}</span> : <input aria-label={field.label} className={inputClass} type={field.data_type === 'text' ? 'text' : field.data_type}
        value={value} placeholder={field.placeholder} step={field.data_type === 'number' ? 'any' : undefined}
        onChange={event => {
          const next = { ...values }
          if (!event.target.value) delete next[field.name]
          else next[field.name] = field.data_type === 'date' && field.date_format !== 'iso' ? event.target.value.replace(/-/g, '') : field.data_type === 'number' ? Number(event.target.value) : event.target.value
          onChange(next)
        }} />}
      {field.description ? <span className="mt-1 block font-normal leading-5 text-slate-600">{field.description}</span> : null}
    </label>
  })}</div>
}

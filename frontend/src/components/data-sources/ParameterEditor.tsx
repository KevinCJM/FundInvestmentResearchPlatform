import { useEffect, useRef, useState } from 'react'
import { buttonClass, inputClass, JsonField } from './EditorFields'

type ValueType = 'text' | 'number' | 'boolean' | 'null' | 'json'
type Row = { id: number; name: string; type: ValueType; value: unknown }
const valueType = (value: unknown): ValueType => value === null ? 'null' : typeof value === 'number' ? 'number' : typeof value === 'boolean' ? 'boolean' : typeof value === 'object' ? 'json' : 'text'

/** Keep incomplete names/values in the form, never silently submit the previous object. */
export default function ParameterEditor({ label, value, onChange, onDraftChange, stringOnly = false }: {
  label: string; value: Record<string, unknown>; onChange: (value: Record<string, unknown>) => void; onDraftChange?: () => void; stringOnly?: boolean
}) {
  const sequence = useRef(0)
  const toRows = (data: Record<string, unknown>) => Object.entries(data).map(([name, item]) => ({ id: sequence.current++, name, value: item, type: stringOnly ? 'text' as const : valueType(item) }))
  const [rows, setRows] = useState<Row[]>(() => toRows(value))
  const serialized = JSON.stringify(value)
  const emitted = useRef(serialized)
  useEffect(() => {
    if (serialized !== emitted.current) {
      emitted.current = serialized
      setRows(toRows(JSON.parse(serialized)))
    }
  }, [serialized, stringOnly])

  const keyError = (row: Row, items: Row[]) => !row.name.trim() ? '请填写参数名称。' : items.some(item => item.id !== row.id && item.name === row.name) ? '参数名称不能重复。' : ''
  const change = (next: Row[]) => {
    onDraftChange?.()
    setRows(next)
    if (next.some(row => keyError(row, next) || (row.type === 'number' && (row.value === '' || !Number.isFinite(Number(row.value)))))) return
    const data = Object.fromEntries(next.map(row => [row.name, row.type === 'number' ? Number(row.value) : row.type === 'null' ? null : row.value]))
    emitted.current = JSON.stringify(data)
    onChange(data)
  }
  const patch = (id: number, next: Partial<Row>) => change(rows.map(row => row.id === id ? { ...row, ...next } : row))

  return <section aria-label={label} className="space-y-3 rounded-xl border border-slate-200 p-3">
    <h4 className="text-sm font-semibold text-slate-800">{label}</h4>
    {!rows.length ? <p className="text-xs leading-5 text-slate-500">暂未设置。接口不需要参数时可以留空。</p> : null}
    {rows.map((row, index) => <div key={row.id} className="grid items-start gap-2 rounded-lg bg-slate-50 p-3 sm:grid-cols-[minmax(0,1fr)_110px_minmax(0,1.4fr)_auto]">
      <label className="text-xs font-semibold text-slate-600">名称<input aria-label={`${label} ${index + 1} 名称`} className={inputClass} value={row.name} required ref={input => { input?.setCustomValidity(keyError(row, rows)) }} onChange={event => patch(row.id, { name: event.target.value })} />{keyError(row, rows) ? <span className="mt-1 block text-rose-700">{keyError(row, rows)}</span> : null}</label>
      <label className="text-xs font-semibold text-slate-600">类型<select aria-label={`${label} ${index + 1} 类型`} className={inputClass} disabled={stringOnly} value={row.type} onChange={event => {
        const type = event.target.value as ValueType
        patch(row.id, { type, value: type === 'number' ? 0 : type === 'boolean' ? false : type === 'null' ? null : type === 'json' ? {} : '' })
      }}><option value="text">文本</option>{!stringOnly ? <><option value="number">数字</option><option value="boolean">是 / 否</option><option value="null">空值</option><option value="json">结构化数据</option></> : null}</select></label>
      {row.type === 'json' ? <JsonField label={`${label} ${index + 1} 值`} objectOnly={false} value={row.value} onChange={item => patch(row.id, { value: item })} /> : <label className="text-xs font-semibold text-slate-600">值
        {row.type === 'boolean' ? <select aria-label={`${label} ${index + 1} 值`} className={inputClass} value={String(row.value)} onChange={event => patch(row.id, { value: event.target.value === 'true' })}><option value="true">是（true）</option><option value="false">否（false）</option></select>
          : row.type === 'null' ? <span className="mt-1 block p-2 text-sm font-normal text-slate-500">空值（null）</span>
            : <input aria-label={`${label} ${index + 1} 值`} className={inputClass} type={row.type === 'number' ? 'number' : 'text'} step={row.type === 'number' ? 'any' : undefined} required={row.type === 'number'} value={String(row.value ?? '')} onChange={event => patch(row.id, { value: event.target.value })} />}
      </label>}
      <button type="button" className={`${buttonClass} sm:mt-5`} aria-label={`删除${label} ${index + 1}`} onClick={() => change(rows.filter(item => item.id !== row.id))}>删除</button>
    </div>)}
    <button type="button" className={buttonClass} onClick={() => change([...rows, { id: sequence.current++, name: '', type: 'text', value: '' }])}>添加{label}</button>
  </section>
}

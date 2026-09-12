import { useEffect, useState, type ReactNode } from 'react'

export interface OutputFieldsValue {
  id: string; label: string; unit: string; display_format: 'number' | 'percent'; precision: number
}
export interface EnumOutputItem { id: string; label: string; color?: string; role?: string; order?: number }

/** Shared output configuration. Rendering a market band is a consumer concern. */
export default function SeriesOutputFields<T extends OutputFieldsValue>({ output, index, onChange, enumItems, onEnumChange, identityReadOnly = false, onIdentifierCommit, children }: {
  output: T; index: number; onChange: (patch: Partial<T>) => void; enumItems?: EnumOutputItem[]
  onEnumChange?: (items: EnumOutputItem[]) => void; identityReadOnly?: boolean; children?: ReactNode
  onIdentifierCommit?: (id: string) => string | undefined
}) {
  const [identifier, setIdentifier] = useState(output.id)
  const [identifierError, setIdentifierError] = useState('')
  useEffect(() => { setIdentifier(output.id); setIdentifierError('') }, [output.id])
  const field = 'mt-1 block min-h-10 w-full rounded-xl border border-slate-200 bg-white px-2 py-2 text-sm disabled:bg-slate-100'
  const patch = (value: Partial<OutputFieldsValue>) => onChange(value as Partial<T>)
  return <div className="space-y-3">
    <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
      <label className="text-xs font-semibold text-slate-600">通道 ID<input aria-label={`输出通道 ${index + 1} ID`} readOnly={identityReadOnly} maxLength={80} value={onIdentifierCommit ? identifier : output.id} onChange={event => onIdentifierCommit ? setIdentifier(event.target.value) : patch({ id: event.target.value })} onBlur={() => { if (onIdentifierCommit) { const error = onIdentifierCommit(identifier); setIdentifierError(error || ''); if (error) setIdentifier(output.id) } }} className={field} />{identifierError && <span role="alert" className="mt-1 block text-rose-700">{identifierError}</span>}</label>
      <label className="text-xs font-semibold text-slate-600">通道名称<input aria-label={`输出通道 ${index + 1} 名称`} value={output.label} onChange={event => patch({ label: event.target.value })} className={field} /></label>
      {enumItems ? <label className="text-xs font-semibold text-slate-600">值类型<input aria-label="输出值类型" readOnly value="枚举时序" className={field} /></label> : <>
        <label className="text-xs font-semibold text-slate-600">单位<input aria-label={`输出通道 ${index + 1} 单位`} value={output.unit} onChange={event => patch({ unit: event.target.value })} className={field} /></label>
        <label className="text-xs font-semibold text-slate-600">展示格式<select aria-label={`输出通道 ${index + 1} 展示格式`} value={output.display_format} onChange={event => patch({ display_format: event.target.value as T['display_format'] })} className={field}><option value="number">数值</option><option value="percent">百分比</option></select></label>
        <label className="text-xs font-semibold text-slate-600">小数位<input aria-label={`输出通道 ${index + 1} 小数位`} type="number" min={0} max={8} value={output.precision} onChange={event => patch({ precision: Number(event.target.value) })} className={field} /></label>
      </>}{children}
    </div>
    {enumItems && <section aria-label="枚举输出设置" className="rounded-xl border border-slate-200 bg-white p-3">
      <div className="flex items-center justify-between gap-2"><h3 className="text-sm font-semibold text-slate-800">枚举项</h3><button type="button" disabled={enumItems.length >= 12 || !onEnumChange} onClick={() => { let n = enumItems.length + 1; while (enumItems.some(item => item.id === `state_${n}`)) n++; onEnumChange?.([...enumItems, { id: `state_${n}`, label: `状态 ${n}`, color: '#64748b', role: 'neutral', order: enumItems.length }]) }} className="min-h-10 rounded-lg border border-slate-200 px-3 text-xs font-semibold disabled:opacity-40">添加状态</button></div>
      <p className="mt-1 text-xs leading-5 text-slate-600">编号表示类别；名称和颜色用于展示。分类规则在计算步骤中配置。</p>
      <div className="mt-3 space-y-3">{enumItems.map((item, i) => <div key={i} className="grid grid-cols-[2.5rem_minmax(0,1fr)_minmax(0,1fr)_auto] items-end gap-2">
        <label className="text-xs text-slate-600">颜色<input type="color" aria-label={`状态${i + 1}颜色`} value={item.color || '#64748b'} onChange={event => onEnumChange?.(enumItems.map((value, j) => i === j ? { ...value, color: event.target.value } : value))} className="mt-1 h-10 w-10 rounded-lg border border-slate-200" /></label>
        <label className="text-xs text-slate-600">稳定编号<input aria-label={`状态${i + 1}编号`} value={item.id} onChange={event => onEnumChange?.(enumItems.map((value, j) => i === j ? { ...value, id: event.target.value } : value))} className={field} /></label>
        <label className="text-xs text-slate-600">显示名称<input aria-label={`状态${i + 1}名称`} value={item.label} onChange={event => onEnumChange?.(enumItems.map((value, j) => i === j ? { ...value, label: event.target.value } : value))} className={field} /></label>
        <button type="button" aria-label={`删除状态${i + 1}`} disabled={enumItems.length <= 2 || !onEnumChange} onClick={() => onEnumChange?.(enumItems.filter((_, j) => i !== j))} className="min-h-10 px-2 text-xs text-rose-600 disabled:opacity-40">删除</button>
      </div>)}</div>
      <p className="mt-3 rounded-lg bg-slate-50 p-2 text-xs text-slate-600">缺失值：未识别。样本不足、未确认区间和无效数据不会归入震荡。</p>
    </section>}
  </div>
}

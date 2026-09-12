import { useEffect, useState } from 'react'

const FIELD_LABELS: Record<string, string> = {
  feature: '特征', feature_id: '特征', column: '字段', field: '字段', operator: '比较方式', op: '比较方式',
  threshold: '阈值', value: '比较值', state_id: '目标状态', state: '目标状态', label: '名称', name: '名称',
  weight: '权重', min: '下限', max: '上限', lower: '下限', upper: '上限', priority: '优先级',
  source: '来源', target: '目标', role: '经济含义', order: '顺序', color: '颜色', condition: '条件',
}

function ScalarField({ label, value, onChange }: { label: string; value: unknown; onChange: (value: unknown) => void }) {
  const [text, setText] = useState(String(value ?? ''))
  useEffect(() => { setText(String(value ?? '')) }, [value])
  if (typeof value === 'boolean') return <label className="flex items-center gap-2 text-xs text-slate-700"><input type="checkbox" aria-label={label} checked={value} onChange={(event) => onChange(event.target.checked)} />{label}</label>
  return <label className="block min-w-0 text-xs font-semibold text-slate-600">{label}<input aria-label={label} type={typeof value === 'number' ? 'number' : 'text'} step="any" value={text} onChange={(event) => { const next = event.target.value; setText(next); if (typeof value !== 'number') onChange(next); else if (next.trim() && Number.isFinite(Number(next))) onChange(Number(next)) }} onBlur={() => { if (typeof value === 'number' && (!text.trim() || !Number.isFinite(Number(text)))) setText(String(value)) }} className="mt-1 min-h-9 w-full min-w-0 rounded-xl border border-slate-300 bg-white px-2 font-normal" /></label>
}

// Edits only existing fields. No schema guessing, conversion, or graph reconstruction.
export default function RegimeStructuredParameter({ label, value, onChange }: { label: string; value: unknown; onChange: (value: unknown) => void }) {
  if (Array.isArray(value)) return <div className="space-y-2">{value.map((item, index) => <div key={index} className="rounded-lg border border-slate-200 bg-slate-50 p-2"><div className="mb-2 flex items-center justify-between gap-2"><span className="text-xs font-bold text-slate-600">第 {index + 1} 项</span><div className="flex gap-2"><button type="button" onClick={() => onChange([...value.slice(0, index + 1), JSON.parse(JSON.stringify(item)), ...value.slice(index + 1)])} className="text-xs font-bold text-accent-700">复制</button><button type="button" aria-label={`删除${label}第${index + 1}项`} onClick={() => onChange(value.filter((_, row) => row !== index))} className="text-xs font-bold text-rose-600">删除</button></div></div><RegimeStructuredParameter label={`${label}第${index + 1}项`} value={item} onChange={(next) => onChange(value.map((entry, row) => row === index ? next : entry))} /></div>)}{!value.length ? <p className="text-xs text-slate-600">当前列表为空。可从模板载入规则，或使用高级编辑添加结构。</p> : null}</div>
  if (value && typeof value === 'object') return <div className="grid min-w-0 gap-2 sm:grid-cols-2">{Object.entries(value as Record<string, unknown>).map(([key, item]) => item !== null && typeof item === 'object' ? <details key={key} className="min-w-0 rounded-lg border border-slate-200 p-2 sm:col-span-2"><summary className="cursor-pointer text-xs font-bold text-slate-600">{FIELD_LABELS[key] || key}</summary><div className="mt-2"><RegimeStructuredParameter label={`${label}${FIELD_LABELS[key] || key}`} value={item} onChange={(next) => onChange({ ...value, [key]: next })} /></div></details> : <ScalarField key={key} label={`${label} · ${FIELD_LABELS[key] || key}`} value={item} onChange={(next) => onChange({ ...value, [key]: next })} />)}</div>
  return <ScalarField label={label} value={value} onChange={onChange} />
}

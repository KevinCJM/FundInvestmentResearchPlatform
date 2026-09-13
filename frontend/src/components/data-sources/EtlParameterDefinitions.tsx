import type { EtlParameter } from '../../services/etl'
import { buttonClass, inputClass, JsonField } from './EditorFields'

export default function EtlParameterDefinitions({ value, onChange }: { value: EtlParameter[]; onChange: (value: EtlParameter[]) => void }) {
  const patch = (index: number, next: EtlParameter) => onChange(value.map((p, i) => i === index ? next : p))
  return <details className="rounded-lg border border-slate-200 p-3"><summary className="cursor-pointer text-sm font-semibold">定义运行参数 · {value.length} 项</summary>
    <p className="mt-3 text-xs leading-5 text-slate-600">这些参数决定每次运行显示哪些输入框；不同流程共用同一套表单。修改参数 ID 后，需要同步调整步骤绑定。</p>
    <div className="mt-3 space-y-3">{value.map((p, index) => <div key={index} className="grid gap-3 rounded-lg bg-slate-50 p-3 sm:grid-cols-2">
      <label className="text-xs">参数标识<input aria-label={`参数 ${index + 1} 标识`} className={inputClass} value={p.id} onChange={e => patch(index, { ...p, id: e.target.value })} /></label>
      <label className="text-xs">显示名称<input aria-label={`参数 ${index + 1} 名称`} className={inputClass} value={p.label} onChange={e => patch(index, { ...p, label: e.target.value })} /></label>
      <label className="text-xs">参数类型<select aria-label={`参数 ${index + 1} 类型`} className={inputClass} value={p.data_type} onChange={e => patch(index, { ...p, data_type: e.target.value as EtlParameter['data_type'] })}><option value="text">文本</option><option value="date">日期</option></select></label>
      <label className="text-xs">默认值<input aria-label={`参数 ${index + 1} 默认值`} className={inputClass} type={p.data_type === 'date' ? 'date' : 'text'} value={p.default} onChange={e => patch(index, { ...p, default: e.target.value })} /></label>
      {p.data_type === 'date' ? <label className="text-xs">发送日期格式<select className={inputClass} value={p.date_format} onChange={e => patch(index, { ...p, date_format: e.target.value as EtlParameter['date_format'] })}><option value="iso">YYYY-MM-DD</option><option value="compact">YYYYMMDD</option></select></label> : null}
      <label className="flex items-center gap-2 text-xs"><input type="checkbox" checked={p.required} onChange={e => patch(index, { ...p, required: e.target.checked })} />运行时必填</label>
      <button type="button" className={buttonClass} onClick={() => onChange(value.filter((_, i) => i !== index))}>删除参数 {index + 1}</button>
    </div>)}</div>
    <button type="button" className={`${buttonClass} mt-3`} disabled={value.length >= 30} onClick={() => onChange([...value, { id: `param_${value.length + 1}`, label: '新运行参数', data_type: 'text', default: '', required: true, date_format: 'iso', description: '' }])}>添加运行参数</button>
    <details className="mt-3"><summary className="cursor-pointer text-xs">高级：参数 JSON</summary><JsonField label="运行参数定义 JSON" objectOnly={false} value={value} onChange={next => { if (!Array.isArray(next) || next.some(p => !p || typeof p.id !== 'string' || typeof p.label !== 'string' || !['text','date'].includes(p.data_type))) throw new Error('请提供有效参数数组。'); onChange(next as EtlParameter[]) }} /></details>
  </details>
}

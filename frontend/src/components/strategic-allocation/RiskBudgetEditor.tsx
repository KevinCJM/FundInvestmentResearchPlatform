import { Field, inputClass, NumberInput, percentText } from '../risk-models/ResearchUI'
import { percentInputValue } from '../../services/strategicAllocation'

export function riskBudgetError(assets: string[], budget?: Record<string, number> | null): string | null {
  if (budget == null) return null
  if (Object.keys(budget).length !== assets.length || assets.some(a => !Number.isFinite(budget[a]) || budget[a] < 0)) return '请填写每项资产的非负风险预算；空白不代表零。'
  if (Math.abs(Object.values(budget).reduce((sum, n) => sum + n, 0) - 1) > 1e-8) return '风险预算百分比必须合计100%。'
  return null
}

export default function RiskBudgetEditor({ assets, value, onChange }: { assets: string[]; value?: Record<string, number> | null; onChange: (value: Record<string, number> | null) => void }) {
  const error = riskBudgetError(assets, value)
  return <div className="space-y-3 border-t border-slate-200 pt-4" aria-label="风险预算比较设置">
    <label className="flex min-h-10 items-center gap-2 text-sm"><input type="checkbox" checked={value != null} onChange={e => onChange(e.target.checked ? Object.fromEntries(assets.map(a => [a, NaN])) : null)} />增加风险预算候选</label>
    {value != null && <>
      <p className="text-sm leading-6 text-slate-600">填写希望各资产承担的风险比例。与原四类候选共用约束和有限搜索；距离最小不代表精确风险平价。</p>
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">{assets.map(a => <Field key={a} label={`${a}风险预算（%）`}><NumberInput className={`${inputClass} tabular-nums placeholder:text-slate-600 placeholder:opacity-100`} value={percentInputValue(value[a])} min={0} max={100} onValueChange={n => onChange({ ...value, [a]: n / 100 })} /></Field>)}</div>
      <p className="text-sm tabular-nums text-slate-600">风险预算合计：{percentText(Object.values(value).reduce((sum, n) => sum + n, 0))}</p>
      {error && <p role="status" className="text-sm text-amber-800">{error}</p>}
    </>}
  </div>
}

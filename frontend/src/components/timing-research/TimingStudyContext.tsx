import type { TimingAdaptation, TimingBaskets, TimingDefinition } from '../../services/timingResearch'
import { timingField } from './TimingRuleEditor'

export type BasketDraft = { market: string; category: string }
export function basketCodes(text: string): string[] { return text.split(/[,，;；\s]+/).map(code => code.trim().toUpperCase()).filter(Boolean) }
export function basketsFromDraft(draft: BasketDraft): TimingBaskets { return { market: basketCodes(draft.market), category: basketCodes(draft.category) } }
export function basketValidation(definition: TimingDefinition, draft: BasketDraft): string {
  for (const group of ['market', 'category'] as const) {
    const codes = basketCodes(draft[group]), label = group === 'market' ? '市场参考篮子' : '同类参考篮子'
    if (codes.some(code => !/^\d{6}\.(SH|SZ)$/.test(code))) return `${label}代码格式应为 510300.SH 或 159919.SZ。`
    if (new Set(codes).size !== codes.length) return `${label}存在重复代码，请移除重复项。`
    if (codes.length > 12 || codes.length === 1) return `${label}需要 2–12 个 ETF；不用时可留空。`
    const required = definition.nodes.some(node => node.op === 'basket_source' && (node.parameters.group || 'market') === group)
    if (required && codes.length < 2) return `算法需要${label}，请显式填写至少 2 个 ETF。`
  }
  return ''
}
export function TimingBasketEditor({ value, onChange, definition }: { value: BasketDraft; onChange: (value: BasketDraft) => void; definition: TimingDefinition }) {
  const required = definition.nodes.some(node => node.op === 'basket_source')
  return <details open={required || undefined} className="border-t border-slate-100 pt-3"><summary className="cursor-pointer py-2 text-sm font-medium text-slate-700">参考 ETF 篮子{required ? ' · 当前算法需要' : ' · 可选'}</summary><p className="mt-2 text-xs leading-5 text-slate-600">只用于广度和横截面条件，不代表全市场。不会自动使用研究对象充当篮子，不合成投资组合。</p><div className="mt-3 space-y-3">{(['market', 'category'] as const).map(group => <label key={group} className="block text-xs text-slate-600">{group === 'market' ? '市场参考篮子代码' : '同类参考篮子代码'}<textarea aria-label={group === 'market' ? '市场参考篮子代码' : '同类参考篮子代码'} className={timingField} rows={3} spellCheck={false} placeholder="逗号或换行分隔，如 510300.SH" value={value[group]} onChange={event => onChange({ ...value, [group]: event.target.value })} /><span className="mt-1 block text-xs text-slate-600">已填 {basketCodes(value[group]).length} 个 · 每篮子 2–12 个 ETF</span></label>)}</div><p className="mt-2 text-xs leading-5 text-slate-600">不自动取日期交集，不补零；缺失保留未知。结果中可按 ETF 成员预览篮子步骤。不同篮子先各自汇总，不能直接逐列混算。</p></details>
}
export function TimingAdaptationNote({ adaptation }: { adaptation?: TimingAdaptation }) {
  if (!adaptation) return null
  return <details className="rounded-xl border border-amber-200 bg-amber-50/50 p-3"><summary className="cursor-pointer text-sm font-medium text-amber-900">ETF 改编说明 · {adaptation.source_experiments.join(' / ')}</summary><p className="mt-2 text-xs leading-5 text-amber-900">{adaptation.version} 是新的 ETF 研究版本，不是原 A 股实验复现，不继承原实验评级或收益。</p><div className="mt-3 grid gap-3 sm:grid-cols-2"><div><h3 className="text-xs font-semibold text-slate-800">保留的思路</h3><ul className="mt-2 list-disc space-y-1 pl-4 text-xs leading-5 text-slate-600">{adaptation.preserved.map((item, index) => <li key={index}>{item}</li>)}</ul></div><div><h3 className="text-xs font-semibold text-slate-800">改编后的差异</h3><ul className="mt-2 list-disc space-y-1 pl-4 text-xs leading-5 text-slate-600">{adaptation.changed.map((item, index) => <li key={index}>{item}</li>)}</ul></div></div></details>
}

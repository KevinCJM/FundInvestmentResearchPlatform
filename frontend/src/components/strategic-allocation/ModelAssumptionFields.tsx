import { Field, inputClass, NumberInput, sectionClass } from '../risk-models/ResearchUI'
import { percentInputValue, type CmaDraft, type EconomicRole } from '../../services/strategicAllocation'

/** Metadata and explicit robust-box assumptions remain independent of the model. */
export default function ModelAssumptionFields({ value, onChange }: { value: CmaDraft; onChange: (value: CmaDraft) => void }) {
  const patch = (index: number, change: Partial<CmaDraft['assets'][number]>) => onChange({ ...value, assets: value.assets.map((a, i) => i === index ? { ...a, ...change } : a) })
  return <section className={`${sectionClass} space-y-4`} aria-label="模型假设口径与经济角色">
    <h2 className="text-lg font-semibold">确认依据与经济用途</h2>
    <div className="grid gap-3 sm:grid-cols-2">
      <Field label="假设版本名称"><input className={inputClass} value={value.name} maxLength={120} onChange={e => onChange({ ...value, name: e.target.value })} /></Field>
      <Field label="假设研究日"><input type="date" className={inputClass} value={value.as_of} onChange={e => onChange({ ...value, as_of: e.target.value })} /></Field>
    </div>
    <p className="text-sm text-slate-600">{value.currency} · {value.horizon_years} 年 · 年化算术总收益。有效收益与风险由服务端模型计算。</p>
    <Field label="预测来源与主要假设"><textarea className={inputClass} value={value.source} onChange={e => onChange({ ...value, source: e.target.value })} /></Field>
    <div className="divide-y divide-slate-200">{value.assets.map((asset, i) => <fieldset key={asset.id} className="min-w-0 py-4"><legend className="text-sm font-semibold">{asset.id}</legend>
      <div className="grid gap-3 sm:grid-cols-2">
        <Field label={`${asset.id}经济角色`}><select className={inputClass} value={asset.role} disabled={Boolean(value.strategic_universe_id)} onChange={e => patch(i, { role: e.target.value as EconomicRole })}><option value="">请选择用途</option>{([['growth', '增长参与'], ['rates', '利率防御'], ['inflation', '通胀分散'], ['credit', '信用收益'], ['liquidity', '流动性储备'], ['diversifier', '其他分散用途']] as const).map(([id, label]) => <option key={id} value={id}>{label}</option>)}</select></Field>
        <Field label={`${asset.id}流动性`}><select className={inputClass} value={asset.liquidity} disabled={Boolean(value.strategic_universe_id)} onChange={e => patch(i, { liquidity: e.target.value as 'liquid' | 'illiquid' })}><option value="">请确认可变现性</option><option value="liquid">可提供流动性</option><option value="illiquid">非流动性资产</option></select></Field>
        <Field label={`${asset.id}分类与代理理由`}><input className={inputClass} value={asset.rationale} onChange={e => patch(i, { rationale: e.target.value })} /></Field>
        <Field label={`${asset.id}均值不确定半宽（百分点）`} hint="显式稳健区间假设，不是 BL 后验标准差。"><NumberInput aria-label={`${asset.id}均值不确定半宽（百分点）`} className={`${inputClass} tabular-nums`} value={percentInputValue(asset.mean_uncertainty)} min={0} max={100} onValueChange={n => patch(i, { mean_uncertainty: n / 100 })} /></Field>
      </div>
    </fieldset>)}</div>
    {value.strategic_universe_id && <p className="text-xs text-slate-600">角色与流动性来自不可变战略范围；更改须先建立新的范围版本。</p>}
    <label className="flex min-h-10 items-start gap-2 text-sm leading-6"><input type="checkbox" className="mt-1.5" checked={value.basis_confirmed} onChange={e => onChange({ ...value, basis_confirmed: e.target.checked })} />我已确认同币种、年化算术总收益、经济角色、流动性与模型依据。</label>
  </section>
}

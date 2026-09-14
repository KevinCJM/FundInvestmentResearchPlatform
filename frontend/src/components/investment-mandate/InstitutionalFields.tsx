import { Field, inputClass, NumberInput } from '../risk-models/ResearchUI'
import type { MandateDefinition } from '../../services/strategicAllocation'
import { newInstitution, reviewTopics, type InstitutionalContext, type BalanceSheet } from '../../services/institutionalContext'

const types: Array<[InstitutionalContext['investor_type'], string]> = [['personal', '个人'], ['family_office', '家族办公室'], ['asset_manager', '资管组合'], ['corporate_treasury', '企业现金']]
const amounts: Array<[keyof Pick<BalanceSheet, 'investable_assets' | 'outside_assets' | 'confirmed_liabilities' | 'uncalled_commitments'>, string]> = [['investable_assets', '可投资资产'], ['outside_assets', '组合外资产'], ['confirmed_liabilities', '已确认负债'], ['uncalled_commitments', '未缴承诺']]
export default function InstitutionalFields({ value, onChange }: { value: MandateDefinition; onChange: (patch: Partial<MandateDefinition>) => void }) {
  const context = value.institutional_context
  const patch = (next: Partial<InstitutionalContext>) => context && onChange({ institutional_context: { ...context, ...next } })
  const balance = context?.balance_sheet
  return <div className="space-y-4 border-b border-slate-200 pb-5">
    <Field label="资金用途场景（可选）" hint="选择场景不会自动调整风险限额。"><select className={inputClass} value={context?.investor_type ?? ''} onChange={e => onChange({ institutional_context: e.target.value ? { ...(context ?? newInstitution(e.target.value as InstitutionalContext['investor_type'])), investor_type: e.target.value as InstitutionalContext['investor_type'] } : null })}><option value="">沿用一般目标</option>{types.map(([id, label]) => <option key={id} value={id}>{label}</option>)}</select></Field>
    {context && <>
      <Field label="资金用途说明"><textarea className={inputClass} value={context.purpose} onChange={e => patch({ purpose: e.target.value })} /></Field>
      <Field label="组合内现金用途下限（%）" hint="仅明确的可流动现金角色满足此硬约束；可交易权益ETF不等于经营或储备现金。"><NumberInput className={inputClass} value={context.cash_reserve_weight * 100} min={0} max={100} onValueChange={n => patch({ cash_reserve_weight: n / 100 })} /></Field>
      <details className="space-y-4"><summary className="min-h-10 cursor-pointer text-sm font-medium">经济状况与人工核验</summary>
        <p className="text-sm leading-6 text-slate-600">税务、监管、对冲、杠杆和特殊流动性未自动建模。研究员已核对不等于独立审批；待核验可以保存研究，但不能交接产品应用。</p>
        <label className="flex min-h-10 items-center gap-2 text-sm"><input type="checkbox" checked={Boolean(balance)} onChange={e => patch({ balance_sheet: e.target.checked ? { as_of: value.as_of, currency: value.currency, source: '', investable_assets: null, outside_assets: null, confirmed_liabilities: null, uncalled_commitments: null } : null })} />提供经济状况研究快照</label>
        {balance && <div className="grid gap-4 sm:grid-cols-2">
          <Field label="经济快照日期"><input type="date" className={inputClass} value={balance.as_of} onChange={e => patch({ balance_sheet: { ...balance, as_of: e.target.value } })} /></Field>
          <Field label="经济快照币种"><input className={inputClass} value={balance.currency} onChange={e => patch({ balance_sheet: { ...balance, currency: e.target.value.toUpperCase() } })} /></Field>
          <Field label="经济快照来源"><input className={inputClass} value={balance.source} onChange={e => patch({ balance_sheet: { ...balance, source: e.target.value } })} /></Field>
          {amounts.map(([key, label]) => <Field key={key} label={`${label}（${balance.currency}）`} hint="未提供保持空白，不自动补零。"><NumberInput className={`${inputClass} tabular-nums`} value={balance[key] ?? NaN} min={0} onValueChange={n => patch({ balance_sheet: { ...balance, [key]: Number.isNaN(n) ? null : n } })} /></Field>)}
          <p className="text-sm text-slate-600 sm:col-span-2">未缴承诺单独披露，不加入资产、不自动视为当前负债。支付计划继续在原现金流区填写；本快照不代替资金目标或法定账簿。</p>
        </div>}
        <div className="divide-y divide-slate-200">{reviewTopics.map(([topic, label]) => {
          const item = context.review_items.find(row => row.topic === topic)!
          const update = (next: Partial<typeof item>) => patch({ review_items: context.review_items.map(row => row.topic === topic ? { ...row, ...next } : row) })
          return <fieldset key={topic} className="min-w-0 space-y-3 py-4"><legend className="text-sm font-semibold">{label}</legend>
            <Field label={`${label}核验状态`}><select className={inputClass} value={item.status} onChange={e => update({ status: e.target.value as typeof item.status })}><option value="not_assessed">未评估</option><option value="pending">待核对</option><option value="researcher_checked">研究员已核对</option><option value="not_applicable">有依据的不适用</option></select></Field>
            {item.status !== 'not_assessed' && <div className="grid gap-3 sm:grid-cols-2"><Field label={`${label}理由`}><input className={inputClass} value={item.reason} onChange={e => update({ reason: e.target.value })} /></Field><Field label={`${label}证据`}><input className={inputClass} value={item.evidence} onChange={e => update({ evidence: e.target.value })} /></Field><Field label={`${label}核验日`}><input type="date" className={inputClass} value={item.reviewed_on ?? ''} max={value.as_of} onChange={e => update({ reviewed_on: e.target.value || null })} /></Field><Field label={`${label}复核日`}><input type="date" className={inputClass} value={item.valid_until ?? ''} onChange={e => update({ valid_until: e.target.value || null })} /></Field></div>}
          </fieldset>
        })}</div>
      </details>
    </>}
  </div>
}

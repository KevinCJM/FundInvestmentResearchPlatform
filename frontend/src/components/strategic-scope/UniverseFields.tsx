import { Button } from '../ui'
import { Field, inputClass } from '../risk-models/ResearchUI'
import { economicRoles, type UniverseDefinition, type StrategicAsset } from '../../services/strategicScope'
export default function UniverseFields({ value, onChange }: { value: UniverseDefinition; onChange: (value: UniverseDefinition) => void }) {
  const patchAsset = (index: number, patch: Partial<StrategicAsset>) => onChange({ ...value, assets: value.assets.map((a, i) => i === index ? { ...a, ...patch } : a) })
  return <div className="space-y-4">
    <div className="grid gap-4 sm:grid-cols-3"><Field label="战略范围名称"><input className={inputClass} value={value.name} onChange={e => onChange({ ...value, name: e.target.value })} /></Field><Field label="战略研究日"><input type="date" className={inputClass} value={value.as_of} onChange={e => onChange({ ...value, as_of: e.target.value })} /></Field><Field label="战略本位币"><input className={inputClass} value={value.currency} maxLength={3} onChange={e => onChange({ ...value, currency: e.target.value.toUpperCase() })} /></Field></div>
    <Field label="战略范围来源"><textarea className={inputClass} value={value.source} onChange={e => onChange({ ...value, source: e.target.value })} /></Field>
    <div className="divide-y divide-slate-200">{value.assets.map((asset, index) => <fieldset key={index} className="min-w-0 space-y-3 py-4"><legend className="text-sm font-semibold">战略资产 {index + 1}</legend><div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
      <Field label={`资产${index + 1}稳定ID`} hint="小写字母开头，可用数字、下划线与短横线；不从产品名推断。"><input className={inputClass} value={asset.id} onChange={e => patchAsset(index, { id: e.target.value })} /></Field>
      <Field label={`资产${index + 1}展示名称`}><input className={inputClass} value={asset.name} onChange={e => patchAsset(index, { name: e.target.value })} /></Field>
      <Field label={`资产${index + 1}风险计价币种`}><input className={inputClass} value={asset.currency} onChange={e => patchAsset(index, { currency: e.target.value.toUpperCase() })} /></Field>
      <Field label={`资产${index + 1}经济角色`}><select className={inputClass} value={asset.role} onChange={e => patchAsset(index, { role: e.target.value as StrategicAsset['role'] })}>{economicRoles.map(([id, label]) => <option key={id} value={id}>{label}</option>)}</select></Field>
      <Field label={`资产${index + 1}流动性`}><select className={inputClass} value={asset.liquidity} onChange={e => patchAsset(index, { liquidity: e.target.value as StrategicAsset['liquidity'] })}><option value="liquid">可提供流动性</option><option value="illiquid">非流动性资产</option></select></Field>
      <Field label={`资产${index + 1}定义理由`}><input className={inputClass} value={asset.rationale} onChange={e => patchAsset(index, { rationale: e.target.value })} /></Field>
      <Field label={`资产${index + 1}来源`}><input className={inputClass} value={asset.source} onChange={e => patchAsset(index, { source: e.target.value })} /></Field>
    </div><Button onClick={() => onChange({ ...value, assets: value.assets.filter((_, i) => i !== index) })}>移除战略资产 {index + 1}</Button></fieldset>)}</div>
    <Button disabled={value.assets.length >= 30} onClick={() => onChange({ ...value, assets: [...value.assets, { id: '', name: '', currency: value.currency, role: 'growth', liquidity: 'liquid', rationale: '', source: '' }] })}>增加战略资产</Button>
    {!value.assets.length && <p className="text-sm text-slate-600">尚未定义战略资产；增加资产后可先做研究，无需选择产品。</p>}
    {value.assets.length >= 30 && <p className="text-xs text-slate-600">每个范围最多30类资产。</p>}
  </div>
}

import { Button } from '../ui'
import { Field, inputClass, NumberInput } from '../risk-models/ResearchUI'
import { percentInputValue as percent, type MandateDefinition, type StrategicCatalog } from '../../services/strategicAllocation'

export default function AssetAuthorizations({ value, catalog, onChange }: {
  value: MandateDefinition; catalog: StrategicCatalog | null; onChange: (value: Partial<MandateDefinition>) => void
}) {
  const scope = value.allocation_scope ?? ''
  const assets = catalog?.allocations.find(a => a.alloc_name === scope)?.assets ?? []
  const groups = value.group_limits ?? []
  return <details className="border-t border-slate-200 pt-4"><summary className="cursor-pointer text-sm font-medium">限定可投大类与联合比例（可选）</summary>
    <div className="mt-4 space-y-4">
      <Field label="授权的大类方案" hint="选择后绑定此方案；后续SAA只能收紧权重边界，不能放宽。清空选择会清除对应的授权边界。"><select className={inputClass} value={scope} onChange={e => {
        const item = catalog?.allocations.find(a => a.alloc_name === e.target.value)
        onChange({ allocation_scope: e.target.value || null, asset_limits: Object.fromEntries((item?.assets ?? []).map(a => [a.id, { min_weight: 0, max_weight: 1, max_abs_tilt: .1 }])), group_limits: [] })
      }}><option value="">暂不限定具体大类方案</option>{catalog?.allocations.map(a => <option key={a.alloc_name} value={a.alloc_name}>{a.alloc_name}</option>)}</select></Field>
      {assets.length > 0 && <>
        <div className="overflow-x-auto"><table aria-label="投资授权资产边界" className="w-full min-w-[470px] text-sm"><thead><tr><th scope="col" className="p-2 text-left">大类</th><th scope="col" className="p-2 text-right">最低（%）</th><th scope="col" className="p-2 text-right">最高（%）</th><th scope="col" className="p-2 text-right">TAA偏离（百分点）</th></tr></thead><tbody>{assets.map(a => <tr key={a.id} className="border-b border-slate-200"><th scope="row" className="p-2 text-left font-medium">{a.name}</th>{(['min_weight', 'max_weight', 'max_abs_tilt'] as const).map((key, i) => <td className="p-2" key={key}><NumberInput aria-label={`${a.name}授权${['最低', '最高', '偏离'][i]}`} className={`${inputClass} text-right tabular-nums`} value={percent(value.asset_limits?.[a.id]?.[key] ?? NaN)} onValueChange={n => onChange({ asset_limits: { ...value.asset_limits, [a.id]: { ...value.asset_limits![a.id], [key]: n / 100 } } })} /></td>)}</tr>)}</tbody></table></div>
        {groups.map((g, index) => <fieldset key={index} className="min-w-0 space-y-3 border-b border-slate-200 pb-4"><legend className="text-sm font-medium">授权联合约束 {index + 1}</legend><div className="grid gap-3 sm:grid-cols-2">{(['lo', 'hi'] as const).map(key => <Field key={key} label={`授权联合约束 ${index + 1}${key === 'lo' ? '最低' : '最高'}（%）`}><NumberInput className={inputClass} value={percent(g[key])} onValueChange={n => onChange({ group_limits: groups.map((item, i) => i === index ? { ...item, [key]: n / 100 } : item) })} /></Field>)}</div><div className="flex flex-wrap gap-4">{assets.map(a => <label key={a.id} className="flex min-h-10 items-center gap-2 text-sm"><input type="checkbox" checked={g.assets.includes(a.id)} onChange={e => onChange({ group_limits: groups.map((item, i) => i === index ? { ...item, assets: e.target.checked ? [...item.assets, a.id] : item.assets.filter(id => id !== a.id) } : item) })} />{a.name}</label>)}</div><Button onClick={() => onChange({ group_limits: groups.filter((_, i) => i !== index) })}>移除授权联合约束 {index + 1}</Button></fieldset>)}
        <Button disabled={groups.length >= 24} onClick={() => onChange({ group_limits: [...groups, { id: `mandate-group-${Date.now()}`, assets: [], lo: 0, hi: 1 }] })}>增加授权联合约束</Button>
      </>}
    </div>
  </details>
}

import { Button } from '../ui'
import { Field } from '../risk-models/ResearchUI'
import { metadata, numericValues, textValue, type ReferenceVersion, type RiskScaleDefinition } from '../../services/riskScales'
import { controlClass, PercentField, pct, useRiskText } from './shared'

type AssetOption = { id: string; name: string }

function versionAssetNames(version: ReferenceVersion) {
  const definition = metadata(version.definition)
  const assets = Array.isArray(definition.assets) ? definition.assets : []
  return new Map(assets.map(item => {
    const asset = metadata(item)
    const id = textValue(asset.id)
    return [id, textValue(asset.name) || id] as const
  }).filter(([id]) => Boolean(id)))
}

function GroupBounds({ value, assets, onChange }: { value: RiskScaleDefinition; assets: AssetOption[]; onChange: (next: RiskScaleDefinition) => void }) {
  const { t } = useRiskText(), groups = value.constraint_profile?.group_limits ?? []
  const change = (next: typeof groups) => onChange({ ...value, constraint_profile: { ...value.constraint_profile, group_limits: next } })
  return <div className="space-y-2"><Button disabled={groups.length >= 28 || !assets.length} onClick={() => change([...groups, { id: `group-${crypto.randomUUID().slice(0, 8)}`, assets: [], lo: 0, hi: 1 }])}>{t('addGroup')}</Button>{groups.map((group, index) => <div className="space-y-2 border-b border-slate-200 py-2" key={index}><div className="grid gap-2 sm:grid-cols-3"><Field label={t('groupName')} required><input required className={controlClass} value={group.id} onChange={event => change(groups.map((item, i) => i === index ? { ...item, id: event.target.value } : item))} /></Field><PercentField required label={t('minWeight')} value={group.lo ?? NaN} onChange={lo => change(groups.map((item, i) => i === index ? { ...item, lo } : item))} /><PercentField required label={t('maxWeight')} value={group.hi ?? NaN} onChange={hi => change(groups.map((item, i) => i === index ? { ...item, hi } : item))} /></div><fieldset><legend className="text-sm font-medium">{t('groupMembers')}</legend><div className="grid gap-1 sm:grid-cols-2 xl:grid-cols-3">{assets.map(asset => <label key={asset.id} className="flex min-h-10 items-center gap-2 text-sm"><input type="checkbox" checked={group.assets.includes(asset.id)} onChange={event => change(groups.map((item, i) => i === index ? { ...item, assets: event.target.checked ? [...item.assets, asset.id] : item.assets.filter(id => id !== asset.id) } : item))} />{asset.name}</label>)}</div></fieldset><Button onClick={() => change(groups.filter((_, i) => i !== index))}>{t('remove')}</Button></div>)}</div>
}

export function FrozenReferenceDetails({ version }: { version: ReferenceVersion }) {
  const { t } = useRiskText()
  const moments = version.moments, quality = version.quality ?? {}, names = versionAssetNames(version)
  return <div className="space-y-2"><p className="text-sm font-medium">{t('intersectionRange', { start: textValue(quality.intersection_start) || t('unavailable'), end: textValue(quality.intersection_end) || t('unavailable') })}</p><p className="text-xs text-slate-600">{t('intersectionMeaning')}</p><div className="overflow-x-auto"><table className="w-full text-sm" aria-label={t('historicalParameterSummary')}><thead><tr>{['assetName', 'historicalAnnualReturn', 'volatility'].map(key => <th scope="col" className="px-2 py-1.5 text-left" key={key}>{t(key)}</th>)}</tr></thead><tbody>{version.ordered_asset_ids.map((id, index) => <tr key={id} className="border-b border-slate-200"><th scope="row" className="px-2 py-1.5 text-left">{names.get(id) ?? id}</th><td className="px-2 py-1.5 text-right tabular-nums">{pct(numericValues(moments?.annual_returns)[index])}</td><td className="px-2 py-1.5 text-right tabular-nums">{pct(numericValues(moments?.annual_volatilities)[index])}</td></tr>)}</tbody></table></div></div>
}

export function BoundsEditor({ value, assets, onChange }: { value: RiskScaleDefinition; assets: AssetOption[]; onChange: (next: RiskScaleDefinition) => void }) {
  const { t } = useRiskText()
  const profile = value.constraint_profile!, configured = Object.keys(profile.asset_limits ?? {}).length
  const changeBound = (id: string, field: 'min_weight' | 'max_weight', nextValue: number) => {
    const current = profile.asset_limits?.[id] ?? { min_weight: 0, max_weight: 1 }
    const next = { ...current, [field]: nextValue }
    const limits = { ...(profile.asset_limits ?? {}) }
    if ((next.min_weight ?? 0) === 0 && (next.max_weight ?? 1) === 1) delete limits[id]
    else limits[id] = next
    onChange({ ...value, constraint_profile: { ...profile, asset_limits: limits } })
  }
  return <section className="space-y-2"><h3 className="text-base font-semibold">{t('constraints')}</h3><p className="text-xs text-slate-600">{t('constraintHint')}</p><details><summary className="min-h-10 cursor-pointer py-2 text-sm font-medium">{t('assetConstraints')}{configured ? ` · ${t('configuredCount', { count: configured })}` : ` · ${t('unconstrained')}`}</summary><div className="divide-y divide-slate-200">{assets.map(asset => { const bound = profile.asset_limits?.[asset.id] ?? { min_weight: 0, max_weight: 1 }; return <div className="grid items-end gap-2 py-2 sm:grid-cols-3" key={asset.id}><span className="break-words text-sm font-medium">{asset.name}</span><PercentField required label={t('minWeight')} value={bound.min_weight ?? 0} onChange={min_weight => changeBound(asset.id, 'min_weight', min_weight)} /><PercentField required label={t('maxWeight')} value={bound.max_weight ?? 1} onChange={max_weight => changeBound(asset.id, 'max_weight', max_weight)} /></div> })}</div></details><details><summary className="min-h-10 cursor-pointer py-2 text-sm font-medium">{t('groupConstraints')}</summary><p className="text-xs text-slate-600">{t('groupConstraintsHint')}</p><GroupBounds value={value} assets={assets} onChange={onChange} /></details></section>
}

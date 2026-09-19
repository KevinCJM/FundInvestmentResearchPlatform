import { useState } from 'react'
import { Button } from '../ui'
import { Field } from '../risk-models/ResearchUI'
import type { ReferenceAsset, ReferenceInputRequest } from '../../services/riskScales'
import { controlClass, PercentField, pct, useRiskText } from './shared'
import { SourcePicker } from './SourcePicker'

export function ReferenceEditor({ value, sourceLabels = {}, onChange, fixedAssets = false, cashEligibleIds }: { value: ReferenceInputRequest; sourceLabels?: Record<string, string>; onChange: (next: ReferenceInputRequest, labels?: Record<string, string>) => void; fixedAssets?: boolean; cashEligibleIds?: string[] }) {
  const { t } = useRiskText(), [selecting, setSelecting] = useState<string | null>(null)
  const update = (index: number, change: Partial<ReferenceAsset>) => onChange({ ...value, assets: value.assets.map((asset, i) => i === index ? { ...asset, ...change } : asset) })
  const newAsset = (id: string, name = '', assetType: ReferenceAsset['asset_type'] = 'market'): ReferenceAsset => assetType === 'cash'
    ? { id, name, asset_type: 'cash', rationale: '', cash_return: 0, components: [], rebalance: null }
    : { id, name, asset_type: 'market', rationale: '', cash_return: null, components: [], rebalance: 'daily' }
  const cashIndex = value.assets.findIndex(asset => asset.asset_type === 'cash')
  const changeType = (index: number, assetType: ReferenceAsset['asset_type']) => {
    setSelecting(null)
    update(index, assetType === 'cash'
      ? { asset_type: 'cash', cash_return: 0, components: [], rebalance: null }
      : { asset_type: 'market', cash_return: null, components: [], rebalance: 'daily' })
  }
  return <div className="space-y-3">
    <div className="divide-y divide-slate-200">{value.assets.map((asset, index) => <section key={asset.id} data-testid="risk-reference-asset" className="space-y-3 py-3">
      <div className="flex flex-wrap items-center justify-between gap-2">{asset.name ? <h3 className="text-base font-semibold">{asset.name}</h3> : <span />}{!fixedAssets && <Button onClick={() => onChange({ ...value, assets: value.assets.filter((_, i) => i !== index) })}>{t('removeAsset')}</Button>}</div>
      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <Field label={t('assetName')} required><input required readOnly={fixedAssets} className={controlClass} value={asset.name} onChange={event => update(index, { name: event.target.value })} /></Field>
        <Field label={t('assetType')} required><select required className={controlClass} value={asset.asset_type} onChange={event => changeType(index, event.target.value as ReferenceAsset['asset_type'])}><option value="market">{t('marketAsset')}</option><option value="cash" disabled={cashIndex >= 0 && cashIndex !== index || Boolean(cashEligibleIds && !cashEligibleIds.includes(asset.id))}>{t('cashAsset')}</option></select></Field>
        {asset.asset_type === 'cash' ? <div className="min-w-0"><PercentField required label={t('cashReturn')} value={asset.cash_return ?? 0} onChange={cash_return => update(index, { cash_return })} /><p className="mt-1 text-xs leading-5 text-slate-600">{t('cashReferenceRate')}</p></div> : <Field label={t('rebalance')} required hint={t(`rebalance.${asset.rebalance ?? 'daily'}`)}><select required className={controlClass} value={asset.rebalance ?? 'daily'} onChange={event => update(index, { rebalance: event.target.value as Exclude<ReferenceAsset['rebalance'], null | undefined> })}>{['daily', 'monthly', 'quarterly', 'yearly', 'buy_and_hold'].map(rule => <option key={rule} value={rule}>{t(`frequency.${rule}`)}</option>)}</select></Field>}
        <Field label={t('assetRationale')}><input className={controlClass} value={asset.rationale ?? ''} onChange={event => update(index, { rationale: event.target.value })} /></Field>
      </div>
      {asset.asset_type === 'cash' ? <p className="text-xs leading-5 text-slate-600">{t('cashHint')}</p> : <>
        <p className="text-xs leading-5 text-slate-600">{t('marketHint')}</p>
        {asset.components.map((component, componentIndex) => { const parts = component.series_id.split(':'), code = parts[parts.length - 1] || component.series_id; return <div key={`${component.series_id}-${componentIndex}`} className="grid items-end gap-2 sm:grid-cols-[minmax(0,1fr)_120px_auto]"><div className="break-words text-sm">{sourceLabels[component.series_id] ?? code}<span className="block text-xs text-slate-600">{code} · {t(`source.${component.kind}`)}</span></div><PercentField required label={t('componentWeight')} value={component.weight} onChange={weight => update(index, { components: asset.components.map((member, i) => i === componentIndex ? { ...member, weight } : member) })} /><Button onClick={() => update(index, { components: asset.components.filter((_, i) => i !== componentIndex) })}>{t('remove')}</Button></div> })}
        <div className="flex flex-wrap items-center gap-2"><Button onClick={() => setSelecting(selecting === asset.id ? null : asset.id)}>{t('addProxy')}</Button><span className="text-xs text-slate-600">{t('weightTotal', { value: pct(asset.components.reduce((sum, item) => sum + item.weight, 0)) })}</span></div>
        {selecting === asset.id && <SourcePicker onSelect={(component, name) => { if (!asset.components.some(member => member.series_id === component.series_id && member.field === component.field)) onChange({ ...value, assets: value.assets.map((item, i) => i === index ? { ...item, components: [...item.components, component] } : item) }, { ...sourceLabels, [component.series_id]: name }); setSelecting(null) }} />}
      </>}
    </section>)}</div>
    {!fixedAssets && <div className="flex flex-wrap items-center gap-2 border-t border-slate-200 pt-3">
      <Button onClick={() => onChange({ ...value, assets: [...value.assets, newAsset(`asset-${crypto.randomUUID().slice(0, 8)}`)] })}>{t('addAsset')}</Button>
      {!value.assets.length && <Button onClick={() => onChange({ ...value, assets: (['cash', 'rates', 'credit', 'equity', 'gold'] as const).map((key, index) => newAsset(key, t(`template.${key}`), index === 0 ? 'cash' : 'market')) })}>{t('useTemplate')}</Button>}
    </div>}
  </div>
}

import { Button } from '../ui'
import { Field, NumberInput } from '../risk-models/ResearchUI'
import { percentInputValue } from '../../services/strategicAllocation'
import type { BlackLittermanRequest } from '../../services/cmaModelTypes'
import { control, useLtcmaText } from '../ltcma/shared'

type View = BlackLittermanRequest['views'][number]
export default function BlackLittermanViews({ value, onChange }: { value: BlackLittermanRequest; onChange: (value: BlackLittermanRequest) => void }) {
  const { t } = useLtcmaText()
  const replace = (index: number, next: View) => onChange({ ...value, views: value.views.map((v, i) => i === index ? next : v) })
  const kind = (index: number, next: View['kind']) => {
    const v = value.views[index]
    const shared = { annual_return: v.annual_return, view_std: v.view_std, observed_on: v.observed_on, available_on: v.available_on, source: v.source }
    replace(index, next === 'basket'
      ? { ...shared, kind: next, basis: 'relative', legs: value.asset_ids.map(asset_id => ({ asset_id, coefficient: 0 })) }
      : { ...shared, kind: next, asset_id: v.kind === 'basket' ? '' : v.asset_id, relative_to: null })
  }
  return <div className="space-y-4">
    <div className="divide-y divide-slate-200">{value.views.map((view, index) => {
      const basis = view.kind === 'basket' ? view.basis : view.kind
      const label = (key: string) => t(key, { index: index + 1 })
      return <fieldset key={index} className="min-w-0 space-y-3 py-4">
        <legend className="text-sm font-semibold">{label('viewTitle')}</legend>
        <div className="grid min-w-0 gap-3 sm:grid-cols-2 xl:grid-cols-3">
          <Field label={label('viewType')}><select className={control} value={view.kind} onChange={e => kind(index, e.target.value as View['kind'])}>
            <option value="absolute">{t('viewAbsolute')}</option><option value="relative">{t('viewRelative')}</option><option value="basket">{t('viewBasket')}</option>
          </select></Field>
          {view.kind !== 'basket' && <Field label={label('viewAsset')}><select className={control} value={view.asset_id} onChange={e => replace(index, { ...view, asset_id: e.target.value })}>
            <option value="">{t('choose')}</option>{value.asset_ids.map(a => <option key={a}>{a}</option>)}
          </select></Field>}
          {view.kind === 'relative' && <Field label={label('viewComparison')}><select className={control} value={view.relative_to ?? ''} onChange={e => replace(index, { ...view, relative_to: e.target.value || null })}>
            <option value="">{t('choose')}</option>{value.asset_ids.map(a => <option key={a} disabled={a === view.asset_id}>{a}</option>)}
          </select></Field>}
          {view.kind === 'basket' && <Field label={label('viewBasis')}><select className={control} value={view.basis} onChange={e => replace(index, { ...view, basis: e.target.value as 'absolute' | 'relative' })}>
            <option value="absolute">{t('viewAbsolute')}</option><option value="relative">{t('viewRelative')}</option>
          </select></Field>}
          <Field label={label(basis === 'absolute' ? 'viewTotalReturn' : 'viewReturnDifference')}><NumberInput className={`${control} tabular-nums`} value={percentInputValue(view.annual_return)} onValueChange={n => replace(index, { ...view, annual_return: n / 100 })} /></Field>
          <Field label={label('viewStd')} hint={t('viewStdHint')}><NumberInput aria-label={label('viewStd')} className={`${control} tabular-nums`} value={percentInputValue(view.view_std)} onValueChange={n => replace(index, { ...view, view_std: n / 100 })} /></Field>
          <Field label={label('viewObserved')}><input type="date" className={control} value={view.observed_on} onChange={e => replace(index, { ...view, observed_on: e.target.value })} /></Field>
          <Field label={label('viewAvailable')}><input type="date" className={control} max={value.as_of} value={view.available_on} onChange={e => replace(index, { ...view, available_on: e.target.value })} /></Field>
          <Field label={label('viewSource')}><input className={control} maxLength={2000} value={view.source} onChange={e => replace(index, { ...view, source: e.target.value })} /></Field>
        </div>
        {view.kind === 'basket' && <div className="space-y-3">
          <p className="text-xs leading-5 text-slate-600">{t('basketHint')}</p>
          <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">{view.legs.map((leg, position) => <Field key={leg.asset_id} label={t('basketCoefficient', { index: index + 1, asset: leg.asset_id })}>
            <NumberInput className={`${control} tabular-nums`} min={-4} max={4} value={leg.coefficient} onValueChange={coefficient => replace(index, { ...view, legs: view.legs.map((v, i) => i === position ? { ...v, coefficient } : v) })} />
          </Field>)}</div>
        </div>}
        <Button onClick={() => onChange({ ...value, views: value.views.filter((_, i) => i !== index) })}>{label('viewRemove')}</Button>
      </fieldset>
    })}</div>
    {!value.views.length && <p className="text-sm text-slate-600">{t('viewEmpty')}</p>}
    <Button disabled={value.views.length >= 60} onClick={() => onChange({ ...value, views: [...value.views, { kind: 'absolute', asset_id: '', relative_to: null, annual_return: NaN, view_std: NaN, observed_on: '', available_on: '', source: '' }] })}>{t('viewAdd')}</Button>
    {value.views.length >= 60 && <p className="text-xs text-slate-600">{t('viewMaximum')}</p>}
  </div>
}

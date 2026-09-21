import { Link } from 'react-router-dom'
import { Field, NumberInput } from '../risk-models/ResearchUI'
import CmaModelEditor from '../strategic-allocation/CmaModelEditor'
import { isStatisticalCma } from '../../services/cmaModelTypes'
import type { CmaDraft } from '../../services/strategicAllocation'
import type { CmaMethodId, LtcmaCapabilities, LtcmaOptions } from '../../services/ltcma'
import LtcmaAssetFields, { LtcmaRiskReference } from './LtcmaAssetFields'
import LtcmaStatisticsFields from './LtcmaStatisticsFields'
import { applyScope, changeMethod, methodOf, scopeKey, updateContext } from './model'
import { control, linkClass, useLtcmaText } from './shared'

type Props = {
  value: CmaDraft; options: LtcmaOptions; capabilities: LtcmaCapabilities; cutoff: string
  onChange: (value: CmaDraft) => void; sourceLabels: Record<string, string>; onLabels: (value: Record<string, string>) => void
}
export default function LtcmaInputFields({ value, options, capabilities, cutoff, onChange, sourceLabels, onLabels }: Props) {
  const { t } = useLtcmaText(), method = methodOf(value)
  const patch = (change: Partial<CmaDraft>) => onChange(updateContext(value, change))
  const available = capabilities.methods.find(item => item.id === method)
  const statistical = isStatisticalCma(value.model)
  const allocation = options.allocations.find(item => item.alloc_name === value.alloc_name)
  return <div className="min-w-0 space-y-6">
    <section className="space-y-4" aria-label={t('basis')}><h2 className="text-lg font-semibold">{t('basis')}</h2>
      <div className="grid gap-3 sm:grid-cols-2"><Field label={t('name')}><input className={control} maxLength={120} value={value.name} onChange={event => patch({ name: event.target.value })} /></Field>
        <Field label={t('asOf')}><input type="date" className={control} max={cutoff} value={value.as_of} onChange={event => patch({ as_of: event.target.value })} /></Field>
        <Field label={t('scope')}><select aria-label={t('scope')} className={control} value={scopeKey(value)} onChange={event => { onLabels({}); onChange(applyScope(value, event.target.value, options)) }}>
          <option value="">{t('chooseScope')}</option>
          <optgroup label={t('productScopes')}>{options.allocations.map(item => <option key={item.alloc_name} value={`allocation:${item.alloc_name}`}>{item.alloc_name}</option>)}</optgroup>
          <optgroup label={t('strategicScopes')}>{options.strategic_universes.map(item => <option key={item.id} value={`universe:${item.id}`}>{item.name} · {item.definition.as_of}</option>)}</optgroup>
          {scopeKey(value) && !allocation && !options.strategic_universes.some(item => item.id === value.strategic_universe_id) && <option value={scopeKey(value)}>{value.alloc_name ?? value.strategic_universe_id} · {t('unavailable')}</option>}
        </select></Field>
        <Field label={t('horizon')} hint={t('horizonHint')}><NumberInput className={control} min={1} max={30} value={value.horizon_years} onValueChange={horizon_years => patch({ horizon_years })} /></Field>
        <Field label={t('currency')}><input className={control} readOnly={Boolean(value.strategic_universe_id)} maxLength={3} value={value.currency} onChange={event => patch({ currency: event.target.value.toUpperCase() })} /></Field>
        <Field label={t('method')}><select aria-label={t('method')} className={control} value={method} onChange={event => onChange(changeMethod(value, event.target.value as CmaMethodId))}>
          {capabilities.methods.map(item => <option key={item.id} value={item.id} disabled={!item.available}>{t(item.id)}{item.available ? '' : ` · ${t('unavailable')}`}</option>)}
        </select></Field>
      </div>
      {!options.allocations.length && !options.strategic_universes.length && <p className="text-sm text-amber-800">{t('scopeMissing')}</p>}
      <Link className={linkClass} to="/pre-investment/product-pool?scope=strategic">{t('openScope')}</Link>
      {available?.available === false && <p className="text-sm text-amber-800">{available.reason ?? t('unavailable')}</p>}
      <Field label={t('source')} hint={t('sourceHint')}><textarea className={control} rows={3} maxLength={2000} value={value.source} onChange={event => patch({ source: event.target.value })} /></Field>
    </section>
    {statistical && <LtcmaStatisticsFields value={value} options={options} onChange={onChange} sourceLabels={sourceLabels} onLabels={onLabels} />}
    {value.model && !statistical && <CmaModelEditor hideMethodChoice context={{ asset_ids: value.assets.map(a => a.id), as_of: value.as_of, currency: value.currency }} value={value.model}
      onChange={model => patch({ model })} />}
    <div className="border-t border-slate-200 pt-5">
      {!value.model && value.alloc_name && <LtcmaRiskReference key={value.alloc_name} value={value} onChange={onChange} coverage={allocation?.coverage ?? undefined} />}
      <LtcmaAssetFields value={value} onChange={onChange} />
    </div>
    <details className="border-t border-slate-200 pt-3"><summary className="min-h-10 cursor-pointer text-sm font-medium">{t('advanced')}</summary>
      <div className="mt-3 grid gap-3 sm:grid-cols-2">
        <Field label={t('momentBasis')}><select aria-label={t('momentBasis')} className={control} disabled={Boolean(value.model)} value={value.moment_semantics ?? 'annualized_periodic_arithmetic'} onChange={event => patch({ moment_semantics: event.target.value as CmaDraft['moment_semantics'] })}>
          <option value="annualized_periodic_arithmetic">{t('annualized_periodic_arithmetic')}</option><option value="one_year_simple">{t('one_year_simple')}</option>
        </select></Field>
        <Field label={t('feeBasis')}><select aria-label={t('feeBasis')} className={control} value={value.fee_basis ?? 'explicit_assumption'} onChange={event => patch({ fee_basis: event.target.value as CmaDraft['fee_basis'] })}><option value="source_embedded_no_additional_fee">{t('embeddedFees')}</option><option value="explicit_assumption">{t('explicitBasis')}</option></select></Field>
        <Field label={t('fxBasis')}><select aria-label={t('fxBasis')} className={control} value={value.fx_hedging_basis ?? 'explicit_assumption'} onChange={event => patch({ fx_hedging_basis: event.target.value as CmaDraft['fx_hedging_basis'] })}><option value="same_currency_no_conversion">{t('noFx')}</option><option value="explicit_assumption">{t('explicitBasis')}</option></select></Field>
      </div>
    </details>
    <label className="flex min-h-10 items-start gap-2 text-sm leading-6"><input className="mt-1.5" type="checkbox" checked={value.basis_confirmed} onChange={event => patch({ basis_confirmed: event.target.checked })} />{t('basisConfirm')}</label>
  </div>
}

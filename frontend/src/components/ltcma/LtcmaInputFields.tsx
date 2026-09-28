import { useId } from 'react'
import { Field } from '../risk-models/ResearchUI'
import CmaModelEditor from '../strategic-allocation/CmaModelEditor'
import { isScenarioCma, isStatisticalCma } from '../../services/cmaModelTypes'
import type { CmaDraft } from '../../services/strategicAllocation'
import type { LtcmaCapabilities, LtcmaOptions } from '../../services/ltcma'
import LtcmaAssetFields, { LtcmaRiskReference } from './LtcmaAssetFields'
import LtcmaStatisticsFields, { LtcmaHistoryWindow } from './LtcmaStatisticsFields'
import LtcmaScenarioFields from './LtcmaScenarioFields'
import LtcmaMethodSelect from './LtcmaMethodSelect'
import { applyScope, changeMethod, methodOf, scopeKey, updateContext } from './model'
import { control, RateInput, useLtcmaText } from './shared'

type Props = {
  value: CmaDraft; options: LtcmaOptions; capabilities: LtcmaCapabilities; cutoff: string; platformDay: string | null | undefined; defaultName: string
  onChange: (value: CmaDraft) => void; sourceLabels: Record<string, string>; onLabels: (value: Record<string, string>) => void
  mandateHorizonDays?: number
  nameError?: string
  methodOptionsReady?: boolean
  methodOptionsError?: string
}
export default function LtcmaInputFields({ value, options, capabilities, cutoff, platformDay, defaultName, onChange, sourceLabels, onLabels, mandateHorizonDays, nameError = '', methodOptionsReady = true, methodOptionsError }: Props) {
  const { t } = useLtcmaText(), method = methodOf(value)
  const methodHelpId = useId(), nameErrorId = useId()
  const patch = (change: Partial<CmaDraft>) => onChange(updateContext(value, change))
  const available = capabilities.methods.find(item => item.id === method)
  const statistical = isStatisticalCma(value.model)
  const scenario = isScenarioCma(value.model)
  const allocation = options.allocations.find(item => item.alloc_name === value.alloc_name)
  const universe = options.strategic_universes.find(item => item.id === value.strategic_universe_id)
  const proxyLabels = { ...Object.assign({}, ...universe?.definition.assets.map(asset => asset.research_proxy?.source_labels ?? {}) ?? []), ...sourceLabels }
  const assetLabels = Object.fromEntries(universe?.definition.assets.map(asset => [asset.id, asset.name]) ?? [])
  const classificationComplete = value.assets.every(asset => asset.role && asset.liquidity)
  const deferClassification = scenario && !value.strategic_universe_id && classificationComplete
  const assetFields = value.assets.length > 0 && (!statistical || !value.strategic_universe_id) && <div className="border-t border-slate-200 pt-4">
    {!value.model && value.alloc_name && <LtcmaRiskReference key={value.alloc_name} value={value} onChange={onChange} coverage={allocation?.coverage ?? undefined} />}
    <LtcmaAssetFields value={value} onChange={onChange} assetLabels={assetLabels} />
  </div>
  return <div className="min-w-0 space-y-6">
    <section className="space-y-4" aria-label={t('basis')}><h2 className="text-lg font-semibold">{t('basis')}</h2>
      {/* 名称是清单里区分同范围、同方法版本的唯一线索，必须在输入首屏可见、可改。 */}
      <Field label={t('name')} hint={t('autoNameHint')}>
        <input aria-label={t('name')} className={control} maxLength={120} placeholder={defaultName} value={value.name}
          aria-invalid={nameError ? true : undefined} aria-describedby={nameError ? nameErrorId : undefined}
          onChange={event => patch({ name: event.target.value })} />
        {nameError && <span id={nameErrorId} role="alert" className="mt-1 block text-xs font-normal text-rose-800">{nameError}</span>}
      </Field>
      <div className="grid gap-3 md:grid-cols-3">
        <Field label={t('scope')}><select aria-label={t('scope')} className={control} value={scopeKey(value)} onChange={event => { onLabels({}); onChange(applyScope(value, event.target.value, options)) }}>
          <option value="">{t('chooseScope')}</option>
          <optgroup label={t('productScopes')}>{options.allocations.map(item => <option key={item.alloc_name} value={`allocation:${item.alloc_name}`}>{item.alloc_name}</option>)}</optgroup>
          <optgroup label={t('strategicScopes')}>{options.strategic_universes.map(item => <option key={item.id} value={`universe:${item.id}`}>{item.name} · {item.definition.as_of}</option>)}</optgroup>
          {scopeKey(value) && !allocation && !options.strategic_universes.some(item => item.id === value.strategic_universe_id) && <option value={scopeKey(value)}>{value.alloc_name ?? value.strategic_universe_id} · {t('unavailable')}</option>}
        </select></Field>
        <LtcmaMethodSelect value={method} methods={capabilities.methods} describedBy={methodHelpId} onChange={method => {
          const next = changeMethod(value, method, universe)
          if (next.model?.method === 'conditional_scenario' && mandateHorizonDays && mandateHorizonDays <= 2520) next.model.horizon_days = mandateHorizonDays
          onChange(next)
        }} />
        <Field label={t('asOf')} hint={t(platformDay === undefined ? 'clockUnknown' : platformDay ? 'pitDateHint' : 'manualDateHint')}>
          <input aria-label={t('asOf')} type="date" className={control} max={cutoff} disabled={platformDay !== null} value={value.as_of} onChange={event => patch({ as_of: event.target.value, basis_confirmed: false })} />
        </Field>
      </div>
      <p id={methodHelpId} aria-live="polite" aria-atomic="true" className="border-l-2 border-accent-200 pl-4 text-sm leading-6 text-slate-600">{t(`methodSummary.${method}`)}</p>
      <p className="text-sm text-slate-600">{t('currency')}：<strong className="font-medium text-slate-900">{value.currency}</strong> · {t('currencyInherited')}</p>
      {!options.allocations.length && !options.strategic_universes.length && <p className="text-sm text-amber-800">{t('scopeMissing')}</p>}
      {available?.available === false && <p className="text-sm text-amber-800">{available.reason ?? t('unavailable')}</p>}
    </section>
    {value.assets.length > 0 && scenario && (methodOptionsError ? null : <LtcmaScenarioFields value={value} options={options} onChange={onChange} sourceLabels={proxyLabels} onLabels={onLabels} mandateHorizonDays={mandateHorizonDays} optionsReady={methodOptionsReady} />)}
    {value.assets.length > 0 && statistical && !scenario && <LtcmaStatisticsFields value={value} options={options} onChange={onChange} sourceLabels={proxyLabels} onLabels={onLabels} optionsReady={methodOptionsReady} optionsError={methodOptionsError} />}
    {value.assets.length > 0 && value.model && !statistical && <CmaModelEditor hideMethodChoice assetLabels={assetLabels} context={{ asset_ids: value.assets.map(a => a.id), as_of: value.as_of, currency: value.currency }} value={value.model}
      onChange={model => patch({ model })} />}
    {!value.assets.length ? <p role="status" className="border-t border-slate-200 pt-4 text-sm text-slate-600">{t('selectScopeFirst')}</p> : !deferClassification && assetFields}
    <details className="border-t border-slate-200 pt-3"><summary className="min-h-10 cursor-pointer text-sm font-medium">{t('optionalNotes')}</summary>
      <div className="mt-2 space-y-3">
        <Field label={t('source')} hint={t('sourceHint')}><textarea className={control} rows={2} maxLength={2000} value={value.source} onChange={event => patch({ source: event.target.value })} /></Field>
        <div className="grid gap-3 sm:grid-cols-2">{value.assets.map((asset, index) => <Field key={asset.id} label={`${assetLabels[asset.id] ?? asset.id} · ${t('rationale')}`}>
          <input className={control} maxLength={1000} value={asset.rationale} onChange={event => patch({ assets: value.assets.map((row, i) => i === index ? { ...row, rationale: event.target.value } : row) })} />
        </Field>)}</div>
      </div>
    </details>
    <details className="border-t border-slate-200 pt-3"><summary className="min-h-10 cursor-pointer text-sm font-medium">{t('advanced')}</summary>
      {scenario && <div className="my-3"><LtcmaHistoryWindow value={value} onChange={onChange} /></div>}
      {deferClassification && assetFields}
      {(value.model?.method === 'historical_statistics' || value.model?.method === 'historical_regime_occupancy') && <div className="my-3"><Field label={t('shrinkage')}><RateInput value={value.model.shrinkage ?? 0} onChange={shrinkage => { if (value.model?.method === 'historical_statistics' || value.model?.method === 'historical_regime_occupancy') patch({ model: { ...value.model, shrinkage } }) }} /></Field></div>}
      <div className="mt-3 grid gap-3 sm:grid-cols-2">
        {!value.strategic_universe_id && <Field label={t('currency')}><input className={control} maxLength={3} value={value.currency} onChange={event => patch({ currency: event.target.value.toUpperCase(), basis_confirmed: false })} /></Field>}
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

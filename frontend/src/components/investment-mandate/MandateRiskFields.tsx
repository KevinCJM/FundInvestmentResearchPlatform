import { useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { Badge, Button, EmptyState } from '../ui'
import { Field, inputClass, NumberInput, percentText } from '../risk-models/ResearchUI'
import { riskScales, type StudyOption, type VersionView } from '../../services/riskScales'
import { percentInputValue, type MandateDefinition, type RiskAuthorization } from '../../services/strategicAllocation'
import { levelReaches } from './model'
import { useMandateText } from './text'

type Props = { value: MandateDefinition; onChange: (patch: Partial<MandateDefinition>) => void; readonly?: boolean }

/**
 * The scale is picked before the objective: it fixes the currency, the asset axis a
 * benchmark can be weighted on, and the C1–C5 caps. It owns the two fetches and hands
 * the frozen version up, so the sections below read one loaded version.
 */
export function ScaleFields({ value, onChange, version, onVersion, readonly = false }: Props & {
  version: VersionView | null; onVersion: (version: VersionView | null) => void
}) {
  const { t } = useMandateText()
  const [items, setItems] = useState<StudyOption[]>([])
  const [loading, setLoading] = useState(false), [error, setError] = useState(''), [reload, setReload] = useState(0)
  const request = useRef<AbortController | null>(null)
  const risk = value.risk_authorization!

  useEffect(() => {
    const controller = new AbortController(); request.current?.abort(); request.current = controller
    setLoading(true); setError(''); setItems([])
    riskScales.studyOptions(value.as_of, controller.signal)
      .then(result => { if (!controller.signal.aborted) setItems(result.items) })
      .catch(reason => { if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message || t('scaleLoadFailed') : t('scaleLoadFailed')) })
      .finally(() => { if (!controller.signal.aborted) setLoading(false) })
    return () => controller.abort()
  }, [value.as_of, reload])

  useEffect(() => {
    const controller = new AbortController(); onVersion(null)
    if (risk.risk_scale_ref) riskScales.version(risk.risk_scale_ref.id, controller.signal)
      .then(result => { if (!controller.signal.aborted) onVersion(result) })
      .catch(reason => { if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message || t('scaleLoadFailed') : t('scaleLoadFailed')) })
    return () => controller.abort()
  }, [risk.risk_scale_ref?.id, risk.risk_scale_ref?.content_hash, reload])

  const selected = items.find(item => item.id === risk.risk_scale_ref?.id)
  const selectScale = (id: string) => {
    const item = items.find(option => option.id === id)
    const nextRisk: RiskAuthorization = { mode: 'manual_level', source: 'risk_scale_selection',
      risk_scale_ref: item ? { id: item.id, content_hash: item.content_hash } : null,
      authorized_max_level: null, selected_max_level: null }
    onChange({ currency: item?.base_currency ?? value.currency, risk_authorization: nextRisk,
      max_volatility: null, benchmark: null })
  }

  return <div className="space-y-5">
    <div className="grid gap-4 sm:grid-cols-2">
      <Field label={t('riskScale')} hint={t('studyScaleHint')}>
        <select className={inputClass} disabled={loading || readonly} value={risk.risk_scale_ref?.id ?? ''} onChange={event => selectScale(event.target.value)}>
          <option value="">{t('chooseStudyScale')}</option>
          {items.map(item => <option key={item.id} value={item.id}>{item.name} · v{item.version_number}</option>)}
          {risk.risk_scale_ref && !items.some(item => item.id === risk.risk_scale_ref!.id) && <option value={risk.risk_scale_ref.id}>{version?.name ?? risk.risk_scale_ref.id}</option>}
        </select>
      </Field>
      <Field label={t('derivedCurrency')} hint={t('derivedCurrencyHint')}><input className={inputClass} value={selected?.base_currency ?? version?.preview.request_echo.definition.base_currency ?? value.currency} readOnly /></Field>
    </div>

    {loading && <div role="status" className="space-y-2"><p className="text-sm text-slate-600">{t('loadingStudyScales')}</p><div className="h-12 animate-pulse rounded-lg bg-slate-200 motion-reduce:animate-none" /></div>}
    {error && <div role="alert" className="flex flex-wrap items-center gap-3 text-sm text-rose-700"><span>{error}</span><Button onClick={() => setReload(token => token + 1)}>{t('retry')}</Button></div>}
    {!loading && !error && !items.length && <EmptyState mascot={false} title={t('noStudyScales')} hint={t('noStudyScalesHint')}
      action={<Link className="inline-flex min-h-10 items-center text-sm font-semibold text-accent-700 underline" to="/settings/risk-scales">{t('openRiskCenter')}</Link>} />}
  </div>
}

export default function MandateRiskFields({ value, onChange, version, readonly = false }: Props & { version: VersionView | null }) {
  const { t } = useMandateText()
  const risk = value.risk_authorization!
  const selectedLevel = risk.selected_max_level
  const levels = version?.preview.result.levels ?? []
  const selectedCap = levels.find(level => level.level_code === `C${selectedLevel}`)?.authorized_volatility_cap
  const selectLevel = (level: number | null) => onChange({ benchmark: null, risk_authorization: {
    ...risk, mode: 'manual_level', source: 'risk_scale_selection', authorized_max_level: level, selected_max_level: level,
  } })

  // Reference-frozen level returns, marked against the stated target. A comparison of
  // published numbers; the target itself is stated below, where the full notice lives.
  const target = value.objective_kind === 'absolute_return' && Number.isFinite(value.target_return) ? value.target_return : null
  const shortfall = (portrait: typeof levels[number]) => levelReaches(portrait.expected_return.value, target) === false ? t('levelBelowTarget') : ''

  return <div className="space-y-5">
    <fieldset className="min-w-0">
      <legend className="text-sm font-medium text-slate-800">{t('maxRiskLevel')}</legend>
      <p className="mt-1 text-xs leading-5 text-slate-600">{t('maxRiskLevelHint')}</p>
      {!version ? <p className="mt-3 text-sm text-slate-600">{t('chooseScaleFirst')}</p>
        : <ul className="mt-3 grid gap-2 sm:grid-cols-2 lg:grid-cols-5">{levels.map((portrait, index) => {
          const level = index + 1, active = selectedLevel === level
          const unavailable = value.objective_kind === 'benchmark_relative' && portrait.representative_node_id == null
          const short = shortfall(portrait)
          return <li key={portrait.level_code}><button type="button" disabled={readonly || unavailable} aria-pressed={active}
            onClick={() => selectLevel(active ? null : level)}
            className={`flex min-h-11 w-full flex-col gap-1 rounded-xl border p-3 text-left disabled:cursor-not-allowed disabled:opacity-50 ${active ? 'border-accent-600 bg-accent-50 ring-2 ring-accent-500' : 'border-slate-300 hover:bg-slate-50'}`}>
            <span className="text-base font-semibold text-slate-900">{portrait.level_code}</span>
            <span className="text-xs text-slate-600">{t('cap')} <b className="font-semibold tabular-nums text-slate-900">{percentText(portrait.authorized_volatility_cap)}</b></span>
            <span className="text-xs text-slate-600">{t('referenceReturn')} <b className="font-semibold tabular-nums text-slate-900">{percentText(portrait.expected_return.value)}</b></span>
            <span className="text-xs text-slate-600">{t('historicalMdd')} <b className="font-semibold tabular-nums text-slate-900">{percentText(portrait.historical_mdd.value)}</b></span>
            {unavailable && <span className="text-xs text-amber-800">{t('noRepresentative')}</span>}
            {!unavailable && short && <span className="text-xs text-amber-800">{short}</span>}
          </button></li>
        })}</ul>}
      {selectedCap !== undefined && <p className="mt-3 text-sm leading-6 text-slate-700">{t('riskLevelResolved', { level: selectedLevel ?? '', cap: percentText(selectedCap) })}</p>}
      <p className="mt-2 text-xs leading-5 text-slate-600">{t('levelPortraitBasis')}</p>
    </fieldset>

    <Field label={t('minCashWeight')} hint={t('minCashWeightHint')}><NumberInput className={inputClass} value={percentInputValue(value.min_cash_weight)} min={0} max={100} onValueChange={number => onChange({ min_cash_weight: number / 100 })} /></Field>

    {version && <details className="border-t border-slate-200 pt-3"><summary className="min-h-10 cursor-pointer text-sm font-medium">{t('riskPortrait')} · {version.name} <Badge>{t('frozen')}</Badge></summary>
      <p className="mt-2 text-xs leading-5 text-slate-600">{t('fixedBandsHint')}</p>
      <div className="mt-3 overflow-x-auto"><table className="w-full min-w-[420px] text-sm" aria-label={t('riskPortrait')}><caption className="sr-only">{t('riskPortrait')}</caption><thead><tr>{['level', 'cap', 'referenceReturn', 'historicalMdd'].map(key => <th key={key} scope="col" className="p-2 text-right first:text-left">{t(key)}</th>)}</tr></thead><tbody>{levels.map(level => <tr key={level.level_code} className="border-b border-slate-200"><th scope="row" className="p-2 text-left">{level.level_code}</th><td className="p-2 text-right tabular-nums">{percentText(level.authorized_volatility_cap)}</td><td className="p-2 text-right tabular-nums">{percentText(level.expected_return.value)}</td><td className="p-2 text-right tabular-nums">{percentText(level.historical_mdd.value)}</td></tr>)}</tbody></table></div>
    </details>}
  </div>
}

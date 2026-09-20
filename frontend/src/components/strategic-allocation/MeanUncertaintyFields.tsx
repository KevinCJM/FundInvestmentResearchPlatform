import type { MeanUncertaintyEvidence, PolicyRequest } from '../../services/strategicAllocation'
import { Field, NumberInput } from '../risk-models/ResearchUI'
import { control, useLtcmaText } from '../ltcma/shared'

export default function MeanUncertaintyFields({ value, available, onChange }: {
  value: PolicyRequest; available: boolean; onChange: (value: PolicyRequest) => void
}) {
  const { t } = useLtcmaText()
  const single = !value.mode || value.mode === 'single'
  const ellipse = value.uncertainty_set === 'ellipsoidal'
  return <div className="space-y-3">
    <div className="grid gap-3 sm:grid-cols-2">
      <Field label={t('uncertaintySet')}><select className={control} value={value.uncertainty_set ?? 'box'} onChange={event => onChange({ ...value,
        uncertainty_set: event.target.value as 'box' | 'ellipsoidal', uncertainty_confidence: null,
        uncertainty_approximation_acknowledged: false, uncertainty_penalty: 1 })}>
        <option value="box">{t('boxSet')}</option><option value="ellipsoidal" disabled={!single || !available}>{t('ellipsoidalSet')}</option>
      </select></Field>
      {ellipse ? <Field label={t('uncertaintyConfidence')}><select className={control} value={value.uncertainty_confidence ?? ''} onChange={event => onChange({ ...value, uncertainty_confidence: (event.target.value || null) as PolicyRequest['uncertainty_confidence'] })}>
        <option value="">{t('chooseConfidence')}</option>{(['68','90','95'] as const).map(value => <option key={value} value={value}>{value}%</option>)}
      </select></Field> : <Field label={t('uncertaintyPenalty')}><NumberInput className={`${control} tabular-nums`} value={value.uncertainty_penalty} min={0} max={5} onValueChange={uncertainty_penalty => onChange({ ...value, uncertainty_penalty })} /></Field>}
    </div>
    <p className="text-xs leading-5 text-slate-600">{t(ellipse ? 'ellipseSetHint' : 'boxSetHint')}</p>
    {(!single || !available) && <p className="text-xs leading-5 text-slate-600">{t(!single ? 'ellipseMultiUnavailable' : 'ellipseSourceUnavailable')}</p>}
    {ellipse && <label className="flex min-h-10 items-start gap-2 text-sm leading-6"><input className="mt-1.5" type="checkbox" checked={value.uncertainty_approximation_acknowledged ?? false}
      onChange={event => onChange({ ...value, uncertainty_approximation_acknowledged: event.target.checked })} />{t('ellipseApproximation')}</label>}
  </div>
}

export function MeanUncertaintySummary({ value }: { value: MeanUncertaintyEvidence }) {
  const { t } = useLtcmaText()
  return <section className="space-y-2" aria-label={t('frozenUncertainty')}>
    <h3 className="text-sm font-semibold">{t('frozenUncertainty')}</h3>
    <p className="text-sm tabular-nums text-slate-700">{t('ellipseSummary', { confidence: value.confidence, radius: value.kappa.toFixed(4) })}</p>
    <p className="text-xs leading-5 text-slate-600">{t(value.calibration)}</p>
    {value.warnings.map((warning, index) => <p key={index} className="text-xs leading-5 text-slate-600">{warning}</p>)}
  </section>
}

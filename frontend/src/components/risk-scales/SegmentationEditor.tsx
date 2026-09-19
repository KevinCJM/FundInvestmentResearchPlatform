import { Button } from '../ui'
import { Field } from '../risk-models/ResearchUI'
import { registeredAlgorithms, type Capabilities, type RiskScaleDefinition } from '../../services/riskScales'
import { controlClass, PercentField, useRiskText } from './shared'

export function SegmentationEditor({ value, capabilities, appliedCaps, busy = false, onChange }: { value: RiskScaleDefinition; capabilities: Capabilities; appliedCaps?: number[]; busy?: boolean; onChange: (next: RiskScaleDefinition) => void }) {
  const { t } = useRiskText()
  const segmentation = value.segmentation!
  const automaticCaps = appliedCaps?.length === 5 ? appliedCaps : undefined
  const adjusted = segmentation.adjusted_caps ?? automaticCaps
  const changeMethod = (algorithm_id: typeof segmentation.algorithm_id) => {
    if (algorithm_id === 'manual_volatility_bands_v1') {
      onChange({ ...value, segmentation: { algorithm_id, manual_caps: automaticCaps ? [...automaticCaps] : [NaN, NaN, NaN, NaN, NaN], rationale: automaticCaps ? t('manualSeedRationale') : '' } })
      return
    }
    onChange({ ...value, segmentation: { algorithm_id } })
  }
  const changeAdjustedCap = (index: number, number: number) => {
    const source = segmentation.adjusted_caps ?? automaticCaps ?? [NaN, NaN, NaN, NaN, NaN]
    onChange({ ...value, segmentation: { ...segmentation, adjusted_caps: source.map((item, i) => i === index ? number : item) } })
  }
  return <div className="space-y-3"><Field label={t('segmentationMethod')} required><select required disabled={busy} className={controlClass} value={segmentation.algorithm_id} onChange={event => changeMethod(event.target.value as typeof segmentation.algorithm_id)}>{registeredAlgorithms(capabilities).map(method => <option value={method.id} key={method.id} disabled={method.available === false}>{t(`algorithm.${method.id}`)}{method.available === false ? ` · ${t('unavailable')}` : ''}</option>)}</select></Field><p className="text-sm text-slate-600">{t(`methodHint.${segmentation.algorithm_id}`)}</p>{registeredAlgorithms(capabilities).filter(method => method.available === false).map(method => <p key={method.id} className="text-xs text-amber-900">{t(`algorithm.${method.id}`)}: {method.reason || t('unsupportedMethod')}</p>)}
    {segmentation.algorithm_id === 'manual_volatility_bands_v1' && <><div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-5">{[0, 1, 2, 3, 4].map(index => <PercentField required key={index} label={t('manualCap', { level: `C${index + 1}` })} value={segmentation.manual_caps?.[index] ?? NaN} onChange={number => onChange({ ...value, segmentation: { ...segmentation, manual_caps: [0, 1, 2, 3, 4].map(i => i === index ? number : segmentation.manual_caps?.[i] ?? NaN) } })} />)}</div><Field label={t('manualRationale')} required><textarea required className={controlClass} value={segmentation.rationale ?? ''} onChange={event => onChange({ ...value, segmentation: { ...segmentation, rationale: event.target.value } })} /></Field><p className="text-xs text-slate-600">{t('manualCapsHint')}</p></>}
    {segmentation.algorithm_id !== 'manual_volatility_bands_v1' && adjusted && <section className="space-y-2 border-t border-slate-200 pt-3"><div className="flex flex-wrap items-center justify-between gap-2"><div><h3 className="text-sm font-semibold">{t('boundaryAdjustment')}</h3><p className="text-xs text-slate-600">{t('boundaryAdjustmentHint')}</p></div>{segmentation.adjusted_caps && <Button disabled={busy} onClick={() => onChange({ ...value, segmentation: { algorithm_id: segmentation.algorithm_id } })}>{t('resetAlgorithmBoundaries')}</Button>}</div><div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-5">{adjusted.map((cap, index) => <PercentField required key={index} label={t('manualCap', { level: `C${index + 1}` })} value={cap} onChange={number => changeAdjustedCap(index, number)} />)}</div></section>}
  </div>
}

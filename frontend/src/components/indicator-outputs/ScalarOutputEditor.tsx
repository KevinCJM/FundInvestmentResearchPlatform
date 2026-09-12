import type { IndicatorDraft, ValidationResponse } from '../../services/customIndicators'
import { useI18n } from '../../i18n/runtime'

const field = 'mt-1 min-h-11 w-full rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-accent-500'

/** A scalar indicator has exactly one result; computation states are not outputs. */
export default function ScalarOutputEditor({ draft, validation, disabled, onPatch }: {
  draft: IndicatorDraft
  validation: ValidationResponse | null
  disabled?: boolean
  onPatch: (patch: Partial<IndicatorDraft>) => void
}) {
  const { s } = useI18n()
  const measure = validation?.output_measure || draft.output_measure
  const date = measure === 'date' || draft.display_format === 'date'
  return <section className="min-w-0 rounded-xl border border-slate-200 bg-white p-5 shadow-sm" aria-label={s('primitiveAuthoring.result', {}, '结果设置')} data-testid="indicator-output-settings">
    <h2 className="font-semibold text-slate-900">{s('primitiveAuthoring.result', {}, '结果设置')}</h2>
    <p className="mt-1 text-xs leading-5 text-slate-600">{s('primitiveAuthoring.singleHint', {}, '一个指标对应一个结果。分别选择多个指标时，系统自动复用相同的计算步骤。')}</p>
    <fieldset disabled={disabled} className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
      <label className="text-xs font-semibold text-slate-700">{s('scalarResult.format', {}, '显示方式')}<select className={field} value={date ? 'date' : draft.display_format} disabled={date || disabled} onChange={event => onPatch({ display_format: event.target.value as IndicatorDraft['display_format'] })}>{date ? <option value="date">{s('primitiveAuthoring.date', {}, '日期（由公式推断）')}</option> : <><option value="number">{s('graph.formatNumber', {}, '数字')}</option><option value="percent">{s('graph.formatPercent', {}, '百分比')}</option></>}</select></label>
      {!date && <label className="text-xs font-semibold text-slate-700">{s('scalarResult.precision', {}, '小数位')}<input aria-label="小数位" type="number" className={field} min={0} max={8} value={draft.precision} onChange={event => onPatch({ precision: Math.max(0, Math.min(8, Number(event.target.value))) })} /></label>}
      {!date && <label className="text-xs font-semibold text-slate-700">{s('graph.unit', {}, '单位')}<input className={field} maxLength={20} value={draft.unit} onChange={event => onPatch({ unit: event.target.value })} /></label>}
      <label className="text-xs font-semibold text-slate-700">{s('scalarResult.direction', {}, '评分方向')}<select className={field} value={date ? 'neutral' : draft.direction} disabled={date || disabled} onChange={event => onPatch({ direction: event.target.value as IndicatorDraft['direction'] })}><option value="neutral">{s('scalarResult.neutral', {}, '仅展示，不判断优劣')}</option>{!date && <><option value="higher_better">{s('scalarResult.higher', {}, '越高越好')}</option><option value="lower_better">{s('scalarResult.lower', {}, '越低越好')}</option></>}</select></label>
    </fieldset>
    {disabled && <p className="mt-2 text-xs text-amber-800">{s('outputAuthoring.locked', {}, '请等待当前操作完成，并应用画布改动后再编辑结果。')}</p>}
  </section>
}

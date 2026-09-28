import { cmaPairReason, cmaChoice } from '../../services/cmaCompatibility'
import { Link } from 'react-router-dom'
import { useI18n } from '../../i18n/runtime'
import type { CmaReference, CmaVersion, StrategicCatalog } from '../../services/strategicAllocation'
import { Button } from '../ui'
import { Field, NumberInput, percentText } from '../risk-models/ResearchUI'
import { cmaMethodText, control, linkClass, useLtcmaText } from '../ltcma/shared'
import { isStatisticalCma } from '../../services/cmaModelTypes'
import { cmaSelectionReason, type CmaSelectionContext } from './LtcmaSelection'
import { ltcmaSaaIssue } from '../../services/ltcma'

export function multiCmaWeightIssue(refs: CmaReference[], common = false) {
  return !refs.length || refs.length > 20 || new Set(refs.map(ref => ref.cma_id)).size !== refs.length ||
    (common ? refs.some(ref => ref.weight != null) : refs.some(ref => typeof ref.weight !== 'number' || !Number.isFinite(ref.weight) || ref.weight < 0) || Math.abs(refs.reduce((sum, ref) => sum + (ref.weight ?? 0), 0) - 1) > 1e-8)
}

export default function MultiCmaSelection({ items, refs, versions, context, busy, common = false, onAdd, onChange, onContinue, onRetry }: {
  common?: boolean
  items: StrategicCatalog['assumptions']; refs: CmaReference[]; versions: CmaVersion[]; context: CmaSelectionContext; busy: boolean
  onAdd: (id: string) => void; onChange: (refs: CmaReference[]) => void; onContinue: () => void; onRetry: () => void
}) {
  const { s } = useI18n(), { t } = useLtcmaText()
  const complete = refs.length > 0 && refs.every(ref => versions.some(version => version.id === ref.cma_id && version.content_hash === ref.content_hash))
  const invalid = multiCmaWeightIssue(refs, common)
  const pairIssue = versions.slice(1).map(version => cmaPairReason(cmaChoice(versions[0]), cmaChoice(version))).find(Boolean)
  const researchIssue = versions.map(ltcmaSaaIssue).find(Boolean) || (pairIssue ? t(pairIssue) : null)
  const remainingItems = items.filter(item => !refs.some(ref => ref.cma_id === item.id))
  return <section className="min-w-0 space-y-4" aria-label={s(common ? 'multiCma.common' : 'multiCma.average')}>
    <p className="text-sm leading-6 text-slate-600">{s(common ? 'multiCma.commonHint' : 'multiCma.hint')}</p>
    {!refs.length ? <p className="text-sm text-slate-600">{s('multiCma.empty')}</p> : <div>
      <div aria-hidden="true" className="hidden grid-cols-[minmax(0,1fr)_8rem_5rem] items-center gap-4 border-b border-slate-200 py-2 text-xs font-medium text-slate-600 sm:grid">
        <span>{t('name')}</span><span className="text-right">{s(common ? 'multiCma.requiredColumn' : 'multiCma.weight')}</span><span className="text-center">{t('actions')}</span>
      </div>
      <div role="list" className="divide-y divide-slate-200">{refs.map(ref => {
      const version = versions.find(item => item.id === ref.cma_id), model = version?.definition.model
      const name = version?.name ?? items.find(item => item.id === ref.cma_id)?.name ?? ref.cma_id
      return <div role="listitem" className="grid min-w-0 grid-cols-[minmax(0,1fr)_auto] items-end gap-3 py-3 sm:grid-cols-[minmax(0,1fr)_8rem_5rem] sm:items-center sm:gap-4" key={ref.cma_id}>
        <div className="col-span-2 min-w-0 sm:col-span-1"><Link className={`${linkClass} !px-0 break-words`} to={`/pre-investment/ltcma/${encodeURIComponent(ref.cma_id)}`}>{name}</Link>{version && <p className="text-sm text-slate-600">{cmaMethodText(t, model?.method ?? 'manual', isStatisticalCma(model) ? model.window?.kind : undefined)}</p>}</div>
        {common ? <p className="text-sm text-slate-600 sm:text-right">{s('multiCma.required')}</p> : <label className="block min-w-0">
          <span className="mb-1 block text-xs font-medium text-slate-600 sm:sr-only">{s('multiCma.weight')}</span>
          <span className="relative block">
            <NumberInput aria-label={`${name} · ${s('multiCma.weight')}`} className={`${control} !mt-0 !pr-8 text-right tabular-nums`} value={typeof ref.weight === 'number' && Number.isFinite(ref.weight) ? Number((ref.weight * 100).toPrecision(12)) : NaN} disabled={!complete} min={0} max={100} onValueChange={weight => onChange(refs.map(item => item.cma_id === ref.cma_id ? { ...item, weight: weight / 100 } : item))} />
            <span aria-hidden="true" className="pointer-events-none absolute inset-y-0 right-3 flex items-center text-sm text-slate-600">%</span>
          </span>
        </label>}
        <Button className="min-h-11" aria-label={s('multiCma.remove', { name })} onClick={() => onChange(refs.filter(item => item.cma_id !== ref.cma_id))}>{s('multiCma.removeShort')}</Button>
      </div>
    })}</div></div>}
    {remainingItems.length > 0 && <>
      <Field label={s('multiCma.add')}><select className={control} value="" disabled={busy || refs.length >= 20 || !context.mandate || !context.allocationName && !context.strategicUniverseId} onChange={event => { if (event.target.value) onAdd(event.target.value) }}>
        <option value="">{s('multiCma.choose')}</option>{remainingItems.map(item => {
          const reason = cmaSelectionReason(item, context) || (versions.length ? cmaPairReason(items.find(x => x.id === versions[0].id) ?? cmaChoice(versions[0]), item) : null)
          return <option key={item.id} value={item.id} disabled={Boolean(reason)}>{item.name} · {item.as_of}{reason ? ` · ${t(reason)}` : ''}</option>
        })}
      </select></Field>
      <p className="text-xs leading-5 text-slate-600">{s('multiCma.limit')}</p>
    </>}
    {refs.length > 0 && <>
      {!common && <p className="border-t border-slate-200 pt-3 text-right text-sm font-medium tabular-nums" aria-live="polite">{s('multiCma.total', { total: percentText(refs.reduce((sum, ref) => sum + (ref.weight ?? 0), 0)) })}</p>}
      {invalid && <p role="status" className="text-sm text-amber-800">{s('multiCma.weightInvalid')}</p>}
      {researchIssue && <p role="status" className="text-sm text-amber-800">{researchIssue}</p>}
      {!complete && <p role="status" className="text-sm text-slate-600">{s('multiCma.loading')}</p>}
      <div className="flex flex-wrap gap-3">{!common && <Button disabled={!complete} onClick={() => onChange(refs.map(ref => ({ ...ref, weight: 1 / refs.length })))}>{s('multiCma.equal')}</Button>}
        {!complete && <Button disabled={busy} onClick={onRetry}>{s('multiCma.retry')}</Button>}
        <Button tone="primary" className="max-w-full whitespace-normal" disabled={busy || !complete || invalid || Boolean(researchIssue)} onClick={onContinue}>{s(common ? 'multiCma.commonContinue' : 'multiCma.continue')}</Button></div>
    </>}
    <Link className={linkClass} to="/pre-investment/ltcma">{s('multiCma.newResearch')}</Link>
  </section>
}

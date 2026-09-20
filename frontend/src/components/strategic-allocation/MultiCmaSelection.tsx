import { Link } from 'react-router-dom'
import { useI18n } from '../../i18n/runtime'
import type { CmaReference, CmaVersion, StrategicCatalog } from '../../services/strategicAllocation'
import { Button } from '../ui'
import { Field, NumberInput, percentText } from '../risk-models/ResearchUI'
import { control, linkClass, useLtcmaText } from '../ltcma/shared'
import { cmaSelectionReason, type CmaSelectionContext } from './LtcmaSelection'

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
  return <section className="min-w-0 space-y-4" aria-label={s(common ? 'multiCma.common' : 'multiCma.average')}>
    <p className="text-sm leading-6 text-slate-600">{s(common ? 'multiCma.commonHint' : 'multiCma.hint')}</p>
    <Field label={s('multiCma.add')}><select className={control} value="" disabled={busy || refs.length >= 20 || !context.mandate || !context.allocationName && !context.strategicUniverseId} onChange={event => { if (event.target.value) onAdd(event.target.value) }}>
      <option value="">{s('multiCma.choose')}</option>{items.filter(item => !refs.some(ref => ref.cma_id === item.id)).map(item => {
        const reason = cmaSelectionReason(item, context)
        return <option key={item.id} value={item.id} disabled={Boolean(reason)}>{item.name} · {item.as_of}{reason ? ` · ${t(reason)}` : ''}</option>
      })}
    </select></Field>
    <p className="text-xs leading-5 text-slate-600">{s('multiCma.limit')}</p>
    {!refs.length ? <p className="text-sm text-slate-600">{s('multiCma.empty')}</p> : <div className="divide-y divide-slate-200">{refs.map(ref => {
      const name = versions.find(item => item.id === ref.cma_id)?.name ?? items.find(item => item.id === ref.cma_id)?.name ?? ref.cma_id
      return <div className="grid min-w-0 gap-3 py-3 sm:grid-cols-[minmax(0,1fr)_10rem_auto] sm:items-end" key={ref.cma_id}>
        <Link className={`${linkClass} break-words`} to={`/pre-investment/ltcma/${encodeURIComponent(ref.cma_id)}`}>{name}</Link>
        {common ? <p className="text-sm text-slate-600">{s('multiCma.required')}</p> : <Field label={`${name} · ${s('multiCma.weight')}`}><NumberInput className={`${control} text-right tabular-nums`} value={typeof ref.weight === 'number' && Number.isFinite(ref.weight) ? Number((ref.weight * 100).toPrecision(12)) : NaN} disabled={!complete} min={0} max={100} onValueChange={weight => onChange(refs.map(item => item.cma_id === ref.cma_id ? { ...item, weight: weight / 100 } : item))} /></Field>}
        <Button aria-label={s('multiCma.remove', { name })} onClick={() => onChange(refs.filter(item => item.cma_id !== ref.cma_id))}>{s('multiCma.removeShort')}</Button>
      </div>
    })}</div>}
    {refs.length > 0 && <>
      {!common && <p className="text-sm tabular-nums" aria-live="polite">{s('multiCma.total', { total: percentText(refs.reduce((sum, ref) => sum + (ref.weight ?? 0), 0)) })}</p>}
      {invalid && <p role="status" className="text-sm text-amber-800">{s('multiCma.weightInvalid')}</p>}
      {!complete && <p role="status" className="text-sm text-slate-600">{s('multiCma.loading')}</p>}
      <div className="flex flex-wrap gap-3">{!common && <Button disabled={!complete} onClick={() => onChange(refs.map(ref => ({ ...ref, weight: 1 / refs.length })))}>{s('multiCma.equal')}</Button>}
        {!complete && <Button disabled={busy} onClick={onRetry}>{s('multiCma.retry')}</Button>}
        <Button tone="primary" className="max-w-full whitespace-normal" disabled={busy || !complete || invalid} onClick={onContinue}>{s(common ? 'multiCma.commonContinue' : 'multiCma.continue')}</Button></div>
    </>}
    <Link className={linkClass} to="/pre-investment/ltcma">{s('multiCma.newResearch')}</Link>
  </section>
}

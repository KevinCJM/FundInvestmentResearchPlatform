import { useEffect, useState } from 'react'
import { Button, DataTable } from '../ui'
import { Field } from '../risk-models/ResearchUI'
import { metadata, textValue, riskScales, type ProxyComponent, type SourceCatalog } from '../../services/riskScales'
import { controlClass, ErrorNotice, Problems, useRiskTask, useRiskText } from './shared'

export interface PendingSource {
  component: ProxyComponent
  name: string
}

const sourceKey = (component: ProxyComponent) => `${component.kind}:${component.series_id}:${component.field}`

/**
 * Inline batch picker: the tray survives type/query/page changes and one
 * confirm applies every selected source atomically through the caller.
 */
export function SourcePicker({ existing = [], onConfirm, onCancel }: {
  existing?: ProxyComponent[]
  onConfirm: (items: PendingSource[]) => void
  onCancel: () => void
}) {
  const { t } = useRiskText()
  const [kind, setKind] = useState('index'), [query, setQuery] = useState(''), [offset, setOffset] = useState(0)
  const [catalog, setCatalog] = useState<SourceCatalog | null>(null)
  const [pending, setPending] = useState<PendingSource[]>([])
  const task = useRiskTask()
  const reload = () => { void task.run(signal => riskScales.sources(kind, query, offset, signal), setCatalog) }
  useEffect(() => { setCatalog(null); const timer = window.setTimeout(reload, 180); return () => { window.clearTimeout(timer); task.invalidate() } }, [kind, query, offset])
  const existingKeys = new Set(existing.map(sourceKey))
  const pendingKeys = new Set(pending.map(item => sourceKey(item.component)))
  const toggle = (item: PendingSource) => setPending(current => current.some(entry => sourceKey(entry.component) === sourceKey(item.component))
    ? current.filter(entry => sourceKey(entry.component) !== sourceKey(item.component))
    : [...current, item])
  return <section className="space-y-3 border-t border-slate-200 pt-3" aria-label={t('sourcePicker')}>
    <div className="grid gap-3 sm:grid-cols-2"><Field label={t('sourceType')} required><select required className={controlClass} value={kind} onChange={event => { setKind(event.target.value); setOffset(0) }}>{['index', 'etf', 'fund'].map(type => <option value={type} key={type}>{t(`source.${type}`)}</option>)}</select></Field><Field label={t('sourceSearch')} optional><input className={controlClass} value={query} onChange={event => { setQuery(event.target.value); setOffset(0) }} /></Field></div>
    <p className="text-xs leading-5 text-slate-600">{t('sourceLimits')}</p><ErrorNotice error={task.error} retry={reload} />
    <DataTable<Record<string, unknown>>
      caption={t('sourcePicker')}
      rows={catalog?.items ?? []}
      rowKey={item => textValue(item.id)}
      minWidth="640px"
      maxHeight="24rem"
      loading={task.busy ? t('loading') : undefined}
      empty={t('noSourcesHint')}
      columns={[
        { header: t('sourceName'), cell: item => {
          const code = textValue(item.code)
          return <span className="block min-w-0"><span className="block break-words font-medium text-slate-900">{textValue(item.name)}</span><span className="block break-all text-xs text-slate-600">{code} · {t(`source.${kind}`)}</span></span>
        } },
        { header: t('dataSemantics'), cell: item => {
          const coverage = metadata(item.coverage)
          const range = [textValue(coverage.start_date ?? coverage.first_date), textValue(coverage.end_date ?? coverage.latest_date)].filter(Boolean).join(' / ')
          return <span className="block break-words text-xs">{t(kind === 'index' ? 'selectedIndexValue' : 'adjustedProductData')}<span className="block break-words">{range || t('coveragePending')}</span></span>
        } },
        { header: t('availability'), cell: item => {
          const capability = metadata(item.reference_capability)
          const supported = Array.isArray(capability.supported_fields) ? capability.supported_fields : []
          const field = kind === 'index' ? (supported.includes('close') ? 'close' : undefined) : ['adj_nav', 'close_hfq'].find(value => supported.includes(value))
          const available = capability.available === true && Boolean(field)
          const key = `${kind}:${textValue(item.id)}:${field}`
          return <span className="block break-words text-xs">{available ? t('available') : !field ? t('adjustedDataRequired') : textValue(capability.reason) || t('unsupportedSource')}{existingKeys.has(key) && <span className="block text-slate-600">{t('alreadyAdded')}</span>}</span>
        } },
        { header: t('select'), cell: item => {
          const capability = metadata(item.reference_capability)
          const supported = Array.isArray(capability.supported_fields) ? capability.supported_fields : []
          const field = kind === 'index' ? (supported.includes('close') ? 'close' : undefined) : ['adj_nav', 'close_hfq'].find(value => supported.includes(value))
          const available = capability.available === true && Boolean(field)
          const component = { kind: kind as ProxyComponent['kind'], series_id: textValue(item.id), field: field as ProxyComponent['field'], weight: 1 }
          const key = sourceKey(component)
          const already = existingKeys.has(key)
          const chosen = pendingKeys.has(key)
          return <label className="flex min-h-10 items-center gap-2"><input type="checkbox" aria-label={`${textValue(item.name)} ${textValue(item.code)}`} disabled={!available || already} checked={already || chosen} onChange={() => toggle({ component, name: textValue(item.name) })} /><span className="text-xs text-slate-600">{already ? t('alreadyAdded') : t('select')}</span></label>
        } },
      ]}
    />
    {catalog && !task.busy && <div className="flex flex-wrap items-center gap-2"><Button disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - 20))}>{t('previous')}</Button><span className="text-xs text-slate-600">{t('sourceCount', { count: catalog.total })}</span><Button disabled={offset + 20 >= catalog.total} onClick={() => setOffset(offset + 20)}>{t('next')}</Button></div>}
    <Problems items={catalog?.problems ?? []} />
    <div className="divide-y divide-slate-200 rounded-xl border border-slate-200">
      <div className="p-3"><p aria-live="polite" className="text-sm font-medium text-slate-800">{t('selectedSources', { count: pending.length })}</p>
        {pending.length
          ? <ul className="mt-1">{pending.map(item => <li key={sourceKey(item.component)} className="flex flex-wrap items-center justify-between gap-2 py-1 text-xs"><span className="min-w-0 break-words">{item.name}<span className="ml-1 break-all text-slate-600">{item.component.series_id}</span></span><button type="button" aria-label={`${t('removeSelected')} ${item.name}`} className="inline-flex min-h-10 items-center text-accent-800 underline" onClick={() => toggle(item)}>{t('removeSelected')}</button></li>)}</ul>
          : <p className="mt-1 text-xs text-slate-600">{t('selectedSourcesEmpty')}</p>}
      </div>
      <div className="flex flex-wrap gap-2 p-3"><Button tone="primary" disabled={!pending.length} onClick={() => onConfirm(pending)}>{t('confirmSelection')}</Button><Button onClick={onCancel}>{t('cancel')}</Button></div>
    </div>
  </section>
}

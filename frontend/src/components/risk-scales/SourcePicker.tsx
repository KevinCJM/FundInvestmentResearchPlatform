import { useEffect, useState } from 'react'
import { Button, EmptyState } from '../ui'
import { Field } from '../risk-models/ResearchUI'
import { metadata, textValue, riskScales, type ProxyComponent, type SourceCatalog } from '../../services/riskScales'
import { controlClass, ErrorNotice, Loading, Problems, useRiskTask, useRiskText } from './shared'

export function SourcePicker({ onSelect }: { onSelect: (item: ProxyComponent, name: string) => void }) {
  const { t } = useRiskText()
  const [kind, setKind] = useState('index'), [query, setQuery] = useState(''), [offset, setOffset] = useState(0)
  const [catalog, setCatalog] = useState<SourceCatalog | null>(null)
  const task = useRiskTask()
  const reload = () => { void task.run(signal => riskScales.sources(kind, query, offset, signal), setCatalog) }
  useEffect(() => { setCatalog(null); const timer = window.setTimeout(reload, 180); return () => { window.clearTimeout(timer); task.invalidate() } }, [kind, query, offset])
  return <section className="space-y-3 border-t border-slate-200 pt-3" aria-label={t('sourcePicker')}>
    <div className="grid gap-3 sm:grid-cols-2"><Field label={t('sourceType')} required><select required className={controlClass} value={kind} onChange={event => { setKind(event.target.value); setOffset(0) }}>{['index', 'etf', 'fund'].map(type => <option value={type} key={type}>{t(`source.${type}`)}</option>)}</select></Field><Field label={t('sourceSearch')}><input className={controlClass} value={query} onChange={event => { setQuery(event.target.value); setOffset(0) }} /></Field></div>
    <p className="text-xs leading-5 text-slate-600">{t('sourceLimits')}</p><ErrorNotice error={task.error} retry={reload} />
    {task.busy && <Loading />}
    {catalog && !task.busy && <><Problems items={catalog.problems ?? []} />{!catalog.items.length ? <EmptyState mascot={false} title={t('noSources')} hint={t('noSourcesHint')} /> : <div className="overflow-x-auto"><table className="w-full text-sm" aria-label={t('sourcePicker')}><thead><tr>{['sourceName', 'dataSemantics', 'availability', 'actions'].map(key => <th scope="col" className="p-2 text-left" key={key}>{t(key)}</th>)}</tr></thead><tbody>{catalog.items.map(item => {
      const capability = metadata(item.reference_capability), coverage = metadata(item.coverage)
      const supported = Array.isArray(capability.supported_fields) ? capability.supported_fields : []
      const field = kind === 'index' ? (supported.includes('close') ? 'close' : undefined) : ['adj_nav', 'close_hfq'].find(value => supported.includes(value))
      const available = capability.available === true && Boolean(field)
      const range = [textValue(coverage.start_date ?? coverage.first_date), textValue(coverage.end_date ?? coverage.latest_date)].filter(Boolean).join(' / ')
      return <tr key={textValue(item.id)} className="border-b border-slate-200"><th scope="row" className="p-2 text-left font-medium"><span className="block">{textValue(item.name)}</span><span className="text-xs text-slate-600">{textValue(item.code)} · {t(`source.${kind}`)}</span></th><td className="p-2 text-xs">{t(kind === 'index' ? 'selectedIndexValue' : 'adjustedProductData')}<span className="block break-words">{range || t('coveragePending')}</span></td><td className="p-2 text-xs">{available ? t('available') : !field ? t('adjustedDataRequired') : textValue(capability.reason) || t('unsupportedSource')}</td><td className="p-2"><Button disabled={!available} onClick={() => onSelect({ kind: kind as ProxyComponent['kind'], series_id: textValue(item.id), field: field as ProxyComponent['field'], weight: 1 }, textValue(item.name))}>{t('select')}</Button></td></tr>
    })}</tbody></table></div>}<div className="flex flex-wrap items-center gap-2"><Button disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - 20))}>{t('previous')}</Button><span className="text-xs text-slate-600">{t('sourceCount', { count: catalog.total })}</span><Button disabled={offset + 20 >= catalog.total} onClick={() => setOffset(offset + 20)}>{t('next')}</Button></div></>}
  </section>
}

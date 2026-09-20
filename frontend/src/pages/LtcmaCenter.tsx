import { useEffect, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { Badge, Button, Card, EmptyState } from '../components/ui'
import { Feedback, Field } from '../components/risk-models/ResearchUI'
import { control, linkClass, useLtcmaTask, useLtcmaText } from '../components/ltcma/shared'
import { ltcma, type CmaDraftView, type CmaListResponse, type LtcmaCapabilities } from '../services/ltcma'

export default function LtcmaCenter() {
  const { t } = useLtcmaText(), task = useLtcmaTask(), navigate = useNavigate()
  const [query, setQuery] = useState(''), [method, setMethod] = useState(''), [retired, setRetired] = useState(false)
  const [offset, setOffset] = useState(0), [tab, setTab] = useState<'versions' | 'drafts'>('versions'), [revision, setRevision] = useState(0)
  const [versions, setVersions] = useState<CmaListResponse | null>(null)
  const [drafts, setDrafts] = useState<CmaDraftView[]>([]), [capabilities, setCapabilities] = useState<LtcmaCapabilities | null>(null)
  const [deleting, setDeleting] = useState<CmaDraftView | null>(null)
  const pageSize = 20
  useEffect(() => {
    task.invalidate(); setVersions(null)
    const timer = setTimeout(() => { void task.run(async signal => {
      const [list, savedDrafts, supported] = await Promise.all([
        ltcma.list({ q: query, method, include_retired: retired, offset, limit: pageSize }, signal),
        ltcma.drafts(signal), ltcma.capabilities(signal),
      ])
      return { list, savedDrafts, supported }
    }, ({ list, savedDrafts, supported }) => { setVersions(list); setDrafts(savedDrafts.items); setCapabilities(supported) }) }, query ? 250 : 0)
    return () => { clearTimeout(timer); task.invalidate() }
  }, [query, method, retired, offset, revision])
  const remove = () => {
    if (!deleting) return
    void task.run(signal => ltcma.deleteDraft(deleting, signal), () => { setDeleting(null); setRevision(value => value + 1) })
  }
  return <div className="min-w-0 space-y-4 text-slate-900">
    <header className="flex flex-col justify-between gap-3 sm:flex-row sm:items-start">
      <div><h1 className="text-2xl font-bold">{t('title')}</h1><p className="mt-2 text-sm leading-6 text-slate-600">{t('description')}</p></div>
      <Button tone="primary" onClick={() => navigate('/pre-investment/ltcma/new')}>{t('new')}</Button>
    </header>
    <div className="flex gap-2" aria-label={t('status')}>
      <Button aria-pressed={tab === 'versions'} onClick={() => setTab('versions')}>{t('saved')}</Button>
      <Button aria-pressed={tab === 'drafts'} onClick={() => setTab('drafts')}>{t('drafts')}</Button>
    </div>
    <Feedback error={task.error} />{task.error && <Button onClick={() => setRevision(value => value + 1)}>{t('retry')}</Button>}
    {tab === 'versions' && <div className="grid items-end gap-3 sm:grid-cols-2 xl:grid-cols-3">
      <Field label={t('search')}><input className={control} value={query} onChange={event => { setOffset(0); setQuery(event.target.value) }} /></Field>
      <Field label={t('method')}><select className={control} value={method} onChange={event => { setOffset(0); setMethod(event.target.value) }}>
        <option value="">{t('allMethods')}</option>{capabilities?.methods.map(item => <option key={item.id} value={item.id}>{t(item.id)}</option>)}
      </select></Field>
      <label className="flex min-h-10 items-center gap-2 text-sm"><input type="checkbox" checked={retired} onChange={event => { setOffset(0); setRetired(event.target.checked) }} />{t('includeRetired')}</label>
    </div>}
    {(task.busy || !versions && !task.error) && <div role="status" className="space-y-2"><p className="text-sm text-slate-600">{t('loading')}</p><div className="h-12 animate-pulse rounded-lg bg-slate-200 motion-reduce:animate-none" /><div className="h-24 animate-pulse rounded-lg bg-slate-100 motion-reduce:animate-none" /></div>}
    {!task.busy && versions && (tab === 'versions' ? versions.items.length ? <Card>
      <div className="overflow-x-auto"><table className="w-full min-w-[720px] text-sm" aria-label={t('saved')}>
        <thead><tr>{['name', 'method', 'scope', 'asOf', 'horizon', 'status', 'actions'].map(key => <th key={key} scope="col" className="px-3 py-3 text-left text-xs text-slate-600">{t(key)}</th>)}</tr></thead>
        <tbody>{versions.items.map(item => <tr key={item.id} className="border-b border-slate-200">
          <th scope="row" className="px-3 py-3 text-left font-medium"><Link className={linkClass} to={`/pre-investment/ltcma/${encodeURIComponent(item.id)}`}>{item.name}</Link></th>
          <td className="px-3 py-3">{t(item.method)}</td><td className="px-3 py-3">{item.scope_name ?? item.asset_ids.join(', ')}</td>
          <td className="whitespace-nowrap px-3 py-3">{item.as_of}</td><td className="px-3 py-3 text-right tabular-nums">{t('horizonValue', { years: item.horizon_years })}</td>
          <td className="px-3 py-3"><Badge tone={item.retired ? 'warning' : 'neutral'}>{t(item.retired ? 'retired' : 'confirmed')}</Badge></td>
          <td className="px-3 py-3"><Link className={linkClass} to={`/pre-investment/ltcma/new?copy=${encodeURIComponent(item.id)}`}>{t('copy')}</Link></td>
        </tr>)}</tbody>
      </table></div>
      <div className="mt-3 flex flex-wrap items-center justify-between gap-2"><p className="text-sm tabular-nums text-slate-600">{t('page', { page: offset / pageSize + 1, total: versions.total })}</p><div className="flex gap-2">
        <Button disabled={offset === 0} onClick={() => setOffset(value => Math.max(0, value - pageSize))}>{t('previous')}</Button>
        <Button disabled={offset + pageSize >= versions.total} onClick={() => setOffset(value => value + pageSize)}>{t('next')}</Button>
      </div></div>
    </Card> : <EmptyState mascot={false} title={t('empty')} hint={t('emptyHint')} /> : drafts.length ? <Card>
      <div className="divide-y divide-slate-200">{drafts.map(item => <div key={item.id} className="flex flex-wrap items-center justify-between gap-2 py-3">
        <div><h2 className="text-sm font-medium">{item.name}</h2><p className="mt-1 text-xs text-slate-600">{item.updated_at}</p></div>
        <div className="flex flex-wrap gap-2"><Link className={linkClass} to={`/pre-investment/ltcma/new?draft=${encodeURIComponent(item.id)}`}>{t('continue')}</Link><Button tone="danger" onClick={() => setDeleting(item)}>{t('deleteDraft')}</Button></div>
      </div>)}</div>
    </Card> : <EmptyState mascot={false} title={t('emptyDrafts')} hint={t('emptyDraftsHint')} />)}
    {deleting && <section className="space-y-3 rounded-xl border border-slate-200 bg-white p-4" aria-label={t('deleteDraft')}>
      <p className="text-sm">{t('deleteConfirm', { name: deleting.name })}</p><div className="flex gap-2"><Button disabled={task.busy} onClick={() => setDeleting(null)}>{t('cancel')}</Button><Button tone="danger" disabled={task.busy} onClick={remove}>{t('confirmDelete')}</Button></div>
    </section>}
  </div>
}

import { Fragment, useEffect, useRef, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { Badge, Button, Card, EmptyState, ErrorPanel, LoadingPanel } from '../components/ui'
import { Feedback, Field } from '../components/risk-models/ResearchUI'
import { cmaMethodText, control, linkClass, useLtcmaTask, useLtcmaText } from '../components/ltcma/shared'
import { ltcma, type CmaDraftView, type CmaListItem, type CmaListResponse, type LtcmaCapabilities } from '../services/ltcma'
import { cmaGroupKey, cmaHandoffIssue, prepareLtcmaHandoff } from '../services/ltcmaHandoff'
import { useResearchDay } from '../app/ResearchContext'
import { updateAllocationJourney } from '../app/allocationJourney'
import { today } from '../components/risk-models/ResearchUI'
import { UpstreamLink, UsabilityNote, VersionTag, upstreamOf } from '../components/versioning'

function MethodSummary({ item }: { item: CmaListItem }) {
  const { t } = useLtcmaText()
  return <span>{cmaMethodText(t, item.method, item.history?.window.kind)}</span>
}

function GroupHeader({ item }: { item: CmaListItem }) {
  const { t } = useLtcmaText()
  const scope = item.upstream.find(ref => ref.kind !== 'mandate')
  const path = item.research_path && t(item.research_path === 'strategy_first' ? 'pathStrategyFirst' : 'pathProductFirst')
  return <tr className="border-b border-slate-200 bg-slate-50"><th scope="rowgroup" colSpan={6} className="px-3 py-2 text-left text-sm font-normal text-slate-700">
    <div className="sticky left-3 flex max-w-[calc(100vw-5rem)] flex-wrap gap-x-6 gap-y-1 sm:max-w-none">
      <span className="inline-flex flex-wrap items-center gap-x-1">{t('groupMandate')}：<UpstreamLink item={upstreamOf(item, 'mandate')} fallback={t('groupUnlinked')} /></span>
      <span className="inline-flex flex-wrap items-center gap-x-1">{t('groupScope')}：{path ? `${path} · ` : ''}<UpstreamLink item={scope} fallback={item.scope_name ?? t('groupUnlinked')} /></span>
    </div>
  </th></tr>
}

export default function LtcmaCenter() {
  const { t } = useLtcmaText(), task = useLtcmaTask(), navigate = useNavigate()
  const [query, setQuery] = useState(''), [method, setMethod] = useState(''), [retired, setRetired] = useState(false)
  const [offset, setOffset] = useState(0), [tab, setTab] = useState<'versions' | 'drafts'>('versions'), [revision, setRevision] = useState(0)
  const [versions, setVersions] = useState<CmaListResponse | null>(null)
  const [drafts, setDrafts] = useState<CmaDraftView[]>([]), [capabilities, setCapabilities] = useState<LtcmaCapabilities | null>(null)
  const [deleting, setDeleting] = useState<CmaDraftView | null>(null)
  const [deletingVersion, setDeletingVersion] = useState<CmaListItem | null>(null), [notice, setNotice] = useState('')
  const retirement = useLtcmaTask(), deleteDialog = useRef<HTMLDialogElement>(null), heading = useRef<HTMLHeadingElement>(null)
  const [selected, setSelected] = useState<CmaListItem[]>([])
  const handoff = useLtcmaTask(), researchDay = useResearchDay()
  const cutoff = researchDay === undefined ? undefined : researchDay && researchDay < today() ? researchDay : today()
  const selectionIssue = cmaHandoffIssue(selected, cutoff)
  useEffect(() => { handoff.invalidate() }, [cutoff])
  useEffect(() => {
    if (!deletingVersion || !deleteDialog.current) return
    const dialog = deleteDialog.current, previous = document.activeElement
    const overflow = document.body.style.overflow
    dialog.showModal(); document.body.style.overflow = 'hidden'
    dialog.querySelector<HTMLButtonElement>('button')?.focus()
    return () => {
      dialog.close(); document.body.style.overflow = overflow
      if (previous instanceof HTMLElement && previous.isConnected) previous.focus()
      else heading.current?.focus()
    }
  }, [deletingVersion])
  const toggle = (item: CmaListItem) => {
    handoff.invalidate()
    setSelected(current => current.some(value => value.id === item.id) ? current.filter(value => value.id !== item.id) : [...current, item])
  }
  const continueToSaa = () => {
    if (selectionIssue) return
    void handoff.run(signal => prepareLtcmaHandoff(selected, cutoff, t, signal), ({ path, journey }) => {
      updateAllocationJourney(journey); navigate(path)
    })
  }
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
  const removeVersion = () => {
    if (!deletingVersion || deletingVersion.retired || retirement.busy) return
    const item = deletingVersion
    void retirement.run(signal => ltcma.retire(item, t('deleteVersionReason'), signal), () => {
      handoff.invalidate()
      setSelected(current => current.filter(value => value.id !== item.id))
      setNotice(t('versionDeleted', { name: item.name })); setDeletingVersion(null)
      if (!retired && versions?.items.length === 1 && offset > 0) setOffset(value => Math.max(0, value - pageSize))
      setVersions(null); setRevision(value => value + 1)
    })
  }
  // 目录没读出来时清单整块是空的，这时把失败说明换成公共错误态；草稿删除这类操作失败时清单还在，仍是纯文字。
  const loadFailed = Boolean(task.error) && !task.busy && !versions
  return <div className="min-w-0 space-y-4 text-slate-900">
    <header className="flex flex-col justify-between gap-3 sm:flex-row sm:items-start">
      <div className="min-w-0 flex-1">
        <h1 ref={heading} tabIndex={-1} className="text-2xl font-bold">{t('title')}</h1>
        <div className="mt-2 max-w-5xl space-y-1 text-sm leading-6 text-slate-600">
          <p>{t('centerDefinition')}</p>
          <p>{t('centerRationale')}</p>
        </div>
      </div>
      <Button tone="primary" className="shrink-0 self-start" onClick={() => navigate('/pre-investment/ltcma/new')}>{t('new')}</Button>
    </header>
    <div className="flex gap-2" aria-label={t('status')}>
      <Button aria-pressed={tab === 'versions'} onClick={() => { handoff.invalidate(); setTab('versions') }}>{t('saved')}</Button>
      <Button aria-pressed={tab === 'drafts'} onClick={() => { handoff.invalidate(); setTab('drafts') }}>{t('drafts')}</Button>
    </div>
    {!loadFailed && <><Feedback error={task.error} notice={notice} />{task.error && <Button onClick={() => setRevision(value => value + 1)}>{t('retry')}</Button>}</>}
    {tab === 'versions' && <div className="grid items-end gap-3 sm:grid-cols-2 xl:grid-cols-3">
      <Field label={t('search')}><input className={control} value={query} onChange={event => { setOffset(0); setQuery(event.target.value) }} /></Field>
      <Field label={t('method')}><select className={control} value={method} onChange={event => { setOffset(0); setMethod(event.target.value) }}>
        <option value="">{t('allMethods')}</option>{capabilities?.methods.map(item => <option key={item.id} value={item.id}>{t(item.id)}</option>)}
      </select></Field>
      <label className="flex min-h-10 items-center gap-2 text-sm"><input type="checkbox" checked={retired} onChange={event => { setOffset(0); setRetired(event.target.checked) }} />{t('includeRetired')}</label>
    </div>}
    {(task.busy || !versions && !task.error) && <LoadingPanel text={t('loading')} />}
    {loadFailed && <ErrorPanel message={task.error} action={<Button onClick={() => setRevision(value => value + 1)}>{t('retry')}</Button>} />}
    {!task.busy && versions && (tab === 'versions' ? versions.items.length ? <Card>
      <div className="overflow-x-auto" role="region" aria-label={t('saved')} tabIndex={0}><table className="w-full min-w-[720px] text-sm" aria-label={t('saved')}>
        <thead><tr><th scope="col" className="px-3 py-3 text-left text-xs text-slate-600">{t('select')}</th>{['name', 'method', 'asOf', 'status', 'actions'].map(key => <th key={key} scope="col" className="px-3 py-3 text-left text-xs text-slate-600">{t(key)}</th>)}</tr></thead>
        <tbody>{versions.items.map((item, index) => {
          const checked = selected.some(value => value.id === item.id)
          const reason = checked ? null : cmaHandoffIssue([...selected, item], cutoff)
          const groupStart = index === 0 || cmaGroupKey(versions.items[index - 1]) !== cmaGroupKey(item)
          return <Fragment key={item.id}>{groupStart && <GroupHeader item={item} />}<tr className={`border-b border-slate-200 ${checked ? 'bg-accent-50' : ''}`}>
          <td className="px-3 py-3"><label className="flex min-h-11 min-w-11 cursor-pointer items-center justify-center"><input type="checkbox" className="h-4 w-4 accent-accent-600 focus-visible:ring-2 focus-visible:ring-accent-500" aria-label={t('selectNamed', { name: item.name })} aria-describedby={`cma-method-${item.id}${reason ? ` cma-reason-${item.id}` : ''}`} checked={checked} disabled={Boolean(reason)} onChange={() => toggle(item)} /></label></td>
          <th scope="row" className="px-3 py-3 text-left font-medium"><Link className={linkClass} to={`/pre-investment/ltcma/${encodeURIComponent(item.id)}`}>{item.name}</Link> <VersionTag version={item.version} /></th>
          <td id={`cma-method-${item.id}`} className="px-3 py-3"><MethodSummary item={item} /></td>
          <td className="whitespace-nowrap px-3 py-3">{item.as_of}</td>
          <td className="px-3 py-3"><Badge tone={item.usable.status === 'ready' ? 'neutral' : 'warning'}>{t(item.retired ? 'retired' : item.usable.status === 'stale' ? 'needsUpdate' : 'confirmed')}</Badge><UsabilityNote usable={item.usable} />{reason && <p id={`cma-reason-${item.id}`} className="mt-1 max-w-64 text-xs leading-5 text-amber-800">{t(reason)}</p>}</td>
          <td className="px-3 py-3"><div className="flex min-w-48 flex-wrap items-center gap-x-2">
            <Link className={linkClass} to={`/pre-investment/ltcma/new?edit=${encodeURIComponent(item.id)}`}>{t('edit')}</Link>
            <Link className={linkClass} to={`/pre-investment/ltcma/new?copy=${encodeURIComponent(item.id)}`}>{t('copy')}</Link>
            <button type="button" className="inline-flex min-h-11 items-center rounded-lg px-2 text-sm font-medium text-rose-700 hover:bg-rose-50 focus-visible:ring-2 focus-visible:ring-accent-500 disabled:cursor-not-allowed disabled:text-slate-500" disabled={item.retired} title={item.retired ? t('retired') : undefined} onClick={() => { handoff.invalidate(); retirement.invalidate(); setNotice(''); setDeletingVersion(item) }}>{t('delete')}</button>
          </div></td>
        </tr></Fragment>})}</tbody>
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
    {tab === 'versions' && <section className="space-y-3 rounded-xl border border-slate-200 bg-white p-4" aria-label={t('saaSelection')}>
      <div className="flex flex-wrap items-center justify-between gap-3"><p className="font-semibold" aria-live="polite">{t('selectedCmas', { count: selected.length })}</p>{selected.length > 0 && <Button onClick={() => { handoff.invalidate(); setSelected([]) }}>{t('clearSelection')}</Button>}</div>
      {selected.length > 0 && <ul className="divide-y divide-slate-200">{selected.map(item => <li key={item.id} className="flex min-w-0 items-center justify-between gap-3 py-2"><div className="min-w-0 text-sm"><p className="break-words font-medium">{item.name}</p><p className="mt-1 text-slate-600"><MethodSummary item={item} /></p></div><Button className="shrink-0" aria-label={t('removeNamed', { name: item.name })} onClick={() => toggle(item)}>{t('removeSelection')}</Button></li>)}</ul>}
      <p className={`text-sm leading-6 ${selectionIssue && selected.length ? 'text-amber-800' : 'text-slate-600'}`} id="cma-selection-hint">{t(selectionIssue ?? (selected.length > 1 ? 'multipleSaaHint' : 'singleSaaHint'))}</p>
      <Feedback error={handoff.error} />
      <Button tone="primary" className="max-w-full whitespace-normal" aria-describedby="cma-selection-hint" disabled={Boolean(selectionIssue) || handoff.busy} onClick={continueToSaa}>{t(handoff.busy ? 'checkingSaa' : 'continueSaa')}</Button>
    </section>}
    {deleting && <section className="space-y-3 rounded-xl border border-slate-200 bg-white p-4" aria-label={t('deleteDraft')}>
      <p className="text-sm">{t('deleteConfirm', { name: deleting.name })}</p><div className="flex gap-2"><Button disabled={task.busy} onClick={() => setDeleting(null)}>{t('cancel')}</Button><Button tone="danger" disabled={task.busy} onClick={remove}>{t('confirmDelete')}</Button></div>
    </section>}
    <dialog ref={deleteDialog} aria-labelledby="cma-delete-title" aria-describedby="cma-delete-impact" aria-busy={retirement.busy} className="m-auto max-h-[calc(100dvh_-_2rem)] w-[calc(100%_-_2rem)] max-w-lg overflow-y-auto rounded-xl border border-slate-200 bg-white p-5 text-slate-900 shadow-xl backdrop:bg-slate-900/40" onCancel={event => { event.preventDefault(); if (!retirement.busy) setDeletingVersion(null) }} onKeyDown={event => {
      if (event.key !== 'Tab') return
      const buttons = event.currentTarget.querySelectorAll<HTMLButtonElement>('button:not([disabled])')
      if (!buttons.length) { event.preventDefault(); return }
      const first = buttons[0], last = buttons[buttons.length - 1]
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus() }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus() }
    }}>
      {deletingVersion && <div className="space-y-4">
        <h2 id="cma-delete-title" className="text-lg font-semibold">{t('deleteVersionTitle')}</h2>
        <p className="break-words text-sm font-medium">{deletingVersion.name}</p>
        <p id="cma-delete-impact" className="text-sm leading-6 text-slate-600">{t('deleteVersionImpact')}</p>
        <Feedback error={retirement.error} />
        {retirement.busy && <p role="status" className="text-sm text-slate-600">{t('deletingVersion')}</p>}
        <div className="flex flex-wrap justify-end gap-2"><Button disabled={retirement.busy} onClick={() => setDeletingVersion(null)}>{t('cancel')}</Button><Button tone="danger" disabled={retirement.busy} onClick={removeVersion}>{t('confirmDelete')}</Button></div>
      </div>}
    </dialog>
  </div>
}

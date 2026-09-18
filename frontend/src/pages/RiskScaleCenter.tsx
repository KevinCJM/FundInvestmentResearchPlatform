import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { Badge, Button, Card, EmptyState } from '../components/ui'
import { ErrorNotice, linkClass, Loading, useRiskTask, useRiskText } from '../components/risk-scales/shared'
import { formatDate } from '../i18n/runtime'
import { riskScales, RiskScaleError, type CatalogResponse, type DefaultsResponse } from '../services/riskScales'

type DeleteTarget = { kind: 'version' | 'draft'; id: string; name: string }

export default function RiskScaleCenter() {
  const { t } = useRiskText()
  const loadTask = useRiskTask(), actionTask = useRiskTask()
  const [catalog, setCatalog] = useState<CatalogResponse | null>(null), [defaults, setDefaults] = useState<DefaultsResponse | null>(null)
  const [pendingDelete, setPendingDelete] = useState<DeleteTarget | null>(null)
  const [clearDefault, setClearDefault] = useState(false), [compared, setCompared] = useState<string[]>([])
  const [includeRetired, setIncludeRetired] = useState(false)
  const [notice, setNotice] = useState('')

  const reload = () => {
    setNotice('')
    setCompared([]); setPendingDelete(null); setClearDefault(false)
    void loadTask.run(async signal => { const [nextCatalog, nextDefaults] = await Promise.all([riskScales.catalog(`limit=100${includeRetired ? '&include_retired=true' : ''}`, signal), riskScales.defaults(signal)]); return { nextCatalog, nextDefaults } }, ({ nextCatalog, nextDefaults }) => { setCatalog(nextCatalog); setDefaults(nextDefaults) })
  }

  useEffect(() => {
    reload()
    return () => { loadTask.invalidate(); actionTask.invalidate() }
  }, [includeRetired])

  const defaultIds = new Set((defaults?.items ?? []).map(item => item.version_id).filter((id): id is string => Boolean(id)))
  const deletingDefault = pendingDelete?.kind === 'version' && defaultIds.has(pendingDelete.id)
  const loadMore = () => {
    if (catalog?.next_offset == null) return
    void loadTask.run(signal => riskScales.catalog(`limit=100&offset=${catalog.next_offset}${includeRetired ? '&include_retired=true' : ''}`, signal), page => {
      setCatalog(current => current ? { ...page,
        items: [...new Map([...current.items, ...page.items].map(item => [item.id, item])).values()] } : page)
    })
  }
  const rows = catalog ? [
    ...catalog.items.map(item => ({ kind: 'version' as const, id: item.id, name: item.name, at: item.created_at, item })),
    ...catalog.drafts.map(item => ({ kind: 'draft' as const, id: item.id, name: item.name, at: item.updated_at, item })),
  ].sort((left, right) => right.at.localeCompare(left.at)) : []

  const remove = () => {
    if (!pendingDelete || !catalog) return
    const target = pendingDelete
    void actionTask.run(async signal => {
      if (target.kind === 'draft') {
        const draft = catalog.drafts.find(item => item.id === target.id)
        if (draft) await riskScales.deleteDraft(draft.id, draft.revision, signal)
        return { target, binding: null }
      }
      const summary = catalog.items.find(item => item.id === target.id)
      if (!summary) return { target, binding: null }
      const [version, defaults] = await Promise.all([riskScales.version(summary.id, signal), riskScales.defaults(signal)])
      const definition = version.preview.request_echo.definition
      const key = `${definition.base_currency}:${definition.risk_basis_id}`
      const binding = defaults.items.find(item => item.key === key)
      if (binding?.version_id === summary.id && !clearDefault) {
        if (!signal.aborted) setDefaults(defaults)
        throw new RiskScaleError('DEFAULT_CLEAR_CONFIRMATION_REQUIRED', t('clearDefault'), null, 409)
      }
      const updated = await riskScales.retire(summary.id, {
        confirm: true,
        expected_revision: binding?.revision ?? 0,
        reason: t('deleteReason'),
        clear_default: binding?.version_id === summary.id && clearDefault,
      }, signal)
      return { target, binding: updated }
    }, ({ target: result, binding }) => {
      setCatalog(current => current ? {
        ...current,
        items: result.kind === 'version' ? includeRetired ? current.items.map(item => item.id === result.id ? { ...item, retired: true } : item) : current.items.filter(item => item.id !== result.id) : current.items,
        drafts: result.kind === 'draft' ? current.drafts.filter(item => item.id !== result.id) : current.drafts,
        total: result.kind === 'version' && !includeRetired ? Math.max(0, current.total - 1) : current.total,
        next_offset: result.kind === 'version' && !includeRetired && current.next_offset != null ? Math.max(0, current.next_offset - 1) : current.next_offset,
      } : current)
      setCompared(current => current.filter(id => id !== result.id))
      setPendingDelete(null)
      setClearDefault(false)
      if (binding) setDefaults(current => ({ items: [...(current?.items ?? []).filter(item => item.key !== binding.key), binding] }))
      setNotice(t('deletedNotice'))
    })
  }

  return <div className="min-w-0 space-y-3 text-slate-900">
    <header className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
      <div><h1 className="text-2xl font-bold">{t('title')}</h1><p className="mt-1 text-sm text-slate-600">{t('descriptionText')}</p></div>
      <Link className={`${linkClass} justify-center bg-accent-600 !px-4 !text-white hover:bg-accent-700`} to="/settings/risk-scales/new">{t('new')}</Link>
    </header>

    <ErrorNotice error={loadTask.error} retry={reload} />
    <ErrorNotice error={actionTask.error} />
    <label className="flex min-h-10 items-center gap-2 text-sm"><input type="checkbox" checked={includeRetired} disabled={actionTask.busy} onChange={event => setIncludeRetired(event.target.checked)} />{t('includeRetired')}</label>
    {notice && <p role="status" className="text-sm text-emerald-800">{notice}</p>}
    {loadTask.busy && <Loading />}
    {catalog && <div className="flex flex-wrap items-center gap-3">
      {compared.length === 2 ? <Link className={linkClass} to={`/settings/risk-scales/compare?left=${encodeURIComponent(compared[0])}&right=${encodeURIComponent(compared[1])}`}>{t('compareSelected')}</Link>
        : <Button disabled>{t('compareSelected')}</Button>}
      <p className="text-sm text-slate-600">{t('exactlyTwo')}</p>
    </div>}

    {catalog && !loadTask.busy && (rows.length ? <Card className="overflow-hidden !p-0">
      <div className="overflow-x-auto">
        <table className="w-full min-w-[680px] text-sm" aria-label={t('configuredTable')}>
          <thead className="bg-slate-50 text-slate-600"><tr>
            <th scope="col" className="px-4 py-3 text-left">{t('name')}</th>
            <th scope="col" className="px-3 py-3 text-left">{t('status')}</th>
            <th scope="col" className="px-3 py-3 text-left">{t('version')}</th>
            <th scope="col" className="px-3 py-3 text-left">{t('updated')}</th>
            <th scope="col" className="px-4 py-3 text-right">{t('actions')}</th>
          </tr></thead>
          <tbody>{rows.map(row => <tr key={`${row.kind}-${row.id}`} className="border-t border-slate-200">
            <th scope="row" className="px-4 py-3 text-left font-medium text-slate-900">
              {row.kind === 'version' && <input type="checkbox" className="mr-2" aria-label={t('selectVersion', { name: row.name, version: row.item.version_number ?? 1 })}
                checked={compared.includes(row.id)} disabled={!compared.includes(row.id) && compared.length >= 2}
                onChange={event => setCompared(current => event.target.checked ? [...current, row.id] : current.filter(id => id !== row.id))} />}
              <Link className="hover:text-accent-700 hover:underline" to={row.kind === 'version' ? `/settings/risk-scales/versions/${encodeURIComponent(row.id)}` : `/settings/risk-scales/drafts/${encodeURIComponent(row.id)}`}>{row.name}</Link>
            </th>
            <td className="px-3 py-3"><div className="flex flex-wrap items-center gap-1"><Badge tone={row.kind === 'version' ? 'success' : 'warning'}>{t(row.kind === 'version' ? row.item.retired ? 'retired' : 'published' : 'draft')}</Badge>{row.kind === 'version' && defaultIds.has(row.id) && <Badge>{t('systemDefault')}</Badge>}{row.kind === 'version' && row.item.review_status === 'upcoming' && <Badge tone="warning">{t('reviewUpcoming')}</Badge>}{row.kind === 'version' && row.item.review_status === 'due' && <Badge tone="danger">{t('reviewDue')}</Badge>}{row.kind === 'version' && row.item.review_due_at && row.item.review_status !== 'none' && <span className="text-xs text-slate-600">{String(row.item.review_due_at)}</span>}</div></td>
            <td className="px-3 py-3 tabular-nums">{row.kind === 'version' ? `v${row.item.version_number ?? 1}` : t('revision', { revision: row.item.revision })}</td>
            <td className="px-3 py-3 whitespace-nowrap text-slate-600">{formatDate(row.at)}</td>
            <td className="px-4 py-2"><div className="flex justify-end gap-1">
              {row.kind === 'version' && <Link className={linkClass} to={`/settings/risk-scales/versions/${encodeURIComponent(row.id)}`}>{t('viewDetails')}</Link>}
              {!(row.kind === 'version' && row.item.retired) && <><Link className={linkClass} to={row.kind === 'version' ? `/settings/risk-scales/new?editFrom=${encodeURIComponent(row.id)}` : `/settings/risk-scales/drafts/${encodeURIComponent(row.id)}`}>{t('editConfig')}</Link>
              <Button tone="danger" disabled={actionTask.busy} onClick={() => { setClearDefault(false); setPendingDelete({ kind: row.kind, id: row.id, name: row.name }) }}>{t('deleteConfig')}</Button></>}
            </div></td>
          </tr>)}</tbody>
        </table>
      </div>
      {catalog.next_offset != null && <div className="border-t border-slate-200 px-4 py-3"><Button disabled={loadTask.busy || actionTask.busy} onClick={loadMore}>{t('loadMore')}</Button></div>}
      {pendingDelete && <div className="flex flex-col gap-2 border-t border-slate-200 bg-slate-50 px-4 py-3 sm:flex-row sm:items-center sm:justify-between">
        <p className="text-sm text-slate-700">{t(pendingDelete.kind === 'version' ? 'deletePublishedConfirm' : 'deleteDraftConfirm', { name: pendingDelete.name })}</p>
        {deletingDefault && <label className="flex min-h-10 items-center gap-2 text-sm"><input type="checkbox" disabled={actionTask.busy} checked={clearDefault} onChange={event => setClearDefault(event.target.checked)} />{t('clearDefault')}</label>}
        <div className="flex gap-2"><Button onClick={() => setPendingDelete(null)}>{t('cancel')}</Button><Button tone="danger" disabled={actionTask.busy || Boolean(deletingDefault && !clearDefault)} onClick={remove}>{t('confirmDelete')}</Button></div>
      </div>}
    </Card> : <EmptyState mascot={false} title={t('noConfigured')} hint={t('noConfiguredHint')} />)}
  </div>
}

import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { Badge, Button, Card, EmptyState } from '../components/ui'
import { ErrorNotice, linkClass, Loading, useRiskTask, useRiskText } from '../components/risk-scales/shared'
import { formatDate } from '../i18n/runtime'
import { riskScales, type CatalogResponse, type DefaultsResponse } from '../services/riskScales'

type DeleteTarget = { kind: 'version' | 'draft'; id: string; name: string }

export default function RiskScaleCenter() {
  const { t } = useRiskText()
  const loadTask = useRiskTask(), actionTask = useRiskTask()
  const [catalog, setCatalog] = useState<CatalogResponse | null>(null), [defaults, setDefaults] = useState<DefaultsResponse | null>(null)
  const [pendingDelete, setPendingDelete] = useState<DeleteTarget | null>(null)
  const [notice, setNotice] = useState('')

  const reload = () => {
    setNotice('')
    void loadTask.run(async signal => { const [nextCatalog, nextDefaults] = await Promise.all([riskScales.catalog('limit=100', signal), riskScales.defaults(signal)]); return { nextCatalog, nextDefaults } }, ({ nextCatalog, nextDefaults }) => { setCatalog(nextCatalog); setDefaults(nextDefaults) })
  }

  useEffect(() => {
    reload()
    return () => { loadTask.invalidate(); actionTask.invalidate() }
  }, [])

  const defaultIds = new Set((defaults?.items ?? []).map(item => item.version_id).filter((id): id is string => Boolean(id)))
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
        return target
      }
      const summary = catalog.items.find(item => item.id === target.id)
      if (!summary) return target
      const [version, defaults] = await Promise.all([riskScales.version(summary.id, signal), riskScales.defaults(signal)])
      const definition = version.preview.request_echo.definition
      const key = `${definition.base_currency}:${definition.risk_basis_id}`
      const binding = defaults.items.find(item => item.key === key)
      await riskScales.retire(summary.id, {
        confirm: true,
        expected_revision: binding?.revision ?? 0,
        reason: t('deleteReason'),
        clear_default: binding?.version_id === summary.id,
      }, signal)
      return target
    }, result => {
      setCatalog(current => current ? {
        ...current,
        items: result.kind === 'version' ? current.items.filter(item => item.id !== result.id) : current.items,
        drafts: result.kind === 'draft' ? current.drafts.filter(item => item.id !== result.id) : current.drafts,
        total: result.kind === 'version' ? Math.max(0, current.total - 1) : current.total,
      } : current)
      setPendingDelete(null)
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
    {notice && <p role="status" className="text-sm text-emerald-800">{notice}</p>}
    {loadTask.busy && <Loading />}

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
              <Link className="hover:text-accent-700 hover:underline" to={row.kind === 'version' ? `/settings/risk-scales/versions/${encodeURIComponent(row.id)}` : `/settings/risk-scales/drafts/${encodeURIComponent(row.id)}`}>{row.name}</Link>
            </th>
            <td className="px-3 py-3"><div className="flex flex-wrap items-center gap-1"><Badge tone={row.kind === 'version' ? 'success' : 'warning'}>{t(row.kind === 'version' ? 'published' : 'draft')}</Badge>{row.kind === 'version' && defaultIds.has(row.id) && <Badge>{t('systemDefault')}</Badge>}{row.kind === 'version' && row.item.review_status === 'upcoming' && <Badge tone="warning">{t('reviewUpcoming')}</Badge>}{row.kind === 'version' && row.item.review_status === 'due' && <Badge tone="danger">{t('reviewDue')}</Badge>}{row.kind === 'version' && row.item.review_due_at && row.item.review_status !== 'none' && <span className="text-xs text-slate-600">{String(row.item.review_due_at)}</span>}</div></td>
            <td className="px-3 py-3 tabular-nums">{row.kind === 'version' ? `v${row.item.version_number ?? 1}` : t('revision', { revision: row.item.revision })}</td>
            <td className="px-3 py-3 whitespace-nowrap text-slate-600">{formatDate(row.at)}</td>
            <td className="px-4 py-2"><div className="flex justify-end gap-1">
              {row.kind === 'version' && <Link className={linkClass} to={`/settings/risk-scales/versions/${encodeURIComponent(row.id)}`}>{t('viewDetails')}</Link>}
              <Link className={linkClass} to={row.kind === 'version' ? `/settings/risk-scales/new?editFrom=${encodeURIComponent(row.id)}` : `/settings/risk-scales/drafts/${encodeURIComponent(row.id)}`}>{t('editConfig')}</Link>
              <Button tone="danger" disabled={actionTask.busy} onClick={() => setPendingDelete({ kind: row.kind, id: row.id, name: row.name })}>{t('deleteConfig')}</Button>
            </div></td>
          </tr>)}</tbody>
        </table>
      </div>
      {pendingDelete && <div className="flex flex-col gap-2 border-t border-slate-200 bg-slate-50 px-4 py-3 sm:flex-row sm:items-center sm:justify-between">
        <p className="text-sm text-slate-700">{t(pendingDelete.kind === 'version' ? 'deletePublishedConfirm' : 'deleteDraftConfirm', { name: pendingDelete.name })}</p>
        <div className="flex gap-2"><Button onClick={() => setPendingDelete(null)}>{t('cancel')}</Button><Button tone="danger" disabled={actionTask.busy} onClick={remove}>{t('confirmDelete')}</Button></div>
      </div>}
    </Card> : <EmptyState mascot={false} title={t('noConfigured')} hint={t('noConfiguredHint')} />)}
  </div>
}

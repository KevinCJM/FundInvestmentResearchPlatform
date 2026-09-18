import { useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { Badge, Button, Card, EmptyState } from '../components/ui'
import { Feedback, percentText } from '../components/risk-models/ResearchUI'
import { amountText } from '../components/investment-mandate/model'
import { useMandateText } from '../components/investment-mandate/text'
import { formatDate } from '../i18n/runtime'
import { deleteMandate, getStrategicCatalog, type MandateDefinition, type MandateVersion } from '../services/strategicAllocation'

const linkClass = 'inline-flex min-h-10 items-center rounded-lg px-3 text-sm font-medium text-accent-700 focus-visible:ring-2 focus-visible:ring-accent-500'

type DeleteTarget = Pick<MandateVersion, 'id' | 'name'>

function objectiveText(value: MandateDefinition, t: (key: string, values?: Record<string, string | number>) => string) {
  const kind = value.objective_kind ?? 'absolute_return'
  if (kind === 'funding_goal') return `${t(kind)} · ${amountText(value.funding_target?.amount)} ${value.currency}`
  if (kind === 'benchmark_relative') return `${t(kind)} · ${percentText(value.target_excess_return)}`
  return `${t(kind)} · ${percentText(value.target_return)}`
}

export default function InvestmentObjectivesCenter() {
  const { t } = useMandateText()
  const [items, setItems] = useState<MandateVersion[]>([])
  const [loading, setLoading] = useState(true), [busy, setBusy] = useState(false)
  const [error, setError] = useState(''), [notice, setNotice] = useState('')
  const [pendingDelete, setPendingDelete] = useState<DeleteTarget | null>(null)
  const generation = useRef(0), request = useRef<AbortController | null>(null)

  const load = () => {
    request.current?.abort(); const controller = new AbortController(); request.current = controller
    const token = ++generation.current; setLoading(true); setError(''); setNotice('')
    getStrategicCatalog(controller.signal).then(result => {
      if (token === generation.current && !controller.signal.aborted) setItems([...result.mandates].sort((a, b) => b.created_at.localeCompare(a.created_at)))
    }).catch(reason => {
      if (token === generation.current && !controller.signal.aborted) setError(reason instanceof Error ? reason.message : t('catalogFailed'))
    }).finally(() => { if (token === generation.current) setLoading(false) })
  }

  useEffect(() => { load(); return () => { generation.current += 1; request.current?.abort() } }, [])

  const remove = () => {
    if (!pendingDelete) return
    const target = pendingDelete
    request.current?.abort(); const controller = new AbortController(); request.current = controller
    const token = ++generation.current; setBusy(true); setError(''); setNotice('')
    deleteMandate(target.id, controller.signal).then(() => {
      if (token !== generation.current || controller.signal.aborted) return
      setItems(current => current.filter(item => item.id !== target.id)); setPendingDelete(null); setNotice(t('deletedNotice'))
    }).catch(reason => {
      if (token === generation.current && !controller.signal.aborted) setError(reason instanceof Error ? reason.message : t('operationFailed'))
    }).finally(() => { if (token === generation.current) setBusy(false) })
  }

  return <div className="min-w-0 space-y-4 text-slate-900">
    <header className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
      <div><h1 className="text-2xl font-bold">{t('title')}</h1><p className="mt-1 text-sm leading-6 text-slate-600">{t('centerDescription')}</p></div>
      <Link className="inline-flex min-h-10 items-center justify-center rounded-lg bg-accent-600 px-4 text-sm font-semibold text-white hover:bg-accent-700 focus-visible:ring-2 focus-visible:ring-accent-500" to="/pre-investment/objectives/new?fresh=1">{t('addObjective')}</Link>
    </header>

    <Feedback error={error} notice={notice} />
    {error && !loading && <Button onClick={load}>{t('retry')}</Button>}
    {loading && <div role="status" aria-live="polite" className="space-y-2"><p className="text-sm text-slate-600">{t('loadingObjectives')}</p>{[0, 1, 2].map(row => <div key={row} className="h-12 animate-pulse rounded-lg bg-slate-200 motion-reduce:animate-none" />)}</div>}

    {!loading && (items.length ? <Card className="overflow-hidden !p-0">
      <div className="overflow-x-auto"><table className="w-full min-w-[760px] text-sm" aria-label={t('objectivesTable')}>
        <caption className="sr-only">{t('objectivesTable')}</caption>
        <thead className="bg-slate-50 text-slate-600"><tr>
          <th scope="col" className="px-4 py-3 text-left">{t('name')}</th>
          <th scope="col" className="px-3 py-3 text-left">{t('objective')}</th>
          <th scope="col" className="px-3 py-3 text-left">{t('researchDate')}</th>
          <th scope="col" className="px-3 py-3 text-right">{t('riskLevel')}</th>
          <th scope="col" className="px-3 py-3 text-right">{t('cashFloor')}</th>
          <th scope="col" className="px-3 py-3 text-left">{t('publishedAt')}</th>
          <th scope="col" className="px-4 py-3 text-right">{t('actions')}</th>
        </tr></thead>
        <tbody>{items.map(item => {
          const definition = item.definition
          const level = definition.risk_authorization?.selected_max_level ?? definition.risk_authorization?.authorized_max_level
          const cash = definition.effective_cash_reserve_weight ?? definition.min_cash_weight ?? 0
          return <tr key={item.id} className="border-t border-slate-200">
            <th scope="row" className="px-4 py-3 text-left font-medium text-slate-900">{item.name}</th>
            <td className="px-3 py-3 text-slate-700">{objectiveText(definition, t)}</td>
            <td className="px-3 py-3 whitespace-nowrap text-slate-700">{definition.as_of}</td>
            <td className="px-3 py-3 text-right tabular-nums">{level ? `C${level}` : '—'}</td>
            <td className="px-3 py-3 text-right tabular-nums">{percentText(cash)}</td>
            <td className="px-3 py-3 whitespace-nowrap text-slate-600">{formatDate(item.created_at)}</td>
            <td className="px-4 py-2"><div className="flex justify-end gap-1">
              <Link className={linkClass} to={`/pre-investment/objectives/new?editFrom=${encodeURIComponent(item.id)}`}>{t('editObjective')}</Link>
              <Button tone="danger" disabled={busy} onClick={() => setPendingDelete({ id: item.id, name: item.name })}>{t('deleteObjective')}</Button>
            </div></td>
          </tr>
        })}</tbody>
      </table></div>
      {pendingDelete && <div className="flex flex-col gap-2 border-t border-slate-200 bg-slate-50 px-4 py-3 sm:flex-row sm:items-center sm:justify-between">
        <p className="text-sm leading-6 text-slate-700">{t('deleteObjectiveConfirm', { name: pendingDelete.name })}</p>
        <div className="flex gap-2"><Button disabled={busy} onClick={() => setPendingDelete(null)}>{t('cancel')}</Button><Button tone="danger" disabled={busy} onClick={remove}>{t('confirmDelete')}</Button></div>
      </div>}
    </Card> : <EmptyState mascot={false} title={t('noObjectives')} hint={t('noObjectivesHint')} action={<Link className={linkClass} to="/pre-investment/objectives/new?fresh=1">{t('addObjective')}</Link>} />)}
  </div>
}

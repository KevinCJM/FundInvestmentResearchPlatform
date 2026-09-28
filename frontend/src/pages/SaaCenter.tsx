import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { actionClass, Card, DataTable, EmptyState, ErrorPanel, LoadingPanel } from '../components/ui'
import { Field, inputClass, percentText } from '../components/risk-models/ResearchUI'
import { listSaaPolicies, type SavedSaaSummary } from '../services/strategicAllocation'
import { useI18n } from '../i18n/runtime'
import { UpstreamLink, UsabilityBadge, UsabilityNote, upstreamOf } from '../components/versioning'

const linkClass = 'inline-flex min-h-10 items-center break-words text-accent-700 underline'

export default function SaaCenter() {
  const { s } = useI18n()
  const [items, setItems] = useState<SavedSaaSummary[] | null>(null)
  const [error, setError] = useState(''), [retry, setRetry] = useState(0)
  const [query, setQuery] = useState('')
  const [newId] = useState(() => crypto.randomUUID())
  const newHref = `/pre-investment/saa/policy?new=${newId}`
  useEffect(() => {
    const controller = new AbortController()
    setItems(null); setError('')
    listSaaPolicies(controller.signal).then(value => {
      if (!controller.signal.aborted) setItems(value.items)
    }).catch(() => { if (!controller.signal.aborted) setError(s('saaCenter.loadFailed')) })
    return () => controller.abort()
  }, [retry])
  const visible = items?.filter(item => [item.name, item.mandate.name, item.scope.name, ...item.cmas.map(cma => cma.name)]
    .some(name => name?.toLocaleLowerCase().includes(query.trim().toLocaleLowerCase()))) ?? []
  const named = (name: string | null) => name || s('saaCenter.nameUnavailable')
  const reference = (name: string | null, href: string | null) => href
    ? <Link className={linkClass} to={href}>{named(name)}</Link> : <span>{named(name)}</span>
  const detailHref = (id: string) => `/pre-investment/saa/policy?baseline=${encodeURIComponent(id)}`

  return <div className="min-w-0 space-y-4 text-slate-900">
    <header className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
      <div className="min-w-0"><h1 className="text-2xl font-bold sm:text-3xl">{s('saaScope.title')}</h1>
        <p className="mt-2 text-sm leading-6 text-slate-600">{s('saaCenter.description')}</p></div>
      <Link className={`${actionClass('primary')} shrink-0`} to={newHref}>{s('saaCenter.create')}</Link>
    </header>
    {error ? <ErrorPanel message={error} onRetry={() => setRetry(value => value + 1)} />
      : !items ? <LoadingPanel text={s('saaCenter.loading')} />
      : !items.length ? <EmptyState title={s('saaCenter.empty')} hint={s('saaCenter.emptyHint')} />
      : <>
        <div className="max-w-xl"><Field label={s('saaCenter.search')}><input className={inputClass} value={query} onChange={event => setQuery(event.target.value)} /></Field></div>
        <Card className="!p-0 overflow-hidden"><DataTable<SavedSaaSummary> caption={s('saaCenter.list')} rows={visible} rowKey={item => item.id} minWidth="920px"
          empty={s('saaCenter.noResults')} columns={[
            { header: s('saaCenter.name'), cell: item => <>{reference(item.name, detailHref(item.id))}<p className="text-xs font-normal text-slate-600">{s('saaCenter.researchDate')} {item.as_of}</p>
              <div className="mt-1"><UsabilityBadge usable={item.usable} /></div><UsabilityNote usable={item.usable} /></> },
            { header: s('saaCenter.mandate'), cell: item => {
              const d = item.mandate.definition
              const cash = d.effective_cash_reserve_weight ?? d.min_cash_weight
              const target = d.effective_target_return ?? d.target_return
              const ref = upstreamOf(item, 'mandate')
              return <>{ref?.id ? <UpstreamLink item={ref} /> : reference(item.mandate.name, item.mandate.id ? `/pre-investment/objectives/new?view=${encodeURIComponent(item.mandate.id)}` : null)}
                <div className="space-y-1 text-xs text-slate-600">
                  {d.objective_kind !== 'funding_goal' && d.objective_kind !== 'benchmark_relative' && typeof target === 'number' && <p>{s('saaCenter.returnFloor', { value: percentText(target) })}</p>}
                  {typeof d.max_volatility === 'number' && <p>{s('saaCenter.riskCap', { value: percentText(d.max_volatility) })}</p>}
                  {typeof cash === 'number' && <p>{s('saaCenter.cashFloor', { value: percentText(cash) })}</p>}
                </div></>
            } },
            { header: s('saaCenter.scope'), cell: item => <>
              <p className="text-xs text-slate-600">{s(item.scope.research_path === 'strategy_first' ? 'ltcma.pathStrategyFirst' : 'ltcma.pathProductFirst')}</p>
              {reference(item.scope.name, item.scope.id ? `/pre-investment/product-pool/new?${item.scope.research_path === 'strategy_first' ? 'scope=strategic&strategic_universe' : 'universe'}=${encodeURIComponent(item.scope.id)}` : null)}
            </> },
            { header: 'LTCMA', cell: item => <>
              <p className="text-xs text-slate-600">{s(item.mode === 'compatible_all_models' ? 'multiCma.common' : item.mode === 'parameter_average' ? 'multiCma.average' : 'saaCenter.single')}</p>
              <div>{item.cmas.map((cma, i) => <div key={cma.id ?? i}>{(ref => ref ? <UpstreamLink item={ref} /> : reference(cma.name, cma.id ? `/pre-investment/ltcma/${encodeURIComponent(cma.id)}` : null))(item.upstream?.find(value => value.kind === 'cma' && value.id === cma.id))}</div>)}</div>
            </> },
            { header: s('saaCenter.actions'), cell: item => <Link className={actionClass('secondary')} to={detailHref(item.id)}>{s('saaCenter.view')}</Link> },
          ]} /></Card>
      </>}
  </div>
}

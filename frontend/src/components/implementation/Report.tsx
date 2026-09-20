import { Badge, Card } from '../ui'
import type { ImplementationReport, ValidationCheck } from '../../services/implementation'
import { amount, percent, useImplementationText } from './shared'

export default function ImplementationReportView({report}: {report: ImplementationReport}) {
  const t = useImplementationText()
  return <div className="space-y-4">
    <div className="flex flex-wrap gap-2" role="status"><Badge tone={report.research_ready ? 'neutral' : 'warning'}>{t(report.research_ready ? 'researchReady' : 'researchBlocked')}</Badge>
      <Badge tone={report.implementation_eligibility==='eligible' ? 'success':'warning'}>{t(report.implementation_eligibility==='eligible'?'implementationReady':'implementationIncomplete')}</Badge>
      <Badge>{t(report.independent_simulation?'independentSimulation':report.validation_mode==='frozen_candidate_validation'?'candidateValidated':'previewOnly')}</Badge></div>
    <Card><h2 className="mb-3 text-lg font-semibold">{t('checks')}</h2>
      <CheckList items={report.checks.filter(check=>check.status!=='passed')}/>
      {report.checks.some(check=>check.status==='passed')&&<details className="mt-3 border-t border-slate-200 pt-3"><summary className="cursor-pointer text-sm font-medium">{t('passedChecks',{count:report.checks.filter(check=>check.status==='passed').length})}</summary><CheckList items={report.checks.filter(check=>check.status==='passed')}/></details>}
    </Card>
    {report.models.length>0 && <Card><h2 className="mb-3 text-lg font-semibold">{t('productRisk')}</h2><p className="mb-2 text-xs text-slate-600 sm:hidden">{t('tableSwipe')}</p><div className="overflow-x-auto" tabIndex={0} aria-label={t('productRisk')}><table className="min-w-[640px] w-full text-sm" aria-label={t('productRisk')}>
      <thead><tr>{['model','expectedReturn','volatility','totalActiveRisk','implementationTe'].map(key=><th scope="col" key={key} className="px-3 py-2 text-right first:text-left">{t(key)}</th>)}</tr></thead>
      <tbody>{report.models.map(row=><tr key={row.model_id} className="border-b border-slate-200"><th scope="row" className="px-3 py-3 text-left font-medium">{row.name}<span className="block text-xs text-slate-600">{t(row.enforced?'requiredModel':'diagnosticModel')}</span></th>{[row.expected_return,row.volatility,row.total_active_risk,row.implementation_tracking_error].map((v,i)=><td key={i} className="whitespace-nowrap px-3 py-3 text-right tabular-nums">{percent(v)}</td>)}</tr>)}</tbody>
    </table></div></Card>}
    {report.transition && <Card><h2 className="mb-3 text-lg font-semibold">{t('costResult')}</h2><dl className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">{[
      ['tradeCost',amount(report.transition.cost)], ['postCostValue',amount(report.transition.post_cost_value)],
      ['grossTrades',amount(report.transition.gross_traded_notional)], ['halfTurnover',percent(report.transition.half_turnover)],
    ].map(([key,value])=><div key={key}><dt className="text-sm text-slate-600">{t(key)}</dt><dd className="mt-1 text-lg font-semibold tabular-nums">{value}</dd></div>)}</dl></Card>}
    {report.funding_results && <Card><h2 className="mb-3 text-lg font-semibold">{t('fundingResult')}</h2><div className="divide-y divide-slate-200">{report.funding_results.map(row=><div key={row.model_id} className="space-y-2 py-3">
      <h3 className="font-medium">{row.name} · {t('remainingMonths',{count:row.remaining_months})}</h3><p className="text-sm tabular-nums">{t('futureProbability')}: {percent(row.future_conditional_success_probability)} · {t('probabilityLower')}: {percent(row.metrics?.probability_lower)}</p>
      <p className="text-sm tabular-nums">{t('terminalMedian')}: {amount(row.metrics?.terminal_median)}</p>
      {row.original_plan_missed_payment && <p className="text-sm font-medium text-amber-900">{t('historicalMissed')}</p>}
    </div>)}</div><p className="mt-3 text-sm text-slate-600">{t('fundingScope')}</p></Card>}
    {report.cash_calendar && <Card><h2 className="mb-3 text-lg font-semibold">{t('cashCalendar')}</h2><div className="divide-y divide-slate-200">{report.cash_calendar.events.map(event=><div key={event.id} className="flex flex-wrap justify-between gap-2 py-2 text-sm"><span>{event.date} · {event.id}</span><span className="tabular-nums">{t('availableCash')}: {amount(event.available_balance)}{event.cash_gap>1e-7 && ` · ${t('cashGap')}: ${amount(event.cash_gap)}`}</span></div>)}</div></Card>}
    {report.product_paths&&<Card><h2 className="text-lg font-semibold">{t('productPaths')}</h2><div className="divide-y divide-slate-200">{report.product_paths.results.map(row=><div key={row.model_id} className="space-y-2 py-3"><h3 className="font-medium">{row.name} · {t(row.status)}</h3><p className="text-sm tabular-nums">{t('futureProbability')}: {percent(row.metrics.success_probability)} · {t('probabilityLower')}: {percent(row.metrics.probability_lower)}</p><p className="text-sm tabular-nums">{t('paymentFailure')}: {percent(row.metrics.payment_failure_probability)} · {t('terminalMedian')}: {amount(row.metrics.terminal_median)}</p><p className="text-sm tabular-nums">{t('futurePathCost')}: {amount(row.metrics.expected_path_cost)}</p>{row.original_plan_missed_payment&&<p className="text-sm text-amber-900">{t('historicalMissed')}</p>}</div>)}</div><details className="mt-3"><summary className="cursor-pointer text-sm font-medium">{t('pathAssumptions')}</summary>{report.product_paths.assumptions.map(text=><p key={text} className="mt-2 text-sm leading-6 text-slate-600">{text}</p>)}</details></Card>}
    {report.scenarios&&report.scenarios.results.length>0&&<Card><h2 className="text-lg font-semibold">{t('publishedScenarios')}</h2><p className="mt-2 text-sm leading-6 text-slate-600">{t('scenarioScope')}</p><div className="divide-y divide-slate-200">{report.scenarios.results.map(({impact})=><div key={impact.id} className="space-y-2 py-3"><h3 className="font-medium">{impact.name}</h3><p className="text-sm tabular-nums">{t('scenarioReturn')}: {percent(impact.summary.terminal_return)} · {t('scenarioDrawdown')}: {percent(impact.summary.max_drawdown)} · {t('scenarioPnl')}: {amount(impact.summary.pnl_amount)}</p></div>)}</div></Card>}
    {report.historical_replay&&<Card><h2 className="text-lg font-semibold">{t('historicalReplay')}</h2><p className="mt-3 text-sm tabular-nums">{t('grossTerminal')}: {amount(report.historical_replay.gross_terminal,6)} · {t('netTerminal')}: {amount(report.historical_replay.net_terminal,6)}</p><p className="mt-2 text-sm leading-6 text-slate-600">{t('historicalReplayScope')}</p></Card>}
    <details className="rounded-xl border border-slate-200 p-4"><summary className="cursor-pointer text-sm font-medium">{t('boundaries')}</summary><div className="mt-3 space-y-2">{report.limitations.map(text=><p key={text} className="text-sm leading-6 text-slate-600">{text}</p>)}<p className="break-all text-xs text-slate-600">{t('candidateFingerprint')}: {report.candidate_hash}</p></div></details>
  </div>
}

function CheckList({items}:{items:ValidationCheck[]}) {
  const t=useImplementationText()
  return <div className="divide-y divide-slate-200">{items.map(check=><div key={check.check_id} className="py-3">
    <div className="flex flex-wrap items-center justify-between gap-2"><h3 className="text-sm font-medium">{check.title}</h3><Badge tone={check.status==='passed'?'success':check.status==='failed'?'danger':check.status==='unavailable'?'warning':'neutral'}>{t(check.status)}</Badge></div>
    <p className="mt-2 break-words text-sm leading-6 text-slate-600">{check.reason}</p>
  </div>)}</div>
}

import { systemText, useI18n } from '../i18n/runtime'
import { useEffect, useMemo, useState } from 'react'
import {
  listHistoricalRegimeRuns,
  type HistoricalRegimeRun,
  type RegimePublication,
} from '../services/historicalRegimes'
import { eligibleRegimePublications } from '../services/regimePublicationEligibility'
import type {
  HistoricalRegimeBacktestReference,
  RegimeConditioningResult,
} from '../services/portfolioRegime'

export type { HistoricalRegimeBacktestReference, RegimeConditioningResult } from '../services/portfolioRegime'

interface EligiblePublication {
  run: HistoricalRegimeRun
  publication: RegimePublication
}

export function eligibleFormalBacktestPublications(runs: HistoricalRegimeRun[]): EligiblePublication[] {
  return runs.flatMap((run) => eligibleRegimePublications(run, 'formal_backtest')
    .map((publication) => ({ run, publication })))
}

function publicationValue(item: EligiblePublication) {
  return `${item.run.id}|${item.publication.id}`
}

export function HistoricalRegimeBacktestSelector({
  value,
  onChange,
  disabled = false,
}: {
  value: HistoricalRegimeBacktestReference | null
  onChange: (value: HistoricalRegimeBacktestReference | null) => void
  disabled?: boolean
}) {
  useI18n()
  const [runs, setRuns] = useState<HistoricalRegimeRun[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    let active = true
    setLoading(true)
    setError('')
    listHistoricalRegimeRuns()
      .then((items) => { if (active) setRuns(items) })
      .catch((caught) => {
        if (active) setError(caught instanceof Error ? caught.message : systemText('preInvestment.historicalRegimeBacktest.unableToLoadHistoricalRegimePublications'))
      })
      .finally(() => { if (active) setLoading(false) })
    return () => { active = false }
  }, [])

  const eligible = useMemo(() => eligibleFormalBacktestPublications(runs), [runs])
  const selected = eligible.find((item) => (
    item.run.id === value?.run_id && item.publication.id === value?.publication_id
  ))

  return <section className="rounded-xl border border-accent-200 bg-accent-50/60 p-3" aria-label={systemText('preInvestment.historicalRegimeBacktest.historicalRegimeConditioning')}>
    <label className="block text-sm font-semibold text-slate-800" htmlFor="historical-regime-backtest">
      {systemText('preInvestment.historicalRegimeBacktest.historicalRegimeConditioningOptional')}</label>
    <select
      id="historical-regime-backtest"
      disabled={disabled || loading}
      value={selected ? publicationValue(selected) : ''}
      onChange={(event) => {
        if (!event.target.value) return onChange(null)
        const item = eligible.find((candidate) => publicationValue(candidate) === event.target.value)
        onChange(item ? { run_id: item.run.id, publication_id: item.publication.id } : null)
      }}
      className="mt-2 w-full rounded-lg border border-accent-200 bg-white px-3 py-2 text-sm text-slate-800 disabled:bg-slate-100"
    >
      <option value="">{systemText('preInvestment.historicalRegimeBacktest.doNotSplitPerformanceByHistoricalRegime')}</option>
      {eligible.map((item) => <option key={publicationValue(item)} value={publicationValue(item)}>
        {item.run.name} {" " + systemText('preInvestment.historicalRegimeBacktest.definitionR')}{item.run.definition_revision ?? '—'} · {item.publication.published_at.slice(0, 10)}
      </option>)}
    </select>
    <p className="mt-2 text-xs leading-5 text-slate-600" aria-live="polite">
      {loading
        ? systemText('preInvestment.historicalRegimeBacktest.loadingVersionsEligibleForFormalBacktesting')
        : error
          ? error
          : eligible.length
            ? systemText('preInvestment.historicalRegimeBacktest.onlyImmutableRealTimeCausalV2Publications')
            : systemText('preInvestment.historicalRegimeBacktest.noHistoricalRegimePublicationsHavePassedReal')}
    </p>
    {selected ? <dl className="mt-2 grid gap-1 rounded-lg bg-white p-2 text-xs text-slate-600 sm:grid-cols-2">
      <div><dt className="inline text-slate-600">{systemText('preInvestment.historicalRegimeBacktest.run')}</dt><dd className="inline font-medium text-slate-800">{selected.run.id}</dd></div>
      <div><dt className="inline text-slate-600">{systemText('preInvestment.historicalRegimeBacktest.publication')}</dt><dd className="inline font-medium text-slate-800">{selected.publication.id}</dd></div>
    </dl> : null}
  </section>
}

function percent(value?: number | null, digits = 2) {
  return typeof value === 'number' && Number.isFinite(value) ? `${(value * 100).toFixed(digits)}%` : '—'
}

function number(value?: number | null, digits = 3) {
  return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : '—'
}

export function RegimeConditioningPanel({ result }: { result?: RegimeConditioningResult | null }) {
  useI18n()
  if (!result) return null
  const performance = Object.entries(result.conditional_performance ?? {})
  const periodStates = result.period_states ?? []
  const periodCount = result.period_states_count ?? periodStates.length

  return <section className="rounded-xl border border-accent-200 bg-accent-50/40 p-4" aria-label={systemText('preInvestment.historicalRegimeBacktest.performanceConditionalOnHistoricalRegimes')}>
    <div className="flex flex-wrap items-start justify-between gap-3">
      <div><h3 className="font-semibold text-slate-900">{systemText('preInvestment.historicalRegimeBacktest.performanceConditionalOnHistoricalRegimes')}</h3><p className="mt-1 text-xs text-slate-600">{systemText('preInvestment.historicalRegimeBacktest.usesRealTimeStatesEffectiveAtThe')}</p></div>
      <span className="rounded-full bg-accent-100 px-3 py-1 text-xs font-semibold text-accent-800">{systemText('preInvestment.historicalRegimeBacktest.versionLockedForFormalBacktest')}</span>
    </div>
    <dl className="mt-3 grid gap-2 rounded-lg bg-white p-3 text-xs sm:grid-cols-2 lg:grid-cols-4">
      <div><dt className="text-slate-600">{systemText('preInvestment.historicalRegimeBacktest.runId')}</dt><dd className="mt-1 break-all font-semibold text-slate-800">{result.binding.run_id}</dd></div>
      <div><dt className="text-slate-600">{systemText('preInvestment.historicalRegimeBacktest.publicationId')}</dt><dd className="mt-1 break-all font-semibold text-slate-800">{result.binding.publication_id}</dd></div>
      <div><dt className="text-slate-600">{systemText('preInvestment.historicalRegimeBacktest.definitionVersion')}</dt><dd className="mt-1 font-semibold text-slate-800">{result.binding.definition_id ?? '—'} · R{result.binding.definition_revision ?? '—'}</dd></div>
      <div><dt className="text-slate-600">{systemText('preInvestment.historicalRegimeBacktest.classificationCoverage')}</dt><dd className="mt-1 font-semibold text-slate-800">{percent(result.coverage.classified_ratio)} · {result.coverage.classified_periods}/{result.coverage.periods} {" " + systemText('preInvestment.historicalRegimeBacktest.periods')}</dd></div>
    </dl>
    <div className="mt-4 space-y-3">
      {performance.map(([strategyName, rows], index) => <details key={strategyName} open={index === 0} className="rounded-xl border border-slate-200 bg-white">
        <summary className="cursor-pointer px-3 py-2 text-sm font-semibold text-slate-800">{strategyName} {" " + systemText('preInvestment.historicalRegimeBacktest.performanceByState')}</summary>
        <div className="overflow-x-auto border-t border-slate-200">
          <table className="min-w-[860px] w-full text-left text-xs">
            <thead className="bg-slate-50 text-slate-600"><tr><th scope="col" className="px-3 py-2">{systemText('preInvestment.historicalRegimeBacktest.status')}</th><th scope="col" className="px-3 py-2">{systemText('preInvestment.historicalRegimeBacktest.intervalCount')}</th><th scope="col" className="px-3 py-2">{systemText('preInvestment.historicalRegimeBacktest.returnObservations')}</th><th scope="col" className="px-3 py-2">{systemText('preInvestment.historicalRegimeBacktest.annualized')}</th><th scope="col" className="px-3 py-2">{systemText('preInvestment.historicalRegimeBacktest.volatility')}</th><th scope="col" className="px-3 py-2">{systemText('preInvestment.historicalRegimeBacktest.maximumDrawdown')}</th><th scope="col" className="px-3 py-2">{systemText('preInvestment.historicalRegimeBacktest.sharpe')}</th><th scope="col" className="px-3 py-2">{systemText('preInvestment.historicalRegimeBacktest.positiveReturnRate')}</th></tr></thead>
            <tbody>{rows.map((row) => <tr key={row.state_id} className="border-t border-slate-100"><td className="px-3 py-2 font-medium text-slate-800">{row.state_label}</td><td className="px-3 py-2">{row.observations}</td><td className="px-3 py-2">{row.return_observations}</td><td className="px-3 py-2">{percent(row.annualized_return)}</td><td className="px-3 py-2">{percent(row.volatility)}</td><td className="px-3 py-2">{percent(row.max_drawdown)}</td><td className="px-3 py-2">{number(row.sharpe)}</td><td className="px-3 py-2">{percent(row.positive_rate)}</td></tr>)}</tbody>
          </table>
        </div>
      </details>)}
      {!performance.length ? <p className="rounded-lg bg-white p-3 text-sm text-slate-600">{systemText('preInvestment.historicalRegimeBacktest.thisRunHasNoStateLevelStatistics')}</p> : null}
    </div>
    <details className="mt-4 rounded-xl border border-slate-200 bg-white">
      <summary className="cursor-pointer px-3 py-2 text-sm font-semibold text-slate-800">{systemText('preInvestment.historicalRegimeBacktest.perPeriodStateDetails')}{periodCount} {" " + systemText('preInvestment.historicalRegimeBacktest.periods2')}</summary>
      {periodStates.length ? <div className="max-h-80 overflow-auto border-t border-slate-200">
        <table className="min-w-[760px] w-full text-left text-xs"><thead className="sticky top-0 bg-slate-50 text-slate-600"><tr><th scope="col" className="px-3 py-2">{systemText('preInvestment.historicalRegimeBacktest.returnInterval')}</th><th scope="col" className="px-3 py-2">{systemText('preInvestment.historicalRegimeBacktest.periodStart')}</th><th scope="col" className="px-3 py-2">{systemText('preInvestment.historicalRegimeBacktest.effectiveState')}</th><th scope="col" className="px-3 py-2">{systemText('preInvestment.historicalRegimeBacktest.stateEffectiveDate')}</th><th scope="col" className="px-3 py-2">{systemText('preInvestment.historicalRegimeBacktest.identificationTime')}</th><th scope="col" className="px-3 py-2">{systemText('preInvestment.historicalRegimeBacktest.confidence')}</th></tr></thead><tbody>{periodStates.map((row) => <tr key={`${row.period_start}-${row.date}`} className="border-t border-slate-100"><td className="px-3 py-2">{row.date}</td><td className="px-3 py-2">{row.period_start}</td><td className="px-3 py-2 font-medium">{row.state_label}</td><td className="px-3 py-2">{row.regime_effective_date ?? '—'}</td><td className="px-3 py-2">{row.regime_recognized_at ?? '—'}</td><td className="px-3 py-2">{percent(row.confidence)}</td></tr>)}</tbody></table>
      </div> : <p className="border-t border-slate-200 p-3 text-xs text-slate-600">{systemText('preInvestment.historicalRegimeBacktest.theListResponseOmitsPerPeriodDetails') + " "}{periodCount} {" " + systemText('preInvestment.historicalRegimeBacktest.periods3')}</p>}
    </details>
  </section>
}

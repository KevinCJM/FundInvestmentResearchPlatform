import { systemText, useI18n } from '../i18n/runtime'
import type { PitAllocationLineage } from '../services/pit'

/**
 * The口径 a backtest actually stood on, printed on the backtest.
 *
 * Two things a reader cannot infer from the curve: whether the class NAV behind
 * it was rebuilt for the research day or carried over from a full-history run,
 * and whether the refits at each rebalance saw only what had been published by
 * then. A run missing either is still shown — it is most of the installed base
 * — but it must not look identical to one that has both.
 */
export default function PitDecisionNotice({ lineage }: { lineage?: PitAllocationLineage | null }) {
  useI18n()
  if (!lineage) return null

  const findings = lineage.universe?.findings ?? []
  const clean = !lineage.hindsight_series && lineage.availability_available && findings.length === 0
  const badges: { text: string; tone: string }[] = [
    lineage.availability_available
      ? { text: systemText('preInvestment.pitDecisionNotice.windowByAnnouncementAvailability'), tone: 'bg-emerald-100 text-emerald-800 border-emerald-200' }
      : { text: systemText('preInvestment.pitDecisionNotice.windowByNavDateIncludesAnnouncementLag'), tone: 'bg-amber-100 text-amber-900 border-amber-200' },
    lineage.hindsight_series
      ? { text: systemText('preInvestment.pitDecisionNotice.fullHistorySeries'), tone: 'bg-rose-100 text-rose-800 border-rose-200' }
      : {
          text: systemText('preInvestment.pitDecisionNotice.seriesConvention', { p0: lineage.series_as_of ?? systemText('preInvestment.pitDecisionNotice.latest') }),
          tone: 'bg-slate-100 text-slate-700 border-slate-200',
        },
    // 产品池的研究日是配置的第二个时间声明，跟净值口径可以差好几年。
    lineage.universe?.snapshot_established_at
      ? {
          text: systemText('preInvestment.pitDecisionNotice.productUniverseResearchDate', { p0: lineage.universe.snapshot_established_at }),
          tone: 'bg-slate-100 text-slate-700 border-slate-200',
        }
      : { text: systemText('preInvestment.pitDecisionNotice.productUniverseNotRecorded'), tone: 'bg-amber-100 text-amber-900 border-amber-200' },
  ]

  return (
    <div
      className={`mt-3 rounded-lg border px-2.5 py-2 text-xs leading-5 ${
        clean ? 'border-slate-200 bg-slate-50 text-slate-600' : 'border-amber-200 bg-amber-50 text-amber-900'
      }`}
      data-testid="pit-decision-notice"
    >
      <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
        <span className="font-semibold">{systemText('preInvestment.pitDecisionNotice.decisionContext')}</span>
        {lineage.as_of ? (
          <span className="tabular-nums">{systemText('preInvestment.pitDecisionNotice.researchDate') + " "}{lineage.as_of}</span>
        ) : (
          <span>{systemText('preInvestment.pitDecisionNotice.researchDateNotSet')}</span>
        )}
        {badges.map((badge) => (
          <span key={badge.text} className={`rounded-lg border px-1.5 py-0.5 ${badge.tone}`}>
            {badge.text}
          </span>
        ))}
        {lineage.rows_dropped_by_as_of > 0 && (
          <span className="tabular-nums text-slate-600">
            {systemText('preInvestment.pitDecisionNotice.excluded') + " "}{lineage.rows_dropped_by_as_of.toLocaleString()} {" " + systemText('preInvestment.pitDecisionNotice.navRowsNotYetAvailableAtThat')}</span>
        )}
      </div>
      {lineage.warnings.map((text) => (
        <p key={text} className="mt-1 text-rose-700">
          {text}
        </p>
      ))}
      {/* 候选集合的前视不在任何一条公式里，后端一直算着也一直传着，这里以前没印。 */}
      {findings.map((finding) => (
        <p key={finding.code + finding.label} className="mt-1 text-rose-700" data-testid="pit-universe-finding">
          {finding.message}
        </p>
      ))}
    </div>
  )
}

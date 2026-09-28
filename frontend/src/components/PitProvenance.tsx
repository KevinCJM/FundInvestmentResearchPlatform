import { systemText, useI18n } from '../i18n/runtime'
import { Link } from 'react-router-dom'
import type { PitRunLineage } from '../services/pit'

/**
 * The口径 a result was actually computed under, printed on the result.
 *
 * Provenance belongs with the data, not with the window frame: a number you can
 * read without knowing its cut-off is the reproducibility problem this whole
 * feature exists to prevent. A run with no PIT口径 is labelled as such — it must
 * not look identical to a strict point-in-time run.
 */
export default function PitProvenance({ lineage }: { lineage?: PitRunLineage | null }) {
  useI18n()
  if (!lineage) return null

  const applied = lineage.as_of_applied
  const strict = lineage.run_mode === 'STRICT_PIT'

  return (
    <div
      className={`mt-3 rounded-lg border px-2.5 py-2 text-xs leading-5 ${
        applied ? 'border-slate-200 bg-slate-50 text-slate-600' : 'border-amber-200 bg-amber-50 text-amber-900'
      }`}
      data-testid="pit-provenance"
    >
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
        <span className="font-semibold">{systemText('preInvestment.pitProvenance.dataContext')}</span>
        {applied ? (
          <>
            <span className="tabular-nums">{systemText('preInvestment.pitProvenance.researchDate') + " "}{lineage.as_of}</span>
            <span>{strict ? systemText('preInvestment.pitProvenance.strictPit') : systemText('preInvestment.pitProvenance.researchMode')}</span>
            <span className="tabular-nums">
              {systemText('preInvestment.pitProvenance.cutOffAt') + " "}{lineage.availability_field} {" " + systemText('preInvestment.pitProvenance.byAnnouncementAvailabilityExcluded') + " "}{lineage.rows_dropped_by_as_of.toLocaleString()} {" " + systemText('preInvestment.pitProvenance.navRowsNotYetAnnounced')}</span>
            <span className="tabular-nums text-slate-600">{systemText('preInvestment.pitProvenance.available') + " "}{lineage.rows_after_cut.toLocaleString()} {" " + systemText('preInvestment.pitProvenance.rows')}</span>
          </>
        ) : (
          <>
            <span className="font-semibold">{systemText('preInvestment.pitProvenance.noPitContext')}</span>
            <span>{systemText('preInvestment.pitProvenance.usesAllOnDiskDataWithoutAnnouncement')}</span>
            <Link to="/settings/pit-snapshots" className="underline hover:no-underline">
              {systemText('preInvestment.pitProvenance.applyADataVersionInPitSettings')}</Link>
          </>
        )}
      </div>
      {lineage.announcement_fallback && (
        <p className="mt-1 text-rose-700">
          {systemText('preInvestment.pitProvenance.ofThese') + " "}{lineage.rows_without_announcement.toLocaleString()} {" " + systemText('preInvestment.pitProvenance.rowsLackAnnouncementDatesAndUseNav')}</p>
      )}
      {/* 产品域的时点问题不在任何一条公式里，只在候选集合里，所以印在口径旁边而不是
          让读者自己去比对产品池的研究日期。 */}
      {(lineage.universe?.findings ?? []).map((finding) => (
        <p key={finding.code + finding.label} className="mt-1 text-rose-700" data-testid="pit-universe-finding">
          {finding.message}
        </p>
      ))}
    </div>
  )
}

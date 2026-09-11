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
  if (!lineage) return null

  const applied = lineage.as_of_applied
  const strict = lineage.run_mode === 'STRICT_PIT'

  return (
    <div
      className={`mt-3 rounded border px-2.5 py-2 text-[11px] leading-5 ${
        applied ? 'border-slate-200 bg-slate-50 text-slate-600' : 'border-amber-200 bg-amber-50 text-amber-900'
      }`}
      data-testid="pit-provenance"
    >
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
        <span className="font-semibold">口径</span>
        {applied ? (
          <>
            <span className="tabular-nums">研究日 {lineage.as_of}</span>
            <span>{strict ? '严格 PIT' : '研究模式'}</span>
            <span className="tabular-nums">
              按 {lineage.availability_field} 公告时点截断，剔除 {lineage.rows_dropped_by_as_of.toLocaleString()} 行当时尚未公告的净值
            </span>
            <span className="tabular-nums text-slate-500">可用 {lineage.rows_after_cut.toLocaleString()} 行</span>
          </>
        ) : (
          <>
            <span className="font-semibold">无 PIT 口径</span>
            <span>使用全部磁盘数据，未按公告时点截断 —— 该结果不具备时点可复现性</span>
            <Link to="/settings/pit-snapshots" className="underline hover:no-underline">
              前往 PIT 设置应用一个数据版本
            </Link>
          </>
        )}
      </div>
      {lineage.announcement_fallback && (
        <p className="mt-1 text-rose-700">
          其中 {lineage.rows_without_announcement.toLocaleString()} 行缺少公告日，已按净值日期近似；该部分不具备严格时点证明。
        </p>
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

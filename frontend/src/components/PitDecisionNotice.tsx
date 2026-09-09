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
  if (!lineage) return null

  const clean = !lineage.hindsight_series && lineage.availability_available
  const badges: { text: string; tone: string }[] = [
    lineage.availability_available
      ? { text: '按公告时点切窗', tone: 'bg-emerald-100 text-emerald-800 border-emerald-200' }
      : { text: '按净值日期切窗 · 含公告滞后穿越', tone: 'bg-amber-100 text-amber-900 border-amber-200' },
    lineage.hindsight_series
      ? { text: '全历史口径序列', tone: 'bg-rose-100 text-rose-800 border-rose-200' }
      : {
          text: `序列口径 ${lineage.series_as_of ?? '最新'}`,
          tone: 'bg-slate-100 text-slate-700 border-slate-200',
        },
  ]

  return (
    <div
      className={`mt-3 rounded border px-2.5 py-2 text-[11px] leading-5 ${
        clean ? 'border-slate-200 bg-slate-50 text-slate-600' : 'border-amber-200 bg-amber-50 text-amber-900'
      }`}
      data-testid="pit-decision-notice"
    >
      <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
        <span className="font-semibold">决策口径</span>
        {lineage.as_of ? (
          <span className="tabular-nums">研究日 {lineage.as_of}</span>
        ) : (
          <span>未设研究日</span>
        )}
        {badges.map((badge) => (
          <span key={badge.text} className={`rounded border px-1.5 py-0.5 ${badge.tone}`}>
            {badge.text}
          </span>
        ))}
        {lineage.rows_dropped_by_as_of > 0 && (
          <span className="tabular-nums text-slate-500">
            剔除 {lineage.rows_dropped_by_as_of.toLocaleString()} 行当时尚未可得的净值
          </span>
        )}
      </div>
      {lineage.warnings.map((text) => (
        <p key={text} className="mt-1 text-rose-700">
          {text}
        </p>
      ))}
    </div>
  )
}

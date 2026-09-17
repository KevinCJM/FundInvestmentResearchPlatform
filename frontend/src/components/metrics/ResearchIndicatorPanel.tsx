import { Link } from 'react-router-dom'
import { useI18n } from '../../i18n/runtime'
import { MetricSelector } from './MetricDisplay'
import MetricPeriodTable from './MetricPeriodTable'
import { indicatorPeriodLabel, indicatorPeriodOptionLabel } from '../../utils/indicatorPeriods'
import type {
  EvaluationResult,
  IndicatorDefinition,
} from '../../services/customIndicators'

export const MAX_RESEARCH_INDICATORS = 8
/** Each column costs one evaluate request, so the width of the board is capped. */
export const MAX_RESEARCH_PERIODS = 6

interface Props {
  indicators: IndicatorDefinition[]
  selectedIds: string[]
  onSelectedIdsChange: (ids: string[]) => void
  periods: string[]
  onPeriodsChange: (periods: string[]) => void
  periodOptions: string[]
  parametersFor: (indicatorId: string) => Record<string, number>
  onParametersChange: (indicatorId: string, values: Record<string, number>) => void
  results: EvaluationResult[]
  loading: boolean
  error: string | null
  asOf: string
  onAsOfChange: (value: string) => void
  asOfHint: string
  onDefinition: (indicator: IndicatorDefinition) => void
  studioHref: string
}

/**
 * Scalar indicators only: rows are indicators, columns are periods.
 *
 * The period used to be a select inside every card, which printed the same
 * control and the same window line once per indicator. It is one question
 * about the whole board, so it is asked once, and answering it with more than
 * one period is what turns a lone number into a term structure.
 */
export default function ResearchIndicatorPanel({
  indicators, selectedIds, onSelectedIdsChange, periods, onPeriodsChange, periodOptions,
  parametersFor, onParametersChange, results, loading, error, asOf, onAsOfChange, asOfHint,
  onDefinition, studioHref,
}: Props) {
  const { s } = useI18n()
  const catalog = indicators.filter(item => (item.result_kind ?? 'scalar') === 'scalar')
  const selected = selectedIds
    .map((id) => catalog.find((item) => item.id === id))
    .filter((item): item is IndicatorDefinition => Boolean(item))
  const available = periodOptions.filter((item) => !periods.includes(item))
  const atPeriodLimit = periods.length >= MAX_RESEARCH_PERIODS || available.length === 0
  const columnHint = atPeriodLimit
    ? `最多同时比较 ${MAX_RESEARCH_PERIODS} 个区间。`
    : periods.length === 1
      ? '再加一个区间即可横向比较同一指标的不同期限。'
      : `已比较 ${periods.length} 个区间，每个区间发起一次计算。`

  return <section className="min-w-0 overflow-hidden rounded-xl border border-slate-200 bg-white shadow-sm" aria-labelledby="research-indicators-title">
    <div className="flex flex-col gap-3 px-5 py-5 sm:flex-row sm:items-end sm:justify-between">
      <div className="min-w-0">
        <h2 id="research-indicators-title" className="text-lg font-semibold text-slate-900">研究指标</h2>
        <p className="mt-1 max-w-3xl text-sm leading-6 text-slate-600">
          用指标中心已保存的版本计算，每个指标在每个区间内给出一个数。{s('indicatorParameters.runtimeHint')}
        </p>
      </div>
      <Link to={studioHref} className="inline-flex min-h-10 shrink-0 items-center justify-center rounded-lg bg-accent-600 px-4 text-sm font-semibold text-white shadow-sm transition hover:bg-accent-700 focus:outline-none focus-visible:ring-2 focus-visible:ring-accent-500 focus-visible:ring-offset-2">
        在指标中心分析
      </Link>
    </div>

    {catalog.length === 0
      ? <div className="border-t border-slate-100 px-5 py-10 text-center">
        <p className="text-sm font-semibold text-slate-700">指标目录里还没有适用于本产品的标量指标</p>
        <p className="mx-auto mt-2 max-w-md text-sm leading-6 text-slate-600">先在指标中心新建，或复制一个内置指标另存为工作区版本，再回到本页选择。</p>
        {error && <p role="alert" className="mt-3 text-sm text-rose-700">{error}</p>}
      </div>
      : <>
        <div className="grid gap-4 border-t border-slate-100 px-5 py-4 lg:grid-cols-[minmax(0,17rem)_minmax(0,1fr)_minmax(0,15rem)]">
          <MetricSelector
            indicators={catalog}
            selectedIds={selectedIds}
            onChange={onSelectedIdsChange}
            maxSelected={MAX_RESEARCH_INDICATORS}
            label="选择研究指标"
          />
          <div className="min-w-0">
            <span id="research-period-columns" className="block text-xs font-medium text-slate-600">计算区间</span>
            <div role="group" aria-labelledby="research-period-columns" className="mt-1 flex flex-wrap items-center gap-2">
              {periods.map((period) => <span key={period} className="inline-flex min-h-10 items-center gap-1 rounded-lg border border-accent-200 bg-accent-50 py-1 pl-3 pr-1 text-sm font-medium text-accent-700">
                {indicatorPeriodLabel(period)}
                <button
                  type="button"
                  disabled={periods.length === 1}
                  onClick={() => onPeriodsChange(periods.filter((item) => item !== period))}
                  aria-label={`移除区间 ${indicatorPeriodLabel(period)}`}
                  title={periods.length === 1 ? '至少保留一个区间' : '移除这一列'}
                  className="inline-flex h-8 w-8 items-center justify-center rounded-lg text-base text-accent-700 transition hover:bg-accent-100 focus:outline-none focus-visible:ring-2 focus-visible:ring-accent-500 disabled:cursor-not-allowed disabled:opacity-50"
                >×</button>
              </span>)}
              <select
                aria-label="添加计算区间"
                value=""
                disabled={atPeriodLimit}
                onChange={(event) => { if (event.target.value) onPeriodsChange([...periods, event.target.value]) }}
                className="min-h-10 rounded-lg border border-dashed border-slate-300 bg-white px-3 text-sm text-slate-700 focus:border-accent-500 focus:outline-none focus-visible:ring-2 focus-visible:ring-accent-500 disabled:cursor-not-allowed disabled:opacity-50"
              >
                <option value="">＋ 添加区间</option>
                {available.map((item) => <option key={item} value={item}>{indicatorPeriodOptionLabel(item)}</option>)}
              </select>
            </div>
            <p className="mt-1 text-xs text-slate-600">{columnHint}</p>
          </div>
          <label className="text-xs font-medium text-slate-600">截止日（可选）
            <input type="date" value={asOf} onChange={(event) => onAsOfChange(event.target.value)} className="mt-1 block min-h-11 w-full rounded-lg border border-slate-200 bg-white px-3 text-sm text-slate-700 focus:border-accent-500 focus:outline-none focus-visible:ring-2 focus-visible:ring-accent-500" />
            <span className="mt-1 block text-xs font-normal leading-5 text-slate-600">{asOfHint}</span>
          </label>
        </div>

        {selected.length === 0 && <div className="border-t border-slate-100 px-5 py-10 text-center">
          <p className="text-sm font-semibold text-slate-700">还没有选择指标</p>
          <p className="mx-auto mt-2 max-w-md text-sm leading-6 text-slate-600">打开上方的“选择研究指标”，挑选本页要计算的指标。每个指标按上面的区间列逐列计算。</p>
        </div>}

        <div aria-live="polite">
          {loading && results.length > 0 && <p className="border-t border-slate-100 px-5 py-3 text-sm text-slate-600">正在基于真实数据批量计算…</p>}
          {error && selected.length > 0 && <p role="alert" className="border-t border-slate-100 px-5 py-3 text-sm text-rose-700">{error}</p>}
          {selected.length > 0 && <div className="border-t border-slate-100">
            <MetricPeriodTable
              indicators={selected}
              periods={periods}
              results={results}
              loading={loading}
              parametersFor={parametersFor}
              onParametersChange={onParametersChange}
              onRemove={(indicatorId) => onSelectedIdsChange(selectedIds.filter((id) => id !== indicatorId))}
              onDefinition={onDefinition}
            />
          </div>}
        </div>
      </>}
  </section>
}

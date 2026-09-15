import { Link } from 'react-router-dom'
import { useI18n } from '../../i18n/runtime'
import { MetricResultCard, MetricSelector } from './MetricDisplay'
import type {
  EvaluationResult,
  IndicatorDefinition,
} from '../../services/customIndicators'

export const MAX_RESEARCH_INDICATORS = 8

interface Props {
  indicators: IndicatorDefinition[]
  selectedIds: string[]
  onSelectedIdsChange: (ids: string[]) => void
  periodFor: (indicatorId: string) => string
  onPeriodChange: (indicatorId: string, period: string) => void
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
 * Scalar indicators only: one number each, one card each.
 *
 * Time-series indicators are drawn on the trend chart instead. Rendering the
 * same indicator in two different grammars on one page was the thing that made
 * this screen hard to read.
 */
export default function ResearchIndicatorPanel({
  indicators, selectedIds, onSelectedIdsChange, periodFor, onPeriodChange, periodOptions,
  parametersFor, onParametersChange, results, loading, error, asOf, onAsOfChange, asOfHint,
  onDefinition, studioHref,
}: Props) {
  const { s } = useI18n()
  const catalog = indicators.filter(item => (item.result_kind ?? 'scalar') === 'scalar')
  const selected = selectedIds
    .map((id) => catalog.find((item) => item.id === id))
    .filter((item): item is IndicatorDefinition => Boolean(item))

  return <section className="min-w-0 overflow-hidden rounded-xl border border-slate-200 bg-white shadow-sm" aria-labelledby="research-indicators-title">
    <div className="flex flex-col gap-3 px-5 py-5 sm:flex-row sm:items-end sm:justify-between">
      <div className="min-w-0">
        <h2 id="research-indicators-title" className="text-lg font-semibold text-slate-900">研究指标</h2>
        <p className="mt-1 max-w-3xl text-sm leading-6 text-slate-600">
          用指标中心已保存的版本计算，每个指标在所选区间内给出一个数。{s('indicatorParameters.runtimeHint')}
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
        <div className="grid gap-3 border-t border-slate-100 px-5 py-4 lg:grid-cols-[minmax(0,20rem)_minmax(0,18rem)]">
          <MetricSelector
            indicators={catalog}
            selectedIds={selectedIds}
            onChange={onSelectedIdsChange}
            maxSelected={MAX_RESEARCH_INDICATORS}
            label="选择研究指标"
          />
          <label className="text-xs font-medium text-slate-600">截止日（可选）
            <input type="date" value={asOf} onChange={(event) => onAsOfChange(event.target.value)} className="mt-1 block min-h-11 w-full rounded-lg border border-slate-200 bg-white px-3 text-sm text-slate-700 focus:border-accent-500 focus:outline-none focus-visible:ring-2 focus-visible:ring-accent-500" />
            <span className="mt-1 block text-xs font-normal leading-5 text-slate-600">{asOfHint}</span>
          </label>
        </div>

        {selected.length === 0 && <div className="border-t border-slate-100 px-5 py-10 text-center">
          <p className="text-sm font-semibold text-slate-700">还没有选择指标</p>
          <p className="mx-auto mt-2 max-w-md text-sm leading-6 text-slate-600">打开上方的“选择研究指标”，挑选本页要计算的指标。每张卡片可以单独调整计算区间和可变参数。</p>
        </div>}

        <div aria-live="polite">
          {loading && <p className="border-t border-slate-100 px-5 py-3 text-sm text-slate-600">正在基于真实数据批量计算…</p>}
          {error && selected.length > 0 && <p role="alert" className="border-t border-slate-100 px-5 py-3 text-sm text-rose-700">{error}</p>}
          {selected.length > 0 && <div className="grid gap-3 border-t border-slate-100 px-5 py-5 md:grid-cols-2 xl:grid-cols-3">
            {selected.map((indicator) => <MetricResultCard
              key={indicator.id}
              indicator={indicator}
              result={results.find((result) => result.indicator_id === indicator.id)}
              period={periodFor(indicator.id)}
              periodOptions={periodOptions}
              onPeriodChange={(period) => onPeriodChange(indicator.id, period)}
              parameters={parametersFor(indicator.id)}
              onParametersChange={(values) => onParametersChange(indicator.id, values)}
              onRemove={() => onSelectedIdsChange(selectedIds.filter((id) => id !== indicator.id))}
              onDefinition={() => onDefinition(indicator)}
            />)}
          </div>}
        </div>
      </>}
  </section>
}

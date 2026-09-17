import { Fragment, useMemo } from 'react'
import {
  MetricStatus,
  MetricUnavailableReason,
  MetricValue,
  resolveMetricPresentation,
} from './MetricDisplay'
import IndicatorParameterInputs from '../indicator-parameters/IndicatorParameterInputs'
import { indicatorPeriodLabel } from '../../utils/indicatorPeriods'
import type { EvaluationResult, IndicatorDefinition } from '../../services/customIndicators'

interface Props {
  indicators: IndicatorDefinition[]
  periods: string[]
  results: EvaluationResult[]
  loading: boolean
  parametersFor: (indicatorId: string) => Record<string, number>
  onParametersChange: (indicatorId: string, values: Record<string, number>) => void
  onRemove: (indicatorId: string) => void
  onDefinition: (indicator: IndicatorDefinition) => void
}

const DIRECTION_LABELS: Record<string, string> = {
  higher_better: '高优先',
  lower_better: '低优先',
  neutral: '仅展示',
}

const categoryOf = (indicator: IndicatorDefinition) =>
  indicator.presentation?.category_label ?? indicator.category_label ?? '未分类'

const windowKey = (result?: EvaluationResult) => result
  ? `${result.window.start_date ?? ''}|${result.window.end_date ?? ''}|${result.window.observation_count}`
  : ''

const parameterSchemaOf = (indicator: IndicatorDefinition, result?: EvaluationResult) =>
  (indicator.parameter_contract_version === '1.0' ? indicator.parameter_schema : null)
    ?? result?.presentation?.parameter_schema
    ?? indicator.presentation?.parameter_schema
    ?? []

/**
 * Rows are indicators, columns are periods.
 *
 * The window and the observation count belong to the column, not to the cell:
 * one product on one period gives every indicator the same window unless an
 * indicator's own input coverage cuts it short. So the header carries them
 * while the column agrees, and the cells carry them only once it does not.
 */
export default function MetricPeriodTable({
  indicators, periods, results, loading, parametersFor, onParametersChange, onRemove, onDefinition,
}: Props) {
  const byKey = useMemo(
    () => new Map(results.map((result) => [`${result.indicator_id}:${result.period}`, result])),
    [results],
  )
  const resultFor = (indicatorId: string, period: string) => byKey.get(`${indicatorId}:${period}`)

  /** One shared window per column, or null when the indicators disagree. */
  const sharedWindow = useMemo(() => new Map(periods.map((period) => {
    const present = indicators.map((item) => resultFor(item.id, period)).filter(Boolean) as EvaluationResult[]
    const keys = new Set(present.map(windowKey))
    return [period, keys.size === 1 ? present[0] : null] as const
  })), [byKey, indicators, periods])

  const columnCount = periods.length + 2
  const dataLatest = results.find((result) => result.window.data_latest_date)?.window.data_latest_date

  if (loading && results.length === 0) {
    return <div className="overflow-x-auto px-5 py-4">
      <table className="w-full min-w-[36rem] text-sm" aria-label="研究指标计算结果（正在计算）">
        <caption className="sr-only">正在基于真实数据批量计算</caption>
        <tbody>
          {indicators.map((indicator) => <tr key={indicator.id}>
            <th scope="row" className="border-b border-slate-100 py-4 pr-3 text-left align-top">
              <span className="block h-4 w-32 rounded bg-slate-100" />
            </th>
            {periods.map((period) => <td key={period} className="border-b border-slate-100 px-3 py-4">
              <span className="ml-auto block h-4 w-20 rounded bg-slate-100" />
            </td>)}
            <td className="border-b border-slate-100" />
          </tr>)}
        </tbody>
      </table>
    </div>
  }

  let previousCategory = ''
  return <div className="overflow-x-auto">
    <table className="w-full min-w-[36rem] text-sm" aria-label="研究指标计算结果">
      <thead>
        <tr>
          <th scope="col" className="sticky left-0 z-10 border-b border-slate-200 bg-white py-3 pl-5 pr-3 text-left align-bottom text-xs font-medium text-slate-600">指标</th>
          {periods.map((period) => {
            const shared = sharedWindow.get(period)
            return <th key={period} scope="col" className="border-b border-slate-200 px-3 py-3 text-right align-bottom">
              <span className="block text-sm font-semibold text-slate-900">{indicatorPeriodLabel(period)}</span>
              {shared
                ? <span className="mt-1 block text-xs font-normal text-slate-600">
                  {shared.window.start_date ?? '—'} 至 {shared.window.end_date ?? '—'}
                  <span className="block">{shared.window.observation_count} 个观察值</span>
                </span>
                : <span className="mt-1 block text-xs font-normal text-slate-600">口径随指标不同</span>}
            </th>
          })}
          <th scope="col" className="border-b border-slate-200 py-3 pl-3 pr-5 text-right align-bottom text-xs font-medium text-slate-600">操作</th>
        </tr>
      </thead>
      <tbody>
        {indicators.map((indicator) => {
          const category = categoryOf(indicator)
          const startsGroup = category !== previousCategory
          previousCategory = category
          const rowResults = periods.map((period) => resultFor(indicator.id, period))
          const anchor = rowResults.find(Boolean)
          const presentation = resolveMetricPresentation(anchor, indicator)
          const schema = parameterSchemaOf(indicator, anchor)
          const unavailable = periods
            .map((period, index) => ({ period, result: rowResults[index] }))
            .filter((item) => item.result && item.result.value === null)
          const cellPadding = startsGroup ? 'pt-5 pb-3' : 'py-3'
          return <Fragment key={indicator.id}>
            <tr className="group hover:bg-slate-50">
              <th scope="row" className={`sticky left-0 z-10 border-b border-slate-100 bg-white pl-5 pr-3 text-left align-top font-normal group-hover:bg-slate-50 ${cellPadding}`}>
                {startsGroup && <span className="mb-1 block text-xs font-medium text-slate-600">{category}</span>}
                <button type="button" onClick={() => onDefinition(indicator)} className="rounded text-left text-sm font-semibold text-slate-900 hover:text-accent-700 focus:outline-none focus-visible:ring-2 focus-visible:ring-accent-500">
                  {presentation.name}
                </button>
                <span className="mt-0.5 block text-xs text-slate-600">
                  {presentation.source === 'built_in' ? '内置' : '工作区'} v{presentation.revision}
                  {' · '}{DIRECTION_LABELS[presentation.direction] ?? DIRECTION_LABELS.neutral}
                </span>
              </th>
              {periods.map((period, index) => {
                const result = rowResults[index]
                const shared = sharedWindow.get(period)
                return <td key={period} className={`border-b border-slate-100 px-3 text-right align-top text-slate-900 ${cellPadding}`}>
                  {result
                    ? <MetricValue value={result.value} presentation={resolveMetricPresentation(result, indicator)} className="text-base font-semibold" />
                    : <span className="text-sm text-slate-600">{loading ? '计算中' : '—'}</span>}
                  {result && !shared && <span className="mt-0.5 block text-xs text-slate-600">{result.window.observation_count} 个观察值</span>}
                  {result && result.status !== 'ok' && <span className="mt-1 flex justify-end"><MetricStatus status={result.status} warnings={result.warnings} /></span>}
                </td>
              })}
              <td className={`border-b border-slate-100 pl-3 pr-5 text-right align-top ${cellPadding}`}>
                <button type="button" onClick={() => onRemove(indicator.id)} aria-label={`移除指标 ${presentation.name}`} title="仅从当前页面移除，不会删除指标定义" className="inline-flex min-h-10 items-center rounded-lg border border-slate-200 px-2.5 text-xs font-medium text-slate-600 transition hover:border-rose-300 hover:bg-rose-50 hover:text-rose-700 focus:outline-none focus-visible:ring-2 focus-visible:ring-accent-500">移除</button>
              </td>
            </tr>
            {schema.length > 0 && <tr>
              <td colSpan={columnCount} className="border-b border-slate-100 px-5 pb-4">
                <IndicatorParameterInputs
                  schema={schema}
                  values={parametersFor(indicator.id)}
                  onApply={(values) => onParametersChange(indicator.id, values)}
                  disabled={loading}
                  effective={anchor?.parameters ?? null}
                />
              </td>
            </tr>}
            {unavailable.length > 0 && <tr>
              <td colSpan={columnCount} className="border-b border-slate-100 px-5 pb-4">
                {periods.length > 1 && <p className="text-xs text-slate-600">
                  {unavailable.map((item) => indicatorPeriodLabel(item.period)).join('、')} 无法计算：
                </p>}
                {/* The blocking input is a property of the product, not of the
                    window, so the reason repeats identically per column. Print
                    it once and name the affected columns above it. */}
                <MetricUnavailableReason result={unavailable[0].result!} compact />
              </td>
            </tr>}
          </Fragment>
        })}
      </tbody>
      {dataLatest && <tfoot>
        <tr>
          <td colSpan={columnCount} className="px-5 py-3 text-xs text-slate-600">数据截至 {dataLatest}</td>
        </tr>
      </tfoot>}
    </table>
  </div>
}

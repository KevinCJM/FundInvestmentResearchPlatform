import { systemText, useI18n, i18n } from '../i18n/runtime'
import { useMemo } from 'react'
import ReactECharts from 'echarts-for-react'
import HorizontalMetricComparison, {
  DEFAULT_METRIC_TABLE_HEIGHT,
  PerformanceQuadrantChart,
} from './HorizontalMetricComparison'
import { buildAnnualMetricRows, type AnnualMetricsResult } from '../utils/performance'
import type { NativeNumericalExecutionAuditLanes } from '../utils/fixedNjitExecution'

export interface ClassFitMetric {
  name: string
  cumulative_return?: number | null
  annual_return?: number | null
  annual_vol?: number | null
  sharpe?: number | null
  var99?: number | null
  es99?: number | null
  max_drawdown?: number | null
  calmar?: number | null
}

export interface ClassFitConsistency {
  name: string
  mean_corr?: number
  pca_evr1?: number
  max_te?: number
}

export interface ClassFitResult {
  dates: string[]
  navs: Record<string, number[]>
  corr: Array<Array<number | null>>
  corr_labels: string[]
  metrics: ClassFitMetric[]
  consistency: ClassFitConsistency[]
  annual_metrics: AnnualMetricsResult
  execution: NativeNumericalExecutionAuditLanes
}

/** Shared NAV / correlation / metric view for a set of asset classes. */
export function buildClassMetricTable(result: Pick<ClassFitResult, 'metrics' | 'annual_metrics'> | null) {
  if (!result?.metrics || !Array.isArray(result.metrics)) {
    return { columns: [] as string[], rows: [] as { label: string; values: number[] }[] }
  }
  const columns = result.metrics.map((metric) => metric.name)
  const cumulativeValues = result.metrics.map((metric) => Number(metric.cumulative_return ?? NaN))
  const rows = [
    { label: systemText('preInvestment.classFitPanel.cumulativeReturn'), values: cumulativeValues },
    { label: systemText('preInvestment.classFitPanel.cumulativeReturn2'), values: cumulativeValues.map((value) => (Number.isFinite(value) ? value * 100 : NaN)) },
    { label: systemText('preInvestment.classFitPanel.annualReturn'), values: result.metrics.map((metric) => Number((metric.annual_return ?? NaN) * 100)) },
    { label: systemText('preInvestment.classFitPanel.annualVolatility'), values: result.metrics.map((metric) => Number((metric.annual_vol ?? NaN) * 100)) },
    { label: systemText('preInvestment.classFitPanel.sharpeRatio'), values: result.metrics.map((metric) => Number(metric.sharpe ?? NaN)) },
    { label: systemText('preInvestment.classFitPanel.99VarDaily'), values: result.metrics.map((metric) => Number((metric.var99 ?? NaN) * 100)) },
    { label: systemText('preInvestment.classFitPanel.99EsDaily'), values: result.metrics.map((metric) => Number((metric.es99 ?? NaN) * 100)) },
    { label: systemText('preInvestment.classFitPanel.maximumDrawdown'), values: result.metrics.map((metric) => Number((metric.max_drawdown ?? NaN) * 100)) },
    { label: systemText('preInvestment.classFitPanel.calmarRatio'), values: result.metrics.map((metric) => Number(metric.calmar ?? NaN)) },
  ]
  const annualRows = buildAnnualMetricRows(columns, result.annual_metrics)
  return { columns, rows: annualRows.length > 0 ? [...rows, ...annualRows] : rows }
}

export default function ClassFitPanel({ result }: { result: ClassFitResult }) {
  useI18n()
  const table = useMemo(() => buildClassMetricTable(result), [result, i18n.language])
  const navKeys = useMemo(() => Object.keys(result.navs), [result.navs])

  return (
    <div className="space-y-6">
      <div>
        <ReactECharts style={{ height: 360 }} option={{
          title: { text: systemText('preInvestment.classFitPanel.syntheticNavStartsAt1'), left: 0, top: 0, textStyle: { fontSize: 13, fontWeight: 600 } },
          tooltip: { trigger: 'axis', valueFormatter: (value: any) => Number(value).toFixed(2) },
          legend: { top: 0, right: 0 },
          grid: { left: 56, right: 16, top: 36, bottom: 86 },
          xAxis: { type: 'category', data: result.dates, axisLabel: { showMaxLabel: true, hideOverlap: true, margin: 12 } },
          yAxis: {
            type: 'value',
            scale: true,
            min: (value: any) => (Number.isFinite(value.min) && Number.isFinite(value.max)) ? value.min - (value.max - value.min) * 0.05 : 'dataMin',
            max: (value: any) => (Number.isFinite(value.min) && Number.isFinite(value.max)) ? value.max + (value.max - value.min) * 0.05 : 'dataMax',
            axisLabel: { formatter: (value: any) => Number(value).toFixed(2) },
          },
          dataZoom: [{ type: 'inside' }, { type: 'slider', bottom: 36, height: 18 }],
          series: navKeys.map((key) => ({
            name: key,
            type: 'line',
            smooth: false,
            symbol: 'none',
            lineStyle: { width: 2 },
            data: result.navs[key],
          })),
        }} />
      </div>

      <div>
        <h3 className="text-sm font-semibold mb-2">{systemText('preInvestment.classFitPanel.correlationMatrix')}</h3>
        <ReactECharts style={{ height: 320 }} option={(() => {
          const labels = result.corr_labels
          const data: any[] = []
          for (let row = 0; row < labels.length; row += 1) {
            for (let column = 0; column < labels.length; column += 1) {
              data.push([row, column, result.corr[row]?.[column] ?? null])
            }
          }
          return {
            tooltip: { position: 'top', formatter: (point: any) => `${labels[point.data[1]]} vs ${labels[point.data[0]]}: ${point.data[2] == null ? '—' : Number(point.data[2]).toFixed(2)}` },
            grid: { left: 80, right: 16, top: 16, bottom: 40 },
            xAxis: { type: 'category', data: labels, axisLabel: { rotate: 30 } },
            yAxis: { type: 'category', data: labels },
            visualMap: {
              min: -1, max: 1, show: false,
              inRange: { color: ['#ffffff', '#e0f2fe', '#bae6fd', '#7dd3fc', '#38bdf8', '#0ea5e9'] },
            },
            series: [{
              type: 'heatmap',
              data,
              label: { show: true, formatter: (point: any) => point.data[2] == null ? '—' : Number(point.data[2]).toFixed(2), color: '#111827' },
              emphasis: { itemStyle: { shadowBlur: 5, shadowColor: 'rgba(0,0,0,0.3)' } },
            }],
          }
        })()} />
      </div>

      <div>
        <h3 className="text-sm font-semibold mb-2">{systemText('preInvestment.classFitPanel.metricComparison')}</h3>
        <HorizontalMetricComparison columns={table.columns} rows={table.rows} height={DEFAULT_METRIC_TABLE_HEIGHT} />
        <div className="mt-4">
          <h4 className="text-sm font-semibold mb-2 text-slate-700">{systemText('preInvestment.classFitPanel.returnRiskQuadrantChart')}</h4>
          <PerformanceQuadrantChart
            columns={table.columns}
            rows={table.rows}
            defaultXAxis={systemText('preInvestment.classFitPanel.annualVolatility')}
            defaultYAxis={systemText('preInvestment.classFitPanel.cumulativeReturn2')}
          />
        </div>
      </div>
    </div>
  )
}

export function ClassConsistencyTable({ rows }: { rows: ClassFitConsistency[] }) {
  useI18n()
  if (!Array.isArray(rows) || rows.length === 0) return null
  return (
    <div className="overflow-auto">
      <table className="text-xs border" style={{ width: '100%', tableLayout: 'fixed' }}>
        <thead>
          <tr>
            <th scope="col" className="border px-2 py-2">{systemText('preInvestment.classFitPanel.assetClass')}</th>
            <th scope="col" className="border px-2 py-2">{systemText('preInvestment.classFitPanel.meanCorrelation')}</th>
            <th scope="col" className="border px-2 py-2">{systemText('preInvestment.classFitPanel.principalComponentExplainedVariance')}</th>
            <th scope="col" className="border px-2 py-2">{systemText('preInvestment.classFitPanel.maximumTrackingError')}</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => {
            const meanCorr = row.mean_corr as number
            const pcaEvr1 = row.pca_evr1 as number
            const maxTe = row.max_te as number
            return (
              <tr key={row.name}>
                <td className="border px-2 py-2">{row.name}</td>
                <td className="border px-2 py-2 text-right" style={Number.isFinite(meanCorr) && meanCorr < 0.6 ? { backgroundColor: '#fee2e2' } : {}}>{Number.isFinite(meanCorr) ? meanCorr.toFixed(3) : '-'}</td>
                <td className="border px-2 py-2 text-right" style={Number.isFinite(pcaEvr1) && pcaEvr1 < 0.8 ? { backgroundColor: '#fee2e2' } : {}}>{Number.isFinite(pcaEvr1) ? (pcaEvr1 * 100).toFixed(2) : '-'}</td>
                <td className="border px-2 py-2 text-right" style={Number.isFinite(maxTe) && maxTe * 100 > 5 ? { backgroundColor: '#fee2e2' } : {}}>{Number.isFinite(maxTe) ? (maxTe * 100).toFixed(2) : '-'}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

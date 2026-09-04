import { useMemo } from 'react'
import ReactECharts from 'echarts-for-react'
import HorizontalMetricComparison, {
  DEFAULT_METRIC_TABLE_HEIGHT,
  PerformanceQuadrantChart,
} from './HorizontalMetricComparison'
import { buildAnnualMetricRows, type AnnualMetricsResult } from '../utils/performance'
import type { FixedNjitExecutionAudit } from '../utils/fixedNjitExecution'

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
  execution: FixedNjitExecutionAudit
}

/** Shared NAV / correlation / metric view for a set of asset classes. */
export function buildClassMetricTable(result: Pick<ClassFitResult, 'metrics' | 'annual_metrics'> | null) {
  if (!result?.metrics || !Array.isArray(result.metrics)) {
    return { columns: [] as string[], rows: [] as { label: string; values: number[] }[] }
  }
  const columns = result.metrics.map((metric) => metric.name)
  const cumulativeValues = result.metrics.map((metric) => Number(metric.cumulative_return ?? NaN))
  const rows = [
    { label: '累计收益率', values: cumulativeValues },
    { label: '累计收益率(%)', values: cumulativeValues.map((value) => (Number.isFinite(value) ? value * 100 : NaN)) },
    { label: '年化收益率(%)', values: result.metrics.map((metric) => Number((metric.annual_return ?? NaN) * 100)) },
    { label: '年化波动率(%)', values: result.metrics.map((metric) => Number((metric.annual_vol ?? NaN) * 100)) },
    { label: '夏普比率', values: result.metrics.map((metric) => Number(metric.sharpe ?? NaN)) },
    { label: '99%VaR(日)(%)', values: result.metrics.map((metric) => Number((metric.var99 ?? NaN) * 100)) },
    { label: '99%ES(日)(%)', values: result.metrics.map((metric) => Number((metric.es99 ?? NaN) * 100)) },
    { label: '最大回撤(%)', values: result.metrics.map((metric) => Number((metric.max_drawdown ?? NaN) * 100)) },
    { label: '卡玛比率', values: result.metrics.map((metric) => Number(metric.calmar ?? NaN)) },
  ]
  const annualRows = buildAnnualMetricRows(columns, result.annual_metrics)
  return { columns, rows: annualRows.length > 0 ? [...rows, ...annualRows] : rows }
}

export default function ClassFitPanel({ result }: { result: ClassFitResult }) {
  const table = useMemo(() => buildClassMetricTable(result), [result])
  const navKeys = useMemo(() => Object.keys(result.navs), [result.navs])

  return (
    <div className="space-y-6">
      <div>
        <ReactECharts style={{ height: 360 }} option={{
          title: { text: '虚拟净值走势（起始=1）', left: 0, top: 0, textStyle: { fontSize: 13, fontWeight: 600 } },
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
        <h3 className="text-sm font-semibold mb-2">相关系数矩阵</h3>
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
        <h3 className="text-sm font-semibold mb-2">横向指标对比</h3>
        <HorizontalMetricComparison columns={table.columns} rows={table.rows} height={DEFAULT_METRIC_TABLE_HEIGHT} />
        <div className="mt-4">
          <h4 className="text-sm font-semibold mb-2 text-gray-700">收益风险象限图</h4>
          <PerformanceQuadrantChart
            columns={table.columns}
            rows={table.rows}
            defaultXAxis="年化波动率(%)"
            defaultYAxis="累计收益率(%)"
          />
        </div>
      </div>
    </div>
  )
}

export function ClassConsistencyTable({ rows }: { rows: ClassFitConsistency[] }) {
  if (!Array.isArray(rows) || rows.length === 0) return null
  return (
    <div className="overflow-auto">
      <table className="text-xs border" style={{ width: '100%', tableLayout: 'fixed' }}>
        <thead>
          <tr>
            <th className="border px-2 py-2">大类</th>
            <th className="border px-2 py-2">相关性均值</th>
            <th className="border px-2 py-2">主成分解释度(%)</th>
            <th className="border px-2 py-2">最大跟踪误差(%)</th>
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

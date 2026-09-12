import { useMemo } from 'react'
import ReactECharts from 'echarts-for-react'
import type { EChartsOption } from 'echarts'
import type { RegimeResultData, RegimeResultInterval } from './regimeResultAdapter'
import { regimeFeatureNumber } from './regimeResultAdapter'

export const UNKNOWN_STATE_COLOR = '#94a3b8'
export function intervalColor(result: RegimeResultData, interval: RegimeResultInterval) {
  return interval.state_id === 'unclassified'
    ? UNKNOWN_STATE_COLOR
    : result.overview.states.find(state => state.id === interval.state_id)?.color ?? UNKNOWN_STATE_COLOR
}

/** Unit-wide trading observations preserve single-day and final-day bands. */
function timelineTooltip(params: unknown, result: RegimeResultData): string {
  const entry = (Array.isArray(params) ? params : [params]).find(value => value && typeof value === 'object' && 'value' in value)
  if (!entry || typeof entry !== 'object' || !('value' in entry) || !Array.isArray(entry.value)) return ''
  const point = result.points[Math.round(Number(entry.value[0]))]
  if (!point) return ''
  const label = result.overview.states.find(state => state.id === point.state_id)?.label ?? '未分类'
  return [point.observation_date, result.overview.primary_series.label + '：' + (point.value == null ? '缺失' : point.value.toLocaleString('zh-CN', { maximumFractionDigits: 6 })), '情景：' + label].join('\n')
}

export function buildRegimeTimelineOption(result: RegimeResultData, selectedId: string | null = null): EChartsOption {
  const { overview, points, intervals } = result
  const dateLabel = (value: number) => points[Math.max(0, Math.min(points.length - 1, Math.round(value)))]?.observation_date ?? ''
  return {
    animation: false,
    aria: { enabled: true, description: '主对照走势及历史情景背景色带。完整日期与状态可在区间明细表中读取。' },
    grid: { left: 64, right: 24, top: 22, bottom: 76 },
    tooltip: {
      trigger: 'axis',
      renderMode: 'richText',
      formatter: params => timelineTooltip(params, result),
      axisPointer: { label: { formatter: params => dateLabel(Number(params.value)) } },
    },
    xAxis: {
      type: 'value', min: -0.5, max: Math.max(0.5, points.length - 0.5), minInterval: 1,
      splitNumber: 6, axisLabel: { formatter: dateLabel, hideOverlap: true }, splitLine: { show: false },
      axisPointer: { snap: true },
    },
    yAxis: { type: 'value', scale: true, name: overview.primary_series.unit ?? '' },
    dataZoom: [{ type: 'inside', filterMode: 'none' }, { type: 'slider', filterMode: 'none', height: 20, bottom: 14, labelFormatter: value => dateLabel(value) }],
    series: [{
      id: 'regime-primary-series', name: overview.primary_series.label, type: 'line',
      showSymbol: false, connectNulls: false, data: points.map((point, index) => [index, point.value]),
      lineStyle: { color: '#334155', width: 1.8 },
      markArea: {
        silent: false,
        label: { show: false },
        data: intervals.map(interval => [{
          name: interval.label,
          xAxis: interval.start_index - 0.5,
          itemStyle: {
            color: intervalColor(result, interval),
            opacity: interval.id === selectedId ? 0.32 : 0.18,
            borderColor: intervalColor(result, interval),
            borderWidth: interval.id === selectedId ? 2 : 0,
          },
        }, { xAxis: interval.end_index + 0.5 }]),
      },
    }],
  }
}

export function buildRegimeProbabilityOption(result: RegimeResultData): EChartsOption | null {
  if (!result.overview.capabilities.probabilities.available) return null
  const availableStates = result.overview.states.filter(state => result.points.some(point => point.probabilities[state.id] != null))
  if (!availableStates.length) return null
  return {
    animation: false, grid: { left: 52, right: 24, top: 40, bottom: 28 },
    legend: { data: availableStates.map(state => state.label) }, tooltip: { trigger: 'axis' },
    xAxis: { type: 'category', data: result.points.map(point => point.observation_date), axisLabel: { hideOverlap: true } },
    yAxis: { type: 'value', min: 0, max: 1, axisLabel: { formatter: value => (value * 100).toFixed(0) + '%' } },
    series: availableStates.map(state => ({ name: state.label, type: 'line', showSymbol: false, connectNulls: false, lineStyle: { color: state.color }, itemStyle: { color: state.color }, data: result.points.map(point => point.probabilities[state.id] ?? null) })),
  }
}

export function buildRegimeTrendOption(result: RegimeResultData, selectedId: string | null = null): EChartsOption | null {
  if (!result.points.some(point => regimeFeatureNumber(point, 'filtered_index') != null)) return null
  // The classification index can differ from the user's evaluation series.
  const diagnosticResult: RegimeResultData = {
    ...result,
    overview: { ...result.overview, primary_series: { ...result.overview.primary_series, label: '识别指数', unit: '点位' } },
    points: result.points.map(point => ({ ...point, value: regimeFeatureNumber(point, 'index_value') })),
  }
  const option = buildRegimeTimelineOption(diagnosticResult, selectedId)
  const series = Array.isArray(option.series) ? option.series : []
  return {
    ...option,
    aria: { enabled: true, description: '用于识别的指数与单向滤波线，背景表示最终市场状态。' },
    grid: { left: 64, right: 24, top: 40, bottom: 76 },
    legend: { data: ['识别指数', '趋势滤波线'] },
    tooltip: {
      trigger: 'axis', renderMode: 'richText',
      formatter: params => {
        const entries = Array.isArray(params) ? params : [params]
        const index = entries[0]?.dataIndex
        const filtered = typeof index === 'number' ? regimeFeatureNumber(result.points[index], 'filtered_index') : null
        return timelineTooltip(params, diagnosticResult) + '\n趋势滤波线：' + (filtered?.toLocaleString('zh-CN', { maximumFractionDigits: 4 }) ?? '预热或缺失')
      },
    },
    series: [...series, {
      name: '趋势滤波线', type: 'line', showSymbol: false, connectNulls: false,
      lineStyle: { color: '#d97706', width: 2 },
      data: result.points.map((point, index) => [index, regimeFeatureNumber(point, 'filtered_index')]),
    }],
  }
}

function clickedIntervalIndex(event: unknown): number | null {
  if (!event || typeof event !== 'object' || !('componentType' in event) || event.componentType !== 'markArea' || !('dataIndex' in event) || typeof event.dataIndex !== 'number') return null
  return event.dataIndex
}

export function buildRegimePeakOption(result: RegimeResultData, selectedId: string | null = null): EChartsOption | null {
  if (!result.points.some(point => regimeFeatureNumber(point, 'boundary_line') != null || regimeFeatureNumber(point, 'pivot_price') != null)) return null
  const source: RegimeResultData = {
    ...result,
    overview: { ...result.overview, primary_series: { ...result.overview.primary_series, label: '定界指数', unit: '点位' } },
    points: result.points.map(point => ({ ...point, value: regimeFeatureNumber(point, 'index_value') })),
  }
  const option = buildRegimeTimelineOption(source, selectedId)
  return {
    ...option, grid: { left: 64, right: 24, top: 42, bottom: 76 },
    legend: { data: ['定界指数', ...(result.points.some(point => regimeFeatureNumber(point, 'boundary_line') != null) ? ['峰谷连线'] : []), '保留峰值', '保留谷值'] },
    series: [
      ...(Array.isArray(option.series) ? option.series : []),
      { name: '峰谷连线', type: 'line', connectNulls: false, showSymbol: false,
        lineStyle: { color: '#7c3aed', type: 'dashed' },
        data: result.points.map((point, index) => [index, regimeFeatureNumber(point, 'boundary_line')]) },
      ...([1, -1] as const).map(kind => ({
        name: kind === 1 ? '保留峰值' : '保留谷值', type: 'scatter' as const, symbolSize: 9,
        itemStyle: { color: kind === 1 ? '#d97706' : '#2563eb' },
        data: result.points.flatMap((point, index) => regimeFeatureNumber(point, 'pivot') === kind ? [[index, regimeFeatureNumber(point, 'index_value')]] : []),
      })),
    ],
  }
}

export default function RegimeTimelineChart({ result, selectedId, onSelect }: { result: RegimeResultData; selectedId: string | null; onSelect: (interval: RegimeResultInterval) => void }) {
  const option = useMemo(() => buildRegimeTimelineOption(result, selectedId), [result, selectedId])
  const probabilityOption = useMemo(() => buildRegimeProbabilityOption(result), [result])
  const trendOption = useMemo(() => buildRegimeTrendOption(result, selectedId), [result, selectedId])
  const peakOption = useMemo(() => buildRegimePeakOption(result, selectedId), [result, selectedId])
  return <div className="min-w-0 space-y-3">
    <div className="flex flex-wrap gap-x-5 gap-y-2 px-2 text-xs" aria-label="情景图例">
      {result.overview.states.map(state => <span key={state.id} className="inline-flex items-center gap-2"><span aria-hidden="true" className="h-3 w-3 rounded-lg" style={{ backgroundColor: state.color }} />{state.label}</span>)}
      {result.overview.summary.unknown > 0 ? <span className="inline-flex items-center gap-2"><span aria-hidden="true" className="h-3 w-3 rounded-lg" style={{ backgroundColor: UNKNOWN_STATE_COLOR }} />未分类</span> : null}
    </div>
    <ReactECharts key={result.overview.run_id} option={option} lazyUpdate style={{ height: 400, width: '100%' }} onEvents={{ click: (event: unknown) => { const index = clickedIntervalIndex(event); if (index != null && result.intervals[index]) onSelect(result.intervals[index]) } }} aria-label="历史情景走势与背景区间" />
    <p className="px-2 text-xs text-slate-600">曲线：{result.overview.primary_series.label}。色带按真实观测日绘制，拖动底部滑块可缩放；点击色带查看该段依据。缺失数值保留断点。</p>
    {trendOption ? <section aria-label="识别指数与趋势滤波线">
      <h3 className="px-2 text-sm font-semibold">识别指数与趋势滤波线</h3>
      <ReactECharts option={trendOption} notMerge lazyUpdate style={{ height: 300 }} />
      <p className="px-2 text-xs text-slate-600">交叉只产生候选，达到幅度、趋势与连续确认门槛后才切换状态；震荡另需平坦趋势和低方向效率。</p>
    </section> : null}
    {peakOption ? <section aria-label="峰谷定界图">
      <h3 className="px-2 text-sm font-semibold">峰谷定界图 · 事后识别</h3>
      <ReactECharts option={peakOption} notMerge lazyUpdate style={{ height: 300 }} />
      <p className="px-2 text-xs text-slate-600">圆点为保留的高低点。拐点需要后续数据确认；首尾未完成区间保留未分类，不能作为当时的交易信号。</p>
    </section> : null}
    {probabilityOption ? <section aria-label="状态概率"><p className="px-2 text-xs text-slate-600">状态概率描述模型对当前分类的判断，不等于投资获利概率。</p><ReactECharts option={probabilityOption} notMerge lazyUpdate style={{ height: 220 }} /></section> : null}
  </div>
}

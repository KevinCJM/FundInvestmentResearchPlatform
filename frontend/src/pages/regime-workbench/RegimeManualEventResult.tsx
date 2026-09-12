import { useMemo } from 'react'
import ReactECharts from 'echarts-for-react'
import type { EChartsOption } from 'echarts'
import type { RegimeManualEventResult, RegimeResultData } from './regimeResultAdapter'

function percent(count: number, total: number) { return total ? `${(count / total * 100).toFixed(1)}%` : '—' }

export function buildManualEventMarketOption(result: RegimeResultData): EChartsOption {
  const points = result.points
  const dateLabel = (value: number) => points[Math.max(0, Math.min(points.length - 1, Math.round(value)))]?.observation_date ?? ''
  return {
    animation: false,
    aria: { enabled: true, description: '观察序列与用户定义的可重叠历史事件区间。事件明细同时提供为文本表格。' },
    grid: { left: 64, right: 24, top: 28, bottom: 76 },
    tooltip: {
      trigger: 'axis',
      formatter: params => {
        const entry = Array.isArray(params) ? params[0] : params
        const index = typeof entry?.dataIndex === 'number' ? entry.dataIndex : -1
        const point = points[index]
        if (!point) return ''
        const active = result.overview.manual_events.filter(event => event.first_observation_index != null && event.last_observation_index != null && index >= event.first_observation_index && index <= event.last_observation_index)
        return [point.observation_date, `${result.overview.primary_series.label}：${point.value == null ? '缺失' : point.value.toLocaleString('zh-CN', { maximumFractionDigits: 6 })}`, `事件：${active.length ? active.map(item => item.label).join('、') : '无'}`].join('\n')
      },
      axisPointer: { label: { formatter: params => dateLabel(Number(params.value)) } },
    },
    xAxis: { type: 'value', min: -0.5, max: Math.max(0.5, points.length - 0.5), minInterval: 1, splitNumber: 6, axisLabel: { formatter: dateLabel, hideOverlap: true }, splitLine: { show: false } },
    yAxis: { type: 'value', scale: true, name: result.overview.primary_series.unit ?? '' },
    dataZoom: [{ type: 'inside', filterMode: 'none' }, { type: 'slider', filterMode: 'none', height: 20, bottom: 14, labelFormatter: value => dateLabel(Number(value)) }],
    series: [{
      name: result.overview.primary_series.label, type: 'line', showSymbol: false, connectNulls: false,
      data: points.map((point, index) => [index, point.value]), lineStyle: { color: '#334155', width: 1.8 },
      markArea: {
        silent: true, label: { show: false },
        data: result.overview.manual_events.flatMap(event => event.first_observation_index == null || event.last_observation_index == null ? [] : [[{
          name: event.label, xAxis: event.first_observation_index - 0.5, itemStyle: { color: event.color, opacity: 0.13 },
        }, { xAxis: event.last_observation_index + 0.5 }]]),
      },
    }],
  }
}

export function buildManualEventLaneOption(result: RegimeResultData): EChartsOption {
  const points = result.points
  const events = result.overview.manual_events
  const dateLabel = (value: number) => points[Math.max(0, Math.min(points.length - 1, Math.round(value)))]?.observation_date ?? ''
  return {
    animation: false,
    aria: { enabled: true, description: '每个历史事件独占一条轨道，因此重叠事件不会相互覆盖。完整事件日期可在下方表格读取。' },
    grid: { left: 150, right: 24, top: 18, bottom: 58, containLabel: false },
    tooltip: {
      trigger: 'item',
      formatter: params => {
        const index = typeof params === 'object' && params && 'dataIndex' in params && typeof params.dataIndex === 'number' ? params.dataIndex : -1
        const event = events[index]
        return event ? `${event.label}\n定义：${event.start_date} 至 ${event.end_date}\n实际覆盖：${event.covered_observations} 个观测` : ''
      },
    },
    xAxis: { type: 'value', min: 0, max: Math.max(1, points.length - 1), axisLabel: { formatter: dateLabel, hideOverlap: true }, splitLine: { show: false } },
    yAxis: { type: 'category', inverse: true, data: events.map(event => event.label), axisLabel: { width: 130, overflow: 'truncate' } },
    dataZoom: [{ type: 'inside', xAxisIndex: 0, filterMode: 'none' }, { type: 'slider', xAxisIndex: 0, filterMode: 'none', height: 18, bottom: 12, labelFormatter: value => dateLabel(Number(value)) }],
    series: [
      { name: '起始偏移', type: 'bar', stack: 'event-range', silent: true, itemStyle: { color: 'transparent' }, emphasis: { disabled: true }, data: events.map(event => event.first_observation_index ?? 0) },
      { name: '事件区间', type: 'bar', stack: 'event-range', barMaxWidth: 22, data: events.map(event => ({ value: event.first_observation_index == null || event.last_observation_index == null ? 0 : event.last_observation_index - event.first_observation_index + 1, itemStyle: { color: event.color } })) },
    ],
  }
}

function EventTable({ events }: { events: RegimeManualEventResult[] }) {
  return <div className="max-h-[420px] overflow-auto rounded-xl border border-slate-200"><table className="min-w-full text-left text-xs" aria-label="人工历史事件明细"><thead className="sticky top-0 bg-slate-50 text-slate-600"><tr>{['事件', '定义开始', '定义结束', '实际覆盖', '说明'].map(label => <th scope="col" key={label} className="whitespace-nowrap px-3 py-2">{label}</th>)}</tr></thead><tbody>{events.map(event => <tr key={event.id} className="border-t border-slate-100"><td className="min-w-40 px-3 py-2"><span className="mr-2 inline-block h-2.5 w-2.5 rounded-lg" style={{ backgroundColor: event.color }} />{event.label}</td><td className="whitespace-nowrap px-3 py-2">{event.start_date}</td><td className="whitespace-nowrap px-3 py-2">{event.end_date}</td><td className="whitespace-nowrap px-3 py-2">{event.covered_observations ? `${event.covered_observations} 个观测 · ${event.first_observation_date} 至 ${event.last_observation_date}` : '当前数据范围外'}</td><td className="max-w-xs px-3 py-2 text-slate-600">{event.description || '—'}</td></tr>)}</tbody></table></div>
}

export default function RegimeManualEventResult({ result, stale = false }: { result: RegimeResultData; stale?: boolean }) {
  const { overview } = result
  const summary = overview.manual_event_summary
  const marketOption = useMemo(() => buildManualEventMarketOption(result), [result])
  const laneOption = useMemo(() => buildManualEventLaneOption(result), [result])
  const laneHeight = Math.max(220, overview.manual_events.length * 38 + 100)
  return <section aria-label="人工历史事件结果" className="min-w-0 space-y-5 p-3 sm:p-5" data-run-id={overview.run_id}>
    <header className="flex flex-wrap items-start justify-between gap-3"><div><h2 className="text-lg font-bold text-slate-950">人工历史事件结果</h2><p className="mt-1 text-xs text-slate-600">{overview.date_range.start ?? '无样本'} 至 {overview.date_range.end ?? '无样本'} · 共 {overview.summary.total} 个观察序列观测{overview.as_of ? ` · 截至 ${overview.as_of}` : ''}</p></div><span className="rounded-full bg-amber-50 px-3 py-1 text-xs font-bold text-amber-800">事后识别 · 多标签事件</span></header>
    {stale ? <p role="status" className="rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-900">此结果使用上一次运行时冻结的事件定义；当前修改需重新运行后才会生效。</p> : null}
    <p className="rounded-xl border border-accent-200 bg-accent-50 px-3 py-2 text-xs leading-5 text-accent-900">事件由人类事后定义，<strong>允许重叠</strong>。每个事件独立保留，不做优先级去重，也不能直接作为当时可交易信号。</p>
    <div className="grid grid-cols-2 gap-3 lg:grid-cols-4" aria-label="人工事件摘要">
      <Summary label="事件数量" value={String(summary.event_count)} detail="事件可互相重叠" />
      <Summary label="任一事件覆盖" value={String(summary.covered_observations)} detail={`${percent(summary.covered_observations, overview.summary.total)} 的观察序列`} />
      <Summary label="重叠观测" value={String(summary.overlap_observations)} detail="同时属于至少两个事件" />
      <Summary label="最大同时事件" value={String(summary.max_concurrent_events)} detail="同一观测点的事件数量" />
    </div>
    {result.points.length ? <section aria-label="观察序列与人工事件背景"><h3 className="mb-2 text-sm font-bold text-slate-900">观察序列</h3><ReactECharts option={marketOption} notMerge lazyUpdate style={{ height: 400, width: '100%' }} /></section> : null}
    {overview.manual_events.length ? <section aria-label="人工事件轨道"><div className="mb-2"><h3 className="text-sm font-bold text-slate-900">事件轨道</h3><p className="mt-1 text-xs text-slate-600">每个事件单独一行；同一日期出现多条色带即表示事件重叠。</p></div><div className="max-h-[720px] overflow-y-auto"><ReactECharts option={laneOption} notMerge lazyUpdate style={{ height: laneHeight, width: '100%' }} /></div></section> : <p className="rounded-xl border border-dashed border-slate-300 p-6 text-center text-sm text-slate-600">本次运行没有定义历史事件。</p>}
    <section><h3 className="mb-2 text-sm font-bold text-slate-900">事件明细</h3><EventTable events={overview.manual_events} /></section>
  </section>
}

function Summary({ label, value, detail }: { label: string; value: string; detail: string }) {
  return <div className="rounded-xl border border-slate-200 bg-white p-3"><p className="text-xs text-slate-600">{label}</p><p className="mt-1 text-xl font-bold text-slate-950">{value}</p><p className="mt-1 text-xs text-slate-600">{detail}</p></div>
}

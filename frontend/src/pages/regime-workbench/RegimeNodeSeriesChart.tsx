import { useEffect, useMemo, useRef, useState } from 'react'
import ReactECharts from 'echarts-for-react'
import type { ECharts, EChartsOption, LineSeriesOption } from 'echarts'
import { getAllRegimePreviewSeries, getRegimeNormalizedChart, type RegimeNormalizedChart, type RegimeSeriesPage, type RegimeUpstreamOutput } from '../../services/regimeGraph'

function chartData(page: RegimeSeriesPage) {
  const candidates = ['value', 'confidence', ...Object.keys(page.items[0] || {}).filter(key => !['date', 'observation_date', 'index', 'recognition_index', 'effective_index', 'reason_code'].includes(key))]
  const key = candidates.find(name => page.items.some(row => typeof row[name] === 'number')) || (page.value_type === 'series<float64>' ? 'value' : '')
  if (!key) return null
  return { key, dates: page.items.map(row => row.date), values: page.items.map(row => typeof row[key] === 'number' && Number.isFinite(row[key]) ? row[key] as number : null) }
}

const numberText = (value: number | null | undefined) => value == null ? '—' : new Intl.NumberFormat('zh-CN', { maximumFractionDigits: 4 }).format(value)
const outputKey = (item: { node_id?: string; port?: string }) => JSON.stringify([item.node_id, item.port || 'value'])
const colors = ['#4f46e5', '#059669', '#d97706', '#e11d48', '#0284c7', '#9333ea', '#475569']
type Placement = 'left' | 'right' | 'subplot'
type Load = { page?: RegimeSeriesPage; error?: string }
type Layer = { id: string; label: string; page?: RegimeSeriesPage; placement: Placement; color: string; error?: string }
type Normalization = { date: string; result?: RegimeNormalizedChart; error?: string }

export default function RegimeNodeSeriesChart({ page }: { page: RegimeSeriesPage }) {
  const main = useMemo(() => chartData(page), [page])
  const upstream = page.upstream_outputs || []
  const [selected, setSelected] = useState<Record<string, Placement>>({})
  const [loads, setLoads] = useState<Record<string, Load>>({})
  const [retry, setRetry] = useState(0)
  const cache = useRef<Record<string, RegimeSeriesPage>>({})
  const [search, setSearch] = useState('')
  const [enabled, setEnabled] = useState(false)
  const [range, setRange] = useState({ start: page.items[0]?.date || '', end: page.items[page.items.length - 1]?.date || '' })
  const [results, setResults] = useState<Record<string, Normalization>>({})
  const selectedKey = JSON.stringify(Object.keys(selected).sort())
  const eligible = page.value_type === 'series<float64>' && !!page.node_id && main?.key === 'value'
  const normalized = enabled && eligible

  useEffect(() => {
    const controller = new AbortController()
    const keys: string[] = JSON.parse(selectedKey)
    for (const key of keys) {
      if (cache.current[key]) continue
      const target = page.upstream_outputs?.find(item => outputKey(item) === key)
      if (!target?.plottable) continue
      setLoads(previous => ({ ...previous, [key]: {} }))
      void getAllRegimePreviewSeries(page.run_id, target.node_id, target.port, controller.signal).then(next => {
        if (controller.signal.aborted) return
        if (next.value_type !== target.value_type) throw new Error('上游数据类型与本次预览不一致，请重新预览。')
        cache.current[key] = next
        setLoads(previous => ({ ...previous, [key]: { page: next } }))
      }).catch(reason => {
        if (!controller.signal.aborted) setLoads(previous => ({ ...previous, [key]: { error: reason instanceof Error ? reason.message : '读取上游数据失败。' } }))
      })
    }
    return () => controller.abort()
  }, [page, selectedKey, retry])

  const layers = useMemo<Layer[]>(() => [
    { id: outputKey(page), label: page.node_label || page.node_id || '当前节点', page, placement: 'left', color: colors[0] },
    ...(page.upstream_outputs || []).filter(item => selected[outputKey(item)]).map((item, index) => ({
      id: outputKey(item), label: `${item.node_label} · ${item.port_label}`, placement: selected[outputKey(item)], color: colors[(index % (colors.length - 1)) + 1], ...loads[outputKey(item)],
    })),
  ], [page, selected, loads])
  // Join by observation date, never by array position; preserve sparse calendars and gaps.
  const plotted = useMemo(() => layers.map(layer => ({ ...layer, data: layer.page ? chartData(layer.page) : null })), [layers])
  const dates = useMemo(() => {
    const first = main?.dates[0], last = main?.dates[main.dates.length - 1]
    return Array.from(new Set(plotted.flatMap(layer => layer.data?.dates || []).filter(date => first && last && date >= first && date <= last))).sort()
  }, [plotted, main])
  const startIndex = Math.max(0, dates.indexOf(range.start))
  const endIndex = Math.max(startIndex, dates.indexOf(range.end))
  // Moving a curve between axes must not reload or normalize it again.
  const normalizationSources = useMemo(() => plotted.filter(layer => layer.page && layer.data).map(layer => ({ id: layer.id, page: layer.page!, data: layer.data! })), [page, loads, selectedKey])

  useEffect(() => {
    setResults({})
    if (!normalized) return
    const controller = new AbortController()
    const timer = window.setTimeout(() => {
      for (const layer of normalizationSources) {
        if (layer.page.value_type !== 'series<float64>' || !layer.page.node_id) continue
        const index = layer.data.dates.indexOf(range.start)
        const save = (value: Omit<Normalization, 'date'>) => {
          if (!controller.signal.aborted) setResults(previous => ({ ...previous, [layer.id]: { date: range.start, ...value } }))
        }
        if (index < 0) { save({ error: '区间首日没有观测，请调整起点或关闭归一化。' }); continue }
        const base = layer.page.offset + index
        void getRegimeNormalizedChart(page.run_id, layer.page.node_id, layer.page.port || 'value', base, controller.signal).then(next => {
          if (next.run_id !== page.run_id || next.node_id !== layer.page.node_id || next.port !== (layer.page.port || 'value') || next.base_index !== base || next.base_date !== range.start || next.values.length !== layer.page.total || next.change_pct.length !== layer.page.total) throw new Error('归一化结果与当前预览不一致，请重新预览。')
          save({ result: next })
        }).catch(reason => save({ error: reason instanceof Error ? reason.message : '归一化失败，请重试。' }))
      }
    }, 150)
    return () => { window.clearTimeout(timer); controller.abort() }
  }, [normalized, range.start, normalizationSources, page.run_id])

  const curves = useMemo(() => plotted.map(layer => {
    const rebased = normalized && layer.page?.value_type === 'series<float64>'
    const response = results[layer.id]?.date === range.start ? results[layer.id] : undefined
    const indices = new Map(layer.data?.dates.map((date, index) => [date, index]))
    const values = dates.map(date => {
      const index = indices.get(date)
      if (index === undefined) return null
      return rebased ? response?.result?.values[(layer.page?.offset || 0) + index] ?? null : layer.data?.values[index] ?? null
    })
    return { ...layer, rebased, response, indices, values }
  }), [plotted, normalized, results, range.start, dates])
  const subplots = curves.filter(layer => layer.placement === 'subplot')
  const onEvents = useMemo(() => ({ datazoom: (_event: unknown, instance: ECharts) => {
    const zoom = (instance.getOption().dataZoom as Array<{ startValue?: number; endValue?: number }> | undefined)?.[0]
    if (typeof zoom?.startValue !== 'number' || typeof zoom.endValue !== 'number') return
    const start = dates[Math.max(0, Math.min(dates.length - 1, Math.round(zoom.startValue)))]
    const end = dates[Math.max(0, Math.min(dates.length - 1, Math.round(zoom.endValue)))]
    setRange(previous => previous.start === start && previous.end === end ? previous : { start, end })
  } }), [dates])

  const option = useMemo<EChartsOption | null>(() => {
    if (!main) return null
    const panels = [curves[0], ...subplots]
    const axisIndices = panels.map((_, index) => index)
    const rightAxis = curves.some(layer => layer.placement === 'right')
    const right = rightAxis ? 58 : 20
    const series: LineSeriesOption[] = curves.map(layer => {
      const panel = layer.placement === 'subplot' ? subplots.findIndex(item => item.id === layer.id) + 1 : 0
      return { id: layer.id, name: layer.label, type: 'line', showSymbol: false, connectNulls: false, xAxisIndex: panel, yAxisIndex: layer.placement === 'right' ? panels.length : panel, data: dates.every(date => layer.indices.has(date)) ? layer.values : dates.flatMap((date, index) => layer.indices.has(date) ? [[date, layer.values[index]]] : []), lineStyle: { color: layer.color, width: 2 }, itemStyle: { color: layer.color } }
    })
    return {
      animation: false,
      title: panels.map((layer, index) => ({ text: `${index ? '' : '主图 · '}${layer.label}${layer.rebased ? '（首日＝1）' : ''}`, top: index * 200, left: 58, textStyle: { color: layer.color, fontSize: 11, width: 210, overflow: 'truncate' } })),
      tooltip: { trigger: 'axis', renderMode: 'richText', confine: true, formatter: params => {
        const item = Array.isArray(params) ? params[0] : params
        if (!item) return ''
        const axisDate = (item as { axisValue?: string }).axisValue
        const index = axisDate ? dates.indexOf(axisDate) : dates.indexOf(item.name)
        return [dates[index], ...curves.map(layer => `${layer.label}${layer.rebased ? '（首日＝1）' : ''}：${numberText(layer.values[index])}`)].join('\n')
      } },
      axisPointer: { link: [{ xAxisIndex: 'all' }] },
      grid: panels.map((_, index) => ({ left: 58, right, top: 26 + index * 200, height: subplots.length ? 166 : 186 })),
      xAxis: panels.map((_, index) => ({ type: 'category', gridIndex: index, data: dates, axisLabel: { hideOverlap: true, show: index === panels.length - 1 }, axisPointer: { show: true, label: { show: index === panels.length - 1 } } })),
      yAxis: [
        ...panels.map((layer, index) => ({ type: 'value' as const, gridIndex: index, scale: true, axisLabel: { hideOverlap: true }, name: '' })),
        ...(rightAxis ? [{ type: 'value' as const, gridIndex: 0, position: 'right' as const, scale: true, splitLine: { show: false }, name: '右轴' }] : []),
      ],
      dataZoom: [
        { type: 'inside', xAxisIndex: axisIndices, startValue: startIndex, endValue: endIndex, filterMode: 'filter' },
        { type: 'slider', xAxisIndex: axisIndices, left: 58, right, startValue: startIndex, endValue: endIndex, height: 17, bottom: 7, realtime: false, filterMode: 'filter' },
      ],
      series,
    }
  }, [main, curves, subplots, dates, startIndex, endIndex])

  if (!option) return null
  const mainCurve = curves[0]
  const mainResult = mainCurve.response?.result
  const mainEnd = mainCurve.indices.get(range.end)
  const options = upstream.filter(item => `${item.node_label} ${item.port_label} ${item.node_id}`.toLowerCase().includes(search.toLowerCase()))
  const select = (item: RegimeUpstreamOutput, checked: boolean) => setSelected(previous => {
    const next = { ...previous }, key = outputKey(item)
    if (checked) next[key] = 'subplot'; else delete next[key]
    return next
  })
  return <section aria-label="节点走势图" className="space-y-3">
    {upstream.length > 0 && <details className="rounded-xl border border-slate-200 p-3">
      <summary className="cursor-pointer text-xs font-semibold text-slate-700">叠加上游数据{Object.keys(selected).length ? `（已选 ${Object.keys(selected).length}）` : ''}</summary>
      <p className="my-2 text-xs text-slate-500">勾选后默认显示在子图。相同量纲可选同图同轴，不同量纲可选右轴或子图。</p>
      {upstream.length > 6 && <input aria-label="搜索上游节点" placeholder="搜索节点或输出名称" value={search} onChange={event => setSearch(event.target.value)} className="mb-2 min-h-9 w-full rounded-lg border px-2 text-xs" />}
      <div className="max-h-64 space-y-2 overflow-auto">
        {options.map(item => <label key={outputKey(item)} className="flex min-h-9 items-start gap-2 rounded-lg bg-slate-50 p-2 text-xs text-slate-700">
          <input type="checkbox" className="mt-0.5" checked={!!selected[outputKey(item)]} disabled={!item.plottable} onChange={event => select(item, event.target.checked)} />
          <span className="min-w-0 break-words">{item.node_label} · {item.port_label}<span className="ml-2 text-slate-400">{item.distance === 1 ? '直接上游' : '更上游'}</span>{!item.plottable && <span className="block text-slate-500">{item.unavailable_reason}</span>}</span>
        </label>)}
        {!options.length && <p className="text-xs text-slate-500">没有匹配的上游节点。</p>}
      </div>
    </details>}
    {curves.length > 1 && <div className="space-y-2" aria-label="曲线展示设置">
      <p className="break-words text-xs font-semibold" style={{ color: mainCurve.color }}>主图：{mainCurve.label}</p>
      {curves.slice(1).map(layer => <div key={layer.id} className="flex flex-wrap items-center gap-2 rounded-lg bg-slate-50 p-2 text-xs">
        <span className="min-w-0 flex-1 basis-32 break-words font-semibold" style={{ color: layer.color }}>{layer.label}</span>
        <select aria-label={`${layer.label}展示位置`} value={layer.placement} onChange={event => setSelected(previous => ({ ...previous, [layer.id]: event.target.value as Placement }))} className="min-h-9 max-w-full rounded-lg border border-slate-300 bg-white px-2">
          <option value="left">同图同轴</option><option value="right">同图右轴</option><option value="subplot">独立子图</option>
        </select>
        <button type="button" aria-label={`移除${layer.label}`} onClick={() => setSelected(previous => { const next = { ...previous }; delete next[layer.id]; return next })} className="min-h-9 px-2 text-slate-500">移除</button>
        {!layer.page && <p className={`basis-full ${layer.error ? 'text-rose-700' : 'text-slate-500'}`}>{layer.error || '正在读取上游数据…'}{layer.error && <button type="button" onClick={() => setRetry(value => value + 1)} className="ml-2 underline">重试</button>}</p>}
      </div>)}
      <p className="text-xs text-slate-500">共用时间轴；按观测日期对齐，缺失值留空，不同频率保留各自观测点。展示范围以主节点为准。</p>
    </div>}
    <div className="flex flex-wrap items-center justify-between gap-2">
      <button type="button" role="switch" aria-checked={normalized} disabled={!eligible} onClick={() => setEnabled(value => !value)} className={`min-h-9 rounded-lg border px-3 text-xs font-semibold disabled:opacity-40 ${normalized ? 'border-violet-600 bg-violet-600 text-white' : 'border-slate-300 bg-white text-slate-700'}`}>区间归一化（首日＝1）</button>
      <span className="text-xs text-slate-500">{eligible ? '各数值曲线以同一首日归一化；表格保留原值。' : '此输出不适用区间归一化。'}</span>
    </div>
    {normalized && <div role="status" className="space-y-1 text-xs text-slate-600">
      <p>{mainCurve.response?.error || (mainResult ? `基准日 ${mainResult.base_date}（原值 ${numberText(mainResult.base_value)}）；区间末值 ${numberText(mainEnd == null ? null : mainResult.values[page.offset + mainEnd])}，涨跌幅 ${numberText(mainEnd == null ? null : mainResult.change_pct[page.offset + mainEnd])}%` : '正在按当前区间归一化…')}</p>
      {curves.slice(1).filter(layer => layer.page).map(layer => <p key={layer.id} className={layer.response?.error ? 'text-rose-700' : ''}>{layer.label}：{!layer.rebased ? '此输出保留原值' : layer.response?.error || (layer.response?.result ? `基准日 ${range.start}，区间末值 ${numberText(layer.values[endIndex])}` : '正在归一化…')}</p>)}
    </div>}
    <ReactECharts option={option} onEvents={onEvents} style={{ height: 260 + subplots.length * 200 }} notMerge lazyUpdate aria-label="历史情景节点结果图" />
    {curves.length > 1 && <details className="rounded-lg border border-slate-200 p-2 text-xs">
      <summary className="cursor-pointer font-semibold text-slate-600">查看对比数据（当前区间最近100条 · 原值）</summary>
      <div className="mt-2 max-h-64 overflow-auto"><table className="min-w-full text-left" aria-label="上游节点对比数据"><thead><tr><th className="whitespace-nowrap p-2">日期</th>{curves.map(layer => <th key={layer.id} className="min-w-32 p-2">{layer.label}</th>)}</tr></thead><tbody>{dates.slice(Math.max(startIndex, endIndex - 99), endIndex + 1).map(date => <tr key={date}><td className="whitespace-nowrap p-2">{date}</td>{curves.map(layer => { const index = layer.indices.get(date); return <td key={layer.id} className="p-2">{numberText(index == null ? null : layer.data?.values[index])}</td> })}</tr>)}</tbody></table></div>
    </details>}
  </section>
}

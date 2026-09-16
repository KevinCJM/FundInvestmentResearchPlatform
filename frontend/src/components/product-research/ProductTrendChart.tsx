import { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import ReactECharts from 'echarts-for-react'
import { useI18n } from '../../i18n/runtime'
import IndicatorParameterInputs from '../indicator-parameters/IndicatorParameterInputs'
import { MetricSelector } from '../metrics/MetricDisplay'
import {
  evaluateTimeSeriesIndicators,
  type IndicatorDefinition,
  type ProductKind,
  type TimeSeriesChannelResult,
  type TimeSeriesIndicatorResult,
} from '../../services/customIndicators'
import {
  fetchProductPriceSeries,
  type ChartBasisOption,
  type ChartPriceBasis,
  type ProductPriceSeries,
} from '../../services/productAnalysis'
import type { HistoricalRegimeRun } from '../../services/historicalRegimes'

/** Where a selected indicator is drawn. `native` only exists when a matching axis does. */
type Placement = 'native' | 'right' | 'panel'

export const MAX_TREND_OVERLAYS = 6

const PRICE_SEMANTIC_BY_BASIS: Record<string, string> = {
  raw_kline: 'raw_market_price',
  adjusted_kline: 'adjusted_market_price',
  adjusted_nav: 'adjusted_nav',
}
const PLACEMENT_LABELS: Record<Placement, string> = {
  native: '同轴同图',
  right: '右轴同图',
  panel: '独立子图',
}

/** Only hex values already used elsewhere in the app: design:check counts distinct ones. */
const OVERLAY_COLORS = ['#0ea5e9', '#f97316', '#10b981', '#3b82f6', '#dc2626', '#34d399', '#94a3b8']

const AXIS_LINE = '#cbd5f5'
const SPLIT_LINE = '#e2e8f0'
const AXIS_LABEL = { color: '#475569', fontSize: 10, hideOverlap: true }

const LAYOUT = { top: 46, price: 300, volume: 90, panel: 110, gap: 26, bottom: 62 }

const normalizeDateKey = (value: string) => {
  const text = value.trim()
  if (/^\d{8}$/.test(text)) return `${text.slice(0, 4)}-${text.slice(4, 6)}-${text.slice(6, 8)}`
  const date = text.slice(0, 10)
  return /^\d{4}-\d{2}-\d{2}$/.test(date) ? date : null
}

const alignedChannelValues = (
  channel: TimeSeriesChannelResult | undefined,
  result: TimeSeriesIndicatorResult | undefined,
  dates: string[],
): Array<number | null> => {
  if (!channel || !result) return dates.map(() => null)
  const byDate = new Map(
    result.dates.map((date, index) => [normalizeDateKey(date) ?? date, channel.values[index] ?? null]),
  )
  const scale = channel.display_format === 'percent' ? 100 : 1
  return dates.map((date) => {
    const value = byDate.get(normalizeDateKey(date) ?? date)
    return value === null || value === undefined ? null : value * scale
  })
}

interface RegimeMarkAreaBoundary {
  name?: string
  xAxis: string
  itemStyle?: { color: string; opacity: number }
}

type RegimeMarkArea = [RegimeMarkAreaBoundary, RegimeMarkAreaBoundary]

const buildRegimeMarkAreas = (run: HistoricalRegimeRun | undefined, dates: string[]): RegimeMarkArea[] => {
  if (!run || dates.length === 0) return []
  const datedCategories = dates
    .map((date) => ({ date, key: normalizeDateKey(date) }))
    .filter((item): item is { date: string; key: string } => item.key !== null)
  const statesById = new Map(run.states.map((state) => [state.id, state]))
  return run.segments.flatMap((segment): RegimeMarkArea[] => {
    const state = statesById.get(segment.state_id)
    const startDate = normalizeDateKey(segment.start_date)
    const endDate = normalizeDateKey(segment.end_date)
    if (!state?.label || !state.color || !startDate || !endDate || startDate > endDate) return []
    const inside = datedCategories.filter(({ key }) => key >= startDate && key <= endDate)
    if (inside.length === 0) return []
    return [[
      { name: state.label, xAxis: inside[0].date, itemStyle: { color: state.color, opacity: 0.12 } },
      { xAxis: inside[inside.length - 1].date },
    ]]
  })
}

interface Props {
  productId: string
  productKind: ProductKind
  /** The single-product catalog, already filtered to this product kind. */
  indicators: IndicatorDefinition[]
  regimeRun?: HistoricalRegimeRun
  regimeStateId: string
  regimeSegment?: { startDate: string; endDate: string }
  windowStart?: string | null
  windowEnd?: string | null
  onDefinition: (indicator: IndicatorDefinition) => void
  studioHref: string
}

/**
 * The price path in one explicit basis, with any saved time-series indicator
 * drawn on the same time axis.
 *
 * The four hard-coded technical toggles this replaces were themselves catalog
 * built-ins; generalising the overlay removed them rather than adding to them.
 */
export default function ProductTrendChart({
  productId, productKind, indicators, regimeRun, regimeStateId, regimeSegment,
  windowStart, windowEnd, onDefinition, studioHref,
}: Props) {
  const { s } = useI18n()
  const [basis, setBasis] = useState<ChartPriceBasis>(productKind === 'etf' ? 'raw_kline' : 'adjusted_nav')
  const [series, setSeries] = useState<{ key: string; data?: ProductPriceSeries; error?: string } | null>(null)
  /** Kept across requests: a failed basis must not take the switch away with it. */
  const [bases, setBases] = useState<ChartBasisOption[]>([])
  const [selectedIds, setSelectedIds] = useState<string[]>([])
  const [parameters, setParameters] = useState<Record<string, Record<string, number>>>({})
  const [placements, setPlacements] = useState<Record<string, Placement>>({})
  const [overlay, setOverlay] = useState<{ key: string; results?: TimeSeriesIndicatorResult[]; error?: string } | null>(null)

  useEffect(() => {
    setBasis(productKind === 'etf' ? 'raw_kline' : 'adjusted_nav')
    setBases([])
    setSelectedIds([])
    setParameters({})
    setPlacements({})
  }, [productId, productKind])

  const seriesKey = JSON.stringify([productId, productKind, basis])
  useEffect(() => {
    // The reset below lands a render later, so skip the request a fund can only 422 on.
    if (!productId || (productKind !== 'etf' && basis !== 'adjusted_nav')) return undefined
    const controller = new AbortController()
    fetchProductPriceSeries(productId, productKind, basis, controller.signal)
      .then((data) => { setSeries({ key: seriesKey, data }); setBases(data.bases) })
      .catch((failure) => {
        if (controller.signal.aborted) return
        setSeries({ key: seriesKey, error: failure instanceof Error ? failure.message : '走势数据加载失败。' })
      })
    return () => controller.abort()
  }, [seriesKey])

  const seriesCatalog = useMemo(
    () => indicators.filter((item) => item.result_kind === 'time_series'),
    [indicators],
  )
  const selected = useMemo(
    () => selectedIds
      .map((id) => seriesCatalog.find((item) => item.id === id))
      .filter((item): item is IndicatorDefinition => Boolean(item)),
    [seriesCatalog, selectedIds],
  )

  const overlayKey = JSON.stringify(selected.map((item) => [item.id, item.revision, parameters[item.id] ?? null]))
  useEffect(() => {
    if (selected.length === 0) { setOverlay(null); return undefined }
    let active = true
    evaluateTimeSeriesIndicators({
      indicator_instances: selected.map((item) => ({
        indicator_id: item.id,
        indicator_revision: item.revision,
        ...(Object.keys(parameters[item.id] ?? {}).length ? { parameters: parameters[item.id] } : {}),
      })),
      target: { kind: productKind, product_id: productId },
      period: 'ALL',
      max_points: 5000,
    })
      .then((response) => { if (active) setOverlay({ key: overlayKey, results: response.results }) })
      .catch((failure) => {
        if (active) setOverlay({ key: overlayKey, error: failure instanceof Error ? failure.message : '叠加指标计算失败。' })
      })
    return () => { active = false }
  }, [productId, productKind, overlayKey])

  const current = series?.key === seriesKey ? series : null
  const payload = current?.data
  const points = useMemo(() => (payload?.available ? payload.points : []), [payload])
  const dates = useMemo(() => points.map((point) => point.date), [points])
  const hasOhlc = points.length > 0 && points.every((point) => (
    point.open !== null && point.high !== null && point.low !== null && Number.isFinite(point.close)
  ))
  const hasVolume = points.some((point) => point.volume !== null && Number.isFinite(point.volume))
  const overlayResults = useMemo(
    () => (overlay?.key === overlayKey ? overlay.results ?? [] : []),
    [overlay, overlayKey],
  )
  const overlayLoading = selected.length > 0 && overlay?.key !== overlayKey

  /**
   * Which axis a set of channels may legitimately share. Derived from the
   * channel contract, never from an id list: a raw-market moving average drawn
   * on an adjusted candle would be a different number pretending to be the
   * same one.
   */
  const priceSemantic = PRICE_SEMANTIC_BY_BASIS[basis] ?? null
  const nativeAxisOf = (channels: TimeSeriesChannelResult[]): 'price' | 'volume' | null => {
    if (channels.length === 0) return null
    if (priceSemantic && channels.every((item) => item.semantic_dimension === priceSemantic)) return 'price'
    if (hasVolume && channels.every((item) => item.semantic_dimension === 'volume')) return 'volume'
    return null
  }

  const rows = useMemo(() => selected.map((indicator) => {
    const result = overlayResults.find((item) => item.indicator_id === indicator.id)
    const channels = result?.channels ?? []
    const native = nativeAxisOf(channels)
    const stored = placements[indicator.id]
    const place: Placement = stored === 'native' && !native ? 'panel' : stored ?? (native ? 'native' : 'panel')
    const schema = indicator.parameter_contract_version === '1.0' ? indicator.parameter_schema ?? [] : []
    return { indicator, result, channels, native, place, schema }
  }), [selected, overlayResults, placements, priceSemantic, hasVolume])

  const regimeMarkAreas = useMemo(() => {
    if (!regimeRun) return []
    const inWindow = dates.filter((date) => (
      (!windowStart || date >= windowStart) && (!windowEnd || date <= windowEnd)
    ))
    return buildRegimeMarkAreas({
      ...regimeRun,
      segments: regimeRun.segments.filter((segment) => (
        (!regimeStateId || segment.state_id === regimeStateId)
        && (!regimeSegment || (segment.start_date <= regimeSegment.endDate && segment.end_date >= regimeSegment.startDate))
      )),
    }, inWindow)
  }, [regimeRun, regimeStateId, regimeSegment, dates, windowStart, windowEnd])

  const chart = useMemo(() => {
    if (dates.length === 0) return null
    const grids: Array<Record<string, unknown>> = []
    const xAxis: Array<Record<string, unknown>> = []
    const yAxis: Array<Record<string, unknown>> = []
    const echartsSeries: Array<Record<string, unknown>> = []
    let top = LAYOUT.top
    const addGrid = (height: number) => {
      grids.push({ left: 12, right: 18, top, height, containLabel: true })
      top += height + LAYOUT.gap
      return grids.length - 1
    }
    const priceGrid = addGrid(LAYOUT.price)
    const volumeGrid = hasVolume ? addGrid(LAYOUT.volume) : -1
    const panelRows = rows.filter((row) => row.place === 'panel' && row.channels.length > 0)
    const panelGrids = panelRows.map(() => addGrid(LAYOUT.panel))

    const axisFor = (gridIndex: number, options: Record<string, unknown> = {}) => {
      yAxis.push({ gridIndex, scale: true, axisLine: { lineStyle: { color: AXIS_LINE } }, splitLine: { lineStyle: { color: SPLIT_LINE } }, axisLabel: AXIS_LABEL, ...options })
      return yAxis.length - 1
    }
    const priceAxis = axisFor(priceGrid)
    const rightRows = rows.filter((row) => row.place === 'right' && row.channels.length > 0)
    const rightAxis = rightRows.length > 0
      ? axisFor(priceGrid, {
        position: 'right',
        name: rightRows[0].channels[0].unit || '',
        splitLine: { show: false },
        axisLabel: { ...AXIS_LABEL, formatter: rightRows[0].channels[0].display_format === 'percent' ? '{value}%' : '{value}' },
      })
      : -1
    const volumeAxis = volumeGrid >= 0 ? axisFor(volumeGrid) : -1
    const panelAxes = panelRows.map((row, index) => axisFor(panelGrids[index], {
      name: row.channels[0].unit || '',
      axisLabel: { ...AXIS_LABEL, formatter: row.channels[0].display_format === 'percent' ? '{value}%' : '{value}' },
    }))

    grids.forEach((_, index) => xAxis.push({
      type: 'category',
      gridIndex: index,
      data: dates,
      boundaryGap: false,
      axisTick: { show: index === grids.length - 1 },
      axisLine: { lineStyle: { color: AXIS_LINE } },
      axisLabel: index === grids.length - 1 ? { ...AXIS_LABEL, showMinLabel: false, showMaxLabel: false } : { show: false },
    }))

    const markArea = regimeMarkAreas.length > 0
      ? { markArea: { silent: true, label: { show: true, position: 'insideTop', color: '#334155', fontSize: 10 }, data: regimeMarkAreas } }
      : {}
    echartsSeries.push(hasOhlc
      ? {
        name: '价格', type: 'candlestick',
        data: points.map((point) => [point.open, point.close, point.low, point.high]),
        itemStyle: { color: '#0ea5e9', color0: '#f87171', borderColor: '#0284c7', borderColor0: '#dc2626' },
        ...markArea,
      }
      : {
        name: '价格', type: 'line', data: points.map((point) => point.close), showSymbol: false,
        lineStyle: { width: 1.6, color: '#0ea5e9' }, ...markArea,
      })
    if (volumeGrid >= 0) {
      echartsSeries.push({
        name: '成交量', type: 'bar', xAxisIndex: volumeGrid, yAxisIndex: volumeAxis, barWidth: '60%',
        data: points.map((point) => ({
          value: point.volume,
          itemStyle: { color: point.open !== null && point.close >= point.open ? '#34d399' : '#94a3b8' },
        })),
      })
    }
    let color = 0
    for (const row of rows) {
      if (row.channels.length === 0) continue
      const panelIndex = panelRows.indexOf(row)
      const gridIndex = row.place === 'panel' ? panelGrids[panelIndex]
        : row.place === 'native' && row.native === 'volume' ? volumeGrid
        : priceGrid
      const axisIndex = row.place === 'panel' ? panelAxes[panelIndex]
        : row.place === 'right' ? rightAxis
        : row.native === 'volume' ? volumeAxis
        : priceAxis
      for (const channel of row.channels) {
        echartsSeries.push({
          name: row.channels.length > 1 ? `${row.indicator.name} ${channel.label}` : channel.label,
          type: 'line', xAxisIndex: gridIndex, yAxisIndex: axisIndex,
          data: alignedChannelValues(channel, row.result, dates),
          smooth: true, showSymbol: false, connectNulls: false,
          lineStyle: { width: 1.4, color: OVERLAY_COLORS[color % OVERLAY_COLORS.length] },
        })
        color += 1
      }
    }

    const defaultStart = Math.max(0, dates.length - 252)
    const startValue = windowStart ? dates.find((date) => date >= windowStart) ?? dates[defaultStart] : dates[defaultStart]
    const endValue = windowEnd ? dates.filter((date) => date <= windowEnd).pop() ?? dates[dates.length - 1] : dates[dates.length - 1]
    const allX = grids.map((_, index) => index)
    return {
      option: {
        backgroundColor: '#ffffff',
        animation: false,
        tooltip: { trigger: 'axis', axisPointer: { type: 'cross', crossStyle: { color: '#94a3b8' } } },
        axisPointer: { link: [{ xAxisIndex: 'all' }] },
        legend: { type: 'scroll', top: 8, left: 'center', icon: 'roundRect', textStyle: { color: '#475569', fontSize: 12 } },
        grid: grids,
        xAxis,
        yAxis,
        dataZoom: [
          { type: 'inside', xAxisIndex: allX, startValue, endValue },
          { show: true, type: 'slider', xAxisIndex: allX, height: 18, bottom: 12, startValue, endValue },
        ],
        series: echartsSeries,
      },
      height: top - LAYOUT.gap + LAYOUT.bottom,
    }
  }, [dates, points, hasOhlc, hasVolume, rows, regimeMarkAreas, windowStart, windowEnd])

  const remove = (indicatorId: string) => setSelectedIds((ids) => ids.filter((id) => id !== indicatorId))

  return <section className="min-w-0 overflow-hidden rounded-xl border border-slate-200 bg-white shadow-sm" aria-labelledby="product-trend-chart-title">
    <div className="flex flex-col gap-3 px-4 py-4 sm:px-5">
      <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
        <h2 id="product-trend-chart-title" className="text-lg font-semibold text-slate-900">走势图</h2>
        <p className="text-xs text-slate-600" aria-live="polite">
          {current
            ? payload?.available
              ? `${payload.label} · ${dates[0] ?? '—'} 至 ${dates[dates.length - 1] ?? '—'} · ${dates.length} 个观察值`
              : payload?.label ?? ''
            : '正在读取走势数据…'}
        </p>
      </div>
      <fieldset className="min-w-0">
        <legend className="text-xs font-medium text-slate-700">价格口径</legend>
        <div className="mt-2 flex flex-wrap gap-2">
          {bases.map((option) => (
            <label key={option.id} title={option.reason ?? option.description}
              className={`inline-flex min-h-10 items-center gap-2 rounded-lg border px-3 text-sm transition ${
                option.id === basis ? 'border-accent-500 bg-accent-50 font-semibold text-accent-800'
                  : option.available ? 'cursor-pointer border-slate-200 bg-white text-slate-700 hover:border-accent-300'
                  : 'cursor-not-allowed border-slate-200 bg-slate-50 text-slate-600'}`}>
              <input type="radio" name="product-trend-basis" value={option.id} checked={option.id === basis}
                disabled={!option.available && option.id !== basis}
                onChange={() => setBasis(option.id as ChartPriceBasis)}
                className="h-4 w-4 accent-accent-600 focus-visible:ring-2 focus-visible:ring-accent-500" />
              {option.label}
            </label>
          ))}
        </div>
        <p className="mt-2 text-xs leading-5 text-slate-600">
          {bases.find((option) => option.id === basis)?.description ?? '按所选口径读取真实行情，不在前端换算。'}
        </p>
      </fieldset>
    </div>

    <div className="border-t border-slate-100 px-4 py-4 sm:px-5">
      {!current
        ? <p role="status" className="py-10 text-center text-sm text-slate-600">正在读取走势数据…</p>
        : current.error
          ? <p role="alert" className="rounded-lg bg-rose-50 px-4 py-3 text-sm text-rose-700">{current.error}</p>
          : !payload?.available
            ? <div className="py-10 text-center">
              <p className="text-sm font-semibold text-slate-700">这个口径暂时没有数据</p>
              <p className="mx-auto mt-2 max-w-lg text-sm leading-6 text-slate-600">{payload?.reason ?? '换一个价格口径，或先在数据中心补齐对应数据集。'}</p>
            </div>
            : chart
              ? <ReactECharts option={chart.option} style={{ height: chart.height }} notMerge lazyUpdate />
              : <p className="py-10 text-center text-sm text-slate-600">暂无可视化数据</p>}
      {payload?.available && basis !== 'adjusted_nav' && (!hasOhlc || !hasVolume) && (
        <p role="status" className="mt-3 rounded-lg bg-amber-50 px-4 py-2 text-xs leading-5 text-amber-800">
          原始数据未完整披露{!hasOhlc ? ' 开高低价' : ''}{!hasVolume ? ' 成交量' : ''}；
          本图只画已披露的真实数值，不使用 close 或 0 伪造缺失字段，缺开高低价时退回收盘价折线。
        </p>
      )}
      {(payload?.warnings ?? []).map((warning, index) => (
        <p key={`${index}-${warning}`} role="status" className="mt-3 rounded-lg bg-amber-50 px-4 py-2 text-xs leading-5 text-amber-800">{warning}</p>
      ))}
    </div>

    <div className="border-t border-slate-100 px-4 py-4 sm:px-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="min-w-0">
          <h3 className="text-sm font-semibold text-slate-800">叠加时序指标</h3>
          <p className="mt-1 text-xs leading-5 text-slate-600">
            指标中心里的时序指标都能画到这张图上；{s('indicatorParameters.runtimeHint')}
          </p>
        </div>
        <div className="shrink-0">
          <MetricSelector indicators={seriesCatalog} selectedIds={selectedIds} onChange={setSelectedIds}
            maxSelected={MAX_TREND_OVERLAYS} label="选择时序指标" />
        </div>
      </div>
      {seriesCatalog.length === 0
        ? <p className="mt-4 text-sm leading-6 text-slate-600">
          指标目录里还没有适用于本产品的时序指标。<Link to={studioHref} className="font-medium text-accent-700 hover:underline">前往指标中心</Link>新建一个，再回到本页叠加。
        </p>
        : rows.length === 0
          ? <p className="mt-4 text-sm leading-6 text-slate-600">还没有叠加指标。点击“选择时序指标”，把滚动波动率、均线、KDJ 等画到上面这张图里。</p>
          : <ul className="mt-3 divide-y divide-slate-100" aria-live="polite">
            {rows.map((row) => (
              <li key={row.indicator.id} className="min-w-0 py-3">
                <div className="flex flex-wrap items-center justify-between gap-x-3 gap-y-2">
                  <div className="min-w-0">
                    <p className="truncate text-sm font-semibold text-slate-900">{row.indicator.name}</p>
                    <p className="mt-0.5 text-xs text-slate-600">
                      {row.indicator.source === 'built_in' ? '内置' : '工作区'} v{row.indicator.revision}
                      {row.channels.length > 0 ? ` · ${row.channels.map((channel) => channel.label).join(' / ')}` : ''}
                    </p>
                  </div>
                  <div className="flex shrink-0 items-center gap-2">
                    <label className="flex items-center gap-2 text-xs text-slate-600">位置
                      <select aria-label={`${row.indicator.name}的显示位置`} value={row.place}
                        onChange={(event) => setPlacements((state) => ({ ...state, [row.indicator.id]: event.target.value as Placement }))}
                        className="min-h-10 rounded-lg border border-slate-200 bg-white px-2 text-sm text-slate-700 focus:border-accent-500 focus:outline-none focus-visible:ring-2 focus-visible:ring-accent-500">
                        <option value="native" disabled={!row.native}>{PLACEMENT_LABELS.native}</option>
                        <option value="right">{PLACEMENT_LABELS.right}</option>
                        <option value="panel">{PLACEMENT_LABELS.panel}</option>
                      </select>
                    </label>
                    <button type="button" onClick={() => onDefinition(row.indicator)}
                      className="min-h-10 rounded-lg px-2 text-xs font-medium text-accent-700 hover:underline focus:outline-none focus-visible:ring-2 focus-visible:ring-accent-500">定义</button>
                    <button type="button" onClick={() => remove(row.indicator.id)} aria-label={`移除指标 ${row.indicator.name}`}
                      title="仅从这张图上移除，不会删除指标定义"
                      className="inline-flex min-h-10 items-center rounded-lg border border-slate-200 px-2.5 text-xs font-medium text-slate-600 hover:border-rose-300 hover:bg-rose-50 hover:text-rose-700 focus:outline-none focus-visible:ring-2 focus-visible:ring-accent-500">移除</button>
                  </div>
                </div>
                {!row.native && <p className="mt-1 text-xs text-slate-600">口径与主图不同，不能与价格共轴；请用右轴或独立子图。</p>}
                {row.result && row.result.status !== 'ok' && row.result.warnings.map((warning, index) => (
                  <p key={`${warning.code}-${index}`} role="status" className="mt-1 text-xs text-amber-800">{warning.message}</p>
                ))}
                {row.schema.length > 0
                  ? <IndicatorParameterInputs schema={row.schema} values={parameters[row.indicator.id] ?? {}}
                    onApply={(values) => setParameters((state) => ({ ...state, [row.indicator.id]: values }))}
                    effective={row.result?.parameters ?? null} />
                  : <p className="mt-1 text-xs text-slate-600">{s('indicatorParameters.fixedHint')}</p>}
              </li>
            ))}
          </ul>}
      {overlayLoading && <p role="status" className="mt-3 text-xs text-slate-600">{s('indicatorParameters.calculating')}</p>}
      {overlay?.key === overlayKey && overlay.error && (
        <p role="alert" className="mt-3 rounded-lg bg-rose-50 px-4 py-2 text-xs leading-5 text-rose-700">{overlay.error}</p>
      )}
    </div>
  </section>
}

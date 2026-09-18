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
import Mascot from '../Mascot'

/**
 * Where one overlay instance is drawn. `native` only exists when a matching axis
 * does; `panel:<key>` names the instance that owns the subplot, so two instances
 * asking for the same owner share one grid instead of getting one each.
 */
type Placement = string

const PANEL_PREFIX = 'panel:'
const panelValue = (instanceKey: string) => `${PANEL_PREFIX}${instanceKey}`
const panelOwnerOf = (placement: Placement) => (
  placement.startsWith(PANEL_PREFIX) ? placement.slice(PANEL_PREFIX.length) : null
)

/** One row of the overlay list. The same indicator may appear more than once. */
interface OverlayInstance {
  key: string
  indicatorId: string
  parameters: Record<string, number>
  /** Unset until the user chooses; the default follows the channel contract. */
  placement?: Placement
  /** A copy shares this subplot only if its resolved default is not a native axis. */
  defaultPanelOwner?: string
}

// Matches EvaluateSeriesRequest / MAX_SERIES_INSTANCES; the list itself is unbounded.
const SERIES_BATCH_SIZE = 50
let instanceSequence = 0
const nextInstanceKey = () => `ov${(instanceSequence += 1)}`

const PRICE_SEMANTIC_BY_BASIS: Record<string, string> = {
  raw_kline: 'raw_market_price',
  adjusted_kline: 'adjusted_market_price',
  adjusted_nav: 'adjusted_nav',
}
const PLACEMENT_LABELS = { native: '同轴同图', right: '右轴同图', panel: '独立子图' } as const

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

/**
 * Names have to stay distinct: two list rows reading alike are indistinguishable,
 * and ECharts links legend entries by name, so two series sharing one would
 * switch on and off together.
 */
const ordinalNamer = () => {
  const seen = new Map<string, number>()
  return (name: string) => {
    const count = (seen.get(name) ?? 0) + 1
    seen.set(name, count)
    return count === 1 ? name : `${name} #${count}`
  }
}

/**
 * What a set of channels is measured in. Two instances may share one vertical
 * axis only when this matches: a percentage and a price on one scale would be
 * two different numbers pretending to be the same one. `null` means the channels
 * disagree among themselves, so the set cannot host or join a shared axis.
 */
const axisIdentity = (channels: TimeSeriesChannelResult[]): string | null => {
  if (channels.length === 0) return null
  const identities = new Set(channels.map((item) => (
    `${item.semantic_dimension ?? ''}|${item.unit}|${item.display_format}`
  )))
  return identities.size === 1 ? [...identities][0] : null
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
  const [instances, setInstances] = useState<OverlayInstance[]>([])
  const [overlay, setOverlay] = useState<{ key: string; results?: TimeSeriesIndicatorResult[]; error?: string } | null>(null)

  useEffect(() => {
    setBasis(productKind === 'etf' ? 'raw_kline' : 'adjusted_nav')
    setBases([])
    setInstances([])
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
  /** Instances whose indicator is still in the catalog, in list order. */
  const requested = useMemo(
    () => instances.flatMap((instance) => {
      const indicator = seriesCatalog.find((item) => item.id === instance.indicatorId)
      return indicator ? [{ instance, indicator }] : []
    }),
    [seriesCatalog, instances],
  )
  const selectedIds = useMemo(
    () => [...new Set(requested.map((item) => item.indicator.id))],
    [requested],
  )

  const overlayKey = JSON.stringify([productId, productKind, requested.map(({ instance, indicator }) => (
    [instance.key, indicator.id, indicator.revision, instance.parameters]
  ))])
  useEffect(() => {
    if (requested.length === 0) { setOverlay(null); return undefined }
    let active = true
    const evaluate = async () => {
      const results: TimeSeriesIndicatorResult[] = []
      // Sequential batches bound server work; superseded requests cannot launch another batch.
      for (let start = 0; start < requested.length && active; start += SERIES_BATCH_SIZE) {
        const response = await evaluateTimeSeriesIndicators({
          indicator_instances: requested.slice(start, start + SERIES_BATCH_SIZE).map(({ instance, indicator }) => ({
            instance_key: instance.key,
            indicator_id: indicator.id,
            indicator_revision: indicator.revision,
            ...(Object.keys(instance.parameters).length ? { parameters: instance.parameters } : {}),
          })),
          target: { kind: productKind, product_id: productId },
          period: 'ALL',
          max_points: 5000,
        })
        results.push(...response.results)
      }
      if (active) setOverlay({ key: overlayKey, results })
    }
    void evaluate().catch((failure) => {
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
  const overlayLoading = requested.length > 0 && overlay?.key !== overlayKey

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

  const rows = useMemo(() => {
    const repeated = new Set(requested
      .map(({ indicator }) => indicator.id)
      .filter((id, index, all) => all.indexOf(id) !== index))
    const nameRow = ordinalNamer()
    const base = requested.map(({ instance, indicator }) => {
      // Matched by the instance's own key: one indicator can be on the chart twice.
      const result = overlayResults.find((item) => item.instance_key === instance.key)
      const channels = result?.channels ?? []
      const native = nativeAxisOf(channels)
      const own = panelValue(instance.key)
      const stored = instance.placement ?? (!native && instance.defaultPanelOwner ? panelValue(instance.defaultPanelOwner) : undefined)
      const place: Placement = stored === 'native' && !native ? own : stored ?? (native ? 'native' : own)
      const schema = indicator.parameter_contract_version === '1.0' ? indicator.parameter_schema ?? [] : []
      // Only a repeated indicator carries its parameters in the name; a lone one
      // keeps reading exactly as before.
      const values = result?.parameters ?? instance.parameters
      const suffix = repeated.has(indicator.id) && schema.length > 0
        ? ` (${schema.map((item) => `${item.label}=${values[item.id] ?? item.default}`).join(' · ')})`
        : ''
      return {
        instance, indicator, result, channels, native, place, schema,
        axisKey: axisIdentity(channels),
        repeated: repeated.has(indicator.id),
        label: nameRow(`${indicator.name}${suffix}`),
      }
    })
    // Merging is one hop deep: only an instance that owns its own subplot can be
    // joined, so A → B → C chains cannot form and no follower is ever orphaned.
    const owners = new Set(base
      .filter((item) => item.place === panelValue(item.instance.key) && item.channels.length > 0)
      .map((item) => item.instance.key))
    return base.map((item) => {
      const owner = panelOwnerOf(item.place)
      if (owner === null || owner === item.instance.key) return { ...item, group: item.instance.key }
      const host = base.find((other) => other.instance.key === owner)
      const shareable = owners.has(owner) && item.axisKey !== null && host?.axisKey === item.axisKey
      return shareable
        ? { ...item, group: owner }
        : { ...item, place: panelValue(item.instance.key), group: item.instance.key }
    })
  }, [requested, overlayResults, priceSemantic, hasVolume])

  const rightRows = useMemo(
    () => rows.filter((row) => row.place === 'right' && row.channels.length > 0),
    [rows],
  )
  /**
   * The unit the right axis may claim. Sharing it is the user's own choice and
   * the scales do give way to each other, but a percent label printed over
   * prices would misread every tick, so a mixed axis stays unlabelled.
   */
  const rightChannel = useMemo(
    () => (rightRows.length > 0 && rightRows.every((row) => row.axisKey !== null && row.axisKey === rightRows[0].axisKey)
      ? rightRows[0].channels[0]
      : null),
    [rightRows],
  )
  const mixedRightAxis = rightRows.length > 0 && rightChannel === null

  /** Subplots that another instance may join: same scale, and not merged already. */
  const panelHosts = useMemo(
    () => rows.filter((row) => row.group === row.instance.key
      && row.place.startsWith(PANEL_PREFIX)
      && row.channels.length > 0),
    [rows],
  )
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
    const panelRows = rows.filter((row) => row.place.startsWith(PANEL_PREFIX) && row.channels.length > 0)
    // One grid per group, not per instance: that is what sharing a subplot means.
    const panelGroups = [...new Set(panelRows.map((row) => row.group))]
    const panelGrids = panelGroups.map(() => addGrid(LAYOUT.panel))

    const axisFor = (gridIndex: number, options: Record<string, unknown> = {}) => {
      yAxis.push({ gridIndex, scale: true, axisLine: { lineStyle: { color: AXIS_LINE } }, splitLine: { lineStyle: { color: SPLIT_LINE } }, axisLabel: AXIS_LABEL, ...options })
      return yAxis.length - 1
    }
    const priceAxis = axisFor(priceGrid)
    const rightAxis = rightRows.length > 0
      ? axisFor(priceGrid, {
        position: 'right',
        name: rightChannel?.unit || '',
        splitLine: { show: false },
        axisLabel: { ...AXIS_LABEL, formatter: rightChannel?.display_format === 'percent' ? '{value}%' : '{value}' },
      })
      : -1
    const volumeAxis = volumeGrid >= 0 ? axisFor(volumeGrid) : -1
    const panelAxes = panelGroups.map((group, index) => {
      const channel = panelRows.find((row) => row.group === group)!.channels[0]
      return axisFor(panelGrids[index], {
        name: channel.unit || '',
        axisLabel: { ...AXIS_LABEL, formatter: channel.display_format === 'percent' ? '{value}%' : '{value}' },
      })
    })

    grids.forEach((_, index) => xAxis.push({
      type: 'category',
      gridIndex: index,
      data: dates,
      boundaryGap: false,
      axisTick: { show: index === grids.length - 1 },
      axisLine: { lineStyle: { color: AXIS_LINE } },
      axisLabel: index === grids.length - 1 ? { ...AXIS_LABEL, showMinLabel: false, showMaxLabel: false } : { show: false },
    }))

    const uniqueName = ordinalNamer()

    const markArea = regimeMarkAreas.length > 0
      ? { markArea: { silent: true, label: { show: true, position: 'insideTop', color: '#334155', fontSize: 10 }, data: regimeMarkAreas } }
      : {}
    echartsSeries.push(hasOhlc
      ? {
        name: uniqueName('价格'), type: 'candlestick',
        data: points.map((point) => [point.open, point.close, point.low, point.high]),
        itemStyle: { color: '#0ea5e9', color0: '#f87171', borderColor: '#0284c7', borderColor0: '#dc2626' },
        ...markArea,
      }
      : {
        name: uniqueName('价格'), type: 'line', data: points.map((point) => point.close), showSymbol: false,
        lineStyle: { width: 1.6, color: '#0ea5e9' }, ...markArea,
      })
    if (volumeGrid >= 0) {
      echartsSeries.push({
        name: uniqueName('成交量'), type: 'bar', xAxisIndex: volumeGrid, yAxisIndex: volumeAxis, barWidth: '60%',
        data: points.map((point) => ({
          value: point.volume,
          itemStyle: { color: point.open !== null && point.close >= point.open ? '#34d399' : '#94a3b8' },
        })),
      })
    }
    let color = 0
    for (const row of rows) {
      if (row.channels.length === 0) continue
      const panelIndex = panelGroups.indexOf(row.group)
      const onPanel = row.place.startsWith(PANEL_PREFIX)
      const gridIndex = onPanel ? panelGrids[panelIndex]
        : row.place === 'native' && row.native === 'volume' ? volumeGrid
        : priceGrid
      const axisIndex = onPanel ? panelAxes[panelIndex]
        : row.place === 'right' ? rightAxis
        : row.native === 'volume' ? volumeAxis
        : priceAxis
      for (const channel of row.channels) {
        const stroke = OVERLAY_COLORS[color % OVERLAY_COLORS.length]
        echartsSeries.push({
          name: uniqueName(row.channels.length > 1
            ? `${row.label} ${channel.label}`
            : row.repeated ? row.label : channel.label),
          type: 'line', xAxisIndex: gridIndex, yAxisIndex: axisIndex,
          data: alignedChannelValues(channel, row.result, dates),
          smooth: true, showSymbol: false, connectNulls: false,
          lineStyle: { width: 1.4, color: stroke },
          // The legend swatch reads from itemStyle: without it the key colour and
          // the drawn line disagree, which is exactly what tells two curves apart.
          itemStyle: { color: stroke },
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
  }, [dates, points, hasOhlc, hasVolume, rows, rightRows, rightChannel, regimeMarkAreas, windowStart, windowEnd])

  const remove = (instanceKey: string) => setInstances((current) => (
    current.filter((item) => item.key !== instanceKey)
  ))

  /** The selector adds and removes indicators; the list owns how many of each. */
  const changeSelection = (ids: string[]) => setInstances((current) => [
    ...current.filter((item) => ids.includes(item.indicatorId)),
    ...ids
      .filter((id) => !current.some((item) => item.indicatorId === id))
      .map((id) => ({ key: nextInstanceKey(), indicatorId: id, parameters: {} })),
  ])

  /**
   * A copy lands next to its source and inherits where it is drawn, so the two
   * curves start out overlaid: comparing parameters is the whole point of asking
   * for a second one. The placement select then splits them apart if wanted.
   */
  const duplicate = (row: { instance: OverlayInstance; place: Placement; channels: TimeSeriesChannelResult[] }) => setInstances((current) => {
    const index = current.findIndex((item) => item.key === row.instance.key)
    if (index < 0) return current
    const copy: OverlayInstance = {
      key: nextInstanceKey(),
      indicatorId: row.instance.indicatorId,
      parameters: { ...row.instance.parameters },
      placement: row.channels.length ? row.place : row.instance.placement,
      defaultPanelOwner: row.instance.defaultPanelOwner ?? row.instance.key,
    }
    return [...current.slice(0, index + 1), copy, ...current.slice(index + 1)]
  })

  const updateInstance = (instanceKey: string, patch: Partial<OverlayInstance>) => setInstances((current) => (
    current.map((item) => (item.key === instanceKey ? { ...item, ...patch } : item))
  ))

  return <section className="min-w-0 overflow-hidden rounded-xl border border-slate-200 bg-white shadow-sm" aria-labelledby="product-trend-chart-title">
    <div className="flex flex-col gap-3 px-4 py-4 sm:px-5">
      <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
        <h2 id="product-trend-chart-title" className="text-lg font-semibold text-slate-900">走势图</h2>
        <p className="text-xs text-slate-600" aria-live="polite">
          {current
            ? payload?.available
              ? `${payload.label} · ${dates[0] ?? '—'} 至 ${dates[dates.length - 1] ?? '—'}`
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
        ? <div role="status" className="flex flex-col items-center gap-3 py-10 text-center">
          <Mascot state="working" />
          <p className="text-sm font-semibold text-slate-700">正在读取走势数据…</p>
          <p className="max-w-lg text-sm leading-6 text-slate-600">按所选价格口径从后端读取真实行情，不在浏览器里换算。</p>
        </div>
        : current.error
          ? <div role="alert" className="flex flex-col items-center gap-3 py-10 text-center">
            <Mascot state="error" />
            <p className="text-sm font-semibold text-slate-800">走势数据没能读出来</p>
            <p className="max-w-lg rounded-lg bg-rose-50 px-4 py-2 text-sm leading-6 text-rose-700">{current.error}</p>
          </div>
          : !payload?.available
            ? <div className="flex flex-col items-center gap-3 py-10 text-center">
              <Mascot state="empty" />
              <p className="text-sm font-semibold text-slate-700">这个口径暂时没有数据</p>
              <p className="max-w-lg text-sm leading-6 text-slate-600">{payload?.reason ?? '换一个价格口径，或先在数据中心补齐对应数据集。'}</p>
            </div>
            : chart
              ? <ReactECharts option={chart.option} style={{ height: chart.height }} notMerge lazyUpdate />
              : <div className="flex flex-col items-center gap-3 py-10 text-center">
                <Mascot state="noresult" />
                <p className="text-sm font-semibold text-slate-700">暂无可视化数据</p>
                <p className="max-w-lg text-sm leading-6 text-slate-600">这个口径读到了数据，但没有可以画成图的观察值。换一个价格口径或放宽时间范围再试。</p>
              </div>}
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
          <h3 className="text-sm font-semibold text-slate-800">
            叠加时序指标{rows.length > 0 && <span className="font-normal tabular-nums text-slate-600"> {rows.length} 条</span>}
          </h3>
          <p className="mt-1 text-xs leading-5 text-slate-600">
            同一个指标可以加多条，用不同参数对比。{s('indicatorParameters.runtimeHint')}
          </p>
        </div>
        <div className="shrink-0">
          <MetricSelector indicators={seriesCatalog} selectedIds={selectedIds} onChange={changeSelection}
            label="选择时序指标" />
        </div>
      </div>
      {seriesCatalog.length === 0
        ? <p className="mt-4 text-sm leading-6 text-slate-600">
          指标目录里还没有适用于本产品的时序指标。<Link to={studioHref} className="font-medium text-accent-700 hover:underline">前往指标中心</Link>新建一个，再回到本页叠加。
        </p>
        : rows.length === 0
          ? <p className="mt-4 text-sm leading-6 text-slate-600">还没有叠加指标。点击“选择时序指标”，把滚动波动率、均线、KDJ 等画到上面这张图里。</p>
          : <ul className="mt-3 divide-y divide-slate-100 border-t border-slate-100" aria-live="polite">
            {rows.map((row) => {
              const hosts = panelHosts.filter((host) => (
                host.instance.key !== row.instance.key && row.axisKey !== null && host.axisKey === row.axisKey
              ))
              const repeatReason = row.schema.length === 0
                ? '固定参数的指标再加一条也是同一条线；需要别的参数请在指标中心另存一个版本'
                : ''
              return <li key={row.instance.key} className="min-w-0 py-2">
                <div className="flex min-w-0 flex-wrap items-center gap-x-3 gap-y-2">
                  {/* 版本号和通道名放在悬停提示里：列表要能一眼扫完，不重复指标名已经说过的话。 */}
                  <p className="min-w-0 max-w-full truncate text-sm font-medium text-slate-900"
                    title={`${row.indicator.source === 'built_in' ? '内置' : '工作区'} v${row.indicator.revision}${
                      row.channels.length > 0 ? ` · ${row.channels.map((channel) => channel.label).join(' / ')}` : ''}`}>{row.label}</p>
                  {/* 参数就长在指标名旁边，用的是这一行本来就空着的地方，不另占一行。 */}
                  {row.schema.length > 0 && (
                    <IndicatorParameterInputs schema={row.schema} values={row.instance.parameters}
                      onApply={(values) => updateInstance(row.instance.key, { parameters: values })}
                      effective={row.result?.parameters ?? null} />
                  )}
                  {/* 四个控件在窄屏换行，不留在一行里被卡片裁掉。 */}
                  <div className="ml-auto flex min-w-0 flex-wrap items-center gap-2">
                    <label className="flex min-w-0 items-center gap-2 whitespace-nowrap text-xs text-slate-600">位置
                      <select aria-label={`${row.label}的显示位置`} value={row.place}
                        onChange={(event) => updateInstance(row.instance.key, { placement: event.target.value })}
                        className="min-h-10 min-w-0 flex-1 max-w-[14rem] rounded-lg border border-slate-200 bg-white px-2 text-sm text-slate-700 focus:border-accent-500 focus:outline-none focus-visible:ring-2 focus-visible:ring-accent-500">
                        <optgroup label="画在主图">
                          <option value="native" disabled={!row.native}>
                            {PLACEMENT_LABELS.native}{row.native ? '' : '（口径与主图不同，不可选）'}
                          </option>
                          <option value="right">{PLACEMENT_LABELS.right}</option>
                        </optgroup>
                        <optgroup label="画在子图">
                          <option value={panelValue(row.instance.key)}>{PLACEMENT_LABELS.panel}</option>
                          {hosts.map((host) => (
                            <option key={host.instance.key} value={panelValue(host.instance.key)}>并入「{host.label}」</option>
                          ))}
                        </optgroup>
                      </select>
                    </label>
                    <button type="button" onClick={() => duplicate(row)} disabled={Boolean(repeatReason)}
                      aria-label={`为 ${row.label} 再加一条`} title={repeatReason || '换一组参数再画一条，可以和这条放同一个子图里比'}
                      className="inline-flex min-h-10 items-center rounded-lg border border-slate-200 px-2.5 text-xs font-medium text-slate-700 transition hover:border-accent-300 hover:bg-accent-50 hover:text-accent-700 focus:outline-none focus-visible:ring-2 focus-visible:ring-accent-500 disabled:cursor-not-allowed disabled:opacity-50">再加一条</button>
                    <button type="button" onClick={() => onDefinition(row.indicator)}
                      className="min-h-10 rounded-lg px-2 text-xs font-medium text-accent-700 hover:underline focus:outline-none focus-visible:ring-2 focus-visible:ring-accent-500">定义</button>
                    <button type="button" onClick={() => remove(row.instance.key)} aria-label={`移除指标 ${row.label}`}
                      title="仅从这张图上移除，不会删除指标定义"
                      className="inline-flex min-h-10 items-center rounded-lg border border-slate-200 px-2.5 text-xs font-medium text-slate-600 hover:border-rose-300 hover:bg-rose-50 hover:text-rose-700 focus:outline-none focus-visible:ring-2 focus-visible:ring-accent-500">移除</button>
                  </div>
                </div>
                {row.group !== row.instance.key && (
                  // 下拉框会把长名字截断，共用了哪一格在这里写全。
                  <p className="mt-1 text-xs text-slate-600">
                    与「{rows.find((host) => host.instance.key === row.group)?.label}」共用一个子图，同一条纵轴。
                  </p>
                )}
                {row.place === 'right' && mixedRightAxis && (
                  <p className="mt-1 text-xs text-slate-600">右轴上还有口径不同的指标，刻度只表示数值大小，不带单位。</p>
                )}
                {row.result && row.result.status !== 'ok' && row.result.warnings.map((warning, index) => (
                  <p key={`${warning.code}-${index}`} role="status" className="mt-1 text-xs text-amber-800">{warning.message}</p>
                ))}
              </li>
            })}
          </ul>}
      {overlayLoading && <p role="status" className="mt-3 text-xs text-slate-600">{s('indicatorParameters.calculating')}</p>}
      {overlay?.key === overlayKey && overlay.error && (
        <p role="alert" className="mt-3 rounded-lg bg-rose-50 px-4 py-2 text-xs leading-5 text-rose-700">{overlay.error}</p>
      )}
    </div>
  </section>
}

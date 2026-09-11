import { useEffect, useMemo, useRef, useState } from 'react'
import ReactECharts from 'echarts-for-react'
import type { EChartsOption } from 'echarts'
import {
  compareResearchSeries,
  getResearchSeriesProfile,
  listResearchSeries,
  type ResearchSeriesCatalogItem,
  type ResearchSeriesComparison,
  type ResearchSeriesCompareSource,
  type ResearchSeriesCatalogResponse,
  type ResearchInlineRow,
  type ResearchSeriesProfile,
} from '../services/researchSeries'

type KindFilter = 'all' | 'index' | 'macro' | 'indicator' | 'upload'
type StatusFilter = 'available' | 'not_downloaded' | 'all'

export interface ResearchDataLabProps {
  embedded?: boolean
  boundSeriesIds?: string[]
  onBindSeries?: (series: ResearchSeriesCatalogItem) => void
  onClose?: () => void
}

const KIND_OPTIONS: Array<{ id: KindFilter; label: string }> = [
  { id: 'all', label: '全部' },
  { id: 'index', label: '指数' },
  { id: 'macro', label: '宏观' },
  { id: 'indicator', label: '指标' },
  { id: 'upload', label: '上传' },
]

const MAX_UPLOAD_ROWS = 20_000
type UploadFrequency = 'daily' | 'weekly' | 'monthly' | 'quarterly' | 'annual' | 'irregular'
type AvailabilityMode = 'point_in_time' | 'latest'

interface ComparisonSelection {
  key: string
  label: string
  source: ResearchSeriesCompareSource
}

function validIsoDate(value: string) {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(value)) return false
  const date = new Date(`${value}T00:00:00Z`)
  return Number.isFinite(date.getTime()) && date.toISOString().slice(0, 10) === value
}

function parseCsvCells(source: string) {
  const rows: string[][] = []
  let row: string[] = []
  let cell = ''
  let quoted = false
  for (let index = 0; index < source.length; index += 1) {
    const character = source[index]
    if (character === '"') {
      if (quoted && source[index + 1] === '"') { cell += '"'; index += 1 } else quoted = !quoted
    } else if (character === ',' && !quoted) {
      row.push(cell); cell = ''
    } else if ((character === '\n' || character === '\r') && !quoted) {
      if (character === '\r' && source[index + 1] === '\n') index += 1
      row.push(cell); cell = ''
      if (row.some((item) => item.trim())) rows.push(row)
      row = []
    } else cell += character
  }
  if (quoted) throw new Error('CSV 存在未闭合的引号。')
  row.push(cell)
  if (row.some((item) => item.trim())) rows.push(row)
  return rows
}

function normalizeUploadRow(raw: Record<string, unknown>, index: number): ResearchInlineRow {
  const date = String(raw.date ?? raw.observation_date ?? '').trim()
  if (!validIsoDate(date)) throw new Error(`第 ${index + 1} 行 date 必须是有效的 YYYY-MM-DD。`)
  if (!Object.prototype.hasOwnProperty.call(raw, 'value')) throw new Error(`第 ${index + 1} 行缺少 value。`)
  const sourceValue = raw.value
  let value: number | null
  if (sourceValue == null || (typeof sourceValue === 'string' && ['', 'null'].includes(sourceValue.trim().toLowerCase()))) value = null
  else {
    if (typeof sourceValue === 'boolean') throw new Error(`第 ${index + 1} 行 value 必须是数值或 null。`)
    value = Number(sourceValue)
    if (!Number.isFinite(value)) throw new Error(`第 ${index + 1} 行 value 必须是有限数值，缺失请使用 null。`)
  }
  const availableAt = raw.available_at == null || String(raw.available_at).trim() === '' ? undefined : String(raw.available_at).trim()
  if (availableAt && !validIsoDate(availableAt)) throw new Error(`第 ${index + 1} 行 available_at 必须是有效的 YYYY-MM-DD。`)
  if (availableAt && availableAt < date) throw new Error(`第 ${index + 1} 行 available_at 不能早于 date。`)
  const vintage = raw.vintage == null || String(raw.vintage).trim() === '' ? undefined : String(raw.vintage).trim()
  if (vintage && vintage.length > 160) throw new Error(`第 ${index + 1} 行 vintage 不能超过 160 个字符。`)
  let revision: number | undefined
  if (raw.revision != null && String(raw.revision).trim() !== '') {
    revision = Number(raw.revision)
    if (!Number.isInteger(revision) || revision < 1) throw new Error(`第 ${index + 1} 行 revision 必须是正整数。`)
  }
  return { date, value, ...(availableAt ? { available_at: availableAt } : {}), ...(vintage ? { vintage } : {}), ...(revision ? { revision } : {}) }
}

export function parseResearchUpload(fileName: string, source: string): ResearchInlineRow[] {
  const normalizedName = fileName.toLowerCase()
  let rows: unknown
  if (normalizedName.endsWith('.json')) {
    const payload = JSON.parse(source) as unknown
    rows = Array.isArray(payload) ? payload : (payload && typeof payload === 'object' ? (payload as { rows?: unknown }).rows : undefined)
  } else if (normalizedName.endsWith('.csv')) {
    const table = parseCsvCells(source)
    if (!table.length) throw new Error('CSV 文件为空。')
    const headers = table[0].map((item, index) => (index === 0 ? item.replace(/^\uFEFF/, '') : item).trim().toLowerCase())
    if (!headers.includes('date') && !headers.includes('observation_date')) throw new Error('CSV 必须包含 date 列。')
    if (!headers.includes('value')) throw new Error('CSV 必须包含 value 列。')
    rows = table.slice(1).map((cells) => Object.fromEntries(headers.map((header, index) => [header, cells[index] ?? ''])))
  } else throw new Error('仅支持 .csv 或 .json 文件。')
  if (!Array.isArray(rows) || !rows.length) throw new Error('上传文件至少需要一条数据。')
  if (rows.length > MAX_UPLOAD_ROWS) throw new Error(`上传文件最多允许 ${MAX_UPLOAD_ROWS.toLocaleString('zh-CN')} 行。`)
  return rows.map((row, index) => {
    if (!row || typeof row !== 'object' || Array.isArray(row)) throw new Error(`第 ${index + 1} 行必须是对象。`)
    return normalizeUploadRow(row as Record<string, unknown>, index)
  })
}

function readUploadText(file: File) {
  return new Promise<string>((resolve, reject) => {
    const reader = new FileReader()
    reader.onerror = () => reject(new Error('文件读取失败，请重新选择。'))
    reader.onload = () => resolve(typeof reader.result === 'string' ? reader.result : '')
    reader.readAsText(file, 'utf-8')
  })
}

function fieldId(field: string | { id?: string; name?: string }) {
  return typeof field === 'string' ? field : field.id || field.name || ''
}

function fieldLabel(field: string | { id?: string; name?: string; label?: string }) {
  return typeof field === 'string' ? field : field.label || field.name || field.id || ''
}

function percentage(value?: number | null) {
  return value == null ? '—' : `${(value * 100).toFixed(2)}%`
}

function compactNumber(value?: number | null) {
  if (value == null || !Number.isFinite(value)) return '—'
  return new Intl.NumberFormat('zh-CN', { maximumFractionDigits: 4 }).format(value)
}

function LabHeader({ embedded, onClose }: Pick<ResearchDataLabProps, 'embedded' | 'onClose'>) {
  return (
    <header className="flex flex-col gap-3 border-b border-slate-200 bg-white px-4 py-4 sm:flex-row sm:items-center sm:justify-between sm:px-5">
      <div>
        <p className="text-[11px] font-bold uppercase tracking-[0.18em] text-indigo-600">Research data lab</p>
        <h2 className="mt-1 text-xl font-bold text-slate-950">研究数据实验室</h2>
        <p className="mt-1 text-xs leading-5 text-slate-500">只展示接口返回的真实数据画像；指数、宏观和指标均保留时点与版本信息。</p>
      </div>
      {embedded && onClose ? <button type="button" onClick={onClose} className="min-h-10 rounded-xl border border-slate-300 px-4 text-sm font-bold text-slate-700 hover:border-indigo-300 hover:text-indigo-700">返回计算图</button> : null}
    </header>
  )
}

function CatalogCard({
  item,
  selected,
  onSelect,
}: {
  item: ResearchSeriesCatalogItem
  selected: boolean
  onSelect: () => void
}) {
  const available = item.status !== 'not_downloaded'
  return (
    <button
      type="button"
      onClick={onSelect}
      aria-pressed={selected}
      className={`w-full rounded-xl border p-3 text-left transition focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 ${selected ? 'border-indigo-500 bg-indigo-50 shadow-sm' : 'border-slate-200 bg-white hover:border-indigo-200'}`}
    >
      <span className="flex items-start justify-between gap-2">
        <span className="min-w-0">
          <span className="block truncate text-sm font-bold text-slate-900">{item.name}</span>
          <span className="mt-0.5 block truncate text-[11px] text-slate-500">{item.code || item.id}</span>
        </span>
        <span className={`shrink-0 rounded-full px-2 py-1 text-[10px] font-bold ${available ? 'bg-emerald-100 text-emerald-800' : 'bg-amber-100 text-amber-900'}`}>{available ? '可研究' : '尚未下载'}</span>
      </span>
      <span className="mt-3 grid grid-cols-2 gap-2 text-[11px] text-slate-500">
        <span>{item.frequency || '频率未知'} · {item.default_field || '默认字段未知'}</span>
        <span className="text-right">{item.coverage?.observations == null ? '样本未知' : `${item.coverage.observations} 条`}</span>
        <span>{item.coverage?.start_date || item.coverage?.first_date || '—'}</span>
        <span className="text-right">至 {item.coverage?.end_date || item.coverage?.last_date || '—'}</span>
      </span>
    </button>
  )
}

function UploadPanel({
  rows,
  fileName,
  name,
  frequency,
  availabilityMode,
  parsing,
  onRows,
  onName,
  onFrequency,
  onAvailabilityMode,
  onParsing,
  onError,
}: {
  rows: ResearchInlineRow[]
  fileName: string
  name: string
  frequency: UploadFrequency
  availabilityMode: AvailabilityMode
  parsing: boolean
  onRows: (rows: ResearchInlineRow[], fileName: string) => void
  onName: (value: string) => void
  onFrequency: (value: UploadFrequency) => void
  onAvailabilityMode: (value: AvailabilityMode) => void
  onParsing: (value: boolean) => void
  onError: (value: string) => void
}) {
  const handleFile = async (file?: File) => {
    if (!file) return
    onParsing(true); onError('')
    try {
      const parsed = parseResearchUpload(file.name, await readUploadText(file))
      onRows(parsed, file.name)
      if (!name.trim()) onName(file.name.replace(/\.(csv|json)$/i, ''))
    } catch (reason) {
      onRows([], '')
      onError(reason instanceof Error ? reason.message : '上传文件解析失败。')
    } finally { onParsing(false) }
  }
  return <section className="mt-4 rounded-xl border border-indigo-200 bg-indigo-50/60 p-3" aria-label="上传研究时序">
    <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between"><div><h4 className="text-sm font-bold text-indigo-950">上传 CSV / JSON 时序</h4><p className="mt-1 text-[11px] leading-5 text-indigo-900/70">必填 date、value；可选 available_at、vintage、revision。前端先校验，后端仍执行最终校验与完整样本 NJIT 画像。</p></div><label className="inline-flex min-h-10 cursor-pointer items-center justify-center rounded-lg bg-indigo-600 px-3 text-xs font-bold text-white">{parsing ? '解析中…' : '选择文件'}<input aria-label="选择研究时序文件" type="file" accept=".csv,.json,text/csv,application/json" disabled={parsing} onChange={(event) => void handleFile(event.target.files?.[0])} className="sr-only" /></label></div>
    <div className="mt-3 grid gap-3 sm:grid-cols-3">
      <label className="text-xs font-bold text-slate-600">序列名称<input aria-label="上传序列名称" value={name} onChange={(event) => onName(event.target.value)} maxLength={160} className="mt-1 min-h-10 w-full rounded-lg border border-slate-300 bg-white px-2 font-normal" /></label>
      <label className="text-xs font-bold text-slate-600">频率<select aria-label="上传序列频率" value={frequency} onChange={(event) => onFrequency(event.target.value as UploadFrequency)} className="mt-1 min-h-10 w-full rounded-lg border border-slate-300 bg-white px-2 font-normal"><option value="daily">日频</option><option value="weekly">周频</option><option value="monthly">月频</option><option value="quarterly">季频</option><option value="annual">年频</option><option value="irregular">不定期</option></select></label>
      <label className="text-xs font-bold text-slate-600">可得性口径<select aria-label="上传可得性口径" value={availabilityMode} onChange={(event) => onAvailabilityMode(event.target.value as AvailabilityMode)} className="mt-1 min-h-10 w-full rounded-lg border border-slate-300 bg-white px-2 font-normal"><option value="point_in_time">按当时可得</option><option value="latest">使用最新修订</option></select></label>
    </div>
    {rows.length ? <div className="mt-3 rounded-lg border border-emerald-200 bg-white p-3"><p role="status" className="text-xs font-bold text-emerald-800">已解析 {rows.length.toLocaleString('zh-CN')} 行 · {fileName}</p><div className="mt-2 overflow-x-auto"><table className="min-w-full text-left text-[10px]" aria-label="上传数据预览"><thead className="text-slate-500"><tr><th className="pr-4">date</th><th className="pr-4">value</th><th className="pr-4">available_at</th><th>vintage / revision</th></tr></thead><tbody>{rows.slice(0, 3).map((row, index) => <tr key={`${row.date || row.observation_date}-${index}`} className="border-t border-slate-100"><td className="py-1 pr-4">{row.date || row.observation_date}</td><td className="pr-4">{row.value == null ? 'null' : row.value}</td><td className="pr-4">{row.available_at || '同 date'}</td><td>{row.vintage || '—'} / {row.revision || 1}</td></tr>)}</tbody></table></div></div> : <p className="mt-3 rounded-lg border border-dashed border-indigo-200 bg-white/70 px-3 py-2 text-[11px] text-slate-600">单次最多 20,000 行；文件只随本次画像请求传输，不会在浏览器伪造数据或宣称已经持久化。</p>}
  </section>
}

function ProfileSummary({ profile }: { profile: ResearchSeriesProfile }) {
  const stats = [
    ['完整样本', profile.coverage?.observations == null ? '—' : String(profile.coverage.observations)],
    ['缺失比例', percentage(profile.missing?.rate ?? profile.missing?.ratio)],
    ['返回点数', String(profile.sampling?.displayed_observations ?? profile.sampling?.returned_observations ?? profile.dates.length)],
    ['抽样方式', profile.sampling?.method || '未说明'],
  ]
  return <dl className="grid grid-cols-2 gap-2 lg:grid-cols-4">{stats.map(([label, value]) => <div key={label} className="rounded-xl border border-slate-200 bg-slate-50 p-3"><dt className="text-[11px] font-semibold text-slate-500">{label}</dt><dd className="mt-1 text-sm font-bold text-slate-900">{value}</dd></div>)}</dl>
}

function ProfileTable({ profile }: { profile: ResearchSeriesProfile }) {
  const metrics = Object.keys(profile.values)
  const hasAvailableAt = Array.isArray(profile.pit?.available_at) && profile.pit.available_at.length > 0
  const vintageValues = profile.vintage && typeof profile.vintage === 'object' && Array.isArray(profile.vintage.values)
    ? profile.vintage.values
    : []
  const hasVintage = vintageValues.length > 0
  const columns = [...metrics, ...(hasAvailableAt ? ['available_at'] : []), ...(hasVintage ? ['vintage'] : [])]
  const rows = profile.dates.map((date, index) => ({ date, values: [...metrics.map((metric) => profile.values[metric]?.[index] ?? null), ...(hasAvailableAt ? [profile.pit?.available_at?.[index] ?? null] : []), ...(hasVintage ? [vintageValues[index] ?? null] : [])] }))
  return (
    <section aria-label="研究序列数据表" className="rounded-xl border border-slate-200 bg-white">
      <div className="border-b border-slate-200 px-4 py-3"><h4 className="text-sm font-bold text-slate-900">抽样数据表</h4><p className="mt-1 text-[11px] text-slate-500">统计使用完整样本，表格仅展示后端返回的等距抽样点。</p></div>
      <div className="space-y-2 p-3 md:hidden" data-testid="research-series-mobile-list">
        {rows.slice(0, 80).map((row) => <article key={row.date} className="rounded-lg bg-slate-50 p-3"><p className="text-xs font-bold text-slate-800">{row.date}</p><dl className="mt-2 grid grid-cols-2 gap-2">{columns.map((metric, index) => <div key={metric}><dt className="truncate text-[10px] text-slate-500">{metric}</dt><dd className="text-xs font-semibold text-slate-800">{typeof row.values[index] === 'number' || row.values[index] == null ? compactNumber(row.values[index] as number | null) : String(row.values[index])}</dd></div>)}</dl></article>)}
      </div>
      <div className="hidden max-h-80 overflow-auto md:block">
        <table className="min-w-full text-left text-xs" aria-label="研究序列抽样数据">
          <thead className="sticky top-0 bg-slate-50 text-slate-500"><tr><th className="px-3 py-2 font-semibold">日期</th>{columns.map((metric) => <th key={metric} className="px-3 py-2 text-right font-semibold">{metric}</th>)}</tr></thead>
          <tbody>{rows.map((row) => <tr key={row.date} className="border-t border-slate-100"><td className="whitespace-nowrap px-3 py-2 font-medium text-slate-700">{row.date}</td>{row.values.map((value, index) => <td key={`${row.date}-${columns[index]}`} className="px-3 py-2 text-right tabular-nums text-slate-700">{typeof value === 'number' || value == null ? compactNumber(value as number | null) : String(value)}</td>)}</tr>)}</tbody>
        </table>
      </div>
    </section>
  )
}

function ProfileWorkspace({
  profile,
  activeMetric,
  onMetric,
}: {
  profile: ResearchSeriesProfile
  activeMetric: string
  onMetric: (metric: string) => void
}) {
  const metrics = Object.keys(profile.values)
  const metric = metrics.includes(activeMetric) ? activeMetric : metrics[0]
  const distribution = metric ? profile.distribution?.[metric] : undefined
  const option = useMemo<EChartsOption>(() => ({
    animation: false,
    tooltip: { trigger: 'axis' },
    grid: { left: 62, right: 22, top: 24, bottom: 52 },
    xAxis: { type: 'category', data: profile.dates, axisLabel: { hideOverlap: true } },
    yAxis: { type: 'value', scale: true },
    dataZoom: [{ type: 'inside' }, { type: 'slider', height: 18, bottom: 8 }],
    series: metric ? [{ name: metric, type: 'line', showSymbol: false, connectNulls: false, data: profile.values[metric], lineStyle: { color: '#4f46e5', width: 2 } }] : [],
  }), [metric, profile])

  return <div className="space-y-4"><ProfileSummary profile={profile} />{metrics.length ? <section className="rounded-xl border border-slate-200 bg-white p-3" aria-label="研究序列图形"><div className="mb-2 flex flex-wrap items-center justify-between gap-2"><h4 className="text-sm font-bold text-slate-900">时序表现</h4><label className="text-xs font-semibold text-slate-600">观察维度 <select aria-label="观察维度" value={metric} onChange={(event) => onMetric(event.target.value)} className="ml-2 min-h-9 rounded-lg border border-slate-300 bg-white px-2 font-normal">{metrics.map((item) => <option key={item} value={item}>{item}</option>)}</select></label></div><ReactECharts option={option} style={{ height: 300 }} notMerge lazyUpdate aria-label={`${profile.series.name}时序图`} />{distribution ? <dl className="mt-2 grid grid-cols-3 gap-2 border-t border-slate-100 pt-3 sm:grid-cols-6">{([['均值', distribution.mean], ['标准差', distribution.std], ['P05', distribution.p05], ['中位数', distribution.median], ['P95', distribution.p95], ['有效样本', distribution.valid_count]] as Array<[string, number | null | undefined]>).map(([label, value]) => <div key={label}><dt className="text-[10px] text-slate-500">{label}</dt><dd className="mt-1 text-xs font-bold text-slate-800">{compactNumber(value)}</dd></div>)}</dl> : null}</section> : <p className="rounded-xl border border-amber-200 bg-amber-50 p-4 text-sm text-amber-900">接口没有返回可绘制的数值列。</p>}<ProfileTable profile={profile} /><p className="rounded-xl border border-emerald-200 bg-emerald-50 px-3 py-2 text-xs font-semibold text-emerald-900">本次统计的固定签名 NJIT 执行证明已通过校验。</p></div>
}

function MultiSeriesComparison({
  selections,
  comparison,
  loading,
  onRemove,
  onRun,
}: {
  selections: ComparisonSelection[]
  comparison: ResearchSeriesComparison | null
  loading: boolean
  onRemove: (key: string) => void
  onRun: () => void
}) {
  const [pairIndex, setPairIndex] = useState(0)
  useEffect(() => { setPairIndex(0) }, [comparison])
  const labels = useMemo(() => new Map(comparison?.series.map((item) => [item.id, item.label]) ?? []), [comparison])
  const overlay = useMemo<EChartsOption | null>(() => comparison ? ({
    animation: false,
    tooltip: { trigger: 'axis' },
    legend: { type: 'scroll', top: 0 },
    grid: { left: 58, right: 22, top: 44, bottom: 52 },
    xAxis: { type: 'category', data: comparison.dates, axisLabel: { hideOverlap: true } },
    yAxis: { type: 'value', name: '标准分', scale: true },
    dataZoom: [{ type: 'inside' }, { type: 'slider', height: 18, bottom: 8 }],
    series: comparison.series.map((item) => ({ name: item.label, type: 'line', showSymbol: false, connectNulls: false, data: item.standardized })),
  }) : null, [comparison])
  const pair = comparison?.scatter_pairs[pairIndex]
  const scatter = useMemo<EChartsOption | null>(() => pair ? ({
    animation: false,
    tooltip: { trigger: 'item', formatter: (params: unknown) => {
      const value = (params as { value?: unknown[] }).value ?? []
      return `${labels.get(pair.left_id) || pair.left_id}: ${value[0] ?? '—'}<br/>${labels.get(pair.right_id) || pair.right_id}: ${value[1] ?? '—'}`
    } },
    grid: { left: 62, right: 20, top: 26, bottom: 48 },
    xAxis: { type: 'value', name: labels.get(pair.left_id) || pair.left_id, scale: true },
    yAxis: { type: 'value', name: labels.get(pair.right_id) || pair.right_id, scale: true },
    dataZoom: [{ type: 'inside' }],
    series: [{ type: 'scatter', symbolSize: 7, data: pair.standardized_x.map((value, index) => [value, pair.standardized_y[index]]) }],
  }) : null, [labels, pair])

  return <section className="rounded-xl border border-slate-200 bg-white p-4" aria-label="多序列研究比较">
    <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between"><div><h3 className="text-sm font-bold text-slate-950">多序列研究比较</h3><p className="mt-1 text-[11px] leading-5 text-slate-500">选择 2–4 条序列。共同日期、标准化、相关性与散点样本全部由后端固定签名 NJIT 计算。</p></div><button type="button" disabled={selections.length < 2 || loading} onClick={onRun} className="min-h-10 rounded-xl bg-slate-950 px-4 text-xs font-bold text-white disabled:opacity-40">{loading ? '正在比较…' : `比较 ${selections.length} 条序列`}</button></div>
    <div className="mt-3 flex flex-wrap gap-2">{selections.map((item) => <span key={item.key} className="inline-flex items-center gap-2 rounded-full bg-indigo-50 px-3 py-1.5 text-[11px] font-bold text-indigo-900">{item.label}<button type="button" aria-label={`移除对比${item.label}`} onClick={() => onRemove(item.key)} className="text-indigo-500 hover:text-rose-600">×</button></span>)}{!selections.length ? <span className="rounded-lg border border-dashed border-slate-300 px-3 py-2 text-xs text-slate-500">从上方当前数据资源加入对比。</span> : null}</div>
    {comparison ? <div className="mt-4 space-y-4">
      <dl className="grid grid-cols-2 gap-2 sm:grid-cols-4"><div className="rounded-lg bg-slate-50 p-3"><dt className="text-[10px] text-slate-500">共同日期</dt><dd className="mt-1 text-sm font-bold text-slate-900">{comparison.alignment.intersected_observations}</dd></div><div className="rounded-lg bg-slate-50 p-3"><dt className="text-[10px] text-slate-500">全部有效</dt><dd className="mt-1 text-sm font-bold text-slate-900">{comparison.common_valid.observation_count}</dd></div><div className="rounded-lg bg-slate-50 p-3"><dt className="text-[10px] text-slate-500">共同有效起点</dt><dd className="mt-1 text-xs font-bold text-slate-900">{comparison.common_valid.start_date || '—'}</dd></div><div className="rounded-lg bg-slate-50 p-3"><dt className="text-[10px] text-slate-500">共同有效终点</dt><dd className="mt-1 text-xs font-bold text-slate-900">{comparison.common_valid.end_date || '—'}</dd></div></dl>
      {overlay ? <section className="rounded-xl border border-slate-200 p-3"><h4 className="text-xs font-bold text-slate-800">标准化时序叠加</h4><ReactECharts option={overlay} style={{ height: 300 }} notMerge lazyUpdate aria-label="多序列标准化叠加图" /></section> : null}
      <div className="grid gap-4 xl:grid-cols-2"><section className="min-w-0 rounded-xl border border-slate-200 p-3"><h4 className="text-xs font-bold text-slate-800">相关性矩阵</h4><div className="mt-2 hidden overflow-x-auto sm:block"><table className="min-w-full text-xs" aria-label="多序列相关性矩阵"><thead><tr><th className="p-2 text-left text-slate-500">序列</th>{comparison.series.map((item) => <th key={item.id} className="p-2 text-right text-slate-500">{item.label}</th>)}</tr></thead><tbody>{comparison.series.map((row, rowIndex) => <tr key={row.id} className="border-t border-slate-100"><th className="p-2 text-left font-semibold text-slate-700">{row.label}</th>{comparison.correlation.matrix[rowIndex].map((value, columnIndex) => <td key={`${row.id}-${columnIndex}`} className="p-2 text-right tabular-nums text-slate-700" title={`共同有效 ${comparison.correlation.observation_counts[rowIndex]?.[columnIndex] ?? 0} 条`}>{compactNumber(value)}</td>)}</tr>)}</tbody></table></div><div className="mt-2 space-y-2 sm:hidden" data-testid="correlation-mobile-list">{comparison.series.flatMap((row, rowIndex) => comparison.series.slice(rowIndex + 1).map((column, offset) => { const columnIndex = rowIndex + offset + 1; return <div key={`${row.id}-${column.id}`} className="flex items-center justify-between rounded-lg bg-slate-50 p-2 text-xs"><span>{row.label} / {column.label}</span><strong>{compactNumber(comparison.correlation.matrix[rowIndex]?.[columnIndex])}</strong></div> }))}</div></section>
      <section className="min-w-0 rounded-xl border border-slate-200 p-3"><div className="flex flex-wrap items-center justify-between gap-2"><h4 className="text-xs font-bold text-slate-800">两两散点</h4><select aria-label="散点序列组合" value={pairIndex} onChange={(event) => setPairIndex(Number(event.target.value))} className="min-h-9 rounded-lg border border-slate-300 bg-white px-2 text-xs">{comparison.scatter_pairs.map((item, index) => <option key={`${item.left_id}-${item.right_id}`} value={index}>{labels.get(item.left_id) || item.left_id} / {labels.get(item.right_id) || item.right_id}</option>)}</select></div>{scatter ? <ReactECharts option={scatter} style={{ height: 300 }} notMerge lazyUpdate aria-label="多序列散点图" /> : <p className="mt-4 text-xs text-slate-500">后端没有返回可绘制的配对。</p>}{pair ? <p className="text-[10px] text-slate-500">相关系数 {compactNumber(pair.correlation)} · 有效配对 {pair.observation_count} 条</p> : null}</section></div>
      <p className="rounded-lg border border-emerald-200 bg-emerald-50 px-3 py-2 text-xs font-semibold text-emerald-900">比较统计的固定签名 NJIT 执行证明已通过校验；浏览器仅编排展示。</p>
    </div> : null}
  </section>
}

export default function ResearchDataLab({ embedded = false, boundSeriesIds = [], onBindSeries, onClose }: ResearchDataLabProps) {
  const [query, setQuery] = useState('')
  const [kind, setKind] = useState<KindFilter>('all')
  const [status, setStatus] = useState<StatusFilter>('available')
  const [catalog, setCatalog] = useState<ResearchSeriesCatalogResponse | null>(null)
  const [selectedId, setSelectedId] = useState('')
  const [field, setField] = useState('')
  const [startDate, setStartDate] = useState('')
  const [endDate, setEndDate] = useState('')
  const [asOf, setAsOf] = useState('')
  const [vintage, setVintage] = useState('')
  const [rollingWindow, setRollingWindow] = useState(20)
  const [uploadRows, setUploadRows] = useState<ResearchInlineRow[]>([])
  const [uploadFileName, setUploadFileName] = useState('')
  const [uploadName, setUploadName] = useState('')
  const [uploadFrequency, setUploadFrequency] = useState<UploadFrequency>('daily')
  const [uploadAvailabilityMode, setUploadAvailabilityMode] = useState<AvailabilityMode>('point_in_time')
  const [uploadParsing, setUploadParsing] = useState(false)
  const [locallyBound, setLocallyBound] = useState<string[]>([])
  const [profile, setProfile] = useState<ResearchSeriesProfile | null>(null)
  const [activeMetric, setActiveMetric] = useState('raw')
  const [comparisonSelections, setComparisonSelections] = useState<ComparisonSelection[]>([])
  const [comparison, setComparison] = useState<ResearchSeriesComparison | null>(null)
  const [comparing, setComparing] = useState(false)
  const [loading, setLoading] = useState(true)
  const [profiling, setProfiling] = useState(false)
  const [error, setError] = useState('')
  const catalogRequest = useRef(0)
  const profileRequest = useRef(0)
  const comparisonRequest = useRef(0)
  const comparisonSequence = useRef(1)

  useEffect(() => {
    const requestId = ++catalogRequest.current
    const controller = new AbortController()
    const timer = window.setTimeout(() => {
      setLoading(true); setError('')
      void listResearchSeries({ query, kind, status }, controller.signal)
        .then((response) => {
          if (requestId !== catalogRequest.current) return
          setCatalog(response)
          setSelectedId((current) => response.items.some((item) => item.id === current) ? current : response.items[0]?.id || '')
        })
        .catch((reason) => {
          if (controller.signal.aborted || requestId !== catalogRequest.current) return
          setError(reason instanceof Error ? reason.message : '研究数据目录加载失败。')
        })
        .finally(() => { if (requestId === catalogRequest.current) setLoading(false) })
    }, 180)
    return () => { window.clearTimeout(timer); controller.abort() }
  }, [kind, query, status])

  const selected = catalog?.items.find((item) => item.id === selectedId) ?? null
  useEffect(() => {
    const nextField = selected?.default_field || (selected?.fields?.[0] ? fieldId(selected.fields[0]) : '')
    setField(nextField)
    setVintage(selected?.vintage?.default || '')
    setProfile(null)
  }, [selectedId]) // reset controls only when the selected series changes

  const requestProfile = async () => {
    if (!selected || selected.status === 'not_downloaded') return
    const isUpload = selected.kind === 'upload'
    if (isUpload && !uploadRows.length) { setError('请先选择并校验 CSV 或 JSON 文件。'); return }
    if (isUpload && !uploadName.trim()) { setError('请填写上传序列名称。'); return }
    const requestId = ++profileRequest.current
    const controller = new AbortController()
    setProfiling(true); setError(''); setProfile(null)
    try {
      const response = await getResearchSeriesProfile({
        series_id: selected.id,
        field: field || undefined,
        ...(isUpload ? { inline_rows: uploadRows, name: uploadName.trim(), frequency: uploadFrequency, availability_mode: uploadAvailabilityMode, register_artifact: true } : {}),
        start_date: startDate || undefined,
        end_date: endDate || undefined,
        as_of: asOf || undefined,
        vintage: vintage || undefined,
        rolling_window: rollingWindow,
        sample_limit: 500,
      }, controller.signal)
      if (requestId !== profileRequest.current) return
      setProfile(response)
      setActiveMetric(Object.keys(response.values)[0] || 'raw')
    } catch (reason) {
      if (requestId === profileRequest.current) setError(reason instanceof Error ? reason.message : '研究序列画像计算失败。')
    } finally {
      if (requestId === profileRequest.current) setProfiling(false)
    }
  }

  const isUpload = selected?.kind === 'upload'
  const bindCandidate = selected && isUpload && profile?.regime_node_type && profile.binding_parameters
    ? { ...selected, ...profile.series, id: selected.id, regime_node_type: profile.regime_node_type, binding_parameters: profile.binding_parameters }
    : selected
  const bindingIdentity = String(bindCandidate?.binding_parameters?.artifact_id || profile?.binding?.fingerprint || bindCandidate?.id || '')
  const canBind = Boolean(bindCandidate && bindCandidate.status !== 'not_downloaded' && bindCandidate.regime_node_type && bindCandidate.binding_parameters && (!isUpload || profile?.binding_parameters))
  const isBound = Boolean(bindCandidate && (boundSeriesIds.includes(bindCandidate.id) || locallyBound.includes(bindingIdentity)))
  const comparisonKey = selected ? `${selected.id}::${field || selected.default_field || 'value'}::${isUpload ? String(profile?.binding_parameters?.artifact_id || '') : ''}` : ''
  const canAddComparison = Boolean(selected && selected.status !== 'not_downloaded' && (!isUpload || profile?.binding_parameters?.artifact_id))

  const addComparison = () => {
    if (!selected || !canAddComparison) return
    if (comparisonSelections.some((item) => item.key === comparisonKey)) { setError('该序列与字段已经在对比篮中。'); return }
    if (comparisonSelections.length >= 4) { setError('一次最多比较 4 条序列。'); return }
    const source: ResearchSeriesCompareSource = isUpload ? {
      id: `source-${comparisonSequence.current++}`,
      label: uploadName.trim() || profile?.series.name || selected.name,
      series_id: selected.id,
      artifact_id: String(profile?.binding_parameters?.artifact_id || ''),
      checksum: String(profile?.binding_parameters?.checksum || ''),
      field: String(profile?.binding_parameters?.value_field || 'value'),
      start_date: startDate || undefined,
      end_date: endDate || undefined,
      as_of: asOf || undefined,
      sample_limit: 500,
    } : {
      id: `source-${comparisonSequence.current++}`,
      label: `${selected.name}${field ? ` · ${field}` : ''}`,
      series_id: selected.id,
      field: field || undefined,
      start_date: startDate || undefined,
      end_date: endDate || undefined,
      as_of: asOf || undefined,
      vintage: vintage || undefined,
      rolling_window: rollingWindow,
      sample_limit: 500,
    }
    setComparisonSelections((current) => [...current, { key: comparisonKey, label: source.label || selected.name, source }])
    setComparison(null); setError('')
  }

  const runComparison = async () => {
    if (comparisonSelections.length < 2) return
    const requestId = ++comparisonRequest.current
    setComparing(true); setError(''); setComparison(null)
    try {
      const response = await compareResearchSeries(comparisonSelections.map((item) => item.source), 500)
      if (requestId === comparisonRequest.current) setComparison(response)
    } catch (reason) {
      if (requestId === comparisonRequest.current) setError(reason instanceof Error ? reason.message : '多序列比较失败。')
    } finally { if (requestId === comparisonRequest.current) setComparing(false) }
  }

  return (
    <div className={`${embedded ? 'min-h-0' : 'min-h-[720px] rounded-2xl border border-slate-200'} overflow-hidden bg-slate-50 shadow-sm`} data-testid="research-data-lab">
      <LabHeader embedded={embedded} onClose={onClose} />
      <div className="grid min-h-0 gap-4 p-3 sm:p-4 xl:grid-cols-[minmax(280px,0.72fr)_minmax(0,1.28fr)]">
        <aside className="min-w-0 space-y-3 rounded-xl border border-slate-200 bg-white p-3" aria-label="研究数据目录">
          <label className="block text-xs font-bold text-slate-600">搜索数据
            <input aria-label="搜索研究数据" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="指数代码、宏观指标或自定义指标" className="mt-1 min-h-10 w-full rounded-lg border border-slate-300 px-3 text-sm font-normal focus:border-indigo-500 focus:outline-none" />
          </label>
          <div role="group" aria-label="数据类型" className="flex gap-1 overflow-x-auto pb-1">{KIND_OPTIONS.map((item) => <button key={item.id} type="button" aria-pressed={kind === item.id} onClick={() => setKind(item.id)} className={`min-h-9 shrink-0 rounded-lg px-3 text-xs font-bold ${kind === item.id ? 'bg-slate-950 text-white' : 'bg-slate-100 text-slate-600 hover:bg-slate-200'}`}>{item.label}</button>)}</div>
          <label className="block text-xs font-bold text-slate-600">下载状态 <select aria-label="下载状态" value={status} onChange={(event) => setStatus(event.target.value as StatusFilter)} className="ml-2 min-h-9 rounded-lg border border-slate-300 bg-white px-2 font-normal"><option value="available">仅可用</option><option value="not_downloaded">尚未下载</option><option value="all">全部</option></select></label>
          <div className="flex items-center justify-between text-[11px] text-slate-500"><span>{loading ? '正在读取目录…' : `${catalog?.total ?? 0} 条数据资源`}</span><span>{catalog?.snapshot?.generation || catalog?.snapshot?.directory || catalog?.snapshot?.id || '快照待确认'}</span></div>
          <div className="max-h-[58vh] space-y-2 overflow-y-auto pr-1 xl:max-h-[680px]">{catalog?.items.map((item) => <CatalogCard key={item.id} item={item} selected={item.id === selectedId} onSelect={() => setSelectedId(item.id)} />)}{!loading && !catalog?.items.length ? <p className="rounded-xl border border-dashed border-slate-300 p-6 text-center text-sm text-slate-500">没有匹配的数据资源。</p> : null}</div>
        </aside>

        <main className="min-w-0 space-y-4">
          {selected ? <section className="rounded-xl border border-slate-200 bg-white p-4" aria-label="数据画像参数">
            <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between"><div><p className="text-xs font-bold text-indigo-600">{selected.category_label || selected.kind}</p><h3 className="mt-1 text-lg font-bold text-slate-950">{selected.name}</h3><p className="mt-1 text-xs leading-5 text-slate-500">{selected.description || `${selected.source_api || selected.dataset || '数据快照'} · ${selected.frequency || '频率未知'}`}</p><div className="mt-2 flex flex-wrap gap-1 text-[10px] font-bold"><span className={`rounded-full px-2 py-1 ${selected.pit?.supported ? 'bg-emerald-100 text-emerald-800' : 'bg-slate-100 text-slate-600'}`}>{selected.pit?.supported ? '支持时点约束' : '无时点元数据'}</span><span className={`rounded-full px-2 py-1 ${selected.vintage?.supported ? 'bg-violet-100 text-violet-800' : 'bg-slate-100 text-slate-600'}`}>{selected.vintage?.supported ? '支持数据版本' : '单版本序列'}</span></div></div><div className="flex flex-wrap gap-2"><button type="button" disabled={!canAddComparison || comparisonSelections.some((item) => item.key === comparisonKey)} onClick={addComparison} className="min-h-10 rounded-xl border border-slate-300 px-4 text-sm font-bold text-slate-700 disabled:opacity-40">加入多序列对比</button><button type="button" disabled={!canBind || isBound} onClick={() => { if (bindCandidate && onBindSeries) { onBindSeries(bindCandidate); setLocallyBound((current) => [...current, bindingIdentity]) } }} className="min-h-10 rounded-xl border border-indigo-200 px-4 text-sm font-bold text-indigo-700 disabled:cursor-not-allowed disabled:opacity-40">{isBound ? '已加入计算图' : '加入计算图'}</button><button type="button" disabled={profiling || selected.status === 'not_downloaded' || (isUpload && uploadParsing)} onClick={() => void requestProfile()} className="min-h-10 rounded-xl bg-indigo-600 px-4 text-sm font-bold text-white disabled:opacity-40">{profiling ? '正在计算画像…' : '计算数据画像'}</button></div></div>
            {isUpload ? <UploadPanel rows={uploadRows} fileName={uploadFileName} name={uploadName} frequency={uploadFrequency} availabilityMode={uploadAvailabilityMode} parsing={uploadParsing} onRows={(rows, fileName) => { setUploadRows(rows); setUploadFileName(fileName); setProfile(null) }} onName={setUploadName} onFrequency={setUploadFrequency} onAvailabilityMode={setUploadAvailabilityMode} onParsing={setUploadParsing} onError={setError} /> : null}
            {selected.kind === 'macro' && selected.status === 'not_downloaded' ? <p className="mt-3 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-950">该宏观序列尚未进入活跃数据快照。<a href="/settings/data-sources" className="ml-1 font-bold underline">前往数据源设置下载</a></p> : null}
            {!canBind && onBindSeries ? <p className="mt-3 rounded-lg bg-amber-50 px-3 py-2 text-xs text-amber-900">{isUpload ? '请先上传文件并完成一次真实数据画像；加入计算图时只使用后端返回的不可变绑定。' : '数据目录尚未返回图谱绑定协议，因此暂不能加入计算图；不会猜测节点类型。'}</p> : null}
            <div className="mt-4 grid gap-3 sm:grid-cols-2 2xl:grid-cols-6">
              <label className="text-xs font-bold text-slate-600">字段 <select aria-label="研究字段" value={field} onChange={(event) => setField(event.target.value)} className="mt-1 min-h-10 w-full rounded-lg border border-slate-300 bg-white px-2 font-normal">{(selected.fields || []).map((item) => <option key={fieldId(item)} value={fieldId(item)}>{fieldLabel(item)}</option>)}</select></label>
              <label className="text-xs font-bold text-slate-600">开始日期 <input aria-label="画像开始日期" type="date" value={startDate} onChange={(event) => setStartDate(event.target.value)} className="mt-1 min-h-10 w-full rounded-lg border border-slate-300 px-2 font-normal" /></label>
              <label className="text-xs font-bold text-slate-600">结束日期 <input aria-label="画像结束日期" type="date" value={endDate} onChange={(event) => setEndDate(event.target.value)} className="mt-1 min-h-10 w-full rounded-lg border border-slate-300 px-2 font-normal" /></label>
              <label className="text-xs font-bold text-slate-600">截至日 <input aria-label="画像截至日" type="date" value={asOf} onChange={(event) => setAsOf(event.target.value)} className="mt-1 min-h-10 w-full rounded-lg border border-slate-300 px-2 font-normal" /></label>
              {selected.vintage?.supported ? <label className="text-xs font-bold text-slate-600">数据版本 <input aria-label="数据版本" value={vintage} onChange={(event) => setVintage(event.target.value)} placeholder="留空取截至日最新版" className="mt-1 min-h-10 w-full rounded-lg border border-slate-300 px-2 font-normal" /></label> : null}
              <label className="text-xs font-bold text-slate-600">滚动窗口 <input aria-label="滚动窗口" type="number" min={2} max={1260} value={rollingWindow} onChange={(event) => setRollingWindow(Number(event.target.value))} className="mt-1 min-h-10 w-full rounded-lg border border-slate-300 px-2 font-normal" /></label>
            </div>
          </section> : <section className="grid min-h-64 place-items-center rounded-xl border border-dashed border-slate-300 bg-white p-8 text-center text-sm text-slate-500">请从左侧选择一个真实数据资源。</section>}
          <MultiSeriesComparison selections={comparisonSelections} comparison={comparison} loading={comparing} onRemove={(key) => { setComparisonSelections((current) => current.filter((item) => item.key !== key)); setComparison(null) }} onRun={() => void runComparison()} />
          {error ? <p role="alert" className="rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-900">{error}</p> : null}
          {profiling ? <div role="status" className="grid min-h-60 place-items-center rounded-xl border border-slate-200 bg-white text-sm font-semibold text-slate-600">后端正在使用固定签名 NJIT 计算完整样本画像…</div> : null}
          {profile && !profiling ? <ProfileWorkspace profile={profile} activeMetric={activeMetric} onMetric={setActiveMetric} /> : null}
          {!profile && !profiling && selected ? <section className="grid min-h-60 place-items-center rounded-xl border border-dashed border-slate-300 bg-white p-8 text-center"><div><p className="font-bold text-slate-800">尚未计算数据画像</p><p className="mt-2 text-sm text-slate-500">设置研究区间后点击“计算数据画像”；这里不会生成示例走势。</p></div></section> : null}
        </main>
      </div>
    </div>
  )
}

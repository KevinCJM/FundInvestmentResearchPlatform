import React, { useCallback, useEffect, useId, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import katex from 'katex'
import { useI18n } from '../../i18n/runtime'
import 'katex/dist/katex.min.css'
import { indicatorPeriodOptionLabel } from '../../utils/indicatorPeriods'
import { indicatorDiagnosticDetail } from '../../utils/indicatorDiagnostics'
import {
  type EvaluationResult,
  type EvaluationWarning,
  type IndicatorDefinition,
  type IndicatorDateContext,
  type MetricPresentation,
} from '../../services/customIndicators'

const fallbackPresentation = (indicator?: IndicatorDefinition): MetricPresentation => indicator?.presentation ?? {
  indicator_id: indicator?.id ?? null,
  revision: indicator?.revision ?? null,
  name: indicator?.name ?? '指标',
  source: indicator?.source ?? 'inline',
  indicator_type: indicator?.indicator_type ?? 'other',
  category: indicator?.category_id ?? 'custom',
  category_label: indicator?.category_label ?? '工作区指标',
  context_kind: indicator?.context_kind ?? 'single_product',
  catalog_status: indicator?.catalog_status ?? 'current',
  display_format: indicator?.display_format ?? 'number',
  precision: indicator?.precision ?? 2,
  unit: indicator?.unit ?? '',
  notation: 'standard',
  value_scale: indicator?.display_format === 'percent' ? 100 : 1,
  output_measure: indicator?.output_measure ?? 'dimensionless',
  direction: indicator?.direction ?? 'higher_better',
  description: indicator?.description ?? '',
  methodology: indicator?.methodology ?? indicator?.description ?? '',
  data_basis: indicator?.data_basis ?? '真实数据、严格窗口、缺失不填充',
  minimum_observations: indicator?.minimum_observations ?? 1,
  applicable_product_kinds: indicator?.applicable_product_kinds ?? ['etf', 'fund'],
}

export const resolveMetricPresentation = (
  result?: EvaluationResult | null,
  indicator?: IndicatorDefinition,
) => result?.presentation ?? fallbackPresentation(indicator)

export const formatMetricValue = (
  value: number | string | null | undefined,
  presentation?: MetricPresentation,
) => {
  const contract = presentation ?? fallbackPresentation()
  if (contract.display_format === 'date') {
    if (typeof value !== 'string' || !/^\d{4}-\d{2}-\d{2}$/.test(value)) return '不可计算'
    const parsed = new Date(`${value}T00:00:00Z`)
    return Number.isFinite(parsed.valueOf()) && parsed.toISOString().slice(0, 10) === value ? value : '不可计算'
  }
  if (typeof value !== 'number' || !Number.isFinite(value)) return '不可计算'
  const scaled = value * (contract.value_scale ?? (contract.display_format === 'percent' ? 100 : 1))
  const formatter = new Intl.NumberFormat('zh-CN', {
    notation: contract.notation ?? 'standard',
    minimumFractionDigits: contract.precision,
    maximumFractionDigits: contract.precision,
  })
  const formatted = formatter.format(scaled)
  if (contract.display_format === 'percent') return `${formatted}%`
  return contract.unit ? `${formatted} ${contract.unit}` : formatted
}

export function MetricValue({
  value,
  presentation,
  className = '',
}: {
  value: number | string | null | undefined
  presentation?: MetricPresentation
  className?: string
}) {
  const formatted = formatMetricValue(value, presentation)
  return <span className={`${formatted === '不可计算' ? 'text-slate-600' : 'tabular-nums'} ${className}`}>{formatted}</span>
}

const statusCopy = (
  status: EvaluationResult['status'],
  warnings: EvaluationWarning[],
) => {
  if (status === 'error') return { label: '计算失败', tone: 'bg-rose-50 text-rose-700' }
  if (status === 'unavailable') return { label: '不可计算', tone: 'bg-amber-50 text-amber-700' }
  if (status === 'warning') return { label: warnings.some((item) => item.code.includes('SAMPLE')) ? '样本不足' : '有警告', tone: 'bg-amber-50 text-amber-700' }
  return { label: '正常', tone: 'bg-emerald-50 text-emerald-700' }
}

type AvailabilityResult = Pick<
  EvaluationResult,
  'value' | 'status' | 'warnings' | 'input_requirements' | 'target_data' | 'data_context'
>

const knownVariableLabels: Record<string, string> = {
  adjusted_nav: '复权净值',
  returns: '普通收益率',
  log_returns: '对数收益率',
  market_open: '开盘价',
  market_high: '最高价',
  market_low: '最低价',
  market_close: '收盘价',
  previous_close: '前收盘价',
  price_change: '价格变动额',
  price_return: '行情涨跌幅',
  volume: '成交量',
  turnover_amount: '成交额',
  unit_nav: '单位净值',
  accumulated_nav: '累计净值',
}

export function IndicatorInputDates({ context }: { context?: IndicatorDateContext | null }) {
  if (!context) return null
  return <section aria-label="本次计算的数据与日期" className="mt-3 rounded-lg border border-amber-200 bg-amber-50 p-3 text-left text-sm text-amber-950">
    <h4 className="font-semibold">本次计算的数据与日期</h4>
    <dl className="mt-2 grid gap-2 sm:grid-cols-2">
      <div><dt className="text-xs text-amber-800">产品成立日期</dt><dd>{context.found_date || '数据源未提供'}</dd></div>
      {context.list_date && <div><dt className="text-xs text-amber-800">上市日期</dt><dd>{context.list_date}</dd></div>}
      <div><dt className="text-xs text-amber-800">本次计算截止日（PIT）</dt><dd>{context.as_of || '未设置，使用本地全部日期'}</dd></div>
    </dl>
    {context.sources.map((source, index) => <div key={`${source.label}-${index}`} className="mt-3 border-t border-amber-200 pt-2">
      <p>{source.label}本地覆盖：{source.first_date || '起点未确认'} 至 {source.latest_date || '终点未确认'}</p>
      {context.as_of && <p className="mt-1 text-xs">原有 {source.rows_before_as_of ?? '未确认'} 条记录 → 日期筛选后 {source.rows_after_date_filter ?? '未确认'} 条 → {source.uses_disclosure_date ? '披露筛选并去重后' : '去重后'} {source.rows_after_as_of ?? '未确认'} 条。</p>}
    </div>)}
    <p className="mt-2 text-xs">成立日期、上市日期和本地数据起点是不同概念；产品已经成立，也可能尚无已下载的数据。</p>
    {context.as_of && <>
      <p className="mt-2">PIT 表示站在截止日查看数据：只使用该日及之前的记录；净值还须确认公告日期不晚于截止日。更晚的数据或公告日期缺失的净值不会用于本次计算。</p>
      <p className="mt-2">做历史研究时，请选择当时已有可用数据的产品。若要查看最新表现，可在顶部 PIT 中选择“关闭 PIT · 查看全部磁盘数据”，仅影响当前标签页；预览中的历史截止日如已填写，也需相应调整。</p>
    </>}
  </section>
}

export function MetricUnavailableReason({
  result,
  compact = false,
}: {
  result: AvailabilityResult
  compact?: boolean
}) {
  const requirements = result.input_requirements
  const blocked = requirements?.blocking_inputs ?? []
  const partial = requirements?.partial_inputs ?? []
  const affected = blocked.length ? blocked : partial
  const dateContext = result.value === null ? <IndicatorInputDates context={result.data_context} /> : null
  if (!affected.length) {
    if (result.value !== null) return null
    return <>{dateContext}{result.warnings[0] && <p className="mt-2 text-xs text-amber-700">{indicatorDiagnosticDetail(result.warnings[0].code, result.warnings[0].message)}</p>}</>
  }
  const labels = affected.map((item) => item.label || knownVariableLabels[item.variable_id] || item.variable_id)
  const alternatives = [...new Map(affected.flatMap((item) => item.alternative_variables ?? []).map((item) => [item.variable_id, item.label])).values()]
  const availableFields = (result.target_data?.available_variables ?? [])
    .map((item) => knownVariableLabels[item] ?? item)
    .filter((item, index, items) => items.indexOf(item) === index)
  const summary = blocked.length
    ? `该指标需要 ${requirements?.required_count ?? blocked.length} 个输入字段，当前产品缺少：${labels.join('、')}。`
    : `本次使用的${labels.join('、')}存在部分缺失，已按共同有效日期计算。`
  return <>{dateContext}<div className={`${compact ? 'mt-1' : 'mt-3'} rounded-lg border border-amber-200 bg-amber-50 p-3 text-left text-xs text-amber-950`}>
    <p className="font-medium">{summary}</p>
    <details className="mt-2" open={!compact}>
      <summary className="cursor-pointer font-semibold text-amber-800">{blocked.length ? '为什么无法计算' : '查看数据覆盖情况'}</summary>
      <ul className="mt-2 space-y-1">
        {affected.map((item) => <li key={item.variable_id}><span className="font-medium">{item.label}</span>：{item.reason || (item.status === 'partial' ? '部分日期缺少有效值。' : '当前没有可用数据。')}</li>)}
      </ul>
      {result.target_data && <div className="mt-3 border-t border-amber-200 pt-2">
        <p className="font-semibold">当前已有数据</p>
        {result.target_data.available_datasets.length > 0 && <p className="mt-1">数据源：{result.target_data.available_datasets.join('、')}</p>}
        {availableFields.length > 0 && <p className="mt-1">可用字段：{availableFields.join('、')}</p>}
        <p className="mt-1">数据截至：{result.target_data.data_latest_date || '暂无'}</p>
      </div>}
      {alternatives.length > 0 && <p className="mt-3 border-t border-amber-200 pt-2"><span className="font-semibold">建议：</span>可改用{alternatives.join('、')}构建适用于当前产品的指标。</p>}
      <details className="mt-2">
        <summary className="cursor-pointer text-amber-700">查看技术详情</summary>
        <ul className="mt-1 space-y-1 font-mono text-xs text-amber-800">
          {affected.map((item) => <li key={item.variable_id}>{item.variable_id} · {item.source_dataset || '运行时派生'}{item.source_field ? `.${item.source_field}` : ''} · {item.reason_code || item.status}</li>)}
        </ul>
      </details>
    </details>
  </div></>
}

export function MetricStatus({
  status,
  warnings = [],
  showReason = false,
}: {
  status: EvaluationResult['status']
  warnings?: EvaluationWarning[]
  showReason?: boolean
}) {
  const copy = statusCopy(status, warnings)
  return <span className="inline-flex flex-col items-start gap-1">
    <span className={`rounded-full px-2 py-0.5 text-xs font-medium ${copy.tone}`}>{copy.label}</span>
    {showReason && warnings[0] && <span className="max-w-xs text-xs text-slate-600">{indicatorDiagnosticDetail(warnings[0].code, warnings[0].message)}</span>}
  </span>
}

export const indicatorOptionLabel = (indicator: IndicatorDefinition) => {
  const source = indicator.source === 'built_in' ? '内置' : '工作区'
  const compatibility = indicator.catalog_status === 'compatibility' ? ' · 兼容' : ''
  return `${indicator.name} · ${source} v${indicator.revision}${compatibility}`
}

type MetricSelectorPosition = {
  left: number
  top?: number
  bottom?: number
  width: number
  maxHeight: number
}

const metricSelectorPosition = (trigger: DOMRect): MetricSelectorPosition => {
  const margin = 16
  const gap = 8
  const viewportWidth = window.innerWidth
  const viewportHeight = window.innerHeight
  const width = Math.max(0, Math.min(576, viewportWidth - margin * 2))
  const left = Math.min(
    Math.max(trigger.left, margin),
    Math.max(margin, viewportWidth - width - margin),
  )
  const spaceBelow = viewportHeight - margin - trigger.bottom - gap
  const spaceAbove = trigger.top - margin - gap
  const openAbove = spaceBelow < 320 && spaceAbove > spaceBelow
  const availableHeight = openAbove ? spaceAbove : spaceBelow

  if (availableHeight < 160) {
    return {
      left,
      top: margin,
      width,
      maxHeight: Math.max(0, viewportHeight - margin * 2),
    }
  }
  return {
    left,
    ...(openAbove
      ? { bottom: viewportHeight - trigger.top + gap }
      : { top: trigger.bottom + gap }),
    width,
    maxHeight: Math.min(480, availableHeight),
  }
}

export function MetricSelector({
  indicators,
  selectedIds,
  onChange,
  maxSelected = 10,
  label = '选择指标',
  disabledReasons = {},
}: {
  indicators: IndicatorDefinition[]
  selectedIds: string[]
  onChange: (ids: string[]) => void
  maxSelected?: number
  label?: string
  disabledReasons?: Record<string, string>
}) {
  const [query, setQuery] = useState('')
  const [indicatorType, setIndicatorType] = useState('all')
  const [indicatorSource, setIndicatorSource] = useState<'all' | 'built_in' | 'custom'>('all')
  const [open, setOpen] = useState(false)
  const [position, setPosition] = useState<MetricSelectorPosition | null>(null)
  const triggerRef = useRef<HTMLButtonElement | null>(null)
  const panelRef = useRef<HTMLDivElement | null>(null)
  const panelId = useId()
  const indicatorTypes = useMemo(() => [...new Map(indicators.map((indicator) => [
    indicator.indicator_type ?? indicator.presentation?.indicator_type ?? indicator.category_id ?? 'other',
    indicator.category_label ?? indicator.presentation?.category_label ?? '其他指标',
  ])).entries()], [indicators])
  const filtered = useMemo(() => {
    const normalized = query.trim().toLowerCase()
    return indicators.filter((indicator) => {
      const type = indicator.indicator_type ?? indicator.presentation?.indicator_type ?? indicator.category_id ?? 'other'
      return (indicatorType === 'all' || type === indicatorType)
        && (indicatorSource === 'all' || indicator.source === indicatorSource)
        && (!normalized || [
        indicator.name,
        indicator.description,
        indicator.category_label,
        indicator.presentation?.category_label,
      ].some((value) => value?.toLowerCase().includes(normalized)))
    })
  }, [indicatorSource, indicatorType, indicators, query])

  const updatePosition = useCallback(() => {
    if (triggerRef.current) {
      setPosition(metricSelectorPosition(triggerRef.current.getBoundingClientRect()))
    }
  }, [])

  useLayoutEffect(() => {
    if (open) updatePosition()
  }, [open, updatePosition])

  useEffect(() => {
    if (!open) return undefined
    const handlePointerDown = (event: MouseEvent) => {
      const target = event.target as Node
      if (!triggerRef.current?.contains(target) && !panelRef.current?.contains(target)) {
        setOpen(false)
      }
    }
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.preventDefault()
        setOpen(false)
        triggerRef.current?.focus()
      }
    }
    document.addEventListener('mousedown', handlePointerDown)
    document.addEventListener('keydown', handleKeyDown)
    window.addEventListener('resize', updatePosition)
    window.addEventListener('scroll', updatePosition, true)
    return () => {
      document.removeEventListener('mousedown', handlePointerDown)
      document.removeEventListener('keydown', handleKeyDown)
      window.removeEventListener('resize', updatePosition)
      window.removeEventListener('scroll', updatePosition, true)
    }
  }, [open, updatePosition])

  const toggle = (indicatorId: string) => {
    if (selectedIds.includes(indicatorId)) {
      onChange(selectedIds.filter((id) => id !== indicatorId))
      return
    }
    if (selectedIds.length < maxSelected && !disabledReasons[indicatorId]) {
      onChange([...selectedIds, indicatorId])
    }
  }

  return <div className="relative">
    <button ref={triggerRef} type="button" aria-expanded={open} aria-controls={panelId} aria-haspopup="dialog" onClick={() => setOpen((current) => !current)} className="flex min-h-11 cursor-pointer items-center justify-between rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm font-medium text-slate-700 focus:outline-none focus-visible:ring-2 focus-visible:ring-accent-500">
      <span>{label}</span><span className="text-xs text-slate-600">已选 {selectedIds.length}/{maxSelected}</span>
    </button>
    {open && position && createPortal(<div ref={panelRef} id={panelId} role="dialog" aria-label={`${label}面板`} style={position} className="fixed z-[70] flex flex-col overflow-hidden rounded-xl border border-slate-200 bg-white shadow-xl">
      <div className="grid shrink-0 gap-2 border-b border-slate-100 p-3 sm:grid-cols-[9rem_9rem_minmax(12rem,1fr)]"><label className="block text-xs font-medium text-slate-600">指标类型<select aria-label="按指标类型筛选" value={indicatorType} onChange={(event) => setIndicatorType(event.target.value)} className="mt-1 min-h-11 w-full rounded-xl border border-slate-200 bg-white px-2 text-sm focus:border-accent-500 focus:outline-none"><option value="all">全部类型</option>{indicatorTypes.map(([id, name]) => <option key={id} value={id}>{name}</option>)}</select></label><label className="block text-xs font-medium text-slate-600">指标来源<select aria-label="按指标来源筛选" value={indicatorSource} onChange={(event) => setIndicatorSource(event.target.value as 'all' | 'built_in' | 'custom')} className="mt-1 min-h-11 w-full rounded-xl border border-slate-200 bg-white px-2 text-sm focus:border-accent-500 focus:outline-none"><option value="all">全部</option><option value="built_in">内置指标</option><option value="custom">工作区指标</option></select></label><label className="block text-xs font-medium text-slate-600">搜索指标
        <input aria-label="搜索指标" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="名称、说明或分类" className="mt-1 min-h-11 w-full rounded-lg border border-slate-200 px-3 text-sm focus:border-accent-500 focus:outline-none" />
      </label></div>
      <div className="min-h-0 flex-1 overflow-auto p-3" role="listbox" aria-multiselectable="true">
        {filtered.map((indicator) => {
          const disabledReason = disabledReasons[indicator.id]
          const checked = selectedIds.includes(indicator.id)
          return <label key={indicator.id} className={`flex min-h-11 gap-3 border-b border-slate-100 px-2 py-2 last:border-0 ${disabledReason ? 'cursor-not-allowed opacity-55' : 'cursor-pointer hover:bg-accent-50'}`}>
            <input type="checkbox" checked={checked} disabled={Boolean(disabledReason) || (!checked && selectedIds.length >= maxSelected)} onChange={() => toggle(indicator.id)} />
            <span className="min-w-0"><span className="block text-sm font-medium text-slate-800">{indicatorOptionLabel(indicator)}</span><span className="block text-xs text-slate-600">{disabledReason ?? indicator.product_kind_hint?.message ?? indicator.presentation?.category_label ?? indicator.category_label ?? '未分类'}</span></span>
          </label>
        })}
        {filtered.length === 0 && <p className="px-2 py-6 text-center text-sm text-slate-600">没有匹配的指标。</p>}
      </div>
      <div className="flex shrink-0 items-center justify-between border-t border-slate-100 px-4 py-2 text-xs text-slate-600"><span>显示 {filtered.length} 项 · 已选 {selectedIds.length}/{maxSelected}</span><button type="button" onClick={() => setOpen(false)} className="min-h-9 px-2 font-medium text-accent-700 hover:underline">完成</button></div>
    </div>, document.body)}
  </div>
}

export function MetricDefinitionDrawer({
  indicator,
  onClose,
}: {
  indicator: IndicatorDefinition | null
  onClose: () => void
}) {
  if (!indicator) return null
  const presentation = fallbackPresentation(indicator)
  const displayFormula = indicator.display_latex?.trim() || ''
  let formulaMarkup: { __html: string } | null = null
  if (displayFormula) {
    try {
      formulaMarkup = {
        __html: katex.renderToString(displayFormula, { throwOnError: false, displayMode: true }),
      }
    } catch {
      formulaMarkup = null
    }
  }
  return <div className="fixed inset-0 z-[100] flex justify-end bg-slate-950/35" role="presentation" onMouseDown={(event) => { if (event.currentTarget === event.target) onClose() }}>
    <aside role="dialog" aria-modal="true" aria-labelledby="metric-definition-title" className="h-full w-full max-w-lg overflow-auto bg-white p-6 shadow-2xl">
      <div className="flex items-start justify-between gap-4"><div><p className="text-xs font-semibold text-accent-600">{presentation.category_label}</p><h2 id="metric-definition-title" className="mt-1 text-2xl font-semibold text-slate-900">{presentation.name}</h2><p className="mt-1 text-sm text-slate-600">{indicatorOptionLabel(indicator)}</p></div><button type="button" onClick={onClose} className="min-h-11 rounded-lg px-3 text-sm text-slate-600 hover:bg-slate-100">关闭</button></div>
      <dl className="mt-6 grid gap-4 text-sm"><div><dt className="font-semibold text-slate-700">说明</dt><dd className="mt-1 text-slate-600">{presentation.description || '—'}</dd></div><div><dt className="font-semibold text-slate-700">方法</dt><dd className="mt-1 text-slate-600">{presentation.methodology || '—'}</dd></div><div><dt className="font-semibold text-slate-700">数据口径</dt><dd className="mt-1 text-slate-600">{presentation.data_basis}</dd></div><div><dt className="font-semibold text-slate-700">方向与样本</dt><dd className="mt-1 text-slate-600">{presentation.direction === 'neutral' ? '仅展示，不判断优劣' : presentation.direction === 'higher_better' ? '数值高优先' : '数值低优先'} · 至少 {presentation.minimum_observations} 个观察值</dd></div><div><dt className="font-semibold text-slate-700">公式</dt><dd className="mt-1">{formulaMarkup ? <div data-testid="metric-formula-latex" className="overflow-x-auto rounded-lg border border-accent-100 bg-accent-50/50 px-3 py-4 text-slate-900" dangerouslySetInnerHTML={formulaMarkup} /> : <p className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-3 text-sm text-slate-500">该兼容指标暂未提供数学符号排版。</p>}{<details className="mt-2"><summary className="cursor-pointer text-xs font-medium text-slate-500 hover:text-accent-700">高级信息：查看公式源码</summary><code className="mt-2 block overflow-auto rounded-lg bg-slate-950 p-3 text-xs text-emerald-200">{indicator.expression}</code></details>}</dd></div></dl>
    </aside>
  </div>
}

export function MetricResultCard({
  result,
  indicator,
  onDefinition,
  onRemove,
  period,
  periodOptions = [],
  onPeriodChange,
}: {
  result?: EvaluationResult
  indicator: IndicatorDefinition
  onDefinition?: () => void
  onRemove?: () => void
  period?: string
  periodOptions?: string[]
  onPeriodChange?: (period: string) => void
}) {
  const { s } = useI18n()
  const presentation = resolveMetricPresentation(result, indicator)
  return <article className="rounded-xl border border-slate-200 bg-white p-4">
    <div className="flex items-start justify-between gap-3"><div><p className="text-xs text-slate-600">{presentation.category_label} · {presentation.source === 'built_in' ? '内置' : '工作区'} v{presentation.revision}</p><h3 className="mt-1 font-semibold text-slate-900">{presentation.name}</h3></div>{onRemove && <button type="button" onClick={onRemove} aria-label={`移除指标 ${presentation.name}`} title="仅从当前页面移除，不会删除指标定义" className="inline-flex min-h-9 items-center rounded-lg border border-slate-200 px-2.5 text-xs font-medium text-slate-600 hover:border-rose-300 hover:bg-rose-50 hover:text-rose-700 focus:outline-none focus-visible:ring-2 focus-visible:ring-accent-500">移除</button>}</div>
    {onPeriodChange && period && <label className="mt-3 block text-xs font-medium text-slate-600">计算区间<select aria-label={`${presentation.name}计算区间`} value={period} onChange={(event) => onPeriodChange(event.target.value)} className="mt-1 min-h-10 w-full rounded-xl border border-slate-200 bg-white px-3 text-sm text-slate-700 focus:border-accent-500 focus:outline-none focus-visible:ring-2 focus-visible:ring-accent-500"><option value={period}>{indicatorPeriodOptionLabel(period)}</option>{periodOptions.filter((item) => item !== period).map((item) => <option key={item} value={item}>{indicatorPeriodOptionLabel(item)}</option>)}</select></label>}
    <div className="mt-4 text-2xl font-semibold text-accent-700"><MetricValue value={result?.value} presentation={presentation} /></div>
    {result ? <p className="mt-3 text-xs text-slate-600">{result.period} · {result.window.start_date ?? '—'} 至 {result.window.end_date ?? '—'} · {result.window.observation_count} 个观察值 · 数据截至 {result.window.data_latest_date ?? '—'}</p> : <p className="mt-3 text-xs text-slate-600">等待计算</p>}
    {result && <MetricUnavailableReason result={result} />}
    {onDefinition && <button type="button" onClick={onDefinition} className="mt-3 text-sm font-medium text-accent-700 hover:underline">查看定义与口径</button>}
  </article>
}

export function MetricMatrix({
  indicators,
  targets,
  results,
  onDefinition,
  periodsByIndicator = {},
  periodOptions = [],
  onPeriodChange,
}: {
  indicators: IndicatorDefinition[]
  targets: Array<{ kind: 'etf' | 'fund'; product_id: string; name: string }>
  results: EvaluationResult[]
  onDefinition?: (indicator: IndicatorDefinition) => void
  periodsByIndicator?: Record<string, string>
  periodOptions?: string[]
  onPeriodChange?: (indicatorId: string, period: string) => void
}) {
  const resultMap = new Map(results.map((result) => [`${result.indicator_id}:${result.target.kind}:${result.target.product_id}`, result]))
  return <div className="overflow-auto rounded-xl border border-slate-200">
    <table className="min-w-[760px] w-full text-sm"><thead className="bg-slate-50 text-left text-slate-600"><tr><th scope="col" className="sticky left-0 bg-slate-50 px-4 py-3">指标</th>{targets.map((target) => <th scope="col" key={`${target.kind}:${target.product_id}`} className="px-4 py-3 text-center">{target.name}<span className="block text-xs font-normal">{target.product_id}</span></th>)}</tr></thead><tbody>{indicators.map((indicator) => {
      const rowResults = targets.map((target) => resultMap.get(`${indicator.id}:${target.kind}:${target.product_id}`))
      const finiteValues = rowResults.flatMap((result) => typeof result?.value === 'number' && Number.isFinite(result.value) ? [result.value] : [])
      const direction = indicator.presentation?.direction ?? indicator.direction
      const bestValue = direction !== 'neutral' && finiteValues.length > 1 ? (direction === 'lower_better' ? Math.min(...finiteValues) : Math.max(...finiteValues)) : null
      const worstValue = direction !== 'neutral' && finiteValues.length > 1 ? (direction === 'lower_better' ? Math.max(...finiteValues) : Math.min(...finiteValues)) : null
      const period = periodsByIndicator[indicator.id]
      return <tr key={indicator.id} className="border-t border-slate-100"><th scope="row" className="sticky left-0 bg-white px-4 py-3 text-left"><button type="button" onClick={() => onDefinition?.(indicator)} className="font-semibold text-slate-800 hover:text-accent-700">{indicator.name}</button><span className="block text-xs font-normal text-slate-600">{indicator.source === 'built_in' ? '内置' : '工作区'} v{indicator.revision} · {direction === 'neutral' ? '仅展示' : direction === 'lower_better' ? '低值优先' : '高值优先'}</span>{onPeriodChange && period && <label className="mt-2 block text-xs font-medium text-slate-600">计算区间<select aria-label={`${indicator.name}计算区间`} value={period} onChange={(event) => onPeriodChange(indicator.id, event.target.value)} className="mt-1 min-h-9 w-full rounded-xl border border-slate-200 bg-white px-2 text-xs text-slate-700 focus:border-accent-500 focus:outline-none focus-visible:ring-2 focus-visible:ring-accent-500"><option value={period}>{indicatorPeriodOptionLabel(period)}</option>{periodOptions.filter((item) => item !== period).map((item) => <option key={item} value={item}>{indicatorPeriodOptionLabel(item)}</option>)}</select></label>}</th>{targets.map((target, index) => {
        const result = rowResults[index]
        const presentation = resolveMetricPresentation(result, indicator)
        const isBest = bestValue !== null && result?.value === bestValue
        const isWorst = worstValue !== null && result?.value === worstValue && worstValue !== bestValue
        return <td key={`${target.kind}:${target.product_id}`} className={`px-4 py-3 text-center ${isBest ? 'bg-emerald-50' : isWorst ? 'bg-rose-50' : ''}`}><MetricValue value={result?.value} presentation={presentation} />{isBest && <span className="mt-1 block text-xs font-semibold text-emerald-700">最佳</span>}{isWorst && <span className="mt-1 block text-xs font-semibold text-rose-700">最弱</span>}{result && <span className="mt-1 block"><MetricStatus status={result.status} warnings={result.warnings} /></span>}{result && result.value === null && result.input_requirements && <MetricUnavailableReason result={result} compact />}</td>
      })}</tr>
    })}</tbody></table>
  </div>
}

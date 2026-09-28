import { systemText, useI18n, i18n } from '../../i18n/runtime'
import React, { useCallback, useEffect, useId, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import katex from 'katex'
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
  name: indicator?.name ?? systemText('preInvestment.metricDisplay.metric'),
  source: indicator?.source ?? 'inline',
  indicator_type: indicator?.indicator_type ?? 'other',
  category: indicator?.category_id ?? 'custom',
  category_label: indicator?.category_label ?? systemText('preInvestment.metricDisplay.workspaceMetric'),
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
  data_basis: indicator?.data_basis ?? systemText('preInvestment.metricDisplay.realDataStrictWindowsNoMissingValue'),
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
    if (typeof value !== 'string' || !/^\d{4}-\d{2}-\d{2}$/.test(value)) return systemText('preInvestment.metricDisplay.cannotCalculate')
    const parsed = new Date(`${value}T00:00:00Z`)
    return Number.isFinite(parsed.valueOf()) && parsed.toISOString().slice(0, 10) === value ? value : systemText('preInvestment.metricDisplay.cannotCalculate')
  }
  if (typeof value !== 'number' || !Number.isFinite(value)) return systemText('preInvestment.metricDisplay.cannotCalculate')
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
  useI18n()
  const formatted = formatMetricValue(value, presentation)
  return <span className={`${formatted === systemText('preInvestment.metricDisplay.cannotCalculate') ? 'text-slate-600' : 'tabular-nums'} ${className}`}>{formatted}</span>
}

const statusCopy = (
  status: EvaluationResult['status'],
  warnings: EvaluationWarning[],
) => {
  if (status === 'error') return { label: systemText('preInvestment.metricDisplay.calculationFailed'), tone: 'bg-rose-50 text-rose-700' }
  if (status === 'unavailable') return { label: systemText('preInvestment.metricDisplay.cannotCalculate'), tone: 'bg-amber-50 text-amber-700' }
  if (status === 'warning') return { label: warnings.some((item) => item.code.includes('SAMPLE')) ? systemText('preInvestment.metricDisplay.insufficientSamples') : systemText('preInvestment.metricDisplay.warnings'), tone: 'bg-amber-50 text-amber-700' }
  return { label: systemText('preInvestment.metricDisplay.valid'), tone: 'bg-emerald-50 text-emerald-700' }
}

type AvailabilityResult = Pick<
  EvaluationResult,
  'value' | 'status' | 'warnings' | 'input_requirements' | 'target_data' | 'data_context'
>

const knownVariableLabels: Record<string, string> = {
  get adjusted_nav() { return systemText('preInvestment.metricDisplay.adjustedNav') },
  get returns() { return systemText('preInvestment.metricDisplay.simpleReturns') },
  get log_returns() { return systemText('preInvestment.metricDisplay.logReturns') },
  get adjusted_open() { return systemText('preInvestment.metricDisplay.adjustedOpen') },
  get adjusted_high() { return systemText('preInvestment.metricDisplay.adjustedHigh') },
  get adjusted_low() { return systemText('preInvestment.metricDisplay.adjustedLow') },
  get adjusted_close() { return systemText('preInvestment.metricDisplay.adjustedClose') },
  get market_open() { return systemText('preInvestment.metricDisplay.open') },
  get market_high() { return systemText('preInvestment.metricDisplay.high') },
  get market_low() { return systemText('preInvestment.metricDisplay.low') },
  get market_close() { return systemText('preInvestment.metricDisplay.close') },
  get previous_close() { return systemText('preInvestment.metricDisplay.previousClose') },
  get price_change() { return systemText('preInvestment.metricDisplay.priceChange') },
  get price_return() { return systemText('preInvestment.metricDisplay.marketReturn') },
  get volume() { return systemText('preInvestment.metricDisplay.volume') },
  get turnover_amount() { return systemText('preInvestment.metricDisplay.turnoverValue') },
  get unit_nav() { return systemText('preInvestment.metricDisplay.unitNav') },
  get accumulated_nav() { return systemText('preInvestment.metricDisplay.cumulativeNav') },
}

export function IndicatorInputDates({ context }: { context?: IndicatorDateContext | null }) {
  useI18n()
  if (!context) return null
  return <section aria-label={systemText('preInvestment.metricDisplay.dataAndDatesUsedInThisCalculation')} className="mt-3 rounded-lg border border-amber-200 bg-amber-50 p-3 text-left text-sm text-amber-950">
    <h4 className="font-semibold">{systemText('preInvestment.metricDisplay.dataAndDatesUsedInThisCalculation')}</h4>
    <dl className="mt-2 grid gap-2 sm:grid-cols-2">
      <div><dt className="text-xs text-amber-800">{systemText('preInvestment.metricDisplay.productInceptionDate')}</dt><dd>{context.found_date || systemText('preInvestment.metricDisplay.notProvidedByTheDataSource')}</dd></div>
      {context.list_date && <div><dt className="text-xs text-amber-800">{systemText('preInvestment.metricDisplay.listingDate')}</dt><dd>{context.list_date}</dd></div>}
      <div><dt className="text-xs text-amber-800">{systemText('preInvestment.metricDisplay.calculationCutoffPit')}</dt><dd>{context.as_of || systemText('preInvestment.metricDisplay.notSetAllLocalDatesUsed')}</dd></div>
    </dl>
    {context.sources.map((source, index) => <div key={`${source.label}-${index}`} className="mt-3 border-t border-amber-200 pt-2">
      <p>{source.label}{systemText('preInvestment.metricDisplay.localCoverage')}{source.first_date || systemText('preInvestment.metricDisplay.startUnconfirmed')} {" " + systemText('preInvestment.metricDisplay.to') + " "}{source.latest_date || systemText('preInvestment.metricDisplay.endUnconfirmed')}</p>
      {context.as_of && (source.disclosure_status === 'required_unavailable'
        ? <p className="mt-1 text-xs">{systemText('preInvestment.metricDisplay.thisSourceLacksAnnouncementDatesSoDisclosure')}</p>
        : <>
          <p className="mt-1 text-xs">{systemText('preInvestment.metricDisplay.originally') + " "}{source.rows_before_as_of ?? systemText('preInvestment.metricDisplay.unconfirmed')} {" " + systemText('preInvestment.metricDisplay.recordsAfterDateFiltering') + " "}{source.rows_after_date_filter ?? systemText('preInvestment.metricDisplay.unconfirmed')} {" " + systemText('preInvestment.metricDisplay.records') + " "}{source.uses_disclosure_date ? systemText('preInvestment.metricDisplay.afterDisclosureFilteringAndDeduplication') : systemText('preInvestment.metricDisplay.afterDeduplication')} {source.rows_after_as_of ?? systemText('preInvestment.metricDisplay.unconfirmed')} {" " + systemText('preInvestment.metricDisplay.items')}</p>
          <p className="mt-1 text-xs">{source.uses_disclosure_date
            ? systemText('preInvestment.metricDisplay.thisSourceAlsoFiltersAnnouncementDatesRecords')
            : systemText('preInvestment.metricDisplay.thisSourceFiltersDataDatesOnlyAnnouncement')}</p>
        </>)}
    </div>)}
    <p className="mt-2 text-xs">{systemText('preInvestment.metricDisplay.inceptionListingAndLocalDataStartDates')}</p>
    {context.as_of && <>
      <p className="mt-2">{systemText('preInvestment.metricDisplay.onlyRecordsOnOrBeforeTheCutoff')}</p>
      <p className="mt-2">{systemText('preInvestment.metricDisplay.forHistoricalResearchSelectProductsWithData')}</p>
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
  useI18n()
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
    ? systemText('preInvestment.metricDisplay.thisMetricRequiresInputFieldsTheCurrent', { p0: requirements?.required_count ?? blocked.length, p1: labels.join('、') })
    : systemText('preInvestment.metricDisplay.someValuesAreMissingCalculationsUseCommon', { p0: labels.join('、') })
  return <>{dateContext}<div className={`${compact ? 'mt-1' : 'mt-3'} rounded-lg border border-amber-200 bg-amber-50 p-3 text-left text-xs text-amber-950`}>
    <p className="font-medium">{summary}</p>
    <details className="mt-2" open={!compact}>
      <summary className="cursor-pointer font-semibold text-amber-800">{blocked.length ? systemText('preInvestment.metricDisplay.whyCalculationIsUnavailable') : systemText('preInvestment.metricDisplay.viewDataCoverage')}</summary>
      <ul className="mt-2 space-y-1">
        {affected.map((item) => <li key={item.variable_id}><span className="font-medium">{item.label}</span>：{item.reason || (item.status === 'partial' ? systemText('preInvestment.metricDisplay.someDatesLackValidValues') : systemText('preInvestment.metricDisplay.noUsableDataCurrentlyAvailable'))}</li>)}
      </ul>
      {result.target_data && <div className="mt-3 border-t border-amber-200 pt-2">
        <p className="font-semibold">{systemText('preInvestment.metricDisplay.currentlyAvailableData')}</p>
        {result.target_data.available_datasets.length > 0 && <p className="mt-1">{systemText('preInvestment.metricDisplay.dataSource')}{result.target_data.available_datasets.join('、')}</p>}
        {availableFields.length > 0 && <p className="mt-1">{systemText('preInvestment.metricDisplay.availableFields')}{availableFields.join('、')}</p>}
        <p className="mt-1">{systemText('preInvestment.metricDisplay.dataThrough')}{result.target_data.data_latest_date || systemText('preInvestment.metricDisplay.none')}</p>
      </div>}
      {alternatives.length > 0 && <p className="mt-3 border-t border-amber-200 pt-2"><span className="font-semibold">{systemText('preInvestment.metricDisplay.suggestion')}</span>{systemText('preInvestment.metricDisplay.youCanInsteadUse')}{alternatives.join('、')}{systemText('preInvestment.metricDisplay.toBuildAMetricSuitableForThis')}</p>}
      <details className="mt-2">
        <summary className="cursor-pointer text-amber-700">{systemText('preInvestment.metricDisplay.viewTechnicalDetails')}</summary>
        <ul className="mt-1 space-y-1 font-mono text-xs text-amber-800">
          {affected.map((item) => <li key={item.variable_id}>{item.variable_id} · {item.source_dataset || systemText('preInvestment.metricDisplay.runtimeDerived')}{item.source_field ? `.${item.source_field}` : ''} · {item.reason_code || item.status}</li>)}
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
  useI18n()
  const copy = statusCopy(status, warnings)
  return <span className="inline-flex flex-col items-start gap-1">
    <span className={`rounded-full px-2 py-0.5 text-xs font-medium ${copy.tone}`}>{copy.label}</span>
    {showReason && warnings[0] && <span className="max-w-xs text-xs text-slate-600">{indicatorDiagnosticDetail(warnings[0].code, warnings[0].message)}</span>}
  </span>
}

export const indicatorOptionLabel = (indicator: IndicatorDefinition) => {
  const source = indicator.source === 'built_in' ? systemText('preInvestment.metricDisplay.builtIn') : systemText('preInvestment.metricDisplay.workspace')
  const compatibility = indicator.catalog_status === 'compatibility' ? " " + systemText('preInvestment.metricDisplay.compatible') : ''
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
  maxSelected,
  label = systemText('preInvestment.metricDisplay.selectMetrics'),
  disabledReasons = {},
}: {
  indicators: IndicatorDefinition[]
  selectedIds: string[]
  onChange: (ids: string[]) => void
  /** Left out where there is no cap: the count then reads as a count, not a quota. */
  maxSelected?: number
  label?: string
  disabledReasons?: Record<string, string>
}) {
  useI18n()
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
    indicator.category_label ?? indicator.presentation?.category_label ?? systemText('preInvestment.metricDisplay.otherMetrics'),
  ])).entries()], [indicators, i18n.language])
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

  const quota = maxSelected === undefined ? '' : `/${maxSelected}`
  const atLimit = maxSelected !== undefined && selectedIds.length >= maxSelected
  const toggle = (indicatorId: string) => {
    if (selectedIds.includes(indicatorId)) {
      onChange(selectedIds.filter((id) => id !== indicatorId))
      return
    }
    if (!atLimit && !disabledReasons[indicatorId]) {
      onChange([...selectedIds, indicatorId])
    }
  }

  return <div className="relative">
    <button ref={triggerRef} type="button" aria-expanded={open} aria-controls={panelId} aria-haspopup="dialog" onClick={() => setOpen((current) => !current)} className="flex min-h-11 cursor-pointer items-center justify-between gap-2 rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm font-medium text-slate-700 focus:outline-none focus-visible:ring-2 focus-visible:ring-accent-500">
      <span>{label}</span><span className="text-xs text-slate-600">{systemText('preInvestment.metricDisplay.selected') + " "}{selectedIds.length}{quota}</span>
    </button>
    {open && position && createPortal(<div ref={panelRef} id={panelId} role="dialog" aria-label={systemText('preInvestment.metricDisplay.panel', { p0: label })} style={position} className="fixed z-[70] flex flex-col overflow-hidden rounded-xl border border-slate-200 bg-white shadow-xl">
      <div className="grid shrink-0 gap-2 border-b border-slate-100 p-3 sm:grid-cols-[9rem_9rem_minmax(12rem,1fr)]"><label className="block text-xs font-medium text-slate-600">{systemText('preInvestment.metricDisplay.metricType')}<select aria-label={systemText('preInvestment.metricDisplay.filterByMetricType')} value={indicatorType} onChange={(event) => setIndicatorType(event.target.value)} className="mt-1 min-h-11 w-full rounded-xl border border-slate-200 bg-white px-2 text-sm focus:border-accent-500 focus:outline-none"><option value="all">{systemText('preInvestment.metricDisplay.allTypes')}</option>{indicatorTypes.map(([id, name]) => <option key={id} value={id}>{name}</option>)}</select></label><label className="block text-xs font-medium text-slate-600">{systemText('preInvestment.metricDisplay.metricSource')}<select aria-label={systemText('preInvestment.metricDisplay.filterByMetricSource')} value={indicatorSource} onChange={(event) => setIndicatorSource(event.target.value as 'all' | 'built_in' | 'custom')} className="mt-1 min-h-11 w-full rounded-xl border border-slate-200 bg-white px-2 text-sm focus:border-accent-500 focus:outline-none"><option value="all">{systemText('preInvestment.metricDisplay.all')}</option><option value="built_in">{systemText('preInvestment.metricDisplay.builtInMetrics')}</option><option value="custom">{systemText('preInvestment.metricDisplay.workspaceMetric')}</option></select></label><label className="block text-xs font-medium text-slate-600">{systemText('preInvestment.metricDisplay.searchMetrics')}<input aria-label={systemText('preInvestment.metricDisplay.searchMetrics')} value={query} onChange={(event) => setQuery(event.target.value)} placeholder={systemText('preInvestment.metricDisplay.nameDescriptionOrCategory')} className="mt-1 min-h-11 w-full rounded-lg border border-slate-200 px-3 text-sm focus:border-accent-500 focus:outline-none" />
      </label></div>
      <div className="min-h-0 flex-1 overflow-auto p-3" role="listbox" aria-multiselectable="true">
        {filtered.map((indicator) => {
          const disabledReason = disabledReasons[indicator.id]
          const checked = selectedIds.includes(indicator.id)
          return <label key={indicator.id} className={`flex min-h-11 gap-3 border-b border-slate-100 px-2 py-2 last:border-0 ${disabledReason ? 'cursor-not-allowed opacity-55' : 'cursor-pointer hover:bg-accent-50'}`}>
            <input type="checkbox" checked={checked} disabled={Boolean(disabledReason) || (!checked && atLimit)} onChange={() => toggle(indicator.id)} />
            <span className="min-w-0"><span className="block text-sm font-medium text-slate-800">{indicatorOptionLabel(indicator)}</span><span className="block text-xs text-slate-600">{disabledReason ?? indicator.product_kind_hint?.message ?? indicator.presentation?.category_label ?? indicator.category_label ?? systemText('preInvestment.metricDisplay.uncategorized')}</span></span>
          </label>
        })}
        {filtered.length === 0 && <p className="px-2 py-6 text-center text-sm text-slate-600">{systemText('preInvestment.metricDisplay.noMatchingMetrics')}</p>}
      </div>
      <div className="flex shrink-0 items-center justify-between border-t border-slate-100 px-4 py-2 text-xs text-slate-600"><span>{systemText('preInvestment.metricDisplay.showing') + " "}{filtered.length} {" " + systemText('preInvestment.metricDisplay.itemsSelected') + " "}{selectedIds.length}{quota}</span><button type="button" onClick={() => setOpen(false)} className="min-h-9 px-2 font-medium text-accent-700 hover:underline">{systemText('preInvestment.metricDisplay.completed')}</button></div>
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
  useI18n()
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
      <div className="flex items-start justify-between gap-4"><div><p className="text-xs font-semibold text-accent-600">{presentation.category_label}</p><h2 id="metric-definition-title" className="mt-1 text-2xl font-semibold text-slate-900">{presentation.name}</h2><p className="mt-1 text-sm text-slate-600">{indicatorOptionLabel(indicator)}</p></div><button type="button" onClick={onClose} className="min-h-11 rounded-lg px-3 text-sm text-slate-600 hover:bg-slate-100">{systemText('preInvestment.metricDisplay.close2')}</button></div>
      <dl className="mt-6 grid gap-4 text-sm"><div><dt className="font-semibold text-slate-700">{systemText('preInvestment.metricDisplay.description')}</dt><dd className="mt-1 text-slate-600">{presentation.description || '—'}</dd></div><div><dt className="font-semibold text-slate-700">{systemText('preInvestment.metricDisplay.method')}</dt><dd className="mt-1 text-slate-600">{presentation.methodology || '—'}</dd></div><div><dt className="font-semibold text-slate-700">{systemText('preInvestment.metricDisplay.dataConvention')}</dt><dd className="mt-1 text-slate-600">{presentation.data_basis}</dd></div><div><dt className="font-semibold text-slate-700">{systemText('preInvestment.metricDisplay.directionAndSample')}</dt><dd className="mt-1 text-slate-600">{presentation.direction === 'neutral' ? systemText('preInvestment.metricDisplay.displayOnlyNoRankingPreference') : presentation.direction === 'higher_better' ? systemText('preInvestment.metricDisplay.higherIsBetter') : systemText('preInvestment.metricDisplay.lowerIsBetter')} {" " + systemText('preInvestment.metricDisplay.atLeast') + " "}{presentation.minimum_observations} {" " + systemText('preInvestment.metricDisplay.observations')}</dd></div><div><dt className="font-semibold text-slate-700">{systemText('preInvestment.metricDisplay.formula')}</dt><dd className="mt-1">{formulaMarkup ? <div data-testid="metric-formula-latex" className="overflow-x-auto rounded-lg border border-accent-100 bg-accent-50/50 px-3 py-4 text-slate-900" dangerouslySetInnerHTML={formulaMarkup} /> : <p className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-3 text-sm text-slate-500">{systemText('preInvestment.metricDisplay.thisCompatibilityMetricHasNoTypesetMathematical')}</p>}{<details className="mt-2"><summary className="cursor-pointer text-xs font-medium text-slate-500 hover:text-accent-700">{systemText('preInvestment.metricDisplay.advancedViewFormulaSource')}</summary><code className="mt-2 block overflow-auto rounded-lg bg-slate-950 p-3 text-xs text-emerald-200">{indicator.expression}</code></details>}</dd></div></dl>
    </aside>
  </div>
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
  useI18n()
  const resultMap = new Map(results.map((result) => [`${result.indicator_id}:${result.target.kind}:${result.target.product_id}`, result]))
  return <div className="overflow-auto rounded-xl border border-slate-200">
    <table className="min-w-[760px] w-full text-sm"><thead className="bg-slate-50 text-left text-slate-600"><tr><th scope="col" className="sticky left-0 bg-slate-50 px-4 py-3">{systemText('preInvestment.metricDisplay.metric')}</th>{targets.map((target) => <th scope="col" key={`${target.kind}:${target.product_id}`} className="px-4 py-3 text-center">{target.name}<span className="block text-xs font-normal">{target.product_id}</span></th>)}</tr></thead><tbody>{indicators.map((indicator) => {
      const rowResults = targets.map((target) => resultMap.get(`${indicator.id}:${target.kind}:${target.product_id}`))
      const finiteValues = rowResults.flatMap((result) => typeof result?.value === 'number' && Number.isFinite(result.value) ? [result.value] : [])
      const direction = indicator.presentation?.direction ?? indicator.direction
      const bestValue = direction !== 'neutral' && finiteValues.length > 1 ? (direction === 'lower_better' ? Math.min(...finiteValues) : Math.max(...finiteValues)) : null
      const worstValue = direction !== 'neutral' && finiteValues.length > 1 ? (direction === 'lower_better' ? Math.max(...finiteValues) : Math.min(...finiteValues)) : null
      const period = periodsByIndicator[indicator.id]
      return <tr key={indicator.id} className="border-t border-slate-100"><th scope="row" className="sticky left-0 bg-white px-4 py-3 text-left"><button type="button" onClick={() => onDefinition?.(indicator)} className="font-semibold text-slate-800 hover:text-accent-700">{indicator.name}</button><span className="block text-xs font-normal text-slate-600">{indicator.source === 'built_in' ? systemText('preInvestment.metricDisplay.builtIn') : systemText('preInvestment.metricDisplay.workspace')} v{indicator.revision} · {direction === 'neutral' ? systemText('preInvestment.metricDisplay.displayOnly') : direction === 'lower_better' ? systemText('preInvestment.metricDisplay.lowerValuesFirst') : systemText('preInvestment.metricDisplay.higherValuesFirst')}</span>{onPeriodChange && period && <label className="mt-2 block text-xs font-medium text-slate-600">{systemText('preInvestment.metricDisplay.calculationInterval')}<select aria-label={systemText('preInvestment.metricDisplay.calculationInterval2', { p0: indicator.name })} value={period} onChange={(event) => onPeriodChange(indicator.id, event.target.value)} className="mt-1 min-h-9 w-full rounded-xl border border-slate-200 bg-white px-2 text-xs text-slate-700 focus:border-accent-500 focus:outline-none focus-visible:ring-2 focus-visible:ring-accent-500"><option value={period}>{indicatorPeriodOptionLabel(period)}</option>{periodOptions.filter((item) => item !== period).map((item) => <option key={item} value={item}>{indicatorPeriodOptionLabel(item)}</option>)}</select></label>}</th>{targets.map((target, index) => {
        const result = rowResults[index]
        const presentation = resolveMetricPresentation(result, indicator)
        const isBest = bestValue !== null && result?.value === bestValue
        const isWorst = worstValue !== null && result?.value === worstValue && worstValue !== bestValue
        return <td key={`${target.kind}:${target.product_id}`} className={`px-4 py-3 text-center ${isBest ? 'bg-emerald-50' : isWorst ? 'bg-rose-50' : ''}`}><MetricValue value={result?.value} presentation={presentation} />{isBest && <span className="mt-1 block text-xs font-semibold text-emerald-700">{systemText('preInvestment.metricDisplay.best')}</span>}{isWorst && <span className="mt-1 block text-xs font-semibold text-rose-700">{systemText('preInvestment.metricDisplay.weakest')}</span>}{result && <span className="mt-1 block"><MetricStatus status={result.status} warnings={result.warnings} /></span>}{result && result.value === null && result.input_requirements && <MetricUnavailableReason result={result} compact />}</td>
      })}</tr>
    })}</tbody></table>
  </div>
}

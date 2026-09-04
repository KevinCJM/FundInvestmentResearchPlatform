import React, { useEffect, useMemo, useState } from 'react'
import FilterDropdown from '../FilterDropdown'
import {
  listInstrumentProducts,
  selectAllInstrumentProducts,
  type InstrumentProductItem,
  type InstrumentProductFilterKey,
  type InstrumentProductFilterState,
  type InstrumentProductsResponse,
  type EvaluationProductSelection,
  type ProductCondition,
  type ProductConditionField,
  type ProductConditionOperator,
  type ProductConditionOperatorOption,
  type ProductKind,
} from '../../services/customIndicators'

interface EvaluationProductSelectorProps {
  productKind: ProductKind
  selectedItems: Record<string, InstrumentProductItem>
  onSelectedItemsChange: (items: Record<string, InstrumentProductItem>) => void
  selectionState: EvaluationProductSelection
  onSelectionStateChange: (state: EvaluationProductSelection) => void
  onError: (message: string | null) => void
}

const INITIAL_FILTERS: InstrumentProductFilterState = {
  fund_type: [],
  invest_type: [],
  qdii_type: [],
  market: [],
  status: [],
  management: [],
  custodian: [],
}

const FILTER_LABELS: Record<InstrumentProductFilterKey, string> = {
  fund_type: '投资类型',
  invest_type: '投资风格',
  qdii_type: 'QDII 属性',
  market: '交易市场',
  status: '产品状态',
  management: '管理人',
  custodian: '托管人',
}

const DEFAULT_OPERATORS: ProductConditionOperatorOption[] = [
  { value: 'gte', label: '大于等于', symbol: '≥' },
  { value: 'lte', label: '小于等于', symbol: '≤' },
  { value: 'gt', label: '大于', symbol: '>' },
  { value: 'lt', label: '小于', symbol: '<' },
  { value: 'eq', label: '等于', symbol: '=' },
]

const PAGE_SIZE_OPTIONS = [10, 20, 50, 100]
const instrumentCode = (item: InstrumentProductItem) => item.ts_code ?? item.code ?? ''

const fallbackConditionFields = (kind: ProductKind): ProductConditionField[] => [{
  field: kind === 'etf' ? 'list_date' : 'found_date',
  label: kind === 'etf' ? '上市日期' : '成立日期',
  data_type: 'date',
  unit_label: null,
  source: 'fund_basic',
  available: true,
}]

export default function EvaluationProductSelector({
  productKind,
  selectedItems,
  onSelectedItemsChange,
  selectionState,
  onSelectionStateChange,
  onError,
}: EvaluationProductSelectorProps) {
  const [response, setResponse] = useState<InstrumentProductsResponse | null>(null)
  const [searchInput, setSearchInput] = useState(selectionState.query)
  const [page, setPage] = useState(1)
  const [pageSize, setPageSize] = useState(PAGE_SIZE_OPTIONS[0])
  const [loading, setLoading] = useState(false)
  const [selectingAll, setSelectingAll] = useState(false)
  const filters = selectionState.filters
  const conditions = selectionState.conditions
  const query = selectionState.query
  const allMatchingSelected = selectionState.selection_mode === 'all_matching'

  useEffect(() => {
    setSearchInput(selectionState.query)
  }, [selectionState.query])

  useEffect(() => {
    const timer = window.setTimeout(() => {
      const nextQuery = searchInput.trim()
      if (nextQuery !== query) {
        onSelectedItemsChange({})
        onSelectionStateChange({
          ...selectionState,
          query: nextQuery,
          selection_mode: 'manual',
        })
      }
      setPage(1)
    }, 300)
    return () => window.clearTimeout(timer)
  }, [onSelectedItemsChange, onSelectionStateChange, query, searchInput, selectionState])

  useEffect(() => {
    const controller = new AbortController()
    setLoading(true)
    listInstrumentProducts({
      kind: productKind,
      query,
      page,
      pageSize,
      filters,
      conditions,
      signal: controller.signal,
    }).then((payload) => {
      setResponse(payload)
    }).catch((caught) => {
      if ((caught as DOMException).name !== 'AbortError') {
        onError('产品筛选失败，请稍后重试。')
      }
    }).finally(() => {
      if (!controller.signal.aborted) setLoading(false)
    })
    return () => controller.abort()
  }, [conditions, filters, onError, page, pageSize, productKind, query])

  const selectedList = useMemo(() => Object.values(selectedItems), [selectedItems])
  useEffect(() => {
    if (selectedList.length === 0 && allMatchingSelected) {
      onSelectionStateChange({ ...selectionState, selection_mode: 'manual' })
    }
  }, [allMatchingSelected, onSelectionStateChange, selectedList.length, selectionState])
  const pageItems = response?.items ?? []
  const currentPageAllSelected = pageItems.length > 0
    && pageItems.every((item) => Boolean(selectedItems[instrumentCode(item)]))
  const totalPages = Math.max(1, Math.ceil((response?.total ?? 0) / pageSize))
  const filterKeys: InstrumentProductFilterKey[] = productKind === 'etf'
    ? ['fund_type', 'invest_type', 'qdii_type', 'market', 'status', 'management', 'custodian']
    : ['fund_type', 'invest_type', 'qdii_type', 'status', 'management', 'custodian']
  const activeFilterCount = filterKeys.reduce((total, key) => total + filters[key].length, 0) + conditions.length
  const conditionFields = response?.condition_fields ?? fallbackConditionFields(productKind)
  const conditionOperators = response?.condition_operators ?? DEFAULT_OPERATORS

  const updateFilter = (key: InstrumentProductFilterKey) => (values: string[]) => {
    onSelectionStateChange({
      ...selectionState,
      filters: { ...filters, [key]: values },
      selection_mode: 'manual',
    })
    onSelectedItemsChange({})
    setPage(1)
  }
  const clearFilters = () => {
    setSearchInput('')
    onSelectionStateChange({
      query: '',
      filters: { ...INITIAL_FILTERS },
      conditions: [],
      selection_mode: 'manual',
    })
    onSelectedItemsChange({})
    setPage(1)
  }
  const toggleItem = (item: InstrumentProductItem) => {
    const code = instrumentCode(item)
    if (!code) return
    const next = { ...selectedItems }
    if (next[code]) {
      delete next[code]
    } else {
      next[code] = { ...item, instrument_type: productKind }
    }
    onSelectedItemsChange(next)
    onSelectionStateChange({ ...selectionState, selection_mode: 'manual' })
  }
  const toggleCurrentPage = () => {
    const next = { ...selectedItems }
    if (currentPageAllSelected) {
      pageItems.forEach((item) => delete next[instrumentCode(item)])
    } else {
      pageItems.forEach((item) => {
        const code = instrumentCode(item)
        if (code) next[code] = { ...item, instrument_type: productKind }
      })
    }
    onSelectedItemsChange(next)
    onSelectionStateChange({ ...selectionState, selection_mode: 'manual' })
  }
  const toggleAllMatching = async () => {
    if (allMatchingSelected) {
      onSelectedItemsChange({})
      onSelectionStateChange({ ...selectionState, selection_mode: 'manual' })
      return
    }
    setSelectingAll(true)
    onError(null)
    try {
      const payload = await selectAllInstrumentProducts({
        kind: productKind,
        query,
        filters,
        conditions,
      })
      onSelectedItemsChange(Object.fromEntries(payload.items.flatMap((item) => {
        const code = instrumentCode(item)
        return code ? [[code, { ...item, instrument_type: productKind }]] : []
      })))
      onSelectionStateChange({ ...selectionState, selection_mode: 'all_matching' })
    } catch {
      onError('全选筛选结果失败，请缩小范围或稍后重试。')
    } finally {
      setSelectingAll(false)
    }
  }

  return (
    <section className="rounded-2xl bg-white p-6 shadow-sm ring-1 ring-slate-100">
      <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
        <div>
          <h2 className="text-xl font-semibold text-slate-900">1. 筛选并选择真实产品</h2>
          <p className="mt-1 text-sm text-slate-500">
            当前仅选择{productKind === 'etf' ? ' ETF' : '场外公募基金'}；筛选条件和选择模式随方案保存，入选产品代码同时锁定。
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2 text-sm">
          <span className="rounded-full bg-emerald-50 px-3 py-1 font-semibold text-emerald-700">已选 {selectedList.length}</span>
          <span className="rounded-full bg-slate-100 px-3 py-1 text-slate-600">筛选结果 {response?.total ?? 0}</span>
        </div>
      </div>

      <div className="mt-5 space-y-4 rounded-xl border border-slate-100 bg-slate-50/60 p-4">
        <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
          <label className="flex w-full max-w-xl items-center rounded-lg border border-slate-200 bg-white px-3 text-sm text-slate-700">
            <span className="sr-only">搜索产品</span>
            <input
              value={searchInput}
              onChange={(event) => setSearchInput(event.target.value)}
              placeholder="按代码、名称或管理人搜索"
              className="min-h-11 w-full bg-transparent outline-none"
            />
          </label>
          <div className="flex items-center gap-3">
            <span className="text-xs font-semibold text-emerald-700">已选条件 {activeFilterCount}</span>
            {(activeFilterCount > 0 || searchInput) && <button type="button" onClick={clearFilters} className="text-sm font-medium text-emerald-700 hover:underline">重置全部筛选</button>}
          </div>
        </div>
        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
          {filterKeys.map((key) => (
            <FilterDropdown
              key={key}
              label={FILTER_LABELS[key]}
              options={response?.available_filters?.[key] ?? []}
              selected={filters[key]}
              onChange={updateFilter(key)}
            />
          ))}
        </div>
        <ConditionFilters
          fields={conditionFields}
          operators={conditionOperators}
          conditions={conditions}
          snapshotStatus={response?.snapshot?.status}
          onChange={(next) => {
            onSelectionStateChange({
              ...selectionState,
              conditions: next,
              selection_mode: 'manual',
            })
            onSelectedItemsChange({})
            setPage(1)
          }}
        />
        <div className="flex flex-wrap gap-2">
          {filterKeys.flatMap((key) => filters[key].map((value) => (
            <button
              key={`${key}:${value}`}
              type="button"
              onClick={() => updateFilter(key)(filters[key].filter((item) => item !== value))}
              className="rounded-full bg-emerald-50 px-3 py-1 text-xs font-semibold text-emerald-700"
            >
              {FILTER_LABELS[key]}：{value} ×
            </button>
          )))}
        </div>
      </div>

      <div className="mt-4 flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap items-center gap-2 text-sm text-slate-600">
          <label className="flex items-center gap-2">
            <span>每页</span>
            <select
              aria-label="评价方案每页产品数量"
              value={pageSize}
              onChange={(event) => { setPageSize(Number(event.target.value)); setPage(1) }}
              className="min-h-10 rounded-lg border border-slate-200 bg-white px-3 text-sm"
            >
              {PAGE_SIZE_OPTIONS.map((size) => <option key={size} value={size}>{size}</option>)}
            </select>
          </label>
          <span className="text-slate-400">共 {response?.total ?? 0} 条</span>
          <button type="button" onClick={toggleCurrentPage} disabled={pageItems.length === 0} className="min-h-10 rounded-lg border border-slate-200 px-3 text-sm font-semibold text-slate-700 disabled:opacity-50">
            {currentPageAllSelected ? '取消本页全选' : '本页全选'}
          </button>
          <button type="button" onClick={() => void toggleAllMatching()} disabled={!response?.total || selectingAll} className={`min-h-10 rounded-lg border px-3 text-sm font-semibold disabled:opacity-50 ${allMatchingSelected ? 'border-emerald-500 bg-emerald-50 text-emerald-700' : 'border-slate-200 text-slate-700'}`}>
            {selectingAll ? '正在全选…' : allMatchingSelected ? '取消全选' : `全选 ${response?.total ?? 0} 条`}
          </button>
          {selectedList.length > 0 && !allMatchingSelected && <button type="button" onClick={() => onSelectedItemsChange({})} className="min-h-10 px-2 text-sm font-medium text-slate-500 hover:text-rose-600">清空选择</button>}
        </div>
        <div className="flex items-center gap-2 text-sm text-slate-600">
          <button type="button" onClick={() => setPage((current) => Math.max(1, current - 1))} disabled={page <= 1} className="rounded border border-slate-200 px-3 py-1 disabled:opacity-40">上一页</button>
          <span>第 {page} / {totalPages} 页</span>
          <button type="button" onClick={() => setPage((current) => Math.min(totalPages, current + 1))} disabled={page >= totalPages} className="rounded border border-slate-200 px-3 py-1 disabled:opacity-40">下一页</button>
        </div>
      </div>
      {allMatchingSelected && <p className="mt-2 text-xs font-medium text-emerald-700" role="status">
        已选择全部符合筛选条件的产品{selectedList.length < (response?.total ?? 0) ? `，排除 ${(response?.total ?? 0) - selectedList.length} 个` : ''}
      </p>}

      <div className="mt-3 max-h-80 overflow-auto rounded-xl border border-slate-100">
        <table className="min-w-[820px] w-full text-sm">
          <caption className="sr-only">{productKind === 'etf' ? 'ETF' : '场外公募基金'}评价候选产品</caption>
          <thead className="sticky top-0 bg-slate-50 text-left text-slate-500">
            <tr><th scope="col" className="px-4 py-3">选择</th><th scope="col" className="px-4 py-3">产品</th><th scope="col" className="px-4 py-3">投资类型 / 风格</th><th scope="col" className="px-4 py-3">管理人</th><th scope="col" className="px-4 py-3">状态</th></tr>
          </thead>
          <tbody>
            {pageItems.map((item) => {
              const code = instrumentCode(item)
              const checked = Boolean(selectedItems[code])
              return <tr key={code} className="border-t border-slate-100">
                <td className="px-4 py-3"><input aria-label={`选择 ${item.name ?? code}`} type="checkbox" checked={checked} onChange={() => toggleItem(item)} /></td>
                <td className="px-4 py-3 font-medium text-slate-800">{item.name ?? '未命名'}<span className="ml-2 text-xs font-normal text-slate-400">{code}</span></td>
                <td className="px-4 py-3 text-slate-600">
                  {item.fund_type ?? '—'}
                  <span className="ml-2 text-xs text-slate-400">{item.invest_type ?? ''}</span>
                  {item.qdii_type && <span className={`ml-2 rounded-full px-2 py-0.5 text-xs font-semibold ${item.qdii_type === 'QDII' ? 'bg-violet-100 text-violet-700' : 'bg-slate-100 text-slate-500'}`}>{item.qdii_type}</span>}
                </td>
                <td className="px-4 py-3 text-slate-600">{item.management ?? '—'}</td>
                <td className="px-4 py-3 text-slate-600">{item.status ?? '—'}</td>
              </tr>
            })}
            {!loading && pageItems.length === 0 && <tr><td colSpan={5} className="px-4 py-10 text-center text-slate-500">没有符合筛选条件的产品。</td></tr>}
          </tbody>
        </table>
        {loading && <div className="p-6 text-center text-sm text-slate-500" aria-live="polite">正在加载筛选结果…</div>}
      </div>
    </section>
  )
}

function ConditionFilters({
  fields,
  operators,
  conditions,
  snapshotStatus,
  onChange,
}: {
  fields: ProductConditionField[]
  operators: ProductConditionOperatorOption[]
  conditions: ProductCondition[]
  snapshotStatus?: string | null
  onChange: (conditions: ProductCondition[]) => void
}) {
  const availableFields = fields.filter((field) => field.available)
  const [fieldName, setFieldName] = useState(availableFields[0]?.field ?? '')
  const [operator, setOperator] = useState<ProductConditionOperator>('gte')
  const [value, setValue] = useState('')

  useEffect(() => {
    if (!availableFields.some((field) => field.field === fieldName)) setFieldName(availableFields[0]?.field ?? '')
  }, [availableFields, fieldName])

  const selectedField = fields.find((field) => field.field === fieldName)
  const addCondition = () => {
    if (!selectedField || !value.trim()) return
    onChange([...conditions, { field: selectedField.field, operator, value: value.trim() }])
    setValue('')
  }

  return <div className="space-y-3 border-t border-slate-200 pt-4">
    <div className="flex flex-wrap items-center justify-between gap-2"><p className="text-sm font-semibold text-slate-700">日期与快照指标筛选</p><span className={`rounded-full px-3 py-1 text-xs font-semibold ${snapshotStatus === 'ready' ? 'bg-emerald-50 text-emerald-700' : 'bg-amber-50 text-amber-700'}`}>指标快照：{snapshotStatus === 'ready' ? '可用' : '未就绪'}</span></div>
    <div className="grid gap-3 lg:grid-cols-[minmax(0,1.4fr)_minmax(0,1fr)_minmax(0,1fr)_auto]">
      <label className="text-xs font-medium text-slate-600">筛选字段<select aria-label="评价方案筛选字段" value={fieldName} onChange={(event) => { setFieldName(event.target.value); setValue('') }} className="mt-1 min-h-11 w-full rounded-lg border border-slate-200 bg-white px-3 text-sm">{fields.map((field) => <option key={field.field} value={field.field} disabled={!field.available}>{field.label}{field.source === 'instrument_metrics_snapshot' ? '（快照）' : ''}{!field.available ? ' · 未就绪' : ''}</option>)}</select></label>
      <label className="text-xs font-medium text-slate-600">比较方式<select aria-label="评价方案比较方式" value={operator} onChange={(event) => setOperator(event.target.value as ProductConditionOperator)} className="mt-1 min-h-11 w-full rounded-lg border border-slate-200 bg-white px-3 text-sm">{operators.map((item) => <option key={item.value} value={item.value}>{item.label}（{item.symbol}）</option>)}</select></label>
      <label className="text-xs font-medium text-slate-600">筛选值{selectedField?.unit_label ? `（${selectedField.unit_label}）` : ''}<input aria-label="评价方案筛选值" type={selectedField?.data_type === 'date' ? 'date' : 'number'} step={selectedField?.data_type === 'number' ? 'any' : undefined} value={value} onChange={(event) => setValue(event.target.value)} className="mt-1 min-h-11 w-full rounded-lg border border-slate-200 bg-white px-3 text-sm" /></label>
      <button type="button" onClick={addCondition} disabled={!selectedField || !value.trim()} className="min-h-11 self-end rounded-lg bg-slate-900 px-4 text-sm font-semibold text-white disabled:bg-slate-200">添加条件</button>
    </div>
    {conditions.length > 0 && <div className="flex flex-wrap gap-2">{conditions.map((condition, index) => {
      const field = fields.find((item) => item.field === condition.field)
      const itemOperator = operators.find((item) => item.value === condition.operator)
      return <button key={`${condition.field}:${condition.operator}:${condition.value}:${index}`} type="button" onClick={() => onChange(conditions.filter((_, itemIndex) => itemIndex !== index))} className="rounded-full bg-violet-50 px-3 py-1 text-xs font-semibold text-violet-700">{field?.label ?? condition.field} {itemOperator?.symbol ?? condition.operator} {condition.value}{field?.unit_label ?? ''} ×</button>
    })}</div>}
  </div>
}

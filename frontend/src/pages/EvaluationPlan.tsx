import React, { useEffect, useMemo, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import EvaluationProductSelector from '../components/evaluation/EvaluationProductSelector'
import {
  createEvaluationPlan,
  deleteEvaluationPlan,
  getCustomIndicatorMeta,
  indicatorsForContext,
  listCustomIndicators,
  listEvaluationPlans,
  runEvaluationPlan,
  updateEvaluationPlan,
  type EvaluationPlan,
  type EvaluationPlanDraft,
  type EvaluationPlanRunResponse,
  type EvaluationProductSelection,
  type IndicatorDefinition,
  type IndicatorDirection,
  type InstrumentProductItem,
  type ProductKind,
} from '../services/customIndicators'
import { indicatorPeriodOptionLabel } from '../utils/indicatorPeriods'
import { formatIndicatorDiagnostic } from '../utils/indicatorDiagnostics'
import {
  MetricSelector,
  MetricStatus,
  MetricUnavailableReason,
  MetricValue,
  indicatorOptionLabel,
} from '../components/metrics/MetricDisplay'

type IndicatorEntry = {
  id: string
  indicatorId: string
  period: string
  weight: number
  direction: IndicatorDirection
}

const instrumentCode = (item: InstrumentProductItem) => item.ts_code ?? item.code ?? ''
const readProductKind = (value: string | null): ProductKind => value === 'fund' ? 'fund' : 'etf'
const entryKey = (entry: Pick<IndicatorEntry, 'indicatorId' | 'period'>, revision: number) => `${entry.indicatorId}:${revision}:${entry.period}`
const emptyProductSelection = (): EvaluationProductSelection => ({
  query: '',
  filters: {
    fund_type: [],
    invest_type: [],
    market: [],
    status: [],
    management: [],
    custodian: [],
  },
  conditions: [],
  selection_mode: 'manual',
})
const normalizeProductSelection = (selection?: EvaluationProductSelection | null): EvaluationProductSelection => {
  const empty = emptyProductSelection()
  if (!selection) return empty
  return {
    query: selection.query ?? '',
    filters: Object.fromEntries(Object.keys(empty.filters).map((key) => [
      key,
      Array.isArray(selection.filters?.[key as keyof typeof empty.filters])
        ? [...selection.filters[key as keyof typeof empty.filters]]
        : [],
    ])) as EvaluationProductSelection['filters'],
    conditions: Array.isArray(selection.conditions) ? selection.conditions.map((condition) => ({ ...condition })) : [],
    selection_mode: selection.selection_mode === 'all_matching' ? 'all_matching' : 'manual',
  }
}

export default function EvaluationPlanPage() {
  const [searchParams, setSearchParams] = useSearchParams()
  const [productKind, setProductKind] = useState<ProductKind>(() => readProductKind(searchParams.get('kind')))
  const [indicators, setIndicators] = useState<IndicatorDefinition[]>([])
  const [runtimePeriods, setRuntimePeriods] = useState<string[]>(['1Y'])
  const [plans, setPlans] = useState<EvaluationPlan[]>([])
  const [selectedPlanId, setSelectedPlanId] = useState('')
  const [name, setName] = useState('')
  const [description, setDescription] = useState('')
  const [selectedTargetItems, setSelectedTargetItems] = useState<Record<string, InstrumentProductItem>>({})
  const [productSelection, setProductSelection] = useState<EvaluationProductSelection>(emptyProductSelection)
  const [entries, setEntries] = useState<IndicatorEntry[]>([])
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [running, setRunning] = useState(false)
  const [message, setMessage] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [runResult, setRunResult] = useState<EvaluationPlanRunResponse | null>(null)

  const selectedPlan = plans.find((plan) => plan.id === selectedPlanId) ?? null
  const selectedTargets = useMemo(() => Object.values(selectedTargetItems), [selectedTargetItems])
  const totalWeight = entries.reduce((total, entry) => total + entry.weight, 0)
  const defaultPeriod = runtimePeriods.includes('1Y') ? '1Y' : runtimePeriods[0] ?? '1Y'
  const kindLabel = productKind === 'etf' ? 'ETF' : '场外公募基金'

  useEffect(() => {
    let active = true
    setLoading(true)
    setError(null)
    Promise.all([
      listCustomIndicators({ contextKind: 'single_product', productKind, includeCompatibility: true }),
      listEvaluationPlans(productKind),
      getCustomIndicatorMeta(),
    ]).then(([indicatorResponse, planResponse, metadata]) => {
      if (!active) return
      const catalog = indicatorsForContext(indicatorResponse.items, 'single_product')
      const periods = metadata.periods.map((item) => item.value)
      const initialPeriod = periods.includes('1Y') ? '1Y' : periods[0] ?? '1Y'
      const first = catalog.find((item) => item.ui_exposed !== false)
      setIndicators(catalog)
      setPlans(planResponse.items)
      setRuntimePeriods(periods)
      setEntries(first ? [{ id: crypto.randomUUID(), indicatorId: first.id, period: initialPeriod, weight: 100, direction: first.direction }] : [])
    }).catch(() => {
      if (active) setError(`无法加载${kindLabel}评价方案或指标库，请确认后端服务已就绪。`)
    }).finally(() => {
      if (active) setLoading(false)
    })
    return () => { active = false }
  }, [kindLabel, productKind])

  const selectableIndicators = useMemo(
    () => indicators.filter((indicator) => indicator.ui_exposed !== false || entries.some((entry) => entry.indicatorId === indicator.id)),
    [entries, indicators],
  )
  const disabledReasons = useMemo(() => ({} as Record<string, string>), [])
  const selectedMetricIds = [...new Set(entries.map((entry) => entry.indicatorId))]

  const updateSelectedMetrics = (ids: string[]) => setEntries((current) => {
    const kept = current.filter((entry) => ids.includes(entry.indicatorId))
    const existing = new Set(kept.map((entry) => entry.indicatorId))
    const added = ids.filter((id) => !existing.has(id)).map((id) => {
      const definition = indicators.find((indicator) => indicator.id === id)
      return {
        id: crypto.randomUUID(),
        indicatorId: id,
        period: defaultPeriod,
        weight: 10,
        direction: definition?.direction ?? 'higher_better' as const,
      }
    })
    return [...kept, ...added]
  })

  const resetDraft = () => {
    const first = indicators.find((item) => item.ui_exposed !== false)
    setSelectedPlanId('')
    setName('')
    setDescription('')
    setSelectedTargetItems({})
    setProductSelection(emptyProductSelection())
    setRunResult(null)
    setMessage(null)
    setError(null)
    setEntries(first ? [{ id: crypto.randomUUID(), indicatorId: first.id, period: defaultPeriod, weight: 100, direction: first.direction }] : [])
  }
  const switchProductKind = (kind: ProductKind) => {
    if (kind === productKind) return
    setProductKind(kind)
    setSearchParams({ kind })
    setSelectedPlanId('')
    setName('')
    setDescription('')
    setSelectedTargetItems({})
    setProductSelection(emptyProductSelection())
    setEntries([])
    setRunResult(null)
    setMessage(null)
    setError(null)
  }

  const draft = (): EvaluationPlanDraft => ({
    name: name.trim(),
    description: description.trim(),
    product_kind: productKind,
    indicators: entries.map((entry) => ({
      indicator_id: entry.indicatorId,
      indicator_revision: indicators.find((item) => item.id === entry.indicatorId)?.revision ?? 1,
      period: entry.period,
      weight: entry.weight,
      direction: entry.direction,
    })),
    targets: selectedTargets.map((item) => ({ kind: productKind, product_id: instrumentCode(item) })),
    product_selection: productSelection,
    missing_policy: 'strict',
  })
  const validateDraft = () => {
    if (!name.trim()) return '请填写评价方案名称。'
    if (entries.length === 0) return '请至少选择一个指标。'
    if (selectedTargets.length === 0) return `请至少选择一个${kindLabel}产品。`
    if (selectedTargets.some((item) => item.instrument_type && item.instrument_type !== productKind)) return '评价方案中存在其他品类产品，请重新选择。'
    if (totalWeight <= 0) return '指标权重合计必须大于 0。'
    const keys = entries.map((entry) => entryKey(entry, indicators.find((indicator) => indicator.id === entry.indicatorId)?.revision ?? 1))
    if (new Set(keys).size !== keys.length) return '同一指标、版本和周期不能重复配置。'
    const incompatible = entries.find((entry) => disabledReasons[entry.indicatorId])
    return incompatible ? disabledReasons[incompatible.indicatorId] : null
  }

  const savePlan = async (): Promise<EvaluationPlan | null> => {
    const validation = validateDraft()
    if (validation) { setError(validation); return null }
    setSaving(true)
    setError(null)
    setMessage(null)
    try {
      const saved = selectedPlan
        ? await updateEvaluationPlan(selectedPlan.id, draft(), selectedPlan.revision)
        : await createEvaluationPlan(draft())
      setPlans((current) => [saved, ...current.filter((plan) => plan.id !== saved.id)])
      setSelectedPlanId(saved.id)
      setMessage(`已保存${kindLabel}方案“${saved.name}”，并锁定当前产品和指标版本。`)
      return saved
    } catch (caught) {
      setError((caught as { status?: number }).status === 409 ? '该方案已被其他修改覆盖，请刷新后再保存。' : '保存评价方案失败，请稍后重试。')
      return null
    } finally {
      setSaving(false)
    }
  }
  const applyPlan = (plan: EvaluationPlan) => {
    setSelectedPlanId(plan.id)
    setName(plan.name)
    setDescription(plan.description)
    setSelectedTargetItems(Object.fromEntries(plan.targets.map((target) => [target.product_id, {
      code: target.product_id,
      ts_code: target.product_id,
      name: target.product_id,
      management: null,
      found_date: null,
      instrument_type: plan.product_kind,
    }])))
    setProductSelection(normalizeProductSelection(plan.product_selection))
    setEntries(plan.indicators.map((entry) => ({
      id: crypto.randomUUID(),
      indicatorId: entry.indicator_id,
      period: entry.period,
      weight: entry.weight,
      direction: entry.direction,
    })))
    setRunResult(null)
    setMessage(plan.product_selection
      ? null
      : '该旧版方案未保存历史筛选条件；已恢复锁定产品，重新设置筛选并保存后即可完整恢复。')
    setError(null)
  }
  const runPlan = async () => {
    const saved = await savePlan()
    if (!saved) return
    setRunning(true)
    setError(null)
    try {
      const result = await runEvaluationPlan(saved.id)
      setRunResult(result)
      setMessage(`已完成运行：${result.ranked_count} 个产品进入排名，${result.excluded_count} 个产品未排名。`)
    } catch {
      setError('方案已保存，但本次排名运行失败。请检查产品净值和样本窗口。')
    } finally {
      setRunning(false)
    }
  }
  const removePlan = async () => {
    if (!selectedPlan || !window.confirm(`删除方案“${selectedPlan.name}”？`)) return
    try {
      await deleteEvaluationPlan(selectedPlan.id, selectedPlan.revision)
      setPlans((current) => current.filter((plan) => plan.id !== selectedPlan.id))
      resetDraft()
      setMessage('已删除评价方案。')
    } catch {
      setError('无法删除该方案。')
    }
  }
  const updateEntry = (id: string, update: Partial<IndicatorEntry>) => setEntries((current) => current.map((entry) => entry.id === id ? { ...entry, ...update } : entry))

  return <div className="mx-auto max-w-7xl space-y-8 px-4 py-8 sm:px-6">
    <header className="flex flex-col gap-5 lg:flex-row lg:items-end lg:justify-between">
      <div>
        <h1 className="text-3xl font-bold text-slate-900">评价方案</h1>
        <p className="mt-2 max-w-3xl text-slate-600">ETF 与场外公募基金分别配置候选范围和适用指标，使用锁定版本及严格完整样本进行独立排名。</p>
        <div className="mt-4 inline-flex rounded-xl border border-slate-200 bg-white p-1" role="tablist" aria-label="评价方案产品类型">
          {(['etf', 'fund'] as ProductKind[]).map((kind) => <button key={kind} type="button" role="tab" aria-selected={productKind === kind} onClick={() => switchProductKind(kind)} className={`min-h-11 rounded-lg px-5 text-sm font-semibold ${productKind === kind ? 'bg-slate-900 text-white' : 'text-slate-600 hover:text-slate-900'}`}>{kind === 'etf' ? 'ETF' : '场外公募基金'}</button>)}
        </div>
      </div>
      <div className="flex flex-wrap gap-2">
        <button type="button" onClick={resetDraft} className="min-h-11 rounded-lg border border-slate-200 px-4 text-sm font-semibold text-slate-700">新建{kindLabel}方案</button>
        <button type="button" disabled={saving} onClick={() => void savePlan()} className="min-h-11 rounded-lg bg-violet-600 px-4 text-sm font-semibold text-white disabled:bg-violet-300">{saving ? '保存中…' : '保存方案'}</button>
        <button type="button" disabled={saving || running} onClick={() => void runPlan()} className="min-h-11 rounded-lg bg-emerald-600 px-4 text-sm font-semibold text-white disabled:bg-emerald-300">{running ? '运行中…' : '保存并运行'}</button>
      </div>
    </header>
    {(message || error) && <div aria-live="polite" className={`rounded-xl border px-4 py-3 text-sm ${error ? 'border-rose-200 bg-rose-50 text-rose-700' : 'border-emerald-200 bg-emerald-50 text-emerald-700'}`}>{error ?? message}</div>}
    {loading ? <div className="rounded-2xl bg-white p-10 text-center text-slate-500 shadow-sm">正在加载{kindLabel}方案和指标…</div> : <>
      <section className="rounded-2xl bg-white p-6 shadow-sm ring-1 ring-slate-100">
        <div className="grid gap-4 md:grid-cols-[minmax(0,1fr)_minmax(0,2fr)]">
          <label className="text-sm font-medium text-slate-700">已保存{kindLabel}方案<select value={selectedPlanId} onChange={(event) => { const plan = plans.find((item) => item.id === event.target.value); if (plan) applyPlan(plan); else resetDraft() }} className="mt-1 block min-h-11 w-full rounded-lg border border-slate-200 px-3"><option value="">新建方案</option>{plans.map((plan) => <option key={plan.id} value={plan.id}>{plan.name} · v{plan.revision}</option>)}</select></label>
          <div className="grid gap-4 sm:grid-cols-2"><label className="text-sm font-medium text-slate-700">方案名称<input aria-label="方案名称" value={name} onChange={(event) => setName(event.target.value)} className="mt-1 block min-h-11 w-full rounded-lg border border-slate-200 px-3" /></label><label className="text-sm font-medium text-slate-700">说明<input value={description} onChange={(event) => setDescription(event.target.value)} className="mt-1 block min-h-11 w-full rounded-lg border border-slate-200 px-3" /></label></div>
        </div>
        {selectedPlan && <button type="button" onClick={() => void removePlan()} className="mt-3 text-sm font-medium text-rose-600 hover:underline">删除当前方案</button>}
      </section>

      <EvaluationProductSelector
        key={`${productKind}:${selectedPlanId || 'new'}`}
        productKind={productKind}
        selectedItems={selectedTargetItems}
        onSelectedItemsChange={setSelectedTargetItems}
        selectionState={productSelection}
        onSelectionStateChange={setProductSelection}
        onError={setError}
      />

      <IndicatorConfiguration productKind={productKind} indicators={indicators} selectableIndicators={selectableIndicators} disabledReasons={disabledReasons} entries={entries} runtimePeriods={runtimePeriods} selectedMetricIds={selectedMetricIds} totalWeight={totalWeight} onSelectedMetricsChange={updateSelectedMetrics} onEntryChange={updateEntry} onEntriesChange={setEntries} />

      <RunResultSection runResult={runResult} />
    </>}
  </div>
}

function IndicatorConfiguration({ productKind, indicators, selectableIndicators, disabledReasons, entries, runtimePeriods, selectedMetricIds, totalWeight, onSelectedMetricsChange, onEntryChange, onEntriesChange }: {
  productKind: ProductKind
  indicators: IndicatorDefinition[]
  selectableIndicators: IndicatorDefinition[]
  disabledReasons: Record<string, string>
  entries: IndicatorEntry[]
  runtimePeriods: string[]
  selectedMetricIds: string[]
  totalWeight: number
  onSelectedMetricsChange: (ids: string[]) => void
  onEntryChange: (id: string, update: Partial<IndicatorEntry>) => void
  onEntriesChange: React.Dispatch<React.SetStateAction<IndicatorEntry[]>>
}) {
  const kindLabel = productKind === 'etf' ? 'ETF' : '场外公募基金'
  return <section className="rounded-2xl bg-white p-6 shadow-sm ring-1 ring-slate-100">
    <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between"><div><h2 className="text-xl font-semibold text-slate-900">2. 配置{kindLabel}研究指标</h2><p className="mt-1 text-sm text-slate-500">指标目录只展示适用于当前品类的定义；输入权重运行时归一化为 100%，保存时锁定版本。</p></div><div className="flex items-center gap-3"><span className="rounded-full bg-violet-50 px-3 py-1 text-sm font-semibold text-violet-700">输入合计 {totalWeight}</span><MetricSelector indicators={selectableIndicators} selectedIds={selectedMetricIds} onChange={onSelectedMetricsChange} maxSelected={10} disabledReasons={disabledReasons} /></div></div>
    <div className="mt-4 overflow-auto"><table className="min-w-[980px] w-full text-sm"><caption className="sr-only">{kindLabel}评价指标配置</caption><thead className="bg-slate-50 text-left text-slate-500"><tr><th scope="col" className="px-3 py-3">指标定义</th><th scope="col" className="px-3 py-3">方向</th><th scope="col" className="px-3 py-3">周期</th><th scope="col" className="px-3 py-3">输入权重</th><th scope="col" className="px-3 py-3">有效占比</th><th scope="col" className="px-3 py-3">操作</th></tr></thead><tbody>{entries.map((entry) => {
      const definition = indicators.find((item) => item.id === entry.indicatorId)
      const overridden = definition && entry.direction !== definition.direction
      return <tr key={entry.id} className="border-t border-slate-100"><td className="px-3 py-3"><p className="font-medium text-slate-800">{definition ? indicatorOptionLabel(definition) : entry.indicatorId}</p><p className="mt-1 text-xs text-slate-500">{definition?.presentation?.category_label ?? definition?.category_label ?? '未分类'} · {definition?.unit || '无单位'} · 最少 {definition?.minimum_observations ?? 1} 个观察值</p>{disabledReasons[entry.indicatorId] && <p className="mt-1 text-xs text-rose-600">{disabledReasons[entry.indicatorId]}</p>}</td><td className="px-3 py-3"><select value={entry.direction} onChange={(event) => onEntryChange(entry.id, { direction: event.target.value as IndicatorDirection })} className="min-h-11 rounded border border-slate-200 px-2"><option value="higher_better">数值高优先</option><option value="lower_better">数值低优先</option></select>{overridden && <p className="mt-1 text-xs text-amber-700">已覆盖指标默认方向</p>}</td><td className="px-3 py-3"><select value={entry.period} onChange={(event) => onEntryChange(entry.id, { period: event.target.value })} className="min-h-11 rounded border border-slate-200 px-2">{runtimePeriods.map((period) => <option key={period} value={period}>{indicatorPeriodOptionLabel(period)}</option>)}</select></td><td className="px-3 py-3"><input aria-label={`${definition?.name ?? '指标'}权重`} type="number" min="0" value={entry.weight} onChange={(event) => onEntryChange(entry.id, { weight: Number(event.target.value) || 0 })} className="min-h-11 w-24 rounded border border-slate-200 px-2 text-right" /></td><td className="px-3 py-3 font-medium text-violet-700">{totalWeight > 0 ? `${(entry.weight / totalWeight * 100).toFixed(1)}%` : '—'}</td><td className="px-3 py-3"><div className="flex gap-2"><button type="button" onClick={() => onEntriesChange((current) => [...current, { ...entry, id: crypto.randomUUID(), period: runtimePeriods.find((period) => !current.some((item) => item.indicatorId === entry.indicatorId && item.period === period)) ?? entry.period }])} className="text-violet-700 hover:underline">复制周期</button><button type="button" onClick={() => onEntriesChange((current) => current.filter((item) => item.id !== entry.id))} className="text-rose-600 hover:underline">删除</button></div></td></tr>
    })}</tbody></table></div>
  </section>
}

const evaluationExclusionReason = (row: EvaluationPlanRunResponse['rows'][number]) => {
  const blockingInputs = row.values.flatMap((value) => value.input_requirements?.blocking_inputs ?? [])
  if (blockingInputs.length) {
    const distinct = [...new Map(blockingInputs.map((item) => [item.variable_id, item])).values()]
    return distinct.map((item) => `${item.label}：${item.reason || '当前没有可用数据'}`).join('；')
  }
  return row.exclusion_reasons?.map((reason) => formatIndicatorDiagnostic(reason.code, reason.message)).join('；') || row.missing_indicators.join('、')
}

function RunResultSection({ runResult }: { runResult: EvaluationPlanRunResponse | null }) {
  return <section className="rounded-2xl bg-white p-6 shadow-sm ring-1 ring-slate-100">
    <div className="flex flex-col gap-2 md:flex-row md:items-end md:justify-between"><div><h2 className="text-xl font-semibold text-slate-900">3. 运行结果</h2><p className="mt-1 text-sm text-slate-500">总分由 0–100 归一化得分乘有效权重后求和；缺少任一值的产品不参与排名。</p></div>{runResult && <p className="text-xs text-slate-500">方案 v{runResult.plan_revision} · {new Date(runResult.run_at).toLocaleString('zh-CN')} · 截止日 {runResult.as_of ?? '最新数据'}</p>}</div>
    {runResult ? <div className="mt-4 overflow-auto"><table className="min-w-[900px] w-full text-sm"><caption className="sr-only">评价方案排名结果</caption><thead className="bg-slate-50 text-left text-slate-500"><tr><th scope="col" className="px-3 py-3">排名</th><th scope="col" className="px-3 py-3">产品</th><th scope="col" className="px-3 py-3">状态</th><th scope="col" className="px-3 py-3">总分</th><th scope="col" className="px-3 py-3">贡献明细</th></tr></thead><tbody>{runResult.rows.map((row) => <tr key={`${row.target.kind}:${row.target.product_id}`} className="border-t border-slate-100 align-top"><td className="px-3 py-3">{row.rank ?? '—'}</td><td className="px-3 py-3 font-medium text-slate-800">{row.target.name}<span className="ml-2 text-xs font-normal text-slate-400">{row.target.product_id}</span></td><td className="px-3 py-3">{row.status === 'ranked' ? <span className="text-emerald-700">已排名</span> : <span className="text-amber-700">不可排名</span>}</td><td className="px-3 py-3 font-semibold text-violet-700">{row.score?.toFixed(2) ?? '—'}</td><td className="px-3 py-3"><details><summary className="cursor-pointer font-medium text-violet-700">{row.status === 'ranked' ? '查看原始值、得分与贡献' : '查看不可排名原因'}</summary><div className="mt-3 space-y-3">{row.values.map((value) => <div key={`${value.indicator_id}:${value.period}`} className="rounded-lg bg-slate-50 p-3"><div className="flex flex-wrap items-center justify-between gap-2"><span className="font-medium text-slate-800">{value.indicator_name} · {value.period}</span><MetricStatus status={value.status} warnings={value.warnings} showReason={value.value === null && !value.input_requirements} /></div><div className="mt-2 grid gap-2 text-xs text-slate-600 sm:grid-cols-4"><span>原始值：<MetricValue value={value.value} presentation={value.presentation} /></span><span>归一化：{value.normalized_score?.toFixed(2) ?? '—'}</span><span>有效权重：{value.effective_weight === null ? '—' : `${(value.effective_weight * 100).toFixed(1)}%`}</span><span>加权贡献：{value.weighted_contribution?.toFixed(2) ?? '—'}</span></div><p className="mt-2 text-xs text-slate-500">实际窗口 {value.window?.start_date ?? '—'} 至 {value.window?.end_date ?? '—'} · {value.window?.observation_count ?? 0} 个观察值</p><MetricUnavailableReason result={value} /></div>)}</div></details></td></tr>)}</tbody></table></div> : <div className="mt-4 rounded-xl bg-slate-50 p-6 text-sm text-slate-500">保存并运行方案后，在这里查看排名、有效权重和贡献拆解。</div>}
    {runResult && runResult.excluded_count > 0 && <div className="mt-4 rounded-xl border border-amber-200 bg-amber-50 p-4"><h3 className="font-semibold text-amber-900">不可排名产品</h3>{runResult.rows.filter((row) => row.status === 'excluded').map((row) => <p key={`${row.target.kind}:${row.target.product_id}`} className="mt-2 text-sm text-amber-800">{row.target.name}：{evaluationExclusionReason(row)}</p>)}</div>}
  </section>
}

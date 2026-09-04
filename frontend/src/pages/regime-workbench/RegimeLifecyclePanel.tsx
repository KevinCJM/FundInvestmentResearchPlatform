import { useEffect, useMemo, useRef, useState } from 'react'
import {
  compareRegimeFormalRuns,
  definitionForRequest,
  getRegimeFormalRun,
  listRegimeFormalRuns,
  prepareRegimeGraph,
  publishRegimeFormalRun,
  runSavedRegimeGraph,
  type RegimeFormalRun,
  type RegimeGraphDefinition,
  type RegimeMode,
  type PreparedRegimeGraph,
  type RegimePublicationUsage,
  type RegimeRunComparison,
} from '../../services/regimeGraph'
import RegimeEvaluationResults from './RegimeEvaluationResults'
import RegimeHelpTip from './RegimeHelpTip'

const USAGES: Array<{ id: RegimePublicationUsage; label: string }> = [
  { id: 'research_display', label: '研究展示' },
  { id: 'product_research', label: '产品研究' },
  { id: 'formal_backtest', label: '正式回测' },
  { id: 'taa', label: '战术资产配置' },
]

function errorText(reason: unknown, fallback: string) {
  if (reason instanceof DOMException && reason.name === 'AbortError') return ''
  return reason instanceof Error ? reason.message : fallback
}

function dateTime(value?: string) {
  if (!value) return '—'
  const date = new Date(value)
  return Number.isFinite(date.getTime()) ? date.toLocaleString('zh-CN', { hour12: false }) : value
}

function percentage(value?: number | null) {
  return value == null ? '—' : `${(value * 100).toFixed(2)}%`
}

export interface RegimeLifecyclePanelProps {
  definition: RegimeGraphDefinition
  dirty: boolean
  valid: boolean
  mode: RegimeMode
  asOf: string
  onError: (message: string) => void
  onNotice: (message: string) => void
}

export default function RegimeLifecyclePanel({ definition, dirty, valid, mode, asOf, onError, onNotice }: RegimeLifecyclePanelProps) {
  const [runs, setRuns] = useState<RegimeFormalRun[]>([])
  const [selectedRunId, setSelectedRunId] = useState('')
  const [selectedRunDetail, setSelectedRunDetail] = useState<RegimeFormalRun | null>(null)
  const [loadingRunDetail, setLoadingRunDetail] = useState(false)
  const [compareIds, setCompareIds] = useState<string[]>([])
  const [referenceRunId, setReferenceRunId] = useState('')
  const [comparison, setComparison] = useState<RegimeRunComparison | null>(null)
  const [usage, setUsage] = useState<RegimePublicationUsage>('research_display')
  const [note, setNote] = useState('')
  const [loadingRuns, setLoadingRuns] = useState(false)
  const [plan, setPlan] = useState<PreparedRegimeGraph | null>(null)
  const [preparing, setPreparing] = useState(false)
  const [running, setRunning] = useState(false)
  const [comparing, setComparing] = useState(false)
  const [publishing, setPublishing] = useState(false)
  const requestId = useRef(0)
  const detailRequestId = useRef(0)

  const selectedRun = selectedRunDetail?.id === selectedRunId ? selectedRunDetail : null
  const eligibleUsages = selectedRun?.causality?.publish_eligible_usages ?? []
  const canPrepare = Boolean(definition.id && definition.revision && valid && !dirty && !preparing && !running)
  const canRun = Boolean(canPrepare && plan?.compile_token)

  const loadRuns = async (signal?: AbortSignal) => {
    const currentRequest = ++requestId.current
    if (!definition.id) { setRuns([]); setSelectedRunId(''); return }
    setLoadingRuns(true)
    try {
      const response = await listRegimeFormalRuns(definition.id, signal)
      if (currentRequest !== requestId.current) return
      setRuns(response)
      setSelectedRunId((current) => response.some((run) => run.id === current) ? current : response[0]?.id || '')
      setCompareIds((current) => current.filter((id) => response.some((run) => run.id === id)))
    } catch (reason) {
      if (!signal?.aborted && currentRequest === requestId.current) onError(errorText(reason, '正式运行记录加载失败。'))
    } finally { if (currentRequest === requestId.current) setLoadingRuns(false) }
  }

  useEffect(() => {
    const controller = new AbortController()
    void loadRuns(controller.signal)
    return () => { controller.abort(); requestId.current += 1 }
  }, [definition.id])

  useEffect(() => {
    const summary = runs.find((run) => run.id === selectedRunId)
    const currentRequest = ++detailRequestId.current
    const controller = new AbortController()
    if (!summary) { setSelectedRunDetail(null); setLoadingRunDetail(false); return () => controller.abort() }
    if (summary.series_included !== false && Array.isArray(summary.series)) {
      setSelectedRunDetail(summary); setLoadingRunDetail(false)
      return () => controller.abort()
    }
    setSelectedRunDetail(null); setLoadingRunDetail(true)
    void getRegimeFormalRun(summary.id, controller.signal)
      .then((detail) => { if (currentRequest === detailRequestId.current) setSelectedRunDetail(detail) })
      .catch((reason) => { if (!controller.signal.aborted && currentRequest === detailRequestId.current) onError(errorText(reason, '正式运行详情加载失败。')) })
      .finally(() => { if (currentRequest === detailRequestId.current) setLoadingRunDetail(false) })
    return () => { controller.abort(); detailRequestId.current += 1 }
  }, [runs, selectedRunId])

  useEffect(() => { setPlan(null) }, [definition.id, definition.revision, dirty, JSON.stringify(definitionForRequest(definition))])

  const prepareSavedVersion = async () => {
    if (!canPrepare) return
    setPreparing(true); onError('')
    try {
      const prepared = await prepareRegimeGraph(definition)
      setPlan(prepared)
      onNotice(`固定签名计划已显式预热 · ${prepared.plan_id}。`)
    } catch (reason) { onError(errorText(reason, '显式预热计划失败。')) } finally { setPreparing(false) }
  }

  const startFormalRun = async () => {
    if (!canRun || !plan) return
    setRunning(true); onError(''); setComparison(null)
    try {
      const next = await runSavedRegimeGraph(definition, plan.compile_token, mode, asOf || undefined)
      setRuns((current) => [next, ...current.filter((run) => run.id !== next.id)])
      setSelectedRunDetail(next)
      setSelectedRunId(next.id)
      onNotice(`正式运行已生成不可变快照 · ${next.id}。`)
    } catch (reason) { onError(errorText(reason, '正式运行失败。')) } finally { setRunning(false) }
  }

  const toggleCompare = (runId: string) => {
    setComparison(null)
    setCompareIds((current) => {
      if (current.includes(runId)) return current.filter((id) => id !== runId)
      if (current.length >= 8) { onError('一次最多比较 8 个正式运行。'); return current }
      const next = [...current, runId]
      if (!referenceRunId) setReferenceRunId(next[0])
      return next
    })
  }

  const compareRuns = async () => {
    if (compareIds.length < 2) return
    setComparing(true); onError('')
    try {
      const response = await compareRegimeFormalRuns(compareIds, compareIds.includes(referenceRunId) ? referenceRunId : compareIds[0])
      setComparison(response)
    } catch (reason) { onError(errorText(reason, '正式运行比较失败。')) } finally { setComparing(false) }
  }

  const publishRun = async () => {
    if (!selectedRun || !eligibleUsages.includes(usage)) return
    setPublishing(true); onError('')
    try {
      const response = await publishRegimeFormalRun(selectedRun.id, usage, note.trim())
      const refreshed = await getRegimeFormalRun(response.run_id)
      setRuns((current) => current.map((run) => run.id === refreshed.id ? refreshed : run))
      setSelectedRunDetail(refreshed)
      setNote('')
      onNotice(`已发布至“${USAGES.find((item) => item.id === usage)?.label || usage}”。`)
    } catch (reason) { onError(errorText(reason, '正式运行发布失败。')) } finally { setPublishing(false) }
  }

  const comparisonPairs = useMemo(() => comparison?.pairwise ?? [], [comparison])

  return <section id="regime-formal-experiments" className="scroll-mt-20 rounded-2xl border border-slate-200 bg-white p-4 shadow-sm" aria-label="历史情景正式实验与治理">
    <div className="flex flex-col gap-3 border-b border-slate-200 pb-3 sm:flex-row sm:items-start sm:justify-between"><div><p className="text-[10px] font-bold tracking-[0.18em] text-indigo-600">版本化正式实验</p><h3 className="mt-1 text-base font-bold text-slate-950">正式实验、比较与发布 <RegimeHelpTip label="正式实验说明" text="试算可随时调参；正式运行固定定义修订、数据快照和计算计划，生成可追溯且不可改写的结果，随后才选择下游发布目标。" /></h3><p className="mt-1 text-xs leading-5 text-slate-500">试算用于调参；正式运行只接受已保存精确修订和用户显式预热的当前计划。</p>{plan ? <p role="status" className="mt-1 text-[10px] font-semibold text-emerald-700">计算计划已就绪，可执行正式运行。</p> : <p className="mt-1 text-[10px] text-amber-700">尚未显式预热；正式运行按钮保持关闭。</p>}</div><div className="flex flex-wrap gap-2"><button type="button" disabled={!definition.id || loadingRuns} onClick={() => void loadRuns()} title="重新读取该定义的不可变正式运行记录" className="min-h-10 rounded-lg border border-slate-300 px-3 text-xs font-bold text-slate-700 disabled:opacity-40">刷新记录</button><button type="button" disabled={!canPrepare} onClick={() => void prepareSavedVersion()} title="为当前已保存精确版本编译并锁定固定签名 NJIT 计划" className="min-h-10 rounded-lg border border-indigo-300 px-3 text-xs font-bold text-indigo-700 disabled:opacity-40">{preparing ? '正在显式预热…' : '显式预热计划'}</button><button type="button" disabled={!canRun} onClick={() => void startFormalRun()} title="使用已保存定义、数据快照和预热计划生成不可变正式结果" className="min-h-10 rounded-lg bg-indigo-600 px-4 text-xs font-bold text-white disabled:opacity-40">{running ? '正在正式运行…' : '运行已保存版本'}</button></div></div>
    {!definition.id ? <p className="mt-3 rounded-xl border border-dashed border-slate-300 bg-slate-50 p-4 text-sm text-slate-600">先在版本管理中保存完整计算图，才能执行正式运行、比较和发布。</p> : dirty ? <p className="mt-3 rounded-xl border border-amber-200 bg-amber-50 p-3 text-xs text-amber-900">当前画布有未保存改动；正式运行仍严格引用 r{definition.revision}，请先保存修订或载回服务端版本。</p> : null}
    <div className="mt-4 grid gap-4 xl:grid-cols-[minmax(300px,.85fr)_minmax(0,1.15fr)]">
      <div className="min-w-0"><div className="flex items-center justify-between"><h4 className="text-sm font-bold text-slate-900">不可变运行记录 <RegimeHelpTip label="不可变运行记录说明" text="每条记录都锁定定义版本、数据快照和计算审计。勾选两条以上可比较状态分歧。" /></h4><span className="text-[10px] text-slate-500">{loadingRuns ? '读取中…' : `${runs.length} 条`}</span></div><div className="mt-2 max-h-96 space-y-2 overflow-y-auto pr-1">{runs.map((run) => <article key={run.id} className={`rounded-xl border p-3 ${run.id === selectedRunId ? 'border-indigo-400 bg-indigo-50' : 'border-slate-200'}`}><div className="flex items-start gap-2"><input aria-label={`选择比较${run.id}`} title="加入版本比较" type="checkbox" checked={compareIds.includes(run.id)} onChange={() => toggleCompare(run.id)} className="mt-1" /><button type="button" onClick={() => setSelectedRunId(run.id)} className="min-w-0 flex-1 text-left"><span className="block truncate text-xs font-bold text-slate-900">{run.name || '历史情景正式运行'}</span><span className="mt-1 block text-[10px] text-slate-500">第 {run.definition_revision} 版 · {run.mode === 'realtime' ? '实时识别' : '事后研究'} · {dateTime(run.created_at)}</span></button><span className="rounded bg-emerald-100 px-1.5 py-1 text-[9px] font-bold text-emerald-800">不可变</span></div></article>)}{!loadingRuns && !runs.length ? <p className="rounded-xl border border-dashed border-slate-300 p-4 text-center text-xs text-slate-500">该定义还没有正式运行记录。</p> : null}</div></div>
      <div className="min-w-0 space-y-4">
        <section className="rounded-xl border border-slate-200 p-3" aria-label="正式运行详情"><h4 className="text-sm font-bold text-slate-900">运行详情</h4>{loadingRunDetail ? <p role="status" className="mt-3 text-xs text-indigo-700">正在按需读取所选运行的完整详情…</p> : selectedRun ? <div className="mt-3 space-y-2 text-xs text-slate-600"><dl className="grid grid-cols-2 gap-2 sm:grid-cols-4"><div><dt className="text-[10px] text-slate-500">运行 ID</dt><dd className="mt-1 truncate font-bold text-slate-800" title={selectedRun.id}>{selectedRun.id}</dd></div><div><dt className="text-[10px] text-slate-500">定义修订</dt><dd className="mt-1 font-bold text-slate-800">r{selectedRun.definition_revision}</dd></div><div><dt className="text-[10px] text-slate-500">序列制品</dt><dd className="mt-1 font-bold text-slate-800">{selectedRun.artifact_manifest?.series?.row_count ?? selectedRun.series?.length ?? '—'} 行</dd><dd className="truncate text-[9px] text-slate-500" title={selectedRun.artifact_manifest?.series?.checksum}>{selectedRun.artifact_manifest?.series?.checksum || selectedRun.artifact_manifest?.checksum || '校验值未返回'}</dd></div><div><dt className="text-[10px] text-slate-500">节点输出制品</dt><dd className="mt-1 font-bold text-slate-800">{selectedRun.artifact_manifest?.node_outputs?.arrays?.length ?? '—'} 个数组</dd><dd className="truncate text-[9px] text-slate-500" title={selectedRun.artifact_manifest?.node_outputs?.checksum}>{selectedRun.artifact_manifest?.node_outputs?.checksum || '校验值未返回'}</dd></div></dl><dl className="grid grid-cols-2 gap-2 rounded-lg bg-slate-50 p-2 sm:grid-cols-4"><div><dt className="text-[9px] text-slate-500">稳定性状态</dt><dd className="font-bold text-slate-800">{selectedRun.stability?.status || '—'}</dd></div><div><dt className="text-[9px] text-slate-500">状态切换</dt><dd className="font-bold text-slate-800">{selectedRun.stability?.state_switches ?? '—'}</dd></div><div><dt className="text-[9px] text-slate-500">分类覆盖</dt><dd className="font-bold text-slate-800">{percentage(selectedRun.stability?.classified_ratio)}</dd></div><div><dt className="text-[9px] text-slate-500">Walk-forward</dt><dd className="font-bold text-slate-800">{selectedRun.walk_forward?.status || '—'} · {selectedRun.walk_forward?.fold_count ?? selectedRun.walk_forward?.folds?.length ?? 0} 折</dd></div></dl><RegimeEvaluationResults results={selectedRun.evaluation_results} mode="formal" />{selectedRun.causality?.blockers?.length ? <ul className="rounded-lg bg-amber-50 p-2 text-amber-900">{selectedRun.causality.blockers.map((item) => <li key={item}>• {item}</li>)}</ul> : null}<p className="text-[10px] text-slate-500">已发布 {selectedRun.publications?.length ?? 0} 次；数值执行审计已在读取结果时强制校验。</p></div> : <p className="mt-3 text-xs text-slate-500">选择一条运行查看血缘和发布门禁。</p>}</section>
        <section className="rounded-xl border border-slate-200 p-3" aria-label="正式运行比较">
          <div className="flex flex-wrap items-end justify-between gap-2"><div><h4 className="text-sm font-bold text-slate-900">版本比较 <RegimeHelpTip label="版本比较说明" text="比较多个正式运行在共同日期上的状态一致率、状态边界距离和连续分歧区间。" /></h4><p className="mt-1 text-[10px] text-slate-500">在左侧勾选 2–8 条运行；一致率、共同样本、边界距离和分歧区间均由后端 NJIT 返回。</p></div><div className="flex gap-2"><label className="text-[10px] font-bold text-slate-600">比较基准<RegimeHelpTip label="比较基准说明" text="其他运行都相对这条记录计算参数差异与状态分歧。" /><select aria-label="比较基准运行" value={compareIds.includes(referenceRunId) ? referenceRunId : compareIds[0] || ''} onChange={(event) => setReferenceRunId(event.target.value)} className="mt-1 block min-h-9 max-w-48 rounded-lg border border-slate-300 bg-white px-2 text-xs font-normal"><option value="">选择基准</option>{compareIds.map((id) => <option key={id} value={id}>{runs.find((run) => run.id === id)?.name || '已选运行'}</option>)}</select></label><button type="button" disabled={compareIds.length < 2 || comparing} onClick={() => void compareRuns()} className="min-h-9 self-end rounded-lg bg-slate-950 px-3 text-xs font-bold text-white disabled:opacity-40">{comparing ? '比较中…' : `比较 ${compareIds.length} 个版本`}</button></div></div>
          {comparison ? <div className="mt-3 space-y-2"><p className="rounded-lg bg-indigo-50 p-2 text-xs font-bold text-indigo-900">整体一致率：{percentage(comparison.agreement_rate)} · 分歧区间 {comparison.disagreement_periods?.length ?? 0} 个</p>{comparisonPairs.length ? <div className="overflow-x-auto"><table className="min-w-full text-left text-[10px]" aria-label="运行两两比较"><thead className="text-slate-500"><tr><th className="py-1">运行组合</th><th className="text-right">共同样本</th><th className="text-right">一致率</th><th className="text-right">边界距离</th></tr></thead><tbody>{comparisonPairs.map((pair) => <tr key={`${pair.left_run_id}-${pair.right_run_id}`} className="border-t border-slate-100"><td className="max-w-52 truncate py-1" title={`${pair.left_run_id} / ${pair.right_run_id}`}>{pair.left_run_id} / {pair.right_run_id}</td><td className="text-right">{pair.common_observations ?? '—'}</td><td className="text-right">{percentage(pair.agreement_rate)}</td><td className="text-right">{pair.boundary_distance ?? '—'}</td></tr>)}</tbody></table></div> : null}{comparison.disagreement_periods?.length ? <div className="max-h-36 overflow-auto rounded-lg border border-slate-100"><table className="min-w-full text-left text-[10px]" aria-label="运行分歧区间"><thead className="sticky top-0 bg-slate-50 text-slate-500"><tr><th className="px-2 py-1">开始</th><th className="px-2 py-1">结束</th><th className="px-2 py-1">各运行状态</th></tr></thead><tbody>{comparison.disagreement_periods.slice(0, 20).map((period, index) => <tr key={`${period.start_date}-${period.end_date}-${index}`} className="border-t border-slate-100"><td className="px-2 py-1">{period.start_date}</td><td className="px-2 py-1">{period.end_date}</td><td className="px-2 py-1 font-mono">{Object.entries(period.states).map(([id, state]) => `${id}: ${state}`).join('；')}</td></tr>)}</tbody></table></div> : <p className="text-[10px] text-emerald-700">后端未识别到分歧区间。</p>}</div> : null}
        </section>
        <section className="rounded-xl border border-slate-200 p-3" aria-label="正式运行发布"><h4 className="text-sm font-bold text-slate-900">发布与下游联动 <RegimeHelpTip label="发布与下游联动说明" text="研究图谱和模板不预设用途；这里只为一条已经固定的正式运行选择实际发布目标，并执行该目标要求的门禁。" /></h4><div className="mt-2 grid gap-2 sm:grid-cols-[minmax(130px,.5fr)_minmax(180px,1fr)_auto]"><label className="text-[10px] font-bold text-slate-600">发布目标<RegimeHelpTip label="发布目标说明" text="选择这次不可变运行要提供给哪个业务。正式回测和战术资产配置要求更严格的时点、因果性和稳定性检查。" /><select aria-label="发布应用目标" value={usage} onChange={(event) => setUsage(event.target.value as RegimePublicationUsage)} className="mt-1 min-h-9 w-full rounded-lg border border-slate-300 bg-white px-2 text-xs font-normal">{USAGES.map((item) => <option key={item.id} value={item.id}>{item.label}</option>)}</select></label><label className="text-[10px] font-bold text-slate-600">发布说明<RegimeHelpTip label="发布说明帮助" text="记录本次发布的用途、假设或审批说明，便于下游审计。" /><input aria-label="发布说明" value={note} onChange={(event) => setNote(event.target.value)} maxLength={500} className="mt-1 min-h-9 w-full rounded-lg border border-slate-300 px-2 text-xs font-normal" /></label><button type="button" disabled={!selectedRun || !eligibleUsages.includes(usage) || publishing} onClick={() => void publishRun()} className="min-h-9 self-end rounded-lg bg-emerald-600 px-3 text-xs font-bold text-white disabled:opacity-40">{publishing ? '发布中…' : '发布运行'}</button></div>{selectedRun && !eligibleUsages.includes(usage) ? <p className="mt-2 text-[10px] text-amber-800">该不可变运行未通过当前发布目标的门禁。</p> : null}</section>
      </div>
    </div>
  </section>
}

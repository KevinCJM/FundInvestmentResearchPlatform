import { useEffect, useMemo, useRef, useState } from 'react'
import type { RegimeGraphConnection } from '../../services/regimeGraph'
import RegimeComparisonPicker, { comparisonKey, comparisonOptions } from './RegimeComparisonPicker'
import { RegimeGraphApiError, cancelRegimePreviewRun, cloneRegimeGraphDefinition, definitionForRequest, getRegimePreviewRun, getAllRegimePreviewSeries, prepareRegimeGraph, startRegimePreviewRun, type RegimeGraphDiagnostic, type RegimeGraphDefinition, type RegimeMode, type RegimeNodeSchema, type RegimePreviewRun, type RegimeSeriesPage } from '../../services/regimeGraph'
import { alignRegimeBinaryInputs } from './regimeGraphEditing'
import { regimePortLabel } from './regimeDisplay'
import RegimeResultDock from './RegimeResultDock'

export default function RegimeNodePreviewPanel({ definition, schemas, initialNodeId, initialMode, initialAsOf, onChange }: {
  definition: RegimeGraphDefinition; schemas: RegimeNodeSchema[]; initialNodeId?: string | null; initialMode: RegimeMode; initialAsOf: string; onChange?: (next: RegimeGraphDefinition) => void
}) {
  const [nodeId, setNodeId] = useState(initialNodeId || definition.graph.nodes[0]?.id || '')
  const node = definition.graph.nodes.find(item => item.id === nodeId)
  const schema = schemas.find(item => [item.id, item.type, item.type_id].includes(node?.type))
  const [port, setPort] = useState('')
  const outputPort = port || schema?.outputs.find(item => item.id === 'value')?.id || schema?.outputs[0]?.id || ''
  const [mode, setMode] = useState(initialMode)
  const [asOf, setAsOf] = useState(initialAsOf)
  const [comparisons, setComparisons] = useState<RegimeGraphConnection[]>([])
  const comparisonChoices = useMemo(() => comparisonOptions(definition, schemas, { node_id: nodeId, port: outputPort }, mode), [definition, schemas, nodeId, outputPort, mode])
  const comparisonTargets = comparisons.filter(ref => ref.node_id !== nodeId || ref.port !== outputPort)
  const comparisonError = comparisonTargets.map(ref => {
    const item = comparisonChoices.find(choice => comparisonKey(choice) === comparisonKey(ref))
    return item ? (item.reason ? `${item.label}：${item.reason}` : '') : '已选对比节点不存在，请清空后重新选择。'
  }).find(Boolean)
  const [run, setRun] = useState<RegimePreviewRun | null>(null)
  const [page, setPage] = useState<RegimeSeriesPage | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [diagnostics, setDiagnostics] = useState<RegimeGraphDiagnostic[]>([])
  const [notice, setNotice] = useState('')
  const [signature, setSignature] = useState('')
  const comparisonPickerRef = useRef<HTMLDetailsElement>(null)
  const openComparisons = () => {
    const picker = comparisonPickerRef.current
    if (!picker) return
    picker.open = true
    picker.querySelector('input')?.focus({ preventScroll: true })
    picker.scrollIntoView?.({ block: 'start', behavior: 'smooth' })
  }
  const controllerRef = useRef<AbortController>()
  const jobRef = useRef('')
  const currentSignature = JSON.stringify([definitionForRequest(definition), nodeId, outputPort, mode, asOf, comparisonTargets])
  const stale = Boolean(signature && currentSignature !== signature)
  const alignmentSchema = schemas.find(item => item.id === 'align.strict_intersection' && item.available !== false)
  const alignmentNodes = definition.graph.nodes.filter(item =>
    ['math.add', 'math.subtract', 'math.multiply', 'math.divide'].includes(item.type) && item.inputs.left && item.inputs.right &&
    diagnostics.some(issue => issue.code === 'EXPLICIT_ALIGNMENT_REQUIRED' && (issue.node_id === item.id || issue.path === `graph.nodes.${item.id}.inputs`)))
  useEffect(() => () => {
    controllerRef.current?.abort()
    if (jobRef.current) void cancelRegimePreviewRun(jobRef.current).catch(() => undefined)
  }, [])

  const preview = async (draft = definition) => {
    if (!node || !outputPort || comparisonError) return
    controllerRef.current?.abort()
    if (jobRef.current) void cancelRegimePreviewRun(jobRef.current).catch(() => undefined)
    const controller = new AbortController(); controllerRef.current = controller
    const snapshot = cloneRegimeGraphDefinition(draft)
    const target = { node_id: nodeId, port: outputPort }
    setBusy(true); setError(''); setDiagnostics([]); setPage(null)
    setSignature(JSON.stringify([definitionForRequest(snapshot), nodeId, outputPort, mode, asOf, comparisonTargets]))
    setRun({ id: '', status: 'preparing', message: '正在检查所选节点及其上游。' })
    try {
      const prepared = await prepareRegimeGraph(snapshot, controller.signal, target, comparisonTargets)
      if (controller.signal.aborted) return
      let next = await startRegimePreviewRun(snapshot, { compileToken: prepared.compile_token, mode, asOf: asOf || undefined, previewTarget: target, comparisonTargets }, controller.signal)
      if (controller.signal.aborted) { void cancelRegimePreviewRun(next.id).catch(() => undefined); return }
      jobRef.current = next.id
      while (!controller.signal.aborted) {
        setRun(next)
        if (['completed', 'failed', 'cancelled'].includes(next.status)) break
        await new Promise(resolve => window.setTimeout(resolve, 350))
        if (controller.signal.aborted) return
        next = await getRegimePreviewRun(next.id, controller.signal)
      }
      if (controller.signal.aborted) return
      jobRef.current = ''
      if (next.status !== 'completed') return
      // Read every page so the chart covers the full timeline, including the latest observations.
      const combined = await getAllRegimePreviewSeries(next.id, target.node_id, target.port, controller.signal)
      if (controller.signal.aborted) return
      setPage(combined)
    } catch (reason) {
      if (!controller.signal.aborted) {
        const message = reason instanceof Error ? reason.message : '节点预览失败，请检查所选节点的输入和参数。'
        setError(message)
        setDiagnostics(reason instanceof RegimeGraphApiError ? reason.diagnostics : [])
        setRun({ id: '', status: 'failed', error: { message } })
      }
    } finally { if (!controller.signal.aborted) setBusy(false) }
  }

  const alignAndPreview = () => {
    if (!onChange || !alignmentSchema || stale || busy) return
    try {
      const next = alignRegimeBinaryInputs(definition, alignmentNodes.map(item => item.id), alignmentSchema.type_version ?? alignmentSchema.version ?? 1)
      onChange(next)
      setNotice('已添加共同日期对齐节点，可在画布中查看或撤销。')
      void preview(next)
    } catch (reason) { setError(reason instanceof Error ? reason.message : '添加日期对齐失败。') }
  }

  return <section className="flex min-h-full flex-col gap-4" aria-label="独立节点预览">
    <p className="text-sm leading-6 text-slate-600">预览主节点，也可选择其他节点一起对比。只计算所选分支及其依赖，无需写完公式或连接最终市场状态。</p>
    <div className="grid gap-3 sm:grid-cols-2">
      <label className="text-xs font-semibold text-slate-600">预览节点<select aria-label="待预览节点" value={nodeId} onChange={event => { setNodeId(event.target.value); setPort('') }} className="mt-1 min-h-10 w-full rounded-xl border border-slate-300 bg-white px-2">{definition.graph.nodes.map(item => <option key={item.id} value={item.id}>{item.label || schemas.find(s => [s.id, s.type, s.type_id].includes(item.type))?.label || item.id}</option>)}</select></label>
      <label className="text-xs font-semibold text-slate-600">输出数据<select aria-label="预览输出端口" value={outputPort} onChange={event => setPort(event.target.value)} className="mt-1 min-h-10 w-full rounded-xl border border-slate-300 bg-white px-2">{schema?.outputs.map(item => <option key={item.id} value={item.id}>{regimePortLabel(item)}</option>)}</select></label>
      <label className="text-xs font-semibold text-slate-600">分析方式<select aria-label="节点预览分析方式" value={mode} onChange={event => setMode(event.target.value as RegimeMode)} className="mt-1 min-h-10 w-full rounded-xl border border-slate-300 bg-white px-2"><option value="realtime">实时分析</option><option value="retrospective">事后研究</option></select></label>
      <label className="text-xs font-semibold text-slate-600">截至日<input aria-label="节点预览截至日" type="date" value={asOf} onChange={event => setAsOf(event.target.value)} className="mt-1 min-h-10 w-full rounded-xl border border-slate-300 bg-white px-2" /></label>
    </div>
    <RegimeComparisonPicker detailsRef={comparisonPickerRef} options={comparisonChoices} selected={comparisonTargets} onChange={setComparisons} disabled={busy} />
    {comparisonError && <p role="alert" className="text-sm text-amber-800">{comparisonError} 请移除该对比项或调整分析方式。</p>}
    <div className="flex items-center gap-3"><button type="button" disabled={!node || !outputPort || busy || !!comparisonError} onClick={() => void preview()} className="min-h-10 rounded-lg bg-accent-600 px-4 text-sm font-semibold text-white disabled:opacity-40">{busy ? '正在预览…' : '预览节点数据'}</button>{busy && <button type="button" onClick={() => { controllerRef.current?.abort(); if (jobRef.current) void cancelRegimePreviewRun(jobRef.current).catch(() => undefined); jobRef.current = ''; setBusy(false); setRun(previous => previous ? { ...previous, status: 'cancelled' } : previous) }} className="min-h-10 px-2 text-sm text-slate-600">取消预览</button>}</div>
    {error && <p role="alert" className="text-sm text-rose-700">{error}</p>}
    {!stale && !busy && onChange && alignmentSchema && alignmentNodes.length > 0 && <div className="space-y-2 rounded-xl border border-amber-200 bg-amber-50 p-3 text-sm text-slate-700">
      <p>将为{alignmentNodes.map(item => `“${item.label || schemas.find(schema => schema.id === item.type)?.label || item.id}”`).join('、')}添加日期对齐节点，只使用两边都有记录的日期。保留缺失值，不自动补值。</p>
      <button type="button" onClick={alignAndPreview} className="min-h-10 rounded-lg bg-accent-600 px-3 font-semibold text-white">按共同日期对齐并预览</button>
    </div>}
    {notice && <p className="text-sm text-slate-600">{notice}</p>}
    {stale && <p role="status" className="text-sm text-amber-800">配置已变化，下方为上次预览结果，请重新预览。</p>}
    <RegimeResultDock onSelectComparisons={openComparisons} nodeOnly run={run} page={page} nodes={[]} schemas={schemas} previewNodeId={nodeId} loadingSeries={busy} onPreviewNode={() => undefined} onLoadSeries={() => undefined} />
  </section>
}

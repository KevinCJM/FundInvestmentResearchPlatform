import RegimeTemporalPanel from './regime-workbench/RegimeTemporalPanel'
import ResourceLibrary from './regime-workbench/RegimeResourceLibrary'
import useRegimeCompositeExpansion from './regime-workbench/useRegimeCompositeExpansion'
import RegimeDefinitionLibrary from './regime-workbench/RegimeDefinitionLibrary'
import RegimeGuidedFormulaPanel from './regime-workbench/RegimeGuidedFormulaPanel'
import RegimeMathPreview from './regime-workbench/RegimeMathPreview'
import HistoricalRegimeDirectory from './HistoricalRegimeDirectory'
import RegimeSeriesOutputs from './regime-workbench/RegimeSeriesOutputs'
import { useEffect, useMemo, useReducer, useRef, useState } from 'react'
import RegimeResultView from './regime-workbench/RegimeResultView'
import RegimeGuidedEditor from './regime-workbench/RegimeGuidedEditor'
import RegimeFormulaEditor from './regime-workbench/RegimeFormulaEditor'
import { regimeConnectionIssue } from './regime-workbench/regimeGraphEditing'
import RegimeWorkbenchDrawer from './regime-workbench/RegimeWorkbenchDrawer'
import { regimeCalculationFingerprint, regimePresentationFingerprint } from './regime-workbench/regimeDraftIdentity'
import ResearchDataLab from './ResearchDataLab'
import RegimeGraphCanvas, {
  type RegimeCanvasEdge,
  type RegimeCanvasNodeChange,
} from './regime-workbench/RegimeGraphCanvas'
import RegimeNodeInspector from './regime-workbench/RegimeNodeInspector'
import RegimeResultDock from './regime-workbench/RegimeResultDock'
import RegimeNodePreviewPanel from './regime-workbench/RegimeNodePreviewPanel'
import RegimeLifecyclePanel from './regime-workbench/RegimeLifecyclePanel'
import RegimeSavePanel from './regime-workbench/RegimeSavePanel'
import RegimeValidationPanel from './regime-workbench/RegimeValidationPanel'
import RegimeGraphAssetsPanel from './regime-workbench/RegimeGraphAssetsPanel'
import RegimeExperimentPanel from './regime-workbench/RegimeExperimentPanel'
import {
  cancelRegimePreviewRun,
  cloneRegimeGraphDefinition,
  createRegimeGraphDefinition,
  createBlankRegimeDefinition,
  definitionForRequest,
  getRegimeGraphDefinition,
  getRegimeGraphTemplates,
  getRegimeNodeCatalog,
  getRegimePreviewRun,
  getRegimePreviewSeries,
  inferRegimeGraph,
  instantiateRegimeTemplate,
  listRegimeGraphDefinitions,
  prepareRegimeGraph,
  RegimeGraphApiError,
  startRegimePreviewRun,
  updateRegimeGraphDefinition,
  type RegimeGraphConnection,
  type RegimeGraphDefinition,
  type RegimeGraphInference,
  type RegimeGraphNode,
  type RegimeGraphTemplate,
  type RegimeMode,
  type RegimeNodeSchema,
  type PreparedRegimeGraph,
  type RegimePreviewRun,
  type RegimeSeriesPage,
} from '../services/regimeGraph'
import RegimeHelpTip from './regime-workbench/RegimeHelpTip'
import type { ResearchSeriesCatalogItem } from '../services/researchSeries'

import { createTimelineReducer, sameDocument as sameDefinition } from '../components/computation-graph/history'

const timelineReducer = createTimelineReducer<RegimeGraphDefinition>(cloneRegimeGraphDefinition)

function schemaId(schema: RegimeNodeSchema) {
  return schema.id || schema.type || ''
}

function schemaParameters(schema: RegimeNodeSchema) {
  return schema.parameter_schema?.properties ?? schema.parameters ?? {}
}

function defaultParameters(schema: RegimeNodeSchema) {
  return Object.fromEntries(Object.entries(schemaParameters(schema)).flatMap(([name, parameter]) => parameter.default === undefined ? [] : [[name, parameter.default]]))
}

function edgesFrom(nodes: RegimeGraphNode[]): RegimeCanvasEdge[] {
  return nodes.flatMap((node) => Object.entries(node.inputs || {}).map(([targetPort, input]) => ({
    id: `${input.node_id}:${input.port}->${node.id}:${targetPort}`,
    source: input.node_id,
    sourcePort: input.port,
    target: node.id,
    targetPort,
  })))
}

function errorText(reason: unknown, fallback: string) {
  if (reason instanceof DOMException && reason.name === 'AbortError') return ''
  return reason instanceof Error ? reason.message : fallback
}

function newNodeId(type: string, sequence: number) {
  const safeType = type.replace(/[^a-zA-Z0-9_-]/g, '-').slice(0, 28) || 'node'
  return `${safeType}-${sequence}`
}

function issueCount(inference: RegimeGraphInference | null) {
  return (inference?.errors.length ?? 0) + (inference?.warnings.length ?? 0)
}

function VersionBar({
  definitions,
  selectedId,
  selectedRevision,
  current,
  dirty,
  valid,
  busy,
  onSelect,
  onRevision,
  onLoad,
  onSave,
  onSaveAs,
}: {
  definitions: RegimeGraphDefinition[]
  selectedId: string
  selectedRevision: number
  current: RegimeGraphDefinition
  dirty: boolean
  valid: boolean
  busy: boolean
  onSelect: (id: string) => void
  onRevision: (revision: number) => void
  onLoad: () => void
  onSave: () => void
  onSaveAs: () => void
}) {
  return <section className="rounded-2xl border border-slate-200 bg-white p-3 shadow-sm" aria-label="历史情景版本管理">
    <div className="grid gap-3 lg:grid-cols-[minmax(220px,1fr)_110px_auto_auto_auto] lg:items-end">
      <label className="text-[11px] font-bold text-slate-600">已保存定义<RegimeHelpTip label="已保存定义说明" text="选择服务器中已保存的历史情景定义。载入不会覆盖当前草稿，确认后才切换版本。" /><select aria-label="已保存历史情景定义" value={selectedId} onChange={(event) => onSelect(event.target.value)} className="mt-1 min-h-10 w-full rounded-lg border border-slate-300 bg-white px-2 font-normal"><option value="">选择已保存定义</option>{definitions.map((item) => <option key={item.id} value={item.id}>{item.name} · 第 {item.revision} 版</option>)}</select></label>
      <label className="text-[11px] font-bold text-slate-600">精确修订<RegimeHelpTip label="精确修订说明" text="指定要载入的不可变历史版本。修改已保存图谱会产生新修订，不会改写旧结果。" /><input aria-label="历史情景修订号" type="number" min={1} value={selectedRevision || 1} onChange={(event) => onRevision(Math.max(1, Math.trunc(Number(event.target.value) || 1)))} className="mt-1 min-h-10 w-full rounded-lg border border-slate-300 px-2 font-normal" /></label>
      <button type="button" disabled={!selectedId || busy} onClick={onLoad} className="min-h-10 rounded-lg border border-slate-300 px-3 text-xs font-bold text-slate-700 disabled:opacity-40">载入版本</button>
      <button type="button" disabled={!valid || !current.id || !current.revision || !dirty || busy} onClick={onSave} className="min-h-10 rounded-lg bg-slate-950 px-3 text-xs font-bold text-white disabled:opacity-40">保存修订</button>
      <button type="button" disabled={!valid || busy} onClick={onSaveAs} className="min-h-10 rounded-lg border border-indigo-300 px-3 text-xs font-bold text-indigo-700 disabled:opacity-40">另存为新定义</button>
    </div>
    <div className="mt-2 flex flex-wrap items-center gap-2 text-[10px]"><span className={`rounded-full px-2 py-1 font-bold ${current.id ? 'bg-emerald-100 text-emerald-800' : 'bg-slate-100 text-slate-600'}`}>{current.id ? `已保存 · ${current.id} · r${current.revision}` : '未保存草稿'}</span><span className={`rounded-full px-2 py-1 font-bold ${dirty ? 'bg-amber-100 text-amber-900' : 'bg-emerald-100 text-emerald-800'}`}>{dirty ? '存在未保存改动' : '与服务端版本一致'}</span><span className="text-slate-500">保存只固化并校验图谱；正式运行前仍需在实验面板显式预热当前版本。</span></div>
  </section>
}

function realtimeNodeBlocked(schema: RegimeNodeSchema | undefined) {
  if (schema?.temporal_contract) return ['manual_hindsight', 'full_input', 'future_confirmation', 'unknown'].includes(schema.temporal_contract.rule)
  // Compatibility for older catalogs; actual graph decisions come from infer.
  return schema?.supports_realtime !== true || schema.causal !== true || schema.repaints !== false
}

const REALTIME_RESTRICTION = '实时识别已禁用事后分析算法；请移除相关节点，或切换到事后研究。'

function InferenceIssues({ inference, status, onNode }: { inference: RegimeGraphInference | null; status: string; onNode: (id: string) => void }) {
  if (status === 'checking') return <div role="status" className="rounded-xl border border-indigo-200 bg-indigo-50 px-3 py-2 text-xs font-semibold text-indigo-900">正在解析类型、连线、因果性与执行策略…</div>
  if (!inference) return null
  const issues = [...inference.errors, ...inference.warnings]
  if (!issues.length) return <div className="rounded-xl border border-emerald-200 bg-emerald-50 px-3 py-2 text-xs font-semibold text-emerald-900">计算图检查通过，可以预热并试算。</div>
  return <section aria-label="图谱检查问题" className="rounded-xl border border-amber-200 bg-amber-50 p-3"><h3 className="text-xs font-bold text-amber-950">图谱检查 · {inference.errors.length} 个错误 / {inference.warnings.length} 个提示</h3><ul className="mt-2 space-y-1">{issues.slice(0, 8).map((issue, index) => <li key={`${issue.code}-${issue.node_id}-${index}`} className="flex items-start justify-between gap-2 text-[11px] text-amber-950"><span>{issue.message}</span>{issue.node_id ? <button type="button" onClick={() => onNode(issue.node_id!)} className="shrink-0 font-bold underline">定位节点</button> : null}</li>)}</ul></section>
}

export interface HistoricalRegimeWorkbenchProps {
  onExit?: () => void
  initialDefinition?: RegimeGraphDefinition
}

export default function HistoricalRegimeWorkbench({ onExit, initialDefinition }: HistoricalRegimeWorkbenchProps) {
  const [timeline, dispatch] = useReducer(timelineReducer, { past: [], present: initialDefinition ? cloneRegimeGraphDefinition(initialDefinition) : createBlankRegimeDefinition(), future: [] })
  const definition = timeline.present
  const [schemas, setSchemas] = useState<RegimeNodeSchema[]>([])
  const [templates, setTemplates] = useState<RegimeGraphTemplate[]>([])
  const [savedDefinitions, setSavedDefinitions] = useState<RegimeGraphDefinition[]>([])
  const [selectedDefinitionId, setSelectedDefinitionId] = useState(initialDefinition?.id || '')
  const [selectedDefinitionRevision, setSelectedDefinitionRevision] = useState(initialDefinition?.revision || 1)
  const [savedDefinitionSignature, setSavedDefinitionSignature] = useState(initialDefinition?.id && initialDefinition.revision ? JSON.stringify(definitionForRequest(initialDefinition)) : '')
  const [savingDefinition, setSavingDefinition] = useState(false)
  const [loadingDefinition, setLoadingDefinition] = useState(false)
  const [selectedTemplate, setSelectedTemplate] = useState('')
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null)
  const [selectedGraphNodeIds, setSelectedGraphNodeIds] = useState<string[]>(initialDefinition?.graph.nodes[0]?.id ? [initialDefinition.graph.nodes[0].id] : [])
  const [inference, setInference] = useState<RegimeGraphInference | null>(null)
  const [inferenceStatus, setInferenceStatus] = useState<'idle' | 'checking' | 'ready' | 'invalid' | 'error'>('idle')
  const [mode, setMode] = useState<RegimeMode>('realtime')
  const [asOf, setAsOf] = useState('')
  const [run, setRun] = useState<RegimePreviewRun | null>(null)
  const [preparedPlan, setPreparedPlan] = useState<PreparedRegimeGraph | null>(null)
  const [runDefinitionSignature, setRunDefinitionSignature] = useState('')
  const [seriesPage, setSeriesPage] = useState<RegimeSeriesPage | null>(null)
  const [previewNodeId, setPreviewNodeId] = useState('')
  const [loadingSeries, setLoadingSeries] = useState(false)
  const [view, setView] = useState<'build' | 'result' | 'experiments'>('build')
  const [viewedRunId, setViewedRunId] = useState('')
  const [displayedResult, setDisplayedResult] = useState<{ id: string; kind: 'preview' | 'formal' } | null>(null)
  const [editorMode, setEditorMode] = useState<'guided' | 'canvas' | 'formula'>('canvas')
  const [formulaPending, setFormulaPending] = useState(false)
  const [formulaBusy, setFormulaBusy] = useState(false)
  const [savingResearch, setSavingResearch] = useState(false)
  const [drawer, setDrawer] = useState<'library' | 'inspector' | 'states' | 'versions' | 'issues' | 'assets' | 'builder' | 'legacy' | 'node-preview' | null>(null)
  const [focused, setFocused] = useState(false)
  const [libraryCollapsed, setLibraryCollapsed] = useState(false)
  const [mobileLibrary, setMobileLibrary] = useState(false)
  const [basicInfoOpen, setBasicInfoOpen] = useState(false)
  const [activeOutputId, setActiveOutputId] = useState('state')
  const definitionLoadRequest = useRef(0)
  const [nodePreviewOpen, setNodePreviewOpen] = useState(false)
  const [runDefinition, setRunDefinition] = useState<RegimeGraphDefinition | null>(null)
  const [dataLabOpen, setDataLabOpen] = useState(false)
  const [loadingTemplate, setLoadingTemplate] = useState(false)
  const [loadingCatalog, setLoadingCatalog] = useState(true)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const nodeSequence = useRef(1)
  const inferenceRequest = useRef(0)
  const runRequest = useRef(0)
  const runAbort = useRef<AbortController | null>(null)
  const seriesAbort = useRef<AbortController | null>(null)
  const seriesRequest = useRef(0)
  const keepEditing = useRef(false)
  const latestApiSignature = useRef('')
  const latestRunContext = useRef('')

  const routeSearch = typeof window === 'undefined' ? '' : window.location.search
  const routeQuery = useMemo(() => new URLSearchParams(routeSearch), [routeSearch])
  const routeDefinitionId = routeQuery.get('definition')?.trim() || ''
  const routeRevisionRaw = routeQuery.get('revision')?.trim() || ''
  const routeTemplateId = routeQuery.get('template')?.trim() || ''
  const routeLoadKey = routeDefinitionId
    ? `definition:${routeDefinitionId}:revision:${routeRevisionRaw}`
    : routeTemplateId ? `template:${routeTemplateId}` : ''

  const apiSignature = useMemo(() => JSON.stringify(definitionForRequest(definition)), [definition])
  latestApiSignature.current = apiSignature
  latestRunContext.current = `${apiSignature}:${mode}:${asOf}`
  const calculationSignature = useMemo(() => regimeCalculationFingerprint(definition, mode, asOf), [definition, mode, asOf])
  const presentationChanged = Boolean(runDefinition && regimePresentationFingerprint(runDefinition) !== regimePresentationFingerprint(definition))
  const evaluationChanged = Boolean(runDefinition && JSON.stringify(runDefinition.evaluation_targets) !== JSON.stringify(definition.evaluation_targets))
  const viewingFormal = view === 'result' && displayedResult?.kind === 'formal'
  const running = Boolean(run && ['queued', 'preparing', 'running'].includes(run.status))
  const valid = Boolean(inference?.valid && inferenceStatus === 'ready' && !formulaPending && !formulaBusy)
  const temporal = inference?.temporal_capability
  const realtimeBlockedNodes = mode === 'realtime' && temporal && !temporal.realtime_supported ? definition.graph.nodes.filter(node => temporal.reasons.some(r => r.node_id === node.id)) : []
  const canRun = valid && !loadingCatalog && (mode !== 'realtime' || !temporal || temporal.realtime_supported)
  const manualEventNode = definition.graph.nodes.find(node => node.type === 'annotation.manual_events')
  const manualEventMode = Boolean(manualEventNode)
  const manualEventCount = Array.isArray(manualEventNode?.parameters.events) ? manualEventNode.parameters.events.length : 0
  const selectableSchemas = mode === 'realtime' ? schemas.map((schema) => realtimeNodeBlocked(schema) ? { ...schema, available: false, unavailable_reason: REALTIME_RESTRICTION } : schema) : schemas

  const edges = useMemo(() => edgesFrom(definition.graph.nodes), [definition.graph.nodes])
  const selectedNode = definition.graph.nodes.find((node) => node.id === selectedNodeId) ?? null
  const selectedSchema = selectedNode ? schemas.find((schema) => schemaId(schema) === selectedNode.type || schema.type === selectedNode.type) : undefined
  const runStale = Boolean(run && runDefinitionSignature && runDefinitionSignature !== calculationSignature)
  const definitionDirty = !savedDefinitionSignature || savedDefinitionSignature !== apiSignature
  const boundSeriesIds = definition.graph.nodes.flatMap((node) => Object.entries(node.parameters).flatMap(([key, value]) => typeof value === 'string' && (key === 'artifact_id' || /^(index|macro|indicator|upload):/.test(value)) ? [value] : []))

  useEffect(() => { setPreparedPlan(null) }, [apiSignature, mode, asOf])

  useEffect(() => {
    const controller = new AbortController()
    setLoadingCatalog(true); setError('')
    void Promise.all([getRegimeNodeCatalog(controller.signal), getRegimeGraphTemplates(controller.signal)])
      .then(([nextSchemas, nextTemplates]) => { setSchemas(nextSchemas); setTemplates(nextTemplates) })
      .catch((reason) => { if (!controller.signal.aborted) setError(errorText(reason, '节点目录或模板加载失败。')) })
      .finally(() => { if (!controller.signal.aborted) setLoadingCatalog(false) })
    return () => controller.abort()
  }, [])


  useEffect(() => {
    const controller = new AbortController()
    void listRegimeGraphDefinitions(controller.signal)
      .then(setSavedDefinitions)
      .catch((reason) => { if (!controller.signal.aborted) setError(errorText(reason, '已保存定义加载失败。')) })
    return () => controller.abort()
  }, [])

  useEffect(() => {
    if (!routeLoadKey || loadingCatalog) return
    const controller = new AbortController()
    setLoadingDefinition(true); setError(''); setNotice('')

    if (routeDefinitionId) {
      const revision = Number(routeRevisionRaw)
      if (!Number.isInteger(revision) || revision < 1) {
        setError('通过目录打开定义时必须提供有效的精确 revision。')
        setLoadingDefinition(false)
        return () => controller.abort()
      }
      void getRegimeGraphDefinition(routeDefinitionId, revision, controller.signal)
        .then((next) => {
          if (controller.signal.aborted) return
          dispatch({ type: 'reset', definition: next })
          setMode((next.default_mode || templates.find(item => item.id === next.template_id)?.default_mode) === 'retrospective' || next.graph.nodes.some(node => realtimeNodeBlocked(schemas.find(schema => schemaId(schema) === node.type))) ? 'retrospective' : 'realtime')
          setSelectedDefinitionId(next.id || routeDefinitionId)
          setSelectedDefinitionRevision(next.revision || revision)
          setSelectedTemplate('')
          setSelectedNodeId(next.graph.nodes[0]?.id ?? null)
          setSelectedGraphNodeIds(next.graph.nodes[0]?.id ? [next.graph.nodes[0].id] : [])
          setSavedDefinitionSignature(JSON.stringify(definitionForRequest(next)))
          clearRunDisplay()
          nodeSequence.current = next.graph.nodes.length + 1
          setNotice(`已从目录载入 ${next.name} · r${next.revision || revision}。`)
        })
        .catch((reason) => { if (!controller.signal.aborted) setError(errorText(reason, '精确定义版本载入失败。')) })
        .finally(() => { if (!controller.signal.aborted) setLoadingDefinition(false) })
      return () => controller.abort()
    }

    void instantiateRegimeTemplate(routeTemplateId, controller.signal)
      .then((next) => {
        if (controller.signal.aborted) return
        const draft = { ...next, id: undefined, revision: undefined, template_id: next.template_id || routeTemplateId }
        dispatch({ type: 'reset', definition: draft })
        setMode(routeQuery.get('mode') === 'retrospective' || (routeQuery.get('mode') !== 'realtime' && templates.find(item => item.id === routeTemplateId)?.default_mode === 'retrospective') || draft.graph.nodes.some(node => realtimeNodeBlocked(schemas.find(schema => schemaId(schema) === node.type))) ? 'retrospective' : 'realtime')
        setSelectedDefinitionId(''); setSelectedDefinitionRevision(1)
        setSelectedTemplate(routeTemplateId)
        setSelectedNodeId(draft.graph.nodes[0]?.id ?? null)
        setSelectedGraphNodeIds(draft.graph.nodes[0]?.id ? [draft.graph.nodes[0].id] : [])
        setSavedDefinitionSignature('')
        clearRunDisplay()
        nodeSequence.current = draft.graph.nodes.length + 1
        setNotice('已从目录实例化模板为独立草稿；保存前不会覆盖模板。')
      })
      .catch((reason) => { if (!controller.signal.aborted) setError(errorText(reason, '目录模板实例化失败。')) })
      .finally(() => { if (!controller.signal.aborted) setLoadingDefinition(false) })
    return () => controller.abort()
  }, [routeDefinitionId, routeLoadKey, routeRevisionRaw, routeTemplateId, loadingCatalog])

  useEffect(() => {
    const requestId = ++inferenceRequest.current
    const controller = new AbortController()
    setInferenceStatus('checking')
    const timer = window.setTimeout(() => {
      void inferRegimeGraph(definition, controller.signal, mode)
        .then((response) => {
          if (requestId !== inferenceRequest.current) return
          setInference(response)
          setInferenceStatus(response.valid ? 'ready' : 'invalid')
        })
        .catch((reason) => {
          if (controller.signal.aborted || requestId !== inferenceRequest.current) return
          setInference(null); setInferenceStatus('error'); setError(errorText(reason, '图谱检查失败。'))
        })
    }, 300)
    return () => { window.clearTimeout(timer); controller.abort() }
  }, [apiSignature, mode])

  useEffect(() => () => {
    runAbort.current?.abort(); runRequest.current += 1
    seriesAbort.current?.abort(); seriesRequest.current += 1
  }, [])

  useEffect(() => {
    seriesAbort.current?.abort(); seriesRequest.current += 1
    setSeriesPage(null); setLoadingSeries(false)
  }, [run?.id, previewNodeId])

  useEffect(() => { if (view === 'result' && displayedResult) setViewedRunId(`${displayedResult.kind}:${displayedResult.id}`) }, [view, displayedResult])

  const changeEditorMode = (next: typeof editorMode) => {
    if (formulaPending || formulaBusy) { setError('请先应用或还原公式草稿，再切换编辑方式。'); return }
    setEditorMode(next); setDrawer(null)
  }

  const selectNode = (id: string) => {
    if (formulaPending || formulaBusy) { setError('请先应用或还原公式草稿。'); return }
    setSelectedNodeId(id); setEditorMode('canvas'); setDrawer('inspector')
  }

  const chooseLibraryItem = (load: () => void) => {
    if (formulaPending || formulaBusy) { setError('请先应用或还原公式草稿。'); return }
    if (timeline.past.length && definitionDirty && !window.confirm('当前有未保存的修改。切换算法将替换草稿，是否继续？')) return
    load()
  }

  const openBuilder = (id?: string) => {
    setSelectedNodeId(id || definition.graph.outputs[activeOutputId]?.node_id || definition.graph.nodes[0]?.id || null)
    setDrawer('builder')
  }

  const changeView = (next: typeof view) => {
    if (next === 'build' && running) keepEditing.current = true
    if (next === 'result' && !displayedResult && run?.status === 'completed') setDisplayedResult({ id: run.id, kind: 'preview' })
    setView(next); setDrawer(null); setMobileLibrary(false)
  }

  const clearRunDisplay = () => {
    runRequest.current += 1; runAbort.current?.abort()
    seriesRequest.current += 1; seriesAbort.current?.abort()
    setRun(null); setRunDefinition(null); setSeriesPage(null); setPreparedPlan(null)
    setNodePreviewOpen(false); setLoadingSeries(false); setView('build'); setDisplayedResult(null); setViewedRunId('')
  }

  const editDefinition = (next: RegimeGraphDefinition) => {
    if (formulaPending) { setError('请先应用或还原公式草稿。'); return }
    if (running) keepEditing.current = true
    dispatch({ type: 'edit', definition: cloneRegimeGraphDefinition(next) }); setNotice('')
    // Preparation remains bound to the server definition hash; positions are UI only.
    if (JSON.stringify(definitionForRequest(next)) !== apiSignature) setPreparedPlan(null)
  }

  const { expand, expanding } = useRegimeCompositeExpansion({
    definition, mode, blocked: formulaPending || formulaBusy, onChange: editDefinition,
    onError: setError, onNotice: setNotice,
    onSelected: id => { setSelectedNodeId(id); setSelectedGraphNodeIds([id]); setDrawer(editorMode === 'guided' ? 'builder' : 'inspector') },
  })

  const addNode = (schema: RegimeNodeSchema, parameterOverrides: Record<string, unknown> = {}, label?: string) => {
    if (expanding || formulaPending || formulaBusy) return
    if (mode === 'realtime' && realtimeNodeBlocked(schema)) { setError(REALTIME_RESTRICTION); return }
    if (schema.available === false || (schema.status && schema.status !== 'available')) { setError(schema.unavailable_reason || `节点 ${schema.label} 当前不可用。`); return }
    const type = schemaId(schema)
    if (!type) { setError('节点目录返回了缺少 id/type 的节点，无法添加。'); return }
    const id = newNodeId(type, nodeSequence.current++)
    const previous = selectedNode
    const firstInput = schema.inputs?.[0]
    const previousSchema = previous ? schemas.find((item) => schemaId(item) === previous.type || item.type === previous.type) : undefined
    const firstOutput = previousSchema?.outputs?.[0]
    const inputs = firstInput && previous && firstOutput && !regimeConnectionIssue([...definition.graph.nodes, { id, type, parameters: {}, inputs: {} }], schemas, { source: previous.id, sourcePort: firstOutput.id, target: id, targetPort: firstInput.id }) ? { [firstInput.id]: { node_id: previous.id, port: firstOutput.id } } : {}
    const typeVersion = schema.type_version ?? schema.version
    const node: RegimeGraphNode = { id, type, ...(typeVersion != null ? { type_version: typeVersion } : {}), label, parameters: { ...defaultParameters(schema), ...parameterOverrides }, inputs, position: { x: definition.graph.nodes.length * 240, y: 0 } }
    const next = { ...definition, graph: { ...definition.graph, nodes: [...definition.graph.nodes, node], exposed_node_ids: [...(definition.graph.exposed_node_ids || []), node.id] } }
    if (schema.granularity?.expandable) { void expand(next, id); return }
    editDefinition(next)
    if (editorMode === 'guided') { setSelectedNodeId(id); setDrawer('builder') } else selectNode(id)
  }

  const handleCanvasChanges = (changes: RegimeCanvasNodeChange[]) => {
    let next = cloneRegimeGraphDefinition(definition)
    changes.forEach((change) => {
      if (change.type === 'select') selectNode(change.id)
      if (change.type === 'position') next.graph.nodes = next.graph.nodes.map((node) => node.id === change.id ? { ...node, position: change.position } : node)
      if (change.type === 'remove') {
        next.graph.exposed_node_ids = next.graph.exposed_node_ids?.filter(id => id !== change.id)
        next.graph.nodes = next.graph.nodes.filter((node) => node.id !== change.id).map((node) => ({ ...node, inputs: Object.fromEntries(Object.entries(node.inputs).filter(([, input]) => input.node_id !== change.id)) }))
        next.graph.outputs = Object.fromEntries(Object.entries(next.graph.outputs).filter(([, output]) => output?.node_id !== change.id))
        if (next.graph.channel_metadata) next.graph.channel_metadata = Object.fromEntries(Object.entries(next.graph.channel_metadata).filter(([id]) => next.graph.outputs[id]))
        if (selectedNodeId === change.id) setSelectedNodeId(null)
      }
    })
    if (!sameDefinition(next, definition)) editDefinition(next)
  }

  const patchSelectedNode = (patch: Partial<RegimeGraphNode>) => {
    if (!selectedNode) return
    editDefinition({ ...definition, graph: { ...definition.graph, nodes: definition.graph.nodes.map((node) => node.id === selectedNode.id ? { ...node, ...patch } : node) } })
  }

  const connectSelectedNode = (port: string, connection: RegimeGraphConnection | null) => {
    if (!selectedNode) return
    const inputs = { ...selectedNode.inputs }
    if (connection) inputs[port] = connection
    else delete inputs[port]
    patchSelectedNode({ inputs })
  }

  const connectCanvasNodes = (connection: { source: string; sourcePort: string; target: string; targetPort: string }) => {
    editDefinition({
      ...definition,
      graph: {
        ...definition.graph,
        nodes: definition.graph.nodes.map((node) => node.id === connection.target ? {
          ...node,
          inputs: { ...node.inputs, [connection.targetPort]: { node_id: connection.source, port: connection.sourcePort } },
        } : node),
      },
    })
    selectNode(connection.target)
  }

  const duplicateNodes = (nodeIds: string[]) => {
    const selected = definition.graph.nodes.filter((node) => nodeIds.includes(node.id))
    if (!selected.length) return
    const occupied = new Set(definition.graph.nodes.map((node) => node.id))
    const idMap = new Map<string, string>()
    selected.forEach((node) => {
      let nextId = newNodeId(node.type, nodeSequence.current++)
      while (occupied.has(nextId)) nextId = newNodeId(node.type, nodeSequence.current++)
      occupied.add(nextId)
      idMap.set(node.id, nextId)
    })
    const copies = selected.map((node) => ({
      ...node,
      id: idMap.get(node.id)!,
      label: `${node.label || schemas.find((schema) => schemaId(schema) === node.type)?.label || node.type} 副本`,
      parameters: JSON.parse(JSON.stringify(node.parameters)) as Record<string, unknown>,
      inputs: Object.fromEntries(Object.entries(node.inputs).map(([port, input]) => [port, { ...input, node_id: idMap.get(input.node_id) || input.node_id }])),
      position: { x: (node.position?.x ?? 0) + 44, y: (node.position?.y ?? 0) + 44 },
    }))
    editDefinition({ ...definition, graph: { ...definition.graph, nodes: [...definition.graph.nodes, ...copies] } })
    setSelectedNodeId(copies[0].id)
    setNotice(`已复制 ${copies.length} 个节点；所选节点之间的内部连线已同步复制。`)
  }

  const setSelectedOutput = (slot: 'state' | 'probabilities' | 'confidence' | 'recognition_index' | 'effective_index' | 'reason_code', port: string | null) => {
    if (!selectedNode) return
    const outputs = { ...definition.graph.outputs }
    if (port) outputs[slot] = { node_id: selectedNode.id, port }
    else delete outputs[slot]
    const channel_metadata = Object.fromEntries(Object.entries(definition.graph.channel_metadata || {}).filter(([id]) => outputs[id]))
    editDefinition({ ...definition, graph: { ...definition.graph, outputs, channel_metadata } })
  }

  const loadTemplate = async (templateId = selectedTemplate) => {
    if (!templateId) return
    const request = ++definitionLoadRequest.current
    const startingSignature = latestApiSignature.current
    setLoadingTemplate(true); setError(''); setNotice('')
    try {
      const next = await instantiateRegimeTemplate(templateId)
      if (request !== definitionLoadRequest.current) return
      if (startingSignature !== latestApiSignature.current) { setNotice('当前草稿已修改，未覆盖你的编辑。'); return }
      setSelectedTemplate(templateId); setSelectedDefinitionId(''); setSelectedDefinitionRevision(1); setMobileLibrary(false)
      setMode(templates.find(item => item.id === templateId)?.default_mode === 'retrospective' || next.graph.nodes.some(node => realtimeNodeBlocked(schemas.find(schema => schemaId(schema) === node.type))) ? 'retrospective' : 'realtime')
      dispatch({ type: 'reset', definition: next }); setSelectedNodeId(next.graph.nodes[0]?.id ?? null); clearRunDisplay(); setSavedDefinitionSignature('')
      setSelectedGraphNodeIds(next.graph.nodes[0]?.id ? [next.graph.nodes[0].id] : [])
      setDrawer(null)
      setNotice('模板已载入。选择研究数据、调整识别规则后即可运行。')
    } catch (reason) { if (request === definitionLoadRequest.current) setError(errorText(reason, '模板载入失败。')) } finally { if (request === definitionLoadRequest.current) setLoadingTemplate(false) }
  }

  const createBlank = () => {
    if ((definition.graph.nodes.length || timeline.past.length) && !window.confirm('当前草稿将被新的空白计算图替换，是否继续？')) return
    definitionLoadRequest.current += 1
    setSelectedDefinitionId(''); setSelectedDefinitionRevision(1); setMobileLibrary(false); setActiveOutputId('state'); setMode('realtime')
    const next = createBlankRegimeDefinition()
    dispatch({ type: 'reset', definition: next }); setSelectedNodeId(null); setSelectedGraphNodeIds([]); setSelectedTemplate(''); clearRunDisplay(); setSavedDefinitionSignature(''); setError(''); setNotice('已创建空白计算图。')
  }

  const selectSavedDefinition = (id: string) => {
    setSelectedDefinitionId(id)
    const selected = savedDefinitions.find((item) => item.id === id)
    setSelectedDefinitionRevision(selected?.revision || 1)
  }

  const loadSavedDefinition = async (id = selectedDefinitionId, revision = selectedDefinitionRevision) => {
    if (!id) return
    const request = ++definitionLoadRequest.current
    const startingSignature = latestApiSignature.current
    setLoadingDefinition(true); setError(''); setNotice('')
    try {
      const next = await getRegimeGraphDefinition(id, revision)
      if (request !== definitionLoadRequest.current) return
      if (startingSignature !== latestApiSignature.current) { setNotice('当前草稿已修改，未覆盖你的编辑。'); return }
      setSelectedDefinitionId(id); setSelectedDefinitionRevision(next.revision || revision); setSelectedTemplate(''); setMobileLibrary(false)
      setMode((next.default_mode || templates.find(item => item.id === next.template_id)?.default_mode) === 'retrospective' || next.graph.nodes.some(node => realtimeNodeBlocked(schemas.find(schema => schemaId(schema) === node.type))) ? 'retrospective' : 'realtime')
      dispatch({ type: 'reset', definition: next })
      setSelectedNodeId(next.graph.nodes[0]?.id ?? null)
      setSelectedGraphNodeIds(next.graph.nodes[0]?.id ? [next.graph.nodes[0].id] : [])
      setSavedDefinitionSignature(JSON.stringify(definitionForRequest(next)))
      clearRunDisplay()
      setNotice(`已载入 ${next.name} · r${next.revision}。`)
    } catch (reason) { if (request === definitionLoadRequest.current) setError(errorText(reason, '定义版本载入失败。')) } finally { if (request === definitionLoadRequest.current) setLoadingDefinition(false) }
  }

  const saveDefinition = async (saveAsNew: boolean) => {
    if (formulaPending || formulaBusy) { setError('请先应用或还原公式草稿再保存。'); return }
    if (!inference?.valid || inferenceStatus !== 'ready') { setError('请先修复图谱检查错误再保存。'); return }
    setSavingDefinition(true); setError(''); setNotice('')
    try {
      const saved = saveAsNew || !definition.id || !definition.revision
        ? await createRegimeGraphDefinition(definition)
        : await updateRegimeGraphDefinition(definition)
      dispatch({ type: 'reset', definition: saved })
      setSavedDefinitionSignature(JSON.stringify(definitionForRequest(saved)))
      setSelectedDefinitionId(saved.id || '')
      setSelectedDefinitionRevision(saved.revision || 1)
      setSavedDefinitions(await listRegimeGraphDefinitions())
      setPreparedPlan(null)
      setNotice(saveAsNew || !definition.id ? `已保存新定义 · r${saved.revision}。` : `已保存修订 · r${saved.revision}。`)
    } catch (reason) { setError(errorText(reason, '定义保存失败。')) } finally { setSavingDefinition(false) }
  }

  const exportDefinition = () => {
    if (formulaPending || formulaBusy) { setError('请先应用或还原公式草稿再导出。'); return }
    const blob = new Blob([JSON.stringify(definitionForRequest(definition), null, 2)], { type: 'application/json;charset=utf-8' })
    const url = URL.createObjectURL(blob)
    const anchor = document.createElement('a'); anchor.href = url; anchor.download = `${definition.name || 'historical-regime'}.json`; anchor.click(); URL.revokeObjectURL(url)
  }

  const pollRun = async (runId: string, requestId: number, controller: AbortController) => {
    while (!controller.signal.aborted && requestId === runRequest.current) {
      const next = await getRegimePreviewRun(runId, controller.signal)
      if (controller.signal.aborted || requestId !== runRequest.current) return
      setRun(next)
      if (['completed', 'failed', 'cancelled'].includes(next.status)) {
        if (next.status === 'failed') setError(typeof next.error === 'string' ? next.error : next.error?.message || next.message || '识别失败，请检查数据和规则。')
        if (next.status === 'completed') {
          setNotice('识别完成。可查看情景走势、完整区间及判断依据。')
          if (!keepEditing.current) { setDisplayedResult({ id: next.id, kind: 'preview' }); setView('result'); setDrawer(null) }
        }
        return
      }
      await new Promise<void>((resolve) => window.setTimeout(resolve, 650))
    }
  }

  const runPreview = async (auditTemporal = false) => {
    if (!canRun) return
    const snapshot = cloneRegimeGraphDefinition(definition)
    const snapshotApiSignature = apiSignature
    const snapshotMode = mode
    const snapshotAsOf = asOf
    const requestId = ++runRequest.current
    runAbort.current?.abort()
    keepEditing.current = false
    setRunDefinition(snapshot)
    setRunDefinitionSignature(regimeCalculationFingerprint(snapshot, snapshotMode, snapshotAsOf))
    setPreviewNodeId(''); setNodePreviewOpen(false); setDisplayedResult(null); setViewedRunId('')
    const controller = new AbortController(); runAbort.current = controller
    setError(''); setNotice(''); setSeriesPage(null)
    setRun({ id: '', status: 'preparing', stage: '准备固定签名计划', progress: 0, message: '正在校验并读取已预热内核。' })
    try {
      const prepared = await prepareRegimeGraph(snapshot, controller.signal)
      if (controller.signal.aborted || requestId !== runRequest.current) return
      if (latestApiSignature.current === snapshotApiSignature && latestRunContext.current === `${snapshotApiSignature}:${snapshotMode}:${snapshotAsOf}`) setPreparedPlan(prepared)
      setRun({ id: '', status: 'preparing', stage: '提交试算任务', progress: 0.05, message: '固定签名计划已验证。' })
      const started = await startRegimePreviewRun(snapshot, { compileToken: prepared.compile_token, mode: snapshotMode, asOf: snapshotAsOf || undefined, ttlSeconds: 1800, auditTemporal }, controller.signal)
      if (controller.signal.aborted || requestId !== runRequest.current) return
      setRun(started)
      await pollRun(started.id, requestId, controller)
    } catch (reason) {
      if (controller.signal.aborted || requestId !== runRequest.current) return
      setRun((current) => ({ id: current?.id || '', status: 'failed', stage: 'failed', progress: current?.progress ?? 0, error: { message: errorText(reason, '历史情景试算失败。'), field: reason instanceof RegimeGraphApiError ? reason.diagnostics.find(item => item.path?.startsWith('graph.nodes.'))?.path : undefined } }))
      setError(errorText(reason, '历史情景试算失败。'))
    }
  }

  const cancelRun = async () => {
    const activeRun = run
    const requestId = ++runRequest.current
    runAbort.current?.abort()
    try { if (activeRun?.id) await cancelRegimePreviewRun(activeRun.id) } catch (reason) { if (requestId === runRequest.current) setError(errorText(reason, '取消试算失败。')); return }
    if (requestId !== runRequest.current) return
    setRun((current) => current ? { ...current, status: 'cancelled', stage: 'cancelled', message: '试算已取消。' } : current); setNotice('识别已取消。')
  }

  const loadSeries = async () => {
    if (!run?.id || run.status !== 'completed') return
    const requestId = ++seriesRequest.current
    seriesAbort.current?.abort()
    const controller = new AbortController(); seriesAbort.current = controller
    const selectedRunId = run.id
    const selectedPreviewNode = previewNodeId
    setLoadingSeries(true); setError('')
    try {
      const page = await getRegimePreviewSeries(selectedRunId, { nodeId: selectedPreviewNode || undefined, limit: 500 }, controller.signal)
      if (!controller.signal.aborted && requestId === seriesRequest.current) setSeriesPage(page)
    } catch (reason) {
      if (!controller.signal.aborted && requestId === seriesRequest.current) setError(errorText(reason, '节点序列读取失败。'))
    } finally {
      if (!controller.signal.aborted && requestId === seriesRequest.current) setLoadingSeries(false)
    }
  }

  const bindSeries = (series: ResearchSeriesCatalogItem) => {
    if (!series.regime_node_type || !series.binding_parameters) { setError('数据目录未返回图谱绑定协议。'); return }
    const schema = schemas.find((item) => schemaId(item) === series.regime_node_type || item.type === series.regime_node_type)
    if (!schema) { setError(`节点目录不支持数据源类型 ${series.regime_node_type}。`); return }
    addNode(schema, series.binding_parameters, series.name); setNotice(`已将“${series.name}”加入计算图。`)
  }

  const loadAssetDefinition = (next: RegimeGraphDefinition) => {
    const draft = { ...next, id: undefined, revision: undefined }
    dispatch({ type: 'reset', definition: draft })
    const firstId = draft.graph.nodes[0]?.id || null
    setSelectedNodeId(firstId); setSelectedGraphNodeIds(firstId ? [firstId] : [])
    setSelectedDefinitionId(''); setSelectedDefinitionRevision(1); setSavedDefinitionSignature('')
    clearRunDisplay()
    nodeSequence.current = draft.graph.nodes.length + 1
  }

  const insertAssetGraph = (graph: RegimeGraphDefinition['graph']) => {
    const occupied = new Set(definition.graph.nodes.map((node) => node.id))
    const sourceIds = new Set(graph.nodes.map((node) => node.id))
    const idMap = new Map<string, string>()
    graph.nodes.forEach((node) => {
      let id = newNodeId(node.type, nodeSequence.current++)
      while (occupied.has(id)) id = newNodeId(node.type, nodeSequence.current++)
      occupied.add(id); idMap.set(node.id, id)
    })
    const inserted = graph.nodes.map((node, index) => {
      const edgeInputs = Object.fromEntries((graph.edges || []).flatMap((edge) => edge.target.node_id === node.id ? [[edge.target.port, edge.source]] : []))
      return {
      ...node,
      id: idMap.get(node.id)!,
      inputs: Object.fromEntries(Object.entries({ ...edgeInputs, ...(node.inputs || {}) }).flatMap(([port, input]) => sourceIds.has(input.node_id) ? [[port, { ...input, node_id: idMap.get(input.node_id)! }]] : [])),
      position: { x: (node.position?.x ?? index * 240) + 48, y: (node.position?.y ?? 0) + 48 },
    }})
    editDefinition({ ...definition, graph: { ...definition.graph, nodes: [...definition.graph.nodes, ...inserted] } })
    const ids = inserted.map((node) => node.id)
    setSelectedGraphNodeIds(ids); setSelectedNodeId(ids[0] || null)
  }


  return (
    <div className={focused ? 'fixed inset-0 z-[100] overflow-auto bg-slate-50 p-3 sm:p-5' : `mx-auto min-w-0 py-6 ${editorMode === 'canvas' ? 'max-w-[1920px]' : 'max-w-[1440px]'}`} data-testid="historical-regime-workbench" onKeyDown={event => { if (focused && !drawer && event.key === 'Escape') { event.preventDefault(); setFocused(false) } }}>
      {!focused && <header aria-label="历史情景工作台命令栏" className="mb-6 flex flex-col gap-4 rounded-2xl bg-gradient-to-r from-slate-950 via-slate-900 to-violet-950 px-5 py-5 text-white shadow-lg sm:px-7 sm:py-6 lg:flex-row lg:items-center lg:justify-between">
        <div><p className="text-sm font-semibold text-violet-200">工作区共享 · 枚举时序算法</p><h1 className="mt-1 text-2xl font-bold tracking-tight sm:text-3xl">历史情景识别</h1><p className="mt-2 max-w-2xl text-sm text-slate-300">通过画布、构建向导或高级公式定义计算；在校验与预览中查看市场状态与区间。</p></div>
        <div className="flex flex-wrap gap-2">
          {onExit && <button type="button" onClick={onExit} className="min-h-10 rounded-lg border border-white/25 px-4 text-sm font-semibold">返回</button>}
          <button type="button" disabled={formulaPending || formulaBusy || loadingDefinition || loadingTemplate} onClick={createBlank} className="min-h-10 rounded-lg border border-white/25 px-4 text-sm font-semibold hover:bg-white/10 disabled:opacity-40">新建情景算法</button>
          {definition.id && <button type="button" disabled={!valid || savingDefinition} onClick={() => void saveDefinition(true)} className="min-h-10 rounded-lg border border-violet-300/70 px-4 text-sm font-semibold text-violet-100 disabled:opacity-40">另存为新算法</button>}
          <button type="button" aria-label="保存" disabled={!valid || savingDefinition || loadingDefinition || loadingTemplate} onClick={() => setDrawer('versions')} className="min-h-10 rounded-lg bg-violet-400 px-4 text-sm font-semibold text-slate-950 hover:bg-violet-300 disabled:opacity-40">保存情景</button>
        </div>
      </header>}
      <div className="mb-4 grid grid-cols-3 gap-1 rounded-xl border border-slate-200 bg-white p-1 md:hidden" role="tablist" aria-label="情景中心区域">
        <button type="button" role="tab" aria-selected={mobileLibrary} onClick={() => setMobileLibrary(true)} className={`min-h-11 rounded-lg text-sm font-semibold ${mobileLibrary ? 'bg-violet-600 text-white' : 'text-slate-600'}`}>算法库</button>
        <button type="button" role="tab" aria-selected={!mobileLibrary && view === 'build'} onClick={() => changeView('build')} className={`min-h-11 rounded-lg text-sm font-semibold ${!mobileLibrary && view === 'build' ? 'bg-violet-600 text-white' : 'text-slate-600'}`}>算法定义</button>
        <button type="button" role="tab" aria-selected={!mobileLibrary && view !== 'build'} onClick={() => changeView('result')} className={`min-h-11 rounded-lg text-sm font-semibold ${!mobileLibrary && view !== 'build' ? 'bg-violet-600 text-white' : 'text-slate-600'}`}>校验与预览</button>
      </div>
      {error && view === 'build' && <p role="alert" className="mb-4 rounded-lg border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-800">{error}</p>}
      {!focused && <div className="mb-3 hidden md:block"><button type="button" onClick={() => setLibraryCollapsed(value => !value)} aria-expanded={!libraryCollapsed} className="min-h-10 rounded-lg border border-slate-200 bg-white px-3 text-sm font-semibold text-slate-600">{libraryCollapsed ? '展开算法库' : '收起算法库'}</button></div>}
      <div className={`grid min-w-0 grid-cols-[minmax(0,1fr)] gap-5 md:items-start ${focused || libraryCollapsed ? 'md:grid-cols-1' : 'md:grid-cols-[minmax(240px,300px)_minmax(0,1fr)]'}`}>
        <aside className={`${mobileLibrary ? 'block' : 'hidden'} min-w-0 md:sticky md:top-5 md:self-start ${focused || libraryCollapsed ? 'md:hidden' : 'md:block'}`}>
          <RegimeDefinitionLibrary definitions={savedDefinitions} templates={templates} schemas={schemas} selectedId={definition.id} selectedTemplate={selectedTemplate} loading={loadingCatalog} busy={savingDefinition || loadingDefinition || loadingTemplate || formulaPending || formulaBusy} onTemplate={id => chooseLibraryItem(() => { void loadTemplate(id) })} onDefinition={item => chooseLibraryItem(() => { void loadSavedDefinition(item.id, item.revision) })} onLegacy={() => setDrawer('legacy')} />
        </aside>
        <main className={`${mobileLibrary ? 'hidden' : 'block'} min-w-0 md:block`}>
          {!focused && <div role="tablist" aria-label="历史情景工作区" className="mb-5 hidden grid-cols-2 gap-2 rounded-2xl border border-slate-200 bg-white p-2 shadow-sm md:grid">
            {([['build', '算法定义', '编辑算法信息、构建公式并完成校验'], ['result', '校验与预览', '选择运行条件，查看市场状态与区间结果']] as const).map(([id, title, help]) => <button type="button" key={id} role="tab" aria-selected={id === 'build' ? view === 'build' : view !== 'build'} onClick={() => changeView(id)} className={`rounded-xl px-4 py-3 text-left transition focus:outline-none focus:ring-2 focus:ring-violet-300 ${(id === 'build' ? view === 'build' : view !== 'build') ? 'bg-violet-600 text-white shadow-sm' : 'text-slate-600 hover:bg-slate-50'}`}><span className="block text-sm font-semibold">{title}</span><span className={`mt-0.5 block text-xs ${(id === 'build' ? view === 'build' : view !== 'build') ? 'text-violet-100' : 'text-slate-400'}`}>{help}</span></button>)}
          </div>}
          <section style={{ display: view === 'build' ? 'block' : 'none' }} aria-label="情景构建视图" className="space-y-5">
            {!focused && <details open={editorMode !== 'canvas' || basicInfoOpen} onToggle={event => { if (editorMode === 'canvas') setBasicInfoOpen(event.currentTarget.open) }}>
              <summary className={editorMode === 'canvas' ? 'cursor-pointer rounded-xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-700' : 'hidden'}><strong>{definition.name || '未命名算法'}</strong><span className="mx-3 text-slate-400">枚举时序 · {definition.id ? `v${definition.revision}` : '未保存'}</span><span className="font-semibold text-violet-700">基本信息与结果设置</span></summary>
              <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
                <h2 className="font-semibold text-slate-900">算法定义</h2><p className="mt-1 text-xs text-slate-500">{definition.id ? `工作区算法 · 当前版本 ${definition.revision}` : selectedTemplate ? '内置算法：保存时将创建工作区副本。' : '未保存草稿'}</p>
                <div className="mt-5 grid gap-4 sm:grid-cols-3">
                  <label className="text-sm font-medium text-slate-700">名称<input aria-label="研究名称" value={definition.name} maxLength={80} onChange={event => editDefinition({ ...definition, name: event.target.value })} className="mt-1 block min-h-10 w-full rounded-lg border border-slate-200 px-3 py-2 focus:outline-none focus:ring-2 focus:ring-violet-100" /></label>
                  <label className="text-sm font-medium text-slate-700">结果类型<input aria-label="结果类型" readOnly value={manualEventMode ? '历史事件区间' : '时序'} className="mt-1 block min-h-10 w-full rounded-lg border border-slate-200 bg-slate-50 px-3 py-2" /></label>
                  <label className="text-sm font-medium text-slate-700">主输出值类型<input aria-label="主输出值类型" readOnly value={manualEventMode ? '多标签区间（允许重叠）' : '有限枚举值'} className="mt-1 block min-h-10 w-full rounded-lg border border-slate-200 bg-slate-50 px-3 py-2" /></label>
                  <label className="text-sm font-medium text-slate-700 sm:col-span-3">说明<textarea aria-label="算法说明" value={definition.description || ''} maxLength={500} rows={2} onChange={event => editDefinition({ ...definition, description: event.target.value })} className="mt-1 block w-full rounded-lg border border-slate-200 px-3 py-2 focus:outline-none focus:ring-2 focus:ring-violet-100" /></label>
                </div>
                <p className="mt-4 rounded-xl border border-sky-100 bg-sky-50 px-3 py-2 text-xs leading-5 text-sky-800">{manualEventMode ? '由人类定义历史事件的日历区间；同一日期可以同时属于多个事件。事件只用于事后研究，不按优先级互相覆盖。' : '每个时点输出一个市场状态，也可附加数值通道。这里定义计算逻辑；市场色带、区间和图表在预览中展示。'}</p>
                <div className="mt-4 grid gap-3 rounded-xl border border-slate-200 bg-slate-50 p-4 text-xs sm:grid-cols-3" aria-label="情景时序输出契约"><div><span className="text-slate-500">日期轴</span><p className="mt-1 font-semibold text-slate-800">随观察序列的观测日期对齐</p></div><div><span className="text-slate-500">{manualEventMode ? '事件定义' : '枚举值域'}</span><p className="mt-1 font-semibold text-slate-800">{manualEventMode ? `${manualEventCount} 个事件 · 允许重叠` : `${definition.states.length} 个状态`}</p></div><div><span className="text-slate-500">{manualEventMode ? '边界口径' : '缺失值'}</span><p className="mt-1 font-semibold text-slate-800">{manualEventMode ? '日历日期闭区间' : '未识别，保留为空'}</p></div></div>
              </div>
            </details>}
            <div className="min-w-0 rounded-2xl border border-slate-200 bg-white p-4 shadow-sm sm:p-5">
              <div className="flex flex-wrap items-start justify-between gap-3"><div><h2 className="font-semibold text-slate-900">算法公式</h2><p className="mt-1 text-xs leading-5 text-slate-500">画布、构建向导和高级公式共用同一份定义。选择适合你的方式编辑。</p></div><button type="button" onClick={() => setDrawer('issues')} className={`min-h-10 rounded-lg px-3 text-xs font-semibold ${valid ? 'bg-emerald-50 text-emerald-800' : 'bg-amber-50 text-amber-800'}`}>{inferenceStatus === 'checking' ? '检查中…' : valid ? `检查通过${issueCount(inference) ? ` · ${issueCount(inference)} 条提示` : ''}` : '查看待修复问题'}</button></div>
              {editorMode !== 'canvas' && drawer !== 'states' && (manualEventMode
                ? <p className="mt-4 rounded-xl border border-violet-200 bg-violet-50 p-3 text-xs leading-5 text-violet-900">人工事件结果由“人工历史事件区间”节点统一输出；事件名称、日期和重叠关系在该节点中编辑，无需配置互斥状态输出。</p>
                : <div className="mt-4"><RegimeSeriesOutputs definition={definition} schemas={schemas} onChange={editDefinition} activeId={activeOutputId} onSelectOutput={setActiveOutputId} /></div>)}
              <div className="mt-4 inline-flex flex-wrap rounded-lg border border-slate-200 bg-slate-50 p-1" role="tablist" aria-label="公式编辑方式">
                {([['canvas', '画布构建'], ['guided', '构建向导'], ['formula', '高级公式']] as const).map(([id, label]) => <button type="button" key={id} role="tab" aria-selected={editorMode === id} onClick={() => changeEditorMode(id)} className={`min-h-10 rounded-md px-3 py-2 text-sm font-semibold focus:outline-none focus:ring-2 focus:ring-violet-300 ${editorMode === id ? 'bg-white text-violet-700 shadow-sm' : 'text-slate-600'}`}>{label}</button>)}
              </div>
              <div className="mt-3 flex flex-wrap gap-2" role="group" aria-label="计算定义编辑历史">
                <button type="button" disabled={!timeline.past.length || formulaPending || formulaBusy} onClick={() => { if (running) keepEditing.current = true; dispatch({ type: 'undo' }) }} className="min-h-10 rounded-lg border border-slate-200 px-3 text-sm disabled:opacity-30">撤销</button>
                <button type="button" disabled={!timeline.future.length || formulaPending || formulaBusy} onClick={() => { if (running) keepEditing.current = true; dispatch({ type: 'redo' }) }} className="min-h-10 rounded-lg border border-slate-200 px-3 text-sm disabled:opacity-30">重做</button>
              </div>
              <div style={{ display: editorMode === 'canvas' ? 'block' : 'none' }} className="mt-4 min-w-0">
                <div className="mb-3 flex flex-wrap items-center gap-2 rounded-xl border border-slate-200 bg-white p-3">
                  <button type="button" disabled={formulaPending || formulaBusy} onClick={() => setDrawer('library')} className="min-h-10 rounded-lg bg-violet-600 px-3 text-sm font-semibold text-white">添加节点</button>
                  {!manualEventMode ? <button type="button" disabled={formulaPending || formulaBusy} onClick={() => setDrawer('states')} className="min-h-10 rounded-lg border border-slate-200 px-3 text-sm font-semibold text-slate-700">输出通道</button> : null}
                  <button type="button" disabled={formulaPending || formulaBusy} onClick={() => setDrawer('assets')} className="min-h-10 px-2 text-xs font-semibold text-slate-600">复用资源</button>
                  <button type="button" disabled={!selectedNode || formulaPending || formulaBusy} onClick={() => setDrawer('node-preview')} className="min-h-10 rounded-lg border border-violet-200 px-3 text-xs font-semibold text-violet-700 disabled:opacity-40">预览所选节点</button>
                  <button type="button" disabled={!run?.id || run.status !== 'completed'} onClick={() => setNodePreviewOpen(value => !value)} className="min-h-10 px-2 text-xs font-semibold text-slate-600 disabled:opacity-40">节点调试</button>
                  <button type="button" onClick={() => { setFocused(value => !value); setDrawer(null) }} className="ml-auto min-h-10 rounded-lg border border-slate-200 px-3 text-sm font-semibold text-slate-700">{focused ? '退出专注' : '专注模式'}</button>
                </div>
                <div className={focused ? 'h-[calc(100dvh-250px)] min-h-[460px]' : 'h-[640px] min-w-0'} data-testid="regime-canvas-workspace"><RegimeGraphCanvas nodes={definition.graph.nodes} edges={edges} schemas={schemas} selectedNodeId={selectedNodeId} onNodesChange={handleCanvasChanges} onConnect={connectCanvasNodes} onDuplicate={duplicateNodes} onSelectionChange={setSelectedGraphNodeIds} /></div>
              </div>
              {editorMode !== 'formula' && <RegimeMathPreview definition={definition} mode={mode} outputId={activeOutputId} />}
              {editorMode === 'guided' && <RegimeGuidedFormulaPanel definition={definition} schemas={schemas} outputId={activeOutputId} onBuild={openBuilder} onResources={() => setDrawer('library')} />}
          {editorMode === 'formula' ? <div className="min-h-0 min-w-0 flex-1 overflow-auto"><RegimeFormulaEditor definition={definition} mode={mode} schemas={schemas} outputId={activeOutputId} onPending={setFormulaPending} onBusy={setFormulaBusy} onApply={(next) => { if (running) keepEditing.current = true; dispatch({ type: 'edit', definition: cloneRegimeGraphDefinition(next) }); setPreparedPlan(null); setNotice('公式已应用，画布和构建向导已同步。') }} /></div> : null}
              {nodePreviewOpen ? <div className="max-h-[45%] shrink-0 overflow-auto border-t border-slate-200 bg-white"><div className="flex items-center justify-between px-3 py-2"><p className="text-xs text-slate-500">中间节点调试 · 仅展示所选运行快照的局部数据</p><button type="button" onClick={() => setNodePreviewOpen(false)} className="text-xs font-bold text-slate-600">关闭节点调试</button></div><RegimeResultDock run={run} page={seriesPage} nodes={runDefinition?.graph.nodes || []} schemas={schemas} previewNodeId={previewNodeId} loadingSeries={loadingSeries} onPreviewNode={setPreviewNodeId} onLoadSeries={() => void loadSeries()} /></div> : null}
              <div className="mt-5 flex flex-wrap items-center justify-between gap-3 border-t border-slate-100 pt-4"><span className="text-xs text-slate-500">{definitionDirty ? '未保存改动' : `已保存 v${definition.revision}`}</span><button type="button" disabled={formulaPending || formulaBusy} onClick={() => changeView('result')} className="min-h-10 rounded-lg bg-violet-600 px-4 text-sm font-semibold text-white disabled:opacity-40">前往校验与预览</button></div>
            </div>
          </section>
          <section style={{ display: view !== 'build' ? 'block' : 'none' }} aria-label="情景校验与预览设置" className="mb-5 space-y-4 rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
            <div className="flex flex-wrap items-start justify-between gap-3"><div><h2 className="font-semibold text-slate-900">校验与预览</h2><p className="mt-1 text-xs leading-5 text-slate-500">选择识别方式和截至日。结果保留运行时的定义与数据快照。</p></div><button type="button" disabled={formulaPending || formulaBusy} onClick={() => setDrawer('versions')} className="min-h-10 rounded-lg border border-slate-200 px-3 text-sm font-semibold text-slate-700">保存情景</button></div>
            <div className="flex flex-wrap items-end gap-4">
              <fieldset><legend className="mb-2 text-xs font-semibold text-slate-600">识别方式</legend><div role="radiogroup" aria-label="V2 识别模式" className="flex rounded-lg border border-slate-200 bg-slate-50 p-1"><button type="button" role="radio" aria-checked={mode === 'realtime'} onClick={() => { if (running) keepEditing.current = true; setMode('realtime') }} className={`min-h-10 rounded-md px-3 text-sm font-semibold ${mode === 'realtime' ? 'bg-white text-emerald-800 shadow-sm' : 'text-slate-500'}`}>实时识别</button><button type="button" role="radio" aria-checked={mode === 'retrospective'} onClick={() => { if (running) keepEditing.current = true; setMode('retrospective') }} className={`min-h-10 rounded-md px-3 text-sm font-semibold ${mode === 'retrospective' ? 'bg-white text-amber-800 shadow-sm' : 'text-slate-500'}`}>事后研究</button></div></fieldset>
              <label className="text-xs font-semibold text-slate-600">截至日<input aria-label="V2 截至日" type="date" value={asOf} onChange={event => { if (running) keepEditing.current = true; setAsOf(event.target.value) }} className="mt-2 block min-h-11 rounded-lg border border-slate-200 bg-white px-3 text-sm" /></label>
              <button type="button" disabled={!canRun || running} onClick={() => void runPreview()} className="min-h-11 rounded-lg bg-violet-600 px-5 text-sm font-semibold text-white disabled:opacity-40">{running ? '识别中…' : '运行识别'}</button>
              <button type="button" onClick={() => setDrawer('issues')} className="min-h-11 rounded-lg border border-slate-200 px-3 text-sm font-semibold text-slate-700">校验定义</button>
            </div>
            {error && view !== 'build' && <p role="alert" className="rounded-lg border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-800">{error}</p>}
            <RegimeTemporalPanel report={run?.result?.temporal_capability && !runStale ? run.result.temporal_capability : temporal} stale={runStale && Boolean(run?.result?.temporal_capability?.verified)} busy={running} disabled={!canRun} onAudit={() => void runPreview(true)} onNode={id => { setView('build'); selectNode(id) }} />
            {!canRun && !running && <InferenceIssues inference={inference} status={inferenceStatus} onNode={id => { setView('build'); selectNode(id) }} />}
            {run && <div role="status" aria-label="识别运行状态" className="space-y-2 rounded-xl bg-slate-50 p-3 text-sm text-slate-700">
              <strong>{running ? '正在识别' : run.status === 'failed' ? '本次识别失败' : run.status === 'cancelled' ? '识别已取消' : '识别完成'}</strong>
              {running && <><p>{run.message || run.stage} · {Math.round((run.progress || 0) * 100)}%</p><progress aria-label="识别进度" max={1} value={run.progress || 0} className="w-full" /><button type="button" onClick={() => void cancelRun()} className="min-h-10 text-rose-700">取消本次识别</button></>}
              {run.status === 'failed' && <button type="button" onClick={() => { changeView('build'); const field = typeof run.error === 'object' ? run.error?.field : undefined; const node = definition.graph.nodes.find(item => field?.startsWith(`graph.nodes.${item.id}.`)); if (node) selectNode(node.id) }} className="min-h-10 font-semibold text-violet-700">检查输入数据</button>}
            </div>}
            <div className="flex flex-wrap gap-3 border-t border-slate-100 pt-3 text-xs"><button type="button" aria-pressed={view === 'result'} onClick={() => changeView('result')} className="min-h-10 font-semibold text-violet-700">识别结果</button><button type="button" aria-pressed={view === 'experiments'} onClick={() => changeView('experiments')} className="min-h-10 font-semibold text-violet-700">实验对比</button><button type="button" onClick={() => setDataLabOpen(true)} className="min-h-10 font-semibold text-slate-600">数据实验室</button></div>
          </section>
          {!loadingCatalog && realtimeBlockedNodes.length > 0 ? <p role="alert" className="shrink-0 border-b border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-900">{REALTIME_RESTRICTION} 涉及：{realtimeBlockedNodes.map((node) => node.label || schemas.find((schema) => schemaId(schema) === node.type)?.label || node.type).join('、')}</p> : null}
      {viewingFormal ? <p className="shrink-0 border-b border-indigo-100 bg-indigo-50 px-3 py-2 text-xs text-indigo-900">正在查看正式运行的冻结版本；其模式、截至日和修订以结果详情为准。</p> : runStale ? <p role="status" className="shrink-0 border-b border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-900">结果与当前配置不同（结果已过期）。当前查看的仍是原运行快照；请重新运行识别以更新结果。</p> : presentationChanged ? <p className="shrink-0 border-b border-indigo-100 bg-indigo-50 px-3 py-2 text-xs text-indigo-900">展示信息已修改；历史结果仍使用运行时的名称和颜色。</p> : null}
      {evaluationChanged && !viewingFormal ? <p role="status" className="shrink-0 border-b border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-900">评估对象已修改；当前结果仍使用原评估快照，分类结果保持不变。</p> : null}
          <section style={{ display: view === 'result' ? 'block' : 'none' }} className="min-h-0 min-w-0 flex-1 overflow-auto p-3" aria-label="情景结果视图">{displayedResult && (view === 'result' || viewedRunId === `${displayedResult.kind}:${displayedResult.id}`) ? <RegimeResultView runId={displayedResult.id} runKind={displayedResult.kind} stale={displayedResult.kind === 'preview' && runStale} /> : !run && <div className="grid h-full min-h-48 place-items-center text-center"><div><h2 className="text-base font-bold text-slate-800">还没有完成的情景结果</h2><p className="mt-2 text-xs text-slate-500">在构建视图选择数据和规则，然后运行识别。</p><button type="button" onClick={() => changeView('build')} className="mt-4 rounded-lg bg-indigo-600 px-4 py-2 text-xs font-bold text-white">返回构建</button></div></div>}</section>
          <section style={{ display: view === 'experiments' ? 'block' : 'none' }} className="min-h-0 min-w-0 flex-1 space-y-4 overflow-auto p-3" aria-label="情景实验对比视图"><RegimeValidationPanel definition={definition} onChange={editDefinition} /><RegimeExperimentPanel definition={definition} schemas={schemas} dirty={definitionDirty} valid={canRun} mode={mode} asOf={asOf} preparedPlan={preparedPlan} onPrepared={setPreparedPlan} onError={setError} onNotice={setNotice} /></section>
        </main>
      </div>
      {drawer ? <RegimeWorkbenchDrawer title={{ library: '添加节点', inspector: '节点参数', states: '输出通道', versions: '保存情景', issues: '检查问题', assets: '复用资源', builder: '公式构建向导', 'node-preview': '节点预览', legacy: '历史版本与旧版迁移目录' }[drawer]} side="right" wide={['versions', 'assets', 'builder', 'legacy', 'states', 'node-preview'].includes(drawer)} onClose={() => { if (!savingResearch) setDrawer(null) }} closeDisabled={savingResearch}>
          {drawer === 'node-preview' ? <RegimeNodePreviewPanel definition={definition} schemas={schemas} initialNodeId={selectedNodeId} initialMode={mode} initialAsOf={asOf} onChange={editDefinition} /> : null}
          {drawer === 'legacy' ? <HistoricalRegimeDirectory /> : null}
          {drawer === 'builder' ? <RegimeGuidedEditor builderOnly onExpandNode={id => { void expand(definition, id) }} expanding={expanding} expandDisabled={formulaPending || formulaBusy || (mode === 'realtime' && realtimeNodeBlocked(selectedSchema))} onPreviewNode={id => { setSelectedNodeId(id); setDrawer('node-preview') }} selectedNodeId={selectedNodeId} onSelectNode={setSelectedNodeId} active={editorMode === 'guided' && view === 'build'} onChange={editDefinition} definition={definition} schemas={schemas} templates={templates} selectedTemplate={selectedTemplate} loadingTemplate={loadingTemplate} onTemplate={setSelectedTemplate} onLoadTemplate={() => void loadTemplate()} onBlank={createBlank} onPatchNode={(id, patch) => editDefinition({ ...definition, graph: { ...definition.graph, nodes: definition.graph.nodes.map((node) => node.id === id ? { ...node, ...patch } : node) } })} onAddNode={() => setDrawer('library')} onRemoveNode={(id) => handleCanvasChanges([{ type: 'remove', id }])} onSetState={(reference) => editDefinition({ ...definition, graph: { ...definition.graph, outputs: { ...definition.graph.outputs, state: reference || undefined } } })} onEditNode={(id) => { changeEditorMode('canvas'); selectNode(id) }} onOpenStates={() => setDrawer('states')} onOpenDataLab={() => setDataLabOpen(true)} onRun={() => void runPreview()} canRun={canRun && !running} /> : null}
          {drawer === 'library' ? <ResourceLibrary schemas={selectableSchemas} onAdd={addNode} onOpenDataLab={() => setDataLabOpen(true)} busy={expanding} /> : null}
          {drawer === 'inspector' ? <><button type="button" disabled={!selectedNode || formulaPending || formulaBusy} onClick={() => setDrawer('node-preview')} className="min-h-10 w-full rounded-lg bg-violet-600 px-3 text-sm font-semibold text-white disabled:opacity-40">预览此节点</button><RegimeNodeInspector states={definition.states} expanding={expanding} expandDisabled={formulaPending || formulaBusy || (mode === 'realtime' && realtimeNodeBlocked(selectedSchema))} onExpand={() => { if (selectedNode) void expand(definition, selectedNode.id) }} node={selectedNode} schema={selectedSchema} nodes={definition.graph.nodes} schemas={schemas} outputs={definition.graph.outputs} inference={inference} preparedPlan={preparedPlan} onPatchNode={patchSelectedNode} onConnect={connectSelectedNode} onSetOutput={setSelectedOutput} onRemove={() => { if (selectedNode) handleCanvasChanges([{ type: 'remove', id: selectedNode.id }]); setDrawer(null) }} />{run?.status === 'completed' && runDefinition?.graph.nodes.some((node) => node.id === selectedNodeId) ? <button type="button" onClick={() => { setPreviewNodeId(selectedNodeId || ''); setNodePreviewOpen(true); setDrawer(null) }} className="w-full rounded-lg border border-indigo-200 bg-white px-3 py-2 text-xs font-bold text-indigo-700">查看此节点的运行结果</button> : null}</> : null}
          {drawer === 'states' ? (manualEventMode ? <p className="rounded-xl bg-violet-50 p-4 text-sm text-violet-900">人工历史事件使用多标签区间，不配置互斥状态输出。请在“人工历史事件区间”节点中维护事件。</p> : <RegimeSeriesOutputs definition={definition} schemas={schemas} onChange={editDefinition} activeId={activeOutputId} onSelectOutput={setActiveOutputId} />) : null}
          {drawer === 'issues' ? <InferenceIssues inference={inference} status={inferenceStatus} onNode={(id) => { setView('build'); setEditorMode('canvas'); selectNode(id) }} /> : null}
          {drawer === 'assets' ? <RegimeGraphAssetsPanel definition={definition} selectedNodeIds={selectedGraphNodeIds} valid={valid} onLoadDefinition={loadAssetDefinition} onInsertGraph={insertAssetGraph} onError={setError} onNotice={setNotice} /> : null}
          {drawer === 'versions' ? <RegimeSavePanel definition={definition} dirty={definitionDirty} valid={canRun} mode={mode} asOf={asOf} onBusy={setSavingResearch} onSaved={saved => { dispatch({ type: 'reset', definition: saved }); setSavedDefinitionSignature(JSON.stringify(definitionForRequest(saved))); setSelectedDefinitionId(saved.id || ''); setSelectedDefinitionRevision(saved.revision || 1); setSavedDefinitions(current => [saved, ...current.filter(item => item.id !== saved.id)]); setPreparedPlan(null) }} onViewResult={id => { setDisplayedResult({ id, kind: 'formal' }); setView('result'); setDrawer(null) }}><VersionBar definitions={savedDefinitions} selectedId={selectedDefinitionId} selectedRevision={selectedDefinitionRevision} current={definition} dirty={definitionDirty} valid={valid} busy={savingDefinition || loadingDefinition || running} onSelect={selectSavedDefinition} onRevision={setSelectedDefinitionRevision} onLoad={() => void loadSavedDefinition()} onSave={() => void saveDefinition(false)} onSaveAs={() => void saveDefinition(true)} /><button type="button" onClick={exportDefinition} className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-xs font-bold text-slate-700">导出定义</button><RegimeLifecyclePanel definition={definition} dirty={definitionDirty} valid={canRun} mode={mode} asOf={asOf} onError={setError} onNotice={setNotice} onViewResult={(id) => { if (running) keepEditing.current = true; setDisplayedResult({ id, kind: 'formal' }); setView('result'); setDrawer(null) }} /></RegimeSavePanel> : null}
        </RegimeWorkbenchDrawer> : null}
        {dataLabOpen ? <div className="fixed inset-0 z-[120] min-w-0 overflow-auto bg-slate-50"><ResearchDataLab embedded boundSeriesIds={boundSeriesIds} onBindSeries={bindSeries} onClose={() => setDataLabOpen(false)} /></div> : null}
      {running || notice || loadingCatalog || loadingDefinition ? <footer role="status" className="flex shrink-0 flex-wrap items-center gap-2 border-t border-slate-200 bg-white px-3 py-2 text-xs text-slate-600"><span className="min-w-0 flex-1">{loadingCatalog ? '正在加载模板与可用节点…' : loadingDefinition ? '正在加载方案…' : running ? `${run?.stage || '正在识别'} · ${Math.round((run?.progress || 0) * 100)}%` : notice}</span>{running ? <button type="button" onClick={() => void cancelRun()} className="font-bold text-rose-700">取消识别</button> : null}{run?.status === 'completed' && view !== 'result' ? <button type="button" onClick={() => { if (run?.id) setDisplayedResult({ id: run.id, kind: 'preview' }); changeView('result') }} className="font-bold text-indigo-700">查看结果</button> : null}{notice && !running ? <button type="button" aria-label="关闭状态提示" onClick={() => setNotice('')} className="px-2 font-bold text-slate-500">关闭</button> : null}</footer> : null}
    </div>
  )
}

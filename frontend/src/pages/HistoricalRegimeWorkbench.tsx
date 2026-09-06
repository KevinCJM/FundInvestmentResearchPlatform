import { useEffect, useMemo, useReducer, useRef, useState } from 'react'
import ResearchDataLab from './ResearchDataLab'
import RegimeGraphCanvas, {
  type RegimeCanvasEdge,
  type RegimeCanvasNodeChange,
} from './regime-workbench/RegimeGraphCanvas'
import RegimeNodeInspector from './regime-workbench/RegimeNodeInspector'
import RegimeResultDock from './regime-workbench/RegimeResultDock'
import RegimeLifecyclePanel from './regime-workbench/RegimeLifecyclePanel'
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
import { listResearchSeries, type ResearchSeriesCatalogItem } from '../services/researchSeries'
import RegimeHelpTip from './regime-workbench/RegimeHelpTip'

interface DefinitionTimeline {
  past: RegimeGraphDefinition[]
  present: RegimeGraphDefinition
  future: RegimeGraphDefinition[]
}

type DefinitionAction =
  | { type: 'edit'; definition: RegimeGraphDefinition }
  | { type: 'reset'; definition: RegimeGraphDefinition }
  | { type: 'undo' }
  | { type: 'redo' }

function sameDefinition(left: RegimeGraphDefinition, right: RegimeGraphDefinition) {
  return JSON.stringify(left) === JSON.stringify(right)
}

function timelineReducer(state: DefinitionTimeline, action: DefinitionAction): DefinitionTimeline {
  if (action.type === 'reset') return { past: [], present: cloneRegimeGraphDefinition(action.definition), future: [] }
  if (action.type === 'undo') {
    const previous = state.past[state.past.length - 1]
    if (!previous) return state
    return { past: state.past.slice(0, -1), present: previous, future: [state.present, ...state.future] }
  }
  if (action.type === 'redo') {
    const next = state.future[0]
    if (!next) return state
    return { past: [...state.past, state.present], present: next, future: state.future.slice(1) }
  }
  if (sameDefinition(state.present, action.definition)) return state
  return { past: [...state.past.slice(-49), state.present], present: action.definition, future: [] }
}

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

function CommandBar({
  definition,
  templates,
  selectedTemplate,
  canUndo,
  canRedo,
  mode,
  asOf,
  inferenceStatus,
  inference,
  loadingTemplate,
  run,
  runStale,
  onName,
  onTemplate,
  onLoadTemplate,
  onBlank,
  onUndo,
  onRedo,
  onMode,
  onAsOf,
  onRun,
  onCancel,
  onExport,
  onExit,
}: {
  definition: RegimeGraphDefinition
  templates: RegimeGraphTemplate[]
  selectedTemplate: string
  canUndo: boolean
  canRedo: boolean
  mode: RegimeMode
  asOf: string
  inferenceStatus: 'idle' | 'checking' | 'ready' | 'invalid' | 'error'
  inference: RegimeGraphInference | null
  loadingTemplate: boolean
  run: RegimePreviewRun | null
  runStale: boolean
  onName: (name: string) => void
  onTemplate: (id: string) => void
  onLoadTemplate: () => void
  onBlank: () => void
  onUndo: () => void
  onRedo: () => void
  onMode: (mode: RegimeMode) => void
  onAsOf: (value: string) => void
  onRun: () => void
  onCancel: () => void
  onExport: () => void
  onExit?: () => void
}) {
  const running = run && ['queued', 'preparing', 'running'].includes(run.status)
  const valid = inferenceStatus === 'ready' && inference?.valid
  return (
    <header className="rounded-2xl border border-slate-800 bg-slate-950 p-3 text-white shadow-lg sm:p-4" aria-label="历史情景工作台命令栏">
      <div className="flex flex-col gap-3 2xl:flex-row 2xl:items-end">
        <div className="min-w-0 2xl:w-72"><p className="text-[10px] font-bold uppercase tracking-[0.18em] text-indigo-300">情景识别研究工作台 <RegimeHelpTip dark label="研究名称说明" text="研究名称用于在定义目录、实验结果和下游引用中识别这套计算图，不影响计算本身。" /></p><input aria-label="研究名称" value={definition.name} onChange={(event) => onName(event.target.value)} className="mt-1 min-h-10 w-full rounded-lg border border-white/15 bg-white/10 px-3 text-sm font-bold text-white placeholder:text-slate-400 focus:border-indigo-300 focus:outline-none" /></div>
        <div className="grid min-w-0 flex-1 gap-2 sm:grid-cols-[minmax(160px,1fr)_auto] 2xl:max-w-xl"><label className="text-[10px] font-bold text-slate-300">模板<RegimeHelpTip dark label="图谱模板说明" text="模板只是可复制的起点。载入后会成为独立草稿，所有节点、连线和参数都可以修改，不会反向改动原模板。" /><select aria-label="图谱模板" value={selectedTemplate} onChange={(event) => onTemplate(event.target.value)} className="mt-1 min-h-10 w-full rounded-lg border border-white/15 bg-slate-900 px-2 text-xs font-normal text-white"><option value="">选择可完全编辑的模板</option>{templates.map((template) => <option key={template.id} value={template.id}>{template.name}</option>)}</select></label><button type="button" disabled={!selectedTemplate || loadingTemplate || Boolean(running)} onClick={onLoadTemplate} className="min-h-10 self-end rounded-lg border border-indigo-300/50 px-3 text-xs font-bold text-indigo-100 disabled:opacity-40">{loadingTemplate ? '载入中…' : '载入模板'}</button></div>
        <div className="flex flex-wrap items-end gap-1.5"><button type="button" disabled={Boolean(running)} onClick={onBlank} className="min-h-10 rounded-lg border border-white/15 px-3 text-xs font-bold text-slate-200 disabled:opacity-40">新建空白</button><button type="button" disabled={!canUndo || Boolean(running)} onClick={onUndo} className="min-h-10 rounded-lg border border-white/15 px-3 text-xs font-bold disabled:opacity-30">撤销</button><button type="button" disabled={!canRedo || Boolean(running)} onClick={onRedo} className="min-h-10 rounded-lg border border-white/15 px-3 text-xs font-bold disabled:opacity-30">重做</button><button type="button" onClick={onExport} className="min-h-10 rounded-lg border border-white/15 px-3 text-xs font-bold">导出定义</button>{onExit ? <button type="button" onClick={onExit} className="min-h-10 rounded-lg border border-white/15 px-3 text-xs font-bold">返回经典版</button> : <a href="/settings/scenario-algorithms" className="inline-flex min-h-10 items-center rounded-lg border border-white/15 px-3 text-xs font-bold">返回情景算法中心</a>}</div>
      </div>
      <div className="mt-3 flex flex-col gap-2 border-t border-white/10 pt-3 lg:flex-row lg:items-center lg:justify-between">
        <div className="flex flex-wrap items-center gap-2"><div className="flex items-center"><div role="radiogroup" aria-label="V2 识别模式" className="flex rounded-lg bg-white/10 p-1"><button type="button" role="radio" aria-checked={mode === 'realtime'} onClick={() => onMode('realtime')} className={`min-h-8 rounded-md px-3 text-[11px] font-bold ${mode === 'realtime' ? 'bg-emerald-500 text-slate-950' : 'text-slate-300'}`}>实时识别</button><button type="button" role="radio" aria-checked={mode === 'retrospective'} onClick={() => onMode('retrospective')} className={`min-h-8 rounded-md px-3 text-[11px] font-bold ${mode === 'retrospective' ? 'bg-amber-400 text-slate-950' : 'text-slate-300'}`}>事后研究</button></div><RegimeHelpTip dark label="识别模式说明" text="实时识别严格按当时可得信息运行；事后研究允许使用最终修订数据或非因果算法，只用于解释历史。" /></div><label className="text-[10px] font-bold text-slate-300">截至日<RegimeHelpTip dark label="截至日说明" text="限制本次检查和试算能看到的数据截止日期；留空表示使用当前快照中的全部可用数据。" /><input aria-label="V2 截至日" type="date" value={asOf} onChange={(event) => onAsOf(event.target.value)} className="ml-2 min-h-9 rounded-lg border border-white/15 bg-slate-900 px-2 font-normal text-white" /></label><span role="status" className={`rounded-full px-2.5 py-1 text-[10px] font-bold ${valid ? 'bg-emerald-400/20 text-emerald-200' : inferenceStatus === 'checking' ? 'bg-indigo-400/20 text-indigo-200' : 'bg-amber-400/20 text-amber-200'}`}>{inferenceStatus === 'checking' ? '300ms 检查中' : valid ? `图谱有效 · ${issueCount(inference)} 条提示` : inferenceStatus === 'invalid' ? `图谱待修复 · ${inference?.errors.length ?? 0} 错误` : '等待图谱检查'}</span>{runStale ? <span className="rounded-full bg-amber-300 px-2.5 py-1 text-[10px] font-bold text-amber-950">结果已过期</span> : null}</div>
        <div className="flex gap-2">{running ? <button type="button" onClick={onCancel} className="min-h-10 rounded-xl border border-rose-300 px-4 text-sm font-bold text-rose-100">取消试算</button> : null}<button type="button" disabled={!valid || Boolean(running)} onClick={onRun} className="min-h-10 rounded-xl bg-indigo-500 px-5 text-sm font-bold text-white shadow disabled:cursor-not-allowed disabled:opacity-40">{running ? `${run?.stage || '正在试算'}…` : '预热并试算'}</button></div>
      </div>
    </header>
  )
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

function ResourceLibrary({ schemas, onAdd, onOpenDataLab }: { schemas: RegimeNodeSchema[]; onAdd: (schema: RegimeNodeSchema) => void; onOpenDataLab: () => void }) {
  const [query, setQuery] = useState('')
  const [category, setCategory] = useState('all')
  const categories = [...new Map(schemas.map((schema) => [schema.category, schema.category_label || schema.category])).entries()]
  const normalized = query.trim().toLowerCase()
  const visible = schemas.filter((schema) => (category === 'all' || schema.category === category) && (!normalized || `${schema.label} ${schema.description || ''} ${schema.tags?.join(' ') || ''}`.toLowerCase().includes(normalized)))
  return <aside className="min-w-0 rounded-2xl border border-slate-200 bg-white p-3 shadow-sm" aria-label="历史情景节点资源库"><div className="flex items-center justify-between"><div><h3 className="text-sm font-bold text-slate-950">节点资源库 <RegimeHelpTip label="节点资源库说明" text="这里列出系统白名单中的数据、算子、模型和状态处理节点。点击添加后可在画布连接，并在右侧调整参数。" /></h3><p className="mt-1 text-[10px] text-slate-500">目录、可用性和参数均来自服务端</p></div><button type="button" onClick={onOpenDataLab} title="浏览并分析已下载的指数、宏观、指标与上传数据" className="min-h-9 rounded-lg bg-indigo-50 px-2 text-[11px] font-bold text-indigo-700">数据实验室</button></div><label className="mt-3 block text-[11px] font-bold text-slate-600">搜索节点<RegimeHelpTip label="搜索节点说明" text="按中文名称、说明或标签筛选节点，不会改变当前计算图。" /><input aria-label="搜索计算节点" value={query} onChange={(event) => setQuery(event.target.value)} className="mt-1 min-h-9 w-full rounded-lg border border-slate-300 px-2 font-normal" placeholder="滤波、回归、HMM…" /></label><div className="mt-2 flex gap-1 overflow-x-auto pb-1"><button type="button" aria-pressed={category === 'all'} onClick={() => setCategory('all')} className={`min-h-8 shrink-0 rounded-md px-2 text-[10px] font-bold ${category === 'all' ? 'bg-slate-950 text-white' : 'bg-slate-100 text-slate-600'}`}>全部</button>{categories.map(([id, label]) => <button key={id} type="button" aria-pressed={category === id} onClick={() => setCategory(id)} className={`min-h-8 shrink-0 rounded-md px-2 text-[10px] font-bold ${category === id ? 'bg-slate-950 text-white' : 'bg-slate-100 text-slate-600'}`}>{label}</button>)}</div><div className="mt-3 max-h-[520px] space-y-2 overflow-y-auto pr-1">{visible.map((schema) => { const available = schema.available !== false && (!schema.status || schema.status === 'available'); return <article key={schemaId(schema)} className={`rounded-xl border p-3 ${available ? 'border-slate-200' : 'border-amber-200 bg-amber-50/60'}`}><div className="flex items-start justify-between gap-2"><div className="min-w-0"><p className="truncate text-xs font-bold text-slate-900">{schema.label} <RegimeHelpTip label={`${schema.label}说明`} text={schema.description || `${schema.label}用于历史情景识别计算。`} /></p><p className="mt-1 line-clamp-2 text-[10px] leading-4 text-slate-500">{schema.description || '系统注册的历史情景计算节点'}</p>{!available ? <p role="note" className="mt-1 text-[10px] font-semibold leading-4 text-amber-800">{schema.unavailable_reason || '当前节点暂不可用'}</p> : null}</div><button type="button" aria-label={`添加${schema.label}`} disabled={!available} title={!available ? schema.unavailable_reason || '当前不可用' : `将${schema.label}加入计算图`} onClick={() => onAdd(schema)} className="min-h-8 shrink-0 rounded-lg bg-indigo-600 px-2 text-[10px] font-bold text-white disabled:cursor-not-allowed disabled:bg-slate-300 disabled:text-slate-600">{available ? '添加' : '不可用'}</button></div><div className="mt-2 flex flex-wrap gap-1"><span className="rounded bg-slate-100 px-1.5 py-0.5 text-[9px] font-bold text-slate-500">{schema.category_label || '计算节点'}</span>{schema.execution_policy?.njit_required ? <span className="rounded bg-emerald-100 px-1.5 py-0.5 text-[9px] font-bold text-emerald-700">NJIT 高性能计算</span> : null}{schema.execution_policy?.third_party_exempt ? <span className="rounded bg-violet-100 px-1.5 py-0.5 text-[9px] font-bold text-violet-700">隔离优化模型</span> : null}{!available ? <span className="rounded bg-amber-100 px-1.5 py-0.5 text-[9px] font-bold text-amber-800">需管理员适配器</span> : null}</div></article> })}{!visible.length ? <p className="rounded-xl border border-dashed border-slate-300 p-4 text-center text-xs text-slate-500">没有匹配节点。</p> : null}</div></aside>
}

function StateDictionary({ definition, onChange }: { definition: RegimeGraphDefinition; onChange: (next: RegimeGraphDefinition) => void }) {
  const addState = () => {
    const index = definition.states.length + 1
    onChange({ ...definition, states: [...definition.states, { id: `state_${index}`, label: `状态 ${index}`, role: 'neutral', order: definition.states.length, color: '#64748b' }] })
  }
  return <section className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm" aria-label="状态语义字典"><div className="flex items-center justify-between"><div><h3 className="text-sm font-bold text-slate-950">状态语义 <RegimeHelpTip label="状态语义说明" text="把模型输出的状态编号解释为牛市、熊市、复苏、滞胀等业务名称。这里不决定算法，只负责结果含义、颜色和下游展示。" /></h3><p className="mt-1 text-[10px] text-slate-500">判定逻辑在模型节点，标签用于解释与下游联动。</p></div><button type="button" onClick={addState} title="新增一个可命名的业务状态" className="min-h-8 rounded-lg bg-slate-100 px-2 text-[10px] font-bold text-slate-700">添加状态</button></div>{definition.states.length ? <div className="mt-3 grid grid-cols-[42px_minmax(0,1fr)_minmax(0,1fr)_auto] gap-1 text-[9px] font-bold text-slate-500"><span>颜色<RegimeHelpTip label="状态颜色说明" text="该状态在图表背景带、区间和图例中的颜色。" /></span><span>状态编号<RegimeHelpTip label="状态编号说明" text="供模型映射和接口引用的稳定编号；发布后不应改变原有编号含义。" /></span><span>显示名称<RegimeHelpTip label="状态显示名称说明" text="用户在图表、条件表现和下游页面中看到的中文名称。" /></span><span /></div> : null}<div className="mt-1 space-y-2">{definition.states.map((state, index) => <div key={`${state.id}-${index}`} className="grid grid-cols-[42px_minmax(0,1fr)_minmax(0,1fr)_auto] gap-1"><input aria-label={`状态${index + 1}颜色`} title="该状态在图表背景带和图例中的颜色" type="color" value={state.color || '#64748b'} onChange={(event) => onChange({ ...definition, states: definition.states.map((item, itemIndex) => itemIndex === index ? { ...item, color: event.target.value } : item) })} className="h-9 w-10 rounded border border-slate-200" /><input aria-label={`状态${index + 1}编号`} title="供模型映射和接口引用的稳定编号" value={state.id} onChange={(event) => onChange({ ...definition, states: definition.states.map((item, itemIndex) => itemIndex === index ? { ...item, id: event.target.value } : item) })} className="min-h-9 min-w-0 rounded-lg border border-slate-300 px-2 text-[11px]" /><input aria-label={`状态${index + 1}名称`} title="用户在图表、分段统计和下游页面中看到的名称" value={state.label} onChange={(event) => onChange({ ...definition, states: definition.states.map((item, itemIndex) => itemIndex === index ? { ...item, label: event.target.value } : item) })} className="min-h-9 min-w-0 rounded-lg border border-slate-300 px-2 text-[11px]" /><button type="button" aria-label={`删除状态${index + 1}`} onClick={() => onChange({ ...definition, states: definition.states.filter((_, itemIndex) => itemIndex !== index) })} className="min-h-9 rounded-lg px-2 text-[10px] font-bold text-rose-600">删除</button></div>)}{!definition.states.length ? <p className="rounded-lg bg-slate-50 p-3 text-xs text-slate-500">尚未定义状态语义。</p> : null}</div></section>
}

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
  const [researchSeries, setResearchSeries] = useState<ResearchSeriesCatalogItem[]>([])
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
  const [resultHeight, setResultHeight] = useState(360)
  const [resultCollapsed, setResultCollapsed] = useState(false)
  const [dataLabOpen, setDataLabOpen] = useState(false)
  const [loadingTemplate, setLoadingTemplate] = useState(false)
  const [loadingCatalog, setLoadingCatalog] = useState(true)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const nodeSequence = useRef(1)
  const inferenceRequest = useRef(0)
  const runRequest = useRef(0)
  const runAbort = useRef<AbortController | null>(null)

  const routeSearch = typeof window === 'undefined' ? '' : window.location.search
  const routeQuery = useMemo(() => new URLSearchParams(routeSearch), [routeSearch])
  const routeDefinitionId = routeQuery.get('definition')?.trim() || ''
  const routeRevisionRaw = routeQuery.get('revision')?.trim() || ''
  const routeTemplateId = routeQuery.get('template')?.trim() || ''
  const routeLoadKey = routeDefinitionId
    ? `definition:${routeDefinitionId}:revision:${routeRevisionRaw}`
    : routeTemplateId ? `template:${routeTemplateId}` : ''

  const apiSignature = useMemo(() => JSON.stringify(definitionForRequest(definition)), [definition])
  const edges = useMemo(() => edgesFrom(definition.graph.nodes), [definition.graph.nodes])
  const selectedNode = definition.graph.nodes.find((node) => node.id === selectedNodeId) ?? null
  const selectedSchema = selectedNode ? schemas.find((schema) => schemaId(schema) === selectedNode.type || schema.type === selectedNode.type) : undefined
  const runStale = Boolean(run && runDefinitionSignature && runDefinitionSignature !== apiSignature)
  const definitionDirty = !savedDefinitionSignature || savedDefinitionSignature !== apiSignature
  const boundSeriesIds = definition.graph.nodes.flatMap((node) => Object.entries(node.parameters).flatMap(([key, value]) => typeof value === 'string' && (key === 'artifact_id' || /^(index|macro|indicator|upload):/.test(value)) ? [value] : []))

  useEffect(() => { setPreparedPlan(null) }, [apiSignature])

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
    void listResearchSeries({ limit: 500 }, controller.signal)
      .then((response) => setResearchSeries(response.items))
      .catch(() => { if (!controller.signal.aborted) setResearchSeries([]) })
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
    if (!routeLoadKey) return
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
          setSelectedDefinitionId(next.id || routeDefinitionId)
          setSelectedDefinitionRevision(next.revision || revision)
          setSelectedTemplate('')
          setSelectedNodeId(next.graph.nodes[0]?.id ?? null)
          setSelectedGraphNodeIds(next.graph.nodes[0]?.id ? [next.graph.nodes[0].id] : [])
          setSavedDefinitionSignature(JSON.stringify(definitionForRequest(next)))
          setRun(null); setSeriesPage(null); setPreparedPlan(null)
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
        setSelectedDefinitionId(''); setSelectedDefinitionRevision(1)
        setSelectedTemplate(routeTemplateId)
        setSelectedNodeId(draft.graph.nodes[0]?.id ?? null)
        setSelectedGraphNodeIds(draft.graph.nodes[0]?.id ? [draft.graph.nodes[0].id] : [])
        setSavedDefinitionSignature('')
        setRun(null); setSeriesPage(null); setPreparedPlan(null)
        nodeSequence.current = draft.graph.nodes.length + 1
        setNotice('已从目录实例化模板为独立草稿；保存前不会覆盖模板。')
      })
      .catch((reason) => { if (!controller.signal.aborted) setError(errorText(reason, '目录模板实例化失败。')) })
      .finally(() => { if (!controller.signal.aborted) setLoadingDefinition(false) })
    return () => controller.abort()
  }, [routeDefinitionId, routeLoadKey, routeRevisionRaw, routeTemplateId])

  useEffect(() => {
    const requestId = ++inferenceRequest.current
    const controller = new AbortController()
    setInferenceStatus('checking')
    const timer = window.setTimeout(() => {
      void inferRegimeGraph(definition, controller.signal)
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
  }, [apiSignature])

  useEffect(() => () => { runAbort.current?.abort(); runRequest.current += 1 }, [])

  const editDefinition = (next: RegimeGraphDefinition) => {
    dispatch({ type: 'edit', definition: cloneRegimeGraphDefinition(next) }); setNotice(''); setSeriesPage(null); setPreparedPlan(null)
  }

  const addNode = (schema: RegimeNodeSchema, parameterOverrides: Record<string, unknown> = {}, label?: string) => {
    if (schema.available === false || (schema.status && schema.status !== 'available')) { setError(schema.unavailable_reason || `节点 ${schema.label} 当前不可用。`); return }
    const type = schemaId(schema)
    if (!type) { setError('节点目录返回了缺少 id/type 的节点，无法添加。'); return }
    const id = newNodeId(type, nodeSequence.current++)
    const previous = selectedNode
    const firstInput = schema.inputs?.[0]
    const previousSchema = previous ? schemas.find((item) => schemaId(item) === previous.type || item.type === previous.type) : undefined
    const firstOutput = previousSchema?.outputs?.[0]
    const inputs = firstInput && previous && firstOutput ? { [firstInput.id]: { node_id: previous.id, port: firstOutput.id } } : {}
    const typeVersion = schema.type_version ?? schema.version
    const node: RegimeGraphNode = { id, type, ...(typeVersion != null ? { type_version: typeVersion } : {}), label, parameters: { ...defaultParameters(schema), ...parameterOverrides }, inputs, position: { x: definition.graph.nodes.length * 240, y: 0 } }
    editDefinition({ ...definition, graph: { ...definition.graph, nodes: [...definition.graph.nodes, node] } })
    setSelectedNodeId(id)
  }

  const handleCanvasChanges = (changes: RegimeCanvasNodeChange[]) => {
    let next = cloneRegimeGraphDefinition(definition)
    changes.forEach((change) => {
      if (change.type === 'select') setSelectedNodeId(change.id)
      if (change.type === 'position') next.graph.nodes = next.graph.nodes.map((node) => node.id === change.id ? { ...node, position: change.position } : node)
      if (change.type === 'remove') {
        next.graph.nodes = next.graph.nodes.filter((node) => node.id !== change.id).map((node) => ({ ...node, inputs: Object.fromEntries(Object.entries(node.inputs).filter(([, input]) => input.node_id !== change.id)) }))
        next.graph.outputs = Object.fromEntries(Object.entries(next.graph.outputs).filter(([, output]) => output?.node_id !== change.id))
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
    setSelectedNodeId(connection.target)
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
    editDefinition({ ...definition, graph: { ...definition.graph, outputs } })
  }

  const loadTemplate = async () => {
    if (!selectedTemplate) return
    setLoadingTemplate(true); setError(''); setNotice('')
    try {
      const next = await instantiateRegimeTemplate(selectedTemplate)
      dispatch({ type: 'reset', definition: next }); setSelectedNodeId(next.graph.nodes[0]?.id ?? null); setRun(null); setSeriesPage(null); setSavedDefinitionSignature('')
      setSelectedGraphNodeIds(next.graph.nodes[0]?.id ? [next.graph.nodes[0].id] : [])
      setNotice('模板已实例化为独立草稿；每个节点均可继续修改。')
    } catch (reason) { setError(errorText(reason, '模板载入失败。')) } finally { setLoadingTemplate(false) }
  }

  const createBlank = () => {
    if ((definition.graph.nodes.length || timeline.past.length) && !window.confirm('当前草稿将被新的空白计算图替换，是否继续？')) return
    const next = createBlankRegimeDefinition()
    dispatch({ type: 'reset', definition: next }); setSelectedNodeId(null); setSelectedGraphNodeIds([]); setSelectedTemplate(''); setRun(null); setSeriesPage(null); setSavedDefinitionSignature(''); setError(''); setNotice('已创建空白计算图。')
  }

  const selectSavedDefinition = (id: string) => {
    setSelectedDefinitionId(id)
    const selected = savedDefinitions.find((item) => item.id === id)
    setSelectedDefinitionRevision(selected?.revision || 1)
  }

  const loadSavedDefinition = async () => {
    if (!selectedDefinitionId) return
    setLoadingDefinition(true); setError(''); setNotice('')
    try {
      const next = await getRegimeGraphDefinition(selectedDefinitionId, selectedDefinitionRevision)
      dispatch({ type: 'reset', definition: next })
      setSelectedNodeId(next.graph.nodes[0]?.id ?? null)
      setSelectedGraphNodeIds(next.graph.nodes[0]?.id ? [next.graph.nodes[0].id] : [])
      setSavedDefinitionSignature(JSON.stringify(definitionForRequest(next)))
      setRun(null); setSeriesPage(null)
      setNotice(`已载入 ${next.name} · r${next.revision}。`)
    } catch (reason) { setError(errorText(reason, '定义版本载入失败。')) } finally { setLoadingDefinition(false) }
  }

  const saveDefinition = async (saveAsNew: boolean) => {
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
      setRun(null); setSeriesPage(null)
      setNotice(saveAsNew || !definition.id ? `已保存新定义 · r${saved.revision}。` : `已保存修订 · r${saved.revision}。`)
    } catch (reason) { setError(errorText(reason, '定义保存失败。')) } finally { setSavingDefinition(false) }
  }

  const exportDefinition = () => {
    const blob = new Blob([JSON.stringify(definitionForRequest(definition), null, 2)], { type: 'application/json;charset=utf-8' })
    const url = URL.createObjectURL(blob)
    const anchor = document.createElement('a'); anchor.href = url; anchor.download = `${definition.name || 'historical-regime'}.json`; anchor.click(); URL.revokeObjectURL(url)
  }

  const pollRun = async (runId: string, requestId: number, controller: AbortController) => {
    while (!controller.signal.aborted && requestId === runRequest.current) {
      const next = await getRegimePreviewRun(runId, controller.signal)
      if (requestId !== runRequest.current) return
      setRun(next)
      if (['completed', 'failed', 'cancelled'].includes(next.status)) {
        setNotice(next.status === 'completed' ? '试算完成。可选择任意节点查看真实序列。' : '')
        return
      }
      await new Promise<void>((resolve) => window.setTimeout(resolve, 650))
    }
  }

  const runPreview = async () => {
    if (!inference?.valid || inferenceStatus !== 'ready') return
    const requestId = ++runRequest.current
    runAbort.current?.abort()
    const controller = new AbortController(); runAbort.current = controller
    setError(''); setNotice(''); setSeriesPage(null)
    setRun({ id: '', status: 'preparing', stage: '准备固定签名计划', progress: 0, message: '正在校验并读取已预热内核。' })
    try {
      const prepared = await prepareRegimeGraph(definition, controller.signal)
      setPreparedPlan(prepared)
      if (requestId !== runRequest.current) return
      setRun({ id: '', status: 'preparing', stage: '提交试算任务', progress: 0.05, message: '固定签名计划已验证。' })
      const started = await startRegimePreviewRun(definition, { compileToken: prepared.compile_token, mode, asOf: asOf || undefined, ttlSeconds: 1800 }, controller.signal)
      if (requestId !== runRequest.current) return
      setRun(started); setRunDefinitionSignature(apiSignature)
      await pollRun(started.id, requestId, controller)
    } catch (reason) {
      if (controller.signal.aborted || requestId !== runRequest.current) return
      setRun((current) => ({ id: current?.id || '', status: 'failed', stage: 'failed', progress: current?.progress ?? 0, error: { message: errorText(reason, '历史情景试算失败。') } }))
      setError(errorText(reason, '历史情景试算失败。'))
    }
  }

  const cancelRun = async () => {
    const activeRun = run
    runRequest.current += 1
    runAbort.current?.abort()
    try { if (activeRun?.id) await cancelRegimePreviewRun(activeRun.id) } catch (reason) { setError(errorText(reason, '取消试算失败。')); return }
    setRun((current) => current ? { ...current, status: 'cancelled', stage: 'cancelled', message: '试算已取消。' } : current)
  }

  const loadSeries = async () => {
    if (!run?.id || run.status !== 'completed') return
    setLoadingSeries(true); setError('')
    try { setSeriesPage(await getRegimePreviewSeries(run.id, { nodeId: previewNodeId || undefined, limit: 500 })) } catch (reason) { setError(errorText(reason, '节点序列读取失败。')) } finally { setLoadingSeries(false) }
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
    setRun(null); setSeriesPage(null); setPreparedPlan(null)
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

  if (dataLabOpen) return <ResearchDataLab embedded boundSeriesIds={boundSeriesIds} onBindSeries={bindSeries} onClose={() => setDataLabOpen(false)} />

  return (
    <div className="space-y-4" data-testid="historical-regime-workbench">
      <CommandBar definition={definition} templates={templates} selectedTemplate={selectedTemplate} canUndo={timeline.past.length > 0} canRedo={timeline.future.length > 0} mode={mode} asOf={asOf} inferenceStatus={inferenceStatus} inference={inference} loadingTemplate={loadingTemplate} run={run} runStale={runStale} onName={(name) => editDefinition({ ...definition, name })} onTemplate={setSelectedTemplate} onLoadTemplate={() => void loadTemplate()} onBlank={createBlank} onUndo={() => dispatch({ type: 'undo' })} onRedo={() => dispatch({ type: 'redo' })} onMode={setMode} onAsOf={setAsOf} onRun={() => void runPreview()} onCancel={() => void cancelRun()} onExport={exportDefinition} onExit={onExit} />
      <VersionBar definitions={savedDefinitions} selectedId={selectedDefinitionId} selectedRevision={selectedDefinitionRevision} current={definition} dirty={definitionDirty} valid={Boolean(inference?.valid && inferenceStatus === 'ready')} busy={savingDefinition || loadingDefinition} onSelect={selectSavedDefinition} onRevision={setSelectedDefinitionRevision} onLoad={() => void loadSavedDefinition()} onSave={() => void saveDefinition(false)} onSaveAs={() => void saveDefinition(true)} />
      {error ? <p role="alert" className="rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-900">{error}</p> : null}
      {notice ? <p role="status" className="rounded-xl border border-indigo-200 bg-indigo-50 px-4 py-3 text-sm text-indigo-950">{notice}</p> : null}
      {loadingCatalog ? <p role="status" className="rounded-xl border border-slate-200 bg-white px-4 py-3 text-sm font-semibold text-slate-600">正在加载节点目录与模板…</p> : null}
      {loadingDefinition ? <p role="status" className="rounded-xl border border-slate-200 bg-white px-4 py-3 text-sm font-semibold text-slate-600">正在加载目录指定的精确版本或模板…</p> : null}
      <InferenceIssues inference={inference} status={inferenceStatus} onNode={setSelectedNodeId} />
      <div className="grid gap-4 xl:grid-cols-[280px_minmax(0,1fr)_340px]">
        <ResourceLibrary schemas={schemas} onAdd={addNode} onOpenDataLab={() => setDataLabOpen(true)} />
        <RegimeGraphCanvas nodes={definition.graph.nodes} edges={edges} schemas={schemas} selectedNodeId={selectedNodeId} onNodesChange={handleCanvasChanges} onConnect={connectCanvasNodes} onDuplicate={duplicateNodes} onSelectionChange={setSelectedGraphNodeIds} />
        <div className="space-y-4"><RegimeNodeInspector node={selectedNode} schema={selectedSchema} nodes={definition.graph.nodes} schemas={schemas} outputs={definition.graph.outputs} inference={inference} preparedPlan={preparedPlan} researchSeries={researchSeries} onPatchNode={patchSelectedNode} onConnect={connectSelectedNode} onSetOutput={setSelectedOutput} onRemove={() => { if (selectedNode) handleCanvasChanges([{ type: 'remove', id: selectedNode.id }]) }} /><StateDictionary definition={definition} onChange={editDefinition} /></div>
      </div>
      <RegimeValidationPanel definition={definition} onChange={editDefinition} />
      <RegimeGraphAssetsPanel definition={definition} selectedNodeIds={selectedGraphNodeIds} valid={Boolean(inference?.valid && inferenceStatus === 'ready')} onLoadDefinition={loadAssetDefinition} onInsertGraph={insertAssetGraph} onError={setError} onNotice={setNotice} />
      <RegimeExperimentPanel definition={definition} schemas={schemas} dirty={definitionDirty} valid={Boolean(inference?.valid && inferenceStatus === 'ready')} mode={mode} asOf={asOf} preparedPlan={preparedPlan} onPrepared={setPreparedPlan} onError={setError} onNotice={setNotice} />
      <RegimeResultDock run={run} page={seriesPage} nodes={definition.graph.nodes} previewNodeId={previewNodeId} loadingSeries={loadingSeries} height={resultHeight} collapsed={resultCollapsed} onPreviewNode={setPreviewNodeId} onLoadSeries={() => void loadSeries()} onHeight={setResultHeight} onCollapsed={setResultCollapsed} />
      <RegimeLifecyclePanel definition={definition} dirty={definitionDirty} valid={Boolean(inference?.valid && inferenceStatus === 'ready')} mode={mode} asOf={asOf} onError={setError} onNotice={setNotice} />
    </div>
  )
}

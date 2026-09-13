import { useMemo, useRef, useState, type KeyboardEvent } from 'react'
import type { SourceCatalog } from '../../services/dataSources'
import { blankStep, stepLabels, planEtlDependencies, type EtlDefinition, type EtlKind } from '../../services/etl'
import GraphCanvas from '../computation-graph/GraphCanvas'
import { graphOrder, layoutGraph } from '../computation-graph/graph'
import type { CanvasConnection, CanvasNodeChange } from '../computation-graph/types'
import { asGraph, connectEtl, connectionProblem, duplicateEtlNodes, etlEdges, etlPositions, removeEtlEdges, removeEtlNodes } from './etlGraphAdapter'
import { buttonClass, inputClass } from './EditorFields'
import EtlParameterDefinitions from './EtlParameterDefinitions'
import EtlNodeInspector from './EtlNodeInspector'

const testIds = { canvas: 'etl-graph-canvas', mobile: 'etl-graph-mobile-list', flow: 'etl-graph-desktop-flow', minimap: 'etl-canvas-minimap' }
const portColor = (type?: string) => type === 'control' ? '#94a3b8' : '#4f46e5'

export default function EtlWorkflowEditor({ definition, catalog, onChange, readOnly = false, canUndo = false, canRedo = false, onUndo, onRedo }: {
  definition: EtlDefinition; catalog: SourceCatalog; onChange: (value: EtlDefinition) => void
  readOnly?: boolean; canUndo?: boolean; canRedo?: boolean; onUndo?: () => void; onRedo?: () => void
}) {
  const graph = useMemo(() => asGraph(definition), [definition])
  const current = useRef(graph); current.current = graph
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [inspectorOpen, setInspectorOpen] = useState(false)
  const [libraryOpen, setLibraryOpen] = useState(!definition.steps.length)
  const [query, setQuery] = useState('')
  const [error, setError] = useState('')
  const [viewKey, setViewKey] = useState(0)
  const [planning, setPlanning] = useState(false)
  const locked = readOnly || !catalog.editing_enabled
  const schemas = catalog.graph_schemas ?? []
  const edges = useMemo(() => etlEdges(graph), [graph])
  const positions = useMemo(() => etlPositions(graph), [graph])
  const order = useMemo(() => { try { return graphOrder(graph.steps.map(s => s.id), edges) } catch { return graph.steps.map(s => s.id) } }, [graph, edges])
  const selected = graph.steps.find(s => s.id === selectedId)
  const nodes = useMemo(() => graph.steps.map(step => {
    const configured = step.kind === 'download' ? Boolean(step.interface_id && catalog.interfaces.find(r => r.config.id === step.interface_id)?.validation?.ready)
      : step.kind === 'task' ? Boolean(step.task_id) : step.kind === 'resolve' ? Boolean(step.table_id && step.inputs.length) : Boolean(step.inputs.length)
    return { id: step.id, type: step.kind, label: step.name, inputs: Object.fromEntries(step.inputs.map(id => [id, id])), position: positions[step.id], ready: configured, statusLabel: `执行 ${order.indexOf(step.id) + 1} · ${configured ? '已配置，待校验' : '待完善配置'}` }
  }), [graph, catalog, positions, order])
  const commit = (next: EtlDefinition) => {
    if (locked) return
    current.current = next; setError(''); onChange(next)
  }
  const action = (operation: () => EtlDefinition) => { if (!locked) { try { commit(operation()) } catch (reason) { setError(reason instanceof Error ? reason.message : '无法修改计算图。') } } }
  const select = (id: string) => { setSelectedId(id); setInspectorOpen(true) }
  const changes = (items: CanvasNodeChange[]) => {
    const selection = items.filter((item): item is Extract<CanvasNodeChange, { type: 'select' }> => item.type === 'select')
    if (selection.length) select(selection[selection.length - 1].id)
    if (locked) return
    const removed = items.filter(item => item.type === 'remove').map(item => item.id)
    const moved = items.filter((item): item is Extract<CanvasNodeChange, { type: 'position' }> => item.type === 'position')
    if (!removed.length && !moved.length) return
    let next = removed.length ? removeEtlNodes(current.current, removed) : current.current
    if (moved.length) next = { ...next, canvas: { version: 1, ...next.canvas, positions: { ...etlPositions(next), ...Object.fromEntries(moved.filter(item => next.steps.some(s => s.id === item.id)).map(item => [item.id, item.position])) } } }
    commit(next)
  }
  const connect = (edge: CanvasConnection) => action(() => connectEtl(current.current, edge, schemas))
  const planDependencies = async () => {
    const original = current.current
    setPlanning(true); setError('')
    try {
      const result = await planEtlDependencies(original)
      if (current.current !== original) { setError('梳理期间流程已修改，请重新梳理；未覆盖你的编辑。'); return }
      commit(result.definition)
    } catch (reason) { setError(reason instanceof Error ? reason.message : '依赖梳理失败。') }
    finally { setPlanning(false) }
  }
  const add = (kind: EtlKind) => {
    if (locked || graph.steps.length >= 40) return
    const step = blankStep(kind)
    if (kind === 'download') step.source_id = catalog.sources[0]?.config.id
    const inputType = schemas.find(s => s.id === kind)?.inputs.find(p => p.id === 'data')?.value_type
    const selectedOutput = schemas.find(s => s.id === selected?.kind)?.outputs.find(p => p.id === 'data')?.value_type
    if (selected && inputType && selectedOutput === inputType) step.inputs = [selected.id]
    if (kind === 'resolve' && selected?.kind === 'map') {
      const upstream = graph.steps.find(s => s.id === selected.inputs[0])
      step.table_id = catalog.interfaces.find(r => r.config.id === upstream?.interface_id)?.config.mappings.find(m => m.enabled)?.target_table
    }
    const origin = selected ? positions[selected.id] : { x: 0, y: 0 }
    commit({ ...graph, steps: [...graph.steps, step], canvas: { version: 1, ...graph.canvas, positions: { ...positions, [step.id]: { x: origin.x + (selected ? 300 : 0), y: origin.y + (selected ? 0 : graph.steps.length * 170) } } } })
    select(step.id)
  }
  const keyboard = (event: KeyboardEvent<HTMLElement>) => {
    if (locked || !(event.metaKey || event.ctrlKey) || (event.target as HTMLElement).closest('input,textarea,select,[contenteditable="true"]')) return
    if (event.key.toLowerCase() === 'z') { event.preventDefault(); if (event.shiftKey) onRedo?.(); else onUndo?.() }
    if (event.key.toLowerCase() === 'y') { event.preventDefault(); onRedo?.() }
  }
  return <section className="min-w-0 space-y-4" aria-label="ETL 流程编辑器" onKeyDown={keyboard}>
    <div className="flex flex-wrap gap-2"><button type="button" className={buttonClass} aria-expanded={libraryOpen} onClick={() => setLibraryOpen(v => !v)}>节点库</button><button type="button" className={buttonClass} disabled={locked || !canUndo} onClick={onUndo}>撤销</button><button type="button" className={buttonClass} disabled={locked || !canRedo} onClick={onRedo}>重做</button><button type="button" className={buttonClass} disabled={locked || !graph.steps.length} onClick={() => { commit({ ...graph, canvas: { version: 1, positions: layoutGraph(graph.steps.map(s => s.id), edges) } }); setViewKey(v => v + 1) }}>自动布局</button>{selected && !inspectorOpen ? <button type="button" className={buttonClass} onClick={() => setInspectorOpen(true)}>打开节点设置</button> : null}</div>
    {libraryOpen ? <section aria-label="ETL 节点库" className="space-y-3 rounded-xl border border-slate-200 bg-slate-50 p-3"><label className="text-xs">查找节点类型<input aria-label="查找节点类型" className={inputClass} value={query} onChange={e => setQuery(e.target.value)} placeholder="下载、映射、取值、快照、数据集" /></label><div className="flex flex-wrap gap-2">{schemas.filter(s => `${s.label} ${s.category_label}`.includes(query)).map(schema => <button type="button" className={buttonClass} key={schema.id} disabled={locked || graph.steps.length >= 40} onClick={() => add(schema.id as EtlKind)}>＋{schema.label}</button>)}</div>{!schemas.length ? <p role="alert" className="text-xs text-amber-900">节点合同未加载，请更新后端并重新加载页面。</p> : null}<p className="text-xs text-slate-600">先选择节点，再添加兼容节点时自动连接。其他依赖可拖线或在检查器中选择。</p></section> : null}
    {error ? <p role="alert" className="rounded-lg bg-rose-50 p-3 text-sm text-rose-800">{error}</p> : null}
    {graph.steps.length > 0 && graph.steps.every(s => s.kind === 'task') ? <div className="space-y-2 rounded-lg bg-accent-50 p-3"><button type="button" className={buttonClass} disabled={locked || planning} onClick={planDependencies}>{planning ? '正在梳理依赖…' : '按数据需求重新梳理依赖'}</button><p className="text-xs text-slate-600">根据服务端任务合同替换当前草稿的连线：保留真实数据依赖，其余改为执行顺序。可撤销；不会修改历史运行或自动下载。</p></div> : null}
    <p className="text-xs leading-5 text-slate-600">实线：必须依赖上游成功，失败则阻断。虚线：只等待上游结束，即使失败也继续。同层按顺序调度，节点内部仍可并发下载。取消、总超时或存储安全错误会停止整个流程。</p>
    <div className="relative min-w-0">
      <GraphCanvas key={viewKey} nodes={nodes} edges={edges} schemas={schemas} selectedNodeId={selected?.id ?? null}
        ariaLabel="ETL 可编辑计算图画布" testIds={testIds} readOnly={locked} portColor={portColor} minZoom={0.05}
        flowClassName="hidden h-[620px] min-w-0 overflow-hidden rounded-xl border border-slate-200 bg-white md:block"
        description="拖动端口连接依赖；点击节点打开设置。运行模式与参数在上方选择。"
        onNodesChange={changes} onConnect={connect} validateConnection={edge => !connectionProblem(current.current, edge, schemas)}
        onDuplicate={ids => action(() => duplicateEtlNodes(current.current, ids))}
        onEdgesRemove={ids => action(() => removeEtlEdges(current.current, ids))}
        viewport={graph.canvas?.viewport ?? undefined} onViewportCommit={viewport => { if (!locked) commit({ ...current.current, canvas: { version: 1, ...current.current.canvas, positions: etlPositions(current.current), viewport } }) }} />
      {selected && inspectorOpen ? <div className="mt-3 max-h-[640px] overflow-y-auto xl:absolute xl:right-5 xl:top-20 xl:mt-0 xl:w-[360px]">
        <EtlNodeInspector step={selected} definition={graph} catalog={catalog} schemas={schemas} readOnly={locked}
          onPatch={step => action(() => ({ ...current.current, steps: current.current.steps.map(s => s.id === step.id ? { ...s, ...step, after: step.after ?? s.after } : s) }))}
          onConnect={connect} onDisconnect={id => action(() => removeEtlEdges(current.current, [id]))}
          onRemove={() => action(() => removeEtlNodes(current.current, [selected.id]))} onClose={() => setInspectorOpen(false)} />
      </div> : null}
    </div>
    <details className="rounded-xl border border-slate-200 p-3"><summary className="cursor-pointer text-sm font-semibold">执行顺序 · {order.length} 个节点</summary><div className="mt-3 grid gap-2 sm:grid-cols-2 lg:grid-cols-3">{order.map((id, i) => <button type="button" className={`${buttonClass} text-left`} key={id} onClick={() => select(id)}>{i + 1}. {graph.steps.find(s => s.id === id)?.name}</button>)}</div></details>
    <details className="rounded-xl border border-slate-200 p-3"><summary className="cursor-pointer text-sm font-semibold">流程名称、说明与运行参数</summary><fieldset disabled={locked} className="mt-3 space-y-3"><div className="grid gap-3 sm:grid-cols-[1fr_200px]"><label className="text-sm font-semibold">流程名称<input className={inputClass} value={definition.name} required onChange={e => commit({ ...graph, name: e.target.value })} /></label><label className="text-sm font-semibold">最长运行时间（秒）<input className={inputClass} type="number" min={10} max={86400} value={definition.max_runtime_seconds} onChange={e => commit({ ...graph, max_runtime_seconds: Number(e.target.value) })} /></label></div><label className="block text-sm">流程说明<textarea className={inputClass} value={definition.description} onChange={e => commit({ ...graph, description: e.target.value })} /></label><EtlParameterDefinitions value={definition.parameters ?? []} onChange={parameters => commit({ ...graph, parameters })} /></fieldset></details>
  </section>
}

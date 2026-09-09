import { useEffect, useMemo, useRef, useState, type KeyboardEvent } from 'react'
import {
  Background, Controls, Handle, MiniMap, Position, ReactFlow, applyNodeChanges, useUpdateNodeInternals,
  type Connection, type Edge, type Node, type NodeChange, type NodeProps, type EdgeChange, type ReactFlowInstance,
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import { nodeAppearance, type NodeAppearance } from './nodeAppearance'
import { systemText, useI18n } from '../../i18n/runtime'
import type { CanvasNode, CanvasEdge, CanvasNodeChange, GraphCanvasProps, GraphNodeSchema, GraphPort } from './types'

const defaultPortLabel = (port: GraphPort) => port.label || port.name || port.id
const defaultTypeLabel = (type?: string, label?: string) => label || type || '数据'
const defaultColor = () => '#4f46e5'
const defaultTestIds = { canvas: 'computation-graph-canvas', mobile: 'computation-graph-mobile-list', flow: 'computation-graph-desktop-flow', minimap: 'computation-graph-minimap' }
type FlowData = {
  label: string; schema?: GraphNodeSchema; ready: boolean; statusLabel?: string; appearance?: NodeAppearance
  portLabel: typeof defaultPortLabel; typeLabel: typeof defaultTypeLabel; portColor: (type?: string) => string
}
type FlowNode = Node<FlowData, 'computation'>

function ComputationNode({ id, data, selected }: NodeProps<FlowNode>) {
  const { s } = useI18n()
  const inputs = data.schema?.inputs ?? []
  const outputs = data.schema?.outputs ?? []
  const updateNodeInternals = useUpdateNodeInternals()
  const signature = [...inputs.map(p => `in:${p.id}`), ...outputs.map(p => `out:${p.id}`)].join('|')
  useEffect(() => {
    const frame = window.requestAnimationFrame(() => updateNodeInternals(id))
    return () => window.cancelAnimationFrame(frame)
  }, [id, signature, updateNodeInternals])
  const { portLabel, typeLabel, portColor } = data
  const appearance = nodeAppearance(data.appearance)
  if (data.appearance === 'constant') return <div data-node-kind="constant" className={`relative min-w-[132px] border px-4 py-3 shadow-sm ${appearance!.className} ${selected ? '!border-indigo-600 ring-2 ring-indigo-200' : !data.ready ? '!border-rose-500 ring-1 ring-rose-200' : ''}`}>
    <p className="text-xs font-bold text-amber-950"><span aria-hidden="true" className="mr-1.5">●</span>{data.label}</p>
    <p className="mt-1 text-[10px] text-amber-800">{data.schema?.category_label || s('graph.fixedNumber')}</p>
    {outputs.map(port => <Handle key={port.id} id={port.id} type="source" position={Position.Right} style={{ width: 10, height: 10, background: appearance!.color, border: '2px solid white' }} title={`${portLabel(port)}：${typeLabel(port.value_type, port.type_label)}`} />)}
  </div>
  return <div data-node-kind={data.appearance} className={`min-w-[210px] border px-3 py-3 shadow-sm ${appearance?.className || 'rounded-xl border-slate-300 bg-white'} ${selected ? '!border-indigo-600 ring-2 ring-indigo-200' : !data.ready ? '!border-rose-500 ring-1 ring-rose-200' : ''}`}>
    <p className="text-xs font-bold text-slate-900">{appearance && <span aria-hidden="true" className="mr-1.5" style={{ color: appearance.color }}>{appearance.glyph}</span>}{data.label}</p>
    <p className="mt-1 text-[9px] font-semibold tracking-wide text-slate-400">{data.schema?.category_label || s('graph.calculationNode', {}, '计算节点')}</p>
    {data.statusLabel ? <p className="mt-1 text-[10px] text-slate-500">{data.statusLabel}</p> : null}
    <div className="mt-3 grid grid-cols-2 gap-x-5 gap-y-1 border-t border-slate-100 pt-2 text-[9px] text-slate-500">
      <div className="space-y-1.5">{inputs.map(port => <div key={port.id} className="relative pl-1"><Handle id={port.id} type="target" position={Position.Left} style={{ position: 'absolute', left: -17, top: '50%', width: 9, height: 9, background: portColor(port.value_type), border: '2px solid white' }} /><span className="block truncate" title={`${portLabel(port)}：${typeLabel(port.value_type, port.type_label)}`}>{portLabel(port)}{port.required ? ' *' : ''}</span></div>)}</div>
      <div className="space-y-1.5 text-right">{outputs.map(port => <div key={port.id} className="relative pr-1"><span className="block truncate" title={`${portLabel(port)}：${typeLabel(port.value_type, port.type_label)}`}>{portLabel(port)}</span><Handle id={port.id} type="source" position={Position.Right} style={{ position: 'absolute', right: -17, top: '50%', width: 9, height: 9, background: portColor(port.value_type), border: '2px solid white' }} /></div>)}</div>
    </div>
  </div>
}
const nodeTypes = { computation: ComputationNode }

function NodeCard({ node, index, count, selected, incoming, schema, onNodesChange, portLabel, readOnly }: {
  node: CanvasNode; index: number; count: number; selected: boolean; incoming: CanvasEdge[]; schema?: GraphNodeSchema
  onNodesChange: GraphCanvasProps['onNodesChange']; portLabel: typeof defaultPortLabel; readOnly: boolean
}) {
  const { s } = useI18n()
  const label = node.label || schema?.label || s('graph.calculationNode', {}, '计算节点')
  const move = (delta: number) => onNodesChange([{ type: 'position', id: node.id, position: { x: (index + delta) * 240, y: node.position?.y ?? 0 } }])
  const appearance = nodeAppearance(node.appearance)
  return <article className={`group relative min-w-0 border p-3 text-left shadow-sm transition ${appearance?.className || 'rounded-xl border-slate-200 bg-white'} ${selected ? '!border-indigo-600 ring-2 ring-indigo-200' : node.ready === false ? '!border-rose-500 ring-1 ring-rose-200' : ''}`} data-node-id={node.id} data-node-kind={node.appearance}>
    <button type="button" onClick={() => onNodesChange([{ type: 'select', id: node.id }])} className="block min-h-12 w-full text-left focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500">
      <span className="flex items-start justify-between gap-2"><span className="min-w-0"><span className="block truncate text-sm font-bold text-slate-900">{appearance && <span aria-hidden="true" className="mr-1.5" style={{ color: appearance.color }}>{appearance.glyph}</span>}{node.label || schema?.label || s('graph.unnamedNode', {}, '未命名节点')}</span><span className="mt-1 block truncate text-[10px] font-semibold tracking-wide text-slate-400">{schema?.category_label || s('graph.calculationNode', {}, '计算节点')}</span></span><span className={`mt-1 h-2.5 w-2.5 shrink-0 rounded-full ${incoming.length || !schema?.inputs.length ? 'bg-emerald-500' : 'bg-amber-400'}`} aria-label={s(incoming.length || !schema?.inputs.length ? 'graph.inputConnected' : 'graph.waitingInput')} /></span>
    </button>
    {node.statusLabel ? <p className="mt-1 text-xs text-slate-500">{node.statusLabel}</p> : null}
    <div className="mt-3 space-y-1 border-t border-slate-100 pt-2 text-[10px] text-slate-500">
      {incoming.length ? incoming.map(edge => <p key={edge.id} className="truncate"><span aria-hidden="true">←</span> {portLabel(schema?.inputs.find(port => port.id === edge.targetPort) || { id: edge.targetPort })} · {s('graph.incoming')}</p>) : <p>{s(schema?.inputs.length ? 'graph.noUpstream' : 'graph.sourceNode')}</p>}
      <p>{schema?.outputs?.map(portLabel).join('、') || s('graph.outputUndefined', {}, '输出待元数据定义')}</p>
    </div>
    <div className="mt-3 flex items-center justify-end gap-1 border-t border-slate-100 pt-2">
      <button type="button" aria-label={s('graph.moveLeft', { name: label })} disabled={readOnly || index === 0} onClick={() => move(-1)} className="min-h-8 rounded-md px-2 text-xs font-bold text-slate-500 hover:bg-slate-100 disabled:opacity-30">←</button>
      <button type="button" aria-label={s('graph.moveRight', { name: label })} disabled={readOnly || index === count - 1} onClick={() => move(1)} className="min-h-8 rounded-md px-2 text-xs font-bold text-slate-500 hover:bg-slate-100 disabled:opacity-30">→</button>
      <button type="button" aria-label={s('graph.deleteNamed', { name: label })} disabled={readOnly} onClick={() => onNodesChange([{ type: 'remove', id: node.id }])} className="min-h-8 rounded-md px-2 text-xs font-bold text-rose-600 hover:bg-rose-50 disabled:opacity-30">{s('common.delete')}</button>
    </div>
  </article>
}

/** Shared interaction implementation extracted from RegimeGraphCanvas. */
export default function GraphCanvas({ nodes, edges, schemas, selectedNodeId, onNodesChange, onConnect, onDuplicate, onSelectionChange,
  onEdgesRemove, validateConnection, portLabel = defaultPortLabel, typeLabel = defaultTypeLabel, portColor = defaultColor,
  readOnly = false, help, toolbar, ariaLabel = systemText('graph.editableAria', {}, '可编辑计算图画布'), description = systemText('graph.defaultInstructions', {}, '点击节点编辑参数；拖动端口连接依赖。'),
  emptyDescription = systemText('graph.emptyHint', {}, '从节点库添加节点，或载入一个可修改的流程。'), testIds = defaultTestIds,
  flowClassName = 'hidden h-[520px] min-w-[680px] overflow-hidden rounded-xl border border-slate-200 bg-white md:block',
  minZoom = 0.25, viewport, onViewportCommit, isPortCompatible, onResourceDrop, onNodeActivate,
}: GraphCanvasProps) {
  const { s, version: translationVersion } = useI18n()
  const flowInstance = useRef<ReactFlowInstance<FlowNode, Edge> | null>(null)
  const [selectedIds, setSelectedIds] = useState<string[]>(selectedNodeId ? [selectedNodeId] : [])
  const [copiedIds, setCopiedIds] = useState<string[]>([])
  const [selectedEdges, setSelectedEdges] = useState<Set<string>>(new Set())
  const nodeIdentity = nodes.map(node => node.id).join('|')
  useEffect(() => {
    const valid = new Set(nodes.map(node => node.id))
    setSelectedIds(current => { const kept = current.filter(id => valid.has(id)); return selectedNodeId && valid.has(selectedNodeId) && !kept.includes(selectedNodeId) ? [selectedNodeId] : kept })
    setCopiedIds(current => current.filter(id => valid.has(id)))
  }, [nodeIdentity, selectedNodeId])
  useEffect(() => { onSelectionChange?.(selectedIds) }, [onSelectionChange, selectedIds])
  const sortedNodes = useMemo(() => [...nodes].sort((left, right) => (left.position?.x ?? nodes.indexOf(left) * 240) - (right.position?.x ?? nodes.indexOf(right) * 240)), [nodes])
  const schemaMap = useMemo(() => new Map(schemas.flatMap(schema => [[schema.id, schema] as const, ...(schema.type ? [[schema.type, schema] as const] : [])])), [schemas])
  const graphFlowNodes = useMemo<FlowNode[]>(() => nodes.map((node, index) => {
    const schema = schemaMap.get(node.type)
    const ready = node.ready ?? (!schema?.inputs?.length || Object.keys(node.inputs || {}).length > 0)
    return { id: node.id, type: 'computation', position: node.position ?? { x: (index % 3) * 260, y: Math.floor(index / 3) * 150 }, initialWidth: node.appearance === 'constant' ? 132 : 210,
      initialHeight: node.appearance === 'constant' ? 66 : Math.max(82, 72 + Math.max(schema?.inputs?.length ?? 0, schema?.outputs?.length ?? 0) * 18),
      data: { label: node.label || schema?.label || s('graph.calculationNode', {}, '计算节点'), schema, ready, statusLabel: node.statusLabel, appearance: node.appearance, portLabel, typeLabel, portColor } }
  }), [nodes, schemaMap, portLabel, typeLabel, portColor, translationVersion])
  const [flowNodes, setFlowNodes] = useState<FlowNode[]>(graphFlowNodes)
  useEffect(() => {
    setFlowNodes(current => { const byId = new Map(current.map(node => [node.id, node])); return graphFlowNodes.map(node => ({ ...node, measured: byId.get(node.id)?.measured, selected: byId.get(node.id)?.selected ?? false })) })
  }, [graphFlowNodes])
  useEffect(() => { const selected = new Set(selectedIds); setFlowNodes(current => current.map(node => node.selected === selected.has(node.id) ? node : { ...node, selected: selected.has(node.id) })) }, [selectedIds])
  const flowEdges = useMemo<Edge[]>(() => edges.map(edge => {
    const source = nodes.find(node => node.id === edge.source)
    const target = nodes.find(node => node.id === edge.target)
    const targetPort = target ? schemaMap.get(target.type)?.inputs.find(port => port.id === edge.targetPort) : undefined
    return { id: edge.id, source: edge.source, sourceHandle: edge.sourcePort, target: edge.target, targetHandle: edge.targetPort,
      label: edge.label ?? (targetPort ? portLabel(targetPort) : s('graph.inputConnection', {}, '输入连接')), animated: false, selected: selectedEdges.has(edge.id),
      style: { stroke: edge.kind === 'control' ? '#94a3b8' : nodeAppearance(source?.appearance)?.color || '#6366f1', strokeWidth: 1.8, ...(edge.kind === 'control' ? { strokeDasharray: '5 4' } : {}) }, labelStyle: { fill: '#475569', fontSize: 10 } }
  }), [edges, nodes, schemaMap, portLabel, selectedEdges, translationVersion])
  const handleFlowChanges = (changes: NodeChange<FlowNode>[]) => {
    setFlowNodes(current => applyNodeChanges(changes, current))
    const selections = changes.filter((change): change is Extract<NodeChange<FlowNode>, { type: 'select' }> => change.type === 'select')
    if (selections.length) setSelectedIds(current => { const next = new Set(current); selections.forEach(change => { if (change.selected) next.add(change.id); else next.delete(change.id) }); return [...next] })
    const next = changes.flatMap<CanvasNodeChange>(change => {
      if (change.type === 'remove' && !readOnly) return [{ type: 'remove', id: change.id }]
      if (change.type === 'select' && change.selected) return [{ type: 'select', id: change.id }]
      return []
    })
    if (next.length) onNodesChange(next)
  }
  const handleEdgeChanges = (changes: EdgeChange[]) => {
    setSelectedEdges(current => { const next = new Set(current); changes.forEach(c => { if (c.type === 'select') { if (c.selected) next.add(c.id); else next.delete(c.id) } }); return next })
    const removed = changes.flatMap(c => c.type === 'remove' ? [c.id] : [])
    if (!readOnly && removed.length) onEdgesRemove?.(removed)
  }
  const commitPositions = (moved: FlowNode[]) => { if (!readOnly && moved.length) onNodesChange(moved.map(node => ({ type: 'position', id: node.id, position: node.position }))) }
  const handleConnect = (connection: Connection) => {
    if (readOnly || !connection.source || !connection.target || !connection.sourceHandle || !connection.targetHandle) return
    onConnect({ source: connection.source, sourcePort: connection.sourceHandle, target: connection.target, targetPort: connection.targetHandle })
  }
  const isValidConnection = (connection: Connection | Edge) => {
    if (readOnly || !connection.source || !connection.target || !connection.sourceHandle || !connection.targetHandle) return false
    const source = nodes.find(node => node.id === connection.source)
    const target = nodes.find(node => node.id === connection.target)
    const output = source ? schemaMap.get(source.type)?.outputs.find(port => port.id === connection.sourceHandle) : undefined
    const input = target ? schemaMap.get(target.type)?.inputs.find(port => port.id === connection.targetHandle) : undefined
    if (!output || !input) return false
    const candidate = { source: connection.source, sourcePort: connection.sourceHandle, target: connection.target, targetPort: connection.targetHandle }
    if (isPortCompatible ? !isPortCompatible(output, input, candidate) : output.value_type !== input.value_type) return false
    return validateConnection ? validateConnection(candidate) : true
  }
  const copySelection = () => setCopiedIds(selectedIds.length ? [...selectedIds] : selectedNodeId ? [selectedNodeId] : [])
  const pasteSelection = () => { if (!readOnly && copiedIds.length) onDuplicate(copiedIds) }
  const handleKeyboard = (event: KeyboardEvent<HTMLElement>) => {
    if (!(event.metaKey || event.ctrlKey) || (event.target as HTMLElement).closest('input,textarea,select,[contenteditable="true"]')) return
    if (event.key.toLowerCase() === 'c') { event.preventDefault(); copySelection() }
    if (event.key.toLowerCase() === 'v') { event.preventDefault(); pasteSelection() }
  }
  return <section aria-label={ariaLabel} data-testid={testIds.canvas} tabIndex={0} onKeyDown={handleKeyboard} className="relative min-h-[420px] overflow-auto rounded-2xl border border-slate-200 bg-slate-50 p-3 sm:p-4" style={{ backgroundImage: 'radial-gradient(#cbd5e1 1px, transparent 1px)', backgroundSize: '18px 18px' }}>
    <div className="mb-4 flex flex-wrap items-center justify-between gap-2 rounded-xl border border-slate-200 bg-white/95 px-3 py-2 backdrop-blur">
      <div><h3 className="text-sm font-bold text-slate-900">{s('graph.canvas')} {help}</h3><p className="mt-0.5 text-[11px] text-slate-500">{description}</p></div>
      <div className="flex flex-wrap items-center gap-2 text-[10px] font-bold text-slate-500">{toolbar}<button type="button" disabled={!selectedIds.length && !selectedNodeId} onClick={copySelection} className="min-h-8 rounded-lg border border-slate-200 px-2 text-slate-700 disabled:opacity-40">{s('graph.copy')}</button><button type="button" disabled={readOnly || !copiedIds.length} onClick={pasteSelection} className="min-h-8 rounded-lg border border-slate-200 px-2 text-slate-700 disabled:opacity-40">{s('graph.paste')}</button><span className="rounded-full bg-slate-100 px-2 py-1">{s('graph.selected', { count: selectedIds.length || (selectedNodeId ? 1 : 0) })}</span><span className="rounded-full bg-slate-100 px-2 py-1">{s('graph.nodes', { count: nodes.length })}</span><span className="rounded-full bg-slate-100 px-2 py-1">{s('graph.edges', { count: edges.length })}</span></div>
    </div>
    {!nodes.length ? <div className="grid min-h-[300px] place-items-center rounded-xl border border-dashed border-slate-300 bg-white/80 p-8 text-center"><div><p className="text-sm font-bold text-slate-800">{s('graph.empty')}</p><p className="mt-2 max-w-sm text-xs leading-5 text-slate-500">{emptyDescription}</p></div></div> : null}
    <div className="space-y-3 md:hidden" data-testid={testIds.mobile}>{sortedNodes.map((node, index) => <NodeCard key={node.id} node={node} index={index} count={sortedNodes.length} selected={node.id === selectedNodeId} incoming={edges.filter(edge => edge.target === node.id)} schema={schemaMap.get(node.type)} onNodesChange={changes => { onNodesChange(changes); changes.forEach(change => { if (change.type === 'select') onNodeActivate?.(change.id) }) }} portLabel={portLabel} readOnly={readOnly} />)}</div>
    {nodes.length ? <div className={flowClassName} data-testid={testIds.flow}><ReactFlow ariaLabelConfig={{ 'controls.zoomIn.ariaLabel': s('graph.zoomIn'), 'controls.zoomOut.ariaLabel': s('graph.zoomOut'), 'controls.fitView.ariaLabel': s('graph.fitView'), 'controls.ariaLabel': s('graph.controls') }} onInit={instance => { flowInstance.current = instance }} onDragOver={event => { if (!readOnly && onResourceDrop && event.dataTransfer.types.includes('application/x-indicator-resource')) { event.preventDefault(); event.dataTransfer.dropEffect = 'copy' } }} onDrop={event => { const resource = event.dataTransfer.getData('application/x-indicator-resource'); if (!readOnly && onResourceDrop && resource && flowInstance.current) { event.preventDefault(); onResourceDrop(resource, flowInstance.current.screenToFlowPosition({ x: event.clientX, y: event.clientY })) } }} nodeTypes={nodeTypes} nodes={flowNodes} edges={flowEdges} onNodesChange={handleFlowChanges} onEdgesChange={onEdgesRemove ? handleEdgeChanges : undefined} onNodeDragStop={(_event, _node, moved) => commitPositions(moved)} onSelectionDragStop={(_event, moved) => commitPositions(moved)} onConnect={handleConnect} isValidConnection={isValidConnection} onNodeClick={(_event, node) => { onNodesChange([{ type: 'select', id: node.id }]); onNodeActivate?.(node.id) }} nodesDraggable={!readOnly} nodesConnectable={!readOnly} selectionOnDrag selectionKeyCode="Shift" multiSelectionKeyCode={['Meta', 'Control']} fitView={!viewport} defaultViewport={viewport} onMoveEnd={(event, value) => { if (event) onViewportCommit?.(value) }} fitViewOptions={{ padding: 0.22 }} minZoom={minZoom} maxZoom={1.8} deleteKeyCode={readOnly ? null : ['Backspace', 'Delete']}><Background gap={18} size={1} color="#cbd5e1" /><MiniMap data-testid={testIds.minimap} ariaLabel={s('graph.minimap')} pannable zoomable nodeBorderRadius={6} nodeColor={node => node.selected ? '#4338ca' : nodeAppearance(node.data.appearance as NodeAppearance | undefined)?.color || '#64748b'} nodeStrokeColor="#ffffff" nodeStrokeWidth={2} maskColor="rgba(99,102,241,.10)" maskStrokeColor="#a5b4fc" maskStrokeWidth={1} className="!rounded-xl !border !border-indigo-200 !bg-indigo-50 shadow-md" style={{ width: 172, height: 118 }} /><Controls showInteractive={false} /></ReactFlow></div> : null}
  </section>
}

import { useEffect, useMemo, useState, type KeyboardEvent } from 'react'
import {
  Background,
  Controls,
  Handle,
  MiniMap,
  Position,
  ReactFlow,
  applyNodeChanges,
  useUpdateNodeInternals,
  type Connection,
  type Edge,
  type Node,
  type NodeChange,
  type NodeProps,
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import type { RegimeGraphNode, RegimeNodeSchema } from '../../services/regimeGraph'
import RegimeHelpTip from './RegimeHelpTip'
import { regimePortLabel, regimeTypeLabel } from './regimeDisplay'

export interface RegimeCanvasEdge {
  id: string
  source: string
  sourcePort: string
  target: string
  targetPort: string
}

export type RegimeCanvasNodeChange =
  | { type: 'select'; id: string }
  | { type: 'remove'; id: string }
  | { type: 'position'; id: string; position: { x: number; y: number } }

export interface RegimeGraphCanvasProps {
  nodes: RegimeGraphNode[]
  edges: RegimeCanvasEdge[]
  schemas: RegimeNodeSchema[]
  selectedNodeId: string | null
  onNodesChange: (changes: RegimeCanvasNodeChange[]) => void
  onConnect: (connection: { source: string; sourcePort: string; target: string; targetPort: string }) => void
  onDuplicate: (nodeIds: string[]) => void
  onSelectionChange?: (nodeIds: string[]) => void
}

type RegimeFlowData = {
  label: string
  schema?: RegimeNodeSchema
  ready: boolean
}

type RegimeFlowNodeType = Node<RegimeFlowData, 'regime'>

function portColor(valueType?: string) {
  if (valueType?.startsWith('state_codes')) return '#f59e0b'
  if (valueType?.startsWith('probabilities')) return '#8b5cf6'
  if (valueType?.startsWith('confidence')) return '#06b6d4'
  return '#4f46e5'
}

function RegimeFlowNode({ id, data, selected }: NodeProps<RegimeFlowNodeType>) {
  const inputs = data.schema?.inputs ?? []
  const outputs = data.schema?.outputs ?? []
  const updateNodeInternals = useUpdateNodeInternals()
  const portSignature = [...inputs.map((port) => `in:${port.id}`), ...outputs.map((port) => `out:${port.id}`)].join('|')
  useEffect(() => {
    const frame = window.requestAnimationFrame(() => updateNodeInternals(id))
    return () => window.cancelAnimationFrame(frame)
  }, [id, portSignature, updateNodeInternals])
  return <div className={`min-w-[210px] rounded-xl border bg-white px-3 py-3 shadow-sm ${selected ? 'border-indigo-500 ring-2 ring-indigo-100' : data.ready ? 'border-slate-300' : 'border-amber-400'}`}>
    <p className="text-xs font-bold text-slate-900">{data.label}</p>
    <p className="mt-1 text-[9px] font-semibold tracking-wide text-slate-400">{data.schema?.category_label || '计算节点'}</p>
    <div className="mt-3 grid grid-cols-2 gap-x-5 gap-y-1 border-t border-slate-100 pt-2 text-[9px] text-slate-500">
      <div className="space-y-1.5">{inputs.map((port) => <div key={port.id} className="relative pl-1"><Handle id={port.id} type="target" position={Position.Left} style={{ position: 'absolute', left: -17, top: '50%', width: 9, height: 9, background: portColor(port.value_type), border: '2px solid white' }} /><span className="block truncate" title={`${regimePortLabel(port)}：${regimeTypeLabel(port.value_type, port.type_label)}`}>{regimePortLabel(port)}{port.required ? ' *' : ''}</span></div>)}</div>
      <div className="space-y-1.5 text-right">{outputs.map((port) => <div key={port.id} className="relative pr-1"><span className="block truncate" title={`${regimePortLabel(port)}：${regimeTypeLabel(port.value_type, port.type_label)}`}>{regimePortLabel(port)}</span><Handle id={port.id} type="source" position={Position.Right} style={{ position: 'absolute', right: -17, top: '50%', width: 9, height: 9, background: portColor(port.value_type), border: '2px solid white' }} /></div>)}</div>
    </div>
  </div>
}

const nodeTypes = { regime: RegimeFlowNode }

function schemaName(node: RegimeGraphNode, schemas: RegimeNodeSchema[]) {
  return node.label || schemas.find((item) => item.id === node.type || item.type === node.type)?.label || '计算节点'
}

function NodeCard({
  node,
  index,
  count,
  selected,
  incoming,
  schema,
  onNodesChange,
}: {
  node: RegimeGraphNode
  index: number
  count: number
  selected: boolean
  incoming: RegimeCanvasEdge[]
  schema?: RegimeNodeSchema
  onNodesChange: RegimeGraphCanvasProps['onNodesChange']
}) {
  const move = (delta: number) => onNodesChange([{
    type: 'position',
    id: node.id,
    position: { x: (index + delta) * 240, y: node.position?.y ?? 0 },
  }])
  return (
    <article
      className={`group relative min-w-0 rounded-xl border bg-white p-3 text-left shadow-sm transition ${selected ? 'border-indigo-500 ring-2 ring-indigo-100' : 'border-slate-200 hover:border-indigo-200'}`}
      data-node-id={node.id}
    >
      <button type="button" onClick={() => onNodesChange([{ type: 'select', id: node.id }])} className="block min-h-12 w-full text-left focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500">
        <span className="flex items-start justify-between gap-2">
          <span className="min-w-0"><span className="block truncate text-sm font-bold text-slate-900">{node.label || schema?.label || '未命名节点'}</span><span className="mt-1 block truncate text-[10px] font-semibold tracking-wide text-slate-400">{schema?.category_label || '计算节点'}</span></span>
          <span className={`mt-1 h-2.5 w-2.5 shrink-0 rounded-full ${incoming.length || !schema?.inputs.length ? 'bg-emerald-500' : 'bg-amber-400'}`} aria-label={incoming.length || !schema?.inputs.length ? '输入已连接' : '等待连接输入'} />
        </span>
      </button>
      <div className="mt-3 space-y-1 border-t border-slate-100 pt-2 text-[10px] text-slate-500">
        {incoming.length ? incoming.map((edge) => <p key={edge.id} className="truncate"><span aria-hidden="true">←</span> {regimePortLabel(schema?.inputs.find((port) => port.id === edge.targetPort) || { id: edge.targetPort })} · 已连接上游</p>) : <p>{schema?.inputs.length ? '尚未连接上游节点' : '数据源 / 起始节点'}</p>}
        <p>{schema?.outputs?.map(regimePortLabel).join('、') || '输出待元数据定义'}</p>
      </div>
      <div className="mt-3 flex items-center justify-end gap-1 border-t border-slate-100 pt-2">
        <button type="button" aria-label={`左移${schemaName(node, schema ? [schema] : [])}`} disabled={index === 0} onClick={() => move(-1)} className="min-h-8 rounded-md px-2 text-xs font-bold text-slate-500 hover:bg-slate-100 disabled:opacity-30">←</button>
        <button type="button" aria-label={`右移${schemaName(node, schema ? [schema] : [])}`} disabled={index === count - 1} onClick={() => move(1)} className="min-h-8 rounded-md px-2 text-xs font-bold text-slate-500 hover:bg-slate-100 disabled:opacity-30">→</button>
        <button type="button" aria-label={`删除${schemaName(node, schema ? [schema] : [])}`} onClick={() => onNodesChange([{ type: 'remove', id: node.id }])} className="min-h-8 rounded-md px-2 text-xs font-bold text-rose-600 hover:bg-rose-50">删除</button>
      </div>
    </article>
  )
}

export default function RegimeGraphCanvas({ nodes, edges, schemas, selectedNodeId, onNodesChange, onConnect, onDuplicate, onSelectionChange }: RegimeGraphCanvasProps) {
  const [selectedIds, setSelectedIds] = useState<string[]>(selectedNodeId ? [selectedNodeId] : [])
  const [copiedIds, setCopiedIds] = useState<string[]>([])
  const nodeIdentity = nodes.map((node) => node.id).join('|')
  useEffect(() => {
    const valid = new Set(nodes.map((node) => node.id))
    setSelectedIds((current) => {
      const kept = current.filter((id) => valid.has(id))
      if (selectedNodeId && valid.has(selectedNodeId) && !kept.includes(selectedNodeId)) return [selectedNodeId]
      return kept
    })
    setCopiedIds((current) => current.filter((id) => valid.has(id)))
  }, [nodeIdentity, selectedNodeId])
  useEffect(() => { onSelectionChange?.(selectedIds) }, [onSelectionChange, selectedIds])
  const sortedNodes = useMemo(() => [...nodes].sort((left, right) => (left.position?.x ?? nodes.indexOf(left) * 240) - (right.position?.x ?? nodes.indexOf(right) * 240)), [nodes])
  const schemaMap = useMemo(() => new Map(schemas.flatMap((schema) => [[schema.id, schema] as const, ...(schema.type ? [[schema.type, schema] as const] : [])])), [schemas])
  const graphFlowNodes = useMemo<RegimeFlowNodeType[]>(() => nodes.map((node, index) => {
    const schema = schemaMap.get(node.type)
    const ready = !schema?.inputs?.length || Object.keys(node.inputs || {}).length > 0
    return {
      id: node.id,
      type: 'regime',
      position: node.position ?? { x: (index % 3) * 260, y: Math.floor(index / 3) * 150 },
      initialWidth: 210,
      initialHeight: Math.max(82, 72 + Math.max(schema?.inputs?.length ?? 0, schema?.outputs?.length ?? 0) * 18),
      data: { label: node.label || schema?.label || '计算节点', schema, ready },
    }
  }), [nodes, schemaMap])
  const [flowNodes, setFlowNodes] = useState<RegimeFlowNodeType[]>(graphFlowNodes)
  useEffect(() => {
    setFlowNodes((current) => {
      const currentById = new Map(current.map((node) => [node.id, node]))
      return graphFlowNodes.map((node) => {
        const previous = currentById.get(node.id)
        return {
          ...node,
          measured: previous?.measured,
          selected: previous?.selected ?? false,
        }
      })
    })
  }, [graphFlowNodes])
  useEffect(() => {
    const selected = new Set(selectedIds)
    setFlowNodes((current) => current.map((node) => node.selected === selected.has(node.id) ? node : { ...node, selected: selected.has(node.id) }))
  }, [selectedIds])
  const flowEdges = useMemo<Edge[]>(() => edges.map((edge) => {
    const target = nodes.find((node) => node.id === edge.target)
    const targetPort = target ? schemaMap.get(target.type)?.inputs.find((port) => port.id === edge.targetPort) : undefined
    return { id: edge.id, source: edge.source, sourceHandle: edge.sourcePort, target: edge.target, targetHandle: edge.targetPort, label: targetPort ? regimePortLabel(targetPort) : '输入连接', animated: false, style: { stroke: '#6366f1', strokeWidth: 1.8 }, labelStyle: { fill: '#475569', fontSize: 10 } }
  }), [edges, nodes, schemaMap])
  const handleFlowChanges = (changes: NodeChange<RegimeFlowNodeType>[]) => {
    setFlowNodes((current) => applyNodeChanges(changes, current))
    const selectionChanges = changes.filter((change): change is Extract<NodeChange<RegimeFlowNodeType>, { type: 'select' }> => change.type === 'select')
    if (selectionChanges.length) setSelectedIds((current) => {
      const next = new Set(current)
      selectionChanges.forEach((change) => { if (change.selected) next.add(change.id); else next.delete(change.id) })
      return [...next]
    })
    const next = changes.flatMap<RegimeCanvasNodeChange>((change) => {
      if (change.type === 'remove') return [{ type: 'remove', id: change.id }]
      if (change.type === 'select' && change.selected) return [{ type: 'select', id: change.id }]
      return []
    })
    if (next.length) onNodesChange(next)
  }
  const commitPositions = (movedNodes: RegimeFlowNodeType[]) => {
    const changes = movedNodes.map<RegimeCanvasNodeChange>((node) => ({ type: 'position', id: node.id, position: node.position }))
    if (changes.length) onNodesChange(changes)
  }
  const handleConnect = (connection: Connection) => {
    if (!connection.source || !connection.target || !connection.sourceHandle || !connection.targetHandle) return
    onConnect({ source: connection.source, sourcePort: connection.sourceHandle, target: connection.target, targetPort: connection.targetHandle })
  }
  const isValidConnection = (connection: Connection | Edge) => {
    if (!connection.source || !connection.target || !connection.sourceHandle || !connection.targetHandle) return false
    const sourceNode = nodes.find((node) => node.id === connection.source)
    const targetNode = nodes.find((node) => node.id === connection.target)
    const sourcePort = sourceNode ? schemaMap.get(sourceNode.type)?.outputs.find((port) => port.id === connection.sourceHandle) : undefined
    const targetPort = targetNode ? schemaMap.get(targetNode.type)?.inputs.find((port) => port.id === connection.targetHandle) : undefined
    return Boolean(sourcePort && targetPort && sourcePort.value_type === targetPort.value_type)
  }
  const copySelection = () => {
    setCopiedIds(selectedIds.length ? [...selectedIds] : selectedNodeId ? [selectedNodeId] : [])
  }
  const pasteSelection = () => {
    if (copiedIds.length) onDuplicate(copiedIds)
  }
  const handleKeyboard = (event: KeyboardEvent<HTMLElement>) => {
    if (!(event.metaKey || event.ctrlKey)) return
    if (event.key.toLowerCase() === 'c') { event.preventDefault(); copySelection() }
    if (event.key.toLowerCase() === 'v') { event.preventDefault(); pasteSelection() }
  }
  return (
    <section
      aria-label="历史情景可编辑节点画布"
      data-testid="regime-graph-canvas"
      tabIndex={0}
      onKeyDown={handleKeyboard}
      className="relative min-h-[420px] overflow-auto rounded-2xl border border-slate-200 bg-slate-50 p-3 sm:p-4"
      style={{ backgroundImage: 'radial-gradient(#cbd5e1 1px, transparent 1px)', backgroundSize: '18px 18px' }}
    >
      <div className="mb-4 flex flex-wrap items-center justify-between gap-2 rounded-xl border border-slate-200 bg-white/95 px-3 py-2 backdrop-blur">
        <div><h3 className="text-sm font-bold text-slate-900">计算图画布 <RegimeHelpTip label="计算图画布说明" text="把数据源、特征、模型和状态处理节点按先后关系连接起来。点击节点可在右侧改参数；右下角小地图用于快速定位大型计算图。" /></h3><p className="mt-0.5 text-[11px] text-slate-500">按节点元数据构建；输入连接与参数在右侧检查器编辑。</p></div>
        <div className="flex flex-wrap items-center gap-2 text-[10px] font-bold text-slate-500"><button type="button" disabled={!selectedIds.length && !selectedNodeId} onClick={copySelection} className="min-h-8 rounded-lg border border-slate-200 px-2 text-slate-700 disabled:opacity-40">复制所选</button><button type="button" disabled={!copiedIds.length} onClick={pasteSelection} className="min-h-8 rounded-lg border border-slate-200 px-2 text-slate-700 disabled:opacity-40">粘贴副本</button><span className="rounded-full bg-slate-100 px-2 py-1">{selectedIds.length || (selectedNodeId ? 1 : 0)} 已选</span><span className="rounded-full bg-slate-100 px-2 py-1">{nodes.length} 节点</span><span className="rounded-full bg-slate-100 px-2 py-1">{edges.length} 连线</span></div>
      </div>
      {!nodes.length ? <div className="grid min-h-[300px] place-items-center rounded-xl border border-dashed border-slate-300 bg-white/80 p-8 text-center"><div><p className="text-sm font-bold text-slate-800">空白计算图</p><p className="mt-2 max-w-sm text-xs leading-5 text-slate-500">从左侧资源库添加数据源、算子或模型，或从顶部载入一个可完全修改的模板。</p></div></div> : null}
      <div className="space-y-3 md:hidden" data-testid="regime-graph-mobile-list">{sortedNodes.map((node, index) => <NodeCard key={node.id} node={node} index={index} count={sortedNodes.length} selected={node.id === selectedNodeId} incoming={edges.filter((edge) => edge.target === node.id)} schema={schemaMap.get(node.type)} onNodesChange={onNodesChange} />)}</div>
      {nodes.length ? <div className="hidden h-[520px] min-w-[680px] overflow-hidden rounded-xl border border-slate-200 bg-white md:block" data-testid="regime-graph-desktop-flow"><ReactFlow nodeTypes={nodeTypes} nodes={flowNodes} edges={flowEdges} onNodesChange={handleFlowChanges} onNodeDragStop={(_event, _node, movedNodes) => commitPositions(movedNodes)} onSelectionDragStop={(_event, movedNodes) => commitPositions(movedNodes)} onConnect={handleConnect} isValidConnection={isValidConnection} onNodeClick={(_event, node) => onNodesChange([{ type: 'select', id: node.id }])} selectionOnDrag selectionKeyCode="Shift" multiSelectionKeyCode={['Meta', 'Control']} fitView fitViewOptions={{ padding: 0.22 }} minZoom={0.25} maxZoom={1.8} deleteKeyCode={['Backspace', 'Delete']}><Background gap={18} size={1} color="#cbd5e1" /><MiniMap data-testid="regime-canvas-minimap" ariaLabel="计算图缩略导航：深色方块代表节点，可拖动快速定位大型计算图" pannable zoomable nodeBorderRadius={6} nodeColor={(node) => node.selected ? '#4338ca' : '#64748b'} nodeStrokeColor="#ffffff" nodeStrokeWidth={2} maskColor="rgba(99,102,241,.10)" maskStrokeColor="#a5b4fc" maskStrokeWidth={1} className="!rounded-xl !border !border-indigo-200 !bg-indigo-50 shadow-md" style={{ width: 172, height: 118 }} /><Controls showInteractive={false} /></ReactFlow></div> : null}
    </section>
  )
}

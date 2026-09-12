import { ReactFlowProvider, useReactFlow } from '@xyflow/react'
import { useEffect, useRef } from 'react'
import GraphCanvas from '../../components/computation-graph/GraphCanvas'
import type { GraphCanvasProps, CanvasEdge, CanvasNodeChange } from '../../components/computation-graph/types'
import type { RegimeGraphNode, RegimeNodeSchema } from '../../services/regimeGraph'
import RegimeHelpTip from './RegimeHelpTip'
import { regimePortLabel, regimeTypeLabel } from './regimeDisplay'
import { layoutRegimeGraph, regimeConnectionIssue } from './regimeGraphEditing'

export type RegimeCanvasEdge = CanvasEdge
export type RegimeCanvasNodeChange = CanvasNodeChange
export interface RegimeGraphCanvasProps {
  nodes: RegimeGraphNode[]
  edges: RegimeCanvasEdge[]
  schemas: RegimeNodeSchema[]
  selectedNodeId: string | null
  onNodesChange: GraphCanvasProps['onNodesChange']
  onConnect: GraphCanvasProps['onConnect']
  onDuplicate: GraphCanvasProps['onDuplicate']
  onSelectionChange?: GraphCanvasProps['onSelectionChange']
}
const testIds = { canvas: 'regime-graph-canvas', mobile: 'regime-graph-mobile-list', flow: 'regime-graph-desktop-flow', minimap: 'regime-canvas-minimap' }
function portColor(valueType?: string) {
  if (valueType?.startsWith('state_codes')) return '#f59e0b'
  if (valueType?.startsWith('probabilities')) return '#8b5cf6'
  if (valueType?.startsWith('confidence')) return '#06b6d4'
  return '#4f46e5'
}

function CanvasTools({ nodes, edges, selectedNodeId, onNodesChange }: Pick<RegimeGraphCanvasProps, 'nodes' | 'edges' | 'selectedNodeId' | 'onNodesChange'>) {
  const { fitView } = useReactFlow()
  const fitButton = useRef<HTMLButtonElement>(null)
  const identity = nodes.map(node => node.id).join('|')
  useEffect(() => {
    const container = fitButton.current?.closest('[data-testid="regime-graph-canvas"]')
    if (!container || typeof ResizeObserver === 'undefined') return
    let frame = 0
    // The workbench establishes its available height after mounting the graph.
    // Refit here once the visible container has a real size; shared canvas stays unchanged.
    const observer = new ResizeObserver(() => {
      if (!container.clientWidth || !container.clientHeight) return
      cancelAnimationFrame(frame)
      frame = requestAnimationFrame(() => { void fitView({ padding: 0.22, duration: 0 }) })
    })
    observer.observe(container)
    return () => { observer.disconnect(); cancelAnimationFrame(frame) }
  }, [identity, fitView])
  const button = 'min-h-8 rounded-xl border border-slate-200 bg-white px-2 text-slate-700 disabled:opacity-40'
  return <><button ref={fitButton} type="button" disabled={!nodes.length} onClick={() => void fitView({ padding: 0.22, duration: 200 })} className={button}>适应全部</button><button type="button" disabled={!selectedNodeId} onClick={() => { if (selectedNodeId) void fitView({ nodes: [{ id: selectedNodeId }], maxZoom: 1, duration: 200 }) }} className={button}>定位选中</button><button type="button" disabled={!nodes.length} onClick={() => onNodesChange(layoutRegimeGraph(nodes, edges))} className={button}>自动布局</button></>
}

/** Keep the shared interaction intact; this workspace alone fills its remaining viewport. */
export default function RegimeGraphCanvas(props: RegimeGraphCanvasProps) {
  return <ReactFlowProvider><div className="h-full min-h-0 min-w-0 [&>section]:flex [&>section]:h-full [&>section]:min-h-0 [&>section]:min-w-0 [&>section]:flex-col [&>section]:overflow-hidden [&>section>div:first-child]:mb-2 [&>section>div:first-child]:shrink-0 [&_[data-testid=regime-graph-mobile-list]]:overflow-auto">
    <GraphCanvas {...props} testIds={testIds} ariaLabel="历史情景可编辑节点画布"
      flowClassName="hidden min-h-0 min-w-0 flex-1 overflow-hidden rounded-xl border border-slate-200 bg-white md:block"
      toolbar={<CanvasTools {...props} />}
      validateConnection={(connection) => regimeConnectionIssue(props.nodes, props.schemas, connection) === null}
      portLabel={regimePortLabel} typeLabel={regimeTypeLabel} portColor={portColor}
      description="连接数据、特征与识别规则；点击节点打开参数，关闭参数即可继续查看全图。"
      emptyDescription="点击“添加节点”，或切换“引导配置”从模板开始。所有节点和连线都可以自由修改。"
      help={<RegimeHelpTip label="计算图画布说明" text="将数据源、特征和识别规则按计算先后连接。拖动节点只改变位置，不影响已有结果；自动布局保留所有连接。" />} />
  </div></ReactFlowProvider>
}

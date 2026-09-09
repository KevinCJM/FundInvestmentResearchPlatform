import type { ReactNode } from 'react'
import type { NodeAppearance } from './nodeAppearance'

export interface GraphPort {
  id: string; name?: string; label?: string; description?: string
  value_type?: string; type?: string; type_label?: string; required?: boolean; multiple?: boolean
}
export interface GraphNodeSchema {
  id: string; type?: string; label: string; category?: string; category_label?: string
  inputs: GraphPort[]; outputs: GraphPort[]
}
export interface CanvasNode {
  id: string; type: string; label?: string; inputs: Record<string, unknown>
  position?: { x: number; y: number }; ready?: boolean; statusLabel?: string
  appearance?: NodeAppearance
}
export interface CanvasEdge {
  id: string; source: string; sourcePort: string; target: string; targetPort: string
  kind?: 'data' | 'control'; label?: string
}
export type CanvasConnection = Omit<CanvasEdge, 'id' | 'kind' | 'label'>
export type CanvasNodeChange =
  | { type: 'select'; id: string }
  | { type: 'remove'; id: string }
  | { type: 'position'; id: string; position: { x: number; y: number } }
export interface GraphViewport { x: number; y: number; zoom: number }
export interface GraphCanvasProps {
  nodes: CanvasNode[]; edges: CanvasEdge[]; schemas: GraphNodeSchema[]; selectedNodeId: string | null
  onNodesChange: (changes: CanvasNodeChange[]) => void
  onConnect: (connection: CanvasConnection) => void
  onDuplicate: (nodeIds: string[]) => void
  onSelectionChange?: (nodeIds: string[]) => void
  onEdgesRemove?: (edgeIds: string[]) => void
  validateConnection?: (connection: CanvasConnection) => boolean
  isPortCompatible?: (output: GraphPort, input: GraphPort, connection: CanvasConnection) => boolean
  onResourceDrop?: (resource: string, position: { x: number; y: number }) => void
  onNodeActivate?: (nodeId: string) => void
  portLabel?: (port: GraphPort) => string
  typeLabel?: (valueType?: string, label?: string) => string
  portColor?: (valueType?: string) => string
  readOnly?: boolean; help?: ReactNode; toolbar?: ReactNode
  ariaLabel?: string; description?: string; emptyDescription?: string
  testIds?: { canvas: string; mobile: string; flow: string; minimap: string }
  flowClassName?: string; minZoom?: number
  viewport?: GraphViewport; onViewportCommit?: (viewport: GraphViewport) => void
}

import { useEffect, useRef, useState } from 'react'
import { graphOrder } from '../../components/computation-graph/graph'
import { cloneRegimeGraphDefinition, expandRegimeComposite, type RegimeCompositeExpansion, type RegimeGraphDefinition, type RegimeMode } from '../../services/regimeGraph'

/** Layout is editor-only; the server remains the sole owner of graph expansion. */
export function positionExpandedDefinition(original: RegimeGraphDefinition, result: RegimeCompositeExpansion): RegimeGraphDefinition {
  const previous = new Map(original.graph.nodes.map(node => [node.id, node]))
  const added = new Set(result.inserted_node_ids)
  const definition = cloneRegimeGraphDefinition(result.definition)
  const byId = new Map(definition.graph.nodes.map(node => [node.id, node]))
  if (result.contract_version !== 1 || !previous.has(result.replaced_node_id) || !added.has(result.primary_node_id)
      || [...added].some(id => previous.has(id) || !byId.has(id)) || byId.has(result.replaced_node_id)) {
    throw new Error('组合展开返回了不匹配的节点，请重新加载节点目录。')
  }
  const anchor = previous.get(result.replaced_node_id)?.position ?? { x: 0, y: 0 }
  const edges = [...added].flatMap(id => Object.values(byId.get(id)!.inputs)
    .filter(ref => added.has(ref.node_id)).map(ref => ({ source: ref.node_id, target: id })))
  const levels = new Map<string, number>()
  const rows = new Map<number, number>()
  for (const id of graphOrder([...added], edges)) {
    const level = 1 + Math.max(-1, ...edges.filter(edge => edge.target === id).map(edge => levels.get(edge.source) ?? -1))
    const row = rows.get(level) ?? 0
    levels.set(id, level); rows.set(level, row + 1)
    byId.get(id)!.position = { x: anchor.x + level * 300, y: anchor.y + row * 180 }
  }
  for (const node of definition.graph.nodes) {
    if (!added.has(node.id) && previous.get(node.id)?.position) node.position = { ...previous.get(node.id)!.position! }
  }
  return definition
}

type Options = {
  definition: RegimeGraphDefinition
  mode: RegimeMode
  blocked: boolean
  onChange: (definition: RegimeGraphDefinition) => void
  onSelected: (id: string) => void
  onError: (message: string) => void
  onNotice: (message: string) => void
}

export default function useRegimeCompositeExpansion(options: Options) {
  const [expanding, setExpanding] = useState(false)
  const controller = useRef<AbortController | null>(null)
  const request = useRef(0)
  const context = useRef('')
  context.current = JSON.stringify([options.definition, options.mode, options.blocked])
  useEffect(() => () => { request.current += 1; controller.current?.abort() }, [])

  const expand = async (source: RegimeGraphDefinition, nodeId: string) => {
    if (controller.current || options.blocked) return
    const startContext = context.current
    const id = ++request.current
    const abort = new AbortController()
    controller.current = abort
    setExpanding(true); options.onError('')
    try {
      const result = await expandRegimeComposite(source, nodeId, options.mode, abort.signal)
      if (abort.signal.aborted || id !== request.current) return
      if (context.current !== startContext) {
        options.onNotice('草稿或识别模式已修改，未应用过期的展开结果。')
        return
      }
      const next = positionExpandedDefinition(source, result)
      options.onChange(next)
      options.onSelected(result.primary_node_id)
      options.onNotice(`已展开为 ${result.inserted_node_ids.length} 个计算步骤，原有连线已保留。可点击“撤销”恢复。`)
    } catch (reason) {
      if (!abort.signal.aborted && id === request.current && context.current === startContext) {
        options.onError(reason instanceof Error ? reason.message : '组合展开失败，原草稿未改变。')
      }
    } finally {
      if (id === request.current) { controller.current = null; setExpanding(false) }
    }
  }
  return { expand, expanding }
}

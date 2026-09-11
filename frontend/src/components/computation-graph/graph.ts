import type { CanvasEdge } from './types'

/** Stable topological traversal; array order is only the tie-breaker. */
export function graphOrder(ids: string[], edges: Pick<CanvasEdge, 'source' | 'target'>[]): string[] {
  if (new Set(ids).size !== ids.length) throw new Error('节点 ID 不可重复。')
  const pending = new Map(ids.map(id => [id, new Set<string>()]))
  edges.forEach(edge => {
    if (!pending.has(edge.source) || !pending.has(edge.target)) throw new Error('连线引用了不存在的节点。')
    pending.get(edge.target)!.add(edge.source)
  })
  const order: string[] = []
  while (pending.size) {
    const id = ids.find(key => pending.has(key) && pending.get(key)!.size === 0)
    if (!id) throw new Error('不能形成循环依赖，请先断开冲突连线。')
    order.push(id); pending.delete(id)
    pending.forEach(inputs => inputs.delete(id))
  }
  return order
}

/** Fold very long chains into bands, retaining level order and avoiding overlap. */
export function layoutGraph(ids: string[], edges: Pick<CanvasEdge, 'source' | 'target'>[]): Record<string, { x: number; y: number }> {
  let ordered: string[]
  try { ordered = graphOrder(ids, edges) } catch { ordered = ids }
  const depth = new Map<string, number>()
  const layers: string[][] = []
  ordered.forEach(id => {
    const parents = edges.filter(edge => edge.target === id).map(edge => depth.get(edge.source) ?? -1)
    const level = parents.length ? Math.max(...parents) + 1 : 0
    depth.set(id, level); (layers[level] ??= []).push(id)
  })
  const positions: Record<string, { x: number; y: number }> = {}
  let y = 0
  for (let start = 0; start < layers.length; start += 5) {
    const band = layers.slice(start, start + 5)
    band.forEach((layer, column) => layer?.forEach((id, row) => { positions[id] = { x: column * 300, y: y + row * 190 } }))
    y += Math.max(1, ...band.map(layer => layer?.length ?? 0)) * 190 + 70
  }
  return positions
}

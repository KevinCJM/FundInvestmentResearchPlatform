import { useState, type RefObject } from 'react'
import type { RegimeGraphConnection, RegimeGraphDefinition, RegimeMode, RegimeNodeSchema } from '../../services/regimeGraph'
import { regimePortLabel } from './regimeDisplay'

export const comparisonKey = (ref: RegimeGraphConnection) => JSON.stringify([ref.node_id, ref.port])
type Option = RegimeGraphConnection & { label: string; relationship: string; reason: string }

export function comparisonOptions(definition: RegimeGraphDefinition, schemas: RegimeNodeSchema[], target: RegimeGraphConnection, mode: RegimeMode): Option[] {
  const nodes = new Map(definition.graph.nodes.map(node => [node.id, node]))
  const schemaFor = (id: string) => schemas.find(item => [item.id, item.type, item.type_id].includes(nodes.get(id)?.type))
  const dependencies = (id: string) => {
    const seen = new Set<string>(), pending = [id]
    for (const key of pending) {
      if (seen.has(key)) continue
      seen.add(key)
      pending.push(...Object.values(nodes.get(key)?.inputs || {}).map(ref => ref.node_id))
      pending.push(...(definition.graph.edges || []).filter(edge => edge.target.node_id === key).map(edge => edge.source.node_id))
    }
    return seen
  }
  const upstream = dependencies(target.node_id)
  return definition.graph.nodes.flatMap(node => {
    const schema = schemaFor(node.id), branch = [...dependencies(node.id)]
    const unavailable = branch.find(id => !schemaFor(id) || schemaFor(id)?.available === false)
    const retrospective = branch.some(id => { const item = schemaFor(id); return item?.causal === false || item?.repaints === true || item?.supports_realtime === false })
    return (schema?.outputs || []).filter(port => node.id !== target.node_id || port.id !== target.port).map(port => ({
      node_id: node.id, port: port.id, label: `${node.label?.trim() || schema?.label || node.id} · ${regimePortLabel(port)}`,
      relationship: node.id === target.node_id ? '同节点其他输出' : upstream.has(node.id) ? '上游节点' : '其他节点',
      reason: !['series<float64>', 'confidence<time>'].includes(port.value_type || port.type || '') ? '此输出不是连续数值序列，请单独预览。'
        : unavailable ? '此节点或依赖当前不可用。' : mode === 'realtime' && retrospective ? '此节点或其上游需要事后分析。' : '',
    }))
  })
}

export default function RegimeComparisonPicker({ options, selected, onChange, disabled, detailsRef }: {
  options: Option[]; selected: RegimeGraphConnection[]; onChange: (refs: RegimeGraphConnection[]) => void; disabled: boolean
  detailsRef: RefObject<HTMLDetailsElement>
}) {
  const [search, setSearch] = useState('')
  const selectedKeys = new Set(selected.map(comparisonKey))
  const visible = options.filter(item => `${item.label} ${item.node_id}`.toLowerCase().includes(search.trim().toLowerCase()))
  return <details ref={detailsRef} className="rounded-xl border border-slate-200 bg-white p-3" aria-label="选择对比节点">
    <summary className="cursor-pointer text-sm font-semibold text-slate-700">叠加对比节点{selected.length ? `（已选 ${selected.length}）` : '（可选）'}</summary>
    <p className="my-2 text-xs leading-5 text-slate-600">可选择画布上的其他分支，最多 7 个输出。选好后点击“预览节点数据”，一起计算并显示；无需连接到主节点。</p>
    <input aria-label="搜索对比节点" placeholder="搜索节点或输出名称" value={search} onChange={event => setSearch(event.target.value)} className="mb-2 min-h-10 w-full rounded-lg border border-slate-300 px-3 text-sm" />
    <div className="max-h-52 space-y-2 overflow-auto">
      {visible.map(item => {
        const checked = selectedKeys.has(comparisonKey(item))
        return <label key={comparisonKey(item)} className="flex items-start gap-2 rounded-lg bg-slate-50 p-2 text-xs text-slate-700">
          <input type="checkbox" className="mt-0.5" checked={checked} disabled={disabled || (!checked && (!!item.reason || selected.length >= 7))} onChange={event => onChange(event.target.checked ? [...selected, { node_id: item.node_id, port: item.port }] : selected.filter(ref => comparisonKey(ref) !== comparisonKey(item)))} />
          <span className="min-w-0 break-words">{item.label}<span className="ml-2 text-slate-600">{item.relationship}</span>{item.reason && <span className="mt-1 block text-amber-800">{item.reason}</span>}</span>
        </label>
      })}
      {!visible.length && <p className="text-xs text-slate-600">没有匹配的其他输出。</p>}
    </div>
    {selected.length > 0 && <button type="button" disabled={disabled} onClick={() => onChange([])} className="mt-2 min-h-9 text-xs text-slate-600">清空对比选择</button>}
  </details>
}

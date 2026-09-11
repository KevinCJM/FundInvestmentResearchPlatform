import { useEffect, useState } from 'react'
import SeriesOutputFields from '../../components/computation-graph/SeriesOutputFields'
import type { RegimeGraphDefinition, RegimeNodeSchema } from '../../services/regimeGraph'
import { regimePortLabel } from './regimeDisplay'

const slots = [
  ['state', '市场状态', 'state_codes<int64>'], ['probabilities', '状态概率', 'probabilities<time,state>'],
  ['confidence', '置信度', 'confidence<time>'], ['recognition_index', '识别时点', 'index<time>'],
  ['effective_index', '生效时点', 'index<time>'], ['reason_code', '原因码', 'reason_codes<int64>'],
] as const
type Slot = string

export default function RegimeSeriesOutputs({ definition, schemas, onChange, activeId, onSelectOutput }: {
  definition: RegimeGraphDefinition; schemas: RegimeNodeSchema[]; onChange: (next: RegimeGraphDefinition) => void
  activeId?: string; onSelectOutput?: (id: string) => void
}) {
  const [localActive, setLocalActive] = useState<Slot>('state')
  const active = activeId ?? localActive
  const setActive = (id: string) => { setLocalActive(id); onSelectOutput?.(id) }
  useEffect(() => { if (!slots.some(item => item[0] === active) && !definition.graph.outputs[active]) setActive('state') }, [active, definition.graph.outputs])
  const allSlots: Array<readonly [string, string, string]> = [...slots.map(([id, label, type]) => [id, definition.graph.channel_metadata?.[id]?.label || label, type] as const), ...Object.keys(definition.graph.channel_metadata || {}).filter(id => !slots.some(item => item[0] === id)).map(id => [id, definition.graph.channel_metadata![id].label || id, 'series<float64>'] as const)]
  const slot = allSlots.find(item => item[0] === active) || [active, active, 'series<float64>']
  const presentation = definition.graph.channel_metadata?.[active]
  const numericChoices = definition.graph.nodes.flatMap(node => (schemas.find(schema => (schema.id || schema.type) === node.type)?.outputs || []).filter(port => (port.value_type || port.type) === 'series<float64>').map(port => ({ node_id: node.id, port: port.id })))
  const patchPresentation = (patch: Partial<{ id: string; label: string; unit: string; display_format: 'number' | 'percent'; precision: number }>) => {
    if (!definition.graph.outputs[active]) return
    const outputs = { ...definition.graph.outputs }; const channel_metadata = { ...definition.graph.channel_metadata }
    const { id = active, ...changes } = patch
    if (id === active && Object.keys(changes).length === 0) return
    if (id !== active && (!/^[A-Za-z][A-Za-z0-9_]*$/.test(id) || outputs[id])) return
    channel_metadata[id] = { label: slot[1], ...presentation, ...changes }; outputs[id] = outputs[active]
    if (id !== active) { delete outputs[active]; delete channel_metadata[active]; setActive(id) }
    onChange({ ...definition, graph: { ...definition.graph, outputs, channel_metadata } })
  }
  const source = definition.graph.outputs[active]
  const choices = definition.graph.nodes.flatMap(node => (schemas.find(schema => (schema.id || schema.type) === node.type)?.outputs || []).filter(port => (port.value_type || port.type) === slot[2]).map(port => ({ id: `${node.id}:${port.id}`, label: `${node.label || schemas.find(schema => (schema.id || schema.type) === node.type)?.label || node.id} · ${regimePortLabel(port)}` })))
  return <section aria-label="时序输出配置" className="space-y-4">
    <div><h3 className="text-sm font-semibold text-slate-900">输出通道</h3><p className="mt-1 text-xs leading-5 text-slate-500">指定每个时点返回的值。市场状态使用枚举；图表与区间明细在结果页展示。</p></div>
    <div role="tablist" aria-label="时序输出通道" className="flex flex-wrap gap-2">{allSlots.filter(item => item[0] === 'state' || definition.graph.outputs[item[0]]).map(item => <button type="button" role="tab" aria-selected={active === item[0]} key={item[0]} onClick={() => setActive(item[0])} className={`min-h-10 rounded-lg border px-3 text-xs font-semibold ${active === item[0] ? 'border-violet-500 bg-violet-600 text-white' : 'border-slate-200 bg-white'}`}>{item[1]}</button>)}<select aria-label="添加输出通道" value="" onChange={event => setActive(event.target.value as Slot)} className="min-h-10 rounded-lg border border-dashed border-violet-300 bg-white px-2 text-xs text-violet-700"><option value="">添加输出通道</option>{slots.filter(item => item[0] !== 'state' && !definition.graph.outputs[item[0]]).map(item => <option key={item[0]} value={item[0]}>{item[1]}</option>)}</select><button type="button" disabled={!numericChoices.length || Object.keys(definition.graph.outputs).length >= 8} onClick={() => { let n = 1; while (definition.graph.outputs[`channel_${n}`]) n++; const id = `channel_${n}`; onChange({ ...definition, graph: { ...definition.graph, outputs: { ...definition.graph.outputs, [id]: numericChoices[0] }, channel_metadata: { ...definition.graph.channel_metadata, [id]: { label: `数值通道 ${n}` } } } }); setActive(id) }} className="min-h-10 rounded-lg border border-dashed border-violet-300 bg-white px-3 text-xs text-violet-700 disabled:opacity-40">新增数值通道</button></div>
    <fieldset className="rounded-xl border border-slate-200 bg-slate-50 p-4"><legend className="px-1 text-sm font-semibold text-slate-800">当前输出通道</legend>
      <SeriesOutputFields output={{ id: active, label: slot[1], unit: '', precision: 4, display_format: 'number' as 'number' | 'percent', ...presentation }} index={0} onChange={patchPresentation} onIdentifierCommit={id => { if (!/^[A-Za-z][A-Za-z0-9_]*$/.test(id) || (id !== active && definition.graph.outputs[id])) return '通道 ID 需以字母开头，且不能重复。'; patchPresentation({ id }); return undefined }} identityReadOnly={slots.some(item => item[0] === active)} enumItems={active === 'state' ? definition.states : undefined} onEnumChange={states => onChange({ ...definition, states })}>
        <label className="text-xs font-semibold text-slate-600">计算结果来源<select aria-label={active === 'state' ? '最终状态来源' : `${slot[1]}来源`} value={source ? `${source.node_id}:${source.port}` : ''} onChange={event => { const outputs = { ...definition.graph.outputs }; const [node_id, port] = event.target.value.split(':'); if (node_id && port) outputs[active] = { node_id, port }; else delete outputs[active]; const channel_metadata = { ...definition.graph.channel_metadata }; if (!outputs[active]) delete channel_metadata[active]; else if (!slots.some(item => item[0] === active)) channel_metadata[active] ||= { label: slot[1] }; onChange({ ...definition, graph: { ...definition.graph, outputs, channel_metadata } }) }} className="mt-1 min-h-10 w-full rounded-lg border border-slate-200 bg-white px-2"><option value="">选择兼容的计算结果</option>{choices.map(choice => <option key={choice.id} value={choice.id}>{choice.label}</option>)}</select></label>
      </SeriesOutputFields>
      {active !== 'state' && <button type="button" onClick={() => { const outputs = { ...definition.graph.outputs }; delete outputs[active]; const channel_metadata = { ...definition.graph.channel_metadata }; delete channel_metadata[active]; setActive('state'); onChange({ ...definition, graph: { ...definition.graph, outputs, channel_metadata } }) }} className="mt-3 min-h-10 rounded-lg border border-rose-100 bg-white px-3 text-xs text-rose-600">移除当前通道</button>}
    </fieldset>
  </section>
}

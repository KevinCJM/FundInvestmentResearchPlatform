import { useMemo, useState } from 'react'
import { Background, Controls, Handle, Position, ReactFlow, type Connection, type NodeProps } from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import type { TimingCatalog, TimingDefinition, TimingNode, TimingOperator, TimingValueType } from '../../services/timingResearch'

export const timingField = 'mt-1 min-h-10 w-full min-w-0 rounded-xl border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 focus:outline-none focus:ring-2 focus:ring-accent-500 disabled:bg-slate-50'
const actionClass = 'min-h-9 rounded-xl border border-slate-300 bg-white px-3 py-1 text-sm text-slate-700 hover:bg-slate-50 disabled:opacity-40'
const typeLabels: Record<TimingValueType, string> = { series: '数值序列', condition: '条件序列', panel: '篮子数值矩阵', condition_panel: '篮子条件矩阵' }
interface Props { definition: TimingDefinition; catalog: TimingCatalog; onChange: (definition: TimingDefinition) => void; disabled?: boolean }
type FlowData = Record<string, unknown> & { label: string; operator?: TimingOperator; selected: boolean }
function FlowStep({ data }: NodeProps) {
  const item = data as FlowData
  return <div className={`w-56 rounded-xl border bg-white p-3 shadow-sm ${item.selected ? 'border-accent-500 ring-2 ring-accent-100' : 'border-slate-300'}`}>
    <strong className="block truncate text-sm text-slate-900">{item.label}</strong><span className="text-xs text-slate-600">{item.operator?.label || '未知步骤'}</span>
    <div className="mt-3 space-y-2">{item.operator?.inputs.map(port => <div key={port.name} className="relative text-left text-xs text-slate-600" title={typeLabels[port.type]}><Handle type="target" id={port.name} position={Position.Left} style={{ left: -18, top: 7 }} />{port.label} · {typeLabels[port.type]}</div>)}</div>
    <div className="mt-3 space-y-2">{item.operator?.outputs.map(port => <div key={port.name} className="relative text-right text-xs text-accent-600" title={typeLabels[port.type]}>{port.label} · {typeLabels[port.type]}<Handle type="source" id={port.name} position={Position.Right} style={{ right: -18, top: 7 }} /></div>)}</div>
  </div>
}
const nodeTypes = { timing: FlowStep }

export default function TimingRuleEditor({ definition, catalog, onChange, disabled }: Props) {
  const [view, setView] = useState<'steps' | 'graph'>('steps')
  const [expanded, setExpanded] = useState<string>('')
  const [operatorId, setOperatorId] = useState('')
  const [selectedEdge, setSelectedEdge] = useState('')
  const byOperator = useMemo(() => new Map(catalog.operators.map(operator => [operator.id, operator])), [catalog.operators])
  const nodeLimit = catalog.limits?.steps ?? 128
  const operatorGroups = useMemo(() => [
    { label: '基础步骤 · 数据、计算与规则', items: catalog.operators.filter(operator => !operator.granularity?.includes('coupled')) },
    { label: '耦合内核 · 不可拆递推或拟合', items: catalog.operators.filter(operator => operator.granularity?.includes('coupled')) },
  ], [catalog.operators])
  const ports = (type: TimingValueType, exclude?: string) => definition.nodes.filter(node => node.id !== exclude).flatMap(node => (byOperator.get(node.op)?.outputs || []).filter(port => port.type === type).map(port => ({ value: `${node.id}.${port.name}`, label: `${node.label} · ${port.label}` })))
  const patchNode = (id: string, patch: Partial<TimingNode>) => onChange({ ...definition, nodes: definition.nodes.map(node => node.id === id ? { ...node, ...patch } : node) })
  const add = () => {
    const operator = byOperator.get(operatorId || catalog.operators[0]?.id)
    if (!operator) return
    let sequence = definition.nodes.length + 1
    while (definition.nodes.some(node => node.id === `step_${sequence}`)) sequence += 1
    const id = `step_${sequence}`
    onChange({ ...definition, nodes: [...definition.nodes, { id, label: operator.label, op: operator.id, inputs: {}, parameters: Object.fromEntries(operator.parameters.map(parameter => [parameter.name, parameter.default])) }] })
    setExpanded(id); setView('steps')
  }
  const remove = (id: string) => {
    const pointsTo = (reference: string | null) => reference?.startsWith(`${id}.`)
    onChange({ ...definition, entry: pointsTo(definition.entry) ? '' : definition.entry, exit: pointsTo(definition.exit) ? null : definition.exit, nodes: definition.nodes.filter(node => node.id !== id).map(node => ({ ...node, inputs: Object.fromEntries(Object.entries(node.inputs).filter(([, ref]) => !pointsTo(ref))) })) })
  }
  const connections = useMemo(() => definition.nodes.flatMap(node => Object.entries(node.inputs).filter(([, ref]) => !!ref).map(([input, ref]) => {
    const split = ref.lastIndexOf('.')
    return { id: `${node.id}:${input}`, source: ref.slice(0, split), sourceHandle: ref.slice(split + 1), target: node.id, targetHandle: input, selected: selectedEdge === `${node.id}:${input}` }
  })), [definition.nodes, selectedEdge])
  const canConnect = (connection: Connection) => {
    const source = definition.nodes.find(node => node.id === connection.source)
    const target = definition.nodes.find(node => node.id === connection.target)
    if (!source || !target || source.id === target.id) return false
    const output = byOperator.get(source.op)?.outputs.find(port => port.name === connection.sourceHandle)
    const input = byOperator.get(target.op)?.inputs.find(port => port.name === connection.targetHandle)
    return !!output && !!input && output.type === input.type
  }
  const connect = (connection: Connection) => {
    const target = definition.nodes.find(node => node.id === connection.target)
    if (!disabled && target && connection.sourceHandle && connection.targetHandle && canConnect(connection)) patchNode(target.id, { inputs: { ...target.inputs, [connection.targetHandle]: `${connection.source}.${connection.sourceHandle}` } })
  }
  const inputSelect = (value: string, type: TimingValueType, exclude: string | undefined, onSelect: (value: string) => void, label: string, optional = false) => <select aria-label={label} className={timingField} value={value} disabled={disabled} onChange={event => onSelect(event.target.value)}>
    <option value="">{optional ? '不使用' : '请选择输入来源'}</option>
    {ports(type, exclude).map(port => <option key={port.value} value={port.value}>{port.label}</option>)}
  </select>
  const renderStep = (node: TimingNode, index: number) => {
    const operator = byOperator.get(node.op)
    const missing = operator?.inputs.filter(port => port.required && !node.inputs[port.name]) || []
    return <section key={node.id} className="min-w-0 rounded-xl border border-slate-200 bg-white">
      <button type="button" aria-expanded={expanded === node.id} onClick={() => setExpanded(expanded === node.id ? '' : node.id)} className="flex w-full items-start gap-3 p-4 text-left">
        <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-accent-50 text-xs font-semibold text-accent-700">{index + 1}</span>
        <span className="min-w-0 flex-1"><strong className="block break-words text-sm text-slate-900">{node.label}</strong><span className="mt-1 block text-xs leading-5 text-slate-600">{operator?.description || '此步骤未在当前目录中注册。'}</span>{missing.length > 0 && <span className="mt-1 block text-xs text-amber-700">待连接：{missing.map(port => port.label).join('、')}</span>}</span>
        <span aria-hidden="true" className="text-slate-600">{expanded === node.id ? '−' : '+'}</span>
      </button>
      {expanded === node.id && <div className="space-y-3 border-t border-slate-100 p-4">
        <label className="block text-xs font-medium text-slate-600">步骤名称<input className={timingField} value={node.label} maxLength={80} disabled={disabled} onChange={event => patchNode(node.id, { label: event.target.value })} /></label>
        <div className="grid gap-3 sm:grid-cols-2">{operator?.inputs.map(port => <label key={port.name} className="min-w-0 text-xs font-medium text-slate-600">{port.label}<span className="ml-1 font-normal text-slate-600">· {typeLabels[port.type]}</span>{inputSelect(node.inputs[port.name] || '', port.type, node.id, value => patchNode(node.id, { inputs: { ...node.inputs, [port.name]: value } }), `${node.label} ${port.label}`, !port.required)}</label>)}
          {operator?.parameters.map(parameter => <label key={parameter.name} className="min-w-0 text-xs font-medium text-slate-600">{parameter.label}{parameter.options?.length ? <select className={timingField} value={node.parameters[parameter.name] ?? parameter.default} disabled={disabled} onChange={event => patchNode(node.id, { parameters: { ...node.parameters, [parameter.name]: parameter.type === 'string' ? event.target.value : Number(event.target.value) } })}>{parameter.options.map(option => <option key={option.value} value={option.value}>{option.label}</option>)}</select> : <input className={timingField} type={parameter.type === 'string' ? 'text' : 'number'} min={parameter.minimum} max={parameter.maximum} step={parameter.type === 'integer' ? 1 : 'any'} value={node.parameters[parameter.name] ?? parameter.default} disabled={disabled} onChange={event => patchNode(node.id, { parameters: { ...node.parameters, [parameter.name]: parameter.type === 'string' ? event.target.value : event.target.value === '' ? '' : Number(event.target.value) } })} />}</label>)}
        </div><button type="button" className="min-h-9 rounded-lg px-2 text-xs text-rose-700 hover:bg-rose-50 disabled:opacity-40" disabled={disabled} onClick={() => remove(node.id)}>移除步骤及关联连线</button>
      </div>}
    </section>
  }
  return <div className="space-y-4">
    <div className="flex flex-wrap items-center justify-between gap-3"><p className="text-sm text-slate-600">点开步骤调整输入和参数，组合出自己的算法。</p><div className="flex gap-1 rounded-lg bg-slate-100 p-1" aria-label="算法编辑方式">{(['steps', 'graph'] as const).map(mode => <button type="button" key={mode} aria-pressed={view === mode} onClick={() => setView(mode)} className={`min-h-8 rounded-lg px-3 text-xs ${view === mode ? 'bg-white font-semibold text-accent-700 shadow-sm' : 'text-slate-600'}`}>{mode === 'steps' ? '步骤编辑' : '连线图'}</button>)}</div></div>
    {!definition.nodes.length && <div className="rounded-xl border border-dashed border-slate-300 p-8 text-center text-sm text-slate-600">从下方添加一个数据或计算步骤，也可以先选择左侧模板。</div>}
    {view === 'graph' && definition.nodes.length > 0 && <div className="h-[480px] min-w-0 overflow-hidden rounded-xl border border-slate-200" aria-label="算法连线图"><ReactFlow nodes={definition.nodes.map((node, index) => ({ id: node.id, type: 'timing', position: { x: index % 3 * 330, y: Math.floor(index / 3) * 280 }, data: { label: node.label, operator: byOperator.get(node.op), selected: node.id === expanded } }))} edges={connections} nodeTypes={nodeTypes} isValidConnection={connection => canConnect(connection)} onConnect={connect} onNodeClick={(_, node) => setExpanded(node.id)} onEdgeClick={(_, edge) => setSelectedEdge(edge.id)} onPaneClick={() => setSelectedEdge('')} onEdgesDelete={edges => {
      if (disabled) return
      const deleted = new Set(edges.map(edge => edge.id))
      onChange({ ...definition, nodes: definition.nodes.map(node => ({ ...node, inputs: Object.fromEntries(Object.entries(node.inputs).filter(([input]) => !deleted.has(`${node.id}:${input}`))) })) })
    }} nodesDraggable={false} nodesConnectable={!disabled} edgesReconnectable={false} deleteKeyCode={disabled ? null : ['Backspace', 'Delete']} fitView minZoom={0.15} maxZoom={1.5} proOptions={{ hideAttribution: true }}><Background /><Controls showInteractive={false} /></ReactFlow></div>}
    <div className="space-y-3">{definition.nodes.filter(node => view === 'steps' || node.id === expanded).map(node => renderStep(node, definition.nodes.indexOf(node)))}</div>
    <div className="flex flex-wrap items-end gap-2 rounded-xl bg-slate-50 p-3"><label className="min-w-[160px] flex-1 text-xs text-slate-600">添加计算或规则<select aria-label="添加计算或规则" className={timingField} value={operatorId || catalog.operators[0]?.id || ''} disabled={disabled} onChange={event => setOperatorId(event.target.value)}>{operatorGroups.filter(group => group.items.length).map(group => <optgroup key={group.label} label={group.label}>{group.items.map(operator => <option key={operator.id} value={operator.id}>{operator.label}</option>)}</optgroup>)}</select></label><button type="button" className={actionClass} disabled={disabled || definition.nodes.length >= nodeLimit || !catalog.operators.length} onClick={add}>添加步骤</button><span className="text-xs text-slate-600">{definition.nodes.length} / {nodeLimit} 步</span></div>
    <div className="grid gap-4 rounded-xl border border-accent-100 bg-accent-50/40 p-4 sm:grid-cols-2"><label className="min-w-0 text-sm font-semibold text-slate-800">什么时候买入{inputSelect(definition.entry, 'condition', undefined, entry => onChange({ ...definition, entry }), '买入条件')}</label><label className="min-w-0 text-sm font-semibold text-slate-800">什么时候退出{inputSelect(definition.exit || '', 'condition', undefined, exit => onChange({ ...definition, exit: exit || null }), '退出条件', true)}<span className="mt-1 block text-xs font-normal text-slate-600">未选择时，按止盈、止损和持有期退出。</span></label></div>
  </div>
}

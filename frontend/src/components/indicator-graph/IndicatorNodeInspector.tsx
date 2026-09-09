import { useEffect, useMemo, useState } from 'react'
import { useI18n } from '../../i18n/runtime'
import type { IndicatorOperator, IndicatorVariable } from '../../services/customIndicators'
import type { AuthoringGraph, AuthoringNode, GraphBinding, GraphOutput, GraphValueType } from '../../services/indicatorGraph'
import { graphConnectionIssue, graphEdges, nodeLabel, outputNodeId, parametersForNode, portsForNode, nodePortKey, nodePortRef } from './indicatorGraphAdapter'
import { boundConstant, constantType } from './indicatorGraphConstants'
import { createGraphTextFormatter, graphAxesLabel, graphParameterLabel, graphTypeLabel, graphVariableLabel } from './indicatorGraphPresentation'

const field = 'mt-1 block min-h-11 w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-violet-300'
const button = 'min-h-10 rounded-lg border border-slate-200 px-3 py-2 text-sm font-semibold hover:bg-slate-50 disabled:opacity-40'
interface Props {
  graph: AuthoringGraph; selectedId: string | null; variables: IndicatorVariable[]; operators: IndicatorOperator[]
  types: Record<string, GraphValueType>; isTimeSeries: boolean
  onNodeChange: (node: AuthoringNode) => void; onOutputChange: (id: string, patch: Partial<GraphOutput>) => void
  onRemove: (id: string) => void; onDuplicate: (id: string) => void
}

export default function IndicatorNodeInspector({ graph, selectedId, variables, operators, types, isTimeSeries, onNodeChange, onOutputChange, onRemove, onDuplicate }: Props) {
  const { s, b, version: translationVersion } = useI18n()
  const text = useMemo(() => createGraphTextFormatter(variables, operators), [variables, operators, translationVersion])
  const node = graph.nodes.find(item => item.id === selectedId)
  const output = graph.outputs.find(item => outputNodeId(item.id) === selectedId)
  const [outputIdDraft, setOutputIdDraft] = useState(output?.id || '')
  const [outputIdError, setOutputIdError] = useState('')
  useEffect(() => { setOutputIdDraft(output?.id || ''); setOutputIdError('') }, [output?.id])
  const commitOutputId = () => {
    if (!output || outputIdDraft === output.id) return
    if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(outputIdDraft) || graph.outputs.some(item => item.id === outputIdDraft && item !== output)) {
      setOutputIdError('graph.outputIdentifierError')
      setOutputIdDraft(output.id)
      return
    }
    setOutputIdError('')
    onOutputChange(output.id, { id: outputIdDraft })
  }
  const options = graph.nodes.flatMap(item => portsForNode(item, operators, types).map(port => ({
    key: nodePortKey(item.id, port.id), node: item, port: port.id,
    label: `${s('graph.stepOption', { index: graph.nodes.indexOf(item) + 1, name: nodeLabel(item, variables, operators) })}${port.id === 'value' ? '' : ` · ${port.label}`}${port.type ? ` · ${graphTypeLabel(port.type)}` : ''}`,
  })))
  const candidates = (parameter: string) => options.filter(item => !graphConnectionIssue(graph, { source: item.node.id, sourcePort: item.port, target: selectedId || '', targetPort: parameter }, operators, types))
  const optionLabel = (item: AuthoringNode) => `${s('graph.stepOption', { index: graph.nodes.indexOf(item) + 1, name: nodeLabel(item, variables, operators) })}${types[item.id] || constantType(item) ? ` · ${graphTypeLabel(types[item.id] || constantType(item))}` : ''}`
  if (output) return <div className="space-y-4" aria-label={s('graph.outputSettings', {}, '输出设置')}>
    <h3 className="font-semibold text-slate-900">{isTimeSeries ? s('graph.outputChannel', {}, '输出通道') : b('nodeKinds.final')}</h3>
    <label className="block text-sm font-medium">{s('graph.outputSource')}<select className={field} aria-label={s('graph.outputSource')} value={output.node_id ? nodePortKey(output.node_id, output.port_id) : ''} onChange={event => onOutputChange(output.id, event.target.value ? { port_id: 'value', ...nodePortRef(event.target.value) } : { node_id: null, port_id: 'value' })}><option value="">{s('graph.selectOutput')}</option>{options.map(item => <option key={item.key} value={item.key}>{item.label}</option>)}</select></label>
    {isTimeSeries && <>
      <label className="block text-sm font-medium">{s('graph.outputName')}<input className={field} value={output.label} maxLength={80} onChange={event => onOutputChange(output.id, { label: event.target.value })} /></label>
      {<label className="block text-sm font-medium">{s('graph.outputId')}<input aria-label={s('graph.outputId')} aria-invalid={Boolean(outputIdError)} className={field} value={outputIdDraft} maxLength={70} onChange={event => setOutputIdDraft(event.target.value)} onBlur={commitOutputId} onKeyDown={event => { if (event.key === 'Enter') { event.preventDefault(); commitOutputId() } }} />{outputIdError && <span role="alert" className="mt-1 block text-xs text-rose-700">{s(outputIdError)}</span>}<span className="mt-1 block text-xs text-slate-500">{s('graph.outputIdentifierHint')}</span></label>}
      <label className="block text-sm font-medium">{s('graph.unit')}<input className={field} value={output.unit} maxLength={20} onChange={event => onOutputChange(output.id, { unit: event.target.value })} /></label>
      <label className="block text-sm font-medium">{s('graph.format')}<select className={field} value={output.display_format} onChange={event => onOutputChange(output.id, { display_format: event.target.value as 'number' | 'percent' })}><option value="number">{s('graph.formatNumber')}</option><option value="percent">{s('graph.formatPercent')}</option></select></label>
      <label className="block text-sm font-medium">{s('graph.precision')}<input className={field} type="number" min={0} max={8} value={output.precision} onChange={event => onOutputChange(output.id, { precision: Math.max(0, Math.min(8, Number(event.target.value))) })} /></label>
      <p className="text-xs leading-5 text-slate-500">{s('graph.outputPlacementHint')}</p>
      <button type="button" className={`${button} text-rose-700`} disabled={graph.outputs.length === 1} onClick={() => onRemove(outputNodeId(output.id))}>{s('graph.deleteOutput')}</button>
    </>}
  </div>
  if (!node) return <p className="text-sm leading-6 text-slate-500">{s('graph.selectStep')}</p>
  const usages = graphEdges(graph).filter(edge => edge.source === node.id).length
  const operator = node.kind === 'operator' ? operators.find(item => item.name === node.operator_id) : undefined
  const description = node.kind === 'operator'
    ? b(`operators.${node.operator_id}.description`, text(operator?.mathematical_essence || operator?.semantic))
    : node.kind === 'variable'
      ? b(`variables.${node.variable_id}.description`, text(variables.find(item => item.name === node.variable_id)?.description))
      : node.kind === 'parameter' ? s('indicatorParameters.parameterNode', { code: node.parameter_id })
      : s('graph.constantHint')
  const updateBinding = (name: string, binding?: GraphBinding) => {
    if (node.kind !== 'operator') return
    const argumentsNext = { ...node.arguments }
    if (binding) argumentsNext[name] = binding
    else delete argumentsNext[name]
    onNodeChange({ ...node, arguments: argumentsNext })
  }
  return <div className="space-y-4" aria-label={s('graph.stepSettings', {}, '步骤设置')}>
    <div><h3 className="font-semibold text-slate-900">{nodeLabel(node, variables, operators)}</h3><p className="mt-1 text-xs leading-5 text-slate-500">{description}</p></div>
    {usages > 1 && <p className="rounded-lg bg-amber-50 p-3 text-xs leading-5 text-amber-900">{s('graph.sharedHint', { count: usages })}</p>}
    <label className="block text-sm font-medium">{s('graph.note')}<input className={field} value={node.label || ''} maxLength={80} placeholder={s('graph.noteHint')} onChange={event => onNodeChange({ ...node, label: event.target.value })} /></label>
    {node.kind === 'variable' && <label className="block text-sm font-medium">{s('graph.variables')}<select className={field} value={node.variable_id} onChange={event => onNodeChange({ ...node, variable_id: event.target.value })}>{variables.map(variable => <option key={variable.name} value={variable.name} disabled={variable.availability === 'unavailable' || variable.availability === 'not_applicable'}>{graphVariableLabel(variable.name, variables)}</option>)}</select></label>}
    {node.kind === 'constant' && (typeof node.value === 'boolean'
      ? <label className="flex min-h-11 items-center gap-2"><input type="checkbox" checked={node.value} onChange={event => onNodeChange({ ...node, value: event.target.checked })} />{s('graph.booleanHint')}</label>
      : <label className="block text-sm font-medium">{s('graph.constantValue')}<input className={field} type="number" step="any" value={node.value ?? ''} onChange={event => onNodeChange({ ...node, value: event.target.value === '' ? null : Number(event.target.value) })} /></label>)}
    {node.kind === 'operator' && <>
      {new Set(operator?.parameter_sets?.map(set => set.arity)).size > 1 && <label className="block text-sm font-medium">{s('graph.parameterConfiguration')}<select aria-label={s('graph.parameterConfiguration')} className={field} value={node.arity ?? Object.keys(node.arguments).length} onChange={event => {
        const arity = Number(event.target.value)
        const parameters = parametersForNode({ ...node, arity }, operators)
        const argumentsNext = Object.fromEntries(parameters.flatMap<[string, GraphBinding]>(parameter => node.arguments[parameter.name] ? [[parameter.name, node.arguments[parameter.name]]] : typeof parameter.default === 'number' ? [[parameter.name, { source: 'constant', value: parameter.default }]] : []))
        onNodeChange({ ...node, arity, arguments: argumentsNext })
      }}>{[...new Set(operator?.parameter_sets?.map(set => set.arity))].sort((a, b) => a - b).map(arity => <option key={arity} value={arity}>{s('graph.parameterCount', { count: arity, names: operator?.parameter_sets?.find(set => set.arity === arity)?.parameters.map((parameter, index) => graphParameterLabel(parameter, index)).join('、') || '' })}</option>)}</select></label>}
      {parametersForNode(node, operators).map((parameter, index) => {
        const binding = node.arguments[parameter.name]
        const fixed = parameter.source_policy === 'fixed_constant'
        const constantNode = boundConstant(graph, binding)
        const constant = binding?.source === 'constant' || Boolean(constantNode)
        const value = constantNode ? constantNode.value : binding?.source === 'constant' ? binding.value : null
        const updateConstant = (value: number | boolean | null) => constantNode
          ? onNodeChange({ ...constantNode, value })
          : updateBinding(parameter.name, { source: 'constant', value })
        const constantUsages = constantNode ? graphEdges(graph).filter(edge => edge.source === constantNode.id).length : 0
        const label = graphParameterLabel(parameter, index, node.operator_id)
        const upstream = binding?.source === 'node' ? graph.nodes.find(item => item.id === binding.node_id) : undefined
        return <fieldset key={parameter.name} className="rounded-xl border border-slate-200 p-3"><legend className="px-1 text-sm font-semibold">{label}</legend>
          {parameter.description && <p className="text-xs leading-5 text-slate-500">{text(parameter.description)}</p>}
          <label className="block text-xs font-medium text-slate-600">{s('graph.inputSource')}<select aria-label={s('graph.parameterSource', { name: label })} className={field} value={binding?.source === 'node' ? nodePortKey(binding.node_id, binding.port_id) : constant ? '__constant__' : ''} onChange={event => updateBinding(parameter.name, event.target.value === '__constant__' ? { source: 'constant', value: null } : event.target.value ? { source: 'node', ...nodePortRef(event.target.value) } : undefined)}><option value="">{s(fixed ? 'graph.selectFixedConstant' : 'graph.selectInput')}</option><option value="__constant__">{s('graph.newConstant')}</option>{candidates(parameter.name).map(item => <option key={item.key} value={item.key}>{item.label}</option>)}</select></label>
          {upstream && <p aria-label={s('graph.currentInput', { name: label }, '{{name}}当前输入')} className="mt-2 break-words text-xs leading-5 text-slate-600">{s('graph.connected', { name: options.find(item => item.key === (binding?.source === 'node' ? nodePortKey(binding.node_id, binding.port_id) : ''))?.label || optionLabel(upstream) })}</p>}
          {upstream?.kind === 'parameter' && <p className="mt-2 text-xs text-cyan-800">{s('indicatorParameters.parameterNode', { code: upstream.parameter_id })}</p>}
          {upstream?.kind !== 'parameter' && (fixed || constant) && (typeof value === 'boolean'
            ? <label className="flex min-h-11 items-center gap-2"><input type="checkbox" checked={value} onChange={event => updateConstant(event.target.checked)} />{label}</label>
            : <input aria-label={s('graph.parameterConstant', { name: label })} className={field} type="number" step={parameter.constant_kind === 'integer' ? 1 : 'any'} min={parameter.minimum} max={parameter.maximum} value={value ?? ''} onChange={event => updateConstant(event.target.value === '' ? null : Number(event.target.value))} />)}
          {constant && <p className="mt-1 text-xs text-slate-600">{s('graph.constantSync', {}, '这里修改的是相连的常量节点，画布与参数同步更新。')}</p>}
          {constantUsages > 1 && <p className="mt-1 text-xs text-amber-800">{s('graph.sharedConstant', { count: constantUsages }, '此常量被 {{count}} 处引用，修改会同步影响这些输入。')}</p>}
          {fixed && upstream?.kind !== 'parameter' && <p className="mt-1 text-xs text-slate-500">{s(parameter.constant_kind === 'integer' ? 'graph.fixedInteger' : 'graph.fixedNumber')}{parameter.minimum !== undefined ? ` · ${s('graph.minimum', { value: parameter.minimum })}` : ''}{parameter.maximum !== undefined ? ` · ${s('graph.maximum', { value: parameter.maximum })}` : ''}</p>}
        </fieldset>
      })}
    </>}
    {types[node.id] && <div className="rounded-lg bg-sky-50 p-3 text-xs leading-5 text-sky-900">{s('graph.typeChecked', { type: graphTypeLabel(types[node.id]) })}{graphAxesLabel(types[node.id]) ? ` · ${s('graph.alignment', { axes: graphAxesLabel(types[node.id]) })}` : ''}</div>}
    <div className="flex flex-wrap gap-2"><button type="button" className={button} onClick={() => onDuplicate(node.id)}>{s('graph.copyStep')}</button><button type="button" className={`${button} text-rose-700`} onClick={() => onRemove(node.id)}>{s('graph.deleteStep')}</button></div>
    <p className="text-xs leading-5 text-slate-500">{s('graph.deleteHint')}</p>
  </div>
}

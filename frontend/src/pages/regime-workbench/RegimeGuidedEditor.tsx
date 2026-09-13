import { useState } from 'react'
import CalculationSteps from '../../components/computation-graph/CalculationSteps'
import RegimeSeriesOutputs from './RegimeSeriesOutputs'
import RegimeGranularityInfo from './RegimeGranularityInfo'
import type { RegimeGraphDefinition, RegimeGraphConnection, RegimeGraphNode, RegimeGraphTemplate, RegimeNodeSchema } from '../../services/regimeGraph'
import RegimeResearchSeriesPicker, { hasResearchSeriesPicker, isEditableSourceParameter, researchSourceParameterPatch, researchSeriesFieldOptions, researchSeriesPickerKey } from './RegimeResearchSeriesPicker'
import { ConnectionEditor, ParameterInput, regimeNodeParameterValue, regimeParameterIsActive } from './RegimeNodeInspector'
import RegimeManualEventEditor from './RegimeManualEventEditor'

export interface RegimeGuidedEditorProps {
  builderOnly?: boolean
  onExpandNode?: (id: string) => void
  expanding?: boolean
  expandDisabled?: boolean
  selectedNodeId?: string | null
  onSelectNode?: (id: string) => void
  active?: boolean
  definition: RegimeGraphDefinition
  schemas: RegimeNodeSchema[]
  templates: RegimeGraphTemplate[]
  selectedTemplate: string
  loadingTemplate: boolean
  onTemplate: (id: string) => void
  onLoadTemplate: () => void
  onBlank: () => void
  onPatchNode: (id: string, patch: Partial<RegimeGraphNode>) => void
  onPreviewNode?: (id: string) => void
  onEditNode: (id: string) => void
  onOpenStates: () => void
  onOpenDataLab: () => void
  onAddNode?: () => void
  onRemoveNode?: (id: string) => void
  onSetState?: (reference: RegimeGraphConnection | null) => void
  onChange?: (definition: RegimeGraphDefinition) => void
  onRun: () => void
  canRun: boolean
}

export default function RegimeGuidedEditor(props: RegimeGuidedEditorProps) {
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const nodes = props.definition.graph.nodes
  const active = nodes.find(node => node.id === (props.selectedNodeId === undefined ? selectedId : props.selectedNodeId)) || nodes[0]
  const schema = props.schemas.find(item => (item.id || item.type) === active?.type)
  const properties = schema?.parameter_schema?.properties || schema?.parameters || {}
  const source = active?.type.startsWith('source.') && active.type !== 'source.constant'
  const entries = active ? Object.entries(properties).filter(([name, parameter]) => !parameter.deprecated && regimeParameterIsActive(active, name) && isEditableSourceParameter(active, name)) : []
  const kind = (type: string) => props.schemas.find(item => item.id === type)?.indicator_reference ? '指标计算' : type === 'source.constant' ? '常量' : type === 'source.indicator' ? '已有指标' : type.startsWith('source.') ? '输入变量' : '计算算子'
  return <div className="mx-auto w-full max-w-6xl space-y-5 p-3 sm:p-5" aria-label="历史情景构建引导">
    {!props.builderOnly && <><div><h2 className="text-lg font-semibold text-slate-950">时序计算逻辑</h2><p className="mt-1 text-xs leading-5 text-slate-600">选择输入，组合计算步骤，再指定输出。与画布和高级公式同步保存。</p></div>
    <details open={!nodes.length} className="rounded-xl border border-slate-200 bg-white p-4"><summary className="cursor-pointer text-sm font-semibold text-slate-700">从模板开始</summary><div className="mt-3 flex flex-wrap gap-2"><select aria-label="图谱模板" value={props.selectedTemplate} onChange={event => props.onTemplate(event.target.value)} className="min-h-10 min-w-0 flex-1 rounded-xl border border-slate-200 bg-white px-2 text-sm"><option value="">选择模板</option>{props.templates.map(template => <option key={template.id} value={template.id}>{template.default_mode === 'retrospective' ? '事后 · ' : ''}{template.name}</option>)}</select><button type="button" disabled={!props.selectedTemplate || props.loadingTemplate} onClick={props.onLoadTemplate} className="min-h-10 rounded-lg bg-accent-600 px-3 text-xs font-semibold text-white disabled:opacity-40">{props.loadingTemplate ? '载入中…' : '载入模板'}</button><button type="button" onClick={props.onBlank} className="min-h-10 rounded-lg border border-slate-200 px-3 text-xs">新建空白</button></div><p className="mt-2 text-xs text-slate-500">{props.templates.find(template => template.id === props.selectedTemplate)?.description}</p></details></>}
    <section aria-label="构建计算步骤" className="rounded-xl border border-slate-200 bg-white p-4">
      <div className="mb-4 flex flex-wrap items-center justify-between gap-2"><div><h3 className="text-sm font-semibold text-slate-900">输入变量与计算步骤</h3><p className="mt-1 text-xs text-slate-600">选择一个步骤修改；共享输入只计算一次。</p></div><button type="button" onClick={props.onAddNode} className="min-h-10 rounded-lg border border-accent-200 px-3 text-xs font-semibold text-accent-700">添加数据、指标或算子</button></div>
      <CalculationSteps compact={props.builderOnly} steps={nodes.map(node => ({ id: node.id, label: node.label || props.schemas.find(item => (item.id || item.type) === node.type)?.label || node.id, kind: kind(node.type) }))} selectedId={active?.id || null} onSelect={id => { setSelectedId(id); props.onSelectNode?.(id) }}>
        {active ? <article className="rounded-xl bg-slate-50 p-4"><div className="flex items-start justify-between gap-3"><div><h4 className="text-sm font-semibold text-slate-900">{schema?.label || active.type}</h4><p className="mt-1 text-xs leading-5 text-slate-600">{schema?.description}</p></div><button type="button" onClick={() => props.onEditNode(active.id)} className="shrink-0 text-xs font-semibold text-accent-700">在画布定位</button></div>
          <RegimeGranularityInfo schema={schema} onExpand={props.onExpandNode ? () => props.onExpandNode?.(active.id) : undefined} expanding={props.expanding} disabled={props.expandDisabled} />
          <label className="mt-4 block text-xs font-semibold text-slate-600">步骤名称<input aria-label="步骤名称" value={active.label || ''} onChange={event => props.onPatchNode(active.id, { label: event.target.value })} className="mt-1 min-h-10 w-full rounded-xl border border-slate-200 bg-white px-2" /></label>
          {source ? <div className="mt-4 space-y-3"><p className="rounded-lg bg-accent-50 p-3 text-xs leading-5 text-accent-800">此变量提供计算所需的时间序列。数据绑定可单独调整，不改变下游计算关系。</p>{hasResearchSeriesPicker(active) ? <RegimeResearchSeriesPicker key={researchSeriesPickerKey(active)} node={active} schema={schema} onPatchNode={patch => props.onPatchNode(active.id, patch)}>{series => <div className="grid gap-3 sm:grid-cols-2">{entries.map(([name, parameter]) => <ParameterInput key={name} name={name} schema={parameter} value={regimeNodeParameterValue(active, name)} options={name === 'field' && researchSeriesFieldOptions(series).length ? researchSeriesFieldOptions(series) : undefined} onChange={value => props.onPatchNode(active.id, researchSourceParameterPatch(active, name, value, series))} />)}</div>}</RegimeResearchSeriesPicker> : <details><summary className="cursor-pointer text-sm font-semibold text-slate-700">数据绑定与字段设置</summary><div className="mt-3 grid gap-3 sm:grid-cols-2">{entries.map(([name, parameter]) => <ParameterInput key={name} name={name} schema={parameter} value={regimeNodeParameterValue(active, name)} onChange={value => props.onPatchNode(active.id, { parameters: { ...active.parameters, [name]: value } })} />)}</div><button type="button" onClick={props.onOpenDataLab} className="mt-3 min-h-10 text-xs font-semibold text-accent-700">从数据实验室选择序列</button></details>}</div> : <>
          {active.type === 'annotation.manual_events'
            ? <div className="mt-4"><RegimeManualEventEditor value={active.parameters.events} onChange={events => props.onPatchNode(active.id, { parameters: { ...active.parameters, events } })} /></div>
            : <div className="mt-4 grid gap-3 sm:grid-cols-2">{entries.map(([name, parameter]) => <ParameterInput key={name} name={name} states={props.definition.states} schema={parameter} value={regimeNodeParameterValue(active, name)} onChange={value => props.onPatchNode(active.id, { parameters: { ...active.parameters, [name]: value } })} />)}</div>}
          <div className="mt-4 grid gap-3 sm:grid-cols-2">{schema?.inputs?.map(port => <ConnectionEditor key={port.id} portId={port.id} label={port.label || port.id} connection={active.inputs[port.id]} node={active} nodes={nodes} schemas={props.schemas} onConnect={(name, reference) => { const inputs = { ...active.inputs }; if (reference) inputs[name] = reference; else delete inputs[name]; props.onPatchNode(active.id, { inputs }) }} />)}</div>
          </>}
          {props.onPreviewNode && <button type="button" onClick={() => props.onPreviewNode?.(active.id)} className="mt-4 mr-3 min-h-10 rounded-lg bg-accent-600 px-3 text-xs font-semibold text-white">预览此节点</button>}
          <button type="button" aria-label={`删除步骤${active.id}`} onClick={() => props.onRemoveNode?.(active.id)} className="mt-4 min-h-10 text-xs text-rose-600">删除此步骤</button>
        </article> : <p className="p-4 text-sm text-slate-600">添加资源后，在这里编辑变量、常量和计算输入。</p>}
      </CalculationSteps>
    </section>
    {!props.builderOnly && <><div className="rounded-xl border border-slate-200 bg-white p-4">{props.onChange && props.active !== false ? <RegimeSeriesOutputs definition={props.definition} schemas={props.schemas} onChange={props.onChange} /> : <button type="button" onClick={props.onOpenStates}>配置输出通道</button>}</div>
    <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl bg-accent-50 p-4"><p className="text-xs leading-5 text-accent-800">定义完成后选择实时或事后模式及截至日，试算查看状态色带与区间明细。</p><button type="button" disabled={!props.canRun} onClick={props.onRun} className="min-h-11 rounded-lg bg-accent-600 px-4 text-sm font-semibold text-white disabled:opacity-40">运行当前方案</button></div></>}
  </div>
}

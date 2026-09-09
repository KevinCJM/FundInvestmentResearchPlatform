import { useEffect, useMemo, useState } from 'react'
import type {
  RegimeGraphConnection,
  RegimeGraphDefinition,
  RegimeGraphInference,
  RegimeGraphNode,
  RegimeNodeSchema,
  RegimeParameterSchema,
  PreparedRegimeGraph,
} from '../../services/regimeGraph'
import type { ResearchSeriesCatalogItem } from '../../services/researchSeries'
import RegimeResearchSeriesPicker, { hasResearchSeriesPicker, isEditableSourceParameter, researchSourceParameterPatch, researchSeriesFieldOptions, researchSeriesPickerKey } from './RegimeResearchSeriesPicker'
import RegimeHelpTip from './RegimeHelpTip'
import RegimeStructuredParameter from './RegimeStructuredParameter'
import { regimeConnectionIssue } from './regimeGraphEditing'
import { regimeEnumLabel, regimeParameterHelp, regimeParameterLabel, regimePhaseLabel, regimePortLabel, regimeTypeLabel } from './regimeDisplay'

function parameterProperties(schema?: RegimeNodeSchema) {
  return schema?.parameter_schema?.properties ?? schema?.parameters ?? {}
}

export function regimeParameterIsActive(node: RegimeGraphNode | null, name: string) {
  if (node?.type !== 'model.range_threshold') return true
  const bound = name === 'upper' ? 'upper_bound' : name === 'lower' ? 'lower_bound' : null
  return !bound || !node.inputs[bound]
}

export function regimeNodeParameterValue(node: RegimeGraphNode, name: string): unknown {
  if (node.type === 'model.peak_trough' && node.parameters[name] === undefined) {
    if (name === 'left_window' || name === 'right_window') return node.parameters.window
    if (name === 'head_window' || name === 'tail_window') return node.parameters.endpoint_window
  }
  return node.parameters[name]
}

function displayValue(value: unknown) {
  if (typeof value === 'string') return value
  return JSON.stringify(value ?? null, null, 2)
}

function StructuredInput({ value, schema, onChange }: { value: unknown; schema: RegimeParameterSchema; onChange: (value: unknown) => void }) {
  const [draft, setDraft] = useState(displayValue(value))
  const [error, setError] = useState('')
  useEffect(() => { setDraft(displayValue(value)); setError('') }, [value])
  const commit = () => {
    try { onChange(JSON.parse(draft)); setError('') } catch { setError('请输入合法 JSON。') }
  }
  return <div><textarea aria-label={schema.label || schema.title || '结构化参数'} value={draft} onChange={(event) => setDraft(event.target.value)} onBlur={commit} rows={4} className="mt-1 w-full rounded-lg border border-slate-300 px-2 py-2 font-mono text-xs focus:border-indigo-500 focus:outline-none" />{error ? <p role="alert" className="mt-1 text-[11px] text-rose-700">{error}</p> : null}</div>
}

export function ParameterInput({ name, schema, value, options, onChange }: { name: string; schema: RegimeParameterSchema; value: unknown; options?: Array<{ value: string; label: string; disabled?: boolean }>; onChange: (value: unknown) => void }) {
  const label = regimeParameterLabel(name, schema)
  const help = regimeParameterHelp(name, schema)
  const choices: Array<{ value: string; label: string; disabled?: boolean }> | undefined = options || schema.enum?.map((option, index) => ({ value: String(option), label: regimeEnumLabel(option, schema, index) }))
  const heading = <>{label}<RegimeHelpTip label={`${label}说明`} text={help} /></>
  if (choices?.length) return <label className="block text-xs font-bold text-slate-600">{heading}<select aria-label={label} value={String(value ?? schema.default ?? '')} onChange={(event) => { const index = schema.enum?.findIndex((item) => String(item) === event.target.value) ?? -1; onChange(index >= 0 ? schema.enum?.[index] : event.target.value) }} className="mt-1 min-h-10 w-full rounded-lg border border-slate-300 bg-white px-2 font-normal"><option value="">请选择</option>{choices.map((option) => <option key={option.value} value={option.value} disabled={option.disabled}>{option.label}</option>)}</select></label>
  if (schema.type === 'boolean') return <label className="flex min-h-10 items-center gap-2 rounded-lg border border-slate-200 px-3 text-xs font-bold text-slate-700"><input aria-label={label} type="checkbox" checked={Boolean(value ?? schema.default)} onChange={(event) => onChange(event.target.checked)} />{heading}</label>
  if (schema.type === 'array' || schema.type === 'object') { const current = value ?? schema.default ?? (schema.type === 'array' ? [] : {}); return <div className="block space-y-2 text-xs font-bold text-slate-600"><p>{heading}</p><RegimeStructuredParameter label={label} value={current} onChange={onChange} /><details><summary className="cursor-pointer text-[11px] font-semibold text-slate-500">高级：编辑完整 JSON</summary><StructuredInput value={current} schema={{ ...schema, label }} onChange={onChange} /></details></div> }
  if (schema.type === 'number' || schema.type === 'integer') return <label className="block text-xs font-bold text-slate-600">{heading}<input aria-label={label} type="number" value={value === null ? '' : typeof value === 'number' ? value : Number(schema.default ?? 0)} min={schema.minimum} max={schema.maximum} step={schema.step ?? (schema.type === 'integer' ? 1 : 'any')} onChange={(event) => onChange(event.target.value === '' ? null : Number(event.target.value))} className="mt-1 min-h-10 w-full rounded-lg border border-slate-300 px-2 font-normal" /></label>
  return <label className="block text-xs font-bold text-slate-600">{heading}<input aria-label={label} value={String(value ?? schema.default ?? '')} placeholder={schema.placeholder} onChange={(event) => onChange(event.target.value)} className="mt-1 min-h-10 w-full rounded-lg border border-slate-300 px-2 font-normal" /></label>
}

function FormulaInput({ label, value, variables, onChange }: { label: string; value: unknown; variables: string[]; onChange: (value: string) => void }) {
  return <label className="block text-xs font-bold text-slate-600">{label}<RegimeHelpTip label={`${label}说明`} text="用可用输入和已登记的算子定义时序计算。检查会指出不支持的函数、数据类型或非因果操作。" /><textarea aria-label={label} value={String(value ?? '')} onChange={(event) => onChange(event.target.value)} rows={6} spellCheck={false} placeholder="例如：rolling_mean(feature_1, 20)" className="mt-1 w-full rounded-lg border border-slate-300 px-2 py-2 font-mono text-xs leading-5 focus:border-indigo-500 focus:outline-none" /><span className="mt-1 block text-[10px] font-normal leading-4 text-slate-500">受限因果公式语言 · 可用输入：{variables.length ? variables.map((item) => regimePortLabel({ id: item })).join('、') : '由输入端口决定'}。运行前会自动检查并准备计算。</span></label>
}

const OUTPUT_SLOTS = [
  ['state', '状态序列'],
  ['probabilities', '状态概率'],
  ['confidence', '置信度'],
  ['recognition_index', '识别索引'],
  ['effective_index', '生效索引'],
  ['reason_code', '原因码'],
] as const

function typedDagNodes(plan: NonNullable<PreparedRegimeGraph['formula_plans']>[string] | undefined) {
  const nodes = plan?.typed_expression?.nodes
  if (Array.isArray(nodes)) return nodes
  return nodes && typeof nodes === 'object' ? Object.values(nodes) : []
}

function outputPorts(node: RegimeGraphNode | undefined, schemas: RegimeNodeSchema[]) {
  if (!node) return []
  return schemas.find((item) => item.id === node.type || item.type === node.type)?.outputs ?? []
}

export function ConnectionEditor({
  portId,
  label,
  connection,
  node,
  nodes,
  schemas,
  onConnect,
}: {
  portId: string
  label: string
  connection?: RegimeGraphConnection
  node: RegimeGraphNode
  nodes: RegimeGraphNode[]
  schemas: RegimeNodeSchema[]
  onConnect: (port: string, value: RegimeGraphConnection | null) => void
}) {
  const compatiblePorts = (candidate: RegimeGraphNode | undefined) => outputPorts(candidate, schemas).filter((port) => candidate && !regimeConnectionIssue(nodes, schemas, { source: candidate.id, sourcePort: port.id, target: node.id, targetPort: portId }))
  const source = nodes.find((item) => item.id === connection?.node_id)
  const ports = outputPorts(source, schemas)
  const issue = source && connection ? regimeConnectionIssue(nodes, schemas, { source: source.id, sourcePort: connection.port, target: node.id, targetPort: portId }) : null
  return <div className="rounded-lg border border-slate-200 p-2"><label className="block text-[11px] font-bold text-slate-600">{label}<RegimeHelpTip label={`${label}输入说明`} text="选择类型兼容且不会形成循环的上游结果。不可选项会说明原因。" /><select aria-label={`${label}上游节点`} value={connection?.node_id || ''} onChange={(event) => { const next = nodes.find((item) => item.id === event.target.value); const firstPort = compatiblePorts(next)[0]; onConnect(portId, next && firstPort ? { node_id: next.id, port: firstPort.id } : null) }} className="mt-1 min-h-9 w-full rounded-md border border-slate-300 bg-white px-2 font-normal"><option value="">未连接</option>{nodes.filter((item) => item.id !== node.id).map((item) => { const eligible = compatiblePorts(item); const first = outputPorts(item, schemas)[0]; const reason = !eligible.length ? first ? regimeConnectionIssue(nodes, schemas, { source: item.id, sourcePort: first.id, target: node.id, targetPort: portId }) : '没有输出' : null; return <option key={item.id} value={item.id} disabled={!eligible.length}>{item.label || schemas.find((schema) => schema.id === item.type || schema.type === item.type)?.label || item.id}{reason ? `（${reason}）` : ''}</option> })}</select></label>{source && ports.length > 1 ? <label className="mt-2 block text-[11px] font-bold text-slate-600">输出端口<RegimeHelpTip label="输出端口说明" text="选择上游节点中与当前输入兼容的结果。" /><select aria-label={`${label}输出端口`} value={connection?.port || ''} onChange={(event) => onConnect(portId, { node_id: source.id, port: event.target.value })} className="mt-1 min-h-9 w-full rounded-md border border-slate-300 bg-white px-2 font-normal">{ports.map((port) => { const reason = regimeConnectionIssue(nodes, schemas, { source: source.id, sourcePort: port.id, target: node.id, targetPort: portId }); return <option key={port.id} value={port.id} disabled={Boolean(reason)}>{regimePortLabel(port)}{reason ? `（${reason}）` : ''}</option> })}</select></label> : null}{issue ? <p role="alert" className="mt-2 text-[11px] text-amber-800">当前连接{issue}，请更换或断开。</p> : null}</div>
}

export interface RegimeNodeInspectorProps {
  node: RegimeGraphNode | null
  schema?: RegimeNodeSchema
  nodes: RegimeGraphNode[]
  schemas: RegimeNodeSchema[]
  outputs: RegimeGraphDefinition['graph']['outputs']
  inference: RegimeGraphInference | null
  preparedPlan: PreparedRegimeGraph | null
  onPatchNode: (patch: Partial<RegimeGraphNode>) => void
  onConnect: (port: string, connection: RegimeGraphConnection | null) => void
  onSetOutput: (slot: typeof OUTPUT_SLOTS[number][0], port: string | null) => void
  onRemove: () => void
}

export default function RegimeNodeInspector({ node, schema, nodes, schemas, outputs, inference, preparedPlan, onPatchNode, onConnect, onSetOutput, onRemove }: RegimeNodeInspectorProps) {
  const [selectedDagNode, setSelectedDagNode] = useState<string>('')
  const properties = parameterProperties(schema)
  const required = new Set(schema?.parameter_schema?.required ?? [])
  const inferred = node ? inference?.inferred?.nodes?.[node.id] : undefined
  const executionLabel = schema?.execution_policy?.third_party_exempt
    ? '第三方优化模型隔离'
    : schema?.execution_policy?.njit_required
      ? '固定签名 NJIT'
      : schema?.execution_policy?.execution_lane === 'data_boundary'
        ? '数据读取边界'
        : '执行策略待接口确认'
  const parameterEntries = useMemo(() => Object.entries(properties).filter(([name, parameter]) => !parameter.deprecated && regimeParameterIsActive(node, name) && (!node || isEditableSourceParameter(node, name))), [properties, node])
  const formulaPlan = node ? preparedPlan?.formula_plans?.[node.id] : undefined
  const dagNodes = typedDagNodes(formulaPlan)
  const dagEdges = formulaPlan?.typed_expression?.edges ?? []
  useEffect(() => { setSelectedDagNode('') }, [node?.id, formulaPlan?.expression_hash])
  if (!node) return <aside className="grid min-h-[420px] place-items-center rounded-2xl border border-dashed border-slate-300 bg-white p-6 text-center"><div><p className="text-sm font-bold text-slate-800">节点检查器 <RegimeHelpTip label="节点检查器说明" text="选择画布中的任一节点后，可在这里挑选数据、调整参数、连接上游并查看该节点的计算约束。" /></p><p className="mt-2 text-xs leading-5 text-slate-500">选择画布节点后，可按服务端说明编辑参数、连接输入并指定图输出。</p></div></aside>
  const renderParameters = (series?: ResearchSeriesCatalogItem) => (<section><h4 className="text-xs font-bold text-slate-800">{hasResearchSeriesPicker(node) ? '数据设置' : '参数'} <RegimeHelpTip label="节点参数说明" text="这些参数直接决定该节点的计算方式。调整后旧试算会标记为过期，需要重新试算。" /></h4><div className="mt-2 space-y-3">{parameterEntries.length ? parameterEntries.map(([name, parameter]) => name === schema?.formula_language?.expression_parameter ? <FormulaInput key={name} label={`${regimeParameterLabel(name, parameter)}${required.has(name) ? ' *' : ''}`} value={regimeNodeParameterValue(node, name)} variables={schema.formula_language?.variables || schema.inputs.map((port) => port.id)} onChange={(value) => onPatchNode(researchSourceParameterPatch(node, name, value, series))} /> : <ParameterInput key={name} name={name} schema={{ ...parameter, label: `${regimeParameterLabel(name, parameter)}${required.has(name) ? ' *' : ''}` }} value={regimeNodeParameterValue(node, name)} options={name === 'field' && researchSeriesFieldOptions(series).length ? researchSeriesFieldOptions(series) : undefined} onChange={(value) => onPatchNode(researchSourceParameterPatch(node, name, value, series))} />) : <p className="rounded-lg bg-slate-50 p-3 text-xs text-slate-500">该节点没有可编辑参数。</p>}</div></section>)
  return (
    <aside className="min-h-[420px] space-y-4 rounded-2xl border border-slate-200 bg-white p-4 shadow-sm" aria-label="节点动态检查器">
      <div className="flex items-start justify-between gap-2"><div><p className="text-[10px] font-bold tracking-wider text-indigo-600">{schema?.category_label || '计算节点'}</p><h3 className="mt-1 text-base font-bold text-slate-950">{schema?.label || '未命名节点'}</h3><p className="mt-1 text-[11px] leading-5 text-slate-500">{schema?.description || '该节点的计算说明由服务端节点目录提供。'}</p></div><button type="button" onClick={onRemove} className="min-h-9 shrink-0 whitespace-nowrap rounded-lg px-2 text-xs font-bold text-rose-600 hover:bg-rose-50">删除</button></div>
      <label className="block text-xs font-bold text-slate-600">节点名称<RegimeHelpTip label="节点名称说明" text="仅改变画布上的显示名称，不会改变节点算法、输入数据或计算结果。" /><input aria-label="节点名称" value={node.label || ''} placeholder={schema?.label || '请输入节点名称'} onChange={(event) => onPatchNode({ label: event.target.value })} className="mt-1 min-h-10 w-full rounded-lg border border-slate-300 px-2 font-normal" /></label>
      {schema?.indicator_reference && <p className="rounded-xl bg-indigo-50 p-3 text-xs leading-5 text-indigo-900">指标中心 · 第 {schema.indicator_reference.revision} 版。连接下方所需输入即可计算，输出时间轴来自上游数据。{schema.indicator_reference.result_kind === 'scalar' ? '该单值指标按滚动窗口逐期计算。' : `提供 ${schema.outputs.length} 个输出，可分别连接下游。`}</p>}
      {hasResearchSeriesPicker(node) && <RegimeResearchSeriesPicker key={researchSeriesPickerKey(node)} node={node} schema={schema} onPatchNode={onPatchNode}>{renderParameters}</RegimeResearchSeriesPicker>}
      <details><summary className="cursor-pointer text-xs font-bold text-slate-600">高级：类型与执行记录</summary><div className="mt-3 space-y-3">
      <div className="flex flex-wrap gap-2 text-[10px] font-bold"><span className={`rounded-full px-2 py-1 ${schema?.execution_policy?.njit_required ? 'bg-emerald-100 text-emerald-800' : schema?.execution_policy?.third_party_exempt ? 'bg-violet-100 text-violet-800' : 'bg-slate-100 text-slate-600'}`}>{executionLabel}</span>{inferred?.causal != null ? <span className={`rounded-full px-2 py-1 ${inferred.causal ? 'bg-sky-100 text-sky-800' : 'bg-amber-100 text-amber-900'}`}>{inferred.causal ? '因果节点' : '可能使用未来数据'}</span> : null}</div>
      <section aria-label="节点执行契约"><h4 className="text-xs font-bold text-slate-800">类型与执行契约 <RegimeHelpTip label="执行契约说明" text="说明节点能接收和输出什么数据、是否可用于实时研究，以及普通数值逻辑是否进入固定签名 NJIT 高性能通道。" /></h4><dl className="mt-2 grid grid-cols-2 gap-2 text-[10px]"><div className="rounded-lg bg-slate-50 p-2"><dt className="text-slate-500">节点版本</dt><dd className="mt-1 font-bold text-slate-800">第 {node.type_version ?? schema?.type_version ?? schema?.version ?? '—'} 版</dd></div><div className="rounded-lg bg-slate-50 p-2"><dt className="text-slate-500">研究阶段 / 实时</dt><dd className="mt-1 font-bold text-slate-800">{regimePhaseLabel(schema?.phase)} / {schema?.supports_realtime == null ? '待确认' : schema.supports_realtime ? '支持实时' : '仅事后研究'}</dd></div><div className="rounded-lg bg-slate-50 p-2"><dt className="text-slate-500">因果 / 重绘</dt><dd className="mt-1 font-bold text-slate-800">{schema?.causal == null ? '待确认' : schema.causal ? '仅使用当时信息' : '可能使用未来信息'} / {schema?.repaints == null ? '待确认' : schema.repaints ? '历史结果会变化' : '历史结果固定'}</dd></div><div className="rounded-lg bg-slate-50 p-2"><dt className="text-slate-500">最少样本</dt><dd className="mt-1 font-bold text-slate-800">{schema?.minimum_samples ?? '待确认'} 期</dd></div><div className="rounded-lg bg-slate-50 p-2"><dt className="text-slate-500">计算引擎</dt><dd className="mt-1 font-bold text-slate-800">{schema?.execution_policy?.njit_required ? '固定签名 NJIT 内核' : schema?.execution_policy?.third_party_exempt ? '隔离的第三方优化模型' : '数据读取边界'}</dd></div><div className="rounded-lg bg-slate-50 p-2"><dt className="text-slate-500">模型版本 / 计算量</dt><dd className="mt-1 font-bold text-slate-800">{schema?.model_version ? `第 ${schema.model_version} 版` : '不适用'} / {schema?.cost_estimate?.class === 'linear' ? '随样本量线性增长' : schema?.cost_estimate?.class === 'io_bound' ? '主要受数据读取影响' : '由模型声明'}</dd></div></dl><div className="mt-2 grid gap-2 text-[10px] sm:grid-cols-2"><p className="rounded-lg border border-slate-200 p-2"><strong className="block text-slate-700">输入类型 <RegimeHelpTip label="输入类型说明" text="输入端口必须接入对应的数据结构。这里展示业务名称；系统内部仍用稳定类型编号做严格校验。" /></strong><span className="mt-1 block text-slate-500">{schema?.inputs.map((port) => `${regimePortLabel(port)}：${regimeTypeLabel(port.value_type || port.type, port.type_label)}`).join('；') || '无输入'}</span></p><p className="rounded-lg border border-slate-200 p-2"><strong className="block text-slate-700">输出类型 <RegimeHelpTip label="输出类型说明" text="输出类型决定它能连接到哪些下游节点，也决定该结果能否被指定为最终状态、概率或置信度。" /></strong><span className="mt-1 block text-slate-500">{schema?.outputs.map((port) => `${regimePortLabel(port)}：${regimeTypeLabel(port.value_type || port.type, port.type_label)}`).join('；') || '无输出'}</span></p></div></section>
      </div></details>
      {schema?.inputs?.length ? <section><h4 className="text-xs font-bold text-slate-800">输入连接 <RegimeHelpTip label="输入连接说明" text="为节点每个必需输入选择上游结果。带星号的输入必须连接，否则图谱检查不会通过。" /></h4><div className="mt-2 space-y-2">{schema.inputs.map((port) => <ConnectionEditor key={port.id} portId={port.id} label={`${regimePortLabel(port)}${port.required ? ' *' : ''}`} connection={node.inputs[port.id]} node={node} nodes={nodes} schemas={schemas} onConnect={onConnect} />)}</div></section> : null}
      {!hasResearchSeriesPicker(node) && (!schema?.indicator_reference || parameterEntries.length > 0) && renderParameters()}
      {schema?.formula_language ? <section aria-label="公式语法树计算图审计" className="rounded-xl border border-indigo-200 bg-indigo-50/50 p-3"><h4 className="text-xs font-bold text-indigo-950">公式与内部计算图 <RegimeHelpTip label="公式内部计算图说明" text="把公式拆成可检查的小算子，展示每一步的数据类型和连接关系，用于确认公式没有绕过白名单或高性能执行约束。" /></h4>{formulaPlan ? <div className="mt-2 space-y-2"><p className="text-[10px] text-indigo-900">已生成并锁定编译计划；定义变化后需重新预热。</p><div className="flex max-h-40 flex-wrap gap-1 overflow-y-auto">{dagNodes.map((item, index) => { const id = String(item.id ?? index); const fragment = String(item.formula_fragment ?? item.label ?? id); return <button type="button" key={id} onClick={() => setSelectedDagNode(id)} className={`rounded border px-2 py-1 text-left font-mono text-[9px] ${selectedDagNode === id ? 'border-indigo-500 bg-indigo-100 text-indigo-950' : 'border-indigo-100 bg-white text-slate-700'}`}><strong>步骤 {index + 1}</strong> {fragment}</button> })}</div>{selectedDagNode ? (() => { const item = dagNodes.find((entry, index) => String(entry.id ?? index) === selectedDagNode); return item ? <dl className="grid grid-cols-2 gap-2 rounded-lg bg-white p-2 text-[9px]"><div><dt className="text-slate-500">公式片段</dt><dd className="mt-1 break-all font-mono text-slate-800">{String(item.formula_fragment ?? '—')}</dd></div><div><dt className="text-slate-500">计算步骤 / 数据类型</dt><dd className="mt-1 text-slate-800">已注册公式算子 / {regimeTypeLabel(typeof item.inferred_type === 'string' ? item.inferred_type : '')}</dd></div></dl> : null })() : null}<p className="text-[9px] text-slate-500">{dagNodes.length} 个带类型计算步骤 · {dagEdges.length} 条内部连线 · 计划与公式内容绑定。</p></div> : <p className="mt-2 text-[10px] leading-4 text-indigo-900">图谱检查只做语义验证；点击“运行识别”或正式实验区的“显式预热计划”后，服务端才会返回内部语法树、计算图、成本和编译凭证。</p>}</section> : null}
      {schema?.indicator_reference && <section><h4 className="text-xs font-bold text-slate-800">计算结果</h4><div className="mt-2 flex flex-wrap gap-2">{schema.outputs.map(port => <span key={port.id} className="rounded-lg bg-indigo-50 px-3 py-2 text-xs text-indigo-800">{regimePortLabel(port)}</span>)}</div><p className="mt-2 text-xs leading-5 text-slate-500">可在节点预览中选择结果，或在画布上连接到下游输入。</p></section>}
      {schema?.outputs?.some(port => OUTPUT_SLOTS.some(([slot]) => port.id === slot || port.name === slot)) ? <section><h4 className="text-xs font-bold text-slate-800">指定计算图输出 <RegimeHelpTip label="计算图输出说明" text="告诉系统最终状态、概率、置信度和识别时点分别来自哪个节点。只有类型相符的端口才会出现。" /></h4><div className="mt-2 space-y-2">{OUTPUT_SLOTS.map(([slot, label]) => { const current = outputs[slot]; const selectedPort = current?.node_id === node.id ? current.port : ''; const compatible = schema.outputs.filter((port) => port.id === slot || port.name === slot); return <label key={slot} className="block text-[11px] font-bold text-slate-600">{label}<RegimeHelpTip label={`${label}输出说明`} text={`选择当前节点是否提供最终的${label}。留空表示由其他节点提供。`} /><select aria-label={`${label}图输出`} value={selectedPort} onChange={(event) => onSetOutput(slot, event.target.value || null)} className="mt-1 min-h-9 w-full rounded-md border border-slate-300 bg-white px-2 font-normal"><option value="">不由本节点输出</option>{compatible.map((port) => <option key={port.id} value={port.id}>{regimePortLabel(port)}</option>)}</select></label> })}</div></section> : null}
    </aside>
  )
}

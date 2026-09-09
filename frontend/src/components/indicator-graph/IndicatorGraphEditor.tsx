import CalculationSteps from '../computation-graph/CalculationSteps'
import { forwardRef, useEffect, useImperativeHandle, useMemo, useRef, useState } from 'react'
import katex from 'katex'
import { useI18n } from '../../i18n/runtime'
import { builtinCatalogs } from '../../i18n/catalogs'
import { composeCustomIndicator, type IndicatorDefinition, type IndicatorVariable } from '../../services/customIndicators'
import { resolveIndicatorFormula, type AuthoringNode, type GraphDocument, type GraphOutput } from '../../services/indicatorGraph'
import GraphCanvas from '../computation-graph/GraphCanvas'
import { NODE_APPEARANCES } from '../computation-graph/nodeAppearance'
import { replaceGraphNode } from './indicatorGraphConstants'
import type { CanvasConnection, CanvasNodeChange } from '../computation-graph/types'
import IndicatorNodeInspector from './IndicatorNodeInspector'
import { useIndicatorGraphEditor, type GraphEditorStateProps } from './useIndicatorGraphEditor'
import { canvasModel, connectGraph, duplicateGraphNodes, emptyOutput, graphConnectionIssue, graphSignature, layoutDocument, localGraphIssues, makeOperator, newNodeId, nodeLabel, outputNodeId, removeGraphEdges, removeGraphNodes } from './indicatorGraphAdapter'
import { createGraphTextFormatter, graphOperatorLabel, graphTypeLabel, graphVariableLabel } from './indicatorGraphPresentation'

export interface IndicatorGraphEditorHandle { persistLayout: (saved: IndicatorDefinition) => Promise<void> }
interface Props extends GraphEditorStateProps {
  variables: IndicatorVariable[]; indicators: IndicatorDefinition[]; disabled?: boolean; definitionDirty?: boolean
}
const button = 'min-h-10 rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm font-semibold text-slate-700 hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-40 focus:outline-none focus-visible:ring-2 focus-visible:ring-violet-400'
const testIds = { canvas: 'indicator-graph-canvas', mobile: 'indicator-graph-mobile', flow: 'indicator-graph-flow', minimap: 'indicator-graph-minimap' }

const IndicatorGraphEditor = forwardRef<IndicatorGraphEditorHandle, Props>(function IndicatorGraphEditor(props, ref) {
  const { s, b, version: translationVersion } = useI18n()
  const editor = useIndicatorGraphEditor(props)
  const { draft, operators, variables, indicators, disabled = false, definitionDirty = false } = props
  const text = useMemo(() => createGraphTextFormatter(variables, operators), [variables, operators, translationVersion])
  const [panel, setPanel] = useState<'resources' | 'inspector' | null>(null)
  const [resourceTab, setResourceTab] = useState<'variables' | 'operators' | 'indicators'>('variables')
  const [query, setQuery] = useState('')
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [fullscreen, setFullscreen] = useState(false)
  const [importing, setImporting] = useState(false)
  const [layoutKey, setLayoutKey] = useState(0)
  const rootRef = useRef<HTMLDivElement>(null)
  const fullButtonRef = useRef<HTMLButtonElement>(null)
  const latest = useRef(editor.document)
  latest.current = editor.document
  const alive = useRef(true)
  useEffect(() => { alive.current = true; return () => { alive.current = false } }, [])
  useImperativeHandle(ref, () => ({ persistLayout: editor.persistLayout }))
  const readOnly = disabled || editor.layoutSaving || importing || !editor.loaded
  const knownTypes = editor.resolution?.valid ? editor.resolution.node_types || {} : {}
  const currentIssues = [...localGraphIssues(editor.document.graph, operators), ...editor.issues]
  const model = useMemo(() => canvasModel(editor.document, variables, operators, knownTypes, currentIssues), [editor.document, variables, operators, editor.resolution, editor.issues, translationVersion])
  const normalizedQuery = query.trim().toLowerCase()
  const match = (label: string, id: string, description = '') => `${label} ${id} ${description} ${Object.values(builtinCatalogs.business[`variables.${id}.label`] || {}).join(' ')} ${Object.values(builtinCatalogs.business[`operators.${id}.label`] || {}).join(' ')}`.toLowerCase().includes(normalizedQuery)
  const pick = (id: string) => { setSelectedId(id); setPanel('inspector') }
  const changeGraph = (graph: GraphDocument['graph']) => editor.change({ ...editor.document, graph })
  const onConnect = (connection: CanvasConnection) => {
    const issue = graphConnectionIssue(editor.document.graph, connection, operators, knownTypes)
    if (issue) { editor.setMessage(issue); return }
    changeGraph(connectGraph(editor.document.graph, connection))
  }
  const onNodesChange = (changes: CanvasNodeChange[]) => {
    let document = editor.document
    for (const change of changes) {
      if (change.type === 'select') { setSelectedId(change.id); continue }
      if (readOnly) continue
      if (change.type === 'position') document = { ...document, positions: { ...document.positions, [change.id]: change.position } }
      else if (change.id.startsWith('output_')) { editor.setMessage('请在输出设置中删除通道；最终结果至少保留一个。') }
      else document = { ...document, graph: removeGraphNodes(document.graph, [change.id]) }
    }
    editor.change(document)
  }
  const addResource = (resource: string, position?: { x: number; y: number }) => {
    if (readOnly || editor.document.graph.nodes.length >= 128) return
    const [kind, id] = resource.split(':', 2)
    let node: AuthoringNode
    if (kind === 'variable') {
      const variable = variables.find(item => item.name === id)
      if (!variable || variable.availability === 'unavailable' || variable.availability === 'not_applicable') return
      node = { id: newNodeId(), kind: 'variable', variable_id: id }
    } else if (kind === 'operator') {
      const operator = operators.find(item => item.name === id)
      if (!operator) return
      node = makeOperator(operator)
    } else if (kind === 'parameter') {
      const parameter = draft.parameter_schema?.find(item => item.id === id)
      if (!parameter || draft.parameter_contract_version !== '1.0') return
      node = { id: newNodeId(), kind: 'parameter', parameter_id: id, label: parameter.label }
    } else if (kind === 'constant') node = { id: newNodeId(), kind: 'constant', value: null }
    else return
    const next = { ...editor.document, graph: { ...editor.document.graph, nodes: [...editor.document.graph.nodes, node] }, positions: { ...editor.document.positions, [node.id]: position || { x: 60 + (editor.document.graph.nodes.length % 3) * 280, y: 100 + Math.floor(editor.document.graph.nodes.length / 3) * 180 } } }
    editor.change(next)
    pick(node.id)
  }
  const duplicate = (ids: string[]) => {
    if (readOnly) return
    const count = editor.document.graph.nodes.filter(node => ids.includes(node.id)).length
    if (editor.document.graph.nodes.length + count > 128) { editor.setMessage('最多保留 128 个计算节点。'); return }
    editor.change(duplicateGraphNodes(editor.document, ids))
  }
  const patchOutput = (id: string, patch: Partial<GraphOutput>) => {
    if (patch.id && (patch.id !== id)) {
      const nextId = outputNodeId(patch.id)
      const positions = { ...editor.document.positions, [nextId]: editor.document.positions[outputNodeId(id)] || { x: 900, y: 0 } }
      delete positions[outputNodeId(id)]
      editor.change({ ...editor.document, positions, graph: { ...editor.document.graph, outputs: editor.document.graph.outputs.map(output => output.id === id ? { ...output, ...patch } : output) } })
      setSelectedId(nextId)
    } else changeGraph({ ...editor.document.graph, outputs: editor.document.graph.outputs.map(output => output.id === id ? { ...output, ...patch } : output) })
  }
  const remove = (id: string) => {
    const output = editor.document.graph.outputs.find(item => outputNodeId(item.id) === id)
    if (output) {
      if (editor.document.graph.outputs.length <= 1) return
      changeGraph({ ...editor.document.graph, outputs: editor.document.graph.outputs.filter(item => item.id !== output.id) })
    } else changeGraph(removeGraphNodes(editor.document.graph, [id]))
    setSelectedId(null)
    setPanel(null)
  }
  const addOutput = () => {
    if (editor.document.graph.outputs.length >= 8) return
    let index = 1
    while (editor.document.graph.outputs.some(output => output.id === `channel_${index}`)) index += 1
    const output = emptyOutput(`channel_${index}`, `输出通道 ${index}`)
    editor.change(layoutDocument({ ...editor.document, graph: { ...editor.document.graph, outputs: [...editor.document.graph.outputs, output] } }))
    pick(outputNodeId(output.id))
  }
  const insertIndicator = async (indicator: IndicatorDefinition) => {
    const snapshot = editor.document
    const signature = graphSignature(snapshot.graph)
    setImporting(true)
    try {
      const expanded = await composeCustomIndicator({ indicator_id: indicator.id, indicator_revision: indicator.revision, arguments: [], context: draft.context_kind || 'single_product', dsl_version: draft.dsl_version, operator_registry_version: draft.operator_registry_version, variable_registry_version: draft.variable_registry_version, data_contract_version: draft.data_contract_version, context_schema_version: draft.context_schema_version })
      const result = await resolveIndicatorFormula({ ...draft, result_kind: 'scalar', expression: expanded.expression || expanded.python_expression || expanded.latex, series_outputs: [] }, 0)
      if (!alive.current || signature !== graphSignature(latest.current.graph)) return
      if (!result.valid || !result.graph) throw new Error(result.diagnostics[0]?.message || '此指标不能展开。')
      const incoming = result.graph
      if (snapshot.graph.nodes.length + incoming.nodes.length > 128) throw new Error('展开后超过 128 个节点，请减少步骤。')
      const mapping = new Map(incoming.nodes.map(node => [node.id, newNodeId()]))
      const rootId = incoming.outputs[0].node_id
      const added = incoming.nodes.map(node => ({ ...node, id: mapping.get(node.id)!, ...(node.id === rootId ? { label: `${indicator.name} v${indicator.revision}` } : {}), ...(node.kind === 'operator' ? { arguments: Object.fromEntries(Object.entries(node.arguments).map(([name, binding]) => [name, binding.source === 'node' ? { ...binding, node_id: mapping.get(binding.node_id)! } : binding])) } : {}) })) as AuthoringNode[]
      editor.change(layoutDocument({ ...snapshot, graph: { ...snapshot.graph, nodes: [...snapshot.graph.nodes, ...added] } }))
      if (rootId) pick(mapping.get(rootId)!)
      editor.setMessage('已展开锁定版本为独立步骤；请把结果连接到下游或最终输出。')
    } catch (failure) { if (alive.current) editor.setMessage(failure instanceof Error ? failure.message : '指标展开失败。') }
    finally { if (alive.current) setImporting(false) }
  }

  useEffect(() => {
    if (!fullscreen) return
    const previousOverflow = document.body.style.overflow
    const previousFocus = document.activeElement as HTMLElement | null
    document.body.style.overflow = 'hidden'
    const frame = requestAnimationFrame(() => rootRef.current?.focus())
    const escape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); setPanel(null); setFullscreen(false) }
    }
    const containFocus = (event: FocusEvent) => {
      if (event.target instanceof Node && !rootRef.current?.contains(event.target)) rootRef.current?.focus()
    }
    document.addEventListener('keydown', escape, true)
    document.addEventListener('focusin', containFocus)
    return () => {
      cancelAnimationFrame(frame)
      document.removeEventListener('keydown', escape, true)
      document.removeEventListener('focusin', containFocus)
      document.body.style.overflow = previousOverflow
      ;(fullButtonRef.current || previousFocus)?.focus()
    }
  }, [fullscreen])
  useEffect(() => { if (!props.active) setFullscreen(false) }, [props.active])

  const resourceDescription = (resource: string, description?: string) => {
    const [kind, id] = resource.split(':', 2)
    return b(`${kind === 'variable' ? 'variables' : 'operators'}.${id}.description`, text(description))
  }
  const resourceButton = (id: string, label: string, description?: string, unavailable = false) => <button key={id} type="button" draggable={!readOnly && !unavailable} onDragStart={event => { event.dataTransfer.setData('application/x-indicator-resource', id); event.dataTransfer.effectAllowed = 'copy' }} disabled={readOnly || unavailable} title={resourceDescription(id, description)} onClick={() => addResource(id)} className="block min-h-12 w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-left hover:border-violet-400 hover:bg-violet-50 disabled:opacity-40"><span className="block text-sm font-semibold text-slate-800">{label}</span>{description && <span className="mt-1 block line-clamp-2 text-xs leading-5 text-slate-500">{resourceDescription(id, description)}</span>}</button>
  const panelContent = panel === 'resources' ? <>
    <input type="search" aria-label={s('graph.search')} value={query} onChange={event => setQuery(event.target.value)} placeholder={s('graph.searchHint')} className="min-h-11 w-full rounded-lg border border-slate-300 px-3 py-2 text-sm" />
    <div className="mt-3 flex flex-wrap gap-1">{([['variables', '输入数据'], ['operators', '计算算子'], ['indicators', '已有指标']] as const).map(([id, label]) => <button key={id} type="button" aria-pressed={resourceTab === id} onClick={() => setResourceTab(id)} className={`min-h-10 rounded-lg px-2 text-xs font-semibold ${resourceTab === id ? 'bg-violet-100 text-violet-800' : 'text-slate-600 hover:bg-slate-100'}`}>{s(`graph.${id}`, {}, label)}</button>)}</div>
    <p className="my-3 text-xs leading-5 text-slate-500">{s('graph.resourceHint')}</p>
    {draft.parameter_contract_version === '1.0' && (draft.parameter_schema ?? []).map(parameter => <button key={`parameter:${parameter.id}`} type="button" className={`${button} mb-2 block w-full text-left`} disabled={readOnly} onClick={() => addResource(`parameter:${parameter.id}`)}>{parameter.label}<span className="ml-2 text-xs text-cyan-800">{parameter.id} = {parameter.default}</span></button>)}
    <div className="space-y-2">{resourceTab === 'variables' ? variables.filter(variable => match(variable.label, variable.name, variable.description)).map(variable => resourceButton(`variable:${variable.name}`, graphVariableLabel(variable.name, variables), variable.description, variable.availability === 'unavailable' || variable.availability === 'not_applicable')) : resourceTab === 'operators' ? operators.filter(operator => match(operator.label, operator.name, operator.semantic)).map(operator => resourceButton(`operator:${operator.name}`, graphOperatorLabel(operator.name, operators), operator.mathematical_essence || operator.semantic)) : indicators.filter(indicator => (indicator.result_kind ?? 'scalar') === 'scalar' && indicator.dsl_version === draft.dsl_version && match(indicator.name, indicator.id, indicator.description)).map(indicator => <button type="button" key={indicator.id} disabled={readOnly} onClick={() => void insertIndicator(indicator)} className={`${button} block w-full text-left`}>{indicator.name}<span className="ml-2 text-xs text-slate-400">v{indicator.revision}</span></button>)}</div>
  </> : <CalculationSteps compact steps={editor.document.graph.nodes.map(node => ({ id: node.id, label: nodeLabel(node, variables, operators) }))} selectedId={selectedId} onSelect={pick}><IndicatorNodeInspector graph={editor.document.graph} selectedId={selectedId} variables={variables} operators={operators} types={knownTypes} isTimeSeries={draft.result_kind === 'time_series'} onNodeChange={node => changeGraph(replaceGraphNode(editor.document.graph, node))} onOutputChange={patchOutput} onRemove={remove} onDuplicate={id => duplicate([id])} /></CalculationSteps>

  return <div ref={rootRef} tabIndex={-1} role={fullscreen ? 'dialog' : 'region'} aria-modal={fullscreen || undefined} aria-label={s(fullscreen ? 'graph.fullscreenAria' : 'graph.editorAria')} className={fullscreen ? 'fixed inset-0 z-[100] overflow-auto bg-slate-50 p-3 sm:p-5' : 'mt-4 min-w-0'} onKeyDown={event => {
    if (fullscreen && event.key === 'Escape') { event.preventDefault(); setFullscreen(false); return }
    if (fullscreen && event.key === 'Tab') {
      const focusable = Array.from(rootRef.current?.querySelectorAll<HTMLElement>('button:not(:disabled),input:not(:disabled),select:not(:disabled),textarea:not(:disabled),[tabindex="0"]') || []).filter(element => element.getClientRects().length > 0)
      const first = focusable[0], last = focusable[focusable.length - 1]
      if (event.shiftKey && (document.activeElement === first || document.activeElement === rootRef.current)) { event.preventDefault(); last?.focus() }
      else if (!event.shiftKey && (document.activeElement === last || document.activeElement === rootRef.current)) { event.preventDefault(); first?.focus() }
    }
    if (!readOnly && (event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'z' && !(event.target as HTMLElement).closest('input,textarea,select,[contenteditable="true"]')) { event.preventDefault(); event.shiftKey ? editor.redo() : editor.undo() }
  }}>
    <div className={`mb-3 flex flex-wrap items-center justify-between gap-3 rounded-xl border border-slate-200 bg-white p-3 ${fullscreen ? 'sticky top-0 z-30' : ''}`}>
      <div><h3 className="text-base font-semibold text-slate-900">{fullscreen ? draft.name : s('graph.title')}</h3><p className="mt-1 text-xs text-slate-500">{s('graph.hint')}</p></div>
      <div className="flex flex-wrap gap-2"><button type="button" className={button} disabled={!editor.loaded || editor.busy || disabled || importing} onClick={() => void editor.check(false)}>{s('graph.check')}</button><button type="button" disabled={!editor.loaded || editor.busy || disabled || importing} onClick={() => void editor.check(true)} className={`${button} !border-violet-600 !bg-violet-600 !text-white`}>{s(editor.busy ? 'graph.checking' : 'graph.apply')}</button><button ref={fullButtonRef} type="button" className={button} aria-pressed={fullscreen} onClick={() => { editor.change({ ...editor.document, viewport: null }); setLayoutKey(value => value + 1); if (fullscreen) setPanel(null); setFullscreen(value => !value) }}>{s(fullscreen ? 'graph.exitFullscreen' : 'graph.fullscreen')}</button></div>
    </div>
    <div role="status" aria-live="polite" className={`mb-3 rounded-lg border px-3 py-2 text-sm leading-6 ${editor.pending ? 'border-amber-200 bg-amber-50 text-amber-900' : 'border-slate-200 bg-white text-slate-600'}`}>{editor.pending && <strong className="mr-2">{s('graph.pending')}</strong>}{editor.message ? s(editor.message, {}, text(editor.message)) : ''}</div>
    <div aria-label={s('graph.typeLegend')} className="mb-3 flex flex-wrap gap-2 text-xs font-semibold text-slate-700">{Object.entries(NODE_APPEARANCES).map(([kind, appearance]) => <span key={kind} className={`inline-flex items-center gap-1.5 border px-3 py-1.5 ${appearance.className}`}><span aria-hidden="true" style={{ color: appearance.color }}>{appearance.glyph}</span>{b(`nodeKinds.${kind}`, appearance.label)}</span>)}</div>
    {!editor.loaded && !editor.busy && <div className="my-4 rounded-xl border border-dashed border-slate-300 bg-white p-5"><p className="mb-3 text-sm text-slate-600">{s('graph.restoreHint', {}, '不会用旧公式覆盖当前输入。可返回高级公式模式修复，或重新构建。')}</p><button type="button" className={button} onClick={editor.startBlank}>{s('graph.startBlank')}</button></div>}
    <div className={`grid min-w-0 gap-3 ${panel ? 'lg:grid-cols-[minmax(0,1fr)_280px]' : ''}`}>
      <div className="min-w-0">{editor.loaded ? <GraphCanvas key={layoutKey} {...model} typeLabel={(type, label) => label || graphTypeLabel(type)} selectedNodeId={selectedId} onNodeActivate={pick} onNodesChange={onNodesChange} onConnect={onConnect} onDuplicate={duplicate} onEdgesRemove={ids => changeGraph(removeGraphEdges(editor.document.graph, ids))} validateConnection={connection => !graphConnectionIssue(editor.document.graph, connection, operators, knownTypes)} isPortCompatible={() => true} onResourceDrop={addResource} readOnly={readOnly} testIds={testIds} viewport={editor.document.viewport || undefined} onViewportCommit={viewport => editor.change({ ...editor.document, viewport })} flowClassName={`hidden min-w-0 overflow-hidden rounded-xl border border-slate-200 bg-white md:block ${fullscreen ? 'h-[calc(100dvh-380px)] min-h-[360px]' : 'h-[clamp(560px,65vh,820px)]'}`} description={s('graph.instructions')} toolbar={<>
        <button type="button" className={button} aria-pressed={panel === 'resources'} onClick={() => setPanel(value => value === 'resources' ? null : 'resources')}>{s('graph.add')}</button>
        <button type="button" className={button} disabled={readOnly} onClick={() => addResource('constant:')}>{s('graph.addConstant')}</button>
        <button type="button" className={button} disabled={readOnly || !editor.history.past.length} onClick={editor.undo}>{s('graph.undo')}</button><button type="button" className={button} disabled={readOnly || !editor.history.future.length} onClick={editor.redo}>{s('graph.redo')}</button>
        <button type="button" className={button} disabled={readOnly} onClick={() => { editor.change(layoutDocument(editor.document)); setLayoutKey(value => value + 1) }}>{s('graph.layout')}</button>
        {draft.result_kind === 'time_series' && <button type="button" className={button} disabled={readOnly || editor.document.graph.outputs.length >= 8} onClick={addOutput}>{s('graph.addOutput')}</button>}
        <button type="button" className={button} disabled={!props.indicator || definitionDirty || editor.pending || readOnly || !editor.layoutDirty} onClick={() => void editor.saveLayout()} title={s(definitionDirty ? 'graph.saveDefinitionFirst' : 'graph.layoutOnly')}>{s(editor.layoutSaving ? 'common.saving' : 'graph.saveLayout')}</button>
      </>} /> : <div className="flex min-h-[400px] items-center justify-center rounded-xl border border-dashed border-slate-300 bg-white p-5 text-sm text-slate-500">{s(editor.busy ? 'graph.preparing' : 'graph.restoreHint')}</div>}</div>
      {panel && <aside className={`fixed inset-x-2 bottom-2 z-[110] max-h-[70dvh] min-w-0 self-start overflow-y-auto rounded-xl border border-slate-200 bg-white p-4 shadow-xl lg:static lg:z-auto lg:shadow-none ${fullscreen ? 'lg:max-h-[calc(100dvh-215px)]' : 'lg:max-h-[740px]'}`} aria-label={s(panel === 'resources' ? 'graph.resources' : 'graph.inspector')}><div className="mb-4 flex items-center justify-between gap-2"><h3 className="text-sm font-semibold">{s(panel === 'resources' ? 'graph.resourceTitle' : 'graph.inspectorTitle')}</h3><button type="button" className={button} onClick={() => setPanel(null)}>{s('graph.collapse')}</button></div><fieldset disabled={readOnly} className="min-w-0">{panelContent}</fieldset></aside>}
    </div>
    {editor.issues.length > 0 && <section aria-label="画布检查结果" className="mt-3 rounded-xl border border-rose-200 bg-rose-50 p-4"><h4 className="font-semibold text-rose-900">{s('graph.checkProblems', { count: editor.issues.length })}</h4><div className="mt-2 max-h-48 space-y-1 overflow-auto">{editor.issues.map((issue, index) => <button key={`${issue.code}-${index}`} type="button" onClick={() => { if (issue.editor_node_id) pick(issue.editor_node_id) }} className="block min-h-10 w-full rounded-lg px-2 py-2 text-left text-sm text-rose-800 hover:bg-rose-100">{index + 1}. {s(`errors.${issue.code}`, {}, text(issue.message))}</button>)}</div></section>}
    {editor.resolution?.valid && <details className="mt-3 rounded-xl border border-slate-200 bg-white p-3"><summary className="cursor-pointer text-sm font-semibold text-slate-700">{s('graph.formulas')}</summary><p className="my-3 text-xs text-slate-500">{s('graph.dependencies', { names: editor.resolution.dependencies?.map(id => graphVariableLabel(id, variables)).join('、') || s('common.none') })}</p>{Object.entries(editor.resolution.display_latex || {}).map(([id, latex]) => <div key={id} className="mt-2 min-w-0 overflow-auto"><p className="text-xs font-semibold">{id === 'result' ? b('nodeKinds.final') : editor.document.graph.outputs.find(output => output.id === id)?.label || s('graph.unnamedOutput')}</p><div dangerouslySetInnerHTML={{ __html: katex.renderToString(latex, { throwOnError: false, displayMode: true, trust: false }) }} /></div>)}</details>}
  </div>
})
export default IndicatorGraphEditor

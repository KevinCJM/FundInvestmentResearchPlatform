import { useEffect, useRef, useState } from 'react'
import type { IndicatorDefinition, IndicatorDraft, IndicatorOperator } from '../../services/customIndicators'
import {
  definitionSourceKey, getIndicatorEditorState, resolveIndicatorFormula, resolveIndicatorGraph, saveIndicatorEditorState,
  type GraphDiagnostic, type GraphDocument, type GraphResolution,
} from '../../services/indicatorGraph'
import { createTimelineReducer, type GraphTimeline, type GraphTimelineAction } from '../computation-graph/history'
import { emptyDocument, emptyOutput, graphSignature, layoutDocument, localGraphIssues, outputPresentation } from './indicatorGraphAdapter'
import { materializeConstantNodes } from './indicatorGraphConstants'

export interface GraphEditorStateProps {
  draft: IndicatorDraft; indicator: IndicatorDefinition | null; operators: IndicatorOperator[]; active: boolean; ready: boolean
  onApply: (patch: Partial<IndicatorDraft>) => void; onPendingChange: (pending: boolean) => void
}
const identityOf = (indicator: Pick<IndicatorDefinition, 'id' | 'revision'> | null) => indicator ? `${indicator.id}@${indicator.revision}` : 'new'
const errorText = (error: unknown) => error instanceof Error ? error.message : '操作失败，请重试。'
const reducer = createTimelineReducer<GraphDocument>()

export function useIndicatorGraphEditor({ draft, indicator, operators, active, ready, onApply, onPendingChange }: GraphEditorStateProps) {
  const [history, setHistory] = useState<GraphTimeline<GraphDocument>>(() => ({ past: [], present: emptyDocument(draft), future: [] }))
  const historyRef = useRef(history)
  const [resolution, setResolution] = useState<GraphResolution | null>(null)
  const [issues, setIssues] = useState<GraphDiagnostic[]>([])
  const [busy, setBusy] = useState(false)
  const [loaded, setLoaded] = useState(false)
  const [message, setMessage] = useState('')
  const [layoutSaving, setLayoutSaving] = useState(false)
  const [layoutSaved, setLayoutSaved] = useState('')
  const requestSequence = useRef(0)
  const abortRef = useRef<AbortController | null>(null)
  const appliedGraph = useRef('')
  const lastSource = useRef('')
  const editorVersion = useRef({ identity: '', revision: 0 })
  const callbacks = useRef({ onApply, onPendingChange })
  callbacks.current = { onApply, onPendingChange }
  const draftRef = useRef(draft)
  draftRef.current = draft
  const sourceKey = definitionSourceKey(draft)
  const identity = identityOf(indicator)
  const pending = loaded && graphSignature(history.present.graph) !== appliedGraph.current

  const transition = (action: GraphTimelineAction<GraphDocument>) => {
    const previous = historyRef.current
    const next = reducer(previous, action)
    if (next === previous) return
    const semanticChanged = graphSignature(next.present.graph) !== graphSignature(previous.present.graph)
    historyRef.current = next
    setHistory(next)
    if (semanticChanged) {
      requestSequence.current += 1
      abortRef.current?.abort()
      setBusy(false)
      setResolution(null)
      setIssues([])
      callbacks.current.onPendingChange(graphSignature(next.present.graph) !== appliedGraph.current)
      setMessage('graph.messages.changed')
    }
  }
  const change = (input: GraphDocument) => {
    const document = materializeConstantNodes(input)
    if (document.graph.nodes.length > 128) { setMessage('含常量在内最多保留 128 个节点，请先减少步骤。'); return }
    const ids = new Set([...document.graph.nodes.map(node => node.id), ...document.graph.outputs.map(output => `output_${output.id}`)])
    const positions = Object.fromEntries(Object.entries(document.positions).filter(([id]) => ids.has(id)))
    transition({ type: 'edit', definition: { ...document, positions } })
  }

  useEffect(() => {
    if (!active || !ready || lastSource.current === `${identity}:${sourceKey}`) return
    const sequence = ++requestSequence.current
    const controller = new AbortController()
    abortRef.current?.abort()
    abortRef.current = controller
    setBusy(true)
    setMessage('graph.messages.restoring')
    void (async () => {
      try {
        const currentDraft = draftRef.current
        let checked: GraphResolution = await resolveIndicatorFormula(currentDraft, sequence, controller.signal)
        if (sequence !== requestSequence.current) return
        if (!checked.valid || !checked.graph) {
          setIssues(checked.diagnostics)
          setLoaded(false)
          setMessage('graph.messages.restoreFailed')
          return
        }
        let document = layoutDocument({ graph: checked.graph, positions: {} })
        let stateWarning = ''
        if (indicator) {
          try {
            const stored = await getIndicatorEditorState(indicator, controller.signal)
            if (sequence !== requestSequence.current) return
            editorVersion.current = { identity, revision: stored.editor_revision }
            if (stored.state && stored.definition_fingerprint === checked.definition_fingerprint) {
              const restored = await resolveIndicatorGraph(draftRef.current, stored.state.graph, sequence, controller.signal)
              if (sequence !== requestSequence.current) return
              if (restored.valid) { document = stored.state; checked = restored }
              else stateWarning = '旧布局无法恢复，已按公式重新排布。'
            }
          } catch (failure) {
            if (controller.signal.aborted) return
            stateWarning = `公式已还原，布局读取失败：${errorText(failure)}`
          }
        }
        if (sequence !== requestSequence.current) return
        const previousLayout = JSON.stringify(document)
        const migrated = materializeConstantNodes(document)
        const layoutMigrated = migrated !== document
        if (migrated.graph.nodes.length > 128) throw new Error('旧布局展开常量后超过 128 个节点，请使用公式模式简化。')
        if (migrated !== document) {
          document = layoutDocument(migrated)
          stateWarning = '旧布局中的常量已展开为独立节点；计算公式保持不变。'
        }
        const metadata = draftRef.current.series_outputs ?? []
        document = { ...document, graph: { ...document.graph, outputs: document.graph.outputs.map(output => {
          const current = metadata.find(item => item.id === output.id)
          if (!current) return output
          return { ...output, ...outputPresentation(current) }
        }) } }
        const next = reducer(historyRef.current, { type: 'reset', definition: document })
        historyRef.current = next
        setHistory(next)
        appliedGraph.current = graphSignature(document.graph)
        lastSource.current = `${identity}:${sourceKey}`
        setLayoutSaved(layoutMigrated ? previousLayout : JSON.stringify(document))
        setResolution(checked)
        setIssues([])
        setLoaded(true)
        callbacks.current.onPendingChange(false)
        setMessage(stateWarning || 'graph.messages.ready')
      } catch (failure) {
        if (sequence !== requestSequence.current || controller.signal.aborted) return
        setLoaded(false)
        setMessage(errorText(failure))
      } finally { if (sequence === requestSequence.current) setBusy(false) }
    })()
    return () => controller.abort()
  }, [active, ready, sourceKey, identity])

  useEffect(() => () => { requestSequence.current += 1; abortRef.current?.abort() }, [])
  useEffect(() => {
    if (!pending) return
    const warn = (event: BeforeUnloadEvent) => { event.preventDefault(); event.returnValue = '' }
    window.addEventListener('beforeunload', warn)
    return () => window.removeEventListener('beforeunload', warn)
  }, [pending])

  const check = async (apply: boolean) => {
    const document = historyRef.current.present
    const localIssues = localGraphIssues(document.graph, operators)
    if (localIssues.length) { setIssues(localIssues); setMessage('graph.messages.incomplete'); return }
    const sequence = ++requestSequence.current
    const controller = new AbortController()
    abortRef.current?.abort()
    abortRef.current = controller
    setBusy(true)
    setIssues([])
    try {
      const checked = await resolveIndicatorGraph(draftRef.current, document.graph, sequence, controller.signal)
      if (sequence !== requestSequence.current) return
      setResolution(checked)
      setIssues(checked.diagnostics)
      if (!checked.valid) { setMessage('graph.messages.invalid'); return }
      if (apply) {
        const expressions = checked.editable_latex || checked.expressions!
        const patch: Partial<IndicatorDraft> = draftRef.current.result_kind === 'time_series'
            ? { series_outputs: document.graph.outputs.map(({ node_id: _nodeId, ...output }) => ({ ...output, expression: expressions[output.id] })), expression: expressions[document.graph.outputs[0].id] }
            : { expression: expressions.result, template_origin: null }
        const nextDraft = { ...draftRef.current, ...patch }
        appliedGraph.current = graphSignature(document.graph)
        lastSource.current = `${identity}:${definitionSourceKey(nextDraft)}`
        callbacks.current.onPendingChange(false)
        callbacks.current.onApply(patch)
        setMessage('graph.messages.applied')
      } else setMessage('graph.messages.valid')
    } catch (failure) {
      if (sequence !== requestSequence.current || controller.signal.aborted) return
      setMessage(errorText(failure))
    } finally { if (sequence === requestSequence.current) setBusy(false) }
  }

  const persistLayout = async (saved: IndicatorDefinition) => {
    // A hidden canvas may describe an older formula edited through another mode.
    if (!loaded || lastSource.current !== `${identity}:${definitionSourceKey(draftRef.current)}`) return
    const document = historyRef.current.present
    if (graphSignature(document.graph) !== appliedGraph.current) throw new Error('画布尚未应用，不能保存布局。')
    const savedIdentity = identityOf(saved)
    let expected = editorVersion.current.revision
    if (savedIdentity !== editorVersion.current.identity) {
      const current = await getIndicatorEditorState(saved)
      expected = current.editor_revision
    }
    const response = await saveIndicatorEditorState(saved, document, expected)
    editorVersion.current = { identity: savedIdentity, revision: response.editor_revision }
    setLayoutSaved(JSON.stringify(document))
  }
  const saveLayout = async () => {
    if (!indicator || pending) return
    setLayoutSaving(true)
    try { await persistLayout(indicator); setMessage('graph.messages.layoutSaved') }
    catch (failure) { setMessage(`布局未保存：${errorText(failure)}`) }
    finally { setLayoutSaving(false) }
  }
  const startBlank = () => {
    if (loaded && historyRef.current.present.graph.nodes.length && !window.confirm('清空当前画布重新构建？可以用撤销恢复。')) return
    lastSource.current = `${identity}:${sourceKey}`
    setLoaded(true)
    const next = layoutDocument(emptyDocument(draftRef.current))
    change(next)
    callbacks.current.onPendingChange(true)
    setMessage('graph.messages.blank')
  }
  return {
    history, document: history.present, resolution, issues, busy, loaded, message, pending, layoutSaving,
    layoutDirty: JSON.stringify(history.present) !== layoutSaved,
    change, undo: () => transition({ type: 'undo' }), redo: () => transition({ type: 'redo' }),
    check, saveLayout, persistLayout, startBlank, setMessage,
  }
}

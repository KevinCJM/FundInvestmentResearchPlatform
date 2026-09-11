import { act, renderHook, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import type { IndicatorDefinition, IndicatorDraft } from '../../services/customIndicators'
import * as api from '../../services/indicatorGraph'
import { useIndicatorGraphEditor, type GraphEditorStateProps } from './useIndicatorGraphEditor'

vi.mock('../../services/indicatorGraph', async importOriginal => ({
  ...await importOriginal<typeof import('../../services/indicatorGraph')>(),
  resolveIndicatorFormula: vi.fn(), resolveIndicatorGraph: vi.fn(),
  getIndicatorEditorState: vi.fn(), saveIndicatorEditorState: vi.fn(),
}))
const draft: IndicatorDraft = { name: '画布测试', description: '', expression: '1', unit: '', display_format: 'number', precision: 4, direction: 'higher_better', annual_risk_free_rate_percent: 0, dsl_version: '2.3.0', result_kind: 'scalar', context_kind: 'single_product' }
const graph = (value = 1): api.AuthoringGraph => ({ graph_version: 1, nodes: [{ id: 'constant', kind: 'constant', value }], outputs: [{ id: 'result', node_id: 'constant', label: '最终结果', unit: '', display_format: 'number', precision: 4, output_measure: 'auto' }] })
const resolved = (value = 1, revision = 0): api.GraphResolution => ({ valid: true, draft_revision: revision, graph: graph(value), diagnostics: [], compile_status: 'not_requested', expressions: { result: String(value) }, editable_latex: { result: String(value) }, definition_fingerprint: String(value), node_types: { constant: { kind: 'scalar' } } })
const saved: IndicatorDefinition = { ...draft, id: 'test', revision: 1, source: 'custom', read_only: false, created_at: '', updated_at: '' }
const props = (): GraphEditorStateProps => ({ draft: { ...draft }, indicator: null, operators: [], active: true, ready: true, onApply: vi.fn(), onPendingChange: vi.fn() })

beforeEach(() => {
  vi.resetAllMocks()
  vi.mocked(api.resolveIndicatorFormula).mockImplementation(async (_draft, revision) => resolved(1, revision))
  vi.mocked(api.resolveIndicatorGraph).mockImplementation(async (_draft, nextGraph, revision) => ({ ...resolved(nextGraph.nodes[0].kind === 'constant' ? Number(nextGraph.nodes[0].value) : 1, revision), graph: nextGraph }))
  vi.mocked(api.getIndicatorEditorState).mockResolvedValue({ indicator_id: 'test', definition_revision: 1, editor_revision: 0, state: null })
  vi.mocked(api.saveIndicatorEditorState).mockImplementation(async (_indicator, document, revision) => ({ indicator_id: 'test', definition_revision: 1, editor_revision: revision + 1, state: document }))
})

describe('authoring state transactions', () => {
  it('editing flags pending immediately; check is not apply', async () => {
    const input = props()
    const { result } = renderHook(() => useIndicatorGraphEditor(input))
    await waitFor(() => expect(result.current.loaded).toBe(true))
    act(() => result.current.change({ ...result.current.document, graph: graph(2) }))
    expect(input.onPendingChange).toHaveBeenLastCalledWith(true)
    await act(async () => { await result.current.check(false) })
    expect(result.current.pending).toBe(true)
    expect(input.onApply).not.toHaveBeenCalled()
    await act(async () => { await result.current.check(true) })
    expect(input.onApply).toHaveBeenCalledWith({ expression: '2', template_origin: null })
    expect(result.current.pending).toBe(false)
  })
  it('positions and notes leave semantic validation intact', async () => {
    const input = props()
    const { result } = renderHook(() => useIndicatorGraphEditor(input))
    await waitFor(() => expect(result.current.loaded).toBe(true))
    const previous = result.current.resolution
    act(() => result.current.change({ ...result.current.document, positions: { constant: { x: 20, y: 70 } }, graph: { ...graph(), nodes: [{ ...graph().nodes[0], label: '说明' }] } }))
    expect(result.current.pending).toBe(false)
    expect(result.current.resolution).toBe(previous)
    expect(api.resolveIndicatorGraph).not.toHaveBeenCalled()
  })
  it('ignores a late response after newer graph edits', async () => {
    const input = props()
    const { result } = renderHook(() => useIndicatorGraphEditor(input))
    await waitFor(() => expect(result.current.loaded).toBe(true))
    let release!: (response: api.GraphResolution) => void
    vi.mocked(api.resolveIndicatorGraph).mockImplementationOnce(() => new Promise(resolve => { release = resolve }))
    act(() => result.current.change({ ...result.current.document, graph: graph(2) }))
    let checking!: Promise<void>
    act(() => { checking = result.current.check(true) })
    act(() => result.current.change({ ...result.current.document, graph: graph(3) }))
    await act(async () => { release(resolved(2)); await checking })
    expect(input.onApply).not.toHaveBeenCalled()
    expect(result.current.document.graph).toEqual(graph(3))
    expect(result.current.pending).toBe(true)
  })
  it('undo and redo restore pending state, and deleted positions are pruned', async () => {
    const { result } = renderHook(() => useIndicatorGraphEditor(props()))
    await waitFor(() => expect(result.current.loaded).toBe(true))
    act(() => result.current.change({ ...result.current.document, graph: graph(2), positions: { removed: { x: 1, y: 2 }, constant: { x: 0, y: 0 } } }))
    expect(result.current.document.positions).not.toHaveProperty('removed')
    act(() => result.current.undo())
    expect(result.current.pending).toBe(false)
    act(() => result.current.redo())
    expect(result.current.pending).toBe(true)
  })
  it('local incomplete nodes never send invalid JSON to the parser', async () => {
    const { result } = renderHook(() => useIndicatorGraphEditor(props()))
    await waitFor(() => expect(result.current.loaded).toBe(true))
    act(() => result.current.change({ ...result.current.document, graph: { ...graph(), nodes: [{ id: 'constant', kind: 'constant', value: null }] } }))
    await act(async () => { await result.current.check(true) })
    expect(api.resolveIndicatorGraph).not.toHaveBeenCalled()
    expect(result.current.issues[0].code).toBe('INVALID_CONSTANT')
  })
  it('saves layout with its own optimistic version and preserves a conflict', async () => {
    const input = { ...props(), indicator: saved }
    const { result } = renderHook(() => useIndicatorGraphEditor(input))
    await waitFor(() => expect(result.current.loaded).toBe(true))
    act(() => result.current.change({ ...result.current.document, positions: { constant: { x: 2, y: 3 } } }))
    await act(async () => { await result.current.saveLayout() })
    expect(api.saveIndicatorEditorState).toHaveBeenLastCalledWith(saved, result.current.document, 0)
    vi.mocked(api.saveIndicatorEditorState).mockRejectedValueOnce(new Error('EDITOR_REVISION_CONFLICT'))
    await act(async () => { await result.current.saveLayout() })
    expect(result.current.message).toContain('EDITOR_REVISION_CONFLICT')
    expect(result.current.document.positions.constant).toEqual({ x: 2, y: 3 })
  })
  it('does not save a stale hidden graph after editing the formula elsewhere', async () => {
    const input = { ...props(), indicator: saved }
    const { result, rerender } = renderHook(parameters => useIndicatorGraphEditor(parameters), { initialProps: input })
    await waitFor(() => expect(result.current.loaded).toBe(true))
    rerender({ ...input, active: false, draft: { ...draft, expression: '20' } })
    await act(async () => { await result.current.persistLayout({ ...saved, revision: 2, expression: '20' }) })
    expect(api.saveIndicatorEditorState).not.toHaveBeenCalled()
  })
})

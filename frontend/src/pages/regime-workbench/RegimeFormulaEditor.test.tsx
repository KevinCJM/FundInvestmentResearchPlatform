import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import RegimeFormulaEditor from './RegimeFormulaEditor'
import { resolveRegimeAuthoring, type RegimeAuthoringResolution, type RegimeGraphDefinition } from '../../services/regimeGraph'

vi.mock('../../services/regimeGraph', async original => ({ ...await original<typeof import('../../services/regimeGraph')>(), resolveRegimeAuthoring: vi.fn() }))
const definition: RegimeGraphDefinition = { schema_version: '2.0', name: '独立计算', description: '',
  graph: { nodes: [{ id: 'value', type: 'source.constant', parameters: { value: .03 }, inputs: {}, position: { x: 99, y: 88 } }], outputs: {} },
  states: [], evaluation_targets: [], validation: {}, usage_intent: 'research_display' }
const result = (source = 'value = source_constant(value=0.03)'): RegimeAuthoringResolution => ({ valid: true, definition, source, diagnostics: [], compile_status: 'not_requested', display_latex: { state: String.raw`S_t=\frac{x_t}{x_{t-1}}-1` } })

describe('RegimeFormulaEditor', () => {
  afterEach(() => { vi.resetAllMocks() })
  it('invalid draft remains visible and cannot replace the graph; valid apply preserves positions', async () => {
    const api = vi.mocked(resolveRegimeAuthoring)
    api.mockResolvedValueOnce(result()).mockResolvedValueOnce({ ...result('bad()'), valid: false, definition: null, diagnostics: [{ code: 'BAD', message: '未知算子' }] }).mockResolvedValueOnce(result('changed'))
    const apply = vi.fn(), pending = vi.fn()
    render(<RegimeFormulaEditor definition={definition} mode="retrospective" schemas={[]} onApply={apply} onPending={pending} onBusy={vi.fn()} />)
    const input = await screen.findByDisplayValue('value = source_constant(value=0.03)')
    await waitFor(() => expect(screen.getByLabelText('当前输出的数学公式').querySelector('.mfrac')).toBeTruthy())
    fireEvent.change(input, { target: { value: 'bad()' } })
    expect(screen.queryByLabelText('当前输出的数学公式')).not.toBeInTheDocument()
    await waitFor(() => expect(pending).toHaveBeenLastCalledWith(true))
    fireEvent.click(screen.getByRole('button', { name: '检查并应用公式' }))
    await screen.findByText('未知算子')
    expect(apply).not.toHaveBeenCalled(); expect(input).toHaveValue('bad()')
    fireEvent.change(input, { target: { value: 'changed' } })
    fireEvent.click(screen.getByRole('button', { name: '检查并应用公式' }))
    await waitFor(() => expect(apply).toHaveBeenCalledTimes(1))
    expect(apply.mock.calls[0][0].graph.nodes[0].position).toEqual({ x: 99, y: 88 })
    expect(pending).toHaveBeenLastCalledWith(false)
    expect(screen.getByLabelText('当前输出的数学公式').querySelector('.katex')).toBeTruthy()
  })

  it('mode or base changes preserve unsaved source and require an explicit reload', async () => {
    vi.mocked(resolveRegimeAuthoring).mockResolvedValue(result())
    const props = { definition, schemas: [], onApply: vi.fn(), onPending: vi.fn(), onBusy: vi.fn() }
    const { rerender } = render(<RegimeFormulaEditor {...props} mode="retrospective" />)
    fireEvent.change(await screen.findByDisplayValue('value = source_constant(value=0.03)'), { target: { value: 'my pending formula' } })
    rerender(<RegimeFormulaEditor {...props} mode="realtime" />)
    expect(screen.getByLabelText('情景计算公式')).toHaveValue('my pending formula')
    expect(screen.getByRole('button', { name: '检查并应用公式' })).toBeDisabled()
    fireEvent.click(screen.getByRole('button', { name: '还原为当前方案' }))
    await screen.findByDisplayValue('value = source_constant(value=0.03)')
    expect(vi.mocked(resolveRegimeAuthoring).mock.calls.slice(-1)[0]?.[1]).toBe('realtime')
  })

  it('late apply response cannot overwrite a newer base definition', async () => {
    let finish!: (value: RegimeAuthoringResolution) => void
    vi.mocked(resolveRegimeAuthoring).mockResolvedValueOnce(result()).mockImplementationOnce(() => new Promise(resolve => { finish = resolve }))
    const props = { schemas: [], onApply: vi.fn(), onPending: vi.fn(), onBusy: vi.fn() }
    const { rerender } = render(<RegimeFormulaEditor {...props} definition={definition} mode="retrospective" />)
    fireEvent.change(await screen.findByDisplayValue('value = source_constant(value=0.03)'), { target: { value: 'changed' } })
    fireEvent.click(screen.getByRole('button', { name: '检查并应用公式' }))
    rerender(<RegimeFormulaEditor {...props} definition={{ ...definition, name: '更新的定义' }} mode="retrospective" />)
    await act(async () => finish(result('obsolete')))
    expect(props.onApply).not.toHaveBeenCalled()
    expect(screen.getByLabelText('情景计算公式')).toHaveValue('changed')
  })
})

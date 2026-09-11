import { act, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import RegimeMathPreview, { RegimeMathDisplay } from './RegimeMathPreview'
import { resolveRegimeAuthoring, type RegimeAuthoringResolution, type RegimeGraphDefinition } from '../../services/regimeGraph'

vi.mock('../../services/regimeGraph', async original => ({ ...await original<typeof import('../../services/regimeGraph')>(), resolveRegimeAuthoring: vi.fn() }))
const definition: RegimeGraphDefinition = { schema_version: '2.0', name: '区间', description: '',
  graph: { nodes: [], outputs: {} }, states: [], evaluation_targets: [], validation: {}, usage_intent: 'research_display' }
const result = (value = '0.03'): RegimeAuthoringResolution => ({ valid: true, definition, source: '', diagnostics: [], compile_status: 'not_requested',
  display_latex: { state: String.raw`S_t=\begin{cases}\text{牛市},&r_t>${value}\\\text{震荡},&|r_t|\le ${value}\end{cases}`, trend: String.raw`R_t=\frac{x_t}{x_{t-1}}-1` },
  formula_steps: { state: [{ node_id: 'r', port: 'value', label: '区间收益', latex: String.raw`R=\frac{x_b}{x_a}-1`, description: '', parameters: [{ label: '上涨门槛', value }] }] },
})

describe('Regime mathematical preview', () => {
  afterEach(() => { vi.resetAllMocks() })
  it('renders real KaTeX cases, fractions and step equations for the selected output', () => {
    const { rerender } = render(<RegimeMathDisplay resolution={result()} />)
    expect(screen.getByLabelText('当前输出的数学公式').querySelector('.katex-mathml math')).toBeTruthy()
    expect(screen.getByLabelText('第 1 步的数学公式').querySelector('.mfrac')).toBeTruthy()
    expect(screen.queryByText(/暂时无法排版/)).not.toBeInTheDocument()
    rerender(<RegimeMathDisplay resolution={result()} outputId="trend" />)
    expect(screen.getByLabelText('当前输出的数学公式').querySelector('.mfrac')).toBeTruthy()
    expect(screen.queryByLabelText('第 1 步的数学公式')).not.toBeInTheDocument()
  })
  it('never passes malformed math or trusted HTML through as a rendered formula', () => {
    render(<RegimeMathDisplay resolution={{ ...result(), display_latex: { state: String.raw`\frac{` } }} />)
    expect(screen.getByText(/暂时无法排版/)).toBeInTheDocument()
    expect(screen.queryByLabelText('当前输出的数学公式')).not.toBeInTheDocument()
  })
  it('hides the previous formula immediately on edits and ignores stale responses', async () => {
    let complete!: (response: RegimeAuthoringResolution) => void
    vi.mocked(resolveRegimeAuthoring).mockResolvedValueOnce(result()).mockImplementationOnce(() => new Promise(resolve => { complete = resolve })).mockResolvedValueOnce(result('0.12'))
    const { rerender } = render(<RegimeMathPreview definition={definition} mode="retrospective" outputId="state" />)
    await screen.findByLabelText('当前输出的数学公式')
    rerender(<RegimeMathPreview definition={{ ...definition, name: '第二版' }} mode="retrospective" outputId="state" />)
    expect(screen.queryByLabelText('当前输出的数学公式')).not.toBeInTheDocument()
    await waitFor(() => expect(resolveRegimeAuthoring).toHaveBeenCalledTimes(2))
    rerender(<RegimeMathPreview definition={{ ...definition, name: '第三版' }} mode="retrospective" outputId="state" />)
    await screen.findByLabelText('当前输出的数学公式')
    await act(async () => complete(result('0.99')))
    const math = screen.getByLabelText('当前输出的数学公式')
    expect(math.querySelector('annotation')?.textContent).toContain('0.12')
    expect(math.querySelector('annotation')?.textContent).not.toContain('0.99')
  })
  it('a pending formula draft does not display the last applied mathematics', () => {
    render(<RegimeMathDisplay resolution={result()} message="公式尚未应用。" />)
    const preview = screen.getByRole('region', { name: '情景数学公式预览' })
    expect(within(preview).getByText('公式尚未应用。')).toBeInTheDocument()
    expect(within(preview).queryByLabelText('当前输出的数学公式')).not.toBeInTheDocument()
  })
})

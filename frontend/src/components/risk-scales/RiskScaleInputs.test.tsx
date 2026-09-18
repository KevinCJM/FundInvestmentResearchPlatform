import { render, screen, fireEvent, act } from '@testing-library/react'
import { describe, it, expect, vi } from 'vitest'
import { useState } from 'react'
import { PercentField, useRiskTask } from './shared'
import { definitionIssue } from './editor'
import { riskDefinition } from '../../test/riskScaleFixtures'
import { SourcePicker } from './SourcePicker'
import { riskScales } from '../../services/riskScales'

describe('risk scale input and generation contracts', () => {
  it('preserves blank, minus and decimal edit states without blur-to-zero', () => {
    function Form() { const [value, setValue] = useState(.05); return <PercentField label="Cap" value={value} onChange={setValue} /> }
    render(<Form />); const input = screen.getByRole('spinbutton')
    for (const value of ['', '-', '.', '-.']) { fireEvent.change(input, { target: { value } }); fireEvent.blur(input); expect(input).toHaveValue(value); expect(input).toHaveAttribute('aria-invalid', 'true') }
    fireEvent.change(input, { target: { value: '5.25' } }); expect(input).toHaveValue('5.25'); expect(input).toHaveAttribute('aria-invalid', 'false')
  })
  it.each([[NaN, .02, .03, .04, .05], [-.01, .02, .03, .04, .05], [.01, .01, .03, .04, .05], [.01, .03, .02, .04, .05]])('rejects invalid manual bands %s', (...caps) => {
    expect(definitionIssue({ ...riskDefinition, segmentation: { algorithm_id: 'manual_volatility_bands_v1', manual_caps: caps, rationale: 'Evidence confirmed' } })).toBe('capsInvalid')
  })
  it('accepts an explicit zero first cap and requires rationale', () => {
    const definition = { ...riskDefinition, segmentation: { algorithm_id: 'manual_volatility_bands_v1' as const, manual_caps: [0, .02, .04, .06, .08], rationale: '' } }
    expect(definitionIssue(definition)).toBe('rationaleRequired'); expect(definitionIssue({ ...definition, segmentation: { ...definition.segmentation, rationale: 'Policy evidence' } })).toBe('')
  })
  it('aborts and rejects a late response even when the transport ignores abort', async () => {
    let resolve!: (value: string) => void; let signal: AbortSignal | undefined
    function Harness() { const task = useRiskTask(), [value, setValue] = useState('current'); return <><button onClick={() => task.run(s => { signal = s; return new Promise<string>(done => { resolve = done }) }, setValue)}>run</button><button onClick={task.invalidate}>edit</button><output>{value}</output></> }
    render(<Harness />); fireEvent.click(screen.getByText('run')); fireEvent.click(screen.getByText('edit')); expect(signal?.aborted).toBe(true)
    await act(async () => resolve('obsolete')); expect(screen.getByText('current')).toBeInTheDocument(); expect(screen.queryByText('obsolete')).not.toBeInTheDocument()
  })
  it('shows an actionable empty source state', async () => {
    vi.spyOn(riskScales, 'sources').mockResolvedValue({ items: [], total: 0, offset: 0, limit: 20, problems: [] })
    render(<SourcePicker onSelect={vi.fn()} />); expect(await screen.findByText('没有匹配的可用来源')).toBeInTheDocument(); vi.restoreAllMocks()
  })
})

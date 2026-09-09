import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import type { InferenceResponse } from '../../services/customIndicators'
import { drawdownDraft, drawdownOperator } from '../../test/drawdownFixtures'
import { scalarDraft } from '../../test/scalarOutputFixtures'
import { composedScalarOutputs, namedPortSource, sharedOutputBinding } from './MultiOutputBuilder'
import ScalarOutputEditor from './ScalarOutputEditor'

function response(input = 'adjusted_nav'): InferenceResponse {
  return {
    latex: '', shape: 'record', inferred_type: 'record', semantic_warnings: [],
    expression: `drawdown_analysis(${input})`,
    output_ports: drawdownOperator.output_ports!.map(port => ({ ...port, expression: `drawdown_analysis(${input}).${port.id}`, editable_latex: '' })),
  }
}

describe('Unified output authoring', () => {
  it('creates named outputs from the authoritative compose response', () => {
    const result = composedScalarOutputs(scalarDraft, response(), undefined, false)
    expect(result.patch.scalar_outputs).toHaveLength(4)
    expect(result.patch.result_kind).toBe('scalar_bundle')
    expect(result.patch.scalar_outputs?.[0].display_format).toBe('percent')
    expect(result.patch.scalar_outputs?.every(item => item.expression.startsWith('drawdown_analysis(adjusted_nav).'))).toBe(true)
  })
  it('updates inputs once without readding removed outputs or changing their identities', () => {
    const current = { ...drawdownDraft, scalar_outputs: [drawdownDraft.scalar_outputs![3], { ...drawdownDraft.scalar_outputs![0], label: '我的回撤', precision: 5 }] }
    const result = composedScalarOutputs(current, response('market_close * 2'), current.scalar_outputs[1].id, true)
    expect(result.patch.scalar_outputs?.map(item => [item.id, item.label, item.precision])).toEqual(current.scalar_outputs.map(item => [item.id, item.label, item.precision]))
    expect(result.patch.scalar_outputs).toHaveLength(2)
    expect(result.patch.scalar_outputs?.every(item => item.expression.includes('(market_close * 2).'))).toBe(true)
    expect(result.selectedId).toBe(current.scalar_outputs[1].id)
  })
  it('expands one selected result without deleting independent siblings', () => {
    const current = { ...drawdownDraft, scalar_outputs: [{ ...drawdownDraft.scalar_outputs![0], id: 'keep', expression: 'mean(returns)' }, { ...drawdownDraft.scalar_outputs![1], id: 'replace', expression: '' }] }
    const result = composedScalarOutputs(current, response(), 'replace', false)
    expect(result.patch.scalar_outputs).toHaveLength(5)
    expect(result.patch.scalar_outputs?.[0]).toEqual(current.scalar_outputs[0])
    expect(result.patch.scalar_outputs?.some(item => item.id === 'replace')).toBe(false)
  })
  it('fails closed for missing output contracts and excessive expansion', () => {
    expect(() => composedScalarOutputs(scalarDraft, { ...response(), output_ports: [] }, undefined, false)).toThrow()
    const current = { ...drawdownDraft, scalar_outputs: Array.from({ length: 8 }, (_, i) => ({ ...drawdownDraft.scalar_outputs![0], id: `output_${i}` })) }
    expect(() => composedScalarOutputs(current, response(), 'output_0', false)).toThrow('8')
  })
  it('recognizes full nested inputs without rewriting identifiers or evaluating expressions', () => {
    expect(namedPortSource('drawdown_analysis(maximum(custom_r_future, 1)).max_drawdown')).toEqual({ operator: 'drawdown_analysis', expression: 'drawdown_analysis(maximum(custom_r_future, 1))', port: 'max_drawdown' })
    expect(namedPortSource(String.raw`\operatorname{drawdown_analysis}\left(\mathbf{p}_{\mathrm{adj}}\right).max_drawdown`)?.operator).toBe('drawdown_analysis')
    expect(namedPortSource('drawdown_analysis(x) + drawdown_analysis(y).max_drawdown')).toBeNull()
    expect(namedPortSource('drawdown_analysis(x).max_drawdown * 2')).toBeNull()
    expect(sharedOutputBinding(drawdownDraft, [drawdownOperator])?.ports).toHaveLength(4)
  })
  it('single-value display edits do not change the return contract', () => {
    const patch = vi.fn()
    render(<ScalarOutputEditor draft={scalarDraft} activeId="" validation={null} onPatch={patch} onSelect={vi.fn()} />)
    fireEvent.change(screen.getByLabelText('小数位'), { target: { value: '5' } })
    expect(patch).toHaveBeenCalledWith(expect.objectContaining({ precision: 5 }))
    expect(patch.mock.calls[0][0]).not.toHaveProperty('result_kind')
    expect(screen.queryByRole('button', { name: '使用此计算' })).not.toBeInTheDocument()
    expect(screen.getByRole('heading', { name: '输出结果' })).toBeInTheDocument()
  })
  it('adds an unused port from the output section with a fresh stable identifier', () => {
    const patch = vi.fn()
    const recovery = response().output_ports![2]
    const current = { ...drawdownDraft, scalar_outputs: drawdownDraft.scalar_outputs!.filter(item => item.id !== 'recovery_periods') }
    render(<ScalarOutputEditor draft={current} activeId="max_drawdown" validation={null} sharedComputation availableOutputs={[recovery]} onPatch={patch} onSelect={vi.fn()} />)
    fireEvent.click(screen.getByRole('button', { name: '＋添加结果' }))
    fireEvent.click(screen.getByRole('button', { name: '最大回撤恢复期数' }))
    const outputs = patch.mock.calls[0][0].scalar_outputs
    expect(outputs).toHaveLength(4)
    expect(outputs[3].expression).toBe(recovery.expression)
    expect(outputs[3].id).not.toBe('recovery_periods')
  })
})

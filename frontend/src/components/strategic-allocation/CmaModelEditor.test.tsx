import { useState } from 'react'
import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import CmaModelEditor, { type CmaModelEditorProps } from './CmaModelEditor'
import { cmaModelInputError, type BlackLittermanRequest, type CmaModelRequest, type ScenarioMixtureRequest } from '../../services/cmaModelTypes'

const context = { asset_ids: ['权益', '债券'], as_of: '2026-09-01', currency: 'CNY' }
const bl = (): BlackLittermanRequest => ({
  ...context, method: 'black_litterman', source: '用户明确风险依据',
  covariance: [[0.04, 0.006], [0.006, 0.01]], risk_covariance_basis: 'input_covariance',
  market_weights: { 权益: 0.6, 债券: 0.4 }, market_weight_source: '明确市场组合来源',
  delta: 2.5, tau: 0.05, risk_free_rate: 0.02,
  views: [{ kind: 'absolute', asset_id: '权益', annual_return: 0.08, view_std: 0.025,
    observed_on: '2026-08-30', available_on: '2026-09-01', source: '研究员独立观点依据' }],
})
const mixture = (): ScenarioMixtureRequest => ({
  ...context, method: 'scenario_mixture', source: '明确情景假设依据', risk_mode: 'shared',
  shared_covariance: [[0.04, 0.006], [0.006, 0.01]], scenarios: [
    { id: '增长', probability: 0.4, annual_returns: { 权益: 0.15, 债券: 0.02 }, source: '增长情景依据' },
    { id: '收缩', probability: 0.6, annual_returns: { 权益: -0.1, 债券: 0.06 }, source: '收缩情景依据' },
  ],
})

function Harness({ initial = null, onEdit = vi.fn(), ...props }: Partial<CmaModelEditorProps> & { initial?: CmaModelRequest | null; onEdit?: (v: CmaModelRequest | null) => void }) {
  const [value, setValue] = useState(initial)
  return <CmaModelEditor context={context} value={value} onChange={v => { onEdit(v); setValue(v) }} {...props} />
}
const edit = (label: string, value: string) => fireEvent.change(screen.getByLabelText(label), { target: { value } })

describe('CmaModelEditor', () => {
  it('keeps manual as default and creates empty opt-in drafts without manufacturing a prior', () => {
    const onEdit = vi.fn(), fetch = vi.spyOn(globalThis, 'fetch')
    const storage = vi.spyOn(Storage.prototype, 'setItem')
    render(<Harness onEdit={onEdit} />)
    expect(screen.getByLabelText('预期生成方法')).toHaveValue('manual')
    expect(onEdit).not.toHaveBeenCalled()
    edit('预期生成方法', 'black_litterman')
    const draft = onEdit.mock.lastCall![0]
    expect(draft.market_weights).toEqual({ 权益: NaN, 债券: NaN })
    expect(draft.covariance).toEqual([[NaN, NaN], [NaN, NaN]])
    expect(screen.getByLabelText('权益市场权重（%）')).toHaveValue('')
    expect(screen.getByLabelText('市场风险厌恶系数 δ')).toHaveValue('')
    expect(draft.views).toEqual([])
    expect(fetch).not.toHaveBeenCalled()
    expect(storage).not.toHaveBeenCalled()
    edit('预期生成方法', 'manual')
    expect(onEdit.mock.lastCall![0]).toBeNull()
  })

  it('converts percent/percentage points once and preserves negative and empty drafts', () => {
    const onEdit = vi.fn(), preview = vi.fn()
    render(<Harness initial={bl()} onEdit={onEdit} onPreview={preview} />)
    expect(screen.getByLabelText('权益市场权重（%）')).toHaveValue('60')
    expect(screen.getByLabelText('年化无风险收益（%）')).toHaveValue('2')
    expect(screen.getByLabelText('观点1标准差（百分点）')).toHaveValue('2.5')
    edit('观点1年化总收益（%）', '-5')
    expect(onEdit.mock.lastCall![0].views[0].annual_return).toBe(-0.05)
    edit('观点1标准差（百分点）', '3')
    expect(onEdit.mock.lastCall![0].views[0].view_std).toBe(0.03)
    fireEvent.click(screen.getByRole('button', { name: '预览模型假设' }))
    expect(preview.mock.lastCall![0].views[0].annual_return).toBe(-0.05)
    edit('观点1年化总收益（%）', '')
    expect(Number.isNaN(onEdit.mock.lastCall![0].views[0].annual_return)).toBe(true)
    expect(screen.getByRole('button', { name: '预览模型假设' })).toBeDisabled()
  })

  it('edits relative views, validates distinct assets and availability, and supports no views', () => {
    const onEdit = vi.fn()
    render(<Harness initial={bl()} onEdit={onEdit} onPreview={vi.fn()} />)
    edit('观点1类型', 'relative')
    expect(screen.getByRole('button', { name: '预览模型假设' })).toBeDisabled()
    edit('观点1比较资产', '债券')
    edit('观点1年化收益差（百分点）', '4')
    expect(onEdit.mock.lastCall![0].views[0]).toMatchObject({ kind: 'relative', annual_return: 0.04, relative_to: '债券' })
    expect(screen.getByRole('button', { name: '预览模型假设' })).toBeEnabled()
    edit('观点1可得日', '2026-09-02')
    expect(screen.getByRole('button', { name: '预览模型假设' })).toBeDisabled()
    fireEvent.click(screen.getByRole('button', { name: '移除观点 1' }))
    expect(screen.getByText('尚无观点，可直接预览先验，或添加有依据的观点。')).toBeVisible()
    expect(screen.getByRole('button', { name: '预览模型假设' })).toBeEnabled()
    fireEvent.click(screen.getByRole('button', { name: '添加观点' }))
    expect(onEdit.mock.lastCall![0].views[0]).toMatchObject({ asset_id: '', annual_return: NaN, view_std: NaN, source: '' })
  })

  it('uses raw covariance units and edits only owned copies symmetrically', () => {
    const original = bl(), onEdit = vi.fn()
    render(<Harness initial={original} onEdit={onEdit} />)
    fireEvent.click(screen.getByText('资产风险协方差', { selector: 'summary' }))
    expect(screen.getByLabelText('资产风险协方差：权益 / 权益')).toHaveValue('0.04')
    edit('资产风险协方差：权益 / 债券', '0.007')
    expect(onEdit.mock.lastCall![0].covariance).toEqual([[0.04, 0.007], [0.007, 0.01]])
    expect(original.covariance[0][1]).toBe(0.006)
  })

  it('shows explicit scenario probability sums, supports full-axis editing and blocks partial inputs', () => {
    const onEdit = vi.fn(), preview = vi.fn()
    render(<Harness initial={mixture()} onEdit={onEdit} onPreview={preview} />)
    expect(screen.getByText('情景概率合计：100.00%')).toBeVisible()
    edit('情景1概率（%）', '30')
    expect(screen.getByText('情景概率合计：90.00%')).toBeVisible()
    expect(screen.getByRole('button', { name: '预览模型假设' })).toBeDisabled()
    edit('情景2概率（%）', '70')
    edit('情景1权益年化收益（%）', '-20')
    fireEvent.click(screen.getByRole('button', { name: '预览模型假设' }))
    expect(preview.mock.lastCall![0].scenarios[0]).toMatchObject({ probability: 0.3, annual_returns: { 权益: -0.2, 债券: 0.02 } })
    fireEvent.click(screen.getByRole('button', { name: '添加情景' }))
    expect(onEdit.mock.lastCall![0].scenarios[2].annual_returns).toEqual({ 权益: NaN, 债券: NaN })
    expect(screen.getByRole('button', { name: '预览模型假设' })).toBeDisabled()
  })

  it('requires explicit risk after mode changes and allows distinct scenario matrices', () => {
    const onEdit = vi.fn()
    render(<Harness initial={mixture()} onEdit={onEdit} onPreview={vi.fn()} />)
    edit('情景风险口径', 'scenario_specific')
    expect(onEdit.mock.lastCall![0].shared_covariance).toBeNull()
    expect(onEdit.mock.lastCall![0].scenarios[0].covariance).toEqual([[NaN, NaN], [NaN, NaN]])
    expect(screen.getByRole('button', { name: '预览模型假设' })).toBeDisabled()
    for (const index of [1, 2]) {
      fireEvent.click(screen.getByText(`情景${index}风险协方差`, { selector: 'summary' }))
      edit(`情景${index}风险协方差：权益 / 权益`, index === 1 ? '0.04' : '0.09')
      edit(`情景${index}风险协方差：权益 / 债券`, '0.006')
      edit(`情景${index}风险协方差：债券 / 债券`, '0.01')
    }
    expect(screen.getByRole('button', { name: '预览模型假设' })).toBeEnabled()
    expect(onEdit.mock.lastCall![0].scenarios[1].covariance[0][0]).toBe(0.09)
  })

  it('keeps saved versions read-only and delegates copy without changing inputs', () => {
    const onChange = vi.fn(), onCopy = vi.fn()
    render(<CmaModelEditor context={context} value={bl()} onChange={onChange} readOnly onCopy={onCopy} onPreview={vi.fn()} />)
    expect(screen.getByLabelText('预期生成方法')).toBeDisabled()
    expect(screen.getByLabelText('权益市场权重（%）')).toBeDisabled()
    expect(screen.getByText(/这是已保存版本/)).toBeVisible()
    fireEvent.click(screen.getByRole('button', { name: '复制为新研究' }))
    expect(onCopy).toHaveBeenCalledOnce()
    expect(onChange).not.toHaveBeenCalled()
  })

  it('blocks stale context and busy requests, and exposes errors with retry', () => {
    const preview = vi.fn()
    const { rerender } = render(<CmaModelEditor context={{ ...context, as_of: '2026-09-02' }} value={bl()} onChange={vi.fn()} onPreview={preview} />)
    expect(screen.getByRole('button', { name: '预览模型假设' })).toBeDisabled()
    expect(screen.getByText(/资产范围、日期或币种已变化/)).toBeVisible()
    rerender(<CmaModelEditor context={context} value={bl()} onChange={vi.fn()} onPreview={preview} busy />)
    expect(screen.getByLabelText('CMA生成方法')).toHaveAttribute('aria-busy', 'true')
    expect(screen.getByRole('button', { name: '预览模型假设' })).toBeDisabled()
    rerender(<CmaModelEditor context={context} value={bl()} onChange={vi.fn()} onPreview={preview} error="矩阵无效，请核对" />)
    expect(screen.getByRole('alert')).toHaveTextContent('矩阵无效，请核对')
    fireEvent.click(screen.getByRole('button', { name: '重试模型预览' }))
    expect(preview).toHaveBeenCalledOnce()
  })

  it('explains missing scope and upstream disabled reasons', () => {
    const { rerender } = render(<Harness context={{ ...context, asset_ids: [] }} />)
    expect(screen.getByLabelText('预期生成方法')).toBeDisabled()
    expect(screen.getByText(/尚未载入资产范围/)).toBeVisible()
    rerender(<Harness initial={bl()} disabledReason="知识截止日尚未确认" />)
    expect(screen.getByText('知识截止日尚未确认')).toBeVisible()
  })
})

describe('CMA input validation', () => {
  it('rejects hidden axes, invalid dates, nonfinite inputs, std underflow, and ambiguous risk', () => {
    const badInputs: CmaModelRequest[] = [
      { ...bl(), market_weights: { 权益: 1 } }, { ...bl(), as_of: '2026-02-31' },
      { ...bl(), tau: NaN }, { ...bl(), source: '' }, { ...bl(), covariance: [[0.04, NaN], [NaN, 0.01]] },
      { ...bl(), views: [{ ...bl().views[0], view_std: 1e-200 }] },
      { ...mixture(), scenarios: [{ ...mixture().scenarios[0], probability: 1, annual_returns: { 权益: 0.1 } }] },
      { ...mixture(), shared_covariance: null },
      { ...mixture(), scenarios: mixture().scenarios.map(s => ({ ...s, covariance: [[0.1, 0], [0, 0.1]] })) },
    ]
    badInputs.forEach(input => expect(cmaModelInputError(input)).not.toBeNull())
    expect(cmaModelInputError(bl())).toBeNull()
    expect(cmaModelInputError(mixture())).toBeNull()
  })
})

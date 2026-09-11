import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import type { IndicatorDraft } from '../../services/customIndicators'
import ScalarOutputEditor from './ScalarOutputEditor'

const draft: IndicatorDraft = { name: '最大回撤率', expression: 'negate(min_value(drawdown_series(adjusted_nav)))', description: '', unit: '%', display_format: 'percent', precision: 2, direction: 'lower_better', annual_risk_free_rate_percent: 0, result_kind: 'scalar' }

describe('Single independent metric result settings', () => {
  it('offers no bundle, result tabs or add-result action', () => {
    render(<ScalarOutputEditor draft={draft} validation={null} onPatch={vi.fn()} />)
    expect(screen.getByRole('heading', { name: '结果设置' })).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /添加结果/ })).not.toBeInTheDocument()
    expect(screen.queryByRole('tab')).not.toBeInTheDocument()
  })
  it('editing display precision never changes metric identity or result kind', () => {
    const patch = vi.fn()
    render(<ScalarOutputEditor draft={draft} validation={null} onPatch={patch} />)
    fireEvent.change(screen.getByLabelText('小数位'), { target: { value: '3' } })
    expect(patch).toHaveBeenLastCalledWith({ precision: 3 })
  })
  it('date results are inferred and cannot accidentally receive scoring direction', () => {
    render(<ScalarOutputEditor draft={{ ...draft, display_format: 'date', output_measure: 'date', direction: 'neutral' }} validation={null} onPatch={vi.fn()} />)
    expect(screen.getByLabelText('显示方式')).toHaveValue('date')
    expect(screen.getByLabelText('显示方式')).toBeDisabled()
    expect(screen.getByLabelText('评分方向')).toBeDisabled()
    expect(screen.queryByLabelText('小数位')).not.toBeInTheDocument()
  })
  it('pending computation disables all result controls', () => {
    render(<ScalarOutputEditor draft={draft} validation={null} disabled onPatch={vi.fn()} />)
    expect(screen.getByLabelText('小数位')).toBeDisabled()
  })
})

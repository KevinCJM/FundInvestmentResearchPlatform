import { render, screen, within } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import LtcmaRegimeResults from './LtcmaRegimeResults'

const audit = {
  state_ids: ['up', 'down', 'stress'], counts: [120, 80, 0],
  detected_probabilities: [.6, .4, 0], applied_probabilities: [.5, .5, 0],
  estimated_states: ['up', 'down'], unestimated_states: ['stress'],
  probability_reason: '保留更多下行情景',
  regime: { state_labels: { up: '上行', down: '下行', stress: '压力' }, unknown_observations: 3 },
}

describe('frozen regime evidence', () => {
  it('separates detected and applied probabilities without recomputing them', () => {
    render(<LtcmaRegimeResults audit={audit} />)
    const table = screen.getByRole('table', { name: '历史状态与应用概率' })
    expect(within(table).getByRole('columnheader', { name: '历史占用率' })).toBeVisible()
    expect(within(table).getByRole('columnheader', { name: '本次应用概率' })).toBeVisible()
    const row = within(table).getByRole('rowheader', { name: '上行' }).closest('tr')!
    expect(within(row).getByText('60.00%')).toBeVisible()
    expect(within(row).getByText('50.00%')).toBeVisible()
    expect(within(row).getByText('120')).toBeVisible()
    expect(screen.getByText(/未分类或缺失状态的收益观察：3 条/)).toBeVisible()
    expect(screen.getByText(/保留更多下行情景/)).toBeVisible()
  })
  it('preserves an unestimated zero-weight state without fabricating estimates', () => {
    render(<LtcmaRegimeResults audit={audit} />)
    const row = screen.getByRole('rowheader', { name: '压力' }).closest('tr')!
    expect(within(row).getByText('未估计')).toBeVisible()
    expect(within(row).getAllByText('0.00%')).toHaveLength(2)
  })
  it('does not fill missing probability or estimation evidence with a success value', () => {
    render(<LtcmaRegimeResults audit={{ state_ids: ['missing'], counts: [], detected_probabilities: [], applied_probabilities: [] }} />)
    const row = screen.getByRole('rowheader', { name: 'missing' }).closest('tr')!
    expect(within(row).getAllByText('—')).toHaveLength(3)
    expect(within(row).queryByText('已估计')).not.toBeInTheDocument()
    expect(within(row).queryByText('0.00%')).not.toBeInTheDocument()
  })
  it('shows an explicit incomplete state instead of an empty result table', () => {
    render(<LtcmaRegimeResults audit={{}} />)
    expect(screen.getByRole('status')).toHaveTextContent('缺少完整的冻结状态证据。')
    expect(screen.queryByRole('table')).not.toBeInTheDocument()
  })
})

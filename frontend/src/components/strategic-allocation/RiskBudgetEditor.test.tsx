import { useState } from 'react'
import { render, screen, fireEvent } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import RiskBudgetEditor, { riskBudgetError } from './RiskBudgetEditor'

function Harness() {
  const [value, setValue] = useState<Record<string, number> | null>(null)
  return <RiskBudgetEditor assets={['股', '债']} value={value} onChange={setValue} />
}
describe('risk budget explicit inputs', () => {
  it('is optional and starts unfilled, with percent display', () => {
    render(<Harness />)
    expect(screen.queryByLabelText('股风险预算（%）')).toBeNull()
    fireEvent.click(screen.getByLabelText('增加风险预算候选'))
    expect(screen.getByLabelText('股风险预算（%）')).toHaveValue('')
    expect(screen.getByText(/空白不代表零/)).toBeInTheDocument()
    fireEvent.change(screen.getByLabelText('股风险预算（%）'), { target: { value: '30' } })
    fireEvent.change(screen.getByLabelText('债风险预算（%）'), { target: { value: '70' } })
    expect(screen.getByText('风险预算合计：100.00%')).toBeInTheDocument()
  })
  it('validates complete axis, finite values and sum without defaults', () => {
    expect(riskBudgetError(['股'], null)).toBeNull()
    expect(riskBudgetError(['股'], { 股: 1 })).toBeNull()
    for (const invalid of [{ 股: NaN }, { 股: -.1 }, { 股: .5 }, { 其他: 1 }] as Record<string, number>[]) expect(riskBudgetError(['股'], invalid)).not.toBeNull()
  })
})

import { useState } from 'react'
import { fireEvent, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { expect, it } from 'vitest'
import InstitutionalFields from './InstitutionalFields'
import { newMandate, mandateIssues } from './model'
import { newInstitution } from '../../services/institutionalContext'
function Harness() {
  const [value, setValue] = useState(newMandate('2026-09-12'))
  return <><InstitutionalFields value={value} onChange={patch => setValue(current => ({ ...current, ...patch }))} /><output data-testid="definition">{JSON.stringify(value)}</output></>
}
it('机构输入按需展开，未提供余额保持null，现金用途不改变原流动性比例', async () => {
  const user = userEvent.setup(); render(<Harness />)
  expect(screen.queryByLabelText('资金用途说明')).not.toBeInTheDocument()
  await user.selectOptions(screen.getByLabelText(/资金用途场景/), 'corporate_treasury')
  await user.type(screen.getByLabelText('资金用途说明'), '企业经营储备')
  fireEvent.change(screen.getByLabelText(/组合内现金用途下限/), { target: { value: '25' } })
  await user.click(screen.getByText('经济状况与人工核验'))
  await user.click(screen.getByRole('checkbox', { name: '提供经济状况研究快照' }))
  expect(screen.getByLabelText(/未缴承诺（CNY）/)).toHaveValue('')
  const value = JSON.parse(screen.getByTestId('definition').textContent!)
  expect(value.institutional_context.cash_reserve_weight).toBe(.25)
  expect(value.min_liquid_weight).toBe(0)
  expect(value.institutional_context.balance_sheet.uncalled_commitments).toBeNull()
  expect(value.institutional_context.review_items.every((i: { status: string }) => i.status === 'not_assessed')).toBe(true)
  expect(screen.getByText(/税务、监管、对冲、杠杆和特殊流动性未自动建模/)).toBeInTheDocument()
})
it('已核对必须有完整证据，未核验能保存研究但不会伪装自动合规', () => {
  const value = { ...newMandate('2026-09-12'), name: '目标', target_return: .02, max_volatility: .1, boundary_reason: '研究确认的边界', institutional_context: { ...newInstitution('family_office'), purpose: '家族长期资金' } }
  expect(mandateIssues(value, '2026-09-12')).toEqual(['', ''])
  value.institutional_context.review_items[0].status = 'researcher_checked'
  expect(mandateIssues(value, '2026-09-12')[0]).toContain('证据')
})

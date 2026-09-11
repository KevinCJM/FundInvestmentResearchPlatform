import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { expect, it, vi } from 'vitest'
import HorizontalMetricComparison, { AllocationMetricsReview } from './HorizontalMetricComparison'

vi.mock('echarts-for-react', () => ({ default: () => <div data-testid="quadrant-chart" /> }))

it('allocation review keeps eight core metrics and reveals annual details on demand without changing the full-table default', async () => {
  const user = userEvent.setup()
  const columns = ['股债方案']
  const rows = ['累计收益率(%)', '年化收益率(%)', '年化波动率(%)', '夏普比率', '99%VaR(日)(%)', '99%ES(日)(%)', '最大回撤(%)', '卡玛比率'].map((label, index) => ({ label, values: [index + 1] }))
  const annualRows = [{ label: '2025累计收益率(%)', values: [12.34] }]
  const full = render(<HorizontalMetricComparison columns={columns} rows={[...rows, ...annualRows]} />)
  expect(screen.getAllByRole('row')).toHaveLength(10)
  expect(screen.getByRole('cell', { name: '2025累计收益率(%)' })).toBeVisible()
  full.unmount()

  render(<AllocationMetricsReview columns={columns} rows={rows} annualRows={annualRows} detailedContent={<p>同类一致性证据</p>} />)
  expect(within(screen.getByRole('region', { name: '核心指标对比' })).getAllByRole('row')).toHaveLength(9)
  expect(screen.queryByRole('cell', { name: '2025累计收益率(%)' })).not.toBeInTheDocument()
  expect(screen.queryByLabelText('X轴指标')).not.toBeInTheDocument()
  expect(screen.queryByText('同类一致性证据')).not.toBeInTheDocument()
  await user.click(screen.getByText('年度指标明细', { selector: 'summary' }))
  expect(await screen.findByRole('cell', { name: '2025累计收益率(%)' })).toBeVisible()
  expect(screen.getByRole('cell', { name: '12.34' })).toBeVisible()
  await user.click(screen.getByText('详细研究：收益风险象限与同类一致性', { selector: 'summary' }))
  const x = await screen.findByLabelText('X轴指标')
  expect(within(x).getAllByRole('option')).toHaveLength(8)
  expect(within(x).queryByRole('option', { name: /2025/ })).not.toBeInTheDocument()
  expect(screen.getByText('同类一致性证据')).toBeVisible()
})

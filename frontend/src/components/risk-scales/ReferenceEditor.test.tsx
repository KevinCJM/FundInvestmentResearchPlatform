import { useState } from 'react'
import { fireEvent, render, screen } from '@testing-library/react'
import { beforeEach, expect, it } from 'vitest'
import { chooseLocale } from '../../i18n/runtime'
import type { ReferenceInputRequest } from '../../services/riskScales'
import { ReferenceEditor } from './ReferenceEditor'

beforeEach(async () => { await chooseLocale('zh-CN') })

const initial: ReferenceInputRequest = {
  name: '测试参考资产', currency: 'CNY', as_of: '2026-09-17',
  calendar: 'SSE', frequency: 'daily', periods_per_year: 252, return_basis: 'selected_index_and_adjusted_product_total_return',
  fee_basis: 'source_embedded_no_additional_fee', fx_basis: 'same_currency_no_conversion',
  assets: [{ id: 'cash', name: '现金', asset_type: 'cash', rationale: '纯现金作为系统风险标尺锚点', cash_return: 0, components: [], rebalance: null }],
}

function Harness() {
  const [value, setValue] = useState(initial)
  return <ReferenceEditor value={value} onChange={setValue} />
}

it('现金只填写年化收益率，不显示代理和再平衡', () => {
  render(<Harness />)
  expect(screen.getByLabelText('大类名称').closest('label')).toHaveAttribute('data-required', 'true')
  expect(screen.getByLabelText('经济定义与依据（可选）').closest('label')).not.toHaveAttribute('data-required')
  expect(screen.getByLabelText('现金预期年化收益率（%）').closest('label')).toHaveAttribute('data-required', 'true')
  expect(screen.getByLabelText('现金预期年化收益率（%）')).toHaveValue('0')
  expect(screen.getByText(/中国银行非定期存款挂牌利率 0\.05%/)).toBeInTheDocument()
  expect(screen.queryByRole('button', { name: '添加或更换代理' })).not.toBeInTheDocument()
  expect(screen.queryByLabelText(/代理再平衡/)).not.toBeInTheDocument()
  expect(screen.queryByText('待配置')).not.toBeInTheDocument()
  expect(screen.queryByText('名称待补充')).not.toBeInTheDocument()
  const asset = screen.getByTestId('risk-reference-asset')
  const add = screen.getByRole('button', { name: '添加参考大类' })
  expect(asset.compareDocumentPosition(add) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
})

it('切换为非现金后默认每日再平衡，且不再要求经济角色', () => {
  render(<Harness />)
  fireEvent.change(screen.getByLabelText('资产类型'), { target: { value: 'market' } })
  expect(screen.queryByLabelText('现金预期年化收益率（%）')).not.toBeInTheDocument()
  expect(screen.queryByLabelText('经济角色')).not.toBeInTheDocument()
  expect(screen.getByLabelText(/代理再平衡/)).toHaveValue('daily')
  expect(screen.getByRole('option', { name: '每季' })).toBeInTheDocument()
  expect(screen.getByRole('option', { name: '每年' })).toBeInTheDocument()
  expect(screen.getByRole('option', { name: '买入持有' })).toBeInTheDocument()
  expect(screen.getByRole('button', { name: '添加或更换代理' })).toBeInTheDocument()
})

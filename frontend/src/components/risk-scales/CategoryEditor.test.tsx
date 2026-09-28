import { useState } from 'react'
import { fireEvent, render, screen, within } from '@testing-library/react'
import { beforeEach, expect, it, vi } from 'vitest'
import { chooseLocale } from '../../i18n/runtime'
import { riskScales, type ReferenceInputRequest } from '../../services/riskScales'
import { CategoryEditor } from './CategoryEditor'

beforeEach(async () => { await chooseLocale('zh-CN') })

const initial: ReferenceInputRequest = {
  name: '测试参考资产', currency: 'CNY', as_of: '2026-09-17',
  calendar: 'SSE', frequency: 'daily', periods_per_year: 252, return_basis: 'selected_index_and_adjusted_product_total_return',
  fee_basis: 'source_embedded_no_additional_fee', fx_basis: 'same_currency_no_conversion',
  assets: [{ id: 'cash', name: '现金', asset_type: 'cash', rationale: '纯现金作为系统风险标尺锚点', cash_return: 0, components: [], rebalance: null }],
}

function Harness(props: { fixedAssets?: boolean; cashEligibleIds?: string[]; value?: ReferenceInputRequest }) {
  const [value, setValue] = useState(props.value ?? initial)
  return <CategoryEditor {...props} value={value} onChange={setValue} />
}

/** 展开某一行的代理与备注区；行内控件在收起状态下是 hidden 的。 */
const expand = (row: HTMLElement, name: string) => fireEvent.click(within(row).getByRole('button', { name: `${name} 的代理与备注` }))

it('现金只填写年化收益率，不显示代理和再平衡', () => {
  render(<Harness />)
  expect(screen.getByLabelText('大类名称')).toBeRequired()
  expect(screen.getByLabelText('现金预期年化收益率（%）')).toHaveValue('0')
  expect(screen.getByText(/中国银行非定期存款挂牌利率 0\.05%/)).toBeInTheDocument()
  expect(screen.getByText('共 1 个大类 · 1 个可用 · 0 个待完成')).toBeInTheDocument()
  expect(screen.queryByLabelText(/代理再平衡/)).not.toBeInTheDocument()

  const asset = screen.getByTestId('risk-reference-asset')
  expand(asset, '现金')
  expect(screen.queryByRole('button', { name: '添加或更换代理' })).not.toBeInTheDocument()
  // 展开区兜底给出经济定义输入框，不必每个调用方自己再写一个。
  expect(screen.getByLabelText('经济定义与依据（可选）').closest('label')).not.toHaveAttribute('data-required')

  // 清单靠汇总与「添加参考大类」起头，新行追加在末尾。
  const add = screen.getByRole('button', { name: '添加参考大类' })
  expect(asset.compareDocumentPosition(add) & Node.DOCUMENT_POSITION_PRECEDING).toBeTruthy()
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
  expect(screen.getByText('待选代理')).toBeInTheDocument()

  expand(screen.getByTestId('risk-reference-asset'), '现金')
  expect(screen.getByRole('button', { name: '添加或更换代理' })).toBeInTheDocument()
  expect(screen.getByText(/每个有值日按目标权重计算当日加权收益率/)).toBeInTheDocument()
})

it('大类被上游冻结时只配代理：名称只读、不能增删、现金仅限可现金的大类', () => {
  const frozen: ReferenceInputRequest = {
    ...initial,
    assets: [
      { id: 'liquidity', name: '流动性', asset_type: 'market', rationale: '', cash_return: null, components: [], rebalance: 'daily' },
      { id: 'equity', name: '权益', asset_type: 'market', rationale: '', cash_return: null, components: [], rebalance: 'daily' },
    ],
  }
  render(<Harness value={frozen} fixedAssets cashEligibleIds={['liquidity']} />)
  expect(screen.getAllByLabelText('大类名称')[0]).toHaveAttribute('readonly')
  expect(screen.queryByRole('button', { name: '添加参考大类' })).not.toBeInTheDocument()
  expect(screen.queryByRole('button', { name: /^移除本草稿大类/ })).not.toBeInTheDocument()
  const rows = screen.getAllByTestId('risk-reference-asset')
  expect(within(rows[0]).getByRole('option', { name: '现金' })).not.toBeDisabled()
  expect(within(rows[1]).getByRole('option', { name: '现金' })).toBeDisabled()
  // 代理仍然可配，冻结的只是大类本身。
  expand(rows[1], '权益')
  expect(within(rows[1]).getByRole('button', { name: '添加或更换代理' })).toBeInTheDocument()
})

it('批量选择跨搜索保留，确认一次原子追加，取消不改动，重复与禁用来源不可选', async () => {
  const market: ReferenceInputRequest = {
    ...initial,
    assets: [
      initial.assets[0],
      { id: 'equity', name: '权益', asset_type: 'market', rationale: '', cash_return: null, components: [], rebalance: 'daily' },
    ],
  }
  function MarketHarness() {
    const [value, setValue] = useState(market)
    const [labels, setLabels] = useState<Record<string, string>>({})
    return <CategoryEditor value={value} sourceLabels={labels} onChange={(next, nextLabels) => { setValue(next); if (nextLabels) setLabels(nextLabels) }} />
  }
  vi.spyOn(riskScales, 'sources').mockImplementation(async (kind: string, _query: string, offset = 0) => ({
    items: kind === 'etf'
      ? [
          { id: 'etf:fund_daily:510300.SH', code: '510300.SH', name: '沪深300ETF', kind: 'etf', coverage: { start_date: '2020-01-02', end_date: '2026-09-17' }, reference_capability: { available: true, supported_fields: ['adj_nav'] } },
          { id: 'etf:fund_daily:000000.SZ', code: '000000.SZ', name: '停用ETF', kind: 'etf', coverage: {}, reference_capability: { available: false, reason: '数据不足' } },
        ]
      : [],
    total: kind === 'etf' ? 2 : 0,
    offset,
    limit: 20,
    problems: [],
  }))
  render(<MarketHarness />)
  const equity = screen.getAllByTestId('risk-reference-asset')[1]
  expand(equity, '权益')
  fireEvent.click(within(equity).getByRole('button', { name: '添加或更换代理' }))
  fireEvent.change(screen.getByLabelText('来源类型'), { target: { value: 'etf' } })
  await screen.findByText('沪深300ETF')

  const availableRow = screen.getByText('沪深300ETF').closest('tr')!
  const disabledRow = screen.getByText('停用ETF').closest('tr')!
  expect(within(disabledRow).getByRole('checkbox')).toBeDisabled()
  fireEvent.click(within(availableRow).getByRole('checkbox'))
  expect(screen.getByText('已选 1 项')).toBeInTheDocument()

  // 搜索或翻页不清空已选托盘。
  fireEvent.change(screen.getByLabelText('搜索名称或代码'), { target: { value: '510300' } })
  await screen.findByText('沪深300ETF')
  expect(screen.getByText('已选 1 项')).toBeInTheDocument()

  fireEvent.click(screen.getByRole('button', { name: '确认选择' }))
  expect(within(equity).getByText('沪深300ETF')).toBeInTheDocument()
  expect(screen.queryByLabelText('来源类型')).not.toBeInTheDocument()

  // 已添加的来源不可重复选择；取消不产生任何修改。
  fireEvent.click(within(equity).getByRole('button', { name: '添加或更换代理' }))
  fireEvent.change(screen.getByLabelText('来源类型'), { target: { value: 'etf' } })
  const picker = screen.getByRole('region', { name: '代理来源选择' })
  await within(picker).findByText('沪深300ETF')
  expect(within(picker).getByRole('checkbox', { name: /沪深300ETF/ })).toBeDisabled()
  expect(screen.getAllByText('已添加').length).toBeGreaterThan(0)
  fireEvent.click(screen.getByRole('button', { name: '取消' }))
  expect(screen.queryByLabelText('来源类型')).not.toBeInTheDocument()
})
